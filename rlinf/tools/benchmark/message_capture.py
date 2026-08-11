# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture the intermediate messages exchanged between the embodied
env / rollout / actor components during a live rollout.

The three components communicate through named channels. This module hooks the
methods that *build* the wire objects (so no core worker code changes) and dumps
each message exactly once, per worker process, on rank 0:

- ``rollout_policy_output``   : the rollout ``PolicyOutput`` sent back to env
- ``env_feedback_envoutput``  : the env ``EnvOutput`` (obs + rewards + dones ...)
- ``env_to_rollout_obs``      : the observation dict env sends to rollout
- ``env_to_actor_trajectory`` : the ``list[Trajectory]`` env sends to the actor

For every message we write three sibling files under
``$RLINF_BENCH_CAPTURE_DIR``:

- ``<name>.schema.txt``  : a human-readable, indented tree (shapes/dtypes/stats)
- ``<name>.schema.json`` : the same structure as JSON (reused later to build fakes)
- ``<name>.pt``          : the raw tensors on CPU (torch.save), for exact replay

Everything here is a no-op unless ``RLINF_BENCH_CAPTURE_DIR`` is set, so the
capture worker subclasses are safe to import / leave in place.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
from typing import Any

import torch

from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

_CAPTURE_DIR_ENV = "RLINF_BENCH_CAPTURE_DIR"
_SAVE_PT_ENV = "RLINF_BENCH_CAPTURE_SAVE_PT"

# Guards so each edge is dumped exactly once per worker process.
_dumped: set[str] = set()
_lock = threading.Lock()


def capture_enabled() -> bool:
    """Whether message capture is active (driven by ``RLINF_BENCH_CAPTURE_DIR``)."""
    return bool(os.environ.get(_CAPTURE_DIR_ENV))


def _save_pt_enabled() -> bool:
    return os.environ.get(_SAVE_PT_ENV, "1") not in ("0", "false", "False", "")


def _should_capture(name: str, rank: int) -> bool:
    """Cheap gate: only rank 0, only when enabled, only if not yet dumped.

    Checked *before* any expensive work (e.g. re-deriving trajectories) so the
    hooks stay free on every call after the first.
    """
    if rank != 0 or not capture_enabled():
        return False
    return name not in _dumped


# --------------------------------------------------------------------------- #
# Structural description
# --------------------------------------------------------------------------- #
def _tensor_desc(t: torch.Tensor) -> dict[str, Any]:
    desc: dict[str, Any] = {
        "kind": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    try:
        if t.numel() == 0:
            desc["stats"] = "empty"
        elif t.dtype == torch.bool:
            desc["stats"] = {"true": int(t.sum().item()), "numel": int(t.numel())}
        else:
            f = t.detach().float()
            desc["stats"] = {
                "min": round(f.min().item(), 6),
                "max": round(f.max().item(), 6),
                "mean": round(f.mean().item(), 6),
                "nan": int(torch.isnan(f).sum().item()),
            }
    except Exception as exc:  # pragma: no cover - stats are best-effort only
        desc["stats"] = f"<unavailable: {exc}>"
    return desc


def describe(obj: Any, _depth: int = 0) -> Any:
    """Recursively describe a message's structure (types/shapes/dtypes/stats)."""
    if obj is None:
        return {"kind": "none"}
    if isinstance(obj, torch.Tensor):
        return _tensor_desc(obj)
    if isinstance(obj, str):
        return {"kind": "str", "len": len(obj), "value": obj[:120]}
    if isinstance(obj, (bool, int, float)):
        return {"kind": "scalar", "type": type(obj).__name__, "value": obj}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "kind": "dataclass",
            "type": type(obj).__name__,
            "fields": {
                f.name: describe(getattr(obj, f.name), _depth + 1)
                for f in dataclasses.fields(obj)
            },
        }
    if isinstance(obj, dict):
        return {
            "kind": "dict",
            "keys": {str(k): describe(v, _depth + 1) for k, v in obj.items()},
        }
    if isinstance(obj, (list, tuple)):
        out: dict[str, Any] = {
            "kind": type(obj).__name__,
            "len": len(obj),
        }
        if obj:
            # Describe the first element; if elements are heterogeneous the
            # first is still the most useful representative for inspection.
            out["elem"] = describe(obj[0], _depth + 1)
        return out
    return {"kind": "other", "type": type(obj).__name__, "repr": repr(obj)[:120]}


def _render_text(desc: Any, name: str) -> str:
    """Render a ``describe()`` dict as an indented, human-readable tree."""
    lines: list[str] = [f"# {name}", ""]

    def walk(node: Any, prefix: str, indent: int) -> None:
        pad = "  " * indent
        kind = node.get("kind") if isinstance(node, dict) else None
        if kind == "tensor":
            stats = node.get("stats")
            lines.append(
                f"{pad}{prefix}: tensor {node['shape']} "
                f"{node['dtype'].replace('torch.', '')} {node['device']} | {stats}"
            )
        elif kind == "str":
            lines.append(f"{pad}{prefix}: str(len={node['len']}) {node['value']!r}")
        elif kind == "scalar":
            lines.append(f"{pad}{prefix}: {node['type']}={node['value']}")
        elif kind == "none":
            lines.append(f"{pad}{prefix}: None")
        elif kind == "dataclass":
            lines.append(f"{pad}{prefix}: {node['type']} (dataclass)")
            for fname, fnode in node["fields"].items():
                walk(fnode, fname, indent + 1)
        elif kind == "dict":
            lines.append(f"{pad}{prefix}: dict[{len(node['keys'])}]")
            for k, v in node["keys"].items():
                walk(v, k, indent + 1)
        elif kind in ("list", "tuple"):
            lines.append(f"{pad}{prefix}: {kind}(len={node['len']})")
            if "elem" in node:
                walk(node["elem"], "[0]", indent + 1)
        else:
            lines.append(f"{pad}{prefix}: {node}")

    walk(desc, "<root>", 0)
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dumping
# --------------------------------------------------------------------------- #
def _dump_once(name: str, obj: Any) -> None:
    """Write schema + raw tensors for ``obj`` exactly once per process."""
    if not capture_enabled():
        return
    with _lock:
        if name in _dumped:
            return
        _dumped.add(name)

    capture_dir = os.environ[_CAPTURE_DIR_ENV]
    os.makedirs(capture_dir, exist_ok=True)

    # The wire dataclasses already move tensors to CPU in __post_init__, but be
    # defensive for plain dict payloads.
    try:
        cpu_obj = obj if dataclasses.is_dataclass(obj) else put_tensor_device(obj, "cpu")
    except Exception:
        cpu_obj = obj

    desc = describe(cpu_obj)

    with open(os.path.join(capture_dir, f"{name}.schema.json"), "w") as f:
        json.dump(desc, f, indent=2)
    with open(os.path.join(capture_dir, f"{name}.schema.txt"), "w") as f:
        f.write(_render_text(desc, name))

    if _save_pt_enabled():
        try:
            torch.save(cpu_obj, os.path.join(capture_dir, f"{name}.pt"))
        except Exception as exc:  # pragma: no cover - raw save is best-effort
            with open(os.path.join(capture_dir, f"{name}.pt.error"), "w") as f:
                f.write(f"torch.save failed: {exc}\n")

    print(f"[bench-capture] dumped '{name}' -> {capture_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Capture worker subclasses (additive; no edits to the base workers)
# --------------------------------------------------------------------------- #
class CaptureRolloutWorker(MultiStepRolloutWorker):
    """Rollout worker that captures its ``PolicyOutput`` (rollout -> env)."""

    def _build_policy_output(self, actions, result, *, final_obs=None):
        out = super()._build_policy_output(actions, result, final_obs=final_obs)
        if _should_capture("rollout_policy_output", self._rank):
            _dump_once("rollout_policy_output", out)
        return out


class CaptureEnvWorker(EnvWorker):
    """Env worker that captures its feedback, the obs it sends to rollout, and
    the trajectories it sends to the actor."""

    def env_interact_step(self, chunk_actions, stage_id):
        out = super().env_interact_step(chunk_actions, stage_id)
        # out == (EnvOutput, env_info, chunk_step_payload)
        if _should_capture("env_feedback_envoutput", self._rank):
            _dump_once("env_feedback_envoutput", out[0])
        return out

    def _build_rollout_input_data(self, env_batch):
        data = super()._build_rollout_input_data(env_batch)
        if _should_capture("env_to_rollout_obs", self._rank):
            _dump_once("env_to_rollout_obs", data)
        return data

    async def send_rollout_trajectories(self, trajectory_builder, channel):
        # Re-derive the split trajectories only when we still need to capture;
        # to_splited_trajectories() is side-effect free (it does not clear the
        # builder), so this is safe and runs at most once.
        if _should_capture("env_to_actor_trajectory", self._rank):
            trajs = trajectory_builder.to_splited_trajectories(self.actor_split_num)
            _dump_once("env_to_actor_trajectory", trajs)
        return await super().send_rollout_trajectories(trajectory_builder, channel)
