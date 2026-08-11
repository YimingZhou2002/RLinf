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

"""Build same-shape fake messages for the batch-size / micro-batch sweeps.

The design goal is to be **model-agnostic**: nothing here names a field or a
non-batch dimension. We take the messages captured from a *default* run (see
``message_capture``) as the source of truth for every non-batch dimension, and
only ever resize the **batch axis (dim 0)** to the requested size.

``resize_batch(obj, N)`` is the single generic primitive:

- ``torch.Tensor`` -> index dim 0 with ``arange(N) % len`` (tile-then-slice;
  preserves dtype / device and keeps values in their captured range).
- ``list``         -> repeat / slice to length ``N`` (e.g. ``task_descriptions``).
- ``dict``         -> recurse over values.
- ``None`` / scalar / other -> passthrough unchanged.

It is used for both the rollout observation dict and the flattened actor
training pool, because after flattening ``[T, B, ...] -> [T*B, ...]`` every field
shares dim 0, exactly like an obs dict.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from rlinf.workers.actor.fsdp_actor_worker import process_nested_dict_for_train
from rlinf.data.schema.embodied_types import (
    Trajectory,
    convert_trajectories_to_batch,
)


# --------------------------------------------------------------------------- #
# Generic batch-axis (dim 0) resize
# --------------------------------------------------------------------------- #
def resize_batch(obj: Any, batch_size: int) -> Any:
    """Return a copy of ``obj`` with its batch axis (dim 0) resized to
    ``batch_size``. Non-batch dimensions and dtypes are preserved exactly."""
    if isinstance(obj, torch.Tensor):
        if obj.shape[0] == batch_size:
            return obj
        idx = torch.arange(batch_size) % obj.shape[0]
        return obj[idx].contiguous()
    if isinstance(obj, dict):
        return {k: resize_batch(v, batch_size) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return obj
        return [obj[i % len(obj)] for i in range(batch_size)]
    if isinstance(obj, tuple):
        if not obj:
            return obj
        return tuple(obj[i % len(obj)] for i in range(batch_size))
    # None, scalars, strings, and anything else are non-batch -> passthrough.
    return obj


def infer_batch_size(obj: Any) -> int:
    """Best-effort discovery of the current batch size (dim 0) of a message.

    Model-agnostic: returns the dim-0 length of the first tensor found, or the
    length of the first list found, walking dicts recursively.
    """
    if isinstance(obj, torch.Tensor):
        return obj.shape[0]
    if isinstance(obj, dict):
        for v in obj.values():
            try:
                return infer_batch_size(v)
            except ValueError:
                continue
    if isinstance(obj, (list, tuple)) and obj:
        return len(obj)
    raise ValueError("Cannot infer batch size from message.")


# --------------------------------------------------------------------------- #
# Rollout: observation dict fed to predict()
# --------------------------------------------------------------------------- #
def load_rollout_obs(pt_path: str) -> dict[str, Any]:
    """Load the captured env->rollout message and return the inner obs dict
    (the argument ``predict`` / ``_predict_rollout_actions`` consumes)."""
    seed = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(seed, dict) and "obs" in seed:
        return seed["obs"]
    return seed


# --------------------------------------------------------------------------- #
# Actor: flattened training pool fed to train_micro_batch()
# --------------------------------------------------------------------------- #
def load_trajectories(pt_path: str) -> list[Trajectory]:
    seed = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(seed, Trajectory):
        return [seed]
    if isinstance(seed, (list, tuple)):
        return list(seed)
    raise TypeError(f"Expected list[Trajectory], got {type(seed)}")


def build_actor_pool(
    trajectories: list[Trajectory],
    compute_adv_fn,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a flattened ``[T*B, ...]`` training pool from captured trajectories,
    reusing the framework's own conversion so every model-specific key/shape is
    produced exactly as in production.

    Args:
        trajectories: captured ``list[Trajectory]``.
        compute_adv_fn: a zero-arg callable that, given ``rollout_batch`` already
            assigned on the actor, populates advantages/returns in place (i.e.
            ``EmbodiedFSDPActor.compute_advantages_and_returns`` bound to the
            live actor). Returns the ``rollout_batch`` dict to flatten.
    """
    rollout_batch = convert_trajectories_to_batch(trajectories)
    # Let the live actor add advantages / returns (pure tensor math, reads and
    # writes the batch it was handed).
    rollout_batch = compute_adv_fn(rollout_batch)

    time_dim, batch_dim = rollout_batch["prev_logprobs"].shape[:2]
    n = time_dim * batch_dim
    g = torch.Generator()
    g.manual_seed(seed)
    shuffle_id = torch.randperm(n, generator=g)
    pool = process_nested_dict_for_train(rollout_batch, shuffle_id)
    return pool
