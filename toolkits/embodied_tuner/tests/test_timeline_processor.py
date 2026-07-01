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

"""Unit tests for :mod:`toolkits.embodied_tuner.timeline_processor`.

These tests exercise the four analyses (A' / C' / D' / raw excerpts)
on hand-crafted synthetic event lists so failure modes are deterministic
and don't depend on a live timeline directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolkits.embodied_tuner.timeline_processor import (
    BLOCKING_TAGS,
    compute_critical_path,
    compute_outliers,
    compute_per_gpu_bubble,
    extract_raw_excerpts,
    is_blocking,
    placement_to_gpu_map,
    process_timeline,
)


# ---------------------------------------------------------------------------
# Synthetic event helpers
# ---------------------------------------------------------------------------


def _event(t0, t1, *, component, rank, tag, global_step=0, **extra):
    return {
        "t0": t0, "t1": t1, "dur": t1 - t0,
        "component": component, "rank": rank, "tag": tag,
        "global_step": global_step,
        **extra,
    }


# ---------------------------------------------------------------------------
# is_blocking / BLOCKING_TAGS
# ---------------------------------------------------------------------------


def test_actor_recv_traj_classified_as_blocking():
    assert is_blocking(_event(0, 10, component="actor", rank=0, tag="actor/recv_traj"))


def test_actor_sync_to_rollout_is_real_busy():
    # NCCL broadcast is real GPU work; only recv_traj is the blocking wait.
    assert not is_blocking(
        _event(0, 10, component="actor", rank=0, tag="actor/sync_model_to_rollout")
    )


def test_rollout_generate_wrapper_is_blocking():
    assert is_blocking(_event(0, 5, component="rollout", rank=0, tag="rollout/generate"))


def test_unknown_tag_defaults_to_real_busy():
    assert not is_blocking(_event(0, 1, component="actor", rank=0, tag="custom_step"))


def test_blocking_tags_includes_known_wrappers():
    # Smoke-test the constant so a typo in the dict surfaces immediately.
    assert "actor/recv_traj" in BLOCKING_TAGS["actor"]
    assert "rollout/generate" in BLOCKING_TAGS["rollout"]


# ---------------------------------------------------------------------------
# placement_to_gpu_map
# ---------------------------------------------------------------------------


def test_placement_to_gpu_map_handles_range_strings():
    out = placement_to_gpu_map(
        {"actor": "0-7", "env": "0-3", "rollout": "4-7"}, num_gpus=8
    )
    assert out["actor"] == {i: i for i in range(8)}
    assert out["env"] == {0: 0, 1: 1, 2: 2, 3: 3}
    assert out["rollout"] == {0: 4, 1: 5, 2: 6, 3: 7}


def test_placement_to_gpu_map_handles_list_values():
    out = placement_to_gpu_map({"actor": [0, 1, 2, 3]}, num_gpus=8)
    assert out["actor"] == {0: 0, 1: 1, 2: 2, 3: 3}


def test_placement_to_gpu_map_skips_unparseable_entries():
    out = placement_to_gpu_map({"actor": "0-7", "weird": 17}, num_gpus=8)
    assert "actor" in out
    assert "weird" not in out


# ---------------------------------------------------------------------------
# A' — compute_critical_path
# ---------------------------------------------------------------------------


def test_critical_path_separates_real_from_blocked():
    events = [
        # rollout does 10s of real work
        _event(0, 10, component="rollout", rank=0, tag="predict"),
        # actor is blocked for the same window
        _event(0, 10, component="actor", rank=0, tag="actor/recv_traj"),
        # actor then does 2s of training
        _event(10, 12, component="actor", rank=0, tag="actor_forward"),
    ]
    cp = compute_critical_path(events)
    step = cp[0]
    lanes = {(l["component"], l["rank"]): l for l in step["real_busy_top"]}
    # rollout has 10s real, 0 blocked
    assert lanes[("rollout", 0)]["real_s"] == 10.0
    assert lanes[("rollout", 0)]["blocked_s"] == 0.0
    # actor has 2s real (forward) + 10s blocked (recv_traj)
    assert lanes[("actor", 0)]["real_s"] == 2.0
    assert lanes[("actor", 0)]["blocked_s"] == 10.0


def test_critical_path_skips_events_without_global_step():
    events = [_event(0, 5, component="env", rank=0, tag="env_interact_step",
                     global_step=None)]
    assert compute_critical_path(events) == {}


def test_critical_path_ranks_lanes_by_real_busy():
    events = [
        _event(0, 5, component="rollout", rank=0, tag="predict"),
        _event(0, 8, component="env", rank=0, tag="env_interact_step"),
        _event(0, 1, component="actor", rank=0, tag="actor_forward"),
    ]
    cp = compute_critical_path(events)
    top = cp[0]["real_busy_top"]
    assert top[0]["component"] == "env"
    assert top[1]["component"] == "rollout"
    assert top[2]["component"] == "actor"


# ---------------------------------------------------------------------------
# C' — compute_outliers
# ---------------------------------------------------------------------------


def test_outlier_carries_knob_hint_for_env_offload_warmup():
    # 30 short events lift P95 high enough that the 45s tail strictly exceeds it.
    events = [
        _event(0.0, 0.1, component="env", rank=0, tag="env_interact_step",
               global_step=None)
        for _ in range(30)
    ]
    events.append(_event(0.0, 45.0, component="env", rank=0,
                         tag="env_interact_step", global_step=None))
    outliers = compute_outliers(events, enable_offload={"env": True})
    assert outliers[0]["tag"] == "env_interact_step"
    assert outliers[0]["dur_s"] == 45.0
    assert "env.enable_offload" in outliers[0]["knob_hint"]


def test_outlier_has_no_hint_when_offload_unknown():
    events = [
        _event(0.0, 0.1, component="env", rank=0, tag="env_interact_step",
               global_step=None)
        for _ in range(30)
    ]
    events.append(_event(0.0, 45.0, component="env", rank=0,
                         tag="env_interact_step", global_step=None))
    outliers = compute_outliers(events, enable_offload=None)
    assert outliers[0]["knob_hint"] is None


def test_outlier_skips_tags_with_too_few_samples():
    events = [_event(0.0, 30.0, component="env", rank=0,
                     tag="env_interact_step", global_step=None)]
    # Below the n<10 threshold → no outliers.
    assert compute_outliers(events) == ()


def test_outlier_min_seconds_threshold():
    # 16 events at 0.05s with one at 0.5s - over P95 but under 1s threshold.
    events = [_event(0.0, 0.05, component="env", rank=0,
                     tag="env_interact_step", global_step=None)
              for _ in range(15)]
    events.append(_event(0.0, 0.5, component="env", rank=0,
                         tag="env_interact_step", global_step=None))
    assert compute_outliers(events) == ()


# ---------------------------------------------------------------------------
# D' — compute_per_gpu_bubble
# ---------------------------------------------------------------------------


def test_per_gpu_bubble_with_hybrid_placement():
    events = [
        # GPU 4-7: rollout busy 0..50, GPU 0-3: env busy 0..30
        _event(0, 50, component="rollout", rank=0, tag="predict"),
        _event(0, 50, component="rollout", rank=1, tag="predict"),
        _event(0, 50, component="rollout", rank=2, tag="predict"),
        _event(0, 50, component="rollout", rank=3, tag="predict"),
        _event(0, 30, component="env", rank=0, tag="env_interact_step"),
        _event(0, 30, component="env", rank=1, tag="env_interact_step"),
        _event(0, 30, component="env", rank=2, tag="env_interact_step"),
        _event(0, 30, component="env", rank=3, tag="env_interact_step"),
    ]
    out = compute_per_gpu_bubble(
        events,
        placement={"actor": "0-7", "env": "0-3", "rollout": "4-7"},
    )
    assert out["wall_s"] == 50.0
    # env GPUs busy 30/50 → bubble 20s
    assert out["per_gpu"]["0"]["bubble_s"] == 20.0
    # rollout GPUs busy 50/50 → bubble 0s
    assert out["per_gpu"]["4"]["bubble_s"] == 0.0
    assert out["env_side_avg_bubble_s"] == 20.0
    assert out["rollout_side_avg_bubble_s"] == 0.0


def test_per_gpu_bubble_skips_blocking_tags():
    # actor on GPU 0 spends 50s "busy" on recv_traj (blocking) — should NOT count.
    events = [
        _event(0, 50, component="actor", rank=0, tag="actor/recv_traj"),
        _event(0, 10, component="env", rank=0, tag="env_interact_step"),
    ]
    out = compute_per_gpu_bubble(
        events,
        placement={"actor": "0-7", "env": "0-3", "rollout": "4-7"},
    )
    # GPU 0 only has env's 10s of real work
    assert out["per_gpu"]["0"]["busy_s"] == 10.0


def test_per_gpu_bubble_returns_empty_without_placement():
    events = [_event(0, 10, component="env", rank=0, tag="env_interact_step")]
    assert compute_per_gpu_bubble(events, placement=None) == {}


# ---------------------------------------------------------------------------
# extract_raw_excerpts
# ---------------------------------------------------------------------------


def test_raw_excerpts_excludes_runner_component():
    events = [
        _event(0, 100, component="runner", rank=0, tag="run"),
        _event(0, 50, component="rollout", rank=0, tag="predict"),
    ]
    excerpts = extract_raw_excerpts(events)
    assert all(e["component"] != "runner" for e in excerpts)
    assert excerpts[0]["component"] == "rollout"


def test_raw_excerpts_sorted_by_duration_desc():
    events = [
        _event(0, 5, component="env", rank=0, tag="env_interact_step"),
        _event(0, 50, component="rollout", rank=0, tag="predict"),
        _event(0, 25, component="actor", rank=0, tag="actor_forward"),
    ]
    excerpts = extract_raw_excerpts(events, top_k=3)
    assert [e["dur_s"] for e in excerpts] == [50.0, 25.0, 5.0]


def test_raw_excerpts_keep_only_useful_fields():
    event = _event(0, 1, component="env", rank=0, tag="env_interact_step",
                   qualname="EnvWorker.step", call_index=3,
                   some_other_field="dropped")
    excerpts = extract_raw_excerpts([event])
    assert "qualname" in excerpts[0]
    assert "call_index" in excerpts[0]
    assert "some_other_field" not in excerpts[0]


# ---------------------------------------------------------------------------
# High-level process_timeline
# ---------------------------------------------------------------------------


def test_process_timeline_empty_dir(tmp_path):
    assert process_timeline(tmp_path) == {}


def test_process_timeline_full_pipeline(tmp_path):
    timeline = tmp_path / "timeline"
    timeline.mkdir()
    (timeline / "env_rank0.jsonl").write_text("\n".join(
        json.dumps({"t0": 0.0, "t1": 30.0, "component": "env", "rank": 0,
                    "tag": "env_interact_step", "global_step": 0})
        for _ in range(1)
    ) + "\n")
    (timeline / "rollout_rank0.jsonl").write_text(
        json.dumps({"t0": 0.0, "t1": 50.0, "component": "rollout", "rank": 0,
                    "tag": "predict", "global_step": 0}) + "\n"
    )
    result = process_timeline(
        timeline,
        placement={"actor": "0-7", "env": "0-3", "rollout": "4-7"},
        enable_offload={"env": True},
    )
    assert "critical_path" in result
    assert "per_gpu_bubble" in result
    assert "outliers" in result
    assert "raw_excerpts" in result
    assert result["per_gpu_bubble"]["wall_s"] == 50.0
