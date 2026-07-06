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

Exercises the analyses (loading, per-tag stats, stall fractions, A' /
C' / D' / raw excerpts) on hand-crafted synthetic event lists so
failure modes are deterministic and don't depend on a live timeline
directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolkits.embodied_tuner.timeline_processor import (
    BLOCKING_TAGS,
    EXCLUDED_COMPONENTS,
    compute_component_call_averages,
    compute_critical_path,
    compute_outliers,
    compute_per_component_bubble,
    compute_stall_fractions,
    compute_tag_stats,
    extract_raw_excerpts,
    is_blocking,
    load_events,
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


def test_runner_is_excluded_from_component_views():
    assert "runner" in EXCLUDED_COMPONENTS


# ---------------------------------------------------------------------------
# load_events
# ---------------------------------------------------------------------------


def test_load_events_normalises_and_skips_bad_lines(tmp_path):
    (tmp_path / "env_rank0.jsonl").write_text(
        json.dumps({"t0": 5.0, "t1": 2.0, "component": "env", "rank": "3",
                    "tag": "env_interact_step"}) + "\n"
        + "not a json line\n"
        + json.dumps({"t0": 0.0, "t1": 1.0, "component": "env", "rank": 0,
                      "worker_timer": "misc"}) + "\n"
    )
    events = load_events(tmp_path)
    assert len(events) == 2
    # First record had reversed timestamps → swapped; rank coerced to int.
    assert events[0]["t0"] == 2.0 and events[0]["t1"] == 5.0
    assert events[0]["rank"] == 3
    assert events[0]["dur"] == 3.0
    # Second record used `worker_timer` as the tag source.
    assert events[1]["tag"] == "misc"


def test_load_events_returns_empty_for_missing_dir(tmp_path):
    assert load_events(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# compute_stall_fractions
# ---------------------------------------------------------------------------


def test_stall_fractions_computed_over_full_window():
    events = [
        _event(0, 40, component="env", rank=0, tag="env_interact_step"),
        _event(0, 100, component="rollout", rank=0, tag="predict"),
    ]
    out = compute_stall_fractions(events)
    # Wall window = [0, 100]. env covers 40/100 → stall 0.6. rollout covers 100/100 → 0.
    assert out["env"] == pytest.approx(0.6)
    assert out["rollout"] == pytest.approx(0.0)


def test_stall_fractions_excludes_runner():
    events = [
        _event(0, 100, component="runner", rank=0, tag="run"),
        _event(0, 40, component="env", rank=0, tag="env_interact_step"),
    ]
    out = compute_stall_fractions(events)
    assert "runner" not in out
    assert "env" in out


def test_stall_fractions_empty_input():
    assert compute_stall_fractions([]) == {}


# ---------------------------------------------------------------------------
# compute_tag_stats
# ---------------------------------------------------------------------------


def test_tag_stats_aggregates_per_component_rank_tag():
    events = [
        _event(0, 1.0, component="env", rank=0, tag="env_interact_step"),
        _event(1, 3.0, component="env", rank=0, tag="env_interact_step"),
        _event(0, 2.0, component="env", rank=1, tag="env_interact_step"),
    ]
    rows = compute_tag_stats(events, headline_tags={"env_interact_step"})
    by_key = {(r["component"], r["rank"]): r for r in rows}
    assert by_key[("env", 0)]["call_count"] == 2
    assert by_key[("env", 0)]["duration_median"] == pytest.approx(1.5)
    assert by_key[("env", 1)]["call_count"] == 1


def test_tag_stats_drops_non_headline_tags():
    events = [_event(0, 1, component="env", rank=0, tag="other_tag")]
    assert compute_tag_stats(events, headline_tags={"env_interact_step"}) == []


def test_tag_stats_excludes_runner():
    events = [_event(0, 100, component="runner", rank=0, tag="run")]
    assert compute_tag_stats(events, headline_tags={"run"}) == []


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


def test_critical_path_excludes_runner_lane():
    events = [
        _event(0, 100, component="runner", rank=0, tag="run"),
        _event(0, 5, component="env", rank=0, tag="env_interact_step"),
    ]
    top = compute_critical_path(events)[0]["real_busy_top"]
    assert all(lane["component"] != "runner" for lane in top)


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
# D' — compute_per_component_bubble
# ---------------------------------------------------------------------------


def test_per_component_bubble_unions_ranks_and_computes_bubble():
    events = [
        # env: 4 ranks all busy 0..30 → union busy 30/wall 50 → bubble 20
        _event(0, 30, component="env", rank=0, tag="env_interact_step"),
        _event(0, 30, component="env", rank=1, tag="env_interact_step"),
        _event(0, 30, component="env", rank=2, tag="env_interact_step"),
        _event(0, 30, component="env", rank=3, tag="env_interact_step"),
        # rollout: 4 ranks all busy 0..50 → union busy 50/wall 50 → bubble 0
        _event(0, 50, component="rollout", rank=0, tag="predict"),
        _event(0, 50, component="rollout", rank=1, tag="predict"),
        _event(0, 50, component="rollout", rank=2, tag="predict"),
        _event(0, 50, component="rollout", rank=3, tag="predict"),
    ]
    out = compute_per_component_bubble(events)
    assert out["wall_s"] == 50.0
    assert out["per_component"]["env"]["busy_s"] == 30.0
    assert out["per_component"]["env"]["bubble_s"] == 20.0
    assert out["per_component"]["env"]["bubble_frac"] == 0.40
    assert out["per_component"]["env"]["num_ranks"] == 4
    assert out["per_component"]["rollout"]["bubble_s"] == 0.0


def test_per_component_bubble_reports_per_rank_detail():
    events = [
        _event(0, 50, component="env", rank=0, tag="env_interact_step"),
        _event(0, 20, component="env", rank=1, tag="env_interact_step"),
    ]
    out = compute_per_component_bubble(events)
    per_rank = out["per_component"]["env"]["per_rank"]
    # rank0 covers the whole wall; rank1 is the straggler bringing avg down.
    assert per_rank["0"]["bubble_s"] == 0.0
    assert per_rank["1"]["bubble_s"] == 30.0


def test_per_component_bubble_excludes_blocking_tags():
    # actor spends 50s on recv_traj (blocking) → must NOT show as busy.
    events = [
        _event(0, 50, component="actor", rank=0, tag="actor/recv_traj"),
        _event(0, 10, component="actor", rank=0, tag="actor_forward"),
    ]
    out = compute_per_component_bubble(events)
    assert out["per_component"]["actor"]["busy_s"] == 10.0
    assert out["per_component"]["actor"]["bubble_s"] == 40.0


def test_per_component_bubble_excludes_runner():
    events = [
        _event(0, 100, component="runner", rank=0, tag="run"),
        _event(0, 30, component="env", rank=0, tag="env_interact_step"),
    ]
    out = compute_per_component_bubble(events)
    assert "runner" not in out["per_component"]
    assert "env" in out["per_component"]


def test_per_component_bubble_returns_empty_on_empty_input():
    assert compute_per_component_bubble([]) == {}


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
    (timeline / "env_rank0.jsonl").write_text(
        json.dumps({"t0": 0.0, "t1": 30.0, "component": "env", "rank": 0,
                    "tag": "env_interact_step", "global_step": 0}) + "\n"
    )
    (timeline / "rollout_rank0.jsonl").write_text(
        json.dumps({"t0": 0.0, "t1": 50.0, "component": "rollout", "rank": 0,
                    "tag": "predict", "global_step": 0}) + "\n"
    )
    result = process_timeline(timeline, enable_offload={"env": True})
    assert "critical_path" in result
    assert "per_component_bubble" in result
    assert "outliers" in result
    assert "raw_excerpts" in result
    assert "component_call_averages" in result
    assert result["per_component_bubble"]["wall_s"] == 50.0
    assert result["per_component_bubble"]["per_component"]["env"]["busy_s"] == 30.0


# ---------------------------------------------------------------------------
# compute_component_call_averages
# ---------------------------------------------------------------------------


def test_component_call_averages_skips_first_two_warmup_events():
    # env: 20s warmup, 15s warmup, then 5s / 5s / 5s steady state -> mean 5.0
    events = [
        _event(0, 20, component="env", rank=0, tag="env_interact_step"),
        _event(20, 35, component="env", rank=0, tag="env_interact_step"),
        _event(35, 40, component="env", rank=0, tag="env_interact_step"),
        _event(40, 45, component="env", rank=0, tag="env_interact_step"),
        _event(45, 50, component="env", rank=0, tag="env_interact_step"),
    ]
    out = compute_component_call_averages(events)
    assert out["env"]["skipped"] == 2
    assert out["env"]["remaining_count"] == 3
    assert out["env"]["mean_duration_s"] == 5.0
    assert out["env"]["call_count_total"] == 5


def test_component_call_averages_pools_across_ranks_by_t0():
    # Two ranks interleaved. Sorted by t0: rank0@0(10s), rank1@5(8s),
    # rank0@10(4s), rank1@13(4s). Skip first 2 -> mean of {4, 4} = 4.
    events = [
        _event(0, 10, component="env", rank=0, tag="env_interact_step"),
        _event(5, 13, component="env", rank=1, tag="env_interact_step"),
        _event(10, 14, component="env", rank=0, tag="env_interact_step"),
        _event(13, 17, component="env", rank=1, tag="env_interact_step"),
    ]
    out = compute_component_call_averages(events)
    assert out["env"]["remaining_count"] == 2
    assert out["env"]["mean_duration_s"] == 4.0


def test_component_call_averages_excludes_blocking_wrapper_tags():
    # `interact` is a wrapper (blocking) — must be dropped or it would
    # double-count with env_interact_step children.
    events = [
        _event(0, 100, component="env", rank=0, tag="interact"),
        _event(0, 10, component="env", rank=0, tag="env_interact_step"),
        _event(10, 20, component="env", rank=0, tag="env_interact_step"),
        _event(20, 25, component="env", rank=0, tag="env_interact_step"),
    ]
    out = compute_component_call_averages(events)
    # Wrapper excluded; children sorted, first 2 skipped, mean of {5} = 5.0
    assert out["env"]["call_count_total"] == 3
    assert out["env"]["mean_duration_s"] == 5.0


def test_component_call_averages_omits_component_with_too_few_events():
    # Only 2 events -> nothing to average after skipping 2.
    events = [
        _event(0, 10, component="env", rank=0, tag="env_interact_step"),
        _event(10, 15, component="env", rank=0, tag="env_interact_step"),
    ]
    assert compute_component_call_averages(events) == {}


def test_component_call_averages_ignores_actor_by_default():
    # Default components tuple is (env, rollout) — actor events are
    # dropped even though they are non-blocking.
    events = [
        _event(0, 1, component="actor", rank=0, tag="actor/sync_model_to_rollout"),
        _event(1, 2, component="actor", rank=0, tag="actor/sync_model_to_rollout"),
        _event(2, 3, component="actor", rank=0, tag="actor/sync_model_to_rollout"),
    ]
    assert compute_component_call_averages(events) == {}


def test_component_call_averages_respects_custom_skip():
    # skip_first=0 -> all events count.
    events = [
        _event(0, 10, component="rollout", rank=0, tag="predict"),
        _event(10, 20, component="rollout", rank=0, tag="predict"),
    ]
    out = compute_component_call_averages(events, skip_first=0)
    assert out["rollout"]["remaining_count"] == 2
    assert out["rollout"]["mean_duration_s"] == 10.0
