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

"""Unit tests for :mod:`toolkits.embodied_tuner.parser`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolkits.embodied_tuner.parser import (
    FailureMode,
    MetricStep,
    ParserInvariantError,
    Status,
    TimelineSummary,
    TrialResult,
    compute_objective,
    parse_metrics_log,
    parse_timeline,
    parse_trial,
    select_best,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_LOG = REPO_ROOT / "logs" / "20260629-07:25:33-maniskill_ppo_openvla"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _metric_block(global_step: int, total_steps: int, step_time: float, num_traj: int | None) -> str:
    """Synthesise a minimal MetricTable block the parser accepts."""
    traj_line = (
        f"│num_trajectories={num_traj}                    │"
        if num_traj is not None
        else "│episode_len=55.5                          │"
    )
    return "\n".join(
        [
            "╭────────────────────────╮",
            "├──── Metric Table ─────┤",
            f"│ Global Step:    {global_step}/{total_steps} │ Step Time: {step_time}s │",
            "├──── Time ────┤",
            "│env/interact=275.4 │ rollout/generate_one_epoch=268.8 │",
            "│actor/run_training=21.371 │ sync_weights=9.231 │",
            "├──── Environment ─────┤",
            traj_line,
            "╰────────────────────────╯",
        ]
    )


def _write_metrics_log(path: Path, steps: list[tuple[int, int, float, int | None]]) -> None:
    blocks = [_metric_block(gs, ts, st, nt) for (gs, ts, st, nt) in steps]
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _write_timeline(timeline_dir: Path, events: list[dict]) -> None:
    timeline_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for event in events:
        grouped.setdefault((event["component"], event["rank"]), []).append(event)
    for (component, rank), records in grouped.items():
        path = timeline_dir / f"{component}_rank{rank}.jsonl"
        with path.open("w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# parse_metrics_log
# ---------------------------------------------------------------------------


def test_parse_metrics_log_against_synthetic_3_blocks(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.log"
    _write_metrics_log(
        metrics,
        [(1, 3, 360.0, 18), (2, 3, 200.0, 18), (3, 3, 210.0, 18)],
    )
    steps = parse_metrics_log(metrics)
    assert len(steps) == 3
    assert steps[0].global_step == 1 and steps[0].total_steps == 3
    assert steps[0].step_time_seconds == pytest.approx(360.0)
    assert steps[0].num_trajectories == 18
    assert steps[1].step_time_seconds == pytest.approx(200.0)
    # Time-section keys are exposed for critic context.
    assert steps[0].time_keys.get("env/interact") == pytest.approx(275.4)


def test_parse_metrics_log_against_live_baseline() -> None:
    assert LIVE_LOG.is_dir(), LIVE_LOG
    steps = parse_metrics_log(LIVE_LOG / "metrics.log")
    # The live log captured a single max_epochs=1 trial; expect ≥1 block.
    assert len(steps) >= 1
    final = steps[-1]
    assert final.num_trajectories == 18
    assert final.step_time_seconds == pytest.approx(359.973)
    assert final.time_keys["env/interact"] == pytest.approx(275.4)


def test_parse_metrics_log_returns_empty_for_text_without_blocks(tmp_path: Path) -> None:
    (tmp_path / "metrics.log").write_text("this file has no MetricTable blocks\n")
    assert parse_metrics_log(tmp_path / "metrics.log") == ()


# ---------------------------------------------------------------------------
# compute_objective
# ---------------------------------------------------------------------------


def _make_steps(values: list[tuple[float, int | None]]) -> tuple[MetricStep, ...]:
    return tuple(
        MetricStep(global_step=i + 1, total_steps=len(values), step_time_seconds=t, num_trajectories=n)
        for i, (t, n) in enumerate(values)
    )


def test_compute_objective_averages_all_steps() -> None:
    steps = _make_steps([(360.0, 18), (200.0, 18), (210.0, 18)])
    obj, avg, reason = compute_objective(steps)
    assert reason is None
    assert avg == pytest.approx((360.0 + 200.0 + 210.0) / 3)
    assert obj == pytest.approx(avg / 18)


def test_compute_objective_accepts_single_step() -> None:
    obj, avg, reason = compute_objective(_make_steps([(360.0, 18)]))
    assert reason is None
    assert avg == pytest.approx(360.0)
    assert obj == pytest.approx(360.0 / 18)


def test_compute_objective_handles_empty_input() -> None:
    obj, avg, reason = compute_objective(())
    assert obj is None and avg is None and reason == "no MetricTable blocks parsed"


def test_compute_objective_handles_missing_final_num_trajectories() -> None:
    obj, avg, reason = compute_objective(_make_steps([(360.0, 18), (200.0, None)]))
    assert obj is None
    assert avg == pytest.approx(280.0)
    assert "num_trajectories" in reason


def test_compute_objective_rejects_non_positive_num_trajectories() -> None:
    obj, _, reason = compute_objective(_make_steps([(360.0, 18), (200.0, 0)]))
    assert obj is None and "not positive" in reason


# ---------------------------------------------------------------------------
# parse_timeline
# ---------------------------------------------------------------------------


def test_parse_timeline_empty_dir_yields_empty_summary(tmp_path: Path) -> None:
    (tmp_path / "timeline").mkdir()
    summary = parse_timeline(tmp_path / "timeline")
    assert isinstance(summary, TimelineSummary)
    assert summary.per_tag == ()
    assert summary.window_start is None


def test_parse_timeline_against_live_baseline() -> None:
    summary = parse_timeline(LIVE_LOG / "timeline")
    assert summary.window_start is not None
    assert summary.window_end > summary.window_start
    assert summary.per_tag, "expected at least one headline tag in live timeline"
    # actor/sync_model_to_rollout appears for ranks 0..7 in the live data.
    tags = {(t.component, t.tag) for t in summary.per_tag}
    assert ("actor", "actor/sync_model_to_rollout") in tags
    # Stall fractions live in [0, 1] for every non-excluded component
    # (runner is dropped — it emits a single wrapper event that gives 0
    # stall by construction).
    for component, fraction in summary.stall_fraction_by_component.items():
        assert 0.0 <= fraction <= 1.0, f"{component}: {fraction}"
    assert "runner" not in summary.stall_fraction_by_component


def test_parse_timeline_synthetic(tmp_path: Path) -> None:
    timeline_dir = tmp_path / "timeline"
    _write_timeline(
        timeline_dir,
        [
            {"t0": 100.0, "t1": 110.0, "tag": "env_interact_step", "component": "env", "rank": 0},
            {"t0": 110.0, "t1": 121.0, "tag": "env_interact_step", "component": "env", "rank": 0},
            {"t0": 100.0, "t1": 100.5, "tag": "predict", "component": "rollout", "rank": 0},
        ],
    )
    summary = parse_timeline(timeline_dir)
    env_tag = [t for t in summary.per_tag if t.tag == "env_interact_step"]
    assert env_tag, "headline tag env_interact_step missing"
    assert env_tag[0].call_count == 2
    assert env_tag[0].duration_max == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# parse_trial classification
# ---------------------------------------------------------------------------


def test_parse_trial_missing_metrics_log(tmp_path: Path) -> None:
    result = parse_trial(tmp_path, returncode=0)
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.METRICS_MISSING


def test_parse_trial_single_step_computes_step_time(tmp_path: Path) -> None:
    _write_metrics_log(tmp_path / "metrics.log", [(1, 1, 360.0, 18)])
    result = parse_trial(tmp_path, returncode=0)
    # timeline/ absent → METRICS_PARTIAL (objective is zeroed for
    # best-config eligibility), but the single MetricTable block is
    # measurable and step_time flows through.
    assert result.status is Status.OK
    assert result.failure_mode is FailureMode.METRICS_PARTIAL
    assert result.step_time_seconds == pytest.approx(360.0)
    assert result.objective is None
    assert "timeline" in result.reason


def test_parse_trial_single_step_with_timeline_is_ok_none(tmp_path: Path) -> None:
    _write_metrics_log(tmp_path / "metrics.log", [(1, 1, 360.0, 18)])
    (tmp_path / "timeline").mkdir()
    result = parse_trial(tmp_path, returncode=0)
    assert result.status is Status.OK
    assert result.failure_mode is FailureMode.NONE
    assert result.step_time_seconds == pytest.approx(360.0)
    assert result.objective == pytest.approx(360.0 / 18)


def test_parse_trial_three_steps_with_timeline_is_ok_none(tmp_path: Path) -> None:
    _write_metrics_log(
        tmp_path / "metrics.log",
        [(1, 3, 360.0, 18), (2, 3, 200.0, 18), (3, 3, 210.0, 18)],
    )
    _write_timeline(
        tmp_path / "timeline",
        [
            {"t0": 0.0, "t1": 1.0, "tag": "env_interact_step", "component": "env", "rank": 0},
            {"t0": 1.0, "t1": 2.0, "tag": "env_interact_step", "component": "env", "rank": 0},
        ],
    )
    result = parse_trial(tmp_path, returncode=0)
    assert result.status is Status.OK
    assert result.failure_mode is FailureMode.NONE
    assert result.objective == pytest.approx((360.0 + 200.0 + 210.0) / 3 / 18)
    assert result.num_trajectories == 18
    assert result.timeline_summary is not None


def test_parse_trial_oom_classification(tmp_path: Path) -> None:
    _write_metrics_log(tmp_path / "metrics.log", [(1, 1, 360.0, 18)])
    stderr_path = tmp_path / "stderr.log"
    stderr_path.write_text("...torch.cuda.OutOfMemoryError: CUDA out of memory...\n")
    result = parse_trial(tmp_path, returncode=1, stderr_path=stderr_path)
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.OOM


def test_parse_trial_worker_crash_classification(tmp_path: Path) -> None:
    _write_metrics_log(tmp_path / "metrics.log", [(1, 1, 360.0, 18)])
    stderr_path = tmp_path / "stderr.log"
    stderr_path.write_text("RayActorError: actor died\n")
    result = parse_trial(tmp_path, returncode=1, stderr_path=stderr_path)
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.WORKER_CRASH


def test_parse_trial_timeout_short_circuits(tmp_path: Path) -> None:
    # Even with a perfectly fine metrics.log, timed_out=True wins.
    _write_metrics_log(
        tmp_path / "metrics.log",
        [(1, 3, 360.0, 18), (2, 3, 200.0, 18), (3, 3, 210.0, 18)],
    )
    result = parse_trial(tmp_path, timed_out=True, returncode=-15)
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.TIMEOUT


def test_parse_trial_failure_mode_override_short_circuits(tmp_path: Path) -> None:
    result = parse_trial(
        tmp_path,
        failure_mode_override=FailureMode.CONFIG_INVALID,
    )
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.CONFIG_INVALID
    assert "CONFIG_INVALID" in result.reason


def test_parse_trial_rejects_override_none(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        parse_trial(tmp_path, failure_mode_override=FailureMode.NONE)


def test_parse_trial_nonzero_returncode_without_oom_is_worker_crash(tmp_path: Path) -> None:
    _write_metrics_log(
        tmp_path / "metrics.log",
        [(1, 3, 360.0, 18), (2, 3, 200.0, 18), (3, 3, 210.0, 18)],
    )
    result = parse_trial(tmp_path, returncode=2, stderr_path=None)
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.WORKER_CRASH


def test_parse_trial_missing_timeline_yields_metrics_partial(tmp_path: Path) -> None:
    _write_metrics_log(
        tmp_path / "metrics.log",
        [(1, 3, 360.0, 18), (2, 3, 200.0, 18), (3, 3, 210.0, 18)],
    )
    result = parse_trial(tmp_path, returncode=0)
    assert result.status is Status.OK
    assert result.failure_mode is FailureMode.METRICS_PARTIAL
    assert "timeline" in result.reason


def test_parse_trial_oom_before_metrics_missing(tmp_path: Path) -> None:
    """Codex Round-2 review: an OOM-killed trial that never wrote
    metrics.log must be classified OOM, NOT METRICS_MISSING.
    """
    # No metrics.log written. stdout_path explicitly contains OOM text.
    stdout = tmp_path / "run_embodiment.log"
    stdout.write_text("torch.cuda.OutOfMemoryError: CUDA out of memory.\n")
    result = parse_trial(tmp_path, returncode=1, stderr_path=stdout)
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.OOM
    # Metrics weren't writeable but that's not the classification reason.
    assert "metrics.log" not in result.reason


def test_parse_trial_defaults_stderr_path_to_run_embodiment_log(tmp_path: Path) -> None:
    """When the caller omits stderr_path, the parser falls back to
    LOG_DIR/run_embodiment.log (the runner's merged stdout+stderr file).
    """
    (tmp_path / "run_embodiment.log").write_text(
        "training started\n"
        "...\n"
        "torch.cuda.OutOfMemoryError: CUDA out of memory\n"
    )
    _write_metrics_log(tmp_path / "metrics.log", [(1, 3, 360.0, 18)])
    result = parse_trial(tmp_path, returncode=1)  # NO stderr_path supplied
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.OOM


def test_parse_trial_worker_crash_via_run_embodiment_log(tmp_path: Path) -> None:
    (tmp_path / "run_embodiment.log").write_text("RayActorError: actor died unexpectedly\n")
    _write_metrics_log(tmp_path / "metrics.log", [(1, 3, 360.0, 18)])
    result = parse_trial(tmp_path, returncode=1)
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.WORKER_CRASH


# ---------------------------------------------------------------------------
# DIVISIBILITY_VIOLATION and error_excerpt
# ---------------------------------------------------------------------------


_ROUTING_ASSERT_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/repo/rlinf/workers/env/env_worker.py", line 934, in _run_interact_once\n'
    "    env_outputs = self._bootstrap_and_send_train(rollout_channel)\n"
    '  File "/repo/rlinf/workers/env/env_worker.py", line 854, in _send_train_bootstrap\n'
    "    self.send_to(\n"
    '  File "/repo/rlinf/scheduler/worker/routing.py", line 139, in get_dst_ranks\n'
    "    assert batch_size % dst_world_size == 0, (\n"
    "AssertionError: batch_size (64) must be divisible by dst_world_size (6).\n"
)


def test_parse_trial_routing_assertion_classified_as_divisibility_violation(
    tmp_path: Path,
) -> None:
    """The exact assertion from wiki §2.6 must classify as DIVISIBILITY_VIOLATION,
    NOT swallowed as WORKER_CRASH by the generic Traceback regex."""
    (tmp_path / "run_embodiment.log").write_text(
        "training started\n" + _ROUTING_ASSERT_TRACEBACK
    )
    result = parse_trial(tmp_path, returncode=1)
    assert result.status is Status.FAILED
    assert result.failure_mode is FailureMode.DIVISIBILITY_VIOLATION


def test_divisibility_violation_carries_error_excerpt(tmp_path: Path) -> None:
    """The LLM prompt gets fed the actual assertion message so it can act on it."""
    (tmp_path / "run_embodiment.log").write_text(
        "some earlier chatter\n" * 100 + _ROUTING_ASSERT_TRACEBACK
    )
    result = parse_trial(tmp_path, returncode=1)
    assert result.failure_mode is FailureMode.DIVISIBILITY_VIOLATION
    assert "batch_size (64) must be divisible by dst_world_size (6)" in result.error_excerpt
    assert "routing.py" in result.error_excerpt


def test_oom_carries_error_excerpt(tmp_path: Path) -> None:
    (tmp_path / "run_embodiment.log").write_text(
        "warmup\n" * 5
        + "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.34 GiB\n"
    )
    result = parse_trial(tmp_path, returncode=1)
    assert result.failure_mode is FailureMode.OOM
    assert "OutOfMemoryError" in result.error_excerpt


def test_metrics_missing_carries_error_excerpt_tail(tmp_path: Path) -> None:
    """When metrics.log is absent, LLM should still see the tail of the stdout log."""
    tail_marker = "hydra: encountered a compose error on this run"
    (tmp_path / "run_embodiment.log").write_text(
        "boot\n" * 10 + tail_marker + "\n"
    )
    result = parse_trial(tmp_path, returncode=0)  # returncode 0 so we hit METRICS_MISSING branch
    assert result.failure_mode is FailureMode.METRICS_MISSING
    assert tail_marker in result.error_excerpt


def test_ok_trial_has_no_error_excerpt(tmp_path: Path) -> None:
    _write_metrics_log(
        tmp_path / "metrics.log",
        [(1, 3, 360.0, 18), (2, 3, 200.0, 18), (3, 3, 210.0, 18)],
    )
    (tmp_path / "run_embodiment.log").write_text("all good\n")
    result = parse_trial(tmp_path, returncode=0)
    # (OK, METRICS_PARTIAL) because timeline is absent — still no excerpt attached.
    assert result.error_excerpt == ""


@pytest.mark.parametrize(
    "assertion_line",
    [
        # Original routing.py shape (dst_world_size).
        "AssertionError: batch_size (64) must be divisible by dst_world_size (6).",
        # Same regex path, src_world_size variant.
        "AssertionError: batch_size (32) must be divisible by src_world_size (3).",
        # validate_embodied_cfg style at rlinf/config.py:962.
        "AssertionError: total_num_envs (128) must be divisible by env_world_size (3)",
        # Actor FSDP branch, modulo shape without the word "divisible".
        "AssertionError: global_batch_size % (micro_batch_size * world_size) == 0",
        # Typo tolerance — "divisable" is a common misspelling in error messages.
        "AssertionError: X must be divisable by Y",
    ],
)
def test_divisibility_regex_generalises_across_rlinf_asserts(
    tmp_path: Path, assertion_line: str
) -> None:
    """Regex must catch divisibility asserts from any RLinf subsystem,
    not just the specific batch_size / dst_world_size wording."""
    (tmp_path / "run_embodiment.log").write_text(
        "Traceback (most recent call last):\n"
        '  File "/repo/rlinf/some/module.py", line 1, in fn\n'
        "    assert cond\n" + assertion_line + "\n"
    )
    result = parse_trial(tmp_path, returncode=1)
    assert result.failure_mode is FailureMode.DIVISIBILITY_VIOLATION, (
        f"failed to classify: {assertion_line!r} → {result.failure_mode}"
    )


def test_divisibility_regex_does_not_swallow_plain_asserts(tmp_path: Path) -> None:
    """A generic AssertionError without divisibility language must stay
    WORKER_CRASH — otherwise the widened regex would over-classify."""
    (tmp_path / "run_embodiment.log").write_text(
        "Traceback (most recent call last):\n"
        "AssertionError: expected 3 shards, got 2\n"
    )
    result = parse_trial(tmp_path, returncode=1)
    assert result.failure_mode is FailureMode.WORKER_CRASH


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------


def _make_result(
    obj: float | None,
    failure_mode: FailureMode = FailureMode.NONE,
    status: Status = Status.OK,
    *,
    log_dir: str = "trial",
) -> TrialResult:
    return TrialResult(
        log_dir=Path(log_dir),
        status=status,
        failure_mode=failure_mode,
        objective=obj,
    )


def test_select_best_picks_lowest_objective_among_ok_none() -> None:
    results = [
        _make_result(50.0, log_dir="t1"),
        _make_result(30.0, log_dir="t2"),
        _make_result(20.0, FailureMode.METRICS_PARTIAL, log_dir="t3"),
        _make_result(None, FailureMode.OOM, Status.FAILED, log_dir="t4"),
    ]
    best = select_best(results)
    assert best is not None
    assert str(best.log_dir) == "t2"


def test_select_best_returns_none_when_no_eligible() -> None:
    results = [
        _make_result(20.0, FailureMode.METRICS_PARTIAL),
        _make_result(None, FailureMode.OOM, Status.FAILED),
    ]
    assert select_best(results) is None


def test_select_best_returns_none_for_empty_input() -> None:
    assert select_best([]) is None


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_failed_with_none_failure_mode_is_invariant_violation() -> None:
    with pytest.raises(ParserInvariantError):
        TrialResult(
            log_dir=Path("/tmp/x"),
            status=Status.FAILED,
            failure_mode=FailureMode.NONE,
        )


# ---------------------------------------------------------------------------
# Timeline processor integration on TimelineSummary
# ---------------------------------------------------------------------------


def test_parse_timeline_populates_new_fields(tmp_path):
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
    ts = parse_timeline(timeline, enable_offload={"env": True})
    # critical_path keyed by global_step
    assert 0 in ts.critical_path
    # per_component_bubble populated without needing placement
    assert ts.per_component_bubble["wall_s"] == 50.0
    assert ts.per_component_bubble["per_component"]["env"]["busy_s"] == 30.0
    # raw_excerpts top-K
    assert len(ts.raw_excerpts) == 2


def test_parse_timeline_excludes_runner_from_stall_fractions(tmp_path):
    timeline = tmp_path / "timeline"
    timeline.mkdir()
    (timeline / "runner_rank0.jsonl").write_text(
        json.dumps({"t0": 0.0, "t1": 100.0, "component": "runner", "rank": 0,
                    "tag": "run", "global_step": 0}) + "\n"
    )
    (timeline / "env_rank0.jsonl").write_text(
        json.dumps({"t0": 0.0, "t1": 30.0, "component": "env", "rank": 0,
                    "tag": "env_interact_step", "global_step": 0}) + "\n"
    )
    ts = parse_timeline(timeline)
    assert "runner" not in ts.stall_fraction_by_component
    assert "runner" not in ts.per_component_bubble.get("per_component", {})


# ---------------------------------------------------------------------------
# GPU-memory summary (MemorySummary / nvitop sidecar)
# ---------------------------------------------------------------------------

def _sample_record(ts, component, rank, pid, mem_gib, util=50.0, mem_util=40.0,
                   gpu_index=0, total_gib=80.0):
    """One normalized nvitop JSONL record (post-_normalize_sample shape)."""
    return {
        "ts": float(ts),
        "component": component,
        "rank": rank,
        "pid": pid,
        "global_step": 0,
        "process_rss_gib": 1.0,
        "gpus": [
            {
                "gpu_index": gpu_index,
                "memory_total_gib": total_gib,
                "memory_used_gib": mem_gib,
                "gpu_util_percent": util,
                "memory_util_percent": mem_util,
                "processes": [
                    {
                        "pid": pid,
                        "gpu_memory_gib": mem_gib,
                        "gpu_sm_util_percent": util,
                        "gpu_memory_util_percent": mem_util,
                    }
                ],
            }
        ],
    }


def _write_nvitop_dir(nvitop_dir: Path, records_by_stem: dict) -> None:
    nvitop_dir.mkdir(parents=True, exist_ok=True)
    for stem, records in records_by_stem.items():
        with (nvitop_dir / f"{stem}.jsonl").open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")


def test_write_nvitop_summary_emits_json_sidecar_with_device_cap(tmp_path):
    from toolkits.embodied_tuner.profiler import plot_nvitop

    nvitop_dir = tmp_path / "nvitop"
    _write_nvitop_dir(
        nvitop_dir,
        {
            "actor_rank0_pid1": [_sample_record(1, "actor", 0, 1, 60.0, 100.0, 57.0)],
            "rollout_rank0_pid2": [_sample_record(1, "rollout", 0, 2, 25.0, 72.0, 30.0, gpu_index=4)],
        },
    )
    samples = plot_nvitop._load_samples(str(nvitop_dir))
    plot_nvitop.write_nvitop_summary(str(nvitop_dir), samples)

    log_path = nvitop_dir / "nvitop_summary.log"
    json_path = nvitop_dir / "nvitop_summary.json"
    assert log_path.is_file()
    assert json_path.is_file()
    side = json.loads(json_path.read_text())
    assert side["gpu_total_gib"] == 80.0
    assert side["samples"] == len(samples)
    components = {p["component"] for p in side["per_process"]}
    assert components == {"actor", "rollout"}
    # actor peak (60) > rollout peak (25) -> global peak is the actor's
    peaks = [p["max_process_gpu_mem"] for p in side["per_process"]]
    assert max(peaks) == 60.0


def test_load_memory_summary_prefers_json_sidecar(tmp_path):
    from toolkits.embodied_tuner.profiler import plot_nvitop
    from toolkits.embodied_tuner.parser import _load_memory_summary

    nvitop_dir = tmp_path / "nvitop"
    _write_nvitop_dir(
        nvitop_dir,
        {
            "actor_rank0_pid1": [
                _sample_record(1, "actor", 0, 1, 60.0, 100.0, 57.0),
                _sample_record(2, "actor", 0, 1, 61.0, 100.0, 57.0),
            ],
            "env_rank0_pid3": [_sample_record(1, "env", 0, 3, 0.5, 16.0, 1.0)],
        },
    )
    # Generate the sidecar (mirrors what a real run leaves on disk).
    plot_nvitop.write_nvitop_summary(
        str(nvitop_dir), plot_nvitop._load_samples(str(nvitop_dir))
    )
    ms = _load_memory_summary(nvitop_dir)
    assert ms is not None
    assert ms.gpu_total_gib == 80.0
    assert ms.peak_gpu_mem_gib == 61.0  # true global max, NOT the first line
    assert ms.peak_mem_util_percent == 57.0
    assert len(ms.per_gpu) >= 1
    assert {p["component"] for p in ms.per_process} == {"actor", "env"}


def test_load_memory_summary_falls_back_to_log_text_without_sidecar(tmp_path):
    """Older trials have no nvitop_summary.json; the .log text must still parse."""
    from toolkits.embodied_tuner.parser import _load_memory_summary

    nvitop_dir = tmp_path / "nvitop"
    nvitop_dir.mkdir()
    # Two process lines; the SECOND has the higher peak. The legacy regex
    # grabbed the FIRST only (component-order bug). The text fallback must
    # take the global max across both.
    (nvitop_dir / "nvitop_summary.log").write_text(
        "nvitop resource summary\n"
        "samples: 10\nspan_s: 5.000\naggregate_bin_s: 1.000\n\n"
        "global_gpu_summary:\n"
        "  gpu0: avg_mem=20.000 GiB, max_mem=25.000 GiB, avg_gpu_util=40.000 %, "
        "max_gpu_util=90.000 %, avg_mem_util=30.000 %, max_mem_util=57.000 %\n\n"
        "process_summary:\n"
        "  actor/r0/pid1: avg_rss=1.000 GiB, max_rss=1.000 GiB, avg_cpu=10.000 %, "
        "max_cpu=10.000 %, avg_process_gpu_mem=10.000 GiB, max_process_gpu_mem=10.000 GiB, "
        "avg_process_gpu_util=33.000 %, max_process_gpu_util=100.000 %, gpu_indices=[0]\n"
        "  rollout/r0/pid2: avg_rss=1.000 GiB, max_rss=1.000 GiB, avg_cpu=10.000 %, "
        "max_cpu=10.000 %, avg_process_gpu_mem=40.000 GiB, max_process_gpu_mem=40.000 GiB, "
        "avg_process_gpu_util=33.000 %, max_process_gpu_util=72.000 %, gpu_indices=[4]\n"
    )
    ms = _load_memory_summary(nvitop_dir)
    assert ms is not None
    # rollout (40) > actor (10): global peak must be rollout's, not actor's.
    assert ms.peak_gpu_mem_gib == 40.0
    assert ms.peak_mem_util_percent == 57.0
    assert ms.gpu_total_gib is None  # text log has no device cap
    assert len(ms.per_process) == 2


def test_parse_trial_carries_memory_summary_on_all_branches(tmp_path):
    """An OOM-killed trial still has nvitop data; the result must carry it."""
    from toolkits.embodied_tuner.profiler import plot_nvitop
    from toolkits.embodied_tuner.nvitop_feed import NvitopFeedMode

    nvitop_dir = tmp_path / "nvitop"
    _write_nvitop_dir(
        nvitop_dir,
        {"actor_rank0_pid1": [_sample_record(1, "actor", 0, 1, 60.0, 100.0, 57.0)]},
    )
    plot_nvitop.write_nvitop_summary(
        str(nvitop_dir), plot_nvitop._load_samples(str(nvitop_dir))
    )
    # metrics.log absent + OOM stderr -> METRICS_MISSING/OOM path.
    (tmp_path / "run_embodiment.log").write_text("CUDA out of memory\n")
    result = parse_trial(tmp_path, returncode=1, stderr_path=tmp_path / "run_embodiment.log")
    assert result.failure_mode is FailureMode.OOM
    assert result.memory_summary is not None
    assert result.peak_gpu_mem_gib == 60.0


def test_parse_trial_raw_nvitop_default_off_per_component_latest_opt_in(tmp_path):
    from toolkits.embodied_tuner.nvitop_feed import NvitopFeedMode

    nvitop_dir = tmp_path / "nvitop"
    _write_nvitop_dir(
        nvitop_dir,
        {
            "actor_rank0_pid1": [_sample_record(1, "actor", 0, 1, 60.0)],
            "actor_rank1_pid2": [_sample_record(1, "actor", 1, 2, 55.0)],
            "rollout_rank0_pid3": [_sample_record(1, "rollout", 0, 3, 25.0)],
        },
    )
    # Default NONE: no raw traces in the prompt.
    result_none = parse_trial(tmp_path, nvitop_feed_mode=NvitopFeedMode.NONE)
    assert result_none.memory_summary is not None
    assert result_none.memory_summary.raw_nvitop_jsonl == {}

    # PER_COMPONENT_LATEST: one rank per component.
    result_latest = parse_trial(tmp_path, nvitop_feed_mode=NvitopFeedMode.PER_COMPONENT_LATEST)
    raw = result_latest.memory_summary.raw_nvitop_jsonl
    assert len(raw) == 2  # one actor rank + one rollout rank
    assert {s.split("_rank")[0] for s in raw} == {"actor", "rollout"}
