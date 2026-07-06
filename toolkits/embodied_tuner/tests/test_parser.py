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
