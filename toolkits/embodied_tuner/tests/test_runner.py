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

"""Unit tests for :mod:`toolkits.embodied_tuner.runner`.

Tests are hermetic: no RLinf launch, no real Ray. They use small Python
``time.sleep`` subprocesses for the timeout/cleanup paths and injected
``ray_stop_hook`` / ``pgrep_runner`` / ``kill_runner`` callables for
deterministic verification.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

from toolkits.embodied_tuner.override_wrapper import LaunchSpec
from toolkits.embodied_tuner.runner import (
    TrialOutcome,
    TrialRunner,
    TrialRunnerError,
)


def _make_spec(
    *,
    log_dir: Path,
    sleep_seconds: float = 0.05,
    trial_id: str = "t0",
    extra_env: dict[str, str] | None = None,
) -> LaunchSpec:
    """Build a :class:`LaunchSpec` that runs a short Python sleep."""
    env = {"RLINF_TUNER_TRIAL_ID": trial_id, "PATH": "/usr/bin:/bin"}
    if extra_env:
        env.update(extra_env)
    argv = (
        sys.executable,
        "-c",
        f"import time; time.sleep({sleep_seconds})",
    )
    return LaunchSpec(
        argv=argv,
        env=env,
        log_dir=log_dir,
        config_name="stub",
        baseline_overrides=(),
        user_overrides=(),
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_launch_completes_within_timeout(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "ok", sleep_seconds=0.02)
    runner = TrialRunner(
        timeout_seconds=5.0,
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    assert isinstance(outcome, TrialOutcome)
    assert outcome.returncode == 0
    assert outcome.timed_out is False
    assert outcome.cleanup_outcome == "ok"
    assert outcome.log_dir.is_dir()
    assert outcome.stdout_path.is_file()
    assert outcome.wall_clock_seconds > 0


def test_launch_uses_supplied_log_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "trials" / "trial-7"
    spec = _make_spec(log_dir=log_dir)
    runner = TrialRunner(
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    assert outcome.log_dir == log_dir
    assert log_dir.is_dir()


# ---------------------------------------------------------------------------
# Timeout / cleanup escalation
# ---------------------------------------------------------------------------


def test_launch_timeout_triggers_sigterm_path(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "slow", sleep_seconds=20.0)
    runner = TrialRunner(
        timeout_seconds=0.2,
        sigterm_grace_seconds=1.0,
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    assert outcome.timed_out is True
    # ``time.sleep`` exits on SIGTERM with a negative returncode equal to
    # ``-SIGTERM`` on POSIX; this means escalation to SIGKILL was NOT
    # required.
    assert outcome.returncode == -signal.SIGTERM
    assert outcome.cleanup_outcome == "ok"


def test_launch_timeout_escalates_to_sigkill_for_term_resistant_child(
    tmp_path: Path,
) -> None:
    spec = LaunchSpec(
        argv=(
            sys.executable,
            "-c",
            (
                "import signal, time;"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                "time.sleep(30)"
            ),
        ),
        env={"RLINF_TUNER_TRIAL_ID": "t-resist", "PATH": "/usr/bin:/bin"},
        log_dir=tmp_path / "resist",
        config_name="stub",
        baseline_overrides=(),
        user_overrides=(),
    )
    runner = TrialRunner(
        timeout_seconds=0.2,
        sigterm_grace_seconds=0.3,
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    assert outcome.timed_out is True
    assert outcome.cleanup_outcome == "sigkill_required"
    # SIGKILL exit: returncode == -SIGKILL.
    assert outcome.returncode == -signal.SIGKILL


def test_launch_rejects_non_positive_timeout(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "t")
    runner = TrialRunner()
    with pytest.raises(TrialRunnerError):
        runner.launch(spec, timeout=0)


# ---------------------------------------------------------------------------
# Profiler env merging
# ---------------------------------------------------------------------------


def test_profiler_env_added_by_default(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "prof")
    runner = TrialRunner(
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    for key in (
        "RLINF_TIMELINE",
        "RLINF_TIMELINE_WORKER_TIMER",
        "RLINF_TIMELINE_ACTOR_TRAINING",
        "RLINF_TIMELINE_DIR",
        "RLINF_NVITOP",
        "RLINF_NVML",
    ):
        assert outcome.spec.env[key] == "1" or key == "RLINF_TIMELINE_DIR"
    assert outcome.spec.env["RLINF_TIMELINE_DIR"] == "auto"


def test_no_profiler_skips_timeline_flags(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "noprof")
    runner = TrialRunner(
        disable_profiler=True,
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    assert "RLINF_TIMELINE" not in outcome.spec.env
    assert "RLINF_TIMELINE_WORKER_TIMER" not in outcome.spec.env
    # Memory telemetry still on.
    assert outcome.spec.env["RLINF_NVITOP"] == "1"


def test_no_collect_memory_skips_nvitop_only(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "nomem")
    runner = TrialRunner(
        disable_memory_telemetry=True,
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    assert outcome.spec.env["RLINF_TIMELINE"] == "1"
    assert "RLINF_NVITOP" not in outcome.spec.env
    assert "RLINF_NVML" not in outcome.spec.env


def test_extra_env_overrides_profiler_defaults(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "extra")
    runner = TrialRunner(
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
        extra_env={"RLINF_TIMELINE_DIR": "/tmp/custom"},
    )
    outcome = runner.launch(spec)
    assert outcome.spec.env["RLINF_TIMELINE_DIR"] == "/tmp/custom"


# ---------------------------------------------------------------------------
# Ray stop hook + scoped orphan sweep
# ---------------------------------------------------------------------------


def test_ray_stop_hook_invoked(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "ray")
    called: dict[str, int] = {"n": 0}

    def hook() -> bool:
        called["n"] += 1
        return True

    runner = TrialRunner(
        ray_stop_hook=hook,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    runner.launch(spec)
    assert called["n"] == 1


def test_ray_stop_hook_failure_sets_cleanup_outcome(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "rayfail")
    runner = TrialRunner(
        ray_stop_hook=lambda: False,
        pgrep_runner=lambda pat: [],
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    assert outcome.cleanup_outcome == "ray_stop_failed"


def test_pgrep_orphans_killed(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "orphans", trial_id="t-orph")
    killed: list[tuple[list[int], int]] = []
    pgrep_calls: list[str] = []

    sweep_state = {"n": 0}

    def pgrep(pat: str) -> list[int]:
        pgrep_calls.append(pat)
        sweep_state["n"] += 1
        # First sweep returns orphans; second sweep (after kill) returns empty.
        return [99001, 99002] if sweep_state["n"] == 1 else []

    def killer(pids: list[int], sig: int) -> None:
        killed.append((list(pids), sig))

    runner = TrialRunner(
        ray_stop_hook=lambda: True,
        pgrep_runner=pgrep,
        kill_runner=killer,
    )
    outcome = runner.launch(spec)
    assert killed == [([99001, 99002], signal.SIGKILL)]
    assert pgrep_calls == ["RLINF_TUNER_TRIAL_ID=t-orph"] * 2
    assert outcome.cleanup_outcome == "orphans_killed"


def test_pgrep_orphans_remain_sets_outcome(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "stuck", trial_id="t-stuck")
    runner = TrialRunner(
        ray_stop_hook=lambda: True,
        pgrep_runner=lambda pat: [42],  # always reports orphans
        kill_runner=lambda pids, sig: None,
    )
    outcome = runner.launch(spec)
    assert outcome.cleanup_outcome == "orphans_remain"


def test_pgrep_runner_error_is_swallowed(tmp_path: Path) -> None:
    spec = _make_spec(log_dir=tmp_path / "pgerr")

    def bad_pgrep(_pat: str) -> list[int]:
        raise RuntimeError("pgrep blew up")

    runner = TrialRunner(
        ray_stop_hook=lambda: True,
        pgrep_runner=bad_pgrep,
        kill_runner=lambda pids, sig: None,
    )
    # Should not raise; cleanup_outcome stays ok because no orphans found.
    outcome = runner.launch(spec)
    assert outcome.cleanup_outcome == "ok"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_log_dir_creation_failure_raises(tmp_path: Path) -> None:
    blocking = tmp_path / "block"
    blocking.write_text("file-where-dir-should-be")
    spec = _make_spec(log_dir=blocking / "nested")
    runner = TrialRunner()
    with pytest.raises(TrialRunnerError):
        runner.launch(spec)
