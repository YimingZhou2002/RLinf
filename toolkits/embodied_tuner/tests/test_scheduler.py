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

"""Unit tests for :mod:`toolkits.embodied_tuner.scheduler`.

All tests are hermetic: the runner and parser are stubbed via small
in-memory factories so the scheduler logic can be exercised without
real subprocesses or RLinf launches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from toolkits.embodied_tuner.critic import CriticOutput, Rationale
from toolkits.embodied_tuner.fake_critic import FakeCritic
from toolkits.embodied_tuner.ledger import Ledger
from toolkits.embodied_tuner.override_wrapper import LaunchSpec
from toolkits.embodied_tuner.parser import (
    FailureMode,
    Status,
    TrialResult,
)
from toolkits.embodied_tuner.runner import TrialOutcome
from toolkits.embodied_tuner.scheduler import (
    BudgetConfig,
    PreflightOutcome,
    Scheduler,
)


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


def _fake_outcome(log_dir: Path, returncode: int = 0) -> TrialOutcome:
    spec = LaunchSpec(
        argv=(),
        env={},
        log_dir=log_dir,
        config_name="stub",
        baseline_overrides=(),
        user_overrides=(),
    )
    return TrialOutcome(
        log_dir=log_dir,
        returncode=returncode,
        timed_out=False,
        wall_clock_seconds=0.0,
        cleanup_outcome="ok",
        stdout_path=log_dir / "run_embodiment.log",
        spec=spec,
    )


def _fake_result(
    log_dir: Path,
    *,
    objective: float | None = None,
    status: Status = Status.OK,
    failure_mode: FailureMode = FailureMode.NONE,
) -> TrialResult:
    return TrialResult(
        log_dir=log_dir,
        status=status,
        failure_mode=failure_mode,
        objective=objective,
        step_time_seconds=200.0 if objective is not None else None,
        num_trajectories=18 if objective is not None else None,
    )


def _ok_preflight(delta: Mapping[str, Any], log_dir: Path) -> PreflightOutcome:
    return PreflightOutcome(
        ok=True,
        errors=(),
        resolved_config_sha="deadbeef",
        log_dir=log_dir,
        delta=delta,
    )


@dataclass
class _SchedulerFactory:
    """Builds a Scheduler against in-memory stubs."""

    tmp_path: Path
    objectives: list[float | None] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)
    statuses: list[Status] = field(default_factory=list)
    preflight_results: list[bool] = field(default_factory=list)
    runner_calls: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    parser_calls: list[Path] = field(default_factory=list)
    clock_ticks: list[float] = field(default_factory=list)

    def build(
        self,
        critic: FakeCritic,
        budget: BudgetConfig | None = None,
        baseline_knobs: dict[str, Any] | None = None,
    ) -> Scheduler:
        ledger = Ledger(self.tmp_path / "ledger.jsonl", fsync_on_append=False)

        def runner_fn(delta: Mapping[str, Any], preflight, trial_idx: int) -> TrialOutcome:
            self.runner_calls.append((trial_idx, dict(delta)))
            log_dir = self.tmp_path / f"trial-{trial_idx}"
            log_dir.mkdir(parents=True, exist_ok=True)
            return _fake_outcome(log_dir)

        def parser_fn(outcome: TrialOutcome) -> TrialResult:
            self.parser_calls.append(outcome.log_dir)
            idx = len(self.parser_calls) - 1
            obj = self.objectives[idx] if idx < len(self.objectives) else None
            fm = self.failure_modes[idx] if idx < len(self.failure_modes) else FailureMode.NONE
            st = self.statuses[idx] if idx < len(self.statuses) else Status.OK
            return _fake_result(outcome.log_dir, objective=obj, status=st, failure_mode=fm)

        def preflight_fn(delta: Mapping[str, Any]) -> PreflightOutcome:
            log_dir = self.tmp_path / f"pf-{len(self.runner_calls)}"
            if self.preflight_results:
                allowed = self.preflight_results.pop(0)
            else:
                allowed = True
            if allowed:
                return _ok_preflight(delta, log_dir)
            return PreflightOutcome(
                ok=False,
                errors=("schema: stub rejection",),
                resolved_config_sha=None,
                log_dir=log_dir,
                delta=delta,
            )

        clock_ticks = iter(self.clock_ticks) if self.clock_ticks else None

        def clock() -> float:
            if clock_ticks is None:
                return 0.0
            try:
                return next(clock_ticks)
            except StopIteration:
                return 0.0

        return Scheduler(
            critic=critic,
            runner_fn=runner_fn,
            parser_fn=parser_fn,
            preflight_fn=preflight_fn,
            ledger=ledger,
            budget=budget or BudgetConfig(max_trials=3, budget_seconds=999.0, max_oom=99),
            baseline_knobs=baseline_knobs or {},
            clock=clock,
        )


# ---------------------------------------------------------------------------
# Budget thresholds
# ---------------------------------------------------------------------------


def test_max_trials_terminates_after_n(tmp_path: Path) -> None:
    factory = _SchedulerFactory(tmp_path=tmp_path, objectives=[100.0, 90.0, 80.0])
    critic = FakeCritic.from_deltas({}, {}, {})
    scheduler = factory.build(critic, BudgetConfig(max_trials=3, budget_seconds=999, max_oom=99))
    result = scheduler.run()
    assert result.stop_reason == "max_trials_reached"
    assert result.trial_count == 3


def test_max_trials_zero_terminates_without_any_trial(tmp_path: Path) -> None:
    factory = _SchedulerFactory(tmp_path=tmp_path)
    critic = FakeCritic.from_deltas({})
    scheduler = factory.build(critic, BudgetConfig(max_trials=0))
    result = scheduler.run()
    assert result.stop_reason == "no_trials_run"
    assert result.trial_count == 0
    assert factory.runner_calls == []


def test_budget_seconds_elapsed_terminates(tmp_path: Path) -> None:
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[100.0, 90.0],
        clock_ticks=[0.0, 5.0, 11.0],  # 11 > budget_seconds=10
    )
    critic = FakeCritic.from_deltas({}, {})
    scheduler = factory.build(critic, BudgetConfig(max_trials=10, budget_seconds=10.0, max_oom=99))
    result = scheduler.run()
    assert result.stop_reason == "budget_seconds_elapsed"


def test_oom_cap_exceeded_terminates(tmp_path: Path) -> None:
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[None, None, None],
        failure_modes=[FailureMode.OOM, FailureMode.OOM, FailureMode.OOM],
        statuses=[Status.FAILED, Status.FAILED, Status.FAILED],
    )
    critic = FakeCritic.from_deltas({}, {}, {}, {})
    scheduler = factory.build(critic, BudgetConfig(max_trials=10, budget_seconds=999, max_oom=2))
    result = scheduler.run()
    assert result.stop_reason == "oom_cap_exceeded"
    assert result.oom_count == 3


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------


def test_plateau_terminates_after_patience_consecutive_small_improvements(tmp_path: Path) -> None:
    # 5 trials with progressively tiny improvements:
    # trial 0: obj 100 (warmup)
    # trial 1: 80 (delta 20%)
    # trial 2: 79.5 (delta 0.6% -- plateau)
    # trial 3: 79.0 (delta 0.6% -- plateau)
    # trial 4: 78.5 (delta 0.6% -- plateau)
    # patience=3 -> after 3 consecutive small improvements, terminate at trial 5
    objectives = [100.0, 80.0, 79.5, 79.0, 78.5, 78.0]
    factory = _SchedulerFactory(tmp_path=tmp_path, objectives=objectives)
    critic = FakeCritic.from_deltas({}, {}, {}, {}, {}, {})
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=10, budget_seconds=999, max_oom=99, patience=3, epsilon=0.02)
    )
    result = scheduler.run()
    assert result.stop_reason == "plateau"
    # Plateau fires after the 5th trial (5 eligible + 1 baseline = 6 entries needed).
    assert result.trial_count == 5


def test_plateau_does_not_fire_on_continued_improvements(tmp_path: Path) -> None:
    # Each step improves by more than epsilon=2%, so plateau never fires;
    # we should terminate via max_trials instead.
    objectives = [100.0, 80.0, 60.0, 40.0]
    factory = _SchedulerFactory(tmp_path=tmp_path, objectives=objectives)
    critic = FakeCritic.from_deltas({}, {}, {}, {})
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=4, budget_seconds=999, max_oom=99, patience=3, epsilon=0.02)
    )
    result = scheduler.run()
    assert result.stop_reason == "max_trials_reached"


# ---------------------------------------------------------------------------
# Critic stagnation
# ---------------------------------------------------------------------------


def test_critic_stagnation_terminates_after_two_consecutive_stop_requests(
    tmp_path: Path,
) -> None:
    factory = _SchedulerFactory(tmp_path=tmp_path, objectives=[100.0, 90.0])
    critic = FakeCritic(
        outputs=[
            CriticOutput(delta={}, rationale=Rationale(summary="ok")),
            CriticOutput(delta={}, rationale=Rationale(summary="ok")),
            CriticOutput(delta={}, rationale=Rationale(summary="stop"), stop_requested=True),
            CriticOutput(delta={}, rationale=Rationale(summary="stop"), stop_requested=True),
        ]
    )
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=10, budget_seconds=999, max_oom=99)
    )
    result = scheduler.run()
    assert result.stop_reason == "critic_stagnation"
    assert result.trial_count == 2  # Only 2 real trials ran.


def test_single_stop_request_does_not_terminate(tmp_path: Path) -> None:
    factory = _SchedulerFactory(tmp_path=tmp_path, objectives=[100.0, 90.0])
    critic = FakeCritic(
        outputs=[
            CriticOutput(delta={}, rationale=Rationale(summary="ok")),
            CriticOutput(delta={}, rationale=Rationale(summary="stop"), stop_requested=True),
            CriticOutput(delta={}, rationale=Rationale(summary="back to work")),
        ]
    )
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=2, budget_seconds=999, max_oom=99)
    )
    result = scheduler.run()
    assert result.stop_reason == "max_trials_reached"


# ---------------------------------------------------------------------------
# Preflight failures do not consume the trial budget
# ---------------------------------------------------------------------------


def test_preflight_failure_burns_critic_retry_not_trial_slot(tmp_path: Path) -> None:
    # First preflight rejects, then accepts; the scheduler should still
    # only consume 2 trial slots even though the critic was called 3 times.
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[100.0, 90.0],
        preflight_results=[False, True, True],  # critic call 1 rejected
    )
    critic = FakeCritic.from_deltas({}, {}, {})
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=2, budget_seconds=999, max_oom=99, preflight_retries=3)
    )
    result = scheduler.run()
    assert result.stop_reason == "max_trials_reached"
    assert result.trial_count == 2
    assert len(factory.runner_calls) == 2


def test_preflight_exhausted_retries_runs_failed_proposal_anyway(tmp_path: Path) -> None:
    # All preflight calls reject. After preflight_retries, the scheduler
    # currently surfaces the failure by still running the runner with the
    # last critic_output (the runner stub doesn't care, so it returns a
    # default outcome and we get a trial entry). This keeps the loop
    # making forward progress.
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[100.0],
        preflight_results=[False, False, False, False],
    )
    critic = FakeCritic.from_deltas({}, {}, {}, {})
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=1, budget_seconds=999, max_oom=99, preflight_retries=3)
    )
    result = scheduler.run()
    assert result.stop_reason == "max_trials_reached"
    assert result.trial_count == 1


# ---------------------------------------------------------------------------
# Ledger integration
# ---------------------------------------------------------------------------


def test_ledger_records_every_trial(tmp_path: Path) -> None:
    factory = _SchedulerFactory(tmp_path=tmp_path, objectives=[50.0, 30.0, 40.0])
    critic = FakeCritic.from_deltas({}, {}, {})
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=3, budget_seconds=999, max_oom=99)
    )
    result = scheduler.run()
    ledger_entries = Ledger(result.ledger_path, fsync_on_append=False).load().entries
    assert len(ledger_entries) == 3
    assert [e.objective for e in ledger_entries] == [50.0, 30.0, 40.0]
    # best_entry should be the lowest-objective entry.
    assert result.best_entry is not None
    assert result.best_entry.objective == 30.0


def test_best_entry_none_when_no_successful_trial(tmp_path: Path) -> None:
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[None, None],
        failure_modes=[FailureMode.OOM, FailureMode.WORKER_CRASH],
        statuses=[Status.FAILED, Status.FAILED],
    )
    critic = FakeCritic.from_deltas({}, {})
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=2, budget_seconds=999, max_oom=99)
    )
    result = scheduler.run()
    assert result.best_entry is None


# ---------------------------------------------------------------------------
# Critic-failure path
# ---------------------------------------------------------------------------


def test_critic_exhaustion_terminates_loop(tmp_path: Path) -> None:
    factory = _SchedulerFactory(tmp_path=tmp_path)
    critic = FakeCritic(outputs=[])  # immediately raises CriticError
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=3, budget_seconds=999, max_oom=99)
    )
    result = scheduler.run()
    assert result.stop_reason == "critic_failure"
    assert result.trial_count == 0
