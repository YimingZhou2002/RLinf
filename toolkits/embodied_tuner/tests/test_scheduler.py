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


def test_preflight_exhausted_stops_loop_without_running_trial(tmp_path: Path) -> None:
    """After preflight_retries exhausted, the scheduler stops with a
    dedicated reason and DOES NOT launch the runner with a known-bad delta.

    This is the Round-3 behaviour change driven by Codex review of Round 2.
    """
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[],  # runner_fn must not be called at all
        preflight_results=[False, False, False, False],
    )
    critic = FakeCritic.from_deltas({}, {}, {}, {})
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=5, budget_seconds=999, max_oom=99, preflight_retries=3)
    )
    result = scheduler.run()
    assert result.stop_reason == "preflight_exhausted"
    assert result.trial_count == 0
    assert factory.runner_calls == [], (
        f"runner_fn must not be called after preflight exhaustion, but was: {factory.runner_calls}"
    )
    # The synthetic CONFIG_INVALID ledger entry is still recorded so the
    # campaign report explains why the loop stopped.
    loaded = Ledger(result.ledger_path, fsync_on_append=False).load()
    assert len(loaded.entries) == 1
    assert loaded.entries[0].status == "FAILED"
    assert loaded.entries[0].failure_mode == "CONFIG_INVALID"


def test_preflight_feedback_reaches_next_critic_call(tmp_path: Path) -> None:
    """Codex Round-2 review: the critic must receive preflight rejection
    reasons as feedback on the next prompt.
    """
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[100.0],
        preflight_results=[False, True],
    )
    critic = FakeCritic.from_deltas({}, {})
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=1, budget_seconds=999, max_oom=99, preflight_retries=3)
    )
    scheduler.run()
    # First propose: no preflight_feedback. Second propose: has it.
    assert critic.calls[0][2] is None
    assert critic.calls[1][2] is not None
    assert "Preflight rejected" in critic.calls[1][2]
    assert "stub rejection" in critic.calls[1][2]


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


# ---------------------------------------------------------------------------
# Bitter-lesson integration
# ---------------------------------------------------------------------------


def _output_with_lesson(delta, trigger, rule):
    from toolkits.embodied_tuner.critic import ProposedLesson

    return CriticOutput(
        delta=dict(delta),
        rationale=Rationale(summary="fake"),
        bitter_lesson=ProposedLesson(trigger=trigger, rule=rule),
    )


def test_scheduler_records_lesson_after_oom(tmp_path: Path) -> None:
    from toolkits.embodied_tuner.lessons import LessonBook, canonical_delta_signature

    # Trial 0: OOM on rollout.enable_offload=False. Trial 1's response
    # writes the lesson explaining what to avoid; trial 2 is a normal
    # recovery move.
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[None, 50.0, 40.0],
        failure_modes=[FailureMode.OOM, FailureMode.NONE, FailureMode.NONE],
        statuses=[Status.FAILED, Status.OK, Status.OK],
    )
    critic = FakeCritic.from_outputs(
        CriticOutput(delta={"rollout.enable_offload": False}, rationale=Rationale(summary="try")),
        _output_with_lesson(
            {"actor.micro_batch_size": 20},
            trigger="trial 0 OOM after rollout.enable_offload=False",
            rule="do not disable rollout offload while envs=8",
        ),
        CriticOutput(delta={"actor.micro_batch_size": 10}, rationale=Rationale(summary="ok")),
    )
    scheduler = factory.build(critic, BudgetConfig(max_trials=3, budget_seconds=999, max_oom=99))
    scheduler.run()

    # Lesson persisted alongside ledger.
    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    lessons = book.load()
    assert len(lessons) == 1
    assert lessons[0].trial_idx == 0
    assert lessons[0].failure_mode == "OOM"
    assert lessons[0].delta_signature == canonical_delta_signature(
        {"rollout.enable_offload": False}
    )
    assert "rollout offload" in lessons[0].rule


def test_scheduler_deduplicates_repeated_oom_lesson(tmp_path: Path) -> None:
    from toolkits.embodied_tuner.lessons import LessonBook

    # Same failing delta OOMs twice. The critic writes the same rule
    # both times — but the book keeps only one entry.
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[None, None, None, 50.0],
        failure_modes=[
            FailureMode.OOM,
            FailureMode.NONE,   # recovery
            FailureMode.OOM,    # same delta again
            FailureMode.NONE,   # recovery
        ],
        statuses=[Status.FAILED, Status.OK, Status.FAILED, Status.OK],
    )
    critic = FakeCritic.from_outputs(
        CriticOutput(delta={"rollout.enable_offload": False}, rationale=Rationale(summary="try")),
        _output_with_lesson({"actor.micro_batch_size": 20},
                            trigger="OOM again", rule="avoid the offload flip"),
        CriticOutput(delta={"rollout.enable_offload": False}, rationale=Rationale(summary="retry")),
        _output_with_lesson({"actor.micro_batch_size": 10},
                            trigger="OOM once more", rule="avoid the offload flip"),
    )
    scheduler = factory.build(critic, BudgetConfig(max_trials=4, budget_seconds=999, max_oom=99))
    scheduler.run()

    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    assert len(book.load()) == 1


def test_scheduler_passes_growing_lessons_to_critic(tmp_path: Path) -> None:
    """After a lesson is recorded, it appears in the critic's bitter_lessons arg
    on every subsequent propose() call — even past the history_window."""
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[None, 50.0, 40.0, 30.0],
        failure_modes=[FailureMode.OOM] + [FailureMode.NONE] * 3,
        statuses=[Status.FAILED] + [Status.OK] * 3,
    )
    critic = FakeCritic.from_outputs(
        CriticOutput(delta={"rollout.enable_offload": False}, rationale=Rationale(summary="try")),
        _output_with_lesson({"actor.micro_batch_size": 20},
                            trigger="oom", rule="do not X"),
        CriticOutput(delta={"actor.micro_batch_size": 10}, rationale=Rationale(summary="ok")),
        CriticOutput(delta={"actor.micro_batch_size": 5}, rationale=Rationale(summary="ok")),
    )
    scheduler = factory.build(
        critic,
        BudgetConfig(max_trials=4, budget_seconds=999, max_oom=99, history_window=1),
    )
    scheduler.run()

    # calls tuples: (history_len, knobs, preflight_feedback, lesson_count)
    lesson_counts = [call[3] for call in critic.calls]
    # Round 0: no lessons yet. Round 1: still none (critic is about to
    # write one now). Round 2 and 3: one lesson each — persisting past
    # history_window=1.
    assert lesson_counts == [0, 0, 1, 1]


def test_scheduler_recovers_lessons_from_disk_on_restart(tmp_path: Path) -> None:
    from toolkits.embodied_tuner.lessons import LessonBook, BitterLesson, canonical_delta_signature

    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    book.add(
        BitterLesson(
            trigger="prior run OOM",
            rule="do not flip rollout offload",
            trial_idx=7,
            failure_mode="OOM",
            delta_signature=canonical_delta_signature({"rollout.enable_offload": False}),
        )
    )

    # Fresh scheduler on the same ledger dir: it should see the prior
    # lesson on the very first propose() call.
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[50.0],
        failure_modes=[FailureMode.NONE],
        statuses=[Status.OK],
    )
    critic = FakeCritic.from_deltas({"actor.micro_batch_size": 20})
    scheduler = factory.build(critic, BudgetConfig(max_trials=1, budget_seconds=999, max_oom=99))
    scheduler.run()

    assert critic.calls[0][3] == 1  # one lesson visible on first prompt


def test_scheduler_ignores_lesson_when_last_trial_was_ok(tmp_path: Path) -> None:
    """A critic that emits a bitter_lesson after a successful trial
    contributes nothing to the persistent store — there is no failure
    to attribute it to."""
    from toolkits.embodied_tuner.lessons import LessonBook

    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[50.0, 40.0],
        failure_modes=[FailureMode.NONE, FailureMode.NONE],
        statuses=[Status.OK, Status.OK],
    )
    critic = FakeCritic.from_outputs(
        CriticOutput(delta={"actor.micro_batch_size": 20}, rationale=Rationale(summary="ok")),
        _output_with_lesson({"actor.micro_batch_size": 10},
                            trigger="nothing failed", rule="ignore me"),
    )
    scheduler = factory.build(critic, BudgetConfig(max_trials=2, budget_seconds=999, max_oom=99))
    scheduler.run()

    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    assert book.load() == ()


# ---------------------------------------------------------------------------
# Delta accumulation across trials
# ---------------------------------------------------------------------------


def test_runner_and_preflight_receive_cumulative_delta(tmp_path: Path) -> None:
    """Each round's runner/preflight input is the cumulative override set.

    A knob set in round 0 must remain in the round 1/2 payloads even
    when the critic's incremental proposal that round doesn't repeat
    it — otherwise the actual training config silently reverts to
    baseline while the critic prompt says it's still changed.
    """
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[100.0, 90.0, 80.0],
    )
    # Round 0: change A. Round 1: change B (A must persist).
    # Round 2: overwrite A (B must persist).
    critic = FakeCritic.from_deltas(
        {"actor.micro_batch_size": 32},
        {"rollout.gpu_memory_utilization": 0.9},
        {"actor.micro_batch_size": 64},
    )
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=3, budget_seconds=999, max_oom=99)
    )
    scheduler.run()

    assert [d for _, d in factory.runner_calls] == [
        {"actor.micro_batch_size": 32},
        {"actor.micro_batch_size": 32, "rollout.gpu_memory_utilization": 0.9},
        {"actor.micro_batch_size": 64, "rollout.gpu_memory_utilization": 0.9},
    ]


def test_ledger_records_effective_and_proposed_delta(tmp_path: Path) -> None:
    """The ledger's ``delta`` field is the cumulative override set that
    actually ran; ``proposed_delta`` captures the critic's incremental
    change for the round (used for lesson attribution and audit)."""
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[100.0, 90.0],
    )
    critic = FakeCritic.from_deltas(
        {"actor.micro_batch_size": 32},
        {"rollout.gpu_memory_utilization": 0.9},
    )
    scheduler = factory.build(
        critic, BudgetConfig(max_trials=2, budget_seconds=999, max_oom=99)
    )
    scheduler.run()

    entries = Ledger(tmp_path / "ledger.jsonl").load().entries
    assert entries[0].delta == {"actor.micro_batch_size": 32}
    assert entries[0].proposed_delta == {"actor.micro_batch_size": 32}
    assert entries[1].delta == {
        "actor.micro_batch_size": 32,
        "rollout.gpu_memory_utilization": 0.9,
    }
    assert entries[1].proposed_delta == {"rollout.gpu_memory_utilization": 0.9}


def test_bitter_lesson_signature_uses_incremental_proposal(tmp_path: Path) -> None:
    """A lesson recorded after a failure is attributed to the specific
    knob flip the critic chose that round, not the whole cumulative
    override stack — otherwise every lesson signature would inflate
    with unrelated prior knobs and dedupe would stop working."""
    from toolkits.embodied_tuner.lessons import LessonBook, canonical_delta_signature

    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=[50.0, None, 40.0],
        failure_modes=[FailureMode.NONE, FailureMode.OOM, FailureMode.NONE],
        statuses=[Status.OK, Status.FAILED, Status.OK],
    )
    # Round 0 sets A successfully. Round 1 adds B and OOMs. Round 2's
    # lesson should point at B alone (the flip), not {A, B}.
    critic = FakeCritic.from_outputs(
        CriticOutput(delta={"actor.micro_batch_size": 32}, rationale=Rationale(summary="ok")),
        CriticOutput(delta={"rollout.enable_offload": False}, rationale=Rationale(summary="try")),
        _output_with_lesson(
            {"actor.micro_batch_size": 10},
            trigger="OOM on rollout offload flip",
            rule="do not disable rollout offload once micro_batch >= 32",
        ),
    )
    scheduler = factory.build(critic, BudgetConfig(max_trials=3, budget_seconds=999, max_oom=99))
    scheduler.run()

    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    lessons = book.load()
    assert len(lessons) == 1
    assert lessons[0].delta_signature == canonical_delta_signature(
        {"rollout.enable_offload": False}
    )
