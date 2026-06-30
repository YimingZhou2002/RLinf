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

"""Scheduler for the embodied auto-tuner trial loop.

The scheduler glues critic + preflight + runner + parser + ledger
together. It owns the budget (``max_trials`` / ``budget_seconds`` /
``max_oom``) and the stopping rule (plateau on ``patience`` consecutive
non-failed trials with relative improvement below ``epsilon``, plus
critic-stagnation when the critic emits ``stop_requested`` twice in a
row).

Per the plan's AC-8 contract:
- Preflight failure DOES NOT count toward ``max_trials``; the critic
  gets feedback and re-proposes, up to ``preflight_retries``.
- Plateau is computed on the last ``patience`` non-failed (``Status.OK``)
  trials with non-``None`` objectives.
- ``oom_cap_exceeded`` fires when the cumulative OOM count exceeds
  ``max_oom``.
- Critic-stagnation fires when ``stop_requested=True`` arrives twice
  consecutively from the critic.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolkits.embodied_tuner.critic import (
    Critic,
    CriticError,
    CriticOutput,
    Rationale,
    TrialHistoryEntry,
)
from toolkits.embodied_tuner.ledger import Ledger, LedgerEntry, make_entry
from toolkits.embodied_tuner.parser import (
    FailureMode,
    Status,
    TrialResult,
)
from toolkits.embodied_tuner.runner import TrialOutcome


@dataclass(frozen=True)
class BudgetConfig:
    """Termination thresholds for one tuning campaign.

    Defaults match the AC-8 commit (``max_trials=20``,
    ``budget_seconds=43200`` (12h), ``max_oom=5``, ``patience=3``,
    ``epsilon=0.02``).
    """

    max_trials: int = 20
    budget_seconds: float = 43_200.0
    max_oom: int = 5
    patience: int = 3
    epsilon: float = 0.02
    preflight_retries: int = 3
    history_window: int = 8  # K from AC-7's "last K=8 trials"


@dataclass(frozen=True)
class CampaignResult:
    """Outcome of :meth:`Scheduler.run`.

    Attributes:
        stop_reason: Which termination predicate fired.
        trial_count: Number of trials that consumed budget (preflight-
            rejected proposals do NOT count).
        oom_count: Cumulative count of OOM failures across the campaign.
        best_entry: The lowest-objective ``(OK, NONE)`` ledger entry, or
            ``None`` when no eligible trial was produced.
        ledger_path: Convenience copy of the ledger file path.
    """

    stop_reason: str
    trial_count: int
    oom_count: int
    best_entry: LedgerEntry | None
    ledger_path: Path


# Type aliases for the injection points.
PreflightFn = Callable[[Mapping[str, Any]], "PreflightOutcome"]
RunnerFn = Callable[[Mapping[str, Any], "PreflightOutcome", int], TrialOutcome]
ParserFn = Callable[[TrialOutcome], TrialResult]


@dataclass(frozen=True)
class PreflightOutcome:
    """Result of the scheduler's pre-trial validation step.

    Attributes:
        ok: ``True`` when the delta passed every preflight check.
        errors: Reasons reported by preflight (empty when ``ok``).
        resolved_config_sha: SHA-256 of the resolved YAML (or ``None``
            when composition failed).
        log_dir: Per-trial log directory the runner should use.
        delta: The delta the critic proposed (for ledger persistence).
    """

    ok: bool
    errors: tuple[str, ...]
    resolved_config_sha: str | None
    log_dir: Path
    delta: Mapping[str, Any]


@dataclass
class Scheduler:
    """Orchestrates the tuning trial loop.

    Attributes:
        critic: Implementation of :class:`Critic` (real Codex critic or
            :class:`FakeCritic` for tests).
        runner_fn: Callable that turns ``(delta, preflight_outcome,
            trial_idx)`` into a :class:`TrialOutcome`. Production wiring
            calls ``OverrideWrapper.build_invocation`` + ``TrialRunner.launch``;
            tests inject a stub.
        parser_fn: Callable that turns a :class:`TrialOutcome` into a
            :class:`TrialResult`. Production calls ``parse_trial``.
        preflight_fn: Callable that runs ``compose_and_validate`` and
            returns a :class:`PreflightOutcome`.
        ledger: The :class:`Ledger` to append every trial to.
        budget: :class:`BudgetConfig` thresholds.
        baseline_knobs: Starting knob values; used for the prompt's
            ``current_knobs`` section before any delta is applied.
        clock: Injectable monotonic clock (defaults to :func:`time.monotonic`).
            Tests use a counter to make plateau / timeout assertions
            deterministic.
    """

    critic: Critic
    runner_fn: RunnerFn
    parser_fn: ParserFn
    preflight_fn: PreflightFn
    ledger: Ledger
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    baseline_knobs: dict[str, Any] = field(default_factory=dict)
    clock: Callable[[], float] = _time.monotonic

    # ----- Public API -----------------------------------------------------

    def run(self) -> CampaignResult:
        """Run the scheduler until a termination predicate fires."""
        start = self.clock()
        current_knobs = dict(self.baseline_knobs)
        history: list[TrialHistoryEntry] = []
        trial_idx = 0
        oom_count = 0
        consecutive_stop_requests = 0
        last_failure_mode: str | None = None
        last_metric_summary: Mapping[str, float] | None = None
        last_timeline_summary: Mapping[str, Any] | None = None

        if self.budget.max_trials <= 0:
            return self._finish("no_trials_run", trial_idx, oom_count)

        while True:
            if trial_idx >= self.budget.max_trials:
                return self._finish("max_trials_reached", trial_idx, oom_count)
            if self.clock() - start >= self.budget.budget_seconds:
                return self._finish("budget_seconds_elapsed", trial_idx, oom_count)
            if oom_count > self.budget.max_oom:
                return self._finish("oom_cap_exceeded", trial_idx, oom_count)

            # 1. Critic proposes (with preflight-retry loop).
            try:
                critic_output, preflight_outcome = self._propose_with_preflight(
                    history=history,
                    current_knobs=current_knobs,
                    last_failure_mode=last_failure_mode,
                    last_metric_summary=last_metric_summary,
                    last_timeline_summary=last_timeline_summary,
                )
            except CriticError as exc:
                self._record_critic_failure(trial_idx, str(exc), start)
                return self._finish("critic_failure", trial_idx, oom_count)

            if critic_output.stop_requested:
                consecutive_stop_requests += 1
                if consecutive_stop_requests >= 2:
                    return self._finish("critic_stagnation", trial_idx, oom_count)
                # Treat the proposed delta as a no-op signal; advance
                # without burning a trial slot OR running the runner.
                continue
            consecutive_stop_requests = 0

            # Bail out cleanly if preflight rejected the delta on every
            # retry. Per Codex review of Round 2, running the runner with
            # a known-invalid config burns a trial slot for no benefit;
            # instead we write a synthetic (FAILED, CONFIG_INVALID)
            # ledger entry and stop the campaign with a dedicated reason.
            if not preflight_outcome.ok:
                self._record_preflight_exhausted(
                    trial_idx=trial_idx,
                    delta=critic_output.delta,
                    preflight=preflight_outcome,
                    rationale=critic_output.rationale,
                    start=start,
                )
                return self._finish("preflight_exhausted", trial_idx, oom_count)

            # 2. Launch the trial via the runner (subprocess + cleanup).
            ts_start = _time.time()
            outcome = self.runner_fn(
                critic_output.delta, preflight_outcome, trial_idx
            )
            ts_end = _time.time()

            # 3. Parse the trial.
            result = self.parser_fn(outcome)

            # 4. Update bookkeeping.
            if result.failure_mode is FailureMode.OOM:
                oom_count += 1
            current_knobs = self._apply_delta(current_knobs, critic_output.delta)
            entry = self._build_ledger_entry(
                trial_idx=trial_idx,
                delta=critic_output.delta,
                preflight=preflight_outcome,
                outcome=outcome,
                result=result,
                rationale=critic_output.rationale,
                ts_start=ts_start,
                ts_end=ts_end,
            )
            self.ledger.append(entry)
            history.append(self._history_entry(entry, critic_output.rationale))
            history = history[-self.budget.history_window :]
            trial_idx += 1
            last_failure_mode = entry.failure_mode
            last_metric_summary = entry.per_component_timings
            last_timeline_summary = entry.timeline_summary

            # 5. Stopping-rule checks that depend on freshly written history.
            if self._is_plateaued(history):
                return self._finish("plateau", trial_idx, oom_count)

    # ----- Internals -----------------------------------------------------

    def _propose_with_preflight(
        self,
        *,
        history: list[TrialHistoryEntry],
        current_knobs: Mapping[str, Any],
        last_failure_mode: str | None,
        last_metric_summary: Mapping[str, float] | None,
        last_timeline_summary: Mapping[str, Any] | None,
    ) -> tuple[CriticOutput, PreflightOutcome]:
        """Ask the critic, run preflight, retry on preflight failures.

        Per AC-8: preflight failures DO NOT count toward ``max_trials``.
        The critic gets the rejection reason as ``preflight_feedback`` on
        the next prompt and re-proposes up to ``preflight_retries`` times.
        After that, the most recent ``(critic_output, preflight_outcome)``
        pair is returned with ``preflight_outcome.ok == False`` so the
        caller can record a synthetic ``CONFIG_INVALID`` ledger entry and
        terminate with the dedicated ``preflight_exhausted`` stop reason.
        """
        attempts = 0
        preflight_feedback: str | None = None
        last_critic_output: CriticOutput | None = None
        last_preflight: PreflightOutcome | None = None
        while attempts <= self.budget.preflight_retries:
            critic_output = self.critic.propose(
                history=history,
                current_knobs=current_knobs,
                last_failure_mode=last_failure_mode,
                last_metric_summary=last_metric_summary,
                last_timeline_summary=last_timeline_summary,
                preflight_feedback=preflight_feedback,
            )
            if critic_output.stop_requested:
                return critic_output, PreflightOutcome(
                    ok=True,
                    errors=(),
                    resolved_config_sha=None,
                    log_dir=Path(""),
                    delta=critic_output.delta,
                )
            preflight = self.preflight_fn(critic_output.delta)
            if preflight.ok:
                return critic_output, preflight
            attempts += 1
            last_critic_output = critic_output
            last_preflight = preflight
            preflight_feedback = (
                "Preflight rejected your previous delta with these errors:\n"
                + "\n".join(f"  - {e}" for e in preflight.errors)
            )
        # Exhausted retries: surface the last failure so the scheduler can
        # write a synthetic CONFIG_INVALID ledger entry and stop with
        # ``preflight_exhausted`` (rather than burn a trial slot launching
        # a known-invalid config — see Round-2 Codex review).
        if last_critic_output is None or last_preflight is None:
            raise CriticError("preflight retry loop produced no output")
        return last_critic_output, last_preflight

    def _is_plateaued(self, history: list[TrialHistoryEntry]) -> bool:
        """Return ``True`` when the last ``patience`` non-failed trials plateaued."""
        eligible = [
            entry
            for entry in history
            if entry.status == Status.OK.value and entry.objective is not None
        ]
        if len(eligible) < self.budget.patience + 1:
            return False
        recent = eligible[-(self.budget.patience + 1) :]
        for prev, curr in zip(recent, recent[1:]):
            if prev.objective <= 0:
                return False
            rel = abs(prev.objective - curr.objective) / prev.objective
            if rel >= self.budget.epsilon:
                return False
        return True

    def _record_critic_failure(self, trial_idx: int, reason: str, start: float) -> None:
        """Persist a ledger entry capturing critic exhaustion."""
        entry = make_entry(
            trial_idx=trial_idx,
            delta={},
            resolved_config_sha=None,
            log_dir="",
            returncode=None,
            status=Status.FAILED.value,
            failure_mode=FailureMode.LAUNCH_FAILURE.value,
            objective=None,
            step_time=None,
            num_trajectories=None,
            critic_rationale={"summary": reason, "metric_table_citations": [], "timeline_citations": []},
            ts_start=start,
            ts_end=_time.time(),
            cleanup_outcome="critic_failure",
        )
        self.ledger.append(entry)

    def _record_preflight_exhausted(
        self,
        *,
        trial_idx: int,
        delta: Mapping[str, Any],
        preflight: PreflightOutcome,
        rationale: Rationale,
        start: float,
    ) -> None:
        """Persist a synthetic ``CONFIG_INVALID`` ledger entry.

        Used when the critic could not produce a preflight-passing delta
        within ``preflight_retries``. The trial slot is NOT consumed
        (``trial_idx`` is the slot that would have been used, but the
        scheduler returns immediately afterwards with ``preflight_exhausted``).
        """
        entry = make_entry(
            trial_idx=trial_idx,
            delta=delta,
            resolved_config_sha=preflight.resolved_config_sha,
            log_dir=str(preflight.log_dir) if preflight.log_dir != Path("") else "",
            returncode=None,
            status=Status.FAILED.value,
            failure_mode=FailureMode.CONFIG_INVALID.value,
            objective=None,
            step_time=None,
            num_trajectories=None,
            critic_rationale=rationale.to_dict(),
            ts_start=start,
            ts_end=_time.time(),
            cleanup_outcome="preflight_exhausted: " + "; ".join(preflight.errors)[:200],
        )
        self.ledger.append(entry)

    def _build_ledger_entry(
        self,
        *,
        trial_idx: int,
        delta: Mapping[str, Any],
        preflight: PreflightOutcome,
        outcome: TrialOutcome,
        result: TrialResult,
        rationale: Rationale,
        ts_start: float,
        ts_end: float,
    ) -> LedgerEntry:
        timeline_dict: dict[str, Any] | None = None
        if result.timeline_summary is not None:
            timeline_dict = {
                "window_start": result.timeline_summary.window_start,
                "window_end": result.timeline_summary.window_end,
                "stall_fraction_by_component": dict(
                    result.timeline_summary.stall_fraction_by_component
                ),
                "per_tag": [
                    {
                        "component": stat.component,
                        "rank": stat.rank,
                        "tag": stat.tag,
                        "call_count": stat.call_count,
                        "duration_min": stat.duration_min,
                        "duration_median": stat.duration_median,
                        "duration_max": stat.duration_max,
                        "duration_total": stat.duration_total,
                    }
                    for stat in result.timeline_summary.per_tag
                ],
            }
        per_component = (
            dict(result.per_step[-1].time_keys)
            if result.per_step
            else {}
        )
        return make_entry(
            trial_idx=trial_idx,
            delta=delta,
            resolved_config_sha=preflight.resolved_config_sha,
            log_dir=str(outcome.log_dir),
            returncode=outcome.returncode,
            status=result.status.value,
            failure_mode=result.failure_mode.value,
            objective=result.objective,
            step_time=result.step_time_seconds,
            num_trajectories=result.num_trajectories,
            per_component_timings=per_component,
            timeline_summary=timeline_dict,
            peak_gpu_mem=result.peak_gpu_mem_gib,
            critic_rationale=rationale.to_dict(),
            ts_start=ts_start,
            ts_end=ts_end,
            cleanup_outcome=outcome.cleanup_outcome,
        )

    def _history_entry(
        self, ledger_entry: LedgerEntry, rationale: Rationale
    ) -> TrialHistoryEntry:
        return TrialHistoryEntry(
            trial_idx=ledger_entry.trial_idx,
            delta=ledger_entry.delta,
            status=ledger_entry.status,
            failure_mode=ledger_entry.failure_mode,
            objective=ledger_entry.objective,
            step_time=ledger_entry.step_time,
            rationale_summary=rationale.summary,
        )

    @staticmethod
    def _apply_delta(
        knobs: Mapping[str, Any], delta: Mapping[str, Any]
    ) -> dict[str, Any]:
        merged = dict(knobs)
        merged.update(delta)
        return merged

    def _finish(self, reason: str, trial_count: int, oom_count: int) -> CampaignResult:
        return CampaignResult(
            stop_reason=reason,
            trial_count=trial_count,
            oom_count=oom_count,
            best_entry=self.ledger.best(),
            ledger_path=self.ledger.path,
        )
