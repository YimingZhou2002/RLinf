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

import logging
import time as _time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolkits.embodied_tuner.critic import (
    Critic,
    CriticError,
    CriticOutput,
    ProposedLesson,
    Rationale,
    TrialHistoryEntry,
)
from toolkits.embodied_tuner.config_dedup_index import (
    ConfigDedupIndex,
    DedupEntry,
)
from toolkits.embodied_tuner.ledger import Ledger, LedgerEntry, make_entry
from toolkits.embodied_tuner.lessons import (
    BitterLesson,
    LessonBook,
    canonical_delta_signature,
)
from toolkits.embodied_tuner.node_store import (
    DAGNode,
    NodeStore,
    NodeStoreIntegrityError,
    ROLLBACK_FAILURE_MODES,
    ROOT_STATUS,
    derive_node_id,
    render_dag_view,
)
from toolkits.embodied_tuner.parser import (
    FailureMode,
    Status,
    TrialResult,
)
from toolkits.embodied_tuner.runner import TrialOutcome


_log = logging.getLogger(__name__)


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
    # Sibling cap for the parent-rollback state machine (AC-5). After
    # this many consecutive launched-trial failures at the same active
    # parent, the active leaf climbs one level further up the DAG. A
    # climb above the baseline root terminates the campaign with the
    # ``rollback_exhausted`` stop reason.
    max_siblings: int = 3


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


@dataclass(frozen=True)
class _ResumedState:
    """Flat scheduler state rebuilt from persisted DAG + Ledger.

    Populated by :meth:`Scheduler._reconstruct_state_from_stores` and
    unpacked by :meth:`Scheduler.run` when the process is restarting
    against a non-empty ``NodeStore``. Every field maps 1:1 to a local
    in ``run`` that would otherwise be reset to a fresh-campaign
    default at process start (see AC-3).
    """

    active_leaf_id: str
    cumulative_delta: dict[str, Any]
    current_knobs: dict[str, Any]
    trial_idx: int
    oom_count: int
    sibling_failures_at_active_parent: int
    history: list[TrialHistoryEntry]
    last_failure_mode: str | None
    last_metric_summary: Mapping[str, float] | None
    last_timeline_summary: Mapping[str, Any] | None
    last_num_trajectories: int | None
    last_failed_trial_idx: int | None
    last_failed_delta: dict[str, Any] | None
    duplicate_counter: int


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
    lesson_book: LessonBook | None = None
    # Optional DAG-structured coexistence store. When set, every trial is
    # mirrored to this store alongside the flat ``ledger``. The scheduler
    # bootstraps a baseline root node on startup if the store is empty.
    # Left ``None`` for backward compat: legacy tests and pre-DAG
    # campaigns continue to work with the flat ``Ledger`` only.
    node_store: NodeStore | None = None
    # Optional persistent config-dedup sidecar. When set (typically
    # alongside ``node_store``), the scheduler consults it after every
    # preflight pass: duplicate-of-OK short-circuits the runner and
    # emits a synthetic DUPLICATE_OF DAGNode; duplicate-of-FAILED is
    # rejected via ``preflight_feedback`` sharing the preflight-retry
    # budget. Left ``None`` for backward compat.
    dedup_index: ConfigDedupIndex | None = None
    # AC-7: cap on the number of sibling / top-K OK / recent-FAILED
    # entries rendered in the DAG-view prompt block. The ancestor
    # chain is always unconditional (never truncated by this cap).
    # Set to 0 to render an empty leaderboard (still well-formed).
    max_dag_nodes: int = 30

    # ----- Public API -----------------------------------------------------

    def run(self) -> CampaignResult:
        """Run the scheduler until a termination predicate fires."""
        start = self.clock()
        current_knobs = dict(self.baseline_knobs)
        # Accumulator of every knob override the campaign has applied on
        # top of baseline so far. Each round's Hydra overrides are
        # ``{**cumulative_delta, **critic_output.delta}`` — without this,
        # a knob changed in round N would silently revert to baseline in
        # round N+1 unless the critic re-emitted it, which contradicts
        # the "current_knobs" view the critic prompt shows.
        cumulative_delta: dict[str, Any] = {}
        history: list[TrialHistoryEntry] = []
        trial_idx = 0
        oom_count = 0
        consecutive_stop_requests = 0
        last_failure_mode: str | None = None
        last_metric_summary: Mapping[str, float] | None = None
        last_timeline_summary: Mapping[str, Any] | None = None
        last_num_trajectories: int | None = None
        # Track the delta that produced last_failure_mode so a lesson
        # emitted in the NEXT round can be attributed to the right
        # trial and delta signature. Cleared once folded into a lesson.
        # Attribution uses the critic's incremental proposal (not the
        # cumulative override set), so the lesson signature identifies
        # the specific knob flip the critic chose that round.
        last_failed_trial_idx: int | None = None
        last_failed_delta: dict[str, Any] | None = None
        # AC-5 sibling counter: how many consecutive launched-trial
        # failures have hit the CURRENT active parent since the last
        # advance / climb. Reset when active_leaf changes (advance on
        # OK, or climb after reaching ``max_siblings``).
        sibling_failures_at_active_parent = 0

        lesson_book = self._resolve_lesson_book()
        lessons: list[BitterLesson] = list(lesson_book.load())

        # Bootstrap the DAG root (if a NodeStore is wired in) BEFORE any
        # stopping-rule checks so a resumed campaign always has a root
        # regardless of ``max_trials``. Returns ``None`` when the store
        # is not configured (backward-compat mode).
        active_leaf_id: str | None = self._bootstrap_root_if_needed(start)
        # Restart-safe reconstruction: when the persisted NodeStore
        # carries content beyond the root (a resumed campaign), rebuild
        # every mutable local that would otherwise reset to fresh-
        # campaign defaults on process start (see AC-3). Fresh runs
        # (empty store or root-only + empty ledger) return ``None`` and
        # keep the fresh-campaign defaults unchanged.
        if active_leaf_id is not None:
            resumed = self._reconstruct_state_from_stores(root_id=active_leaf_id)
            if resumed is not None:
                active_leaf_id = resumed.active_leaf_id
                cumulative_delta = resumed.cumulative_delta
                current_knobs = resumed.current_knobs
                trial_idx = resumed.trial_idx
                oom_count = resumed.oom_count
                sibling_failures_at_active_parent = (
                    resumed.sibling_failures_at_active_parent
                )
                history = resumed.history
                last_failure_mode = resumed.last_failure_mode
                last_metric_summary = resumed.last_metric_summary
                last_timeline_summary = resumed.last_timeline_summary
                last_num_trajectories = resumed.last_num_trajectories
                last_failed_trial_idx = resumed.last_failed_trial_idx
                last_failed_delta = resumed.last_failed_delta
        # Load / rebuild the persistent dedup sidecar (if wired). The
        # NodeStore is the authoritative source; the sidecar rebuild
        # step ensures a lost or corrupt sidecar file is not fatal.
        # Also seeds the counter used to synthesise unique node ids
        # for duplicate-of short-circuits.
        duplicate_counter = 0
        if active_leaf_id is not None and self.node_store is not None:
            # Seed the counter from the resumed state so newly-synthesised
            # DUPLICATE_OF ids never collide with previously-persisted ones.
            # Recompute here (rather than threading through the else branch
            # above) so a run without a resume also lands on 0.
            duplicate_counter = sum(
                1
                for n in self.node_store.all_nodes()
                if n.failure_mode == FailureMode.DUPLICATE_OF.value
            )
        if self.dedup_index is not None:
            self.dedup_index.load_or_rebuild(self.node_store)

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
                    cumulative_delta=cumulative_delta,
                    last_failure_mode=last_failure_mode,
                    last_metric_summary=last_metric_summary,
                    last_timeline_summary=last_timeline_summary,
                    last_num_trajectories=last_num_trajectories,
                    bitter_lessons=lessons,
                    active_leaf_id=active_leaf_id,
                )
            except CriticError as exc:
                self._record_critic_failure(trial_idx, str(exc), start)
                return self._finish("critic_failure", trial_idx, oom_count)

            # Persist the critic's per-attempt prompt+response under the
            # trial log dir so a human debugger can audit what Codex saw
            # and returned for this round. Best-effort; a failure here
            # must not abort the trial loop.
            self._persist_critic_transactions(preflight_outcome.log_dir)

            # Fold any proposed bitter_lesson into the persistent book
            # BEFORE we run the new trial: the lesson describes the
            # PREVIOUS failure, so waiting until after this trial would
            # delay its influence by one extra round.
            if (
                critic_output.bitter_lesson is not None
                and last_failed_trial_idx is not None
                and last_failed_delta is not None
                and last_failure_mode is not None
            ):
                inserted = self._record_bitter_lesson(
                    lesson_book=lesson_book,
                    lessons=lessons,
                    proposed=critic_output.bitter_lesson,
                    trial_idx=last_failed_trial_idx,
                    failure_mode=last_failure_mode,
                    delta=last_failed_delta,
                )
                if inserted:
                    # A single failure yields one lesson at most; clear
                    # the pointer so a re-emission on a later round
                    # does not double-count.
                    last_failed_trial_idx = None
                    last_failed_delta = None

            if critic_output.stop_requested:
                consecutive_stop_requests += 1
                if consecutive_stop_requests >= 2:
                    return self._finish("critic_stagnation", trial_idx, oom_count)
                # Treat the proposed delta as a no-op signal; advance
                # without burning a trial slot OR running the runner.
                continue
            consecutive_stop_requests = 0

            # Cumulative override set actually applied this round. The
            # runner (and the preflight it just passed) both saw this
            # merged view, not just the critic's incremental proposal.
            effective_delta = {**cumulative_delta, **critic_output.delta}

            # Bail out cleanly if preflight rejected the delta on every
            # retry. Per Codex review of Round 2, running the runner with
            # a known-invalid config burns a trial slot for no benefit;
            # instead we write a synthetic (FAILED, CONFIG_INVALID)
            # ledger entry and stop the campaign with a dedicated reason.
            if not preflight_outcome.ok:
                self._record_preflight_exhausted(
                    trial_idx=trial_idx,
                    delta=effective_delta,
                    proposed_delta=critic_output.delta,
                    preflight=preflight_outcome,
                    rationale=critic_output.rationale,
                    start=start,
                )
                return self._finish("preflight_exhausted", trial_idx, oom_count)

            # Duplicate-of-OK short-circuit. When the (validated)
            # resolved-config SHA matches a prior clean-OK trial, skip
            # the runner entirely: emit a synthetic DUPLICATE_OF DAG
            # node that back-references the ORIGINAL non-duplicate and
            # copies its objective. AC-6: does NOT consume a
            # max_trials slot; does NOT create a Ledger entry (the
            # flat Ledger stays free of synthetic rows so existing
            # consumers like `plot_step_time_vs_trajectories` are
            # untouched). The DAG's ``active_leaf`` advances to the
            # new duplicate node so subsequent Codex proposals treat
            # it like any other expansion point.
            if (
                self.dedup_index is not None
                and preflight_outcome.resolved_config_sha is not None
            ):
                origin = self.dedup_index.lookup(
                    preflight_outcome.resolved_config_sha
                )
                if origin is not None and origin.is_ok():
                    duplicate_counter += 1
                    new_dup_id = self._record_duplicate_of_ok(
                        origin=origin,
                        parent_id=active_leaf_id,
                        delta_from_parent=critic_output.delta,
                        cumulative_delta=effective_delta,
                        resolved_config_sha=preflight_outcome.resolved_config_sha,
                        rationale=critic_output.rationale,
                        duplicate_seq=duplicate_counter,
                    )
                    if new_dup_id is not None:
                        # AC-6: dup-of-OK is an OK-equivalent advance —
                        # sync the FLAT scheduler state (active leaf +
                        # cumulative delta + current knobs) to the
                        # synthetic node so the next preflight builds
                        # from the same cumulative view the DAG says
                        # is now active. Without this the DAG advances
                        # to the duplicate while the next round's
                        # preflight silently runs against the previous
                        # flat state (split-brain).
                        active_leaf_id = new_dup_id
                        cumulative_delta = dict(effective_delta)
                        current_knobs = self._apply_delta(
                            dict(self.baseline_knobs), cumulative_delta
                        )
                        # Successful advance clears the parent's
                        # sibling-failure counter, mirroring the OK-trial
                        # branch below.
                        sibling_failures_at_active_parent = 0
                    # Do not advance trial_idx; do not touch oom_count.
                    # Loop back and let the critic propose again from
                    # the updated (post-dedup-visible) active leaf.
                    continue

            # 2. Launch the trial via the runner (subprocess + cleanup).
            ts_start = _time.time()
            outcome = self.runner_fn(
                effective_delta, preflight_outcome, trial_idx
            )
            ts_end = _time.time()

            # 3. Parse the trial.
            result = self.parser_fn(outcome)

            # 4. Update bookkeeping.
            if result.failure_mode is FailureMode.OOM:
                oom_count += 1
            entry = self._build_ledger_entry(
                trial_idx=trial_idx,
                delta=effective_delta,
                proposed_delta=critic_output.delta,
                preflight=preflight_outcome,
                outcome=outcome,
                result=result,
                rationale=critic_output.rationale,
                ts_start=ts_start,
                ts_end=ts_end,
            )
            self.ledger.append(entry)
            # Mirror to the DAG store (if wired). The parent id here is
            # the CURRENT ``active_leaf_id``, which is either the
            # baseline root, a prior OK/DUPLICATE_OF node, or an
            # ancestor reached by an earlier rollback. AC-5's state
            # machine (below) rewinds ``active_leaf_id`` after the
            # mirror step when this trial is a rollback failure.
            new_node_id = self._mirror_to_node_store(
                entry=entry,
                parent_id=active_leaf_id,
                ts_start=ts_start,
            )
            if new_node_id is not None:
                # Register this real launched trial in the dedup index so
                # a subsequent proposal that resolves to the same SHA
                # can short-circuit (OK) or be rejected (FAILED).
                if self.dedup_index is not None and self.node_store is not None:
                    origin_node = self.node_store.get(new_node_id)
                    if origin_node is not None:
                        self.dedup_index.add(node=origin_node)

            # AC-5 state machine: any launched-trial failure in
            # ROLLBACK_FAILURE_MODES rewinds ``active_leaf`` to the
            # failing node's parent (and, if the sibling cap has been
            # reached, one more level up). Only OK trials advance the
            # active leaf; only OK trials update the flat
            # ``cumulative_delta`` / ``current_knobs`` accumulators (a
            # rollback failure must not silently leave its knobs
            # applied for the next round). Preflight rejections
            # (CONFIG_INVALID / DIVISIBILITY_VIOLATION) are handled
            # above via ``preflight_exhausted`` and never reach this
            # block — they do not create a DAG node.
            is_rollback_failure = entry.failure_mode in ROLLBACK_FAILURE_MODES
            if self.node_store is not None and new_node_id is not None and is_rollback_failure:
                # Rewind: undo the tentative advance and reset the
                # flat accumulators to the parent's cumulative state.
                parent = self.node_store.parent_of(new_node_id)
                if parent is not None:
                    active_leaf_id = parent.node_id
                    cumulative_delta = dict(parent.cumulative_delta)
                    current_knobs = self._apply_delta(
                        dict(self.baseline_knobs), cumulative_delta
                    )
                else:
                    # Should be unreachable — every launched trial's
                    # parent is at least the root — but degrade safely.
                    _log.warning(
                        "rollback: failing trial %s has no parent in the"
                        " NodeStore; skipping state machine update",
                        new_node_id,
                    )
                sibling_failures_at_active_parent += 1
                if sibling_failures_at_active_parent >= self.budget.max_siblings:
                    grandparent = (
                        self.node_store.parent_of(active_leaf_id)
                        if active_leaf_id is not None
                        else None
                    )
                    if grandparent is None:
                        # We were rewound to root (or above); no
                        # further climb possible. Terminate the
                        # campaign with the dedicated stop reason.
                        # AC-8: this trial was launched (runner ran,
                        # ledger entry appended, NodeStore mirrored),
                        # so it counts toward the campaign's
                        # trial_count. The post-trial ``trial_idx += 1``
                        # bookkeeping below never runs on this stop
                        # path, so we pass ``trial_idx + 1`` explicitly
                        # to keep ``result.trial_count == len(runner_calls)``.
                        return self._finish(
                            "rollback_exhausted", trial_idx + 1, oom_count
                        )
                    active_leaf_id = grandparent.node_id
                    cumulative_delta = dict(grandparent.cumulative_delta)
                    current_knobs = self._apply_delta(
                        dict(self.baseline_knobs), cumulative_delta
                    )
                    sibling_failures_at_active_parent = 0
            else:
                # OK trial (or backward-compat mode without node_store):
                # advance the flat state and, when we have a DAG store,
                # advance active_leaf to the new node too. Reset the
                # sibling counter — this parent has produced a success.
                current_knobs = self._apply_delta(current_knobs, critic_output.delta)
                cumulative_delta = effective_delta
                if new_node_id is not None:
                    active_leaf_id = new_node_id
                sibling_failures_at_active_parent = 0
            history.append(
                self._history_entry(
                    entry, critic_output.rationale, critic_output.delta
                )
            )
            history = history[-self.budget.history_window :]
            trial_idx += 1
            last_failure_mode = entry.failure_mode
            last_metric_summary = entry.per_component_timings
            last_timeline_summary = entry.timeline_summary
            last_num_trajectories = entry.num_trajectories
            if entry.failure_mode != FailureMode.NONE.value:
                last_failed_trial_idx = entry.trial_idx
                last_failed_delta = dict(critic_output.delta)
            else:
                last_failed_trial_idx = None
                last_failed_delta = None

            # 5. Stopping-rule checks that depend on freshly written history.
            if self._is_plateaued(history):
                return self._finish("plateau", trial_idx, oom_count)

    # ----- Internals -----------------------------------------------------

    def _propose_with_preflight(
        self,
        *,
        history: list[TrialHistoryEntry],
        current_knobs: Mapping[str, Any],
        cumulative_delta: Mapping[str, Any],
        last_failure_mode: str | None,
        last_metric_summary: Mapping[str, float] | None,
        last_timeline_summary: Mapping[str, Any] | None,
        last_num_trajectories: int | None = None,
        bitter_lessons: Sequence[BitterLesson] = (),
        active_leaf_id: str | None = None,
    ) -> tuple[CriticOutput, PreflightOutcome]:
        """Ask the critic, run preflight, retry on preflight failures.

        Per AC-8: preflight failures DO NOT count toward ``max_trials``.
        The critic gets the rejection reason as ``preflight_feedback`` on
        the next prompt and re-proposes up to ``preflight_retries`` times.
        After that, the most recent ``(critic_output, preflight_outcome)``
        pair is returned with ``preflight_outcome.ok == False`` so the
        caller can record a synthetic ``CONFIG_INVALID`` ledger entry and
        terminate with the dedicated ``preflight_exhausted`` stop reason.

        ``cumulative_delta`` is the override set already accepted by
        earlier rounds; each retry's preflight is run against
        ``{**cumulative_delta, **critic.delta}`` so that a delta the
        critic proposes here is validated in the context that will
        actually reach the runner, not in isolation.
        """
        attempts = 0
        preflight_feedback: str | None = None
        last_critic_output: CriticOutput | None = None
        last_preflight: PreflightOutcome | None = None
        # AC-7: render the compact DAG view once per proposal round so
        # every retry in this preflight loop sees the same DAG state
        # (Codex should not observe the DAG shifting mid-retry). When
        # the NodeStore is not wired in the block is left empty and
        # ``CriticPrompt`` renders nothing for it.
        dag_block = ""
        if self.node_store is not None:
            dag_block = render_dag_view(
                self.node_store,
                active_leaf_id=active_leaf_id,
                max_dag_nodes=self.max_dag_nodes,
            )
        while attempts <= self.budget.preflight_retries:
            critic_output = self.critic.propose(
                history=history,
                current_knobs=current_knobs,
                last_failure_mode=last_failure_mode,
                last_metric_summary=last_metric_summary,
                last_timeline_summary=last_timeline_summary,
                last_num_trajectories=last_num_trajectories,
                bitter_lessons=bitter_lessons,
                preflight_feedback=preflight_feedback,
                dag_block=dag_block,
            )
            if critic_output.stop_requested:
                return critic_output, PreflightOutcome(
                    ok=True,
                    errors=(),
                    resolved_config_sha=None,
                    log_dir=Path(""),
                    delta=critic_output.delta,
                )
            effective_delta = {**cumulative_delta, **critic_output.delta}
            preflight = self.preflight_fn(effective_delta)
            if preflight.ok:
                # Dedup guard for previously-FAILED configs: if the
                # cumulative resolved-config SHA matches a prior failed
                # trial, reject this proposal via ``preflight_feedback``
                # sharing the preflight-retry budget. Duplicate-of-OK
                # is handled outside this helper (see ``run``): it
                # short-circuits the runner rather than looping the
                # critic.
                if (
                    self.dedup_index is not None
                    and preflight.resolved_config_sha is not None
                ):
                    prior = self.dedup_index.lookup(preflight.resolved_config_sha)
                    if prior is not None and not prior.is_ok():
                        attempts += 1
                        feedback_msg = (
                            "The resolved config for your proposed delta has "
                            "already been attempted in this campaign (node "
                            f"{prior.origin_node_id!r}) and failed with "
                            f"{prior.failure_mode!r}. Propose a different "
                            "delta that produces a different resolved config."
                        )
                        preflight_feedback = feedback_msg
                        last_critic_output = critic_output
                        # Track a synthesised rejection so the exit
                        # path still surfaces preflight_exhausted (the
                        # semantic is identical: no acceptable config
                        # was produced within the retry budget).
                        last_preflight = PreflightOutcome(
                            ok=False,
                            errors=(feedback_msg,),
                            resolved_config_sha=preflight.resolved_config_sha,
                            log_dir=preflight.log_dir,
                            delta=preflight.delta,
                        )
                        continue
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
        """Return ``True`` when the last ``patience`` non-failed trials plateaued.

        Excludes DUPLICATE_OF entries: a synthetic duplicate copies the
        original trial's objective, so treating it as a distinct
        improvement (or lack thereof) would artificially satisfy the
        plateau predicate.
        """
        eligible = [
            entry
            for entry in history
            if entry.status == Status.OK.value
            and entry.failure_mode == FailureMode.NONE.value
            and entry.objective is not None
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

    def _persist_critic_transactions(self, log_dir: Path) -> None:
        """Dump the critic's per-attempt prompt + response under ``log_dir``.

        Reads ``self.critic.transaction_log`` (present on ``CodexCritic``
        and any test critic that opts in). Silently skips when the
        critic does not expose it, or when the log dir is empty (the
        stop-requested branch supplies ``Path("")``). Any IO error is
        swallowed with a log line — persisting the transcript is a
        debugging convenience, not a correctness requirement.
        """
        log = getattr(self.critic, "transaction_log", None)
        if not log:
            return
        if log_dir == Path(""):
            return
        try:
            critic_dir = Path(log_dir) / "critic"
            critic_dir.mkdir(parents=True, exist_ok=True)
            for record in log:
                attempt = record.get("attempt", 0)
                prompt_path = critic_dir / f"attempt-{attempt:02d}-prompt.md"
                response_path = critic_dir / f"attempt-{attempt:02d}-response.txt"
                header_parts = [f"# Critic transaction — attempt {attempt}"]
                if record.get("parse_error"):
                    header_parts.append(f"parse_error: {record['parse_error']}")
                else:
                    header_parts.append(
                        f"validation_ok: {record.get('validation_ok')}"
                    )
                    if record.get("validation_reason"):
                        header_parts.append(
                            f"validation_reason: {record['validation_reason']}"
                        )
                header = "\n".join(header_parts) + "\n\n"
                prompt_path.write_text(
                    header + record.get("prompt_debug", ""), encoding="utf-8"
                )
                response_path.write_text(
                    record.get("response", ""), encoding="utf-8"
                )
        except OSError as exc:
            _log.warning("failed to persist critic transactions to %s: %s", log_dir, exc)

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
        proposed_delta: Mapping[str, Any],
        preflight: PreflightOutcome,
        rationale: Rationale,
        start: float,
    ) -> None:
        """Persist a synthetic ``CONFIG_INVALID`` ledger entry.

        Used when the critic could not produce a preflight-passing delta
        within ``preflight_retries``. The trial slot is NOT consumed
        (``trial_idx`` is the slot that would have been used, but the
        scheduler returns immediately afterwards with ``preflight_exhausted``).

        ``delta`` is the cumulative override set that would have been
        applied; ``proposed_delta`` is the last incremental proposal the
        critic emitted (stored separately for lesson attribution and
        debugging).
        """
        entry = make_entry(
            trial_idx=trial_idx,
            delta=delta,
            proposed_delta=proposed_delta,
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
            # Surface the exact preflight errors so on a resumed campaign
            # or downstream analysis the LLM sees why the delta was
            # rejected, not just "CONFIG_INVALID".
            error_excerpt="preflight errors:\n" + "\n".join(preflight.errors),
        )
        self.ledger.append(entry)

    def _build_ledger_entry(
        self,
        *,
        trial_idx: int,
        delta: Mapping[str, Any],
        proposed_delta: Mapping[str, Any],
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
                        "duration_median": stat.duration_median,
                        "duration_max": stat.duration_max,
                        "duration_total": stat.duration_total,
                    }
                    for stat in result.timeline_summary.per_tag
                ],
                "critical_path": dict(result.timeline_summary.critical_path),
                "outliers": list(result.timeline_summary.outliers),
                "per_component_bubble": dict(result.timeline_summary.per_component_bubble),
                "raw_excerpts": list(result.timeline_summary.raw_excerpts),
                "raw_jsonl": dict(result.timeline_summary.raw_jsonl),
                "plot_paths": {
                    fmt: str(path)
                    for fmt, path in result.timeline_summary.plot_paths.items()
                },
                "component_call_averages": dict(
                    result.timeline_summary.component_call_averages
                ),
            }
        per_component = (
            dict(result.per_step[-1].time_keys)
            if result.per_step
            else {}
        )
        return make_entry(
            trial_idx=trial_idx,
            delta=delta,
            proposed_delta=proposed_delta,
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
            error_excerpt=result.error_excerpt,
        )

    def _history_entry(
        self,
        ledger_entry: LedgerEntry,
        rationale: Rationale,
        proposed_delta: Mapping[str, Any],
    ) -> TrialHistoryEntry:
        # The history block in the critic prompt shows what the critic
        # *chose to change* each round, not the cumulative override set
        # (which is redundant with the "Current knob values" section).
        return TrialHistoryEntry(
            trial_idx=ledger_entry.trial_idx,
            delta=dict(proposed_delta),
            status=ledger_entry.status,
            failure_mode=ledger_entry.failure_mode,
            objective=ledger_entry.objective,
            step_time=ledger_entry.step_time,
            rationale_summary=rationale.summary,
            error_excerpt=ledger_entry.error_excerpt,
        )

    @staticmethod
    def _apply_delta(
        knobs: Mapping[str, Any], delta: Mapping[str, Any]
    ) -> dict[str, Any]:
        merged = dict(knobs)
        merged.update(delta)
        return merged

    def _resolve_lesson_book(self) -> LessonBook:
        """Return the campaign :class:`LessonBook`, defaulting alongside the ledger.

        Persistence lives next to ``tuner_ledger.jsonl`` (i.e. the ledger
        directory the CLI writes into) so an operator can inspect
        ``<ledger_dir>/bitter_lessons.jsonl`` without hunting for it.
        """
        if self.lesson_book is not None:
            return self.lesson_book
        book = LessonBook(path=self.ledger.path.parent / "bitter_lessons.jsonl")
        self.lesson_book = book
        return book

    def _record_bitter_lesson(
        self,
        *,
        lesson_book: LessonBook,
        lessons: list[BitterLesson],
        proposed: ProposedLesson,
        trial_idx: int,
        failure_mode: str,
        delta: Mapping[str, Any],
    ) -> bool:
        """Attach scheduler-owned attribution to a critic-proposed lesson.

        The critic emits ``trigger`` and ``rule``; the scheduler stamps
        the actual failed trial's index, failure mode, and delta
        signature before handing the assembled :class:`BitterLesson` to
        :meth:`LessonBook.add`. Returns whether a new lesson was
        persisted (``False`` when the delta signature had already been
        recorded under the same failure mode).
        """
        lesson = BitterLesson(
            trigger=proposed.trigger.strip(),
            rule=proposed.rule.strip(),
            trial_idx=trial_idx,
            failure_mode=failure_mode,
            delta_signature=canonical_delta_signature(delta),
        )
        if not lesson.trigger or not lesson.rule:
            return False
        inserted = lesson_book.add(lesson)
        if inserted:
            lessons.append(lesson)
            # LessonBook may have evicted the oldest entry to enforce
            # the cap; re-sync the caller's in-memory copy from the
            # authoritative store so the next prompt reflects that.
            del lessons[:]
            lessons.extend(lesson_book.all())
        return inserted

    def _finish(self, reason: str, trial_count: int, oom_count: int) -> CampaignResult:
        return CampaignResult(
            stop_reason=reason,
            trial_count=trial_count,
            oom_count=oom_count,
            best_entry=self.ledger.best(),
            ledger_path=self.ledger.path,
        )

    # ----- DAG coexistence -----------------------------------------------

    def _bootstrap_root_if_needed(self, start: float) -> str | None:
        """Ensure the DAG root node exists; return the active leaf id or ``None``.

        Called once at the top of :meth:`run`. Behaviour depends on
        whether a :class:`NodeStore` has been wired in:

        - ``self.node_store is None``: DAG coexistence is disabled.
          Return ``None`` and skip every subsequent mirror step.
        - Store already has a root (resumed campaign): return the root
          id. The rollback-aware active-leaf derivation is deferred to
          :meth:`_reconstruct_state_from_stores`, which the caller
          invokes when the persisted DAG has non-root content.
        - Store is empty (fresh campaign): run preflight with an empty
          delta to compute the baseline ``resolved_config_sha``, append
          a root :class:`DAGNode`, and return its id.
        """
        if self.node_store is None:
            return None
        existing_root = self.node_store.root()
        if existing_root is not None:
            # Return the root id unconditionally. If the store carries
            # non-root nodes (a resumed campaign), the caller replaces
            # this with the rollback-aware active leaf via
            # :meth:`_reconstruct_state_from_stores`.
            return existing_root.node_id
        # Fresh campaign: compute baseline SHA via preflight so root's
        # resolved_config_sha matches the same hash the runner would
        # observe. Preflight is by contract Ray/GPU-free, so this is
        # cheap and safe at scheduler startup.
        baseline_preflight = self.preflight_fn({})
        root_ts = _time.time()
        root_id = derive_node_id(
            parent_id=None,
            delta_from_parent={},
            trial_idx=None,
            ts_start=root_ts,
        )
        root_node = DAGNode(
            node_id=root_id,
            parent_id=None,
            delta_from_parent={},
            cumulative_delta={},
            trial_idx=None,
            resolved_config_sha=baseline_preflight.resolved_config_sha,
            log_dir="",
            returncode=None,
            status=ROOT_STATUS,
            failure_mode=FailureMode.NONE.value,
            objective=None,
            step_time=None,
            num_trajectories=None,
            per_component_timings={},
            timeline_summary=None,
            peak_gpu_mem=None,
            critic_rationale=None,
            ts_start=root_ts,
            ts_end=root_ts,
            cleanup_outcome="ok",
        )
        self.node_store.append(root_node)
        # Seed the dedup index with the baseline root's SHA so a Codex
        # proposal that resolves back to the pristine baseline short-
        # circuits to a duplicate rather than re-running the baseline.
        if self.dedup_index is not None:
            self.dedup_index.load_or_rebuild(self.node_store)
        return root_id

    def _reconstruct_state_from_stores(
        self, *, root_id: str
    ) -> "_ResumedState | None":
        """Rebuild the flat scheduler state from persisted DAG + Ledger.

        Called from :meth:`run` after :meth:`_bootstrap_root_if_needed`
        when the persisted ``NodeStore`` carries content beyond the
        root. Restores every mutable variable that ``run`` would
        otherwise reset to fresh-campaign defaults on process start:
        ``active_leaf_id``, ``cumulative_delta``, ``current_knobs``,
        ``trial_idx``, ``oom_count``, the sibling-failure counter, the
        rolling ``history`` window, and the ``last_*`` failure /
        metric / delta attribution slots.

        The active leaf is derived by replaying the AC-5 rollback rules
        via :meth:`NodeStore.active_state`. When the replay walks above
        the root (``rollback_exhausted`` waiting to fire), the resumed
        state anchors at the root; the next launched failure will
        surface the stop reason on the normal path.

        Returns ``None`` when there is nothing to reconstruct (store
        has only the root and the ledger is empty), letting the caller
        keep fresh-campaign defaults untouched.
        """
        if self.node_store is None:
            return None
        all_nodes = self.node_store.all_nodes()
        ledger_result = self.ledger.load()
        # Ledger and NodeStore together define "has this campaign done
        # any work?". If both are empty apart from the root, resume is
        # a no-op and the caller keeps fresh-campaign defaults.
        if len(all_nodes) <= 1 and not ledger_result.entries:
            return None

        active_id, sibling_failures = self.node_store.active_state(
            self.budget.max_siblings
        )
        # Climb walked above the root during replay. The next launched
        # failure will fire ``rollback_exhausted`` on the normal path,
        # so anchor state at the root and let the loop drive it.
        if active_id is None:
            active_id = root_id
            sibling_failures = 0
        active_node = self.node_store.get(active_id)
        if active_node is None:
            # Defensive: active-state replay pointed at something the
            # store doesn't have (corrupted line skipped between the
            # replay and this lookup). Fall back to the root and log.
            _log.warning(
                "resume: active id %r not found in NodeStore; anchoring"
                " at root %r",
                active_id,
                root_id,
            )
            active_id = root_id
            sibling_failures = 0
            active_node = self.node_store.get(root_id)
            if active_node is None:
                return None

        cumulative_delta = dict(active_node.cumulative_delta)
        current_knobs = self._apply_delta(
            dict(self.baseline_knobs), cumulative_delta
        )

        # Only launched trials (non-negative ``trial_idx``) count toward
        # ``trial_idx`` / ``oom_count``. Root has ``trial_idx=None`` and
        # DUPLICATE_OF nodes use negative sentinels.
        launched_nodes = [
            n
            for n in all_nodes
            if n.trial_idx is not None and n.trial_idx >= 0
        ]
        if launched_nodes:
            trial_idx = max(n.trial_idx for n in launched_nodes) + 1
        else:
            trial_idx = 0
        oom_count = sum(
            1
            for n in launched_nodes
            if n.failure_mode == FailureMode.OOM.value
        )

        # Reconstruct the rolling history window from the flat Ledger.
        # Ledger contains every launched trial and only launched trials
        # (DUPLICATE_OF nodes and the root are NodeStore-only), which
        # matches how ``history`` is populated in the live loop.
        history: list[TrialHistoryEntry] = []
        for entry in ledger_result.entries[-self.budget.history_window :]:
            rationale_dict = entry.critic_rationale or {}
            rationale = Rationale(
                summary=str(rationale_dict.get("summary", "")),
                metric_table_citations=tuple(
                    rationale_dict.get("metric_table_citations", ()) or ()
                ),
                timeline_citations=tuple(
                    rationale_dict.get("timeline_citations", ()) or ()
                ),
            )
            history.append(
                self._history_entry(
                    entry, rationale, entry.proposed_delta or {}
                )
            )

        # Attribution fields: mirror how the live loop sets them at the
        # end of the trial iteration, using the most recent Ledger entry
        # (which is always a launched trial).
        last_failure_mode: str | None = None
        last_metric_summary: Mapping[str, float] | None = None
        last_timeline_summary: Mapping[str, Any] | None = None
        last_num_trajectories: int | None = None
        last_failed_trial_idx: int | None = None
        last_failed_delta: dict[str, Any] | None = None
        if ledger_result.entries:
            latest = ledger_result.entries[-1]
            last_failure_mode = latest.failure_mode
            last_metric_summary = latest.per_component_timings
            last_timeline_summary = latest.timeline_summary
            last_num_trajectories = latest.num_trajectories
            if latest.failure_mode != FailureMode.NONE.value:
                last_failed_trial_idx = latest.trial_idx
                last_failed_delta = dict(latest.proposed_delta or {})

        # Seed the ``duplicate_counter`` so newly-synthesised DUPLICATE_OF
        # node ids do not collide with previously-persisted duplicates.
        duplicate_counter = sum(
            1
            for n in all_nodes
            if n.failure_mode == FailureMode.DUPLICATE_OF.value
        )

        return _ResumedState(
            active_leaf_id=active_id,
            cumulative_delta=cumulative_delta,
            current_knobs=current_knobs,
            trial_idx=trial_idx,
            oom_count=oom_count,
            sibling_failures_at_active_parent=sibling_failures,
            history=history,
            last_failure_mode=last_failure_mode,
            last_metric_summary=last_metric_summary,
            last_timeline_summary=last_timeline_summary,
            last_num_trajectories=last_num_trajectories,
            last_failed_trial_idx=last_failed_trial_idx,
            last_failed_delta=last_failed_delta,
            duplicate_counter=duplicate_counter,
        )

    def _mirror_to_node_store(
        self,
        *,
        entry: LedgerEntry,
        parent_id: str | None,
        ts_start: float,
    ) -> str | None:
        """Project ``entry`` into a :class:`DAGNode` and append it.

        Called after every ``ledger.append`` in :meth:`run`. When
        ``self.node_store`` is ``None`` (backward-compat mode) this is a
        no-op returning ``None``. When the node store is wired but a
        parent id is missing (should never happen once the root is
        bootstrapped), the mirror step is skipped defensively rather
        than raising — the flat Ledger remains the authoritative
        campaign record for existing consumers.
        """
        if self.node_store is None:
            return None
        if parent_id is None:
            _log.warning(
                "DAG coexistence: no parent_id available for trial_idx=%s;"
                " skipping node_store mirror",
                entry.trial_idx,
            )
            return None
        node_id = derive_node_id(
            parent_id=parent_id,
            delta_from_parent=entry.proposed_delta or {},
            trial_idx=entry.trial_idx,
            ts_start=ts_start,
        )
        node = DAGNode(
            node_id=node_id,
            parent_id=parent_id,
            delta_from_parent=dict(entry.proposed_delta or {}),
            cumulative_delta=dict(entry.delta),
            trial_idx=entry.trial_idx,
            resolved_config_sha=entry.resolved_config_sha,
            log_dir=entry.log_dir,
            returncode=entry.returncode,
            status=entry.status,
            failure_mode=entry.failure_mode,
            objective=entry.objective,
            step_time=entry.step_time,
            num_trajectories=entry.num_trajectories,
            per_component_timings=dict(entry.per_component_timings),
            timeline_summary=(
                dict(entry.timeline_summary)
                if entry.timeline_summary is not None
                else None
            ),
            peak_gpu_mem=entry.peak_gpu_mem,
            critic_rationale=(
                dict(entry.critic_rationale)
                if entry.critic_rationale is not None
                else None
            ),
            ts_start=entry.ts_start,
            ts_end=entry.ts_end,
            cleanup_outcome=entry.cleanup_outcome,
            error_excerpt=entry.error_excerpt,
        )
        try:
            self.node_store.append(node)
        except NodeStoreIntegrityError as exc:
            _log.warning(
                "DAG coexistence: node_store rejected trial_idx=%s (%s);"
                " Ledger remains authoritative for this trial",
                entry.trial_idx,
                exc,
            )
            return None
        return node_id

    def _record_duplicate_of_ok(
        self,
        *,
        origin: DedupEntry,
        parent_id: str | None,
        delta_from_parent: Mapping[str, Any],
        cumulative_delta: Mapping[str, Any],
        resolved_config_sha: str,
        rationale: Rationale,
        duplicate_seq: int,
    ) -> str | None:
        """Append a synthetic ``DUPLICATE_OF`` DAGNode short-circuiting the runner.

        Emitted when the resolved-config SHA of a preflight-passing
        proposal has already been observed with a clean-OK outcome
        (see AC-6). The synthetic node:

        - Uses ``status = "OK"`` and ``failure_mode = "DUPLICATE_OF"``.
        - Sets ``duplicate_of_node_id`` to the ORIGINAL non-duplicate
          node's id (never to another duplicate — the ``ConfigDedupIndex``
          already collapses chains).
        - Copies the original trial's ``objective`` for continuity in
          downstream leaderboards.
        - Uses a negative ``trial_idx`` (``-duplicate_seq``) so it
          renders as ``dup<seq>-<hash>`` in the DAG viewer, is easy to
          distinguish from real launched trials, and never collides
          with a positive ``trial_idx``.

        No :class:`Ledger` entry is written — the flat Ledger stays
        clean of synthetic rows so ``_emit_best_artefacts`` and
        ``plot_step_time_vs_trajectories`` continue to consume only
        real launched trials. The DAG is the audit trail for
        duplicates.
        """
        if self.node_store is None:
            return None
        if parent_id is None:
            _log.warning(
                "duplicate-of-OK short-circuit: no parent_id available;"
                " skipping DUPLICATE_OF DAG node emission for SHA %s",
                resolved_config_sha,
            )
            return None
        ts = _time.time()
        synthetic_trial_idx = -duplicate_seq
        node_id = derive_node_id(
            parent_id=parent_id,
            delta_from_parent=delta_from_parent,
            trial_idx=synthetic_trial_idx,
            ts_start=ts,
        )
        dup_node = DAGNode(
            node_id=node_id,
            parent_id=parent_id,
            delta_from_parent=dict(delta_from_parent),
            cumulative_delta=dict(cumulative_delta),
            trial_idx=synthetic_trial_idx,
            resolved_config_sha=resolved_config_sha,
            log_dir="",
            returncode=None,
            status=Status.OK.value,
            failure_mode=FailureMode.DUPLICATE_OF.value,
            objective=origin.objective,
            step_time=None,
            num_trajectories=None,
            per_component_timings={},
            timeline_summary=None,
            peak_gpu_mem=None,
            critic_rationale=rationale.to_dict(),
            ts_start=ts,
            ts_end=ts,
            cleanup_outcome="ok",
            duplicate_of_node_id=origin.origin_node_id,
        )
        try:
            self.node_store.append(dup_node)
        except NodeStoreIntegrityError as exc:
            _log.warning(
                "duplicate-of-OK short-circuit: node_store rejected"
                " synthetic node for SHA %s (%s); ledger and dedup"
                " state remain consistent",
                resolved_config_sha,
                exc,
            )
            return None
        return node_id
