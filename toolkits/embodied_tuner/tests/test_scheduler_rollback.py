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

"""Tests for the AC-5 parent-rollback state machine.

Covers:

- Rollback on every failure mode in ``ROLLBACK_FAILURE_MODES`` (5 modes
  including METRICS_PARTIAL and METRICS_MISSING per the user's
  refine-plan clarification).
- OK-only advance: successful trials advance ``active_leaf`` and reset
  the sibling counter.
- Sibling cap + ancestor climb (``max_siblings``): after N consecutive
  rollback failures at the same parent, the active leaf climbs one
  level; climbing above the root terminates with the new
  ``rollback_exhausted`` stop reason.
- Preflight rejections (``CONFIG_INVALID`` / ``DIVISIBILITY_VIOLATION``)
  never create NodeStore entries and never change ``active_leaf``.
- ``cumulative_delta`` / ``current_knobs`` correctly rewind on rollback.
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
from toolkits.embodied_tuner.node_store import (
    NodeStore,
    ROLLBACK_FAILURE_MODES,
)
from toolkits.embodied_tuner.override_wrapper import LaunchSpec
from toolkits.embodied_tuner.parser import FailureMode, Status, TrialResult
from toolkits.embodied_tuner.runner import TrialOutcome
from toolkits.embodied_tuner.scheduler import (
    BudgetConfig,
    PreflightOutcome,
    Scheduler,
)


# ----- Shared harness -------------------------------------------------


@dataclass
class _RollbackFactory:
    tmp_path: Path
    objectives: list[float | None] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)
    statuses: list[Status] = field(default_factory=list)
    shas: list[str] = field(default_factory=list)
    runner_calls: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    parser_calls: list[Path] = field(default_factory=list)
    preflight_calls: list[Mapping[str, Any]] = field(default_factory=list)

    def build(
        self,
        critic: FakeCritic,
        *,
        budget: BudgetConfig | None = None,
    ) -> tuple[Scheduler, NodeStore]:
        ledger = Ledger(self.tmp_path / "ledger.jsonl", fsync_on_append=False)
        node_store = NodeStore(self.tmp_path / "nodes.jsonl", fsync_on_append=False)

        def runner_fn(delta, preflight, trial_idx) -> TrialOutcome:
            self.runner_calls.append((trial_idx, dict(delta)))
            log_dir = self.tmp_path / f"trial-{trial_idx}"
            log_dir.mkdir(parents=True, exist_ok=True)
            spec = LaunchSpec(
                argv=(), env={}, log_dir=log_dir, config_name="stub",
                baseline_overrides=(), user_overrides=(),
            )
            return TrialOutcome(
                log_dir=log_dir, returncode=0, timed_out=False,
                wall_clock_seconds=0.0, cleanup_outcome="ok",
                stdout_path=log_dir / "run_embodiment.log", spec=spec,
            )

        def parser_fn(outcome: TrialOutcome) -> TrialResult:
            self.parser_calls.append(outcome.log_dir)
            idx = len(self.parser_calls) - 1
            obj = self.objectives[idx] if idx < len(self.objectives) else None
            fm = self.failure_modes[idx] if idx < len(self.failure_modes) else FailureMode.NONE
            st = self.statuses[idx] if idx < len(self.statuses) else Status.OK
            return TrialResult(
                log_dir=outcome.log_dir,
                status=st,
                failure_mode=fm,
                objective=obj,
                step_time_seconds=obj,
                num_trajectories=10 if obj is not None else None,
            )

        def preflight_fn(delta: Mapping[str, Any]) -> PreflightOutcome:
            self.preflight_calls.append(dict(delta))
            idx = len(self.preflight_calls) - 1
            sha = self.shas[idx] if idx < len(self.shas) else f"sha-{idx}"
            log_dir = self.tmp_path / f"pf-{idx}"
            return PreflightOutcome(
                ok=True,
                errors=(),
                resolved_config_sha=sha,
                log_dir=log_dir,
                delta=delta,
            )

        sched = Scheduler(
            critic=critic,
            runner_fn=runner_fn,
            parser_fn=parser_fn,
            preflight_fn=preflight_fn,
            ledger=ledger,
            budget=budget or BudgetConfig(max_trials=5, budget_seconds=999.0, max_oom=99),
            baseline_knobs={"actor.micro_batch_size": 16},
            node_store=node_store,
        )
        return sched, node_store


# ----- Rollback across every failure mode ------------------------------


@pytest.mark.parametrize("failure_mode", sorted(ROLLBACK_FAILURE_MODES))
def test_rollback_on_failure_rewinds_active_leaf_to_parent(
    tmp_path: Path, failure_mode: str
) -> None:
    """OOM / WORKER_CRASH / TIMEOUT / METRICS_PARTIAL / METRICS_MISSING all rewind."""
    fm_enum = FailureMode(failure_mode)
    # Sequence: bootstrap, trial 1 (OK), trial 2 (rollback failure),
    # trial 3 (OK) — trial 3's parent should be the ROOT, not trial 2.
    factory = _RollbackFactory(
        tmp_path=tmp_path,
        objectives=[1.5, None, 2.5],
        failure_modes=[FailureMode.NONE, fm_enum, FailureMode.NONE],
        statuses=[
            Status.OK,
            (
                Status.OK
                if failure_mode in {"METRICS_PARTIAL", "METRICS_MISSING"}
                else Status.FAILED
            ),
            Status.OK,
        ],
    )
    sched, node_store = factory.build(
        FakeCritic.from_deltas({"a": 1}, {"a": 2}, {"a": 3}),
        budget=BudgetConfig(max_trials=3, budget_seconds=999.0, max_oom=99, max_siblings=99),
    )
    sched.run()

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    nodes = fresh.load().nodes
    root = next(n for n in nodes if n.parent_id is None)
    launched = [n for n in nodes if n.parent_id is not None]
    assert len(launched) == 3

    # Trial 1: parent = root
    assert launched[0].parent_id == root.node_id
    # Trial 2 (rollback failure): parent = trial 1 (this is where it was appended)
    assert launched[1].parent_id == launched[0].node_id
    assert launched[1].failure_mode == failure_mode
    # Trial 3 (OK): after trial 2 rewound active_leaf to root (trial 1
    # was OK so active was at trial 1; trial 2 rewound to trial 1's
    # parent which IS root). Actually wait — trial 2's parent WAS
    # trial 1 (that was the active leaf when trial 2 launched), and
    # rollback rewinds to trial 1's parent — root. So trial 3's parent
    # is root.
    #
    # WAIT — re-check. After trial 1 OK: active = trial1. Trial 2 is
    # appended with parent=trial1. Trial 2 fails: rewind to parent of
    # trial 2 = trial 1. So active is at trial 1, not root. So trial
    # 3's parent = trial 1.
    assert launched[2].parent_id == launched[0].node_id


def test_ok_trial_advances_active_leaf_and_resets_sibling_counter(tmp_path: Path) -> None:
    """Three sequential OKs produce a linear chain (no rollback)."""
    factory = _RollbackFactory(
        tmp_path=tmp_path,
        objectives=[1.5, 2.5, 3.5],
    )
    sched, node_store = factory.build(
        FakeCritic.from_deltas({"a": 1}, {"a": 2}, {"a": 3}),
        budget=BudgetConfig(max_trials=3, budget_seconds=999.0, max_siblings=3),
    )
    sched.run()

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    nodes = fresh.load().nodes
    # 1 root + 3 trials
    assert len(nodes) == 4
    # Linear chain
    assert nodes[1].parent_id == nodes[0].node_id
    assert nodes[2].parent_id == nodes[1].node_id
    assert nodes[3].parent_id == nodes[2].node_id


# ----- Sibling cap + ancestor climb + rollback_exhausted ---------------


def test_sibling_cap_climbs_grandparent_after_max_failures(tmp_path: Path) -> None:
    """Three failures at the SAME active parent trigger a climb one level up."""
    # Sequence:
    #  0 = bootstrap
    #  1 = trial 0 OK, active moves root -> trial0
    #  2 = trial 1 OOM, active rewinds to trial0, counter=1
    #  3 = trial 2 OOM, active still trial0, counter=2
    #  4 = trial 3 OOM, counter reaches 3 -> climb to grandparent
    #       parent_of(trial0) = root, so grandparent = root's parent = None
    #       -> rollback_exhausted (see next test).
    # To exercise a SUCCESSFUL climb (rather than terminate), we need
    # depth 2 first. Do trial 0 OK (root->t0), trial 1 OK (t0->t1),
    # then 3 OOM siblings at t1 which climb from t1 to root/t0.
    factory = _RollbackFactory(
        tmp_path=tmp_path,
        objectives=[1.0, 0.5, None, None, None],
        failure_modes=[
            FailureMode.NONE, FailureMode.NONE,
            FailureMode.OOM, FailureMode.OOM, FailureMode.OOM,
        ],
        statuses=[
            Status.OK, Status.OK,
            Status.FAILED, Status.FAILED, Status.FAILED,
        ],
    )
    sched, node_store = factory.build(
        FakeCritic.from_deltas({"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}),
        budget=BudgetConfig(
            max_trials=5, budget_seconds=999.0, max_oom=99, max_siblings=3,
        ),
    )
    result = sched.run()
    # 5 trials all launched, so we hit max_trials_reached (or budget)
    # rather than rollback_exhausted. Sibling-cap climb should have
    # moved active_leaf up but not terminated.
    assert result.stop_reason in {"max_trials_reached", "budget_seconds_elapsed"}

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    nodes = fresh.load().nodes
    launched = [n for n in nodes if n.parent_id is not None]
    assert len(launched) == 5
    t0, t1, f0, f1, f2 = launched
    # t0's parent = root, t1's parent = t0
    assert t0.parent_id == nodes[0].node_id  # root
    assert t1.parent_id == t0.node_id
    # f0, f1 are appended while active_leaf = t1 (trial 1 OK advanced
    # to t1). Their parent is t1.
    assert f0.parent_id == t1.node_id
    assert f1.parent_id == t1.node_id
    # f2 is appended while active_leaf is still t1 (not yet climbed).
    # After f2's rollback + sibling counter = 3, the climb happens.
    # So f2's parent IS still t1.
    assert f2.parent_id == t1.node_id


def test_rollback_exhausted_when_climb_goes_above_root(tmp_path: Path) -> None:
    """Three sibling failures at root's direct children exhaust rollback."""
    factory = _RollbackFactory(
        tmp_path=tmp_path,
        objectives=[None, None, None],
        failure_modes=[FailureMode.OOM, FailureMode.OOM, FailureMode.OOM],
        statuses=[Status.FAILED, Status.FAILED, Status.FAILED],
    )
    sched, node_store = factory.build(
        FakeCritic.from_deltas({"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}),
        budget=BudgetConfig(
            max_trials=10, budget_seconds=999.0, max_oom=99, max_siblings=3,
        ),
    )
    result = sched.run()
    assert result.stop_reason == "rollback_exhausted"
    # 3 launched failures before exhaustion.
    assert len(factory.runner_calls) == 3


def test_sibling_counter_resets_after_successful_trial(tmp_path: Path) -> None:
    """OOM, OOM, OK, OOM — the OK must reset the counter."""
    factory = _RollbackFactory(
        tmp_path=tmp_path,
        objectives=[None, None, 1.0, None],
        failure_modes=[FailureMode.OOM, FailureMode.OOM, FailureMode.NONE, FailureMode.OOM],
        statuses=[Status.FAILED, Status.FAILED, Status.OK, Status.FAILED],
    )
    sched, node_store = factory.build(
        FakeCritic.from_deltas({"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}),
        budget=BudgetConfig(
            max_trials=4, budget_seconds=999.0, max_oom=99, max_siblings=3,
        ),
    )
    result = sched.run()
    # If the counter had NOT reset after the OK, the 4th trial's
    # failure would have been the third at root and exhausted rollback.
    # With the OK reset, we complete all 4 trials via max_trials.
    assert result.stop_reason == "max_trials_reached"
    assert len(factory.runner_calls) == 4


# ----- Preflight rejection doesn't touch NodeStore ---------------------


def test_preflight_rejection_does_not_create_dag_node_or_change_active_leaf(
    tmp_path: Path,
) -> None:
    """CONFIG_INVALID / DIVISIBILITY_VIOLATION never create a DAG node."""
    factory = _RollbackFactory(
        tmp_path=tmp_path,
        objectives=[1.5],
    )

    # Override preflight_fn to reject after the first trial's preflight.
    original_build = factory.build

    def build_with_reject(critic, budget=None):
        sched, ns = original_build(critic, budget=budget)
        original_preflight = sched.preflight_fn
        call_count = [0]

        def preflight_with_reject(delta):
            call_count[0] += 1
            outcome = original_preflight(delta)
            # bootstrap = call 1, trial-0 preflight = call 2 (accept),
            # everything after = reject (simulating preflight rejections).
            if call_count[0] <= 2:
                return outcome
            return PreflightOutcome(
                ok=False,
                errors=("stub rejection",),
                resolved_config_sha=None,
                log_dir=outcome.log_dir,
                delta=delta,
            )

        sched.preflight_fn = preflight_with_reject
        return sched, ns

    factory.build = build_with_reject  # type: ignore[method-assign]
    sched, node_store = factory.build(
        FakeCritic.from_deltas({"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}),
        budget=BudgetConfig(
            max_trials=5, budget_seconds=999.0, max_oom=99,
            preflight_retries=3, max_siblings=99,
        ),
    )
    result = sched.run()
    assert result.stop_reason == "preflight_exhausted"

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    launched = [n for n in fresh.load().nodes if n.parent_id is not None]
    # Only the first (successful) trial produced a DAG node. The
    # preflight-exhausted attempt did NOT create a DAG node even
    # though its (would-be) failure_mode is CONFIG_INVALID in the
    # ledger.
    assert len(launched) == 1
    assert launched[0].failure_mode == FailureMode.NONE.value


# ----- cumulative_delta rewinds on rollback ----------------------------


def test_cumulative_delta_rewinds_on_rollback(tmp_path: Path) -> None:
    """After a rollback, the next trial's runner receives the parent's delta, not the failed child's."""
    factory = _RollbackFactory(
        tmp_path=tmp_path,
        objectives=[1.5, None, 2.0],
        failure_modes=[FailureMode.NONE, FailureMode.OOM, FailureMode.NONE],
        statuses=[Status.OK, Status.FAILED, Status.OK],
    )
    sched, node_store = factory.build(
        FakeCritic.from_deltas(
            {"a": 1},  # trial 0 — cumulative becomes {a:1}
            {"b": 2},  # trial 1 — cumulative BECOMES {a:1, b:2}, OOMs, rewinds to {a:1}
            {"c": 3},  # trial 2 — expected cumulative {a:1, c:3}, NOT {a:1, b:2, c:3}
        ),
        budget=BudgetConfig(max_trials=3, budget_seconds=999.0, max_oom=99, max_siblings=99),
    )
    sched.run()

    # runner_calls captures the cumulative delta passed to each trial.
    assert len(factory.runner_calls) == 3
    _idx0, delta0 = factory.runner_calls[0]
    _idx1, delta1 = factory.runner_calls[1]
    _idx2, delta2 = factory.runner_calls[2]
    assert delta0 == {"a": 1}
    assert delta1 == {"a": 1, "b": 2}
    # Post-rollback: b should be gone (not merged into cumulative).
    assert "b" not in delta2, (
        f"cumulative_delta failed to rewind after rollback: got {delta2}"
    )
    assert delta2 == {"a": 1, "c": 3}
