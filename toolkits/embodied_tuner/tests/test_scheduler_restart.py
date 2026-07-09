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

"""Tests for restart-safe scheduler state reconstruction (AC-3).

The scheduler MUST derive its live state (active leaf, cumulative
delta, current knobs, trial_idx, oom_count, sibling-failure counter,
rolling history, and last-failure attribution) from the persisted
:class:`NodeStore` + :class:`Ledger` when a fresh process starts
against an existing campaign directory. Without this, a crash between
appending a rollback failure and starting the next iteration would
cause the next process to expand from the failed node instead of
rolling back to its parent.

Every case here follows the same pattern:

1. Build a first :class:`Scheduler`, run it to produce a persisted
   ``nodes.jsonl`` + ``ledger.jsonl`` in ``tmp_path``.
2. Instantiate a SECOND :class:`Scheduler` over the same paths.
3. Assert on the reconstructed state — via
   :meth:`Scheduler._reconstruct_state_from_stores` for the resumed
   snapshot and via a single follow-up trial's ``parent_id`` /
   ``cumulative_delta`` for the end-to-end contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from toolkits.embodied_tuner.fake_critic import FakeCritic
from toolkits.embodied_tuner.ledger import Ledger
from toolkits.embodied_tuner.node_store import NodeStore
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
class _RestartFactory:
    """Deterministic sequence-driven scheduler builder for restart tests.

    Callers list the desired ``(objective, failure_mode, status)`` per
    trial index and get back a ``build(critic)`` factory that produces
    a fresh :class:`Scheduler` bound to ``tmp_path``. Building twice
    over the same ``tmp_path`` yields a second scheduler that will
    read the first run's persisted state on ``run()``.
    """

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
        baseline_knobs: Mapping[str, Any] | None = None,
    ) -> tuple[Scheduler, NodeStore]:
        ledger = Ledger(self.tmp_path / "ledger.jsonl", fsync_on_append=False)
        node_store = NodeStore(
            self.tmp_path / "nodes.jsonl", fsync_on_append=False
        )

        def runner_fn(delta, preflight, trial_idx) -> TrialOutcome:
            self.runner_calls.append((trial_idx, dict(delta)))
            log_dir = self.tmp_path / f"trial-{trial_idx}"
            log_dir.mkdir(parents=True, exist_ok=True)
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
                returncode=0,
                timed_out=False,
                wall_clock_seconds=0.0,
                cleanup_outcome="ok",
                stdout_path=log_dir / "run_embodiment.log",
                spec=spec,
            )

        def parser_fn(outcome: TrialOutcome) -> TrialResult:
            self.parser_calls.append(outcome.log_dir)
            idx = len(self.parser_calls) - 1
            obj = self.objectives[idx] if idx < len(self.objectives) else None
            fm = (
                self.failure_modes[idx]
                if idx < len(self.failure_modes)
                else FailureMode.NONE
            )
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
            budget=budget
            or BudgetConfig(
                max_trials=99, budget_seconds=999.0, max_oom=99, max_siblings=3
            ),
            baseline_knobs=dict(baseline_knobs or {"actor.micro_batch_size": 16}),
            node_store=node_store,
        )
        return sched, node_store


def _seed_first_run(
    tmp_path: Path,
    *,
    deltas: list[Mapping[str, Any]],
    objectives: list[float | None],
    failure_modes: list[FailureMode],
    statuses: list[Status],
    budget: BudgetConfig | None = None,
    baseline_knobs: Mapping[str, Any] | None = None,
) -> _RestartFactory:
    """Run one scheduler campaign end-to-end and return the factory."""
    factory = _RestartFactory(
        tmp_path=tmp_path,
        objectives=objectives,
        failure_modes=failure_modes,
        statuses=statuses,
    )
    sched, _ = factory.build(
        FakeCritic.from_deltas(*deltas),
        budget=budget,
        baseline_knobs=baseline_knobs,
    )
    sched.run()
    return factory


# ----- Restart derivations ---------------------------------------------


def test_restart_after_single_ok_active_leaf_is_the_ok_child(tmp_path: Path) -> None:
    """After ``root -> OK``, restart resumes at the OK child."""
    _seed_first_run(
        tmp_path,
        deltas=[{"actor.micro_batch_size": 32}],
        objectives=[1.0],
        failure_modes=[FailureMode.NONE],
        statuses=[Status.OK],
        budget=BudgetConfig(
            max_trials=1, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    factory_2 = _RestartFactory(tmp_path=tmp_path)
    sched_2, node_store_2 = factory_2.build(FakeCritic.from_deltas({"x": 0}))
    root_id = sched_2._bootstrap_root_if_needed(start=0.0)
    assert root_id is not None
    resumed = sched_2._reconstruct_state_from_stores(root_id=root_id)
    assert resumed is not None

    launched = [n for n in node_store_2.all_nodes() if n.parent_id is not None]
    assert len(launched) == 1
    ok_child = launched[0]

    assert resumed.active_leaf_id == ok_child.node_id
    assert resumed.cumulative_delta == {"actor.micro_batch_size": 32}
    assert resumed.current_knobs == {"actor.micro_batch_size": 32}
    assert resumed.trial_idx == 1
    assert resumed.oom_count == 0
    assert resumed.sibling_failures_at_active_parent == 0
    assert resumed.last_failure_mode == FailureMode.NONE.value
    assert resumed.last_failed_trial_idx is None
    assert resumed.last_failed_delta is None
    assert len(resumed.history) == 1
    assert resumed.history[0].status == Status.OK.value
    assert resumed.duplicate_counter == 0


def test_restart_after_ok_then_oom_rolls_back_to_ok_child(tmp_path: Path) -> None:
    """After ``root -> OK(A) -> OOM(B)``, restart resumes at A, not B (AC-3 core case)."""
    _seed_first_run(
        tmp_path,
        deltas=[{"actor.micro_batch_size": 32}, {"actor.micro_batch_size": 64}],
        objectives=[1.0, None],
        failure_modes=[FailureMode.NONE, FailureMode.OOM],
        statuses=[Status.OK, Status.FAILED],
        budget=BudgetConfig(
            max_trials=2, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    factory_2 = _RestartFactory(tmp_path=tmp_path)
    sched_2, node_store_2 = factory_2.build(FakeCritic.from_deltas({"x": 0}))
    root_id = sched_2._bootstrap_root_if_needed(start=0.0)
    assert root_id is not None
    resumed = sched_2._reconstruct_state_from_stores(root_id=root_id)
    assert resumed is not None

    launched = [n for n in node_store_2.all_nodes() if n.parent_id is not None]
    ok_a, oom_b = launched
    assert oom_b.failure_mode == FailureMode.OOM.value

    # AC-3: the failing OOM node MUST NOT be the active leaf on restart.
    assert resumed.active_leaf_id == ok_a.node_id
    # Cumulative delta reflects the OK ancestor, not the failed proposal.
    assert resumed.cumulative_delta == {"actor.micro_batch_size": 32}
    assert resumed.current_knobs == {"actor.micro_batch_size": 32}
    # trial_idx counts launched trials: next launched will be #2.
    assert resumed.trial_idx == 2
    assert resumed.oom_count == 1
    # One OOM sibling has hit the OK parent so far; sibling cap == 3
    # means one more failure will accumulate to 2 (still under cap).
    assert resumed.sibling_failures_at_active_parent == 1
    assert resumed.last_failure_mode == FailureMode.OOM.value
    assert resumed.last_failed_trial_idx == 1
    assert resumed.last_failed_delta == {"actor.micro_batch_size": 64}


def test_restart_then_next_trial_child_is_ok_ancestor_not_failed_leaf(
    tmp_path: Path,
) -> None:
    """End-to-end: after restart, next launched trial's parent is the OK ancestor."""
    _seed_first_run(
        tmp_path,
        deltas=[{"actor.micro_batch_size": 32}, {"actor.micro_batch_size": 64}],
        objectives=[1.0, None],
        failure_modes=[FailureMode.NONE, FailureMode.OOM],
        statuses=[Status.OK, Status.FAILED],
        budget=BudgetConfig(
            max_trials=2, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    # Second run: budget headroom for one more launched trial on top of
    # the 2 already persisted. ``max_trials`` is a campaign-wide cap
    # (compared against reconstructed ``trial_idx``), not a per-session
    # window, so we allow 3.
    factory_2 = _RestartFactory(
        tmp_path=tmp_path,
        objectives=[2.0],
        failure_modes=[FailureMode.NONE],
        statuses=[Status.OK],
    )
    sched_2, node_store_2 = factory_2.build(
        FakeCritic.from_deltas({"actor.micro_batch_size": 24}),
        budget=BudgetConfig(
            max_trials=3, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )
    sched_2.run()

    launched = [n for n in node_store_2.all_nodes() if n.parent_id is not None]
    ok_a, oom_b, next_trial = launched
    assert next_trial.parent_id == ok_a.node_id
    assert next_trial.trial_idx == 2
    # Cumulative delta layered on top of the OK ancestor's cumulative
    # state, NOT on top of the failed proposal.
    assert next_trial.cumulative_delta == {"actor.micro_batch_size": 24}


def test_restart_after_two_failed_siblings_carries_sibling_counter(
    tmp_path: Path,
) -> None:
    """Two consecutive failures at the same parent must preserve the counter across restart."""
    _seed_first_run(
        tmp_path,
        deltas=[
            {"actor.micro_batch_size": 32},
            {"actor.micro_batch_size": 64},
            {"actor.micro_batch_size": 128},
        ],
        objectives=[1.0, None, None],
        failure_modes=[FailureMode.NONE, FailureMode.OOM, FailureMode.OOM],
        statuses=[Status.OK, Status.FAILED, Status.FAILED],
        budget=BudgetConfig(
            max_trials=3, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    factory_2 = _RestartFactory(tmp_path=tmp_path)
    sched_2, node_store_2 = factory_2.build(FakeCritic.from_deltas({"x": 0}))
    root_id = sched_2._bootstrap_root_if_needed(start=0.0)
    resumed = sched_2._reconstruct_state_from_stores(root_id=root_id)
    assert resumed is not None

    launched = [n for n in node_store_2.all_nodes() if n.parent_id is not None]
    ok_a, _oom_b, _oom_c = launched
    assert resumed.active_leaf_id == ok_a.node_id
    # Two sibling failures under the OK ancestor — a third would trip
    # the cap (max_siblings=3) and climb.
    assert resumed.sibling_failures_at_active_parent == 2
    assert resumed.oom_count == 2
    assert resumed.trial_idx == 3


def test_restart_after_sibling_cap_reached_climbs_to_root(tmp_path: Path) -> None:
    """Three failures at the OK ancestor climb to root; restart replays the climb."""
    _seed_first_run(
        tmp_path,
        deltas=[
            {"actor.micro_batch_size": 32},
            {"actor.micro_batch_size": 64},
            {"actor.micro_batch_size": 128},
            {"actor.micro_batch_size": 256},
        ],
        objectives=[1.0, None, None, None],
        failure_modes=[
            FailureMode.NONE,
            FailureMode.OOM,
            FailureMode.OOM,
            FailureMode.OOM,
        ],
        statuses=[Status.OK, Status.FAILED, Status.FAILED, Status.FAILED],
        budget=BudgetConfig(
            max_trials=4, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    factory_2 = _RestartFactory(tmp_path=tmp_path)
    sched_2, node_store_2 = factory_2.build(FakeCritic.from_deltas({"x": 0}))
    root_id = sched_2._bootstrap_root_if_needed(start=0.0)
    resumed = sched_2._reconstruct_state_from_stores(root_id=root_id)
    assert resumed is not None

    # After the third failure at OK-ancestor, the climb walks one level:
    # from OK-ancestor to its parent (root). Counter resets to 0.
    assert resumed.active_leaf_id == root_id
    assert resumed.cumulative_delta == {}
    assert resumed.current_knobs == {"actor.micro_batch_size": 16}
    assert resumed.sibling_failures_at_active_parent == 0
    assert resumed.oom_count == 3
    assert resumed.trial_idx == 4


def test_restart_after_climb_above_root_anchors_at_root(tmp_path: Path) -> None:
    """When the AC-5 replay walks above root, resume anchors at root not None."""
    # 3 direct-of-root sibling failures with max_siblings=3 exhaust
    # rollback (the first campaign terminates with rollback_exhausted).
    _seed_first_run(
        tmp_path,
        deltas=[
            {"actor.micro_batch_size": 32},
            {"actor.micro_batch_size": 64},
            {"actor.micro_batch_size": 128},
        ],
        objectives=[None, None, None],
        failure_modes=[FailureMode.OOM, FailureMode.OOM, FailureMode.OOM],
        statuses=[Status.FAILED, Status.FAILED, Status.FAILED],
        budget=BudgetConfig(
            max_trials=3, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    factory_2 = _RestartFactory(tmp_path=tmp_path)
    sched_2, _ = factory_2.build(FakeCritic.from_deltas({"x": 0}))
    root_id = sched_2._bootstrap_root_if_needed(start=0.0)
    resumed = sched_2._reconstruct_state_from_stores(root_id=root_id)
    assert resumed is not None
    # active_state() returns (None, 0) after climb-above-root; resume
    # must anchor at root so the next launched failure fires
    # rollback_exhausted on the normal path.
    assert resumed.active_leaf_id == root_id
    assert resumed.sibling_failures_at_active_parent == 0
    assert resumed.oom_count == 3
    assert resumed.trial_idx == 3


def test_restart_current_knobs_equals_baseline_plus_active_cumulative(
    tmp_path: Path,
) -> None:
    """AC-3 invariant: current_knobs == apply_delta(baseline, active.cumulative_delta)."""
    _seed_first_run(
        tmp_path,
        deltas=[
            {"actor.micro_batch_size": 32},
            {"rollout.gpu_memory_utilization": 0.8},
        ],
        objectives=[1.0, 0.5],
        failure_modes=[FailureMode.NONE, FailureMode.NONE],
        statuses=[Status.OK, Status.OK],
        budget=BudgetConfig(
            max_trials=2, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    baseline = {"actor.micro_batch_size": 16, "rollout.gpu_memory_utilization": 0.6}
    factory_2 = _RestartFactory(tmp_path=tmp_path)
    sched_2, node_store_2 = factory_2.build(
        FakeCritic.from_deltas({"x": 0}), baseline_knobs=baseline
    )
    root_id = sched_2._bootstrap_root_if_needed(start=0.0)
    resumed = sched_2._reconstruct_state_from_stores(root_id=root_id)
    assert resumed is not None

    active_node = node_store_2.get(resumed.active_leaf_id)
    assert active_node is not None
    expected = dict(baseline)
    expected.update(active_node.cumulative_delta)
    assert resumed.current_knobs == expected
    assert resumed.cumulative_delta == dict(active_node.cumulative_delta)


def test_restart_on_empty_store_returns_none(tmp_path: Path) -> None:
    """Fresh campaign (only root, empty ledger) needs no reconstruction."""
    factory = _RestartFactory(tmp_path=tmp_path)
    sched, _ = factory.build(FakeCritic.from_deltas({"x": 0}))
    root_id = sched._bootstrap_root_if_needed(start=0.0)
    assert root_id is not None
    # Nothing to reconstruct — the caller keeps fresh-campaign defaults.
    assert sched._reconstruct_state_from_stores(root_id=root_id) is None


def test_restart_history_window_matches_ledger_tail(tmp_path: Path) -> None:
    """Reconstructed history mirrors the flat ledger's tail up to history_window."""
    _seed_first_run(
        tmp_path,
        deltas=[
            {"actor.micro_batch_size": 32},
            {"actor.micro_batch_size": 64},
            {"actor.micro_batch_size": 128},
        ],
        objectives=[1.0, 0.5, 0.25],
        failure_modes=[FailureMode.NONE, FailureMode.NONE, FailureMode.NONE],
        statuses=[Status.OK, Status.OK, Status.OK],
        budget=BudgetConfig(
            max_trials=3, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    factory_2 = _RestartFactory(tmp_path=tmp_path)
    # history_window default = 4 (via BudgetConfig); test caps it to 2 to
    # verify tail-window semantics on the reconstructed history.
    sched_2, _ = factory_2.build(
        FakeCritic.from_deltas({"x": 0}),
        budget=BudgetConfig(
            max_trials=1,
            budget_seconds=999.0,
            max_oom=99,
            max_siblings=3,
            history_window=2,
        ),
    )
    root_id = sched_2._bootstrap_root_if_needed(start=0.0)
    resumed = sched_2._reconstruct_state_from_stores(root_id=root_id)
    assert resumed is not None
    assert len(resumed.history) == 2
    # Tail preserved in order.
    assert [h.trial_idx for h in resumed.history] == [1, 2]


def test_bootstrap_no_longer_returns_last_appended_node(tmp_path: Path) -> None:
    """Regression guard: bootstrap must return root id, not the last node.

    Round-0 code returned ``all_nodes[-1].node_id`` from
    ``_bootstrap_root_if_needed`` when the store already had content —
    which silently violated AC-3 by re-anchoring at the failing node.
    The current contract is: bootstrap always returns the ROOT id; the
    rollback-aware active leaf is derived by
    ``_reconstruct_state_from_stores``.
    """
    _seed_first_run(
        tmp_path,
        deltas=[{"actor.micro_batch_size": 32}, {"actor.micro_batch_size": 64}],
        objectives=[1.0, None],
        failure_modes=[FailureMode.NONE, FailureMode.OOM],
        statuses=[Status.OK, Status.FAILED],
        budget=BudgetConfig(
            max_trials=2, budget_seconds=999.0, max_oom=99, max_siblings=3
        ),
    )

    factory_2 = _RestartFactory(tmp_path=tmp_path)
    sched_2, node_store_2 = factory_2.build(FakeCritic.from_deltas({"x": 0}))
    root_id = sched_2._bootstrap_root_if_needed(start=0.0)
    assert root_id is not None
    # The last-appended node is the OOM child. Bootstrap MUST NOT return it.
    launched = [n for n in node_store_2.all_nodes() if n.parent_id is not None]
    last_appended = launched[-1]
    assert last_appended.failure_mode == FailureMode.OOM.value
    assert root_id != last_appended.node_id
    root_node = node_store_2.root()
    assert root_node is not None
    assert root_id == root_node.node_id
