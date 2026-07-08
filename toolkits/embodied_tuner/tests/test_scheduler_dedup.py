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

"""Integration tests for :class:`Scheduler` + :class:`ConfigDedupIndex`.

Covers AC-6: duplicate-OK short-circuits the runner and emits a
synthetic ``DUPLICATE_OF`` DAGNode with ``duplicate_of_node_id``
pointing at the ORIGINAL non-duplicate; duplicate-FAILED is rejected
via ``preflight_feedback`` sharing the preflight-retry budget;
DUPLICATE_OF entries are excluded from best-selection and plateau
eligibility.

The scheduler tests use a preflight stub that can be programmed to
return different resolved-config SHAs across attempts so we can
exercise the dedup path deterministically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from toolkits.embodied_tuner.config_dedup_index import ConfigDedupIndex
from toolkits.embodied_tuner.fake_critic import FakeCritic
from toolkits.embodied_tuner.ledger import Ledger
from toolkits.embodied_tuner.node_store import (
    DUPLICATE_OF_FAILURE_MODE,
    NodeStore,
)
from toolkits.embodied_tuner.override_wrapper import LaunchSpec
from toolkits.embodied_tuner.parser import FailureMode, Status, TrialResult
from toolkits.embodied_tuner.runner import TrialOutcome
from toolkits.embodied_tuner.scheduler import (
    BudgetConfig,
    PreflightOutcome,
    Scheduler,
)


# ----- Programmable preflight stub -------------------------------------


@dataclass
class _DedupFactory:
    """Scheduler harness where the preflight stub can vary SHAs per call."""

    tmp_path: Path
    shas: list[str] = field(default_factory=list)
    objectives: list[float | None] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)
    statuses: list[Status] = field(default_factory=list)
    runner_calls: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    parser_calls: list[Path] = field(default_factory=list)
    preflight_calls: list[Mapping[str, Any]] = field(default_factory=list)

    def build(
        self,
        critic: FakeCritic,
        *,
        budget: BudgetConfig | None = None,
        node_store: NodeStore | None = None,
        dedup_index: ConfigDedupIndex | None = None,
    ) -> Scheduler:
        ledger = Ledger(self.tmp_path / "ledger.jsonl", fsync_on_append=False)

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
            sha = self.shas[idx] if idx < len(self.shas) else "sha-default"
            log_dir = self.tmp_path / f"pf-{idx}"
            return PreflightOutcome(
                ok=True,
                errors=(),
                resolved_config_sha=sha,
                log_dir=log_dir,
                delta=delta,
            )

        return Scheduler(
            critic=critic,
            runner_fn=runner_fn,
            parser_fn=parser_fn,
            preflight_fn=preflight_fn,
            ledger=ledger,
            budget=budget or BudgetConfig(max_trials=5, budget_seconds=999.0, max_oom=99),
            baseline_knobs={},
            node_store=node_store,
            dedup_index=dedup_index,
        )


# ----- Duplicate-of-OK short-circuit -----------------------------------


def test_duplicate_of_ok_short_circuits_runner(tmp_path: Path) -> None:
    """A proposal whose SHA matches a prior OK trial must skip the runner."""
    node_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dedup = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    factory = _DedupFactory(
        tmp_path=tmp_path,
        # 1 bootstrap preflight + 2 trial preflights, both resolving to
        # the SAME SHA. The second must short-circuit.
        shas=["sha-baseline", "sha-A", "sha-A"],
        objectives=[1.5],  # only 1 real trial launches
    )
    scheduler = factory.build(
        FakeCritic.from_deltas({"micro_batch_size": 32}, {"micro_batch_size": 32}),
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
        node_store=node_store,
        dedup_index=dedup,
    )
    result = scheduler.run()

    # Runner was invoked exactly once (for the first trial); the
    # second attempt short-circuited.
    assert len(factory.runner_calls) == 1
    assert result.trial_count == 1  # duplicate-OK does not consume a slot

    fresh_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    nodes = fresh_store.load().nodes
    # 1 root + 1 real trial + 1 synthetic duplicate = 3 nodes.
    assert len(nodes) == 3
    launched = [n for n in nodes if n.trial_idx is not None and n.trial_idx >= 0]
    duplicates = [n for n in nodes if n.failure_mode == DUPLICATE_OF_FAILURE_MODE]
    assert len(launched) == 1
    assert len(duplicates) == 1
    dup = duplicates[0]
    assert dup.duplicate_of_node_id == launched[0].node_id
    assert dup.objective == launched[0].objective


def test_duplicate_of_ok_back_reference_points_at_original_even_across_chains(
    tmp_path: Path,
) -> None:
    """Third duplicate points at the ORIGINAL, not the second duplicate (AC-6)."""
    node_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dedup = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    # bootstrap + trial1 (real) + trial2 (duplicate) + trial3 (duplicate)
    # trial4 (real, breaks the loop) so max_trials fires cleanly.
    factory = _DedupFactory(
        tmp_path=tmp_path,
        shas=["sha-baseline", "sha-A", "sha-A", "sha-A", "sha-B"],
        objectives=[1.5, 2.5],
    )
    scheduler = factory.build(
        FakeCritic.from_deltas({"x": 1}, {"x": 1}, {"x": 1}, {"x": 2}),
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
        node_store=node_store,
        dedup_index=dedup,
    )
    scheduler.run()

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    nodes = fresh.load().nodes
    duplicates = [n for n in nodes if n.failure_mode == DUPLICATE_OF_FAILURE_MODE]
    launched = [n for n in nodes if n.trial_idx is not None and n.trial_idx >= 0]
    assert len(duplicates) == 2, "expected two duplicates of sha-A"
    original = launched[0]  # the first trial that actually ran sha-A
    for dup in duplicates:
        assert dup.duplicate_of_node_id == original.node_id
        assert dup.objective == original.objective


def test_duplicate_of_ok_excluded_from_best_selection(tmp_path: Path) -> None:
    """`NodeStore.best_ok_leaf()` and `Ledger.best()` must skip DUPLICATE_OF."""
    node_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dedup = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    factory = _DedupFactory(
        tmp_path=tmp_path,
        shas=["sha-baseline", "sha-A", "sha-A"],
        objectives=[3.0],
    )
    scheduler = factory.build(
        FakeCritic.from_deltas({"x": 1}, {"x": 1}),
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
        node_store=node_store,
        dedup_index=dedup,
    )
    result = scheduler.run()
    # Best is the single real trial; the duplicate must not appear.
    best = result.best_entry
    assert best is not None
    assert best.failure_mode == FailureMode.NONE.value
    # NodeStore side of best selection also excludes DUPLICATE_OF.
    fresh_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    best_leaf = fresh_store.best_ok_leaf()
    assert best_leaf is not None
    assert best_leaf.failure_mode == FailureMode.NONE.value


def test_duplicate_of_ok_does_not_write_ledger_entry(tmp_path: Path) -> None:
    """Ledger stays free of synthetic DUPLICATE_OF rows (plot compatibility)."""
    node_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dedup = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    # Sequence: bootstrap, trial 1 (real OK sha-A), dup attempt (sha-A
    # short-circuits), trial 2 (real OK sha-B) so max_trials=2 fires
    # naturally after 2 real launched trials + 1 duplicate.
    factory = _DedupFactory(
        tmp_path=tmp_path,
        shas=["sha-baseline", "sha-A", "sha-A", "sha-B"],
        objectives=[1.5, 2.5],
    )
    scheduler = factory.build(
        FakeCritic.from_deltas({"x": 1}, {"x": 1}, {"x": 2}),
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
        node_store=node_store,
        dedup_index=dedup,
    )
    scheduler.run()

    fresh_ledger = Ledger(scheduler.ledger.path, fsync_on_append=False)
    ledger_entries = fresh_ledger.load().entries
    # Exactly two real trials should have written Ledger rows; the
    # dedup short-circuit in between must not have appended anything.
    assert len(ledger_entries) == 2, (
        "duplicate-OK short-circuit must not append a Ledger row; "
        f"got failure_modes={[e.failure_mode for e in ledger_entries]}"
    )
    for entry in ledger_entries:
        assert entry.failure_mode == FailureMode.NONE.value


# ----- Duplicate-of-FAILED rejection -----------------------------------


def test_duplicate_of_failed_rejects_via_preflight_feedback(tmp_path: Path) -> None:
    """A SHA that previously failed must be rejected before reaching the runner."""
    node_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dedup = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    # Preflight sequence:
    #  0: bootstrap
    #  1: trial 1 (FAILED, OOM) — records sha-A as failed in dedup
    #  2..5: 4 retries of trial 2, all resolving to sha-A (rejected each time)
    #     preflight_retries=3 -> 4 attempts before exhausted -> preflight_exhausted
    factory = _DedupFactory(
        tmp_path=tmp_path,
        shas=[
            "sha-baseline",
            "sha-A",  # trial 1 (runs, FAILED)
            "sha-A", "sha-A", "sha-A", "sha-A",  # trial 2 (all dedup-rejected)
        ],
        objectives=[None],
        failure_modes=[FailureMode.OOM],
        statuses=[Status.FAILED],
    )
    scheduler = factory.build(
        FakeCritic.from_deltas({"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}),
        budget=BudgetConfig(
            max_trials=5,
            budget_seconds=999.0,
            preflight_retries=3,
            max_oom=99,
        ),
        node_store=node_store,
        dedup_index=dedup,
    )
    result = scheduler.run()
    assert result.stop_reason == "preflight_exhausted"
    # Only trial 1 launched; every subsequent attempt was rejected via
    # preflight_feedback (dedup-FAILED short-circuit).
    assert len(factory.runner_calls) == 1


def test_duplicate_of_failed_does_not_create_dag_node(tmp_path: Path) -> None:
    node_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dedup = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    factory = _DedupFactory(
        tmp_path=tmp_path,
        shas=[
            "sha-baseline",
            "sha-A",
            "sha-A", "sha-A", "sha-A", "sha-A",
        ],
        objectives=[None],
        failure_modes=[FailureMode.OOM],
        statuses=[Status.FAILED],
    )
    scheduler = factory.build(
        FakeCritic.from_deltas({"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}),
        budget=BudgetConfig(
            max_trials=5,
            budget_seconds=999.0,
            preflight_retries=3,
            max_oom=99,
        ),
        node_store=node_store,
        dedup_index=dedup,
    )
    scheduler.run()

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    launched = [n for n in fresh.load().nodes if n.trial_idx is not None and n.trial_idx >= 0]
    # Only the OOM'd real trial is in the DAG. Dedup-FAILED rejections
    # never create a DAG node — that behaviour matches AC-5's "preflight
    # rejections do not touch NodeStore" semantics for a similar reason:
    # the runner never launched.
    assert len(launched) == 1
    assert launched[0].failure_mode == FailureMode.OOM.value


# ----- Dedup index rebuild across restarts -----------------------------


def test_dedup_index_rebuilt_on_scheduler_restart(tmp_path: Path) -> None:
    """A fresh Scheduler on the same directory must rebuild the dedup index."""
    node_store_a = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dedup_a = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    factory_a = _DedupFactory(
        tmp_path=tmp_path,
        shas=["sha-baseline", "sha-A"],
        objectives=[1.5],
    )
    scheduler_a = factory_a.build(
        FakeCritic.from_deltas({"x": 1}),
        budget=BudgetConfig(max_trials=1, budget_seconds=999.0),
        node_store=node_store_a,
        dedup_index=dedup_a,
    )
    scheduler_a.run()

    # Fresh scheduler on same directory. Delete the sidecar to force
    # rebuild-from-node-store (a lost sidecar must recover cleanly).
    (tmp_path / "cdi.jsonl").unlink()
    node_store_b = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dedup_b = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    factory_b = _DedupFactory(
        tmp_path=tmp_path,
        shas=["sha-baseline", "sha-A"],  # second attempt at sha-A
        objectives=[],
    )
    scheduler_b = factory_b.build(
        FakeCritic.from_deltas({"x": 1}),
        budget=BudgetConfig(max_trials=1, budget_seconds=999.0),
        node_store=node_store_b,
        dedup_index=dedup_b,
    )
    scheduler_b.run()

    # After rebuild, the second attempt at sha-A must have short-
    # circuited (no runner call in factory_b).
    assert len(factory_b.runner_calls) == 0


# ----- Backward compat: no dedup_index --------------------------------


def test_scheduler_without_dedup_index_behaves_unchanged(tmp_path: Path) -> None:
    """When dedup_index is None, the scheduler must not consult it or short-circuit."""
    node_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    factory = _DedupFactory(
        tmp_path=tmp_path,
        shas=["sha-baseline", "sha-A", "sha-A"],  # 2 identical proposals
        objectives=[1.5, 2.5],
    )
    scheduler = factory.build(
        FakeCritic.from_deltas({"x": 1}, {"x": 2}),
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
        node_store=node_store,
        dedup_index=None,
    )
    scheduler.run()
    # Both trials launched (no dedup short-circuit).
    assert len(factory.runner_calls) == 2
