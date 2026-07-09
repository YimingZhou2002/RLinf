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

"""Coexistence tests for :class:`Scheduler` <-> :class:`NodeStore` wiring.

The Scheduler mirrors every trial's ``LedgerEntry`` into an
authoritative DAG :class:`NodeStore`. This test module verifies:

- Backward compatibility: when ``node_store`` is left as its default
  ``None``, the Scheduler behaves exactly as before (no bootstrap, no
  mirror step, no interaction with any nodes.jsonl on disk).
- Root bootstrap: on the first ``run()`` invocation, a root DAGNode is
  written using the baseline preflight SHA; the root is idempotent
  across restarts.
- Per-trial mirror: every launched trial produces one DAGNode whose
  fields carry the same information as the corresponding LedgerEntry,
  with parent_id chaining back through the campaign in insertion order.
- Preflight-rejected trials do not create a DAGNode (they never reach
  the runner and therefore aren't launched trials).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from toolkits.embodied_tuner.fake_critic import FakeCritic
from toolkits.embodied_tuner.ledger import Ledger
from toolkits.embodied_tuner.node_store import (
    NodeStore,
    ROOT_STATUS,
)
from toolkits.embodied_tuner.parser import FailureMode, Status
from toolkits.embodied_tuner.scheduler import BudgetConfig

from toolkits.embodied_tuner.tests.test_scheduler import _SchedulerFactory


def _build_with_node_store(
    tmp_path: Path,
    critic: FakeCritic,
    *,
    objectives: list[float | None] | None = None,
    failure_modes: list[FailureMode] | None = None,
    statuses: list[Status] | None = None,
    preflight_results: list[bool] | None = None,
    budget: BudgetConfig | None = None,
) -> tuple[_SchedulerFactory, NodeStore]:
    """Convenience: return ``(factory, node_store)`` with node_store wired in.

    Reuses :class:`_SchedulerFactory` from ``test_scheduler`` so the
    stub runner/parser/preflight behaviour matches every other
    scheduler test in this repo. The returned factory has the scheduler
    already built and its ``node_store`` attribute set.
    """
    factory = _SchedulerFactory(
        tmp_path=tmp_path,
        objectives=objectives or [],
        failure_modes=failure_modes or [],
        statuses=statuses or [],
        preflight_results=preflight_results or [],
    )
    node_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    scheduler = factory.build(critic, budget=budget)
    scheduler.node_store = node_store
    factory.scheduler = scheduler  # type: ignore[attr-defined]
    return factory, node_store


# ----- Backward-compat: node_store=None ------------------------------


def test_default_scheduler_does_not_touch_node_store(tmp_path: Path) -> None:
    """When ``node_store`` is None the scheduler must not create nodes.jsonl."""
    factory = _SchedulerFactory(tmp_path=tmp_path, objectives=[100.0, 90.0])
    critic = FakeCritic.from_deltas({}, {})
    scheduler = factory.build(critic, budget=BudgetConfig(max_trials=2, budget_seconds=999.0))
    result = scheduler.run()
    assert result.trial_count == 2
    # nodes.jsonl must NOT be created when the store is not wired in.
    assert not (tmp_path / "nodes.jsonl").exists()


# ----- Root bootstrap -------------------------------------------------


def test_bootstrap_writes_root_node_on_first_run(tmp_path: Path) -> None:
    factory, node_store = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas({}, {}),
        objectives=[100.0, 90.0],
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
    )
    factory.scheduler.run()  # type: ignore[attr-defined]

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    result = fresh.load()
    # 1 root + 2 launched trials = 3 nodes total.
    assert len(result.nodes) == 3
    root = result.nodes[0]
    assert root.parent_id is None
    assert root.status == ROOT_STATUS
    assert root.objective is None
    assert root.trial_idx is None
    # Root's resolved_config_sha must match the baseline preflight SHA
    # (the factory's _ok_preflight returns "deadbeef" for every delta,
    # including the empty baseline delta).
    assert root.resolved_config_sha == "deadbeef"


def test_bootstrap_is_idempotent_across_restarts(tmp_path: Path) -> None:
    factory, node_store = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas({}, {}),
        objectives=[100.0, 90.0],
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
    )
    factory.scheduler.run()  # type: ignore[attr-defined]

    # Re-run against the SAME directory: the root must not be duplicated.
    factory2, node_store2 = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas({}, {}),
        objectives=[80.0, 70.0],
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
    )
    factory2.scheduler.run()  # type: ignore[attr-defined]

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    result = fresh.load()
    root_nodes = [n for n in result.nodes if n.parent_id is None]
    assert len(root_nodes) == 1, "root must be unique across restarts"


# ----- Per-trial mirror -----------------------------------------------


def test_every_trial_produces_a_dag_node(tmp_path: Path) -> None:
    factory, node_store = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas({}, {}, {}),
        objectives=[100.0, 90.0, 80.0],
        budget=BudgetConfig(max_trials=3, budget_seconds=999.0),
    )
    factory.scheduler.run()  # type: ignore[attr-defined]

    fresh = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    nodes = fresh.load().nodes
    # 1 root + 3 trials = 4 total.
    assert len(nodes) == 4
    launched = [n for n in nodes if n.parent_id is not None]
    assert len(launched) == 3
    # Linear chain under Milestone-1 semantics: each trial's parent is
    # the previous node (root -> trial0 -> trial1 -> trial2).
    parent_chain = [n.parent_id for n in launched]
    prior_ids = [nodes[i].node_id for i in range(len(nodes) - 1)]
    assert parent_chain == prior_ids


def test_dag_node_fields_match_ledger_entry(tmp_path: Path) -> None:
    factory, node_store = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas(
            {"actor.micro_batch_size": 32},
            {"actor.micro_batch_size": 64},
        ),
        objectives=[100.0, 90.0],
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
    )
    factory.scheduler.run()  # type: ignore[attr-defined]

    fresh_ledger = Ledger(factory.scheduler.ledger.path, fsync_on_append=False)  # type: ignore[attr-defined]
    ledger_entries = fresh_ledger.load().entries
    fresh_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    dag_nodes = [n for n in fresh_store.load().nodes if n.parent_id is not None]

    assert len(ledger_entries) == len(dag_nodes) == 2
    for entry, node in zip(ledger_entries, dag_nodes):
        # Semantic equivalences documented in AC-1 / AC-2:
        assert node.trial_idx == entry.trial_idx
        assert node.cumulative_delta == dict(entry.delta)
        assert node.delta_from_parent == dict(entry.proposed_delta or {})
        assert node.status == entry.status
        assert node.failure_mode == entry.failure_mode
        assert node.objective == entry.objective
        assert node.step_time == entry.step_time
        assert node.num_trajectories == entry.num_trajectories
        assert node.resolved_config_sha == entry.resolved_config_sha
        assert node.log_dir == entry.log_dir


def test_preflight_exhausted_trial_does_not_create_dag_node(tmp_path: Path) -> None:
    """Preflight rejection is NOT a launched trial and does NOT touch NodeStore."""
    factory, node_store = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas({}, {}, {}, {}, {}),
        objectives=[100.0],
        # Preflight budget consumed in order: (1) bootstrap call for the
        # baseline root, (2) trial 0's initial attempt, then (3-6) four
        # rejections for trial 1's retry loop (preflight_retries=3 gives
        # attempts 0..3, i.e. 4 total). Six entries in all.
        preflight_results=[True, True, False, False, False, False],
        budget=BudgetConfig(
            max_trials=5,
            budget_seconds=999.0,
            preflight_retries=3,
        ),
    )
    result = factory.scheduler.run()  # type: ignore[attr-defined]
    assert result.stop_reason == "preflight_exhausted"

    fresh_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    nodes = fresh_store.load().nodes
    # 1 root + 1 launched (successful) trial. The preflight-exhausted
    # attempt appended a synthetic CONFIG_INVALID entry to the Ledger
    # per AC-8, but that must NOT create a DAGNode (AC-5: preflight
    # rejections do not touch NodeStore).
    launched = [n for n in nodes if n.parent_id is not None]
    assert len(launched) == 1, (
        "preflight-exhausted attempt must not create a DAGNode; "
        f"got {[(n.node_id, n.failure_mode) for n in launched]}"
    )
    # And the ledger DOES record the synthetic entry (unchanged from
    # existing scheduler behaviour), so coexistence is asymmetric here
    # by design.
    fresh_ledger = Ledger(factory.scheduler.ledger.path, fsync_on_append=False)  # type: ignore[attr-defined]
    ledger_failure_modes = [e.failure_mode for e in fresh_ledger.load().entries]
    assert FailureMode.CONFIG_INVALID.value in ledger_failure_modes


def test_ledger_and_node_store_stay_in_sync_across_failed_trials(tmp_path: Path) -> None:
    """Failed launched trials (OOM here) still get mirrored to NodeStore."""
    factory, node_store = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas({}, {}, {}),
        objectives=[100.0, None, 80.0],
        failure_modes=[FailureMode.NONE, FailureMode.OOM, FailureMode.NONE],
        statuses=[Status.OK, Status.FAILED, Status.OK],
        budget=BudgetConfig(max_trials=3, budget_seconds=999.0, max_oom=99),
    )
    factory.scheduler.run()  # type: ignore[attr-defined]

    fresh_ledger = Ledger(factory.scheduler.ledger.path, fsync_on_append=False)  # type: ignore[attr-defined]
    fresh_store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    ledger_entries = fresh_ledger.load().entries
    dag_launched = [n for n in fresh_store.load().nodes if n.parent_id is not None]
    assert len(ledger_entries) == len(dag_launched) == 3
    for entry, node in zip(ledger_entries, dag_launched):
        assert node.failure_mode == entry.failure_mode


# ----- AC-9 artefact contract: every campaign emits the four artefacts -


def test_clean_campaign_emits_bitter_lessons_file_even_without_lessons(
    tmp_path: Path,
) -> None:
    """AC-9: bitter_lessons.jsonl MUST exist after every campaign.

    Round-3 Codex review flagged: ``LessonBook.add()`` was the only
    write path, so a clean campaign whose FakeCritic never proposed a
    ``bitter_lesson`` finished with no ``bitter_lessons.jsonl`` on
    disk. That violates the AC-9 artefact contract ("`bitter_lessons
    .jsonl` continues to be emitted after every campaign") and would
    trip any resume-time consumer that assumes the file exists.
    ``LessonBook.ensure_file()``, invoked from scheduler startup,
    materialises an empty JSONL — the "no lessons yet" state — so
    the file is always observable.
    """
    factory, _ = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas({}, {}),
        objectives=[100.0, 90.0],
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
    )
    factory.scheduler.run()  # type: ignore[attr-defined]

    # LessonBook path is derived from the ledger's parent dir by
    # ``Scheduler._resolve_lesson_book`` — read it back from the
    # scheduler rather than hard-coding a filename convention.
    ledger_path = factory.scheduler.ledger.path  # type: ignore[attr-defined]
    lessons_path = ledger_path.parent / "bitter_lessons.jsonl"
    assert lessons_path.exists(), (
        "AC-9: bitter_lessons.jsonl must exist after every campaign"
    )
    # A clean campaign leaves the file present but empty. Any
    # future consumer (resume-time reload, packaging, log
    # collection) can therefore ``open()`` it unconditionally.
    assert lessons_path.read_bytes() == b""


def test_clean_campaign_emits_all_scheduler_owned_artefacts(
    tmp_path: Path,
) -> None:
    """AC-9: every scheduler-owned artefact is on disk after a clean campaign.

    Covers the DAG-native subset of the AC-9 artefact contract that
    the scheduler is responsible for (as opposed to the CLI-level
    ``best_config.yaml`` / ``best_trial.json`` / plot, covered by
    the legacy-ledger-compat test): ``<ledger>.jsonl``,
    ``nodes.jsonl``, ``bitter_lessons.jsonl``. Both authoritative
    stores are populated; the lesson file exists as an empty
    JSONL when no lesson was proposed.
    """
    factory, node_store = _build_with_node_store(
        tmp_path,
        FakeCritic.from_deltas({}, {}),
        objectives=[100.0, 90.0],
        budget=BudgetConfig(max_trials=2, budget_seconds=999.0),
    )
    factory.scheduler.run()  # type: ignore[attr-defined]

    ledger_path = factory.scheduler.ledger.path  # type: ignore[attr-defined]
    nodes_path = tmp_path / "nodes.jsonl"
    lessons_path = ledger_path.parent / "bitter_lessons.jsonl"
    assert ledger_path.is_file()
    assert nodes_path.is_file()
    assert lessons_path.is_file()
    # Structural checks: Ledger has both trials, NodeStore has root +
    # both trials, LessonBook file is present (empty for this
    # zero-lesson campaign).
    fresh_ledger = Ledger(ledger_path, fsync_on_append=False)
    fresh_store = NodeStore(nodes_path, fsync_on_append=False)
    assert len(fresh_ledger.load().entries) == 2
    node_result = fresh_store.load()
    assert len(node_result.nodes) == 3  # root + 2 launched
    assert node_result.skipped_lines == 0
    # Empty lesson file — a fully-clean campaign proposed no lessons.
    assert lessons_path.read_bytes() == b""
