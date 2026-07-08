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

"""Tests for ``toolkits.embodied_tuner.node_store``.

Mirrors the patterns in ``tests/test_ledger.py`` and
``tests/test_lessons.py``: round-trip, corruption tolerance, fsync,
schema enforcement, dedup / duplicate detection, ancestor walk, cycle
rejection, RUNNING-status rejection, no-LRU retention across many
nodes, and the ``active_leaf`` derivation that simulates the AC-5
rollback state machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from toolkits.embodied_tuner.node_store import (
    DUPLICATE_OF_FAILURE_MODE,
    DAGNode,
    NodeStore,
    NodeStoreIntegrityError,
    NodeStoreSchemaError,
    ROLLBACK_FAILURE_MODES,
    ROOT_STATUS,
    derive_node_id,
)


# ----- Test helpers -----------------------------------------------------


def _make_root(node_id: str = "root", *, sha: str = "sha-baseline") -> DAGNode:
    return DAGNode(
        node_id=node_id,
        parent_id=None,
        delta_from_parent={},
        cumulative_delta={},
        trial_idx=None,
        resolved_config_sha=sha,
        log_dir="",
        returncode=None,
        status=ROOT_STATUS,
        failure_mode="NONE",
        objective=None,
        step_time=None,
        num_trajectories=None,
        per_component_timings={},
        timeline_summary=None,
        peak_gpu_mem=None,
        critic_rationale=None,
        ts_start=1000.0,
        ts_end=1000.0,
        cleanup_outcome="ok",
    )


def _make_child(
    node_id: str,
    parent_id: str,
    *,
    trial_idx: int = 0,
    status: str = "OK",
    failure_mode: str = "NONE",
    objective: float | None = 1.0,
    delta_from_parent: dict[str, Any] | None = None,
    cumulative_delta: dict[str, Any] | None = None,
    sha: str | None = None,
    log_dir: str = "logs/trial-0",
    duplicate_of_node_id: str | None = None,
) -> DAGNode:
    return DAGNode(
        node_id=node_id,
        parent_id=parent_id,
        delta_from_parent=delta_from_parent or {"actor.micro_batch_size": 32},
        cumulative_delta=cumulative_delta or {"actor.micro_batch_size": 32},
        trial_idx=trial_idx,
        resolved_config_sha=sha or f"sha-{node_id}",
        log_dir=log_dir,
        returncode=0 if status == "OK" else 1,
        status=status,
        failure_mode=failure_mode,
        objective=objective,
        step_time=objective,
        num_trajectories=10,
        per_component_timings={},
        timeline_summary=None,
        peak_gpu_mem=None,
        critic_rationale=None,
        ts_start=2000.0 + trial_idx,
        ts_end=2100.0 + trial_idx,
        cleanup_outcome="ok",
        duplicate_of_node_id=duplicate_of_node_id,
    )


# ----- DAGNode schema tests -------------------------------------------


def test_dag_node_to_dict_roundtrips() -> None:
    node = _make_root()
    other = DAGNode.from_dict(node.to_dict())
    assert other == node


def test_dag_node_from_dict_rejects_missing_fields() -> None:
    payload = _make_root().to_dict()
    payload.pop("node_id")
    with pytest.raises(NodeStoreSchemaError) as exc:
        DAGNode.from_dict(payload)
    assert "node_id" in str(exc.value)


def test_dag_node_from_dict_accepts_missing_optional_fields() -> None:
    payload = _make_root().to_dict()
    payload.pop("duplicate_of_node_id", None)
    payload.pop("error_excerpt", None)
    node = DAGNode.from_dict(payload)
    assert node.duplicate_of_node_id is None
    assert node.error_excerpt == ""


def test_dag_node_is_root() -> None:
    assert _make_root().is_root()
    assert not _make_child("n1", "root").is_root()


# ----- NodeStore.append + load round-trip -----------------------------


def test_append_then_load_round_trips(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    root = _make_root()
    child = _make_child("n1", "root")
    store.append(root)
    store.append(child)

    reloaded = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    result = reloaded.load()
    assert result.skipped_lines == 0
    assert result.nodes == (root, child)


def test_append_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "nodes.jsonl"
    store = NodeStore(nested, fsync_on_append=False)
    store.append(_make_root())
    assert nested.is_file()


def test_load_returns_empty_when_path_missing(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    result = store.load()
    assert result == store.load()
    assert result.nodes == ()
    assert result.skipped_lines == 0


def test_load_tolerates_corrupted_line(tmp_path: Path) -> None:
    path = tmp_path / "nodes.jsonl"
    store = NodeStore(path, fsync_on_append=False)
    root = _make_root()
    child_a = _make_child("n1", "root", trial_idx=0)
    child_b = _make_child("n2", "root", trial_idx=1)
    store.append(root)
    store.append(child_a)
    # Inject a corrupt line between the two valid children.
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{ this is not json\n")
    store.append(child_b)

    reloaded = NodeStore(path, fsync_on_append=False)
    result = reloaded.load()
    assert result.skipped_lines == 1
    assert set(n.node_id for n in result.nodes) == {"root", "n1", "n2"}


def test_load_tolerates_schema_violating_line(tmp_path: Path) -> None:
    path = tmp_path / "nodes.jsonl"
    store = NodeStore(path, fsync_on_append=False)
    store.append(_make_root())
    with path.open("a", encoding="utf-8") as fh:
        # Well-formed JSON missing required fields.
        fh.write(json.dumps({"node_id": "orphan"}) + "\n")
    store.append(_make_child("n1", "root"))

    reloaded = NodeStore(path, fsync_on_append=False).load()
    assert reloaded.skipped_lines == 1
    assert [n.node_id for n in reloaded.nodes] == ["root", "n1"]


# ----- NodeStore.append integrity rules -------------------------------


def test_append_rejects_running_status(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    bad = _make_child("n1", "root", status="RUNNING", failure_mode="NONE")
    with pytest.raises(NodeStoreIntegrityError) as exc:
        store.append(bad)
    assert "non-terminal status" in str(exc.value)


def test_append_rejects_duplicate_node_id(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root"))
    with pytest.raises(NodeStoreIntegrityError) as exc:
        store.append(_make_child("n1", "root"))
    assert "duplicate node_id" in str(exc.value)


def test_append_rejects_missing_parent(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    orphan = _make_child("n1", "does-not-exist")
    with pytest.raises(NodeStoreIntegrityError) as exc:
        store.append(orphan)
    assert "missing parent" in str(exc.value)


def test_append_rejects_second_root(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    with pytest.raises(NodeStoreIntegrityError) as exc:
        store.append(_make_root("root-2"))
    assert "second root" in str(exc.value)


def test_load_rejects_cyclic_line(tmp_path: Path) -> None:
    # A cycle cannot be created via ``append`` (would require a node
    # whose parent references a node not yet in the store), so we
    # craft the on-disk state directly and verify ``load`` skips the
    # cyclic tail.
    path = tmp_path / "nodes.jsonl"
    # Two nodes that reference each other as parents; neither has a
    # valid parent chain to a root.
    a = DAGNode.from_dict(_make_child("a", "b").to_dict())
    b = DAGNode.from_dict(_make_child("b", "a").to_dict())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(a.to_dict(), sort_keys=True) + "\n")
        fh.write(json.dumps(b.to_dict(), sort_keys=True) + "\n")
    result = NodeStore(path, fsync_on_append=False).load()
    # Both rows fail the "parent must already exist" check on load, so
    # they are skipped rather than poisoning the index.
    assert result.skipped_lines == 2
    assert result.nodes == ()


# ----- Ancestor walk + children lookup --------------------------------


def test_ancestors_returns_root_to_leaf_chain(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    root = _make_root()
    n1 = _make_child("n1", "root", trial_idx=0)
    n2 = _make_child("n2", "n1", trial_idx=1)
    n3 = _make_child("n3", "n2", trial_idx=2)
    for node in (root, n1, n2, n3):
        store.append(node)
    chain = store.ancestors("n3")
    assert [n.node_id for n in chain] == ["root", "n1", "n2", "n3"]


def test_ancestors_returns_empty_for_missing_node(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    assert store.ancestors("does-not-exist") == ()


def test_children_of_returns_direct_children_in_order(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    root = _make_root()
    a = _make_child("a", "root", trial_idx=0)
    b = _make_child("b", "root", trial_idx=1)
    c = _make_child("c", "a", trial_idx=2)
    for n in (root, a, b, c):
        store.append(n)
    assert [n.node_id for n in store.children_of("root")] == ["a", "b"]
    assert [n.node_id for n in store.children_of("a")] == ["c"]
    assert store.children_of("b") == ()


def test_root_returns_root_or_none(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    assert store.root() is None
    root = _make_root()
    store.append(root)
    assert store.root() == root


def test_parent_of(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    root = _make_root()
    child = _make_child("n1", "root")
    store.append(root)
    store.append(child)
    assert store.parent_of("n1") == root
    assert store.parent_of("root") is None
    assert store.parent_of("does-not-exist") is None


# ----- No-LRU retention -----------------------------------------------


def test_no_lru_retention_across_many_nodes(tmp_path: Path) -> None:
    """Retain every appended node regardless of any in-memory cache limit.

    NodeStore has no semantic LRU. This test asserts that after
    appending far more nodes than any conceivable cache size, every id
    is still resolvable via ``get``, ``ancestors``, and
    ``children_of``.
    """
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    # Build a chain: root -> n0 -> n1 -> ... -> n999 (1000 non-root nodes)
    N = 1000
    prev = "root"
    for i in range(N):
        node = _make_child(f"n{i}", prev, trial_idx=i)
        store.append(node)
        prev = f"n{i}"
    # Spot-check retention across the full range.
    for i in (0, 1, 100, 500, 999):
        assert store.get(f"n{i}") is not None
    # Ancestor chain from the last node must still walk back to root.
    chain = store.ancestors(f"n{N - 1}")
    assert len(chain) == N + 1
    assert chain[0].node_id == "root"
    assert chain[-1].node_id == f"n{N - 1}"
    # Children lookup must still work at every step.
    for i in range(N - 1):
        kids = store.children_of(f"n{i}")
        assert len(kids) == 1 and kids[0].node_id == f"n{i + 1}"


# ----- best_ok_leaf ---------------------------------------------------


def test_best_ok_leaf_returns_lowest_objective_ok_node(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root", trial_idx=0, objective=2.0))
    store.append(_make_child("n2", "root", trial_idx=1, objective=1.5))
    store.append(_make_child("n3", "root", trial_idx=2, objective=3.0))
    best = store.best_ok_leaf()
    assert best is not None and best.node_id == "n2"


def test_best_ok_leaf_excludes_duplicate_of(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("orig", "root", trial_idx=0, objective=1.0))
    # A duplicate node copies the original's objective but must not be
    # eligible for best-selection.
    dup = _make_child(
        "dup",
        "root",
        trial_idx=1,
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
        duplicate_of_node_id="orig",
    )
    store.append(dup)
    best = store.best_ok_leaf()
    assert best is not None and best.node_id == "orig"


def test_best_ok_leaf_returns_none_when_no_eligible(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root", status="FAILED", failure_mode="OOM", objective=None))
    assert store.best_ok_leaf() is None


# ----- active_leaf derivation (AC-5) ----------------------------------


def test_active_leaf_empty_store_is_none(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    assert store.active_leaf(max_siblings=3) is None


def test_active_leaf_after_root_only_returns_root(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    assert store.active_leaf(max_siblings=3) == "root"


def test_active_leaf_after_ok_child_advances_to_child(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root", status="OK", failure_mode="NONE"))
    assert store.active_leaf(max_siblings=3) == "n1"


@pytest.mark.parametrize("failure_mode", sorted(ROLLBACK_FAILURE_MODES))
def test_active_leaf_after_rollback_failure_returns_parent(
    tmp_path: Path, failure_mode: str
) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(
        _make_child(
            "n1",
            "root",
            status="FAILED" if failure_mode not in {"METRICS_PARTIAL", "METRICS_MISSING"} else "OK",
            failure_mode=failure_mode,
            objective=None,
        )
    )
    # Any launched-trial failure mode rewinds to the parent (root here).
    assert store.active_leaf(max_siblings=3) == "root"


def test_active_leaf_after_soft_failure_treated_as_rollback(tmp_path: Path) -> None:
    """METRICS_PARTIAL / METRICS_MISSING trigger rollback just like OOM."""
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    store.append(
        _make_child(
            "n1",
            "root",
            status="OK",
            failure_mode="METRICS_PARTIAL",
            objective=None,
        )
    )
    assert store.active_leaf(max_siblings=3) == "root"
    store.append(
        _make_child(
            "n2",
            "root",
            status="OK",
            failure_mode="METRICS_MISSING",
            objective=None,
        )
    )
    assert store.active_leaf(max_siblings=3) == "root"


def test_active_leaf_after_sibling_cap_climbs_grandparent(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    # root -> a (OK, becomes active)
    store.append(_make_child("a", "root", status="OK", failure_mode="NONE"))
    assert store.active_leaf(max_siblings=3) == "a"
    # Two sibling failures at a: active stays at a until the cap trips.
    store.append(
        _make_child("a1", "a", status="FAILED", failure_mode="OOM", objective=None)
    )
    store.append(
        _make_child("a2", "a", status="FAILED", failure_mode="OOM", objective=None)
    )
    # After two failures, active is at a (parent of each failed child).
    assert store.active_leaf(max_siblings=3) == "a"
    # Third failure trips the cap: climb one more level, to root.
    store.append(
        _make_child("a3", "a", status="FAILED", failure_mode="OOM", objective=None)
    )
    assert store.active_leaf(max_siblings=3) == "root"


def test_active_leaf_duplicate_of_treated_as_advance(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    dup = _make_child(
        "dup",
        "root",
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
        duplicate_of_node_id="root",
    )
    store.append(dup)
    # DUPLICATE_OF is NOT in ROLLBACK_FAILURE_MODES, so it advances.
    assert store.active_leaf(max_siblings=3) == "dup"


def test_active_leaf_rejects_bad_max_siblings(tmp_path: Path) -> None:
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root())
    with pytest.raises(ValueError):
        store.active_leaf(max_siblings=0)


# ----- Persistence + reload -------------------------------------------


def test_load_recovers_prior_nodes(tmp_path: Path) -> None:
    path = tmp_path / "nodes.jsonl"
    a = NodeStore(path, fsync_on_append=False)
    a.append(_make_root())
    a.append(_make_child("n1", "root"))
    # Fresh instance must see the same nodes on load.
    b = NodeStore(path, fsync_on_append=False)
    result = b.load()
    assert [n.node_id for n in result.nodes] == ["root", "n1"]
    assert b.get("n1") is not None


def test_reload_does_not_double_count(tmp_path: Path) -> None:
    path = tmp_path / "nodes.jsonl"
    store = NodeStore(path, fsync_on_append=False)
    store.append(_make_root())
    store.append(_make_child("n1", "root"))
    # Explicit reload should reset the index to the on-disk state, not
    # accumulate duplicates.
    result = store.load()
    assert len(result.nodes) == 2
    assert len(store.all_nodes()) == 2


# ----- derive_node_id -------------------------------------------------


def test_derive_node_id_deterministic() -> None:
    a = derive_node_id(
        parent_id="root",
        delta_from_parent={"a": 1},
        trial_idx=5,
        ts_start=1.0,
    )
    b = derive_node_id(
        parent_id="root",
        delta_from_parent={"a": 1},
        trial_idx=5,
        ts_start=1.0,
    )
    assert a == b
    assert a.startswith("n5-")


def test_derive_node_id_prefix_for_root() -> None:
    node_id = derive_node_id(
        parent_id=None,
        delta_from_parent={},
        trial_idx=None,
        ts_start=1.0,
    )
    assert node_id.startswith("root-")


def test_derive_node_id_prefix_for_duplicate() -> None:
    node_id = derive_node_id(
        parent_id="root",
        delta_from_parent={"a": 1},
        trial_idx=-3,
        ts_start=1.0,
    )
    assert node_id.startswith("dup3-")
