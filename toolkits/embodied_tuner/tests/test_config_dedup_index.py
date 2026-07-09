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

"""Tests for :mod:`toolkits.embodied_tuner.config_dedup_index`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolkits.embodied_tuner.config_dedup_index import (
    ConfigDedupIndex,
    DedupEntry,
    DedupIndexSchemaError,
)
from toolkits.embodied_tuner.node_store import (
    DUPLICATE_OF_FAILURE_MODE,
    DAGNode,
    NodeStore,
    ROOT_STATUS,
)
from toolkits.embodied_tuner.tests.test_node_store import (
    _make_child,
    _make_root,
)


# ----- DedupEntry schema -----------------------------------------------


def test_entry_roundtrips_via_dict() -> None:
    entry = DedupEntry(
        resolved_config_sha="sha-abc",
        origin_node_id="n5",
        status="OK",
        failure_mode="NONE",
        objective=1.5,
    )
    assert DedupEntry.from_dict(entry.to_dict()) == entry


def test_entry_from_dict_rejects_missing_fields() -> None:
    with pytest.raises(DedupIndexSchemaError):
        DedupEntry.from_dict({"resolved_config_sha": "s"})


def test_is_ok() -> None:
    ok = DedupEntry(
        resolved_config_sha="s",
        origin_node_id="n",
        status="OK",
        failure_mode="NONE",
        objective=1.0,
    )
    assert ok.is_ok()

    failed = DedupEntry(
        resolved_config_sha="s",
        origin_node_id="n",
        status="FAILED",
        failure_mode="OOM",
        objective=None,
    )
    assert not failed.is_ok()

    dup = DedupEntry(
        resolved_config_sha="s",
        origin_node_id="n",
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
    )
    assert not dup.is_ok()

    ok_no_objective = DedupEntry(
        resolved_config_sha="s",
        origin_node_id="n",
        status="OK",
        failure_mode="NONE",
        objective=None,
    )
    assert not ok_no_objective.is_ok()


# ----- ConfigDedupIndex add + lookup -----------------------------------


def test_add_records_new_entry(tmp_path: Path) -> None:
    idx = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    node = _make_child("n1", "root", trial_idx=0, objective=1.0, sha="sha-1")
    inserted = idx.add(node=node)
    assert inserted is True
    found = idx.lookup("sha-1")
    assert found is not None
    assert found.origin_node_id == "n1"
    assert found.is_ok()


def test_add_is_first_write_wins(tmp_path: Path) -> None:
    idx = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    first = _make_child("n1", "root", trial_idx=0, sha="sha-shared")
    second = _make_child("n2", "root", trial_idx=1, sha="sha-shared")
    idx.add(node=first)
    inserted = idx.add(node=second)
    assert inserted is False
    assert idx.lookup("sha-shared").origin_node_id == "n1"


def test_add_skips_none_sha(tmp_path: Path) -> None:
    idx = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    orphan = DAGNode(
        node_id="n1",
        parent_id="root",
        delta_from_parent={},
        cumulative_delta={},
        trial_idx=0,
        resolved_config_sha=None,
        log_dir="",
        returncode=0,
        status="OK",
        failure_mode="NONE",
        objective=1.0,
        step_time=None,
        num_trajectories=None,
        per_component_timings={},
        timeline_summary=None,
        peak_gpu_mem=None,
        critic_rationale=None,
        ts_start=0.0,
        ts_end=0.0,
        cleanup_outcome="ok",
    )
    assert idx.add(node=orphan) is False


def test_add_skips_duplicate_of_source(tmp_path: Path) -> None:
    """A DUPLICATE_OF node must never become the origin for its SHA."""
    idx = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    dup = _make_child(
        "dup",
        "root",
        trial_idx=-1,
        sha="sha-x",
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
        duplicate_of_node_id="original",
    )
    assert idx.add(node=dup) is False
    assert idx.lookup("sha-x") is None


# ----- ConfigDedupIndex persistence ------------------------------------


def test_load_recovers_prior_entries(tmp_path: Path) -> None:
    path = tmp_path / "cdi.jsonl"
    a = ConfigDedupIndex(path, fsync_on_append=False)
    a.add(node=_make_child("n1", "root", trial_idx=0, sha="sha-1"))
    a.add(node=_make_child("n2", "root", trial_idx=1, sha="sha-2"))
    b = ConfigDedupIndex(path, fsync_on_append=False)
    b.load_or_rebuild()
    assert b.lookup("sha-1") is not None
    assert b.lookup("sha-2") is not None
    assert b.lookup("sha-missing") is None


def test_load_tolerates_corrupt_line(tmp_path: Path) -> None:
    path = tmp_path / "cdi.jsonl"
    idx = ConfigDedupIndex(path, fsync_on_append=False)
    idx.add(node=_make_child("n1", "root", trial_idx=0, sha="sha-1"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    idx.add(node=_make_child("n2", "root", trial_idx=1, sha="sha-2"))
    fresh = ConfigDedupIndex(path, fsync_on_append=False)
    fresh.load_or_rebuild()
    assert fresh.lookup("sha-1") is not None
    assert fresh.lookup("sha-2") is not None


def test_rebuild_from_node_store_when_sidecar_missing(tmp_path: Path) -> None:
    """A lost sidecar must be reconstructed from the authoritative NodeStore."""
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root(sha="sha-baseline"))
    store.append(_make_child("n1", "root", trial_idx=0, sha="sha-1", objective=1.0))
    store.append(_make_child("n2", "root", trial_idx=1, sha="sha-2", objective=2.0))
    # Note the sidecar path does NOT exist yet.
    idx = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    idx.load_or_rebuild(store)
    assert {e.origin_node_id for e in idx.all_entries()} == {"root", "n1", "n2"}


def test_rebuild_skips_duplicate_of_nodes(tmp_path: Path) -> None:
    """Rebuild must not point origin_node_id at a DUPLICATE_OF node."""
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root(sha="sha-baseline"))
    store.append(_make_child("n1", "root", trial_idx=0, sha="sha-shared", objective=1.0))
    dup = _make_child(
        "dup1",
        "root",
        trial_idx=-1,
        sha="sha-shared",
        status="OK",
        failure_mode=DUPLICATE_OF_FAILURE_MODE,
        objective=1.0,
        duplicate_of_node_id="n1",
    )
    store.append(dup)
    idx = ConfigDedupIndex(tmp_path / "cdi.jsonl", fsync_on_append=False)
    idx.load_or_rebuild(store)
    entry = idx.lookup("sha-shared")
    assert entry is not None
    assert entry.origin_node_id == "n1"  # NOT "dup1"


def test_rebuild_prefers_node_store_over_stale_sidecar(tmp_path: Path) -> None:
    """AC-6 / task6: NodeStore is authoritative; a stale sidecar row is ignored.

    Round-1 (Codex-flagged) contract: when both a sidecar and a
    NodeStore are supplied, the DAG NodeStore is the ground truth for
    which node originated a given resolved-config SHA. A hand-edited
    or partially rolled-back sidecar that claims a different origin
    for a SHA that NodeStore knows about must NOT be trusted. Without
    this, a semantic-sidecar-corruption fixture (parseable, but
    pointing at the wrong origin) would silently break duplicate-of
    back-references.
    """
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root(sha="sha-baseline"))
    # NodeStore says sha-1 was first attempted by n1.
    store.append(_make_child("n1", "root", trial_idx=0, sha="sha-1", objective=2.0))
    # Sidecar claims sha-1's origin is a non-existent n0 (stale from
    # an older campaign or hand-editing accident).
    path = tmp_path / "cdi.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "resolved_config_sha": "sha-1",
                    "origin_node_id": "n0",
                    "status": "OK",
                    "failure_mode": "NONE",
                    "objective": 1.0,
                }
            )
            + "\n"
        )
    idx = ConfigDedupIndex(path, fsync_on_append=False)
    idx.load_or_rebuild(store)
    entry = idx.lookup("sha-1")
    assert entry is not None
    # NodeStore wins — origin_node_id is n1 (the real launched node),
    # not n0 (the stale sidecar claim). Objective also reflects the
    # NodeStore payload, not the sidecar's stale copy.
    assert entry.origin_node_id == "n1"
    assert entry.objective == 2.0


def test_rebuild_backfills_sidecar_when_node_store_silent_for_sha(
    tmp_path: Path,
) -> None:
    """A sidecar row for a SHA absent from NodeStore is retained (defensive fallback)."""
    store = NodeStore(tmp_path / "nodes.jsonl", fsync_on_append=False)
    store.append(_make_root(sha="sha-baseline"))
    # Sidecar has a row for sha-orphan that NodeStore doesn't know
    # about. Keep it — the sidecar is the only source and losing it
    # would silently allow a re-launch of the orphan config.
    path = tmp_path / "cdi.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "resolved_config_sha": "sha-orphan",
                    "origin_node_id": "n_orphan",
                    "status": "OK",
                    "failure_mode": "NONE",
                    "objective": 3.5,
                }
            )
            + "\n"
        )
    idx = ConfigDedupIndex(path, fsync_on_append=False)
    idx.load_or_rebuild(store)
    entry = idx.lookup("sha-orphan")
    assert entry is not None
    assert entry.origin_node_id == "n_orphan"
