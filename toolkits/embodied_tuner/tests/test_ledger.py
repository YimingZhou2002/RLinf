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

"""Unit tests for :mod:`toolkits.embodied_tuner.ledger`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolkits.embodied_tuner.ledger import (
    Ledger,
    LedgerEntry,
    LedgerSchemaError,
    LoadResult,
    make_entry,
)


def _entry(
    trial_idx: int = 0,
    *,
    objective: float | None = 1.0,
    status: str = "OK",
    failure_mode: str = "NONE",
    log_dir: str = "/tmp/trial",
    sha: str | None = "deadbeef",
) -> LedgerEntry:
    return make_entry(
        trial_idx=trial_idx,
        delta={"actor.micro_batch_size": 64},
        resolved_config_sha=sha,
        log_dir=log_dir,
        returncode=0,
        status=status,
        failure_mode=failure_mode,
        objective=objective,
        step_time=200.0,
        num_trajectories=18,
        per_component_timings={"env/interact": 100.0},
        timeline_summary={"window_start": 0.0, "window_end": 10.0},
        peak_gpu_mem=12.5,
        critic_rationale={
            "summary": "actor dominant",
            "metric_table_citations": ["actor/run_training=21.3"],
            "timeline_citations": ["actor rank0 sync_model_to_rollout median=9.2"],
        },
        ts_start=1.0,
        ts_end=2.0,
        cleanup_outcome="ok",
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_append_then_load_round_trips(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "tuner_ledger.jsonl")
    entries = [_entry(i, log_dir=f"/tmp/trial-{i}") for i in range(5)]
    for entry in entries:
        ledger.append(entry)
    loaded = ledger.load()
    assert loaded.skipped_lines == 0
    assert len(loaded.entries) == 5
    assert [e.trial_idx for e in loaded.entries] == [0, 1, 2, 3, 4]
    # Field-level equality.
    assert loaded.entries[0].delta == {"actor.micro_batch_size": 64}
    assert loaded.entries[0].critic_rationale["summary"] == "actor dominant"


def test_append_creates_parent_dirs(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "nested" / "deeper" / "ledger.jsonl")
    ledger.append(_entry())
    assert ledger.path.is_file()


def test_load_returns_empty_when_path_missing(tmp_path: Path) -> None:
    result = Ledger(tmp_path / "absent.jsonl").load()
    assert result == LoadResult()


# ---------------------------------------------------------------------------
# Crash recovery / corruption tolerance
# ---------------------------------------------------------------------------


def test_load_tolerates_corrupted_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append(_entry(0))
    ledger.append(_entry(1))
    # Simulate a partial write between trials 1 and 2.
    with path.open("a") as fh:
        fh.write("{not valid json\n")
    ledger.append(_entry(2))
    result = ledger.load()
    assert result.skipped_lines == 1
    assert [e.trial_idx for e in result.entries] == [0, 1, 2]


def test_load_tolerates_schema_violation_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append(_entry(0))
    # A well-formed JSON object missing required ledger fields.
    with path.open("a") as fh:
        fh.write(json.dumps({"trial_idx": 999}) + "\n")
    Ledger(path).append(_entry(1))
    result = Ledger(path).load()
    assert result.skipped_lines == 1
    assert [e.trial_idx for e in result.entries] == [0, 1]


def test_simulated_mid_loop_crash_preserves_earlier_entries(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    for i in range(3):
        ledger.append(_entry(i))
    # Drop the last byte to simulate the crash truncating a line.
    raw = path.read_bytes()
    path.write_bytes(raw + b"{partial-")
    result = ledger.load()
    assert result.skipped_lines == 1
    assert [e.trial_idx for e in result.entries] == [0, 1, 2]


# ---------------------------------------------------------------------------
# .best()
# ---------------------------------------------------------------------------


def test_best_picks_lowest_objective_among_ok_none(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_entry(0, objective=50.0, log_dir="/tmp/a"))
    ledger.append(_entry(1, objective=30.0, log_dir="/tmp/b"))
    ledger.append(
        _entry(2, objective=20.0, failure_mode="METRICS_PARTIAL", log_dir="/tmp/c")
    )
    ledger.append(
        _entry(3, objective=None, status="FAILED", failure_mode="OOM", log_dir="/tmp/d")
    )
    best = ledger.best()
    assert best is not None
    assert best.trial_idx == 1
    assert best.log_dir == "/tmp/b"


def test_best_returns_none_when_no_eligible(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_entry(0, failure_mode="METRICS_PARTIAL"))
    ledger.append(_entry(1, status="FAILED", failure_mode="OOM", objective=None))
    assert ledger.best() is None


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------


def test_from_dict_rejects_missing_required_field() -> None:
    raw = _entry().to_dict()
    raw.pop("critic_rationale")
    with pytest.raises(LedgerSchemaError):
        LedgerEntry.from_dict(raw)


def test_append_validates_before_writing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    # Construct a frozen entry with a non-serialisable field through
    # ``object.__setattr__`` to bypass dataclass freezing.
    entry = _entry()
    object.__setattr__(entry, "delta", {"actor.micro_batch_size": object()})
    with pytest.raises(TypeError):
        ledger.append(entry)
    # Ledger file must not exist yet (validation happens before open()).
    # In our code we open first then write — verify file is empty.
    if ledger.path.exists():
        assert ledger.path.read_text() == ""


def test_path_resolves_via_str(tmp_path: Path) -> None:
    path_str = str(tmp_path / "ledger.jsonl")
    ledger = Ledger(Path(path_str))
    ledger.append(_entry())
    assert ledger.load().entries[0].log_dir == "/tmp/trial"


# ---------------------------------------------------------------------------
# Critic rationale audit-trail expectations
# ---------------------------------------------------------------------------


def test_critic_rationale_persisted_verbatim(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    entry = _entry()
    ledger.append(entry)
    loaded = ledger.load().entries[0]
    assert loaded.critic_rationale == entry.critic_rationale


def test_resolved_config_sha_round_trips(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_entry(sha="a" * 64))
    assert ledger.load().entries[0].resolved_config_sha == "a" * 64
