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

"""Unit tests for :mod:`toolkits.embodied_tuner.lessons`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolkits.embodied_tuner.lessons import (
    BitterLesson,
    LessonBook,
    LessonSchemaError,
    canonical_delta_signature,
)


def _make_lesson(idx: int, mode: str = "OOM", delta: dict | None = None) -> BitterLesson:
    payload = delta if delta is not None else {"rollout.enable_offload": False}
    return BitterLesson(
        trigger=f"trial {idx} failed with {mode}",
        rule=f"avoid the delta from trial {idx}",
        trial_idx=idx,
        failure_mode=mode,
        delta_signature=canonical_delta_signature(payload),
    )


# ---------------------------------------------------------------------------
# canonical_delta_signature
# ---------------------------------------------------------------------------


def test_signature_key_order_independent() -> None:
    a = canonical_delta_signature({"a": 1, "b": 2})
    b = canonical_delta_signature({"b": 2, "a": 1})
    assert a == b


def test_signature_recurses_into_nested_mappings() -> None:
    a = canonical_delta_signature({"placement": {"actor": "0-3", "env": "4-7"}})
    b = canonical_delta_signature({"placement": {"env": "4-7", "actor": "0-3"}})
    assert a == b


# ---------------------------------------------------------------------------
# BitterLesson
# ---------------------------------------------------------------------------


def test_from_dict_rejects_missing_fields() -> None:
    with pytest.raises(LessonSchemaError):
        BitterLesson.from_dict({"trigger": "x", "rule": "y"})


def test_to_dict_roundtrips() -> None:
    lesson = _make_lesson(3)
    assert BitterLesson.from_dict(lesson.to_dict()) == lesson


# ---------------------------------------------------------------------------
# LessonBook append + dedup
# ---------------------------------------------------------------------------


def test_add_persists_lesson_and_returns_true(tmp_path: Path) -> None:
    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    assert book.add(_make_lesson(1)) is True
    assert len(book.all()) == 1
    with (tmp_path / "bitter_lessons.jsonl").open() as fh:
        line = fh.readline().strip()
    payload = json.loads(line)
    assert payload["trial_idx"] == 1
    assert payload["failure_mode"] == "OOM"


def test_add_deduplicates_by_failure_mode_and_delta_signature(tmp_path: Path) -> None:
    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    first = _make_lesson(1)
    duplicate = _make_lesson(11)  # same failure_mode + same delta
    assert book.add(first) is True
    assert book.add(duplicate) is False
    assert len(book.all()) == 1
    assert (tmp_path / "bitter_lessons.jsonl").read_text().count("\n") == 1


def test_add_different_failure_mode_is_not_duplicate(tmp_path: Path) -> None:
    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    book.add(_make_lesson(1, mode="OOM"))
    assert book.add(_make_lesson(2, mode="WORKER_CRASH")) is True
    assert len(book.all()) == 2


def test_add_different_delta_is_not_duplicate(tmp_path: Path) -> None:
    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl")
    book.add(_make_lesson(1, delta={"rollout.enable_offload": False}))
    assert (
        book.add(_make_lesson(2, delta={"actor.micro_batch_size": 160})) is True
    )
    assert len(book.all()) == 2


# ---------------------------------------------------------------------------
# LessonBook LRU cap
# ---------------------------------------------------------------------------


def test_add_evicts_oldest_when_cap_exceeded(tmp_path: Path) -> None:
    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl", max_lessons=2)
    book.add(_make_lesson(1, delta={"a": 1}))
    book.add(_make_lesson(2, delta={"b": 2}))
    book.add(_make_lesson(3, delta={"c": 3}))
    live = book.all()
    assert [l.trial_idx for l in live] == [2, 3]
    # An eviction marker was appended to the file (audit trail).
    lines = (tmp_path / "bitter_lessons.jsonl").read_text().strip().splitlines()
    assert any(json.loads(line).get("__evicted__") for line in lines)


def test_evicted_signature_can_be_re_added(tmp_path: Path) -> None:
    book = LessonBook(path=tmp_path / "bitter_lessons.jsonl", max_lessons=1)
    book.add(_make_lesson(1, delta={"a": 1}))
    book.add(_make_lesson(2, delta={"b": 2}))  # evicts trial 1
    # The trial-1 signature is now gone from the dedup set, so a fresh
    # lesson at the same signature can be re-added if the failure
    # recurs. (We do NOT want the LRU to silently swallow recurrences.)
    assert book.add(_make_lesson(3, delta={"a": 1})) is True


# ---------------------------------------------------------------------------
# LessonBook load + crash recovery
# ---------------------------------------------------------------------------


def test_load_from_empty_or_missing_path_returns_no_lessons(tmp_path: Path) -> None:
    book = LessonBook(path=tmp_path / "does-not-exist.jsonl")
    assert book.load() == ()


def test_load_recovers_prior_lessons(tmp_path: Path) -> None:
    path = tmp_path / "bitter_lessons.jsonl"
    first_book = LessonBook(path=path)
    first_book.add(_make_lesson(5))
    first_book.add(_make_lesson(6, mode="WORKER_CRASH", delta={"a": 1}))

    # Fresh instance simulating a scheduler restart.
    second_book = LessonBook(path=path)
    loaded = second_book.load()
    assert [l.trial_idx for l in loaded] == [5, 6]
    # And the dedup keys are populated so re-adds are still rejected.
    assert second_book.add(_make_lesson(99)) is False


def test_load_skips_corrupt_and_schema_violating_lines(tmp_path: Path) -> None:
    path = tmp_path / "bitter_lessons.jsonl"
    path.write_text(
        json.dumps(_make_lesson(1).to_dict())
        + "\nnot json\n"
        + json.dumps({"trigger": "x"})  # missing required fields
        + "\n"
        + json.dumps(_make_lesson(2, delta={"z": 9}).to_dict())
        + "\n"
    )
    book = LessonBook(path=path)
    loaded = book.load()
    assert [l.trial_idx for l in loaded] == [1, 2]
