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

"""Persistent "bitter lessons" store for the embodied auto-tuner.

Trials that fail (OOM, WORKER_CRASH, TIMEOUT, CONFIG_INVALID) fall out
of the critic's rolling history window after ``budget.history_window``
subsequent rounds. Without a longer-lived memory the critic re-proposes
the same failing delta again — exactly the pattern the
``maniskill_ppo_openvla`` campaign showed with three OOMs on the same
``rollout.enable_offload=False`` move.

A :class:`BitterLesson` is the critic's own one-line write-up of that
failure. :class:`LessonBook` persists the accumulated lessons as an
append-only JSONL file alongside the ledger, deduplicates by
``(failure_mode, delta_signature)`` on insert, and caps the total to
:attr:`LessonBook.max_lessons` (LRU: oldest ``trial_idx`` evicted
first) so the critic prompt does not grow without bound.

Persistence rules mirror :mod:`toolkits.embodied_tuner.ledger`:

- One JSON object per line, ``fsync`` after each append so the store
  survives a mid-loop SIGKILL.
- A corrupted line does not invalidate subsequent lines: :meth:`load`
  returns whatever it can decode.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class LessonSchemaError(ValueError):
    """Raised when a :class:`BitterLesson` payload is missing required fields."""


_REQUIRED_FIELDS = (
    "trigger",
    "rule",
    "trial_idx",
    "failure_mode",
    "delta_signature",
)


@dataclass(frozen=True)
class BitterLesson:
    """One persistent lesson learned from a failing trial.

    Attributes:
        trigger: One-line description of what happened, in the critic's
            own words. E.g. ``"OOM after rollout.enable_offload=False at
            total_num_envs=8, actor.micro_batch_size=40"``.
        rule: Directive the critic committed to for future rounds. E.g.
            ``"Do not disable rollout offload while total_num_envs >= 8
            unless actor.micro_batch_size <= 20."``.
        trial_idx: Index of the FAILED trial that produced this lesson
            (not the trial that emitted it — the critic learns *from*
            trial N in trial N+1's response).
        failure_mode: The failure mode string from the ledger (``"OOM"``,
            ``"WORKER_CRASH"``, ``"TIMEOUT"``, ``"CONFIG_INVALID"``).
        delta_signature: Canonical JSON of the delta that failed, used
            as the dedup key. Two lessons with the same
            ``(failure_mode, delta_signature)`` are considered duplicates.
    """

    trigger: str
    rule: str
    trial_idx: int
    failure_mode: str
    delta_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BitterLesson:
        missing = [name for name in _REQUIRED_FIELDS if name not in raw]
        if missing:
            raise LessonSchemaError(
                f"bitter lesson is missing required fields: {missing}"
            )
        return cls(**{name: raw[name] for name in _REQUIRED_FIELDS})

    def dedup_key(self) -> tuple[str, str]:
        return (self.failure_mode, self.delta_signature)


def canonical_delta_signature(delta: Mapping[str, Any]) -> str:
    """Return the canonical JSON string used for lesson dedup.

    Keys are sorted so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` map
    to the same signature. Nested mappings (e.g. ``component_placement``)
    are also canonicalised because ``json.dumps(..., sort_keys=True)``
    recurses.
    """
    return json.dumps(dict(delta), sort_keys=True, default=_json_default)


@dataclass
class LessonBook:
    """Append-only, deduplicating store of :class:`BitterLesson` records.

    Attributes:
        path: JSONL file. Parent directories are created lazily on the
            first :meth:`add`.
        max_lessons: Hard cap on retained lessons. When adding would
            exceed the cap, the oldest lesson (by ``trial_idx``, tie-
            broken by insertion order) is dropped and the eviction is
            appended as a marker line so the on-disk file remains a
            complete audit trail. Defaults to 30 — enough to cover a
            long campaign without ballooning the critic prompt.
        fsync_on_append: When ``True`` (default) each append is
            ``fsync``-ed. Tests may disable this for speed.
    """

    path: Path
    max_lessons: int = 30
    fsync_on_append: bool = True
    _lessons: list[BitterLesson] = field(default_factory=list, init=False, repr=False)
    _dedup_keys: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    # ----- Public API -----------------------------------------------------

    def load(self) -> tuple[BitterLesson, ...]:
        """Populate the in-memory list from disk and return it.

        Called at scheduler startup so a resumed campaign inherits its
        prior lessons. Corrupt or schema-violating lines are counted and
        skipped, matching :meth:`Ledger.load`'s tolerance policy.
        """
        self._lessons.clear()
        self._dedup_keys.clear()
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if raw.get("__evicted__"):
                        # An eviction marker; ignore for the live set
                        # (the record it evicted is already absent above).
                        continue
                    try:
                        lesson = BitterLesson.from_dict(raw)
                    except LessonSchemaError:
                        continue
                    key = lesson.dedup_key()
                    if key in self._dedup_keys:
                        # Duplicate on disk (e.g. from an older build
                        # without dedup); collapse.
                        continue
                    self._lessons.append(lesson)
                    self._dedup_keys.add(key)
        self._loaded = True
        return tuple(self._lessons)

    def ensure_file(self) -> None:
        """Create parent directory and an empty JSONL file if missing.

        Called from scheduler startup so a clean campaign that emits no
        critic-proposed lessons still leaves ``bitter_lessons.jsonl``
        on disk. Downstream consumers (``AC-9`` artefact contract, any
        resume-time reload) can then always ``open()`` the file — an
        empty file is the valid "no lessons yet" representation, not
        a missing file. Idempotent: if the file already exists it is
        left untouched (no truncate, no touch-mtime).
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_bytes(b"")

    def all(self) -> tuple[BitterLesson, ...]:
        """Return the current in-memory lessons (loads on first call)."""
        if not self._loaded:
            self.load()
        return tuple(self._lessons)

    def add(self, lesson: BitterLesson) -> bool:
        """Add ``lesson`` if not a duplicate. Returns whether it was inserted.

        Duplicates (same ``(failure_mode, delta_signature)``) return
        ``False`` and do not write to disk. When adding pushes the store
        over :attr:`max_lessons`, the oldest lesson is evicted from
        memory and an ``{"__evicted__": true, ...}`` marker is appended
        to the file so the audit trail is complete.
        """
        if not self._loaded:
            self.load()
        key = lesson.dedup_key()
        if key in self._dedup_keys:
            return False
        self._lessons.append(lesson)
        self._dedup_keys.add(key)
        self._write_line(lesson.to_dict())
        while len(self._lessons) > self.max_lessons:
            evicted = self._lessons.pop(0)
            self._dedup_keys.discard(evicted.dedup_key())
            self._write_line(
                {
                    "__evicted__": True,
                    "trial_idx": evicted.trial_idx,
                    "failure_mode": evicted.failure_mode,
                    "delta_signature": evicted.delta_signature,
                }
            )
            _log.info(
                "LessonBook evicted lesson from trial %d (%s) — cap %d reached",
                evicted.trial_idx,
                evicted.failure_mode,
                self.max_lessons,
            )
        return True

    def extend(self, lessons: Iterable[BitterLesson]) -> int:
        """Add each lesson via :meth:`add`. Returns the number inserted."""
        return sum(1 for lesson in lessons if self.add(lesson))

    # ----- Internals -----------------------------------------------------

    def _write_line(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=True, default=_json_default)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            if self.fsync_on_append:
                os.fsync(fh.fileno())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")
