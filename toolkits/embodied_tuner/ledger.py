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

"""Append-only JSONL ledger for embodied auto-tuner trials.

Persistence rules (per the plan's AC-9):

- One JSON object per line; ``append`` opens in ``"a"`` mode, writes the
  encoded line, and ``fsync``s before returning so a mid-loop SIGKILL
  cannot truncate the entry.
- A corrupted line on disk does not invalidate subsequent entries:
  :func:`Ledger.load` returns a :class:`LoadResult` carrying both the
  successfully decoded entries and the count of skipped lines.
- The structured ``critic_rationale`` (``{summary,
  metric_table_citations, timeline_citations}``) is persisted verbatim
  so an operator can grep the ledger to see WHICH MetricTable observation
  AND WHICH timeline observation drove each placement change — the
  audit trail the plan calls out under "Placement Decision Audit Trail".
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


class LedgerSchemaError(ValueError):
    """Raised when a :class:`LedgerEntry` is missing required fields."""


# Fields a :class:`LedgerEntry` MUST carry. ``critic_rationale`` and a
# handful of metric/timeline fields may be ``None`` for failed trials.
_REQUIRED_FIELDS = (
    "trial_idx",
    "delta",
    "resolved_config_sha",
    "log_dir",
    "returncode",
    "status",
    "failure_mode",
    "objective",
    "step_time",
    "num_trajectories",
    "per_component_timings",
    "timeline_summary",
    "peak_gpu_mem",
    "critic_rationale",
    "ts_start",
    "ts_end",
    "cleanup_outcome",
)


@dataclass(frozen=True)
class LedgerEntry:
    """One trial's record, persisted as a single JSONL line.

    Mirrors the AC-9 field list. ``critic_rationale`` is the structured
    ``{summary, metric_table_citations, timeline_citations}`` payload AC-7
    produces. ``per_component_timings`` holds the MetricTable Time-section
    keys (e.g. ``env/interact=275.4``). ``timeline_summary`` is the
    :class:`TimelineSummary`-shaped dict from the parser.
    """

    trial_idx: int
    delta: Mapping[str, Any]
    resolved_config_sha: str | None
    log_dir: str
    returncode: int | None
    status: str
    failure_mode: str
    objective: float | None
    step_time: float | None
    num_trajectories: int | None
    per_component_timings: Mapping[str, float]
    timeline_summary: Mapping[str, Any] | None
    peak_gpu_mem: float | None
    critic_rationale: Mapping[str, Any] | None
    ts_start: float
    ts_end: float
    cleanup_outcome: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` suitable for ``json.dumps``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LedgerEntry:
        missing = [name for name in _REQUIRED_FIELDS if name not in raw]
        if missing:
            raise LedgerSchemaError(
                f"ledger entry is missing required fields: {missing}"
            )
        return cls(**{name: raw[name] for name in _REQUIRED_FIELDS})


@dataclass(frozen=True)
class LoadResult:
    """Outcome of :meth:`Ledger.load`.

    Attributes:
        entries: Successfully decoded entries, in file order.
        skipped_lines: Count of lines that failed JSON decode or schema
            validation (a corrupted line, or a partial write from a
            crashed run).
    """

    entries: tuple[LedgerEntry, ...] = ()
    skipped_lines: int = 0


@dataclass(frozen=True)
class Ledger:
    """Append-only JSONL ledger persisted at :attr:`path`.

    The ledger is stateless apart from its file path; instances are
    cheap to construct and may be re-instantiated from a recovered path
    after a crash.

    Attributes:
        path: Absolute path of the ``.jsonl`` file. Parent directories
            are created lazily on first ``append``.
        fsync_on_append: When ``True`` (the default) every append is
            ``fsync``-ed so the entry survives a process crash. Tests
            may set this ``False`` for speed; production should leave
            it on.
    """

    path: Path
    fsync_on_append: bool = True

    # ----- Public API -----------------------------------------------------

    def append(self, entry: LedgerEntry) -> None:
        """Persist ``entry`` as a single JSONL line."""
        # Validate eagerly so we don't write half-formed records.
        LedgerEntry.from_dict(entry.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry.to_dict(), sort_keys=True, default=_json_default)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            if self.fsync_on_append:
                os.fsync(fh.fileno())

    def load(self) -> LoadResult:
        """Return every readable entry in file order plus a skipped count."""
        if not self.path.is_file():
            return LoadResult()
        entries: list[LedgerEntry] = []
        skipped = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    entries.append(LedgerEntry.from_dict(raw))
                except (json.JSONDecodeError, LedgerSchemaError, TypeError):
                    skipped += 1
                    continue
        return LoadResult(entries=tuple(entries), skipped_lines=skipped)

    def best(self) -> LedgerEntry | None:
        """Return the lowest-objective entry with ``(OK, NONE)`` semantics.

        Mirrors :func:`toolkits.embodied_tuner.parser.select_best`'s
        eligibility rule: only ``status == "OK"``, ``failure_mode == "NONE"``,
        and non-``None`` objective qualify.
        """
        eligible = [
            entry
            for entry in self.load().entries
            if entry.status == "OK"
            and entry.failure_mode == "NONE"
            and entry.objective is not None
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda e: (e.objective, e.log_dir))


def make_entry(
    *,
    trial_idx: int,
    delta: Mapping[str, Any],
    resolved_config_sha: str | None,
    log_dir: str | Path,
    returncode: int | None,
    status: str,
    failure_mode: str,
    objective: float | None,
    step_time: float | None,
    num_trajectories: int | None,
    per_component_timings: Mapping[str, float] | None = None,
    timeline_summary: Mapping[str, Any] | None = None,
    peak_gpu_mem: float | None = None,
    critic_rationale: Mapping[str, Any] | None = None,
    ts_start: float,
    ts_end: float,
    cleanup_outcome: str = "ok",
) -> LedgerEntry:
    """Construct a :class:`LedgerEntry` with sensible defaults for missing fields.

    Convenience helper used by the scheduler to assemble a ledger entry
    from the parser's :class:`TrialResult` plus the runner's
    :class:`TrialOutcome`.
    """
    return LedgerEntry(
        trial_idx=trial_idx,
        delta=dict(delta),
        resolved_config_sha=resolved_config_sha,
        log_dir=str(log_dir),
        returncode=returncode,
        status=status,
        failure_mode=failure_mode,
        objective=objective,
        step_time=step_time,
        num_trajectories=num_trajectories,
        per_component_timings=dict(per_component_timings or {}),
        timeline_summary=dict(timeline_summary) if timeline_summary is not None else None,
        peak_gpu_mem=peak_gpu_mem,
        critic_rationale=dict(critic_rationale) if critic_rationale is not None else None,
        ts_start=ts_start,
        ts_end=ts_end,
        cleanup_outcome=cleanup_outcome,
    )


def _json_default(obj: Any) -> Any:
    """Fallback encoder for objects ``json`` doesn't natively support."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):  # enum-like
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")
