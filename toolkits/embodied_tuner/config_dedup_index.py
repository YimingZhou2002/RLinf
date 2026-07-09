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

"""Persistent config-dedup index for the embodied auto-tuner.

Keyed by the cumulative-config SHA-256 that :mod:`preflight` already
computes (:attr:`ValidationResult.resolved_config_sha`). The index is
consulted inside :meth:`Scheduler._propose_with_preflight` immediately
after preflight passes: if the proposal's SHA matches an entry, the
runner is short-circuited (duplicate-of-OK) or the proposal is rejected
via ``preflight_feedback`` (duplicate-of-FAILED).

Persistence rules mirror :mod:`toolkits.embodied_tuner.node_store`:

- One JSON object per line; append-only, fsync-per-write, corruption-
  tolerant load.
- **No LRU eviction.** Every previously observed cumulative config is
  retained; recall correctness is a hard requirement (otherwise Codex
  could re-launch a config that fell out of a size-capped cache).
- Rebuildable from the authoritative :class:`NodeStore` at scheduler
  startup, so a lost or corrupt sidecar file is a warning, not a
  campaign-level failure.
- Duplicates always back-reference the ORIGINAL non-duplicate node for
  a given SHA — chains of duplicates are forbidden (see AC-6).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from toolkits.embodied_tuner.node_store import (
    DAGNode,
    NodeStore,
)
from toolkits.embodied_tuner.parser import FailureMode

_log = logging.getLogger(__name__)


class DedupIndexSchemaError(ValueError):
    """Raised when a :class:`DedupEntry` payload is missing required fields."""


_REQUIRED_FIELDS: tuple[str, ...] = (
    "resolved_config_sha",
    "origin_node_id",
    "status",
    "failure_mode",
    "objective",
)


@dataclass(frozen=True)
class DedupEntry:
    """One row in the :class:`ConfigDedupIndex` sidecar.

    Attributes:
        resolved_config_sha: SHA-256 of the resolved (Hydra-composed)
            YAML for the ORIGINAL trial that first attempted this
            cumulative config. Serves as the dedup key.
        origin_node_id: ``node_id`` of the ORIGINAL non-duplicate
            :class:`DAGNode` — never the id of a subsequent duplicate.
            Follows the AC-6 rule: no chains of duplicates.
        status: Status string of the original trial (typically ``"OK"``
            or ``"FAILED"``). Determines the short-circuit branch
            taken on a subsequent duplicate proposal.
        failure_mode: Failure mode of the original trial. Used to
            surface a specific rejection message via
            ``preflight_feedback`` for duplicate-of-FAILED.
        objective: The original trial's objective, copied verbatim onto
            synthetic ``DUPLICATE_OF`` DAGNodes so downstream
            leaderboards keep working.
    """

    resolved_config_sha: str
    origin_node_id: str
    status: str
    failure_mode: str
    objective: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DedupEntry:
        missing = [name for name in _REQUIRED_FIELDS if name not in raw]
        if missing:
            raise DedupIndexSchemaError(
                f"dedup entry is missing required fields: {missing}"
            )
        return cls(**{name: raw[name] for name in _REQUIRED_FIELDS})

    def is_ok(self) -> bool:
        """True when the original trial was a clean OK (not FAILED, not DUPLICATE_OF)."""
        return (
            self.status == "OK"
            and self.failure_mode == FailureMode.NONE.value
            and self.objective is not None
        )


@dataclass
class ConfigDedupIndex:
    """Persistent JSONL sidecar keyed by ``resolved_config_sha``.

    Instances are cheap to construct; call :meth:`load_or_rebuild` (or
    any accessor that transitively calls it) to populate the in-memory
    map from disk. Duplicates on disk (e.g. from an older build without
    dedup) are collapsed to the first occurrence.

    Attributes:
        path: Absolute path of ``config_dedup_index.jsonl``. Parent
            directories are created lazily on first ``add``.
        fsync_on_append: When ``True`` (the default) each append is
            ``fsync``-ed. Tests may disable this for speed; production
            should leave it on.
    """

    path: Path
    fsync_on_append: bool = True
    _entries: dict[str, DedupEntry] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    # ----- Public API -----------------------------------------------------

    def load_or_rebuild(self, node_store: NodeStore | None = None) -> None:
        """Populate the in-memory map from disk; optionally rebuild from ``node_store``.

        Two paths, keyed on whether a :class:`NodeStore` is supplied:

        - **NodeStore-authoritative (``node_store is not None``, the
          production path):** The DAG store is the ground truth for
          "which trials happened, with what SHA, in what order". Build
          the in-memory dedup map from NodeStore first, first-write-
          wins on SHA (so ``origin_node_id`` always points at the
          ORIGINAL non-duplicate). Then scan the sidecar file for
          SHAs that NodeStore does not know about (defensive; sidecar
          should be a strict subset). For any SHA already in the map,
          the sidecar row is IGNORED — a stale sidecar cannot
          override the authoritative record. Disagreements are logged
          so operators can inspect corruption.
        - **Sidecar-only (``node_store is None``, backward-compat
          path):** Load whatever the sidecar file contains, collapsing
          duplicate rows to the first occurrence. Used by unit tests
          that don't wire in a NodeStore; production always supplies
          one.

        Nodes with ``failure_mode == DUPLICATE_OF`` are SKIPPED during
        NodeStore rebuild so ``origin_node_id`` never points at a
        synthetic duplicate.

        See :attr:`ConfigDedupIndex` docstring for the persistence
        contract and AC-6 (chain-prevention) for the invariant.
        """
        self._entries.clear()

        if node_store is not None:
            # 1. NodeStore first — authoritative. Insertion order
            #    naturally implements first-write-wins on SHA.
            for node in node_store.all_nodes():
                entry = self._maybe_entry_from_node(node)
                if entry is None:
                    continue
                # setdefault so the first launched trial for this SHA
                # wins (matches AC-6's "back-reference to ORIGINAL
                # non-duplicate" rule).
                self._entries.setdefault(entry.resolved_config_sha, entry)

            # 2. Sidecar fallback for SHAs NodeStore doesn't know
            #    about. For SHAs already claimed by NodeStore, IGNORE
            #    the sidecar row — a stale or hand-edited sidecar
            #    must never override the authoritative DAG record.
            if self.path.is_file():
                with self.path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                            entry = DedupEntry.from_dict(raw)
                        except (json.JSONDecodeError, DedupIndexSchemaError, TypeError) as exc:
                            _log.warning(
                                "config_dedup_index: skipping corrupt line: %s", exc
                            )
                            continue
                        existing = self._entries.get(entry.resolved_config_sha)
                        if existing is None:
                            self._entries[entry.resolved_config_sha] = entry
                        elif existing.origin_node_id != entry.origin_node_id:
                            _log.warning(
                                "config_dedup_index: sidecar row for SHA %s "
                                "disagrees with NodeStore (sidecar_origin=%s, "
                                "node_store_origin=%s); NodeStore wins",
                                entry.resolved_config_sha,
                                entry.origin_node_id,
                                existing.origin_node_id,
                            )
            self._loaded = True
            return

        # Sidecar-only path (backward compat). No NodeStore to consult.
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        entry = DedupEntry.from_dict(raw)
                    except (json.JSONDecodeError, DedupIndexSchemaError, TypeError) as exc:
                        _log.warning(
                            "config_dedup_index: skipping corrupt line: %s", exc
                        )
                        continue
                    self._entries.setdefault(entry.resolved_config_sha, entry)
        self._loaded = True

    def _maybe_entry_from_node(self, node: DAGNode) -> DedupEntry | None:
        """Project a DAGNode into a DedupEntry, or ``None`` if the node is not indexable."""
        if node.resolved_config_sha is None:
            return None
        if node.failure_mode == FailureMode.DUPLICATE_OF.value:
            return None
        if node.is_root():
            # Baseline root's SHA is real but the root itself is not a
            # launched trial. Surface it so a proposal that resolves
            # back to baseline short-circuits.
            return DedupEntry(
                resolved_config_sha=node.resolved_config_sha,
                origin_node_id=node.node_id,
                status="OK",
                failure_mode=FailureMode.NONE.value,
                objective=node.objective,
            )
        return DedupEntry(
            resolved_config_sha=node.resolved_config_sha,
            origin_node_id=node.node_id,
            status=node.status,
            failure_mode=node.failure_mode,
            objective=node.objective,
        )

    def lookup(self, resolved_config_sha: str) -> DedupEntry | None:
        """Return the entry for ``resolved_config_sha`` or ``None``."""
        if not self._loaded:
            self.load_or_rebuild()
        return self._entries.get(resolved_config_sha)

    def add(
        self,
        *,
        node: DAGNode,
    ) -> bool:
        """Record ``node`` as the origin for its ``resolved_config_sha``.

        No-op (returns ``False``) when:

        - ``node.resolved_config_sha`` is ``None`` (can't be keyed).
        - ``node.failure_mode == DUPLICATE_OF`` — synthetic duplicates
          must never become the origin (AC-6 chain-prevention rule).
        - The SHA is already present in the index (first-write wins).

        Otherwise appends the derived :class:`DedupEntry` to the JSONL
        file with fsync and updates the in-memory map. Returns whether
        the entry was newly inserted.
        """
        if not self._loaded:
            self.load_or_rebuild()
        sha = node.resolved_config_sha
        if sha is None:
            return False
        if node.failure_mode == FailureMode.DUPLICATE_OF.value:
            return False
        if sha in self._entries:
            return False
        entry = DedupEntry(
            resolved_config_sha=sha,
            origin_node_id=node.node_id,
            status=node.status,
            failure_mode=node.failure_mode,
            objective=node.objective,
        )
        self._entries[sha] = entry
        self._write_line(entry.to_dict())
        return True

    def all_entries(self) -> tuple[DedupEntry, ...]:
        if not self._loaded:
            self.load_or_rebuild()
        return tuple(self._entries.values())

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
