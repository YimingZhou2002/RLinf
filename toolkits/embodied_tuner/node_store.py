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

"""DAG-structured trial store for the embodied auto-tuner.

Persistence and lifecycle rules:

- One JSON object per line; nodes are appended only after their final
  ``(status, failure_mode)`` is known. No in-place status mutation, no
  ``RUNNING`` persistence — a node is written exactly once.
- ``append`` opens in ``"a"`` mode, writes the encoded line, and
  ``fsync``s before returning so a mid-loop SIGKILL cannot truncate the
  entry (mirrors the guarantee in :mod:`toolkits.embodied_tuner.ledger`).
- A corrupted line on disk does not invalidate subsequent entries.
- No semantic LRU eviction: the store retains every appended node
  indefinitely. Any in-memory index is a non-authoritative cache; every
  lookup falls back to the persisted log if needed.

The store coexists with the flat :class:`Ledger`. The scheduler writes
both on every trial. The ``NodeStore`` is authoritative for DAG state
(``parent_id``, ancestor walks, dedup rebuild, active-leaf derivation);
the ``Ledger`` remains the compatibility source for existing consumers
such as ``_emit_best_artefacts``, ``plot_step_time_vs_trajectories``,
and any historical ``tuner_ledger.jsonl`` from before the DAG landed.
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


# Failure modes that trigger active-leaf rollback to the failing node's
# parent. Includes every launched-trial failure mode (runtime and
# soft-metric alike). Preflight rejections (``CONFIG_INVALID``,
# ``DIVISIBILITY_VIOLATION``) are NOT included: those never create a
# NodeStore entry in the first place — they are proposal-validation
# failures handled by ``_propose_with_preflight``'s retry loop.
ROLLBACK_FAILURE_MODES: frozenset[str] = frozenset(
    {"OOM", "WORKER_CRASH", "TIMEOUT", "METRICS_PARTIAL", "METRICS_MISSING"}
)

# Sentinel used for the baseline root node's ``status``. Non-root nodes
# use the status strings produced by
# :class:`toolkits.embodied_tuner.parser.Status` (typically ``"OK"`` or
# ``"FAILED"``).
ROOT_STATUS: str = "ROOT"

# Sentinel used for the ``failure_mode`` field of duplicate-of nodes
# synthesised by the ``ConfigDedupIndex`` short-circuit. See AC-6.
DUPLICATE_OF_FAILURE_MODE: str = "DUPLICATE_OF"


class NodeStoreSchemaError(ValueError):
    """Raised when a :class:`DAGNode` payload is missing required fields."""


class NodeStoreIntegrityError(ValueError):
    """Raised on append if a node violates a store-level invariant.

    Concrete conditions:

    - The proposed node's ``node_id`` already exists in the store (no
      in-place update, no duplicate append).
    - The proposed node's ``parent_id`` does not resolve to a node in
      the store (dangling parent pointer).
    - Appending the node would introduce a cycle (its ``parent_id``
      transitively resolves to the node itself).
    - The node's ``status`` is non-terminal — persisted nodes must
      always carry a final status.
    - A second root node (``parent_id is None``) is proposed after a
      root already exists.
    """


# Required fields for :class:`DAGNode`. Kept close to
# :data:`toolkits.embodied_tuner.ledger._REQUIRED_FIELDS` so the
# coexistence writer can trivially cross-project the same source
# information into both stores.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "node_id",
    "parent_id",
    "delta_from_parent",
    "cumulative_delta",
    "trial_idx",
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

# Optional fields kept off :data:`_REQUIRED_FIELDS` for backward
# compatibility. Older on-disk lines may pre-date a field being added;
# ``from_dict`` accepts either presence or absence.
_OPTIONAL_FIELDS: tuple[str, ...] = (
    "error_excerpt",
    "duplicate_of_node_id",
    "memory_summary",
)

# Statuses that must never be persisted. Everything else (including any
# custom project-specific status) is accepted so long as it is a
# non-empty string.
_NON_TERMINAL_STATUSES: frozenset[str] = frozenset({"RUNNING", "IN_PROGRESS", ""})


@dataclass(frozen=True)
class DAGNode:
    """One node in the trial DAG, persisted as a single JSONL line.

    A node represents either the baseline root (``parent_id is None``
    and ``status == ROOT_STATUS``) or a completed trial reachable from
    the root by walking ``parent_id`` pointers. Every field that also
    lives on :class:`toolkits.embodied_tuner.ledger.LedgerEntry` carries
    the same meaning and type so the coexisting Ledger mirror stays
    trivially in sync.

    Attributes:
        node_id: Stable identifier for the node. Any string is legal so
            long as it is unique within the store. Callers typically use
            :func:`derive_node_id` for deterministic ids or supply a
            UUID/hash of their choice.
        parent_id: ``node_id`` of the parent node, or ``None`` for the
            root. Non-root nodes must reference a node already present
            in the store.
        delta_from_parent: The critic's incremental proposal that
            created this node (equivalent to
            :attr:`LedgerEntry.proposed_delta`). Empty ``{}`` for the
            root.
        cumulative_delta: The full override set applied to the baseline
            for this node (equivalent to :attr:`LedgerEntry.delta`).
            Empty ``{}`` for the root.
        trial_idx: The launched-trial index for correlation with the
            Ledger mirror, or ``None`` for the root (no runner was
            launched for it).
        duplicate_of_node_id: For a duplicate-of-OK synthetic node, the
            ``node_id`` of the ORIGINAL (non-duplicate) node that
            resolved to the same ``resolved_config_sha``. AC-6 forbids
            chains: this field always points to the true source, never
            to another duplicate.

    See :mod:`toolkits.embodied_tuner.ledger` for the meaning of the
    other fields, which mirror :class:`LedgerEntry`.
    """

    node_id: str
    parent_id: str | None
    delta_from_parent: Mapping[str, Any]
    cumulative_delta: Mapping[str, Any]
    trial_idx: int | None
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
    error_excerpt: str = ""
    # Full nvitop ``MemorySummary`` projection (per-GPU / per-process
    # breakdown + soft-pressure fields), mirroring
    # :attr:`LedgerEntry.memory_summary`. Optional for backward compat:
    # nodes persisted before this field was added load as ``None``. The
    # scheduler reverts the critic's ``last_memory_summary`` to the parent
    # node's value on a rollback failure, so the critic sees the
    # expand-from parent's memory — not the failed sibling's.
    memory_summary: Mapping[str, Any] | None = None
    duplicate_of_node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` suitable for ``json.dumps``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DAGNode:
        missing = [name for name in _REQUIRED_FIELDS if name not in raw]
        if missing:
            raise NodeStoreSchemaError(
                f"DAG node is missing required fields: {missing}"
            )
        kwargs: dict[str, Any] = {name: raw[name] for name in _REQUIRED_FIELDS}
        for name in _OPTIONAL_FIELDS:
            if name in raw:
                kwargs[name] = raw[name]
        return cls(**kwargs)

    def is_root(self) -> bool:
        return self.parent_id is None


@dataclass(frozen=True)
class LoadResult:
    """Outcome of :meth:`NodeStore.load`.

    Attributes:
        nodes: Successfully decoded nodes, in file order.
        skipped_lines: Count of lines that failed JSON decode or schema
            validation.
    """

    nodes: tuple[DAGNode, ...] = ()
    skipped_lines: int = 0


@dataclass
class NodeStore:
    """Append-only JSONL store of :class:`DAGNode` records.

    Unlike :class:`toolkits.embodied_tuner.lessons.LessonBook`, this
    store has NO LRU eviction: authoritative node retention is a
    correctness requirement (ancestor walk, dedup rebuild, active-leaf
    recovery would all break under a size cap). Any in-memory structure
    below is a non-authoritative cache — every lookup falls back to the
    persisted log if the cache does not know the answer.

    Instances are cheap to construct; call :meth:`load` (or any accessor
    that transitively calls it) to populate the in-memory index from
    disk. Repeated ``append`` calls maintain the index incrementally so
    a subsequent ``load`` is not required.

    Attributes:
        path: Absolute path of the ``nodes.jsonl`` file. Parent
            directories are created lazily on first ``append``.
        fsync_on_append: When ``True`` (the default) every append is
            ``fsync``-ed so the entry survives a process crash. Tests
            may set this ``False`` for speed; production should leave
            it on.
    """

    path: Path
    fsync_on_append: bool = True
    _by_id: dict[str, DAGNode] = field(default_factory=dict, init=False, repr=False)
    _children: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _insertion_order: list[str] = field(default_factory=list, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    # ----- Public API -----------------------------------------------------

    def append(self, node: DAGNode) -> None:
        """Persist ``node`` as a single JSONL line and update the index.

        Raises:
            NodeStoreSchemaError: The node fails eager ``from_dict``
                round-trip validation.
            NodeStoreIntegrityError: The node violates a store-level
                invariant. See :class:`NodeStoreIntegrityError` for the
                exact conditions.
        """
        # Validate schema eagerly so we never write half-formed records.
        DAGNode.from_dict(node.to_dict())
        if not self._loaded:
            self.load()
        self._validate_integrity(node)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(node.to_dict(), sort_keys=True, default=_json_default)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            if self.fsync_on_append:
                os.fsync(fh.fileno())

        # Update in-memory index only after the write has succeeded, so
        # a crash mid-append leaves the cache consistent with disk.
        self._index(node)

    def load(self) -> LoadResult:
        """Read every readable node from disk and rebuild the in-memory index.

        Called automatically on first access. Reloading a store that has
        already been mutated in memory (via :meth:`append`) is a full
        rebuild — the on-disk log is treated as authoritative.
        """
        self._by_id.clear()
        self._children.clear()
        self._insertion_order.clear()
        nodes: list[DAGNode] = []
        skipped = 0
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        node = DAGNode.from_dict(raw)
                    except (json.JSONDecodeError, NodeStoreSchemaError, TypeError) as exc:
                        skipped += 1
                        _log.warning(
                            "NodeStore load: skipping corrupt line %d of %s (%s)",
                            line_no,
                            self.path,
                            exc,
                        )
                        continue
                    # Enforce integrity on load too — a JSONL row whose
                    # ``parent_id`` does not resolve, or which cycles,
                    # or which is a duplicate ``node_id`` or persisted
                    # ``RUNNING`` line, is treated as corruption and
                    # skipped rather than poisoning the index.
                    try:
                        self._validate_integrity(node, on_load=True)
                    except NodeStoreIntegrityError as exc:
                        skipped += 1
                        _log.warning(
                            "NodeStore load: skipping corrupt line %d of %s "
                            "(node_id=%r, %s)",
                            line_no,
                            self.path,
                            node.node_id,
                            exc,
                        )
                        continue
                    self._index(node)
                    nodes.append(node)
        self._loaded = True
        return LoadResult(nodes=tuple(nodes), skipped_lines=skipped)

    def get(self, node_id: str) -> DAGNode | None:
        """Return the node with ``node_id``, or ``None`` if not present."""
        if not self._loaded:
            self.load()
        return self._by_id.get(node_id)

    def all_nodes(self) -> tuple[DAGNode, ...]:
        """Return every node in insertion order."""
        if not self._loaded:
            self.load()
        return tuple(self._by_id[nid] for nid in self._insertion_order)

    def root(self) -> DAGNode | None:
        """Return the root node (``parent_id is None``) or ``None`` if the store is empty."""
        if not self._loaded:
            self.load()
        for node_id in self._insertion_order:
            node = self._by_id[node_id]
            if node.is_root():
                return node
        return None

    def parent_of(self, node_id: str) -> DAGNode | None:
        """Return the parent of ``node_id`` or ``None`` if it is the root / missing."""
        node = self.get(node_id)
        if node is None or node.parent_id is None:
            return None
        return self.get(node.parent_id)

    def children_of(self, parent_id: str) -> tuple[DAGNode, ...]:
        """Return every direct child of ``parent_id`` in insertion order."""
        if not self._loaded:
            self.load()
        child_ids = self._children.get(parent_id, [])
        return tuple(self._by_id[cid] for cid in child_ids)

    def ancestors(self, node_id: str) -> tuple[DAGNode, ...]:
        """Return the root-to-``node_id`` chain in insertion (root-first) order.

        Returns an empty tuple if ``node_id`` is not present. The
        returned chain includes ``node_id`` itself as the last element.
        """
        node = self.get(node_id)
        if node is None:
            return ()
        chain: list[DAGNode] = [node]
        current = node
        visited = {node.node_id}
        while current.parent_id is not None:
            parent = self._by_id.get(current.parent_id)
            if parent is None or parent.node_id in visited:
                # Malformed chain; stop walking rather than raise. Load
                # would already have rejected true cycles, so hitting a
                # cycle here means the on-disk state was manipulated
                # after load — defensive.
                break
            chain.append(parent)
            visited.add(parent.node_id)
            current = parent
        chain.reverse()
        return tuple(chain)

    def best_ok_leaf(self) -> DAGNode | None:
        """Return the lowest-objective OK leaf, excluding DUPLICATE_OF.

        Mirrors :meth:`toolkits.embodied_tuner.ledger.Ledger.best`
        semantics: only nodes with ``status == "OK"``,
        ``failure_mode == "NONE"``, and a non-``None`` ``objective``
        qualify. DUPLICATE_OF nodes (which carry the original's
        objective but are not independent evidence) are excluded.
        """
        if not self._loaded:
            self.load()
        eligible = [
            node
            for node in self._by_id.values()
            if node.status == "OK"
            and node.failure_mode == "NONE"
            and node.objective is not None
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda n: (n.objective, n.log_dir))

    def active_leaf(self, max_siblings: int) -> str | None:
        """Reconstruct the active-leaf id by replaying rollback rules from disk.

        Convenience wrapper around :meth:`active_state` that returns only
        the active-leaf id. See :meth:`active_state` for the full replay
        semantics and the shared implementation.

        Args:
            max_siblings: Sibling cap consulted for the climb rule
                (matches :attr:`BudgetConfig.max_siblings`).

        Returns:
            The ``node_id`` of the active leaf, or ``None`` when the
            store is empty or the climb has walked above the root.
        """
        active, _sibling_failures = self.active_state(max_siblings)
        return active

    def active_state(self, max_siblings: int) -> tuple[str | None, int]:
        """Replay AC-5 rollback rules and return ``(active_id, sibling_failures)``.

        Walks every node in insertion order (starting from the root) and
        simulates the state machine described in AC-5:

        - Start at the root.
        - For each subsequent node ``n``:

          - If ``n.failure_mode`` is in :data:`ROLLBACK_FAILURE_MODES`,
            treat ``n`` as a rollback child. Increment the sibling
            counter at the current active parent; if it reaches
            ``max_siblings`` climb one level further (with the counter
            reset for the new active).
          - Otherwise (``n`` is OK, ROOT, or DUPLICATE_OF), advance the
            active leaf to ``n`` and reset the sibling counter.

        Duplicate-of-OK nodes are treated as advances because they
        carry the original OK trial's objective. Climb-above-root
        situations return ``(None, 0)`` — callers interpret that as
        ``rollback_exhausted``.

        This is the single source of truth used both by
        :meth:`active_leaf` and by ``Scheduler._reconstruct_state_from_stores``:
        the scheduler needs the sibling counter to preserve
        rollback-cap semantics across restart, not just the active id.

        Args:
            max_siblings: Sibling cap consulted for the climb rule
                (matches :attr:`BudgetConfig.max_siblings`).

        Returns:
            ``(active_id, sibling_failures)``. ``active_id`` is ``None``
            when the store is empty or a climb walked above the root;
            in that case ``sibling_failures`` is ``0`` (moot).
        """
        if max_siblings < 1:
            raise ValueError(
                f"max_siblings must be >= 1, got {max_siblings}"
            )
        if not self._loaded:
            self.load()
        root = self.root()
        if root is None:
            return None, 0
        active: str | None = root.node_id
        # Sibling failure counter is keyed by the CURRENT active id and
        # reset whenever the active leaf changes.
        sibling_failures = 0
        for node_id in self._insertion_order:
            node = self._by_id[node_id]
            if node.is_root():
                continue
            if node.failure_mode in ROLLBACK_FAILURE_MODES:
                # Rollback: walk to the failing node's parent.
                new_active: str | None = node.parent_id
                sibling_failures += 1
                if sibling_failures >= max_siblings:
                    # Sibling cap tripped: climb one more level.
                    grandparent = (
                        self._by_id.get(new_active).parent_id
                        if new_active is not None
                        and new_active in self._by_id
                        else None
                    )
                    new_active = grandparent
                    sibling_failures = 0
                active = new_active
            else:
                # OK / DUPLICATE_OF / any non-rollback: advance.
                active = node.node_id
                sibling_failures = 0
        if active is None:
            # Climb walked above the root during replay. The counter is
            # meaningless in this state; callers should treat this as
            # rollback_exhausted on the next failure.
            return None, 0
        return active, sibling_failures

    # ----- Internals -----------------------------------------------------

    def _validate_integrity(self, node: DAGNode, *, on_load: bool = False) -> None:
        """Check every store-level invariant that must hold for ``node``.

        Runs symmetrically on append and on load. The ``on_load`` flag is
        preserved as a hook for future load-only relaxations, but every
        invariant enforced on append (non-terminal status, duplicate id,
        parent-must-exist, no-cycle, single-root) is also enforced on
        load. A persisted ``RUNNING`` line or duplicate ``node_id`` on
        disk is therefore treated as corruption by :meth:`load` — the
        line is skipped rather than allowed to poison the authoritative
        cache used for restart-time active-leaf reconstruction.
        """
        del on_load  # retained for API stability; enforcement is symmetric
        if node.status in _NON_TERMINAL_STATUSES:
            raise NodeStoreIntegrityError(
                "cannot append node with non-terminal status "
                f"{node.status!r}; append only after final "
                "(status, failure_mode) is known"
            )
        if node.node_id in self._by_id:
            raise NodeStoreIntegrityError(
                f"duplicate node_id {node.node_id!r}: no in-place "
                "update; nodes are appended exactly once"
            )
        if node.parent_id is None:
            # Root node: check we do not already have one.
            for existing in self._by_id.values():
                if existing.is_root() and existing.node_id != node.node_id:
                    raise NodeStoreIntegrityError(
                        "cannot append a second root node "
                        f"({node.node_id!r}); existing root is "
                        f"{existing.node_id!r}"
                    )
            return
        # Non-root node: parent must exist and no cycle.
        if node.parent_id not in self._by_id:
            raise NodeStoreIntegrityError(
                f"node {node.node_id!r} references missing parent "
                f"{node.parent_id!r}"
            )
        # Cycle check by walking the proposed parent chain.
        seen = {node.node_id}
        current: DAGNode | None = self._by_id[node.parent_id]
        while current is not None:
            if current.node_id in seen:
                raise NodeStoreIntegrityError(
                    f"node {node.node_id!r} would introduce a cycle "
                    f"via parent chain reaching {current.node_id!r}"
                )
            seen.add(current.node_id)
            current = (
                self._by_id.get(current.parent_id)
                if current.parent_id is not None
                else None
            )

    def _index(self, node: DAGNode) -> None:
        """Insert ``node`` into the in-memory cache."""
        self._by_id[node.node_id] = node
        self._insertion_order.append(node.node_id)
        if node.parent_id is not None:
            self._children.setdefault(node.parent_id, []).append(node.node_id)


def derive_node_id(
    *,
    parent_id: str | None,
    delta_from_parent: Mapping[str, Any],
    trial_idx: int | None,
    ts_start: float,
) -> str:
    """Compute a deterministic, human-inspectable ``node_id`` for a new node.

    Format: ``"n<TRIAL>-<HEX8>"``, where ``<TRIAL>`` is the trial index
    (``root`` for the root, ``dup<idx>`` when ``trial_idx`` is negative
    which is the convention for synthetic DUPLICATE_OF entries) and
    ``<HEX8>`` is the first 8 hex characters of the SHA-256 over
    ``(parent_id, delta_from_parent, ts_start)``. Callers may supply
    their own ``node_id`` if a different scheme is preferred; this
    helper is offered for convenience only.
    """
    import hashlib

    payload = json.dumps(
        {
            "parent_id": parent_id,
            "delta_from_parent": dict(delta_from_parent),
            "ts_start": ts_start,
        },
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    if parent_id is None:
        prefix = "root"
    elif trial_idx is None:
        prefix = "unk"
    elif trial_idx < 0:
        prefix = f"dup{-trial_idx}"
    else:
        prefix = f"n{trial_idx}"
    return f"{prefix}-{digest}"


# ---------------------------------------------------------------------------
# DAG view rendering for the Codex prompt
# ---------------------------------------------------------------------------


def _fmt_delta(delta: Mapping[str, Any] | None) -> str:
    """Compact single-line JSON rendering of a delta for the DAG view."""
    if not delta:
        return "{}"
    return json.dumps(dict(delta), sort_keys=True, default=_json_default)


def _fmt_objective(objective: float | None) -> str:
    if objective is None:
        return "n/a"
    return f"{objective:.4g}"


def _fmt_node_line(node: DAGNode, *, marker: str = "") -> str:
    """Render one DAGNode as a single markdown-friendly line."""
    tag = "root" if node.is_root() else f"trial={node.trial_idx}"
    parts = [
        f"{marker}{node.node_id}",
        f"({tag})",
        f"status={node.status}",
        f"failure_mode={node.failure_mode}",
        f"objective={_fmt_objective(node.objective)}",
        f"delta_from_parent={_fmt_delta(node.delta_from_parent)}",
    ]
    if node.duplicate_of_node_id is not None:
        parts.append(f"duplicate_of={node.duplicate_of_node_id}")
    return "  ".join(parts)


def render_dag_view(
    node_store: NodeStore,
    *,
    active_leaf_id: str | None,
    max_dag_nodes: int = 30,
) -> str:
    """Return a compact markdown rendering of the search DAG for Codex.

    Layout (per AC-7):

    1. **Ancestor chain** — root → active_leaf. Always rendered in
       full; never truncated even when the chain exceeds
       ``max_dag_nodes``. Callers who want a hard cap should render
       an empty view (or set ``active_leaf_id`` to a shallow node).
    2. **Sibling attempts** — every other child of the active leaf's
       parent, in insertion order.
    3. **Top-K OK leaderboard** — OK, non-DUPLICATE_OF nodes sorted by
       objective ascending. Ties broken by insertion order.
    4. **Recent FAILED leaves** — nodes whose ``failure_mode`` is not
       ``NONE``/``DUPLICATE_OF``, ordered by recency (most recent
       first).
    5. **Recent duplicate config attempts** — DUPLICATE_OF nodes,
       ordered by recency. Ensures Codex sees repeated proposals of
       already-attempted configs even when the duplicate is neither
       an active ancestor nor a current-parent sibling. Each entry
       renders ``node_id``, ``duplicate_of_node_id``, ``objective``,
       and ``delta_from_parent``.

    The combined budget for sections 2 + 3 + 4 + 5 is ``max_dag_nodes``.
    A ``max_dag_nodes = 0`` renders only the ancestor chain plus a
    single-line note stating the leaderboard is empty; the block is
    still well-formed.

    The rendered block starts with a stable ``## Search DAG`` header
    so tests and downstream tooling can locate it precisely.
    """
    if max_dag_nodes < 0:
        raise ValueError(f"max_dag_nodes must be >= 0, got {max_dag_nodes}")

    lines: list[str] = ["## Search DAG"]

    root = node_store.root()
    all_nodes = node_store.all_nodes()
    if root is None:
        lines.append("(store empty — no root node yet)")
        return "\n".join(lines)

    # --- Section 1: ancestor chain -------------------------------------
    lines.append("")
    lines.append(f"### Active branch (root → active_leaf: {active_leaf_id or 'unknown'})")
    ancestors = node_store.ancestors(active_leaf_id) if active_leaf_id else ()
    if not ancestors and active_leaf_id is None:
        lines.append("(no active leaf; scheduler will expand from root)")
        ancestors = (root,)
    elif not ancestors:
        lines.append(
            f"(active leaf {active_leaf_id!r} not found in store; showing root only)"
        )
        ancestors = (root,)
    for i, node in enumerate(ancestors):
        marker = "-> " if node.node_id == active_leaf_id else "   "
        lines.append(f"{marker}{_fmt_node_line(node)}")

    # --- Section 2: sibling attempts at the active leaf's parent -------
    active_node = node_store.get(active_leaf_id) if active_leaf_id else None
    remaining_budget = max_dag_nodes
    lines.append("")
    lines.append("### Sibling attempts at parent")
    if active_node is None or active_node.parent_id is None:
        lines.append("(active leaf is the root — no siblings)")
    else:
        siblings = [
            n
            for n in node_store.children_of(active_node.parent_id)
            if n.node_id != active_node.node_id
        ]
        siblings = siblings[:remaining_budget]
        remaining_budget -= len(siblings)
        if not siblings:
            lines.append("(no other children at this parent)")
        else:
            for s in siblings:
                lines.append(_fmt_node_line(s))

    # --- Section 3: top-K OK leaderboard -------------------------------
    lines.append("")
    lines.append("### Top-K OK leaves (lowest objective first)")
    if remaining_budget <= 0:
        lines.append("(budget exhausted by sibling section)")
    else:
        ok_leaves = [
            n
            for n in all_nodes
            if n.status == "OK"
            and n.failure_mode == "NONE"
            and n.objective is not None
            and not n.is_root()
        ]
        ok_leaves.sort(key=lambda n: (n.objective, n.node_id))
        ok_leaves = ok_leaves[:remaining_budget]
        remaining_budget -= len(ok_leaves)
        if not ok_leaves:
            lines.append("(no OK leaves yet)")
        else:
            for n in ok_leaves:
                lines.append(_fmt_node_line(n))

    # --- Section 4: recent FAILED leaves (by recency) ------------------
    lines.append("")
    lines.append("### Recent failure leaves")
    if remaining_budget <= 0:
        lines.append("(budget exhausted by OK leaderboard)")
    else:
        failed_leaves = [
            n
            for n in reversed(all_nodes)  # recency-first
            if n.failure_mode not in {"NONE", DUPLICATE_OF_FAILURE_MODE}
            and not n.is_root()
        ]
        failed_leaves = failed_leaves[:remaining_budget]
        remaining_budget -= len(failed_leaves)
        if not failed_leaves:
            lines.append("(no failures observed yet)")
        else:
            for n in failed_leaves:
                lines.append(_fmt_node_line(n))

    # --- Section 5: recent DUPLICATE_OF attempts (by recency) ----------
    # Guarantees that Codex sees repeated proposals of already-attempted
    # configs even when the duplicate is neither an active ancestor nor
    # a current-parent sibling. DUPLICATE_OF remains excluded from the
    # top-K OK leaderboard (section 3) and the recent-failure section
    # (section 4); this dedicated section is the only place duplicates
    # are guaranteed to surface.
    lines.append("")
    lines.append("### Recent duplicate config attempts")
    if remaining_budget <= 0:
        lines.append("(budget exhausted by failure section)")
    else:
        duplicate_leaves = [
            n
            for n in reversed(all_nodes)  # recency-first
            if n.failure_mode == DUPLICATE_OF_FAILURE_MODE
        ]
        duplicate_leaves = duplicate_leaves[:remaining_budget]
        if not duplicate_leaves:
            lines.append("(no duplicate attempts yet)")
        else:
            for n in duplicate_leaves:
                lines.append(_fmt_node_line(n))

    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    """Fallback encoder for objects ``json`` doesn't natively support."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):  # enum-like
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")
