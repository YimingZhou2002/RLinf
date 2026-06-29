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

"""Legal placement enumeration for the embodied auto-tuner.

The enumerator generates `cluster.component_placement` candidates that
respect the contiguity and non-overlap rules enforced by
``rlinf.utils.placement.ModelParallelComponentPlacement`` (see the
``assert self._actor_gpus == list(range(...))`` and the ``_is_disaggregated``
checks in ``rlinf/utils/placement.py``).

Scope: single-node, 8-GPU enumeration. Multi-node placement is ``FUT-1``.

This module does not instantiate ``Cluster`` or any RLinf worker; it only
parses range strings and applies legality predicates. Preflight (AC-3)
calls into ``is_legal_placement`` to reject malformed user-supplied
placements before any GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Component names the placement YAML may reference. The toolkit currently
# tunes the three embodied workers; critic/inference are pinned to actor
# in this loop and thus do not need separate placements.
COMPONENT_NAMES = ("actor", "env", "rollout")


class PlacementParseError(ValueError):
    """Raised when a placement range string cannot be parsed."""


class PlacementLegalityError(ValueError):
    """Raised when a placement violates contiguity or non-overlap."""


@dataclass(frozen=True)
class PlacementSpec:
    """A concrete ``cluster.component_placement`` candidate.

    Attributes:
        actor: GPU ids assigned to the actor worker, ascending and contiguous.
        env: GPU ids assigned to the env worker, ascending and contiguous.
        rollout: GPU ids assigned to the rollout worker, ascending and contiguous.
        kind: Categorical descriptor used for telemetry / critic context;
            one of ``"collocated"``, ``"disaggregated"``, ``"hybrid"``,
            or ``"all"``.

    The triple-list form is the canonical internal representation. Use
    :meth:`to_yaml_dict` to obtain the dict form Hydra writes into
    ``cluster.component_placement``.
    """

    actor: tuple[int, ...]
    env: tuple[int, ...]
    rollout: tuple[int, ...]
    kind: str

    def to_yaml_dict(self) -> dict[str, str]:
        """Return the YAML-shaped mapping suitable for Hydra overrides."""
        return {
            "actor": _format_range(self.actor),
            "env": _format_range(self.env),
            "rollout": _format_range(self.rollout),
        }


# ---------------------------------------------------------------------------
# Range parsing
# ---------------------------------------------------------------------------


def parse_range_spec(spec: str, num_gpus: int = 8) -> tuple[int, ...]:
    """Parse a single-component GPU range string.

    Accepts ``"all"`` (expands to ``0..num_gpus-1``), single ids
    (``"3"``), and inclusive ranges (``"0-7"``). Comma-separated lists
    such as ``"0,2,4-7"`` are also parsed, but the resulting tuple is
    only useful when callers explicitly want to detect non-contiguity —
    legality checks reject non-contiguous results.

    Args:
        spec: Raw range string from YAML.
        num_gpus: Total GPU count on the node (used to expand ``"all"``).

    Returns:
        Sorted, deduplicated tuple of GPU ids.

    Raises:
        PlacementParseError: when ``spec`` cannot be parsed.
    """
    if not isinstance(spec, str):
        raise PlacementParseError(
            f"placement range must be a string, got {type(spec).__name__}"
        )
    text = spec.strip()
    if not text:
        raise PlacementParseError("placement range is empty")
    if text.lower() == "all":
        return tuple(range(num_gpus))

    ids: set[int] = set()
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            raise PlacementParseError(f"empty segment in placement range {spec!r}")
        if "-" in piece:
            low_str, _, high_str = piece.partition("-")
            try:
                low = int(low_str)
                high = int(high_str)
            except ValueError as exc:
                raise PlacementParseError(
                    f"non-integer bound in placement range {spec!r}: {piece!r}"
                ) from exc
            if low > high:
                raise PlacementParseError(
                    f"inverted range in placement {spec!r}: {piece!r} (low > high)"
                )
            ids.update(range(low, high + 1))
        else:
            try:
                ids.add(int(piece))
            except ValueError as exc:
                raise PlacementParseError(
                    f"non-integer id in placement range {spec!r}: {piece!r}"
                ) from exc

    if any(i < 0 or i >= num_gpus for i in ids):
        raise PlacementParseError(
            f"placement range {spec!r} contains ids outside [0, {num_gpus - 1}]"
        )
    return tuple(sorted(ids))


def _format_range(ids: tuple[int, ...]) -> str:
    """Format a contiguous id tuple as a YAML-friendly range string."""
    if not ids:
        raise PlacementLegalityError("empty placement range")
    if len(ids) == 1:
        return str(ids[0])
    return f"{ids[0]}-{ids[-1]}"


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


def _is_contiguous(ids: tuple[int, ...]) -> bool:
    if not ids:
        return False
    return list(ids) == list(range(ids[0], ids[-1] + 1))


def _classify(actor: tuple[int, ...], env: tuple[int, ...], rollout: tuple[int, ...]) -> str:
    if actor == env == rollout:
        if len(actor) > 0 and actor[0] == 0:
            return "collocated"
        return "collocated"
    actor_s = set(actor)
    env_s = set(env)
    rollout_s = set(rollout)
    if actor_s.isdisjoint(env_s) and actor_s.isdisjoint(rollout_s) and env_s.isdisjoint(rollout_s):
        return "disaggregated"
    return "hybrid"


def is_legal_placement(
    placement: Mapping[str, str] | PlacementSpec, num_gpus: int = 8
) -> tuple[bool, str]:
    """Return ``(is_legal, reason)`` for a candidate placement.

    The function mirrors the runtime checks performed by
    ``ModelParallelComponentPlacement`` (contiguous GPU ranges, all ids
    inside ``[0, num_gpus)``) plus the structural expectation from
    ``_is_disaggregated`` for the disaggregated case. It does NOT attempt
    to encode every RLinf placement edge case; it is a fast preflight
    filter, not a replacement for ``ModelParallelComponentPlacement``.

    Args:
        placement: Either a mapping ``{component: range_string}`` or a
            :class:`PlacementSpec`.
        num_gpus: Total GPU count on the node.

    Returns:
        Tuple ``(is_legal, reason)``. When ``is_legal`` is ``True`` the
        reason is the classified kind (``"collocated"``,
        ``"disaggregated"``, ``"hybrid"``, or ``"all"``).
    """
    if isinstance(placement, PlacementSpec):
        actor, env, rollout = placement.actor, placement.env, placement.rollout
    else:
        missing = [name for name in COMPONENT_NAMES if name not in placement]
        if missing:
            return False, f"missing components in placement: {missing}"
        try:
            actor = parse_range_spec(placement["actor"], num_gpus=num_gpus)
            env = parse_range_spec(placement["env"], num_gpus=num_gpus)
            rollout = parse_range_spec(placement["rollout"], num_gpus=num_gpus)
        except PlacementParseError as exc:
            return False, f"parse error: {exc}"

    for name, ids in (("actor", actor), ("env", env), ("rollout", rollout)):
        if not ids:
            return False, f"component {name!r} has empty GPU range"
        if not _is_contiguous(ids):
            return False, (
                f"component {name!r} GPU range {list(ids)} is non-contiguous; "
                "ModelParallelComponentPlacement requires contiguous ranges"
            )

    # Hybrid placements are allowed (e.g. actor:0-7, env:0-3, rollout:4-7
    # where env and rollout are disjoint but both overlap actor). Reject
    # PARTIAL overlap of env and rollout (one shares some but not all
    # GPUs with the other), which has no valid embodied semantics. Full
    # equality (collocated / "all") is fine.
    env_set, rollout_set = set(env), set(rollout)
    if env_set & rollout_set and env_set != rollout_set:
        return False, (
            f"env and rollout GPU ranges partially overlap: env={list(env)} "
            f"rollout={list(rollout)}; only full equality or full disjointness is allowed"
        )

    kind = _classify(actor, env, rollout)
    if isinstance(placement, Mapping) and str(placement.get("actor", "")).strip().lower() == "all" \
            and str(placement.get("env", "")).strip().lower() == "all" \
            and str(placement.get("rollout", "")).strip().lower() == "all":
        return True, "all"
    return True, kind


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_placements(num_gpus: int = 8) -> list[PlacementSpec]:
    """Return a curated set of legal placements for a single ``num_gpus`` node.

    The returned list always contains at least one collocated, one
    disaggregated, one hybrid, and one ``"all"`` placement. It is
    intentionally short — the LLM critic picks from it (or proposes a
    near-neighbour) rather than exploring an exhaustive combinatorial
    space. Larger sweeps belong to FUT-3 (BO / evolutionary proposer).
    """
    if num_gpus < 3:
        raise ValueError(
            f"enumerate_placements requires num_gpus >= 3 (got {num_gpus}); "
            "the three embodied workers need at least one GPU each."
        )
    all_gpus = tuple(range(num_gpus))
    specs: list[PlacementSpec] = []

    # Collocated: everyone shares all GPUs.
    specs.append(PlacementSpec(actor=all_gpus, env=all_gpus, rollout=all_gpus, kind="all"))

    # Disaggregated patterns: three disjoint contiguous ranges that cover [0, num_gpus).
    # Pick a couple of representative splits.
    quarter = max(num_gpus // 4, 1)
    half = num_gpus // 2
    disagg_partitions = [
        # actor:0..half-1, env:half..3*quarter-1, rollout:3*quarter..num_gpus-1
        (tuple(range(0, half)), tuple(range(half, half + quarter)), tuple(range(half + quarter, num_gpus))),
        # actor:0..quarter-1, env:quarter..half-1, rollout:half..num_gpus-1
        (tuple(range(0, quarter)), tuple(range(quarter, half)), tuple(range(half, num_gpus))),
    ]
    for actor, env, rollout in disagg_partitions:
        if not (actor and env and rollout):
            continue
        legal, _ = is_legal_placement({
            "actor": _format_range(actor),
            "env": _format_range(env),
            "rollout": _format_range(rollout),
        }, num_gpus=num_gpus)
        if legal:
            specs.append(PlacementSpec(actor=actor, env=env, rollout=rollout, kind="disaggregated"))

    # Hybrid: actor on all GPUs, env and rollout split disjointly.
    hybrid_partitions = [
        (all_gpus, tuple(range(0, half)), tuple(range(half, num_gpus))),
        (all_gpus, tuple(range(0, quarter)), tuple(range(quarter, num_gpus))),
    ]
    for actor, env, rollout in hybrid_partitions:
        legal, _ = is_legal_placement({
            "actor": _format_range(actor),
            "env": _format_range(env),
            "rollout": _format_range(rollout),
        }, num_gpus=num_gpus)
        if legal:
            specs.append(PlacementSpec(actor=actor, env=env, rollout=rollout, kind="hybrid"))

    return specs
