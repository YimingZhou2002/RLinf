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

"""Tunable knob schema for the embodied auto-tuner.

The schema declares the legal domain of each knob the LLM critic may
mutate, the pinned knobs whose tuning is deferred (see ``FUT-5`` in the
plan), and exposes a ``validate(delta)`` API used by preflight before any
real RLinf trial is launched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# Tunable knob identifiers (Hydra dotted paths) — the critic may mutate these.
KNOB_PLACEMENT = "cluster.component_placement"
KNOB_TOTAL_NUM_ENVS = "env.train.total_num_envs"
KNOB_ROLLOUT_EPOCH = "env.train.rollout_epoch"
KNOB_MICRO_BATCH_SIZE = "actor.micro_batch_size"
KNOB_ENV_OFFLOAD = "env.train.enable_offload"
KNOB_ROLLOUT_OFFLOAD = "rollout.enable_offload"
KNOB_ACTOR_OFFLOAD = "actor.enable_offload"

# Pinned knobs — listed in the schema so the validator rejects them with
# a clear ``KnobNotTunableError`` and so the FUT-5 un-pin path is a
# simple ``pinned=False`` flip.
KNOB_GLOBAL_BATCH_SIZE = "actor.global_batch_size"
KNOB_PIPELINE_STAGE_NUM = "rollout.pipeline_stage_num"
KNOB_NUM_ACTION_CHUNKS = "actor.model.num_action_chunks"


class KnobSchemaError(ValueError):
    """Base exception raised by :class:`KnobSchema` validation."""


class UnknownKnobError(KnobSchemaError):
    """Raised when a delta references a knob the schema does not declare."""


class KnobOutOfRangeError(KnobSchemaError):
    """Raised when a knob value is outside its declared legal domain."""


class KnobNotTunableError(KnobSchemaError):
    """Raised when a delta references a pinned (non-tunable) knob."""


_KnobValue = Any  # Knobs hold heterogeneous Python types (int, bool, str/dict).


@dataclass(frozen=True)
class KnobDomain:
    """Declared legal domain of a single knob.

    Attributes:
        knob: Hydra dotted path identifying the knob.
        kind: One of ``"int"``, ``"bool"``, ``"placement"``.
        pinned: ``True`` when the knob is reserved (not tunable in this loop).
        min_value: Inclusive lower bound for ``kind="int"`` knobs.
        max_value: Inclusive upper bound for ``kind="int"`` knobs.
        notes: Free-form note used in error messages.

    The schema deliberately keeps domain semantics simple: integers carry
    explicit ``min_value`` / ``max_value``; booleans rely on ``kind="bool"``;
    placement strings/dicts are checked structurally by an external
    placement-legality module (see ``placement_enum``). The schema does
    not attempt to enforce cross-knob divisibility — that is preflight's
    job (``preflight.compose_and_validate`` re-implements the targeted
    divisibility checks locally because ``rlinf.config.validate_cfg``
    would otherwise instantiate ``Cluster()``/``ray.init`` and break the
    "no GPU work" guarantee).
    """

    knob: str
    kind: str
    pinned: bool = False
    min_value: int | None = None
    max_value: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"int", "bool", "placement"}:
            raise ValueError(
                f"KnobDomain {self.knob!r}: invalid kind={self.kind!r}; "
                "expected one of int/bool/placement"
            )
        if self.kind == "int" and (self.min_value is None or self.max_value is None):
            raise ValueError(
                f"KnobDomain {self.knob!r}: int knobs require min_value and max_value"
            )
        if self.kind == "int" and self.min_value > self.max_value:
            raise ValueError(
                f"KnobDomain {self.knob!r}: min_value={self.min_value} > "
                f"max_value={self.max_value}"
            )


def _default_domains() -> dict[str, KnobDomain]:
    """Return the canonical knob-domain table used by :class:`KnobSchema`.

    Numeric bounds are deliberately wide: the goal is to catch typos and
    sign errors, not to encode workload-specific feasibility (preflight
    handles that). Bounds reflect the maniskill-class single-node 8xA800
    envelope cited in the plan.
    """
    return {
        # Placement is structural; legality is checked by placement_enum.
        KNOB_PLACEMENT: KnobDomain(
            knob=KNOB_PLACEMENT,
            kind="placement",
            notes="Structural placement string/dict; legality checked by placement_enum.",
        ),
        KNOB_TOTAL_NUM_ENVS: KnobDomain(
            knob=KNOB_TOTAL_NUM_ENVS,
            kind="int",
            min_value=1,
            max_value=4096,
            notes="Per-trial env count; divisibility against env_world_size handled by preflight.",
        ),
        KNOB_ROLLOUT_EPOCH: KnobDomain(
            knob=KNOB_ROLLOUT_EPOCH,
            kind="int",
            min_value=1,
            max_value=16,
            notes="Number of rollout epochs per training step.",
        ),
        KNOB_MICRO_BATCH_SIZE: KnobDomain(
            knob=KNOB_MICRO_BATCH_SIZE,
            kind="int",
            min_value=1,
            max_value=4096,
            notes="Actor micro batch; divisibility against global_batch_size handled by preflight.",
        ),
        KNOB_ENV_OFFLOAD: KnobDomain(knob=KNOB_ENV_OFFLOAD, kind="bool"),
        KNOB_ROLLOUT_OFFLOAD: KnobDomain(knob=KNOB_ROLLOUT_OFFLOAD, kind="bool"),
        KNOB_ACTOR_OFFLOAD: KnobDomain(knob=KNOB_ACTOR_OFFLOAD, kind="bool"),
        # Pinned knobs — reserved so the un-pin path is a single flag flip.
        KNOB_GLOBAL_BATCH_SIZE: KnobDomain(
            knob=KNOB_GLOBAL_BATCH_SIZE,
            kind="int",
            pinned=True,
            min_value=1,
            max_value=65_536,
            notes="Pinned in this loop; un-pinning is FUT-5.",
        ),
        KNOB_PIPELINE_STAGE_NUM: KnobDomain(
            knob=KNOB_PIPELINE_STAGE_NUM,
            kind="int",
            pinned=True,
            min_value=1,
            max_value=16,
            notes="Pinned in this loop; un-pinning is FUT-5.",
        ),
        KNOB_NUM_ACTION_CHUNKS: KnobDomain(
            knob=KNOB_NUM_ACTION_CHUNKS,
            kind="int",
            pinned=True,
            min_value=1,
            max_value=64,
            notes="Pinned in this loop; un-pinning is FUT-5.",
        ),
    }


@dataclass(frozen=True)
class KnobSchema:
    """Declarative schema of all tunable knobs the auto-tuner may mutate.

    The schema is the single source of truth consumed by the LLM critic
    prompt builder, the preflight validator, and the trial ledger.
    """

    domains: Mapping[str, KnobDomain] = field(default_factory=_default_domains)

    def list_knobs(self) -> list[str]:
        """Return the dotted-path names of knobs the critic is allowed to set."""
        return [name for name, dom in self.domains.items() if not dom.pinned]

    def list_pinned_knobs(self) -> list[str]:
        """Return the dotted-path names of knobs reserved (not tunable)."""
        return [name for name, dom in self.domains.items() if dom.pinned]

    def validate(self, delta: Mapping[str, _KnobValue]) -> None:
        """Validate ``delta`` against the declared schema.

        Args:
            delta: Mapping from Hydra dotted path to proposed value.

        Raises:
            UnknownKnobError: when a key is not declared in the schema.
            KnobNotTunableError: when a key is declared but pinned.
            KnobOutOfRangeError: when a value falls outside its declared
                ``min_value`` / ``max_value`` (int) or wrong type (bool).
        """
        for knob, value in delta.items():
            domain = self.domains.get(knob)
            if domain is None:
                raise UnknownKnobError(
                    f"unknown knob {knob!r}; not declared in KnobSchema"
                )
            if domain.pinned:
                raise KnobNotTunableError(
                    f"knob {knob!r} is pinned in this loop ({domain.notes or 'see FUT-5'})"
                )
            self._validate_value(domain, value)

    @staticmethod
    def _validate_value(domain: KnobDomain, value: _KnobValue) -> None:
        if domain.kind == "bool":
            if not isinstance(value, bool):
                raise KnobOutOfRangeError(
                    f"knob {domain.knob!r}: expected bool, got {type(value).__name__}"
                )
            return
        if domain.kind == "int":
            # Reject bool here because ``bool`` is a subclass of ``int`` in Python.
            if isinstance(value, bool) or not isinstance(value, int):
                raise KnobOutOfRangeError(
                    f"knob {domain.knob!r}: expected int, got {type(value).__name__}"
                )
            if value < domain.min_value or value > domain.max_value:
                raise KnobOutOfRangeError(
                    f"knob {domain.knob!r}: value {value} outside "
                    f"[{domain.min_value}, {domain.max_value}]"
                )
            return
        if domain.kind == "placement":
            # Structural check: must be a string or a mapping. Legality
            # (contiguity, non-overlap) is validated by placement_enum.
            if not isinstance(value, (str, Mapping)):
                raise KnobOutOfRangeError(
                    f"knob {domain.knob!r}: expected placement string or mapping, "
                    f"got {type(value).__name__}"
                )
            return
        raise AssertionError(f"unreachable: unknown kind {domain.kind!r}")
