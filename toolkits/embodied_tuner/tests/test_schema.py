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

"""Unit tests for :mod:`toolkits.embodied_tuner.schema`."""

from __future__ import annotations

import pytest

from toolkits.embodied_tuner.schema import (
    KNOB_ACTOR_OFFLOAD,
    KNOB_ENV_OFFLOAD,
    KNOB_GLOBAL_BATCH_SIZE,
    KNOB_MICRO_BATCH_SIZE,
    KNOB_NUM_ACTION_CHUNKS,
    KNOB_PIPELINE_STAGE_NUM,
    KNOB_PLACEMENT,
    KNOB_ROLLOUT_EPOCH,
    KNOB_ROLLOUT_OFFLOAD,
    KNOB_TOTAL_NUM_ENVS,
    KnobDomain,
    KnobNotTunableError,
    KnobOutOfRangeError,
    KnobSchema,
    UnknownKnobError,
)


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


def test_list_knobs_returns_only_tunable_knobs() -> None:
    schema = KnobSchema()
    knobs = schema.list_knobs()
    assert set(knobs) == {
        KNOB_PLACEMENT,
        KNOB_TOTAL_NUM_ENVS,
        KNOB_ROLLOUT_EPOCH,
        KNOB_MICRO_BATCH_SIZE,
        KNOB_ENV_OFFLOAD,
        KNOB_ROLLOUT_OFFLOAD,
        KNOB_ACTOR_OFFLOAD,
    }


def test_list_pinned_knobs_returns_only_pinned_knobs() -> None:
    schema = KnobSchema()
    assert set(schema.list_pinned_knobs()) == {
        KNOB_GLOBAL_BATCH_SIZE,
        KNOB_PIPELINE_STAGE_NUM,
        KNOB_NUM_ACTION_CHUNKS,
    }


def test_validate_accepts_baseline_shaped_delta() -> None:
    schema = KnobSchema()
    # Mirrors the values used by maniskill_ppo_openvla.yaml.
    schema.validate(
        {
            KNOB_PLACEMENT: {"actor": "0-7", "env": "0-3", "rollout": "4-7"},
            KNOB_TOTAL_NUM_ENVS: 128,
            KNOB_ROLLOUT_EPOCH: 1,
            KNOB_MICRO_BATCH_SIZE: 80,
            KNOB_ENV_OFFLOAD: True,
            KNOB_ROLLOUT_OFFLOAD: True,
            KNOB_ACTOR_OFFLOAD: True,
        }
    )


def test_validate_accepts_empty_delta() -> None:
    KnobSchema().validate({})


def test_validate_accepts_placement_as_string() -> None:
    KnobSchema().validate({KNOB_PLACEMENT: "actor,env,rollout: 0-7"})


def test_validate_accepts_boundary_int_values() -> None:
    schema = KnobSchema()
    schema.validate({KNOB_MICRO_BATCH_SIZE: 1})
    schema.validate({KNOB_MICRO_BATCH_SIZE: 4096})
    schema.validate({KNOB_TOTAL_NUM_ENVS: 1})
    schema.validate({KNOB_ROLLOUT_EPOCH: 16})


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_validate_rejects_out_of_range_int() -> None:
    schema = KnobSchema()
    with pytest.raises(KnobOutOfRangeError) as exc:
        schema.validate({KNOB_MICRO_BATCH_SIZE: -1})
    msg = str(exc.value)
    assert KNOB_MICRO_BATCH_SIZE in msg
    assert "-1" in msg


def test_validate_rejects_int_above_upper_bound() -> None:
    schema = KnobSchema()
    with pytest.raises(KnobOutOfRangeError):
        schema.validate({KNOB_ROLLOUT_EPOCH: 9999})


def test_validate_rejects_pinned_knob() -> None:
    schema = KnobSchema()
    with pytest.raises(KnobNotTunableError) as exc:
        schema.validate({KNOB_GLOBAL_BATCH_SIZE: 1024})
    assert KNOB_GLOBAL_BATCH_SIZE in str(exc.value)


def test_validate_rejects_unknown_knob() -> None:
    schema = KnobSchema()
    with pytest.raises(UnknownKnobError):
        schema.validate({"nonexistent.knob": 7})


def test_validate_rejects_wrong_type_for_int_knob() -> None:
    schema = KnobSchema()
    with pytest.raises(KnobOutOfRangeError):
        schema.validate({KNOB_MICRO_BATCH_SIZE: "80"})


def test_validate_rejects_bool_for_int_knob() -> None:
    """``bool`` is a subclass of ``int`` in Python; the schema must catch this."""
    schema = KnobSchema()
    with pytest.raises(KnobOutOfRangeError):
        schema.validate({KNOB_MICRO_BATCH_SIZE: True})


def test_validate_rejects_non_bool_for_bool_knob() -> None:
    schema = KnobSchema()
    with pytest.raises(KnobOutOfRangeError):
        schema.validate({KNOB_ENV_OFFLOAD: 1})


def test_validate_rejects_non_string_non_mapping_placement() -> None:
    schema = KnobSchema()
    with pytest.raises(KnobOutOfRangeError):
        schema.validate({KNOB_PLACEMENT: 7})


def test_validate_raises_on_first_invalid_knob() -> None:
    """When multiple knobs are wrong, ``validate`` raises on the first failure."""
    schema = KnobSchema()
    with pytest.raises(KnobNotTunableError):
        schema.validate(
            {
                KNOB_GLOBAL_BATCH_SIZE: 1024,  # pinned
                "unknown.knob": 1,  # also wrong
            }
        )


# ---------------------------------------------------------------------------
# KnobDomain sanity
# ---------------------------------------------------------------------------


def test_knob_domain_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError):
        KnobDomain(knob="x", kind="float")


def test_knob_domain_int_requires_bounds() -> None:
    with pytest.raises(ValueError):
        KnobDomain(knob="x", kind="int")


def test_knob_domain_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        KnobDomain(knob="x", kind="int", min_value=10, max_value=1)


# ---------------------------------------------------------------------------
# Pinned-knob declarations (specific to the plan)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "knob",
    [KNOB_GLOBAL_BATCH_SIZE, KNOB_PIPELINE_STAGE_NUM, KNOB_NUM_ACTION_CHUNKS],
)
def test_plan_pinned_knobs_are_declared_pinned(knob: str) -> None:
    schema = KnobSchema()
    assert schema.domains[knob].pinned is True
