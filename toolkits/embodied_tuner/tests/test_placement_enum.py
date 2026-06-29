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

"""Unit tests for :mod:`toolkits.embodied_tuner.placement_enum`."""

from __future__ import annotations

import pytest

from toolkits.embodied_tuner.placement_enum import (
    PlacementParseError,
    PlacementSpec,
    enumerate_placements,
    is_legal_placement,
    parse_range_spec,
)


# ---------------------------------------------------------------------------
# parse_range_spec — positive
# ---------------------------------------------------------------------------


def test_parse_range_spec_single_id() -> None:
    assert parse_range_spec("3") == (3,)


def test_parse_range_spec_contiguous_range() -> None:
    assert parse_range_spec("0-7") == tuple(range(8))


def test_parse_range_spec_all_token() -> None:
    assert parse_range_spec("all") == tuple(range(8))
    assert parse_range_spec("ALL", num_gpus=4) == (0, 1, 2, 3)


def test_parse_range_spec_comma_separated() -> None:
    assert parse_range_spec("0,2,4-6") == (0, 2, 4, 5, 6)


def test_parse_range_spec_strips_whitespace() -> None:
    assert parse_range_spec("  0-3  ") == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# parse_range_spec — negative
# ---------------------------------------------------------------------------


def test_parse_range_spec_rejects_non_string() -> None:
    with pytest.raises(PlacementParseError):
        parse_range_spec(7)  # type: ignore[arg-type]


def test_parse_range_spec_rejects_empty() -> None:
    with pytest.raises(PlacementParseError):
        parse_range_spec("")


def test_parse_range_spec_rejects_inverted_range() -> None:
    with pytest.raises(PlacementParseError):
        parse_range_spec("3-1")


def test_parse_range_spec_rejects_non_integer() -> None:
    with pytest.raises(PlacementParseError):
        parse_range_spec("a-b")


def test_parse_range_spec_rejects_out_of_bounds() -> None:
    with pytest.raises(PlacementParseError):
        parse_range_spec("0-8", num_gpus=8)


def test_parse_range_spec_rejects_empty_segment() -> None:
    with pytest.raises(PlacementParseError):
        parse_range_spec("0,,2")


# ---------------------------------------------------------------------------
# is_legal_placement — positive
# ---------------------------------------------------------------------------


def test_is_legal_placement_disaggregated() -> None:
    ok, kind = is_legal_placement({"actor": "0-3", "env": "4-5", "rollout": "6-7"})
    assert ok and kind == "disaggregated"


def test_is_legal_placement_hybrid() -> None:
    ok, kind = is_legal_placement({"actor": "0-7", "env": "0-3", "rollout": "4-7"})
    assert ok and kind == "hybrid"


def test_is_legal_placement_collocated_via_all() -> None:
    ok, kind = is_legal_placement({"actor": "all", "env": "all", "rollout": "all"})
    assert ok and kind == "all"


def test_is_legal_placement_collocated_via_explicit_ranges() -> None:
    ok, kind = is_legal_placement({"actor": "0-7", "env": "0-7", "rollout": "0-7"})
    assert ok and kind == "collocated"


def test_is_legal_placement_accepts_placement_spec() -> None:
    spec = PlacementSpec(
        actor=tuple(range(8)), env=tuple(range(0, 4)), rollout=tuple(range(4, 8)), kind="hybrid"
    )
    ok, _ = is_legal_placement(spec)
    assert ok


# ---------------------------------------------------------------------------
# is_legal_placement — negative
# ---------------------------------------------------------------------------


def test_is_legal_placement_rejects_env_rollout_partial_overlap() -> None:
    ok, reason = is_legal_placement({"actor": "0-7", "env": "0-3", "rollout": "2-5"})
    assert not ok
    assert "overlap" in reason.lower()


def test_is_legal_placement_rejects_malformed_range() -> None:
    ok, reason = is_legal_placement({"actor": "0-7", "env": "3-1", "rollout": "4-7"})
    assert not ok
    assert "parse" in reason.lower() or "inverted" in reason.lower()


def test_is_legal_placement_rejects_non_contiguous_range() -> None:
    ok, reason = is_legal_placement(
        {"actor": "0,2,4-7", "env": "0-3", "rollout": "4-7"},
    )
    assert not ok
    assert "non-contiguous" in reason.lower() or "contiguous" in reason.lower()


def test_is_legal_placement_rejects_missing_component() -> None:
    ok, reason = is_legal_placement({"actor": "0-7", "env": "0-3"})  # no rollout
    assert not ok
    assert "rollout" in reason


def test_is_legal_placement_allows_actor_env_partial_overlap_when_env_rollout_disjoint() -> None:
    """The plan's hybrid example pattern: actor covers all GPUs; env and rollout split disjointly.

    actor and env (or actor and rollout) may overlap; only env and rollout
    are required to be either equal or disjoint.
    """
    ok, kind = is_legal_placement({"actor": "0-7", "env": "0-3", "rollout": "4-7"})
    assert ok and kind == "hybrid"


# ---------------------------------------------------------------------------
# enumerate_placements
# ---------------------------------------------------------------------------


def test_enumerate_placements_yields_each_pattern() -> None:
    specs = enumerate_placements(num_gpus=8)
    kinds = {s.kind for s in specs}
    assert {"all", "disaggregated", "hybrid"} <= kinds


def test_enumerate_placements_all_legal() -> None:
    for spec in enumerate_placements(num_gpus=8):
        ok, reason = is_legal_placement(spec)
        assert ok, f"enumerated spec {spec} reported illegal: {reason}"


def test_enumerate_placements_to_yaml_dict() -> None:
    specs = enumerate_placements(num_gpus=8)
    for spec in specs:
        yaml_dict = spec.to_yaml_dict()
        assert set(yaml_dict) == {"actor", "env", "rollout"}
        ok, _ = is_legal_placement(yaml_dict)
        assert ok


def test_enumerate_placements_rejects_tiny_num_gpus() -> None:
    with pytest.raises(ValueError):
        enumerate_placements(num_gpus=2)
