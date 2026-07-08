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

"""Unit tests for :mod:`toolkits.embodied_tuner.preflight`.

These tests use the real ``maniskill_ppo_openvla.yaml`` baseline from
the RLinf repository so the assertions reflect actual Hydra composition
behaviour. No GPU work and no Ray is started.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from toolkits.embodied_tuner.preflight import (
    PreflightError,
    ValidationResult,
    compose_and_validate,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "examples" / "embodiment" / "config" / "maniskill_ppo_openvla.yaml"


# ---------------------------------------------------------------------------
# Infrastructure preconditions
# ---------------------------------------------------------------------------


def test_baseline_exists() -> None:
    assert BASELINE.is_file(), BASELINE


def test_missing_baseline_raises_preflight_error(tmp_path: Path) -> None:
    with pytest.raises(PreflightError):
        compose_and_validate(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# Positive: known-legal delta
# ---------------------------------------------------------------------------


def test_baseline_alone_passes_preflight() -> None:
    """The committed baseline must pass preflight with an empty delta."""
    result = compose_and_validate(BASELINE, delta={})
    assert result.ok, f"baseline must pass preflight, got errors: {result.errors}"
    assert result.resolved_cfg is not None
    assert result.resolved_config_sha is not None
    assert len(result.resolved_config_sha) == 64  # SHA-256 hex


def test_legal_offload_flip_passes_preflight() -> None:
    """Flipping an offload flag is a no-op for divisibility — must still pass."""
    result = compose_and_validate(
        BASELINE,
        delta={"env.train.enable_offload": False},
    )
    assert result.ok, result.errors


def test_resolved_config_sha_is_stable_for_identical_inputs() -> None:
    a = compose_and_validate(BASELINE, delta={})
    b = compose_and_validate(BASELINE, delta={})
    assert a.resolved_config_sha == b.resolved_config_sha


def test_resolved_config_sha_changes_when_delta_changes() -> None:
    a = compose_and_validate(BASELINE, delta={})
    b = compose_and_validate(BASELINE, delta={"actor.micro_batch_size": 64})
    assert a.resolved_config_sha != b.resolved_config_sha


def test_placement_kind_reported_on_success() -> None:
    result = compose_and_validate(BASELINE)
    assert result.ok
    # Baseline has actor:0-7, env:0-3, rollout:4-7 (hybrid).
    assert result.placement_kind == "hybrid"


# ---------------------------------------------------------------------------
# Negative: divisibility violations
# ---------------------------------------------------------------------------


def test_micro_batch_size_violation_rejected() -> None:
    """``global_batch_size % (micro_batch_size * actor_world_size) != 0`` must fail."""
    # Baseline: actor.global_batch_size=640, actor_world_size=8 (actor:0-7).
    # 640 / 8 = 80. A micro_batch_size that doesn't divide 80 evenly should fail.
    # 81 doesn't divide 640/8=80 -> 640 % (81*8)=640 % 648 != 0.
    result = compose_and_validate(
        BASELINE,
        delta={"actor.micro_batch_size": 81},
    )
    assert not result.ok
    assert any("global_batch_size" in e for e in result.errors)


def test_total_num_envs_not_divisible_by_env_world_size_rejected() -> None:
    """``total_num_envs % env_world_size != 0`` must fail."""
    # env:0-3 -> env_world_size=4. 7 % 4 = 3, so it's a violation.
    result = compose_and_validate(
        BASELINE,
        delta={"env.train.total_num_envs": 7},
    )
    assert not result.ok
    assert any("total_num_envs" in e and "env_world_size" in e for e in result.errors)


def test_per_rank_envs_not_divisible_by_pipeline_stage_num_rejected() -> None:
    """``per_rank % pipeline_stage_num != 0`` must fail."""
    # env:0-3 -> env_world_size=4. baseline pipeline_stage_num=2.
    # total_num_envs=12 -> per_rank=3, 3 % 2 != 0.
    result = compose_and_validate(
        BASELINE,
        delta={"env.train.total_num_envs": 12},
    )
    assert not result.ok
    assert any("pipeline_stage_num" in e for e in result.errors)


def test_routing_env_to_rollout_divisibility_rejected() -> None:
    """Reproduces the runtime crash from wiki §04.2.6.

    With ``env=0-1`` (env_world_size=2), ``rollout=2-7``
    (rollout_world_size=6), and baseline ``total_num_envs=128``, the
    per-env-rank batch is 64. 64 % 6 != 0, which trips
    ``CommMapper.get_dst_ranks`` at
    ``rlinf/scheduler/worker/routing.py:139`` and kills the trial. All
    other Tier-1 checks pass, so this is the check that must catch it.
    """
    result = compose_and_validate(
        BASELINE,
        delta={
            "cluster.component_placement": {
                "actor": "0-7",
                "env": "0-1",
                "rollout": "2-7",
            }
        },
    )
    assert not result.ok
    assert any(
        "rollout_world_size" in e and "routing.py" in e for e in result.errors
    ), result.errors


def test_routing_env_to_actor_divisibility_rejected() -> None:
    """A placement where ``per_rank`` divides ``rollout_world_size`` but not
    ``actor_world_size`` must still be rejected — the same routing
    assertion fires on the rollout→actor hop."""
    # env=0-1 (2), rollout=4-7 (4), actor=0-2 (3). total_num_envs=48:
    # per_rank=24, 24 % 2 (pipeline_stage_num) == 0, 24 % 4 == 0
    # (rollout OK), but 24 % 3 == 0 too — need a case where actor
    # violates. Use actor=0-4 (5): 24 % 5 != 0.
    result = compose_and_validate(
        BASELINE,
        delta={
            "env.train.total_num_envs": 48,
            "cluster.component_placement": {
                "actor": "0-4",
                "env": "0-1",
                "rollout": "4-7",
            },
        },
    )
    assert not result.ok
    assert any(
        "actor_world_size" in e and "routing.py" in e for e in result.errors
    ), result.errors


def test_routing_divisibility_passes_when_fixed() -> None:
    """The suggested fix from wiki §04.2.6 (rollout=2-5, 4 GPUs) must pass."""
    result = compose_and_validate(
        BASELINE,
        delta={
            "cluster.component_placement": {
                "actor": "0-7",
                "env": "0-1",
                "rollout": "2-5",
            }
        },
    )
    # per_rank=64, 64%4==0 (rollout) and 64%8==0 (actor) and 64%2==0
    # (pipeline_stage_num) — all divisibility must pass.
    assert result.ok, result.errors


# ---------------------------------------------------------------------------
# Negative: placement
# ---------------------------------------------------------------------------


def test_illegal_placement_rejected() -> None:
    """env+rollout partial overlap must be caught by the legality check."""
    result = compose_and_validate(
        BASELINE,
        delta={
            "cluster.component_placement": {
                "actor": "0-7",
                "env": "0-3",
                "rollout": "2-5",
            }
        },
    )
    assert not result.ok
    assert any("placement" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Negative: schema
# ---------------------------------------------------------------------------


def test_pinned_knob_in_delta_rejected_by_schema() -> None:
    """Schema rejects a pinned knob BEFORE we attempt Hydra composition."""
    result = compose_and_validate(
        BASELINE,
        delta={"actor.global_batch_size": 1024},
    )
    assert not result.ok
    assert any("schema" in e for e in result.errors)


def test_unknown_knob_in_delta_rejected_by_schema() -> None:
    result = compose_and_validate(
        BASELINE,
        delta={"nonexistent.knob": 1},
    )
    assert not result.ok
    assert any("schema" in e for e in result.errors)


# ---------------------------------------------------------------------------
# ValidationResult contract
# ---------------------------------------------------------------------------


def test_validation_result_is_frozen() -> None:
    result = compose_and_validate(BASELINE)
    assert isinstance(result, ValidationResult)
    with pytest.raises(Exception):
        result.ok = False  # type: ignore[misc]
