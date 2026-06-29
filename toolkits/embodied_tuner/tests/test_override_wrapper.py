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

"""Unit tests for :mod:`toolkits.embodied_tuner.override_wrapper`.

Tests use the real ``examples/embodiment/`` layout under the RLinf repo;
they do NOT launch the train script. Hydra precedence is verified by
checking the override token order, which is the contract Hydra honours
(later override token wins).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from toolkits.embodied_tuner.override_wrapper import (
    LaunchSpec,
    OverrideWrapper,
    OverrideWrapperError,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def wrapper() -> OverrideWrapper:
    return OverrideWrapper.for_repo(REPO_ROOT)


def test_repo_layout_assumed_by_wrapper_exists(wrapper: OverrideWrapper) -> None:
    """Sanity-check the wrapper's assumed file layout against the live repo."""
    assert wrapper.train_script.is_file(), wrapper.train_script
    assert wrapper.config_dir.is_dir(), wrapper.config_dir
    assert (wrapper.config_dir / "maniskill_ppo_openvla.yaml").is_file()


# ---------------------------------------------------------------------------
# build_invocation — positive
# ---------------------------------------------------------------------------


def test_build_invocation_emits_expected_argv_skeleton(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        overrides=[],
        log_dir=tmp_path / "trial-1",
        trial_id="trial-1",
    )
    argv = spec.argv
    assert argv[1] == str(wrapper.train_script)
    assert argv[2] == "--config-path"
    assert argv[3] == f"{wrapper.config_dir}/"
    assert argv[4] == "--config-name"
    assert argv[5] == "maniskill_ppo_openvla"
    assert argv[6].startswith("runner.logger.log_path=")
    assert argv[6].endswith("trial-1")


def test_build_invocation_appends_user_overrides_after_log_path(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        overrides=["actor.micro_batch_size=64"],
        log_dir=tmp_path / "trial-2",
        trial_id="trial-2",
    )
    # User overrides must come AFTER the stock-script log_path injection
    # so Hydra precedence (later wins) lets user overrides win if they
    # ever collide with the baseline injection.
    log_path_idx = spec.argv.index(f"runner.logger.log_path={tmp_path / 'trial-2'}")
    mbs_idx = spec.argv.index("actor.micro_batch_size=64")
    assert mbs_idx > log_path_idx


def test_build_invocation_accepts_dict_overrides(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        overrides={"actor.micro_batch_size": 64, "env.train.enable_offload": True},
        log_dir=tmp_path / "trial-3",
        trial_id="trial-3",
    )
    assert "actor.micro_batch_size=64" in spec.argv
    # Booleans must be lowercased to match Hydra/OmegaConf conventions.
    assert "env.train.enable_offload=true" in spec.argv


def test_build_invocation_accepts_tuple_overrides(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        overrides=[("actor.micro_batch_size", 32), ("env.train.rollout_epoch", 2)],
        log_dir=tmp_path / "trial-4",
        trial_id="trial-4",
    )
    assert "actor.micro_batch_size=32" in spec.argv
    assert "env.train.rollout_epoch=2" in spec.argv


def test_build_invocation_preserves_override_order(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    overrides = [
        "actor.micro_batch_size=64",
        "actor.micro_batch_size=128",  # later override must win per Hydra
    ]
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        overrides=overrides,
        log_dir=tmp_path / "trial-5",
        trial_id="trial-5",
    )
    assert spec.user_overrides == ("actor.micro_batch_size=64", "actor.micro_batch_size=128")
    first = spec.argv.index("actor.micro_batch_size=64")
    second = spec.argv.index("actor.micro_batch_size=128")
    assert second > first


def test_build_invocation_returns_correct_log_dir(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    log_dir = tmp_path / "logs" / "trial-6"
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        log_dir=log_dir,
        trial_id="trial-6",
    )
    assert spec.log_dir == log_dir
    assert spec.config_name == "maniskill_ppo_openvla"


def test_build_invocation_sets_trial_id_env(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        log_dir=tmp_path / "trial-7",
        trial_id="trial-7",
    )
    assert spec.env["RLINF_TUNER_TRIAL_ID"] == "trial-7"


def test_build_invocation_sets_stock_env_defaults(
    wrapper: OverrideWrapper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unset to verify defaults are populated.
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("ROBOT_PLATFORM", raising=False)
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        log_dir=tmp_path / "trial-8",
        trial_id="trial-8",
    )
    assert spec.env["MUJOCO_GL"] == "egl"
    assert spec.env["ROBOT_PLATFORM"] == "LIBERO"
    assert str(wrapper.repo_path) in spec.env["PYTHONPATH"]


def test_build_invocation_extra_env_overrides_defaults(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        log_dir=tmp_path / "trial-9",
        trial_id="trial-9",
        extra_env={"ROBOT_PLATFORM": "ALOHA"},
    )
    assert spec.env["ROBOT_PLATFORM"] == "ALOHA"


def test_overrides_in_order_returns_baseline_then_user(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        overrides=["actor.micro_batch_size=64"],
        log_dir=tmp_path / "trial-10",
        trial_id="trial-10",
    )
    order = spec.overrides_in_order()
    assert order == (
        f"runner.logger.log_path={tmp_path / 'trial-10'}",
        "actor.micro_batch_size=64",
    )


# ---------------------------------------------------------------------------
# build_invocation — negative
# ---------------------------------------------------------------------------


def test_build_invocation_rejects_missing_train_script(tmp_path: Path) -> None:
    bogus = OverrideWrapper.for_repo(tmp_path)  # empty directory
    with pytest.raises(OverrideWrapperError):
        bogus.build_invocation(
            "maniskill_ppo_openvla",
            log_dir=tmp_path / "trial",
            trial_id="trial",
        )


def test_build_invocation_rejects_empty_trial_id(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    with pytest.raises(OverrideWrapperError):
        wrapper.build_invocation(
            "maniskill_ppo_openvla",
            log_dir=tmp_path / "trial",
            trial_id="",
        )


def test_build_invocation_rejects_override_without_equals(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    with pytest.raises(OverrideWrapperError):
        wrapper.build_invocation(
            "maniskill_ppo_openvla",
            overrides=["bare_token_without_equals"],
            log_dir=tmp_path / "trial",
            trial_id="t",
        )


def test_build_invocation_rejects_malformed_tuple(
    wrapper: OverrideWrapper, tmp_path: Path
) -> None:
    with pytest.raises(OverrideWrapperError):
        wrapper.build_invocation(
            "maniskill_ppo_openvla",
            overrides=[("only-one-element",)],  # type: ignore[list-item]
            log_dir=tmp_path / "trial",
            trial_id="t",
        )


# ---------------------------------------------------------------------------
# LaunchSpec contract
# ---------------------------------------------------------------------------


def test_launch_spec_is_frozen(wrapper: OverrideWrapper, tmp_path: Path) -> None:
    spec = wrapper.build_invocation(
        "maniskill_ppo_openvla",
        log_dir=tmp_path / "trial",
        trial_id="t",
    )
    assert isinstance(spec, LaunchSpec)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        spec.config_name = "other"  # type: ignore[misc]
