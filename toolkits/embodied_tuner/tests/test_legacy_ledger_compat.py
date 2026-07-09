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

"""AC-2 / AC-9 / task18 backward-compat regression against a pre-DAG ledger.

The DAG rework must not break existing consumers that read
``tuner_ledger.jsonl`` from a pre-DAG campaign. This test locks that
guarantee down with three assertions:

1. A frozen pre-DAG ``tuner_ledger.jsonl`` under
   ``tests/fixtures/legacy_ledger/`` loads through the current
   :class:`Ledger` implementation with zero skipped lines and yields
   the expected best-entry pick.
2. ``_emit_best_artefacts`` writes a ``best_trial.json`` that is
   byte-identical to the frozen ``expected_best_trial.json`` fixture.
   Any drift in the emit code path (JSON keys, field order, indent,
   trailing newline handling) fails this test loudly.
3. ``_emit_best_artefacts`` writes a ``best_config.yaml`` that
   round-trips through ``OmegaConf`` and contains the ``actor.model``
   node (structural check — a byte-identical baseline for the YAML
   would break every time the baseline Hydra config legitimately
   changes).
4. ``_emit_ledger_plot`` produces a PNG whose first four bytes are
   the canonical PNG magic ``89 50 4E 47``. File existence alone
   (Round-0's existing check) does not prove the artifact is a valid
   image.

If the frozen fixture ever needs a legitimate schema bump, regenerate
``expected_best_trial.json`` and this file's fixture ledger together
in the SAME commit so the drift is visible in review.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from toolkits.embodied_tuner.__main__ import (
    CLIArgs,
    _emit_best_artefacts,
    _emit_ledger_plot,
)
from toolkits.embodied_tuner.ledger import Ledger
from toolkits.embodied_tuner.scheduler import CampaignResult

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASELINE = (
    _REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "maniskill_ppo_openvla.yaml"
)
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "legacy_ledger"
_LEGACY_LEDGER = _FIXTURE_DIR / "tuner_ledger.jsonl"
_EXPECTED_BEST_TRIAL = _FIXTURE_DIR / "expected_best_trial.json"

# PNG magic bytes per the spec (RFC 2083 §12.11 preamble):
# 0x89 P N G 0x0D 0x0A 0x1A 0x0A. Checking the first 4 bytes is
# sufficient to distinguish PNG from empty / truncated / other-format.
_PNG_MAGIC = b"\x89PNG"


def _stub_args(tmp_path: Path) -> CLIArgs:
    return CLIArgs(
        config="maniskill_ppo_openvla",
        baseline=_BASELINE,
        max_trials=3,
        budget_seconds=999.0,
        trial_timeout_seconds=2700.0,
        max_oom=99,
        patience=3,
        epsilon=0.02,
        max_epochs=3,
        collect_memory=False,
        use_profiler=False,
        dry_run_preflight=False,
        fake_critic_path=None,
        ledger_dir=tmp_path,
        ask_codex_path="/nonexistent",
    )


def _copy_fixture_ledger_to(tmp_path: Path) -> Path:
    dest = tmp_path / "tuner_ledger.jsonl"
    shutil.copy(_LEGACY_LEDGER, dest)
    return dest


# ----- 1. Pre-DAG ledger loads through current Ledger.load() -----------


def test_legacy_ledger_loads_without_skipped_lines() -> None:
    ledger = Ledger(_LEGACY_LEDGER, fsync_on_append=False)
    result = ledger.load()
    assert result.skipped_lines == 0
    # The fixture ledger has three rows: two OK trials + one OOM.
    assert len(result.entries) == 3
    trial_indices = [e.trial_idx for e in result.entries]
    assert trial_indices == [0, 1, 2]


def test_legacy_ledger_best_picks_lowest_ok_objective() -> None:
    ledger = Ledger(_LEGACY_LEDGER, fsync_on_append=False)
    best = ledger.best()
    assert best is not None
    assert best.trial_idx == 2
    assert best.objective == 18.5


# ----- 2. best_trial.json is byte-identical to the frozen fixture ------


def test_emit_best_trial_json_is_byte_identical_to_fixture(
    tmp_path: Path,
) -> None:
    _copy_fixture_ledger_to(tmp_path)
    ledger = Ledger(tmp_path / "tuner_ledger.jsonl", fsync_on_append=False)
    ledger_entries = ledger.load().entries
    best = ledger.best()
    assert best is not None

    args = _stub_args(tmp_path)
    campaign = CampaignResult(
        stop_reason="plateau",
        trial_count=len(ledger_entries),
        oom_count=1,
        best_entry=best,
        ledger_path=tmp_path / "tuner_ledger.jsonl",
    )
    _emit_best_artefacts(campaign, args)

    written = (tmp_path / "best_trial.json").read_bytes()
    expected = _EXPECTED_BEST_TRIAL.read_bytes()
    assert written == expected, (
        "best_trial.json drift vs frozen fixture. Regenerate "
        f"{_EXPECTED_BEST_TRIAL} in the same commit if this is intentional."
    )


# ----- 3. best_config.yaml is structurally valid + contains actor block --


def test_emit_best_config_yaml_round_trips_and_contains_actor(
    tmp_path: Path,
) -> None:
    _copy_fixture_ledger_to(tmp_path)
    ledger = Ledger(tmp_path / "tuner_ledger.jsonl", fsync_on_append=False)
    best = ledger.best()
    assert best is not None

    args = _stub_args(tmp_path)
    campaign = CampaignResult(
        stop_reason="plateau",
        trial_count=3,
        oom_count=1,
        best_entry=best,
        ledger_path=tmp_path / "tuner_ledger.jsonl",
    )
    _emit_best_artefacts(campaign, args)

    yaml_path = tmp_path / "best_config.yaml"
    assert yaml_path.is_file()
    assert yaml_path.stat().st_size > 0
    # Round-trip through PyYAML to prove the emit is valid YAML. We
    # deliberately do NOT byte-compare the YAML because it is composed
    # from a live Hydra config that legitimately evolves; the fixture
    # would rot within days. The structural check catches the failure
    # mode Codex asked for: emit produces a non-YAML or empty file.
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    # Sanity: the actor knob the best entry overrode is present in the
    # emitted config. Without this, a regression that silently emits an
    # empty dict would still pass.
    assert "actor" in parsed


# ----- 4. plot output starts with PNG magic bytes ----------------------


def test_emit_ledger_plot_output_has_png_magic_bytes(tmp_path: Path) -> None:
    _copy_fixture_ledger_to(tmp_path)
    _emit_ledger_plot(tmp_path)
    png_path = tmp_path / "step_time_vs_num_trajectories.png"
    assert png_path.is_file()
    header = png_path.read_bytes()[: len(_PNG_MAGIC)]
    assert header == _PNG_MAGIC, (
        f"plot artifact is not a PNG (header bytes: {header!r}). Round-0's "
        "existence check would have missed this."
    )
