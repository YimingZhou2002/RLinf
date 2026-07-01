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

"""Unit tests for the embodied_tuner CLI (`__main__`).

These tests do NOT launch RLinf; they exercise:
- argument parsing (defaults, error paths).
- `--dry-run-preflight` against the real baseline.
- The shim launcher's `--help` invocation.

The full end-to-end smoke test (FakeCritic + mock runner) lives in
:mod:`tests.test_smoke`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from toolkits.embodied_tuner.__main__ import (
    CLIArgs,
    CLIError,
    _campaign_id,
    _emit_best_artefacts,
    _load_fake_critic,
    _preflight_adapter,
    _stable_delta_token,
    main,
    parse_cli_args,
)
from toolkits.embodied_tuner.fake_critic import FakeCritic
from toolkits.embodied_tuner.ledger import LedgerEntry
from toolkits.embodied_tuner.scheduler import CampaignResult


REPO_ROOT = Path(__file__).resolve().parents[3]
SHIM_PATH = REPO_ROOT / "examples" / "embodiment" / "run_embodied_tuner.sh"
BASELINE = REPO_ROOT / "examples" / "embodiment" / "config" / "maniskill_ppo_openvla.yaml"


# ---------------------------------------------------------------------------
# parse_cli_args
# ---------------------------------------------------------------------------


def test_parse_cli_args_defaults() -> None:
    args = parse_cli_args(["--config", "maniskill_ppo_openvla"])
    assert isinstance(args, CLIArgs)
    assert args.config == "maniskill_ppo_openvla"
    assert args.baseline == BASELINE
    assert args.max_trials == 20
    assert args.budget_seconds == 43_200.0
    assert args.trial_timeout_seconds == 2700.0
    assert args.max_oom == 5
    assert args.patience == 3
    assert args.epsilon == 0.02
    assert args.max_epochs == 3
    assert args.collect_memory is False
    assert args.use_profiler is True
    assert args.dry_run_preflight is False
    assert args.fake_critic_path is None
    assert args.ledger_dir.is_absolute()


def test_parse_cli_args_overrides() -> None:
    args = parse_cli_args(
        [
            "--config",
            "maniskill_ppo_openvla",
            "--max-trials",
            "5",
            "--budget-seconds",
            "60",
            "--trial-timeout-seconds",
            "120",
            "--max-oom",
            "1",
            "--patience",
            "1",
            "--epsilon",
            "0.1",
            "--max-epochs",
            "2",
            "--no-profiler",
            "--no-collect-memory",
        ]
    )
    assert args.max_trials == 5
    assert args.budget_seconds == 60.0
    assert args.trial_timeout_seconds == 120.0
    assert args.max_oom == 1
    assert args.patience == 1
    assert args.epsilon == 0.1
    assert args.max_epochs == 2
    assert args.use_profiler is False
    assert args.collect_memory is False


def test_parse_cli_args_missing_required_flag() -> None:
    with pytest.raises(CLIError):
        parse_cli_args([])


def test_parse_cli_args_baseline_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CLIError) as exc:
        parse_cli_args(
            [
                "--config",
                "stub",
                "--baseline",
                str(tmp_path / "does_not_exist.yaml"),
            ]
        )
    assert "baseline" in exc.value.message


# ---------------------------------------------------------------------------
# --dry-run-preflight end-to-end
# ---------------------------------------------------------------------------


def test_dry_run_preflight_on_real_baseline(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--config",
            "maniskill_ppo_openvla",
            "--dry-run-preflight",
            "--ledger-dir",
            str(tmp_path / "ledger"),
        ]
    )
    assert exit_code == 0


def test_dry_run_preflight_via_python_m(tmp_path: Path) -> None:
    """Equivalent of the CLI surface the shim launcher will hit."""
    env = {
        "PYTHONPATH": str(REPO_ROOT),
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolkits.embodied_tuner",
            "--config",
            "maniskill_ppo_openvla",
            "--dry-run-preflight",
            "--ledger-dir",
            str(tmp_path / "ledger"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "dry-run preflight: OK" in result.stderr or "dry-run preflight: OK" in result.stdout


# ---------------------------------------------------------------------------
# _emit_best_artefacts
# ---------------------------------------------------------------------------


def _stub_args(tmp_path: Path) -> CLIArgs:
    return CLIArgs(
        config="maniskill_ppo_openvla",
        baseline=BASELINE,
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


def _stub_best_entry() -> LedgerEntry:
    from toolkits.embodied_tuner.ledger import make_entry

    return make_entry(
        trial_idx=2,
        delta={"actor.micro_batch_size": 64},
        resolved_config_sha="deadbeef",
        log_dir="/tmp/trial-2",
        returncode=0,
        status="OK",
        failure_mode="NONE",
        objective=18.5,
        step_time=333.0,
        num_trajectories=18,
        per_component_timings={"env/interact": 100.0},
        timeline_summary=None,
        peak_gpu_mem=None,
        critic_rationale={"summary": "shrink mbs", "metric_table_citations": [], "timeline_citations": []},
        ts_start=0.0,
        ts_end=1.0,
        cleanup_outcome="ok",
    )


def test_emit_best_artefacts_writes_yaml_and_json(tmp_path: Path) -> None:
    args = _stub_args(tmp_path)
    campaign = CampaignResult(
        stop_reason="plateau",
        trial_count=3,
        oom_count=0,
        best_entry=_stub_best_entry(),
        ledger_path=tmp_path / "tuner_ledger.jsonl",
    )
    _emit_best_artefacts(campaign, args)
    best_config = tmp_path / "best_config.yaml"
    best_trial = tmp_path / "best_trial.json"
    assert best_config.is_file()
    assert best_trial.is_file()
    payload = json.loads(best_trial.read_text())
    assert payload == {
        "objective": 18.5,
        "denominator_source": "num_trajectories (final MetricTable block)",
        "step_range_used": "steps 2..3 (step 1 warmup)",
        "exclusion_reasons": [],
        "source_trial_idx": 2,
    }


def test_emit_best_artefacts_no_eligible_trial(tmp_path: Path) -> None:
    args = _stub_args(tmp_path)
    campaign = CampaignResult(
        stop_reason="oom_cap_exceeded",
        trial_count=2,
        oom_count=2,
        best_entry=None,
        ledger_path=tmp_path / "tuner_ledger.jsonl",
    )
    _emit_best_artefacts(campaign, args)
    best_trial = tmp_path / "best_trial.json"
    assert best_trial.is_file()
    payload = json.loads(best_trial.read_text())
    assert payload["objective"] is None
    assert payload["source_trial_idx"] is None
    assert any("stop_reason=" in r for r in payload["exclusion_reasons"])


# ---------------------------------------------------------------------------
# _load_fake_critic
# ---------------------------------------------------------------------------


def test_load_fake_critic_array_of_deltas(tmp_path: Path) -> None:
    path = tmp_path / "fake.json"
    path.write_text(
        json.dumps(
            [
                {"delta": {"actor.micro_batch_size": 64}},
                {"delta": {"actor.micro_batch_size": 32}, "stop_requested": True},
            ]
        )
    )
    critic = _load_fake_critic(path)
    assert isinstance(critic, FakeCritic)
    assert critic.outputs[0].delta == {"actor.micro_batch_size": 64}
    assert critic.outputs[1].stop_requested is True


def test_load_fake_critic_rejects_non_array(tmp_path: Path) -> None:
    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"not": "array"}))
    with pytest.raises(CLIError):
        _load_fake_critic(path)


# ---------------------------------------------------------------------------
# Shim launcher
# ---------------------------------------------------------------------------


def test_shim_launcher_exists_and_is_executable() -> None:
    assert SHIM_PATH.is_file(), SHIM_PATH
    assert SHIM_PATH.stat().st_mode & 0o111, "shim must be executable"


def test_shim_launcher_help(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SHIM_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "python -m toolkits.embodied_tuner" in result.stdout


# ---------------------------------------------------------------------------
# Codex Round-3 review fixes
# ---------------------------------------------------------------------------


def test_preflight_adapter_handles_dict_valued_placement_delta(tmp_path: Path) -> None:
    """Regression: placement deltas are dict-valued; ``hash(frozenset(...))``
    used to crash with ``TypeError: unhashable type: 'dict'``.
    """
    from toolkits.embodied_tuner.schema import KNOB_PLACEMENT

    delta = {KNOB_PLACEMENT: {"actor": "0-7", "env": "0-3", "rollout": "4-7"}}
    # Should NOT raise — pre-fix this called hash(frozenset(delta.items())).
    outcome = _preflight_adapter(
        delta,
        baseline=BASELINE,
        max_epochs=3,
        ledger_dir=tmp_path,
    )
    assert outcome.ok is True
    # The log_dir should still incorporate a delta-derived token so two
    # different deltas land in different directories.
    assert "trial-" in str(outcome.log_dir)


def test_stable_delta_token_is_deterministic_and_handles_dicts() -> None:
    """The token must be stable across calls and tolerate unhashable values."""
    delta = {
        "cluster.component_placement": {"actor": "0-7", "env": "0-3", "rollout": "4-7"},
        "actor.micro_batch_size": 64,
    }
    a = _stable_delta_token(delta)
    b = _stable_delta_token(delta)
    assert a == b
    assert len(a) == 8
    # Different delta -> different token.
    other = _stable_delta_token({"actor.micro_batch_size": 32})
    assert other != a


def test_campaign_id_is_unique_per_ledger_dir(tmp_path: Path) -> None:
    """Two campaigns with different ledger dirs MUST get different ids
    so their RLINF_TUNER_TRIAL_ID tags don't collide on a shared host.
    """
    a = _campaign_id(tmp_path / "campaign-a")
    b = _campaign_id(tmp_path / "campaign-b")
    assert a != b
    # And re-derivation for the same path is stable.
    assert _campaign_id(tmp_path / "campaign-a") == a


def test_campaign_id_short_and_hex(tmp_path: Path) -> None:
    cid = _campaign_id(tmp_path / "c")
    assert len(cid) == 12
    assert all(c in "0123456789abcdef" for c in cid)


def test_default_ledger_dir_includes_random_nonce() -> None:
    """Codex final review: two same-second same-config launches MUST
    NOT share a default ledger_dir (would collide on campaign id).
    """
    a = parse_cli_args(["--config", "maniskill_ppo_openvla"])
    b = parse_cli_args(["--config", "maniskill_ppo_openvla"])
    # Even within the same wall-clock second, the random nonce makes
    # the two ledger_dirs distinct.
    assert a.ledger_dir != b.ledger_dir
    # And their derived campaign ids are therefore distinct too.
    assert _campaign_id(a.ledger_dir) != _campaign_id(b.ledger_dir)
