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

"""End-to-end smoke test for the embodied auto-tuner (AC-11).

Drives the full scheduler loop with :class:`FakeCritic` and mock
runner/parser/preflight callables. No real RLinf launch; no Ray; no
Codex call. The smoke test passes in CI within seconds.

Two scenarios are exercised:

1. A clean 4-trial run where every trial succeeds with monotonically
   improving objectives. Verifies that the ledger, best_config.yaml, and
   best_trial.json artefacts are all well-formed and that the chosen
   best trial is the lowest-objective ``(OK, NONE)`` entry.

2. A failure-injected run where trial 2 OOMs and trial 1 has a parser
   crash; the loop must still complete trials 3 and 4 and select the
   best among the eligible ones.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from toolkits.embodied_tuner.critic import CriticOutput, Rationale
from toolkits.embodied_tuner.fake_critic import FakeCritic
from toolkits.embodied_tuner.ledger import Ledger
from toolkits.embodied_tuner.override_wrapper import LaunchSpec
from toolkits.embodied_tuner.parser import (
    FailureMode,
    Status,
    TimelineSummary,
    TrialResult,
)
from toolkits.embodied_tuner.runner import TrialOutcome
from toolkits.embodied_tuner.scheduler import (
    BudgetConfig,
    PreflightOutcome,
    Scheduler,
)
from toolkits.embodied_tuner.__main__ import _emit_best_artefacts, CLIArgs


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "examples" / "embodiment" / "config" / "maniskill_ppo_openvla.yaml"


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _stub_outcome(log_dir: Path, returncode: int = 0) -> TrialOutcome:
    spec = LaunchSpec(
        argv=(),
        env={"RLINF_TUNER_TRIAL_ID": "x"},
        log_dir=log_dir,
        config_name="stub",
        baseline_overrides=(),
        user_overrides=(),
    )
    return TrialOutcome(
        log_dir=log_dir,
        returncode=returncode,
        timed_out=False,
        wall_clock_seconds=0.0,
        cleanup_outcome="ok",
        stdout_path=log_dir / "run_embodiment.log",
        spec=spec,
    )


def _stub_result(
    log_dir: Path,
    *,
    objective: float | None,
    status: Status,
    failure_mode: FailureMode,
) -> TrialResult:
    return TrialResult(
        log_dir=log_dir,
        status=status,
        failure_mode=failure_mode,
        objective=objective,
        step_time_seconds=200.0 if objective is not None else None,
        num_trajectories=18 if objective is not None else None,
        timeline_summary=TimelineSummary(),
    )


def _ok_preflight(delta: Mapping[str, Any], log_dir: Path) -> PreflightOutcome:
    return PreflightOutcome(
        ok=True,
        errors=(),
        resolved_config_sha="cafebabe",
        log_dir=log_dir,
        delta=delta,
    )


# ---------------------------------------------------------------------------
# Smoke scenarios
# ---------------------------------------------------------------------------


def test_smoke_e2e_clean_run(tmp_path: Path) -> None:
    """4 trials, monotonically improving objectives, no failures."""
    objectives = [40.0, 30.0, 25.0, 20.0]
    statuses = [Status.OK] * 4
    failure_modes = [FailureMode.NONE] * 4

    parser_calls: list[Path] = []

    def runner_fn(delta, preflight, trial_idx):
        log_dir = tmp_path / f"trial-{trial_idx}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return _stub_outcome(log_dir)

    def parser_fn(outcome):
        parser_calls.append(outcome.log_dir)
        idx = len(parser_calls) - 1
        return _stub_result(
            outcome.log_dir,
            objective=objectives[idx],
            status=statuses[idx],
            failure_mode=failure_modes[idx],
        )

    def preflight_fn(delta):
        return _ok_preflight(delta, tmp_path / f"pf-{len(parser_calls)}")

    ledger_path = tmp_path / "tuner_ledger.jsonl"
    critic = FakeCritic.from_deltas(
        {"actor.micro_batch_size": 64},
        {"actor.micro_batch_size": 32},
        {"actor.micro_batch_size": 16},
        {"actor.enable_offload": True},
    )

    scheduler = Scheduler(
        critic=critic,
        runner_fn=runner_fn,
        parser_fn=parser_fn,
        preflight_fn=preflight_fn,
        ledger=Ledger(ledger_path, fsync_on_append=False),
        budget=BudgetConfig(
            max_trials=4,
            budget_seconds=999,
            max_oom=99,
            patience=99,  # disable plateau for this scenario
            epsilon=0.0001,
            preflight_retries=3,
        ),
    )

    result = scheduler.run()

    # Loop terminated cleanly.
    assert result.stop_reason == "max_trials_reached"
    assert result.trial_count == 4

    # Ledger holds every trial.
    loaded = Ledger(ledger_path).load()
    assert loaded.skipped_lines == 0
    assert len(loaded.entries) == 4

    # Best trial = trial 3 (lowest objective 20.0).
    assert result.best_entry is not None
    assert result.best_entry.trial_idx == 3
    assert result.best_entry.objective == 20.0

    # Emit CLI artefacts and verify their shape.
    args = CLIArgs(
        config="maniskill_ppo_openvla",
        baseline=BASELINE,
        max_trials=4,
        budget_seconds=999.0,
        trial_timeout_seconds=2700.0,
        max_oom=99,
        patience=99,
        epsilon=0.0001,
        max_epochs=3,
        collect_memory=False,
        use_profiler=False,
        dry_run_preflight=False,
        fake_critic_path=None,
        ledger_dir=tmp_path,
        ask_codex_path="/nonexistent",
    )
    _emit_best_artefacts(result, args)
    best_config = tmp_path / "best_config.yaml"
    best_trial = tmp_path / "best_trial.json"
    assert best_config.is_file()
    assert best_trial.is_file()

    cfg = OmegaConf.load(best_config)
    # The emitted YAML reflects the best trial's delta (enable_offload=True).
    assert OmegaConf.select(cfg, "actor.enable_offload") is True

    payload = json.loads(best_trial.read_text())
    assert payload["objective"] == 20.0
    assert payload["source_trial_idx"] == 3
    assert payload["exclusion_reasons"] == []


def test_smoke_e2e_with_oom_and_parser_crash(tmp_path: Path) -> None:
    """Mocked failures must not poison subsequent trials."""
    # Index 0 = parser crash; Index 1 = OOM; Index 2 = OK; Index 3 = OK.
    objectives = [None, None, 30.0, 25.0]
    statuses = [Status.FAILED, Status.FAILED, Status.OK, Status.OK]
    failure_modes = [
        FailureMode.METRICS_MISSING,  # mocked parser crash
        FailureMode.OOM,
        FailureMode.NONE,
        FailureMode.NONE,
    ]

    parser_calls: list[Path] = []

    def runner_fn(delta, preflight, trial_idx):
        log_dir = tmp_path / f"trial-{trial_idx}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return _stub_outcome(log_dir, returncode=0 if trial_idx >= 2 else 1)

    def parser_fn(outcome):
        parser_calls.append(outcome.log_dir)
        idx = len(parser_calls) - 1
        return _stub_result(
            outcome.log_dir,
            objective=objectives[idx],
            status=statuses[idx],
            failure_mode=failure_modes[idx],
        )

    def preflight_fn(delta):
        return _ok_preflight(delta, tmp_path / f"pf-{len(parser_calls)}")

    ledger_path = tmp_path / "tuner_ledger.jsonl"
    critic = FakeCritic.from_deltas(
        {"actor.micro_batch_size": 80},
        {"actor.micro_batch_size": 64},
        {"actor.micro_batch_size": 32},
        {"actor.micro_batch_size": 16},
    )

    scheduler = Scheduler(
        critic=critic,
        runner_fn=runner_fn,
        parser_fn=parser_fn,
        preflight_fn=preflight_fn,
        ledger=Ledger(ledger_path, fsync_on_append=False),
        budget=BudgetConfig(
            max_trials=4,
            budget_seconds=999,
            max_oom=99,
            patience=99,
            epsilon=0.0001,
            preflight_retries=3,
        ),
    )

    result = scheduler.run()
    assert result.stop_reason == "max_trials_reached"
    assert result.trial_count == 4
    assert result.oom_count == 1

    loaded = Ledger(ledger_path).load()
    assert len(loaded.entries) == 4
    # Best among eligible = trial 3 (objective 25.0).
    assert result.best_entry is not None
    assert result.best_entry.trial_idx == 3
    assert result.best_entry.objective == 25.0


def test_smoke_emits_best_trial_when_no_eligible(tmp_path: Path) -> None:
    """Every trial fails → best_trial.json explains the empty result."""
    objectives = [None] * 3
    statuses = [Status.FAILED] * 3
    failure_modes = [FailureMode.OOM, FailureMode.OOM, FailureMode.OOM]

    parser_calls: list[Path] = []

    def runner_fn(delta, preflight, trial_idx):
        log_dir = tmp_path / f"trial-{trial_idx}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return _stub_outcome(log_dir, returncode=1)

    def parser_fn(outcome):
        parser_calls.append(outcome.log_dir)
        idx = len(parser_calls) - 1
        return _stub_result(
            outcome.log_dir,
            objective=objectives[idx],
            status=statuses[idx],
            failure_mode=failure_modes[idx],
        )

    def preflight_fn(delta):
        return _ok_preflight(delta, tmp_path / f"pf-{len(parser_calls)}")

    critic = FakeCritic.from_deltas({}, {}, {})
    ledger_path = tmp_path / "tuner_ledger.jsonl"
    scheduler = Scheduler(
        critic=critic,
        runner_fn=runner_fn,
        parser_fn=parser_fn,
        preflight_fn=preflight_fn,
        ledger=Ledger(ledger_path, fsync_on_append=False),
        budget=BudgetConfig(
            max_trials=3, budget_seconds=999, max_oom=10, patience=99, epsilon=0.0001
        ),
    )
    result = scheduler.run()
    assert result.best_entry is None
    args = CLIArgs(
        config="maniskill_ppo_openvla",
        baseline=BASELINE,
        max_trials=3,
        budget_seconds=999.0,
        trial_timeout_seconds=2700.0,
        max_oom=10,
        patience=99,
        epsilon=0.0001,
        max_epochs=3,
        collect_memory=False,
        use_profiler=False,
        dry_run_preflight=False,
        fake_critic_path=None,
        ledger_dir=tmp_path,
        ask_codex_path="/nonexistent",
    )
    _emit_best_artefacts(result, args)
    payload = json.loads((tmp_path / "best_trial.json").read_text())
    assert payload["objective"] is None
    assert payload["source_trial_idx"] is None
    assert any("no (OK, NONE) trial" in r for r in payload["exclusion_reasons"])
