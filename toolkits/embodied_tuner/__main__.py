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

"""CLI entrypoint for the embodied auto-tuner.

Invoke as::

    python -m toolkits.embodied_tuner --config maniskill_ppo_openvla [...]

or via the bundled shim launcher::

    bash examples/embodiment/run_embodied_tuner.sh --config maniskill_ppo_openvla

Both forms expect ``RLinf/`` on ``PYTHONPATH``; the shim sets this up
automatically and prints a remediation hint if it cannot.

On termination, the CLI emits two artefacts alongside the ledger:

- ``best_config.yaml`` — the Hydra-resolved YAML of the baseline with
  the best trial's delta applied. Operators can promote this directly
  into ``examples/embodiment/config/`` as a new entry.
- ``best_trial.json`` — ``{objective, denominator_source,
  step_range_used, exclusion_reasons, source_trial_idx}`` so operators
  understand WHY the chosen trial won.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
import time as _time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from toolkits.embodied_tuner.critic import (
    CodexCritic,
    Critic,
    TrialHistoryEntry,
)
from toolkits.embodied_tuner.fake_critic import FakeCritic
from toolkits.embodied_tuner.ledger import Ledger
from toolkits.embodied_tuner.override_wrapper import LaunchSpec, OverrideWrapper
from toolkits.embodied_tuner.parser import TrialResult, parse_trial
from toolkits.embodied_tuner.preflight import (
    ValidationResult,
    compose_and_validate,
)
from toolkits.embodied_tuner.runner import TrialOutcome, TrialRunner
from toolkits.embodied_tuner.scheduler import (
    BudgetConfig,
    CampaignResult,
    PreflightOutcome,
    Scheduler,
)
from toolkits.embodied_tuner.schema import KnobSchema


_LOGGER = logging.getLogger("toolkits.embodied_tuner")


class CLIError(SystemExit):
    """Structured error that maps to a non-zero CLI exit code."""

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(exit_code)
        self.message = message


@dataclass(frozen=True)
class CLIArgs:
    """Parsed CLI arguments. Keeps `main` testable."""

    config: str
    baseline: Path
    max_trials: int
    budget_seconds: float
    trial_timeout_seconds: float
    max_oom: int
    patience: int
    epsilon: float
    max_epochs: int
    collect_memory: bool
    use_profiler: bool
    dry_run_preflight: bool
    fake_critic_path: Path | None
    ledger_dir: Path
    ask_codex_path: str


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the :mod:`argparse` parser for ``python -m toolkits.embodied_tuner``."""
    parser = argparse.ArgumentParser(
        prog="python -m toolkits.embodied_tuner",
        description=(
            "Iteratively tune an embodied RLinf training config to minimise "
            "step_time / num_trajectories under memory constraints."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Hydra config-name under examples/embodiment/config/ (e.g. maniskill_ppo_openvla).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Optional explicit baseline YAML path. Defaults to "
            "examples/embodiment/config/<config>.yaml under the active repo."
        ),
    )
    parser.add_argument("--max-trials", type=int, default=20, help="Max trials before stop (default 20).")
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=43_200.0,
        help="Wall-clock budget in seconds (default 12h).",
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=float,
        default=2700.0,
        help=(
            "Per-trial wall-clock budget in seconds (default 2700 = 45min). "
            "When a trial exceeds this, the runner escalates SIGTERM → SIGKILL "
            "and the trial is classified (FAILED, TIMEOUT)."
        ),
    )
    parser.add_argument("--max-oom", type=int, default=5, help="Cumulative OOM tolerance (default 5).")
    parser.add_argument("--patience", type=int, default=3, help="Plateau patience window (default 3).")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="Plateau improvement threshold (default 0.02 = 2%%).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=3,
        help="runner.max_epochs Hydra override per trial (default 3; every step contributes to the averaged objective).",
    )
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--collect-memory",
        dest="collect_memory",
        action="store_true",
        default=False,
        help="Export RLINF_NVITOP/RLINF_NVML (default).",
    )
    memory_group.add_argument(
        "--no-collect-memory",
        dest="collect_memory",
        action="store_false",
        help="Skip NVITOP/NVML memory telemetry.",
    )
    parser.add_argument(
        "--no-profiler",
        action="store_true",
        help="Skip RLINF_TIMELINE* env exports; trials run without timeline JSONL.",
    )
    parser.add_argument(
        "--dry-run-preflight",
        action="store_true",
        help="Compose the baseline + delta via Hydra and exit without spawning the trial.",
    )
    parser.add_argument(
        "--fake-critic",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file used to bypass the real CodexCritic. The "
            "file must contain a JSON array; each element is an object "
            '{"delta": {<knob>: <value>, ...}, "stop_requested": <bool, optional>}. '
            "Rationale fields are ignored (FakeCritic synthesises a minimal "
            "summary). If ANY element has stop_requested=true, the loader "
            "uses FakeCritic.stop_after so the final response signals "
            "campaign termination. Used by the AC-11 smoke test."
        ),
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=None,
        help=(
            "Directory for tuner_ledger.jsonl, best_config.yaml, best_trial.json. "
            "Defaults to a timestamped directory under the repo's logs/."
        ),
    )
    parser.add_argument(
        "--critic-backend",
        choices=("codex", "claude"),
        default="codex",
        help=(
            "Which vendored ask-*.sh backend to use for the LLM critic. "
            "Only takes effect when --ask-codex-path is left at its default. "
            "'codex' uses scripts/ask-codex.sh; 'claude' uses scripts/ask-claude.sh."
        ),
    )
    parser.add_argument(
        "--ask-codex-path",
        default=None,
        help=(
            "Override path to the critic transport script. When left unset the "
            "path is derived from --critic-backend (scripts/ask-<backend>.sh "
            "next to this module). The name is kept for backwards compatibility."
        ),
    )
    return parser


def parse_cli_args(argv: list[str] | None = None) -> CLIArgs:
    """Parse argv into a typed :class:`CLIArgs`. Raises :class:`CLIError`."""
    parser = build_parser()
    try:
        ns = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already wrote a message; convert to our structured error
        # so callers can re-raise consistently.
        if exc.code == 0:
            raise
        raise CLIError("invalid arguments", exit_code=int(exc.code or 2)) from exc

    repo_root = _detect_repo_root()
    baseline = ns.baseline
    if baseline is None:
        baseline = repo_root / "examples" / "embodiment" / "config" / f"{ns.config}.yaml"
    baseline = Path(baseline).expanduser()
    if not baseline.is_file():
        raise CLIError(f"baseline config not found: {baseline}")

    ledger_dir = ns.ledger_dir
    if ledger_dir is None:
        # Append a short urandom suffix to the timestamp so two same-user
        # same-config campaigns launched within the same second still get
        # distinct ledger directories (and therefore distinct campaign
        # ids and distinct RLINF_TUNER_TRIAL_ID prefixes). Without the
        # nonce, the orphan-cleanup tag could match a sibling campaign's
        # Ray workers and kill them.
        import os as _os

        stamp = _time.strftime("%Y%m%d-%H:%M:%S")
        nonce = _os.urandom(3).hex()  # 6 hex chars
        ledger_dir = repo_root / "logs" / f"tuner-{stamp}-{nonce}-{ns.config}"
    ledger_dir = Path(ledger_dir).expanduser()

    ask_script_path = ns.ask_codex_path
    if ask_script_path is None:
        ask_script_path = str(
            Path(__file__).resolve().parent
            / "scripts"
            / f"ask-{ns.critic_backend}.sh"
        )

    return CLIArgs(
        config=ns.config,
        baseline=baseline,
        max_trials=ns.max_trials,
        budget_seconds=ns.budget_seconds,
        trial_timeout_seconds=ns.trial_timeout_seconds,
        max_oom=ns.max_oom,
        patience=ns.patience,
        epsilon=ns.epsilon,
        max_epochs=ns.max_epochs,
        collect_memory=ns.collect_memory,
        use_profiler=not ns.no_profiler,
        dry_run_preflight=ns.dry_run_preflight,
        fake_critic_path=ns.fake_critic,
        ledger_dir=ledger_dir,
        ask_codex_path=ask_script_path,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Module entry point. Returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        args = parse_cli_args(argv)
    except CLIError as exc:
        sys.stderr.write(f"embodied_tuner: {exc.message}\n")
        return exc.code

    args.ledger_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(
        "embodied_tuner: config=%s baseline=%s ledger_dir=%s",
        args.config,
        args.baseline,
        args.ledger_dir,
    )

    if args.dry_run_preflight:
        return _run_dry_run_preflight(args)

    return _run_campaign(args)


def _campaign_id(ledger_dir: Path) -> str:
    """Return a campaign-unique id derived from the ledger directory.

    The orphan-cleanup tag (``RLINF_TUNER_TRIAL_ID=<campaign>-<trial>``)
    MUST be unique across concurrent campaigns on a shared host, otherwise
    one campaign's cleanup could kill another's Ray workers. We derive
    the campaign id from the absolute ledger directory path because the
    CLI guarantees a fresh timestamped ``ledger_dir`` per campaign.
    """
    import hashlib

    digest = hashlib.sha1(str(ledger_dir.resolve()).encode("utf-8")).hexdigest()
    return digest[:12]


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def _run_dry_run_preflight(args: CLIArgs) -> int:
    """Compose baseline (no delta) + max_epochs override and report status."""
    overrides = (f"runner.max_epochs={args.max_epochs}",)
    result = compose_and_validate(args.baseline, delta={}, hydra_overrides=overrides)
    if result.ok:
        _LOGGER.info(
            "dry-run preflight: OK (placement_kind=%s, sha=%s)",
            result.placement_kind,
            result.resolved_config_sha,
        )
        return 0
    sys.stderr.write("dry-run preflight FAILED:\n")
    for err in result.errors:
        sys.stderr.write(f"  - {err}\n")
    return 1


def _run_campaign(args: CLIArgs) -> int:
    """Run the full tuner loop, then emit best_config.yaml + best_trial.json."""
    schema = KnobSchema()
    critic = _build_critic(args, schema)

    repo_root = _detect_repo_root()
    wrapper = OverrideWrapper.for_repo(repo_root)
    runner = TrialRunner(
        timeout_seconds=args.trial_timeout_seconds,
        disable_profiler=not args.use_profiler,
        disable_memory_telemetry=not args.collect_memory,
    )
    ledger_path = args.ledger_dir / "tuner_ledger.jsonl"
    ledger = Ledger(ledger_path)
    budget = BudgetConfig(
        max_trials=args.max_trials,
        budget_seconds=args.budget_seconds,
        max_oom=args.max_oom,
        patience=args.patience,
        epsilon=args.epsilon,
    )
    campaign_id = _campaign_id(args.ledger_dir)
    _LOGGER.info("embodied_tuner: campaign_id=%s", campaign_id)

    preflight_fn = functools.partial(
        _preflight_adapter,
        baseline=args.baseline,
        max_epochs=args.max_epochs,
        ledger_dir=args.ledger_dir,
    )
    runner_fn = functools.partial(
        _runner_adapter,
        wrapper=wrapper,
        runner=runner,
        config_name=args.config,
        max_epochs=args.max_epochs,
        ledger_dir=args.ledger_dir,
        campaign_id=campaign_id,
    )
    parser_fn = functools.partial(_parser_adapter)

    baseline_knobs = _extract_baseline_knobs(args.baseline, schema)
    scheduler = Scheduler(
        critic=critic,
        runner_fn=runner_fn,
        parser_fn=parser_fn,
        preflight_fn=preflight_fn,
        ledger=ledger,
        budget=budget,
        baseline_knobs=baseline_knobs,
    )
    campaign = scheduler.run()
    _emit_best_artefacts(campaign, args)
    _LOGGER.info(
        "embodied_tuner: stop_reason=%s trials=%d oom=%d",
        campaign.stop_reason,
        campaign.trial_count,
        campaign.oom_count,
    )
    return 0 if campaign.best_entry is not None else 3


# ---------------------------------------------------------------------------
# Adapters bridging the production modules to the Scheduler's injection points
# ---------------------------------------------------------------------------


def _preflight_adapter(
    delta: Mapping[str, Any],
    *,
    baseline: Path,
    max_epochs: int,
    ledger_dir: Path,
) -> PreflightOutcome:
    overrides = (f"runner.max_epochs={max_epochs}",)
    result: ValidationResult = compose_and_validate(
        baseline, delta=delta, hydra_overrides=overrides
    )
    # Use a hash that tolerates unhashable values (dict-valued placement
    # deltas would otherwise raise ``TypeError: unhashable type: 'dict'``
    # if we used ``hash(frozenset(delta.items()))``).
    delta_token = _stable_delta_token(delta)
    stamp = _time.strftime("%Y%m%d-%H:%M:%S")
    log_dir = ledger_dir / f"trial-{stamp}-{delta_token}"
    return PreflightOutcome(
        ok=result.ok,
        errors=result.errors,
        resolved_config_sha=result.resolved_config_sha,
        log_dir=log_dir,
        delta=delta,
    )


def _stable_delta_token(delta: Mapping[str, Any]) -> str:
    """Return a short, stable token for ``delta`` usable in a log-dir name.

    Tolerates unhashable values (e.g. dict-valued placement deltas) by
    serialising via ``json.dumps`` with ``sort_keys=True``.
    """
    import hashlib

    payload = json.dumps(dict(delta), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _runner_adapter(
    delta: Mapping[str, Any],
    preflight: PreflightOutcome,
    trial_idx: int,
    *,
    wrapper: OverrideWrapper,
    runner: TrialRunner,
    config_name: str,
    max_epochs: int,
    ledger_dir: Path,
    campaign_id: str,
) -> TrialOutcome:
    overrides = [f"runner.max_epochs={max_epochs}"]
    for key, value in delta.items():
        overrides.append((key, value))
    log_dir = preflight.log_dir if preflight.log_dir != Path("") else ledger_dir / f"trial-{trial_idx}"
    # Trial id MUST be unique across concurrent campaigns on a shared host,
    # otherwise the orphan-cleanup pgrep/proc scan in TrialRunner could
    # match Ray workers from another campaign and kill them.
    spec = wrapper.build_invocation(
        config_name,
        overrides=overrides,
        log_dir=log_dir,
        trial_id=f"{campaign_id}-{trial_idx}",
    )
    return runner.launch(spec)


def _parser_adapter(outcome: TrialOutcome) -> TrialResult:
    enable_offload = _extract_trial_context(outcome.log_dir)
    return parse_trial(
        outcome.log_dir,
        returncode=outcome.returncode,
        timed_out=outcome.timed_out,
        stderr_path=outcome.stdout_path,
        enable_offload=enable_offload,
        plot_formats=("png", "html"),
    )


def _extract_trial_context(
    log_dir: Path,
) -> Mapping[str, bool] | None:
    """Recover this trial's effective offload knobs.

    The embodied training entrypoint writes the resolved Hydra config
    to ``<log_dir>/tensorboard/config.yaml`` (the file consumed by the
    Tensorboard sidecar). We read it best-effort so the outlier
    ``knob_hint`` field in :class:`TimelineSummary` reflects the
    trial's actual ``enable_offload`` state. When the file is absent
    (e.g. test fixtures, dry-runs) the value is ``None`` and outliers
    ship without hints.
    """
    cfg_path = log_dir / "tensorboard" / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception:  # noqa: BLE001 — best-effort
        return None
    enable_offload: dict[str, bool] = {}
    for component in ("env", "rollout", "actor"):
        # env.train.enable_offload lives under env.train; rollout/actor at top.
        if component == "env":
            value = OmegaConf.select(cfg, "env.train.enable_offload")
        else:
            value = OmegaConf.select(cfg, f"{component}.enable_offload")
        if value is not None:
            enable_offload[component] = bool(value)
    return enable_offload or None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_repo_root() -> Path:
    """Return the RLinf repository root (two levels above this module)."""
    return Path(__file__).resolve().parents[2]


def _extract_baseline_knobs(baseline: Path, schema: KnobSchema) -> dict[str, Any]:
    """Read the baseline YAML and pluck the current tunable-knob values.

    Does NOT trigger Hydra composition (cheap; no env-var dependence).
    """
    raw = OmegaConf.load(baseline)
    out: dict[str, Any] = {}
    for knob in schema.list_knobs():
        value = OmegaConf.select(raw, knob)
        if value is not None:
            out[knob] = (
                OmegaConf.to_container(value, resolve=False)
                if hasattr(value, "_content")
                else value
            )
    return out


def _build_critic(args: CLIArgs, schema: KnobSchema) -> Critic:
    if args.fake_critic_path is not None:
        return _load_fake_critic(args.fake_critic_path)
    return CodexCritic(schema=schema, ask_codex_path=args.ask_codex_path)


def _load_fake_critic(path: Path) -> FakeCritic:
    """Load a list of CriticOutput-shaped dicts from a JSON file."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise CLIError(f"fake-critic file {path} must contain a JSON array")
    deltas = []
    stop_after = False
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise CLIError("fake-critic entries must be JSON objects")
        deltas.append(entry.get("delta", {}))
        stop_after = bool(entry.get("stop_requested", stop_after))
    if stop_after:
        return FakeCritic.stop_after(*deltas)
    return FakeCritic.from_deltas(*deltas)


def _emit_best_artefacts(campaign: CampaignResult, args: CLIArgs) -> None:
    """Write ``best_config.yaml`` and ``best_trial.json`` next to the ledger."""
    best_config_path = args.ledger_dir / "best_config.yaml"
    best_trial_path = args.ledger_dir / "best_trial.json"

    if campaign.best_entry is None:
        # Still emit best_trial.json with explicit no-result information.
        best_trial_path.write_text(
            json.dumps(
                {
                    "objective": None,
                    "denominator_source": "num_trajectories (final MetricTable block)",
                    "step_range_used": "steps 1..N (all blocks averaged)",
                    "exclusion_reasons": [
                        f"stop_reason={campaign.stop_reason}",
                        "no (OK, NONE) trial produced",
                    ],
                    "source_trial_idx": None,
                },
                indent=2,
            )
        )
        return

    best = campaign.best_entry
    # Compose the best config via Hydra so the emitted YAML reflects
    # everything the runner would have seen. The YAML is the
    # Hydra-COMPOSED form (unresolved), so ``${oc.env:...}`` interpolations
    # stay symbolic and the file is portable across hosts with different
    # env-var settings.
    result = compose_and_validate(
        args.baseline,
        delta=dict(best.delta),
        hydra_overrides=(f"runner.max_epochs={args.max_epochs}",),
    )
    if result.resolved_cfg is not None:
        best_config_path.write_text(
            OmegaConf.to_yaml(result.resolved_cfg, sort_keys=True, resolve=False)
        )
    best_trial_path.write_text(
        json.dumps(
            {
                "objective": best.objective,
                "denominator_source": "num_trajectories (final MetricTable block)",
                "step_range_used": f"steps 1..{args.max_epochs} (all blocks averaged)",
                "exclusion_reasons": [],
                "source_trial_idx": best.trial_idx,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
