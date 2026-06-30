# embodied_tuner

An LLM-critic-driven auto-tuner for RLinf embodied training configs. Given a
baseline embodied config (e.g. `examples/embodiment/config/maniskill_ppo_openvla.yaml`),
the tuner iteratively proposes Hydra-override deltas, validates them mechanically,
runs RLinf trials, parses `metrics.log` + `timeline/*.jsonl`, and converges on a
config that minimises `step_time / num_trajectories` subject to memory and
feasibility constraints.

## Quick start

From the RLinf repo root:

```bash
# Show CLI help (shim sets PYTHONPATH=REPO_PATH for you):
bash examples/embodiment/run_embodied_tuner.sh --help

# Validate the baseline + a max_epochs=3 override via Hydra, no GPU work:
bash examples/embodiment/run_embodied_tuner.sh \
    --config maniskill_ppo_openvla \
    --dry-run-preflight

# Run a 20-trial / 12-hour tuning campaign on the maniskill baseline:
bash examples/embodiment/run_embodied_tuner.sh \
    --config maniskill_ppo_openvla \
    --max-trials 20 \
    --budget-seconds 43200
```

Direct module invocation (equivalent; the shim is just a PYTHONPATH wrapper):

```bash
cd <RLinf root>
PYTHONPATH=. python -m toolkits.embodied_tuner --config maniskill_ppo_openvla
```

## What the tuner does each trial

1. **Critic proposes** a knob delta with structured rationale
   `{summary, metric_table_citations, timeline_citations}`.
2. **Preflight validates** the delta: composes baseline + delta via Hydra,
   runs the targeted divisibility checks from `rlinf/config.py:962/965/980/1363-1368`
   plus a placement-legality check. **Never starts Ray** (preflight is GPU-free
   and Ray-free by contract).
3. **Trial runner** launches the trial as a subprocess in its own POSIX
   process group with a per-trial timeout, escalates SIGTERM → SIGKILL on
   timeout, invokes `ray stop --force`, and sweeps `/proc/<pid>/environ` +
   `pgrep -f` for orphan Ray workers tagged with the trial id.
4. **Parser** reads `metrics.log` (the MetricTable) + `timeline/*.jsonl`
   (per-component events), classifies the trial with
   `(Status, FailureMode)`, computes the objective as
   `step_time / num_trajectories` averaged over steps 2..N (step 1 = warmup),
   and surfaces a per-component timeline summary the next critic prompt
   consumes.
5. **Ledger** persists each trial as one JSONL line, including the structured
   `critic_rationale` payload — the audit trail that lets operators trace
   which MetricTable observation AND which timeline observation drove each
   placement change.

## The dual-source rationale rule (critic contract)

Any critic-proposed delta that touches `cluster.component_placement` **must**
cite at least one observation from the MetricTable AND at least one
observation from `timeline/*.jsonl`. The output schema is:

```json
{
  "delta": {"cluster.component_placement": {"actor": "0-7", "env": "0-3", "rollout": "4-7"}},
  "rationale": {
    "summary": "Env is the bottleneck (env/interact dominates); shrink env's GPU range to free GPUs for rollout.",
    "metric_table_citations": ["env/interact=275.4", "rollout/generate_one_epoch=268.8"],
    "timeline_citations": ["env rank0 env_interact_step median=15.2s stall_fraction=0.40"]
  },
  "stop_requested": false
}
```

Outputs that lack either citation array on a placement-touching delta are
rejected by `CriticOutputValidator` and the critic retries (up to 3 times)
with the rejection reason as feedback.

Non-placement deltas (e.g. `actor.micro_batch_size=64`, `env.train.enable_offload=true`)
need only a non-empty `summary`; citations are optional.

## CLI flags

| Flag                          | Default                                  | Purpose                                                                  |
|-------------------------------|------------------------------------------|--------------------------------------------------------------------------|
| `--config NAME`               | (required)                               | Hydra `--config-name` under `examples/embodiment/config/`.               |
| `--baseline PATH`             | `examples/embodiment/config/<config>.yaml` | Override the baseline YAML path.                                        |
| `--max-trials N`              | `20`                                     | Stop after N completed trials.                                           |
| `--budget-seconds SEC`        | `43200` (12h)                            | Wall-clock budget.                                                       |
| `--max-oom N`                 | `5`                                      | Stop when cumulative OOM count exceeds N.                                |
| `--patience N`                | `3`                                      | Plateau window: stop when last N non-failed trials improved <`epsilon`.   |
| `--epsilon FRAC`              | `0.02`                                   | Plateau improvement threshold (relative).                                |
| `--max-epochs N`              | `3`                                      | Hydra override `runner.max_epochs=N`. Step 1 is dropped as warmup.       |
| `--collect-memory`            | on                                       | Export `RLINF_NVITOP=1`/`RLINF_NVML=1` per trial.                         |
| `--no-collect-memory`         | —                                        | Skip the two memory-telemetry env vars.                                  |
| `--no-profiler`               | off                                      | Skip all `RLINF_TIMELINE*` env vars. Resulting trials are flagged `(OK, METRICS_PARTIAL)` and ineligible for best-config selection. |
| `--dry-run-preflight`         | off                                      | Compose baseline + `max_epochs` override via Hydra; exit 0 without launching a trial. Useful for sanity-checking the config. |
| `--fake-critic FILE`          | —                                        | JSON array of `{delta, stop_requested?}` entries. Bypasses the real Codex critic — used by the smoke test. |
| `--ledger-dir DIR`            | `logs/tuner-<timestamp>-<nonce>-<config>` | Where `tuner_ledger.jsonl`, `best_config.yaml`, `best_trial.json` are written. |
| `--ask-codex-path PATH`       | bundled                                  | Override the path to `ask-codex.sh` (the Codex transport).               |

## Outputs

All written to `<ledger_dir>/`:

- **`tuner_ledger.jsonl`** — one JSON object per trial. Fields:
  `trial_idx, delta, resolved_config_sha, log_dir, returncode, status,
  failure_mode, objective, step_time, num_trajectories,
  per_component_timings, timeline_summary, peak_gpu_mem, critic_rationale,
  ts_start, ts_end, cleanup_outcome`. Append-only with per-line `fsync` so a
  mid-loop crash never truncates a prior entry.
- **`best_config.yaml`** — the Hydra-COMPOSED (unresolved) YAML of the
  baseline with the winning trial's delta applied. `${oc.env:...}`
  interpolations stay symbolic, so the file is portable. Promote it into
  `examples/embodiment/config/` as a new entry whenever you're ready.
- **`best_trial.json`** —
  `{objective, denominator_source, step_range_used, exclusion_reasons,
  source_trial_idx}`. Explains *why* the chosen trial won.

When no trial qualifies (`status=OK, failure_mode=NONE`), `best_trial.json`
still gets written with `objective=null` and the campaign's stop reason in
`exclusion_reasons`.

## Tunable knobs

The critic may mutate any of these (see `schema.py`):

- `cluster.component_placement` — GPU-range strings for actor / env / rollout,
  e.g. `{actor: "0-7", env: "0-3", rollout: "4-7"}`. Validated against
  `ModelParallelComponentPlacement`'s contiguity + env/rollout disjoint-or-equal
  rules.
- `env.train.total_num_envs`, `env.train.rollout_epoch`
- `actor.micro_batch_size`
- `env.train.enable_offload`, `rollout.enable_offload`, `actor.enable_offload`

These knobs are **pinned** (rejected by the schema) and tracked under `FUT-5`:
`actor.global_batch_size`, `rollout.pipeline_stage_num`,
`actor.model.num_action_chunks`. Touching them triggers a `KnobNotTunableError`.

## Trial classification

`(Status, FailureMode)` grid:

| Status | FailureMode        | Meaning                                                                       | Best-config eligible? |
|--------|--------------------|--------------------------------------------------------------------------------|------------------------|
| OK     | NONE               | Full `metrics.log` MetricTable AND `timeline/*.jsonl` present.                | yes                    |
| OK     | METRICS_PARTIAL    | Trial completed but data is incomplete (e.g. `timeline/` missing because of `--no-profiler`, or `num_trajectories` row absent). | no                     |
| FAILED | OOM                | `stderr`/`run_embodiment.log` matched `CUDA out of memory` etc.; counts toward `--max-oom`. | no                     |
| FAILED | WORKER_CRASH       | Ray actor death / Python traceback / SIGKILL.                                  | no                     |
| FAILED | TIMEOUT            | Per-trial timeout exceeded; subprocess was SIGKILL'd.                          | no                     |
| FAILED | METRICS_MISSING    | `metrics.log` is absent and no failure signal was found.                       | no                     |
| FAILED | CONFIG_INVALID     | Preflight rejected the delta (used by the `preflight_exhausted` stop reason). | no                     |
| FAILED | LAUNCH_FAILURE     | Subprocess never started.                                                      | no                     |
| FAILED | NONE               | **Invariant violation** — the parser refuses to construct this.                | n/a                    |

## Stopping rules

The scheduler terminates with one of:

- `max_trials_reached` — `--max-trials` consumed.
- `budget_seconds_elapsed` — wall-clock exceeded `--budget-seconds`.
- `oom_cap_exceeded` — cumulative OOM count exceeded `--max-oom`.
- `plateau` — last `--patience` non-failed trials all improved less than
  `--epsilon` (relative).
- `critic_stagnation` — critic returned `stop_requested=true` twice in a row.
- `preflight_exhausted` — critic produced `preflight_retries+1` consecutive
  invalid deltas; a synthetic `(FAILED, CONFIG_INVALID)` ledger entry is
  written and no runner is launched.
- `critic_failure` — `CodexCritic` could not produce a parseable / valid
  output within `max_retries` (transport error, malformed JSON, dual-source
  failure).
- `no_trials_run` — `--max-trials=0`.

## Architecture

```
critic.py              CriticPrompt + build_prompt + CriticOutputValidator + CodexCritic
fake_critic.py         Deterministic FakeCritic used by tests and the smoke harness
schema.py              KnobSchema + KnobDomain + KnobSchemaError hierarchy
placement_enum.py      parse_range_spec / is_legal_placement / enumerate_placements
override_wrapper.py    OverrideWrapper.build_invocation → LaunchSpec
preflight.py           compose_and_validate (Hydra compose + local divisibility checks)
runner.py              TrialRunner.launch (subprocess + timeout + scoped cleanup + profiler env)
parser.py              parse_trial → TrialResult; select_best; TimelineSummary
ledger.py              Ledger.append/.load/.best — append-only JSONL with fsync
scheduler.py           Scheduler.run → CampaignResult (orchestrates the loop)
__main__.py            CLI entrypoint (python -m toolkits.embodied_tuner)
tests/                 197 passing unit tests (Round 0..4)
```

The toolkit deliberately does **not** import `toolkits.auto_placement`
(enforced by an AST walker bundled into the smoke-test suite) and does **not**
modify the stock `examples/embodiment/run_embodiment.sh`.

## Running the tests

```bash
cd <RLinf root>
PYTHONPATH=. python -m pytest toolkits/embodied_tuner/tests/
```

Expect `202 passed`. All tests are hermetic — no real RLinf launch, no Ray,
no GPU. The smoke test (`tests/test_smoke.py`) drives the scheduler
end-to-end via `FakeCritic` + mock runner_fn / parser_fn / preflight_fn.

## Integration with humanize / RLCR

This toolkit was built via the humanize + RLCR pipeline (`/humanize:gen-plan`
→ `/humanize:refine-plan` → `/humanize:start-rlcr-loop`). The plan lives at
the repo root's `docs/plan.md` (RLinf copy: `RLinf/docs/plan.md`); the loop
artefacts (`goal-tracker.md`, `round-{0,1,2,3,4}-{contract,summary}.md`) are
under `.humanize/rlcr/2026-06-29_15-37-02/`.

The tuner itself is plain Python — it is **not** an RLCR loop. Specifically:
one RLCR round of the build process equals "the entire plan is finished",
not "one RLinf trial". The trial loop happens inside `Scheduler.run()`.

## Known limitations (FUT-1 … FUT-9)

The plan explicitly defers these:

- Multi-node placement (`FUT-1`). Current scope is single-node 8xA800.
- Async pipelined trials (`FUT-2`).
- Bayesian / evolutionary fallback proposer for when the LLM critic
  plateaus (`FUT-3`).
- Parametric-surrogate warm start (`FUT-4`).
- Un-pinning `actor.global_batch_size`, `rollout.pipeline_stage_num`,
  `actor.model.num_action_chunks` (`FUT-5`).
- Evaluation / checkpoint during trials (`FUT-6`).
- Cross-config tuning (`FUT-7`).
- Deterministic memory-shrink retry on OOM (`FUT-8`).
- Additional baseline configs beyond `maniskill_ppo_openvla` (`FUT-9`).

Codex review also flagged three polish items (non-blocking):

- Mirror broader `validate_cfg` rules in preflight (group_size divisibility,
  eval-side divisibility, supported model_type, etc.). Current preflight
  mirrors the 4 divisibility checks the plan explicitly tests.
- Softer evidence requirement for memory-sensitive non-placement deltas
  (`enable_offload`, `total_num_envs`, `micro_batch_size`).
- Opt-in `ray stop --force` (currently default-on for safety on shared hosts).

## Environment expectations

- Python 3.10+; the rest of the RLinf stack.
- `hydra-core` and `OmegaConf` (already in `pyproject.toml`).
- For the production critic: `ask-codex.sh` at the path supplied by
  `--ask-codex-path` (default points at the bundled humanize plugin install).
  `CodexCritic` requires `CLAUDE_PROJECT_DIR` to be set when invoked outside
  a trusted git repo, and `HUMANIZE_CODEX_BYPASS_SANDBOX=1` if the project
  directory is not in Codex's trusted-directory list.
- For real RLinf trials: whatever embodied stack `run_embodiment.sh` would
  need (`MUJOCO_GL=egl`, `ROBOT_PLATFORM`, etc.). The tuner forwards these
  via `OverrideWrapper`'s default env block.
- For orphan-cleanup `/proc/<pid>/environ` scanning: POSIX Linux (graceful
  fallback to `pgrep -f` only on other platforms).
