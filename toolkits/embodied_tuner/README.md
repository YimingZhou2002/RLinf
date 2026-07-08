# embodied_tuner

An LLM-critic-driven auto-tuner for RLinf embodied training configs. Given a
baseline embodied config (e.g. `examples/embodiment/config/maniskill_ppo_openvla.yaml`),
the tuner iteratively proposes Hydra-override deltas, validates them mechanically,
runs RLinf trials, parses `metrics.log` + `timeline/*.jsonl`, and converges on a
config that minimises `step_time / num_trajectories` subject to memory and
feasibility constraints.

**See also:** [`ARCHITECTURE.md`](./ARCHITECTURE.md) — internals, per-file
reference, design decisions, extension points.

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

# Run a 20-trial / 12-hour / 1 epoch per trial tuning campaign on the maniskill baseline:
bash examples/embodiment/run_embodied_tuner.sh \
    --config maniskill_ppo_openvla \
    --max-trials 20 \
    --budget-seconds 43200 \
    --max-epochs 1
```

Direct module invocation (equivalent; the shim is just a PYTHONPATH wrapper):

```bash
cd <RLinf root>
PYTHONPATH=. python -m toolkits.embodied_tuner --config maniskill_ppo_openvla
```

## What the tuner does each trial

1. **Critic proposes** a knob delta with structured rationale
   `{summary, metric_table_citations, timeline_citations}`. The critic
   prompt is augmented with a compact **DAG view** (ancestor chain of
   the active leaf, sibling attempts, top-K OK leaderboard, recent
   failures) so Codex can see the shape of the search so far.
2. **Preflight validates** the delta: composes baseline + delta via Hydra,
   runs the targeted divisibility checks from `rlinf/config.py:962/965/980/1363-1368`
   plus a placement-legality check. **Never starts Ray** (preflight is GPU-free
   and Ray-free by contract).
3. **Config dedup** (post-preflight): the resolved-config SHA is
   looked up in `config_dedup_index.jsonl`. If the cumulative config
   has already been attempted OK, the runner is short-circuited and a
   synthetic `DUPLICATE_OF` DAG node is emitted (no Ledger row, no
   trial-slot burn). If it has already FAILED, the proposal is
   rejected via `preflight_feedback` sharing the `preflight_retries`
   budget.
4. **Trial runner** launches the trial as a subprocess in its own POSIX
   process group with a per-trial timeout, escalates SIGTERM → SIGKILL on
   timeout, invokes `ray stop --force`, and sweeps `/proc/<pid>/environ` +
   `pgrep -f` for orphan Ray workers tagged with the trial id.
5. **Parser** reads `metrics.log` (the MetricTable) + `timeline/*.jsonl`
   (per-component events), classifies the trial with
   `(Status, FailureMode)`, computes the objective as
   `step_time / num_trajectories` averaged over every parsed MetricTable
   block (single-step trials are measurable),
   and surfaces a per-component timeline summary the next critic prompt
   consumes.
6. **Ledger + NodeStore** persist each trial. The flat
   `tuner_ledger.jsonl` is written unchanged (existing consumers
   continue to work) plus a `DAGNode` is appended to `nodes.jsonl` with
   an explicit `parent_id`. On a rollback failure
   (`OOM`/`WORKER_CRASH`/`TIMEOUT`/`METRICS_PARTIAL`/`METRICS_MISSING`),
   the scheduler rewinds `active_leaf` to the failing node's parent so
   the next round's proposal starts from a clean context; a
   sibling-cap counter (`--max-siblings`, default 3) drives ancestor
   climbing after repeated failures at the same parent, and a climb
   above the root terminates the campaign with the new
   `rollback_exhausted` stop reason.

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

## Bitter lessons (persistent failure memory)

The critic's rolling `history_window` (default 8) means a failed trial
falls out of the prompt after 8 subsequent rounds. Without a longer-
lived store the critic re-proposes the same failing delta again — the
`maniskill_ppo_openvla` campaign hit the same `rollout.enable_offload=False`
OOM three times for exactly this reason.

Whenever the previous trial's `failure_mode` is one of `OOM`,
`WORKER_CRASH`, `TIMEOUT`, or `CONFIG_INVALID`, the critic's response
MUST include a `bitter_lesson` payload:

```json
{
  "delta": {"actor.micro_batch_size": 20},
  "rationale": {"summary": "..."},
  "bitter_lesson": {
    "trigger": "trial 2 OOMed immediately after rollout.enable_offload=False at total_num_envs=128",
    "rule": "Do not disable rollout offload while total_num_envs >= 8 unless actor.micro_batch_size <= 20."
  }
}
```

The scheduler stamps the failed trial's `trial_idx`, `failure_mode` and
canonical delta signature onto the lesson, deduplicates by
`(failure_mode, delta_signature)`, and appends it to
**`<ledger_dir>/bitter_lessons.jsonl`** with fsync-on-write. Every
future critic prompt is prepended with a `## Bitter Lessons` section
that lists the accumulated rules verbatim — permanent memory the LLM
must respect unless it can cite concrete evidence that the memory or
feasibility envelope has changed since the failure. The store is
capped at `LessonBook.max_lessons` (default 30) via an LRU eviction
that appends an audit marker line to the same file. A scheduler
restart re-loads lessons from disk so a resumed campaign inherits its
prior failures.

`CriticOutputValidator` enforces the rule: a response after a failing
trial that omits `bitter_lesson` (or leaves either field blank) is
rejected and the critic retries with the reason as feedback, sharing
the existing `max_retries` retry loop with dual-source violations.

## CLI flags

| Flag                          | Default                                  | Purpose                                                                  |
|-------------------------------|------------------------------------------|--------------------------------------------------------------------------|
| `--config NAME`               | (required)                               | Hydra `--config-name` under `examples/embodiment/config/`.               |
| `--baseline PATH`             | `examples/embodiment/config/<config>.yaml` | Override the baseline YAML path.                                        |
| `--max-trials N`              | `20`                                     | Stop after N completed trials.                                           |
| `--budget-seconds SEC`        | `43200` (12h)                            | Wall-clock budget.                                                       |
| `--trial-timeout-seconds SEC` | `2700` (45min)                           | Per-trial wall-clock budget. On timeout the runner escalates SIGTERM → SIGKILL and the trial is classified `(FAILED, TIMEOUT)`. |
| `--max-oom N`                 | `5`                                      | Stop when cumulative OOM count exceeds N.                                |
| `--patience N`                | `3`                                      | Plateau window: stop when last N non-failed trials improved <`epsilon`.   |
| `--epsilon FRAC`              | `0.02`                                   | Plateau improvement threshold (relative).                                |
| `--max-epochs N`              | `3`                                      | Hydra override `runner.max_epochs=N`. All steps contribute to the averaged objective.       |
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
- **`bitter_lessons.jsonl`** — append-only, deduplicated store of
  `{trigger, rule, trial_idx, failure_mode, delta_signature}` records.
  See "Bitter lessons" above.

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
lessons.py             LessonBook / BitterLesson — persistent failure memory
scheduler.py           Scheduler.run → CampaignResult (orchestrates the loop)
__main__.py            CLI entrypoint (python -m toolkits.embodied_tuner)
tests/                 unit tests
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

## Known limitations (FUT-1 … FUT-12)

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
- **Algorithmic frontier-selection over the DAG** (`FUT-10`).
  Introduce a `FrontierPolicy` protocol with concrete implementations
  (`BestFirstPolicy`, `UCB1Policy`, `BeamPolicy`, `PlateauBacktrackPolicy`);
  replace the current "always expand active leaf" order with
  policy-driven expansion; adjust plateau / critic-stagnation semantics
  for graph search. Corresponds to Alt-5 from the DAG-store design
  draft. Prerequisites are the current-loop DAG infrastructure
  (`AC-1` NodeStore, `AC-5` rollback state machine, `AC-6` dedup
  index). A negative-space AST walker
  (`tests/test_no_frontier_policy.py`) guards against accidental
  drift: introducing any class / function / import named
  `FrontierPolicy`, `FrontierScheduler`, `BestFirstPolicy`,
  `UCB1Policy`, `BeamPolicy`, or `PlateauBacktrackPolicy` inside
  `toolkits/embodied_tuner/**/*.py` will immediately fail the test
  suite until this FUT is explicitly promoted in a future RLCR loop.
- **Operator-facing DAG visualization + manual branch injection**
  (`FUT-11`). ASCII / Plotly-HTML DAG viewer alongside every campaign
  and a `--inject --from-trial N --delta '{...}'` CLI flag for
  human-directed branching. Corresponds to Alt-4 from the DAG-store
  design draft. Prerequisite: `AC-1` (`NodeStore` provides the
  read model).
- **True multi-parent DAG for duplicate-config convergence**
  (`FUT-12`). Model unique resolved configs as nodes keyed by
  `resolved_config_sha` with multiple incoming edges when different
  parent paths converge, replacing the single-parent tree +
  `duplicate_of_node_id` back-reference the current loop uses (see
  `AC-6`). Prerequisite: `AC-1`.

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
