# RLCR-Built Auto-Tuner for RLinf Embodied Configs

## Goal Description

Build a Python orchestrator under `RLinf/toolkits/embodied_tuner/` (importable as `toolkits.embodied_tuner` via the existing `RLinf/toolkits/__init__.py`) that, given a baseline embodied RLinf config (default: `RLinf/examples/embodiment/config/maniskill_ppo_openvla.yaml`), iteratively proposes config deltas, validates them mechanically, runs RLinf trials, parses logs and `timeline/*.jsonl`, and converges on a configuration that minimizes `step_time / num_trajectories` subject to memory and feasibility constraints (no OOM, no worker crash, no validate_cfg violation).

The tunable knob set is `cluster.component_placement` (env/actor/rollout/all GPU range strings), `env.train.total_num_envs`, `env.train.rollout_epoch`, `actor.micro_batch_size`, and the three `enable_offload` flags (env/rollout/actor). `actor.global_batch_size`, `rollout.pipeline_stage_num`, and `actor.num_action_chunks` are pinned in this current loop to keep `validate_cfg` divisibility ripples tractable; loosening them is `FUT-5`.

The humanize + RLCR pipeline is used ONCE to build this tuner. The tuner's per-trial loop is plain Python — it is NOT itself an RLCR loop, and one RLCR round does NOT correspond to one RLinf trial. This avoids the round-boundary mismatch the draft flagged ("agent believes entire plan is finished" vs. one trial). The existing `RLinf/toolkits/auto_placement/` package is NOT imported or extended; it requires `config.profile_data` for cold start, which embodied configs lack, and the user has explicitly excluded it.

Every placement-optimization decision made by the LLM critic during the tuning loop MUST be grounded in BOTH coarse evidence from `metrics.log` (MetricTable per-component aggregates) AND fine-grained evidence parsed from `timeline/*.jsonl` (per-rank min/median/max timings, stall fractions, call counts). A critic-output validator rejects any proposed delta that touches `cluster.component_placement` without citing at least one observation from each source.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.
The `AC-*` items are current RLCR completion gates for this implementation loop.

- AC-1: Knob schema implemented and unit-tested. A `KnobSchema` dataclass (or equivalent) enumerates the legal domain for each tunable knob and the dependencies between them (e.g. env GPU count derives from `cluster.component_placement`). Knobs that are pinned in this loop (`actor.global_batch_size`, `rollout.pipeline_stage_num`, `actor.num_action_chunks`) are reserved in the schema with a `pinned=True` marker so the un-pinning path in FUT-5 is a schema flip.
  - Positive Tests (expected to PASS):
    - A baseline-shaped delta with every knob inside its declared domain passes `KnobSchema.validate(delta)`.
    - `KnobSchema.list_knobs()` returns exactly the committed tunable knob set.
  - Negative Tests (expected to FAIL):
    - A delta containing an out-of-range value (e.g. `actor.micro_batch_size = -1`) is rejected with a structured error naming the knob and the violated bound.
    - A delta containing a pinned knob (e.g. `actor.global_batch_size = 1024`) is rejected with `KnobNotTunableError`.

- AC-2: Override wrapper invokes `RLinf/examples/embodiment/run_embodiment.sh` such that arbitrary user-supplied Hydra overrides reach `train_embodied_agent.py`, are reflected in the resolved Hydra config dumped at `LOG_DIR/.hydra/config.yaml`, and the captured `LOG_DIR` matches the directory actually used by the trial. The stock script only injects `runner.logger.log_path=${LOG_DIR}`; the wrapper extends the override list.
  - Positive Tests:
    - Passing `actor.micro_batch_size=64` through the wrapper produces `actor.micro_batch_size: 64` in `LOG_DIR/.hydra/config.yaml`.
    - When `actor.micro_batch_size` is set in the baseline AND via override, the override value wins in `.hydra/config.yaml` (Hydra precedence check).
    - The wrapper returns the exact `LOG_DIR` that contains the produced `.hydra/config.yaml`.
  - Negative Tests:
    - Passing a syntactically-invalid Hydra override (e.g. `nonexistent.key=1`) results in either pre-launch `CONFIG_INVALID` classification (preferred) or, if it bypasses preflight, a launch failure classified as `LAUNCH_FAILURE`. The wrapper MUST NOT silently fall back to the baseline value.

- AC-3: Preflight validator composes baseline + delta via Hydra, runs `RLinf/rlinf/config.py::validate_cfg`, runs `validate_embodied_cfg`, and runs the placement legality check WITHOUT launching any GPU work.
  - Positive Tests:
    - A known-legal delta (e.g. flip `env.train.enable_offload` from False to True) passes preflight.
    - The fully resolved config returned by preflight matches what `train_embodied_agent.py` would see if launched.
  - Negative Tests:
    - A delta setting `actor.micro_batch_size` to a value that violates `global_batch_size % (micro_batch_size * actor_world_size) == 0` (per `rlinf/config.py:1363-1368`) is rejected before any subprocess is spawned.
    - A delta setting `env.train.total_num_envs` to a value violating `total_num_envs % env_world_size == 0` (per `rlinf/config.py:962`) is rejected before launch.

- AC-4: Placement enumerator generates only legal placements respecting the contiguity and non-overlap rules enforced by `ModelParallelComponentPlacement` (per `RLinf/rlinf/utils/placement.py` contiguous-GPU assertions and `_is_disaggregated` non-overlap logic). Enumeration covers collocated, disaggregated, hybrid, and "all" patterns within single-node 8 GPU scope.
  - Positive Tests:
    - Every enumerated placement string parses successfully into a `ModelParallelComponentPlacement` instance under a mock cluster of 8 GPUs.
    - The enumerator emits at least one collocated, one disaggregated, one hybrid, and one "all" pattern.
  - Negative Tests:
    - A hand-crafted overlapping placement `env:0-3, actor:2-5` is excluded by the enumerator and, if fed in via `--baseline`, is rejected by preflight (AC-3) before launch.
    - A malformed range string `env:3-1` (high < low) is excluded.
    - A non-contiguous range string `env:0,2,4-7` is excluded.

- AC-5: Trial runner launches a real RLinf trial via the wrapper, captures `LOG_DIR`, enforces a configurable per-trial timeout (default 2700 seconds), and unconditionally exports profiler env vars (`RLINF_TIMELINE=1`, `RLINF_TIMELINE_WORKER_TIMER=1`, `RLINF_TIMELINE_ACTOR_TRAINING=1`, `RLINF_TIMELINE_DIR=auto`, `RLINF_NVITOP=1`, `RLINF_NVML=1`) by default. The CLI exposes `--no-profiler` (opts out of all `RLINF_TIMELINE*` flags) and `--no-collect-memory` (opts out of `RLINF_NVITOP`/`RLINF_NVML` only). On trial failure the runner performs (a) subprocess process-group SIGTERM then SIGKILL, (b) invokes a configurable `ray stop --force` cleanup hook, (c) verifies via `pgrep -f "RLINF_TUNER_TRIAL_ID=<trial_idx>"` (scoped to this trial via the unique env tag the runner exports) that no orphan workers remain, (d) records cleanup outcome in the trial ledger.
  - Positive Tests:
    - A normal trial run with default flags produces `LOG_DIR/timeline/*.jsonl` AND `LOG_DIR/nvitop/`; the runner returns the correct `LOG_DIR`; the ledger entry records `cleanup_outcome: "ok"`; the `RLINF_TUNER_TRIAL_ID` env tag appears in the subprocess environment.
    - A trial launched with `--no-collect-memory` produces `LOG_DIR/timeline/*.jsonl` but NO `LOG_DIR/nvitop/`.
  - Negative Tests:
    - A trial launched with `--no-profiler --no-collect-memory` produces a `LOG_DIR` with NO `timeline/` and NO `nvitop/` and is later classified `(OK, METRICS_PARTIAL)` by the parser (AC-6).
    - A trial that deliberately sleeps past the timeout is killed; `pgrep -f "RLINF_TUNER_TRIAL_ID=<trial_idx>"` returns empty after cleanup; the next trial starts without inheriting orphan workers.
    - A trial whose subprocess refuses SIGTERM is escalated to SIGKILL and the ledger records `cleanup_outcome: "sigkill_required"`.

- AC-6: Log/timeline parser produces a structured `(Status, FailureMode, Objective)` result for every trial directory. `Status ∈ {OK, FAILED}`. `FailureMode ∈ {NONE, METRICS_PARTIAL, METRICS_MISSING, CONFIG_INVALID, LAUNCH_FAILURE, OOM, WORKER_CRASH, TIMEOUT}`. `(OK, NONE)` means complete data including `metrics.log` MetricTable AND `timeline/*.jsonl`. `(OK, METRICS_PARTIAL)` means trial completed but some non-fatal data is absent (e.g. missing `num_trajectories` row, missing `timeline/` because `--no-profiler` was set, only step 1 was produced). `METRICS_MISSING` means `metrics.log` itself is absent or yields no usable objective. `(FAILED, NONE)` is invalid by construction (every FAILED result must carry a non-NONE FailureMode). OOM detection rubric: subprocess nonzero exit AND any of (a) `stderr` matches `CUDA out of memory` or `torch.cuda.OutOfMemoryError`, or (b) Ray actor-death log lines mention an OOM keyword. Memory telemetry from `nvitop/` is best-effort metadata, NOT a feasibility constraint. The parser also computes the trial objective as `step_time / num_trajectories` where `step_time` is averaged across steps 2-N (step 1 dropped as warmup; default `max_epochs=3` yields steps 2-3) and `num_trajectories` is taken from the FINAL MetricTable block in `metrics.log` (last `num_trajectories=N` line, no silent fallback denominator). Best-config selection picks the trial with the lowest objective among trials whose `(Status, FailureMode) == (OK, NONE)`; trials with `(OK, METRICS_PARTIAL)` are usable as critic context but INELIGIBLE for best-config selection.
  - Positive Tests:
    - A successful real trial with profiler ON yields `(OK, NONE)`, a full MetricTable plus per-component timeline summary, and a non-null objective.
    - Synthetic 3-step `metrics.log` with per-step times and final `num_trajectories=18` yields the expected averaged objective.
    - Among synthetic trials `{(OK,NONE, obj=50), (OK,NONE, obj=30), (OK,METRICS_PARTIAL, obj=20), (FAILED,OOM, obj=None)}`, best-config selection returns trial 2 (objective=30), NOT trial 3 (excluded by partial) and NOT trial 4 (excluded by failure).
    - The parser emits a structured per-component timeline summary (per-rank min/median/max for `env_interact_step`, `generate_one_epoch`, `run_training`, `sync_weights`, plus stall-fraction and call counts) consumable by AC-7.
  - Negative Tests:
    - A trial directory with no `metrics.log` is classified `(FAILED, METRICS_MISSING)` without raising.
    - A synthetic stderr containing `CUDA out of memory` plus nonzero exit produces `(FAILED, OOM)`.
    - A trial with `metrics.log` present but no `num_trajectories=` line is flagged `(OK, METRICS_PARTIAL)` and excluded from best-config selection.
    - A trial completing only step 1 (1-epoch run) is flagged `(OK, METRICS_PARTIAL)` because warmup exclusion leaves zero data points.
    - Returning `(FAILED, NONE)` from the parser triggers an internal invariant error in tests (sentinel for the "every FAILED has a reason" rule).

- AC-7: LLM critic prompt builder emits a prompt that contains (i) a bottleneck rubric mapping `env vs rollout vs actor dominates -> propose specific knob` (encodes Alt-2 from the draft), (ii) the last K=8 trials' summarized MetricTable + per-component timeline summary (from AC-6) + outcomes + critic rationale (or an explicit empty-history marker on round 0), (iii) current knob values + legal ranges from `KnobSchema`, (iv) hard constraints paraphrased from `validate_cfg`/`validate_embodied_cfg`, (v) a memory-pressure flag when the last trial was OOM. Critic output is parsed as structured JSON `{delta: {knob: value, ...}, rationale: {summary: "...", metric_table_citations: ["..."], timeline_citations: ["..."]}}`. A critic-output validator enforces a dual-source rationale rule: when the proposed `delta` touches `cluster.component_placement`, the rationale MUST contain AT LEAST ONE non-empty `metric_table_citations` entry AND AT LEAST ONE non-empty `timeline_citations` entry; otherwise the output is rejected. Invalid output (malformed JSON OR missing-source-rationale on a placement delta) triggers up to 3 retries with feedback. Scheduler reactivity is tested with a deterministic `fake_critic`, NOT the live LLM.
  - Positive Tests:
    - A snapshot test verifies the prompt string contains every required section in order: rubric, history (or empty marker), current values, constraints, memory-pressure flag (when applicable), and a per-component timeline summary block.
    - With a `fake_critic` that returns a delta reducing `actor.micro_batch_size` whenever the previous trial showed actor-dominant time, a synthetic high-actor-time trial leads the scheduler to apply that delta in the next iteration.
    - With a `fake_critic` that returns a placement delta accompanied by both `metric_table_citations` (e.g. `["env/interact=120s"]`) and `timeline_citations` (e.g. `["env_rank0:env_interact_step median=15s, stall_fraction=0.4"]`), the validator accepts the output and the scheduler applies the delta.
  - Negative Tests:
    - A `fake_critic` that returns malformed JSON triggers 3 retries with feedback strings appended, then fails the trial proposal without launching GPU.
    - A `fake_critic` that returns a placement-touching delta with empty `timeline_citations` (only MetricTable rationale) is rejected by the dual-source validator and triggers retries.
    - A `fake_critic` that returns a placement-touching delta with empty `metric_table_citations` (only timeline rationale) is also rejected.
    - Empty trial history is rendered as an explicit `## Trial History\n(none — first round)` marker, not silently omitted.

- AC-8: Scheduler honors the trial budget (`max_trials=20`, `budget_seconds=43200`, `max_oom=5`; CLI-overridable) and the stopping rule: terminate when ANY of (max_trials reached, budget_seconds elapsed, max_oom exceeded, the last `patience=3` non-failed trials show relative improvement < `epsilon=0.02`, or the critic returns `"no_further_improvement"` twice in a row).
  - Positive Tests:
    - A budget of 3 trials with a stub runner terminates after exactly 3 trials.
    - A 5-trial run where trials 3, 4, 5 each improve objective by < 2% terminates at trial 5 (plateau).
    - A `fake_critic` that returns `"no_further_improvement"` on trials 4 and 5 terminates the loop at trial 5.
  - Negative Tests:
    - A budget of 0 trials terminates without launching any trial and emits a structured "no trials run" report.
    - Exceeding `max_oom=5` with a mock-OOM runner terminates the loop with stop reason `oom_cap_exceeded`.

- AC-9: Trial ledger persists every trial as append-only JSONL at `LOG_DIR/../tuner_ledger.jsonl` with the fields: `trial_idx, delta, resolved_config_sha, log_dir, returncode, status, failure_mode, objective, step_time, num_trajectories, per_component_timings, timeline_summary, peak_gpu_mem (nulla/ble), critic_rationale, ts_start, ts_end`. `resolved_config_sha` is the SHA-256 of `OmegaConf.to_yaml(resolved_cfg, sort_keys=True)` computed after Hydra composition in preflight. `critic_rationale` stores the structured `{summary, metric_table_citations, timeline_citations}` payload from AC-7 so historical placement decisions remain auditable.
  - Positive Tests:
    - All ledger entries round-trip via `json.loads(json.dumps(entry)) == entry` for a 5-trial stub run.
    - A simulated mid-loop crash (SIGKILL of the orchestrator between trials 3 and 4) leaves trials 1-3 readable in the ledger.
    - Two trials with identical resolved configs produce identical `resolved_config_sha` values.
    - Replaying the ledger for a trial whose delta touched placement shows the `metric_table_citations` and `timeline_citations` that drove the decision.
  - Negative Tests:
    - A single corrupted line in the ledger (manually truncated) does not prevent loading subsequent valid lines.
    - A trial missing a required field is rejected by the ledger schema check before write.

- AC-10: CLI entrypoint `python -m toolkits.embodied_tuner` accepts `--config <config_name>` (defaults to `maniskill_ppo_openvla`), `--baseline <path>` (defaults to the matching file under `RLinf/examples/embodiment/config/`), `--max-trials`, `--budget-seconds`, `--max-oom`, `--patience`, `--epsilon`, `--max-epochs` (default 3), `--collect-memory`/`--no-collect-memory`, `--no-profiler`, `--dry-run-preflight`. The module invocation path resolves against `RLinf/` as the PYTHONPATH root, matching `RLinf/examples/embodiment/run_placement_autotune.sh`. A shim launcher at `RLinf/examples/embodiment/run_embodied_tuner.sh` exports `REPO_PATH` and `PYTHONPATH=${REPO_PATH}:${PYTHONPATH}` and forwards arguments to the CLI. The orchestrator emits `best_config.yaml` and `best_trial.json` at deterministic locations alongside `tuner_ledger.jsonl`.
  - Positive Tests:
    - A CLI run with `fake_critic` and a mock trial runner on a stub config produces `best_config.yaml` parseable as YAML and `best_trial.json` containing `{objective, denominator_source, step_range_used, exclusion_reasons, source_trial_idx}`.
    - `--dry-run-preflight` exits cleanly after composing the config and running validators, without spawning the trial subprocess.
    - `--baseline /tmp/custom_stub.yaml` is honored when the file exists and parses.
    - `bash RLinf/examples/embodiment/run_embodied_tuner.sh --help` prints the CLI usage.
    - Invoking the shim with `--dry-run-preflight --config maniskill_ppo_openvla` exits cleanly after preflight.
  - Negative Tests:
    - Missing `--config` fails with a structured error mentioning the missing flag.
    - An unreadable `--baseline` path fails before launch with a structured error mentioning the path.
    - Invoking `python -m toolkits.embodied_tuner --help` from a working directory without `RLinf/` on `PYTHONPATH` fails with a clear `ModuleNotFoundError` (or `ImportError`), and the shim preflight detects this and prints a structured remediation hint.

- AC-11: End-to-end smoke test using `fake_critic` and a mock trial runner (no real GPU) verifies that the orchestrator runs N=4 synthetic trials, the ledger is well-formed, `best_config.yaml` and `best_trial.json` are emitted, a mocked OOM trial does not poison subsequent trials, and a deliberate `from toolkits.auto_placement import DataFitter` planted in any file under `RLinf/toolkits/embodied_tuner/` is caught by the import-boundary sub-test (an AST walker bundled into the same `tests/` directory that scans every Python file in the new toolkit and asserts no `import` or `from ... import` statement references `toolkits.auto_placement` or the bare name `auto_placement`).
  - Positive Tests:
    - The smoke test passes in CI within seconds.
    - The emitted best trial corresponds to the synthetic trial with the lowest objective among `(OK, NONE)` trials.
    - The import-boundary AST walker passes on the clean implementation with zero violations.
  - Negative Tests:
    - A smoke test variant that mocks an OOM at trial 2 still completes trials 3 and 4 and selects the best among the eligible ones.
    - A smoke test variant that mocks a parser crash on trial 1 marks trial 1 `(FAILED, *)` and continues.
    - A deliberate `from toolkits.auto_placement import DataFitter` added under `RLinf/toolkits/embodied_tuner/` causes the import-boundary sub-test to fail with a message naming the offending file and line.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The implementation includes the full orchestrator with: knob schema (with `pinned` markers for the FUT-5 knobs), placement enumerator, Hydra-override wrapper, preflight validator (calling `validate_cfg` + `validate_embodied_cfg` + placement check), trial runner with timeout + scoped cleanup + ray-stop hook + default-on profiler env exports + opt-out flags, log+timeline parser with full `(Status, FailureMode)` taxonomy, OOM detection rubric, objective computation, best-config selection, and a per-component timeline summary builder, LLM critic prompt builder with bottleneck rubric, full trial-history block, timeline-summary block, memory-pressure flag, structured-rationale schema, AND a dual-source rationale validator that rejects placement-touching deltas without paired MetricTable and timeline citations, deterministic `fake_critic` for tests, scheduler with budget + plateau + critic-stagnation stopping, append-only JSONL ledger with SHA-256 resolved-config hashing and full critic-rationale persistence, CLI entrypoint with all flags listed in AC-10, shim launcher script with PYTHONPATH-missing remediation hint, smoke test using `fake_critic` and mock runner, AND a bundled import-boundary AST walker as a sub-test under the smoke test directory. All ACs verified.

### Lower Bound (Minimum Acceptable Scope)
The implementation includes the core path: knob schema, Hydra-override wrapper, preflight validator, trial runner with timeout, basic cleanup, and default-on profiler env exports, log/timeline parser handling at least `(OK, NONE)`, `(OK, METRICS_PARTIAL)`, `(FAILED, OOM)`, `(FAILED, METRICS_MISSING)`, `(FAILED, TIMEOUT)`, objective computation, best-config selection, AND a per-component timeline summary builder, LLM critic prompt builder with bottleneck rubric and timeline-summary block, structured-rationale schema, dual-source validator on placement deltas, and `fake_critic`, scheduler honoring `max_trials` + `max_oom` + plateau, JSONL ledger with the required fields and critic-rationale persistence, CLI entrypoint, shim launcher, smoke test, bundled import-boundary AST sub-test. All ACs still verified, but the implementation can omit optional polish such as `--dry-run-preflight`, the explicit empty-history marker variant tests, and the multi-baseline `--baseline` arbitrary path support beyond the default.

### Allowed Choices
- Can use:
  - Python 3.10+ stdlib (`subprocess`, `signal`, `pathlib`, `json`, `dataclasses`, `argparse` or `click`).
  - `OmegaConf` and `hydra-core` (already RLinf dependencies) for config composition and SHA-256 input formatting.
  - `pytest` for tests (matches `RLinf/tests/` convention).
  - The existing `ask-codex.sh` script in `humanize/` for the LLM critic; the project root must be passed via `CLAUDE_PROJECT_DIR` (or invoked from inside the repo) so `ask-codex.sh` can locate it.
  - Existing RLinf modules read-only: `rlinf.config.validate_cfg`, `rlinf.config.validate_embodied_cfg`, `rlinf.utils.placement.ModelParallelComponentPlacement`, `rlinf.utils.metric_utils` keys.
  - Existing profiler env vars from `profiler/enable2.sh` plus the two `RLINF_NVITOP`/`RLINF_NVML` flags that script leaves commented.
- Cannot use:
  - Any `import` from `toolkits.auto_placement` or its submodules (`AutoPlacementWorker`, `DataFitter`, `EnvProfiler`, `MegatronNode`, etc.). The user has explicitly excluded reuse for embodied tuning. This rule is enforced by the AST walker bundled into the smoke-test suite (AC-11).
  - New required dependencies (Optuna, SMAC, DEAP, scikit-learn). `scipy` and `numpy` are already present and may be used; nothing else may be added to `pyproject.toml`.
  - The standard humanize RLCR loop as the per-trial execution loop. The tuner's per-trial loop is plain Python.
  - Modifying `RLinf/examples/embodiment/run_embodiment.sh` itself. The wrapper composes a sibling launcher or shim; the stock script remains untouched.
  - Allowing a placement-touching critic delta to enter preflight without a paired MetricTable citation AND timeline citation in its rationale. The critic-output validator (AC-7) rejects such deltas mechanically.

> **Note on Deterministic Designs**: The draft pins the search target (`step_time / num_trajectories`), the toolkit location (`RLinf/toolkits/embodied_tuner/`), the knob set, and the prohibition on reusing `auto_placement`. The user-requested refinement further pins the evidence rule for placement decisions (dual-source rationale). These narrow the path boundaries: the implementation MAY choose internal module decomposition and test framing, but MUST NOT change the toolkit location, the knob set, the auto_placement prohibition, or the dual-source placement-rationale rule.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

```
class KnobSchema:                     # AC-1
    domains: dict[str, KnobDomain]    # one entry per tunable knob; pinned knobs are reserved
    def validate(delta) -> Result

class PlacementEnumerator:            # AC-4
    def enumerate(num_gpus=8) -> list[PlacementSpec]
    # respects ModelParallelComponentPlacement contiguity + non-overlap

class OverrideWrapper:                # AC-2
    def build_launcher(baseline_cfg, overrides) -> shim_path, log_dir
    # composes a sibling shim that invokes train_embodied_agent.py with
    # ${overrides} appended to the stock --config-name <name> runner.logger.log_path=$LOG_DIR

class Preflight:                      # AC-3
    def compose_and_validate(baseline, delta) -> (resolved_cfg, sha256, ValidationResult)
    # Hydra compose -> validate_cfg -> validate_embodied_cfg -> placement check

class TrialRunner:                    # AC-5
    def run(resolved_cfg, log_dir, timeout, profile_env) -> (returncode, log_dir)
    # default exports RLINF_TIMELINE*, RLINF_NVITOP, RLINF_NVML; opt-out flags supported.
    # exports RLINF_TUNER_TRIAL_ID=<idx>; on timeout/SIGTERM/SIGKILL escalates;
    # invokes ray_stop_hook; pgrep -f RLINF_TUNER_TRIAL_ID=<idx> must return empty.

class LogParser:                      # AC-6
    def parse(log_dir) -> TrialResult(status, failure_mode, objective,
                                      metric_table_summary, timeline_summary, ...)
    # final MetricTable block; OOM rubric; (OK, METRICS_PARTIAL) when timeline absent.
    # timeline_summary holds per-rank min/median/max + stall fractions per component.
    def select_best(trials) -> TrialResult | None
    # picks lowest objective among (OK, NONE) trials only.

class CriticOutputValidator:          # AC-7
    def validate(delta, rationale) -> Result
    # if delta touches cluster.component_placement:
    #   require >=1 non-empty metric_table_citation AND >=1 non-empty timeline_citation
    # else: any non-empty rationale.summary is sufficient.

class Critic:                         # AC-7
    def propose(history, current_knobs, memory_pressure,
                metric_table_summary, timeline_summary) -> Delta
    # Real impl shells out to ask-codex.sh with the structured prompt;
    # parses {delta, rationale={summary, metric_table_citations, timeline_citations}};
    # CriticOutputValidator runs before returning; up to 3 retries with feedback.
    # fake_critic returns deterministic deltas (and structured rationale for tests).

class Scheduler:                      # AC-8
    def run(budget, stopping_rule) -> CampaignResult
    # loops: preflight -> trial -> parse -> ledger.append -> critic.propose -> next.

class Ledger:                         # AC-9
    def append(entry) -> None
    def load() -> list[TrialEntry]
    def best() -> TrialEntry | None   # picks lowest objective among (OK, NONE)
    # entry.critic_rationale persists {summary, metric_table_citations, timeline_citations}.

# CLI: python -m toolkits.embodied_tuner   (AC-10)
# Shim: RLinf/examples/embodiment/run_embodied_tuner.sh   (AC-10)
# Tests: AC-11 smoke (fake_critic + mock runner + bundled import-boundary AST walker).
```

Per-trial flow:
1. Critic proposes Delta with structured rationale (round 0 = empty delta = baseline; or a small seed perturbation). CriticOutputValidator enforces dual-source rationale for placement-touching deltas.
2. Preflight composes + validates; on failure record `(FAILED, CONFIG_INVALID)` and ask critic for a new delta (NOT counted as a real trial against budget, with a configurable retry cap).
3. TrialRunner launches subprocess with profiler env + RLINF_TUNER_TRIAL_ID tag, awaits with timeout.
4. Parser reads `LOG_DIR/metrics.log` + `LOG_DIR/timeline/*.jsonl` (best-effort `LOG_DIR/nvitop/`), builds metric_table_summary AND timeline_summary for the critic to consume next round.
5. Ledger appends (including structured critic_rationale so placement decisions remain auditable).
6. Scheduler checks stopping rule; if not stopped, hand history + summaries to Critic and loop.

### Relevant References

- `RLinf/examples/embodiment/run_embodiment.sh` — stock entry script (only injects `runner.logger.log_path=${LOG_DIR}`; the wrapper extends overrides).
- `RLinf/examples/embodiment/train_embodied_agent.py` — Hydra-driven training entry; the override wrapper targets this binary.
- `RLinf/examples/embodiment/config/maniskill_ppo_openvla.yaml` — default baseline; exposes all tunable knobs.
- `RLinf/rlinf/config.py` — `validate_cfg` and `validate_embodied_cfg` containing the divisibility assertions the preflight enforces.
- `RLinf/rlinf/utils/placement.py` — `ModelParallelComponentPlacement` (contiguity + non-overlap rules); `HybridComponentPlacement` (mode auto-detect; does NOT itself enforce legality).
- `RLinf/rlinf/utils/metric_utils.py` — MetricTable rendering and `Step Time = elapsed_time / steps_done` semantics; the parser must NOT double-divide by steps.
- `RLinf/rlinf/runners/embodied_runner.py` — `num_steps_per_epoch = 1`, so `max_epochs=N` yields N global steps (informs the `--max-epochs=3` default with warmup drop).
- `profiler/enable2.sh` — pattern for the timeline env vars; the tuner adds the two memory-telemetry flags this script leaves commented.
- `profiler/rlinf_timeline/autopatch.py` and `profiler/rlinf_timeline/writer.py` — JSONL format the parser consumes for per-component timing and stall analysis.
- `RLinf/examples/embodiment/run_placement_autotune.sh` — invocation pattern for the shim launcher (`python ${REPO_PATH}/toolkits/<name>/...`).
- `RLinf/toolkits/auto_placement/` — read-only reference; MUST NOT be imported by the new toolkit (enforced by AST walker bundled under AC-11).
- `humanize/commands/start-rlcr-loop.md` — the RLCR machinery that BUILDS this tuner.
- `humanize/commands/ask-codex.md` and `scripts/ask-codex.sh` — the LLM critic transport.
- `mlsys2026-flashinfer-contest/` — precedent flow: `gen-idea -> gen-plan -> start-rlcr-loop`.
- `RLinf/logs/20260629-07:25:33-maniskill_ppo_openvla/` — current log shape; contains `metrics.log` + `timeline/*.jsonl` + `tensorboard/`, no `nvitop/` (verified, because `enable2.sh` leaves NVITOP flag commented).
- `RLinf/demo_logs/maniskill_ppo_openvla_envs192_mbs80_env0-1_roll2-7_act0-7.log` — historical MetricTable example with `num_trajectories=448`.

## Dependencies and Sequence

### Milestones

1. Milestone A — Foundations (no GPU required).
   - Step A1: Knob schema (`schema.py`) + tests. (task1)
   - Step A2: Placement enumerator (`placement_enum.py`) + tests. (task4)
   - Step A3: Override wrapper (`override_wrapper.py`) + tests with the stock script. (task2)
   - Step A4: Preflight validator (`preflight.py`) using `validate_cfg` + `validate_embodied_cfg` + placement check. (task3)

2. Milestone B — Trial execution + parsing.
   - Step B1: Trial runner (`runner.py`) with timeout + scoped cleanup + `RLINF_TUNER_TRIAL_ID` env tag + default-on profiler env exports. (task5)
   - Step B2: Log+timeline parser (`parser.py`) with full taxonomy, OOM rubric, objective computation, best-config selection, AND per-component timeline summary builder. (task6)

3. Milestone C — Loop control + critic.
   - Step C1: Trial ledger (`ledger.py`) with SHA-256 resolved-config hashing and structured critic-rationale persistence. (task9)
   - Step C2: Critic prompt builder (`critic.py`) with bottleneck rubric + timeline summary + dual-source rationale validator + `fake_critic.py`. (task7)
   - Step C3: Scheduler (`scheduler.py`) with budget + plateau + critic-stagnation. (task8)

4. Milestone D — User-facing surface + guards.
   - Step D1: CLI entrypoint (`__main__.py`) + best-config + best_trial emitters + shim launcher script. (task10)
   - Step D2: End-to-end smoke test with `fake_critic` + mock runner + bundled import-boundary AST walker as a sub-test. (task11)

Dependencies between components (not time):
- preflight depends on knob schema, placement enumerator.
- runner depends on override wrapper (profiler env exports live inside the runner now).
- parser depends on runner output convention (LOG_DIR layout).
- critic depends on knob schema, parser (consumes the timeline summary the parser builds).
- scheduler depends on critic, runner, parser, ledger, preflight.
- CLI depends on scheduler + ledger + best-config emitter.
- smoke test (including the import-boundary AST sub-test) depends on the toolkit package existing.

## Task Breakdown

Each task is tagged `coding`. Every `AC-*` is covered by at least one task and every task targets a current-scope `AC-*`.

| Task ID | Description                                                                                       | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|---------------------------------------------------------------------------------------------------|-----------|---------------------------|------------|
| task1   | Implement `schema.py` (knob dataclass + legal-domain + dependency declarations + pinned-knob markers) + unit tests. | AC-1 | coding | -          |
| task2   | Implement `override_wrapper.py` (shim that forwards arbitrary Hydra overrides and returns LOG_DIR) + tests including Hydra precedence and `LOG_DIR` capture. | AC-2 | coding | task1 |
| task3   | Implement `preflight.py` (Hydra compose + `validate_cfg` + `validate_embodied_cfg` + placement check) + tests. | AC-3 | coding | task1, task4 |
| task4   | Implement `placement_enum.py` (legal placement enumeration for 8 GPUs, respecting `ModelParallelComponentPlacement` rules) + tests. | AC-4 | coding | task1 |
| task5   | Implement `runner.py` (subprocess, timeout, SIGTERM->SIGKILL, ray_stop hook, scoped `pgrep -f RLINF_TUNER_TRIAL_ID=...` cleanup verification, default-on profiler env exports, `--no-profiler`/`--no-collect-memory` opt-outs) + tests. | AC-5 | coding | task2 |
| task6   | Implement `parser.py` (`metrics.log` + `timeline/*.jsonl`, `(Status, FailureMode)` taxonomy, OOM detection rubric, METRICS_PARTIAL vs METRICS_MISSING distinction, objective computation with warmup exclusion, best-config selection, AND per-component timeline summary builder consumed by AC-7) + tests including objective-correctness and best-config-selection cases. | AC-6 | coding | task5 |
| task7   | Implement `critic.py` (prompt builder with bottleneck rubric, history, current knobs, constraints, memory-pressure flag, timeline summary block) + structured-rationale JSON schema + dual-source rationale validator for placement deltas + `fake_critic.py` + snapshot tests + scheduler-reactivity tests + validator-rejection tests (placement delta missing one citation source). | AC-7 | coding | task1, task6 |
| task8   | Implement `scheduler.py` (loop, budget, plateau patience + epsilon, critic-stagnation stop) + tests. | AC-8 | coding | task3, task5, task6, task7, task9 |
| task9   | Implement `ledger.py` (append-only JSONL, SHA-256 of `OmegaConf.to_yaml(resolved_cfg, sort_keys=True)`, structured critic_rationale persistence, schema check) + tests including mid-loop crash recovery and rationale round-trip. | AC-9 | coding | task3 |
| task10  | Implement `__main__.py` (CLI with all flags listed in AC-10) + best-config emitter + `best_trial.json` writer + shim launcher `RLinf/examples/embodiment/run_embodied_tuner.sh` (exports `REPO_PATH`, `PYTHONPATH`, prints PYTHONPATH-missing remediation hint) + CLI/module-invocation tests. | AC-10 | coding | task8, task9 |
| task11  | Implement end-to-end smoke test using `fake_critic` and mock trial runner (N=4 trials including mocked OOM and mocked parser-crash) + bundle an import-boundary AST-walker sub-test in the same `tests/` directory that scans every Python file under `RLinf/toolkits/embodied_tuner/` and fails on any reference to `toolkits.auto_placement` or the bare `auto_placement` name. | AC-11 | coding | task10 |

## Future Work / Out of Scope

Future, deferred, post-work, successor-loop, and out-of-scope items.

- FUT-1: Multi-node placement support. Current loop is single-node 8xA800 only.
  - Source DEC: scope decision committed in Goal (single-node only).
  - Current-loop handoff: none.
  - Promotion trigger: when the user adopts multi-node embodied training and a cross-node placement notation needs to enter `cluster.component_placement`.
- FUT-2: Async pipelined trials (multiple trials in flight, e.g. trial N+1 preflight while trial N executes).
  - Source DEC: none (explicit scope choice).
  - Current-loop handoff: none.
  - Promotion trigger: when wall-clock per tuning campaign exceeds an operator threshold the synchronous loop can no longer absorb.
- FUT-3: Bayesian / evolutionary fallback proposer that swaps in for the LLM critic when patience is exhausted (per draft Alts 1 and 4).
  - Source DEC: none.
  - Current-loop handoff: AC-7 already abstracts critic behind a `Critic` interface, so the swap is mechanical.
  - Promotion trigger: when LLM critic plateau patience is exhausted in a single campaign and the user wants a second proposer to try.
- FUT-4: Parametric-surrogate warm start (per draft Alt-3): a calibration sweep fits `env_time = a·N + b` and the actor U-shape, seeds the first critic round with quantitative priors.
  - Source DEC: none.
  - Current-loop handoff: ledger schema already records per-component timings + timeline summaries sufficient to fit surrogates offline.
  - Promotion trigger: when enough historical campaigns accumulate to fit a surrogate.
- FUT-5: Tunable `actor.global_batch_size`, `rollout.pipeline_stage_num`, `actor.num_action_chunks`.
  - Source DEC: knob domain commitment in Goal (pinned for tractable `validate_cfg` ripples).
  - Current-loop handoff: knob schema (AC-1) reserves these names with a `pinned=True` marker; un-pinning is a schema flip + extra divisibility solver.
  - Promotion trigger: when divisibility wrangling logic is added to handle the cross-knob ripples.
- FUT-6: Evaluation / checkpoint during trials. Current trials skip eval and checkpoint to keep step-time signal clean.
  - Source DEC: none.
  - Current-loop handoff: none.
  - Promotion trigger: when the user wants the tuner output to include quality metrics alongside throughput.
- FUT-7: Cross-config tuning (transfer tuning state across embodiments, e.g. seeding a `libero_*` tuning campaign from a `maniskill_*` ledger).
  - Source DEC: none.
  - Current-loop handoff: ledger JSONL is self-contained per campaign.
  - Promotion trigger: when multiple embodiments are tuned and operators want transfer.
- FUT-8: Deterministic memory-shrink retry on OOM. Current loop classifies OOM, counts toward `max_oom`, prompts the critic with a memory-pressure flag, and does NOT retry the same delta.
  - Source DEC: OOM-behavior default committed in plan body during convergence (AC-6 + scheduler section).
  - Current-loop handoff: AC-6 records OOM with detection rubric; AC-7 prompts critic with memory-pressure flag.
  - Promotion trigger: when operators want the tuner to mechanically shrink memory-heavy knobs and retry, rather than waiting on the critic.
- FUT-9: Additional baseline configs beyond `maniskill_ppo_openvla` (e.g. `maniskill_ppo_openpi`, `libero_10_grpo_openpi`, `calvin_*`). Current `--baseline` accepts arbitrary paths but explicit per-config testing is FUT.
  - Source DEC: scope decision committed in Phase 6 (single-baseline acceptance scope).
  - Current-loop handoff: AC-10 verifies `--baseline` accepts an arbitrary stub path; the loop is config-agnostic where validators agree.
  - Promotion trigger: when operators run the tuner against a non-maniskill embodied config and find a config-specific gap.

## Claude-Codex Deliberation

### Agreements

- Build the tuner ONCE via humanize+RLCR; the tuner's per-trial loop is plain Python, not RLCR. Avoids the "one round = whole plan finished" mismatch the draft flagged.
- `RLinf/toolkits/auto_placement/` is read-only reference and must NOT be imported, given its cold-start dependency on `config.profile_data` that embodied configs lack.
- `actor.global_batch_size`, `rollout.pipeline_stage_num`, and `actor.num_action_chunks` are pinned in this loop to keep `validate_cfg` divisibility ripples tractable.
- Placement legality is enforced against `ModelParallelComponentPlacement`'s contiguous-GPU + non-overlap rules; `HybridComponentPlacement` itself only sets mode and does not enforce legality.
- AC/Task bidirectional coverage holds: after the user-requested AC consolidation, 11 ACs map to 11 tasks, all targeting current-scope ACs only.

### Resolved Disagreements

- Codex disagreed that "one RLCR round = one RLinf trial" was workable. Resolution: Goal commits to RLCR-as-builder; the tuner's per-trial loop is plain Python.
- Codex disagreed that the sample denominator should be configured `rollout_epoch * total_num_envs`. Resolution: objective is `step_time / num_trajectories` parsed from the FINAL MetricTable block in `metrics.log`. Configured-count fallback is explicitly forbidden.
- Codex disagreed that `max_epochs=1` was a usable signal. Resolution: tuner injects `runner.max_epochs=3` Hydra override by default, drops step 1 as warmup, averages steps 2-3. CLI exposes `--max-epochs`.
- Codex disagreed that AC-2's negative test should hinge on the stock script's inability to forward overrides. Resolution: AC-2 negative tests target the WRAPPER's behavior on invalid overrides and add a Hydra precedence test.
- Codex disagreed that "timeline always on" was implementable alongside a `--no-profiler` opt-out. Resolution: profiler env exports (now folded into AC-5) are default-on with explicit `--no-profiler`/`--no-collect-memory` opt-outs; partial-telemetry trials are flagged `(OK, METRICS_PARTIAL)` and excluded from best-config selection.
- Codex disagreed that `(Status, FailureMode)` semantics were precise enough. Resolution: explicit grid in AC-6 — `(OK, NONE)` eligible, `(OK, METRICS_PARTIAL)` usable-but-ineligible, `(FAILED, NONE)` invalid by construction.
- Codex disagreed that "clean up Ray + GPU" was deterministic. Resolution: scoped `pgrep -f "RLINF_TUNER_TRIAL_ID=<idx>"` plus configurable `ray stop --force` hook plus explicit SIGTERM→SIGKILL escalation plus cleanup-outcome ledger record.
- Codex disagreed that "synthetic high-actor-time trial → critic reduces actor work" depended on a live LLM. Resolution: scheduler reactivity is tested with deterministic `fake_critic`; prompt content is tested with snapshot-style assertions.
- Codex disagreed that an import-guard could be a one-shot grep. Resolution: it is now an AST walker bundled into the smoke-test suite (AC-11).
- Codex disagreed that the CLI module path was specified. Resolution: AC-10 commits `python -m toolkits.embodied_tuner` against `RLinf/` PYTHONPATH plus a shim launcher script.
- Codex disagreed that OOM default behavior was a real PENDING decision once architecture/scheduler/parser had committed it. Resolution: deterministic memory-shrink retry pulled into FUT-8.

### Refinements from User Comments

- CMT-1.1 (`删除AC-13`): The standalone import-boundary AC was deleted; the safeguard test was folded into AC-11 as a bundled AST-walker sub-test under the smoke-test suite, and the import prohibition remains in Path Boundaries.
- CMT-1.2 (`适当减少AC的总数量`): Three natural consolidations were applied — profiler-env enablement merged into AC-5 (trial runner), objective computation + best-config selection merged into AC-6 (parser), module invocation path + shim launcher merged into AC-10 (CLI). Net AC count went from 15 to 11; task count likewise.
- CMT-1.3 (`强化RLCR循环过程中放置优化决定基于log+timeline`): AC-7 now requires the critic prompt to include a per-component timeline summary block (produced by AC-6), requires the critic output to follow a structured `{summary, metric_table_citations, timeline_citations}` rationale schema, and adds a `CriticOutputValidator` that mechanically rejects placement-touching deltas whose rationale lacks BOTH a non-empty MetricTable citation AND a non-empty timeline citation. Goal Description and Path Boundaries also encode the dual-source rule. AC-9 persists the structured rationale in the ledger so historical placement decisions remain auditable.

### Convergence Status

- Final Status: `converged`
- Rounds executed: 3 (during gen-plan). Refinement applied user-requested changes consistently across all sections; no new disagreements introduced.

## Pending User Decisions

(None. All decisions from drafting, convergence, and the user-requested refinement are committed in the plan body, with deferred alternatives linked to `FUT-*` entries.)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. name a function `select_best_trial`, not `best_select`; name the rationale validator `PlacementRationaleValidator`, not by an AC number).

### Repository Facts Worth Preserving in Code Comments (where the WHY is non-obvious)
- `run_embodiment.sh` injects only `runner.logger.log_path=${LOG_DIR}`; the override wrapper exists because of this.
- `EmbodiedRunner.num_steps_per_epoch = 1`, so `max_epochs=N` yields N global steps. The `--max-epochs=3` default + warmup drop exists because of this.
- `profiler/enable2.sh` leaves `RLINF_NVITOP=1` / `RLINF_NVML=1` commented; the tuner sets them by default.
- `auto_placement_worker.py` requires `config.profile_data.env_profile_data` / `rollout_profile_data` which embodied configs lack; this is why we do not import that toolkit. The bundled AST walker (AC-11) enforces this at test time.

### Codex Invocation Note
- `ask-codex.sh` requires `CLAUDE_PROJECT_DIR` to be set when invoked outside a trusted git repo, and requires `HUMANIZE_CODEX_BYPASS_SANDBOX=1` when the project directory is not registered as a Codex trusted directory. Document this in the critic implementation so deployment on fresh machines works.

### Placement Decision Audit Trail
- The structured `critic_rationale = {summary, metric_table_citations, timeline_citations}` is persisted in `tuner_ledger.jsonl` (AC-9). Operators reviewing a tuning campaign can grep the ledger to see WHICH MetricTable observation AND WHICH timeline observation drove each placement change. This audit trail is the implementation surface for the dual-source rule defined in the Goal Description and enforced by AC-7's validator.

--- Original Design Draft Start ---

# RLCR-Orchestrated Auto-Tuning Loop For RLinf Embodied Configs

## Original Idea

现在我想要在/mnt/public/zhouyiming/humanize/RLinf，使用humanize你这个plugin和RLCR，你可以学习一下mlsys2026这个工作使用humanize的方式，对RLinf进行迭代尝试式自动参数配置,可以配置的参数有config文件中的env actor 和rollout的placement方式，total_env_num和rollout_epcoh和micro batch size各个组件是否enable offload等, 核心目标是，在满足显存约束的情况下，获得最优的step time/ sample , RLinf/logs中有log的示例，在打开了profiler的情况下，log文件夹还会有详细的timeline等数据，虽然目前有RLinf有一个autoplacement的util，但是在具身下表现不好，且不能冷启动，不建议使用，如果坚持使用必须针对具身进行优化；初步的profile结果表示env rollout的运行时间会大致随着step生成样本数量的增长线性增长，ax+b；actor的时间会随着step生成样本数量的增长先线性增长，后非线性增长；随着microbatch size先减小后增大

for
	运行目标config的RLinf训练1轮
	获得step time/ per sample(rollout_epoch*total_env_num)(核心优化指标)
	获得profile的结果和log（包含step time、是否OOM或者挂掉）
	
	根据profile的结果分析bottleneck
	根据bottleneck调整config配置文件中的配置
	进入下一轮，重新运行

## Primary Direction: RLCR Trial-Refine Orchestrator

### Rationale

Drive the auto-tuning loop with humanize's existing RLCR (Review-Critique-Refine) machinery — each iteration runs one RLinf trial, ingests logs+profile, and an LLM critic proposes the next config delta. This is the path most aligned with the user's stated "use humanize + RLCR" mechanism.

### Approach Summary

Build an RLCR loop that drives iterative RLinf config tuning for embodied tasks. Each round: (1) the loop applies a YAML config delta to a target embodiment config file; (2) runs `bash examples/embodiment/run_embodiment.sh <config_name>` for 1 epoch (`max_epochs: 1`); (3) parses the resulting log under `logs/<timestamp>-<config_name>/` to extract the MetricTable (step time, env/interact, rollout/generate_one_epoch, actor/run_training) and the per-component timeline JSONL events from the `timeline/` subdirectory; (4) an LLM critic (Codex via `/humanize:ask-codex`) analyzes the logs, identifies the bottleneck component, and proposes a config delta adjusting `cluster.component_placement` (env/actor/rollout GPU ranges), `env.train.total_num_envs`, `env.train.rollout_epoch`, `actor.micro_batch_size`, and `enable_offload` flags for env/rollout/actor; (5) the delta is applied and the next round begins. The loop terminates when step_time / (rollout_epoch * total_num_envs) reaches a plateau or the user cancels. This uses humanize's existing `/humanize:start-rlcr-loop` machinery: the plan file enumerates tunable knobs and their valid ranges, and each round in the RLCR loop corresponds to one trial+critique cycle. The existing `rlinf_timeline` autopatch system and `MetricTable` rendering in `metric_utils.py` provide the structured data the critic needs without new instrumentation.

### Objective Evidence

- `/mnt/public2/zhouyiming/humanize/humanize/commands/start-rlcr-loop.md` -- the full RLCR loop command, including setup script, round definition ("one round = agent believes entire plan is finished"), goal tracker, task routing (coding/analyze), Codex review hooks. This is the core machinery to extend.
- `/mnt/public2/zhouyiming/humanize/humanize/commands/gen-idea.md` and `/mnt/public2/zhouyiming/humanize/humanize/commands/gen-plan.md` -- the idea-draft and plan-generation pipeline that precedes RLCR. The mlsys2026 contest used this exact flow: phase1.md -> `/humanize:gen-idea` -> `/humanize:gen-plan` -> `/humanize:start-rlcr-loop` with the plan file. See `/mnt/public2/zhouyiming/humanize/mlsys2026-flashinfer-contest/prompts/dsa-sparse-attention/phase1.md` through `phase3.md` for the precedent pattern of "write draft -> gen-plan -> start-rlcr-loop".
- `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/run_embodiment.sh` -- the entry script. It constructs the log directory as `logs/<timestamp>-<config_name>` and runs Hydra-based config resolution. This is the single invocation point per trial round.
- `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/config/maniskill_ppo_openvla.yaml` -- exemplar config showing all tunable knobs: `cluster.component_placement` (lines 18-21, GPU range assignments like "0-7", "0-3", "4-7"), `env.train.total_num_envs` (line 73), `env.train.rollout_epoch` (line 72), `env.train.enable_offload` (line 78), `rollout.enable_offload` (line 106), `actor.micro_batch_size` (line 116), `actor.enable_offload` (line 119). These are the knobs the critic would mutate.
- `/mnt/public2/zhouyiming/humanize/RLinf/demo_logs/maniskill_ppo_openvla_envs192_mbs80_env0-1_roll2-7_act0-7.log` -- contains the MetricTable output (lines 758-791) with `step=1473.3`, `env/interact=877.8`, `rollout/generate_one_epoch=876.2`, `actor/run_training=573.2`, `rollout/predict=451.5`, `sync_weights=7.346`. This is the structured data the critic parses. Line 761 shows `Step Time: 1473.315s`.
- `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/metric_utils.py` (lines 297-348) -- `print_metrics_table_async` renders the MetricTable with `Step Time` computed as `elapsed_time / steps_done`. The "Time" category includes `env/interact`, `rollout/generate_one_epoch`, `actor/run_training`, `step`, `sync_weights`.
- `/mnt/public2/zhouyiming/humanize/profiler/rlinf_timeline/autopatch.py` (1400+ LOC) -- the import-hook-based instrumentation that patches `Worker.timer`, `EmbodiedFSDPActor.run_training`, env/rollout workers at import time. Produces per-component JSONL timeline files (`timeline/actor_rank*.jsonl`, `timeline/env_rank*.jsonl`, `timeline/rollout_rank*.jsonl`) under the log directory. The timeline JSONL entries contain `t0/t1` timestamps, `tag` (e.g. `env_interact_step`, `generate_one_epoch`, `run_training`), `call_index`, `rollout_epoch`, `chunk_step`, `stage_id`, and exception info. This provides sub-step granularity the critic can use.
- `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/` (584 LOC total) -- existing auto-placement utility using `AutoPlacementWorker` with a workflow DAG (`env -> env_rollout -> actor`). The user explicitly states this does not work well for embodied tasks and cannot cold-start, so this is NOT the primary mechanism. However, it serves as prior art for the optimization problem formulation (cost models, GPU allocation, pipeline vs collocated scheduling).
- `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/placement.py` (975 LOC) -- `ModelParallelComponentPlacement` class handling COLLOCATED, DISAGGREGATED, HYBRID, AUTO modes. The `component_placement` YAML section directly maps to these placement strategies. The critic's GPU-range proposals must respect the continuous-GPU and non-overlap constraints enforced here (lines 138-150).
- `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/config/profile/default.yaml` -- nsight-based profiler backend config with `worker_groups: [ActorGroup, RolloutGroup, EnvGroup]`. This shows that profiling all three worker groups is already supported.
- `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/scheduler/worker/worker.py` (lines 1305-1349) -- `Worker.timer` decorator and `worker_timer` context manager. These accumulate `_timer_metrics` dict that feeds into the MetricTable. The critic can rely on these metrics being present.

### Known Risks

- Each trial round runs a full RLinf training epoch, which can take 20+ minutes on 8xA800. The RLCR loop's round-boundary semantic ("agent believes entire plan is finished") is mismatched with a trial-refine cycle where each round is one trial, not completion of a multi-task plan. This requires either adapting the round definition or using a custom loop outside the standard RLCR machinery.
- OOM or crashes may terminate a trial before log collection, leaving no MetricTable to parse. The critic must handle missing or partial logs.
- The tunable knob space is combinatorial (placement ranges, env count, rollout epoch, micro_batch_size, offload flags). Exhaustive search via RLCR rounds is impractical; the critic must propose focused deltas, but LLM critics may lack the quantitative precision to converge efficiently.
- Hydra config resolution via defaults list (e.g. `defaults: [env/maniskill..., model/openvla@actor.model, ...]`) means knob changes must respect the layered composition. Mutating a top-level YAML may be overridden by a default sub-config unless the override syntax is correct.
- The rlinf_timeline instrumentation must be enabled (env vars `RLINF_TIMELINE=1`, `RLINF_TIMELINE_WORKER_TIMER=1`, etc.) for timeline JSONL to appear. If not set, only the coarse MetricTable is available, reducing the critic's diagnostic resolution.

## Alternative Directions Considered

### Alt-1: Bayesian Optimization Tuner
- Gist: An Optuna-based black-box optimizer treats the embodiment config as a mixed search space (categorical placement strings + integer total_num_envs/rollout_epoch/micro_batch_size + boolean offload flags), uses TPE to propose the next config under an OOM/crash feasibility constraint, and marks failed trials PRUNED so the optimizer avoids nearby infeasible regions. Each trial runs one RLinf epoch via Hydra overrides, parses `metrics.log` and timeline JSONL for step_time / (rollout_epoch * total_num_envs), and calls `optuna.tell()`; the existing `toolkits/auto_placement/fitter.py` curve-fits can warm-start the study from any prior profile data. No LLM is in the inner loop, which makes the search reproducible and statistically well-grounded.
- Objective Evidence:
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/fitter.py` (169 LOC): existing `DataFitter` class that fits power_law, exponential, logarithmic, polynomial curves to profile data using `scipy.optimize.curve_fit`. Directly reusable as a warm-start surrogate model or for post-trial bottleneck fitting within the BO loop.
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/auto_placement_worker.py` (241 LOC): existing `AutoPlacementWorker` that enumerates placement strategies using a DAG-of-components model. Confirms search-space structure, but its analytical optimizer is the part the user warns against; BO replaces it with a black-box approach.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/config/maniskill_ppo_openvla.yaml` and `maniskill_ppo_openpi.yaml`: tunable parameters in real configs (`cluster.component_placement` actor/env/rollout dict, `env.train.total_num_envs`, `env.train.rollout_epoch`, `actor.micro_batch_size`, `*.enable_offload`) and the "all" placement variant.
  - `/mnt/public2/zhouyiming/humanize/RLinf/logs/20260626-08:08:46-maniskill_ppo_openvla/metrics.log`: contains `Step Time: 947.166s`, `num_trajectories=318`, per-component timing (`env/interact=536.1`, `actor/run_training=348.5`, `rollout/predict=428.1`). The objective is computed directly from these values.
  - `/mnt/public2/zhouyiming/humanize/RLinf/logs/20260626-08:08:46-maniskill_ppo_openvla/timeline/*.jsonl`: per-call durations (`t0`, `t1`, `component`, `tag`, `global_step`, `call_index`) for actor/rollout/env/runner, used for fine-grained bottleneck features.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/run_embodiment.sh`: already generates timestamped log directories and accepts Hydra overrides in the CMD string.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/profiler.py` (245 LOC): `PyTorchProfiler.from_config()` factory with Chrome / TensorBoard trace export, configurable per trial.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/metric_utils.py`: `print_train_metrics()` computes `Step Time = elapsed_time / steps_done` and writes `metrics.log` (also logged to tensorboard).
  - `/mnt/public2/zhouyiming/humanize/RLinf/pyproject.toml`: no Optuna/SMAC dependency listed; `scipy` is already present. Optuna would be a new dependency.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/placement.py`: `HybridComponentPlacement` handles both "actor:0-7, env:0-3, rollout:4-7" and "actor,env,rollout:all" formats — the two placement families to encode categorically.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/scheduler/cluster/config.py`: `ClusterConfig` dataclass with `component_placement` as `list[dict[str, str]]`, defining the structural constraints any BO-generated config must respect.
- Why not primary: Removes the LLM critic from the inner loop entirely, departing from the user's explicit "use humanize + RLCR" framing, and introduces a new Optuna dependency plus a non-trivial categorical encoder for the combinatorial GPU-range placement space.

### Alt-2: Analytical Bottleneck-Driven Heuristic
- Gist: Skip statistical search entirely and encode the user's stated cost model directly as deterministic rules — env_rollout ≈ a·N+b; actor is piecewise-linear-then-nonlinear in N and U-shaped in micro_batch_size. Each round, a `CostModel` module estimates per-component time from the latest run's MetricTable, a `ConstraintChecker` enforces `validate_cfg`'s divisibility rules and a peak-memory OOM guard, and a `GradientStep` solver moves the next config along the bottleneck's gradient (if env_rollout dominates → reduce total_num_envs or grow env GPU count; if actor dominates → step micro_batch_size toward the U-shape minimum or grow actor GPUs; if rollout inference dominates → grow rollout GPU count or reduce pipeline_stage_num). Interpretable, fast per round, cheap memory cost. Implemented as a new script under `toolkits/` parallel to `auto_placement`.
- Objective Evidence:
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/fitter.py`: Existing `DataFitter` fits power_law, exponential, logarithmic, polynomial curves via `scipy.optimize.curve_fit`; precedent for curve fitting but does not support piecewise models or cold-start.
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/node.py`: `MegatronNode._estimate_cost()` (lines 85-98) uses a deterministic heuristic formula (linear + sublinear correction) for actor cost — the exact pattern to extend with piecewise and U-shape.
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/auto_placement_worker.py`: `_find_schedule()` does exhaustive search over collocated/disaggregated placements — replaced here with a directed gradient step.
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/util.py`: `init_global_config_env()` (lines 67-105) requires `config.profile_data` which is absent from maniskill configs, confirming the auto_placement cold-start failure.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/config/maniskill_ppo_openvla.yaml`: target config exposing all tunable parameters (`cluster.component_placement`, `env.train.total_num_envs=128`, `env.train.rollout_epoch=1`, `actor.micro_batch_size=80`, `actor.global_batch_size=640`, `*.enable_offload`, `rollout.pipeline_stage_num=2`).
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/config.py` (lines 958-983): `validate_cfg` divisibility constraints any adjustment must respect (`total_num_envs % env_world_size == 0`, etc.).
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/runners/embodied_runner.py` (lines 497-560): `ScopedTimer` keys "step", "sync_weights", "generate_rollouts", "cal_adv_and_returns" — the per-component timing data each round extracts.
  - `/mnt/public2/zhouyiming/humanize/profiler/rlinf_timeline/autopatch.py`: sidecar emits per-step timeline JSONL with `component`, `rank`, `tag`, `t0`, `t1`, plus nvml/nvitop GPU memory samplers for OOM constraint data.
  - `/mnt/public2/zhouyiming/humanize/RLinf/logs/20260626-08:29:39-maniskill_ppo_openvla/nvitop/nvitop_summary.log`: per-process GPU memory (rollout avg 23 GiB / max ~25 GiB; actor avg 1.7 GiB / max ~6 GiB; env 0.457 GiB) — the data for the OOM constraint.
  - `/mnt/public2/zhouyiming/humanize/RLinf/demo_logs/maniskill_ppo_openvla_envs192_mbs80_env0-1_roll2-7_act0-7.log`: reference for a successful configuration (192 envs, mbs 80, env on 0-1, rollout on 2-7, actor on 0-7).
- Why not primary: Drops the LLM critic and relies entirely on a fixed three-component cost model — vulnerable when bottlenecks are unmodeled (weight sync, channel contention, NCCL stalls), and gives up the qualitative reasoning that motivates the "use humanize" framing.

### Alt-3: Parametric Surrogate + Offline Search
- Gist: Spend a fixed calibration budget on K probe runs that each sweep one axis of the config space, collect step_time and per-component timings from timeline JSONL and nvitop, fit the user's stated parametric forms (env=a·N+b, actor piecewise + U-shape over mbs), then optimize `step_time / (rollout_epoch * total_num_envs)` analytically over the surrogates under a peak-memory OOM constraint. Only the top few Pareto-optimal candidates predicted by the surrogate touch GPUs again for validation. Minimizes total real-run count by frontloading sample budget into a structured calibration phase rather than iterative trials.
- Objective Evidence:
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/fitter.py` (169 LOC): `DataFitter` with scipy.optimize.curve_fit; extensible to add piecewise-linear and constrained-U-shape fitters matching the user's parametric forms.
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/auto_placement_worker.py` (241 LOC): `AutoPlacementWorker` uses `EnvProfiler` + `DataFitter` for cost estimation across GPU counts — architectural precedent for profile-driven surrogates.
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/node.py` (211 LOC): `EnvProfiler` class feeds profile data into `DataFitter` — the pattern needed for per-axis probes.
  - `/mnt/public2/zhouyiming/humanize/profiler/rlinf_timeline/autopatch.py`: produces per-component JSONL events with timestamps for surrogate fitting input.
  - `/mnt/public2/zhouyiming/humanize/profiler/rlinf_timeline/writer.py`: `append_event` writes structured JSONL records (`component`, `rank`, `tag`, `t0`, `t1`, `global_step`).
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/distributed.py` line 1272: `ScopedTimer` records `time/step`, `time/env/*`, `time/rollout/*`, `time/actor/*` durations used by `EmbodiedRunner`.
  - `/mnt/public2/zhouyiming/humanize/RLinf/logs/*/nvitop/*.jsonl`: per-process peak memory (`max_process_gpu_mem`) — e.g. actor peak 5.7 GiB, rollout peak 24.9 GiB on A800 80 GiB. The OOM constraint surface.
  - `/mnt/public2/zhouyiming/humanize/RLinf/logs/*/nvitop/nvitop_summary.log`: aggregated GPU memory summary per GPU and per-process.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/config/maniskill_ppo_openvla.yaml`: base config with all tunable parameters used for Hydra-override probe generation.
  - `/mnt/public2/zhouyiming/humanize/RLinf/demo_logs/maniskill_ppo_openvla_envs192_mbs80_env0-1_roll2-7_act0-7.log`: an existing single-axis probe, evidence the user has already begun manual sweeps that could seed calibration.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/runners/embodied_runner.py` lines 344-360: per-step aggregation of `time/env`, `time/rollout`, `time/actor` via ScopedTimer — exactly the decomposition the surrogate fits.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/metric_utils.py` `print_metrics_table`: prints "Step Time: X.XXs" per step.
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/` (1234 LOC): structural template (fitter + node profiler + workflow scheduler) but must not be reused directly per user's instruction.
- Why not primary: Larger upfront code surface (new piecewise / U-shape fitters, surrogate solver, separate probe runner) and assumes the parametric forms hold uniformly; weaker fit with the user's iterative RLCR pattern since the optimization happens offline rather than per-round.

### Alt-4: Evolutionary Population Search
- Gist: Maintain a population of N config genomes (discrete genes for placement strings, continuous for `total_num_envs`/`rollout_epoch`/`micro_batch_size`/offload flags). Each generation evaluates the fittest by running 1 epoch of RLinf, fitness = `step_time / (rollout_epoch * total_num_envs)`, OOM/worker-death produces infinite cost (lethal mutation). Selection picks the top K; constraint-aware crossover swaps GPU-range partitions between two parents while preserving non-overlap/contiguity; Gaussian mutation on continuous genes is filtered through `validate_cfg`'s divisibility checks; boolean genes flip independently. Population diversity tolerates the non-smooth discrete-heavy landscape (especially placement) better than gradient-style heuristics.
- Objective Evidence:
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/`: existing placement-search toolkit (~242 LOC in `auto_placement_worker.py`) — closest precedent for automated placement search, though its analytical optimizer fails on embodiment.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/run_placement_autotune.sh`: shell script that invokes `auto_placement_worker.py` with Hydra config — the pattern to follow for invoking a search script with the same config-path/config-name mechanism.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/config/*.yaml` (~100 files): diverse placement patterns — collocated ("env,rollout,actor: 0-3"), hybrid ("actor: 0-7, env: 0-3, rollout: 4-7"), all-collocated ("actor,env,rollout: all"). The discrete gene pool for crossover.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/placement.py`: `HybridComponentPlacement` (lines 86-96) and `ModelParallelComponentPlacement` (lines 99-218) — modes COLLOCATED, DISAGGREGATED, HYBRID, AUTO; `_is_collocated`, `_is_disaggregated`, `_generate_placements` define the constraints any crossover output must satisfy.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/config.py`: ~80 assertions in `validate_cfg` (line 1285) and `validate_embodied_cfg` enforce divisibility: `total_num_envs % env_world_size == 0`, `total_num_envs % env_world_size % pipeline_stage_num == 0`, `global_batch_size % (micro_batch_size * actor_world_size) == 0`, `max_steps_per_rollout_epoch % num_action_chunks == 0`. Any mutation must preserve these.
  - `/mnt/public2/zhouyiming/humanize/RLinf/logs/*/timeline/*.jsonl`: per-component JSONL with `t0/t1`, metadata (`configured_pipeline_stages`, `rollout_epoch`) — fitness-signal source for bottleneck-aware selection pressure.
  - `/mnt/public2/zhouyiming/humanize/RLinf/demo_logs/maniskill_ppo_openvla_envs192_mbs80_env0-1_roll2-7_act0-7.log`: MetricTable with Step Time (1473.315s), per-component timings, `num_trajectories=448` — the fitness signal extraction format.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/timers.py` (196 LOC): `NamedTimer`/`ScopedTimer` provide the timing infrastructure.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/runners/embodied_runner.py`: `_log_step_metrics` (line 331) consumes per-component durations logged under `time/env`, `time/rollout`, `time/actor` — the metric namespaces for fitness aggregation.
  - `/mnt/public2/zhouyiming/humanize/RLinf/toolkits/auto_placement/fitter.py`: `DataFitter` precedent for cost-modeling but evolutionary search avoids analytical models by direct measurement.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/metric_utils.py`: `step_time_str = elapsed_time / steps_done` in the MetricTable.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/config/profile/default.yaml`: nsight profiling for ActorGroup/RolloutGroup/EnvGroup confirms timeline collection is built in.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/scheduler/cluster/config.py`: `ClusterConfig` (lines 163-487) defines validation rules placement genomes must respect.
  - `/mnt/public2/zhouyiming/humanize/RLinf/tests/unit_tests/test_auto_placement.py` (341 LOC): testing pattern to follow.
- Why not primary: Population-of-N × generations-of-G evaluations multiply the (already 20+ min/trial) GPU cost; lacks the directed-feedback signal that the LLM critic provides in the primary; constraint-aware crossover/mutation operators are non-trivial to implement correctly given the ~80 assertions in `validate_cfg`.

### Alt-5: Hierarchical Placement → Env → Actor Decomposition
- Gist: Rather than picking a search algorithm, restructure the search space by phase. Stage 1 selects the placement strategy (collocated vs disaggregated vs hybrid) via cheap proxy probes (1-2 training steps, minimal env count) since placement is the highest-impact, costliest-to-change decision. Stage 2 sweeps env-side (`total_num_envs`, `rollout_epoch`, `env.enable_offload`) exploiting the linear `env_time = a·N + b`. Stage 3 sweeps actor-side (`micro_batch_size`, `enable_offload`, `gradient_checkpointing`) exploiting the U-shaped profile of `micro_batch_size`. Any inner search algorithm (RLCR critic, BO, heuristic, evolutionary) can live inside a stage; the structural choice is independent. Reduces effective dimensionality at each phase and avoids futile combinations.
- Objective Evidence:
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/placement.py` lines 28-41: `PlacementMode` enum defines COLLOCATED, DISAGGREGATED, HYBRID, AUTO — exactly the Stage-1 taxonomy. Lines 86-96: `HybridComponentPlacement` auto-detects mode from YAML, directly reusable for Stage 1.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/train_embodied_agent.py` line 42: uses `HybridComponentPlacement(cfg, cluster)` as the single placement entry point for embodiment — the code location to feed different component_placement YAMLs per iteration.
  - `/mnt/public2/zhouyiming/humanize/RLinf/examples/embodiment/config/`: YAML files demonstrate all three placement patterns in production — collocated, disaggregated, hybrid.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/config.py` lines 959-978: `total_num_envs % env_world_size % pipeline_stage_num == 0` and divisibility by `group_size` — discrete grid for Stage-2.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/config.py` line 1053 and lines 931-978: `micro_batch_size` validation and pipeline_stage_num interactions with env-side parameter divisibility.
  - `/mnt/public2/zhouyiming/humanize/RLinf/logs/20260626-08:29:39-maniskill_ppo_openvla/timeline/*.jsonl`: structured per-component timing (`t0`, `t1`, `component`, `rank`, `tag`, `configured_rollout_epochs`, `configured_train_chunk_steps`, `configured_pipeline_stages`) for per-stage bottleneck extraction.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/metric_utils.py` line 310: `step_time_str = f"Step Time: {elapsed_time / steps_done:.3f}s"` — existing metric the optimizer reads.
  - `/mnt/public2/zhouyiming/humanize/RLinf/rlinf/utils/placement.py` lines 220-248: `_is_auto()` requires Megatron-style layout and `cluster.auto_scheduler` — confirms the existing AUTO mode does NOT support embodiment, motivating new hierarchical search rather than extending AUTO.
  - `/mnt/public2/zhouyiming/humanize/humanize/commands/start-rlcr-loop.md`: humanize's RLCR loop provides the outer "run → observe → adjust" infrastructure that each stage's inner loop can reuse.
  - `/mnt/public2/zhouyiming/humanize/mlsys2026-flashinfer-contest/CLAUDE.md` lines 34-35: references `humanize` as an external skill, illustrating the iterative prompt-driven workflow this idea applies to RLinf.
- Why not primary: This is a search-space schedule, not a search algorithm — it still needs an inner mutation strategy per stage, so it composes with the primary rather than replacing it; on its own it does not specify how to mutate within a phase.

## Synthesis Notes

The primary direction is an orchestration shell, not an inner search algorithm — every alternative composes naturally inside it rather than competing with it. Alt-2's bottleneck-detection rules belong inside the LLM critic prompt as a structured rubric (`env vs actor vs rollout dominates → propose this knob`), anchoring critique in measurable parametric signals instead of free-form judgment. Alt-3's parametric surrogate becomes a cold-start phase: a small calibration sweep fits `env_time = a·N + b` and the actor U-shape, then seeds the first RLCR round with a quantitatively grounded proposal rather than blind exploration. Alt-5's hierarchical decomposition becomes a phase schedule for the loop — early rounds tune placement only, middle rounds tune env-side knobs, late rounds tune actor-side — which keeps each round's mutation space small enough for the critic to reason about reliably. Alt-1 (Bayesian) and Alt-4 (evolutionary) are alternative inner proposers if the LLM critic underperforms: both can be plugged in by swapping the propose-next-config step while keeping the same trial-runner, log-parser, and convergence check — useful as fallback once enough trials accumulate to train a surrogate or seed a population.

--- Original Design Draft End ---
