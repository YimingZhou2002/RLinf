# embodied_tuner — Architecture & Component Reference

This document describes **how the auto-tuner is built and why**: the
end-to-end data flow, the role of each module, the design decisions
that resolve non-obvious tensions (no-Ray preflight, env-tagged orphan
cleanup, dual-source rationale, etc.), and the integration boundary
with RLinf and the humanize plugin.

For operator-facing usage, see [`README.md`](./README.md).
For the original design draft, see `<repo>/draft_run.md`.
For the implementation plan that drove this build, see
`<repo>/docs/plan.md` (RLinf copy: `docs/plan.md`).

---

## 1. Overview

### 1.1 The problem

RLinf embodied training (e.g. `examples/embodiment/run_embodiment.sh
maniskill_ppo_openvla`) is sensitive to several config knobs whose
optimum is workload- and hardware-dependent:

- `cluster.component_placement` (which GPU ranges run actor / env / rollout)
- `env.train.total_num_envs`, `env.train.rollout_epoch`
- `actor.micro_batch_size`
- `env / rollout / actor . enable_offload`

The existing `RLinf/toolkits/auto_placement/` toolkit needs
`config.profile_data` for cold start, which embodied configs do not
provide. We need a **cold-start auto-tuner** that:

- Iteratively explores the knob space toward minimum
  `step_time / num_trajectories` under memory/feasibility constraints.
- Catches divisibility / placement violations **before** spending
  ~20 minutes of GPU time on an invalid trial.
- Recovers cleanly from OOMs, timeouts, and worker crashes without
  leaving orphan Ray actors behind.
- Lets a human operator audit every tuning decision (which signal
  justified which delta).

### 1.2 The design

- An **LLM critic** (Codex via `ask-codex.sh`) proposes the next config
  delta from the last trial's evidence. Placement-touching deltas must
  cite **both** MetricTable evidence AND `timeline/*.jsonl` evidence —
  the *dual-source rationale rule*. A `CriticOutputValidator` enforces
  this mechanically; the critic retries up to 3 times with feedback.
- A **preflight validator** composes baseline + delta via Hydra and
  runs the targeted divisibility checks from `rlinf/config.py`. It
  deliberately **does not call `validate_cfg` / `validate_embodied_cfg`**
  because those instantiate `Cluster()` which calls `ray.init`, and
  preflight is contractually GPU/Ray-free.
- A **trial runner** launches RLinf in its own POSIX process group,
  enforces a per-trial timeout, escalates SIGTERM→SIGKILL on hang,
  invokes `ray stop --force`, and sweeps both `pgrep -f` *and*
  `/proc/<pid>/environ` for orphan workers tagged with a
  campaign-unique trial id.
- A **log + timeline parser** classifies each trial with
  `(Status, FailureMode)`, computes the objective with warmup
  exclusion, and surfaces a per-component timeline summary the next
  critic prompt consumes.
- An **append-only JSONL ledger** persists every trial, including the
  structured `critic_rationale` payload — the **audit trail** that lets
  operators trace WHY each placement decision was taken.
- A **scheduler** orchestrates the loop with explicit budget +
  stopping rules; preflight failures don't consume trial slots; if the
  critic can't produce a valid delta within `preflight_retries`, the
  loop terminates with `preflight_exhausted` rather than launching a
  known-bad config.
- A **CLI + shim launcher** glue everything to `examples/embodiment/run_embodiment.sh`
  and emit `best_config.yaml` + `best_trial.json` next to the ledger.

The humanize + RLCR pipeline was used **once to build this tuner**.
The tuner's per-trial loop is **plain Python**, not an RLCR loop.

---

## 2. End-to-end data flow

A single trial flows through the modules as follows:

```
       ┌─────────────────────────────────────────────────────────┐
       │                                                         │
       │             (1) Scheduler.run() → loop iteration        │
       │                                                         │
       └─────────────┬───────────────────────────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────┐
   │ (2) Critic.propose(history,         │
   │     current_knobs, prev_metrics,    │
   │     prev_timeline,                  │
   │     preflight_feedback?)            │ ◀────────┐
   │                                     │          │
   │  CodexCritic shells out to          │          │ feedback string
   │  ask-codex.sh; parses JSON;         │          │ on validation fail
   │  CriticOutputValidator enforces:    │          │
   │   - knob schema                     │          │
   │   - dual-source rule (placement)    │          │
   │   - retry up to 3x with feedback    │          │
   └─────────────────┬───────────────────┘          │
                     │ CriticOutput                 │
                     │ {delta, rationale,           │
                     │  stop_requested}             │
                     ▼                              │
   ┌─────────────────────────────────────┐          │
   │ (3) Preflight.compose_and_validate  │          │
   │                                     │──────────┘ rejection feedback
   │  - Hydra compose baseline + delta   │            (next critic call)
   │  - KnobSchema.validate(delta)       │
   │  - placement_enum.is_legal_placement│
   │  - targeted divisibility checks     │
   │    (rlinf/config.py:962/965/980/1363)
   │  No Ray. No GPU.                    │
   └─────────────────┬───────────────────┘
                     │ PreflightOutcome
                     │ {ok, errors, sha,
                     │  log_dir, delta}
                     ▼
            ┌────────┴────────┐
        ok? │                 │ retries exhausted
            ▼                 ▼
   ┌─────────────────────┐   ┌──────────────────────────────┐
   │ (4) Runner.launch   │   │ Scheduler writes synthetic    │
   │                     │   │ (FAILED, CONFIG_INVALID)      │
   │ OverrideWrapper     │   │ ledger entry → stop:          │
   │ .build_invocation   │   │ "preflight_exhausted"         │
   │     ↓               │   └──────────────────────────────┘
   │ subprocess.Popen    │
   │  (POSIX pgroup,     │
   │   profiler env,     │
   │   RLINF_TUNER_      │
   │     TRIAL_ID=       │
   │   <campaign>-<idx>) │
   │                     │
   │ wait/timeout/       │
   │ SIGTERM→SIGKILL/    │
   │ ray stop --force/   │
   │ /proc env scan +    │
   │ pgrep -f orphan     │
   │ sweep               │
   └─────────────────┬───┘
                     │ TrialOutcome
                     │ {returncode, timed_out,
                     │  cleanup_outcome, ...}
                     ▼
   ┌─────────────────────────────────────┐
   │ (5) Parser.parse_trial              │
   │                                     │
   │  - reads metrics.log MetricTable    │
   │  - reads timeline/*.jsonl           │
   │  - OOM/crash rubric BEFORE          │
   │    METRICS_MISSING on returncode≠0  │
   │  - objective = avg(step_time[2..N]) │
   │             / num_trajectories      │
   │  - per-component timeline summary   │
   │    (consumed by next critic prompt) │
   └─────────────────┬───────────────────┘
                     │ TrialResult
                     │ {status, failure_mode,
                     │  objective, summary, ...}
                     ▼
   ┌─────────────────────────────────────┐
   │ (6) Ledger.append(LedgerEntry)      │
   │                                     │
   │  append-only JSONL; fsync per line; │
   │  structured critic_rationale        │
   │  persisted verbatim                 │
   └─────────────────┬───────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────┐
   │ (7) Scheduler stopping-rule check   │
   │                                     │
   │  max_trials | budget_seconds_elapsed│
   │  | oom_cap_exceeded | plateau       │
   │  | critic_stagnation | critic_failure
   │  | preflight_exhausted              │
   └─────────────────┬───────────────────┘
                     │
            still going? loop back to (1)
            done? → CampaignResult →
                    __main__._emit_best_artefacts
                    writes best_config.yaml + best_trial.json
```

---

## 3. Module reference

Each toolkit file is independently testable; cross-references between
them go through small public dataclasses (`LaunchSpec`, `TrialOutcome`,
`TrialResult`, `LedgerEntry`, `CriticOutput`) defined as frozen
dataclasses for clarity and immutability.

### 3.1 `schema.py` — knob schema (AC-1)

- **Role.** Single source of truth for which knobs the critic may
  mutate, what values are legal, and which knobs are pinned.
- **Public API.**
  - `KnobDomain` — per-knob dataclass; `kind ∈ {int, bool, placement}`,
    optional `min_value`/`max_value`, `pinned: bool`.
  - `KnobSchema` — owns the canonical knob → `KnobDomain` mapping.
    Methods: `list_knobs()`, `list_pinned_knobs()`, `validate(delta)`.
  - Exception hierarchy: `KnobSchemaError` → `UnknownKnobError`,
    `KnobOutOfRangeError`, `KnobNotTunableError`.
- **Design notes.**
  - `bool` values are rejected for `int` knobs (Python treats `bool` as
    a subclass of `int`, which would otherwise silently pass).
  - The 3 pinned knobs (`actor.global_batch_size`,
    `rollout.pipeline_stage_num`, `actor.model.num_action_chunks`) are
    declared in the schema so the un-pinning path in `FUT-5` is a
    `pinned=False` flip — not a code edit.
  - Cross-knob divisibility (e.g. `global_batch_size %
    (micro_batch_size * actor_world_size) == 0`) is **not** the
    schema's job — that's preflight.

### 3.2 `placement_enum.py` — placement enumeration & legality (AC-4)

- **Role.** Parse and validate `cluster.component_placement` strings;
  produce a curated set of legal candidate placements for the critic /
  fallback proposers to choose from.
- **Public API.**
  - `parse_range_spec("0-7" | "all" | "0,2,4-6")` → `tuple[int, ...]`.
  - `PlacementSpec` — frozen dataclass holding three contiguous GPU-id
    tuples + a `kind ∈ {collocated, disaggregated, hybrid, all}`.
  - `is_legal_placement(mapping_or_spec, num_gpus=8)` →
    `(ok, reason_or_kind)`.
  - `enumerate_placements(num_gpus=8)` → curated `list[PlacementSpec]`
    guaranteed to include at least one of each kind.
- **Design notes.**
  - Mirrors RLinf's runtime checks: contiguous GPU ranges (per
    `ModelParallelComponentPlacement`'s assertions at
    `rlinf/utils/placement.py:138-160`) and env/rollout being either
    **equal** (collocated) or **fully disjoint** — *partial* overlap
    is the only forbidden case.
  - Hybrid placements where actor covers all GPUs and env+rollout
    split disjointly (the maniskill baseline) are explicitly allowed.

### 3.3 `override_wrapper.py` — Hydra-override launcher (AC-2)

- **Role.** Build a `LaunchSpec` (argv + env + log_dir) that invokes
  `train_embodied_agent.py` with arbitrary Hydra overrides appended
  AFTER the stock script's `runner.logger.log_path=...` injection.
- **Public API.**
  - `OverrideWrapper.for_repo(repo_path)` — factory with the canonical
    RLinf layout.
  - `OverrideWrapper.build_invocation(config_name, overrides, log_dir,
    trial_id, extra_env=None)` → `LaunchSpec`.
  - `LaunchSpec` — frozen dataclass: `argv`, `env`, `log_dir`,
    `config_name`, `baseline_overrides`, `user_overrides`, plus a
    helper `.overrides_in_order()`.
- **Design notes.**
  - The stock `run_embodiment.sh` only injects
    `runner.logger.log_path=${LOG_DIR}`. We don't modify it
    (Path Boundaries forbid). The wrapper replicates the stock script's
    env-var defaults (`MUJOCO_GL=egl`, `ROBOT_PLATFORM=LIBERO`, etc.)
    and prepends `REPO_PATH` to `PYTHONPATH`.
  - User overrides go AFTER baseline overrides so **Hydra precedence**
    (later wins) lets the critic always override the stock script's
    injections.
  - Exports `RLINF_TUNER_TRIAL_ID=<id>` for the orphan-cleanup tag.

### 3.4 `preflight.py` — config validation without Ray (AC-3)

- **Role.** Compose baseline + delta via Hydra, run the knob schema,
  validate placement legality, and apply the divisibility checks that
  `validate_cfg` / `validate_embodied_cfg` would — without spinning up
  Ray.
- **Public API.**
  - `compose_and_validate(baseline_path, delta, hydra_overrides=(),
    schema=None, num_gpus=8)` → `ValidationResult`.
  - `ValidationResult` — frozen dataclass: `ok`, `errors`,
    `resolved_cfg`, `resolved_config_sha`, `placement_kind`.
- **Design notes.**
  - The big tension: `rlinf.config.validate_embodied_cfg` does
    `HybridComponentPlacement(cfg, Cluster())` at line 922, and
    `Cluster.__init__` calls `ray.init` (`rlinf/scheduler/cluster/cluster.py:332`).
    Calling it would violate the "no GPU work" contract preflight has
    by construction. **Resolution:** mirror the targeted divisibility
    checks locally — they all depend only on
    `cluster.component_placement` (parsed by `placement_enum`) plus
    scalar config fields, so they're computable from the composed
    `DictConfig` alone.
  - Mirrored assertions (file:line in `rlinf/config.py`):
    - `:962` — `env.train.total_num_envs % env_world_size == 0`
    - `:965` — `per_rank % rollout.pipeline_stage_num == 0`
    - `:980` — `max_steps_per_rollout_epoch % num_action_chunks == 0`
    - `:1363-1368` — `global_batch_size % (micro_batch_size * actor_world_size) == 0`
  - Hydra composition requires `EMBODIED_PATH` for `${oc.env:EMBODIED_PATH}`
    interpolations in `hydra.searchpath`. The module sets `EMBODIED_PATH`
    and `REPO_PATH` from the baseline path itself before `compose()`,
    and uses `OmegaConf.to_yaml(..., resolve=False)` for the SHA so
    other env-interpolations stay symbolic.

### 3.5 `runner.py` — trial subprocess + cleanup (AC-5)

- **Role.** Launch a `LaunchSpec` as a subprocess, enforce a per-trial
  timeout, clean up orphan Ray workers scoped to this trial.
- **Public API.**
  - `TrialRunner.launch(spec, timeout=None)` → `TrialOutcome`.
  - `TrialOutcome` — frozen dataclass: `log_dir`, `returncode`,
    `timed_out`, `wall_clock_seconds`, `cleanup_outcome`,
    `stdout_path`, `spec`.
- **Design notes.**
  - `start_new_session=True` puts the trial in its own POSIX process
    group so we can `os.killpg` the whole tree on timeout.
  - SIGTERM → wait `sigterm_grace_seconds` → SIGKILL → wait again →
    classify `cleanup_outcome` as `ok | sigkill_required | orphans_remain`.
  - `ray stop --force` invoked via `_default_ray_stop_hook` after reap.
    Treated as best-effort; failure surfaces as `ray_stop_failed`.
  - **Orphan-cleanup scope** is the subtle part. We export
    `RLINF_TUNER_TRIAL_ID=<id>` as an env var, but `pgrep -f` matches
    *argv*, not environment. Ray workers spawned by the trial inherit
    the env var but their argv typically doesn't carry it, so
    `pgrep -f` alone misses them. Fix: scan `/proc/<pid>/environ`
    directly (`_pids_with_env_match`) and union with `pgrep -f`.
    Recorded as BitLesson `BL-2026-06-30-pgrep-env-vs-argv`.
  - Profiler env vars (`RLINF_TIMELINE*`, `RLINF_NVITOP`, `RLINF_NVML`)
    are default-on. Opt out via `disable_profiler` /
    `disable_memory_telemetry`.

### 3.6 `parser.py` — log + timeline parsing (AC-6)

- **Role.** Turn a trial's `LOG_DIR/{metrics.log, timeline/*.jsonl,
  nvitop/*}` into a structured `TrialResult` the scheduler and critic
  can consume.
- **Public API.**
  - `parse_trial(log_dir, returncode=None, timed_out=False,
    failure_mode_override=None, stderr_path=None)` → `TrialResult`.
  - `parse_metrics_log(path)` → `tuple[MetricStep, ...]`.
  - `parse_timeline(timeline_dir)` → `TimelineSummary`.
  - `compute_objective(per_step)` → `(objective, avg_step_time, partial_reason)`.
  - `select_best(results)` → `TrialResult | None`.
  - Enums: `Status ∈ {OK, FAILED}`,
    `FailureMode ∈ {NONE, METRICS_PARTIAL, METRICS_MISSING,
    CONFIG_INVALID, LAUNCH_FAILURE, OOM, WORKER_CRASH, TIMEOUT}`.
  - Dataclasses: `MetricStep`, `TagStats`, `TimelineSummary`,
    `TrialResult`. `TrialResult` enforces `(FAILED, NONE)` is impossible
    via `ParserInvariantError`.
- **Design notes.**
  - MetricTable blocks are delineated by Unicode box-draw characters
    (`╭` … `╰`). The parser splits on those, then extracts
    `Global Step: X/Y`, `Step Time: NNN.NNNs`, and
    `key=value` cells (Time + Environment sections).
  - **OOM precedence**: when `returncode != 0`, the OOM/crash rubric
    runs **before** the `metrics.log` presence check. An OOM-killed
    trial that never wrote `metrics.log` is correctly classified
    `(FAILED, OOM)` not `(FAILED, METRICS_MISSING)`.
  - `stderr_path` defaults to `log_dir/run_embodiment.log` — the
    runner's merged stdout+stderr file — so callers don't have to
    remember to pass it.
  - Objective averaging: drop step 1 as warmup; average step_time
    across steps 2..N. With `--max-epochs=3` this is steps 2 and 3.
  - `num_trajectories` comes from the **final** MetricTable block (no
    silent fallback denominator).
  - `select_best` requires `(OK, NONE)` strictly — `(OK,
    METRICS_PARTIAL)` is critic context, not a best-config candidate.
  - `TimelineSummary` carries per-rank `TagStats` for a verified
    headline-tag set (`env_interact_step`, `actor/sync_model_to_rollout`,
    `rollout/generate`, `predict`, etc. — the actual tags emitted by
    `profiler/rlinf_timeline/autopatch.py`, **not** the MetricTable
    aggregate keys), plus per-component stall fractions.

### 3.7 `ledger.py` — append-only JSONL persistence (AC-9)

- **Role.** Persist every trial as one JSONL line; survive mid-loop
  crashes; serve as the audit trail for placement decisions.
- **Public API.**
  - `Ledger(path, fsync_on_append=True)` — frozen dataclass.
  - `Ledger.append(LedgerEntry)`, `Ledger.load()` → `LoadResult`,
    `Ledger.best()` → `LedgerEntry | None`.
  - `LedgerEntry` — frozen dataclass with the 17 fields from AC-9
    (incl. structured `critic_rationale`). `from_dict` validates that
    every required field is present.
  - `make_entry(...)` — convenience constructor used by the scheduler.
  - `LoadResult(entries, skipped_lines)` so a single corrupted line is
    counted but doesn't break loading the rest.
- **Design notes.**
  - `append()` validates the entry, opens in `"a"`, writes one line,
    `flush()`, `os.fsync()`. A SIGKILL between trials never truncates
    a previously-written entry.
  - `load()` tolerates JSON decode errors AND schema violations per
    line (counts them in `skipped_lines`).
  - `.best()` mirrors `parser.select_best`'s eligibility rule
    (`status="OK", failure_mode="NONE"`, non-None objective).
  - `critic_rationale` is persisted as a plain dict
    (`{summary, metric_table_citations, timeline_citations}`) — the
    audit-trail surface the plan calls "Placement Decision Audit
    Trail". Operators can `jq` the ledger to see exactly which
    observations drove each placement change.

### 3.8 `critic.py` — prompt + validator + Codex transport (AC-7)

- **Role.** Turn parsed trial outcomes into a structured prompt;
  invoke Codex; parse the JSON response; enforce the dual-source rule;
  retry on validator failures.
- **Public API.**
  - `CriticPrompt` — frozen dataclass with section blocks (rubric,
    history, current_knobs, constraints, memory_pressure,
    timeline_summary, feedback). `__str__` assembles the prompt.
  - `build_prompt(history, current_knobs, schema, last_failure_mode,
    last_metric_summary, last_timeline_summary, feedback=None,
    preflight_feedback=None)` → `CriticPrompt`.
  - `Rationale` — frozen `{summary,
    metric_table_citations, timeline_citations}`.
  - `CriticOutput` — frozen `{delta, rationale, stop_requested}`.
  - `TrialHistoryEntry` — per-trial summary the critic prompt repeats.
  - `parse_critic_output(text)` → `CriticOutput`. Tolerates raw JSON,
    Markdown code fences, and brace-bracketed JSON. **Rejects** bare
    strings / non-list / non-string-element citation arrays.
  - `CriticOutputValidator(schema)` with `.validate(output)` →
    `ValidationResult`. Enforces the dual-source rule.
  - `Critic` Protocol with a `propose(...)` method.
  - `CodexCritic` — production implementation shelling out to
    `ask-codex.sh`. Retries up to `max_retries` (default 3) on JSON or
    validator failure, with the failure reason appended as feedback.
- **Design notes.**
  - **Dual-source rule.** When the proposed `delta` contains
    `cluster.component_placement`, the rationale MUST contain at least
    one non-empty `metric_table_citations` entry AND at least one
    non-empty `timeline_citations` entry. Otherwise, the validator
    rejects and the critic retries. This is the design's primary lever
    against ungrounded placement reasoning — a hallucinated "I think
    this is faster" placement delta cannot proceed without naming
    actual observations.
  - **Citation type validation.** `parse_critic_output` rejects bare
    strings in `metric_table_citations` because the dual-source check
    would otherwise iterate a string character-by-character (each char
    "non-empty"), silently bypassing the rule.
  - **Preflight feedback wiring.** `build_prompt` accepts a
    `preflight_feedback` block; the scheduler passes preflight
    rejection reasons via this channel so the critic actually learns
    from divisibility / placement violations.
  - **Transport injection.** `CodexCritic.transport` is injectable for
    tests (no real Codex call); production uses
    `ask-codex.sh` via `subprocess.run`.

### 3.9 `fake_critic.py` — deterministic test critic (AC-7 helper)

- **Role.** A deterministic `Critic` for tests and the AC-11 smoke
  harness. No LLM, no network.
- **Public API.**
  - `FakeCritic(outputs=[CriticOutput, ...])`.
  - `FakeCritic.from_deltas(*deltas)` — builds a critic that returns
    each delta with a minimal valid rationale.
  - `FakeCritic.stop_after(*deltas)` — same, but the final response
    has `stop_requested=True`.
  - `.calls` captures `(history_len, current_knobs, preflight_feedback)`
    tuples so tests can assert on what the critic was called with.
- **Design notes.** Lives in a separate file so it can be imported
  independently (the smoke test composes it via the CLI's
  `--fake-critic` flag).

### 3.10 `scheduler.py` — campaign orchestration (AC-8)

- **Role.** Glue critic + preflight + runner + parser + ledger into
  the per-trial loop. Own the budget and the stopping rules.
- **Public API.**
  - `BudgetConfig` — defaults `max_trials=20, budget_seconds=43200,
    max_oom=5, patience=3, epsilon=0.02, preflight_retries=3,
    history_window=8`.
  - `Scheduler(critic, runner_fn, parser_fn, preflight_fn, ledger,
    budget, baseline_knobs, clock)`. `runner_fn` / `parser_fn` /
    `preflight_fn` are callable injection points — production wires
    them to the real `TrialRunner.launch` / `parse_trial` /
    `compose_and_validate`; tests inject stubs.
  - `Scheduler.run()` → `CampaignResult`.
  - `CampaignResult(stop_reason, trial_count, oom_count, best_entry,
    ledger_path)`.
  - `PreflightOutcome` — passed between `preflight_fn` and `runner_fn`.
- **Design notes.**
  - **Preflight failures don't consume trial slots.** The scheduler
    calls the critic, runs preflight, and retries (with feedback) up
    to `preflight_retries`. Only AFTER preflight passes does the
    runner launch.
  - **Preflight-exhaustion behaviour.** If `preflight_retries+1`
    consecutive critic proposals all fail preflight, the scheduler
    writes a synthetic `(FAILED, CONFIG_INVALID)` ledger entry and
    stops with `stop_reason="preflight_exhausted"`. It does NOT
    launch the runner with a known-bad config. (This is the
    Round-3 behaviour change driven by Codex review.)
  - **Plateau** is computed over the last `patience` non-failed
    trials with non-`None` objectives. If every consecutive relative
    improvement is below `epsilon`, stop.
  - **Critic stagnation** requires *two consecutive*
    `stop_requested=True` outputs — a single one is treated as
    "I think you should stop, but I'm not insistent".
  - **History window** for the next critic prompt is capped at
    `history_window=8` (matches AC-7's last-K=8 rule).
  - `clock` is injectable so `budget_seconds_elapsed` tests are
    deterministic.

### 3.11 `__main__.py` — CLI entrypoint (AC-10)

- **Role.** Parse CLI args, wire the production modules into
  `Scheduler`, run the loop, emit `best_config.yaml` +
  `best_trial.json` next to `tuner_ledger.jsonl`.
- **Public API.** `main(argv)` → exit code. Internal helpers:
  `parse_cli_args`, `_run_dry_run_preflight`, `_run_campaign`, the
  three adapter `_preflight_adapter` / `_runner_adapter` /
  `_parser_adapter`, `_emit_best_artefacts`, `_load_fake_critic`,
  `_campaign_id`, `_stable_delta_token`.
- **Design notes.**
  - **Default `ledger_dir`** carries a 6-hex-char `os.urandom` nonce
    in addition to a timestamp + config name, so two same-second
    same-config launches don't collide.
  - **`_campaign_id`** is a 12-char SHA-1 of the absolute ledger_dir
    path. Combined with `trial_idx` it forms the
    `RLINF_TUNER_TRIAL_ID=<campaign>-<idx>` tag the orphan-cleanup
    scan uses. Concurrent campaigns are guaranteed-disjoint.
  - **`_stable_delta_token`** uses
    `sha1(json.dumps(delta, sort_keys=True, default=str))[:8]` so
    dict-valued placement deltas don't crash the log-dir naming (the
    previous `hash(frozenset(delta.items()))` raised `TypeError:
    unhashable type: 'dict'`).
  - **`best_config.yaml`** is the Hydra-COMPOSED (unresolved) YAML —
    `${oc.env:...}` interpolations stay symbolic so the file is
    portable across hosts.
  - **`--dry-run-preflight`** composes the baseline + `max_epochs`
    override and exits 0 without launching anything. Useful for CI /
    sanity checks.
  - **`--fake-critic`** loads a JSON array of
    `{delta, stop_requested?}` entries and instantiates `FakeCritic`.
    If any entry sets `stop_requested=true`, uses `FakeCritic.stop_after`.

### 3.12 `examples/embodiment/run_embodied_tuner.sh` — shim launcher

- **Role.** Wrap `python -m toolkits.embodied_tuner` with the
  `PYTHONPATH` / `REPO_PATH` setup the toolkit needs, matching the
  convention of `examples/embodiment/run_placement_autotune.sh`.
- **Design notes.**
  - Exports `EMBODIED_PATH`, `REPO_PATH`, and prepends `REPO_PATH` to
    `PYTHONPATH`.
  - If `import toolkits.embodied_tuner` fails after the setup, prints
    a structured remediation hint (paths to check, exact command to
    re-try) and exits with code 4.
  - `exec`s the Python module so signals propagate cleanly.

---

## 4. Cross-cutting design decisions

### 4.1 The dual-source rationale rule

**Where:** `CriticOutputValidator` (`critic.py:337+`).
**Why:** Placement decisions are the highest-impact, costliest-to-change
class of mutation. Without a forcing function, an LLM critic will
sometimes propose plausible-sounding placement shifts that are not
grounded in either the coarse MetricTable evidence or the fine-grained
timeline evidence. The dual-source rule turns "be grounded" from a
prompt suggestion into a mechanical contract — invalid outputs are
rejected before they reach preflight, and the critic gets the rejection
reason as feedback on its retry.
**Tested by:** `test_validator_accepts_placement_delta_with_dual_source`,
`test_validator_rejects_placement_delta_missing_{metric,timeline}_citation`,
`test_validator_rejects_empty_citations_for_placement_delta`,
`test_codex_critic_retries_after_dual_source_failure`.

### 4.2 Preflight without Ray

**Where:** `preflight.compose_and_validate` (`preflight.py`).
**Why:** `rlinf.config.validate_cfg` / `validate_embodied_cfg`
instantiate `Cluster()` (`rlinf/config.py:922`) which calls `ray.init`
(`rlinf/scheduler/cluster/cluster.py:332`). Starting Ray to validate a
single config delta is wasteful (~seconds per trial) and breaks on
hosts without Ray's prerequisites. Preflight's contract is "no GPU
work, no Ray". The mirrored divisibility checks at
`rlinf/config.py:962/965/980/1363-1368` are computable from a Hydra-
composed `DictConfig` alone — we don't need `Cluster()` to derive
`actor_world_size` / `env_world_size`, we derive them from
`cluster.component_placement` directly (which is what
`HybridComponentPlacement` does internally anyway).
**Trade-off:** We mirror only the 4 divisibility checks the plan
explicitly tests, not all ~80 assertions in `validate_cfg`. Broader
coverage is a queued polish item (Codex review SUGGESTION).
**Tested by:** `test_baseline_alone_passes_preflight`,
`test_micro_batch_size_violation_rejected`, etc.

### 4.3 Env-tagged orphan cleanup via `/proc`

**Where:** `runner._pids_with_env_match` + `runner._default_pgrep_runner`.
**Why:** `pgrep -f <pattern>` matches *argv*, not environment.
`RLINF_TUNER_TRIAL_ID=<id>` lives in the subprocess env block at
`/proc/<pid>/environ`, not in the argv string. Ray workers spawned by
the trial inherit the env var but their argv typically doesn't carry
it — so `pgrep -f` alone returns no matches and the cleanup silently
misses every orphan. Fix: scan `/proc/<pid>/environ` directly and
union with `pgrep -f`. Recorded as BitLesson
`BL-2026-06-30-pgrep-env-vs-argv`.
**Scope:** POSIX Linux only (graceful fallback to `pgrep -f` on
other platforms). Best-effort; never raises from the cleanup path.

### 4.4 Campaign-unique trial id

**Where:** `__main__._campaign_id` + `_runner_adapter`.
**Why:** Two same-user campaigns on a shared host using
`trial_id="0"`, `"1"`, etc. would have their orphan-cleanup scans
match each other's Ray workers and kill them. We use
`f"{campaign_id}-{trial_idx}"` where `campaign_id` is a 12-hex-char
SHA-1 of the absolute ledger_dir path. And the default ledger_dir
carries a 6-hex-char `os.urandom` nonce so even same-second same-config
launches don't collide on the directory itself.

### 4.5 OOM-before-METRICS_MISSING precedence

**Where:** `parser.parse_trial`.
**Why:** An OOM-killed trial frequently dies before writing
`metrics.log`. If the parser checked metrics presence first, it would
classify those trials as `METRICS_MISSING` — burying the actual
failure mode. The runner already writes a merged stdout+stderr to
`LOG_DIR/run_embodiment.log`; the parser scans it by default and runs
the OOM/crash rubric BEFORE the metrics-presence check whenever
`returncode != 0`.

### 4.6 Append-only ledger with per-line fsync

**Where:** `ledger.Ledger.append`.
**Why:** Tuning campaigns run for hours; a SIGKILL between trials
shouldn't corrupt the prior trial's record. `append()` opens in
`"a"`, writes one line, `flush()`, `os.fsync()`. `load()` tolerates
JSON decode + schema violations per line (counts them in
`skipped_lines`), so a partial write on crash is detected but doesn't
prevent loading the earlier entries.

### 4.7 `(FAILED, NONE)` is an enforced invariant

**Where:** `TrialResult.__post_init__` (`parser.py`).
**Why:** A `FAILED` trial must always carry a non-`NONE` failure
reason — otherwise the audit trail loses the *why* of the failure.
The dataclass raises `ParserInvariantError` at construction time if
the invariant is violated, surfacing bugs immediately rather than
silently losing information.

### 4.8 Preflight-exhausted ≠ trial slot

**Where:** `scheduler.Scheduler.run` + `_record_preflight_exhausted`.
**Why:** If the critic produces `preflight_retries+1` consecutive
invalid deltas, that's a signal the critic is stuck — launching the
last one anyway burns ~20 minutes of GPU time on a config we already
know will fail `validate_cfg` in production. Instead the scheduler
writes a synthetic `(FAILED, CONFIG_INVALID)` ledger entry (preserves
the audit trail) and stops with `stop_reason="preflight_exhausted"`.

---

## 5. Test architecture

```
tests/test_schema.py                       21 tests  (AC-1)
tests/test_placement_enum.py               25 tests  (AC-4)
tests/test_override_wrapper.py             16 tests  (AC-2)
tests/test_preflight.py                    14 tests  (AC-3)
tests/test_runner.py                       15 tests  (AC-5)
tests/test_parser.py                       27 tests  (AC-6)
tests/test_ledger.py                       13 tests  (AC-9)
tests/test_critic.py                       30 tests  (AC-7)
tests/test_scheduler.py                    13 tests  (AC-8)
tests/test_cli.py                          17 tests  (AC-10)
tests/test_smoke.py                         3 tests  (AC-11)
tests/test_no_auto_placement_import.py      7 tests  (AC-11)
────────────────────────────────────────────────────────────
total                                     201 tests
```

All tests are **hermetic** — no real RLinf launch, no Ray, no GPU,
no Codex network call. The classes of test:

- **Unit tests** (`test_{schema,placement_enum,override_wrapper,
  parser,ledger,critic}.py`) — pure-Python assertions on individual
  modules.
- **Subprocess tests** (`test_runner.py`) — use small `python -c
  "import time; time.sleep(...)"` subprocesses to exercise timeout /
  SIGTERM / SIGKILL paths.
- **Hydra-against-real-baseline tests** (`test_preflight.py`,
  `test_cli.py::test_dry_run_preflight_on_real_baseline`) — compose
  the live `maniskill_ppo_openvla.yaml` via Hydra to confirm the
  divisibility checks fire on real config shapes.
- **Scheduler stub tests** (`test_scheduler.py`) — inject
  `runner_fn` / `parser_fn` / `preflight_fn` mocks via a small
  factory so every stop-reason path is verified deterministically.
- **End-to-end smoke** (`test_smoke.py`) — drives the full
  `Scheduler.run()` via `FakeCritic` + mock runner/parser/preflight
  for N=4 synthetic trials, including OOM and parser-crash
  injection.
- **AST-walker import guard** (`test_no_auto_placement_import.py`) —
  walks every `.py` under the toolkit and asserts no
  `toolkits.auto_placement` imports. Includes a planted-offender
  regression test.

Run with: `cd <RLinf root>; PYTHONPATH=. python -m pytest toolkits/embodied_tuner/tests/`.

---

## 6. Extension points

| Extension              | Where to plug in                                                                                          |
|------------------------|------------------------------------------------------------------------------------------------------------|
| New tunable knob       | Add to `_default_domains()` in `schema.py`; the critic prompt auto-renders its legal range; preflight may need a new divisibility check. |
| New pinned knob        | Same as above but set `pinned=True`. The schema rejects it with `KnobNotTunableError`.                    |
| New `FailureMode`      | Add to the `FailureMode` enum in `parser.py` and extend `_classify_failure` if it has a regex trigger.    |
| Alternative critic     | Implement the `Critic` protocol (`critic.py::Critic`). For Bayesian/EA fallback (FUT-3), the scheduler is critic-agnostic — just swap the `critic` argument. |
| Mirror more `validate_cfg` rules | Add new checks in `preflight._check_divisibility`. They must be computable without `Cluster()`. |
| New best-config selection rule | Change `parser.select_best`. The scheduler and ledger both delegate to it.                          |
| Multi-node placement   | Generalise `placement_enum.is_legal_placement` to accept node-qualified ranges. (FUT-1)                   |
| Async pipelined trials | Change `Scheduler.run` to a producer/consumer model. The runner/parser/ledger are already stateless. (FUT-2) |

---

## 7. Integration boundary

### 7.1 RLinf modules used (read-only)

- `rlinf.config.validate_cfg` / `validate_embodied_cfg` — referenced
  by line number, NOT called (would start Ray).
- `rlinf.utils.placement.ModelParallelComponentPlacement` — its
  contiguity rules and `_is_disaggregated` logic are mirrored in
  `placement_enum.is_legal_placement`.
- `rlinf.utils.metric_utils` — MetricTable rendering convention
  (`Step Time = elapsed / steps_done`) drives the parser's interpretation.
- `examples/embodiment/run_embodiment.sh` — unchanged; wrapped by
  `OverrideWrapper`.
- `examples/embodiment/train_embodied_agent.py` — the Hydra entry
  point the wrapper points argv at.
- `profiler/rlinf_timeline/autopatch.py` — emits the JSONL events
  the parser consumes. The runner exports the env flags that turn
  the autopatch on.

### 7.2 RLinf modules deliberately NOT used

- `toolkits.auto_placement` — forbidden by the plan, enforced by an
  AST walker bundled into the smoke-test suite. The toolkit requires
  `config.profile_data` for cold start which embodied configs lack.
- `rlinf.utils.logging.get_logger` — returns `Worker.logger`, a
  Worker-context abstraction. The CLI/orchestrator runs outside any
  Worker so it uses stdlib `logging`.

### 7.3 humanize integration

- `humanize/commands/start-rlcr-loop.md` — the RLCR loop machinery
  that BUILT this tuner over 5 rounds.
- `humanize/commands/ask-codex.md` + `scripts/ask-codex.sh` — the
  Codex transport `CodexCritic` shells out to.
- `mlsys2026-flashinfer-contest/` — the precedent contest workflow
  (`gen-idea → gen-plan → start-rlcr-loop`) this build followed.
- BitLesson: `RLinf/.humanize/bitlesson.md` carries one entry
  (`BL-2026-06-30-pgrep-env-vs-argv`) added during this build.
- Loop artefacts: `RLinf/.humanize/rlcr/2026-06-29_15-37-02/`
  (goal-tracker.md, round-{0,1,2,3,4}-{contract,summary}.md, state.md,
  plan.md).

### 7.4 Repository facts the implementation depends on (worth preserving)

- `run_embodiment.sh` injects only `runner.logger.log_path=${LOG_DIR}`.
  The override wrapper exists because of this — user overrides must
  go AFTER this baseline injection so Hydra precedence honours them.
- `EmbodiedRunner.num_steps_per_epoch = 1` in
  `rlinf/runners/embodied_runner.py`, so `max_epochs=N` yields N
  global steps. The `--max-epochs=3` default + warmup drop exists
  because of this.
- `profiler/enable2.sh` leaves `RLINF_NVITOP=1` / `RLINF_NVML=1`
  commented out by default. The tuner's runner sets them
  unconditionally so memory telemetry is present.
- Hydra search paths in embodied baselines use
  `${oc.env:EMBODIED_PATH}/config/`. Preflight sets `EMBODIED_PATH` /
  `REPO_PATH` before `compose()` so the interpolation resolves.

---

## 8. Origin

This toolkit was specified, built, and validated entirely through the
humanize + RLCR pipeline:

1. **`docs/plan.md`** was generated from `draft_run.md` via
   `/humanize:gen-plan`.
2. The plan was refined via `/humanize:refine-plan` (added the
   dual-source rule; consolidated 15 ACs → 11 ACs).
3. The toolkit was implemented over 4 rounds of
   `/humanize:start-rlcr-loop`, with `/humanize:ask-codex`
   reviewing each round's summary.
4. Codex final review on commit `8eb645b0` issued
   `COMPLETE_ELIGIBLE: yes`.

Commit history (in `RLinf`):

```
5b9e28b9 docs(toolkits/embodied_tuner): add README covering CLI, outputs, and architecture
3f54419c docs(rlcr): mark embodied_tuner RLCR loop COMPLETE in round-4-summary
8eb645b0 fix(toolkits/embodied_tuner): default ledger_dir now carries a random nonce
4e500491 fix(toolkits/embodied_tuner): close 2 IMPORTANT items from final Codex review
ef650e37 feat(toolkits): add embodied_tuner Milestone D — CLI, smoke test, AST walker; fold Codex Round-2 review fixes
7639f2dc feat(toolkits): add embodied_tuner Milestone C — ledger, critic, scheduler
563e8dc6 feat(toolkits): add embodied_tuner Milestone B — trial runner + log/timeline parser
8eb5a9fb feat(toolkits): add embodied_tuner Milestone A — schema, placement enum, override wrapper, preflight
```

For an operator-facing usage guide, see [`README.md`](./README.md).
