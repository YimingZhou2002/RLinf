# embodied_tuner — Architecture & Component Reference

This document describes **how the auto-tuner is built and why**: the
end-to-end data flow, the role of each module, the design decisions
that resolve non-obvious tensions (no-Ray preflight, env-tagged orphan
cleanup, dual-source rationale, cross-window bitter-lesson memory,
wiki-driven prompts, block-tag-aware timeline analytics, etc.), and
the integration boundary with RLinf and the humanize plugin.

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
- Remembers failures **beyond** the critic's rolling context window so
  the same failing delta is not re-proposed once the trial that
  produced it rolls out of history.
- Lets a human operator audit every tuning decision (which signal
  justified which delta) and re-read the exact prompt Codex saw.

### 1.2 The design

- An **LLM critic** (Codex by default via `scripts/ask-codex.sh`;
  Claude also available via `scripts/ask-claude.sh`) proposes the next
  config delta from the last trial's evidence. Placement-touching
  deltas must cite **both** MetricTable evidence AND `timeline/*.jsonl`
  evidence — the *dual-source rationale rule*. A
  `CriticOutputValidator` enforces this mechanically; the critic
  retries up to 3 times with feedback appended.
- The critic prompt is assembled from a **numbered wiki** of markdown
  files under `wiki/` (bottleneck rubric, placement critical paths,
  per-knob optimization directions, timeline signals, hard
  constraints). Editing prompt guidance is a markdown edit, not a
  Python edit.
- After any trial that failed with **OOM, WORKER_CRASH, TIMEOUT, or
  CONFIG_INVALID**, the critic MUST return a `bitter_lesson`
  `{trigger, rule}` payload. The scheduler stamps it with the failed
  trial's index/mode/delta signature and appends the resulting
  `BitterLesson` to a persistent **LessonBook**
  (`<ledger_dir>/bitter_lessons.jsonl`). Every subsequent prompt
  prepends the accumulated lessons — the critic cannot forget a
  failure just because it fell out of the rolling `history_window`.
- A **preflight validator** composes baseline + delta via Hydra and
  runs the targeted divisibility checks from `rlinf/config.py`. It
  deliberately **does not call `validate_cfg` / `validate_embodied_cfg`**
  because those instantiate `Cluster()` which calls `ray.init`, and
  preflight is contractually GPU/Ray-free.
- A **trial runner** launches RLinf in its own POSIX process group,
  enforces a per-trial timeout (default 45 min, `--trial-timeout-seconds`),
  escalates SIGTERM→SIGKILL on hang, invokes `ray stop --force`, and
  sweeps both `pgrep -f` *and* `/proc/<pid>/environ` for orphan
  workers tagged with a campaign-unique trial id. Timeline env vars
  are exported by default; memory-telemetry env vars (`RLINF_NVITOP`,
  `RLINF_NVML`) require opt-in via `--collect-memory`.
- A **log + timeline parser** classifies each trial with
  `(Status, FailureMode)`, computes the objective by averaging
  `step_time` across every MetricTable block (single-step trials are
  measurable), and hands the loaded events to
  `timeline_processor` for the analytical views the critic consumes.
- A dedicated **timeline processor** derives seven analytical views
  from `timeline/*.jsonl`: per-component stall fractions, per-tag
  stats, a per-step critical-path summary, P95 outliers with a
  `knob_hint` derived from the trial's `enable_offload` state, a
  per-component union-busy / bubble breakdown, top-K raw excerpts, and
  a warmup-skipped per-component steady-state call average. Blocking
  wrappers (e.g. `actor/recv_traj`) are declared in `BLOCKING_TAGS`
  and excluded from busy totals so `actor` is not mis-labelled the
  bottleneck when it is actually waiting.
- A **timeline feed selector** picks which raw JSONL trace files are
  embedded verbatim in the prompt. Default `PER_COMPONENT_LATEST` picks
  one representative rank per component (the straggler — most
  informative). `PER_COMPONENT_RANK0`, `ALL`, and `NONE` are also
  supported. A best-effort Gantt PNG (and optional HTML) is rendered
  alongside every trial via `profiler/plot_timeline.py`.
- An **append-only JSONL ledger** persists every trial, including the
  structured `critic_rationale` payload — the **audit trail** that
  lets operators trace WHY each placement decision was taken.
- A **scheduler** orchestrates the loop with explicit budget +
  stopping rules; preflight failures don't consume trial slots; if the
  critic can't produce a valid delta within `preflight_retries`, the
  loop terminates with `preflight_exhausted` rather than launching a
  known-bad config. Every critic exchange is persisted under
  `<trial_log_dir>/critic/attempt-NN-{prompt,response}` so a human
  debugger can replay exactly what the LLM saw and returned.
- A **CLI + shim launcher** glue everything to
  `examples/embodiment/run_embodiment.sh` and emit `best_config.yaml`
  + `best_trial.json` next to the ledger.

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
   │     prev_timeline, prev_num_traj,   │
   │     bitter_lessons,                 │ ◀────────┐
   │     preflight_feedback?)            │          │
   │                                     │          │ feedback string
   │  CodexCritic (or ClaudeCritic)      │          │ on validation fail
   │  builds prompt from wiki/*.md +     │          │
   │  bitter lessons + rolling history + │          │
   │  compact & verbose timeline blocks; │          │
   │  shells out to scripts/ask-*.sh;    │          │
   │  parses JSON; CriticOutputValidator │          │
   │  enforces:                          │          │
   │   - knob schema                     │          │
   │   - dual-source rule (placement)    │          │
   │   - bitter_lesson after OOM /       │          │
   │     WORKER_CRASH / TIMEOUT /        │          │
   │     CONFIG_INVALID                  │          │
   │   - retry up to 3x with feedback    │          │
   │  Every attempt (prompt+response)    │          │
   │  is captured on transaction_log.    │          │
   └─────────────────┬───────────────────┘          │
                     │ CriticOutput                 │
                     │ {delta, rationale,           │
                     │  stop_requested,             │
                     │  bitter_lesson?}             │
                     ▼                              │
   ┌─────────────────────────────────────┐          │
   │ (2b) Scheduler.persist_critic_      │          │
   │      transactions → writes          │          │
   │   <log_dir>/critic/attempt-NN-      │          │
   │   {prompt.md, response.txt}         │          │
   └─────────────────┬───────────────────┘          │
                     │                              │
                     ▼                              │
   ┌─────────────────────────────────────┐          │
   │ (2c) Scheduler.record_bitter_lesson │          │
   │      (only after a failed prev      │          │
   │      trial). Stamps trial_idx /     │          │
   │      failure_mode / delta_signature │          │
   │      and appends to LessonBook.     │          │
   │      Dedup by (mode, signature);    │          │
   │      LRU-cap at 30.                 │          │
   └─────────────────┬───────────────────┘          │
                     │                              │
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
   │ wait/timeout        │
   │  (default 2700s) /  │
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
   │  - reads timeline/*.jsonl (via      │
   │    timeline_processor)              │
   │  - OOM/crash rubric BEFORE          │
   │    METRICS_MISSING on returncode≠0  │
   │  - objective = avg(step_time[1..N]) │
   │             / num_trajectories      │
   │  - runs 7 analytical views:         │
   │    stall_fractions, tag_stats,      │
   │    critical_path (A'),              │
   │    per_component_bubble (D'),       │
   │    outliers (C'), raw_excerpts,     │
   │    component_call_averages          │
   │  - collects raw JSONL for feed      │
   │  - renders timeline.png / .html     │
   │  - attaches knob_hint via effective │
   │    enable_offload state (read from  │
   │    tensorboard/config.yaml)         │
   └─────────────────┬───────────────────┘
                     │ TrialResult
                     │ {status, failure_mode,
                     │  objective, summary,
                     │  peak_gpu_mem_gib, ...}
                     ▼
   ┌─────────────────────────────────────┐
   │ (6) Ledger.append(LedgerEntry)      │
   │                                     │
   │  append-only JSONL; fsync per line; │
   │  structured critic_rationale        │
   │  persisted verbatim; timeline_      │
   │  summary carries per-view payloads  │
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
`TrialResult`, `LedgerEntry`, `CriticOutput`, `BitterLesson`,
`ProposedLesson`, `PreflightOutcome`) defined as frozen dataclasses for
clarity and immutability.

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
  - `TrialRunner(timeout_seconds=..., disable_profiler=..., disable_memory_telemetry=...)`.
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
  - **Profiler env vars.** Timeline flags (`RLINF_TIMELINE`,
    `RLINF_TIMELINE_WORKER_TIMER`, `RLINF_TIMELINE_ACTOR_TRAINING`,
    `RLINF_TIMELINE_DIR=auto`) are on by default and can be
    turned off via `disable_profiler` (CLI `--no-profiler`).
    Memory-telemetry flags (`RLINF_NVITOP`, `RLINF_NVML`) are
    opt-in via `disable_memory_telemetry=False` (CLI
    `--collect-memory`); the default is *off* because a long
    campaign accumulates hundreds of MB of nvitop/NVML traces.

### 3.6 `parser.py` — log + timeline parsing (AC-6)

- **Role.** Turn a trial's `LOG_DIR/{metrics.log, timeline/*.jsonl,
  nvitop/*}` into a structured `TrialResult` the scheduler and critic
  can consume. Delegates every derivation from `timeline/*.jsonl` to
  `timeline_processor`.
- **Public API.**
  - `parse_trial(log_dir, returncode=None, timed_out=False,
    failure_mode_override=None, stderr_path=None,
    enable_offload=None, jsonl_feed_mode=None,
    plot_formats=("png",))` → `TrialResult`.
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
  - Objective averaging: average `step_time` across every parsed
    MetricTable block. Single-step trials are measurable (their lone
    `step_time` is used directly); with `--max-epochs=3` all three
    blocks are averaged.
  - `num_trajectories` comes from the **final** MetricTable block (no
    silent fallback denominator). It is passed to the critic prompt
    verbatim as the objective's denominator.
  - `select_best` requires `(OK, NONE)` strictly — `(OK,
    METRICS_PARTIAL)` is critic context, not a best-config candidate.
  - `TimelineSummary` carries seven views produced by
    `timeline_processor` (`per_tag`, `stall_fraction_by_component`,
    `critical_path`, `outliers`, `per_component_bubble`,
    `raw_excerpts`, `component_call_averages`) plus the raw JSONL
    payload selected by `timeline_feed.collect_raw_jsonl` and the
    Gantt plot paths returned by
    `timeline_feed.render_default_plots`.

### 3.7 `timeline_processor.py` — timeline analytics (new)

- **Role.** Owns every derivation from `timeline/*.jsonl`. The parser
  is a thin adapter that calls this module and packs the results into
  `TimelineSummary`.
- **Public API.**
  - `load_events(timeline_dir)` — read + normalize every JSONL event.
  - `is_blocking(event)`, `BLOCKING_TAGS`, `EXCLUDED_COMPONENTS`.
  - `compute_stall_fractions(events)` — per-component fraction of the
    observation window NOT covered by any of its events.
  - `compute_tag_stats(events, headline_tags)` — per-`(component,
    rank, tag)` count / min / median / max / total duration.
  - `compute_critical_path(events)` (A') — per-`global_step` summary
    of `(component, rank)` lanes ranked by REAL busy time, with a
    matching `blocked_s` column.
  - `compute_outliers(events, enable_offload=None)` (C') — top-K
    events above per-tag P95 and above 1 s, each carrying a
    `knob_hint` derived from the trial's `enable_offload` state
    (e.g. `env.enable_offload=False → try True`).
  - `compute_per_component_bubble(events)` (D') — per-component union
    of REAL busy intervals across all its ranks + complementary
    bubble fraction; per-rank detail for straggler diagnosis.
  - `extract_raw_excerpts(events, k)` — top-K longest raw events
    verbatim so the critic can inspect full call context.
  - `compute_component_call_averages(events, components=("env",
    "rollout"), skip_first=2)` — steady-state per-call duration after
    dropping bootstrap warmup calls.
  - `process_timeline(timeline_dir, enable_offload=None)` — entry
    point returning all view payloads in one dict.
- **Design notes.**
  - `BLOCKING_TAGS` (e.g. `actor/recv_traj`,
    `actor/recv_rollout_trajectories`, `rollout/generate`,
    `rollout/generate_one_epoch`, `env/interact`) look like busy
    intervals but are actually component A waiting on component B.
    Excluding them from A'/D' is the difference between the critic
    seeing "actor is 97% busy" (wrong) and "actor is mostly idle
    waiting for rollout" (correct).
  - `EXCLUDED_COMPONENTS = {"runner"}` because runner emits a single
    `run` event spanning the whole trial — including it makes every
    per-component ranking trivial.
  - `DEFAULT_OUTLIER_K = 12`, `DEFAULT_RAW_EXCERPTS_K = 15`,
    `OUTLIER_MIN_SECONDS = 1.0`. Bumping these inflates every prompt
    token-for-token; the defaults keep the four new sections together
    under ~3 KB.
  - `DEFAULT_SKIP_FIRST_CALLS = 2` for
    `compute_component_call_averages` — the first two per-component
    calls carry offload page-in, JIT compile, and first-CUDA-kernel
    init cost and would otherwise inflate the mean by orders of
    magnitude.
  - The view suite is deliberately concrete — each view answers a
    specific question the critic prompt asks in a labelled block. If
    a new question surfaces during a campaign (e.g. "why is env rank
    3 late?"), add a new view here, then wire a matching
    `_render_*` block in `critic._render_timeline_verbose`.

### 3.8 `timeline_feed.py` — raw JSONL selection + Gantt rendering (new)

- **Role.** Two responsibilities: pick which raw JSONL files the
  critic sees, and render a best-effort Gantt plot after every trial.
- **Public API.**
  - `JsonlFeedMode` enum: `PER_COMPONENT_LATEST` (default —
    per-component pick the rank whose last event ends latest),
    `PER_COMPONENT_RANK0`, `ALL`, `NONE`.
  - `collect_raw_jsonl(timeline_dir, mode=..., max_bytes_per_file=None,
    selector=None)` → `dict[str, str]` mapping file stem to text.
  - `render_timeline_plot(timeline_dir, output_path=None, fmt="png",
    timeout_seconds=120, extra_args=())` — invokes
    `profiler/plot_timeline.py`.
  - `render_default_plots(timeline_dir, formats=("png",))` — convenience
    wrapper for the common "PNG for the critic + HTML for humans" case.
- **Design notes.**
  - `PER_COMPONENT_LATEST` reads only the tail of each JSONL to fetch
    the last `t1` — no whole-file reads. This is the informative-but-
    bounded default. `ALL` is included so we can flip a knob later
    when long-context critics land, but it's ~600 K tokens at 8 GPUs
    and is not the default for that reason.
  - `render_timeline_plot` is best-effort: every failure (missing
    script, subprocess crash, timeout) is logged and `None` returned.
    The trial loop must never abort because a plot didn't render.
  - Plotting is text-only in the current critic transport — the path
    is surfaced to the prompt via `## Last trial — timeline Gantt
    renders`. A future multimodal critic can lift `plot_paths["png"]`
    into an image content block.

### 3.9 `ledger.py` — append-only JSONL persistence (AC-9)

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
  - `timeline_summary` in the ledger is the full seven-view payload
    from `TimelineSummary` (per_tag, stall_fraction_by_component,
    critical_path, outliers, per_component_bubble, raw_excerpts,
    raw_jsonl, plot_paths, component_call_averages) so an operator
    can replay the exact numbers Codex saw for any past round.

### 3.10 `lessons.py` — persistent BitterLesson store (new)

- **Role.** Give the critic a permanent, cross-window memory of
  failed trials so the same failing delta is not re-proposed once
  it falls out of `history_window`.
- **Public API.**
  - `BitterLesson(trigger, rule, trial_idx, failure_mode,
    delta_signature)` — frozen dataclass. `from_dict` validates every
    required field.
  - `LessonBook(path, max_lessons=30, fsync_on_append=True)` —
    append-only JSONL store. Methods: `load()`, `all()`,
    `add(lesson)` (returns whether inserted), `extend(iterable)`.
  - `canonical_delta_signature(delta)` — `json.dumps(dict(delta),
    sort_keys=True, default=_json_default)`; recurses into nested
    mappings so `component_placement` deltas dedup correctly.
  - `LessonSchemaError` — raised on malformed on-disk payloads.
- **Design notes.**
  - **Why this exists.** The `maniskill_ppo_openvla` campaign showed
    the critic proposing the same
    `rollout.enable_offload=False` OOM three times in a row because
    the failing trial rolled out of the 8-round `history_window`. A
    persistent lesson survives that window.
  - **Dedup key** is `(failure_mode, delta_signature)`. Two proposals
    of the same delta under the same failure mode are one lesson.
  - **LRU cap of 30 lessons.** When adding pushes over the cap, the
    oldest lesson (by `trial_idx`, tie-broken by insertion order) is
    dropped in memory and an `{"__evicted__": true, ...}` marker is
    appended to the file so the on-disk audit trail is complete.
  - **Persistence rules mirror `Ledger`.** Per-line fsync, tolerance
    of JSON decode + schema violations per line, corrupted lines
    counted but not fatal to the rest of the file.
  - The critic writes only `{trigger, rule}` in its
    `ProposedLesson`; the scheduler stamps `trial_idx`,
    `failure_mode`, and `delta_signature` from the actual failed
    trial before handing the assembled `BitterLesson` to
    `LessonBook.add`. This split prevents the critic from lying
    about which trial the lesson came from.

### 3.11 `critic.py` — prompt + validator + Codex/Claude transport (AC-7)

- **Role.** Turn parsed trial outcomes into a structured prompt;
  invoke the chosen backend; parse the JSON response; enforce the
  dual-source rule AND the mandatory-bitter-lesson rule; retry on
  validator failures. Persist every attempt for later human audit.
- **Public API.**
  - `CriticPrompt` — frozen dataclass with section blocks
    (`wiki_block`, `bitter_lessons_block`, `history_block`,
    `current_knobs_block`, `constraints_block`,
    `memory_pressure_block`, `metric_summary_block` (compact),
    `timeline_verbose_block`, `schema_doc`, `feedback_block`).
    `__str__` assembles the full prompt; `to_debug_text()` renders a
    compact form (wiki + schema doc + verbose timeline excluded) used
    for the per-attempt `attempt-NN-prompt.md` file that gets
    persisted next to the trial.
  - `build_prompt(history, current_knobs, schema, last_failure_mode,
    last_metric_summary, last_timeline_summary,
    last_num_trajectories=None, bitter_lessons=(),
    feedback=None, preflight_feedback=None)` → `CriticPrompt`.
  - `Rationale` — frozen `{summary, metric_table_citations,
    timeline_citations}`.
  - `ProposedLesson` — frozen `{trigger, rule}`. The critic-facing
    half of `BitterLesson`.
  - `CriticOutput` — frozen `{delta, rationale, stop_requested,
    bitter_lesson}`.
  - `TrialHistoryEntry` — per-trial summary the critic prompt repeats
    (per `history_window`).
  - `parse_critic_output(text)` → `CriticOutput`. Tolerates raw JSON,
    Markdown code fences, and brace-bracketed JSON. **Rejects** bare
    strings / non-list / non-string-element citation arrays. Coerces
    `bitter_lesson` if present.
  - `CriticOutputValidator(schema, last_failure_mode)` with
    `.validate(output)` → `ValidationResult`. Enforces the dual-source
    rule AND (when `last_failure_mode` is OOM / WORKER_CRASH /
    TIMEOUT / CONFIG_INVALID) the mandatory bitter-lesson rule.
  - `Critic` Protocol with a `propose(...)` method taking
    `bitter_lessons` and `last_num_trajectories`.
  - `CodexCritic` — production implementation shelling out to
    `scripts/ask-codex.sh`. Retries up to `max_retries` (default 3) on
    JSON or validator failure, with the failure reason appended as
    feedback. `transaction_log` field captures per-attempt
    `{attempt, prompt_debug, response, parse_error?, validation_ok,
    validation_reason}` for the scheduler to persist.
- **Design notes.**
  - **Wiki-driven prompt.** The static optimization context lives in
    numbered markdown files under `wiki/` (`01-concepts.md`
    → `08-gotchas.md`, loaded in that order). Editing prompt
    guidance is a markdown edit — Python constants no longer hold it.
    A missing wiki file is a build error (raised at import time), not
    a runtime warning.
  - **Dual-source rule.** When the proposed `delta` contains
    `cluster.component_placement`, the rationale MUST contain at least
    one non-empty `metric_table_citations` entry AND at least one
    non-empty `timeline_citations` entry. Otherwise, the validator
    rejects and the critic retries. Non-placement deltas require a
    non-empty `rationale.summary` (a lower bar, matching the fact that
    non-placement mutations are cheaper to undo).
  - **Mandatory bitter-lesson rule.** When
    `last_failure_mode ∈ {OOM, WORKER_CRASH, TIMEOUT, CONFIG_INVALID}`
    and the response is not `stop_requested`, the validator requires a
    populated `bitter_lesson.trigger` AND `bitter_lesson.rule`. Without
    this the persistent memory silently drops the round's lesson.
  - **Citation type validation.** `parse_critic_output` rejects bare
    strings in `metric_table_citations` because the dual-source check
    would otherwise iterate a string character-by-character (each char
    "non-empty"), silently bypassing the rule.
  - **Preflight feedback wiring.** `build_prompt` accepts a
    `preflight_feedback` block; the scheduler passes preflight
    rejection reasons via this channel so the critic actually learns
    from divisibility / placement violations.
  - **Transport injection.** `CodexCritic.transport` is injectable for
    tests (no real Codex call); production uses `subprocess.run`
    against `scripts/ask-codex.sh` (or `ask-claude.sh` when
    `--critic-backend=claude`).
  - **Transaction persistence.** Every `propose(...)` clears and
    populates `transaction_log`. The scheduler drains it into
    `<log_dir>/critic/attempt-NN-{prompt.md, response.txt}` — the
    prompt file uses `to_debug_text()` so it stays inspection-friendly
    (~4 KB) instead of dumping the ~36 KB wiki+verbose timeline.

### 3.12 `fake_critic.py` — deterministic test critic (AC-7 helper)

- **Role.** A deterministic `Critic` for tests and the AC-11 smoke
  harness. No LLM, no network.
- **Public API.**
  - `FakeCritic(outputs=[CriticOutput, ...])`.
  - `FakeCritic.from_deltas(*deltas)` — builds a critic that returns
    each delta with a minimal valid rationale (and a placeholder
    `bitter_lesson` when needed to satisfy the validator).
  - `FakeCritic.stop_after(*deltas)` — same, but the final response
    has `stop_requested=True`.
  - `.calls` captures `(history_len, current_knobs,
    preflight_feedback, bitter_lesson_count)` tuples so tests can
    assert on what the critic was called with.
- **Design notes.** Lives in a separate file so it can be imported
  independently (the smoke test composes it via the CLI's
  `--fake-critic` flag).

### 3.13 `scheduler.py` — campaign orchestration (AC-8)

- **Role.** Glue critic + preflight + runner + parser + ledger +
  lesson book into the per-trial loop. Own the budget and the
  stopping rules. Persist critic transactions and bitter lessons at
  the right moment in the loop.
- **Public API.**
  - `BudgetConfig` — defaults `max_trials=20, budget_seconds=43200,
    max_oom=5, patience=3, epsilon=0.02, preflight_retries=3,
    history_window=8`.
  - `Scheduler(critic, runner_fn, parser_fn, preflight_fn, ledger,
    budget, baseline_knobs, clock, lesson_book=None)`. `runner_fn` /
    `parser_fn` / `preflight_fn` are callable injection points —
    production wires them to the real `TrialRunner.launch` /
    `parse_trial` / `compose_and_validate`; tests inject stubs.
    `lesson_book` defaults to a `LessonBook` at
    `<ledger.path.parent>/bitter_lessons.jsonl`.
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
    launch the runner with a known-bad config.
  - **Plateau** is computed over the last `patience` non-failed
    trials with non-`None` objectives. If every consecutive relative
    improvement is below `epsilon`, stop.
  - **Critic stagnation** requires *two consecutive*
    `stop_requested=True` outputs — a single one is treated as
    "I think you should stop, but I'm not insistent". A
    `stop_requested` proposal does NOT burn a trial slot.
  - **History window** for the next critic prompt is capped at
    `history_window=8` (matches AC-7's last-K=8 rule). BitterLessons
    survive outside this window and are prepended to every prompt.
  - **Bitter-lesson fold** happens BEFORE running the next trial (not
    after). The lesson describes the *previous* failure; delaying its
    persistence by one round would let the critic re-propose the same
    failing delta in the very round its lesson should have prevented.
  - **Critic transaction persistence.** After `_propose_with_preflight`
    returns, the scheduler drains `critic.transaction_log` into
    `<preflight_outcome.log_dir>/critic/`. Best-effort; a failure here
    must not abort the trial loop.
  - `clock` is injectable so `budget_seconds_elapsed` tests are
    deterministic.

### 3.14 `__main__.py` — CLI entrypoint (AC-10)

- **Role.** Parse CLI args, wire the production modules into
  `Scheduler`, run the loop, emit `best_config.yaml` +
  `best_trial.json` next to `tuner_ledger.jsonl`.
- **Public API.** `main(argv)` → exit code. Internal helpers:
  `parse_cli_args`, `_run_dry_run_preflight`, `_run_campaign`, the
  three adapter `_preflight_adapter` / `_runner_adapter` /
  `_parser_adapter`, `_emit_best_artefacts`, `_load_fake_critic`,
  `_campaign_id`, `_stable_delta_token`, `_extract_trial_context`,
  `_build_critic`.
- **CLI flags** (`bash examples/embodiment/run_embodied_tuner.sh --help`):
  - `--config` (required) / `--baseline` (defaults to
    `examples/embodiment/config/<config>.yaml`)
  - `--max-trials` (default 20) / `--budget-seconds` (default 43200 = 12h)
  - `--trial-timeout-seconds` (default 2700 = 45 min)
  - `--max-oom` (default 5) / `--patience` (default 3) / `--epsilon` (default 0.02)
  - `--max-epochs` (default 3) — passed as `runner.max_epochs=` to every trial
  - `--collect-memory` / `--no-collect-memory` (default OFF)
  - `--no-profiler` — opt out of timeline env exports (default on)
  - `--critic-backend {codex,claude}` (default `codex`) — picks which
    vendored `scripts/ask-*.sh` script to use
  - `--ask-codex-path <path>` — explicit script override (kept for
    backwards compatibility; the flag name predates the multi-backend
    switch)
  - `--dry-run-preflight` — compose baseline + `runner.max_epochs`
    override and exit 0 without launching anything
  - `--fake-critic <json>` — bypass the LLM with a JSON array of
    `{delta, stop_requested?}` entries
  - `--ledger-dir <path>` — override the default timestamped output dir
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
  - **`_extract_trial_context`** reads the trial's resolved config
    from `<log_dir>/tensorboard/config.yaml` (the file the Tensorboard
    sidecar consumes) to recover the trial's effective
    `enable_offload` state per component. This state is threaded into
    `parse_trial(enable_offload=...)` so
    `timeline_processor.compute_outliers` can attach a `knob_hint`
    ("env.enable_offload=False → try True") to each outlier row.
    When the file is absent (test fixtures / dry-runs) the hints are
    simply omitted.
  - **`best_config.yaml`** is the Hydra-COMPOSED (unresolved) YAML —
    `${oc.env:...}` interpolations stay symbolic so the file is
    portable across hosts.
  - **`--dry-run-preflight`** composes the baseline + `max_epochs`
    override and exits 0 without launching anything. Useful for CI /
    sanity checks.
  - **`--fake-critic`** loads a JSON array of
    `{delta, stop_requested?}` entries and instantiates `FakeCritic`.
    If any entry sets `stop_requested=true`, uses `FakeCritic.stop_after`.
  - **`--critic-backend`** is a thin selector: `codex` maps to
    `scripts/ask-codex.sh`, `claude` maps to `scripts/ask-claude.sh`.
    Only takes effect when `--ask-codex-path` is left at its default.

### 3.15 `wiki/` — critic optimization context (new)

- **Role.** Natural-language optimization context loaded verbatim into
  every critic prompt. Complements the mechanical inputs (MetricTable,
  timeline JSONL, current knob values, trial history) with a concepts
  glossary, placement critical paths, prompt-input contract, timeline
  signal glossary, decision recipe, per-knob playbook, hard-constraint
  list, and consolidated anti-patterns.
- **Files** (loaded in numeric order by `critic._load_wiki_context`):
  - [`README.md`](./wiki/README.md) — read order + editorial rules
  - [`01-concepts.md`](./wiki/01-concepts.md) — objective, notation
    (`T_env / T_rol / T_act / T_sync / R`), placement modes, runner
    modes, and the trajectory-scaling model
  - [`02-paths.md`](./wiki/02-paths.md) — critical-path formula per
    `(placement, runner mode)` combination with quick-reference table
    and timeline signatures
  - [`03-inputs.md`](./wiki/03-inputs.md) — the data blocks the critic
    receives, their schemas, block-priority-on-conflict rule, and
    `FailureMode` catalog
  - [`04-signals.md`](./wiki/04-signals.md) — timeline tag reference,
    `stall_fraction` semantics, wrapper tags to ignore, missing-data
    handling
  - [`05-recipe.md`](./wiki/05-recipe.md) — step-by-step decision
    flow: identify → locate → verify → choose → validate → cite;
    dual-source and failed-trial revert-bundle rules
  - [`06-playbook.md`](./wiki/06-playbook.md) — per-knob playbook:
    what each knob moves, when to grow/shrink, per-knob preflight
    checks, cross-knob patterns
  - [`07-constraints.md`](./wiki/07-constraints.md) — Tier 1 preflight
    and Tier 2 runtime rules with per-delta-type checklists; §2.6
    documents the routing-divisibility trap that produces synthetic
    `DIVISIBILITY_VIOLATION`
  - [`08-gotchas.md`](./wiki/08-gotchas.md) — consolidated
    anti-patterns with cross-references to the rule detail
- **Design notes.**
  - **Why markdown, not Python.** Prompt tuning is workload knowledge
    (which knob helps which bottleneck), not code. Keeping it in
    markdown lets a domain expert edit without touching `critic.py`.
    The critic prompt builder reads each file verbatim into a
    `wiki_block` section — no templating, no substitution.
  - **A missing wiki file is a build error.** `_read_wiki_file` raises
    `FileNotFoundError` at import time so tests and real trials fail
    fast rather than producing a silently-weaker critic prompt.
  - **The wiki is excluded from `to_debug_text()`** so per-attempt
    `attempt-NN-prompt.md` files stay compact. The debug view is what
    a human reads when auditing a decision; the full wiki is
    reproducible from the git-tracked files.

### 3.16 `scripts/ask-codex.sh` + `scripts/ask-claude.sh` — vendored transports (new)

- **Role.** Shell scripts that take a prompt on stdin, invoke the
  respective LLM binary, and stream the response to stdout. Vendored
  next to the toolkit so a shared-humanize install is not required.
- **Design notes.**
  - `CodexCritic.ask_codex_path` defaults to the script sitting next
    to the module (`scripts/ask-codex.sh`); `--critic-backend=claude`
    swaps it for `scripts/ask-claude.sh`.
  - The scripts mirror the interface of humanize's `ask-codex.sh` /
    `ask-claude.sh`; if the shared version diverges, this vendored
    copy can be updated independently.

### 3.17 `profiler/` — vendored timeline + memory profiler (new)

- **Role.** Vendored copy of the `rlinf_timeline` autopatch machinery,
  the NVITOP / NVML samplers, and the Gantt/memory plotting scripts
  (`plot_timeline.py`, `plot_nvitop.py`, `plot_nvml.py`).
- **How the tuner uses it.**
  - `TrialRunner._with_profiler_env` exports
    `RLINF_TIMELINE*` (default on) and `RLINF_NVITOP` /
    `RLINF_NVML` (opt-in) so the autopatch turns on without the user
    having to `source profiler/enable2.sh`.
  - `timeline_feed.render_default_plots` invokes
    `profiler/plot_timeline.py` via `subprocess.run` (best-effort;
    every failure is logged and `None` returned).
- See [`profiler/README.md`](./profiler/README.md) for the standalone
  usage of these tools (which pre-dates the tuner).

### 3.18 `examples/embodiment/run_embodied_tuner.sh` — shim launcher

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

**Where:** `CriticOutputValidator` (`critic.py`).
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

### 4.2 The mandatory-bitter-lesson rule

**Where:** `CriticOutputValidator._LESSON_REQUIRED_FAILURE_MODES`
(`critic.py`), `Scheduler._record_bitter_lesson` (`scheduler.py`),
`LessonBook` (`lessons.py`).
**Why:** The critic sees only the last `history_window=8` trials in its
prompt. In the maniskill campaign, the critic proposed the same
`rollout.enable_offload=False` OOM three times in a row because the
failing trial rolled out of that window. Rather than growing the window
(which makes every prompt heavier), we force the critic to *write down
what it just learned* after every OOM/WORKER_CRASH/TIMEOUT/CONFIG_INVALID
failure. Those lessons are appended to `<ledger_dir>/bitter_lessons.jsonl`
and prepended to every subsequent prompt — even long after the failing
trial has fallen out of history. Two lessons with the same `(failure_mode,
delta_signature)` collapse to one so the block stays bounded, and an
LRU cap of 30 prevents runaway campaigns from ballooning the prompt.
**Tested by:** `test_lessons.py` (dedup, LRU eviction, canonical
signature recursion), `test_critic.py` (validator refuses missing
lessons after failed trials), `test_scheduler.py` (fold happens before
the next trial, not after).

### 4.3 Preflight without Ray

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

### 4.4 Env-tagged orphan cleanup via `/proc`

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

### 4.5 Campaign-unique trial id

**Where:** `__main__._campaign_id` + `_runner_adapter`.
**Why:** Two same-user campaigns on a shared host using
`trial_id="0"`, `"1"`, etc. would have their orphan-cleanup scans
match each other's Ray workers and kill them. We use
`f"{campaign_id}-{trial_idx}"` where `campaign_id` is a 12-hex-char
SHA-1 of the absolute ledger_dir path. And the default ledger_dir
carries a 6-hex-char `os.urandom` nonce so even same-second same-config
launches don't collide on the directory itself.

### 4.6 OOM-before-METRICS_MISSING precedence

**Where:** `parser.parse_trial`.
**Why:** An OOM-killed trial frequently dies before writing
`metrics.log`. If the parser checked metrics presence first, it would
classify those trials as `METRICS_MISSING` — burying the actual
failure mode. The runner already writes a merged stdout+stderr to
`LOG_DIR/run_embodiment.log`; the parser scans it by default and runs
the OOM/crash rubric BEFORE the metrics-presence check whenever
`returncode != 0`.

### 4.7 Append-only ledger with per-line fsync

**Where:** `ledger.Ledger.append`.
**Why:** Tuning campaigns run for hours; a SIGKILL between trials
shouldn't corrupt the prior trial's record. `append()` opens in
`"a"`, writes one line, `flush()`, `os.fsync()`. `load()` tolerates
JSON decode + schema violations per line (counts them in
`skipped_lines`), so a partial write on crash is detected but doesn't
prevent loading the earlier entries. The same policy is applied
verbatim to the LessonBook (`lessons.LessonBook._write_line`).

### 4.8 `(FAILED, NONE)` is an enforced invariant

**Where:** `TrialResult.__post_init__` (`parser.py`).
**Why:** A `FAILED` trial must always carry a non-`NONE` failure
reason — otherwise the audit trail loses the *why* of the failure.
The dataclass raises `ParserInvariantError` at construction time if
the invariant is violated, surfacing bugs immediately rather than
silently losing information.

### 4.9 Preflight-exhausted ≠ trial slot

**Where:** `scheduler.Scheduler.run` + `_record_preflight_exhausted`.
**Why:** If the critic produces `preflight_retries+1` consecutive
invalid deltas, that's a signal the critic is stuck — launching the
last one anyway burns ~20 minutes of GPU time on a config we already
know will fail `validate_cfg` in production. Instead the scheduler
writes a synthetic `(FAILED, CONFIG_INVALID)` ledger entry (preserves
the audit trail) and stops with `stop_reason="preflight_exhausted"`.

### 4.10 Blocking-tag-aware timeline analytics

**Where:** `timeline_processor.BLOCKING_TAGS` +
`compute_critical_path` / `compute_per_component_bubble`.
**Why:** Timeline events like `actor/recv_traj` and
`actor/recv_rollout_trajectories` *look* like busy intervals but are
actually the actor blocked waiting on rollout+env to produce
trajectories. If we count them as real work, the critic sees "actor
is 97% busy" and rules out actor-side deltas — when actor is in fact
mostly idle. Every A' / D' aggregation subtracts blocking events
before computing busy totals. The prompt renders a
`_BLOCKING_TAGS_EXPLAINER` note so the LLM interprets `real_s` vs
`blocked_s` correctly. Extending the block list is a domain decision
and requires human review (per plan BitLesson).

### 4.11 Wiki-in-git, not string constants in Python

**Where:** `wiki/*.md` + `critic._load_wiki_context`.
**Why:** The earlier build had a `_BOTTLENECK_RUBRIC` string constant
inline in `critic.py`. Every prompt edit needed a Python change, and
non-Python contributors couldn't touch prompt guidance. Moving the
context to numbered markdown files under `wiki/` makes editing the
prompt an ordinary docs edit; the same files are readable as
standalone documentation for anyone learning the tuner. A missing
wiki file raises at import time — no silent degradation to a
weaker prompt.

### 4.12 Critic transaction persistence

**Where:** `scheduler.Scheduler._persist_critic_transactions` +
`critic.CodexCritic.transaction_log`.
**Why:** Reading a ledger entry tells you WHAT the critic proposed and
WHY (via `critic_rationale`). It does not tell you what the critic
*saw* nor what it originally returned before the dual-source /
bitter-lesson retry loop reshaped the response. The scheduler drains
`transaction_log` after every `propose(...)` into
`<log_dir>/critic/attempt-NN-{prompt.md, response.txt}` so a human
debugger can replay the full conversation. The prompt file uses
`CriticPrompt.to_debug_text()` (compact form, wiki+schema-doc+verbose-
timeline excluded) so the persisted file stays inspection-friendly
instead of a 36 KB dump.

### 4.13 Multi-backend critic transport

**Where:** `__main__._build_critic` + `--critic-backend {codex,claude}`
+ `scripts/ask-{codex,claude}.sh`.
**Why:** Codex and Claude have different failure modes on the
structured-JSON contract the tuner requires. Rather than hard-code one
provider, the CLI picks the transport script by name and hands it to
`CodexCritic.ask_codex_path`. The class name is kept for backwards
compatibility; the wiring is generic (any script that reads a prompt
on stdin and writes the response on stdout works).

### 4.14 Memory telemetry off by default

**Where:** `__main__` (`--collect-memory` mutex group; default `False`)
+ `TrialRunner.disable_memory_telemetry`.
**Why:** A 12-hour, 20-trial campaign accumulates hundreds of megabytes
of nvitop/NVML JSONL. That's fine for a single debug run but wasteful
for routine campaigns, and the tuner doesn't consume the samples during
the loop (only the operator does, post-hoc). Timeline env exports
remain on by default because `timeline/*.jsonl` IS consumed every
round by the critic.

---

## 4.15 DAG-structured trial store (NodeStore, dedup, rollback, prompt view)

Alongside the flat `Ledger`, the tuner maintains a **DAG-structured
NodeStore** that persists every trial as a graph node with an
explicit `parent_id`. The DAG substrate powers three related
mechanisms — parent-rollback on failure, resolved-config dedup, and
a DAG-aware Codex prompt view — that could not be built on the flat
Ledger alone. All three are wired in coexistence mode: the flat
Ledger continues to be written unchanged (so existing consumers such
as `_emit_best_artefacts` and `plot_step_time_vs_trajectories` are
untouched), while the NodeStore is authoritative for DAG state.

### 4.15.1 On-disk schema (`nodes.jsonl`, `config_dedup_index.jsonl`)

Two sidecar files land next to `tuner_ledger.jsonl`:

- `nodes.jsonl`: one `DAGNode` per line, append-only, fsync-per-write,
  corruption-tolerant load (mirrors the `LessonBook` persistence
  contract). A node is written to disk ONLY after its final
  `(status, failure_mode)` is known — the scheduler never persists an
  in-flight `RUNNING` state. There is **no** semantic LRU eviction:
  authoritative node retention is a correctness requirement (ancestor
  walk, dedup rebuild, active-leaf recovery all need it).
- `config_dedup_index.jsonl`: one `DedupEntry` per line, keyed by
  `resolved_config_sha`. Rebuildable from `nodes.jsonl` at scheduler
  startup so a lost / corrupt sidecar is a warning, not a
  campaign-level failure. First-write-wins on any given SHA so
  `origin_node_id` always points at the ORIGINAL non-duplicate node
  for that SHA — never at another `DUPLICATE_OF` synthetic entry.

Integrity guards enforced by `NodeStore.append`:

- reject `status == "RUNNING"` (or any non-terminal status);
- reject duplicate `node_id` (no in-place updates);
- reject `parent_id` referencing a node not already in the store;
- reject cycles;
- reject a second root (a node with `parent_id is None`).

Load-time integrity guards are looser: dangling `parent_id` or cycles
on disk are counted as `skipped_lines` rather than raised.

### 4.15.2 Rollback state machine (AC-5)

`Scheduler.run()` tracks an `active_leaf_id` local variable alongside
`cumulative_delta` and `current_knobs`. On each launched trial:

- **OK trial**: advance `active_leaf` to the new child, merge the
  incremental delta into `cumulative_delta`, and reset the per-parent
  sibling-failure counter.
- **Rollback failure** (`failure_mode` in `ROLLBACK_FAILURE_MODES =
  {OOM, WORKER_CRASH, TIMEOUT, METRICS_PARTIAL, METRICS_MISSING}`):
  rewind `active_leaf` to the failing node's parent, restore
  `cumulative_delta` and `current_knobs` from that parent's state
  (so the failing knob is not silently carried forward), and
  increment the sibling counter. When the counter reaches
  `BudgetConfig.max_siblings` (default 3, CLI flag
  `--max-siblings`) the active leaf climbs one more level up the
  DAG; a climb above the root terminates the campaign with the new
  `rollback_exhausted` stop reason.

Preflight rejections (`CONFIG_INVALID`, `DIVISIBILITY_VIOLATION`)
never reach this branch — they are handled by the pre-existing
`_propose_with_preflight` retry loop, never create a `DAGNode`, and
never mutate `active_leaf`.

### 4.15.3 Dedup index and duplicate-OF short-circuit (AC-6)

After every preflight pass, `_propose_with_preflight` consults the
`ConfigDedupIndex` keyed by the resolved SHA:

- **Duplicate-of-FAILED**: the proposal is rejected via
  `preflight_feedback` sharing the `preflight_retries` budget. The
  critic sees a message naming the prior origin node and its failure
  mode, and must propose a different delta.
- **Duplicate-of-OK**: `run()` short-circuits the runner and appends
  a synthetic `DAGNode` with `status = "OK"`,
  `failure_mode = "DUPLICATE_OF"`, `duplicate_of_node_id` pointing at
  the ORIGINAL non-duplicate node (never a chain), and the original
  trial's objective copied for continuity. **No** Ledger entry is
  written for this case — the flat Ledger stays free of synthetic
  rows so `plot_step_time_vs_trajectories` and best-config selection
  behave identically to pre-DAG campaigns. Best-config selection,
  plateau eligibility, and the top-K OK leaderboard in the DAG prompt
  view all exclude `DUPLICATE_OF` entries.

### 4.15.4 DAG-aware Codex prompt block (AC-7)

`CriticPrompt` gains a `dag_block` string field, positioned between
`bitter_lessons_block` and `history_block` in both the full
`__str__` and the human-readable `to_debug_text()` renderings. The
scheduler pre-renders this block once per proposal round via
`render_dag_view(node_store, active_leaf_id, max_dag_nodes)` and
passes it through `Critic.propose()` → `build_prompt()`. Layout:

1. **Active branch** (ancestor chain of `active_leaf`, always
   unconditional).
2. **Sibling attempts** at the active leaf's parent.
3. **Top-K OK leaves** by objective ascending (excluding
   `DUPLICATE_OF`).
4. **Recent failure leaves** by recency.

Sections 2 + 3 + 4 are jointly capped by
`Scheduler.max_dag_nodes` (default 30). Section 1 is never
truncated. A `max_dag_nodes = 0` still renders a well-formed block
with the ancestor chain plus empty leaderboard sections.

The wiki file `wiki/09-dag-search.md` documents this block to Codex
and is loaded via `_WIKI_CONTEXT_FILES`.

### 4.15.5 Coexistence contract

The Scheduler dataclass has both `node_store: NodeStore | None` and
`dedup_index: ConfigDedupIndex | None` as optional fields with a
default of `None`. When both are `None` (backward-compat mode), the
scheduler behaves exactly as it did before the DAG substrate landed:
no coexistence writes, no bootstrap, no rollback rewind (state
advances unconditionally as before), no dedup lookups, no DAG prompt
block. This preserves every pre-DAG test and every pre-DAG campaign
directory. The production CLI (`__main__.py::_run_campaign`) wires
both fields on every run; internal tests pass ``None`` to isolate
behaviour.

---

## 5. Test architecture

```
tests/test_schema.py                       19 tests  (AC-1)
tests/test_placement_enum.py               25 tests  (AC-4)
tests/test_override_wrapper.py             16 tests  (AC-2)
tests/test_preflight.py                    14 tests  (AC-3)
tests/test_runner.py                       15 tests  (AC-5)
tests/test_parser.py                       31 tests  (AC-6)
tests/test_timeline_processor.py           38 tests  (AC-6/new)
tests/test_ledger.py                       13 tests  (AC-9)
tests/test_lessons.py                      13 tests  (new — BitterLesson store)
tests/test_critic.py                       51 tests  (AC-7)
tests/test_scheduler.py                    19 tests  (AC-8)
tests/test_cli.py                          17 tests  (AC-10)
tests/test_smoke.py                         3 tests  (AC-11)
tests/test_no_auto_placement_import.py      7 tests  (AC-11)
────────────────────────────────────────────────────────────
total                                     281 tests
```

All tests are **hermetic** — no real RLinf launch, no Ray, no GPU,
no Codex/Claude network call. The classes of test:

- **Unit tests** (`test_{schema,placement_enum,override_wrapper,
  parser,ledger,critic,lessons,timeline_processor}.py`) — pure-Python
  assertions on individual modules.
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
  Also asserts the bitter-lesson fold happens BEFORE the next trial.
- **Timeline processor tests** (`test_timeline_processor.py`) —
  synthetic event lists exercise every view (stall_fractions, tag
  stats, critical path, outliers, per-component bubble, raw excerpts,
  component call averages, BLOCKING_TAGS exclusion).
- **LessonBook tests** (`test_lessons.py`) — canonical signature
  recursion, dedup by `(mode, signature)`, LRU cap with eviction
  markers, fsync-per-append, corruption tolerance.
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
| New `FailureMode`      | Add to the `FailureMode` enum in `parser.py`, extend `_classify_failure` if it has a regex trigger, and consider whether it should be added to `critic._LESSON_REQUIRED_FAILURE_MODES`. |
| Alternative critic     | Implement the `Critic` protocol (`critic.py::Critic`). For Bayesian/EA fallback (FUT-3), the scheduler is critic-agnostic — just swap the `critic` argument. |
| New critic backend     | Drop a `scripts/ask-<name>.sh` next to the module (stdin=prompt, stdout=response), then extend the `--critic-backend` choices in `__main__.build_parser`. |
| Mirror more `validate_cfg` rules | Add new checks in `preflight._check_divisibility`. They must be computable without `Cluster()`. |
| New timeline analysis view | Add a `compute_*` function to `timeline_processor.py`, wire it through `process_timeline`, add a matching `TimelineSummary` field in `parser.py`, and render a labelled block in `critic._render_timeline_verbose`. |
| Widen `BLOCKING_TAGS`  | `timeline_processor.BLOCKING_TAGS`. Human review required; a wrong entry silently deflates a component's busy total.  |
| Update critic prompt guidance | Edit the relevant file under `wiki/` — no Python change needed. Order matters; keep numeric prefixes stable.  |
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
  point the wrapper points argv at. The trial's resolved
  `<log_dir>/tensorboard/config.yaml` is read back by
  `__main__._extract_trial_context` to recover the effective
  `enable_offload` state.
- `profiler/rlinf_timeline/autopatch.py` (vendored under
  `toolkits/embodied_tuner/profiler/`) — emits the JSONL events
  `timeline_processor` consumes. The runner exports the env flags
  that turn the autopatch on.
- `profiler/plot_timeline.py` (vendored) — invoked best-effort by
  `timeline_feed.render_timeline_plot` to produce Gantt PNGs alongside
  every trial.

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
- `humanize/commands/ask-codex.md` — the humanize command whose
  interface the vendored `scripts/ask-codex.sh` matches. The shared
  `scripts/ask-*.sh` scripts are copies vendored next to the toolkit
  so a shared-humanize install is not required.
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
  global steps. The `--max-epochs=3` default exists because of this;
  parser averages `step_time` across all N blocks (single-step trials
  are still measurable).
- `profiler/enable2.sh` leaves `RLINF_NVITOP=1` / `RLINF_NVML=1`
  commented out by default. The tuner's runner exposes these behind
  `--collect-memory` (opt-in) so routine campaigns don't accumulate
  hundreds of MB of memory telemetry; timeline env vars remain on by
  default because the critic consumes them every round.
- Hydra search paths in embodied baselines use
  `${oc.env:EMBODIED_PATH}/config/`. Preflight sets `EMBODIED_PATH` /
  `REPO_PATH` before `compose()` so the interpolation resolves.
- Every trial writes its resolved Hydra config to
  `<log_dir>/tensorboard/config.yaml`. `__main__._extract_trial_context`
  reads that file to recover the effective `enable_offload` per
  component; without it, `timeline_processor.compute_outliers` would
  ship without the `knob_hint` column.

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
5. Subsequent iterations added the persistent BitterLesson store,
   the wiki-driven critic prompt, the seven-view timeline processor,
   the JSONL feed selector + Gantt rendering, per-attempt critic
   transaction persistence, the `--trial-timeout-seconds` +
   `--critic-backend` CLI knobs, and moved memory telemetry behind
   `--collect-memory`.

For an operator-facing usage guide, see [`README.md`](./README.md).
