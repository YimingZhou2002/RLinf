# Round 2 Contract

## Mainline Objective

Establish Milestone C — loop control + critic. Three production modules under `RLinf/toolkits/embodied_tuner/`, completing AC-7 (critic prompt + dual-source rationale validator + fake_critic), AC-8 (scheduler with budget + plateau + critic-stagnation), AC-9 (append-only JSONL ledger with SHA-256 hashing + structured critic_rationale persistence).

After this round only Milestone D (CLI + smoke test + bundled AST walker) remains.

## Target ACs

- AC-9 (append-only JSONL ledger; comes first so critic and scheduler can both depend on it)
- AC-7 (LLM critic prompt builder + `CriticOutputValidator` enforcing the dual-source rule for placement-touching deltas + `fake_critic` for hermetic tests)
- AC-8 (scheduler with `max_trials=20, budget_seconds=43200, max_oom=5, patience=3, epsilon=0.02` defaults; stopping rule covers budget exhaustion / plateau / critic-stagnation)

AC-10 and AC-11 remain `[queued]` for Round 3.

## Mainline Tasks (in dependency order)

1. **task9 — `ledger.py`** (no deps inside the toolkit; uses `OmegaConf` for SHA computation, but that's already exercised by preflight). Public surface:
   - `Ledger(path)` — append-only JSONL writer. `.append(entry: LedgerEntry)`, `.load() -> list[LedgerEntry]`, `.best() -> LedgerEntry | None` (delegates to parser's `select_best` semantics).
   - `LedgerEntry` frozen dataclass mirroring the AC-9 field list:
     `trial_idx, delta, resolved_config_sha, log_dir, returncode, status, failure_mode,
      objective, step_time, num_trajectories, per_component_timings, timeline_summary,
      peak_gpu_mem, critic_rationale, ts_start, ts_end, cleanup_outcome`.
   - `critic_rationale` is a `dict` carrying `{summary, metric_table_citations, timeline_citations}` exactly as AC-7 produces.
   - Append is atomic per line (open in `"a"`, `flush`, `os.fsync` so a mid-loop crash leaves prior trials intact).
   - `load()` tolerates one corrupted line without losing subsequent valid lines; a counter of skipped lines is exposed on the result for telemetry.

2. **task7 — `critic.py` + `fake_critic.py`** (deps: task9 ledger schema, parser `TimelineSummary` / `MetricStep`, schema `KnobSchema`). Public surface:
   - `CriticPrompt` — pure data class holding the rendered prompt sections (bottleneck rubric, history, current knobs, constraints, memory-pressure flag, timeline summary block). `__str__` returns the assembled prompt.
   - `build_prompt(history, current_knobs, schema, last_failure_mode, last_metric_summary, last_timeline_summary)` returns a `CriticPrompt`.
   - `Rationale` dataclass `{summary, metric_table_citations, timeline_citations}`.
   - `CriticOutput` dataclass `{delta, rationale, stop_requested: bool}`.
   - `parse_critic_output(text) -> CriticOutput` parses structured JSON from a Codex response (tolerates Markdown code fences and surrounding prose).
   - `CriticOutputValidator(schema)` with `.validate(output) -> ValidationResult`: enforces the dual-source rule (placement-touching deltas MUST cite ≥1 non-empty `metric_table_citations` AND ≥1 non-empty `timeline_citations`) and re-runs `schema.validate(delta)`.
   - `Critic` Protocol with `propose(history, current_knobs, last_outcome) -> CriticOutput`. The default real implementation `CodexCritic` shells out to `ask-codex.sh`; up to 3 validator-driven retries with feedback strings.
   - `FakeCritic` (in `fake_critic.py`) deterministic critic that returns canned `CriticOutput`s from a list, supports `"no_further_improvement"` via `stop_requested=True`. Sufficient for AC-8 scheduler tests and AC-11 smoke test.

3. **task8 — `scheduler.py`** (deps: task9 ledger, task7 critic, task6 parser, task5 runner, task3 preflight). Public surface:
   - `BudgetConfig(max_trials, budget_seconds, max_oom, patience, epsilon)` dataclass with the AC-8 defaults.
   - `Scheduler(critic, runner, parser_fn, preflight_fn, ledger, budget)` orchestrates the loop. `parser_fn` and `preflight_fn` are injectable so tests can stub them; production wiring uses `parse_trial` and `compose_and_validate`.
   - `.run() -> CampaignResult` runs until termination. Returns `CampaignResult(stop_reason, trial_count, best_entry)`.
   - Stop reasons: `max_trials_reached | budget_seconds_elapsed | oom_cap_exceeded | plateau | critic_stagnation | critic_failure | no_trials_run`.
   - Preflight failure does NOT count toward the trial budget (per the plan); critic gets feedback and proposes a new delta, up to a configurable retry cap (default 3).
   - Plateau check: among the last `patience` non-failed trials, if every consecutive relative improvement is below `epsilon`, terminate.

## Blocking Side Issues in Scope

None known at start.

## Queued Side Issues Out of Scope

- AC-10 CLI + shim launcher.
- AC-11 end-to-end smoke test + bundled AST-walker import-boundary sub-test.
- The real `CodexCritic.propose` will not be GPU-tested in this round; it relies on `ask-codex.sh` which needs network/Codex CLI. Tests for it use a `subprocess.run` stub so the suite stays hermetic.

## Round Success Criteria

- `ledger.py` writes/reads append-only JSONL; mid-loop crash recovery test passes; corrupted-line tolerance test passes; `.best()` delegates to the parser's `(OK, NONE)`-only rule.
- `critic.py` and `fake_critic.py` implement the API above. The `CriticOutputValidator` rejects placement-touching deltas with empty `metric_table_citations` AND/OR empty `timeline_citations`; accepts non-placement deltas with just a non-empty `summary`. `parse_critic_output` parses JSON inside Markdown code fences. The real `CodexCritic` is exercised via a fake `subprocess.run` and verifies the 3-retry-with-feedback loop.
- `scheduler.py` honours the budget defaults and every stopping rule. Tests cover: budget of 3 trials terminates after 3; plateau termination at trial 5 when last 3 improvements are <2%; `"no_further_improvement"` twice consecutively terminates; `max_oom=5` exceeded terminates with `oom_cap_exceeded`; preflight failure burns a critic retry but NOT a trial budget slot.
- Goal Tracker updated to mark task7/task8/task9 `completed (pending verification)`.
- `round-2-summary.md` written with `## BitLesson Delta`.
- All changes committed locally with a Conventional-Commit signed-off message.

## Out-of-Scope (Explicit)

- No real Codex call. `CodexCritic` is implemented but tests use injected fakes.
- No real RLinf trial. Scheduler tests use injected fake runner / fake preflight / fake parser; the integration with the Round-1 modules will be exercised by the AC-11 smoke test in Round 3.
- No CLI; no shim launcher. Both arrive in Round 3.
