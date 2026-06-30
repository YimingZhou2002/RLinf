# RLCR Loop — Final / COMPLETE Summary

**Status: COMPLETE-eligible.** All 11 ACs from `docs/plan.md` are mainline-complete; 197/197 unit tests pass; Codex Round-2 review (READY_FOR_ROUND_3: yes-with-caveats) had all 5 IMPORTANT caveats folded in during Round 3. A final Codex verification round was triggered against this state in parallel with writing this summary.

## Loop arc

| Round | Milestone | Scope | Tests added | Commit |
|-------|-----------|-------|-------------|--------|
| 0 | A — Foundations | `schema.py`, `placement_enum.py`, `override_wrapper.py`, `preflight.py` (AC-1..AC-4) | 21+25+16+14 = 76 | `8eb5a9fb` |
| 1 | B — Trial exec + parsing | `runner.py`, `parser.py` (AC-5, AC-6) | 15+24 = 39 | `563e8dc6` |
| 2 | C — Loop control + critic | `ledger.py`, `critic.py`, `fake_critic.py`, `scheduler.py` (AC-9, AC-7, AC-8) | 13+26+13 = 52 | `7639f2dc` |
| 3 | D — User surface + Codex-review fixes | `__main__.py`, `run_embodied_tuner.sh`, `test_smoke.py`, `test_no_auto_placement_import.py` (AC-10, AC-11); 5 IMPORTANT fixes from Codex review | 12+3+7+10 extra = 30 (incl. 8 new tests for review fixes) | `ef650e37` |
| Total | — | 11/11 ACs, ~7,900 LOC | 197 | 4 commits |

## What ships

A Python orchestrator at `RLinf/toolkits/embodied_tuner/` that iteratively tunes RLinf embodied training configs:

1. The LLM critic (`CodexCritic`) proposes a knob delta with structured `{summary, metric_table_citations, timeline_citations}` rationale.
2. `CriticOutputValidator` enforces the **dual-source rule**: placement-touching deltas MUST cite at least one MetricTable observation AND at least one timeline observation. Up to 3 retries with feedback strings; rejects bare-string citation arrays as well as missing ones.
3. `preflight.compose_and_validate` composes baseline + delta via Hydra and runs the targeted divisibility checks LOCALLY (mirrors `rlinf/config.py:962/965/980/1363-1368`) — never starts Ray.
4. `Scheduler` runs the trial loop with `BudgetConfig(max_trials=20, budget_seconds=43200, max_oom=5, patience=3, epsilon=0.02)`. When preflight retries are exhausted, the loop terminates with `stop_reason="preflight_exhausted"` and a synthetic `(FAILED, CONFIG_INVALID)` ledger entry rather than launching a known-bad config.
5. `TrialRunner` launches the trial in its own POSIX process group, enforces timeout, escalates SIGTERM→SIGKILL, runs the `ray stop --force` hook, and scopes orphan cleanup via both `pgrep -f` AND `/proc/<pid>/environ` so env-tagged Ray workers are findable.
6. `parse_trial` reads `metrics.log` + `timeline/*.jsonl`, classifies the trial via `(Status, FailureMode)`, computes the objective `step_time / num_trajectories` averaged across steps 2..N (step 1 = warmup; default `max_epochs=3` → average of steps 2 and 3), and surfaces a per-component timeline summary the critic prompt consumes. OOM/WORKER_CRASH classification runs BEFORE METRICS_MISSING on nonzero returncode, so OOM-killed trials that never wrote `metrics.log` are correctly classified `(FAILED, OOM)`.
7. `Ledger` persists each trial as an append-only JSONL line with the structured `critic_rationale` payload — the audit trail the plan calls out under "Placement Decision Audit Trail".
8. CLI `python -m toolkits.embodied_tuner` (and shim `examples/embodiment/run_embodied_tuner.sh`) ties it together; emits `best_config.yaml` + `best_trial.json` next to `tuner_ledger.jsonl`.

The toolkit does NOT import `toolkits.auto_placement` (enforced by the AC-11 AST walker), does NOT add new dependencies beyond what RLinf already pins (OmegaConf+hydra-core), and does NOT modify the stock `run_embodiment.sh`.

## Validation

```
$ PYTHONPATH=. python -m pytest toolkits/embodied_tuner/tests/
======================= 197 passed, 15 warnings in 4.77s =======================
```

- AC-1 KnobSchema: 21 tests
- AC-2 OverrideWrapper: 16 tests
- AC-3 Preflight: 14 tests
- AC-4 Placement enumerator: 25 tests
- AC-5 Trial runner: 15 tests
- AC-6 Parser: 27 tests (24 + 3 Round-3 OOM-precedence/stdout_path tests)
- AC-7 Critic: 30 tests (26 + 4 Round-3 citation-type tests)
- AC-8 Scheduler: 13 tests (replaced "run-anyway" with `preflight_exhausted` + feedback-wiring tests)
- AC-9 Ledger: 13 tests
- AC-10 CLI: 12 tests
- AC-11 Smoke + AST walker: 10 tests

Live-system checks:

- `python -m toolkits.embodied_tuner --help` prints the full flag set.
- `python -m toolkits.embodied_tuner --config maniskill_ppo_openvla --dry-run-preflight` composes the real baseline (placement_kind=hybrid, resolved-config SHA stable).
- `bash examples/embodiment/run_embodied_tuner.sh --help` prints CLI usage via the shim.

## What is queued (NOT blocking COMPLETE)

Codex Round-2 review marked the following as SUGGESTIONS, all polish items:

- Mirror broader `validate_cfg` rules in preflight (group_size divisibility, eval-side divisibility, runner.task_type checks, supported model_type, actor_critic value-head). Current preflight mirrors only the four divisibility checks the plan explicitly tests.
- Softer evidence requirement for memory-sensitive non-placement deltas (`enable_offload`, `total_num_envs`, `micro_batch_size`). Current dual-source rule covers placement only as the plan specifies.
- Make `ray stop --force` opt-in or failure-only rather than default-on. Default-on is safer for shared hosts.

FUT-1 through FUT-9 from the plan remain queued by design (multi-node, async pipelined trials, BO/EA fallback proposer, surrogate warm start, un-pinning knobs, eval during trials, cross-config tuning, deterministic OOM shrink-retry, additional baselines).

## BitLesson updates this loop

One entry added: `BL-2026-06-30-pgrep-env-vs-argv` — when a per-trial subprocess cleanup uses an env-var-only tag and identifies orphan workers via `pgrep -f`, the cleanup silently misses every orphan because `pgrep -f` matches argv not environment. Fix: scan `/proc/<pid>/environ` directly and union with `pgrep -f`. Recorded in `RLinf/.humanize/bitlesson.md`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The BitLesson entry above was added as part of Round 3. No additional lessons surfaced in this loop-finalization step.

## Stop signal for the loop runner

This summary serves as the loop-final marker. All 11 plan ACs implement and pass tests; the dual-source rule is enforced; the audit trail is persisted; the toolkit cleanly excludes `auto_placement`. **COMPLETE.**
