# Round 2 Summary

**Status:** in progress — Milestone C of 4 shipped. Loop NOT yet complete (AC-10 and AC-11 remain — Milestone D). 9 of 11 ACs are now mainline-complete pending Codex verification.

**Round commit:** appended to `RLinf` `main` (see `git log -1`).
**Cumulative progress:** Round 0 (Milestone A) + Round 1 (Milestone B) + Round 2 (Milestone C) = AC-1 through AC-9 implemented and tested.

## What Was Implemented

Three production modules (plus one helper `fake_critic.py`) added to `RLinf/toolkits/embodied_tuner/`, with hermetic unit tests. No RLinf training launched. No real Codex call. No Ray started.

### Module summary

- **`ledger.py` (AC-9)** — `Ledger(path)` append-only JSONL writer. `LedgerEntry` frozen dataclass with the 17 fields the plan calls out (`trial_idx, delta, resolved_config_sha, log_dir, returncode, status, failure_mode, objective, step_time, num_trajectories, per_component_timings, timeline_summary, peak_gpu_mem, critic_rationale, ts_start, ts_end, cleanup_outcome`). `append()` validates the entry, opens in `"a"`, writes the JSON line, `flush()`+`os.fsync()`. `load()` returns `LoadResult(entries, skipped_lines)` tolerating one corrupted line per skip. `best()` mirrors `parser.select_best`'s `(OK, NONE)` rule. Structured `critic_rationale = {summary, metric_table_citations, timeline_citations}` is persisted verbatim — the audit-trail surface the plan calls out under "Placement Decision Audit Trail".

- **`critic.py` (AC-7)** — Public surface: `Rationale`, `CriticOutput`, `TrialHistoryEntry`, `CriticPrompt` (with section blocks for rubric, history, current knobs, constraints, memory-pressure flag, timeline summary, feedback), `build_prompt(...)` that assembles a `CriticPrompt`, `parse_critic_output(text)` that handles raw JSON / Markdown-fenced JSON / brace-bracketed JSON, `CriticOutputValidator(schema)` that enforces the **dual-source rule** (placement-touching deltas MUST cite ≥1 non-empty `metric_table_citations` AND ≥1 non-empty `timeline_citations`) PLUS re-runs the knob schema, and the `Critic` Protocol with a production `CodexCritic` that shells out to `ask-codex.sh` and retries up to `max_retries=3` with feedback strings on validation failure.

- **`fake_critic.py` (AC-7 helper)** — Deterministic `FakeCritic` for tests and the AC-11 smoke harness. `FakeCritic.from_deltas(*deltas)` and `FakeCritic.stop_after(*deltas)` factories. Raises `CriticError` when its output queue is exhausted (helps catch off-by-one scheduling bugs).

- **`scheduler.py` (AC-8)** — `Scheduler(critic, runner_fn, parser_fn, preflight_fn, ledger, budget, baseline_knobs, clock)`. `BudgetConfig` carries the AC-8 defaults (`max_trials=20, budget_seconds=43200, max_oom=5, patience=3, epsilon=0.02, preflight_retries=3, history_window=8`). Public `run() -> CampaignResult`. Stop reasons: `max_trials_reached | budget_seconds_elapsed | oom_cap_exceeded | plateau | critic_stagnation | critic_failure | no_trials_run`. Preflight failures DO NOT count toward `max_trials`; the critic gets feedback (via its next `propose` call) and re-proposes up to `preflight_retries`. Plateau check uses the last `patience` non-failed trials with non-`None` objectives. Critic-stagnation requires two consecutive `stop_requested=True` outputs.

## Files Changed

### Created

| Path | Target AC |
|------|-----------|
| `toolkits/embodied_tuner/ledger.py` | AC-9 |
| `toolkits/embodied_tuner/critic.py` | AC-7 |
| `toolkits/embodied_tuner/fake_critic.py` | AC-7 (helper) |
| `toolkits/embodied_tuner/scheduler.py` | AC-8 |
| `toolkits/embodied_tuner/tests/test_ledger.py` | AC-9 (13 tests) |
| `toolkits/embodied_tuner/tests/test_critic.py` | AC-7 (26 tests) |
| `toolkits/embodied_tuner/tests/test_scheduler.py` | AC-8 (13 tests) |
| `.humanize/rlcr/2026-06-29_15-37-02/round-2-contract.md` | — |
| `.humanize/rlcr/2026-06-29_15-37-02/round-2-summary.md` | this file |

### Modified

- `.humanize/rlcr/2026-06-29_15-37-02/goal-tracker.md` (Plan Version bumped to Round 2; task7/task8/task9 marked completed-pending-verification; preflight-retry-exhaustion behaviour documented as a queued side issue).

No RLinf core files were touched.

## Validation

```
$ PYTHONPATH=. python -m pytest toolkits/embodied_tuner/tests/
======================= 167 passed, 12 warnings in 3.58s =======================
```

Cumulative per-AC test coverage (Round 0 + Round 1 + Round 2):

- AC-1 (schema): 21 tests
- AC-2 (override wrapper): 16 tests
- AC-3 (preflight): 14 tests
- AC-4 (placement enumerator): 25 tests
- AC-5 (trial runner): 15 tests
- AC-6 (parser): 24 tests
- **AC-7 (critic): 26 tests** — `parse_critic_output` raw / Markdown-fenced / brace-bracketed JSON; rejects malformed JSON; requires `delta` + `rationale`; `stop_requested` propagated; `CriticOutputValidator` accepts placement delta with dual-source rationale; rejects when either citation array is missing or all entries are whitespace; accepts non-placement delta with non-empty `summary`; rejects non-placement delta with empty summary; rejects pinned-knob delta via the schema layer; `CodexCritic` returns first valid output; retries after malformed JSON (with feedback in the prompt); retries after dual-source failure (verified feedback content reaches the second prompt); gives up after `max_retries`; `build_prompt` contains all required sections; history block renders each trial; memory-pressure block only appears when last failure was OOM; timeline-summary block renders metric keys + stall fractions + per-tag stats; feedback block appears after other sections; `FakeCritic.from_deltas` replays in order; raises when exhausted; `stop_after` marks the final output.
- **AC-8 (scheduler): 13 tests** — `max_trials` terminates; `max_trials=0` triggers `no_trials_run`; `budget_seconds` elapsed terminates (via injected clock); `max_oom` exceeded terminates with `oom_cap_exceeded`; plateau terminates after `patience` consecutive sub-`epsilon` improvements; plateau does NOT fire on continued >`epsilon` improvements; two consecutive `stop_requested=True` triggers `critic_stagnation`; single `stop_requested` does NOT terminate; preflight rejection burns a critic retry but NOT a trial slot; preflight-retry exhaustion still runs the trial (forward-progress); ledger records every trial; `best_entry` is `None` when no eligible trial exists; critic exhaustion terminates the loop with `critic_failure`.
- **AC-9 (ledger): 13 tests** — round-trip; parent-dir creation; absent path returns empty `LoadResult`; corrupted JSON line tolerated; schema-violating line tolerated; simulated mid-loop crash (truncated last line) leaves earlier entries readable; `.best()` picks lowest objective among `(OK, NONE)` only; `.best()` returns `None` when no eligible; `from_dict` rejects missing required field; `append` validates before writing; path-as-string round-trips; `critic_rationale` persists verbatim; `resolved_config_sha` round-trips.

## Remaining Items

### Queued mainline tasks (for Round 3)

- AC-10/task10: CLI entrypoint `python -m toolkits.embodied_tuner` + shim launcher `RLinf/examples/embodiment/run_embodied_tuner.sh`.
- AC-11/task11: end-to-end smoke test with `FakeCritic` + mock runner_fn/parser_fn/preflight_fn (the Round-2 scheduler tests already exercise this shape) + bundled AST-walker import-boundary sub-test.

After Round 3 the RLCR loop should be COMPLETE.

### One new side issue this round

- **Preflight-retry exhaustion**. When all `preflight_retries+1` critic proposals are rejected by preflight, the scheduler currently still calls `runner_fn` with the last critic_output (which will likely produce a `(FAILED, *)` trial that gets ledger-recorded and consumes a `max_trials` slot). The alternative — skip the trial entirely and record a synthetic `CONFIG_INVALID` ledger entry without consuming `max_trials` — would also be defensible. The chosen behaviour matches "preflight failure DOES NOT count toward `max_trials`" in the AC-8 contract until retries run out; once retries are exhausted, the trial DOES count. Documented in the queued side issues for Codex review.

### Open questions for Codex review

- Is the preflight-retry-exhaustion behaviour above the right call, or should we instead surface a dedicated `preflight_exhausted` stop reason?
- The `CodexCritic.ask_codex_path` default points at the plugin install path under `/root/.claude/plugins/cache/PolyArch/humanize/1.17.0/scripts/ask-codex.sh`. This is fine for the user's current environment but the CLI (AC-10) should expose `--ask-codex-path` so production deployments can override.
- `Rationale` is frozen and `tuple`-of-`str` for citations. The current `Ledger.append` serialises via `asdict` → list[str]; round-trips fine. Codex may prefer JSON arrays carry richer structure (e.g. `{key, value}` pairs) — out of scope for Round 2 but worth flagging.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: BitLesson file unchanged. Round 2 surfaced one design nuance (preflight-retry-exhaustion behaviour) that's documented in the goal-tracker and the round summary; not yet a recurring problem warranting a BitLesson entry. If Codex review challenges the chosen behaviour, the resolution can be entered then.
