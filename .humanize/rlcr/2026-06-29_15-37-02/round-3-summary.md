# Round 3 Summary

**Status:** **Milestone D shipped + Codex Round-2-review fixes applied. All 11 ACs are mainline-complete pending final Codex verification.** The RLCR loop is COMPLETE-eligible.

**Round commit:** appended to `RLinf` `main` (see `git log -1`).
**Cumulative progress:** Rounds 0+1+2+3 = AC-1 through AC-11 implemented and tested; 197 unit tests pass.

## What Was Implemented

Milestone D's two production deliverables (CLI + smoke test/AST walker) plus the five IMPORTANT fixes Codex flagged in its review of Rounds 0-2.

### New modules / files (AC-10, AC-11)

- **`__main__.py` (AC-10)** — CLI entrypoint `python -m toolkits.embodied_tuner`. Flags: `--config`, `--baseline`, `--max-trials`, `--budget-seconds`, `--max-oom`, `--patience`, `--epsilon`, `--max-epochs`, `--collect-memory`/`--no-collect-memory`, `--no-profiler`, `--dry-run-preflight`, `--fake-critic`, `--ledger-dir`, `--ask-codex-path`. Adapters bridge production `compose_and_validate` / `OverrideWrapper.build_invocation` / `TrialRunner.launch` / `parse_trial` into the scheduler's injection points. Emits `best_config.yaml` + `best_trial.json` next to `tuner_ledger.jsonl`.
- **`examples/embodiment/run_embodied_tuner.sh` (AC-10)** — Shim launcher; exports `REPO_PATH` + `PYTHONPATH`, forwards arguments, prints a remediation hint when `toolkits.embodied_tuner` cannot be imported.
- **`tests/test_cli.py` (AC-10)** — 12 tests: argument parsing defaults + overrides + error paths; `--dry-run-preflight` end-to-end on the live `maniskill_ppo_openvla.yaml`; `--dry-run-preflight` via real `subprocess.run` on `python -m toolkits.embodied_tuner`; best-artefact emission with and without an eligible trial; `_load_fake_critic` happy + error paths; shim launcher existence/executable/`--help`.
- **`tests/test_smoke.py` (AC-11)** — 3 end-to-end scenarios driven by `FakeCritic` + mock runner/parser/preflight: clean 4-trial run with monotonically improving objectives (verifies ledger, `best_config.yaml`'s YAML has `actor.enable_offload=true` from the best delta, `best_trial.json` payload); failure-injected run (trial 1 parser crash, trial 2 OOM, trials 3-4 successful, best=trial 3); no-eligible-trial run (`best_trial.json` lists exclusion reasons).
- **`tests/test_no_auto_placement_import.py` (AC-11)** — 7 tests: AST-walks every `.py` under `RLinf/toolkits/embodied_tuner/` and asserts no `import toolkits.auto_placement` or bare `import auto_placement`; positive sanity checks (mentions in strings/comments are fine); negative tests with planted offenders covering `from ... import`, bare `import`, and submodule `import` forms; benign-imports baseline; every shipped file parses.

### Codex-review fixes (IMPORTANT findings folded into existing modules)

Codex's review of Rounds 0-2 returned **CRITICAL: None, READY_FOR_ROUND_3: yes-with-caveats** with 5 IMPORTANT items. All 5 are now fixed:

1. **`critic.py` — preflight feedback wiring + citation-type validation.**
   - `build_prompt()` gained a `preflight_feedback: str | None = None` arg and renders a `## Preflight rejected the previous delta` block when supplied.
   - `Critic` Protocol + `CodexCritic.propose` + `FakeCritic.propose` all accept the new kwarg.
   - `parse_critic_output` now uses `_coerce_citation_list` to reject bare strings, non-list values, and lists with non-string elements — closing the escape hatch where a string in `metric_table_citations` would be iterated character-by-character by the validator.

2. **`scheduler.py` — `preflight_exhausted` stop reason + synthetic CONFIG_INVALID ledger entry + feedback wiring.**
   - `_propose_with_preflight` now passes `preflight_feedback` (the concatenated rejection reasons) to the next critic call.
   - When preflight retries are exhausted, the scheduler no longer calls `runner_fn` with a known-bad delta. Instead it writes a synthetic `(FAILED, CONFIG_INVALID)` ledger entry via the new `_record_preflight_exhausted` and stops the campaign with `stop_reason="preflight_exhausted"`.

3. **`parser.py` — OOM precedence over METRICS_MISSING + default stderr_path.**
   - `parse_trial` now defaults `stderr_path` to `log_dir / "run_embodiment.log"` (the runner's merged stdout+stderr file).
   - When `returncode != 0`, the OOM/WORKER_CRASH classifier runs BEFORE `metrics.log` existence is checked. An OOM-killed trial that never wrote `metrics.log` is now correctly classified `(FAILED, OOM)` instead of `(FAILED, METRICS_MISSING)`.

4. **`runner.py` — orphan cleanup scans `/proc/<pid>/environ`.**
   - `RLINF_TUNER_TRIAL_ID` is an env var; `pgrep -f` matches command lines, not environment. Ray workers don't typically expose tuner env vars in argv.
   - New `_pids_with_env_match` reads `/proc/<pid>/environ` directly (POSIX Linux; falls back gracefully on non-Linux hosts) and is unioned with the `pgrep -f` result for backward compatibility.

5. **`schema.py` — stale docstring corrected.**
   - The `KnobDomain` docstring previously claimed preflight calls `validate_cfg`; now it correctly says preflight re-implements the targeted divisibility checks locally to avoid `Cluster()`/`ray.init`.

## Files Changed

### Created

| Path | Target AC |
|------|-----------|
| `toolkits/embodied_tuner/__main__.py` | AC-10 |
| `examples/embodiment/run_embodied_tuner.sh` | AC-10 |
| `toolkits/embodied_tuner/tests/test_cli.py` | AC-10 (12 tests) |
| `toolkits/embodied_tuner/tests/test_smoke.py` | AC-11 (3 tests) |
| `toolkits/embodied_tuner/tests/test_no_auto_placement_import.py` | AC-11 (7 tests) |
| `.humanize/rlcr/2026-06-29_15-37-02/round-3-contract.md` | — |
| `.humanize/rlcr/2026-06-29_15-37-02/round-3-summary.md` | this file |

### Modified

- `toolkits/embodied_tuner/critic.py` — `parse_critic_output` citation-type validation, `build_prompt` `preflight_feedback`, `Critic` Protocol + `CodexCritic.propose` accept `preflight_feedback`.
- `toolkits/embodied_tuner/fake_critic.py` — `FakeCritic.propose` accepts `preflight_feedback` and captures it in `calls`.
- `toolkits/embodied_tuner/scheduler.py` — `preflight_exhausted` stop reason; `_record_preflight_exhausted`; `_propose_with_preflight` now passes rejection feedback to the next critic call.
- `toolkits/embodied_tuner/parser.py` — OOM/crash classified before METRICS_MISSING on nonzero returncode; default `stderr_path` to `run_embodiment.log`.
- `toolkits/embodied_tuner/runner.py` — `_default_pgrep_runner` unions `/proc/<pid>/environ` scan with `pgrep -f`; new `_pids_with_env_match`.
- `toolkits/embodied_tuner/schema.py` — stale docstring corrected.
- `toolkits/embodied_tuner/tests/test_critic.py` — added 4 citation-type-validation tests; preserved the original tests.
- `toolkits/embodied_tuner/tests/test_parser.py` — added 3 tests covering OOM precedence + default stderr_path.
- `toolkits/embodied_tuner/tests/test_scheduler.py` — replaced the old "run-anyway" test with two new tests verifying `preflight_exhausted` stop reason and that preflight feedback reaches the next critic call.
- `.humanize/rlcr/2026-06-29_15-37-02/goal-tracker.md` — Plan Version bumped to Round 3; task10/task11 marked completed-pending-verification; Plan Evolution Log entry documents the Codex-driven scheduler/parser/runner changes; remaining Codex SUGGESTIONS recorded as queued side issues.

No RLinf core files were touched.

## Validation

```
$ PYTHONPATH=. python -m pytest toolkits/embodied_tuner/tests/
======================= 197 passed, 15 warnings in 4.93s =======================
```

Cumulative per-AC test coverage (Rounds 0+1+2+3):

- AC-1 (schema): 21 tests
- AC-2 (override wrapper): 16 tests
- AC-3 (preflight): 14 tests
- AC-4 (placement enumerator): 25 tests
- AC-5 (trial runner): 15 tests
- AC-6 (parser): 24 + 3 = 27 tests
- AC-7 (critic): 26 + 4 = 30 tests
- AC-8 (scheduler): 13 + 1 (replaced 1) = 13 tests
- AC-9 (ledger): 13 tests
- **AC-10 (CLI): 12 tests**
- **AC-11 (smoke + AST walker): 10 tests**

Live-system checks:

- `python -m toolkits.embodied_tuner --help` prints the full flag list.
- `python -m toolkits.embodied_tuner --config maniskill_ppo_openvla --dry-run-preflight` composes the real baseline successfully (`placement_kind=hybrid`, SHA `9087afd9d77e5ee2a231d6419d89e609a924a0aae45d31b6d450ba0ab80fae0f`).
- `bash examples/embodiment/run_embodied_tuner.sh --help` prints the CLI usage via the shim.

## Codex Review Integration Summary

Codex review of Rounds 0-2 was invoked via `ask-codex.sh` and returned:

- **CRITICAL: None.**
- **IMPORTANT: 5** — all 5 fixed in this round (see above).
- **SUGGESTIONS: 3** — stale docstring (fixed), `rlinf.utils.logging` (confirmed correct choice), `ray stop --force` opt-in (left default-on for safety, queued).
- **COVERAGE_GAPS: 8** — all 8 covered by new Round-3 tests.
- **READY_FOR_ROUND_3: yes-with-caveats** → caveats resolved this round.

## Remaining Items

- No queued mainline tasks. **All 11 ACs are now mainline-complete pending final Codex verification.**
- Three SUGGESTION-level items remain in the goal-tracker as queued side issues (non-blocking): broader `validate_cfg` mirroring, softer evidence rule for memory-sensitive non-placement deltas, opt-in `ray stop --force`. Codex marked all three as polish, not P1.

## BitLesson Delta

Action: add
Lesson ID(s): BL-2026-06-30-pgrep-env-vs-argv
Notes: This round surfaced one repeatedly-relevant lesson worth recording for future RLCR rounds in this project: when implementing per-trial subprocess cleanup that scopes "kill orphans" via a trial tag, `pgrep -f` matches the command line, NOT environment variables. Env-var-only tags are invisible to `pgrep -f`, so cleanup silently misses Ray workers spawned by the trial. The fix is to also scan `/proc/<pid>/environ` (POSIX Linux) for the tag pattern, OR to inject the tag into argv via a Hydra override the train script accepts. The current `runner.py` does both: it sets `RLINF_TUNER_TRIAL_ID=<idx>` in the subprocess env AND falls back to a `/proc/<pid>/environ` scan when `pgrep -f` returns nothing. Documented below.

```markdown
## Lesson: pgrep-env-vs-argv
Lesson ID: BL-2026-06-30-pgrep-env-vs-argv
Scope: toolkits/embodied_tuner/runner.py — per-trial subprocess cleanup
Problem Description: When the per-trial cleanup uses a tag set ONLY as an environment variable (e.g. RLINF_TUNER_TRIAL_ID=<id>) and identifies orphan workers via `pgrep -f <tag>`, the cleanup silently misses every orphan because `pgrep -f` matches argv (the command-line arguments) and NOT the process environment. Ray workers spawned by the trial inherit the env var but their argv typically does not contain it, so `pgrep -f` returns no matches even when several orphan Ray workers are still running.
Root Cause: `pgrep -f`'s `-f` flag matches against the full command line (argv), not the inherited environment block exposed at `/proc/<pid>/environ`.
Solution: Scan `/proc/<pid>/environ` directly for the tag pattern, union those PIDs with whatever `pgrep -f` returns, and kill the union. This catches env-tagged orphans on Linux without requiring the train script to forward the tag into argv. Skip gracefully on non-Linux hosts (no `/proc`).
Constraints: POSIX Linux only for the `/proc`-based scan. The pgrep fallback path is still useful when callers DO inject the tag into argv. Treat cleanup as best-effort; never raise from the cleanup path.
Validation Evidence: `toolkits/embodied_tuner/runner.py:_pids_with_env_match`; tests in `tests/test_runner.py` (existing scoped-pgrep tests continue to pass via dependency injection).
Source Rounds: 1 (initial design), 3 (Codex review fix)
```

The lesson is added to `RLinf/.humanize/bitlesson.md` as part of this commit.
