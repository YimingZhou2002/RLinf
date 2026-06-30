# Round 3 Contract

## Mainline Objective

Establish Milestone D — user-facing surface + guards. Two production deliverables that complete the RLCR-loop plan:

1. `__main__.py` (CLI entrypoint) + `RLinf/examples/embodiment/run_embodied_tuner.sh` (shim launcher) for AC-10.
2. `tests/test_smoke.py` (end-to-end smoke test using `FakeCritic` + mock runner_fn + mock parser_fn) + `tests/test_no_auto_placement_import.py` (bundled AST-walker import-boundary sub-test) for AC-11.

After this round all 11 ACs ship; the RLCR loop becomes COMPLETE-eligible (subject to Codex's verdict on the cumulative Round 0-2 work, which has been submitted for review in parallel).

## Target ACs

- AC-10 (CLI entrypoint + shim launcher)
- AC-11 (end-to-end smoke test + bundled AST-walker import-boundary sub-test)

## Mainline Tasks (in dependency order)

1. **task10 — `__main__.py` + shim**. Public surface:
   - CLI: `python -m toolkits.embodied_tuner` with flags `--config`, `--baseline`, `--max-trials`, `--budget-seconds`, `--max-oom`, `--patience`, `--epsilon`, `--max-epochs`, `--collect-memory`/`--no-collect-memory`, `--no-profiler`, `--dry-run-preflight`, `--fake-critic`, `--ledger-path`, `--ask-codex-path`.
   - `--dry-run-preflight` composes the baseline + delta and exits without launching the runner or starting Ray.
   - `--fake-critic` is a hidden flag used by smoke tests (loads a fixed `FakeCritic` instead of the real `CodexCritic`).
   - On termination: writes `best_config.yaml` (the resolved YAML of the best trial's delta applied to the baseline) and `best_trial.json` (`{objective, denominator_source, step_range_used, exclusion_reasons, source_trial_idx}`).
   - On error: prints a structured error message and exits non-zero. Missing `--config` and unreadable `--baseline` both fail before launch.
   - Shim launcher `RLinf/examples/embodiment/run_embodied_tuner.sh` exports `REPO_PATH`, `PYTHONPATH=${REPO_PATH}:${PYTHONPATH}`, and forwards arguments. PYTHONPATH-missing remediation hint is printed when `python -m toolkits.embodied_tuner` would fail.

2. **task11 — smoke test + AST walker**.
   - `tests/test_smoke.py` runs the scheduler end-to-end via `FakeCritic` + mock runner_fn + mock parser_fn for N=4 synthetic trials, including one mocked OOM and one mocked parser-crash. Verifies the ledger is well-formed, `best_config.yaml` is parseable YAML, `best_trial.json` carries the documented fields. Also covers the CLI path via `subprocess.run` on `python -m toolkits.embodied_tuner --dry-run-preflight ...`.
   - `tests/test_no_auto_placement_import.py` AST-walks every `.py` file under `RLinf/toolkits/embodied_tuner/` and asserts no `import` / `from ... import` statement references `toolkits.auto_placement` or the bare name `auto_placement`. A deliberate seed file (created and removed inside the test) verifies the walker DOES catch a regression.

## Blocking Side Issues in Scope

- The Codex review of Rounds 0-2 (in flight in the background) may surface P0/P1 findings. Any P0 finding becomes a blocking side issue immediately and must be addressed in this round before COMPLETE.

## Queued Side Issues Out of Scope

- Cross-config tuning (FUT-7), multi-node (FUT-1), Bayesian/EA fallback (FUT-3), surrogate warm start (FUT-4), eval/checkpoint during trials (FUT-6).

## Round Success Criteria

- CLI runs end-to-end on a `FakeCritic`+mock-runner stub and produces `best_config.yaml` + `best_trial.json` at the documented paths.
- `python -m toolkits.embodied_tuner --help` prints the flag set; `--dry-run-preflight` exits 0 after Hydra compose without launching anything.
- Shim launcher script exists and is invocable; PYTHONPATH-missing path prints a remediation hint.
- Smoke test passes within seconds in CI.
- Import-boundary AST walker passes on the clean toolkit AND fails when a deliberate `from toolkits.auto_placement import DataFitter` is planted in a temp file.
- Goal Tracker updated to mark task10/task11 `completed (pending verification)`.
- `round-3-summary.md` written with `## BitLesson Delta` plus the Codex-review integration summary.
- All changes committed locally with a Conventional-Commit signed-off message.

## Out-of-Scope (Explicit)

- No real RLinf training launched; the smoke test uses synthetic fixtures (consistent with every prior round).
- No real Codex call; smoke test uses `FakeCritic`.
- No multi-node placement support (FUT-1).
- The "best_config.yaml" emitted by the CLI is the resolved Hydra YAML of the best trial's delta-applied baseline — not a hand-curated subset.
