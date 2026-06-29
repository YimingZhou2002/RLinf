# Round 1 Summary

**Status:** in progress — Milestone B of 4 shipped. Loop NOT yet complete (AC-7 through AC-11 remain). Round 1 scope was Milestone B (trial runner + log/timeline parser); 6 of 11 ACs are now mainline-complete pending Codex verification.

**Round commit:** appended to `RLinf` `main` (see `git log -1`).
**Cumulative progress:** Round 0 (Milestone A) + Round 1 (Milestone B) = AC-1 through AC-6 implemented and tested. AC-7 through AC-11 remain queued.

## What Was Implemented

Two production modules added to `RLinf/toolkits/embodied_tuner/`, with hermetic unit tests. No RLinf training was launched. No Ray was started.

### Module summary

- **`runner.py` (AC-5)** — `TrialRunner` dataclass with `launch(spec, timeout)` API. Launches the `LaunchSpec` from `OverrideWrapper` in its own POSIX process group (`start_new_session=True`), enforces a per-trial timeout, escalates SIGTERM → SIGKILL after `sigterm_grace_seconds`, invokes a configurable `ray_stop_hook`, and runs a scoped `pgrep -f "RLINF_TUNER_TRIAL_ID=<id>"` orphan sweep with kill-then-recheck. Profiler env exports (`RLINF_TIMELINE*` + `RLINF_NVITOP`/`RLINF_NVML`) are merged into the launch env by default; `disable_profiler=True` and `disable_memory_telemetry=True` provide the CLI opt-outs the plan calls for. `TrialOutcome` is a frozen dataclass carrying `returncode`, `timed_out`, `wall_clock_seconds`, `cleanup_outcome` (`ok|sigkill_required|ray_stop_failed|orphans_killed|orphans_remain`), `stdout_path`, and the effective `LaunchSpec`.

- **`parser.py` (AC-6)** — `parse_trial(log_dir, returncode, timed_out, stderr_path, failure_mode_override)` returns a `TrialResult`. Parses `metrics.log` MetricTable blocks (`╭...╰` Unicode-box-drawn frames; extracts `Global Step: X/Y`, `Step Time: NNN.NNNs`, `num_trajectories=N`, and Time-section `key=value` pairs). Parses `timeline/*.jsonl` events into a `TimelineSummary` with per-rank `TagStats` for a verified headline-tag set (`env_interact_step`, `env/bootstrap_step`, `actor/recv_traj`, `actor/sync_model_to_rollout`, `actor/compute_adv`, `rollout/generate`, `predict`) plus per-component stall fractions. `(Status, FailureMode)` classifier with the full enum from AC-6; `(FAILED, NONE)` is an enforced invariant. OOM rubric: nonzero `returncode` AND stderr matches `CUDA out of memory | torch.cuda.OutOfMemoryError | Ray actor ... died|killed ... OOM`. Worker-crash rubric: nonzero `returncode` AND stderr matches `RayActorError | ActorDiedError | killed by signal | Traceback`. Objective `step_time / num_trajectories` averaged across steps 2..N; best-config selection (`select_best`) requires `(OK, NONE)`. Best-effort `nvitop_summary.log` peak-GPU-memory read.

## Files Changed

### Created

| Path | Target AC |
|------|-----------|
| `toolkits/embodied_tuner/runner.py` | AC-5 |
| `toolkits/embodied_tuner/parser.py` | AC-6 |
| `toolkits/embodied_tuner/tests/test_runner.py` | AC-5 (15 tests) |
| `toolkits/embodied_tuner/tests/test_parser.py` | AC-6 (24 tests) |
| `.humanize/rlcr/2026-06-29_15-37-02/round-1-contract.md` | — |
| `.humanize/rlcr/2026-06-29_15-37-02/round-1-summary.md` | this file |

### Modified

- `.humanize/rlcr/2026-06-29_15-37-02/goal-tracker.md` (Plan Version bumped to Round 1; task5/task6 marked completed-pending-verification; queued side issues updated with Round-1 discoveries).

No RLinf core files were touched.

## Validation

```
$ PYTHONPATH=. python -m pytest toolkits/embodied_tuner/tests/
======================= 115 passed, 12 warnings in 3.52s =======================
```

Cumulative per-AC test coverage (Round 0 + Round 1):

- AC-1 (schema): 21 tests
- AC-2 (override wrapper): 16 tests
- AC-3 (preflight): 14 tests
- AC-4 (placement enumerator): 25 tests
- **AC-5 (trial runner): 15 tests** — success path, log_dir capture, timeout → SIGTERM, SIGTERM-resistant child → SIGKILL escalation, profiler env defaults / `--no-profiler` / `--no-collect-memory` / `extra_env` overrides, ray_stop_hook invocation + failure-sets-outcome, scoped pgrep orphan kill + remain detection, pgrep error swallowing, log_dir creation failure raises.
- **AC-6 (parser): 24 tests** — synthetic 3-block parse; live baseline parse; objective averaged over steps 2..N; objective requires ≥2 steps; missing final num_trajectories handled; non-positive num_trajectories rejected; empty/missing timeline handled; live timeline parsed end-to-end; OOM stderr classification; WORKER_CRASH classification; TIMEOUT short-circuit; failure_mode_override path; nonzero returncode without stderr → WORKER_CRASH; missing timeline → METRICS_PARTIAL; select_best picks lowest objective among (OK, NONE) only; select_best returns None when no eligible / empty input; (FAILED, NONE) invariant violation raises.

The 12 warnings are pre-existing Hydra "Defaults list is missing `_self_`" notices on `maniskill_ppo_openvla.yaml`.

## Remaining Items

### Queued mainline tasks (for Round 2+)

- AC-7/task7: LLM critic prompt builder + structured `{summary, metric_table_citations, timeline_citations}` rationale schema + `CriticOutputValidator` (dual-source rule for placement deltas) + `fake_critic`. Will consume `parser.TimelineSummary` directly.
- AC-8/task8: scheduler with budget defaults `max_trials=20, budget_seconds=43200, max_oom=5, patience=3, epsilon=0.02`.
- AC-9/task9: append-only JSONL ledger with SHA-256 hashing + structured `critic_rationale` persistence (uses the SHA computed by `preflight._sha256_of_resolved`).
- AC-10/task10: CLI `python -m toolkits.embodied_tuner` + shim launcher.
- AC-11/task11: end-to-end smoke test with `fake_critic` + mock runner + bundled AST-walker import-boundary sub-test.

### Side issues this round

- **Timeline tag names.** The plan's example tag names (`env/env_interact_step`, `rollout/generate_one_epoch`, `actor/run_training`) don't match what `profiler/rlinf_timeline/autopatch.py` actually emits in `*.jsonl` (the real tags are `env_interact_step`, `rollout/generate`, `actor/sync_model_to_rollout`, etc.). Resolved by updating `_HEADLINE_TAGS` in `parser.py` to the verified event-tag set against the live `logs/20260629-07:25:33-*` directory. The MetricTable still surfaces the aggregate per-component times (`env/interact`, `actor/run_training`) via `MetricStep.time_keys`, so the AC-7 critic prompt can still cite both signals.
- **`nvitop/` directory currently absent.** The live `logs/20260629-07:25:33-*` directory has no `nvitop/` because `profiler/enable2.sh` leaves the NVITOP flag commented. The Round-1 runner now exports `RLINF_NVITOP=1`/`RLINF_NVML=1` by default; trials launched via the auto-tuner will have a `nvitop/` directory, and `parser._read_peak_gpu_mem` reads `nvitop_summary.log` best-effort. The test `test_parse_trial_three_steps_with_timeline_is_ok_none` uses synthetic fixtures that omit `nvitop/`; the parser correctly populates `peak_gpu_mem_gib=None` without complaint.

### Open questions for Codex review

- Should the runner's `cleanup_outcome` precedence rules be reordered? Current order: `sigkill_required` (from SIGTERM-resistant child) > `ray_stop_failed` (from hook returning False) > `orphans_killed` / `orphans_remain` (from pgrep sweep). Specifically, when SIGKILL was required AND ray_stop fails AND orphans remain, the final outcome is `orphans_remain`. That seems right (the worst-case wins) but I want Codex to sanity-check.
- The parser's OOM rubric currently matches OOM patterns on a stderr file. RLinf's actual OOM output may also appear in the trial's `metrics.log` or `run_embodiment.log` (the runner's `stdout_path`). Should the rubric also scan `stdout_path` by default? Currently the caller (scheduler in task8) is responsible for passing the right path.
- `peak_gpu_mem_gib` is read from `nvitop_summary.log`'s `max_process_gpu_mem` field via a regex. The actual nvitop schema may differ; this is a best-effort read documented as such. Codex review: is there a canonical nvitop file we should prefer?

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: BitLesson file still contains only the template. Round 1 surfaced two integration nuances (timeline event tags differ from MetricTable aggregate keys; `nvitop/` is opt-in via the NVITOP env flag) but both are documented in `parser.py` and the goal-tracker's Queued Side Issues. They are not recurring problems yet; if a future round encounters either pattern again, an entry can be added.
