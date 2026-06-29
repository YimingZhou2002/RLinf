# Round 1 Contract

## Mainline Objective

Establish Milestone B — trial execution + log/timeline parsing. Two production modules under `RLinf/toolkits/embodied_tuner/` plus their unit tests, completing AC-5 (trial runner) and AC-6 (log/timeline parser, objective computation, best-config selection, timeline summary builder).

## Target ACs

- AC-5 (trial runner with timeout + scoped cleanup + default-on profiler env exports)
- AC-6 (log/timeline parser with `(Status, FailureMode)` taxonomy, OOM rubric, objective computation, best-config selection, per-component timeline summary)

AC-7 through AC-11 remain `[queued]` for Round 2+.

## Mainline Tasks (in dependency order)

1. **task5 — `runner.py`** (deps: task2 OverrideWrapper from Round 0). `TrialRunner` class with:
   - `launch(spec: LaunchSpec, timeout: float) -> TrialOutcome` that runs the subprocess in its own process group (`os.setsid`), pipes stdout/stderr to the trial's log dir, escalates SIGTERM → SIGKILL on timeout, invokes a configurable `ray_stop_hook`, runs `pgrep -f "RLINF_TUNER_TRIAL_ID=<id>"` to detect orphan workers, kills any remaining orphans, records the cleanup outcome.
   - Profiler env exports (`RLINF_TIMELINE=1`, `RLINF_TIMELINE_WORKER_TIMER=1`, `RLINF_TIMELINE_ACTOR_TRAINING=1`, `RLINF_TIMELINE_DIR=auto`, `RLINF_NVITOP=1`, `RLINF_NVML=1`) merged into the spec env unless `disable_profiler=True` (mapped to CLI `--no-profiler`) or `disable_memory_telemetry=True` (mapped to `--no-collect-memory`).
   - `TrialOutcome` carries returncode, log_dir, stdout/stderr paths, wall_clock, timed_out, cleanup_outcome, and the launched spec.
   - Tests use real subprocesses (`python -c "import time; time.sleep(...)"`) for the timeout/cleanup paths, plus injected `ray_stop_hook` and `pgrep_runner` to keep tests hermetic. NO RLinf launch in tests.

2. **task6 — `parser.py`** (deps: task5 LOG_DIR convention). Functions and dataclasses for:
   - `parse_trial(log_dir, returncode=None, timed_out=False, stderr_path=None)` → `TrialResult`.
   - Parse `metrics.log` MetricTable blocks (top-of-block sentinel `Metric Table`, key-value lines like `│num_trajectories=18│`, the `Step Time: X.XXXs` header line). Extract per-block `(global_step, step_time, num_trajectories)`.
   - Parse `timeline/*.jsonl` per-rank/per-tag records (`t0`, `t1`, `tag`, `component`, `rank`). Build `TimelineSummary` with per-rank min/median/max for the four headline tags `env/env_interact_step`, `rollout/generate_one_epoch`, `actor/run_training`, plus the cross-component `sync_weights` if present in actor events. Add call_count and stall_fraction (gap-between-events / total-window) where derivable.
   - `(Status, FailureMode)` classifier with the full enum from AC-6: `Status ∈ {OK, FAILED}`, `FailureMode ∈ {NONE, METRICS_PARTIAL, METRICS_MISSING, CONFIG_INVALID, LAUNCH_FAILURE, OOM, WORKER_CRASH, TIMEOUT}`. OOM rubric: nonzero returncode AND stderr matches `CUDA out of memory|torch\.cuda\.OutOfMemoryError|Ray actor.*died.*OOM` (case-insensitive on the latter).
   - `compute_objective(per_step)` averaging steps 2..N (step 1 is warmup); returns `None` when fewer than 2 successful steps were produced (→ `METRICS_PARTIAL`).
   - `select_best(results)` picks the trial with the lowest objective among `(OK, NONE)` only.

## Blocking Side Issues in Scope

None known at start.

## Queued Side Issues Out of Scope

- AC-7 critic + dual-source validator, AC-8 scheduler, AC-9 ledger, AC-10 CLI + shim, AC-11 smoke test.
- `nvitop/` parsing for `peak_gpu_mem` is best-effort metadata per the plan; the parser populates the field when the directory exists and leaves it `None` otherwise. Full nvitop schema verification is queued (no current `nvitop/` directory exists in `RLinf/logs/20260629-07:25:33-maniskill_ppo_openvla/` since `profiler/enable2.sh` leaves the NVITOP flag commented).

## Round Success Criteria

- `runner.py` implements `TrialRunner.launch` with timeout/cleanup as above. Tests for the success path, timeout-triggered cleanup, SIGKILL escalation, `pgrep` orphan detection, ray-stop hook invocation, profiler env merging, opt-out flags — all hermetic (no RLinf launch).
- `parser.py` correctly parses the live `RLinf/logs/20260629-07:25:33-maniskill_ppo_openvla/metrics.log` baseline. Tests cover: `(OK, METRICS_PARTIAL)` for single-step warmup-only trials; `(FAILED, METRICS_MISSING)` for empty trial dirs; `(FAILED, OOM)` on synthetic CUDA OOM stderr; `(FAILED, TIMEOUT)` when runner sets `timed_out=True`; `(OK, NONE)` synthesized via fixture covering ≥3 MetricTable blocks; objective averaged over steps 2..N; best-config selection excludes `(OK, METRICS_PARTIAL)` and `(FAILED, *)` candidates.
- Goal Tracker updated to mark task5/task6 `completed (pending verification)`.
- `round-1-summary.md` written with `## BitLesson Delta`.
- All changes committed locally with a Conventional-Commit-style signed-off message.

## Out-of-Scope (Explicit)

- No real RLinf training invocation. Runner tests use stub Python subprocesses (`time.sleep`); parser tests use synthetic `metrics.log`/`timeline/*.jsonl` fixtures and the live read-only `logs/20260629-07:25:33-*` directory.
- No Codex calls during Round 1.
- No critic logic; the parser exposes `TimelineSummary` shape but does NOT wrap it into critic-prompt content (that is task7 in Round 2).
- No persistence layer; the parser returns in-memory dataclasses. JSONL ledger ships in task9 (Round 2).
