# 04 Signals — timeline tags and stall_fraction

The `TimelineSummary` and its sub-blocks (see `03-inputs.md §9`) are
derived from raw JSONL events. This file explains what each tag
means so that citations translate to concrete remediations.

## 1. Files and lanes

Every worker process writes one JSONL file:

    <log_dir>/timeline/<component>_rank<RANK>_pid<PID>.jsonl

Each event is a `{ts_start, ts_end, tag, component, rank, ...extra}`
row. `parser.py` folds these into per-rank statistics keyed by tag.

Component lanes in the current embodied stack:

- `runner` — `EmbodiedRunner.run`. Outermost lane. Wrapper only, do
  not cite as a bottleneck (excluded by default via
  `RLINF_TIMELINE_WORKER_TIMER_EXCLUDE_TAGS`).
- `env` — env-worker interact loops.
- `rollout` — the HuggingFace multi-step rollout worker's predict
  calls.
- `actor` — FSDP actor training (forward, policy loss, backward,
  optimizer step) when `RLINF_TIMELINE_ACTOR_TRAINING=1`.

## 2. Tag reference

| Tag                          | Component | Interpretation                                                                              |
|------------------------------|-----------|---------------------------------------------------------------------------------------------|
| `run`                        | runner    | Outer wrapper. Ignore. Only useful as an absolute timeline scale.                          |
| `interact`, `run_interact_once` | env / rollout | Wrapper timers that also include wait time — noisy, excluded by default.               |
| `env_interact_step`          | env       | One env `step(action)` call. Sum over R ≈ per-step env cost.                               |
| `prefetch_train_bootstrap`   | env       | Startup cost. Ignore beyond step 1.                                                        |
| `predict`                    | rollout   | One rollout forward pass. Sum over R ≈ per-step rollout cost.                              |
| `generate`, `generate_one_epoch` | rollout | Wrapper timers, exclude by default.                                                       |
| `recv_rollout_results`, `recv_rollout_trajectories` | actor | Actor waiting for rollout/env data. See §2.1 for placement-dependent interpretation. |
| `run_training`               | actor     | Full actor training step. Wrapper — excluded by default; per-phase tags below are truthful.|
| `actor_forward`              | actor     | Model forward pass in training.                                                            |
| `actor_policy_loss`          | actor     | Policy loss computation.                                                                   |
| `actor_backward`             | actor     | Backward pass.                                                                             |
| `actor_optimizer_step`       | actor     | Optimizer step (FSDP all-gather + step).                                                   |
| `compute_advantages_and_returns` | actor | Advantage computation; usually small — a large value here suggests a config regression.    |

### 2.1 `recv_rollout_*` semantics (placement-dependent)

`actor/recv_rollout_trajectories` and `actor/recv_rollout_results` are
**blocking waits**, not actor GPU work. Their duration reflects the
interact-loop cost (env + rollout), not the actor. Interpretation
depends on placement:

- **Hybrid** (actor occupies all GPUs): a high value is expected — the
  actor is simply idle while rollout and env interact on the same GPUs
  it will later use. **Ignore.**
- **Disaggregated**: a high value means a starved actor (rollout slow
  or transfer stalled). Corroborate with the busy component's
  `stall_fraction ≈ 0`.

Never cite `recv_rollout_*` as evidence of actor being slow. Cite the
actor's own phase tags (`actor_forward`, `actor_backward`,
`actor_optimizer_step`, `compute_advantages_and_returns`) instead.

## 3. Derived per-tag statistics

For each `(component, rank, tag)` triple the parser reports:

- `median` — median event duration.
- `p90` — 90th-percentile duration. A high `p90 / median` ratio
  signals straggler ranks (usually caused by all-gather / all-reduce
  imbalance).
- `stall_fraction` — fraction of the lane's wall-clock window during
  which this tag was **not** active.
  - `stall_fraction ≈ 0` on the busiest tag identifies the bottleneck.
  - `stall_fraction ≈ 1` on any worker points at a starved consumer
    (see `02-paths.md §3` and `§4` for placement-specific readings).
- `count` — event count; sanity-check against `R` and the number of
  training steps.

The plotter and worker-timer sidecar both exclude wrapper tags by
default; when the parser emits a citation-worthy tag, it has already
survived that filter.

## 4. Which tags are wrappers (ignore)

Never cite these as bottleneck evidence — they span waits:

- `runner:run`
- `interact`, `run_interact_once`
- `generate`, `generate_one_epoch`
- `run_training`
- `recv_rollout_results`, `recv_rollout_trajectories` — see §2.1

The parser's outlier and critical-path blocks already exclude the
noisiest wrappers, so if you see one of these listed, it survived
because a specific analysis needed it (e.g. blocked-vs-real split in
the critical path per step).

## 5. Missing data — where the tag stream is empty or partial

See `03-inputs.md §12` for the full missing-data catalog. Short form:

- `--no-profiler` → no `RLINF_TIMELINE*` env vars → empty timeline
  verbose block, `METRICS_PARTIAL` trial.
- `RLINF_TIMELINE_ACTOR_TRAINING=0` → `actor_forward` /
  `actor_backward` missing; only `run_training` wrapper is emitted.
  `enable_timeline.sh` in this toolkit enables it.
- Directory exists but tag map is empty → early crash before first
  flush; reclassify from stderr.
