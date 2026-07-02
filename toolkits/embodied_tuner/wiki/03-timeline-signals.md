# 03 Timeline signals — from raw events to conclusions

The critic receives a `timeline_summary` derived from `timeline/*.jsonl`
by `parser.TimelineSummary`. This file explains what each signal means,
so citations in `rationale.timeline_citations` translate to concrete
remediations.

## 03.1 Files and lanes

Every worker process writes one JSONL file:

    <log_dir>/timeline/<component>_rank<RANK>_pid<PID>.jsonl

Each event is a `{ts_start, ts_end, tag, component, rank, ...extra}`
row. `parser.py` folds these into per-rank statistics keyed by tag.

Component lanes in the current embodied stack:

- `runner` — `EmbodiedRunner.run`. Outermost lane. Wrapper only, do not
  cite as a bottleneck (excluded by default via
  `RLINF_TIMELINE_WORKER_TIMER_EXCLUDE_TAGS`).
- `env` — env-worker interact loops.
- `rollout` — the HuggingFace multi-step rollout worker's predict calls.
- `actor` — FSDP actor training (forward, policy loss, backward,
  optimizer step) when `RLINF_TIMELINE_ACTOR_TRAINING=1`.

## 03.2 Tag reference (embodied)

| Tag                          | Component | Interpretation                                                                              |
|------------------------------|-----------|---------------------------------------------------------------------------------------------|
| `run`                        | runner    | Outer wrapper. Ignore. Only useful as an absolute timeline scale.                          |
| `interact`, `run_interact_once` | env / rollout | Wrapper timers that also include wait time — noisy, excluded by default.                |
| `env_interact_step`          | env       | One env `step(action)` call. Sum over R ≈ per-step env cost.                               |
| `prefetch_train_bootstrap`   | env       | Startup cost. Ignore beyond step 1.                                                        |
| `predict`                    | rollout   | One rollout forward pass. Sum over R ≈ per-step rollout cost.                              |
| `generate`, `generate_one_epoch` | rollout | Wrapper timers, exclude by default.                                                       |
| `recv_rollout_results`, `recv_rollout_trajectories` | actor | Actor waiting for rollout/env data. Under **hybrid** (actor occupies all GPUs) ignore this value — the actor is simply idle while rollout and env interact on their own GPUs, so it is not "starved". Under **disaggregated**, a high value does indicate a starved actor (rollout slow or transfer stalled). |
| `run_training`               | actor     | Full actor training step. Wrapper — excluded by default; per-phase tags below are truthful.|
| `actor_forward`              | actor     | Model forward pass in training.                                                            |
| `actor_policy_loss`          | actor     | Policy loss computation.                                                                   |
| `actor_backward`             | actor     | Backward pass.                                                                             |
| `actor_optimizer_step`       | actor     | Optimizer step (FSDP all-gather + step).                                                   |
| `compute_advantages_and_returns` | actor | Advantage computation; usually small — a large value here suggests a config regression.    |

The plotter and worker-timer sidecar both exclude the wrapper tags by
default. When the parser emits a `timeline_citations` recommendation, it
means the tag was not wrapper-suppressed.

## 03.3 Derived statistics per tag

For each `(component, rank, tag)` triple the parser reports:

- `median` — median event duration.
- `p90` — 90th-percentile duration; high `p90 / median` ratios signal
  straggler ranks (usually caused by all-gather / all-reduce imbalance).
- `stall_fraction` — the fraction of the lane's wall-clock window
  during which this tag was **not** active. `stall_fraction ≈ 0` on the
  busiest tag identifies the bottleneck; `stall_fraction ≈ 1` on any
  worker points at a starved consumer (see hybrid / disaggregated
  interpretations below).
- `count` — event count; sanity-check against `R` and the number of
  training steps.

## 03.4 Bottleneck decision tree

1. **Ignore wrapper tags** (`interact`, `run_interact_once`, `generate`,
   `generate_one_epoch`, `run_training`, `recv_rollout_*`, `runner:run`)
   for the bottleneck decision — their durations sum wait time in.
2. Under **collocated**: pick the largest of `env_interact_step * R`,
   `predict * R`, and `actor_backward + actor_forward + actor_optimizer_step`.
   That's the dominant term.
3. Under **hybrid**: env and rollout run on disjoint GPU subsets during
   the interact loop.
   - If `env_interact_step_median * R > predict_median * R`, env is the
     interact-side bottleneck. Rollout ranks will show a matching
     `predict.stall_fraction > 0`.
   - Compare `R * max(env_interact_step_median, predict_median)` with
     the sum of the actor training tags. Whichever is larger is the
     placement's critical path term.
4. Under **disaggregated**: all three lanes run in parallel. Whichever
   lane has `stall_fraction ≈ 0` on its main tag is the bottleneck; the
   other two show `stall_fraction > 0` on their consumer-side tags
   (`recv_rollout_*` on actor, or the interact waits).

## 03.5 When timeline data is missing

- `--no-profiler` disables all `RLINF_TIMELINE*` env vars and the trial
  is classified `(OK, METRICS_PARTIAL)`. It cannot win best-config.
- `RLINF_TIMELINE_ACTOR_TRAINING=0` (default for third-party
  patch files) suppresses `actor_forward` / `actor_backward` — only
  wrapper-level `run_training` will exist. `enable_timeline.sh` in this
  toolkit turns actor training on for exactly this reason.
- If the timeline directory exists but the tag map is empty, the trial
  probably crashed before the first event was flushed. Reclassify by
  checking `stderr` / `run_embodiment.log` — this is usually a
  worker-crash or OOM masquerading as METRICS_PARTIAL.

## 03.6 Do not

- Do not cite wrapper tags as bottlenecks — they include wait time and
  will mislead the critic.
- Do not average across all ranks when investigating stragglers; use
  the `p90 / median` ratio per rank.
- Do not treat `run_interact_once` durations as evidence of env-worker
  cost. That timer wraps the whole interact chunk including the wait
  for rollout's predict.
