# 00 Bottleneck rubric

How to read the runtime data blocks the tuner injects into your prompt
(`## Last trial — MetricTable ...`, `## Last trial — critical path per
global_step`, `## Last trial — per-GPU bubble ...`, `## Last trial —
outlier events ...`, `## Last trial — raw timeline excerpts ...`) and
turn them into a delta whose expected effect on `step_time /
num_trajectories` is non-zero.

This document is read first and must be self-contained. Detailed
definitions and edge cases appear later in `01 placement-critical-paths`,
`02 optimization-directions`, `03 timeline-signals`, and
`04 constraints`.

## 00.1 Identify the runner mode and current placement

The critical-path formula changes between `run` (default) and
`run_pipeline`, and between collocated / hybrid / disaggregated. The
prompt's current knobs section shows `cluster.component_placement`
directly.

Runner mode is fixed by the baseline (`runner.use_training_pipeline`)
and is NOT a tunable knob in this loop.

Placement mode semantics (each mode uses **all** GPUs — the difference is
how the three components share them):

- **collocated**: env, rollout, and actor all occupy every GPU and
  time-share the hardware. No two components run at the same time; the
  three phases execute serially per rollout step.
- **hybrid**: env and rollout are split onto disjoint GPU shards (their
  shards together cover all GPUs) and run in parallel (pipeline) during
  interact; the actor is collocated with one side (typically rollout)
  and blocks while interact runs, then reclaims those GPUs for training.
- **disaggregated**: env, rollout, and actor each own a disjoint GPU
  shard (the three shards together cover all GPUs). With `run_pipeline`
  all three run concurrently; with plain `run` the actor shard sits idle
  during interact.

Resulting critical-path formulas:

- `run` + collocated:   `T_sync + R*(T_env + T_rol) + T_act`
- `run` + hybrid:       `T_sync + R*max(T_env, T_rol) + T_act`
- `run` + disaggregated:`T_sync + R*max(T_env, T_rol) + T_act`   (actor GPUs idle during interact — usually worse than hybrid)
- `run_pipeline` + disaggregated: `T_sync + max(R*T_env, R*T_rol, T_act)`

## 00.2 Locate the bottleneck term

Read in this order:

1. **Per-GPU bubble** (`## Last trial — per-GPU bubble ...`). Bubble = wall
   − union(real-busy intervals). A large bubble on GPUs assigned to one
   side (env-side or rollout-side) is a strong indicator that the OTHER
   side is the current chunk-level bottleneck. The `env_side_avg_bubble_s`
   and `rollout_side_avg_bubble_s` summary lines are the fastest read.
2. **Critical path per global_step** (`## Last trial — critical path per
   global_step`). Each lane row separates `real_s` (actual GPU work) from
   `blocked_s` (waiting on another component). **The bottleneck is the
   lane with the largest `real_s`.** A lane with big `blocked_s` and
   small `real_s` is a downstream consumer, not a bottleneck; do NOT
   propose to shrink it.

    Common trap: `actor/recv_traj` / `recv_rollout_trajectories` is a
    **blocking wait** — its duration reflects the interact-loop cost
    (env + rollout), not actor work. Cite `actor_forward`,
    `actor_backward`, `actor_optimizer_step`, or
    `compute_advantages_and_returns` for actor bottleneck claims. Use
    `actor/run_training` only as MetricTable sanity-check evidence when
    actor phase tags are missing; never cite recv tags as actor compute.

3. **MetricTable time keys** (`## Last trial — MetricTable Time-section
   keys`). Sanity-check the `real_s` decision: `env/interact`,
   `rollout/generate`, and `actor/run_training` should agree with the
   timeline's per-step attribution within ~10%. Divergence means one of
   the tags is wrapper-noisy — trust the timeline `real_s` over the
   MetricTable in that case.
4. **Outlier events** (`## Last trial — outlier events ...`). Rows carry
   `knob_hint` when the parser can map the stall to a knob. Use these as
   corroboration for the delta you already chose in step 3 — not as a
   substitute for reading the critical path.
5. **Raw timeline excerpts** (`## Last trial — raw timeline excerpts
   ...`). The verbatim JSONL of the longest events. Cite one of these
   whenever the delta touches `cluster.component_placement` (dual-source
   rule).

## 00.3 Confirm the delta will move the critical-path term

Before proposing a knob, check that its target term is on the current
critical path.

- Under `run` + hybrid: shrinking the smaller of `T_env` / `T_rol`
  buys **zero** — the term is `max(T_env, T_rol)`. Skip to a different
  knob.
- Under `run` + disaggregated: same as above. Also, the actor GPUs are
  idle during interact — proposing to grow actor's GPU share cannot
  reduce `T_act`'s scheduling slot, so its only effect is `T_act`
  compute speedup.
- Under `run_pipeline` + disaggregated: shrinking a non-max term buys
  zero.
- Under collocated: every term contributes additively, so any shrink
  helps. Rank by magnitude and pick the largest.

## 00.4 Choose the knob

Consult [`02-optimization-directions.md`](02-optimization-directions.md) for
knob-by-knob effects. Summary of first-line moves:

| Bottleneck term (largest `real_s` on critical path) | First-line knob                    | Second-line knob                            |
|-----------------------------------------------------|------------------------------------|---------------------------------------------|
| `env_interact_step` on env ranks                    | move `env.train.total_num_envs` only if normalized `step_time / num_trajectories` should improve | reallocate env GPUs (placement)             |
| `predict` on rollout ranks                          | reallocate rollout GPUs (placement)| `env.train.rollout_epoch` down only if normalized objective improves |
| `actor_forward`, `actor_backward`, `actor_optimizer_step` on actor ranks | `actor.micro_batch_size` up if memory allows; down only for OOM / high memory | reallocate actor GPUs / `rollout_epoch` up if normalized objective improves |
| `actor/sync_model_to_rollout` (T_sync)              | (not tunable in this loop) — investigate `weight_sync_interval` in the baseline | flag as FUT                                 |
| `compute_advantages_and_returns`                    | `env.train.rollout_epoch` down only if normalized objective improves | flag if config regression |

Rules of thumb:

- Prefer knobs that touch a single component. Placement changes are
  the highest-variance moves — reserve them for when a component-local
  knob has plateaued.
- Optimize `step_time / num_trajectories`, not raw `step_time`. A delta
  that reduces `step_time` but reduces `num_trajectories` by the same or
  larger fraction is not an improvement.
- Under memory pressure (last trial `FailureMode=OOM`), the memory
  triage cascade is: `actor.micro_batch_size` down → `env.train.total_num_envs`
  down → `enable_offload=true` on the OOM component. Do NOT reach for
  placement in the OOM branch.
- Only one knob per delta unless the two are logically inseparable
  (e.g. shrinking `total_num_envs` while flipping `env.train.enable_offload`
  because both are the response to the same OOM). Multi-knob deltas
  make the ledger harder to interpret and slow convergence.

## 00.5 Justify the delta with dual sources when placement moves

When `delta` contains `cluster.component_placement`:

- `rationale.metric_table_citations` must include at least one line of
  the form `key=value` copied from the MetricTable Time-section block
  (e.g. `env/interact=275.4`, `actor/run_training=210.1`).
- `rationale.timeline_citations` must include at least one observation
  copied from `## Last trial — per-GPU bubble ...`, the critical path
  block, or the raw excerpts block, expressed as free-form text
  (e.g. `env rank0 env_interact_step median=15.2s stall_fraction=0.40`).

For non-placement deltas, only `rationale.summary` is required — but
the summary should still name the specific term you are moving and
where you read it.

## 00.6 What not to do

- Do not cite `actor/recv_traj` as evidence of actor being slow. It is a
  wait on the interact loop; its duration is the env-rollout cost.
- Do not cite wrapper tags (`interact`, `run_interact_once`, `generate`,
  `generate_one_epoch`, `run_training`, `step`, `generate_rollouts`)
  as bottleneck evidence. They span waits. The parser's outlier and
  critical-path blocks already exclude the noisiest wrappers.
- Do not propose more than one placement move per delta. Placement
  moves interact with divisibility constraints (`env.train.total_num_envs`
  vs `env_world_size`) and often need a follow-up shrink; keep those
  in separate rounds so the ledger attributes cause and effect.
- Do not propose `enable_offload=true` on the current critical-path
  component — offloading is a memory rescue knob, not a throughput
  knob. It widens the wall time of whichever component it applies to.
- Do not propose a knob that has already been tried at the same value
  in the recent trial history without new evidence to justify it. The
  history block is deduplicated for exactly this check.
