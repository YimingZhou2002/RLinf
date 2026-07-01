# Optimization directions per knob

Each knob's effect on the three critical-path terms and the memory
budget. Use this together with `placement-critical-paths.md` to decide
whether a proposed delta will actually reduce `step_time /
num_trajectories`.

The overall objective `step_time / num_trajectories` can be decomposed
into reducing the per-trajectory time of each component on the critical
path (`T_env / traj`, `T_rol / traj`, `T_act / traj`), which in turn
decomposes into the sub-tasks each knob targets.

## `cluster.component_placement`

- **What it moves:** reshapes the critical path itself.
- **When to touch:**
  - Under collocated when one of `T_env`, `T_rol`, `T_act` dominates and
    the other two barely register — the parallelism won by disaggregating
    is worth the memory pressure of dedicated GPU sets.
  - Under hybrid when the timeline shows large `stall_fraction` on the
    faster of env / rollout — reallocate GPUs from the idle side to the
    busy side.
- **Common failure mode:** proposing a disaggregated split that leaves
  actor with too few GPUs; `T_act` grows past every other term and now
  the actor is bound, not helping.
- **Dual-source rationale required** (see the README): any placement
  delta must cite one MetricTable line AND one timeline observation.

## `env.train.total_num_envs`

- **What it moves:** batches env stepping. Larger → fewer env kernel
  launches per trajectory element, so `T_env` per collected sample
  usually decreases; but GPU memory used by env increases linearly.
- **When to grow:** env is the critical-path bottleneck AND env GPU
  memory has headroom (nvml_total_used_gib well under the device cap).
- **When to shrink:** OOM on the env process, or env memory has grown
  past ~85% of the device cap.
- **Watch for:** `T_env` also depends on `group_size` and env-side
  divisibility; preflight rejects splits that violate the env-worker
  contract, so a shrink may fail preflight rather than the trial.
- **Divisibility interaction:** `total_num_envs` must divide by
  `env_world_size` (`rlinf/config.py:962` family). Ledger will show a
  `FAILED, CONFIG_INVALID` if this is violated.
- **Scaling behaviour:** with placement held constant, `T_env` grows
  roughly linearly with `total_num_envs` (i.e.
  `T_env = a * env_num + b`), but beyond a certain env count the growth
  becomes super-linear — doubling `env_num` may more than double `T_env`
  (e.g. quadruple it) due to kernel scheduling contention and
  memory-bandwidth saturation.

## `env.train.rollout_epoch` (denoted `R` above)

- **What it moves:** trades trajectory buffer size (favouring the actor's
  gradient step efficiency) for env/rollout wall time. `step_time`
  contains an `R * ...` term under every placement.
- **When to shrink:** critical path is dominated by the env/rollout
  chunk term (`R * max(T_env, T_rol)` under hybrid, or `R * T_env` /
  `R * T_rol` under disaggregated). Halving R halves that term but
  doubles the number of actor calls per unit of trajectory.
- **When to grow:** actor is the bottleneck under hybrid or
  disaggregated (`T_act > R * max(T_env, T_rol)`) — larger R amortises
  actor overhead over more collected samples.
- **Interaction with `num_trajectories`:** the objective normalises by
  `num_trajectories`, and `num_trajectories` scales linearly with R.
  A smaller R buys smaller `step_time` but also smaller
  `num_trajectories`; watch that `step_time / num_trajectories` actually
  improves.
- **Scaling behaviour:** with placement held constant, `T_rol` grows
  roughly linearly with total trajectory count (i.e.
  `T_rol = a * traj_num + b`), but beyond a certain scale the growth
  becomes super-linear — doubling trajectories may more than double
  `T_rol` (e.g. quadruple it) due to KV-cache pressure and
  memory-bandwidth saturation.

## `actor.micro_batch_size`

- **What it moves:** memory footprint of one actor forward+backward pass,
  and the number of micro-batches per `global_batch_size`.
- **When to grow:** actor OOM headroom exists AND actor is on the
  critical path — fewer micro-batches = fewer kernel launches, higher
  arithmetic intensity, lower `T_act`.
- **When to shrink:** actor OOM occurred, or `nvml_total_used_gib` on
  actor ranks is above ~85% of the device cap.
- **Divisibility:** `global_batch_size % micro_batch_size == 0` is
  required (`rlinf/config.py:965`, `1363-1368`). Preflight will reject a
  non-divisor; propose a divisor instead of a nearby number.
- **Note on pinned `global_batch_size`:** the schema pins
  `actor.global_batch_size` (FUT-5), so the critic can only move
  `micro_batch_size` within its divisors.
- **Scaling behaviour:** with placement held constant, `T_act` grows
  roughly linearly with total trajectory count (i.e.
  `T_act = a * traj_num + b`), but beyond a certain scale the growth
  becomes super-linear — doubling trajectories may more than double
  `T_act` (e.g. quadruple it) due to memory-bandwidth contention and
  cache pressure.
- **Non-monotonic effect of mbs:** when GPU memory allows, increasing
  `micro_batch_size` first *decreases* `T_act` (fewer micro-batches,
  higher arithmetic intensity). However, once memory usage approaches
  the device cap or crosses a critical threshold, further increasing
  `micro_batch_size` *increases* `T_act` instead — fragmentation,
  spilling, and scheduling overhead dominate. The optimal mbs is
  therefore not the largest divisor but the one at the inflection point
  before the super-linear slowdown.

## `env.train.enable_offload`

- **What it moves:** moves env-side state between GPU and CPU/host
  between phases. Cuts env GPU memory at the cost of extra transfer
  latency per phase.
- **When to enable:** env-side OOM under the current placement, or env
  is not on the critical path and its memory is squeezing another
  worker's headroom.
- **When to disable:** env is the critical-path bottleneck; the offload
  overhead widens `T_env` and worsens the objective.

## `rollout.enable_offload`

- **What it moves:** symmetric to env offload but for rollout weights and
  KV cache. Rollout offload is expensive in wall time (moves a large
  model checkpoint), so it is a memory rescue knob, not a throughput
  knob.
- **When to enable:** rollout OOM only. If rollout is on the critical
  path, offload will almost certainly regress `step_time`.
- **When to disable:** rollout is bottleneck and enough memory exists to
  keep the model resident.

## `actor.enable_offload`

- **What it moves:** offloads actor optimiser state / weights between
  training steps. Big memory saving, big wall-time cost.
- **When to enable:** actor OOM only.
- **When to disable:** actor is on the critical path or nvml curves show
  large downward-then-upward memory swings around each `run_training`
  call, which is the offload signature.

## Pinned in this loop (FUT-5)

These are declared in the schema but rejected by the validator with
`KnobNotTunableError`. Do not propose them:

- `actor.global_batch_size`
- `rollout.pipeline_stage_num`
- `actor.model.num_action_chunks`

## Cross-knob patterns

- **Memory-triage cascade** on repeated OOM: `micro_batch_size` down →
  `total_num_envs` down → `enable_offload` on the OOM component. Prefer
  cheap-to-revert knobs first.
- **Rebalance under hybrid**: when the timeline shows env-rollout
  stall imbalance ≥ 0.3, move GPUs from the idle side to the busy side
  via `component_placement`. Do not grow `total_num_envs` at the same
  time — one move at a time makes the ledger interpretable.
- **Actor-bound recovery**: if `T_act` dominates and actor memory is
  tight, growing `micro_batch_size` may OOM. Consider disabling
  `actor.enable_offload` (if on) before growing the batch size — offload
  is often the reason `T_act` is bloated in the first place.
