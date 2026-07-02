# 02 Optimization directions per knob

Each knob's effect on the three critical-path terms and the memory
budget. Use this together with `01-placement-critical-paths.md` to decide
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

- **What it moves:** batches env stepping and changes both `step_time`
  and `num_trajectories`. Larger usually increases raw `T_env` and GPU
  memory, but may reduce `T_env / trajectory` if batching efficiency
  improves.
- **When to grow:** env is the critical-path bottleneck, env GPU memory
  has headroom, and trial history or MetricTable normalisation suggests
  `step_time / num_trajectories` will decrease.
- **When to shrink:** env OOM, env memory has grown past ~85% of the
  device cap, or recent trials show super-linear `T_env` growth such
  that fewer envs should improve `step_time / num_trajectories`.
- **Watch for:** `T_env` also depends on `group_size` and env-side
  divisibility. Some violations fail preflight, while group-size and
  routing violations may crash at runtime; apply the `04 constraints`
  checklist before changing this knob.
- **Divisibility interaction:** `total_num_envs` must divide by
  `env_world_size` (`rlinf/config.py:962` family). Ledger will show a
  `FAILED, CONFIG_INVALID` if this is violated.
- **Evidence gate:** treat linear / super-linear scaling as a hypothesis
  unless trial history contains adjacent env counts, p90/median worsens,
  or memory approaches the cap. Prefer the move whose expected
  `step_time / num_trajectories` is lower.

## `env.train.rollout_epoch` (denoted `R` above)

- **What it moves:** changes the number of interact chunks and collected
  trajectories per training step. `step_time` contains an `R * ...`
  term under every placement, while `num_trajectories` also scales with
  R.
- **When to shrink:** critical path is dominated by the env/rollout
  chunk term (`R * max(T_env, T_rol)` under hybrid, or `R * T_env` /
  `R * T_rol` under disaggregated) AND the expected reduction in
  `step_time` is larger than the reduction in `num_trajectories`.
- **When to grow:** actor is the bottleneck under hybrid or
  disaggregated (`T_act > R * max(T_env, T_rol)`) AND larger R should
  reduce actor cost per trajectory by amortising actor overhead over
  more collected samples.
- **Interaction with `num_trajectories`:** the objective normalises by
  `num_trajectories`, and `num_trajectories` scales linearly with R.
  A smaller R buys smaller `step_time` but also smaller
  `num_trajectories`; watch that `step_time / num_trajectories` actually
  improves.
- **Evidence gate:** treat rollout super-linear scaling as a hypothesis
  unless trial history, p90/median, KV-cache pressure, or memory curves
  support it. Optimise the normalised objective, not raw `step_time`.

## `actor.micro_batch_size`

- **What it moves:** memory footprint of one actor forward+backward pass,
  and the number of micro-batches per `global_batch_size`.
- **When to grow:** actor OOM headroom exists AND actor is on the
  critical path — fewer micro-batches = fewer kernel launches, higher
  arithmetic intensity, lower `T_act`.
- **When to shrink:** actor OOM occurred, or `nvml_total_used_gib` on
  actor ranks is above ~85% of the device cap.
- **Divisibility:** `actor.global_batch_size %
  (actor.micro_batch_size * actor_world_size) == 0` is required
  (`rlinf/config.py:1363-1368`). Preflight will reject a non-divisor;
  propose a divisor of `global_batch_size / actor_world_size` instead
  of a nearby number.
- **Note on pinned `global_batch_size`:** the schema pins
  `actor.global_batch_size` (FUT-5), so the critic can only move
  `micro_batch_size` within its divisors.
- **Non-monotonic effect of mbs:** when GPU memory allows, increasing
  `micro_batch_size` first *decreases* `T_act` (fewer micro-batches,
  higher arithmetic intensity). However, once memory usage approaches
  the device cap or crosses a critical threshold, further increasing
  `micro_batch_size` *increases* `T_act` instead — fragmentation,
  spilling, and scheduling overhead dominate. The optimal mbs is
  therefore not the largest divisor but the one at the inflection point
  before the super-linear slowdown.
- **Evidence gate:** use the non-monotonic rule only when memory curves,
  OOM history, or adjacent mbs trials support it. Without that evidence,
  prefer increasing mbs for actor-bound throughput and decreasing it for
  actor OOM / high memory.

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
