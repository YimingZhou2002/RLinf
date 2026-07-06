# 02 Optimization directions per knob

Each knob's effect on the three critical-path terms and the memory
budget. Use this together with `01-placement-critical-paths.md` to decide
whether a proposed delta will actually reduce `step_time /
num_trajectories`.

The overall objective `step_time / num_trajectories` can be decomposed
into reducing the per-trajectory time of each component on the critical
path (`T_env / traj`, `T_rol / traj`, `T_act / traj`), which in turn
decomposes into the sub-tasks each knob targets.

**Trajectory-scaling model.** All three components' wall times are
functions of the total trajectories produced per step,
`num_trajectories = env.train.total_num_envs * env.train.rollout_epoch *
env.train.max_steps_per_rollout_epoch * group_size` (see
`num_trajectories` in the MetricTable). The scaling shape differs by
component and drives whether growing / shrinking those knobs helps the
objective:

- `T_env` and `T_rol`: roughly **linear** in `num_trajectories` — each
  extra traj is one more env step and one more rollout inference of
  fixed size (`T_env ≈ a_env * num_trajectories + b_env`, symmetric for
  rollout). Doubling `total_num_envs` or `rollout_epoch` roughly doubles
  each term.
- `T_act`: **linear at first, then super-linear** past a threshold.
  While the traj count fits comfortably in actor memory, more traj = more
  micro-batches at fixed cost per batch. Once the accumulated activation
  / optimizer-state footprint approaches the device cap, allocator
  retries, fragmentation, and gradient-checkpointing recompute kick in
  and per-traj actor time grows.

Two consequences for the objective:

- Growing a traj-generating knob (`total_num_envs`, `rollout_epoch`)
  usually leaves `T_env / traj` and `T_rol / traj` unchanged, but only
  reduces `T_act / traj` while actor is still in the linear regime; past
  the inflection it *increases* `T_act / traj`.
- Shrinking a traj-generating knob to escape actor's super-linear regime
  is a valid throughput move even though `num_trajectories` drops — the
  win is `T_act / traj` falling faster than `num_trajectories`.

## 02.1 `cluster.component_placement`

- **What it moves:** reshapes the critical path itself.
- **When to touch:**
  - Under collocated when one of `T_env`, `T_rol`, `T_act` dominates and
    the other two barely register — the parallelism won by disaggregating
    is worth the memory pressure of dedicated GPU sets.
  - Under hybrid when the timeline shows large `stall_fraction` on the
    faster of env / rollout — reallocate GPUs from the idle side to the
    busy side.
  - Under hybrid when, **excluding the first two interact chunks of a
    step** (offload/onload warmup skews their timings), the steady-state
    per-chunk `T_env` and `T_rol` are imbalanced — shift GPUs toward the
    side with the longer interact time. Exclude the warmup chunks
    explicitly because using them would bias the decision toward the
    side that pays the onload cost, not the side that is actually the
    steady-state bottleneck.
- **Common failure mode:** proposing a disaggregated split that leaves
  actor with too few GPUs; `T_act` grows past every other term and now
  the actor is bound, not helping.
- **Dual-source rationale required** (see the README): any placement
  delta must cite one MetricTable line AND one timeline observation.

## 02.2 `env.train.total_num_envs`

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
- **Cross-component coupling:** `total_num_envs` is not env-local — every
  interact chunk feeds `total_num_envs` trajectories into rollout, so
  rollout per-chunk work scales roughly linearly with it (`T_rol ≈ a *
  total_num_envs + b`, where `b` is the fixed rollout overhead per
  chunk). Growing `total_num_envs` to shrink env can therefore push
  rollout past env on the hybrid `max(T_env, T_rol)` critical path;
  check the projected `T_rol` against current `T_env` before proposing
  the delta, and cite both terms in the rationale.
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

## 02.3 `env.train.rollout_epoch` (denoted `R` above)

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

## 02.4 `actor.micro_batch_size`

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
  `actor.global_batch_size`, so the critic can only move
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

## 02.5 `env.train.enable_offload`

- **What it moves:** moves env-side state between GPU and CPU/host
  between phases. Cuts env GPU memory at the cost of extra transfer
  latency per phase.
- **When to enable:** env-side OOM under the current placement, or env
  is not on the critical path and its memory is squeezing another
  worker's headroom.
- **When to disable:** env is the critical-path bottleneck; the offload
  overhead widens `T_env` and worsens the objective.

## 02.6 `rollout.enable_offload`

- **What it moves:** symmetric to env offload but for rollout weights and
  KV cache. Rollout offload is expensive in wall time (moves a large
  model checkpoint), so it is a memory rescue knob, not a throughput
  knob.
- **When to enable:** rollout OOM. Also enable on **actor OOM** when
  actor and rollout share GPU ranks (typical hybrid: `actor: 0-7`,
  `rollout: 4-7`) — offloading rollout weights frees the shared GPUs'
  memory during actor training, so the OOM rescue works even though the
  failing component is the actor. If rollout is on the critical path,
  offload will almost certainly regress `step_time`; only accept the
  regression when the OOM alternative is a crash.
- **When to disable:** rollout is bottleneck and enough memory exists to
  keep the model resident.

## 02.7 `actor.enable_offload`

- **What it moves:** offloads actor optimiser state / weights between
  training steps. Big memory saving, big wall-time cost.
- **When to enable:** actor or rollout OOM only.
- **When to disable:** actor is on the critical path or nvml curves show
  large downward-then-upward memory swings around each `run_training`
  call, which is the offload signature.

## 02.8 Pinned in this loop (FUT-5)

These are declared in the schema but rejected by the validator with
`KnobNotTunableError`. Do not propose them:

- `actor.global_batch_size`
- `rollout.pipeline_stage_num`
- `actor.model.num_action_chunks`

## 02.9 Cross-knob patterns

- **Memory-triage cascade** on repeated OOM: `enable_offload=true` on
  the OOM component (or on rollout if actor OOMed and actor-rollout
  share GPU ranks) → shrink `actor.micro_batch_size` → shrink
  `env.train.total_num_envs`. Offload first because it is the largest
  single memory win and is fully revertible with one flag; only reach
  for `mbs` / `total_num_envs` shrinks after offload has failed to
  rescue, since those shrinks change the batching regime and directly
  reduce `num_trajectories` or arithmetic intensity.
- **Rebalance under hybrid**: when the timeline shows env-rollout
  stall imbalance ≥ 0.3, move GPUs from the idle side to the busy side
  via `component_placement`. Do not grow `total_num_envs` at the same
  time — one move at a time makes the ledger interpretable.
- **Actor-bound recovery**: if `T_act` dominates and actor memory is
  tight, growing `micro_batch_size` may OOM. Consider disabling
  `actor.enable_offload` and `rollout.enable_offload` (if either is on
  and actor-rollout share GPU ranks) before growing the batch size —
  offload is often the reason `T_act` is bloated in the first place,
  and turning it off frees the shared GPUs' memory so a larger
  `micro_batch_size` fits.
