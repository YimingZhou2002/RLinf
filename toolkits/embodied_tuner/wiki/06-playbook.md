# 06 Knob playbook

Per-knob effects on the three critical-path terms and the memory
budget. Read alongside `02-paths.md` (which formula applies) and
`05-recipe.md §4` (which knob is first-line for which bottleneck).

Per-knob preflight checks are inlined here; the full constraint
catalog is `07-constraints.md`.

## 1. `cluster.component_placement`

- **What it moves:** reshapes the critical path itself
  (`02-paths.md`).
- **GPU coverage requirement:** on the current single-node target
  (`num_gpus=8`), placements must cover every GPU:
  - Under **collocated**, every component covers all 8 GPUs.
  - Under **hybrid**, `env_world_size + rollout_world_size ==
    num_gpus` (env and rollout disjoint and together spanning `0..num_gpus-1`);
    actor spans all GPUs. A hybrid with `env={0}, rollout={1-4}`
    leaves GPUs 5-7 idle during interact — that is a throughput
    regression, not a legal fallback, even though preflight accepts
    it. Prefer to shift `total_num_envs` (see §8.2) so the "busy"
    side can absorb the freed GPU.
  - Under **disaggregated**, the three components' shards together
    cover all GPUs.
- **When to touch:**
  - Under collocated when one of `T_env`, `T_rol`, `T_act` dominates
    and the other two barely register — the parallelism won by
    disaggregating is worth the memory pressure of dedicated GPU sets.
  - Under hybrid when the timeline shows a large `stall_fraction` on
    the faster of env / rollout — reallocate GPUs from the idle side
    to the busy side.
  - Under hybrid when, **excluding the first two interact chunks of a
    step** (offload/onload warmup skews their timings), the
    steady-state per-chunk `T_env` and `T_rol` are imbalanced — shift
    GPUs toward the side with the longer interact time. The
    steady-state view is the "component call averages" sub-block
    (`03-inputs.md §9.4`), which already drops the first 2 events.
- **Common failure modes:**
  - Disaggregated split that leaves actor with too few GPUs; `T_act`
    grows past every other term and now the actor is bound, not
    helping.
  - Hybrid placement that leaves GPUs uncovered (`env_world_size +
    rollout_world_size < num_gpus`), chosen because the "cleanly
    divisible" world_size wastes GPUs. Bundle a `total_num_envs`
    adjustment (§8.2, `05-recipe.md §7`) instead of leaving GPUs
    idle.
- **Dual-source rationale required:** any placement delta must cite
  one MetricTable line AND one timeline observation (`05-recipe.md §6`).
- **Preflight checks (`07-constraints.md §4.1`):** every range
  contiguous and in `[0, num_gpus)`; env and rollout equal or fully
  disjoint; downstream divisibility (`total_num_envs %
  env_world_size`, routing divisibility from `07-constraints.md §2.6`,
  and the actor `global_batch_size / (micro_batch_size *
  actor_world_size)` divisor).

## 2. `env.train.total_num_envs`

- **What it moves:** batches env stepping and changes both `step_time`
  and `num_trajectories`. Larger usually increases raw `T_env` and GPU
  memory, but may reduce `T_env / trajectory` if batching efficiency
  improves.
- **When to grow:** env is the critical-path bottleneck, env GPU
  memory has headroom, and trial history or MetricTable normalisation
  suggests `step_time / num_trajectories` will decrease.
- **When to shrink:** env OOM, env memory has grown past ~85% of the
  device cap, or recent trials show super-linear `T_env` growth such
  that fewer envs should improve `step_time / num_trajectories`.
- **Cross-component coupling:** every interact chunk feeds
  `total_num_envs` trajectories into rollout, so rollout per-chunk work
  scales roughly linearly with it (`T_rol ≈ a * total_num_envs + b`,
  where `b` is the fixed rollout overhead per chunk). Growing
  `total_num_envs` to shrink env can push rollout past env on the
  hybrid `max(T_env, T_rol)` critical path; check the projected
  `T_rol` against current `T_env` before proposing the delta, and cite
  both terms.
- **Preflight checks (`07-constraints.md §4.2`):** in `[1, 4096]`;
  `total_num_envs % env_world_size == 0` (`07 §1.3`); routing
  divisibility (`07 §2.6`); `group_size` divisibility (`07 §2.1`);
  matching eval-side check when `runner.val_check_interval > 0`.
- **Evidence gate:** treat linear / super-linear scaling as a
  hypothesis unless trial history contains adjacent env counts,
  p90/median worsens, or memory approaches the cap. Prefer the move
  whose expected `step_time / num_trajectories` is lower.

## 3. `env.train.rollout_epoch` (denoted `R`)

- **What it moves:** changes the number of interact chunks and
  collected trajectories per training step. `step_time` contains an
  `R * ...` term under every placement, while `num_trajectories` also
  scales with R.
- **When to shrink:** critical path is dominated by the env/rollout
  chunk term (`R * max(T_env, T_rol)` under hybrid, or `R * T_env` /
  `R * T_rol` under disaggregated) AND the expected reduction in
  `step_time` is larger than the reduction in `num_trajectories`.
- **When to grow:** actor is the bottleneck under hybrid or
  disaggregated (`T_act > R * max(T_env, T_rol)`) AND larger R should
  reduce actor cost per trajectory by amortising actor overhead over
  more collected samples.
- **Preflight checks:** in `[1, 16]`.
- **Interaction with `num_trajectories`:** the objective normalises by
  `num_trajectories`, and `num_trajectories` scales linearly with R.
  A smaller R buys smaller `step_time` but also smaller
  `num_trajectories`; watch that `step_time / num_trajectories`
  actually improves.
- **Evidence gate:** treat rollout super-linear scaling as a
  hypothesis unless trial history, p90/median, KV-cache pressure, or
  memory curves support it. Optimise the normalised objective, not
  raw `step_time`.

## 4. `actor.micro_batch_size`

- **What it moves:** memory footprint of one actor forward+backward
  pass, and the number of micro-batches per `global_batch_size`.
- **When to grow:** actor OOM headroom exists AND actor is on the
  critical path — fewer micro-batches = fewer kernel launches, higher
  arithmetic intensity, lower `T_act`.
- **When to shrink:** actor OOM occurred, or `nvml_total_used_gib` on
  actor ranks is above ~85% of the device cap.
- **Preflight checks (`07-constraints.md §4.3`):** in `[1, 4096]`;
  `actor.global_batch_size % (actor.micro_batch_size *
  actor_world_size) == 0` — propose a divisor of `global_batch_size /
  actor_world_size` instead of a nearby number.
- **Non-monotonic effect:** when GPU memory allows, increasing
  `micro_batch_size` first *decreases* `T_act` (fewer micro-batches,
  higher arithmetic intensity). Once memory usage approaches the
  device cap or crosses a critical threshold, further increases
  *increase* `T_act` instead — fragmentation, spilling, and scheduling
  overhead dominate. The optimal mbs is the inflection point before
  the super-linear slowdown, not the largest divisor. If `T_act /
  num_trajectories` worsened after the last bump, roll back one step
  rather than pushing further.
- **Evidence gate:** use the non-monotonic rule only when memory
  curves, OOM history, or adjacent mbs trials support it. Without that
  evidence, prefer increasing mbs for actor-bound throughput and
  decreasing it for actor OOM / high memory.

## 5. `env.train.enable_offload`

- **What it moves:** moves env-side state between GPU and CPU/host
  between phases. Cuts env GPU memory at the cost of extra transfer
  latency per phase.
- **When to enable:** env-side OOM under the current placement, or env
  is not on the critical path and its memory is squeezing another
  worker's headroom.
- **When to disable:** env is the critical-path bottleneck; the
  offload overhead widens `T_env` and worsens the objective. Disabling
  is a valid throughput lever under hybrid when env GPUs have
  headroom.
- **Runtime side-effect (`07-constraints.md §2.3`):**
  `runner.overlap_env_bootstrap` is silently forced to `False` when
  this is `true`, which under hybrid can widen the actor's serial gap.
  Do not enable env offload as a throughput knob.

## 6. `rollout.enable_offload`

- **What it moves:** symmetric to env offload but for rollout weights
  and KV cache. Expensive in wall time (moves a large model
  checkpoint), so it is a memory rescue knob, not a throughput knob.
- **When to enable:** rollout OOM. Also enable on **actor OOM** when
  actor and rollout share GPU ranks (typical hybrid: `actor: 0-7`,
  `rollout: 4-7`) — offloading rollout weights frees the shared GPUs'
  memory during actor training, so the OOM rescue works even though
  the failing component is the actor. If rollout is on the critical
  path, offload will almost certainly regress `step_time`; only accept
  the regression when the OOM alternative is a crash.
- **When to disable:** rollout is bottleneck and enough memory exists
  to keep the model resident.

## 7. `actor.enable_offload`

- **What it moves:** offloads actor optimiser state / weights between
  training steps. Big memory saving, big wall-time cost.
- **When to enable:** actor or rollout OOM only.
- **When to disable:** actor is on the critical path, or nvml curves
  show large downward-then-upward memory swings around each
  `run_training` call (the offload signature).

## 8. Cross-knob patterns

### 8.1 Memory triage cascade (on repeated OOM)

`enable_offload=true` on the OOM component (or on rollout if actor
OOMed and actor-rollout share GPU ranks) → shrink
`actor.micro_batch_size` → shrink `env.train.total_num_envs`.

Offload first because it is the largest single memory win and is fully
revertible with one flag; only reach for `mbs` / `total_num_envs`
shrinks after offload has failed to rescue, since those shrinks change
the batching regime and directly reduce `num_trajectories` or
arithmetic intensity.

### 8.2 Rebalance under hybrid

When the timeline shows env-rollout stall imbalance ≥ 0.3, move GPUs
from the idle side to the busy side via `component_placement`. The
new placement should still cover every GPU on the node — under hybrid
`env_world_size + rollout_world_size == num_gpus` (see §1), otherwise
the leftover GPUs sit idle during interact and the delta buys less
than the placement change suggests.

If the new `env_world_size` or `rollout_world_size` does not divide
the current `total_num_envs` (see `07-constraints.md §1.3, §2.6`),
bundle a `total_num_envs` adjustment into the same delta rather than
falling back to a partial-coverage placement. This is the placement
exception to the one-knob-per-delta rule (`05-recipe.md §7`). Name
both moves in `rationale.summary` so the ledger stays attributable.
Do NOT bundle any other knob (offload, mbs, rollout_epoch) — the
exception is scoped to placement + `total_num_envs`.

### 8.3 Actor-bound recovery

If `T_act` dominates and actor memory is tight, growing
`micro_batch_size` may OOM. Consider disabling `actor.enable_offload`
and `rollout.enable_offload` (if either is on and actor-rollout share
GPU ranks) before growing the batch size — offload is often the
reason `T_act` is bloated in the first place, and turning it off frees
the shared GPUs' memory so a larger `micro_batch_size` fits.

## 9. Pinned knobs (do not touch in this loop)

Declared in the schema but rejected by the validator with
`KnobNotTunableError`. Do not include them in a delta:

- `actor.global_batch_size`
- `rollout.pipeline_stage_num`
- `actor.model.num_action_chunks`

Note: because `actor.global_batch_size` is pinned, `micro_batch_size`
can only move within its divisors (see §4 preflight check).
