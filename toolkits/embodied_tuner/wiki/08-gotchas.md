# 08 Gotchas — consolidated anti-patterns

The scattered "do not" rules from the rest of the wiki, gathered in
one place. Each item names the failure mode, the reason, and where
the underlying rule lives.

## 1. Do not cite blocking-wait tags as bottlenecks

`actor/recv_traj`, `actor/recv_rollout_trajectories`, `actor/recv_rollout_results`
are waits on the interact loop, not actor work. Under hybrid the actor
is simply idle while rollout and env interact on the same GPUs.

Cite the actor's own phase tags: `actor_forward`, `actor_backward`,
`actor_optimizer_step`, `compute_advantages_and_returns`. Use
`actor/run_training` only as MetricTable sanity-check when the phase
tags are missing.

See `04-signals.md §2.1`.

## 2. Do not cite wrapper tags

Wrapper tags span waits and inflate durations:

- `runner:run`
- `interact`, `run_interact_once`
- `generate`, `generate_one_epoch`
- `run_training`
- `recv_rollout_*` (also §1)

The parser's outlier and critical-path blocks already exclude the
noisiest wrappers. If one appears, it survived because a specific
analysis (e.g. blocked-vs-real split) needed it.

See `04-signals.md §4`.

## 3. Do not propose disaggregated under `run` mode expecting parallelism

Under `use_training_pipeline=false` the actor gates on
`recv_rollout_trajectories.wait()`, so a disaggregated placement's
actor GPUs sit idle during interact. The critical path stays
`T_sync + R*max(T_env, T_rol) + T_act`, not `T_sync + max(R*T_env,
R*T_rol, T_act)`. Cite the runner mode in the rationale.

See `02-paths.md §4.1`.

## 4. Do not propose more than one placement move per delta

Placement moves interact with divisibility constraints
(`total_num_envs % env_world_size`, routing divisibility from
`07-constraints.md §2.6`) and often need a follow-up shrink. Keep
those in separate rounds so the ledger attributes cause and effect.

See `05-recipe.md §7`.

## 5. Do not propose `enable_offload=true` on the current critical-path component

Offloading is a memory rescue knob, not a throughput knob. It widens
the wall time of whichever component it applies to.

- Env offload additionally forces `runner.overlap_env_bootstrap=False`
  (`07-constraints.md §2.3`) which can widen the actor's serial gap
  under hybrid.
- Rollout / actor offload move large model state and are wall-time
  expensive.

Propose them when the last trial hit `FailureMode=OOM` on that
component (or on actor when actor and rollout share GPU ranks — see
`06-playbook.md §6`) and the more effective memory rescues are
exhausted.

Offload is also acceptable *without* a crash: when the expand-from
parent is under high memory occupancy (the §7.1 soft-pressure
`WARNING`, `max_mem_occ >= 95%`) and offloading a component opens headroom that enables a delta with a clearly better
throughput payoff. The offload is the enabler for that stronger
optimization, not a throughput move in itself. The hard rule stays:
never offload the component currently on the critical path.

See `06-playbook.md §§5-7`.

## 6. Do not propose a knob that has already been tried at the same value

The trial history block is deduplicated for exactly this check. Same
value = no new evidence unless the surrounding placement / memory /
knob environment has changed.

See `05-recipe.md §7`.

## 7. Do not reach for placement in the OOM branch

Under memory pressure (the `## Memory pressure` block present, i.e. the
expand-from parent itself failed with OOM — see `03-inputs.md §7` for
why a reverted sibling OOM does NOT count), follow the memory triage
cascade in `06-playbook.md §8.1` first:
`enable_offload=true` on the OOM component (or on rollout if actor
OOMed and actor-rollout share GPU ranks) → shrink
`actor.micro_batch_size` → shrink `env.train.total_num_envs`.

Placement moves are the highest-variance change in the toolkit and
rarely fix an OOM by themselves.

**Distinguish a reverted sibling OOM from current pressure.** If the
only OOM you see is a `Recent failure leaf` whose delta the scheduler
already rolled back (the `## Current knobs` reflect the parent, not
the failed sibling), you are NOT in the OOM branch — the config you
are expanding from did not OOM. Record the failure as a `bitter_lesson`
and propose forward from the parent; do not run the triage cascade and
do not reach for placement.

See `05-recipe.md §4` (rules of thumb).

## 8. Do not optimize raw `step_time`

The objective is `step_time / num_trajectories`. A delta that reduces
`step_time` but reduces `num_trajectories` by the same or larger
fraction is not an improvement.

`env.train.total_num_envs` and `env.train.rollout_epoch` both change
the denominator; every proposal touching them must project the
normalised objective, not just the numerator.

See `01-concepts.md §1` and `06-playbook.md §§2-3`.

## 9. Do not bundle unrelated knobs

**Default: one knob per delta.** Two knobs only when they are
logically inseparable — e.g. adjusting `cluster.component_placement`
together with `env.train.total_num_envs`, because the new
`env_world_size` / `rollout_world_size` would violate the divisibility
constraints (`07-constraints.md §1.3, §2.6`) unless `total_num_envs`
moves at the same time.

**Exception: failed-trial revert bundle.** When the last trial
failed, bundle the rollback and the next move into a single delta.
Name both moves in `rationale.summary`.

See `05-recipe.md §7`.

## 10. Do not treat the MetricTable as the primary bottleneck source

When MetricTable Time-section keys disagree with the critical-path
per-step block by more than ~10%, trust the timeline. The MetricTable
key is usually a wrapper (`env/interact`, `rollout/generate`,
`actor/run_training`) that sums wait time in.

Priority order: critical path per global_step > per-component bubble >
component call averages > MetricTable Time-section.

See `03-inputs.md §10`.

## 11. Do not omit dual-source citations on placement deltas

Any delta that touches `cluster.component_placement` must cite at
least one MetricTable key=value line AND at least one timeline
observation. The validator rejects placement deltas with either array
empty.

See `05-recipe.md §6` and `critic.py:CriticOutputValidator.validate`.

## 12. Do not skip the bitter lesson after a failure

When the previous trial's `failure_mode` is
`OOM, WORKER_CRASH, TIMEOUT, CONFIG_INVALID, DIVISIBILITY_VIOLATION`,
the response must include a non-empty `bitter_lesson.trigger` AND
`bitter_lesson.rule`. Otherwise the failure is forgotten after the
history window rolls over and the same delta may be re-proposed.

## 13. Do not leave GPUs uncovered under hybrid to satisfy divisibility

Under hybrid, `env_world_size + rollout_world_size` must equal
`num_gpus`. Preflight accepts partial coverage
(e.g. `env={0}, rollout={1-4}` on an 8-GPU node) because contiguity
and disjointness both hold, but GPUs 5-7 then sit idle during the
interact loop and the placement change buys less than the timeline
suggested.

If the desired world_size (e.g. rollout=5) does not divide the current
`total_num_envs`, do NOT retreat to a divisor-safe world_size that
leaves GPUs idle. Bundle a `total_num_envs` adjustment into the same
delta — this is the one placement-scoped exception to the
one-knob-per-delta rule (`05-recipe.md §7`, `06-playbook.md §8.2`).

See `06-playbook.md §1` for the GPU-coverage requirement and its
common failure modes.

See `05-recipe.md §8` and `03-inputs.md §3`.
