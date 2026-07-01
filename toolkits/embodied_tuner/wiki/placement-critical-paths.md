# Placement critical paths

The single most important reasoning tool for the critic. Shortening a
component only reduces `step_time` when that component sits on the
critical path for the current placement AND the current runner mode.
This file gives the closed-form critical path and its timeline
signature.

Notation:

- `T_env` — wall time of one env `env_interact_step`.
- `T_rol` — wall time of one rollout `predict` (a.k.a. `generate`).
- `T_act` — wall time of one actor `run_training` (recv_traj + compute_adv +
  forward + policy_loss + backward + optimizer_step).
- `T_sync` — wall time of `actor/sync_model_to_rollout` when
  `weight_sync_interval=1`; zero otherwise.
- `R` — `env.train.rollout_epoch` × `env.train.max_steps_per_rollout_epoch`
  effectively — the number of interact chunks per training step.
- `step_time` — per-`global_step` wall time reported in `metrics.log`
  under `time/step_time`.

The three components' contributions to a single training step are:

- **Env chunk** — repeated once per interact chunk (R total per step).
- **Rollout chunk** — repeated once per interact chunk (R total per
  step), interleaved with env chunks via channels.
- **Actor training** — runs once per step, on the R accumulated chunks.

## Runner mode matters

The runner exposes two loop shapes (see `rlinf/runners/embodied_runner.py`):

- **`EmbodiedRunner.run`** (default when `runner.use_training_pipeline=false`).
  Interact loop and actor training are **serial within a step** because
  actor training is preceded by
  `self.actor.recv_rollout_trajectories(...).wait()` (line 518). The
  actor cannot start work until every trajectory of the current step has
  been received.
- **`EmbodiedRunner.run_pipeline`** (opt-in via
  `runner.use_training_pipeline=true`). Actor training of step N
  overlaps the interact loop of step N. Enables the `max(...)` critical
  path under disaggregated placements.

The tuner does **not** currently flip `use_training_pipeline` — treat
whatever the baseline sets as fixed.

## 1. Collocated (all components share the same GPU set)

Every worker time-slices on the same GPUs, so nothing overlaps.

    step_time ≈ T_sync + R * (T_env + T_rol) + T_act

Every component is on the critical path. Any shrink helps. The right
question is which one dominates:

    dominant_component = argmax(R * T_env, R * T_rol, T_act)

Timeline signature: env / rollout / actor bars never overlap in wall
time on any rank. The lane sequence per step is
`sync_model_to_rollout → { env_interact_step, predict } × R → actor forward/backward`.

Typical `component_placement`: `{actor: all, env: all, rollout: all}`
or `{actor: 0-7, env: 0-7, rollout: 0-7}`.

## 2. Hybrid — env and rollout on disjoint GPU subsets, actor spans both

Canonical embodied hybrid: `actor: 0-7`, `env: 0-3`, `rollout: 4-7`. Env
and rollout run on disjoint GPU subsets so they can execute concurrently
as a producer-consumer pair through the env-channel and rollout-channel.
Actor still runs on all 8 GPUs but only after the R chunks finish (see
`recv_rollout_trajectories.wait()` line 518).

For each of the R interact chunks, the steady-state per-chunk time is
approximately `max(T_env, T_rol)`.

Both `run` and `run_pipeline` share the same interact-side critical path
here; `run_pipeline` additionally overlaps actor with the interact loop
of the SAME step, but under hybrid the actor uses all 8 GPUs and
therefore contends with env and rollout for those GPUs — the pipeline
gain is small and often negative.

**Under `run`:**

    step_time ≈ T_sync + R * max(T_env, T_rol) + T_act

**Under `run_pipeline`:** approximately the same (actor GPU set overlaps
env+rollout GPU sets, so overlap is minimal).

Consequences:

- If `T_env > T_rol` **and** rollout GPUs sit idle waiting on env,
  shrinking `T_rol` does **not** help — env is the bottleneck. Cite
  `env/interact` in the MetricTable and env rank `stall_fraction ≈ 0`
  (env is busy) alongside a growing `predict.stall_fraction` on rollout
  ranks. Remedies: reduce `env.train.total_num_envs`, offload env, or
  reallocate GPUs from rollout to env.
- If `T_rol > T_env`, symmetric argument in reverse.
- If `T_act` alone dominates `R * max(T_env, T_rol)`, the R-fold
  parallel gain is wasted — reduce `micro_batch_size` (memory
  permitting), or reduce R so the actor gets called more often on
  smaller trajectories.

Timeline signature: on env ranks, `env_interact_step` bars overlap in
wall time with `predict` bars on rollout ranks. Actor bars appear only
after the R-th chunk completes on both subsets.

## 3. Disaggregated — actor, env, rollout on three disjoint GPU subsets

Example: `actor: 0-3`, `env: 4-5`, `rollout: 6-7`. All three GPU sets
are disjoint, so env, rollout, and actor **can** run concurrently.
Whether they actually do depends on the runner mode.

**Under `run` (default, `use_training_pipeline=false`):**
Actor still gates on `recv_rollout_trajectories.wait()`. So the critical
path is exactly the same as hybrid — the disjoint actor GPUs simply sit
idle during the interact loop:

    step_time ≈ T_sync + R * max(T_env, T_rol) + T_act

If the baseline is `run` mode, disaggregated placement wastes the actor's
GPUs whenever the interact loop is running. Prefer collocating the actor
onto all GPUs (i.e. use hybrid) unless you have a memory reason to
reserve dedicated actor GPUs.

**Under `run_pipeline` (`use_training_pipeline=true`):**
Actor training of step N overlaps the interact loop of step N. Steady
state:

    step_time ≈ T_sync + max( R * T_env, R * T_rol, T_act )

Add one `R * max(T_env, T_rol)` for pipeline drain at the very last step
of the run — negligible over long runs. This is the pattern where
disaggregation actually pays off.

Timeline signature (pipeline mode): env, rollout, and actor tag lanes
all show overlapping bars in the same wall-time window. `stall_fraction`
is the primary signal: whichever component has `stall_fraction ≈ 0` is
the bottleneck; the other two show `stall_fraction > 0`.

## Quick reference

| Placement                               | Critical path per step                    | What shrinking a non-bottleneck buys |
|-----------------------------------------|-------------------------------------------|--------------------------------------|
| Collocated                              | `T_sync + R*(T_env+T_rol) + T_act`        | direct — every component on the path |
| Hybrid (both runner modes)              | `T_sync + R*max(T_env,T_rol) + T_act`     | zero for the shorter of env / rol    |
| Disaggregated + `run`                   | `T_sync + R*max(T_env,T_rol) + T_act`     | zero for the shorter of env / rol; actor GPUs also idle during interact |
| Disaggregated + `run_pipeline`          | `T_sync + max(R*T_env, R*T_rol, T_act)`   | zero for non-max components          |

## Anti-patterns

- Proposing `env.train.enable_offload=true` when the timeline shows
  env is already the fastest of the three under hybrid — this trades
  T_env for a memory saving that does not help the objective.
- Moving GPUs from actor to rollout under hybrid when `T_rol < T_env`
  — hybrid critical path is `max(T_env, T_rol)`, so growing rollout's
  GPU share cannot beat env, and shrinking actor slows the `T_act`
  term on the same critical path.
- Growing `env.train.total_num_envs` to reduce per-env stalls when the
  MetricTable already shows `env/interact` is the smallest term of the
  three — you'll gain nothing but risk OOM.
- Proposing a disaggregated placement under `run` (non-pipeline) mode
  expecting a `max(...)` critical path — it stays additive. Cite the
  runner mode in the rationale.
