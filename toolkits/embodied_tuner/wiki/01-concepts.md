# 01 Concepts

The vocabulary every later file uses. Read once; later files assume
these definitions without repeating them.

## 1. Objective

Every delta is judged by

    objective = step_time / num_trajectories

not raw `step_time`. A delta that shrinks `step_time` but shrinks
`num_trajectories` by the same fraction is a wash; a delta that shrinks
`step_time` less than it shrinks `num_trajectories` is a regression.

- `step_time` — per-`global_step` wall time, reported in `metrics.log`
  under `time/step_time`.
- `num_trajectories` — trajectories collected per training step:

        num_trajectories =
            env.train.total_num_envs
          * env.train.rollout_epoch
          * env.train.max_steps_per_rollout_epoch
          * env.train.group_size

  Present verbatim in the `## Last trial — MetricTable Environment-section`
  block of the prompt.

## 2. Notation

| Symbol   | Meaning                                                                                           |
|----------|---------------------------------------------------------------------------------------------------|
| `T_env`  | wall time of one env `env_interact_step`                                                          |
| `T_rol`  | wall time of one rollout `predict` (a.k.a. `generate`)                                            |
| `T_act`  | wall time of one actor `run_training` (recv_traj + compute_adv + forward + policy_loss + backward + optimizer_step) |
| `T_sync` | wall time of `actor/sync_model_to_rollout` when `weight_sync_interval=1`; zero otherwise          |
| `R`      | `env.train.rollout_epoch * env.train.max_steps_per_rollout_epoch` — interact chunks per training step |

## 3. Per-step component contributions

Every training step consists of:

- **Env chunk** — one `env_interact_step` per env-side rank, repeated
  `R` times.
- **Rollout chunk** — one `predict` per rollout-side rank, interleaved
  with env chunks via channels, `R` times.
- **Actor training** — one `run_training` on the `R` accumulated
  chunks, once per step.

The runner also injects a `sync_model_to_rollout` at the start of each
step (`T_sync`) when the weight-sync interval fires.

## 4. Placement modes

Every mode uses **all** GPUs. What differs is how the three components
share them. See `cluster.component_placement` in the current-knobs block.

- **collocated** — env, rollout, and actor all occupy every GPU and
  time-share the hardware. No two components run at the same time; the
  three phases execute serially per rollout step.
  Typical: `{actor: 0-7, env: 0-7, rollout: 0-7}` or `{actor: all, env: all, rollout: all}`.
- **hybrid** — env and rollout on disjoint GPU shards (together
  covering all GPUs); actor spans both sides and blocks while interact
  runs, then reclaims those GPUs for training.
  Typical: `{actor: 0-7, env: 0-3, rollout: 4-7}`.
- **disaggregated** — env, rollout, and actor each own a disjoint GPU
  shard. All three can run concurrently in principle; whether they
  actually do depends on the runner mode (see §5).
  Typical: `{actor: 0-3, env: 4-5, rollout: 6-7}`.

## 5. Runner modes

The runner (`rlinf/runners/embodied_runner.py`) exposes two loop shapes:

- **`EmbodiedRunner.run`** — default when
  `runner.use_training_pipeline=false`. Interact loop and actor training
  are **serial within a step**: actor training is gated by
  `self.actor.recv_rollout_trajectories(...).wait()` (line 518). The
  actor cannot start work until every trajectory of the current step
  has been received.
- **`EmbodiedRunner.run_pipeline`** — opt-in via
  `runner.use_training_pipeline=true`. Actor training of step N
  overlaps the interact loop of step N. Enables the `max(...)` critical
  path shape under disaggregated placements.

The tuner does **not** flip `use_training_pipeline` — treat whatever
the baseline sets as fixed. Read the current value from
`runner.use_training_pipeline` in the current-knobs block.

## 6. Trajectory-scaling model

Each component's wall time scales differently with `num_trajectories`:

- `T_env` and `T_rol`: roughly **linear** in `num_trajectories`. Each
  extra traj is one more env step and one more rollout inference of
  fixed size (`T_env ≈ a_env * num_trajectories + b_env`, symmetric for
  rollout). Doubling `total_num_envs` or `rollout_epoch` roughly
  doubles each term.
- `T_act`: **linear at first, then super-linear** past a threshold.
  While the traj count fits comfortably in actor memory, more traj =
  more micro-batches at fixed cost per batch. Once the accumulated
  activation / optimizer-state footprint approaches the device cap,
  allocator retries, fragmentation, and gradient-checkpointing recompute
  kick in and per-traj actor time grows.

Two consequences for the objective:

- Growing a traj-generating knob (`total_num_envs`, `rollout_epoch`)
  usually leaves `T_env / traj` and `T_rol / traj` unchanged, but only
  reduces `T_act / traj` while actor is still in the linear regime;
  past the inflection it *increases* `T_act / traj`.
- Shrinking a traj-generating knob to escape actor's super-linear
  regime is a valid throughput move even though `num_trajectories`
  drops — the win is `T_act / traj` falling faster than
  `num_trajectories`.
