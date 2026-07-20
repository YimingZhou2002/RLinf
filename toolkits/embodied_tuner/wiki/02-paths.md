# 02 Critical paths

Given the notation from `01-concepts.md`, the critical path is the
closed-form model of `step_time` under each `(placement, runner_mode)`
combination. Shortening a component reduces `step_time` only when that
component sits on the current critical path; otherwise the delta buys
zero.

The objective is still `step_time / num_trajectories`; these formulas
give you the numerator.

## 1. Quick reference

| Placement                          | Runner mode      | Critical path per step                          | Shrinking a non-max term buys |
|------------------------------------|------------------|--------------------------------------------------|-------------------------------|
| Collocated                         | either           | `T_sync + O_env+O_rol+R*(T_env + T_rol) + T_act+O_act`             | direct — every term on path   |
| Hybrid                             | either           | `T_sync + max(O_env+O_rol)+R*max(T_env, T_rol) + T_act`           | 0 for the shorter of env / rol |
| Disaggregated                      | `run`            | `T_sync + max(O_env+O_rol)+R*max(T_env, T_rol) + T_act`           | 0 for shorter of env / rol; actor GPUs also idle during interact |
| Disaggregated                      | `run_pipeline`   | `T_sync + max(O_env+O_rol) + max(R*T_env, R*T_rol, T_act+O_act)`          | 0 for non-max components      |

## 2. Collocated

Every worker time-slices on the same GPUs, so nothing overlaps.

    step_time ≈ T_sync + O_env+O_rol+R*(T_env + T_rol) + T_act+O_act

Every component is on the critical path; any shrink helps. Pick the
dominant term:

    dominant_component = argmax(R*T_env, R*T_rol, T_act)

**Timeline signature.** Env / rollout / actor bars never overlap in
wall time on any rank. Per-step lane sequence:
`sync_model_to_rollout → { env_interact_step, predict } × R → actor forward/backward`.

## 3. Hybrid

Canonical embodied hybrid: `actor: 0-7`, `env: 0-3`, `rollout: 4-7`.
Env and rollout run on disjoint GPU subsets and execute concurrently as
producer/consumer through the env-channel and rollout-channel. Actor
runs on all 8 GPUs but only after the R chunks finish (see
`recv_rollout_trajectories.wait()`, `embodied_runner.py:518`).

For each of the R interact chunks, steady-state per-chunk time is
approximately `max(T_env, T_rol)`.

Both runner modes share the same interact-side critical path here.
`run_pipeline` additionally overlaps actor with the interact loop of
the SAME step, but under hybrid the actor uses all 8 GPUs and contends
with env/rollout — the pipeline gain is small and often negative.

    step_time ≈ T_sync + max(O_env+O_rol)+R*max(T_env, T_rol) + T_act    (both runner modes)

**Consequences.**

- If `T_env > T_rol` and rollout GPUs sit idle waiting on env, shrinking
  `T_rol` does **not** help — env is the bottleneck. Cite `env/interact`
  in the MetricTable and `env` rank `stall_fraction ≈ 0` (env is busy)
  alongside a growing `predict.stall_fraction` on rollout ranks.
  Remedies: change `env.train.total_num_envs` only when
  `step_time / num_trajectories` should improve, or reallocate GPUs
  from rollout to env.
- Symmetric reasoning if `T_rol > T_env`.
- If `T_act` dominates, `actor.micro_batch_size` up (memory permitting)
  — subject to the non-monotonic caveat in `06-playbook.md §5`.

**Timeline signature.** On env ranks, `env_interact_step` bars overlap
in wall time with `predict` bars on rollout ranks. Actor bars appear
only after the R-th chunk completes on both subsets.

## 4. Disaggregated

Example: `actor: 0-3`, `env: 4-5`, `rollout: 6-7`. All three GPU sets
are disjoint, so env, rollout, and actor **can** run concurrently.
Whether they actually do depends on the runner mode.

### 4.1 `run` mode (default, `use_training_pipeline=false`)

Actor still gates on `recv_rollout_trajectories.wait()`, so the
critical path is identical to hybrid — the disjoint actor GPUs simply
sit idle during the interact loop:

    step_time ≈ T_sync + max(O_env+O_rol)+R*max(T_env, T_rol) + T_act

**Consequence.** Under `run`, disaggregated wastes the actor's GPUs
during the interact loop. Prefer hybrid (actor collocated on all GPUs)
unless a memory reason forces dedicated actor GPUs. See
`08-gotchas.md §3` for the anti-pattern.

### 4.2 `run_pipeline` mode (`use_training_pipeline=true`)

Actor training of step N overlaps the interact loop of step N. Steady
state:

    step_time ≈ T_sync + max(O_env+O_rol) + max(R*T_env, R*T_rol, T_act+O_act)

Add one `R*max(T_env, T_rol)` for pipeline drain at the very last step
of the run — negligible over long runs. This is the pattern where
disaggregation actually pays off.

**Timeline signature (pipeline mode).** Env, rollout, and actor tag
lanes all show overlapping bars in the same wall-time window.
`stall_fraction` is the primary signal: whichever component has
`stall_fraction ≈ 0` is the bottleneck; the other two show
`stall_fraction > 0`.
