# 07 Constraints and hard rules

Every rule that will refuse or crash a delta. Read this before
proposing so you do not waste retries on constraint violations.

Two tiers:

- **Tier 1 — Preflight-enforced** (§1): mechanically checked in
  `toolkits/embodied_tuner/preflight.py`. Violations produce a
  synthetic `(FAILED, CONFIG_INVALID)` ledger entry and the critic
  retries with the reason as feedback.
- **Tier 2 — Runtime-enforced** (§2): checked by RLinf's own
  `validate_cfg` / `validate_embodied_cfg` (`rlinf/config.py`) or the
  placement runtime (`rlinf/utils/placement.py`). Preflight does NOT
  re-implement all of these, so the critic can propose a delta that
  preflight accepts but that crashes the trial. Cite this section in
  the rationale to avoid it.

The `rlinf/config.py` checks below are cited by the assertion
expression and its enclosing function (`validate_embodied_cfg`), which
are stable and greppable; precise line numbers drift with every edit
to the source and are omitted deliberately. `rlinf/utils/placement.py`
citations retain line ranges as of the current HEAD.

Per-delta checklists live in §4.

## 1. Tier 1 — Preflight-enforced (fail fast, no trial launched)

### 1.1 Knob schema (`schema.py`)

- Only knobs listed in `KnobSchema` may appear in `delta`. Others
  raise `UnknownKnobError`.
- Pinned knobs always raise `KnobNotTunableError` regardless of value.
  See `06-playbook.md §9`.
- Integer knobs are bounded (`schema.py:_default_domains`):
  - `env.train.total_num_envs ∈ [1, 4096]`
  - `env.train.rollout_epoch ∈ [1, 16]`
  - `actor.micro_batch_size ∈ [1, 4096]`
- Boolean knobs must be JSON `true` / `false`; the schema rejects
  strings or ints for booleans.

### 1.2 Placement structure (`placement_enum.is_legal_placement`)

- Every component GPU range must be **contiguous**. `[0, 1, 3]` is
  rejected. RLinf enforces this at
  `rlinf/utils/placement.py:138-143`.
- All ids must fall within `[0, num_gpus)` where `num_gpus=8` for the
  current single-node target.
- **Env vs rollout must be either equal or fully disjoint.** Partial
  overlap is rejected (e.g. env=`0-3`, rollout=`2-5` — reason: no
  valid embodied semantics).
- No component may have an empty range.

### 1.3 Divisibility (mirrors `rlinf/config.py`)

Preflight computes `env_world_size` = size of env GPU range,
`actor_world_size` = size of actor GPU range from the composed
placement.

- `env.train.total_num_envs % env_world_size == 0`.
- `(env.train.total_num_envs // env_world_size) %
  rollout.pipeline_stage_num == 0`. (`pipeline_stage_num`
  is pinned but the check still applies to the current baseline
  value.)
- `env.train.max_steps_per_rollout_epoch % actor.model.num_action_chunks == 0`.
- `actor.global_batch_size % (actor.micro_batch_size * actor_world_size) == 0`
  (FSDP branch). Since `global_batch_size` is
  pinned, moving `micro_batch_size` must land on a divisor of
  `global_batch_size / actor_world_size`.

## 2. Tier 2 — Runtime-enforced (crashes the trial if violated)

Preflight does not (yet) mirror these. If the critic's delta touches a
knob or placement affected by them, cite this section in the rationale
so the delta stays inside the safe envelope.

### 2.1 Additional divisibility (`validate_embodied_cfg`)

- `(env.train.total_num_envs // env_world_size // pipeline_stage_num) > 0`.
  Together with §1.3, per-rank per-stage env count must
  be a positive integer. Trivially violated if `total_num_envs` is
  smaller than `env_world_size * pipeline_stage_num`.
- `(env.train.total_num_envs // env_world_size // pipeline_stage_num) %
  env.train.group_size == 0`. `env.train.group_size`
  is not currently a tunable knob; the divisibility depends on the
  baseline value. When shrinking `total_num_envs` under memory
  pressure, walk down to divisors of `env_world_size *
  pipeline_stage_num * group_size`.
- `env.eval.total_num_envs` carries the same four assertions when
  `runner.val_check_interval > 0`. Deltas that touch
  training-side knobs must not accidentally invalidate the eval-side
  divisibility.

### 2.2 Weight-sync interval (`validate_embodied_cfg`)

- `runner.weight_sync_interval > 0`. Not a tunable knob;
  do not attempt to set it to `0`.

### 2.3 Overlap-env-bootstrap gate (`validate_embodied_cfg`)

- `runner.overlap_env_bootstrap` is silently forced to `False` when
  `env.train.enable_offload=true`. Consequence:
  turning on env offload also disables the actor↔env-bootstrap overlap
  optimisation. Under hybrid this can widen the actor's serial gap.
  Do not enable env offload as a throughput knob.

### 2.4 Placement runtime rules (`ModelParallelComponentPlacement`)

- All the contiguity checks preflight already runs (see §1.2) — cited
  at `rlinf/utils/placement.py:138-143`.
- **TP size ≤ world size** for actor and rollout — lines 211-215:
  - `actor.model.tensor_model_parallel_size ≤ actor_world_size`
  - `rollout.model.tensor_model_parallel_size ≤ rollout_world_size`
  Neither TP knob is currently tunable, but shrinking a component's
  GPU range below its TP size will crash Ray init.
- `padded_vocab_size % actor_tp_size == 0` (Megatron branch). Not
  applicable to the FSDP-based embodied baselines.

### 2.5 Placement mode discovery (`ModelParallelComponentPlacement`)

RLinf classifies every legal placement as EXACTLY one of collocated /
disaggregated (with a hybrid third path only for embodied
`HybridComponentPlacement`). A placement that is neither raises
`ValueError` at `rlinf/utils/placement.py:204`. `is_legal_placement`
already screens for the partial-overlap case; other unclassifiable
patterns should not be reachable given the contiguity +
full-equal-or-full-disjoint rules.

### 2.6 Routing divisibility (`CommMapper.get_dst_ranks`)

When env workers send bootstrap / trajectory data to rollout workers,
the routing layer (`rlinf/scheduler/worker/routing.py:139`) asserts:

- `(env.train.total_num_envs // env_world_size) % rollout_world_size == 0`

i.e. the per-env-rank batch size must be divisible by the rollout
world size. This is **not** mirrored by `validate_embodied_cfg`; the
tuner mirrors it as a synthetic `DIVISIBILITY_VIOLATION` inside
preflight so the critic gets a fast-fail retry, but a lucky bypass of
preflight still hits this assertion at runtime.

Concrete example that crashed a campaign trial: with
`total_num_envs=128`, `env_world_size=2`, `rollout_world_size=6`:
per-rank batch = 128/2 = 64, and 64 % 6 ≠ 0 →
`AssertionError: batch_size (64) must be divisible by dst_world_size
(6)` → Ray kills the rollout and actor groups → trial exits with
returncode 255.

The same assertion also applies in the reverse direction
(rollout→actor) with `src_world_size` = rollout world size and
`dst_world_size` = actor world size, so the full safe envelope is:

- `(total_num_envs // env_world_size) % rollout_world_size == 0`
- `(total_num_envs // env_world_size) % actor_world_size == 0`

When rebalancing GPUs between env and rollout, the critic must verify
that the new `rollout_world_size` divides `total_num_envs //
env_world_size`. For `total_num_envs=128` the legal rollout world
sizes are divisors of 128 (1, 2, 4, 8, …); 6 is not legal. If a
non-divisor rollout count is desired, `total_num_envs` must be
adjusted simultaneously so that the quotient is divisible.

## 3. Placement dictionary shape

Preflight reads `cluster.component_placement` as an OmegaConf DictConfig
with keys `actor`, `env`, `rollout`. Each value is a range string
(`"0-7"`, `"0"`, or `"all"`). The delta form the critic emits should
be a JSON object with exactly the same keys, e.g.

    {"actor": "0-7", "env": "0-3", "rollout": "4-7"}

Do **not** emit list-of-dicts placements or add extra components — the
preflight parser rejects unknown types
(`preflight._placement_to_str_map`).

## 4. Delta-type checklists

Use the checklist matching every knob in the proposed delta.
Substitute the **new** values after applying the delta, not the
baseline values.

### 4.1 If delta touches `cluster.component_placement`

1. Parse `actor_world_size`, `env_world_size`, and `rollout_world_size`
   from the new ranges.
2. Verify every range is contiguous, in `[0, num_gpus)`, non-empty,
   and that env and rollout are either equal or fully disjoint.
3. Verify `env.train.total_num_envs % env_world_size == 0`.
4. Verify `(env.train.total_num_envs // env_world_size) %
   rollout.pipeline_stage_num == 0`.
5. Verify `(env.train.total_num_envs // env_world_size //
   rollout.pipeline_stage_num) % env.train.group_size == 0`.
6. Verify `actor.global_batch_size %
   (actor.micro_batch_size * actor_world_size) == 0`.
7. Verify actor and rollout world sizes are at least their tensor
   parallel sizes.
8. If env or rollout GPU ranges change, verify routing divisibility
   (§2.6): `(total_num_envs // env_world_size) % rollout_world_size == 0`
   and `(total_num_envs // env_world_size) % actor_world_size == 0`.

### 4.2 If delta touches `env.train.total_num_envs`

1. Verify it is in `[1, 4096]`.
2. Verify `total_num_envs % env_world_size == 0`.
3. Verify `(total_num_envs // env_world_size) %
   rollout.pipeline_stage_num == 0`.
4. Verify `(total_num_envs // env_world_size //
   rollout.pipeline_stage_num) > 0`.
5. Verify `(total_num_envs // env_world_size //
   rollout.pipeline_stage_num) % env.train.group_size == 0`.
6. If placement is hybrid or disaggregated, verify routing
   divisibility (§2.6): `(total_num_envs // env_world_size) %
   rollout_world_size == 0` and `(total_num_envs // env_world_size) %
   actor_world_size == 0`.

### 4.3 If delta touches `actor.micro_batch_size` or actor GPU count

1. Verify `actor.micro_batch_size` is in `[1, 4096]`.
2. Verify `actor.global_batch_size %
   (actor.micro_batch_size * actor_world_size) == 0`.
3. If the actor GPU range shrinks, verify
   `actor.model.tensor_model_parallel_size <= actor_world_size`.

### 4.4 If delta touches `env.train.rollout_epoch`

1. Verify `env.train.rollout_epoch` is in `[1, 16]`.
2. Remember the optimisation objective is `step_time /
   num_trajectories`; changing rollout_epoch changes both numerator
   and denominator.

### 4.5 If delta enables any `*.enable_offload`

1. Use offload as a memory rescue knob, not a throughput knob.
2. If enabling `env.train.enable_offload`, note that
   `runner.overlap_env_bootstrap` is forced to `False` (§2.3).
3. Do not enable offload on the current critical-path component
   unless the last trial failed or nearly failed from memory pressure.

Missing citations to constraints do not cause validation to fail on
their own, but every crash caused by a constraint the critic could
have foreseen slows the campaign — treat this file as required reading
before proposing.
