# 05 Decision recipe

The step-by-step for turning the prompt into a delta whose expected
effect on `step_time / num_trajectories` is non-zero. Each step names
which prompt block(s) to consult and cross-refs the wiki file that
owns the underlying rule.

## Strategy — early rounds: profile before you exploit

When the campaign is young — short Trial History, the active leaf still
at or near the root in the `## Search DAG` — spend the first few rounds
*characterizing* the knob landscape rather than greedily chasing the
first bottleneck you see. Each such round applies a single-knob delta to
a different tunable knob and reads back two things from the resulting
trial:

- **Cost:** how the knob moved wall time and GPU memory — the
  per-component cost blocks (`03-inputs.md §9.1`, `§9.2`) and the GPU
  memory summary (`§7.1`).
- **Payoff:** how it moved the objective `step_time / num_trajectories`
  (`01-concepts.md §1`).

The goal is an empirical cost/sensitivity map — which knobs actually
shift the objective, in which direction, and at what memory cost — so
later rounds can exploit the highest-payoff knob backed by evidence
instead of a guess. This is the exploration phase; the
bottleneck-driven exploitation in §§2-4 is what you switch to once the
map exists.

Guardrails while probing:

- Every probe is still **one knob per delta** (§7) and must pass the
  constraint checklist (§5).
- Prefer probing knobs that plausibly touch the current critical-path
  component first (§2) — a probe on an off-path knob teaches little.
- Do not re-probe a knob at a value already in the Trial History / DAG
  (§7, `09-dag-search.md`); vary the value so each probe adds new
  evidence.
- Stop exploring and commit to exploitation once the dominant term's
  sensitivity is clear, or the remaining round budget is short.

## 1. Identify placement and runner mode

Read `cluster.component_placement` and `runner.use_training_pipeline`
from the current-knobs block (`03-inputs.md §5`). Classify placement
as one of collocated / hybrid / disaggregated per
`01-concepts.md §4`, and runner mode per `01-concepts.md §5`.

Runner mode is fixed by the baseline and is NOT a tunable knob in this
loop.

## 2. Locate the bottleneck

Read the timeline verbose block (`03-inputs.md §9`) in the priority
order from `03-inputs.md §10`:

1. **Critical path per global_step** (`§9.4`). Each lane row separates
   `real_s` (actual GPU work) from `blocked_s` (waiting on another
   component). **The bottleneck is the lane with the largest `real_s`.**
   A lane with big `blocked_s` and small `real_s` is a downstream
   consumer, not a bottleneck; do NOT propose to shrink it.

    Common trap: `actor/recv_traj` / `recv_rollout_trajectories` is a
    **blocking wait** (`04-signals.md §2.1`) — its duration reflects
    interact-loop cost, not actor work. Cite `actor_forward`,
    `actor_backward`, `actor_optimizer_step`, or
    `compute_advantages_and_returns` for actor bottleneck claims. Use
    `actor/run_training` only as MetricTable sanity-check evidence when
    actor phase tags are missing; never cite recv tags as actor
    compute.
2. **Per-component bubble** (`§9.5`). Largest `bubble_frac` = most
   idle wall-clock; usually the side whose GPU budget can be reduced.
3. **Component call averages** (`§9.1`). Typical per-call cost after
   the first 2 warmup calls are dropped.
4. **MetricTable Time-section keys** (`§8.1`). Sanity-check against
   the timeline's per-step attribution within ~10%. Divergence means
   one of the tags is wrapper-noisy — trust the timeline `real_s`.
5. **Outlier events** (`§9.6`). Rows carry `knob_hint` when the parser
   can map the stall to a knob. Use as corroboration for the delta
   chosen in step 1 — not as a substitute for reading the critical
   path.
6. **Raw excerpts / JSONL** (`§9.7`). Verbatim events. Cite one of
   these whenever the delta touches `cluster.component_placement`
   (dual-source rule — see §6).

## 3. Confirm the delta will move the critical-path term

Cross-check the current placement's formula from `02-paths.md §1`:

- **`run` + hybrid**: shrinking the smaller of `T_env` / `T_rol` buys
  zero — the term is `max(T_env, T_rol)`.
- **`run` + disaggregated**: same as hybrid; also, the actor GPUs are
  idle during interact, so proposing to grow actor's GPU share cannot
  reduce `T_act`'s scheduling slot, only its compute cost.
- **`run_pipeline` + disaggregated**: shrinking a non-max term buys
  zero.
- **Collocated**: every term contributes additively; any shrink helps.
  Rank by magnitude and pick the largest.

If the intended term is not on the critical path, either pick a
different term or a different knob.

## 4. Choose the knob

Consult `06-playbook.md` for knob-by-knob effects. First-line moves:

| Bottleneck term (largest `real_s` on critical path)                     | First-line knob                                          | Second-line knob                                                                            |
|-------------------------------------------------------------------------|----------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `env_interact_step` on env ranks                                        | reallocate env GPUs (`cluster.component_placement`)      | `env.train.total_num_envs` only if normalized `step_time / num_trajectories` should improve |
| `predict` on rollout ranks                                              | reallocate rollout GPUs (`cluster.component_placement`)  | `env.train.rollout_epoch` down only if normalized objective improves                        |
| `actor_forward`, `actor_backward`, `actor_optimizer_step` on actor ranks | `actor.micro_batch_size` up if memory allows; down only for OOM / high memory | reallocate actor GPUs / `rollout_epoch` up if normalized objective improves            |
| `actor/sync_model_to_rollout` (`T_sync`)                                | (not tunable in this loop) — investigate `weight_sync_interval` in the baseline | flag as FUT                                                                                 |
| `compute_advantages_and_returns`                                        | `env.train.rollout_epoch` down only if normalized objective improves | flag if config regression                                                                    |

Rules of thumb:

- Prefer knobs that touch a single component. Placement changes are
  the highest-variance moves — reserve them for when a component-local
  knob has plateaued.
- Optimize `step_time / num_trajectories`, not raw `step_time`. A
  delta that reduces `step_time` but reduces `num_trajectories` by the
  same or larger fraction is not an improvement.
- Under memory pressure (last trial `FailureMode=OOM`), follow the
  memory triage cascade in `06-playbook.md §8` first.
  Do NOT reach for placement in the OOM branch.

## 5. Validate against constraints

Before writing the delta, walk the checklist in `07-constraints.md §4`
for every knob the delta touches. Substitute the **new** values (after
applying the delta), not the baseline values.

If any check fails, either pick a divisor-safe value or drop the
delta.

## 6. Cite the evidence

The rationale schema (see the `## Required output JSON shape` block of
the prompt) requires:

- Always: `rationale.summary` — one paragraph naming the specific term
  you are moving and where you read it.
- If the delta contains `cluster.component_placement`
  ("dual-source rule"):
  - `rationale.metric_table_citations` must include at least one
    `key=value` line copied from the MetricTable Time-section block
    (e.g. `env/interact=275.4`, `actor/run_training=210.1`).
  - `rationale.timeline_citations` must include at least one
    observation copied from the per-component bubble, the critical
    path block, or the raw excerpts block, as free-form text
    (e.g. `env rank0 env_interact_step median=15.2s stall_fraction=0.40`).

The validator enforces both citation arrays are non-empty for
placement deltas (`critic.py:CriticOutputValidator.validate`).

## 7. One knob per delta (with a failed-trial exception)

**Default: one knob per delta.** Two knobs per delta only when they
are logically inseparable — e.g. adjusting `cluster.component_placement`
together with `env.train.total_num_envs`, because the new
`env_world_size` / `rollout_world_size` would violate the divisibility
constraints (`07-constraints.md §1.3, §2.6`) unless `total_num_envs`
moves at the same time. Multi-knob deltas make the ledger harder to
interpret and slow convergence.

**Do not propose a knob that has already been tried at the same value
in the recent trial history** without new evidence to justify it. The
history block is deduplicated for exactly this check.

**Failed-trial revert bundle.** When the last trial failed
(`FailureMode != NONE` — OOM / crash / timeout / config invalid /
divisibility), the one-knob-per-delta rule is relaxed: bundle the
rollback and the next move into a single delta. First revert the
failed knob to its previous known-good value, then apply the next
adjustment on top of that baseline in the same delta. This avoids
spending a whole trial re-establishing the prior baseline before
making forward progress. The ledger entry must name both moves
(revert + new) in `rationale.summary` so cause and effect stay
attributable.

## 8. Emit a bitter lesson when a failure requires it

When `last_failure_mode` is one of
`OOM, WORKER_CRASH, TIMEOUT, CONFIG_INVALID, DIVISIBILITY_VIOLATION`,
the response MUST include a non-empty `bitter_lesson.trigger` AND
`bitter_lesson.rule`. The scheduler persists it under
`<ledger_dir>/bitter_lessons.jsonl` and prepends every future prompt
with the accumulated lessons so the same failing delta is not
re-proposed (see `03-inputs.md §3`).

Omit the field on a successful follow-up unless the trial revealed a
durable constraint worth persisting.
