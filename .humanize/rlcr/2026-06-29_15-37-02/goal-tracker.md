# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Build a Python orchestrator under `RLinf/toolkits/embodied_tuner/` (importable as `toolkits.embodied_tuner` via the existing `RLinf/toolkits/__init__.py`) that, given a baseline embodied RLinf config (default: `RLinf/examples/embodiment/config/maniskill_ppo_openvla.yaml`), iteratively proposes config deltas, validates them mechanically, runs RLinf trials, parses logs and `timeline/*.jsonl`, and converges on a configuration that minimizes `step_time / num_trajectories` subject to memory and feasibility constraints (no OOM, no worker crash, no validate_cfg violation).

The tunable knob set is `cluster.component_placement` (env/actor/rollout/all GPU range strings), `env.train.total_num_envs`, `env.train.rollout_epoch`, `actor.micro_batch_size`, and the three `enable_offload` flags (env/rollout/actor). `actor.global_batch_size`, `rollout.pipeline_stage_num`, and `actor.num_action_chunks` are pinned in this current loop to keep `validate_cfg` divisibility ripples tractable; loosening them is `FUT-5`.

The humanize + RLCR pipeline is used ONCE to build this tuner. The tuner's per-trial loop is plain Python — it is NOT itself an RLCR loop, and one RLCR round does NOT correspond to one RLinf trial. The existing `RLinf/toolkits/auto_placement/` package is NOT imported or extended.

Every placement-optimization decision made by the LLM critic during the tuning loop MUST be grounded in BOTH coarse evidence from `metrics.log` (MetricTable per-component aggregates) AND fine-grained evidence parsed from `timeline/*.jsonl` (per-rank min/median/max timings, stall fractions, call counts). A critic-output validator rejects any proposed delta that touches `cluster.component_placement` without citing at least one observation from each source.

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->

- AC-1: Knob schema + tests (`schema.py`). Pinned-knob markers reserved for FUT-5.
- AC-2: Override wrapper around `run_embodiment.sh` + tests (Hydra precedence; LOG_DIR capture).
- AC-3: Preflight validator (Hydra compose + `validate_cfg` + `validate_embodied_cfg` + placement legality) + tests, no GPU launch.
- AC-4: Placement enumerator respecting `ModelParallelComponentPlacement` contiguity + non-overlap + tests.
- AC-5: Trial runner with timeout + scoped cleanup + default-on profiler env exports + tests.
- AC-6: Log/timeline parser with `(Status, FailureMode)` taxonomy, OOM rubric, objective computation, best-config selection, per-component timeline summary + tests.
- AC-7: LLM critic prompt builder + structured-rationale schema + dual-source rationale validator + `fake_critic` + tests.
- AC-8: Scheduler with budget + plateau + critic-stagnation stop + tests.
- AC-9: Append-only JSONL ledger with SHA-256 resolved-config hashing + structured critic_rationale persistence + tests.
- AC-10: CLI entrypoint + shim launcher + tests.
- AC-11: End-to-end smoke test using `fake_critic` + mock runner + bundled import-boundary AST walker sub-test.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | gen-plan + refine-plan output | - |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| [mainline] task1: `schema.py` (KnobSchema dataclass + pinned-knob markers + tests) | AC-1 | completed (pending verification) | coding | claude | 21 unit tests pass |
| [mainline] task4: `placement_enum.py` (legal placement enumeration for 8 GPUs + tests) | AC-4 | completed (pending verification) | coding | claude | 25 unit tests pass |
| [mainline] task2: `override_wrapper.py` (Hydra-override shim + LOG_DIR capture + tests) | AC-2 | completed (pending verification) | coding | claude | 16 unit tests pass; Hydra precedence verified |
| [mainline] task3: `preflight.py` (Hydra compose + targeted divisibility + placement legality, no GPU) + tests | AC-3 | completed (pending verification) | coding | claude | 14 unit tests pass; no Ray started |
| [queued] task5: `runner.py` (trial runner with timeout + scoped cleanup + profiler env exports) | AC-5 | pending | coding | claude | Round 1+ — Milestone B |
| [queued] task6: `parser.py` (log/timeline parser + objective + best-config + timeline summary) | AC-6 | pending | coding | claude | Round 1+ — Milestone B |
| [queued] task7: `critic.py` (prompt + rationale schema + dual-source validator + fake_critic) | AC-7 | pending | coding | claude | Round 1+ — Milestone C |
| [queued] task8: `scheduler.py` (loop + budget + plateau + critic-stagnation) | AC-8 | pending | coding | claude | Round 1+ — Milestone C |
| [queued] task9: `ledger.py` (append-only JSONL + SHA-256 + critic_rationale persistence) | AC-9 | pending | coding | claude | Round 1+ — Milestone C |
| [queued] task10: `__main__.py` (CLI + shim launcher) | AC-10 | pending | coding | claude | Round 1+ — Milestone D |
| [queued] task11: end-to-end smoke test + bundled import-boundary AST walker | AC-11 | pending | coding | claude | Round 1+ — Milestone D |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| `validate_embodied_cfg` instantiates `Cluster()` at `rlinf/config.py:922`, which calls `ray.init` (`rlinf/scheduler/cluster/cluster.py:332`) — preflight cannot call it directly. **Resolved in Round 0** by re-implementing the targeted divisibility checks locally (mirrors `rlinf/config.py:962/965/980/1363-1368`); resolved. | 0 | Resolved in Round 0 | Codex review may surface deeper RLinf rules not covered by our local checks |
| Embodied baselines reference `${oc.env:EMBODIED_PATH}` in `hydra.searchpath`. **Resolved in Round 0** by setting `EMBODIED_PATH`/`REPO_PATH` in `_compose_cfg` before invoking Hydra compose. | 0 | Resolved in Round 0 | None |
| `RLinf/toolkits/` is excluded from `[tool.setuptools.packages.find]` in `pyproject.toml`, so the new toolkit is not pip-installable. Tests still run via `pytest` from repo root with `PYTHONPATH=RLinf/` set. | 0 | Matches existing convention | Revisit only if pip-install becomes desired |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
