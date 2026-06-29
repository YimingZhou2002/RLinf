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

### Plan Version: 1 (Updated: Round 1)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | gen-plan + refine-plan output | - |
| 1 | None — pure implementation of queued tasks | - | - |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| [mainline] task1: `schema.py` (KnobSchema dataclass + pinned-knob markers + tests) | AC-1 | completed (pending verification) | coding | claude | Round 0, 21 unit tests |
| [mainline] task4: `placement_enum.py` (legal placement enumeration for 8 GPUs + tests) | AC-4 | completed (pending verification) | coding | claude | Round 0, 25 unit tests |
| [mainline] task2: `override_wrapper.py` (Hydra-override shim + LOG_DIR capture + tests) | AC-2 | completed (pending verification) | coding | claude | Round 0, 16 unit tests |
| [mainline] task3: `preflight.py` (Hydra compose + targeted divisibility + placement legality, no GPU) + tests | AC-3 | completed (pending verification) | coding | claude | Round 0, 14 unit tests |
| [mainline] task5: `runner.py` (trial runner: timeout, SIGTERM→SIGKILL, ray-stop hook, scoped pgrep cleanup, profiler env exports) + tests | AC-5 | completed (pending verification) | coding | claude | Round 1, 15 unit tests; hermetic (no real RLinf launch) |
| [mainline] task6: `parser.py` (metrics.log + timeline JSONL parser, (Status, FailureMode), objective + best-config, timeline summary) + tests | AC-6 | completed (pending verification) | coding | claude | Round 1, 24 unit tests; validated against live `logs/20260629-07:25:33-*` |
| [queued] task7: `critic.py` (prompt + rationale schema + dual-source validator + fake_critic) | AC-7 | pending | coding | claude | Round 2+ — Milestone C |
| [queued] task8: `scheduler.py` (loop + budget + plateau + critic-stagnation) | AC-8 | pending | coding | claude | Round 2+ — Milestone C |
| [queued] task9: `ledger.py` (append-only JSONL + SHA-256 + critic_rationale persistence) | AC-9 | pending | coding | claude | Round 2+ — Milestone C |
| [queued] task10: `__main__.py` (CLI + shim launcher) | AC-10 | pending | coding | claude | Round 2+ — Milestone D |
| [queued] task11: end-to-end smoke test + bundled import-boundary AST walker | AC-11 | pending | coding | claude | Round 2+ — Milestone D |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| `validate_embodied_cfg` instantiates `Cluster()` (`rlinf/config.py:922`) which calls `ray.init` (`rlinf/scheduler/cluster/cluster.py:332`). **Resolved Round 0** by re-implementing targeted divisibility checks locally in `preflight.py`. | 0 | Resolved | Codex review may surface deeper RLinf rules not covered |
| Embodied baselines reference `${oc.env:EMBODIED_PATH}` in `hydra.searchpath`. **Resolved Round 0** by setting `EMBODIED_PATH`/`REPO_PATH` in `_compose_cfg` before Hydra compose. | 0 | Resolved | None |
| `RLinf/toolkits/` is excluded from `[tool.setuptools.packages.find]` — tests still run via `PYTHONPATH=.`. Matches existing convention used by `run_placement_autotune.sh`. | 0 | Matches convention | Revisit only if pip-install becomes desired |
| Timeline JSONL tag names differ from the MetricTable per-component aggregates (e.g. `env_interact_step` (no slash) in events vs `env/interact` in the MetricTable). **Resolved Round 1** by setting `_HEADLINE_TAGS` in `parser.py` to the verified event-tag set. | 1 | Resolved | None |
| `nvitop/` is absent from the current logs because `profiler/enable2.sh` leaves the NVITOP/NVML flags commented. **Round 1** runner now exports `RLINF_NVITOP=1`/`RLINF_NVML=1` by default; future trials launched via the tuner will populate `nvitop/`. Parser already reads `nvitop_summary.log` best-effort. | 1 | Future trials populate it; current logs don't have it | When task11 smoke test verifies a real trial dir |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
