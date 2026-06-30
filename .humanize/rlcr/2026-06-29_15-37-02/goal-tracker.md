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

### Plan Version: 1 (Updated: Round 3)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | gen-plan + refine-plan output | - |
| 1 | None — pure implementation of queued tasks | - | - |
| 2 | None — pure implementation of queued tasks | - | - |
| 3 | Scheduler: added `preflight_exhausted` stop reason + synthetic CONFIG_INVALID ledger; preflight feedback now reaches the next critic prompt; OOM/crash classification scans `run_embodiment.log` by default and runs before METRICS_MISSING; citation arrays must be `list[str]`; orphan scan reads `/proc/<pid>/environ` not just `pgrep -f`. | Codex Round-2 review (READY_FOR_ROUND_3: yes-with-caveats). | Strengthens AC-7 (citation type validation), AC-6 (OOM rubric), AC-5 (orphan scan), AC-8 (preflight exhaustion). No new ACs. |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| [mainline] task1: `schema.py` (KnobSchema dataclass + pinned-knob markers + tests) | AC-1 | completed (pending verification) | coding | claude | Round 0, 21 unit tests |
| [mainline] task4: `placement_enum.py` (legal placement enumeration for 8 GPUs + tests) | AC-4 | completed (pending verification) | coding | claude | Round 0, 25 unit tests |
| [mainline] task2: `override_wrapper.py` (Hydra-override shim + LOG_DIR capture + tests) | AC-2 | completed (pending verification) | coding | claude | Round 0, 16 unit tests |
| [mainline] task3: `preflight.py` (Hydra compose + targeted divisibility + placement legality, no GPU) + tests | AC-3 | completed (pending verification) | coding | claude | Round 0, 14 unit tests |
| [mainline] task5: `runner.py` (trial runner: timeout, SIGTERM→SIGKILL, ray-stop hook, scoped orphan cleanup via `/proc/<pid>/environ` + `pgrep -f`, profiler env exports) + tests | AC-5 | completed (pending verification) | coding | claude | Round 1, 15 unit tests; Round 3 strengthened orphan cleanup to scan `/proc/<pid>/environ` (env vars don't appear in `pgrep -f`) |
| [mainline] task6: `parser.py` (metrics.log + timeline JSONL parser, (Status, FailureMode), objective + best-config, timeline summary, OOM-before-METRICS_MISSING precedence) + tests | AC-6 | completed (pending verification) | coding | claude | Round 1, 24 unit tests; Round 3 hardened OOM/crash classification to scan `run_embodiment.log` by default and run BEFORE METRICS_MISSING when returncode != 0 |
| [mainline] task9: `ledger.py` (append-only JSONL + SHA-256 + critic_rationale persistence) + tests | AC-9 | completed (pending verification) | coding | claude | Round 2, 13 unit tests |
| [mainline] task7: `critic.py` + `fake_critic.py` (prompt + rationale schema + dual-source validator + Codex transport + retries + preflight feedback) + tests | AC-7 | completed (pending verification) | coding | claude | Round 2, 26 unit tests + Round 3 added 4 citation-type-validation tests; preflight feedback now threaded into prompt |
| [mainline] task8: `scheduler.py` (loop + budget + plateau + critic-stagnation + preflight-exhaustion) + tests | AC-8 | completed (pending verification) | coding | claude | Round 2, 13 unit tests + Round 3 added 2 preflight tests (exhaustion stops without launching; feedback reaches critic). Replaced old "run anyway" test |
| [mainline] task10: `__main__.py` (CLI) + `examples/embodiment/run_embodied_tuner.sh` (shim) + tests | AC-10 | completed (pending verification) | coding | claude | Round 3, 12 unit tests; --dry-run-preflight composes the real baseline cleanly |
| [mainline] task11: end-to-end smoke test (FakeCritic + mock runner/parser/preflight) + bundled AST-walker import-boundary sub-test | AC-11 | completed (pending verification) | coding | claude | Round 3, 10 unit tests; smoke covers clean run, OOM+parser-crash, no-eligible-trial; AST walker catches planted forbidden imports |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| `validate_embodied_cfg` instantiates `Cluster()` (`rlinf/config.py:922`) which calls `ray.init`. **Resolved Round 0** by re-implementing targeted divisibility checks locally in `preflight.py`. | 0 | Resolved | Codex review may surface deeper RLinf rules not covered |
| Embodied baselines reference `${oc.env:EMBODIED_PATH}` in `hydra.searchpath`. **Resolved Round 0** by setting `EMBODIED_PATH`/`REPO_PATH` in `_compose_cfg` before Hydra compose. | 0 | Resolved | None |
| `RLinf/toolkits/` is excluded from `[tool.setuptools.packages.find]` — tests still run via `PYTHONPATH=.`. Matches existing convention used by `run_placement_autotune.sh`. | 0 | Matches convention | Revisit only if pip-install becomes desired |
| Timeline JSONL tag names differ from the MetricTable aggregate keys. **Resolved Round 1** by setting `_HEADLINE_TAGS` in `parser.py` to the verified event-tag set. | 1 | Resolved | None |
| `nvitop/` directory absent in current logs because `profiler/enable2.sh` leaves NVITOP/NVML flags commented. **Round 1** runner exports them by default. | 1 | Future trials populate it | When task11 smoke test verifies a real trial dir |
| Preflight-retry exhaustion: previously ran the runner anyway. **Resolved Round 3** by adding `preflight_exhausted` stop reason + synthetic CONFIG_INVALID ledger entry; no runner call. | 2 | Resolved Round 3 | None |
| Codex review of Rounds 0-2 (Round 3): non-placement memory-sensitive deltas (`enable_offload`, `total_num_envs`, `micro_batch_size`) could optionally require softer evidence. Codex marked SUGGESTION; left as future enhancement. | 3 | Suggestion; current dual-source rule covers placement only as the plan specifies | Revisit if operators want stricter evidence rule |
| Codex review (Round 3): preflight could mirror more `validate_cfg` checks (group_size divisibility, eval-side divisibility, runner.task_type, supported model_type, actor_critic value-head). Left as future enhancement; current 4 targeted checks match what the plan explicitly tests. | 3 | Current checks match plan; broader coverage is a polish item | Revisit if a real trial leaks past preflight |
| Codex review (Round 3): `ray stop --force` runs after every trial regardless of cleanup state. Suggested making it failure-only or opt-in. Left as default-on for safety. | 3 | Default-on is safer for shared hosts; opt-in flag can be added cheaply later | Revisit if shared-host operators complain |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
