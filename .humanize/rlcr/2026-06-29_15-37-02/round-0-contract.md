# Round 0 Contract

## Mainline Objective

Establish Milestone A — Foundations of `RLinf/toolkits/embodied_tuner/`: the four non-GPU modules and their unit tests that every later milestone depends on.

## Target ACs

- AC-1 (knob schema)
- AC-4 (placement enumerator)
- AC-2 (Hydra-override wrapper)
- AC-3 (preflight validator)

AC-5 through AC-11 are explicitly **out of scope** for this round and remain `[queued]` in the Goal Tracker.

## Mainline Tasks (in dependency order)

1. **task1 — `schema.py`** (no deps): `KnobSchema` dataclass enumerating the 7 tunable knobs (placement, total_num_envs, rollout_epoch, micro_batch_size, env/rollout/actor enable_offload) with declared domains; the 3 pinned knobs reserved with `pinned=True`. Public surface: `KnobSchema.validate(delta)`, `KnobSchema.list_knobs()`, exception types `KnobOutOfRangeError`, `KnobNotTunableError`, `UnknownKnobError`.
2. **task4 — `placement_enum.py`** (deps: task1): Enumerate legal placement strings for a fixed-size 8-GPU node respecting `ModelParallelComponentPlacement`'s contiguity and non-overlap rules. Covers collocated, disaggregated, hybrid, and "all" patterns. Pure parsing/legality logic — does not instantiate `Cluster`. Public surface: `PlacementSpec` dataclass, `enumerate_placements(num_gpus=8)`, `parse_range_spec(s)`, `is_legal_placement(spec)`.
3. **task2 — `override_wrapper.py`** (deps: task1): Compose a launcher that invokes `examples/embodiment/run_embodiment.sh`-equivalent semantics with arbitrary Hydra overrides forwarded. Because the stock script only injects `runner.logger.log_path=${LOG_DIR}`, the wrapper writes a sibling shell shim that re-uses the stock script's env-var setup and appends user overrides. Returns the exact `LOG_DIR` actually used. Public surface: `OverrideWrapper.build_invocation(config_name, overrides, log_dir_root)`, `LaunchSpec` dataclass.
4. **task3 — `preflight.py`** (deps: task1, task4): Compose baseline + delta via Hydra, run `rlinf.config.validate_cfg` + `validate_embodied_cfg`, run a local placement-legality check (using `placement_enum.is_legal_placement` rather than instantiating a full `Cluster`, to keep "no GPU work" honest). Returns `(resolved_cfg, sha256, ValidationResult)`. Public surface: `compose_and_validate(baseline_path, delta, hydra_overrides=())`, `ValidationResult` dataclass.

## Blocking Side Issues in Scope

None known at start.

## Queued Side Issues Out of Scope

- AC-5..AC-11 modules (runner, parser, critic, scheduler, ledger, CLI, smoke test).
- `validate_embodied_cfg` instantiates `Cluster()` at `rlinf/config.py:924` to derive env_world_size. This MAY require a running Ray cluster or careful mock during preflight. Plan: in task3, first try `Cluster()` inside an `init_local_cluster` context (existing `rlinf.scheduler.cluster.Cluster` may auto-init Ray locally); if that requires real GPUs, fall back to a `LocalClusterShim` that returns a precomputed world-size dict. Resolution must happen INSIDE task3 because preflight cannot ship without working `validate_embodied_cfg`.
- Pip-installability of the new toolkit (`pyproject.toml` excludes `toolkits/`).

## Round Success Criteria

- Every Round-0 mainline task is implemented with tests under `RLinf/toolkits/embodied_tuner/tests/`.
- `pytest RLinf/toolkits/embodied_tuner/tests/` (or equivalent invocation with `PYTHONPATH=RLinf/`) passes for all task1/task2/task3/task4 tests.
- `KnobSchema` exposes the exact knob set from the plan; pinned-knob rejection is tested.
- `enumerate_placements(num_gpus=8)` returns ≥1 collocated, ≥1 disaggregated, ≥1 hybrid, and ≥1 "all" entry, each parseable by the local-legality check; tests for the negative cases (overlap, malformed, non-contiguous) pass.
- `OverrideWrapper.build_invocation(...)` returns a `LaunchSpec` carrying (a) the exact command to execute the stock entry script with user overrides appended after `runner.logger.log_path=${LOG_DIR}`, (b) the `LOG_DIR` to be created, and (c) override precedence is verified by a test.
- `preflight.compose_and_validate(...)` composes a known-legal delta on the real baseline (`maniskill_ppo_openvla.yaml`) and returns `ok=True`; rejects the `actor.micro_batch_size`-violates-divisibility delta with a structured reason; rejects the `total_num_envs % env_world_size != 0` delta.
- Goal Tracker updated to reflect Round 0 task statuses; `round-0-summary.md` written with `## BitLesson Delta` (Action: none).
- All changes committed locally with a Conventional-Commit-style message.

## Out-of-Scope (Explicit)

- No real RLinf training invocation. Preflight tests use Hydra composition only.
- No Codex calls during Round 0.
- No multi-node placement (FUT-1).
- No un-pinning of `actor.global_batch_size`, `rollout.pipeline_stage_num`, `actor.num_action_chunks` (FUT-5).
