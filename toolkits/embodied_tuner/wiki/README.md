# Tuner wiki — ordered optimization context for the critic

This directory holds natural-language context the LLM critic reads
verbatim at the top of every prompt. Files are ordered so that
concepts precede data-block schemas, data precedes decisions, and
decisions precede per-knob actions.

## Read order

The critic reads the numbered files below as one wiki block, in
ascending order. Earlier files do not assume later ones.

1. [`01-concepts.md`](01-concepts.md) — objective, notation
   (`T_env / T_rol / T_act / T_sync / R`), placement modes, runner
   modes, and the trajectory-scaling model. Every later file assumes
   these definitions.
2. [`02-paths.md`](02-paths.md) — critical-path formula per
   `(placement, runner mode)` combination, with quick-reference table
   and timeline signatures.
3. [`03-inputs.md`](03-inputs.md) — the data blocks the critic
   actually receives in the prompt, their schemas, and which block is
   authoritative for what. Includes the priority rule when blocks
   disagree and the `FailureMode` catalog.
4. [`04-signals.md`](04-signals.md) — timeline tag reference,
   `stall_fraction` semantics, wrapper tags to ignore, and
   missing-data handling.
5. [`05-recipe.md`](05-recipe.md) — the step-by-step decision flow:
   identify placement → locate bottleneck → verify shrinkable →
   choose knob → validate → cite. Includes the dual-source rule,
   one-knob-per-delta rule, and failed-trial revert bundle.
6. [`06-playbook.md`](06-playbook.md) — per-knob playbook: what each
   knob moves, when to grow/shrink, per-knob preflight checks, and
   cross-knob patterns (memory triage cascade, hybrid rebalance,
   actor-bound recovery).
7. [`07-constraints.md`](07-constraints.md) — Tier 1 preflight and
   Tier 2 runtime rules, plus per-delta-type checklists. §2.6
   documents the routing-divisibility trap that produces synthetic
   `DIVISIBILITY_VIOLATION` failures.
8. [`08-gotchas.md`](08-gotchas.md) — consolidated anti-patterns
   with cross-references to the rule detail.
9. [`09-dag-search.md`](09-dag-search.md) — the `## Search DAG` block:
   active branch, sibling attempts, top-K OK leaderboard, recent
   failure / duplicate leaves, and how to expand from the active leaf.

## Prompt contract

Every file is intended to be consumed **verbatim** by
`critic.build_prompt`. Keep them short, declarative, and free of
speculation — the critic treats these lines as ground truth.

The list of files loaded and their order are pinned in
`critic._WIKI_CONTEXT_FILES` (`toolkits/embodied_tuner/critic.py`).
Adding, removing, or renaming a file requires updating that tuple.

If you want the critic to weight a hypothesis differently, edit the
wiki before adjusting `_BOTTLENECK_RUBRIC` (if any) or the prompt
builders in `critic.py`. This keeps runtime code general and defers
workload-specific priors here.

## Externally referenced anchor

Runtime error messages and tests reference
`07-constraints.md §2.6` (routing divisibility). Preserve the anchor,
or update every citation site:
- `critic.py:_render_constraints`
- `preflight.py` (multiple sites in `_check_routing_divisibility`)
- `parser.py` `_infer_failure_mode` neighborhood
- `tests/test_preflight.py`, `tests/test_parser.py`
