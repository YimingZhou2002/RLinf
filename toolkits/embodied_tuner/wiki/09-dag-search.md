# DAG search view

This section documents the `## Search DAG` block that appears between
`## Bitter Lessons` and `## Trial History` in your prompt. It shows
the persistent DAG of trials the campaign has explored so far. Every
entry corresponds to a node in the on-disk `nodes.jsonl` file; node
identifiers are stable across rounds.

## What each subsection means

- **Active branch (root → active_leaf):** The ancestor chain of the
  scheduler's current active leaf, root first. This is the parent
  context your proposed delta will be applied on top of. The arrow
  `->` marks the current active leaf. Chain is rendered in full, even
  when it exceeds `max_dag_nodes`.
- **Sibling attempts at parent:** Other children of the active leaf's
  parent that have already been tried. Reviewing these tells you
  which nearby deltas have been explored and how they turned out.
- **Top-K OK leaves (lowest objective first):** The best `objective`
  (i.e. `step_time / num_trajectories`) achieved so far, sorted from
  best to worst, excluding `DUPLICATE_OF` synthetic nodes. A new
  proposal that would land inside this leaderboard's SHA space would
  short-circuit to a duplicate.
- **Recent failure leaves:** The most recent trials whose
  `failure_mode` is not `NONE`/`DUPLICATE_OF`, ordered by recency.
  Each line shows the failure mode alongside the delta that failed.
- **Recent duplicate config attempts:** The most recent synthetic
  `DUPLICATE_OF` nodes, ordered by recency. A duplicate is emitted
  whenever the resolved cumulative config of a proposal matches a
  prior clean-OK trial, so this section surfaces repeated proposals
  that would waste budget by re-running an already-known result. The
  `duplicate_of=<node_id>` field points back at the original OK trial
  whose objective the duplicate reuses.

## How to use the DAG view

1. **Do not re-propose a delta that matches a failed leaf's
   `proposed_delta` unless you can cite concrete evidence** (a new
   MetricTable or timeline signal) that the memory / feasibility
   envelope has changed since that failure.
2. **Prefer expansion at the current active leaf.** The scheduler
   already rewound the active pointer if the previous trial rolled
   back; your next proposal is layered on top of the parent context
   the scheduler chose. Do not try to explicitly change the parent —
   simply propose a delta on top of `current_knobs` and the
   scheduler will thread the parent for you. This rewind is also why
   the `## Memory pressure` (§7) and `## Last trial — GPU memory`
   (§7.1) blocks reflect the *parent's* state after a rollback, not
   the failed sibling's: the sibling's OOM / inflated occupancy was
   produced by the very delta that was reverted, so it is shown only
   as a `Recent failure leaf` + a required `bitter_lesson`, never as
   current pressure. Propose forward from the parent; do not "fix" the
   reverted sibling by lowering memory.
3. **Watch the top-K OK leaderboard.** If your proposal would resolve
   to the same cumulative config as an entry there, the scheduler
   will short-circuit to a `DUPLICATE_OF` DAG node rather than
   re-launching. Save your budget by proposing a delta that shifts
   the resolved-config SHA.
4. **Failed leaves are stronger evidence than bitter lessons alone.**
   Bitter lessons are one-line rules the critic wrote after past
   failures. Failed leaves in the DAG view are the actual delta +
   failure-mode observations. Both are shown; both matter.
5. **Duplicate config attempts signal wasted budget.** Every time the
   `Recent duplicate config attempts` section grows, it means a
   proposal resolved to an already-attempted cumulative config. When
   you see a duplicate you emitted appear there, deliberately shift
   the resolved SHA on the next proposal (change a different knob,
   change the magnitude of the change, or explicitly propose
   `stop_requested`).
6. **Early rounds are for breadth, not depth.** When the active branch
   is still shallow (the root or only a handful of nodes deep), a spread
   of single-knob children off the same parent — each probing a
   different knob — builds the cost/sensitivity map faster than drilling
   one branch deep. Once the DAG's top-K OK leaderboard shows which knob
   moves the objective most, switch to depth: keep expanding the best OK
   leaf. See the early-rounds strategy note at the top of
   `05-recipe.md`.

## Node line format

Each line renders as:

```
<node_id> (<tag>) status=<STATUS> failure_mode=<MODE> objective=<OBJ> delta_from_parent=<DELTA>
```

- `<tag>` is `root` for the baseline root and `trial=<idx>` for
  launched trials (trial indices are negative on synthetic
  `DUPLICATE_OF` entries).
- `<STATUS>` is `ROOT`, `OK`, or `FAILED`.
- `<MODE>` is one of `NONE`, `OOM`, `WORKER_CRASH`, `TIMEOUT`,
  `METRICS_PARTIAL`, `METRICS_MISSING`, `CONFIG_INVALID`,
  `DIVISIBILITY_VIOLATION`, `LAUNCH_FAILURE`, or `DUPLICATE_OF`.
- `<OBJ>` is `step_time / num_trajectories` (`n/a` for failures /
  root).
- `<DELTA>` is the incremental knob change that produced this node
  relative to its parent, as a compact single-line JSON.
- `duplicate_of=<node_id>` appears only on `DUPLICATE_OF` synthetic
  entries and points at the ORIGINAL non-duplicate node whose
  objective was recycled.
