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
   scheduler will thread the parent for you.
3. **Watch the top-K OK leaderboard.** If your proposal would resolve
   to the same cumulative config as an entry there, the scheduler
   will short-circuit to a `DUPLICATE_OF` DAG node rather than
   re-launching. Save your budget by proposing a delta that shifts
   the resolved-config SHA.
4. **Failed leaves are stronger evidence than bitter lessons alone.**
   Bitter lessons are one-line rules the critic wrote after past
   failures. Failed leaves in the DAG view are the actual delta +
   failure-mode observations. Both are shown; both matter.

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
