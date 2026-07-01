# Tuner wiki — optimization context for the critic

This directory holds natural-language context the LLM critic can cite when
proposing knob deltas. It complements the mechanical inputs the critic
already receives (MetricTable, `timeline/*.jsonl`, current knob values,
trial history) with:

1. **A model of the critical path** under each placement mode, so the
   critic can reason "shortening component X only helps if X is on the
   critical path in this placement".
2. **Per-knob optimization directions** — which knobs move which timings
   and memory footprints, and their common failure modes.
3. **A signal-to-conclusion decoder** for timeline events — which tag
   means what, and which tags are noisy wrappers to be ignored.

## Files

- [`bottleneck-rubric.md`](bottleneck-rubric.md) — reading order, when
  to trust which block, and the term→knob decision table. This is the
  extracted-and-rewritten former `_BOTTLENECK_RUBRIC` constant from
  `critic.py`.
- [`placement-critical-paths.md`](placement-critical-paths.md) — critical
  path per placement mode (collocated / hybrid / disaggregated) and per
  runner mode (`run` vs `run_pipeline`), with short-form formulas.
- [`optimization-directions.md`](optimization-directions.md) — knob-by-knob
  playbook: what each knob shifts, when to touch it, common OOM patterns.
- [`timeline-signals.md`](timeline-signals.md) — what each timeline tag
  means, how `stall_fraction` is defined, and which tags to ignore.
- [`constraints.md`](constraints.md) — every rule that will refuse a
  delta, split into preflight-enforced (fail fast, no trial) and
  runtime-enforced (crashes the trial). Consult before proposing any
  delta that touches divisibility-sensitive knobs or placement.

## For the critic prompt builder

Every file is intended to be consumed **verbatim** by
`critic.build_prompt`. Keep them short, declarative, and free of
speculation — the critic will treat these lines as ground truth.

If you want the critic to weight a hypothesis differently, edit the wiki
before adjusting the rubric. This keeps `_BOTTLENECK_RUBRIC` in
`critic.py` general and defers workload-specific priors here.
