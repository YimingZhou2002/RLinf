"""Render a human-readable timeline summary txt from a ``timeline/`` dir.

Dumps every view in :mod:`toolkits.embodied_tuner.timeline_processor`
(stall fractions, per-tag stats, offload cost, per-component bubble,
steady-state call averages, outliers, critical path, top raw excerpts)
into one text file. Intended for debugging a single trial by hand —
the LLM-facing rendering lives in :mod:`toolkits.embodied_tuner.critic`.

Usage::

    python toolkits/embodied_tuner/summarize_timeline.py <timeline_dir> [-o out.txt]

If ``-o`` is omitted, writes ``<timeline_dir>/timeline_summary.txt``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a plain script from anywhere: make the package dir
# importable so `import timeline_processor` resolves.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import timeline_processor as tp  # noqa: E402


_HEADLINE_TAGS: tuple[str, ...] = (
    "env_interact_step",
    "env/bootstrap_step",
    "actor/recv_traj",
    "actor/sync_model_to_rollout",
    "actor/compute_adv",
    "rollout/generate",
    "predict",
    # offload-bearing tags so the headline table shows their per-call cost
    "onload",
    "offload",
    "reload_model",
    "offload_model",
    "load_weight_and_optimizer",
)


def _fmt(value, spec: str = ".3f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "n/a"


def _render_header(timeline_dir: Path, events: list[dict]) -> list[str]:
    n = len(events)
    if n:
        t0 = min(float(e["t0"]) for e in events)
        t1 = max(float(e["t1"]) for e in events)
        span = t1 - t0
    else:
        span = 0.0
    lines = [
        "=" * 78,
        f"Timeline summary — {timeline_dir}",
        f"events={n}  wall_span={_fmt(span)}s",
        "=" * 78,
        "",
    ]
    return lines


def _render_stalls(stalls: dict[str, float]) -> list[str]:
    lines = ["## Stall fractions (idle / window, per component)"]
    if not stalls:
        lines.append("  (none)")
    for component in sorted(stalls):
        lines.append(f"  - {component}: {_fmt(stalls[component], '.3f')}")
    lines.append("")
    return lines


def _render_tag_stats(rows: list[dict]) -> list[str]:
    lines = ["## Per-tag stats (component / rank / tag / count / min / median / max / total)"]
    if not rows:
        lines.append("  (no headline-tag events)")
    for row in rows:
        lines.append(
            f"  - {row['component']} r{row['rank']} {row['tag']}  "
            f"count={row['call_count']}  "
            f"min={_fmt(row['duration_min'])}  "
            f"median={_fmt(row['duration_median'])}  "
            f"max={_fmt(row['duration_max'])}  "
            f"total={_fmt(row['duration_total'])}"
        )
    lines.append("")
    return lines


def _render_offload_block(oc: dict) -> list[str]:
    lines = ["## Offload cost (CPU<->GPU transfer, per component / direction)"]
    if not oc:
        lines.append("  (no offload-bearing events found — enable_offload likely False)")
        lines.append("")
        return lines
    wall = oc.get("wall_s")
    totals = oc.get("totals", {})
    lines.append(
        f"  wall={_fmt(wall)}s  "
        f"onload_total={_fmt(totals.get('onload_total_s'))}s  "
        f"offload_total={_fmt(totals.get('offload_total_s'))}s  "
        f"combined={_fmt(totals.get('combined_total_s'))}s "
        f"({_fmt(totals.get('combined_frac_of_wall') * 100, '.1f')}% of wall)"
    )
    per_component = oc.get("per_component", {})
    for component in sorted(per_component):
        info = per_component[component]
        onload = info.get("onload", {})
        offload = info.get("offload", {})
        lines.append(
            f"  - {component}: combined={_fmt(info.get('combined_total_s'))}s "
            f"({_fmt(info.get('combined_frac_of_wall') * 100, '.1f')}% of wall)"
        )
        if onload.get("count"):
            lines.append(
                f"      onload : count={onload['count']}  "
                f"total={_fmt(onload['total_s'])}s  mean={_fmt(onload['mean_s'])}s  "
                f"median={_fmt(onload['median_s'])}s  "
                f"min={_fmt(onload['min_s'])}  max={_fmt(onload['max_s'])}"
            )
        if offload.get("count"):
            lines.append(
                f"      offload: count={offload['count']}  "
                f"total={_fmt(offload['total_s'])}s  mean={_fmt(offload['mean_s'])}s  "
                f"median={_fmt(offload['median_s'])}s  "
                f"min={_fmt(offload['min_s'])}  max={_fmt(offload['max_s'])}"
            )
    lines.append("")
    return lines


def _render_bubble(bubble: dict) -> list[str]:
    lines = ["## Per-component bubble (real-busy vs idle)"]
    if not bubble:
        lines.append("  (none)")
        lines.append("")
        return lines
    lines.append(f"  wall={_fmt(bubble.get('wall_s'))}s")
    for component in sorted(bubble.get("per_component", {})):
        info = bubble["per_component"][component]
        lines.append(
            f"  - {component}: busy={_fmt(info.get('busy_s'))}s  "
            f"bubble={_fmt(info.get('bubble_s'))}s  "
            f"bubble_frac={_fmt(info.get('bubble_frac'), '.2f')}  "
            f"ranks={info.get('num_ranks')}"
        )
    lines.append("")
    return lines


def _render_call_avgs(avgs: dict) -> list[str]:
    lines = ["## Steady-state per-call duration (first 2 warmup calls dropped)"]
    if not avgs:
        lines.append("  (none)")
        lines.append("")
        return lines
    for component in sorted(avgs):
        info = avgs[component]
        lines.append(
            f"  - {component}: mean={_fmt(info.get('mean_duration_s'))}s  "
            f"min={_fmt(info.get('min_duration_s'))}  "
            f"max={_fmt(info.get('max_duration_s'))}  "
            f"n={info.get('remaining_count')} "
            f"(skipped={info.get('skipped')} of {info.get('call_count_total')})"
        )
    lines.append("")
    return lines


def _render_outliers(outliers) -> list[str]:
    lines = ["## Outlier events (per-tag P95, >1s)"]
    if not outliers:
        lines.append("  (none)")
        lines.append("")
        return lines
    for o in outliers:
        hint = o.get("knob_hint")
        lines.append(
            f"  - {o.get('component')}/r{o.get('rank')} {o.get('tag')}  "
            f"step={o.get('global_step')}  dur={_fmt(o.get('dur_s'))}s"
            + (f"  hint={hint}" if hint else "")
        )
    lines.append("")
    return lines


def _render_critical_path(cp: dict) -> list[str]:
    lines = ["## Critical path per global_step (real-busy top lanes)"]
    if not cp:
        lines.append("  (none — no global_step on any event, bootstrap only)")
        lines.append("")
        return lines
    for raw_step in sorted(cp, key=lambda k: int(k)):
        step = cp[raw_step]
        lines.append(f"  - step={raw_step}  span={_fmt(step.get('step_span_s'), '.1f')}s")
        for lane in step.get("real_busy_top", ()):
            lines.append(
                f"      {lane['component']}/r{lane['rank']}: "
                f"real={_fmt(lane['real_s'], '.1f')}s  "
                f"blocked={_fmt(lane['blocked_s'], '.1f')}s  "
                f"real_frac={_fmt(lane['real_frac'], '.2f')}"
            )
    lines.append("")
    return lines


def _render_raw_excerpts(excerpts) -> list[str]:
    import json
    lines = ["## Top raw excerpts (longest events, runner excluded)"]
    if not excerpts:
        lines.append("  (none)")
        lines.append("")
        return lines
    for e in excerpts:
        lines.append("  - " + json.dumps(e, sort_keys=True))
    lines.append("")
    return lines


def build_summary(timeline_dir: Path, *, enable_offload=None) -> str:
    events = tp.load_events(timeline_dir)
    sections: list[str] = []
    sections += _render_header(timeline_dir, events)
    if not events:
        sections.append("(no events)")
        return "\n".join(sections) + "\n"
    # Lead with the two views the critic weighs most heavily: steady-state
    # per-call cost and CPU<->GPU offload cost. Keep the txt ordering in sync
    # with the critic verbose block (see critic._render_timeline_verbose).
    sections += _render_call_avgs(tp.compute_component_call_averages(events))
    sections += _render_offload_block(tp.compute_offload_cost(events))
    sections += _render_stalls(tp.compute_stall_fractions(events))
    sections += _render_tag_stats(
        tp.compute_tag_stats(events, headline_tags=_HEADLINE_TAGS)
    )
    sections += _render_bubble(tp.compute_per_component_bubble(events))
    sections += _render_outliers(
        tp.compute_outliers(events, enable_offload=enable_offload)
    )
    sections += _render_critical_path(tp.compute_critical_path(events))
    sections += _render_raw_excerpts(tp.extract_raw_excerpts(events))
    return "\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a timeline summary txt.")
    parser.add_argument("timeline_dir", help="Directory containing *_rank*.jsonl")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output txt path (default: <timeline_dir>/timeline_summary.txt)",
    )
    args = parser.parse_args(argv)

    timeline_dir = Path(args.timeline_dir)
    if not timeline_dir.is_dir():
        print(f"error: {timeline_dir} is not a directory", file=sys.stderr)
        return 1

    text = build_summary(timeline_dir)
    out_path = Path(args.output) if args.output else timeline_dir / "timeline_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
