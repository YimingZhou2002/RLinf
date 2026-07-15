"""
Plot timeline JSONL traces as a Gantt chart (one row per component/rank lane).
"""
from __future__ import annotations

import os
import sys

# Running `python .../rlinf/utils/plot_timeline.py` puts this directory first on sys.path,
# so `import logging` resolves to rlinf/utils/logging.py and breaks PIL/matplotlib.
if sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import json
from collections import defaultdict
from glob import glob
from typing import Iterable

# Default palette for known components; others cycle from tab10
_COMPONENT_COLORS: dict[str, str] = {
    "env": "#27ae60",
    "env_micro": "#58d68d",
    "rollout": "#2980b9",
    "rollout_preprocess": "#85c1e9",
    "rollout_vlm": "#2471a3",
    "rollout_ae": "#8e44ad",
    "rollout_ae_detail": "#bb8fce",
    "rollout_postprocess": "#d2b4de",
    "kv_transfer": "#16a085",
    "kv_wait": "#16a085",
    "kv_queue": "#76d7c4",
    "kv_unpack": "#48c9b0",
    "actor": "#d35400",
}

_COMPONENT_ORDER: dict[str, int] = {
    "runner": -1,
    "env": 0,
    "env_micro": 1,
    "rollout": 2,
    "rollout_preprocess": 3,
    "rollout_vlm": 4,
    "kv_transfer": 5,
    "kv_wait": 5,
    "kv_queue": 6,
    "kv_unpack": 7,
    "rollout_ae_detail": 8,
    "rollout_ae": 9,
    "rollout_postprocess": 10,
    "actor": 11,
}

_TAG_ORDER: dict[str, int] = {
    "run": 0,
    # env onload/offload (enable_offload=True 时 env 在 step 前 onload、interact 后 offload)
    "onload": 9,
    "interact": 10,
    "run_interact_once": 11,
    "offload": 12,
    "generate": 20,
    "generate_one_epoch": 21,
    "recv_rollout_results": 30,
    "predict": 31,
    "compute_bootstrap_rewards": 32,
    "env_interact_step": 33,
    "prefetch_train_bootstrap": 34,
    # rollout 模型搬运（reload 在 generate 头、offload 在尾）
    "reload_model": 35,
    "offload_model": 36,
    "recv_rollout_trajectories": 40,
    "compute_advantages_and_returns": 41,
    "run_training": 42,
    # actor 权重/优化器搬运（run_training 内、forward 前加载）
    "load_weight_and_optimizer": 43,
    "actor_forward": 44,
    "actor_policy_loss": 45,
    "actor_backward": 46,
    "actor_optimizer_step": 47,
}

_TAG_COLORS: dict[str, str] = {
    "run": "#7f8c8d",
    "interact": "#2ecc71",
    "run_interact_once": "#27ae60",
    "env_interact_step": "#1abc9c",
    "recv_rollout_results": "#58d68d",
    "compute_bootstrap_rewards": "#a9dfbf",
    "prefetch_train_bootstrap": "#16a085",
    "generate": "#21618c",
    "generate_one_epoch": "#2874a6",
    "predict": "#5dade2",
    "recv_rollout_trajectories": "#e67e22",
    "compute_advantages_and_returns": "#f5b041",
    "run_training": "#d35400",
    # 搬运：to-GPU 用暖色，to-CPU 用冷灰/蓝
    "onload": "#138d75",                # env 加载回 GPU（深青）
    "offload": "#a6acaf",               # env 卸载到 CPU（银灰）
    "reload_model": "#ff8f40",          # rollout 模型加载回 GPU（琥珀）
    "offload_model": "#5d6d7e",         # rollout 模型卸载到 CPU（蓝灰）
    "load_weight_and_optimizer": "#cb4335",  # actor 权重+优化器加载回 GPU（暗红）
    "actor_forward": "#c0392b",
    "actor_policy_loss": "#f39c12",
    "actor_backward": "#8e44ad",
    "actor_optimizer_step": "#2c3e50",
}

_DEFAULT_EXCLUDE_TAGS: set[str] = {
    "interact",
    "run_interact_once",
    "generate",
    "generate_one_epoch",
    "recv_rollout_results",
    "recv_rollout_trajectories",
    "run_training",
}


def _load_events(timeline_dir: str) -> list[dict]:
    events: list[dict] = []
    pattern = os.path.join(timeline_dir, "*.jsonl")
    for path in sorted(glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t0, t1 = float(rec["t0"]), float(rec["t1"])
                if t1 < t0:
                    t0, t1 = t1, t0
                rec["t0"], rec["t1"] = t0, t1
                rec["duration"] = t1 - t0
                events.append(rec)
    return events


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _filter_events(
    events: list[dict],
    *,
    include_tags: set[str] | None = None,
    exclude_tags: set[str] | None = None,
    include_components: set[str] | None = None,
    exclude_components: set[str] | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> list[dict]:
    filtered: list[dict] = []
    for rec in events:
        tag = str(rec.get("tag", ""))
        component = str(rec.get("component", ""))
        duration = float(rec["duration"])
        if include_tags and tag not in include_tags:
            continue
        if exclude_tags and tag in exclude_tags:
            continue
        if include_components and component not in include_components:
            continue
        if exclude_components and component in exclude_components:
            continue
        if min_duration is not None and duration < min_duration:
            continue
        if max_duration is not None and duration > max_duration:
            continue
        filtered.append(rec)
    return filtered


def _effective_exclude_tags(
    include_tags: set[str] | None,
    exclude_tags: set[str] | None,
) -> set[str] | None:
    if include_tags:
        return exclude_tags
    if exclude_tags is None:
        return set(_DEFAULT_EXCLUDE_TAGS)
    return exclude_tags


def _event_tag(rec: dict) -> str:
    return str(rec.get("tag", ""))


def _lane_key(rec: dict, lane_mode: str = "tag") -> tuple[str, int, str]:
    component = str(rec["component"])
    rank = int(rec["rank"])
    if lane_mode == "rank":
        return component, rank, ""
    return component, rank, _event_tag(rec)


def _lane_label(lane: tuple[str, int, str], lane_mode: str = "tag") -> str:
    component, rank, tag = lane
    if lane_mode == "rank":
        return f"{component}/r{rank}"
    return f"{component}/r{rank}/{tag}"


def _component_sort_key(component: str) -> tuple[int, str]:
    return (_COMPONENT_ORDER.get(component, 999), component)


def _tag_sort_key(tag: str) -> tuple[int, str]:
    return (_TAG_ORDER.get(tag, 999), tag)


def _lane_sort_key(lane: tuple[str, int, str]) -> tuple[int, str, int, int, str]:
    component, rank, tag = lane
    order, name = _component_sort_key(component)
    tag_order, tag_name = _tag_sort_key(tag)
    return (order, name, rank, tag_order, tag_name)


def _color_key(rec: dict, color_by: str) -> str:
    if color_by == "tag":
        return _event_tag(rec)
    return str(rec["component"])


def _build_color_map(keys: Iterable[str], color_by: str, palette: list[str]) -> dict[str, str]:
    colors = dict(_TAG_COLORS if color_by == "tag" else _COMPONENT_COLORS)
    ordered = sorted(set(keys), key=_tag_sort_key if color_by == "tag" else _component_sort_key)
    for i, key in enumerate(ordered):
        if key not in colors:
            colors[key] = palette[i % len(palette)]
    return colors


def plot_timeline(
    timeline_dir: str,
    output_path: str | None = None,
    fig_width: float = 14.0,
    lane_height: float = 0.65,
    dpi: int = 150,
    lane_mode: str = "tag",
    color_by: str = "tag",
    exclude_tags: set[str] | None = None,
) -> str:
    """
    Draw a Gantt chart: x = time (relative to first event), y = component/rank lane.

    Returns path to saved figure.
    """
    return plot_timeline_html(
        timeline_dir,
        output_path=output_path,
        lane_mode=lane_mode,
        color_by=color_by,
        exclude_tags=exclude_tags,
    )

def plot_timeline_png(
    timeline_dir: str,
    output_path: str | None = None,
    fig_width: float = 14.0,
    lane_height: float = 0.65,
    dpi: int = 150,
    lane_mode: str = "tag",
    color_by: str = "tag",
    target_aspect: float = 1.35,
    include_tags: set[str] | None = None,
    exclude_tags: set[str] | None = None,
    include_components: set[str] | None = None,
    exclude_components: set[str] | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> str:
    """Matplotlib PNG output."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    events = _filter_events(
        _load_events(timeline_dir),
        include_tags=include_tags,
        exclude_tags=_effective_exclude_tags(include_tags, exclude_tags),
        include_components=include_components,
        exclude_components=exclude_components,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    if not events:
        raise ValueError(f"No events found under {timeline_dir!r}")

    t_min = min(e["t0"] for e in events)
    t_max = max(e["t1"] for e in events)

    lanes = sorted({_lane_key(e, lane_mode) for e in events}, key=_lane_sort_key)
    lane_index = {lk: i for i, lk in enumerate(lanes)}

    color_keys = sorted({_color_key(e, color_by) for e in events})
    tab = plt.get_cmap("tab10")
    palette = [tab(i % tab.N) for i in range(max(tab.N, len(color_keys)))]
    colors = _build_color_map(color_keys, color_by, palette)

    by_lane: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for e in events:
        by_lane[_lane_key(e, lane_mode)].append(e)
    for lk in by_lane:
        by_lane[lk].sort(key=lambda r: r["t0"])

    nlanes = len(lanes)
    fig_h = max(4.0, 1.2 + nlanes * 0.55)
    # Keep the chart landscape: widen with lane count so tall tag-mode timelines
    # do not end up much taller than wide.
    fig_w = max(fig_width, fig_h * target_aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_h = min(lane_height, 0.85)
    for e in events:
        lk = _lane_key(e, lane_mode)
        y = lane_index[lk]
        x0 = e["t0"] - t_min
        w = e["t1"] - e["t0"]
        color = colors[_color_key(e, color_by)]
        ax.barh(
            y,
            w,
            left=x0,
            height=bar_h,
            color=color,
            alpha=0.78,
            linewidth=0,
            align="center",
        )

    yticklabels = [_lane_label(lane, lane_mode) for lane in lanes]
    ax.set_yticks(range(nlanes))
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time from trace start (s)")
    ax.set_title(
        f"Timeline Gantt · {len(events)} events · "
        f"span {t_max - t_min:.2f}s · {os.path.basename(timeline_dir.rstrip(os.sep))}"
    )
    ax.set_xlim(0, max(t_max - t_min, 1e-6))
    ax.set_ylim(-0.6, nlanes - 0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    handles = [
        mpatches.Patch(color=colors[c], label=c, alpha=0.78)
        for c in sorted(color_keys, key=_tag_sort_key if color_by == "tag" else _component_sort_key)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(timeline_dir, "timeline_gantt.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_timeline_html(
    timeline_dir: str,
    output_path: str | None = None,
    *,
    lane_height: float = 0.55,
    width_px: int = 1200,
    target_aspect: float = 1.35,
    lane_mode: str = "tag",
    color_by: str = "tag",
    include_tags: set[str] | None = None,
    exclude_tags: set[str] | None = None,
    include_components: set[str] | None = None,
    exclude_components: set[str] | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> str:
    """
    Interactive HTML output (pan/zoom + hover) using Plotly.
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotly is required for --format html. Install it with: pip install plotly"
        ) from e

    if not hasattr(go, "Figure") or not hasattr(go, "Bar"):
        raise RuntimeError(
            "plotly.graph_objects is incomplete; please reinstall plotly in this environment"
        )

    events = _filter_events(
        _load_events(timeline_dir),
        include_tags=include_tags,
        exclude_tags=_effective_exclude_tags(include_tags, exclude_tags),
        include_components=include_components,
        exclude_components=exclude_components,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    if not events:
        raise ValueError(f"No events found under {timeline_dir!r}")

    t_min = min(e["t0"] for e in events)
    t_max = max(e["t1"] for e in events)

    lanes = sorted({_lane_key(e, lane_mode) for e in events}, key=_lane_sort_key)
    lane_labels = [_lane_label(lk, lane_mode) for lk in lanes]
    lane_label_by_key = {lk: _lane_label(lk, lane_mode) for lk in lanes}

    plotly_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_keys = {_color_key(e, color_by) for e in events}
    colors = _build_color_map(color_keys, color_by, plotly_palette)

    # Group events by color key into separate traces (legend toggles).
    events_by_color: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        events_by_color[_color_key(e, color_by)].append(e)

    fig = go.Figure()
    sorted_color_keys = sorted(
        color_keys,
        key=_tag_sort_key if color_by == "tag" else _component_sort_key,
    )
    for color_key in sorted_color_keys:
        trace_events = events_by_color.get(color_key, [])
        if not trace_events:
            continue
        y = [lane_label_by_key[_lane_key(e, lane_mode)] for e in trace_events]
        base = [e["t0"] - t_min for e in trace_events]
        dur = [e["t1"] - e["t0"] for e in trace_events]
        tag = [str(e.get("tag", "")) for e in trace_events]
        step = [e.get("global_step", None) for e in trace_events]
        component = [str(e.get("component", "")) for e in trace_events]
        qualname = [str(e.get("qualname", "")) for e in trace_events]
        rollout_epoch = [e.get("rollout_epoch", None) for e in trace_events]
        chunk_step = [e.get("chunk_step", None) for e in trace_events]
        stage_id = [e.get("stage_id", None) for e in trace_events]
        phase = [e.get("phase", None) for e in trace_events]
        mode = [e.get("mode", None) for e in trace_events]
        call_index = [e.get("call_index", None) for e in trace_events]

        fig.add_trace(
            go.Bar(
                name=color_key,
                orientation="h",
                y=y,
                x=dur,
                base=base,
                marker=dict(color=colors[color_key]),
                opacity=0.78,
                customdata=list(
                    zip(
                        component,
                        tag,
                        dur,
                        base,
                        step,
                        qualname,
                        rollout_epoch,
                        chunk_step,
                        stage_id,
                        phase,
                        mode,
                        call_index,
                    )
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "component=%{customdata[0]}<br>"
                    "tag=%{customdata[1]}<br>"
                    "t0=%{customdata[3]:.3f}s<br>"
                    "dt=%{customdata[2]:.3f}s<br>"
                    "global_step=%{customdata[4]}<br>"
                    "rollout_epoch=%{customdata[6]}<br>"
                    "chunk_step=%{customdata[7]}<br>"
                    "stage_id=%{customdata[8]}<br>"
                    "phase=%{customdata[9]}<br>"
                    "mode=%{customdata[10]}<br>"
                    "call_index=%{customdata[11]}<br>"
                    "%{customdata[5]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="overlay",
        bargap=0.15,
        height=max(360, int(120 + len(lanes) * 30)),
        width=max(width_px, int(max(360, int(120 + len(lanes) * 30)) * target_aspect)),
        title=(
            f"Timeline Gantt · {len(events)} events · "
            f"span {t_max - t_min:.2f}s · {os.path.basename(timeline_dir.rstrip(os.sep))}"
        ),
        xaxis=dict(title="Time from trace start (s)", rangeslider=dict(visible=True)),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=lane_labels,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=210 if lane_mode == "tag" else 120, r=20, t=60, b=40),
    )

    # make bars slightly thinner by adjusting marker line / gaps; Plotly doesn't have direct lane height
    fig.update_traces(marker_line_width=0)

    if output_path is None:
        output_path = os.path.join(timeline_dir, "timeline_gantt.html")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot timeline JSONL as a Gantt chart.")
    parser.add_argument(
        "timeline_dir",
        nargs="?",
        default="/mnt/public/zengwen/RLinf/logs/20260331-03:46:32-behavior_ppo_openpi_pi05/timeline",
        help="Directory containing *_rank*.jsonl timeline files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default depends on --format)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "html"],
        default="html",
        help="Output format: png (static) or html (interactive zoom/pan)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Alias for --format html",
    )
    parser.add_argument("--width", type=float, default=14.0, help="Figure width in inches")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    parser.add_argument(
        "--target-aspect",
        type=float,
        default=1.35,
        help="Target width/height ratio; width auto-grows with lane count so the chart stays landscape (avoids tall-thin Gantt charts)",
    )
    parser.add_argument(
        "--lane-mode",
        choices=["tag", "rank"],
        default="tag",
        help="Rows: tag separates nested worker timers; rank preserves the old component/rank rows",
    )
    parser.add_argument(
        "--color-by",
        choices=["tag", "component"],
        default="tag",
        help="Legend/color grouping",
    )
    parser.add_argument("--include-tags", default=None, help="Comma-separated tag allow-list")
    parser.add_argument(
        "--exclude-tags",
        default=None,
        help=(
            "Comma-separated tags to hide. By default hides outer wrapper tags: "
            + ",".join(sorted(_DEFAULT_EXCLUDE_TAGS))
        ),
    )
    parser.add_argument(
        "--include-components", default=None, help="Comma-separated component allow-list"
    )
    parser.add_argument(
        "--exclude-components", default=None, help="Comma-separated components to hide"
    )
    parser.add_argument("--min-duration", type=float, default=None, help="Hide shorter events")
    parser.add_argument("--max-duration", type=float, default=None, help="Hide longer events")
    args = parser.parse_args()

    include_tags = _parse_csv(args.include_tags)
    exclude_tags = (
        set(_DEFAULT_EXCLUDE_TAGS)
        if not include_tags and args.exclude_tags is None
        else _parse_csv(args.exclude_tags)
    )
    include_components = _parse_csv(args.include_components)
    exclude_components = _parse_csv(args.exclude_components)

    out_format = "html" if args.interactive else args.format
    if out_format == "html":
        out = plot_timeline_html(
            args.timeline_dir,
            output_path=args.output,
            lane_mode=args.lane_mode,
            color_by=args.color_by,
            target_aspect=args.target_aspect,
            include_tags=include_tags or None,
            exclude_tags=exclude_tags,
            include_components=include_components or None,
            exclude_components=exclude_components or None,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    else:
        out = plot_timeline_png(
            args.timeline_dir,
            output_path=args.output,
            fig_width=args.width,
            dpi=args.dpi,
            lane_mode=args.lane_mode,
            color_by=args.color_by,
            target_aspect=args.target_aspect,
            include_tags=include_tags or None,
            exclude_tags=exclude_tags,
            include_components=include_components or None,
            exclude_components=exclude_components or None,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    print(out)


if __name__ == "__main__":
    main()
