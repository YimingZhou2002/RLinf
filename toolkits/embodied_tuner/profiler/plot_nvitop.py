"""
Plot nvitop JSONL traces as CPU, RAM, and GPU resource curves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob
from statistics import mean
from typing import Any

if sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)


_COMPONENT_COLORS: dict[str, str] = {
    "runner": "#7f8c8d",
    "actor": "#d35400",
    "rollout": "#2980b9",
    "env": "#27ae60",
    "reward": "#8e44ad",
    "behavior_subproc": "#16a085",
}

_COMPONENT_ORDER: dict[str, int] = {
    "runner": 0,
    "actor": 1,
    "rollout": 2,
    "env": 3,
    "reward": 4,
    "behavior_subproc": 5,
}

_GPU_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _component_sort_key(component: str) -> tuple[int, str]:
    return (_COMPONENT_ORDER.get(component, 999), component)


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _load_samples(nvitop_dir: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    pattern = os.path.join(nvitop_dir, "*.jsonl")
    for path in sorted(glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_path"] = path
                rec["ts"] = float(rec["ts"])
                _normalize_sample(rec)
                samples.append(rec)
    samples.sort(
        key=lambda rec: (
            rec["ts"],
            str(rec.get("component", "")),
            int(rec.get("rank", 0)),
            int(rec.get("pid", 0)),
        )
    )
    return samples


def _normalize_sample(rec: dict[str, Any]) -> None:
    if "process_rss_gib" not in rec and "process_rss_bytes" in rec:
        rec["process_rss_gib"] = float(rec["process_rss_bytes"]) / (2**30)
    if "system_memory_used_gib" not in rec and "system_memory_used_bytes" in rec:
        rec["system_memory_used_gib"] = float(rec["system_memory_used_bytes"]) / (2**30)

    pid = int(rec.get("pid", 0))
    gpu_mem = 0.0
    gpu_indices: list[int] = []
    gpu_sm_utils: list[float] = []
    gpu_mem_utils: list[float] = []
    for gpu in rec.get("gpus", []) or []:
        gpu_index = gpu.get("gpu_index")
        for proc in gpu.get("processes", []) or []:
            if int(proc.get("pid", -1)) != pid:
                continue
            value = _to_float(proc.get("gpu_memory_gib"))
            if value is None and "gpu_memory_bytes" in proc:
                value = float(proc["gpu_memory_bytes"]) / (2**30)
            if value is not None:
                gpu_mem += value
            if gpu_index is not None:
                try:
                    gpu_indices.append(int(gpu_index))
                except Exception:
                    pass
            sm = _to_float(proc.get("gpu_sm_util_percent"))
            mem_util = _to_float(proc.get("gpu_memory_util_percent"))
            if sm is not None:
                gpu_sm_utils.append(sm)
            if mem_util is not None:
                gpu_mem_utils.append(mem_util)
    rec["process_gpu_memory_gib"] = gpu_mem if gpu_mem > 0 else None
    rec["process_gpu_indices"] = sorted(set(gpu_indices))
    rec["process_gpu_sm_util_percent"] = max(gpu_sm_utils) if gpu_sm_utils else None
    rec["process_gpu_memory_util_percent"] = (
        max(gpu_mem_utils) if gpu_mem_utils else None
    )


def _filter_samples(
    samples: list[dict[str, Any]],
    *,
    include_components: set[str] | None = None,
    exclude_components: set[str] | None = None,
    include_ranks: set[int] | None = None,
    include_pids: set[int] | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for rec in samples:
        component = str(rec.get("component", ""))
        rank = int(rec.get("rank", 0))
        pid = int(rec.get("pid", 0))
        if include_components and component not in include_components:
            continue
        if exclude_components and component in exclude_components:
            continue
        if include_ranks and rank not in include_ranks:
            continue
        if include_pids and pid not in include_pids:
            continue
        filtered.append(rec)
    return filtered


def _process_key(rec: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(rec.get("component", "")),
        int(rec.get("rank", 0)),
        int(rec.get("pid", 0)),
    )


def _process_label(key: tuple[str, int, int]) -> str:
    component, rank, pid = key
    return f"{component}/r{rank}/pid{pid}"


def _process_colors(keys: list[tuple[str, int, int]]) -> dict[tuple[str, int, int], str]:
    palette = [
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
    colors: dict[tuple[str, int, int], str] = {}
    ordered = sorted(keys, key=lambda key: (_component_sort_key(key[0]), key[1], key[2]))
    next_palette = 0
    seen_components: set[str] = set()
    for key in ordered:
        component = key[0]
        base = _COMPONENT_COLORS.get(component)
        if base and component not in seen_components:
            colors[key] = base
            seen_components.add(component)
        else:
            colors[key] = palette[next_palette % len(palette)]
            next_palette += 1
    return colors


def _default_output_path(nvitop_dir: str, out_format: str) -> str:
    ext = "html" if out_format == "html" else "png"
    return os.path.join(nvitop_dir, f"nvitop_resources.{ext}")


def _aggregate_system(samples: list[dict[str, Any]], t0: float, bin_s: float) -> list[dict]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        buckets[int((rec["ts"] - t0) / bin_s)].append(rec)
    out = []
    for bucket, records in sorted(buckets.items()):
        cpu_values = [
            value
            for value in (_to_float(rec.get("system_cpu_percent")) for rec in records)
            if value is not None
        ]
        mem_values = [
            value
            for value in (_to_float(rec.get("system_memory_used_gib")) for rec in records)
            if value is not None
        ]
        if not cpu_values and not mem_values:
            continue
        out.append(
            {
                "x": bucket * bin_s,
                "system_cpu_percent": mean(cpu_values) if cpu_values else None,
                "system_memory_used_gib": mean(mem_values) if mem_values else None,
            }
        )
    return out


def _aggregate_gpu(
    samples: list[dict[str, Any]],
    t0: float,
    bin_s: float,
) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        bucket = int((rec["ts"] - t0) / bin_s)
        for gpu in rec.get("gpus", []) or []:
            gpu_index = gpu.get("gpu_index")
            if gpu_index is None:
                continue
            try:
                key = (int(gpu_index), bucket)
            except Exception:
                continue
            buckets[key].append(gpu)

    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (gpu_index, bucket), records in sorted(buckets.items()):
        mem_values = [
            value
            for value in (_to_float(gpu.get("memory_used_gib")) for gpu in records)
            if value is not None
        ]
        util_values = [
            value
            for value in (_to_float(gpu.get("gpu_util_percent")) for gpu in records)
            if value is not None
        ]
        mem_util_values = [
            value
            for value in (_to_float(gpu.get("memory_util_percent")) for gpu in records)
            if value is not None
        ]
        if not mem_values and not util_values and not mem_util_values:
            continue
        by_gpu[gpu_index].append(
            {
                "x": bucket * bin_s,
                "memory_used_gib": max(mem_values) if mem_values else None,
                "gpu_util_percent": max(util_values) if util_values else None,
                "memory_util_percent": max(mem_util_values) if mem_util_values else None,
            }
        )
    return by_gpu


def _clean_values(values: list[Any]) -> list[float]:
    cleaned = []
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            cleaned.append(parsed)
    return cleaned


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{value:.3f}{suffix}"


def _summary_stats(values: list[Any]) -> dict[str, float | None]:
    cleaned = _clean_values(values)
    if not cleaned:
        return {"avg": None, "max": None, "min": None}
    return {
        "avg": mean(cleaned),
        "max": max(cleaned),
        "min": min(cleaned),
    }


def write_nvitop_summary(
    nvitop_dir: str,
    samples: list[dict[str, Any]],
    *,
    include_gpus: set[int] | None = None,
    aggregate_bin_s: float = 1.0,
    output_path: str | None = None,
) -> str:
    if not samples:
        raise ValueError("Cannot write nvitop summary without samples")

    t0 = min(rec["ts"] for rec in samples)
    t1 = max(rec["ts"] for rec in samples)
    process_keys = sorted(
        {_process_key(rec) for rec in samples},
        key=lambda key: (_component_sort_key(key[0]), key[1], key[2]),
    )

    by_process: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        by_process[_process_key(rec)].append(rec)

    gpu_series = _aggregate_gpu(samples, t0, aggregate_bin_s)
    if include_gpus:
        gpu_series = {
            gpu_index: records
            for gpu_index, records in gpu_series.items()
            if gpu_index in include_gpus
        }

    lines = [
        "nvitop resource summary",
        f"source_dir: {nvitop_dir}",
        f"samples: {len(samples)}",
        f"span_s: {t1 - t0:.3f}",
        f"aggregate_bin_s: {aggregate_bin_s:.3f}",
        "",
        "global_gpu_summary:",
    ]

    all_gpu_mem = []
    all_gpu_util = []
    all_gpu_mem_util = []
    for gpu_index, records in sorted(gpu_series.items()):
        mem_stats = _summary_stats([row.get("memory_used_gib") for row in records])
        util_stats = _summary_stats([row.get("gpu_util_percent") for row in records])
        mem_util_stats = _summary_stats(
            [row.get("memory_util_percent") for row in records]
        )
        all_gpu_mem.extend(row.get("memory_used_gib") for row in records)
        all_gpu_util.extend(row.get("gpu_util_percent") for row in records)
        all_gpu_mem_util.extend(row.get("memory_util_percent") for row in records)
        lines.append(
            "  "
            f"gpu{gpu_index}: "
            f"avg_mem={_fmt(mem_stats['avg'], 'GiB')}, "
            f"max_mem={_fmt(mem_stats['max'], 'GiB')}, "
            f"avg_gpu_util={_fmt(util_stats['avg'], '%')}, "
            f"max_gpu_util={_fmt(util_stats['max'], '%')}, "
            f"avg_mem_util={_fmt(mem_util_stats['avg'], '%')}, "
            f"max_mem_util={_fmt(mem_util_stats['max'], '%')}"
        )

    overall_mem = _summary_stats(all_gpu_mem)
    overall_util = _summary_stats(all_gpu_util)
    overall_mem_util = _summary_stats(all_gpu_mem_util)
    active_gpu_mem = []
    active_gpu_util = []
    active_gpu_mem_util = []
    for records in gpu_series.values():
        mem_values = _clean_values([row.get("memory_used_gib") for row in records])
        util_values = _clean_values([row.get("gpu_util_percent") for row in records])
        if not mem_values and not util_values:
            continue
        avg_mem = mean(mem_values) if mem_values else 0.0
        avg_util = mean(util_values) if util_values else 0.0
        if avg_mem < 0.1 and avg_util < 1.0:
            continue
        active_gpu_mem.extend(mem_values)
        active_gpu_util.extend(util_values)
        active_gpu_mem_util.extend(
            row.get("memory_util_percent") for row in records
        )
    active_mem = _summary_stats(active_gpu_mem)
    active_util = _summary_stats(active_gpu_util)
    active_mem_util = _summary_stats(active_gpu_mem_util)
    lines.extend(
        [
            "  overall_across_selected_gpus: "
            f"avg_mem={_fmt(overall_mem['avg'], 'GiB')}, "
            f"max_mem={_fmt(overall_mem['max'], 'GiB')}, "
            f"avg_gpu_util={_fmt(overall_util['avg'], '%')}, "
            f"max_gpu_util={_fmt(overall_util['max'], '%')}, "
            f"avg_mem_util={_fmt(overall_mem_util['avg'], '%')}, "
            f"max_mem_util={_fmt(overall_mem_util['max'], '%')}",
            "  overall_active_gpus: "
            f"avg_mem={_fmt(active_mem['avg'], 'GiB')}, "
            f"max_mem={_fmt(active_mem['max'], 'GiB')}, "
            f"avg_gpu_util={_fmt(active_util['avg'], '%')}, "
            f"max_gpu_util={_fmt(active_util['max'], '%')}, "
            f"avg_mem_util={_fmt(active_mem_util['avg'], '%')}, "
            f"max_mem_util={_fmt(active_mem_util['max'], '%')}",
            "",
            "process_summary:",
        ]
    )

    for key in process_keys:
        records = by_process[key]
        label = _process_label(key)
        rss_stats = _summary_stats([rec.get("process_rss_gib") for rec in records])
        cpu_stats = _summary_stats([rec.get("process_cpu_percent") for rec in records])
        proc_gpu_stats = _summary_stats(
            [rec.get("process_gpu_memory_gib") for rec in records]
        )
        proc_gpu_util_stats = _summary_stats(
            [rec.get("process_gpu_sm_util_percent") for rec in records]
        )
        gpu_indices = sorted(
            {
                gpu_index
                for rec in records
                for gpu_index in (rec.get("process_gpu_indices") or [])
            }
        )
        lines.append(
            "  "
            f"{label}: "
            f"avg_rss={_fmt(rss_stats['avg'], 'GiB')}, "
            f"max_rss={_fmt(rss_stats['max'], 'GiB')}, "
            f"avg_cpu={_fmt(cpu_stats['avg'], '%')}, "
            f"max_cpu={_fmt(cpu_stats['max'], '%')}, "
            f"avg_process_gpu_mem={_fmt(proc_gpu_stats['avg'], 'GiB')}, "
            f"max_process_gpu_mem={_fmt(proc_gpu_stats['max'], 'GiB')}, "
            f"avg_process_gpu_util={_fmt(proc_gpu_util_stats['avg'], '%')}, "
            f"max_process_gpu_util={_fmt(proc_gpu_util_stats['max'], '%')}, "
            f"gpu_indices={gpu_indices or 'n/a'}"
        )

    if output_path is None:
        output_path = os.path.join(nvitop_dir, "nvitop_summary.log")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def plot_nvitop_html(
    nvitop_dir: str,
    output_path: str | None = None,
    *,
    include_components: set[str] | None = None,
    exclude_components: set[str] | None = None,
    include_ranks: set[int] | None = None,
    include_pids: set[int] | None = None,
    include_gpus: set[int] | None = None,
    aggregate_bin_s: float = 1.0,
    width_px: int = 1350,
    height_px: int | None = None,
    summary_output: str | None = None,
) -> str:
    samples = _filter_samples(
        _load_samples(nvitop_dir),
        include_components=include_components,
        exclude_components=exclude_components,
        include_ranks=include_ranks,
        include_pids=include_pids,
    )
    if not samples:
        raise ValueError(f"No nvitop samples found under {nvitop_dir!r}")
    write_nvitop_summary(
        nvitop_dir,
        samples,
        include_gpus=include_gpus,
        aggregate_bin_s=aggregate_bin_s,
        output_path=summary_output,
    )

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Plotly is required for --format html. Install it with: pip install plotly"
        ) from exc

    t0 = min(rec["ts"] for rec in samples)
    process_keys = sorted(
        {_process_key(rec) for rec in samples},
        key=lambda key: (_component_sort_key(key[0]), key[1], key[2]),
    )
    colors = _process_colors(process_keys)

    by_process: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        by_process[_process_key(rec)].append(rec)
    for key in by_process:
        by_process[key].sort(key=lambda rec: rec["ts"])

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=(
            "Process RSS",
            "Process CPU",
            "Process GPU Memory",
            "System CPU / Memory",
            "Global GPU Memory / Utilization",
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],
    )

    for key in process_keys:
        records = by_process[key]
        label = _process_label(key)
        color = colors[key]
        x = [rec["ts"] - t0 for rec in records]
        step = [rec.get("global_step") for rec in records]
        worker_name = [rec.get("worker_name") for rec in records]
        threads = [rec.get("process_threads") for rec in records]
        gpu_indices = [",".join(map(str, rec.get("process_gpu_indices") or [])) for rec in records]

        rss = [_to_float(rec.get("process_rss_gib")) for rec in records]
        cpu = [_to_float(rec.get("process_cpu_percent")) for rec in records]
        process_gpu_mem = [_to_float(rec.get("process_gpu_memory_gib")) for rec in records]

        custom = list(zip(step, worker_name, threads, gpu_indices))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rss,
                mode="lines",
                name=label,
                legendgroup=label,
                line=dict(color=color, width=2),
                customdata=custom,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "t=%{x:.3f}s<br>"
                    "rss=%{y:.3f} GiB<br>"
                    "global_step=%{customdata[0]}<br>"
                    "worker=%{customdata[1]}<br>"
                    "threads=%{customdata[2]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cpu,
                mode="lines",
                name=f"{label} cpu",
                legendgroup=label,
                showlegend=False,
                line=dict(color=color, width=1.6),
                customdata=custom,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "t=%{x:.3f}s<br>"
                    "cpu=%{y:.1f}%<br>"
                    "global_step=%{customdata[0]}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        if any(value is not None for value in process_gpu_mem):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=process_gpu_mem,
                    mode="lines",
                    name=f"{label} gpu_mem",
                    legendgroup=label,
                    showlegend=False,
                    line=dict(color=color, width=1.8),
                    customdata=custom,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "t=%{x:.3f}s<br>"
                        "process_gpu_mem=%{y:.3f} GiB<br>"
                        "gpu_indices=%{customdata[3]}<br>"
                        "global_step=%{customdata[0]}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )

    system = _aggregate_system(samples, t0, aggregate_bin_s)
    if system:
        x = [row["x"] for row in system]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[row.get("system_memory_used_gib") for row in system],
                mode="lines",
                name="system memory",
                line=dict(color="#34495e", width=2),
                hovertemplate="t=%{x:.3f}s<br>system_memory=%{y:.3f} GiB<extra></extra>",
            ),
            row=4,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[row.get("system_cpu_percent") for row in system],
                mode="lines",
                name="system cpu",
                line=dict(color="#95a5a6", width=1.6, dash="dash"),
                hovertemplate="t=%{x:.3f}s<br>system_cpu=%{y:.1f}%<extra></extra>",
            ),
            row=4,
            col=1,
            secondary_y=True,
        )

    gpu_series = _aggregate_gpu(samples, t0, aggregate_bin_s)
    for gpu_index, records in sorted(gpu_series.items()):
        if include_gpus and gpu_index not in include_gpus:
            continue
        color = _GPU_COLORS[gpu_index % len(_GPU_COLORS)]
        x = [row["x"] for row in records]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[row.get("memory_used_gib") for row in records],
                mode="lines",
                name=f"gpu{gpu_index} memory",
                legendgroup=f"gpu{gpu_index}",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>gpu{gpu_index}</b><br>"
                    "t=%{x:.3f}s<br>"
                    "memory=%{y:.3f} GiB<extra></extra>"
                ),
            ),
            row=5,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[row.get("gpu_util_percent") for row in records],
                mode="lines",
                name=f"gpu{gpu_index} util",
                legendgroup=f"gpu{gpu_index}",
                showlegend=False,
                line=dict(color=color, width=1.3, dash="dash"),
                hovertemplate=(
                    f"<b>gpu{gpu_index}</b><br>"
                    "t=%{x:.3f}s<br>"
                    "gpu_util=%{y:.1f}%<extra></extra>"
                ),
            ),
            row=5,
            col=1,
            secondary_y=True,
        )

    if output_path is None:
        output_path = _default_output_path(nvitop_dir, "html")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    fig.update_layout(
        title=(
            f"nvitop Resource Curves · {len(samples)} samples · "
            f"{os.path.basename(nvitop_dir.rstrip(os.sep))}"
        ),
        width=width_px,
        height=height_px or 1200,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=76, r=58, t=90, b=50),
    )
    fig.update_xaxes(title_text="Time from trace start (s)", row=5, col=1)
    fig.update_yaxes(title_text="GiB", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="GiB", row=3, col=1)
    fig.update_yaxes(title_text="GiB", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="%", row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text="GiB", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text="%", row=5, col=1, secondary_y=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def plot_nvitop_png(
    nvitop_dir: str,
    output_path: str | None = None,
    *,
    include_components: set[str] | None = None,
    exclude_components: set[str] | None = None,
    include_ranks: set[int] | None = None,
    include_pids: set[int] | None = None,
    include_gpus: set[int] | None = None,
    fig_width: float = 14.0,
    dpi: int = 150,
    summary_output: str | None = None,
    aggregate_bin_s: float = 1.0,
) -> str:
    samples = _filter_samples(
        _load_samples(nvitop_dir),
        include_components=include_components,
        exclude_components=exclude_components,
        include_ranks=include_ranks,
        include_pids=include_pids,
    )
    if not samples:
        raise ValueError(f"No nvitop samples found under {nvitop_dir!r}")
    write_nvitop_summary(
        nvitop_dir,
        samples,
        include_gpus=include_gpus,
        aggregate_bin_s=aggregate_bin_s,
        output_path=summary_output,
    )

    import matplotlib.pyplot as plt

    t0 = min(rec["ts"] for rec in samples)
    process_keys = sorted(
        {_process_key(rec) for rec in samples},
        key=lambda key: (_component_sort_key(key[0]), key[1], key[2]),
    )
    colors = _process_colors(process_keys)
    by_process: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        by_process[_process_key(rec)].append(rec)
    for key in by_process:
        by_process[key].sort(key=lambda rec: rec["ts"])

    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 10), sharex=True)
    for key in process_keys:
        records = by_process[key]
        label = _process_label(key)
        color = colors[key]
        x = [rec["ts"] - t0 for rec in records]
        axes[0].plot(x, [_to_float(rec.get("process_rss_gib")) for rec in records], label=label, color=color)
        axes[1].plot(x, [_to_float(rec.get("process_cpu_percent")) for rec in records], color=color)
        gpu_mem = [_to_float(rec.get("process_gpu_memory_gib")) for rec in records]
        if any(value is not None for value in gpu_mem):
            axes[2].plot(x, gpu_mem, color=color)

    axes[0].set_title(
        f"nvitop Resource Curves · {len(samples)} samples · {os.path.basename(nvitop_dir.rstrip(os.sep))}"
    )
    axes[0].set_ylabel("RSS (GiB)")
    axes[1].set_ylabel("CPU (%)")
    axes[2].set_ylabel("Process GPU (GiB)")
    axes[2].set_xlabel("Time from trace start (s)")
    for ax in axes:
        ax.grid(axis="both", linestyle=":", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if output_path is None:
        output_path = _default_output_path(nvitop_dir, "png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot nvitop JSONL traces as CPU, RAM, and GPU resource curves."
    )
    parser.add_argument(
        "nvitop_dir",
        nargs="?",
        default="/mnt/public/zengwen/RLinf/logs/overlap/20260425-10:54:37-libero_10_ppo_openpi_pi05/nvitop",
        help="Directory containing *_rank*_pid*.jsonl nvitop files",
    )
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument(
        "--format",
        choices=["png", "html"],
        default="html",
        help="Output format: png or html",
    )
    parser.add_argument("--interactive", action="store_true", help="Alias for --format html")
    parser.add_argument("--width", type=float, default=14.0, help="PNG figure width in inches")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    parser.add_argument(
        "--include-components",
        default=None,
        help="Comma-separated component allow-list",
    )
    parser.add_argument(
        "--exclude-components",
        default=None,
        help="Comma-separated components to hide",
    )
    parser.add_argument("--include-ranks", default=None, help="Comma-separated rank allow-list")
    parser.add_argument("--include-pids", default=None, help="Comma-separated pid allow-list")
    parser.add_argument("--include-gpus", default=None, help="Comma-separated GPU index allow-list")
    parser.add_argument(
        "--aggregate-bin",
        type=float,
        default=1.0,
        help="Seconds per bucket for system/GPU global curves",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary log path (default: <nvitop_dir>/nvitop_summary.log)",
    )
    args = parser.parse_args()

    include_components = _parse_csv(args.include_components)
    exclude_components = _parse_csv(args.exclude_components)
    include_ranks = {int(value) for value in _parse_csv(args.include_ranks)}
    include_pids = {int(value) for value in _parse_csv(args.include_pids)}
    include_gpus = {int(value) for value in _parse_csv(args.include_gpus)}

    out_format = "html" if args.interactive else args.format
    if out_format == "html":
        out = plot_nvitop_html(
            args.nvitop_dir,
            output_path=args.output,
            include_components=include_components or None,
            exclude_components=exclude_components or None,
            include_ranks=include_ranks or None,
            include_pids=include_pids or None,
            include_gpus=include_gpus or None,
            aggregate_bin_s=args.aggregate_bin,
            summary_output=args.summary_output,
        )
    else:
        out = plot_nvitop_png(
            args.nvitop_dir,
            output_path=args.output,
            include_components=include_components or None,
            exclude_components=exclude_components or None,
            include_ranks=include_ranks or None,
            include_pids=include_pids or None,
            include_gpus=include_gpus or None,
            fig_width=args.width,
            dpi=args.dpi,
            summary_output=args.summary_output,
            aggregate_bin_s=args.aggregate_bin,
        )
    print(out)


if __name__ == "__main__":
    main()
