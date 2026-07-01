"""
Plot NVML JSONL traces as per-process GPU memory curves.
"""
from __future__ import annotations

import os
import sys

if sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import json
from collections import defaultdict
from glob import glob
from typing import Any


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


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _component_sort_key(component: str) -> tuple[int, str]:
    return (_COMPONENT_ORDER.get(component, 999), component)


def _load_samples(nvml_dir: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    pattern = os.path.join(nvml_dir, "*.jsonl")
    for path in sorted(glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_path"] = path
                rec["ts"] = float(rec["ts"])
                rec["nvml_total_gib"] = float(
                    rec.get(
                        "nvml_total_used_gib",
                        float(rec.get("nvml_total_used_bytes", 0.0)) / (2**30),
                    )
                )
                if "torch_allocated_bytes" in rec:
                    rec["torch_allocated_gib"] = float(rec["torch_allocated_bytes"]) / (
                        2**30
                    )
                else:
                    rec["torch_allocated_gib"] = None
                if "torch_reserved_bytes" in rec:
                    rec["torch_reserved_gib"] = float(rec["torch_reserved_bytes"]) / (
                        2**30
                    )
                else:
                    rec["torch_reserved_gib"] = None
                samples.append(rec)
    samples.sort(key=lambda r: (r["ts"], str(r.get("component", "")), int(r.get("rank", 0))))
    return samples


def _apply_nvml_scale(samples: list[dict[str, Any]], scale: float) -> None:
    if scale == 1.0:
        return
    for rec in samples:
        rec["nvml_total_gib"] *= scale


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
    return (str(rec.get("component", "")), int(rec.get("rank", 0)), int(rec.get("pid", 0)))


def _process_label(key: tuple[str, int, int]) -> str:
    component, rank, pid = key
    return f"{component}/r{rank}/pid{pid}"


def _build_color_map(keys: list[tuple[str, int, int]]) -> dict[tuple[str, int, int], str]:
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
    ordered = sorted(
        keys,
        key=lambda key: (_component_sort_key(key[0]), key[1], key[2]),
    )
    next_palette = 0
    for key in ordered:
        component = key[0]
        base = _COMPONENT_COLORS.get(component)
        if base and component not in {k[0] for k in colors}:
            colors[key] = base
        else:
            colors[key] = palette[next_palette % len(palette)]
            next_palette += 1
    return colors


def _default_output_path(nvml_dir: str, out_format: str) -> str:
    ext = "html" if out_format == "html" else "png"
    return os.path.join(nvml_dir, f"nvml_memory.{ext}")


def plot_nvml_html(
    nvml_dir: str,
    output_path: str | None = None,
    *,
    include_components: set[str] | None = None,
    exclude_components: set[str] | None = None,
    include_ranks: set[int] | None = None,
    include_pids: set[int] | None = None,
    width_px: int = 1300,
    height_px: int | None = None,
    nvml_scale: float = 1.0,
) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotly is required for --format html. Install it with: pip install plotly"
        ) from e

    samples = _load_samples(nvml_dir)
    _apply_nvml_scale(samples, nvml_scale)
    samples = _filter_samples(
        samples,
        include_components=include_components,
        exclude_components=exclude_components,
        include_ranks=include_ranks,
        include_pids=include_pids,
    )
    if not samples:
        raise ValueError(f"No NVML samples found under {nvml_dir!r}")

    t0 = min(rec["ts"] for rec in samples)
    process_keys = sorted({_process_key(rec) for rec in samples}, key=lambda key: (_component_sort_key(key[0]), key[1], key[2]))
    colors = _build_color_map(process_keys)

    by_process: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        by_process[_process_key(rec)].append(rec)
    for key in by_process:
        by_process[key].sort(key=lambda rec: rec["ts"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Process NVML Total Memory", "PyTorch Allocated / Reserved"),
    )

    for key in process_keys:
        trace_samples = by_process[key]
        label = _process_label(key)
        color = colors[key]
        x = [rec["ts"] - t0 for rec in trace_samples]
        nvml_gib = [rec["nvml_total_gib"] for rec in trace_samples]
        global_steps = [rec.get("global_step") for rec in trace_samples]
        backend = [rec.get("backend") for rec in trace_samples]
        device_list = [
            ",".join(str(device.get("gpu_index")) for device in rec.get("devices", []))
            for rec in trace_samples
        ]
        worker_name = [rec.get("worker_name") for rec in trace_samples]

        hover_rows = list(
            zip(
                global_steps,
                backend,
                device_list,
                worker_name,
                nvml_gib,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=nvml_gib,
                mode="lines",
                name=label,
                legendgroup=label,
                line=dict(color=color, width=2),
                customdata=hover_rows,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "t=%{x:.3f}s<br>"
                    "nvml_total=%{y:.3f} GiB<br>"
                    "global_step=%{customdata[0]}<br>"
                    "backend=%{customdata[1]}<br>"
                    "gpu_indices=%{customdata[2]}<br>"
                    "worker_name=%{customdata[3]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        torch_alloc = [rec.get("torch_allocated_gib") for rec in trace_samples]
        torch_reserved = [rec.get("torch_reserved_gib") for rec in trace_samples]
        has_torch = any(value is not None for value in torch_alloc) or any(
            value is not None for value in torch_reserved
        )
        if not has_torch:
            continue

        fig.add_trace(
            go.Scatter(
                x=x,
                y=torch_alloc,
                mode="lines",
                name=f"{label} allocated",
                legendgroup=label,
                showlegend=False,
                line=dict(color=color, width=1.8),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "t=%{x:.3f}s<br>"
                    "torch_allocated=%{y:.3f} GiB<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=torch_reserved,
                mode="lines",
                name=f"{label} reserved",
                legendgroup=label,
                showlegend=False,
                line=dict(color=color, width=1.4, dash="dash"),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "t=%{x:.3f}s<br>"
                    "torch_reserved=%{y:.3f} GiB<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    if output_path is None:
        output_path = _default_output_path(nvml_dir, "html")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    fig.update_layout(
        title=(
            f"NVML Memory Curves · {len(samples)} samples · "
            f"{os.path.basename(nvml_dir.rstrip(os.sep))}"
        ),
        width=width_px,
        height=height_px or 900,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=20, t=80, b=50),
    )
    fig.update_xaxes(title_text="Time from trace start (s)", row=2, col=1)
    fig.update_yaxes(title_text="GiB", row=1, col=1)
    fig.update_yaxes(title_text="GiB", row=2, col=1)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def plot_nvml_png(
    nvml_dir: str,
    output_path: str | None = None,
    *,
    include_components: set[str] | None = None,
    exclude_components: set[str] | None = None,
    include_ranks: set[int] | None = None,
    include_pids: set[int] | None = None,
    fig_width: float = 14.0,
    dpi: int = 150,
    nvml_scale: float = 1.0,
) -> str:
    import matplotlib.pyplot as plt

    samples = _load_samples(nvml_dir)
    _apply_nvml_scale(samples, nvml_scale)
    samples = _filter_samples(
        samples,
        include_components=include_components,
        exclude_components=exclude_components,
        include_ranks=include_ranks,
        include_pids=include_pids,
    )
    if not samples:
        raise ValueError(f"No NVML samples found under {nvml_dir!r}")

    t0 = min(rec["ts"] for rec in samples)
    process_keys = sorted({_process_key(rec) for rec in samples}, key=lambda key: (_component_sort_key(key[0]), key[1], key[2]))
    colors = _build_color_map(process_keys)

    by_process: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in samples:
        by_process[_process_key(rec)].append(rec)
    for key in by_process:
        by_process[key].sort(key=lambda rec: rec["ts"])

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    for key in process_keys:
        trace_samples = by_process[key]
        label = _process_label(key)
        color = colors[key]
        x = [rec["ts"] - t0 for rec in trace_samples]
        ax_top.plot(x, [rec["nvml_total_gib"] for rec in trace_samples], label=label, color=color, linewidth=2)

        torch_alloc = [rec.get("torch_allocated_gib") for rec in trace_samples]
        torch_reserved = [rec.get("torch_reserved_gib") for rec in trace_samples]
        if any(value is not None for value in torch_alloc):
            ax_bottom.plot(x, torch_alloc, color=color, linewidth=1.8)
        if any(value is not None for value in torch_reserved):
            ax_bottom.plot(x, torch_reserved, color=color, linewidth=1.2, linestyle="--")

    ax_top.set_title(
        f"NVML Memory Curves · {len(samples)} samples · {os.path.basename(nvml_dir.rstrip(os.sep))}"
    )
    ax_top.set_ylabel("NVML Total (GiB)")
    ax_bottom.set_ylabel("Torch Alloc/Reserved (GiB)")
    ax_bottom.set_xlabel("Time from trace start (s)")
    ax_top.grid(axis="both", linestyle=":", alpha=0.4)
    ax_bottom.grid(axis="both", linestyle=":", alpha=0.4)
    ax_top.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if output_path is None:
        output_path = _default_output_path(nvml_dir, "png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot NVML JSONL traces as per-process GPU memory curves."
    )
    parser.add_argument(
        "nvml_dir",
        nargs="?",
        default="/mnt/public/zengwen/RLinf/logs/overlap/nvml_debug",
        help="Directory containing *_rank*_pid*.jsonl NVML files",
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
    parser.add_argument(
        "--include-ranks",
        default=None,
        help="Comma-separated rank allow-list",
    )
    parser.add_argument(
        "--include-pids",
        default=None,
        help="Comma-separated pid allow-list",
    )
    parser.add_argument(
        "--nvml-scale",
        type=float,
        default=1.0,
        help="Multiply NVML GiB values by this factor when plotting",
    )
    args = parser.parse_args()

    include_components = _parse_csv(args.include_components)
    exclude_components = _parse_csv(args.exclude_components)
    include_ranks = {int(value) for value in _parse_csv(args.include_ranks)}
    include_pids = {int(value) for value in _parse_csv(args.include_pids)}

    out_format = "html" if args.interactive else args.format
    if out_format == "html":
        out = plot_nvml_html(
            args.nvml_dir,
            output_path=args.output,
            include_components=include_components or None,
            exclude_components=exclude_components or None,
            include_ranks=include_ranks or None,
            include_pids=include_pids or None,
            nvml_scale=args.nvml_scale,
        )
    else:
        out = plot_nvml_png(
            args.nvml_dir,
            output_path=args.output,
            include_components=include_components or None,
            exclude_components=exclude_components or None,
            include_ranks=include_ranks or None,
            include_pids=include_pids or None,
            fig_width=args.width,
            dpi=args.dpi,
            nvml_scale=args.nvml_scale,
        )
    print(out)


if __name__ == "__main__":
    main()
