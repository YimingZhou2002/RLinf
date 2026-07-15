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


def _mem_occ_percent(mem_gib: float | None, total_gib: float | None) -> float | None:
    """Memory occupancy ratio (used / total) as a percent.

    Distinct from NVML's ``memory_util_percent`` (memory-controller busy ratio):
    this is how full the device's memory actually got. Returns ``None`` when
    either side is missing or the total is non-positive.
    """
    if mem_gib is None or not total_gib or total_gib <= 0:
        return None
    return mem_gib / total_gib * 100.0


def _summary_stats(values: list[Any]) -> dict[str, float | None]:
    cleaned = _clean_values(values)
    if not cleaned:
        return {"avg": None, "max": None, "min": None}
    return {
        "avg": mean(cleaned),
        "max": max(cleaned),
        "min": min(cleaned),
    }


def _compute_nvitop_summary(
    samples: list[dict[str, Any]],
    *,
    include_gpus: set[int] | None = None,
    aggregate_bin_s: float = 1.0,
    source_dir: str | None = None,
) -> dict[str, Any]:
    """Aggregate raw nvitop samples into a structured, machine-readable dict.

    This is the single source of truth for the nvitop summary: both the
    human-readable ``nvitop_summary.log`` (via :func:`write_nvitop_summary`)
    and the ``nvitop_summary.json`` sidecar are derived from this dict so
    they can never drift apart. The auto-tuner parser
    (:func:`toolkits.embodied_tuner.parser._load_memory_summary`) reads the
    JSON sidecar directly instead of re-parsing the text log.

    Returns a dict with ``samples`` / ``span_s`` / ``aggregate_bin_s`` /
    ``gpu_total_gib`` (device cap from ``memory_total_gib``) / ``per_gpu``
    (per-GPU avg+max mem & util, plus each GPU's ``memory_total_gib`` and the
    occupancy ratios ``avg_mem_occ_percent`` / ``max_mem_occ_percent`` — the
    used/total memory ratio, distinct from NVML's controller-busy
    ``mem_util``) / ``overall`` / ``active`` (each also carrying
    ``avg_mem_occ_percent`` / ``max_mem_occ_percent``) / ``per_process``
    (per-(component,rank,pid) avg+max rss / cpu / process_gpu_mem /
    process_gpu_util + ``gpu_indices``).
    """
    if not samples:
        raise ValueError("Cannot compute nvitop summary without samples")

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

    # Device cap (per-GPU memory_total_gib is a constant for the run).
    # Take the max across every GPU seen in the first sample that carries
    # any; hetero setups get the largest device's cap (conservative for the
    # peak/cap soft-pressure ratio).
    gpu_total_gib: float | None = None
    for rec in samples:
        for gpu in rec.get("gpus", []) or []:
            total = _to_float(gpu.get("memory_total_gib"))
            if total is None and "memory_total_bytes" in gpu:
                total = float(gpu["memory_total_bytes"]) / (2**30)
            if total is not None and (gpu_total_gib is None or total > gpu_total_gib):
                gpu_total_gib = total
        if gpu_total_gib is not None:
            break

    per_gpu: list[dict[str, Any]] = []
    all_gpu_mem: list[Any] = []
    all_gpu_util: list[Any] = []
    all_gpu_mem_util: list[Any] = []
    for gpu_index, records in sorted(gpu_series.items()):
        mem_stats = _summary_stats([row.get("memory_used_gib") for row in records])
        util_stats = _summary_stats([row.get("gpu_util_percent") for row in records])
        mem_util_stats = _summary_stats(
            [row.get("memory_util_percent") for row in records]
        )
        all_gpu_mem.extend(row.get("memory_used_gib") for row in records)
        all_gpu_util.extend(row.get("gpu_util_percent") for row in records)
        all_gpu_mem_util.extend(row.get("memory_util_percent") for row in records)
        per_gpu.append(
            {
                "index": gpu_index,
                "avg_mem": mem_stats["avg"],
                "max_mem": mem_stats["max"],
                "avg_gpu_util": util_stats["avg"],
                "max_gpu_util": util_stats["max"],
                "avg_mem_util": mem_util_stats["avg"],
                "max_mem_util": mem_util_stats["max"],
                "memory_total_gib": gpu_total_gib,
                # Occupancy (used/total) ratios — how full the device's memory
                # actually got. NOT the same as mem_util (controller busy ratio).
                "avg_mem_occ_percent": _mem_occ_percent(mem_stats["avg"], gpu_total_gib),
                "max_mem_occ_percent": _mem_occ_percent(mem_stats["max"], gpu_total_gib),
            }
        )

    overall = _summary_stats(all_gpu_mem)
    overall_util = _summary_stats(all_gpu_util)
    overall_mem_util = _summary_stats(all_gpu_mem_util)
    # Occupancy (used/total) ratios pooled across every (gpu, bin) sample.
    all_gpu_mem_occ = [
        _mem_occ_percent(v, gpu_total_gib) for v in all_gpu_mem
    ]
    overall_mem_occ = _summary_stats(all_gpu_mem_occ)
    overall_summary = {
        "avg_mem": overall["avg"],
        "max_mem": overall["max"],
        "avg_gpu_util": overall_util["avg"],
        "max_gpu_util": overall_util["max"],
        "avg_mem_util": overall_mem_util["avg"],
        "max_mem_util": overall_mem_util["max"],
        "avg_mem_occ_percent": overall_mem_occ["avg"],
        "max_mem_occ_percent": overall_mem_occ["max"],
    }

    active_gpu_mem: list[Any] = []
    active_gpu_util: list[Any] = []
    active_gpu_mem_util: list[Any] = []
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
    # Occupancy (used/total) ratios pooled across active (gpu, bin) samples.
    active_gpu_mem_occ = [
        _mem_occ_percent(v, gpu_total_gib) for v in active_gpu_mem
    ]
    active_mem_occ = _summary_stats(active_gpu_mem_occ)
    active_summary = {
        "avg_mem": active_mem["avg"],
        "max_mem": active_mem["max"],
        "avg_gpu_util": active_util["avg"],
        "max_gpu_util": active_util["max"],
        "avg_mem_util": active_mem_util["avg"],
        "max_mem_util": active_mem_util["max"],
        "avg_mem_occ_percent": active_mem_occ["avg"],
        "max_mem_occ_percent": active_mem_occ["max"],
    }

    per_process: list[dict[str, Any]] = []
    for key in process_keys:
        records = by_process[key]
        component, rank, pid = key
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
        per_process.append(
            {
                "label": _process_label(key),
                "component": component,
                "rank": rank,
                "pid": pid,
                "avg_rss": rss_stats["avg"],
                "max_rss": rss_stats["max"],
                "avg_cpu": cpu_stats["avg"],
                "max_cpu": cpu_stats["max"],
                "avg_process_gpu_mem": proc_gpu_stats["avg"],
                "max_process_gpu_mem": proc_gpu_stats["max"],
                "avg_process_gpu_util": proc_gpu_util_stats["avg"],
                "max_process_gpu_util": proc_gpu_util_stats["max"],
                "gpu_indices": gpu_indices,
            }
        )

    return {
        "source_dir": source_dir,
        "samples": len(samples),
        "span_s": t1 - t0,
        "aggregate_bin_s": aggregate_bin_s,
        "gpu_total_gib": gpu_total_gib,
        "per_gpu": per_gpu,
        "overall": overall_summary,
        "active": active_summary,
        "per_process": per_process,
    }


def _summary_to_log_lines(summary: dict[str, Any]) -> list[str]:
    """Format a :func:`_compute_nvitop_summary` dict as the ``.log`` text.

    Mirrors the legacy inline formatting; the ``avg_mem_occ`` / ``max_mem_occ``
    fields are appended after ``max_mem_util`` on each GPU and overall line.
    They report the used/total occupancy ratio (how full device memory got),
    distinct from NVML's ``mem_util`` (memory-controller busy ratio).
    """
    def _f(value: Any, unit: str = "") -> str:
        return _fmt(value, unit)

    lines = [
        "nvitop resource summary",
        f"source_dir: {summary.get('source_dir') or ''}",
        f"samples: {summary['samples']}",
        f"span_s: {summary['span_s']:.3f}",
        f"aggregate_bin_s: {summary['aggregate_bin_s']:.3f}",
        "",
        "global_gpu_summary:",
    ]
    for gpu in summary["per_gpu"]:
        lines.append(
            "  "
            f"gpu{gpu['index']}: "
            f"avg_mem={_f(gpu['avg_mem'], 'GiB')}, "
            f"max_mem={_f(gpu['max_mem'], 'GiB')}, "
            f"avg_gpu_util={_f(gpu['avg_gpu_util'], '%')}, "
            f"max_gpu_util={_f(gpu['max_gpu_util'], '%')}, "
            f"avg_mem_util={_f(gpu['avg_mem_util'], '%')}, "
            f"max_mem_util={_f(gpu['max_mem_util'], '%')}, "
            f"avg_mem_occ={_f(gpu.get('avg_mem_occ_percent'), '%')}, "
            f"max_mem_occ={_f(gpu.get('max_mem_occ_percent'), '%')}"
        )
    overall = summary["overall"]
    active = summary["active"]
    lines.extend(
        [
            "  overall_across_selected_gpus: "
            f"avg_mem={_f(overall['avg_mem'], 'GiB')}, "
            f"max_mem={_f(overall['max_mem'], 'GiB')}, "
            f"avg_gpu_util={_f(overall['avg_gpu_util'], '%')}, "
            f"max_gpu_util={_f(overall['max_gpu_util'], '%')}, "
            f"avg_mem_util={_f(overall['avg_mem_util'], '%')}, "
            f"max_mem_util={_f(overall['max_mem_util'], '%')}, "
            f"avg_mem_occ={_f(overall.get('avg_mem_occ_percent'), '%')}, "
            f"max_mem_occ={_f(overall.get('max_mem_occ_percent'), '%')}",
            "  overall_active_gpus: "
            f"avg_mem={_f(active['avg_mem'], 'GiB')}, "
            f"max_mem={_f(active['max_mem'], 'GiB')}, "
            f"avg_gpu_util={_f(active['avg_gpu_util'], '%')}, "
            f"max_gpu_util={_f(active['max_gpu_util'], '%')}, "
            f"avg_mem_util={_f(active['avg_mem_util'], '%')}, "
            f"max_mem_util={_f(active['max_mem_util'], '%')}, "
            f"avg_mem_occ={_f(active.get('avg_mem_occ_percent'), '%')}, "
            f"max_mem_occ={_f(active.get('max_mem_occ_percent'), '%')}",
            "",
            "process_summary:",
        ]
    )
    for proc in summary["per_process"]:
        gpu_indices = proc["gpu_indices"]
        lines.append(
            "  "
            f"{proc['label']}: "
            f"avg_rss={_f(proc['avg_rss'], 'GiB')}, "
            f"max_rss={_f(proc['max_rss'], 'GiB')}, "
            f"avg_cpu={_f(proc['avg_cpu'], '%')}, "
            f"max_cpu={_f(proc['max_cpu'], '%')}, "
            f"avg_process_gpu_mem={_f(proc['avg_process_gpu_mem'], 'GiB')}, "
            f"max_process_gpu_mem={_f(proc['max_process_gpu_mem'], 'GiB')}, "
            f"avg_process_gpu_util={_f(proc['avg_process_gpu_util'], '%')}, "
            f"max_process_gpu_util={_f(proc['max_process_gpu_util'], '%')}, "
            f"gpu_indices={gpu_indices or 'n/a'}"
        )
    return lines


def write_nvitop_summary(
    nvitop_dir: str,
    samples: list[dict[str, Any]],
    *,
    include_gpus: set[int] | None = None,
    aggregate_bin_s: float = 1.0,
    output_path: str | None = None,
    write_json_sidecar: bool = True,
) -> str:
    """Write ``nvitop_summary.log`` (and a ``nvitop_summary.json`` sidecar).

    The structured sidecar is the machine-readable twin of the text log —
    both produced from :func:`_compute_nvitop_summary` so they cannot drift.
    The auto-tuner reads the JSON sidecar; humans read the log.
    """
    summary = _compute_nvitop_summary(
        samples,
        include_gpus=include_gpus,
        aggregate_bin_s=aggregate_bin_s,
        source_dir=nvitop_dir,
    )

    if output_path is None:
        output_path = os.path.join(nvitop_dir, "nvitop_summary.log")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    lines = _summary_to_log_lines(summary)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if write_json_sidecar:
        log_path = os.path.abspath(output_path)
        if log_path.endswith(".log"):
            json_path = log_path[:-4] + ".json"
        else:
            json_path = log_path + ".json"
        # Strip runtime-only fields (source_dir is absolute) before dumping
        # so the sidecar is stable across machines. Keep everything else.
        sidecar = {k: v for k, v in summary.items() if k != "source_dir"}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, sort_keys=True)
            f.write("\n")

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
