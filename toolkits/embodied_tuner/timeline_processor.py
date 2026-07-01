# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Higher-level timeline analyses consumed by the Codex critic prompt.

The existing :mod:`toolkits.embodied_tuner.parser` already produces
per-rank :class:`TagStats` and per-component stall fractions from
``timeline/*.jsonl``. Those signals describe *what each component spent
time doing* but not *why the step took as long as it did*. This module
adds four orthogonal views, derived from feedback that the critic was
making placement decisions without enough structural evidence:

1. ``compute_critical_path`` (A')
   Per-``global_step`` summary of which ``(component, rank)`` lanes did
   the most REAL work versus how long they spent BLOCKED waiting on
   another component. ``actor/recv_traj`` looks busy in the raw timeline
   but is actually the actor waiting for rollout's trajectories — under
   the hybrid placement (actor on every GPU, env/rollout sharing
   sub-ranges) actor only does real GPU work during sync / forward /
   backward / optimizer / compute_adv steps, not during ``recv_traj``.
   :data:`BLOCKING_TAGS` enumerates the tag names that count as
   blocking-wait so the critical-path view excludes them from
   "real busy" totals.

2. ``compute_outliers`` (C')
   The longest events whose duration exceeds the per-tag P95 *and*
   exceeds 1s. Each outlier carries a ``knob_hint`` derived from the
   current ``enable_offload`` flags so the critic can see "this 45s
   ``env_interact_step`` was during warmup AND ``env.enable_offload``
   was true — flipping the flag should remove this stall."

3. ``compute_per_gpu_bubble`` (D')
   GPU-by-GPU breakdown under the current ``cluster.component_placement``.
   For each physical GPU id we union the REAL-busy intervals from every
   component whose rank-on-this-GPU is doing real work and report the
   complementary bubble fraction. The output includes per-GPU detail
   *and* env-side / rollout-side averages so the critic can answer
   "should I shift GPU budget from one side to the other?" directly.

4. ``extract_raw_excerpts``
   Top-K longest raw events copied verbatim from the JSONL stream so
   Codex can inspect the full call context (``qualname``, ``call_index``,
   ``configured_*`` fields, etc.) and reason about anomalies the
   aggregated summaries alone cannot explain.

The public entry point :func:`process_timeline` loads ``timeline_dir``
and runs all four. The result is a JSON-serialisable dict that the
parser splices into :class:`TimelineSummary` and the scheduler dumps
verbatim into the ledger's ``timeline_summary`` payload, which then
flows back into the next critic prompt via
:func:`toolkits.embodied_tuner.critic.build_prompt`.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from glob import glob
from pathlib import Path
from typing import Any

from toolkits.embodied_tuner.placement_enum import (
    PlacementParseError,
    parse_range_spec,
)


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------


# Tags that look like busy intervals in the timeline but are actually a
# component blocking-on-another-component. They MUST be excluded from
# "real busy" totals in A' and D', otherwise the critic sees actor as
# busy 97% of step time when in fact actor is mostly idle waiting for
# rollout/env to finish producing trajectories.
#
# Mapping: ``component -> {tag, ...}``. Tags not in this map count as
# real work. Keep this list short and explicit; widening it is a domain
# decision and should be reviewed by a human (per BL note in the plan).
BLOCKING_TAGS: dict[str, frozenset[str]] = {
    "actor": frozenset({
        "actor/recv_traj",
        "actor/recv_rollout_trajectories",
    }),
    "rollout": frozenset({
        # `generate*` wrap the per-chunk `predict` events; counting both
        # would double-count rollout time.
        "rollout/generate",
        "rollout/generate_one_epoch",
        "recv_rollout_results",
    }),
    "env": frozenset({
        # `interact` / `run_interact_once` wrap the per-step
        # `env_interact_step` events.
        "interact",
        "run_interact_once",
    }),
}


# Number of top-K outliers and raw excerpts to keep. Bumping these
# inflates every critic prompt token-for-token; the defaults are tuned
# so the four new sections together stay under ~3 KB.
DEFAULT_OUTLIER_K: int = 12
DEFAULT_RAW_EXCERPTS_K: int = 15
OUTLIER_MIN_SECONDS: float = 1.0  # ignore sub-second "outliers"


# Fields kept on raw-excerpt records. The full JSONL line can be very
# long (>200 chars when ``configured_*`` is present); we drop the ones
# the critic does not reason about to keep the prompt compact.
_RAW_EXCERPT_FIELDS: tuple[str, ...] = (
    "t0", "t1", "component", "rank", "tag", "global_step",
    "qualname", "call_index", "rollout_epoch", "chunk_step",
    "stage_id", "phase",
)


# ---------------------------------------------------------------------------
# Loading + classification
# ---------------------------------------------------------------------------


def _load_events(timeline_dir: Path | str) -> list[dict]:
    """Return every JSONL event under ``timeline_dir``.

    Best-effort: malformed lines and unreadable files are skipped. Each
    event has ``t0 <= t1`` and a derived ``dur`` field.
    """
    events: list[dict] = []
    for path in sorted(glob(f"{Path(timeline_dir)}/*.jsonl")):
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        try:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    t0 = float(rec["t0"]); t1 = float(rec["t1"])
                except (KeyError, TypeError, ValueError):
                    continue
                if t1 < t0:
                    t0, t1 = t1, t0
                rec["t0"], rec["t1"] = t0, t1
                rec["dur"] = t1 - t0
                events.append(rec)
        finally:
            fh.close()
    return events


def is_blocking(event: Mapping[str, Any]) -> bool:
    """Return ``True`` when ``event`` represents blocking-wait, not real work."""
    component = event.get("component")
    tag = event.get("tag")
    if component is None or tag is None:
        return False
    return tag in BLOCKING_TAGS.get(component, frozenset())


# ---------------------------------------------------------------------------
# Placement parsing
# ---------------------------------------------------------------------------


def placement_to_gpu_map(
    placement: Mapping[str, Any],
    num_gpus: int = 8,
) -> dict[str, dict[int, int]]:
    """Resolve ``cluster.component_placement`` into ``{component: {rank: gpu}}``.

    Accepts the YAML-native form where each value is either a range
    string (``"0-3"``, ``"all"``, ``"0,2,4-6"``) or already a list of
    GPU ids. Components missing from ``placement`` are returned empty.
    Unparseable entries are skipped — production callers should treat
    a None/empty result as "no per-GPU view available".
    """
    out: dict[str, dict[int, int]] = {}
    for component, value in placement.items():
        gpus: tuple[int, ...]
        if isinstance(value, str):
            try:
                gpus = parse_range_spec(value, num_gpus=num_gpus)
            except (PlacementParseError, ValueError):
                continue
        elif isinstance(value, (list, tuple)):
            try:
                gpus = tuple(int(g) for g in value)
            except (TypeError, ValueError):
                continue
        else:
            continue
        out[component] = {rank: gpu for rank, gpu in enumerate(gpus)}
    return out


# ---------------------------------------------------------------------------
# A' — Critical path per global_step
# ---------------------------------------------------------------------------


def compute_critical_path(
    events: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> dict[int, dict[str, Any]]:
    """Per-``global_step`` real-busy vs blocked busy summary.

    Returns ``{global_step: {step_span_s, real_busy_top: [...]}}``.
    Events without a ``global_step`` (bootstrap warmup) are ignored;
    the bootstrap stalls show up in :func:`compute_outliers` instead.

    The ``real_busy_top`` list ranks ``(component, rank)`` lanes by
    REAL busy seconds (blocking tags excluded). Each entry also reports
    ``blocked_s`` so the critic can see "rollout did 165s real work
    while actor was blocked on it for 268s" without inferring that
    relationship from raw event counts.
    """
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for event in events:
        gs = event.get("global_step")
        if gs is None:
            continue
        try:
            by_step.setdefault(int(gs), []).append(event)
        except (TypeError, ValueError):
            continue

    out: dict[int, dict[str, Any]] = {}
    for gs in sorted(by_step):
        evs = by_step[gs]
        if not evs:
            continue
        t0 = min(float(e["t0"]) for e in evs)
        t1 = max(float(e["t1"]) for e in evs)
        span = t1 - t0
        real: dict[tuple[str, int], float] = {}
        blocked: dict[tuple[str, int], float] = {}
        for event in evs:
            key = (str(event["component"]), int(event["rank"]))
            target = blocked if is_blocking(event) else real
            target[key] = target.get(key, 0.0) + float(event["dur"])
        ranked = sorted(real.items(), key=lambda kv: -kv[1])
        out[gs] = {
            "step_span_s": round(span, 1),
            "real_busy_top": [
                {
                    "component": c,
                    "rank": r,
                    "real_s": round(s, 1),
                    "blocked_s": round(blocked.get((c, r), 0.0), 1),
                    "real_frac": round(s / span, 2) if span > 0 else 0.0,
                }
                for (c, r), s in ranked[:top_k]
            ],
        }
    return out


# ---------------------------------------------------------------------------
# C' — Outliers with knob hint
# ---------------------------------------------------------------------------


def _knob_hint(
    event: Mapping[str, Any],
    enable_offload: Mapping[str, bool] | None,
) -> str | None:
    """Map an outlier event to the knob most likely responsible.

    The rules below were derived from BL feedback: long bootstrap-stage
    ``env_interact_step`` events on configs with ``env.enable_offload=True``
    are explained by the offload page-in/out cost. Similar reasoning
    holds for rollout-side and actor-side offload. The hint is a hint,
    not a verdict — the critic still cites it and reasons.
    """
    if not enable_offload:
        return None
    tag = event.get("tag", "") or ""
    component = event.get("component", "") or ""
    gs = event.get("global_step")
    if component == "env" and gs is None and enable_offload.get("env"):
        return "env.enable_offload=True → warmup paged GPU<->CPU"
    if component == "rollout" and tag in {"predict", "rollout/generate"} \
            and enable_offload.get("rollout"):
        return "rollout.enable_offload=True → weight/activation reload"
    if component == "actor" and (
        tag.startswith("actor_") or tag == "actor/sync_model_to_rollout"
    ) and enable_offload.get("actor"):
        return "actor.enable_offload=True → optimizer/grad page-in"
    return None


def compute_outliers(
    events: Sequence[Mapping[str, Any]],
    *,
    enable_offload: Mapping[str, bool] | None = None,
    top_k: int = DEFAULT_OUTLIER_K,
    min_seconds: float = OUTLIER_MIN_SECONDS,
) -> tuple[dict[str, Any], ...]:
    """Return the longest events above the per-tag P95 and ``min_seconds``."""
    by_tag: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        tag = event.get("tag")
        if tag is None:
            continue
        by_tag.setdefault(str(tag), []).append(event)

    outliers: list[Mapping[str, Any]] = []
    for tag, evs in by_tag.items():
        # Need a handful of samples for P95 to be meaningful.
        if len(evs) < 10:
            continue
        durs = sorted(float(e["dur"]) for e in evs)
        p95 = durs[int(len(durs) * 0.95)]
        for event in evs:
            if float(event["dur"]) > p95 and float(event["dur"]) > min_seconds:
                outliers.append(event)
    outliers.sort(key=lambda e: -float(e["dur"]))

    enriched: list[dict[str, Any]] = []
    for event in outliers[:top_k]:
        enriched.append({
            "tag": event.get("tag"),
            "component": event.get("component"),
            "rank": event.get("rank"),
            "global_step": event.get("global_step"),
            "dur_s": round(float(event["dur"]), 2),
            "knob_hint": _knob_hint(event, enable_offload),
        })
    return tuple(enriched)


# ---------------------------------------------------------------------------
# D' — Per-GPU bubble under placement
# ---------------------------------------------------------------------------


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Union-merge a list of half-open intervals. Stable in input order."""
    sorted_iv = sorted(intervals)
    out: list[list[float]] = []
    for s, t in sorted_iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], t)
        else:
            out.append([s, t])
    return [(s, t) for s, t in out]


def compute_per_gpu_bubble(
    events: Sequence[Mapping[str, Any]],
    *,
    placement: Mapping[str, Any] | None,
    num_gpus: int = 8,
) -> dict[str, Any]:
    """GPU-by-GPU bubble breakdown under ``placement``.

    Returns ``{}`` when ``placement`` is None or no events match. The
    output schema includes per-GPU detail *and* env-side / rollout-side
    averages so the critic can read both granularities without rerunning
    the analysis.
    """
    if not placement:
        return {}
    gpu_map = placement_to_gpu_map(placement, num_gpus=num_gpus)
    if not gpu_map:
        return {}

    intervals_by_gpu: dict[int, list[tuple[float, float]]] = {}
    for event in events:
        if is_blocking(event):
            continue
        component = event.get("component")
        rank = event.get("rank")
        if component not in gpu_map:
            continue
        try:
            rank_int = int(rank)
        except (TypeError, ValueError):
            continue
        gpu = gpu_map[component].get(rank_int)
        if gpu is None:
            continue
        intervals_by_gpu.setdefault(gpu, []).append(
            (float(event["t0"]), float(event["t1"]))
        )

    if not events:
        return {}
    t_min = min(float(e["t0"]) for e in events)
    t_max = max(float(e["t1"]) for e in events)
    total = t_max - t_min
    if total <= 0:
        return {}

    per_gpu: dict[int, dict[str, Any]] = {}
    all_gpus = set(intervals_by_gpu)
    for component, mapping in gpu_map.items():
        all_gpus.update(mapping.values())
    for gpu in sorted(all_gpus):
        busy = sum(t - s for s, t in _merge_intervals(intervals_by_gpu.get(gpu, [])))
        bubble = max(total - busy, 0.0)
        residents = sorted(
            comp for comp, mapping in gpu_map.items() if gpu in mapping.values()
        )
        per_gpu[gpu] = {
            "residents": residents,
            "busy_s": round(busy, 1),
            "bubble_s": round(bubble, 1),
            "bubble_frac": round(bubble / total, 2),
        }

    # env-side = the union of GPUs where env lives; rollout-side = where
    # rollout lives. These two sets may overlap (collocated placement);
    # we report the average bubble across the union so the critic gets a
    # single comparable number per side.
    env_gpus = set(gpu_map.get("env", {}).values())
    rollout_gpus = set(gpu_map.get("rollout", {}).values())
    env_side_avg = (
        sum(per_gpu[g]["bubble_s"] for g in env_gpus) / len(env_gpus)
        if env_gpus else None
    )
    rollout_side_avg = (
        sum(per_gpu[g]["bubble_s"] for g in rollout_gpus) / len(rollout_gpus)
        if rollout_gpus else None
    )
    return {
        "wall_s": round(total, 1),
        "per_gpu": {str(g): info for g, info in per_gpu.items()},
        "env_side_avg_bubble_s": (
            round(env_side_avg, 1) if env_side_avg is not None else None
        ),
        "rollout_side_avg_bubble_s": (
            round(rollout_side_avg, 1) if rollout_side_avg is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Raw excerpts
# ---------------------------------------------------------------------------


def extract_raw_excerpts(
    events: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_RAW_EXCERPTS_K,
) -> tuple[dict[str, Any], ...]:
    """Top-K longest events, projected onto a short field set.

    Only fields the critic actually reasons about (declared in
    :data:`_RAW_EXCERPT_FIELDS`) are copied through. Keeps every
    excerpt under ~200 chars when serialised.

    Events from the ``runner`` component are dropped — its single
    ``run`` event spans the whole trial and would otherwise dominate
    the top-K list with zero signal value.
    """
    filtered = [e for e in events if e.get("component") != "runner"]
    ordered = sorted(filtered, key=lambda e: -float(e.get("dur", 0.0)))
    out: list[dict[str, Any]] = []
    for event in ordered[:top_k]:
        excerpt: dict[str, Any] = {}
        for field in _RAW_EXCERPT_FIELDS:
            if field in event:
                excerpt[field] = event[field]
        if "dur" not in excerpt and "t0" in excerpt and "t1" in excerpt:
            excerpt["dur_s"] = round(float(event["dur"]), 3)
        else:
            excerpt["dur_s"] = round(float(event.get("dur", 0.0)), 3)
        out.append(excerpt)
    return tuple(out)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def process_timeline(
    timeline_dir: Path | str,
    *,
    placement: Mapping[str, Any] | None = None,
    enable_offload: Mapping[str, bool] | None = None,
    num_gpus: int = 8,
) -> dict[str, Any]:
    """Run all four analyses and return a JSON-serialisable dict.

    Empty input → ``{}``. Missing ``placement`` skips the per-GPU
    bubble view (the critic falls back to the existing
    ``stall_fraction_by_component`` signal). Missing ``enable_offload``
    means outliers ship without a ``knob_hint``.
    """
    events = _load_events(timeline_dir)
    if not events:
        return {}
    return {
        "critical_path": compute_critical_path(events),
        "outliers": list(compute_outliers(events, enable_offload=enable_offload)),
        "per_gpu_bubble": compute_per_gpu_bubble(
            events, placement=placement, num_gpus=num_gpus
        ),
        "raw_excerpts": list(extract_raw_excerpts(events)),
    }
