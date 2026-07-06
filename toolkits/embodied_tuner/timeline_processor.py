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

"""Timeline analyses consumed by the Codex critic prompt.

This module owns every derivation from ``timeline/*.jsonl`` — event
loading, per-tag aggregation, stall fractions, and the four analytical
views. :mod:`toolkits.embodied_tuner.parser` is a thin adapter that
calls into this module and packs the results into
:class:`TimelineSummary`.

Views produced:

1. ``compute_stall_fractions``
   Per-component fraction of the observation window NOT covered by any
   of that component's events. Captures pipeline / channel waits at the
   coarsest level; the compact block in the critic prompt renders this.

2. ``compute_tag_stats``
   Per-``(component, rank, tag)`` call count + min/median/max/total
   duration for the headline tag list supplied by the parser.

3. ``compute_critical_path`` (A')
   Per-``global_step`` summary of which ``(component, rank)`` lanes did
   the most REAL work versus how long they spent BLOCKED waiting on
   another component. Uses :data:`BLOCKING_TAGS` to exclude wait-
   disguised-as-work events.

4. ``compute_outliers`` (C')
   The longest events whose duration exceeds the per-tag P95 *and*
   exceeds 1s. Each outlier carries a ``knob_hint`` derived from the
   current ``enable_offload`` flags.

5. ``compute_per_component_bubble`` (D')
   For each component (env/rollout/actor), the union of REAL-busy
   intervals across all its ranks and the complementary bubble fraction
   over wall time. Also reports per-rank detail so the critic can spot
   stragglers. Replaces the old per-GPU view — same signal but keyed on
   workload identity, which is what the critic actually reasons about.

6. ``extract_raw_excerpts``
   Top-K longest raw events copied verbatim so the critic can inspect
   full call context (``qualname``, ``call_index``, ``configured_*``,
   etc.) when the aggregations don't explain an anomaly.

7. ``compute_component_call_averages``
   Per-component (default: env, rollout) mean per-call duration after
   dropping the first :data:`DEFAULT_SKIP_FIRST_CALLS` events (warmup
   skew). Pools events across all ranks/tags for the component, excludes
   :data:`BLOCKING_TAGS` wrappers to avoid double-counting, sorts by
   ``t0``, then averages the tail. Steady-state per-call cost the critic
   uses when reasoning about whether a knob affected the *typical* call
   or only the warmup call.

The public entry point :func:`process_timeline` loads a timeline dir
and runs (3)/(4)/(5)/(6)/(7); the parser calls (1)/(2) directly against
the loaded events. All component-oriented views exclude the ``runner``
component — its single ``run`` event spans the whole trial and adds no
signal to per-component analysis.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from glob import glob
from pathlib import Path
from typing import Any


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


# Components excluded from every per-component aggregation. `runner`
# emits a single `run` event spanning the whole trial — it dominates
# rankings and reports 0.000 stall by construction. Dropping it makes
# the compact block and per-component bubble legible.
EXCLUDED_COMPONENTS: frozenset[str] = frozenset({"runner"})


# Number of top-K outliers and raw excerpts to keep. Bumping these
# inflates every critic prompt token-for-token; the defaults are tuned
# so the four new sections together stay under ~3 KB.
DEFAULT_OUTLIER_K: int = 12
DEFAULT_RAW_EXCERPTS_K: int = 15
OUTLIER_MIN_SECONDS: float = 1.0  # ignore sub-second "outliers"


# Components + skip count for :func:`compute_component_call_averages`.
# env and rollout are the two components whose per-call cost the critic
# actually reasons about — actor per-call cost is already captured by
# the ``per_tag`` headline stats. Skipping the first 2 calls removes the
# bootstrap warmup (offload page-in, JIT compile, first-CUDA-kernel
# init) which otherwise inflates the mean by orders of magnitude.
DEFAULT_CALL_AVERAGE_COMPONENTS: tuple[str, ...] = ("env", "rollout")
DEFAULT_SKIP_FIRST_CALLS: int = 2


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


def load_events(timeline_dir: Path | str) -> list[dict]:
    """Return every JSONL event under ``timeline_dir``, normalized.

    Each returned dict has ``t0 <= t1``, an integer ``rank``, and a
    derived ``dur = t1 - t0``. Missing ``t0/t1/component/rank/tag``
    events, malformed lines, and unreadable files are silently skipped.
    ``tag`` falls back to ``worker_timer`` when present (matches what
    ``profiler/rlinf_timeline/autopatch.py`` emits for some paths).
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
                tag = rec.get("tag") or rec.get("worker_timer")
                component = rec.get("component")
                rank = rec.get("rank")
                if tag is None or component is None or rank is None:
                    continue
                try:
                    t0 = float(rec["t0"]); t1 = float(rec["t1"])
                    rank_int = int(rank)
                except (KeyError, TypeError, ValueError):
                    continue
                if t1 < t0:
                    t0, t1 = t1, t0
                rec["t0"] = t0
                rec["t1"] = t1
                rec["tag"] = tag
                rec["component"] = component
                rec["rank"] = rank_int
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


def _is_excluded(component: Any) -> bool:
    return component in EXCLUDED_COMPONENTS


# ---------------------------------------------------------------------------
# Stall fractions (moved from parser._stall_fraction)
# ---------------------------------------------------------------------------


def compute_stall_fractions(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Per-component fraction of the observation window NOT covered by any event.

    Coarsest-grained idle signal — does NOT exclude blocking tags (that
    is what :func:`compute_per_component_bubble` is for). The compact
    block in the critic prompt renders this because it is easy to
    explain: "when is *nothing at all* happening on this component?".

    Excludes :data:`EXCLUDED_COMPONENTS`.
    """
    if not events:
        return {}
    window_start = min(float(e["t0"]) for e in events)
    window_end = max(float(e["t1"]) for e in events)
    total = window_end - window_start
    if total <= 0:
        return {}
    intervals_by_component: dict[str, list[tuple[float, float]]] = {}
    for event in events:
        component = event.get("component")
        if _is_excluded(component):
            continue
        intervals_by_component.setdefault(str(component), []).append(
            (float(event["t0"]), float(event["t1"]))
        )
    out: dict[str, float] = {}
    for component, intervals in intervals_by_component.items():
        covered = sum(t - s for s, t in _merge_intervals(intervals))
        stall = max(total - covered, 0.0)
        out[component] = stall / total
    return out


# ---------------------------------------------------------------------------
# Per-tag stats (moved from parser)
# ---------------------------------------------------------------------------


def compute_tag_stats(
    events: Sequence[Mapping[str, Any]],
    *,
    headline_tags: Iterable[str],
) -> list[dict[str, Any]]:
    """Per-``(component, rank, tag)`` aggregation for the headline tag list.

    Returns dict rows keyed by the same fields as the parser's
    :class:`TagStats` dataclass so the parser can wrap them in one pass.
    Only tags in ``headline_tags`` are summarised, to keep the critic
    prompt compact. Excludes :data:`EXCLUDED_COMPONENTS`.
    """
    wanted = frozenset(headline_tags)
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for event in events:
        component = event.get("component")
        if _is_excluded(component):
            continue
        tag = event.get("tag")
        rank = event.get("rank")
        if tag not in wanted:
            continue
        try:
            rank_int = int(rank)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(
            (str(component), rank_int, str(tag)), []
        ).append(float(event.get("dur", 0.0)))
    out: list[dict[str, Any]] = []
    for (component, rank, tag) in sorted(grouped):
        durs = grouped[(component, rank, tag)]
        if not durs:
            continue
        out.append({
            "component": component,
            "rank": rank,
            "tag": tag,
            "call_count": len(durs),
            "duration_min": min(durs),
            "duration_median": statistics.median(durs),
            "duration_max": max(durs),
            "duration_total": sum(durs),
        })
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
    Events from :data:`EXCLUDED_COMPONENTS` are dropped.

    The ``real_busy_top`` list ranks ``(component, rank)`` lanes by
    REAL busy seconds (blocking tags excluded). Each entry also reports
    ``blocked_s`` so the critic can see "rollout did 165s real work
    while actor was blocked on it for 268s" without inferring that
    relationship from raw event counts.
    """
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for event in events:
        if _is_excluded(event.get("component")):
            continue
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
# D' — Per-component bubble
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


def compute_per_component_bubble(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-component real-busy vs bubble breakdown.

    For each component (excluding :data:`EXCLUDED_COMPONENTS`, and
    excluding blocking-wait tags via :func:`is_blocking`):

    - ``busy_s`` — union of real-busy intervals across every rank of
      this component. Answers "how much wall time is *at least one*
      rank of this component doing real work?".
    - ``bubble_s`` = wall - busy_s, ``bubble_frac`` = bubble_s / wall.
      Lower bubble = component's ranks collectively covered more of
      the trial wall.
    - ``per_rank[rank]`` — same three stats per rank so the critic can
      spot a straggler (one rank with much higher busy_s than the rest).

    Returns ``{}`` when there are no events or the wall window is
    degenerate. No placement required — the "which component sits on
    which GPU" mapping is not needed to answer "is this component the
    pipeline bottleneck".
    """
    if not events:
        return {}
    t_min = min(float(e["t0"]) for e in events)
    t_max = max(float(e["t1"]) for e in events)
    wall = t_max - t_min
    if wall <= 0:
        return {}

    intervals_by_c: dict[str, list[tuple[float, float]]] = {}
    intervals_by_cr: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for event in events:
        component = event.get("component")
        if _is_excluded(component):
            continue
        if is_blocking(event):
            continue
        try:
            rank = int(event.get("rank"))
        except (TypeError, ValueError):
            continue
        iv = (float(event["t0"]), float(event["t1"]))
        intervals_by_c.setdefault(str(component), []).append(iv)
        intervals_by_cr.setdefault((str(component), rank), []).append(iv)

    if not intervals_by_c:
        return {}

    per_component: dict[str, dict[str, Any]] = {}
    for component in sorted(intervals_by_c):
        busy = sum(t - s for s, t in _merge_intervals(intervals_by_c[component]))
        bubble = max(wall - busy, 0.0)
        ranks = sorted(r for c, r in intervals_by_cr if c == component)
        per_rank: dict[str, dict[str, float]] = {}
        for rank in ranks:
            r_busy = sum(
                t - s
                for s, t in _merge_intervals(intervals_by_cr[(component, rank)])
            )
            r_bubble = max(wall - r_busy, 0.0)
            per_rank[str(rank)] = {
                "busy_s": round(r_busy, 1),
                "bubble_s": round(r_bubble, 1),
                "bubble_frac": round(r_bubble / wall, 2),
            }
        per_component[component] = {
            "num_ranks": len(ranks),
            "busy_s": round(busy, 1),
            "bubble_s": round(bubble, 1),
            "bubble_frac": round(bubble / wall, 2),
            "per_rank": per_rank,
        }
    return {
        "wall_s": round(wall, 1),
        "per_component": per_component,
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

    Events from :data:`EXCLUDED_COMPONENTS` are dropped — the runner's
    single ``run`` event spans the whole trial and would otherwise
    dominate the top-K list with zero signal value.
    """
    filtered = [e for e in events if not _is_excluded(e.get("component"))]
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
# Per-component steady-state call averages
# ---------------------------------------------------------------------------


def compute_component_call_averages(
    events: Sequence[Mapping[str, Any]],
    *,
    components: Iterable[str] = DEFAULT_CALL_AVERAGE_COMPONENTS,
    skip_first: int = DEFAULT_SKIP_FIRST_CALLS,
) -> dict[str, dict[str, Any]]:
    """Mean per-call duration per component after dropping the first N calls.

    For each component in ``components`` (default: ``env``, ``rollout``):

    - Collect every event of that component EXCLUDING :data:`BLOCKING_TAGS`
      wrappers (`interact` / `run_interact_once` / `rollout/generate` /
      `rollout/generate_one_epoch` — counting these alongside their
      per-step children would double-count).
    - Also excludes :data:`EXCLUDED_COMPONENTS`.
    - Pool across all ranks and tags of the component, sort by ``t0``,
      drop the first ``skip_first`` events, average ``dur`` over the
      remainder.

    Skipping the first ``skip_first`` calls filters out bootstrap warmup
    (offload page-in, JIT compile, first-CUDA-kernel init) that would
    otherwise dominate the mean. The critic uses this to see the
    *steady-state* per-call cost — i.e. whether a knob affected the
    typical call or only the warmup call.

    Returns ``{component: {call_count_total, skipped, remaining_count,
    mean_duration_s, min_duration_s, max_duration_s}}``. Components with
    ``<= skip_first`` matching events are omitted (nothing to average).
    """
    requested = frozenset(components)
    per_component: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        component = event.get("component")
        if _is_excluded(component) or component not in requested:
            continue
        if is_blocking(event):
            continue
        per_component.setdefault(str(component), []).append(event)

    out: dict[str, dict[str, Any]] = {}
    for component, evs in per_component.items():
        if len(evs) <= skip_first:
            continue
        evs_sorted = sorted(evs, key=lambda e: float(e["t0"]))
        tail = evs_sorted[skip_first:]
        durs = [float(e["dur"]) for e in tail]
        out[component] = {
            "call_count_total": len(evs_sorted),
            "skipped": skip_first,
            "remaining_count": len(tail),
            "mean_duration_s": round(sum(durs) / len(durs), 3),
            "min_duration_s": round(min(durs), 3),
            "max_duration_s": round(max(durs), 3),
        }
    return out


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def process_timeline(
    timeline_dir: Path | str,
    *,
    enable_offload: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Run the four analyses and return a JSON-serialisable dict.

    Empty input → ``{}``. Missing ``enable_offload`` means outliers
    ship without a ``knob_hint``. The parser wraps this together with
    ``stall_fractions`` and ``tag_stats`` — those two are recomputable
    from the same event stream but are packed into
    :class:`TimelineSummary` separately because they feed the compact
    block of the critic prompt.
    """
    events = load_events(timeline_dir)
    if not events:
        return {}
    return {
        "critical_path": compute_critical_path(events),
        "outliers": list(compute_outliers(events, enable_offload=enable_offload)),
        "per_component_bubble": compute_per_component_bubble(events),
        "raw_excerpts": list(extract_raw_excerpts(events)),
        "component_call_averages": compute_component_call_averages(events),
    }
