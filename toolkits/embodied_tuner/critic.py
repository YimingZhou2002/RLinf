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

"""LLM critic for the embodied auto-tuner.

This module turns parsed trial outcomes into a structured prompt for
Codex, parses the JSON response, and enforces the dual-source rationale
rule the plan calls out: when a proposed delta touches
``cluster.component_placement``, the rationale MUST cite at least one
observation from the MetricTable AND at least one observation from the
timeline JSONL. Critic output that fails the rule is rejected and the
real critic retries up to ``max_retries`` times with the failure reason
appended as feedback.

Public surface:

- :class:`CriticPrompt` — rendered prompt sections + ``__str__``.
- :func:`build_prompt` — assemble a :class:`CriticPrompt`.
- :class:`Rationale` — structured rationale payload.
- :class:`CriticOutput` — ``{delta, rationale, stop_requested}``.
- :func:`parse_critic_output` — JSON parser, tolerates Markdown fences.
- :class:`CriticOutputValidator` — dual-source rule + schema check.
- :class:`Critic` — protocol with ``propose``.
- :class:`CodexCritic` — production implementation; shells out to
  ``ask-codex.sh`` (the binary path is injectable for testing).

The deterministic :class:`FakeCritic` lives in
:mod:`toolkits.embodied_tuner.fake_critic` so it can be imported
independently by tests and the AC-11 smoke harness.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from toolkits.embodied_tuner.lessons import BitterLesson
from toolkits.embodied_tuner.schema import (
    KNOB_PLACEMENT,
    KnobSchema,
    KnobSchemaError,
)


# --------------------------------------------------------------------------
# Wiki loader — optimization context lives in numbered ``wiki/*.md`` files
# alongside this module, NOT as inline string constants. Editing markdown
# does not require touching Python, and the same files are used as human
# documentation.
# --------------------------------------------------------------------------

_WIKI_DIR = Path(__file__).resolve().parent / "wiki"

# Files pulled into the ``wiki_block`` section of the prompt, in this
# order.
_WIKI_CONTEXT_FILES: tuple[str, ...] = (
    "00-bottleneck-rubric.md",
    "01-placement-critical-paths.md",
    "02-optimization-directions.md",
    "03-timeline-signals.md",
    "04-constraints.md",
)


def _read_wiki_file(name: str) -> str:
    """Return the contents of ``wiki/<name>``.

    A missing wiki file is a build error, not a runtime warning: the
    tuner cannot produce sensible critic prompts without them. Raising
    at import time fails fast in tests and in real trials alike.
    """
    path = _WIKI_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"critic wiki file missing: {path}. The tuner requires the "
            f"wiki/ directory to be shipped alongside critic.py."
        )
    return path.read_text(encoding="utf-8").rstrip() + "\n"


def _load_wiki_context() -> str:
    """Concatenate the wiki context files behind a common header."""
    sections = ["## Optimization context (from tuner wiki)"]
    for name in _WIKI_CONTEXT_FILES:
        sections.append(_read_wiki_file(name))
    return "\n\n".join(s.rstrip() for s in sections) + "\n"


_RATIONALE_SCHEMA_DOC = (
    "## Required output JSON shape\n"
    '{\n'
    '  "delta": {"<knob>": <value>, ...},\n'
    '  "rationale": {\n'
    '    "summary": "<one-paragraph reasoning>",\n'
    '    "metric_table_citations": ["<key=value snippet>", ...],\n'
    '    "timeline_citations": ["<component rank tag observation>", ...]\n'
    '  },\n'
    '  "stop_requested": false,\n'
    '  "bitter_lesson": {                       // required after any failed trial\n'
    '    "trigger": "<one line describing what the failed trial did and how it failed>",\n'
    '    "rule": "<durable directive for future rounds — what to avoid or require>"\n'
    '  }\n'
    '}\n'
    "Rule (dual-source): when `delta` contains `cluster.component_placement`, both\n"
    "citation arrays MUST contain at least one non-empty entry.\n"
    "\n"
    "Rule (bitter lesson): when the previous trial's failure_mode is one of\n"
    "OOM, WORKER_CRASH, TIMEOUT, CONFIG_INVALID, or DIVISIBILITY_VIOLATION, the\n"
    "response MUST include a non-empty `bitter_lesson.trigger` and\n"
    "`bitter_lesson.rule`. The scheduler persists it under\n"
    "`<ledger_dir>/bitter_lessons.jsonl` and prepends every future prompt with\n"
    "the accumulated lessons so the same failing delta is not re-proposed. Omit\n"
    "the field on a successful follow-up unless the trial revealed a durable\n"
    "constraint worth persisting.\n"
    "\n"
    "The wrapper rejects outputs that violate either rule and retries.\n"
)


class CriticError(RuntimeError):
    """Raised when the critic transport (e.g. Codex) fails entirely."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rationale:
    """Structured rationale persisted alongside each delta."""

    summary: str = ""
    metric_table_citations: tuple[str, ...] = ()
    timeline_citations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "metric_table_citations": list(self.metric_table_citations),
            "timeline_citations": list(self.timeline_citations),
        }


@dataclass(frozen=True)
class ProposedLesson:
    """The critic's raw ``bitter_lesson`` payload before scheduler enrichment.

    The critic writes ``trigger`` and ``rule`` only. The scheduler
    attaches ``trial_idx``, ``failure_mode`` and ``delta_signature``
    from the actual failed trial before handing the resulting
    :class:`~toolkits.embodied_tuner.lessons.BitterLesson` to the
    :class:`~toolkits.embodied_tuner.lessons.LessonBook`.
    """

    trigger: str
    rule: str

    def to_dict(self) -> dict[str, Any]:
        return {"trigger": self.trigger, "rule": self.rule}


@dataclass(frozen=True)
class CriticOutput:
    """The critic's structured proposal."""

    delta: Mapping[str, Any]
    rationale: Rationale
    stop_requested: bool = False
    bitter_lesson: ProposedLesson | None = None


@dataclass(frozen=True)
class TrialHistoryEntry:
    """Minimal trial summary the critic prompt repeats per past trial."""

    trial_idx: int
    delta: Mapping[str, Any]
    status: str
    failure_mode: str
    objective: float | None
    step_time: float | None
    rationale_summary: str = ""
    metric_table_excerpt: str = ""
    timeline_excerpt: str = ""
    # Short tail-of-log slice for OOM / WORKER_CRASH / METRICS_MISSING /
    # DIVISIBILITY_VIOLATION trials. Empty otherwise.
    error_excerpt: str = ""


@dataclass(frozen=True)
class CriticPrompt:
    """The fully rendered prompt sections."""

    wiki_block: str = field(default_factory=_load_wiki_context)
    schema_doc: str = _RATIONALE_SCHEMA_DOC
    bitter_lessons_block: str = ""  # permanent, cross-window failure memory
    history_block: str = "## Trial History\n(none — first round)\n"
    current_knobs_block: str = ""
    constraints_block: str = ""
    memory_pressure_block: str = ""
    metric_summary_block: str = ""  # compact: MetricTable time keys + stall fractions
    timeline_verbose_block: str = ""  # verbose: critical path / bubble / outliers / raw excerpts
    feedback_block: str = ""  # appended on retries

    def __str__(self) -> str:
        sections = [
            self.wiki_block,
            self.bitter_lessons_block,
            self.history_block,
            self.current_knobs_block,
            self.constraints_block,
            self.memory_pressure_block,
            self.metric_summary_block,
            self.timeline_verbose_block,
            self.schema_doc,
        ]
        if self.feedback_block:
            sections.append(self.feedback_block)
        return "\n".join(s.rstrip() for s in sections if s)

    def to_debug_text(self) -> str:
        """Render only the sections a human debugger needs.

        Excludes the static wiki block, the schema doc, and the
        verbose per-step timeline dump (critical path,
        per-GPU bubble, outliers, raw JSONL excerpts). Keeps the
        compact ``metric_summary_block`` because a debugger reading
        the round's decision without those keys is missing the
        primary signal the critic saw, and keeps
        ``bitter_lessons_block`` because it is the permanent memory
        the critic acts on.
        """
        sections = [
            self.bitter_lessons_block,
            self.history_block,
            self.current_knobs_block,
            self.memory_pressure_block,
            self.metric_summary_block,
        ]
        if self.feedback_block:
            sections.append(self.feedback_block)
        return "\n".join(s.rstrip() for s in sections if s)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :meth:`CriticOutputValidator.validate`."""

    ok: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    *,
    history: Sequence[TrialHistoryEntry],
    current_knobs: Mapping[str, Any],
    schema: KnobSchema,
    last_failure_mode: str | None,
    last_metric_summary: Mapping[str, float] | None,
    last_timeline_summary: Mapping[str, Any] | None,
    last_num_trajectories: int | None = None,
    bitter_lessons: Sequence[BitterLesson] = (),
    feedback: str | None = None,
    preflight_feedback: str | None = None,
) -> CriticPrompt:
    """Assemble a :class:`CriticPrompt` from current scheduler state.

    Args:
        history: Last K trials (caller decides K; AC-7 default is 8).
        current_knobs: Current value of every tunable knob.
        schema: Knob schema; used to render the legal-range block.
        last_failure_mode: ``"OOM"`` triggers the memory-pressure flag,
            and (together with WORKER_CRASH / TIMEOUT / CONFIG_INVALID)
            switches on the mandatory ``bitter_lesson`` output rule.
        last_metric_summary: ``MetricStep.time_keys`` from the last
            successful trial (``env/interact``, ``actor/run_training``, etc.).
        last_timeline_summary: ``TimelineSummary``-shaped dict (per-rank
            stats + stall fractions) from the last successful trial.
        last_num_trajectories: ``num_trajectories`` from the final
            MetricTable block of the last successful trial. Rendered as a
            sibling ``## Last trial — MetricTable Environment-section keys``
            sub-section so the critic sees the denominator of
            ``objective = avg_step_time / num_trajectories`` explicitly.
        bitter_lessons: Full campaign lessons from
            :class:`~toolkits.embodied_tuner.lessons.LessonBook`, rendered
            above the rolling history so they survive the ``history_window``.
        feedback: Optional retry feedback for invalid critic OUTPUT
            (malformed JSON or validator failure).
        preflight_feedback: Optional retry feedback for valid critic
            output that was REJECTED by preflight (e.g. divisibility
            violation, placement legality). Surfaced as its own prompt
            section so the critic can correct the proposed delta on its
            next attempt.
    """
    combined_feedback = ""
    if feedback:
        combined_feedback += f"## Feedback on previous response\n{feedback}\n"
    if preflight_feedback:
        combined_feedback += (
            f"## Preflight rejected the previous delta\n{preflight_feedback}\n"
            "Propose a different delta that does NOT violate the same constraints.\n"
        )
    return CriticPrompt(
        bitter_lessons_block=_render_bitter_lessons(bitter_lessons),
        history_block=_render_history(history),
        current_knobs_block=_render_current_knobs(current_knobs, schema),
        constraints_block=_render_constraints(),
        memory_pressure_block=_render_memory_pressure(last_failure_mode),
        metric_summary_block=_render_metric_summary_compact(
            last_metric_summary, last_timeline_summary, last_num_trajectories
        ),
        timeline_verbose_block= _render_timeline_verbose(last_timeline_summary),
        feedback_block=combined_feedback,
    )


def _render_bitter_lessons(lessons: Sequence[BitterLesson]) -> str:
    if not lessons:
        return ""
    lines = [
        "## Bitter Lessons (permanent — do NOT repeat these mistakes)",
        (
            "Each lesson was written by an earlier round of this campaign after a "
            "failed trial. They persist beyond the rolling trial history window. "
            "Treat every rule below as a hard constraint on the next delta unless "
            "you can point to concrete evidence (metric or timeline citation) that "
            "the memory / feasibility envelope has changed since the failure."
        ),
    ]
    for lesson in lessons:
        lines.append(
            f"- [trial {lesson.trial_idx}, {lesson.failure_mode}] "
            f"trigger: {lesson.trigger}"
        )
        lines.append(f"    rule: {lesson.rule}")
    return "\n".join(lines) + "\n"


def _render_history(history: Sequence[TrialHistoryEntry]) -> str:
    if not history:
        return "## Trial History\n(none — first round)\n"
    lines = ["## Trial History (most recent last)"]
    for entry in history:
        lines.append(
            f"- trial {entry.trial_idx}: ({entry.status},{entry.failure_mode}) "
            f"delta={dict(entry.delta)} objective={entry.objective} "
            f"step_time={entry.step_time}"
        )
        if entry.rationale_summary:
            lines.append(f"    rationale: {entry.rationale_summary}")
        if entry.metric_table_excerpt:
            lines.append(f"    metric_excerpt: {entry.metric_table_excerpt}")
        if entry.timeline_excerpt:
            lines.append(f"    timeline_excerpt: {entry.timeline_excerpt}")
        if entry.error_excerpt:
            lines.append("    error_log_excerpt (tail of run_embodiment.log):")
            lines.append("    ```")
            for raw_line in entry.error_excerpt.splitlines():
                lines.append(f"    {raw_line}")
            lines.append("    ```")
    return "\n".join(lines) + "\n"


def _render_current_knobs(knobs: Mapping[str, Any], schema: KnobSchema) -> str:
    lines = ["## Current knob values and legal ranges"]
    for knob in schema.list_knobs():
        domain = schema.domains[knob]
        if domain.kind == "int":
            range_str = f"int in [{domain.min_value}, {domain.max_value}]"
        elif domain.kind == "bool":
            range_str = "bool"
        else:
            range_str = "placement string or mapping"
        lines.append(f"- {knob}: current={knobs.get(knob)!r} legal={range_str}")
    lines.append("\nPinned knobs (DO NOT touch in this loop):")
    for knob in schema.list_pinned_knobs():
        lines.append(f"- {knob}")
    return "\n".join(lines) + "\n"


def _render_constraints() -> str:
    return (
        "## Hard constraints (preflight will reject violations)\n"
        "- env.train.total_num_envs % env_world_size == 0 (rlinf/config.py:962)\n"
        "- (env.train.total_num_envs / env_world_size) % rollout.pipeline_stage_num == 0 (line 965)\n"
        "- env.train.max_steps_per_rollout_epoch % actor.model.num_action_chunks == 0 (line 980)\n"
        "- actor.global_batch_size % (actor.micro_batch_size * actor_world_size) == 0 (lines 1363-1368)\n"
        "- (env.train.total_num_envs / env_world_size) % rollout_world_size == 0 AND\n"
        "  (env.train.total_num_envs / env_world_size) % actor_world_size == 0 —\n"
        "  routing-layer assertion at rlinf/scheduler/worker/routing.py:139; if\n"
        "  violated, preflight synthesises a DIVISIBILITY_VIOLATION failure. See\n"
        "  toolkits/embodied_tuner/wiki/04-constraints.md §04.2.6.\n"
        "- cluster.component_placement components use contiguous GPU ranges; env and rollout are either equal or disjoint.\n"
    )


def _render_memory_pressure(last_failure_mode: str | None) -> str:
    if last_failure_mode != "OOM":
        return ""
    return (
        "## Memory pressure (last trial failed with OOM)\n"
        "Prefer memory-reducing knobs in the next delta: enable_offload flips "
        "(env/rollout/actor), lower env.train.total_num_envs, or lower "
        "actor.micro_batch_size. Avoid placement deltas that grow the actor or "
        "rollout GPU footprint.\n"
    )


def _render_metric_summary_compact(
    metric_summary: Mapping[str, float] | None,
    timeline_summary: Mapping[str, Any] | None,
    num_trajectories: int | None = None,
) -> str:
    """Compact block: MetricTable time keys + per-component stall fractions.

    Also renders ``num_trajectories`` (from the final MetricTable's
    Environment section) as its own ``- key=value`` sibling sub-section
    so the critic sees the denominator of
    ``objective = avg_step_time / num_trajectories`` explicitly.

    This is the block that survives ``CriticPrompt.to_debug_text()`` — the
    smallest possible summary of what the critic saw for last-trial
    performance, useful for a human debugger reading the critic's decision.
    """
    sections: list[str] = []
    if metric_summary:
        lines = ["## Last trial — MetricTable Time-section keys"]
        for key in sorted(metric_summary):
            lines.append(f"- {key}={metric_summary[key]}")
        sections.append("\n".join(lines))
    if num_trajectories is not None:
        lines = [
            "## Last trial — MetricTable Environment-section keys",
            f"- num_trajectories={num_trajectories}",
        ]
        sections.append("\n".join(lines))
    if timeline_summary:
        stalls = timeline_summary.get("stall_fraction_by_component", {})
        if stalls:
            lines = ["## Last trial — per-component stall fractions (idle / total window)"]
            for component in sorted(stalls):
                lines.append(f"  - {component}: {stalls[component]:.3f}")
            sections.append("\n".join(lines))
    if not sections:
        return ""
    return "\n\n".join(sections) + "\n"


def _render_timeline_verbose(
    timeline_summary: Mapping[str, Any] | None,
) -> str:
    """Verbose block: per-tag stats, critical path, per-component bubble,
    outliers, raw excerpts. Excluded from ``to_debug_text()``.
    """
    if not timeline_summary:
        return ""
    sections: list[str] = []

    per_tag = timeline_summary.get("per_tag", ())
    if per_tag:
        lines = ["## Last trial — headline tag stats (component / rank / tag / count / median)"]
        for stat in per_tag[:24]:  # keep the prompt compact
            lines.append(
                f"  - {stat['component']} rank{stat['rank']} {stat['tag']} "
                f"count={stat['call_count']} median={stat['duration_median']:.3f}"
            )
        sections.append("\n".join(lines))

    critical_path = timeline_summary.get("critical_path") or {}
    outliers = timeline_summary.get("outliers") or ()
    per_component_bubble = timeline_summary.get("per_component_bubble") or {}
    raw_excerpts = timeline_summary.get("raw_excerpts") or ()
    raw_jsonl = timeline_summary.get("raw_jsonl") or {}
    component_call_averages = timeline_summary.get("component_call_averages") or {}

    if critical_path:
        sections.append(_render_critical_path(critical_path))
    if per_component_bubble:
        sections.append(_render_per_component_bubble(per_component_bubble))
    if component_call_averages:
        sections.append(_render_component_call_averages(component_call_averages))
    if outliers:
        sections.append(_render_outliers(outliers))
    if raw_excerpts:
        sections.append(_render_raw_excerpts(raw_excerpts))
    if raw_jsonl:
        sections.append(_render_raw_jsonl(raw_jsonl))
    if not sections:
        return ""
    return "\n\n".join(sections) + "\n"


def _render_plot_paths(plot_paths: Mapping[str, str]) -> str:
    """Point the critic at the rendered Gantt file(s) sitting on disk.

    We do not embed the image itself in the prompt (Codex's shell
    transport is text-only). The path lets a human debugger open it,
    and a future multimodal critic can lift it into an image content
    block via ``plot_paths["png"]``.
    """
    lines = ["## Last trial — timeline Gantt renders"]
    for fmt in sorted(plot_paths):
        lines.append(f"  - {fmt}: {plot_paths[fmt]}")
    return "\n".join(lines)


def _render_raw_jsonl(raw_jsonl: Mapping[str, str]) -> str:
    """Full-fidelity JSONL text block, one section per selected trace.

    The selector on the parser side keeps this bounded (default: one
    representative rank per component). Each section is fenced with the
    file stem so the critic can cite events unambiguously
    (e.g. ``env_rank3 line 47``). Files are emitted in insertion order,
    which the selector chooses deliberately (e.g. actor / rollout / env).
    """
    lines = [
        "## Last trial — raw timeline JSONL",
        (
            "Each fenced block below is the verbatim contents of one "
            "`<component>_rank<N>.jsonl` file from `<log_dir>/timeline/`. "
            "Every line is one event `{t0, t1, component, rank, tag, "
            "global_step, ...}`. Cite specific lines by their file stem "
            "and line number when the aggregated views above are not "
            "enough to justify a delta."
        ),
    ]
    for name in raw_jsonl:  # insertion-ordered
        body = raw_jsonl[name].rstrip()
        lines.append(f"\n### {name}")
        lines.append("```jsonl")
        lines.append(body)
        lines.append("```")
    return "\n".join(lines)


# Concept-explanation prefix the critic must see exactly once per prompt so
# it interprets the A'/D' tables correctly. Without it the LLM will read
# `actor/recv_traj=259s` as "actor is the bottleneck" when in fact actor
# is idle waiting for rollout to finish producing trajectories.
_BLOCKING_TAGS_EXPLAINER = (
    "Note: in the hybrid placement the actor lives on every GPU but only "
    "does real GPU work during sync_model_to_rollout / compute_adv / "
    "forward / backward / optimizer_step. `actor/recv_traj` is a "
    "blocking-wait on rollout+env producing trajectories — its duration "
    "is the rollout/env cost, not actor work. The tables below split "
    "'real' (actual GPU work) from 'blocked' (waiting on another "
    "component)."
)


def _render_critical_path(critical_path: Mapping[Any, Mapping[str, Any]]) -> str:
    """A' — per-step real-busy lane ranking with blocked context."""
    lines = ["## Last trial — critical path per global_step", _BLOCKING_TAGS_EXPLAINER]
    for raw_step in sorted(critical_path, key=lambda k: int(k)):
        step = critical_path[raw_step]
        lines.append(f"- step={raw_step}  step_span_s={step.get('step_span_s')}")
        for lane in step.get("real_busy_top", ()):
            lines.append(
                f"    {lane['component']}/r{lane['rank']}: "
                f"real={lane['real_s']}s  blocked={lane['blocked_s']}s  "
                f"real_frac={lane['real_frac']}"
            )
    return "\n".join(lines)


def _render_per_component_bubble(per_component_bubble: Mapping[str, Any]) -> str:
    """D' — per-component (env/rollout/actor) bubble under this trial."""
    wall = per_component_bubble.get("wall_s")
    lines = [
        "## Last trial — per-component bubble",
        f"wall_s={wall}",
        "Bubble = wall - union(real-busy intervals across all ranks of this "
        "component); blocking-wait tags (e.g. `actor/recv_traj`) are excluded "
        "so busy_s reflects real GPU work only. The component with the "
        "largest bubble_frac is the one whose ranks were idle the most "
        "wall-clock — usually the side whose GPU budget can be reduced.",
    ]
    per_component = per_component_bubble.get("per_component") or {}
    for component in sorted(per_component):
        info = per_component[component]
        lines.append(
            f"  - {component}: busy_s={info.get('busy_s')} "
            f"bubble_s={info.get('bubble_s')} "
            f"bubble_frac={info.get('bubble_frac')} "
            f"ranks={info.get('num_ranks')}"
        )
        per_rank = info.get("per_rank") or {}
        for raw_rank in sorted(per_rank, key=lambda k: int(k)):
            rank_info = per_rank[raw_rank]
            lines.append(
                f"      r{raw_rank}: busy_s={rank_info.get('busy_s')} "
                f"bubble_s={rank_info.get('bubble_s')} "
                f"bubble_frac={rank_info.get('bubble_frac')}"
            )
    return "\n".join(lines)


def _render_component_call_averages(
    component_call_averages: Mapping[str, Mapping[str, Any]],
) -> str:
    """Per-component steady-state mean per-call duration (first 2 events skipped).

    Renders the ``compute_component_call_averages`` view — the typical
    per-call cost for env / rollout after the bootstrap warmup calls
    are dropped. Complements per_component_bubble (which sums busy time)
    by exposing the *typical* per-call number the critic can compare
    against outliers.
    """
    lines = [
        "## Last trial — env/rollout steady-state per-call duration "
        "(first 2 calls dropped as warmup)",
        "For each component: pool every non-blocking event across all "
        "ranks/tags, sort by t0, drop the first 2 (bootstrap warmup — "
        "offload page-in, JIT compile, first-CUDA-kernel init), then "
        "average dur over the remainder. Wrapper tags "
        "(`interact`, `run_interact_once`, `rollout/generate`, "
        "`rollout/generate_one_epoch`) are excluded to avoid double-"
        "counting per-step children.",
    ]
    for component in sorted(component_call_averages):
        info = component_call_averages[component]
        lines.append(
            f"  - {component}: mean={info.get('mean_duration_s')}s "
            f"min={info.get('min_duration_s')}s "
            f"max={info.get('max_duration_s')}s "
            f"n={info.get('remaining_count')} "
            f"(skipped={info.get('skipped')} of "
            f"total={info.get('call_count_total')})"
        )
    return "\n".join(lines)


def _render_outliers(outliers: Sequence[Mapping[str, Any]]) -> str:
    """C' — longest events above per-tag P95 with knob hint."""
    lines = [
        "## Last trial — outlier events (per-tag P95, >1s)",
        "Each row: component / rank / tag / step / duration / knob hint (if any). "
        "knob_hint links the stall back to a tunable knob the critic can flip.",
    ]
    for outlier in outliers:
        hint = outlier.get("knob_hint")
        lines.append(
            f"  - {outlier.get('component')}/r{outlier.get('rank')} "
            f"{outlier.get('tag')} step={outlier.get('global_step')} "
            f"dur_s={outlier.get('dur_s')}"
            + (f"  hint={hint}" if hint else "")
        )
    return "\n".join(lines)


def _render_raw_excerpts(excerpts: Sequence[Mapping[str, Any]]) -> str:
    """Top-K longest raw events copied as-is so the critic sees full context."""
    lines = [
        "## Last trial — raw timeline excerpts (top-K longest events, runner wrapper excluded)",
        "These are verbatim JSONL events. Use the qualname / call_index / "
        "configured_* fields to reason about cases the aggregated tables miss.",
    ]
    for excerpt in excerpts:
        # One compact JSON line per event so the critic can cite it directly.
        lines.append("  - " + json.dumps(excerpt, sort_keys=True))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_critic_output(text: str) -> CriticOutput:
    """Parse a critic response into a :class:`CriticOutput`.

    Accepts:
    - Raw JSON (whole response).
    - JSON inside a Markdown ```json ... ``` fence.
    - JSON inside any ``` ... ``` fence (first one wins).

    Raises:
        CriticError: when no JSON can be located OR the JSON does not
            carry the required ``delta`` and ``rationale`` keys, OR when
            the rationale's citation arrays are not lists of strings
            (a single string would otherwise be iterated character-by-
            character by the dual-source validator — exactly the escape
            hatch the AC-7 spec forbids).
    """
    candidate = _extract_json_candidate(text)
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise CriticError(f"critic output is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise CriticError(f"critic output must be a JSON object, got {type(raw).__name__}")
    delta = raw.get("delta")
    rationale_raw = raw.get("rationale")
    if not isinstance(delta, Mapping):
        raise CriticError("critic output missing required 'delta' object")
    if not isinstance(rationale_raw, Mapping):
        raise CriticError("critic output missing required 'rationale' object")

    rationale = Rationale(
        summary=str(rationale_raw.get("summary", "")),
        metric_table_citations=_coerce_citation_list(
            rationale_raw.get("metric_table_citations"),
            field_name="metric_table_citations",
        ),
        timeline_citations=_coerce_citation_list(
            rationale_raw.get("timeline_citations"),
            field_name="timeline_citations",
        ),
    )
    bitter_lesson = _coerce_bitter_lesson(raw.get("bitter_lesson"))
    return CriticOutput(
        delta=dict(delta),
        rationale=rationale,
        stop_requested=bool(raw.get("stop_requested", False)),
        bitter_lesson=bitter_lesson,
    )


def _coerce_bitter_lesson(value: object) -> ProposedLesson | None:
    """Validate an optional ``bitter_lesson`` object on a critic response.

    Returns ``None`` when the field is absent, ``null``, or an empty
    object. Raises :class:`CriticError` when the payload is malformed
    (not a JSON object, or ``trigger`` / ``rule`` present but not a
    string). Whether the field is REQUIRED for the current round is a
    scheduler-level decision (see
    :meth:`CriticOutputValidator.validate`); this helper only checks
    shape.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CriticError(
            f"bitter_lesson must be a JSON object, got {type(value).__name__}"
        )
    if not value:
        return None
    trigger = value.get("trigger", "")
    rule = value.get("rule", "")
    if not isinstance(trigger, str):
        raise CriticError(
            f"bitter_lesson.trigger must be a string, got {type(trigger).__name__}"
        )
    if not isinstance(rule, str):
        raise CriticError(
            f"bitter_lesson.rule must be a string, got {type(rule).__name__}"
        )
    if not trigger.strip() and not rule.strip():
        return None
    return ProposedLesson(trigger=trigger, rule=rule)


def _coerce_citation_list(value: object, *, field_name: str) -> tuple[str, ...]:
    """Validate that ``value`` is a (possibly missing) list of strings.

    Returns an empty tuple when ``value`` is ``None`` (the dual-source
    validator catches that for placement deltas). Raises :class:`CriticError`
    when ``value`` is a bare string, a number, or a list containing
    non-string elements — these are the silent escape hatches the
    validator would otherwise miss.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        raise CriticError(
            f"{field_name} must be a JSON array of strings, got a bare string "
            f"(this would be silently iterated char-by-char by the validator)"
        )
    if not isinstance(value, list):
        raise CriticError(
            f"{field_name} must be a JSON array of strings, got {type(value).__name__}"
        )
    for entry in value:
        if not isinstance(entry, str):
            raise CriticError(
                f"{field_name} entries must be strings, got {type(entry).__name__}"
            )
    return tuple(value)


def _extract_json_candidate(text: str) -> str:
    """Return the substring of ``text`` most likely to be JSON."""
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    # Bare JSON: trust the input. Otherwise, hunt for the first { and last }.
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    open_idx = text.find("{")
    close_idx = text.rfind("}")
    if open_idx != -1 and close_idx != -1 and close_idx > open_idx:
        return text[open_idx : close_idx + 1]
    return stripped


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


_LESSON_REQUIRED_FAILURE_MODES: frozenset[str] = frozenset(
    {"OOM", "WORKER_CRASH", "TIMEOUT", "CONFIG_INVALID", "DIVISIBILITY_VIOLATION"}
)


@dataclass(frozen=True)
class CriticOutputValidator:
    """Enforces the dual-source rationale rule and re-runs the knob schema."""

    schema: KnobSchema
    last_failure_mode: str | None = None

    def validate(self, output: CriticOutput) -> ValidationResult:
        try:
            self.schema.validate(output.delta)
        except KnobSchemaError as exc:
            return ValidationResult(ok=False, reason=f"schema: {exc}")
        if KNOB_PLACEMENT in output.delta:
            if not _has_non_empty(output.rationale.metric_table_citations):
                return ValidationResult(
                    ok=False,
                    reason=(
                        "dual-source rule: placement-touching delta must cite at least "
                        "one non-empty metric_table_citations entry"
                    ),
                )
            if not _has_non_empty(output.rationale.timeline_citations):
                return ValidationResult(
                    ok=False,
                    reason=(
                        "dual-source rule: placement-touching delta must cite at least "
                        "one non-empty timeline_citations entry"
                    ),
                )
        else:
            if not output.rationale.summary.strip():
                return ValidationResult(
                    ok=False,
                    reason="non-placement deltas require a non-empty rationale.summary",
                )
        if (
            self.last_failure_mode in _LESSON_REQUIRED_FAILURE_MODES
            and not _lesson_is_populated(output.bitter_lesson)
            and not output.stop_requested
        ):
            return ValidationResult(
                ok=False,
                reason=(
                    f"previous trial failed with failure_mode={self.last_failure_mode}; "
                    "response must include a non-empty bitter_lesson.trigger AND "
                    "bitter_lesson.rule so the failure is not forgotten after the "
                    "history window rolls over"
                ),
            )
        return ValidationResult(ok=True)


def _lesson_is_populated(lesson: ProposedLesson | None) -> bool:
    return (
        lesson is not None
        and bool(lesson.trigger.strip())
        and bool(lesson.rule.strip())
    )


def _has_non_empty(items: Sequence[str]) -> bool:
    return any(isinstance(s, str) and s.strip() for s in items)


# ---------------------------------------------------------------------------
# Critic protocol + Codex implementation
# ---------------------------------------------------------------------------


class Critic(Protocol):
    """Implementations propose a delta given the current loop state."""

    def propose(
        self,
        *,
        history: Sequence[TrialHistoryEntry],
        current_knobs: Mapping[str, Any],
        schema: KnobSchema,
        last_failure_mode: str | None,
        last_metric_summary: Mapping[str, float] | None,
        last_timeline_summary: Mapping[str, Any] | None,
        last_num_trajectories: int | None = None,
        bitter_lessons: Sequence[BitterLesson] = (),
        preflight_feedback: str | None = None,
    ) -> CriticOutput:
        ...


@dataclass(frozen=True)
class CodexCritic:
    """Critic backed by the humanize ``ask-codex.sh`` script.

    Attributes:
        ask_codex_path: Path to ``ask-codex.sh``. Tests inject a fake
            shell-callable via ``transport``.
        max_retries: Maximum dual-source / schema retries before raising
            :class:`CriticError`. Mirrors the AC-7 default of 3.
        transport: Optional injection point. When set, called as
            ``transport(prompt) -> str``; otherwise the script is invoked
            via :func:`subprocess.run`.
        transaction_log: Per-``propose`` capture of each transport
            exchange, for the scheduler to persist under
            ``<trial_log_dir>/critic/``. Reset at the start of each
            ``propose`` call. Records carry the DEBUG view of the
            prompt (see :meth:`CriticPrompt.to_debug_text`) so the
            saved file is inspection-friendly rather than a 36 KB dump
            of the wiki + verbose timeline. The field is a
            mutable list; ``frozen=True`` still holds because dataclass
            freezing only blocks ``__setattr__`` on the container, not
            mutation of a list's contents.
    """

    schema: KnobSchema
    ask_codex_path: str = str(
        Path(__file__).resolve().parent / "scripts" / "ask-codex.sh"
    )
    max_retries: int = 3
    transport: Callable[[str], str] | None = None
    transaction_log: list[dict[str, Any]] = field(default_factory=list)

    def propose(
        self,
        *,
        history: Sequence[TrialHistoryEntry],
        current_knobs: Mapping[str, Any],
        schema: KnobSchema | None = None,
        last_failure_mode: str | None,
        last_metric_summary: Mapping[str, float] | None,
        last_timeline_summary: Mapping[str, Any] | None,
        last_num_trajectories: int | None = None,
        bitter_lessons: Sequence[BitterLesson] = (),
        preflight_feedback: str | None = None,
    ) -> CriticOutput:
        active_schema = schema or self.schema
        validator = CriticOutputValidator(
            schema=active_schema, last_failure_mode=last_failure_mode
        )
        feedback: str | None = None
        last_error: str = ""

        # Reset per-propose so each trial round's log stands alone.
        self.transaction_log.clear()

        for attempt in range(self.max_retries + 1):
            prompt = build_prompt(
                history=history,
                current_knobs=current_knobs,
                schema=active_schema,
                last_failure_mode=last_failure_mode,
                last_metric_summary=last_metric_summary,
                last_timeline_summary=last_timeline_summary,
                last_num_trajectories=last_num_trajectories,
                bitter_lessons=bitter_lessons,
                feedback=feedback,
                preflight_feedback=preflight_feedback,
            )
            debug_prompt = prompt.to_debug_text()
            response = self._invoke_transport(str(prompt))
            record: dict[str, Any] = {
                "attempt": attempt,
                "prompt_debug": debug_prompt,
                "response": response,
                "parse_error": None,
                "validation_ok": False,
                "validation_reason": "",
            }
            try:
                output = parse_critic_output(response)
            except CriticError as exc:
                last_error = str(exc)
                record["parse_error"] = last_error
                self.transaction_log.append(record)
                feedback = (
                    f"Your previous response could not be parsed as the required JSON "
                    f"object: {exc}. Re-emit the JSON exactly per the schema above."
                )
                continue
            verdict = validator.validate(output)
            record["validation_ok"] = verdict.ok
            record["validation_reason"] = verdict.reason
            self.transaction_log.append(record)
            if verdict.ok:
                return output
            last_error = verdict.reason
            feedback = (
                f"Your previous response failed validation: {verdict.reason}. "
                "Re-emit a corrected JSON object that satisfies the rule."
            )
        raise CriticError(
            f"critic could not produce a valid output after {self.max_retries + 1} attempts; "
            f"last error: {last_error}"
        )

    # ----- Transport -------------------------------------------------------

    def _invoke_transport(self, prompt: str) -> str:
        if self.transport is not None:
            return self.transport(prompt)
        try:
            result = subprocess.run(  # noqa: S603 — known argv
                [self.ask_codex_path, "--stdin"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            raise CriticError(f"ask-codex.sh transport failed: {exc}") from exc
        if result.returncode != 0:
            raise CriticError(
                f"ask-codex.sh exited with code {result.returncode}: "
                f"{result.stderr[:500]}"
            )
        return result.stdout
