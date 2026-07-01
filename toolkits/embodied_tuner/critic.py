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
from typing import Any, Callable, Protocol

from toolkits.embodied_tuner.schema import (
    KNOB_PLACEMENT,
    KnobSchema,
    KnobSchemaError,
)


_BOTTLENECK_RUBRIC = (
    "## Bottleneck rubric (Alt-2 in the design draft)\n"
    "Read the timeline sections below in this order:\n"
    "1. `per-GPU bubble`: side with higher bubble has slack; consider "
    "moving GPUs away from it.\n"
    "2. `critical path per global_step`: the lane with the largest `real_s` "
    "is the one whose work limits step time. Lanes with large `blocked_s` "
    "and small `real_s` are NOT bottlenecks — they are downstream consumers.\n"
    "3. `outlier events`: any row with a `knob_hint` is a stall the critic "
    "can directly fix by flipping the named knob.\n"
    "4. `raw timeline excerpts`: cite specific events to ground a placement "
    "delta (mandatory dual-source rule).\n"
    "Knob-side heuristics:\n"
    "- If `actor_forward` / `actor_backward` dominate REAL busy: shrink "
    "`actor.micro_batch_size` or grow actor GPU count.\n"
    "- If `env_interact_step` is the largest real lane: reduce "
    "`env.train.total_num_envs` or grow env GPU range.\n"
    "- If `predict` is the largest real lane: grow rollout GPU range.\n"
    "- If memory_pressure flag is set, prefer enable_offload flips or "
    "shrink env/micro batch.\n"
)


_RATIONALE_SCHEMA_DOC = (
    "## Required output JSON shape\n"
    '{\n'
    '  "delta": {"<knob>": <value>, ...},\n'
    '  "rationale": {\n'
    '    "summary": "<one-paragraph reasoning>",\n'
    '    "metric_table_citations": ["<key=value snippet>", ...],\n'
    '    "timeline_citations": ["<component rank tag observation>", ...]\n'
    '  },\n'
    '  "stop_requested": false\n'
    '}\n'
    "Rule: when `delta` contains `cluster.component_placement`, both citation\n"
    "arrays MUST contain at least one non-empty entry. The wrapper rejects\n"
    "outputs that violate this rule and retries.\n"
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
class CriticOutput:
    """The critic's structured proposal."""

    delta: Mapping[str, Any]
    rationale: Rationale
    stop_requested: bool = False


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


@dataclass(frozen=True)
class CriticPrompt:
    """The fully rendered prompt sections."""

    rubric: str = _BOTTLENECK_RUBRIC
    schema_doc: str = _RATIONALE_SCHEMA_DOC
    history_block: str = "## Trial History\n(none — first round)\n"
    current_knobs_block: str = ""
    constraints_block: str = ""
    memory_pressure_block: str = ""
    timeline_summary_block: str = ""
    feedback_block: str = ""  # appended on retries

    def __str__(self) -> str:
        sections = [
            self.rubric,
            self.history_block,
            self.current_knobs_block,
            self.constraints_block,
            self.memory_pressure_block,
            self.timeline_summary_block,
            self.schema_doc,
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
    feedback: str | None = None,
    preflight_feedback: str | None = None,
) -> CriticPrompt:
    """Assemble a :class:`CriticPrompt` from current scheduler state.

    Args:
        history: Last K trials (caller decides K; AC-7 default is 8).
        current_knobs: Current value of every tunable knob.
        schema: Knob schema; used to render the legal-range block.
        last_failure_mode: ``"OOM"`` triggers the memory-pressure flag.
        last_metric_summary: ``MetricStep.time_keys`` from the last
            successful trial (``env/interact``, ``actor/run_training``, etc.).
        last_timeline_summary: ``TimelineSummary``-shaped dict (per-rank
            stats + stall fractions) from the last successful trial.
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
        history_block=_render_history(history),
        current_knobs_block=_render_current_knobs(current_knobs, schema),
        constraints_block=_render_constraints(),
        memory_pressure_block=_render_memory_pressure(last_failure_mode),
        timeline_summary_block=_render_timeline_summary(
            last_metric_summary, last_timeline_summary
        ),
        feedback_block=combined_feedback,
    )


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


def _render_timeline_summary(
    metric_summary: Mapping[str, float] | None,
    timeline_summary: Mapping[str, Any] | None,
) -> str:
    sections: list[str] = []
    if metric_summary:
        lines = ["## Last trial — MetricTable Time-section keys"]
        for key in sorted(metric_summary):
            lines.append(f"- {key}={metric_summary[key]}")
        sections.append("\n".join(lines))
    if timeline_summary:
        per_tag = timeline_summary.get("per_tag", ())
        stalls = timeline_summary.get("stall_fraction_by_component", {})
        lines = ["## Last trial — per-component timeline summary"]
        if stalls:
            lines.append("Stall fractions (idle / total window):")
            for component in sorted(stalls):
                lines.append(f"  - {component}: {stalls[component]:.3f}")
        if per_tag:
            lines.append("Headline tag stats (component / rank / tag / count / median):")
            for stat in per_tag[:24]:  # keep the prompt compact
                lines.append(
                    f"  - {stat['component']} rank{stat['rank']} {stat['tag']} "
                    f"count={stat['call_count']} median={stat['duration_median']:.3f}"
                )
        sections.append("\n".join(lines))

        critical_path = timeline_summary.get("critical_path") or {}
        outliers = timeline_summary.get("outliers") or ()
        per_gpu_bubble = timeline_summary.get("per_gpu_bubble") or {}
        raw_excerpts = timeline_summary.get("raw_excerpts") or ()

        if critical_path:
            sections.append(_render_critical_path(critical_path))
        if per_gpu_bubble:
            sections.append(_render_per_gpu_bubble(per_gpu_bubble))
        if outliers:
            sections.append(_render_outliers(outliers))
        if raw_excerpts:
            sections.append(_render_raw_excerpts(raw_excerpts))
    if not sections:
        return ""
    return "\n\n".join(sections) + "\n"


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


def _render_per_gpu_bubble(per_gpu_bubble: Mapping[str, Any]) -> str:
    """D' — GPU-by-GPU bubble under the trial's placement."""
    wall = per_gpu_bubble.get("wall_s")
    env_avg = per_gpu_bubble.get("env_side_avg_bubble_s")
    rollout_avg = per_gpu_bubble.get("rollout_side_avg_bubble_s")
    lines = [
        "## Last trial — per-GPU bubble under this trial's placement",
        f"wall_s={wall}  env_side_avg_bubble_s={env_avg}  "
        f"rollout_side_avg_bubble_s={rollout_avg}",
        "Bubble = wall - union(real-busy intervals from components on this GPU). "
        "Lower bubble = more useful work. The side with the larger bubble is "
        "the one whose GPU budget can be reduced without hurting throughput.",
    ]
    per_gpu = per_gpu_bubble.get("per_gpu") or {}
    for raw_gpu in sorted(per_gpu, key=lambda k: int(k)):
        info = per_gpu[raw_gpu]
        residents = "+".join(info.get("residents", []))
        lines.append(
            f"  - GPU{raw_gpu} ({residents}): busy_s={info.get('busy_s')} "
            f"bubble_s={info.get('bubble_s')} bubble_frac={info.get('bubble_frac')}"
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
    return CriticOutput(
        delta=dict(delta),
        rationale=rationale,
        stop_requested=bool(raw.get("stop_requested", False)),
    )


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


@dataclass(frozen=True)
class CriticOutputValidator:
    """Enforces the dual-source rationale rule and re-runs the knob schema."""

    schema: KnobSchema

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
        return ValidationResult(ok=True)


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
    """

    schema: KnobSchema
    ask_codex_path: str = "/root/.claude/plugins/cache/PolyArch/humanize/1.17.0/scripts/ask-codex.sh"
    max_retries: int = 3
    transport: Callable[[str], str] | None = None

    def propose(
        self,
        *,
        history: Sequence[TrialHistoryEntry],
        current_knobs: Mapping[str, Any],
        schema: KnobSchema | None = None,
        last_failure_mode: str | None,
        last_metric_summary: Mapping[str, float] | None,
        last_timeline_summary: Mapping[str, Any] | None,
        preflight_feedback: str | None = None,
    ) -> CriticOutput:
        active_schema = schema or self.schema
        validator = CriticOutputValidator(active_schema)
        feedback: str | None = None
        last_error: str = ""

        for attempt in range(self.max_retries + 1):
            prompt = build_prompt(
                history=history,
                current_knobs=current_knobs,
                schema=active_schema,
                last_failure_mode=last_failure_mode,
                last_metric_summary=last_metric_summary,
                last_timeline_summary=last_timeline_summary,
                feedback=feedback,
                preflight_feedback=preflight_feedback,
            )
            response = self._invoke_transport(str(prompt))
            try:
                output = parse_critic_output(response)
            except CriticError as exc:
                last_error = str(exc)
                feedback = (
                    f"Your previous response could not be parsed as the required JSON "
                    f"object: {exc}. Re-emit the JSON exactly per the schema above."
                )
                continue
            verdict = validator.validate(output)
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
                [self.ask_codex_path, prompt],
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
