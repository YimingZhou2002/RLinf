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
    "- If `actor/run_training` dominates step time: shrink `actor.micro_batch_size` "
    "or grow actor GPU count.\n"
    "- If `env/interact` or `env_interact_step` dominates: reduce `env.train.total_num_envs` "
    "or grow env GPU count.\n"
    "- If `rollout/generate_one_epoch` or `predict` dominates: grow rollout GPU count or "
    "lower `rollout.pipeline_stage_num` (note: pipeline_stage_num is pinned in this loop).\n"
    "- If memory_pressure flag is set, prefer enable_offload flips or shrink env/micro batch.\n"
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
        feedback: Optional retry feedback (e.g. "previous response was
            malformed JSON" or "placement delta missing timeline_citations").
    """
    return CriticPrompt(
        history_block=_render_history(history),
        current_knobs_block=_render_current_knobs(current_knobs, schema),
        constraints_block=_render_constraints(),
        memory_pressure_block=_render_memory_pressure(last_failure_mode),
        timeline_summary_block=_render_timeline_summary(
            last_metric_summary, last_timeline_summary
        ),
        feedback_block=f"## Feedback on previous response\n{feedback}\n" if feedback else "",
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
    if not sections:
        return ""
    return "\n\n".join(sections) + "\n"


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
            carry the required ``delta`` and ``rationale`` keys.
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
        metric_table_citations=tuple(
            str(x) for x in rationale_raw.get("metric_table_citations") or ()
        ),
        timeline_citations=tuple(
            str(x) for x in rationale_raw.get("timeline_citations") or ()
        ),
    )
    return CriticOutput(
        delta=dict(delta),
        rationale=rationale,
        stop_requested=bool(raw.get("stop_requested", False)),
    )


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
