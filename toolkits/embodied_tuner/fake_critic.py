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

"""Deterministic critic stand-in used by tests and the AC-11 smoke test.

The fake critic returns a pre-programmed sequence of
:class:`CriticOutput` values. It exists so the scheduler can be tested
without depending on Codex network / process state, and so the smoke
test can drive the orchestrator end-to-end without GPUs or LLM calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from toolkits.embodied_tuner.critic import (
    CriticError,
    CriticOutput,
    ProposedLesson,
    Rationale,
    TrialHistoryEntry,
)
from toolkits.embodied_tuner.lessons import BitterLesson
from toolkits.embodied_tuner.schema import KnobSchema


@dataclass
class FakeCritic:
    """A deterministic :class:`~toolkits.embodied_tuner.critic.Critic`.

    Attributes:
        outputs: Pre-programmed responses. ``propose`` returns the next
            one each time. Raise :class:`CriticError` when the list is
            exhausted (helps tests catch off-by-one scheduling bugs).
        calls: Captured ``(history_length, current_knobs,
            preflight_feedback, bitter_lesson_count)`` tuples for
            assertion in tests.
    """

    outputs: list[CriticOutput]
    calls: list[tuple[int, dict[str, Any], str | None, int]] = field(
        default_factory=list
    )

    def propose(
        self,
        *,
        history: Sequence[TrialHistoryEntry],
        current_knobs: Mapping[str, Any],
        schema: KnobSchema | None = None,
        last_failure_mode: str | None = None,
        last_metric_summary: Mapping[str, float] | None = None,
        last_timeline_summary: Mapping[str, Any] | None = None,
        last_memory_summary: Mapping[str, Any] | None = None,
        last_num_trajectories: int | None = None,
        bitter_lessons: Sequence[BitterLesson] = (),
        preflight_feedback: str | None = None,
        dag_block: str = "",
        last_sibling_failure_mode: str | None = None,
    ) -> CriticOutput:
        del (
            schema,
            last_failure_mode,
            last_metric_summary,
            last_timeline_summary,
            last_memory_summary,
            last_num_trajectories,
            dag_block,
            last_sibling_failure_mode,
        )
        self.calls.append(
            (len(history), dict(current_knobs), preflight_feedback, len(bitter_lessons))
        )
        if not self.outputs:
            raise CriticError("FakeCritic.outputs exhausted")
        return self.outputs.pop(0)

    @classmethod
    def from_deltas(cls, *deltas: Mapping[str, Any]) -> FakeCritic:
        """Build a :class:`FakeCritic` whose responses are placement-free deltas.

        Each delta gets a minimal valid rationale (non-empty ``summary``,
        no citations) so the :class:`CriticOutputValidator` accepts it.
        """
        outputs = [
            CriticOutput(
                delta=dict(d),
                rationale=Rationale(summary="fake critic deterministic delta"),
            )
            for d in deltas
        ]
        return cls(outputs=outputs)

    @classmethod
    def from_outputs(cls, *outputs: CriticOutput) -> FakeCritic:
        """Build a :class:`FakeCritic` from fully-formed outputs.

        Prefer this when tests need to attach ``bitter_lesson`` payloads
        or bespoke ``rationale`` shapes; :meth:`from_deltas` covers the
        common placement-free / no-lesson case.
        """
        return cls(outputs=list(outputs))

    @classmethod
    def stop_after(cls, *deltas: Mapping[str, Any]) -> FakeCritic:
        """Like :meth:`from_deltas`, but the final response sets
        ``stop_requested=True`` so the scheduler terminates after applying it.
        """
        outputs = [
            CriticOutput(
                delta=dict(d),
                rationale=Rationale(summary="fake critic deterministic delta"),
            )
            for d in deltas
        ]
        if outputs:
            last = outputs[-1]
            outputs[-1] = CriticOutput(
                delta=last.delta,
                rationale=last.rationale,
                stop_requested=True,
            )
        return cls(outputs=outputs)


def make_output_with_lesson(
    delta: Mapping[str, Any],
    *,
    trigger: str,
    rule: str,
    summary: str = "fake critic deterministic delta",
) -> CriticOutput:
    """Convenience for tests that need an output carrying a ``bitter_lesson``."""
    return CriticOutput(
        delta=dict(delta),
        rationale=Rationale(summary=summary),
        bitter_lesson=ProposedLesson(trigger=trigger, rule=rule),
    )
