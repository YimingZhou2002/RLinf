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

"""Unit tests for :mod:`toolkits.embodied_tuner.critic` and ``fake_critic``."""

from __future__ import annotations

import json

import pytest

from toolkits.embodied_tuner.critic import (
    CodexCritic,
    CriticError,
    CriticOutput,
    CriticOutputValidator,
    ProposedLesson,
    Rationale,
    TrialHistoryEntry,
    build_prompt,
    parse_critic_output,
)
from toolkits.embodied_tuner.fake_critic import FakeCritic
from toolkits.embodied_tuner.lessons import BitterLesson, canonical_delta_signature
from toolkits.embodied_tuner.schema import (
    KNOB_ACTOR_OFFLOAD,
    KNOB_GLOBAL_BATCH_SIZE,
    KNOB_MICRO_BATCH_SIZE,
    KNOB_PLACEMENT,
    KnobSchema,
)


# ---------------------------------------------------------------------------
# parse_critic_output
# ---------------------------------------------------------------------------


def test_parse_critic_output_raw_json() -> None:
    text = json.dumps(
        {
            "delta": {"actor.micro_batch_size": 64},
            "rationale": {"summary": "shrink mbs", "metric_table_citations": [], "timeline_citations": []},
        }
    )
    out = parse_critic_output(text)
    assert out.delta == {"actor.micro_batch_size": 64}
    assert out.rationale.summary == "shrink mbs"
    assert out.stop_requested is False


def test_parse_critic_output_inside_markdown_fence() -> None:
    text = (
        "Here is the JSON:\n"
        "```json\n"
        '{"delta": {"actor.enable_offload": true}, '
        '"rationale": {"summary": "offload actor", '
        '"metric_table_citations": [], "timeline_citations": []}}\n'
        "```\n"
        "End of message.\n"
    )
    out = parse_critic_output(text)
    assert out.delta == {"actor.enable_offload": True}
    assert out.rationale.summary == "offload actor"


def test_parse_critic_output_inside_unlabeled_fence() -> None:
    text = '```\n{"delta": {"x": 1}, "rationale": {"summary": "s"}}\n```'
    out = parse_critic_output(text)
    assert out.delta == {"x": 1}


def test_parse_critic_output_braces_only() -> None:
    text = (
        "Some preamble.\n"
        '{"delta": {"x": 2}, "rationale": {"summary": "ok"}}\n'
        "Some postamble."
    )
    out = parse_critic_output(text)
    assert out.delta == {"x": 2}


def test_parse_critic_output_propagates_stop_requested() -> None:
    text = json.dumps(
        {
            "delta": {},
            "rationale": {"summary": "no further improvement"},
            "stop_requested": True,
        }
    )
    out = parse_critic_output(text)
    assert out.stop_requested is True


def test_parse_critic_output_raises_on_malformed_json() -> None:
    with pytest.raises(CriticError):
        parse_critic_output("```json\n{not valid json\n```")


def test_parse_critic_output_requires_delta_and_rationale() -> None:
    with pytest.raises(CriticError):
        parse_critic_output(json.dumps({"delta": {}}))
    with pytest.raises(CriticError):
        parse_critic_output(json.dumps({"rationale": {"summary": ""}}))


def test_parse_critic_output_rejects_string_citations() -> None:
    """Codex Round-2 review: a bare string in a citation field would be
    iterated char-by-char by the validator (escape hatch). Reject it.
    """
    text = json.dumps(
        {
            "delta": {"actor.micro_batch_size": 64},
            "rationale": {
                "summary": "ok",
                "metric_table_citations": "single string instead of list",
                "timeline_citations": [],
            },
        }
    )
    with pytest.raises(CriticError) as exc:
        parse_critic_output(text)
    assert "metric_table_citations" in str(exc.value)
    assert "bare string" in str(exc.value).lower()


def test_parse_critic_output_rejects_non_list_citations() -> None:
    text = json.dumps(
        {
            "delta": {"actor.micro_batch_size": 64},
            "rationale": {
                "summary": "ok",
                "metric_table_citations": 7,  # not a list at all
                "timeline_citations": [],
            },
        }
    )
    with pytest.raises(CriticError) as exc:
        parse_critic_output(text)
    assert "metric_table_citations" in str(exc.value)


def test_parse_critic_output_rejects_non_string_citation_elements() -> None:
    text = json.dumps(
        {
            "delta": {"actor.micro_batch_size": 64},
            "rationale": {
                "summary": "ok",
                "metric_table_citations": ["env/interact=275", 12345],
                "timeline_citations": [],
            },
        }
    )
    with pytest.raises(CriticError) as exc:
        parse_critic_output(text)
    assert "metric_table_citations" in str(exc.value)


def test_parse_critic_output_accepts_missing_citation_fields() -> None:
    """Missing citation fields are fine — the validator catches placement-delta
    cases that require them; non-placement deltas only need a summary.
    """
    text = json.dumps(
        {
            "delta": {"actor.micro_batch_size": 32},
            "rationale": {"summary": "shrink mbs"},
        }
    )
    out = parse_critic_output(text)
    assert out.rationale.metric_table_citations == ()
    assert out.rationale.timeline_citations == ()


# ---------------------------------------------------------------------------
# CriticOutputValidator
# ---------------------------------------------------------------------------


def _placement_delta() -> dict[str, object]:
    return {KNOB_PLACEMENT: {"actor": "0-7", "env": "0-3", "rollout": "4-7"}}


def test_validator_accepts_placement_delta_with_dual_source() -> None:
    schema = KnobSchema()
    validator = CriticOutputValidator(schema)
    output = CriticOutput(
        delta=_placement_delta(),
        rationale=Rationale(
            summary="rebalance",
            metric_table_citations=("env/interact=275.4",),
            timeline_citations=("env rank0 env_interact_step median=15s, stall=0.4",),
        ),
    )
    result = validator.validate(output)
    assert result.ok, result.reason


def test_validator_rejects_placement_delta_missing_metric_citation() -> None:
    validator = CriticOutputValidator(KnobSchema())
    output = CriticOutput(
        delta=_placement_delta(),
        rationale=Rationale(
            summary="rebalance",
            metric_table_citations=(),
            timeline_citations=("env rank0 env_interact_step median=15s",),
        ),
    )
    result = validator.validate(output)
    assert not result.ok
    assert "metric_table_citations" in result.reason


def test_validator_rejects_placement_delta_missing_timeline_citation() -> None:
    validator = CriticOutputValidator(KnobSchema())
    output = CriticOutput(
        delta=_placement_delta(),
        rationale=Rationale(
            summary="rebalance",
            metric_table_citations=("env/interact=275.4",),
            timeline_citations=(),
        ),
    )
    result = validator.validate(output)
    assert not result.ok
    assert "timeline_citations" in result.reason


def test_validator_rejects_empty_citations_for_placement_delta() -> None:
    validator = CriticOutputValidator(KnobSchema())
    output = CriticOutput(
        delta=_placement_delta(),
        rationale=Rationale(
            summary="rebalance",
            metric_table_citations=("   ", ""),  # all empty / whitespace
            timeline_citations=("env rank0 ...",),
        ),
    )
    result = validator.validate(output)
    assert not result.ok


def test_validator_accepts_non_placement_delta_with_summary_only() -> None:
    validator = CriticOutputValidator(KnobSchema())
    output = CriticOutput(
        delta={KNOB_MICRO_BATCH_SIZE: 32},
        rationale=Rationale(summary="shrink mbs"),
    )
    assert validator.validate(output).ok


def test_validator_rejects_non_placement_delta_with_empty_summary() -> None:
    validator = CriticOutputValidator(KnobSchema())
    output = CriticOutput(
        delta={KNOB_MICRO_BATCH_SIZE: 32},
        rationale=Rationale(summary="   "),
    )
    assert not validator.validate(output).ok


def test_validator_rejects_pinned_knob_via_schema() -> None:
    validator = CriticOutputValidator(KnobSchema())
    output = CriticOutput(
        delta={KNOB_GLOBAL_BATCH_SIZE: 1024},
        rationale=Rationale(summary="should be rejected by schema"),
    )
    result = validator.validate(output)
    assert not result.ok
    assert "schema" in result.reason


# ---------------------------------------------------------------------------
# CodexCritic with injected transport
# ---------------------------------------------------------------------------


def _make_codex_response(delta: dict, summary: str, metric_cits=(), timeline_cits=()) -> str:
    return json.dumps(
        {
            "delta": delta,
            "rationale": {
                "summary": summary,
                "metric_table_citations": list(metric_cits),
                "timeline_citations": list(timeline_cits),
            },
        }
    )


def test_codex_critic_returns_first_valid_output() -> None:
    schema = KnobSchema()
    response = _make_codex_response({KNOB_ACTOR_OFFLOAD: True}, "offload actor")
    critic = CodexCritic(schema=schema, transport=lambda _p: response)
    output = critic.propose(
        history=[],
        current_knobs={},
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    assert output.delta == {KNOB_ACTOR_OFFLOAD: True}


def test_codex_critic_forwards_session_flag(monkeypatch) -> None:
    """codex_session is passed through to ask-codex.sh as --codex-session;
    left unset, the flag is absent (default one-shot behaviour)."""
    import types

    from toolkits.embodied_tuner import critic as critic_mod

    response = _make_codex_response({KNOB_ACTOR_OFFLOAD: True}, "s")
    seen: dict[str, list[str]] = {}

    def fake_run(argv, **_kwargs):
        seen["argv"] = argv
        return types.SimpleNamespace(returncode=0, stdout=response, stderr="")

    monkeypatch.setattr(critic_mod.subprocess, "run", fake_run)
    schema = KnobSchema()

    CodexCritic(schema=schema, codex_session="camp-1")._invoke_transport("p")
    assert "--codex-session" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--codex-session") + 1] == "camp-1"

    CodexCritic(schema=schema)._invoke_transport("p")
    assert "--codex-session" not in seen["argv"]


def test_codex_critic_retries_after_malformed_json() -> None:
    schema = KnobSchema()
    responses = [
        "this is not json at all",
        _make_codex_response({KNOB_MICRO_BATCH_SIZE: 32}, "shrink"),
    ]

    def transport(_prompt: str) -> str:
        return responses.pop(0)

    critic = CodexCritic(schema=schema, transport=transport, max_retries=3)
    output = critic.propose(
        history=[],
        current_knobs={KNOB_MICRO_BATCH_SIZE: 64},
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    assert output.delta == {KNOB_MICRO_BATCH_SIZE: 32}


def test_codex_critic_retries_after_dual_source_failure() -> None:
    schema = KnobSchema()
    bad = _make_codex_response(
        _placement_delta(),
        "rebalance",
        metric_cits=("env/interact=275.4",),
        timeline_cits=(),  # missing timeline -> validator rejects
    )
    good = _make_codex_response(
        _placement_delta(),
        "rebalance",
        metric_cits=("env/interact=275.4",),
        timeline_cits=("env rank0 ...",),
    )
    responses = [bad, good]

    def transport(prompt: str) -> str:
        # The second call MUST include feedback about timeline_citations.
        if not responses:
            raise AssertionError("transport called more times than expected")
        if len(responses) == 1:
            assert "timeline_citations" in prompt or "validation" in prompt
        return responses.pop(0)

    critic = CodexCritic(schema=schema, transport=transport, max_retries=3)
    output = critic.propose(
        history=[],
        current_knobs={},
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    assert KNOB_PLACEMENT in output.delta


def test_codex_critic_gives_up_after_max_retries() -> None:
    schema = KnobSchema()
    # Always returns malformed JSON.
    critic = CodexCritic(
        schema=schema,
        transport=lambda _p: "not json",
        max_retries=2,
    )
    with pytest.raises(CriticError) as exc:
        critic.propose(
            history=[],
            current_knobs={},
            last_failure_mode=None,
            last_metric_summary=None,
            last_timeline_summary=None,
        )
    assert "attempts" in str(exc.value)


def test_codex_critic_records_transport_failure_in_transaction_log() -> None:
    schema = KnobSchema()

    def transport(_prompt: str) -> str:
        raise CriticError("ask-codex.sh exited with code 1: InvalidParameter")

    critic = CodexCritic(
        schema=schema, transport=transport, max_retries=3, transport_retries=0
    )
    with pytest.raises(CriticError):
        critic.propose(
            history=[],
            current_knobs={},
            last_failure_mode=None,
            last_metric_summary=None,
            last_timeline_summary=None,
        )
    # A transport failure must still leave the failing attempt on the
    # transaction log so the scheduler can persist it for debugging.
    assert len(critic.transaction_log) == 1
    record = critic.transaction_log[0]
    assert record["attempt"] == 0
    assert "InvalidParameter" in record["parse_error"]
    assert record["prompt_debug"]  # captured the prompt Codex saw


def test_codex_critic_retries_transient_transport_failure() -> None:
    schema = KnobSchema()
    good = _make_codex_response({KNOB_MICRO_BATCH_SIZE: 32}, "shrink")
    calls = {"n": 0}

    def transport(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:  # two transient failures, then success
            raise CriticError("ask-codex.sh exited with code 1: InvalidParameter")
        return good

    critic = CodexCritic(
        schema=schema,
        transport=transport,
        transport_retries=2,
        transport_retry_backoff_s=0,  # no real sleep in tests
    )
    output = critic.propose(
        history=[],
        current_knobs={KNOB_MICRO_BATCH_SIZE: 64},
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    assert output.delta == {KNOB_MICRO_BATCH_SIZE: 32}
    assert calls["n"] == 3  # 1 initial + 2 retries
    # The transient failures are transparent to the propose loop: only the
    # single successful transport exchange lands on the transaction log.
    assert len(critic.transaction_log) == 1
    assert critic.transaction_log[0]["validation_ok"] is True


def test_codex_critic_gives_up_after_transport_retries_exhausted() -> None:
    schema = KnobSchema()
    calls = {"n": 0}

    def transport(_prompt: str) -> str:
        calls["n"] += 1
        raise CriticError("ask-codex.sh exited with code 1: InvalidParameter")

    critic = CodexCritic(
        schema=schema,
        transport=transport,
        transport_retries=2,
        transport_retry_backoff_s=0,
    )
    with pytest.raises(CriticError):
        critic.propose(
            history=[],
            current_knobs={},
            last_failure_mode=None,
            last_metric_summary=None,
            last_timeline_summary=None,
        )
    assert calls["n"] == 3  # 1 initial + 2 retries, then give up




def test_build_prompt_contains_all_required_sections() -> None:
    schema = KnobSchema()
    prompt = build_prompt(
        history=[],
        current_knobs={KNOB_MICRO_BATCH_SIZE: 80},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    text = str(prompt)
    assert "Concepts" in text
    assert "Trial History" in text  # "(none — first round)" branch
    assert "Current knob values" in text
    assert "Hard constraints" in text
    assert "Required output JSON shape" in text


def test_build_prompt_history_block_renders_each_trial() -> None:
    schema = KnobSchema()
    history = [
        TrialHistoryEntry(
            trial_idx=0,
            delta={KNOB_MICRO_BATCH_SIZE: 80},
            status="OK",
            failure_mode="NONE",
            objective=20.0,
            step_time=360.0,
            rationale_summary="baseline",
        ),
        TrialHistoryEntry(
            trial_idx=1,
            delta={KNOB_MICRO_BATCH_SIZE: 64},
            status="OK",
            failure_mode="NONE",
            objective=18.0,
            step_time=324.0,
            rationale_summary="shrink",
        ),
    ]
    prompt = build_prompt(
        history=history,
        current_knobs={KNOB_MICRO_BATCH_SIZE: 64},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    text = str(prompt)
    assert "trial 0" in text and "trial 1" in text
    assert "baseline" in text and "shrink" in text


def test_build_prompt_memory_pressure_only_when_last_failure_was_oom() -> None:
    schema = KnobSchema()
    prompt_oom = build_prompt(
        history=[],
        current_knobs={},
        schema=schema,
        last_failure_mode="OOM",
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    prompt_ok = build_prompt(
        history=[],
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    # The wiki (03-inputs.md) quotes both the block header and describes
    # the block's purpose, so substring assertions on the header alone
    # can't distinguish "block rendered" from "wiki describes the block".
    # Assert on body text unique to _render_memory_pressure.
    assert "Prefer memory-reducing knobs in the next delta" in str(prompt_oom)
    assert "Prefer memory-reducing knobs in the next delta" not in str(prompt_ok)


def test_build_prompt_timeline_summary_rendered_when_supplied() -> None:
    schema = KnobSchema()
    timeline = {
        "stall_fraction_by_component": {"env": 0.4, "rollout": 0.1, "actor": 0.05},
        "per_tag": [
            {"component": "env", "rank": 0, "tag": "env_interact_step", "call_count": 12, "duration_median": 0.5},
        ],
    }
    metric_summary = {"env/interact": 275.4, "actor/run_training": 21.3}
    text = str(
        build_prompt(
            history=[],
            current_knobs={},
            schema=schema,
            last_failure_mode=None,
            last_metric_summary=metric_summary,
            last_timeline_summary=timeline,
        )
    )
    assert "env/interact=275.4" in text
    assert "env_interact_step" in text
    assert "env: 0.400" in text


def test_build_prompt_feedback_block_appears_after_other_sections() -> None:
    schema = KnobSchema()
    text = str(
        build_prompt(
            history=[],
            current_knobs={},
            schema=schema,
            last_failure_mode=None,
            last_metric_summary=None,
            last_timeline_summary=None,
            feedback="please re-emit valid JSON",
        )
    )
    assert "Feedback on previous response" in text
    assert text.rfind("Feedback") > text.rfind("Required output JSON shape")


# ---------------------------------------------------------------------------
# FakeCritic
# ---------------------------------------------------------------------------


def test_fake_critic_replays_outputs_in_order() -> None:
    critic = FakeCritic.from_deltas(
        {KNOB_MICRO_BATCH_SIZE: 64},
        {KNOB_MICRO_BATCH_SIZE: 32},
    )
    schema = KnobSchema()
    out1 = critic.propose(
        history=[],
        current_knobs={KNOB_MICRO_BATCH_SIZE: 80},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    out2 = critic.propose(
        history=[],
        current_knobs={KNOB_MICRO_BATCH_SIZE: 64},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    assert out1.delta == {KNOB_MICRO_BATCH_SIZE: 64}
    assert out2.delta == {KNOB_MICRO_BATCH_SIZE: 32}
    assert len(critic.calls) == 2
    assert critic.calls[1][1] == {KNOB_MICRO_BATCH_SIZE: 64}


def test_fake_critic_raises_when_exhausted() -> None:
    critic = FakeCritic.from_deltas({KNOB_MICRO_BATCH_SIZE: 32})
    schema = KnobSchema()
    critic.propose(history=[], current_knobs={}, schema=schema, last_failure_mode=None, last_metric_summary=None, last_timeline_summary=None)
    with pytest.raises(CriticError):
        critic.propose(history=[], current_knobs={}, schema=schema, last_failure_mode=None, last_metric_summary=None, last_timeline_summary=None)


def test_fake_critic_stop_after_marks_final_output() -> None:
    critic = FakeCritic.stop_after(
        {KNOB_MICRO_BATCH_SIZE: 64},
        {KNOB_MICRO_BATCH_SIZE: 32},
    )
    schema = KnobSchema()
    o1 = critic.propose(history=[], current_knobs={}, schema=schema, last_failure_mode=None, last_metric_summary=None, last_timeline_summary=None)
    o2 = critic.propose(history=[], current_knobs={}, schema=schema, last_failure_mode=None, last_metric_summary=None, last_timeline_summary=None)
    assert o1.stop_requested is False
    assert o2.stop_requested is True


# ---------------------------------------------------------------------------
# Timeline summary rendering (new sections from timeline_processor)
# ---------------------------------------------------------------------------


def test_prompt_renders_per_component_bubble_section() -> None:
    schema = KnobSchema()
    summary = {
        "stall_fraction_by_component": {},
        "per_tag": [],
        "critical_path": {},
        "outliers": [],
        "per_component_bubble": {
            "wall_s": 100.0,
            "per_component": {
                "env": {
                    "num_ranks": 4, "busy_s": 30.0, "bubble_s": 70.0,
                    "bubble_frac": 0.70,
                    "per_rank": {
                        "0": {"busy_s": 30.0, "bubble_s": 70.0, "bubble_frac": 0.70},
                        "1": {"busy_s": 30.0, "bubble_s": 70.0, "bubble_frac": 0.70},
                    },
                },
                "rollout": {
                    "num_ranks": 4, "busy_s": 90.0, "bubble_s": 10.0,
                    "bubble_frac": 0.10,
                    "per_rank": {
                        "0": {"busy_s": 90.0, "bubble_s": 10.0, "bubble_frac": 0.10},
                    },
                },
            },
        },
        "raw_excerpts": [],
    }
    prompt = build_prompt(
        history=(), current_knobs={}, schema=schema,
        last_failure_mode=None, last_metric_summary=None,
        last_timeline_summary=summary,
    )
    text = str(prompt)
    assert "per-component bubble" in text
    assert "wall_s=100.0" in text
    assert "env: busy_s=30.0 bubble_s=70.0 bubble_frac=0.7 ranks=4" in text
    assert "rollout: busy_s=90.0 bubble_s=10.0 bubble_frac=0.1 ranks=4" in text
    assert "r0: busy_s=30.0" in text


def test_prompt_renders_critical_path_with_blocking_explainer() -> None:
    schema = KnobSchema()
    summary = {
        "stall_fraction_by_component": {},
        "per_tag": [],
        "critical_path": {
            0: {"step_span_s": 100.0, "real_busy_top": [
                {"component": "rollout", "rank": 0, "real_s": 90.0,
                 "blocked_s": 0.0, "real_frac": 0.9},
                {"component": "actor", "rank": 0, "real_s": 10.0,
                 "blocked_s": 90.0, "real_frac": 0.1},
            ]},
        },
        "outliers": [], "per_component_bubble": {}, "raw_excerpts": [],
    }
    prompt = build_prompt(
        history=(), current_knobs={}, schema=schema,
        last_failure_mode=None, last_metric_summary=None,
        last_timeline_summary=summary,
    )
    text = str(prompt)
    # The blocking-wait concept must be explained so the critic doesn't
    # read recv_traj as actor work
    assert "actor/recv_traj" in text
    assert "blocking-wait" in text
    assert "real=90.0s  blocked=0.0s" in text


def test_prompt_renders_outliers_with_knob_hint() -> None:
    schema = KnobSchema()
    summary = {
        "stall_fraction_by_component": {},
        "per_tag": [],
        "critical_path": {},
        "outliers": [
            {"tag": "env_interact_step", "component": "env", "rank": 0,
             "global_step": None, "dur_s": 45.0,
             "knob_hint": "env.enable_offload=True"},
        ],
        "per_component_bubble": {}, "raw_excerpts": [],
    }
    prompt = build_prompt(
        history=(), current_knobs={}, schema=schema,
        last_failure_mode=None, last_metric_summary=None,
        last_timeline_summary=summary,
    )
    text = str(prompt)
    assert "outlier events" in text
    assert "env.enable_offload=True" in text


def test_prompt_renders_raw_excerpts_as_jsonl() -> None:
    schema = KnobSchema()
    summary = {
        "stall_fraction_by_component": {},
        "per_tag": [],
        "critical_path": {},
        "outliers": [],
        "per_component_bubble": {},
        "raw_excerpts": [
            {"component": "rollout", "rank": 0, "tag": "rollout/generate",
             "global_step": 0, "dur_s": 273.6, "qualname": "MSR.generate",
             "call_index": 0},
        ],
    }
    prompt = build_prompt(
        history=(), current_knobs={}, schema=schema,
        last_failure_mode=None, last_metric_summary=None,
        last_timeline_summary=summary,
    )
    text = str(prompt)
    assert "raw timeline excerpts" in text
    # Each excerpt rendered as a JSON line the critic can cite verbatim
    assert '"qualname": "MSR.generate"' in text


# ---------------------------------------------------------------------------
# bitter_lesson: parse + validate + prompt rendering
# ---------------------------------------------------------------------------


def _make_lesson(idx: int = 3, mode: str = "OOM") -> BitterLesson:
    return BitterLesson(
        trigger=f"trial {idx} OOMed after rollout.enable_offload=False",
        rule="do not disable rollout offload while total_num_envs >= 8",
        trial_idx=idx,
        failure_mode=mode,
        delta_signature=canonical_delta_signature({"rollout.enable_offload": False}),
    )


def test_parse_critic_output_accepts_bitter_lesson() -> None:
    text = json.dumps(
        {
            "delta": {"actor.micro_batch_size": 20},
            "rationale": {"summary": "shrink mbs"},
            "bitter_lesson": {
                "trigger": "trial 2 OOMed",
                "rule": "avoid rollout.enable_offload=False under total_num_envs>=8",
            },
        }
    )
    out = parse_critic_output(text)
    assert out.bitter_lesson is not None
    assert out.bitter_lesson.trigger == "trial 2 OOMed"
    assert "rollout.enable_offload" in out.bitter_lesson.rule


def test_parse_critic_output_accepts_missing_bitter_lesson() -> None:
    text = json.dumps(
        {
            "delta": {"actor.micro_batch_size": 20},
            "rationale": {"summary": "shrink mbs"},
        }
    )
    assert parse_critic_output(text).bitter_lesson is None


def test_parse_critic_output_treats_empty_bitter_lesson_as_absent() -> None:
    text = json.dumps(
        {
            "delta": {"actor.micro_batch_size": 20},
            "rationale": {"summary": "shrink mbs"},
            "bitter_lesson": {"trigger": "", "rule": ""},
        }
    )
    assert parse_critic_output(text).bitter_lesson is None


def test_parse_critic_output_rejects_non_object_bitter_lesson() -> None:
    text = json.dumps(
        {
            "delta": {"x": 1},
            "rationale": {"summary": "s"},
            "bitter_lesson": "not an object",
        }
    )
    with pytest.raises(CriticError):
        parse_critic_output(text)


def test_parse_critic_output_rejects_non_string_lesson_fields() -> None:
    text = json.dumps(
        {
            "delta": {"x": 1},
            "rationale": {"summary": "s"},
            "bitter_lesson": {"trigger": 42, "rule": "r"},
        }
    )
    with pytest.raises(CriticError):
        parse_critic_output(text)


def test_validator_requires_lesson_after_oom() -> None:
    schema = KnobSchema()
    validator = CriticOutputValidator(schema=schema, last_failure_mode="OOM")
    output = CriticOutput(
        delta={"actor.micro_batch_size": 20},
        rationale=Rationale(summary="ok"),
    )
    result = validator.validate(output)
    assert result.ok is False
    assert "bitter_lesson" in result.reason


def test_validator_accepts_lesson_after_oom() -> None:
    schema = KnobSchema()
    validator = CriticOutputValidator(schema=schema, last_failure_mode="OOM")
    output = CriticOutput(
        delta={"actor.micro_batch_size": 20},
        rationale=Rationale(summary="ok"),
        bitter_lesson=ProposedLesson(
            trigger="trial 2 OOMed", rule="avoid rollout offload disable"
        ),
    )
    assert validator.validate(output).ok is True


def test_validator_does_not_require_lesson_after_ok_trial() -> None:
    schema = KnobSchema()
    validator = CriticOutputValidator(schema=schema, last_failure_mode="NONE")
    output = CriticOutput(
        delta={"actor.micro_batch_size": 20},
        rationale=Rationale(summary="ok"),
    )
    assert validator.validate(output).ok is True


def test_validator_does_not_require_lesson_when_first_round() -> None:
    schema = KnobSchema()
    validator = CriticOutputValidator(schema=schema, last_failure_mode=None)
    output = CriticOutput(
        delta={"actor.micro_batch_size": 20},
        rationale=Rationale(summary="ok"),
    )
    assert validator.validate(output).ok is True


def test_validator_allows_stop_requested_without_lesson_after_failure() -> None:
    """A critic that concedes the campaign is over shouldn't be blocked."""
    schema = KnobSchema()
    validator = CriticOutputValidator(schema=schema, last_failure_mode="OOM")
    output = CriticOutput(
        delta={},
        rationale=Rationale(summary="no more moves"),
        stop_requested=True,
    )
    assert validator.validate(output).ok is True


def test_validator_rejects_lesson_with_blank_trigger() -> None:
    schema = KnobSchema()
    validator = CriticOutputValidator(schema=schema, last_failure_mode="OOM")
    output = CriticOutput(
        delta={"actor.micro_batch_size": 20},
        rationale=Rationale(summary="ok"),
        bitter_lesson=ProposedLesson(trigger="   ", rule="do not X"),
    )
    assert validator.validate(output).ok is False


def test_validator_requires_lesson_for_config_invalid_failure() -> None:
    schema = KnobSchema()
    validator = CriticOutputValidator(
        schema=schema, last_failure_mode="CONFIG_INVALID"
    )
    output = CriticOutput(
        delta={"actor.micro_batch_size": 20},
        rationale=Rationale(summary="ok"),
    )
    assert validator.validate(output).ok is False


def test_build_prompt_renders_bitter_lessons_before_history() -> None:
    schema = KnobSchema()
    lessons = [_make_lesson(2), _make_lesson(6, mode="WORKER_CRASH")]
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
        bitter_lessons=lessons,
    )
    text = str(prompt)
    assert "Bitter Lessons" in text
    assert "[trial 2, OOM]" in text
    assert "[trial 6, WORKER_CRASH]" in text
    # Lessons section should appear before trial history.
    assert text.index("Bitter Lessons") < text.index("Trial History")


def test_build_prompt_omits_bitter_lessons_block_when_empty() -> None:
    schema = KnobSchema()
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
        bitter_lessons=(),
    )
    # Same reasoning as memory-pressure test above: 03-inputs.md quotes the
    # Bitter Lessons header, so assert on body text unique to
    # _render_bitter_lessons instead.
    assert "Each lesson was written by an earlier round" not in str(prompt)


def test_debug_text_includes_bitter_lessons_block() -> None:
    schema = KnobSchema()
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
        bitter_lessons=(_make_lesson(2),),
    )
    assert "Bitter Lessons" in prompt.to_debug_text()


def test_codex_critic_retries_when_lesson_missing_after_failure() -> None:
    """First response omits bitter_lesson after an OOM; validator rejects,
    critic retries with feedback, second response includes it and is accepted."""
    schema = KnobSchema()
    responses = [
        json.dumps(
            {
                "delta": {"actor.micro_batch_size": 20},
                "rationale": {"summary": "shrink mbs"},
            }
        ),
        json.dumps(
            {
                "delta": {"actor.micro_batch_size": 20},
                "rationale": {"summary": "shrink mbs"},
                "bitter_lesson": {
                    "trigger": "trial 2 OOM after rollout.enable_offload=False",
                    "rule": "do not disable rollout offload at total_num_envs>=8",
                },
            }
        ),
    ]
    prompts_seen: list[str] = []

    def transport(prompt: str) -> str:
        prompts_seen.append(prompt)
        return responses.pop(0)

    critic = CodexCritic(schema=schema, transport=transport)
    out = critic.propose(
        history=[],
        current_knobs={},
        last_failure_mode="OOM",
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    assert out.bitter_lesson is not None
    assert out.bitter_lesson.trigger.startswith("trial 2 OOM")
    # Second prompt carries the retry feedback quoting the validator reason.
    assert "bitter_lesson" in prompts_seen[1]


def test_build_prompt_forwards_bitter_lessons_arg_default_is_empty() -> None:
    schema = KnobSchema()
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
    )
    assert prompt.bitter_lessons_block == ""


# ---------------------------------------------------------------------------
# GPU-memory prompt blocks (memory_summary_block / memory_verbose_block)
# ---------------------------------------------------------------------------

def _mem_summary(**overrides):
    base = {
        "samples": 100,
        "span_s": 50.0,
        "gpu_total_gib": 80.0,
        "peak_gpu_mem_gib": 61.0,
        "peak_mem_util_percent": 57.0,
        "per_gpu": (
            {"index": 0, "avg_mem": 24.0, "max_mem": 61.0, "avg_gpu_util": 69.0,
             "max_gpu_util": 100.0, "avg_mem_util": 15.0, "max_mem_util": 57.0,
             "memory_total_gib": 80.0},
        ),
        "per_process": (
            {"label": "actor/r0/pid1", "component": "actor", "rank": 0, "pid": 1,
             "avg_rss": 5.7, "max_rss": 7.8, "avg_cpu": 46.0, "max_cpu": 102.0,
             "avg_process_gpu_mem": 17.9, "max_process_gpu_mem": 61.0,
             "avg_process_gpu_util": 33.0, "max_process_gpu_util": 100.0,
             "gpu_indices": [0]},
        ),
        "raw_nvitop_jsonl": {},
        "plot_paths": {},
    }
    base.update(overrides)
    return base


def test_memory_summary_block_rendered_when_supplied() -> None:
    schema = KnobSchema()
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
        last_memory_summary=_mem_summary(),
    )
    txt = str(prompt)
    # Assert on body text unique to the rendered block — the wiki
    # (03-inputs.md §7.5) quotes the headers verbatim, so a header-only
    # substring can't tell "block rendered" from "wiki describes block"
    # (same gotcha the OOM memory-pressure test calls out).
    assert "peak_process_gpu_mem=61.000 GiB" in txt
    assert "device_cap=80.000 GiB" in txt
    assert "actor/r0/pid1" in txt
    # Per-process row carries max_gpu_util alongside memory + gpu_indices.
    assert "max_process_gpu_mem=61.000 GiB max_gpu_util=100.0% [gpu 0]" in txt
    # 61/80 = 76% occupancy < 95% threshold -> no soft-pressure WARNING.
    # (Use the rendered form — the wiki §7.5 describes the warning prefix
    # verbatim, so the bare prefix matches the wiki too.)
    assert "WARNING: memory pressure — peak=" not in txt
    # Default NONE -> raw nvitop block absent (no fenced raw dump).
    assert "### actor_rank0_pid1" not in txt


def test_memory_summary_block_soft_pressure_warning() -> None:
    schema = KnobSchema()
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,  # NOT OOM — soft pressure must fire anyway
        last_metric_summary=None,
        last_timeline_summary=None,
        last_memory_summary=_mem_summary(
            peak_mem_util_percent=90.0, peak_gpu_mem_gib=77.0,
        ),
    )
    txt = str(prompt)
    assert "WARNING: memory pressure — peak=" in txt
    assert "96% of cap" in txt
    assert "did not OOM" in txt


def test_memory_summary_block_absent_when_none() -> None:
    schema = KnobSchema()
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode="OOM",
        last_metric_summary=None,
        last_timeline_summary=None,
        last_memory_summary=None,
    )
    txt = str(prompt)
    # OOM directive still present.
    assert "Prefer memory-reducing knobs in the next delta" in txt
    # But no numeric memory block (body marker unique to the rendered
    # block; the wiki does not quote the rendered line format).
    assert "peak_process_gpu_mem=" not in txt
    assert "device_cap=" not in txt


def test_memory_verbose_block_only_when_raw_nvitop_present() -> None:
    schema = KnobSchema()
    raw = {"actor_rank0_pid1": '{"ts": 1.0, "gpus": []}\n'}
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
        last_memory_summary=_mem_summary(raw_nvitop_jsonl=raw),
    )
    txt = str(prompt)
    # The fenced raw dump + per-stem heading only appear when raw is injected.
    assert "```jsonl" in txt
    assert "### actor_rank0_pid1" in txt


def test_to_debug_text_keeps_memory_summary_drops_verbose() -> None:
    schema = KnobSchema()
    raw = {"actor_rank0_pid1": '{"ts": 1.0, "gpus": []}\n'}
    prompt = build_prompt(
        history=(),
        current_knobs={},
        schema=schema,
        last_failure_mode=None,
        last_metric_summary=None,
        last_timeline_summary=None,
        last_memory_summary=_mem_summary(raw_nvitop_jsonl=raw),
    )
    debug = prompt.to_debug_text()
    # Compact summary survives (debugger needs the peak signal).
    assert "peak_process_gpu_mem=61.000 GiB" in debug
    # Verbose raw dump dropped (mirrors timeline_verbose_block).
    assert "### actor_rank0_pid1" not in debug
    assert "```jsonl" not in debug
