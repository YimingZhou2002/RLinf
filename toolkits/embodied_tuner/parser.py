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

"""Log + timeline parser for the embodied auto-tuner.

Inputs (per trial, all live under ``LOG_DIR``):

- ``metrics.log``: RLinf's MetricTable output, one block per global step.
  See ``rlinf/utils/metric_utils.py`` and a live sample at
  ``logs/20260629-07:25:33-maniskill_ppo_openvla/metrics.log``. Each
  block looks like::

      ╭─...─╮
      ├──── Metric Table ────┤
      │ Global Step:    1/3 │ ... │ Step Time: 359.973s ...
      ├──── Time ────┤
      │env/interact=275.4 │ rollout/generate_one_epoch=268.8 │ ...
      ├──── Environment ────┤
      │num_trajectories=18 │ ...
      ╰─...─╯

- ``timeline/*.jsonl`` (one file per component+rank): per-call records
  ``{"t0", "t1", "tag", "component", "rank", "call_index", ...}`` emitted
  by ``profiler/rlinf_timeline/autopatch.py``.

Outputs:

- :class:`TrialResult` carrying ``Status``, ``FailureMode``, objective,
  per-step values, the timeline summary the critic prompt consumes
  (AC-7), and an optional ``peak_gpu_mem`` field (best-effort metadata).

The objective is computed as ``mean(step_time[1:N]) / num_trajectories``
where step indices are 0-based: step 0 (the first ``Global Step: 1/N``
block) is dropped as warmup, the remainder is averaged, and
``num_trajectories`` is taken from the FINAL MetricTable block — exactly
the rule documented under AC-6 in the plan.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Status(str, Enum):
    """Coarse-grained trial outcome."""

    OK = "OK"
    FAILED = "FAILED"


class FailureMode(str, Enum):
    """Reason a trial is not ``(OK, NONE)``.

    ``(FAILED, NONE)`` is invariant-violating; the parser refuses to
    construct that combination via :func:`make_result`.
    """

    NONE = "NONE"
    METRICS_PARTIAL = "METRICS_PARTIAL"
    METRICS_MISSING = "METRICS_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"
    LAUNCH_FAILURE = "LAUNCH_FAILURE"
    OOM = "OOM"
    WORKER_CRASH = "WORKER_CRASH"
    TIMEOUT = "TIMEOUT"


class ParserInvariantError(AssertionError):
    """Raised when constructing an invalid ``(Status, FailureMode)`` pair."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricStep:
    """A single MetricTable block extracted from ``metrics.log``.

    Attributes:
        global_step: The 1-based index parsed from ``Global Step: X/Y``.
        total_steps: The ``Y`` denominator.
        step_time_seconds: Cumulative ``elapsed / steps_done`` reported on
            this block (NOT the per-step delta — that's a quirk of how
            ``rlinf/utils/metric_utils.py`` renders Step Time).
        num_trajectories: ``num_trajectories`` parsed from the Environment
            section (``None`` when absent).
        time_keys: Raw ``key=value`` pairs from the Time section (e.g.
            ``env/interact=275.4``). Used for critic context only.
    """

    global_step: int
    total_steps: int
    step_time_seconds: float | None
    num_trajectories: int | None
    time_keys: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TagStats:
    """Per-rank stats for one timeline tag."""

    component: str
    rank: int
    tag: str
    call_count: int
    duration_min: float
    duration_median: float
    duration_max: float
    duration_total: float


@dataclass(frozen=True)
class TimelineSummary:
    """Aggregated timeline events consumed by the AC-7 critic prompt.

    Attributes:
        per_tag: Stats grouped by ``(component, rank, tag)``.
        window_start: Earliest ``t0`` observed across all events
            (``None`` when no events exist).
        window_end: Latest ``t1`` observed across all events.
        stall_fraction_by_component: For each component, the fraction of
            the observation window NOT covered by any of its events.
            Captures pipeline / channel waits.
    """

    per_tag: tuple[TagStats, ...] = ()
    window_start: float | None = None
    window_end: float | None = None
    stall_fraction_by_component: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    """Outcome of parsing a single trial directory.

    Best-config selection (see :func:`select_best`) requires
    ``status == OK`` AND ``failure_mode == NONE``. ``(OK, METRICS_PARTIAL)``
    trials remain usable as critic context but are NEVER selected as best.
    """

    log_dir: Path
    status: Status
    failure_mode: FailureMode
    reason: str = ""
    step_time_seconds: float | None = None  # averaged across steps 2..N
    num_trajectories: int | None = None  # from the FINAL MetricTable block
    objective: float | None = None
    per_step: tuple[MetricStep, ...] = ()
    timeline_summary: TimelineSummary | None = None
    peak_gpu_mem_gib: float | None = None
    returncode: int | None = None

    def __post_init__(self) -> None:
        if self.status is Status.FAILED and self.failure_mode is FailureMode.NONE:
            raise ParserInvariantError(
                "TrialResult(FAILED, NONE) is invalid: every FAILED trial must "
                "carry a non-NONE failure_mode"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_OOM_REGEX = re.compile(
    r"(CUDA out of memory|torch\.cuda\.OutOfMemoryError|Ray actor.*?(died|killed).*OOM)",
    re.IGNORECASE | re.DOTALL,
)
_WORKER_CRASH_REGEX = re.compile(
    r"(RayActorError|ActorDiedError|killed by signal|Traceback \(most recent call last\))",
    re.IGNORECASE,
)


def parse_trial(
    log_dir: Path | str,
    *,
    returncode: int | None = None,
    timed_out: bool = False,
    failure_mode_override: FailureMode | None = None,
    stderr_path: Path | str | None = None,
) -> TrialResult:
    """Parse a trial directory and return a :class:`TrialResult`.

    Args:
        log_dir: The trial's log directory (the path the runner returned).
        returncode: Subprocess exit code (``None`` means the subprocess
            never started — caller should pass ``LAUNCH_FAILURE`` via
            ``failure_mode_override``).
        timed_out: When ``True`` the runner timed the trial out; the
            parser will classify ``(FAILED, TIMEOUT)`` regardless of what
            ``metrics.log`` looks like.
        failure_mode_override: Optional escape hatch for early-failure
            modes the scheduler knows about before parsing
            (``CONFIG_INVALID``, ``LAUNCH_FAILURE``). Setting this short-
            circuits classification.
        stderr_path: Optional path to a captured stderr/stdout log used
            for the OOM and worker-crash rubrics.

    Returns:
        A :class:`TrialResult`. Always returns; never raises on missing
        files (those map to ``METRICS_MISSING``).
    """
    log_dir = Path(log_dir)

    if failure_mode_override is not None:
        if failure_mode_override is FailureMode.NONE:
            raise ValueError(
                "failure_mode_override=NONE is meaningless; pass None instead"
            )
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=failure_mode_override,
            reason=f"override: {failure_mode_override.value}",
            returncode=returncode,
        )

    if timed_out:
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=FailureMode.TIMEOUT,
            reason="runner timed out the trial",
            returncode=returncode,
        )

    metrics_path = log_dir / "metrics.log"
    timeline_dir = log_dir / "timeline"

    per_step: tuple[MetricStep, ...]
    if not metrics_path.is_file():
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=FailureMode.METRICS_MISSING,
            reason=f"metrics.log not found at {metrics_path}",
            returncode=returncode,
        )
    try:
        per_step = parse_metrics_log(metrics_path)
    except Exception as exc:  # noqa: BLE001 — surface parse errors as METRICS_MISSING
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=FailureMode.METRICS_MISSING,
            reason=f"failed to parse metrics.log: {exc}",
            returncode=returncode,
        )

    # OOM / worker-crash rubric: only applies when the subprocess exited
    # with a non-zero return code AND we have a stderr/stdout log to
    # inspect. The classifier returns the first matching mode.
    if returncode is not None and returncode != 0:
        oom_mode = _classify_failure(stderr_path)
        if oom_mode is not None:
            return TrialResult(
                log_dir=log_dir,
                status=Status.FAILED,
                failure_mode=oom_mode,
                reason=f"detected via {stderr_path}",
                returncode=returncode,
                per_step=per_step,
            )

    # If no MetricTable blocks parsed, treat as METRICS_MISSING.
    if not per_step:
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=FailureMode.METRICS_MISSING,
            reason="metrics.log contained no MetricTable blocks",
            returncode=returncode,
            per_step=(),
        )

    # Compute objective with warmup exclusion.
    objective, avg_step_time, partial_reason = compute_objective(per_step)
    final_num_traj = per_step[-1].num_trajectories

    # Timeline summary (best-effort; missing → METRICS_PARTIAL).
    timeline_summary: TimelineSummary | None = None
    timeline_partial_reason: str | None = None
    if timeline_dir.is_dir():
        try:
            timeline_summary = parse_timeline(timeline_dir)
        except Exception as exc:  # noqa: BLE001
            timeline_partial_reason = f"timeline parse error: {exc}"
    else:
        timeline_partial_reason = f"timeline/ directory absent at {timeline_dir}"

    # Decide (Status, FailureMode).
    reasons: list[str] = []
    if partial_reason:
        reasons.append(partial_reason)
    if timeline_partial_reason:
        reasons.append(timeline_partial_reason)
    if final_num_traj is None:
        reasons.append("final MetricTable block has no num_trajectories field")

    peak_mem = _read_peak_gpu_mem(log_dir / "nvitop")

    if returncode not in (None, 0):
        # Non-OOM, non-crash failure with usable metrics → WORKER_CRASH.
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=FailureMode.WORKER_CRASH,
            reason=f"subprocess exited with non-zero returncode={returncode}",
            returncode=returncode,
            per_step=per_step,
            timeline_summary=timeline_summary,
            peak_gpu_mem_gib=peak_mem,
        )

    if reasons:
        return TrialResult(
            log_dir=log_dir,
            status=Status.OK,
            failure_mode=FailureMode.METRICS_PARTIAL,
            reason="; ".join(reasons),
            step_time_seconds=avg_step_time,
            num_trajectories=final_num_traj,
            objective=None,  # not eligible for best-config selection
            per_step=per_step,
            timeline_summary=timeline_summary,
            peak_gpu_mem_gib=peak_mem,
            returncode=returncode,
        )

    return TrialResult(
        log_dir=log_dir,
        status=Status.OK,
        failure_mode=FailureMode.NONE,
        step_time_seconds=avg_step_time,
        num_trajectories=final_num_traj,
        objective=objective,
        per_step=per_step,
        timeline_summary=timeline_summary,
        peak_gpu_mem_gib=peak_mem,
        returncode=returncode,
    )


def select_best(results: Iterable[TrialResult]) -> TrialResult | None:
    """Return the trial with the lowest objective among ``(OK, NONE)`` only."""
    eligible = [
        r
        for r in results
        if r.status is Status.OK
        and r.failure_mode is FailureMode.NONE
        and r.objective is not None
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (r.objective, str(r.log_dir)))


# ---------------------------------------------------------------------------
# metrics.log parsing
# ---------------------------------------------------------------------------


# Pattern for the ``Global Step:    X/Y`` header line inside the table.
_GLOBAL_STEP_RE = re.compile(r"Global Step:\s*(\d+)\s*/\s*(\d+)")
_STEP_TIME_RE = re.compile(r"Step Time:\s*([0-9.]+)\s*s")
# Box-drawing characters used as block delimiters in ``metric_utils.py``.
_BLOCK_START_PREFIX = "╭"  # ╭
_BLOCK_END_PREFIX = "╰"  # ╰


def parse_metrics_log(path: Path) -> tuple[MetricStep, ...]:
    """Parse all MetricTable blocks in ``path``.

    Returns the blocks in document order; an empty tuple when the file
    contains no recognised blocks (caller maps this to
    ``METRICS_MISSING``).
    """
    text = path.read_text(errors="replace")
    blocks = _split_into_blocks(text)
    steps: list[MetricStep] = []
    for block in blocks:
        parsed = _parse_block(block)
        if parsed is not None:
            steps.append(parsed)
    return tuple(steps)


def _split_into_blocks(text: str) -> list[str]:
    """Slice ``text`` into MetricTable blocks demarcated by ``╭...╮``/``╰...╯``."""
    blocks: list[str] = []
    current: list[str] = []
    in_block = False
    for line in text.splitlines():
        if line.startswith(_BLOCK_START_PREFIX):
            current = [line]
            in_block = True
            continue
        if in_block:
            current.append(line)
            if line.startswith(_BLOCK_END_PREFIX):
                blocks.append("\n".join(current))
                current = []
                in_block = False
    return blocks


def _parse_block(block: str) -> MetricStep | None:
    """Parse a single MetricTable block.

    Returns ``None`` when the block lacks both ``Global Step:`` and any
    salvageable data; otherwise returns a :class:`MetricStep` with the
    fields we could extract (``None`` for the rest).
    """
    step_match = _GLOBAL_STEP_RE.search(block)
    if step_match is None:
        # Not a real MetricTable block (e.g. a stray header line).
        return None
    global_step = int(step_match.group(1))
    total_steps = int(step_match.group(2))

    step_time_match = _STEP_TIME_RE.search(block)
    step_time = float(step_time_match.group(1)) if step_time_match else None

    time_keys, env_keys = _extract_key_value_sections(block)
    num_traj_value = env_keys.get("num_trajectories")
    try:
        num_traj = int(float(num_traj_value)) if num_traj_value is not None else None
    except (TypeError, ValueError):
        num_traj = None

    return MetricStep(
        global_step=global_step,
        total_steps=total_steps,
        step_time_seconds=step_time,
        num_trajectories=num_traj,
        time_keys=time_keys,
    )


# A section header like ``├──── Time ────┤``.
_SECTION_HEADER_RE = re.compile(
    r"^├[─\-]+\s*([A-Za-z][A-Za-z/_ ]*?)\s*[─\-]+┤"
)
# A ``│key=value│`` cell. The pipe is U+2502; ``|`` is accepted as a fallback.
_KV_RE = re.compile(r"([A-Za-z][A-Za-z0-9_./]*)=([-+]?[0-9.eE]+|nan|inf)")


def _extract_key_value_sections(block: str) -> tuple[dict[str, float], dict[str, str]]:
    """Return ``(time_keys, env_keys)`` from the Time and Environment sections."""
    time_keys: dict[str, float] = {}
    env_keys: dict[str, str] = {}
    current_section: str | None = None
    for raw_line in block.splitlines():
        header = _SECTION_HEADER_RE.match(raw_line)
        if header is not None:
            current_section = header.group(1).strip().lower()
            continue
        if current_section is None:
            continue
        for key, value in _KV_RE.findall(raw_line):
            if current_section == "time":
                try:
                    time_keys[key] = float(value)
                except ValueError:
                    continue
            elif current_section == "environment":
                env_keys[key] = value
    return time_keys, env_keys


def compute_objective(
    per_step: Sequence[MetricStep],
) -> tuple[float | None, float | None, str | None]:
    """Return ``(objective, avg_step_time_seconds, partial_reason)``.

    ``objective`` is ``avg_step_time / num_trajectories`` where the
    average excludes step 1 (warmup). When fewer than 2 successful steps
    exist, or ``num_trajectories`` is missing on the final block, the
    objective is ``None`` and ``partial_reason`` explains why.
    """
    if len(per_step) < 2:
        return None, None, (
            "fewer than 2 MetricTable blocks: warmup exclusion leaves zero data points"
        )
    tail = per_step[1:]
    tail_times = [s.step_time_seconds for s in tail if s.step_time_seconds is not None]
    if not tail_times:
        return None, None, "no step_time values parsed from MetricTable blocks 2..N"
    avg_step_time = sum(tail_times) / len(tail_times)
    final_num_traj = per_step[-1].num_trajectories
    if final_num_traj is None:
        return None, avg_step_time, (
            "final MetricTable block has no num_trajectories field"
        )
    if final_num_traj <= 0:
        return None, avg_step_time, f"final num_trajectories={final_num_traj} not positive"
    return avg_step_time / final_num_traj, avg_step_time, None


# ---------------------------------------------------------------------------
# timeline parsing
# ---------------------------------------------------------------------------


# Tags we always summarise for the critic prompt (AC-7 consumes this).
# The list reflects what actually appears in ``timeline/*.jsonl`` emitted
# by ``profiler/rlinf_timeline/autopatch.py`` (verified against
# ``logs/20260629-07:25:33-maniskill_ppo_openvla/timeline/``). Note that
# these are finer-grained than the MetricTable per-component aggregates
# (``env/interact``, ``rollout/generate_one_epoch``, ``actor/run_training``,
# ``sync_weights``); the aggregates are available from
# :attr:`MetricStep.time_keys` and the timeline supplements them with
# per-rank min/median/max plus stall-fraction signals.
_HEADLINE_TAGS: tuple[str, ...] = (
    "env_interact_step",
    "env/bootstrap_step",
    "actor/recv_traj",
    "actor/sync_model_to_rollout",
    "actor/compute_adv",
    "rollout/generate",
    "predict",
)


def parse_timeline(timeline_dir: Path) -> TimelineSummary:
    """Aggregate every ``*.jsonl`` file under ``timeline_dir``."""
    events: list[dict] = []
    for path in sorted(timeline_dir.glob("*.jsonl")):
        try:
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    if not events:
        return TimelineSummary()

    window_start = min(float(e.get("t0", 0.0)) for e in events)
    window_end = max(float(e.get("t1", 0.0)) for e in events)

    grouped: dict[tuple[str, int, str], list[float]] = {}
    intervals_by_component: dict[str, list[tuple[float, float]]] = {}
    for event in events:
        tag = event.get("tag") or event.get("worker_timer")
        component = event.get("component")
        rank = event.get("rank")
        if tag is None or component is None or rank is None:
            continue
        try:
            t0 = float(event["t0"])
            t1 = float(event["t1"])
        except (KeyError, TypeError, ValueError):
            continue
        duration = max(t1 - t0, 0.0)
        grouped.setdefault((component, int(rank), tag), []).append(duration)
        intervals_by_component.setdefault(component, []).append((t0, t1))

    per_tag: list[TagStats] = []
    for (component, rank, tag) in sorted(grouped):
        # Only summarise the headline tags by default to keep the critic
        # prompt compact; the full event list is recomputable on demand.
        if tag not in _HEADLINE_TAGS:
            continue
        durations = grouped[(component, rank, tag)]
        if not durations:
            continue
        per_tag.append(
            TagStats(
                component=component,
                rank=rank,
                tag=tag,
                call_count=len(durations),
                duration_min=min(durations),
                duration_median=statistics.median(durations),
                duration_max=max(durations),
                duration_total=sum(durations),
            )
        )

    stall_fractions = {
        comp: _stall_fraction(intervals, window_start, window_end)
        for comp, intervals in intervals_by_component.items()
    }

    return TimelineSummary(
        per_tag=tuple(per_tag),
        window_start=window_start,
        window_end=window_end,
        stall_fraction_by_component=stall_fractions,
    )


def _stall_fraction(
    intervals: list[tuple[float, float]],
    window_start: float,
    window_end: float,
) -> float:
    """Fraction of ``[window_start, window_end]`` NOT covered by ``intervals``."""
    total = window_end - window_start
    if total <= 0 or not intervals:
        return 0.0
    sorted_iv = sorted(intervals)
    merged: list[list[float]] = []
    for start, end in sorted_iv:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(max(end - start, 0.0) for start, end in merged)
    stall = max(total - covered, 0.0)
    return stall / total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_failure(stderr_path: Path | str | None) -> FailureMode | None:
    """Inspect ``stderr_path`` (if any) and return OOM / WORKER_CRASH if matched."""
    if stderr_path is None:
        return None
    path = Path(stderr_path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    if _OOM_REGEX.search(text):
        return FailureMode.OOM
    if _WORKER_CRASH_REGEX.search(text):
        return FailureMode.WORKER_CRASH
    return None


def _read_peak_gpu_mem(nvitop_dir: Path) -> float | None:
    """Best-effort peak-GPU-memory read; ``None`` when ``nvitop/`` is absent."""
    summary = nvitop_dir / "nvitop_summary.log"
    if not summary.is_file():
        return None
    try:
        text = summary.read_text(errors="replace")
    except OSError:
        return None
    # nvitop_summary.log contains lines like ``max_process_gpu_mem=25.3 GiB``;
    # tolerate either form to keep this parser robust to schema drift.
    match = re.search(r"max_process_gpu_mem[\s:=]+([0-9.]+)\s*GiB", text, re.IGNORECASE)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
