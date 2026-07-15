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

The objective is computed as ``mean(step_time[0:N]) / num_trajectories``:
every parsed MetricTable block contributes to the averaged step_time
(including the first / ``Global Step: 1/N`` block), and
``num_trajectories`` is taken from the FINAL MetricTable block. A trial
with a single step is measurable — its lone step_time is used directly.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from toolkits.embodied_tuner.nvitop_feed import (
    NvitopFeedMode,
    collect_raw_nvitop_jsonl,
    discover_curve_plots,
    render_nvitop_summary,
)
from toolkits.embodied_tuner.timeline_feed import (
    JsonlFeedMode,
    collect_raw_jsonl,
    render_default_plots,
)


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
    DIVISIBILITY_VIOLATION = "DIVISIBILITY_VIOLATION"
    # Synthetic mode set on a DAGNode that short-circuits the runner
    # because its cumulative resolved-config SHA has already been
    # attempted in the campaign (see ConfigDedupIndex). The
    # accompanying ``duplicate_of_node_id`` back-reference on the
    # DAGNode always points at the ORIGINAL non-duplicate that
    # produced the recycled objective — never at a chain of duplicates.
    DUPLICATE_OF = "DUPLICATE_OF"


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
        critical_path: Per-``global_step`` summary of real-busy vs
            blocked busy time per ``(component, rank)`` lane. Produced
            by :func:`timeline_processor.compute_critical_path`. Empty
            ``{}`` when no events carry a global_step.
        outliers: Top-K longest events above per-tag P95 with an
            optional ``knob_hint`` linking the stall to a knob the
            critic can flip (e.g. ``env.enable_offload``).
        per_component_bubble: Per-component (env/rollout/actor) union-
            busy vs bubble breakdown, plus per-rank detail. Uses
            :data:`~toolkits.embodied_tuner.timeline_processor.BLOCKING_TAGS`
            to exclude wait-disguised-as-work events. Empty ``{}`` when
            no events remain after exclusion. Replaces the earlier per-
            GPU view — same signal but keyed on workload identity.
        raw_excerpts: Top-K longest raw events copied verbatim from the
            JSONL stream (runner-wrapper events excluded) so the critic
            sees full call context (``qualname``, ``call_index``, etc.).
        raw_jsonl: Optional mapping ``{"<component>_rank<N>": <file text>}``
            with unabridged JSONL contents. Populated by
            :func:`~toolkits.embodied_tuner.timeline_feed.collect_raw_jsonl`
            when a :class:`~toolkits.embodied_tuner.timeline_feed.JsonlFeedMode`
            other than ``NONE`` is active. Empty ``{}`` when disabled or
            when no JSONL files were selected.
        plot_paths: Optional mapping ``{fmt: path}`` recording the Gantt
            renders produced by
            :func:`~toolkits.embodied_tuner.timeline_feed.render_default_plots`
            (e.g. ``{"png": Path(".../timeline.png")}``). Empty ``{}`` when
            plotting was skipped or every format failed.
        component_call_averages: Steady-state per-call duration for
            ``env`` / ``rollout`` after skipping the first two events
            (drops bootstrap warmup). Produced by
            :func:`~toolkits.embodied_tuner.timeline_processor.compute_component_call_averages`.
            Empty ``{}`` when a component has <=2 non-blocking events.
    """

    per_tag: tuple[TagStats, ...] = ()
    window_start: float | None = None
    window_end: float | None = None
    stall_fraction_by_component: dict[str, float] = field(default_factory=dict)
    critical_path: dict[int, dict] = field(default_factory=dict)
    outliers: tuple[dict, ...] = ()
    per_component_bubble: dict[str, object] = field(default_factory=dict)
    raw_excerpts: tuple[dict, ...] = ()
    raw_jsonl: dict[str, str] = field(default_factory=dict)
    plot_paths: dict[str, Path] = field(default_factory=dict)
    component_call_averages: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class MemorySummary:
    """Aggregated GPU-memory summary for one trial (mirrors TimelineSummary).

    Built by :func:`_load_memory_summary` from the structured
    ``nvitop_summary.json`` sidecar produced by
    :func:`profiler.plot_nvitop.write_nvitop_summary` (with a text-log
    fallback for older trials). Unlike the legacy single-scalar
    ``peak_gpu_mem_gib``, this carries per-GPU / per-component breakdowns
    and a soft-pressure signal so the critic can reason quantitatively
    about memory — not just the boolean "OOM happened".

    The ``raw_nvitop_jsonl`` slot is a reserved insertion point for raw
    per-sample nvitop traces; it is ``{}`` by default (NvitopFeedMode.NONE)
    so raw traces do NOT enter the critic prompt unless explicitly opted
    into — exactly mirroring ``TimelineSummary.raw_jsonl``.
    """

    samples: int | None = None
    span_s: float | None = None
    aggregate_bin_s: float | None = None
    gpu_total_gib: float | None = None
    peak_gpu_mem_gib: float | None = None
    peak_mem_util_percent: float | None = None
    # Peak memory occupancy ratio (used/total) across all GPUs. This is the
    # "how full did device memory get" signal, distinct from
    # ``peak_mem_util_percent`` which is NVML's memory-controller busy ratio
    # (bandwidth). Soft-pressure logic keys off this field, not the
    # controller-util one.
    peak_mem_occ_percent: float | None = None
    per_gpu: tuple[dict, ...] = ()
    per_process: tuple[dict, ...] = ()
    raw_nvitop_jsonl: dict[str, str] = field(default_factory=dict)
    plot_paths: dict[str, Path] = field(default_factory=dict)


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
    step_time_seconds: float | None = None  # averaged across all MetricTable blocks
    num_trajectories: int | None = None  # from the FINAL MetricTable block
    objective: float | None = None
    per_step: tuple[MetricStep, ...] = ()
    timeline_summary: TimelineSummary | None = None
    peak_gpu_mem_gib: float | None = None
    memory_summary: MemorySummary | None = None
    returncode: int | None = None
    # Short (~40-line) tail-around-error slice for OOM / WORKER_CRASH /
    # DIVISIBILITY_VIOLATION / METRICS_MISSING trials. Empty for OK
    # trials and for override-classified failures (CONFIG_INVALID /
    # LAUNCH_FAILURE) where the caller already owns the reason string.
    error_excerpt: str = ""

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
# Divisibility violations show up in multiple RLinf subsystems:
#   - CommMapper.get_dst_ranks / get_src_ranks at
#     rlinf/scheduler/worker/routing.py:137-141:
#       "AssertionError: batch_size (64) must be divisible by dst_world_size (6)."
#   - validate_embodied_cfg at rlinf/config.py:962-978:
#       "AssertionError: total_num_envs ... must be divisible by env_world_size ..."
#   - Actor FSDP / rollout branches with modulo asserts like
#       "AssertionError: global_batch_size % (micro_batch_size * world_size) == 0"
#
# We match either an ``AssertionError`` whose message contains a
# "divisible"-shaped word (tolerating the common ``divisable`` typo) or
# a modulo comparison of the form ``X % Y == 0`` / ``!= 0``. Anchoring
# on ``AssertionError:`` keeps stray ``%`` in unrelated log chatter or
# Python source lines from misclassifying trials.
_DIVISIBILITY_REGEX = re.compile(
    r"AssertionError:[^\n]{0,500}?"
    r"(?:divis[ia]ble|%[^\n]{0,80}?[!=]=\s*0)",
    re.IGNORECASE,
)

# Number of lines to keep when extracting the tail-around-error for the
# critic prompt. ~40 lines is enough to cover the deepest Traceback we
# see from Ray-wrapped errors without ballooning the prompt.
_ERROR_EXCERPT_MAX_LINES = 40


def parse_trial(
    log_dir: Path | str,
    *,
    returncode: int | None = None,
    timed_out: bool = False,
    failure_mode_override: FailureMode | None = None,
    stderr_path: Path | str | None = None,
    enable_offload: Mapping[str, bool] | None = None,
    jsonl_feed_mode: JsonlFeedMode | None = None,
    nvitop_feed_mode: NvitopFeedMode | None = None,
    plot_formats: Iterable[str] = ("png",),
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
            for the OOM and worker-crash rubrics. When ``None`` the
            parser falls back to ``log_dir / "run_embodiment.log"`` (the
            file the runner writes by default with merged stdout+stderr).
        enable_offload: This trial's effective offload knob state
            ({"env": True, "rollout": False, "actor": True}). Used to
            attach knob hints to outliers.
        jsonl_feed_mode: Which JSONL files to load into
            ``TimelineSummary.raw_jsonl`` for the critic prompt. ``None``
            (default) means ``PER_COMPONENT_LATEST`` — per-component pick
            the rank whose events end latest. Set to ``NONE`` to skip
            (keeps prior behaviour). ``ALL`` dumps every file (may be
            hundreds of KB per trial — only viable for long-context
            critics).
        nvitop_feed_mode: Which raw nvitop JSONL files to load into
            ``MemorySummary.raw_nvitop_jsonl`` for the critic's
            ``memory_verbose_block``. ``None`` (default) means
            :attr:`~toolkits.embodied_tuner.nvitop_feed.NvitopFeedMode.NONE`
            — raw per-sample GPU traces do NOT enter the prompt; the critic
            sees only the aggregated :class:`MemorySummary` (per-GPU /
            per-component peak+avg, device cap, soft-pressure flag). Unlike
            the timeline, raw nvitop traces are large (~660 KB each), so raw
            injection is opt-in. ``PER_COMPONENT_LATEST`` picks the
            straggler rank per component; ``ALL`` dumps every file.
        plot_formats: Which Gantt formats to render alongside the trial
            via :func:`profiler.plot_timeline`. Defaults to ``("png",)``
            because that is what critics can attach as an image. Pass
            ``("png", "html")`` to also emit the interactive plot for
            human debugging, or ``()`` to skip plotting entirely.

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

    # GPU-memory summary is best-effort and computed once, early, so EVERY
    # downstream branch (TIMEOUT / OOM / WORKER_CRASH / METRICS_MISSING /
    # OK) can carry it. An OOM-killed trial that ran for a while still has
    # nvitop samples up to the crash — the critic needs that pre-crash
    # peak, so we do NOT gate this on success.
    nvitop_dir = log_dir / "nvitop"
    memory_summary = _load_memory_summary(nvitop_dir)
    active_nvitop_mode = (
        nvitop_feed_mode
        if nvitop_feed_mode is not None
        else NvitopFeedMode.NONE
    )
    if memory_summary is not None and active_nvitop_mode is not NvitopFeedMode.NONE:
        raw_nvitop = collect_raw_nvitop_jsonl(nvitop_dir, mode=active_nvitop_mode)
        if raw_nvitop:
            memory_summary = dataclasses.replace(
                memory_summary, raw_nvitop_jsonl=raw_nvitop
            )
    peak_mem = (
        memory_summary.peak_gpu_mem_gib if memory_summary is not None else None
    )

    if timed_out:
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=FailureMode.TIMEOUT,
            reason="runner timed out the trial",
            returncode=returncode,
            memory_summary=memory_summary,
        )

    metrics_path = log_dir / "metrics.log"
    timeline_dir = log_dir / "timeline"

    # Default stderr_path to the runner's merged stdout/stderr log so
    # OOM and worker-crash rubrics fire even when the caller forgets to
    # pass the path explicitly (per Round-2 Codex review).
    effective_stderr_path: Path | str | None = stderr_path
    if effective_stderr_path is None:
        default_stdout = log_dir / "run_embodiment.log"
        if default_stdout.is_file():
            effective_stderr_path = default_stdout

    # On nonzero return code we ALWAYS check the failure rubric first.
    # Otherwise an OOM-killed trial that never wrote ``metrics.log`` would
    # be classified as ``METRICS_MISSING`` instead of ``OOM`` — exactly
    # the misclassification Codex flagged.
    if returncode is not None and returncode != 0:
        oom_mode = _classify_failure(effective_stderr_path)
        if oom_mode is not None:
            per_step_for_failure: tuple[MetricStep, ...] = ()
            if metrics_path.is_file():
                try:
                    per_step_for_failure = parse_metrics_log(metrics_path)
                except Exception:  # noqa: BLE001
                    per_step_for_failure = ()
            return TrialResult(
                log_dir=log_dir,
                status=Status.FAILED,
                failure_mode=oom_mode,
                reason=f"detected via {effective_stderr_path}",
                returncode=returncode,
                per_step=per_step_for_failure,
                peak_gpu_mem_gib=peak_mem,
                memory_summary=memory_summary,
                error_excerpt=_extract_error_excerpt(effective_stderr_path, oom_mode),
            )

    per_step: tuple[MetricStep, ...]
    if not metrics_path.is_file():
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=FailureMode.METRICS_MISSING,
            reason=f"metrics.log not found at {metrics_path}",
            returncode=returncode,
            peak_gpu_mem_gib=peak_mem,
            memory_summary=memory_summary,
            error_excerpt=_extract_error_excerpt(
                effective_stderr_path, FailureMode.METRICS_MISSING
            ),
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
            peak_gpu_mem_gib=peak_mem,
            memory_summary=memory_summary,
            error_excerpt=_extract_error_excerpt(
                effective_stderr_path, FailureMode.METRICS_MISSING
            ),
        )

    # OOM / worker-crash rubric already ran above (before parsing
    # metrics.log) to ensure OOM-killed trials are classified as OOM
    # rather than METRICS_MISSING. By the time we reach this point a
    # nonzero returncode means the rubric did NOT match, so fall through
    # to the WORKER_CRASH branch later when we know more about the trial.

    # If no MetricTable blocks parsed, treat as METRICS_MISSING.
    if not per_step:
        return TrialResult(
            log_dir=log_dir,
            status=Status.FAILED,
            failure_mode=FailureMode.METRICS_MISSING,
            reason="metrics.log contained no MetricTable blocks",
            returncode=returncode,
            per_step=(),
            peak_gpu_mem_gib=peak_mem,
            memory_summary=memory_summary,
            error_excerpt=_extract_error_excerpt(
                effective_stderr_path, FailureMode.METRICS_MISSING
            ),
        )

    # Compute objective across all parsed MetricTable blocks.
    objective, avg_step_time, partial_reason = compute_objective(per_step)
    final_num_traj = per_step[-1].num_trajectories

    # Timeline summary (best-effort; missing → METRICS_PARTIAL).
    timeline_summary: TimelineSummary | None = None
    timeline_partial_reason: str | None = None
    if timeline_dir.is_dir():
        try:
            timeline_summary = parse_timeline(
                timeline_dir,
                enable_offload=enable_offload,
            )
        except Exception as exc:  # noqa: BLE001
            timeline_partial_reason = f"timeline parse error: {exc}"
    else:
        timeline_partial_reason = f"timeline/ directory absent at {timeline_dir}"

    # Best-effort side-effects on the timeline dir: render the Gantt
    # plot(s) and load raw JSONL text for the critic prompt. Both are
    # additive to TimelineSummary and never affect trial classification.
    if timeline_summary is not None and timeline_dir.is_dir():
        active_mode = (
            jsonl_feed_mode
            if jsonl_feed_mode is not None
            else JsonlFeedMode.PER_COMPONENT_LATEST
        )
        raw_jsonl = collect_raw_jsonl(timeline_dir, mode=active_mode)
        plot_paths = render_default_plots(timeline_dir, formats=tuple(plot_formats))
        # ``TimelineSummary`` is frozen; rebuild with the extra fields.
        timeline_summary = dataclasses.replace(
            timeline_summary,
            raw_jsonl=raw_jsonl,
            plot_paths=plot_paths,
        )

    # Decide (Status, FailureMode).
    reasons: list[str] = []
    if partial_reason:
        reasons.append(partial_reason)
    if timeline_partial_reason:
        reasons.append(timeline_partial_reason)
    if final_num_traj is None:
        reasons.append("final MetricTable block has no num_trajectories field")

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
            memory_summary=memory_summary,
            error_excerpt=_extract_error_excerpt(
                effective_stderr_path, FailureMode.WORKER_CRASH
            ),
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
            memory_summary=memory_summary,
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
        memory_summary=memory_summary,
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
    average is taken over every parsed MetricTable block (the first
    block is not treated as warmup). When ``per_step`` is empty, or no
    block reports ``step_time``, or ``num_trajectories`` is missing /
    non-positive on the final block, the objective is ``None`` and
    ``partial_reason`` explains why.
    """
    if not per_step:
        return None, None, "no MetricTable blocks parsed"
    times = [s.step_time_seconds for s in per_step if s.step_time_seconds is not None]
    if not times:
        return None, None, "no step_time values parsed from MetricTable blocks"
    avg_step_time = sum(times) / len(times)
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


def parse_timeline(
    timeline_dir: Path,
    *,
    enable_offload: Mapping[str, bool] | None = None,
) -> TimelineSummary:
    """Aggregate every ``*.jsonl`` file under ``timeline_dir``.

    Thin adapter over :mod:`toolkits.embodied_tuner.timeline_processor` —
    the module owns event loading and every derivation from those events.
    This function packs the results into :class:`TimelineSummary` and
    supplies the headline-tag list.

    Args:
        timeline_dir: directory containing ``*.jsonl`` per-component traces.
        enable_offload: per-component offload state; when provided,
            outlier records carry a ``knob_hint`` linking the stall back
            to ``env/rollout/actor.enable_offload``.
    """
    # Defer the import so the processor module's dependencies don't
    # bloat the parser's import surface.
    from toolkits.embodied_tuner.timeline_processor import (
        compute_component_call_averages,
        compute_critical_path,
        compute_outliers,
        compute_per_component_bubble,
        compute_stall_fractions,
        compute_tag_stats,
        extract_raw_excerpts,
        load_events,
    )

    events = load_events(timeline_dir)
    if not events:
        return TimelineSummary()

    window_start = min(float(e["t0"]) for e in events)
    window_end = max(float(e["t1"]) for e in events)

    per_tag = tuple(
        TagStats(**row)
        for row in compute_tag_stats(events, headline_tags=_HEADLINE_TAGS)
    )

    return TimelineSummary(
        per_tag=per_tag,
        window_start=window_start,
        window_end=window_end,
        stall_fraction_by_component=compute_stall_fractions(events),
        critical_path=compute_critical_path(events),
        outliers=compute_outliers(events, enable_offload=enable_offload),
        per_component_bubble=compute_per_component_bubble(events),
        raw_excerpts=extract_raw_excerpts(events),
        component_call_averages=compute_component_call_averages(events),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_failure(stderr_path: Path | str | None) -> FailureMode | None:
    """Inspect ``stderr_path`` (if any) and return OOM / WORKER_CRASH /
    DIVISIBILITY_VIOLATION if matched.

    The divisibility check runs BEFORE the generic worker-crash regex so
    a routing-layer assertion isn't swallowed as a plain worker crash —
    the LLM critic needs the specific classification to know the fix is
    a placement / total_num_envs adjustment (wiki §2.6), not a memory
    or code bug.
    """
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
    if _DIVISIBILITY_REGEX.search(text):
        return FailureMode.DIVISIBILITY_VIOLATION
    if _WORKER_CRASH_REGEX.search(text):
        return FailureMode.WORKER_CRASH
    return None


def _extract_error_excerpt(
    log_path: Path | str | None,
    failure_mode: FailureMode,
    *,
    max_lines: int = _ERROR_EXCERPT_MAX_LINES,
) -> str:
    """Return a short tail-of-log slice for the critic prompt.

    For OOM / WORKER_CRASH / DIVISIBILITY_VIOLATION we anchor the slice
    on the LAST regex hit and grab the trailing ``max_lines`` lines so
    the LLM sees the actual assertion / exception message + a bit of
    surrounding traceback. For METRICS_MISSING we just grab the tail of
    the log — there is no anchor to search for, but the runner's final
    lines usually explain why (e.g. Ray init failure, Hydra error).

    Returns an empty string when the log is missing, unreadable, or
    when the failure mode isn't one that benefits from a log excerpt.
    """
    if log_path is None:
        return ""
    path = Path(log_path)
    if not path.is_file():
        return ""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""

    anchor_end: int | None = None
    if failure_mode is FailureMode.DIVISIBILITY_VIOLATION:
        matches = list(_DIVISIBILITY_REGEX.finditer(text))
        if matches:
            anchor_end = matches[-1].end()
    elif failure_mode is FailureMode.OOM:
        matches = list(_OOM_REGEX.finditer(text))
        if matches:
            anchor_end = matches[-1].end()
    elif failure_mode is FailureMode.WORKER_CRASH:
        matches = list(_WORKER_CRASH_REGEX.finditer(text))
        if matches:
            anchor_end = matches[-1].end()
    elif failure_mode is FailureMode.METRICS_MISSING:
        # No anchor — fall through to plain tail.
        pass
    else:
        return ""

    if anchor_end is not None:
        # Extend the anchor to the end of its current line so the
        # assertion message stays contiguous in the extracted excerpt
        # (a non-greedy match on ``divisible`` / ``% ... == 0`` would
        # otherwise land mid-line and split the message across
        # ``preceding``/``following``).
        newline_pos = text.find("\n", anchor_end)
        anchor_end = newline_pos if newline_pos != -1 else len(text)
        # Include a few lines BEFORE the anchor so the traceback frames
        # leading up to the assertion / OOM are visible.
        prefix_lines = 8
        preceding = text[:anchor_end].splitlines()
        following = text[anchor_end:].splitlines()
        start = max(0, len(preceding) - prefix_lines)
        selected = preceding[start:] + following[: max_lines - min(prefix_lines, len(preceding) - start)]
    else:
        selected = text.splitlines()[-max_lines:]

    excerpt = "\n".join(selected).strip()
    if not excerpt:
        return ""
    return excerpt


def _parse_nvitop_summary_log_text(text: str) -> MemorySummary | None:
    """Back-compat fallback: parse a legacy ``nvitop_summary.log`` text.

    Used only when the structured ``nvitop_summary.json`` sidecar is absent
    (trials produced before the sidecar existed). Parses the
    ``global_gpu_summary`` and ``process_summary`` sections into the same
    :class:`MemorySummary` shape the sidecar would have yielded, and —
    unlike the legacy ``_read_peak_gpu_mem`` regex — takes the true global
    peak as ``max`` across ALL per-process ``max_process_gpu_mem`` lines
    (not just the first one, which was component-order-dependent and could
    silently report the wrong component).
    """
    if not text:
        return None

    def _num(token: str) -> float | None:
        token = token.strip()
        if not token or token == "n/a":
            return None
        try:
            return float(token)
        except ValueError:
            return None

    per_gpu: list[dict] = []
    per_process: list[dict] = []
    samples: int | None = None
    span_s: float | None = None
    aggregate_bin_s: float | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("samples:"):
            try:
                samples = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("span_s:"):
            span_s = _num(line.split(":", 1)[1])
        elif line.startswith("aggregate_bin_s:"):
            aggregate_bin_s = _num(line.split(":", 1)[1])
        elif line.startswith("gpu") and ":" in line and "avg_mem=" in line:
            kv = dict(
                _kv_pair(part)
                for part in line.split(",")
                if "=" in part
            )
            try:
                index = int(line.split(":", 1)[0].replace("gpu", "").strip())
            except ValueError:
                continue
            per_gpu.append(
                {
                    "index": index,
                    "avg_mem": _num(kv.get("avg_mem", "")),
                    "max_mem": _num(kv.get("max_mem", "")),
                    "avg_gpu_util": _num(kv.get("avg_gpu_util", "")),
                    "max_gpu_util": _num(kv.get("max_gpu_util", "")),
                    "avg_mem_util": _num(kv.get("avg_mem_util", "")),
                    "max_mem_util": _num(kv.get("max_mem_util", "")),
                    # Occupancy (used/total) ratio. Absent on legacy logs → None,
                    # which the soft-pressure path tolerates.
                    "avg_mem_occ_percent": _num(kv.get("avg_mem_occ", "")),
                    "max_mem_occ_percent": _num(kv.get("max_mem_occ", "")),
                    "memory_total_gib": None,
                }
            )
        elif "/" in line and line.split(":", 1)[0].count("/") >= 2 and ":" in line:
            # process_summary line: "component/rN/pidM: avg_rss=..., ..."
            label_part, _, rest = line.partition(":")
            label = label_part.strip()
            kv = dict(_kv_pair(part) for part in rest.split(",") if "=" in part)
            component_rank_pid = label.split("/")
            if len(component_rank_pid) != 3:
                continue
            component = component_rank_pid[0]
            try:
                rank = int(component_rank_pid[1].replace("r", ""))
            except ValueError:
                rank = -1
            try:
                pid = int(component_rank_pid[2].replace("pid", ""))
            except ValueError:
                pid = -1
            gpu_idx_token = kv.get("gpu_indices", "n/a")
            if gpu_idx_token and gpu_idx_token != "n/a":
                try:
                    gpu_indices = [
                        int(x.strip())
                        for x in gpu_idx_token.strip("[]").split(",")
                        if x.strip()
                    ]
                except ValueError:
                    gpu_indices = []
            else:
                gpu_indices = []
            per_process.append(
                {
                    "label": label,
                    "component": component,
                    "rank": rank,
                    "pid": pid,
                    "avg_rss": _num(kv.get("avg_rss", "")),
                    "max_rss": _num(kv.get("max_rss", "")),
                    "avg_cpu": _num(kv.get("avg_cpu", "")),
                    "max_cpu": _num(kv.get("max_cpu", "")),
                    "avg_process_gpu_mem": _num(kv.get("avg_process_gpu_mem", "")),
                    "max_process_gpu_mem": _num(kv.get("max_process_gpu_mem", "")),
                    "avg_process_gpu_util": _num(kv.get("avg_process_gpu_util", "")),
                    "max_process_gpu_util": _num(kv.get("max_process_gpu_util", "")),
                    "gpu_indices": gpu_indices,
                }
            )

    if not per_gpu and not per_process:
        return None

    peak = _safe_max([p.get("max_process_gpu_mem") for p in per_process])
    peak_util = _safe_max([g.get("max_mem_util") for g in per_gpu])
    peak_occ = _safe_max([g.get("max_mem_occ_percent") for g in per_gpu])
    return MemorySummary(
        samples=samples,
        span_s=span_s,
        aggregate_bin_s=aggregate_bin_s,
        gpu_total_gib=None,
        peak_gpu_mem_gib=peak,
        peak_mem_util_percent=peak_util,
        peak_mem_occ_percent=peak_occ,
        per_gpu=tuple(per_gpu),
        per_process=tuple(per_process),
    )


def _kv_pair(part: str) -> tuple[str, str]:
    """Split ``avg_mem=24.231 GiB`` -> ``("avg_mem", "24.231")``."""
    key, _, value = part.partition("=")
    # Strip the unit suffix (``GiB`` / ``%``); _num tolerates bare numbers.
    value = value.strip().split()[0] if value.strip() else ""
    return key.strip(), value


def _safe_max(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    return max(nums) if nums else None


def _load_memory_summary(nvitop_dir: Path) -> MemorySummary | None:
    """Build a :class:`MemorySummary` for a trial's ``nvitop/`` dir.

    Resolution order (mirrors the timeline sidecar pattern, but with a
    text-log back-compat fallback):

    1. Read the structured ``nvitop_summary.json`` sidecar (produced by
       :func:`profiler.plot_nvitop.write_nvitop_summary`). Zero regex,
       zero drift, carries ``gpu_total_gib`` (device cap).
    2. If the sidecar is absent (legacy trial), parse the
       ``nvitop_summary.log`` text via :func:`_parse_nvitop_summary_log_text`.
    3. If neither exists but raw ``*.jsonl`` traces do, call
       :func:`nvitop_feed.render_nvitop_summary` to generate the sidecar
       on the fly, then read it.

    Returns ``None`` when no nvitop data exists at all. Best-effort: never
    raises — a missing memory summary must not change trial classification.
    """
    if not nvitop_dir.is_dir():
        return None

    plot_paths = discover_curve_plots(nvitop_dir)

    sidecar = nvitop_dir / "nvitop_summary.json"
    if sidecar.is_file():
        try:
            import json

            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if data is not None:
            return _memory_summary_from_sidecar(data, plot_paths)

    log_path = nvitop_dir / "nvitop_summary.log"
    if log_path.is_file():
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            text = ""
        summary = _parse_nvitop_summary_log_text(text)
        if summary is not None:
            return dataclasses.replace(summary, plot_paths=plot_paths)

    # Last resort: generate the sidecar from raw traces, then read it.
    generated = render_nvitop_summary(nvitop_dir)
    if generated is not None and sidecar.is_file():
        try:
            import json

            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if data is not None:
            return _memory_summary_from_sidecar(data, plot_paths)

    return None


def _memory_summary_from_sidecar(
    data: dict, plot_paths: dict
) -> MemorySummary:
    """Build a :class:`MemorySummary` from a ``nvitop_summary.json`` dict."""
    per_gpu = tuple(data.get("per_gpu", ()))
    per_process = tuple(data.get("per_process", ()))
    peak = _safe_max([p.get("max_process_gpu_mem") for p in per_process])
    peak_util = _safe_max([g.get("max_mem_util") for g in per_gpu])
    # True occupancy peak (used/total); falls back to deriving it from
    # max_mem / memory_total_gib when the sidecar predates the
    # max_mem_occ_percent field.
    peak_occ = _safe_max([g.get("max_mem_occ_percent") for g in per_gpu])
    if peak_occ is None:
        for g in per_gpu:
            mm = g.get("max_mem")
            tot = g.get("memory_total_gib")
            if mm is not None and tot:
                occ = mm / tot * 100.0
                if peak_occ is None or occ > peak_occ:
                    peak_occ = occ
    return MemorySummary(
        samples=data.get("samples"),
        span_s=data.get("span_s"),
        aggregate_bin_s=data.get("aggregate_bin_s"),
        gpu_total_gib=data.get("gpu_total_gib"),
        peak_gpu_mem_gib=peak,
        peak_mem_util_percent=peak_util,
        peak_mem_occ_percent=peak_occ,
        per_gpu=per_gpu,
        per_process=per_process,
        plot_paths=plot_paths,
    )
