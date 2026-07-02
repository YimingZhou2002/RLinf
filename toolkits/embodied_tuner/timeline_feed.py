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

"""Timeline post-processing helpers for the embodied auto-tuner.

Two responsibilities:

1. :func:`render_timeline_plot` — best-effort invocation of
   ``profiler/plot_timeline.py`` after each trial. Produces
   ``<log_dir>/timeline.png`` (and optionally ``timeline.html``) alongside
   the trial dir so a human debugger can eyeball the Gantt chart.

2. :func:`collect_raw_jsonl` — pluggable selector that reads one JSONL
   trace file per component from ``<log_dir>/timeline/`` and returns the
   raw text, keyed by the file's stem. This is what gets fed into the
   critic prompt so the LLM sees actual event streams (not just the
   aggregated :class:`TimelineSummary`).

The selector is intentionally an :class:`Enum` (not a bool) so we can
extend it later — e.g. ``ALL`` for full-fidelity dumps once critic-side
context budgets grow, or a callable escape hatch.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL selector
# ---------------------------------------------------------------------------


class JsonlFeedMode(str, Enum):
    """Which timeline JSONL files to feed into the critic prompt.

    ``PER_COMPONENT_LATEST`` (default)
        For each component, pick a single representative rank: prefer the
        rank whose last event ended latest (i.e. the straggler — most
        informative for stall / critical-path reasoning); tie-break on
        rank0. Keeps prompt size bounded regardless of world size.

    ``PER_COMPONENT_RANK0``
        Always pick ``<component>_rank0.jsonl``. Cheaper to reason about
        (canonical), but may miss the slowest rank's story.

    ``ALL``
        Concatenate every ``*.jsonl`` file under ``timeline/``. Highest
        fidelity, but at 8 GPU it's ~600K tokens — only viable for
        long-context critics. Included so we can flip a knob later.

    ``NONE``
        Skip raw JSONL injection entirely; the critic sees only the
        aggregated :class:`TimelineSummary` (pre-existing behavior).
    """

    PER_COMPONENT_LATEST = "per_component_latest"
    PER_COMPONENT_RANK0 = "per_component_rank0"
    ALL = "all"
    NONE = "none"


# Selector callable takes the timeline dir and returns the list of paths
# to read. Tests inject fakes here.
SelectorFn = Callable[[Path], list[Path]]


def _select_per_component_latest(timeline_dir: Path) -> list[Path]:
    """Pick, per component, the rank whose events finish latest.

    Reads only the last non-empty line of each ``*.jsonl`` and parses
    ``t1`` to avoid loading the whole file. Files that fail to yield a
    ``t1`` fall back to rank0 semantics (earliest rank wins).
    """
    per_component: dict[str, tuple[float, int, Path]] = {}
    for path in sorted(timeline_dir.glob("*.jsonl")):
        component, rank = _split_component_rank(path.stem)
        if component is None:
            continue
        t_end = _tail_t1(path)
        key = (t_end if t_end is not None else float("-inf"), -rank, path)
        existing = per_component.get(component)
        if existing is None or key > existing:
            per_component[component] = key
    return [entry[2] for entry in per_component.values()]


def _select_per_component_rank0(timeline_dir: Path) -> list[Path]:
    """Pick ``<component>_rank0.jsonl`` (or the lowest-rank file present)."""
    per_component: dict[str, tuple[int, Path]] = {}
    for path in sorted(timeline_dir.glob("*.jsonl")):
        component, rank = _split_component_rank(path.stem)
        if component is None:
            continue
        existing = per_component.get(component)
        if existing is None or rank < existing[0]:
            per_component[component] = (rank, path)
    return [entry[1] for entry in per_component.values()]


def _select_all(timeline_dir: Path) -> list[Path]:
    return sorted(timeline_dir.glob("*.jsonl"))


_MODE_TO_SELECTOR: dict[JsonlFeedMode, SelectorFn] = {
    JsonlFeedMode.PER_COMPONENT_LATEST: _select_per_component_latest,
    JsonlFeedMode.PER_COMPONENT_RANK0: _select_per_component_rank0,
    JsonlFeedMode.ALL: _select_all,
    JsonlFeedMode.NONE: lambda _dir: [],
}


def _split_component_rank(stem: str) -> tuple[str | None, int]:
    """Split ``env_rank3`` -> ``("env", 3)``. Returns ``(None, -1)`` on mismatch."""
    marker = "_rank"
    idx = stem.rfind(marker)
    if idx < 0:
        return None, -1
    try:
        rank = int(stem[idx + len(marker):])
    except ValueError:
        return None, -1
    return stem[:idx], rank


def _tail_t1(path: Path) -> float | None:
    """Return ``t1`` from the last non-empty line of ``path``.

    Reads the tail of the file (up to 4 KB) to avoid loading whole
    traces. Returns ``None`` on any parse / IO error.
    """
    import json

    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line for line in tail.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        record = json.loads(lines[-1])
    except (ValueError, TypeError):
        return None
    t1 = record.get("t1")
    if not isinstance(t1, (int, float)):
        return None
    return float(t1)


def collect_raw_jsonl(
    timeline_dir: Path | str,
    *,
    mode: JsonlFeedMode = JsonlFeedMode.PER_COMPONENT_LATEST,
    max_bytes_per_file: int | None = None,
    selector: SelectorFn | None = None,
) -> dict[str, str]:
    """Read the JSONL text that should be fed into the critic prompt.

    Args:
        timeline_dir: Directory containing ``<component>_rank<N>.jsonl``
            files. When absent, returns ``{}``.
        mode: Which subset of files to include. See :class:`JsonlFeedMode`.
        max_bytes_per_file: If set, truncate each file's text to this many
            bytes and append ``\\n<TRUNCATED — N of M bytes shown>``.
            ``None`` (default) keeps the full contents.
        selector: Advanced escape hatch — a callable that returns the
            list of paths to read. When provided, ``mode`` is ignored.

    Returns:
        A dict mapping ``<path.stem>`` (e.g. ``"env_rank3"``) to the
        file's text. Empty when no files are selected or the directory is
        missing. Preserves insertion order (dict is ordered in Python 3.7+),
        which callers can rely on for stable prompt rendering.
    """
    timeline_dir = Path(timeline_dir)
    if not timeline_dir.is_dir():
        return {}
    fn = selector if selector is not None else _MODE_TO_SELECTOR[mode]
    out: dict[str, str] = {}
    for path in fn(timeline_dir):
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            _log.warning("failed to read timeline file %s: %s", path, exc)
            continue
        if max_bytes_per_file is not None and len(text) > max_bytes_per_file:
            original = len(text)
            text = text[:max_bytes_per_file] + (
                f"\n<TRUNCATED — {max_bytes_per_file} of {original} bytes shown>\n"
            )
        out[path.stem] = text
    return out


# ---------------------------------------------------------------------------
# Gantt plot generation
# ---------------------------------------------------------------------------


_PLOT_SCRIPT = Path(__file__).resolve().parent / "profiler" / "plot_timeline.py"


def render_timeline_plot(
    timeline_dir: Path | str,
    *,
    output_path: Path | str | None = None,
    fmt: str = "png",
    timeout_seconds: float = 120.0,
    extra_args: Iterable[str] = (),
) -> Path | None:
    """Invoke ``profiler/plot_timeline.py`` on ``timeline_dir``.

    Best-effort: any failure (missing script, subprocess crash, timeout)
    is logged and ``None`` returned. The trial loop should never abort
    because a plot didn't render.

    Args:
        timeline_dir: The directory containing ``*_rank*.jsonl`` traces.
        output_path: Where to write the plot. Defaults to
            ``<timeline_dir.parent>/timeline.<fmt>`` (i.e. sitting next
            to the trial's ``timeline/`` folder in ``log_dir``).
        fmt: ``"png"`` or ``"html"``. PNG is the default because it is
            what the critic can consume as an image content block; HTML
            is human-only.
        timeout_seconds: Subprocess timeout. Real trials on ~5K events
            plot in <2 s; 120 s is a large safety margin.
        extra_args: Passthrough CLI flags to ``plot_timeline.py``
            (e.g. ``("--lane-mode", "rank")``).

    Returns:
        The absolute path to the written file on success, ``None`` on
        failure. Callers can safely persist the returned path into the
        ledger entry / critic prompt.
    """
    timeline_dir = Path(timeline_dir)
    if not timeline_dir.is_dir():
        return None
    if not _PLOT_SCRIPT.is_file():
        _log.warning("plot_timeline.py not found at %s", _PLOT_SCRIPT)
        return None

    if output_path is None:
        output_path = timeline_dir.parent / f"timeline.{fmt}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_PLOT_SCRIPT),
        str(timeline_dir),
        "--format", fmt,
        "-o", str(output_path),
        *extra_args,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _log.warning("plot_timeline.py timed out on %s", timeline_dir)
        return None
    except OSError as exc:
        _log.warning("plot_timeline.py could not start: %s", exc)
        return None

    if completed.returncode != 0:
        _log.warning(
            "plot_timeline.py exited %d on %s: stderr=%s",
            completed.returncode,
            timeline_dir,
            completed.stderr[-400:] if completed.stderr else "",
        )
        return None
    if not output_path.is_file():
        _log.warning(
            "plot_timeline.py returned success but %s missing", output_path
        )
        return None
    return output_path


def render_default_plots(
    timeline_dir: Path | str,
    *,
    formats: Iterable[str] = ("png",),
) -> dict[str, Path]:
    """Render one plot per format; return ``{fmt: path}`` on success.

    Convenience wrapper for the common "always render PNG, also HTML for
    humans" case: pass ``formats=("png", "html")``.
    """
    out: dict[str, Path] = {}
    for fmt in formats:
        path = render_timeline_plot(timeline_dir, fmt=fmt)
        if path is not None:
            out[fmt] = path
    return out
