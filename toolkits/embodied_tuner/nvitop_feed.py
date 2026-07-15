# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""nvitop resource-trace selectors for the embodied auto-tuner.

Mirrors :mod:`toolkits.embodied_tuner.timeline_feed` for GPU-memory
records. Two responsibilities:

1. :func:`collect_raw_nvitop_jsonl` — pluggable selector that reads a
   subset of ``<log_dir>/nvitop/<component>_rank<N>_pid<PID>.jsonl``
   files and returns their verbatim text, keyed by stem. This is what
   gets fed into the critic's ``memory_verbose_block`` so the LLM can
   see actual per-sample GPU memory streams (not just the aggregated
   :class:`~toolkits.embodied_tuner.parser.MemorySummary`).

2. :func:`render_nvitop_summary` — best-effort generation of the
   ``nvitop_summary.log`` + ``nvitop_summary.json`` sidecar from raw
   JSONL when neither is already on disk (the fallback path in
   :func:`toolkits.embodied_tuner.parser._load_memory_summary`).

The selector is an :class:`Enum` (not a bool) mirroring
:class:`~toolkits.embodied_tuner.timeline_feed.JsonlFeedMode`, but it
defaults to :attr:`NvitopFeedMode.NONE` — unlike the timeline (which
defaults to ``PER_COMPONENT_LATEST``) — because a single raw nvitop
trace is ~660 KB (vs. a few KB for timeline traces), so raw injection
is opt-in to keep the critic prompt bounded by default.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from enum import Enum
from pathlib import Path

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL selector
# ---------------------------------------------------------------------------


class NvitopFeedMode(str, Enum):
    """Which nvitop JSONL files to feed into the critic prompt.

    ``NONE`` (default)
        Skip raw nvitop JSONL injection entirely; the critic sees only
        the aggregated :class:`~toolkits.embodied_tuner.parser.MemorySummary`
        (per-GPU / per-process avg+max memory & util, peak, device cap,
        soft-pressure flag). This is the bounded-by-default behaviour.

    ``PER_COMPONENT_LATEST``
        For each component, pick a single representative rank: prefer the
        rank whose last sample has the largest ``ts`` (i.e. the straggler
        that ran longest — most informative for memory-peak reasoning);
        tie-break on the lowest rank. Keeps prompt size bounded regardless
        of world size.

    ``ALL``
        Concatenate every ``*.jsonl`` file under ``nvitop/``. Highest
        fidelity, but at 8 GPU it's megabytes of text — only viable for
        long-context critics. Included so we can flip a knob later.
    """

    NONE = "none"
    PER_COMPONENT_LATEST = "per_component_latest"
    ALL = "all"


# Selector callable takes the nvitop dir and returns the list of paths
# to read. Tests inject fakes here.
NvitopSelectorFn = Callable[[Path], list[Path]]


def _split_stem(stem: str) -> tuple[str | None, int, int]:
    """Split ``actor_rank0_pid329570`` -> ``("actor", 0, 329570)``.

    Returns ``(None, -1, -1)`` on mismatch so non-conforming files are
    skipped by the selectors.
    """
    rank_marker = "_rank"
    pid_marker = "_pid"
    ridx = stem.rfind(rank_marker)
    if ridx < 0:
        return None, -1, -1
    rest = stem[ridx + len(rank_marker):]
    pidx = rest.rfind(pid_marker)
    if pidx < 0:
        return None, -1, -1
    try:
        rank = int(rest[:pidx])
        pid = int(rest[pidx + len(pid_marker):])
    except ValueError:
        return None, -1, -1
    return stem[:ridx], rank, pid


def _tail_ts(path: Path) -> float | None:
    """Return ``ts`` from the last non-empty line of ``path``.

    Reads the tail of the file (up to 4 KB) to avoid loading whole
    multi-hundred-KB traces. Returns ``None`` on any parse / IO error.
    """
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
    ts = record.get("ts")
    if not isinstance(ts, (int, float)):
        return None
    return float(ts)


def _select_per_component_latest(nvitop_dir: Path) -> list[Path]:
    """Pick, per component, the rank whose last sample has the largest ts."""
    per_component: dict[str, tuple[float, int, Path]] = {}
    for path in sorted(nvitop_dir.glob("*.jsonl")):
        component, rank, _pid = _split_stem(path.stem)
        if component is None:
            continue
        ts_end = _tail_ts(path)
        # Largest ts wins; tie-break on lowest rank.
        key = (ts_end if ts_end is not None else float("-inf"), -rank, path)
        existing = per_component.get(component)
        if existing is None or key > existing:
            per_component[component] = key
    return [entry[2] for entry in per_component.values()]


def _select_all(nvitop_dir: Path) -> list[Path]:
    return sorted(nvitop_dir.glob("*.jsonl"))


_MODE_TO_SELECTOR: dict[NvitopFeedMode, NvitopSelectorFn] = {
    NvitopFeedMode.PER_COMPONENT_LATEST: _select_per_component_latest,
    NvitopFeedMode.ALL: _select_all,
    NvitopFeedMode.NONE: lambda _dir: [],
}


def collect_raw_nvitop_jsonl(
    nvitop_dir: Path | str,
    *,
    mode: NvitopFeedMode = NvitopFeedMode.NONE,
    max_bytes_per_file: int | None = None,
    selector: NvitopSelectorFn | None = None,
) -> dict[str, str]:
    """Read the nvitop JSONL text to feed into the critic prompt.

    Args:
        nvitop_dir: Directory containing
            ``<component>_rank<N>_pid<PID>.jsonl`` files. When absent,
            returns ``{}``.
        mode: Which subset of files to include. See :class:`NvitopFeedMode`.
            Defaults to :attr:`NvitopFeedMode.NONE` — no raw injection.
        max_bytes_per_file: If set, truncate each file's text to this
            many bytes and append a truncation marker. ``None`` keeps
            full contents.
        selector: Advanced escape hatch — a callable returning the list
            of paths to read. When provided, ``mode`` is ignored.

    Returns:
        A dict mapping each file's ``stem`` (e.g.
        ``"actor_rank0_pid329570"``) to the file's text. Empty when no
        files are selected or the directory is missing. Insertion-
        ordered (dict is ordered in Python 3.7+) for stable prompt
        rendering.
    """
    nvitop_dir = Path(nvitop_dir)
    if not nvitop_dir.is_dir():
        return {}
    fn = selector if selector is not None else _MODE_TO_SELECTOR[mode]
    out: dict[str, str] = {}
    for path in fn(nvitop_dir):
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            _log.warning("failed to read nvitop file %s: %s", path, exc)
            continue
        if max_bytes_per_file is not None and len(text) > max_bytes_per_file:
            original = len(text)
            text = text[:max_bytes_per_file] + (
                f"\n<TRUNCATED — {max_bytes_per_file} of {original} bytes shown>\n"
            )
        out[path.stem] = text
    return out


# ---------------------------------------------------------------------------
# Sidecar generation (fallback when plot_nvitop did not run)
# ---------------------------------------------------------------------------


def _newest_raw_mtime(nvitop_dir: Path) -> float | None:
    """Newest mtime among ``*.jsonl`` under ``nvitop_dir``; ``None`` if none."""
    newest: float | None = None
    for path in nvitop_dir.glob("*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def render_nvitop_summary(
    nvitop_dir: Path | str,
    *,
    aggregate_bin_s: float = 1.0,
) -> Path | None:
    """Ensure ``nvitop_summary.{log,json}`` exist for ``nvitop_dir``.

    Idempotent: if ``nvitop_summary.json`` already exists AND is at least
    as new as the newest raw ``*.jsonl``, return immediately without
    reloading the (potentially multi-MB) traces. Otherwise load the raw
    samples via :func:`plot_nvitop._load_samples` and call
    :func:`plot_nvitop.write_nvitop_summary` to (re)write both the
    ``.log`` and the ``.json`` sidecar.

    This is the fallback path in
    :func:`toolkits.embodied_tuner.parser._load_memory_summary` for trials
    whose training run never invoked ``plot_nvitop``. It is best-effort:
    any failure is logged and ``None`` returned so the trial loop never
    aborts over a missing memory summary.
    """
    nvitop_dir = Path(nvitop_dir)
    if not nvitop_dir.is_dir():
        return None
    sidecar = nvitop_dir / "nvitop_summary.json"
    newest_raw = _newest_raw_mtime(nvitop_dir)
    if sidecar.is_file() and newest_raw is not None:
        try:
            if sidecar.stat().st_mtime >= newest_raw:
                return sidecar
        except OSError:
            pass

    # Import lazily so importing nvitop_feed never pulls plotly / matplotlib
    # into the tuner's hot path.
    try:
        from toolkits.embodied_tuner.profiler import plot_nvitop
    except Exception as exc:  # noqa: BLE001
        _log.warning("could not import plot_nvitop: %s", exc)
        return None

    try:
        samples = plot_nvitop._load_samples(str(nvitop_dir))
        if not samples:
            return None
        plot_nvitop.write_nvitop_summary(
            str(nvitop_dir),
            samples,
            aggregate_bin_s=aggregate_bin_s,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("render_nvitop_summary failed on %s: %s", nvitop_dir, exc)
        return None
    return sidecar if sidecar.is_file() else None


def discover_curve_plots(nvitop_dir: Path | str) -> dict[str, Path]:
    """Point at any pre-rendered nvitop curve plots sitting on disk.

    ``plot_nvitop`` writes ``nvitop_resources.<png|html>`` (or
    ``nvitop_curves.png``) into ``<log_dir>`` (the parent of the
    ``nvitop/`` dir). We do NOT render curves here (avoids pulling
    plotly/matplotlib into the parser); we just surface paths that the
    training run already produced so the critic prompt can point a human
    debugger at them.
    """
    nvitop_dir = Path(nvitop_dir)
    parent = nvitop_dir.parent
    candidates = {
        "png": [parent / "nvitop_curves.png", parent / "nvitop_resources.png",
                nvitop_dir / "nvitop_resources.png"],
        "html": [parent / "nvitop_resources.html", nvitop_dir / "nvitop_resources.html"],
    }
    out: dict[str, Path] = {}
    for fmt, paths in candidates.items():
        for path in paths:
            if path.is_file():
                out[fmt] = path
                break
    return out
