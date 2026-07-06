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

"""Render a per-trial line chart of step_time / num_trajectories.

Consumes the tuner's ``tuner_ledger.jsonl`` (one JSON row per trial)
and produces a two-panel PNG:

- Top: raw ``step_time`` (left axis) and ``num_trajectories`` (right axis).
- Bottom: efficiency ratio ``step_time / num_trajectories`` (s per traj).

FAILED / TIMEOUT trials (no ``step_time`` / ``num_trajectories``) are
skipped for the line and marked with a red dotted vertical guide.

Usable both as a library (``plot_ledger``) and as a CLI. The tuner
CLI invokes ``plot_ledger`` at the end of a campaign so operators get
the chart alongside ``best_config.yaml`` / ``best_trial.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_LEDGER_NAME = "tuner_ledger.jsonl"
DEFAULT_OUTPUT_NAME = "step_time_vs_num_trajectories.png"


def plot_ledger(ledger_path: Path, out_path: Path) -> Path | None:
    """Render the chart from ``ledger_path`` to ``out_path``.

    Returns the output path on success, or ``None`` when there are no
    successful trials to plot (empty ledger, all FAILED). Missing
    ledger files raise ``FileNotFoundError``.
    """
    ledger_path = Path(ledger_path)
    out_path = Path(out_path)
    if not ledger_path.is_file():
        raise FileNotFoundError(f"ledger not found: {ledger_path}")

    trials = []
    with ledger_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))

    if not trials:
        return None

    idx = list(range(len(trials)))
    step_time = [t.get("step_time") for t in trials]
    num_traj = [t.get("num_trajectories") for t in trials]
    status = [t.get("status") for t in trials]
    ratio = [
        (s / n) if (s is not None and n) else None
        for s, n in zip(step_time, num_traj)
    ]

    ok_idx = [
        i for i, (st, s, n) in enumerate(zip(status, step_time, num_traj))
        if st == "OK" and s is not None and n
    ]
    fail_idx = [i for i in idx if i not in ok_idx]
    if not ok_idx:
        return None

    # Import matplotlib lazily so importing this module (e.g. from the
    # tuner CLI) does not pull in the plotting stack unless we actually
    # render a chart. Use the non-interactive backend explicitly — the
    # tuner runs headless.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    color_st = "#1f77b4"
    color_nt = "#ff7f0e"
    color_r = "#2ca02c"

    ax1.plot(
        ok_idx,
        [step_time[i] for i in ok_idx],
        "o-", color=color_st, label="step_time (s)",
    )
    ax1.set_ylabel("step_time (s)", color=color_st)
    ax1.tick_params(axis="y", labelcolor=color_st)
    ax1.grid(alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(
        ok_idx,
        [num_traj[i] for i in ok_idx],
        "s--", color=color_nt, label="num_trajectories",
    )
    ax1b.set_ylabel("num_trajectories", color=color_nt)
    ax1b.tick_params(axis="y", labelcolor=color_nt)

    for i in fail_idx:
        ax1.axvline(i, color="red", alpha=0.25, linestyle=":")
        ax1.text(
            i, ax1.get_ylim()[1] * 0.02, "FAIL", color="red",
            ha="center", fontsize=8, rotation=90, va="bottom",
        )

    ax1.set_title("Per-trial step_time and num_trajectories")

    ax2.plot(
        ok_idx,
        [ratio[i] for i in ok_idx],
        "d-", color=color_r, label="step_time / num_trajectories (s/traj)",
    )
    ax2.set_ylabel("step_time / num_trajectories (s per traj)", color=color_r)
    ax2.tick_params(axis="y", labelcolor=color_r)
    ax2.set_xlabel("Trial index")
    ax2.grid(alpha=0.3)

    for i in fail_idx:
        ax2.axvline(i, color="red", alpha=0.25, linestyle=":")

    ax2.set_title("Per-trial efficiency: seconds per trajectory")
    ax2.set_xticks(idx)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ledger_dir(ledger_dir: Path) -> Path | None:
    """Convenience wrapper: locate ``tuner_ledger.jsonl`` under ``ledger_dir``.

    Writes ``step_time_vs_num_trajectories.png`` next to the ledger.
    """
    ledger_dir = Path(ledger_dir)
    return plot_ledger(
        ledger_dir / DEFAULT_LEDGER_NAME,
        ledger_dir / DEFAULT_OUTPUT_NAME,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m toolkits.embodied_tuner.utils.plot_step_time_vs_trajectories",
        description="Render step_time / num_trajectories chart from a tuner ledger.",
    )
    parser.add_argument(
        "ledger_dir",
        type=Path,
        help=(
            "Directory containing tuner_ledger.jsonl (typically the campaign "
            "ledger dir). The PNG is written next to the ledger."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional explicit output PNG path. Defaults to "
            f"<ledger_dir>/{DEFAULT_OUTPUT_NAME}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ledger_path = args.ledger_dir / DEFAULT_LEDGER_NAME
    out_path = args.out or (args.ledger_dir / DEFAULT_OUTPUT_NAME)
    result = plot_ledger(ledger_path, out_path)
    if result is None:
        sys.stderr.write(
            f"plot_step_time_vs_trajectories: nothing to plot (empty ledger "
            f"or no OK trials): {ledger_path}\n"
        )
        return 1
    print(f"Saved: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
