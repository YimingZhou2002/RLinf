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

"""Aggregate every critic response of a tuner campaign into one file.

For each trial the scheduler persisted a ``critic/attempt-NN-response.txt``
file under the trial's ``log_dir`` (see
``Scheduler._persist_critic_transactions``). This module concatenates them
into a single ``all_responses.txt`` next to the ledger, in chronological
order, with a per-trial header:

    =================================================================
    TRIAL #00: trial-20260713-08:02:58-0559a747   |   attempt-00 / response
    =================================================================
    { ...critic JSON... }

Trial order is the sorted list of ``trial-*`` subdirectories. The
``trial-{stamp}-...`` prefix uses a zero-padded ``YYYYMMDD-HH:MM:SS``
timestamp, so a plain lexical sort is chronological. This is preferred
over ``tuner_ledger.jsonl`` order because a trial whose critic ran but
which the scheduler later discarded (validation failure, preflight
rejection, ...) is never written to the ledger — yet its response is
persisted on disk. Globbing the directories captures every such orphan
trial too, which is the point of an "all responses" aggregate.

Usable both as a library (``emit_all_responses``) and as a CLI. The
tuner CLI invokes ``emit_all_responses`` at the end of a campaign so
operators get the transcript alongside ``best_config.yaml`` /
``best_trial.json``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_OUTPUT_NAME = "all_responses.txt"

_ATTEMPT_RE = re.compile(r"attempt-(\d+)-response\.txt$")

_HEADER_BAR = "=" * 64


def _attempt_sort_key(path: Path) -> tuple[int, str]:
    m = _ATTEMPT_RE.search(path.name)
    return (int(m.group(1)) if m else 0, str(path))


def _trial_responses(trial_log_dir: Path) -> list[Path]:
    """Return this trial's ``critic/attempt-NN-response.txt`` files, sorted."""
    critic_dir = trial_log_dir / "critic"
    if not critic_dir.is_dir():
        return []
    return sorted(critic_dir.glob("attempt-*-response.txt"), key=_attempt_sort_key)


def _ordered_trial_dirs(ledger_dir: Path) -> list[str]:
    """Sorted ``trial-*`` directory names under ``ledger_dir``.

    Lexical sort is chronological because each name is prefixed with a
    zero-padded ``trial-YYYYMMDD-HH:MM:SS-...`` stamp.
    """
    return sorted(
        p.name
        for p in ledger_dir.iterdir()
        if p.is_dir() and p.name.startswith("trial-")
    )


def emit_all_responses(ledger_dir: Path, out_path: Path | None = None) -> Path | None:
    """Write ``all_responses.txt`` aggregating every critic response.

    Returns the output path, or ``None`` when no trial responses were found
    (nothing to aggregate). A per-trial IO error is swallowed with a log
    line so a single corrupt trial does not abort the whole aggregate.
    """
    ledger_dir = Path(ledger_dir)
    out_path = Path(out_path) if out_path else (ledger_dir / DEFAULT_OUTPUT_NAME)

    blocks: list[str] = []
    for trial_idx, trial_name in enumerate(_ordered_trial_dirs(ledger_dir)):
        trial_dir = ledger_dir / trial_name
        for resp in _trial_responses(trial_dir):
            m = _ATTEMPT_RE.search(resp.name)
            attempt_tag = f"attempt-{int(m.group(1)):02d}" if m else resp.stem
            try:
                body = resp.read_text(encoding="utf-8")
            except OSError as exc:
                # A single unreadable response must not kill the aggregate.
                print(
                    f"emit_all_responses: skipping unreadable {resp}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not body.endswith("\n"):
                body += "\n"
            blocks.append(
                f"{_HEADER_BAR}\n"
                f"TRIAL #{trial_idx:02d}: {trial_name}   |   {attempt_tag} / response\n"
                f"{_HEADER_BAR}\n"
                f"{body}"
            )

    if not blocks:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(blocks), encoding="utf-8")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m toolkits.embodied_tuner.utils.emit_all_responses",
        description="Aggregate every critic response of a tuner campaign.",
    )
    parser.add_argument(
        "ledger_dir",
        type=Path,
        help=(
            "Campaign ledger directory (the one containing tuner_ledger.jsonl "
            "and trial-* subdirs). all_responses.txt is written here."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional explicit output path. Defaults to "
            f"<ledger_dir>/{DEFAULT_OUTPUT_NAME}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = emit_all_responses(args.ledger_dir, args.out)
    if result is None:
        sys.stderr.write(
            f"emit_all_responses: nothing to aggregate (no critic responses "
            f"found under {args.ledger_dir})\n"
        )
        return 1
    print(f"Saved: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
