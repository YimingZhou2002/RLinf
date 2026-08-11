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

"""Dynamic sweep-size generation.

Sweep points are never hardcoded as absolute values. They are derived from the
batch size discovered in the *default* run, scaled by relative multipliers, so
the same code produces sensible sweeps for any model / env.
"""

from __future__ import annotations


def parse_multipliers(spec: str | None, default: str = "0.25,0.5,1,2,4") -> list[float]:
    """Parse a comma-separated multiplier spec (e.g. ``"0.5,1,2,4"``)."""
    text = spec if spec else default
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [1.0]


def make_sizes(
    default: int,
    multipliers: list[float],
    *,
    divisor_of: int | None = None,
    cap: int | None = None,
    minimum: int = 1,
) -> list[int]:
    """Generate sorted, de-duplicated sweep sizes from ``default * multiplier``.

    Args:
        default: the batch size discovered from the default run.
        multipliers: relative scale factors applied to ``default``.
        divisor_of: if set, keep only sizes that evenly divide this value and do
            not exceed it (used for the actor, whose global batch is pinned so a
            micro size must divide the per-rank global batch).
        cap: if set, drop sizes larger than this.
        minimum: floor for each size (default 1).
    """
    sizes = {max(minimum, int(round(default * m))) for m in multipliers}

    if divisor_of is not None:
        sizes = {s for s in sizes if s <= divisor_of and divisor_of % s == 0}
        # Always include the pinned global itself (single micro-batch) and the
        # discovered default if it is valid, so the baseline is present.
        if divisor_of >= minimum:
            sizes.add(divisor_of)
        if default <= divisor_of and divisor_of % default == 0:
            sizes.add(default)
    if cap is not None:
        sizes = {s for s in sizes if s <= cap}

    return sorted(s for s in sizes if s >= minimum)
