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

"""AC-11 hermetic-suite guard: no Ray / network imports in tests.

Every test under ``toolkits/embodied_tuner/tests/**/*.py`` is
declared hermetic (no Ray, no GPU, no network). This walker enforces
the "no Ray / no network" half of the promise by rejecting any
``import`` statement whose module name matches a known heavyweight
runtime or a network client. GPU imports are policed by test-time
skips (already handled by upstream ``pytest`` fixtures) rather than
by static analysis, since some GPU libraries also expose CPU-only
paths that are legitimately used in fixtures.

The guard uses ``ast`` to walk each test file — string matches would
be too easy to defeat and would false-positive on strings/comments.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Top-level module names that must never be imported by hermetic
# tests. Sub-module imports are matched by leading-component equality
# (e.g. ``import ray.util`` matches ``ray``).
_FORBIDDEN_TOP_LEVEL_MODULES: frozenset[str] = frozenset(
    {
        "ray",
        "sglang",
        "vllm",
        "requests",
        "httpx",
        "aiohttp",
    }
)

# Specific from-imports that are forbidden regardless of module top
# level. Currently empty; extend as needed.
_FORBIDDEN_FROM_TARGETS: frozenset[tuple[str, str]] = frozenset()

_TESTS_ROOT = Path(__file__).resolve().parent


def _module_top_level(name: str | None) -> str:
    if not name:
        return ""
    return name.split(".", 1)[0]


def _iter_test_files() -> list[Path]:
    return sorted(
        p for p in _TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def test_no_forbidden_imports_in_hermetic_test_suite() -> None:
    offenders: list[tuple[str, int, str]] = []
    for path in _iter_test_files():
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            offenders.append((str(path), exc.lineno or 0, f"parse error: {exc}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = _module_top_level(alias.name)
                    if top in _FORBIDDEN_TOP_LEVEL_MODULES:
                        offenders.append(
                            (str(path), node.lineno, f"forbidden import: {alias.name}")
                        )
            elif isinstance(node, ast.ImportFrom):
                top = _module_top_level(node.module)
                if top in _FORBIDDEN_TOP_LEVEL_MODULES:
                    offenders.append(
                        (
                            str(path),
                            node.lineno,
                            f"forbidden from-import: from {node.module} ...",
                        )
                    )
                for alias in node.names:
                    if (node.module or "", alias.name) in _FORBIDDEN_FROM_TARGETS:
                        offenders.append(
                            (
                                str(path),
                                node.lineno,
                                f"forbidden from-import target: from {node.module} import {alias.name}",
                            )
                        )
    assert not offenders, (
        "AC-11 hermetic-suite violation. Offending imports:\n"
        + "\n".join(f"  {p}:{ln}: {msg}" for p, ln, msg in offenders)
    )
