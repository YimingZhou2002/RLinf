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

"""AC-10 negative-space guard: no frontier-selection policy in this loop.

Alt-5 (algorithmic frontier-selection over the DAG — Best-First,
UCB1, Beam, Plateau-Backtrack policies) is explicitly recorded as
FUT-1 in the plan (see the README's "Known limitations" section) and
MUST NOT be implemented in the current loop. This test walks every
Python source file under ``toolkits/embodied_tuner/**/*.py`` at import
time and enforces two invariants:

1. No class, function, method, protocol, factory, or import symbol
   is named one of the reserved frontier-policy identifiers.
2. No source file's name contains the substring ``frontier``
   (case-insensitive).

Markdown / RST / wiki / README references are allowlisted (they are
not Python source and merely document the future work).

This test file is itself allowlisted from rule (1) because it
mentions the reserved names as string literals; the AST walker skips
this file explicitly.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Identifiers reserved for FUT-1 (Alt-5 frontier-selection). If any of
# these ever appear as a class name, function name, or imported symbol
# under ``toolkits/embodied_tuner/**/*.py`` the AC-10 handoff-AC is
# violated and this test fails immediately.
_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "FrontierPolicy",
        "FrontierScheduler",
        "BestFirstPolicy",
        "UCB1Policy",
        "BeamPolicy",
        "PlateauBacktrackPolicy",
    }
)

# Files whose contents legitimately mention the reserved names as
# string literals or documentation (this test itself, plus any doc-
# adjacent Python module — currently just this test). Enforced as
# ``.name`` comparison against the file path.
_ALLOWLISTED_FILES: frozenset[str] = frozenset(
    {
        "test_no_frontier_policy.py",
    }
)

_TUNER_ROOT = Path(__file__).resolve().parents[1]  # toolkits/embodied_tuner/


def _iter_tuner_py_files() -> list[Path]:
    return sorted(
        p
        for p in _TUNER_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_no_frontier_policy_class_or_function_defined() -> None:
    """Walk every AST and reject reserved class/function names."""
    offenders: list[tuple[str, int, str]] = []
    for path in _iter_tuner_py_files():
        if path.name in _ALLOWLISTED_FILES:
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            offenders.append((str(path), exc.lineno or 0, f"parse error: {exc}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in _RESERVED_NAMES:
                    offenders.append(
                        (str(path), node.lineno, f"reserved definition: {node.name}")
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _RESERVED_NAMES:
                        offenders.append(
                            (str(path), node.lineno, f"reserved import: {alias.name}")
                        )
    assert not offenders, (
        "AC-10 violation: found frontier-policy identifiers in current-loop"
        " source (Alt-5 must remain FUT-1). Offending sites:\n"
        + "\n".join(f"  {p}:{ln}: {msg}" for p, ln, msg in offenders)
    )


def test_no_python_source_filename_contains_frontier() -> None:
    """Reject any .py file whose name suggests it hosts frontier-selection code."""
    offenders = [
        p
        for p in _iter_tuner_py_files()
        if p.name not in _ALLOWLISTED_FILES and "frontier" in p.name.lower()
    ]
    assert not offenders, (
        "AC-10 violation: found Python source file(s) named with 'frontier':\n"
        + "\n".join(f"  {p}" for p in offenders)
    )
