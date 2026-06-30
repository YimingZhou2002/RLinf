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

"""Bundled import-boundary AST walker (AC-11 sub-test).

The plan forbids the embodied auto-tuner from importing
``toolkits.auto_placement`` (which requires ``config.profile_data`` for
cold start, a field embodied configs do not provide). This test walks
every ``.py`` file under ``RLinf/toolkits/embodied_tuner/`` and asserts
no ``import`` / ``from ... import`` statement references
``toolkits.auto_placement`` or the bare module name ``auto_placement``.

A separate test plants a deliberate offending import in a temp file and
verifies the walker catches it (regression-safety).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FULL = "toolkits.auto_placement"
FORBIDDEN_BARE = "auto_placement"


def _iter_python_files(root: Path):
    yield from sorted(root.rglob("*.py"))


def _violations_in(path: Path) -> list[tuple[int, str]]:
    """Return ``[(lineno, message), ...]`` for every forbidden import in ``path``."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover — every shipped file parses
        return [(getattr(exc, "lineno", 0) or 0, f"syntax error: {exc}")]

    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    out.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_forbidden(module):
                names = ", ".join(a.name for a in node.names)
                out.append((node.lineno, f"from {module} import {names}"))
    return out


def _is_forbidden(module_name: str) -> bool:
    if not module_name:
        return False
    parts = module_name.split(".")
    if parts == [FORBIDDEN_BARE]:
        return True
    return module_name == FORBIDDEN_FULL or module_name.startswith(FORBIDDEN_FULL + ".")


# ---------------------------------------------------------------------------
# Positive: clean toolkit has no forbidden imports
# ---------------------------------------------------------------------------


def test_no_auto_placement_import_in_toolkit() -> None:
    offenders: dict[Path, list[tuple[int, str]]] = {}
    for path in _iter_python_files(TOOLKIT_ROOT):
        # Skip THIS test file (it mentions the forbidden name as a string
        # constant, not an import).
        if path.name == Path(__file__).name:
            continue
        violations = _violations_in(path)
        if violations:
            offenders[path] = violations
    if offenders:
        formatted = "\n".join(
            f"  {p}:{ln} -> {msg}"
            for p, vs in offenders.items()
            for ln, msg in vs
        )
        pytest.fail(
            "embodied_tuner contains forbidden imports of "
            f"{FORBIDDEN_FULL!r} / bare {FORBIDDEN_BARE!r}:\n{formatted}"
        )


def test_walker_distinguishes_bare_name_from_attribute_access() -> None:
    """``auto_placement`` as a string in a comment or docstring is OK."""
    sample = (
        '"""auto_placement is mentioned here in a docstring."""\n'
        "x = 1  # auto_placement in a comment is fine too\n"
        "name = 'auto_placement'  # so is a string literal\n"
    )
    tree = ast.parse(sample)
    # Recreate the walker's logic against this in-memory tree.
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if _is_forbidden(node.module or ""):
                violations.append(node.module)
    assert violations == []


# ---------------------------------------------------------------------------
# Negative: planted offender is caught
# ---------------------------------------------------------------------------


def test_walker_catches_planted_violation(tmp_path: Path) -> None:
    """A deliberate `from toolkits.auto_placement import DataFitter` must be flagged."""
    offender = tmp_path / "offender.py"
    offender.write_text(
        "# This file intentionally imports the forbidden module.\n"
        "from toolkits.auto_placement import DataFitter  # noqa\n"
        "_ = DataFitter\n"
    )
    violations = _violations_in(offender)
    assert violations, "walker failed to detect a planted forbidden import"
    assert violations[0][1].startswith("from toolkits.auto_placement import DataFitter")


def test_walker_catches_bare_import_form(tmp_path: Path) -> None:
    offender = tmp_path / "bare_offender.py"
    offender.write_text("import auto_placement\n")
    violations = _violations_in(offender)
    assert any("import auto_placement" in msg for _, msg in violations)


def test_walker_catches_submodule_import(tmp_path: Path) -> None:
    offender = tmp_path / "sub_offender.py"
    offender.write_text(
        "from toolkits.auto_placement.fitter import DataFitter\n"
    )
    violations = _violations_in(offender)
    assert any("toolkits.auto_placement.fitter" in msg for _, msg in violations)


def test_walker_ignores_unrelated_imports(tmp_path: Path) -> None:
    benign = tmp_path / "benign.py"
    benign.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from toolkits.embodied_tuner.schema import KnobSchema\n"
    )
    assert _violations_in(benign) == []


# ---------------------------------------------------------------------------
# Sanity: every shipped toolkit file is at least walkable
# ---------------------------------------------------------------------------


def test_every_toolkit_file_parses() -> None:
    for path in _iter_python_files(TOOLKIT_ROOT):
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
