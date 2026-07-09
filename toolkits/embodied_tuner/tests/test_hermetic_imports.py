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

"""AC-11 hermetic-suite guard: no Ray / network imports or socket calls.

Every test under ``toolkits/embodied_tuner/tests/**/*.py`` is
declared hermetic (no Ray, no GPU, no network). This walker enforces
the "no Ray / no network" half of the promise by rejecting:

- ``import`` statements whose module name matches a known heavyweight
  runtime or a network client (top-level modules like ``ray``,
  ``requests``);
- ``import`` statements naming a forbidden dotted stdlib path
  (``urllib.request``, ``urllib.error``, ``urllib.parse`` remaining
  legitimate for URL manipulation);
- socket-connect call patterns — direct ``socket.socket(...).connect(
  ...)`` chains AND assigned forms like ``s = socket.socket(); s
  .connect(...)``.

GPU imports are policed by test-time skips (upstream ``pytest``
fixtures) rather than by static analysis, since some GPU libraries
also expose CPU-only paths that are legitimately used in fixtures.

The walker is factored into :func:`walk_source_for_violations` so it
can be exercised both on the real filesystem AND on synthetic
snippet strings in self-tests — the self-tests prove the detector
catches each forbidden pattern without ever adding a real forbidden
import to any test module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


# Top-level module names that must never be imported by hermetic
# tests. Sub-module imports match by leading-component equality
# (``import ray.util`` matches ``ray``).
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

# Forbidden dotted stdlib paths. Matched with dotted-prefix semantics:
# ``import urllib.request`` and ``from urllib.request import Request``
# both hit ``urllib.request``. ``from urllib import parse`` does NOT
# hit (``urllib.parse`` isn't listed).
_FORBIDDEN_DOTTED_MODULES: frozenset[str] = frozenset(
    {
        "urllib.request",
        "urllib.error",
    }
)

# Specific from-imports forbidden regardless of module top level.
# Example: ``from urllib import request`` — the module is ``urllib``
# (not forbidden as top-level) and the name is ``request`` (forbidden
# via this table). Keyed on (module, target_name).
_FORBIDDEN_FROM_TARGETS: frozenset[tuple[str, str]] = frozenset(
    {
        ("urllib", "request"),
        ("urllib", "error"),
    }
)

_TESTS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class _Violation:
    path: str
    lineno: int
    message: str


def _module_top_level(name: str | None) -> str:
    if not name:
        return ""
    return name.split(".", 1)[0]


def _matches_dotted_path(name: str | None, forbidden: frozenset[str]) -> bool:
    """Return True when ``name`` equals or is a sub-path of any forbidden dotted module."""
    if not name:
        return False
    return any(name == f or name.startswith(f + ".") for f in forbidden)


class _SocketConnectDetector(ast.NodeVisitor):
    """Track ``socket`` module aliases + flag every ``.connect(...)`` call chain.

    Recognises:
        import socket
        import socket as sk
        from socket import socket
        from socket import socket as make_sock

    Then flags any ``Call`` whose function attribute is ``connect`` and
    whose receiver chain traces back to one of those imports:

        socket.socket(...).connect(...)
        s = socket.socket(); s.connect(...)
        sock = socket.socket(...); sock.connect(...)
        sock_factory().connect(...)          # NOT flagged (unknown receiver)
    """

    def __init__(self) -> None:
        # Names bound to the ``socket`` module or the ``socket.socket`` class.
        self._socket_module_aliases: set[str] = set()
        self._socket_class_aliases: set[str] = set()
        # Local variables bound to a ``socket.socket(...)`` return value.
        self._socket_instances: set[str] = set()
        self.violations: list[tuple[int, str]] = []

    # -- import tracking ---------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "socket":
                self._socket_module_aliases.add(alias.asname or "socket")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "socket":
            for alias in node.names:
                if alias.name == "socket":
                    self._socket_class_aliases.add(alias.asname or "socket")
        self.generic_visit(node)

    # -- assignment tracking ----------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        # ``s = socket.socket(...)`` or ``s = SocketAlias(...)``.
        if isinstance(node.value, ast.Call) and self._call_yields_socket(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    self._socket_instances.add(tgt.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            node.value is not None
            and isinstance(node.value, ast.Call)
            and self._call_yields_socket(node.value)
            and isinstance(node.target, ast.Name)
        ):
            self._socket_instances.add(node.target.id)
        self.generic_visit(node)

    # -- connect() detection ----------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        # Look for a ``.connect(...)`` where the receiver is either an
        # inline socket-creation Call OR a Name previously bound to
        # such a Call.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
            receiver = node.func.value
            if self._receiver_is_socket(receiver):
                self.violations.append(
                    (node.lineno, "forbidden socket.socket(...).connect(...) call")
                )
        self.generic_visit(node)

    # -- helpers ----------------------------------------------------------

    def _call_yields_socket(self, call: ast.Call) -> bool:
        """Return True if ``call`` looks like a ``socket.socket(...)`` invocation."""
        func = call.func
        # Case: socket.socket(...) (module dotted access)
        if isinstance(func, ast.Attribute) and func.attr == "socket":
            if isinstance(func.value, ast.Name) and func.value.id in self._socket_module_aliases:
                return True
        # Case: bare ``socket(...)`` from ``from socket import socket``
        if isinstance(func, ast.Name) and func.id in self._socket_class_aliases:
            return True
        return False

    def _receiver_is_socket(self, receiver: ast.AST) -> bool:
        # Inline: socket.socket(...).connect(...) or SocketAlias(...).connect(...)
        if isinstance(receiver, ast.Call) and self._call_yields_socket(receiver):
            return True
        # Bound variable: previously assigned from socket.socket(...).
        if isinstance(receiver, ast.Name) and receiver.id in self._socket_instances:
            return True
        return False


def walk_source_for_violations(source: str, filename: str) -> list[_Violation]:
    """Parse ``source`` and return every AC-11 violation it contains.

    Used by both the filesystem walker below and the snippet-based
    self-tests. When ``source`` fails to parse the sole violation
    returned is a synthetic ``parse error`` entry.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [_Violation(filename, exc.lineno or 0, f"parse error: {exc}")]

    offenders: list[_Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_top_level(alias.name) in _FORBIDDEN_TOP_LEVEL_MODULES:
                    offenders.append(
                        _Violation(filename, node.lineno, f"forbidden import: {alias.name}")
                    )
                if _matches_dotted_path(alias.name, _FORBIDDEN_DOTTED_MODULES):
                    offenders.append(
                        _Violation(
                            filename,
                            node.lineno,
                            f"forbidden dotted import: {alias.name}",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if _module_top_level(node.module) in _FORBIDDEN_TOP_LEVEL_MODULES:
                offenders.append(
                    _Violation(
                        filename,
                        node.lineno,
                        f"forbidden from-import: from {node.module} ...",
                    )
                )
            if _matches_dotted_path(node.module, _FORBIDDEN_DOTTED_MODULES):
                offenders.append(
                    _Violation(
                        filename,
                        node.lineno,
                        f"forbidden dotted from-import: from {node.module} ...",
                    )
                )
            for alias in node.names:
                if (node.module or "", alias.name) in _FORBIDDEN_FROM_TARGETS:
                    offenders.append(
                        _Violation(
                            filename,
                            node.lineno,
                            f"forbidden from-import target: from {node.module} import {alias.name}",
                        )
                    )

    # Socket-connect AST walk.
    detector = _SocketConnectDetector()
    detector.visit(tree)
    for lineno, message in detector.violations:
        offenders.append(_Violation(filename, lineno, message))

    return offenders


def _iter_test_files() -> list[Path]:
    return sorted(
        p for p in _TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def test_no_forbidden_imports_in_hermetic_test_suite() -> None:
    offenders: list[_Violation] = []
    for path in _iter_test_files():
        source = path.read_text(encoding="utf-8")
        offenders.extend(walk_source_for_violations(source, str(path)))
    assert not offenders, (
        "AC-11 hermetic-suite violation. Offending imports/calls:\n"
        + "\n".join(f"  {o.path}:{o.lineno}: {o.message}" for o in offenders)
    )


# ----- Self-tests: snippet-based verification of the detector ----------
#
# These tests parse synthetic source strings so we can verify the
# walker catches each forbidden form WITHOUT ever adding a real
# forbidden import to any test module (which would trip the walker on
# itself). Each snippet lives entirely in a string literal.


def _has_violation_matching(source: str, substring: str) -> bool:
    return any(substring in v.message for v in walk_source_for_violations(source, "snippet"))


def test_detector_flags_import_urllib_request() -> None:
    src = "import urllib.request\n"
    assert _has_violation_matching(src, "urllib.request")


def test_detector_flags_from_urllib_import_request() -> None:
    src = "from urllib import request\n"
    assert _has_violation_matching(src, "urllib")


def test_detector_flags_from_urllib_request_import_urlopen() -> None:
    src = "from urllib.request import urlopen\n"
    assert _has_violation_matching(src, "urllib.request")


def test_detector_permits_from_urllib_import_parse() -> None:
    """urllib.parse is legitimate (URL manipulation, no I/O)."""
    src = "from urllib import parse\n"
    assert not walk_source_for_violations(src, "snippet")


def test_detector_flags_socket_socket_connect_direct() -> None:
    src = (
        "import socket\n"
        "socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('h', 80))\n"
    )
    assert _has_violation_matching(src, "socket.socket(...).connect(...)")


def test_detector_flags_socket_socket_connect_assigned() -> None:
    src = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.connect(('h', 80))\n"
    )
    assert _has_violation_matching(src, "socket.socket(...).connect(...)")


def test_detector_flags_socket_socket_connect_aliased_module() -> None:
    src = (
        "import socket as sk\n"
        "s = sk.socket()\n"
        "s.connect(('h', 80))\n"
    )
    assert _has_violation_matching(src, "socket.socket(...).connect(...)")


def test_detector_flags_socket_socket_connect_from_import_alias() -> None:
    src = (
        "from socket import socket as make_sock\n"
        "s = make_sock()\n"
        "s.connect(('h', 80))\n"
    )
    assert _has_violation_matching(src, "socket.socket(...).connect(...)")


def test_detector_permits_unrelated_connect_call() -> None:
    """A .connect() call on a non-socket receiver must not trip the detector."""
    src = (
        "class Foo:\n"
        "    def connect(self): pass\n"
        "Foo().connect()\n"
    )
    assert not walk_source_for_violations(src, "snippet")


def test_detector_permits_bare_socket_import_without_connect() -> None:
    """Importing socket alone (no ``.connect()`` call) is legal."""
    src = "import socket\ns = socket.socket()\n"
    assert not walk_source_for_violations(src, "snippet")
