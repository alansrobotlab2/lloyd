"""An optional solver stays out of import time, so the suite's counts do not
depend on which venv ran it (#1073).

The automod gate's `tests` rung floors are absolute: `PYTEST_MIN_COLLECTED =
1000` and `PYTEST_MIN_PASSED = 1000` (`scripts/automod/gate.py:56` and `:62`).
A module-scope `import z3` in anything pytest collects makes a *missing optional
package* a collection error, and a collection error takes the whole run under
the floor: the round that touches requirements is the round whose candidate venv
gets rebuilt from `requirements.lock`, and the lock is where a solver is
deliberately never written. So the one round that would notice is the round that
has no reason to expect it.

The rule this enforces is the shape, not the package: solver access is resolved
where it runs. `pytest.importorskip("z3")` is the sanctioned idiom **inside a
test or a fixture** — and NOT at module scope, which is the trap this guard's
first draft fell into: a module-scope `importorskip` raises `Skipped` during
*collection*, so the module's tests never enter the collected count at all when
the package is absent. That is a different `collected` number with and without
z3, which is exactly what the floors cannot absorb — the same failure as an
`ImportError` at import time, wearing a skip.

The corpus is therefore whatever pytest is configured to collect — the
`testpaths` line of `pytest.ini`, parsed, not a hand-copied list — because a
collected directory this file does not scan is a hole in the clause, not a
narrower guard. Runtime application code is NOT scanned on purpose: a service
that needs a solver and cannot get one should fail loudly, not skip quietly.

The guard is an AST walk, not a grep: a string literal in a docstring or a
parametrised fixture that contains the words `import z3` is not an import, and
this file carries several of them.
"""

from __future__ import annotations

import ast
import configparser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Top-level package names that are optional in `.venvs/lloyd`. Adding a second
# solver means adding its root here and nothing else.
OPTIONAL_SOLVER_ROOTS = frozenset({"z3"})
# Calls that import at the moment the surrounding statement runs, so they are
# import-time work even though they are not `import` statements.
RUNTIME_IMPORT_CALLEES = frozenset({"importorskip", "import_module", "__import__"})


def _testpaths() -> tuple[str, ...]:
    """The trees pytest collects, read from `pytest.ini` rather than copied."""
    cp = configparser.ConfigParser()
    cp.read(REPO_ROOT / "pytest.ini")
    return tuple(cp.get("pytest", "testpaths", fallback="").split())


TESTPATHS = _testpaths()


def _root_of(dotted: str) -> str:
    return dotted.split(".")[0]


def _callee_name(func: ast.expr) -> str:
    """`pytest.importorskip` -> `importorskip`; `__import__` -> `__import__`."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _requested_module(call: ast.Call) -> str | None:
    """The package name a dynamic-import call asks for, if it is a literal."""
    candidates: list[ast.expr] = list(call.args[:1])
    candidates += [kw.value for kw in call.keywords if kw.arg in ("name", "module")]
    for node in candidates:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
    return None


def _executed_calls(node: ast.stmt) -> list[ast.Call]:
    """Calls that run when *this statement* runs.

    Only the statement's own expression parts count: a decorator's arguments and
    a default value evaluate at definition time, so a call sitting in them is
    import-time work. A call inside a nested statement, or inside a lambda's
    body, does not — that is reached by the walk when it gets to the statement
    that owns it, or never.
    """
    out: list[ast.Call] = []

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.Lambda):
            visit(n.args)              # defaults evaluate now; the body does not
            return
        if isinstance(n, ast.Call):
            out.append(n)
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)):
                continue
            visit(child)

    visit(node)
    return out


def module_scope_solver_imports(source: str, filename: str = "<source>") -> list[str]:
    """Import-time solver imports: every way of binding the package that runs
    when the module is imported.

    "Import time" is the question the floors actually ask, so the walk descends
    through the containers that execute at import — `if`, `try`, `with`,
    `for`, `while`, `match`, `class` — and stops at a function or lambda body,
    which does not run until something calls it. A module-level
    `try: import z3 / except ImportError` is therefore still reported: it is
    import-time work whose absence has to be turned into a per-test skip by the
    test that wrote it, not into a collection-time event.
    """
    found: list[str] = []

    def report(node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root_of(alias.name) in OPTIONAL_SOLVER_ROOTS:
                    found.append(f"{filename}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if _root_of(node.module) in OPTIONAL_SOLVER_ROOTS:
                found.append(f"{filename}:{node.lineno}: from {node.module} import ...")
        for call in _executed_calls(node):
            if _callee_name(call.func) not in RUNTIME_IMPORT_CALLEES:
                continue
            requested = _requested_module(call)
            if requested and _root_of(requested) in OPTIONAL_SOLVER_ROOTS:
                found.append(f"{filename}:{call.lineno}: dynamic import of {requested}")

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            report(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue          # deferred: runs when called, not at import
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    walk([child])
                elif isinstance(child, (ast.excepthandler, ast.match_case)):
                    walk(child.body)   # `except ...:` and `case ...:` bodies

    tree = ast.parse(source, filename=filename)
    walk(tree.body)
    return found


def _scanned_files() -> list[Path]:
    return sorted({p for d in TESTPATHS for p in (REPO_ROOT / d).rglob("*.py")})


def test_the_corpus_is_what_pytest_actually_collects():
    """A 0-hit walk over a path that resolved to nothing is not a clean bill.

    The class this guards against is the false zero, and the cheapest instance
    of it is a glob that matched no file at all — which is why the corpus is
    read out of `pytest.ini` and then verified to resolve, rather than listed
    by hand here. A testpath that contributes no file is a hole, not a verdict.
    """
    assert TESTPATHS, "pytest.ini has no `testpaths`, so this guard scans nothing"
    assert len(TESTPATHS) >= 3, f"testpaths shrank to {TESTPATHS}; the guard narrowed with it"
    files = _scanned_files()
    assert len(files) > 100, f"expected the collected trees to be on disk, got {len(files)}"
    assert any(p.name == "test_solver_import_is_lazy.py" for p in files), \
        "this file is itself part of the corpus it scans"
    for tp in TESTPATHS:
        assert any(p.is_relative_to(REPO_ROOT / tp) for p in files), \
            f"`testpaths` names {tp!r} but no collected file resolves under it"


def test_nothing_at_import_time_imports_the_solver():
    """The clause: collected and passed counts are identical with or without
    the package, so `PYTEST_MIN_COLLECTED`/`PYTEST_MIN_PASSED` hold either way.
    """
    offenders = []
    for path in _scanned_files():
        offenders += module_scope_solver_imports(
            path.read_text(encoding="utf-8"), filename=str(path.relative_to(REPO_ROOT)))
    assert not offenders, "solver imported at module scope: " + "; ".join(offenders)


@pytest.mark.parametrize("source,expected", [
    # Plain imports.
    ("import z3\n", 1),
    ("import os\nfrom z3 import Solver\n", 1),
    ("from z3.z3 import Solver\n", 1),
    ("if True:\n    import z3\n", 1),
    ("with open('x') as f:\n    import z3\n", 1),
    ("try:\n    import z3\nexcept ImportError:\n    z3 = None\n", 1),
    ("class C:\n    import z3\n", 1),
    # Dynamic imports at module scope: same import-time effect, and with
    # `importorskip` the effect is a *collection* event, which is why a skip is
    # not a substitute for resolving it inside the test.
    ("import pytest\npytest.importorskip('z3')\n", 1),
    ("import pytest\nz3 = pytest.importorskip('z3')\n", 1),
    ("import importlib\nimportlib.import_module('z3')\n", 1),
    ("z3 = __import__('z3')\n", 1),
    ("class C:\n    z3 = __import__('z3')\n", 1),
    # A decorator's arguments evaluate at definition time, so this is import time.
    ("import pytest\n@pytest.mark.parametrize('m', pytest.importorskip('z3').cases)\n"
     "def test_a(m):\n    assert m\n", 1),
    # Deferred: not import time, so not this guard's business.
    ("def f():\n    import z3\n    return z3\n", 0),
    ("async def f():\n    from z3 import Solver\n", 0),
    ("import pytest\n\n\n@pytest.fixture\ndef z3mod():\n"
     "    return pytest.importorskip('z3')\n", 0),
    ("import pytest\n@pytest.fixture(params=['a'])\ndef z3mod(request):\n"
     "    import z3\n    return z3, request.param\n", 0),
    ("import importlib\n\ndef f():\n    return importlib.import_module('z3')\n", 0),
    # The sanctioned idiom: a per-test skip naming the reason, deferred.
    ("def test_something(z3):\n    assert z3\n", 0),
    # A different optional package is not this guard's package.
    ("import pytest\npytest.importorskip('numpy')\n", 0),
    # A name that merely shares the prefix is not the solver.
    ("import z3plus\nfrom z3_solver import x\nimport myz3\n", 0),
    # A relative import of a local module is not a package import.
    ("from . import z3shim\n", 0),
    # An unrelated call that happens to take a string.
    ("import importlib\nimportlib.reload(importlib)\n", 0),
])
def test_the_guard_reads_import_time_not_the_spelling(source: str, expected: int):
    """The positive controls. Without these, the corpus test above is a grep
    that could be returning 0 because the pattern matches nothing."""
    assert len(module_scope_solver_imports(source)) == expected, source


def test_the_guard_parses_its_own_file():
    """This file quotes `import z3` and `importorskip('z3')` in prose and in
    fixtures; the guard is an AST walk precisely so that quoting is not
    importing, and a corpus that included this file must still come back clean."""
    src = (REPO_ROOT / "tests" / "test_solver_import_is_lazy.py").read_text(encoding="utf-8")
    assert module_scope_solver_imports(src, "tests/test_solver_import_is_lazy.py") == []
