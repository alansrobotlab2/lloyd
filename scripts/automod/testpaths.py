"""One answer to "is this path a test file?", read from `pytest.ini`.

The review rung, its prechecks, the grader prompt, the tests rung's partial
narrowing and the after-the-fact review route each used to decide it with a
literal `startswith("tests/")`, while `pytest.ini`'s `testpaths` has said
`tests app/harness/tests scripts` for as long as the harness suite has
existed. The tests rung runs bare `pytest` from the worktree, so a test under
`app/harness/tests/` is collected and measured green — and then the review's
met-node rail refused every clause pinned to one (#1322: round
`SM_20260921_030016` had five clauses graded `met` and four downgraded for
where their test file lived). Widening one site moves the refusal to the
next, which is why this is a module rather than a second prefix.

Two shapes of testpath, because `scripts` is one. A testpath whose directory
is a test tree (its last component is `tests`) holds nothing but tests and
their helpers, so every `.py` under it is a test file — `conftest.py` and
`_helpers.py` included, exactly as root `tests/` was always read. Any other
testpath is ordinary code that pytest merely *walks*, and only a file it would
collect (`python_files`, default `test_*.py *_test.py`) is a test file there:
`scripts/automod/review.py` sits under a testpath and is not a test.

`pytest.ini` is read, never written — it is on the loop's never-touch list.
No file, or no `testpaths` key, is the root `tests/` reading the tree grew up
with, so a stripped fixture repo behaves as it did.

No public name here starts with `test`: pytest's default `python_functions`
is that prefix, so a test module importing one by name would collect it.
"""
from __future__ import annotations

import configparser
import fnmatch
from pathlib import Path

DEFAULT_TESTPATHS: tuple[str, ...] = ("tests",)
DEFAULT_PYTHON_FILES: tuple[str, ...] = ("test_*.py", "*_test.py")
TEST_TREE_NAMES: tuple[str, ...] = ("tests",)

# This module's own checkout: the tree whose pytest.ini the gate runs under
# when no worktree is named. Code, not data — `app.paths.LLOYD_HOME` is the
# same derivation.
_OWN_ROOT = Path(__file__).resolve().parents[2]

# (path, mtime_ns) → parsed. The callers ask per path in list comprehensions.
_cache: dict[tuple[Path, int], tuple[tuple[str, ...], tuple[str, ...]]] = {}


def _norm(p: str) -> str:
    return str(p).strip().replace("\\", "/").strip("/")


def _read_ini(root: Path | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(testpaths, python_files)` from `<root>/pytest.ini`, defaults when
    either is absent."""
    ini = Path(root or _OWN_ROOT) / "pytest.ini"
    try:
        key = (ini, ini.stat().st_mtime_ns)
    except OSError:
        return DEFAULT_TESTPATHS, DEFAULT_PYTHON_FILES
    hit = _cache.get(key)
    if hit is not None:
        return hit
    tps, pfs = DEFAULT_TESTPATHS, DEFAULT_PYTHON_FILES
    try:
        cp = configparser.ConfigParser(interpolation=None)
        cp.read(ini, encoding="utf-8")
        if cp.has_section("pytest"):
            raw = cp.get("pytest", "testpaths", fallback="")
            found = tuple(_norm(t) for t in raw.split() if _norm(t))
            if found:
                tps = found
            raw = cp.get("pytest", "python_files", fallback="")
            found = tuple(t.strip() for t in raw.split() if t.strip())
            if found:
                pfs = found
    except (configparser.Error, OSError, UnicodeDecodeError):
        pass
    _cache[key] = (tps, pfs)
    return tps, pfs


def read_testpaths(root: Path | None = None) -> tuple[str, ...]:
    """The `testpaths` pytest runs from `root`, in `pytest.ini` order."""
    return _read_ini(root)[0]


def is_test_tree(testpath: str) -> bool:
    """Whether every `.py` under `testpath` is a test file (`tests/`,
    `app/harness/tests/`) rather than code pytest merely walks (`scripts/`)."""
    return _norm(testpath).rsplit("/", 1)[-1] in TEST_TREE_NAMES


def owning_testpath(rel: str, root: Path | None = None) -> str:
    """The testpath `rel` sits under — the path itself counts — or ""."""
    rel = _norm(rel)
    if not rel:
        return ""
    for tp in read_testpaths(root):
        if rel == tp or rel.startswith(tp + "/"):
            return tp
    return ""


def is_test_path(rel: str, root: Path | None = None) -> bool:
    """`rel` belongs to the test suite: any path inside a test tree (a `.py`,
    a fixture, a data file), or a collectable test module under any other
    testpath. A testpath itself (`tests`, `scripts`) qualifies, being what a
    suite-level `pytest <dir>` names."""
    rel = _norm(rel)
    tp = owning_testpath(rel, root)
    if not tp:
        return False
    if rel == tp or is_test_tree(tp):
        return True
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatchcase(name, pat) for pat in _read_ini(root)[1])


def is_test_file(rel: str, root: Path | None = None) -> bool:
    """A `.py` file the suite owns — the set the review's changed-test list,
    the honesty prechecks and the tests rung's narrowing are built from."""
    rel = _norm(rel)
    return rel.endswith(".py") and rel != owning_testpath(rel, root) and is_test_path(rel, root)


def pick_test_files(paths, root: Path | None = None) -> list[str]:
    """The test files among `paths`, order kept."""
    return [p for p in paths if is_test_file(p, root)]
