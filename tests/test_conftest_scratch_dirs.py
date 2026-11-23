"""The per-test scratch directories must not make setup cost grow with the suite.

Six autouse fixtures in `tests/conftest.py` give every test a fresh directory. Until
2026-11-22 each asked `tmp_path_factory.mktemp(<prefix>)`, which lists the whole base
temp to pick the next number; the directories are kept for the session, so every test
made every later test's six listings longer. Measured on 6,516 tests in one process:
564.6 s before, 175.3 s after the directories moved to `conftest._fresh_scratch_dir`
(`architecture/testing.md`, "Where the tests rung's time goes").

Two halves. The behaviour: the helper still hands out what the fixtures promise — a new,
empty, existing directory nobody else was given — and puts it one level below the base
temp. The rail: no function-scoped autouse fixture in `conftest.py` may call a numbered
`mktemp` again, because that is a per-test listing of a directory that only grows, and it
is invisible in any one test's timing.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import conftest

CONFTEST = Path(conftest.__file__)
PREFIXES = ("kg", "research", "lloyd-records", "usage", "dedupe", "automod-backlog")


def test_the_helper_hands_out_a_new_empty_directory_below_the_base_temp(tmp_path_factory):
    base = Path(tmp_path_factory.getbasetemp())
    top_level_before = set(os.listdir(base))
    made = [conftest._fresh_scratch_dir(tmp_path_factory, p) for p in PREFIXES for _ in range(2)]

    assert len(set(made)) == len(made), "two callers were handed the same directory"
    for path in made:
        assert path.is_dir() and not any(path.iterdir()), path
        assert path.parent == base / "isolate", (
            f"{path} is not one level below the base temp, so `tmp_path`'s numbering scan "
            "would walk it")
        assert (path.stat().st_mode & 0o777) == 0o700, oct(path.stat().st_mode)
    assert {p.name.rsplit("-", 1)[0] for p in made} == set(PREFIXES), made
    assert set(os.listdir(base)) - top_level_before <= {"isolate"}, (
        "the helper added entries to the base temp's top level, which is the listing every "
        "numbered mktemp in the session pays for")


def test_the_defaults_a_test_is_given_are_the_helpers_directories(tmp_path_factory):
    """The six fixtures are wired to the helper, read off what this very test was given."""
    from app import event_log, kg_store, research_store, sessions_io, usage_store

    iso = Path(tmp_path_factory.getbasetemp()) / "isolate"
    given = {
        "kg": Path(kg_store._default_path).parent,
        "research": Path(research_store._default_path).parent,
        "lloyd-records": Path(sessions_io.SESSIONS_DIR).parent,
        "usage": Path(usage_store.DB_PATH).parent,
    }
    assert Path(event_log.EVENT_LOGS_DIR).parent == given["lloyd-records"]
    for prefix, path in given.items():
        assert path.parent == iso and path.name.rsplit("-", 1)[0] == prefix, (prefix, path)
        assert path.is_dir(), f"the {prefix} default points into a directory that does not exist"
    assert len(set(given.values())) == len(given)


def _function_scoped_autouse_fixtures(tree: ast.Module) -> list[ast.FunctionDef]:
    out = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and ast.unparse(dec.func) == "pytest.fixture"):
                continue
            kw = {k.arg: ast.unparse(k.value) for k in dec.keywords}
            if kw.get("autouse") == "True" and kw.get("scope", "'function'").strip("'\"") == "function":
                out.append(node)
    return out


def _numbered_mktemp_calls(fn: ast.FunctionDef) -> list[int]:
    return [n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "mktemp"]


def test_no_per_test_autouse_fixture_lists_the_base_temp():
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    fixtures = _function_scoped_autouse_fixtures(tree)
    assert len(fixtures) >= 10, (
        f"only {len(fixtures)} function-scoped autouse fixtures were found in conftest.py; "
        "the scan below would be passing over nothing")
    offenders = {f.name: _numbered_mktemp_calls(f) for f in fixtures if _numbered_mktemp_calls(f)}
    assert not offenders, (
        "a per-test autouse fixture calls a numbered mktemp, which lists the whole base temp "
        f"once per test — use conftest._fresh_scratch_dir: {offenders}")
    users = [f.name for f in fixtures
             if any(isinstance(n, ast.Name) and n.id == "_fresh_scratch_dir" for n in ast.walk(f))]
    assert len(users) == len(PREFIXES), users


def test_the_rail_can_see_the_call_it_forbids():
    """The scan above goes red on the shape it replaced, so its green is a reading."""
    old = ast.parse(
        "import pytest\n"
        "@pytest.fixture(autouse=True)\n"
        "def _isolate_usage_store(tmp_path_factory, monkeypatch):\n"
        "    monkeypatch.setattr('app.usage_store.DB_PATH',\n"
        "                        tmp_path_factory.mktemp('usage') / 'usage.db')\n"
        "@pytest.fixture(scope='session', autouse=True)\n"
        "def _once(tmp_path_factory):\n"
        "    return tmp_path_factory.mktemp('djev-shadow')\n")
    fixtures = _function_scoped_autouse_fixtures(old)
    assert [f.name for f in fixtures] == ["_isolate_usage_store"], (
        "a session-scoped fixture pays for one listing, not one per test, and is not the defect")
    assert _numbered_mktemp_calls(fixtures[0]) == [5]
