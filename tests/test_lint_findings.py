"""One definition of "is this finding new?", shared by the gate and the model.

The gate judges pyflakes and tsc as deltas because the tree carries ~69
tolerated pyflakes findings; an absolute bar would fail every round forever.
Post-edit diagnostics have to normalise identically or the model is told
about findings the gate will not mind — or, worse, not told about one it
will. Two private copies of that rule is exactly how they would drift.
"""

from __future__ import annotations

import inspect
from collections import Counter

from app import lint_findings as LF
from scripts.automod import gate as G


# ── the normalisers themselves ──────────────────────────────────────────────

def test_pyflakes_line_drops_position_but_keeps_path_and_message():
    """A finding that merely moved down the file is not a new finding."""
    assert LF.normalize_pyflakes_line("app/x.py:12:5: undefined name 'bar'") == \
        "app/x.py: undefined name 'bar'"
    assert LF.normalize_pyflakes_line("app/x.py:900:5: undefined name 'bar'") == \
        "app/x.py: undefined name 'bar'"


def test_pyflakes_keeps_an_unparseable_line_verbatim():
    assert LF.normalize_pyflakes_line("could not compile") == "could not compile"
    assert LF.normalize_pyflakes_line("   ") is None


def test_pyflakes_findings_are_a_set():
    text = ("a.py:1:1: 'os' imported but unused\n"
            "a.py:9:1: 'os' imported but unused\n"
            "b.py:2:1: undefined name 'x'\n")
    assert LF.parse_pyflakes(text) == {
        "a.py: 'os' imported but unused",
        "b.py: undefined name 'x'",
    }


def test_tsc_findings_are_a_multiset():
    """A second copy of an existing error in one file IS a new error."""
    text = ("src/a.tsx(10,3): error TS2339: Property 'x' does not exist.\n"
            "src/a.tsx(40,3): error TS2339: Property 'x' does not exist.\n")
    c = LF.parse_tsc(text)
    assert c["src/a.tsx: error TS2339: Property 'x' does not exist."] == 2


def test_tsc_ignores_non_error_output():
    assert LF.parse_tsc("Found 0 errors.\nwatching for changes\n") == Counter()


def test_split_tsc_by_file_groups_on_the_path_the_normaliser_produced():
    c = LF.parse_tsc(
        "src/a.tsx(1,1): error TS1: one\n"
        "src/a.tsx(2,1): error TS2: two\n"
        "src/b.ts(3,1): error TS3: three\n")
    by_file = LF.split_tsc_by_file(c)
    assert set(by_file) == {"src/a.tsx", "src/b.ts"}
    assert sum(by_file["src/a.tsx"].values()) == 2


def test_node_env_prefixes_path_and_drops_node_options(monkeypatch):
    monkeypatch.setenv("NODE_OPTIONS", "--inspect")
    env = LF.node_env()
    assert env["PATH"].startswith("/usr/local/bin:/usr/bin:/bin:")
    assert "NODE_OPTIONS" not in env


# ── the gate uses them, rather than keeping its own copy ────────────────────

def test_the_gate_delegates_rather_than_reimplementing():
    assert G._parse_tsc is LF.parse_tsc
    assert G._node_env is LF.node_env
    src = inspect.getsource(G._pyflakes)
    assert "lint_findings.parse_pyflakes" in src
    assert "re.match" not in src, "the gate grew a second normaliser again"


def test_the_aggregator_never_imports_the_automod_package():
    """`app/lint_findings.py` exists so this stays true.

    `scripts.automod.gate` pulls in worktrees, promotion and ledger state.
    An aggregator that imported it would put the whole self-modification
    package behind every tool call in every session.
    """
    from pathlib import Path
    import agent_mcp
    pkg = Path(agent_mcp.__file__).parent
    offenders = [
        p.name for p in sorted(pkg.glob("*.py"))
        if "scripts.automod.gate" in p.read_text()
        or "from scripts.automod import gate" in p.read_text()
    ]
    assert offenders == [], offenders


def test_lint_findings_imports_nothing_from_the_app_package():
    """It is shared with the aggregator, so it must stay dependency-free."""
    from pathlib import Path
    src = Path(LF.__file__).read_text()
    for line in src.splitlines():
        if line.startswith(("import ", "from ")):
            assert not line.startswith(("import app", "from app",
                                        "import scripts", "from scripts")), line
