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


def test_a_position_inside_the_message_is_not_part_of_the_findings_identity():
    """#1217. pyflakes writes "redefinition of unused '_local' from line 445".
    With the finding's own line:col stripped and that one kept, a three-line
    edit above a pre-existing redefinition made all eight in one file read as
    new, and refused a round whose review had passed."""
    from app import lint_findings as L
    before = "tests/t.py:594:1: redefinition of unused '_local' from line 445\n"
    after = "tests/t.py:597:1: redefinition of unused '_local' from line 448\n"
    assert L.parse_pyflakes(before) == L.parse_pyflakes(after)
    assert L.normalize_pyflakes_line(before) == "tests/t.py: redefinition of unused '_local' from line N"
    # …and a different name is still a different finding.
    other = "tests/t.py:597:1: redefinition of unused '_session' from line 448\n"
    assert L.parse_pyflakes(other) != L.parse_pyflakes(after)


# ── #734's contract, clause by clause ───────────────────────────────────────

def test_734_redefinitions_normalise_equal_across_a_line_shift():
    from app import lint_findings as L
    a = L.normalize_pyflakes_line("app/x.py:1:1: redefinition of unused 'x' from line 5")
    b = L.normalize_pyflakes_line("app/x.py:12:1: redefinition of unused 'x' from line 16")
    assert a == b


def test_734_shadowed_import_and_enclosing_scope_messages_survive_a_line_shift():
    from app import lint_findings as L
    for before, after in [
        ("app/x.py:2:5: import 'os' from line 1 shadowed by loop variable",
         "app/x.py:9:5: import 'os' from line 8 shadowed by loop variable"),
        ("app/x.py:3:11: local variable 'y' defined in enclosing scope on line 1 referenced before assignment",
         "app/x.py:10:11: local variable 'y' defined in enclosing scope on line 8 referenced before assignment"),
    ]:
        assert L.normalize_pyflakes_line(before) == L.normalize_pyflakes_line(after), before


def test_734_a_different_name_is_a_different_finding_and_digitless_findings_are_untouched():
    from app import lint_findings as L
    x = L.normalize_pyflakes_line("app/x.py:1:1: redefinition of unused 'x' from line 5")
    y = L.normalize_pyflakes_line("app/x.py:1:1: redefinition of unused 'y' from line 5")
    assert x != y
    assert L.normalize_pyflakes_line("app/x.py:7:3: undefined name 'bar'") == "app/x.py: undefined name 'bar'"


def test_734_inserting_lines_above_a_redefinition_is_an_empty_delta_both_ways():
    """The shape `gate.rung_static` compares: post - base and base - post."""
    import subprocess
    import sys
    import tempfile
    import textwrap
    from pathlib import Path
    from app import lint_findings as L
    body = textwrap.dedent("""\
        def f():
            return 1


        def f():
            return 2
        """)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.py"
        runs = []
        for text in (body, "# one\n# two\n# three\n" + body):
            p.write_text(text)
            r = subprocess.run([sys.executable, "-m", "pyflakes", p.name], cwd=d, capture_output=True, text=True)
            runs.append(L.parse_pyflakes(r.stdout + r.stderr))
    base, post = runs
    assert base, "positive control: pyflakes does report the redefinition"
    assert post - base == set() and base - post == set()


# ── parse_pyright (#2450) ───────────────────────────────────────────────────

def _pyright_payload(diagnostics: list[dict], *, analyzed: int = 2) -> str:
    """A `pyright --outputjson` payload of the shape 1.1.409 actually prints.

    The `range` object is what a key must NOT contain: pyright reports the
    finding's own line and column there, and a key that carried them would
    report every finding below an inserted line as new (#1210's pyflakes
    failure, in a different format).
    """
    import json
    return json.dumps({
        "version": "1.1.409", "time": "1791511167724",
        "generalDiagnostics": diagnostics,
        "summary": {"filesAnalyzed": analyzed, "errorCount": len(diagnostics),
                    "warningCount": 0, "informationCount": 0, "timeInSec": 0.9},
    })


def _diag(file: str, rule: str, message: str, line: int, char: int = 0,
          **extra: object) -> dict:
    d: dict = {"file": file, "severity": "error", "message": message, "rule": rule,
               "range": {"start": {"line": line, "character": char},
                         "end": {"line": line, "character": char + 4}}}
    d.update(extra)
    return d


def test_parse_pyright_returns_a_multiset_keyed_on_rule_and_file_not_line():
    """#2450 clause 1, in both directions the key has to get right.

    Two identical findings on different lines are two findings (a set would
    hide the second one, which is why `parse_tsc` is a Counter and
    `parse_pyflakes` is not), and the same finding at a different line is the
    same key (which is what makes a head-minus-base delta mean anything).
    """
    from collections import Counter

    from app import lint_findings as L
    payload = _pyright_payload([
        _diag("/wt/caller.py", "reportCallIssue", 'No parameter named "env"', 3),
        _diag("/wt/caller.py", "reportCallIssue", 'No parameter named "env"', 9, 12),
        _diag("/wt/app/service.py", "reportArgumentType",
              'Argument of type "str" is incompatible with parameter of type "int"', 41),
    ])
    got = L.parse_pyright(payload, root="/wt")
    assert isinstance(got, Counter)
    assert got == Counter({
        'caller.py: reportCallIssue: No parameter named "env"': 2,
        'app/service.py: reportArgumentType: Argument of type "str" is '
        'incompatible with parameter of type "int"': 1,
    }), dict(got)
    # The line is nowhere in the key: neither of the two line numbers (3, 9)
    # nor the columns (0, 12) appear in any key on their own.
    assert all("3" not in k.split(": ")[0] for k in got)


def test_parse_pyright_of_the_same_tree_at_two_roots_gives_the_same_keys():
    """The delta compares a run in the worktree against a run at the round's
    base commit in a different directory. Absolute paths in the key would make
    every one of those a difference, so the root is stripped."""
    from app import lint_findings as L
    head = _pyright_payload([
        _diag("/home/alansrobotlab/lloyd-work/SM_T/home/lloyd/caller.py",
              "reportCallIssue", 'No parameter named "env"', 3)])
    base = _pyright_payload([
        _diag("/home/alansrobotlab/lloyd-work/SM_T/gate-state/checkout-base/caller.py",
              "reportCallIssue", 'No parameter named "env"', 3)])
    h = L.parse_pyright(head, root="/home/alansrobotlab/lloyd-work/SM_T/home/lloyd")
    b = L.parse_pyright(base, root="/home/alansrobotlab/lloyd-work/SM_T/gate-state/checkout-base")
    assert h == b
    assert list(h) == ['caller.py: reportCallIssue: No parameter named "env"']


def test_parse_pyright_keeps_a_diagnostic_with_no_rule_and_drops_nothing():
    """A syntax error carries no `rule`; a project-level diagnostic carries no
    file. Neither may vanish, because an unreported finding is a check that
    quietly stopped existing."""
    from app import lint_findings as L
    syntax = _diag("/wt/one.py", "reportInvalidSyntax", "Expected expression", 1)
    syntax.pop("rule")                       # pyright omits it on a syntax error
    got = L.parse_pyright(_pyright_payload([
        syntax,
        {"file": "", "severity": "error", "message": '"basic" is not a valid typeCheckingMode'},
    ]), root="/wt")
    assert sum(got.values()) == 2, dict(got)
    assert got["one.py: (no rule): Expected expression"] == 1
    assert got["(project): (no rule): \"basic\" is not a valid typeCheckingMode"] == 1


def test_parse_pyright_of_text_that_is_not_a_payload_is_empty_not_wrong():
    """A pyright that crashed prints prose, and prose that parses to zero
    findings must not be readable as "the tree is clean" — the caller checks
    the payload before believing an empty result, and this is the fact it
    checks against."""
    from app import lint_findings as L
    assert L.parse_pyright("") == {}
    assert L.parse_pyright('File or directory "x.py" does not exist\n') == {}
    assert L.parse_pyright('{"generalDiagnostics": "not a list"}') == {}
