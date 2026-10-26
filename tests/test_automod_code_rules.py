"""The content-rule rung (#2574): its rule file, its runner, and its zero-violator control.

`scripts/automod/code_rules.toml` is the first surface where Lloyd states, in the
tree, "this code shape must never appear in a diff, mechanically, in any round",
with the incident that produced each rule sitting next to the pattern. The
runner is stdlib `ast` only — Semgrep is not installed, `~/lloyd/.venvs/` is a
protected path a round cannot pip-install into, and a rung that needs a wheel
the round cannot fetch is a rung that silently stops running.

Four things are pinned here, one per acceptance clause, and each one is a claim
about a *shape*, never about a file count (the lesson `test_automod_vet.py`
learned: a control that pins a count rots within days).
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.automod import code_rules as CR  # noqa: E402


RULES_PATH = ROOT / "scripts" / "automod" / "code_rules.toml"


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(tree), *args],
                          capture_output=True, text=True, check=True).stdout


def _scratch_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    """A real git repo with one commit, then `files` committed on top."""
    tree = tmp_path / "wt"
    tree.mkdir()
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "config", "user.email", "rules@example.invalid")
    _git(tree, "config", "user.name", "rules")
    (tree / "README.md").write_text("base\n", encoding="utf-8")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "base")
    base = _git(tree, "rev-parse", "HEAD").strip()
    for rel, text in files.items():
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "the round")
    return tree, base


# ---------------------------------------------------------------------------
# Planting corpus: one violation per rule that exists in the rule file.
#
# The line each fixture plants is the line the rule must report, so a rule whose
# matcher quietly stopped matching shows up as "no hit for this rule" rather than
# as a green suite. `PLANTED` is keyed by rule id and the test is parametrized
# over the *file's* rules, which means adding a rule to the TOML without adding
# a fixture here is a red test, not an untested rule.
# ---------------------------------------------------------------------------

#: Every id the rule file ships, read at import. The four tests below are parametrized
#: over THIS and never over `PLANTED`'s keys. A rule in the TOML with no fixture entry is
#: the one case this suite promises to catch — "adding a rule to the TOML without adding a
#: fixture here is a red test, not an untested rule" — and driving the parametrization off
#: the fixture dict answers that question with a green run whatever the rule file holds.
SHIPPED_RULE_IDS = sorted(r.id for r in CR.load_rules(RULES_PATH))

PLANTED: dict[str, tuple[str, str, int]] = {
    # rule id -> (repo path, source, 1-based line of the offending shape)
    "guardian-import-hygiene": (
        "agent-services/guardian/planted.py",
        "import json\nimport yaml\n\ndef go():\n    return json\n",
        2,
    ),
    "opens-kg-store-outside-kg-store": (
        "app/planted_store.py",
        "import sqlite3\n\n"
        "con = sqlite3.connect(TREE / '_pipeline' / 'vault-derived' / 'kg.sqlite')\n",
        3,
    ),
    "broad-except-swallow-same-line": (
        "app/planted_swallow.py",
        "def go(q):\n"
        "    try: q.put_nowait(None)\n"
        "    except Exception: pass\n",
        3,
    ),
    "naive-now-in-artifact-name": (
        "app/planted_stamp.py",
        "from datetime import datetime\n\n"
        'name = LOGS / f"run-{datetime.now():%Y%m%d}.json"\n',
        3,
    ),
    "exports-lloyd-data-to-process-environ": (
        "app/planted_env.py",
        "import os\n\nos.environ['LLOYD_DATA'] = str(root)\n",
        3,
    ),
}

#: The second arm of `opens-kg-store-outside-kg-store`: a store path that reached the
#: `connect` through a name rather than a literal. It gets its own fixture dict keyed by
#: the *binder*, because `PLANTED` is keyed by rule id and holds one source per rule, and
#: because the firing test over that dict asserts `counts == 1` for this rule — folding a
#: second shape into it would stop pinning that the original literal spelling still fires
#: at its own line (#3357 clause 4). Both live non-owner openers are of this shape, which
#: is why the arm exists: `agent-services/guardian/guardian.py:2897` is the parameter
#: binder, `scripts/automod/promote.py:502` the assignment binder.
#: binder -> (repo path, source, 1-based line of the `connect`, the name the hit must
#: name back, the spelling the hit must credit as the source).
#:
#: The last two fields are what make each parametrized node a test of *its own* binder
#: rather than a test that the arm fires somewhere in this file's neighbourhood: a hit
#: that named the wrong binder — the assignment when the parameter is under test, say —
#: would still put one hit on the expected line. `as_written` is the identifier as it
#: appears inside the `connect`, `source_spelling` the binder the matcher resolved it to,
#: and the reported detail is pinned as the sentence carrying both.
#:
#: Every fixture here is a module that parses AND defines what it uses: the runner only
#: calls `ast.parse`, so an undefined name would not stop the arm firing, but it would
#: leave a reader running the snippet by hand and blaming the NameError for the hit.
PLANTED_RESOLVED: dict[str, tuple[str, str, int, str, str]] = {
    "assigned-from-kg_store_for_root": (
        "app/planted_resolved_root.py",
        "import sqlite3\n"
        "from app.data_root import kg_store_for_root\n"
        "\n"
        "def rows(root):\n"
        "    db = kg_store_for_root(root)\n"
        "    return sqlite3.connect(f\"file:{db}?mode=ro\", uri=True)\n",
        6,
        "db", "kg_store_for_root(...)",
    ),
    "assigned-from-kg_store_for_tree": (
        "app/planted_resolved_tree.py",
        "import sqlite3\n"
        "from app.data_root import kg_store_for_tree\n"
        "\n"
        "def rows():\n"
        "    store = kg_store_for_tree()\n"
        "    con = sqlite3.connect(store)\n"
        "    return con\n",
        6,
        "store", "kg_store_for_tree(...)",
    ),
    "assigned-from-KG_DB_RELATIVE": (
        "app/planted_resolved_relative.py",
        "import sqlite3\n"
        "from app.data_root import KG_DB_RELATIVE\n"
        "\n"
        "def open_it(root):\n"
        "    rel = KG_DB_RELATIVE\n"
        "    return sqlite3.connect(str(root / rel))\n",
        6,
        "rel", "KG_DB_RELATIVE",
    ),
    "assigned-from-VAULT_KG_DB": (
        "app/planted_resolved_vault.py",
        "import sqlite3\n"
        "from app.paths import VAULT_KG_DB\n"
        "\n"
        "def dump():\n"
        "    path = VAULT_KG_DB\n"
        "    con = sqlite3.connect(path)\n"
        "    return con\n",
        6,
        "path", "VAULT_KG_DB",
    ),
    # The guardian's shape: the open sits in a function, and the store arrives as the
    # argument this same file's own call passes it. A matcher that only reads assignments
    # cannot see it, which is the whole of #3357. What is under test here is the
    # PARAMETER, so the constant on line 4 is deliberately spelled as a plain `Path`
    # join — an expression the arm does not read as a store spelling (no
    # `kg_store_for_*` call, no `KG_DB`/`KG_DB_RELATIVE`/`VAULT_KG_DB` on the right-hand
    # side), so this module has exactly one binder and it is `db_path`. The node's
    # `as_written="db_path"` is what pins that: were the module-level assignment ever
    # counted as the binder, the detail would credit the assignment, not the parameter.
    "parameter-filled-by-this-file-s-call": (
        "app/planted_resolved_param.py",
        "import sqlite3\n"
        "from pathlib import Path\n"
        "\n"
        "VAULT_KG_DB = Path('_pipeline') / 'vault-derived' / 'kg.sqlite'\n"
        "\n"
        "\n"
        "def count_rows(db_path):\n"
        "    return sqlite3.connect(f\"file:{db_path}?mode=ro\", uri=True)\n"
        "\n"
        "\n"
        "print(count_rows(VAULT_KG_DB))\n",
        8,
        "db_path", "VAULT_KG_DB",
    ),
}

#: The remaining limit, planted as a fixture so the arm cannot silently grow into
#: cross-file flow the runner's contract refuses: here the path crosses a module
#: boundary before it reaches the name, and the matcher must say nothing.
PLANTED_CROSS_FILE = (
    "app/planted_resolved_imported.py",
    "import sqlite3\n"
    "from somewhere_else import store_path\n"
    "\n"
    "def rows(root):\n"
    "    db = store_path(root)\n"
    "    return sqlite3.connect(db)\n",
    6,
)


# ---------------------------------------------------------------------------
# Clause 1 — the rule file exists, is cited, and is capped.
# ---------------------------------------------------------------------------

def test_the_rule_file_exists_and_every_rule_cites_its_incident():
    """A rule with no citation cannot be added, so the file is checked for one.

    The citation is what keeps a hand-maintained rule set from decaying into
    #662's closed shape (a general anti-pattern catalogue): the rule is deletable
    when its incident stops being real, and undeletable-but-tunable when it is
    not — the guard is that every rule names a checkable event, item or doc claim.
    """
    assert RULES_PATH.is_file(), f"missing {RULES_PATH}"
    rules = CR.load_rules(RULES_PATH)
    assert rules, "the rule file loaded zero rules"
    for rule in rules:
        assert rule.id, f"rule with an empty id in {RULES_PATH.name}"
        assert rule.cite.strip(), f"rule {rule.id!r} carries no citation"
        assert CR.CITATION_RE.search(rule.cite), (
            f"rule {rule.id!r} cites {rule.cite!r}, which names no item (#NNNN), "
            "no round (SM_YYYYMMDD_*), no doc (CLAUDE.md / architecture/…), and "
            "no dated memory class rule")


def test_the_rule_file_holds_at_most_eight_rules():
    """The cap is enforced by the loader, not by the reviewer's memory."""
    rules = CR.load_rules(RULES_PATH)
    assert len(rules) <= CR.MAX_RULES, f"{len(rules)} rules, cap is {CR.MAX_RULES}"


def test_a_rule_with_no_citation_is_a_load_error(tmp_path):
    """Clause 1's negative half: an uncited rule is refused at load, not warned."""
    bad = tmp_path / "uncited.toml"
    bad.write_text(
        "[[rule]]\n"
        'id = "uncited-shape"\n'
        'matcher = "broad_except_swallow_same_line"\n'
        'summary = "a swallow"\n'
        'cite = ""\n'
        'scope = ["app/**"]\n',
        encoding="utf-8")
    with pytest.raises(CR.CodeRulesError) as exc:
        CR.load_rules(bad)
    assert "cite" in str(exc.value)


def test_the_cap_is_enforced_by_the_loader(tmp_path):
    """Nine cited rules is a load error, so the cap cannot be drifted past."""
    body = "".join(
        "[[rule]]\n"
        f'id = "r{n}"\n'
        'matcher = "broad_except_swallow_same_line"\n'
        'summary = "s"\n'
        'cite = "#2574 planted"\n'
        'scope = ["app/**"]\n'
        for n in range(CR.MAX_RULES + 1))
    big = tmp_path / "nine.toml"
    big.write_text(body, encoding="utf-8")
    with pytest.raises(CR.CodeRulesError) as exc:
        CR.load_rules(big)
    assert f"cap is {CR.MAX_RULES}" in str(exc.value), exc.value


def test_an_unknown_matcher_is_a_load_error(tmp_path):
    """A rule naming a matcher that does not exist would silently never fire."""
    bad = tmp_path / "typo.toml"
    bad.write_text(
        "[[rule]]\n"
        'id = "typo"\n'
        'matcher = "no_such_matcher"\n'
        'summary = "s"\n'
        'cite = "#2574 planted"\n'
        'scope = ["app/**"]\n',
        encoding="utf-8")
    with pytest.raises(CR.CodeRulesError):
        CR.load_rules(bad)


def test_every_allow_entry_names_its_reason(tmp_path):
    """An allowlist entry with no reason is how an allowlist becomes a default."""
    bad = tmp_path / "bare-allow.toml"
    bad.write_text(
        "[[rule]]\n"
        'id = "unexplained"\n'
        'matcher = "broad_except_swallow_same_line"\n'
        'summary = "s"\n'
        'cite = "#2574 planted"\n'
        'scope = ["app/**"]\n'
        'allow = ["app/kept.py"]\n',
        encoding="utf-8")
    with pytest.raises(CR.CodeRulesError) as exc:
        CR.load_rules(bad)
    assert "why" in str(exc.value)


# ---------------------------------------------------------------------------
# Clause 2 — the runner: stdlib only, per-rule hits over the round's diff, and
# `unevaluated` (never a clean-looking empty list) when a read fails.
# ---------------------------------------------------------------------------

def test_the_runner_adds_no_third_party_import():
    """The rung must run on the interpreter the gate already has.

    Semgrep is absent from every venv under `~/lloyd/.venvs` and that directory
    is protected at the tool layer, so a wheel-based runner could only ever be
    installed by hand. The import table is therefore a checked property, not a
    style preference.
    """
    tree = ast.parse((ROOT / "scripts" / "automod" / "code_rules.py").read_text(encoding="utf-8"))
    stdlib = set(sys.stdlib_module_names)
    local = {"scripts", "agent_mcp", "app", "workers"}
    roots: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            roots += [a.name.split(".")[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            if n.level:                      # relative: same package
                continue
            roots.append((n.module or "").split(".")[0])
    for r in roots:
        assert r in stdlib or r in local, f"code_rules.py imports {r!r}, which is neither stdlib nor local"
    assert "semgrep" not in roots


@pytest.mark.parametrize("rule_id", SHIPPED_RULE_IDS)
def test_a_rule_file_pinned_fixture_exists_for_every_shipped_rule(rule_id):
    """Adding a rule to the TOML without a planted violation is a red test here.

    The path has to be one the rule's own `scope` accepts. The firing test below asks
    "did this rule report the line", and a rule whose matcher had silently stopped
    matching answers that with zero hits — so a fixture planted outside the scope looks
    exactly as green as one that a live matcher is being checked against, and the
    direction this suite exists to guard (a rule that no longer fires) is the direction
    that stays invisible.
    """
    shipped = {r.id: r for r in CR.load_rules(RULES_PATH)}
    assert rule_id in shipped, f"{rule_id} has a fixture but no rule in the file"
    assert set(SHIPPED_RULE_IDS) <= set(PLANTED), (
        f"the rule file ships {sorted(set(SHIPPED_RULE_IDS) - set(PLANTED))} with no "
        "planted violation: the firing, suppression and live-tree tests are parametrized "
        "over the rule FILE's ids, so a rule without a fixture is a red test here rather "
        "than an untested rule")
    path = PLANTED[rule_id][0]
    assert shipped[rule_id].covers(path), (
        f"{rule_id}'s fixture is planted at {path!r}, which its own scope "
        f"{list(shipped[rule_id].scope)!r} does not cover — the firing test below "
        "would pass on a rule that can never see the file")


@pytest.mark.parametrize("rule_id", SHIPPED_RULE_IDS)
def test_each_shipped_rule_fires_on_its_planted_violation(tmp_path, rule_id):
    """One planted violation per rule, reported over the round's own diff.

    The repo path and the line both have to come back: a rule that reports the
    wrong site is as useless as one that reports nothing, and the reviewer reads
    these labels off the gate report.
    """
    shipped = {r.id: r for r in CR.load_rules(RULES_PATH)}
    assert rule_id in shipped, f"{rule_id} is in PLANTED but not in the rule file"
    rel, src, line = PLANTED[rule_id]
    tree, base = _scratch_repo(tmp_path, {rel: src})

    res = CR.run(base, tree, rules_path=RULES_PATH)

    assert res["status"] == CR.EVALUATED, res
    hits = [h for h in res["hits"] if h["rule"] == rule_id]
    assert [h["path"] for h in hits] == [rel], res["hits"]
    assert hits[0]["line"] == line, hits
    assert f"{rule_id}:{rel}" in res["labels"]
    assert res["counts"][rule_id] == 1, res["counts"]


@pytest.mark.parametrize("rule_id", SHIPPED_RULE_IDS)
def test_a_suppression_comment_does_not_silence_a_rule(tmp_path, rule_id):
    """Clause 3: suppressions are counted, never honoured.

    `# nosemgrep` is how a team defangs a rule rung one commit at a time. The
    runner is deliberately `--disable-nosemgrep`-by-construction: a marker on the
    offending line changes the reported count and nothing else.
    """
    rel, src, line = PLANTED[rule_id]
    lines = src.splitlines()
    lines[line - 1] = lines[line - 1].rstrip() + "  # noqa: E501"
    tree, base = _scratch_repo(tmp_path, {rel: "\n".join(lines) + "\n"})

    res = CR.run(base, tree, rules_path=RULES_PATH)

    hits = [h for h in res["hits"] if h["rule"] == rule_id]
    assert len(hits) == 1, (rule_id, res["hits"])
    assert res["totals"]["suppression_markers"] == 1, res["totals"]


def test_the_result_reports_suppression_markers_across_the_scanned_files(tmp_path):
    """The count is what keeps a clean record from being an illusion."""
    rel, src, _ = PLANTED["broad-except-swallow-same-line"]
    tree, base = _scratch_repo(tmp_path, {
        rel: src.replace("pass\n", "pass  # noqa: E501\n"),
        "app/other.py": "import re\nx = re  # type: ignore\n",
    })
    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["totals"]["suppression_markers"] == 2, res["totals"]


def test_a_missing_rule_file_is_unevaluated_and_never_clean(tmp_path):
    """A pass that could not load its rules must not read as a zero-hit pass."""
    tree, base = _scratch_repo(tmp_path, {"app/x.py": "X = 1\n"})
    res = CR.run(base, tree, rules_path=tmp_path / "absent.toml")
    assert res["status"] == CR.UNEVALUATED
    assert res["reason"], res
    # The absence that matters: a record with an empty hit list is what a caller reads as
    # "checked, found nothing". `counts`-shaped assertions self-satisfy when the key is
    # simply missing, so they cannot be the guard.
    assert "hits" not in res, "an unevaluated run must not carry a hit list"


def test_a_failed_git_read_is_unevaluated_not_clean(tmp_path):
    """Clause 2's contract, copied from `vet.py`: never report clean on failure."""
    tree, _ = _scratch_repo(tmp_path, {"app/x.py": "X = 1\n"})
    res = CR.run("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", tree, rules_path=RULES_PATH)
    assert res["status"] == CR.UNEVALUATED, res
    assert res["reason"], "an unevaluated run must say which read failed"


def test_an_unreadable_changed_file_is_unevaluated_not_clean(tmp_path, monkeypatch):
    """The other read in the loop: a file the runner cannot open is a gap it says so about."""
    rel, src, _ = PLANTED["broad-except-swallow-same-line"]
    tree, base = _scratch_repo(tmp_path, {rel: src})
    real_read_text = Path.read_text

    def _boom(self, *a, **k):
        if self.name == "planted_swallow.py":
            raise OSError("simulated EIO")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _boom)
    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["status"] == CR.UNEVALUATED, res
    assert "planted_swallow.py" in res["reason"], res


def test_the_scan_is_scoped_to_the_rounds_changed_python_files(tmp_path):
    """A violation sitting at base is not this round's; only added lines are."""
    rel, src, _ = PLANTED["broad-except-swallow-same-line"]
    tree = tmp_path / "wt"
    tree.mkdir()
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "config", "user.email", "rules@example.invalid")
    _git(tree, "config", "user.name", "rules")
    (tree / rel.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
    (tree / rel).write_text(src, encoding="utf-8")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "base already had it")
    base = _git(tree, "rev-parse", "HEAD").strip()
    (tree / "app" / "unrelated.py").write_text("Y = 2\n", encoding="utf-8")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "the round touches something else")

    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["status"] == CR.EVALUATED, res
    assert res["hits"] == [], res["hits"]
    # The positive control beside that zero. Asserting only the empty hit list would also
    # pass on a scan that read no files, on a rule whose matcher had died, or on a fixture
    # that had stopped containing the shape at all — so the swallow is asserted to still be
    # sitting in the tree: the zero here means "not this round's line", nothing weaker.
    assert "except Exception: pass" in (tree / rel).read_text(encoding="utf-8"), (
        f"{rel} no longer contains the planted swallow, so this zero proves nothing about "
        "diff-scoping")


def test_a_file_outside_a_rules_scope_is_not_scanned(tmp_path):
    """Scope is in the rule file, so a rule cannot silently widen to the tree.

    The guardian's import rule is the shipped rule with a directory-wide scope,
    and the same violation one directory over must be nothing: the runner's whole
    standing claim is that a rule's reach is written down, not inferred.
    """
    _rel, src, _ = PLANTED["guardian-import-hygiene"]
    tree, base = _scratch_repo(tmp_path, {"app/not_the_guardian.py": src})
    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["status"] == CR.EVALUATED, res
    assert [h for h in res["hits"]
            if h["rule"] == "guardian-import-hygiene"] == [], res["hits"]


# ---------------------------------------------------------------------------
# #3357 — the store-ownership rule learns to follow a path one hop.
#
# Before this, `opens_kg_store` fired only when the literal `kg.sqlite` sat inside the
# `sqlite3.connect(...)` call. Evaluated over every tracked `.py` at HEAD, its whole
# population was one test fixture: both non-owner openers that actually exist resolve
# the path into a name first, so the guard could not report them and so could not ask
# either to carry an exemption. Six kinds of node here: the new arm fires on each binder it
# claims and names that binder back in its own detail; it stops at the module boundary it
# claims to stop at; it leaves the literal arm's site and message exactly as they were
# (#3357's clause 4, whose named node runs the literal fixture alone); it survives the
# call path the gate's own rung takes, diff-scoped and line-filtered; the rule file's cite
# describes the two openers that exist rather than the one that does not; and the two
# witnesses are exempted in writing with the read-only handle named as the reason.
# ---------------------------------------------------------------------------

KG_RULE = "opens-kg-store-outside-kg-store"


@pytest.mark.parametrize("binder", sorted(PLANTED_RESOLVED))
def test_the_resolved_path_arm_fires_on_its_own_planted_violation(tmp_path, binder):
    """Clause 1: a store path carried into `connect` by a name is the same open.

    The line the hit lands on is the `connect`, not the assignment — the finding is about
    opening a second handle, and a reader looking at the assignment has not been shown the
    write-shaped call. The detail is pinned as a whole sentence rather than as a substring,
    because it is the only part of the hit that says *which* binder fired: `db` on its own
    names nothing, and a fixture with more than one binder in it (the guardian's shape has
    a constant in the module as well as the parameter) would otherwise pass with a hit
    credited to the wrong one. The reported sentence is
    `… through `<name>`, one hop from <spelling>`.
    """
    rel, src, line, as_written, source_spelling = PLANTED_RESOLVED[binder]
    tree, base = _scratch_repo(tmp_path, {rel: src})
    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["status"] == CR.EVALUATED, res

    hits = [h for h in res["hits"] if h["rule"] == KG_RULE]
    assert [h["path"] for h in hits] == [rel], res["hits"]
    assert hits[0]["line"] == line, (binder, hits[0])
    assert res["counts"][KG_RULE] == 1, res["counts"]
    assert hits[0]["detail"] == ("opens the knowledge-graph store outside its owner "
                                f"through `{as_written}`, one hop from {source_spelling}"), (
        f"the {binder} fixture's hit does not name the binder this node plants: "
        f"{hits[0]['detail']}")


def test_the_resolved_path_arm_stops_at_the_module_boundary_the_runner_refuses(tmp_path):
    """The arm's limit, planted: a path that crossed a file to reach the name is silent.

    `code_rules.py`'s own contract is that it parses one file at a time and does not do
    cross-module analysis, so the honest description of this arm is "one hop inside the
    file". A fixture whose store path arrives from an imported helper must produce nothing:
    if it fired, the matcher would be doing flow analysis it cannot do reliably and the
    docstring's promise would be the lie.
    """
    rel, src, _ = PLANTED_CROSS_FILE
    tree, base = _scratch_repo(tmp_path, {rel: src})
    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["status"] == CR.EVALUATED, res
    # The zero below is only a verdict if the matcher actually saw the file: `run`
    # skips a source it cannot parse and counts it in `parse_errors`, so an
    # unparseable fixture would report "no hit" while proving nothing. Asserted
    # before the emptiness, and it is the reason this fixture's module name is
    # `somewhere_else` and not something with a keyword in it.
    assert res["totals"]["parse_errors"] == 0, (
        f"the runner could not parse the fixture ({rel}), so the empty hit list "
        f"below is a skipped file, not a matcher that stopped at the boundary: "
        f"{res['totals']}")
    assert [h for h in res["hits"] if h["rule"] == KG_RULE] == [], res["hits"]


def test_the_literal_arm_still_reports_its_own_line_beside_the_resolved_one(tmp_path):
    """Clause 4: the arm that already existed is unchanged, line for line and word for word.

    The sibling node `test_each_shipped_rule_fires_on_its_planted_violation[
    opens-kg-store-outside-kg-store]` runs the literal fixture alone and pins its
    `counts == 1`. Alone, that node cannot see an interference the widening could introduce:
    a `connect` whose call carries both a `kg.sqlite` literal and a resolved name is one
    expression the two arms both claim, and an implementation that appended from both would
    report that open twice while every single-fixture node stayed green. So both fixtures
    sit in one tree here. Three claims, all of them about the literal hit: it is still
    reported, it is still reported at line 3 of its own file, and its message is still the
    sentence it has always carried — no hop clause bolted onto it. The resolved file next to
    it keeps its own hop-clause message, and the rule's count over the two files is exactly
    2, which is the assertion that no open was counted twice.
    """
    lit_rel, lit_src, lit_line = PLANTED[KG_RULE]
    res_rel, res_src, res_line = PLANTED_RESOLVED["assigned-from-kg_store_for_root"][:3]
    tree, base = _scratch_repo(tmp_path, {lit_rel: lit_src, res_rel: res_src})
    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["status"] == CR.EVALUATED, res

    hits = [h for h in res["hits"] if h["rule"] == KG_RULE]
    by_path = {h["path"]: h for h in hits}
    assert len(hits) == 2 and set(by_path) == {lit_rel, res_rel}, res["hits"]
    assert res["counts"][KG_RULE] == 2, res["counts"]
    assert by_path[lit_rel]["line"] == lit_line, by_path[lit_rel]
    assert by_path[res_rel]["line"] == res_line, by_path[res_rel]
    assert by_path[lit_rel]["detail"] == "opens the knowledge-graph store outside its owner", (
        "the literal arm's message moved while the resolved arm was added; clause 4 asks "
        "for the original form unchanged — the hop clause belongs to the new arm only: "
        f"{by_path[lit_rel]['detail']}")
    assert "one hop" not in by_path[lit_rel]["detail"], by_path[lit_rel]
    assert "one hop from" in by_path[res_rel]["detail"], by_path[res_rel]
    for rel in (lit_rel, res_rel):
        assert f"{KG_RULE}:{rel}" in res["labels"], (rel, res["labels"])


def test_the_kg_store_cite_names_the_two_openers_that_exist_and_the_hop_the_rule_follows():
    """Clause 3: the cite is the surface a reader trusts about who touches the store.

    What it said before was false on its face: "The one allowed opener, the guardian's
    read-only `count_kg_rows` tripwire ... opens no handle through `sqlite3.connect` on a
    restated path (#1525)" — that tripwire is a `sqlite3.connect` on a restated path, and a
    second file does the same thing the sentence never mentions. `architecture/autonomy-jobs.md`
    §Build the graph had already been rewritten to name two openers, so the rule file was
    the stale half of a pair. Pinned by substring rather than byte-for-byte
    (`test_the_rule_gained_the_three_script_roots_and_its_cite_did_not_move` pins that
    other rule's cite byte-for-byte) because this clause is a list of claims, and a reader
    has to be able to see which claim a rewrite dropped.
    """
    rule = {r.id: r for r in CR.load_rules(RULES_PATH)}[KG_RULE]
    cite = rule.cite

    assert "opens no handle through `sqlite3.connect` on a restated path" not in cite, (
        f"the refuted sentence is still the cite's account of the guardian:\n{cite}")
    for who in ("agent-services/guardian/guardian.py", "scripts/automod/promote.py",
                "count_kg_rows"):
        assert who in cite, f"the cite never names {who}, one of the two openers it must:\n{cite}"
    assert cite.count("?mode=ro") >= 2, (
        "both witnesses are read-only handles and that is the property that makes each "
        f"exemption survivable; the cite says it {cite.count('?mode=ro')} time(s):\n{cite}")
    assert "one hop" in cite, f"the cite does not say how far the matcher follows:\n{cite}"
    for spelling in ("kg_store_for_root", "kg_store_for_tree", "KG_DB_RELATIVE",
                     "VAULT_KG_DB"):
        assert spelling in cite, (
            f"the cite omits {spelling}, one of the spellings the arm resolves:\n{cite}")
    assert CR.CITATION_RE.search(cite), "a cite that names no item cannot be loaded anyway"


def test_the_two_read_only_witnesses_are_exempted_by_name_with_the_ro_handle_as_the_why():
    """Clause 2's other half: the exemptions exist, and each says what makes it safe.

    The two witnesses held no allow entry before #3357 for the worst available reason: not
    that they were judged and cleared, but that the guard could not see them to exempt them.
    Routing them through the allow list is what makes a *future* unlisted reader cost its
    author a written `why` — an exemption that names `?mode=ro` is checkable by anyone,
    because `uri=True` plus `mode=ro` is what forbids the write the invariant is about. The
    sibling node `test_the_live_tree_scores_zero_or_exactly_the_named_allowlist[
    opens-kg-store-outside-kg-store]` is what pins the live tree to exactly these paths.
    """
    rule = {r.id: r for r in CR.load_rules(RULES_PATH)}[KG_RULE]
    by_path = {a.path: a.why for a in rule.allow}

    assert set(by_path) == {"tests/test_data_home.py",
                            "agent-services/guardian/guardian.py",
                            "scripts/automod/promote.py"}, sorted(by_path)
    for path in ("agent-services/guardian/guardian.py", "scripts/automod/promote.py"):
        why = by_path[path]
        assert why.strip(), f"{path} is exempted with no reason at all"
        assert "?mode=ro" in why, f"{path}'s exemption never names its read-only handle: {why}"
        assert "uri=True" in why, (
            f"{path}'s exemption names `?mode=ro` without `uri=True`, and mode=ro is only "
            f"a query string until uri=True makes SQLite read it: {why}")


def test_the_gate_s_own_call_form_carries_the_resolved_open_on_added_lines(tmp_path):
    """The seam the review named, put under test: the rung calls `CR.run(base, worktree)`.

    `scripts/automod/gate.py::rung_code_rules` invokes the runner with the two arguments a
    round has and nothing else — no `files=`, no `whole_tree=True` — so the runner itself
    enumerates the round's changed `.py` and then keeps only hits whose line the diff ADDED
    (`_added_lines`; a site that predates the base is not this round's). That is a different
    call path from every other node in this section, which either hands `run` a scratch diff
    or hands it `git ls-files '*.py'` with `whole_tree=True`, and it is the path the soak's
    denominator is actually built from. Line-based scoping is where a name-resolution arm
    could go quietly wrong, because the binder and the `connect` are two different lines:
    a hit filtered on the *binder's* line having been added would report nothing for the
    common shape of a new `connect` written under an existing assignment.

    Two files, then. `app/legacy_store_reader.py` is committed BEFORE the base, so its
    resolved open predates the round: a diff-scoped run must say nothing about it, which is
    the control that proves this run was scoped to the diff and not to the tree.
    `app/planted_resolved_tree.py` (the same fixture the binder nodes plant) is added by the
    round's commit with the assignment and the open both new, and it must come back as one
    hit on the `connect`'s own line, which is line 6 of that source.
    """
    tree = tmp_path / "wt"
    tree.mkdir()
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "config", "user.email", "rules@example.invalid")
    _git(tree, "config", "user.name", "rules")
    legacy_rel = "app/legacy_store_reader.py"
    (tree / "README.md").write_text("base\n", encoding="utf-8")
    p = tree / legacy_rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("import sqlite3\n"
                 "from app.data_root import kg_store_for_root\n"
                 "\n"
                 "def rows(root):\n"
                 "    db = kg_store_for_root(root)\n"
                 "    return sqlite3.connect(f\"file:{db}?mode=ro\", uri=True)\n",
                 encoding="utf-8")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "base")
    base = _git(tree, "rev-parse", "HEAD").strip()

    new_rel, new_src, new_line = PLANTED_RESOLVED["assigned-from-kg_store_for_tree"][:3]
    p = tree / new_rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new_src, encoding="utf-8")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "the round")

    res = CR.run(base, tree, rules_path=RULES_PATH)      # exactly the rung's call form
    assert res["status"] == CR.EVALUATED, res
    assert res["scope"]["kind"] == "diff", res["scope"]
    assert res["totals"]["parse_errors"] == 0, (
        f"a fixture the runner could not parse would make the assertions below a skipped "
        f"file rather than a verdict: {res['totals']}")
    assert res["totals"]["files"] == 1, (
        f"the diff-scoped runner was supposed to consider the one .py the round added, not "
        f"{res['totals']['files']}: {res['totals']}")

    hits = [h for h in res["hits"] if h["rule"] == KG_RULE]
    assert [(h["path"], h["line"]) for h in hits] == [(new_rel, new_line)], res["hits"]
    assert legacy_rel not in {h["path"] for h in res["hits"] + res["allowed"]}, (
        "a pre-existing open was reported by a diff-scoped run, which means the run read "
        f"the tree and the zero-hit half of this node proves nothing: {res['allowed']}")


# ---------------------------------------------------------------------------
# Clause 4 — the live-tree control: per rule, zero hits or exactly the allowlist
# that rule itself names. Shape-based, and it fails on a stale allowlist entry
# too, because an allowlist nobody can delete is how a rule stops meaning anything.
# ---------------------------------------------------------------------------

def _tracked_python(tree: Path) -> list[str]:
    return [p for p in _git(tree, "ls-files", "*.py").splitlines() if p]


@pytest.mark.parametrize("rule_id", SHIPPED_RULE_IDS)
def test_the_live_tree_scores_zero_or_exactly_the_named_allowlist(rule_id):
    rules = {r.id: r for r in CR.load_rules(RULES_PATH)}
    rule = rules[rule_id]
    res = CR.run("HEAD", ROOT, rules_path=RULES_PATH,
                 files=_tracked_python(ROOT), whole_tree=True)
    assert res["status"] == CR.EVALUATED, res
    assert res["totals"]["files"] > 100, "the control must actually read the tree"

    hits = [h for h in res["hits"] if h["rule"] == rule_id]
    allowed = [h for h in res["allowed"] if h["rule"] == rule_id]
    assert hits == [], (
        f"rule {rule_id} has new violators on the live tree: "
        f"{[h['path'] + ':' + str(h['line']) for h in hits]}")
    assert {h["path"] for h in allowed} == {a.path for a in rule.allow}, (
        f"rule {rule_id}: the tree's allowed sites are "
        f"{sorted({h['path'] for h in allowed})} but the rule file names "
        f"{sorted(a.path for a in rule.allow)} — a site that stopped existing "
        "means the allow entry gets deleted, not kept")


def test_the_runner_reports_its_scope_kind(tmp_path):
    """Diff-scoped is the default and the artifact says so on every record."""
    tree, base = _scratch_repo(tmp_path, {"app/x.py": "X = 1\n"})
    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["scope"] == {"kind": "diff", "base": base}, res
    whole = CR.run("HEAD", ROOT, rules_path=RULES_PATH,
                   files=["scripts/automod/code_rules.py"], whole_tree=True)
    assert whole["scope"]["kind"] == "whole_tree", whole


def test_the_guardian_rule_asks_the_guardians_interpreter(tmp_path):
    """The one thing a clean guardian record cannot be allowed to hide.

    `guardian-import-hygiene` decides "is this module stdlib" — and
    `sys.stdlib_module_names` is a property of an interpreter, not of Python. The
    gate runs on the venv (3.12, 301 names); the unit runs on `/usr/bin/python3`
    (3.14, 297 names, read here). A module in the venv's set and not the unit's is
    therefore *not* importable by the watchdog that exists for the machine already
    down, and `uu`/`telnetlib`/`crypt`/`cgi` are exactly that set on this box: a
    rule that consulted the gate's interpreter would pass the commit that broke the
    guardian. Measured, not remembered — the two sets are printed by the command in
    the runner's comment and asserted against below.
    """
    unit = CR.stdlib_names(CR.GUARDIAN_PYTHON)
    gate = frozenset(__import__("sys").stdlib_module_names)
    assert unit, "the probe returned nothing, so the rule would allow everything"
    # The two universes genuinely differ, or this whole node is decoration: on a box
    # where the gate and the unit ran the same interpreter the bug is unreachable and
    # the test should say so rather than silently pin nothing.
    assert unit != gate, (
        "the guardian's interpreter and the gate's answer the same stdlib set, so the "
        "interpreter split is not exercised on this box — do not claim it is")
    wrongly_allowed = sorted(gate - unit)
    assert "uu" in wrongly_allowed and "telnetlib" in wrongly_allowed, (
        f"the 3.12-only modules this rule used to wave through changed: "
        f"{wrongly_allowed}")

    # The shape the old matcher waved through: a 3.12-stdlib import in the unit's dir.
    tree, base = _scratch_repo(tmp_path,
                               {"agent-services/guardian/probe.py": "import uu\n"})
    res = CR.run(base, tree, rules_path=RULES_PATH)
    assert res["status"] == CR.EVALUATED, res
    hits = [h for h in res["hits"] if h["rule"] == "guardian-import-hygiene"]
    assert [h["line"] for h in hits] == [1], (
        "an import that is stdlib for the gate and not for the unit was let through — "
        "the matcher is consulting the wrong interpreter again")
    assert "uu" in hits[0]["detail"], hits[0]
    assert res["totals"]["interpreters"] == [CR.GUARDIAN_PYTHON], (
        "the record does not say which interpreter its guardian verdict used, so a "
        f"clean row is unattributable: {res['totals']['interpreters']}")

    # And a probe failure is `unevaluated`, never a clean pass: a matcher that could
    # not ask must not answer.
    def boom(_interp):
        raise CR.StdlibProbeError("no interpreter here")
    monkey = tmp_path / "monkey.py"
    monkey.write_text("")
    orig = CR.stdlib_names
    CR.stdlib_names = boom
    try:
        failed = CR.run(base, tree, rules_path=RULES_PATH)
    finally:
        CR.stdlib_names = orig
    assert failed["status"] == CR.UNEVALUATED, failed
    assert "no interpreter here" in failed["reason"], failed
    assert "hits" not in failed, "an unevaluated run must not carry a hit list"


def test_the_live_tree_control_asks_what_it_actually_covered():
    """A zero is not a clean tree: it is a scan of the files some rule's scope accepted.

    Most of the shipped rules carry a scope that admits every tracked `.py`; the
    narrow-scope ones see one directory each. The control's headline file figure is
    therefore the *union* of the rules' scopes, and a rule that saw a handful of files
    cannot be quoted as a tree-wide zero — which is what a per-rule census is for. Without
    it, a rule whose pattern had quietly stopped matching and a rule whose real violation
    list had been swept away are the same number: zero.

    The census is read from the runner's own `totals.coverage` and checked against a count
    this node computes from `rule.covers`. Two paths to one number is the point: the figure
    on the record is the one a reader of a landing event months from now has, and it has to
    agree with the scope the rule actually carries. It is printed per rule so the census is
    legible at the point of use, and every figure it shows is measured at run time — an
    absolute file count quoted in this docstring would rot inside a week, which is the
    lesson of `tests/test_automod_vet.py`.

    The assertions are shape-based per rule: `test_..._allowlist` already pins which files
    each rule may allow, so nothing here needs to pin how many files exist.
    """
    tracked = _tracked_python(ROOT)
    rules = CR.load_rules(RULES_PATH)
    res = CR.run("HEAD", ROOT, rules_path=RULES_PATH, files=tracked, whole_tree=True)
    assert res["status"] == CR.EVALUATED, res
    assert res["totals"]["files"] > 100, "the control must actually read the tree"

    covered = {r.id: sum(1 for p in tracked if r.covers(p)) for r in rules}
    record = res["totals"]["coverage"]
    assert set(record) == set(covered), (
        f"the census names {sorted(record)}, not the rules that loaded: {sorted(covered)}")
    for rule_id, n in covered.items():
        got = record[rule_id]
        assert got["of"] == len(tracked), (
            f"{rule_id}'s census denominator is {got['of']}, not the {len(tracked)} "
            "tracked .py the control offered — so the ratio would not be a tree figure")
        assert got["covered"] == n, (
            f"{rule_id}: the runner counted {got['covered']} files inside its scope, this "
            f"node counts {n} — the figure on the durable record is not the scope the rule "
            "carries")
        assert got["covered"] > 0, (
            f"{rule_id} covers no tracked file at all: its zero is meaningless")
        print(f"coverage: {rule_id} covered {got['covered']} of {got['of']} tracked .py")

    # Exactly one rule is narrow, and its census is the number of files in the directory
    # its scope names — so a scope typo that silently narrowed a broad rule, or widened a
    # narrow one, surfaces here instead of as a rule that quietly sees nothing.
    narrow = [r for r in rules if not r.covers("app/anything.py")]
    assert [r.id for r in narrow] == ["guardian-import-hygiene"], \
        {r.id: covered[r.id] for r in narrow}
    assert covered[narrow[0].id] == sum(
        1 for f in tracked if f.startswith("agent-services/guardian/")), covered
    assert res["totals"]["interpreters"] == [CR.GUARDIAN_PYTHON], (
        "the guardian rule ran without recording the interpreter it asked")
    assert res["totals"]["suppression_markers"] > 0, (
        "the live tree carries `# noqa` and `type: ignore` markers by the thousand; a "
        "control that counts none is not reading the files")


# ---------------------------------------------------------------------------
# #2810 — the naive-stamp rule earns the scripts tree, and the sites it names get
# converted. Five nodes: the grep that filed the item now comes back empty; every
# stamp left in those roots carries an explicit UTC zone; the rule's scope grew and
# its cite did not; the widening is visible as a census figure and only for that
# rule; and the whole-tree control is still zero hits over the allowed sites it names
# (five as #2810 left them, seven since #3357 made the two read-only kg-store witnesses
# visible to the rule that had to excuse them).
# ---------------------------------------------------------------------------

#: The roots the widened scope reaches. `scripts/` at large is NOT one of them: the
#: triage measured the naive sites living in these three, and a rule scoped to all
#: of `scripts/` would have to be vouched for site by site across files nobody read.
STAMP_ROOTS = ("scripts/memory", "scripts/benchmark.py", "scripts/maintenance")

#: The grep the item was filed on, as re-worded by its own triage. The wording in
#: the original clause — `now[()].*%Y%m%dT%H%M%SZ` — also spans an aware call, so a
#: perfect conversion still returned 4 hits and the clause could not be satisfied;
#: and it never matched `scripts/benchmark.py:287`, whose stamp has no `Z`.
NAIVE_STAMP_GREP = r"(dt\.)?datetime\.now\(\).*%Y%m%d"

#: A `now()` call naming the zone it is measured in. Both spellings the tree uses
#: (`datetime.now(timezone.utc)` and `dt.datetime.now(dt.timezone.utc)`) match, and
#: a bare `now()` does not.
AWARE_NOW_RE = re.compile(
    r"\bnow\(\s*(?:[A-Za-z_]\w*\.)?timezone\.(?:utc|UTC)\s*\)"
    r"|\bnow\(\s*(?:[A-Za-z_]\w*\.)?UTC\s*\)")

#: The rule's cite as shipped at dcf22067, whitespace collapsed the way `load_rules`
#: collapses it. Clause 2 asks for it UNCHANGED, so it is pinned verbatim: widening
#: the reach of a rule must not quietly rewrite the incident that justified it.
CITE_NAIVE_NOW_AT_FILING = (
    "#548's findings for round SM_20260916_022959: a `--rerender` flag took "
    "`datetime.now()` for both the artifact filename and its `captured:` line and so "
    "minted a second committed table claiming a capture time it never captured. The "
    "same class is the loaded-memory rule of 2026-09-21 ('a naive local timestamp "
    "inside a machine-facing payload is read as UTC by every later reader').")

#: The four other rules' scopes as shipped at dcf22067. Three of them are `**/*.py`
#: and so already covered every file this widening adds, which is why clause 3's
#: "every other rule's denominator is unchanged" is not checkable as a number —
#: those numbers cannot move. It is checkable as the one thing that would move them:
#: a scope edit. Pinning the four scope lists is the falsifiable form of that claim.
OTHER_RULE_SCOPES_AT_FILING = {
    "guardian-import-hygiene": ("agent-services/guardian/*.py",),
    "opens-kg-store-outside-kg-store": ("**/*.py",),
    "broad-except-swallow-same-line": ("**/*.py",),
    "exports-lloyd-data-to-process-environ": ("**/*.py",),
}

#: The naive rule's own scope before #2810. Rebuilt here as a `Rule` so the census
#: delta is measured against the scope the rule actually shipped with, not against a
#: number quoted from the item — 354 of 1377 was true at filing and the tree grows.
NAIVE_NOW_SCOPE_AT_FILING = ("app/**", "agent_mcp/**", "workers/**", "server.py",
                             "scripts/automod/**", "agent-services/**/*.py")


def _stamp_grep(tree: Path) -> list[str]:
    """The item's own acceptance grep, as lines of `path:line:source`."""
    proc = subprocess.run(["git", "-C", str(tree), "grep", "-nE", NAIVE_STAMP_GREP,
                           "--", *STAMP_ROOTS], capture_output=True, text=True)
    # git grep exits 1 when nothing matches, which is the pass condition here, and
    # 2 on a real failure (a bad pattern, an unreadable tree) — those must not read
    # as "clean", the same trap `code_rules.run` refuses with `unevaluated`.
    assert proc.returncode in (0, 1), (
        f"the acceptance grep could not run: rc={proc.returncode} "
        f"{proc.stderr.strip()[:200]}")
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def test_the_naive_stamp_grep_that_filed_the_item_now_returns_nothing():
    """Clause 1: the 12 sites are gone from the tree the grep reads."""
    sites = _stamp_grep(ROOT)
    assert sites == [], (
        "these lines still build an artifact name from a no-arg `datetime.now()`, "
        "which is local wall clock and gets read as UTC by every later reader:\n"
        + "\n".join(sites))


def test_every_name_stamp_in_the_widened_roots_names_its_zone():
    """Clause 1's positive half: converted, not deleted, and no comment exemption.

    The grep above goes green three ways — the sites converted, the sites deleted, or
    the pattern broken — so this one reads the same roots line by line and asks the
    other question: a line that builds a `%Y%m%d` stamp out of a `now()` call has to
    say which zone that clock is in. It is the difference between "the defect is
    absent" and "the code says what it does", and only the second one survives the
    next person editing the line.
    """
    offenders: list[str] = []
    for rel in _tracked_python(ROOT):
        if not any(rel == r or rel.startswith(r.rstrip("/") + "/") for r in STAMP_ROOTS):
            continue
        if not rel.endswith(".py"):
            continue
        for i, line in enumerate((ROOT / rel).read_text(encoding="utf-8",
                                                        errors="replace").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "%Y%m%d" not in line or "now(" not in line:
                continue
            if not AWARE_NOW_RE.search(line):
                offenders.append(f"{rel}:{i}: {stripped}")
    assert offenders == [], (
        "a stamp-bearing `now()` call that does not name its zone: "
        + "\n".join(offenders))


def test_the_rule_gained_the_three_script_roots_and_its_cite_did_not_move():
    """Clause 2: scope widened, still exactly 5 rules, cite byte-identical."""
    rules = CR.load_rules(RULES_PATH)
    assert len(rules) == 5, (
        f"the rule file loads {len(rules)} rules; #2810 was to widen a scope, not add "
        f"a rule: {[r.id for r in rules]}")
    rule = {r.id: r for r in rules}["naive-now-in-artifact-name"]
    assert rule.cite == CITE_NAIVE_NOW_AT_FILING, (
        "the cite moved. The clause asks for the scope to widen with the incident that "
        f"justified the rule left alone; got:\n{rule.cite}")
    for root in ("scripts/memory/**", "scripts/benchmark.py", "scripts/maintenance/**"):
        assert root in rule.scope, (
            f"{root} is not in the rule's scope line; scope is {list(rule.scope)}")
    for probe in ("scripts/memory/kg_rebuild.py", "scripts/benchmark.py",
                  "scripts/maintenance/qmd_index_maintenance.py"):
        assert rule.covers(probe), f"the widened scope still does not cover {probe}"


def test_the_widening_shows_up_as_this_rules_census_and_nobody_elses():
    """Clause 3: the coverage figure moves for one rule and its scope is the reason.

    Printed, not just asserted, because the point of the census is that a reader of a
    landing event months from now can see what a zero was measured over. The baseline
    is the pre-widening scope rebuilt as a `Rule` and run through the same `covers()`,
    so the delta is arithmetic over one mechanism rather than a comparison with a
    number the item happened to record on the day it was filed.
    """
    tracked = _tracked_python(ROOT)
    rule = {r.id: r for r in CR.load_rules(RULES_PATH)}["naive-now-in-artifact-name"]
    before = CR.Rule(id="naive-now-in-artifact-name", matcher="naive_now_in_artifact_name",
                     summary="", cite="#2810", scope=NAIVE_NOW_SCOPE_AT_FILING,
                     exclude=())
    now_cov = sorted(f for f in tracked if rule.covers(f))
    was_cov = sorted(f for f in tracked if before.covers(f))
    gained = sorted(set(now_cov) - set(was_cov))
    print(f"coverage: naive-now-in-artifact-name covers {len(now_cov)} of {len(tracked)} "
          f"tracked .py; at the scope shipped at filing it covered {len(was_cov)} "
          f"— this widening added {len(gained)}")
    assert len(now_cov) > len(was_cov), (
        f"the rule covers {len(now_cov)} files, no more than the {len(was_cov)} the "
        "pre-widening scope covered — the widening is not in the rule file")
    assert gained, "the widening added no file at all"
    for probe in ("scripts/memory/kg_rebuild.py", "scripts/benchmark.py",
                  "scripts/maintenance/qmd_index_maintenance.py"):
        assert probe in gained, (
            f"{probe} was already covered before the widening, so the scope line did "
            f"not gain the root that names it: gained={gained[:12]}")

    others = {r.id: r.scope for r in CR.load_rules(RULES_PATH)
              if r.id != "naive-now-in-artifact-name"}
    assert others == OTHER_RULE_SCOPES_AT_FILING, (
        "another rule's scope changed in the same breath as this one, so its census "
        f"figure is no longer the one its own clause was graded against: {others}")

    res = CR.run("HEAD", ROOT, rules_path=RULES_PATH, files=tracked, whole_tree=True)
    assert res["status"] == CR.EVALUATED, res
    recorded = res["totals"]["coverage"]["naive-now-in-artifact-name"]
    assert recorded["covered"] == len(now_cov), (
        f"the durable census says {recorded}, this node counted {len(now_cov)} — the "
        "figure a landing event carries is not the scope the rule carries")


def test_the_whole_tree_control_is_still_zero_hits_over_exactly_seven_allowed_sites():
    """Clause 4 as #3357 moved it: HEAD reports nothing, and now names what it excuses.

    The seven sites are the whole allow population of the five rules — two
    `broad-except-swallow-same-line` drains in `app/sessions_io.py`, the test suite's
    scratch-root exports, and the three kg-store opens: the scratch-fixture open in
    `tests/test_data_home.py`, plus the two read-only `?mode=ro` `count_kg_rows`
    witnesses in `agent-services/guardian/guardian.py` and
    `scripts/automod/promote.py`. Those two are new to the count because #3357 widened
    the matcher to follow a store path one hop: before that they were not excused, they
    were unseen, which is a different thing and a worse one. The line numbers move
    whenever those files are edited, so what is pinned is the count and the paths, which
    is the part that carries the claim "no new violator slipped in behind the widening".
    """
    res = CR.run("HEAD", ROOT, rules_path=RULES_PATH,
                 files=_tracked_python(ROOT), whole_tree=True)
    assert res["status"] == CR.EVALUATED, res
    assert res["hits"] == [], (
        "new violators on the live tree under the widened scope: "
        + "; ".join(f"{h['rule']} at {h['path']}:{h['line']}" for h in res["hits"]))
    sites = sorted((h["rule"], h["path"]) for h in res["allowed"])
    print("allowed sites: " + "; ".join(f"{r} @ {p}" for r, p in sites))
    assert len(sites) == 7, f"the control reports {len(sites)} allowed sites, not 7: {sites}"
    assert sorted(p for _, p in sites) == [
        "agent-services/guardian/guardian.py", "app/sessions_io.py",
        "app/sessions_io.py", "scripts/automod/promote.py", "tests/conftest.py",
        "tests/test_data_home.py", "tests/test_youtube_artifact_phase.py"], sites
    for rid, n in sorted(res["counts"].items()):
        print(f"hits for {rid}: {n}")
    assert res["totals"]["parse_errors"] == 0, res["totals"]


# ---------------------------------------------------------------------------
# #2809 — the soak #2574 owed is recorded in the rule file, not in a closed item.
#
# Every clause here is a claim about prose in the comment block above `[meta]`, so
# each node answers it with a DIFFERENT mechanism than the prose itself: `git` for
# whether the landing commit the header cites really added this file and is an
# ancestor of HEAD; `bash`+`jq` for whether the recompute block the header names
# executes and can report a nonzero flag count; the `gate.py` source for whether the
# flip bound in the header adds to the published one instead of competing with it; the
# loader for whether the ids, matchers and cap the header describes are the ones the
# file ships. Scope lines are pinned by #2810's three nodes in this file.
#
# What is deliberately NOT asserted anywhere below is the VALUE of a live soak figure.
# The gate-row and landing counts in the header were true when written and age by the
# day (~51 gate rows a day as of 2026-10-19). WHO OWNS RAISING THEM IS SETTLED, and the
# two sentences of it below are quoted verbatim from the header's soak block, where
# #3263 wrote them into the rule file:
#
#   The owed-check pass alone decides that this soak record has gone stale, and it
#   alone raises the figures, by publishing the fresh measurement in its decision.
#   A round changes these comment lines only inside an organic edit it is already
#   making for another reason.
#
# `test_the_ownership_ruling_is_the_same_two_sentences_in_both_files` reads this file's
# own source and the rule file's header and requires the pair in BOTH, so the ruling
# cannot move in one place and be contradicted in the other — which is exactly the state
# #2992 left behind, this block ending on an open question its own owed-check pass had
# already answered. Neither route makes a node read the ledger: #2992's
# nodes pin FLOORS (see below), because a node that demanded today's flags still be 0
# would enforce this instrument through the test rung — and a rule that fails a round is
# precisely the flip #679 ruled has to be a human decision, so the suite must not make
# it.
# ---------------------------------------------------------------------------

#: The landing that put these rules on the gate, as the header cites it. Clause 1's
#: acceptance check is `git grep -n 4592296e scripts/automod/code_rules.toml`, which
#: is worth nothing unless the sha names the commit that actually added the file.
RULES_LANDED_AT = "4592296e"

#: The five ids and matchers as shipped when #2809 recorded the soak. A legitimate
#: rule change updates this list on purpose, exactly like #2810's scope pins.
RULE_IDS_AT_RECORDING = [
    "guardian-import-hygiene",
    "opens-kg-store-outside-kg-store",
    "broad-except-swallow-same-line",
    "naive-now-in-artifact-name",
    "exports-lloyd-data-to-process-environ",
]
RULE_MATCHERS_AT_RECORDING = [
    ("guardian-import-hygiene", "guardian_nonstdlib_import"),
    ("opens-kg-store-outside-kg-store", "opens_kg_store"),
    ("broad-except-swallow-same-line", "broad_except_swallow_same_line"),
    ("naive-now-in-artifact-name", "naive_now_in_artifact_name"),
    ("exports-lloyd-data-to-process-environ", "process_environ_lloyd_data"),
]

#: The two machine-readable figures in the header, parsed rather than counted by eye.
RECORDED_FIGURE_RE = {
    "SOAK_GATE_ROWS_AT_RECORDING": re.compile(
        r"SOAK_GATE_ROWS_AT_RECORDING\s*=\s*(\d+)"),
    "LANDED_PY_COMMITS_AT_RECORDING": re.compile(
        r"LANDED_PY_COMMITS_AT_RECORDING\s*=\s*(\d+)"),
}


def _header_lines() -> list[str]:
    """The comment block above the first `[table]` header, comment markers intact.

    Everything above `[meta]` is prose: an edit confined to that region cannot change
    a rule, a matcher, a scope or the cap, which is what #2809's comment-only clause
    turns on.
    """
    lines = RULES_PATH.read_text(encoding="utf-8").splitlines()
    head: list[str] = []
    for ln in lines:
        if ln.lstrip().startswith("["):
            return head
        head.append(ln)
    raise AssertionError("the rule file has no [table] header, so it has no rule section")


def _header_prose() -> str:
    """That block as one whitespace-normalised string, `#NNNN` refs kept.

    Joined, because a claim of record is one sentence and a prose assertion must not
    hinge on where a rewrap happens to fall.
    """
    return " ".join(ln.lstrip("#").strip() for ln in _header_lines())


def _recorded_figure(header: str, key: str) -> int:
    m = RECORDED_FIGURE_RE[key].search(header)
    assert m, f"the header carries no machine-readable {key}"
    return int(m.group(1))


def _fixture_row(created_at: str, counts: dict[str, int],
                 covered: dict[str, int],
                 round_id: str = "SM_FIXTURE") -> str:
    """A `gate` ledger row shaped like the ones `rung_code_rules` writes.

    `round_id` defaults to one round because the nodes that planted this row before #3263
    ask nothing of it; the window node needs a spread of them, because the arm that counts
    distinct rounds is one of the four the bound has to cover.
    """
    return json.dumps({
        "event": "gate", "rung": "code_rules", "round_id": round_id,
        "created_at": created_at,
        "code_rules": {
            "status": "evaluated", "counts": counts, "labels": [],
            "totals": {"files": 2,
                       "coverage": {rid: {"covered": n, "of": 2}
                                    for rid, n in covered.items()}}}})


def _recompute_body() -> str:
    """The header's RECOMPUTE block with its comment markers stripped: what bash runs.

    Cut exactly the way `_run_recompute_block` cuts it, so a structural claim about the
    block — that it binds a variable, that a command filters on it — is a claim about the
    script the figures come out of, not about a commented copy of it.
    """
    lines = _header_lines()
    starts = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("# RECOMPUTE-BEGIN")]
    ends = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("# RECOMPUTE-END")]
    assert starts and ends and ends[0] > starts[0], (
        "the soak record names no RECOMPUTE-BEGIN/END block, so its figures cannot be "
        "recomputed by a reader holding only this file")
    return "\n".join(ln[1:] if ln.startswith("#") else ln
                     for ln in lines[starts[0] + 1:ends[0]])


def _run_recompute_block(tmp_path: Path, ledger: Path) -> str:
    """Extract the header's RECOMPUTE block and run it against `ledger`."""
    body = _recompute_body()
    script = tmp_path / "recompute.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body + "\n",
                      encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin",
           "HOME": str(tmp_path),
           "CODE_RULES_LEDGER": str(ledger),
           "CODE_RULES_SOAK_SINCE": "2026-10-13T03:07:36Z"}
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path))
    assert proc.returncode == 0, (
        f"the recompute block the header names does not run (rc={proc.returncode}):\n"
        f"{body}\n--- stderr ---\n{proc.stderr.strip()[:400]}")
    return proc.stdout


def test_the_header_names_the_soak_and_the_cited_landing_commit_is_the_one_that_added_the_file():
    """Clause 1: 0 flags, a denominator, the command, the span, the flag count.

    Three mechanisms, because the prose alone is not evidence: the acceptance grep has
    to exit 0, the sha it finds has to be the commit that ADDED this file (a soak
    "since <sha>" is a sentence about that sha), and the landing figure the header
    claims is checked as a LOWER bound — landings accumulate, so a claim above the live
    count was never measured, while one below it has merely aged.
    """
    header = _header_prose()
    grep = subprocess.run(["git", "-C", str(ROOT), "grep", "-n", RULES_LANDED_AT,
                           "scripts/automod/code_rules.toml"],
                          capture_output=True, text=True)
    assert grep.returncode == 0, (
        "the acceptance check `git grep -n 4592296e scripts/automod/code_rules.toml` "
        "found no landing commit: the header states a soak without naming what it is a "
        "soak of")
    assert f"{RULES_LANDED_AT} is the commit that put these rules on the gate" in header

    added = _git(ROOT, "log", "--diff-filter=A", "--format=%H", "--",
                 "scripts/automod/code_rules.toml").split()
    assert len(added) == 1, f"expected one commit adding the rule file, got {added}"
    assert added[0].startswith(RULES_LANDED_AT), (
        f"the header cites {RULES_LANDED_AT} as the landing, but {added[0]} is the "
        "commit that added this file — the soak's denominator would start at a commit "
        "that has nothing to do with these rules")
    anc = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                          RULES_LANDED_AT, "HEAD"], capture_output=True, text=True)
    assert anc.returncode == 0, (
        f"{RULES_LANDED_AT} is not an ancestor of HEAD: {anc.stderr.strip()[:160]}")

    assert "0 flags" in header, "the header states no flag count"
    assert "promotions.jsonl" in header and "jq" in header, (
        "the denominator names no command that recomputes it, so the figure is a "
        "caption and not a measurement")
    assert "git log --oneline 4592296e..HEAD" in header, (
        "the landing-count denominator names no command either")
    assert "`status: evaluated`" in header and "unevaluated" in header, (
        "the denominator has to say what kind of row it counts and that none came back "
        "unevaluated — otherwise a zero only means the instrument did not run")

    rows = _recorded_figure(header, "SOAK_GATE_ROWS_AT_RECORDING")
    assert rows >= 20, (
        f"the header records a soak of {rows} gate rows, under the 20-landing bound a "
        "flip waits on — the record is advertising a verdict it cannot support")
    landed = _recorded_figure(header, "LANDED_PY_COMMITS_AT_RECORDING")
    live = len(_git(ROOT, "log", "--oneline", f"{RULES_LANDED_AT}..HEAD", "--",
                    "*.py").splitlines())
    assert live >= landed, (
        f"the header claims {landed} promoted .py landings since {RULES_LANDED_AT}; the "
        f"command it names returns {live} now. Landings only accumulate, so the claim "
        "was never measured — the owed-check pass may raise the figure, never invent it")
    print(f"landings since {RULES_LANDED_AT}: recorded {landed}, measured now {live}")

    span = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) \.\. "
                     r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", header)
    assert span, "the header names no date span for the soak"
    start, stop = span.group(1), span.group(2)
    assert start < stop, span.groups()
    assert start >= "2026-10-13", (
        f"the soak's first row is stamped {start}, before the landing "
        f"{RULES_LANDED_AT} (2026-10-13T03:07:36Z UTC) could have run a rule")


def test_the_recompute_block_the_header_names_runs_and_reports_a_nonzero_flag_count(tmp_path):
    """Clause 1's real half: the named command is a command, not a caption.

    The figures in the header age, so nothing here asserts them. What is asserted is
    the thing that does not age — that the block between `RECOMPUTE-BEGIN` and
    `RECOMPUTE-END` executes, that it excludes rows before `SINCE`, that it can report
    a flag when one exists (a program that could only ever print `flags 0` would keep
    this record looking true forever), and that its coverage arm counts rows and files.
    The ledger is a fixture in `tmp_path`, never the live one: reading live flags into
    the suite would make an observe-only rule blocking by way of the test rung.
    """
    planted = [
        _fixture_row("2026-10-12T00:00:00Z", {}, {"wide": 1}),   # before SINCE
        json.dumps({"event": "promoted", "round_id": "SM_X",
                    "created_at": "2026-10-14T00:00:00Z"}),      # not a gate row
        _fixture_row("2026-10-14T00:00:00Z", {}, {"wide": 2, "narrow": 0}),
        _fixture_row("2026-10-15T00:00:00Z", {"wide": 1}, {"wide": 1, "narrow": 0}),
    ]
    flagged = tmp_path / "flagged.jsonl"
    flagged.write_text("\n".join(planted) + "\n", encoding="utf-8")
    out = _run_recompute_block(tmp_path, flagged)
    print(out)
    head = out.splitlines()[0]
    assert "rows 2" in head, (
        f"the two in-window `gate` rows are not what the block counted: {head!r} — a "
        "figure from this command is only trustworthy if the command scopes it")
    assert "flags 1" in head, (
        f"the block cannot report a nonzero flag count: {head!r}. A recompute that can "
        "only ever print 0 is not a check, it is a caption for the answer already "
        "recorded")
    assert "2026-10-14T00:00:00Z .. 2026-10-15T00:00:00Z" in head, (
        f"the span the block reports is not its own rows': {head!r}")
    assert "wide: 2/2 rows saw >=1 file, 3 file-reads" in out, (
        f"the coverage arm does not count rows-and-files as the header's table claims "
        f"it does:\n{out}")
    assert "narrow: 0/2 rows saw >=1 file, 0 file-reads" in out, (
        f"a rule that covered nothing is reported as if it were indistinguishable from "
        f"one that covered everything:\n{out}")

    clean = tmp_path / "clean.jsonl"
    clean.write_text("\n".join(planted[:3]) + "\n", encoding="utf-8")
    out_clean = _run_recompute_block(tmp_path, clean)
    assert "flags 0" in out_clean.splitlines()[0], (
        f"with no flagged row planted the block still reports a flag: {out_clean!r}")


def test_the_header_records_that_nothing_flipped_and_679_is_the_precedent_for_that():
    """Clause 2: no rule flips, on #679's zero-finding-soak close.

    Quoting the precedent is the point: #679 was ruled, delegated by a human, over a
    far bigger zero (790 `vet` rows, 400 promoted landing events) and came out
    observe-only. A header that said "nothing flips" without it would be a ruling with
    no case behind it, and the next reader could not tell it from a default.
    """
    header = _header_prose()
    low = header.lower()
    assert "no rule has flipped to blocking" in low, (
        "the header records a soak without recording the flip decision it supports")
    assert "#679" in header, "the no-flip ruling names no precedent"
    assert "flip nothing to blocking" in header, (
        "the precedent's own ruling is not quoted, so the reader cannot compare the "
        "two soaks — only the conclusion is in the file")
    assert "not the defect class here" in header, (
        "#679's zero-finding-soak close is not quoted: the phrase that settled a "
        "comparable soak is the thing a future reader needs")
    assert "790" in header and "400 promoted landing" in header, (
        "the comparison's own denominators are missing, so this soak's "
        f"{_recorded_figure(header, 'SOAK_GATE_ROWS_AT_RECORDING')} rows cannot be read "
        "against the one that ruled first")
    assert "nothing flips" in low, (
        "the header quotes precedent but never states this file's own conclusion")


def test_the_flip_bound_in_the_header_adds_to_the_one_gate_py_publishes():
    """Clause 3: one published bound, two conditions, no competing number.

    `rung_code_rules` already states the >=20-landings condition in the tree, and this
    round touches only the `.toml`, so the header has to ADD to that bound rather than
    restate it — two files carrying different flip thresholds is how a threshold stops
    meaning anything. The check therefore runs both ways: gate.py must still say what
    the header cites, and every digit-before-landings figure in the header must be 20.
    """
    gate = (ROOT / "scripts" / "automod" / "gate.py").read_text(encoding="utf-8")
    assert "at least 20 landings" in gate, (
        "rung_code_rules no longer publishes the >=20-landings bound the header cites, "
        "so condition (a) points at a number that moved — move it in the same change, "
        "not by editing the sentence here")
    header = _header_prose()
    assert "rung_code_rules" in header, (
        "the header names no published bound to add itself to")
    assert "at least 20 landings" in header, (
        "condition (a) does not quote gate.py's bound, so a reader cannot tell which "
        "number is the published one")
    assert "at least 1 confirmed real catch" in header, (
        "condition (b) is missing: >=20 landings with zero catches is #679's outcome, "
        "not a licence to flip")
    assert "in live diff" in header or "confirmed real catch in a live diff" in header, (
        "condition (b) must be about a catch in a real diff, not in a fixture")
    assert "allow list" in header and "SAME diff" in header, (
        "a flip that ships without the allow list the rule will need refuses every "
        "round until the exceptions land — including the round fixing them")
    nums = set(re.findall(r"(\d+)\s?[- ]?\blandings\b", header))
    assert nums == {"20"}, (
        f"the header publishes flip bounds of its own: {sorted(nums)}, beside gate.py's "
        "20 — a threshold in two places is a threshold nobody can satisfy")


@pytest.mark.parametrize("rule_id", SHIPPED_RULE_IDS)
def test_the_header_carries_a_coverage_figure_for_every_rule_the_file_ships(rule_id):
    """Clause 4: the zero is priced per rule, and a new rule cannot dodge that.

    Parametrized over the FILE's ids, so adding a rule without recording what it
    covered is a red node here rather than a rule whose zero nobody can price — the
    same shape as the fixture-coverage test above, on the other half of the record.
    """
    lines = [ln.lstrip("#").strip() for ln in _header_lines() if rule_id in ln]
    assert lines, f"the header never names {rule_id}, so its zero is unattributed"
    cov = [ln for ln in lines if re.search(r"\d+/\d+ rows", ln)]
    assert cov, (
        f"{rule_id} appears in the soak record with no `N/M rows` coverage figure: "
        f"{lines}. A rule that covered two rows of two hundred has a zero that means "
        "something else entirely, and the table is where that difference lives")
    m = re.search(r"(\d+)/(\d+) rows", cov[0])
    assert m, cov[0]
    covered, total = int(m.group(1)), int(m.group(2))
    assert total > 0 and 0 <= covered <= total, cov[0]
    assert "file-reads" in cov[0], (
        f"{rule_id}'s coverage line counts rows but not files, so a rule that saw one "
        f"file in 200 rows and one that saw 200 files each look identical: {cov[0]}")
    print(f"header coverage: {rule_id} {covered}/{total} rows")


def test_the_header_prices_the_zero_and_the_rules_it_prices_are_the_rules_that_ship():
    """Clause 4's other half: the coverage table's id set IS the shipped rule set.

    A record that prices four of five rules reads exactly as complete as one that
    prices five, which is the gap this node closes; and the sentence that makes the
    table load-bearing has to be in the file, or the table is decoration beside a
    headline number a reader will quote on its own.
    """
    header = _header_prose()
    assert "a rule's zero is only as strong as its coverage" in header.lower(), (
        "the header states a flag count without pricing it — the sentence the whole "
        "coverage table exists to carry")
    table_ids = set()
    for ln in _header_lines():
        m = re.match(r"^([a-z0-9][a-z0-9-]+)\s+\d+/\d+ rows", ln.lstrip("# ").strip())
        if m:
            table_ids.add(m.group(1))
    assert table_ids == set(SHIPPED_RULE_IDS), (
        f"the coverage table prices {sorted(table_ids)} but the file ships "
        f"{sorted(SHIPPED_RULE_IDS)} — an unpriced rule's zero is the one that gets "
        "quoted as if it were a soak")
    narrow = [rid for rid in ("guardian-import-hygiene", "naive-now-in-artifact-name")
              if any(rid in ln for ln in _header_lines()
                     if re.search(r"\d+/\d+ rows", ln))]
    assert narrow == ["guardian-import-hygiene", "naive-now-in-artifact-name"], (
        "the two rules whose coverage is the weak point of this record are not the two "
        "the record names")


def test_the_header_rejects_669_stale_close_with_the_reason_that_rejects_it():
    """Clause 5: #669's stale-close is refused, with its reason, so it is not reopened.

    #669 is the other zero-finding soak in this area, and it ended `hand sweep
    2026-09-24 (stale)` after two automod triages had CONFIRMED it — a sweep that
    overrode a settled ruling and left no measurement behind it (the owed-check that
    went looking for its artifact found `scripts/code_quality_metrics.py` absent,
    because nothing had ever been measured). The reason is the deliverable: a bare
    "not #669" would be re-litigated by the next person who saw the parallel.
    """
    header = _header_prose()
    assert "#669" in header, "the header records no answer to the other closure route"
    assert "hand sweep 2026-09-24 (stale)" in header, (
        "the closure being refused is not named, so a reader cannot check the refusal")
    assert "two automod triages had confirmed" in header, (
        "the reason has to name what the sweep overrode — a stale close of an "
        "unconfirmed item would be defensible, and this one is not")
    assert "erases a settled ruling" in header, (
        "the consequence is missing: the objection is a ruling erased, not a wording")
    assert "code_quality_metrics.py" in header, (
        "the refusal should cite the artifact that never existed, which is the proof "
        "that this closure route measures nothing")
    assert "stale-close is not a" in header, (
        "the header never says what it is refusing to become")
    assert "confirmed real catch" in header, (
        "the record must also say what WOULD retire it, or it reads as permanent by "
        "default rather than by ruling")


def test_the_soak_record_is_a_comment_edit_and_the_rule_surface_it_describes_is_unchanged():
    """Clause 6 of #2809's contract: a record about rules that changes none of them.

    Three checks, none of them a count of how many lines the block has. (1) Everything
    above `[meta]` is a comment, so the region this round edited cannot carry a rule.
    (2) The ids and matchers still load as the five that shipped — scope lines are
    pinned by #2810's three nodes, and the whole-tree control by the two above. (3) The
    cap the header quotes is the loader's cap, so the prose and the code cannot drift
    apart in a change that claims to be only prose.
    """
    lines = _header_lines()
    non_comment = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    assert non_comment == [], (
        f"non-comment lines above the first table header: {non_comment} — the soak "
        "record is specified as a comment-only edit precisely so it cannot add, widen "
        "or soften a rule while it reports on one")

    rules = CR.load_rules(RULES_PATH)
    assert [r.id for r in rules] == RULE_IDS_AT_RECORDING, (
        f"the rule ids moved: {[r.id for r in rules]} vs "
        f"{RULE_IDS_AT_RECORDING}. A round that records a soak ships no rule; if a "
        "rule genuinely changed, that is a different item and this list moves with it")
    assert [(r.id, r.matcher) for r in rules] == RULE_MATCHERS_AT_RECORDING, (
        f"a matcher was swapped under cover of a comment edit: "
        f"{[(r.id, r.matcher) for r in rules]}")
    assert len(rules) <= CR.MAX_RULES, f"{len(rules)} rules, cap is {CR.MAX_RULES}"

    header = _header_prose()
    quoted_cap = re.search(r"`MAX_RULES` in code_rules\.py is (\d+)", header)
    assert quoted_cap, (
        "the header no longer states the cap in a form this node can read, so the "
        "prose and the loader stopped being checkable against each other")
    assert int(quoted_cap.group(1)) == CR.MAX_RULES, (
        f"the header says the cap is {quoted_cap.group(1)} and the loader says "
        f"{CR.MAX_RULES}")


# ---------------------------------------------------------------------------
# #2992 — the soak figures are re-measured, and the pins are floors, not values.
#
# #2809 wrote the record; #2992 refreshed it. What the refresh must not do is turn a
# figure back into an equality: rows accumulate (~51 gate rows a day as of 2026-10-19),
# so a node that demanded the header match TODAY's ledger would be red by tomorrow and
# green for exactly one day — and reading live flags into the suite is the flip #679
# ruled a human has to make. So every figure node below pins a FLOOR: the value
# measured at 2026-10-19T00:35Z, which the header can only ever move up, plus internal
# consistency the prose itself cannot fake (a measurement dated before the last row it
# counts is not a measurement, and a table whose denominator differs from the headline
# denominator is two different soaks).
#
# The one node that leaves the file entirely reads `code_rules_census.py`'s own source,
# because clause 4's claim is about what that program denominates over.
# ---------------------------------------------------------------------------

#: Measured 2026-10-19T00:35Z by the header's own commands, from #2992's triage. Floors
#: because rows, rounds, landings and file-reads all accumulate over a fixed window.
FLOOR_GATE_ROWS = 377
FLOOR_DISTINCT_ROUNDS = 246
FLOOR_LANDED_PY_COMMITS = 200
FLOOR_SPAN_END = "2026-10-19T00:30:09Z"
#: Per rule: (rows that saw >=1 file, file-reads), same instant. The three `**/*.py`
#: rules see every row; the two narrow ones are the reason the zero is priced at all.
FLOOR_COVERAGE = {
    "opens-kg-store-outside-kg-store": (376, 1167),
    "broad-except-swallow-same-line": (376, 1167),
    "exports-lloyd-data-to-process-environ": (376, 1167),
    "naive-now-in-artifact-name": (155, 245),
    "guardian-import-hygiene": (6, 6),
}
#: #2810 widened `naive-now-in-artifact-name`'s scope in 1168247e, whose commit time is
#: 2026-10-16T09:03:23Z UTC. Only rows from then on were gated against the wider scope.
SCOPE_WIDENED_AT = "2026-10-16T09:03:23Z"
FLOOR_ROWS_ON_WIDER_SCOPE = 157

#: The shipped scopes as #2810 left them and #2992's comment edits found them: one
#: guardian-scoped rule, three over every .py, and #2810's nine-entry widened scope.
RULE_SCOPES_AT_RECORDING = [
    ("agent-services/guardian/*.py",),
    ("**/*.py",),
    ("**/*.py",),
    ("app/**", "agent_mcp/**", "workers/**", "server.py", "scripts/automod/**",
     "agent-services/**/*.py", "scripts/memory/**", "scripts/benchmark.py",
     "scripts/maintenance/**"),
    ("**/*.py",),
]

#: The last commit that changed anything below the header block. A comment-only edit
#: leaves the non-comment bytes exactly where this sha left them; a rule change moves
#: this anchor on purpose, in the same shape as `RULE_IDS_AT_RECORDING` above.
#:
#: #3357 moved it, and moved it twice. Once for the rule change itself (the cite rewrite
#: and the two new `allow` entries are non-comment bytes), and once again when the round
#: was resumed: `automod_start(from_branch=…)` rebases the branch and deletes the old one,
#: which rewrites every sha in it and left the pin naming a commit no ref can reach. An
#: unreachable sha is not a failing test today — `git show` still resolves a loose object —
#: it is a test that goes red on whoever runs `git gc` first, which is why this one now
#: names a commit on the branch that is about to be `main`. The mechanism cannot be made
#: sha-proof from inside the change: the sha of the commit that changes the rule file does
#: not exist while that commit is still being written, so the anchor is always set one
#: commit late (#2992's `c06cd68e` did the same thing).
NON_COMMENT_SURFACE_ANCHOR = "b136d97a"


def _non_comment_lines(text: str) -> list[str]:
    """`grep -v '^#'`, byte for byte: even blank lines survive it."""
    return [ln for ln in text.splitlines() if not ln.startswith("#")]


def _table_rows() -> dict[str, tuple[int, int, int]]:
    """Every coverage table line as `rule -> (rows covered, denominator, file-reads)`."""
    out: dict[str, tuple[int, int, int]] = {}
    for ln in _header_lines():
        m = re.match(r"^([a-z0-9][a-z0-9-]+)\s+(\d+)/(\d+) rows saw >=1 file,\s*(\d+) file-reads",
                     ln.lstrip("# ").strip())
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return out


def test_the_refreshed_soak_figures_carry_the_2026_10_19_floor():
    """Clause 1: every soak figure is at or above the fresh measurement, and is dated.

    Four figures, one node, because they are one measurement: rows, distinct rounds, the
    `AS MEASURED ON` date and the span's end all come out of the same run of the
    RECOMPUTE block, so a header that moved three of them and left one behind is a
    header that did not re-measure. Two of the four are checked against each other
    rather than against a number — a record "measured on 2026-10-16" whose last row is
    stamped 2026-10-19 is arithmetically impossible, and that check still works in a
    year when no floor means anything.
    """
    header = _header_prose()
    rows = _recorded_figure(header, "SOAK_GATE_ROWS_AT_RECORDING")
    assert rows >= FLOOR_GATE_ROWS, (
        f"the header records {rows} gate rows; the same command returned "
        f"{FLOOR_GATE_ROWS} at 2026-10-19T00:35Z and rows only accumulate. A stale-low "
        "denominator is invisible to CI by design (this file pins floors, never "
        "values), which is how 224 survived three days")

    m = re.search(r"(\d+) `?gate`? rows from (\d+) distinct rounds", header)
    assert m, (
        "the header's row count and its distinct-round count are no longer one phrase, "
        "so the prose denominator cannot be read next to the rounds it spans")
    prose_rows, rounds = int(m.group(1)), int(m.group(2))
    assert prose_rows == rows, (
        f"the prose says {prose_rows} rows and the machine-readable denominator says "
        f"{rows} — two denominators in one record means one of them is stale")
    assert rounds >= FLOOR_DISTINCT_ROUNDS, (
        f"the header spans {rounds} distinct rounds against a floor of "
        f"{FLOOR_DISTINCT_ROUNDS}; rounds only accumulate over a fixed window")

    landed = _recorded_figure(header, "LANDED_PY_COMMITS_AT_RECORDING")
    assert landed >= FLOOR_LANDED_PY_COMMITS, (
        f"the header claims {landed} promoted .py commits since {RULES_LANDED_AT} "
        "against a floor of " + str(FLOOR_LANDED_PY_COMMITS) + "; the command it names "
        "(`git log --oneline 4592296e..HEAD -- '*.py'`) returned 200 at 2026-10-19T00:35Z")

    stamped = re.search(r"AS MEASURED ON (\d{4}-\d{2}-\d{2})", header)
    assert stamped, "the header names no measurement date, so its figures cannot age visibly"
    span = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) \.\. "
                     r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", header)
    assert span, "the header names no date span for the soak"
    stop = span.group(2)
    assert stop >= FLOOR_SPAN_END, (
        f"the soak's last row is stamped {stop}, older than the {FLOOR_SPAN_END} read at "
        "2026-10-19T00:35Z — a span that stops short is an unrefreshed window")
    assert stamped.group(1) >= stop[:10], (
        f"'AS MEASURED ON {stamped.group(1)}' cannot describe a window whose last row is "
        f"{stop}: the date has to be the day the block was re-run, not the day the "
        "record was first written")
    print(f"soak floor ok: rows {rows} (>= {FLOOR_GATE_ROWS}), rounds {rounds}, "
          f"landed {landed}, span ends {stop}, measured {stamped.group(1)}")


def test_the_coverage_table_and_the_sentences_naming_it_share_one_denominator():
    """Clause 2: the five lines are fresh, and the prose quotes the SAME denominator.

    A refreshed table with a stale sentence beside it is worse than a stale table,
    because the sentence is the line people quote. Both prose figures are pulled out of
    the header and compared to the table rather than to a literal, so the node survives
    the next refresh and still catches a half-refresh.
    """
    header = _header_prose()
    rows = _recorded_figure(header, "SOAK_GATE_ROWS_AT_RECORDING")
    table = _table_rows()
    assert set(table) == set(SHIPPED_RULE_IDS), (
        f"the coverage table reads {sorted(table)}, not the five shipped rules "
        f"{sorted(SHIPPED_RULE_IDS)} — an unparsed line is an unpriced rule")
    for rid, (covered, denom, reads) in sorted(table.items()):
        floor_cov, floor_reads = FLOOR_COVERAGE[rid]
        assert denom == rows, (
            f"{rid}'s line divides by {denom} while the record's denominator is {rows}")
        assert covered >= floor_cov, (
            f"{rid} covered {covered} rows against a floor of {floor_cov} at "
            "2026-10-19T00:35Z")
        assert reads >= floor_reads, (
            f"{rid} reports {reads} file-reads against a floor of {floor_reads}; reads "
            "accumulate over a fixed window, so a lower figure is an older measurement")

    narrow = table["naive-now-in-artifact-name"][0]
    priced = re.search(r"which is (\d+) rows of the (\d+)", header)
    assert priced, (
        "the sentence that prices `naive-now-in-artifact-name`'s zero no longer states "
        f"its fraction; the table says {narrow}/{rows}")
    assert (int(priced.group(1)), int(priced.group(2))) == (narrow, rows), (
        f"the prose says '{priced.group(0)}' while the table says "
        f"{narrow}/{rows} — the refreshed table and the stale sentence are both in "
        "the file, which is the exact half-refresh this item was filed over")

    over = re.search(r"— (\d+) rows over (?:the window|.*?days)", header)
    assert over, (
        f"the #679 comparison still has no row count for this soak, so it cannot be "
        f"read against #679's 790; the record's denominator is {rows}")
    assert int(over.group(1)) == rows, (
        f"'{over.group(0)}' names {over.group(1)} rows beside a denominator of {rows}")


def test_the_wider_scope_sub_figure_is_restated_over_the_refreshed_total():
    """Clause 3: #2810's widening covers a restated sub-figure, not '4 rows out of 224'.

    The old figure was not stale, it was unmeasurable: it counted rows in a window
    whose start moved. The replacement has to carry its own bound — the widened scope
    began at 1168247e's commit time, 2026-10-16T09:03:23Z UTC — so that a reader can
    re-run the header's own `select` with that second bound and get the same number.
    """
    header = _header_prose()
    rows = _recorded_figure(header, "SOAK_GATE_ROWS_AT_RECORDING")
    assert "4 rows out of 224" not in header, (
        "the widened-scope sub-figure is still the pre-refresh one")
    m = re.search(r"(\d+) rows? (?:of the |out of )(\d+) .{0,80}wider scope|"
                  r"wider scope at all[^.]*\((\d+) of (\d+)", header)
    assert m, (
        "no sentence states how many of the refreshed rows were gated against #2810's "
        "wider scope; without it the reader cannot tell how much of the zero belongs to "
        "the widened rule")
    n, denom = (int(m.group(1)), int(m.group(2))) if m.group(2) and m.group(1) else \
        (int(m.group(3)), int(m.group(4)))
    assert denom == rows, (
        f"the wider-scope sub-figure divides by {denom}, not the record's {rows}")
    assert n >= FLOOR_ROWS_ON_WIDER_SCOPE, (
        f"{n} rows were gated against the wider scope against a floor of "
        f"{FLOOR_ROWS_ON_WIDER_SCOPE} measured 2026-10-19 — 4 rows of 224 is neither "
        "the old measurement nor a possible new one")
    assert SCOPE_WIDENED_AT in header, (
        f"the sub-figure names no bound: {SCOPE_WIDENED_AT} is 1168247e's commit time, "
        "and without it the count cannot be recomputed, only re-quoted")


def test_the_census_keeps_the_rules_under_whole_tree_watch_not_the_fractions_current():
    """Clause 4: the fractions come from the ledger block; the census watches the RULES.

    The header used to close by saying the weekly census 'is what keeps them current',
    and it does not: `scripts/maintenance/code_rules_census.py` denominates over
    tracked .py (`tracked_python`, run with `whole_tree=True`, reported as
    `"kind": "whole_tree"` beside `"of"`), and it never opens a ledger — `promotions`
    does not occur in its source, which is what this node asserts rather than trusting.
    A reader following the old sentence would have looked in the census for the gate-row
    fractions and found a different denominator entirely.
    """
    census = (ROOT / "scripts" / "maintenance" / "code_rules_census.py").read_text(
        encoding="utf-8")
    assert "def tracked_python" in census and "whole_tree" in census, (
        "the census no longer denominates over tracked .py; if its denominator moved, "
        "the sentence in the header moves with it, in the same change")
    assert "promotions" not in census, (
        "the census now reads the automod ledger, so the claim that it never touches "
        "gate rows has to be re-read before this node keeps passing")

    header = _header_prose()
    assert "keeps them current" not in header, (
        "the header still sends a reader to the census for the gate-row fractions")
    low = header.lower()
    assert "code_rules_census.py" in header, "the census is no longer named at all"
    assert "whole" in low and ("tracked .py" in low or "tracked python" in low), (
        "the census sentence must say what it denominates over — tracked .py over the "
        "whole tree — or the two instruments are still conflated")
    fractions_to_block = re.search(
        r"(?:re-?run(?:ning)?|the recompute block|RECOMPUTE-BEGIN)[^.]{0,160}(?:fractions|coverage)",
        header) or re.search(
        r"(?:fractions|coverage)[^.]{0,160}(?:recompute|re-?run)", header)
    assert fractions_to_block, (
        "nothing in the header says where the per-rule fractions come from now that the "
        "census sentence is gone; the answer is the RECOMPUTE block's second command")


@pytest.mark.parametrize("rule_id", SHIPPED_RULE_IDS)
def test_the_coverage_floor_names_the_rule_it_prices(rule_id):
    """Clause 2's table half, one rule at a time: a floor exists for what ships.

    Parametrized over the shipped ids and keyed on `FLOOR_COVERAGE`, so a sixth rule
    arrives with no floor and this node says so instead of the table quietly growing a
    row nobody measured.
    """
    table = _table_rows()
    assert rule_id in FLOOR_COVERAGE, (
        f"{rule_id} ships but has no 2026-10-19 floor here — record what it covered and "
        "add the floor, or its zero is unpriced")
    assert rule_id in table, f"{rule_id} is missing from the coverage table"
    covered, _denom, reads = table[rule_id]
    floor_cov, floor_reads = FLOOR_COVERAGE[rule_id]
    assert covered >= floor_cov and reads >= floor_reads, (
        f"{rule_id}: the header records ({covered} rows, {reads} file-reads), under the "
        f"({floor_cov}, {floor_reads}) measured at 2026-10-19T00:35Z")


def test_the_rule_surface_below_the_comment_block_is_the_one_the_anchor_shipped():
    """Clause 5: `diff <(grep -v '^#' new) <(git show <anchor>:… | grep -v '^#')` is empty.

    Run as the clause states it, against the last commit that legitimately touched a
    rule line rather than against HEAD — the point is that the rule surface has not
    moved since #2810 widened a scope, through #2809's soak record and #2992's refresh,
    both of which were prose. A rule change updates this anchor deliberately, exactly
    like the id and matcher lists; it must never be a side effect of a comment edit.
    """
    shown = _git(ROOT, "show", f"{NON_COMMENT_SURFACE_ANCHOR}:scripts/automod/code_rules.toml")
    before = _non_comment_lines(shown)
    after = _non_comment_lines(RULES_PATH.read_text(encoding="utf-8"))
    assert before, f"{NON_COMMENT_SURFACE_ANCHOR} shipped an empty rule file"
    moved = [(i, b, a) for i, (b, a) in enumerate(zip(before, after)) if b != a]
    assert not moved and len(before) == len(after), (
        f"the non-comment surface differs from {NON_COMMENT_SURFACE_ANCHOR}: "
        f"{moved[:3] or f'{len(after)} lines vs {len(before)}'} — a comment-only edit "
        f"cannot move a rule, and if a rule genuinely changed, move "
        f"NON_COMMENT_SURFACE_ANCHOR in the same commit that changed it")

    rules = CR.load_rules(RULES_PATH)
    assert [r.id for r in rules] == RULE_IDS_AT_RECORDING, (
        "the ids moved under cover of a comment edit")
    assert [(r.id, r.matcher) for r in rules] == RULE_MATCHERS_AT_RECORDING, (
        "a matcher was swapped under cover of a comment edit")
    assert [tuple(r.scope) for r in rules] == RULE_SCOPES_AT_RECORDING, (
        f"the scopes moved: {[tuple(r.scope) for r in rules]} vs "
        f"{RULE_SCOPES_AT_RECORDING}. #2810's widened scope is pinned by name in its own "
        "three nodes; nothing in a comment edit licenses changing one")


# ---------------------------------------------------------------------------
# #3263 — the ownership ruling written into the tree, a soak window that replays, no
# dangling section pointer, and the four allow entries the file has always carried.
#
# Each node answers one clause with a mechanism that is not the prose it checks: this
# file's own bytes on disk for the ruling, `bash`+`jq` over a synthetic ledger for the
# window, the loader for the allow list, and a positive control for the pointer resolver
# so an assertion of absence cannot pass by never firing.
# ---------------------------------------------------------------------------

#: The two sentences of the settled figure-owner ruling, verbatim, which BOTH the rule
#: file's soak header and this file's comment block above `RULES_LANDED_AT` must carry.
#: Quoted rather than paraphrased: two paraphrases of one ruling drift, and the drift is
#: the bug — the header said the owed-check pass owns the refresh while this block said a
#: person had to settle it.
OWNERSHIP_RULING = (
    "The owed-check pass alone decides that this soak record has gone stale, and it "
    "alone raises the figures, by publishing the fresh measurement in its decision.",
    "A round changes these comment lines only inside an organic edit it is already "
    "making for another reason.",
)

#: The item both files name as the organic edit that carried the ruling in. An organic
#: edit is named, never implied, or the next round claims the licence by analogy.
OWNERSHIP_RULING_ITEM = "#3263"

#: Every allow entry the five rules ship, as #3357 left them. A comment-only edit adds
#: none: an exemption for a shape the matcher cannot even see is a dead entry, and the
#: rule it would blank is the one whose whole point is that nobody exports the data root.
#:
#: The list moved on purpose exactly once since #2810, in #3357, and it moved in the only
#: direction this pin is meant to allow: the matcher learned to follow a store path one
#: hop, so the two read-only `?mode=ro` `count_kg_rows` witnesses became sites the rule
#: reports and therefore sites that need an exemption in writing. Each was invisible
#: before, which is not the same as excused.
ALLOW_ENTRIES_AT_RECORDING = [
    ("broad-except-swallow-same-line", "app/sessions_io.py"),
    ("exports-lloyd-data-to-process-environ", "tests/conftest.py"),
    ("exports-lloyd-data-to-process-environ", "tests/test_youtube_artifact_phase.py"),
    ("opens-kg-store-outside-kg-store", "agent-services/guardian/guardian.py"),
    ("opens-kg-store-outside-kg-store", "scripts/automod/promote.py"),
    ("opens-kg-store-outside-kg-store", "tests/test_data_home.py"),
]


def _prose(lines: list[str]) -> str:
    """Comment lines as one string, with bare `#` lines collapsed out of it.

    `_header_prose` keeps a space for every comment line including the blank ones, so a
    sentence that wraps near one comes out double-spaced there; text that has to match in
    two files is compared in a form where one paragraph is one run of single spaces.
    """
    return " ".join(t for t in (ln.lstrip("#").strip() for ln in lines) if t)


def _own_comment_block_lines() -> list[str]:
    """This file's own comment block above `RULES_LANDED_AT`, read off disk.

    Read from disk and not restated, because the clause is about what the shipped file
    says: a node that paraphrased its own module would keep passing after the comment it
    certifies had been edited away.
    """
    lines = Path(__file__).resolve().read_text(encoding="utf-8").splitlines()
    hits = [i for i, ln in enumerate(lines) if ln.startswith("RULES_LANDED_AT = ")]
    assert hits, "RULES_LANDED_AT is no longer a module-level constant in this file"
    block: list[str] = []
    for ln in reversed(lines[:hits[0]]):
        if ln.lstrip().startswith("#"):
            block.append(ln)
        elif ln.strip():
            break
    assert block, "nothing comments the RULES_LANDED_AT anchor any more"
    return list(reversed(block))


def _dangling_section_refs(text: str) -> list[str]:
    """Every `see <SECTION>` in `text` whose section is not defined in that same text.

    A pointer inside a file has to land in that file: the reader holding one rule file has
    no reason to know another subsystem spells the same word, and a section nobody wrote
    is worse than no pointer, because it reads as a checked reference.
    """
    return [name for name in re.findall(r"\bsee ([A-Z][A-Z0-9_]{2,})\b", text)
            if not re.search(rf"(?m)^\s*#*\s*{name}\b", text)]


def test_the_ownership_ruling_is_the_same_two_sentences_in_both_files():
    """Clauses 1 and 2: one ruling, quoted identically in the rule file and here.

    Three surfaces. (1) The rule file's soak header carries both sentences and names the
    item that carried them in, so the answer to "whose figures are these?" lives in the
    file the figures live in. (2) This file's comment block carries the same two sentences
    quoted, not restated. (3) The open question is gone from this block, so the block
    cannot be read as licensing a round to raise a figure on its own initiative.
    """
    header = _prose(_header_lines())
    own = _prose(_own_comment_block_lines())
    for sentence in OWNERSHIP_RULING:
        assert sentence in header, (
            f"the rule file's soak header no longer carries, verbatim: {sentence!r} — "
            "with the ownership question open again, every round has to guess whether it "
            "may touch these lines")
        assert sentence in own, (
            f"the comment block above RULES_LANDED_AT no longer quotes: {sentence!r} — "
            "two files disagreeing about who owns the figures is the state #3263 was "
            "filed to close")
    assert OWNERSHIP_RULING_ITEM in header, (
        f"the header states the ruling without naming {OWNERSHIP_RULING_ITEM} as the "
        "organic edit it came in on, so the exception has no precedent to be read "
        "against")
    assert OWNERSHIP_RULING_ITEM in own, (
        "this file's block names no item, so the ruling quoted here cannot be checked "
        "against the one in the rule file")
    assert "still has to settle" not in own, (
        "the comment block above RULES_LANDED_AT still asks a person to settle an "
        "ownership question its own owed-check pass had already ruled")


def test_the_recompute_block_bounds_its_window_on_every_command(tmp_path):
    """Clause 3: the block replays the recorded window, on all four of its arms.

    The header's figures describe rows between `SINCE` and `UNTIL`. Unbound at the top the
    same command counts rows the record never measured, so a reader who "verifies the
    soak" gets a different soak and a zero that belongs to neither. Planted here: exactly
    the recorded number of rows spread across the recorded span, over the recorded number
    of distinct rounds, every one `evaluated` and flag-free — plus one flagged `gate` row
    past the bound, from a round the record never counted. If the bound is not honoured,
    four figures move at once (the rows, the flags, the denominator on every coverage line,
    the widened-scope cut), which is why each is asserted on its own.
    """
    header = _prose(_header_lines())
    rows = _recorded_figure(header, "SOAK_GATE_ROWS_AT_RECORDING")
    m = re.search(r"(\d+) `?gate`? rows from (\d+) distinct rounds", header)
    assert m, "the header's row count and distinct-round count are no longer one phrase"
    rounds = int(m.group(2))
    assert rows >= rounds, (
        f"the record claims {rows} rows across {rounds} distinct rounds — more rounds "
        "than rows is not a measurement this fixture can plant")
    span = re.search(r"Date span (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) \.\. "
                     r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", header)
    assert span, "the header names no `Date span`, so the window has no recorded end"
    start, stop = span.group(1), span.group(2)

    body = _recompute_body()
    bound = re.search(r'UNTIL="\$\{CODE_RULES_SOAK_UNTIL:-'
                      r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\}"', body)
    assert bound, (
        "the RECOMPUTE block binds no UNTIL default, so re-running it counts rows the "
        "record never measured and prints a soak that never happened")
    assert bound.group(1) == stop, (
        f"UNTIL defaults to {bound.group(1)} while the recorded span ends {stop}: the "
        "bound has to be the end of the window the figures above it describe, or the "
        "block replays a different soak than the one it sits under")

    cmds: list[str] = []
    cur: str | None = None
    for ln in body.splitlines():
        if ln.lstrip().startswith("jq "):
            if cur is not None:
                cmds.append(cur)
            cur = ln
        elif cur is not None:
            cur += "\n" + ln
    if cur is not None:
        cmds.append(cur)
    assert len(cmds) == 4, (
        f"the block runs {len(cmds)} jq commands, not the four the record's figures come "
        "from — a fifth arm is an unbound figure nobody has to filter")
    for i, cmd in enumerate(cmds, 1):
        assert '--arg u "$UNTIL"' in cmd, (
            f"command {i} is never given UNTIL, so it cannot honour the bound")
        assert ".created_at<=$u" in cmd, (
            f"command {i} does not filter .created_at<=$u, so its figure counts rows past "
            "the recorded window while the figures above it do not")

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    span_start = datetime.strptime(start, fmt).replace(tzinfo=timezone.utc)
    span_stop = datetime.strptime(stop, fmt).replace(tzinfo=timezone.utc)
    whole = int((span_stop - span_start).total_seconds())

    def stamp(i: int) -> str:
        """One in-window row time, with the recorded endpoints hit exactly."""
        if i == 0:
            return start
        if i == rows - 1:
            return stop
        offset = timedelta(seconds=whole * i // (rows - 1))
        return (span_start + offset).strftime(fmt)

    every_rule = {rid: 1 for rid in SHIPPED_RULE_IDS}
    planted = [stamp(i) for i in range(rows)]
    rows_jsonl = [_fixture_row(t, {}, dict(every_rule), round_id=f"SM_R{i % rounds:04d}")
                  for i, t in enumerate(planted)]
    rows_jsonl.append(_fixture_row(
        "2026-10-24T18:46:41Z", {"opens-kg-store-outside-kg-store": 1}, dict(every_rule),
        round_id="SM_PAST_THE_BOUND"))
    ledger = tmp_path / "window.jsonl"
    ledger.write_text("\n".join(rows_jsonl) + "\n", encoding="utf-8")

    out = _run_recompute_block(tmp_path, ledger)
    head = out.splitlines()[0]
    assert f"rows {rows}  flags 0  span {start} .. {stop}" in head, (
        f"over a ledger of the recorded window plus one row past the bound, the block's "
        f"headline line is {head!r} instead of the recorded {rows} rows, zero flags and "
        "the recorded span — the later row is being counted")
    for rid in SHIPPED_RULE_IDS:
        assert f"{rid}: {rows}/{rows} rows saw >=1 file" in out, (
            f"{rid}'s coverage line does not divide by the window's {rows} rows, so the "
            "table's denominator and the headline figure are two different soaks")
    assert f"rows {rows}  distinct_rounds {rounds}  unevaluated 0" in out, (
        f"the distinct-round arm does not report the recorded {rows} rows over {rounds} "
        "rounds with nothing unevaluated:\n"
        + "\n".join(ln for ln in out.splitlines() if "distinct_rounds" in ln))
    wide = sum(1 for t in planted if t >= SCOPE_WIDENED_AT)
    assert f"rows at or after the #2810 widening: {wide}" in out, (
        f"the widened-scope arm should count the {wide} in-window rows at or after "
        f"{SCOPE_WIDENED_AT}; anything it finds after the bound makes the sub-figure "
        "unrecomputable:\n"
        + "\n".join(ln for ln in out.splitlines() if "widening" in ln))


def test_the_header_promises_no_section_that_the_rule_file_does_not_have():
    """Clause 4: a `see <SECTION>` has to land in the file it is written in.

    `see AGING, last` promised a section its own round never wrote: `AGING` occurred once
    in the rule file, in the pointer itself, and every other use of that word in the repo
    names a queue-priority constant in `workers/queue.py` (re-named in
    `architecture/workers.md`) — a reader who follows the sentence arrives in another
    subsystem. The positive control below is what keeps this a test and not an echo of a
    grep: the resolver must still call the removed pointer dangling when it is handed the
    sentence that carried it.
    """
    text = RULES_PATH.read_text(encoding="utf-8")
    assert text.count("AGING") == 0 or re.search(r"(?m)^\s*#\s*AGING\b", text), (
        f"`AGING` appears {text.count('AGING')} time(s) in the rule file with no `AGING` "
        "section defining it, so the pointer sends the reader somewhere that does not "
        "exist")
    assert _dangling_section_refs("the figures it had aged out of — see AGING, last") \
        == ["AGING"], "the resolver does not catch the pointer this clause removed"
    dangling = _dangling_section_refs(text)
    assert dangling == [], (
        f"the rule file points at sections it does not have: {dangling}")


def test_the_comment_only_edit_carries_no_new_allow_entry():
    """Clause 5's unpinned half: the exemption list is the list on this pin, no more.

    The rest of clause 5 — non-comment bytes, the five ids, matchers and scopes, the
    pinned strings, one published flip bound — is held by the anchor node, the precedent
    node and the bound node above. Nothing pinned the `allow` list, which is the part of a
    comment-only change that moves a matcher without touching code: an entry for a file
    the rule cannot report is dead weight that blanks that rule over that file, and
    `load_rules` checks only that an entry carries a why, never that the exception is
    needed by a hit.

    The equality is over `ALLOW_ENTRIES_AT_RECORDING`, which is six as #3357 left them,
    and that is the whole of how a change to the list is allowed: not by editing this
    node's prose, but by making the matcher report the site first, which the live-tree
    control then shows.
    """
    rules = CR.load_rules(RULES_PATH)
    entries = sorted((r.id, a.path) for r in rules for a in r.allow)
    assert entries == ALLOW_ENTRIES_AT_RECORDING, (
        f"the allow list is {entries}, not the "
        f"{len(ALLOW_ENTRIES_AT_RECORDING)} entries of ALLOW_ENTRIES_AT_RECORDING: a "
        "soak record has no licence to exempt a file. An exception needs a hit the "
        "matcher reports, which is measurable, not a shape a reader believes is harmless")
    missing_why = [(r.id, a.path) for r in rules for a in r.allow if not a.why.strip()]
    assert missing_why == [], f"allow entries with no why: {missing_why}"
