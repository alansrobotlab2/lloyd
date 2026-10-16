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
# rule; and the whole-tree control is still zero hits over the five allowed sites.
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


def test_the_whole_tree_control_is_still_zero_hits_over_exactly_five_allowed_sites():
    """Clause 4: with the scope widened, HEAD still has nothing to report.

    The five sites are the whole allow population of the five rules at filing — two
    `broad-except-swallow-same-line` drains in `app/sessions_io.py`, the test suite's
    scratch-root exports, and the guardian's read-only store tripwire in
    `tests/test_data_home.py`. The line numbers move whenever those files are edited,
    so what is pinned is the count and the paths, which is the part that carries the
    claim "no new violator slipped in behind the widening".
    """
    res = CR.run("HEAD", ROOT, rules_path=RULES_PATH,
                 files=_tracked_python(ROOT), whole_tree=True)
    assert res["status"] == CR.EVALUATED, res
    assert res["hits"] == [], (
        "new violators on the live tree under the widened scope: "
        + "; ".join(f"{h['rule']} at {h['path']}:{h['line']}" for h in res["hits"]))
    sites = sorted((h["rule"], h["path"]) for h in res["allowed"])
    print("allowed sites: " + "; ".join(f"{r} @ {p}" for r, p in sites))
    assert len(sites) == 5, f"the control reports {len(sites)} allowed sites, not 5: {sites}"
    assert sorted(p for _, p in sites) == [
        "app/sessions_io.py", "app/sessions_io.py", "tests/conftest.py",
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
# "224 gate rows" and "120 landings" were true when written and are re-measured by the
# owed-check pass, which is the only thing allowed to update them. A node that
# re-read the ledger and demanded its flags still be 0 would enforce this instrument
# through the test suite — and a rule that fails a round is precisely the flip #679
# ruled has to be a human decision, so the suite must not make it.
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
                 covered: dict[str, int]) -> str:
    """A `gate` ledger row shaped like the ones `rung_code_rules` writes."""
    return json.dumps({
        "event": "gate", "rung": "code_rules", "round_id": "SM_FIXTURE",
        "created_at": created_at,
        "code_rules": {
            "status": "evaluated", "counts": counts, "labels": [],
            "totals": {"files": 2,
                       "coverage": {rid: {"covered": n, "of": 2}
                                    for rid, n in covered.items()}}}})


def _run_recompute_block(tmp_path: Path, ledger: Path) -> str:
    """Extract the header's RECOMPUTE block and run it against `ledger`."""
    lines = _header_lines()
    starts = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("# RECOMPUTE-BEGIN")]
    ends = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("# RECOMPUTE-END")]
    assert starts and ends and ends[0] > starts[0], (
        "the soak record names no RECOMPUTE-BEGIN/END block, so its figures cannot be "
        "recomputed by a reader holding only this file")
    body = "\n".join(ln[1:] if ln.startswith("#") else ln
                     for ln in lines[starts[0] + 1:ends[0]])
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
        "the comparison's own denominators are missing, so this soak's 224 rows cannot "
        "be read against the one that ruled first")
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
