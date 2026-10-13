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
