"""The review rung refused 86% of what it graded. This is why, and the fix.

Over 2026-09-11 the rung graded 14 autocode rounds and blocked six of them
twice each, ending the round. The findings were real; most were also
unfixable *inside* the round:

  * a clause that can only be observed with live traffic, which had no word
    of its own and came out `partial` — a refusal for a round that did the
    work (#859, refused twice with the mechanism complete on both commits);
  * a seam across an HTTP boundary that no test in this repo can cross until
    the change is live;
  * "test files changed but no test function was added", on rounds that
    tightened existing assertions;
  * five `met`s downgraded for an `evidence_path` that was real but lived in
    the vault;
  * an amendment answered from the ledger and never re-graded (866-c);
  * a grader 503 caused by a *sibling* round's landing (866-a).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.automod import review as RV


def _parsed(**kw):
    base = dict(premise="sound", clauses=[], test_honesty=[],
                seams_unverified=[], downgraded=[], summary="ok")
    base.update(kw)
    return base


def _clause(idx, verdict, **kw):
    c = dict(clause=idx, verdict=verdict, evidence_path="app/x.py",
             evidence_line=1, test_node_id="tests/t.py::test_x",
             how_verified="ran", note="n")
    c.update(kw)
    return c


# ---------------------------------------------------------------------------
# post_landing
# ---------------------------------------------------------------------------

def test_post_landing_is_a_clause_verdict():
    assert "post_landing" in RV.CLAUSE_VERDICTS
    enum = RV.REVIEW_SCHEMA["properties"]["clauses"]["items"]["properties"]["verdict"]["enum"]
    assert enum == list(RV.CLAUSE_VERDICTS), "schema is built from the one list"


def test_a_post_landing_clause_does_not_refuse_the_round():
    """#859's exact shape: the mechanism is in the diff and correct, and the
    clause needs traffic that does not exist until it is live.
    """
    kind, findings = RV.decide(
        _parsed(clauses=[_clause(1, "post_landing", note="needs a day of traffic")]), [])
    assert kind == "pass"
    assert "after landing" in findings
    assert "needs-human" in findings


def test_a_post_landing_clause_needs_a_pinned_mechanism(tmp_path):
    """Without evidence it is a claim about a mechanism nobody has seen —
    which is exactly "not done", and must not land as though it were.
    """
    obj = {"premise": "sound", "clauses": [
        {"clause": 1, "verdict": "post_landing", "evidence_path": "",
         "evidence_line": 0, "test_node_id": "", "how_verified": "inferred",
         "note": "later"}]}
    parsed = RV.parse_review(obj, worktree=tmp_path, changed_tests=[], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "partial"
    assert any("post_landing without a pinned mechanism" in d
               for d in parsed["clauses"][0]["downgraded"])


def test_a_post_landing_clause_with_a_real_path_survives(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("x = 1")
    obj = {"premise": "sound", "clauses": [
        {"clause": 1, "verdict": "post_landing", "evidence_path": "app/x.py",
         "evidence_line": 1, "test_node_id": "", "how_verified": "read",
         "note": "needs traffic"}]}
    parsed = RV.parse_review(obj, worktree=tmp_path, changed_tests=[], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "post_landing"


def test_unmet_still_refuses_beside_a_post_landing():
    kind, _ = RV.decide(_parsed(clauses=[
        _clause(1, "post_landing"), _clause(2, "unmet")]), [])
    assert kind == "retry"


# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------

def test_a_seam_blocks_on_the_first_attempt():
    """Blocking once is what keeps it honest: the author is told, with a
    chance to add the test.
    """
    kind, findings = RV.decide(
        _parsed(seams_unverified=[{"seam": "loopback POST",
                                   "testable_before_landing": True}]),
        [], attempt=1)
    assert kind == "retry"
    assert "seam unverified" in findings


def test_a_seam_advises_on_the_second():
    kind, findings = RV.decide(
        _parsed(seams_unverified=[{"seam": "loopback POST",
                                   "testable_before_landing": True}]),
        [], attempt=2)
    assert kind == "pass"
    assert "not refusing again" in findings
    assert "loopback POST" in findings, "the finding still rides into the report"


@pytest.mark.parametrize("policy,attempt,blocks", [
    ("first", 1, True), ("first", 2, False), ("first", 3, False),
    ("always", 1, True), ("always", 2, True),
    ("never", 1, False), ("never", 2, False),
    ("", 1, True),          # empty falls back to `first`
    ("nonsense", 1, True),  # so does an unknown policy
])
def test_the_seam_policy_table(policy, attempt, blocks):
    assert RV.seams_block(policy, attempt) is blocks


def test_an_untestable_seam_never_blocks_even_on_attempt_one():
    kind, findings = RV.decide(
        _parsed(seams_unverified=[{"seam": "a real pool tick",
                                   "testable_before_landing": False}]),
        [], attempt=1)
    assert kind == "pass"
    assert "post-landing seam" in findings


# ---------------------------------------------------------------------------
# precheck severities
# ---------------------------------------------------------------------------

def test_the_dishonest_test_patterns_still_block():
    for pat, why, sev in RV._HONESTY_PATTERNS:
        assert sev == "blocking", why
    kind, _ = RV.decide(_parsed(), [
        {"file": "tests/t.py", "line": 3, "problem": "`or True` …",
         "severity": "blocking"}])
    assert kind == "retry"


def test_no_new_test_function_is_advisory():
    """A round that tightens an existing test's assertions has pinned exactly
    what it should and added no `def test_`.
    """
    assert RV._NO_NEW_TEST_SEVERITY == "advisory"
    kind, findings = RV.decide(_parsed(), [
        {"file": "tests/t.py", "line": 0,
         "problem": "test files changed but no test function was added while "
                    "the item has acceptance clauses to pin",
         "severity": "advisory"}])
    assert kind == "pass"
    assert "advisory" in findings


def test_a_precheck_with_no_severity_still_blocks():
    """An entry written before severities existed keeps the old reading."""
    kind, _ = RV.decide(_parsed(), [
        {"file": "tests/t.py", "line": 3, "problem": "something"}])
    assert kind == "retry"


def test_a_dishonest_pattern_precheck_is_blocking(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text(
        "def test_a():\n    assert 1 == 1 or True\n")
    monkeypatch.setattr(RV, "_git", lambda *a, **k: "")
    out = RV.honesty_prechecks(tmp_path, "BASE", ["tests/t.py", "app/x.py"],
                               n_clauses=1)
    or_true = next(o for o in out if "or True" in o["problem"])
    assert or_true["severity"] == "blocking"


def test_the_no_new_test_precheck_is_advisory(tmp_path, monkeypatch):
    """The round that tightens an existing test's assertions: a real
    observation and a bad refusal.
    """
    (tmp_path / "tests").mkdir()
    # Post has the same test function as pre, with a tighter assertion — so
    # `added_tests` is 0 and this is the shape that used to refuse.
    (tmp_path / "tests" / "t.py").write_text(
        "def test_a():\n    assert compute() == 7\n")
    monkeypatch.setattr(
        RV, "_git",
        lambda *a, **k: "def test_a():\n    assert compute() is not None\n")
    out = RV.honesty_prechecks(tmp_path, "BASE", ["tests/t.py", "app/x.py"],
                               n_clauses=1)
    no_test = next(o for o in out if "no test function was added" in o["problem"])
    assert no_test["severity"] == "advisory"
    # ...and it does not refuse.
    kind, _ = RV.decide(_parsed(), out)
    assert kind == "pass"


# ── a grader that denies tests the diff demonstrably added (#1442) ─────────
# The review that refused round SM_20260924_104307's last attempt wrote "This
# round's diff adds no such test" about a diff whose own `git diff --stat
# 78f0637f..37179d71` lists four test files, and its prompt contained the line
# `build_prompt` emits naming those four files. The `def test_` delta was
# already computed deterministically by `honesty_prechecks`; nothing compared a
# grader note against it.

def test_the_added_test_delta_counts_only_what_the_diff_added(tmp_path, monkeypatch):
    """Post minus pre per changed test file, floored at zero, so a round that
    tightens assertions is 0 and a round that added four nodes is 4."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text(
        "def test_a():\n    assert 1\n\n\ndef test_b():\n    assert 2\n\n\ndef test_c():\n    assert 3\n")
    monkeypatch.setattr(RV, "_git", lambda *a, **k: "def test_a():\n    assert 1\n")
    assert RV.def_test_delta(tmp_path, "BASE", ["tests/t.py", "app/x.py"]) == 2
    # Same count on both sides: assertions tightened, nothing added.
    monkeypatch.setattr(RV, "_git", lambda *a, **k:
                        "def test_a():\n    assert 1\n\n\ndef test_b():\n    assert 2\n"
                        "\n\ndef test_c():\n    assert 3\n")
    assert RV.def_test_delta(tmp_path, "BASE", ["tests/t.py"]) == 0
    # Nodes removed are not negative: the delta is what the diff ADDED.
    monkeypatch.setattr(RV, "_git", lambda *a, **k:
                        "\n".join(f"def test_{c}():\n    assert 1\n" for c in "abcde"))
    assert RV.def_test_delta(tmp_path, "BASE", ["tests/t.py"]) == 0


def test_a_note_denying_the_added_tests_is_a_contradiction_python_can_see():
    """Verbatim shapes from the 2026-09-24 refusal, both clauses."""
    parsed = {"clauses": [
        {"clause": 3, "verdict": "partial",
         "note": "This round's diff adds no such test; the clause is satisfied by a test "
                 "in a prior landing (agent_mcp/facts.py:520-540, commit 08a4f4f0)"},
        {"clause": 4, "verdict": "partial",
         "note": "Same as clause 3: the pin exists at agent_mcp/facts.py:540-560 from commit "
                 "08a4f4f0, not in this diff's tests."}]}
    reasons = RV.added_test_denials(parsed, added_tests=4)
    assert len(reasons) == 2, reasons
    assert "clause 3" in reasons[0] and "clause 4" in reasons[1]
    assert all("adds no test" in r for r in reasons), reasons
    assert all("4" in r for r in reasons), "the reason names the delta it contradicts"
    # Zero delta: the same note is a true observation, not a contradiction.
    assert RV.added_test_denials(parsed, added_tests=0) == []


def test_a_note_naming_the_test_it_ran_is_not_read_as_a_denial():
    """The rail reads a denial of the diff's tests, not any sentence with the
    word `test` in it."""
    parsed = {"clauses": [
        {"clause": 1, "verdict": "met",
         "note": "the test the diff adds, test_the_guard_trips, was run and is green"},
        {"clause": 2, "verdict": "met",
         "note": "the diff adds nothing to the config and the existing test still pins it"}]}
    assert RV.added_test_denials(parsed, added_tests=3) == []


# ---------------------------------------------------------------------------
# evidence roots
# ---------------------------------------------------------------------------

def test_a_worktree_path_still_wins(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("x")
    assert RV.normalize_evidence_path("app/x.py", tmp_path) == "app/x.py"


def test_a_vault_relative_path_resolves_under_the_roots(tmp_path):
    root = tmp_path / "vault"
    (root / "lloyd").mkdir(parents=True)
    (root / "lloyd" / "SOUL.md").write_text("soul")
    got = RV.normalize_evidence_path("lloyd/SOUL.md", tmp_path / "wt",
                                     roots=(root,))
    assert got == str(root / "lloyd" / "SOUL.md")


def test_a_line_suffix_is_still_stripped_under_the_roots(tmp_path):
    root = tmp_path / "vault"
    (root / "lloyd").mkdir(parents=True)
    (root / "lloyd" / "SOUL.md").write_text("soul")
    got = RV.normalize_evidence_path("lloyd/SOUL.md:44-51", tmp_path / "wt",
                                     roots=(root,))
    assert got.endswith("SOUL.md")


def test_a_path_in_neither_place_is_still_empty(tmp_path):
    assert RV.normalize_evidence_path("nowhere/at/all.py", tmp_path,
                                      roots=(tmp_path / "vault",)) == ""


# ---------------------------------------------------------------------------
# a root-level module cited under a package dir this checkout does not have
# (#1252: four `met`s on #832 were downgraded for this citation alone)
# ---------------------------------------------------------------------------

def test_a_root_module_cited_under_a_package_dir_that_is_not_there_resolves(tmp_path):
    """`autonomy.py` is at the repo root and `app/` is routers only, but the
    grader's mental model is a package, so it writes `app/autonomy.py:1605-1620`
    — a true claim about a path that is on no disk. The `met` is right; the
    resolver was the thing that could not see it."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "config.py").write_text("x")
    (tmp_path / "autonomy.py").write_text("x")
    assert RV.normalize_evidence_path("app/autonomy.py:1605-1620", tmp_path,
                                      roots=(tmp_path / "vault",)) == "autonomy.py"


def test_the_prefix_retry_runs_only_after_the_written_path_and_one_deep(tmp_path):
    """The retry is a fallback, not a rewrite: a path that exists as written
    keeps winning, and a made-up path stays refused because stripping stops at
    one segment."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("x")
    (tmp_path / "x.py").write_text("x")
    n = RV.normalize_evidence_path
    # Both places hold a file: the citation as written resolves.
    assert n("app/x.py:5", tmp_path, roots=()) == "app/x.py"
    # One segment deep and no further — `nonexistent/thing.py` is what is left
    # after stripping `app/`, and the bare `thing.py` is never tried.
    assert n("app/nonexistent/thing.py:10", tmp_path, roots=()) == ""
    # A fabricated name under a real package dir still resolves to nothing.
    assert n("app/no_such_module.py:1", tmp_path, roots=()) == ""


def test_a_second_citation_token_resolves_when_the_first_token_misses(tmp_path):
    """The grader puts two locations in one field. Only the head token was ever
    tried — and `;` was not even in its separator set, which is why
    `app/autonomy.py:1605; tests/test_autonomy_scheduler.py::test_x` resolved
    to nothing with a real test path attached to it."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_autonomy_scheduler.py").write_text("x")
    n = RV.normalize_evidence_path
    assert n("app/autonomy.py:1605; tests/test_autonomy_scheduler.py::test_x",
             tmp_path, roots=()) == "tests/test_autonomy_scheduler.py"
    # A space-separated pair behaves the same.
    assert n("app/autonomy.py:1605 tests/test_autonomy_scheduler.py:120",
             tmp_path, roots=()) == "tests/test_autonomy_scheduler.py"
    # Order still decides: once the head token names something real it wins,
    # so this is a fallback across tokens, not a search for the best one.
    (tmp_path / "autonomy.py").write_text("x")
    assert n("app/autonomy.py:1605; tests/test_autonomy_scheduler.py::test_x",
             tmp_path, roots=()) == "autonomy.py"


def test_every_citation_that_resolved_before_resolves_identically(tmp_path):
    """The no-regression half: neither the prefix retry nor the extra tokens
    may move a citation that already resolved, including the spellings the
    other files pin (`tests/test_automod_review.py::
    test_evidence_paths_are_normalized_before_they_are_judged`,
    `tests/test_review_transport.py::
    test_evidence_paths_with_symbols_anchors_or_a_dead_absolute_prefix_still_resolve`)
    and both `REVIEW_EVIDENCE_ROOTS` shapes."""
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "config.py").write_text("x")
    (tmp_path / "tests" / "test_autonomy_scheduler.py").write_text("x")
    vault = tmp_path / "vault"
    (vault / "lloyd").mkdir(parents=True)
    (vault / "lloyd" / "SOUL.md").write_text("soul")
    n = RV.normalize_evidence_path
    assert n("app/config.py:358", tmp_path, roots=(vault,)) == "app/config.py"
    assert n("tests/test_autonomy_scheduler.py:100", tmp_path,
             roots=(vault,)) == "tests/test_autonomy_scheduler.py"
    assert n("app/config.py::helper", tmp_path, roots=(vault,)) == "app/config.py"
    assert n("./app/config.py:3", tmp_path, roots=(vault,)) == "app/config.py"
    # An absolute path into a checkout that has since moved: worktree tail.
    assert n("/gone/checkout/review-abc/app/config.py", tmp_path,
             roots=(vault,)) == "app/config.py"
    # A real absolute path, two locations in one field, prose after them.
    assert n(str(vault / "lloyd" / "SOUL.md") + " + " + str(vault / "lloyd")
             + " (the trim is vault-side)", tmp_path, roots=()) == str(vault / "lloyd" / "SOUL.md")
    # The vault-relative fallback, unchanged.
    assert n("lloyd/SOUL.md:44-51", tmp_path / "wt",
             roots=(vault,)) == str(vault / "lloyd" / "SOUL.md")
    # And the refusals stay refusals.
    assert n("nowhere/at/all.py", tmp_path, roots=(vault,)) == ""
    assert n("/nope/x.md", tmp_path, roots=(vault,)) == ""
    assert n("", tmp_path, roots=(vault,)) == ""


def test_the_default_roots_name_the_vault():
    assert any(str(r).endswith("obsidian") for r in RV.REVIEW_EVIDENCE_ROOTS)


# ---------------------------------------------------------------------------
# a citation naming a leading-dot file (#1362). The rail trimmed a cited path
# with `cand.lstrip("./")`, and `str.lstrip` takes a CHARACTER SET, so
# `'.gitignore'.lstrip('./')` is `'gitignore'` — a file on no disk. Any clause
# about a dotfile was therefore ungradeable: round SM_20260922_065953 (item
# #759, whose clause 1 is *about* `.gitignore:33`) was downgraded `met`→partial
# with "grader wrote '.gitignore:33'", and SM_20260912_184007 (#472,
# `.gitignore:103`, `.gitignore:25`) ten days before it.
# ---------------------------------------------------------------------------

def test_a_citation_naming_a_leading_dot_file_resolves(tmp_path):
    """The file is in the tree and the citation names it; the rail returned ""
    anyway, and the caller reads "" as "the grader cited nothing"."""
    (tmp_path / ".gitignore").write_text("*.db\n*.json\n")
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config" / "settings.json").write_text("{}")
    n = RV.normalize_evidence_path
    assert n(".gitignore:33", tmp_path, roots=()) == ".gitignore"
    assert n(".gitignore", tmp_path, roots=()) == ".gitignore"
    assert n("./.gitignore:33", tmp_path, roots=()) == ".gitignore"
    # A dotfile as a directory name, and one as a file inside it.
    assert n(".config/settings.json:1", tmp_path, roots=()) == ".config/settings.json"


def test_a_dotfile_clause_is_graded_met_and_not_downgraded(tmp_path, monkeypatch):
    """The downgrade, end to end through the parser that the gate's review rung
    actually calls: the grader wrote `met` and pointed at a real dotfile, and
    the rail overrode it. The control beside it is the half that must stay —
    a dotfile that is NOT in the tree is still no evidence. `parse_review`
    always consults `REVIEW_EVIDENCE_ROOTS`, and the real vault has a
    `.gitignore` of its own, so the roots are pinned empty here to keep both
    readings about the worktree."""
    monkeypatch.setattr(RV, "REVIEW_EVIDENCE_ROOTS", ())
    (tmp_path / ".gitignore").write_text("*.db\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_review_policy.py").write_text("x")
    obj = {"premise": "sound", "summary": "ok", "test_honesty": [],
           "seams_unverified": [],
           "clauses": [{"clause": 1, "verdict": "met",
                        "evidence_path": ".gitignore:33", "evidence_line": 33,
                        "test_node_id": "tests/test_review_policy.py::test_a",
                        "how_verified": "ran", "note": "the rule fires"}]}
    parsed = RV.parse_review(obj, worktree=tmp_path,
                             changed_tests=["tests/test_review_policy.py"],
                             n_clauses=1, tests_passed=True,
                             changed_paths=[".gitignore",
                                            "tests/test_review_policy.py"])
    row = parsed["clauses"][0]
    assert row["verdict"] == "met", row.get("downgraded")
    assert row["evidence_path"] == ".gitignore"
    assert not row.get("downgraded")

    (tmp_path / ".gitignore").unlink()
    gone = RV.parse_review(obj, worktree=tmp_path,
                           changed_tests=["tests/test_review_policy.py"],
                           n_clauses=1, tests_passed=True,
                           changed_paths=["tests/test_review_policy.py"])
    row = gone["clauses"][0]
    assert row["verdict"] == "partial"
    assert any(w.startswith("evidence_path missing or not on disk")
               and "'.gitignore:33'" in w for w in row["downgraded"]), row["downgraded"]


def test_the_vault_root_fallback_resolves_a_dotfile_too(tmp_path):
    """Second site, same character set: the fallback that accepts a
    vault-relative citation stripped with its own `lstrip("./")`, so
    `.hidden-note.md:3` vanished while `plain.md` in the SAME root resolved."""
    (tmp_path / "wt").mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "plain.md").write_text("p")
    (vault / ".hidden-note.md").write_text("h")
    n = RV.normalize_evidence_path
    assert n(".hidden-note.md:3", tmp_path / "wt", roots=(vault,)) \
        == str(vault / ".hidden-note.md")
    assert n("plain.md", tmp_path / "wt", roots=(vault,)) == str(vault / "plain.md")
    assert n(".absent-note.md", tmp_path / "wt", roots=(vault,)) == ""


def test_evidence_of_absence_accepts_a_changed_dotfile():
    """A diff that deletes a dotfile cannot cite it as a file on disk, so the
    waiver for changed-and-gone files is the only honest evidence — and it
    never fired, because `first` had its dots eaten before the changed-path
    lookup: `evidence_of_absence('.gitignore:33', ['.gitignore', …])` was
    False with `.gitignore` sitting in `changed_paths`."""
    assert RV.evidence_of_absence(".gitignore:33",
                                 [".gitignore", "tests/test_x.py"]) is True
    assert RV.evidence_of_absence("./.gitignore:33", [".gitignore"]) is True
    # Unchanged readings, both directions.
    assert RV.evidence_of_absence("tests/test_x.py", ["tests/test_x.py"]) is True
    assert RV.evidence_of_absence("tests/test_x.py", ["app/x.py"]) is False
    assert RV.evidence_of_absence("old/note.md (absent)", []) is True
    assert RV.evidence_of_absence("", [".gitignore"]) is False


def test_the_prefix_trim_is_a_prefix_and_not_a_character_set(tmp_path):
    """The no-widening half. Every spelling that resolved under `lstrip("./")`
    resolves identically, and a citation to a dotfile that is not there is
    still nothing — the fix removes the bug, not the rail. The path-level
    assertions are therefore true at base as well, by design: the clause they
    pin is "unchanged". The helper's own contract is asserted beside them
    because it is the one claim here a wrong fix would break."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("x")
    n = RV.normalize_evidence_path
    assert n("./app/x.py:3", tmp_path, roots=()) == "app/x.py"
    assert n("././app/x.py", tmp_path, roots=()) == "app/x.py"
    assert n("app/x.py", tmp_path, roots=()) == "app/x.py"
    # A `..`-prefixed citation keeps the resolution lstrip gave it: the tail
    # inside the tree, not a file in some sibling checkout.
    assert n("../app/x.py", tmp_path, roots=()) == "app/x.py"
    # Dot-only tokens name a directory, not evidence.
    assert n(".", tmp_path, roots=()) == ""
    assert n("..", tmp_path, roots=()) == ""
    # The helper's own contract. A leading-dot name keeps its dots; `./` and
    # `../` come off as prefixes, repeated prefixes included; a dot-only token
    # is "", the value every caller reads as "no evidence". `lstrip("./")`
    # mapped `..gitignore` to `gitignore` — a different file — and no caller
    # depends on that mapping, so the prefix reading is what ships.
    t = RV._trim_citation_prefix
    assert t("./app/.gitignore") == "app/.gitignore"
    assert t("..gitignore") == "..gitignore"
    assert t("././x") == "x"
    assert t("../x") == "x"
    assert t("./") == ""
    assert t(".") == ""
    assert t("..") == ""
    assert t("") == ""
    # An absent dotfile is still an unresolvable citation, in either root.
    vault = tmp_path / "vault"
    vault.mkdir()
    assert n(".gitignore", tmp_path, roots=(vault,)) == ""
    assert n(".env:7", tmp_path, roots=(vault,)) == ""
    assert n(".gone.md", tmp_path / "wt", roots=(vault,)) == ""


def test_the_node_and_honesty_rails_read_a_leading_dot_the_same_way(tmp_path):
    """The last two sites of the same character set. Neither gates a dotfile's
    existence — one asks whether a cited test is under `tests/`, the other
    whether a dishonesty remark is about a test — but both ate leading dots
    before asking, which is how a citation to `.tests/test_x.py` (no such
    directory) was judged against the real `tests/test_x.py`."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("x")
    holds, reason = RV._node_rail("./tests/test_x.py::test_a", worktree=tmp_path,
                                  changed=set(), how="ran", tests_passed=True)
    assert holds, reason
    assert RV._node_rail("./.tests/test_x.py::test_a", worktree=tmp_path,
                         changed=set(), how="ran",
                         tests_passed=True)[0] is False
    parsed = RV.parse_review(
        {"premise": "sound", "summary": "ok", "clauses": [], "seams_unverified": [],
         "test_honesty": [
             {"file": "./tests/test_x.py", "line": 3, "severity": "blocking",
              "problem": "asserts a constant"},
             {"file": ".tests/test_x.py", "line": 3, "severity": "blocking",
              "problem": "names a directory that is not there"},
             {"file": ".gitignore", "line": 33, "severity": "blocking",
              "problem": "not a test"}]},
        worktree=tmp_path, changed_tests=["tests/test_x.py"], n_clauses=1)
    severities = {h["file"]: h["severity"] for h in parsed["test_honesty"]}
    assert severities == {"./tests/test_x.py": "blocking",
                          ".tests/test_x.py": "advisory",
                          ".gitignore": "advisory"}


def _prompt_text() -> str:
    return RV.build_prompt(
        contract={"clauses": ["c1"], "amendments": [], "human_clauses": [],
                  "id": 1, "title": "t", "body": "b"},
        diff="diff", diff_truncated=False, changed_tests=["tests/t.py"],
        test_counts={"passed": 1}, worktree=Path("/wt"),
        run_tests=Path("/wt/run"))


def test_the_prompt_no_longer_forbids_what_the_parser_accepts():
    """The prompt said paths are worktree-relative "not `~/…`" while the
    parser already accepted them, and five `met`s were downgraded on
    2026-09-11 for paths that were real.
    """
    text = _prompt_text()
    assert "not `~/…`" not in text
    assert "a vault path you actually read is" in text
    # The three shapes `parse_review` accepts besides a changed test are
    # named, or the grader keeps writing its honest suite-level node and
    # watching Python downgrade it (#860's clause 8, three times).
    flat = " ".join(text.split())
    assert "a suite-level run cited as `tests/ -k <expr>` that you `ran`" in flat
    assert "an existing test outside this diff that you `ran`" in flat
    assert "an evidence_path naming the deleted file marked `(deleted)`" in flat


def test_the_prompt_teaches_post_landing_rather_than_unsatisfiable():
    """The two are opposite remedies: `post_landing` lands the change and
    waits for a person; `unsatisfiable` refuses and amends the contract.
    Telling the grader to use the second for the first is what parked #859.
    """
    text = _prompt_text()
    assert "is `post_landing`" in text
    assert "does not refuse the round" in text
    # `unsatisfiable` is still taught, for the contract-defect case.
    assert "no diff could EVER satisfy" in text


# ---------------------------------------------------------------------------
# the grader is not always up
# ---------------------------------------------------------------------------

def _stream_503(*_a, **_k):
    import urllib.error
    raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)


def test_a_503_before_the_stream_opens_is_retried(tmp_path, monkeypatch):
    """Round 866-a: the grader hit a 503 because ANOTHER round was landing at
    that moment. The rung recorded `external`, the turn ended without ever
    re-gating, and it was reaped 30 minutes later.
    """
    attempts = {"n": 0}

    def _stream(url, payload, timeout):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            _stream_503()
        yield "done", {"response": "r", "stop_reason": "stop",
                       "structured": {"premise": "sound", "clauses": []}}

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: None)
    monkeypatch.setattr(RV, "write_session", lambda *a, **k: "sess-1")

    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=600)
    assert rep["ok"] is True
    assert rep["retries"] == 2
    assert attempts["n"] == 3


def test_the_session_id_is_reused_across_retries(tmp_path, monkeypatch):
    """Three retries must read as one grading of one commit, not as three."""
    minted = []

    def _mint(*a, **k):
        minted.append(1)
        return f"sess-{len(minted)}"

    attempts = {"n": 0}

    def _stream(url, payload, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            _stream_503()
        yield "done", {"response": "r", "stop_reason": "stop",
                       "structured": {"premise": "sound", "clauses": []}}

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: None)
    monkeypatch.setattr(RV, "write_session", _mint)
    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=600)
    assert len(minted) == 1
    assert rep["session_id"] == "sess-1"


def test_a_failure_mid_stream_is_never_retried(tmp_path, monkeypatch):
    """Once events have arrived the turn ran and cost the round. Re-POSTing
    would run a second grading turn whose verdict duplicates a judgment
    already partly made.
    """
    attempts = {"n": 0}

    def _stream(url, payload, timeout):
        attempts["n"] += 1
        yield "token", {"text": "thinking"}
        _stream_503()

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: None)
    monkeypatch.setattr(RV, "write_session", lambda *a, **k: "sess-1")
    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=600)
    assert rep["ok"] is False
    assert attempts["n"] == 1
    assert rep["retries"] == 0


def test_a_non_availability_error_is_not_retried(tmp_path, monkeypatch):
    """A 400 is a refusal of THIS request and waiting changes nothing."""
    import urllib.error
    attempts = {"n": 0}

    def _stream(url, payload, timeout):
        attempts["n"] += 1
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
        yield  # pragma: no cover

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: None)
    monkeypatch.setattr(RV, "write_session", lambda *a, **k: "sess-1")
    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=600)
    assert attempts["n"] == 1
    assert rep["retries"] == 0


def test_the_wait_is_bounded(tmp_path, monkeypatch):
    slept = []

    def _stream(url, payload, timeout):
        _stream_503()
        yield  # pragma: no cover

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(RV, "write_session", lambda *a, **k: "sess-1")
    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=60)
    assert rep["ok"] is False
    assert sum(slept) <= 60
    assert "still unavailable" in rep["error"]


@pytest.mark.parametrize("err,expected", [
    ("HTTP 503: lloyd is landing a code update; retry in 42s", 42.0),
    ("HTTP 503: retry in 9999s", 60.0),        # capped
    ("HTTP 503: no hint at all", 15.0),        # first backoff
])
def test_the_retry_delay_reads_the_servers_own_hint(err, expected):
    assert RV._retry_delay(err, 0) == expected


@pytest.mark.parametrize("err,unavailable", [
    ("HTTP 503: Service Unavailable", True),
    ("URLError: Connection refused", True),
    ("lloyd is landing a code update; not starting a worker turn", True),
    ("HTTP 400: Bad Request", False),
    ("HTTP 500: Internal Server Error", False),
    ("", False),
])
def test_what_counts_as_unavailable(err, unavailable):
    assert RV._is_unavailable(err) is unavailable


# ---------------------------------------------------------------------------
# human_paths, and the post_landing round trip onto the item
# ---------------------------------------------------------------------------

from scripts.automod import backlog as B  # noqa: E402
from scripts.automod import spec as SP    # noqa: E402


@pytest.fixture(autouse=True)
def _seams_first(monkeypatch):
    """This file tests the review rung's mechanics — seams by attempt,
    precheck severities, amendment handling — so it pins `seams_block: first`,
    under which a testable seam still refuses on attempt 1. The shipped
    setting is `never` (2026-09-24, `tests/test_review_grader_policy.py`).
    Set through the config `review.seams_policy` reads, not by patching it.
    (These tests ran under the `table` policy until it was retired the same
    day; they decide under the grader policy, the only one there is.)"""
    from app.config import CONFIG
    automod = dict(CONFIG.get("automod") or {})
    review = dict(automod.get("review") or {})
    review["seams_block"] = "first"
    automod["review"] = review
    monkeypatch.setitem(CONFIG, "automod", automod)



def test_a_scope_refusal_names_the_move():
    """A round told only that it may not have a path reached for `git add -f`,
    which defeats the check rather than reporting past it.
    """
    ok, why, _ = SP.check_scope([".gitignore"])
    assert ok is False
    assert "human_paths" in why
    assert "git add -f" in why
    assert "state dir" in why


def test_human_paths_survive_parse_outcome():
    out = B.parse_outcome({
        "acceptance": "met", "landed": True, "deferred_to": [], "summary": "s",
        "spawned": [], "clause_outcomes": [],
        "human_paths": [{"path": ".gitignore", "reason": "needs a rule"}]})
    assert out["human_paths"] == [{"path": ".gitignore", "reason": "needs a rule"}]


def test_the_outcome_schema_offers_human_paths():
    props = B.IMPLEMENT_OUTCOME_SCHEMA["properties"]
    assert "human_paths" in props
    assert props["human_paths"]["items"]["required"] == ["path", "reason"]
    assert "maxLength" not in json_dumps(B.IMPLEMENT_OUTCOME_SCHEMA)


def json_dumps(x):
    import json
    return json.dumps(x)


def test_apply_post_landing_rescues_a_deferred_clause():
    """The review rung decides a clause is observable only after landing; the
    implementer, inside the round, cannot know that and honestly says
    `deferred`. Re-read, the round has done its job.
    """
    out = {"acceptance": "not_met", "clause_outcomes": [
        {"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []},
        {"clause": 2, "outcome": "deferred", "evidence": "", "deferred_to": []},
    ]}
    got = B.apply_post_landing(out, [2])
    assert got["acceptance"] == "met"
    assert got["post_landing_clauses"] == [2]
    assert got["clause_outcomes"][1]["outcome"] == "met"
    assert got["clause_outcomes"][1]["post_landing"] is True


def test_apply_post_landing_never_rescues_an_unrelated_clause():
    out = {"acceptance": "not_met", "clause_outcomes": [
        {"clause": 1, "outcome": "not_met", "evidence": "", "deferred_to": []},
        {"clause": 2, "outcome": "deferred", "evidence": "", "deferred_to": []},
    ]}
    got = B.apply_post_landing(out, [2])
    assert got["acceptance"] == "not_met", "clause 1 is still genuinely not met"


def test_apply_post_landing_leaves_a_met_claim_alone():
    """A clause the implementer called `met` stands on its own and needs no
    rescue — marking it post_landing would hold an item open for nothing.
    """
    out = {"acceptance": "met", "clause_outcomes": [
        {"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []}]}
    got = B.apply_post_landing(out, [1])
    assert "post_landing_clauses" not in got
    assert got is out


def test_apply_post_landing_is_a_noop_with_no_marks():
    out = {"acceptance": "deferred", "clause_outcomes": []}
    assert B.apply_post_landing(out, []) is out
    assert B.apply_post_landing(None, [1]) is None
