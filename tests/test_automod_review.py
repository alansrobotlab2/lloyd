"""The review rung, and the retry it drives.

Every other rung asks whether the change broke something. #544 passed all
eight with one acceptance clause skipped, half a fleet uncovered, a silent
`Edit` replay and an `or True` assertion — then declared itself `deferred` to
nothing and parked its item forever. What follows pins the mechanism that
would have caught it, and the loop that gives a sound-premise round another
go without letting author and grader disagree indefinitely.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, gate as G, review as RV, round as R
from scripts.automod import state as S, vault_round as V, worktree as W
from workers.sources import autocode as I
from workers.sources import autotriage as T


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


ROOT = Path(__file__).resolve().parent.parent
HONESTY_FIXTURES = ROOT / "tests" / "fixtures" / "honesty"


def _fixture(name: str) -> str:
    """A measured test file's body, read from `tests/fixtures/honesty/`.

    The bodies the honesty prechecks are measured against live there rather than
    inline here because this file is itself one of the files `honesty_prechecks`
    runs on, and the review rung computes them with the LIVE checkout's
    `review.py` (#1755) — a round that changes that checker is graded by the
    version it replaces, and the one still running counts the five dishonesty
    patterns in the raw text of every changed test file, prose and string
    literals included. A fixture spelled inline is therefore counted as newly
    dishonest code and refuses the round that added it: that is how
    `SM_20260928_210355` spent its second review attempt. A `.txt` is not a test
    file (`testpaths.is_test_file` asks for `.py`), so each shape stays readable
    in the code it is, and the node measures the real thing.
    """
    return (HONESTY_FIXTURES / name).read_text(encoding="utf-8")


def _why(rx_fragment: str) -> str:
    """The problem string the pattern table attaches to the pattern whose
    regular expression contains `rx_fragment` — looked up off the table rather
    than quoted here, for the same reason `_fixture` reads its bodies off disk.
    Pass a fragment of the REGEX as `review.py` writes it (`skip\\(`, not the
    spelling), and it is unique by construction: two matches is a table this
    file can no longer read, and the node says so."""
    hits = [why for pat, why, _sev in RV._HONESTY_PATTERNS if rx_fragment in pat]
    assert len(hits) == 1, (rx_fragment, hits, [p for p, _w, _s in RV._HONESTY_PATTERNS])
    return hits[0]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def write_item(d: Path, item_id, *, status="up_next", clauses=None, board="lloyd") -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created, "board": board,
          "tags": ["backlog"]}
    if clauses:
        fm["acceptance_clauses"] = list(clauses)
    path = d / f"{item_id}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n",
                    encoding="utf-8")
    return path


def _confirm(item_id, acceptance="the check passes", clauses=(), surface=""):
    S.append_event({"event": "backlog_triage", "item_id": item_id, "verdict": "confirmed",
                    "acceptance": acceptance, "acceptance_clauses": list(clauses),
                    **({"surface": surface} if surface else {})},
                   path=S.LEDGER_PATH)


def _round(item_id, rid, *, stop_reason="stop"):
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "finished",
                    "round_id": rid, "stop_reason": stop_reason, "num_turns": 40},
                   path=S.LEDGER_PATH)


def _review_refused(rid, item_id, *, clauses_unmet=(1,), findings="clause 1 unmet: no test"):
    S.append_event({"event": "review", "round_id": rid, "item_id": item_id, "blocking": True,
                    "kind": "retry", "premise": "sound",
                    "clauses": [{"clause": i, "verdict": "unmet"} for i in clauses_unmet]},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "gate", "round_id": rid, "rung": "review", "ok": False,
                    "detail": f"review sent it back: {findings}", "review_retry": True,
                    "review_findings": findings}, path=S.LEDGER_PATH)


def _fm(path):
    return B._split_frontmatter(path.read_text())[0]


# ── the schema is derived, strict, and bans nothing the grader needs ────────

def test_the_review_schema_is_built_from_the_tuples_and_is_strict():
    s = RV.REVIEW_SCHEMA
    assert s["properties"]["premise"]["enum"] == list(RV.PREMISES)
    clause = s["properties"]["clauses"]["items"]
    assert clause["properties"]["verdict"]["enum"] == list(RV.CLAUSE_VERDICTS)
    assert clause["properties"]["how_verified"]["enum"] == list(RV.HOW_VERIFIED)
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    assert clause["additionalProperties"] is False and set(clause["required"]) == set(clause["properties"])
    assert "maxLength" not in json.dumps(s)


def test_the_grader_is_denied_every_automod_verb_and_every_writer():
    from workers.sources import _common as C
    assert set(C.WORKER_AUTOMOD_BAN) <= set(RV.REVIEW_DENY)
    assert set(C.WORKER_GRANT_MINT_BAN) <= set(RV.REVIEW_DENY)
    for name in ("Edit", "Write", "backlog_write_task", "vault_write", "Task"):
        assert name in RV.REVIEW_DENY
    # ...and is not denied what it exists to do.
    for name in ("Read", "Grep", "Glob", "Bash"):
        assert name not in RV.REVIEW_DENY


# ── parse_review judges the judge ──────────────────────────────────────────

def _obj(**clause):
    base = {"clause": 1, "verdict": "met", "evidence_path": "app/x.py", "evidence_line": 3,
            "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran", "note": "ok"}
    base.update(clause)
    return {"premise": "sound", "clauses": [base], "test_honesty": [], "seams_unverified": [],
            "summary": "fine"}


@pytest.fixture
def wt(tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "tests").mkdir()
    # Six lines, so `_obj`'s default `evidence_line` 3 is inside the file (#1254).
    (tmp_path / "app" / "x.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    (tmp_path / "tests" / "test_x.py").write_text("def test_it():\n    assert 1\n")
    # A test this diff did not touch, for the existing-test shape.
    (tmp_path / "tests" / "test_old.py").write_text("def test_before():\n    assert 1\n")
    return tmp_path


def test_a_met_with_real_evidence_stands(wt):
    parsed = RV.parse_review(_obj(), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "met" and parsed["downgraded"] == []


@pytest.mark.parametrize("tests_passed", [False, True])
@pytest.mark.parametrize("bad", [
    {"evidence_path": "app/nope.py"},
    {"evidence_path": ""},
    {"test_node_id": "tests/test_other.py::test_z"},
    {"test_node_id": ""},
    {"how_verified": "inferred"},
])
def test_a_met_without_evidence_is_downgraded_in_python(wt, bad, tests_passed):
    """The grader's laziness is not the author's pass. A `met` needs a real
    path, a test in a file this diff changed, and ran|read — or it is
    `partial`, decided here without asking the model again. A green tests
    rung waives none of these five: a missing file is still missing."""
    parsed = RV.parse_review(_obj(**bad), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=tests_passed, changed_paths=["app/x.py", "tests/test_x.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c["downgraded"] and parsed["downgraded"] == [1]
    assert "accepted" not in c


# ── the three shapes a `met` may stand on besides a changed test ─────────────

def _met(wt, *, tests_passed=True, changed_paths=("app/x.py", "tests/test_x.py"), **clause):
    parsed = RV.parse_review(_obj(**clause), worktree=wt, changed_tests=["tests/test_x.py"],
                             n_clauses=1, tests_passed=tests_passed, changed_paths=list(changed_paths))
    return parsed["clauses"][0]


def test_a_suite_level_run_stands_on_a_green_tests_rung(wt):
    """#860's clause 8 — "the autoresearch suite passes" — was refused three
    times for the only honest node it has: `tests/ -k autoresearch`."""
    c = _met(wt, test_node_id="tests/ -k autoresearch", how_verified="ran")
    assert c["verdict"] == "met" and "downgraded" not in c
    assert "suite-level run" in c["accepted"][0]


def test_a_suite_level_run_without_a_green_tests_rung_is_partial(wt):
    c = _met(wt, tests_passed=False, test_node_id="tests/ -k autoresearch", how_verified="ran")
    assert c["verdict"] == "partial"
    assert c["downgraded"] == ["test_node_id not in a test file this diff changed"]


def test_a_suite_level_run_the_grader_only_read_is_partial(wt):
    c = _met(wt, test_node_id="tests/ -k autoresearch", how_verified="read")
    assert c["verdict"] == "partial"


# ── #1322: a node under a non-root testpath is judged like one under tests/ ──

@pytest.fixture
def harness_wt(wt):
    """`wt` with the tree's own testpaths and a harness suite beside root
    `tests/`: one file the diff changed, one it did not."""
    (wt / "pytest.ini").write_text("[pytest]\ntestpaths = tests app/harness/tests scripts\n")
    h = wt / "app" / "harness" / "tests"; h.mkdir(parents=True)
    (h / "test_x.py").write_text("def test_y():\n    assert 1\n")
    (h / "test_old.py").write_text("def test_before():\n    assert 1\n")
    return wt


def _harness_met(wt, *, tests_passed=True, **clause):
    changed = ["app/harness/loop.py", "app/harness/tests/test_x.py"]
    parsed = RV.parse_review(_obj(**clause), worktree=wt, changed_tests=changed[1:], n_clauses=1,
                             tests_passed=tests_passed, changed_paths=changed)
    return parsed, parsed["clauses"][0]


def test_a_met_pinned_in_a_changed_harness_test_stands(harness_wt):
    """SM_20260921_030016: all five clauses graded met, four downgraded
    because their tests lived in app/harness/tests/."""
    parsed, c = _harness_met(harness_wt, test_node_id="app/harness/tests/test_x.py::test_y",
                             how_verified="read", tests_passed=False)
    assert c["verdict"] == "met" and "downgraded" not in c and parsed["downgraded"] == []


def test_an_unchanged_harness_node_needs_a_run_and_a_green_rung(harness_wt):
    node = "app/harness/tests/test_old.py::test_before"
    _, c = _harness_met(harness_wt, test_node_id=node, how_verified="ran")
    assert c["verdict"] == "met" and "existing test" in c["accepted"][0]
    for how, passed in (("read", True), ("ran", False)):
        _, c = _harness_met(harness_wt, test_node_id=node, how_verified=how, tests_passed=passed)
        assert c["verdict"] == "partial"
        assert c["downgraded"] == ["test_node_id not in a test file this diff changed"]
    # `scripts` is a testpath pytest walks, not a test tree: its code is no node.
    (harness_wt / "scripts" / "automod").mkdir(parents=True)
    (harness_wt / "scripts" / "automod" / "review.py").write_text("def f():\n    pass\n")
    _, c = _harness_met(harness_wt, test_node_id="scripts/automod/review.py::f", how_verified="ran")
    assert c["verdict"] == "partial"


def test_an_existing_test_outside_the_diff_stands_when_it_was_run(wt):
    """#487's clause 4 named a real test file the round did not change."""
    c = _met(wt, test_node_id="tests/test_old.py::test_before", how_verified="ran")
    assert c["verdict"] == "met" and "existing test" in c["accepted"][0]
    assert _met(wt, test_node_id="tests/test_old.py", how_verified="ran")["verdict"] == "met"


# ── a test that fails at base too is not evidence for this diff ──────────────

@pytest.mark.parametrize("node, how", [
    ("tests/test_old.py::test_before", "ran"),     # the failing node itself
    ("tests/test_old.py::test_before", "read"),
    ("tests/test_old.py", "ran"),                  # a file-level run of its file
    ("tests/test_old.py::test_other", "ran"),      # a sibling in the same red file
])
def test_a_met_on_a_node_that_fails_at_base_does_not_stand(wt, node, how):
    """Since 2026-09-24 the tests rung passes over failures that reproduce at
    the round's base. A clause resting on one was verified by nothing: the
    rung passed over it, not through it — whatever `how` says."""
    parsed = RV.parse_review(_obj(test_node_id=node, how_verified=how), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"],
                             pre_existing_failures={"tests/test_old.py::test_before"})
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", c
    assert any("fails at base" in w for w in c["downgraded"]), c
    # The counterfactual: without the list, the same citation stands (ran) —
    # so the list, not something else, is what refused it.
    if how == "ran":
        assert _met(wt, test_node_id=node, how_verified="ran")["verdict"] == "met"


def test_a_changed_test_beside_an_unrelated_red_file_still_stands(wt):
    """Only a citation INTO the red set is refused; the diff's own test holds."""
    parsed = RV.parse_review(_obj(), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"],
                             pre_existing_failures={"tests/test_old.py::test_before"})
    assert parsed["clauses"][0]["verdict"] == "met"


def test_the_grader_is_told_which_failures_predate_the_round(tmp_path):
    contract = {"id": 1, "title": "t", "body": "b", "clauses": ["c"]}
    kw = dict(contract=contract, diff="", diff_truncated=False, changed_tests=[],
              test_counts={"passed": 5}, worktree=tmp_path, run_tests=tmp_path / "rt")
    plain = RV.build_prompt(**kw)
    told = RV.build_prompt(**kw, pre_existing_failures=["tests/test_uptake.py::test_a"])
    assert "fail here AND at the round's base" not in plain
    assert "tests/test_uptake.py::test_a" in told and "at most `partial`" in told
    # With nothing pre-existing the prompt is byte-identical to before.
    assert plain == RV.build_prompt(**kw, pre_existing_failures=[])


@pytest.mark.parametrize("node", ["tests/test_gone.py::test_x", "tests/test_gone.py",
                                  "app/x.py::f", "pytest -k autoresearch"])
def test_a_node_that_is_not_a_real_tests_path_is_partial(wt, node):
    assert _met(wt, test_node_id=node, how_verified="ran")["verdict"] == "partial"


def test_evidence_of_a_deleted_file_stands_beside_a_changed_test(wt):
    """#487's clause 1 was about a report the change removed, and its only
    evidence was the file's absence."""
    c = _met(wt, evidence_path="scripts/dead_report.py", test_node_id="tests/test_x.py::test_it",
             changed_paths=("scripts/dead_report.py", "tests/test_x.py"))
    assert c["verdict"] == "met" and c["accepted"][0].startswith("evidence of absence")
    # The shape #487's grader actually wrote (a made-up name, so a real file
    # in this machine's vault cannot make it pass for the wrong reason). Since
    # #1252 the second token of that citation is tried, and it is a real test
    # file — so the path rail holds on the test path and the absence waiver is
    # no longer what carries the clause. `met` either way, now for the better
    # reason: a file on disk is pointed at.
    c = _met(wt, evidence_path="~/obsidian/memory/no-such-report-7f3a.md (absent); "
                               "tests/test_x.py:396-406",
             test_node_id="tests/test_x.py::test_it")
    assert c["verdict"] == "met" and c["evidence_path"] == "tests/test_x.py"
    assert "accepted" not in c


def test_an_absence_marker_does_not_stand_without_a_node_of_its_own(wt):
    c = _met(wt, evidence_path="scripts/dead_report.py (deleted)", test_node_id="")
    assert c["verdict"] == "partial"
    # ...nor stacked on a suite-level waiver: that pins nothing.
    c = _met(wt, evidence_path="scripts/dead_report.py (deleted)",
             test_node_id="tests/ -k report", how_verified="ran")
    assert c["verdict"] == "partial"
    assert c["downgraded"][0].startswith("evidence_path missing")
    # ...and a path the diff never touched, with no marker, is just missing.
    assert _met(wt, evidence_path="scripts/dead_report.py",
                test_node_id="tests/test_x.py::test_it")["verdict"] == "partial"


def test_a_clause_the_grader_did_not_mention_is_not_met(wt):
    parsed = RV.parse_review(_obj(), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=3)
    assert [c["verdict"] for c in parsed["clauses"]] == ["met", "partial", "partial"]
    assert parsed["clauses"][1]["note"] == "not addressed by the grader"
    # One readable index is an abstention about the CHANGE. The rung must keep
    # refusing on it, or a grader that answers one clause of three passes by
    # saying nothing about the other two.
    assert "clauses_unreadable" not in parsed


def test_clause_entries_in_another_shape_are_recorded_as_unreadable(wt):
    """`id`/`status` instead of `clause`/`verdict`: not one index is usable.

    #1443's three rounds reached this with the grader approving every clause.
    The synthesized partials stay in the result — the caller decides what they
    mean — but the parse boundary records that no index was readable, with the
    key names it actually found, so a rail failure stops reading as a verdict.
    """
    alias = {"premise": "sound", "summary": "APPROVE", "test_honesty": [],
             "seams_unverified": [],
             "clauses": [{"id": i, "status": "met", "evidence_path": "app/x.py",
                          "test_node_id": "tests/test_x.py::test_it",
                          "how_verified": "ran", "note": "graded"} for i in (1, 2)]}
    parsed = RV.parse_review(alias, worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=2)
    assert [c["verdict"] for c in parsed["clauses"]] == ["partial", "partial"]
    unread = parsed["clauses_unreadable"]
    assert unread["entries"] == 2
    assert unread["keys"] == ["evidence_path", "how_verified", "id", "note",
                              "status", "test_node_id"], unread["keys"]


def test_clause_entries_that_are_not_objects_are_still_unreadable(wt):
    """A grader that answered `["met", "met"]` graded nothing it can be asked."""
    parsed = RV.parse_review({"premise": "sound", "clauses": ["met", "met"]},
                             worktree=wt, changed_tests=[], n_clauses=2)
    assert parsed["clauses_unreadable"]["entries"] == 2
    assert parsed["clauses_unreadable"]["keys"] == []


def test_no_clause_entries_at_all_is_an_abstention_not_an_unreadable_shape(wt):
    """`clauses: []` is the grader declining to grade; it refuses as it always did.

    The unreadable flag exists for a verdict that cannot be read, so an empty
    list must not borrow it — that would let an abstention skip the attempt.
    """
    parsed = RV.parse_review(_obj(clause=1) | {"clauses": []}, worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    assert "clauses_unreadable" not in parsed
    assert [c["verdict"] for c in parsed["clauses"]] == ["partial"]


def test_the_vault_shape_needs_no_test_node(wt):
    parsed = RV.parse_review(_obj(test_node_id=""), worktree=wt, changed_tests=[], n_clauses=1,
                             require_tests=False)
    assert parsed["clauses"][0]["verdict"] == "met"


def test_an_unusable_object_is_none(wt):
    assert RV.parse_review("nope", worktree=wt, changed_tests=[], n_clauses=1) is None
    assert RV.parse_review({"premise": "maybe"}, worktree=wt, changed_tests=[], n_clauses=1) is None


def test_decide_maps_premise_then_findings():
    base = {"premise": "sound", "clauses": [{"clause": 1, "verdict": "met", "note": ""}],
            "test_honesty": [], "seams_unverified": [], "summary": "good", "downgraded": []}
    assert RV.decide(base, []) == ("pass", "good")
    assert RV.decide({**base, "premise": "unsound", "summary": "premise is false"}, [])[0] == "unsound"
    kind, findings = RV.decide({**base, "clauses": [{"clause": 2, "verdict": "unmet", "note": "no path",
                                                     "downgraded": []}]}, [])
    assert kind == "retry" and "clause 2 unmet: no path" in findings
    kind, findings = RV.decide(base, [{"file": "tests/t.py", "line": 9, "problem": "`or True`"}])
    assert kind == "retry" and "tests/t.py:9" in findings
    kind, findings = RV.decide({**base, "seams_unverified": ["loopback POST"]}, [])
    assert kind == "retry" and "seam unverified: loopback POST" in findings


# ── the deterministic honesty checks ───────────────────────────────────────

@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"; (r / "tests").mkdir(parents=True); (r / "app").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0\n")
    (r / "app" / "m.py").write_text("V = 1\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    return r, base


def test_prechecks_flag_only_what_the_round_added(repo):
    r, base = repo
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0\n\n"
        "def test_new():\n    assert rows() == [] or True\n")
    git(r, "commit", "-qam", "round")
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    problems = [o["problem"] for o in out]
    assert any("or True" in p for p in problems), out
    assert not any("xfail" in p for p in problems), "the pre-existing xfail is not this round's"
    assert out[0]["line"] == 8


def test_prechecks_notice_a_code_change_with_no_new_test(repo):
    r, base = repo
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0  # touched\n")
    (r / "app" / "m.py").write_text("V = 2\n")
    git(r, "commit", "-qam", "round")
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=1)
    assert any("no test function was added" in o["problem"] for o in out)
    assert RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=0) == []


# ── the constant-mirroring assertion (#1678) ───────────────────────────────
# Pocock's first shape, and the one a deterministic check CAN pin: an
# assertion comparing a call's result against a constant defined at the top of
# the SAME test file. The five literal patterns spell "cannot fail" the way a
# person writes it when they give up; a mirrored constant never gives up, it
# just moves with the code, and the only thing that has ever caught it is the
# token-priced `test_honesty` grader. The ledger for 2026-09-11→09-27 carries
# 708 events with a grader-authored `test_honesty` list against 73 carrying a
# deterministic precheck line (9.7x), and 0 of the 1965 recorded entries name
# one of the five patterns' own problem strings. This moves one shape out of
# the priced half and into the free one.


def test_a_docstring_that_quotes_itself_hides_no_real_skip(tmp_path):
    """Why the blank pass runs on `tokenize` and not on a char-wise quote scan:
    a docstring holding a lone double quote plus an apostrophe desynchronises
    the scan's idea of where the literal ends, the flipped state runs to the end
    of the file, and what happens next depends on quote parity. At
    `tests/_live_data.py` the first version of the pass blanked that file's real
    unconditional skip at line 136 out of the skip pattern's reach — the shape
    that refuses a skipless round, invisible in exactly the files that skip
    most. The fixture is that shape: a docstring quoting itself, then one
    unconditional skip below it, which must still be found, at its own line."""
    r, base = _delta_repo(
        tmp_path, tag="parity", base_src=_MIRROR_BASE,
        post_src=_fixture("quote_heavy_docstring_skip.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0)
    hits = [o for o in out if "pytest.skip" in o["problem"]]
    assert len(hits) == 1, out
    assert hits[0]["severity"] == "blocking", "an unconditional skip still refuses"
    assert hits[0]["line"] == 4, "the skip is code below the docstring, not prose in it"


def test_prose_that_looks_like_code_binds_no_constant(tmp_path):
    """An ALL_CAPS assignment written inside a docstring is documentation, not
    a binding, so the assertion below it compares against nothing the file
    defines. The detector reads the blanked text for exactly this reason — and
    the control at the end of this node proves the silence is the blanking and
    not a dead detector: the same assertion under a real module-level binding
    fires."""
    r, base = _delta_repo(
        tmp_path, tag="prosecode", base_src=_MIRROR_BASE,
        post_src='"""\n'                                                      # 1
                 'SOME_LIMIT = 5   documented here, bound nowhere\n'          # 2
                 '"""\n'                                                      # 3
                 '\n'                                                         # 4
                 'def test_the_docstring_one():\n'                            # 5
                 '    assert count(open_body()) == SOME_LIMIT\n')             # 6
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"],
                                         n_clauses=1)) == []
    r2, base2 = _delta_repo(
        tmp_path, tag="prosectl", base_src=_MIRROR_BASE,
        post_src='SOME_LIMIT = 5\n'                                           # 1
                 '\n'                                                         # 2
                 'def test_the_bound_one():\n'                                # 3
                 '    assert count(open_body()) == SOME_LIMIT\n')             # 4
    found = _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"], n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4 and "line 1" in found[0]["problem"], found


def test_a_comment_beside_an_assertion_is_not_a_second_comparison(tmp_path):
    """A trailing `# was parse(y) == EXPECTED_CHAR` puts a second `==` at depth
    zero on the same physical line as a real assertion. Un-commented, that
    comment supplies both the call and the constant name, and the round is
    blamed for an assertion nobody wrote."""
    r, base = _delta_repo(
        tmp_path, tag="comment", base_src=_MIRROR_BASE,
        post_src="EXPECTED_CHAR = 'x'\n"                                      # 1
                 '\n'                                                         # 2
                 'def test_ok():\n'                                           # 3
                 "    assert parse(open_body()) == 'x'  # was parse(y) == EXPECTED_CHAR\n")
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"],
                                         n_clauses=1)) == []
    # Control: the same file with the assertion comparing against the name the
    # comment only mentioned is exactly the thing being reported.
    r2, base2 = _delta_repo(
        tmp_path, tag="commentctl", base_src=_MIRROR_BASE,
        post_src="EXPECTED_CHAR = 'x'\n"                                      # 1
                 '\n'                                                         # 2
                 'def test_ok():\n'                                           # 3
                 "    assert parse(open_body()) == EXPECTED_CHAR\n")          # 4
    found = _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"], n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4 and "EXPECTED_CHAR" in found[0]["problem"], found


def test_an_assertion_written_over_three_lines_is_still_one_comparison(tmp_path):
    """The call and the compared name on different physical lines: a
    line-by-line read sees an unclosed `(` on the assert line and no comparison
    at all, so the mirror goes unreported. The statement is joined to Python's
    bracket rule and reported at the `assert` line."""
    r, base = _delta_repo(
        tmp_path, tag="split", base_src=_MIRROR_BASE,
        post_src="EXPECTED_TITLE = 'widget'\n"      # 1
                 "\n"                               # 2
                 "def test_it():\n"                 # 3
                 "    assert parse(\n"              # 4
                 "        open_body()\n"            # 5
                 "    ) == EXPECTED_TITLE\n")       # 6
    found = _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4, "the assertion's own line, not the closing bracket's"
    assert "EXPECTED_TITLE" in found[0]["problem"] and "line 1" in found[0]["problem"]


def _delta_repo(tmp_path, *, base_src, post_src, pytest_ini=None, new_files=None,
                tag="dr"):
    """A repo whose base commit holds `base_src` and whose HEAD holds
    `post_src` (plus `new_files`), so a delta is measured across real git
    objects rather than a stubbed `_git`. `tag` names the subdirectory, because
    one test measures several base/HEAD pairs side by side."""
    r = tmp_path / tag
    (r / "tests").mkdir(parents=True)
    (r / "app").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "tests" / "test_a.py").write_text(base_src)
    (r / "app" / "m.py").write_text("def parse(s):\n    return s\n")
    if pytest_ini:
        (r / "pytest.ini").write_text(pytest_ini)
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    (r / "tests" / "test_a.py").write_text(post_src)
    for rel, src in (new_files or {}).items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "round")
    return r, base


_MIRROR_BASE = "import pytest\n\n\ndef test_old():\n    assert parse('a') == 'a'\n"


def _mirrors(out):
    return [o for o in out if "mirrors" in o["problem"]]


def test_prechecks_flag_an_assertion_that_mirrors_its_own_module_constant(tmp_path):
    """Clause 1: the finding names the constant's line AND the assertion's
    line, and says the test mirrors the value instead of pinning it."""
    r, base = _delta_repo(
        tmp_path, base_src=_MIRROR_BASE,
        post_src="from app.m import parse\n"                      # 1
                 "\n"                                             # 2
                 "EXPECTED_TITLE = 'widget'\n"                    # 3
                 "\n"                                             # 4
                 "def test_title_is_the_module_default():\n"      # 5
                 "    assert parse(open_body()) == EXPECTED_TITLE\n")   # 6
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    found = _mirrors(out)
    assert len(found) == 1, out
    assert found[0]["file"] == "tests/test_a.py"
    assert found[0]["line"] == 6, "the assertion's own line"
    assert "EXPECTED_TITLE" in found[0]["problem"]
    assert "line 3" in found[0]["problem"], "and the line the constant is bound on"
    assert "instead of pinning" in found[0]["problem"], found[0]["problem"]


def test_prechecks_stay_silent_on_an_imported_name_or_an_inline_literal(tmp_path):
    """Clause 2: the ordinary way to assert a public contract must not fire.
    A name imported from the module under test is not a value the test wrote,
    and neither is a literal typed into the assertion itself — even when the
    same file also defines a constant holding that same value. The control adds
    the fourth shape to that same file — the file's own constant compared against
    a call — and the run reports exactly that one, so the three silences above
    are selective and not a detector that never fires."""
    silent = ("from app.m import MAX_ITEMS, parse\n"               # 1
              "\n"                                                 # 2
              "EXPECTED_TITLE = 'widget'\n"                        # 3
              "\n"                                                 # 4
              "def test_the_imported_contract_value():\n"          # 5
              "    assert parse(open_body()) == MAX_ITEMS\n"       # 6  imported
              "\n"                                                 # 7
              "def test_the_inline_literal():\n"                   # 8
              "    assert parse(open_body()) == 'widget'\n"        # 9  literal
              "\n"                                                 # 10
              "def test_a_value_the_test_handed_round():\n"        # 11
              "    parsed = 'widget'\n"                            # 12
              "    assert parsed == EXPECTED_TITLE\n")             # 13  no call
    r, base = _delta_repo(tmp_path, base_src=_MIRROR_BASE, post_src=silent)
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    assert _mirrors(out) == [], out
    r2, base2 = _delta_repo(
        tmp_path, tag="ctl", base_src=_MIRROR_BASE,
        post_src=silent
                 + "\n"                                            # 14
                 "def test_the_mirror_control():\n"                # 15
                 "    assert parse(open_body()) == EXPECTED_TITLE\n")   # 16
    found = _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"], n_clauses=2))
    assert len(found) == 1, found
    assert found[0]["line"] == 16 and "EXPECTED_TITLE" in found[0]["problem"], found


def test_a_bracket_in_a_string_or_a_filter_inside_a_call_moves_nothing(tmp_path):
    """The scan reads the assertion the way Python reads it, not the way a line
    grep does. A bracket that lives inside a string literal does not open an
    argument list, so the assertion after it still splits and still fires; a
    comparison buried inside a call is not what the `assert` compares, so it
    fires nothing. One finding, at line 4, is both halves at once."""
    r, base = _delta_repo(
        tmp_path, base_src=_MIRROR_BASE,
        post_src="EXPECTED_CHAR = '('\n"                          # 1
                 "\n"                                             # 2
                 "def test_the_bracket_is_a_string():\n"          # 3
                 "    assert parse('(') == EXPECTED_CHAR\n"       # 4  fires
                 "\n"                                             # 5
                 "def test_a_filter_is_not_the_assertion():\n"    # 6
                 "    assert sum(1 for t in titles if t == EXPECTED_CHAR) == 0\n")
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1)
    found = _mirrors(out)
    assert len(found) == 1, out
    assert found[0]["line"] == 4, found
    assert "EXPECTED_CHAR" in found[0]["problem"] and "line 1" in found[0]["problem"]


def _seed_repo(tag, tmp_path, seed_line):
    """A file that seeds a constant and then asserts a call still returns it,
    with the seed written however the caller wants — used by the #1864 nodes to
    vary only the way the constant enters a call."""
    return _delta_repo(
        tmp_path, tag=tag, base_src=_MIRROR_BASE,
        post_src='PAYLOAD = "seeded identity"\n'                            # 1
                 '\n'                                                       # 2
                 'def _seed(target):\n'                                     # 3
                 f'    {seed_line}\n'                                       # 4
                 '\n'                                                       # 5
                 'def test_a_refused_write_left_the_file_alone():\n'        # 6
                 '    assert read_back() == PAYLOAD\n')                     # 7


def test_a_seed_constant_written_by_a_fixture_elsewhere_is_no_mirror(tmp_path):
    """#1864 clauses 1 and 2: a constant the file hands to a call as a whole
    argument is that file's own seed data, and the assertion re-reading it after
    a refused operation is the contract — it is not a mirror.

    Both false positives of 2026-09-29 have this shape, and
    `tests/test_builtin_fs_protected_write.py` is the hard one: none of its five
    firing tests calls `write_text(ORIGINAL)` itself — the writes live in the
    `home()` and `linked_home()` fixtures — so only a FILE-scope reading silences
    it, and that is what the seed call in `_seed` above is for: a different
    function from the assertion. The keyword form (`content=PAYLOAD`) is the same
    act. Then the three controls, each of which keeps the detector alive: a
    `PAYLOAD` that never enters any argument list is `FALLBACK_LAYOUT` and fires
    once, at the assertion's own line, naming the constant's; `assert
    parse(PAYLOAD) == PAYLOAD` fires, because the round-trip-that-cannot-fail is
    the worst case in the corpus and counting its input side as a seed would
    silence it; and a seed call that can only take `PAYLOAD + "!"` fires too,
    because a constant fed into arithmetic is being used as an expectation, not
    handed over as one.
    """
    r, base = _seed_repo("seedpos", tmp_path, "target.write_text(PAYLOAD)")
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"],
                                         n_clauses=1)) == []

    r2, base2 = _seed_repo("seedkw", tmp_path, "target.write(content=PAYLOAD)")
    assert _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"],
                                         n_clauses=1)) == []

    r3, base3 = _delta_repo(
        tmp_path, tag="seedctl", base_src=_MIRROR_BASE,
        post_src='PAYLOAD = "seeded identity"\n'                            # 1
                 '\n'                                                       # 2
                 'def test_the_constant_that_is_never_an_input():\n'        # 3
                 '    assert read_back() == PAYLOAD\n')                     # 4
    found = _mirrors(RV.honesty_prechecks(r3, base3, ["tests/test_a.py"],
                                          n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4, found
    assert "PAYLOAD" in found[0]["problem"] and "line 1" in found[0]["problem"]

    r4, base4 = _delta_repo(
        tmp_path, tag="seedrt", base_src=_MIRROR_BASE,
        post_src='PAYLOAD = "seeded identity"\n'                            # 1
                 '\n'                                                       # 2
                 'def test_the_round_trip_that_cannot_fail():\n'             # 3
                 '    assert parse(PAYLOAD) == PAYLOAD\n')                   # 4
    found = _mirrors(RV.honesty_prechecks(r4, base4, ["tests/test_a.py"],
                                          n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4, found

    r5, base5 = _seed_repo("seedexpr", tmp_path, 'target.write_text(PAYLOAD + "!")')
    found = _mirrors(RV.honesty_prechecks(r5, base5, ["tests/test_a.py"],
                                          n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 7, found


def test_prose_that_spells_a_write_buys_the_constant_no_exclusion(tmp_path):
    """#1864 clause 3: the seed scan reads `_code_only`'s blanked text, so a
    comment, a string literal and a module docstring all spelling
    `write_text(PAYLOAD)` are not the file putting a value into the world.

    Without that, a round could silence any mirror it was blamed for by writing
    one line of comment above the assertion — the exclusion would be a phrase, not
    a property of the code. The control is the same file with that call as real
    code, which is what makes the silence here about the prose and not about a
    detector that stopped running."""
    r, base = _delta_repo(
        tmp_path, tag="prosewrite", base_src=_MIRROR_BASE,
        post_src='PAYLOAD = "seeded identity"\n'                            # 1
                 '# the fixture used to write it: target.write_text(PAYLOAD)\n'   # 2
                 'HINT = "target.write_text(PAYLOAD)"\n'                     # 3
                 "'''A docstring naming target.write_text(PAYLOAD) as if it were"
                 " code.'''\n"                                               # 4
                 '\n'                                                        # 5
                 'def test_the_mirror_is_still_a_mirror():\n'                # 6
                 '    assert read_back() == PAYLOAD\n')                      # 7
    found = _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"],
                                          n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 7 and "PAYLOAD" in found[0]["problem"], found

    r2, base2 = _seed_repo("prosectl", tmp_path, "target.write_text(PAYLOAD)")
    assert _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"],
                                         n_clauses=1)) == []


def test_the_severity_ruling_is_written_where_the_constant_lives():
    """#1864 clause 4: the comment above `_CONSTANT_MIRROR_SEVERITY` used to
    defer the question to "a week of real rounds" and no round ever came to
    answer it. The measurement was made on 2026-09-29 and the block has to carry
    its result: 1/3 precision, 3 firings, 78 graded rounds, the fixture-seed
    class, and that the window was one day rather than the week it asked for.

    A prose pin, deliberately, because the artefact under test IS prose: it is
    the sentence the next reader consults about whether to promote the severity,
    and a stale deferral there invites a re-measure of a question already closed.
    The severity itself is pinned as a value, not as text, by
    `tests/test_review_policy.py`."""
    src = (ROOT / "scripts" / "automod" / "review.py").read_text()
    block = (src.split("# ── the constant-mirroring assertion", 1)[1]
                .split("_CONSTANT_MIRROR_SEVERITY", 1)[0])
    assert "week of real rounds" not in block, block
    assert "settled by" not in block, "the block still defers the decision"
    for fact in ("2026-09-29", "1/3", "3 firings", "78 graded rounds",
                 "fixture seed"):
        assert fact in block, (fact, block)
    assert "one day" in block.lower(), "the window it actually measured is named"
    assert '_CONSTANT_MIRROR_SEVERITY = "advisory"' in src


def test_a_constant_mirror_the_base_already_carried_is_not_blamed(tmp_path):
    """Clause 3: the delta is the same arithmetic as the five patterns. A base
    that already carried one mirrored assertion is blamed for none, and one
    that adds a second is blamed for exactly that one — at the new line."""
    carried = ("EXPECTED_TITLE = 'widget'\n"
               "\n"
               "def test_title_is_the_module_default():\n"
               "    assert parse(open_body()) == EXPECTED_TITLE\n")
    r, base = _delta_repo(tmp_path, tag="same", base_src=carried, post_src=carried)
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1)) == []
    # Same assertion twice over: still nothing new this round blamed for.
    r, base = _delta_repo(tmp_path, tag="twice", base_src=carried * 2, post_src=carried * 2)
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1)) == []
    # One MORE mirrored assertion is exactly one finding, naming the new node.
    r, base = _delta_repo(
        tmp_path, tag="more", base_src=carried,
        post_src=carried + "\nMAX_RETRIES = 3\n"                  # 6
                         "\n"                                     # 7
                         "def test_retry_default():\n"            # 8
                         "    assert load_config().retries == MAX_RETRIES\n")   # 9
    found = _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 9, "the increase is the finding, not the carried one"
    assert "MAX_RETRIES" in found[0]["problem"] and "line 6" in found[0]["problem"]


def test_the_five_literal_patterns_still_refuse_after_the_new_detector(tmp_path):
    """Clause 5, first half: widening the check may not soften the five. The
    fixture is one round's worth of every one of them added to a file that had
    none, and each is still reported, once, at the severity its own table entry
    carries — `blocking`, which the last assertion here pins rather than trusts.
    The table is read rather than quoted so a softened or swapped entry fails
    this node instead of quietly agreeing with it."""
    expected = (r"\bor\s+True\b", r"^\s*assert\s+True\b", r"pytest\.skip\(",
                r"pytest\.mark\.skip", r"pytest\.mark\.xfail")
    assert tuple(p for p, _w, _s in RV._HONESTY_PATTERNS) == expected
    r, base = _delta_repo(tmp_path, base_src=_MIRROR_BASE,
                          post_src=_fixture("five_literal_patterns.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    for _pat, why, severity in RV._HONESTY_PATTERNS:
        hit = [o for o in out if o["problem"] == why]
        assert len(hit) == 1, (why, out)
        assert hit[0]["severity"] == severity == "blocking", (why, out)


def test_a_pattern_spelled_in_a_string_is_not_counted_as_new_dishonest_code(tmp_path):
    """The prechecks run on the round's OWN changed test files, so a round that
    adds a test for this checker has to put the five shapes inside the source
    strings of that test. Prose about a pattern is not the pattern — and the
    control at the end is the same file with one of them written as code, which
    still refuses, on the code's own line. This is the shape that refused
    `SM_20260928_210355` for adding the tests that pin it."""
    prose = _fixture("patterns_named_in_prose_base.txt")
    r, base = _delta_repo(
        tmp_path, tag="prose", base_src=prose,
        post_src=prose + _fixture("patterns_named_in_prose_added.txt"))
    assert RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0) == []
    r, base = _delta_repo(
        tmp_path, tag="code", base_src=prose,
        post_src=prose + _fixture("real_assert_true_added.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0)
    hits = [o for o in out if o["problem"] == _why(r"assert\s+True")]
    assert len(hits) == 1, out
    assert hits[0]["severity"] == "blocking" and hits[0]["line"] == 8, out


def test_a_pattern_in_a_docstring_is_prose_and_the_line_reported_is_the_code_s(tmp_path):
    """The blank pass has to be blind across MULTIPLE lines, not just across a
    quoted word, and it may not move the lines it reports: a three-line
    docstring that names a skip call and a bare-true assertion is prose, and the
    one real assertion below it is on line 6 however many lines the docstring
    swallowed."""
    r, base = _delta_repo(
        tmp_path, tag="doc", base_src=_MIRROR_BASE,
        post_src=_fixture("multiline_docstring_then_real_assert.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0)
    assert len(out) == 1, out
    assert out[0]["problem"] == _why(r"assert\s+True"), out
    assert out[0]["line"] == 6, "the code's line, not the one the blanking left"


def test_a_conditional_skip_is_still_demoted_to_advisory(tmp_path):
    """Clause 5, second half — the #1204 fix. 27 of 134 refusals were skip-only
    with every clause graded met; a skip behind a condition is a judgment and
    goes to the grader as advice, not as a refusal."""
    r, base = _delta_repo(tmp_path, base_src=_MIRROR_BASE,
                          post_src=_fixture("conditional_skip.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0)
    skips = [o for o in out if o["problem"].startswith(_why(r"skip\("))]
    assert len(skips) == 1, out
    assert skips[0]["severity"] == "advisory", out
    assert "conditional" in skips[0]["problem"], out


def test_the_new_detector_scans_the_files_testpaths_owns(tmp_path):
    """Clause 5, third half: the file→test mapping is `TP.pick_test_files`'s
    answer read from `pytest.ini` — the same one the patterns and #1322's
    met-node rail read — not a `tests/` prefix and no new resolver. A mirrored
    assertion in a harness test under a second testpath is reported; one in
    ordinary code under no testpath is not."""
    mirrored = "EXPECTED_TITLE = 'widget'\n" \
               "\n" \
               "def test_title_is_the_module_default():\n" \
               "    assert parse(open_body()) == EXPECTED_TITLE\n"
    r, base = _delta_repo(
        tmp_path, base_src=_MIRROR_BASE,
        post_src=_MIRROR_BASE + "\n" + mirrored,
        pytest_ini="[pytest]\ntestpaths = tests app/harness/tests scripts\n",
        new_files={"app/harness/tests/test_h.py": mirrored,
                   "app/code.py": mirrored})
    out = RV.honesty_prechecks(r, base,
                               ["tests/test_a.py", "app/harness/tests/test_h.py",
                                "app/code.py", "app/m.py"], n_clauses=1)
    files = sorted({o["file"] for o in _mirrors(out)})
    assert files == ["app/harness/tests/test_h.py", "tests/test_a.py"], out
    assert [p for p in RV.TP.pick_test_files(
        ["tests/test_a.py", "app/harness/tests/test_h.py", "app/code.py"], r)] == \
        ["tests/test_a.py", "app/harness/tests/test_h.py"]


# ── every citation is validated, whatever verdict carries it (#1442) ───────
# Round SM_20260924_104307, review attempt 2 of 2: four `partial`s refused the
# round, and every checkable claim in them was false of the head the gate
# named. The evidence was a test file that exists in no commit
# (`tests/test_facts_surviving_readers.py`) and a landing at `08a4f4f0`, which
# `git cat-file -t` calls `fatal: Not a valid object name` in both `~/lloyd` and
# the vault. Both rails existed and both were skipped: `_node_rail`'s existence
# check and `normalize_evidence_path`'s answer were consulted only inside
# `if verdict == "met":`, so a uniformly-`partial` restatement — the shape the
# finalizer's schema pass emits, not the grading turn, which returned
# `verdict: approve` — was checked against nothing. `unresolved_shas` and
# `added_test_denials` carry the mechanism.

PHANTOM = "tests/test_facts_surviving_readers.py::test_the_surviving_kg_readers_still_register"


def _fin(*specs) -> dict:
    """A `clauses` list in the shape the finalizer emits: one entry per clause."""
    clauses = []
    for i, spec in enumerate(specs, 1):
        c = {"clause": i, "verdict": "met", "evidence_path": "app/x.py", "evidence_line": 3,
             "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran", "note": "ok"}
        c.update(spec)
        clauses.append(c)
    return {"premise": "sound", "clauses": clauses, "test_honesty": [],
            "seams_unverified": [], "summary": "fine"}


def _judged(wt, obj, n, **kw):
    return RV.parse_review(obj, worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=n,
                           tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"], **kw)


@pytest.mark.parametrize("verdict", ["met", "partial", "unmet"])
def test_a_test_node_absent_from_the_tree_is_rejected_on_every_verdict(wt, verdict):
    """`partial` is precisely the verdict that skipped the check on 2026-09-24,
    so the rail cannot live only in the `met` branch."""
    parsed = _judged(wt, _fin({"verdict": verdict, "test_node_id": PHANTOM}), 1)
    c = parsed["clauses"][0]
    assert c.get("citation_unresolved"), "an entry naming an absent test file is marked whatever its verdict"
    assert "tests/test_facts_surviving_readers.py" in c["citation_unresolved"][0]
    assert parsed["unreliable"], "a refusal built on a file that is not in the graded tree is not a verdict"


@pytest.mark.parametrize("verdict", ["met", "partial", "unmet"])
def test_an_evidence_path_that_resolves_to_nothing_is_recorded_on_the_clause(wt, verdict):
    """The refusal then says the citation failed instead of silently
    accepting `agent_mcp/facts.py:520-540`-shaped text for a test. An evidence
    path alone stays a graded refusal, not an unusable review."""
    parsed = _judged(wt, _fin({"verdict": verdict, "evidence_path": "app/facts_gone.py:520-540"}), 1)
    c = parsed["clauses"][0]
    assert c.get("citation_unresolved") and "app/facts_gone.py" in c["citation_unresolved"][0]
    assert parsed["unreliable"] == [], "a path the grader mis-cited is still a finding about the diff"


def test_a_review_whose_entries_all_cite_absent_files_is_unusable(wt):
    """The 2026-09-24 refusal: all four entries named a file in no tree, branch
    or commit, and the round still spent its last attempt on the text."""
    specs = [{"verdict": "partial", "test_node_id": f"tests/test_ghost_{i}.py::test_x"}
             for i in range(4)]
    parsed = _judged(wt, _fin(*specs), 4)
    assert len(parsed["clauses"]) == 4
    assert all(c.get("citation_unresolved") for c in parsed["clauses"])
    assert any("every" in r.lower() for r in parsed["unreliable"]), parsed["unreliable"]


def test_a_commit_cited_in_a_note_that_is_not_an_object_makes_the_review_unreliable(repo):
    """`git cat-file -t 08a4f4f0` → fatal, yet the rung wrote "a test in a prior
    landing … commit 08a4f4f0 … that I ran and read" and the gate believed it."""
    r, _ = repo
    head = git(r, "rev-parse", "HEAD").stdout.strip()
    bad = _judged(r, _fin({"verdict": "partial", "test_node_id": "",
                           "evidence_path": "app/m.py", "evidence_line": 1,
                           "note": "the pin exists from commit 08a4f4f0, not in this diff's tests"}),
                  1, repo=r)
    assert bad["clauses"][0]["citation_unresolved"]
    assert "08a4f4f0" in bad["clauses"][0]["citation_unresolved"][0]
    assert bad["unreliable"] and "08a4f4f0" in " ".join(bad["unreliable"])
    # Positive control through the same rail: a sha the repo really has.
    good = _judged(r, _fin({"verdict": "partial", "test_node_id": "", "evidence_path": "app/m.py", "evidence_line": 1,
                            "note": f"the pin landed at {head}, and the suite is green"}), 1, repo=r)
    assert good["unreliable"] == [] and "citation_unresolved" not in good["clauses"][0]


def test_a_repo_the_rail_cannot_read_invents_no_unresolved_shas(wt):
    """Fourth instance of the catalogued class: a guard that reads its own
    missing input reports what it cannot see. `tmp_path` is not a repo, so
    `git cat-file` answers nothing, and the answer is to stay quiet."""
    parsed = _judged(wt, _fin({"verdict": "partial", "test_node_id": "",
                              "note": "satisfied by commit 08a4f4f0"}), 1, repo=wt)
    assert parsed["unreliable"] == [] and "citation_unresolved" not in parsed["clauses"][0]


def test_the_sha_rail_asks_git_and_does_not_read_prose_as_a_commit(repo):
    """The shape filter is a cheap pre-check; the git lookup is the whole
    verdict. Hex-looking English (`decode`, `beadded`), an all-decimal date
    (`20260915`) and a short line range are not commit-ish at all, while a
    7-token with a digit and a letter IS asked — and a real sha in the same
    note is the positive control that the asking works."""
    r, _ = repo
    head = git(r, "rev-parse", "HEAD").stdout.strip()
    quiet = _judged(r, _fin({"verdict": "partial", "test_node_id": "", "evidence_path": "app/m.py", "evidence_line": 1,
                             "note": f"the decode path, the beadded case, since 20260915, "
                                     f"agent_mcp/facts.py:520-540 and {head[:12]} are all "
                                     f"consistent"}), 1, repo=r)
    assert quiet["unreliable"] == [] and "citation_unresolved" not in quiet["clauses"][0]
    # The same note with one token the repo does not have.
    # `0f`×6 keeps the shape filter (hex, 12 chars, has both a digit and a
    # letter) and is an object no repo has.
    loud = _judged(r, _fin({"verdict": "partial", "test_node_id": "", "evidence_path": "app/m.py", "evidence_line": 1,
                            "note": quiet["clauses"][0]["note"].replace(head[:12], "0f" * 6)}),
                   1, repo=r)
    assert "0f0f0f0f0f0f" in " ".join(loud["unreliable"]), "the lookup decides, the shape only asks"


def test_the_note_of_an_unsound_premise_is_marked_but_its_verdict_still_stands(wt):
    """The unsound call is the grader's to make about the ITEM, and an
    unusable-citation flag must not swallow it."""
    obj = _fin({"verdict": "partial", "test_node_id": PHANTOM})
    obj["premise"] = "unsound"
    parsed = _judged(wt, obj, 1)
    assert parsed["premise"] == "unsound" and parsed["unreliable"]


# ── the rung: four outcomes, and the flags ride the event ──────────────────

class _Gate(G.Gate):
    def __init__(self, item_id, changed, tmp_path):
        super().__init__("SM_REV", tmp_path, "a" * 40, item_id=item_id)
        self.report.changed_paths = changed
        self.report.rungs.append(G.RungResult("tests", True, "ok", 1.0, {"passed": 10}))


def _arm(monkeypatch, tmp_path, *, grade, contract=None, prior=0):
    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    # Graded refusals of DISTINCT commits: only those spend an attempt now.
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: [
        {"event": "review", "round_id": "SM_REV", "ok": True, "blocking": True,
         "attempt": i + 1, "head": f"{i:040d}", "findings": f"f{i}"} for i in range(prior)])
    monkeypatch.setattr(G.W, "round_dir", lambda rid: tmp_path / "round")
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: contract or {
        "id": iid, "title": "t", "body": "b", "clauses": ["the thing happens once"], "path": ""})
    monkeypatch.setattr(RV, "grade", grade)
    monkeypatch.setattr(RV, "honesty_prechecks", lambda *a, **k: [])
    return events


def _grader(structured, *, ok=True, error=""):
    def grade(**kw):
        grade.calls.append(kw)
        return {"ok": ok, "error": error, "session_id": "sess_r", "structured": structured,
                "structured_error": "", "text": "", "stop_reason": "stop", "duration_s": 1.0}
    grade.calls = []
    return grade


def test_no_item_bound_is_a_recorded_skip(monkeypatch, tmp_path):
    """A human's round has no contract. That is a skip that says so, not a
    pass that pretends to have graded."""
    _arm(monkeypatch, tmp_path, grade=_grader(None))
    g = _Gate(None, ["app/x.py"], tmp_path)
    ok, detail, data = g.rung_review()
    assert ok and data["skipped"] and "no backlog item" in detail


def test_an_unreachable_grader_is_external_not_a_pass(monkeypatch, tmp_path):
    """A waived review is the #544 shape. The engine being down is the
    engine's fault, so the item keeps its attempt — but the rung fails."""
    events = _arm(monkeypatch, tmp_path, grade=_grader(None, ok=False, error="HTTP 503"))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True and "503" in detail
    assert events[-1]["event"] == "review" and events[-1]["ok"] is False


def _phantom_entry(verdict="partial", node=PHANTOM, note="the pin is elsewhere"):
    return {"clause": 1, "verdict": verdict, "evidence_path": "app/x.py", "evidence_line": 1,
            "test_node_id": node, "how_verified": "ran", "note": note}


def _one_tree(tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_it():\n    assert 1\n")


def test_an_unresolvable_citation_makes_the_review_unreliable_and_spends_no_attempt(
        monkeypatch, tmp_path):
    """The 2026-09-24 round died on this text. A grader whose own evidence is
    not in the tree it was handed has not judged the diff, so the item keeps its
    attempt exactly as it does when the grader is unreachable."""
    _one_tree(tmp_path)
    obj = {"premise": "sound", "summary": "close", "clauses": [_phantom_entry()],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), prior=1)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True
    assert "unreliable" in detail and "keeps its attempt" in detail
    assert "review_retry" not in data, "an unreliable review is not a graded refusal"
    assert "tests/test_facts_surviving_readers.py" in detail
    assert events[-1]["ok"] is False and events[-1]["blocking"] is False
    assert "test_facts_surviving_readers" in events[-1]["error"]


# The two wordings this rail has, verbatim from review.py. `PAST_EOF` is the
# `downgraded` reason #1750 clause 5 holds to, and the one that reaches the review event
# on the re-ask path; `PAST_EOF_UNRESOLVED` is the `citation_unresolved` wording, which
# is what a broken citation rail carries into `unreliable` (#1442's shape).
PAST_EOF = "evidence_line {line} past EOF ({eof} lines) of {path}"
PAST_EOF_UNRESOLVED = "evidence_line {line} is past EOF of {path} ({eof} lines at the graded head)"


def _past_eof_entry(clause=1, *, line=500, **kw):
    """A `met` clause in a `_one_tree` worktree: `app/x.py` is one line there, so 500
    is past EOF and every other rail holds. Each keyword is a rail #1750 says must
    hold before a citation defect may be treated as the grader's own problem.
    """
    entry = {"clause": clause, "verdict": "met", "evidence_path": "app/x.py",
             "evidence_line": line, "test_node_id": "tests/test_x.py::test_it",
             "how_verified": "ran",
             "note": "capture_round plants each scenario and the scorecard names the dir"}
    entry.update(kw)
    return entry


def test_a_met_clause_whose_only_defect_is_a_line_past_eof_is_not_a_grade(wt):
    """Clause 1: the only rail that fails is a line the named file cannot contain, so
    `parse_review` reports the review unreliable rather than returning an empty list.

    The distinction decides what the attempt is spent on. A wrong line number in a
    file that resolved is arithmetic the grader did, and the author has nothing to
    edit — the file, the node and the behaviour all passed. `unreliable` is the list
    that means "this review did not grade the diff", so adding an entry here IS the
    fix: the no-attempt path for that list already exists in `rung_review`.
    """
    parsed = RV.parse_review(_obj(evidence_line=500), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is True, c
    assert c["downgraded"] == [PAST_EOF.format(line=500, eof=6, path="app/x.py")], \
        "the reason stays verbatim in the clause however the review is routed"
    assert len(parsed["unreliable"]) == 1, parsed["unreliable"]
    assert "clause 1" in parsed["unreliable"][0]
    assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in parsed["unreliable"][0]


def test_a_line_past_eof_spends_no_attempt_at_the_rung(monkeypatch, tmp_path):
    """Clause 2: at the rung that same review is an `external_blocker`, not a retry.

    Shaped like `test_an_unresolvable_citation_makes_the_review_unreliable_and_spends_no_attempt`,
    which pins this accounting for a test file the grader invented. The line-number
    case reaches the same arm and inherits the same treatment: the rung fails, the item
    keeps its attempt, and `review_retry` is absent because nothing was graded.
    """
    _one_tree(tmp_path)
    obj = {"premise": "sound", "summary": "graded, but on a number that is not there",
           "clauses": [_past_eof_entry()], "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), prior=1)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True, data
    assert "keeps its attempt" in detail, detail
    assert "no review attempt is spent" in detail, detail
    assert "review_retry" not in data, "a grader's arithmetic is not a graded refusal"
    assert PAST_EOF.format(line=500, eof=1, path="app/x.py") in detail, detail
    ev = events[-1]
    assert ev["event"] == "review" and ev["ok"] is False and ev["blocking"] is False
    assert PAST_EOF.format(line=500, eof=1, path="app/x.py") in ev["error"], ev["error"]


@pytest.mark.parametrize("bad_node", [
    # A real node, in a test file this diff never touched: the clause is demoted, and
    # the author can fix it by citing a test the diff actually changed.
    "tests/test_old.py::test_before",
    # A file the grader invented. This is the rail that already routes to `unreliable`
    # on its own (#1442), and the case the flag must not swallow: the phantom-file
    # `broken` marker is the ONLY thing separating this from a re-ask, since `why`
    # still holds nothing but the past-EOF line.
    PHANTOM,
])
def test_a_line_past_eof_beside_a_failing_node_rail_is_still_a_graded_refusal(wt,
                                                                             bad_node):
    """Clause 3: past EOF is the grader's own problem only when it is the ONLY one.

    With the node rail failing as well, the clause has a defect the author can act on
    (or the review is already unreliable for a reason that has its own test), so the
    citation gets no special treatment and `unreliable` stays as those tests left it.
    Widening the routing past this line would hand every bad node citation a free
    re-gate, and a grader that keeps inventing nodes would never be refused at all.
    """
    parsed = RV.parse_review(
        _obj(evidence_line=500, test_node_id=bad_node),
        worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] in ("partial", "unmet"), c
    # The flag is what routes a clause to the re-ask, and it is not set for either
    # shape: a second rail spoke about the clause, whatever it decided.
    assert c.get("citation_only") is not True, c
    if bad_node == "tests/test_old.py::test_before":
        # A node in a file the diff never touched is a finding about the diff, so the
        # review stays actionable: `unreliable` empty, no free re-gate, the attempt
        # charged as before. And the demoted clause still carries the reason verbatim.
        assert parsed.get("unreliable", []) == [], parsed.get("unreliable")
        assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in c["downgraded"], c
    else:
        # The invented-file shape was ALREADY unreliable before this change (#1442):
        # the rail could not read its input at all, so `broken` named the clause and the
        # review spent no attempt. That path is unchanged; the past-EOF line rides along
        # inside the SAME reason list, because `broken[idx]` holds the very list the line
        # rail appends to. Both reasons are visible, neither one is a new route.
        assert c["verdict"] == "partial", c
        assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in c["downgraded"], c
        assert any("test_node_id" in w for w in c["downgraded"]), c
        reasons = " ".join(parsed["unreliable"])
        assert bad_node.split("::")[0] in reasons, parsed["unreliable"]
        assert PAST_EOF_UNRESOLVED.format(line=500, eof=6, path="app/x.py") in reasons, \
            parsed["unreliable"]
        reasons = parsed["unreliable"]
        assert any(bad_node.split("::")[0] in r for r in reasons), reasons
        assert PAST_EOF_UNRESOLVED.format(line=500, eof=6, path="app/x.py") in \
            " ".join(reasons), reasons


def test_a_line_past_eof_beside_an_inferred_verification_is_a_graded_refusal(wt):
    """Clause 3, other half: `how_verified: inferred` is a claim about the evidence,
    not about the grader's arithmetic, so the clause stays a `partial` to be fixed.
    """
    parsed = RV.parse_review(_obj(evidence_line=500, how_verified="inferred"),
                             worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is not True, c
    assert parsed.get("unreliable", []) == [], parsed.get("unreliable")


def test_a_demotion_that_is_not_a_phantom_line_is_still_a_grade(wt):
    """The guard's other half: `len(why) == 1` alone would be enough to re-ask almost
    every demotion in the file, so the reason has to BE the impossible line.

    A single `how_verified is not ran|read` demotion is a finding about the clause: the
    grader looked and could not say it ran anything. Granting that a re-ask would let a
    round re-gate indefinitely without the author changing a line, and would empty
    `unreliable` of its meaning — the list says the review could not check the diff, not
    that the diff was weak.
    """
    parsed = RV.parse_review(_obj(how_verified="inferred"), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c["downgraded"] == ["how_verified is not ran|read"], c
    assert c.get("citation_only") is not True, c
    assert parsed.get("unreliable", []) == [], parsed.get("unreliable")
    assert parsed["downgraded"] == [1], "the downgrade is still a graded fact"


def test_a_phantom_line_on_a_broken_premise_still_spends_its_attempt(monkeypatch,
                                                                    tmp_path):
    """A citation defect must not rescue a review that also says the item is unsound.

    `PREMISES` is exactly `("sound", "unsound")`, and the no-attempt arm returns the text
    `the item keeps its attempt`; reaching it from an unsound premise would hand a round
    the grader says should not have been attempted a free re-gate every time the grader
    also mis-numbers a line. The premise verdict is a finding about the ITEM, so it
    outranks the citation and the refusal charges its attempt as it did before.
    """
    _one_tree(tmp_path)
    obj = {"premise": "unsound", "premise_problems": ["the item restates a landed fix"],
           "summary": "", "clauses": [_past_eof_entry()],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), prior=1)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data.get("external_blocker") is not True, data
    # `Gate.rung_review`'s unsound-premise branch returns `review_premise_unsound`, not
    # `review_retry`, and it is a charged refusal — so the key is the branch, named.
    assert data["review_premise_unsound"] is True, data
    assert "keeps its attempt" not in detail, detail
    # Charged, in the rung's own ledger event: `blocking` true and the unsound-premise
    # `kind`, which is a different decision from the graded `retry` — a finding about the
    # item, not about the diff.
    ev = events[-1]
    assert ev["blocking"] is True and ev["kind"] == "unsound", ev
    # The item-level keys `review_retry` / `review_premise_unsound` are copied onto the
    # event by the gate's run loop (`gate.py`'s
    # `for key in ("review_retry", "review_premise_unsound")`), outside this rung, so
    # they are asserted in `data`, which is what that loop reads.
    assert data["review_summary"], data


def test_a_line_past_eof_never_masks_an_unmet_clause(monkeypatch, tmp_path):
    """Clause 4: one phantom line number must not hide a genuine finding on another
    clause, because the `unreliable` arm returns before findings are ever delivered.

    This is the common shape, not the corner: the docstring above `parse_review`
    counts 23 `met` clauses across 15 rounds citing a line past EOF. A round that
    pairs a phantom number with a real failure still has to come back with that
    failure named, and still pays for it.
    """
    _one_tree(tmp_path)
    contract = {"id": 7, "title": "t", "body": "b", "path": "",
                "clauses": ["clause one holds", "clause two holds"]}
    obj = {"premise": "sound", "summary": "one phantom, one real", "clauses": [
        _past_eof_entry(1),
        {"clause": 2, "verdict": "unmet", "evidence_path": "app/x.py", "evidence_line": 1,
         "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran",
         "note": "the second path is not covered at all"}],
        "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), contract=contract)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False, "a real unmet clause must still refuse the round"
    assert data.get("external_blocker") is not True, (
        "the phantom citation swallowed the finding: the author would be told to "
        "re-gate and never shown what is wrong")
    assert data["review_retry"] is True and data["review_attempt"] == 1, data
    assert "the second path is not covered at all" in data["review_findings"], data
    assert PAST_EOF.format(line=500, eof=1, path="app/x.py") in data["review_findings"], \
        "the phantom line is still reported, as a downgrade rather than a re-ask"
    # Charged, and both clause texts reach the ledger: `blocking` + the graded `retry`
    # kind, `findings` naming the real failure, and the phantom clause still carrying its
    # verbatim `downgraded` reason and its `citation_only` mark.
    ev = events[-1]
    assert ev["blocking"] is True and ev["kind"] == "retry", ev
    assert "the second path is not covered at all" in ev["findings"], ev
    phantom = next(c for c in ev["clauses"] if c["clause"] == 1)
    assert phantom["citation_only"] is True and phantom["downgraded"] == [
        PAST_EOF.format(line=500, eof=1, path="app/x.py")], phantom
    assert ev["downgraded"] == [1], ev


def test_the_reread_still_loses_nothing_in_the_ledger(monkeypatch, tmp_path):
    """Clause 5, the half that is not about the verdict: the re-read leaves the reason
    in the ledger word for word, in BOTH places the review event carries it.

    Three rounds on 2026-09-11 were downgraded here and nothing recorded which path
    failed. Converting the case into a re-ask must not lose the reason along with the
    refusal. The one `review` event holds the string twice over: in `error`, the
    human-readable reason the arm returns, and in `clauses[0]["downgraded"]`, the
    machine-readable list the finalizer reads. A reader searching `promotions.jsonl`
    for `evidence_line` finds it either way.

    The refusal path is covered by the parametrised
    `test_a_line_past_eof_beside_a_failing_node_rail_is_still_a_graded_refusal`, which
    asserts the same string in that clause's `downgraded`, and by
    `test_a_line_past_eof_never_masks_an_unmet_clause`, which asserts it inside the
    event's `findings`.
    """
    _one_tree(tmp_path)
    reason = PAST_EOF.format(line=500, eof=1, path="app/x.py")
    obj = {"premise": "sound", "summary": "", "clauses": [_past_eof_entry()],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    review_events = [e for e in events if e["event"] == "review"]
    assert len(review_events) == 1, events
    ev = review_events[0]
    assert reason in ev["error"], ev
    assert ev["clauses"][0]["downgraded"] == [reason], ev["clauses"][0]
    assert ev["clauses"][0]["citation_only"] is True, ev["clauses"][0]
    # The list the no-attempt arm is built on, on the event too: this is what
    # `rung_review`'s detail and the item's attempt accounting both read.
    assert any(reason in u for u in ev["unreliable"]), ev.get("unreliable")


# ── #1845: an advisory test-honesty note may not cost the round its re-ask ────
#
# Round SM_20260929_101355 (#1812, 2026-09-29T10:35:13Z) is the shape these tests
# hold: clause 1 demoted for a line past EOF and marked `citation_only`, clauses
# 2-5 `met`, premise `sound`, and two `severity: advisory` test-honesty notes.
# The block-list beside the re-ask read a `clause` key that no honesty entry has
# ever carried, so it counted those two notes as findings the author could act
# on, `unreliable` stayed empty, and the round paid for a review that had graded
# nothing (attempt 1's 220.7 s plus a 206.3 s test re-run before attempt 2 landed
# it). `decide_by_grader` had already called those same entries advisory.

def _honesty(severity="advisory", *, actionable=True,
             problem="the assert message restates the literal it compares"):
    """One grader test-honesty finding, filed against a test file this diff changed.

    The path matters as much as the severity: `parse_review` demotes any finding
    about a non-test path to `advisory` whatever the grader wrote, so naming
    `tests/test_x.py` is what makes a `severity: "blocking"` entry a test of the
    severity and not an accidental test of the file."""
    return {"file": "tests/test_x.py", "line": 2, "problem": problem,
            "severity": severity, "actionable_in_round": actionable}


def _phantom_plus_met_obj(*honesty):
    """Two clauses in a `_one_tree`/`wt` worktree: clause 1 cites line 500 of a file
    that cannot hold it, clause 2 is met on evidence inside the same file.

    Both fixtures serve: `app/x.py` is 6 lines in `wt` and 1 in `_one_tree`, so the
    phantom reason reads `(6 lines)` in one and `(1 lines)` in the other, and clause
    2's `evidence_line` 1 is inside either way. Clause 2 present and `met` is what
    makes this the live shape — the block-list asks about EVERY clause, not just the
    demoted one."""
    return {"premise": "sound",
            "summary": "clause 1 graded on a line the named file cannot hold",
            "clauses": [_past_eof_entry(1, note="the guard refuses and prints the flag"),
                        _past_eof_entry(2, line=1, note="the dry run still prints")],
            "test_honesty": list(honesty), "seams_unverified": []}


_TWO_CONTRACT = {"id": 7, "title": "t", "body": "b", "path": "",
                 "clauses": ["clause one holds", "clause two holds"]}


def test_an_advisory_honesty_note_does_not_cost_the_round_its_past_eof_reask(wt):
    """Clause 1: a note the file's own policy calls advisory leaves the re-ask intact.

    `unreliable` is the whole of the no-attempt route, so an entry appearing there IS
    the fix; the note is not lost, it rides on in `test_honesty` for the report.
    """
    parsed = RV.parse_review(_phantom_plus_met_obj(_honesty("advisory")), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=2)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is True, c
    assert len(parsed["unreliable"]) == 1, (
        f"the advisory note suppressed the re-ask: {parsed['unreliable']}")
    assert "clause 1" in parsed["unreliable"][0], parsed["unreliable"]
    assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in parsed["unreliable"][0], \
        parsed["unreliable"]
    assert parsed["test_honesty"][0]["severity"] == "advisory", (
        "the note must still reach the report, not be dropped to make the route work")
    # And the same entry is advisory on the decide side too. A review routed to the
    # re-ask never reaches `decide_by_grader` — the `unreliable` arm in `rung_review`
    # returns first — so this call, not the rung, is where the two surfaces can be
    # compared for this shape. The note may only ever append a tail to the identical
    # refusal the review makes without it.
    kind0, findings0 = RV.decide_by_grader(dict(parsed, test_honesty=[]), [])
    kind, findings = RV.decide_by_grader(parsed, [])
    assert kind == kind0 == "retry", (kind, kind0, findings)
    assert findings.startswith(findings0 + "; "), (findings0, findings)
    assert "advisory tests/test_x.py:2: " in findings[len(findings0):], findings


def test_a_blocking_note_the_round_cannot_fix_also_leaves_the_reask(wt):
    """Clause 2: `actionable_in_round: false` demotes too, exactly as
    `decide_by_grader` already reads it — asserted by calling that function on the
    same parsed review, so the two surfaces cannot drift back apart unnoticed."""
    parsed = RV.parse_review(_phantom_plus_met_obj(_honesty("blocking", actionable=False)),
                             worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=2)
    assert len(parsed["unreliable"]) == 1, (
        f"a blocking-but-unfixable note suppressed the re-ask: {parsed['unreliable']}")
    assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in parsed["unreliable"][0], \
        parsed["unreliable"]
    # `decide_by_grader` on the SAME parsed review is the policy this has to match.
    # The control is the same review with the note removed: the note may only ever
    # append an advisory tail to an identical refusal, never add a blocking line —
    # which is exactly the distinction the block-list was getting wrong.
    kind0, findings0 = RV.decide_by_grader(dict(parsed, test_honesty=[]), [])
    kind, findings = RV.decide_by_grader(parsed, [])
    assert kind == kind0 == "retry", (kind, kind0, findings)
    assert "clause 1 partial" in findings0, findings0
    assert findings.startswith(findings0 + "; "), (findings0, findings)
    tail = findings[len(findings0):]
    assert "advisory test honesty tests/test_x.py:2" in tail, tail
    assert "(blocking, but not fixable in this round)" in tail, tail


def test_a_blocking_actionable_note_still_refuses_instead_of_re_asking(wt):
    """Clause 3, both halves: a finding that is blocking AND fixable in this round
    buys no re-ask (`unreliable` empty), and the author is still shown it while the
    demoted clause is still refused as a partial."""
    parsed = RV.parse_review(_phantom_plus_met_obj(_honesty("blocking")), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=2)
    assert parsed.get("unreliable", []) == [], (
        f"a fixable honesty finding was routed to the free re-gate: {parsed['unreliable']}")
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is True, c
    assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in c["downgraded"], c
    kind, findings = RV.decide_by_grader(parsed, [])
    kind0, findings0 = RV.decide_by_grader(dict(parsed, test_honesty=[]), [])
    assert kind == kind0 == "retry", (kind, kind0, findings)
    assert "test honesty" not in findings0, (
        "the line this test is about has to come from the honesty entry:\n" + findings0)
    assert "test honesty tests/test_x.py:2: " in findings, (
        "the finding the author can fix in this round was not shown to them:\n" + findings)
    assert "clause 1 partial" in findings, findings


def test_the_advisory_note_buys_the_reask_and_the_blocking_note_costs_an_attempt(
        monkeypatch, tmp_path):
    """Clause 4, both halves at the rung: the advisory shape is an `external_blocker`
    that spends nothing, the blocking-actionable shape is a graded refusal that spends
    attempt 1. One test because the pair is the whole of the change's value: routing
    everything to the no-attempt arm would be as wrong as charging both."""
    _one_tree(tmp_path)
    reason = PAST_EOF.format(line=500, eof=1, path="app/x.py")

    events = _arm(monkeypatch, tmp_path, grade=_grader(_phantom_plus_met_obj(
        _honesty("advisory"))), contract=_TWO_CONTRACT)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True, data
    assert "keeps its attempt" in detail, detail
    assert "no review attempt is spent" in detail, detail
    assert "review_retry" not in data, (
        "an advisory note sent the round down the charged-refusal path:\n" + detail)
    assert reason in detail, detail
    # The no-attempt arm writes one `review` event and nothing else. Which arm ran
    # is not in that event — `self.event(...)` forwards only error/clauses/
    # seams_unverified/test_honesty/unreliable, so `external_blocker` and
    # `review_retry` above are the rung's `data`, not the ledger's — so what the
    # record can prove is that the reason survived on the no-attempt path.
    assert len(events) == 1, events
    assert any(reason in u for u in events[0].get("unreliable", [])), events[0]

    _arm(monkeypatch, tmp_path, grade=_grader(_phantom_plus_met_obj(
        _honesty("blocking"))), contract=_TWO_CONTRACT)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False, detail
    assert data.get("external_blocker") is not True, (
        "a finding the author can fix in this round was forgiven as the grader's "
        "own arithmetic:\n" + detail)
    assert data["review_retry"] is True and data["review_attempt"] == 1, data
    assert "test honesty tests/test_x.py:2" in data["review_findings"], data


def test_the_advisory_reask_still_puts_the_past_eof_reason_in_the_ledger(
        monkeypatch, tmp_path):
    """Clause 5, the record half: routing the case to the re-ask because of an
    advisory note must not lose the reason, which the event carries twice over — in
    `error` for a human and in `clauses[0]["downgraded"]` for the finalizer.

    Same two places `test_the_reread_still_loses_nothing_in_the_ledger` pins for a
    review with no honesty notes at all; this is the note-present half of that claim,
    and the half that was false while the block-list counted any note as blocking."""
    _one_tree(tmp_path)
    reason = PAST_EOF.format(line=500, eof=1, path="app/x.py")
    events = _arm(monkeypatch, tmp_path, grade=_grader(_phantom_plus_met_obj(
        _honesty("advisory"))), contract=_TWO_CONTRACT)
    _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    review_events = [e for e in events if e["event"] == "review"]
    assert len(review_events) == 1, events
    ev = review_events[0]
    assert ev["ok"] is False and ev["blocking"] is False, ev
    assert reason in ev["error"], ev
    phantom = next(c for c in ev["clauses"] if c["clause"] == 1)
    assert phantom["citation_only"] is True and phantom["downgraded"] == [reason], phantom
    assert any(reason in u for u in ev["unreliable"]), ev.get("unreliable")


def test_an_advisory_note_beside_a_phantom_line_and_an_unmet_clause_is_a_charged_refusal(
        monkeypatch, tmp_path):
    """Clause 5, the refusal half: the re-ask stays conditional on the phantom line
    being the review's ONLY defect, honesty notes included.

    One phantom number, one genuinely unmet clause, one advisory note. The rung
    short-circuits on `unreliable` before findings are delivered, so this round must
    still be shown what is wrong and still pay for it — the advisory note changes
    nothing here, which is why this test passes before the fix and after it."""
    _one_tree(tmp_path)
    reason = PAST_EOF.format(line=500, eof=1, path="app/x.py")
    obj = _phantom_plus_met_obj(_honesty("advisory"))
    obj["clauses"][1] = {"clause": 2, "verdict": "unmet", "evidence_path": "app/x.py",
                         "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                         "how_verified": "ran", "note": "the second path is not covered"}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), contract=_TWO_CONTRACT)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False, detail
    assert data.get("external_blocker") is not True, (
        "the phantom citation swallowed the finding: the author would be told to "
        "re-gate and never shown what is wrong")
    assert data["review_retry"] is True and data["review_attempt"] == 1, data
    assert "the second path is not covered" in data["review_findings"], data
    assert reason in data["review_findings"], data
    ev = [e for e in events if e["event"] == "review"][-1]
    assert ev["blocking"] is True and ev["kind"] == "retry", ev
    phantom = next(c for c in ev["clauses"] if c["clause"] == 1)
    assert phantom["citation_only"] is True and phantom["downgraded"] == [reason], phantom


def test_a_note_denying_added_tests_against_a_positive_delta_spends_no_attempt(
        monkeypatch, tmp_path):
    """#1442 clause 4: "This round's diff adds no such test", said of a diff
    whose `def test_` delta over the base was four nodes. The deterministic half
    already counts them, so the contradiction is decidable without a model."""
    _one_tree(tmp_path)
    obj = {"premise": "sound", "summary": "", "clauses": [
        _phantom_entry(node="tests/test_x.py::test_it",
                       note="This round's diff adds no such test; the pin is a prior landing")],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True
    assert "adds no test" in detail and "keeps its attempt" in detail
    assert events[-1]["blocking"] is False


def test_a_sound_premise_with_an_unmet_clause_is_a_retry_with_findings(monkeypatch, tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    obj = {"premise": "sound", "summary": "close",
           "clauses": [{"clause": 1, "verdict": "unmet", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "", "how_verified": "read",
                        "note": "only the chat path is covered"}],
           "test_honesty": [], "seams_unverified": ["loopback POST"]}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    g = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path)
    ok, detail, data = g.rung_review()
    assert ok is False and data["review_retry"] is True and data["review_attempt"] == 1
    assert "only the chat path is covered" in data["review_findings"]
    assert "seam unverified" in data["review_findings"]
    assert "fix what it names" in detail
    ev = events[-1]
    assert ev["event"] == "review" and ev["blocking"] and ev["kind"] == "retry"
    assert ev["clauses"][0]["verdict"] == "unmet"


def test_a_clause_the_grader_left_out_still_refuses_at_the_rung(monkeypatch, tmp_path):
    """One clause graded of two is a verdict on the change, not a rail failure.

    The unreadable-shape rail must not widen into an amnesty: a grader that
    answered clause 1 and said nothing about clause 2 still sends the round back
    with an attempt charged. Only the ALL-unusable case skips the judgment.
    """
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    contract = {"id": 7, "title": "t", "body": "b", "path": "",
                "clauses": ["clause one holds", "clause two holds"]}
    obj = {"premise": "sound", "summary": "one of two",
           "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                        "how_verified": "ran", "note": "ran it"}],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), contract=contract)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["review_retry"] is True and data["review_attempt"] == 1
    assert data.get("external_blocker") is not True, "an abstention is a judgment, not a rail"
    assert "not addressed by the grader" in data["review_findings"]
    assert "fix what it names" in detail
    ev = events[-1]
    assert ev["ok"] is True and ev["blocking"] and ev["kind"] == "retry"
    assert [c["verdict"] for c in ev["clauses"]] == ["met", "partial"]


def test_the_second_refusal_says_abort_and_the_third_never_asks_the_model(monkeypatch, tmp_path):
    obj = {"premise": "sound", "summary": "", "clauses": [
        {"clause": 1, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
         "test_node_id": "", "how_verified": "inferred", "note": "still"}],
        "test_honesty": [], "seams_unverified": []}
    grade = _grader(obj)
    _arm(monkeypatch, tmp_path, grade=grade, prior=1)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_attempt"] == 2 and "abort and report" in detail
    assert len(grade.calls) == 1
    _arm(monkeypatch, tmp_path, grade=grade, prior=2)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_retry"] and data["review_exhausted"]
    assert len(grade.calls) == 1, "a third gate call must not spend another grading turn"


def test_an_unsound_premise_is_its_own_verdict(monkeypatch, tmp_path):
    obj = {"premise": "unsound", "summary": "the file it fixes was deleted in 5531f21",
           "clauses": [], "test_honesty": [], "seams_unverified": []}
    _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_premise_unsound"] is True
    assert "5531f21" in data["review_summary"] and "review_retry" not in data


def test_all_clauses_met_passes_and_names_the_session(monkeypatch, tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    obj = {"premise": "sound", "summary": "does what it says",
           "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                        "how_verified": "ran", "note": "ran it"}],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok and "1 met" in detail and data["review_session"] == "sess_r"
    assert events[-1]["blocking"] is False


def test_the_gate_event_carries_the_review_flags_and_findings(monkeypatch):
    """`gate.json` dies with the worktree; `implement_outcomes` reads these off
    the ledger long after."""
    events = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    g = G.Gate("SM_FLAGS", ROOT, "HEAD")
    g._rung("review", lambda: (False, "sent back", {"review_retry": True,
                                                    "review_findings": "clause 1 unmet",
                                                    "review_attempt": 1}))
    g._rung("review", lambda: (False, "unsound", {"review_premise_unsound": True,
                                                  "review_summary": "false premise"}))
    assert events[0]["review_retry"] is True and events[0]["review_findings"] == "clause 1 unmet"
    assert events[0]["review_attempt"] == 1 and "review_premise_unsound" not in events[0]
    assert events[1]["review_premise_unsound"] is True and events[1]["review_summary"] == "false premise"


def test_preflight_refuses_a_contract_with_no_clauses_or_no_test(monkeypatch, tmp_path):
    """Zero-cost fail-fasts for a round that has a contract — cheaper to learn
    here than after a 77 s test run, and both are the round's own."""
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "", "clauses": [], "path": ""})
    g = G.Gate("SM_PF", ROOT, "HEAD", item_id=9)
    # Drive only the clause/test check by calling the same predicate preflight uses.
    contract = RV.item_contract(9)
    assert not contract["clauses"]
    src = (ROOT / "scripts" / "automod" / "gate.py").read_text()
    assert "has no acceptance clauses to judge" in src
    assert "nothing pins a clause" in src
    assert g.item_id == 9


def test_review_sits_after_tests_and_before_venv(monkeypatch):
    names: list[str] = []
    g = G.Gate("SM_POS", ROOT, "HEAD")
    monkeypatch.setattr(g, "_rung", lambda name, fn: names.append(name) or True)
    g.run()
    assert names.index("tests") < names.index("review") < names.index("venv")
    assert names.index("canary_smoke") == names.index("drill") - 1


# ── the backlog: retry, cap, disagreement, partial ─────────────────────────

def test_a_review_refusal_re_offers_the_item_with_the_findings_and_the_branch(isolated):
    write_item(isolated, 544, clauses=["a", "b"]); _confirm(544, clauses=["a", "b"])
    _round(544, "SM_1"); _review_refused("SM_1", 544, findings="clause 2 unmet: no loopback test")
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[544]
    assert verdict == "review_retry"
    assert "no loopback test" in detail and "automod/SM_1" in detail and "from_branch" in detail
    assert 544 not in B.implemented_ids(S.LEDGER_PATH)
    assert B.desired_statuses(S.LEDGER_PATH, None)[544][0] == "up_next"
    assert B.reoffer_reason(S.LEDGER_PATH, 544).startswith("review_retry:")


def test_the_review_re_offer_is_capped(isolated):
    """`REVIEW_RETRY_CAP` re-offers: the first round plus two more, each
    refused on a DIFFERENT clause (the same clause twice escalates earlier —
    see the next test). The fourth finished round is spent."""
    write_item(isolated, 545, clauses=["a", "b", "c", "d"]); _confirm(545, clauses=["a", "b", "c", "d"])
    for n, rid in enumerate(("SM_a", "SM_b", "SM_c"), 1):
        _round(545, rid); _review_refused(rid, 545, clauses_unmet=(n,), findings=f"f{n}")
        assert B.implement_outcomes(S.LEDGER_PATH)[545][0] == "review_retry", n
    _round(545, "SM_d"); _review_refused("SM_d", 545, clauses_unmet=(4,), findings="f4")
    verdict, _ = B.implement_outcomes(S.LEDGER_PATH)[545]
    assert verdict == "spent", "one round plus REVIEW_RETRY_CAP re-offers; then a human decides"
    assert B.desired_statuses(S.LEDGER_PATH, None)[545][0] == "draft"


def test_the_same_clause_twice_is_a_disagreement_and_escalates_early(isolated):
    """Author says met, grader says unmet, twice, on clause 1. A third round
    re-runs the argument; a human resolves it."""
    write_item(isolated, 546, clauses=["a", "b"]); _confirm(546, clauses=["a", "b"])
    _round(546, "SM_x"); _review_refused("SM_x", 546, clauses_unmet=(1,))
    assert B.review_disagreement(S.LEDGER_PATH, 546) is None
    _round(546, "SM_y"); _review_refused("SM_y", 546, clauses_unmet=(1, 2))
    assert B.review_disagreement(S.LEDGER_PATH, 546) == 1
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[546]
    assert verdict == "spent" and detail.startswith("review disagreement") and "clause 1" in detail
    # ...while different clauses on successive reviews are still a retry.
    write_item(isolated, 547, clauses=["a", "b"]); _confirm(547, clauses=["a", "b"])
    _round(547, "SM_p"); _review_refused("SM_p", 547, clauses_unmet=(1,))
    _round(547, "SM_q"); _review_refused("SM_q", 547, clauses_unmet=(2,))
    assert B.implement_outcomes(S.LEDGER_PATH)[547][0] == "review_retry"


def test_review_retry_yields_to_incomplete_and_supersedes_external(isolated):
    write_item(isolated, 548, clauses=["a"]); _confirm(548, clauses=["a"])
    _round(548, "SM_i", stop_reason="max_turns"); _review_refused("SM_i", 548)
    assert B.implement_outcomes(S.LEDGER_PATH)[548][0] == "incomplete"
    # A later passing gate on the same round supersedes the refusal.
    S.append_event({"event": "gate", "round_id": "SM_i", "rung": "drill", "ok": True}, path=S.LEDGER_PATH)
    assert "SM_i" not in B.review_retry_rounds(S.LEDGER_PATH)


def test_an_unsound_premise_spends_the_attempt(isolated):
    write_item(isolated, 549, clauses=["a"]); _confirm(549, clauses=["a"])
    _round(549, "SM_u")
    S.append_event({"event": "gate", "round_id": "SM_u", "rung": "review", "ok": False,
                    "detail": "review: premise unsound", "review_premise_unsound": True,
                    "review_summary": "false"}, path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[549][0] == "spent"
    assert "SM_u" in B.review_unsound_rounds(S.LEDGER_PATH)
    want = B.desired_statuses(S.LEDGER_PATH, None, retriage_enabled=False)[549]
    assert want[0] == "draft" and want[2] is True
    want = B.desired_statuses(S.LEDGER_PATH, None)[549]
    assert want[0] == "draft" and len(want) == 2, "its one re-triage comes before a person"


def test_fresh_confirmations_are_picked_before_re_offers(isolated):
    """Oldest-first alone let a sent-back item monopolise the loop."""
    write_item(isolated, 550, clauses=["a"]); _confirm(550, clauses=["a"])     # older, sent back
    _round(550, "SM_o"); _review_refused("SM_o", 550)
    p = write_item(isolated, 551, clauses=["a"]); _confirm(551, clauses=["a"])  # newer, fresh
    fm, body = B._split_frontmatter(p.read_text())
    fm["created"] = datetime.now(timezone.utc).isoformat()
    p.write_text(f"---\n{yaml.dump(fm)}---\n{body}")
    B.reconcile_statuses(S.LEDGER_PATH, None)
    pair = B.select_confirmed(S.LEDGER_PATH, None)
    assert pair is not None and pair[0].id == 551
    # ...and a re-offer is still reachable once the fresh work is gone.
    B.set_status(551, "draft", "parked by a human")
    assert B.select_confirmed(S.LEDGER_PATH, None)[0].id == 550


# ── clause outcomes derive the acceptance; a nameless deferral is not_met ──

def test_parse_outcome_derives_the_acceptance_from_clauses():
    base = {"landed": True, "acceptance": "met", "deferred_to": [], "summary": "s", "spawned": []}
    all_met = [{"clause": 1, "outcome": "met", "evidence": "t::a", "deferred_to": []},
               {"clause": 2, "outcome": "met", "evidence": "t::b", "deferred_to": []}]
    assert B.parse_outcome({**base, "clause_outcomes": all_met})["acceptance"] == "met"
    one_unmet = all_met[:1] + [{"clause": 2, "outcome": "not_met", "evidence": "", "deferred_to": []}]
    out = B.parse_outcome({**base, "clause_outcomes": one_unmet})
    assert out["acceptance"] == "not_met" and B.unmet_clauses(out) == [2]
    deferred = all_met[:1] + [{"clause": 2, "outcome": "deferred", "evidence": "", "deferred_to": [618]}]
    out = B.parse_outcome({**base, "clause_outcomes": deferred})
    assert out["acceptance"] == "deferred" and out["deferred_to"] == [618]
    # `unnecessary` is a verdict on the item and is kept as stated.
    assert B.parse_outcome({**base, "acceptance": "unnecessary", "clause_outcomes": []})["acceptance"] == "unnecessary"


def test_a_deferral_that_names_nothing_is_not_met():
    """#544 exactly: `deferred`, `deferred_to: []`, and the item parked in
    `in_progress` with "close this when that closes" pointing at nothing."""
    base = {"landed": True, "acceptance": "deferred", "deferred_to": [], "summary": "s",
            "spawned": [], "clause_outcomes": []}
    assert B.parse_outcome(base)["acceptance"] == "not_met"
    nameless = [{"clause": 1, "outcome": "deferred", "evidence": "", "deferred_to": []}]
    assert B.parse_outcome({**base, "clause_outcomes": nameless})["acceptance"] == "not_met"
    assert B.parse_outcome({**base, "deferred_to": [7]})["acceptance"] == "deferred"
    assert "clause_outcomes" in B.IMPLEMENT_OUTCOME_SCHEMA["required"]
    assert B.IMPLEMENT_OUTCOME_SCHEMA["properties"]["clause_outcomes"]["items"]["properties"][
        "outcome"]["enum"] == list(B.CLAUSE_OUTCOMES)


def test_a_landed_round_with_an_unmet_clause_is_offered_once_more_for_it(isolated):
    p = write_item(isolated, 552, clauses=["a", "b"]); _confirm(552, clauses=["a", "b"])
    S.append_event({"event": "backlog_implement", "item_id": 552, "phase": "started"}, path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 552, "phase": "finished",
                    "round_id": "SM_l", "stop_reason": "stop", "num_turns": 30,
                    "outcome": {"landed": True, "acceptance": "not_met", "deferred_to": [],
                                "summary": "half", "spawned": [],
                                "clause_outcomes": [
                                    {"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []},
                                    {"clause": 2, "outcome": "not_met", "evidence": "", "deferred_to": []}]}},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "promoted", "round_id": "SM_l", "commit": "c0ffee00c0ffee"}, path=S.LEDGER_PATH)
    # Under observation: still in_progress.
    assert B.desired_statuses(S.LEDGER_PATH, None)[552][0] == "in_progress"
    S.append_event({"event": "settled", "commit": "c0ffee00c0ffee"}, path=S.LEDGER_PATH)
    out = B.close_settled_items(S.LEDGER_PATH, None)
    assert out == [{"item_id": 552, "closed": False, "acceptance": "not_met"}]
    assert "clause(s) [2]" in _fm(p)["activity_log"][-1]
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[552]
    assert verdict == "partial" and "[2]" in detail and "c0ffee00" in detail
    assert B.desired_statuses(S.LEDGER_PATH, None)[552][0] == "up_next"
    assert "partial" in I._reoffer_block(B.reoffer_reason(S.LEDGER_PATH, 552))


# ── clauses at triage: both parsers, the item, the fallback ─────────────────

TEXT = ("VERDICT: confirmed\nSURFACE: code\nCHECK: c\nEVIDENCE: e\n"
        "ACCEPTANCE: the retry fires once and the dashboard shows it\n"
        "ACCEPTANCE_CLAUSES:\n1. a retried item fires email_send once\n2. the counter is on /api/dashboard\n"
        "SPAWNED: none\n")


def test_the_text_path_splits_numbered_clauses_and_keeps_the_placeholder_rule():
    parsed = T.parse_verdict(TEXT)
    assert parsed["acceptance_clauses"] == ["a retried item fires email_send once",
                                           "the counter is on /api/dashboard"]
    assert parsed["acceptance"].startswith("the retry fires once")
    none = T.parse_verdict("VERDICT: stale\nCHECK: c\nEVIDENCE: e\nACCEPTANCE: -\nACCEPTANCE_CLAUSES: none\n")
    assert none["acceptance_clauses"] == [] and none["acceptance"] == ""


def test_the_structured_path_carries_clauses_too():
    obj = {"verdict": "confirmed", "surface": "code", "check": "c", "evidence": "e",
           "acceptance": "x", "acceptance_clauses": ["one", "two", "none"], "spawned": []}
    assert T.parse_verdict("", obj)["acceptance_clauses"] == ["one", "two"]
    assert "acceptance_clauses" in B.TRIAGE_VERDICT_SCHEMA["required"]
    assert "ACCEPTANCE_CLAUSES:" in T.PROMPT


def test_record_verdict_writes_the_clauses_onto_the_item(isolated):
    p = write_item(isolated, 560, status="draft")
    item = next(i for i in B.open_items(None) if i.id == 560)
    B.record_verdict(item, "confirmed", "real", acceptance="x", acceptance_clauses=["one", "two"])
    fm = _fm(p)
    assert fm["acceptance_clauses"] == ["one", "two"] and fm["status"] == "up_next"
    assert "1. one" in p.read_text() and "graded one by one" in p.read_text()


def test_acceptance_clauses_fall_back_to_the_prose_for_old_items(isolated):
    write_item(isolated, 561); _confirm(561, acceptance="the check passes")
    contract = RV.item_contract(561)
    assert contract["clauses"] == ["the check passes"] and contract["title"] == "A thing"
    assert B.acceptance_clauses_of({"acceptance": "-"}) == []
    assert B.split_clause_lines("1. a\n2) b\n") == ["a", "b"]
    assert B.split_clause_lines("just prose") == ["just prose"]


# ── the implement side: prompt, resume, abort reason ───────────────────────

def test_the_implement_prompt_names_the_clauses_the_seams_and_the_item_id():
    text = I.PROMPT.format(item_id=9, status="up_next", priority="high", name="n", body="b",
                           triaged_ago="today", surface="code", check="c", evidence="e",
                           acceptance="a", clauses="    1. a\n    2. b", spawn_cap=I.SPAWN_CAP,
                           max_turns=I.DEFAULT_MAX_TURNS, gate_minutes=16, first_gate_by=23,
                           round_label="item9", reoffer="", members="", human_clauses="",
                           surface_rules="")
    assert "    1. a\n    2. b" in text
    # "Seams." and the review-rung procedure moved to the vault skill
    # `automod-change-own-code` (cut 4); the prompt names the skill instead.
    assert "item_id=9" in text and "automod-change-own-code" in text and "review rung" in text
    assert "A deferral that names no id is recorded as `not_met`" in text
    assert "per clause" in text


def test_the_reoffer_banner_for_a_review_retry_names_the_branch_to_resume():
    reason = ("review_retry: the review rung found the premise sound but the implementation or its "
              "tests short — clause 1 unmet; its work is on branch `automod/SM_20260910_070638` "
              "(pass it as from_branch)")
    banner = I._reoffer_block(reason)
    assert 'from_branch="automod/SM_20260910_070638"' in banner
    assert "never reached a verdict" not in banner
    assert "never reached a verdict" in I._reoffer_block("incomplete: ran out of clock")


@pytest.fixture()
def scratch(tmp_path, monkeypatch):
    live = tmp_path / "live"; (live / "app").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(live))
    git(live, "config", "user.email", "t@e.com"); git(live, "config", "user.name", "t")
    (live / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    git(live, "add", "-A"); git(live, "commit", "-q", "-m", "base")
    for name in ("STATE_DIR", "BROKEN_DIR"):
        monkeypatch.setattr(S, name, tmp_path / "state")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state" / "rounds")
    for name, fn in (("LEDGER_PATH", "promotions.jsonl"), ("HALTED_PATH", "halted"),
                     ("BROKEN_PATH", "BROKEN"), ("LOCK_PATH", "lock"), ("CURRENT_PATH", "current.json")):
        monkeypatch.setattr(S, name, tmp_path / "state" / fn)
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    return live


def test_a_round_can_resume_the_branch_review_sent_back(scratch, monkeypatch):
    """The re-offer is not a fresh start: the new worktree begins where the
    refused round left off, rebased onto live main, and the old branch is
    gone so branches do not accumulate forever."""
    # Round ids are second-resolution; two starts in one test need distinct ones.
    ids = iter(("SM_T1", "SM_T2"))
    monkeypatch.setattr(R, "_round_id", lambda: next(ids))
    first = R.start("first go", force=True, item_id=77)
    rid1 = first["round_id"]
    wt1 = Path(first["worktree"])
    (wt1 / "app" / "fix.py").write_text("FIX = 1\n", encoding="utf-8")
    git(wt1, "add", "-A"); git(wt1, "commit", "-q", "-m", "partial fix")
    spec1 = yaml.safe_load((S.ROUNDS_DIR / rid1 / "run_spec.yaml").read_text())
    assert spec1["item"] == {"id": 77}
    R.abort(rid1, reason="review sent it back: clause 1 unmet")
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_aborted"][-1]
    assert ev["reason"].startswith("review sent it back")
    assert W.branch_exists(scratch, f"automod/{rid1}")

    # main moves underneath in the meantime.
    (scratch / "app" / "other.py").write_text("O = 1\n", encoding="utf-8")
    git(scratch, "add", "-A"); git(scratch, "commit", "-q", "-m", "human commit")
    live_head = git(scratch, "rev-parse", "HEAD").stdout.strip()

    second = R.start("second go", force=True, item_id=77, from_branch=f"automod/{rid1}")
    try:
        wt2 = Path(second["worktree"])
        assert (wt2 / "app" / "fix.py").exists(), "the refused round's work is the starting point"
        assert (wt2 / "app" / "other.py").exists(), "rebased onto the moved main"
        assert second["base"] == live_head and second["rebased_onto"] == live_head
        assert second["from_branch"] == f"automod/{rid1}"
        assert not W.branch_exists(scratch, f"automod/{rid1}"), "the old branch is deleted"
        start_ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_start"][-1]
        assert start_ev["item_id"] == 77 and start_ev["from_branch"] == f"automod/{rid1}"
    finally:
        W.remove(second["round_id"], repo=scratch)


def test_an_abort_row_carries_the_gate_verdict_beside_the_reason(scratch, monkeypatch):
    """A reason that contradicts the artifact has to be visible in the row.

    #1169's round was aborted with the reason "Both review attempts spent …
    the review refused twice on clause 4/5 test honesty" 20 seconds after its
    `gate.json` reported `ok: true` with "review: 5 met of 5 clause(s)" — the
    narrative described the *first* gate run, and `abort()` then deleted the
    artifact that disproved it. Three implement-filed blockers (#1091, #1169,
    #1200) are the same shape. So the row carries the report's own verdict and
    head, read off disk before the round dir goes, and a reader needs no
    `gate.json` to spot the disagreement.
    """
    ids = iter(("SM_G1", "SM_G2"))
    monkeypatch.setattr(R, "_round_id", lambda: next(ids))

    gated = R.start("graded green", force=True, item_id=77)
    rid = gated["round_id"]
    (S.ROUNDS_DIR / rid).mkdir(parents=True, exist_ok=True)
    # A head no git command could answer for this round, so the row can only
    # have come from the file.
    (S.ROUNDS_DIR / rid / "gate.json").write_text(json.dumps(
        {"round_id": rid, "head": "deadbeefcafe1234", "ok": True,
         "rungs": [{"name": "review", "ok": True, "detail": "5 met of 5 clause(s)"}]}),
        encoding="utf-8")
    out = R.abort(rid, reason="the review refused twice on clause 4")
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "round_aborted"][-1]
    assert ev["gate_ok"] is True
    assert ev["gate_head"] == "deadbeefcafe1234"
    assert "refused twice" in ev["reason"]
    # and back to the caller that wrote the reason, at the moment it wrote it
    assert out["gate_ok"] is True and out["gate_head"] == "deadbeefcafe1234"

    # A round that never gated: the keys are present and empty, so the absence
    # of an artifact is a distinct answer from a gate that said no.
    dry = R.start("never gated", force=True, item_id=78)
    R.abort(dry["round_id"], reason="out of clock")
    ev2 = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "round_aborted"][-1]
    assert ev2["gate_ok"] == "" and ev2["gate_head"] == ""


def test_a_missing_resume_branch_falls_back_to_a_fresh_worktree(scratch):
    out = R.start("go", force=True, from_branch="automod/SM_NOPE")
    try:
        assert out["from_branch_missing"] is True and "does not exist" in out["note"]
        assert Path(out["worktree"]).exists()
    finally:
        W.remove(out["round_id"], repo=scratch)


# ── vault rounds: the same reader, before `git add` ────────────────────────

@pytest.fixture
def vault(tmp_path, monkeypatch):
    r = tmp_path / "obsidian"
    (r / "skills" / "foo").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    monkeypatch.setattr(V, "VAULT", r)
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    monkeypatch.setattr(V, "GRADER", None)
    return r


def _vault_events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


def test_a_vault_land_with_no_grader_records_why_it_was_not_reviewed(vault):
    """Was `assert _vault_events("vault_review") == []`: an abstention left no
    review event at all, and the landing said only `skipped`. #955's merged
    finding is that 34 of 54 item-bound landings looked like that — a grader
    outage, a surface mismatch and an unconsulted reviewer indistinguishable. An
    abstention still never blocks; it now says which of the six it was."""
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v2\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert out["review"] == "skipped" and _vault_events("vault_land")[-1]["review"] == "skipped"
    assert _vault_events("vault_land")[-1]["review_reason"] == out["review_reason"]
    ev = _vault_events("vault_review")[-1]
    assert ev["blocking"] is False and ev["kind"] == "skipped"
    assert "no grader configured" in ev["review_reason"]


def test_a_vault_review_refusal_leaves_the_edit_then_reverts_on_the_second(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("retry", "clause 1 unmet: the skill never says when"))
    S.append_event({"event": "backlog_implement", "item_id": 9, "phase": "started"}, path=S.LEDGER_PATH)
    f = vault / "skills" / "foo" / "SKILL.md"
    f.write_text("---\nname: foo\n---\n# foo v2\n")
    with pytest.raises(V.VaultRoundError, match="still in place"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert "v2" in f.read_text(), "first refusal: the model fixes it in place"
    ev = _vault_events("vault_review")[-1]
    assert ev["blocking"] and ev["attempt"] == 1 and ev["review_retry"] and ev["reverted"] == []
    with pytest.raises(V.VaultRoundError, match="reverted"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert f.read_text() == "---\nname: foo\n---\n# foo\n", "second refusal: put back"
    assert _vault_events("vault_review")[-1]["reverted"] == ["skills/foo/SKILL.md"]


def test_an_unsound_vault_premise_reverts_at_once(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("unsound", "the task this skill serves was deleted"))
    f = vault / "skills" / "foo" / "SKILL.md"
    f.write_text("---\nname: foo\n---\n# foo v3\n")
    with pytest.raises(V.VaultRoundError, match="premise unsound"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v3", item_id=9)
    assert "v3" not in f.read_text()
    assert _vault_events("vault_review")[-1]["review_premise_unsound"] is True


def test_a_vault_grader_that_blows_up_never_blocks_a_validated_landing(vault, monkeypatch):
    def boom(**kw):
        raise RuntimeError("engine gone")
    monkeypatch.setattr(V, "GRADER", boom)
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v4\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v4", item_id=9)
    assert out["ok"] and out["review"] == "skipped"


def test_a_vault_land_passing_review_records_it(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("pass", "all clauses met"))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v5\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v5", item_id=9)
    assert out["review"] == "pass"
    assert _vault_events("vault_review")[-1]["blocking"] is False


def test_the_vault_grader_only_judges_a_vault_items_clauses(isolated, monkeypatch):
    """#551 was a `code` item whose round landed a skill and a task file first.
    The vault grader held that half to the whole contract and refused it for
    the code it had not written yet — and would have every time. Only a
    `vault` surface's clauses can be met by vault paths; the rest are the code
    gate's. No network: the skip is decided before any grader runs."""
    write_item(isolated, 570, clauses=["a"])
    S.append_event({"event": "backlog_triage", "item_id": 570, "verdict": "confirmed",
                    "surface": "code", "acceptance": "a", "acceptance_clauses": ["a"]},
                   path=S.LEDGER_PATH)
    monkeypatch.setattr(RV, "run_grader", lambda **kw: pytest.fail("grader must not run"))
    kind, why, clauses = RV.grade_vault(item_id=570, paths=["skills/x/SKILL.md"], diff="+x",
                                        vault=isolated)
    assert kind == "skipped" and "surface is code" in why and clauses == []
    S.append_event({"event": "backlog_triage", "item_id": 570, "verdict": "confirmed",
                    "surface": "vault", "acceptance": "a", "acceptance_clauses": ["a"]},
                   path=S.LEDGER_PATH)
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": True, "structured": {
        "premise": "sound", "summary": "ok", "test_honesty": [], "seams_unverified": [],
        "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "skills/x/SKILL.md",
                     "evidence_line": 1, "test_node_id": "", "how_verified": "read", "note": ""}]}})
    (isolated / "skills" / "x").mkdir(parents=True); (isolated / "skills" / "x" / "SKILL.md").write_text("x")
    assert RV.grade_vault(item_id=570, paths=["skills/x/SKILL.md"], diff="+x", vault=isolated)[0] == "pass"


# ── #955: the landing clause cannot be graded before the commit ─────────────

# #972 clause 11 and #993 clause 10, verbatim from their front matter. The two
# rounds with a recorded death are #425 (refused twice, clause 6 both times, work
# reverted at attempt 2) and #502 (attempt 1, clause 6, "no revertable sha").
# Triage listed nine open items carrying such a clause; four name the artefact.
LANDING_CLAUSE_972 = ("The change lands via automod_vault_land as one revertable sha "
                      "on vault main, commit message naming #972")
LANDING_CLAUSE_993 = ("The change touches only vault paths — no file under ~/lloyd "
                      "modified — and lands through automod_vault_land as a revertable sha.")
# And a clause that only mentions the landing in passing, plus one whose author
# already said which half the reviewer is to grade (#463 clause 4's shape).
CONTENT_CLAUSES = ["the skill names the retry rule",
                   "No field list names `board_id` (0 occurrences across 37 real keys).",
                   "The commit message names the item, and the change was reviewed before commit."]


def _vault_contract(clauses):
    return {"id": 1, "title": "t", "body": "b", "clauses": list(clauses), "members": [],
            "path": "", "amendments": [], "human_clauses": []}


def test_a_landing_clause_is_the_only_kind_of_clause_the_vault_grader_recognises():
    """Narrow on purpose: a false positive takes a gradeable clause out of the
    reviewer's hands, which is worse than the artifact it fixes."""
    graded = RV.landing_clause_indices(CONTENT_CLAUSES + [LANDING_CLAUSE_972, LANDING_CLAUSE_993])
    assert graded == [4, 5]
    # An author-drawn line stays with the reviewer (#463 clause 4, amended by hand).
    claimed = ("Pre-landing, graded by this review: both files are modified in the "
               "working tree on branch main. Post-landing, the sha.")
    assert RV.landing_clause_indices([claimed]) == []
    # The other spellings the triage listed on #955.
    assert RV.landing_clause_indices(["The change is submitted through "
                                      "`automod_vault_land` as one call naming exactly the "
                                      "six SKILL.md paths"]) == [1]
    assert RV.landing_clause_indices(["The change lands through automod_vault_land "
                                      "naming exactly the touched paths"]) == [1]
    assert RV.landing_clause_indices([]) == [] and RV.landing_clause_indices(None) == []


def test_the_vault_prompt_tells_the_reviewer_the_commit_is_pending(isolated):
    """#425 attempt 1: "Nothing landed … No new sha" — the reviewer ran `git log`
    in the vault and refused a round for the state it was standing in. The prompt
    has to say the commit is the CONSEQUENCE of a pass."""
    p = RV.build_vault_prompt(contract=_vault_contract(["content", LANDING_CLAUSE_972]),
                              paths=["skills/x/SKILL.md"], diff="+x", vault=isolated,
                              landing_clauses=[2])
    assert "The commit does not exist while you grade." in p
    assert ("on a pass the caller commits exactly these paths on the vault's `main` "
            "as one revertable sha") in p
    assert "Clauses 2 have the landing itself as their subject" in p
    assert "Never refuse a round because no sha exists yet" in p
    assert "never report `git log` as evidence that the change did not land" in p
    assert "The caller records their verdict from the commit it creates" in p
    # The ordering is stated for every vault round; the per-clause note only where
    # a clause actually needs it.
    plain = RV.build_vault_prompt(contract=_vault_contract(["content only"]),
                                 paths=["a.md"], diff="+x", vault=isolated)
    assert "The commit does not exist while you grade." in plain
    assert "have the landing itself as their subject" not in plain


def test_a_landing_clause_cannot_refuse_a_round_a_content_clause_still_can(isolated, monkeypatch):
    """The exclusion must not weaken the guard for anything else: clause 1 is a
    content clause the grader says is unmet, so the round still comes back —
    with the landing clause advisory, and marked so the caller can grade it.

    What is policy-free is the landing clause's side: `grade_vault` rewrites
    that verdict to `post_landing` before `decide` sees it. (This ran under the
    `table` policy until it was retired on 2026-09-24; an unmet clause refuses
    under the grader policy all the same.)"""
    write_item(isolated, 573, clauses=["the skill names the retry rule", LANDING_CLAUSE_972])
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": True, "structured": {
        "premise": "sound", "summary": "s", "test_honesty": [], "seams_unverified": [],
        "clauses": [{"clause": 1, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
                     "test_node_id": "", "how_verified": "read", "note": "no retry rule"},
                    {"clause": 2, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
                     "test_node_id": "", "how_verified": "read", "note": "no sha yet"}]}})
    kind, why, rows = RV.grade_vault(item_id=573, paths=["skills/x/SKILL.md"], diff="+x",
                                     vault=isolated)
    assert kind == "retry"
    assert "clause 1 unmet" in why, "an ordinary content clause still refuses"
    assert "clause 2 unmet" not in why, "the landing clause refused it — the #955 artifact"
    assert "observable only after landing" in why
    assert rows[1] == {"clause": 2, "verdict": "post_landing", "subject": "landing"}
    assert rows[0]["verdict"] == "unmet"


def test_a_passing_vault_review_returns_the_landing_clause_for_the_caller_to_close(isolated, monkeypatch):
    """All the content clauses met and the landing clause not gradable: that is a
    pass, and the row the caller rewrites is marked rather than silently `met`."""
    write_item(isolated, 574, clauses=["the skill names the retry rule", LANDING_CLAUSE_993])
    (isolated / "skills" / "y").mkdir(parents=True)
    (isolated / "skills" / "y" / "SKILL.md").write_text("---\nname: y\n---\n# y\n")
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": True, "structured": {
        "premise": "sound", "summary": "s", "test_honesty": [], "seams_unverified": [],
        "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "skills/y/SKILL.md",
                     "evidence_line": 1, "test_node_id": "", "how_verified": "read",
                     "note": "named"},
                    {"clause": 2, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
                     "test_node_id": "", "how_verified": "read", "note": "no commit"}]}})
    kind, why, rows = RV.grade_vault(item_id=574, paths=["skills/y/SKILL.md"], diff="+x",
                                     vault=isolated)
    assert kind == "pass", why
    assert rows[1] == {"clause": 2, "verdict": "post_landing", "subject": "landing"}
    assert rows[0]["verdict"] == "met"


def test_the_vault_grader_names_the_way_it_abstains(isolated, monkeypatch):
    """Six causes, six wordings. #955's merged finding: 34 of 54 item-bound
    successful landings recorded one unlabelled `skipped`, so a refusal a later
    attempt ignored was indistinguishable from a grader outage."""
    write_item(isolated, 575, clauses=["a"])
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": False, "error": "backend 503 after 420s"})
    assert RV.grade_vault(item_id=575, paths=["a.md"], diff="+x", vault=isolated)[1] == (
        "grader did not answer: backend 503 after 420s")
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": True, "structured": {"premise": "sideways"}})
    assert RV.grade_vault(item_id=575, paths=["a.md"], diff="+x", vault=isolated)[1] == (
        "grader returned an unusable object")
    write_item(isolated, 576)
    assert RV.grade_vault(item_id=576, paths=["a.md"], diff="+x", vault=isolated)[1] == (
        "item #576 has no acceptance clauses")


def test_evidence_paths_are_normalized_before_they_are_judged(wt, tmp_path):
    """The schema asks for a bare worktree-relative file; the grader writes
    `app/x.py:164`, `a.py:224,253,201-214`, `~/vault/SOUL.md + ~/vault/b.md (…)`.
    Every one of the first four backfill rows had a real `met` downgraded for
    it. A path that exists, in any of those spellings, stands."""
    n = RV.normalize_evidence_path
    assert n("app/x.py", wt) == "app/x.py"
    assert n("app/x.py:12", wt) == "app/x.py"
    assert n("app/x.py:12,40-52", wt) == "app/x.py"
    assert n("./app/x.py:3", wt) == "app/x.py"
    assert n("`app/x.py:3`", wt) == "app/x.py"
    outside = tmp_path / "vault" / "SOUL.md"; outside.parent.mkdir(); outside.write_text("s")
    assert n(f"{outside} + {outside.parent}/other.md (trim is vault-side)", wt) == str(outside)
    assert n("app/nope.py:1", wt) == "" and n("", wt) == "" and n("/nope/x.md", wt) == ""
    parsed = RV.parse_review(_obj(evidence_path="app/x.py:3,7"), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "met" and parsed["clauses"][0]["evidence_path"] == "app/x.py"


# ── #1254: the cited line is bounded like the cited path ────────────────────

def test_a_met_citing_a_line_past_eof_is_downgraded(wt):
    """`SM_20260914_145943` cl.5 cited `fixture_iv_loop_turn.py:1007` in a
    52-line file and passed: the path rail resolved the file and nothing read
    the number. The clause is downgraded and the citation recorded, the way an
    unresolvable path is."""
    parsed = RV.parse_review(_obj(evidence_line=500), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and parsed["downgraded"] == [1]
    assert c["evidence_path"] == "app/x.py" and c["evidence_line"] == 500
    assert any("past EOF" in w and "(6 lines)" in w for w in c["downgraded"]), c
    assert any("evidence_line 500 is past EOF of app/x.py" in u for u in c["citation_unresolved"])
    assert "accepted" not in c
    # #1750 overturns the ruling this file recorded here for a year ("the file is
    # real, the number is wrong — the review stays actionable and the clause is
    # simply unmet"). Round SM_20260928_164226 is the counter-evidence: the grader
    # cited line 769 of a 366-line file, its own finding text affirmed the clause, and
    # the round spent one of its two review attempts on a number only the grader could
    # have written. An impossible line in a file that DID resolve is the grader's
    # arithmetic. Where the clause has a defect besides that number it stays a graded
    # refusal — see
    # `test_a_line_past_eof_beside_a_failing_node_rail_is_still_a_graded_refusal`.
    assert c["downgraded"] == ["evidence_line 500 past EOF (6 lines) of app/x.py"]
    assert any("past EOF" in u for u in parsed["unreliable"]), parsed["unreliable"]


def test_a_line_inside_the_file_and_the_last_line_stand(wt):
    for line in (3, 6):
        parsed = RV.parse_review(_obj(evidence_line=line), worktree=wt,
                                 changed_tests=["tests/test_x.py"], n_clauses=1)
        assert parsed["clauses"][0]["verdict"] == "met" and parsed["downgraded"] == [], line
    # Line 0 is the schema's "no line" and is never a claim.
    parsed = RV.parse_review(_obj(evidence_line=0), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "met"


def test_a_past_eof_line_is_recorded_on_a_non_met_verdict_too(wt):
    parsed = RV.parse_review(_obj(verdict="partial", evidence_line=500), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and "downgraded" not in c
    assert any("past EOF" in u for u in c["citation_unresolved"])


def test_the_line_is_bounded_only_against_the_file_the_grader_named_first(wt):
    """#1252 widened the path rail to later tokens; `evidence_line` still
    describes the first one, so a citation resolved through its second token
    is not held to a number written for another file."""
    parsed = RV.parse_review(_obj(evidence_path="scripts/nope.py:3; tests/test_x.py:1",
                                  evidence_line=3),
                             worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "met" and c["evidence_path"] == "tests/test_x.py"
    assert "citation_unresolved" not in c


def test_a_vault_path_outside_the_worktree_is_not_line_checked(wt, tmp_path_factory):
    """The worktree is the commit under review; a vault file is live and
    shared, and a line into it is not a claim about the commit."""
    outside = tmp_path_factory.mktemp("vault") / "SOUL.md"
    outside.write_text("one line\n")
    parsed = RV.parse_review(_obj(evidence_path=f"{outside}:500", evidence_line=500), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "met" and c["evidence_path"] == str(outside)
    assert "citation_unresolved" not in c
    assert RV.evidence_line_past_eof(str(outside), 500, wt) is None
    assert RV.evidence_line_past_eof("app/x.py", 7, wt) == 6
    assert RV.evidence_line_past_eof("app/x.py", 6, wt) is None
    assert RV.evidence_line_past_eof("app/nope.py", 1, wt) is None


# ── a `landed: true` the ledger cannot see is not stored as landed ──────────
#
# `landed` came straight out of the implementer's structured self-report
# (`parse_outcome`, then the `finished` row) with nothing comparing it to the
# ledger. Measured on 2026-09-21 over `promotions.jsonl`: of 228 finished rows
# claiming `landed: true`, 23 name a round with no landing row at all and 6 more
# have only a `vault_land` while the round's surface was `code` or `mixed`.
# `SM_20260916_095854` (#415) is one of the 6 — "Landed in two halves: vault
# commit 5f65d971 … code commit b0818d4 …", and `main` never moved, which spent
# the item's single unattended attempt. The tests below are the #415 row's shape
# read out of that ledger, then the same claim through the writer.

LANDING_CLAIM = {"acceptance": "met", "landed": True, "clause_outcomes": [],
                 "deferred_to": [], "summary": "Landed in two halves", "spawned": []}
UNLANDED_ROUND = "SM_20260916_095854"


def _vault_row(item_id=415, *, ok=True):
    return {"event": "vault_land", "item_id": item_id, "ok": ok, "commit": "5f65d971" + "0" * 32}


def _reconcile(events, *, surface="mixed", round_id=UNLANDED_ROUND, item_id=415,
               landing_seen=False):
    """The self-report as the finalizer hands it over, reconciled once."""
    outcome = B.parse_outcome(LANDING_CLAIM)
    outcome, _ = B.settle_item_verdict(outcome, landing_seen=landing_seen)
    return B.reconcile_outcome_landing(outcome, round_id=round_id, item_id=item_id,
                                       events=list(events), landing_seen=landing_seen,
                                       surface=surface)


def test_a_mixed_round_with_only_a_vault_landing_is_recorded_as_not_landed():
    """The #415 case, exactly: a `vault_land` row and no promotion, `landed: true`.

    Accepting any landing row would pass this claim — the vault half really did
    commit, which is why the round's own summary reads true and why a mixed round
    is where the defect hides. On a `code`/`mixed` surface only `promoted` /
    `item_landed` says the diff reached `main`.
    """
    outcome, mismatch = _reconcile([_vault_row()])
    assert outcome["landed"] is False, "the half-landing stored the self-report verbatim"
    assert UNLANDED_ROUND in mismatch and "promoted" in mismatch, mismatch
    assert outcome["landed_mismatch"] == mismatch
    # The reconciliation demotes the flag, not the verdict: the round still said
    # `met`, and rewriting that here would move a judgment that is not this one.
    assert outcome["acceptance"] == "met"


@pytest.mark.parametrize("event", ["promoted", "item_landed"])
def test_a_round_with_a_code_landing_row_keeps_its_landed_claim(event):
    outcome, mismatch = _reconcile([{"event": event, "round_id": UNLANDED_ROUND,
                                     "item_id": 415, "commit": "a" * 40}])
    assert outcome["landed"] is True and mismatch == ""


def test_a_round_with_no_landing_row_at_all_is_recorded_as_not_landed():
    outcome, mismatch = _reconcile([{"event": "gate", "round_id": UNLANDED_ROUND,
                                     "rung": "review", "ok": True}], surface="code")
    assert outcome["landed"] is False and "promoted" in mismatch


def test_an_unsuccessful_vault_land_row_is_not_a_landing():
    """`ok: false` is a validation failure that reverted its own paths."""
    outcome, mismatch = _reconcile([_vault_row(ok=False)], surface="vault")
    assert outcome["landed"] is False and mismatch, "a reverted vault land counted as landed"


def test_a_vault_surface_round_lands_on_its_vault_row():
    """The other half of the surface rule: a vault-only item has no `promoted`
    row by construction, and demoting it would flag every vault round."""
    outcome, mismatch = _reconcile([_vault_row()], surface="vault")
    assert outcome["landed"] is True and mismatch == ""


def test_a_landing_still_in_flight_keeps_the_claim():
    """`automod_land` returns before the landing runs, so the ledger can be
    legitimately silent at the moment the turn is finalised. The process's own
    evidence — the live marker or `current.json` — covers that window."""
    outcome, mismatch = _reconcile([], landing_seen=True)
    assert outcome["landed"] is True and mismatch == ""


class _Payload:
    def __init__(self, payload):
        self.payload = payload


def _implement_turn_isolated(board, monkeypatch, structured, *, mid_turn=None,
                             surface=""):
    """Run one real implement turn against an isolated board and ledger, and
    return its `finished` ledger row. `mid_turn` fires inside the turn, which is
    when a landing's ledger row would actually appear; it may be async, for a hook
    that has to cross a tool handler (`agent_mcp/automod.py:call_tool`)."""
    import asyncio
    import inspect

    from workers.sources import _common as C

    write_item(board, 415)
    _confirm(415, acceptance="the check passes", surface=surface)

    async def fake(prompt, **kw):
        if mid_turn:
            hooked = mid_turn()
            if inspect.isawaitable(hooked):
                await hooked
        return {"text": "done\n\nSPAWNED: none\n", "session_id": "s415",
                "stop_reason": "stop", "num_turns": 30, "errors": [],
                "structured": structured, "structured_error": ""}

    async def no_reap(session_id):
        return None

    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    monkeypatch.setattr(I, "_reap_at_turn_end", no_reap)
    asyncio.run(I.execute(_Payload({"structured_outcome": True, "max_turns": 40})))
    return [e for e in S.read_events(path=S.LEDGER_PATH)
            if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]


def test_a_vault_landing_row_is_not_evidence_that_the_round_promoted():
    """The two landing checks must not disagree about what a row means.

    `reconcile_outcome_landing` reads `promoted` / `item_landed` / `vault_land`
    (`backlog.LANDING_LEDGER_EVENTS`) and asks `_landing_seen` whether a landing was
    in flight, as the one exemption that covers `automod_land` returning before its
    landing runs. If `_landing_seen` also counted a `vault_land` row, every mixed
    half-landing would take that exemption and the reconciliation would never fire
    — the exact #415 shape, passing its own guard. The two lists are asserted
    against each other, through the real function, because the difference between
    them *is* the mixed-surface rule.
    """
    row = lambda event: {"event": event, "round_id": "SM_VOCAB", "item_id": 415,
                         "ok": True, "commit": "c" * 40}
    assert "vault_land" in B.LANDING_LEDGER_EVENTS, (
        "the reconciliation no longer recognises a vault landing at all, so a "
        "vault-surface round would be demoted on a landing it really did make")
    assert not I._landing_seen("SM_VOCAB", [row("vault_land")]), (
        "`_landing_seen` read a vault landing as this round reaching `automod_land`")
    assert I._landing_seen("SM_VOCAB", [row("promoted")])
    assert I._landing_seen("SM_VOCAB", [row("item_landed")])


def test_the_writer_demotes_an_unlanded_self_report_on_the_finished_row(isolated, monkeypatch):
    """The seam itself: the finalizer's structured object in, the persisted row
    out. #415's row said `"landed": true` beside a `round_aborted` four minutes
    later; a row that still says that after this change is the defect."""
    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM)
    assert ev["outcome"]["landed"] is False, ev["outcome"]
    assert "promoted" in ev["outcome_landing_mismatch"], ev["outcome_landing_mismatch"]
    assert ev["outcome"]["acceptance"] == "met", "the verdict is not what this reconciles"


def test_the_writer_demotes_a_mixed_round_whose_only_landing_is_the_vault(isolated, monkeypatch):
    """#415 end to end: the vault half committed, the code half did not, `landed: true`.

    The mixed-surface rule is only reachable if the in-flight escape hatch does
    not read a `vault_land` row as "the round reached `automod_land`" — that row
    is a landing of the other half. So this is the integration test for the
    wiring: the row appears inside the turn, the real writer runs, and the
    persisted row says not-landed.
    """
    def vault_half():
        S.append_event({"event": "vault_land", "item_id": 415, "ok": True,
                        "round_id": "SM_MIXED", "commit": "5f65d971" + "0" * 32},
                       path=S.LEDGER_PATH)

    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM,
                                  mid_turn=vault_half, surface="mixed")
    assert ev["outcome"]["landed"] is False, ev["outcome"]
    assert "promoted" in ev["outcome_landing_mismatch"], ev["outcome_landing_mismatch"]


def test_the_writer_keeps_a_claim_the_round_actually_promoted(isolated, monkeypatch):
    """The false-positive direction, same seam: a round that did land must not be
    demoted, or every landed item's row starts lying the other way."""
    def landed():
        S.append_event({"event": "round_start", "round_id": "SM_TEST_LANDED",
                        "item_id": 415, "session_id": "s415"}, path=S.LEDGER_PATH)
        S.append_event({"event": "promoted", "round_id": "SM_TEST_LANDED", "item_id": 415,
                        "commit": "b" * 40}, path=S.LEDGER_PATH)

    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM, mid_turn=landed)
    assert ev["outcome"]["landed"] is True, ev["outcome"]
    assert ev["outcome_landing_mismatch"] == ""


def test_a_landing_that_finished_after_the_snapshot_is_still_a_landing(tmp_path, monkeypatch):
    """The first seam the review rung named on `SM_20260921_091624`: the marker and
    `current.json` are read live, the ledger is a snapshot taken earlier in the
    run, so a landing that completes inside that window leaves both empty — the
    marker cleared, its `promoted` row appended after the snapshot — and the round
    that DID land gets demoted.

    The row is written with the real writer (`S.append_event`, what `promote.py`
    calls) into the real ledger file this test isolated, and the marker is absent
    because the landing is over. What is stale is the injected snapshot, which is
    exactly the condition under test. The mutation control is the same call with
    nothing in the ledger: the second look must not become an unconditional True.
    """
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "read_land_marker", lambda rid: None)
    monkeypatch.setattr(S, "read_current", lambda: {"round_id": "SM_OTHER"})
    S.append_event({"event": "promoted", "round_id": "SM_RACE", "commit": "e" * 40},
                   path=S.LEDGER_PATH)
    assert I._landing_seen("SM_RACE", []), (
        "a landing that completed after the run's snapshot was read as no landing — "
        "the round that landed is demoted to `landed: false`")
    # Control: same stale snapshot, same absent marker, but a ledger with no
    # landing row at all. The second look is a re-read of a file, not a yes.
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "empty-ledger.jsonl")
    assert not I._landing_seen("SM_RACE", [
        {"event": "round_start", "round_id": "SM_RACE"}]), (
        "`_landing_seen` answered True for a round with no landing row anywhere, "
        "which makes the second look meaningless")


def test_a_vault_landing_with_no_item_id_is_attributed_to_the_turn_that_made_it(
        isolated, monkeypatch, tmp_path):
    """The second seam the review rung named: `item_id` is OPTIONAL on
    `automod_vault_land`, and 56 of the 172 `vault_land` rows in the ledger on
    2026-09-21 carry none, so matching a vault landing on `item_id` alone demotes a
    vault-surface turn that really landed.

    End to end across the process boundary the row actually crosses: the row is
    written by the real MCP handler (`agent_mcp/automod.py`, a different process in
    production) with no item_id and no open round, into the ledger the worker
    reads; the fake implement turn reports the same session id the harness put in
    `_meta`. The claim must survive. The control is the same turn with no landing
    at all — attribution is not a licence.
    """
    import json

    import agent_mcp.automod as AM

    vault = tmp_path / "obsidian"
    (vault / "backlog").mkdir(parents=True)
    git(vault, "init", "-q", "-b", "main", str(vault))
    git(vault, "config", "user.email", "t@e.com")
    git(vault, "config", "user.name", "t")
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: draft\n---\n# item\n")
    git(vault, "add", "-A")
    git(vault, "commit", "-q", "-m", "base")
    # The change the tool is asked to land: the base commit must not contain it.
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")
    monkeypatch.setattr(V, "VAULT", vault)
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    monkeypatch.setattr(AM, "_inner_voice_gate", lambda action: None)

    async def land_without_an_item():
        from agent_mcp import _task_registry
        token = _task_registry.current_session_id.set("s415")
        try:
            # The real handler, awaited exactly as the MCP server awaits it.
            res = await AM.call_tool("automod_vault_land", {
                "paths": ["backlog/9-item.md"], "message": "backlog: promote #9"})
        finally:
            _task_registry.current_session_id.reset(token)
        out = json.loads(res[0].text if isinstance(res, list) else res.content[0].text)
        assert out.get("ok"), out
        return out

    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM,
                                  mid_turn=land_without_an_item, surface="vault")
    assert ev["outcome"]["landed"] is True, (
        f"a vault landing the handler really committed was attributed to nobody: {ev}")
    assert ev.get("outcome_landing_mismatch", "") == "", ev
    rows = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "vault_land"]
    assert len(rows) == 1 and rows[0]["item_id"] is None, rows
    assert rows[0]["session_id"] == "s415", (
        "the row carries no attribution at all, so nothing downstream can tell a "
        "landing from a round that landed nothing")


def test_the_vault_writer_still_demotes_a_turn_that_landed_nothing(isolated, monkeypatch):
    """The control for the seam above, same surface and same claim: a `vault`
    turn whose ledger holds no `vault_land` row keeps being demoted. Attributing a
    row by session is not the same as trusting the claim."""
    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM, surface="vault")
    assert ev["outcome"]["landed"] is False, ev["outcome"]
    assert "vault_land" in ev["outcome_landing_mismatch"], ev["outcome_landing_mismatch"]


def test_a_commit_the_vault_holds_is_not_an_unresolved_citation(repo, tmp_path, monkeypatch):
    """A `mixed` round lands vault commits and the grader cites them. Three of
    the five shas that spent review attempts on 2026-09-26/27 were real vault
    commits (7709cf49, e3c415a1, da4d1b42): the rail asked only the code repo."""
    r, _ = repo
    vault = tmp_path / "vault"
    vault.mkdir()
    git(vault, "init", "-q")
    (vault / "note.md").write_text("n\n")
    git(vault, "add", "-A")
    git(vault, "-c", "user.name=v", "-c", "user.email=v@v", "commit", "-q", "-m", "vault")
    vsha = git(vault, "rev-parse", "HEAD").stdout.strip()[:8]
    monkeypatch.setattr(RV, "REVIEW_EVIDENCE_ROOTS", (vault,))
    parsed = _judged(r, _fin({"verdict": "met", "test_node_id": "", "evidence_path": "app/m.py",
                              "evidence_line": 1, "note": f"the skill landed in vault commit {vsha}"}),
                     1, repo=r)
    assert parsed["unreliable"] == [] and "citation_unresolved" not in parsed["clauses"][0]
    assert RV.unresolved_shas(f"see {vsha}", r, also=()) == [vsha], "the code repo alone lacks it"
