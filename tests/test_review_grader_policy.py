"""Cut 2 of senses-not-supervision: the grader judges its own history.

`review.decide()` had become a policy table with a row per incident — seams
first/always/never, precheck severities, amendment exemptions, attempts keyed
by head then patch-id, two caps. Every row answered one of two questions on
the grader's behalf, from outside the evidence: *is this the same refusal as
last time?* and *can the author fix this inside the round?* The grader holds
the diff, the clauses, its own prior reviews and the tree. It is the one
placed to answer them, and under `automod.review.policy: grader` it does.

The property that makes the flip safe: an object written before the two
fields existed decides IDENTICALLY under both policies.
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
    c = dict(clause=idx, verdict=verdict, evidence_path="app/x.py", evidence_line=1,
             test_node_id="tests/t.py::test_x", how_verified="ran", note="n")
    c.update(kw)
    return c


def _honesty(actionable=True, same=False, **kw):
    h = dict(file="tests/t.py", line=3, severity="blocking", problem="p",
             actionable_in_round=actionable, same_as_prior=same)
    h.update(kw)
    return h


def _seam(actionable=True, same=False, testable=True):
    return dict(seam="loopback POST", testable_before_landing=testable,
                actionable_in_round=actionable, same_as_prior=same)


# ---------------------------------------------------------------------------
# the fields
# ---------------------------------------------------------------------------

def test_the_schema_asks_for_both_judgments_on_every_finding():
    props = RV.REVIEW_SCHEMA["properties"]
    for key in ("test_honesty", "seams_unverified"):
        item = props[key]["items"]
        assert "actionable_in_round" in item["required"], key
        assert "same_as_prior" in item["required"], key


def test_old_objects_default_to_the_old_reading():
    """Absent `actionable_in_round` reads True — every finding blocked — and
    absent `same_as_prior` reads False. So an object from before the fields
    existed is not silently reinterpreted.
    """
    assert RV._judgments({}) == {"actionable_in_round": True, "same_as_prior": False}
    assert RV._judgments({"actionable_in_round": "yes"})["actionable_in_round"] is True
    assert RV._judgments({"actionable_in_round": False})["actionable_in_round"] is False


def test_parse_review_carries_the_fields_through(tmp_path):
    obj = {"premise": "sound", "clauses": [],
           "test_honesty": [{"file": "tests/t.py", "line": 1, "severity": "blocking",
                             "problem": "x", "actionable_in_round": False,
                             "same_as_prior": True}],
           "seams_unverified": [{"seam": "s", "testable_before_landing": True,
                                 "actionable_in_round": False, "same_as_prior": False}]}
    parsed = RV.parse_review(obj, worktree=tmp_path, changed_tests=[], n_clauses=0)
    assert parsed["test_honesty"][0]["actionable_in_round"] is False
    assert parsed["test_honesty"][0]["same_as_prior"] is True
    assert parsed["seams_unverified"][0]["actionable_in_round"] is False


def test_a_bare_string_seam_still_parses_as_actionable(tmp_path):
    obj = {"premise": "sound", "clauses": [], "test_honesty": [],
           "seams_unverified": ["a loopback POST"]}
    parsed = RV.parse_review(obj, worktree=tmp_path, changed_tests=[], n_clauses=0)
    assert parsed["seams_unverified"][0]["actionable_in_round"] is True


# ---------------------------------------------------------------------------
# decide_by_grader
# ---------------------------------------------------------------------------

def test_an_actionable_finding_refuses():
    kind, findings = RV.decide_by_grader(_parsed(test_honesty=[_honesty(actionable=True)]), [])
    assert kind == "retry"
    assert "test honesty" in findings


def test_a_non_actionable_finding_is_advisory():
    """The finding the table refused twice and aborted on: a seam only
    production can cross. The grader says so; the round lands with it in the
    report.
    """
    kind, findings = RV.decide_by_grader(_parsed(seams_unverified=[_seam(actionable=False)]), [])
    assert kind == "pass"
    assert "loopback POST" in findings, "still in the report"


def test_a_repeat_is_marked_in_the_findings():
    kind, findings = RV.decide_by_grader(
        _parsed(test_honesty=[_honesty(actionable=True, same=True)]), [])
    assert kind == "retry"
    assert "[repeat]" in findings


def test_an_unmet_clause_refuses_whatever_the_findings_say():
    kind, _ = RV.decide_by_grader(_parsed(clauses=[_clause(1, "unmet")]), [])
    assert kind == "retry"


def test_post_landing_and_unsatisfiable_keep_their_meanings():
    kind, findings = RV.decide_by_grader(_parsed(clauses=[_clause(1, "post_landing")]), [])
    assert kind == "pass" and "after landing" in findings
    kind, findings = RV.decide_by_grader(_parsed(clauses=[_clause(1, "unsatisfiable")]), [])
    assert kind == "retry" and "automod_amend_clause" in findings


def test_a_precheck_is_a_fact_and_blocks_by_its_severity():
    """The pattern findings are not the grader's judgment to override."""
    kind, _ = RV.decide_by_grader(_parsed(), [
        {"file": "tests/t.py", "line": 1, "problem": "`or True`", "severity": "blocking"}])
    assert kind == "retry"
    kind, _ = RV.decide_by_grader(_parsed(), [
        {"file": "tests/t.py", "line": 1, "problem": "no new test", "severity": "advisory"}])
    assert kind == "pass"


def test_unsound_is_still_the_graders_call_alone():
    kind, _ = RV.decide_by_grader(_parsed(premise="unsound", summary="no"), [])
    assert kind == "unsound"


# ---------------------------------------------------------------------------
# the equivalence that makes the flip safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("parsed,pre", [
    (_parsed(), []),
    (_parsed(clauses=[_clause(1, "met")]), []),
    (_parsed(clauses=[_clause(1, "unmet")]), []),
    (_parsed(clauses=[_clause(1, "partial", downgraded=["x"])]), []),
    (_parsed(test_honesty=[dict(file="tests/t.py", line=1, severity="blocking", problem="p")]), []),
    (_parsed(seams_unverified=[dict(seam="s", testable_before_landing=True)]), []),
    (_parsed(seams_unverified=[dict(seam="s", testable_before_landing=False)]), []),
    (_parsed(), [{"file": "tests/t.py", "line": 1, "problem": "`or True`", "severity": "blocking"}]),
    (_parsed(premise="unsound", summary="no"), []),
])
def test_an_object_without_the_fields_decides_the_same_under_both_policies(parsed, pre):
    """Every calibration case and every backfill row predates the fields.
    Under `grader` they must come out exactly as under `table` on attempt 1
    (where the table blocks a testable seam too) — the flip changes nothing
    about the past, only what the grader is asked from now on.
    """
    import copy
    a = RV.decide(copy.deepcopy(parsed), list(pre), mode="table", attempt=1, policy="first")
    b = RV.decide(copy.deepcopy(parsed), list(pre), mode="grader")
    assert a[0] == b[0], (a, b)


def test_decide_dispatches_on_the_configured_policy(monkeypatch):
    monkeypatch.setattr(RV, "review_policy", lambda: "grader")
    parsed = _parsed(seams_unverified=[_seam(actionable=False)])
    assert RV.decide(parsed, [])[0] == "pass"
    monkeypatch.setattr(RV, "review_policy", lambda: "table")
    assert RV.decide(parsed, [], attempt=1)[0] == "retry"


def test_the_shipped_policy_is_grader():
    """Flipped 2026-09-12 after the calibration suite scored the two policies
    equal on identical grader output. The `table` code stays for a revert."""
    from app.config import CONFIG
    assert ((CONFIG.get("automod") or {}).get("review") or {}).get("policy") == "grader"


# ---------------------------------------------------------------------------
# the grader sees its own history
# ---------------------------------------------------------------------------

def test_the_prior_reviews_block_renders_verdicts_and_findings():
    block = RV._prior_reviews_block([
        {"ok": True, "attempt": 1, "head": "abcdef1234", "blocking": True,
         "clauses": [{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "unmet"}],
         "findings": "clause 2 unmet: no test"},
        {"ok": False, "error": "grader down"},
    ])
    assert "<prior_reviews>" in block
    assert "attempt 1 on abcdef12: refused; c1=met, c2=unmet" in block
    assert "clause 2 unmet: no test" in block
    assert "grader down" not in block, "an ungraded attempt is not a review"


def test_no_prior_reviews_means_no_block():
    assert RV._prior_reviews_block([]) == ""


def test_the_prompt_leads_with_the_prior_reviews():
    text = RV.build_prompt(
        contract={"clauses": ["c1"], "amendments": [], "human_clauses": [], "id": 1,
                  "title": "t", "body": "b"},
        diff="d", diff_truncated=False, changed_tests=[], test_counts={},
        worktree=Path("/wt"), run_tests=Path("/wt/run"),
        prior_reviews=[{"ok": True, "attempt": 1, "head": "abc", "blocking": True,
                        "clauses": [], "findings": "seam unverified: X"}])
    assert text.startswith("<prior_reviews>")
    assert "same_as_prior" in text and "actionable_in_round" in text
