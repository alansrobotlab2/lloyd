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
# severity decides (landing pace, 2026-09-13)
# ---------------------------------------------------------------------------

def test_an_advisory_finding_never_refuses_however_actionable():
    """The grader era's first day: 60 advisory test-honesty entries against
    one blocking, 43 of the advisory ones "actionable" — a docstring number
    is always fixable — and test honesty led 15 of 21 refusals. #484 met
    every clause four rounds running and was refused on a new nit each time.
    """
    kind, findings = RV.decide_by_grader(
        _parsed(clauses=[_clause(1, "met")],
                test_honesty=[_honesty(actionable=True, severity="advisory",
                                       problem="docstring says 118, the fixture makes 119")]), [])
    assert kind == "pass"
    assert "advisory tests/t.py:3: docstring says 118" in findings, "still in the report"


def test_a_blocking_finding_the_round_cannot_fix_is_demoted_not_dropped():
    kind, findings = RV.decide_by_grader(
        _parsed(test_honesty=[_honesty(actionable=False, severity="blocking",
                                       problem="the node is live_vault")]), [])
    assert kind == "pass"
    assert "the node is live_vault (blocking, but not fixable in this round)" in findings


def test_a_blocking_actionable_finding_refuses():
    kind, findings = RV.decide_by_grader(
        _parsed(test_honesty=[_honesty(actionable=True, severity="blocking",
                                       problem="`x == [] or True`")]), [])
    assert kind == "retry" and findings.startswith("test honesty tests/t.py:3: `x == [] or True`")


def test_a_refused_amendment_refuses_under_the_grader_policy_too():
    """`decide` emitted this finding and `decide_by_grader` dropped it: a
    weakened clause the grader refused could pass on the weakened text."""
    parsed = _parsed(clauses=[_clause(1, "met")], amendments_ok=False,
                     amendments_note="the new clause drops the A/B")
    kind, findings = RV.decide(parsed, [], amendments=[{"clause": 1}], mode="grader")
    assert kind == "retry"
    assert "amendment of clause(s) 1 refused" in findings and "drops the A/B" in findings
    # No amendment shown: the flag is inert, as under the table.
    assert RV.decide(parsed, [], amendments=[], mode="grader")[0] == "pass"


# ---------------------------------------------------------------------------
# seams keep the attempt rule, and repeats advise
# ---------------------------------------------------------------------------

def test_a_testable_seam_refuses_on_the_first_attempt():
    kind, findings = RV.decide(_parsed(seams_unverified=[_seam()]), [], mode="grader",
                               attempt=1, policy="first")
    assert kind == "retry" and findings.startswith("seam unverified: loopback POST")


def test_a_testable_seam_advises_on_the_second_attempt():
    """`seams_block` was dead code under `grader`: a testable seam refused
    every attempt, decisive in 4 of the first 21 refusals."""
    kind, findings = RV.decide(_parsed(seams_unverified=[_seam()]), [], mode="grader",
                               attempt=2, policy="first")
    assert kind == "pass"
    assert "seam unverified (attempt 2, not refusing again): loopback POST" in findings
    # `always` still means always.
    assert RV.decide(_parsed(seams_unverified=[_seam()]), [], mode="grader",
                     attempt=2, policy="always")[0] == "retry"


@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_a_repeated_seam_never_refuses(attempt):
    kind, findings = RV.decide(_parsed(seams_unverified=[_seam(same=True)]), [], mode="grader",
                               attempt=attempt, policy="always")
    assert kind == "pass"
    assert "(repeat, not refusing again)" in findings


# ---------------------------------------------------------------------------
# the equivalence that makes the flip safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attempt", [1, 2])
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
def test_an_object_without_the_fields_decides_the_same_under_both_policies(parsed, pre, attempt):
    """Every calibration case and every backfill row predates the fields.
    Under `grader` they must come out exactly as under `table` — on attempt
    1, where the table blocks a testable seam, and on attempt 2, where it
    advises — so the flip changes nothing about the past, only what the
    grader is asked from now on. It held at attempt 1 only until the seam
    rule honoured `seams_block`.
    """
    import copy
    a = RV.decide(copy.deepcopy(parsed), list(pre), mode="table", attempt=attempt, policy="first")
    b = RV.decide(copy.deepcopy(parsed), list(pre), mode="grader", attempt=attempt, policy="first")
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


def test_the_prior_reviews_block_names_each_reviews_round():
    """The history spans rounds now, so a row that did not say whose review
    it was would read as this round's refusal."""
    block = RV._prior_reviews_block([
        {"ok": True, "round_id": "SM_20260913_173156", "attempt": 2, "head": "a" * 40,
         "blocking": True, "clauses": [{"clause": 4, "verdict": "partial"}], "findings": "c4"},
        {"ok": True, "round_id": "SM_20260913_180142", "attempt": 1, "head": "b" * 40,
         "blocking": False, "clauses": [{"clause": 4, "verdict": "met"}], "findings": ""},
    ])
    assert "- round SM_20260913_173156 attempt 2 on aaaaaaaa: refused; c4=partial" in block
    assert "- round SM_20260913_180142 attempt 1 on bbbbbbbb: passed; c4=met" in block
    assert "Earlier reviews of THIS item" in block


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


# ---------------------------------------------------------------------------
# redecide: the rules measured on recorded grader output
# ---------------------------------------------------------------------------

def _write_ledger(path, events):
    import json
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_redecide_replays_recorded_reviews_under_todays_rules(tmp_path):
    """The acceptance measurement for the landing-pace change: re-decide the
    grader era's refusals on the output the grader actually recorded, with no
    model in the loop."""
    from scripts.automod import review_tools as RT
    ledger = tmp_path / "promotions.jsonl"
    nit = {"file": "tests/t.py", "line": 3, "severity": "advisory", "problem": "docstring number",
           "actionable_in_round": True, "same_as_prior": False}
    _write_ledger(ledger, [
        # too old: outside --since
        {"event": "review", "ok": True, "ts": 100.0, "round_id": "SM_old", "item_id": 1,
         "attempt": 1, "kind": "retry", "premise": "sound", "clauses": [], "test_honesty": [nit]},
        # #484's shape: every clause met, refused on an advisory nit
        {"event": "review", "ok": True, "ts": 1000.0, "round_id": "SM_a", "item_id": 484,
         "attempt": 1, "kind": "retry", "blocking": True, "premise": "sound",
         "clauses": [{"clause": 1, "verdict": "met"}], "test_honesty": [nit],
         "seams_unverified": [], "seams_untestable": []},
        # #860's shape: a suite-level node Python downgraded
        {"event": "review", "ok": True, "ts": 1001.0, "round_id": "SM_b", "item_id": 860,
         "attempt": 1, "kind": "retry", "blocking": True, "premise": "sound",
         "clauses": [{"clause": 8, "verdict": "partial", "test_node_id": "tests/ -k autoresearch",
                      "how_verified": "ran",
                      "downgraded": ["test_node_id not in a test file this diff changed"]}],
         "test_honesty": [], "seams_unverified": [], "seams_untestable": []},
        # an old event whose testable seam still refuses on attempt 1
        {"event": "review", "ok": True, "ts": 1002.0, "round_id": "SM_c", "item_id": 9,
         "attempt": 1, "kind": "retry", "blocking": True, "premise": "sound",
         "clauses": [{"clause": 1, "verdict": "met"}], "test_honesty": [],
         "seams_unverified": ["loopback POST"], "seams_untestable": []},
        # a recorded pass, and an ungraded row that is not a review at all
        {"event": "review", "ok": True, "ts": 1003.0, "round_id": "SM_d", "item_id": 9,
         "attempt": 1, "kind": "pass", "premise": "sound",
         "clauses": [{"clause": 1, "verdict": "met"}], "test_honesty": [], "seams_unverified": []},
        {"event": "review", "ok": False, "ts": 1004.0, "round_id": "SM_e", "error": "grader down"},
    ])
    out = RT.redecide(since_ts=500.0, ledger=ledger, policy="grader", seams_policy="first")
    got = {r["round_id"]: (r["recorded"], r["redecided"]) for r in out["rows"]}
    assert got == {"SM_a": ("retry", "pass"), "SM_b": ("retry", "pass"),
                   "SM_c": ("retry", "retry"), "SM_d": ("pass", "pass")}
    assert (out["refusals"], out["refusals_now_pass"]) == (3, 2)
    assert (out["passes"], out["passes_now_refused"]) == (1, 0)
    assert out["approximated_clauses"] == 1 and "approximated" in out["note"]
    # Without the approximation, the suite-level clause stays what Python recorded.
    strict = RT.redecide(since_ts=500.0, ledger=ledger, policy="grader", seams_policy="first",
                         approximate=False)
    assert {r["round_id"]: r["redecided"] for r in strict["rows"]}["SM_b"] == "retry"


def test_redecide_reads_full_seam_judgments_when_the_event_carries_them():
    from scripts.automod import review_tools as RT
    ev = {"premise": "sound", "clauses": [], "seams_unverified": ["s"], "seams_untestable": [],
          "seams": [{"seam": "s", "testable_before_landing": True, "actionable_in_round": False,
                     "same_as_prior": True}]}
    parsed, _ = RT.parsed_from_event(ev)
    assert parsed["seams_unverified"][0]["same_as_prior"] is True
    old = {k: v for k, v in ev.items() if k != "seams"}
    parsed, _ = RT.parsed_from_event({**old, "seams_untestable": ["s"]})
    assert parsed["seams_unverified"] == [{"seam": "s", "testable_before_landing": False,
                                           "actionable_in_round": True, "same_as_prior": False}]
