"""#588 clause 2: the retired-decision replay's grader and its safety rails.

No engine: the grading rule and the session-id mint are pure, and they are
what the committed count rests on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "decision_replay_588", ROOT / "eval" / "decision_replay_588.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load()
CASES = {c["key"]: c for c in R.load_cases()}


def test_eight_cases_each_with_markers_and_a_source():
    assert len(CASES) == 8
    for case in CASES.values():
        assert case["retirement_markers"], case["key"]
        assert case["source"] and case["retired"] and case["body"].strip()


def test_no_case_body_leaks_its_own_retirement():
    """A naive item must not carry the answer: if the body already names the
    retirement, a `caught` grade measures the prompt, not the triage."""
    for case in CASES.values():
        text = case["title"] + "\n" + case["body"]
        assert R.cites_retirement(case, text) == [], case["key"]


def test_confirmed_without_citation_is_a_re_proposal():
    case = CASES["kg-voice-merge"]
    g = R.grade(case, "confirmed", "VERDICT: confirmed\nmerge the three entities")
    assert g["grade"] == R.RE_PROPOSED


def test_confirmed_that_names_the_retirement_is_caught():
    case = CASES["kg-voice-merge"]
    g = R.grade(case, "confirmed", "VERDICT: confirmed — but commit 06f0e41 guards this")
    assert g["grade"] == R.CAUGHT and g["markers"] == ["06f0e41"]


@pytest.mark.parametrize("verdict", ["stale", "already_done", "unverifiable", "not_code"])
def test_any_other_verdict_is_caught(verdict):
    assert R.grade(CASES["djev-canvas"], verdict, "")["grade"] == R.CAUGHT


def test_no_verdict_is_counted_apart():
    assert R.grade(CASES["djev-canvas"], None, "")["grade"] == R.NO_VERDICT
    s = R.summarize([{"key": "djev-canvas", "rep": 1, "verdict": None},
                     {"key": "djev-canvas", "rep": 2, "verdict": "confirmed", "verdict_text": "x"},
                     {"key": "barge-in", "verdict": "stale", "verdict_text": ""}],
                    list(CASES.values()))
    assert (s["n"], s[R.RE_PROPOSED], s[R.CAUGHT], s[R.NO_VERDICT]) == (3, 1, 1, 1)


def test_every_minted_session_is_sandboxed():
    from agent_mcp._tool_sandbox import is_sandboxed_session
    for key in CASES:
        sid = R.new_session_id(key)
        assert is_sandboxed_session(sid)
        assert R.require_sandboxed(sid) == sid


def test_an_unsandboxed_id_is_refused():
    with pytest.raises(RuntimeError):
        R.require_sandboxed("20260924_120000_autotriage_abcd")


def test_latest_row_per_rep_wins_and_an_override_needs_a_reason(tmp_path):
    rows = [{"key": "barge-in", "rep": 1, "verdict": None},
            {"key": "barge-in", "rep": 1, "verdict": "stale", "verdict_text": ""},
            {"key": "task68-restore", "rep": 1, "verdict": "confirmed",
             "verdict_text": "predates the #1127 guard"}]
    s = R.summarize(rows, list(CASES.values()))
    assert (s["n"], s[R.CAUGHT], s[R.NO_VERDICT]) == (2, 2, 0)
    ov = {"task68-restore#1": {"grade": R.RE_PROPOSED, "reason": "marker matched incidentally"}}
    s = R.summarize(rows, list(CASES.values()), ov)
    row = [r for r in s["rows"] if r["key"] == "task68-restore"][0]
    assert row["grade"] == R.RE_PROPOSED and row["auto"] == R.CAUGHT
    bad = tmp_path / "o.yaml"
    bad.write_text("x#1: {grade: re-proposed}\n")
    with pytest.raises(ValueError):
        R.load_overrides(bad)
