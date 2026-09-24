"""A finished gate says what it is first, and a passed round is not aborted by accident.

Four rounds in four days aborted a change 20–90 seconds after it passed every
rung, each reporting a refusal that is on no ledger row:

  * #1131, 2026-09-15 — "both review attempts spent", on a 5-of-5 pass;
  * #1190, 2026-09-16 — "review refused clause 4", on a first-attempt pass and
    a green rollback drill;
  * #1053, 2026-09-18, twice — "REVIEW refused: fixable problems", a phrase that
    occurs in no tool result of that session, only in the model's own reasoning.

The model had the whole report each time (`transcript_entries` clamps the
stored transcript, never what the model is sent). What it had was `gate.json`
verbatim: `"ok": true` on line five, and several hundred lines later a list of
the reviewer's ADVISORY findings, worded the way a refusal's are.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

from agent_mcp import automod as T
from scripts.automod import round as R
from scripts.automod import state as S, worktree as W


# The report #1053's third round was handed (SM_20260918_185909), with the
# prose shortened. It PASSED — review attempt 2 of 2, four advisory seams, five
# advisory findings — and was aborted 42 seconds later as "refused twice".
HEAD = "9a12e722fe411c3a67576939631aa7bc9f38cf8b"
PATHS = ["agent_mcp/aggregator_auth.py", "agent_mcp/main.py", "tests/test_aggregator_auth.py"]
PASSED = {
    "round_id": "SM_P", "base": "4d4ff1c9f0a5", "head": HEAD, "ok": True,
    "changed_paths": PATHS, "venv": None,
    "rungs": [
        {"name": "preflight", "ok": True, "detail": "3 file(s) in scope", "seconds": 0.17,
         "data": {"buckets": {"allowed": list(PATHS), "protected": [], "denied": [], "unlisted": []},
                  "item_id": 1053, "clauses": 4}},
        {"name": "tests", "ok": True, "detail": "5799 passed, 2 xfailed, 13 skipped", "seconds": 600.0},
        {"name": "review", "ok": True, "seconds": 281.0,
         "detail": "review: 4 met of 4 clause(s); The credential is one raw ASGI middleware",
         "data": {"review_session": "20260918_124122_review_4abe", "review_attempt": 2,
                  "clauses": [{"clause": n, "verdict": "met"} for n in (1, 2, 3, 4)],
                  "post_landing_clauses": [],
                  "advisory_seams": ["MCPPool._open_session — no test opens a persistent session "
                                     "against a guarded server"],
                  "advisory_findings": ["tests/test_aggregator_auth.py:158: assert rec.calls == [] "
                                        "cannot fail: the fixture drives the app with no lifespan",
                                        "tests/test_aggregator_auth.py:313: five caller legs are "
                                        "pinned by a source-text grep; dashboard is still grep-only"],
                  "amendments_ratified": []}},
        {"name": "drill", "ok": True, "detail": "no protected paths touched — drill not required",
         "seconds": 0.0},
    ],
}


def _refused(attempt: int) -> dict:
    rep = copy.deepcopy(PASSED)
    rep["ok"] = False
    nxt = ("fix what it names, commit, and gate again" if attempt < 2
           else "abort and report — the item comes back with these findings and your branch")
    rep["rungs"] = rep["rungs"][:2] + [{
        "name": "review", "ok": False, "seconds": 300.0,
        "detail": f"review sent it back ({attempt}/2; {nxt}): seam unverified: liveness probes",
        "data": {"review_retry": True, "review_findings": "seam unverified: liveness probes",
                 "review_attempt": attempt, "review_session": "s"}}]
    return rep


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    return tmp_path


def _finished(rid: str, report: dict) -> dict:
    d = S.ROUNDS_DIR / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "gate.json").write_text(json.dumps({**report, "round_id": rid}), encoding="utf-8")
    return T._gate_wait(rid, wait_seconds=0)


# ── the report ──────────────────────────────────────────────────────────────

def test_a_pass_says_passed_and_land_before_anything_else():
    out = _finished("SM_P", PASSED)
    assert list(out)[:2] == ["verdict", "next"], list(out)[:4]
    assert out["verdict"].startswith("PASSED") and "4 met of 4" in out["verdict"]
    assert 'automod_land("SM_P")' in out["next"] and "end your turn" in out["next"]
    assert "Nothing in this report is a refusal" in out["next"]
    assert out["ok"] is True and out["gate_finished"] is True, "the report itself is all still there"


def test_a_passes_advisories_are_labelled_as_notes_not_left_in_the_review_rung():
    out = _finished("SM_P", PASSED)
    notes = out["notes_that_did_not_block"]
    assert "did not block" in notes["what"] and "not a reason to edit, re-gate or abort" in notes["what"]
    assert len(notes["seams"]) == 1 and len(notes["findings"]) == 2
    review = next(r for r in out["rungs"] if r["name"] == "review")
    assert "advisory_findings" not in review["data"] and "advisory_seams" not in review["data"]
    assert review["data"]["clauses"], "the grading stays where it was"
    # Order is the point: the notes come after the verdict and before the rungs.
    keys = list(out)
    assert keys.index("notes_that_did_not_block") < keys.index("rungs")
    # And the second copy of `changed_paths` is gone from the top of the report.
    preflight = next(r for r in out["rungs"] if r["name"] == "preflight")
    assert "allowed" not in preflight["data"]["buckets"]
    assert preflight["data"]["buckets"]["protected"] == []


def test_a_pass_with_nothing_to_note_has_no_notes_key():
    rep = copy.deepcopy(PASSED)
    rep["rungs"][2]["data"]["advisory_seams"] = []
    rep["rungs"][2]["data"]["advisory_findings"] = []
    assert "notes_that_did_not_block" not in _finished("SM_Q", rep)


@pytest.mark.parametrize("attempt, verdict, nxt", [
    (1, "REFUSED at review — attempt 1 of 2", "One attempt is left"),
    (2, "REFUSED at review — attempt 2 of 2, the last", "automod_abort"),
])
def test_a_refusal_says_which_attempt_it_was_and_what_that_leaves(attempt, verdict, nxt):
    out = _finished("SM_R", _refused(attempt))
    assert out["verdict"] == verdict and nxt in out["next"]
    assert "notes_that_did_not_block" not in out
    assert "automod_land" not in out["next"]


def test_an_unsound_premise_and_a_grader_that_could_not_run_are_told_apart():
    rep = _refused(1)
    rep["rungs"][-1]["data"] = {"review_premise_unsound": True, "review_summary": "already true"}
    assert "premise unsound" in _finished("SM_U", rep)["verdict"]
    rep["rungs"][-1]["data"] = {"review_session": "s"}
    rep["rungs"][-1]["detail"] = "review could not run: finalizer failed"
    out = _finished("SM_G", rep)
    assert out["verdict"].startswith("NOT GRADED") and "automod_abort" not in out["next"]


def test_an_external_blocker_keeps_its_instruction_and_gets_a_verdict():
    rep = _refused(1)
    rep["rungs"][-1] = {"name": "preflight", "ok": False, "detail": "live tree is dirty",
                        "data": {"external_blocker": True, "retry_after_s": 90,
                                 "external_reason": "uncommitted live edits"}}
    out = _finished("SM_E", rep)
    assert out["verdict"].startswith("NOT JUDGED") and "preflight rung" in out["verdict"]
    assert "Wait 90s" in out["next"] and "Do NOT abort" in out["next"] and out["retry_after_s"] == 90


def test_pre_existing_failures_ride_a_pass_as_a_note_naming_their_owner():
    """A tests rung that passed over failures predating the round says whose
    they are, so the round does not spend its clock fixing the tree."""
    rep = copy.deepcopy(PASSED)
    ids = ["tests/test_uptake.py::test_a", "tests/test_uptake.py::test_b"]
    rep["rungs"][1]["data"] = {"pre_existing_failures": ids, "red_tree_item": 1501}
    out = _finished("SM_P", rep)
    assert out["verdict"].startswith("PASSED")
    pre = out["notes_that_did_not_block"]["pre_existing_failures"]
    assert pre["ids"] == ids
    assert "tracked by item #1501" in pre["note"] and "do not fix them in this round" in pre["note"]
    # Without advisories the note still appears, with its own explanation.
    rep["rungs"][2]["data"]["advisory_seams"] = []
    rep["rungs"][2]["data"]["advisory_findings"] = []
    out = _finished("SM_P2", rep)
    assert out["notes_that_did_not_block"]["pre_existing_failures"]["ids"] == ids
    assert "did not block" in out["notes_that_did_not_block"]["what"]


def test_the_red_tree_items_own_round_is_told_to_fix_them_not_to_leave_them():
    """#1454's round was told "tracked by item #1454 — do not fix them in this
    round" about its own contract (2026-09-24)."""
    rep = copy.deepcopy(PASSED)
    ids = ["tests/test_uptake.py::test_a"]
    rep["rungs"][1]["data"] = {"pre_existing_failures": ids, "red_tree_item": 1454,
                               "red_tree_item_is_own": True}
    note = _finished("SM_O", rep)["notes_that_did_not_block"]["pre_existing_failures"]["note"]
    assert "#1454" in note and "this round's job" in note and "do not fix" not in note


def test_another_rungs_failure_names_the_rung():
    rep = _refused(1)
    rep["rungs"] = rep["rungs"][:1] + [{"name": "static", "ok": False,
                                       "detail": "2 new pyflakes finding(s)", "data": {}}]
    out = _finished("SM_S", rep)
    assert out["verdict"] == "FAILED at static" and "static rung's `detail`" in out["next"]


def test_the_gate_tool_no_longer_says_seven_to_twelve_minutes(monkeypatch, tmp_path):
    for n in range(6):
        t = 1_000_000 + n * 5000
        for rung, secs, detail in (("preflight", 0.2, "ok"), ("tests", 540.0, "5830 passed"),
                                   ("review", 420.0, "review: 4 met of 4 clause(s)")):
            t += secs
            S.append_event({"event": "gate", "round_id": f"SM_{n}", "rung": rung, "ok": True,
                            "seconds": secs, "detail": detail, "ts": t}, path=S.LEDGER_PATH)
    note = T._gate_minutes_note()
    assert "about 16 minutes now" in note and "median of the last 6" in note
    # The measured figure rides the tool's RESULT. Its description carries no
    # number at all: it is part of every turn's cached prompt prefix, and a
    # figure that moved with the ledger would re-prefill every session.
    gate_tool = next(t for t in asyncio_run(T.list_tools()) if t.name == "automod_gate")
    assert "seven to twelve" not in gate_tool.description
    assert not any(ch.isdigit() for ch in gate_tool.description), gate_tool.description


# ── the abort ───────────────────────────────────────────────────────────────

@pytest.fixture()
def passed_round(tmp_path, monkeypatch):
    rid = "SM_P"
    (S.ROUNDS_DIR / rid).mkdir(parents=True)
    (S.ROUNDS_DIR / rid / "gate.json").write_text(json.dumps(PASSED), encoding="utf-8")
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setattr(W, "worktree_path", lambda r: wt)
    state = {"head": HEAD}
    monkeypatch.setattr(W, "head", lambda path: state["head"])
    aborted: list = []
    monkeypatch.setattr(R, "abort", lambda r, reason="": aborted.append((r, reason)) or
                        {"aborted": r, "branch_kept": f"automod/{r}"})
    return rid, state, aborted


def _abort(**arguments) -> dict:
    import asyncio
    res = asyncio.run(T.call_tool("automod_abort", arguments))
    return json.loads(res[0].text if isinstance(res, list) else res.content[0].text)


def test_aborting_a_round_whose_gate_passed_is_refused_with_the_verdict(passed_round, monkeypatch):
    """#1053's own words to the tool, 42 s after the pass: "Both review
    attempts spent (resume gate + this gate), both refusals on accept-side
    seam coverage only"."""
    monkeypatch.setattr(S, "require_enabled", lambda *a, **k: None)
    rid, _state, aborted = passed_round
    out = _abort(round_id=rid, reason="Both review attempts spent (resume gate + this gate)")
    assert aborted == [], "a landable change was discarded"
    assert "PASSED its gate at 9a12e722" in out["error"] and "4 met of 4" in out["error"]
    assert 'automod_land("SM_P")' in out["next"] and "discard_passed_gate=true" in out["next"]


def test_a_round_that_means_it_can_still_abandon_a_passed_change(passed_round, monkeypatch):
    monkeypatch.setattr(S, "require_enabled", lambda *a, **k: None)
    rid, _state, aborted = passed_round
    out = _abort(round_id=rid, reason="measured no gain after the gate", discard_passed_gate=True)
    assert out["aborted"] == rid and aborted == [(rid, "measured no gain after the gate")]


def test_a_commit_made_after_the_gate_is_not_a_passed_round(passed_round, monkeypatch):
    """The pass was of another commit: nothing landable is being discarded."""
    monkeypatch.setattr(S, "require_enabled", lambda *a, **k: None)
    rid, state, aborted = passed_round
    state["head"] = "f" * 40
    assert _abort(round_id=rid, reason="reworked after the gate")["aborted"] == rid
    assert len(aborted) == 1


def test_a_refused_or_ungated_round_aborts_as_before(tmp_path, monkeypatch, passed_round):
    monkeypatch.setattr(S, "require_enabled", lambda *a, **k: None)
    rid, _state, aborted = passed_round
    (S.ROUNDS_DIR / rid / "gate.json").write_text(json.dumps(_refused(2)), encoding="utf-8")
    assert _abort(round_id=rid, reason="refused twice")["aborted"] == rid
    (S.ROUNDS_DIR / rid / "gate.json").unlink()
    assert _abort(round_id=rid, reason="never gated")["aborted"] == rid
    assert len(aborted) == 2


def test_an_abort_under_a_running_landing_is_refused_whatever_it_is_told(passed_round, monkeypatch):
    """`round.abort` removes the worktree; the promoter is using it."""
    monkeypatch.setattr(S, "require_enabled", lambda *a, **k: None)
    rid, _state, aborted = passed_round
    S.write_land_marker(rid, pid=os.getpid(), by="automod_land")
    out = _abort(round_id=rid, reason="changed my mind", discard_passed_gate=True)
    assert aborted == [] and "a landing is running" in out["error"]


def test_the_reaper_and_the_cli_are_not_asked():
    """The guard is the tool's. `round.abort` is what the reaper calls to close
    a round nobody will finish, and it must never be refused."""
    import inspect
    assert "_abort_refusal" not in inspect.getsource(R.abort)
    tool = next(t for t in asyncio_run(T.list_tools()) if t.name == "automod_abort")
    schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema")
    assert schema["properties"]["discard_passed_gate"]["type"] == "boolean"
    assert schema["required"] == ["round_id"]


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)
