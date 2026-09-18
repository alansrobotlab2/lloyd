"""An item verdict is checked against the landing before it closes anything.

`unnecessary` and `rejected` close an item when the turn ends — no landing to
wait for, no re-triage after — and `parse_outcome` keeps them "as stated
whatever the clauses say". `rejected` was recorded twice in its first two days
(2026-09-16 → 09-18) and was wrong both times: #1242 and #1053 both LANDED, and
both were closed "tried it and rejected it on the evidence" minutes before
their promotions. The settle sweep walks open items only, so neither landing
ever reached its item, and had either promotion failed the item would not have
been offered again.

The two incident outcomes below are the ledger's, not paraphrases.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, state as S
from workers.sources import _common as C
from workers.sources import autocode as I


# The finalizer objects as recorded (`backlog_implement` / `finished`, 2026-09-18).
OUTCOME_1053 = {
    "landed": True, "acceptance": "rejected", "deferred_to": [], "summary": "", "spawned": [],
    "clause_outcomes": [
        {"clause": n, "outcome": "met", "evidence": f"tests/test_aggregator_auth.py::t{n}",
         "deferred_to": []} for n in (1, 2, 3, 4)],
}
OUTCOME_1242 = {
    "landed": False, "acceptance": "rejected", "clause_outcomes": [], "deferred_to": [],
    "summary": ("This turn asks for a transcription, but I cannot state `landed: true` on evidence "
                "I do not have: `automod_land` returned immediately with `landing: "
                "SM_20260918_170516, pid 1018929` and its own instruction was to end the turn "
                "without polling, because the landing restarts the backend."),
    "spawned": [],
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    monkeypatch.setattr(S, "read_current", lambda: None)
    return d


def _write_item(d: Path, item_id: int, status="up_next") -> Path:
    fm = {"status": status, "priority": "medium", "board": "lloyd", "tags": ["backlog"],
          "created": (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()}
    path = d / f"{item_id}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n",
                    encoding="utf-8")
    S.append_event({"event": "backlog_triage", "item_id": item_id, "verdict": "confirmed",
                    "check": "grep -n x", "evidence": "still there",
                    "acceptance": "the check no longer reproduces"}, path=S.LEDGER_PATH)
    return path


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text(encoding="utf-8"))[0]


class _Item:
    def __init__(self, payload=None):
        self.payload = payload or {}


# ── the rule ────────────────────────────────────────────────────────────────

def test_1053s_outcome_is_met_not_rejected():
    """`landed: true`, four clauses `met`, an empty summary, `rejected`."""
    out, why = B.settle_item_verdict(B.parse_outcome(OUTCOME_1053), landing_seen=True)
    assert out["acceptance"] == "met" and out["item_verdict_refused"] == "rejected"
    assert "no measurement" in why
    # The outcome's own `landed: true` is enough; the marker need not be seen.
    out, why = B.settle_item_verdict(
        {**B.parse_outcome(OUTCOME_1053), "summary": "measured nothing"}, landing_seen=False)
    assert out["acceptance"] == "met" and "every clause reported met" in why


def test_1242s_outcome_is_unreported_not_rejected():
    """`landed: false` and no clauses — while its own `automod_land` was in
    flight. Nothing to derive from, so the acceptance is unreported and the
    settle sweep asks the review rung instead."""
    out, why = B.settle_item_verdict(B.parse_outcome(OUTCOME_1242), landing_seen=True)
    assert out["acceptance"] == "" and out["item_verdict_refused"] == "rejected"
    assert out["landed"] is True and "`landed: false`" in why


@pytest.mark.parametrize("structured, landing_seen", [
    # Built it, measured it, no gain, landed nothing: Alan's rule, 2026-09-16.
    ({"acceptance": "rejected", "landed": False, "deferred_to": [], "spawned": [],
      "summary": "MRR 0.500 -> 0.497 over 3 runs; within noise, not adopted",
      "clause_outcomes": [{"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []}]}, False),
    # The premise no longer holds; every clause is already true on the live tree.
    ({"acceptance": "unnecessary", "landed": False, "deferred_to": [], "spawned": [],
      "summary": "", "clause_outcomes": []}, False),
    ({"acceptance": "unnecessary", "landed": False, "deferred_to": [], "spawned": [],
      "summary": "already true at HEAD",
      "clause_outcomes": [{"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []}]}, False),
    # Landed the instrument, rejected the idea: the gain clause is not met and
    # the measurement is there. A landing alone does not void a rejection.
    ({"acceptance": "rejected", "landed": True, "deferred_to": [], "spawned": [],
      "summary": "harness landed; ndcg10 0.611 -> 0.609, inside the 0.006 noise floor",
      "clause_outcomes": [{"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []},
                          {"clause": 2, "outcome": "not_met", "evidence": "eval", "deferred_to": []}]},
     True),
])
def test_a_verdict_the_landing_does_not_contradict_still_stands(structured, landing_seen):
    parsed = B.parse_outcome(structured)
    out, why = B.settle_item_verdict(parsed, landing_seen=landing_seen)
    assert why == "" and out == parsed


def test_a_rejection_with_no_measurement_is_not_a_rejection():
    parsed = B.parse_outcome({"acceptance": "rejected", "landed": False, "deferred_to": [],
                              "spawned": [], "summary": "  ",
                              "clause_outcomes": [{"clause": 1, "outcome": "not_met",
                                                   "evidence": "t", "deferred_to": []}]})
    out, why = B.settle_item_verdict(parsed, landing_seen=False)
    assert out["acceptance"] == "not_met" and "no measurement" in why


def test_an_ordinary_outcome_passes_through_untouched():
    parsed = B.parse_outcome({"acceptance": "met", "landed": True, "deferred_to": [], "spawned": [],
                              "summary": "landed", "clause_outcomes": []})
    assert B.settle_item_verdict(parsed, landing_seen=True) == (parsed, "")
    assert B.settle_item_verdict(None, landing_seen=True) == (None, "")


# ── through `execute`: the item is not closed under a landing ────────────────

def _run_turn(monkeypatch, tmp_path, item_id, structured, *, rid="SM_T", land=True):
    from scripts.automod import round as R, worktree as W
    import app.sessions_io as sio
    wt = tmp_path / "wt"
    wt.mkdir(exist_ok=True)
    monkeypatch.setattr(W, "worktree_path", lambda r: wt)
    monkeypatch.setattr(R, "abort", lambda r, reason="": {"aborted": r})
    monkeypatch.setattr(sio, "active_sessions_snapshot", lambda: [])
    monkeypatch.setattr(C, "source_inner_voice", lambda source, default=True: False)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))

    async def turn(prompt, **kw):
        S.append_event({"event": "round_start", "round_id": rid, "item_id": item_id,
                        "session_id": "s-turn"}, path=S.LEDGER_PATH)
        if land:
            # What `automod_land` leaves behind before it returns.
            import os
            S.write_land_marker(rid, pid=os.getpid(), by="automod_land")
        return {"text": "landed\n\nSPAWNED: none\n", "session_id": "s-turn", "stop_reason": "stop",
                "num_turns": 40, "errors": [], "structured": structured, "structured_error": ""}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    asyncio.run(I.execute(_Item({"structured_outcome": True})))
    return [e for e in S.read_events(path=S.LEDGER_PATH)
            if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]


@pytest.mark.parametrize("structured, derived", [(OUTCOME_1053, "met"), (OUTCOME_1242, "")])
def test_a_turn_that_called_automod_land_cannot_close_its_item_as_rejected(
        isolated, monkeypatch, tmp_path, structured, derived):
    p = _write_item(isolated, 1053)
    fin = _run_turn(monkeypatch, tmp_path, 1053, structured)
    fm = _fm(p)
    assert fm["status"] != "done" and B.REJECTED_TAG not in fm.get("tags", []), (
        "closed as tried-and-rejected while its own landing was in flight")
    assert not [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_closed"]
    assert fin["outcome"]["acceptance"] == derived
    assert fin["outcome"]["item_verdict_refused"] == "rejected" and fin["outcome_refused"]


def test_the_same_verdict_with_nothing_landing_still_closes_the_item(isolated, monkeypatch, tmp_path):
    """The control: the guard keys on the landing, not on the word."""
    p = _write_item(isolated, 726)
    structured = {"acceptance": "rejected", "landed": False, "deferred_to": [], "spawned": [],
                  "clause_outcomes": [], "summary": "p50 412 ms -> 409 ms; inside the noise, not adopted"}
    fin = _run_turn(monkeypatch, tmp_path, 726, structured, land=False)
    assert _fm(p)["status"] == "done" and B.REJECTED_TAG in _fm(p)["tags"]
    assert fin["outcome"]["acceptance"] == "rejected" and fin["outcome_refused"] == ""


# ── the settle sweep: the review rung stands in for what was not reported ────

def _landing(item_id, rid, commit, outcome, *, clauses=("met", "met"), blocking=False):
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "review", "round_id": rid, "item_id": item_id, "ok": True,
                    "blocking": blocking, "attempt": 1,
                    "clauses": [{"clause": i, "verdict": v} for i, v in enumerate(clauses, 1)]},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "finished",
                    "round_id": rid, "stop_reason": "stop", "num_turns": 40, "outcome": outcome},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "promoted", "round_id": rid, "commit": commit}, path=S.LEDGER_PATH)
    S.append_event({"event": "settled", "commit": commit}, path=S.LEDGER_PATH)


def test_a_refused_verdict_with_nothing_to_derive_closes_on_the_reviews_all_met(isolated):
    """#1242 as it would go now: the turn's `rejected` is not taken, the
    outcome is unreported, and the review rung graded every clause met."""
    p = _write_item(isolated, 1242)
    refused, _ = B.settle_item_verdict(B.parse_outcome(OUTCOME_1242), landing_seen=True)
    _landing(1242, "SM_1242", "e2fc0754a215", refused, clauses=("met",) * 4)
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 1242, "closed": True, "acceptance": "met"}]
    fm = _fm(p)
    assert fm["status"] == "done" and fm["automod_landed"] == "e2fc0754a215"
    assert B.REJECTED_TAG not in fm["tags"]
    assert "review rung graded every acceptance clause met" in fm["activity_log"][-1]
    assert "`rejected` was not taken" in fm["activity_log"][-1]
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_landed"][-1]
    assert ev["acceptance_source"] == "code_review" and ev["closed"] is True


def test_a_turn_that_died_without_a_finalizer_closes_on_the_reviews_all_met(isolated):
    p = _write_item(isolated, 900)
    _landing(900, "SM_900", "aaaa1111bbbb", None, clauses=("met", "met", "met"))
    assert B.close_settled_items(S.LEDGER_PATH)[0] == {"item_id": 900, "closed": True,
                                                       "acceptance": "met"}
    assert "the turn ended without reporting an outcome" in _fm(p)["activity_log"][-1]


@pytest.mark.parametrize("clauses, blocking", [
    (("met", "unmet"), False), (("met", "post_landing"), False), (("met", "met"), True)])
def test_the_review_stands_in_only_when_it_graded_every_clause_met(isolated, clauses, blocking):
    p = _write_item(isolated, 901)
    _landing(901, "SM_901", "cccc2222dddd", None, clauses=clauses, blocking=blocking)
    assert B.close_settled_items(S.LEDGER_PATH)[0]["closed"] is False
    assert _fm(p)["status"] == "up_next" and "a human decides" in _fm(p)["activity_log"][-1]


def test_an_outcome_the_turn_did_report_is_never_overridden(isolated):
    """The author said a clause is not met; the grader said all met. The
    author's word stands, as it does for a vault landing."""
    p = _write_item(isolated, 902)
    reported = {"acceptance": "not_met", "landed": True, "deferred_to": [], "summary": "",
                "spawned": [], "clause_outcomes": [
                    {"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []},
                    {"clause": 2, "outcome": "not_met", "evidence": "t", "deferred_to": []}]}
    _landing(902, "SM_902", "eeee3333ffff", reported)
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 902, "closed": False, "acceptance": "not_met"}]
    assert _fm(p)["status"] == "up_next"


def test_the_newest_review_of_the_round_is_the_one_that_counts(isolated):
    S.append_event({"event": "review", "round_id": "SM_N", "item_id": 5, "ok": True, "ts": 100.0,
                    "blocking": False, "clauses": [{"clause": 1, "verdict": "met"}]},
                   path=S.LEDGER_PATH)
    assert B.code_review_outcome(S.LEDGER_PATH, "SM_N")["acceptance"] == "met"
    S.append_event({"event": "review", "round_id": "SM_N", "item_id": 5, "ok": True, "ts": 200.0,
                    "blocking": True, "clauses": [{"clause": 1, "verdict": "unmet"}]},
                   path=S.LEDGER_PATH)
    assert B.code_review_outcome(S.LEDGER_PATH, "SM_N") is None
    assert B.code_review_outcome(S.LEDGER_PATH, "SM_other") is None
    assert B.code_review_outcome(S.LEDGER_PATH, "") is None


def test_the_model_is_told_what_landed_means_where_it_answers(isolated, monkeypatch, tmp_path):
    """Three places, by when they are read. The template says how a clause is
    judged, because that shapes the turn's own report. What `landed` means is
    said by `automod_land`'s result — the last thing read before the turn ends
    — and by the finalizer's own prompt, which is read as the object is
    written; neither costs the length-bounded template a character."""
    import inspect
    from agent_mcp import automod as T
    flat = " ".join(I.PROMPT.split())
    assert "`not_met` means the change does not satisfy it, never that it is not promoted yet" in flat
    assert "`landed` is TRUE" in inspect.getsource(T._land_detached)

    _write_item(isolated, 77)
    seen: dict = {}

    async def turn(prompt, **kw):
        seen.update(kw)
        return {"text": "SPAWNED: none\n", "session_id": "s77", "stop_reason": "stop",
                "num_turns": 3, "errors": [], "structured": None, "structured_error": ""}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    asyncio.run(I.execute(_Item({"structured_outcome": True})))
    finalizer = " ".join(str(seen["final_schema_prompt"]).split())
    assert "`landed` is true if you called automod_land on a passed gate" in finalizer
    assert "no reason for `landed: false`, `not_met` or `rejected`" in finalizer
    assert "neither fits a round you landed with every clause met" in finalizer
