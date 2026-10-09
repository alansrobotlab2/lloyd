"""A write of `status: up_next` is answered by the ledger, not undone by it.

2026-10-04: of 168 `status_moved` rows in seven days, 77 were the reconciler
moving a promoted item back to `draft` (53 "never triaged", 24 "not for the
unattended loop"). The nightly promotion pass (autonomy task 35) wrote a status
the ledger did not back, the reconciler undid it minutes later, and the
reconciler's own line read to the next night's pass as a reason to promote
again — #2030 four nights running, held in quarantine with no reader at all.
#2041 lost its only owner the same way: owed-check answered `reopen`, the entry
was settled, and the `up_next` it wrote lasted two minutes.

These pin the replacement (`B.promotion_ruling`): the status moves only where
the reconciler would leave it, and a refused promotion does the thing that can
actually advance the item.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as BL
from scripts.automod import backlog as B, owed as O, state as S


def write_item(d: Path, item_id, *, status="draft", tags=("backlog",), extra=None) -> Path:
    fm = {"status": status, "priority": "medium", "board": "lloyd", "tags": list(tags),
          "created": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}
    fm.update(extra or {})
    path = d / f"{item_id}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nBody.\n",
                    encoding="utf-8")
    return path


def fm_of(path: Path) -> dict:
    return B._split_frontmatter(path.read_text(encoding="utf-8"))[0]


def _ev(**row):
    S.append_event(row, path=S.LEDGER_PATH)


def _write(**args) -> dict:
    return json.loads(BL._handle_write(args))


@pytest.fixture(autouse=True)
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


SPAWNED = ("spawned-by-triage",)


# ── The ruling ──────────────────────────────────────────────────────────────

def test_a_confirmed_contract_moves(board):
    write_item(board, 10)
    _ev(event="backlog_triage", item_id=10, verdict="confirmed", acceptance="it passes")
    ruling = B.promotion_ruling(10)
    assert ruling["move"] and ruling["outcome"] == "allowed"


def test_a_held_self_filed_item_is_released_to_triage_not_moved(board):
    p = write_item(board, 11, tags=SPAWNED)
    assert B.triage_pool(S.LEDGER_PATH) == ([], 1), "held: the state #2030 sat in"
    ruling = B.promotion_ruling(11, session_class="background")
    assert not ruling["move"] and ruling["outcome"] == "request"
    assert B.triage_pool(S.LEDGER_PATH) == ([], 1), "the ruling is pure: nothing released yet"
    B.apply_promotion_ruling(11, ruling, "blocks live work", session_class="background")
    pool, held = B.triage_pool(S.LEDGER_PATH)
    assert [i.id for i in pool] == [11] and held == 0
    assert fm_of(p)["status"] == "draft"
    # Asked again the next night: it is a candidate already, and no second row is written.
    again = B.promotion_ruling(11, session_class="background")
    assert again["outcome"] == "pooled"
    B.apply_promotion_ruling(11, again, "again", session_class="background")
    assert len(B._ledger_events(S.LEDGER_PATH, B.TRIAGE_REQUEST_EVENT)) == 1


def test_an_unattended_caller_is_capped_and_a_person_is_not(board):
    for i in range(20, 20 + B.TRIAGE_REQUEST_CAP + 1):
        write_item(board, i, tags=SPAWNED)
    for i in range(20, 20 + B.TRIAGE_REQUEST_CAP):
        ruling = B.promotion_ruling(i, session_class="background")
        assert ruling["outcome"] == "request"
        B.apply_promotion_ruling(i, ruling, "why", session_class="background")
    last = 20 + B.TRIAGE_REQUEST_CAP
    assert B.promotion_ruling(last, session_class="background")["outcome"] == "capped"
    assert B.promotion_ruling(last, session_class="chat")["outcome"] == "request"
    # The window is 24 h of the unattended callers' own rows.
    later = datetime.now(timezone.utc).timestamp() + B.TRIAGE_REQUEST_WINDOW_S + 1
    assert B.promotion_ruling(last, session_class="background", now=later)["outcome"] == "request"


def test_an_untriaged_item_already_in_the_pool_just_stays(board):
    write_item(board, 30)
    ruling = B.promotion_ruling(30, session_class="background")
    assert not ruling["move"] and ruling["outcome"] == "pooled"
    B.apply_promotion_ruling(30, ruling, "why")
    assert B._ledger_events(S.LEDGER_PATH, B.TRIAGE_REQUEST_EVENT) == []


@pytest.mark.parametrize("verdict", ["not_code", "unverifiable"])
def test_triaged_out_of_the_loop_with_nothing_owed_gets_an_owner(board, verdict):
    p = write_item(board, 31)
    _ev(event="backlog_triage", item_id=31, verdict=verdict)
    ruling = B.promotion_ruling(31)
    assert not ruling["move"] and ruling["outcome"] == "owe"
    assert "not for the unattended loop" in ruling["message"]
    B.apply_promotion_ruling(31, ruling, "the run is executable now")
    owed = O.entries_of(fm_of(p))
    assert len(owed) == 1 and owed[0]["kind"] == "decide"
    assert "the run is executable now" in owed[0]["what"]
    assert fm_of(p)["status"] == "draft"


def test_triaged_out_of_the_loop_and_already_owed_is_left_to_its_owner(board):
    p = write_item(board, 32, extra={"owed": [{"what": "decide it", "kind": "decide"}]})
    _ev(event="backlog_triage", item_id=32, verdict="not_code")
    ruling = B.promotion_ruling(32)
    assert not ruling["move"] and ruling["outcome"] == "ledger"
    B.apply_promotion_ruling(32, ruling, "why")
    assert [e["what"] for e in O.entries_of(fm_of(p))] == ["decide it"]


def test_an_item_the_loop_does_not_read_is_written_as_asked(board):
    write_item(board, 33, extra={"board": "alfie"})
    assert B.promotion_ruling(33)["move"]
    assert B.promotion_ruling(999)["move"], "not an open item: no opinion"


# ── The tool ────────────────────────────────────────────────────────────────

def test_the_tool_says_what_happened_and_keeps_the_rest_of_the_write(board):
    p = write_item(board, 40, tags=SPAWNED)
    out = _write(task_id=40, status="up_next", priority="high",
                 activity="promoted: premise re-measured live")
    assert out["success"] and out["status"] == "draft" and out["promotion"] == "request"
    assert "NOT moved" in out["message"] and "released to single triage" in out["message"]
    fm = fm_of(p)
    assert fm["status"] == "draft" and fm["priority"] == "high"
    assert any("status not moved to up_next" in str(line) for line in fm["activity_log"])
    assert [i.id for i in B.triage_pool(S.LEDGER_PATH)[0]] == [40]
    # What the reconciler would have had to undo is never written.
    assert 40 not in B.desired_statuses(S.LEDGER_PATH)


def test_the_tool_moves_a_confirmed_item_and_reports_it_plainly(board):
    p = write_item(board, 41)
    _ev(event="backlog_triage", item_id=41, verdict="confirmed", acceptance="it passes")
    out = _write(task_id=41, status="up_next")
    assert out == {"success": True, "task_id": 41, "created": False, "message": "Task updated"}
    assert fm_of(p)["status"] == "up_next"


def test_the_owed_entry_survives_the_tools_own_save(board):
    p = write_item(board, 42)
    _ev(event="backlog_triage", item_id=42, verdict="not_code")
    out = _write(task_id=42, status="up_next", activity="promoted: executable now")
    assert out["promotion"] == "owe"
    fm = fm_of(p)
    assert fm["status"] == "draft" and len(O.entries_of(fm)) == 1


def test_a_create_at_up_next_is_filed_as_draft_and_releases_nothing(board):
    out = _write(name="Blocks #7: a new finding", description="Body text.", board="lloyd",
                 status="up_next", tags=["spawned-by-autocode"])
    assert out["success"] and out["created"] and out["status"] == "draft"
    assert out["promotion"] == "created_draft"
    assert fm_of(next(board.glob("*.md")))["status"] == "draft"
    assert B._ledger_events(S.LEDGER_PATH, B.TRIAGE_REQUEST_EVENT) == []


def test_a_broken_ruling_is_the_old_write(board, monkeypatch):
    p = write_item(board, 43, tags=SPAWNED)
    monkeypatch.setattr(B, "promotion_ruling", lambda *a, **k: 1 / 0)
    assert _write(task_id=43, status="up_next")["message"] == "Task updated"
    assert fm_of(p)["status"] == "up_next"


# ── owed-check's reopen ─────────────────────────────────────────────────────

def _owing(d: Path, item_id: int) -> tuple[Path, list[dict]]:
    p = write_item(d, item_id)
    O.add_owed(p, ["decide"], kind="decide")
    return p, O.entries_of(fm_of(p))


def test_a_reopen_the_ledger_cannot_grant_keeps_the_entry_owed(board):
    p, entries = _owing(board, 50)
    _ev(event="backlog_triage", item_id=50, verdict="not_code")
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "reopen", "evidence": "x",
                                        "ruling": "the run is executable now"}], item_id=50)
    fm = fm_of(p)
    assert out["moved"] == "" and fm["status"] == "draft"
    assert out["remaining"] == 1 and not fm.get(O.SETTLED_KEY), "the #2041 hole: still owned"
    kept = O.entries_of(fm)[0]
    assert kept["rechecks"] == 1 and kept["recheck_after"]
    assert "a round cannot take it" in kept["note"]


def test_a_reopen_refused_past_the_recheck_bound_is_ruled_not_asked_again(board):
    p = write_item(board, 51, extra={"owed": [{"what": "decide", "kind": "decide",
                                               "rechecks": O.MAX_RECHECKS}]})
    _ev(event="backlog_triage", item_id=51, verdict="not_code")
    O.apply_verdict(p, O.entries_of(fm_of(p)),
                    [{"n": 1, "outcome": "reopen", "evidence": "x", "ruling": "again"}], item_id=51)
    fm = fm_of(p)
    assert not O.entries_of(fm)
    assert "not reopenable" in fm[O.SETTLED_KEY][-1]["ruling"]


def test_a_reopen_with_an_attempt_on_record_is_still_granted(board):
    p, entries = _owing(board, 52)
    _ev(event="backlog_triage", item_id=52, verdict="confirmed", acceptance="it passes")
    _ev(event="backlog_implement", item_id=52, phase="started")
    _ev(event="backlog_implement", item_id=52, phase="finished", outcome="aborted")
    out = O.apply_verdict(p, entries, [{"n": 1, "outcome": "reopen", "evidence": "x",
                                        "ruling": "try again"}], item_id=52)
    assert out["moved"] == "reopened"


def test_the_prompt_shows_why_the_last_answer_was_not_applied():
    from workers.sources import owed_check as OC
    block = OC._entries_block([{"what": "decide", "kind": "decide", "since": "2026-10-02",
                                "rechecks": 1, "note": "a round cannot take it"}], [0])
    assert "last answer not applied: a round cannot take it" in block
    assert "do it now" in OC.PROMPT


def test_a_missing_measurement_holds_nothing_out_of_promotion(board):
    """#2394 clause 2: the row that holds an item out must be a verdict.

    #2378 is the live case: its only triage row is `verdict: unverifiable,
    verdict_source: none` — the finalizer writing an empty turn as a judgement —
    and `promotion_ruling` answered `owe` with the parked reason, which is the
    same answer an item triage genuinely retired gets. A row with no measurement
    behind it is not a ruling, so it must not produce this outcome; the item
    stays an ordinary draft awaiting a triage that actually runs.

    The sibling rows below are the controls: a real `unverifiable` with a
    rendered source, and a row with no `verdict_source` field at all (the shape
    every pre-existing ledger row has), both still hold the item out.
    """
    p = write_item(board, 60)
    _ev(event="backlog_triage", item_id=60, verdict="unverifiable", check="",
        verdict_source="none",
        evidence="triage turn ended (stop) with no parseable verdict block")

    assert B.triaged_ids(S.LEDGER_PATH) == {}
    ruling = B.promotion_ruling(60, session_class="worker")
    assert ruling["outcome"] == "pooled", ruling
    assert not ruling["move"]
    assert "no triage verdict yet" in ruling["message"], ruling["message"]
    assert "not for the unattended loop" not in ruling["message"]
    assert B.desired_statuses(S.LEDGER_PATH).get(60) is None, \
        "nothing parks it"
    assert B.select_candidate(S.LEDGER_PATH).id == 60, \
        "and it is still what the next triage pass will read"
    assert fm_of(p)["status"] == "draft"

    _ev(event="backlog_triage", item_id=61, verdict="unverifiable",
        verdict_source="structured")
    write_item(board, 61)
    assert B.promotion_ruling(61)["outcome"] == "owe"
    _ev(event="backlog_triage", item_id=62, verdict="unverifiable")
    write_item(board, 62)
    assert B.triaged_ids(S.LEDGER_PATH)[62] == "unverifiable", \
        "a row with no verdict_source is a rendered verdict, as all history has"
