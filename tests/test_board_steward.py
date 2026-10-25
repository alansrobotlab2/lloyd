"""Cut 3 of senses-not-supervision: one steward turn beside the state machine.

Shipped with `apply: false`. Every pass records how far the steward's answer
agrees with what `desired_statuses` would have done, and that record — not a
week of hope — is what decides whether `apply` flips. The one rule that is
never the steward's: it cannot set `done`.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.automod import backlog as B
from workers.sources import board_steward as W


# ---------------------------------------------------------------------------
# the schema and the one hard rule
# ---------------------------------------------------------------------------

def test_done_is_not_a_status_the_steward_can_set():
    enum = W.STEWARD_SCHEMA["properties"]["moves"]["items"]["properties"]["status"]["enum"]
    assert "done" not in enum
    assert enum == list(W.STEWARD_STATUSES)


def test_parse_drops_a_done_move_even_if_the_grammar_let_it_through():
    parsed = W.parse_steward({"moves": [{"item_id": 5, "status": "done", "tags_add": [],
                                         "tags_remove": [], "note": "finished"},
                                        {"item_id": 6, "status": "up_next", "tags_add": [],
                                         "tags_remove": [], "note": "ok"}],
                              "next_pick": 6, "next_pick_reason": "r", "summary": "s"})
    assert [m["item_id"] for m in parsed["moves"]] == [6]


def test_apply_moves_refuses_done_belt_and_braces(monkeypatch):
    calls = []
    monkeypatch.setattr(B, "set_status", lambda *a, **k: calls.append(a) or True)
    monkeypatch.setattr(B, "tag_item", lambda *a, **k: True)
    W.apply_moves([{"item_id": 5, "status": "done", "tags_add": [], "tags_remove": [], "note": ""}])
    assert calls == []


def test_parse_clamps_and_tolerates_junk():
    assert W.parse_steward("nope") is None
    p = W.parse_steward({"moves": [{"item_id": "x"}, {"item_id": 3, "status": "draft",
                                                       "tags_add": ["a"] * 20, "tags_remove": [],
                                                       "note": "n " * 400}],
                         "next_pick": "bad", "next_pick_reason": "", "summary": ""})
    assert len(p["moves"]) == 1
    assert len(p["moves"][0]["tags_add"]) == 6
    assert len(p["moves"][0]["note"]) <= 300
    assert p["next_pick"] == 0


# ---------------------------------------------------------------------------
# agreement with the state machine
# ---------------------------------------------------------------------------

def test_agreement_separates_agree_disagree_abstain_and_missed():
    """The first dry-run counted the machine's deliberate abstentions (the
    stranded landings it parks for a human) as the steward being wrong. They
    are different things and are scored apart."""
    expected = {1: ("up_next", "why"), 2: ("draft", "why", True), 3: ("in_progress", "why")}
    current = {1: "draft", 2: "up_next", 3: "up_next", 4: "draft"}
    moves = [{"item_id": 1, "status": "up_next"},    # agrees
             {"item_id": 2, "status": "up_next"},    # machine says draft: disagree
             {"item_id": 4, "status": "up_next"}]    # machine has no opinion
    a = W.agreement(moves, expected, current)
    assert a["agree"] == [1]
    assert a["disagree"] == [2]
    assert a["no_opinion"] == [4]
    assert a["missed"] == [2, 3]
    assert a["machine_moves"] == 3
    # rate is over decisions both sides made: {1,2,3} judged, 1 agrees
    assert a["judged"] == 3
    assert a["rate"] == pytest.approx(1 / 3)


def test_the_steward_is_shown_every_pending_machine_move():
    """#898: confirmed, parked in draft, wanted in up_next by the machine —
    and invisible to the steward because nothing recent had touched it."""
    items = [_item(898, "draft"), _item(2, "up_next"), _item(5, "draft")]
    shown = W.board_view(items, touched=set(), pending={898}, max_items=10)
    assert [i.id for i in shown] == [898, 2]


def test_a_pending_move_survives_a_pool_bigger_than_the_cap():
    """The third dry-run's miss: 91 pool items, cap 80, the one pending item
    listed after them and cut off. Pending goes first, whatever the cap."""
    items = [_item(i, "up_next") for i in range(1, 92)] + [_item(898, "draft")]
    shown = W.board_view(items, touched=set(), pending={898}, max_items=80)
    assert shown[0].id == 898
    assert len(shown) == 80


def test_events_include_each_shown_items_own_history(tmp_path):
    ledger = tmp_path / "l.jsonl"
    rows = [{"event": "backlog_triage", "ts": 1, "item_id": 898, "verdict": "confirmed"},
            {"event": "backlog_triage", "ts": 2, "item_id": 7, "verdict": "stale"},
            {"event": "backlog_implement", "ts": 100, "item_id": 3, "phase": "started"}]
    ledger.write_text("\n".join(json.dumps(r) for r in rows))
    out = W.events_since(ledger, since_ts=50, for_items={898})
    assert [e["item_id"] for e in out] == [898, 3], "898's old triage rides along; 7's does not"


def test_agreement_has_no_rate_when_nothing_was_judged():
    """#1688 clause 1. This used to assert `rate == 1.0`, and that line was the
    bug: 76 of 93 live ticks judged nothing and every one of them read as
    perfect agreement on the record the `apply` flip is judged from. Nothing
    judged is no measurement, and the count says so."""
    a = W.agreement([], {1: ("draft", "w")}, {1: "draft"})
    assert a["rate"] is None
    assert a["judged"] == 0
    # One judged decision is a measurement again, whichever way it went.
    one = W.agreement([{"item_id": 1, "status": "up_next"}], {1: ("up_next", "w")}, {1: "draft"})
    assert one["judged"] == 1 and one["agree"] == [1] and one["rate"] == 1.0
    missed = W.agreement([], {1: ("up_next", "w")}, {1: "draft"})
    assert missed["judged"] == 1 and missed["missed"] == [1] and missed["rate"] == 0.0


# ---------------------------------------------------------------------------
# what the steward is shown
# ---------------------------------------------------------------------------

def _item(i, status="draft", tags=(), group=None):
    return SimpleNamespace(id=i, status=status, name=f"item {i}", created="2026-09-01T00:00:00",
                           body="b" * 1000, tags=list(tags), group=group, members=[], parent=None)


def test_board_view_is_pool_first_then_touched_drafts_then_flagged():
    items = [_item(1, "draft"), _item(2, "up_next"), _item(3, "in_progress"),
             _item(4, "draft"), _item(5, "draft", tags=("needs-human",))]
    shown = W.board_view(items, touched={4}, max_items=10)
    assert [i.id for i in shown] == [2, 3, 4, 5]


def test_board_view_is_bounded():
    items = [_item(i, "up_next") for i in range(200)]
    assert len(W.board_view(items, touched=set(), max_items=80)) == 80


def test_events_since_reads_only_shown_events_after_the_cursor(tmp_path):
    ledger = tmp_path / "l.jsonl"
    rows = [{"event": "backlog_triage", "ts": 10, "item_id": 1},
            {"event": "gate", "ts": 11, "round_id": "R"},          # not shown
            {"event": "backlog_implement", "ts": 12, "item_id": 1, "phase": "started"},
            {"event": "promoted", "ts": 5, "commit": "abc"}]        # before cursor
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    out = W.events_since(ledger, since_ts=9)
    assert [e["event"] for e in out] == ["backlog_triage", "backlog_implement"]


def test_the_prompt_names_the_vocabulary_and_forbids_done():
    text = W.build_prompt(events=[{"event": "backlog_triage", "ts": 1, "item_id": 1,
                                   "verdict": "confirmed"}],
                          items=[_item(1, "up_next")], n_open=5, since_ts=0)
    for phrase in ("`draft`", "`up_next`", "`in_progress`", "`done` is NOT yours",
                   "next_pick", "#1 [up_next]", "backlog_triage",
                   # the rule the first dry-run missed, and the umbrella/member split
                   "A triage verdict moves the item", "An `umbrella` is an ordinary confirmed item",
                   # the converse, missed on the first tick under the shared rule
                   "no triage verdict that sits in `up_next` goes back to",
                   # the rule #860 was refused on, forty-four ticks running
                   "A review refusal on the contract spends nothing either"):
        assert phrase in text, phrase
    assert len(text) < 20_000


# ---------------------------------------------------------------------------
# the pick reaches select_confirmed only when applying
# ---------------------------------------------------------------------------

def _cfg(monkeypatch, apply: bool):
    from app.config import CONFIG
    workers = dict(CONFIG.get("workers") or {})
    sources = dict(workers.get("sources") or {})
    sources["board-steward"] = {"enabled": True, "apply": apply}
    workers["sources"] = sources
    monkeypatch.setitem(CONFIG, "workers", workers)


def test_a_dry_run_pick_is_never_the_order(monkeypatch, tmp_path):
    _cfg(monkeypatch, apply=False)
    p = tmp_path / "pick.json"
    p.write_text(json.dumps({"item_id": 7, "ts": time.time()}))
    assert B.steward_pick(p) is None


def test_an_applied_fresh_pick_is_the_order(monkeypatch, tmp_path):
    _cfg(monkeypatch, apply=True)
    p = tmp_path / "pick.json"
    p.write_text(json.dumps({"item_id": 7, "ts": time.time()}))
    assert B.steward_pick(p) == 7


def test_a_stale_pick_is_ignored(monkeypatch, tmp_path):
    """A pick from a board that has since moved is worse than the sort."""
    _cfg(monkeypatch, apply=True)
    p = tmp_path / "pick.json"
    p.write_text(json.dumps({"item_id": 7, "ts": time.time() - 3 * 3600}))
    assert B.steward_pick(p) is None


def test_the_shipped_state_is_dry_run():
    from app.config import CONFIG
    cfg = CONFIG["workers"]["sources"]["board-steward"]
    assert cfg["enabled"] is True
    assert cfg["apply"] is False
    assert cfg["model"] == "primary"      # since the first live tick; see the source
    assert int(cfg["max_turns"]) >= 16


def test_the_prompt_tells_it_not_to_read_its_way_through_the_board():
    text = W.build_prompt(events=[], items=[_item(1, "up_next")], n_open=1, since_ts=0)
    assert "Do not open items or run tools" in text


def test_the_source_is_registered():
    from workers.sources import SOURCE_REGISTRY
    assert "board-steward" in SOURCE_REGISTRY


def test_the_steward_turn_cannot_write(monkeypatch):
    """A judgment, not an action: every writer and the automod tools are
    denied. The parse of its answer is the only thing that touches the board.
    """
    import inspect
    src = inspect.getsource(W.execute)
    for tool in ("Bash", "Edit", "Write", "backlog_write_task", "automod_start", "automod_land"):
        assert f'"{tool}"' in src, tool


def test_the_steward_reads_only_the_loops_boards():
    """The machine scopes itself to `DEFAULT_BOARDS`; a steward shown every
    board proposes moves on items that are not the loop's to touch, and the
    metric filed them under `no_opinion` every tick on 2026-09-13."""
    import inspect
    src = inspect.getsource(W.execute)
    assert "B.open_items, B.DEFAULT_BOARDS" in src
    assert "B.open_items, None" not in src


# ---------------------------------------------------------------------------
# board health: one definition, and the steward reads it
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

import yaml  # noqa: E402

from scripts.automod import state as S  # noqa: E402


def _write(d, iid, status, *, tags=("backlog",), hours_old=48, **fm):
    created = (datetime.now(timezone.utc) - timedelta(hours=hours_old)).isoformat()
    data = {"status": status, "priority": "medium", "created": created, "board": "lloyd",
            "tags": list(tags), **fm}
    (d / f"{iid}-item-{iid}.md").write_text(
        f"---\n{yaml.dump(data, default_flow_style=False)}---\n\n# item {iid}\n\nbody\n")


@pytest.fixture
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    return d


def _ev(**row):
    S.append_event(row, path=S.LEDGER_PATH)


def _ten_item_board(d):
    spawn = ("backlog", "spawned-by-triage")
    _write(d, 1, "draft", hours_old=1)                                   # pool (human)
    _write(d, 2, "draft", tags=spawn)                                    # quarantined
    _write(d, 3, "draft", tags=spawn + ("grouped",), group=9)            # grouped beats quarantined
    _write(d, 4, "draft", tags=("backlog", "needs-human"))               # needs-human beats triaged
    _ev(event="backlog_triage", item_id=4, verdict="confirmed", acceptance="a")
    _write(d, 5, "draft")                                                # triaged and parked
    _ev(event="backlog_triage", item_id=5, verdict="unverifiable")
    _write(d, 6, "draft", tags=spawn)                                    # released by a keep: pool
    _ev(event="backlog_group_triage", cluster_id="c-1", judged={"6": "keep"})
    _write(d, 7, "up_next")                                              # ready
    _ev(event="backlog_triage", item_id=7, verdict="confirmed", acceptance="it passes")
    _write(d, 8, "up_next")                                              # no acceptance: unready
    _ev(event="backlog_triage", item_id=8, verdict="confirmed", acceptance="")
    _write(d, 9, "up_next", tags=("backlog", "umbrella"), members=[3])   # umbrella, re-offered
    _ev(event="backlog_triage", item_id=9, verdict="confirmed", acceptance="all of them")
    _ev(event="backlog_implement", item_id=9, phase="finished", round_id="SM_9",
        stop_reason="max_turns", num_turns=150)
    done_at = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")
    _write(d, 10, "done", hours_old=240, completed=done_at)              # closed today


def test_board_health_partitions_the_board(board):
    _ten_item_board(board)
    h = B.board_health(S.LEDGER_PATH)
    assert h["open"] == {"draft": 6, "up_next": 3}
    assert h["draft"] == {"pool": 2, "quarantined": 1, "grouped": 1, "needs_human": 1,
                          "held": 0, "parked": 0, "triaged": 1, "total": 6}
    assert h["draft"]["total"] == sum(v for k, v in h["draft"].items() if k != "total")
    assert h["up_next"] == {"total": 3, "umbrellas": 1, "singles": 2, "never_attempted": 2,
                            "ready": 2, "unready": 1}
    assert h["flow"]["24h"] == {"created": 1, "closed": 1, "net": 0}
    assert h["flow"]["7d"] == {"created": 9, "closed": 1, "net": 8}
    assert h["self_spawned_open"] == 3
    assert h["implement_pool"] == {"ready": 2, "bound": 20, "floor": 20}


def test_board_flow_reads_each_stamp_in_its_writers_clock(board):
    """Both eras of a naive stamp, each read in the clock its own writer used.

    `app/backlog_move.LOCAL_STAMP_CUTOVER` — 2026-09-26T04:00, #1517 — splits the
    store's naive stamps. Below it a `created`/`updated` came from a box-clock
    surface (the MCP store, Mission Control's router, this module's old `new_item`)
    and reads as local; at or above it every writer stamps naive UTC
    (`app/backlog_move.now_stamp()`), so the numerals are the instant already.
    `completed` is UTC on both sides, since only `backlog.py`'s closers ever wrote it.

    The stamps are anchored to that constant, not to the wall clock this run happens
    to execute at. Deriving them from `datetime.now()` is what made this node expire:
    a legacy stamp's numerals run seven hours behind its instant (PDT), so once real
    `now` passed the cut-off plus 27 h (2026-09-27T07:00Z) the same fixture landed
    *above* the cut-off, was read as UTC — correctly — aged to 27 h, and `created`
    fell to 0 while `net` went negative. The behaviour under test never changed; the
    fixture's era did.

    Row 3 is the half this node never pinned, and the harm #1517 itself fixed: a
    naive `created` at or above the cut-off must not be shifted into local, which
    would place it seven hours in the future and outside a window whose cut is
    `t <= now` — the reading that ran from 2026-09-14 to #1517.
    """
    import os
    import time as _time
    from app.backlog_move import LOCAL_STAMP_CUTOVER
    old = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    _time.tzset()
    try:
        # Three hours past the cut-off, so one row's numerals sit below it and
        # another's above, permanently, whatever date this runs on.
        now = (LOCAL_STAMP_CUTOVER + timedelta(hours=3)).replace(tzinfo=timezone.utc)
        local = lambda h: (now - timedelta(hours=h)).astimezone().replace(tzinfo=None).isoformat()
        utc = lambda h: (now - timedelta(hours=h)).replace(tzinfo=None).isoformat()
        out_of_week = utc(7 * 24 + 24)  # 192 h: clear of the 7 d floor even shifted local
        _write(board, 1, "draft", created=local(20))   # in; read as UTC it is 27 h old
        _write(board, 2, "draft", created=local(30))   # out at 24 h, in at 7 d
        _write(board, 3, "draft", created=utc(2))      # in; shifted local it is +5 h
        _write(board, 4, "done", created=out_of_week, completed=utc(3))                  # closed: in
        _write(board, 5, "done", created=out_of_week, updated=local(26))                 # no completed: 26 h
        _write(board, 6, "done", created=out_of_week, completed=utc(30), updated=local(1))  # completed wins
        flow = B.board_flow(backlog_dir=board, now=now.timestamp())
        assert flow["24h"] == {"created": 2, "closed": 1, "net": 1}
        # Every excluded row is excluded by the window's edge, not by a stamp the
        # reader could not place: at 7 d all four dated events land.
        assert flow["7d"] == {"created": 3, "closed": 3, "net": 0}
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        _time.tzset()


def test_a_held_confirmation_is_its_own_draft_bucket_and_the_prompt_names_it(board):
    """Held is a verdict waiting for pool room, not a parked triage: it counts
    once, below needs-human and above triaged, and the steward is told."""
    _write(board, 1, "draft", tags=("backlog", B.HELD_TAG))
    _ev(event="backlog_triage", item_id=1, verdict="confirmed", acceptance="a", held=True)
    _write(board, 2, "draft", tags=("backlog", "needs-human", B.HELD_TAG))
    _ev(event="backlog_triage", item_id=2, verdict="confirmed", acceptance="a", held=True)
    _write(board, 3, "draft")                                              # released: triaged
    _ev(event="backlog_triage", item_id=3, verdict="confirmed", acceptance="a", held=True)
    _ev(event="backlog_confirm_released", item_id=3, reason="room", moved=False)
    h = B.board_health(S.LEDGER_PATH)
    assert (h["draft"]["held"], h["draft"]["needs_human"], h["draft"]["triaged"]) == (1, 1, 1)
    assert h["draft"]["total"] == sum(v for k, v in h["draft"].items() if k != "total") == 3
    text = W.build_prompt(events=[], items=[_item(1, "draft")], n_open=1, since_ts=0, health=h)
    assert "1 confirmed and held for implement-pool room" in text


def test_the_prompt_carries_board_health_and_renders_without_it():
    h = {"open": {"draft": 6}, "draft": {"total": 6, "pool": 2, "quarantined": 1, "grouped": 1,
                                         "needs_human": 1, "triaged": 1},
         "up_next": {"total": 3, "umbrellas": 1, "singles": 2, "never_attempted": 2,
                     "ready": 2, "unready": 1},
         "flow": {"24h": {"created": 5, "closed": 2, "net": 3}, "7d": {}},
         "self_spawned_open": 3, "landed_items_7d": 4,
         "implement_pool": {"ready": 2, "bound": 20, "floor": 20}}
    text = W.build_prompt(events=[], items=[_item(1, "up_next")], n_open=1, since_ts=0, health=h)
    block = text[text.index("<board_health>"):text.index("</board_health>")]
    assert "draft 6: 2 triageable, 1 quarantined" in block
    assert "flow 24h: 5 created, 2 closed, net +3" in block
    assert "Lead your `summary` with the 24-hour net flow" in text
    bare = W.build_prompt(events=[], items=[_item(1, "up_next")], n_open=1, since_ts=0)
    assert "<board_health>\n(unavailable)\n</board_health>" in bare


def test_the_steward_row_records_the_board_health(board, monkeypatch):
    from workers.sources import _common as C
    _ten_item_board(board)
    seen = {}

    async def turn(prompt, **kw):
        seen["prompt"] = prompt
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": [], "next_pick": 0, "next_pick_reason": "",
                               "summary": "net 0 today"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out["status"] == "success"
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "board_steward"][-1]
    assert row["board_health"]["draft"]["total"] == 6
    assert "<board_health>\nopen: draft 6, up_next 3" in seen["prompt"]


def test_the_steward_is_told_a_held_confirmation_stays_draft_and_sees_its_release(tmp_path):
    text = W.build_prompt(events=[], items=[_item(1, "draft", tags=(B.HELD_TAG,))], n_open=1, since_ts=0)
    assert "held: true" in text and "Do not move a held item to" in text
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps({"event": "backlog_confirm_released", "ts": 10, "item_id": 1}) + "\n")
    assert [e["event"] for e in W.events_since(ledger, since_ts=0)] == ["backlog_confirm_released"]


# ── #1210: closed items a person still owes, and the steward's rule for them ──

def test_board_health_counts_closed_items_a_person_still_owes(board):
    """#1210 clause 4. `draft.needs_human` was the only number on the board that
    spoke for 'a person owes this', and it could only see drafts — so the moment
    a landed-met item closes carrying the tag, every count the steward reads
    stops showing it. `closed_needs_human` is the closed-side half of that
    bucket, and it does not leak into the open counts."""
    _ten_item_board(board)                              # item 4: draft + needs-human
    _write(board, 11, "done", tags=("backlog", "needs-human"))
    _write(board, 12, "done")
    h = B.board_health(S.LEDGER_PATH)
    assert h["closed_needs_human"] == 1, "only the closed item carrying the tag"
    assert h["draft"]["needs_human"] == 1, "the draft-side bucket is unchanged"
    assert h["open"] == {"draft": 6, "up_next": 3}, "a closed item is not open work"


def test_the_steward_is_told_a_met_landing_closes_carrying_needs_human(board):
    """#1210 clause 5. The prompt told the steward that a `met` landing with a
    person owed a check belongs in `draft` + `needs-human` — the rule #1210
    removed, and the one the steward's agreement is scored against. It now says
    `done` carrying `needs-human`, and `done` is still not a status it can set:
    the closer is the sweep, not the judgment."""
    prompt = W.build_prompt(events=[], items=[], n_open=0, since_ts=0.0, health=None)
    assert "acceptance `met` with a person still owed a check → `draft` + `needs-human`" not in prompt
    assert "`met` with a person still owed a check → `done` carrying `needs-human`" in prompt
    assert "done" not in W.STEWARD_STATUSES
    assert W.parse_steward({"moves": [{"item_id": 4, "status": "done", "tags_add": [],
                                       "tags_remove": [], "note": "x"}]})["moves"] == []


def test_the_summary_leads_with_the_steward_s_judgment_not_the_counts(board, monkeypatch):
    """Clause 4 of #1606: the outcome has to survive the 500-character column.

    The record used to read
    `0 move(s) proposed (dry run), agreement 100% (0 agree, …, 0 missed); next
    pick #1596: <the board state>` — ~110 characters of machine-derived counts
    in front of the one sentence only the steward wrote. `parse_steward` allows
    that sentence 400 characters and the counts take ~110, so the record overflows
    on a good day, and what falls off the end is the tail of the judgment: 77 of
    the 154 board-steward rows in `runs` are exactly 500 characters and 63 of them
    are cut mid-word ("…guardian observati"). Leading with the counts therefore
    deleted the only outcome in the row and kept the boilerplate that explains it.
    """
    import asyncio

    from workers.pool import normalize_result
    from workers.queue import SUMMARY_MARKER, SUMMARY_MAX_CHARS
    from workers.sources import _common as C

    _ten_item_board(board)
    judgment = "Net flow 24 h is +43, " * 18            # 396 characters, under the 400 cap
    assert len(judgment) < 400, "parse_steward caps the sentence at 400"

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": [], "next_pick": 0, "next_pick_reason": "",
                               "summary": judgment}}

    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out["status"] == "success"

    # The steward's sentence is first, and the counts follow it.
    assert out["summary"].startswith(judgment), \
        f"the judgment is not at the head: {out['summary'][:60]!r}"
    assert out["summary"].index("move(s)") > len(judgment)
    assert out["summary"].index("agreement") > out["summary"].index("move(s)")

    # And through the cap the column applies, the judgment is still whole and
    # the counts are the part that got cut — which is the whole point of the
    # reordering, and the half that a source-side assertion cannot see.
    recorded = normalize_result(_run_item(), out)["summary"]
    assert len(recorded) <= SUMMARY_MAX_CHARS
    assert recorded.endswith(SUMMARY_MARKER), "this record is long enough to truncate"
    assert recorded.startswith(judgment), "the outcome did not survive the cap"
    assert recorded.index(SUMMARY_MARKER) > len(judgment) + len(" — 0 move(s)")
    assert "move(s)" in recorded, "the counts were dropped instead of the overflow"


def _run_item():
    """A stand-in queue item for the pool's normaliser.

    `_item` above is this module's backlog-item factory and takes an id, so the
    run-record helper has its own name. `normalize_result` reads `source` for the
    declined marker and `payload` for the task id.
    """
    return SimpleNamespace(source=W.NAME, payload={})


# ── #1688: a tick that judged nothing reports no agreement percentage ─────

def _steward_tick(board, monkeypatch, *, moves=(), expected=None):
    """Run one dry-run tick on the ten-item board and return (result, ledger row).

    `expected` replaces `desired_statuses` so a test can hand the machine an
    opinion (or none) without building the ledger history that produces one.
    """
    import asyncio

    from workers.sources import _common as C

    _ten_item_board(board)
    if expected is not None:
        monkeypatch.setattr(B, "desired_statuses", lambda ledger: expected)

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": list(moves), "next_pick": 0, "next_pick_reason": "",
                               "summary": "net 0 today"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out["status"] == "success"
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "board_steward"][-1]
    return out, row


def test_a_tick_that_judged_nothing_says_agreement_was_not_measured(board, monkeypatch):
    """#1688 clauses 2 and 3. The live row read `agreement 100% (0 agree, 0
    disagree, 0 where the machine abstains, 0 missed)`; it now names the count
    and prints no percentage, and the meta and the ledger row carry `judged: 0`
    beside a `rate` of None."""
    out, row = _steward_tick(board, monkeypatch, expected={})
    assert "0 judged" in out["summary"]
    assert "not measured" in out["summary"]
    assert "%" not in out["summary"], out["summary"]
    assert out["meta"]["agreement"]["judged"] == 0
    assert out["meta"]["agreement"]["rate"] is None
    assert row["agreement"]["judged"] == 0
    assert row["agreement"]["rate"] is None


def test_a_tick_that_judged_something_still_prints_the_percentage(board, monkeypatch):
    """#1688 clauses 2 and 3, the other side: one agree, one missed — two
    judged decisions, 50%, and the count rides in the meta and the ledger row
    so the flip condition is a count rather than a rate."""
    expected = {7: ("draft", "w"), 1: ("up_next", "w")}           # both are pending moves
    moves = [{"item_id": 7, "status": "draft", "tags_add": [], "tags_remove": [], "note": "x"}]
    out, row = _steward_tick(board, monkeypatch, moves=moves, expected=expected)
    assert "agreement 50% of 2 judged" in out["summary"], out["summary"]
    assert "not measured" not in out["summary"]
    assert out["meta"]["agreement"]["judged"] == 2
    assert out["meta"]["agreement"]["rate"] == pytest.approx(0.5)
    assert row["agreement"]["judged"] == 2
    assert row["agreement"]["rate"] == pytest.approx(0.5)


# ── #2197: the grammar has a ceiling, so a truncated tick is not a budget ─────
#
# 18 board-steward runs between 2026-10-01 and 2026-10-04 failed with
# `output truncated at 8192 tokens — raise harness.finalizer.max_tokens` (20
# failed vs 134 success over the window; `backlog/data/workers-runs-2197.db` in the
# vault is the extracted witness — 176 board-steward runs since 10-01, 18 of them
# naming that message and none naming the other one, and `backlog/data/workers.db`
# is another item's witness whose `runs` table must stay empty, which is why
# this extract has its own file: see `backlog/data/workers-runs-2197.witness.md`,
# amended clause 5, and vault `d9c3b5b0` for the same collision on #2087).
# What those 18 got before they stopped, counted off the witness bytes: 17 emitted
# an empty moves list and stopped their opener at `"next_pick":`, 1 reached one real
# move, and of the 200-character excerpts the finalizer stores, 11 are whitespace
# after that opener while 7 go on to spit stray digits and text (`2026`, `185`,
# `20261009`). No shape is a long legitimate answer, and `max_tokens` was already
# 8192 and was never the cause: an uncapped node is read as "this could go on
# forever", so the finalizer could not rule the budget out and blamed it (#1706
# found exactly this for deep-research; this is the same source's turn). The caps
# below make the object ungrammatical past the answer, and — the part the 18
# failures could not get — make the message describe the failure.

#: One of the 18 failures, replayed as a SHAPE rather than as stored bytes, because
#: the store cannot give the bytes back: `app/harness/finalizer.py:279-285` files the
#: failure with `content[:200]!r`, so 200 characters is all a run ever records and
#: what survives is an excerpt, not a completion.
#:
#: How much of this fixture is captured bytes, measured against the stored excerpts
#: in the witness (the longest common prefix, character by character, over all 18):
#: 30 for the 2026-10-04T08:00 run, which this fixture opens like
#: (`{"moves":[],"next_pick":` then six whitespace characters); 24 or more for 10 of
#: the 18; one at 10, which is the row that gets a real move into `moves`; and 9 for
#: the 7 that write the empty list SPACED — `{"moves": [], "next_pick":` — and so
#: part company at `{"moves":`. Everything past the prefix is the shape EXTENDED: the
#: same 11-character whitespace block repeated out to ~8.8 KB, where the real
#: excerpts' whitespace is irregular and ends where the store cuts it.
#:
#: That substitution is the right one because the finalizer's branch reads the shape
#: (content that cannot close an open object, `finish_reason: length`) and
#: `_schema_is_bounded(schema)`, never a character count — and the token count its
#: message quotes comes from the faked `usage`, not from this string's length.
_CAPTURED_TRAILING_WHITESPACE = ('{"moves":[],"next_pick":'
                                 + "\n\n\t\n\n\n \n  \n" * 800)


def _unbound(node):
    """A copy of a schema with every cap stripped — the schema as it was before
    #2197, used as the control beside the replay."""
    if isinstance(node, dict):
        return {k: _unbound(v) for k, v in node.items()
                if k not in ("maxLength", "maxItems", "minItems")}
    if isinstance(node, list):
        return [_unbound(v) for v in node]
    return node


def _normalised(length: int) -> str:
    """A whitespace-normal string of exactly `length` characters."""
    s = " ".join(["word"] * (length // 5))
    return s + "x" * (length - len(s))


#: The five caps `app.harness.finalizer._node_is_bounded` actually consults. It
#: asks a string for `maxLength` and asks an array only whether its ITEMS are
#: bounded, so it never reads `maxItems` at all — which is why `tags_add` needed
#: a `maxLength` on the tag string as well as a count cap: with the count capped
#: and the item open, the helper still called the whole schema open-ended.
_STRING_CAP_SITES = (
    ("properties", "moves", "items", "properties", "note"),
    ("properties", "moves", "items", "properties", "tags_add", "items"),
    ("properties", "moves", "items", "properties", "tags_remove", "items"),
    ("properties", "next_pick_reason"),
    ("properties", "summary"),
)


def _descend(node, path):
    for key in path:
        node = node[key]
    return node


def test_every_node_of_the_steward_schema_is_bounded():
    """Clause 1: the helper the finalizer itself consults says this schema has a
    ceiling, every node the item names carries a positive cap, and re-opening any
    one of them in a copy turns the helper red.

    The mutation loop is the half that keeps the pin honest — the drift it exists
    for is one field quietly losing its cap while the rest still reads as bounded.
    It runs over the five `maxLength` sites, because `_node_is_bounded` ignores
    `maxItems` entirely; the three count caps are pinned by the value assertions
    and by the parse clamp, and the two tag arrays are bounded for the helper only
    because their ITEM string is capped."""
    import copy

    from app.harness import finalizer as F

    props = W.STEWARD_SCHEMA["properties"]
    move_props = props["moves"]["items"]["properties"]
    assert F._schema_is_bounded(W.STEWARD_SCHEMA), (
        "the finalizer still reads this schema as open-ended, so a truncated "
        "tick is blamed on harness.finalizer.max_tokens")
    assert props["moves"]["maxItems"] == W.MOVES_MAX > 0
    for name in ("tags_add", "tags_remove"):
        assert move_props[name]["maxItems"] == W.TAGS_MAX > 0, name
        assert move_props[name]["items"]["maxLength"] == W.TAG_MAX > 0, name
    assert move_props["note"]["maxLength"] == W.NOTE_MAX > 0
    assert props["next_pick_reason"]["maxLength"] == W.REASON_MAX > 0
    assert props["summary"]["maxLength"] == W.SUMMARY_MAX > 0

    for path in _STRING_CAP_SITES:
        probe = copy.deepcopy(W.STEWARD_SCHEMA)
        _descend(probe, path).pop("maxLength")
        assert not F._schema_is_bounded(probe), (
            f"re-opening {'.'.join(path)} left the helper saying bounded, so "
            "this test is not reading the caps")


def test_the_caps_sit_above_the_largest_reply_a_real_tick_ever_sent():
    """Clause 2: the caps are not tight, and a reply AT them survives the parse
    path unchanged.

    Two populations, both named. The run population the caps must fit inside — 176
    board-steward runs since 2026-10-01, 20 failed / 134 success / 22 skipped, 18 of
    them naming the budget message and 0 the divergence one — is the extracted
    witness `backlog/data/workers-runs-2197.db`, whose marker
    `backlog/data/workers-runs-2197.witness.md` carries the re-derive commands. The
    shape population is the 378 `board_steward` rows in the promotion ledger
    (`~/.local/state/lloyd-automod/promotions.jsonl`, measured 2026-10-04 — the run
    store cannot supply this population because it never keeps the reply: `
    response_json` is empty in all 176 witness rows): the largest legitimate reply is
    27 moves, the longest `note` and the longest
    `next_pick_reason` are 300 characters each and the longest `summary` is 400 —
    all three sit AT `parse_steward`'s ceilings because that function has been
    cutting them there all along. The most tags ever carried in one direction in
    one move is 2, across 12 distinct tag names whose longest is
    `review-disagreement` at 19 characters. So `MOVES_MAX` 40 is above the
    observed 27, `TAGS_MAX` 6 above the observed 2 and `TAG_MAX` 40 above the
    observed 19, while the three prose caps are the parse ceilings themselves: a
    cap ABOVE a parse ceiling would let the grammar admit text the parser then
    throws away, and the record of the steward's decision would stop being the
    answer it gave."""
    for length in (300, 400, W.NOTE_MAX, W.REASON_MAX, W.SUMMARY_MAX):
        built = _normalised(length)
        assert len(built) == length and built == " ".join(built.split()), (
            f"{length} is not buildable whitespace-normal")
    moves = [{"item_id": 1000 + i, "status": "up_next",
              "tags_add": [f"tag-{i}"] * 1 if i % 2 else [],
              "tags_remove": ["needs-human", "confirmed-held"] if i % 3 else [],
              "note": _normalised(W.NOTE_MAX)} for i in range(W.MOVES_MAX)]
    reply = {"moves": moves, "next_pick": 2197,
             "next_pick_reason": _normalised(W.REASON_MAX),
             "summary": _normalised(W.SUMMARY_MAX)}
    assert len(reply["moves"]) == W.MOVES_MAX > 27, (
        "the cap is not above the observed maximum")
    parsed = W.parse_steward(reply)
    assert parsed == reply, "the parse path altered an answer that is legal at the caps"
    # And the largest reply a real tick has ever sent — 27 moves, 13 under the cap —
    # has to survive the parse path whole as well, not just a reply at the cap.
    assert W.parse_steward({**reply, "moves": moves[:27]})["moves"] == moves[:27]


def test_a_reply_past_the_caps_is_still_clamped_by_the_parser():
    """The clamp side of the same number, so the caps are not only a request to
    the decoder: guided decoding is asked to stop at 40 moves and 300 characters,
    and if it does not, the parse path still does."""
    reply = {"moves": [{"item_id": i, "status": "draft", "tags_add": ["t"] * 9,
                        "tags_remove": [], "note": "n" * 400} for i in range(1, 42)],
             "next_pick": 5, "next_pick_reason": "r" * 500, "summary": "s" * 500}
    parsed = W.parse_steward(reply)
    assert len(parsed["moves"]) == W.MOVES_MAX == 40, len(parsed["moves"])
    assert len(parsed["moves"][0]["note"]) == W.NOTE_MAX
    assert len(parsed["moves"][0]["tags_add"]) == W.TAGS_MAX
    assert len(parsed["next_pick_reason"]) == W.REASON_MAX
    assert len(parsed["summary"]) == W.SUMMARY_MAX


async def test_a_truncated_tick_is_blamed_on_divergence_and_not_on_the_budget(
        monkeypatch):
    """Clause 3: the captured completion, replayed through the real finalizer
    against the real schema, with only the HTTP leg faked.

    Two assertions, and the second is the one that proves the diff did it: the
    same bytes against the same schema with its caps stripped must STILL say
    `raise harness.finalizer.max_tokens`. Content did not change between the 18
    failures and this replay; the schema did, and the branch follows the schema."""
    from app.harness import finalizer as F

    assert _CAPTURED_TRAILING_WHITESPACE.startswith('{"moves":[],"next_pick":')
    assert _CAPTURED_TRAILING_WHITESPACE[24:].strip() == "", (
        "the fixture must be the captured shape: an opener that never closes and "
        "nothing but whitespace after it, which is what the 18 stored excerpts hold")
    with pytest.raises(json.JSONDecodeError):
        json.loads(_CAPTURED_TRAILING_WHITESPACE)
    monkeypatch.setattr(F.httpx, "AsyncClient", _fake_engine(
        _CAPTURED_TRAILING_WHITESPACE, finish_reason="length", completion_tokens=8192))
    kwargs = dict(base_url="http://engine:8097", model="primary",
                  chat_messages=[{"role": "user", "content": "Restate it."}],
                  tools=None, max_tokens=8192)

    obj, err, usage = await F.run_finalizer(schema=W.STEWARD_SCHEMA, **kwargs)
    assert obj is None and usage["output_tokens"] == 8192, usage
    assert "diverged" in err, err
    assert "not a budget" in err, err
    assert "raise harness.finalizer.max_tokens" not in err, err

    obj2, err2, _ = await F.run_finalizer(schema=_unbound(W.STEWARD_SCHEMA), **kwargs)
    assert obj2 is None
    assert "raise harness.finalizer.max_tokens" in err2, (
        "the control no longer reproduces the 18 failures, so this test would "
        "pass even if the caps were removed")


class _Resp:
    def __init__(self, content, finish_reason, completion_tokens):
        self.status_code = 200
        self.text = ""
        self._c, self._f, self._t = content, finish_reason, completion_tokens

    def json(self):
        return {"choices": [{"message": {"content": self._c},
                             "finish_reason": self._f}],
                "usage": {"completion_tokens": self._t}}


def _fake_engine(content, *, finish_reason="stop", completion_tokens=180):
    """The engine minus the network, same shape as
    `tests/test_deep_research_source.py::_fake_engine`."""
    resp = _Resp(content, finish_reason, completion_tokens)

    class _Cli:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return resp

    return _Cli


def test_the_finalizer_budget_stays_8192_on_both_sides_of_the_reader():
    """Clause 4: 8192 in the code default and in `config.yaml`, checked through
    the reader production uses (`app.mcp_discovery._get_harness_kwargs`).

    The two sides are `app/harness/options.py:374` (`finalizer_max_tokens: int =
    8192`) and `config.yaml:744` (`max_tokens: 8192`).

    This is not the pin at `app/harness/tests/test_finalizer.py::
    test_the_default_budget_is_8192_and_config_agrees` repeated: that node asserts
    both NUMBERS — the `RunOptions` default, and a regex over the `harness.finalizer`
    block of the config TEXT — but never the leg between them, which is what this
    adds: the config side is read back through the code that consumes it. The failure
    mode a regex cannot see is a config whose value sits under a key the reader
    ignores — that node stays green and the loop silently runs on the code default.
    It matters to THIS diff because 8192 is the budget the caps are sized to fit
    inside: the loop-forbidden fix for a truncated tick is to raise it, and the caps
    are the reason that is not the fix.
    """
    from app.harness.options import RunOptions
    from app.mcp_discovery import _get_harness_kwargs

    assert RunOptions(model="primary").finalizer_max_tokens == 8192
    assert _get_harness_kwargs()["finalizer_max_tokens"] == 8192, (
        "config.yaml no longer reaches the option the loop reads")


# ── #2370: a diverged tick is drawn twice, and the row says which draw failed ─
#
# Only the HTTP leg is faked below: the tick runs the real `_maybe_finalize`
# against the real `STEWARD_SCHEMA`, so the error the steward publishes is the
# one the loop produced, attempt count included. `_fake_engine` cannot serve
# these — it returns the SAME response to every POST, and the whole question is
# how many posts a tick made.

def _drive_steward_tick(board, monkeypatch, responses, *, seed=True):
    """One board-steward tick over a queued engine; returns the row and the bodies.

    `run_prompt_in_session` is replaced by a call into the REAL
    `loop._maybe_finalize`, which is where the re-sample lives: faking it with a
    canned dict would let this file assert an `attempts=2` string no code wrote.
    The returned list is every finalizer request body in order, so a count of
    re-samples is a count of requests on the wire.

    `seed=False` skips `_ten_item_board`, which is how #3088's nodes drive a
    SECOND tick over the window the first was shown: the fixture stamps its
    events with `time.time()`, so re-seeding between ticks would move the window
    and make two rows sharing a `since_ts` trivially different things.
    """
    from app.harness import finalizer as F
    from app.harness import loop as L
    from app.harness.options import RunOptions
    from workers.sources import _common as C

    posted: list[dict] = []
    left = list(responses)

    class _Cli:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            posted.append(json)
            assert left, "the tick issued more finalizer requests than were queued"
            return left.pop(0)

    monkeypatch.setattr(F.httpx, "AsyncClient", _Cli)

    async def turn(prompt, **kw):
        parsed, error = await L._maybe_finalize(
            options=RunOptions(model="primary", final_schema=W.STEWARD_SCHEMA,
                               finalizer_max_tokens=8192),
            stop_reason="stop",
            chat_messages=[{"role": "user", "content": prompt}],
            tools=None, total_usage={})
        return {"text": "", "session_id": "s", "stop_reason": "stop",
                "num_turns": 1, "structured": parsed, "structured_error": error}

    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    if seed:
        _ten_item_board(board)
    return asyncio.run(W.execute(SimpleNamespace(payload={"apply": False}))), posted


_DIVERGED_TICK = _Resp(_CAPTURED_TRAILING_WHITESPACE, "length", 8192)


def test_a_tick_that_fails_after_the_re_sample_publishes_attempts_2(board, monkeypatch):
    """Clause 5. Both draws ended at the cap inside a capped grammar, so the row
    has to say the tick was drawn twice AND keep saying the token budget is not
    the reason — the phrase #1706 put there is what tells the next reader not to
    raise `harness.finalizer.max_tokens`.

    `attempts=2` is asserted inside the first 500 characters on purpose: that is
    the `runs.summary` column, and the message ends in 200 characters of
    captured completion, so a marker after them is the first thing a real row
    drops."""
    out, posted = _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK,
                                                           _DIVERGED_TICK])
    assert out["status"] == "failed", out
    assert len(posted) == 2, f"one re-sample expected, {len(posted)} requests made"
    assert "not a budget" in out["summary"], out["summary"]
    assert "diverged at 8192 tokens" in out["summary"], out["summary"]
    assert "attempts=2" in out["summary"][:500], (
        "the attempt count did not survive the column: " + repr(out["summary"]))
    assert [b["max_tokens"] for b in posted] == [8192, 8192], (
        "the re-sample asked for a different budget than the attempt it replaced")


def test_a_tick_that_did_not_need_the_re_sample_publishes_no_attempt_count(
        board, monkeypatch):
    """Clause 5's other shape, and the exclusion holding end to end: the first
    draw never left thinking, which is the BUDGET under a capped grammar
    (#1431), so no re-sample is issued and the row says nothing about
    attempts — one request, and the advice is still `raise
    harness.finalizer.max_tokens`."""
    out, posted = _drive_steward_tick(
        board, monkeypatch,
        [_Resp(content=None, finish_reason="length", completion_tokens=8192)])
    assert out["status"] == "failed", out
    assert len(posted) == 1, "a budget truncation must not be re-drawn"
    assert "raise harness.finalizer.max_tokens" in out["summary"], out["summary"]
    assert "attempts" not in out["summary"], out["summary"]


def test_a_tick_the_re_sample_recovers_publishes_no_failure_at_all(board, monkeypatch):
    """The payoff, at the seam that writes the row. The second draw parses, so
    the tick succeeds and there is no `no structured answer` row to read: 6 of
    the 10 divergences of 2026-10-04→10-07 got this recovery on the NEXT
    scheduled tick, ~15 minutes late and with the watermark un-advanced so the
    window re-read whole; the re-sample gets it in the same tick, and `moves`
    and `next_pick` reach the ledger of the run that drew them."""
    out, posted = _drive_steward_tick(
        board, monkeypatch,
        [_DIVERGED_TICK,
         _Resp(json.dumps({"moves": [], "next_pick": 0, "next_pick_reason": "",
                           "summary": "nothing to move this pass"}), "stop", 180)])
    assert len(posted) == 2, out
    assert out["status"] == "success", out
    assert "no structured answer" not in out["summary"], out["summary"]
    assert "attempts" not in out["summary"], out["summary"]


# ── #3088: a diverged tick publishes itself, and keeps the window it was shown ─
#
# 19 divergence failures over 511 board-steward ticks from 2026-10-05 to
# 2026-10-20 wrote no `board_steward` ledger row at all: `execute` returned the
# failed record two lines before `_write_state` and `S.append_event`. Each node
# below drives a REAL diverged tick — the captured shape drawn twice through the
# real `loop._maybe_finalize` against the real `STEWARD_SCHEMA` — and reads back
# the promotion ledger and `board_steward.json` themselves, so what is asserted is
# the artefact a later reader finds, not the return value that already existed.

#: One live divergence message, as the queue store records it. Re-read it with:
#: `sqlite3 -readonly ~/lloyd-data/workers.db "SELECT substr(summary,1,120) FROM
#: runs WHERE source='board-steward' AND summary LIKE '%diverged%' ORDER BY
#: started_at DESC LIMIT 1"`. The source string is `board-steward` WITH A HYPHEN:
#: `source='board_steward'` matches 0 rows, which reads as "the steward never runs".
_LIVE_DIVERGENCE = ("[attempts=2] finalizer failed: generation diverged at 8192 "
                    "tokens — every field this schema admits is capped")


def _steward_rows():
    """Every `board_steward` row in the fixture's ledger, oldest first."""
    return [e for e in S.read_events(path=S.LEDGER_PATH, limit=1000)
            if e.get("event") == "board_steward"]


def _two_diverged_ticks_over_one_window(board, monkeypatch):
    """Two ticks, each drawn twice into the same divergence, over one un-moving window.

    The watermark is seeded an hour back so `since_ts` is a number with meaning
    rather than the `0.0` of a missing state file, and the second tick is driven
    `seed=False`: re-seeding would stamp a fresh set of events, and the two ticks
    would no longer have been shown the same window.
    """
    since = time.time() - 3600.0
    W._write_state({"last_ts": since, "last_run": since, "last_session": "seeded"})
    first, _ = _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK, _DIVERGED_TICK])
    second, _ = _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK, _DIVERGED_TICK],
                                    seed=False)
    assert (first["status"], second["status"]) == ("failed", "failed"), (first, second)
    return _steward_rows(), W._read_state(), since


def test_a_diverged_tick_publishes_a_ledger_row_naming_the_divergence(board, monkeypatch):
    """Clause 1: a tick whose finalizer returns no structured answer appends a
    `board_steward` event saying the tick diverged and carrying the finalizer's own
    `structured_error`, where today no row is written at all.

    The row is the whole item: four sampled live failures had zero `board_steward`
    rows inside their own [started_at, completed_at] windows, so the ledger could
    not show that any of them had been tried.
    """
    out, _ = _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK, _DIVERGED_TICK])
    assert out["status"] == "failed", out
    rows = _steward_rows()
    assert len(rows) == 1, f"one diverged tick, {len(rows)} board_steward rows: {rows}"
    row = rows[0]
    assert row["diverged"] is True, row
    assert "diverged" in row["summary"], row
    err = out["summary"].split("no structured answer: ", 1)[1]
    assert row["structured_error"] == err[:W.DIVERGENCE_ERROR_MAX], row["structured_error"]
    assert _LIVE_DIVERGENCE in row["structured_error"], row["structured_error"]
    assert "attempts=2" in row["structured_error"], row["structured_error"]
    # A tick that judged nothing must not read as a verdict: the row carries the
    # empty answer, so a later reader cannot mistake it for `moves: []` from a
    # steward that looked at the board and decided to move nothing.
    assert row["moves"] == [] and row["applied"] == [] and row["next_pick"] == 0, row
    assert row["events"] > 0 and row["items_shown"] > 0, row
    assert row["apply"] is False, row


def test_two_diverged_ticks_over_one_window_share_their_since_ts(board, monkeypatch):
    """Clause 2: the row records the window the failed tick was shown, so a
    re-presentation is two rows carrying one `since_ts`.

    This is the shape of the 2026-10-07 pair the item was filed on — failed at
    05:25:50, failed again at 05:41:24 one 900 s interval later over the 11 events
    the first tick had already been shown plus 6 new ones. Before #3088 neither
    tick left a row, so 'the same window was retried' was arithmetic on the run
    store rather than something the ledger could say.
    """
    rows, state, since = _two_diverged_ticks_over_one_window(board, monkeypatch)
    assert len(rows) == 2, rows
    assert [r["since_ts"] for r in rows] == [since, since], rows
    assert {r["newest_ts"] for r in rows} == {rows[0]["newest_ts"]}, rows
    assert rows[0]["newest_ts"] > since, "the row must describe a real window"
    assert state.get("last_ts", 0.0) == rows[-1]["since_ts"], (
        "the row's `since_ts` has to be the watermark the tick actually read")


def test_a_diverged_tick_records_the_attempt_and_leaves_last_ts_alone(board, monkeypatch):
    """Clause 3: the attempt lands in `board_steward.json` with a per-window count,
    and the watermark does not move — because the re-presentation IS the recovery
    mechanism. 6 of the 10 divergences of 2026-10-04→10-07 were repaired by the
    next scheduled tick reading the same window again; advancing `last_ts` here
    would trade a visible retry for a window the steward never judges.
    """
    rows, state, since = _two_diverged_ticks_over_one_window(board, monkeypatch)
    assert state["last_ts"] == since, "a failed tick moved the watermark"
    assert state["divergent_since_ts"] == since, state
    assert state["divergent_streak"] == 2, state
    assert [r["streak"] for r in rows] == [1, 2], rows
    assert state["divergent_newest_ts"] == rows[-1]["newest_ts"] == rows[0]["newest_ts"], state
    assert state["last_session"] == "s", state
    # The counter is per window, not global: a failure over a different `since_ts`
    # starts again at 1 rather than inheriting this one's streak.
    W._write_state({**state, "divergent_since_ts": since + 5.0})
    out, _ = _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK, _DIVERGED_TICK],
                                 seed=False)
    assert out["status"] == "failed", out
    after = W._read_state()
    assert after["divergent_streak"] == 1, after
    assert after["last_ts"] == since, "resetting the counter moved the watermark"


def test_a_diverged_tick_still_returns_the_record_it_returned_before(board, monkeypatch):
    """Clause 4: the record the queue stores is the one #2370 left, and publishing
    the failure does not disturb it.

    `attempts=2` is asserted inside the first 500 characters because that is the
    `runs.summary` column, and the message ends in up to 200 characters of
    captured completion — the same limit that makes the ledger's own copy capped at
    `DIVERGENCE_ERROR_MAX` characters keep the diagnosis and drop only the excerpt.
    """
    out, posted = _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK, _DIVERGED_TICK])
    assert out["status"] == "failed", out
    assert out["meta"] == {"session_id": "s"}, out
    assert out["summary"].startswith("no structured answer: "), out["summary"]
    assert _LIVE_DIVERGENCE in out["summary"], out["summary"]
    assert "diverged at 8192 tokens" in out["summary"], out["summary"]
    assert "not a budget" in out["summary"], out["summary"]
    assert "attempts=2" in out["summary"][:500], repr(out["summary"])
    assert len(posted) == 2, f"publishing changed the retry policy: {len(posted)} draws"


def test_a_ledger_or_state_write_failure_cannot_replace_the_failed_record(board, monkeypatch):
    """Clause 4's other half: `S.append_event` RAISES by design
    (`scripts/automod/state.py:176-177`), and #3088 puts a call to it on the
    failure path — so a ledger that will not take the row has to leave the queue
    with the same `failed` record rather than an exception out of `execute`.
    """
    real_append = S.append_event

    def _refuse_steward_row(entry, path=None):
        if entry.get("event") == "board_steward":
            raise RuntimeError("the ledger refuses the steward's row")
        return real_append(entry, path=path)

    def _refuse_state(d):
        raise RuntimeError("state is unwritable")

    monkeypatch.setattr(S, "append_event", _refuse_steward_row)
    monkeypatch.setattr(W, "_write_state", _refuse_state)
    out, posted = _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK, _DIVERGED_TICK])
    assert out["status"] == "failed", out
    assert "diverged at 8192 tokens" in out["summary"], out["summary"]
    assert "not a budget" in out["summary"], out["summary"]
    assert "attempts=2" in out["summary"][:500], repr(out["summary"])
    assert len(posted) == 2, posted
    assert _steward_rows() == [], "the ledger append was supposed to raise"
    assert W._read_state() == {}, "the state write was supposed to raise"


def test_the_budget_the_caps_and_the_resample_guard_are_untouched(board, monkeypatch):
    """Clause 5: what this item forbids is where #2197 and #2370 left it, and the
    publish brought no budget knob of its own.

    `test_the_finalizer_budget_stays_8192_on_both_sides_of_the_reader` and
    `test_every_node_of_the_steward_schema_is_bounded` above already pin the number
    through the reader and the caps through the finalizer's own helper, mutation
    loop included. What only this round could break, and what this node adds, is
    that neither the changed failure path nor the new helper passes a token budget
    anywhere, and that the text the ledger now stores is still the text
    `should_resample_divergence` fires on — a reworded or over-truncated copy would
    silently stop the re-sample, which is the retry this item chose to keep.
    """
    import inspect

    from app.harness import finalizer as F
    from app.harness.options import RunOptions

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml")
                         .read_text(encoding="utf-8"))
    assert cfg["harness"]["finalizer"]["max_tokens"] == 8192
    assert RunOptions(model="primary").finalizer_max_tokens == 8192
    for fn in (W.execute, W._record_divergence):
        src = inspect.getsource(fn)
        assert "max_tokens" not in src, f"{fn.__name__} now names a token budget"
        assert "finalizer_max_tokens" not in src, f"{fn.__name__} now sets a budget"

    props = W.STEWARD_SCHEMA["properties"]
    move_props = props["moves"]["items"]["properties"]
    assert props["moves"]["maxItems"] == W.MOVES_MAX == 40
    assert (move_props["note"]["maxLength"], props["next_pick_reason"]["maxLength"],
            props["summary"]["maxLength"]) == (W.NOTE_MAX, W.REASON_MAX, W.SUMMARY_MAX) \
        == (300, 300, 400)
    for name in ("tags_add", "tags_remove"):
        assert move_props[name]["maxItems"] == W.TAGS_MAX == 6, name
        assert move_props[name]["items"]["maxLength"] == W.TAG_MAX == 40, name
    assert F._schema_is_bounded(W.STEWARD_SCHEMA)
    assert not F._schema_is_bounded(_unbound(W.STEWARD_SCHEMA))

    out, _ = _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK, _DIVERGED_TICK])
    row = _steward_rows()[-1]
    assert F.should_resample_divergence(row["structured_error"], W.STEWARD_SCHEMA) is True, \
        row["structured_error"]
    assert F.should_resample_divergence(
        row["structured_error"] + " raise harness.finalizer.max_tokens.",
        W.STEWARD_SCHEMA) is False, "the budget exclusion stopped firing"
    assert F.should_resample_divergence(row["structured_error"],
                                        _unbound(W.STEWARD_SCHEMA)) is False, (
        "the unbounded-schema exclusion stopped firing")


# ── #2197 clause 5: the failure counts have committed bytes behind them ──────

#: The extract of the queue store's `runs` rows for board-steward since 2026-10-01,
#: and the marker beside it. Item-scoped paths on purpose: `backlog/data/workers.db`
#: is #1946's and #1949's witness and two other tests assert its `runs` table holds
#: ZERO rows, so a runs extract written there breaks them — the same collision #2087
#: hit (`0f8e6a3d`), which `d9c3b5b0` fixed by giving the extract its own file.
WITNESS = Path.home() / "obsidian" / "backlog" / "data" / "workers-runs-2197.db"
MARKER = WITNESS.with_suffix(".witness.md")


def test_the_committed_witness_bytes_reproduce_the_failure_counts():
    """Clause 5: the 18, the 176 and the 0 are re-derived from committed rows.

    `~/lloyd-data/workers.db` has no history — one mutable file the running queue
    writes to and a retention sweep prunes — so a count quoted from it is
    unreproducible the week it is written. `backlog/data/workers-runs-2197.db` is the
    read-only extract those figures were read from, whole rows, `summary` included,
    because the figure being witnessed is a count over `status` and `summary` and the
    `summary` has to be the bytes that produced it.

    What the run store cannot supply is the reply SHAPE — `response_json` is empty in
    all 176 rows — which is why the caps are sized off the promotion ledger instead
    (see `test_the_caps_sit_above_the_largest_reply_a_real_tick_ever_sent`).

    It also asserts the divergence message is absent from these bytes. That absence
    is the premise: with an uncapped node the finalizer could not rule the budget
    out, so all 18 landed on the budget branch and none on the one this diff turns on.
    """
    for path in (WITNESS, MARKER):
        assert path.is_file(), f"missing witness artifact {path}"
    meta = json.loads(MARKER.read_text(encoding="utf-8").split("```json")[1]
                      .split("```")[0])

    with sqlite3.connect(f"file:{WITNESS}?mode=ro", uri=True) as db:
        def one(sql):
            return db.execute(sql).fetchone()[0]

        rows = one("SELECT count(*) FROM runs")
        by_status = dict(db.execute(
            "SELECT status, count(*) FROM runs GROUP BY status"))
        truncated = one(
            "SELECT count(*) FROM runs WHERE status='failed'"
            " AND summary LIKE '%truncated at 8192%'")
        diverged = one("SELECT count(*) FROM runs WHERE summary LIKE"
                       " '%not a budget%'")
        objects = one("SELECT count(*) FROM sqlite_master")

    assert rows == 176 == meta["rows"]["runs"], rows
    assert by_status == {"failed": 20, "success": 134, "skipped": 22}, by_status
    assert truncated == 18 == meta["figures"]["failed_naming_truncated_at_8192"]
    assert diverged == 0 == meta["figures"]["failed_naming_not_a_budget"], diverged
    # The clause's re-derive command: the object count of the committed file.
    assert objects == 3 == meta["figures"]["sqlite_master_objects"], objects
    assert meta["rows"]["by_status"] == by_status, meta["rows"]["by_status"]
    assert (meta["figures"]["truncated_by_day"]
            == {"2026-10-02": 6, "2026-10-03": 5, "2026-10-04": 7}), (
        meta["figures"]["truncated_by_day"])


# ── #3089: the rolling judgement aggregate the `apply` flip is gated on ─────────
#
# #1688 made a tick's judgement recordable (`agreement()` carries `judged`, and
# `rate` is None over an empty judged set) but left it unread: the 7-day figures
# the flip depends on had to be derived by hand with a sqlite query over
# workers.db. These nodes pin the aggregate that replaces that hand-run query —
# its arithmetic over a fixture, which rows are allowed to touch it at all, and
# the window rule that decides what is in the week.

import datetime as _dt

from workers.queue import WorkQueue as _WorkQueue
from workers import dispatch_watch as DW


def _iso(days_ago: float, *, hours: int = 0) -> str:
    """An ISO instant `days_ago` days behind NOW, with an explicit +00:00 offset.

    Built the way the column is actually written (`record_run` stores whatever the
    pool hands it, and the pool hands `datetime.now(timezone.utc).isoformat()`, so
    every live `completed_at` carries `+00:00`), because the window rule this
    aggregates over is a STRING comparison and a fixture written as `Z` or as a
    naive local stamp would test a different column than the one production has.
    """
    return (_dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(days=days_ago, hours=hours)).isoformat()


def _seed_tick(queue, *, days_ago: float, agreement: dict | None = None,
               status: str = "success", extra_meta: dict | None = None):
    """Write one run row as the pool writes it, at `days_ago`, with `meta.agreement`.

    `agreement=None` means the row has no `agreement` block at all, which is what a
    `failed` or `skipped` tick looks like live (175 such rows since #1688 landed);
    pass a dict verbatim for the judged case and `{"agree": [...]}` with no
    `judged` key for a row written before #1688 existed.
    """
    meta = dict(extra_meta or {})
    if agreement is not None:
        meta["agreement"] = agreement
    queue.record_run(
        run_id=f"run_steward_{days_ago}_{status}", queue_id=None, source=W.NAME,
        status=status, started_at=_iso(days_ago, hours=1), completed_at=_iso(days_ago),
        duration_seconds=120.0, summary="seeded", meta_json=json.dumps(meta))


def _judged(n: int, agree: int, disagree: int | None = None) -> dict:
    """A per-tick `agreement()` block with `n` judged decisions, `agree` of them agreed."""
    disagree = (n - agree) if disagree is None else disagree
    return {"agree": list(range(agree)), "disagree": list(range(agree, agree + disagree)),
            "no_opinion": [], "missed": [], "machine_moves": n, "proposed": n,
            "judged": n, "rate": (agree / n) if n else None}


def test_the_aggregate_sums_judged_ticks_and_never_invents_a_rate(tmp_path):
    """Clause 3: zero-judged ticks add nothing, and an all-zero week has NO rate.

    Three ticks that judged nothing plus one that judged 2 and agreed on 1 must
    read summed judged 2 and rate 0.5. The failure this pins is the arithmetic
    #1688 removed one level down and that a naive sum re-introduces one level up:
    `agree / ticks` over four ticks, or `(1) / (1 judged tick)`, either of which
    reads 100% while the source judged twice in a week. An unmeasured week has to
    stay visibly unmeasured (`rate` None), because the flip gate is `rate >= 0.90`
    and a phantom 1.0 clears a threshold nobody earned.
    """
    q = _WorkQueue(tmp_path / "workers.db")
    for i in range(3):
        _seed_tick(q, days_ago=i + 1, agreement=_judged(0, 0))
    _seed_tick(q, days_ago=4, agreement=_judged(2, 1))

    agg = DW.judgement_aggregate(q)
    assert agg["judged"] == 2, agg
    assert agg["agree"] == 1 and agg["disagree"] == 1, agg
    assert agg["rate"] == 0.5, (
        "three unjudged ticks must not dilute or inflate the one judged tick: " + str(agg))
    assert agg["ticks_in_window"] == 4, agg
    assert agg["ticks_with_judged"] == 4, (
        "a zero-judged tick still JUDGED (it just had nothing to judge), so it "
        "belongs in the keyed count and not in the excluded one: " + str(agg))

    q2 = _WorkQueue(tmp_path / "zero.db")
    for i in range(4):
        _seed_tick(q2, days_ago=i + 1, agreement=_judged(0, 0))
    zero = DW.judgement_aggregate(q2)
    assert zero["judged"] == 0 and zero["rate"] is None, (
        "a week of ticks that judged nothing must report NO rate; any number here, "
        "and 1.0 above all, is the phantom that clears the flip gate by default: "
        + str(zero))


def test_only_rows_carrying_judged_reach_the_aggregate(tmp_path):
    """Clause 4: a row without `judged` is excluded from BOTH sides of the fraction.

    Two live populations do this. 175 rows since #1688 carry no `agreement` block
    at all (failed / skipped / interrupted ticks), and 119 older rows carry an
    `agreement` block written before #1688 added the key — 13 of those with a
    NON-EMPTY `agree` list. Summing `len(agree)` over everything while taking the
    denominator only from keyed rows puts 2 agreements over 2 judged decisions on
    a fixture that judged nothing new: the rate climbs without any new judgement,
    which is exactly the inflation that would flip `apply` on stale evidence.
    """
    q = _WorkQueue(tmp_path / "workers.db")
    _seed_tick(q, days_ago=1, agreement=_judged(2, 1))                 # keyed
    _seed_tick(q, days_ago=2, agreement=None, status="failed")          # no block
    _seed_tick(q, days_ago=3, agreement=None, status="skipped")         # no block
    _seed_tick(q, days_ago=4, agreement=None, status="interrupted")     # no block
    # A pre-#1688 row: an `agreement` block with two agreements and NO `judged`.
    _seed_tick(q, days_ago=5, agreement={"agree": [7, 8], "disagree": [],
                                         "no_opinion": [], "missed": [],
                                         "machine_moves": 2, "proposed": 2,
                                         "rate": 1.0})
    _seed_tick(q, days_ago=6, agreement=_judged(0, 0))                  # keyed, zero
    q.record_run(run_id="run_steward_unparseable", queue_id=None, source=W.NAME,
                 status="done", started_at=_iso(1, hours=1), completed_at=_iso(1),
                 duration_seconds=1.0, summary="seeded", meta_json="{not json")

    agg = DW.judgement_aggregate(q)
    assert agg["judged"] == 2 and agg["agree"] == 1 and agg["disagree"] == 1, agg
    assert agg["rate"] == 0.5, (
        "the rate moved without any new judgement — the 2 agreements on the "
        "pre-#1688 row joined the numerator while its absence from `judged` left "
        "the denominator alone: " + str(agg))
    assert agg["ticks_in_window"] == 7, agg
    assert agg["ticks_with_judged"] == 2, (
        "only the keyed rows may be counted as read; the two block-less statuses "
        "and the pre-#1688 row must show up as read-but-not-keyed: " + str(agg))


def test_the_window_is_a_string_comparison_on_completed_at(tmp_path):
    """Clause 5: 6 days old is inside a 7-day window, 8 days old is outside.

    The column holds an ISO-8601 string with a `+00:00` offset, and
    `list_runs_joined` compares it lexicographically against a computed ISO cutoff
    — which works precisely because both sides are UTC with the same offset. The
    trap is the "obvious" alternative: `completed_at >= <epoch float>` selects ZERO
    rows (SQLite compares text against a numeric as different types, text always
    losing), and an empty aggregate reads as a source with no traffic rather than
    as a broken query. So the boundary tick is the test: a row that is 8 days old
    has to be excluded and a 6-day row included, with the counts saying which
    happened.
    """
    q = _WorkQueue(tmp_path / "workers.db")
    _seed_tick(q, days_ago=1, agreement=_judged(3, 3))
    _seed_tick(q, days_ago=6, agreement=_judged(2, 2))     # inside
    _seed_tick(q, days_ago=8, agreement=_judged(50, 0))    # outside, and disagreeing
    _seed_tick(q, days_ago=30, agreement=_judged(9, 9))    # far outside

    agg = DW.judgement_aggregate(q)
    assert agg["ticks_in_window"] == 2, agg
    assert agg["judged"] == 5 and agg["agree"] == 5, (
        "the window boundary moved: the 8-day row's 50 disagreements are the tell "
        "that the cutoff stopped matching, and a 0-row read would be the "
        "epoch-comparison bug announcing itself: " + str(agg))
    assert agg["rate"] == 1.0 and agg["window_days"] == 7, agg
    assert agg["window_since"].startswith(
        (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=7)).isoformat()[:13]), (
        "the window is not named on the figure, so a reader cannot tell which "
        "week this is: " + str(agg))
    # 6 days inside and 8 outside is a 1-day margin at the boundary, which is wide
    # enough that this node does not care what time of day it runs at; the cutoff
    # itself is asserted, not inferred from the count.
    assert agg["window_since"] <= _iso(6), (
        "the cutoff is later than a tick that is 6 days old, so a 6-day row would "
        "be cut out of a 7-day window: " + agg["window_since"])
    assert agg["window_since"] > _iso(8), (
        "the cutoff is at or before a tick that is 8 days old, so the window is "
        "wider than the 7 days it claims: " + agg["window_since"])


def test_the_config_states_both_flip_thresholds_and_when_they_were_measured():
    """Clause 5 (second half), re-pinned by #3329: the gate lives in the file that
    would flip it, and it states the numbers that were RULED.

    `config.yaml`'s board-steward comment has always said the flip waits for a
    summed `judged` count to clear "a minimum" with a rate that "holds over it" —
    a gate with no numbers in it, so no reader can tell cleared from unmet without
    a query. #3089 put numbers in; owed-check ruled them on 2026-10-25, KEEPING
    `rate` >= 0.90 and the 7-day window and REPLACING the summed-judged floor 100
    with 50. The node therefore pins four things a re-worded comment can fail one
    at a time:

     - the ruled floor, and the ABSENCE of the superseded one — a reader obeys the
       number written beside the flag, so a retired 100 left in the prose is still
       a live instruction to the next round;
     - the half of the gate that did not move (`>= 0.90`, the 7-day window, the date
       the figures were read) plus WHICH half today's reading leaves unmet: judged
       86 clears a 50 floor, so the old "NEITHER threshold is met" went false the
       moment the floor dropped;
     - that the numbers are settled, on the ruling's own reason — the floor sits
       ~20% above a measured ~80/week plateau (weekly judged 163/82/78/82 for the
       four weeks ending 2026-10-25) — and NOT on "100 was unreachable", which the
       ruling's own 163-judged week refutes;
     - that `judgement.rate` is a pre-flip-only measure, with its mechanism named:
       once `apply: true` has written a wrong status the item sits at the steward's
       value and is never re-proposed, so the standing divergence lands in `missed`
       and in the `judged` denominator only and never arrives as a fresh
       `disagree`.

    And `apply` is still read as YAML and asserted false: stating a threshold beside
    a flipped flag would be a flip dressed as documentation.
    """
    text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    lines = text.splitlines()
    keys = [i for i, ln in enumerate(lines) if ln.strip() == "board-steward:"]
    assert len(keys) == 1, (
        f"the source block this gate is written into must be findable by its own key, "
        f"found {len(keys)}")
    # The gate lives in the comment ABOVE the key, so the block is that comment run
    # and the key's own lines — not a fixed window of text, which would drift into
    # the neighbouring source's comment and let its prose satisfy this node.
    start = keys[0]
    while start > 0 and (lines[start - 1].strip().startswith("#")
                         or not lines[start - 1].strip()):
        start -= 1
    end = keys[0] + 1
    while end < len(lines) and (lines[end].startswith("      ")
                                or lines[end].strip().startswith("#")):
        end += 1
    block = "\n".join(lines[start:end])
    assert "apply: false" in block, (
        "the slice did not capture the flag the comment gates: " + block[-200:])
    # Comment lines wrap at ~76 columns, so every phrase below is pinned against
    # the comment's PROSE — comment markers stripped, whitespace collapsed. The
    # markers have to go, not just the newlines: leaving them in re-joins the text
    # as "the agree-rate is the # half that does not", so a pinned phrase that
    # happens to wrap would fail for where the author broke the line. The wording
    # is the contract; the line breaks are not.
    prose = " ".join(ln.lstrip().lstrip("#").rstrip() for ln in block.splitlines())
    flat = " ".join(prose.split())

    # The ruled floor is the stated floor, and the superseded one is gone.
    assert "judged` >= 50" in block, (
        "the judged-count threshold in the board-steward block is not the ruled "
        "number — owed-check set the floor to 50 on 2026-10-25: " + block[:200])
    assert "judged` >= 100" not in block, (
        "the block still states the floor #3089 proposed and the 2026-10-25 ruling "
        "replaced. Beside a flag that flips on what this comment says, a retired "
        "number left in prose is an instruction, not history")
    # The half of the gate the ruling did NOT move. The window is pinned as the
    # phrase attached to the two numbers, not as "7-day" or "7 days" somewhere in
    # the block: the block also says the rate "read 0.7317 over the trailing 7
    # days", and that incidental phrase satisfied the loose check while the
    # thresholds' own window silently became 30.
    assert ">= 0.90" in block, "the agree-rate threshold is not stated as a number"
    assert "both over the rolling 7-day window" in flat, (
        "the thresholds are meaningless without the window they are measured over, "
        "and it has to be the window attached to THEM")
    # And the prose window is the code window: `judgement_aggregate` measures what
    # this comment promises, so a drift between the two is the seam failing, not a
    # wording preference.
    from workers import dispatch_watch as DW
    assert DW.JUDGEMENT_WINDOW_DAYS == 7, (
        "`JUDGEMENT_WINDOW_DAYS` is no longer 7, so the window this comment states "
        "and the one `judgement_aggregate` measures have come apart")
    # The reading needs a timestamp, not merely some date in the block: the
    # comment also carries the ruling's date and the plateau weeks' end dates, and
    # any one of them satisfied a bare `2026-\d\d-\d\d`. Only the measurement is
    # dated to the minute.
    assert re.search(r"2026-\d\d-\d\dT\d\d:\d\dZ", block), (
        "the figures need the moment they were measured, or the next reader cannot "
        "tell a stale reading from a current one")
    assert "not cleared" in block.lower(), (
        "the comment must say what today's figures read, so an unmet gate is not "
        "mistaken for an absent one")
    # …and which of the two halves that leaves unmet. Lowering the floor made the
    # #3089 wording false rather than merely stale: judged clears, rate does not.
    assert "neither" not in flat.lower(), (
        "the comment still says NEITHER threshold is met. A 50 floor made that "
        "false — the live reading clears it — and a comment naming the wrong unmet "
        "half is how the wrong bar gets lowered next time")
    assert "the agree-rate is the half that does not" in flat, (
        "naming the gate NOT cleared is not enough; the block has to say which half "
        "of it today's figures leave unmet")
    # Settled, with the ruling's reason and not a refuted one. An assert on "owed"
    # proves nothing here because "owed-check" contains it, so the settled wording
    # itself is what has to be present.
    assert "settled by the owed-check ruling of 2026-10-25" in flat, (
        "the two numbers have to read as ruled, not as proposals still on trial")
    assert "still owed" not in flat.lower(), (
        "the block still calls the numbers a ruling owed to owed-check, which #3089 "
        "settled on 2026-10-25")
    assert "80/week plateau" in flat, (
        "the reason 50 is the floor is the measured plateau under the old one; a "
        "number with no reason beside it gets re-litigated by the next reader")
    assert "163/82/78/82" in flat, (
        "the four weekly judged counts behind that plateau are the evidence for the "
        "floor, so they belong with the number they justify")
    assert "unreachable" not in flat.lower(), (
        "the block repeats a claim the ruling's own first week (judged 163, over "
        "the old floor) refutes — the reason is a plateau under the floor, not an "
        "unreachable floor")
    assert "rate bar was deliberately not lowered" in flat.lower(), (
        "keeping 0.90 while moving the floor was a decision, and the comment has to "
        "say it was one")
    assert "0.7317" in flat and "0.80" in flat, (
        "the rate readings the bar was deliberately NOT fitted to are the whole "
        "point of that sentence")
    # The rate is only evidence while the steward is still dry-running, and the
    # file that would end the dry-run has to carry the reason.
    assert "pre-flip-only" in flat.lower(), (
        "the block has to say that judgement.rate stops meaning agreement at the flip")
    assert "never as a fresh `disagree`" in flat, (
        "the pre-flip-only claim without its mechanism reads as an opinion: an "
        "applied wrong status is never re-proposed, so no fresh `disagree` is ever "
        "scored on it")
    assert "`missed` and in the `judged` denominator only" in flat, (
        "and where the divergence does go instead — `missed`, plus the denominator, "
        "which is what pulls the rate down while nothing new disagrees")
    # The flag itself is read as YAML, not as prose: the comment says
    # "Flip to `apply: true` only after …" and a substring test would trip on its
    # own instruction. The value is the only thing that answers "is it flipped?".
    import yaml
    cfg = yaml.safe_load(text)
    assert cfg["workers"]["sources"]["board-steward"]["apply"] is False, (
        "this item exposes the gate; it does not open it")


#: The three ways the tree actually writes the retired floor. #3089's own absence
#: grep was `>= 100`, which saw one of them: `config.yaml` and the architecture doc
#: put the backtick AFTER `judged` (one of them with a Unicode `≥`), and
#: `workers/dispatch_watch.py`'s docstring puts it BEFORE and wraps the number onto
#: the next comment line. Collapsing whitespace and making the backtick optional
#: covers all three; the trailing guard stops an unrelated `judged 1000`-shaped
#: count from reading as the floor.
RETIRED_FLOOR = re.compile(r"judged[`'\"\u201d\u2019]{0,2}\s*(?:>=|≥)\s*100(?![\d.])")
#: The sentence claiming the two numbers are still on trial, anchored on the flip's
#: own figures. The anchor is load-bearing: "is a ruling owed" is ordinary English
#: in this tree (#3109's guardian raise, #2584's loopback allowance, #2644's
#: restored verdicts), and those are decisions this item does not own.
RULING_STILL_OWED = re.compile(
    r"(?:judged|judgement|0\.90)[^.]{0,140}?\b(?:is|remains|stays)\s+a\s+ruling\b"
    r"[^.]{0,90}?(?:owed|owes)")
#: Suffixes that carry prose, so the scan reaches `.md` docs, `.yaml` config and
#: `.json` fixtures without trying to decode an image.
TEXT_SUFFIXES = {".py", ".md", ".rst", ".yaml", ".yml", ".txt", ".json",
                 ".ts", ".tsx", ".toml", ".cfg", ".ini", ".sh"}


def test_no_shipped_file_still_states_the_old_flip_floor_or_an_owed_ruling():
    """#3329 clause 5: after this round the only 100 left is the one inside the
    assertion that it is gone.

    The item's check was ``git grep ">= 100"``, and that pattern is what let three
    sites outlive #3089: the `judgement_aggregate` docstring keeps the backtick on
    the other side of `judged`, `architecture/workers-jobs.md` writes the
    comparison as `≥` in two places, and one of those is a dated drift record no
    bare grep can tell from a live claim. So this node collapses each file's
    whitespace before matching, covers the ASCII and Unicode spellings, and
    enumerates with `git ls-files` rather than a list of files it happens to
    remember — an allowlist cannot close a property that has to hold over every
    file, which is the class rule this loop wrote on 2026-09-20 after exactly this
    failure. `tests/` is the one excluded tree, because the node above has to
    spell the retired literal in order to assert its absence, and that is the
    single place a 100 belongs.
    """
    # The instrument proves itself first: a pattern that matches nothing would
    # green this node against any corpus, and that is the failure this class of
    # guard has to defend against before it checks anybody else.
    assert RETIRED_FLOOR.search("summed `judged` >= 100 AND agree `rate`")
    assert RETIRED_FLOOR.search("whether `judged\n    >= 100` and `rate >= 0.90`")
    assert RETIRED_FLOOR.search("both numbers — summed `judged` ≥ 100 and agree")
    assert RETIRED_FLOOR.search("gate is stated since #3089 (judged ≥ 100, rate ≥ 0.90)")
    assert not RETIRED_FLOOR.search("summed `judged` >= 50 AND agree `rate` >= 0.90")
    assert not RETIRED_FLOOR.search("`judged` totals 1000 rows")
    assert RULING_STILL_OWED.search("Whether 100 and 0.90 are the right two numbers "
                                    "is a ruling still owed to owed-check on #3089")
    assert RULING_STILL_OWED.search("whether `judged >= 100` and `rate >= 0.90` are "
                                    "the right numbers is a ruling #1688 owes")
    assert not RULING_STILL_OWED.search("Whether that stays report-only is a ruling "
                                        "no round makes (owed on #3109).")

    repo = Path(__file__).resolve().parents[1]
    listed = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"],
                            capture_output=True, check=True).stdout.split(b"\0")
    scanned, offenders = 0, []
    for raw in listed:
        if not raw:
            continue
        rel = Path(raw.decode())
        if rel.parts[0] == "tests" or rel.suffix not in TEXT_SUFFIXES:
            continue
        try:
            body = (repo / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scanned += 1
        flat = " ".join(body.split())
        for rx, label in ((RETIRED_FLOOR, "states the retired 100 floor"),
                          (RULING_STILL_OWED, "calls the two numbers a ruling still owed")):
            hit = rx.search(flat)
            if hit:
                offenders.append(f"{rel}: {label}: …"
                                 f"{flat[max(0, hit.start() - 60):hit.end() + 50]}…")
    # A scan that read nothing is a green instrument that checked nothing, so the
    # corpus size is asserted alongside the emptiness of the offender list.
    assert scanned > 1000, (
        f"the scan read {scanned} tracked text files, which is not this corpus "
        "(~2,200 outside tests/) — the enumeration, not the match, is broken")
    assert not offenders, ("shipped prose still states the superseded gate:\n"
                           + "\n".join(offenders))


def test_the_judgement_reader_lives_in_the_watcher_and_names_the_writers_keys():
    """The two costs of hosting the reader away from the writer, both paid back.

    `judgement_aggregate` sits in `workers/dispatch_watch.py`, not beside the
    `agreement()` that emits the block, because importing a module containing the
    board-steward WRITER (`apply_moves`) into the backend's import closure would put
    a live backlog writer inside a tool agent's reach as a bare import — the edge
    `tests/fixtures/worker_co_presence_baseline.json` exists to keep out. Two things
    follow, and both are pinned here rather than left to the reader's good intentions:
     - the source name the reader selects rows for is a literal in the watcher, so it
       is asserted against the source module's own `NAME`; a drifted string selects 0
       rows and the gate reads an empty week, not a broken query;
     - the keys the reader counts are the keys the writer emits, checked by calling
       `agreement()` and reading what it returns, so renaming a key on either side
       fails a node instead of silently dropping a term out of the flip gate.
    """
    assert DW.BOARD_STEWARD_SOURCE == W.NAME, (
        "the reader selects a source the writer is not named "
        f"{DW.BOARD_STEWARD_SOURCE!r} vs {W.NAME!r}: the aggregate would read 0 rows "
        "and report a week in which nothing was judged")
    emitted = W.agreement(moves=[{"item_id": 1, "status": "up_next"},
                                  {"item_id": 2, "status": "done"}],
                           expected={1: ("up_next", "triaged"),
                                     2: ("up_next", "triaged")},
                           current={1: "up_next", 2: "up_next"})
    for key in ("judged", "agree", "disagree"):
        assert key in emitted, (
            f"agreement() no longer emits {key!r}, which is the term the aggregate "
            "sums — the reader is in another module and cannot fail when this one "
            "renames a key")
    src = (Path(__file__).resolve().parents[1] / "app" / "routers"
           / "workers.py").read_text(encoding="utf-8")
    assert "dispatch_watch.judgement_aggregate" in src, (
        "the health route stopped reading the aggregate from the watcher, so the "
        "closure reason for the placement is gone and it should be revisited")
    assert "from workers.sources.board_steward import" not in src, (
        "the backend route imports the board-steward source module again, which is "
        "the co-presence the placement exists to avoid — the guard in "
        "tests/test_worker_capability_allowlist.py fails with it")


# ── #3322: a quiet tick may not rewind the watermark ─────────────────────────
#
# `newest` is a max over `events`, and `events` is the `for_items`-widened list:
# `execute` calls `events_since(..., for_items={i.id for i in shown})` at
# `workers/sources/board_steward.py:641-643`, and `events_since` appends up to
# `per_item` rows for each shown item "whatever their age" (:383-387). So a tick
# whose window holds NOTHING newer than `last_ts` can still be handed a shown
# item's history, the plain max over that list lands BELOW `since_ts`, and the
# single `last_ts` write (:702) moves the watermark backwards.
#
# `default=since_ts` is not a clamp: it fires only on an empty list, and the
# quiet tick's list is not empty — that is precisely the case the skip at :649
# does not catch, because it fires on `not events` and the widened list is full.
#
# Live on 2026-10-25, as #3322 records it: the tick at 16:41:54Z left `last_ts`
# at 03:49:23Z where it had been 16:36:32Z (the item measures that rewind as
# 12 h 49 m and counts 172 already-consumed events re-exposed), and the same
# re-exposure shape appears three times that day — 132 events at 13:03:38, 180 at
# 16:38:19, 181 at 16:58:13, against a same-day norm of 9-41.


def _quiet_window_with_one_item_of_history(board, *, hours_old: float = 12.0):
    """Seed `board` so its ONLY events are one shown item's history, all older
    than the watermark. Returns `(since_ts, history_ts)`.

    The three rows are the kinds the live rewind rode — a `status_moved`, an
    `item_landed` and a `backlog_implement`, all on item 7 — stamped
    `hours_old` BELOW the watermark. Item 7 is `up_next`, which does two jobs: it
    puts the item in `board_view`, and `board_view` membership is the only reason
    `events_since` is asked for its history at all; and the same status is what
    keeps the `not events` skip at :649 from firing on an empty window.

    The watermark is an hour back rather than `0.0` of a missing state file, so
    `since_ts` is a number the assertion can be exact about.
    """
    since = time.time() - 3600.0
    history_ts = since - hours_old * 3600.0
    _write(board, 7, "up_next")
    _ev(event="status_moved", item_id=7, ts=history_ts, to="up_next", by="owed-check")
    _ev(event="item_landed", item_id=7, ts=history_ts + 1.0, commit="deadbeef")
    _ev(event="backlog_implement", item_id=7, ts=history_ts + 2.0, phase="finished",
        round_id="SM_QUIET")
    W._write_state({"last_ts": since, "last_run": since, "last_session": "seeded"})
    return since, history_ts


def _drive_a_success_tick(board, monkeypatch) -> str:
    """One board-steward tick that answers, over whatever `board` holds now.

    Only the turn is canned; the ledger, `events_since`, `board_view`,
    `parse_steward`, the state write and the ledger row are production. Returns
    the prompt the steward was shown, so a caller can see what the tick really
    read rather than what it was said to read.
    """
    from workers.sources import _common as C
    seen: dict[str, str] = {}

    async def turn(prompt, **kw):
        seen["prompt"] = prompt
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": [], "next_pick": 0, "next_pick_reason": "",
                               "summary": "net 0 today, nothing to move this pass"}}

    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out["status"] == "success", out
    return seen["prompt"]


def test_a_quiet_tick_with_history_on_a_shown_item_leaves_last_ts_where_it_was(board,
                                                                               monkeypatch):
    """Clause 1: the watermark can never move backwards.

    The window holds no event newer than `last_ts`, item 7's history is older than
    it, and the tick still runs and still answers — so the write happens, and the
    clamp is the only thing that can keep `last_ts` at `since_ts`.
    """
    since, history_ts = _quiet_window_with_one_item_of_history(board)
    # The precondition, from the real `events_since`: the window is empty, yet the
    # tick is handed three rows. A test that only asserted the resulting
    # `last_ts` would also pass on a `events_since` that returned nothing.
    widened = W.events_since(S.LEDGER_PATH, since_ts=since, for_items={7})
    assert [d["event"] for d in widened] == ["status_moved", "item_landed",
                                             "backlog_implement"], widened
    assert max(d["ts"] for d in widened) == history_ts + 2.0 < since, widened
    assert W.events_since(S.LEDGER_PATH, since_ts=since, for_items=frozenset()) == [], (
        "the control: no shown item, no events — so the three rows above are the "
        "per-item history and not the window")

    prompt = _drive_a_success_tick(board, monkeypatch)

    state = W._read_state()
    assert state["last_ts"] == since, (
        f"a quiet tick rewound the watermark by {since - state['last_ts']:.0f} s, "
        f"from {since} to {state['last_ts']}")
    assert state["last_session"] == "s", state
    row = _steward_rows()[-1]
    assert row["events"] == 3, f"the tick recorded {row['events']} events: {row}"
    assert "event=status_moved item_id=7" in prompt, prompt[-400:]


def test_a_tick_shown_one_newer_event_still_advances_last_ts_to_its_newest_ts(board,
                                                                             monkeypatch):
    """Clause 2: the clamp is a floor, not a freeze.

    Same seeded history twelve hours under the watermark, plus two events inside
    the window. `last_ts` has to land on the newest event the tick was SHOWN
    (`since + 200`), not on `since_ts`, not on the last row of a list sorted
    oldest-first, and not on the newest row of the window if history were newer.
    """
    since, history_ts = _quiet_window_with_one_item_of_history(board)
    newest_ts = since + 200.0
    _ev(event="item_landed", item_id=7, ts=since + 100.0, commit="cafebdd0")
    _ev(event="backlog_implement", item_id=7, ts=newest_ts, phase="started",
        round_id="SM_LIVE")

    _drive_a_success_tick(board, monkeypatch)

    state = W._read_state()
    assert state["last_ts"] == newest_ts, (
        f"last_ts={state['last_ts']} but the newest event shown was {newest_ts} "
        f"(watermark before the tick: {since}, oldest history: {history_ts})")
    assert state["last_ts"] > since, "a tick with new events stalled the watermark"


def test_a_diverged_quiet_tick_writes_no_last_ts_and_its_record_is_unchanged(board,
                                                                             monkeypatch):
    """Clause 3: #3088's divergence path is untouched by the clamp.

    The same quiet window, drawn twice into the captured divergence through the
    real finalizer. Two things must stay exactly as they are today: the watermark
    is not written at all on the failure path, and the record it publishes is
    UNCLAMPED — `divergent_newest_ts` and the ledger row's `newest_ts` are the
    plain max over the shown events, twelve hours BELOW the watermark. Clamping
    the value passed at :681 would lift both of them to `since_ts`, which is the
    change #3322's triage explicitly dropped from its own contract.
    """
    since, history_ts = _quiet_window_with_one_item_of_history(board)
    out, posted = _drive_steward_tick(board, monkeypatch,
                                      [_DIVERGED_TICK, _DIVERGED_TICK], seed=False)
    assert out["status"] == "failed", out
    assert len(posted) == 2, f"{len(posted)} finalizer draws: the re-sample moved"

    state = W._read_state()
    assert state["last_ts"] == since, "a failed tick moved the watermark"
    assert state["divergent_since_ts"] == since, state
    assert state["divergent_streak"] == 1, state
    assert state["divergent_newest_ts"] == history_ts + 2.0, state

    rows = _steward_rows()
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["diverged"] is True and row["streak"] == 1, row
    assert row["since_ts"] == since, row
    assert row["newest_ts"] == history_ts + 2.0 == state["divergent_newest_ts"], row
    assert row["newest_ts"] < state["last_ts"], (
        "the divergence record is clamped: its newest_ts is no longer below the "
        "watermark it was recorded over")
    assert row["events"] == 3, row
    assert row["summary"] == (f"tick diverged, no structured answer "
                             f"({row['events']} events shown)"), row
    assert out["summary"].startswith("no structured answer: "), out["summary"]


def test_the_clamp_left_every_input_of_the_stewards_tick_alone(board, monkeypatch):
    """Clause 4: the fix is the clamp at the write and nothing else.

    The discriminating half is the first assertion. Dropping the old rows inside
    `events_since` would ALSO stop the rewind and would satisfy clause 1 — and it
    would take from the steward exactly what the widening exists to give it: "an
    item the steward is shown with none of its history is one it can only guess
    about" (`workers/sources/board_steward.py:361-362`). So the quiet tick must
    still SEE the twelve-hour-old rows while the watermark stays put. The rest
    pins the numbers the item names as out of scope: the 300-event and 80-item
    caps, `events_since`'s own `per_item` default, and the schema whose caps
    `parse_steward` shares (#2197).
    """
    import inspect

    since, history_ts = _quiet_window_with_one_item_of_history(board)
    prompt = _drive_a_success_tick(board, monkeypatch)

    assert "event=status_moved item_id=7" in prompt, (
        "the quiet tick no longer sees the shown item's history: the fix moved out "
        "of the last_ts write and into events_since")
    assert 'events="3"' in prompt, prompt[:400]

    assert (W.DEFAULT_MAX_EVENTS, W.DEFAULT_MAX_ITEMS) == (300, 80), (
        "the event and item caps moved with the clamp")
    assert inspect.signature(W.events_since).parameters["per_item"].default == 6, (
        "events_since's per-item history width moved with the clamp")
    assert W.STEWARD_SCHEMA["properties"]["moves"]["maxItems"] == W.MOVES_MAX == 40, (
        "the schema and the parser's move cap are no longer one number, or the "
        "number is no longer 40 (#2197)")
    assert W.STEWARD_SCHEMA["properties"]["summary"]["maxLength"] == W.SUMMARY_MAX == 400
    assert W.STEWARD_STATUSES == ("draft", "up_next", "in_progress")


# ── #3321: a tick killed on the wire is recorded, and still dies in the pool ──
#
# `execute` catches `DrainActive` and `TurnTimeout` and nothing else, and
# `workers/sources/_common.py` opens its stream with `httpx.AsyncClient` /
# `client.stream` and no `except httpx.*` in the module, so a connect failure
# leaves `execute` entirely and the promotion ledger never learns the window was
# tried. Triage measured it on 2026-10-25: 13 board-steward runs in the trailing
# 30 days died that way (11 `ConnectTimeout` + 2 `ConnectError`; 12 = 11 + 1 since
# 2026-10-05), and ZERO `board_steward` rows fall inside any of those 13
# [started_at, completed_at] windows — while `_record_divergence` (#3088) does
# fire on the parse path, with 4 `diverged: True` rows on 10-20, 10-23 and
# 10-25. So the ledger can show a tick the finalizer never answered and cannot
# show a tick the network never delivered.
#
# Only `run_prompt_in_session` is replaced below. The events, the board view, the
# ledger and `board_steward.json` are the real ones, because the thing under test
# is what a dead tick LEAVES BEHIND, not what it was shown.

import httpx  # noqa: E402  (the class the live deaths are all instances of)


def _dying_engine(board, monkeypatch, exc_cls, *, message, seed=True):
    """`run_prompt_in_session` replaced by one that raises `exc_cls` on the wire.

    `execute` imports it from `workers.sources._common` inside the function, so
    patching the module attribute is the seam production uses, not a copy of it.
    """
    from workers.sources import _common as C

    async def turn(prompt, **kw):
        raise exc_cls(message)

    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    if seed:
        _ten_item_board(board)


def _seed_window_an_hour_back():
    """A watermark an hour back, so `since_ts` is a window and not `0.0`."""
    since = time.time() - 3600.0
    W._write_state({"last_ts": since, "last_run": since, "last_session": "seeded"})
    return since


def test_a_tick_killed_on_the_wire_appends_one_row_naming_the_error_class(board, monkeypatch):
    """Clause 1: the death of a tick on the wire is one `board_steward` row, and
    it is on disk before the exception leaves `execute`.

    `since_ts` is the join key that makes a retried window readable, and the
    `events` count is cross-checked against the same window driven to success, so
    the number in a death row is the number a living tick would have been shown —
    not a count of nothing.
    """
    since = _seed_window_an_hour_back()
    _dying_engine(board, monkeypatch, httpx.ConnectTimeout, message="connect timed out")
    with pytest.raises(httpx.ConnectTimeout):
        asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))

    rows = _steward_rows()
    assert len(rows) == 1, f"one dead tick, {len(rows)} board_steward rows: {rows}"
    row = rows[0]
    assert row["error_class"] == "ConnectTimeout", row
    assert "ConnectTimeout" in row["summary"], row
    assert row["since_ts"] == since, row
    assert row["events"] > 0 and row["items_shown"] > 0, row
    assert row["apply"] is False, row
    # A death judged nothing, so it must not read as a verdict.
    assert row["moves"] == [] and row["applied"] == [] and row["next_pick"] == 0, row
    # The same window, this time answered: the counts must agree, or the death
    # row is describing a window nobody could re-derive.
    out, _ = _drive_steward_tick(
        board, monkeypatch,
        [_Resp(json.dumps({"moves": [], "next_pick": 0, "next_pick_reason": "",
                           "summary": "net 0 today"}), "stop", 40)], seed=False)
    assert out["status"] == "success", out
    rows = _steward_rows()
    assert len(rows) == 2, rows
    assert rows[0]["events"] == rows[1]["events"], (
        f"the death row counted {rows[0]['events']} events, the living tick over "
        f"the same window was shown {rows[1]['events']}")
    assert rows[0]["items_shown"] == rows[1]["items_shown"], rows


def test_a_tick_killed_on_the_wire_still_raises_it_to_the_pool_unchanged(board, monkeypatch):
    """Clause 2: recording must not convert the death into a returned record.

    This is the correction triage put on the item's own proposal. The pool's
    generic arm (`workers/pool.py:2043`, `error_msg = f"{type(e).__name__}: {e}"`)
    is what calls `mark_failed` and stamps `meta.failure_kind`; a returned dict
    goes to `mark_completed` with no retry. Live proof the retry is load-bearing:
    11 of the 13 failed queue rows carry `attempts=2` and their SECOND run
    succeeded — the pool recovers these windows inside about a minute, and a
    swallowed exception would trade that for the next 900 s tick.

    The stamping is checked through the real classifier the pool uses, not a copy
    of its rule: `_failure_kind_of` reads `type(exc).__name__` against
    `_INFRA_EXC_NAMES`, which holds both transport names.
    """
    from app.autonomy import _failure_kind_of
    _seed_window_an_hour_back()
    _dying_engine(board, monkeypatch, httpx.ConnectError, message="connection refused")
    with pytest.raises(httpx.ConnectError):
        asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    # The class survives the record, so the pool's arm still sees the transport
    # error and not a recorder's own failure.
    assert _failure_kind_of(httpx.ConnectError("connection refused")) == "infra"
    assert _failure_kind_of(httpx.ConnectTimeout("connect timed out")) == "infra"


def test_two_wire_deaths_over_one_window_count_1_then_2_without_moving_last_ts(
        board, monkeypatch):
    """Clause 3: `last_ts` stays exactly where the tick read it, and the streak is
    per window, so two deaths over one window read 1 then 2 and a death over a
    different `since_ts` restarts at 1.

    The watermark must not move: the immediate retry is what recovers these
    deaths, and advancing the window would silently drop the events the dead tick
    was never shown. #3322 is the same rule on the success path from the other
    side — a write may never move `last_ts` backwards.
    """
    since = _seed_window_an_hour_back()
    for expect_streak in (1, 2):
        _dying_engine(board, monkeypatch, httpx.ConnectTimeout,
                      message="connect timed out", seed=(expect_streak == 1))
        with pytest.raises(httpx.ConnectTimeout):
            asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
        state = W._read_state()
        assert state["last_ts"] == since, (
            f"death {expect_streak} moved the watermark from {since} to "
            f"{state['last_ts']}")
        assert state["death_streak"] == expect_streak, state
        assert state["death_since_ts"] == since, state
    rows = _steward_rows()
    assert [r["streak"] for r in rows] == [1, 2], rows
    assert [r["since_ts"] for r in rows] == [since, since], rows
    # The counter is keyed on the window, not global: a death over another
    # `since_ts` starts again at 1 rather than inheriting this one's streak.
    W._write_state({**W._read_state(), "death_since_ts": since + 5.0})
    _dying_engine(board, monkeypatch, httpx.ConnectTimeout, message="t", seed=False)
    with pytest.raises(httpx.ConnectTimeout):
        asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    after = W._read_state()
    assert after["death_streak"] == 1, after
    assert after["last_ts"] == since, "resetting the counter moved the watermark"


@pytest.mark.parametrize("surface", ["state", "ledger"])
def test_a_broken_record_cannot_replace_the_error_the_pool_receives(board, monkeypatch,
                                                                    surface):
    """Clause 4: if the record itself fails, the class the pool catches is still
    the transport error.

    Both write surfaces are driven, because a guard on only one of two is not a
    guard: `_write_state` can fail on a full disk and `S.append_event` RAISES by
    design (`scripts/automod/state.py`, unlike autoresearch's best-effort append).
    On the ledger surface the row is legitimately absent — that is the record
    failing — but the death still travels.

    The window and the ten-item board are seeded BEFORE the surface is poisoned:
    `_ten_item_board` writes its events through `S.append_event`, so poisoning
    first would break the setup instead of the path under test.
    """
    since = _seed_window_an_hour_back()
    _dying_engine(board, monkeypatch, httpx.ConnectTimeout, message="connect timed out")
    if surface == "state":
        def boom(*a, **kw):
            raise OSError("state disk full")
        monkeypatch.setattr(W, "_write_state", boom)
    else:
        def boom(*a, **kw):
            raise RuntimeError("ledger locked")
        monkeypatch.setattr(S, "append_event", boom)
    with pytest.raises(httpx.ConnectTimeout):
        asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    if surface == "ledger":
        assert _steward_rows() == [], "a refused append cannot manufacture a row"
    else:
        assert W._read_state()["last_ts"] == since, (
            "the state write failed, so the file must still hold the tick's own read")


def test_a_wire_death_row_is_not_a_divergence_row_and_the_divergence_row_is_unchanged(
        board, monkeypatch):
    """Clause 5: the two failure kinds are distinguishable from the row alone, and
    the #3088 row's keys and summary are the ones it writes today.

    A transport death stamped `diverged: True` with the `tick diverged, no
    structured answer` summary would publish a network failure as a statement
    about the finalizer — which never ran. `ts` and `created_at` are excluded from
    the key sets because `S.append_event` stamps them on every row.
    """
    appended = {"ts", "created_at"}
    since = _seed_window_an_hour_back()
    _dying_engine(board, monkeypatch, httpx.ConnectTimeout, message="connect timed out")
    with pytest.raises(httpx.ConnectTimeout):
        asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    dead = _steward_rows()[0]
    assert "diverged" not in dead, dead
    assert dead["died"] is True, dead
    assert "no structured answer" not in dead["summary"], dead["summary"]

    # The same window, this time reaching the finalizer and diverging there.
    _drive_steward_tick(board, monkeypatch, [_DIVERGED_TICK, _DIVERGED_TICK], seed=False)
    rows = _steward_rows()
    assert len(rows) == 2, rows
    div = rows[1]
    assert div["diverged"] is True and "died" not in div, div
    assert div["summary"] == (f"tick diverged, no structured answer "
                             f"({div['events']} events shown)"), div["summary"]
    assert set(div) - appended == {
        "event", "apply", "diverged", "streak", "structured_error", "since_ts",
        "newest_ts", "events", "items_shown", "moves", "next_pick", "next_pick_reason",
        "applied", "session_id", "summary"}, sorted(set(div) - appended)
    assert set(dead) - appended == {
        "event", "apply", "died", "error_class", "error", "streak", "since_ts",
        "events", "items_shown", "moves", "next_pick", "next_pick_reason",
        "applied", "session_id", "summary"}, sorted(set(dead) - appended)
    # Two counters, one per failure kind: a death does not lengthen a divergence
    # streak, and does not shorten or replace it.
    state = W._read_state()
    assert (state["death_streak"], state["divergent_streak"]) == (1, 1), state
    assert state["last_ts"] == since, state


def test_the_drain_skip_still_writes_neither_state_nor_a_row(board, monkeypatch):
    """Clause 5's other half, and the path this diff must not touch: a 503 drain
    is a `skipped` tick that already promised to leave no trace — the backend is
    busy, not broken, and the window is re-read by the next tick anyway.
    """
    from workers.sources._common import DrainActive
    _seed_window_an_hour_back()
    before = W._read_state()
    _dying_engine(board, monkeypatch, DrainActive, message="backend draining (503)")
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out["status"] == "skipped", out
    assert out["summary"] == "backend draining (503)", out
    assert _steward_rows() == [], "the drain path promised to write no row"
    assert W._read_state() == before, "the drain path promised to write no state"


def test_the_timeout_branch_still_returns_the_string_it_returned_before(board,
                                                                        monkeypatch):
    """The scope line this round did not cross, pinned rather than asserted in
    prose: the dead `TurnTimeout` branch keeps its own returned string.

    Whether it should get the same recording treatment is owed-check's ruling from
    #3088's owed entry, not this diff's — and the branch is measurably dead, 0
    fires in 1071 board-steward runs as triage counted on 2026-10-25. Pinning the
    string is what makes that ruling a change someone has to make on purpose.
    """
    from workers.sources._common import TurnTimeout
    _seed_window_an_hour_back()
    _dying_engine(board, monkeypatch, TurnTimeout, message="120s")
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out == {"status": "failed", "summary": "steward turn timed out: 120s"}, out
    assert _steward_rows() == [], "the timeout branch writes no row today, still"
