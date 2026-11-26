"""Cut 3 of senses-not-supervision: one steward turn beside the state machine.

Shipped with `apply: false`. Every pass records how far the steward's answer
agrees with what `desired_statuses` would have done, and that record — not a
week of hope — is what decides whether `apply` flips. The one rule that is
never the steward's: it cannot set `done`.
"""

from __future__ import annotations

import difflib
import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.automod import backlog as B
from workers import board_disagreement as D
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
                   # #3766 re-pointed this rule at the field it had no way to read;
                   # the sentence below is the rule it replaced, kept in the assert so a
                   # reader can see the condition is now `triage=none` and not a silence.
                   "whose line reads `triage=none` goes back to",
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


def test_the_shipped_state_does_not_apply_after_the_stop_condition_fired():
    """What the shipped source is configured to do, read off the real config: NOT apply.

    This node was `test_the_shipped_state_applies` and its body said
    `assert cfg["apply"] is True`. #4409 re-pinned it to False, and moved the name with
    the value, because the name was a claim too: a passing node called
    `..._ships_the_flag_on_...` over `apply: false` tells the next reader the steward
    writes, which is the exact false reading this item was filed to close.

    `apply` was false from Cut 3 (#3088) until #4147 flipped it at 2026-11-16T00:06Z on
    the gate the block beside it states (summed judged 140 and rate 0.9714, both over the
    rolling 7-day window), and the STOP CONDITION in that block fired 18h48m53s later, on
    2026-11-16T18:54:53Z — the trip record beside the flag now says so, and closes the
    flip as 'tried, didn't work out'. What this node is for has not changed across
    either direction: it is the YAML-leg counterpart of the comment pins, the thing that
    catches a block which recites figures beside a value they no longer justify, and the
    reason a revert of this flag is a real edit rather than a stray keystroke. Whether
    the flag may go back to true, and on what bar, is owed-check's ruling (#4409's owed
    entry), not this node's.
    """
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

#: An unmeasured window: the #3519 reader files only on a MEASURED disagreement,
#: so the default here is inert and every test that does not ask about the reader
#: cannot file an item or open the queue. Its own tests pass a real one.
NO_WINDOW = {"judged": 0, "agree": 0, "disagree": 0, "rate": None,
             "ticks_in_window": 0, "ticks_with_judged": 0, "truncated": False,
             "window_since": "2026-10-20T00:00:00+00:00"}


def _steward_tick(board, monkeypatch, *, moves=(), expected=None, window=None):
    """Run one dry-run tick on the ten-item board and return (result, ledger row).

    `expected` replaces `desired_statuses` so a test can hand the machine an
    opinion (or none) without building the ledger history that produces one.
    `window` is the trailing-7-day judgement the tick's #3519 reader sees; the
    default is an unmeasured window, so the reader stays inert for every test
    that is not about it.
    """
    import asyncio

    from workers.sources import _common as C
    from workers import board_disagreement as D

    monkeypatch.setattr(D, "window_now", lambda **kw: dict(window or NO_WINDOW))
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
    # #4233: the three prose caps are `GRAMMAR_HEADROOM` ABOVE the parse ceilings,
    # so an over-ceiling answer is admissible and `parse_steward` stays the cut
    # that decides what is stored. The literal is pinned beside the constant so a
    # cap widened further still has to be signed off twice.
    assert move_props["note"]["maxLength"] == W.NOTE_MAX + W.GRAMMAR_HEADROOM == 332
    assert (props["next_pick_reason"]["maxLength"]
            == W.REASON_MAX + W.GRAMMAR_HEADROOM == 332)
    assert props["summary"]["maxLength"] == W.SUMMARY_MAX + W.GRAMMAR_HEADROOM == 432
    assert move_props["tags_add"]["items"]["maxLength"] == W.TAG_MAX == 40, (
        "the tag cap widened with the prose — a tag is a key the board matches by "
        "string, so a word-boundary cut would corrupt it, not shorten it")

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
    observed 19. The three prose GRAMMAR caps now sit `GRAMMAR_HEADROOM` (32)
    characters above the parse ceilings (#4233), so the parser and not the grammar
    is the cut — and what this node pins is the ceiling beside that: a reply AT
    300/300/400, the size the longest legitimate note, reason and summary in the
    ledger sit at, still comes back byte for byte the answer the steward gave,
    because `_fit_prose` is called with these ceilings, never with the grammar
    caps."""
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


# ---------------------------------------------------------------------------
# #3991: the prose cut. A steward field that overflows its cap is stored cut on a
# word boundary with the ASCII marker inside the cap, and never ends on a
# non-ASCII fragment; a field that fits is stored byte-identical.
# ---------------------------------------------------------------------------

def _reply_with_prose(note: str = "", reason: str = "", summary: str = "") -> dict:
    """A steward answer whose three prose fields are exactly `note`/`reason`/`summary`."""
    return {"moves": [{"item_id": 7, "status": "draft", "tags_add": [],
                       "tags_remove": [], "note": note}],
            "next_pick": 0, "next_pick_reason": reason, "summary": summary}


def _overflow_with_cjk_at_the_cut(cap: int) -> str:
    """Text LONGER than `cap` whose non-ASCII run ends EXACTLY at its cut boundary.

    `shorten_summary` cuts at index `cap - 3` and backs off to the previous word
    only when the character AT that index is not whitespace. So the alignment that
    tests #3991's strip is a fragment ENDING at `cap - 4` with a SPACE at
    `cap - 3`: the back-off finds its boundary exactly where the fragment stops and
    keeps the fragment. Any other over-`cap` string is fixed by the back-off alone,
    and a test built from one would pass green with the strip deleted — which is
    the trap #3991's triage names. The assertions are this fixture's own contract:
    if an edit stops them holding, the test that uses it has stopped proving.
    """
    from workers.queue import SUMMARY_MARKER
    room = cap - len(SUMMARY_MARKER)                     # where `shorten_summary` cuts
    frag_len = room % 6 or 6                             # room - frag_len ≡ 0 (mod 6)
    frag = ("机器" * 4)[:frag_len]
    head = " ".join(["alpha"] * ((room - frag_len) // 6))   # each word is 5 chars + 1 space
    text = " ".join([head, frag, " ".join(["omega"] * 20)])
    assert len(text) > cap, f"the fixture does not overflow the cap {cap}: {len(text)}"
    assert text == " ".join(text.split()), "the fixture must already be whitespace-normal"
    assert ord(text[room - 1]) > 127, (
        f"the fragment has to occupy the character BEFORE the cut; index {room - 1} "
        f"holds {text[room - 1]!r}, so a green run here would not prove the strip exists")
    assert text[room] == " ", (
        f"the character AT the cut has to be the space that stops the back-off; "
        f"index {room} holds {text[room]!r}")
    return text


def test_a_note_over_the_cap_is_cut_on_a_word_boundary_with_the_marker():
    """Clause 1: an overflowing `note` is stored cut at a word boundary, ends with
    the truncation marker, and the marker counts against `NOTE_MAX`.

    The line this replaces was `" ".join(...split())[:NOTE_MAX]` — a character
    slice that ended the note where 300 characters ran out. The fixture is whole
    5-character words on a 6-character stride, so index 297 falls inside the word
    at 294-298: the blind slice stops at `…alph`, the guard has to back off to the
    space at 293, and the two answers are different strings. A cut that lands on a
    boundary is also SHORTER than the cap, which is the shape the post-landing
    ledger scan looks for (no new value sitting exactly at the cap).
    """
    from workers.queue import SUMMARY_MARKER
    note = " ".join(["alpha"] * 100)                     # 599 characters of whole words
    assert len(note) > W.NOTE_MAX
    stored = W.parse_steward(_reply_with_prose(note=note))["moves"][0]["note"]
    assert stored.endswith(SUMMARY_MARKER), "a note that was cut has to say so"
    body = stored[:-len(SUMMARY_MARKER)]
    assert len(stored) <= W.NOTE_MAX, "the marker has to count against the cap"
    assert len(stored) < W.NOTE_MAX, (
        "a word-boundary cut is SHORTER than the cap — a value sitting at exactly "
        f"NOTE_MAX is the blind slice, not the guard: tail {stored[-12:]!r}")
    assert note.startswith(body) and note[len(body)] == " ", (
        f"the cut landed mid-word, not on a boundary: tail {stored[-12:]!r}")
    assert not body.endswith("alph"), "half a word survived the cut"


def test_a_cut_note_never_ends_on_a_non_ascii_fragment():
    """Clause 2: the degenerate tail a cut leaves behind is dropped, not stored.

    The ledger the item was filed from holds four cap-length `moves[].note` values
    ending in CJK — newest 2026-11-09T00:49:54Z, "…so a draft proposal is机器" —
    and the note is a human-facing field on the item's activity log, so a
    mid-token non-Latin fragment is unreadable to every later reader.

    Three alignments, because only the first needs the new strip: the fragment ENDING
    at the cut (where `shorten_summary`'s back-off does nothing), the fragment
    STRADDLING the cut (where the back-off already handles it and the guard must not
    un-handle it), and a reply that overflowed while non-ASCII THROUGHOUT, which has
    no readable prefix to keep and gets the marker alone rather than a fragment.
    """
    from workers.queue import SUMMARY_MARKER
    stored = W.parse_steward(_reply_with_prose(
        note=_overflow_with_cjk_at_the_cut(W.NOTE_MAX)))["moves"][0]["note"]
    assert all(ord(c) <= 127 for c in stored), (
        f"a cut note was stored ending {stored[-8:]!r} — the fragment at the cut has "
        "to be dropped along with the text after it")
    assert stored.endswith(SUMMARY_MARKER), "dropping the fragment must still say so"
    assert len(stored) < W.NOTE_MAX
    assert _overflow_with_cjk_at_the_cut(W.NOTE_MAX).startswith(
        stored[:-len(SUMMARY_MARKER)]), "the kept text was reflowed"

    straddles = " ".join(["alpha"] * 49) + " 机器机器机器机器 " + " ".join(["omega"] * 20)
    assert len(straddles) > W.NOTE_MAX
    cut_mid = W.parse_steward(_reply_with_prose(note=straddles))["moves"][0]["note"]
    assert all(ord(c) <= 127 for c in cut_mid), f"straddling run kept: {cut_mid[-10:]!r}"
    assert cut_mid.endswith(SUMMARY_MARKER)

    all_cjk = " ".join(["机器机器"] * 80)                 # 399 chars, no ASCII anywhere
    assert len(all_cjk) > W.NOTE_MAX
    assert W.parse_steward(_reply_with_prose(note=all_cjk))["moves"][0]["note"] == (
        SUMMARY_MARKER), "a field with nothing readable to keep must store the marker alone"


def test_the_reason_and_the_summary_get_the_same_cut_as_the_note():
    """Clause 3: the same cut-and-drop at each field's own cap — `next_pick_reason`
    at `REASON_MAX` (300) and `summary` at `SUMMARY_MAX` (400).

    All three are cut by the same expression in `parse_steward`, and the live-ledger
    scan that filed this item counted 4 cap-length CJK-terminated `note` values
    beside 9 `next_pick_reason` and 13 `summary` ones, so a guard on `note` alone
    would have left the majority of the observed rows exactly as degenerate.
    """
    from workers.queue import SUMMARY_MARKER
    for label, cap in (("note", W.NOTE_MAX), ("next_pick_reason", W.REASON_MAX),
                       ("summary", W.SUMMARY_MAX)):
        text = _overflow_with_cjk_at_the_cut(cap)
        parsed = W.parse_steward(_reply_with_prose(
            note=text if label == "note" else "",
            reason=text if label == "next_pick_reason" else "",
            summary=text if label == "summary" else ""))
        got = (parsed["moves"][0]["note"] if label == "note" else
               parsed["next_pick_reason"] if label == "next_pick_reason" else
               parsed["summary"])
        assert got, f"{label} came out empty"
        assert got.endswith(SUMMARY_MARKER), f"{label} was cut with no marker"
        assert len(got) < cap, f"{label} sits at its cap ({len(got)}={cap}) mid-token"
        assert all(ord(c) <= 127 for c in got), (
            f"{label} was stored ending {got[-8:]!r}, the shape the ledger scan counted")
        assert text.startswith(got[:-len(SUMMARY_MARKER)]), f"{label} was reflowed"


def test_a_field_at_or_under_its_cap_is_stored_byte_identical():
    """Clause 4: the guard triggers on overflow only, so an answer that is legal at
    the caps is recorded as the answer the steward gave, with its non-ASCII kept.

    Two `>=` bugs, both pinned. A field of EXACTLY its cap — the size the longest
    legitimate replies in the ledger sit at, which is why `test_the_caps_sit_above_
    the_largest_reply_a_real_tick_ever_sent` asserts `parsed == reply` for exact-cap
    input — has to survive whole. And a field that FITS while ending in CJK has to
    keep that CJK: the strip belongs to a cut, and a short note written in Chinese
    is the steward writing a real sentence, not degenerating. (The clause as written
    cites a node named `test_a_reply_at_the_caps_survives_the_parser_whole`; that
    name is not in this file and never was — `git grep -n 'parsed == reply'` on the
    base commit lands on the assertion above, which is what the clause means.)
    """
    from workers.queue import SUMMARY_MARKER
    exact_note, exact_reason, exact_summary = (_normalised(W.NOTE_MAX),
                                               _normalised(W.REASON_MAX),
                                               _normalised(W.SUMMARY_MAX))
    parsed = W.parse_steward(_reply_with_prose(note=exact_note, reason=exact_reason,
                                               summary=exact_summary))
    assert parsed["moves"][0]["note"] == exact_note
    assert parsed["next_pick_reason"] == exact_reason
    assert parsed["summary"] == exact_summary
    assert SUMMARY_MARKER not in exact_summary, "a field that fits got a marker"

    ends_cjk = _normalised(W.NOTE_MAX - 2) + "机器"        # exactly at the cap, ends non-ASCII
    assert len(ends_cjk) == W.NOTE_MAX
    kept = W.parse_steward(_reply_with_prose(note=ends_cjk))["moves"][0]["note"]
    assert kept == ends_cjk, (
        f"a note AT the cap was re-cut (tail {kept[-10:]!r}): the trigger is "
        "len(text) > cap, never >=")

    short = "净流量 +43，needs-human 保持 parked — 已确认"
    assert len(short) < W.NOTE_MAX
    assert W.parse_steward(_reply_with_prose(note=short))["moves"][0]["note"] == short, (
        "legitimate non-ASCII in a field that fits must survive untouched")


# ---------------------------------------------------------------------------
# #4233: `STEWARD_SCHEMA` caps its prose 32 characters ABOVE the ceilings
# `_fit_prose` cuts to, so an over-ceiling answer ARRIVES at the parser instead of
# being sliced by the grammar at the ceiling. With the two numbers equal, #3991's
# cut could not run: over 1,931 `board_steward` ledger rows only 2 carry the "..."
# marker (both before #3991 landed), while 179 post-landing values sit at exactly
# their cap and 6 end on a lone CJK codepoint — the newest booked 2026-11-17.
# ---------------------------------------------------------------------------

def _admissible_overflow(ceiling: int, grammar_max: int) -> str:
    """Whole-word prose strictly OVER `ceiling` and AT OR UNDER `grammar_max`.

    Whole 5-character words on a 6-character stride, so a length of `6n - 1`: the
    shortest such length past the ceiling. Whole words because the shipped value
    has to be judgeable against a WORD boundary, and inside `grammar_max` because
    a fixture the widened grammar still rejects would be testing an answer the
    decoder could never have sent.
    """
    text = " ".join(["alpha"] * ((ceiling + 4) // 6 + 1))
    assert ceiling < len(text) <= grammar_max, (
        f"{len(text)} characters is neither over the ceiling {ceiling} nor inside "
        f"the grammar cap {grammar_max}: the fixture stopped being admissible")
    return text


def _tick_with_prose(board, monkeypatch, note: str, reason: str, summary: str) -> dict:
    """One real steward tick whose model answer carries exactly these three prose
    fields; returns its `board_steward` ledger row, read back off the file.

    `run_prompt_in_session` is the only fake: the parse path, the row writer and
    the ledger are production, because what is under test is what a live tick books
    once the grammar stops cutting first.
    """
    from workers.sources import _common as C

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": _reply_with_prose(note=note, reason=reason,
                                                summary=summary)}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out["status"] == "success", out
    return [e for e in S.read_events(path=S.LEDGER_PATH)
            if e.get("event") == "board_steward"][-1]


def test_the_grammar_admits_an_answer_over_each_prose_ceiling():
    """#4233 clause 1 (with clause 2's ceiling): the schema caps the three prose
    fields `GRAMMAR_HEADROOM` above the ceiling `_fit_prose` is called with, that
    ceiling still bounds everything shipped, and the comment that used to forbid
    the gap is gone.

    The RELATION is the half that matters: `GRAMMAR_HEADROOM` has to be positive or
    the grammar slices at the ceiling first and `_fit_prose`'s `len(text) > limit`
    is never true in production. The count caps and the tag cap stay EXACTLY at
    their parse ceilings, and the assertion here is what keeps them there — a tag
    is a key the board matches by string, so a word-boundary cut would corrupt it
    rather than shorten it, and a count has nothing to cut.

    Clause 2 is measured rather than read off the source: every length from one
    character past a ceiling out to its grammar cap goes through `parse_steward`
    and comes back at or under the ceiling. Handing `_fit_prose` the grammar cap
    instead of the ceiling would ship a 332-character note, and this loop is red on
    the first such field.
    """
    import inspect

    assert W.GRAMMAR_HEADROOM == 32 > 0, (
        "a headroom of 0 puts the grammar back at the ceiling, where its own slice "
        "makes _fit_prose's word-boundary cut unreachable")
    for name, kw, ceiling, grammar_max in (
            ("note", "note", W.NOTE_MAX, W.NOTE_GRAMMAR_MAX),
            ("next_pick_reason", "reason", W.REASON_MAX, W.REASON_GRAMMAR_MAX),
            ("summary", "summary", W.SUMMARY_MAX, W.SUMMARY_GRAMMAR_MAX)):
        assert grammar_max == ceiling + W.GRAMMAR_HEADROOM, name
        for length in (ceiling + 1, ceiling + 8, grammar_max):
            text = _normalised(length)
            assert len(text) == length and text == " ".join(text.split())
            parsed = W.parse_steward(_reply_with_prose(**{kw: text}))
            got = parsed["moves"][0]["note"] if name == "note" else parsed[name]
            assert len(got) <= ceiling, (
                f"a {name} of {length} characters — admissible grammar, over the "
                f"ceiling — shipped {len(got)} characters, over {ceiling}")
            assert got, f"a {name} of {length} characters shipped empty"

    src = inspect.getsource(W)
    for stale in ("exactly the ceiling the parse path imposes",
                  "The two have to be one number"):
        assert stale not in src, (
            f"{stale!r} is back: that is the equal-caps rule #4233 reversed, and "
            "the next reader who trusts it re-breaks the guard")


def test_the_parse_ceilings_and_the_three_cap_comments_stay_where_they_are():
    """#4233 clause 1 (placement) and clause 2 (the call sites): `_fit_prose` is
    still called with `NOTE_MAX`/`REASON_MAX`/`SUMMARY_MAX` — 300/300/400 — and the
    three places that document the relation name the parser as the cutter.

    The source form is what this node can pin that the behaviour loop above cannot:
    `parse_steward` has three `_fit_prose` call sites, and the failure it is here
    for is one of them quietly switching to the grammar cap while the other two
    stay. The discrimininator is exact — `NOTE_MAX)` cannot match inside
    `NOTE_GRAMMAR_MAX)` — so a call site widened to the grammar cap drops the
    count for that ceiling to 0 and this node is red.
    """
    import inspect

    assert (W.NOTE_MAX, W.REASON_MAX, W.SUMMARY_MAX) == (300, 300, 400)
    src = inspect.getsource(W.parse_steward)
    for ceiling in ("NOTE_MAX", "REASON_MAX", "SUMMARY_MAX"):
        assert src.count(f"{ceiling})") == 1, (
            f"`parse_steward` no longer hands _fit_prose exactly one {ceiling}, so "
            "a shipped field can be longer than the ceiling the ledger scan reads")
    for fn in (W._fit_prose, W.parse_steward):
        doc = fn.__doc__ or ""
        assert "#4233" in doc, f"{fn.__name__}'s docstring stopped stating the rule"
        assert "GRAMMAR_HEADROOM" in doc, (
            f"{fn.__name__} no longer names the headroom that makes its own branch "
            "reachable")


def test_an_admissible_over_ceiling_answer_is_booked_already_cut(board, monkeypatch):
    """#4233 clauses 3 and 4, across the seam this round opened: prose longer than
    its ceiling but inside the widened grammar goes through a real tick and is
    booked cut at a word boundary, under the ceiling, ending on the ASCII marker
    with no trailing non-ASCII run — while a field that fits the ceiling is booked
    byte for byte, marker-free, CJK tail and all.

    Driving `parse_steward` directly cannot show this, because what changed is what
    the grammar lets reach it: the ledger the item was filed from is the proof the
    old order never produced a cut value. The tick is the seam — model answer,
    parse, row, ledger — so a guard that only fires in the unit test would leave
    this red.
    """
    from workers.queue import SUMMARY_MARKER
    _ten_item_board(board)

    fields = (("note", W.NOTE_MAX, W.NOTE_GRAMMAR_MAX),
              ("next_pick_reason", W.REASON_MAX, W.REASON_GRAMMAR_MAX),
              ("summary", W.SUMMARY_MAX, W.SUMMARY_GRAMMAR_MAX))
    texts = {name: _admissible_overflow(ceiling, grammar_max)
             for name, ceiling, grammar_max in fields}
    row = _tick_with_prose(board, monkeypatch, note=texts["note"],
                           reason=texts["next_pick_reason"],
                           summary=texts["summary"])
    for name, ceiling, _grammar_max in fields:
        got = (row["moves"][0]["note"] if name == "note" else row[name])
        text = texts[name]
        assert len(text) > ceiling, f"the {name} fixture stopped overflowing"
        assert got.endswith(SUMMARY_MARKER), (
            f"a {name} over the ceiling was booked uncut, so the reader cannot tell "
            "a truncation from a sentence")
        body = got[:-len(SUMMARY_MARKER)]
        assert len(got) < ceiling, (
            f"a {name} sits at {len(got)} == {ceiling}: that is the grammar's blind "
            f"slice, not a word-boundary cut. tail {got[-12:]!r}")
        assert text.startswith(body) and text[len(body)] == " ", (
            f"the {name} cut landed mid-word: tail {got[-12:]!r}")
        assert all(ord(c) <= 127 for c in got), (
            f"a cut {name} ends {got[-8:]!r} — non-ASCII at the tail is the shape "
            "the ledger scan counted")

    fits = _normalised(W.NOTE_MAX - 2) + "机器"
    assert len(fits) == W.NOTE_MAX
    kept = _tick_with_prose(board, monkeypatch, note=fits,
                            reason=_normalised(W.REASON_MAX),
                            summary=_normalised(W.SUMMARY_MAX))
    assert kept["moves"][0]["note"] == fits, (
        f"a note AT the ceiling was re-cut (tail {kept['moves'][0]['note'][-10:]!r})")
    assert SUMMARY_MARKER not in kept["moves"][0]["note"], "a field that fits got a marker"
    assert kept["next_pick_reason"] == _normalised(W.REASON_MAX)
    assert kept["summary"] == _normalised(W.SUMMARY_MAX)


def _degenerate_prose_reply() -> dict:
    """A model answer whose every prose field overflows its cap with a non-ASCII
    fragment sitting AT the cut — the shape the 2026-11-09 ledger scan counted."""
    return {"moves": [{"item_id": 1, "status": "up_next", "tags_add": [],
                       "tags_remove": [], "note": _overflow_with_cjk_at_the_cut(W.NOTE_MAX)}],
            "next_pick": 0,
            "next_pick_reason": _overflow_with_cjk_at_the_cut(W.REASON_MAX),
            "summary": _overflow_with_cjk_at_the_cut(W.SUMMARY_MAX)}


def _tick_with_degenerate_prose(board, monkeypatch, apply: bool) -> dict:
    """One steward tick with `apply` and the degenerate answer above; returns its
    `board_steward` ledger row, read back off the file the writer appended to."""
    from workers.sources import _common as C

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": _degenerate_prose_reply()}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": apply})))
    assert out["status"] == "success", out
    row = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "board_steward"][-1]
    assert row["apply"] is apply, f"the {apply} tick booked the other kind of row"
    return row


def test_a_dry_run_row_and_an_apply_row_book_the_identical_prose(board, monkeypatch):
    """Clause 5: the guard sits at the one normalisation site, so the `apply: false`
    row (what ships today) and the `apply: true` row carry the same sanitised strings
    for the same model reply.

    `parse_steward` is where all three fields are cut and `execute` has a single
    `S.append_event` for both row kinds, so a guard placed beside the write or on the
    apply branch would still reach one of these two rows — which is what this test
    can catch: the pair has to be equal AND the dry-run row has to be sanitised, since
    equality alone is satisfied by two unsanitised copies. The model answer is the
    only fake; the board, the flag split, the agreement and the ledger are production.
    """
    from workers.queue import SUMMARY_MARKER
    _ten_item_board(board)
    dry = _tick_with_degenerate_prose(board, monkeypatch, apply=False)
    live = _tick_with_degenerate_prose(board, monkeypatch, apply=True)
    assert dry["moves"] == live["moves"], "the two rows disagree about the note"
    assert dry["next_pick_reason"] == live["next_pick_reason"], (
        "the two rows disagree about next_pick_reason")
    assert dry["summary"] == live["summary"], "the two rows disagree about the summary"
    for value in (dry["moves"][0]["note"], dry["next_pick_reason"], dry["summary"]):
        assert value.endswith(SUMMARY_MARKER), f"a dry-run value was not cut: {value[-10:]!r}"
        assert all(ord(c) <= 127 for c in value), (
            f"a dry-run value ends {value[-8:]!r} — the strip ran on the apply row only")


def test_the_activity_line_an_apply_tick_writes_is_the_sanitised_note(board, monkeypatch):
    """The one destination of the note that a human reads: `apply_moves` puts
    `m["note"]` into `B.set_status(..., why=...)`, which writes the item's activity
    line and rewrites the item's markdown file — a second store, reached through a
    different writer than the ledger row, so the ledger assertions alone would not
    see a fragment escape that way.

    Asserts the move really landed first: an applied-empty tick would leave the file
    untouched and the rest of this would pass by never having tested anything.
    """
    from workers.queue import SUMMARY_MARKER
    _ten_item_board(board)
    row = _tick_with_degenerate_prose(board, monkeypatch, apply=True)
    landed = [a for a in row["applied"] if a["item_id"] == 1 and a["applied"]]
    assert landed, f"the move never reached item 1, so this file proves nothing: {row['applied']}"
    note = landed[0]["note"]
    assert note.endswith(SUMMARY_MARKER) and len(note) < W.NOTE_MAX
    # The activity entry is YAML front matter, and `yaml.dump` wraps a long scalar
    # across lines, so compare on whitespace-collapsed text rather than bytes.
    flat = " ".join((board / "1-item-1.md").read_text(encoding="utf-8").split())
    assert f"board steward: {note}" in flat, (
        f"the activity line did not carry the note the parser produced: {note[-30:]!r}")
    foreign = {c for c in _degenerate_prose_reply()["moves"][0]["note"] if ord(c) > 127}
    assert foreign, "the fixture's reply must end in a non-ASCII fragment"
    assert not any(ch in flat for ch in foreign), (
        "a CJK fragment from the reply reached the item's own activity log, the "
        "human-facing field this item exists to protect")


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
            props["summary"]["maxLength"]) == (W.NOTE_GRAMMAR_MAX,
                                               W.REASON_GRAMMAR_MAX,
                                               W.SUMMARY_GRAMMAR_MAX) \
        == (332, 332, 432), "the schema's prose caps moved off #4233's headroom"
    assert (W.NOTE_MAX, W.REASON_MAX, W.SUMMARY_MAX) == (300, 300, 400), (
        "the parse ceilings above the schema caps are what `_fit_prose` still cuts "
        "to; the 32-character gap between the two sets is what makes that cut "
        "reachable at all (#4233)")
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


#: The gate reading #4147 published into `config.yaml`'s board-steward block, measured
#: through the shipped reader (`workers.dispatch_watch.judgement_aggregate` over
#: `~/lloyd-data/workers.db` — the same call `GET /api/workers/health` publishes as
#: `judgement` on this source's row) at 2026-11-16T00:06Z: judged 140 of the 50 the leg
#: needs, rate 0.9714 of the 0.90 it needs. The reading one 900 s poll earlier (judged
#: 124, agree 120, disagree 3, rate 0.9677, 96 ticks, truncated false at
#: 2026-11-15T07:46Z) is in the block beside it, so the block carries the trend and not
#: only the last number. These are the figures the flip RAN on, not a bar: the bar is
#: `` `judged` >= 50 `` AND `` `rate` >= 0.90 `` over the rolling 7-day window, and this
#: round moved neither number nor any other source's `apply`.
FLIP_READING = {"judged": 140, "agree": 136, "disagree": 3, "rate": 0.9714,
                "ticks": 93, "truncated": "false", "at": "2026-11-16T00:06"}

#: The cleared reading as the block writes it, in the block's own order, against
#: marker-stripped prose so a re-wrapped comment still reads (the markers have to go, not
#: just the newlines: leaving them in re-joins the text as "the agree-rate is the # half
#: that does not", so a pinned phrase that happens to wrap would fail for where the
#: author broke the line). Every figure the clause names — summed judged, agree,
#: disagree, rate, the tick count, `truncated false` — plus the minute it was measured,
#: in ONE expression: a block missing any one of them is not a reading.
CLEARED_READING_RE = re.compile(
    r"summed judged (\d+), agree (\d+), disagree (\d+), rate ([01]\.\d+), (\d+) ticks "
    r"in the window, truncated (false|true) — both over the rolling 7-day window, "
    r"measured (2026-\d\d-\d\dT\d\d:\d\d)Z")


def _board_steward_block(text: str) -> str:
    """The `board-steward:` source's own lines plus the comment run above its key.

    Not a fixed window of text: that drifts into the neighbouring source's comment and
    lets its prose satisfy a node about this flag. The gate lives in the comment ABOVE
    the key, which is why the slice runs backwards over the comment run.
    """
    lines = text.splitlines()
    keys = [i for i, ln in enumerate(lines) if ln.strip() == "board-steward:"]
    assert len(keys) == 1, (
        f"the source block this gate is written into must be findable by its own key, "
        f"found {len(keys)}")
    start = keys[0]
    while start > 0 and (lines[start - 1].strip().startswith("#")
                         or not lines[start - 1].strip()):
        start -= 1
    end = keys[0] + 1
    while end < len(lines) and (lines[end].startswith("      ")
                                or lines[end].strip().startswith("#")):
        end += 1
    return "\n".join(lines[start:end])


def _flip_block_problems(block: str, *, apply_value) -> list[str]:
    """Which pins the flip block FAILS, each labelled. Empty list == every pin holds.

    The pins are the ones #3329 shipped, moved to the cleared side by #4147, plus the
    caveat and the stop condition the flip owes. They live in a function rather than in
    the node's body for one reason: `test_every_flip_block_pin_reddens_on_its_own_
    mutation` has to run them against a MUTATED copy of the same block, and a mutation
    test that re-implements the predicate proves nothing about the node it imitates. The
    label on each entry is what lets that node say which pin a mutation tripped.
    """
    from workers import dispatch_watch as DW
    prose = " ".join(ln.lstrip().lstrip("#").rstrip() for ln in block.splitlines())
    flat = " ".join(prose.split())
    low = flat.lower()
    problems: list[str] = []

    # The flag itself. Prose is no proof of a value: the comment says "Flip to
    # `apply: true` only after …", so a substring test would trip on its own instruction.
    # `apply_value` is the YAML-parsed value, and this first pin only proves the slice
    # captured the key line the parsed value came from.
    if not re.search(r"^\s+apply:\s*(?:true|false)\s*$", block, re.MULTILINE):
        problems.append("[flag key captured] the slice did not capture the flag the "
                        "comment gates: " + block[-200:])
    if apply_value is not False:
        problems.append(
            f"[apply ships off] `workers.sources.board-steward.apply` reads "
            f"{apply_value!r}, not False — the STOP CONDITION below FIRED on "
            f"2026-11-16T18:54:53Z, and the round that closes the flip (#4409) set the "
            f"flag back to false, so a value of True here is the flip re-opened by a "
            f"round with no business re-opening it: whether it may be flipped again, "
            f"and on what bar, is owed-check's to rule")

    # The ruled floor is the stated floor, and the superseded one is gone. A reader
    # obeys the number written beside the flag, so a retired 100 left in the prose is
    # still a live instruction to the next round.
    if "judged` >= 50" not in block:
        problems.append(
            "[ruled floor] the judged-count threshold in the board-steward block is not "
            "the ruled number — owed-check set the floor to 50 on 2026-10-25: "
            + block[:200])
    if "judged` >= 100" in block:
        problems.append(
            "[retired floor gone] the block still states the floor #3089 proposed and "
            "the 2026-10-25 ruling replaced")
    # The half of the gate the ruling did NOT move. The window is pinned as the phrase
    # attached to the two numbers, not as "7-day" somewhere in the block: the block also
    # says the rate "read 0.7317 over the trailing 7 days", and that incidental phrase
    # satisfied the loose check while the thresholds' own window silently became 30.
    if ">= 0.90" not in block:
        problems.append("[rate floor stated] the agree-rate threshold is not stated as a number")
    if "both over the rolling 7-day window" not in flat:
        problems.append("[one rolling window] the thresholds are meaningless without the "
                        "window they are measured over, and it has to be the window "
                        "attached to THEM")
    # And the prose window is the code window: a drift between the two is the seam
    # failing, not a wording preference.
    if DW.JUDGEMENT_WINDOW_DAYS != 7:
        problems.append(
            "[window is the code window] `JUDGEMENT_WINDOW_DAYS` is no longer 7, so the "
            "window this comment states and the one `judgement_aggregate` measures have "
            "come apart")

    # The figures below describe a DRY RUN, and say so. #4147 ruled them kept — they are
    # what the flip was decided on — and #4409 relabelled them, because with `apply` back
    # to false the steward writes nothing again and a paragraph headed as today's reading
    # is the false absence #3089 set out to remove, one flip later: a reader who takes
    # `rate` 0.9714 as the steward's CURRENT standing believes an agreement figure
    # measured while the flag was off.
    if "pre-flip-only reading" not in low:
        problems.append(
            "[reading labelled pre-flip-only] the cleared figures are no longer labelled "
            "as the reading the flip RAN ON, so they read as the steward's current "
            "standing — which `apply: false` says it does not have")

    # The reading: every figure, in one expression, with the minute it was measured. A
    # bare `2026-\d\d-\d\d` is not enough — the block also carries the ruling's date and
    # the plateau weeks' end dates, and only the measurement is dated to the minute.
    if not re.search(r"2026-\d\d-\d\dT\d\d:\d\dZ", block):
        problems.append("[reading stamped to the minute] the figures need the moment they "
                        "were measured, or the next reader cannot tell a stale reading "
                        "from a current one")
    m = CLEARED_READING_RE.search(flat)
    if not m:
        problems.append(
            "[cleared reading] the block states no reading carrying summed judged, "
            "agree, disagree, rate, the tick count, `truncated false` and the measured "
            "minute in the block's one reading form")
    else:
        got = {"judged": int(m.group(1)), "agree": int(m.group(2)),
               "disagree": int(m.group(3)), "rate": float(m.group(4)),
               "ticks": int(m.group(5)), "truncated": m.group(6), "at": m.group(7)}
        for key, want in FLIP_READING.items():
            if got[key] != want:
                problems.append(
                    f"[cleared reading] figure `{key}`: the block says {got[key]!r} and "
                    f"the round measured {want!r} through the shipped reader")
        if got["judged"] < 50 or got["rate"] < 0.90 or got["truncated"] != "false":
            problems.append(
                "[cleared reading] its own figures do not clear the bar beside them "
                "(`judged` >= 50 AND `rate` >= 0.90, truncated false), so the block "
                "calls an unmet gate cleared")
    # And it says so in words. The banned pair is the other side of the same pin: the
    # block that says CLEARED and still carries the unmet wording is two readings.
    if "the gate reads cleared" not in low:
        problems.append("[gate reads cleared] the comment must say what today's figures "
                        "read, so a cleared gate is not mistaken for an absent one")
    if "not cleared" in low:
        problems.append("[unmet wording gone] the block still reads NOT cleared beside a "
                        "flag that is now true — the flip's own file cannot hold both")
    if "the agree-rate is the half that does not" in low:
        problems.append("[retired half-sentence gone] the block still names the "
                        "agree-rate the half that does not clear, which is the reading "
                        "#4147 replaced")
    # The reading before this one stays quoted: the trend is the evidence that the gate
    # cleared by ageing AND by agreement, and #3329's bar was fitted to neither.
    if "0.9677" not in flat or "judged 124" not in flat:
        problems.append("[previous reading kept] the 900 s poll before the flip read "
                        "judged 124 / rate 0.9677 and the block is supposed to carry it")

    # Settled, with the ruling's reason and not a refuted one. An assert on "owed"
    # proves nothing here because "owed-check" contains it, so the settled wording itself
    # is what has to be present.
    if "settled by the owed-check ruling of 2026-10-25" not in flat:
        problems.append("[numbers settled] the two numbers have to read as ruled, not as "
                        "proposals still on trial")
    if "still owed" in low:
        problems.append("[not on trial] the block still calls the numbers a ruling owed "
                        "to owed-check, which #3089 settled on 2026-10-25")
    if "80/week plateau" not in flat:
        problems.append("[floor's reason] the reason 50 is the floor is the measured "
                        "plateau under the old one; a number with no reason beside it "
                        "gets re-litigated by the next reader")
    if "163/82/78/82" not in flat:
        problems.append("[plateau evidence] the four weekly judged counts behind that "
                        "plateau are the evidence for the floor")
    if "unreachable" in low:
        problems.append("[no refuted reason] the block repeats a claim the ruling's own "
                        "first week (judged 163, over the old floor) refutes")
    if "rate bar was deliberately not lowered" not in low:
        problems.append("[rate bar not lowered] keeping 0.90 while moving the floor was a "
                        "decision, and the comment has to say it was one")
    if "0.7317" not in flat or "0.80" not in flat:
        problems.append("[history kept] the rate readings the bar was deliberately NOT "
                        "fitted to are the whole point of that sentence")

    # The rate is only evidence while the steward is still dry-running, the mechanism is
    # named beside the claim, and the post-flip measure and the stop condition are in the
    # same comment as the flag they govern.
    if "pre-flip-only" not in low:
        problems.append("[pre-flip-only caveat] the block has to say that judgement.rate "
                        "stops meaning agreement at the flip")
    if "never as a fresh `disagree`" not in flat:
        problems.append("[mechanism named] the pre-flip-only claim without its mechanism "
                        "reads as an opinion: an applied wrong status is never "
                        "re-proposed, so no fresh `disagree` is ever scored on it")
    if "`missed` and in the `judged` denominator only" not in flat:
        problems.append("[where it goes instead] and where the divergence does go — "
                        "`missed`, plus the denominator, which is what pulls the rate "
                        "down while nothing new disagrees")
    if "post-flip 0.90 is a phantom" not in low:
        problems.append("[phantom stated] the caveat has to say what a post-flip 0.90 IS: "
                        "a phantom, because a wrong applied status scores no fresh "
                        "disagreement")
    if "reverted_moves" not in flat:
        problems.append("[post-flip leg named] the block has to name what measures the "
                        "steward after the flip — `reverted_moves`, the applied move the "
                        "machine puts back inside the window")
    if "'tried, didn't work out'" not in flat:
        problems.append("[stop condition] the flip and the route back have to be in the "
                        "same comment: if the alarm fires within 7 days, `apply` goes "
                        "back to false and this reading is recorded as tried, didn't "
                        "work out")
    # Not a substring of the instruction: its bytes. #4409's trip record re-uses the
    # condition's own phrase "`apply` back to false" to report the value the round wrote,
    # so a substring pin could be satisfied by the RECORD while the INSTRUCTION was
    # softened — which is the drift #4443 pinned these bytes against. The constant is
    # defined below this function and read at call time.
    if STOP_CONDITION_VERBATIM not in block:
        problems.append("[stop condition moves the flag] the STOP CONDITION sentence is no "
                        "longer the bytes it was written as, so the instruction the trip "
                        "record now quotes may itself have been edited — and it has to name "
                        "the value it writes, not just the alarm it watches")
    # ── the TRIP RECORD the stop condition asked for, written by #4409 ──────────
    # The sentence above is an instruction to a future round; #4409 is that round, and a
    # round that only flips the value leaves the NEXT reader to re-derive why from a
    # 15,633-row ledger. So the record is pinned in the same predicate as the flag it
    # explains: the condition's own words, the count, the two stamps that bound it, and
    # the two reverts that were real — separated from the pairs the broken pairing
    # manufactured, which is the difference between closing a flip and closing an
    # instrument error.
    if "stop condition fired" not in low:
        problems.append("[trip recorded] the block must open its record with the "
                        "condition's own words — that it FIRED — or a later reader finds "
                        "a reverted flag with no stated trigger")
    if "fired 6 times" not in low:
        problems.append("[trip counts its firings] the record has to carry the number of "
                        "revert-leg firings, which is the trigger the condition was "
                        "written to count")
    if "2026-11-16T18:54:53Z" not in block or "2026-11-20T07:27:31Z" not in block:
        problems.append("[trip bounds its window] the record has to name the first and "
                        "the last firing, which is what makes 'inside 7 days' checkable "
                        "instead of a claim")
    if "2 of the 6 firings' pairs were genuine" not in flat:
        problems.append("[trip separates genuine from mis-paired] 2 of the 6 firings rode "
                        "a real revert and the rest rode the pairing bug; a record that "
                        "says only 'it fired' leaves the mis-paired rows looking like "
                        "steward errors")
    for stamp, who in (("2026-11-16T07:17:31Z", "#4172"),
                       ("2026-11-17T03:47:29Z", "#4151")):
        if who not in block or stamp not in block:
            problems.append(f"[trip names {who}] the genuine reverts are the reason for "
                            f"the flag, so {who} and its stamp {stamp} have to be in the "
                            "record beside it")
    if "were mis-paired rows" not in flat:
        problems.append("[trip names the mis-paired items] #4240 and #4244 are the shape "
                        "the old pairing mistook for a revert; the record says which items "
                        "were never reverted, and why the leg now pairs on the latest row")
    # BY ID, in the block's own bytes — not paraphrased as "the others". An item the record
    # does not name is an item nobody can look up in the store, and #4409's whole claim is
    # that 2 of the 6 firings were false, which is only checkable if the false ones are
    # findable. `#`-prefixed like the genuine pair above, so both directions read alike.
    separating = flat[flat.find("mis-paired"):] if "mis-paired" in flat.lower() else ""
    for mispaired in ("#4240", "#4244"):
        if mispaired not in separating:
            problems.append(f"[trip names {mispaired}] the record does not carry "
                            f"{mispaired} itself, so the pair the shipped pairing "
                            f"manufactured for it cannot be audited against the store")
    return problems


def test_the_config_ships_the_flag_off_and_states_the_pre_flip_reading():
    """#4409: the stop condition fired, so the flag is off again, and the block says so
    in the trip record's own words.

    Every pin in `_flip_block_problems`, asserted on the real file. #3329 shipped this
    node pinning the UNMET half (`not cleared`, the agree-rate named as the failing leg,
    the flag read as False); #3386 and #3651 re-pinned it to their own readings; #4147
    pinned it to the cleared side with the flag True; #4409 brings it back to the reverted
    side — the flag parses False, the figures that decided the flip stay in the block with
    their historical values but are labelled the PRE-FLIP reading they are, and the trip
    record beside them carries the 6 revert-leg firings, the two genuine reverts (#4172
    and #4151, each with the stamp the machine wrote) and the two pairs the broken pairing
    manufactured (#4240, #4244). The name moved with the value because the name was a
    claim: a node reading 'ships the flag on' over `apply: false` is the false absence
    this item was filed to close.

    The bar itself is unchanged and this node is the witness that it is: `judged` >= 50,
    `>= 0.90` and 'both over the rolling 7-day window' are still in the block, because
    the flip consumed the gate, it did not move it — and a future flip has to clear the
    same two numbers owed-check ruled, on a ruling that is owed-check's to make (#4409's
    owed entry), not a round's.
    """
    text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    import yaml
    cfg = yaml.safe_load(text)
    apply_value = cfg["workers"]["sources"]["board-steward"]["apply"]
    problems = _flip_block_problems(_board_steward_block(text), apply_value=apply_value)
    assert problems == [], problems


def test_every_flip_block_pin_reddens_on_its_own_mutation():
    """A pin that cannot fail is not a pin: each mutation below breaks exactly one
    claim about the flip and has to be caught by the label that guards it.

    Two of these are #4409's own words — restore `not cleared` into the block, or flip
    `apply` back to `true` on a value the trip record says is false — and the rest are the
    pins a careless later round is most likely to break while editing the comment for
    another reason: the measured minute, the window attached to the two numbers, a figure,
    the stop condition, the retired half-sentence typed back in, and the trip record's
    count, stamps and named reverts.
    """
    text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    import yaml
    assert yaml.safe_load(text)["workers"]["sources"]["board-steward"]["apply"] is False, (
        "the mutations below are written against the reverted block; on a file still "
        "reading True this node would be proving the wrong thing")
    block = _board_steward_block(text)
    assert _flip_block_problems(block, apply_value=False) == [], (
        "the committed block does not pass its own pins")

    def labels(mutated: str, *, apply_value=False) -> str:
        return " ".join(_flip_block_problems(mutated, apply_value=apply_value))

    # `not cleared` typed back in, wherever the comment wrap happens to fall.
    restored = re.sub(r"reads\s+(?:#\s*)?CLEARED", "reads NOT cleared", block, count=1)
    assert restored != block and "[unmet wording gone]" in labels(restored), restored
    assert "[gate reads cleared]" in labels(restored)
    # The flag value, flipped back on. Prose untouched: this is the YAML leg failing
    # alone, and it is the leg that makes a re-flip a decision rather than an accident.
    assert "[apply ships off]" in labels(block, apply_value=True)
    # The reading's own label, generalised back into today's standing.
    relabelled = block.replace("PRE-FLIP-ONLY READING", "TODAY'S READING")
    assert relabelled != block and "[reading labelled pre-flip-only]" in labels(relabelled), \
        labels(relabelled)
    # The trip record, thinned to 'it fired': the count, then the two stamps, then the
    # two items that were never reverted. Each is a separate claim, so each reddens
    # separately.
    uncounted = block.replace("fired 6 times", "fired")
    assert uncounted != block and "[trip counts its firings]" in labels(uncounted), \
        labels(uncounted)
    unbounded = block.replace("2026-11-16T18:54:53Z", "the first one")
    assert "[trip bounds its window]" in labels(unbounded), labels(unbounded)
    unnamed = block.replace("#4172 applied `up_next`", "one item applied `up_next`")
    assert "[trip names #4172]" in labels(unnamed), labels(unnamed)
    unpaired = block.replace("were mis-paired rows", "were also reported")
    assert unpaired != block and "[trip names the mis-paired items]" in labels(unpaired), \
        labels(unpaired)
    # The ids paraphrased away: the sentence survives, the auditability does not.
    unnamed = re.sub(r"#42(?:40|44)", "another item", block)
    assert unnamed != block and "[trip names #4240]" in labels(unnamed), labels(unnamed)
    assert "[trip names #4244]" in labels(unnamed), labels(unnamed)
    # (`2 of` and `the 6 firings` land on opposite sides of the comment wrap, so a plain
    # string replace would match nothing and the mutation would prove nothing).
    merged = re.sub(r"2\s+of\s+(?:#\s*)?the\s+(?:#\s*)?6", "several", block, count=1)
    assert merged != block and "[trip separates genuine from mis-paired]" in labels(merged), \
        labels(merged)
    unrecorded = block.replace("STOP CONDITION FIRED", "STOP CONDITION")
    assert unrecorded != block and "[trip recorded]" in labels(unrecorded), labels(unrecorded)
    # The retired half-sentence, typed back in at the end of the comment run.
    half = block + "\n    # the agree-rate is the half that does not\n"
    assert "[retired half-sentence gone]" in labels(half)
    # The measured minute dropped from the reading (the comment wraps across it).
    unstamped = re.sub(r",\s+(?:#\s*)?measured\s+(?:#\s*)?2026-\d\d-\d\dT\d\d:\d\dZ",
                       "", block)
    assert unstamped != block and (
            "[reading stamped to the minute]" in labels(unstamped)
            or "[cleared reading]" in labels(unstamped)), labels(unstamped)
        # Either pin may be the one that reddens: the reading regex carries the minute
        # inside its one expression, so dropping the stamp can fail the whole reading or
        # only the stamp pin — but the mutation must not read clean.
    # The window that the two numbers are measured over, widened by a reader in a hurry.
    widened = block.replace("both over the rolling 7-day window",
                            "both over the rolling 30-day window")
    assert "[one rolling window]" in labels(widened)
    # A figure that no longer clears the leg it is quoted under.
    thin = re.sub(r"judged\s+(?:#\s*)?140", "judged 40", block, count=1)
    got = labels(thin)
    assert "[cleared reading]" in got, got
    # The stop condition, dropped as too much prose.
    no_stop = block.replace("'tried, didn't work out'", "'shipped'")
    assert "[stop condition]" in labels(no_stop)
    no_stop2 = block.replace("a round sets `apply`", "a round considers whether it sets")
    assert "[stop condition moves the flag]" in labels(no_stop2)
    # The phantom caveat, softened into an opinion.
    soft = block.replace("a post-flip 0.90 is a PHANTOM", "a post-flip 0.90 is a number")
    assert "[phantom stated]" in labels(soft)
    # The post-flip measure, un-named.
    unnamed = block.replace("`reverted_moves`", "the alarm")
    assert "[post-flip leg named]" in labels(unnamed)
    # The retired floor, typed back in beside the flag.
    old_floor = block.replace("summed `judged` >= 50", "summed `judged` >= 100")
    assert "[retired floor gone]" in labels(old_floor)


#: The #4443 caveat's opening words, as committed to `config.yaml`'s board-steward
#: block. That sentence is the ONE piece of prose #4443 allowed into a comment run
#: #4409's round also rewrites, so its own words are the slice anchor: the run around it
#: is 30-odd lines of other rounds' pinned prose, and a marker matched anywhere in there
#: could stay satisfied while the sentence itself went missing.
CAVEAT_4443_ANCHOR = "ONE CAVEAT ABOUT A REVERT (#4443)"

#: The STOP CONDITION sentence, byte for byte as it sits in `config.yaml` at this
#: round's base — comment wrap and hashes included. Writing its trip record is #4409's
#: job, not this round's, so #4443 pinned the bytes rather than hoping two rounds editing
#: one comment block would notice each other: a later round that legitimately rewrites
#: the trip, or merely re-wraps it, has to pass over this node's corpse in its own diff.
STOP_CONDITION_VERBATIM = (
    "STOP CONDITION: if that alarm\n"
    "    # fires within 7 days of this flip (2026-11-16T00:06Z), a round sets `apply`\n"
    "    # back to false and records this reading as 'tried, didn't work out'. The\n"
    "    # revert is the finding, and the round that reverses this one is a normal\n"
    "    # outcome here, not a failure to hide.")


def _caveat_4443_problems(block: str) -> list[str]:
    """Which pins on the #4443 applied-move caveat FAIL, each labelled. Empty == all hold.

    The same shape as `_flip_block_problems`, for the same reason: the deletion proof
    below has to run THIS predicate over a mutated block, and a predicate re-implemented
    there would prove nothing about the node it imitates. The slice is the caveat's own
    comment run — its anchor line down to the first non-comment line, which is the
    `board-steward:` key it sits directly above — so no marker can be satisfied by the
    two paragraphs that neighbour it (one of which already names the flip date).
    """
    lines = block.splitlines()
    hits = [i for i, ln in enumerate(lines) if CAVEAT_4443_ANCHOR in ln]
    if len(hits) != 1:
        return [f"[caveat present] the block holds {len(hits)} lines opening with the "
                f"caveat's anchor, not exactly 1 — the sentence was deleted or typed "
                f"twice, and a reader who cannot find it cannot be told what to do with "
                f"a silent steward"]
    start = hits[0]
    end = start + 1
    while end < len(lines) and lines[end].lstrip().startswith("#"):
        end += 1
    flat = " ".join(" ".join(ln.lstrip().lstrip("#").rstrip()
                             for ln in lines[start:end]).split())
    problems: list[str] = []
    for label, want in (
            ("[row is applied-only]",
             "written only for a move the steward actually APPLIES"),
            ("[window opens when apply goes false]",
             "the window it bounds opens the moment `apply` goes back to false"),
            ("[silence is not evidence]",
             "neither an absent `steward_move` row nor a zero `up_next -> draft: never "
             "triaged` count is evidence that #4412"),
            ("[the one reading]", "rows > 0 with the park-line count at 0"),
            ("[zero rows is unmeasured]", "rows = 0 is UNMEASURED"),
            ("[no absence claim before a revert]",
             "claims nothing about any window before such a revert"),
            ("[flip date named beside the window]", "2026-11-16T00:06Z"),
            ("[row count carries its own date]",
             "47 of them as measured 2026-11-20, newest 2026-11-20T15:47:54Z")):
        if want not in flat:
            problems.append(f"{label} the caveat no longer states {want!r}")
    return problems


def test_the_config_states_that_only_an_applied_move_writes_a_steward_move_row():
    """#4443: the block says a `steward_move` row exists only for an APPLIED move, so a
    steward switched back to `apply: false` writes none and its silence proves nothing.

    The confound is measured, not feared: #4412's volume clause is graded on
    `up_next -> draft: never triaged` park lines, while `apply_moves` runs only under
    `if apply:` and the row is appended only under `if moved_status:` inside it — so a
    round that reverts the flag drives the park count to 0 by itself. Rows > 0 with the
    park count at 0 is the fix; rows = 0 is UNMEASURED, and the sentence has to say that
    or the next `apply: false` round is read as the repair. #4409 landed the revert this
    caveat was written for, so the tree is now on the window's OTHER side: `apply` is
    false again from 2026-11-21, the park count is 0 by construction, and the warning is
    the live reading rather than a hypothetical. The sentence stays honest about the side
    it is not on: rows were live from the 2026-11-16T00:06Z flip (47
    as measured 2026-11-20, newest 2026-11-20T15:47:54Z), so it claims nothing about a
    window before a revert. The YAML leg beside it is what stops the prose being pinned
    while the flag quietly moved.
    """
    text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    import yaml
    assert yaml.safe_load(text)["workers"]["sources"]["board-steward"]["apply"] is False, (
        "the caveat holds on both sides of a revert, but #4409 landed the revert, so this "
        "tree is the `apply: false` side — a file reading True here is the flip re-opened "
        "past its own stop condition, and these pins would be guarding the wrong half")
    block = _board_steward_block(text)
    problems = _caveat_4443_problems(block)
    assert problems == [], problems
    assert STOP_CONDITION_VERBATIM in block, (
        "the STOP CONDITION sentence is no longer the bytes #4443 pinned: its trip "
        "record is #4409's to write, so a round that re-wrapped or restated it here has "
        "taken #4409's clause")


def test_the_applied_move_caveat_pin_reddens_on_its_own_deletion():
    """A pin that cannot fail is not a pin: removing the caveat, or any one of its
    load-bearing halves, has to redden the label that guards it.

    The first mutation is the one the item asked for — the whole sentence gone, which is
    also what #4409's round does by accident if it rewrites this comment run without
    seeing the caveat inside it. The rest are the quiet half-edits a later prose edit
    makes while intending something else: the mechanism softened, the window's start
    dropped, the flip date typed off, the UNMEASURED reading deleted across the comment
    wrap, and the no-absence-before-a-revert half turned into a universal claim. The last
    pair proves the byte pin bites on a re-wrap alone.
    """
    text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    block = _board_steward_block(text)
    assert _caveat_4443_problems(block) == [], (
        "the committed caveat does not pass its own pins")
    assert STOP_CONDITION_VERBATIM in block

    lines = block.splitlines()
    start = next(i for i, ln in enumerate(lines) if CAVEAT_4443_ANCHOR in ln)
    end = start + 1
    while end < len(lines) and lines[end].lstrip().startswith("#"):
        end += 1
    deleted = "\n".join(lines[:start] + lines[end:])
    assert deleted != block
    assert "[caveat present]" in " ".join(_caveat_4443_problems(deleted))

    def labels(mutated: str) -> str:
        return " ".join(_caveat_4443_problems(mutated))

    # The mechanism, softened from applied to proposed: the whole point of the caveat.
    assert "[row is applied-only]" in labels(
        block.replace("actually APPLIES", "is proposed"))
    # The window's start, generalised into prose about windows in general.
    assert "[window opens when apply goes false]" in labels(
        block.replace("opens the moment `apply` goes back to false", "opens at a revert"))
    # The two silence-is-not-evidence claims, dropped as too much prose.
    assert "[silence is not evidence]" in labels(
        block.replace("neither an absent", "and not an absent"))
    # The flip date, typed off everywhere — the caveat's own slice has to lose it.
    assert "[flip date named beside the window]" in labels(
        block.replace("2026-11-16T00:06Z", "the flip"))
    # The UNMEASURED half, deleted across the comment wrap.
    assert "[zero rows is unmeasured]" in labels(
        re.sub(r"rows\s*=\s*0(?:\s+#\s*)?is\s+UNMEASURED", "and silence settles it", block))
    # The honesty half, turned into a claim about every window.
    assert "[no absence claim before a revert]" in labels(
        block.replace("claims nothing about any window before such a revert",
                      "claims the same for every window"))
    # A re-wrap of the STOP CONDITION, with its words untouched: the byte pin bites.
    rewrapped = block.replace(STOP_CONDITION_VERBATIM,
                              STOP_CONDITION_VERBATIM.replace("\n    # ", " "))
    assert rewrapped != block and STOP_CONDITION_VERBATIM not in rewrapped, (
        "the byte pin cannot fail, so it pins nothing")


#: owed-check's re-flip bar, ruled 2026-11-22 on #4409's owed entry, as committed to
#: `config.yaml`'s board-steward block. The two boundary strings are that paragraph's own
#: first and last words, and the slice runs between them rather than over a fixed window
#: of lines: the comment run above `board-steward:` is ~140 contiguous lines belonging to
#: six other rounds, every one of them pinned elsewhere, so a window would drift into
#: their prose and let THEIR words satisfy a node about this ruling.
REFLIP_BAR_ANCHOR = "THE RE-FLIP BAR"
REFLIP_BAR_TAIL = "the dry-run costs nothing today."
#: The false gloss the item carried: `>= 46 proposals` was attributed to the flip window's
#: own volume ("the 11-16 -> 11-20 volume"). It cannot come from that window — proposals
#: were persisted only once #4381 landed on 2026-11-19 — so a sentence tying the 46 to a
#: flip-window volume is a claim about rows that were never written down. Bounded by
#: characters rather than by sentence because the spelling uses `..` and `->`, both of
#: which a `[^.]` class would refuse to cross.
REFLIP_WINDOW_GLOSS = re.compile(r"(?:11-16|2026-11-20)[^\n]{0,45}?\bvolume\b")


def _reflip_bar_problems(block: str) -> list[str]:
    """Which pins on owed-check's re-flip bar FAIL, each labelled. Empty == all hold.

    The same shape as `_flip_block_problems` and `_caveat_4443_problems`, for the same
    reason: `test_the_ruled_reflip_bar_pin_reddens_on_its_own_deletion` has to run THIS
    predicate over a mutated block, and a predicate re-implemented there would prove
    nothing about the node it imitates. Pins run on the comment-wrap-collapsed form
    because the bar is prose a later round may legitimately re-wrap; the two boundary
    markers are the exception, and are matched per line because bounding the paragraph is
    exactly what a re-wrap must not disturb.
    """
    lines = block.splitlines()
    heads = [i for i, ln in enumerate(lines) if REFLIP_BAR_ANCHOR in ln]
    tails = [i for i, ln in enumerate(lines) if REFLIP_BAR_TAIL in ln]
    if len(heads) != 1 or len(tails) != 1 or tails[0] < heads[0]:
        return [f"[bar present] the block holds {len(heads)} anchor line(s) and "
                f"{len(tails)} tail line(s), not exactly one of each in order — the "
                f"ruling was deleted, typed twice, or its closing sentence moved, and a "
                f"reader who cannot bound the paragraph cannot tell the ruled bar from "
                f"the question it replaced"]
    flat = " ".join(" ".join(ln.lstrip().lstrip("#").rstrip()
                             for ln in lines[heads[0]:tails[0] + 1]).split())
    low = flat.lower()
    whole = " ".join(" ".join(ln.lstrip().lstrip("#").rstrip() for ln in lines).split())
    problems: list[str] = []

    # Ruled, dated, and attributed to the one party that may rule it. #4409's block said
    # "Whether `apply` may be flipped again, and on what bar, is owed-check's to rule";
    # owed-check answered on 2026-11-22, and a bar with no date and no author is a
    # proposal again the moment its author is forgotten.
    if "ruled by owed-check" not in low:
        problems.append("[bar attributed] the block does not name the ruling's author — "
                        "owed-check rules this flag, and no round rules it")
    if "dated 2026-11-22" not in low:
        problems.append("[bar dated] the ruling has no date, so a later reader cannot "
                        "tell it from the pre-flip rulings of 2026-10-25 and 2026-11-16")
    if "`apply` stays false" not in flat:
        problems.append("[bar keeps the flag off] the ruling's first clause is that "
                        "`apply` stays false; a bar that does not say so reads as "
                        "permission")
    if "no round may re-flip it on its own" not in low:
        problems.append("[bar closes self-service] the ruling says no round may re-flip "
                        "it on its own — the sentence this one replaced existed precisely "
                        "so a round could not grant itself the flip")

    # Three parts, present and IN ORDER: the order is the ruling's (a) rail, (b) measure,
    # (c) process, and a re-labelled or reordered bar is a different bar.
    marks = {p: flat.find(p) for p in ("(a)", "(b)", "(c)")}
    if min(marks.values()) < 0:
        problems.append(f"[bar has three parts] the ruling's parts are (a), (b) and (c) "
                        f"and the block carries {sorted(marks)} — one part gone is one "
                        f"precondition gone")
    elif not marks["(a)"] < marks["(b)"] < marks["(c)"]:
        problems.append("[bar parts in order] the ruling numbered its parts for a reason: "
                        "the rail comes before the measure and the measure before the "
                        "round that may spend them")

    # Part (a): BOTH rail classes the ledger proves, by their mechanism and by the item
    # that proved each. Naming only one is a rail against one of the two real mis-moves.
    if "whose store already carries a round row before the move" not in low:
        problems.append("[rail: round row] the first rail is the #4172 class — the store "
                        "already had a round row when the steward moved, so the guard "
                        "reads rows and not the model's 'no round live' note")
    if "reason says attempts are spent / owed-check decides" not in low:
        problems.append("[rail: spent reasons] the second rail is the #4151 class — the "
                        "latest `status_moved` reason says the loop is done trying, which "
                        "is the judgment error the first rail cannot catch")
    # Cited WITHOUT the `#`: the trip record's own bytes are pinned elsewhere by a
    # mutation that deletes its single `#4172`, and a second `#4172` anywhere in the
    # block would make that pin unreddenable. The phrases below are unique to this bar.
    for who in ("item 4172", "item 4151"):
        if who not in flat:
            problems.append(f"[rail names {who}] the ruling proves each rail from a real "
                            f"mis-move, so {who} has to be in the bar beside its class")
    # And the bar says plainly that neither rail is built. A precondition written as an
    # indicative reads as a protection that stands, which is the false absence this loop
    # wrote its class rule about after exactly that failure.
    if "neither rail exists in the tree today" not in low:
        problems.append("[rail unbuilt stated] the bar must say the rails are not in the "
                        "tree: the ruling filed only #4513 and #4514, and neither builds "
                        "a guard, so a reader must not find the flip already fenced")

    # Part (b): the inadmissible measure WITH ITS REASON and the figures that proved the
    # reason, then the admissible one. A ban with no mechanism is re-litigated by the next
    # reader who has a rate to quote.
    if "`judgement.rate` is inadmissible for a re-flip" not in low:
        problems.append("[rate inadmissible] the bar has to say the agree-rate cannot "
                        "decide a RE-flip, not merely that it decided the first one")
    if "`agreement()` never re-scores an applied status" not in low:
        problems.append("[rate mechanism] why the rate is inadmissible: an applied wrong "
                        "status is never re-proposed, so it scores no fresh disagreement")
    if "cleared judged 140 / rate 0.9714" not in low:
        problems.append("[rate failure evidence] the figures the first flip was cleared "
                        "on are the counter-example: they read 140 and 0.9714")
    if "2 put-backs inside the first 3 applied moves" not in low:
        problems.append("[rate failure counted] and what those cleared figures bought: 2 "
                        "reverts inside the first 3 applied moves of 129")
    if "`runs.response_json`" not in flat or "#4381" not in flat:
        problems.append("[replay source] the admissible measure has to name what is "
                        "replayed (`runs.response_json`) and since when it exists (#4381)")
    if "through the fixed `reverted_moves`" not in low:
        problems.append("[replay reader named] the replay is said to run through the "
                        "FIXED `reverted_moves`, the same reader the stop condition runs "
                        "on — the unfixed pairing is what manufactured 4 of the 6 firings "
                        "this block counts, so naming the reader names the fix")
    if ">= 46 proposals with 0 pairs" not in flat:
        problems.append("[replay bar stated] the measure is a number: >= 46 proposals "
                        "replayed with 0 reverted pairs")
    if "accumulates forward from" not in low:
        problems.append("[replay accumulates] 46 cannot be met yet — proposals are "
                        "persisted only since #4381, so the floor accumulates forward")
    if REFLIP_WINDOW_GLOSS.search(flat):
        problems.append("[no flip-window gloss] the block attributes the 46 to the "
                        "11-16 flip window's volume, and nothing before #4381 was ever "
                        "persisted — the figure is forward-looking, not a volume the flip "
                        "supplied")

    # Part (c): the flip is its own round carrying the SAME stop condition plus the
    # live-window replay — the two things that made the first flip unreviewable if it had
    # gone quietly.
    if "carrying this same 7-day stop condition" not in low:
        problems.append("[re-flip carries the stop condition] a future flip has to carry "
                        "the same 7-day stop condition, or the alarm that closed this one "
                        "watches a flag no longer bounded by it")
    if "live-window `reverted_moves` replay" not in low:
        problems.append("[re-flip carries the replay] and the live-window replay #4409's "
                        "owed entry settled as unobservable while `apply` is false — a "
                        "zero there today is not evidence")

    # The question the ruling answered must be gone, everywhere in the block: two
    # sentences, one asking and one answering, is one reading too many.
    if "is owed-check's to rule" in whole:
        problems.append("[question replaced] the block still poses the question the "
                        "2026-11-22 ruling answered — a later round reads the open "
                        "question, not the ruling beneath it")
    return problems


def test_the_config_records_the_ruled_reflip_bar_beside_the_flip_it_governs():
    """#4513: owed-check ruled the re-flip bar on 2026-11-22, and the ruling now sits in
    `config.yaml`'s board-steward block instead of only in an owed record.

    The block used to end its trip record with 'Whether `apply` may be flipped again, and
    on what bar, is owed-check's to rule on 2 genuine mis-moves in 4 days; no round
    decides it here.' owed-check then answered it — `apply` stays false, and a future flip
    needs (a) a code-side rail against the two mis-move classes the ledger proves, (b) a
    proposal-replay measure instead of `judgement.rate`, and (c) the flip as its own round
    carrying the same 7-day stop condition. A ruling that lives only in a ledger row is
    invisible to the next reader of the flag it governs, which is this loop's own class
    rule about a verdict that is only printed. So the answer replaces the question, in the
    same comment run, in the ruling's own three parts.

    What this node does NOT do is touch the flip's own record: `apply` still parses False,
    the STOP CONDITION sentence is still the bytes #4443 pinned, the #4443 caveat anchor
    still appears exactly once, and both neighbouring predicates still report no problem —
    the item forbade bundling the re-flip with recording the bar, and forbade re-writing
    the 6-firing trip record while recording it.
    """
    text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    import yaml
    assert yaml.safe_load(text)["workers"]["sources"]["board-steward"]["apply"] is False, (
        "the ruling this node pins is that `apply` STAYS false; a tree reading True here "
        "has a round that re-flipped the flag while recording the bar it was told not to "
        "spend, and these pins would be guarding the wrong half")
    block = _board_steward_block(text)
    problems = _reflip_bar_problems(block)
    assert problems == [], problems

    # The instrument proves itself before it grades the file: a gloss pattern that cannot
    # match the sentence it was written for would green any wording.
    assert REFLIP_WINDOW_GLOSS.search(">= 46 proposals (the 11-16..11-20 volume)")
    assert REFLIP_WINDOW_GLOSS.search("the 11-16 -> 11-20 volume of the flip window")
    assert not REFLIP_WINDOW_GLOSS.search("that 46 accumulates forward from 2026-11-19, "
                                          "the date #4381 landed the persistence")
    # The neighbours this round was told not to disturb, checked from here so the diff
    # cannot pass its own node while quietly editing them.
    assert STOP_CONDITION_VERBATIM in block, (
        "the STOP CONDITION sentence is no longer the bytes #4443 pinned — recording the "
        "ruling was a comment edit beside it, never a re-write of it")
    assert sum(1 for ln in block.splitlines() if CAVEAT_4443_ANCHOR in ln) == 1, (
        "the #4443 caveat anchor is no longer unique: the re-flip bar was typed inside "
        "it, which puts this paragraph inside the caveat's own slice")
    assert _flip_block_problems(block, apply_value=False) == [], (
        "the new paragraph broke a pin on the flip and its trip record")
    assert _caveat_4443_problems(block) == [], (
        "the new paragraph broke the #4443 applied-move caveat")


def test_the_ruled_reflip_bar_pin_reddens_on_its_own_deletion():
    """A pin that cannot fail is not a pin: delete the recorded bar, or thin any single
    part of it, and the label that guards that part has to fire.

    The first mutation is the one the item asked for as the acceptance proof — the whole
    paragraph gone — and must redden a node that is green with it present. The rest are
    the ways a later round quietly reopens the question without editing the flag: drop the
    date, re-order the parts, keep one rail and lose the other, describe the rails as
    built, soften `judgement.rate` from inadmissible to merely old, un-number the replay
    floor, attribute that floor to the flip window's volume, or type the old open question
    back in beside the answer.
    """
    text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
    block = _board_steward_block(text)
    assert _reflip_bar_problems(block) == [], "the committed bar does not pass its own pins"

    lines = block.splitlines()
    head = next(i for i, ln in enumerate(lines) if REFLIP_BAR_ANCHOR in ln)
    tail = next(i for i, ln in enumerate(lines) if REFLIP_BAR_TAIL in ln)
    deleted = "\n".join(lines[:head] + lines[tail + 1:])
    assert deleted != block, "the bar's own markers bound nothing, so nothing was deleted"
    gone = " ".join(_reflip_bar_problems(deleted))
    assert "[bar present]" in gone, gone

    def labels(mutated: str) -> str:
        return " ".join(_reflip_bar_problems(mutated))

    # The date and the author, thinned into an undated standing rule.
    assert "[bar dated]" in labels(block.replace("dated 2026-11-22", "dated recently"))
    assert "[bar attributed]" in labels(
        block.replace("THE RE-FLIP BAR, RULED BY OWED-CHECK", "THE RE-FLIP BAR"))
    # The ruling read as permission rather than as a refusal.
    assert "[bar keeps the flag off]" in labels(
        block.replace("`apply` stays false", "`apply` may return to true"))
    assert "[bar closes self-service]" in labels(
        block.replace("no round may re-flip it on its own", "a round may re-flip it"))
    # The three parts, reordered: the rail moved behind the measure it is supposed to
    # gate. (`(b)` and `(a)` swap in one pass through a placeholder, so neither marker is
    # ever absent and only the ORDER is what the mutation changes.)
    swapped = block.replace("(a) A code-side rail", "(z) A code-side rail")
    swapped = swapped.replace("(b) Pre-flip evidence", "(a) Pre-flip evidence")
    swapped = swapped.replace("(z) A code-side rail", "(b) A code-side rail")
    assert swapped != block and "[bar parts in order]" in labels(swapped), labels(swapped)
    # The (c) marker dropped: the process half survives as prose and stops being a part.
    assert "[bar has three parts]" in labels(re.sub(r"\(c\) The flip lands", "The flip lands", block))
    # One rail kept, the other deleted: each class is a different mis-move.
    assert "[rail: round row]" in labels(
        block.replace("carries a round row", "carries a note"))
    assert "[rail: spent reasons]" in labels(
        block.replace("attempts are spent / owed-check decides", "attempts are mentioned"))
    assert "[rail names item 4172]" in labels(
        block.replace("item 4172, mis-bound", "one item, mis-bound"))
    assert "[rail names item 4151]" in labels(
        block.replace("(item 4151, moved", "(one item, moved"))
    # The rails described as though they were already built.
    assert "[rail unbuilt stated]" in labels(
        block.replace("Neither rail exists in the tree today", "Both rails are in place"))
    # The inadmissible measure softened, its mechanism dropped, its figures dropped.
    assert "[rate inadmissible]" in labels(
        block.replace("`judgement.rate` is INADMISSIBLE for a re-flip",
                      "`judgement.rate` is one input to a re-flip"))
    assert "[rate mechanism]" in labels(
        block.replace("`agreement()` never re-scores an applied status",
                      "the rate is a noisy signal"))
    assert "[rate failure evidence]" in labels(
        block.replace("cleared judged 140 / rate 0.9714", "earlier cleared readings"))
    assert "[rate failure counted]" in labels(
        block.replace("2 put-backs inside the first 3 applied moves", "some put-backs"))
    # The replay: its source, its reader, its number, its direction, and the false gloss.
    assert "[replay source]" in labels(block.replace("`runs.response_json`", "the ledger"))
    assert "[replay reader named]" in labels(
        block.replace("through the fixed `reverted_moves`", "through the alarm"))
    assert "[replay bar stated]" in labels(
        block.replace(">= 46 proposals with 0 pairs", ">= 46 proposals with a few pairs"))
    both = labels(block.replace("accumulates forward from 2026-11-19",
                               "is the 11-16 -> 11-20 volume"))
    assert "[replay accumulates]" in both, both
    assert "[no flip-window gloss]" in both, both
    # Part (c)'s two halves.
    assert "[re-flip carries the stop condition]" in labels(
        block.replace("carrying this same 7-day stop condition", "carrying a stop condition"))
    assert "[re-flip carries the replay]" in labels(
        block.replace("live-window `reverted_moves` replay", "later review"))
    # The old open question, typed back in beside the answer that replaced it.
    asked = block + "\n    # Whether `apply` may be flipped again, and on what bar, is\n" \
                   "    # owed-check's to rule; no round decides it here.\n"
    assert "[question replaced]" in labels(asked), labels(asked)


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
    assert (W.STEWARD_SCHEMA["properties"]["summary"]["maxLength"]
            == W.SUMMARY_GRAMMAR_MAX == 432), (
        "the schema's summary cap moved off its #4233 headroom over the parse "
        "ceiling")
    assert (W.SUMMARY_MAX, W.NOTE_MAX, W.REASON_MAX) == (400, 300, 300), (
        "the parse ceilings `parse_steward` shares with the schema moved (#2197)")
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
    generic arm (`workers/pool.py:2069`, `error_msg = f"{type(e).__name__}: {e}"`)
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


# ─────────────────────────── #3519: a reader for the disagreement ───────────
#
# The figures the reader triggers on are the live ones it was filed on, from
# `GET /api/workers/health?days=7` on 2026-10-27: judged 127, agree 101, disagree
# 26, rate 0.7953. Volume leg cleared (127 ≥ 50), agreement leg under the 0.90 the
# flip needs (#3089 asked, #3329 ruled), and no code path read the 26 ids.


def _window7(**over):
    """The published rolling-7-day judgement block, with overrides."""
    w = {"judged": 127, "agree": 101, "disagree": 26, "rate": 0.7953,
         "ticks_in_window": 279, "ticks_with_judged": 267, "truncated": False,
         "window_since": "2026-10-20T23:49:13+00:00"}
    w.update(over)
    return w


def _agree(*ids):
    """The tick's agreement block, with the keys `agreement()` actually returns."""
    return {"judged": len(ids), "agree": [], "disagree": sorted(ids),
            "no_opinion": [], "missed": [], "machine_moves": len(ids),
            "proposed": len(ids), "rate": 0.0}


def _filed(board_dir):
    """The disagreement items on the board, by tag rather than by filename."""
    return [i for i in B.all_items(("lloyd",), backlog_dir=board_dir)
            if "board-steward-disagreement" in (i.tags or ())]


#: One tick's worth of the real disagreement: the steward moved five `up_next`
#: items to `draft` for having no triage verdict in their history, and the state
#: machine wanted them left `up_next` — run `board-steward` 2026-10-27 22:56:40Z,
#: agreement 0% of 5 judged.
DIS_AGREE = [3024, 3041, 3068, 3196, 3284]
DIS_MOVES = [{"item_id": i, "status": "draft", "note": "no triage verdict in its history"}
             for i in DIS_AGREE]
DIS_EXPECTED = {i: ("up_next", "has a confirmed triage with acceptance text")
                for i in DIS_AGREE}


# ── the post-flip leg: an applied move the machine puts back (#4147) ────────
#
# `below_floor` used to have one leg, and after this round's flip that leg can only ever
# read as agreement: `agreement()` scores a `disagree` on the tick that asks, so once
# `apply: true` has written a wrong status the item sits there and is never re-proposed —
# the divergence lands in `missed` and in the `judged` denominator only and NEVER arrives
# as a fresh `disagree` (`config.yaml`'s pre-flip-only caveat). A steward whose every
# applied move the state machine silently put back would hold `rate` 1.0 and the alarm
# would have no leg left to stand on. These nodes are that leg: the pair of ledger rows
# the flip itself produces, read back.


def _post_flip_window(**over):
    """A window that CLEARS the old gate — `judged` over the floor and `rate` at or above
    0.90 — which is precisely the reading the flip leaves behind as its own.

    Built on `_window7()` so the shape stays the shipped reader's, with the figures
    `workers.dispatch_watch.judgement_aggregate` returned at 2026-11-16T00:06Z (judged 140,
    agree 136, disagree 3, rate 0.9714, 93 ticks, truncated false). The leg is exercised on
    the dict production publishes, not on a fixture invented for the node.
    """
    w = _window7()
    w.update({"judged": 140, "agree": 136, "disagree": 3, "rate": 0.9714,
              "ticks_in_window": 93, "ticks_with_judged": 93, "truncated": False,
              "window_since": "2026-11-15T00:00:00+00:00"})
    w.update(over)
    return w


#: The item the leg pairs, and the two statuses of one real revert. #3024 is a live
#: disagreement in the same window (`judgement_aggregate`'s `disagree_ids` today reads
#: [3024, 4035]), and `up_next` → `draft` is the direction #3024's own tick ran: the
#: steward parks an item with no triage verdict at `draft`, the state machine wants it
#: `up_next`. Post-flip the steward's `draft` is WRITTEN, and the machine's revert is a
#: `status_moved` row back to `up_next` — the pair below.
REVERT_ITEM = 3024
REVERT_APPLIED = "draft"
REVERT_TO = "up_next"
REVERT_REASON = "triage verdict re-read; the steward parked an already-triaged item"


def _flip_ledger(tmp_path, *, applied=REVERT_APPLIED, moved_to=REVERT_TO,
                 order=("steward_move", "status_moved"), same_item=True,
                 revert_at="2026-11-16T01:15:00Z", apply_at="2026-11-16T01:00:00Z",
                 shape="real"):
    """A `promotions.jsonl` holding an applied steward move and the machine's revert of it.

    Every row goes through the REAL writer, `S.append_event(entry, path=…)`, with the
    payload copied from the two writers verbatim: `steward_move` from
    `workers/sources/board_steward.py:1606` (`from`/`to`/`note`/`tick`) and `status_moved`
    from `scripts/automod/backlog.py:6046` (`from`/`to`/`reason`/`needs_human`/`owed`).
    Writing through the writer is the point: `append_event` is what supplies the stamp
    (`scripts/automod/state.py:187` builds `{..., "created_at": now_iso(), **entry}`), so a
    row carries `created_at` and never `at`, and the entry's own keys win — which is how
    only the stamp is back-dated here (the machine's row 15 minutes after the apply: one
    `autocode .interval_seconds: 900` poll, the delay a real revert has). #4147's first
    round hand-rolled its rows with `at` and `to_status`, names NO writer emits, and its
    node passed while the shipped reader found nothing on the production store — #3150's
    failure, reproduced in a new module. A fixture produced by the writer cannot drift from
    the store a third time.

    `shape="invented"` writes the WRONG spellings instead, and the reader must find
    nothing: that is the negative half of the field-name proof, held by
    `test_the_pair_is_found_only_in_the_field_names_the_writers_actually_stamp`.
    """
    from scripts.automod import state as S
    stamp_field = "created_at" if shape == "real" else "at"
    rows = []
    for kind in order:
        if kind == "steward_move":
            rows.append({"event": "steward_move", "item_id": REVERT_ITEM,
                         "from": "up_next", "to": applied,
                         "note": "no triage verdict in its history", "tick": "run-3024",
                         stamp_field: apply_at})
        else:
            rows.append({"event": "status_moved",
                         "item_id": REVERT_ITEM if same_item else REVERT_ITEM + 1,
                         "from": applied, "to": moved_to,
                         "reason": REVERT_REASON, "needs_human": False, "owed": False,
                         # `to_status` is the invented spelling of `to` and the ONLY key
                         # re-spelled: renaming `from` too would drop a key the reader never
                         # claimed to read, and this node proves the pair is unreadable, not
                         # that a required key went missing.
                         **({stamp_field: revert_at} if shape == "real" else
                            {stamp_field: revert_at, "to_status": moved_to})})
    led = tmp_path / "ledger.jsonl"
    if shape == "real":
        for r in rows:
            S.append_event(r, path=led)
        return led
    # The invented shape is written RAW, because that is exactly what round 1's fixture did:
    # `append_event` CANNOT produce it — it injects `created_at` into every payload
    # (state.py:187), so a hand-rolled line is the only way to hold a ledger whose stamp is
    # named `at`. That is the store round 1 believed it was testing, and the reader must be
    # silent on it.
    led.parent.mkdir(parents=True, exist_ok=True)
    led.write_text("".join(json.dumps(r) + chr(10) for r in rows), encoding="utf-8")
    return led


def test_the_pair_is_found_only_in_the_field_names_the_writers_actually_stamp(tmp_path):
    """The schema claim, proved both directions on one ledger written two ways (#4147).

    #4147's first round got this wrong in the direction that HIDES rather than the one that
    crashes: `reverted_moves` read `at` for the stamp and `to_status` for the reverted
    status, while the stamp is `created_at` on EVERY row (`scripts/automod/state.py:187`,
    injected by `append_event`, never passed by a writer) and BOTH kinds write the status
    pair as `from`/`TO` (`workers/sources/board_steward.py:1606`;
    `scripts/automod/backlog.py:5994`, `:6046`, `:7202` — `from_status`/`to_status` appear
    on neither). The cause was reading this module's own `_event`, which does add `at` to a
    `board_steward_disagreement` row, and assuming it described the rows this leg reads.

    The witness is the store a writer really wrote. A row carrying the invented spellings
    is not evidence of a revert, so the alarm stays silent on it — and a reader that kept
    `at` as a fallback would PASS here and go blind post-flip, which is precisely the
    outcome the flip's 7-day stop condition depends on not having.
    """
    real = _flip_ledger(tmp_path / "real", shape="real")
    invented = _flip_ledger(tmp_path / "invented", shape="invented")
    rows = [json.loads(line) for line in real.read_text().splitlines() if line.strip()]
    # Both rows carry the stamp the reader reads, and neither carries the one it must not.
    # This is the assertion that reddens if `append_event` ever stops supplying
    # `created_at`, which would otherwise un-arm the leg with nothing going red.
    assert {frozenset(r) for r in rows} == {
        frozenset({"ts", "created_at", "event", "item_id", "from", "to", "note", "tick"}),
        frozenset({"ts", "created_at", "event", "item_id", "from", "to", "reason",
                   "needs_human", "owed"})}
    assert not [r for r in rows if "at" in r or "to_status" in r]

    hit, why, _pairs = D.below_floor(_post_flip_window(), real)
    assert hit is True and f"#{REVERT_ITEM}" in why, why
    miss, why2, _pairs = D.below_floor(_post_flip_window(), invented)
    assert miss is False and "at or above" in why2, why2
    assert D.reverted_moves(_post_flip_window(), invented) == []


#: The machine's own abstention echo, copied from `scripts/automod/backlog.py:5738-5740`
#: (`desired_statuses`, the #3467 handover) and from the live row the store carries for
#: #4240 at 2026-11-18T02:27:28Z. It is stamped `status_moved` — the very kind the revert
#: leg reads — while saying the opposite of a put-back, which is #4409's second false pair.
#: `test_the_abstention_echo_filter_is_read_from_the_writer_that_stamps_it` re-reads both
#: fragments out of the writer's own source, so this constant cannot drift.
ECHO_REASON = ("the board steward moved it to in_progress (`steward_move`: Board line "
               "carries round_flag=round_in_flight with round_sta); the machine abstains "
               "until this item lands, is promoted or is re-triaged")

#: A park reason that QUOTES the steward's claim without being the machine's abstention:
#: `status_pipeline`'s spent-attempt line, copied from the live #4151 row at
#: 2026-11-16T21:13:45Z. It must stay a revert, which is why the filter needs BOTH
#: fragments and not just the phrase "the board steward moved it".
MACHINE_PARK_REASON = ("its unattended attempts are spent; owed-check decides whether to "
                       "grant another (reopen) or close it as tried")


def _pair_ledger(tmp_path, rows):
    """A `promotions.jsonl` of caller-supplied rows, written through the real writer.

    Same rule `_flip_ledger` states, and for the same reason: every row goes through
    `S.append_event(entry, path=…)`, which is what supplies `created_at` while the entry's
    own keys win, so the stamps below are the live ones and the fixture cannot drift from
    the store a third time (#4147's first round hand-rolled rows; #3150 is that failure).
    """
    from scripts.automod import state as S
    led = tmp_path / "ledger.jsonl"
    led.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        S.append_event(dict(r), path=led)
    return led


def _applied(item, at, frm, to, note):
    """A `steward_move` row with the payload `board_steward.py:1606` writes."""
    return {"event": "steward_move", "item_id": item, "from": frm, "to": to,
            "note": note, "tick": "run-4409", "created_at": at}


def _machine_moved(item, at, frm, to, reason):
    """A `status_moved` row with the payload `scripts/automod/backlog.py:6046` writes."""
    return {"event": "status_moved", "item_id": item, "from": frm, "to": to,
            "reason": reason, "needs_human": False, "owed": False, "created_at": at}


def test_a_superseded_applied_move_is_not_a_revert_the_latest_one_is(tmp_path, monkeypatch):
    """#4409 clause 4: the pairing is against the item's LATEST applied move, not its first.

    The shape is #4240's live rows, at their live stamps. The steward moved it
    `draft -> up_next` at 2026-11-17T23:27:39Z and then `up_next -> in_progress` at
    2026-11-18T00:50:10Z, and at 02:27:28Z the machine moved it to `in_progress` — the
    status the steward's NEWEST row had just asked for. The shipped pairing took the FIRST
    apply (`up_next`) and scored that agreement as a put-back, so the disagreement alarm
    fired on 2026-11-18T00:50:16Z over a steward that had been right. Three applied rows
    for the window and two machine rows, one file, both directions:

    * the superseded pair must be absent, and `below_floor` on a window whose rate CLEARS
      must stay silent on it (that is the false alarm's own window, cleared leg and all);
    * a genuine pair in the same store (#4172's live rows) must still come back, so the
      silence above is the supersession rule and not a reader that found nothing;
    * pairing on the FIRST `steward_move` — `D._supersedes` replaced by the rule that
      shipped until this round — must manufacture the #4240 pair, which is what makes this
      node a witness and not a tautology.

    The reason carries no abstention marker here on purpose: clause 5's echo filter cannot
    be what silences this ledger, so the two filters stay separately proved.
    """
    win = _post_flip_window()
    led = _pair_ledger(tmp_path / "supersede", [
        _applied(4240, "2026-11-17T23:27:39Z", "draft", "up_next",
                 "draft with triage=confirmed and round_flag=none; last event is the "
                 "confirmation, so the verdict moves it (#898)."),
        _applied(4240, "2026-11-18T00:50:10Z", "up_next", "in_progress",
                 "Board line carries round_flag=round_in_flight with round_state=worktree"
                 "+gate on round SM_20261117_233457; only in_progress is proposable here."),
        _machine_moved(4240, "2026-11-18T02:27:28Z", "up_next", "in_progress",
                       "the round's worktree and gate are alive; the item is being worked"),
        # A genuine revert for a different item, in the same store: applied `up_next` while
        # a round of its own was in flight, machine back to `in_progress` 10 minutes later.
        _applied(4172, "2026-11-16T07:07:00Z", "in_progress", "up_next",
                 "Fresh red-tree item, confirmed 06:59, no round live."),
        _machine_moved(4172, "2026-11-16T07:17:31Z", "up_next", "in_progress",
                       "an automod round is in flight for it"),
    ])

    pairs = D.reverted_moves(win, led)
    assert [p["item_id"] for p in pairs] == [4172], (
        "the revert leg keyed on something other than the genuine pair: "
        f"{pairs} — #4240's machine row moved it to the status its NEWEST `steward_move` "
        "asked for, so pairing it is the false alarm the 2026-11-18T00:50:16Z firing was")
    assert pairs[0]["applied"] == "up_next" and pairs[0]["reverted_to"] == "in_progress"
    assert pairs[0]["at"] == "2026-11-16T07:17:31Z", pairs[0]

    miss, why, _pairs = D.below_floor(win, led)
    assert miss is True and "#4172" in why and "#4240" not in why, why

    # The mutation: pair on the FIRST applied move, which is what shipped until #4409.
    # Replacing the shipped rule with the old one has to bring the false pair back, or the
    # assertion above was satisfied by something other than latest-pairing.
    monkeypatch.setattr(D, "_supersedes", lambda cur, ev: False)
    first = D.reverted_moves(win, led)
    assert [p["item_id"] for p in first] == [4240, 4172], (
        "pairing on the FIRST `steward_move` did not manufacture the #4240 pair, so this "
        f"node proves nothing about which row it pairs: {first}")
    assert first[0]["applied"] == "up_next" and first[0]["reverted_to"] == "in_progress"


def test_the_machines_abstention_echo_is_not_a_revert_even_when_its_to_differs(
        tmp_path, monkeypatch):
    """#4409 clause 5: the machine's `the machine abstains` row is agreement, not a put-back.

    `desired_statuses` stamps its #3467 abstention into a `status_moved` row, so the kind
    the leg reads for a reversal is also the kind the machine writes when the steward WON.
    Latest-pairing alone does not save the leg here: this ledger's echo carries
    `to: up_next` while the item's latest applied move is `in_progress`, exactly the case a
    steward that moved an item twice inside one 900-second poll produces. Three directions,
    one file:

    * the echo is not a pair, and the alarm stays silent on a window whose rate clears;
    * deleting the filter (`ABSTENTION_ECHO_MARKS` emptied) manufactures the pair, so the
      silence is the filter and not the fixture's shape;
    * a reason that QUOTES the steward without being the machine's own abstention — the
      spent-attempt park line, which contains "the board steward moved it" in ordinary
      prose about a claim — is still a revert. Only the abstention half is the machine's
      first-person sentence, and this is the direction that keeps the filter from eating
      real findings.
    """
    win = _post_flip_window()
    rows = [
        _applied(4240, "2026-11-17T23:27:39Z", "draft", "up_next",
                 "draft with triage=confirmed and round_flag=none."),
        _applied(4240, "2026-11-18T00:50:10Z", "up_next", "in_progress",
                 "round_flag=round_in_flight with round_state=worktree+gate; only "
                 "in_progress is proposable here."),
        _machine_moved(4240, "2026-11-18T03:00:00Z", "in_progress", "up_next", ECHO_REASON),
    ]
    led = _pair_ledger(tmp_path / "echo", rows)
    assert D.reverted_moves(win, led) == []
    miss, why, _pairs = D.below_floor(win, led)
    assert miss is False and "at or above" in why, (
        "the abstention echo fired the disagreement alarm: " + why)

    # Deleting the filter reddens the whole direction, on the same file. An EMPTY tuple
    # would invert `all()` and filter every row, so the mutation is a mark no row can
    # carry: the filter stays in place and matches nothing, which is what removing it does.
    monkeypatch.setattr(D, "ABSTENTION_ECHO_MARKS", ("no writer stamps this phrase",))
    pairs = D.reverted_moves(win, led)
    assert [p["item_id"] for p in pairs] == [4240], (
        f"with the abstention filter removed the echo still is not a pair: {pairs} — the "
        "node's silence would then be coming from somewhere else")
    assert pairs[0]["applied"] == "in_progress" and pairs[0]["reverted_to"] == "up_next"

    # A park reason that mentions the steward in prose is a real revert. `to` differs from
    # the latest applied status and the machine spoke for itself, so the filter must not
    # touch it — the false-negative direction, which is the one that would hide #4172.
    quoted = _pair_ledger(tmp_path / "quoted", rows[:2] + [
        _machine_moved(4240, "2026-11-18T03:00:00Z", "in_progress", "up_next",
                       "the board steward moved it once, and " + MACHINE_PARK_REASON)])
    hits = D.reverted_moves(win, quoted)
    assert [p["item_id"] for p in hits] == [4240], (
        "a machine park reason that merely mentions the steward was swallowed as an "
        f"abstention echo: {hits}")
    assert D.below_floor(win, quoted)[0] is True


def test_the_abstention_echo_filter_is_read_from_the_writer_that_stamps_it():
    """The filter's fragments live in the writer's source, not only in this module's.

    #4147's first round lost this leg because its reader used field names no writer emits,
    and its node passed anyway: the failure was invisible because both sides of the seam
    were invented together. This node closes the same seam for the new filter — it reads
    `desired_statuses`' own source and requires BOTH fragments `ABSTENTION_ECHO_MARKS`
    matches on to appear there, so a writer that re-words its abstention sentence reddens
    here rather than re-arming the false pair silently. It also requires the filter to be
    more than one fragment: a filter that matched the phrase "the board steward moved it"
    alone would match a park reason quoting the steward's claim, and that is how a fix for
    a false positive becomes a false negative.
    """
    import inspect

    marks = D.ABSTENTION_ECHO_MARKS
    assert len(marks) >= 2, (
        f"{marks}: one fragment is not a filter — the phrase 'the board steward moved it' "
        "appears in ordinary prose about a claim, and matching on it alone eats genuine "
        "reverts")
    src = inspect.getsource(B.desired_statuses)
    for mark in marks:
        assert mark in src, (
            f"`ABSTENTION_ECHO_MARKS` matches on {mark!r}, which no longer appears in "
            "`desired_statuses` — the writer re-worded its abstention sentence and the "
            "filter is now silently matching rows that are not abstentions")
    assert "status_moved" in src, (
        "`desired_statuses` no longer stamps its abstention as a `status_moved` row, so "
        "`EVENT_KIND_REVERT` is the wrong kind for this filter and both sides need re-read")



#: The shipped config, read as text so the YAML leg and the comment leg come from one file.
CFG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"


def test_the_stop_condition_trip_is_recorded_beside_the_flag_that_reverted():
    """#4409 clause 1's own witness: the flag parses False, and the stop condition's block
    records the trip in the condition's own words.

    One node per clause, so a reader (or a gate) can answer "did the record land?" without
    reading the shared predicate. Everything asserted here is also a label inside
    `_flip_block_problems`; what this node adds is that the labels are read against the
    parsed YAML value beside the prose, so the record and the flag cannot disagree.
    """
    text = CFG_YAML.read_text(encoding="utf-8")
    import yaml
    apply_value = yaml.safe_load(text)["workers"]["sources"]["board-steward"]["apply"]
    labels = " ".join(_flip_block_problems(_board_steward_block(text),
                                           apply_value=apply_value))
    assert "[trip recorded]" not in labels, labels
    assert "[trip counts the firings]" not in labels, labels
    assert "[trip bounds its window]" not in labels, labels
    assert "[stop condition moves the flag]" not in labels, labels
    assert apply_value is False, (
        "the record says the condition fired and a round set the flag back; the parsed "
        "value beside it has to agree, which is the YAML leg #4443 pinned")


def test_the_trip_record_names_both_genuine_reverts_and_both_mis_paired_items():
    """#4409 clause 2's own witness: the record names the reason for the flag.

    Both directions of the count, each with the row that settles it: #4172 and its
    `in_progress` stamp of 2026-11-16T07:17:31Z, #4151 and its `draft` stamp of
    2026-11-17T03:47:29Z, the phrase that says only 2 of the 6 firings rode a real revert,
    and the two items the shipped pairing mis-paired, #4240 and #4244, named by id. An item
    the record does not name cannot be looked up in the store, and a count without them
    reads as six reverts.
    """
    block = _board_steward_block(CFG_YAML.read_text(encoding="utf-8"))
    flat = " ".join(block.split())
    for stamp, who in (("2026-11-16T07:17:31Z", "#4172"),
                       ("2026-11-17T03:47:29Z", "#4151")):
        assert who in block and stamp in block, (who, stamp)
    assert re.search(r"(?:2|two)\s+(?:#\s*)?of\s+(?:#\s*)?(?:the\s+)?6\s+"
                     r"(?:firings|fires)", flat), flat
    assert "pairs were genuine" in flat, flat
    assert "mis-paired" in flat.lower(), flat
    for mispaired in ("#4240", "#4244"):
        assert mispaired in block, mispaired


def test_no_shipped_test_still_asserts_the_board_steward_flag_is_on():
    """#4409 clause 3's other half: a NAME is a claim, and two of them said the flag ships
    on while the body underneath them was being changed to False.

    Scans every test file for a node that pins THIS source's parsed `apply` key to True
    (the subscript-then-`is True` form, built below by concatenation so this node cannot
    match its own prose) or whose NAME asserts the flag ships on. `tests/test_autocode_off_loop.py` asserts
    `cfg["apply"] is True` for the autocode source and must keep doing so, which is why the
    scan is scoped to nodes naming this source rather than to the whole repo. The two nodes
    this round renamed are the ones the scan exists to keep out.
    """
    here_dir = Path(__file__).resolve().parent
    tracked = subprocess.run(["git", "-C", str(here_dir.parent), "ls-files", "tests"],
                             capture_output=True, text=True).stdout.split()
    names = sorted(Path(n).name for n in tracked) or sorted(
        q.name for q in here_dir.glob("test_*.py"))
    offenders = []
    for fname in names:
        path = here_dir / fname
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        for whole, name in re.findall(r"(def (test_[a-z0-9_]+)\([\s\S]*?)(?=\ndef |\Z)", src):
            if name == "test_no_shipped_test_still_asserts_the_board_steward_flag_is_on":
                continue              # this node: it quotes the form it hunts for
            if "board-steward" not in whole and "board_steward" not in name:
                continue
            # Only ASSERT lines: a node's docstring may QUOTE the retired claim to explain
            # why it is gone, and prose is not an assertion.
            asserts = " ".join(l.strip() for l in whole.splitlines()
                               if l.strip().startswith("assert"))
            if ('apply"] is True' in asserts or "apply'] is True" in asserts) \
                    or "ships_the_flag_on" in name or name.endswith("_state_applies"):
                offenders.append(f"{path.name}::{name}")
    assert offenders == [], (
        "a node still asserts board-steward's apply is True; the stop condition fired on "
        f"2026-11-16T18:54:53Z and #4409 reverted the flag: {offenders}")
    here = Path(__file__).read_text(encoding="utf-8")
    for renamed in ("def test_the_shipped_state_does_not_apply_after_the_stop_condition",
                    "def test_the_config_ships_the_flag_off_and_states_the_pre_flip"):
        assert renamed in here, renamed
    # Only DEF statements: this file's own docstrings name the retired nodes to explain
    # why they are gone, and a substring test over the whole file would read that as the
    # nodes still being there.
    defs = set(re.findall(r"^def (test_[a-z0-9_]+)\(", here, re.M))
    assert "test_the_shipped_state_applies" not in defs
    assert "test_the_config_ships_the_flag_on_and_states_the_cleared_reading" not in defs


def test_a_reverted_applied_move_fires_the_alarm_on_a_window_the_rate_clears(tmp_path):
    """Clause 3 of #4147 (and the purpose witness #4409 clause 5 requires to stay green):
    rate at or above 0.90, and a reverted applied move fires.

    This is the whole point of the leg. Post-flip the rate can only drift DOWN and never
    scores a fresh `disagree`, so without this second leg a steward that is being
    systematically reverted reports a rate that only decays, never trips. The window here
    is the shape production publishes today (`judged` 140, `rate` 0.9714), so the rate's
    own floor passes and the ONLY thing that can fire the alarm is the ledger pair.
    """
    led = _flip_ledger(tmp_path / "fire")
    hit, why, _pairs = D.below_floor(_post_flip_window(), led)
    assert hit is True, why
    assert f"#{REVERT_ITEM}" in why and REVERT_TO in why and REVERT_APPLIED in why, why
    # The moment and the machine's own reason are in the string. The stop condition counts
    # days from the flip, so a reason that cannot say WHEN the machine reverted, or WHY, is
    # unusable by the round that has to reverse the flag.
    assert "2026-11-16T01:15:00Z" in why, why
    assert REVERT_REASON in why, why
    assert D.reverted_moves(_post_flip_window(), led) == [
        {"item_id": REVERT_ITEM, "applied": REVERT_APPLIED, "reverted_to": REVERT_TO,
         "at": "2026-11-16T01:15:00Z", "reason": REVERT_REASON}]


def test_the_same_post_flip_window_with_no_reverted_move_does_not_fire(tmp_path):
    """Clause 4, first half: the same window with no reverted move stays silent.

    One pair of rows per ledger (`name=`), because `append_event` appends: two ledgers built
    into one file would carry the first pair into the second assertion, and the negative
    half of this clause would be measuring a revert it just wrote itself.
    """
    agree = _flip_ledger(tmp_path / "agree", moved_to=REVERT_APPLIED)
    miss, why, _pairs = D.below_floor(_post_flip_window(), agree)
    assert miss is False and "at or above" in why, why

    other = _flip_ledger(tmp_path / "other", same_item=False)
    miss, why, _pairs = D.below_floor(_post_flip_window(), other)
    assert miss is False and "at or above" in why, why

    # The leg does not reach outside the window either: a revert stamped BEFORE
    # `window_since` is history, not a live disagreement, and re-fires every tick otherwise.
    old = _flip_ledger(tmp_path / "old", revert_at="2026-11-10T01:15:00Z")
    miss, why, _pairs = D.below_floor(_post_flip_window(), old)
    assert miss is False and "at or above" in why, why

    # And an apply the machine never touched cannot fire, however the rate reads.
    quiet = _flip_ledger(tmp_path / "quiet", order=("steward_move",))
    miss, why, _pairs = D.below_floor(_post_flip_window(), quiet)
    assert miss is False and "at or above" in why, why

    # The rate's own leg is UNCHANGED by the flip: a window under the floor fires with the
    # rate's reason and no ledger at all. This is the node that keeps the flip from
    # replacing the pre-flip alarm instead of adding to it.
    under, why, _pairs = D.below_floor(_post_flip_window(rate=0.5))
    assert under is True and "under the" in why, why

    # Precedence when both would fire: the rate names the measurement, and the leg is the
    # post-flip instrument, so a reader must not have to open the ledger to learn which
    # half of the gate moved.
    both, why, _pairs = D.below_floor(_post_flip_window(rate=0.5), _flip_ledger(tmp_path / "both"))
    assert both is True and "rate 0.5 under the" in why and "reverted_moves" not in why, why


def test_a_quiet_window_still_reports_the_volume_leg_not_agreement(tmp_path):
    """Clause 4, second half: judged below the floor still reads as volume, with the leg on.

    The pre-flip ordering is kept — volume before rate before the leg — because a steward
    that has run three ticks has not been shown to be wrong, and filing disagreement off
    three ticks is what the volume leg exists to stop. With `apply: true` the volume leg
    gets MORE important, not less: the ticks are now writing to the board.
    """
    miss, why, _pairs = D.below_floor(_post_flip_window(judged=30, rate=0.5),
                                      _flip_ledger(tmp_path))
    assert miss is False and "volume leg unmet" in why and "judged 30" in why, why
    miss2, why2, _pairs = D.below_floor(_post_flip_window(truncated=True),
                                        _flip_ledger(tmp_path / "trunc"))
    assert miss2 is False and "truncated" in why2, why2


def test_a_revert_files_the_alarm_end_to_end_with_the_flag_state_on_its_body(
        board, tmp_path, monkeypatch):
    """The alarm on the path a tick actually takes, with the flip state on its body.

    The steward source ends its tick with `_file_quietly`, which resolves the window, the
    backlog dir and the ledger and then asks `file_or_update`. Calling `file_or_update`
    directly leaves `_file_quietly`'s own forwarding unexercised, and that function
    swallows EVERY exception: an `apply`/`rules_note` kwarg that never reached its filer
    would have to turn a green tick red here, or nothing at all would show it. `window_now`
    is monkeypatched to the published reading so the node measures forwarding rather than
    the scratch queue this run's conftest points at — and the patch is asserted, not
    assumed: the body has to carry these numbers.

    The ledger argument is the seam the review caught (finding 3): the trigger now reads the
    store it is handed, so a test that pointed the trigger at one file and the rows at
    another could pass with the pair never sitting in one place. Here the revert rows AND
    the filed event row are `tmp_path/"route"/ledger.jsonl`, and the last assertion reads
    the event back off that same file.
    """
    from workers.board_disagreement import _file_quietly
    monkeypatch.setattr(D, "window_now", lambda: _post_flip_window())
    led = _flip_ledger(tmp_path / "route")
    out = _file_quietly(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                        backlog_dir=board, ledger=led,
                        rules_note="apply: true since 2026-11-16T00:06Z (#4147)")
    assert out is not None and out["action"] == "filed", out
    body = _filed(board)[0].body
    assert "apply: true since 2026-11-16T00:06Z" in body, body
    # The reason is the LEG's, not the rate's: a body filed by the pre-flip trigger says the
    # agree-rate fell under the floor, this one has to say the machine reverted an applied
    # move. That distinction is what the 7-day stop condition gets read off.
    assert f"#{REVERT_ITEM}" in body and REVERT_REASON in body, body
    assert "0.9714" in body, body          # the window it was forwarded, not a re-read
    events = [json.loads(line) for line in led.read_text().splitlines() if line.strip()]
    filed = [e for e in events if e.get("event") == "board_steward_disagreement"]
    assert len(filed) == 1, filed
    # The ledger row says WHICH leg fired, as a selectable field: the 7-day stop
    # condition is read off this store later, and a reader must be able to pick out the
    # revert-leg rows without parsing prose or re-measuring an aged window.
    assert filed[0]["leg"] == "post-flip revert", filed[0]
    assert "post-flip revert" in body or "steward_move applied" in body, body


def test_the_alarm_body_states_a_dry_run_flag_when_the_tick_wrote_nothing(board, tmp_path,
                                                                         monkeypatch):
    """The body distinguishes an alarm raised while the steward was APPLYING from one
    raised while it only PROPOSED.

    #4147's owed entry asks whether the alarm fired within 7 days of the flip; the answer is
    only readable if the item records the flag it was filed under. A tick on `apply: false`
    wrote nothing, so a revert could not have happened under it and the body says so — a
    reader who took that as post-flip evidence would set the flag back to false on a dry run.
    """
    monkeypatch.setattr(D, "window_now", lambda: _post_flip_window())
    out = D._file_quietly(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                          backlog_dir=board, ledger=_flip_ledger(tmp_path / "dry"),
                          rules_note="apply: false (dry run: this tick wrote nothing)")
    assert out is not None
    body = _filed(board)[0].body
    assert "apply: false (dry run" in body, body


# ── the DIRECTION sentence: which leg fired is what the body says (#4251) ───
#
# `_create`'s header used to be ONE unconditional f-string declaring the trailing-7-day
# judgement "was measured **below** owed-check's flip floor". The revert leg is reached
# only AFTER the volume floor and the rate floor have both passed (`below_floor`, and its
# docstring's "A CLEARED window with a revert in it is the case this alarm was kept alive
# for"), so every revert-leg filing printed the opposite of its own figures. Both
# post-flip filings the flip has produced read that way — #4184 (`rate` 0.968) and #4247
# (`rate` 0.9691), each printing "below … floor" against a floor of 0.9 in the same body
# whose `board_steward_disagreement` ledger row, written by the same call, recorded
# `leg: "post-flip revert"`. The reader of that body is the reader of `config.yaml`'s
# 7-day stop condition for the apply flip, so a false direction is a false reading of the
# flip's health. These nodes pin the branch, the one derivation behind both surfaces, and
# the scope of what may NOT change.

#: The sub-floor header sentence exactly as HEAD 9156e6f9 rendered it for `_window7()`,
#: frozen as a literal rather than re-derived: clauses 2 and 5 ask that a below-floor
#: filing keep the text it has, so the thing to compare against is the old bytes, not a
#: second copy of the rule.
RATE_LEG_HEADER = (
    "Filed by `workers/board_disagreement.py` at the end of a board-steward tick whose "
    "trailing-7-day judgement was measured **below** owed-check's flip floor: "
    "`judged=127` (the leg it needs is 50) with agree `rate=0.7953` against a floor of "
    "0.9, over the window since 2026-10-20T23:49:13+00:00. The volume leg is cleared, so "
    "this is a measured disagreement and not an unmeasured one — which is the case "
    "nothing used to read."
)

#: Below-floor wording, as one constant, so a node asserting its ABSENCE is asserting the
#: absence of the exact sentence that shipped and not of a paraphrase of it.
BELOW_FLOOR_CLAIM = "measured **below** owed-check's flip floor"


def _header_para(body: str) -> str:
    """The filing's DIRECTION paragraph alone, whatever titles ride above it.

    Scoped by the next paragraph's own opening words rather than by a line count, so a
    rewrap of the header cannot move the window the assertions read through. The last block
    before that marker, rather than everything before it, because `new_item` writes its own
    H1 above the body the filer composes and the filer's body starts with one too — the
    loaded item's `body` carries the title twice, and neither title is the sentence under
    test.
    """
    head, sep, _rest = body.partition("\n\nEach row names BOTH sides")
    assert sep, f"the body's second paragraph moved; can't scope the header:\n{body[:400]}"
    return head.rsplit("\n\n", 1)[-1]


def _leg_rows(ledger):
    """The `leg` field of every disagreement row in a ledger, in the order written."""
    return [e["leg"] for e in S.read_events(path=ledger)
            if e.get("event") == D.EVENT]


def _new_board(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def test_a_revert_leg_filing_on_a_cleared_window_says_cleared_not_below_floor(board,
                                                                             tmp_path):
    """Clause 1: the direction the body states is the direction that actually fired.

    The window here is `_post_flip_window()` — `judged` 140 over the volume floor of 50 and
    `rate` 0.9714 over the agree-rate floor of 0.90 — so BOTH floors cleared, and the only
    thing that can file is the revert pair `_flip_ledger` writes. The header must therefore
    say the window cleared both floors and name the post-flip revert as what fired, and must
    not print the below-floor claim at all, anywhere in the body.
    """
    led = _flip_ledger(tmp_path / "revert-body")
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_post_flip_window(), backlog_dir=board, ledger=led)
    assert out is not None and out["action"] == "filed", out
    body = _filed(board)[0].body
    para = _header_para(body)
    assert BELOW_FLOOR_CLAIM not in para, para
    assert BELOW_FLOOR_CLAIM not in body, body
    assert "cleared both" in para, para
    assert "the volume leg and the agree-rate leg" in para, (
        "the header never names WHICH floors cleared inside its own cleared clause. The "
        "word `volume` alone proves nothing: `The volume leg is cleared` prints on BOTH "
        f"legs, so only this naming clause constrains the revert header:\n{para}")
    assert D.LEG_REVERT in para, f"the header never names the leg that fired:\n{para}"
    assert "`judged=140`" in para and "`rate=0.9714`" in para, (
        f"the cleared reading's own figures are gone from the header:\n{para}")
    assert "against a floor of 0.9" in para, (
        "the floor it cleared is not stated. Bare `0.9` would be satisfied by the "
        f"`rate` figure itself, so it proves nothing:\n{para}")


def test_a_sub_floor_filing_keeps_the_measured_below_floor_header(board, tmp_path):
    """Clause 2 and clause 5's create half: the rate leg's text is not touched.

    `_window7()` is the 2026-10-27 published reading — `judged` 127 (volume cleared),
    `rate` 0.7953 (under the 0.90 the flip needs) — which is the filing the below-floor
    wording describes. The whole direction paragraph is compared for equality against the
    bytes HEAD 9156e6f9 wrote, so "unchanged in substance" is checked as "unchanged", and a
    branch that re-worded the rate leg to share the revert leg's sentence fails here.
    """
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED, window=_window7(),
                           backlog_dir=board, ledger=tmp_path / "rate-body.jsonl")
    assert out is not None and out["action"] == "filed", out
    para = _header_para(_filed(board)[0].body)
    assert para == RATE_LEG_HEADER, (
        "the sub-floor header moved: clause 2 asks that this filing keep today's wording "
        f"and its figures.\n--- filed ---\n{para}\n--- frozen ---\n{RATE_LEG_HEADER}")
    assert BELOW_FLOOR_CLAIM in para
    assert "`judged=127`" in para and "`rate=0.7953`" in para and "floor of 0.9" in para


def test_the_bodies_direction_and_the_ledger_leg_are_one_derivation(board, tmp_path,
                                                                    monkeypatch):
    """Clause 3: the body's wording and the ledger row's `leg` cannot contradict.

    Three parts, because "one derivation" is a claim about code shape and only a patch can
    show it. FIRST the key itself: the derivation reads the `below_floor` trigger string, so
    the sentence `below_floor` emits has to be the one the prefix selects, in both
    directions — a re-worded revert reason that stopped starting with the prefix would
    silently re-label every revert filing as a rate filing. SECOND, one real filing per leg,
    read back from BOTH stores that call writes: the backlog file and the ledger row, with
    the two direction surfaces compared. THIRD, `leg_from_trigger` is patched to answer for
    a sub-floor window, and both surfaces move together — which is the only way to show a
    reader that neither one keeps its own copy of the rule.
    """
    hit, revert_reason, _pairs = D.below_floor(_post_flip_window(), _flip_ledger(tmp_path / "k1"))
    assert hit is True and revert_reason.startswith(D.REVERT_TRIGGER_PREFIX), revert_reason
    hit2, rate_reason, _pairs = D.below_floor(_post_flip_window(rate=0.5))
    assert hit2 is True and not rate_reason.startswith(D.REVERT_TRIGGER_PREFIX), rate_reason
    assert D.leg_from_trigger(revert_reason) == D.LEG_REVERT
    assert D.leg_from_trigger(rate_reason) == D.LEG_RATE
    assert D.leg_from_trigger("") == D.LEG_RATE, "an absent trigger is not a revert"

    rev_board, rev_led = _new_board(tmp_path, "b-revert"), _flip_ledger(tmp_path / "l-rev")
    filed_r = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                               window=_post_flip_window(), backlog_dir=rev_board,
                               ledger=rev_led)
    assert filed_r is not None and filed_r["action"] == "filed", filed_r
    rate_board, rate_led = _new_board(tmp_path, "b-rate"), tmp_path / "l-rate.jsonl"
    filed_v = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                               window=_window7(), backlog_dir=rate_board, ledger=rate_led)
    assert filed_v is not None and filed_v["action"] == "filed", filed_v

    rev_leg, rate_leg = _leg_rows(rev_led)[0], _leg_rows(rate_led)[0]
    rev_body, rate_body = _filed(rev_board)[0].body, _filed(rate_board)[0].body
    assert rev_leg == D.LEG_REVERT and rate_leg == D.LEG_RATE, (rev_leg, rate_leg)
    # The invariant clause 3 asks for: no filing whose body says below while its row says
    # revert, or the reverse. Stated once per surface pair rather than as two separate
    # "not in" asserts, so a future leg name cannot satisfy one and break the other.
    for leg, body in ((rev_leg, rev_body), (rate_leg, rate_body)):
        assert (leg == D.LEG_REVERT) == (BELOW_FLOOR_CLAIM not in body), (
            f"a filing whose ledger leg is {leg!r} carries the below-floor claim in its "
            f"body: the two direction surfaces are derived twice\n{body[:400]}")

    monkeypatch.setattr(D, "leg_from_trigger", lambda trigger: D.LEG_REVERT)
    patched_board, patched_led = _new_board(tmp_path, "b-patched"), tmp_path / "l-p.jsonl"
    filed_p = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                               window=_window7(), backlog_dir=patched_board,
                               ledger=patched_led)
    assert filed_p is not None and filed_p["action"] == "filed", filed_p
    patched_body = _filed(patched_board)[0].body
    assert _leg_rows(patched_led)[0] == D.LEG_REVERT, (
        "patching the one derivation did not move the ledger row, so `_event` derives the "
        "leg somewhere else")
    assert BELOW_FLOOR_CLAIM not in patched_body and "cleared both" in _header_para(
        patched_body), (
        "patching the one derivation did not move the body, so `_create` words its header "
        "from somewhere else")


def test_a_widening_names_the_leg_that_widened_it_and_keeps_its_figures(board, tmp_path):
    """Clause 4, and clause 5's merge half: `### Added` says which leg widened it.

    `_merge` already forwards `trigger` to `_event`, so the ledger row for a widening always
    recorded its leg while the appended text named nothing — a reader of a merged body saw
    the ORIGINAL create's leg and could not tell a revert-leg widening from a rate-leg one.
    One item is widened twice, once per leg, so the two headings sit in one file and have to
    be distinguishable from each other. The heading's `judged`/`rate` figures are matched
    against the exact prefix today's bytes produce: clause 5 says only the leg text is
    added, the numbers do not move.
    """
    led = _flip_ledger(tmp_path / "merge-leg")
    first = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                             window=_window7(), backlog_dir=board, ledger=led)
    assert first is not None and first["action"] == "filed", first
    path = _filed(board)[0].path
    assert _leg_rows(led)[-1] == D.LEG_RATE, "the create was not a rate-leg filing"

    moves2 = DIS_MOVES + [{"item_id": 3119, "status": "draft",
                           "note": "the vec0 index needs a hand-side copy rebuild"}]
    want2 = dict(DIS_EXPECTED)
    want2[3119] = ("up_next", "in_progress by no round, ready by its acceptance")
    second = D.file_or_update(_agree(3119), moves2, want2, window=_post_flip_window(),
                              backlog_dir=board, ledger=led)
    assert second is not None and second["action"] == "merged", second

    moves3 = DIS_MOVES + [{"item_id": 3201, "status": "draft",
                           "note": "no triage verdict in its history"}]
    want3 = dict(DIS_EXPECTED)
    want3[3201] = ("up_next", "has a confirmed triage with acceptance text")
    third = D.file_or_update(_agree(3201), moves3, want3, window=_window7(rate=0.70),
                             backlog_dir=board, ledger=led)
    assert third is not None and third["action"] == "merged", third

    body = path.read_text(encoding="utf-8")
    added = [ln for ln in body.splitlines() if ln.startswith("### Added")]
    assert len(added) == 2, f"expected one heading per widening:\n{added}"
    assert D.LEG_REVERT in added[0], (
        f"a revert-leg widening names nothing but the original create's leg:\n{added[0]}")
    assert D.LEG_RATE in added[1], f"the rate-leg widening is unlabelled:\n{added[1]}"
    assert re.match(r"^### Added \S+ \(window `judged=140`, `rate=0\.9714`\)", added[0]), (
        f"the widening's figures stopped printing as they printed today:\n{added[0]}")
    assert re.match(r"^### Added \S+ \(window `judged=127`, `rate=0\.7`\)", added[1]), (
        f"the widening's figures stopped printing as they printed today:\n{added[1]}")
    assert _leg_rows(led) == [D.LEG_RATE, D.LEG_REVERT, D.LEG_RATE], _leg_rows(led)


def test_only_the_direction_phrase_moves_between_the_two_legs_bodies(tmp_path):
    """Clause 5: with the window held fixed, the ONLY difference is the direction.

    Both bodies are `_create`'s own output for the same window, the same rows and the same
    stamp, differing in nothing but the trigger sentence — and both triggers are the real
    `below_floor` output, not invented text. Both bodies come back through the real reader,
    so what is compared is the file a person opens. The direction paragraph and the
    `**What tripped it**` line (which necessarily differs: it quotes the trigger) are cut
    out, and everything else has to be byte-identical — the check that the `## Coverage`
    window line, the both-sides prose, the `What to decide` section and the flag-state line
    were not rewritten along the way. The cleared-versus-below wording is the only thing
    clause 5 lets move.
    """
    w = _window7()
    _hit, revert_trigger, _pairs = D.below_floor(_post_flip_window(), _flip_ledger(tmp_path / "sk"))
    _hit2, rate_trigger, _pairs = D.below_floor(w)
    assert revert_trigger.startswith(D.REVERT_TRIGGER_PREFIX), revert_trigger
    assert not rate_trigger.startswith(D.REVERT_TRIGGER_PREFIX), rate_trigger

    def create(name, trigger):
        d = _new_board(tmp_path, name)
        out = D._create(B, DIS_AGREE, DIS_MOVES, DIS_EXPECTED, w,
                        "2026-11-18T00:00:00+00:00", backlog_dir=d,
                        ledger=tmp_path / f"{name}.jsonl", trigger=trigger)
        assert out["action"] == "filed", out
        return _filed(d)[0].body

    rate_body, rev_body = create("b-rate-only", rate_trigger), create("b-rev-only",
                                                                      revert_trigger)
    assert BELOW_FLOOR_CLAIM in rate_body and BELOW_FLOOR_CLAIM not in rev_body

    def skeleton(text):
        """The body with its direction paragraph and its trigger line cut out."""
        head, sep, rest = text.partition("\n\nEach row names BOTH sides")
        assert sep, f"the body's second paragraph moved:\n{text[:400]}"
        before, sep2, _tail = rest.partition("**What tripped it**")
        assert sep2, f"no `What tripped it` line to cut the body at:\n{text[:400]}"
        return head.rsplit("\n\n", 1)[0] + before

    if skeleton(rate_body) != skeleton(rev_body):
        diff = "\n".join(list(difflib.unified_diff(skeleton(rate_body).splitlines(),
                                                   skeleton(rev_body).splitlines(),
                                                   "rate-leg", "revert-leg", lineterm=""))
                         [:60])
        raise AssertionError(
            "the two filings differ somewhere outside the direction sentence and the "
            f"trigger line:\n{diff}")

    window_line = (f"- the window itself: {w['judged']} judged, `rate={w['rate']}` "
                   f"against owed-check's {D.FLIP_MIN_RATE} floor, since "
                   f"{w['window_since']}.")
    assert window_line in rate_body and window_line in rev_body, (
        "the `## Coverage` window line stopped printing the same figures on one leg")
    cleared = ("The volume leg is cleared, so this is a measured disagreement and not an "
               "unmeasured one")
    assert cleared in rate_body and cleared in rev_body, (
        "the volume clause is true on both legs and had to survive on both")
    tripped_r = [ln for ln in rate_body.splitlines() if ln.startswith("**What tripped it**")]
    tripped_v = [ln for ln in rev_body.splitlines() if ln.startswith("**What tripped it**")]
    assert len(tripped_r) == len(tripped_v) == 1, (tripped_r, tripped_v)
    assert tripped_r[0] != tripped_v[0]
    assert rate_trigger in tripped_r[0] and revert_trigger in tripped_v[0]


def test_the_post_flip_leg_silences_itself_on_a_missing_ledger_or_window(tmp_path):
    """A missing ledger, a missing window, and an unreadable one all keep the alarm quiet.

    The filing is allowed to be silent and never allowed to be wrong: firing disagreement
    because a store was briefly absent would set the stop condition off on an outage, and the
    stop condition's whole consequence is turning the steward off. `window_since` is what the
    filter needs, so a window without one is refused before any read.
    """
    missing = tmp_path / "nope.jsonl"
    assert D.reverted_moves(_post_flip_window(), missing) == []
    assert D.below_floor(_post_flip_window(), missing)[0] is False
    assert D.reverted_moves(_post_flip_window(window_since=""),
                            _flip_ledger(tmp_path / "ws")) == []
    # A ledger with no parseable row at all is a missing ledger (`read_events` skips what
    # it cannot decode, and returns [] for the whole file on an OSError).
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "ledger.jsonl").write_text("not json\n{also bad\n")
    assert D.reverted_moves(_post_flip_window(), junk / "ledger.jsonl") == []


def test_below_floor_needs_both_legs_of_the_flip_bar():
    """The gate is owed-check's bar restated, not a new number: volume AND
    agreement, so a thin or unmeasured window cannot file and a well-measured
    passing one cannot either. `truncated` means the aggregate hit its scan cap,
    which is the same 'not yet evidence' state #1688 stopped summarising as 100%.
    """
    from workers.board_disagreement import below_floor

    assert below_floor(_window7())[0] is True
    assert below_floor(_window7(rate=0.90))[0] is False, "0.90 is the floor, not under it"
    assert below_floor(_window7(rate=0.95))[0] is False
    assert below_floor(_window7(judged=49, rate=0.10))[0] is False, "volume leg unmet"
    assert below_floor(_window7(judged=50, rate=0.10))[0] is True, "50 judged is the leg"
    assert below_floor(_window7(rate=0.10, truncated=True))[0] is False
    assert below_floor(_window7(judged=0, rate=None))[0] is False


def test_a_measured_below_floor_tick_files_one_item_naming_both_sides(board, tmp_path):
    """Clause 1. 127 judged at rate 0.7953 is a measured fail, and the artifact
    has to carry both sides: the status the steward proposed and the status
    `desired_statuses` wants, so a reader holding only the item can rule without
    hand-running sqlite. The move is not nowhere — the tick's `board_steward` ledger
    event carries it (#3602) — but that event is the worker's own write, and the item
    is what a triage pass reads."""
    from workers.board_disagreement import file_or_update

    out = file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                         window=_window7(), backlog_dir=board,
                         ledger=tmp_path / "ledger.jsonl")
    assert out is not None and out["action"] == "filed"
    items = _filed(board)
    assert len(items) == 1, f"expected exactly one item, got {[i.id for i in items]}"
    it = items[0]
    assert it.status == "draft" and it.board == "lloyd"
    for i in DIS_AGREE:
        assert f"#{i}" in it.body, f"#{i} missing from the filed body"
    assert "up_next" in it.body and "draft" in it.body
    assert "steward proposed `draft`" in it.body
    assert "the state machine wants `up_next`" in it.body
    assert "has a confirmed triage with acceptance text" in it.body, "the machine's why"


def test_nothing_is_filed_above_the_floor_or_before_the_volume_leg(board, tmp_path):
    """Clause 1's negatives, on the same board and the same disagreement: an
    agreement that clears the bar is a steward doing its job, and 49 judged is the
    unmeasured region the item's own text warns is 'not yet evidence of anything'."""
    from workers.board_disagreement import file_or_update

    for w in (_window7(rate=0.94), _window7(judged=49, rate=0.10),
              _window7(judged=0, rate=None)):
        assert file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                             window=w, backlog_dir=board,
                             ledger=tmp_path / "ledger.jsonl") is None
    assert _filed(board) == []


def test_filing_again_adds_the_new_ids_to_the_same_item(board, tmp_path):
    """Clause 2. A second below-floor tick naming three ids it has not named
    before extends the one open item; a second item on the same subject is the
    duplicate this clause exists to prevent, and the 500-char summary cap (#1606)
    means the item IS the record."""
    from workers.board_disagreement import file_or_update

    first = file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(), backlog_dir=board,
                           ledger=tmp_path / "ledger.jsonl")
    assert first["action"] == "filed"
    more = DIS_AGREE + [3119]
    moves = DIS_MOVES + [{"item_id": 3119, "status": "draft",
                          "note": "the vec0 index needs a hand-side copy rebuild"}]
    want = dict(DIS_EXPECTED)
    want[3119] = ("up_next", "in_progress by no round, ready by its acceptance")
    second = file_or_update(_agree(3119), moves, want, window=_window7(),
                            backlog_dir=board, ledger=tmp_path / "ledger.jsonl")
    assert second["action"] == "merged"
    items = _filed(board)
    assert len(items) == 1, f"the second tick filed a second item: {[i.id for i in items]}"
    assert items[0].id == first["item_id"]
    for i in more:
        assert f"#{i}" in items[0].body


def test_an_item_a_round_is_working_is_left_unrewritten(board, tmp_path):
    """Clause 2's other half: the same rule `file_red_tree_item` keeps for a
    round holding a `blocked: red-tree` item. A round working the disagreement
    item already IS the reader this window wanted."""
    from workers import board_disagreement as D

    first = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                             window=_window7(), backlog_dir=board,
                             ledger=tmp_path / "ledger.jsonl")
    it = _filed(board)[0]
    B.update_frontmatter(it.path, {"status": "in_progress"})
    before = it.path.read_bytes()
    assert D.file_or_update(_agree(3119), DIS_MOVES, DIS_EXPECTED, window=_window7(),
                            backlog_dir=board,
                            ledger=tmp_path / "ledger.jsonl") is None
    assert it.path.read_bytes() == before, "the in_progress item was rewritten"
    assert len(_filed(board)) == 1
    assert first["action"] == "filed"


def test_each_file_or_merge_appends_one_ledger_event_with_the_window(tmp_path):
    """Clause 3. The run summary is capped at 500 characters and 77 of the 154
    board-steward rows already stop at exactly 500, so the counts cannot live
    there; one row per write carries that window's judged, disagree and rate."""
    from workers.board_disagreement import file_or_update

    led = tmp_path / "ledger.jsonl"
    board = tmp_path / "backlog"
    board.mkdir()
    file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED, window=_window7(),
                   backlog_dir=board, ledger=led)
    file_or_update(_agree(3119), DIS_MOVES, DIS_EXPECTED, window=_window7(rate=0.70),
                   backlog_dir=board, ledger=led)
    rows = [e for e in S.read_events(path=led)
            if e.get("event") == "board_steward_disagreement"]
    assert len(rows) == 2, rows
    assert [r["action"] for r in rows] == ["filed", "merged"]
    assert [r["judged"] for r in rows] == [127, 127]
    assert [r["disagree"] for r in rows] == [26, 26]
    assert [r["rate"] for r in rows] == [0.7953, 0.70]
    assert rows[0]["item_id"] == rows[1]["item_id"]
    assert rows[0]["tick_item_ids"] == DIS_AGREE
    assert rows[1]["new_item_ids"] == [3119]


# ── the revert pairs ON THE ROW: which item the leg keyed on (#4514) ────────
#
# A `board_steward_disagreement` row carries `leg`, `judged`, `rate` and `window_since`, but
# never the pair the revert leg keyed on. `item_id` is the disagreement item being filed and
# `tick_item_ids` are the tick's disagree ids, and for each of the six `leg: "post-flip
# revert"` firings of 2026-11-16..20 (#4184, #4247, #4251, #4377, #4397, #4405) the ids the
# leg actually keyed on intersect NEITHER. Reading them back means replaying `reverted_moves`
# over the store — and the replay is unfaithful, not merely extra work: `reverted_moves` reads
# the whole file (`workers/board_disagreement.py` `S.read_events(ledger, 0)`) and bounds it
# only BELOW by `window_since`, so replaying the 2026-11-16 firing today also returns a pair
# stamped six days later. Only a row written at the firing's own instant is a record. These
# nodes pin `revert_pairs`, the field that makes an item-by-item audit one read of the row.

#: The two genuine reverts inside the post-flip window, copied from the live rows of
#: `~/.local/state/lloyd-automod/promotions.jsonl` — both are named in `config.yaml`'s
#: stop-condition record. #4172: applied `up_next` at 2026-11-16T07:07:00Z, put back to
#: `in_progress` ten minutes later. #4151: applied `up_next` at 2026-11-17T00:59:08Z, parked
#: to `draft` at 03:47:29Z. The order below is the store's first-appearance order, which is
#: the order `reverted_moves` returns and the order `below_floor`'s reason reads its one
#: named pair from (#4381) — the row carries both, the trigger names the first.
EXPECTED_REVERT_PAIRS = [
    {"item_id": 4172, "applied": "up_next", "reverted_to": "in_progress",
     "at": "2026-11-16T07:17:31Z"},
    {"item_id": 4151, "applied": "up_next", "reverted_to": "draft",
     "at": "2026-11-17T03:47:29Z"},
]

#: #4172's put-back row verbatim — the text the trigger sentence quotes and clause 4 asserts.
IN_FLIGHT_REASON = "an automod round is in flight for it"

#: The nine non-clock fields of the rate-leg row as they shipped before #4514. Every field
#: the filer can know before it writes: `item_id` comes from the fixture and is added at the
#: assert, and the three stamps `_row_fields` drops move with the clock. Clause 2's "every
#: other key and value exactly as today" is one comparison against this, not nine assertions.
RATE_LEG_ROW_FIELDS = {
    "event": D.EVENT, "action": "filed", "judged": 127, "disagree": 26, "rate": 0.7953,
    "window_since": "2026-10-20T23:49:13+00:00", "tick_item_ids": DIS_AGREE,
    "new_item_ids": DIS_AGREE, "leg": D.LEG_RATE,
}


def _four_keys(pairs):
    """`reverted_moves`' own pairs, projected to the four keys the row stores.

    The reader also returns the machine's `reason`, which `_event` deliberately does not copy
    onto the row (the trigger sentence already quotes it), so comparing the reader's raw
    output to the row's list is a category error. This is the same projection `_event`
    applies, which is what makes a row-level equality below a statement about the reader too.
    """
    return [{k: p.get(k) for k in ("item_id", "applied", "reverted_to", "at")} for p in pairs]


def _two_revert_ledger(tmp_path, name):
    """A `promotions.jsonl` holding a genuine revert for TWO different items.

    Built by `_pair_ledger` from the same real-payload rows `_applied` and `_machine_moved`
    write, so the stamps are the live ones and the field names are the ones the two writers
    emit (#4147's first round invented `at`/`to_status`, and its node passed while the shipped
    reader found nothing on the production store). #4151 also carries its earlier spent-park
    `status_moved` of 2026-11-16T21:13:45Z, which is stamped BEFORE that item's latest
    `steward_move`: the pairing rule must skip it, so the row's list is the reader's output
    and not a constant this file hand-wrote.
    """
    return _pair_ledger(tmp_path / name, [
        _applied(4172, "2026-11-16T07:07:00Z", "in_progress", "up_next",
                 "Fresh red-tree item, confirmed 06:59, no round live."),
        _machine_moved(4172, "2026-11-16T07:17:31Z", "up_next", "in_progress",
                       IN_FLIGHT_REASON),
        _machine_moved(4151, "2026-11-16T21:13:45Z", "in_progress", "draft",
                       MACHINE_PARK_REASON),
        _applied(4151, "2026-11-17T00:59:08Z", "draft", "up_next",
                 "the spent-park of 11-16T21:13 was answered by a reopened row granting "
                 "another attempt; the line still carries its triage verdict"),
        _machine_moved(4151, "2026-11-17T03:47:29Z", "in_progress", "draft",
                       "landed with a clause not met and its one unattended attempt is "
                       "spent; owed-check decides (reopen_item to grant another)"),
    ])


def _row_fields(row):
    """A disagreement row minus the three stamps that differ on every run.

    `ts` and `created_at` are prepended by `append_event`
    (`scripts/automod/state.py:181` builds `{"ts": …, "created_at": now_iso(), **entry}`) and
    `at` is this filer's own `_stamp()`, so all three move with the wall clock. Every OTHER
    key is a value a test can pin, which is what makes a whole-dict comparison the instrument
    for "nothing else about this row changed".
    """
    return {k: v for k, v in row.items() if k not in ("ts", "created_at", "at")}


def _pair_rows(ledger):
    """The `revert_pairs` field of every disagreement row in a ledger, in the order written.

    Subscripted, not `.get`: a row that lost the key has to raise. That is what makes
    deleting the key from `_event` redden a node on BOTH filing paths rather than leaving the
    merge path mute, which is the gap #4251 left open for the fields it did not pin.
    """
    return [e["revert_pairs"] for e in S.read_events(path=ledger)
            if e.get("event") == D.EVENT]


def test_a_revert_leg_row_carries_every_pair_the_leg_keyed_on(tmp_path):
    """Clause 1: the row names every pair `reverted_moves` returned, item by item.

    Driven end to end through `file_or_update` on `_post_flip_window()` — both floors cleared,
    so the revert leg is the only thing that can file — against a ledger whose reverts span TWO
    item ids. The row's `revert_pairs` has to hold BOTH, and the comparison is whole-dict
    equality on each entry: `item_id`, `applied`, `reverted_to` and `at` exactly, so a missing
    key reddens it as surely as a fifth one (`reverted_moves` also returns the machine's own
    `reason`, and the body already quotes that verbatim in the trigger sentence — a row that
    copied it again would be storing one sentence in two places of a store this module is
    asked to keep readable).

    The two assertions under that comparison are the premise the field exists to close. The
    keyed ids appear in no other field of the row, so before this field the item the leg
    keyed on was unreadable from it; and the trigger names only the FIRST pair, which is
    #4381's asymmetry — the reason the row carries the whole list rather than one item.
    """
    board = _new_board(tmp_path, "b-pairs")
    led = _two_revert_ledger(tmp_path, "two-reverts")
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_post_flip_window(), backlog_dir=board, ledger=led)
    assert out is not None and out["action"] == "filed", out
    rows = [e for e in S.read_events(path=led) if e.get("event") == D.EVENT]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["leg"] == D.LEG_REVERT, row["leg"]
    assert row["revert_pairs"] == EXPECTED_REVERT_PAIRS, json.dumps(row, sort_keys=True)
    keyed = {p["item_id"] for p in row["revert_pairs"]}
    assert keyed == {4172, 4151}, keyed
    assert not keyed & ({row["item_id"]} | set(row["tick_item_ids"])), (
        "the premise is gone: the keyed item is reachable from another field, so the field "
        f"above is a copy rather than the only record — {row}")

    hit, why, _pairs = D.below_floor(_post_flip_window(), led)
    assert hit is True and "#4172" in why and "#4151" not in why, why
    assert "`up_next`" in why and "`in_progress`" in why, why


def test_a_rate_leg_row_carries_an_empty_revert_pairs_and_nothing_else_moves(tmp_path):
    """Clause 2: the rate leg's row gains the key and no other field of it moves.

    `_window7()` is the 2026-10-27 published reading — `rate` 0.7953 under the 0.90 floor — so
    the filing is a rate-leg one. The assertion is a whole-dict equality over the row's
    non-clock fields, which pins both directions at once: `revert_pairs` must be present and
    empty, and the other ten non-clock fields — `RATE_LEG_ROW_FIELDS`' nine plus this row's own
    `item_id` — must be exactly what they were before the key existed.

    The second half is what an empty list does NOT mean. `below_floor` returns for a sub-floor
    rate BEFORE it calls `reverted_moves`, so on a rate-leg row `[]` says this leg never
    evaluated, not that nothing was put back: the ledger beside the filing below holds two
    genuine reverts and the row still reads `[]`. `leg` is the only discriminator, and this is
    the witness for the comment beside the new key — a later reader must not SELECT on the
    empty list.
    """
    plain = tmp_path / "rate-plain.jsonl"
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED, window=_window7(),
                           backlog_dir=_new_board(tmp_path, "b-rate"), ledger=plain)
    assert out is not None and out["action"] == "filed", out
    rows = [e for e in S.read_events(path=plain) if e.get("event") == D.EVENT]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["revert_pairs"] == [], json.dumps(row, sort_keys=True)
    pinned = {**RATE_LEG_ROW_FIELDS, "item_id": row["item_id"]}
    assert _row_fields(row) == {**pinned, "revert_pairs": []}, (
        "the rate-leg row changed by more than the one new key: "
        + json.dumps(_row_fields(row), sort_keys=True))
    assert set(_row_fields(row)) - set(pinned) == {"revert_pairs"}, (
        "the field set grew by something other than `revert_pairs`")

    with_reverts = _two_revert_ledger(tmp_path, "rate-with-reverts")
    out2 = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                            window=_window7(), backlog_dir=_new_board(tmp_path, "b-rate-2"),
                            ledger=with_reverts)
    assert out2 is not None and out2["action"] == "filed", out2
    rows2 = [e for e in S.read_events(path=with_reverts) if e.get("event") == D.EVENT]
    assert len(rows2) == 1, rows2
    assert rows2[0]["leg"] == D.LEG_RATE and rows2[0]["revert_pairs"] == [], (
        "a rate-leg row's empty list stopped meaning 'leg never evaluated': "
        + json.dumps(rows2[0], sort_keys=True))


def test_revert_pairs_is_pinned_on_a_merged_row_as_well_as_on_a_created_one(tmp_path):
    """Clause 3: a merge can speak about this field, so losing the key reddens both paths.

    #4251 fixed the DIRECTION of a widening but left the per-row fields asserted on the create
    only, which is why a merge could not be said to carry them. One create and one merge run on
    the SAME revert-leg ledger and the SAME board here, and `_pair_rows` subscripts the field
    on both rows — `_create` and `_merge` reach the store through the one `_event`, so the
    field is written by both and this assertion is what makes that load-bearing rather than
    incidental. Deleting the key from `_event` reddens this node and the two above it, one on
    each path.
    """
    board = _new_board(tmp_path, "b-merge")
    led = _two_revert_ledger(tmp_path, "merge-reverts")
    first = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                             window=_post_flip_window(), backlog_dir=board, ledger=led)
    assert first is not None and first["action"] == "filed", first
    more = DIS_MOVES + [{"item_id": 3119, "status": "draft",
                         "note": "the vec0 index needs a hand-side copy rebuild"}]
    want = dict(DIS_EXPECTED)
    want[3119] = ("up_next", "in_progress by no round, ready by its acceptance")
    second = D.file_or_update(_agree(3119), more, want, window=_post_flip_window(),
                              backlog_dir=board, ledger=led)
    assert second is not None and second["action"] == "merged", second
    actions = [e["action"] for e in S.read_events(path=led) if e.get("event") == D.EVENT]
    assert actions == ["filed", "merged"], actions
    assert _pair_rows(led) == [EXPECTED_REVERT_PAIRS, EXPECTED_REVERT_PAIRS], (
        "the widening lost the pairs the create wrote: "
        + json.dumps(_pair_rows(led), sort_keys=True))


def test_the_reason_the_pairs_hands_back_alongside_is_still_the_old_sentence(tmp_path):
    """Clause 4: `below_floor` gained a third return element; its sentence did not move.

    Widening the tuple is a shape change on the ONE surface the trigger sentence is built
    from, so the sentence is pinned here against the rendered text as well as through the body
    the existing nodes read — the pair set and the reason come off the SAME call, the row
    carries both reverts, and the sentence still names #4172 alone. That asymmetry is #4381's
    finding, and this round deliberately leaves WHICH pair to name as owed-check's ruling.
    """
    led = _two_revert_ledger(tmp_path, "reason-wording")
    hit, why, pairs = D.below_floor(_post_flip_window(), led)
    assert hit is True, why
    assert _four_keys(pairs) == EXPECTED_REVERT_PAIRS, json.dumps(pairs, sort_keys=True)
    first = EXPECTED_REVERT_PAIRS[0]
    assert why == (
        f"{D.REVERT_TRIGGER_PREFIX} to #{first['item_id']} as `{first['applied']}` was moved "
        f"back to `{first['reverted_to']}` by the state machine on {first['at']} "
        f"(reason: {IN_FLIGHT_REASON}): the steward's applied move was undone inside the "
        f"window, which is the disagreement `rate` can no longer score post-flip"), why
    assert "#4151" not in why and "2026-11-17" not in why, why


def test_a_filing_that_raises_cannot_change_the_tick(board, monkeypatch):
    """Clause 4. `agreement` is computed from tick SUCCESS, so a reader that can
    break the instrument makes the flip harder to earn: every failure mode is
    caught, logged and swallowed, and nothing below the call is skipped."""
    from workers.sources import _common as C
    from workers import board_disagreement as D

    _ten_item_board(board)
    monkeypatch.setattr(D, "window_now", lambda **kw: _window7())

    def explode(*a, **k):
        raise RuntimeError("backlog is a smoking crater")
    monkeypatch.setattr(D, "file_or_update", explode)

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": [], "next_pick": 0, "next_pick_reason": "",
                               "summary": "net 0 today"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out["status"] == "success", out
    # `apply` is not a meta key; the dry-run proof is the summary's own wording
    # plus the fact that no item on the board moved.
    assert "dry run" in out["summary"], out["summary"]
    assert _filed(board) == [], "a filing that raised must leave no partial item"


def test_a_below_floor_tick_files_the_item_and_moves_nothing(board, monkeypatch):
    """Clauses 1 and 4 end to end: the reader runs as the last act of an ordinary
    tick, and the tick's outcome, its ledger row, the flip flag and every item's
    status are exactly what they were without it — #7 stays `up_next`, which is
    the disagreement itself, left for a person or autotriage to rule on."""
    from workers.sources import _common as C
    from workers import board_disagreement as D

    _ten_item_board(board)
    monkeypatch.setattr(B, "desired_statuses",
                        lambda ledger: {7: ("up_next", "confirmed with acceptance")})
    monkeypatch.setattr(D, "window_now", lambda **kw: _window7())

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": [{"item_id": 7, "status": "draft",
                                          "note": "never triaged"}],
                               "next_pick": 0, "next_pick_reason": "",
                               "summary": "net 0 today"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))

    assert out["status"] == "success", out
    assert "dry run" in out["summary"], out["summary"]
    assert out["meta"]["agreement"]["disagree"] == [7]
    items = _filed(board)
    assert len(items) == 1
    assert "#7" in items[0].body
    seven = next(B.load_item(q) for q in board.glob("7-*.md"))
    assert seven.status == "up_next", "the reader moved an item"
    rows = [e for e in S.read_events(path=S.LEDGER_PATH)
            if e.get("event") == "board_steward_disagreement"]
    assert len(rows) == 1 and rows[0]["rate"] == 0.7953
    ticks = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "board_steward"]
    assert len(ticks) == 1, "the reader wrote a second tick row"


# ─────────── #3546: the steward is shown the round state its prompt reads ───────────
#
# The figures are the ones #3546 was filed on. The board-steward tick at
# 2026-10-29T17:27:13Z proposed `up_next` for two items `desired_statuses` wanted
# held `in_progress`: #3518 (round SM_20261029_161921, `review kind=retry` at
# 17:08:17 and again at 17:36:16, worktree still open) and #3524 (round
# SM_20261029_150548 promoted 16:47:56Z at commit 752cca1a…, unsettled until
# `item_landed` closed:true at 17:34:32Z). The steward's own note for #3518 was
# "round SM_20261029_161921 finished 17:11 … no promotion, no settlement, no
# item_landed, so no round is running; unspent, offered again" — a correct read
# of the events it was GIVEN and wrong about the round. It was given:
#   * a board line (`_item_line`) carrying id, status, name, created, tags and
#     body — no round id and no round state, and
#   * a `promoted` ledger row that names a round and a commit but no item: 0 of
#     the 400 `promoted` rows in the live ledger carry an `item_id`, and 0 of the
#     381 `settled` rows carry a `round_id`, so `events_since`' item join (which
#     reads the row's own `item_id`) can never attach either to #3524.
# Every number below is one of those two rounds' or of the window that filed the
# item (judged 119, agree 90, disagree 28, rate 0.7563 since 2026-10-22T17:48:54).

DIS_ROUND = "SM_20261029_150548"           # #3524's promoted round
DIS_COMMIT = "752cca1a" * 5                # its landing commit, 40 hex
DIS_ROUND_3518 = "SM_20261029_161921"      # #3518's round, gate retried, worktree open


def _led(led: Path, **row):
    """One ledger row, written the way the writers write it. The path is a
    parameter on purpose: these tests also read the row shapes back with
    `item_refs`, and a helper that silently wrote the LIVE ledger would put a
    fixture round into the file every reader in the fleet scans."""
    S.append_event(row, path=led)


def _promotion(led: Path, item_id: int, round_id: str, commit: str, *,
               settle: bool, base: float) -> None:
    """An implement turn that finished, its promotion, and its settlement or absence."""
    _led(led, event="backlog_implement", item_id=item_id, phase="started", ts=base)
    _led(led, event="backlog_implement", item_id=item_id, phase="finished",
         round_id=round_id, ts=base + 1)
    _led(led, event="promoted", round_id=round_id, commit=commit, ts=base + 2)
    if settle:
        _led(led, event="settled", commit=commit, ts=base + 3)


def test_an_item_whose_round_is_unfinished_is_shown_the_round_and_the_state(
        board, tmp_path, monkeypatch):
    """Clause 1. The states that make a round unfinished, in the machine's
    own words ("a gate, landing, worktree or observation is still live",
    `items_with_unfinished_rounds`), have to be visible on the board line —
    #3518's worktree was open and #3524's promotion unsettled at the filing tick
    and NEITHER was renderable from what the steward was given. A promotion that
    settled AND was recorded on the item is a finished round and renders nothing:
    the same grounding that holds a re-offer back must not park every item that ever
    landed. #3602 split that sentence in two — between the `settled` row and the
    `item_landed` one the round is NOT over, and this test's #3600 case is exactly
    that interval, which used to render nothing at all."""
    _write(board, 3524, "up_next")
    _write(board, 3518, "up_next")
    _write(board, 3610, "up_next")
    _write(board, 3620, "up_next")
    _write(board, 3600, "up_next")
    led = S.LEDGER_PATH          # the board fixture points this at tmp_path
    _promotion(led, 3524, DIS_ROUND, DIS_COMMIT, settle=False, base=1)
    _led(led, event="backlog_implement", item_id=3518, phase="finished",
         round_id=DIS_ROUND_3518, ts=10)
    _led(led, event="backlog_implement", item_id=3610, phase="finished",
         round_id="SM_GATE", ts=20)
    _led(led, event="backlog_implement", item_id=3620, phase="finished",
         round_id="SM_LAND", ts=30)
    _promotion(led, 3600, "SM_SETTLED", "a" * 40, settle=True, base=40)

    live = tmp_path / "wt" / DIS_ROUND_3518 / "home" / "lloyd"
    live.mkdir(parents=True)
    monkeypatch.setattr("scripts.automod.worktree.worktree_path",
                        lambda rid: live if rid == DIS_ROUND_3518 else tmp_path / "no-such-wt")
    monkeypatch.setattr(S, "gate_in_progress",
                        lambda rid: {"pid": 4242} if rid == "SM_GATE" else None)
    monkeypatch.setattr(S, "land_in_progress",
                        lambda rid: {"pid": 4343} if rid == "SM_LAND" else None)

    ids = {3524, 3518, 3610, 3620, 3600}
    st = W.unfinished_rounds(S.LEDGER_PATH, ids)
    assert st[3524] == (DIS_ROUND, ("promoted",)), "promoted and unsettled"
    assert st[3518] == (DIS_ROUND_3518, ("worktree",)), "its worktree is still open"
    assert st[3610] == ("SM_GATE", ("gate",)), "a gate process is running"
    assert st[3620] == ("SM_LAND", ("landing",)), "a landing is running"
    # #3600's promotion settled at base+3 and the sweep has written nothing on the
    # item yet, which is the interval #3450's tick fell through — `promoted` is gone
    # (the settlement withdrew it, correctly) and the round is still not over.
    assert st[3600] == ("SM_SETTLED", ("closing",)), "settled, and no `item_landed` yet"
    # Once the landing IS recorded the window shuts, and that is the half of the old
    # assertion this one replaces: a landed item renders nothing, or every item that
    # ever landed would read as unfinished.
    _led(led, event="item_landed", item_id=3600, round_id="SM_SETTLED",
         commit="a" * 40, ts=50)
    assert 3600 not in W.unfinished_rounds(S.LEDGER_PATH, ids), \
        "a promotion that SETTLED and was recorded on the item is a round that is over"

    line = next(l for l in W.build_prompt(
        events=[], items=[_item(3524, "up_next")], n_open=5, since_ts=0,
        round_state=st).splitlines() if l.startswith("#3524 "))
    assert f"round={DIS_ROUND}" in line, line
    assert "round_state=promoted" in line, line
    # The bracketed status is the item's OWN (`up_next` on disk), which is the
    # fact the steward needs. `desired_statuses` wants `in_progress` for this
    # item — that is the disagreement #3546 exists to settle — and printing the
    # machine's answer on the line would score agreement off the prompt.
    assert "[up_next]" in line, line
    for leak in ("desired", "want", "machine", "expected"):
        assert leak not in line, f"the machine's verdict leaked onto the line: {line}"
    # And structurally: the grounding renderer is never even handed the table.
    import inspect
    assert "expected" not in inspect.signature(W._item_line).parameters
    assert "expected" not in inspect.signature(W.build_prompt).parameters


def test_a_promoted_or_settled_event_line_names_the_item_its_round_belongs_to(tmp_path):
    """Clause 2. `promoted` carries `round_id` + `commit` and no `item_id` (0 of
    400 rows carry one), `settled` carries `commit` and no `round_id` (0 of 381
    rows carry one), and `events_since` attaches a row to a shown item only
    through the row's own `item_id` — so the observation rule the prompt states
    was applied to a row that could not be joined to any item. The join runs
    through `round_id` to the ledger's `backlog_implement` rows, which DO carry
    an item; a round no `backlog_implement` row names renders no item."""
    led = tmp_path / "ledger.jsonl"
    _promotion(led, 3524, DIS_ROUND, DIS_COMMIT, settle=True, base=1)
    # A promotion whose round no `backlog_implement` row names: its implement turn
    # diverged, so the round exists and its item does not appear anywhere.
    _led(led, event="promoted", round_id="SM_ORPHAN", commit="b" * 40, ts=50)

    def row(**over):
        d = {"ts": 3, "event": "promoted", "round_id": DIS_ROUND, "commit": DIS_COMMIT}
        d.update(over)
        return d

    refs = W.item_refs(led)
    assert refs[DIS_ROUND] == 3524, "a round id is a key"
    assert refs[DIS_COMMIT] == 3524, "so is the commit, which is all a settled row names"

    line = W._event_line(row(), item_refs=refs)
    assert "event=promoted" in line and f"round_id={DIS_ROUND}" in line, line
    assert "item_id=3524" in line, f"a promotion must name its item: {line}"
    settled = W._event_line(row(event="settled", ts=4), item_refs=refs)
    assert "item_id=3524" in settled, f"a settlement must name its item: {settled}"

    orphan = W._event_line(row(round_id="SM_ORPHAN", commit="b" * 40), item_refs=refs)
    assert "SM_ORPHAN" in orphan, orphan
    assert "item_id=" not in orphan, f"a round with no implement row got a guessed item: {orphan}"
    assert "item_id=" not in W._event_line(row()), "no refs passed: the line is as it was"


def test_the_prompt_says_a_finished_turn_is_not_the_end_of_a_round():
    """Clause 3. The prompt's in-flight rule was turn-level — "a
    `backlog_implement` `started` with nothing after it is a round in flight" —
    while the machine's is round-level, and #3518's proposal note is the gap
    verbatim: a `finished` row read as the end of the round when its gate was
    retried 25 minutes later. The prompt must say the turn/round distinction and
    define every `round_state` token it now points at."""
    text = W.PROMPT
    assert "`finished`" in text and "not the end of a round" in text
    assert "stays `in_progress`" in text
    assert "NEVER proposed `up_next`" in text, "the up_next trap must be closed in words"
    assert "round_state=" in text, "the prompt must point at the marker on the board line"
    for state in W.ROUND_STATES:
        assert f"`{state}`" in text, f"`{state}` is not defined in the prompt"
    # Five since #3602: `closing` is the window the four above do not name, and the
    # prompt defines it too — a state rendered on a line and undefined in the prompt
    # is the inference the model has to guess at, which is what #3546 was for.
    assert W.ROUND_STATES == ("promoted", "closing", "worktree", "gate", "landing")


def test_the_disagreement_item_says_how_much_of_the_window_it_does_not_cover(board, tmp_path):
    """Clause 4. `file_or_update` takes its ids from the CURRENT tick while the
    numbers it cites are the 7-day aggregate: the item filed at 17:27:13Z names 2
    rows against a window holding 28 disagreements over 25 distinct items, so a
    ruler reading it sees 2 and believes the case is small. The body must state
    the arithmetic — window `disagree` minus the rows it writes — and say so again
    after a merge adds rows."""
    from workers.board_disagreement import file_or_update

    out = file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                         window=_window7(), backlog_dir=board,
                         ledger=tmp_path / "ledger.jsonl")
    assert out["action"] == "filed" and out["rows"] == 5
    body = _filed(board)[0].body
    assert "## Coverage" in body
    assert "21 of the window's 26 disagreement rows are not represented here" in body, body

    more = file_or_update(_agree(3119), DIS_MOVES + [
        {"item_id": 3119, "status": "draft", "note": "no triage verdict"}],
        {**DIS_EXPECTED, 3119: ("up_next", "confirmed with acceptance")},
        window=_window7(), backlog_dir=board, ledger=tmp_path / "ledger.jsonl")
    assert more["action"] == "merged" and more["new_ids"] == [3119]
    body = _filed(board)[0].body
    assert "now names 6 of the window's 26 disagreement rows" in body, body
    assert "20 are not represented" in body, body

    # A tick that names the whole window says so instead of claiming a gap.
    small = tmp_path / "small"
    small.mkdir()
    file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                   window=_window7(disagree=5), backlog_dir=small,
                   ledger=tmp_path / "small.jsonl")
    covered = _filed(small)[0].body
    assert "all 5 of the window's 5 disagreement rows are represented" in covered, covered


# ---------------------------------------------------------------------------
# #3602 — the window between a promotion's `settled` row and the item's
# `item_landed` row, which #3546's grounding rendered as no state at all
# ---------------------------------------------------------------------------

def _closing_ledger(tmp_path, *, item=3450, round_id="SM_T_CLOSING", ts_landed=None,
                    landed_names_round=True, settle=True, rolled_back=False):
    """A ledger in #3450's exact shape, written to `tmp_path`, in its real order.

    The live rows it replays (`~/.local/state/lloyd-automod/promotions.jsonl`,
    2026-10-30): `backlog_implement` started 20:38:13Z → the worker's TURN killed by
    the 3540s cap at 21:36:40 → `land_rescued` 21:36:41 ("gate passed at cdd105aa
    and no landing; the loop started one") → `promoted` 21:38:06 → `settled` 21:43:07
    → the steward's tick 21:51:37 → `item_landed` 21:56:27. Times here are relative
    seconds on the same skeleton; the ts an assertion turns on is the settlement
    sitting between the promotion and the landing.

    `ts_landed=None` is the tick's own instant: everything up to the settlement and
    no landing recorded on the item yet. `rolled_back` swaps the `settled` row for a
    `rollback_succeeded`, which must NOT read as a closing window — the prompt sends
    a rolled-back item to `up_next`, so holding it would contradict a rule that is
    right.
    """
    rows = [{"ts": 1000.0, "event": "backlog_implement", "action": "started",
             "item_id": item, "round_id": round_id},
            {"ts": 3400.0, "event": "backlog_implement", "action": "finished",
             "item_id": item, "round_id": round_id, "outcome": "aborted",
             "note": "worker turn exceeded 3540s; backend cancel accepted"},
            {"ts": 3401.0, "event": "land_rescued", "item_id": item,
             "round_id": round_id, "commit": "cdd105aa" + "0" * 32,
             "detail": "gate passed and no landing; the loop started one"},
            {"ts": 3500.0, "event": "promoted", "round_id": round_id,
             "commit": "9e5324b7" + "0" * 32}]
    if rolled_back:
        rows.append({"ts": 3800.0, "event": "rollback_succeeded", "round_id": round_id,
                     "item_id": item, "commit": "9e5324b7" + "0" * 32})
    elif settle:
        rows.append({"ts": 3800.0, "event": "settled", "commit": "9e5324b7" + "0" * 32})
    if ts_landed is not None:
        row = {"ts": ts_landed, "event": "item_landed", "item_id": item,
               "commit": "9e5324b7" + "0" * 32}
        if landed_names_round:
            row["round_id"] = round_id
        rows.append(row)
    p = tmp_path / "promotions.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _shown_item(item_id: int = 3450, status: str = "in_progress"):
    """A board line's worth of an item, for `_item_line` only."""
    return SimpleNamespace(id=item_id, name="board-steward disagreed with the state "
                           "machine and nobody has ruled which side is wrong",
                           status=status, created="2026-10-30", body="# Serve",
                           tags=["spawned-by-steward"], group=None, members=[],
                           parent=None)


def test_an_item_whose_settled_landing_has_no_item_landed_row_yet_shows_closing(tmp_path):
    """Clause 1. #3450 at its 2026-10-30T21:51:37Z tick: promoted 21:38:06, settled
    21:43:07, `item_landed` not until 21:56:27 — and the steward's move note for that
    tick says the only thing the line could say, "Board line carries NO round_state=,
    so the round is over". The window needs a state of its own, so `ROUND_STATES`
    gains `closing` and the round keeps its id on the line.
    """
    ledger = _closing_ledger(tmp_path)

    got = W.unfinished_rounds(ledger, {3450})
    assert got == {3450: ("SM_T_CLOSING", ("closing",))}, (
        f"the settled → item_landed window must name the round and the new state, got {got}")
    assert "closing" in W.ROUND_STATES and W.ROUND_STATES.index("closing") == 1, W.ROUND_STATES

    line = W._item_line(_shown_item(), round_state=got[3450])
    assert "round=SM_T_CLOSING" in line, line
    assert "round_state=closing" in line, line


def test_a_promotion_that_settled_and_a_turn_that_aborted_is_not_read_as_a_finished_round(tmp_path):
    """The premise the acceptance check measures, stated as the shape the steward
    used to be shown: the SAME ledger, with the state arm removed, says nothing.

    This is the replay of the live ledger at the tick's instant, kept as a red line:
    the abort row is present in both, so the only thing that can tell the steward
    "the outcome is the landing, not the abort" is the state on the line. If
    `_item_line` were rendered without it the text would read exactly as #3450's did
    — a `finished` row and no `round_state` — and the prompt's own rule sends that to
    `up_next`.
    """
    ledger = _closing_ledger(tmp_path)
    state = W.unfinished_rounds(ledger, {3450}).get(3450)

    line = W._item_line(_shown_item(), round_state=state)
    assert "round_state=" in line, (
        "a board line with an aborted `finished` row and NO round_state is the text "
        "that made #3450's tick propose up_next for an item already being landed")

    # And the state is not `promoted`: re-adding the settled commit to
    # `_unsettled_promotions` would "fix" this by rendering every item that ever
    # landed as unfinished forever, which `_unsettled_promotions`' docstring rules
    # out. The settlement is still visible as a separate fact.
    assert state is not None and state[1] != ("promoted",), state
    assert "SM_T_CLOSING" not in W._unsettled_promotions(ledger)


def test_a_landing_the_sweep_recorded_on_the_item_leaves_no_round_state(tmp_path):
    """Clause 2. Once `item_landed` exists the window is shut, and an item that
    landed must never be shown as unfinished — the opposite failure
    `_unsettled_promotions`' docstring calls "a worse one", because the line would
    then be telling the steward never to offer the item again.

    Both join arms are tested, because either alone leaves a hole: the `round_id`
    arm is exact but a row without the field must still close the window, and the
    per-item ts arm must not let an OLDER landing hide a NEWER round's window.
    """
    landed = _closing_ledger(tmp_path / "named", ts_landed=4100.0)
    assert W.unfinished_rounds(landed, {3450}) == {}, "item_landed naming the round shuts it"

    unnamed = _closing_ledger(tmp_path / "unnamed", ts_landed=4100.0,
                              landed_names_round=False)
    assert W.unfinished_rounds(unnamed, {3450}) == {}, \
        "a landing row that lost round_id must still close the window by its ts"

    # The ts arm's own negative control: a landing EARLIER than this promotion is a
    # previous round's landing and closes nothing — so the window stays open.
    older = _closing_ledger(tmp_path / "older", ts_landed=900.0,
                            landed_names_round=False)
    assert W.unfinished_rounds(older, {3450}) == {3450: ("SM_T_CLOSING", ("closing",))}, \
        "an older landing must not hide a newer round's window"


def test_a_rolled_back_landing_is_not_a_closing_window(tmp_path):
    """`closing` keys on a settlement, and a `rollback_succeeded` is the other
    verdict a commit gets. The prompt's rule for that verdict is `up_next`, so
    rendering the item as closing would hold in place a change the guardian already
    reverted — and `_unsettled_promotions` drops it for the same reason, so the two
    readers must not disagree about what a revert is.
    """
    ledger = _closing_ledger(tmp_path, rolled_back=True)
    assert W.unfinished_rounds(ledger, {3450}) == {}, \
        "a rolled-back landing must leave the item offerable, not held"


def test_an_older_rounds_closing_window_is_named_when_the_newest_round_is_over(tmp_path):
    """The same re-run shape #3524 pinned, now on the settled side: #3538's
    `SM_20261029_145233` landed and settled while `SM_20261029_161851` was abandoned,
    and the newest row names only the re-run. Naming the newest round with no state
    would hide the live window, so the older round is named instead.
    """
    ledger = _closing_ledger(tmp_path)
    rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.append({"ts": 3900.0, "event": "backlog_implement", "action": "started",
                 "item_id": 3450, "round_id": "SM_T_RERUN"})
    rows.append({"ts": 3910.0, "event": "backlog_implement", "action": "finished",
                 "item_id": 3450, "round_id": "SM_T_RERUN", "outcome": "aborted",
                 "note": "aborted for a rebase conflict"})
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    assert W.unfinished_rounds(ledger, {3450}) == {3450: ("SM_T_CLOSING", ("closing",))}, \
        "the state must be named on the round that holds it"


def test_the_prompt_governs_the_closing_window():
    """Clause 3. The prompt already routes "a `finished` row and NO `round_state`" to
    "read its outcome", and "a round that aborted for a reason outside its diff" to
    `up_next` — two rules that are each right and that #3450's tick composed into the
    wrong move, because the window it sat in rendered neither the settlement nor the
    landing. The new rule has to say all three things the tick needed: the item stays
    where it is, the abort row is not the round's outcome, and `up_next` is not
    available while a promotion or a closing window is on the item's lines.
    """
    text = W.PROMPT
    assert "round_state=closing" in text, "the prompt must define the state it now renders"

    rule = text[text.index("- **`round_state=closing`"):]
    rule = rule[:rule.index("- A round that aborted for a reason outside")]
    for want in ("`settled`", "item_landed", "propose NO move", "not `up_next`",
                 "not `draft`", "stays `in_progress`"):
        assert want in rule, f"{want!r} missing from the closing rule: {rule}"
    assert "turn-timeout/abort row is NOT the round's outcome" in rule, rule
    assert "land_rescued" in rule, \
        "the row that proves the landing belongs to the round must be named"
    assert "#3450" in rule and "21:56:27" in rule, \
        "the case the rule is for must be citable from the prompt itself"

    # The aborted-outside-diff rule now carries the condition the tick lacked.
    aborted = text[text.index("- A round that aborted for a reason outside"):]
    aborted = aborted[:aborted.index("- **A review refusal")]
    assert "NO line of this item names `promoted` or" in aborted, aborted


#: The claim #4387 retires, built here from two fragments rather than written whole so
#: that this file does not itself trip the acceptance it enforces: `git grep -n` for the
#: phrase over the tracked tree had to return 0 lines and returned 4 before this round,
#: and one of those 4 was this file asserting the phrase was PRESENT. Every node below
#: asserts this string ABSENT, and
#: `test_the_retired_claim_survives_nowhere_in_the_text_that_publishes_it` folds each
#: file's line wraps away first — which is the hole #4387's own check fell into, since
#: the module docstring held the claim across a wrap and the item's grep never saw it.
REPLY_CLAIM_RETIRED = ("`response_json` is empty in every " "board-steward `runs` row")


def test_the_disagreement_body_names_the_ledger_event_as_where_the_move_lives(board,
                                                                              tmp_path):
    """Clause 4. The body claimed "the proposed move is stored nowhere else … so this
    item is the record". That is false: the same tick's `board_steward` ledger event
    carries `moves` with the `item_id`, the `status` and the `note` the steward wrote
    (#3602's triage read the 2026-10-30T21:51:37Z row and found all three).

    #4387 is the same failure in the paragraph after that one, and this node was part of
    it. The body went on to publish the model's raw reply as kept nowhere at all, and
    until this round this node asserted that sentence WAS there — a pinned false claim is
    worse than an unpinned one, because the filing site and the test agree to keep
    re-issuing it. `43cc6ba7` (landed 2026-11-19) is the commit that started writing the
    reply into `runs.response_json`, so what the body owes a reader now is the boundary
    and the way back into an older tick's reply, not a permanent absence. The
    hyphenated-`source` caveat stays: filtering `runs` by the ledger's `board_steward`
    spelling answers 0 rows, which a reader meets as "the store keeps nothing".
    """
    from workers.board_disagreement import REPLY_PROVENANCE, file_or_update

    file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED, window=_window7(),
                   backlog_dir=board, ledger=tmp_path / "ledger.jsonl")
    body = _filed(board)[0].body

    assert "stored nowhere else" not in body, body
    assert "this item is the record" not in body, body
    assert "`board_steward` event" in body, body
    for want in ("`moves`", "`item_id`", "`status`", "`note`"):
        assert want in body, f"{want!r} missing: the ledger's own field names are the way back"
    assert REPLY_PROVENANCE in body, body
    assert REPLY_CLAIM_RETIRED not in body, (
        "the retired absolute is back in the rendered filing body, so the next item this "
        "module files publishes a pre-`43cc6ba7` fact as the store's live limitation")
    for want in ("`meta_json.session_id`", "`~/lloyd-data/sessions/<id>.json`",
                 "`43cc6ba7`", "2026-11-19"):
        assert want in body, f"{want!r} missing: the body must name the boundary and the " \
                             f"route back to any tick's raw reply"
    assert "hyphenated" in body and "matches no runs row at all" in body, (
        "the hyphenated-`source` caveat is still true and still the only thing stopping a "
        "reader concluding from a 0-row `board_steward` query that nothing is kept")
    assert "`desired_statuses`" in body, "the machine's side must still be named"


# ─────────── #3651: the filing reads the WINDOW, not the tick that tripped it ───────────
#
# The figures are the ones this item was filed on. At 2026-11-01T02:46Z the published
# judgement was `judged 122, agree 96, disagree 25, rate 0.7869` over seven days, and the
# item it filed named exactly ONE row (#3418), because `file_or_update` took its ids from
# the current tick's `agreement()["disagree"]`. A recount over the same window's `runs`
# rows names 23 distinct ids, and every one of them has its proposed status in a
# `board_steward` ledger event of that window (361 events, 45 carrying non-empty `moves`,
# 23 of 23 recovered) — so the one row was never a limit of the store, only of the reader.

#: `_window7()`'s five, plus two ids an earlier tick disagreed about.
DIS_WINDOW = DIS_AGREE + [3310, 3355]
#: Inside `_window7()["window_since"]` = 2026-10-20T23:49:13+00:00, so the window filter
#: must keep it; a `created_at` is what `state.append_event` writes, and passing one here
#: is what makes an "older tick" reproducible instead of always-now.
DIS_OLDER_STAMP = "2026-10-26T01:02:03Z"


def _row_count(body: str) -> int:
    """How many rows the `## Rows` section holds — the section, not every `- ` line in the
    body, because `## Coverage` opens its bullets the same way."""
    return sum(1 for line in body.split("## Rows\n\n")[-1].split("## Coverage")[0]
               .splitlines() if line.startswith("- #"))


def _steward_event(led: Path, ids: list[int], *, status: str = "up_next",
                   created_at: str = DIS_OLDER_STAMP, moves: bool = True) -> None:
    """One tick's own `board_steward` ledger row — where its proposals live.

    `moves=False` is the shape of a tick whose recorded move list is empty: the ids in the
    window then have no proposal anywhere, which is the only genuinely unrecoverable case.
    """
    _led(led, event="board_steward", apply=False, created_at=created_at,
         moves=[{"item_id": i, "status": status, "note": "unspent, offered again"}
                for i in ids] if moves else [])


def test_the_filing_names_every_window_id_whose_move_is_in_the_window(board, tmp_path,
                                                                       monkeypatch):
    """Clause 1. This tick disagreed about five items; seven disagreed across the window,
    and the two older ones' proposals are sitting in an earlier tick's ledger event. A
    filing that names five and cites `rate=0.7869` over 25 is a report on its own tick:
    the ruler was handed one row and asked to judge a week, and never told the rest of
    the window existed (#3651)."""
    from workers.board_disagreement import file_or_update

    led = tmp_path / "ledger.jsonl"
    _steward_event(led, DIS_WINDOW[5:])          # the older tick, ids this tick never saw
    monkey_expected = {i: ("up_next", "has a confirmed triage with acceptance text")
                       for i in DIS_WINDOW}
    monkeypatch.setattr(B, "desired_statuses", lambda ledger: monkey_expected)
    out = file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                         window=_window7(disagree_ids=DIS_WINDOW),
                         backlog_dir=board, ledger=led)
    assert out is not None and out["action"] == "filed", out
    it = _filed(board)[0]
    for i in DIS_WINDOW:
        assert f"#{i}" in it.body, f"#{i} is in the window's disagree set and not in the rows"
    assert _row_count(it.body) == len(DIS_WINDOW), (
        f"the Rows section must hold the window's ids and only those: {it.body}")
    assert "- 0 of the named ids are unrecoverable" in it.body, it.body
    assert "steward proposed `up_next`" in it.body, (
        "the two older ids' side comes from the earlier tick's move, and that move was "
        "`up_next` — if it reads `(not in its moves)` the ledger read is not happening")


def test_an_id_no_move_proposed_is_named_and_counted_as_unrecoverable(board, tmp_path,
                                                                        monkeypatch):
    """Clause 2. A disagreeing id with no proposal in ANY move of the window is still
    named — the machine's half is the half a reader can act on — and Coverage says how
    many named ids are ruled blind, so the gap is visible instead of silent."""
    from workers.board_disagreement import file_or_update

    led = tmp_path / "ledger.jsonl"
    _steward_event(led, [], moves=False)         # the older tick recorded no moves at all
    monkey_expected = {i: ("up_next", "has a confirmed triage with acceptance text")
                       for i in DIS_WINDOW}
    monkeypatch.setattr(B, "desired_statuses", lambda ledger: monkey_expected)
    out = file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                         window=_window7(disagree_ids=DIS_WINDOW),
                         backlog_dir=board, ledger=led)
    assert out is not None and out["action"] == "filed", out
    it = _filed(board)[0]
    for i in DIS_WINDOW[5:]:
        assert f"#{i}: steward proposed `(not in its moves)`" in it.body, it.body
    assert "2 of the named ids carry `(not in its moves)`" in it.body, it.body


def test_rows_stay_capped_at_forty_and_the_body_states_the_truncation(board, tmp_path):
    """Clause 3. 45 window ids is more than ROWS_MAX can write, and the cap is the only
    thing between this item and a body nobody reads. Dropping five ids silently would be
    the same failure as naming one tick's rows and citing the window: the reader would
    not know how much it was not being shown."""
    from workers.board_disagreement import ROWS_MAX, file_or_update

    many = list(range(4000, 4000 + ROWS_MAX + 5))
    expected = {i: ("up_next", "has a confirmed triage with acceptance text")
                for i in many}
    moves = [{"item_id": i, "status": "draft", "note": "no triage verdict"} for i in many]
    out = file_or_update(_agree(*many), moves, expected,
                         window=_window7(disagree=len(many), disagree_ids=many),
                         backlog_dir=board, ledger=tmp_path / "ledger.jsonl")
    assert out is not None and out["action"] == "filed", out
    it = _filed(board)[0]
    assert out["rows"] == ROWS_MAX, out
    assert _row_count(it.body) == ROWS_MAX, (
        f"{it.body.count(chr(10) + '- #')} rows written against a cap of {ROWS_MAX}")
    assert (f"the rows are capped at {ROWS_MAX}, so {ROWS_MAX} of {len(many)} named ids "
            f"are written and {len(many) - ROWS_MAX} are not") in it.body, it.body


def test_the_architecture_page_says_the_ruling_can_be_answered_from_the_store():
    """Clause 4. The page told a reader not to bother: the proposed move was "stored
    nowhere else" and the question "which side is wrong" "cannot be answered from the
    store". Both halves are false — 23 of the window's 23 disagreeing ids have their
    proposal in that window's `board_steward` events — and a claim that the evidence does
    not exist is what stops anyone going to look for it (#3651).

    The page's third sentence went the same way (#4387): it told the reader the raw reply
    is kept nowhere, and justified that with a count of rows on this box. A count in prose
    is the reason the sentence outlived its own evidence — the figure had rotted before
    the next reader checked it, and `43cc6ba7` (2026-11-19) had already falsified the
    absolute. So the page now carries the same boundary sentence the module files and the
    same recovery route, from the same constant, and quotes no count at all."""
    from workers.board_disagreement import REPLY_PROVENANCE

    doc = " ".join((Path(__file__).resolve().parents[1] / "architecture"
                    / "workers-jobs.md").read_text(encoding="utf-8").split())
    assert "stored nowhere else" not in doc, doc[-900:]
    assert "cannot be answered from the store" not in doc
    assert "answerable from the store" in doc, (
        "the replacement has to state the positive, or a later reader re-derives the "
        "doubt the sentence it replaced was built on")
    assert REPLY_PROVENANCE in doc, (
        "the page states the boundary the filer states, from the module's own constant, "
        "or the two documents drift and a reader of one re-derives the retired claim")
    assert REPLY_CLAIM_RETIRED not in doc, doc[-900:]
    assert "1,123" not in doc, (
        "the page quotes a row count again: this is the figure that had rotted while it "
        "still read as a live measurement, and the sentence needs no count to stand")


def test_the_module_text_states_the_reply_boundary_rather_than_an_absence():
    """Clause 2. `disagreement_rows`' docstring told a reader the store "genuinely never
    kept" the reply, an unqualified present-tense claim about EVERY board-steward `runs`
    row (#4387). It is read by whoever edits the filer, which is exactly the reader a
    stale limitation misleads, so it now carries the same boundary as the body and the
    page.

    Asserted through `inspect.getdoc` and not the raw module source on purpose: the
    constant holding the sentence lives in that same module, so a source-text `in` check
    would be satisfied by the constant's own definition and could not see whether the
    docstring still says otherwise.
    """
    import inspect

    from workers import board_disagreement as D
    from workers.board_disagreement import REPLY_PROVENANCE

    doc = " ".join(inspect.getdoc(D.disagreement_rows).split())
    assert REPLY_PROVENANCE in doc, doc
    assert REPLY_CLAIM_RETIRED not in doc, (
        "the docstring reverted to the unqualified claim about every row")


def test_the_retired_claim_survives_nowhere_in_the_text_that_publishes_it():
    """#4387's acceptance check is a `git grep` over the tracked tree, and the item's own
    addendum records why that check needed widening: at the docstring the claim sat across
    a line wrap, so the item's pattern matched one of its two homes only, and a round that
    satisfied the grep while leaving the docstring absolute would still have shipped the
    false claim. Folding each file's whitespace before comparing closes that hole, and this
    node is the acceptance check in a form that runs on every suite rather than only when
    somebody remembers to grep.

    Three files, because the claim had three publishers: the filer's docstring, the
    architecture page, and this file — a test asserting the false sentence is present is
    how a false sentence keeps being filed and re-triaged as confirmed.
    """
    root = Path(__file__).resolve().parents[1]
    for rel in ("workers/board_disagreement.py", "architecture/workers-jobs.md",
                "tests/test_board_steward.py"):
        flat = " ".join((root / rel).read_text(encoding="utf-8").split())
        assert REPLY_CLAIM_RETIRED not in flat, (
            f"{rel} still publishes the retired claim once its line wraps are folded — "
            f"the wrapped shape is exactly what the acceptance grep could not see")
    for rel in ("workers/board_disagreement.py", "architecture/workers-jobs.md"):
        flat = " ".join((root / rel).read_text(encoding="utf-8").split())
        assert "1,123" not in flat, (
            f"{rel} carries a hard-coded board-steward row count again")


async def test_the_reader_never_consults_the_machine_for_a_rows_half_only_its_removal(
        board, tmp_path, monkeypatch):
    """The reader is a REPORTER: it may consult the machine's live table to REMOVE a row and
    never to build one.

    #3914 changed what "at all" means here, so read this node as a direction and not an
    absence. An id whose ruling has already landed is not an open question, and only a read
    of the board and of what the machine answers NOW can know that — the four ids #3433,
    #3537, #3656 and #3704 are exactly that case, since commit `6119589c` settled them by
    changing what `desired_statuses` answers (`draft`: its hand_run is still unexecuted)
    rather than by moving the board to the `up_next` their rows recorded. So the live read is
    back at this seam with one permission: subtract a row, never supply either half of one.
    `boom` still raises on ANY call, which now proves two things: the abstention guard
    reaches the seam at all (an absence assertion alone would be satisfied by a guard that
    never ran), and a consultation that FAILS prints every window row rather than losing one
    — a guard that cannot decide degrades to the old behaviour, never to silence. The byte
    half of the direction — that a row which SURVIVES the read is the bytes this build
    already printed — is
    `test_rows_that_survive_the_abstention_are_the_bytes_this_build_already_printed`.

    The retired half is unchanged: `desired_statuses` computes the machine's moves; applying
    them is what a round does. This node used to prove the READER only CALLS
    `desired_statuses` and never `apply_moves`, and stubbed it to observe the call (#3602).
    #3677 retired that read for a row's machine half: `desired_statuses` is the board's LIVE
    state at filing time, and answering "what does the machine want" about a disagreement
    scored up to seven days earlier produced 12 rows on that item printing `(the machine
    abstains)` — a sentence `agreement()` makes impossible for an id it filed as `disagree` —
    plus 2 printing the machine agreeing with the steward. A row's machine half still comes
    from the disagreeing tick's own record. The rest of the rail is unchanged — no
    `apply_moves`, no item status written, and the widened read taken from the INJECTED
    ledger only.
    """
    from workers import board_disagreement as D
    import scripts.automod.backlog as BL

    fake_calls: list = []
    live_reads: list = []

    def boom(ledger):
        fake_calls.append(ledger)
        raise AssertionError(
            "the disagreement reader consulted the machine's LIVE table at filing time;"
            " #3677 removed that read because a row must name the side the disagreeing"
            " tick scored, not the side the board happens to want now")

    monkeypatch.setattr(BL, "desired_statuses", boom)
    led = tmp_path / "led.jsonl"
    _ten_item_board(board)                      # writes board/round events to S.LEDGER_PATH
    monkeypatch.setattr(S, "LEDGER_PATH", led)  # the widened read must come from HERE
    real_read, real_append = S.read_events, S.append_event

    def spy_read(path=None, limit=100):
        live_reads.append(path)
        return real_read(path, limit)

    monkeypatch.setattr(S, "read_events", spy_read)
    monkeypatch.setattr(S, "append_event",
                        lambda entry, path=None: real_append(entry, path or led))
    # An id whose disagreement lives in this window's events: the proposal `draft` and the
    # machine's side, both as that tick filed them. The only source a widened row may use.
    # #7 is in the same tick's `disagree` too — this node is about #3914's abstention, and an
    # id that disagreed for the first time this tick would be withheld by #4378's persistence
    # guard before the abstention was ever reached (the guard has its own nodes further down).
    S.append_event({"event": "board_steward",
                    "agreement": {"disagree": [7, 3], "disagree_sides": [
                        {"item_id": 7, "status": "up_next",
                         "why": "ready by its acceptance"},
                        {"item_id": 3, "status": "up_next",
                         "why": "in the offered set, not started"}]},
                    "moves": [{"item_id": 3, "status": "draft", "note": "its reason"}]},
                   path=led)

    want = {7: ("up_next", "ready by its acceptance")}
    before = {it.id: it.status for it in B.all_items(("lloyd",), backlog_dir=board)}
    out = D.file_or_update({"judged": 1, "rate": 0.0, "agree": [], "disagree": [7],
                            "no_opinion": [], "missed": [], "machine_moves": 1,
                            "proposed": 1},
                           [{"item_id": 7, "status": "in_progress",
                             "note": "the n-gram table memory guard"}],
                           want, window=_window7(disagree_ids=[3]), backlog_dir=board,
                           ledger=led)
    assert out is not None, "the reader filed nothing"
    assert fake_calls, (
        "#3914's abstention guard never reached the machine's live table, so a filing whose "
        "window contains an id whose ruling has landed would still name it as an open "
        "question — the phantom #3858 and #3913 re-filed. (#3677's prohibition is below: "
        "ONE call site, and it is the removal guard.)")
    src = (Path(__file__).resolve().parents[1] / "workers" / "board_disagreement.py") \
        .read_text(encoding="utf-8")
    assert "apply_moves" not in src and "apply(" not in src, (
        "the reporter grew a call into the machine's writer")
    assert src.count("desired_statuses(") == 1, (
        f"`desired_statuses` is called {src.count('desired_statuses(')} times from the "
        f"reporter; #3914 permits exactly ONE live consultation, inside `ruled_ids`, whose "
        f"only effect is to remove a row. Any second call site is #3677's filing-time "
        f"recompute coming back under #3914's name")
    ids = out["ids"]
    assert ids[:2] == [7, 3], f"current tick first, then the window: {ids}"
    after = {it.id: it.status for it in B.all_items(("lloyd",), backlog_dir=board)}
    moved = {i: (before[i], after[i]) for i in before if after.get(i) != before[i]}
    assert not moved, f"the reader moved an item: {moved}"
    assert live_reads and all(pth == led for pth in live_reads), (
        f"the widened read did not use the injected ledger: {live_reads}")
    body = _filed(board)[0].body
    assert "#3" in body, "the window's id must still be named"
    assert "(the machine abstains)" not in body
    row3 = [ln for ln in body.splitlines() if ln.startswith("- #3")][0]
    assert "the state machine wants `up_next` — in the offered set, not started" in row3, (
        f"the widened row did not print the side its own tick recorded: {row3}")

    # The fail-safe, from `boom` raising above: a consultation that cannot be made is NOT a
    # licence to drop rows. Every window row printed, and Coverage reports the abstention it
    # actually made — zero — rather than a number it could not establish.
    assert ids == [7, 3], f"a consultation that raised still removed a row: {ids}"
    assert "- 0 of the window's ids were abstained as already ruled." in \
        body[body.index("## Coverage"):], (
        "the body claims an abstention the guard could not have made: "
        f"{body[body.index('## Coverage'):]}")


# ── #3677: a row must name the machine's side as the disagreeing tick recorded it ──
#
# `agreement()` puts an id in `disagree` only when the machine HAD an opinion about it
# (`workers/sources/board_steward.py:920`; abstention is the separate `no_opinion` bucket
# at `:921`), but the block stored only id lists — key set
# `('agree','disagree','judged','machine_moves','missed','no_opinion','proposed','rate')`,
# no per-id status — so the filer re-ran `desired_statuses` at filing time
# (`workers/board_disagreement.py:225`) and printed `(the machine abstains)` for ids whose
# tick had one. Item #3677 measured 12 of its own 22 rows that way, and 2 more printed as
# agreement inside the disagreement list. A ruler is told to judge the machine's side; the
# side it was shown was a later read of a ledger the tick itself had moved.
#
# The four nodes below are the four clauses. Each is pinned at the seam the bug sat on:
# the block the tick emits, the JSON the ledger stores, the renderer that prints a row,
# and the lock that was supposed to protect a person's ruling.


def _agree_with_sides(ids, sides):
    """An agreement block carrying per-id recorded machine `sides` ({id: (status, why)}).

    Built through `agreement()` rather than hand-typed where that is possible; this
    hand-built form exists so a test can present a tick that predates the record (an
    id in `disagree` with no side) without faking the classifier.
    """
    agree = _agree(*ids)
    agree["disagree_sides"] = [{"item_id": i, "status": s, "why": w}
                               for i, (s, w) in sides.items()]
    return agree


def test_a_tick_records_the_machine_side_for_every_id_it_called_a_disagreement():
    """Clause 1, first half: the block that reaches the ledger event and `meta_json`."""
    agree = W.agreement(moves=[{"item_id": 3456, "status": "in_progress"},
                               {"item_id": 3433, "status": "draft"},
                               {"item_id": 3024, "status": "up_next"},
                               {"item_id": 3551, "status": "up_next"}],
                        expected={3456: ("up_next", "offered again — clock ran out"),
                                  3433: ("up_next", "triage confirmed it"),
                                  3024: ("up_next", "triage confirmed it")},
                        current={3456: "in_progress", 3433: "up_next",
                                 3024: "up_next"})
    assert agree["disagree"] == [3433, 3456], (
        "the two conflicts, by construction, published sorted")
    assert 3551 not in agree["disagree"], "the abstainer is `no_opinion`, not `disagree`"
    recorded = {int(r["item_id"]): (r["status"], r["why"])
                for r in agree["disagree_sides"]}
    assert recorded == {3456: ("up_next", "offered again — clock ran out"),
                        3433: ("up_next", "triage confirmed it")}, (
        "the machine's side must be recorded for every id the tick itself put in "
        f"`disagree`, and for no other: {recorded}")
    assert [r["item_id"] for r in agree["disagree_sides"]] == agree["disagree"], (
        "the record must line up with the published id list row for row, since both "
        "reach the ledger side by side")
    assert json.loads(json.dumps(agree["disagree_sides"])) == agree["disagree_sides"], (
        "the record must survive the JSON of the ledger line and of `meta_json` "
        "unchanged — a shape that changes across the write reads back as a different "
        "side")

    src = (Path(__file__).resolve().parents[1] / "workers" / "sources"
           / "board_steward.py").read_text(encoding="utf-8")
    handed = src.count('"agreement": agree')
    assert handed == 2, (
        f"the tick hands the agreement block to {handed} sink(s), not both: the "
        "`board_steward` ledger event and the run's `meta_json` must each receive the "
        "whole block, or the record reaches the reader that needs it and not the "
        "dashboard that reports the rate")
    assert 'S.append_event({"event": "board_steward"' in src and '"meta": {' in src, (
        "one of the two sinks stopped existing under its own name, so the count above "
        "would match two writes to the same place")


def test_the_recorded_side_is_not_rendered_into_the_prompt_or_the_board_line(board):
    """Clause 1, second half: recording the side must not hand it to the model.

    #3547's own rail is that the steward's prompt and board line carry no machine
    target status, because the disagreement rate IS the measurement of the steward's
    judgement and printing the answer into its prompt would move the measurement it is
    measured by. So the record lives in the block written after the turn, and the two
    renderers that build what the model reads must stay unable to see it even now that
    it exists — pinned both on their signatures (they cannot be *passed* the machine's
    table) and on the rendered text (the reason string does not appear).

    #3546's `round_state` argument is deliberately not in the forbidden set: it carries
    the machine-derived FLAGS `in_flight`/`observing`, which is #3547's remedy, not a
    target status.
    """
    import inspect

    for fn in (W.build_prompt, W.board_view):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"expected", "sides", "machine", "machine_sides"}, (
            f"{fn.__name__} now takes the machine's table as an argument, which is how "
            "the target status gets rendered into what the model reads")

    _write(board, 3456, "in_progress", tags=("triage-confirmed",),
           name="the n-gram table keeps being double-booted")
    items = B.all_items(("lloyd",), backlog_dir=board)
    why = "offered again — clock ran out before a verdict"
    prompt = W.build_prompt(events=[], items=W.board_view(items, set(), pending={3456}),
                            n_open=len(items), since_ts=0.0)
    assert why not in prompt, (
        "the machine's reason for the status it wants reached the prompt the steward "
        "reads before it forms a view — that is the answer to the question it is "
        "measured on")
    assert "#3456" in prompt, "the fixture stopped being in the rendered board"


def test_the_recorded_side_survives_the_ledger_line_the_filer_reads(tmp_path):
    """The process boundary: a later tick recovers an earlier tick's side from JSON.

    This is the seam the record exists for. `S.append_event` writes one JSON line and
    `S.read_events` hands back parsed objects, so a record keyed by an int item id would
    come back keyed by the STRING "3456" and every lookup would miss — which is the
    failure this node is really about, because it would look like an absent record and
    the row would print `not recorded` for a side that was filed.
    """
    agree = W.agreement(moves=[{"item_id": 3456, "status": "in_progress"}],
                        expected={3456: ("up_next", "offered again — clock ran out")},
                        current={3456: "in_progress"})
    path = tmp_path / "promotions.jsonl"
    S.append_event({"event": "board_steward", "agreement": agree,
                    "moves": [{"item_id": 3456, "status": "in_progress",
                               "note": "its own reason"}]}, path=path)
    events = S.read_events(path)
    assert len(events) == 1, "the ledger line did not read back"
    moves = D._window_moves(_window7(), path)
    assert moves == {3456: "in_progress"}, "the steward's side still recovers"
    sides = D._window_sides(_window7(), path)
    assert sides == {3456: ("up_next", "offered again — clock ran out")}, (
        f"a recorded side must survive the JSON round trip keyed by an int id: {sides}")
    assert D._window_sides(_window7(), tmp_path / "absent.jsonl") == {}, (
        "a missing ledger is an empty record, never a guess about the machine's side")
    S.append_event({"event": "board_steward", "agreement": _agree(3456)}, path=path)
    assert D._window_sides(_window7(), path) == {3456: ("up_next",
                                                        "offered again — clock ran out")}, (
        "a tick that predates the record must not overwrite one that has it")


def test_no_disagreement_row_prints_an_abstention_and_a_row_without_a_record_says_so(
        board, tmp_path):
    """Clause 2: the row is rendered from the record, and an unrecorded side is admitted.

    Both halves matter. The first is the bug: `desired_statuses` re-run at filing time
    answers abstention for an id its tick scored as a conflict. The second is what stops
    the fix from being a lie in the other direction — a row for a tick that predates the
    record must say the side was NOT RECORDED, which is a fact about the artifact, and
    must not assert the machine abstained, which was the claim that made 12 of #3677's
    rows look ruleable-by-abstention when they were never judged that way.
    """
    unrecorded = 3551
    out = D.file_or_update(
        _agree(*DIS_AGREE, unrecorded), DIS_MOVES, DIS_EXPECTED, window=_window7(),
        backlog_dir=board, ledger=tmp_path / "ledger.jsonl")
    assert out is not None and out["action"] == "filed"
    body = _filed(board)[0].body
    assert "(the machine abstains)" not in body, (
        "an id in `disagree` printed as an abstention — by construction the tick had a "
        "machine side for it, so the row was rendered from a later read, not from the tick")
    assert "#3551" in body
    row = [ln for ln in body.splitlines() if ln.startswith("- #3551")]
    assert len(row) == 1 and "not recorded" in row[0], (
        f"a row with no tick record must state that: {row}")
    assert "abstain" not in row[0].lower(), (
        f"the row still asserts an abstention: {row[0]}")
    for i in DIS_AGREE:
        assert "not recorded" not in [ln for ln in body.splitlines()
                                      if ln.startswith(f"- #{i}")][0], (
            f"#{i} has a machine side in the tick's own table and must print it")


def test_a_row_is_rendered_from_a_record_only_the_ledger_carries(board, tmp_path):
    """Clause 2's widening half: an id THIS tick did not judge still gets its side.

    The whole reason the record is in the ledger event and not only in the tick's
    arguments is that a merge names ids from earlier ticks. With no record anywhere the
    row admits it; with one in the window's events the row prints it, and no fresh
    `desired_statuses` read is involved — which is what makes the row about the tick
    rather than about the moment the file was written.
    """
    ledger = tmp_path / "promotions.jsonl"
    # The status PROPOSED has to differ from the side RECORDED, because no honest tick ever
    # writes both for one status: `agreement()` puts an id in `disagree` only where
    # `expected[i][0] != s`, and `disagree_sides` is built from exactly that list. An event
    # whose `moves` say `up_next` and whose `disagree_sides` also say `up_next` is a tick
    # that could not have run, and the filer now refuses to print such a pair — #3752's
    # fourth clause is that no filed row may name the same status on both sides — so the
    # row this node is about would render **not recorded** and the assertion at the bottom
    # would be grading an impossible artifact rather than the ledger read.
    S.append_event({"event": "board_steward",
                    "agreement": _agree_with_sides(
                        [3450], {3450: ("up_next", "triaged, no verdict yet")}),
                    "moves": [{"item_id": 3450, "status": "in_progress",
                               "note": "its own reason"}]}, path=ledger)
    # The tick scored only DIS_AGREE; 3450 enters the filing the way every widened row
    # does — named by the window's `disagree_ids` — and its machine side is NOT in this
    # tick's `expected` (it was not this tick's disagreement). It IS recorded in the
    # window's events, which is the only place the row can get it from.
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(disagree_ids=list(DIS_AGREE) + [3450]),
                           backlog_dir=board, ledger=ledger)
    assert out is not None and 3450 in out["ids"], (
        f"the window's id was not widened into the filing: {out}")
    body = _filed(board)[0].body
    assert "(the machine abstains)" not in body
    assert "not recorded" not in [ln for ln in body.splitlines()
                                  if ln.startswith("- #3450")][0], (
        "#3450's machine side is recorded in the window's events and the row ignored it")


def test_the_coverage_section_counts_the_rows_that_carry_a_recorded_machine_side(
        board, tmp_path):
    """Clause 3: Coverage may not say every row is ruleable while some rows are not.

    #3677's own Coverage bullet read "0 of the named ids are unrecoverable: every one
    carries the status a `board_steward` event in this window proposed, so every row here
    can be ruled from the item alone." The first half is true and about the STEWARD's
    side; the second was inferred from it, and the machine's half was missing for 12 of
    the 22. So the count of rows carrying a recorded machine side is now its own published
    number, and the ruleability sentence may only be made when both halves are there.
    """
    out = D.file_or_update(_agree(*DIS_AGREE, 3551), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(), backlog_dir=board,
                           ledger=tmp_path / "ledger.jsonl")
    assert out is not None
    body = _filed(board)[0].body
    cov = body[body.index("## Coverage"):]
    assert "5 of the 6 named rows carry the machine's side as its own tick recorded it" \
        in cov, f"the recorded-side count is missing or wrong:\n{cov}"
    assert "1" in cov and "not recorded" in cov, (
        "the bullet must name the shortfall, not just the covered count")
    assert "every row here can be ruled from the item alone" not in cov, (
        "Coverage still claims full ruleability while one row has no machine side")

    all_ruleable = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                                    window=_window7(), backlog_dir=tmp_path / "b2",
                                    ledger=tmp_path / "l2.jsonl")
    assert all_ruleable is not None
    cov2 = (b := _filed(tmp_path / "b2")[0].body)[b.index("## Coverage"):]
    assert "5 of the 5 named rows carry the machine's side as its own tick recorded it" \
        in cov2, f"with every side recorded the count must say so:\n{cov2}"
    assert "so every row here can be ruled from the item alone" in cov2, (
        "the sentence is legitimate only when it is true, and here it is: both halves "
        "of both sides are in the artifact")


def test_a_merged_coverage_counts_the_rows_the_body_will_actually_hold(board, tmp_path):
    """Clause 3's merge half: the recorded-side count is taken over the rows the
    denominator printed beside it counts, not over the ids this filing widened.

    The review rung sent #3677's round back on exactly this (clause 3 partial, blocking):
    "in a merge the numerator counts this filing's ids while the denominator is the file's
    total named rows, so the written bullet claims N rows have no recorded side while
    those same rows in the body each print a machine side; no test covers the merged
    figure". Here the first filing records all five sides and one row is then given the
    legacy `(the machine abstains)` form, so four of the six rows the merged body will
    hold carry a recorded side; the second filing widens to one id with no record at all.
    The figure has to come from the same rows the bullet counts, so both numbers are read
    out of the bullet and re-counted off the body rather than written as literals — and a
    legacy abstention must NOT count, because an abstention a filing-time recompute
    asserted is not a side its own tick recorded.
    """
    first = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                             window=_window7(), backlog_dir=board,
                             ledger=tmp_path / "ledger.jsonl")
    assert first is not None and first["action"] == "filed"
    path = _filed(board)[0].path

    aged = f"- #{DIS_AGREE[0]}: steward proposed `draft`; the state machine wants "
    legacy = f"- #{DIS_AGREE[0]}: steward proposed `draft`; (the machine abstains)"
    lines = [legacy if ln.startswith(aged) else ln for ln in path.read_text().splitlines()]
    assert sum(1 for ln in lines if ln == legacy) == 1, (
        "the age had to land on exactly one filed row for the figure to mean anything")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    second = D.file_or_update(
        _agree(3119),
        DIS_MOVES + [{"item_id": 3119, "status": "draft",
                      "note": "the vec0 index needs a hand-side copy rebuild"}],
        {}, window=_window7(), backlog_dir=board, ledger=tmp_path / "ledger.jsonl")
    assert second is not None and second["action"] == "merged"

    body = path.read_text(encoding="utf-8")
    cov = body[body.index("## Coverage"):]
    row_lines = [ln for ln in body.splitlines() if D._ROW_RE.match(ln)]
    named = len(row_lines)
    with_side = sum(1 for ln in row_lines if "the state machine wants" in ln)
    assert named == 6 and with_side == 4, (
        f"the fixture itself moved: {named} rows, {with_side} naming a machine side")

    m = re.search(r"(\d+) of the (\d+) named rows carry the machine's side", cov)
    assert m is not None, f"the merged bullet lost its count:\n{cov}"
    assert int(m.group(2)) == named, (
        f"the denominator counts {m.group(2)} rows and the body holds {named}: "
        "the two figures stopped describing the same population")
    assert int(m.group(1)) == with_side, (
        f"the bullet claims {m.group(1)} rows carry a recorded side while {with_side} "
        f"of the body's rows print one — the numerator is being counted over this "
        f"filing's ids instead of over the rows the file holds:\n{cov}")
    assert "2 do not" in cov, "the shortfall must be the merged one, not this tick's"
    assert "**not recorded**" in body, "the widened row has to say its side is absent"
    assert "every row here can be ruled from the item alone" not in cov, (
        "two of the six rows have no machine half, so the merge may not claim "
        "full ruleability")


# ── #3914: a ruling that has landed must stop being printed as an open question ──
#
# The widening takes its ids from the window's per-tick history and asked nobody whether the
# item still disagreed, so six consecutive filings (#3752, #3766, #3784, #3791, #3858,
# #3913) re-named the same ids after commit `6119589c` had resolved four of them on
# 2026-11-05T11:47:50Z: `backlog/3858-*.md:152-155` and `backlog/3913-*.md:62-65` both
# still print `- #3433: steward proposed draft; the state machine wants up_next`, which by
# then was false. Measured 2026-11-07T21:13Z, 0 of the window's 11 distinct disagreeing ids
# still disagreed, and `below_floor` was still True on a trailing-7d `judged=151
# rate=0.8344` while the trailing 2d (0.9464) and 1d (0.9474) legs were above the floor —
# the sub-floor week being the Nov 1/4/5 ticks, every one of which has since been ruled.
#
# The five nodes below are the five clauses, and each is a SEEDED BODY: the 7-day window is
# only sub-floor because of ticks that age out by ~2026-11-12, after which `below_floor`
# reads False with no fix applied at all. A live trigger re-run proves nothing either way;
# the row-building path is the durable evidence.

#: The four ids `6119589c` (#3791) resolved. Each is on disk as `draft` while the row it
#: printed said the machine wanted `up_next` — and that commit settled them by changing
#: what the MACHINE answers (`draft`: its hand_run is still unexecuted), NOT by moving the
#: board to the recorded `up_next`. Which is why clause 2 cannot be folded into clause 1:
#: an abstention that only compares the row against the board would leave all four here.
RULED_FOUR = [3433, 3537, 3656, 3704]

#: What those four ticks scored as the machine's side, from the filed rows themselves.
RULED_SIDES = {i: ("up_next", "has a confirmed triage with acceptance text")
               for i in RULED_FOUR}

#: A live id: on disk `draft`, and the state machine still wants `up_next` for it, so it is
#: a disagreement after the fix as well as before it. The control in every node here.
LIVE_ID = 3310


def _ruled_tick(led: Path, ids, *, proposed: str = "draft",
                sides: dict | None = None, proposals: dict | None = None) -> None:
    """One `board_steward` tick whose proposals and recorded sides the rows will render.

    Both halves have to be in the ledger row: `moves` is the steward's half, and
    `agreement.disagree_sides` is the machine's (#3680) — a window id with no side renders
    **not recorded**, which would let a node pass on a row nobody could rule.
    `proposals` overrides `proposed` per id (#4405's nodes need a steward half that is not
    `draft`).
    """
    sides = sides if sides is not None else {}
    proposals = proposals or {}
    _led(led, event="board_steward", apply=False, created_at=DIS_OLDER_STAMP,
         moves=[{"item_id": i, "status": proposals.get(i, proposed),
                 "note": "unspent, offered again"} for i in ids],
         agreement={"disagree": list(ids), "judged": len(ids), "agree": [],
                    "disagree_sides": [{"item_id": i, "status": s, "why": w}
                                       for i, (s, w) in sides.items()]})


def _rule_filing(board, monkeypatch, led, *, win_ids, on_disk, machine,
                 tick_ids=DIS_AGREE, proposals=None, sides=None):
    """File one disagreement item over `win_ids`, then hand back its body.

    `on_disk` writes each item at the status the board holds NOW (the live half of the
    abstention), and `machine` is this test's stand-in for what `desired_statuses` answers
    NOW — stubbed rather than derived, because the state machine's live answer is an input
    to the guard, not the thing under test, and #3677 forbids a test that computes a row's
    half from a later instant.
    """
    from workers.board_disagreement import file_or_update

    Path(board).mkdir(parents=True, exist_ok=True)
    for iid, status in on_disk.items():
        _write(board, iid, status)
        assert (Path(board) / f"{iid}-item-{iid}.md").exists(), (
            f"#{iid} never reached the board, so the live read would find nothing to "
            f"abstain and the node would pass on an empty fixture")
    # #4378's persistence guard is cleared here on purpose: the preceding tick is written as
    # having disagreed about EVERY id this filing will name — the tick's as well as the
    # window's — so these nodes keep testing abstention and row provenance rather than
    # silently becoming guard tests. The guard's own nodes are below.
    sides = {i: (sides or {}).get(i) or RULED_SIDES.get(
                 i, ("up_next", "has a confirmed triage with acceptance text"))
             for i in list(tick_ids) + list(win_ids)}
    _ruled_tick(led, list(tick_ids) + list(win_ids), sides=sides, proposals=proposals)
    monkeypatch.setattr(B, "desired_statuses", lambda ledger: dict(machine))
    out = file_or_update(_agree(*tick_ids), DIS_MOVES, DIS_EXPECTED,
                         window=_window7(disagree_ids=list(tick_ids) + list(win_ids)),
                         backlog_dir=board, ledger=led)
    assert out is not None and out["action"] == "filed", out
    return _filed(board)[0].body


def _rows(body: str) -> list[str]:
    """Every row line the body prints, in file order."""
    return [ln for ln in body.splitlines() if D._ROW_RE.match(ln)]


def test_a_closed_item_prints_no_row_while_a_live_disagreement_still_prints(
        board, tmp_path, monkeypatch):
    """Clause 1. An id whose item is off the board is not an open question: #3418, #3456,
    #3508, #3570 and #3852 are `done`, so the machine has no opinion about them at all, and
    naming one as a disagreement spends a sweep and a triage pass on an item nobody can
    move. The live id in the same window must still print — a guard that removes everything
    would pass a test that only looks for the removed one."""
    led = tmp_path / "ledger.jsonl"
    body = _rule_filing(
        board, monkeypatch, led, win_ids=[RULED_FOUR[0], LIVE_ID],
        # #3433 closed, and the machine is stubbed to want `up_next` for it anyway, so the
        # only thing that can abstain it is the board fact — not a status comparison.
        on_disk={RULED_FOUR[0]: "done", LIVE_ID: "draft"},
        machine={RULED_FOUR[0]: ("up_next", "has a confirmed triage with acceptance text"),
                 LIVE_ID: ("up_next", "has a confirmed triage with acceptance text")})

    assert not [ln for ln in _rows(body) if ln.startswith(f"- #{RULED_FOUR[0]}")], (
        "a `done` item is still printing a disagreement row; the ruling landed and the "
        "filing is re-asking a question about an item nobody can move")
    live = [ln for ln in _rows(body) if ln.startswith(f"- #{LIVE_ID}")]
    assert len(live) == 1 and "the state machine wants `up_next`" in live[0], (
        f"the id that genuinely still disagrees must print with both halves: {live}")
    for i in DIS_AGREE:
        assert any(ln.startswith(f"- #{i}") for ln in _rows(body)), (
            f"tick id #{i} disappeared; only window ids are candidates for abstention")
    assert _row_count(body) == len(DIS_AGREE) + 1, (
        f"expected the five tick rows plus the one live id: {_rows(body)}")
    cov = body[body.index("## Coverage"):]
    assert f"#{RULED_FOUR[0]}" in cov, "the abstention has to name the id it removed"
    assert f"#{LIVE_ID}" not in cov, "the live id is not abstained and may not be listed"


def test_a_window_id_whose_status_now_matches_the_machine_is_abstained(
        board, tmp_path, monkeypatch):
    """Clause 2, and the case clause 1 cannot reach. All four ids of `RULED_FOUR` are on
    disk as `draft` while their rows record the machine wanting `up_next`, so the row and
    the board still disagree by the LETTER — and an abstention that compares the row's
    recorded side against the board leaves all four printing. `6119589c` settled them from
    the other side: it changed what `desired_statuses` answers (`draft`, its hand_run is
    still unexecuted), so only a read of the LIVE machine answer sees them.
    """
    led = tmp_path / "ledger.jsonl"
    on_disk = {i: "draft" for i in RULED_FOUR}
    on_disk[LIVE_ID] = "draft"
    machine = {i: ("draft", "its hand_run is still unexecuted") for i in RULED_FOUR}
    machine[LIVE_ID] = ("up_next", "has a confirmed triage with acceptance text")
    body = _rule_filing(board, monkeypatch, led, win_ids=RULED_FOUR + [LIVE_ID],
                        on_disk=on_disk, machine=machine)

    # The fixture has to stay the case, or this node quietly becomes clause 1 again: read
    # the two states back and confirm the row's recorded half still differs from the board.
    disk = {it.id: it.status for it in B.all_items(("lloyd",), backlog_dir=board)}
    rec = {r["item_id"]: r["status"]
           for e in S.read_events(S.LEDGER_PATH)
           if e.get("event") == "board_steward"
           for r in (e.get("agreement") or {}).get("disagree_sides") or []}
    assert all(disk[i] == "draft" and rec[i] == "up_next" for i in RULED_FOUR), (
        f"the board must hold `draft` while the recorded machine side is `up_next`; if "
        f"both say `draft` this node is testing clause 1's equality and clause 2 is "
        f"unpinned: {[(i, disk[i], rec[i]) for i in RULED_FOUR]}")
    assert all(f"- #{i}" not in body for i in RULED_FOUR), (
        "the four ids whose ruling landed are still named: the guard compared the row "
        "against its recorded side instead of asking the live table what the machine "
        "answers now")
    live = [ln for ln in _rows(body) if ln.startswith(f"- #{LIVE_ID}")]
    assert len(live) == 1, (
        f"#{LIVE_ID} is on disk at `draft` and the machine wants `up_next`: it still "
        f"disagrees and its row must print: {_rows(body)}")
    assert _row_count(body) == len(DIS_AGREE) + 1, _rows(body)
    cov = body[body.index("## Coverage"):]
    assert f"{len(RULED_FOUR)} of the window's ids were **abstained, not dropped**" in cov, cov
    assert cov.count("its status on disk (`draft`) is what the state machine answers "
                     "for it now") == len(RULED_FOUR), (
        f"all four must say which arm abstained them: {cov}")


# ── #4405: a row whose two sides now agree is retired, whatever the file says ──
#
# Post-flip `desired_statuses` abstains on an item the steward wrote and echoes its status,
# while the file moves on `reconcile_statuses`' own 900 s clock. #4394 read `draft` on disk
# for 74 minutes after its `steward_move` to `up_next` while the live answer was already
# `up_next`; the disk-equality arm saw `draft` != `up_next` and kept the row, and #4397 was
# filed on it alone. Seeded here in #4394's shape, since the live instance self-healed.

#: #4394's shape: the steward's recorded proposal, the tick's recorded machine side, and
#: what the machine answers NOW.
STEWARD_ECHO_ID = 4394
STEWARD_ECHO_SIDE = ("draft", "triage confirmed it; held until the implement pool has room")
STEWARD_ECHO_LIVE = ("up_next", "the board steward moved it to up_next; the machine abstains")


def test_a_window_id_the_machine_now_answers_as_the_steward_proposed_is_abstained(
        board, tmp_path, monkeypatch):
    """#4405 clause 1. File at `draft`, steward proposal `up_next`, live answer `up_next`:
    `ruled_ids` gives a reason for it instead of `{}`, and the filing stops printing its
    row. Without the row's steward side (the pre-#4405 call) the same read returns `{}`,
    which is the before half of the pair."""
    led = tmp_path / "ledger.jsonl"
    on_disk = {STEWARD_ECHO_ID: "draft", LIVE_ID: "draft"}
    machine = {STEWARD_ECHO_ID: STEWARD_ECHO_LIVE,
               LIVE_ID: ("up_next", "has a confirmed triage with acceptance text")}
    body = _rule_filing(board, monkeypatch, led, win_ids=[STEWARD_ECHO_ID, LIVE_ID],
                        on_disk=on_disk, machine=machine,
                        proposals={STEWARD_ECHO_ID: "up_next"},
                        sides={STEWARD_ECHO_ID: STEWARD_ECHO_SIDE})

    assert D.ruled_ids([STEWARD_ECHO_ID], backlog_dir=board, ledger=led) == {}, (
        "the fixture is not #4394's shape: the disk-equality arm already retires it")
    got = D.ruled_ids([STEWARD_ECHO_ID], backlog_dir=board, ledger=led,
                      proposed={STEWARD_ECHO_ID: "up_next"})
    assert got.get(STEWARD_ECHO_ID), (
        f"the machine now answers the steward's `up_next`, so the row's two sides agree "
        f"and the id is ruled; got {got}")
    assert "`up_next`" in got[STEWARD_ECHO_ID]
    assert STEWARD_ECHO_LIVE[1] not in got[STEWARD_ECHO_ID], (
        "the reason was taken from the live table's `why`; it must come from the row")

    assert not [ln for ln in _rows(body) if ln.startswith(f"- #{STEWARD_ECHO_ID}")], (
        f"#{STEWARD_ECHO_ID}'s row still prints although its steward side and the live "
        f"machine answer agree: {_rows(body)}")
    assert [ln for ln in _rows(body) if ln.startswith(f"- #{LIVE_ID}")], (
        "the genuinely live id must still print")


def test_a_row_whose_steward_side_differs_from_the_live_answer_still_prints(
        board, tmp_path, monkeypatch):
    """#4405 clause 2. The new arm abstains only where the row's steward side and the live
    answer agree: steward `up_next`, machine `in_progress`, file `draft` (equal to
    neither) is a live disagreement and prints."""
    led = tmp_path / "ledger.jsonl"
    body = _rule_filing(
        board, monkeypatch, led, win_ids=[STEWARD_ECHO_ID, LIVE_ID],
        on_disk={STEWARD_ECHO_ID: "draft", LIVE_ID: "draft"},
        machine={STEWARD_ECHO_ID: ("in_progress", "its round is not over"),
                 LIVE_ID: ("up_next", "has a confirmed triage with acceptance text")},
        proposals={STEWARD_ECHO_ID: "up_next"},
        sides={STEWARD_ECHO_ID: STEWARD_ECHO_SIDE})
    assert D.ruled_ids([STEWARD_ECHO_ID], backlog_dir=board, ledger=led,
                       proposed={STEWARD_ECHO_ID: "up_next"}) == {}
    row = [ln for ln in _rows(body) if ln.startswith(f"- #{STEWARD_ECHO_ID}")]
    assert len(row) == 1 and "steward proposed `up_next`" in row[0], (
        f"a row whose two sides still differ was abstained: {_rows(body)}")
    assert "- 0 of the window's ids were abstained as already ruled." in body, body


def test_the_steward_side_arm_only_subtracts_and_fails_open(board, tmp_path, monkeypatch):
    """#4405 clause 5. An id THIS tick scored is printed even in the new arm's shape, and an
    unreadable board or ledger returns `{}` so every row prints."""
    from workers.board_disagreement import file_or_update

    led = tmp_path / "ledger.jsonl"
    tick = DIS_AGREE[0]
    _write(board, tick, "draft")
    _ruled_tick(led, list(DIS_AGREE), proposals={tick: "up_next"},
                sides={i: STEWARD_ECHO_SIDE for i in DIS_AGREE})
    monkeypatch.setattr(B, "desired_statuses",
                        lambda ledger: {tick: STEWARD_ECHO_LIVE})
    assert tick in D.ruled_ids([tick], backlog_dir=board, ledger=led,
                               proposed={tick: "up_next"}), (
        "the fixture is not the case: the new arm must call this tick's id ruled")
    moves = [dict(m, status="up_next") if m["item_id"] == tick else m for m in DIS_MOVES]
    out = file_or_update(_agree(*DIS_AGREE), moves, DIS_EXPECTED,
                         window=_window7(disagree_ids=list(DIS_AGREE)),
                         backlog_dir=board, ledger=led)
    assert out is not None and out["action"] == "filed", out
    body = _filed(board)[0].body
    assert any(ln.startswith(f"- #{tick}") for ln in _rows(body)), (
        "an id this tick scored was dropped by the steward-side arm")

    def boom(ledger):
        raise OSError("ledger unreadable")
    monkeypatch.setattr(B, "desired_statuses", boom)
    assert D.ruled_ids([tick], backlog_dir=board, ledger=led,
                       proposed={tick: "up_next"}) == {}, (
        "an unreadable ledger must fail OPEN to `{}`, never retire a row")
    monkeypatch.setattr(B, "desired_statuses", lambda ledger: {tick: STEWARD_ECHO_LIVE})
    monkeypatch.setattr(B, "all_items", lambda *a, **kw: (_ for _ in ()).throw(
        OSError("board unreadable")))
    assert D.ruled_ids([tick], backlog_dir=board, ledger=led,
                       proposed={tick: "up_next"}) == {}, (
        "an unreadable board must fail OPEN to `{}`, never retire a row")


def test_coverage_states_the_abstention_figure_and_names_each_abstained_id(
        board, tmp_path, monkeypatch):
    """Clause 3. Removing a row is a claim about the window, and the Coverage section is
    where this filing already states what its rows do not cover. An id that vanished with
    no figure beside it is indistinguishable from a read that failed, which is the
    difference between abstaining and losing rows."""
    led = tmp_path / "ledger.jsonl"
    body = _rule_filing(
        board, monkeypatch, led, win_ids=[RULED_FOUR[0], RULED_FOUR[1], RULED_FOUR[2],
                                          LIVE_ID],
        on_disk={RULED_FOUR[0]: "done",          # off the board
                 RULED_FOUR[1]: "draft",         # equals the live machine answer
                 RULED_FOUR[2]: "draft",         # #4405: equals neither side
                 LIVE_ID: "draft"},
        # #4405's shape for RULED_FOUR[2]: the steward proposed `up_next`, its tick's
        # machine side was a pool hold at `draft`, and the live machine now echoes the
        # steward's `up_next` while the file still lags at `draft`.
        proposals={RULED_FOUR[2]: "up_next"},
        sides={RULED_FOUR[2]: ("draft", "triage confirmed it; held until the implement "
                                        "pool has room")},
        machine={RULED_FOUR[1]: ("draft", "its hand_run is still unexecuted"),
                 RULED_FOUR[2]: ("up_next", "the machine abstains until this item lands"),
                 LIVE_ID: ("up_next", "has a confirmed triage with acceptance text")})
    cov = body[body.index("## Coverage"):]

    assert "- 3 of the window's ids were **abstained, not dropped**" in cov, cov
    assert f"#{RULED_FOUR[0]} (its item is off the board (`done`))" in cov, cov
    assert f"#{RULED_FOUR[1]} (its status on disk (`draft`) is what the state machine " \
           "answers for it now)" in cov, cov
    # #4405: the steward-side arm names the RECORDED proposal and claims nothing about the
    # file, which is at `draft` and matches neither side.
    assert f"#{RULED_FOUR[2]} (the state machine now answers `up_next`, the status the " \
           "steward proposed in its row)" in cov, cov
    bullet = cov[cov.index(f"#{RULED_FOUR[2]} ("):]
    bullet = bullet[:bullet.index(")") + 1]
    assert "its status on disk" not in bullet and "draft" not in bullet, (
        f"the steward-side abstention borrowed the disk arm's claim: {bullet}")
    assert "the machine abstains until this item lands" not in body, (
        "the live table's `why` reached the body; the reason must come from the row")
    assert f"#{LIVE_ID}" not in cov, "a live id may not appear in the abstention list"
    # Beside the existing figures, not after them: the count above moves the `un` in the
    # "not represented here" bullet, so the two have to be read together.
    assert (cov.index("not represented here") < cov.index("abstained, not dropped")
            < cov.index("distinct items")), (
        f"the abstention bullet moved away from the figures it subtracts from:\n{cov}")

    # A filing with nothing to abstain says so, in the same slot, rather than going quiet:
    # the figure is what tells a reader the guard ran and found nothing.
    quiet = _rule_filing(
        tmp_path / "b2" / "backlog", monkeypatch, tmp_path / "b2-ledger.jsonl",
        win_ids=[LIVE_ID], on_disk={LIVE_ID: "draft"},
        machine={LIVE_ID: ("up_next", "has a confirmed triage with acceptance text")})
    assert "- 0 of the window's ids were abstained as already ruled." in quiet, quiet
    assert "abstained, not dropped" not in quiet, quiet


def test_the_live_read_only_removes_rows_and_never_supplies_one(board, tmp_path,
                                                                monkeypatch):
    """Clause 4. The live read is allowed to subtract and nothing else: an id THIS tick
    scored as a disagreement stays (dropping it would be the filer overruling its own
    measurement, which is #3752's job inside a tick), and every row that survives has to be
    the bytes this build already printed — no re-derived machine half, no `(the machine
    abstains)`, no reason string from the live table reaching the body."""
    led = tmp_path / "ledger.jsonl"
    sentinel = "LIVE-TABLE-WHY-3914-must-never-be-printed"
    on_disk = {i: "draft" for i in RULED_FOUR}
    on_disk.update({LIVE_ID: "draft", DIS_AGREE[0]: "up_next"})
    machine = {i: ("draft", sentinel) for i in RULED_FOUR}
    machine.update({LIVE_ID: ("up_next", sentinel),
                    DIS_AGREE[0]: ("up_next", sentinel)})
    # `DIS_AGREE[0]` is at the status the live table answers for it, so the guard itself
    # calls it ruled — and it is STILL printed, because this tick scored it.
    assert D.ruled_ids([DIS_AGREE[0]], backlog_dir=board,
                       ledger=led) == {}          # before the files exist
    body = _rule_filing(board, monkeypatch, led, win_ids=RULED_FOUR + [LIVE_ID],
                        on_disk=on_disk, machine=machine)
    assert DIS_AGREE[0] in D.ruled_ids([DIS_AGREE[0]], backlog_dir=board,
                                       ledger=led), (
        "the fixture is not the case: the live read has to consider this tick's id ruled, "
        "or printing it proves nothing about the guard's scope")

    rows = _rows(body)
    assert any(ln.startswith(f"- #{DIS_AGREE[0]}") for ln in rows), (
        "an id this tick scored in its own `disagree` was dropped by the live read")
    assert all(sentinel not in ln for ln in rows), (
        "a reason string from the live table reached a row: the read may only remove")
    assert sentinel not in body, "the live table's wording reached the body at all"
    assert "(the machine abstains)" not in body, (
        "#3677's sentence is back; it is only ever produced by a filing-time recompute")
    assert "**not recorded**" not in body, (
        "a row that survived the read lost the machine half its tick recorded")


def test_rows_that_survive_the_abstention_are_the_bytes_this_build_already_printed(
        tmp_path, monkeypatch):
    """Clause 4's byte half, run as a pair of filings rather than one.

    Two boards, the same tick, the same window, the same ledger rows — differing only in
    whether the board has moved under one id's ruling. Every row the abstaining filing
    prints must appear in the other filing's output byte for byte, and the difference must
    be exactly the one removed row: a re-derived machine half would show up as a surviving
    row that is not identical.
    """
    from workers.board_disagreement import file_or_update

    # Only #3433 and the live id are on the board at all; the rest of the window's ids have
    # no file either way, so the single difference between the two copies is whether the
    # board moved under #3433's ruling.
    win = [RULED_FOUR[0], LIVE_ID]
    machine = {RULED_FOUR[0]: ("draft", "its hand_run is still unexecuted"),
               LIVE_ID: ("up_next", "has a confirmed triage with acceptance text")}
    bodies = {}
    for tag, disk in (("ruled", "draft"), ("live", "up_next")):
        d = tmp_path / tag
        d.mkdir()
        led = tmp_path / f"{tag}-ledger.jsonl"
        monkeypatch.setattr(B, "BACKLOG_DIR", d)
        monkeypatch.setattr(S, "LEDGER_PATH", led)
        _write(d, RULED_FOUR[0], disk)
        _write(d, LIVE_ID, "draft")
        # #4405: the steward's half is `up_next`, not `_ruled_tick`'s default `draft`. A
        # `draft` proposal equals the live machine answer below, and the steward-side arm
        # would then retire the row on BOTH boards, whatever the file says; `up_next` keeps
        # the board's move the single difference between the two copies.
        _ruled_tick(led, win, proposals={RULED_FOUR[0]: "up_next"},
                    sides={RULED_FOUR[0]: ("draft", "its hand_run is still unexecuted"),
                           LIVE_ID: ("up_next", "has a confirmed triage")})
        monkeypatch.setattr(B, "desired_statuses", lambda ledger: dict(machine))
        out = file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                             window=_window7(disagree_ids=list(DIS_AGREE) + list(win)),
                             backlog_dir=d, ledger=led)
        assert out is not None and out["action"] == "filed", (tag, out)
        bodies[tag] = _filed(d)[0].body

    cut, kept = _rows(bodies["ruled"]), _rows(bodies["live"])
    assert len(cut) == len(kept) - 1, (
        f"the live read removed {len(kept) - len(cut)} rows and exactly one was abstained:"
        f"\ncut={cut}\nkept={kept}")
    removed = [ln for ln in kept if ln not in cut]
    assert len(removed) == 1 and removed[0].startswith(f"- #{RULED_FOUR[0]}"), (
        f"the row that disappeared is not the ruled one: {removed}")
    assert all(ln in kept for ln in cut), (
        "a surviving row changed its bytes, which is a half being re-derived: "
        f"{[ln for ln in cut if ln not in kept]}")
    assert f"- #{LIVE_ID}" in bodies["live"] and f"- #{LIVE_ID}" in bodies["ruled"]
    assert "- 1 of the window's ids were **abstained, not dropped**" in bodies["ruled"]
    assert "- 0 of the window's ids were abstained as already ruled." in bodies["live"]


def test_the_abstention_moves_no_aggregate_and_no_below_floor_verdict(
        board, tmp_path, monkeypatch):
    """Clause 5. Whether the aggregate should also stop counting ticks whose disagreement
    was ruled is owed-check's to rule (#3089 asked, #3329 set the bar), so this change may
    not touch a single figure in it: `below_floor`'s verdict on a given window dict is the
    same with five rows abstained as with none, and `judged`, `rate` and the window's own
    row and item counts are the same numbers the body printed before.
    """
    import copy as _copy

    win = _window7(disagree_ids=list(DIS_AGREE) + RULED_FOUR + [LIVE_ID])
    held = _copy.deepcopy(win)
    assert D.below_floor(win)[0] is True, "the fixture window has to trip the trigger"

    body = _rule_filing(
        board, monkeypatch, tmp_path / "ledger.jsonl",
        win_ids=RULED_FOUR + [LIVE_ID],
        on_disk={**{i: "draft" for i in RULED_FOUR}, LIVE_ID: "draft"},
        machine={**{i: ("draft", "its hand_run is still unexecuted") for i in RULED_FOUR},
                 LIVE_ID: ("up_next", "has a confirmed triage with acceptance text")})

    assert len([ln for ln in _rows(body) if "**abstained" in ln]) == 0
    assert "4 of the window's ids were **abstained, not dropped**" in body, body
    assert win == held, "the abstention mutated the window dict it was handed"
    assert D.below_floor(win) == D.below_floor(held), (
        "the same window dict reads a different verdict after a filing")
    assert D.below_floor(_window7())[0] is True
    assert (f"{win['judged']} judged, `rate={win['rate']}`" in body
            or f"{win['judged']} judged" in body and f"rate={win['rate']}" in body), (
        f"Coverage stopped citing the window it was handed: {body[body.index('## Coverage'):]}")
    distinct = len(set(win["disagree_ids"]))
    assert f"are {distinct} distinct items" in body, (
        f"the window's distinct-item figure moved with the abstention (it counts the "
        f"window, not the rows this filing wrote): expected {distinct}")


def test_a_merge_counts_the_rows_an_earlier_filing_recorded(board, tmp_path):
    """The other direction of the same figure: rows a PREVIOUS filing wrote count
    towards the covered total, or the count is only ever pessimistic and the merged
    item can never say it is whole.

    Six ids, five recorded by the first filing and the sixth by this one, so the merged
    bullet has to read six of six — which it can only do by carrying the earlier rows
    over, since `expected` here holds the widened id alone.
    """
    first = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                             window=_window7(), backlog_dir=board,
                             ledger=tmp_path / "ledger.jsonl")
    assert first is not None and first["action"] == "filed"

    second = D.file_or_update(
        _agree(3119),
        DIS_MOVES + [{"item_id": 3119, "status": "draft",
                      "note": "the vec0 index needs a hand-side copy rebuild"}],
        {3119: ("up_next", "open at `up_next`, no branch, never worked")},
        window=_window7(), backlog_dir=board, ledger=tmp_path / "ledger.jsonl")
    assert second is not None and second["action"] == "merged"

    cov = _filed(board)[0].path.read_text(encoding="utf-8")
    cov = cov[cov.index("## Coverage"):]
    assert "6 of the 6 named rows carry the machine's side" in cov, (
        f"the five rows already in the body were not counted:\n{cov}")
    assert ("as its own tick recorded it. Both halves of every row are in the artifact, "
            "so every row here can be ruled from the item alone.") in cov, (
        "every row the merged item names prints both halves, and the bullet has to say "
        "so — that sentence is what a ruler reads to know the item needs no ledger")
    assert "do not:" not in cov, "nothing is short, so no shortfall sentence may appear"


def test_a_ruling_written_on_the_item_while_a_merge_computes_survives_it(board,
                                                                        tmp_path,
                                                                        monkeypatch):
    """Clause 4: the merge appends to what the file said UNDER ITS LOCK.

    `_write_body` re-reads the file under `commit_lock` into `fm_now, _body_now` and then
    writes `body` — the text built from the read taken BEFORE the lock, across a window
    long enough to matter (`S.read_events` over an 18,479-row ledger plus
    `desired_statuses`). `_body_now` was assigned and never used, and the comment three
    lines above the write said "keeping their text and adding the rows is the safe
    answer". This node fires a human edit inside the lock, which is the exact instant the
    pre-lock read goes stale: before the fix the ruling is erased while the commit claims
    to have kept it, and the thing erased is a judgment about which side of a
    disagreement was wrong — the one thing on this item that cannot be regenerated.
    """
    D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED, window=_window7(),
                     backlog_dir=board, ledger=tmp_path / "ledger.jsonl")
    it = _filed(board)[0]
    ruling = ("## Ruling\n\n"
              "- #3024: the steward is wrong. Its own note says \"no triage verdict in "
              "its history\" and its history has one.\n")
    real_lock = B.commit_lock

    # `_write_body` imports `commit_lock` from `app.atomic_io` INSIDE the function
    # (function-local imports are that module's rule), so the patch belongs on the
    # module it comes from and not on `B`: patched on the wrong one the fake never
    # runs, the ruling is never written at all, and the node fails for a reason that
    # has nothing to do with the code under test.
    from app import atomic_io

    real_lock = atomic_io.commit_lock

    def racy_commit_lock(target, timeout=None):
        """The real lock, with a person's edit landing once it is held.

        Yields the real lock's own token, because `_write_body` hands that token to
        `BW.write_item(..., lock=...)` — substituting a stand-in would test a call the
        code never makes.
        """
        inner = real_lock(target, timeout=timeout)

        class _Racy:
            def __enter__(self):
                token = inner.__enter__()
                it.path.write_text(it.path.read_text(encoding="utf-8").replace(
                    "## What to decide", ruling + "\n## What to decide"),
                    encoding="utf-8")
                return token

            def __exit__(self, *exc):
                return inner.__exit__(*exc)

        return _Racy()

    monkeypatch.setattr(atomic_io, "commit_lock", racy_commit_lock)
    moves = DIS_MOVES + [{"item_id": 3119, "status": "draft",
                          "note": "the vec0 index needs a hand-side copy rebuild"}]
    want = dict(DIS_EXPECTED)
    want[3119] = ("up_next", "in_progress by no round, ready by its acceptance")
    assert D.file_or_update(_agree(3119), moves, want, window=_window7(),
                            backlog_dir=board,
                            ledger=tmp_path / "ledger.jsonl") is not None
    after = it.path.read_text(encoding="utf-8")
    assert "#3024: the steward is wrong" in after, (
        "the merge overwrote a ruling written while it was computing — the item is the "
        "record of who decided which side was wrong, and the overwrite is silent")
    assert "#3119" in after, "keeping the ruling must not cost the merge its own row"
    assert after.count("## Coverage") == 1, (
        "the re-read re-introduced the stale Coverage section, so the file now carries "
        "two and a reader cannot tell which is current")
    assert "(the machine abstains)" not in after


# ---------------------------------------------------------------------------
# #3683 — a tick that held the item its own re-offer had just returned is not a
# disagreement: the state machine now agrees with the round state it rendered
# ---------------------------------------------------------------------------

def test_a_tick_holding_an_item_with_a_live_round_records_no_disagreement(board, monkeypatch,
                                                                         tmp_path):
    """Clause 5. The steward's half of the #3683 class, run through the SAME
    `agreement()` that scored the four 2026-11-01 ticks and wrote their `disagree` rows
    (#3418 02:16:27Z, #3456 14:29:42Z, #3508 19:25:00Z and 19:29:20Z).

    The fixture is the machine's own shape: an implement turn that died at
    `turn_timeout` with its round's worktree standing, no promotion and no gate row, so
    `implement_outcomes` reads `incomplete` and the board line renders
    `round_state=worktree`. The steward, following the unconditional rule in its prompt
    (while any round state holds, the item stays `in_progress` and is NEVER proposed
    `up_next`), proposes `in_progress`. Before the machine-side fix `expected` answered
    `up_next` / `offered again — incomplete`, and that is what was scored — the control
    below hands `agreement()` exactly that `expected` and shows the row was real, so this
    node cannot pass by the metric having been edited.

    `rate` and `judged` are asserted to be `agreement()`'s own arithmetic
    (`len(agree) / judged`, asserted below). Neither that function nor
    `workers.dispatch_watch.judgement_aggregate` is touched by this change: the fix moves
    the number only by changing what the state machine answers, never how the answer is
    measured. And the ceiling stays what the item says it is — clearing those 4 of the
    window's 121 judged gives 99/121 = 0.8182, still under the 0.90 floor — so a passing
    node here is a partial move, not a crossing.
    """
    from scripts.automod import worktree as WT
    _write(board, 3418, "in_progress")
    _ev(event="backlog_implement", item_id=3418, phase="started", ts=2_000.0)
    _ev(event="backlog_implement", item_id=3418, phase="finished",
        round_id="SM_20261101_011407", session_id="s-3418",
        stop_reason="turn_timeout", num_turns=250)
    wt = tmp_path / "wt"
    wt.mkdir(exist_ok=True)
    monkeypatch.setattr(WT, "worktree_path", lambda rid: wt)
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "gate_in_progress", lambda rid: False)

    assert B.implement_outcomes(S.LEDGER_PATH)[3418][0] == "incomplete"
    expected = B.desired_statuses(S.LEDGER_PATH)
    assert expected[3418][0] == "in_progress", \
        f"the machine re-offered a round the board line calls live: {expected[3418]}"

    a = W.agreement(moves=[{"item_id": 3418, "status": "in_progress"}],
                    expected=expected, current={3418: "in_progress"})
    assert a["disagree"] == [] and a["disagree_sides"] == [], \
        f"the tick still records the machine contradicting its own round_state: {a}"
    assert a["agree"] == [3418] and a["missed"] == []
    assert a["judged"] == 1 and a["rate"] == pytest.approx(len(a["agree"]) / a["judged"]), \
        "the tick's rate stopped being agreement()'s own arithmetic"

    pre_fix = {3418: ("up_next", "offered again — incomplete: the turn ran out of clock "
                                "before reaching a verdict; its work is on branch "
                                "`automod/SM_20261101_011407`")}
    scored = W.agreement(moves=[{"item_id": 3418, "status": "in_progress"}],
                         expected=pre_fix, current={3418: "in_progress"})
    assert scored["disagree"] == [3418] and scored["rate"] == 0.0, \
        "the pre-fix table does not even reproduce the row this node closes"
    assert scored["disagree_sides"] == [{"item_id": 3418, "status": "up_next",
                                        "why": pre_fix[3418][1]}]


# ─────────── #3547: the machine's live round state as ONE named flag ───────────
#
# The figures are the ones the item was filed on, measured 2026-10-29T17:54Z at
# HEAD `752cca1a`: `GET /api/workers/health?days=7` reported for board-steward
# judged 119, agree 90, disagree 28, rate 0.7563, and an independent read-only
# recount of `runs.meta_json.agreement` over ~/lloyd-data/workers.db found those
# same 28 disagreeing rows across 25 distinct ids. The item's own headline cites
# 26 / 0.7759, which was the reading when it was written; both readings sit under
# the 0.90 the `apply` flip waits for, and 28 is the number quoted below because it
# is the one a second command reproduces.
#
# What the prompt already says about a round's FIVE states is #3546's shipped work
# and is not re-tested here. What was missing is the one field that says, per shown
# item, whether a round is live at all — and the machine's own refusal of a move
# that contradicts it.

STATUS_WORDS = ("draft", "up_next", "in_progress", "done")


def _load(board, iid: int):
    """The item as its front matter now reads on disk."""
    hits = list(board.glob(f"{iid}-*.md"))
    assert len(hits) == 1, f"{iid} matched {len(hits)} files"
    return B.load_item(hits[0])


def _line(prompt: str, iid: int) -> str:
    hits = [l for l in prompt.splitlines() if l.startswith(f"#{iid} ")]
    assert len(hits) == 1, f"item {iid} rendered {len(hits)} board lines"
    return hits[0]


def _flags(prompt: str) -> dict[int, str]:
    out = {}
    for line in prompt.splitlines():
        m = re.match(r"#(\d+) ", line)
        if not m:
            continue
        got = re.findall(r"\bround_flag=(\S+)", line)
        assert len(got) == 1, f"{line}\nexpected exactly one round_flag=, found {len(got)}"
        out[int(m.group(1))] = got[0]
    return out


#: The destination words a board line must never print: the four statuses an item's
#: front matter takes (`draft`, `up_next`, `in_progress`, `done`) plus `ready`, which
#: is how `board_health` counts a takeable implement-pool item. #3329's
#: no-answer-sheet rail says `triage=` names an OBSERVATION about the ledger and never
#: what `desired_statuses` wants, because the tick's proposal is scored against exactly
#: that table.
_STATUS_WORDS = ("draft", "up_next", "in_progress", "done", "ready")


def _board_lines(prompt: str) -> list[str]:
    """Every board line in a rendered prompt — the ONE pattern for it here.

    A board line is `#<id> [status] …` (`_item_line`'s f-string), so this is the same
    prefix `_line` and `_flags` match on. `^#` with no leading dash on purpose: no
    rendered line begins `- #`, event lines open with a `%m-%d %H:%M UTC` stamp and
    health lines are prose, which is why a `- #` guard silently selects nothing.
    """
    return [l for l in prompt.splitlines() if re.match(r"#\d+ ", l)]


def _assert_no_line_names_a_destination(prompt: str, n_items: int) -> None:
    """#3329's rail, swept over every board line, with the sweep's own reach pinned.

    Two count assertions, not one property: a filter that stops matching empties the
    loop and the property then passes over nothing, which is exactly how the first
    commit's guard read — the review rung caught it by checking whether the body could
    run, not by reading its intent. So: the sweep must find one line per shown item, and
    one `triage=` per line, before the status check means anything.
    """
    lines = _board_lines(prompt)
    assert len(lines) == n_items, (
        f"the sweep matched {len(lines)} board lines for {n_items} shown items, so it "
        f"is not looking at them: {prompt[:300]!r}")
    checked = 0
    for line in lines:
        for v in re.findall(r"\btriage=(\S+)", line):
            checked += 1
            assert v not in _STATUS_WORDS, (
                f"the triage field named a status, which is the answer sheet: {line}")
    assert checked == n_items, (
        f"{n_items} items shown but {checked} `triage=` values found: a board line lost "
        "its verdict field, so this sweep is not sweeping")


def test_every_shown_item_s_board_line_carries_one_derived_round_flag(board, monkeypatch):
    """Clause 1. Four items, four ledger situations, one `round_flag=` each.

    The ledger is written through `S.append_event` on the fixture's
    `S.LEDGER_PATH` — the same file, with the same row shapes, that
    `desired_statuses` reads — and the flag is derived by the same
    `unfinished_rounds` call that supplies #3546's `round_state=` field:

    The four states are the ones `unfinished_rounds` is verified to produce, written
    the way the #3546 node writes them (`_promotion` for the promotion pair,
    `gate_in_progress` for a running gate, an absent worktree everywhere else):

      #7   `backlog_implement`/`phase=finished` whose round has a `promoted` row no
           `rollback_succeeded` reverted, no `settled` row — state `promoted`, the
           `observing` set, the promotion under observation → `observation_live`.
           This is #3518's and #3524's situation at the #3546 filing tick.
      #8   a finished implement whose round has a live gate process — state `gate`,
           an unfinished round → `round_in_flight`
      #9   a round that finished and promoted nothing → `none`
      #10  `promoted`, `settled`, and the sweep has written `item_landed`: the
           `closing` interval is over → `none`

    And a flag may not contain a status word. #3547's triage names the hazard: print
    the machine's target status and the model copies it, so `rate` climbs with no
    more correct judgements — and a flag that read `in_progress` would be exactly
    that leak, because `in_progress` is also a status the steward may propose.
    """
    from scripts.automod import backlog as B, state as S

    for iid, st in ((7, "in_progress"), (8, "in_progress"), (9, "up_next"),
                    (10, "done")):
        _write(board, iid, st)
    led = S.LEDGER_PATH           # the board fixture points this at tmp_path
    _promotion(led, 7, "R7", "a" * 40, settle=False, base=1)
    _led(led, event="backlog_implement", item_id=8, phase="finished",
         round_id="SM_GATE", ts=20)
    _led(led, event="backlog_implement", item_id=9, phase="finished",
         round_id="SM_OVER", ts=30)
    _promotion(led, 10, "R10", "b" * 40, settle=True, base=40)
    _led(led, event="item_landed", round_id="R10", item_id=10, ts=44)
    monkeypatch.setattr("scripts.automod.worktree.worktree_path",
                        lambda rid: board.parent / "no-such-worktree")
    monkeypatch.setattr(S, "gate_in_progress",
                        lambda rid: {"pid": 4242} if rid == "SM_GATE" else None)
    monkeypatch.setattr(S, "land_in_progress", lambda rid: None)

    ids = {7, 8, 9, 10}
    prompt = W.build_prompt(events=[], items=[_item(i, "in_progress") for i in sorted(ids)],
                           n_open=len(ids), since_ts=0,
                           round_state=W.unfinished_rounds(S.LEDGER_PATH, ids))
    flags = _flags(prompt)
    assert flags == {7: "observation_live", 8: "round_in_flight",
                     9: "none", 10: "none"}, flags
    for iid, value in flags.items():
        assert value in W.ROUND_FLAG_VALUES, (iid, value)
        assert not [w for w in STATUS_WORDS if w in value], (iid, value)
    # One ledger, one read: the rows that put #7's flag at `observation_live` are the
    # rows that make the state machine hold the same item. Asserted against
    # `desired_statuses` itself rather than asserted in prose.
    assert B.desired_statuses(S.LEDGER_PATH)[7][0] == "in_progress", \
        "the flag is non-none on a ledger the state machine does not hold for"
    # #3546's finer field is unchanged: still only on a line whose round is
    # unfinished, and never on the settled one.
    assert "round_state=" in _line(prompt, 8)
    assert "round_state=" not in _line(prompt, 10), \
        "the settled promotion gained #3546's field, which says it is unsettled"


def test_the_flag_projection_answers_none_only_for_an_item_with_no_live_round():
    """Clause 1's vocabulary: three values, and the projection fails toward holding.

    A state outside both named sets answers `round_in_flight`, never `none`. The
    direction is the point: `none` is the value that licenses `up_next` and `draft`,
    so a sixth state `ROUND_STATES` may later grow must fail toward holding the item.
    An entry is taken as `(round_id, states)` — handing the function a bare states
    tuple would read the round id as a state and answer `round_in_flight` for a
    promotion under observation, which is the wrong direction on the one case the
    filed rows are about.
    """
    assert W.ROUND_FLAG_VALUES == ("observation_live", "round_in_flight", "none")
    assert W.ROUND_STATES == ("promoted", "closing", "worktree", "gate", "landing")
    for value in W.ROUND_FLAG_VALUES:
        assert not [w for w in STATUS_WORDS if w in value], value
    assert W.round_flag(None) == "none"
    for state in ("promoted", "closing"):
        assert W.round_flag(("R1", (state,))) == "observation_live", state
    for state in ("worktree", "gate", "landing"):
        assert W.round_flag(("R1", (state,))) == "round_in_flight", state
    assert W.round_flag(("R1", ("promoted", "gate"))) == "round_in_flight", \
        "a live process and an unsettled promotion both hold; the process is named"
    assert W.round_flag(("R1", ("invented_state",))) == "round_in_flight", \
        "an unmapped state must not release the item"
    # The flag is a projection of the entry, so the line and any gate on it cannot
    # disagree about whether a round is live.
def test_a_shown_item_with_no_live_round_still_asks_its_flag():
    """The line cost of clause 1, printed instead of claimed in a docstring.

    `_item_line`'s comment says an item with no live round pays 16 characters and a
    line under observation pays 28. Both figures are measured here off the one
    difference between the two renderings, because a byte count quoted from prose is
    the kind of figure this tree has already been burned on.

    The 16 is what #3546's "a None renders the line exactly as it read before" gave
    up: an absent entry now answers `none` on the line instead of answering nothing,
    so a rule can read the field on every item rather than infer from its absence.
    """
    plain = W._item_line(_item(7, "up_next"))
    live = W._item_line(_item(7, "up_next"), round_state=("R7", ("promoted",)))
    both = W._item_line(_item(7, "in_progress"), round_state=("R7", ("gate",)))
    assert "round_flag=none" in plain
    assert "round_flag=observation_live" in live
    assert "round_flag=round_in_flight" in both
    for line in (plain, live, both):
        assert line.count("round_flag=") == 1, line
    assert len(" round_flag=none") == 16, len(" round_flag=none")
    assert len(" round_flag=observation_live") == 28, len(" round_flag=observation_live")
    # What the 16 characters bought, and what they did not: the fine-grained field is
    # still absent where nothing is unfinished, so a `none` line pays for the flag and
    # for nothing else — #3546's rendering is unchanged below it.
    assert "round_state=" not in plain and "round=" not in plain.replace("round_flag=", "")
    assert "round=R7 round_state=promoted" in live


def test_a_move_the_flag_refuses_is_never_written_yet_still_scored(board, monkeypatch):
    """Clause 2 end to end: the flag gates the WRITE, and never the SCORE.

    One tick, `apply: true`, three moves, ruled on three ways by the flag:

      #7  on the board at `up_next`, its promotion unsettled so its flag is
          `observation_live`, and the steward proposes `up_next` — leave it there.
          The machine holds `in_progress` (`observing`), so the proposal
          contradicts the flag: it must not reach the board. #3518 and #3524 are
          the rows this is for.
      #8  flag `none`, machine wants `draft` (confirmed with an empty acceptance).
          Its move is honoured and written — the control that proves #7 and #9 were
          stopped by their flags and not by something switched off wholesale.
      #9  its `SM_9` promotion unsettled so its flag is `observation_live`, the
          machine holds it at `up_next` where it already sits, and the steward
          proposes `draft`. Suppressed, and the one that makes the score's shape
          measurable — see the rate paragraph.

    Then the score, which is the rest of the clause. `agreement()` is called on the
    moves the steward PROPOSED, so #7 and #9 both land in `disagree`, `judged` is 3
    and `rate` is one third. Re-running `agreement()` on ONLY the writable move —
    what filtering the list before the call would have scored — gives `disagree`
    EMPTY, `disagree_sides` empty and `rate` 0.5: higher, on a smaller denominator,
    for an identical steward answer. That is the inflation #3547's triage names, and
    it is a rate the `apply` flip reads (judged >= 50 AND rate >= 0.90). The two
    numbers below are computed by the same `agreement()` over the same
    `expected`/`current`, so the ORDERING of the gate is pinned by arithmetic rather
    than by a comment claiming it. `judged` is also reported: 3 against 2, because
    dropping a proposal removes the decision it was a judgement about.
    """
    from workers.sources import _common as C
    from scripts.automod import backlog as B, state as S
    from workers import board_disagreement as D

    _ten_item_board(board)
    led = S.LEDGER_PATH
    _promotion(led, 7, "R7", "a" * 40, settle=False, base=1)
    # #9's round is the `SM_9` one `_ten_item_board` already finished. A `promoted`
    # row with no `settled` after it is #3518's and #3524's shape exactly.
    _led(led, event="promoted", round_id="SM_9", commit="c" * 40, ts=99)
    monkeypatch.setattr("scripts.automod.worktree.worktree_path",
                        lambda rid: board.parent / "no-such-worktree")
    raw = [{"item_id": 7, "status": "up_next", "note": "stale, needs a human"},
           {"item_id": 8, "status": "draft", "note": "never triaged"},
           {"item_id": 9, "status": "draft", "note": "umbrella, re-offer it"}]
    expected = {7: ("in_progress", "promotion under observation"),
                8: ("draft", "confirmed with no acceptance"),
                9: ("up_next", "already in the pool, and staying")}
    current = {7: "up_next", 8: "up_next", 9: "up_next"}
    parsed = W.parse_steward({"moves": raw})
    monkeypatch.setattr(B, "desired_statuses", lambda *a, **k: dict(expected))
    monkeypatch.setattr(D, "window_now", lambda **kw: _window7())

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": raw, "next_pick": 0, "next_pick_reason": "",
                               "summary": "net 0 today"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": True})))

    assert out["status"] == "success", out
    assert _load(board, 8).status == "draft", "the none-flagged move was not written"
    for iid in (7, 9):
        assert _load(board, iid).status == "up_next", \
            f"#{iid}'s front matter was rewritten on the strength of a live round"
    row = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "board_steward"][-1]
    assert [m["item_id"] for m in row["moves"]] == [7, 8, 9], "the row lost a proposal"
    assert [m["item_id"] for m in row["suppressed"]] == [7, 9]
    assert [m["item_id"] for m in row["applied"]] == [8]
    assert "2 refused by round_flag" in out["summary"], out["summary"]

    agree = row["agreement"]
    assert agree["disagree"] == [7, 9] and agree["agree"] == [8]
    assert agree["judged"] == 3 and agree["rate"] == pytest.approx(1 / 3)
    assert agree == W.agreement(parsed["moves"], expected, current), \
        "the recorded score is not the score of the steward's own answer"
    clamped = W.agreement([m for m in parsed["moves"] if m["item_id"] == 8],
                          expected, current)
    assert clamped["disagree"] == [] and clamped["disagree_sides"] == [], \
        "pre-clamping left the disagreement in the record — it was supposed to"
    assert clamped["judged"] == 2 and clamped["rate"] == pytest.approx(0.5)
    assert clamped["rate"] > agree["rate"], \
        "filtering before the score must be the worse-sounding option, or the rail " \
        "this node stands on is not the one the item describes"
    # #3677: `disagree_sides` is the MACHINE's side of each disagreement — the
    # status `expected` wanted and why — recorded as the tick scored it.
    assert [d["item_id"] for d in agree["disagree_sides"]] == [7, 9]
    assert agree["disagree_sides"][0] == {"item_id": 7, "status": "in_progress",
                                          "why": "promotion under observation"}


def test_the_gate_refuses_a_demotion_and_a_leave_it_alone_but_still_allows_a_claim():
    """Clause 2's shape, on both live values and with the controls in the same loop.

    `up_next` and `draft` are the two destinations the filed rows show. Leaving an
    item where it sits is the third and the one a prompt that says "no move is also
    an answer" invites: on an item the machine holds at `in_progress`, a proposed
    `up_next` is the machine's no-op and the steward's claim that the round is over
    — so it is refused on the same terms as a demotion. The `none` rows are the
    positive control that the gate is not a freeze, and a live item stays claimable.
    """
    def move(iid, status):
        return {"item_id": iid, "status": status, "note": "n",
                "tags_add": [], "tags_remove": []}

    for flag in ("observation_live", "round_in_flight"):
        live = {7: flag}
        for status in ("up_next", "draft"):
            writable, suppressed = W.split_writable([move(7, status)], live)
            assert writable == [] and [m["item_id"] for m in suppressed] == [7], (flag, status)
        writable, suppressed = W.split_writable([move(7, "in_progress")], live)
        assert len(writable) == 1 and suppressed == [], f"{flag}: a live round must be claimable"
    none = {7: "none"}
    for status in ("up_next", "draft", "in_progress"):
        writable, suppressed = W.split_writable([move(7, status)], none)
        assert len(writable) == 1 and suppressed == [], status
    # An item the flag map does not name is `none`, not a refusal: an item the
    # machine never looked at is not evidence of a live round.
    assert W.split_writable([move(99, "draft")], {})[1] == []
    # Two equal moves for one item are both refused — the partition is by position,
    # and an equality filter would have dropped only one of a duplicated pair.
    writable, suppressed = W.split_writable([move(7, "draft"), move(7, "draft")],
                                            {7: "observation_live", 8: "none"})
    assert writable == [] and len(suppressed) == 2


def test_the_prompt_names_the_three_flag_values_and_refuses_the_move_they_block():
    """Clause 3: the shipped prompt states the flag's vocabulary and the refusal.

    Asserted inside the one bullet the flag owns — cut at the next bullet, so the
    vocabulary cannot be satisfied by an unrelated sentence elsewhere — except the
    third value, which is a value and so is asserted on the prompt as a whole: a
    test that demanded `none` inside the bullet would be asking for the word "none"
    to appear where the flag is defined, which is what the assertion then checks.
    """
    text = W.PROMPT
    assert "- **`round_flag=`" in text, "the field is never named to the steward"
    start = text.index("- **`round_flag=`")
    nxt = text.find("\n- ", start + 5)
    bullet = text[start:nxt if nxt > 0 else len(text)]
    for value in ("observation_live", "round_in_flight"):
        assert f"`{value}`" in bullet, value
    assert "`none`" in bullet, "the flag's third value is undefined prose"
    assert "**`in_progress` and nothing else**" in bullet
    for phrase in ("refused by the machine", "scored against you"):
        assert phrase in bullet, phrase
    # Nothing in the bullet may state a target status as the flag's own value: the
    # words are in the sentence because the sentence names the move that is refused,
    # so the check is on the flag's definition, which is the three backticked values.
    assert [v for v in W.ROUND_FLAG_VALUES if f"`{v}`" in bullet] == list(W.ROUND_FLAG_VALUES)


def test_the_architecture_page_names_the_flag_its_ledger_source_and_the_unchanged_bar():
    """Clause 4: `architecture/workers-jobs.md` §board-steward documents the flag.

    The flip bar is asserted as both of its numbers, so a page that dropped one or
    moved either cannot pass; `≥` rather than `>=` follows what this page already
    prints. The retired 100 floor is asserted absent for the same reason the file's
    own banned-string node asserts it tree-wide.
    """
    doc = (Path(__file__).resolve().parents[1]
           / "architecture" / "workers-jobs.md").read_text(encoding="utf-8")
    start = doc.index("### `board-steward`")
    nxt = doc.find("\n### ", start + 5)
    sec = doc[start:nxt if nxt > 0 else len(doc)]
    assert "`round_flag=`" in sec
    for value in W.ROUND_FLAG_VALUES:
        assert f"`{value}`" in sec, value
    for source in ("desired_statuses", "items_with_unfinished_rounds", "promoted"):
        assert f"`{source}`" in sec, f"the flag's ledger source {source} is not named"
    assert "≥ 50" in sec and "≥ 0.90" in sec, "both legs of the flip bar must be stated"
    assert not re.search(r"judged[`'\"\\u201d\\u2019]{0,2}\s*(?:>=|≥)\s*100(?![\d.])", sec)


# ===========================================================================
# The row that makes `apply: true` safe to flip (#3467)
#
# Before this, the only trace of a move the steward WROTE was the activity line in
# the item's own front matter — `**ts** — up_next → draft: board steward: <note>`,
# written by `app/backlog_move.py` and parsed by nobody: before these nodes,
# `git grep -n "board steward:" -- '*.py'` returned this module's writer and four
# logger strings, and no ledger reader. So
# `desired_statuses` could not tell a move the steward wrote from a move it merely
# suggested, and `reconcile_statuses` reverted the steward's park on its very next
# poll (900 s) while its own `status_moved` row recorded the revert as a machine
# move. These nodes pin the WRITER half: one `steward_move` row per move that
# actually changed an item. The reader half — the abstention that consumes the row
# — is pinned in `tests/test_backlog_unattended.py`.
# ===========================================================================

def _move_rows():
    """Every `steward_move` row in the fixture's ledger, oldest first."""
    return [e for e in S.read_events(path=S.LEDGER_PATH, limit=1000)
            if e.get("event") == "steward_move"]


def _fm_of(d: Path, iid: int) -> dict:
    return B._split_frontmatter((d / f"{iid}-item-{iid}.md").read_text())[0]


def test_an_applied_move_gets_its_own_ledger_row_not_the_machine_s(board):
    """Clause 1: the row carries item, from, to, note and the tick that wrote it."""
    _write(board, 7, "up_next")
    out = W.apply_moves([{"item_id": 7, "status": "draft", "tags_add": [], "tags_remove": [],
                          "note": "spent its re-offer, parked for Alan"}],
                        tick="board-steward:tick#41")
    assert [m["applied"] for m in out] == [True]
    assert _fm_of(board, 7)["status"] == "draft"
    rows = _move_rows()
    assert len(rows) == 1, f"exactly one row per applied move, got {rows}"
    row = rows[0]
    assert (row["item_id"], row["from"], row["to"]) == (7, "up_next", "draft")
    assert row["note"] == "spent its re-offer, parked for Alan"
    assert row["tick"] == "board-steward:tick#41"
    # A name of its own, because `status_moved` means "the machine moved it" to
    # `scripts/automod/board_decisions.py` and `landed_outcomes.py`, which count it
    # into the dashboard's decision figures. A steward move booked under that name
    # would flatter the machine on the very dashboard the flip's bar is read from.
    machine = [e for e in S.read_events(path=S.LEDGER_PATH, limit=1000)
               if e.get("event") == "status_moved"]
    assert machine == [], f"the steward must not borrow the machine's row name: {machine}"


def test_a_move_the_writer_refused_or_that_changed_nothing_writes_no_row(board):
    """Clause 1's other half: the row records a move, not an intention.

    999 does not exist, so `set_status` declines it; item 8 is already `draft`, so
    the same writer returns False for a no-op. Both come back `applied: False` and
    neither may leave a row: a row for a move that never happened would hand
    `desired_statuses` a status nobody wrote, and the abstention would then protect
    a board state from a machine that is the only thing standing between it and
    `done`.
    """
    _write(board, 8, "draft")
    out = W.apply_moves([
        {"item_id": 999, "status": "up_next", "tags_add": [], "tags_remove": [], "note": "ghost"},
        {"item_id": 8, "status": "draft", "tags_add": [], "tags_remove": [], "note": "no change"},
    ], tick="board-steward:tick#42")
    assert [m["applied"] for m in out] == [False, False]
    assert _move_rows() == [], "a refused or empty move must not claim a status"
    assert _fm_of(board, 8)["status"] == "draft"


def test_two_applied_moves_write_two_rows_each_naming_its_own_item(board):
    """One row per MOVE, not one per tick — the aggregate `board_steward` row
    already carries the tick, and it carries PROPOSED moves on a dry-run tick with
    `applied: []` whenever `apply: false`, so it cannot answer which moves happened."""
    _write(board, 11, "draft")
    _write(board, 12, "up_next")
    W.apply_moves([{"item_id": 11, "status": "up_next", "tags_add": [], "tags_remove": [],
                    "note": "ready, spend refreshed"},
                   {"item_id": 12, "status": "draft", "tags_add": [], "tags_remove": [],
                    "note": "parked"}], tick="board-steward:tick#43")
    assert [(r["item_id"], r["from"], r["to"], r["tick"]) for r in _move_rows()] == [
        (11, "draft", "up_next", "board-steward:tick#43"),
        (12, "up_next", "draft", "board-steward:tick#43")]


def test_a_move_that_only_changed_tags_endorses_a_status_without_claiming_one(board):
    """The row is a STATUS claim, so a tags-only application writes none.

    `set_status` declines a move whose status the item already carries and the
    caller then applies the tags alone (`apply_moves`'s second branch). Writing a
    `from`/`to` row for that would say the steward set a status it never moved, and
    because the abstention holds until a machine event lands, the echo would freeze
    the item at whatever status it happened to be at while a tag went on — a
    stronger power than the steward has been given anywhere else.
    """
    _write(board, 13, "draft")
    out = W.apply_moves([{"item_id": 13, "status": "draft", "tags_add": ["needs-human"],
                          "tags_remove": [], "note": "needs Alan, leave it parked"}],
                        tick="board-steward:tick#44")
    assert [m["applied"] for m in out] == [True], "the tag did go on: the move was applied"
    assert "needs-human" in _fm_of(board, 13)["tags"]
    assert _move_rows() == [], "no status changed, so no status claim"


# ───────────────────── #3752: the table has an instant, the board does not ─────────────────────
#
# The figures are #3752's own, measured on the tree it was triaged on. The board-steward
# tick that filed it ran 2026-11-04T04:13:50.44Z → 04:16:17.06Z (146.6 s), and #3570's
# `backlog_implement phase=started` row is stamped 04:14:04Z — 14 s after that tick read
# `desired_statuses` (`workers/sources/board_steward.py`) and 132 s before its reply was
# scored against that same table. Re-running `desired_statuses` over the same ledger at
# 04:47Z answers `('in_progress', 'an automod round is in flight for it')` for #3570: the
# STEWARD's answer. The window that filed it: judged 99, agree 86, disagree 11, rate
# 0.8687, `below_floor()` True. Scanning every window tick's `started_at`→`completed_at`
# against each item's own ledger rows puts 3 of the window's 11 disagreement instances in
# this class — #3524 (`land_rescued`, `round_abandoned`), #3418 (`land_rescued`,
# `status_moved`) and #3570 (`backlog_implement`, `round_start`) — because the pool opens
# rounds continuously while the steward is thinking.
#
# The second class the same item measured: #3433's row printed `steward proposed up_next`
# beside `the state machine wants up_next`. Its disagreeing tick
# (2026-10-28T06:26:33Z) proposed `draft` and recorded no side; the `up_next` came from an
# AGREEING tick at 2026-11-03T23:43:23Z, and the machine's side came from the filing
# tick's live table — an agreement, printed inside the disagreement list.

STALE_READ = "2026-11-04T04:13:50Z"        # the instant `expected` was read
STALE_MOVE = "2026-11-04T04:14:04Z"        # 14 s later: inside the same tick
STALE_BEFORE = "2026-11-04T04:13:40Z"      # 10 s earlier: the board was still still

#: Every row of a `## Rows` section, for the invariant no single row can show alone.
_ROW_STATUSES = re.compile(r"steward proposed `(?P<steward>[a-z_]+)`; the state machine "
                           r"wants `(?P<machine>[a-z_]+)`")


def _rows_of(body: str) -> list[str]:
    return [ln for ln in body.splitlines() if ln.startswith("- #")]


def test_an_item_whose_ledger_moved_after_the_table_read_abstains_rather_than_disagreeing(tmp_path):
    """#3752 clause 1: `agreement()` is given the instant its `expected` was read, and an
    id with a ledger row of its own after that instant is an abstention, not a disagreement.

    The id being weighed is #3570, and the row is the one this item was filed on: a round
    starting is exactly the event `desired_statuses`' `in_flight` arm reads its answer
    from, so a row after the read is proof the table has stopped describing the board the
    steward reasoned over. `stale` is published BESIDE `disagree` rather than folded into
    it, because a reader of the aggregate has to be able to tell a steward that stopped
    disagreeing from a guard that stopped counting.
    """
    led = tmp_path / "ledger.jsonl"
    _led(led, event="backlog_implement", item_id=3570, phase="started",
         round_id="SM_20261104_041404", created_at=STALE_MOVE)
    agree = W.agreement(moves=[{"item_id": 3570, "status": "in_progress"},
                               {"item_id": 3433, "status": "in_progress"}],
                        expected={3570: ("up_next", "triage confirmed it with an "
                                                    "acceptance check"),
                                  3433: ("up_next", "triage confirmed it with an "
                                                    "acceptance check")},
                        current={3570: "up_next", 3433: "up_next"},
                        table_read_at=STALE_READ, ledger=led)
    assert agree["stale"] == [3570], (
        "a round start 14 s after the table read must not score as the steward disagreeing")
    assert agree["disagree"] == [3433], "the id the board did not move is untouched"
    assert [s["item_id"] for s in agree["disagree_sides"]] == [3433], (
        "an abstention has no machine side to record, because it was never scored")
    assert agree["judged"] == 1 and agree["rate"] == 0.0, (
        "the stale id leaves the DENOMINATOR: an abstention is #1688's shape — a decision "
        "not measured — and never an agreement credited to the steward")


def test_a_round_started_while_the_steward_was_answering_is_not_scored_as_a_disagreement(
        board, monkeypatch):
    """#3752 clause 1 over the seam the bug lives on, not the function that fixes it.

    The row is appended INSIDE the canned turn, because that is where a round start
    happens in a real tick: `execute` reads the table, hands the prompt to the model, and
    the pool opens a round while it is answering. A unit test of `agreement()` alone could
    pass with a `table_read_at` that was never the read's instant, which is the whole
    failure — the instant has to travel from the read to the score.
    """
    from workers.sources import _common as C
    from workers import board_disagreement as D

    _ten_item_board(board)
    monkeypatch.setattr(B, "desired_statuses",
                        lambda ledger: {7: ("up_next", "confirmed with acceptance")})
    monkeypatch.setattr(D, "window_now", lambda **kw: _window7())
    started_at = (datetime.now(timezone.utc)
                  + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")

    async def turn(prompt, **kw):
        _ev(event="backlog_implement", item_id=7, phase="started", round_id="SM_3752",
            created_at=started_at)          # 14 s in, where #3570's row landed
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": [{"item_id": 7, "status": "draft",
                                          "note": "never triaged"}],
                               "next_pick": 0, "next_pick_reason": "",
                               "summary": "net 0 today"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))

    assert out["status"] == "success", out
    blk = out["meta"]["agreement"]
    assert blk["stale"] == [7] and blk["disagree"] == [], blk
    assert blk["judged"] == 0 and blk["rate"] is None, (
        "the tick judged a board its table had stopped describing: that is an unmeasured "
        "tick, and saying so is the only honest output it has")
    assert "1 abstained" in out["summary"], out["summary"]
    assert _filed(board) == [], "a disagreement the board moved under may not reach a ruler"
    ticks = [e for e in S.read_events(path=S.LEDGER_PATH)
             if e.get("event") == "board_steward"]
    assert len(ticks) == 1 and ticks[0]["agreement"]["stale"] == [7], (
        "the abstention has to be in the ledger row the next reader sums, or the guard is "
        "invisible to the instrument it protects")


def test_a_disagreement_the_board_did_not_move_still_lands_with_its_recorded_side(tmp_path):
    """#3752 clause 2: the guard abstains, and a guard that also blinded the instrument
    would raise the flip bar by deleting the evidence.

    Two rows that are NOT proof the board moved are weighed here: #3508's `status_moved`
    is stamped 10 s BEFORE the read (the table was read after it, so it describes the
    board), and #3456's `round_abandoned` has both stamps mangled — `created_at` and the
    epoch `ts` that `state.append_event` writes beside it are neither a parsable date nor a
    number — so it can neither prove nor disprove that the board moved. One readable stamp
    would be enough to date a row, which is why the row's OTHER half has to be blanked too. Both disagreements stand, with the
    machine's side recorded for each. The last assertion is the rail for every caller that
    has no instant to give: with no `table_read_at` the counts are exactly what they were
    before the guard existed.
    """
    led = tmp_path / "ledger.jsonl"
    _led(led, event="status_moved", item_id=3508, to="in_progress", created_at=STALE_BEFORE)
    _led(led, event="round_abandoned", item_id=3456, created_at="not a timestamp",
         ts="not a number")
    kwargs = dict(moves=[{"item_id": 3508, "status": "in_progress"},
                         {"item_id": 3456, "status": "draft"}],
                  expected={3508: ("up_next", "triage confirmed it with an acceptance"),
                            3456: ("up_next", "triage confirmed it with an acceptance")},
                  current={3508: "up_next", 3456: "up_next"})
    agree = W.agreement(table_read_at=STALE_READ, ledger=led, **kwargs)
    assert agree["stale"] == [], "a row from before the read is not the board moving after it"
    assert agree["disagree"] == [3456, 3508], agree
    assert [s["item_id"] for s in agree["disagree_sides"]] == [3456, 3508]
    assert agree["judged"] == 2 and agree["rate"] == 0.0
    assert W.agreement(**kwargs)["disagree"] == [3456, 3508], (
        "the guard is opt-in on the instant: a caller that passes none scores as before")


def test_no_filed_row_borrows_the_machine_side_of_the_filing_tick_for_an_id_it_never_judged(
        board, tmp_path):
    """#3752 clause 3: `expected` is THIS tick's answer, never a table to seed the window with.

    The id being weighed is #3433 and both halves of its real row are reproduced: the tick
    that disagreed about it (`2026-10-28T06:26:33Z`) proposed `draft` and recorded no
    machine side, while the filing tick's live `desired_statuses` — which is the very
    object passed in as `expected` — answers `('up_next', 'triage confirmed it with an
    acceptance check')` for it. #3677 retired the RE-RUN of `desired_statuses` for window
    ids, and the seed `wide_expected = dict(expected or {})` survived it, which is how that
    side still reached the row. So the assertion is not just that the row admits it: this
    exact status string, the filing tick's own answer for an item it had not judged, may
    appear nowhere in the body. The Coverage figure counts only ids whose side the ledger
    holds, so 3433 is not in it either — 5 of the 8 named rows, the five this tick scored.
    """
    led = tmp_path / "ledger.jsonl"
    _led(led, event="board_steward", apply=False, created_at="2026-10-28T06:26:33+00:00",
         disagree=[3433], moves=[{"item_id": 3433, "status": "draft",
                                 "note": "no triage verdict in its history"}])
    expected = {**DIS_EXPECTED,
                3433: ("up_next", "triage confirmed it with an acceptance check")}
    ids, rows, wide = D._widen(_agree(*DIS_AGREE), DIS_MOVES, expected,
                              _window7(disagree_ids=DIS_WINDOW + [3433]), led)
    assert 3433 in ids, "the window's id is still named — a side is not required to name it"
    assert 3433 not in wide, "the filing tick's table was seeded into the window's rows"
    assert {"item_id": 3433, "status": "draft"} in rows, (
        "its proposal is still the disagreeing tick's own move")

    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, expected,
                           window=_window7(disagree_ids=DIS_WINDOW + [3433]),
                           backlog_dir=board, ledger=led)
    assert out is not None and 3433 in out["ids"]
    body = _filed(board)[0].body
    assert "triage confirmed it with an acceptance check" not in body, (
        "the machine's side was taken from the filing tick's live `desired_statuses`, not "
        "from the tick that scored the disagreement")
    row = [ln for ln in body.splitlines() if ln.startswith("- #3433")][0]
    assert "not recorded" in row and "`draft`" in row, row
    assert "- 5 of the 8 named rows carry the machine's side" in body, (
        "Coverage counted a side the ledger does not hold")


def test_a_row_that_could_only_be_built_from_two_ticks_prints_the_newer_side_it_never_had(
        board, tmp_path):
    """#3752 clause 4: one tick, both halves — an agreement cannot be printed inside the
    disagreement list, whatever the store holds.

    Both rows here are #3433's real ledger: the disagreeing tick at
    `2026-10-28T06:26:33Z` proposed `draft`, and a LATER tick at `2026-11-03T23:43:23Z`
    scored the item as `agree` with a `up_next` proposal — the two separate ledger walks
    kept the newer proposal and the older (unrecorded) side, and printed an agreement.
    `_window_pairs` walks once and takes both halves from the one tick that disagreed, so
    the row prints the `draft` that was actually disagreed with and admits no machine side
    was recorded. The loop after it is the clause as an invariant over the whole body, not
    a property of one row: no line may name the same status on both sides.
    """
    led = tmp_path / "ledger.jsonl"
    # Both rows carry an `agreement` block, because that block is how the window counted
    # them at all: `judgement_aggregate` sums `agreement.disagree`, so a window id with no
    # scoring block in its tick is a row this filer never sees. Neither carries
    # `disagree_sides`: the record shipped with #3680, after both these ticks ran.
    _led(led, event="board_steward", apply=False, created_at="2026-10-28T06:26:33+00:00",
         agreement=_agree(3433),
         moves=[{"item_id": 3433, "status": "draft",
                 "note": "no triage verdict in its history"}])
    _led(led, event="board_steward", apply=False, created_at="2026-11-03T23:43:23+00:00",
         agreement={"judged": 1, "agree": [3433], "disagree": [], "no_opinion": [],
                    "missed": [], "machine_moves": 1, "proposed": 1, "rate": 1.0},
         moves=[{"item_id": 3433, "status": "up_next", "note": "triaged, ready"}])
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(disagree_ids=DIS_WINDOW + [3433]),
                           backlog_dir=board, ledger=led)
    assert out is not None and 3433 in out["ids"]
    body = _filed(board)[0].body
    row = [ln for ln in body.splitlines() if ln.startswith("- #3433")][0]
    assert "`draft`" in row, f"the row printed the agreeing tick's proposal: {row}"
    assert "not recorded" in row, row
    for ln in _rows_of(body):
        pair = _ROW_STATUSES.search(ln)
        if pair and "not recorded" not in ln:
            assert pair.group("steward") != pair.group("machine"), (
                f"an agreement inside the disagreement list: {ln}")


def test_a_pair_whose_two_halves_name_the_same_status_is_dropped_rather_than_printed(
        board, tmp_path):
    """#3752 clause 4, the rail on the rendering itself.

    No honest tick writes this row — `agreement()` records a side only for an id whose
    status differs from the proposal it made — so a ledger event that carries both halves
    for one status is a mangled read or a future writer bug, not evidence. It used to
    render as a disagreement naming `up_next` twice; now the pair is dropped and the row
    says what is true of it, that no tick recorded a side, and the body-wide invariant
    still holds. Contrast
    `test_a_row_is_rendered_from_a_record_only_the_ledger_carries`, which is the same
    shape with honest numbers and must still print the recorded side.
    """
    led = tmp_path / "ledger.jsonl"
    _led(led, event="board_steward", apply=False, created_at=DIS_OLDER_STAMP,
         agreement=_agree_with_sides([3450], {3450: ("up_next", "triaged, no verdict yet")}),
         moves=[{"item_id": 3450, "status": "up_next", "note": "its own reason"}])
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(disagree_ids=DIS_WINDOW + [3450]),
                           backlog_dir=board, ledger=led)
    assert out is not None and 3450 in out["ids"], "the id stays named even with no pair"
    body = _filed(board)[0].body
    row = [ln for ln in body.splitlines() if ln.startswith("- #3450")][0]
    assert "not recorded" in row, f"the same status printed on both sides: {row}"
    for ln in _rows_of(body):
        pair = _ROW_STATUSES.search(ln)
        if pair and "not recorded" not in ln:
            assert pair.group("steward") != pair.group("machine"), (
                f"an agreement inside the disagreement list: {ln}")


# ─────────────────────────── triage= on the board line (#3766) ─────────────────
#
# The four items this is for are #3433 #3537 #3656 and #3704: one steward tick
# (runs row completed 2026-11-04T18:09:24.134070+00:00, `judged=7 agree=3
# disagree=[3433,3537,3656,3704]`) proposed `draft` for all four while
# `desired_statuses` wanted `up_next` for all four. All four ARE triage-confirmed.
# The steward was never told: `_item_line` rendered status, name, created, tags,
# round_flag and 280 body characters, and none of those carries a verdict — while
# the prompt ordered the model to move "an item with no triage verdict that sits in
# `up_next`" back to `draft` (that bullet is `workers/sources/board_steward.py:299-311`
# now, having moved twice under #3766 and #3784; the same sentence is quoted verbatim
# at :1022 of that file). A rule whose
# condition is a field that is not on the line is a rule that fires on everything,
# which is the same defect #3547 fixed for rounds by putting `round_flag=` on every
# line. Trailing-7-day agreement at filing: judged=99, rate=0.8788 against
# owed-check's 0.90 floor; re-measured 2026-11-04 at judged=106, rate=0.8491.

def _triage_item(d, iid, status, *, verdict="confirmed", acceptance="step: do the thing",
                 hours_old=48, held=False, **fm):
    """An item on the board plus the ledger row that makes it what it is.

    `held=True` puts `held: true` on the CONFIRMATION row itself, which is what
    `backlog.held_confirmations` selects and what `confirmed_verdicts` drops — the
    pair of readings the `triage=` field is derived from.
    """
    _write(d, iid, status, hours_old=hours_old, **fm)
    if verdict:
        _ev(event="backlog_triage", item_id=iid, verdict=verdict, acceptance=acceptance,
            **({"held": True} if held else {}))


def test_every_board_line_names_the_verdict_confirmed_verdicts_derives(board):
    """Clause 1. The line carries the verdict, derived from the machine's own reader.

    Four items, four ledger situations: #11 confirmed, #12 confirmed and HELD by a
    landing, #13 confirmed-then-re-triaged (so `confirmed_verdicts` no longer returns
    it), #14 never triaged. Each line's `triage=` must be what
    `backlog.confirmed_verdicts`/`held_confirmations` say about that item right now,
    and the word must never be a status — the no-answer-sheet rail (#3329) holds only
    if the new field names an OBSERVATION and not a destination.
    """
    _triage_item(board, 11, "up_next")
    # `confirmed-held` is the confirmation carrying `held: true` — the same row
    # `held_confirmations` returns and `confirmed_verdicts` therefore drops.
    _triage_item(board, 12, "up_next", held=True)
    _triage_item(board, 13, "up_next")
    _ev(event="backlog_retriage", item_id=13, ts=time.time() + 5, reason="premise re-checked")
    _write(board, 14, "up_next")

    items = B.all_items(("lloyd",), backlog_dir=board)
    assert len(W.board_view(items, set())) == 4
    board_text = W.build_prompt(events=[], items=items, n_open=4, since_ts=0.0)
    verdicts = B.confirmed_verdicts(S.LEDGER_PATH)
    held = B.held_confirmations(S.LEDGER_PATH)
    for iid, want in ((11, "confirmed"), (12, "confirmed-held"), (13, "none"),
                      (14, "none")):
        line = _line(board_text, iid)
        found = re.findall(r"\btriage=(\S+)", line)
        assert len(found) == 1, f"{line}\nexpected exactly one triage=, found {found}"
        assert found[0] == want, f"#{iid}: line says {found[0]!r}, ledger says {want!r}"
        # The field is derived, not authored: it agrees with the reader the machine
        # itself consults, item by item.
        assert ((iid in held) if want == "confirmed-held" else
                (iid in verdicts) if want == "confirmed" else iid not in verdicts), \
            f"#{iid}: `triage={want}` contradicts confirmed_verdicts/held_confirmations"
    # #3329's rail, over every board line. This was a `line.startswith("- #")` loop,
    # which matched nothing at all — so the docstring's "never be a status" was pinned
    # by an assertion that could not fail. `_assert_no_line_names_a_destination` is the
    # one pattern for a board line here, and it fails loudly when its own reach is zero.
    _assert_no_line_names_a_destination(board_text, 4)


def test_a_confirmed_verdict_older_than_the_retriage_mark_reads_none(board):
    """Clause 2. One measurement, so the model and the machine cannot disagree.

    #15 was confirmed six hours ago and re-triaged one hour ago. `confirmed_verdicts`
    drops it (`ts` before the newest `backlog_retriage` mark), so the line must read
    `triage=none` and not `confirmed`. The direction of the risk is worth naming: a
    line that showed a superseded verdict would be the STEWARD holding an item the
    machine considers untriaged, while a line that hides a live verdict is what
    produced #3433 #3537 #3656 #3704. Hiding a stale one is correct only because it is
    the same answer the machine gives — the clause is "one measurement", not "less
    information".
    """
    _triage_item(board, 15, "up_next")
    _ev(event="backlog_retriage", item_id=15, ts=time.time(), reason="premises re-checked")
    assert 15 not in B.confirmed_verdicts(S.LEDGER_PATH), \
        "the fixture no longer holds: a re-triaged item must drop out of confirmed_verdicts"

    items = B.all_items(("lloyd",), backlog_dir=board)
    assert len(W.board_view(items, set())) == 1, "the item must be shown to be ruled on"
    board_text = W.build_prompt(events=[], items=items, n_open=1, since_ts=0.0)
    assert "triage=none" in _line(board_text, 15), _line(board_text, 15)
    assert "triage=confirmed" not in _line(board_text, 15)


def test_the_up_next_to_draft_rule_can_only_fire_on_a_line_reading_triage_none(board):
    """Clause 3. The rule's condition is now a field the model can actually read.

    The shipped rule (prompt lines 278-283) said "An item with no triage verdict that
    sits in `up_next` goes back to `draft`" — and the line carried no verdict field, so
    the model could only satisfy it by guessing, and guessed `draft` for four confirmed
    items in one tick. The rewrite must (a) name `triage=none` as the ONLY trigger,
    (b) forbid the move on a line reading `triage=confirmed` or `confirmed-held`, and
    (c) still not become an order to move every confirmed item, which is the rail the
    #3547 rule already holds.
    """
    _triage_item(board, 16, "up_next")
    _write(board, 17, "up_next")
    items = B.all_items(("lloyd",), backlog_dir=board)
    prompt = W.build_prompt(events=[], items=items, n_open=2, since_ts=0.0)

    start = prompt.index("- **An `up_next` item")
    bullet = prompt[start:prompt.index("\n- ", start + 1)]
    assert "`triage=none`" in bullet, f"the rule's trigger is not the field: {bullet}"
    assert "`triage=confirmed`" in bullet, \
        f"the rule never says the move is forbidden on a confirmed line: {bullet}"
    assert "no triage verdict" not in bullet, \
        f"the rule still reads on the absence of a field the line now always carries: {bullet}"
    # The field is documented where the other fields are documented, or the model
    # sees a token it has no rule for.
    assert "- **`triage=`" in prompt, "the field is never named to the steward"
    field_doc = prompt[prompt.index("- **`triage=`"):]
    field_doc = field_doc[:field_doc.index("\n\n")]
    for value in W.TRIAGE_VALUES:
        assert value in field_doc, f"`{value}` is rendered but never defined: {field_doc}"


def test_a_confirmed_item_the_steward_parks_is_still_scored_as_a_disagreement(
        board, monkeypatch):
    """Clause 4. Grounding informs and does not silence.

    #18 is `up_next`, confirmed, and the machine wants it to stay `up_next`. The
    steward proposes `draft` — the exact move #3433 made. Telling the model the
    verdict is only a fix if a wrong answer still costs: the flip gate is
    `judged >= 50 AND rate >= 0.90` (#3089 asked, #3329 ruled), so any change that
    quietly removed these proposals from the denominator would lift the rate by
    deleting its evidence. This asserts the disagreement survives end-to-end:
    the move is in the row's `moves`, in `agreement.disagree`, and
    `agreement.disagree_sides` names the machine's side for it.
    """
    from scripts.automod import state as S, state as ST
    from workers.sources import _common as C

    _triage_item(board, 18, "up_next")
    expected = {18: ("up_next", "confirmed, acceptance names an actionable step")}
    monkeypatch.setattr(B, "desired_statuses", lambda *a, **k: dict(expected))

    raw = [{"item_id": 18, "status": "draft", "note": "looks parked"}]
    seen: dict[str, str] = {}

    async def turn(prompt, **kw):
        seen["prompt"] = prompt
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": raw, "next_pick": 0, "next_pick_reason": "",
                               "summary": "net 0 today"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)

    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": True})))
    assert out["status"] == "success", out

    row = [e for e in ST.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "board_steward"][-1]
    assert 18 in row["agreement"]["disagree"], \
        f"grounding a proposal removed it from the score: {row['agreement']}"
    sides = [r for r in row["agreement"]["disagree_sides"] if r["item_id"] == 18]
    assert len(sides) == 1, \
        f"disagree_sides lost the machine's half of this disagreement: {sides}"
    assert sides[0]["status"] == "up_next", sides[0]
    assert row["moves"][0]["item_id"] == 18 and row["moves"][0]["status"] == "draft", \
        row["moves"]
    assert "triage=confirmed" in seen["prompt"], \
        "the steward was scored on a line that still never named the verdict"


# ── #3784: a human-only confirmation is a different observation ────────────
#
# #3766 put the verdict on every board line, and the disagreement rate did not
# recover, because a bare `confirmed` is not one observation but two machine
# branches wearing one word. `desired_statuses` asks a second question of a
# confirmed verdict — `elif iid in confirmed and not is_human_only(...)`
# (scripts/automod/backlog.py:5939) — and a human-only confirmation falls past
# it into `("draft", "triaged confirmed; not for the unattended loop")`
# (scripts/automod/backlog.py:5953), while a plain one goes to `up_next`.
# `_triage_map` never asked that second question: it mapped every confirmed id to
# `confirmed`, and the prompt's `up_next` -> `draft` bullet in
# workers/sources/board_steward.py:305-308 then orders the model to answer a
# confirmed line with `up_next` or
# nothing. On #3782, whose acceptance opens
# `human-only: ~/obsidian/autonomy/105-check-health-census-weekly.md:14`, that
# is a manufactured disagreement: the steward was told to propose the one status
# the machine has already refused. Measured live on the ledger at triage, 10 open
# items get the "not for the unattended loop" answer and 7 of them (#3119 #3359
# #3446 #3466 #3477 #3645 #3782) read bare `triage=confirmed` — seven rows the
# next under-floor tick will score as the model's error.


#: A marker whose named path the gate would really refuse, so `is_human_only` is
#: True by the #3027 rule — the shape #3782's own acceptance opens with.
GUARD_MARKER = "human-only: ~/obsidian/autonomy/105-check-health-census-weekly.md:14"

#: A marker that names only a runtime-data path, which `_under_runtime_data_root`
#: answers True for so `_names_a_guard` skips the token
#: (`scripts/automod/backlog.py:403-411`, #3027), and `is_human_only` is therefore
#: False: the item stays implementable. The contrast is the point — the fourth value
#: is keyed on the guard question, not on the marker's presence.
NON_GUARD_MARKER = "human-only: ~/lloyd-data/_pipeline/thing.json"


def test_a_human_only_confirmation_reads_confirmed_human_only_and_the_rest_do_not_move(
        board):
    """Clause 1. The field gains the machine's second question, and nothing else moves.

    Four items, four situations, one assertion each. #21 is confirmed behind a guard,
    which is the case the field could not express. #22 is a plain confirmation and #23
    a held one — those two are #3766's values and must come back byte-identical, since
    the round that shipped them is the reason the field exists. #24 is the trap: its
    acceptance OPENS with `human-only:` but names only a `~/lloyd-data/**` path, which
    `_names_a_guard` discards by decision (#3027), so the machine implements it and the
    line must still say `confirmed`. A fix that keyed on the marker's presence instead
    of on `is_human_only` would reddens #24 — and would park a landable item with a
    decision nobody asked for, the exact failure the 2026-10-05 ruling removed.
    """
    from scripts.automod.backlog import is_human_only

    assert W.TRIAGE_VALUES == ("confirmed", "confirmed-held",
                               "confirmed-human-only", "none"), W.TRIAGE_VALUES
    assert len(W.TRIAGE_VALUES) == 4, (
        "the field's whole safety property is that each value maps to exactly one "
        "machine branch; a fifth would need its own rule in the prompt")

    _triage_item(board, 21, "up_next", acceptance=GUARD_MARKER)
    _triage_item(board, 22, "up_next", acceptance="step: do the thing")
    _triage_item(board, 23, "up_next", acceptance="step: do the thing", held=True)
    _triage_item(board, 24, "up_next", acceptance=NON_GUARD_MARKER)

    # The premise, re-checked rather than assumed: the fourth value exists only
    # because `is_human_only` answers differently for #21 and #24.
    confirmed = B.confirmed_verdicts(S.LEDGER_PATH)
    assert is_human_only(confirmed[21].get("acceptance")), "fixture lost its guard"
    assert not is_human_only(confirmed[24].get("acceptance")), (
        "#3027's discard stopped holding, and #24 is no longer the contrast case")

    tm = W._triage_map(S.LEDGER_PATH)
    assert tm[21] == "confirmed-human-only", tm
    assert tm[22] == "confirmed", tm
    assert tm[23] == "confirmed-held", tm
    assert tm[24] == "confirmed", tm


#: The shape #4230 makes a guard: a marker head naming the loop's runtime data root
#: AND a value asserting no round/agent/scheduled run may perform the act. #4009's
#: verbatim live acceptance is pinned as `ACC_4009_LIVE` in
#: `tests/test_backlog_unattended.py`; this is that row's own head cut and its own
#: assertion, with the packet's byte count and the `Done when` tail dropped.
#:
#: The em dash after the store path is load-bearing and is why this fixture was rewritten
#: once. `_head_text` stops at the first segment separator, so the head carries the root
#: token and NOTHING ELSE — the rail path `skills/entity-resolution-sweep/SKILL.md` sits
#: below the cut, exactly as in the live row. Trim the dash away and that token joins the
#: head, `spec.classify` calls it `unlisted`, `_names_a_guard` returns True at the
#: repo-path branch, and this whole node passes at base while proving nothing about the
#: rule it is here for. The node asserts both halves of that, plus the stub-off.
RUNTIME_ROOT_ASSERTED = (
    "human-only: a person runs §4 of the merge packet against the live store, because "
    "the merge writes `~/lloyd-data/_pipeline/vault-derived/kg.sqlite` — outside both "
    "the repo and the vault — and `skills/entity-resolution-sweep/SKILL.md:59`/`:101` "
    "forbid the SUFFIX tier to any agent or scheduled run")

#: #2983's real head: a file under the same root, no assertion against the loop. The
#: contrast is the node — #3027's discard has to keep holding here, or the steward
#: starts calling landable work human-only, which is the false-positive the
#: 2026-10-05 ruling priced as the expensive direction.
RUNTIME_ROOT_EMIT = ("human-only: "
                     "~/lloyd-data/_pipeline/research/behavioural_pairs/"
                     "NOISE_FLOOR_NOTE.draft.md")


def test_a_runtime_root_clause_that_asserts_no_round_may_run_it_renders_parked(board,
                                                                              monkeypatch):
    """#4230 clause 5: the steward agrees with dispatch without being told to.

    `_triage_map` asks `B.is_human_only` and nothing else (`workers/sources/
    board_steward.py:897`), so the new rule reaches the board line for free — and the
    node is worth having because the SAME call is what stops the line from lying: #4009
    sat `triage=confirmed` while `select_confirmed` was being spent on it twice.
    Two-sided, then: the parked item renders `confirmed-human-only`, and the item that
    merely EMITS under that root still renders plain `confirmed`, because the rule
    gained an assertion test and not a path test.

    Three premises before the render, because the render would otherwise pass at base.
    The parked fixture's head carries the store token and no other path (`SKILL.md` below
    the cut, as in the live row), and with `_runtime_root_assertion_holds` stubbed off its
    verdict goes False — so the line this node grades comes from #4230's rule and not from
    the unlisted-repo-path branch that a sloppier fixture walks into."""
    from scripts.automod.backlog import is_human_only

    _triage_item(board, 31, "up_next", acceptance=RUNTIME_ROOT_ASSERTED)
    _triage_item(board, 32, "up_next", acceptance=RUNTIME_ROOT_EMIT)
    head = B._head_text(RUNTIME_ROOT_ASSERTED)
    assert "_pipeline/vault-derived/kg.sqlite" in head, (
        "the fixture no longer names the runtime root in its head")
    assert "entity-resolution-sweep" not in head, (
        f"the head reached past the em dash into the rail clause, so this node's verdict "
        f"would come from the unlisted-repo-path branch: {head!r}")
    assert is_human_only(RUNTIME_ROOT_ASSERTED) is True, (
        "the fixture stopped carrying a root token or the assertion")
    assert is_human_only(RUNTIME_ROOT_EMIT) is False, (
        "#3027's emit-under-the-root contrast case stopped being one")
    # Restored by hand, not by `monkeypatch.undo()`: the `board` fixture was handed the
    # SAME function-scoped monkeypatch, so undo() would take the fixture's own patches
    # with it and `_triage_map` would be reading the real ledger from here on.
    real_rule = B._runtime_root_assertion_holds
    try:
        monkeypatch.setattr(B, "_runtime_root_assertion_holds", lambda acceptance: False)
        assert is_human_only(RUNTIME_ROOT_ASSERTED) is False, (
            "the parked verdict came from a branch other than #4230's rule")
    finally:
        monkeypatch.setattr(B, "_runtime_root_assertion_holds", real_rule)

    tm = W._triage_map(S.LEDGER_PATH)
    assert tm[31] == "confirmed-human-only", tm
    assert tm[32] == "confirmed", tm

    items = B.all_items(("lloyd",), backlog_dir=board)
    prompt = W.build_prompt(events=[], items=items, n_open=2, since_ts=0.0)
    assert re.findall(r"\btriage=(\S+)", _line(prompt, 31)) == ["confirmed-human-only"]
    assert re.findall(r"\btriage=(\S+)", _line(prompt, 32)) == ["confirmed"]

    # The machine side, stated so a passing render cannot hide a disagreement: the
    # steward's line and `desired_statuses` must want the same thing for each item.
    want = B.desired_statuses(S.LEDGER_PATH)
    assert want[31][0] == "draft" and "not for the unattended loop" in want[31][1], (
        f"the line says parked and the state machine says otherwise: {want.get(31)}")
    assert want[32][0] == "up_next", (
        f"the emit item left the implement pool beside it: {want.get(32)}")


def test_a_human_only_confirmed_line_renders_its_value_and_never_the_status_wanted(
        board):
    """Clause 2. The new value reaches the wire, and the answer sheet stays hidden.

    #25 sits `up_next` with a confirmation behind a guard, so the machine wants
    `draft`. The line has to carry the OBSERVATION (`triage=confirmed-human-only`) and
    must not carry the destination: #3329's rule is that the board line may name a fact
    the model checks its proposal against and never the status `desired_statuses` wants,
    because this tick is scored against that table. So the assertion is two-sided — the
    value is present on this item's line, and no `triage=` on any line of this board
    names a status, which is what would make the disagreement rate meaningless.
    """
    import re as _re

    _triage_item(board, 25, "up_next", acceptance=GUARD_MARKER)
    items = B.all_items(("lloyd",), backlog_dir=board)
    assert len(items) == 1, "the item must be shown to be ruled on"
    want = B.desired_statuses(S.LEDGER_PATH)
    assert want[25][0] == "draft" and "not for the unattended loop" in want[25][1], (
        f"the machine no longer parks a human-only confirmation, and this node is "
        f"measuring nothing: {want.get(25)}")

    prompt = W.build_prompt(events=[], items=items, n_open=1, since_ts=0.0)
    line = _line(prompt, 25)
    found = _re.findall(r"\btriage=(\S+)", line)
    assert len(found) == 1, f"expected exactly one triage= on {line!r}, got {found}"
    assert found[0] == "confirmed-human-only", (
        f"#{25} is confirmed behind a guard and the line says {found[0]!r}; the model "
        "reading it has no way to know the loop may not implement it")
    # No `!= want[25][0]` assertion here: the review named that as unfailable,
    # since `found[0] == "confirmed-human-only"` and `want[25][0] == "draft"`
    # already entail it. The no-answer-sheet property is pinned by the sweep
    # below, over EVERY board line against the set of status names.
    #
    # The sweep, and why it is not a loop over a guessed prefix: the first commit's
    # `l.startswith("- #")` matched nothing at all — event lines open with a
    # `%m-%d %H:%M UTC` stamp, health lines are prose, and no rendered line begins
    # `- #` — so its body never ran and its assertion could not fail. The helper keys on
    # `#<id> ` like `_line` and `_flags` do, and pins its own reach by count, so a
    # filter that stops matching reddens the node instead of emptying the loop.
    _assert_no_line_names_a_destination(prompt, len(items))


def test_the_prompt_names_four_triage_values_and_rules_the_human_only_line():
    """Clause 3. The model is told what the fourth value means, and what it forbids.

    A value the prompt never defines is a token the model guesses at, which is the
    failure #3547's own triage names. So the field doc must enumerate all four and say
    the operative thing: on a `confirmed-human-only` line the loop may not implement the
    item, so `up_next` is never the proposal for it — `draft`, or no move. The old
    sentence "Three values, no others" has to go, because it is a false statement about
    the field the moment the fourth exists.

    The existing `up_next` -> `draft` bullet is deliberately NOT re-worded, and that is
    the second half of this clause: its trigger is `triage=none` and its prohibition
    names `confirmed` and `confirmed-held`, so it simply does not fire on the new value
    — the least-diff place for the rule to live that cannot regress #3766's fix. The
    assertions below are the ones that node already makes, restated here so a future
    edit that quietly widens that bullet to the new value reddens something.
    """
    import re as _re

    prompt = W.build_prompt(events=[], items=[], n_open=0, since_ts=0.0)

    assert "- **`triage=`" in prompt, "the field is never named to the steward"
    doc = prompt[prompt.index("- **`triage=`"):]
    doc = doc[:doc.index("\n- ")]
    assert "Three values, no others" not in doc, (
        f"the field doc still asserts a cardinality the field does not have: {doc}")
    for value in W.TRIAGE_VALUES:
        assert value in doc, f"`{value}` is rendered but never defined: {doc}"

    # The rule has to be pinned to ITS OWN sentence, and the only scope that
    # does that is the bolded sentence itself: a character window around the
    # value cannot test this (the review refused the first commit for exactly
    # that — 700 back from the value plus the rest of the bullet still contained
    # `up_next` and `draft` from the bullet's own header, so deleting the whole
    # rule left the node green), and splitting on terminal punctuation cannot
    # either, because the four-value enumeration is itself one sentence ending in
    # `)`, so the definition and the rule land in the same fragment. So: find the
    # rule by its own anchor, and fail when the anchor is absent — which is what
    # deleting the sentence now does.
    anchor = "**On a `confirmed-human-only` line"
    k = doc.find(anchor)
    assert k >= 0, (
        "the value is named in the enumeration but the sentence that says what it "
        "forbids is gone: the model has a token and no instruction, which is the "
        "shape that produced the row this item is about")
    close = doc.find("**", k + len(anchor))
    assert close > k, "the rule sentence is not closed, so its scope is unreadable"
    rule = doc[k:close]
    # Clause scope, not sentence scope, for the prohibition itself: the rule
    # sentence also forbids `done`, and a sentence-level `"never" in rule` was
    # satisfied by that second clause — measured, softening "`up_next` is never
    # the proposal" to "`up_next` is the proposal" left the node green because
    # "never `done`" was still in the same sentence. So the negation has to be in
    # the same clause as the move it forbids.
    # The negation must come AFTER the move it refuses. A `"may not" in clause`
    # test passed on the softened text — "the loop may not implement the item, so
    # `up_next` is the proposal for it" — because `may not` was the REASON half of
    # the same clause. So the check is on what follows the name.
    after = [c[c.index("up_next") + len("up_next"):]
             for c in _re.split(r"[;—]| - ", rule) if "up_next" in c]
    assert after, f"no clause of the rule addresses `up_next`: {rule}"
    assert any(("never" in a or "not" in a) for a in after), (
        f"the rule does not forbid `up_next` after naming it: {after}")
    assert "draft" in rule or "no move" in rule or "nothing" in rule, (
        f"the rule forbids up_next without saying what to do instead: {rule}")
    assert "implement" in rule, (
        "the rule does not say WHY the move is forbidden — the loop may not "
        f"implement the item is the fact the model needs: {rule}")

    bullet_start = prompt.index("- **An `up_next` item")
    bullet = prompt[bullet_start:prompt.index("\n- ", bullet_start + 1)]
    assert "`triage=none`" in bullet, (
        "the rule's trigger drifted off the field, which is #3766's fix")
    assert "`triage=confirmed`" in bullet and "`triage=confirmed-held`" in bullet, (
        "the confirmed prohibition drifted, which is #3766's fix")
    assert "confirmed-human-only" not in bullet, (
        "the up_next -> draft bullet must not acquire the human-only value: that case "
        "is ruled where the values are defined, and naming it twice is how the two "
        "copies disagree")


def test_nothing_the_machine_parks_as_not_for_the_loop_reads_bare_confirmed(board):
    """Clause 4. The property, over a ledger, not a fixture item.

    The item's acceptance is stated as a universal: for every open item
    `desired_statuses` answers `(draft, ...'not for the unattended loop')` for, the
    `triage=` value derived from the same ledger rows is not bare `confirmed`. That is
    the whole defect in one line — one word standing for two machine branches — and a
    four-item table would only pin the four cases someone thought of. So this walks the
    table the two functions agree on, over a board deliberately populated with every
    verdict the parked branch can be reached by: a guard marker, a re-triaged guard,
    an `unverifiable`, a non-guard marker (which must NOT be caught), and a plain
    confirmation (which must not be caught either).

    The direction of the risk is worth naming: a false positive here (an implementable
    item reading `confirmed-human-only`) would park work the loop could do, which is the
    2026-10-05 ruling's whole subject. A false negative costs agreement points and
    re-files this item, which is what has happened eight times.
    """
    _triage_item(board, 31, "up_next", acceptance=GUARD_MARKER)
    _triage_item(board, 32, "up_next", acceptance=GUARD_MARKER + " (and a second path)")
    _triage_item(board, 33, "up_next", verdict="unverifiable", acceptance="")
    _triage_item(board, 34, "up_next", acceptance=NON_GUARD_MARKER)
    _triage_item(board, 35, "up_next", acceptance="step: do the thing")

    want = B.desired_statuses(S.LEDGER_PATH)
    tm = W._triage_map(S.LEDGER_PATH)
    parked = sorted(iid for iid, tup in want.items()
                    if tup[0] == "draft" and "not for the unattended loop" in tup[1])
    assert parked, (
        "the fixture reaches the parked branch for nobody, so the loop below is an "
        "empty conjunction and proves nothing")

    for iid in parked:
        got = tm.get(iid, "none")
        assert got != "confirmed", (
            f"#{iid} is parked by the machine as not for the unattended loop, but its "
            f"board line says {got!r} and the prompt orders up_next for that word")
        assert got in W.TRIAGE_VALUES, f"#{iid}: {got!r} is not a defined value"

    # And the two shapes that must survive untouched, checked by id so a blanket
    # "not confirmed" over the board could not pass this node by accident.
    assert tm[34] == "confirmed", "a non-guard marker became a hold"
    assert tm[35] == "confirmed", "a plain confirmation became a hold"


# ── #3791: a carrier is the same observation as a human-only confirmation ────
#
# #3784 gave `triage=` its fourth value for ONE of the two ways a confirmed item
# is refused: `is_human_only` names a decision the gate would really refuse. The
# other way is a `hand_run:` receipt that does not pass, and `ready_confirmed`
# refuses the OFFER of a round over it (#3289) while the line went on reading
# `triage=confirmed` — a word the prompt defines as "the loop MAY implement it".
# That is the row #3791 was filed from: four carriers (#3433 #3537 #3656 #3704),
# each told the loop may implement it, each answered `draft`, 8 of the window's
# 19 disagreement rows, reproduced on two scored ticks AFTER #3766 put the
# verdict on every line — which is the proof that showing `triage=` was never
# going to be enough unless the VALUE carried the refusal.

#: A receipt that exits 1, so `owed.hand_run_pending` answers pending. The command
#: is the shape every live carrier's is: an attended apply over runtime data.
PENDING_HAND_RUN = {"set": ["PS-9 -> PS-8"],
                    "cmd": "cd ~/lloyd && .venvs/lloyd/bin/python "
                           "scripts/memory/link_stranded_entities.py --apply",
                    "receipt": "exit 1", "since": "2026-10-29T09:04:35+00:00"}

#: The same block with a receipt that passes: the attended run happened, so this
#: item is implementable and its line must stay bare `confirmed`. The pair is the
#: clause — the field keys on the receipt measurement, not on the block's presence.
PAID_HAND_RUN = {**PENDING_HAND_RUN, "receipt": "exit 0"}


def test_a_confirmed_carrier_reads_confirmed_human_only_and_the_other_values_stay_put(board):
    """Clause 1 of #3791 (the field). Five items, one new value, four unchanged.

    #41 is the carrier: confirmed, `up_next`, a receipt exiting 1. #42 is the
    contrast the clause turns on — the identical block whose receipt exits 0 — and
    must still read plain `confirmed`, because a paid debt is not a refusal and a
    field that parked paid carriers would withhold work the loop can do. #43 has no
    block at all. #44 is a HELD confirmation that also carries a pending block, and
    keeps `confirmed-held`: `desired_statuses` reaches its pool-room hold before any
    carrier check, so the field has to answer the same question in the same order or
    the two surfaces disagree about which hold is speaking. #45 is #3784's guard
    marker, unchanged.

    `TRIAGE_VALUES` is asserted by value AND length: the carrier deliberately reuses
    the existing fourth value instead of adding a fifth, because a fifth value needs
    a fifth rule in the prompt and the observation the model needs is one sentence
    either way ("the loop may not be the one that acts on this").
    """
    from scripts.automod.backlog import is_human_only

    assert W.TRIAGE_VALUES == ("confirmed", "confirmed-held",
                               "confirmed-human-only", "none"), W.TRIAGE_VALUES
    assert len(W.TRIAGE_VALUES) == 4, (
        "the carrier was given a fifth value; one word per machine branch is the "
        f"field's whole safety property and the field now has {W.TRIAGE_VALUES}")

    _triage_item(board, 41, "up_next", hand_run=dict(PENDING_HAND_RUN))
    _triage_item(board, 42, "up_next", hand_run=dict(PAID_HAND_RUN))
    _triage_item(board, 43, "up_next")
    _triage_item(board, 44, "up_next", held=True, hand_run=dict(PENDING_HAND_RUN))
    _triage_item(board, 45, "up_next", acceptance=GUARD_MARKER)

    # The premise, re-checked: the receipt, not the block, splits #41 from #42, and
    # the guard marker of #45 is what makes ITS value, not this round's predicate.
    from scripts.automod import owed as O
    fm_pending, _ = B._split_frontmatter((board / "41-item-41.md").read_text(encoding="utf-8"))
    fm_paid, _ = B._split_frontmatter((board / "42-item-42.md").read_text(encoding="utf-8"))
    assert O.hand_run_pending(fm_pending)[0], "fixture lost its pending receipt"
    assert not O.hand_run_pending(fm_paid)[0], "fixture lost its passing receipt"
    assert is_human_only(B.confirmed_verdicts(S.LEDGER_PATH)[45].get("acceptance")), \
        "fixture lost its guard"

    tm = W._triage_map(S.LEDGER_PATH)
    assert tm[41] == "confirmed-human-only", tm
    assert tm[42] == "confirmed", tm
    assert tm[43] == "confirmed", tm
    assert tm[44] == "confirmed-held", tm
    assert tm[45] == "confirmed-human-only", tm


def test_no_carrier_the_machine_parks_is_ever_shown_as_a_line_the_loop_may_implement(board):
    """Clause 2 of #3791 (the wire). The property over a ledger, and the line on it.

    The clause is a universal, so it is measured as one: for every item
    `desired_statuses` parks with its hand-run reason, the `triage=` value derived
    from the same board is not bare `confirmed`, and the rendered line carries that
    value and no destination. The parked set is asserted non-empty first, because an
    empty conjunction is exactly how a guard like this passes while proving nothing.

    `build_prompt` is called with no `triage=` argument on purpose: that default
    path is the one `execute` takes when a caller forgets to pass a map, and it is
    the path #3766 had to fix for the same reason.
    """
    _triage_item(board, 51, "up_next", hand_run=dict(PENDING_HAND_RUN))
    _triage_item(board, 52, "up_next", hand_run=dict(PENDING_HAND_RUN))
    _triage_item(board, 53, "up_next", hand_run=dict(PAID_HAND_RUN))

    want = B.desired_statuses(S.LEDGER_PATH)
    tm = W._triage_map(S.LEDGER_PATH)
    parked = sorted(iid for iid, tup in want.items()
                    if tup[0] == "draft" and "hand_run is still unexecuted" in tup[1])
    assert parked == [51, 52], (
        f"the machine no longer parks these carriers, so the loop below measures "
        f"nothing: {parked} against {dict(want)}")

    for iid in parked:
        got = tm.get(iid, "none")
        assert got != "confirmed", (
            f"#{iid} is parked by the machine as unexecuted, but its line says "
            f"{got!r} — the word the prompt defines as MAY implement")
        assert got in W.TRIAGE_VALUES, f"#{iid}: {got!r} is not a defined value"
    assert tm[53] == "confirmed", "a paid carrier became a hold"

    items = B.all_items(("lloyd",), backlog_dir=board)
    prompt = W.build_prompt(events=[], items=items, n_open=len(items), since_ts=0.0)
    for iid in parked:
        found = re.findall(r"\btriage=(\S+)", _line(prompt, iid))
        assert found == ["confirmed-human-only"], f"#{iid}: {found} on its own line"
    _assert_no_line_names_a_destination(prompt, len(items))


def test_the_prompt_definition_of_confirmed_human_only_names_the_unexecuted_hand_run():
    """Clause 3 of #3791 (the prose). The value the model is told about matches the
    predicate that derives it.

    A field value whose definition does not mention one of the two ways it is
    derived is a token that lies about half the items carrying it: the model reads
    "its acceptance names a decision the gate would really refuse", sees an
    acceptance with no marker, and concludes the loop may implement. So the
    definition has to name the unexecuted hand_run — the command, the receipt, and
    the reason an implement turn is not the one that runs it.

    Each value's definition is read as ITS OWN parenthetical, and the check is
    two-sided: `hand_run` is in the carrier value's, and in no other value's. A
    character window around the value would pass on a definition that had drifted
    onto a neighbour, which is the failure #3784's own review caught.
    """
    prompt = W.build_prompt(events=[], items=[], n_open=0, since_ts=0.0)
    doc = prompt[prompt.index("- **`triage=`"):]
    doc = doc[:doc.index("\n- ")]

    defs = {}
    for value in W.TRIAGE_VALUES:
        anchor = f"`{value}` ("
        k = doc.find(anchor)
        assert k >= 0, f"`{value}` is enumerated without a definition: {doc}"
        close = doc.find(")", k)
        assert close > k, f"`{value}`'s definition never closes, so its scope is unreadable"
        defs[value] = doc[k + len(anchor):close]

    carrier = defs["confirmed-human-only"]
    assert "hand_run" in carrier, (
        f"the value a carrier now carries is never defined as one: {carrier}")
    assert "receipt" in carrier, (
        f"the definition does not say what makes a hand_run a refusal: {carrier}")
    for value, text in defs.items():
        if value == "confirmed-human-only":
            continue
        assert "hand_run" not in text, (
            f"`{value}`'s definition names the hand_run case, so the model has two "
            f"words for it and one rule: {text}")


# ---------------------------------------------------------------------------
# #3858 — the `next_pick` rule excluded an item on its ACCEPTANCE TEXT ("never
# one whose acceptance begins `human-only:`"), which is the marker-only rule the
# 2026-10-05 ruling retired (#2205/#2211/#2214, scripts/automod/backlog.py:332-345).
# The machine reads the predicate `is_human_only` instead, so on tick
# run_board-steward_20261106_170235_5e4f5e the steward proposed `draft` for #3852
# — whose acceptance names only `~/lloyd-data/eval/2258/**`, which
# `_names_a_guard` discards by ruled decision (#3027, backlog.py:269-272) — while
# `desired_statuses` wanted `up_next`. That row is one of the 20 disagreements
# keeping the trailing-7-day rate (judged=132, rate=0.8182) under owed-check's
# 0.9 flip floor, and the prompt, not the board, is what produced it.
# ---------------------------------------------------------------------------

def _next_pick_para(prompt: str) -> str:
    """The `next_pick` instruction of a rendered prompt, and nothing around it.

    Scoped by its own anchor to the blank line that closes the paragraph, and
    pinned for length: a window that stopped matching would hand the assertions
    below an empty string, which is the failure #3784's review caught.
    """
    k = prompt.index("`next_pick`:")
    seg = prompt[k:]
    end = seg.index("\n\n")
    para = seg[:end]
    assert len(para) > 100, f"the next_pick paragraph is not the rule: {para!r}"
    return para


def test_the_next_pick_rule_no_longer_excludes_an_item_by_its_acceptance_text():
    """Clause 1. The raw-marker exclusion is gone from the shipped prompt.

    The sentence to remove is verbatim what the prompt shipped: `never one whose
    acceptance begins` followed by the `human-only:` marker in backticks. Both
    halves are asserted absent from the WHOLE rendered prompt (not just the
    paragraph) because the exclusion's harm did not depend on where it sat: a
    model that reads "never one whose acceptance begins human-only:" applies it to
    any item whose acceptance OPENS with the marker, and on #3852 that was an
    acceptance naming only runtime-data paths, which `is_human_only` answers False
    for. A re-worded rule that still keys the exclusion on the acceptance text is
    the same defect, so the paragraph is also asserted to mention the marker
    nowhere — the marker token in backticks with its colon, which the verdict
    value `confirmed-human-only` does not contain.
    """
    prompt = W.build_prompt(events=[], items=[], n_open=0, since_ts=0.0)
    para = _next_pick_para(prompt)

    assert "never one whose acceptance begins" not in prompt, (
        "the acceptance-text exclusion is still shipped, so the steward keeps "
        "applying the marker-only rule the 2026-10-05 ruling retired")
    assert "`human-only:`" not in prompt, (
        f"the marker itself is still in the prompt, wherever the rule sits: {para}")
    assert "acceptance" not in para, (
        f"the next_pick rule still reads the acceptance to decide a pick: {para}")


def test_the_next_pick_human_only_exclusion_is_keyed_on_the_rendered_verdict():
    """Clause 2. The exclusion survives, re-keyed on the value the line prints.

    Dropping the sentence would be the wrong fix: a `confirmed-human-only` item
    must NOT be a next pick, because `desired_statuses` parks it at `draft` "not
    for the unattended loop" and picking it would be a disagreement of the opposite
    sign. So the rule is asserted to name the rendered value — the exact phrase
    `triage=confirmed-human-only`, which is what `_item_line` prints and what
    `triage_value_for` derives from `B.is_human_only` — and to forbid it with a
    `never`. Both scoped to the next_pick paragraph, so a sentence elsewhere in the
    prompt cannot satisfy the node, and the negation scoped to the one
    comma-delimited clause that holds the value, because a paragraph-level `never`
    would still pass on a rule that merely names the value as a fourth thing to
    prefer.
    """
    prompt = W.build_prompt(events=[], items=[], n_open=0, since_ts=0.0)
    para = _next_pick_para(prompt)

    assert "triage=confirmed-human-only" in para, (
        "the next_pick rule names no verdict value, so the model has nothing to "
        f"key the human-only exclusion on: {para}")
    # The value must sit in a clause that FORBIDS it, not merely in the paragraph.
    # A character window, or a `"never" in para`, would pass on "or a line reading
    # `triage=confirmed-human-only`" — which reads as a fourth thing to prefer and
    # is the softening that would put a parked item back in the running. So the
    # comma-delimited clause holding the value is what has to open with the
    # negation.
    holding = [c for c in para.split(",") if "triage=confirmed-human-only" in c]
    assert len(holding) == 1, (
        f"the value is named by {len(holding)} clauses of one rule: {para}")
    assert holding[0].strip().lower().startswith("never"), (
        f"the rule names the value without forbidding it: {holding[0]!r}")
    assert "confirmed-human-only" in W.TRIAGE_VALUES, (
        f"the value the rule keys on is not the field's human-only branch: "
        f"{W.TRIAGE_VALUES}")


def test_a_non_guard_human_only_confirmation_stays_pickable_and_unparked(board):
    """Clause 3. Re-keying does not widen what counts as human-only.

    The trap #3027 set and #3784's fixture already names: an acceptance that OPENS
    `human-only:` but names only a `~/lloyd-data/**` path is NOT a guard — the
    runtime-data root is the loop's own, so `_names_a_guard` skips the token and
    `is_human_only` answers False. #31 below is that shape and #32 is the real
    guard (`~/obsidian/...`, a vault path), so the two must land on opposite sides
    of BOTH surfaces at once: the board line's `triage=` and
    `desired_statuses`'s destination. A fix that keyed next_pick on the marker's
    presence would leave #31 rendered `confirmed` while dropping it from the pick
    list, which is a silent second rail the machine does not have — so the
    assertion is the cross-product, not each half alone: the value the pick rule
    forbids is exactly the value the real guard's line renders, and is not the
    value the runtime-data item's line renders.
    """
    _triage_item(board, 31, "up_next", acceptance=NON_GUARD_MARKER)
    _triage_item(board, 32, "up_next", acceptance=GUARD_MARKER)
    items = B.all_items(("lloyd",), backlog_dir=board)
    assert len(items) == 2, "both items must be on the board to be compared"

    want = B.desired_statuses(S.LEDGER_PATH)
    assert want[31][0] == "up_next", (
        f"the machine stopped wanting to implement an item whose acceptance names "
        f"only runtime-data paths — the fix widened the guard: {want.get(31)}")
    assert want[32][0] == "draft" and "not for the unattended loop" in want[32][1], (
        f"the real guard stopped parking: {want.get(32)}")

    prompt = W.build_prompt(events=[], items=items, n_open=2, since_ts=0.0)
    para = _next_pick_para(prompt)
    v31 = re.findall(r"\btriage=(\S+)", _line(prompt, 31))
    v32 = re.findall(r"\btriage=(\S+)", _line(prompt, 32))
    assert len(v31) == 1 and len(v32) == 1, (f"{_line(prompt, 31)}\n{_line(prompt, 32)}")
    assert v31[0] == "confirmed", (
        f"#31 names no guard and its line says {v31[0]!r}; a widened guard would "
        "park a landable item on a decision nobody made (#3027)")
    assert v32[0] == "confirmed-human-only", f"#32's line says {v32[0]!r}"
    # Whole tokens, never substrings: `triage=confirmed` is a prefix of
    # `triage=confirmed-human-only`, so a `not in para` test on the shorter value
    # fails on a correct prompt — measured on the first run of this node. Each
    # value is compared as the token a board line would print for it.
    forbidden = set(re.findall(r"triage=([a-z][a-z-]*)", para))
    assert forbidden == {"confirmed-human-only"}, (
        f"the pick rule forbids {sorted(forbidden)}, not the one human-only value")
    assert v31[0] not in forbidden, (
        "the pick rule forbids the value the runtime-data item's line renders, so "
        "re-keying became a widening")
    assert v32[0] in forbidden, (
        "the pick rule does not forbid the value the real guard's line renders")


# ── #3962: a confirmed line's verdict disagreement is a note, never a move ─────
#
# #3766 put the verdict on every line, #3784 gave it its fourth value and #3858
# re-keyed `next_pick` onto the rendered value — and the sentence that defines what a
# disagreement MEANS was never touched. It ended "that is worth a proposal with a
# `reason` saying so", and the steward obeyed it against a correct line:
# `~/lloyd-data/sessions/20261107_155534_boardsteward_de38.json` records the model
# writing "**Per instruction**, that's worth a proposal with a reason saying so"
# while proposing `draft` for #3908, whose bare `triage=confirmed` line was right
# (`NON_GUARD_MARKER` is #3908's shape — a `human-only:` acceptance naming only
# runtime-data paths, so `is_human_only` is False and `desired_statuses` wants
# `up_next`). Triage measured three `disagree_sides` rows naming #3908 with the
# machine side `up_next` in the trailing 7-day window (2026-11-07T20:45:17,
# 21:56:56 and 22:01:57 UTC) inside judged=171 / rate=0.8596, under the
# `judged >= 50 AND rate >= 0.90` flip bar the steward is still dry-run on.
#
# The wording cannot be hidden instead: `_item_line` prints `BODY_CHARS` = 280
# characters of body and no acceptance at all (workers/sources/board_steward.py),
# so the model derived "human-only" from the visible first sentence of the body. A
# rule that keys on an excerpt the machine never read is the defect; the binding has
# to forbid the MOVE, and it has to sit in the same field-doc bullet, because that
# bullet's own sentence is the one the session quoted.


def _triage_field_sentence(prompt: str, anchor: str) -> str:
    """The one sentence of the `triage=` field doc that holds `anchor`.

    Scoped to the field-doc bullet (`- **`triage=`` to the next bullet) and to the
    sentence holding the anchor, which is the only scope that grades "the same
    sentence must also say X". #3784's review refused a character window for exactly
    this: a window wide enough to reach the rule is wide enough to contain the
    bullet's header, so a deleted rule still passes. A sentence is delimited here by
    `". "` — the bullet's only terminator — and the result is asserted to hold no
    second one, so a rule split across two sentences reddens the caller instead of
    silently satisfying a per-sentence assertion from the wrong half.
    """
    assert "- **`triage=`" in prompt, "the field is never named to the steward"
    doc = prompt[prompt.index("- **`triage=`"):]
    doc = doc[:doc.index("\n- ")]
    k = doc.find(anchor)
    assert k >= 0, (
        f"{anchor!r} is gone from the `triage=` field doc, so the rule that grounds a "
        f"verdict disagreement has no home there at all: {doc}")
    start = doc.rfind(". ", 0, k) + 2
    end = doc.find(". ", k)
    assert end > k, f"the sentence holding {anchor!r} is unterminated: {doc[k:k+200]!r}"
    sentence = doc[start:end]
    assert anchor in sentence and ". " not in sentence, (
        f"sentence extraction failed for {anchor!r}: {sentence!r}")
    return sentence


def test_a_suspected_mis_rendered_confirmed_verdict_is_a_note_and_never_a_move():
    """Clause 1-2. The disagreement sentence forbids the move and names a real field.

    The old ending — "that is worth a proposal with a `reason` saying so" — is an
    invitation to do the one thing that cost three disagreement rows, and it named a
    field (`reason`) that the reply schema does not have: `STEWARD_SCHEMA` requires
    `["moves","next_pick","next_pick_reason","summary"]` at the top level and
    `["item_id","status","tags_add","tags_remove","note"]` per move, so there is no
    move-less note channel except `summary` and `next_pick_reason`. A binding that
    said "record it as a note" without naming one of those is unsatisfiable, and a
    model that cannot record an observation except as a move will keep making the
    move — which is what #3908's session shows. Hence three assertions on ONE
    sentence: it still says a disagreement means one side is broken, it forbids a
    status move for that item, and it names a reply field that exists. The
    schema membership check is the seam: prompt prose that points at a field the
    grammar will not accept is the bug that produced this item.
    """
    prompt = W.build_prompt(events=[], items=[], n_open=0, since_ts=0.0)
    sentence = _triage_field_sentence(prompt, "one of them is broken")

    assert re.search(r"never[^.]{0,60}\bmove\b", sentence), (
        "the disagreement sentence does not forbid a move — with the negation absent "
        "the field is an invitation again, which is the sentence the steward quoted: "
        f"{sentence!r}")
    named = [f for f in ("summary", "next_pick_reason") if f"`{f}`" in sentence]
    assert named, (
        "the rule sends a suspected mis-rendering nowhere: with no reply field named "
        "there is no way to record it that is not a move, so the note loses to the "
        f"move: {sentence!r}")
    for field in named:
        assert field in W.STEWARD_SCHEMA["properties"], (
            f"the rule names `{field}`, which the reply grammar does not accept — an "
            "instruction the steward cannot render is the one #3908 worked around")

    # The invitation itself has to be gone from the whole bullet, not merely
    # outweighed by a later clause: the model cited this exact phrasing.
    doc = prompt[prompt.index("- **`triage=`"):]
    doc = doc[:doc.index("\n- ")]
    assert "worth a proposal" not in doc, (
        "the bullet still ends its disagreement sentence on an offer to propose, "
        f"which is what the 2026-11-07 ticks followed: {doc}")


def test_the_confirmed_line_binding_names_its_verdict_as_a_whole_token(board):
    """Clause 3. The binding is keyed on the value, whole — never on a prefix.

    #3858's whole-token lesson, applied to the new sentence: `triage=confirmed` is a
    prefix of `triage=confirmed-human-only`, so a substring test cannot tell "the
    rule binds the bare confirmation" from "the rule binds the fourth value" — and
    binding the wrong one is what #3784 exists to prevent, since a
    `confirmed-human-only` line's proposal IS `draft`. So the sentence's named values
    are taken with `re.findall(r"triage=([a-z][a-z-]*)", …)` as a SET and compared to
    the whole token, and the set is then cross-checked against the two board lines
    that actually print those values: the non-guard `human-only:` item, whose line
    renders `confirmed` and which the rule must therefore bind, and the real guard,
    whose line renders `confirmed-human-only` and which it must not.
    """
    prompt = W.build_prompt(events=[], items=[], n_open=0, since_ts=0.0)
    sentence = _triage_field_sentence(prompt, "one of them is broken")

    named = set(re.findall(r"triage=([a-z][a-z-]*)", sentence))
    assert named == {"confirmed"}, (
        f"the binding names {sorted(named)}; it has to name exactly the bare "
        "`confirmed` — naming the human-only value would tell the steward to park "
        "the item the machine wants implemented, which is the #3784 defect in "
        f"reverse, and naming nothing leaves the rule unkeyed: {sentence!r}")
    assert "confirmed" in W.TRIAGE_VALUES and len(W.TRIAGE_VALUES) == 4, (
        f"`confirmed` is not one of the field's values any more: {W.TRIAGE_VALUES}")

    _triage_item(board, 41, "up_next", acceptance=NON_GUARD_MARKER)
    _triage_item(board, 42, "up_next", acceptance=GUARD_MARKER)
    items = B.all_items(("lloyd",), backlog_dir=board)
    assert len(items) == 2, (
        "both sides of the guard pair have to be on the board for the cross-check to "
        "be a comparison and not a guess")
    board_prompt = W.build_prompt(events=[], items=items, n_open=2, since_ts=0.0)
    non_guard = re.findall(r"\btriage=(\S+)", _line(board_prompt, 41))[0]
    guard = re.findall(r"\btriage=(\S+)", _line(board_prompt, 42))[0]
    assert non_guard == "confirmed" and guard == "confirmed-human-only", (
        f"the guard pair drifted ({non_guard!r} / {guard!r}), so the cross-check "
        "below is comparing the rule against values no line prints")
    assert non_guard in named, (
        f"the rule does not bind the value a non-guard human-only line prints "
        f"({non_guard!r}) — which is #3908's shape and the row this item is about")
    assert guard not in named, (
        f"the rule names {guard!r}, the value whose proposal IS `draft`: a confirmed-"
        "human-only line moved to `draft` is the machine's own answer, and a sentence "
        "forbidding a move on it contradicts the bullet above it")


def test_the_binding_does_not_stop_scoring_a_draft_proposal_for_a_non_guard_line(board,
                                                                                  monkeypatch):
    """Clause 4, the half that could only be broken by this change.

    Telling the steward "never propose `draft` for a confirmed line" is only safe if
    a wrong `draft` proposal still costs: the flip gate is `judged >= 50 AND rate >=
    0.90`, so a rail that quietly dropped these proposals from the denominator would
    lift the rate by deleting its own evidence — the same reason #3766's clause 4
    insisted its disagreement survive end-to-end. #41 is that case in #3908's shape:
    confirmed, `NON_GUARD_MARKER` acceptance, machine side `up_next`, line rendering
    bare `confirmed`, and a steward answer of `draft` for the reason the session gave
    out loud. It runs the real `desired_statuses` (not a stub) and the real scoring,
    so the assertion crosses the whole seam: board line → model reply → ledger row.
    A node that only re-read the prompt could not catch a fix that silenced the score.
    """
    from scripts.automod import state as S, state as ST
    from workers.sources import _common as C

    _triage_item(board, 41, "up_next", acceptance=NON_GUARD_MARKER)
    want = B.desired_statuses(S.LEDGER_PATH)
    assert want[41][0] == "up_next", (
        f"the machine stopped wanting to implement a non-guard human-only item, so "
        f"this fixture is no longer #3908: {want.get(41)}")

    items = B.all_items(("lloyd",), backlog_dir=board)
    prompt = W.build_prompt(events=[], items=items, n_open=1, since_ts=0.0)
    assert re.findall(r"\btriage=(\S+)", _line(prompt, 41)) == ["confirmed"], (
        "the line stopped rendering the bare confirmation, so the steward is no "
        "longer being tested on the shape that produced the item")

    raw = [{"item_id": 41, "status": "draft",
            "note": "acceptance names a human-only path"}]

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": raw, "next_pick": 0, "next_pick_reason": "",
                               "summary": "one parked"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)

    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": True})))
    assert out["status"] == "success", out

    row = [e for e in ST.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "board_steward"][-1]
    assert 41 in row["agreement"]["disagree"], (
        f"the new binding deleted a wrong proposal from the score: {row['agreement']}")
    sides = [r for r in row["agreement"]["disagree_sides"] if r["item_id"] == 41]
    assert len(sides) == 1 and sides[0]["status"] == "up_next", (
        f"disagree_sides lost the machine's half of the row this item was filed "
        f"from: {sides}")
    assert row["moves"][0]["item_id"] == 41 and row["moves"][0]["status"] == "draft", \
        row["moves"]


def _tick_with_moves(board, monkeypatch, moves: list[dict]):
    """One applied tick whose answer is `moves`; everything else is production."""
    from workers.sources import _common as C

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": moves, "next_pick": 0,
                               "next_pick_reason": "", "summary": "moved two, picked none"}}

    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    return asyncio.run(W.execute(SimpleNamespace(payload={"apply": True})))


def test_a_tick_that_moved_items_names_the_one_it_moved_first(board, monkeypatch):
    """#3992 clause 4: the column gets the backlog reference for the item this tick
    moved, in the steward's own reported order.

    Of the 939 board-steward successes in the 30-day window read on 2026-11-09, 88
    moved exactly one item and 91 moved more than one, so a reference column that
    holds one value names the first and points at the ledger row for the rest.
    """
    from workers.queue import backlog_artifact
    _ten_item_board(board)
    out = _tick_with_moves(board, monkeypatch, [
        {"item_id": 7, "status": "up_next", "note": "ready: acceptance recorded"},
        {"item_id": 4, "status": "draft", "note": "needs-human, parked"},
    ])

    assert out["status"] == "success"
    assert out["artifact_path"] == backlog_artifact(7) == "backlog#7"
    assert out["artifact_path"] != backlog_artifact(4), (
        "the column holds one reference, so it is the first reported move and not an "
        "arbitrary one — the full set is in `meta.agreement` and the ledger row")


def test_a_tick_that_moved_nothing_says_so_rather_than_leaving_the_column_empty(
        board, monkeypatch):
    """The load-bearing half of clause 4, and the reason this source needed the token
    more than any other: 760 of the 939 ticks in the 30-day window moved nothing, and
    until now those 760 were indistinguishable in the table from the 179 that moved
    something. A census reading `artifact_path <> ''` scored board-steward as producing
    nothing on every run; with the token it scores 760 stated nothings and 179 yields.

    `runs.summary` already distinguishes them for a human ("0 move(s)"), but `pool.py`
    only warns when BOTH summary and artifact are missing, so no log signal ever fired
    and nothing was countable in SQL.
    """
    from workers.queue import NO_ARTIFACT
    _ten_item_board(board)
    out = _tick_with_moves(board, monkeypatch, [])

    assert out["status"] == "success"
    assert out["artifact_path"] == NO_ARTIFACT == "__none__"
    assert out["artifact_path"] != "", (
        "an empty string here is the unsaid case again — the whole point of the token "
        "is that attested-nothing and unsaid are two values, not two readings of one")


# --------------------------------------------------------------------------------------
# #4080 — a move whose note cites a round its item has never run is not a judgement about
# that item. The row this is filed on: the tick at 2026-11-12T00:30:46Z proposed
# `{item_id: 3024, status: in_progress}` with a note about round SM_20261111_233110, which
# belongs to #4074, and #3024 was scored as disagreeing with the state machine about its
# own status. Four clauses: never written even with `apply` on, abstained out of both
# `disagree` and `judged`, a real policy disagreement still counted, and #3547's flag gate
# unchanged for a `none` item.
# --------------------------------------------------------------------------------------

#: A round id in the shape `scripts.automod.state` mints. #4's, in these nodes.
OWNED_ROUND = "SM_20261010_010203"


def test_a_move_citing_a_round_its_item_never_ran_is_never_written_even_with_apply_on(
        board, monkeypatch):
    """Clause 1 end to end: the citation gates the WRITE, on an `apply: true` tick.

    #4 owns `SM_20261010_010203` — its `backlog_implement` row names it — and #8 owns no
    round at all, so #8's flag is `none` and nothing else on the board could stop it.
    Two moves, one tick, `apply: true` (a status write, which is the only thing #3024 was
    spared by: the disagreeing tick ran with `apply: false`, so a dry run hid the fact
    that a move about the wrong row would have been written to the board):

      #8  proposed `in_progress` with a note citing #4's round. #8 owns no round, so the
          citation is #3024's exactly: prose about one row, booked on another. It must
          not reach the front matter.
      #7  proposed `in_progress` with no round cited, flag `none`. Written, which is what
          makes #8's refusal a refusal: the same writer, the same tick, the same flag,
          one withheld and one applied.
    """
    from scripts.automod import state as S
    from workers import board_disagreement as D
    _ten_item_board(board)
    _promotion(S.LEDGER_PATH, 4, OWNED_ROUND, "d" * 40, settle=True, base=1)
    # Unmeasured window: this node rules a write, it does not open a disagreement item
    # on the live board off a fixture tick's disagreement (#3089's bar is the gate).
    monkeypatch.setattr(D, "window_now",
                        lambda **kw: _window7(judged=10, agree=10, disagree=0, rate=1.0))

    out = _tick_with_moves(board, monkeypatch, [
        {"item_id": 8, "status": "in_progress",
         "note": f"round {OWNED_ROUND} is under observation; held"},
        {"item_id": 7, "status": "in_progress", "note": "ready: acceptance recorded"},
    ])
    assert out["status"] == "success", out

    row = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "board_steward"][-1]
    assert _load(board, 8).status == "up_next", (
        "#8's front matter was rewritten on the strength of a round it has never run")
    assert _load(board, 7).status == "in_progress", (
        "the un-cited move was withheld too, so #8 proves nothing about a refusal")
    assert [m["item_id"] for m in row["applied"]] == [7]
    assert row["agreement"]["misbound"] == [8]
    assert [m["item_id"] for m in row["moves"]] == [8, 7], (
        "the proposal itself stays in the row: this is a withheld write, not a move "
        "the steward never made")
    assert "1 mis-bound to another item's round" in out["summary"], out["summary"]


def test_the_ownership_behind_the_citation_is_read_from_the_ledger_and_nowhere_else(
        tmp_path):
    """Clause 1's boundary: what is caught, and the four shapes that are not.

    The detector is one predicate over the same map `unfinished_rounds` reads at
    `board_steward.py:717` to name an item's rounds on the board line, so this node
    pins the reading and not just the outcome: an item the ledger gives a round is
    never caught citing it, and an item it gives none is caught with an empty set
    because there is no round it could own.

    Four shapes stay uncaught, and each is a decision rather than an accident:

      * a note that cites a round the item DOES own — the legitimate case, and the
        one the board line itself renders ("round=… is in flight");
      * a note with no round id in it, including a missing `note`, which
        `parse_steward` defaults to `""` — the gate is about a claim specific enough
        to falsify and about nothing else;
      * a token that is not the shape, `SM_2026_1010_01020`: `state.make_round_id`
        is `SM_` plus `%Y%m%d_%H%M%S`, so 8 digits and 6, and a near-miss is prose;
      * two ids where one is owned: `cited <= owned` is the predicate, so naming a
        round of yours alongside one that is not theirs is still caught.
    """
    led = tmp_path / "ledger.jsonl"
    _promotion(led, 4, OWNED_ROUND, "d" * 40, settle=True, base=1)
    cite = lambda iid, note: [{"item_id": iid, "status": "in_progress", "note": note}]

    assert W.misbound_moves(cite(4, f"round {OWNED_ROUND} is under observation"),
                            led) == set(), "an item owns the round it cites"
    assert W.misbound_moves(cite(8, f"round {OWNED_ROUND} is under observation"),
                            led) == {8}, "#3024's shape: a citation of another row's round"
    assert W.misbound_moves(cite(8, "no round cited, just judgement"), led) == set()
    assert W.misbound_moves([{"item_id": 8, "status": "in_progress"}], led) == set(), \
        "an omitted note is an empty one, and an empty one cites nothing"
    assert W.misbound_moves(cite(8, "round SM_2026_1010_01020 is in flight"), led) == set(), \
        "a token that is not the minted shape is prose, not a citation"
    assert W.misbound_moves(cite(8, f"rounds {OWNED_ROUND} and SM_9 both open"),
                            led) == {8}, (
        "one owned citation must not launder a foreign one: the predicate is a subset, "
        "not an intersection")
    # The authority is `implement_history`, and #4's ownership is one row: deleting it
    # takes #4 with the rest of them, which is the accepted cost the function's
    # docstring names — a round that has written no `backlog_implement` row cannot be
    # cited without an abstention, whether or not it is real.
    bare = tmp_path / "bare.jsonl"
    _led(bare, event="promoted", round_id=OWNED_ROUND, commit="d" * 40, ts=2)
    assert W.misbound_moves(cite(4, f"round {OWNED_ROUND} is under observation"),
                            bare) == {4}, (
        "a `promoted` row alone does not make the round the item's: the map is the one "
        "the board line reads, so an item absent from it is an item with no rounds")


def test_a_mis_bound_move_is_abstained_out_of_disagree_and_out_of_judged(board,
                                                                        monkeypatch):
    """Clause 2: the abstention leaves the score's numerator AND its denominator.

    Three moves, one tick, `apply: false` (what ships today). `expected` is the machine
    the disagreeing tick faced: #7 and #8 both held at `up_next` where they sit, #9 held
    at `up_next` too.

      #7  proposes `up_next` — agrees.
      #8  proposes `in_progress` citing #4's round — mis-bound, abstained.
      #9  proposes `draft` where the machine wants `up_next` — a real disagreement, and
          the control that keeps clause 2 from being clause 3 in disguise.

    Three numbers, all computed by this one `agreement()` over one `expected`:

      scored WITH a ledger (what ships after this change): `disagree` [9] only,
        `judged` 2, `rate` 0.5.
      scored WITHOUT one (what the code did before it): `disagree` [8, 9], `judged` 3,
        `rate` one third. #3024 sat in that list and in that denominator, and
        `judgement_aggregate` sums both into the figure that decides the `apply` flip.
      the same tick with the mis-bound move NEVER SENT: `judged` 2, `rate` 0.5 — equal to
        the shipped case, which is what "leaving `rate` unchanged by it" means, stated as
        arithmetic instead of asserted.

    `no_opinion` and `missed` keep it out too: both are built from the surviving moves,
    so a row cannot be reported as a row the steward failed to address.
    """
    from scripts.automod import state as S
    from scripts.automod import backlog as B
    from workers import board_disagreement as D
    from workers.sources import _common as C
    _ten_item_board(board)
    led = S.LEDGER_PATH
    _promotion(led, 4, OWNED_ROUND, "d" * 40, settle=True, base=1)
    # An unmeasured window, so this tick's disagreement files nothing (#3089's bar):
    # these nodes score a tick, they do not open an item on the live board.
    monkeypatch.setattr(D, "window_now",
                        lambda **kw: _window7(judged=10, agree=10, disagree=0, rate=1.0))
    expected = {7: ("up_next", "ready: confirmed with an acceptance"),
                8: ("up_next", "unconfirmed, held"),
                9: ("up_next", "already in the pool")}
    current = {7: "up_next", 8: "up_next", 9: "up_next"}
    monkeypatch.setattr(B, "desired_statuses", lambda *a, **k: dict(expected))
    moves = [{"item_id": 7, "status": "up_next", "note": "ready, leave it"},
             {"item_id": 8, "status": "in_progress",
              "note": f"round {OWNED_ROUND} is under observation; held"},
             {"item_id": 9, "status": "draft", "note": "umbrella, park it"}]

    async def turn(prompt, **kw):
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": moves, "next_pick": 0, "next_pick_reason": "",
                               "summary": "one held, one parked, one mis-bound"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out["status"] == "success", out
    row = [e for e in S.read_events(path=led) if e.get("event") == "board_steward"][-1]
    agree = row["agreement"]

    assert agree["misbound"] == [8]
    assert agree["disagree"] == [9], "the mis-bound id is still inside the disagreement"
    assert agree["judged"] == 2 and agree["rate"] == pytest.approx(0.5)
    assert 8 not in agree["no_opinion"] and 8 not in agree["missed"], (
        "the abstention has to reach the two lists built from the surviving moves, or "
        "the row is still scored, just scored as a gap instead of as a disagreement")
    assert agree["stale"] == [], "the two abstentions stay separate lists"
    assert agree["disagree_sides"] == [{"item_id": 9, "status": "up_next",
                                        "why": "already in the pool"}], (
        "the machine's half is recorded for the disagreement that remains, and for no "
        "other id — a side for an abstained row would put it back in the disagreement")
    assert [m["item_id"] for m in row["moves"]] == [7, 8, 9], (
        "the ledger event keeps the proposal and the abstention beside it; publishing "
        "the abstention is the point, so neither half may be dropped")
    assert "1 mis-bound to another item's round" in out["summary"], out["summary"]

    before = W.agreement(moves, expected, current)
    assert before["disagree"] == [8, 9] and before["misbound"] == [], \
        "without a ledger the check cannot run, which is the pre-change score exactly"
    assert before["judged"] == 3 and before["rate"] == pytest.approx(1 / 3), \
        "#3024's own arithmetic: 3 judged, one third, on a move that was never about it"
    assert agree["rate"] > before["rate"], (
        "abstaining the row must RAISE the rate, since the pre-change code was "
        "depressing it with a decision nobody made about this item")

    never_sent = W.agreement([m for m in moves if m["item_id"] != 8], expected, current,
                             ledger=led)
    assert (never_sent["judged"], never_sent["rate"]) == (agree["judged"], agree["rate"]), \
        "the tick must score exactly as it would have with the mis-bound move absent"


def test_a_genuine_policy_disagreement_still_counts_after_the_mis_bound_abstention(
        board, monkeypatch):
    """Clause 3: the abstention is narrow, and a real disagreement still costs the rate.

    Same board, same machine, one tick with the mis-bound move REMOVED and the genuine
    one kept — #9 proposed `draft` where `desired_statuses` holds `up_next`. It lands in
    `disagree`, it is in `judged`, and `rate` is 0.0 against the 1.0 of the tick that
    proposed only its agreeing move. Those two numbers in the same node are what clause 3
    claims: an abstention that also swallowed a policy disagreement would keep `judged`
    at 1 and `rate` at 1.0 here, and the whole instrument would quietly stop measuring
    the thing it exists to measure.
    """
    from scripts.automod import state as S
    from scripts.automod import backlog as B
    from workers import board_disagreement as D
    _ten_item_board(board)
    _promotion(S.LEDGER_PATH, 4, OWNED_ROUND, "d" * 40, settle=True, base=1)
    monkeypatch.setattr(D, "window_now",
                        lambda **kw: _window7(judged=10, agree=10, disagree=0, rate=1.0))
    expected = {7: ("up_next", "ready: confirmed with an acceptance"),
                9: ("up_next", "already in the pool")}
    current = {7: "up_next", 9: "up_next"}
    monkeypatch.setattr(B, "desired_statuses", lambda *a, **k: dict(expected))

    policy = W.agreement([{"item_id": 9, "status": "draft", "note": "umbrella, park it"}],
                         expected, current, ledger=S.LEDGER_PATH)
    only_agreement = W.agreement([{"item_id": 7, "status": "up_next",
                                   "note": "ready, leave it"}],
                                 expected, current, ledger=S.LEDGER_PATH)

    assert policy["disagree"] == [9] and policy["agree"] == []
    assert policy["judged"] == 1, "a policy disagreement that is not counted is clause 3 " \
                                  "inverted: the instrument would measure nothing"
    assert policy["misbound"] == [] and policy["stale"] == [], \
        "nothing about this move cites a round and nothing moved mid-tick"
    assert policy["rate"] == 0.0 and only_agreement["rate"] == 1.0, (
        f"a real disagreement must LOWER the rate: {policy['rate']} against "
        f"{only_agreement['rate']}")


def test_the_new_gate_leaves_what_the_round_flag_gate_refuses_alone(board, monkeypatch):
    """Clause 4: #3547's write gate is not widened, and the two refusals stay named apart.

    `move_contradicts_flag` returns False for `flag == "none"` on purpose, and the reason
    is a window: a round that has opened but not yet written its `backlog_implement` row
    reads `none`, so refusing `in_progress` then would stop a legitimate claim. That
    predicate is not one character different after this round, and this node is the
    witness on both sides of the seam:

      * the predicate itself, `in_progress` on a `none` item whose note cites a round, and
        the same move with no note at all — False on both.
      * the attribution on a real tick: #8's withheld move arrives with `suppressed`
        EMPTY and `misbound` holding it, which is only true because the flag gate let it
        through and the citation gate is the sole thing that stopped it. Fold the new
        refusal into `suppressed` and this node goes red, which is the difference between
        two gates and one gate pretending to be two.
    """
    from scripts.automod import state as S
    _ten_item_board(board)
    _promotion(S.LEDGER_PATH, 4, OWNED_ROUND, "d" * 40, settle=True, base=1)

    for note in (f"round {OWNED_ROUND} is in flight", ""):
        assert W.move_contradicts_flag({"item_id": 8, "status": "in_progress",
                                        "note": note}, "none") is False, (
            "the flag gate grew a second condition, which is the widening #3547's "
            "reason for the `none` arm was written to prevent")
    assert W.move_contradicts_flag({"item_id": 4, "status": "draft", "note": "park it"},
                                   "round_in_flight") is True, \
        "the arm that DOES refuse is still refusing"

    # The attribution on a real `apply: true` tick. #8's flag is `none` — its only
    # ledger row is #4's promotion, which names no round of #8's — so the flag gate
    # passes it, `suppressed` is empty, and the sole thing standing between that move
    # and the front matter is the citation gate. A `suppressed` list holding #8 would
    # mean the two refusals had been merged into one number.
    from workers import board_disagreement as D
    monkeypatch.setattr(D, "window_now",
                        lambda **kw: _window7(judged=10, agree=10, disagree=0, rate=1.0))
    _tick_with_moves(board, monkeypatch, [
        {"item_id": 8, "status": "in_progress",
         "note": f"round {OWNED_ROUND} is under observation; held"}])
    row = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "board_steward"][-1]

    assert row["suppressed"] == [], (
        "the flag gate took the refusal, which is the widening clause 4 forbids")
    assert row["agreement"]["misbound"] == [8] and row["applied"] == []
    assert _load(board, 8).status == "up_next"


def test_both_abstentions_are_published_in_the_same_block_and_the_same_row(board,
                                                                          monkeypatch,
                                                                          tmp_path):
    """#4080 clause 2, end to end: the two abstentions coexist in one tick.

    #3752's `stale` and #4080's `misbound` are separate predicates — a row can be
    abstained by one and not the other — so a block carrying both is a claim that the
    same `agreement()` call published both lists and that the ledger row carried them.
    A tick with only one of them would still pass if the second list were overwritten
    by the first, or if the writer picked one key.

    One turn, three moves, `apply: true`, and #10 has a ledger row of its own after the
    table read:

      #7   proposed `up_next`, the machine wants `up_next` — agrees, and is the only
           thing left in `judged`, so `rate` is 1.0 over exactly one scored decision.
      #8   proposed `in_progress` citing round SM_20261010_010203, which #4 owns —
           mis-bound (#4080), and its front matter is untouched.
      #10  proposed `up_next` where the machine wants `draft`, and a `backlog_triage`
           row for it lands DURING the turn, stamped after the table read — stale
           (#3752), so it leaves `disagree` for the reason it has always left for.

    #8 is mis-bound and NOT stale (it owns no row after the read); #10 is stale and NOT
    mis-bound (its note cites no round). Neither list may absorb the other's id, and the
    one thing `missed` does NOT do is lose #10: that list asks whether the steward
    proposed the status the machine wants, which is a question about the answer, not
    about the freshness of the table — so it keeps counting the stale row, unchanged by
    this round, while #8 stays out of it (a move was made about #8, so this tick cannot
    claim the steward skipped it).
    """
    from scripts.automod import backlog as B
    from scripts.automod import state as S
    from workers import board_disagreement as D
    from workers.sources import _common as C
    _ten_item_board(board)
    led = S.LEDGER_PATH
    _promotion(led, 4, OWNED_ROUND, "d" * 40, settle=True, base=1)
    monkeypatch.setattr(D, "window_now",
                        lambda **kw: _window7(judged=10, agree=10, disagree=0, rate=1.0))
    expected = {7: ("up_next", "ready: confirmed with an acceptance"),
                8: ("up_next", "unconfirmed, held"),
                10: ("draft", "triage stale (>72h)")}
    monkeypatch.setattr(B, "desired_statuses", lambda *a, **k: dict(expected))
    moves = [{"item_id": 7, "status": "up_next", "note": "ready, leave it"},
             {"item_id": 8, "status": "in_progress",
              "note": f"round {OWNED_ROUND} is under observation; held"},
             {"item_id": 10, "status": "up_next", "note": "looks ready to me"}]

    async def turn(prompt, **kw):
        # #10's ruling lands mid-turn, stamped ahead of the instant the tick read the
        # table — the row that makes the disagreement stale rather than real.
        _ev(event="backlog_triage", item_id=10, verdict="confirmed",
            acceptance="a check exists",
            created_at=(datetime.now(timezone.utc)
                        + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 1,
                "structured": {"moves": moves, "next_pick": 0, "next_pick_reason": "",
                               "summary": "one agree, one mis-bound, one stale"}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": True})))
    assert out["status"] == "success", out
    row = [e for e in S.read_events(path=led) if e.get("event") == "board_steward"][-1]
    agree = row["agreement"]

    assert agree["stale"] == [10] and agree["misbound"] == [8], agree
    assert agree["disagree"] == [] and agree["agree"] == [7]
    assert agree["judged"] == 1 and agree["rate"] == pytest.approx(1.0), \
        "with both abstentions out, one scored decision is left and it agreed"
    assert agree["missed"] == [10] and 8 not in agree["missed"], agree
    assert agree["disagree_sides"] == [], \
        "no disagreement survived to have a side recorded"
    # Published once, in one place: the `agreement` block inside the event is the copy
    # `board_disagreement` sums, and it is where `stale` has always lived. A second,
    # flat copy of the same list beside `suppressed` was considered and rejected — two
    # writers of one measurement is the drift this file keeps having to fix.
    assert row["agreement"]["misbound"] == [8] == out["meta"]["agreement"]["misbound"], (
        "the event's block and the run meta have to carry the same abstention; the row "
        "is the copy a later reader sums")
    assert _load(board, 8).status == "up_next", (
        "the mis-bound move was written on an applied tick, which is clause 1 undone")
    assert "1 abstained (the board moved mid-tick)" in out["summary"], out["summary"]
    assert "1 mis-bound to another item's round" in out["summary"], out["summary"]


# ── #4381: the reply the model gave has to reach the run record ───────────
#
# `pool.py:2426` writes `response_json=norm["response"]` for every source, and
# `pool.py:526` defaults a missing key to `""` — so a source that never sets the
# key stores an empty string and nothing ever complains. That is how all 913
# board-steward rows came to hold no reply (`workers/pool.py`'s writer vs
# `workers/sources/autoresearch.py:107`, which has always set it). What is stored
# here is the half the ledger does not keep: the `board_steward` event holds the
# moves as APPLIED and, since #3677, the machine's `disagree_sides`; the reply is
# what `workers/board_disagreement.py:317` says it never had.


def _one_reply_tick(board, monkeypatch, *, text, structured, structured_error=""):
    """One real steward tick whose engine call returns exactly this answer.

    `run_prompt_in_session` is the only fake: the parse, the ledger row and the
    record handed back to the pool are production, because what is under test is
    the record — the three fields `steward_response` builds and the status and
    summary that must not move.
    """
    from workers.sources import _common as C

    async def turn(prompt, **kw):
        return {"text": text, "session_id": "sess-1", "stop_reason": "stop",
                "num_turns": 3, "structured": structured,
                "structured_error": structured_error}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    return asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))


def _steward_ledger_row():
    return [e for e in S.read_events(path=S.LEDGER_PATH)
            if e.get("event") == "board_steward"][-1]


def test_a_successful_tick_returns_the_reply_as_parseable_json(board, monkeypatch):
    """#4381 clause 1: the record carries what the model ANSWERED, as JSON.

    The reply has to survive as parseable JSON rather than as prose, because the
    reader is SQL: `WorkQueue.run_rollup_by_source` counts liveness with
    `json_valid(response_json)`, and a value that is not JSON is invisible to the
    check this column exists to satisfy.

    The stored `structured` is the finalizer's dict BEFORE `parse_steward` clamped
    it, and the ledger's copy after: a ruler comparing the steward against the
    state machine has to see what was said, not what fitted. The note here is
    deliberately over `NOTE_MAX` so the two copies differ and the difference is
    the assertion.
    """
    _ten_item_board(board)
    long_note = _admissible_overflow(W.NOTE_MAX, W.NOTE_GRAMMAR_MAX)
    answered = {"moves": [{"item_id": 7, "status": "up_next", "note": long_note,
                           "tags_add": [], "tags_remove": []}],
                "next_pick": 7, "next_pick_reason": "oldest ready item",
                "summary": "one move, no objections"}
    said = "Item 7 is confirmed with an acceptance, so it goes up_next."
    out = _one_reply_tick(board, monkeypatch, text=said, structured=answered)

    assert out["status"] == "success", out
    blob = out.get("response")
    assert blob, "absent `response` is what booked an empty column for 913 runs"
    stored = json.loads(blob)
    assert stored["text"] == said, "the raw reply, not a restatement of it"
    assert stored["structured"]["next_pick"] == 7
    assert stored["structured"]["summary"] == "one move, no objections"
    assert stored["structured"]["moves"][0]["note"] == long_note, (
        "the run record is the answer as given; clamping it here would make the "
        "two copies indistinguishable and the seam untestable")
    row = _steward_ledger_row()
    assert row["moves"][0]["note"] != long_note and len(row["moves"][0]["note"]) < len(long_note), (
        "the ledger half stayed the clamped answer, so the two copies are one value")
    assert "truncated" not in stored, "a reply inside the bound must not read as cut"


def test_a_rejected_reply_is_returned_with_the_validators_complaint(board, monkeypatch):
    """#4381 clause 3: a bad answer is inspectable too.

    The `no structured answer` path is the one a ruler most needs the reply for —
    it is where the steward said something the finalizer could not shape — and all
    it recorded was that a divergence happened. The status, summary and meta stay
    exactly as #2370 and #3088 left them (`test_a_diverged_tick_still_returns_the_
    record_it_returned_before` pins the rest); the reply is additive.
    """
    _ten_item_board(board)
    complaint = "structured output does not match the schema: 'moves' is required"
    out = _one_reply_tick(board, monkeypatch,
                          text="I think item 7 should go up_next, but here is prose.",
                          structured=None, structured_error=complaint)

    assert out["status"] == "failed", out
    stored = json.loads(out["response"])
    assert stored["structured_error"] == complaint, (
        "the complaint is what tells a ruler the reply was malformed, not absent")
    assert stored["text"] == "I think item 7 should go up_next, but here is prose."
    assert stored["structured"] is None, (
        "there was no structured answer, and an empty dict would invent one")


def test_the_three_returns_without_a_reply_keep_an_empty_response(board, monkeypatch):
    """#4381 clause 5: no turn, no reply — and no invented one.

    The skipped, drain-skip and timeout returns happen before or without a `run`,
    so their `response` stays the pool's default `""`, which is what every
    historical board-steward row holds. `""` is also what must NOT be readable as
    an empty reply from a turn that ran: the two are told apart by `status`.
    Asserted as whole dicts, because a stray key here is the same bug in reverse.
    """
    from workers.pool import normalize_result
    from workers.queue import WorkQueue
    from workers.sources._common import DrainActive, TurnTimeout

    # (a) skipped: nothing happened since the last pass — an empty ledger and no
    # item ready to move, so the tick never reaches the engine at all.
    _write(board, 12, "draft")
    out = asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))
    assert out == {"status": "skipped",
                   "summary": "nothing happened since the last pass"}, out

    def _raise(exc):
        from workers.sources import _common as C

        async def turn(prompt, **kw):
            raise exc
        monkeypatch.setattr(C, "run_prompt_in_session", turn)
        return asyncio.run(W.execute(SimpleNamespace(payload={"apply": False})))

    _ten_item_board(board)                      # now there IS something to consider
    drained = _raise(DrainActive("draining for a landing"))
    assert drained == {"status": "skipped", "summary": "draining for a landing"}, drained
    timed = _raise(TurnTimeout("900 s elapsed"))
    assert timed == {"status": "failed", "summary": "steward turn timed out: 900 s elapsed"}, timed

    for record in (out, drained, timed):
        assert "response" not in record, record

    # The pool's own default is what those three rows get: an absent key becomes
    # `""`, so `runs.response_json` keeps the value it has always had for them.
    q = WorkQueue(board.parent / "w.db")
    item = q.get(q.enqueue(source=W.NAME, kind="tick", payload={}))
    for record in (out, drained, timed):
        assert normalize_result(item, record)["response"] == "", record


def test_an_over_bound_reply_is_cut_without_breaking_the_json():
    """#4381 clause 4's policy, at the unit: the cap is fitted to, not hit.

    `queue.record_run` slices `response_json[:50000]` and `pool.normalize_result`
    slices at the same width one frame earlier. A value that arrives over it is cut
    mid-JSON, and `json_valid()` — the very predicate `run_rollup_by_source` measures with
    — goes to 0 for the biggest replies, which is the worst possible time to lose
    the row. So `steward_response` shortens `text` first and keeps the object whole.
    """
    blob = W.steward_response(text="q" * (W.RESPONSE_MAX_CHARS + 5_000),
                              structured={"moves": [], "next_pick": 4, "summary": "net 0"})
    assert len(blob) <= W.RESPONSE_MAX_CHARS, len(blob)
    stored = json.loads(blob)
    assert stored["truncated"] is True
    assert stored["text"].endswith(W.RESPONSE_TRUNCATED_MARK), "the cut is marked, not silent"
    assert stored["structured"]["summary"] == "net 0", (
        "the cheapest field is the one cut: the answer outlives the prose")


def test_an_answer_too_big_to_fit_still_stays_parseable():
    """The shrink arm behind `text`: prose and complaint together can still exceed
    the bound, and the last field has to be cuttable too — collapsed to its JSON
    text, which is what lets it be sliced at all — rather than left for the writer's
    blind slice.
    """
    blob = W.steward_response(text="t" * 1_000,
                              structured={"moves": [{"note": "z" * 60_000}]},
                              structured_error="e" * 20_000)
    assert len(blob) <= W.RESPONSE_MAX_CHARS, len(blob)
    stored = json.loads(blob)
    assert stored["truncated"] is True
    assert isinstance(stored["structured"], str), (
        "an answer too big to fit is kept as its own JSON text, so it is kept")


# ── #4412 clause 4: the abstention reason quotes the steward, not 60 characters of it ──

def test_the_abstention_reason_carries_the_steward_note_in_full(tmp_path, monkeypatch):
    """`desired_statuses` quotes a `steward_move` note whole, so the item's own activity
    line says why the steward moved it.

    The note here is copied from the live ledger row for item 3068 —
    `draft with triage=confirmed and round_flag=none; last event is the confirmation, so
    the verdict moves it (#898).` — 111 characters, which is why the old `claim[2][:60]`
    read so much like a claim the steward had left unfinished: the slice lands on exactly
    60 characters and ends `...; last event `. A reader of the item concluded from it that
    the steward believed `triage=confirmed` while the machine's own confirmed map did not,
    and filed that as the root cause of the round-trip (#4412's own triage had to correct
    it). Nothing was truncated on the steward's side; only the quote was.

    `parse_steward` caps a note at 400 characters, so the reason stays bounded without a
    slice at the call site."""
    from scripts.automod import state as S
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    led = tmp_path / "promotions.jsonl"
    (d / "9999-churning-item.md").write_text(
        "---\nstatus: up_next\npriority: medium\nboard: lloyd\ntags: [backlog]\n"
        "created: '2026-11-01T00:00:00+00:00'\n---\n\n# Churning item\n\nBody.\n",
        encoding="utf-8")
    note = ("draft with triage=confirmed and round_flag=none; last event is the "
            "confirmation, so the verdict moves it (#898).")
    assert len(note) == 112, "the slice under test cuts at 60; this note must clear it"
    S.append_event({"event": "backlog_triage", "item_id": 9999, "verdict": "unverifiable",
                    "check": "grep -n x", "evidence": "unreadable store"}, path=led)
    S.append_event({"event": "steward_move", "item_id": 9999, "from": "draft",
                    "to": "up_next", "note": note, "tick": "board-steward:tick#3068",
                    "ts": time.time() + 600}, path=led)

    want = B.desired_statuses(led, None)
    assert want[9999][0] == "up_next", want[9999]
    why = want[9999][1]
    assert note in why, why
    assert f"{note[:60]});" not in why, f"the note is sliced at 60 again: {why}"


# ─────────── #4378: a row is earned on a SECOND consecutive disagreeing tick ───────────
#
# `file_or_update` took its ids from THIS tick alone, so a single tick's disagreement — one
# reading of one board by one model — was enough to put a row on the lloyd board and page a
# ruling. #4378's bar: an id earns a body row and a `new_item_ids` entry only if the
# IMMEDIATELY PRECEDING `board_steward` ledger event scored it as a disagree too.
#
# The size of that bar on live traffic, measured rather than assumed. Walking the ledger
# (`/home/alansrobotlab/.local/state/lloyd-automod/promotions.jsonl`, 16,632 rows by
# `wc -l` on 2026-11-24) — each of its 270 `board_steward` events against ITS OWN
# immediately-preceding steward row, which is what `prior_tick_disagree` sees at filing time —
# 17 ticks scored a non-empty `disagree`, naming 21 id-ticks over 14 ids: 4 id-ticks earn
# (every one of them #4276, disagreeing on two ticks in a row), 1 is withheld (#4548, new
# against its preceding tick), and 16 land on the no-comparison fallback because the tick
# before them recorded no disagreement at all. So on this week's traffic the guard bites once
# and the fallback answers sixteen times — the clause-3 shape is the common case, not a
# curiosity, and a node that only ever exercised the withholding path would describe none of
# it. The nodes below pin all three outcomes.
# The prior tick is read from that event's `agreement` block through the store, never from a
# new state file, and a tick with nothing to compare against files exactly as it did before.
#
# The one-tick delay is the cost, and it is not small: the median gap between consecutive
# steward ticks measured 21.1 minutes on 2026-11-19 and its p90 was 112.9 minutes, so a real
# disagreement is sometimes held back for about two hours before it earns its row. Whether
# that lag is acceptable is owed-check's ruling, recorded as owed on #4378; these nodes only
# pin that the bar is what was ruled.

#: Sentinel for `agreement=_NO_BLOCK`: the key is absent, as it is on every row a tick
#: wrote before #1688 shipped the block.
_NO_BLOCK = object()


def _steward_row(led, *, disagree=(), session_id="tick-prior", agreement=_NO_BLOCK,
                 moves=None):
    """One `board_steward` LEDGER row, written the way a tick writes it.

    The payload is copied from `workers/sources/board_steward.py:1903-1915` — the same
    keys, the real `S.append_event` writer, and `agreement` built by the house `_agree()`
    helper — because every failure mode of this guard is a mismatch between what THAT
    writer puts on the row and what the reader asks for. Not `_steward_tick`, which runs a
    whole tick: this writes only the row a preceding tick left in the store.

    `agreement=` overrides the built block: `_NO_BLOCK` omits the key (a row written
    before #1688 shipped the record), `None` or a string is a block that is not a dict,
    and a dict names its ids exactly.
    """
    from scripts.automod import state as S
    row = {"event": "board_steward", "apply": False, "events": 3,
           "items_shown": len(disagree),
           "moves": (DIS_MOVES if moves is None else moves) if disagree else [],
           "suppressed": [], "next_pick": 0, "next_pick_reason": "",
           "applied": [], "session_id": session_id,
           "summary": "tick over 5 shown items"}
    row["agreement"] = _agree(*disagree) if agreement is _NO_BLOCK else agreement
    S.append_event(row, path=led)
    return led


def test_an_id_that_disagreed_only_this_tick_earns_no_row(tmp_path):
    """Clause 1: one tick's disagreement buys nothing.

    The preceding tick disagreed about two of this tick's five ids. Those two earn their
    rows; the three that appear for the first time this tick get no row, are absent from
    the created event's `new_item_ids`, and are named in Coverage instead.
    """
    board = _new_board(tmp_path, "b-4378-earn")
    led = tmp_path / "ledger.jsonl"
    _steward_row(led, disagree=DIS_AGREE[:2], session_id="tick-a")

    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(), backlog_dir=board, ledger=led,
                           session_id="tick-b")
    assert out is not None and out["action"] == "filed", out
    held = DIS_AGREE[2:]
    assert out["ids"] == DIS_AGREE[:2], out
    assert out["rows"] == len(DIS_AGREE[:2]), out

    body = _filed(board)[0].body
    rows = "\n".join(_rows(body))
    for i in DIS_AGREE[:2]:
        assert f"# {i}:" in rows or f"#{i}:" in rows, f"earned id {i} lost its row:\n{rows}"
    for i in held:
        assert f"#{i}" not in rows, f"single-tick id {i} earned a row:\n{rows}"

    from scripts.automod import state as S
    events = [e for e in S.read_events(led, 0)
              if e.get("event") == "board_steward_disagreement"]
    assert len(events) == 1, events
    assert events[0]["new_item_ids"] == DIS_AGREE[:2], events[0]
    assert not (set(held) & set(events[0]["new_item_ids"])), events[0]


def test_a_merged_filing_also_withholds_an_id_that_has_not_persisted(tmp_path):
    """Clause 1's MERGED leg: the guard is upstream of both writers, not just of `_create`.

    An item is already open (filed off a store with no steward row at all, so the guard had
    nothing to compare and the first filing is the pre-#4378 shape). The tick now disagrees
    about all five ids with a preceding tick that disagreed about two: the merge may add only
    those two, and the three that have not persisted appear in its body and its `new_item_ids`
    never — with the merge's own Coverage naming them.
    """
    from scripts.automod import state as S
    board = _new_board(tmp_path, "b-4378-merge")
    one = DIS_AGREE[0]
    first = D.file_or_update(_agree(one),
                             [m for m in DIS_MOVES if m["item_id"] == one],
                             {one: DIS_EXPECTED[one]},
                             window=_window7(), backlog_dir=board,
                             ledger=tmp_path / "led-first.jsonl", session_id="tick-0")
    assert first and first["action"] == "filed" and first["ids"] == [one], first

    led = _steward_row(tmp_path / "led-merge.jsonl", disagree=DIS_AGREE[:2],
                       session_id="tick-a")
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(), backlog_dir=board, ledger=led,
                           session_id="tick-b")
    assert out is not None and out["ids"] == DIS_AGREE[:2], out
    held = set(DIS_AGREE[2:])

    body = _filed(board)[0].body
    rows = "\n".join(_rows(body))
    for i in held:
        assert f"#{i}" not in rows, f"a single-tick id merged in:\n{rows}"
    cov = body[body.index("## Coverage"):]
    assert "3 of this tick's ids were withheld as **not-yet-earned**" in cov, cov
    assert f"#{DIS_AGREE[2]}" in cov and f"#{DIS_AGREE[4]}" in cov, cov

    events = [e for e in S.read_events(led, 0)
              if e.get("event") == "board_steward_disagreement"]
    assert events, "the merge wrote no ledger row"
    assert not (held & set(events[-1]["new_item_ids"] or [])), events[-1]


def test_an_id_that_disagreed_on_the_preceding_tick_earns_the_same_bytes_as_today(
        tmp_path):
    """Clause 2: the guard is a gate on WHO is named, not a re-write of what a row says.

    Two boards, the same five ids, the same window. One has a preceding tick that disagreed
    about every id, the other has no `board_steward` row at all — the shape that falls back
    to the pre-#4378 path — so the second filing IS what this build shipped before the
    change. The created bodies must be byte-identical, and the two ledger rows must agree
    field by field on everything but the instant.
    """
    from scripts.automod import state as S
    led_prior = _steward_row(tmp_path / "led-prior.jsonl", disagree=list(DIS_AGREE),
                             session_id="tick-a")
    led_none = tmp_path / "led-none.jsonl"

    with_prior = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                                  window=_window7(),
                                  backlog_dir=_new_board(tmp_path, "b-4378-prior"),
                                  ledger=led_prior, session_id="tick-b")
    no_prior = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                                window=_window7(),
                                backlog_dir=_new_board(tmp_path, "b-4378-none"),
                                ledger=led_none, session_id="tick-b")
    assert with_prior and no_prior and with_prior["ids"] == no_prior["ids"], (
        f"{with_prior} vs {no_prior}")

    b_prior = _filed(tmp_path / "b-4378-prior")[0].body
    b_none = _filed(tmp_path / "b-4378-none")[0].body
    if b_prior != b_none:
        diff = "\n".join(list(difflib.unified_diff(
            b_none.splitlines(), b_prior.splitlines(), "no-prior-tick", "prior-tick",
            lineterm=""))[:60])
        raise AssertionError(f"a guarded filing rewrote the body:\n{diff}")

    fields = ("judged", "disagree", "rate", "window_since", "tick_item_ids",
              "new_item_ids", "leg", "action", "item_id")
    ev_p = [e for e in S.read_events(led_prior, 0)
            if e.get("event") == "board_steward_disagreement"][-1]
    ev_n = [e for e in S.read_events(led_none, 0)
            if e.get("event") == "board_steward_disagreement"][-1]
    assert {k: ev_p.get(k) for k in fields} == {k: ev_n.get(k) for k in fields}, (
        f"{ev_p} vs {ev_n}")


def test_the_preceding_tick_is_found_past_the_hundred_event_read_window(tmp_path):
    """Clause 3's trap, held by a fixture that makes the naive read blind on purpose.

    `S.read_events` defaults to the newest 100 rows, and it counts EVERY event, not steward
    events: over the live ledger's 321 gaps between consecutive steward ticks the median gap
    is 22 intervening rows and 30 of them exceed 100, so a default read finds no preceding
    tick and silently takes the fallback — the guard would look green while never firing.
    This fixture packs 120 `backlog_triage` rows between the two ticks, which is inside that
    measured range, and asserts first that the default read really does see no steward row
    at all. Then the guard must still withhold three of this tick's five ids.
    """
    from scripts.automod import state as S
    led = tmp_path / "ledger.jsonl"
    _steward_row(led, disagree=DIS_AGREE[:2], session_id="tick-a")
    for n in range(120):
        S.append_event({"event": "backlog_triage", "item_id": 9000 + n,
                        "verdict": "confirmed", "acceptance": "the check passes"},
                       path=led)
    assert len([e for e in S.read_events(led)
                if e.get("event") == "board_steward"]) == 0, (
        "the fixture stopped being the trap: a default read sees a steward row, so this "
        "node no longer distinguishes a whole-file read from a bounded one")
    assert D.prior_tick_disagree(led, session_id="tick-b") == set(DIS_AGREE[:2])

    board = _new_board(tmp_path, "b-4378-deep")
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(), backlog_dir=board, ledger=led,
                           session_id="tick-b")
    assert out is not None and out["ids"] == DIS_AGREE[:2], out
    solo = _steward_row(tmp_path / "led-solo.jsonl", disagree=list(DIS_AGREE),
                        session_id="tick-b")
    assert D.prior_tick_disagree(solo, session_id="tick-b") is None, (
        "the store holds one steward row and it is THIS tick's: excluded by session, "
        "nothing is left, so the answer is that there is no preceding tick at all — "
        "reading that row as the preceding one would be the tick comparing itself "
        "with itself")


def test_the_guard_keeps_its_state_in_the_ledger_it_already_read(tmp_path):
    """Clause 3's other half: no new store. A second file holding the last tick's ids is a
    second truth that can disagree with the ledger, and the ledger already carries the ids —
    `agreement.disagree` is what `_window_pairs` reads for the same purpose.
    """
    led = tmp_path / "ledger.jsonl"
    _steward_row(led, disagree=[3024], session_id="tick-a")
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert D.prior_tick_disagree(led, session_id="tick-b") == {3024}
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before, (
        "reading the preceding tick created a file; the guard's only store is the ledger")


def test_a_tick_with_no_usable_preceding_tick_files_exactly_as_before(tmp_path):
    """Clause 3's fallback, on every shape that has no prior data to compare against.

    No `board_steward` row at all; the row predates #1688 and carries no `agreement` key;
    the key holds a string instead of a dict; the block is a dict that names no disagreeing
    id. Each of those must produce the body and the `new_item_ids` the build shipped before
    #4378 produces — a guard that cannot answer must not also mute the alarm, or the
    instrument that decides the flip would lose its reader every time a tick diverged.
    """
    from scripts.automod import state as S

    def filing(name, led):
        board = _new_board(tmp_path, f"b-4378-{name}")
        out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                               window=_window7(), backlog_dir=board, ledger=led,
                               session_id="tick-b")
        assert out and out["ids"] == DIS_AGREE, f"{name}: {out}"
        assert out["rows"] == len(DIS_AGREE), f"{name}: {out}"
        return _filed(board)[0].body

    baseline = filing("base", tmp_path / "led-base.jsonl")

    # (a) no `board_steward` row anywhere in the store: only a triage row.
    led = tmp_path / "led-none.jsonl"
    S.append_event({"event": "backlog_triage", "item_id": 3024, "verdict": "confirmed",
                    "acceptance": "the check passes"}, path=led)
    assert filing("no-steward-row", led) == baseline

    # (b) the row predates the record: it carries no `agreement` key at all, which is the
    # shape `_window_pairs` calls "cannot be attributed either way".
    led = tmp_path / "led-absent.jsonl"
    S.append_event({"event": "board_steward", "apply": False, "events": 3,
                    "items_shown": 5, "moves": DIS_MOVES, "applied": [],
                    "session_id": "tick-a", "summary": "a row from before #1688"},
                   path=led)
    assert filing("block-absent", led) == baseline

    # (c) the key holds a string, so there is no id list to read out of it.
    led = _steward_row(tmp_path / "led-string.jsonl", session_id="tick-a",
                       agreement="judged 5, agree 0")
    assert filing("block-not-a-dict", led) == baseline

    # (d) a dict naming no disagreement: an agreeing preceding tick is not evidence of
    # persistence, and withholding this tick on that reading would be inventing a bar the
    # ruling did not set — #4378 files it as shipped instead.
    led = _steward_row(tmp_path / "led-empty.jsonl", session_id="tick-a", agreement=_agree())
    assert filing("block-names-nothing", led) == baseline
    assert D.prior_tick_disagree(led, session_id="tick-b") is None


def test_the_current_tick_s_own_ledger_row_is_not_the_preceding_tick(tmp_path):
    """The seam across `workers/sources/board_steward.py:1903`.

    The tick appends its OWN `board_steward` row and only then files (`:1929`), so the newest
    steward event when `file_or_update` runs is the filing tick itself. Read naively, the
    guard would compare the tick with its own answer and let every id through — a gate that
    never closes and never says so. The row is therefore excluded by the `session_id` the
    tick already writes on its own row, and this node holds both halves: with the session
    handed down, only the id the EARLIER tick disagreed about earns; without it, the tick's
    own row is read as the preceding tick and all five earn, which is the failure.
    """
    led = tmp_path / "ledger.jsonl"
    _steward_row(led, disagree=DIS_AGREE[:2], session_id="tick-a")
    _steward_row(led, disagree=list(DIS_AGREE), session_id="tick-b")

    guarded = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                               window=_window7(),
                               backlog_dir=_new_board(tmp_path, "b-4378-self"),
                               ledger=led, session_id="tick-b")
    assert guarded and guarded["ids"] == DIS_AGREE[:2], guarded

    naive = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                             window=_window7(),
                             backlog_dir=_new_board(tmp_path, "b-4378-naive"),
                             ledger=led)
    assert naive and naive["ids"] == DIS_AGREE, (
        "without the session id the newest steward row IS the current tick's, so this is "
        "the shape that must fail open to today's behaviour rather than withhold at random")


def test_the_steward_tick_hands_its_session_id_to_the_filer(tmp_path, monkeypatch):
    """The wiring, from the writer's side: `_file_quietly` must carry `session_id` down.

    A keyword the caller never passes is a guard that silently never fires in production
    while every node above stays green, so the pin is on the forwarding function and on the
    call site in the source that makes it — not on `file_or_update` alone. `_file_quietly`
    reads the window itself, unlike `file_or_update`, so `window_now` is pinned here rather
    than at the call site below, which is about a keyword, not about the trigger.
    """
    monkeypatch.setattr(D, "window_now", lambda **kw: _window7())
    led = tmp_path / "ledger.jsonl"
    _steward_row(led, disagree=DIS_AGREE[:2], session_id="tick-a")
    out = D._file_quietly(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                          backlog_dir=_new_board(tmp_path, "b-4378-wired"),
                          ledger=led, session_id="tick-b")
    assert out and out["ids"] == DIS_AGREE[:2], out

    src = (Path(__file__).resolve().parents[1] / "workers" / "sources"
           / "board_steward.py").read_text(encoding="utf-8")
    call = src[src.index("_file_quietly, agree"):]
    assert "session_id=" in call.split("rules_note=")[0], (
        "the tick stopped passing its own session id into the filing, so its own ledger "
        f"row becomes the preceding tick:\n{call[:400]}")


def test_coverage_names_each_id_the_persistence_guard_withheld(tmp_path):
    """Clause 5, first half: a withheld id is named, never subtracted.

    A count without the ids is the silent subtraction this filing already swears against —
    a reader could not tell a disagreement that has not persisted from a read that failed.
    The bullet sits beside the abstained one, and the zero is stated like every other figure.
    """
    led = tmp_path / "ledger.jsonl"
    _steward_row(led, disagree=DIS_AGREE[:2], session_id="tick-a")
    board = _new_board(tmp_path, "b-4378-cov")
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(), backlog_dir=board, ledger=led,
                           session_id="tick-b")
    assert out is not None, out
    cov = _filed(board)[0].body
    cov = cov[cov.index("## Coverage"):]
    assert "3 of this tick's ids were withheld as **not-yet-earned**" in cov, cov
    for i in DIS_AGREE[2:]:
        assert f"#{i}" in cov, f"withheld id {i} not named: {cov}"
    assert "did not score them as a disagreement" in cov, cov
    assert (cov.index("not represented here") < cov.index("were abstained")
            < cov.index("withheld as **not-yet-earned**")
            < cov.index("named ids are unrecoverable")), (
        f"the bullet moved away from the abstention it sits beside:\n{cov}")

    earned_all = _steward_row(tmp_path / "led-all.jsonl", disagree=list(DIS_AGREE),
                              session_id="tick-a")
    board2 = _new_board(tmp_path, "b-4378-cov2")
    D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED, window=_window7(),
                     backlog_dir=board2, ledger=earned_all, session_id="tick-b")
    cov2 = _filed(board2)[0].body
    cov2 = cov2[cov2.index("## Coverage"):]
    assert "- 0 of this tick's ids were withheld as **not-yet-earned**." in cov2, cov2


def test_a_tick_whose_ids_are_all_withheld_files_nothing_at_all(tmp_path):
    """Clause 5, second half: the guard can empty the filing, so it must empty it cleanly.

    Every id this tick scored disagreed about is new since the preceding tick, so there is
    no row to write: no item, no `### Added` section on the one that is already open, and no
    `board_steward_disagreement` ledger row. `abstained` cannot save it either — that list
    is built from ids OUTSIDE this tick's disagreement, which is exactly why its comment
    could claim the filing never empties while `tick_ids` survived the abstention.
    """
    from scripts.automod import state as S
    led = tmp_path / "ledger.jsonl"
    _steward_row(led, disagree=[999], session_id="tick-a")
    board = _new_board(tmp_path, "b-4378-empty")
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED,
                           window=_window7(), backlog_dir=board, ledger=led,
                           session_id="tick-b")
    assert out is None, out
    assert _filed(board) == [], f"an all-withheld tick filed: {_filed(board)}"
    assert [e for e in S.read_events(led, 0)
            if e.get("event") == "board_steward_disagreement"] == []


def test_the_persistence_guard_leaves_the_window_figures_and_leg_wording_untouched(
        tmp_path):
    """Clause 4: the guard is not allowed to move a measure or re-name a leg.

    A fixture whose ids all clear the bar. `judged`, `rate` and the window sentence are the
    aggregate's own numbers — withholding a row changes who is named, never how much of the
    window was measured — and the abstained list and #4251's direction wording are pinned to
    the bytes they shipped with.
    """
    from scripts.automod import state as S
    w = _window7()
    led = _steward_row(tmp_path / "ledger.jsonl", disagree=list(DIS_AGREE),
                       session_id="tick-a")
    out = D.file_or_update(_agree(*DIS_AGREE), DIS_MOVES, DIS_EXPECTED, window=w,
                           backlog_dir=_new_board(tmp_path, "b-4378-figs"),
                           ledger=led, session_id="tick-b")
    assert out and out["action"] == "filed", out
    body = _filed(tmp_path / "b-4378-figs")[0].body
    window_line = (f"- the window itself: {w['judged']} judged, `rate={w['rate']}` "
                   f"against owed-check's {D.FLIP_MIN_RATE} floor, since "
                   f"{w['window_since']}.")
    assert window_line in body, body[body.index("## Coverage"):]
    assert "- 0 of the window's ids were abstained as already ruled." in body, body
    assert BELOW_FLOOR_CLAIM in body, (
        "the rate-leg direction sentence moved: " + body[:600])

    ev = [e for e in S.read_events(led, 0)
          if e.get("event") == "board_steward_disagreement"][-1]
    assert ev["judged"] == w["judged"] and ev["rate"] == w["rate"], ev
    assert ev["leg"] == D.LEG_RATE, ev
    assert ev["disagree"] == w["disagree"], ev


# ── #4591: the disagreement this filing was about stops being one ────────────

def test_a_steward_move_back_to_draft_after_a_re_triage_is_booked_as_agreement(board):
    """#4591 clause 4: the tick scores the fix as agreement, on the #4276 shape.

    #4591 was filed because a tick booked `steward_move` -> `up_next` for #4276 against a
    table that answered `up_next` ONLY by abstaining — "the machine abstains until this
    item lands, is promoted or is re-triaged" — while the steward, reading a board line of
    `triage=none`, proposed `draft`. That pair is the disagreement, and the item's own row
    is 12 of the 21 rows in the window that measured the sub-floor rate.

    So the ending has to be visible from the scorer's side too. The fixture is the live
    item's shape exactly: `status: up_next` with no `triage:` key, a confirmed verdict,
    the steward's `up_next` row, and a `backlog_retriage` mark after it. `expected` is the
    real `desired_statuses` output, not a hand-typed pair, because the clause is about
    what the tick books, and a hand-typed table would score whatever it was told.

    Before the fix this node reddens on its own first line: the table answers `up_next`
    and the proposal lands in `disagree` with the abstention sentence in
    `disagree_sides` — which is the row the filing quoted verbatim.
    """
    base = time.time() + 3600
    _write(board, 9276, "up_next")                      # no `triage:` key, like #4276
    _ev(event="backlog_triage", item_id=9276, verdict="confirmed",
        acceptance="prints draft", ts=base)
    _ev(event="steward_move", item_id=9276, frm="draft", to="up_next", ts=base + 600,
        note="triage=confirmed and round_flag=none; back to draft for its second triage",
        tick="t-fixture")
    _ev(event="backlog_retriage", item_id=9276, round_id="SM_FIXTURE", ts=base + 1200,
        reason="its one unattended attempt is spent; the loop re-triages")

    expected = B.desired_statuses(S.LEDGER_PATH, None)
    assert expected[9276][0] == "draft", (
        f"the table still answers the steward's own status: {expected[9276]}")
    assert "abstains" not in expected[9276][1], expected[9276][1]

    a = W.agreement([{"item_id": 9276, "status": "draft"}], expected, {9276: "up_next"})
    assert a["disagree"] == [] and a["disagree_sides"] == [], (
        f"the tick still books the move as a disagreement: {a['disagree']} / "
        f"{a['disagree_sides']}")
    assert a["agree"] == [9276], a
    assert a["judged"] == 1 and a["rate"] == 1.0, (
        f"a booked agreement that left the denominator moves the rate the flip is "
        f"judged from: {a['judged']} / {a['rate']}")
    assert a["machine_moves"] == 1, (
        "the table does want a move on this item, so the tick has a decision to score")
