"""Cut 3 of senses-not-supervision: one steward turn beside the state machine.

Shipped with `apply: false`. Every pass records how far the steward's answer
agrees with what `desired_statuses` would have done, and that record — not a
week of hope — is what decides whether `apply` flips. The one rule that is
never the steward's: it cannot set `done`.
"""

from __future__ import annotations

import json
import sqlite3
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
