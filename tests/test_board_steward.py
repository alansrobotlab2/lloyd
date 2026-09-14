"""Cut 3 of senses-not-supervision: one steward turn beside the state machine.

Shipped with `apply: false`. Every pass records how far the steward's answer
agrees with what `desired_statuses` would have done, and that record — not a
week of hope — is what decides whether `apply` flips. The one rule that is
never the steward's: it cannot set `done`.
"""

from __future__ import annotations

import json
import time
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


def test_agreement_is_perfect_when_nothing_should_move():
    assert W.agreement([], {1: ("draft", "w")}, {1: "draft"})["rate"] == 1.0


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
                          "held": 0, "triaged": 1, "total": 6}
    assert h["draft"]["total"] == sum(v for k, v in h["draft"].items() if k != "total")
    assert h["up_next"] == {"total": 3, "umbrellas": 1, "singles": 2, "never_attempted": 2,
                            "ready": 2, "unready": 1}
    assert h["flow"]["24h"] == {"created": 1, "closed": 1, "net": 0}
    assert h["flow"]["7d"] == {"created": 9, "closed": 1, "net": 8}
    assert h["self_spawned_open"] == 3
    assert h["implement_pool"] == {"ready": 2, "bound": 20, "floor": 20}


def test_board_flow_reads_each_stamp_in_its_writers_clock(board):
    """`created` is naive local (the MCP store and the Mission Control router
    use `datetime.now()`), `completed` is naive UTC (this module's closers),
    and a close with no `completed` came from a local-time writer. Read all
    as UTC, the two sides of the 24 h window sat seven hours apart here."""
    import os
    import time as _time
    old = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    _time.tzset()
    try:
        now = datetime.now(timezone.utc)
        local = lambda h: (now - timedelta(hours=h)).astimezone().replace(tzinfo=None).isoformat()
        utc = lambda h: (now - timedelta(hours=h)).replace(tzinfo=None).isoformat()
        _write(board, 1, "draft", created=local(20))          # in
        _write(board, 2, "draft", created=local(30))          # out; read as UTC it was 23 h old
        _write(board, 3, "done", hours_old=240, completed=utc(3))                      # in
        _write(board, 4, "done", hours_old=240, updated=local(26))                     # out; as UTC, 19 h
        _write(board, 5, "done", hours_old=240, completed=utc(30), updated=local(1))  # out: completed wins
        flow = B.board_flow(backlog_dir=board, now=now.timestamp())
        assert flow["24h"] == {"created": 1, "closed": 1, "net": 0}
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
