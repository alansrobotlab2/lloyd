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
                   "A triage verdict moves the item", "An `umbrella` is an ordinary confirmed item"):
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
    assert cfg["model"] == "secondary"


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
