"""#904: the board-decision measure joins queue decisions to their outcomes.

Everything runs over a fixture ledger in `tmp_path`; nothing reads the live
automod state dir (conftest points it at scratch anyway)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.automod import board_decisions as BD

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc).timestamp()
H = 3600.0
D = 86400.0


def _ledger(tmp_path, rows):
    path = tmp_path / "ledger.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _item(iid, status):
    return SimpleNamespace(id=iid, status=status)


def _confirm(iid, ts, **kw):
    return {"event": "backlog_triage", "item_id": iid, "verdict": "confirmed", "held": False,
            "closed": False, "verdict_source": "structured", "ts": ts, **kw}


def _moved(iid, ts, frm, to, reason, **kw):
    # `by` is NULL on most real rows; the measure must never need it.
    return {"event": "status_moved", "item_id": iid, "from": frm, "to": to,
            "reason": reason, "ts": ts, **kw}


def _entries(reading):
    return {e["item_id"]: e for e in reading["entries"]}


# ── clause 1: each entry, its deciding event, its terminal state ───────────


def test_each_promotion_carries_its_deciding_event_and_terminal_state(tmp_path):
    t0 = NOW - 3 * D
    rows = [
        # 1: single triage confirms straight into the pool, then lands and closes.
        _confirm(1, t0),
        {"event": "backlog_implement", "item_id": 1, "phase": "started", "ts": t0 + H},
        {"event": "item_landed", "item_id": 1, "closed": True, "acceptance": "met", "ts": t0 + 5 * H},
        # 2: confirmed into a full pool (held), released a day later, handed to a person.
        _confirm(2, t0, held=True),
        {"event": "backlog_confirm_released", "item_id": 2, "moved": True, "ts": t0 + D},
        _moved(2, t0 + D, "draft", "up_next", "released from hold: the implement pool has room"),
        _moved(2, t0 + D + 2 * H, "in_progress", "draft",
               "its one unattended attempt is spent; a human decides (reopen_item to grant another)"),
        # 3: the steward, applying, moves an item into the pool; still there.
        {"event": "board_steward", "apply": True, "ts": t0 + 2 * H,
         "applied": [{"item_id": 3, "status": "up_next", "tags_add": [], "tags_remove": [],
                      "note": "x", "applied": True}]},
        # 4: a group triage's umbrella.
        _confirm(4, t0 + 3 * H, umbrella=True, members=[40, 41]),
        # 5: a round starts on an item no ledger event put in the pool.
        {"event": "backlog_implement", "item_id": 5, "phase": "started", "ts": t0 + 4 * H},
        # 6: a dry-run steward proposes, and proposals are not decisions.
        {"event": "board_steward", "apply": False, "ts": t0 + 5 * H,
         "moves": [{"item_id": 6, "status": "up_next"}], "applied": []},
    ]
    r = BD.board_decisions(_ledger(tmp_path, rows), now=NOW,
                           items=[_item(1, "done"), _item(2, "draft"), _item(3, "up_next"),
                                  _item(4, "up_next"), _item(5, "in_progress")])
    e = _entries(r)
    assert set(e) == {1, 2, 3, 4, 5}, "a dry-run steward proposal moved nothing"

    assert (e[1]["decider"], e[1]["via"], e[1]["terminal"]) == ("triage", "confirm", "landed")
    assert e[1]["entered_at"] == BD._iso(t0) and e[1]["terminal_at"] == BD._iso(t0 + 5 * H)
    assert e[1]["dwell_s"] == 5 * H, "entry and terminal timestamps make dwell a subtraction"

    assert (e[2]["decider"], e[2]["via"]) == ("triage", "released")
    assert e[2]["decision_at"] == BD._iso(t0), "a release is credited to the confirm it held"
    assert e[2]["entered_at"] == BD._iso(t0 + D)
    assert e[2]["terminal"] == "needs_human" and e[2]["dwell_s"] == 2 * H

    assert e[3]["decider"] == "steward" and e[3]["terminal"] is None and e[3]["dwell_s"] is None
    assert e[4]["decider"] == "group_triage"
    assert (e[5]["decider"], e[5]["decision_at"]) == ("unrecorded", None)
    assert r["promotions"]["by_decider"] == {"triage": 2, "steward": 1, "group_triage": 1,
                                             "unrecorded": 1}


def test_a_sweep_retirement_and_a_reoffer_are_attributed_from_their_source_events(tmp_path):
    t0 = NOW - 2 * D
    rows = [
        _confirm(7, t0),
        # A re-offer inside the pool is not a second promotion.
        _moved(7, t0 + H, "in_progress", "up_next", "offered again — infra: the turn never reported"),
        # The sweep retires it from the pool: the terminal state is a close.
        {"event": "backlog_triage", "item_id": 7, "verdict": "stale", "closed": True,
         "sweep_batch": "sw-1", "auto": True, "ts": t0 + 2 * H},
        # A person's reopen_item re-offer is its own decision.
        _moved(8, t0 + 3 * H, "draft", "up_next", "offered again — reopened: a person decided"),
    ]
    r = BD.board_decisions(_ledger(tmp_path, rows), now=NOW)
    e = _entries(r)
    assert len(r["entries"]) == 2
    assert e[7]["terminal"] == "closed" and e[7]["dwell_s"] == 2 * H
    assert e[8]["decider"] == "reopen"


def test_an_open_entry_whose_item_was_closed_off_the_ledger_reads_closed(tmp_path):
    r = BD.board_decisions(_ledger(tmp_path, [_confirm(9, NOW - D)]), now=NOW,
                           items=[_item(9, "done")])
    (e,) = r["entries"]
    assert e["terminal"] == "closed_off_ledger" and e["terminal_at"] is None


def test_promotions_outside_the_window_are_not_reported(tmp_path):
    r = BD.board_decisions(_ledger(tmp_path, [_confirm(1, NOW - 8 * D), _confirm(2, NOW - D)]),
                           now=NOW, days=7)
    assert [e["item_id"] for e in r["entries"]] == [2]


# ── clause 2: fewer than 5 promotions is insufficient, with the count ──────


@pytest.mark.parametrize("n", [0, 1, 4])
def test_fewer_than_five_promotions_is_insufficient_with_no_rate(tmp_path, n):
    rows = [_confirm(i, NOW - H * (i + 1)) for i in range(n)]
    o = BD.board_decisions(_ledger(tmp_path, rows), now=NOW)["promotions"]["outcome"]
    assert o["verdict"] == "insufficient"
    assert o["promotions"] == n
    assert not any(k.endswith("rate") for k in o), f"no rate below the minimum: {o}"
    assert "dwell_h_median" not in o


def test_five_promotions_are_measured(tmp_path):
    rows = [_confirm(i, NOW - H * (i + 2)) for i in range(5)]
    rows += [{"event": "item_landed", "item_id": 0, "closed": True, "ts": NOW - H},
             {"event": "item_landed", "item_id": 1, "closed": True, "ts": NOW - H}]
    o = BD.board_decisions(_ledger(tmp_path, rows), now=NOW)["promotions"]["outcome"]
    assert o["verdict"] == "measured" and o["promotions"] == 5
    assert o["done_rate"] == pytest.approx(0.4) and o["landed_rate"] == pytest.approx(0.4)
    assert o["terminal"] == {"landed": 2, "open": 3}


def test_the_insufficient_line_names_the_count(tmp_path):
    s = BD.summary(BD.board_decisions(_ledger(tmp_path, [_confirm(1, NOW - H)]), now=NOW))
    line = BD.health_line(s)
    assert "outcome insufficient (1 < 5)" in line and "reached done" not in line


# ── clause 3: per-day counts, zero days included ───────────────────────────


def test_zero_promotion_days_appear_as_a_streak_of_days(tmp_path):
    # Promotions six days ago and three days ago; nothing since.
    rows = [_confirm(1, NOW - 6 * D), _confirm(2, NOW - 6 * D + H), _confirm(3, NOW - 3 * D)]
    r = BD.board_decisions(_ledger(tmp_path, rows), now=NOW, days=7)
    days = r["per_day"]
    assert [d["day"] for d in days] == ["2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20",
                                        "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]
    assert [d["promotions"] for d in days] == [0, 2, 0, 0, 1, 0, 0, 0]
    assert days[1]["by_decider"] == {"triage": 2}
    assert r["zero_streak"] == {"longest": 3, "current": 3}


def test_an_empty_window_is_all_zero_days_not_no_rows(tmp_path):
    r = BD.board_decisions(_ledger(tmp_path, []), now=NOW, days=3)
    assert len(r["per_day"]) == 4 and all(d["promotions"] == 0 for d in r["per_day"])
    assert r["zero_streak"] == {"longest": 4, "current": 4}


# ── clause 4: retired out of the pool, later moved back ────────────────────


def test_retired_items_later_moved_back_are_counted_with_their_ids(tmp_path):
    t0 = NOW - 4 * D
    spent = "its one unattended attempt is spent; a human decides (reopen_item to grant another)"
    rows = [
        # 10: handed to a person, then a person's reopen puts it back in the pool.
        _confirm(10, t0),
        _moved(10, t0 + H, "in_progress", "draft", spent, needs_human=True),
        _moved(10, t0 + 2 * H, "draft", "up_next", "offered again — reopened: grant another"),
        # 11: handed to a person, the loop's re-triage confirms it again.
        _confirm(11, t0),
        _moved(11, t0 + H, "in_progress", "draft", spent),        # legacy row: no flag
        {"event": "backlog_retriage", "item_id": 11, "ts": t0 + 2 * H},
        _confirm(11, t0 + 3 * H),
        # 12: parked by the sweep, later confirmed into the pool.
        {"event": "backlog_sweep", "batch_id": "sw", "item_ids": [12], "judged": {"12": "keep"},
         "ranked": {"12": "low/small"}, "parked": 1, "ts": t0},
        _confirm(12, t0 + D),
        # 13: closed by triage; a person reopened it in Mission Control (no ledger row).
        {"event": "backlog_triage", "item_id": 13, "verdict": "stale", "closed": True, "ts": t0},
        # 14: closed by triage and still closed: not a reversal.
        {"event": "backlog_triage", "item_id": 14, "verdict": "stale", "closed": True, "ts": t0},
        # 15: handed to a person and still waiting: not a reversal.
        _moved(15, t0, "in_progress", "draft", spent, needs_human=True),
        # 16: a flag of False overrides the reason text.
        _moved(16, t0, "in_progress", "draft", spent, needs_human=False),
    ]
    r = BD.board_decisions(_ledger(tmp_path, rows), now=NOW,
                           items=[_item(13, "draft"), _item(14, "done"), _item(15, "draft")])
    rr = r["retire_reopen"]
    assert rr["ids"] == [10, 11, 12, 13]
    assert rr["reopened"] == 4 and rr["retired"] == 6
    assert rr["by_kind"]["needs_human"] == {"retired": 3, "reopened": 2, "after_retriage": 1,
                                            "by_reopen_item": 1}
    assert rr["by_kind"]["parked"]["reopened"] == 1
    assert rr["by_kind"]["closed"] == {"retired": 2, "reopened": 1, "after_retriage": 0,
                                       "by_reopen_item": 0}
    back = {b["item_id"]: b for b in r["reopened"]}
    assert back[13]["how"] == "board status draft" and back[13]["back_at"] is None
    assert back[11]["after_retriage"] is True and back[10]["via"] == "reopen"


def test_a_reopen_before_the_retirement_is_not_counted(tmp_path):
    rows = [_confirm(20, NOW - 2 * D),
            _moved(20, NOW - D, "in_progress", "draft", "a human decides", needs_human=True)]
    assert BD.board_decisions(_ledger(tmp_path, rows), now=NOW)["retire_reopen"]["reopened"] == 0


def test_summary_drops_the_listings_but_keeps_the_ids(tmp_path):
    rows = [_confirm(1, NOW - H)]
    s = BD.summary(BD.board_decisions(_ledger(tmp_path, rows), now=NOW))
    assert "entries" not in s and "reopened" not in s
    assert "ids" in s["retire_reopen"]
    json.dumps(s)   # it rides a ledger row and a JSON payload
