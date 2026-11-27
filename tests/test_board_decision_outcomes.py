"""#904: the board-decision measure joins queue decisions to their outcomes.

Everything runs over a fixture ledger in `tmp_path`; nothing reads the live
automod state dir (conftest points it at scratch anyway)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from scripts.automod import backlog as B, state as S, worktree as W
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


# ── #4621: the landed attempt-spent park yields to a round the store counts unfinished ──
#
# `backlog.desired_statuses` parked #4616 `draft` + needs-human on 2026-11-27 while the
# same store still counted one of its rounds unfinished: the board steward scored that
# tick as a disagreement, and the reaper-side note in `workers/sources/autocode.py`
# (`_unfinished_round_note`) refused to move the item at all one minute later with
# "`items_with_unfinished_rounds` still counts this item mid-round". These nodes read the
# reconcile's own table over fixture rows, the way `tests/test_backlog_unattended.py`
# does for the rest of that function.

def _board(tmp_path, monkeypatch):
    """A scratch board plus ledger, the two globals `desired_statuses` reads."""
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def _open_item(d, iid, *, status="draft"):
    created = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": "lloyd", "tags": ["backlog"]}
    path = d / f"{iid}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\n"
                    f"Do the thing.\n", encoding="utf-8")
    return path


NOT_MET = {"acceptance": "not_met", "landed": True, "deferred_to": [], "summary": "",
           "spawned": [], "clause_outcomes": [
               {"clause": 1, "outcome": "not_met", "evidence": "", "deferred_to": []}]}


def _spent_landed_park(d, iid, *, rolled_back=True):
    """An item the landed park fires on, built the way the landing sweep leaves it.

    Round one lands with a `not_met` clause (promoted, settled, swept — which writes
    the item's `automod_landed` marker and its `item_landed` row and leaves the item
    open); round two then runs and never promotes, which is what makes
    `implement_outcomes` answer `spent`: the documented rule offers a not_met landing
    exactly one more unattended attempt.

    `rolled_back=True` appends the `rollback_succeeded` row for the landing's commit,
    and that is what makes the fixture usable for clause 2 at all: the shared
    predicate's promotion arm holds every item whose promoted commit no rollback took
    off `main` — measured on the live store on 2026-11-27, 264 of the 266 items it
    holds are held by that arm alone — so a fixture that left the promotion standing
    could never show the park firing on an item with no unfinished round.
    `rolled_back=False` is #4616's own sequence: the promotion stands, and with it the
    predicate's hold.
    """
    _open_item(d, iid)
    S.append_event({"event": "backlog_implement", "item_id": iid, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": iid, "phase": "finished",
                    "round_id": f"SM_{iid}_ONE", "stop_reason": "stop", "num_turns": 40,
                    "outcome": NOT_MET}, path=S.LEDGER_PATH)
    commit = f"{iid:040x}"
    S.append_event({"event": "promoted", "round_id": f"SM_{iid}_ONE", "commit": commit},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "settled", "commit": commit}, path=S.LEDGER_PATH)
    B.close_settled_items(S.LEDGER_PATH)
    if rolled_back:
        S.append_event({"event": "rollback_succeeded", "commit": commit}, path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": iid, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": iid, "phase": "finished",
                    "round_id": f"SM_{iid}_TWO", "stop_reason": "stop", "num_turns": 60},
                   path=S.LEDGER_PATH)
    return f"SM_{iid}_TWO"


def _worktree_stands(tmp_path, monkeypatch):
    """The latest round's worktree on disk, and no `current.json` round.

    The state the reaper takes away with `round.abort`, so a hold keyed on it decays
    instead of pinning an item; `read_current` is pinned so the pass cannot see the
    round this very session opened.
    """
    wt = tmp_path / "wt"
    wt.mkdir(exist_ok=True)
    monkeypatch.setattr(W, "worktree_path", lambda rid: wt)
    monkeypatch.setattr(S, "read_current", lambda: None)
    return wt


def test_the_landed_spent_park_yields_to_a_round_the_shared_predicate_still_counts(
        tmp_path, monkeypatch):
    """Clause 1 — #4616's row, on the machine's side.

    A landed `not_met` item whose one re-offer is spent must not be parked while
    `items_with_unfinished_rounds` counts one of its rounds unfinished: the answer is
    the same `in_progress` the `in_flight` guard at the head of that chain gives, and
    it is one tuple shared between the two positions, not a paraphrase — the reason is
    what the board, owed-check and the steward's scoring all read.
    """
    d = _board(tmp_path, monkeypatch)
    _spent_landed_park(d, 9101, rolled_back=False)      # #4616's own sequence
    _worktree_stands(tmp_path, monkeypatch)             # round two's tree is still open

    assert B.implement_outcomes(S.LEDGER_PATH)[9101][0] == "spent", \
        "the fixture must be the park's own verdict, or the node proves nothing"
    assert 9101 in B.items_with_unfinished_rounds(S.LEDGER_PATH), \
        "a round of it is unfinished on the ONE shared predicate"

    want = B.desired_statuses(S.LEDGER_PATH, None)[9101]
    assert want == ("in_progress", "an automod round is in flight for it"), \
        f"the park fired on a mid-round item: {want}"


def test_a_spent_landed_item_with_no_unfinished_round_still_parks_for_owed_check(
        tmp_path, monkeypatch):
    """Clause 2 — the gate cannot strand a truly spent item in `in_progress`.

    The same fixture with the landing's commit rolled back: nothing is mid-round, so
    the park still fires, still as `draft` + needs-human, and still naming the route
    that grants another attempt. Without this half the new guard would be a way for a
    spent item to sit `in_progress` with nothing running and never reach the board a
    person reads.
    """
    d = _board(tmp_path, monkeypatch)
    _spent_landed_park(d, 9102, rolled_back=True)
    monkeypatch.setattr(S, "read_current", lambda: None)

    assert 9102 not in B.items_with_unfinished_rounds(S.LEDGER_PATH), \
        "nothing is mid-round: the guard must not fire here"

    want = B.desired_statuses(S.LEDGER_PATH, None)[9102]
    assert want[0] == "draft" and want[2] is True, f"a spent item stopped parking: {want}"
    assert "owed-check decides (reopen_item to grant another)" in want[1], want[1]


def test_the_park_guard_is_the_shared_predicates_and_walks_nothing_else(
        tmp_path, monkeypatch):
    """Clause 3 — one measurement, asked once, and only where a park candidate stands.

    Three readings of one stubbed `items_with_unfinished_rounds`:

    * a board with no park candidate asks it NOTHING — the predicate probes a worktree
      and a gate and land marker per item it holds (265 items on the live store on
      2026-11-27), and the reconcile runs every 900 s, so a guard that asked per item,
      or asked eagerly at the head of the pass, would pay that walk board-wide;
    * a board with one park candidate asks it exactly ONCE per pass;
    * with the stub answering EMPTY while the worktree still stands, the park fires
      anyway — so the decision belongs to that predicate and to no liveness rule of
      this function's own, which is the drift the two surfaces were caught in.
    """
    calls: list = []
    stubbed: set = set()

    def counting(ledger, **kw):
        calls.append(ledger)
        return set(stubbed)

    monkeypatch.setattr(B, "items_with_unfinished_rounds", counting)
    monkeypatch.setattr(S, "read_current", lambda: None)

    # 1. No park candidate on the board: no walk at all.
    d = _board(tmp_path, monkeypatch)
    _open_item(d, 9201, status="up_next")
    B.desired_statuses(S.LEDGER_PATH, None)
    assert calls == [], "the pass walked round state for a board with nothing parked"

    # 2/3. One park candidate: asked once, and the answer is the stub's, not a local one.
    stubbed = {9202}
    _spent_landed_park(d, 9202, rolled_back=False)
    _worktree_stands(tmp_path, monkeypatch)
    want = B.desired_statuses(S.LEDGER_PATH, None)[9202]
    assert want == ("in_progress", "an automod round is in flight for it"), want
    assert len(calls) == 1, f"one shared measurement per pass, not one per item: {calls}"

    # The control: the same live worktree, the predicate saying nothing is unfinished.
    stubbed = set()
    want = B.desired_statuses(S.LEDGER_PATH, None)[9202]
    assert want[0] == "draft" and "owed-check decides" in want[1], \
        f"the guard holds its own liveness rule and ignores the predicate: {want}"
