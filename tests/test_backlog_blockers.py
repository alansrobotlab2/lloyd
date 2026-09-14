"""A blocker has to reach someone before it expires.

An implement round may file exactly one item of its own: a *blocker*, a
finding that stops one of its clauses from becoming true, tagged
`spawned-by-autocode` + `blocker`, first line "Blocks #N". The round defers
the clause to it. Until 2026-09-14 nothing but write-time dedupe read the tag,
so a blocker was an ordinary self-spawn: quarantined, so no triage gave it a
contract and it never reached `up_next`; then expired at seven days, orphaning
the clause that waited on it. That day 13 of the 101 quarantined drafts were
blockers, four of them in front of items already in the implement pool.

A blocker is *live* while the item it blocks is open. Live: triaged first,
never held by the depth gate, taken early by autocode, never expired. Once the
blocked item closes it is an ordinary self-spawn again — quarantined, bound by
expiry — not closed outright, because the finding can be real on its own.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, state as S
from workers.sources import _common as C
from workers.sources import autotriage as M


def write_item(d: Path, item_id, *, status="draft", days_old=100, tags=("backlog",),
               name="A thing", body="Do the thing.", board="lloyd") -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": board, "tags": list(tags)}
    path = d / f"{item_id}-{name.lower().replace(' ', '-').replace('#', '')}.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
                    encoding="utf-8")
    return path


def blocker(d: Path, item_id, blocks: int | None, *, days_old=0, titled=True, **kw) -> Path:
    """What autocode's prompt asks a round to file."""
    if blocks is None:
        name, body = "An obstacle", "Something is in the way."
    elif titled:
        name, body = f"Blocks #{blocks}: an obstacle", f"Blocks #{blocks}\n\nThe handoff."
    else:
        name, body = "An obstacle", f"Blocks #{blocks} (clause 2).\n\nThe handoff."
    return write_item(d, item_id, days_old=days_old, name=name, body=body,
                      tags=("backlog", "spawned-by-autocode", B.BLOCKER_TAG), **kw)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def _ev(**row):
    S.append_event(row, path=S.LEDGER_PATH)


class _QItem:
    def __init__(self, payload=None):
        self.payload = payload or {}


def _ready(d, n, start=1000):
    for i in range(start, start + n):
        write_item(d, i, status="up_next", name=f"ready {i}")
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="it passes")


# ===========================================================================
# Which item a blocker blocks
# ===========================================================================

def test_the_target_comes_from_the_title_the_first_line_or_the_round_that_filed_it(isolated):
    blocker(isolated, 500, 42)
    blocker(isolated, 501, 43, titled=False)
    blocker(isolated, 502, None)
    _ev(event="backlog_implement", item_id=44, phase="finished", spawned=[502])
    targets = B.blocker_targets(S.LEDGER_PATH)
    assert B.blocked_item_of(B.item_by_id(500), targets) == 42
    assert B.blocked_item_of(B.item_by_id(501), targets) == 43
    assert B.blocked_item_of(B.item_by_id(502), targets) == 44
    assert B.blocked_item_of(B.item_by_id(502)) is None


def test_a_blocker_is_live_only_while_its_target_is_open(isolated):
    write_item(isolated, 42, status="up_next")
    write_item(isolated, 43, status="done")
    blocker(isolated, 500, 42)
    blocker(isolated, 501, 43)
    blocker(isolated, 502, None)          # names nothing: triage finds out
    blocker(isolated, 503, 9999)          # names an item not on disk: likewise
    write_item(isolated, 504, name="Blocks #42: untagged", tags=("backlog", "spawned-by-autocode"))
    assert B.live_blockers(S.LEDGER_PATH) == {500: 42, 502: None, 503: 9999}


def test_a_full_board_read_answers_the_same_as_lookups_by_id(isolated):
    write_item(isolated, 42, status="up_next")
    write_item(isolated, 43, status="done")
    blocker(isolated, 500, 42)
    blocker(isolated, 501, 43)
    everything = B.all_items()
    assert (B.live_blockers(S.LEDGER_PATH, everything=everything)
            == B.live_blockers(S.LEDGER_PATH) == {500: 42})


# ===========================================================================
# Triage reaches a live blocker, first
# ===========================================================================

def test_a_live_blocker_is_a_triage_candidate_and_goes_ahead_of_the_oldest(isolated):
    write_item(isolated, 42, status="up_next")
    write_item(isolated, 10, days_old=300)                 # the oldest real item
    blocker(isolated, 500, 42, days_old=0)
    candidates, held = B.triage_pool(S.LEDGER_PATH)
    assert sorted(i.id for i in candidates) == [10, 500] and held == 0
    assert B.select_candidate(S.LEDGER_PATH).id == 500


def test_a_blocker_of_a_closed_item_is_quarantined_like_any_self_spawn(isolated):
    write_item(isolated, 42, status="done")
    blocker(isolated, 500, 42)
    assert B.triage_pool(S.LEDGER_PATH) == ([], 1)
    assert B.select_candidate(S.LEDGER_PATH) is None


def test_without_the_blocker_rule_the_same_blocker_is_held_back(isolated, monkeypatch):
    """The counterfactual, so the tests above cannot pass for a lazy reason."""
    write_item(isolated, 42, status="up_next")
    blocker(isolated, 500, 42)
    monkeypatch.setattr(B, "BLOCKER_TAG", "not-a-tag-anyone-writes")
    assert B.triage_pool(S.LEDGER_PATH) == ([], 1)


def test_the_origin_block_tells_triage_what_the_blocker_holds_up(isolated):
    write_item(isolated, 42, status="up_next")
    blocker(isolated, 500, 42)
    text = M.render_prompt(B.item_by_id(500), ledger=S.LEDGER_PATH)
    origin = text[text.index("<origin"):text.index("</origin>")]
    assert "a BLOCKER of #42 (up_next)" in origin
    blocker(isolated, 501, None)
    text = M.render_prompt(B.item_by_id(501), ledger=S.LEDGER_PATH)
    assert "a BLOCKER, naming no item it blocks" in text


def test_an_ordinary_item_carries_no_blocker_line(isolated):
    write_item(isolated, 7)
    assert "BLOCKER" not in M.render_prompt(B.item_by_id(7), ledger=S.LEDGER_PATH)


# ===========================================================================
# Expiry
# ===========================================================================

def test_a_live_blocker_never_expires_and_expires_once_its_target_closes(isolated):
    target = write_item(isolated, 42, status="up_next")
    blocker(isolated, 500, 42, days_old=B.spawn_expiry_days() + 3)
    assert B.expire_stale_spawns(S.LEDGER_PATH) == []
    assert B.item_by_id(500).status == "draft"
    target.write_text(target.read_text().replace("status: up_next", "status: done", 1))
    assert [r["item_id"] for r in B.expire_stale_spawns(S.LEDGER_PATH)] == [500]
    assert B.item_by_id(500).status == "done"


def test_the_over_bound_gauge_reads_the_same_rule_as_expiry(isolated):
    """"`over_bound` should read 0" — so an item expiry deliberately skips must
    not count, or the gauge reports a sweep failure that is the sweep working."""
    from scripts.automod import scorecard as SC
    old = B.spawn_expiry_days() + 3
    target = write_item(isolated, 42, status="up_next")
    blocker(isolated, 500, 42, days_old=old)
    blocker(isolated, 501, None, days_old=old)            # unnamed target: live
    write_item(isolated, 502, days_old=old, name="Blocks #42 — ordinary",
               tags=("backlog", "spawned-by-autocode"))   # not tagged: an ordinary spawn
    now = datetime.now(timezone.utc).timestamp()
    events = S.read_events(path=S.LEDGER_PATH) if S.LEDGER_PATH.exists() else []
    assert SC._self_spawned_gauge(events, isolated, now=now)["over_bound"] == 1
    assert [r["item_id"] for r in B.expire_stale_spawns(S.LEDGER_PATH)] == [502]
    target.write_text(target.read_text().replace("status: up_next", "status: done", 1))
    assert SC._self_spawned_gauge(events, isolated, now=now)["over_bound"] == 1   # 500 now; 502 closed
    assert [r["item_id"] for r in B.expire_stale_spawns(S.LEDGER_PATH)] == [500]


def test_the_gauge_resolves_a_target_from_the_ledger_rows_it_was_handed(isolated):
    from scripts.automod import scorecard as SC
    old = B.spawn_expiry_days() + 3
    write_item(isolated, 44, status="done")
    blocker(isolated, 502, None, days_old=old)
    events = [{"event": "backlog_implement", "item_id": 44, "phase": "finished", "spawned": [502]}]
    now = datetime.now(timezone.utc).timestamp()
    assert SC._self_spawned_gauge(events, isolated, now=now)["over_bound"] == 1, \
        "its target is closed, so it is an ordinary over-bound spawn"
    assert SC._self_spawned_gauge([], isolated, now=now)["over_bound"] == 0, \
        "with no row naming a target it counts as live"


# ===========================================================================
# The depth gate does not hold a live blocker
# ===========================================================================

def _stub_confirmed(monkeypatch):
    async def turn(prompt, **kw):
        turn.calls.append(prompt)
        return {"text": "VERDICT: confirmed\nSURFACE: code\nCHECK: grep -n x app.py\nEVIDENCE: e\n"
                        "ACCEPTANCE: it passes\nSPAWNED: none\n",
                "session_id": "s", "stop_reason": "stop", "num_turns": 3, "errors": []}
    turn.calls = []
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    return turn


def test_a_confirmed_live_blocker_enters_a_full_pool(isolated, monkeypatch):
    _ready(isolated, 40)
    blocker(isolated, 500, 1000)          # blocks an item already in the pool
    write_item(isolated, 7, days_old=300)
    turn = _stub_confirmed(monkeypatch)
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["item_id"] == 500 and out["verdict"] == "confirmed" and out["held"] is False
    assert len(turn.calls) == 1 and "a BLOCKER of #1000 (up_next)" in turn.calls[0]
    item = B.item_by_id(500)
    assert item.status == "up_next" and B.HELD_TAG not in item.tags
    assert B.held_confirmations(S.LEDGER_PATH) == {}


def test_the_same_verdict_on_an_ordinary_item_is_still_held(isolated, monkeypatch):
    """The gate itself is untouched."""
    _ready(isolated, 40)
    write_item(isolated, 7, days_old=300)
    _stub_confirmed(monkeypatch)
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["item_id"] == 7 and out["held"] is True


def test_a_blocker_held_before_this_rule_is_released_without_room(isolated):
    _ready(isolated, 5)
    write_item(isolated, 42, status="up_next")
    for iid, path in ((500, blocker(isolated, 500, 42)), (7, write_item(isolated, 7))):
        B.record_verdict(B.load_item(path), "confirmed", "real", acceptance="it passes", hold=True)
        _ev(event="backlog_triage", item_id=iid, verdict="confirmed", acceptance="it passes", held=True)
    out = B.release_held_confirmations(S.LEDGER_PATH, floor=3)       # full: 6 ready >= 3
    assert [(r["item_id"], r["moved"]) for r in out] == [(500, True)]
    assert "live blocker of #42" in out[0]["reason"]
    assert B.item_by_id(500).status == "up_next"
    assert set(B.held_confirmations(S.LEDGER_PATH)) == {7}, "an ordinary item still waits"


# ===========================================================================
# Autocode takes a live blocker early
# ===========================================================================

def test_select_confirmed_takes_a_live_blocker_before_an_older_confirmation(isolated):
    _ready(isolated, 1, start=1000)                      # older ordinary confirmation
    write_item(isolated, 42, status="draft")             # the blocked item, still open
    blocker(isolated, 500, 42, status="up_next", days_old=0)
    _ev(event="backlog_triage", item_id=500, verdict="confirmed", acceptance="it passes")
    item, _ev_row = B.select_confirmed(S.LEDGER_PATH)
    assert item.id == 500


def test_a_blocker_of_a_closed_item_takes_its_ordinary_place(isolated):
    _ready(isolated, 1, start=1000)
    write_item(isolated, 42, status="done")
    blocker(isolated, 500, 42, status="up_next", days_old=0)
    _ev(event="backlog_triage", item_id=500, verdict="confirmed", acceptance="it passes")
    item, _ev_row = B.select_confirmed(S.LEDGER_PATH)
    assert item.id == 1000


# ===========================================================================
# The board says how many there are
# ===========================================================================

def test_board_health_counts_live_blockers_outside_the_quarantined_bucket(isolated):
    write_item(isolated, 42, status="up_next")
    write_item(isolated, 43, status="done")
    blocker(isolated, 500, 42)
    blocker(isolated, 501, 43)
    blocker(isolated, 502, 42)
    _ev(event="backlog_triage", item_id=502, verdict="unverifiable")
    h = B.board_health(S.LEDGER_PATH)
    assert h["live_blockers"] == {"open": 2, "untriaged": 1}
    assert h["draft"]["pool"] == 1 and h["draft"]["quarantined"] == 1 and h["draft"]["triaged"] == 1
