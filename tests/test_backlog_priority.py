"""The human's `priority` orders every pool, and the default is `low`.

Until 2026-09-16 `priority` was on every item and read by nothing: the MCP
writer stamped `medium` on everything the loop filed, the HTTP route stamped
`none`, and the pools ordered on the sweep's `worth`. Alan's rule: the pace
of the loop is tolerable if the tag is honoured, for autotriage and autocode
both. So `priority_key` sorts ahead of `rank_key` in every pool, both writers
default `low` (a value nobody chose must not outrank one somebody did), a
missing or unknown value reads as `low`, and `backfill_priority` writes the
word onto the files that had none.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as BL
from app.routers import backlog as BR
from scripts.automod import backlog as B, cluster as CL, state as S


def write_item(d: Path, item_id, *, status="draft", days_old=10, tags=("backlog",),
               name=None, body="Do the thing.", board="lloyd", priority="medium",
               **fm_extra) -> Path:
    name = name or f"Item {item_id}"
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "created": created, "board": board, "tags": list(tags), **fm_extra}
    if priority is not None:
        fm["priority"] = priority
    p = d / f"{item_id}-{name.lower().replace(' ', '-')[:30]}.md"
    p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
                 encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(CL, "CLUSTERS_PATH", tmp_path / "clusters.json")
    return d


def _ev(**row):
    S.append_event(row, path=S.LEDGER_PATH)


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text())[0]


def _path(d: Path, item_id) -> Path:
    return next(d.glob(f"{item_id}-*.md"))


def _it(priority, **kw) -> B.Item:
    base = dict(path=Path("x"), id=1, name="n", status="draft", created="", body="")
    return B.Item(**{**base, **kw, "priority": priority})


# ── the key and the loader ──────────────────────────────────────────────────

def test_priority_key_orders_high_medium_low_and_reads_anything_else_as_low():
    assert [B.priority_key(_it(p)) for p in ("high", "medium", "low")] == [0, 1, 2]
    for junk in ("none", "", None, "urgent", "HIGH "):
        assert B.priority_key(_it(junk)) == B.priority_key(_it("low")) or junk == "HIGH "
    assert B.priority_key(_it("HIGH ")) == 0, "case and whitespace are not a priority"


def test_load_item_normalises_priority_and_defaults_low(isolated):
    write_item(isolated, 1, priority=None)
    write_item(isolated, 2, priority="none")
    write_item(isolated, 3, priority="High")
    write_item(isolated, 4, priority="whatever")
    assert [B.item_by_id(i).priority for i in (1, 2, 3, 4)] == ["low", "low", "high", "low"]


def test_priority_beats_is_strict_and_false_against_nothing():
    assert B.priority_beats(_it("high"), [_it("medium"), _it("low")])
    assert not B.priority_beats(_it("high"), [_it("high"), _it("low")])
    assert not B.priority_beats(_it("high"), [])


# ── every pool ──────────────────────────────────────────────────────────────

def test_single_triage_takes_priority_over_the_sweeps_rank_then_rank_then_age(isolated):
    write_item(isolated, 1, days_old=300, priority="low", tags=("backlog", B.SWEPT_TAG),
               worth="high", size="small")                       # best rank, low priority
    write_item(isolated, 2, days_old=5, priority="high")          # unranked, high priority
    write_item(isolated, 3, days_old=50, priority="high", tags=("backlog", B.SWEPT_TAG),
               worth="medium", size="large")                     # high priority, ranked
    write_item(isolated, 4, days_old=400, priority="medium")
    write_item(isolated, 5, days_old=200, priority="medium", tags=("backlog", B.SWEPT_TAG),
               worth="high", size="small")
    assert B.select_candidate(S.LEDGER_PATH).id == 2, \
        "high first; within high the newest, whatever the sweep ranked"
    B.record_verdict(B.item_by_id(2), "stale", "x", close=True)
    assert B.select_candidate(S.LEDGER_PATH).id == 3
    B.record_verdict(B.item_by_id(3), "stale", "x", close=True)
    assert B.select_candidate(S.LEDGER_PATH).id == 5, "medium before low whatever the sweep said; " \
                                                      "within medium the sweep's rank, then age"
    B.record_verdict(B.item_by_id(5), "stale", "x", close=True)
    assert B.select_candidate(S.LEDGER_PATH).id == 4


def test_a_high_draft_goes_before_a_live_blocker_which_goes_before_the_rest(isolated):
    write_item(isolated, 1, priority="high", days_old=1)
    write_item(isolated, 2, status="up_next", priority="low")
    _ev(event="backlog_triage", item_id=2, verdict="confirmed", acceptance="x")
    write_item(isolated, 3, priority="low", tags=("backlog", B.BLOCKER_TAG, "spawned-by-autocode"),
               body="Blocks #2. The precondition.")
    write_item(isolated, 4, priority="medium", days_old=500)
    assert B.select_candidate(S.LEDGER_PATH).id == 1
    assert B.select_urgent(S.LEDGER_PATH).id == 1
    B.record_verdict(B.item_by_id(1), "stale", "x", close=True)
    assert B.select_candidate(S.LEDGER_PATH).id == 3, "the blocker, before the old medium"
    assert B.select_urgent(S.LEDGER_PATH) is None, "urgent means high, not merely first"


def test_the_sweep_never_reads_a_high_item(isolated, monkeypatch):
    monkeypatch.setattr(B, "sweep_enabled", lambda: True)
    write_item(isolated, 1, priority="high")
    write_item(isolated, 2, priority="medium")
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [2]
    assert B.sweep_pending(S.LEDGER_PATH) == 1


def test_a_held_item_raised_to_high_is_released_without_room(isolated, monkeypatch):
    write_item(isolated, 1, priority="low", tags=("backlog", B.HELD_TAG))
    write_item(isolated, 2, priority="high", tags=("backlog", B.HELD_TAG))
    for i in (1, 2):
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x", held=True)
    monkeypatch.setattr(B, "implement_pool_full", lambda *a, **k: {"full": True, "ready": 73, "bound": 73,
                                                                   "floor": 20, "landed_items_7d": 73})
    out = B.release_held_confirmations(S.LEDGER_PATH)
    assert [(r["item_id"], r["moved"]) for r in out] == [(2, True)]
    assert "never held" in out[0]["reason"]
    assert _fm(_path(isolated, 2))["status"] == "up_next" and _fm(_path(isolated, 1))["status"] == "draft"


def test_implement_takes_priority_above_the_near_tier_and_the_rank(isolated):
    def confirmed(i, *, clauses, priority, days_old=50, **fm):
        write_item(isolated, i, status="up_next", days_old=days_old, priority=priority,
                   acceptance_clauses=clauses, **fm)
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x",
            acceptance_clauses=clauses)
    confirmed(1, clauses=["a"] * 2, priority="low", tags=("backlog", B.SWEPT_TAG),
              worth="high", size="small")
    confirmed(2, clauses=["a"] * 6, priority="medium")
    confirmed(3, clauses=["a"] * 3, priority="medium", tags=("backlog", B.SWEPT_TAG),
              worth="medium", size="large")
    # #1 is also one fix cycle from landing: a graded review met every clause.
    _ev(event="backlog_implement", item_id=1, round_id="SM_1", status="started")
    _ev(event="backlog_implement", item_id=1, round_id="SM_1", status="finished",
        outcome="review_refused", review_attempt=1)
    _ev(event="review", round_id="SM_1", ok=True, all_met=True, refused=False, head="abc",
        clauses=[{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "met"}])
    ready = B.ready_confirmed(S.LEDGER_PATH)
    assert {p[0].id for p in ready} >= {2, 3}
    picked, _ = B.select_confirmed(S.LEDGER_PATH)
    assert picked.id == 3, "medium before low even when the low is near landing; then rank"
    order = [p[0].id for p in sorted(ready, key=lambda p: (B.priority_key(p[0]), B.rank_key(p[0])))]
    assert order.index(3) < order.index(2) < order.index(1) if 1 in order else order[:2] == [3, 2]


def test_a_high_item_just_submitted_is_the_next_round_whatever_is_already_high(isolated):
    """87 open highs on the day this landed, 17 of them ready: oldest-first
    within the tier would have queued a new high behind every one."""
    def confirmed(i, *, clauses, priority, days_old, **fm):
        write_item(isolated, i, status="up_next", days_old=days_old, priority=priority,
                   acceptance_clauses=clauses, **fm)
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x",
            acceptance_clauses=clauses)
    confirmed(1, clauses=["a"] * 2, priority="high", days_old=120, tags=("backlog", B.SWEPT_TAG),
              worth="high", size="small")                                  # old high, best rank
    confirmed(2, clauses=["a"] * 6, priority="high", days_old=0)           # just submitted, unranked
    confirmed(3, clauses=["a"] * 1, priority="medium", days_old=300, tags=("backlog", B.SWEPT_TAG),
              worth="high", size="small")
    picked, _ = B.select_confirmed(S.LEDGER_PATH)
    assert picked.id == 2
    # A high re-offer that is one fix cycle from landing still goes first:
    # it is the same kind of ask, nearly done.
    _ev(event="backlog_implement", item_id=1, round_id="SM_1", status="started")
    _ev(event="backlog_implement", item_id=1, round_id="SM_1", status="finished",
        outcome="review_refused", review_attempt=1)
    _ev(event="review", round_id="SM_1", ok=True, all_met=True, refused=False, head="abc",
        clauses=[{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "met"}])
    if 1 in {p[0].id for p in B.ready_confirmed(S.LEDGER_PATH)} and B.last_review_all_met(S.LEDGER_PATH):
        assert B.select_confirmed(S.LEDGER_PATH)[0].id == 1
    assert B.recency_key(B.item_by_id(2)) < B.recency_key(B.item_by_id(1))
    assert B.recency_key(B.item_by_id(3)) < B.recency_key(_it("medium", created="2030-01-01T00:00:00+00:00", id=9))


def test_a_re_offered_high_is_not_queued_behind_every_fresh_high(isolated):
    """2026-09-17: #1199, re-offered with its branch after two rounds, sat 27th
    behind 26 never-attempted highs. Within `high`, recency decides."""
    def confirmed(i, *, priority, days_old, **fm):
        write_item(isolated, i, status="up_next", days_old=days_old, priority=priority,
                   acceptance_clauses=["a"], **fm)
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x",
            acceptance_clauses=["a"])
    for i in (1, 2, 3):
        confirmed(i, priority="high", days_old=10 + i)                        # fresh, older
    confirmed(9, priority="high", days_old=1)                                 # newest, re-offered
    _ev(event="backlog_implement", item_id=9, round_id="SM_9", status="started", phase="started")
    _ev(event="backlog_implement", item_id=9, round_id="SM_9", phase="finished",
        stop_reason="max_turns", outcome=None)
    _ev(event="round_abandoned", round_id="SM_9", item_id=9, branch="automod/SM_9")
    out = B.implement_outcomes(S.LEDGER_PATH)
    assert out.get(9), "the re-offer is recorded"
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 9
    # Medium keeps fresh-before-re-offer: the guard against a sent-back item
    # monopolising the loop still holds outside the high tier.
    for i in (1, 2, 3, 9):
        B.update_frontmatter(_path(isolated, i), {"priority": "medium"})
    assert B.select_confirmed(S.LEDGER_PATH)[0].id != 9


def test_held_confirmations_are_released_by_priority_then_rank(isolated, monkeypatch):
    rows = {1: ("low", "high", "small", 40), 2: ("high", "", "", 30), 3: ("medium", "high", "small", 20),
            4: ("high", "low", "large", 50)}
    for i, (pr, w, s, age) in rows.items():
        write_item(isolated, i, days_old=age, priority=pr,
                   tags=("backlog", B.HELD_TAG) + ((B.SWEPT_TAG,) if w else ()),
                   worth=w or None, size=s or None)
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x", held=True)
    monkeypatch.setattr(B, "implement_pool_full", lambda *a, **k: {"full": False, "ready": 0, "bound": 1,
                                                                   "floor": 1, "landed_items_7d": 0})
    out = B.release_held_confirmations(S.LEDGER_PATH)
    assert [r["item_id"] for r in out if r["moved"]] == [2, 4, 3], \
        "both highs first without spending the room (unranked before low-ranked), " \
        "then the one slot goes to the medium; the low waits"
    assert _fm(_path(isolated, 1))["status"] == "draft"


def test_the_sweep_reads_never_judged_first_then_by_priority(isolated, monkeypatch):
    monkeypatch.setattr(B, "sweep_enabled", lambda: True)
    write_item(isolated, 1, days_old=300, priority="low")
    write_item(isolated, 2, days_old=10, priority="medium")
    write_item(isolated, 3, days_old=100, priority="medium")
    _ev(event="backlog_triage", item_id=3, verdict="confirmed", acceptance="x")
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [2, 1, 3]


def test_group_triage_takes_the_cluster_with_the_best_member_and_keeps_it_through_the_trim(isolated):
    for i in (1, 2, 3, 4, 5):
        write_item(isolated, i, days_old=100 - i, priority="low")
    write_item(isolated, 6, days_old=1, priority="high")       # youngest: trimmed off before
    write_item(isolated, 7, days_old=90, priority="low")
    clusters = {"clusters": [
        {"id": "big", "item_ids": [1, 2, 3, 4, 5], "duplicates": []},
        {"id": "small", "item_ids": [6, 7], "duplicates": []},
    ]}
    c, members = B.select_cluster(S.LEDGER_PATH, clusters, min_size=2, max_size=4)
    assert c["id"] == "small" and [m.id for m in members] == [6, 7]
    B.update_frontmatter(_path(isolated, 6), {"priority": "low"})
    c, members = B.select_cluster(S.LEDGER_PATH, clusters, min_size=2, max_size=4)
    assert c["id"] == "big", "equal priority: the largest cluster, as before"
    B.update_frontmatter(_path(isolated, 5), {"priority": "high"})
    B.update_frontmatter(_path(isolated, 7), {"priority": "high"})
    c, members = B.select_cluster(S.LEDGER_PATH, clusters, min_size=2, max_size=4)
    assert c["id"] == "big" and 5 in [m.id for m in members], \
        "a high member survives the trim to max_size although it is the youngest"


@pytest.mark.asyncio
async def test_a_single_that_outranks_the_cluster_runs_before_group_triage(isolated, monkeypatch):
    from workers.sources import autotriage as M
    for i in (1, 2, 3):
        write_item(isolated, i, priority="low")
    write_item(isolated, 4, priority="high")
    CL.CLUSTERS_PATH.write_text(json.dumps({"clusters": [
        {"id": "c1", "item_ids": [1, 2, 3], "duplicates": []}]}))
    from workers.sources import _common as C
    ran = {}

    async def fake_group(item, cluster, members):
        ran["group"] = [m.id for m in members]
        return {"status": "ok", "summary": "group"}

    # The single path is inline in `execute` and ends in a session turn; stub
    # the turn itself so the test never reaches the backend, and read which
    # item the prompt was about.
    async def fake_turn(prompt, **kw):
        ran["single_prompt"] = prompt
        raise RuntimeError("stop here: the choice was made before this call")
    monkeypatch.setattr(M, "_execute_group", fake_group)
    monkeypatch.setattr(C, "run_prompt_in_session", fake_turn)
    monkeypatch.setattr(B, "implement_pool_full",
                        lambda *a, **k: {"full": False, "ready": 0, "bound": 20, "floor": 20,
                                         "landed_items_7d": 0})

    class _Q:
        payload = {"group_triage": True, "sweep": False}
        id = 1
    with pytest.raises(RuntimeError):
        await M.execute(_Q())
    assert "group" not in ran and "#4" in ran["single_prompt"], \
        "the high single outranks a cluster of lows and is triaged first"
    B.update_frontmatter(_path(isolated, 4), {"priority": "low"})
    ran.clear()
    assert (await M.execute(_Q()))["summary"] == "group"
    assert ran.get("group") == [1, 2, 3] and "single_prompt" not in ran, \
        "equal priority: the cluster wins as before"


# ── the writers and the backfill ────────────────────────────────────────────

def test_the_mcp_writer_defaults_low_and_keeps_a_named_priority(isolated, monkeypatch):
    monkeypatch.setattr(BL, "BACKLOG_DIR", isolated)
    out = json.loads(BL._handle_write({"name": "Fresh", "description": "Body.", "board": "lloyd"}))
    assert out["success"], out
    assert _fm(_path(isolated, out["task_id"]))["priority"] == "low"
    out = json.loads(BL._handle_write({"name": "Urgent", "description": "Body.", "board": "lloyd",
                                       "priority": "high", "force": True}))
    assert _fm(_path(isolated, out["task_id"]))["priority"] == "high"
    import asyncio
    tool = next(t for t in asyncio.run(BL.list_tools()) if t.name == "backlog_write_task")
    schema = getattr(tool, "inputSchema", None) or tool.input_schema
    assert schema["properties"]["priority"]["enum"] == ["high", "medium", "low"]


class _FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_the_http_create_route_defaults_low_and_lists_a_missing_priority_as_low(isolated, monkeypatch):
    monkeypatch.setattr(BR, "_BACKLOG_DIR", isolated)
    monkeypatch.setattr(BR, "_backlog_board_map", lambda: {})
    resp = await BR.backlog_task_create(_FakeRequest({"name": "From the UI", "board_id": "lloyd"}))
    assert resp.status_code == 200, resp.body
    assert _fm(next(isolated.glob("*.md")))["priority"] == "low"
    resp = await BR.backlog_task_create(_FakeRequest({"name": "Named", "board_id": "lloyd",
                                                      "priority": "high"}))
    assert resp.status_code == 200
    assert sorted(_fm(p)["priority"] for p in isolated.glob("*.md")) == ["high", "low"]


def test_backfill_writes_low_onto_absent_none_and_unknown_and_leaves_the_rest(isolated):
    write_item(isolated, 1, priority=None)
    write_item(isolated, 2, priority="none", status="done")
    write_item(isolated, 3, priority="high")
    write_item(isolated, 4, priority="medium", board="alfie")
    write_item(isolated, 5, priority="urgent")
    p6 = isolated / "6-broken.md"
    p6.write_text("---\nstatus: draft\ntags: [a\n---\n\n# Broken\n")
    dry = B.backfill_priority(None, dry_run=True)
    assert sorted(r["item_id"] for r in dry) == [1, 2, 5, 6] and not any(r["written"] for r in dry)
    assert _fm(_path(isolated, 1)).get("priority") is None, "a dry run writes nothing"
    rows = B.backfill_priority(None)
    by = {r["item_id"]: r for r in rows}
    assert {i for i, r in by.items() if r["written"]} == {1, 2, 5}
    assert by[6]["written"] is False and by[6]["skipped"]
    assert p6.read_text().startswith("---\nstatus: draft\ntags: [a\n"), "broken YAML is never rewritten"
    for i in (1, 2, 5):
        fm = _fm(_path(isolated, i))
        assert fm["priority"] == "low" and any("priority set to low" in a for a in fm["activity_log"])
    assert _fm(_path(isolated, 3))["priority"] == "high" and _fm(_path(isolated, 4))["priority"] == "medium"
    assert B.backfill_priority(None) == [{"item_id": 6, "status": "draft", "was": None, "now": "low",
                                         "written": False, "skipped": "frontmatter did not parse"}], \
        "idempotent: a second pass finds only the file it could not read"


def test_reset_open_writes_low_onto_open_items_only_and_says_what_it_overwrote(isolated):
    write_item(isolated, 1, priority="high")
    write_item(isolated, 2, priority="high", status="done")
    write_item(isolated, 3, priority="medium", status="up_next", board="alfie")
    write_item(isolated, 4, priority="low")
    dry = B.backfill_priority(None, dry_run=True, reset_open=True)
    assert sorted(r["item_id"] for r in dry) == [1, 3] and _fm(_path(isolated, 1))["priority"] == "high"
    rows = B.backfill_priority(None, reset_open=True)
    assert {r["item_id"]: r["was"] for r in rows if r["written"]} == {1: "high", 3: "medium"}
    assert _fm(_path(isolated, 1))["priority"] == "low" and _fm(_path(isolated, 2))["priority"] == "high"
    assert any("reset from 'high' by hand" in a for a in _fm(_path(isolated, 1))["activity_log"])
    assert B.backfill_priority(None, reset_open=True) == []
