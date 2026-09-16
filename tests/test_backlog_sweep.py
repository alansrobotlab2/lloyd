"""The sweep: every open item read once, retired or ranked, nothing filed.

On 2026-09-15 the board held 560 open items. Single triage confirmed 73% of
what it read and the implement loop landed ~10 a day, so 156 confirmed items
were queued, 82 quarantined drafts had never been read by anything (27 due to
expire unread within a day), and 56 umbrellas of 8-12 clauses — a shape that
lands one time in five — held 158 members out of every pool. The sweep is
the gear change: a batch per turn, quarantine lifted, each item either
retired or ranked (`worth` × `size`), with `parked` replacing expiry as the
exit for an item nobody asked for. The rank then orders every pool.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, cluster as CL, state as S
from workers.sources import _common as C
from workers.sources import autocode as AC
from workers.sources import autotriage as M


def write_item(d: Path, item_id, *, status="draft", days_old=10, tags=("backlog",),
               name=None, body="Do the thing.", board="lloyd", **fm_extra) -> Path:
    name = name or f"Item {item_id}"
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": board, "tags": list(tags), **fm_extra}
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
    monkeypatch.setattr(B, "sweep_enabled", lambda: True)
    return d


def _ev(**row):
    S.append_event(row, path=S.LEDGER_PATH)


def _fm(path: Path) -> dict:
    return B._split_frontmatter(path.read_text())[0]


def _path(d: Path, item_id) -> Path:
    return next(d.glob(f"{item_id}-*.md"))


def _item(item_id) -> B.Item:
    return B.item_by_id(item_id)


class _Q:
    def __init__(self, payload=None):
        self.payload = payload or {}


# ── the pool ────────────────────────────────────────────────────────────────

def test_the_pool_reads_quarantined_held_and_confirmed_items_and_skips_the_judged_shapes(isolated):
    write_item(isolated, 1)                                                  # plain draft
    write_item(isolated, 2, tags=("backlog", "spawned-by-triage"))           # quarantined: read anyway
    write_item(isolated, 3, tags=("backlog", "youtube-eval"))                # quarantined: read anyway
    write_item(isolated, 4, status="up_next")                                # confirmed: ranked too
    _ev(event="backlog_triage", item_id=4, verdict="confirmed", acceptance="x")
    write_item(isolated, 5, tags=("backlog", B.HELD_TAG))                    # held: ranked too
    _ev(event="backlog_triage", item_id=5, verdict="confirmed", acceptance="x", held=True)
    write_item(isolated, 6, tags=("backlog", "grouped"), group=99)           # member: its umbrella is the unit
    write_item(isolated, 7, status="up_next", tags=("backlog", "umbrella"), members=[6])
    write_item(isolated, 8, tags=("backlog", B.NEEDS_HUMAN_TAG))
    write_item(isolated, 9, tags=("backlog", B.SWEPT_TAG), worth="high", size="small")
    write_item(isolated, 10, tags=("backlog", B.SWEPT_TAG, B.PARKED_TAG), worth="low")
    write_item(isolated, 11, status="in_progress")
    write_item(isolated, 12, status="done")
    _ev(event="backlog_sweep", batch_id="sw-x", item_ids=[13], verdict="abandoned", judged={})
    write_item(isolated, 13)
    ids = {i.id for i in B.sweep_pool(S.LEDGER_PATH)}
    assert ids == {1, 2, 3, 4, 5}


def test_a_batch_takes_never_judged_items_first_then_oldest(isolated):
    write_item(isolated, 1, days_old=50, status="up_next")     # confirmed: judged once already
    _ev(event="backlog_triage", item_id=1, verdict="confirmed", acceptance="x")
    write_item(isolated, 2, days_old=5)
    write_item(isolated, 3, days_old=30, tags=("backlog", "spawned-by-triage"))
    write_item(isolated, 4, days_old=20)
    batch = B.select_sweep_batch(S.LEDGER_PATH, 3)
    assert [i.id for i in batch] == [3, 4, 2]
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [3, 4, 2, 1]
    assert B.sweep_batch_id([4, 3, 2]) == B.sweep_batch_id([2, 3, 4])


def test_pending_is_zero_when_the_sweep_is_off(isolated, monkeypatch):
    write_item(isolated, 1)
    assert B.sweep_pending(S.LEDGER_PATH) == 1
    monkeypatch.setattr(B, "sweep_enabled", lambda: False)
    assert B.sweep_pending(S.LEDGER_PATH) == 0


# ── parsing ─────────────────────────────────────────────────────────────────

def test_parse_prefers_the_structured_object_and_validates_it():
    structured = {"items": [
        {"item_id": 1, "verdict": "keep", "duplicate_of": 0, "worth": "high", "size": "small", "evidence": "e1"},
        {"item_id": 2, "verdict": "duplicate_of", "duplicate_of": 1, "worth": "low", "size": "large", "evidence": "e2"},
        {"item_id": 3, "verdict": "fold", "duplicate_of": 0, "worth": "high", "size": "small", "evidence": "no such verdict"},
        {"item_id": 4, "verdict": "keep", "duplicate_of": 0, "worth": "HUGE", "size": "small", "evidence": "bad level"},
        {"item_id": 99, "verdict": "keep", "duplicate_of": 0, "worth": "high", "size": "small", "evidence": "not a member"},
    ]}
    out = M.parse_sweep_verdict("SWEEP_VERDICTS:\n#1: stale worth=low size=small — ignored", structured, [1, 2, 3, 4, 5])
    assert out["source"] == "structured"
    assert out["items"][1] == {"verdict": "keep", "duplicate_of": 0, "worth": "high", "size": "small", "evidence": "e1"}
    assert out["items"][2]["verdict"] == "duplicate_of" and out["items"][2]["duplicate_of"] == 1
    assert out["items"][4]["worth"] == "", "an unknown level is dropped, not guessed"
    assert 3 not in out["items"] and 99 not in out["items"]
    assert out["unjudged"] == [3, 5], "an unlisted or unparseable member stays unswept"


def test_parse_falls_back_to_the_block_and_reads_worth_and_size_in_any_order():
    text = """thinking...

SWEEP_VERDICTS:
#1: keep worth=high size=small — the check at loop.py:40 still fails
#2 -> already_done size: medium worth: low — fixed by abc123
- #3: duplicate_of #1 worth=medium size=medium — same finding as #1
#4: keep — no labels on this one
#5: fold worth=high size=small — not a sweep verdict
"""
    out = M.parse_sweep_verdict(text, None, [1, 2, 3, 4, 5])
    assert out["source"] == "regex"
    assert out["items"][1] == {"verdict": "keep", "duplicate_of": 0, "worth": "high", "size": "small",
                               "evidence": "the check at loop.py:40 still fails"}
    assert out["items"][2]["verdict"] == "already_done" and out["items"][2]["worth"] == "low" \
        and out["items"][2]["size"] == "medium" and out["items"][2]["evidence"] == "fixed by abc123"
    assert out["items"][3]["verdict"] == "duplicate_of" and out["items"][3]["duplicate_of"] == 1
    assert out["items"][4] == {"verdict": "keep", "duplicate_of": 0, "worth": "", "size": "",
                               "evidence": "no labels on this one"}
    assert out["unjudged"] == [5]
    assert M.parse_sweep_verdict("no block here", None, [1]) is None


# ── recording ───────────────────────────────────────────────────────────────

def test_record_retires_closes_duplicates_ranks_and_parks(isolated):
    write_item(isolated, 1)
    write_item(isolated, 2)
    write_item(isolated, 3)
    write_item(isolated, 4)
    write_item(isolated, 5, status="up_next")
    _ev(event="backlog_triage", item_id=5, verdict="confirmed", acceptance="x")
    write_item(isolated, 6)
    write_item(isolated, 7)
    write_item(isolated, 20)                       # open, outside the batch
    write_item(isolated, 21, status="done")        # closed, outside the batch
    members = [_item(i) for i in (1, 2, 3, 4, 5, 6, 7)]
    verdicts = {
        1: {"verdict": "stale", "worth": "low", "size": "small", "evidence": "gone"},
        2: {"verdict": "duplicate_of", "duplicate_of": 3, "worth": "medium", "size": "small", "evidence": "same as 3"},
        3: {"verdict": "keep", "worth": "high", "size": "small", "evidence": "real"},
        4: {"verdict": "keep", "worth": "low", "size": "large", "evidence": "someday"},
        5: {"verdict": "keep", "worth": "low", "size": "large", "evidence": "confirmed but low"},
        6: {"verdict": "duplicate_of", "duplicate_of": 20, "worth": "low", "size": "small", "evidence": "same as 20"},
        7: {"verdict": "duplicate_of", "duplicate_of": 21, "worth": "medium", "size": "medium", "evidence": "same as a closed item"},
    }
    res = B.record_sweep_verdicts("sw-t", members, verdicts, session_id="sess")
    assert res["retired"] == 1 and res["duplicates"] == 2 and res["kept"] == 4 and res["parked"] == 1

    assert _fm(_path(isolated, 1))["status"] == "done"
    assert _fm(_path(isolated, 2))["status"] == "done" and _fm(_path(isolated, 2))["duplicate_of"] == 3
    fm3 = _fm(_path(isolated, 3))
    assert fm3["status"] == "draft" and fm3["worth"] == "high" and fm3["size"] == "small"
    assert B.SWEPT_TAG in fm3["tags"] and B.PARKED_TAG not in fm3["tags"]
    fm4 = _fm(_path(isolated, 4))
    assert fm4["worth"] == "low" and B.PARKED_TAG in fm4["tags"] and B.SWEPT_TAG in fm4["tags"]
    assert "parked" in fm4["activity_log"][-1]
    fm5 = _fm(_path(isolated, 5))
    assert fm5["status"] == "up_next" and fm5["worth"] == "low" and B.PARKED_TAG not in fm5["tags"], \
        "a confirmed item is ranked, never parked: the implement order sorts it last instead"
    assert _fm(_path(isolated, 6))["status"] == "done" and _fm(_path(isolated, 6))["duplicate_of"] == 20, \
        "a duplicate may point at any open item on the board"
    fm7 = _fm(_path(isolated, 7))
    assert fm7["status"] == "draft" and fm7["worth"] == "medium" and B.SWEPT_TAG in fm7["tags"], \
        "a duplicate of a closed item is a keep"

    rows = S.read_events(path=S.LEDGER_PATH)
    closes = [r for r in rows if r["event"] == "backlog_triage" and r.get("sweep_batch") == "sw-t"]
    assert {r["item_id"]: r["verdict"] for r in closes} == {1: "stale", 2: "stale", 6: "stale"}
    assert all(r["closed"] for r in closes)
    summary = [r for r in rows if r["event"] == "backlog_sweep"][-1]
    assert summary["judged"] == {"1": "stale", "2": "duplicate_of", "3": "keep", "4": "keep",
                                 "5": "keep", "6": "duplicate_of", "7": "keep"}
    assert summary["ranked"] == {"3": "high/small", "4": "low/large", "5": "low/large", "7": "medium/medium"}
    assert B.swept_ids(S.LEDGER_PATH) == {1, 2, 3, 4, 5, 6, 7}
    assert B.sweep_pool(S.LEDGER_PATH) == [] or {i.id for i in B.sweep_pool(S.LEDGER_PATH)} == {20}
    # The reconciler agrees with the file: nothing moves a swept draft.
    desired = B.desired_statuses(S.LEDGER_PATH)
    assert desired.get(3) is None and desired.get(5, ("up_next",))[0] == "up_next"


def test_a_swept_self_spawn_is_released_a_parked_one_is_out_of_every_pool_and_neither_expires(isolated):
    write_item(isolated, 1, days_old=40, tags=("backlog", "spawned-by-triage"))
    write_item(isolated, 2, days_old=40, tags=("backlog", "spawned-by-triage"))
    write_item(isolated, 3, days_old=40, tags=("backlog", "spawned-by-triage"))   # never swept
    members = [_item(1), _item(2)]
    B.record_sweep_verdicts("sw-r", members, {
        1: {"verdict": "keep", "worth": "medium", "size": "small", "evidence": "e"},
        2: {"verdict": "keep", "worth": "low", "size": "small", "evidence": "e"}}, session_id="s")
    fresh, held = B.triage_pool(S.LEDGER_PATH)
    assert [i.id for i in fresh] == [1], "swept: released from quarantine; parked: out of the pool"
    assert held == 1, "the unswept self-spawn is still quarantined"
    assert B.select_candidate(S.LEDGER_PATH).id == 1
    assert B.select_cluster(S.LEDGER_PATH, {"clusters": [
        {"id": "c-1", "item_ids": [1, 2, 3], "duplicates": []}]}, min_size=2)[1] == [_item(1), _item(3)] \
        or [i.id for i in B.select_cluster(S.LEDGER_PATH, {"clusters": [
            {"id": "c-1", "item_ids": [1, 2, 3], "duplicates": []}]}, min_size=2)[1]] == [1, 3]
    expired = B.expire_stale_spawns(S.LEDGER_PATH, max_age_days=7)
    assert [e["item_id"] for e in expired] == [3], "read and ranked is the exit; only the unread item expires"
    assert _fm(_path(isolated, 2))["status"] == "draft"


# ── the rank orders every pool ──────────────────────────────────────────────

def test_rank_key_orders_high_small_first_and_unranked_between_medium_and_low():
    def it(w, s):
        return B.Item(path=Path("x"), id=1, name="n", status="draft", priority="m", created="",
                      body="", worth=w, size=s)
    order = sorted([it("low", "small"), it("", ""), it("medium", "large"), it("high", "large"),
                    it("high", "small"), it("medium", "small")], key=B.rank_key)
    assert [(i.worth, i.size) for i in order] == [("high", "small"), ("high", "large"),
                                                  ("medium", "small"), ("medium", "large"),
                                                  ("", ""), ("low", "small")]


def test_single_triage_takes_the_best_ranked_candidate_and_age_breaks_ties(isolated):
    write_item(isolated, 1, days_old=300)                                          # unranked, oldest
    write_item(isolated, 2, days_old=10, tags=("backlog", B.SWEPT_TAG), worth="medium", size="large")
    write_item(isolated, 3, days_old=5, tags=("backlog", B.SWEPT_TAG), worth="high", size="large")
    write_item(isolated, 4, days_old=20, tags=("backlog", B.SWEPT_TAG), worth="high", size="large")
    assert B.select_candidate(S.LEDGER_PATH).id == 4, "high before medium before unranked; older high first"


def test_implement_takes_the_best_ranked_shortest_contract_after_the_near_tier(isolated):
    def confirmed(i, *, clauses, days_old=50, **fm):
        write_item(isolated, i, status="up_next", days_old=days_old, acceptance_clauses=clauses, **fm)
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x",
            acceptance_clauses=clauses)
    confirmed(1, clauses=["a"] * 10, days_old=300)                                        # unranked umbrella-shaped
    confirmed(2, clauses=["a"] * 6, tags=("backlog", B.SWEPT_TAG), worth="high", size="medium")
    confirmed(3, clauses=["a"] * 3, tags=("backlog", B.SWEPT_TAG), worth="high", size="medium")
    confirmed(4, clauses=["a"] * 2, tags=("backlog", B.SWEPT_TAG), worth="low", size="small")
    picked, ev = B.select_confirmed(S.LEDGER_PATH)
    assert picked.id == 3, "same rank: the shorter contract; low sorts last whatever its size"
    assert picked.clause_count == 3
    # Rank beats age: the oldest item is the 10-clause unranked one and it
    # sorts behind every ranked high, ahead of the low.
    order = sorted(B.ready_confirmed(S.LEDGER_PATH),
                   key=lambda p: (B.rank_key(p[0]), p[0].clause_count))
    assert [p[0].id for p in order] == [3, 2, 1, 4]


def test_held_confirmations_are_released_best_first(isolated, monkeypatch):
    for i, (w, s, age) in {1: ("low", "small", 40), 2: ("high", "large", 30), 3: ("high", "small", 20),
                           4: ("", "", 50)}.items():
        write_item(isolated, i, days_old=age, tags=("backlog", B.HELD_TAG) + ((B.SWEPT_TAG,) if w else ()),
                   worth=w or None, size=s or None)
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x", held=True)
    monkeypatch.setattr(B, "implement_pool_full", lambda *a, **k: {"full": False, "ready": 0, "bound": 2,
                                                                   "floor": 2, "landed_items_7d": 0})
    out = B.release_held_confirmations(S.LEDGER_PATH)
    assert [r["item_id"] for r in out if r["moved"]] == [3, 2]
    assert _fm(_path(isolated, 3))["status"] == "up_next" and _fm(_path(isolated, 1))["status"] == "draft"


# ── the source ──────────────────────────────────────────────────────────────

def _sweep_turn(monkeypatch, text, *, structured=None, stop_reason="stop"):
    async def fake(prompt, **kw):
        fake.calls.append({"prompt": prompt, **kw})
        return {"text": text, "session_id": "sess_sweep", "stop_reason": stop_reason,
                "num_turns": 12, "errors": [], "structured": structured}
    fake.calls = []
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    return fake


def test_execute_sweeps_first_read_only_and_records_the_batch(isolated, monkeypatch):
    for i in (1, 2, 3):
        write_item(isolated, i, tags=("backlog", "spawned-by-triage"), body=f"Claim {i}.")
    turn = _sweep_turn(monkeypatch, "SWEEP_VERDICTS:\n#1: stale worth=low size=small — gone\n"
                                    "#2: keep worth=high size=small — real\n"
                                    "#3: keep worth=low size=large — someday\n")
    out = asyncio.run(M.execute(_Q({"sweep": True, "sweep_batch": 8, "group_triage": True})))
    assert out["status"] == "success" and out["batch_id"].startswith("sw-")
    assert out["retired"] == 1 and out["kept"] == 2 and out["parked"] == 1
    call = turn.calls[0]
    assert "SWEEPING 3 items" in call["prompt"] and '<item id="1"' in call["prompt"]
    assert call["max_turns"] == M.DEFAULT_SWEEP_MAX_TURNS
    for tool in ("backlog_write_task", "Edit", "Write", "automod_start", "vault_write"):
        assert tool in call["extra_disallowed"]
    assert call["final_schema"] is B.SWEEP_SCHEMA
    assert _fm(_path(isolated, 1))["status"] == "done"
    assert _fm(_path(isolated, 2))["worth"] == "high"
    assert B.PARKED_TAG in _fm(_path(isolated, 3))["tags"]
    # Everything read: the next run falls through to the ordinary passes.
    assert B.sweep_pending(S.LEDGER_PATH) == 0
    turn2 = _sweep_turn(monkeypatch, "VERDICT: unverifiable\nSURFACE: code\nCHECK: x\nEVIDENCE: e\n"
                                     "ACCEPTANCE: none\nSPAWNED: none\n")
    out2 = asyncio.run(M.execute(_Q({"sweep": True, "group_triage": False})))
    assert out2["status"] == "success" and out2["item_id"] == 2, \
        "the ranked high item is single-triaged next; the parked one never is"
    assert "SWEEPING" not in turn2.calls[0]["prompt"]


def test_a_batch_that_reaches_no_verdict_is_offered_once_more_then_abandoned(isolated, monkeypatch):
    for i in (1, 2):
        write_item(isolated, i)
    _sweep_turn(monkeypatch, "ran out of room", stop_reason="max_turns")
    out = asyncio.run(M.execute(_Q({"sweep": True})))
    assert out["status"] == "skipped" and "incomplete" in out["summary"]
    assert _fm(_path(isolated, 1)).get("worth") is None, "nothing is written on the items"
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [1, 2], "still unswept: offered again"
    out = asyncio.run(M.execute(_Q({"sweep": True})))
    assert out["status"] == "success" and "abandoned" in out["summary"]
    assert B.sweep_pool(S.LEDGER_PATH) == [], "abandoned: left to the ordinary passes"
    fresh, _ = B.triage_pool(S.LEDGER_PATH)
    assert {i.id for i in fresh} == {1, 2}


def test_an_unjudged_member_stays_unswept(isolated, monkeypatch):
    for i in (1, 2):
        write_item(isolated, i)
    _sweep_turn(monkeypatch, "SWEEP_VERDICTS:\n#1: keep worth=medium size=small — fine\n")
    out = asyncio.run(M.execute(_Q({"sweep": True})))
    assert out["kept"] == 1
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [2]


def test_with_umbrellas_off_a_group_triage_folds_nothing(isolated, monkeypatch):
    for i in (2, 5):
        write_item(isolated, i, tags=("backlog", "spawned-by-triage"))
    members = [_item(2), _item(5)]
    cluster = {"id": CL.cluster_id([2, 5]), "item_ids": [2, 5], "reason": "paths", "anchor_paths": [],
               "duplicates": []}

    async def fake(prompt, **kw):
        fake.calls.append(prompt)
        return {"text": "GROUP_VERDICTS:\n#2: fold — same work\n#5: fold — same work\nUMBRELLA: none\n"
                        "UMBRELLA_MEMBERS: 2 5\nSURFACE: code\nCHECK: c\nEVIDENCE: e\nACCEPTANCE: none\n"
                        "ACCEPTANCE_CLAUSES: none\nSPAWNED: none\n",
                "session_id": "g", "stop_reason": "stop", "num_turns": 9, "errors": [], "structured": None}
    fake.calls = []
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    out = asyncio.run(M._execute_group(_Q({"form_umbrellas": False}), cluster, members))
    assert out["folded"] == 0 and out["kept"] == 2 and out["umbrella_id"] is None
    assert "UMBRELLAS ARE OFF" in fake.calls[0]
    assert "grouped" not in _fm(_path(isolated, 2))["tags"]
    assert 2 in B.group_kept_ids(S.LEDGER_PATH)


def test_autocode_yields_to_a_pending_sweep_and_resumes_when_it_is_done(isolated, monkeypatch):
    from workers.sources import DECLINED

    class Q:
        def __init__(self):
            self.enqueued = []
        def wm_get(self, *a): return datetime.now(timezone.utc).isoformat()
        def wm_set(self, *a): pass
        def has_live(self, key): return False
        def enqueue(self, **kw):
            self.enqueued.append(kw)
            return 1
    monkeypatch.setattr(AC, "_loop_is_free", lambda: (True, ""))
    monkeypatch.setitem(AC._boot_settled, "done", True)
    write_item(isolated, 1)
    q = Q()
    assert asyncio.run(AC.enqueue_if_due(q, {"yield_to_sweep": True})) == DECLINED
    assert q.enqueued == []
    assert asyncio.run(AC.enqueue_if_due(q, {"yield_to_sweep": False})) is None, \
        "off: the sweep and the loop share the engine; nothing confirmed, so nothing queued"
    B.record_sweep_verdicts("sw-1", [_item(1)], {1: {"verdict": "keep", "worth": "high", "size": "small",
                                                    "evidence": "e"}}, session_id="s")
    assert asyncio.run(AC.enqueue_if_due(q, {"yield_to_sweep": True})) is None


def test_the_sweep_keys_ride_in_the_payload():
    src = inspect.getsource(M.enqueue_if_due)
    for key in ('"sweep"', '"sweep_batch"', '"sweep_max_turns"', '"form_umbrellas"'):
        assert key in src


# ── oversized umbrellas ─────────────────────────────────────────────────────

def test_unfold_oversized_releases_never_attempted_big_contracts_only(isolated):
    def umbrella(i, *, clauses, members, attempted=False, landed=False):
        text = [f"clause {k} of #{i} holds — tests/test_x.py" for k in range(1, clauses + 1)]
        fm = {"acceptance_clauses": text, "members": members}
        if landed:
            fm[B.LANDED_MARKER] = "abc"
        write_item(isolated, i, status="up_next", tags=("backlog", "umbrella", "spawned-by-triage"), **fm)
        _ev(event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x",
            acceptance_clauses=text, umbrella=True, members=members)
        for m in members:
            write_item(isolated, m, tags=("backlog", "grouped", "spawned-by-triage"), group=i)
            _ev(event="backlog_triage", item_id=m, verdict=B.FOLDED, group=i)
        if attempted:
            _ev(event="backlog_implement", item_id=i, phase="started", round_id=f"SM_{i}")
    umbrella(10, clauses=12, members=[11, 12])
    umbrella(20, clauses=5, members=[21, 22])
    umbrella(30, clauses=9, members=[31], attempted=True)
    umbrella(40, clauses=10, members=[41], landed=True)
    dry = B.unfold_oversized_umbrellas(S.LEDGER_PATH, min_clauses=8, dry_run=True)
    assert [d["umbrella_id"] for d in dry] == [10] and _fm(_path(isolated, 10))["status"] == "up_next"
    out = B.unfold_oversized_umbrellas(S.LEDGER_PATH, min_clauses=8)
    assert [(d["umbrella_id"], d["released"]) for d in out] == [(10, [11, 12])]
    assert _fm(_path(isolated, 10))["status"] == "done" and B.UNFOLDED_TAG in _fm(_path(isolated, 10))["tags"]
    fm11 = _fm(_path(isolated, 11))
    assert fm11["status"] == "draft" and "grouped" not in fm11["tags"] and fm11.get("group") is None
    assert {i.id for i in B.sweep_pool(S.LEDGER_PATH)} >= {11, 12}, "released members are read by the sweep"
    assert _fm(_path(isolated, 20))["status"] == "up_next"
    assert _fm(_path(isolated, 30))["status"] == "up_next"
    assert _fm(_path(isolated, 40))["status"] == "up_next"
    rows = [r for r in S.read_events(path=S.LEDGER_PATH) if r["event"] == "backlog_umbrella_unfolded"]
    assert len(rows) == 1 and rows[0]["oversized"] and rows[0]["clauses"] == 12
    # The reconciler leaves a released member where the sweep will find it.
    assert B.desired_statuses(S.LEDGER_PATH).get(11) is None


# ── the board's shape ───────────────────────────────────────────────────────

def test_board_health_counts_parked_and_the_sweeps_coverage(isolated):
    write_item(isolated, 1)
    write_item(isolated, 2, tags=("backlog", B.SWEPT_TAG, B.PARKED_TAG), worth="low", size="small")
    write_item(isolated, 3, tags=("backlog", B.SWEPT_TAG), worth="high", size="small")
    h = B.board_health(S.LEDGER_PATH)
    assert h["draft"]["parked"] == 1 and h["draft"]["pool"] == 2
    assert h["sweep"] == {"unswept": 1, "swept": 2, "parked": 1,
                          "worth": {"high": 1, "medium": 0, "low": 1}}


def test_rank_key_puts_a_proposal_after_a_defect_of_the_same_rank():
    """Within one worth/size a YouTube-digest proposal sorts after a defect.
    Proposal rounds landed 2 of 29 in the week to 2026-09-16 against 30 of 96
    for defects, and equal rank had handed half of `up_next` to them."""
    def it(i, tags):
        return B.Item(path=Path("x"), id=i, name="n", status="draft", priority="m", created="",
                      body="", worth="high", size="small", tags=tuple(tags))
    order = sorted([it(1, ("backlog", "youtube-eval", B.SWEPT_TAG)), it(2, ("backlog", B.SWEPT_TAG))],
                   key=B.rank_key)
    assert [i.id for i in order] == [2, 1]
    # Rank still beats the proposal tie-break: a high proposal before a medium defect.
    hi_prop = it(3, ("backlog", "youtube-eval")); mid_bug = it(4, ("backlog",)); mid_bug.worth = "medium"
    assert sorted([mid_bug, hi_prop], key=B.rank_key)[0].id == 3
