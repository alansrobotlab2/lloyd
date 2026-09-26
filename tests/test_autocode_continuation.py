"""The next item on a slot continues in the previous round's session.

Every round started cold — a fresh session, the ~55k-token prompt, and a
median 64 tool calls before its first gate (week to 2026-09-25), much of it
re-learning the tree, the test commands and the gate protocol the previous
round on that slot had just used. The hand sweeps of 2026-09-24 cleared 60
items in one session. A finished round's session rebuilds to 20–55k tokens
(tool results are stored as 2 KB pointers), so the next item starts there:
one round per item and one turn per pool job as before, only the session
shared — and only after a turn that ended on its own with its item decided.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import pytest

import app.paths as paths
import app.sessions_io as sio
from scripts.automod import backlog as B, state as S
from workers.sources import _common as C
from workers.sources import autocode as I

from tests.test_backlog_unattended import _confirm, write_item

SLOT = "autocode:round"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(paths, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(sio, "active_sessions_snapshot", lambda: [])
    monkeypatch.setattr(I, "_source_cfg", lambda name: {})
    world = {"backlog": d, "sessions": sessions, "tokens": 30_000}

    async def rebuilt(path, model=""):
        return {"history": [{"role": "user", "content": "x"}], "tokens_after": world["tokens"]}
    import app.compaction as comp
    monkeypatch.setattr(comp, "load_and_compact_session", rebuilt)
    return world


def _session(env, sid):
    (env["sessions"] / f"{sid}.json").write_text(json.dumps({"session_id": sid, "messages": []}))


def _row(phase, *, sid="sess_a", slot=SLOT, continuable=True, chain=1, item=10,
         round_id="SM_A", age=60):
    S.append_event({"event": "backlog_implement", "phase": phase, "item_id": item,
                    "slot": slot, "session_id": sid, "continuable": continuable,
                    "chain": chain, "round_id": round_id, "ts": time.time() - age},
                   path=S.LEDGER_PATH)


def _warm():
    return asyncio.run(I._warm_session(SLOT))


# ── when it continues ───────────────────────────────────────────────────────

def test_a_clean_finish_on_the_slot_is_continued(env):
    _session(env, "sess_a")
    _row("finished")
    w = _warm()
    assert w == {"session_id": "sess_a", "chain": 2, "history_tokens": 30_000,
                 "prev_item": 10, "prev_round": "SM_A"}


def test_another_slots_session_is_not_this_slots(env):
    _session(env, "sess_a")
    _row("finished", slot="autocode:round:1")
    assert _warm() is None


@pytest.mark.parametrize("why", ["not_continuable", "started_after", "infra_failed",
                                 "chain_full", "too_old", "busy", "no_file", "too_big",
                                 "switch_off", "no_slot"])
def test_it_starts_cold_when_it_should(env, monkeypatch, why):
    _session(env, "sess_a")
    _row("finished", continuable=(why != "not_continuable"),
         chain=(I.CONTINUE_MAX_ITEMS if why == "chain_full" else 1),
         age=(I.CONTINUE_WITHIN_S + 60 if why == "too_old" else 60))
    if why == "started_after":
        _row("started", sid="")            # a turn in flight, or one that died unrecorded
    if why == "infra_failed":
        _row("infra_failed")
    if why == "busy":
        monkeypatch.setattr(sio, "active_sessions_snapshot", lambda: [{"session_id": "sess_a"}])
    if why == "no_file":
        (env["sessions"] / "sess_a.json").unlink()
    if why == "too_big":
        env["tokens"] = I.CONTINUE_MAX_HISTORY_TOKENS + 1
    if why == "switch_off":
        monkeypatch.setattr(I, "_source_cfg", lambda name: {"continue_session": False})
    if why == "no_slot":
        assert asyncio.run(I._warm_session(None)) is None
        return
    assert _warm() is None, why


def test_the_limits_are_configurable(env, monkeypatch):
    _session(env, "sess_a")
    _row("finished", chain=3)
    assert _warm() is None
    monkeypatch.setattr(I, "_source_cfg", lambda name: {"continue_max_items": 5})
    assert _warm()["chain"] == 4


# ── end to end through `execute` ────────────────────────────────────────────

class _Item:
    def __init__(self):
        self.payload = {}
        self.dedup_key = SLOT


def _landing_turn(env):
    """A turn that opens a round and lands it, in whatever session it is given."""
    calls = []

    async def turn(prompt, **kw):
        sid = kw.get("session_id") or f"sess_{len(calls)}"
        calls.append({"prompt": prompt, **kw, "used_session": sid})
        rid = f"SM_{len(calls)}"
        item_id = int(re.search(r"Backlog item #(\d+)", prompt).group(1))
        S.append_event({"event": "round_start", "round_id": rid, "session_id": sid,
                        "item_id": item_id}, path=S.LEDGER_PATH)
        S.append_event({"event": "promoted", "round_id": rid, "commit": f"c{len(calls)}"},
                       path=S.LEDGER_PATH)
        _session(env, sid)
        return {"text": "Landed.", "session_id": sid, "stop_reason": "stop",
                "num_turns": 20, "errors": [], "structured": None}
    turn.calls = calls
    return turn


def test_two_items_on_one_slot_share_one_session(env, monkeypatch):
    for n in (21, 22):
        write_item(env["backlog"], n, status="up_next")
        _confirm(n)
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))
    turn = _landing_turn(env)
    monkeypatch.setattr(C, "run_prompt_in_session", turn)

    first = asyncio.run(I.execute(_Item()))
    second = asyncio.run(I.execute(_Item()))

    assert first["status"] == second["status"] == "success"
    a, b = turn.calls
    assert a.get("session_id") is None and not a["prompt"].startswith("<next_item")
    assert b["session_id"] == a["used_session"], "the second item started cold"
    assert b["prompt"].startswith("<next_item continuing_session=\"true\"")
    assert f"#{first['item_id']}" in b["prompt"].split("</next_item>")[0]
    rows = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "backlog_implement"]
    started = [r for r in rows if r["phase"] == "started"]
    finished = [r for r in rows if r["phase"] == "finished"]
    assert "continues_session" not in started[0]
    assert started[1]["continues_session"] == a["used_session"] and started[1]["chain"] == 2
    assert [f["chain"] for f in finished] == [1, 2]
    assert all(f["continuable"] and f["slot"] == SLOT for f in finished)


def test_a_turn_that_left_its_round_open_is_not_continued(env, monkeypatch):
    """No landing seen at turn end: the next item starts cold, whatever the
    reaper then does with the round."""
    write_item(env["backlog"], 31, status="up_next")
    _confirm(31)
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))

    async def stalled(prompt, **kw):
        S.append_event({"event": "round_start", "round_id": "SM_S", "session_id": "sess_s",
                        "item_id": 31}, path=S.LEDGER_PATH)
        return {"text": "I made the change.", "session_id": "sess_s", "stop_reason": "stop",
                "num_turns": 30, "errors": [], "structured": None}
    monkeypatch.setattr(C, "run_prompt_in_session", stalled)
    monkeypatch.setattr(I, "reap_abandoned_rounds", lambda *a, **k: [])
    asyncio.run(I.execute(_Item()))
    fin = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e["event"] == "backlog_implement" and e["phase"] == "finished"][-1]
    assert fin["continuable"] is False
    assert _warm() is None
