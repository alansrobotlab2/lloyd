"""autocode and autotriage keep their blocking work off the event loop.

The backend's event loop serves every HTTP request and streams every chat
turn, and both sources run inside it. A whole-board walk (~2 s), a ledger scan
or a `git worktree prune` (a git lock held by a landing stalls it for up to
120 s) run synchronously in `execute` or `enqueue_if_due` stalls all of that.
The sibling sources (board_steward, owed_check, backlog_cluster) thread this
work; these tests pin the same for the two loop sources.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading

import pytest

import app.paths as paths
import app.sessions_io as sio
from scripts.automod import backlog as B, state as S
from workers.sources import _common as C
from workers.sources import autocode as I
from workers.sources import autotriage as M

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
    B._IMPLEMENT_CLAIMS.clear()
    B._TRIAGE_CLAIMS.clear()
    yield {"backlog": d, "sessions": sessions}
    B._IMPLEMENT_CLAIMS.clear()
    B._TRIAGE_CLAIMS.clear()


def _on_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _watch(monkeypatch, owner, names, seen):
    """Wrap `owner.<name>` so every call records whether it ran on the loop thread."""
    for name in names:
        real = getattr(owner, name)

        def wrapped(*a, __real=real, __name=name, **kw):
            seen.setdefault(__name, []).append(_on_loop())
            return __real(*a, **kw)
        monkeypatch.setattr(owner, name, wrapped)


class _Item:
    def __init__(self):
        self.payload = {}
        self.dedup_key = SLOT


def _landing_turn(env):
    async def turn(prompt, **kw):
        sid = kw.get("session_id") or "sess_a"
        item_id = int(re.search(r"Backlog item #(\d+)", prompt).group(1))
        S.append_event({"event": "round_start", "round_id": "SM_1", "session_id": sid,
                        "item_id": item_id}, path=S.LEDGER_PATH)
        S.append_event({"event": "promoted", "round_id": "SM_1", "commit": "c1"},
                       path=S.LEDGER_PATH)
        (env["sessions"] / f"{sid}.json").write_text(json.dumps({"session_id": sid}))
        return {"text": "Landed.", "session_id": sid, "stop_reason": "stop",
                "num_turns": 20, "errors": [], "structured": None}
    return turn


def test_an_implement_turn_walks_the_board_and_the_ledger_off_the_event_loop(env, monkeypatch):
    write_item(env["backlog"], 21, status="up_next")
    _confirm(21)
    seen: dict[str, list[bool]] = {}
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))
    monkeypatch.setattr(C, "run_prompt_in_session", _landing_turn(env))
    _watch(monkeypatch, I, ["_loop_is_free", "_pacing_marks", "_reoffer_for",
                            "_maybe_flush", "reap_abandoned_rounds"], seen)
    _watch(monkeypatch, B, ["select_confirmed", "max_item_id", "set_status",
                            "close_settled_items", "reconcile_statuses",
                            "settle_item_verdict"], seen)

    res = asyncio.run(I.execute(_Item()))

    assert res["status"] == "success", res
    for name in ("_loop_is_free", "select_confirmed", "max_item_id", "set_status",
                 "_pacing_marks", "_reoffer_for", "reconcile_statuses", "_maybe_flush",
                 "settle_item_verdict", "reap_abandoned_rounds"):
        assert seen.get(name), f"{name} was never called: the probe is not wired"
    on_loop = sorted(n for n, calls in seen.items() if any(calls))
    assert on_loop == [], f"ran on the event loop: {on_loop}"


def test_a_hung_worktree_list_does_not_hold_the_loop_in_enqueue_or_execute(env, monkeypatch):
    """`_loop_is_free` is held on an event only another task on the loop can set:
    on the loop thread it would wait out its timeout, off it the loop answers."""
    seen = {}

    def blocking_free(depth=None):
        seen["released"] = release.wait(timeout=5)
        return False, "held for the test"

    monkeypatch.setattr(I, "_loop_is_free", blocking_free)
    monkeypatch.setattr(I, "_housekeeping_due", lambda q, c: False)
    monkeypatch.setattr(I, "_release_stranded_quietly", lambda: None)
    monkeypatch.setattr(I, "_maybe_flush", lambda *a, **k: None)
    monkeypatch.setattr(I, "_reap_quietly", lambda: None)
    monkeypatch.setitem(I._boot_settled, "done", True)

    async def other_task():
        await asyncio.sleep(0.05)
        release.set()

    async def go(coro):
        _, out = await asyncio.gather(other_task(), coro)
        return out

    release = threading.Event()
    asyncio.run(go(I.enqueue_if_due(object(), {})))
    assert seen["released"] is True, "enqueue_if_due ran the worktree read on the loop"

    release = threading.Event()
    out = asyncio.run(go(I.execute(_Item())))
    assert seen["released"] is True, "execute ran the worktree read on the loop"
    assert out == {"status": "skipped", "summary": "held for the test"}


class _QItem:
    def __init__(self, payload=None):
        self.payload = payload or {}
        self.id = 1


def test_a_triage_turn_selects_off_the_event_loop(env, monkeypatch):
    write_item(env["backlog"], 31, status="draft")
    seen: dict[str, list[bool]] = {}

    async def turn(prompt, **kw):
        return {"text": "VERDICT: unverifiable\nSURFACE: code\nCHECK: grep -n x app.py\n"
                        "EVIDENCE: e\nACCEPTANCE: none\nSPAWNED: none\n",
                "session_id": "s", "stop_reason": "stop", "num_turns": 3, "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    _watch(monkeypatch, B, ["select_candidate", "triage_pool", "max_item_id",
                            "select_cluster"], seen)

    out = asyncio.run(M.execute(_QItem({"group_triage": True, "implement_pool_floor": 40})))

    assert out.get("item_id") == 31, out
    for name in ("select_candidate", "triage_pool", "max_item_id", "select_cluster"):
        assert seen.get(name), f"{name} was never called: the probe is not wired"
    on_loop = sorted(n for n, calls in seen.items() if any(calls))
    assert on_loop == [], f"ran on the event loop: {on_loop}"
