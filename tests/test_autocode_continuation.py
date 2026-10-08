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
import subprocess
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


# ── a red-tree item is probed before its attempt is spent ───────────────────
# The only closer for an already-healed red tree was `close_healed_red_tree`,
# reachable only from a FULL green `tests` rung, so #2384 was picked up 11
# minutes after a vault commit had healed its node and spent 22 turns opening no
# round; #1846 spent 14 turns the same way nine days earlier. These two tests
# drive the real `execute` → `close_healed_red_tree_at_pickup` → gate seam and
# stop one layer short of the subprocess: which nodes are still red is decided by
# the probe, and the probe against a real pytest is pinned in
# tests/test_automod_gate.py::test_a_pickup_probe_reruns_the_nodes_it_is_given.

from tests.test_backlog_red_tree import RED, RED2, _fm  # noqa: E402

RED_NODES = [RED, RED2]


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def _red_tree_item(tmp_path, monkeypatch):
    """A real red-tree item on a real git tree `c1 -> c2`, and the loop pointed at
    THAT tree: pickup reads HEAD and checks ancestry against `autocode.LIVE_ROOT`,
    so the fake tree has to be the live one for the item's own commits to mean
    anything. Returns (live tree, item id, the head the probe will answer with).

    The two-commit tree is built here rather than imported as a fixture from
    tests/test_backlog_red_tree.py: a cross-module fixture import is an unused
    name to pyflakes, and the static rung counts it against the diff.
    """
    live = tmp_path / "live"
    live.mkdir()
    _git(live, "init", "-q", "-b", "main")
    _git(live, "config", "user.email", "t@e.com")
    _git(live, "config", "user.name", "t")
    shas = {}
    for name in ("c1", "c2"):
        (live / f"{name}.txt").write_text(name)
        _git(live, "add", "-A")
        _git(live, "commit", "-q", "-m", name)
        shas[name] = _git(live, "rev-parse", "HEAD")
    iid = B.file_red_tree_item(shas["c1"], list(RED_NODES), "SM_RED", [],
                               live_root=live)["item_id"]
    monkeypatch.setattr(I, "LIVE_ROOT", live)
    return live, iid, shas["c2"]


def _probe_answers(live, head, unresolved, *, conclusive=True):
    """Stands in for `gate.red_tree_state_at_head`, called as production calls it
    — one list of node ids plus the tree whose HEAD to read — and stops there:
    which nodes are still red is the probe's answer, and the probe aimed at a real
    pytest is pinned in tests/test_automod_gate.py."""
    def probe(nodes, *, live_root=None):
        assert str(live_root) == str(live), \
            "pickup must ask the tree the loop actually runs against"
        return {"head": head, "probed": list(nodes), "unresolved": list(unresolved),
                "conclusive": conclusive,
                "note": f"probed {len(nodes)} node(s) at base {head[:8]}: "
                        f"{len(unresolved)} already failing"}
    return probe


def _implement_rows(item_id):
    return [e for e in S.read_events(path=S.LEDGER_PATH)
            if e.get("event") == "backlog_implement" and e.get("item_id") == item_id]


def _ledger_rows(event):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == event]


def test_a_red_tree_whose_nodes_all_pass_at_head_is_never_started(env, monkeypatch, tmp_path):
    """Clause 1: the nodes are re-run before the turn, and a green answer starts
    no implement turn — the `started` row that spends the attempt is never
    written, which is the #1846/#2384 signature this item exists to stop."""
    from scripts.automod import gate as _G
    live, iid, head = _red_tree_item(tmp_path, monkeypatch)
    assert B.item_by_id(iid).status == "up_next"
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))
    turn = _landing_turn(env)
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    monkeypatch.setattr(_G, "red_tree_state_at_head", _probe_answers(live, head, []))

    res = asyncio.run(I.execute(_Item()))
    assert res["status"] == "skipped" and res["item_id"] == iid, res
    assert turn.calls == [], "no implement turn, so nothing to record a turn against"
    assert [r["phase"] for r in _implement_rows(iid)] == ["skipped"], \
        "no `started` row: the attempt was never spent"
    assert B.item_by_id(iid).status == "done"
    assert _fm(B.item_by_id(iid))["autotriage_retired"] == "already_done"


def test_a_red_tree_whose_nodes_still_fail_is_started_exactly_as_today(env, monkeypatch, tmp_path):
    """Clause 3 end to end: one node still red at the probed head, and pickup is
    invisible — the `started` row is written and the turn runs."""
    from scripts.automod import gate as _G
    live, iid, head = _red_tree_item(tmp_path, monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))
    turn = _landing_turn(env)
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    monkeypatch.setattr(_G, "red_tree_state_at_head",
                        _probe_answers(live, head, [RED2]))

    res = asyncio.run(I.execute(_Item()))
    assert res["status"] == "success" and len(turn.calls) == 1
    assert [r["phase"] for r in _implement_rows(iid)][:1] == ["started"]
    assert _ledger_rows("red_tree_closed") == [], \
        "the probe found red, so it credited nothing"


def test_an_inconclusive_pickup_probe_starts_the_attempt_and_closes_nothing(
        env, monkeypatch, tmp_path):
    """Clause 4 end to end: the probe answers with no failures AND no answer, and
    the loop proceeds as it did before this existed."""
    from scripts.automod import gate as _G
    live, iid, head = _red_tree_item(tmp_path, monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))
    turn = _landing_turn(env)
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    monkeypatch.setattr(_G, "red_tree_state_at_head",
                        _probe_answers(live, head, [], conclusive=False))

    res = asyncio.run(I.execute(_Item()))
    assert res["status"] == "success" and len(turn.calls) == 1
    assert [r["phase"] for r in _implement_rows(iid)][:1] == ["started"]
    assert _ledger_rows("red_tree_closed") == []


def test_a_probe_that_raises_does_not_stop_the_loop(env, monkeypatch, tmp_path):
    """A pickup probe that cannot even run is not a reason to idle the loop: the
    item is attempted, which is what an unbuilt probe always costs."""
    from scripts.automod import gate as _G
    live, iid, head = _red_tree_item(tmp_path, monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))
    turn = _landing_turn(env)
    monkeypatch.setattr(C, "run_prompt_in_session", turn)

    def boom(nodes):
        raise RuntimeError("no git here")
    monkeypatch.setattr(_G, "red_tree_state_at_head", boom)

    res = asyncio.run(I.execute(_Item()))
    assert res["status"] == "success" and len(turn.calls) == 1
    assert [r["phase"] for r in _implement_rows(iid)][:1] == ["started"]


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
