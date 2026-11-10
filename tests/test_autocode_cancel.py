"""A cancelled implement turn still ends: a terminal row, a stopped backend turn.

The pool cancels `execute` when its own `max_duration_seconds` runs out, and on
a pool stop. `CancelledError` skipped the `TurnTimeout` handler, so the item kept
a `started` row with nothing after it (`items_with_unfinished_rounds` held it
until the next boot) and the backend turn ran on with nobody owning it. The turn
clock also started after the pre-turn work, which runs on the pool's clock.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

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
    monkeypatch.setattr(I, "_loop_is_free", lambda depth=None: (True, "free"))
    B._IMPLEMENT_CLAIMS.clear()
    write_item(d, 21, status="up_next")
    _confirm(21)
    yield d
    B._IMPLEMENT_CLAIMS.clear()


class _Item:
    def __init__(self, claimed_at=None):
        self.payload = {}
        self.dedup_key = SLOT
        self.claimed_at = claimed_at


def _phases(item_id=21):
    return [(e["phase"], e) for e in S.read_events(path=S.LEDGER_PATH, limit=10**6)
            if e.get("event") == "backlog_implement" and e.get("item_id") == item_id]


def test_a_pool_timeout_writes_the_terminal_row_and_releases_the_item(monkeypatch):
    async def hangs(prompt, **kw):
        await asyncio.sleep(30)
    monkeypatch.setattr(C, "run_prompt_in_session", hangs)
    monkeypatch.setattr(I, "_pool_cap_seconds", lambda: 0.4)   # the cancel IS the cap

    async def pool():
        await asyncio.wait_for(I.execute(_Item()), timeout=0.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(pool())

    rows = _phases()
    assert [p for p, _ in rows] == ["started", "finished"], rows
    end = rows[-1][1]
    assert end["cancelled"] is True and end["stop_reason"] == "turn_timeout", end
    assert 21 not in B.items_with_unfinished_rounds(S.LEDGER_PATH), \
        "a `started` row with nothing after it holds the item until the next boot"
    assert B.implement_claimed_ids() == set()


class _StoppedPool:
    _running = False


@pytest.mark.parametrize("how", ["pool_flag", "before_the_cap"])
def test_a_pool_stop_is_an_outage_row_not_an_attempt(monkeypatch, how):
    """A UI stop or a backend shutdown cancels `execute` too. Before the restart
    `settle_orphaned_turns` charged that once to the outage budget; read as the
    pool's timeout it spent one of the item's attempts."""
    async def hangs(prompt, **kw):
        await asyncio.sleep(30)
    monkeypatch.setattr(C, "run_prompt_in_session", hangs)
    import workers.pool as pool_mod
    if how == "pool_flag":
        monkeypatch.setattr(pool_mod, "get_pool", lambda: _StoppedPool())
        monkeypatch.setattr(I, "_pool_cap_seconds", lambda: 0.4)
        monkeypatch.setattr(I, "_backend_boot_ts", lambda: 1234.0)
    # else: the default cap is an hour, and the cancel comes half a second in.

    async def pool():
        await asyncio.wait_for(I.execute(_Item()), timeout=0.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(pool())

    rows = _phases()
    assert [p for p, _ in rows] == ["started", "infra_failed"], rows
    end = rows[-1][1]
    assert end["stop_reason"] == "pool_stopped", end
    if how == "pool_flag":
        assert end["backend_boot"] == 1234.0, end
    assert B.implement_outcomes(S.LEDGER_PATH)[21][0] != "spent"
    assert 21 not in B.implemented_ids(S.LEDGER_PATH), "a stop spent the item's attempt"
    assert 21 not in B.items_with_unfinished_rounds(S.LEDGER_PATH)


def test_a_cancel_after_the_turn_recorded_writes_no_second_terminal_row(monkeypatch):
    """Cancelled during the post-turn bookkeeping: the turn's own row stands, and
    a second `finished` would charge the item a second attempt."""
    async def turn(prompt, **kw):
        return {"text": "done", "session_id": "sess_a", "stop_reason": "stop",
                "num_turns": 3, "errors": [], "structured": None}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    monkeypatch.setattr(I, "_escalate_review_disagreement",
                        lambda *a, **k: time.sleep(1.0))

    async def pool():
        await asyncio.wait_for(I.execute(_Item()), timeout=0.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(pool())
    assert [p for p, _ in _phases()] == ["started", "finished"]
    assert "cancelled" not in _phases()[-1][1]


def test_the_turn_clock_is_what_the_pool_has_left_since_the_claim(monkeypatch):
    seen = []

    async def turn(prompt, **kw):
        seen.append(kw.get("timeout_seconds"))
        return {"text": "done", "session_id": "sess_a", "stop_reason": "stop",
                "num_turns": 3, "errors": [], "structured": None}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    monkeypatch.setattr(C, "turn_timeout_for", lambda source: 3540.0)

    claimed = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
    asyncio.run(I.execute(_Item(claimed_at=claimed)))
    assert seen and 2930 <= seen[0] <= 2941, seen


async def test_a_cancelled_worker_turn_is_stopped_in_the_backend(monkeypatch):
    """The pool's cancel reaches the HTTP request, not the turn: the chat path
    keeps running when its client goes away, so the worker asks it to stop."""
    cancelled: list[str] = []

    async def fake_cancel(backend, session_id):
        cancelled.append(session_id)
        return True
    monkeypatch.setattr(C, "_cancel_session_turn", fake_cancel)

    async def never_finishes(*a, **k):
        await asyncio.sleep(30)
    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", never_finishes)

    task = asyncio.ensure_future(
        C.run_prompt_in_session("go", title="t", source="autocode", timeout_seconds=600))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    assert len(cancelled) == 1 and cancelled[0], "the backend turn was left running"
    assert excinfo.value.session_id == cancelled[0], "the caller cannot name the session"
    assert excinfo.value.backend_cancelled is True


@pytest.mark.parametrize("accepted", [True, False])
def test_a_turn_timeout_names_its_session_and_reaps_only_after_an_accepted_cancel(
        monkeypatch, accepted):
    """A cancel the backend refused leaves the turn running; reaping under it can
    run `commit_pending` or `worktree remove --force` beneath a live Bash child."""
    async def times_out(prompt, **kw):
        raise C.TurnTimeout("exceeded", session_id="sess_t", cancelled=accepted)
    monkeypatch.setattr(C, "run_prompt_in_session", times_out)
    reaps = []
    monkeypatch.setattr(I, "reap_abandoned_rounds",
                        lambda *a, **k: reaps.append(k.get("finished_session")) or [])

    res = asyncio.run(I.execute(_Item()))

    assert res["status"] == "failed", res
    end = _phases()[-1][1]
    assert end["phase"] == "finished" and end["session_id"] == "sess_t", end
    assert reaps == (["sess_t"] if accepted else []), reaps


def test_a_cancel_while_the_turn_writes_its_row_waits_for_that_row(monkeypatch):
    """The cancel lands while `_record_run`'s thread is still writing `finished`:
    the cancel path must see that row, not race it and write a second one."""
    async def turn(prompt, **kw):
        return {"text": "done", "session_id": "sess_a", "stop_reason": "stop",
                "num_turns": 3, "errors": [], "structured": None}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    real = I._record_run

    def slow_record(*a, **k):
        time.sleep(1.0)
        return real(*a, **k)
    monkeypatch.setattr(I, "_record_run", slow_record)

    async def pool():
        await asyncio.wait_for(I.execute(_Item()), timeout=0.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(pool())
    rows = _phases()
    assert [p for p, _ in rows] == ["started", "finished"], [p for p, _ in rows]
    assert "cancelled" not in rows[-1][1]
    assert I._TERMINAL_WRITES == {}


@pytest.mark.parametrize("cap", [0.4, None])     # the pool's timeout; a pool stop
def test_a_pool_cancel_row_names_the_session_and_whether_its_cancel_was_taken(
        monkeypatch, cap):
    async def hangs(prompt, **kw):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError as exc:   # what `run_prompt_in_session` attaches
            exc.session_id, exc.backend_cancelled = "sess_c", False
            raise
    monkeypatch.setattr(C, "run_prompt_in_session", hangs)
    if cap is not None:
        monkeypatch.setattr(I, "_pool_cap_seconds", lambda: cap)

    async def pool():
        await asyncio.wait_for(I.execute(_Item()), timeout=0.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(pool())
    end = _phases()[-1][1]
    assert end["phase"] == ("finished" if cap else "infra_failed"), end
    assert end["session_id"] == "sess_c" and end["cancel_accepted"] is False, end


def test_the_reaper_holds_a_round_whose_cancel_the_backend_refused(monkeypatch, tmp_path):
    """The snapshot can miss a turn the backend is still unwinding; a refused
    cancel says it may still be running, so its round is not reaped at once."""
    from scripts.automod import worktree as W
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setattr(W, "worktree_path", lambda rid: wt)
    monkeypatch.setattr(S, "gate_in_progress", lambda rid: False)
    monkeypatch.setattr(S, "land_in_progress", lambda rid: False)
    monkeypatch.setattr(S, "read_current", lambda: None)
    landed = []
    monkeypatch.setattr(I, "_land_if_passed",
                        lambda rid, e, ev: landed.append(rid) or {"round_id": rid,
                                                                  "reason": "x"})
    now = time.time()
    S.append_event({"event": "backlog_implement", "item_id": 21, "phase": "finished",
                    "round_id": "SM_R", "session_id": "sess_c", "cancel_accepted": False,
                    "stop_reason": "turn_timeout"}, path=S.LEDGER_PATH)
    assert I.reap_abandoned_rounds(now=now + 60) == [] and landed == []
    I.reap_abandoned_rounds(now=now + I.ORPHAN_ROUND_MIN_AGE_SECONDS + 60)
    assert landed == ["SM_R"], "the hold is bounded"
