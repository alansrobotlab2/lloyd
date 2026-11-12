"""`run_round._off_loop`: a round's blocking promotion/rollback leaves the backend's
event loop free, and a cancellation still waits for the call it started.

2026-11-12: `restore_for_decline` -> `vault_round.land` -> the vault-guards pytest
probe ran inline on the backend's loop, and /health stopped answering for the
length of that suite.
"""

from __future__ import annotations

import asyncio
import threading
import time

from scripts.autoresearch import run_round


async def test_the_loop_keeps_running_while_the_call_blocks():
    release = threading.Event()
    ticks = 0

    def blocking(x, *, y):
        release.wait(5)
        return x + y

    async def ticker():
        nonlocal ticks
        while not release.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    t = asyncio.create_task(ticker())
    call = asyncio.create_task(run_round._off_loop(blocking, 1, y=2))
    await asyncio.sleep(0.2)
    seen = ticks
    release.set()
    assert await call == 3
    await t
    assert seen >= 5, f"the event loop ticked {seen} times while the call blocked"


async def test_a_cancellation_waits_for_the_started_call_to_finish():
    finished = threading.Event()

    def slow():
        time.sleep(0.3)
        finished.set()
        return "landed"

    try:
        await asyncio.wait_for(run_round._off_loop(slow), timeout=0.05)
    except (asyncio.TimeoutError, TimeoutError):
        pass
    else:
        raise AssertionError("wait_for should have timed out")
    assert finished.is_set(), (
        "the timeout was delivered before the promotion/rollback it interrupted had "
        "finished; the pool would free the slot while it still runs")


async def test_an_exception_in_the_call_reaches_the_caller():
    def boom():
        raise ValueError("refused")

    try:
        await run_round._off_loop(boom)
    except ValueError as exc:
        assert str(exc) == "refused"
    else:
        raise AssertionError("the exception was swallowed")
