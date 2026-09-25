"""Event-loop stalls from session-file I/O, on a synthetic 1 MB session (P13.5).

A watchdog coroutine wakes every 1 ms and records how late each wake-up
was (late by more than 10 ms = a stall); anything the loop thread did synchronously in between shows up as lateness.
Three workloads run against one ~1 MB session JSON, each under its own watchdog:

* ``append`` — 40 `_append_messages` calls (every persisted row of a turn goes
  through `mutate_session`: parse, mutate, dump with indent=2, atomic write);
* ``post``   — 10 real `POST /api/message/stream` handler runs up to the enqueue,
  with prefetch, the system prompt and MCP discovery stubbed (they are not
  session I/O), so what is left is the handler's own reads and its meta save;
* ``refresh`` — 60 calls of the per-iteration plan-mode refresher read
  (`_load_session_plan`), which a long turn pays once per harness iteration.

Run from a worktree root, never exporting LLOYD_DATA; it writes only under a
mkdtemp:

    .venvs/lloyd/bin/python eval/measure_session_io_stalls.py

Prints one line per workload: wall time (including a 5 ms gap per operation), wake-ups late by >10/25/50/100 ms, the
worst stall, and the summed lateness.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: The watchdog wakes every millisecond; a wake-up more than 10 ms late is a
#: stall (the plan's "10 ms watchdog" is the threshold, not the period — a 10 ms
#: period would miss most of a stall that starts just after a wake-up).
TICK_S = 0.001
#: Between operations, so the watchdog wakes once per operation and a stall is
#: attributed to one operation rather than to the whole workload.
GAP_S = 0.005


class _Watchdog:
    def __init__(self) -> None:
        self.lags: list[float] = []
        self._stop = False

    async def run(self) -> None:
        while not self._stop:
            t0 = time.perf_counter()
            await asyncio.sleep(TICK_S)
            self.lags.append(max(0.0, time.perf_counter() - t0 - TICK_S))

    def stop(self) -> None:
        self._stop = True

    def summary(self) -> dict:
        ms = [x * 1000 for x in self.lags]
        return {
            "wakeups": len(ms),
            **{f">{t}ms": sum(1 for x in ms if x > t) for t in (10, 25, 50, 100)},
            "max_ms": round(max(ms, default=0.0), 1),
            "sum_late_ms": round(sum(x for x in ms if x > 10), 1),
        }


def _synthetic_session(sid: str, target_bytes: int = 1_000_000) -> dict:
    msgs = []
    filler = ("The quick brown fox jumps over the lazy dog. " * 50)[:2000]
    i = 0
    while True:
        msgs.append({"id": f"m{i}", "role": "user" if i % 2 == 0 else "assistant",
                     "source": "user",
                     "content": [{"type": "text", "text": f"{i} {filler}"}],
                     "timestamp": "2026-09-24T12:00:00"})
        i += 1
        if i % 50 == 0 and len(json.dumps({"messages": msgs}, indent=2)) >= target_bytes:
            break
    return {"session_id": sid, "model": "primary", "platform": "mission-control",
            "todos": [{"content": "t", "status": "pending"}],
            "plan": {"plan_mode": False}, "messages": msgs}


async def _measure(name: str, work) -> None:
    dog = _Watchdog()
    task = asyncio.create_task(dog.run())
    await asyncio.sleep(0.05)
    dog.lags.clear()
    t0 = time.perf_counter()
    await work()
    wall = time.perf_counter() - t0
    dog.stop()
    await task
    print(f"{name:8s} wall={wall:6.2f}s  " + "  ".join(
        f"{k}={v}" for k, v in dog.summary().items()), flush=True)


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="p13-stalls-"))
    import app.routers.automod as automod
    import app.routers.messages as messages
    import app.sessions_io as sessions_io
    import app.routers._messages_inner_voice as iv

    mods = [messages, sessions_io, iv]
    try:
        import app.routers.turn_options as turn_options
        mods.append(turn_options)
    except ImportError:
        turn_options = None
    for mod in mods:
        mod.SESSIONS_DIR = tmp

    async def _prefetch(text, session_id="", plan_mode=False, **_):
        return text

    async def _enqueue(session_id, turn, *, consumer_factory=None):
        return {}

    fakes = {
        "prefetch_context_async": _prefetch,
        "build_system_prompt": lambda **kw: "SYSTEM",
        "enqueue_turn": _enqueue,
        "set_last_user_session": lambda sid: None,
        "_get_mcp_servers": lambda: {},
        "_get_disallowed_tools": lambda plan_mode=False: [],
        "_get_harness_kwargs": lambda: {},
        "log_turn_prompt_budget": lambda *a, **k: None,
    }
    for mod in (messages, turn_options):
        if mod is None:
            continue
        for k, v in fakes.items():
            setattr(mod, k, v)
    automod.drain_active = lambda: False

    sid = "20260924_120000_stall"
    path = tmp / f"{sid}.json"
    path.write_text(json.dumps(_synthetic_session(sid), indent=2))
    print(f"session: {path.stat().st_size / 1e6:.2f} MB, "
          f"turn_options={'yes' if turn_options else 'no'}", flush=True)

    class _Req:
        def __init__(self, body):
            self._b = body

        async def json(self):
            return self._b

    async def appends():
        for i in range(40):
            await sessions_io._append_messages(sid, [{
                "id": f"a{i}", "role": "assistant",
                "content": [{"type": "text", "text": "x" * 500}]}])
            await asyncio.sleep(GAP_S)

    async def posts():
        for i in range(10):
            await messages.post_message_stream(_Req({"text": f"hi {i}", "session_id": sid}))
            await asyncio.sleep(GAP_S)

    async def refreshes():
        for _ in range(60):
            messages._load_session_plan(sid)
            await asyncio.sleep(GAP_S)

    await _measure("append", appends)
    await _measure("post", posts)
    await _measure("refresh", refreshes)


if __name__ == "__main__":
    asyncio.run(main())
