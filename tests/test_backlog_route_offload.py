"""The backlog list handlers run in the threadpool, so a corpus scan cannot
take the whole backend with it.

Item #1199, cause #2. Both list handlers were `async def` while doing
synchronous `read_text()` + `yaml.safe_load()` over all 1,137 backlog files —
a 1.78 s parse pass, twice per `/tasks` call. A coroutine that blocks does not
merely run slowly, it runs *exclusively*: nothing else on the event loop is
scheduled until it returns. Measured on the live box 2026-09-16: `GET /health`
answered in 0.9-3.1 ms idle and in 3.496 s when fired 0.4 s into a backlog
load. That is why the sluggishness was felt outside the backlog tab — every
other Mission Control tab and every concurrent agent API call queued behind
the page's own render.

`def` instead of `async def` is the whole fix: FastAPI recognises a
non-coroutine endpoint and dispatches it with `run_in_threadpool`, where
blocking is what threads are for. These tests pin the shape the clause names
(`asyncio.iscoroutinefunction` is false for both handlers) and the behaviour —
an `/health`-style probe must be served *while* a scan is still parked inside
`_backlog_parse_fm`, which is impossible on the loop and ordinary in a thread.

The probe test is deliberately mechanism-first: it blocks inside the parse
itself and waits to be released by the probe, rather than racing two timers.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import backlog as BR

PARK_SECONDS = 3.0


def write_item(d: Path, item_id: int, *, body: str = "Do the thing.") -> Path:
    p = d / f"{item_id}-item-{item_id}.md"
    fm = {
        "type": "backlog", "segment": "backlog", "status": "draft",
        "priority": "low", "board": "lloyd", "blocked": False,
        "assigned": False, "position": item_id * 1000,
    }
    p.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# Item {item_id}\n\n{body}\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def backlog_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    for i in range(1, 6):
        write_item(d, i)
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(BR, "_FM_CACHE", {})
    return d


def _probe_alongside(handler_path: str, backlog_dir, monkeypatch) -> dict:
    """Fetch `handler_path` and a probe together; return both latencies.

    `_backlog_parse_fm` is wrapped so the *first* file parks on an event only
    the probe can set. A coroutine handler is holding the event loop inside
    that park, so the probe cannot be dispatched and the park runs to
    `PARK_SECONDS`; a threadpool handler leaves the loop free and the park is
    released in about the probe's own 0.1 s.
    """
    real = BR._backlog_parse_fm
    scan_parked = threading.Event()
    probe_ran = threading.Event()

    def blocking_parse(*args, **kwargs):
        if not scan_parked.is_set():
            scan_parked.set()
            probe_ran.wait(PARK_SECONDS)
        return real(*args, **kwargs)

    monkeypatch.setattr(BR, "_backlog_parse_fm", blocking_parse)

    app = FastAPI()
    app.include_router(BR.router)

    @app.get("/health-probe")
    async def probe():
        await asyncio.sleep(0.1)  # proves the loop is free to run this at all
        probe_ran.set()
        return {"ok": True}

    timings: dict[str, float] = {}
    with TestClient(app) as client:
        def fire(name: str, path: str) -> None:
            t0 = time.monotonic()
            r = client.get(path)
            assert r.status_code == 200, f"{name} -> {r.status_code}"
            timings[name] = time.monotonic() - t0

        with ThreadPoolExecutor(max_workers=2) as pool:
            scan = pool.submit(fire, "scan", handler_path)
            time.sleep(0.15)  # let the scan reach its park
            probe_call = pool.submit(fire, "probe", "/health-probe")
            scan.result(timeout=60)
            probe_call.result(timeout=60)

    timings["probe_ran"] = bool(probe_ran.is_set())
    return timings


# ── clause 4, the literal half ───────────────────────────────────────────────

def test_both_list_handlers_are_plain_def_so_fastapi_uses_the_threadpool():
    for handler in (BR.backlog_tasks, BR.backlog_boards):
        assert not asyncio.iscoroutinefunction(handler), (
            f"{handler.__name__} is still `async def`: it will run on the event "
            "loop and hold it for the whole corpus scan"
        )
        assert inspect.isfunction(handler)


# ── clause 4, the behaviour ──────────────────────────────────────────────────

def test_a_probe_is_served_while_a_cold_tasks_scan_is_parked(backlog_dir, monkeypatch):
    timings = _probe_alongside("/api/backlog/tasks", backlog_dir, monkeypatch)

    assert timings["probe_ran"], (
        "the probe was never served while the scan was parked, so the scan held "
        "the event loop"
    )
    assert timings["probe"] < 1.0, (
        f"probe took {timings['probe']:.2f}s against a 3.1 ms idle baseline: it "
        "waited for the backlog scan to finish, i.e. it was not offloaded"
    )


def test_a_probe_is_served_while_a_cold_boards_scan_is_parked(backlog_dir, monkeypatch):
    """"/boards" is a third full-corpus pass; it gets the same treatment."""
    timings = _probe_alongside("/api/backlog/boards", backlog_dir, monkeypatch)

    assert timings["probe_ran"], "/boards held the event loop"
    assert timings["probe"] < 1.0, (
        f"/boards probe took {timings['probe']:.2f}s; it queued behind the scan"
    )
