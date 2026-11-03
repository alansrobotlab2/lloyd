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
import re
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


# ── #3410 clause 3: the rationale is a ratio, and the reasoning survives ─────
#
# This file's own docstring records why the handlers are plain `def`, and so does
# the comment above the boards route. Both used to lean on a pair of absolutes —
# "0.033 s to read, 1.78 s to parse" — and that pair is exactly what rots: on
# 2026-10-26 the read was 0.14 s and the pass 16.0 s, so the numbers are 4x and 9x
# off while the *ratio* is the one thing that held (54x then, 111x now — the same
# "two orders of magnitude" claim). A rewrite that only refreshed the digits would
# be re-rotted by the next 500 items, so the prose now states the gap as a ratio
# and keeps the dated figures in exactly one place, the module docstring's
# re-measurement line (pinned in tests/test_backlog_doc_claims.py).
#
# These checks read the source rather than the imported module because the thing
# under test is prose: a comment cannot be introspected, and the failure being
# prevented is a comment drifting away from the behaviour it describes.

ROUTE = Path(__file__).resolve().parents[1] / "app" / "routers" / "backlog.py"


def _comment_block_above(needle: str) -> str:
    """The contiguous `#` lines sitting directly above the first line with `needle`.

    Scoped to the block rather than the whole file on purpose: `in src` would pass
    the moment the phrase appeared anywhere, and the claim here is that *this*
    rationale still gives this route its reason for being a plain `def`.
    """
    lines = ROUTE.read_text(encoding="utf-8").splitlines()
    at = [i for i, ln in enumerate(lines) if needle in ln]
    assert at, (f"no line of app/routers/backlog.py contains {needle!r}; the comment "
                f"this node checks is gone or the anchor was renamed")
    out: list[str] = []
    i = at[0] - 1
    while i >= 0 and lines[i].lstrip().startswith("#"):
        out.append(lines[i].lstrip().lstrip("#").strip())
        i -= 1
    assert out, f"no comment block above {needle!r}"
    # Flattened, because these are prose checks on wrapped text: the sentence
    # "costs roughly two orders of magnitude more" is legal to break between "two"
    # and "orders", and a check that failed on the wrap would be an instrument that
    # punishes good line-wrapping until someone re-wraps the comment to please it.
    return " ".join(" ".join(reversed(out)).split())


def _module_docstring() -> str:
    """The module docstring, off the parse, flattened for prose checks.

    Off the parse rather than off the bytes around line 1: slicing the file's head
    would pass on any leading comment or encoding line, and would check the
    docstring's neighbours instead of the docstring.
    """
    import ast

    doc = ast.get_docstring(ast.parse(ROUTE.read_text(encoding="utf-8"))) or ""
    return " ".join(doc.split())


def test_the_parse_versus_io_fact_is_stated_as_a_ratio_not_two_absolutes():
    doc = _module_docstring()
    assert doc, "app/routers/backlog.py lost its module docstring"
    assert "two orders of magnitude" in doc, (
        "the docstring no longer states the parse-to-I/O gap as a ratio. The pair "
        "it used to quote — 0.033 s read against a 1.78 s pass — is 4x and 9x low "
        "as at 2026-10-26, while their ratio moved only from 54x to 111x"
    )
    assert "111x" in doc and "54x" in doc, (
        "the ratio must carry both measurements (54x on 2026-09-16, 111x on "
        "2026-10-26): one point is an assertion, two points are a trend, and the "
        "trend is what tells the next reader whether the gap itself is drifting"
    )
    assert "1.78 s" not in doc and "0.033 s" not in doc, (
        "the rotted absolutes are back in the docstring, where a reader would take "
        "them for the live cost of a corpus pass"
    )


def test_the_plain_def_rationale_and_the_lock_note_survive_the_rewording():
    """The reasoning must outlive the numbers it was written beside.

    Two claims, both load-bearing: corpus-reading handlers are plain `def` because
    a blocking coroutine holds the loop, and `_FM_CACHE_LOCK` is deliberately not
    held across the parse. The second is the one a rewording loses, because it is a
    note about a cost ("1.78 s over 1,137 files") that this item removed from the
    file — so it is checked by name here.
    """
    boards = _comment_block_above('/api/backlog/boards')
    assert "not `async def`" in boards, "the block no longer states the shape it argues for"
    assert "run_in_threadpool" in boards, (
        "the mechanism is gone: without it the block says 'def' but not why FastAPI obeys")
    assert "exclusively" in boards, (
        "the reason a blocking handler is worse than a slow one has been reworded away")
    assert "two orders of magnitude" in boards, (
        "the boards comment quotes an absolute corpus size again instead of the ratio")
    assert "test_backlog_route_offload.py" in boards, (
        "the block's own witness citation was dropped, and a prose citation that "
        "names the file keeping it honest is the difference between a claim and a "
        "guarantee (#2069)")

    lock = _comment_block_above("_FM_CACHE_LOCK = threading.Lock()")
    assert "never across `_backlog_parse_fm`" in lock, (
        "the note that the parse runs outside the lock is gone — the design reason "
        "for the lock's narrow scope, and the one that a cost figure's removal "
        "would otherwise take with it")
    assert "ms a file" not in lock and "ms/file" not in lock, (
        "a per-file cost is quoted as an absolute again; mean file size doubled "
        "between the two measurements, so it was wrong twice over")
    assert not re.search(r"1[\d,]{2,}\s+files", lock), (
        "the lock comment states a corpus file count again, which is the figure "
        "that tripled and the reason this item exists")
