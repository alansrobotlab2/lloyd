"""#1133: the Mission Control mirror's timestamp is a change-time, and says so.

`mc-state.json` carried one `last_updated`, stamped by `set_state` — which the
frontend calls only when the tab, focus or IDE state CHANGES (`useMcStateSync.ts`
has no heartbeat). On 2026-09-20 it read 3 h 24 m old while the mirrored value
was still correct, so the name asserted a liveness the field cannot support, no
per-tab recency was derivable (`focus_by_tab` entries carried only
`{kind, id, label?}`), and `get_focus_snapshot()` advertised a sync caller that
was never written. One test per acceptance clause, in the item's order.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app import mc_state
from app.routers import mc_ui
from agent_mcp import mission_control_ui

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fresh_mirror(tmp_path, monkeypatch):
    """A blank in-memory mirror persisting to a scratch file."""
    monkeypatch.setattr(mc_state, "_STATE_PATH", tmp_path / "mc-state.json")
    monkeypatch.setattr(mc_state, "_state", {
        "tab": "inner_voice", "focus_by_tab": {}, "ide": None, "last_changed": None,
    })
    return tmp_path / "mc-state.json"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(mc_ui.router)
    return app


# ── clause 1: named for what it is ────────────────────────────────────


def test_snapshot_exposes_last_changed_and_no_last_updated(fresh_mirror):
    async def run():
        await mc_state.set_state("backlog", {"kind": "backlog", "id": "7"})
        return await mc_state.get_state()
    snap = asyncio.run(run())
    assert snap["last_changed"] is not None
    assert "last_updated" not in snap
    # The definition says what the field is NOT, beside the field itself.
    src = (ROOT / "app" / "mc_state.py").read_text()
    lines = src.splitlines()
    i = next(n for n, ln in enumerate(lines) if '"last_changed": None' in ln)
    comment = " ".join(ln.strip() for ln in lines[max(0, i - 12):i]
                       if ln.strip().startswith("#"))
    assert "not" in comment.lower() and "liveness" in comment.lower(), comment
    assert "change" in comment.lower()


# ── clause 2: a restart over the old on-disk key keeps the change-time ──


def test_hydrating_a_legacy_last_updated_file_keeps_the_change_time(fresh_mirror):
    fresh_mirror.write_text(json.dumps({
        "tab": "tools",
        "focus_by_tab": {"tools": {"kind": "tool", "id": "Bash"}},
        "ide": None,
        "last_updated": "2026-09-20T03:16:36+00:00",
    }))
    mc_state._load_from_disk()
    assert mc_state._state["last_changed"] == "2026-09-20T03:16:36+00:00"
    assert mc_state._state["tab"] == "tools"


# ── clause 3: per-entry recency ───────────────────────────────────────


def test_each_focus_entry_carries_its_own_ordered_changed_at(fresh_mirror):
    async def run():
        await mc_state.set_state("backlog", {"kind": "backlog", "id": "7"})
        await mc_state.set_state("tools", {"kind": "tool", "id": "Bash",
                                           # a client-sent stamp is not trusted
                                           "changed_at": "1999-01-01T00:00:00+00:00"})
        return await mc_state.get_state()
    snap = asyncio.run(run())
    by_tab = snap["focus_by_tab"]
    a, b = by_tab["backlog"]["changed_at"], by_tab["tools"]["changed_at"]
    assert a and b and a != b
    assert a < b, (a, b)   # ISO-8601 UTC, so string order is time order
    assert b == snap["last_changed"]
    most_recent = max(by_tab, key=lambda t: by_tab[t]["changed_at"])
    assert most_recent == "tools"
    # The stamp is also what reaches disk, so a restart keeps the ordering.
    on_disk = json.loads(fresh_mirror.read_text())
    assert on_disk["focus_by_tab"]["backlog"]["changed_at"] == a


# ── clause 4: the route and the tool both carry it ───────────────────


def test_route_and_tool_carry_the_change_time_and_per_entry_stamps(
        fresh_mirror, monkeypatch):
    app = _app()

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mc") as c:
            r = await c.post("/api/mc/state",
                             json={"tab": "backlog", "focus": {"kind": "backlog", "id": "7"}})
            assert r.status_code == 200, r.text
            r = await c.get("/api/mc/state")
            route = r.json()
        # The tool reads the same route over HTTP; hand it the ASGI client.
        monkeypatch.setattr(mission_control_ui, "LLOYD_API", "http://mc")
        monkeypatch.setattr(
            mission_control_ui, "make_http_client",
            lambda timeout=None, **kw: httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://mc"))
        tool = await mission_control_ui._mc_get_state({})
        return route, tool

    route, tool = asyncio.run(run())
    for name, payload in (("route", route), ("tool", tool)):
        assert payload["last_changed"], name
        assert "last_updated" not in payload, name
        assert payload["focus_by_tab"]["backlog"]["changed_at"] == payload["last_changed"], name
        assert payload["focus"]["changed_at"] == payload["last_changed"], name
    # The tool's description tells the model what the stamp is, and is not.
    desc = next(t.description for t in asyncio.run(mission_control_ui.list_tools())
                if t.name == "mc_get_state")
    assert "last_changed" in desc and "last_updated" not in desc
    assert "changed_at" in desc


# ── clause 5: the dead accessor is gone ──────────────────────────────


def test_no_focus_snapshot_symbol_remains_in_tracked_python():
    assert not hasattr(mc_state, "get_focus_snapshot")
    out = subprocess.run(
        ["git", "grep", "-l", "focus_snapshot", "--", "*.py"],
        cwd=ROOT, capture_output=True, text=True)
    hits = [h for h in out.stdout.split() if h != "tests/test_mc_state_freshness.py"]
    assert hits == [], hits
