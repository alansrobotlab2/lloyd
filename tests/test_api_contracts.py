"""API contract tests — pin the response shapes web/src/api.ts depends on.

The frontend types in api.ts are hand-maintained against untyped dict
JSONResponses, so a renamed key ships silently and breaks the UI at
click-time. These tests make that class of drift fail in pytest instead.

Uses httpx.ASGITransport with a loopback client address so the mTLS
allowlist middleware's same-host bypass applies. No lifespan is run, so
startup hooks (autonomy ticker, worker pool, watchers) stay off.
"""
import json
import sys
import uuid
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402
from app.routers import sessions as sessions_router  # noqa: E402


@pytest.fixture
def client():
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 9999))
    return httpx.AsyncClient(transport=transport, base_url="http://lloyd-test")


@pytest.fixture
def fixture_session(tmp_path, monkeypatch):
    """One session JSON in an isolated SESSIONS_DIR (the real dir holds
    thousands of files — the list endpoint reads every one)."""
    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    session_id = f"contract-test-{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{session_id}.json").write_text(json.dumps({
        "session_id": session_id,
        "model": "primary",
        "preview": "contract test",
        "platform": "mission-control",
        "messages": [],
        "todos": [{"content": "x", "status": "pending", "activeForm": "x"}],
        "experiment_id": None,
        "inner_voice": False,
    }))
    return session_id


# ── Sessions list — SessionsPage + sidebar ───────────────────────────────────

async def test_sessions_list_shape(client, fixture_session):
    r = await client.get("/api/sessions")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"sessions", "count"}
    entry = next(s for s in body["sessions"] if s["id"] == fixture_session)
    # api.ts Session type — renaming any of these breaks the session list UI
    assert set(entry) >= {
        "id", "session_key", "preview", "last_active", "platform",
        "model", "experiment_id", "inner_voice",
    }


async def test_session_todos_shape(client, fixture_session):
    r = await client.get(f"/api/sessions/{fixture_session}/todos")
    assert r.status_code == 200
    todos = r.json()["todos"]
    assert todos and set(todos[0]) >= {"content", "status"}


async def test_session_plan_shape(client, fixture_session):
    r = await client.get(f"/api/sessions/{fixture_session}/plan")
    assert r.status_code == 200
    assert "plan" in r.json()


async def test_session_queue_shape(client):
    r = await client.get("/api/sessions/no-such-session/queue")
    assert r.status_code == 200
    # api.ts QueueState — consumed by the chat header queue indicator
    assert set(r.json()) >= {"current", "pending_user", "pending_ambient", "depth"}


async def test_missing_session_404s(client, fixture_session):
    r = await client.get("/api/sessions/definitely-not-a-session/todos")
    assert r.status_code == 404


# ── Tools tab ────────────────────────────────────────────────────────────────

async def test_tool_discovery_shape(client):
    r = await client.get("/api/tool-discovery")
    assert r.status_code == 200
    # api.ts ToolDiscovery — Tools page summary chips
    assert set(r.json()) >= {
        "enabled", "threshold_tools", "baseline_tools",
        "max_results_default", "max_results_cap", "total_tools", "active",
    }


async def test_tool_toggle_validates_server(client):
    r = await client.post("/api/tool-toggle", json={
        "type": "tool", "server": "no-such-server", "tool": "Bash", "enabled": False,
    })
    assert r.status_code == 404


async def test_tool_toggle_rejects_unknown_type(client):
    r = await client.post("/api/tool-toggle", json={"type": "bogus", "enabled": True})
    assert r.status_code == 400


# ── Autonomy tab ─────────────────────────────────────────────────────────────

async def test_autonomy_tasks_shape(client):
    r = await client.get("/api/autonomy/tasks")
    assert r.status_code == 200
    body = r.json()
    assert "tasks" in body
    for task in body["tasks"][:3]:
        # api.ts AutonomyTask essentials — AutonomyPage table columns
        assert set(task) >= {"id", "name", "status"}
