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


@pytest.fixture
def fixture_background_sessions(tmp_path, monkeypatch):
    """One autonomy run and one worker run, named the way the recorder names
    them — four underscore-separated parts, which is the fast path the
    listings read before opening anything."""
    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    ids = []
    for sid, platform, source in (
            ("20260910_120001_autonomy_ab12", "autonomy", "autonomy-task:80"),
            ("20260910_120002_autocode_cd34", "worker", "autocode")):
        (tmp_path / f"{sid}.json").write_text(json.dumps({
            "session_id": sid, "title": f"{source} run", "preview": "p",
            "model": "primary", "platform": platform, "source": source,
            "inner_voice": False, "messages": [],
            "last_active": "2026-09-10T12:00:00",
        }))
        ids.append(sid)
    # A chat, to prove the two listings really are complements.
    (tmp_path / "20260910_120003_ef5678.json").write_text(json.dumps({
        "session_id": "20260910_120003_ef5678", "platform": "mission-control",
        "messages": [], "preview": "a real conversation"}))
    return ids


async def test_background_sessions_shape(client, fixture_background_sessions):
    r = await client.get("/api/background/sessions")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"sessions", "count"}
    got = {s["id"] for s in body["sessions"]}
    assert got == set(fixture_background_sessions)
    entry = body["sessions"][0]
    # api.ts BackgroundSession type.
    assert set(entry) >= {
        "id", "session_key", "title", "preview", "platform", "source",
        "model", "inner_voice", "message_count", "last_active",
    }


async def test_the_two_listings_are_complements(client,
                                                fixture_background_sessions):
    """Chat history is conversations and the Background tab is everything
    else. A session in both would be the bug this split exists to fix, in the
    other direction."""
    chats = {s["id"] for s in (await client.get("/api/sessions")).json()["sessions"]}
    background = {s["id"] for s in
                  (await client.get("/api/background/sessions")).json()["sessions"]}
    assert chats == {"20260910_120003_ef5678"}
    assert not (chats & background)


async def test_workers_health_shape(client):
    r = await client.get("/api/workers/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"initialized", "days", "sources"}
    for src in body["sources"]:
        # api.ts WorkerSourceHealth type.
        assert set(src) >= {"name", "enabled", "inner_voice", "depth",
                            "health", "recent"}
        # `health` is None for a source with no runs in the window — a rate
        # over zero runs is unknown, not 0%.
        assert src["health"] is None or set(src["health"]) >= {
            "total", "ok", "failed", "fail_rate", "gpu_hours"}


async def test_created_sessions_are_never_background_shaped(client, tmp_path,
                                                          monkeypatch):
    """`POST /api/sessions/create` is how the Inner Voice "+ new chat" button and the
    right-hand chat sidebar create a user session. `/api/sessions` skips a
    four-part id without opening it, so a session minted here in that shape
    would never appear in the history — and nothing would say so."""
    from app.sessions_io import is_background_session_name

    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    r = await client.post("/api/sessions/create", json={
        "inner_voice": True, "inner_voice_evaluate_user_turns": True})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert not is_background_session_name(sid)
    listed = {s["id"] for s in (await client.get("/api/sessions")).json()["sessions"]}
    assert sid in listed


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


async def test_autonomy_blocked_agrees_with_dispatch_at_one_instant(
        client, tmp_path, monkeypatch):
    """#870 clause 6, measured AT the endpoint and not only at the function.

    `blocked` is the board's answer to "why is this not running"; the queue is
    the scheduler's. They were computed from two different `depends_on`
    resolution sets, so the autonomy tab could show `waiting on #1` on a task the
    ticker was dispatching in the same second. This crosses the real seam — an
    ASGI request through the router, which reads the directory and imports
    autonomy inside the handler — with the clock pinned to one instant on both
    sides, and asserts the two answers cannot part company for ANY task listed.
    """
    import datetime as dt

    import autonomy
    from app.routers import autonomy as autonomy_router

    when = dt.datetime(2026, 9, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(autonomy, "_utcnow", lambda: when, raising=False)
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path)
    monkeypatch.setattr(autonomy_router, "_AUTONOMY_DIR", tmp_path)

    def write(tid, **over):
        fm = {"id": tid, "name": f"task{tid}", "status": "up_next",
              "frequency": "daily", "priority": "medium", "skill_name": "some-skill",
              "timeout_seconds": 600, "max_retries": 3, "failure_count": 0}
        fm.update(over)
        head = "\n".join(f"{k}: {v}" for k, v in fm.items())
        (tmp_path / f"{tid}-task{tid}.md").write_text(f"---\n{head}\n---\n\nb\n")

    # Every task here is PAST its own interval before the biconditional is
    # asserted, on purpose: `hold_reason` answers "is anything holding this", not
    # "is its interval up", so a task comfortably inside its own window has no
    # hold reason and does not dispatch either. Asking one instant to settle both
    # questions is fine; asking `blocked is None` to mean "dispatching now" is
    # only sound once no task is sitting in its window.
    stale = (when - dt.timedelta(days=3)).isoformat()
    write(1, status="paused", last_run=(when - dt.timedelta(days=2)).isoformat())
    write(2, depends_on=1, last_run=stale)          # held by #870's fix
    # 3 runs every 6 h and last ran 8 h ago: due on its own clock, and fresh
    # enough (under task 4's 12 h half-interval bound) to satisfy 4.
    write(3, frequency="every_6_hours",
          last_run=(when - dt.timedelta(hours=8)).isoformat())
    write(4, depends_on=3, last_run=stale)          # upstream met → dispatches
    write(5, depends_on=99, last_run=stale)         # id with no file → unresolved

    r = await client.get("/api/autonomy/tasks")
    assert r.status_code == 200
    listed = {t["id"]: t for t in r.json()["tasks"]}
    due = {t["id"] for t in autonomy.get_due_tasks()}
    assert set(listed) == {1, 2, 3, 4, 5}
    # The contradiction this item exists to kill, asserted unconditionally: no
    # task is reported as held by the board and dispatched by the scheduler at
    # the same instant.
    for tid, task in listed.items():
        assert not (task["blocked"] is not None and tid in due), (
            f"task {tid}: board said {task['blocked']!r} while "
            f"get_due_tasks() dispatched it")
    # And with every task past its interval, the two answers coincide outright.
    for tid, task in listed.items():
        assert (task["blocked"] is None) == (tid in due), (
            f"task {tid}: board said {task['blocked']!r}, "
            f"get_due_tasks says dispatched={tid in due}")
    # Both branches taken, so neither loop above is green by everything landing
    # on one side: a dependency-held dependent AND a dispatched dependent.
    assert listed[2]["blocked"] == "waiting on #1" and 2 not in due
    assert listed[4]["blocked"] is None and 4 in due


# ── Memory page: entities, entity detail, entity graph ───────────────────────

@pytest.fixture
def entity_world(tmp_path, monkeypatch):
    """A temp facts tree + store behind the entity endpoints."""
    import yaml
    from app import kg_store
    from app.routers import entities as entities_router

    facts_root = tmp_path / "facts"
    for name in ("Lloyd", "vLLM"):
        d = facts_root / name
        d.mkdir(parents=True)
        (d / f"{name}-overview.md").write_text(
            f"---\ntype: overview\nentity: {name}\ndefinition: {name} is a thing.\n---\n\n# Summary\n\nProse about {name}.\n")
    fm = {"type": "facts", "entity": "Lloyd", "category": "state", "facts": [
        {"id": "stat-001", "fact": "Lloyd runs on vLLM", "confidence": 0.9,
         "created_at": "2026-01-01T00:00:00+00:00", "source_doc": "knowledge/lloyd.md",
         "provenance": "EXTRACTED"},
        {"id": "stat-002", "fact": "Lloyd used Ollama", "confidence": 0.8,
         "created_at": "2025-06-01T00:00:00+00:00", "expired_at": "2026-02-01T00:00:00+00:00",
         "provenance": "EXTRACTED"},
    ]}
    (facts_root / "Lloyd" / "Lloyd-state.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# Lloyd - state\n")

    monkeypatch.setattr(entities_router, "_FACTS_ROOT", facts_root)
    st = kg_store.configure(tmp_path / "kg.sqlite")
    st.facts_idx.reindex(root=facts_root)
    st.entities.register("vLLM")
    st.entities.register("Orphan Concept")     # registered, no edges, no facts
    st.entities.backfill_kinds(overwrite=True)
    st.aliases.set("lloyd-mc", "Lloyd", kind="punct", origin="test")
    st.edges.add({"source": "Lloyd", "target": "vLLM", "type": "uses",
                  "confidence": 0.95, "provenance": "STATED",
                  "evidence": "Lloyd runs on vLLM"}, origin="test")
    st.edges.add({"source": "vLLM", "target": "Lloyd", "type": "part_of",
                  "confidence": 0.7, "provenance": "STATED"}, origin="test")
    st.edges.add({"source": "Lloyd", "target": "Noise", "type": "mentions",
                  "confidence": 0.8}, origin="test")
    entities_router._ENTITIES_CACHE.clear()
    entities_router._GRAPH_CACHE.clear()
    yield st
    kg_store.reset()
    entities_router._ENTITIES_CACHE.clear()
    entities_router._GRAPH_CACHE.clear()


async def test_entities_list_shape_and_limit(client, entity_world):
    r = await client.get("/api/entities?limit=1")
    assert r.status_code == 200
    body = r.json()
    # api.ts EntitiesListData
    assert set(body) >= {"entities", "total", "offset", "limit", "returned", "query"}
    assert body["returned"] == 1 and body["total"] == 3
    e = body["entities"][0]
    assert set(e) >= {"name", "factCount", "kind", "categories"}
    # Most-populated first: an alphabetical list put `#160` at the top of a
    # 23,564-row sidebar.
    assert e["name"] == "Lloyd" and e["factCount"] == 1


async def test_entities_list_query_filter(client, entity_world):
    r = await client.get("/api/entities?q=vll")
    assert r.status_code == 200
    assert [e["name"] for e in r.json()["entities"]] == ["vLLM"]
    assert (await client.get("/api/entities?q=nothingmatches")).json()["total"] == 0


async def test_entity_detail_filters_expired_by_default(client, entity_world):
    r = await client.get("/api/entity?name=Lloyd")
    body = r.json()
    assert set(body) >= {"name", "kind", "facts", "factCount", "relationships",
                         "outbound", "inbound", "aliases", "definition", "summary",
                         "includeExpired"}
    assert [f["id"] for f in body["facts"]] == ["stat-001"]
    assert body["includeExpired"] is False

    r2 = await client.get("/api/entity?name=Lloyd&include_expired=1")
    body2 = r2.json()
    assert sorted(f["id"] for f in body2["facts"]) == ["stat-001", "stat-002"]
    assert body2["includeExpired"] is True


async def test_entity_detail_splits_direction_and_names_the_other_end(client, entity_world):
    body = (await client.get("/api/entity?name=Lloyd")).json()
    assert [r["type"] for r in body["outbound"]] == ["uses"]
    assert [r["type"] for r in body["inbound"]] == ["part_of"]
    # `other` is the endpoint that is not the viewed entity — the UI printed
    # `target` for both directions, so inbound edges showed Lloyd's own name.
    assert body["outbound"][0]["other"] == "vLLM"
    assert body["inbound"][0]["other"] == "vLLM"
    assert body["outbound"][0]["evidence"] == "Lloyd runs on vLLM"
    # mentions edges are co-occurrence noise, not a stated relationship
    assert all(r["type"] != "mentions" for r in body["relationships"])


async def test_entity_detail_carries_aliases_and_a_resolved_name(client, entity_world):
    body = (await client.get("/api/entity?name=lloyd-mc")).json()
    assert body["name"] == "Lloyd"
    assert [a["surface"] for a in body["aliases"]] == ["lloyd-mc"]


async def test_entity_graph_excludes_isolated_nodes_by_default(client, entity_world):
    body = (await client.get("/api/entity-graph")).json()
    assert set(body) >= {"nodes", "edges", "nodeCount", "edgeCount",
                         "includeIsolated", "minConfidence"}
    # `Noise` only has a mentions edge, which the graph drops.
    assert sorted(n["id"] for n in body["nodes"]) == ["Lloyd", "vLLM"]
    node = next(n for n in body["nodes"] if n["id"] == "Lloyd")
    # api.ts EntityGraphNode. `type` is the registry kind, not a fact category.
    assert set(node) >= {"id", "label", "type", "factCount", "definition"}
    assert node["type"] in ("system", "project", "concept", "person", "skill",
                            "task", "doc", "entity")
    assert node["definition"] is None, "definitions are lazy; 23,565 file opens per build"


async def test_entity_graph_collapses_a_pair_and_marks_it_bidirectional(client, entity_world):
    body = (await client.get("/api/entity-graph")).json()
    assert body["edgeCount"] == 1
    edge = body["edges"][0]
    assert edge["bidirectional"] is True, "both directions genuinely exist"
    assert edge["type"] == "uses" and edge["weight"] == 0.95   # dominant direction
    assert set(edge) >= {"source", "target", "type", "weight", "bidirectional",
                         "provenance", "created_at"}


async def test_entity_graph_include_isolated_and_min_confidence(client, entity_world):
    """Isolated entities are off by default: including the 20,000 that have no
    edges made a 5.7 MB payload the browser then had to lay out."""
    default = (await client.get("/api/entity-graph")).json()
    assert "Orphan Concept" not in {n["id"] for n in default["nodes"]}
    body = (await client.get("/api/entity-graph?include_isolated=1")).json()
    assert "Orphan Concept" in {n["id"] for n in body["nodes"]}
    strict = (await client.get("/api/entity-graph?min_confidence=0.99")).json()
    assert strict["edgeCount"] == 0 and strict["nodeCount"] == 0
