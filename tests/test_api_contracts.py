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
