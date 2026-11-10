"""`fact_relationships` returns one ranked, trimmed page (2026-11-10 review).

It used to return every active edge with every column: `Lloyd` had 1,363
(about 840 KB), which the harness spilled to a file, so the model got a
pointer instead of an answer for exactly the hub entities it asked about most.
And its `type` filter was not canonicalised, so `depends-on` matched nothing
and read as "no such edges".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_mcp._shared as shared            # noqa: E402
from agent_mcp import facts as facts_mod      # noqa: E402
from agent_mcp import retrieval               # noqa: E402
from app import kg_store                      # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    facts = tmp_path / "facts"
    facts.mkdir()
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts)
    monkeypatch.setattr(shared, "FACTS_ROOT", facts)
    shared._invalidate_entity_dirs_cache()
    s = kg_store.configure(tmp_path / "kg.sqlite")
    yield s
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()


def _edge(s, t, typ="uses", conf=0.9, **kw):
    return {"source": s, "target": t, "type": typ, "confidence": conf,
            "provenance": "STATED", **kw}


def test_hub_entity_gets_a_ranked_capped_page(store):
    for i in range(60):
        store.edges.add(_edge("Hub", f"T{i:02d}", conf=0.5,
                              evidence="x" * 500,
                              classifier_meta='{"big": "blob"}'), origin="test")
    store.edges.add(_edge("Hub", "Strong", typ="depends_on", conf=0.99), origin="test")
    out = facts_mod._fact_relationships({"entity": "Hub"})
    assert out["count"] == 61
    assert out["returned"] == facts_mod.RELATIONSHIPS_DEFAULT_LIMIT
    assert out["truncated"] is True
    assert out["by_type"] == {"uses": 60, "depends_on": 1}
    assert out["edges"][0]["target"] == "Strong"
    e = out["edges"][1]
    assert "classifier_meta" not in e and "expired_at" not in e
    assert len(e["evidence"]) <= facts_mod.RELATIONSHIPS_EVIDENCE_CHARS + 1


def test_limit_is_clamped(store):
    for i in range(5):
        store.edges.add(_edge("Hub", f"T{i}"), origin="test")
    assert facts_mod._fact_relationships({"entity": "Hub", "limit": 2})["returned"] == 2
    assert facts_mod._fact_relationships({"entity": "Hub", "limit": 0})["returned"] == 1
    out = facts_mod._fact_relationships({"entity": "Hub", "limit": 10_000})
    assert out["returned"] == 5 and out["truncated"] is False


def test_type_filter_is_canonicalised(store):
    store.edges.add(_edge("A", "B", typ="depends_on"), origin="test")
    store.edges.add(_edge("A", "C", typ="uses"), origin="test")
    out = facts_mod._fact_relationships({"entity": "A", "type": "depends-on"})
    assert out["count"] == 1 and out["edges"][0]["target"] == "B"
    assert out["type"] == "depends_on"


def test_an_unknown_type_is_refused_not_an_empty_list(store):
    store.edges.add(_edge("A", "B"), origin="test")
    out = facts_mod._fact_relationships({"entity": "A", "type": "frobnicates"})
    assert "error" in out and "uses" in out["error"]
