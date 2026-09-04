"""Readers and writers of the edge graph, now that it lives in app.kg_store.

Replaces test_relationships_cache.py: the mtime-keyed JSON cache it pinned is
gone (the store memoises on `PRAGMA data_version`, covered in
test_kg_store.py). What still needs pinning is the boundary this file now
tests — that an unreadable store never reads as an empty graph to a writer,
and that read-only ranking paths degrade instead of failing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import facts as facts_mod  # noqa: E402
from agent_mcp import retrieval  # noqa: E402
from app import kg_store  # noqa: E402
from app.kg_store import StoreUnavailable  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the process-default store at a temp file, and the facts tree with
    it so entity resolution does not reach into the real corpus."""
    facts = tmp_path / "facts"
    facts.mkdir()
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts)
    import agent_mcp._shared as shared
    monkeypatch.setattr(shared, "FACTS_ROOT", facts)
    shared._invalidate_entity_dirs_cache()
    s = kg_store.configure(tmp_path / "kg.sqlite")
    yield s
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()


@pytest.fixture
def broken_store(tmp_path, monkeypatch):
    p = tmp_path / "kg.sqlite"
    p.write_bytes(b"not a database file, just 64 bytes of garbage ..............")
    monkeypatch.setattr(kg_store, "_default_path", p)
    kg_store.reset()
    yield p
    kg_store.reset()


def _edge(s, t, typ="uses", **kw):
    return {"source": s, "target": t, "type": typ, "confidence": 0.9, "provenance": "STATED", **kw}


# ── read view ────────────────────────────────────────────────────────────────

def test_load_relationships_returns_the_legacy_shape(store):
    assert retrieval.load_relationships() == {"edges": [], "schema_version": 1}
    store.edges.add(_edge("A", "B"), origin="test")
    data = retrieval.load_relationships()
    assert len(data["edges"]) == 1
    assert data["edges"][0]["source"] == "A"
    assert "id" in data["edges"][0] and "origin" in data["edges"][0]


def test_load_relationships_includes_expired_edges(store):
    i = store.edges.add(_edge("A", "B"), origin="test")
    store.edges.expire(i, "test")
    edges = retrieval.load_relationships()["edges"]
    assert len(edges) == 1 and edges[0]["expired_at"]


def test_degree_and_ci_degree(store):
    store.edges.add(_edge("vLLM", "Ray"), origin="test")
    store.edges.add(_edge("vllm", "Ray"), origin="test")
    assert retrieval.get_entity_edge_counts() == {"vLLM": 1, "vllm": 1, "Ray": 2}
    # `vault._graph_rerank` looks up lowercased names; the cased map missed
    # every one, so its god-node penalty was a constant divide (2026-09-03).
    assert retrieval.get_entity_edge_counts_ci() == {"vllm": 2, "ray": 2}


def test_graph_expansion_walks_both_directions(store):
    store.edges.add(_edge("A", "B"), origin="test")
    store.edges.add(_edge("C", "B"), origin="test")
    assert sorted(retrieval.graph_expand_entities(["A"], hops=1)) == ["B"]
    assert sorted(retrieval.graph_expand_entities(["A"], hops=2)) == ["B", "C"]


def test_weighted_neighbors_prefer_typed_edges(store):
    store.edges.add(_edge("A", "Typed", "uses"), origin="test")
    store.edges.add(_edge("A", "Weak", "mentions"), origin="test")
    ranked = retrieval.graph_weighted_neighbors(["A"], top_k=2)
    assert [n for n, _ in ranked] == ["Typed", "Weak"]
    assert ranked[0][1] > ranked[1][1]


def test_expired_edges_are_not_traversed(store):
    i = store.edges.add(_edge("A", "B"), origin="test")
    store.edges.expire(i, "test")
    assert retrieval.graph_expand_entities(["A"]) == []
    assert retrieval.graph_weighted_neighbors(["A"]) == []
    assert retrieval.get_entity_edge_counts() == {}


# ── fail closed ──────────────────────────────────────────────────────────────

def test_an_unreadable_store_raises_rather_than_reading_as_empty(broken_store):
    """The empty graph is a legitimate value. `I cannot read the graph` is not
    the same value, and conflating them is how a writer persists one edge over
    thousands."""
    assert retrieval.RelationshipsCorrupt is StoreUnavailable
    with pytest.raises(StoreUnavailable):
        retrieval.get_entity_edge_counts()
    with pytest.raises(StoreUnavailable):
        retrieval.load_relationships()


def test_ranking_paths_degrade_on_an_unreadable_store(broken_store):
    """Degree is a tie-break and a god-node penalty, never a stored value —
    a corrupt store costs ranking quality, not a failed recall."""
    assert retrieval._edge_counts_or_empty() == {}
    assert retrieval._edge_counts_or_empty(ci=True) == {}
    assert retrieval.graph_expand_entities(["A"]) == []
    assert retrieval.graph_weighted_neighbors(["A"]) == []


def test_fact_relate_refuses_to_write_to_an_unreadable_store(broken_store):
    result = facts_mod._fact_relate({"source": "Lloyd", "target": "vLLM", "type": "uses"})
    assert "error" in result
    assert "unreadable" in result["error"]


def test_fact_relate_writes_an_edge_with_its_origin(store):
    r = facts_mod._fact_relate({"source": "Lloyd", "target": "vLLM", "type": "uses",
                                "source_doc": "notes/x.md"})
    assert r["action"] == "created"
    edge = store.edges.by_id(r["edge_id"])
    assert edge["origin"] == "fact_relate" and edge["source_doc"] == "notes/x.md"
    again = facts_mod._fact_relate({"source": "Lloyd", "target": "vLLM", "type": "uses"})
    assert again["action"] == "already_exists" and again["edge_id"] == r["edge_id"]
    assert store.edges.count() == 1


def test_fact_relate_refuses_a_self_loop(store):
    r = facts_mod._fact_relate({"source": "Lloyd", "target": "Lloyd", "type": "uses"})
    assert "error" in r and "resolve to" in r["error"]
    assert store.edges.count(active_only=False) == 0


# ── handlers read through the store ──────────────────────────────────────────

def test_fact_relationships_direction_and_type_filters(store):
    store.edges.add(_edge("A", "B", "uses"), origin="test")
    store.edges.add(_edge("C", "A", "part_of"), origin="test")
    out = facts_mod._fact_relationships({"entity": "A"})
    assert out["count"] == 2
    assert facts_mod._fact_relationships({"entity": "A", "direction": "out"})["count"] == 1
    assert facts_mod._fact_relationships({"entity": "A", "direction": "in"})["count"] == 1
    assert facts_mod._fact_relationships({"entity": "A", "type": "uses"})["count"] == 1


def test_fact_path_finds_a_route_and_reports_a_miss(store):
    store.edges.add(_edge("A", "B"), origin="test")
    store.edges.add(_edge("B", "C"), origin="test")
    hit = facts_mod._fact_path({"source": "A", "target": "C"})
    assert hit["found"] and hit["path"] == ["A", "B", "C"] and hit["hops"] == 2
    assert facts_mod._fact_path({"source": "A", "target": "Nowhere"})["found"] is False


def test_fact_neighbors_honours_min_confidence_and_caps(store):
    store.edges.add(_edge("A", "Strong", confidence=0.9), origin="test")
    store.edges.add(_edge("A", "Weak", confidence=0.2), origin="test")
    out = facts_mod._fact_neighbors({"entity": "A", "min_confidence": 0.5})
    assert out["nodes"] == ["A", "Strong"]
    capped = facts_mod._fact_neighbors({"entity": "A", "min_confidence": 0.0, "max_edges": 1})
    assert capped["truncated"] is True and "hint" in capped


# ── conversation relation linking (#51) ──────────────────────────────────────

def test_conversation_relations_dedupe_reads_the_key_the_index_writes(tmp_path, monkeypatch):
    """The index writes `relationships`; this read `edges`, so the dedupe was
    a no-op against 486,961 rows (2026-09-03 review)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "conversation_relations", ROOT / "scripts/memory/conversation_relations.py")
    cr = importlib.util.module_from_spec(spec)
    sys.modules["conversation_relations"] = cr
    spec.loader.exec_module(cr)

    import json as _json
    idx = tmp_path / "relations-index.json"
    idx.write_text(_json.dumps({"relationships": [{"source": "a.md", "target": "b.md"}]}))
    monkeypatch.setattr(cr, "RELATIONS_INDEX", idx)

    props = [{"source": "a.md", "target": "b.md"}, {"source": "b.md", "target": "a.md"},
             {"source": "c.md", "target": "d.md"}]
    assert cr.deduplicate_against_index(props) == [{"source": "c.md", "target": "d.md"}]


def test_approved_conversation_links_land_as_edges(store):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "conversation_relations2", ROOT / "scripts/memory/conversation_relations.py")
    cr = importlib.util.module_from_spec(spec)
    sys.modules["conversation_relations2"] = cr
    spec.loader.exec_module(cr)

    props = [
        {"source": "a.md", "target": "b.md", "status": "approved", "confidence": 0.9,
         "evidence_trajectory": "traj/2026-09-01.jsonl", "reason": "co-read twice"},
        {"source": "c.md", "target": "d.md", "status": "pending", "confidence": 0.95},
    ]
    assert cr.land_approved_edges(props) == 1
    edge = store.edges.find_active("a.md", "b.md", "co_accessed")
    assert edge["provenance"] == "INFERRED" and edge["origin"] == "conversation"
    assert edge["source_doc"] == "traj/2026-09-01.jsonl" and edge["evidence"] == "co-read twice"
    assert props[0]["edge_id"] == edge["id"]
    # idempotent: a second pass lands nothing new
    assert cr.land_approved_edges(props) == 0
    assert store.edges.count() == 1
