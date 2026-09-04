"""agent_mcp.retrieval + the fact tools' ranking behaviour.

There were no tests here at all, which is how `_graph_rerank`'s god-node
penalty stayed a no-op for four months and `fact_profile` kept returning
5,489 facts for one entity.
"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import facts as facts_mod  # noqa: E402
from agent_mcp import retrieval, vault  # noqa: E402
from app import kg_store  # noqa: E402


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A temp facts tree + store, wired into every module that reads them."""
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    import agent_mcp._shared as shared
    monkeypatch.setattr(shared, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(facts_mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None
    st = kg_store.configure(tmp_path / "kg.sqlite")
    yield facts_root, st
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None


def _write_facts(root, entity, category, facts):
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity, "category": category, "facts": facts}
    (d / f"{entity}-{category}.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n")


# ── _graph_rerank ────────────────────────────────────────────────────────────

def test_graph_rerank_penalty_applies_to_a_cased_hub_entity(world):
    """`voters` is keyed lowercase; the degree map was cased, so every lookup
    missed and every voter was divided by log(1+1+e) — the same constant.
    The penalty existed in the code and did nothing."""
    _, st = world
    # `Lloyd` is a hub (many edges); `Alfie` is specific (one edge).
    for i in range(40):
        st.edges.add({"source": "Lloyd", "target": f"N{i}", "type": "mentions"}, origin="t")
    st.edges.add({"source": "Alfie", "target": "Block", "type": "uses"}, origin="t")

    ci = retrieval.get_entity_edge_counts_ci()
    assert ci["lloyd"] == 40 and ci["alfie"] == 1, ci

    docs = [{"path": "a.md", "snippet": "a note about Lloyd", "score": 1.0},
            {"path": "b.md", "snippet": "a note about Alfie", "score": 1.0}]
    out = vault._graph_rerank(list(docs), ["Lloyd", "Alfie"], [], alpha=0.0)
    # The specific entity's document must outrank the hub's.
    assert [d["path"] for d in out] == ["b.md", "a.md"]
    assert out[0]["_topo_score"] > out[1]["_topo_score"]

    # And with the penalty absent, the two are indistinguishable — which is
    # exactly what the cased-map lookup produced for four months.
    import agent_mcp.vault as v
    orig = v._edge_counts_or_empty
    try:
        v._edge_counts_or_empty = lambda ci=False: {}
        flat = v._graph_rerank(list(docs), ["Lloyd", "Alfie"], [], alpha=0.0)
        assert flat[0]["_topo_score"] == flat[1]["_topo_score"]
    finally:
        v._edge_counts_or_empty = orig


def test_graph_rerank_is_a_noop_without_voters(world):
    docs = [{"path": "a.md", "text": "x", "score": 1.0}]
    assert vault._graph_rerank(docs, [], [], alpha=0.3) is docs


def test_production_defaults_are_what_the_eval_imports():
    """The eval ran rerank off / alpha 0.5 while production ran on / 0.3, so
    a retrieval regression could not appear in the nightly numbers."""
    import eval.run_eval as ev
    assert ev.RECALL_GRAPH_RERANK is vault.RECALL_GRAPH_RERANK
    assert ev.RECALL_RERANK_ALPHA == vault.RECALL_RERANK_ALPHA
    assert ev.RECALL_GRAPH_TOP_K == vault.RECALL_GRAPH_TOP_K
    assert ev.RECALL_GRAPH_HOPS == vault.RECALL_GRAPH_HOPS


# ── extract_entities_from_query ──────────────────────────────────────────────

def test_generic_single_words_do_not_win_a_full_name_match(world):
    """`memory`, `graph`, `session` are real entity dirs. A single generic
    word scored 5.0 — the full-name bonus — and pushed the query's actual
    subject out of the top-k."""
    root, _ = world
    for name in ("memory", "graph", "Knowledge Graph Store"):
        _write_facts(root, name, "state", [{"fact": "x", "id": "stat-001"}])
    ranked = dict(retrieval.extract_entities_from_query(
        "how does the knowledge graph store work in memory"))
    assert "Knowledge Graph Store" in ranked
    assert ranked.get("memory", 0) < 5.0
    assert ranked.get("graph", 0) < 5.0
    assert max(ranked, key=ranked.get) == "Knowledge Graph Store"


def test_task_ids_dispatch_to_the_canonical_form(world):
    root, _ = world
    _write_facts(root, "Task #67", "state", [{"fact": "x", "id": "stat-001"}])
    ranked = dict(retrieval.extract_entities_from_query("what happened with task 67"))
    assert ranked["Task #67"] == 10.0


def test_degree_breaks_ties_deterministically(world):
    root, st = world
    for name in ("Alpha One", "Alpha Two"):
        _write_facts(root, name, "state", [{"fact": "x", "id": "stat-001"}])
    st.edges.add({"source": "Alpha Two", "target": "Hub", "type": "uses"}, origin="t")
    ranked = retrieval.extract_entities_from_query("alpha")
    names = [n for n, _ in ranked]
    assert names.index("Alpha Two") < names.index("Alpha One")


# ── graph_weighted_neighbors ─────────────────────────────────────────────────

def test_weighted_neighbors_decay_by_hop_and_skip_expired(world):
    _, st = world
    st.edges.add({"source": "A", "target": "B", "type": "uses", "confidence": 1.0}, origin="t")
    st.edges.add({"source": "B", "target": "C", "type": "uses", "confidence": 1.0}, origin="t")
    gone = st.edges.add({"source": "A", "target": "Gone", "type": "uses"}, origin="t")
    st.edges.expire(gone, "test")

    one = dict(retrieval.graph_weighted_neighbors(["A"], top_k=5, hops=1))
    assert set(one) == {"B"}
    two = dict(retrieval.graph_weighted_neighbors(["A"], top_k=5, hops=2))
    assert set(two) == {"B", "C"}
    assert two["C"] < two["B"], "second hop must decay"
    assert "Gone" not in two


def test_edge_type_weights_order_neighbours(world):
    _, st = world
    st.edges.add({"source": "A", "target": "Typed", "type": "depends_on", "confidence": 1.0}, origin="t")
    st.edges.add({"source": "A", "target": "Weak", "type": "co_mentioned", "confidence": 1.0}, origin="t")
    ranked = retrieval.graph_weighted_neighbors(["A"], top_k=2)
    assert [n for n, _ in ranked] == ["Typed", "Weak"]


# ── get_facts_sync temporal filters ──────────────────────────────────────────

def test_get_facts_sync_temporal_filters(world):
    root, _ = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "current"},
        {"id": "stat-002", "fact": "expired", "expired_at": "2026-02-01"},
        {"id": "stat-003", "fact": "invalid", "invalid_at": "2026-02-01"},
        {"id": "stat-004", "fact": "future", "valid_at": "2027-01-01"},
    ])
    now = [f["id"] for f in retrieval.get_facts_sync("Lloyd")["facts"]]
    assert now == ["stat-001", "stat-004"]        # valid_at is not a future gate by default
    at = [f["id"] for f in retrieval.get_facts_sync("Lloyd", as_of="2026-01-15")["facts"]]
    assert at == ["stat-001", "stat-002", "stat-003"]
    every = retrieval.get_facts_sync("Lloyd", include_expired=True)["facts"]
    assert len(every) == 4
    assert retrieval.get_facts_sync("Nobody")["facts"] == []


# ── fact_profile ─────────────────────────────────────────────────────────────

def test_fact_profile_caps_a_god_node(world):
    """Uncapped, this returned every fact an entity had — 5,489 for `Lloyd`
    — straight into the model's context."""
    root, _ = world
    _write_facts(root, "Lloyd", "state",
                 [{"id": f"stat-{i:03d}", "fact": f"fact number {i}", "category": "state"}
                  for i in range(1, 61)])
    out = facts_mod._fact_profile({"entity": "Lloyd"})
    assert out["fact_count"] == 60
    assert len(out["categories"]["state"]) == retrieval.FACT_RANK_CAP_SEED
    assert out["truncated_categories"] == {"state": 60}
    assert "hint" in out and "showing" in out["summary"]


def test_fact_profile_ranks_by_query_when_given(world):
    root, _ = world
    _write_facts(root, "Lloyd", "state",
                 [{"id": f"stat-{i:03d}", "fact": f"filler {i}", "category": "state"}
                  for i in range(1, 40)]
                 + [{"id": "stat-099", "category": "state",
                     "fact": "Lloyd serves models through vLLM"}])
    out = facts_mod._fact_profile({"entity": "Lloyd", "query": "vLLM serving"})
    assert out["categories"]["state"][0]["id"] == "stat-099"


# ── fact_resolve ─────────────────────────────────────────────────────────────

def test_fact_resolve_reports_by_default(world):
    """It defaulted to auto_resolve=True, so a call that reads like a query
    silently expired facts."""
    root, _ = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "the feature is enabled", "confidence": 0.9},
        {"id": "stat-002", "fact": "the feature is disabled", "confidence": 0.5},
    ])
    out = facts_mod._fact_resolve({"entity": "Lloyd"})
    assert out["resolved"] == 0 and out["remaining"] >= 1
    fm = yaml.safe_load((root / "Lloyd" / "Lloyd-state.md").read_text().split("---")[1])
    assert all(not f.get("invalid_at") and not f.get("expired_at") for f in fm["facts"])


def test_fact_resolve_sets_invalid_at_only(world):
    root, _ = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "the feature is enabled", "confidence": 0.9},
        {"id": "stat-002", "fact": "the feature is disabled", "confidence": 0.5},
    ])
    out = facts_mod._fact_resolve({"entity": "Lloyd", "auto_resolve": True})
    assert out["resolved"] == 1
    fm = yaml.safe_load((root / "Lloyd" / "Lloyd-state.md").read_text().split("---")[1])
    by_id = {f["id"]: f for f in fm["facts"]}
    assert by_id["stat-002"]["invalid_at"] and not by_id["stat-002"].get("expired_at")
    assert not by_id["stat-001"].get("invalid_at")


def test_fact_resolve_leaves_equal_confidence_alone(world):
    root, _ = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "the feature is enabled", "confidence": 0.9},
        {"id": "stat-002", "fact": "the feature is disabled", "confidence": 0.9},
    ])
    assert facts_mod._fact_resolve({"entity": "Lloyd", "auto_resolve": True})["resolved"] == 0


def test_the_contradiction_scan_is_refused_on_a_god_node(world):
    """O(n squared) over 5,489 facts is 15 million comparisons — measured at
    113 seconds through MCP, and it reported 32,857 `contradictions` that
    were almost all the overlap heuristic firing on similar phrasing.
    Refused in BOTH modes: the report path ran the scan too."""
    root, _ = world
    _write_facts(root, "Lloyd", "state",
                 [{"id": f"stat-{i:03d}", "fact": f"fact {i} is enabled",
                   "confidence": 0.9, "category": "state"}
                  for i in range(1, retrieval.FACT_GODNODE_THRESHOLD + 5)])
    for params in ({"entity": "Lloyd", "auto_resolve": True}, {"entity": "Lloyd"}):
        out = facts_mod._fact_resolve(params)
        assert "error" in out and "refused" in out["error"], params
    check = facts_mod._fact_check({"entity": "Lloyd"})
    assert "error" in check and "refused" in check["error"]
    # a narrower slice is still scannable
    _write_facts(root, "Lloyd", "goal", [
        {"id": "goal-001", "fact": "the goal is enabled", "confidence": 0.9, "category": "goal"},
        {"id": "goal-002", "fact": "the goal is disabled", "confidence": 0.5, "category": "goal"},
    ])
    retrieval.invalidate_fact_file_cache()
    narrow = facts_mod._fact_check({"entity": "Lloyd", "category": "goal"})
    assert "error" not in narrow and narrow["checked"] == 2


# ── entity kind ──────────────────────────────────────────────────────────────

def test_system_names_are_not_typed_as_people():
    """`Knowledge Graph`, `Claude Code` and `Isaac Lab` all satisfy the
    `First Last` shape. The v4 classifier's PERSON test ran before its SYSTEM
    test, so all three were typed PERSON — and the type gates
    ROLE_BLOCKED_VERBS, which is how `created_by` between two systems passed.
    """
    from app.entity_kind import derive_entity_type, derive_kind
    for name in ("Knowledge Graph", "Claude Code", "Isaac Lab", "Intel Pipeline",
                 "QMD Daemon", "Morning Briefing System"):
        assert derive_kind(name) == "system", name
        assert derive_entity_type(name) == "SYSTEM", name


def test_person_needs_corroboration(monkeypatch):
    from app import entity_kind
    monkeypatch.setattr(entity_kind, "_people_cache", {"jane doe"})
    assert entity_kind.derive_kind("Jane Doe") == "person"
    # same shape, no note and no people/ source -> not a person
    assert entity_kind.derive_kind("Isaac Lab") == "system"
    # the source document settles it either way
    assert entity_kind.derive_kind("Someone New", "people/someone-new.md") == "person"
    assert entity_kind.derive_kind("Jane Doe", "projects/x.md") == "project"


def test_kind_recognises_tasks_docs_and_skills():
    from app.entity_kind import derive_kind
    assert derive_kind("Task #67") == "task"
    assert derive_kind("Autonomy Task #33") == "task"
    assert derive_kind("backlog_item_235") == "task"
    assert derive_kind("server.py") == "doc"
    assert derive_kind("knowledge/ai/note.md") == "doc"
    assert derive_kind("nightly-reflection") == "skill"
    assert derive_kind("vLLM") == "system"


# ── every fact write updates the index ───────────────────────────────────────

def test_fact_invalidate_updates_the_index(world):
    """The markdown is the fact layer, but facts_idx is what the router and
    fact_profile read. fact_invalidate wrote expired_at to the file and left
    the index alone, so the Memory page went on showing the fact as current."""
    root, st = world
    _write_facts(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "Lloyd runs on Ollama", "confidence": 0.9, "category": "state"},
        {"id": "stat-002", "fact": "Lloyd runs on vLLM", "confidence": 0.9, "category": "state"},
    ])
    st.facts_idx.reindex(root=root)
    assert len(st.facts_idx.for_entity("Lloyd")) == 2

    out = facts_mod._fact_invalidate({
        "entity": "Lloyd", "ended": "2026-09-04", "fact_substring": "ollama",
        "reason": "moved to vLLM"})
    assert out["expired_count"] == 1

    live = st.facts_idx.for_entity("Lloyd")
    assert [f["fact_id"] for f in live] == ["stat-002"], "the index still serves the expired fact"
    everything = st.facts_idx.for_entity("Lloyd", include_expired=True)
    assert len(everything) == 2
    expired = next(f for f in everything if f["fact_id"] == "stat-001")
    assert expired["expired_at"] == "2026-09-04"


def test_fact_add_and_resolve_also_update_the_index(world):
    root, st = world
    add = facts_mod._fact_add({"entity": "Newthing", "category": "state", "fact": "it exists"})
    assert add["success"]
    assert [f["fact"] for f in st.facts_idx.for_entity("Newthing")] == ["it exists"]

    _write_facts(root, "Pair", "state", [
        {"id": "stat-001", "fact": "the flag is enabled", "confidence": 0.9, "category": "state"},
        {"id": "stat-002", "fact": "the flag is disabled", "confidence": 0.4, "category": "state"},
    ])
    st.facts_idx.reindex(root=root)
    retrieval.invalidate_fact_file_cache()
    assert facts_mod._fact_resolve({"entity": "Pair", "auto_resolve": True})["resolved"] == 1
    assert [f["fact_id"] for f in st.facts_idx.for_entity("Pair")] == ["stat-001"]
