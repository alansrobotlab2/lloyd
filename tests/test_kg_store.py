"""app.kg_store — the one write path for edges, aliases, entities, fact index.

Pins the properties the JSON files could not offer: a crash or a race cannot
lose an edge, history survives a retype or a merge, a merge is exactly
revertable, caches invalidate on cross-process commits.
"""
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import kg_store  # noqa: E402
from app.kg_store import KGStore, StoreUnavailable  # noqa: E402


@pytest.fixture
def db(tmp_path) -> KGStore:
    s = KGStore(tmp_path / "kg.sqlite")
    yield s
    s.close()


def _edge(s, t, typ="uses", **kw):
    return {"source": s, "target": t, "type": typ, "confidence": 0.9,
            "provenance": "STATED", **kw}


# ── schema / basics ──────────────────────────────────────────────────────────

def test_schema_init_is_idempotent_and_wal(tmp_path):
    p = tmp_path / "kg.sqlite"
    a = KGStore(p); a.close()
    b = KGStore(p)
    assert b.meta_get("schema_version") == "1"
    assert b.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    b.close()


def test_add_dedupes_against_active_unique_index(db):
    i1 = db.edges.add(_edge("A", "B"), origin="test")
    i2 = db.edges.add(_edge("A", "B"), origin="test")
    assert i1 == i2
    assert db.edges.count() == 1
    # a different type is a different edge
    i3 = db.edges.add(_edge("A", "B", "part_of"), origin="test")
    assert i3 != i1 and db.edges.count() == 2
    # once expired, the key is free again
    db.edges.expire(i1, "test")
    i4 = db.edges.add(_edge("A", "B"), origin="test")
    assert i4 != i1 and db.edges.count() == 2 and db.edges.count(active_only=False) == 3


def test_add_refuses_self_loops_and_blanks(db):
    with pytest.raises(ValueError):
        db.edges.add(_edge("A", "A"))
    with pytest.raises(ValueError):
        db.edges.add({"source": "A", "target": "", "type": "uses"})


def test_every_edge_carries_origin_and_created_at(db):
    i = db.edges.add(_edge("A", "B"), origin="fact_relate")
    e = db.edges.by_id(i)
    assert e["origin"] == "fact_relate" and e["created_at"]


def test_retype_keeps_history_and_collapses_the_pair(db):
    m = db.edges.add(_edge("A", "B", "mentions", provenance="EXTRACTED"), origin="seed")
    stray = db.edges.add(_edge("A", "B", "related_to"), origin="test")   # second active type on the pair
    new = db.edges.retype(m, {"type": "uses", "confidence": 0.95, "provenance": "EXTRACTED_CLASSIFIER_V4"},
                          origin="classifier", reason="v4")
    assert db.edges.by_id(m)["expired_at"] and db.edges.by_id(m)["expired_reason"] == "v4"
    assert db.edges.by_id(stray)["expired_at"]                          # pair has one typed relation
    e = db.edges.by_id(new)
    assert e["type"] == "uses" and e["superseded_edge_id"] == m and e["expired_at"] is None
    assert db.edges.active(source="A", target="B") == [e]


def test_rewrite_endpoint_returns_pairs_and_revert_is_exact(db):
    a = db.edges.add(_edge("vllm", "Ray"), origin="test")
    b = db.edges.add(_edge("Lloyd", "vllm", "depends_on"), origin="test")
    c = db.edges.add(_edge("vLLM", "Ray"), origin="test")              # already exists on the canonical
    pairs = db.edges.rewrite_endpoint("vllm", "vLLM", origin="sweep")
    assert sorted(p[0] for p in pairs) == [a, b]
    by_old = dict(pairs)
    assert by_old[a] == c                                              # collision → existing id
    assert db.edges.by_id(by_old[b])["target"] == "vLLM"
    assert db.edges.by_id(by_old[b])["superseded_edge_id"] == b
    assert db.edges.active(either="vllm") == []
    assert {e["id"] for e in db.edges.active(either="vLLM")} == {c, by_old[b]}

    n = db.edges.revert_rewrites(pairs, reason="revert")
    assert n == 2
    assert {e["id"] for e in db.edges.active(either="vllm")} == {a, b}
    assert db.edges.by_id(by_old[b])["expired_at"]                     # the rewritten copy is gone
    assert db.edges.by_id(c)["expired_at"] is None                     # the pre-existing one stays


def test_rewrite_drops_edges_that_would_self_loop(db):
    i = db.edges.add(_edge("Intel Pipeline", "Intel"), origin="test")
    pairs = db.edges.rewrite_endpoint("Intel Pipeline", "Intel", origin="sweep")
    assert pairs == [] and db.edges.by_id(i)["expired_at"]


def test_reactivate_expires_a_competing_active_row(db):
    a = db.edges.add(_edge("A", "B"), origin="test")
    db.edges.expire(a, "x")
    b = db.edges.add(_edge("A", "B"), origin="test")
    assert db.edges.reactivate(a)
    assert db.edges.by_id(a)["expired_at"] is None and db.edges.by_id(b)["expired_at"]


# ── aliases / entities ───────────────────────────────────────────────────────

def test_resolve_is_case_insensitive_and_prefers_exact(db):
    db.entities.register("vLLM")
    db.aliases.set("vllm-engine", "vLLM", kind="punct", origin="sweep")
    assert db.aliases.resolve("VLLM-ENGINE") == "vLLM"
    assert db.resolve("VLLM") == "vLLM"                  # registry, case-insensitive
    assert db.resolve("vllm-engine") == "vLLM"
    assert db.resolve("nope") is None
    # two case variants registered: exact wins, else earliest
    db.entities.register("OpenClaw"); db.entities.register("openclaw")
    assert db.resolve("openclaw") == "openclaw"
    assert db.resolve("OPENCLAW") == "OpenClaw"


def test_self_alias_becomes_an_entity_not_an_alias(db):
    db.aliases.set("Lloyd", "Lloyd", kind="self", origin="test")
    assert db.aliases.count() == 0 and db.entities.exists("Lloyd")


def test_alias_set_upserts_and_remove_where_reports(db):
    db.aliases.set("intel pipeline system", "Intel", kind="suffix", origin="sweep", report_path="r1")
    db.aliases.set("intel pipeline system", "Intel Pipeline System", kind="case", origin="revert")
    assert db.aliases.resolve("Intel Pipeline System") == "Intel Pipeline System"
    db.aliases.set("Intel Pipeline System", "Intel", kind="suffix", origin="sweep")
    removed = db.aliases.remove_where(canonical="Intel", surface_lc="intel pipeline system")
    assert removed == ["Intel Pipeline System"]
    assert db.aliases.resolve("Intel Pipeline System") == "Intel Pipeline System"


def test_entities_rename_and_kind(db):
    assert db.entities.register("Foo", kind="project") == "Foo"
    assert db.entities.register("Foo") is None
    assert db.entities.rename("Foo", "Bar") and not db.entities.exists("Foo")
    assert db.entities.get("Bar")["kind"] == "project"


# ── export / import ──────────────────────────────────────────────────────────

def test_export_import_round_trip_on_fixture_graph(db, tmp_path):
    legacy_edges = [
        {"source": "A", "target": "B", "type": "mentions", "confidence": 0.8, "provenance": "EXTRACTED",
         "created_at": "2026-01-01T00:00:00+00:00", "expired_at": "2026-02-01T00:00:00+00:00", "source_doc": None},
        {"source": "A", "target": "B", "type": "uses", "confidence": 0.95, "provenance": "EXTRACTED_CLASSIFIER_V4",
         "created_at": "2026-02-01T00:00:00+00:00", "expired_at": None, "source_doc": None,
         "reason": "because", "superseded_edge": {"type": "mentions"}, "classifier_model": "primary",
         "classifier_meta": {"prompt_version": "v4"}},
        {"source": "C", "target": "D", "type": "related_to", "confidence": 0.7, "provenance": "STATED",
         "created_at": "2026-03-01T00:00:00+00:00", "expired_at": None, "source_doc": "notes/x.md"},
        # legacy duplicate active key → second one lands expired, nothing lost
        {"source": "C", "target": "D", "type": "related_to", "confidence": 0.6, "provenance": "STATED",
         "created_at": "2026-03-02T00:00:00+00:00", "expired_at": None, "source_doc": None},
    ]
    rel = tmp_path / "_relationships.json"
    rel.write_text(json.dumps({"schema_version": 1, "edges": legacy_edges}))
    al = tmp_path / "entity-aliases.json"
    al.write_text(json.dumps({"A": "A", "a": "A", "b-thing": "B", "C": "C", "D": "D"}))

    stats = db.import_json(rel, al)
    assert stats == {"edges": 4, "edges_skipped": 0, "aliases": 2, "entities": 3}
    assert db.edges.count(active_only=False) == 4 and db.edges.count() == 2
    dup = [e for e in db.edges.all() if e["created_at"].startswith("2026-03-02")][0]
    assert dup["expired_reason"] == "migration: duplicate active edge"
    typed = db.edges.find_active("A", "B", "uses")
    assert typed["reason"] == "because" and typed["classifier_meta"] == {"prompt_version": "v4"}
    assert typed["superseded_edge"] == {"type": "mentions"}

    # idempotent
    again = db.import_json(rel, al)
    assert again["edges"] == 0 and again["edges_skipped"] == 4

    out = db.export_json(tmp_path / "out")
    exported = json.loads(out["relationships"].read_text())
    assert len(exported["edges"]) == 4
    assert sum(1 for e in exported["edges"] if not e.get("expired_at")) == 2
    for k in ("source", "target", "type", "confidence", "provenance", "created_at", "expired_at", "source_doc"):
        assert all(k in e for e in exported["edges"]), k
    alias_out = json.loads(out["aliases"].read_text())
    # B was only ever an alias target, never registered: no self-identity is
    # invented for it (55 legacy aliases point at canonicals with no dir).
    assert alias_out == {"A": "A", "a": "A", "b-thing": "B", "C": "C", "D": "D"}


# ── fact index ───────────────────────────────────────────────────────────────

def _fact_file(root, entity, cat, facts):
    d = root / entity; d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity, "category": cat, "facts": facts}
    p = d / f"{entity}-{cat}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity} - {cat}\n")
    return p


def test_facts_idx_reindex_and_temporal_filters(db, tmp_path):
    root = tmp_path / "facts"
    _fact_file(root, "Lloyd", "state", [
        {"id": "stat-001", "fact": "Lloyd runs on vLLM", "confidence": 0.9, "created_at": "2026-01-01"},
        {"id": "stat-002", "fact": "Lloyd used Ollama", "confidence": 0.8, "expired_at": "2026-02-01"},
        {"id": "stat-003", "fact": "Lloyd will use X", "confidence": 0.5, "valid_at": "2026-06-01"},
    ])
    _fact_file(root, "Lloyd", "goal", [{"id": "goal-001", "fact": "be useful"}])
    (root / "Lloyd" / "Lloyd-overview.md").write_text("---\ntype: overview\nentity: Lloyd\n---\n# Summary\n")
    (root / "Broken").mkdir()
    (root / "Broken" / "Broken-state.md").write_text("---\nfacts: [unclosed\n---\n")
    stats = db.facts_idx.reindex(root=root)
    assert stats["facts"] == 4 and stats["entities_registered"] == 2   # every dir seen is an entity, Broken included
    assert db.entities.exists("Lloyd")
    # reindex walks files in sorted order, so row order is stable across runs
    assert [f["fact_id"] for f in db.facts_idx.for_entity("Lloyd")] == ["goal-001", "stat-001", "stat-003"]
    assert [f["fact_id"] for f in db.facts_idx.for_entity("Lloyd", category="state", as_of="2026-01-15")] == ["stat-001", "stat-002"]
    assert len(db.facts_idx.for_entity("Lloyd", include_expired=True)) == 4
    assert db.facts_idx.entity_fact_counts() == {"Lloyd": 3}
    assert db.facts_idx.categories_for("Lloyd") == ["goal", "state"]
    assert db.facts_idx.last_reindex()

    # single-file update replaces that file's rows only
    p = _fact_file(root, "Lloyd", "goal", [{"id": "goal-001", "fact": "be useful"}, {"id": "goal-002", "fact": "be fast"}])
    assert db.facts_idx.update_file(p, root=root) == 2
    assert db.facts_idx.count(entity="Lloyd") == 5
    p.unlink()
    db.facts_idx.update_file(p, root=root)
    assert db.facts_idx.count(entity="Lloyd") == 3


# ── caching / versioning ─────────────────────────────────────────────────────

def test_data_version_cache_invalidates_on_cross_process_commit(tmp_path):
    p = tmp_path / "kg.sqlite"
    a = KGStore(p)
    a.edges.add(_edge("A", "B"), origin="test")
    assert a.edges.degree() == {"A": 1, "B": 1}
    assert a.edges.adjacency()["A"][0]["target"] == "B"
    # another connection (as another process would) commits
    b = KGStore(p)
    b.edges.add(_edge("B", "C"), origin="test")
    b.close()
    assert a.edges.degree() == {"A": 1, "B": 2, "C": 1}
    assert {e["target"] for e in a.edges.adjacency()["B"]} == {"B", "C"}
    # own writes invalidate too
    a.aliases.set("c", "C", kind="case", origin="test")
    assert a.aliases.all_lower()["c"] == "C"
    a.close()


def test_transaction_rolls_back_whole_batch(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.edges.add(_edge("A", "B"), origin="test")
            db.edges.add(_edge("C", "D"), origin="test")
            raise RuntimeError("boom")
    assert db.edges.count(active_only=False) == 0


def test_unreadable_store_raises_store_unavailable(tmp_path):
    p = tmp_path / "kg.sqlite"
    p.write_bytes(b"this is not a database file, it is 64 bytes of garbage.........")
    with pytest.raises(StoreUnavailable):
        KGStore(p)


def _writer(path, prefix, n):
    s = KGStore(path)
    for i in range(n):
        s.edges.add({"source": f"{prefix}{i}", "target": "hub", "type": "uses"}, origin="p")
    s.close()


def test_concurrent_writers_from_two_processes_never_lose_an_edge(tmp_path):
    p = tmp_path / "kg.sqlite"
    KGStore(p).close()
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=_writer, args=(p, "x", 300)), ctx.Process(target=_writer, args=(p, "y", 300))]
    for q in ps: q.start()
    for q in ps: q.join(60)
    assert all(q.exitcode == 0 for q in ps)
    s = KGStore(p)
    assert s.edges.count() == 600
    assert s.integrity_check() == "ok"
    s.close()


def test_kill_dash_nine_mid_write_leaves_a_consistent_store(tmp_path):
    p = tmp_path / "kg.sqlite"
    KGStore(p).close()
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from app.kg_store import KGStore\n"
        "s = KGStore(%r)\n"
        "i = 0\n"
        "while True:\n"
        "    s.edges.add({'source': 'n%%d' %% i, 'target': 'hub', 'type': 'uses'}, origin='k')\n"
        "    i += 1\n"
    ) % (str(ROOT), str(p))
    proc = subprocess.Popen([sys.executable, "-c", code])
    time.sleep(1.0)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    s = KGStore(p)
    assert s.integrity_check() == "ok"
    n = s.edges.count()
    assert n > 0
    # every row is whole: no half-written edge without endpoints
    assert all(e["source"] and e["target"] for e in s.edges.all())
    s.close()


def test_backup_is_a_consistent_copy(db, tmp_path):
    db.edges.add(_edge("A", "B"), origin="test")
    dest = db.backup(tmp_path / "bak" / "kg.sqlite")
    c = KGStore(dest)
    assert c.edges.count() == 1 and c.integrity_check() == "ok"
    c.close()


def test_default_store_can_be_pointed_at_a_path(tmp_path):
    s = kg_store.configure(tmp_path / "kg.sqlite")
    assert kg_store.store() is s
    kg_store.reset()
