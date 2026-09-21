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


def _variant_fact_file(root, dir_name, entity_tag, cat, facts):
    """Fact file that SITS IN `dir_name` but whose `entity:` tag is `entity_tag`.

    That mismatch is the whole defect: the file already lives in the canonical
    directory while its own tag still spells a variant, which is how
    `_rows_for_file` came to key 17,202 active rows under 1,430 alias surfaces
    on the live index (measured 2026-09-21).
    """
    d = root / dir_name; d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity_tag, "category": cat, "facts": facts}
    p = d / f"{entity_tag}-{cat}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {dir_name} - {cat}\n")
    return p


def _indexed(db, sql="SELECT fact_id, file_path, entity FROM facts_idx"):
    return sorted(tuple(r) for r in db.conn.execute(sql))


def test_reindex_keys_facts_under_the_canonical_the_alias_table_declares(db, tmp_path):
    """#957 clause 1: the stored key is `aliases.resolve(name)`, literal when no row exists.

    Covers both sources of the key — the file's own `entity:` frontmatter tag
    and a per-fact `entity:` override — and pins the no-alias fallback, because
    folding a name nobody declared is the `06f0e41` failure mode. The fold
    lives at index-build time so the per-file write path (`update_file`, which
    every `fact_add` triggers) keys canonically too; a one-off UPDATE would
    revert on the next write.
    """
    root = tmp_path / "facts"
    db.aliases.set("TencentDB-Agent-Memory", "TencentDB Agent Memory", kind="punct", origin="test")
    db.aliases.set("autonomy-data-pipeline", "Autonomy Data Pipeline", kind="punct", origin="test")
    # (a) the file-level tag is the declared variant
    variant_file = _variant_fact_file(
        root, "TencentDB Agent Memory", "TencentDB-Agent-Memory", "architecture",
        [{"id": "arch-001", "fact": "keeps a vector store"}])
    # (b) the file tag is already canonical; the per-fact override is the variant
    _variant_fact_file(
        root, "Autonomy Data Pipeline", "Autonomy Data Pipeline", "usage",
        [{"id": "u-001", "fact": "runs nightly", "entity": "autonomy-data-pipeline"}])
    # (c) no alias row for this tag: the literal name is the key
    _variant_fact_file(root, "Loamgate", "Loamgate Archived", "state",
                       [{"id": "l-001", "fact": "no alias row names it"}])
    db.facts_idx.reindex(root=root)
    assert sorted(r[0] for r in _indexed(db, "SELECT entity, fact_id FROM facts_idx")) == [
        "Autonomy Data Pipeline", "Loamgate Archived", "TencentDB Agent Memory"]
    assert db.facts_idx.count(entity="TencentDB Agent Memory", active_only=True) == 1
    assert db.facts_idx.count(entity="TencentDB-Agent-Memory", active_only=True) == 0
    # and a single-file re-read after a write keys the same way
    db.facts_idx.update_file(variant_file, root=root)
    assert db.facts_idx.count(entity="TencentDB Agent Memory", active_only=True) == 1


def test_a_chained_alias_folds_to_the_end_of_the_chain_not_one_hop(db, tmp_path):
    """One hop would leave rows under a surface, which is the defect restated.

    The live alias table holds 107 rows whose canonical is itself a surface
    (measured 2026-09-21), e.g. `Codemode` -> `CodeMode` -> `Code Mode`, and 81
    active rows sat two hops from home. Every hop is still a declared row, so
    following the chain is still not a guess.
    """
    root = tmp_path / "facts"
    db.aliases.set("LloydBot", "lloyd-bot", kind="punct", origin="sweep")
    db.aliases.set("lloyd-bot", "Lloyd Bot", kind="case", origin="sweep")
    _variant_fact_file(root, "Lloyd Bot", "LloydBot", "state",
                       [{"id": "b-001", "fact": "greets in the morning"}])
    db.facts_idx.reindex(root=root)
    assert db.facts_idx.count(entity="Lloyd Bot", active_only=True) == 1
    assert db.facts_idx.count(entity="lloyd-bot", active_only=True) == 0
    assert db.facts_idx.count(entity="LloydBot", active_only=True) == 0
    assert db.facts_idx.for_entity("Lloyd Bot")[0]["fact_id"] == "b-001"


def test_canonical_read_covers_a_family_whose_files_keep_the_variant_tag(db, tmp_path):
    """#957 clause 2: for_entity(canonical) returns the whole family, not one file's rows.

    Before the fix the canonical dir's two files split across two keys, so the
    entity API (`app/routers/entities.py` → `for_entity`) served the canonical
    file's rows only while the variant's rows were unreachable — the route
    normalises the request name first, so the variant key cannot be asked for.
    """
    root = tmp_path / "facts"
    db.aliases.set("TencentDB-Agent-Memory", "TencentDB Agent Memory", kind="punct", origin="test")
    _fact_file(root, "TencentDB Agent Memory", "architecture",
               [{"id": "a-001", "fact": "keeps a vector store"}])
    _variant_fact_file(root, "TencentDB Agent Memory", "TencentDB-Agent-Memory", "usage",
                       [{"id": "u-001", "fact": "serves the memory api"},
                        {"id": "u-002", "fact": "backs fact_add"}])
    db.facts_idx.reindex(root=root)
    rows = db.facts_idx.for_entity("TencentDB Agent Memory")
    assert [f["fact_id"] for f in rows] == ["a-001", "u-001", "u-002"]      # 3 of 3, was 1 of 3
    assert db.facts_idx.for_entity("TencentDB-Agent-Memory") == []          # that key no longer exists
    assert all(f["file_path"].startswith("TencentDB Agent Memory/") for f in rows)


def test_entity_fact_counts_reports_one_entry_per_family_and_no_alias_key(db, tmp_path):
    """#957 clause 3: the per-entity totals the MC entity list and graph read carry one entry per family.

    `entity_fact_counts` groups the raw key, so a family split across the
    canonical and a variant surface showed up twice — `Autonomy Data Pipeline`
    179 next to `autonomy-data-pipeline` 2,661.
    """
    root = tmp_path / "facts"
    db.aliases.set("TencentDB-Agent-Memory", "TencentDB Agent Memory", kind="punct", origin="test")
    _fact_file(root, "TencentDB Agent Memory", "architecture",
               [{"id": "a-001", "fact": "keeps a vector store"}])
    _variant_fact_file(root, "TencentDB Agent Memory", "TencentDB-Agent-Memory", "usage",
                       [{"id": "u-001", "fact": "serves the memory api"},
                        {"id": "u-002", "fact": "backs fact_add"}])
    _fact_file(root, "Lloyd", "state", [{"id": "s-001", "fact": "runs on vLLM"}])
    db.facts_idx.reindex(root=root)
    counts = db.facts_idx.entity_fact_counts()
    assert counts == {"TencentDB Agent Memory": 3, "Lloyd": 1}
    surfaces = {r["surface"] for r in db.aliases.rows() if r["surface"] != r["canonical"]}
    assert surfaces == {"TencentDB-Agent-Memory"}
    assert not (set(counts) & surfaces)                 # no count is filed under an alias surface


def test_retagging_index_keys_rewrites_no_fact_file_and_moves_no_file_path(db, tmp_path):
    """#957 clause 5: only the key moves — files stay byte-identical, and the
    contamination gate, which reads those files, reports the same thing."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kg_hygiene", ROOT / "scripts/memory/kg_hygiene.py")
    kgh = importlib.util.module_from_spec(spec); spec.loader.exec_module(kgh)

    root = tmp_path / "facts"
    _variant_fact_file(root, "TencentDB Agent Memory", "TencentDB-Agent-Memory", "architecture",
                       [{"id": "a-001", "fact": "keeps a vector store"},
                        {"id": "a-002", "fact": "serves the memory api"}])
    # 832 of the 17,243 rows sat in directories that are THEMSELVES alias
    # surfaces (`The Graph/`, `Lloyd Knowledge Graph/`, …), so after the fold a
    # canonical key can name a different entity than the directory holding its
    # own file. Pinned here, because it is the one shape that could make the
    # file-based gate disagree with the index.
    _variant_fact_file(root, "The Graph", "The-Graph", "state",
                       [{"id": "g-001", "fact": "one node per entity"}])
    bytes_before = {p: p.read_bytes() for p in sorted(root.rglob("*.md"))}
    db.facts_idx.reindex(root=root)                     # no alias row yet: literal key
    literal = _indexed(db)
    assert [r[2] for r in literal] == ["TencentDB-Agent-Memory", "TencentDB-Agent-Memory", "The-Graph"]
    contam_before = kgh.contamination(root)             # normalize_punct already equates the two spellings
    assert contam_before["foreign_facts"] == 0

    db.aliases.set("TencentDB-Agent-Memory", "TencentDB Agent Memory", kind="punct", origin="test")
    db.aliases.set("The-Graph", "Knowledge Graph", kind="punct", origin="test")
    db.aliases.set("The Graph", "Knowledge Graph", kind="punct", origin="test")
    db.facts_idx.reindex(root=root)
    after = _indexed(db)
    assert [(r[0], r[1]) for r in after] == [(r[0], r[1]) for r in literal]  # same rows, same files
    assert [r[2] for r in after] == ["TencentDB Agent Memory", "TencentDB Agent Memory",
                                     "Knowledge Graph"]                      # only the key moved
    # The directory keeps its own variant name under a canonical key: decoupled
    # on purpose, and exactly what no-fact-file-rewritten buys.
    assert ("g-001", "The Graph/The-Graph-state.md", "Knowledge Graph") in after
    assert {p: p.read_bytes() for p in sorted(root.rglob("*.md"))} == bytes_before
    assert kgh.contamination(root) == contam_before


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


def test_default_store_refuses_an_absent_database_and_creates_no_file(tmp_path, monkeypatch):
    """An absent database is refused, not answered with zeros (#1236).

    `sqlite3.connect` CREATES the file and `_init_schema()` fills it with empty
    tables, so before the guard `store().facts_idx.count()` returned 0 against a
    path holding no database — the false clean bill rule 7 of
    `architecture/knowledge-graph.md` exists to stop, reachable from any
    self-mod worktree because `_pipeline/` is gitignored.
    """
    store_dir = tmp_path / "vault-derived"
    store_dir.mkdir()
    absent = store_dir / "kg.sqlite"
    monkeypatch.setattr(kg_store, "_default_path", absent)
    kg_store.reset()

    with pytest.raises(StoreUnavailable) as exc:
        kg_store.store()

    assert str(absent.resolve()) in str(exc.value)   # names the path it looked at
    assert not absent.exists()
    assert list(store_dir.iterdir()) == []           # nothing created as a side effect


def test_default_store_refuses_without_making_the_missing_directory(tmp_path, monkeypatch):
    """The refusal happens before `_open()`, which mkdirs the parent (#1236)."""
    absent = tmp_path / "never-made" / "kg.sqlite"
    monkeypatch.setattr(kg_store, "_default_path", absent)
    kg_store.reset()

    with pytest.raises(StoreUnavailable):
        kg_store.store()

    assert not (tmp_path / "never-made").exists()


def test_a_reader_process_refuses_an_absent_store_and_leaves_no_file(tmp_path):
    """The documented recipe, run the way it is written, at an absent path (#1236).

    `architecture/knowledge-graph.md` tells a reader to run
    `from app.kg_store import store; store().facts_idx.count()`. That crosses a
    process boundary — `LLOYD_KG_DB` in the environment, `app.paths` reading it
    at import, the module-level reader refusing — so it is run as a subprocess
    rather than through in-process patching, and must leave no file behind.
    """
    store_dir = tmp_path / "vault-derived"
    store_dir.mkdir()
    absent = store_dir / "kg.sqlite"
    env = dict(os.environ, PYTHONPATH=str(ROOT), LLOYD_KG_DB=str(absent))
    env.pop("LLOYD_FACTS_ROOT", None)

    proc = subprocess.run(
        [sys.executable, "-c", "from app.kg_store import store; print(store().facts_idx.count())"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)

    assert proc.returncode != 0
    assert "StoreUnavailable" in proc.stderr
    assert str(absent.resolve()) in proc.stderr
    assert not absent.exists()
    assert list(store_dir.iterdir()) == []


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


def test_the_provisioning_routes_still_create_an_absent_database(tmp_path):
    """The guard sits on the reader `store()`, not on the routes that provision.

    `KGStore(path)` and `configure(path)` are how a rebuild, a migration or a
    test says "make this database": both still create a missing file, mkdirs and
    all, and the store they hand back is readable at zero rows.
    """
    s = KGStore(tmp_path / "a" / "kg.sqlite")
    assert s.path.is_file() and s.edges.count() == 0
    s.close()

    c = kg_store.configure(tmp_path / "b" / "kg.sqlite")
    assert c.path.is_file()
    assert kg_store.store() is c
    assert c.facts_idx.count() == 0
    kg_store.reset()
