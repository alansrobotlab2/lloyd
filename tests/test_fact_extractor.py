"""fact_extractor + nightly_extraction — the write path that owns the corpus.

Pins the seven defects the 2026-09-03 review found in this module: restarted
fact IDs, missing provenance, an LLM error that looked like a clean empty
extraction, a YAML error that wiped an entity's history, categories
registered as entities, junk names registered before they were rejected, and
edges that only ever appeared when someone ran a script by hand.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "scripts" / "memory" / "next-gen-memory"))

from app import kg_store  # noqa: E402
from app.fact_ids import assign_ids, next_fact_id, category_prefix  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fx = _load("fact_extractor", "scripts/memory/next-gen-memory/fact_extractor.py")


@pytest.fixture
def extractor(tmp_path, monkeypatch):
    """A FactExtractor writing into a temp tree, with a temp store."""
    facts = tmp_path / "facts"
    facts.mkdir()
    kg_store.configure(tmp_path / "kg.sqlite")
    monkeypatch.setattr(fx, "FACTS_DIR", facts)
    e = fx.FactExtractor()
    e.facts_dir = facts
    yield e
    kg_store.reset()


def _fact(text, **kw):
    return {"fact": text, "confidence": 0.9, "category": "state", **kw}


def _read(path):
    return yaml.safe_load(path.read_text().split("---")[1])


# ── fact IDs ─────────────────────────────────────────────────────────────────

def test_ids_continue_instead_of_restarting():
    """43% of fact files carried duplicate IDs because numbering restarted at
    1 on every run and only filled facts that had none."""
    facts = [{"id": "pref-001", "fact": "a"}, {"id": "pref-002", "fact": "b"},
             {"fact": "c"}, {"fact": "d"}]
    assign_ids(facts, "preference")
    assert [f["id"] for f in facts] == ["pref-001", "pref-002", "pref-003", "pref-004"]
    assert len({f["id"] for f in facts}) == 4


def test_ids_survive_a_second_pass(extractor):
    e = extractor
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("one"), _fact("two")]})
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("three")]})
    fm = _read(e.facts_dir / "Lloyd" / "Lloyd-state.md")
    ids = [f["id"] for f in fm["facts"]]
    assert ids == ["stat-001", "stat-002", "stat-003"]
    assert len(set(ids)) == 3


def test_id_scheme_is_shared_with_fact_add(extractor, monkeypatch):
    """fact_add minted `pref-a3f9` from a UUID; the extractor minted
    `pref-003`. One scheme now, so a file's IDs sort and continue."""
    from agent_mcp import facts as facts_mod
    import agent_mcp._shared as shared
    e = extractor
    monkeypatch.setattr(shared, "FACTS_ROOT", e.facts_dir)
    monkeypatch.setattr(facts_mod, "FACTS_ROOT", e.facts_dir)
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("one")]})
    shared._invalidate_entity_dirs_cache()
    try:
        r = facts_mod._fact_add({"entity": "Lloyd", "category": "state", "fact": "added in chat"})
        assert r.get("fact_id") == "stat-002", r
    finally:
        shared._invalidate_entity_dirs_cache()


def test_category_prefix_folds_plurals():
    assert category_prefix("preferences") == category_prefix("preference") == "pref"
    assert next_fact_id([], "goal") == "goal-001"


# ── provenance ───────────────────────────────────────────────────────────────

def test_every_written_fact_carries_provenance(extractor):
    e = extractor
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("runs on vLLM")]},
                      source_doc="knowledge/lloyd.md", source_hash="abc123")
    fm = _read(e.facts_dir / "Lloyd" / "Lloyd-state.md")
    f = fm["facts"][0]
    assert f["provenance"] == "EXTRACTED"
    assert f["created_at"] and f["source_doc"] == "knowledge/lloyd.md"
    assert f["source_hash"] == "abc123"
    assert f["expired_at"] is None and f["invalid_at"] is None
    assert fm["source_doc"] == "knowledge/lloyd.md"


def test_event_date_becomes_valid_at(extractor):
    e = extractor
    e.write_fact_file("Lloyd", "event", {"facts": [_fact("shipped", event_date="2026-05-01")]})
    f = _read(e.facts_dir / "Lloyd" / "Lloyd-event.md")["facts"][0]
    assert f["valid_at"] == "2026-05-01" and f["created_at"] != "2026-05-01"


# ── merge / dedupe ───────────────────────────────────────────────────────────

def test_merge_facts_dedupes_by_text(extractor):
    e = extractor
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("same text")]})
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("same text"), _fact("new text")]})
    fm = _read(e.facts_dir / "Lloyd" / "Lloyd-state.md")
    assert [f["fact"] for f in fm["facts"]] == ["same text", "new text"]


# ── LLM failure ──────────────────────────────────────────────────────────────

def test_llm_failure_raises_rather_than_returning_empty(extractor, monkeypatch):
    """An empty fact list was indistinguishable from `this doc has no facts`,
    so a transient vLLM error marked the document extracted forever."""
    def boom(url, timeout=0):
        raise OSError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(fx.ExtractionFailed):
        extractor._call_llm("anything")


def test_a_failed_file_is_not_hashed(tmp_path, monkeypatch):
    ne = _load("nightly_extraction", "scripts/memory/next-gen-memory/nightly_extraction.py")
    kg_store.configure(tmp_path / "kg.sqlite")
    x = ne.NightlyExtraction()
    doc = tmp_path / "doc.md"
    doc.write_text("some content")
    monkeypatch.setattr(ne, "VAULT", tmp_path)

    def fail(*a, **k):
        raise fx.ExtractionFailed("vLLM wedged")
    monkeypatch.setattr(x.extractor, "extract_from_document", fail)
    assert x._process_single_file(doc, True, 1, 1) == (0, 0, False)

    # a successful, factless extraction IS recorded — it is genuinely done
    monkeypatch.setattr(x.extractor, "extract_from_document",
                        lambda *a, **k: {"entity": "X", "category": "state", "facts": []})
    assert x._process_single_file(doc, True, 1, 1) == (1, 0, True)
    kg_store.reset()


# ── corrupt frontmatter ──────────────────────────────────────────────────────

def test_corrupt_file_is_quarantined_not_wiped(extractor):
    """`existing_facts = []` on a YAML error, followed immediately by a write,
    deleted an entity's whole history over one bad character."""
    e = extractor
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("valuable history")]})
    target = e.facts_dir / "Lloyd" / "Lloyd-state.md"
    target.write_text("---\nfacts: [unclosed\nentity: Lloyd\n---\n\nbody\n")
    before = target.read_text()

    assert e.write_fact_file("Lloyd", "state", {"facts": [_fact("new fact")]}) is None
    assert not target.exists()
    quarantined = list((e.facts_dir / "Lloyd").glob("*.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == before


def test_missing_fence_is_quarantined(extractor):
    e = extractor
    d = e.facts_dir / "Lloyd"; d.mkdir()
    (d / "Lloyd-state.md").write_text("no frontmatter at all\n")
    assert e.write_fact_file("Lloyd", "state", {"facts": [_fact("x")]}) is None
    assert list(d.glob("*.corrupt-*"))


def test_an_empty_file_is_not_corrupt(extractor):
    e = extractor
    d = e.facts_dir / "Lloyd"; d.mkdir()
    (d / "Lloyd-state.md").write_text("")
    assert e.write_fact_file("Lloyd", "state", {"facts": [_fact("x")]}) is not None


# ── categories ───────────────────────────────────────────────────────────────

def test_category_vocabulary_collapses_spellings():
    """287 category spellings existed because the model's answer was written
    through verbatim, and each spelling made its own fact file."""
    assert fx.normalize_category("States") == "state"
    assert fx.normalize_category("current state") == "state"
    assert fx.normalize_category("Config") == "configuration"
    assert fx.normalize_category("relationship notes") == "relationship"
    assert fx.normalize_category("") == "general"
    assert fx.normalize_category("entirely unrelated") == "general"
    assert all(fx.normalize_category(c) == c for c in fx.CATEGORY_VOCAB)


def test_a_category_never_becomes_an_entity(extractor):
    e = extractor
    e.write_fact_file("Lloyd", "Current State", {"facts": [_fact("x")]})
    assert (e.facts_dir / "Lloyd" / "Lloyd-state.md").exists()
    assert not (e.facts_dir / "Current State").exists()
    assert kg_store.store().entities.lookup("state") is None


# ── junk guard ───────────────────────────────────────────────────────────────

def test_junk_entity_is_rejected_before_registration(extractor):
    """Names were registered as canonicals first and rejected at write time,
    which is how 921 run-named entities entered the alias table."""
    e = extractor
    assert e.write_fact_file("Sweep Run 313", "state", {"facts": [_fact("x")]}) is None
    assert not (e.facts_dir / "Sweep Run 313").exists()
    assert kg_store.store().entities.lookup("Sweep Run 313") is None


# ── edge emission ────────────────────────────────────────────────────────────

def test_extractor_emits_edges_for_named_entities(extractor):
    """The growth path: edges used to appear only when someone ran
    seed_relationship_edges.py by hand, so the nightly chain added none."""
    st = kg_store.store()
    e = extractor
    for name in ("Lloyd", "vLLM", "Isaac Lab"):
        st.entities.register(name)
        (e.facts_dir / name).mkdir(exist_ok=True)

    e.write_fact_file(
        "Lloyd", "relationship",
        {"facts": [_fact("Lloyd serves models through vLLM and trains in Isaac Lab",
                         category="relationship")]},
        source_doc="knowledge/lloyd.md",
    )
    edges = {(x["source"], x["target"]): x for x in st.edges.active()}
    assert ("Lloyd", "vLLM") in edges and ("Lloyd", "Isaac Lab") in edges
    edge = edges[("Lloyd", "vLLM")]
    assert edge["type"] == "mentions" and edge["provenance"] == "EXTRACTED"
    assert edge["origin"] == "extractor"
    assert edge["source_doc"] == "knowledge/lloyd.md"
    assert "vLLM" in edge["evidence"]


def test_edge_emission_skips_self_and_dedupes(extractor):
    st = kg_store.store()
    e = extractor
    for name in ("Lloyd", "vLLM"):
        st.entities.register(name)
    body = {"facts": [_fact("Lloyd uses vLLM")]}
    e.write_fact_file("Lloyd", "state", body, source_doc="a.md")
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("Lloyd uses vLLM again")]}, source_doc="b.md")
    assert st.edges.count() == 1                    # deduped by the active unique index
    assert not st.edges.active(source="Lloyd", target="Lloyd")


def test_facts_are_indexed_as_they_are_written(extractor):
    st = kg_store.store()
    extractor.write_fact_file("Lloyd", "state", {"facts": [_fact("a"), _fact("b")]},
                              source_doc="knowledge/lloyd.md")
    rows = st.facts_idx.for_entity("Lloyd")
    assert len(rows) == 2
    assert rows[0]["source_doc"] == "knowledge/lloyd.md" and rows[0]["provenance"] == "EXTRACTED"


# ── concurrency ──────────────────────────────────────────────────────────────

def test_two_threads_writing_one_file_lose_nothing(extractor):
    """Four extractor threads plus fact_add all target the same file; without
    the lock the later read-modify-write drops the earlier one's facts."""
    import threading
    e = extractor
    errors = []

    def writer(n):
        try:
            for i in range(10):
                e.write_fact_file("Lloyd", "state", {"facts": [_fact(f"thread {n} fact {i}")]})
        except Exception as exc:   # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads: t.start()
    for t in threads: t.join(60)
    assert not errors
    fm = _read(e.facts_dir / "Lloyd" / "Lloyd-state.md")
    assert len(fm["facts"]) == 40
    assert len({f["id"] for f in fm["facts"]}) == 40


# ── corpus allow-list ────────────────────────────────────────────────────────

def test_corpus_comes_from_the_config_allow_list(tmp_path, monkeypatch):
    ne = _load("nightly_extraction2", "scripts/memory/next-gen-memory/nightly_extraction.py")
    vault = tmp_path / "obsidian"
    for rel in ("knowledge/a.md", "projects/b.md", "skills/c.md", "autonomy/d.md",
                "youtube/e.md", "memory/2020-01-01.md", "facts/f.md"):
        p = vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    cfg = tmp_path / "pipeline_config.yaml"
    cfg.write_text(yaml.dump({"sources": {
        "paths": ["knowledge", "projects", "memory", "/etc"],
        "exclude_patterns": [".git/"],
    }}))
    monkeypatch.setattr(ne, "VAULT", vault)
    monkeypatch.setattr(ne, "CONFIG_PATH", cfg)
    kg_store.configure(tmp_path / "kg.sqlite")
    x = ne.NightlyExtraction()
    got = {str(Path(p).relative_to(vault)) for p in x._eligible_files(full_mode=True)}
    assert got == {"knowledge/a.md", "projects/b.md", "memory/2020-01-01.md"}
    kg_store.reset()


def test_empty_sources_is_an_error_not_a_whole_vault_walk(tmp_path, monkeypatch):
    ne = _load("nightly_extraction3", "scripts/memory/next-gen-memory/nightly_extraction.py")
    cfg = tmp_path / "pipeline_config.yaml"
    cfg.write_text(yaml.dump({"sources": {"paths": []}}))
    monkeypatch.setattr(ne, "CONFIG_PATH", cfg)
    kg_store.configure(tmp_path / "kg.sqlite")
    with pytest.raises(RuntimeError, match="no usable sources.paths"):
        ne.NightlyExtraction()._eligible_files(full_mode=True)
    kg_store.reset()


# ── the rebuild's write flag ─────────────────────────────────────────────────

def test_fact_writes_can_be_disabled_for_a_rebuild(extractor, monkeypatch):
    """A fact added during the rebuild would land in a tree about to be
    renamed to facts-quarantine-<ts>."""
    from agent_mcp import facts as facts_mod
    monkeypatch.setattr(facts_mod, "_writes_enabled", lambda: False)
    add = facts_mod._fact_add({"entity": "Lloyd", "category": "state", "fact": "x"})
    assert "error" in add and "rebuild" in add["error"]
    relate = facts_mod._fact_relate({"source": "A", "target": "B", "type": "uses"})
    assert "error" in relate and "rebuild" in relate["error"]


def test_content_hasher_honours_the_env_override(tmp_path, monkeypatch):
    """Without its own index the rebuild would skip every file the live tree
    had already extracted and produce an empty tree."""
    idx = tmp_path / "rebuild-hashes.json"
    monkeypatch.setenv("LLOYD_CONTENT_HASHES", str(idx))
    ch = _load("content_hasher_env", "scripts/memory/content_hasher.py")
    h = ch.ContentHasher()
    assert h.index_path == idx
    doc = tmp_path / "a.md"
    doc.write_text("hello")
    assert h.has_changed(doc)
    h.update_hashes([doc]); h.save()
    assert not ch.ContentHasher().has_changed(doc)
    assert idx.exists()
