"""fact_extractor + nightly_extraction — the write path that owns the corpus.

Pins the seven defects the 2026-09-03 review found in this module: restarted
fact IDs, missing provenance, an LLM error that looked like a clean empty
extraction, a YAML error that wiped an entity's history, categories
registered as entities, junk names registered before they were rejected, and
edges that only ever appeared when someone ran a script by hand.
"""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "scripts" / "memory" / "next-gen-memory"))

from app import entity_naming as en  # noqa: E402
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


def test_one_pass_repairs_a_duplicate_already_in_the_file(extractor):
    """#499 clause 4. 708 fact files hold the same text twice *inside one file*:
    `_merge_facts` started from `merged = existing.copy()`, so it refused a
    repeat coming IN but preserved the repeat already THERE, and one more
    extraction pass re-indexed both copies. A pass over such a file now leaves
    one copy in the markdown and one row in `facts_idx`."""
    e = extractor
    path = e.write_fact_file("Lloyd", "state", {"facts": [_fact("twin text")]})
    fm = _read(path)
    fm["facts"].append({**fm["facts"][0], "id": "stat-002"})   # the legacy pair
    path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\nbody\n", encoding="utf-8")
    assert [f["fact"] for f in _read(path)["facts"]] == ["twin text", "twin text"]

    e.write_fact_file("Lloyd", "state", {"facts": [_fact("one more")]})

    assert [f["fact"] for f in _read(path)["facts"]] == ["twin text", "one more"]
    rows = kg_store.store().facts_idx.for_entity("Lloyd", include_expired=True)
    assert [r["fact"] for r in rows] == ["twin text", "one more"], rows


def test_merge_facts_keeps_the_copy_that_carries_an_expiry(extractor):
    """Repairing a pair must not retire anything: #499 records that expiring
    one copy of a pair is what deleted useful facts, so an expired or invalid
    member of the pair has to survive the merge on its own id."""
    existing = [
        {"fact": "dated claim", "id": "stat-001", "confidence": 0.9,
         "expired_at": "2026-01-01T00:00:00+00:00"},
        {"fact": "dated claim", "id": "stat-002", "confidence": 0.9, "expired_at": None},
        {"fact": "keep me", "id": "stat-003", "confidence": 0.9},
    ]
    merged = extractor._merge_facts(existing, [{"fact": "new", "id": "stat-004",
                                                "confidence": 0.9}])
    assert [(f["fact"], f["id"], f.get("expired_at")) for f in merged] == [
        ("dated claim", "stat-001", "2026-01-01T00:00:00+00:00"),
        ("dated claim", "stat-002", None),
        ("keep me", "stat-003", None),
        ("new", "stat-004", None),
    ]


# ── cross-category duplicate refusal (#1144 clause 1) ────────────────────────

def test_extracting_a_text_the_entity_holds_elsewhere_leaves_one_row(extractor):
    """#1144 clause 1: the write path that owns the corpus refuses on the store's
    key, not on the file it is about to append to.

    Of the same-entity duplicate groups on the live store at 2026-09-21 07:34Z,
    95 have BOTH copies created on or after 2026-09-14 — after #499's refusal
    settled — because that guard lives in `_fact_add` and nightly extraction
    writes through here, where `_merge_facts` consults one file. A fact filed
    under `state` and then re-extracted under `relationship` is the shape it
    took, so that is the shape this pins.
    """
    e = extractor
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("serves the bench model")]})

    e.write_fact_file("Lloyd", "relationship",
                      {"facts": [_fact("serves the bench model",
                                       category="relationship")]})

    rows = kg_store.store().facts_idx.for_entity("Lloyd", include_expired=True)
    assert [r["fact"] for r in rows] == ["serves the bench model"], rows
    assert rows[0]["category"] == "state", rows
    assert not (e.facts_dir / "Lloyd" / "Lloyd-relationship.md").exists(), \
        "the refused write still created its category file"


def test_case_and_padding_variants_refuse_on_the_stores_own_key(extractor):
    """The extractor now refuses on `text_hash` — strip + casefold + sha256 —
    so it refuses exactly what `facts_idx` counts as a duplicate, no looser and
    not stricter: a re-worded claim is a different claim and still writes."""
    e = extractor
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("Re-indexes the vault")]})

    e.write_fact_file("Lloyd", "event",
                      {"facts": [_fact("  re-indexes the vault  ", category="event")]})
    e.write_fact_file("Lloyd", "event",
                      {"facts": [_fact("Re-indexes the vault nightly", category="event")]})

    rows = kg_store.store().facts_idx.for_entity("Lloyd", include_expired=True)
    assert sorted(r["fact"] for r in rows) == ["Re-indexes the vault",
                                              "Re-indexes the vault nightly"], rows


def test_a_different_category_still_gets_a_genuinely_new_fact(extractor):
    """The guard is keyed on text, not on the category: an earlier `state` file
    must not swallow the entity's first `event` fact."""
    e = extractor
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("runs on vLLM")]})
    path = e.write_fact_file("Lloyd", "event",
                             {"facts": [_fact("shipped 0.9 on Tuesday", category="event")]})

    assert path == e.facts_dir / "Lloyd" / "Lloyd-event.md"
    rows = kg_store.store().facts_idx.for_entity("Lloyd", include_expired=True)
    assert len(rows) == 2, rows


def test_an_all_duplicate_batch_does_not_create_an_empty_category_file(extractor):
    """A refusal that opened its file would leave an empty fact file per refused
    write — churn the index then has to carry. `_fact_add`'s refusal never opens
    the file; this one matches, and only in that case."""
    e = extractor
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("holds this text")]})

    assert e.write_fact_file("Lloyd", "goal",
                             {"facts": [_fact("holds this text", category="goal")]}) is None
    assert not (e.facts_dir / "Lloyd" / "Lloyd-goal.md").exists()


def test_the_guard_narrows_to_the_file_when_the_store_is_unreadable(extractor, monkeypatch):
    """`_index_and_link` already degrades to a warning when the store is down, so
    the duplicate check cannot live only in the store. With no index to consult
    the extractor behaves exactly as it did before #1144: same-file copies are
    still folded by `_merge_facts`, cross-category ones are not refused. Failing
    toward writing is the deliberate direction — a guard that dropped facts
    because it could not read would be a worse defect than the one it prevents."""
    def no_store():
        raise kg_store.StoreUnavailable("probe: store is closed")
    monkeypatch.setattr(fx, "_kg_store", no_store)
    e = extractor

    e.write_fact_file("Lloyd", "state", {"facts": [_fact("captured without an index")]})
    e.write_fact_file("Lloyd", "state", {"facts": [_fact("captured without an index")]})
    e.write_fact_file("Lloyd", "event",
                      {"facts": [_fact("captured without an index", category="event")]})

    assert [f["fact"] for f in _read(e.facts_dir / "Lloyd" / "Lloyd-state.md")["facts"]] == \
        ["captured without an index"]
    assert (e.facts_dir / "Lloyd" / "Lloyd-event.md").exists(), \
        "the degraded guard stopped writing"


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
    # The 4th element is the digest of the bytes read; there are none here, so
    # it is "" and the caller's `ok=False` branch keeps it out of the index.
    # Elements 5 and 6 are chars_covered/chars_total (#1151 clause 1); a file
    # that never got to the model reports none of either.
    assert x._process_single_file(doc, True, 1, 1) == (0, 0, False, "", 0, 0)

    # a successful, factless extraction IS recorded — it is genuinely done, and
    # what it records is the digest of the bytes it read (#482 clause 4). The
    # gate stores that digest instead of re-hashing the file at flush time.
    # This double returns no coverage keys, which means "everything I was given
    # was read" (see the `chars_covered` fallback in `_process_single_file`), so
    # both coverage elements are the document's own length.
    read_digest = hashlib.sha256(doc.read_bytes()).hexdigest()
    monkeypatch.setattr(x.extractor, "extract_from_document",
                        lambda *a, **k: {"entity": "X", "category": "state", "facts": []})
    assert x._process_single_file(doc, True, 1, 1) == (1, 0, True, read_digest, 12, 12)
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


def test_extractor_leaves_a_pair_the_classifier_already_typed_alone(extractor):
    """#1246 clause 1. `edges.add` dedupes on the exact (source, target, type),
    so a `mentions` row beside an active `uses` was accepted, and the next
    nightly apply re-typed the pair to the type it already carried — 165 such
    pairs on 2026-09-20, every one a no-op retype. The typed verdict stays,
    the extractor adds nothing beside it, and it counts what it skipped."""
    st = kg_store.store()
    e = extractor
    for name in ("Lloyd", "vLLM", "Isaac Lab"):
        st.entities.register(name)
        (e.facts_dir / name).mkdir(exist_ok=True)
    typed_id = st.edges.add({
        "source": "Lloyd", "target": "vLLM", "type": "uses", "confidence": 0.9,
        "provenance": "EXTRACTED_CLASSIFIER_V4"}, origin="classifier")

    e.write_fact_file(
        "Lloyd", "relationship",
        {"facts": [_fact("Lloyd serves models through vLLM and trains in Isaac Lab",
                         category="relationship")]},
        source_doc="knowledge/lloyd.md",
    )

    on_pair = st.edges.active(source="Lloyd", target="vLLM")
    assert [(x["id"], x["type"]) for x in on_pair] == [(typed_id, "uses")], (
        f"the typed pair gained a row: {on_pair}")
    # Clause 2's other half in the same run: the untyped pair still gets its edge.
    assert st.edges.find_active("Lloyd", "Isaac Lab", "mentions") is not None
    assert e.link_stats == {"mentions_linked": 1, "mentions_skipped_typed": 1}


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


# ── a fact that quotes the fence (#1400) ─────────────────────────────────────

FENCE_FACT = ("OKF uses UTF-8 markdown files with YAML frontmatter delimited by "
              "'---' on the first line and a closing '---' before the body.")
CODE_FACT = "Obsidian uses inline `---` code spans to show a fence."


def _facts_in_file(path):
    return kg_store.parse_fact_file(path)[1]


def test_a_fence_quoting_fact_survives_the_next_write_whole(extractor):
    """`split("---", 2)` cut the YAML inside the quoted scalar; the truncated
    list was re-dumped, leaving "…delimited by '" with no provenance."""
    e = extractor
    e.write_fact_file("OKF", "state", {"facts": [_fact(FENCE_FACT)]},
                      source_doc="knowledge/okf.md")
    target = e.facts_dir / "OKF" / "OKF-state.md"
    e.write_fact_file("OKF", "state", {"facts": [_fact("OKF is versioned.")]},
                      source_doc="knowledge/okf.md")
    texts = [f["fact"] for f in _facts_in_file(target)]
    assert FENCE_FACT in texts, texts
    assert not any(t != FENCE_FACT and t.startswith("OKF uses UTF-8") for t in texts)


def test_six_facts_with_one_fence_quote_become_seven_all_with_provenance(extractor):
    e = extractor
    first = [_fact(f"plain fact number {i}") for i in range(2)]
    first += [_fact(FENCE_FACT), _fact(CODE_FACT)]
    first += [_fact(f"later fact number {i}") for i in range(2)]
    e.write_fact_file("OKF", "state", {"facts": first}, source_doc="knowledge/okf.md")
    e.write_fact_file("OKF", "state", {"facts": [_fact("the seventh fact")]},
                      source_doc="knowledge/other.md")
    target = e.facts_dir / "OKF" / "OKF-state.md"
    facts = _facts_in_file(target)
    assert len(facts) == 7, [f["fact"] for f in facts]
    for f in facts:
        assert f.get("created_at") and f.get("source_doc"), f
    rows = kg_store.store().facts_idx.for_entity("OKF")
    assert len(rows) == 7
    for r in rows:
        assert r["created_at"] and r["source_doc"], r


def test_the_extractors_read_agrees_with_the_stores_reader(extractor):
    e = extractor
    e.write_fact_file("OKF", "state", {"facts": [_fact("a"), _fact(FENCE_FACT),
                                                 _fact(CODE_FACT), _fact("z")]})
    target = e.facts_dir / "OKF" / "OKF-state.md"
    ours = [f["fact"] for f in e._read_existing_facts(target)]
    theirs = [f["fact"] for f in kg_store.parse_fact_file(target)[1]]
    assert ours == theirs and len(ours) == 4


def test_existing_facts_block_carries_the_whole_fence_quoting_fact(extractor):
    e = extractor
    e.write_fact_file("OKF", "state", {"facts": [_fact(FENCE_FACT), _fact("after it")]})
    block = e.get_existing_facts("OKF", "state")
    # yaml.dump may fold a long scalar; compare on whitespace-normalised text
    assert " ".join(FENCE_FACT.split()) in " ".join(block.split())
    assert "after it" in block


def test_an_opening_fence_with_no_closing_fence_is_still_quarantined(extractor):
    e = extractor
    d = e.facts_dir / "OKF"; d.mkdir()
    target = d / "OKF-state.md"
    target.write_text("---\nentity: OKF\nfacts:\n- fact: 'no closing --- line'\n")
    assert e.write_fact_file("OKF", "state", {"facts": [_fact("x")]}) is None
    assert not target.exists()
    assert list(d.glob("*.corrupt-*"))


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


# ── self-written derived indexes (#487) ──────────────────────────────────────

NE_PATH = "scripts/memory/next-gen-memory/nightly_extraction.py"
# Reports whose generators died in 0b3f00b ("kg phase 7: delete what nothing
# runs", 2026-09-03). Nothing regenerates them; nothing read them either.
DEAD_DERIVED_INDEXES = ("semantic-relationships.md", "relationship-proposals.md")
VAULT = Path.home() / "obsidian"


def test_exclusion_set_holds_only_the_live_writer():
    """Clauses 2 and 3 (the set itself) — neither dead name in
    `_SELF_WRITTEN_MEMORY_NOTES`, `skills-index.md` still in it.

    A name belongs in `_SELF_WRITTEN_MEMORY_NOTES` only while something
    still regenerates it. Both dead reports stayed in the set after their
    generators were deleted, so the extractor kept skipping files it had
    stopped writing, and the relationship counts those dead `type: report`
    notes asserted in `memory/` were never recomputed by anything (#487)."""
    ne = _load("nightly_extraction_exclset", NE_PATH)
    for name in DEAD_DERIVED_INDEXES:
        assert name not in ne._SELF_WRITTEN_MEMORY_NOTES, (
            f"{name} has no generator left; excluding it protects a dead file "
            "instead of letting a stale one be noticed")
    assert "skills-index.md" in ne._SELF_WRITTEN_MEMORY_NOTES, \
        ("skills-index.md still has a live writer (memory/skills-index.md, "
         "written nightly), so the basename entry stays. It is defence in "
         "depth for roots other than `memory/`, not the sole reason the live "
         "file stays out: `nightly_extraction.py:401` also drops any `memory/` "
         "note whose stem sorts >= today, and 'skills-index' always does. The "
         "behaviour that depends on THIS entry is pinned by "
         "test_the_dead_names_stop_being_excluded_from_the_corpus, under a "
         "non-`memory/` root where the date rule cannot interfere")


def test_the_dead_names_stop_being_excluded_from_the_corpus(tmp_path, monkeypatch):
    """Clauses 2 and 3 (as behaviour) — the extractor's eligible set no longer
    drops either dead name, and still drops `skills-index.md`.

    The same claim as behaviour, not as set membership: a file under either
    dead name is now an ordinary document, while `skills-index.md` — the one
    entry with a live writer — must still never enter the corpus.

    The fixtures sit under `knowledge/`, not `memory/`: the walk also drops any
    `memory/` file whose stem sorts >= today's date, and `'semantic…' >
    '2026-09-13'`, so in their real directory every one of these names would
    be excluded by that rule whatever the set contained — the check would
    report a verdict it could not see. The set matches on basename, so
    anywhere else exercises the identical branch.

    Why the store guard here is `try/finally` and the four older sites in this
    file are bare `kg_store.configure(...)` / `kg_store.reset()` statements
    (:43, :149, :322, :334): `tests/conftest.py` `_isolate_default_store` is an
    autouse function-scoped fixture that calls `kg_store.reset()` before every
    test, so a leak at those sites is cleared by the *next* test rather than by
    the one that leaked — survivable, but only to a reader who already knows
    the fixture exists. This test covers itself so that knowledge is not a
    precondition for reading it safely.
    """
    ne = _load("nightly_extraction_deadnames", NE_PATH)
    vault = tmp_path / "obsidian"
    for rel in ("knowledge/skills-index.md", "knowledge/semantic-relationships.md",
                "knowledge/relationship-proposals.md", "knowledge/control.md"):
        p = vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    cfg = tmp_path / "pipeline_config.yaml"
    cfg.write_text(yaml.dump({"sources": {"paths": ["knowledge"]}}))
    monkeypatch.setattr(ne, "VAULT", vault)
    monkeypatch.setattr(ne, "CONFIG_PATH", cfg)
    kg_store.configure(tmp_path / "kg.sqlite")
    try:
        x = ne.NightlyExtraction()
        got = {Path(p).name for p in x._eligible_files(full_mode=True)}
    finally:
        # `finally`, not the next statement: `_eligible_files` raises on a bad
        # config, and a raise here would otherwise leave the temp store
        # configured for every later test in the process — a leaked fixture
        # turns an unrelated failure into this test's name.
        kg_store.reset()
    assert got == {"semantic-relationships.md", "relationship-proposals.md",
                   "control.md"}, (
        "the dead reports are ordinary documents now; skills-index.md is the "
        "only self-written index that may stay out of the corpus")


def test_nothing_else_in_the_checkout_names_the_removed_notes():
    """Clause 4 — removing the two entries breaks no reference to them.

    An exclusion-set entry is just a name, so the thing that breaks when it
    goes is a *spelling*: a second deny-list, a report generator, or a job that
    reads one of the paths by hand. `0b3f00b` deleted the generators and left
    nothing pointing at either file, which is why this removal is safe; this
    pins that no future reference creeps back in. It is the suite half of the
    clause — the rest is the two tests above plus the pre-existing
    `_eligible_files` tests, which pass unchanged.

    `git grep` over tracked content is the denominator, not an `rglob`: the
    tracked checkout is what a reader or a scheduled job can be pointed at, and
    `_pipeline/` is ignored by git, so the generated artefacts that legitimately
    carry these names cannot inflate the count. Within the tracked tree the
    scope is ENUMERATED, not defaulted, and the enumeration is the whole
    claim — `scripts/`, `app/`, `workers/`, `*.py`, `*.yaml`, `*.sh`. Not
    "all code and config": a pointer spelled in a file type outside the list
    would pass uncaught, and files of any kind inside the three swept
    directories are in scope. Prose and pathspec name ONE list on purpose;
    the earlier phrasing ("code and config only") was an adjective over a
    list, and adjectives are how denominators drift. Everything
    prose-shaped stays out.
    Markdown, because prose is exactly where this removal gets recorded — an
    architecture note or changelog naming the two files by name is the
    *expected* outcome, and a fact-extractor test must not fail one. Data
    corpora for the same reason in a different extension: the first
    implementation grepped `.` minus `*.md` rather than the enumerated
    surfaces, and two eval corpora that landed in September 2026 read as
    references — `eval/djev/name_prior_corpus.jsonl` (one dedupe row, line 9,
    quoting a backlog body that names both files) and
    `eval/iv/recovered-2026-08-22..09-21.jsonl` (eight recovered-transcript
    lines, 77–84, quoting work logs from before the deletion). Neither points
    at a path; a name quoted inside a corpus row is data the eval reads, not
    code the checkout runs. Exit status 1 is git's "no matches" — a real
    answer, distinct from a failed git call, which is asserted rather than
    swallowed.
    """
    root = Path(__file__).resolve().parents[1]
    me = str(Path(__file__).resolve().relative_to(root))
    probe = ["git", "-C", str(root), "grep", "-l", "--fixed-strings",
             "-e", DEAD_DERIVED_INDEXES[0], "-e", DEAD_DERIVED_INDEXES[1]]
    # One pathspec shared by the control and the check below: two hand copies
    # of a denominator is exactly how this grep drifted from the surfaces its
    # own docstring enumerates.
    surfaces = ["scripts", "app", "workers", "*.py", "*.yaml", "*.sh",
                ":(exclude)*.md"]

    # Positive control first: this file names both strings and is a `*.py`
    # surface, so a grep that cannot see it cannot report an honest "nothing
    # else does". Without this the empty result below would be
    # indistinguishable from a silently broken call, and a check that cannot
    # fail is not a pin.
    control = subprocess.run(probe + ["--", *surfaces],
                             capture_output=True, text=True)
    assert control.returncode == 0 and me in control.stdout.split(), (
        f"the reference grep cannot see its own file (rc={control.returncode}, "
        f"{control.stdout.split()}); the empty-result assertion below would be "
        "unevaluable")

    hits = subprocess.run(
        probe + ["--", *surfaces, f":(exclude){me}"],
        capture_output=True, text=True)
    assert hits.returncode in (0, 1), (
        f"git grep failed ({hits.stderr.strip()[:160]}); the reference check is "
        "unevaluable and must not report a pass")
    others = sorted(hits.stdout.split())
    assert others == [], (
        f"{others} name a note whose generator died in 0b3f00b and which the "
        "extractor no longer excludes — whatever reads that path is reading a "
        "file nothing writes (#487)")


@pytest.mark.live_vault
def test_the_two_dead_reports_are_out_of_the_vault():
    """Clause 1 — neither report exists in the vault working tree.

    The mechanism for this clause is `automod_vault_land`, which landed vault
    commit `e4ad7729`; a code round's diff cannot carry a vault path, so this
    is the pin, and it reads the tree a reader actually opens. Marked
    `live_vault` per pytest.ini: the gate's `tests` rung runs `-m "not
    live_vault"` because a live-vault assertion judges whatever the previous
    writer left, not the candidate under test — run it directly, or through
    the vault route that writes it.

    Two things are asserted, because absence alone is the weaker claim: a file
    deleted in a dirty tree comes back on the next `git checkout`, so the
    deletion has to be committed as well as present. The directory is asserted
    first — a check that cannot see its own input reports a pass it cannot
    justify, and `ls-files` answers the empty string for a broken git call just
    as it does for a deleted file.
    """
    mem = VAULT / "memory"
    assert mem.is_dir(), f"{mem} is not readable — this check cannot report"
    for name in DEAD_DERIVED_INDEXES:
        assert not (mem / name).exists(), (
            f"memory/{name} is back: a report no generator writes, asserting a "
            "relationship count nothing recomputes (#487)")
        tracked = subprocess.run(
            ["git", "-C", str(VAULT), "ls-files", "--", f"memory/{name}"],
            capture_output=True, text=True)
        assert tracked.returncode == 0, (
            f"git ls-files failed ({tracked.stderr.strip()[:120]}); the "
            "committed-deletion half of this check is unevaluable")
        assert tracked.stdout.strip() == "", (
            f"memory/{name} is still tracked in the vault: an untracked "
            "deletion is one `git checkout` away from restoring it")


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


# ── the content-hash gate records the bytes that were extracted (#482) ──────
#
# `update_hash` re-read the file, so what the gate stored for a document was
# the digest of whatever it looked like at FLUSH time — and for a run under
# `CHECKPOINT_EVERY = 25` the only flush is the final one, so the window is the
# whole run: 545-1666s, the durations quoted at
# tests/test_extraction_single_instance.py:11-13. Anything appended to a note
# inside that window was recorded as already extracted and never was. The
# provenance half of the same hazard is fixed in `app/atomic_io.py::hash_bytes`;
# these pin the gate half.
#
# Clause key (clause N of the four graded acceptance clauses of #482):
#   1 digest-or-reread, CLI route unchanged →
#     test_update_hash_records_the_digest_it_was_given,
#     test_update_hashes_takes_path_digest_pairs_and_bare_paths,
#     test_the_cli_update_route_still_hashes_files_it_never_extracted
#   2 the swallow is closed →
#     test_an_append_after_the_digest_is_recorded_still_reads_as_changed
#   3 the index holds the digest read, workers=1 and workers=2 →
#     test_the_gate_stores_the_digest_of_the_bytes_extracted[1] / [2]
#   4 a raised ExtractionFailed gets no index entry, both branches, and the
#     return shape is pinned →
#     test_a_failed_extraction_still_gets_no_index_entry[1] / [2],
#     test_a_failed_file_is_not_hashed
#

def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_index(tmp_path, tag):
    """The real ContentHasher, loaded fresh, over an index of the test's own.

    `index_path` is passed explicitly rather than through LLOYD_CONTENT_HASHES
    so no test can write `_pipeline/content-hashes.json`, which is the resume
    point of a live extractor run.
    """
    ch = _load(f"content_hasher_{tag}", "scripts/memory/content_hasher.py")
    return ch, tmp_path / f"gate-{tag}.json"


def _extraction_run(tmp_path, monkeypatch, tag):
    """A NightlyExtraction that never calls the model, gated on a tmp index.

    `_eligible_files` takes its corpus from `pipeline_config.yaml`; the harness
    points it at `docs/` under a patched `VAULT`, so the corpus is exactly the
    files the test wrote and the flush path is the real one.
    """
    ne = _load(f"nightly_extraction_{tag}",
               "scripts/memory/next-gen-memory/nightly_extraction.py")
    kg_store.configure(tmp_path / "kg.sqlite")
    facts = tmp_path / "facts"
    facts.mkdir()
    monkeypatch.setattr(fx, "FACTS_DIR", facts)
    x = ne.NightlyExtraction()
    x.extractor.facts_dir = facts
    monkeypatch.setattr(ne, "VAULT", tmp_path)
    monkeypatch.setattr(ne, "_load_pipeline_config",
                        lambda: {"sources": {"paths": ["docs"]}})
    ch, idx = _gate_index(tmp_path, tag)
    monkeypatch.setattr(ne, "ContentHasher",
                        lambda: ch.ContentHasher(index_path=idx))
    (tmp_path / "docs").mkdir()
    return x, ch, idx


def _one_fact(md_file, *a, **k):
    name = Path(md_file).name
    return {"entity": "Gate", "category": "state",
            "facts": [{"fact": f"{name} was extracted through the gate",
                       "confidence": 0.9}]}


def test_update_hash_records_the_digest_it_was_given(tmp_path):
    """Clause 1: a caller that already hashed the bytes must not have them
    re-read — and a caller with nothing to offer still gets the file hashed."""
    ch, idx = _gate_index(tmp_path, "supplied")
    h = ch.ContentHasher(index_path=idx)
    doc = tmp_path / "note.md"
    doc.write_bytes(b"the bytes that were read\n")
    digest = _sha(doc)
    doc.write_bytes(b"the bytes that were read\nplus a line appended mid-run\n")

    h.update_hash(doc, digest)
    h.save()
    fresh = ch.ContentHasher(index_path=idx)
    assert fresh._hashes[str(doc)]["sha256"] == digest, (
        "the gate stored a re-read of the file, not the digest it was handed")
    assert fresh.has_changed(doc), (
        "the appended line is marked extracted and will never be fact-extracted")

    # With no digest the file on disk is hashed. That is the CLI `--update`
    # route (content_hasher.py:141), which hashes files it never extracted.
    cli = tmp_path / "cli.md"
    cli.write_bytes(b"hashed from disk, never extracted\n")
    h.update_hash(cli)
    h.save()
    fresh = ch.ContentHasher(index_path=idx)
    assert fresh._hashes[str(cli)]["sha256"] == _sha(cli)
    assert not fresh.has_changed(cli)


def test_update_hashes_takes_path_digest_pairs_and_bare_paths(tmp_path):
    """Clause 1 again, one level up: `update_hashes` takes the `(path,
    digest)` pairs the extraction checkpoint now carries, and a bare path still
    means "hash this file as it is on disk", which is what the CLI route
    passes."""
    ch, idx = _gate_index(tmp_path, "pairs")
    h = ch.ContentHasher(index_path=idx)
    carried = tmp_path / "carried.md"
    carried.write_bytes(b"v1\n")
    digest_of_read = _sha(carried)
    carried.write_bytes(b"v1\nv2 appended after the read\n")
    bare = tmp_path / "bare.md"
    bare.write_bytes(b"only ever hashed from disk\n")

    h.update_hashes([(carried, digest_of_read), bare])
    h.save()
    fresh = ch.ContentHasher(index_path=idx)
    assert fresh._hashes[str(carried)]["sha256"] == digest_of_read
    assert fresh._hashes[str(bare)]["sha256"] == _sha(bare)
    assert fresh.has_changed(carried)
    assert not fresh.has_changed(bare)


def test_an_append_after_the_digest_is_recorded_still_reads_as_changed(tmp_path):
    """Clause 2: record, then append, then save and reload. Against the
    pre-fix tree the digest is dropped and the appended-to file is what gets
    recorded, so a fresh hasher reports it unchanged forever."""
    ch, idx = _gate_index(tmp_path, "swallow")
    h = ch.ContentHasher(index_path=idx)
    doc = tmp_path / "daily.md"
    doc.write_bytes(b"morning entry\n")
    h.update_hash(doc, _sha(doc))
    doc.write_bytes(b"morning entry\nevening entry, appended after the digest\n")
    h.save()
    assert ch.ContentHasher(index_path=idx).has_changed(doc), (
        "the evening entry is permanently marked extracted")


@pytest.mark.parametrize("workers", [1, 2])
def test_the_gate_stores_the_digest_of_the_bytes_extracted(tmp_path, monkeypatch,
                                                          workers):
    """Clause 3: end to end through `_extract_all_facts`, in the sequential and
    the parallel branch both, a note appended to while the run is still walking
    the vault must leave the gate holding the digest of the bytes that were
    read — and therefore must still read as changed on the next run."""
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, f"gate{workers}")
    docs = []
    for i in range(3):
        d = tmp_path / "docs" / f"n{i}.md"
        d.write_text(f"note {i} says something worth keeping\n", encoding="utf-8")
        docs.append(d)
    read = {d: _sha(d) for d in docs}

    def append_mid_run(md_file, content, existing_facts="", start_offset=0):
        # `start_offset` is the resume point #1151 added to the extractor's
        # seam; a double that ignored it could not tell a resumed pass from a
        # restart, which is what the coverage tests are for.
        # The vault is live: someone appends between the read and the flush.
        Path(md_file).write_text(content + "\nappended mid-run\n", encoding="utf-8")
        return _one_fact(md_file)

    monkeypatch.setattr(x.extractor, "extract_from_document", append_mid_run)
    x._extract_all_facts(full_mode=False, workers=workers)

    stored = json.loads(idx.read_text())["hashes"]
    fresh = ch.ContentHasher(index_path=idx)
    for d in docs:
        assert stored[str(d)]["sha256"] == read[d], (
            f"{d}: the gate holds a re-read taken after the append, not the "
            f"digest of the {read[d][:8]} bytes that were extracted")
        assert fresh.has_changed(d), (
            f"{d}: the mid-run append is marked extracted and never re-processed")
    assert x.last_files_processed == 3
    kg_store.reset()


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failed_extraction_still_gets_no_index_entry(tmp_path, monkeypatch,
                                                      workers):
    """Clause 4: `ok=False` must keep a file out of `pending` whatever the new
    signature is — hashing a document whose extraction raised marks a transient
    vLLM error as done forever."""
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, f"fail{workers}")
    good = tmp_path / "docs" / "good.md"
    good.write_text("the model answered for this one\n", encoding="utf-8")
    bad = tmp_path / "docs" / "bad.md"
    bad.write_text("the model wedged on this one\n", encoding="utf-8")

    def wedge(md_file, content, existing_facts="", start_offset=0):
        # `start_offset` is the resume point #1151 added to the extractor's
        # seam; a double that ignored it could not tell a resumed pass from a
        # restart, which is what the coverage tests are for.
        if Path(md_file).name == "bad.md":
            raise fx.ExtractionFailed("vLLM wedged")
        return _one_fact(md_file)

    monkeypatch.setattr(x.extractor, "extract_from_document", wedge)
    x._extract_all_facts(full_mode=False, workers=workers)

    stored = json.loads(idx.read_text())["hashes"]
    assert str(good) in stored
    assert str(bad) not in stored, (
        "a failed extraction reached the index: bad.md is marked done forever")
    assert x.last_failed_files == 1
    kg_store.reset()


def test_the_cli_update_route_still_hashes_files_it_never_extracted(tmp_path):
    """The one caller with no digest to offer is the CLI, and it is a separate
    process: `--update` walks a directory and hashes everything it finds. Run
    as the process it is, so a signature change that broke it is caught at the
    boundary rather than by re-calling the same Python function."""
    idx = tmp_path / "cli-index.json"
    scan = tmp_path / "scan"
    scan.mkdir()
    note = scan / "a.md"
    note.write_text("alpha\n", encoding="utf-8")

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "memory" / "content_hasher.py"),
         str(scan), "--update"],
        capture_output=True, text=True, timeout=120,
        env=dict(os.environ, LLOYD_CONTENT_HASHES=str(idx)))
    assert r.returncode == 0, r.stderr[-2000:]
    stored = json.loads(idx.read_text())["hashes"]
    assert stored[str(note)]["sha256"] == _sha(note), (
        "`--update` recorded something other than the file on disk; it has no "
        "digest to carry and must keep hashing")


# ── backlog-citation entities (#743) ────────────────────────────────────────
# Prose that says "see backlog #1095" names a row in a tracker, not a thing the
# graph knows about. The extractor minted an entity for every such number: the
# The 2026-09-15 triage measured 187 rows carrying 641 active facts, every one
# `provenance='EXTRACTED'`, 12 of them created on 09-13 alone, and new rows were
# still being created on 2026-09-22, the day this rule was written — a live mint,
# not legacy. (Live counts move daily and cannot be re-measured from a worktree;
# the post-landing check is rows created after the landing sha, not a flat
# total.) The name is a plain multi-word Capitalized string with no extension, so
# every extension-based rule in `looks_like_junk_entity` is blind to it, and
# `task #N` was the only tracker shape the pipeline-exhaust rule covered.

#: Every whole-name spelling the clause demands refused.
CITATION_SPELLINGS = (
    "Backlog Item #338",   # triage's own example
    "backlog item 338",    # no sigil, all lowercase
    "backlog #1095",       # the newest row at triage, created 2026-09-15
    "Backlog #945",        # bare board noun plus a sigil
    "Board Item #12",      # the other tracker noun
)

#: The escape hatch `looks_like_junk_entity` carries, exercised on both sides.
PROJECT_NOTE = "projects/lloyd/voice/x.md"


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    """The candidates sidecar, redirected off the real tree."""
    p = tmp_path / "entity-candidates.jsonl"
    monkeypatch.setattr(en, "ENTITY_CANDIDATES_PATH", p)
    # The seen-name set is process-wide, so a name an earlier test recorded
    # would be silently dropped here and this file's sidecar assertions would
    # pass on nothing.
    monkeypatch.setattr(en, "_KNOWN_CANDIDATES", set())
    monkeypatch.setattr(en, "_KNOWN_CANDIDATES_LOADED", False)
    en.reset_identity_schema_cache()
    yield p
    en.reset_identity_schema_cache()


def _candidates(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _answer(prompt_payload):
    """A stand-in `_call_llm` that answers every chunk with one canned JSON."""
    body = json.dumps(prompt_payload)
    return lambda prompt: body


# ── clause 1: the predicate refuses the whole name, in every spelling ────────

def test_a_backlog_citation_is_junk_in_every_spelling():
    for name in CITATION_SPELLINGS:
        assert en.looks_like_junk_entity(name) is True, name
    assert en.is_valid_entity_name("Backlog Item #338") is False


# ── clause 2: no projects/ exemption for a citation ──────────────────────────

def test_the_citation_rule_is_not_exempted_by_a_project_note():
    """`_EXHAUST_RE` stands down when `source_doc` starts with `projects/`,
    because a project note may legitimately be about a task. A citation is a
    citation whoever wrote the note — two of the rows counted at triage came
    from `projects/` prose — so this rule ignores the argument."""
    for source_doc in (None, "", PROJECT_NOTE):
        for name in CITATION_SPELLINGS:
            assert en.looks_like_junk_entity(name, source_doc) is True, (name, source_doc)
    # Positive control that the argument is not simply ignored everywhere, which
    # would make the loop above pass without the citation rule existing: the
    # exhaust rule still bends for a project note.
    assert en.looks_like_junk_entity("Sweep Run 313") is True
    assert en.looks_like_junk_entity("Sweep Run 313", PROJECT_NOTE) is False


# ── clause 3: precision ──────────────────────────────────────────────────────

def test_the_citation_rule_keeps_the_names_the_graph_and_the_eval_need():
    """Digits are required and the pattern is anchored at both ends, because
    `Backlog System` is the gold entity for the backlog-overview eval, and
    `Backlog` / `Backlog Item` are live digitless entities the 2026-09-15 triage
    counted at 82 and 89 active facts."""
    for name in ("Backlog", "Backlog Item", "Backlog System", "Lloyd Backlog",
                 "Autonomy Backlog Pipeline"):
        assert en.looks_like_junk_entity(name) is False, name
    # A citation inside a longer name is a mention of one, not the entity's own
    # name, so it stays writable.
    assert en.looks_like_junk_entity("Summary for backlog item #338 in review") is False
    assert en.looks_like_junk_entity("Triage of backlog item #338") is False


def test_a_long_name_merely_containing_a_citation_is_not_refused_by_this_rule():
    """Nine words is a sentence to this module, and this name is nine words: the
    composite predicate does refuse it, but by the pre-existing
    whole-name-is-a-sentence rule (`len(s.split()) >= 8`), not by the citation
    rule, which this change does not let reach that far. The citation sits
    mid-string, so only an anchored whole name is a citation."""
    name = "Triage of backlog item #338 in the nightly chain"
    assert en.is_backlog_citation_entity(name) is False
    assert en.looks_like_junk_entity(name) is True


# ── clause 4: the mint is refused before the typed_new branch ────────────────

def test_extraction_refuses_a_fresh_backlog_citation_typed_as_a_task(extractor, sidecar,
                                                                     monkeypatch):
    """The model may assert any legal `entity_type` — the extraction prompt
    demands one for every fact and `task` is a legal `SCHEMA_TYPES` value, which
    is precisely how these rows were minted — and the name still must not reach
    `gate_entity_name`'s `typed_new` branch, and must not be dropped silently."""
    e = extractor
    citation = "Backlog Item #744001"
    monkeypatch.setattr(e, "_call_llm", _answer({
        "entity": "Lloyd", "entity_type": "system", "category": "state",
        "facts": [_fact("the triage note cites a board row",
                        entity=citation, entity_type="task")]}))

    out = e.extract_from_document(Path(PROJECT_NOTE), "cites backlog item 744001\n")

    assert out["facts"] == [], "the citation fact was filed anyway"
    assert not (e.facts_dir / citation).exists()
    assert kg_store.store().entities.lookup(citation) is None
    cands = _candidates(sidecar)
    assert [c["name"] for c in cands] == [citation]
    assert cands[0]["declared_type"] == "task"
    assert "citation" in cands[0]["reason"]
    # A caller that skips extraction is refused a second time, and under a
    # projects/ source_doc, which is the write_fact_file route's `enforce=False`.
    assert e.write_fact_file(citation, "state", {"facts": [_fact("x")]},
                             source_doc=PROJECT_NOTE) is None
    assert not (e.facts_dir / citation).exists()


def test_the_gate_itself_refuses_a_citation_before_its_typed_new_branch(extractor,
                                                                        sidecar):
    """The seam one level down: `gate_entity_name` is the mint site, so no
    caller reaching it directly — with a legal type, from any source document —
    can register the name either."""
    for name, declared in (("Board Item #744002", "task"),
                           ("backlog #744003", "concept"),
                           ("Backlog Item #744004", None)):
        entity, verdict = en.gate_entity_name(name, declared_type=declared,
                                              source_doc=PROJECT_NOTE)
        assert (entity, verdict) == ("", "candidate"), name
        assert kg_store.store().entities.lookup(name) is None
    assert sorted(c["name"] for c in _candidates(sidecar)) == [
        "Backlog Item #744004", "Board Item #744002", "backlog #744003"]


def test_the_mcp_fact_add_route_refuses_a_citation_name(tmp_path, monkeypatch):
    """The seam one process further out. `fact_add` runs in the MCP server
    process, and its own junk check (`agent_mcp/facts.py:455`) is the predicate
    clause 1 widens — so a caller filing interactively under a whole-name
    citation is now refused where it previously registered a row. Driven through
    the real handler, because the facts tree and the store are on the far side of
    that boundary and a call to the predicate is not a call across it.

    The handler resolves every path from ITS OWN module-global `FACTS_ROOT`
    (imported from `agent_mcp._shared`, which `app.paths` anchors at the real
    vault facts tree), so both copies are redirected here. Asserting against the
    extraction fixture's directory instead would be an assertion that cannot
    fail: nothing on this path ever writes there.

    The second half is the positive control, and it is load-bearing: one accepted
    non-citation name proves the directory and store assertions below CAN fail,
    which is the only thing that makes their negations evidence."""
    from agent_mcp import _shared, facts as mcp_facts

    root = tmp_path / "mcp-facts"
    root.mkdir()
    kg_store.configure(tmp_path / "kg.sqlite")
    monkeypatch.setattr(_shared, "FACTS_ROOT", root)
    monkeypatch.setattr(mcp_facts, "FACTS_ROOT", root)
    _shared._invalidate_entity_dirs_cache()

    control = "Zephyr Widget Framework"
    ok = mcp_facts._fact_add({"entity": control, "category": "state",
                              "fact": "the control entity got its fact"})
    assert ok.get("success") and "error" not in ok, ok
    assert (root / control).is_dir(), "the control wrote no directory, so the " \
        "assertions below would pass on an unwritable tree and prove nothing"
    assert kg_store.store().entities.lookup(control) is not None

    citation = "Backlog Item #744005"
    result = mcp_facts._fact_add({"entity": citation, "category": "state",
                                  "fact": "the triage note cites a board row",
                                  "source_doc": PROJECT_NOTE})
    assert result.get("code") == "INVALID_PARAM", result
    assert citation in result["error"]
    assert not (root / citation).exists()
    assert kg_store.store().entities.lookup(citation) is None
    monkeypatch.undo()
    kg_store.reset()
    _shared._invalidate_entity_dirs_cache()


# ── clause 5: purpose preserved ──────────────────────────────────────────────

def test_extraction_still_mints_a_genuinely_new_typed_entity(extractor, sidecar,
                                                             monkeypatch):
    """Widening the junk predicate must not turn `typed_new` into a blanket
    refusal: an undeclared, non-citation entity named by the model with a legal
    type still gets its row and its fact file."""
    e = extractor
    monkeypatch.setattr(e, "_call_llm", _answer({
        "entity": "Lloyd", "entity_type": "system", "category": "state",
        "facts": [_fact("the router fronts the alert channel",
                        entity="Foghorn Signal Router", entity_type="system")]}))

    out = e.extract_from_document(Path("knowledge/software/x.md"), "the router\n")

    assert [f["entity"] for f in out["facts"]] == ["Foghorn Signal Router"]
    assert kg_store.store().entities.lookup("Foghorn Signal Router") == "Foghorn Signal Router"
    assert kg_store.store().entities.kinds().get("Foghorn Signal Router") == "system"
    written = e.write_fact_file("Foghorn Signal Router", "state",
                               {"facts": out["facts"]})
    assert written is not None and written.exists()
    assert _candidates(sidecar) == []


# ── per-document coverage: the chunk cap is a result, not a print (#1151) ────
#
# `FactExtractor._chunk_content` windows one document into at most `MAX_CHUNKS`
# chunks, so one pass over one document reads at most `CHUNK_BUDGET_CHARS`
# (47,000) chars. Until this section the shortfall left that function as a bare
# `print`: `_process_single_file` returned ok=True for a 955k-char feed, the run
# hashed it, and `ContentHasher.has_changed` skipped it forever — the unread
# tail was never revisited and `failed=0` read as full coverage. Live cost at
# triage: 18 allow-listed documents sit past the ceiling and 6-15 of them
# carried a stored sha equal to the file's live sha in
# `_pipeline/content-hashes.json` (`knowledge/tools/openclaw/updates.md`,
# 957,489 chars, 4.9 % of it ever sent to the model).
#
# Clause key (clause N of the five graded acceptance clauses of #1151):
#   1 coverage is a field of the result →
#     test_extraction_reports_the_chars_it_covered
#   2 the next pass starts at the offset where coverage ended →
#     test_a_truncated_document_resumes_where_coverage_ended
#   3 no complete record while the tail is unread, and coverage advances to a
#     fixed point → test_an_uncovered_tail_is_not_recorded_as_complete
#   4 `PIPELINE_RESULT` carries the shortfall →
#     tests/test_extraction_single_instance.py::
#         test_the_pipeline_result_names_documents_whose_tail_went_unread
#   5 the full-coverage claim is corrected at both sites →
#     test_the_full_coverage_claim_is_gone_from_both_sites,
#     test_one_pass_over_one_document_covers_the_budget_it_claims
#

DOC_CELL = 100                              # chars per cell of a synthetic doc
DOC_MARKER = "MARKER-PAST-ONE-CHUNK-PASS"    # 25 chars, marks the unread tail
ONE_PASS_CHARS = 47_000                     # (MAX_CHUNKS-1)*7800 + CHUNK_SIZE


def _tall_doc(n_chars: int, marker_at: int | None = None) -> str:
    """A document of exactly `n_chars` whose every 100-char cell is distinct.

    Distinct cells are what make a chunk's source offset recoverable: the
    stubbed model locates the chunk it was sent by searching the document for
    it, and a repeating filler would alias onto an earlier offset. `marker_at`
    plants `DOC_MARKER` inside the region one pass cannot reach, so a test can
    tell "resumed" from "restarted at 0".
    """
    doc = "".join(f"CELL{c:06d} " + "x" * (DOC_CELL - 11)
                  for c in range(-(-n_chars // DOC_CELL)))[:n_chars]
    if marker_at is not None:
        doc = (doc[:marker_at] + DOC_MARKER
               + doc[marker_at + len(DOC_MARKER):])
    assert len(doc) == n_chars, "the document must be exactly n_chars long"
    return doc


# The extraction template delimits the document block with these two headings.
CONTENT_HEAD = "\nContent:\n"
CONTENT_TAIL = "\n\nKnown entities"
NO_FACTS = '{"entity": "", "category": "state", "facts": []}'


def _model_that_records_offsets(content: str, log: list):
    """A stand-in `FactExtractor._call_llm` recording the source offset of every
    chunk it is sent, and answering with no facts.

    The offset comes back out of the prompt — the content block is sliced at the
    template's own headings and located in the document — so what these tests
    pin is what the model was *handed*, not what the chunker claims to have
    returned.
    """
    def _call(prompt: str) -> str:
        block = prompt.split(CONTENT_HEAD, 1)[1].split(CONTENT_TAIL, 1)[0]
        log.append((content.find(block), block))
        return NO_FACTS
    return _call


def test_one_pass_over_one_document_covers_the_budget_it_claims():
    """The ceiling every clause below is written against: 6 chunks of 8,000
    chars stepping 7,800 (200 chars of overlap carried across each cut) reach
    47,000 chars, which is why a 955k-char feed was 4.9 % covered."""
    step = fx.CHUNK_SIZE - fx.CHUNK_OVERLAP
    assert fx.CHUNK_BUDGET_CHARS == ONE_PASS_CHARS
    assert fx.CHUNK_BUDGET_CHARS == (fx.MAX_CHUNKS - 1) * step + fx.CHUNK_SIZE

    # ...and the chunker honours it: the constant is not the only thing the
    # clauses below rest on, so the last chunk actually handed over ends there.
    long_doc = _tall_doc(ONE_PASS_CHARS + 5_000)
    chunks = fx.FactExtractor._chunk_content(None, long_doc)
    assert len(chunks) == fx.MAX_CHUNKS
    assert chunks[-1][0] + len(chunks[-1][1]) == fx.CHUNK_BUDGET_CHARS, (
        "_chunk_content stops somewhere other than the budget the test names")
    short = _tall_doc(1_200)
    assert fx.FactExtractor._chunk_content(None, short) == [(0, short)], (
        "a document inside the budget is not returned whole from offset 0")


def test_extraction_reports_the_chars_it_covered(extractor, monkeypatch):
    """Clause 1: coverage is a field of the extraction result, not a line on
    stdout. `chars_total` is the whole document; `chars_covered` is the end
    offset of the last chunk that reached the model — the document's own length
    when it fits the budget, `CHUNK_BUDGET_CHARS` when it does not."""
    e = extractor
    monkeypatch.setattr(e, "_call_llm", lambda p: NO_FACTS)

    short = _tall_doc(3000)
    out = e.extract_from_document(Path("docs/short.md"), short)
    assert out["chars_total"] == len(short) == 3000
    assert out["chars_covered"] == 3000, "a doc inside the budget is fully covered"

    tall = _tall_doc(60_000, marker_at=50_000)
    out = e.extract_from_document(Path("docs/tall.md"), tall)
    assert out["chars_total"] == len(tall) == 60_000
    assert out["chars_covered"] == fx.CHUNK_BUDGET_CHARS == ONE_PASS_CHARS


def test_a_truncated_document_resumes_where_coverage_ended(extractor, monkeypatch):
    """Clause 2: the first pass over a 60,000-char document sends the six chunks
    starting at 0, 7800 … 39000 and stops at 47,000; the next pass, given the
    offset that pass reported, starts at 47,000. The marker planted at 50,000 is
    the proof: it is unread by pass 1 and read by pass 2, which a restart from 0
    could not do within the same budget."""
    e = extractor
    doc = _tall_doc(60_000, marker_at=50_000)
    step = fx.CHUNK_SIZE - fx.CHUNK_OVERLAP

    seen: list = []
    monkeypatch.setattr(e, "_call_llm", _model_that_records_offsets(doc, seen))
    first = e.extract_from_document(Path("docs/feed.md"), doc)
    assert [off for off, _ in seen] == [i * step for i in range(fx.MAX_CHUNKS)]
    assert first["chars_covered"] == ONE_PASS_CHARS
    assert not any(DOC_MARKER in b for _, b in seen), "one pass already read the tail"

    seen.clear()
    second = e.extract_from_document(Path("docs/feed.md"), doc,
                                     start_offset=first["chars_covered"])
    assert seen[0][0] == ONE_PASS_CHARS, (
        f"the resumed pass started at {seen[0][0]}, not at the offset pass 1 "
        "reported: coverage restarts and the tail is never reached")
    assert any(DOC_MARKER in b for _, b in seen), "the resumed pass never reached the marker"
    assert second["chars_covered"] == second["chars_total"] == 60_000


def test_an_uncovered_tail_is_not_recorded_as_complete(tmp_path, monkeypatch):
    """Clause 3, mirroring `test_a_failed_file_is_not_hashed`: a document whose
    tail went unread keeps reading as changed and its stored offset grows, and
    the pass that reaches the end records it complete. Neither skipped forever
    nor re-read from 0 forever — and a third run extracts nothing at all.
    """
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, "coverage")
    doc = tmp_path / "docs" / "feed.md"
    doc.write_text(_tall_doc(60_000, marker_at=50_000), encoding="utf-8")
    seen: list = []
    monkeypatch.setattr(x.extractor, "_call_llm",
                        _model_that_records_offsets(doc.read_text(encoding="utf-8"), seen))

    x._extract_all_facts(full_mode=False)
    stored = json.loads(idx.read_text())["hashes"][str(doc)]
    assert stored["covered_through"] == ONE_PASS_CHARS
    assert stored["complete"] is False
    assert ch.ContentHasher(index_path=idx).has_changed(doc), (
        "the unread tail is recorded as done: the 47,000 chars read are the "
        "whole document as far as the next run is concerned")
    assert x.last_truncated_files == ["docs/feed.md"]
    assert seen[0][0] == 0, "a document with no stored offset must start at 0"

    seen.clear()
    x._extract_all_facts(full_mode=False)
    assert seen[0][0] >= ONE_PASS_CHARS, "pass 2 re-read the prefix instead of resuming"
    stored = json.loads(idx.read_text())["hashes"][str(doc)]
    assert stored["covered_through"] == 60_000
    assert stored["complete"] is True
    assert not ch.ContentHasher(index_path=idx).has_changed(doc), (
        "the whole document has been read and the index still calls it changed: "
        "it would now be re-extracted every night forever")
    assert x.last_truncated_files == []

    seen.clear()
    x._extract_all_facts(full_mode=False)
    assert seen == [], "a covered, unchanged document was extracted a third time"
    kg_store.reset()


def test_a_shortened_document_restarts_coverage_at_zero(tmp_path, monkeypatch):
    """The resume offset is keyed to the recorded content hash, so a feed that
    is rewritten shorter cannot be "resumed" past its own new end and recorded
    complete with its tail unread — the same laundering one layer down. The
    digest no longer matches, so coverage restarts at 0."""
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, "shorten")
    doc = tmp_path / "docs" / "feed.md"
    doc.write_text(_tall_doc(60_000, marker_at=50_000), encoding="utf-8")
    seen: list = []
    monkeypatch.setattr(x.extractor, "_call_llm",
                        _model_that_records_offsets(doc.read_text(encoding="utf-8"), seen))
    x._extract_all_facts(full_mode=False)
    assert json.loads(idx.read_text())["hashes"][str(doc)]["covered_through"] == ONE_PASS_CHARS

    rewritten = _tall_doc(55_000)
    doc.write_text(rewritten, encoding="utf-8")
    monkeypatch.setattr(x.extractor, "_call_llm",
                        _model_that_records_offsets(rewritten, seen))
    seen.clear()
    x._extract_all_facts(full_mode=False)
    assert seen[0][0] == 0, (
        "coverage resumed into bytes of a document that no longer contains them")
    kg_store.reset()


def test_the_full_coverage_claim_is_gone_from_both_sites():
    """Clause 5: the comment above the chunk constants and the `_chunk_content`
    docstring both asserted that long docs get "full coverage", which was only
    true up to `MAX_CHUNKS`. Neither may keep the claim, and the docstring must
    name the budget that replaces it."""
    src = (ROOT / "scripts" / "memory" / "next-gen-memory"
           / "fact_extractor.py").read_text(encoding="utf-8")
    assert "full coverage" not in src.lower(), (
        "a full-coverage claim survives in fact_extractor.py: the next reader "
        "inherits it as a guarantee")
    assert "CHUNK_BUDGET_CHARS" in src, "the budget must be named where the claim was"
    doc = fx.FactExtractor._chunk_content.__doc__ or ""
    assert "MAX_CHUNKS" in doc and "CHUNK_BUDGET_CHARS" in doc, (
        "_chunk_content's docstring does not state the cap it enforces")


def test_a_digest_without_coverage_is_trusted_only_below_the_budget(tmp_path, monkeypatch):
    """The migration rule, pinned. Every entry in the live index was written
    before coverage was tracked and says nothing about how much of its file was
    read. Distrusting all of them would put the whole recorded corpus back through
    the model in one pass — every entry in the index, whatever its file's size;
    trusting all of them keeps the oversized feeds skipped forever. Size separates
    them: a document under `CHUNK_BUDGET_CHARS` could only
    have been read whole by the pass that recorded it, so its digest still means
    done.
    """
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, "migration")
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    short = docs / "short.md"
    short.write_text(_tall_doc(3000), encoding="utf-8")
    big = docs / "big.md"
    big.write_text(_tall_doc(52_000), encoding="utf-8")

    legacy = {str(p): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                       "checked_at": "2026-09-14T03:18:01Z"}
              for p in (short, big)}
    idx.write_text(json.dumps({"hashes": legacy}), encoding="utf-8")

    calls: list = []
    monkeypatch.setattr(x.extractor, "_call_llm",
                        lambda p: (calls.append(p), NO_FACTS)[1])
    x._extract_all_facts(full_mode=False)

    assert x.last_files_processed == 1, (
        "the short file's pre-#1151 digest was distrusted too: every legacy entry "
        "in the index just went back through the model")
    assert len(calls) == fx.MAX_CHUNKS, (
        "the oversized file was not given exactly one bounded pass")
    stored = json.loads(idx.read_text())["hashes"]
    assert stored[str(short)]["checked_at"] == "2026-09-14T03:18:01Z", (
        "the unchanged short file was rewritten anyway")
    assert stored[str(big)]["covered_through"] == 47_000, (
        "the oversized file kept a digest that asserts nothing and stays skipped")
    kg_store.reset()


def test_an_untrustworthy_record_re_enters_from_outside_the_window(tmp_path, monkeypatch):
    """A digest that cannot be trusted is owed work whatever the file's mtime.

    The nightly run reads files touched in the last 24 hours, which is a policy
    about *changed* documents and says nothing about a record that predates
    coverage tracking. `knowledge/tools/isaac-gr00t/prs.md` (53,966 bytes,
    hash-skipped with a live-matching digest at triage) is exactly that shape: a
    feed that stopped being edited has nothing left to re-trigger it, so under a
    window-only sweep it would sit unread forever and the acceptance check would
    still find it in the index. The short file beside it, equally stale, must
    stay out of the run — the window is untouched for every record that is
    judgeable.
    """
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, "stale")
    docs = tmp_path / "docs"
    stale_big = docs / "stale-feed.md"
    stale_big.write_text(_tall_doc(52_000), encoding="utf-8")
    stale_short = docs / "stale-note.md"
    stale_short.write_text(_tall_doc(2_000), encoding="utf-8")
    three_days_ago = time.time() - 3 * 86400
    for stale in (stale_big, stale_short):
        os.utime(stale, (three_days_ago, three_days_ago))
    legacy = {str(p): {"sha256": _sha(p), "checked_at": "2026-09-14T03:18:01Z"}
              for p in (stale_big, stale_short)}
    idx.write_text(json.dumps({"hashes": legacy}), encoding="utf-8")

    calls: list = []
    monkeypatch.setattr(x.extractor, "_call_llm",
                        lambda p: (calls.append(p), NO_FACTS)[1])
    x._extract_all_facts(full_mode=False)

    assert x.last_files_processed == 1, (
        "the window admitted the stale short file too, or the stale oversized one "
        "was skipped: " + str(x.last_truncated_files))
    assert x.last_truncated_files == ["docs/stale-feed.md"]
    assert len(calls) == fx.MAX_CHUNKS
    stored = json.loads(idx.read_text())["hashes"]
    assert stored[str(stale_big)]["covered_through"] == ONE_PASS_CHARS
    assert stored[str(stale_short)]["checked_at"] == "2026-09-14T03:18:01Z", (
        "a stale document whose record is judgeable was pulled into the run: the "
        "24-hour window is not what this sweep is for")
    kg_store.reset()


def test_a_capped_run_leaves_an_unreached_record_exactly_as_it_was(tmp_path, monkeypatch):
    """The migration sweep is a read, so `--limit` cannot cost a file its record.

    `--limit` truncates the eligible list *after* the gate. Re-admission used to
    be done by deleting the entry so `get_changed_files` would return its file,
    which meant the document the cap passed by had its pre-#1151 digest erased
    and came back next run as a path the index has never seen — the same
    judgment, made by amnesia instead of by size, and invisible to anyone
    reading the index. Both feeds here are oversized and coverage-less; a
    one-file run reads one of them and must leave the other's entry byte-identical.
    """
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, "cap")
    docs = tmp_path / "docs"
    feeds = [docs / "a-feed.md", docs / "b-feed.md"]
    for feed, size in zip(feeds, (52_000, 61_000)):
        feed.write_text(_tall_doc(size), encoding="utf-8")
    legacy = {str(p): {"sha256": _sha(p), "checked_at": "2026-09-14T03:18:01Z"}
              for p in feeds}
    idx.write_text(json.dumps({"hashes": legacy}), encoding="utf-8")

    monkeypatch.setattr(x.extractor, "_call_llm", lambda p: NO_FACTS)
    x._extract_all_facts(full_mode=False, limit=1)

    stored = json.loads(idx.read_text())["hashes"]
    read = [p for p in feeds if stored[str(p)].get("covered_through") is not None]
    untouched = [p for p in feeds if p not in read]
    assert len(read) == 1, (
        f"a limit=1 run was to read exactly one oversized file, read: {read}")
    assert stored[str(untouched[0])] == legacy[str(untouched[0])], (
        "the sweep erased the record of a file the cap never reached: the next "
        "run can no longer tell a pre-#1151 digest from a file it has never seen")
    assert x.last_truncated_files == [f"docs/{read[0].name}"]

    # Keep taking capped passes until both feeds are covered. Which one a given
    # pass reads is walk order, so the pin is the invariant rather than a
    # sequence: a pass changes exactly one file's record, and the file it did not
    # reach keeps the record it had — which is what the delete-to-re-admit sweep
    # broke, and what makes `--limit` safe to run against this index at all.
    before = json.loads(idx.read_text())["hashes"]
    for _ in range(4):
        if all(before[str(p)].get("complete") for p in feeds):
            break
        x._extract_all_facts(full_mode=False, limit=1)
        after = json.loads(idx.read_text())["hashes"]
        moved = [p for p in feeds if after[str(p)] != before[str(p)]]
        assert len(moved) == 1, (
            f"a limit=1 pass rewrote {len(moved)} of 2 records: {moved}")
        skipped = [p for p in feeds if p not in moved][0]
        assert after[str(skipped)] == before[str(skipped)], (
            "a pass that never reached a file changed its record")
        before = after

    assert all(before[str(p)].get("complete") for p in feeds), (
        "four one-file passes did not finish two 52k/61k documents, so coverage "
        "does not reach a fixed point under a cap: " + json.dumps(before))
    for feed in feeds:
        assert before[str(feed)]["covered_through"] == len(
            feed.read_text(encoding="utf-8")), (
            "a capped pass stopped short of the document's length yet recorded it "
            "complete")
    kg_store.reset()


# --------------------------------------------------------------------------- #
# The seam this change did not touch and had to (review of round SM_20260923_163408)
# --------------------------------------------------------------------------- #

def _kg_rebuild():
    """`scripts/memory/kg_rebuild.py`, loaded the way the other scripts here are.

    Its own suite (`tests/test_kg_rebuild_freeze.py`) does not load the module, so
    importing it from this file is what keeps the import the coverage fix needs
    (clause 3: a record with an unread tail is not a complete record) from silently
    breaking the rebuild path.
    """
    sys.path.insert(0, str(ROOT / "scripts" / "memory"))
    return _load("kg_rebuild_under_test", "scripts/memory/kg_rebuild.py")


def test_the_rebuild_extracted_count_refuses_a_record_whose_own_coverage_says_otherwise(tmp_path, monkeypatch):
    """`kg_rebuild` decides whether a rebuilt knowledge tree may be promoted from
    the count of documents extracted, read from the very index this step writes —
    and `scripts/memory/kg_rebuild.py` was not in the diff that gave entries a
    `complete` key, so the count can be satisfied by documents nobody read to the
    end. That is clause 3's rule at a second process boundary.

    The gate compares the count against `_corpus_size()` and requires 95 %
    (`GATE['corpus_coverage_pct']`), so the assertion is stated against that
    threshold rather than against a number of entries: with all four documents
    recorded complete the check sees 4 of 4, with the two capped passes recorded
    the way clause 3 requires it sees 2 of 4, and a gate that cannot tell those
    apart promotes a corpus built from unread tails.
    """
    ch, idx = _gate_index(tmp_path, "rebuild_gate")
    docs = tmp_path / "docs"
    docs.mkdir()
    whole = docs / "note.md"
    whole.write_text("a short note\n", encoding="utf-8")
    also_whole = docs / "other.md"
    also_whole.write_text("another short note\n", encoding="utf-8")
    capped = [docs / f"feed{i}.md" for i in (1, 2)]
    for feed in capped:
        feed.write_text(_tall_doc(ONE_PASS_CHARS + 900), encoding="utf-8")

    h = ch.ContentHasher(index_path=idx)
    for doc in (whole, also_whole):
        h.update_hash(doc, covered_through=len(doc.read_text(encoding="utf-8")),
                      complete=True)
    for feed in capped:
        h.update_hash(feed, covered_through=ONE_PASS_CHARS, complete=False)
    h.save()
    assert json.loads(idx.read_text())["file_count"] == 4, (
        "the index counts four documents; the point is that the scalar is not a "
        "count of documents *extracted*")

    kr = _kg_rebuild()
    monkeypatch.setattr(kr, "VAULT_DERIVED_ROOT", tmp_path)
    (tmp_path / "rebuild-content-hashes.json").write_text(
        idx.read_text(encoding="utf-8"), encoding="utf-8")

    assert kr._hashed_count() == 2, (
        "the promotion gate counted two documents whose own record says the tail "
        "is owed, so `corpus_coverage_pct` can be satisfied by unread text")
    assert kr._corpus_size() >= 4
    coverage = kr._hashed_count() / kr._corpus_size()
    assert coverage < kr.GATE["corpus_coverage_pct"] / 100.0, (
        "the fixture no longer separates the two counts, so this test proves "
        "nothing about the gate")

    # And the other half of clause 3: once the tail is covered the same document
    # is counted again, so the gate is not simply made harder to pass.
    h2 = ch.ContentHasher(index_path=idx)
    for feed in capped:
        h2.update_hash(feed, covered_through=len(feed.read_text(encoding="utf-8")),
                       complete=True)
    h2.save()
    (tmp_path / "rebuild-content-hashes.json").write_text(
        idx.read_text(encoding="utf-8"), encoding="utf-8")
    assert kr._hashed_count() == 4, (
        "a document read to the end is still refused, so coverage can never "
        "satisfy the gate")


def test_the_readme_documents_every_key_the_pipeline_result_line_carries():
    """The line is consumed by `skills/autonomy-data-pipeline/SKILL.md` as the
    nightly chain's gate, and that file and this module's `main()` were both
    outside the round that changed the line's shape. A reader with a fixed key set
    treats a dropped or renamed key as a zero, and a `truncated` that reads as 0 is
    exactly the misreading the count was added to prevent.

    `SKILL.md` is vault content this round may not write, so the contract is
    checked against the in-tree README, which states the line for both readers.
    """
    import inspect
    import re as _re
    import nightly_extraction as ne

    src = inspect.getsource(ne.main)
    emitted = set(_re.findall(r"PIPELINE_RESULT (.*?)\n", src))
    assert emitted, "main() no longer prints a PIPELINE_RESULT line"
    # Every line the function can print — the `status=locked` refusal is a
    # PIPELINE_RESULT line too, and a reader that only handles one of them will
    # not know the run never started.
    keys = sorted({m.group(1) for line in emitted
                   for m in _re.finditer(r"(\w+)=", line)})
    assert keys == ["facts", "failed", "files_processed", "status", "truncated"], keys

    readme = (ROOT / "scripts" / "memory" / "next-gen-memory" / "README.md")
    text = readme.read_text(encoding="utf-8")
    sample = [ln for ln in text.splitlines() if ln.startswith("PIPELINE_RESULT ")]
    assert sample, "the README no longer shows the line, so a reader cannot copy it"
    documented = sorted({m.group(1) for m in _re.finditer(r"(\w+)=", sample[0])})
    assert documented == keys, (
        f"the README documents {documented} but main() emits {keys}")
    assert "truncated" in text and "failed=0" in text, (
        "the README does not tell a reader that failed=0 stops meaning full "
        "coverage, which is the whole point of the count")


# ── the [N/M] progress denominator is the worked queue (#1011) ──────────────


_PROGRESS = re.compile(r"^\[(\d+)/(\d+)\] (Processing|FAILED|ERROR):", re.M)


def _progress(out):
    return [(int(n), int(m), kind) for n, m, kind in _PROGRESS.findall(out)]


@pytest.mark.parametrize("workers", [1, 2])
def test_progress_denominator_excludes_hash_skipped_files(tmp_path, monkeypatch,
                                                          capsys, workers):
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, f"prog{workers}")
    docs = []
    for i in range(4):
        d = tmp_path / "docs" / f"p{i}.md"
        d.write_text(f"note {i} holds a fact\n", encoding="utf-8")
        docs.append(d)
    monkeypatch.setattr(x.extractor, "extract_from_document",
                        lambda md, c, existing_facts="", start_offset=0: _one_fact(md))
    x._extract_all_facts(full_mode=False, workers=workers)
    capsys.readouterr()

    # Second run: two notes changed, one of which the model fails on; two skip.
    docs[1].write_text("note 1 changed\n", encoding="utf-8")
    docs[3].write_text("note 3 changed\n", encoding="utf-8")

    def one_fails(md_file, content, existing_facts="", start_offset=0):
        if Path(md_file).name == "p3.md":
            raise fx.ExtractionFailed("vLLM wedged")
        return _one_fact(md_file)

    monkeypatch.setattr(x.extractor, "extract_from_document", one_fails)
    x._extract_all_facts(full_mode=False, workers=workers)
    out = capsys.readouterr().out
    lines = _progress(out)
    assert {k for _, _, k in lines} == {"Processing", "FAILED"}, out
    assert all(m == 2 for _, m, _ in lines), out
    # the scan size is still reported, on its own lines
    assert "Found 4 eligible files" in out
    assert "Skipped 2 unchanged files" in out
    kg_store.reset()


def test_progress_denominator_is_the_limit_capped_queue(tmp_path, monkeypatch, capsys):
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, "proglimit")
    for i in range(5):
        (tmp_path / "docs" / f"l{i}.md").write_text(f"note {i}\n", encoding="utf-8")
    monkeypatch.setattr(x.extractor, "extract_from_document",
                        lambda md, c, existing_facts="", start_offset=0: _one_fact(md))
    x._extract_all_facts(full_mode=False, workers=1, limit=2)
    out = capsys.readouterr().out
    lines = _progress(out)
    assert [(n, m) for n, m, _ in lines] == [(1, 2), (2, 2)], out
    assert "Found 5 eligible files" in out
    kg_store.reset()


# ── #1246 clause 3: the skip count reaches the nightly run summary ───────────

def test_the_nightly_summary_prints_the_typed_pairs_the_extractor_skipped(
        tmp_path, monkeypatch):
    """The guard is only worth having if a reader can see it doing work: one
    line in the nightly log carries the count, and the grep-able
    PIPELINE_RESULT line keeps its key names untouched."""
    x, ch, idx = _extraction_run(tmp_path, monkeypatch, "skiplog")
    x.log_file = tmp_path / "nightly-extraction.log"
    st = kg_store.store()
    for name in ("Lloyd", "vLLM"):
        st.entities.register(name)
    st.edges.add({"source": "Lloyd", "target": "vLLM", "type": "uses",
                  "confidence": 0.9, "provenance": "EXTRACTED_CLASSIFIER_V4"},
                 origin="classifier")
    (tmp_path / "docs" / "lloyd.md").write_text("Lloyd serves models through vLLM\n")

    def typed_pair(md_file, content, existing_facts="", start_offset=0):
        return {"entity": "Lloyd", "category": "relationship",
                "facts": [{"fact": "Lloyd serves models through vLLM",
                           "confidence": 0.9, "category": "relationship"}]}

    monkeypatch.setattr(x.extractor, "extract_from_document", typed_pair)
    monkeypatch.setattr(x.rel_generator, "rebuild", lambda: {"total_relationships": 0})
    monkeypatch.setattr(x, "_regenerate_entity_overviews", lambda: 0)

    result = x.run_full_extraction(full_mode=True)

    assert result["success"] and result["files_processed"] == 1, result
    log = x.log_file.read_text()
    assert "Linked 0 mentions edges; 1 skipped on pairs the classifier had already typed" in log, log
    assert st.edges.active(source="Lloyd", target="vLLM")[0]["type"] == "uses"
    src = (ROOT / NE_PATH).read_text()
    line = src[src.index('f"PIPELINE_RESULT '):]
    line = line[:line.index('status=')]
    assert all(k in line for k in ("files_processed=", "facts=", "failed=", "truncated=")), line
    assert "mentions" not in line
    kg_store.reset()
