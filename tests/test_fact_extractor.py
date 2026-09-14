"""fact_extractor + nightly_extraction — the write path that owns the corpus.

Pins the seven defects the 2026-09-03 review found in this module: restarted
fact IDs, missing provenance, an LLM error that looked like a clean empty
extraction, a YAML error that wiped an entity's history, categories
registered as entities, junk names registered before they were rejected, and
edges that only ever appeared when someone ran a script by hand.
"""
import importlib.util
import json
import subprocess
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
    carry these names cannot inflate the count. Markdown is excluded from the
    search, because prose is exactly where this removal gets recorded — an
    architecture note or changelog naming the two files by name is the
    *expected* outcome, and a fact-extractor test must not fail one. Every
    code and config surface (`scripts/`, `app/`, `workers/`, `*.py`, `*.yaml`,
    `*.sh`) stays in scope; those are the files that can actually point at a
    path. Exit status 1 is git's "no matches" — a real answer, distinct from a
    failed git call, which is asserted rather than swallowed.
    """
    root = Path(__file__).resolve().parents[1]
    me = str(Path(__file__).resolve().relative_to(root))
    probe = ["git", "-C", str(root), "grep", "-l", "--fixed-strings",
             "-e", DEAD_DERIVED_INDEXES[0], "-e", DEAD_DERIVED_INDEXES[1]]

    # Positive control first: this file names both strings, so a grep that
    # cannot see it cannot report an honest "nothing else does". Without this
    # the empty result below would be indistinguishable from a silently
    # broken call, and a check that cannot fail is not a pin.
    control = subprocess.run(probe + ["--", ".", ":(exclude)*.md"],
                             capture_output=True, text=True)
    assert control.returncode == 0 and me in control.stdout.split(), (
        f"the reference grep cannot see its own file (rc={control.returncode}, "
        f"{control.stdout.split()}); the empty-result assertion below would be "
        "unevaluable")

    hits = subprocess.run(
        probe + ["--", ".", ":(exclude)*.md", f":(exclude){me}"],
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
