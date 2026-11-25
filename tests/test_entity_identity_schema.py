"""#537 — schema-gated entity identity on the extraction write path.

The gate ports one component of OaK's task kernel (arXiv 2608.22974): identity
is *declared* in a hand-authored schema and extraction attaches to it, instead
of the extractor minting a fresh entity row per surface-name variant.

Pins, in order:
  1. the schema file is typed, deduplicated, and agrees with app.entity_kind
     (no second taxonomy of the same column);
  2. a declared alias attaches to the canonical row and registers a
     `kind='semantic'` alias — the acceptance check's second clause;
  3. a declared key is never registered a second time as its own entity;
  4. an unknown name WITH a declared type creates a typed entity;
  5. an unknown name WITHOUT one goes to the candidates sidecar and mints
     nothing;
  6. nothing is ever merged by string similarity — the whole point;
  7. the extractor's write path actually routes through the gate;
  8. `register_schema_keys` RECONCILES the `origin='schema'` alias population
     (#1234) — a withdrawn declaration is retracted, a spelling variant the
     gate itself minted is kept, every other provenance is untouched, and an
     unbalanced-bracket surface is never written.
"""
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "scripts" / "memory" / "next-gen-memory"))

from app import entity_naming as en  # noqa: E402
from app import entity_kind, kg_store  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fx = _load("fact_extractor", "scripts/memory/next-gen-memory/fact_extractor.py")


@pytest.fixture
def store(tmp_path):
    st = kg_store.configure(tmp_path / "kg.sqlite")
    en.reset_identity_schema_cache()
    yield st
    kg_store.reset()
    en.reset_identity_schema_cache()


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    p = tmp_path / "entity-candidates.jsonl"
    monkeypatch.setattr(en, "ENTITY_CANDIDATES_PATH", p)
    return p


def _candidates(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _semantic_rows(st):
    return st._query("SELECT surface, canonical, origin FROM aliases WHERE kind='semantic'")


# ── 1. the schema artifact itself ────────────────────────────────────────────

def test_schema_exists_and_is_a_typed_declaration():
    schema = en.load_identity_schema()
    assert schema["entities"], "schema declares no entities"
    assert len(schema["entities"]) >= 20, (
        "the acceptance check samples 20 canonical entities; the schema must cover them")
    for e in schema["entities"]:
        assert e.get("canonical") and e.get("type"), e
        assert e["type"] in schema["types"], f"{e['canonical']}: undeclared type {e['type']}"


def test_schema_types_do_not_fork_the_entity_kind_vocabulary():
    """`entities.kind` already carries these values; a second, divergent
    vocabulary for the same column is a new defect, not a schema."""
    schema = en.load_identity_schema()
    assert set(entity_kind.KINDS) <= set(schema["types"])
    by_name = {e["canonical"]: e for e in schema["entities"]}
    assert len(by_name) == len(schema["entities"]), "duplicate canonical in schema"
    # The invariant that actually matters: every declared surface resolves to
    # the entry that declared it. An alias declared by two entries, or one that
    # is also a canonical, makes that answer depend on iteration order.
    for canon, e in by_name.items():
        assert en.schema_identity(canon) == canon, canon
        for a in e.get("aliases", []):
            assert a not in by_name, f"{a} is both alias and canonical"
            assert en.schema_identity(a) == canon, f"{a} did not resolve to {canon}"


# ── 2-3. declared identity attaches; it is not re-minted ─────────────────────

def test_declared_alias_attaches_to_the_canonical_row(store):
    en.register_schema_keys()
    for surface in ("Relationship Graph", "The Graph", "Autonomy Pipeline"):
        entity, verdict = en.gate_entity_name(surface)
        assert verdict == "schema", surface
    assert en.gate_entity_name("Relationship Graph")[0] == "Knowledge Graph"
    assert en.gate_entity_name("autonomy pipeline")[0] == "Autonomy Data Pipeline"
    # Attaching must not have created the variant as an entity of its own.
    assert store.aliases.resolve("Relationship Graph") == "Knowledge Graph"
    assert "Relationship Graph" not in store.entities.all()
    assert "Autonomy Pipeline" not in store.entities.all()


def test_declared_keys_register_as_semantic_aliases(store):
    """The acceptance check: `select count(*) from aliases where
    kind='semantic'` must rise as declared keys register."""
    assert len(_semantic_rows(store)) == 0
    n = en.register_schema_keys()
    rows = _semantic_rows(store)
    assert n > 0 and len(rows) >= n
    origins = {r["origin"] for r in rows}
    assert origins == {"schema"}, origins
    surfaces = {r["surface"] for r in rows}
    assert "The Graph" in surfaces and "Knowledge Graph" not in surfaces
    # Idempotent: re-registering must not add rows.
    assert en.register_schema_keys() == 0
    assert len(_semantic_rows(store)) == len(rows)


def test_declared_canonical_is_registered_with_its_declared_type(store):
    en.gate_entity_name("Knowledge Graph")
    assert store.entities.kinds().get("Knowledge Graph") == "system"


# ── 4-5. new names: typed passes, untyped goes to the sidecar ────────────────

def test_unknown_name_with_a_declared_type_is_created_typed(store, sidecar):
    entity, verdict = en.gate_entity_name("Foghorn Signal Router", declared_type="system")
    assert (entity, verdict) == ("Foghorn Signal Router", "typed_new")
    assert store.entities.kinds()["Foghorn Signal Router"] == "system"
    assert _candidates(sidecar) == []


def test_unknown_name_without_a_declared_type_becomes_a_candidate(store, sidecar):
    entity, verdict = en.gate_entity_name("Foghorn Signal Router", declared_type="")
    assert (entity, verdict) == ("", "candidate")
    assert "Foghorn Signal Router" not in store.entities.all()
    cands = _candidates(sidecar)
    assert len(cands) == 1
    assert cands[0]["name"] == "Foghorn Signal Router"
    assert cands[0]["reason"] == "no declared type"


def test_a_type_outside_the_schema_is_not_a_declared_type(store, sidecar):
    entity, verdict = en.gate_entity_name("Foghorn Signal Router", declared_type="banana")
    assert (entity, verdict) == ("", "candidate")
    assert _candidates(sidecar)[0]["declared_type"] == "banana"


def test_candidates_are_append_only_and_deduplicated_by_name(store, sidecar):
    en.gate_entity_name("Foghorn Signal Router", declared_type="")
    en.gate_entity_name("Foghorn Signal Router", declared_type="")
    en.gate_entity_name("Kestrel Batch Runner", declared_type=None,
                        source_doc="projects/x/y.md")
    cands = _candidates(sidecar)
    assert [c["name"] for c in cands] == ["Foghorn Signal Router", "Kestrel Batch Runner"]
    assert cands[1]["source_doc"] == "projects/x/y.md"


# ── 6. no similarity merge, ever ─────────────────────────────────────────────

def test_sibling_shaped_name_is_not_merged_by_string_similarity(store, sidecar):
    """`Autonomy Data Pipeline` exists and is the nearest name in the store.
    The gate must not attach to it — that inference is exactly what minted the
    sibling families this item is about, so it stays available as a fresh
    typed entity instead."""
    store.entities.register("Autonomy Data Pipeline", kind="pipeline")
    entity, verdict = en.gate_entity_name("Autonomy Data Pipeline Runner",
                                          declared_type="pipeline")
    assert (entity, verdict) == ("Autonomy Data Pipeline Runner", "typed_new")


def test_the_gate_never_consults_a_similarity_matcher(store, monkeypatch):
    """The strongest form of the contract: no string metric is reachable from
    the write gate at all. If someone later "improves" the gate by adding the
    fuzzy helper it used to avoid, this fails at the import rather than
    quietly merging two systems again.

    Read mode is the opposite and must stay that way — #400/#512 own read-side
    identity, so pinning that the reader still reaches the matcher is what
    keeps this round from having narrowed it.
    """
    import agent_mcp._shared as shared

    def boom(*a, **k):
        raise AssertionError("the write gate must not run a similarity matcher")

    monkeypatch.setattr(shared, "_fuzzy_entity_match", boom)
    en.register_schema_keys()
    for name in ("Autonomy Data Pipeline", "Autonomy Data Pipelin",
                 "Autonomy Data Pipeline Runner", "Knowledge Grap", "The Graph"):
        for declared in (None, "", "system", "pipeline"):
            en.gate_entity_name(name, declared_type=declared)

    # Sanity: the loop above actually exercised the declared path, so the test
    # is not passing because nothing ran.
    assert store.aliases.resolve("The Graph") == "Knowledge Graph"


# ── 7. the extractor write path is gated ─────────────────────────────────────

@pytest.fixture
def extractor(tmp_path, monkeypatch):
    facts = tmp_path / "facts"
    facts.mkdir()
    kg_store.configure(tmp_path / "kg.sqlite")
    en.reset_identity_schema_cache()
    monkeypatch.setattr(fx, "FACTS_DIR", facts)
    monkeypatch.setattr(en, "ENTITY_CANDIDATES_PATH", tmp_path / "entity-candidates.jsonl")
    e = fx.FactExtractor()
    e.facts_dir = facts
    yield e, tmp_path / "entity-candidates.jsonl"
    kg_store.reset()
    en.reset_identity_schema_cache()


def test_extractor_sanitiser_attaches_declared_variants(extractor):
    e, _ = extractor
    assert e._sanitize_entity("Relationship Graph", enforce=True) == "Knowledge Graph"
    assert e._sanitize_entity("The Graph", enforce=True) == "Knowledge Graph"


def test_extractor_sanitiser_routes_untyped_unknowns_to_candidates(extractor):
    e, sidecar = extractor
    assert e._sanitize_entity("Foghorn Signal Router", declared_type="", enforce=True) == ""
    assert _candidates(sidecar)[0]["name"] == "Foghorn Signal Router"
    # Rejected subject ⇒ no fact file, and no entity dir on disk.
    assert not (e.facts_dir / "Foghorn Signal Router").exists()


def test_enforcement_is_the_default_and_reads_opt_out(extractor):
    """The gate must not be something a caller remembers to switch on — but a
    read that refused a name for want of a type would withhold existing facts
    from the prompt, so `get_existing_facts`/`write_fact_file` opt out."""
    e, sidecar = extractor
    assert e._sanitize_entity("Foghorn Signal Router") == ""          # write default
    assert _candidates(sidecar)[0]["name"] == "Foghorn Signal Router"
    assert e._sanitize_entity("Foghorn Signal Router", enforce=False) == "Foghorn Signal Router"
    assert e.get_existing_facts("Foghorn Signal Router", "state") == ""
    # Opting out must not double-record the candidate.
    assert len(_candidates(sidecar)) == 1


def test_prompt_asks_the_model_for_a_declared_type():
    """Schema-constrained extraction needs the type on the wire, or every
    unknown name lands in the sidecar. Checked on the RENDERED prompt, so the
    vocabulary is pinned to coming from SCHEMA_TYPES rather than being typed
    into the prompt a second time and drifting from it."""
    rendered = fx.EXTRACTION_PROMPT.format(
        content="body", existing_facts="None", known_entities="(none recognised)",
        entity_types=" | ".join(fx.SCHEMA_TYPES))
    assert '"entity_type"' in rendered
    for t in ("system", "pipeline", "concept", "person"):
        assert t in rendered, t
    assert "is not created at all" in rendered
    # An unrendered template must not be what reaches the model.
    assert "{entity_types}" not in rendered


# ── 8. a withdrawn declaration is retracted, not merely superseded (#1234) ────

def _declared_aliases():
    """Every declared alias surface → its canonical, straight from the schema."""
    schema = en.load_identity_schema()
    return {a.strip(): e["canonical"]
            for e in schema["entities"] for a in (e.get("aliases") or [])}


# The six rows the live store still carries: every one of them is stamped
# `2026-09-10T04:55:14…Z`, the bulk registration under the feature's earlier
# shape (`extraction_schema.yaml`, commit `13fcc71`, not an ancestor of HEAD),
# and the landed JSON schema dropped all six. Reproduced verbatim, because the
# retractor has to cover the truncated surface as well as the withdrawn ones.
LIVE_WITHDRAWN_ROWS = (
    ("Task", "Entity Resolution Sweep"),
    ("Entity Resolution Sweep (Task", "Entity Resolution Sweep"),
    ("Knowledge Graphs", "Knowledge Graph"),
    ("Periodic Memory Capture", "Memory Capture"),
    ("Periodic Memory Capture Skill", "Memory Capture"),
    ("memory capture skill", "Memory Capture"),
)


def _seed_withdrawn_declaration(st):
    """Put the live store's six undeclared `semantic`/`schema` rows in a test
    store, with the canonicals they route to."""
    for canon, kind in (("Entity Resolution Sweep", "pipeline"),
                        ("Knowledge Graph", "system"),
                        ("Memory Capture", "pipeline")):
        st.entities.register(canon, kind=kind)
    for surface, canon in LIVE_WITHDRAWN_ROWS:
        st.aliases.set(surface, canon, kind="semantic", origin="schema")


def test_a_withdrawn_declaration_is_retracted_by_the_next_register(store):
    """Clause 1. `register_schema_keys` used only to INSTALL, so a declaration
    that was withdrawn left its row behind permanently: six rows survived the
    move from `extraction_schema.yaml` to the JSON schema, and because
    `gate_entity_name` answers `alias` before `typed_new`, the withdrawn
    mapping outranked every live judgement — a generic word, `Task`, filed
    facts against the entity-resolution pipeline. Registering must reconcile
    the population, and it must not lose a declaration while doing it."""
    _seed_withdrawn_declaration(store)
    assert store.aliases.resolve("Task") == "Entity Resolution Sweep"
    en.register_schema_keys()
    assert store.aliases.resolve("Task") is None
    # The truncated surface is the one row the FOLD alone would keep: `_schema_key`
    # drops the parenthesis, so `Entity Resolution Sweep (Task` folds to the
    # declared alias `Entity Resolution Sweep Task` and reads as declared. Only
    # the shape half of the predicate retracts it, which makes this the assertion
    # that pins that half — and the bracketed surface the schema really declares
    # has to survive the same call.
    assert store.aliases.resolve("Entity Resolution Sweep (Task") is None, (
        "the truncated surface survived: the bracket half of the predicate is not running")
    assert store.aliases.resolve("Knowledge Graph (KG)") == "Knowledge Graph"
    declared = _declared_aliases()
    # 42 as of #4561: `claude-code` -> `Claude Code`, the stack-update digester's
    # package slug, which had become a fact entity beside the tool it names. It is the
    # one declared alias whose TEXT is redundant — the slug folds onto its canonical's
    # key, so the loader answers it from the canonical line whether or not a row exists
    # — and whose ROW is indispensable, because the only previous fix was exactly such a
    # row (#893, 2026-09-16, `origin='triage-893'`) and a store rebuild deleted it with
    # nothing left to write it back. That asymmetry is why the declaration is a line of
    # this file rather than a call into the store.
    assert len(declared) == 42, f"schema declares {len(declared)} aliases, expected 42"
    for surface, canon in declared.items():
        assert store.aliases.resolve(surface) == canon, surface


def test_retraction_folds_the_normalisation_the_gate_installs_with(store):
    """Clause 2. The predicate cannot be a literal (surface, canonical) pair
    test: the schema spells `KG`, `gate_entity_name` legitimately mints `kg`
    for that same declaration, and deleting a row the next gate call re-mints
    is the churn reconciling exists to prevent. Keeping a row here means its
    `created_at` is untouched too — a nightly run must not re-date it."""
    en.register_schema_keys()
    assert en.gate_entity_name("kg", declared_type="system") == ("Knowledge Graph", "schema")
    row = next(r for r in store.aliases.rows() if r["surface"] == "kg")
    assert (row["canonical"], row["kind"], row["origin"]) == (
        "Knowledge Graph", "semantic", "schema")
    assert row["report_path"] is None
    before_rows = store.aliases.rows()
    assert en.register_schema_keys() == 0, "a reconcile over a settled store must write nothing"
    assert store.aliases.rows() == before_rows, "nothing added, nothing retracted"
    assert store.aliases.resolve("kg") == "Knowledge Graph"


def test_retraction_leaves_every_other_provenance_alone(store, tmp_path):
    """Clause 3. The retractor's scope is `origin='schema'` with no report, and
    nothing wider — the table also carries migration/sweep/test/triage rows,
    28 of the live unbalanced-bracket surfaces among them, which punct-stripping
    produced legitimately and this change has no business deleting. Nor may it
    take a `schema` row that names the run that authorized it: #475's
    verification is exactly 'apply-origin rows that name their report'.

    Two rows are seeded so that a wrong key shows up as a named loss rather than
    a matching count: a case-variant of a withdrawn surface, which a
    (canonical, surface_lc) delete would take with it, and a reported row on a
    surface the declaration does not resolve, which only `report_path IS NULL`
    spares."""
    en.register_schema_keys()
    report = tmp_path / "entity-merges-applied-2026-09-13-010000Z.json"
    report.write_text("{}")
    store.aliases.set("RWKV)", "RWKV", kind="punct", origin="migration")
    store.aliases.set("TypeScript)", "TypeScript", kind="punct", origin="sweep")
    store.aliases.set("vllm-engine", "vLLM", kind="suffix", origin="test")
    store.aliases.set("claude-code", "Claude Code", kind="punct",
                      origin="triage-893")   # the one live triage row, verbatim
    # A foreign row sharing the FOLDED surface of a withdrawn one, carrying the
    # (kind, origin) shape of 2,540 live `case`/`migration` rows. The primary key
    # is `surface`, so `task` and `Task` are two rows standing in one store, and
    # that is what makes this the discriminating seed: a retractor built on
    # `remove_where`, which keys on (canonical, surface_lc), deletes both. The one
    # this change ships keys on surface AND canonical AND origin, so this row has
    # to be here afterwards.
    store.aliases.set("task", "Entity Resolution Sweep", kind="case",
                      origin="migration")
    # A #475 apply-provenance row on a surface the declaration does NOT resolve.
    # Seeding it on a kept surface (the `kg` row) would protect it twice over and
    # test neither guard: this one is undeclared, so the only thing standing
    # between it and the DELETE is `report_path IS NULL`.
    store.aliases.set("Memory Capture Skill", "Memory Capture", kind="semantic",
                      origin="schema", report_path=str(report))
    store.aliases.set("kg", "Knowledge Graph", kind="semantic", origin="schema",
                      report_path=str(report))
    _seed_withdrawn_declaration(store)
    before = store.aliases.rows()
    foreign_before = [r for r in before if r["origin"] != "schema"
                      or r["report_path"] is not None]
    en.register_schema_keys()
    after = store.aliases.rows()
    foreign_after = [r for r in after if r["origin"] != "schema"
                     or r["report_path"] is not None]
    # A population, not a count. One row in `foreign_before` is #893's
    # `claude-code` -> `Claude Code` (`origin='triage-893'`), seeded here because the
    # live store carries it. #4561 DECLARED that pair, so the install adopts it —
    # re-stamped `schema`, which is precisely what a declaration is for — and it leaves
    # this population on purpose. The invariant is therefore "nothing else moved, and
    # anything that did is a pair the file now declares", which a single count cannot say.
    before_keys = {(r["surface"], r["canonical"]) for r in foreign_before}
    after_keys = {(r["surface"], r["canonical"]) for r in foreign_after}
    assert after_keys <= before_keys, (
        "a row with another provenance changed shape: " f"{sorted(after_keys - before_keys)}")
    moved = before_keys - after_keys
    assert moved <= set(_declared_aliases().items()), (
        f"rows left the other-provenance population that the schema does not declare: "
        f"{sorted(moved - set(_declared_aliases().items()))}")
    # Whatever stays foreign keeps every field it had. `created_at` is excluded on
    # purpose: the one row this round legitimately touches is #893's, and it is
    # re-stamped rather than rewritten-elsewhere — which `moved` above already pinned as
    # a declared pair. Comparing full dicts would also fail on the adopted row's clock.
    fields = ("surface", "surface_lc", "canonical", "kind", "origin", "report_path")
    stayed = before_keys & after_keys
    shape = lambda rs: {((r["surface"], r["canonical"])): {f: r[f] for f in fields}
                        for r in rs if (r["surface"], r["canonical"]) in stayed}
    assert shape(foreign_after) == shape(foreign_before), (
        "content of other-provenance rows moved")
    # Named, not just aggregated: the two rows the guards exist for.
    assert [(r["canonical"], r["kind"], r["origin"]) for r in after if r["surface"] == "task"] == [
        ("Entity Resolution Sweep", "case", "migration")], "the case-variant row moved"
    assert [(r["canonical"], r["origin"], r["report_path"]) for r in after
            if r["surface"] == "Memory Capture Skill"] == [
        ("Memory Capture", "schema", str(report))], "the #475 reported row was retracted"
    # Exact surface, not `resolve`: `resolve` answers case-insensitively, so once
    # a `task` row stands — and it must, clause 3's whole point is that it
    # survives — `resolve("Task")` reaches that row and answers a canonical. What
    # has to be gone is the withdrawn ROW.
    assert "Task" not in store.aliases.all(), "the withdrawn row is still retracted"


def test_an_unbalanced_bracket_surface_is_never_written(store, tmp_path, caplog):
    """Clause 4. `Entity Resolution Sweep (Task` is in the live alias table and
    is permanently canonical: a heading read too far by a parser, stored because
    nothing checked the shape. Unbalanced brackets are evidence the string was
    cut, so the declaration path refuses one and says so — while
    `Knowledge Graph (KG)`, the one bracketed surface the schema declares,
    installs as normal."""
    schema = json.loads((ROOT / "scripts" / "memory" / "entity_identity_schema.json")
                        .read_text(encoding="utf-8"))
    schema["entities"].append({"canonical": "Foo Widget", "type": "system",
                               "aliases": ["Foo (Bar"]})
    copy = tmp_path / "entity_identity_schema.json"
    copy.write_text(json.dumps(schema), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="app.entity_naming"):
        en.register_schema_keys(copy)
    assert "Foo (Bar" not in store.aliases.all(), "unbalanced surface was stored"
    assert "Foo (Bar" in caplog.text, "the refusal was not reported"
    assert store.aliases.resolve("Knowledge Graph (KG)") == "Knowledge Graph"
    # `gate_entity_name` is the OTHER producer of `origin='schema'` rows — it
    # mints one whenever a raw name matches a declared key spelled differently
    # from the canonical — so the guard has to hold there too. It does, because
    # it sits on the shared write: the name still resolves to its canonical, it
    # just acquires no alias row for the truncated spelling.
    assert en.gate_entity_name("Foo (Bar", declared_type="system",
                               path=copy) == ("Foo Widget", "schema")
    assert "Foo (Bar" not in store.aliases.all()


def test_the_extractor_no_longer_files_task_under_the_pipeline(extractor):
    """The seam this whole item is about: the nightly extractor runs in its own
    process against the live store, and its first gated name is what installs
    the declarations. Seeding the withdrawn row first is that process's start
    state. Before the fix `Task` filed under `Entity Resolution Sweep`; after
    it, the row is retracted by the install the extractor itself triggers, so
    the extractor's write path judges `Task` as `typed_new`."""
    st = kg_store.store()          # the extractor fixture owns this store
    _seed_withdrawn_declaration(st)
    e, _ = extractor
    assert e._sanitize_entity("Task", declared_type="system") == "Task"
    assert e._sanitize_entity("Relationship Graph", enforce=True) == "Knowledge Graph"


# ── 10. the digester's package slug must not be an entity of its own (#4561) ──
#
# skills/documentation-digester/SKILL.md:165 writes knowledge/stack-updates/<date>-
# {key}.md and its own note at :168-178 warns that `{key}` is not the fact-store entity
# name and that using it "silently creates a second entity". The warning cannot hold,
# because the writer is not the digester run's `fact_add` — it is the nightly deep
# extraction pass, which derives an entity from the note's own slug text and files what
# it gets. Measured 2026-11-25 on the live store:
# `curl -s localhost:8080/api/entity?name=claude-code` answered `factCount: 1` (stat-001
# "The path does not exist (No such file).", provenance EXTRACTED, source_doc
# knowledge/stack-updates/2026-11-25-claude-code.md) beside `Claude Code`'s 4,609, and
# `facts/claude-code/` carried two files with 2026-11-24 mtimes. A reader asking about
# the tool by the name the digester writes was served last night's fragment and none of
# the history. #893 closed this split on 2026-09-16 with a hand-written store row
# (`kind='punct', origin='triage-893'`) and verified `resolve('claude-code')`; that row
# is gone, its 4,609 facts are not, and the 2026-10-02 owed-check ruled the split
# settled. These nodes pin the correction: the declaration lives in a human-edited file
# every process installs from, not in a store a rebuild may replace.
CLAUDE_CANON = "Claude Code"
CLAUDE_SLUG = "claude-code"
STACK_NOTE = "knowledge/stack-updates/2026-11-25-claude-code.md"
SCHEMA_PATH = ROOT / "scripts/memory/entity_identity_schema.json"


def alias_row(st, surface: str, canonical: str):
    """One alias row by its FULL key.

    `resolve(surface)` cannot stand in for this: the schema answers the folded surface on
    its own now, so resolution succeeds whether or not a row was ever written, and a node
    that asked it would pass on the store's silence.
    """
    for r in st.aliases.rows():
        if r["surface"] == surface and r["canonical"] == canonical:
            return r
    return None


@pytest.fixture
def fragment(store):
    """The live defect, present BEFORE the schema is read: the slug is an entity.

    The order is the mechanism, not decoration. `claude-code` self-registered in 2026-09
    through the unclaimed-name path, and a self-registered row is what makes the split
    feed itself: `known_entities_in_text` offers every registered name to the next pass,
    so each cycle's note re-arms its own fragment. A nightly pass reads the schema
    afterwards, which is the state this fixture reaches; installing first on a clean store
    would skip the collision branch a fragment actually lands in.
    """
    store.entities.register(CLAUDE_SLUG, kind="system")
    return store


def test_the_declaration_is_accepted_and_folds_the_slug_onto_the_canonical(store):
    """Clause 1. The file declares the pair, the loader accepts it, the fold is real.

    The pair is unusual and the reason belongs in the open: `_schema_key` folds case,
    punctuation and whitespace, so `claude-code` and `Claude Code` reduce to the same key,
    `claude code`, and the loader's index answers the slug from the canonical line with no
    alias row in between. The alias is declared anyway because the ROW is what a rebuild
    needs re-installed, and it is what #893 had and lost. The fold assertion is what keeps
    the two counts in this file honest about what the declaration actually adds: it is the
    reason `test_the_shipped_alias_population_is_exactly_what_the_store_should_carry`
    counts 42 rather than 41.
    """
    assert en._schema_key(CLAUDE_SLUG) == en._schema_key(CLAUDE_CANON) == "claude code"
    loaded = en.load_identity_schema()
    by_name = {e["canonical"]: e for e in loaded["entities"]}
    assert CLAUDE_CANON in by_name, "the loader dropped the declared canonical"
    assert CLAUDE_SLUG in (by_name[CLAUDE_CANON].get("aliases") or []), (
        "the slug is not among the canonical's own alias lines")
    assert en.schema_identity(CLAUDE_SLUG) == CLAUDE_CANON, (
        "the folded key no longer answers the canonical, so the write gate cannot fold "
        "the slug and tonight's pass files into the fragment again")
    assert en.schema_type_of(CLAUDE_CANON) == "system", (
        "the declared type is not drawn from the file's own types list, or has moved off "
        "the `system` kind the store already carries for this entity")
    assert CLAUDE_SLUG in _declared_aliases(), (
        "the alias line is gone, so the installed row and the pinned count are stale")


def test_the_write_gate_folds_the_slug_before_the_fragment_can_be_used(store):
    """Clause 2 at the gate the extractor actually calls.

    `fx._identity_gate` is that module's own import of `gate_entity_name`, reached by
    `_gate_entity` at its line 530 with the model-supplied name and the note's path —
    exactly what the nightly pass hands it for a stack-update note. The fragment entity
    already exists here, which is the state the live store is in, and `store` is a fresh
    tmp KGStore, so a second registered spelling means a second name that could be filed
    under. The schema is consulted first, so the self-registered row no longer wins.
    """
    store.entities.register(CLAUDE_SLUG, kind="system")
    assert store.entities.get(CLAUDE_SLUG) is not None, (
        "fixture lost its teeth: the fragment is no longer a registered entity")

    name, verdict = fx._identity_gate(CLAUDE_SLUG, declared_type=None,
                                      source_doc=STACK_NOTE)
    assert (name, verdict) == (CLAUDE_CANON, "schema"), (
        f"the write gate answers {name!r} by {verdict!r}; anything but 'schema' is a "
        f"route a rebuild can delete")
    # The canonical keeps answering as itself — the fold redirects a variant, it does
    # not rename the entity that holds 4,609 facts.
    assert fx._identity_gate(CLAUDE_CANON, declared_type="system",
                             source_doc=STACK_NOTE) == (CLAUDE_CANON, "schema")


def test_the_gate_redirects_the_slug_without_moving_a_single_fact_row(extractor):
    """Clause 2's second half: nothing is created under `facts/claude-code/`.

    Written, not asserted away — a fact is filed under the name the gate returned, in a
    tmp facts tree, so the assertion that no slug-named file exists is a measurement of
    where writes go rather than of the tree being empty to begin with. And the half that
    makes this a safe round rather than a convenient one: a declaration attaches FUTURE
    extractions and migrates nothing. The live `facts/claude-code/` directory, and the
    one row the owed-check now owns, are untouched by any of this; the read side still
    serves the fragment while the directory stands, because `agent_mcp/_shared.py`'s
    `_resolve_entity` probes the fact tree before the alias table (#754 is the precedent
    for knowingly stranding one such sibling).
    """
    e, _sidecar = extractor
    e.facts_dir.mkdir(parents=True, exist_ok=True)
    kg_store.store().entities.register(CLAUDE_SLUG, kind="system")
    name, _verdict = e._gate_entity(CLAUDE_SLUG, STACK_NOTE)
    assert (name, _verdict) == (CLAUDE_CANON, "schema"), (name, _verdict)
    e.write_fact_file(name, "state", {"facts": [
        {"fact": "armed nightly since 2026-11-23", "confidence": 0.9,
         "category": "state", "provenance": "EXTRACTED",
         "source_doc": STACK_NOTE}]})
    facts = e.facts_dir

    assert (facts / CLAUDE_CANON / f"{CLAUDE_CANON}-state.md").is_file(), (
        "the write did not land under the name the gate returned; the tree is "
        f"{sorted(q.name for q in facts.iterdir())}")
    assert not (facts / CLAUDE_SLUG).exists(), (
        "the slug got a directory of its own, which is the split this declaration "
        "exists to stop")
    assert not any(q.name == CLAUDE_SLUG for q in facts.iterdir()), (
        f"something was created under the slug anyway: {sorted(q.name for q in facts.iterdir())}")
    assert en.schema_identity(CLAUDE_CANON) == CLAUDE_CANON, (
        "the fold went the wrong way and renamed the entity holding 4,609 facts")


def test_the_pair_installs_over_the_fragment_and_the_second_install_writes_nothing(
        fragment):
    """Clause 3. Installed with the two fields that matter, over the fragment, twice.

    `kind='semantic'` preserves behaviour: the slug's own row carries `system`, and alias
    kinds rank BELOW a stored type in the precedence `entity_kind` applies, so the alias
    can never downgrade the canonical's type. `origin='schema'` preserves the row: it is
    the origin the reconciler recognises as schema-owned, and the one #893's row lacked
    (`triage-893`), which is why a rebuild could delete a fix with nothing left to say it
    had been there. The collision branch is present by construction here — the surface
    already names an entity — so asserting the row exists is also asserting the install
    was not skipped as a collision. A second call writing 0 rows is what lets every
    process install on its first gate call without churning `created_at`.
    """
    assert en.register_schema_keys() > 0, "nothing was installed at all"
    row = alias_row(fragment, CLAUDE_SLUG, CLAUDE_CANON)
    assert row is not None, (
        "no row for the pair: the install skipped it as a collision against the "
        "fragment's own entity row, which is exactly the branch a self-registered slug "
        "lands in, and the store would keep answering the slug by itself")
    assert row["kind"] == "semantic" and row["origin"] == "schema", row
    assert fragment.aliases.resolve(CLAUDE_SLUG) == CLAUDE_CANON
    assert fragment.entities.get(CLAUDE_SLUG) is not None, (
        "the declaration deleted the entity row too, which is migration work this round "
        "does not own")

    assert en.register_schema_keys() == 0, (
        "a second install wrote rows, so installing from the file is not the no-op the "
        "every-process call site depends on")


def test_a_store_replaced_from_scratch_regains_the_pair_and_keeps_the_row_it_declares(
        tmp_path):
    """The durability half of the acceptance, measured as a rebuild.

    #893's fix was a row: `store.aliases.set('claude-code', 'Claude Code',
    kind='punct', origin='triage-893')`, verified by `resolve()`, and gone today while
    the entity's 4,609 facts remain. Facts are re-derived from the markdown tree by every
    pass; an alias row is only ever data in one file. `kg_store.wipe_provenance_rows`
    deliberately SPARES aliases ("they are declared identity, not extraction output"), so
    the loss cannot be that DELETE either: what happened is the store being replaced whole
    by a rebuild, which carries what can be re-derived and nothing else. So the test is a
    second store, not a second call — and what it must show is the pair coming back from
    the file, and the file-declared row standing in the store that the rebuild itself
    writes.
    """
    before = kg_store.configure(tmp_path / "kg-before.sqlite")
    en.reset_identity_schema_cache()
    before.entities.register(CLAUDE_SLUG, kind="system")
    before.aliases.set(CLAUDE_SLUG, CLAUDE_CANON, kind="punct", origin="triage-893")
    assert before.aliases.resolve(CLAUDE_SLUG) == CLAUDE_CANON
    kg_store.reset()

    rebuilt = kg_store.configure(tmp_path / "kg-after.sqlite")   # the 2026-09-22/23 state
    en.reset_identity_schema_cache()
    assert rebuilt.aliases.resolve(CLAUDE_SLUG) is None, (
        "a fresh store carries the pair already, so this node measures nothing")

    assert en.register_schema_keys() > 0, (
        "nothing came back after a rebuild, so the file is not the route being trusted")
    row = alias_row(rebuilt, CLAUDE_SLUG, CLAUDE_CANON)
    assert row is not None and row["kind"] == "semantic" \
        and row["origin"] == "schema", row
    assert rebuilt.aliases.resolve(CLAUDE_SLUG) == CLAUDE_CANON

    # And it survives the reconciliation that runs inside every install: `origin
    # = 'schema'` is the mark that says "the file owns this, keep it", so a second pass
    # writes nothing and retracts nothing here either.
    assert en.register_schema_keys() == 0, "a reconcile wrote rows over a settled store"
    assert alias_row(rebuilt, CLAUDE_SLUG, CLAUDE_CANON) is not None, (
        "reconciliation retracted the row the schema declares")
    en.reset_identity_schema_cache()
    kg_store.reset()
