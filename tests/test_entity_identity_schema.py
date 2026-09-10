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
  7. the extractor's write path actually routes through the gate.
"""
import importlib.util
import json
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
