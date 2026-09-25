"""app.entity_naming.known_entities_in_text — extraction-time entity hints."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import entity_naming as en  # noqa: E402
from app import kg_store  # noqa: E402


@pytest.fixture
def aliases(tmp_path):
    st = kg_store.configure(tmp_path / "kg.sqlite")
    names = ["Intel Pipeline", "Intel", "vLLM", "segment", "active", "stack-updates",
             "Alfie", "GR00T N1", "Node.js", "GPU", "qmd-sdk", "Claude Agent SDK"]
    for n in names:
        st.entities.register(n)
    st.aliases.set("intel pipeline", "Intel Pipeline", kind="case", origin="test")
    yield st
    kg_store.reset()


def test_multi_token_names_match_case_insensitively(aliases):
    hits = en.known_entities_in_text("the INTEL pipeline scans arxiv and the claude agent sdk streams")
    assert hits == ["Intel Pipeline", "Claude Agent SDK"]


def test_single_token_names_need_proper_casing_or_acronym(aliases):
    assert en.known_entities_in_text("intel corp shipped a gpu") == []          # lowercase surface ≠ Intel/GPU
    assert en.known_entities_in_text("Intel shipped a new GPU today") == ["Intel", "GPU"]
    assert en.known_entities_in_text("Alfie picked up the block") == ["Alfie"]


def test_lowercase_slug_and_frontmatter_words_are_never_hints(aliases):
    text = "---\nsegment: knowledge\nstatus: active\ntags: [stack-updates]\n---\nqmd-sdk wraps QMD"
    assert en.known_entities_in_text(text) == []


def test_longest_match_wins_and_order_is_first_occurrence(aliases):
    hits = en.known_entities_in_text("GR00T N1 runs on vLLM; Intel Pipeline uses Node.js")
    assert hits == ["GR00T N1", "vLLM", "Intel Pipeline", "Node.js"]


def test_limit_and_empty(aliases):
    assert en.known_entities_in_text("") == []
    assert en.known_entities_in_text("Intel vLLM Alfie", limit=2) == ["Intel", "vLLM"]


def test_index_refreshes_when_the_store_changes(aliases):
    assert en.known_entities_in_text("Blackwell rocks") == []
    aliases.entities.register("Blackwell")
    assert en.known_entities_in_text("Blackwell rocks") == ["Blackwell"]


def test_normalize_and_register_round_trip(aliases):
    assert en.normalize("INTEL PIPELINE") == "Intel Pipeline"
    assert en.normalize("nothing known") == "nothing known"
    assert en.normalize_and_register("Brand New Thing") == "Brand New Thing"
    assert aliases.entities.exists("Brand New Thing")
    # a second call resolves rather than re-registering
    assert en.normalize_and_register("brand new thing") == "Brand New Thing"
    assert aliases.entities.count() == 13


def test_set_alias_records_kind_and_origin(aliases):
    en.set_alias("vllm-engine", "vLLM", origin="test")
    row = aliases.aliases.for_canonical("vLLM")[0]
    assert row["surface"] == "vllm-engine" and row["origin"] == "test"
    # tokenised on non-alphanumerics, so this reads as a suffix difference
    # rather than an unrelated semantic merge
    assert row["kind"] == "suffix"
    assert en.normalize("VLLM-Engine") == "vLLM"


def test_alias_kind_shapes():
    from app.kg_store import alias_kind
    assert alias_kind("vLLM", "vLLM") == "self"
    assert alias_kind("VLLM", "vLLM") == "case"
    assert alias_kind("swe-bench", "SWE Bench") == "punct"
    assert alias_kind("Intel Pipeline System", "Intel Pipeline") == "suffix"
    assert alias_kind("Groundskeeper", "Intel") == "semantic"


def test_a_declaration_cannot_erase_an_apply_runs_provenance(tmp_path):
    """#475: an authorized apply's evidence was one nightly extraction away from
    vanishing. `Aliases.set` upserts on `surface` and takes `origin`/`report_path`
    with it, so the schema-declaration path rewrote an applied row to
    origin='schema' with a NULL report — the mapping survived and the record of who
    authorized it did not, which is how a store that had been backfilled could go
    on reading as if it never had been."""
    st = kg_store.configure(tmp_path / "kg.sqlite")
    report = tmp_path / "entity-merges-applied-2026-09-13-010000Z.json"
    report.write_text("{}")
    try:
        st.entities.register("vLLM")
        st.aliases.set("vllm", "vLLM", kind="punct", origin="sweep", report_path=str(report))
        assert en._ensure_alias("vllm", "vLLM", kind="semantic", origin="schema") is False
        row = next(r for r in st.aliases.rows() if r["surface"] == "vllm")
        assert (row["origin"], row["report_path"]) == ("sweep", str(report))
        # Only rows carrying a run's provenance are protected. An inherited row is
        # still the declaration's to route — that is what this function is for.
        st.aliases.set("vllm-engine", "vLLM", kind="punct", origin="migration")
        assert en._ensure_alias("vllm-engine", "vLLM", kind="semantic", origin="schema") is True
        assert next(r for r in st.aliases.rows()
                    if r["surface"] == "vllm-engine")["origin"] == "schema"
    finally:
        kg_store.reset()


# ── #1486: rule-derived alias regeneration ────────────────────────────────────

def _regen_module():
    import importlib.util
    p = Path(__file__).resolve().parent.parent / "scripts" / "memory" / "regenerate_aliases.py"
    spec = importlib.util.spec_from_file_location("regenerate_aliases", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def regen_store(tmp_path):
    st = kg_store.configure(tmp_path / "kg.sqlite")
    for n in ["Causal Agent Replay (CAR)", "Qwen3.8-Flash-Next", "Ameli et al. (2024)",
              "Hierarchical Diffusion Policy (Ma et al.)",
              "Hierarchical Diffusion Policy (Wang et al.)", "Knowledge Graph", "SWE-bench",
              "SWE bench"]:
        st.entities.register(n)
    yield st
    kg_store.reset()


def test_regeneration_writes_through_the_store_with_a_rule_origin_and_a_date(regen_store, tmp_path):
    """Clause 1: the entry point writes alias rows through `app.kg_store`'s one
    writer, each with a non-empty `origin` naming the rule and a `created_at`."""
    rg = _regen_module()
    report = tmp_path / "regen.json"
    out = rg.run(apply=True, report=report)
    rows = [r for r in regen_store.aliases.rows() if r["origin"].startswith("regen:")]
    assert rows and out["written"] == len(rows)
    for r in rows:
        assert r["origin"].split(":", 1)[1] in rg.RULES, r
        assert r["created_at"], r
        assert r["report_path"] == str(report)
    assert json.loads(report.read_text())["written"] == len(rows)


def test_a_rule_alias_resolves_where_it_resolved_to_nothing_before(regen_store):
    """Clause 2: a punctuation variant and a parenthetical acronym of an existing
    canonical resolve to it through `store().resolve` after regeneration, and to
    nothing before."""
    rg = _regen_module()
    probes = {"Qwen3.8 Flash Next": "Qwen3.8-Flash-Next",
              "CAR": "Causal Agent Replay (CAR)",
              "Causal Agent Replay": "Causal Agent Replay (CAR)"}
    assert all(kg_store.store().resolve(s) is None for s in probes)
    rg.run(apply=True)
    for surface, canonical in probes.items():
        assert kg_store.store().resolve(surface) == canonical, surface


def test_regeneration_never_routes_a_real_name_or_an_ambiguous_one(regen_store):
    """The guard rails: a surface that is itself an entity (`SWE bench`), a
    citation year, and a surface two canonicals both produce are never written —
    an alias that routes a real name elsewhere is a merge by another name."""
    rg = _regen_module()
    p = rg.plan(regen_store.entities.all(), set())
    surfaces = {r["surface"].lower() for r in p["rows"]}
    assert "swe bench" not in surfaces            # an entity row already
    assert "ameli et al." not in surfaces         # (2024) is a citation, never stripped
    assert "hierarchical diffusion policy" not in surfaces
    assert p["collisions"] == [{"surface": "hierarchical diffusion policy",
                                "canonicals": ["Hierarchical Diffusion Policy (Ma et al.)",
                                               "Hierarchical Diffusion Policy (Wang et al.)"]}]


def test_a_dry_run_writes_nothing(regen_store):
    rg = _regen_module()
    before = regen_store.aliases.count()
    out = rg.run(apply=False)
    assert out["planned"] > 0 and out["written"] == 0
    assert regen_store.aliases.count() == before
