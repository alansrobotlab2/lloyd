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
