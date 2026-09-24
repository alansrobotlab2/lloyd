"""#546 — one closed edge-type vocabulary, asserted rather than prose-synced.

Six partial lists of edge types lived in six files, joined only by "keep in
sync with ..." comments, and nothing refused a type outside all of them:
`fact_relate` wrote whatever string it was handed. `app.kg_store.EDGE_TYPES`
is now the one set. These tests hold every list a writer or ranker keys on to
it, by importing the list itself, so a type added to one of them without the
set fails here instead of drifting.

`seed_relationship_edges._EDGE_VOCAB` is deliberately not held to it: it is a
list of words that are *not entity names* (hyphenated spellings and `used_by`
included), used to reject noise entities, and writes no edge type.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.kg_store import EDGE_TYPES, canonical_edge_type  # noqa: E402

MEMORY = ROOT / "scripts" / "memory"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_the_set_is_already_in_its_stored_spelling():
    """The store folds every type through `canonical_edge_type` (#1161), so a
    member in any other spelling could never match a stored row."""
    assert EDGE_TYPES
    assert {canonical_edge_type(t) for t in EDGE_TYPES} == set(EDGE_TYPES)


def test_retrieval_weights_and_directional_types_are_members():
    from agent_mcp import retrieval
    assert set(retrieval.EDGE_TYPE_WEIGHTS) <= EDGE_TYPES, \
        set(retrieval.EDGE_TYPE_WEIGHTS) - EDGE_TYPES
    assert set(retrieval.DIRECTIONAL_EDGE_TYPES) <= EDGE_TYPES, \
        set(retrieval.DIRECTIONAL_EDGE_TYPES) - EDGE_TYPES


def test_the_relation_classifier_vocabulary_is_a_member():
    mod = _load("classify_relationships_546", MEMORY / "classify-relationships.py")
    assert set(mod.VOCABULARY) <= EDGE_TYPES, set(mod.VOCABULARY) - EDGE_TYPES
    # An out-of-vocabulary answer is coerced to `mentions`, which must itself
    # be a member or the coercion writes the drift it exists to prevent.
    assert "mentions" in mod.VOCABULARY


def test_the_conversation_linker_types_are_members():
    mod = _load("conversation_relations_546", MEMORY / "conversation_relations.py")
    assert set(mod.VALID_RELATION_TYPES) <= EDGE_TYPES, \
        set(mod.VALID_RELATION_TYPES) - EDGE_TYPES
    assert mod.DEFAULT_RELATION_TYPE in EDGE_TYPES
    # The type a landed link falls back to when a proposal carries none.
    assert "co_accessed" in EDGE_TYPES


def test_the_structural_types_the_entity_graph_hides_are_members():
    from app.routers import entities
    assert set(entities._STRUCTURAL_EDGE_TYPES) <= EDGE_TYPES


@pytest.mark.parametrize("word", ["informs", "ships", "shares_mechanism_with",
                                  "upgrade_candidate_for", "mitigates"])
def test_the_one_off_types_fact_relate_minted_are_not_members(word):
    """The count-1 `fact_relate` types the item measured. Admitting them would
    make the set a record of drift rather than a bound on it."""
    assert word not in EDGE_TYPES
