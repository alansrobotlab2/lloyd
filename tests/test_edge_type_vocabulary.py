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


def _failing(line: str) -> str:
    """Everything after `conditions failing: `, so an attribution assertion cannot be
    satisfied by the operand columns printed before it."""
    return line.split("conditions failing: ", 1)[1]


def test_the_ceiling_and_the_vocabulary_still_fail_the_line_after_the_tolerance():
    """#2760 clause 4: what the trend clause was protecting still fails it, untouched.

    #2760 weakened the trend half of the share condition twice over — the baseline is
    now the prior night's snapshot nearest this run's own time of day instead of that
    night's LAST snapshot (which was always the post-#74 trough), and a rise has to
    clear `EDGE_TYPE_TREND_TOLERANCE` rather than being any rise at all. Both were
    needed: over 2026-10-10 to 10-15 the line FAILed every night while the level fell
    from 34.2% to 31.8%. A weakened trigger is only safe if the two conditions this
    verdict exists to catch still catch, so they are asserted here through the real
    function, at the constants the module actually reads rather than copies of them.

    Level alone: 85 of 100 on `mentions` FAILs with NO baseline at all, and the line
    says no previous night was available — so the FAIL is the ceiling's doing, not the
    trend's, which is the whole point of #1658's "a trend with a level under it". The
    same 85% also FAILs when the baseline sits ABOVE it (0.90): the ceiling is tested
    first and independently, so neither the phase alignment nor the tolerance can buy
    an escape from a runaway catch-all.

    Vocabulary at any count: `ships` at exactly 5 uses — at the floor, therefore not a
    rare type at all — FAILs as a type outside `EDGE_TYPES`, and `ships` at 1 use
    FAILs too, attributed to the 5-use floor because #1820 hands an off-vocabulary name
    below the floor to the floor condition and only names the vocabulary for a name at
    or above it. Either way a minted type cannot pass, which is what has to survive a
    tolerance introduced somewhere else in the line.

    The PASS rows keep these assertions honest: an in-vocabulary spread under the
    ceiling with no history still PASSes with `conditions failing: none`, so the four
    FAILs above are the conditions firing and not a line that fails everything.
    """
    khr = _load("khr_vocabulary_2760", MEMORY / "knowledge-health-report.py")
    assert khr.EDGE_TYPE_MAX_SHARE == 0.8, "the ceiling is not #2760's to move"
    assert khr.EDGE_TYPE_MIN_USES == 5
    assert khr.EDGE_TYPE_TREND_TOLERANCE == 0.005, (
        "#2760 fixed the tolerance at 0.5 pp for this round; deriving it from the "
        "measured intra-day spread is owed to the owed-check job")

    ceiling = khr.edge_type_cardinality({"mentions": 85, "uses": 15})
    assert ceiling.startswith("Edge-type cardinality: FAIL — "), ceiling
    assert "dominant type share (mentions 85.0% > 80%)" in _failing(ceiling), ceiling
    assert "no previous night's share available" in ceiling, ceiling

    under_a_falling_baseline = khr.edge_type_cardinality(
        {"mentions": 85, "uses": 15},
        previous={"date": "2026-10-14", "share": 0.90})
    assert under_a_falling_baseline.startswith(
        "Edge-type cardinality: FAIL — "), under_a_falling_baseline
    assert "dominant type share (mentions 85.0% > 80%)" in _failing(
        under_a_falling_baseline), under_a_falling_baseline

    off_vocab_at_the_floor = khr.edge_type_cardinality(
        {"uses": 400, "related_to": 350, "part_of": 245, "ships": 5})
    assert off_vocab_at_the_floor.startswith("Edge-type cardinality: FAIL — "), \
        off_vocab_at_the_floor
    assert ("types outside the edge-type vocabulary (ships (5))"
            in _failing(off_vocab_at_the_floor)), off_vocab_at_the_floor

    off_vocab_once = khr.edge_type_cardinality(
        {"uses": 400, "related_to": 350, "part_of": 249, "ships": 1})
    assert off_vocab_once.startswith("Edge-type cardinality: FAIL — "), off_vocab_once
    assert ("types below the 5-use floor (ships (1))"
            in _failing(off_vocab_once)), off_vocab_once

    clean = khr.edge_type_cardinality({"uses": 40, "part_of": 35, "related_to": 25})
    assert clean.startswith("Edge-type cardinality: PASS — "), clean
    assert _failing(clean).strip() == "none", clean
    falling = khr.edge_type_cardinality(
        {"uses": 400, "related_to": 350, "part_of": 250},
        previous={"date": "2026-10-14", "share": 0.4010})
    assert falling.startswith("Edge-type cardinality: PASS — "), falling
    assert _failing(falling).strip() == "none", falling
