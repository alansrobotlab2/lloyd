"""`eval/djev/schemas.py`: option order is part of the schema, and the hash
is what makes that a test rather than a convention.

Measured 2026-09-20: reversing the option order of one question moved P by
0.324 on average and 0.851 at worst, and dropped accuracy at a fixed 0.5
cutoff from 0.80 to 0.50. Every framing had a different optimal threshold
(0.39, 0.03, 0.30, 0.01). So a threshold belongs to an exact wording AND an
exact option order, and a rewording that kept the old number would be
reporting a measurement of a different question.
"""

from __future__ import annotations

import pytest

from eval.djev import schemas


def test_every_schema_is_current():
    """The whole point of `calibrated_hash`. A wording or option-order edit
    that leaves the numbers behind fails here and nowhere else."""
    for name, s in schemas.SCHEMAS.items():
        s.check()   # raises SchemaDrift


def test_reordering_options_changes_the_hash():
    """`sort_keys=False` in `Schema.hash` is load-bearing, not an oversight:
    sorting would make the one change most likely to invalidate a threshold
    invisible to the check that exists to catch it."""
    q = schemas.ENTITY.spec["same_entity"]
    reversed_criteria = dict(reversed(list(q["criteria"].items())))
    flipped = schemas.Schema(
        name="entity", seam="entity",
        spec={"same_entity": {**q, "criteria": reversed_criteria}})
    assert flipped.hash != schemas.ENTITY.hash
    with pytest.raises(schemas.SchemaDrift):
        schemas.Schema(name="x", spec=flipped.spec,
                       calibrated_hash=schemas.ENTITY.hash).check()


def test_rewording_changes_the_hash():
    q = schemas.DEDUPE.spec["same_finding"]
    reworded = schemas.Schema(
        name="dedupe",
        spec={"same_finding": {**q, "instructions": q["instructions"] + "?"}})
    assert reworded.hash != schemas.DEDUPE.hash


def test_the_negative_option_is_listed_first_in_every_pair_schema():
    """The arm that measured AUC 0.833, against 0.795 for the reverse and an
    optimal threshold of 0.03 — a number that is noise. Order-AVERAGING is
    not the fix: it lands between the two arms and `acc@0.5` stays 0.53."""
    for name in ("dedupe", "entity", "clusters"):
        q = next(iter(schemas.SCHEMAS[name].spec.values()))
        first = list(q["criteria"])[0]
        assert first in ("different", "distinct"), f"{name} lists {first} first"


def test_no_schema_may_gate_a_production_decision_yet():
    """`gate_ready` is a separate field from `threshold is not None` because a
    measured threshold and a threshold that may DECIDE something are
    different claims. Every schema here carries a measured number and none
    may act on it."""
    for name, s in schemas.SCHEMAS.items():
        assert s.gate_ready is False, name
        assert s.gate_blocked_reason.strip(), name


def test_a_calibrated_schema_names_what_it_was_calibrated_on():
    """A threshold with no provenance cannot be re-checked, and three of
    these were measured against another MODEL's opinions rather than against
    truth."""
    for name, s in schemas.SCHEMAS.items():
        if s.threshold is not None or s.label_mass_floor is not None:
            assert s.calibrated_on.strip(), name
            assert s.calibrated_hash == s.hash, name


def test_the_rerank_schema_has_no_floor_and_says_why():
    """The one place the design's own rule was tested rather than restated:
    listwise `label_mass` at n=12-16 has measured 0.446, 0.807, 0.965 and
    1.000 across four runs on different corpora. No floor read off one of
    them means anything for the others, so the shadow rows set it."""
    assert schemas.RERANK.label_mass_floor is None
    assert schemas.RERANK.threshold is None


def test_every_live_seam_has_a_schema():
    """A seam with no schema records nothing, which is how a new seam fails:
    silently uninstrumented rather than instrumented against a shape nobody
    froze."""
    from app import djev_shadow
    for seam in djev_shadow.SEAMS:
        assert seam in schemas.BY_SEAM, seam
        assert schemas.hash_for(seam)


def test_rank_levels_are_ordered_worst_first():
    """`score` returns an expected value over the level INDEX, so reversing
    this list reverses every ranking silently."""
    from app import djev
    assert list(djev.RANK_LEVELS) == schemas.RANK_LEVELS
    assert schemas.RANK_LEVELS[0] == "irrelevant"
    assert schemas.RANK_LEVELS[-1] == "directly answers it"


def test_the_clusters_schema_asks_what_its_corpus_labels():
    """382 `distinct`, 221 `related`, 6 `same`, and no `duplicate` verdict at
    any row. A `duplicate`-vs-`distinct` framing measured 12 usable pairs out
    of 609 and reported AUC 1.000 off the six `same` rows."""
    options = list(schemas.CLUSTERS.spec["related_item"]["criteria"])
    assert options == ["distinct", "related"]
