"""kg_health.py's `edges.by_type` — one key per relation (#1161 clause 4).

The 2026-09-15 snapshot that surfaced this item listed `related_to` 5,046 and
`related-to` 49 as two relations, because `by_type` counted the raw `type`
column. A histogram that splits one relation in two makes the vocabulary look
bigger than it is and lets any per-type rule silently see half of it. The
snapshot is a file a person reads and a metric later reports quote, so the fold
belongs before the count, not in whoever opens the JSON afterwards.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "kg_health_types", ROOT / "scripts" / "memory" / "kg_health.py")
kg_health = importlib.util.module_from_spec(_spec)
sys.modules["kg_health_types"] = kg_health
_spec.loader.exec_module(kg_health)

from app import kg_store  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A real store and a facts root with one entity directory, so
    `build_snapshot` runs end to end over data this test owns."""
    facts = tmp_path / "facts"
    (facts / "vllm").mkdir(parents=True)
    monkeypatch.setattr(kg_health, "VAULT_FACTS_ROOT", facts)
    s = kg_store.configure(tmp_path / "kg.sqlite")
    yield s
    kg_store.reset()


def _legacy_row(s, source, target, typ):
    """A row in the spelling the pre-fold store wrote. `edges.add` folds now, so
    a direct INSERT is the only way to hand the snapshot a non-canonical type —
    which is exactly the case clause 4 is about."""
    s.conn.execute(
        "INSERT INTO edges(source, target, type, confidence, provenance,"
        " created_at, origin) VALUES (?,?,?,0.9,'INFERRED',"
        " '2026-09-18T08:38:03+00:00', 'conversation')",
        (source, target, typ))
    s.conn.commit()


def test_by_type_reports_each_relation_under_one_key(db):
    """Clause 4: no snapshot may list `related_to` and `related-to` side by
    side. Both rows are the same relation, so the histogram holds one key."""
    _legacy_row(db, "vllm", "Ray", "related-to")
    db.edges.add({"source": "vllm", "target": "Kubernetes", "type": "related_to",
                  "confidence": 0.9}, origin="test")

    snap = kg_health.build_snapshot()
    assert snap["edges"]["by_type"] == {"related_to": 2}, snap["edges"]["by_type"]
    assert snap["edges"]["count"] == 2
    assert [t for t in snap["edges"]["by_type"] if "-" in t] == []


def test_by_type_folds_a_spelling_the_store_never_recorded(db):
    """The canonicalisation is a rule over the type, not a table of the two
    spellings that happened to be live in September: `depends-on` folds, and a
    name in its canonical form is passed through untouched."""
    _legacy_row(db, "vllm", "Ray", "depends-on")
    _legacy_row(db, "vllm", "SGLang", "requires")

    assert kg_health.build_snapshot()["edges"]["by_type"] == {
        "depends_on": 1, "requires": 1}


def test_by_type_does_not_merge_two_relations_that_differ_by_more_than_a_hyphen(db):
    """`conflicts_with` and `competes_with` are two relations (#546 owns whether
    they should be one), so the histogram keeps two keys even though the first
    arrived hyphenated."""
    _legacy_row(db, "vllm", "SGLang", "conflicts-with")
    _legacy_row(db, "vllm", "TensorRT-LLM", "competes_with")

    assert kg_health.build_snapshot()["edges"]["by_type"] == {
        "conflicts_with": 1, "competes_with": 1}
