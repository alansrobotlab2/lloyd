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


def _kg_hygiene():
    """The same module object `kg_health._hygiene_section` imports, by the same
    path, so a baseline written here is the one the snapshot reads."""
    sys.path.insert(0, str(ROOT / "scripts" / "memory"))
    import kg_hygiene
    return kg_hygiene


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


def test_captured_at_is_utc_with_an_offset(db):
    """#822: the stamp was naive local time, so a 20:58 PDT run read as the
    previous UTC day's snapshot and no reader could tell which clock it was."""
    import datetime as dt
    stamp = dt.datetime.fromisoformat(kg_health.build_snapshot()["captured_at"])
    assert stamp.utcoffset() == dt.timedelta(0), stamp
    assert abs((dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()) < 300


# ── #1535: the hygiene block passes the regrowth reference through ────────────

REGROWTH_KEYS = {"days", "new_dirs", "near_dup_new", "by_tier", "samples",
                 "skipped_vanished",                                    # every key already on disk
                 "dirs_at_baseline", "baseline_at", "baseline_path", "no_baseline_reason"}


def test_hygiene_regrowth_carries_its_baseline_reference_through_the_snapshot(db, tmp_path):
    """#1535 clause 3: the section names the reference it diffed against —
    `dirs_at_baseline` at `baseline_at` — and the health block passes the new
    keys through untouched. `_hygiene_section` filters the top level of
    kg_hygiene's snapshot and reshapes nothing else; a key dropped in there is a
    key no reader of the JSON will ever see again."""
    import datetime as dt
    kg_hygiene = _kg_hygiene()
    facts = kg_health.VAULT_FACTS_ROOT                 # the fixture's one-dir tree ("vllm")
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(facts, base)
    (facts / "VLLM").mkdir()                           # its case twin, created afterwards

    r = kg_health.build_snapshot(baseline_path=base)["hygiene"]["regrowth"]
    assert set(r) == REGROWTH_KEYS, sorted(r)
    assert r == kg_hygiene.regrowth(facts, 7, baseline_path=base)   # passed through unchanged
    assert r["days"] == 7
    assert (r["new_dirs"], r["near_dup_new"]) == (1, 1)
    assert r["samples"] == ["VLLM"]
    assert r["dirs_at_baseline"] == 1                  # the denominator, from the same walk
    assert r["no_baseline_reason"] is None
    assert dt.datetime.fromisoformat(r["baseline_at"]).utcoffset() == dt.timedelta(0)


def test_hygiene_regrowth_reports_none_rather_than_the_store_size(db, tmp_path):
    """With no baseline the snapshot carries None and the reason, which is the
    one state the old check can no longer be reproduced in: this tree has one
    entity directory, and before #1535 that is precisely where the snapshot
    printed `new_dirs: 1` beside `entities.count: 1` because a 7-day window over
    rebuilt `created_at` stamps covers any tree built this week."""
    r = kg_health.build_snapshot(baseline_path=tmp_path / "absent.json")["hygiene"]["regrowth"]
    assert r["new_dirs"] is None and r["near_dup_new"] is None
    assert "absent.json" in r["no_baseline_reason"]
    assert r["dirs_at_baseline"] is None
    assert kg_health.build_snapshot(baseline_path=tmp_path / "absent.json")["entities"]["count"] == 1


def test_the_summary_prints_the_reference_beside_the_regrowth_number(db, tmp_path, capsys):
    """kg_health's own line, the surface the item named: it used to read
    `near-dup regrowth 7d  51 of 12027 new dirs`, where 12,027 was the whole
    store. The window is gone from the line because the number is no longer a
    window."""
    kg_hygiene = _kg_hygiene()
    facts = kg_health.VAULT_FACTS_ROOT
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(facts, base)
    (facts / "VLLM").mkdir()

    kg_health.print_summary(kg_health.build_snapshot(baseline_path=base))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "regrowth" in ln]
    assert len(lines) == 1, lines
    assert "1 of 1 new dirs" in lines[0], lines[0]
    assert "baseline" in lines[0] and "over 1 dirs" in lines[0], lines[0]
    assert "7d" not in lines[0], lines[0]


def test_the_summary_renders_a_pre_fix_snapshot_without_a_bare_count(db, tmp_path, capsys):
    """The snapshots already on disk under `_pipeline/metrics/` have no baseline
    keys and must still render — with their number labelled as having no
    reference, rather than as the growth of 11,959 directories in a week."""
    snap = kg_health.build_snapshot(baseline_path=tmp_path / "absent.json")
    snap["hygiene"]["regrowth"] = {"days": 7, "new_dirs": 11959, "near_dup_new": 51,
                                   "by_tier": {"CASE": 51}, "samples": [], "skipped_vanished": 0}

    kg_health.print_summary(snap)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "regrowth" in ln]
    assert len(lines) == 1, lines
    assert "51 of 11,959 new dirs" in lines[0], lines[0]
    assert "no baseline recorded" in lines[0], lines[0]
