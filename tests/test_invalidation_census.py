"""#3927 — the invalidation census: what it counts, on which root, and what it refuses.

`scripts/maintenance/invalidation_census.py` is the measurement the fact store has
never had: how often a record retires itself, and whether that number is even a
number. Two failure modes are the whole reason this module exists, so both are
pinned here rather than assumed away:

  * **the false zero.** The item's step-1 path — the `facts` directory under the
    vault — is an empty directory; the live tree is `app.paths.VAULT_FACTS_ROOT`,
    which resolves under the data root. A census that spells the retired path
    reports zero invalidations and fires this item's own stop gate on an artefact.
    So the root is resolved through `app.paths` at call time, and a node below
    redirects it to prove the row reports whatever the resolver says.
  * **the bulk job counted as evidence.** One sweep stamped 442 facts with a
    single `expired_at`. As rows that clears a 200-event bar; as events it is
    one. Every count that feeds the stop gate is therefore a count of DISTINCT
    stamps, and the raw row count is reported beside it, never inside it.

Fact fields are nested in a `facts:` YAML list, which is why a line-anchored grep
over the tree returns nothing; the fixtures below are written in that real shape
and read back through the same parser the index is built with, so a fixture that
only *looks* like a fact file cannot pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import kg_store, paths                                    # noqa: E402
from scripts.maintenance import invalidation_census as cens        # noqa: E402


def _scalar(v) -> str:
    """A YAML scalar in the shape the live tree writes: quoted ISO strings and a
    literal `null` for an unset field, never a bare word."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))


def _write_fact_file(root: Path, entity: str, category: str, facts: list[dict]) -> Path:
    """One markdown fact file in the shipped shape: every field NESTED under a
    `facts:` list, `expired_at: null` on a live fact."""
    lines = ["---", "type: facts", f"entity: {_scalar(entity)}",
             f"category: {_scalar(category)}", "facts:"]
    for f in facts:
        lines.append(f"- entity: {_scalar(f.get('entity', entity))}")
        lines.append(f"  fact: {_scalar(f.get('fact'))}")
        for key in ("confidence", "category", "provenance", "created_at", "valid_at",
                    "source_doc", "expired_at", "invalid_at", "id"):
            if key in f:
                lines.append(f"  {key}: {_scalar(f[key])}")
    lines += ["last_extracted: null", "---", "", f"# {entity}", ""]
    path = root / entity / f"{entity}-{category}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fact(text: str, created: str, *, expired=None, invalid=None) -> dict:
    return {"fact": text, "confidence": 0.9, "provenance": "STATED",
            "created_at": created, "expired_at": expired, "invalid_at": invalid,
            "id": abs(hash(text)) % 10_000}


@pytest.fixture
def fact_tree(tmp_path: Path) -> Path:
    """Seven facts, eight days of spread, two expiry stamps.

    Alpha and Beta carry four expired facts over TWO distinct stamps: three
    share `2026-03-12` (a sweep), one stands on `2026-03-13`. One fact is
    invalidated, two are live. Two entities are affected. Every age in days:
    70, 69, 40 (all inside the 31-90 bucket) and 1 (inside 1-7).
    """
    root = tmp_path / "facts"
    _write_fact_file(root, "Alpha", "capability", [
        _fact("Alpha runs on GPU 1.", "2026-01-01T00:00:00+00:00",
              expired="2026-03-12T00:00:00+00:00"),
        _fact("Alpha ships a nightly build.", "2026-01-02T00:00:00+00:00",
              expired="2026-03-12T00:00:00+00:00"),
        _fact("Alpha is written in Python.", "2026-01-03T00:00:00+00:00"),
    ])
    _write_fact_file(root, "Beta", "state", [
        _fact("Beta is at version 2.", "2026-02-01T00:00:00+00:00",
              expired="2026-03-13T00:00:00+00:00"),
        _fact("Beta was renamed.", "2026-02-02T00:00:00+00:00",
              invalid="2026-04-01T00:00:00+00:00"),
        _fact("Beta was rebuilt.", "2026-03-11T00:00:00+00:00",
              expired="2026-03-12T00:00:00+00:00"),
        _fact("Beta is maintained by Alan.", "2026-02-03T00:00:00+00:00"),
    ])
    return root


@pytest.fixture
def kg(tmp_path: Path):
    """A provisioned store, reached the only way the census is allowed to reach
    it: it is the process default that `app.kg_store.store()` returns."""
    original = kg_store._default_path
    store = kg_store.configure(tmp_path / "kg-store-for-census.db")
    yield store
    kg_store.reset()
    kg_store._default_path = original


def _seed_edges(store) -> None:
    """Three expired edges over TWO distinct days (two of them the same day an
    hour apart, which is what a nightly sweep looks like) plus two live edges."""
    for source, target, at in (("A", "B", "2026-10-01T00:00:00+00:00"),
                              ("C", "D", "2026-10-02T05:00:00+00:00"),
                              ("E", "F", "2026-10-02T06:00:00+00:00")):
        edge_id = store.edges.add({"source": source, "target": target, "type": "uses",
                                  "confidence": 0.9, "provenance": "STATED"}, origin="test")
        assert store.edges.expire(edge_id, "census test", at=at)
    for source, target in (("G", "H"), ("I", "J")):
        store.edges.add({"source": source, "target": target, "type": "uses",
                        "confidence": 0.9, "provenance": "STATED"}, origin="test")


NO_TREE = Path("/nonexistent-fact-tree-for-this-node")


# ── clause 1: the root is resolved, the database is never opened here ─────────

def test_facts_root_is_the_resolved_one_and_not_the_retired_literal(tmp_path, monkeypatch,
                                                                   fact_tree, kg):
    """Clause 1: `facts_root` in the emitted row equals
    `str(paths.VAULT_FACTS_ROOT)` under a redirected root, and the totals it
    measured over that root are non-zero."""
    monkeypatch.setattr(paths, "VAULT_FACTS_ROOT", fact_tree)
    row = cens.build_row()
    assert row["facts_root"] == str(paths.VAULT_FACTS_ROOT) == str(fact_tree)
    assert row["fact_stream"]["total_facts"] == 7, (
        "a census that names the right root but reports zero facts is the false "
        "zero this item was filed to kill")
    assert row["facts_root_is_default"] is False, (
        "the row must say whether it read the built-in location or an override (#700)")


def test_the_root_the_script_falls_back_to_is_the_shipped_resolver():
    """The default root is `app.paths.VAULT_FACTS_ROOT`, so a later move of the
    tree moves the census with it instead of leaving it counting a corpse."""
    assert paths.VAULT_FACTS_ROOT == paths.VAULT_FACTS_ROOT_DEFAULT, (
        "LLOYD_FACTS_ROOT is set in this environment; the fallback is only "
        "meaningful against the built-in default")
    assert paths.VAULT_ROOT not in (paths.VAULT_FACTS_ROOT,
                                    *paths.VAULT_FACTS_ROOT.parents), (
        f"{paths.VAULT_FACTS_ROOT} sits under the vault root {paths.VAULT_ROOT}: "
        "that is the retired spelling, and it holds no fact files at all")
    assert str(paths.VAULT_FACTS_ROOT_DEFAULT).endswith("/facts")


def test_the_script_spells_no_path_and_opens_no_database():
    """Clause 1, the negative half: the module contains neither the retired
    literal fact path nor any route to the database file. The store is reached
    only through `app.kg_store.store()`, the one sanctioned opener."""
    source = Path(cens.__file__).read_text(encoding="utf-8")
    for banned in ("obsidian", "sqlite", "connect(", "execute("):
        assert banned not in source, f"{banned!r} appears in the census module"
    assert "kg_store.store()" in source, "the store must be reached through store()"
    assert "paths.VAULT_FACTS_ROOT" in source, "the root must be read off app.paths"


# ── clause 2: four separate fact-side numbers ────────────────────────────────

def test_fact_side_publishes_four_separate_counts(fact_tree):
    """Clause 2: total facts, facts with `expired_at` set, facts with
    `invalid_at` set, and distinct entities affected — four numbers, never one
    blended rate."""
    fs = cens.scan_fact_tree(fact_tree)
    assert fs["total_facts"] == 7
    assert fs["facts_with_expired_at"] == 4
    assert fs["facts_with_invalid_at"] == 1
    assert fs["distinct_entities_affected"] == 2, (
        "Alpha and Beta are the entities carrying an expiry or an invalidation")
    assert fs["files_with_fact_lists"] == 2 and fs["files_scanned"] == 2


def test_a_live_fact_with_a_null_stamp_is_not_an_invalidation(tmp_path):
    """`expired_at: null` is the shipped spelling of "this fact is current".
    Reading it as set would report every live fact as retired — which is what a
    string-keyed grep over the tree does, and this parser must not."""
    root = tmp_path / "null-tree"
    _write_fact_file(root, "Gamma", "state", [
        _fact("Gamma is current.", "2026-05-01T00:00:00+00:00")])
    fs = cens.scan_fact_tree(root)
    assert fs["total_facts"] == 1
    assert fs["facts_with_expired_at"] == 0 and fs["facts_with_invalid_at"] == 0
    assert fs["distinct_expired_at_values"] == 0
    assert fs["distinct_entities_affected"] == 0


# ── clause 3: an event is a stamp, not a row ─────────────────────────────────

def test_distinct_stamps_are_the_event_count_not_rows(fact_tree):
    """Clause 3: the row reports DISTINCT `expired_at` values, so four expired
    facts carrying two stamps is two events."""
    fs = cens.scan_fact_tree(fact_tree)
    assert fs["facts_with_expired_at"] == 4
    assert fs["distinct_expired_at_values"] == 2


def test_a_bulk_job_expiring_hundreds_of_facts_is_one_event(tmp_path, kg):
    """Clause 3's point, at the size that decides the item: 250 expired facts
    sharing 3 stamps is 250 ROWS and 3 EVENTS. Rows clear the 200-event bar,
    events do not, and the verdict has to follow the events."""
    root = tmp_path / "bulk"
    for entity in ("Bulk1", "Bulk2"):
        facts = [_fact(f"{entity} claim {i}.", "2026-08-01T00:00:00+00:00",
                       expired=f"2026-09-{(i % 3) + 1:02d}T00:00:00+00:00")
                 for i in range(125)]
        _write_fact_file(root, entity, "state", facts)
    row = cens.build_row(facts_root=root)
    assert row["fact_stream"]["facts_with_expired_at"] == 250
    assert row["fact_stream"]["distinct_expired_at_values"] == 3
    assert row["stop_gate"]["independent_event_count"] == 3
    assert row["stop_gate"]["verdict"] == "stop_census_only", (
        "the gate was computed on rows: 250 expired facts from 3 sweeps is not "
        "250 independent observations")


# ── clause 4: the edge stream ────────────────────────────────────────────────

def test_edge_stream_reports_total_active_expired_and_expiry_days(kg):
    """Clause 4: the edge-side numbers come from `store().stats()` plus one
    aggregate read — edges_total, edges_active, the expired count, and the
    number of DISTINCT expiry days, which is this stream's honest unit."""
    _seed_edges(kg)
    stats = kg_store.store().stats()
    row = cens.build_row(facts_root=NO_TREE)
    edge = row["edge_stream"]
    assert edge["edges_total"] == stats["edges_total"] == 5
    assert edge["edges_active"] == stats["edges_active"] == 2
    assert edge["edges_expired"] == 3
    assert edge["expired_edge_distinct_expiry_days"] == 2, (
        "two edges expired an hour apart on one day are one day of a sweep")
    assert edge["expired_edge_distinct_expiry_timestamps"] == 3
    assert edge["agree"] is True, (
        "the stats() delta and the direct expired count must agree; if they do "
        "not, the store grew an expiry mechanism the census does not know about")
    assert row["kg_db_resolved_by"] == "app.kg_store.store()"


def test_the_edge_stream_is_judged_against_the_same_threshold(kg):
    """The substrate question the item leaves open has to be answerable FROM the
    artifact: the same threshold applied to the edge stream's independent unit,
    printed beside the fact stream's verdict."""
    _seed_edges(kg)
    row = cens.build_row(facts_root=NO_TREE)
    assert row["edge_stream_gate"]["threshold"] == row["stop_gate"]["threshold"]
    assert row["edge_stream_gate"]["independent_event_count"] == 2
    assert row["edge_stream_gate"]["verdict"] == "stop_census_only"
    assert "edge stream" in row["open_ruling"]


# ── clause 5: the gate, the unbuilt arms, and the censoring ──────────────────

def test_stop_gate_threshold_is_published_and_named(kg, fact_tree):
    """Clause 5: the verdict is computed against a NAMED, published threshold,
    and the row cites that name so a reader can tell a gate from a judgement."""
    assert cens.STOP_GATE_NAME == "STOP_GATE_MIN_INDEPENDENT_EVENTS"
    assert cens.STOP_GATE_MIN_INDEPENDENT_EVENTS == 200
    row = cens.build_row(facts_root=fact_tree)
    assert row["stop_gate"]["name"] == cens.STOP_GATE_NAME
    assert row["stop_gate"]["threshold"] == 200
    assert row["stop_gate"]["independent_event_count"] == 2
    assert row["stop_gate"]["verdict"] == "stop_census_only"
    assert "DISTINCT" in row["stop_gate"]["basis"]
    assert cens.stop_gate(200)["verdict"] == "build_arms", (
        "the bar is >= threshold: 200 events is the item's 'if there are >=200'")
    assert cens.stop_gate(199)["verdict"] == "stop_census_only"


def test_unbuilt_arms_state_the_independent_event_count_as_the_reason(kg, fact_tree):
    """Clause 5: with the gate closed both arms are `not_built`, and the reason
    names the event count — not the row count — as the cause."""
    arms = cens.build_row(facts_root=fact_tree)["arms"]
    assert arms["hazard_rate"]["status"] == "not_built"
    assert arms["prediction"]["status"] == "not_built"
    assert "2 independent invalidation events" in arms["hazard_rate"]["reason"], (
        arms["hazard_rate"]["reason"])
    assert arms["hazard_rate"]["horizons_days"] == [30, 60, 90, 180]


def test_when_the_gate_opens_the_arms_stay_unbuilt_and_say_why(kg, fact_tree):
    """Clearing the gate does not build an arm in this round: the reason turns to
    the substrate ruling, so the artifact can never read as a finished fit."""
    row = cens.build_row(facts_root=fact_tree, threshold=2)
    assert row["stop_gate"]["verdict"] == "build_arms"
    assert row["arms"]["hazard_rate"]["status"] == "not_built"
    assert "substrate" in row["arms"]["hazard_rate"]["reason"], (
        row["arms"]["hazard_rate"]["reason"])


def test_censoring_reports_the_day_distribution_and_the_bulk_share(kg, fact_tree):
    """Clause 5: the row carries the day-distribution of
    (expired_at - created_at) and the share of expiries carried by the single
    most common stamp."""
    cen = cens.build_row(facts_root=fact_tree)["censoring"]
    buckets = cen["expiry_minus_created_day_buckets"]
    assert sum(buckets.values()) == 4, "every expired fact in the fixture has a creation date"
    assert buckets["31_to_90_days"] == 3, "70, 69 and 40 days to the stamp"
    assert buckets["1_to_7_days"] == 1, "Beta was rebuilt one day before the sweep stamped it"
    assert buckets["same_day"] == 0 and buckets["negative"] == 0
    assert cen["deltas_measured"] == 4
    assert cen["most_common_expired_at"] == "2026-03-12T00:00:00+00:00"
    assert cen["most_common_expired_at_facts"] == 3
    assert cen["share_carried_by_single_most_common_stamp"] == pytest.approx(0.75, abs=1e-4)


def test_an_expiry_written_before_its_fact_was_created_is_counted_not_clamped(tmp_path):
    """A backfill can stamp an expiry older than the row's own creation date.
    That belongs in the `negative` bucket and stays visible: clamping it to zero
    would inflate the same-day share, which is the number that decides whether a
    hazard curve means anything at all."""
    root = tmp_path / "negative"
    _write_fact_file(root, "Delta", "state", [
        _fact("Delta was stamped before it was written.", "2026-06-10T00:00:00+00:00",
              expired="2026-06-01T00:00:00+00:00")])
    buckets = cens.censoring_row(cens.scan_fact_tree(root))["expiry_minus_created_day_buckets"]
    assert buckets["negative"] == 1
    assert buckets["same_day"] == 0


# ── the second instrument: the store's index of the same tree ────────────────

def test_the_store_index_is_a_second_instrument_and_the_tree_wins(kg, fact_tree):
    """`facts_idx` is BUILT from the tree, so the row publishes both and names one
    as the source of record. Indexed from the fixture tree, the index has to say
    the same four things the tree says."""
    kg.facts_idx.reindex(root=fact_tree)
    row = cens.build_row(facts_root=fact_tree)
    idx = row["fact_index_from_store"]
    assert idx["index_rows"] == 7
    assert idx["rows_with_expired_at"] == 4
    assert idx["rows_with_invalid_at"] == 1
    assert idx["distinct_expired_at_values"] == 2
    xc = row["cross_check"]
    assert xc["agree_on_totals"] and xc["agree_on_expired"] and xc["agree_on_invalid"]
    assert "fact_stream" in xc["source_of_record_for_the_four_fact_counts"]


def test_a_stale_index_disagrees_loudly_instead_of_becoming_the_answer(kg, fact_tree):
    """The tree's counts must never be replaced by the index's. An unindexed store
    reports zero fact rows, and a census that took that as its fact side is the
    false zero all over again — so the disagreement has to be visible and the
    tree's 7 facts still have to be the headline."""
    row = cens.build_row(facts_root=fact_tree)
    xc = row["cross_check"]
    assert row["fact_index_from_store"]["index_rows"] == 0
    assert row["fact_stream"]["total_facts"] == 7
    assert xc["tree_total_facts"] == 7
    assert xc["agree_on_totals"] is False, (
        "the cross-check stopped meaning anything: both instruments now agree by "
        "construction rather than by measurement")


# ── the artifact itself ──────────────────────────────────────────────────────

def test_main_writes_one_dated_row_under_the_baselines_dir(tmp_path, monkeypatch, fact_tree, kg):
    """The deliverable: one dated JSON artifact beside the nightly baselines,
    naming the command that regenerates it."""
    out = tmp_path / "baselines"
    monkeypatch.setattr(paths, "EVAL_BASELINES_DIR", out)
    assert cens.main(["--facts-root", str(fact_tree)]) == 0
    files = sorted(out.glob("invalidation-census-*.json"))
    assert len(files) == 1, files
    row = json.loads(files[0].read_text(encoding="utf-8"))
    assert files[0].name == cens.artifact_name(row["date"])
    assert row["date"] in files[0].name
    assert row["facts_root"] == str(fact_tree)
    assert row["regenerate"]["command"] == "python -m scripts.maintenance.invalidation_census"
    assert row["stop_gate"]["verdict"] in ("stop_census_only", "build_arms")
    assert row["censoring"]["expiry_minus_created_day_buckets"], (
        "the artifact must carry the censoring measurement, not just the counts")


def test_the_default_output_dir_is_the_eval_baselines_dir():
    resolved = cens.resolve_out_dir(paths.EVAL_BASELINES_DIR)
    assert resolved == Path(paths.EVAL_BASELINES_DIR).resolve()
    assert resolved.name == "baselines" and resolved.parent.name == "eval"


def test_an_output_dir_inside_the_code_tree_is_a_refusal():
    """#2378's ruling: runtime state never reaches a commit, so an in-tree
    destination is a refusal rather than a warning."""
    with pytest.raises(ValueError):
        cens.resolve_out_dir(ROOT / "eval" / "baselines")
    with pytest.raises(ValueError):
        cens.resolve_out_dir(ROOT)
