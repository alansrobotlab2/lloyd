"""#541 — the counterfactual-perturbation rater on the retrieval eval.

Two numbers per query, both over entity/fact-level output only:

  counterfactual_moved_rate   the changed constraint actually moved what the
                              retriever pulled;
  counterfactual_pinned_rate  the constraints that did NOT change pulled the
                              same rows they did before.

The pinned half is the one a doc-level metric cannot see: the doc corpus is the
same vault either way, so a forward-only "did it change" rater rewards an
extractor that churns everything on every edit. Both directions, or neither
number means anything.

These tests pin the committed perturbation records (one per query, deterministic,
never re-derived from live graph data at eval time so a nightly diff stays a
diff) and the two metric definitions.
"""
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.counterfactual as cf  # noqa: E402
import eval.run_eval as ev  # noqa: E402

QUERIES = ROOT / "eval" / "vault_recall_queries.yaml"
RECORDS = ROOT / "eval" / "counterfactual_perturbations.yaml"
GENERATOR = ROOT / "eval" / "counterfactual.py"


def _specs():
    return yaml.safe_load(QUERIES.read_text())["queries"]


# The rater is sized against the live corpus, never a literal (#763 clause 4).
# This used to carry `MIN_CORPUS_N = 78` — the floor the trend audit's 80%-power
# claim needs (#1319, the exact McNemar search in scripts/eval_trend_stats.py) —
# and that assertion belonged where the corpus is under review, not in the
# rater's suite: here it could only fail for a reason this file has no power to
# fix, so every corpus resize reddened the counterfactual tests, and a suite
# that reddens on an unrelated change is a suite nobody reads. The power floor is
# still enforced, once, in tests/test_eval_corpus_guard.py::
# test_the_gold_set_is_big_enough_for_its_own_power_claim via
# GOLD_SET_MIN_QUERIES.
CORPUS_N = len(_specs())


# ── the committed perturbation set ───────────────────────────────────────────

def test_one_perturbation_per_query_with_the_declared_shape():
    """One variant per query, one changed axis each, over the WHOLE corpus.

    The count is the live corpus size, never a literal 20. The generator was
    written for the 20-query corpus and its records raise `ValueError` for an id
    with no PLAN entry, so a corpus change that forgets the rater leaves the
    nightly scoring one number of queries while its summary reports
    `counterfactual_n_moved` for another — one summary describing two different
    corpora. The corpus-size floor behind the trend audit's power claim (#1319)
    is guarded in `tests/test_eval_corpus_guard.py`, where resizing the corpus is
    the change under review; asserting it here could only redden the rater's own
    suite for a reason this file has no power to fix (#763 clause 4).
    """
    specs = {s["id"]: s for s in _specs()}
    recs = cf.load_records(RECORDS)
    assert len(recs) == len(specs) == CORPUS_N, (
        f"records {len(recs)}, corpus {CORPUS_N}")
    assert set(recs) == set(specs)
    for qid, rec in recs.items():
        for key in ("axis_changed", "old_value", "new_value",
                    "expected_to_move", "expected_pinned"):
            assert key in rec, f"{qid} missing {key}"
        assert rec["axis_changed"] in cf.AXES, (qid, rec["axis_changed"])
        assert isinstance(rec["expected_to_move"], list)
        assert isinstance(rec["expected_pinned"], list)


def test_plan_covers_every_query_and_nothing_outside_it():
    """The PLAN/corpus join, as a set difference that names EVERY gap, plus the
    size comparison a corpus resize moves together with it (#763 clause 4).

    `build_perturbations` raises on the first id with no plan entry, so it stops
    there; a growth that added 67 queries would be reported one id at a time.
    """
    specs = _specs()
    ids = {s["id"] for s in specs}
    assert sorted(set(cf.PLAN) - ids) == [], "PLAN entries with no query in the corpus"
    assert sorted(ids - set(cf.PLAN)) == [], "corpus queries with no PLAN entry"
    assert len(cf.PLAN) == len(specs), (
        f"the plan holds {len(cf.PLAN)} entries, the live corpus "
        f"{len(specs)} — the rater and the eval are describing different sets")


def test_cli_check_reports_no_drift():
    """`python eval/counterfactual.py --check`: the committed file IS the generator.

    Goes through the same entry point a person runs rather than only the
    in-process comparison, so a `--write` that emits something `--check` cannot
    reproduce — a header, a key order, a path the generator would not write — is
    caught here.
    """
    out = subprocess.run([sys.executable, str(GENERATOR), "--check"],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "records match the generator" in out.stdout, out.stdout


def test_alias_resolver_indexes_the_distinct_canonical_row(tmp_path):
    """`select distinct canonical` returns a ROW, not a string.

    `{c.lower() for c in con.execute("select distinct canonical ...")}` calls
    `.lower()` on a one-column tuple and raises AttributeError, so `--verify` —
    the only audit of an entity swap's chosen sibling — had never run against a
    real store. Every sibling claim in this file rested on `_resolver`, a
    hand-written dict that cannot fail this way. Built on a real sqlite file
    because the defect IS the row shape.
    """
    db = tmp_path / "kg.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table aliases (surface_lc text, canonical text)")
    con.executemany("insert into aliases values (?, ?)",
                    [("thunderbird mcp", "Thunderbird Service"),
                     ("qwen3-tts", "Qwen3-TTS")])
    con.commit()
    con.close()
    resolve = cf.alias_resolver(db)
    assert resolve("thunderbird mcp") == "Thunderbird Service"   # via a surface row
    assert resolve("Qwen3-TTS") == "Qwen3-TTS"                   # via the canonical set
    assert resolve("no such thing") is None


def test_verify_siblings_reaches_a_verdict_on_the_committed_corpus(tmp_path):
    """The audit runs, and returns 0 unverified swaps, on the real alias table.

    With the resolver fixed, every entity-axis record must name a swapped-in
    value that is a DISTINCT canonical — the risk-1 rule #537 spells out for
    exactly these labels. Falls back to a hand-built store when the derived
    knowledge store is absent, so the assertion is about the code path rather
    than the build; the entity swaps in the committed corpus are the ones the
    fallback store cannot vouch for, and it says so by being named here.
    """
    db = cf.kg_db_path()
    if not db.exists():
        db = tmp_path / "kg.sqlite"
        con = sqlite3.connect(db)
        con.execute("create table aliases (surface_lc text, canonical text)")
        con.executemany("insert into aliases values (?, ?)",
                        [(cf._norm(r["new_value"]), r["new_value"])
                         for r in cf.load_records(RECORDS).values()
                         if r["axis_changed"] == cf.ENTITY_AXIS])
        con.commit()
        con.close()
    recs = list(cf.load_records(RECORDS).values())
    unverified = cf.verify_siblings(recs, cf.alias_resolver(db))
    assert [r["id"] for r in unverified] == [], unverified


def test_a_scored_run_reports_counterfactual_n_over_the_whole_corpus(monkeypatch):
    """One scored run over the live corpus rates EVERY query, not 20 of them.

    The clause this pins is the one a yaml-only growth cannot satisfy: the
    summary reports `counterfactual_n_moved`, and if any query id is missing
    from the plan the runtime degrades it to `counterfactual_error: "no
    perturbation record for this query id"` and drops it from the denominator.
    The nightly then prints one number over 20 queries and another over 87 and
    reads as though it described a single corpus. So the whole production loop
    runs here — `run_eval` with only the retriever injected, then `summarize` —
    and its denominator has to equal the corpus size.
    """
    specs = _specs()

    def retrieve(params):
        # The rater reads only fact-attributed entities and their text, so the
        # stub echoes the query text as the ATTRIBUTED ENTITY: an entity swap
        # then finds its swapped-in sibling among the rows the variant added, a
        # swap with no named target sees a different attributed set, and a
        # pinned constraint matches the same single text in both arms.
        return {"documents": [{"path": "knowledge/whatever.md"}],
                "entities": [], "entity_facts": [],
                "facts": [{"entity": params["query"], "text": "one fact"}],
                "graph_neighbors_used": []}

    monkeypatch.setattr(ev, "_vault_recall", lambda params, **kw: retrieve(params))
    out = ev.summarize(ev.run_eval(specs, limit=len(specs)))["overall"]
    assert out["n_queries"] == CORPUS_N
    assert out["counterfactual_n_moved"] == CORPUS_N, (
        "the scored run rated fewer queries than the corpus holds — the rater "
        "is describing a smaller corpus than the eval is")
    assert out["counterfactual_moved_rate"] == 1.0
    # A twin that was never recalled would leave the pinned leg empty, so a
    # non-zero pinned denominator is the evidence both arms ran.
    assert out["counterfactual_n_pinned"] > 0


def test_each_perturbation_changes_exactly_one_named_constraint():
    """The variant differs from the original by exactly one substitution.

    A generator that quietly changes two things measures two things, and the
    axis attribution — the whole point of the rater — is gone.
    """
    specs = {s["id"]: s for s in _specs()}
    for qid, rec in cf.load_records(RECORDS).items():
        query = specs[qid]["query"]
        old, new = rec["old_value"], rec["new_value"]
        assert old.lower() in query.lower(), f"{qid}: old_value not in the query"
        rebuilt = cf.apply_perturbation(query, rec)
        assert rebuilt != query, f"{qid}: perturbation changed nothing"
        assert rebuilt == cf.apply_perturbation(query, rec), f"{qid}: not deterministic"
        before = query.lower().count(old.lower())
        after = rebuilt.lower().count(old.lower())
        # an overlap (QMD -> QMD Index) legitimately leaves the substring inside
        # the replacement; anything else means the swap did not take.
        assert after == before - 1 or old.lower() in new.lower(), qid
        assert rebuilt.lower().count(new.lower()) >= 1, qid


def test_generator_is_deterministic_and_matches_the_committed_file():
    """The generator is committed and reproducible; the records are committed.

    A record set that re-derived itself from live graph data each night would
    change the measurement underneath the trend line — the same defect the
    nightly compare step guards against for the query set itself.
    """
    built = {r["id"]: r for r in cf.build_perturbations(_specs())}
    committed = cf.load_records(RECORDS)
    assert set(built) == set(committed)
    for qid, rec in built.items():
        assert rec == committed[qid], f"{qid}: committed record drifted from generator"


def test_entity_swap_siblings_are_declared_as_alias_table_pairs():
    """Risk 1 in #541: an unfair sibling manufactures false failures, so every
    entity-axis pair declares where its sibling came from. Resolution against
    the live table is a separate, runnable check (verify_siblings) — this one
    stays hermetic so the suite does not need the store."""
    recs = cf.load_records(RECORDS)
    entity_axis = {q: r for q, r in recs.items() if r["axis_changed"] == "entity"}
    assert entity_axis, "no entity-swap perturbations — the #537 evidence cannot exist"
    for qid, rec in entity_axis.items():
        assert rec.get("sibling_source") == "kg_alias_table", qid
        assert rec["old_value"] != rec["new_value"], qid
    for rec in recs.values():
        if rec["axis_changed"] != "entity":
            assert "sibling_source" not in rec, "non-entity axis claims a sibling"


def test_sibling_verifier_flags_a_pair_that_is_not_in_the_table():
    """The audit half of risk 1, tested against an injected resolver rather than
    the live 3.9k-row table."""
    table = {"knowledge graph": "Knowledge Graph", "entity graph": "Entity Graph",
             "kg": "Knowledge Graph", "qmd": "QMD"}
    resolve = lambda name: table.get((name or "").strip().lower())
    good = {"axis_changed": "entity", "old_value": "Knowledge Graph",
            "new_value": "Entity Graph"}
    invented = {"axis_changed": "entity", "old_value": "Knowledge Graph",
                "new_value": "Frobnicator Prime"}
    # two surfaces of ONE canonical is a rename, not a sibling swap
    rename = {"axis_changed": "entity", "old_value": "Knowledge Graph", "new_value": "KG"}
    assert cf.verify_siblings([good], resolve) == []
    bad = cf.verify_siblings([invented, rename, good], resolve)
    assert [r["new_value"] for r in bad] == ["Frobnicator Prime", "KG"]


def test_non_entity_axes_are_not_held_to_the_sibling_audit():
    """date/qualifier/artifact swaps name no sibling; the audit has nothing to
    check and must not report them."""
    rec = {"axis_changed": "date", "old_value": "this week", "new_value": "last quarter"}
    assert cf.verify_siblings([rec], lambda name: None) == []


def test_perturbation_generation_touches_no_ground_truth(tmp_path):
    """Guardrail in #541: generation must not touch the query set or the corpus.

    The generator is a pure function of the query text plus the checked-in axis
    plan; it opens no store and writes only where it is told to.
    """
    before = QUERIES.read_text()
    built = cf.build_perturbations(_specs())
    assert QUERIES.read_text() == before
    out = tmp_path / "perts.yaml"
    cf.emit_records(built, out)
    assert cf.load_records(out) == {r["id"]: r for r in built}


# ── the two metrics ──────────────────────────────────────────────────────────

def _result(entities, fact_text="f"):
    return {"facts": [{"entity": e, "text": f"{fact_text} {e}"} for e in entities],
            "graph_expanded_facts": [], "graph_neighbors_used": [],
            "documents": [{"path": "a/b.md"}]}


def test_moved_is_scored_on_retrieved_entities_not_doc_paths():
    """Risk 3 in #541: the doc corpus is the same vault, so a doc-level pinned
    comparison would read as spurious success. Only entity/fact output counts."""
    rec = {"axis_changed": "entity", "old_value": "Knowledge Graph",
           "new_value": "Entity Graph", "expected_to_move": ["Entity Graph"],
           "expected_pinned": ["Data Pipeline"]}
    orig = _result(["Knowledge Graph", "Data Pipeline"])
    moved_entities = _result(["Entity Graph", "Data Pipeline"])
    same_entities = _result(["Knowledge Graph", "Data Pipeline"])

    out = cf.score_pair(rec, orig, ["Old Seed"], moved_entities, ["New Seed"])
    assert out["counterfactual_moved"] is True
    assert out["counterfactual_pinned"] is True

    stayed = cf.score_pair(rec, orig, ["Old Seed"], same_entities, ["New Seed"])
    assert stayed["counterfactual_moved"] is False
    assert stayed["counterfactual_pinned"] is True
    assert stayed["retrieved_unchanged"] is True
    # every arm above returned the identical document path: docs are not signal
    assert stayed["retrieved"] == stayed["retrieved_variant"]


def test_pinned_fails_when_an_unchanged_axis_churns():
    """The backward half. A retriever that re-pulls everything on any edit is
    not sensitive to the constraint — it is noise, and a moved-only rater would
    score it a perfect 1.0."""
    rec = {"axis_changed": "entity", "old_value": "vLLM", "new_value": "TensorRT-LLM",
           "expected_to_move": ["TensorRT-LLM"], "expected_pinned": ["lloyd"]}
    orig = _result(["vLLM", "Lloyd"], fact_text="same")
    churned = _result(["TensorRT-LLM", "Lloyd"], fact_text="different")
    out = cf.score_pair(rec, orig, ["vLLM"], churned, ["TensorRT-LLM"])
    assert out["counterfactual_moved"] is True
    assert out["counterfactual_pinned"] is False


def test_unscored_pinned_is_none_and_never_a_vacuous_pass():
    """A query with nothing left to pin contributes to moved_rate only. Scoring
    it as a pass would inflate pinned_rate with queries that tested nothing."""
    rec = {"axis_changed": "entity", "old_value": "QMD", "new_value": "QMD Index",
           "expected_to_move": ["QMD Index"], "expected_pinned": []}
    out = cf.score_pair(rec, _result(["QMD"]), ["QMD"], _result(["QMD Index"]), ["QMD Index"])
    assert out["counterfactual_moved"] is True
    assert out["counterfactual_pinned"] is None
    assert out["pinned_unscored"] is True


def test_non_entity_axes_score_moved_as_any_change():
    """date / qualifier / artifact axes name no new entity row, so the moved
    half is "the pulled entity-level output differs at all"."""
    rec = {"axis_changed": "date", "old_value": "this week", "new_value": "last quarter",
           "expected_to_move": [], "expected_pinned": ["Autonomy System"]}
    changed = cf.score_pair(rec, _result(["Autonomy System"]), ["s"],
                            _result(["Autonomy Data Pipeline"]), ["s"])
    assert changed["counterfactual_moved"] is True
    unchanged = cf.score_pair(rec, _result(["Autonomy System"]), ["s"],
                              _result(["Autonomy System"]), ["s"])
    assert unchanged["counterfactual_moved"] is False
    assert unchanged["counterfactual_pinned"] is True


def test_move_tolerates_the_stores_canonical_name_but_not_an_old_row():
    """Two things pull in opposite directions and the added-set settles both.

    Tolerant, because the store's canonical for 'Semantic Entity Resolution' is
    'semantic-entity-resolution-via-graph-embeddings' — refusing to call that a
    move manufactures a false failure (risk 1). Restricted to the rows the
    variant attributed and the original did not, because 'QMD' was already
    attributed when the query said QMD, and crediting it for a swap to
    'QMD Index' would score a no-change run as sensitive.
    """
    rec = {"axis_changed": "entity", "old_value": "QMD", "new_value": "QMD Search",
           "expected_to_move": ["QMD Search"], "expected_pinned": []}

    def res(entities):
        return _result(entities)

    # already-attributed 'QMD' must not satisfy a swap to 'QMD Search'
    no_move = cf.score_pair(rec, res(["QMD"]), ["QMD"], res(["QMD"]), ["QMD Search"])
    assert no_move["counterfactual_moved"] is False

    # a newly-attributed row whose canonical is longer still counts
    long_canon = {"facts": [{"entity": "semantic-entity-resolution-via-graph-embeddings",
                             "text": "x"}],
                  "graph_expanded_facts": [], "graph_neighbors_used": [],
                  "documents": []}
    out = cf.score_pair({"axis_changed": "entity", "old_value": "Entity Resolution Sweep",
                        "new_value": "Semantic Entity Resolution",
                        "expected_to_move": ["Semantic Entity Resolution"],
                        "expected_pinned": []},
                       res(["Entity Resolution Sweep"]), ["s"], long_canon, ["s"])
    assert out["counterfactual_moved"] is True


# ── the record the nightly compare step reads ────────────────────────────────

def test_summarize_carries_both_rates_and_its_own_n():
    """Each rate is averaged over the queries scoreable for that half, so
    summary carries those denominators beside the numbers."""
    def rec(moved, pinned):
        return {"id": "q", "category": "single", "latency_ms": 100.0, "error": None,
                "scoring": {"entity_hit": True, "doc_hit": True, "entity_recall": 1.0,
                            "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 1.0,
                            "fact_entity_recall": 1.0, "first_doc_rank": 1,
                            "counterfactual_moved_rate": moved,
                            "counterfactual_pinned_rate": pinned}}
    o = ev.summarize([rec(1.0, 1.0), rec(0.0, None), rec(1.0, 0.0)])["overall"]
    assert o["counterfactual_moved_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert o["counterfactual_pinned_rate"] == pytest.approx(0.5)
    assert o["counterfactual_n_moved"] == 3
    assert o["counterfactual_n_pinned"] == 2


def test_summarize_emits_both_keys_even_with_no_counterfactual_data():
    """Old records and the automod baseline arm carry no perturbation block.
    The keys must still be present — so the nightly compare reads a key that
    exists — with null rather than 0.0: absent is not zero."""
    rec = {"id": "q", "category": "single", "latency_ms": 10.0, "error": None,
           "scoring": {"entity_hit": True, "doc_hit": True, "entity_recall": 1.0,
                       "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 1.0,
                       "fact_entity_recall": 1.0, "first_doc_rank": 1}}
    o = ev.summarize([rec])["overall"]
    assert "counterfactual_moved_rate" in o and "counterfactual_pinned_rate" in o
    assert o["counterfactual_moved_rate"] is None
    assert o["counterfactual_n_moved"] == 0


def test_failure_labels_name_the_swaps_that_left_the_seed_set_alone():
    """The deliverable is a labelled defect list, and the #537-relevant label is
    an entity-name swap the seed extractor did not notice."""
    labelled = cf.label_failures([
        {"id": "swap-blind", "counterfactual": {"axis_changed": "entity",
                                                "seed_moved": False,
                                                "counterfactual_moved": False,
                                                "counterfactual_pinned": True}},
        {"id": "swap-seen", "counterfactual": {"axis_changed": "entity",
                                               "seed_moved": True,
                                               "counterfactual_moved": True,
                                               "counterfactual_pinned": True}},
        {"id": "pinned-churn", "counterfactual": {"axis_changed": "qualifier",
                                                  "seed_moved": False,
                                                  "counterfactual_moved": True,
                                                  "counterfactual_pinned": False}},
    ])
    by_id = {row["id"]: row for row in labelled}
    assert by_id["swap-blind"]["label"] == "entity_swap_seed_set_unchanged"
    assert "swap-blind" in cf.identity_keying_evidence(labelled)
    assert "swap-seen" not in cf.identity_keying_evidence(labelled)
    assert by_id["swap-seen"]["label"] is None
    assert by_id["pinned-churn"]["label"] == "pinned_axis_churned"


# ── #763: the pinned leg's two coverage errors, and its denominator ──────────

# The three entity swaps whose swapped-in sibling keeps part of the original
# term, so the surviving part is a constraint the perturbed query really does
# still name. `backlog-363` pins "Backlog" rather than the two-word
# "Backlog Item" for a measured reason, recorded at the PLAN entry: the original
# arm attributes `backlogtask363`, in which the normalized "backlogitem" is not a
# substring, so the two-word form scores `present=False->True` — the clause-2
# defect imported instead of removed.
SURVIVING_TYPE_PINS = {"qmd": "QMD", "backlog-363": "Backlog",
                       "entity-resolution-sweep": "Entity Resolution"}

# The rows each arm attributed for these three queries in
# `eval/baselines/nightly-20260921-20260921-063958.json`
# (`counterfactual.retrieved` and `.retrieved_variant`), copied here so the
# non-vacuity claim below is checkable without a store: a pin must match a row on
# BOTH sides, because `score_pair` also passes a pin that matched nothing in
# either arm (`present=False->False`) — the vacuity #763 leaves to a person.
MEASURED_ROWS = {
    "qmd": (["bun", "ldlibrarypath", "lexicalsearchbackend", "qmd",
             "qwen3embedding", "structuredsearch"],
            ["industrialaiassistant", "ldlibrarypath", "lexicalsearchbackend",
             "qmd", "qmdsearch", "qwen3embedding", "structuredsearch"]),
    "backlog-363": (["backlogtask363", "currenttask", "task363",
                     "task363tgsragmultihopreasoning", "task380kgdensification",
                     "tgsrag", "tgsragretrievallevers"],
                    ["313", "backlogitem313", "task313"]),
    "entity-resolution-sweep": (
        ["aliastable", "cleanup", "entityresolution", "entityresolutionscript",
         "entityresolutionsweep", "lloydv4classifier", "task320"],
        ["autonomytask67", "cleanup", "entityresolution", "entityresolutionscript",
         "lloydv4classifier", "robomd",
         "semanticentityresolutionviagraphembeddings"]),
}


def _perturbed_text(qid: str) -> str:
    """The variant text the generator produces for one corpus query."""
    spec = next(s for s in _specs() if s["id"] == qid)
    _axis, old, new, _move, _pins = cf.PLAN[qid]
    return cf.apply_perturbation(spec["query"],
                                 {"old_value": old, "new_value": new, "id": qid})


def _stubbed_run(monkeypatch, specs):
    """The production loop with only the retriever injected.

    The stub echoes the query text as the ATTRIBUTED ENTITY, so presence is
    judged against each arm's own text and nothing else — the same shape
    `test_a_scored_run_reports_counterfactual_n_over_the_whole_corpus` uses.
    """
    def retrieve(params):
        return {"documents": [{"path": "knowledge/whatever.md"}],
                "entities": [], "entity_facts": [],
                "facts": [{"entity": params["query"], "text": "one fact"}],
                "graph_neighbors_used": []}

    monkeypatch.setattr(ev, "_vault_recall", lambda params, **kw: retrieve(params))
    records = ev.run_eval(specs, limit=len(specs))
    return records, ev.summarize(records)


def test_the_three_entity_swaps_whose_type_survives_their_swap_are_scored(monkeypatch):
    """Clause 1: `qmd`, `backlog-363` and `entity-resolution-sweep` each carry a
    non-empty `expected_pinned` that survives into its own perturbed query, so
    those three ids no longer score `nothing_pinned`.

    Three things are asserted, because the first alone would pass on a pin
    invented out of thin air:

      * the pin is the PLAN's declared pin, and a case-insensitive substring of
        the variant text the generator will actually run;
      * it matches a row BOTH arms attributed in the shipping baseline, so the
        pass is evidence about retrieval rather than a pin that matched nothing
        in either arm;
      * the production loop now returns a non-null pinned leg for the id, which
        is what takes it out of `nothing_pinned` in `counterfactual_failures`.
    """
    specs = _specs()
    for qid, pin in SURVIVING_TYPE_PINS.items():
        assert cf.PLAN[qid][4] == [pin], (qid, cf.PLAN[qid][4])
        perturbed = _perturbed_text(qid)
        assert pin.lower() in perturbed.lower(), (qid, pin, perturbed)
        orig_rows, var_rows = MEASURED_ROWS[qid]
        assert cf._match(pin, {cf._norm(r) for r in orig_rows}), (qid, orig_rows)
        assert cf._match(pin, {cf._norm(r) for r in var_rows}), (qid, var_rows)

    records, summary = _stubbed_run(monkeypatch, specs)
    by_id = {r["id"]: r for r in records}
    labels = {row["id"]: row["label"] for row in cf.label_failures(records)}
    for qid in SURVIVING_TYPE_PINS:
        block = by_id[qid]["counterfactual"]
        assert block["pinned_unscored"] is False, qid
        assert block["counterfactual_pinned"] is not None, qid
        assert labels[qid] != "nothing_pinned", (qid, labels[qid])
    # The denominator still means "actually scored", so it equals the entries
    # that declare a pin — derived from the plan, never a number in this file.
    declared = sum(1 for entry in cf.PLAN.values() if entry[4])
    o = summary["overall"]
    assert o["counterfactual_n_pinned"] == declared, (o["counterfactual_n_pinned"],
                                                     declared)
    assert o["counterfactual_n_moved"] == len(specs), o["counterfactual_n_moved"]


def test_no_plan_entry_pins_a_term_its_own_swap_deletes_beyond_the_five_named():
    """Clause 2: the only entries whose declared pin is absent from their own
    perturbed query are the five enumerated non-entity ones, so
    `autonomy-pipeline` no longer pins "autonomy".

    `autonomy-pipeline` is the live false alarm this clause closes. Its swap
    rewrites "describe the autonomy pipeline end to end" into "describe the Data
    Pipeline end to end", so the variant text holds no "autonomy" to hold
    constant, and `nightly-20260917` and `nightly-20260921` each booked the row as
    `pinned_axis_churned / autonomy: present=True->False` — a defect reported on
    the one axis that was supposed to move. The five survivors are qualifier and
    artifact entries whose pin names an entity the query implies rather than
    repeats; they are legitimate expectations and `PINS_ABSENT_BY_DESIGN` says so
    per entry. No ENTITY-axis entry may be in that set, because on that axis a pin
    outside the variant text is by construction the constraint that moved.
    """
    specs = _specs()
    offenders = cf.pins_absent_from_their_own_swap(specs)
    assert sorted(offenders) == sorted(cf.PINS_ABSENT_BY_DESIGN), offenders
    assert set(cf.PINS_ABSENT_BY_DESIGN) == {
        "qwen38-local-serving", "yaml-scalar-block-indent",
        "check-that-cannot-see-its-input", "self-referential-check-catalogue",
        "skill-mining-to-promotion"}
    assert cf.PLAN["autonomy-pipeline"][4] == [], cf.PLAN["autonomy-pipeline"]
    assert "autonomy-pipeline" not in offenders
    # The pin is gone; the swap's own `old_value` keeps the word, because the
    # word is what the swap moves.
    assert "autonomy" not in str(cf.PLAN["autonomy-pipeline"][4]), \
        cf.PLAN["autonomy-pipeline"][4]
    for qid in cf.PINS_ABSENT_BY_DESIGN:
        assert cf.PLAN[qid][0] != cf.ENTITY_AXIS, (
            qid, "an entity-axis entry pins a term its own swap deletes")

    # The audit must be able to FAIL. `autonomy-pipeline` is the offender this
    # clause removes, so it is injected back as a synthetic sixth entry here and
    # has to come back named — otherwise the five could be passing because the
    # function never finds anything.
    probe_specs = [{"id": "probe", "query": "describe the autonomy pipeline"}]
    probe_plan = {"probe": ("entity", "autonomy pipeline", "Data Pipeline",
                            ["Data Pipeline"], ["autonomy"])}
    assert cf.pins_absent_from_their_own_swap(probe_specs, probe_plan) == {
        "probe": ["autonomy"]}
    # ... and a pin that DOES survive its own swap is not reported.
    ok_plan = {"probe": ("entity", "autonomy pipeline", "Data Pipeline",
                         ["Data Pipeline"], ["pipeline"])}
    assert cf.pins_absent_from_their_own_swap(probe_specs, ok_plan) == {}


def test_the_printed_counterfactual_line_shows_each_rate_out_of_the_query_set(monkeypatch,
                                                                              capsys):
    """Clause 3: the console summary prints `pinned=0.90 (n=50/81)`, and moved's
    out-of-total is on the same line.

    `(n=50)` alone — what `eval/run_eval.py` has printed since `af1e8c1`
    (2026-09-09) — shows the count but not the coverage. The two legs of this
    rater are scored over DIFFERENT query populations by design (an entry with no
    `expected_pinned` feeds moved only), so without the out-of-total a reader
    comparing the two trend lines is comparing an unknown pair of populations,
    which is the reading #763 was filed on. The unscored remainder has to stay
    VISIBLE: the fraction carries the shortfall, and the residual pair is not
    quietly promoted into the denominator.
    """
    specs = _specs()
    records, summary = _stubbed_run(monkeypatch, specs)
    ev.print_table(records, summary)
    out = capsys.readouterr().out
    o = summary["overall"]
    total = o["n_queries"]
    line = next((ln for ln in out.splitlines() if "counterfactual:" in ln), None)
    assert line is not None, out
    # Both denominators are fractions of the run's own query total, in the order
    # (moved, pinned), with no bare `(n=N)` left behind.
    assert re.findall(r"\(n=(\d+)/(\d+)\)", line) == [
        (str(o["counterfactual_n_moved"]), str(total)),
        (str(o["counterfactual_n_pinned"]), str(total))], line
    # The residual unscored pair shows as a shortfall rather than as coverage:
    # the pinned denominator is strictly under the total, and the two entries the
    # item leaves to a person are why.
    assert o["counterfactual_n_pinned"] < total, line
    for qid in ("inner-voice", "vault-recall"):
        assert cf.PLAN[qid][4] == [], qid


def test_the_counterfactual_suite_is_sized_by_the_corpus_it_reads():
    """Clause 4: no hard-coded corpus size anywhere in this file, so a corpus
    resize cannot redden the rater's suite.

    This file used to carry `MIN_CORPUS_N = 78` and assert the live corpus was at
    least it, which meant the rater's own tests failed whenever an unrelated job
    trimmed or grew `eval/vault_recall_queries.yaml`. 78 is the exact-McNemar
    floor for a 0.10 paired change at 80 % power (`scripts/eval_trend_stats.py`),
    a property of the CORPUS, and it is guarded once in
    `tests/test_eval_corpus_guard.py::test_the_gold_set_is_big_enough_for_its_own_power_claim`,
    where a resize is the change under review. What belongs HERE is agreement:
    plan entries, committed records and corpus queries all one size.
    """
    specs = _specs()
    assert len(cf.PLAN) == len(specs), (len(cf.PLAN), len(specs))
    assert len(cf.load_records(RECORDS)) == len(specs), (
        len(cf.load_records(RECORDS)), len(specs))
    src = Path(__file__).read_text()
    literals = re.findall(r"^(?:MIN_)?CORPUS_N\s*=\s*\d+", src, re.MULTILINE)
    assert literals == [], (
        f"the suite hard-codes a corpus size: {literals} — assert against "
        "`len(_specs())` instead")
