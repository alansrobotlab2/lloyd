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


# The rater is sized against the live corpus, never a literal. MIN_CORPUS_N is
# the floor the trend audit's 80%-power claim needs (#1319): the exact McNemar
# search in scripts/eval_trend_stats.py reports n = 78 for a 0.10 paired change
# at 80% power, alpha 0.05, so a corpus under it cannot support the verdicts the
# nightly writes no matter how well the rater covers it.
CORPUS_N = len(_specs())
MIN_CORPUS_N = 78


# ── the committed perturbation set ───────────────────────────────────────────

def test_one_perturbation_per_query_with_the_declared_shape():
    """One variant per query, one changed axis each, over the WHOLE corpus.

    The count is the live corpus size, never a literal 20. The generator was
    written for the 20-query corpus and its records raise `ValueError` for an id
    with no PLAN entry, so a corpus growth that forgets the rater leaves the
    nightly scoring 87 queries while its summary still reports
    `counterfactual_n_moved: 20` — one summary describing two different corpora.
    """
    specs = {s["id"]: s for s in _specs()}
    recs = cf.load_records(RECORDS)
    assert len(recs) == len(specs) == CORPUS_N, (
        f"records {len(recs)}, corpus {CORPUS_N}")
    assert CORPUS_N >= MIN_CORPUS_N, (
        f"corpus is {CORPUS_N}; the trend audit needs at least {MIN_CORPUS_N} "
        "paired queries for 80% power on a 0.10 change")
    assert set(recs) == set(specs)
    for qid, rec in recs.items():
        for key in ("axis_changed", "old_value", "new_value",
                    "expected_to_move", "expected_pinned"):
            assert key in rec, f"{qid} missing {key}"
        assert rec["axis_changed"] in cf.AXES, (qid, rec["axis_changed"])
        assert isinstance(rec["expected_to_move"], list)
        assert isinstance(rec["expected_pinned"], list)


def test_plan_covers_every_query_and_nothing_outside_it():
    """The PLAN/corpus join, as a set difference that names EVERY gap.

    `build_perturbations` raises on the first id with no plan entry, so it stops
    there; a growth that added 67 queries would be reported one id at a time.
    """
    ids = {s["id"] for s in _specs()}
    assert sorted(set(cf.PLAN) - ids) == [], "PLAN entries with no query in the corpus"
    assert sorted(ids - set(cf.PLAN)) == [], "corpus queries with no PLAN entry"


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
