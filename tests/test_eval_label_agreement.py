"""The retrieval-blind second labeler, and the two numbers it produces (#654).

Clauses 1-3 of item #654. What each test refuses to let the instrument become:

1. `label_corpus` must label EVERY query in the corpus, narrowing candidates
   deterministically from the entity-name table and the vault paths, shuffling under
   a recorded seed, and stamping which model produced the labels. A ceiling whose
   labeler is unknown is a number about an unnamed model.
2. The two agreement numbers must be over the CORPUS'S OWN label counts — the file's
   94 entity labels across 66 queries and 139 doc labels, not the 43 an earlier
   draft of the item quoted — with every disagreeing label named, and both numbers
   must recompute from the artifact alone.
3. The labeling payload must contain no retrieval output. A labeler shown what
   retrieval returned agrees with retrieval, not with the gold, and the ceiling
   becomes a mirror that reports 1.0 for any retriever.

Everything runs against an injected stub callable, so none of this needs an engine
up. That is deliberate: `secondary_enabled: false` means the item's named engine is
a stopped process, and an instrument that cannot be exercised without a human
switching a GPU on is an instrument nobody can test.
"""
import contextlib
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Resolved, never assumed: `.venvs/` is gitignored, so a worktree — including the one
# the gate runs the suite in — has no venv of its own, and a CLI test that assumed one
# would be skipping exactly where a CLI regression would otherwise be caught.
_VENV = ROOT / ".venvs" / "lloyd" / "bin" / "python"
PY = _VENV if _VENV.exists() else Path(sys.executable)

import eval.label_agreement_ceiling as lac  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────

ENTITIES = ["Knowledge Graph", "Backlog Item #363", "TGS-RAG Implementation",
            "Qwen3.6-35B-A3B", "Mission Control", "Inner Voice", "LiveKit",
            "Zeroclaw", "vLLM", "Whisper"]

VAULT_DOCS = ["knowledge/software/kg-maintenance-tasks.md",
              "projects/lloyd/voice/voice-source-dave-cullen.md",
              "backlog/363-tgs-rag-implementation.md",
              "knowledge/software/mission-control-dashboard.md",
              "memory/learnings/DAILY_NOTES.md",
              "projects/lloyd/retrieval/entity-resolution.md"]


def _corpus(tmp_path):
    """A four-query corpus with 5 entity labels and 4 doc labels.

    Deliberately NOT 94/139: the point of clause 2 is that the denominators come
    from whatever corpus is on disk, so the fixture has its own small counts and
    every assertion below names them.
    """
    queries = [
        {"id": "kg-tasks", "query": "What maintenance tasks does the Knowledge Graph "
                                    "need?",
         "category": "knowledge",
         "expect_entities": ["Knowledge Graph", "KG Maintenance Tasks"],
         "expect_docs": ["knowledge/software/kg-maintenance-tasks.md"]},
        {"id": "backlog-363", "query": "What was decided about Backlog Item 363?",
         "category": "projects",
         "expect_entities": ["Backlog Item #363"],
         "expect_docs": ["backlog/363"]},
        {"id": "voice-clone", "query": "Which voice model is the clone source?",
         "category": "voice",
         "expect_entities": ["Voice Model"],
         "expect_docs": ["projects/lloyd/voice/voice-source-dave-cullen.md"]},
        {"id": "mc-dashboard", "query": "How does Mission Control show the dashboard?",
         "category": "software",
         "expect_entities": ["Mission Control"],
         "expect_docs": ["knowledge/software/mission-control-dashboard.md"]},
    ]
    path = tmp_path / "queries.yaml"
    import yaml
    path.write_text(yaml.safe_dump({"queries": queries}, sort_keys=False),
                    encoding="utf-8")
    return queries, path


def _run(tmp_path, *, pick=None, seed=lac.SEED, identity=None):
    queries, corpus_path = _corpus(tmp_path)
    labeler = lac.stub_labeler(pick)
    art = lac.label_corpus(
        labeler=labeler,
        labeler_identity=identity or {"kind": "stub", "model": "stub-model-x",
                                      "endpoint": "in-process"},
        queries=queries, entity_names=ENTITIES, vault_paths=VAULT_DOCS,
        seed=seed, corpus_path=corpus_path)
    return art, queries, corpus_path


@contextlib.contextmanager
def _sandbox_store(path: Path):
    """Point the process-wide KG default at `path`, then move the pointer back.

    The restore is a pointer move plus `reset()`, never `configure(saved)`.
    `configure` is the *provisioning* route — `app/kg_store.py:1281` says so in
    its own docstring: "unlike `store()` it creates an absent database" — and
    inside an automod round the saved path is
    `<round>/home/lloyd-data/_pipeline/vault-derived/kg.sqlite`, which does not
    exist there. Restoring with `configure` therefore CREATED a 0-row database
    that `tests/test_eval_corpus_guard.py`, run later in the same data root, read
    as the store of record: gate rung `tests` failed
    `test_every_expect_entities_name_is_contained_in_a_store_entity_name` and
    `test_the_default_route_is_the_live_store_the_eval_scores_against` on all
    three attempts of round SM_20260924_063720, and the base probe filed them as
    PRE-EXISTING because the probe re-ran in that same poisoned data root (its
    own shape is #1436). `tests/conftest.py:342` `_isolate_default_store` is the
    precedent: reset, move `_default_path`, never provision on the way out.
    """
    from app import kg_store
    saved = kg_store._default_path
    kg_store.reset()
    try:
        yield kg_store.configure(path)
    finally:
        kg_store.reset()
        kg_store._default_path = saved


# ── clause 1: label every query, deterministically, from the two namespaces ──

def test_labels_every_query_and_records_the_labeler_identity(tmp_path):
    art, queries, _ = _run(tmp_path)
    assert [r["id"] for r in art["queries"]] == [q["id"] for q in queries], (
        "a ceiling measured over a subset of the corpus is a ceiling for a "
        "different corpus; every query must be labelled")
    assert art["labeler"] == {"kind": "stub", "model": "stub-model-x",
                              "endpoint": "in-process"}, (
        "which model produced the ceiling determines what the number means, so the "
        "artifact that carries the ceiling must carry the model name and endpoint")
    assert art["seed"] == lac.SEED, (
        "the shuffle is only reproducible if the seed that produced it is stored")
    for row in art["queries"]:
        assert row["entity_candidates"] and row["doc_candidates"]
        assert row["primary_entities"] == next(
            q["expect_entities"] for q in queries if q["id"] == row["id"])


def test_candidates_are_deterministic_and_shuffled_under_the_recorded_seed(tmp_path):
    art_a, *_ = _run(tmp_path)
    art_b, *_ = _run(tmp_path)
    assert ([r["entity_candidates"] for r in art_a["queries"]]
            == [r["entity_candidates"] for r in art_b["queries"]]), (
        "the same seed must reproduce the same candidate ORDER, or 'recompute from "
        "the artifact' is a claim about a run that cannot be repeated")
    art_c, *_ = _run(tmp_path, seed=lac.SEED + 1)
    differs = any(a["entity_candidates"] != c["entity_candidates"]
                  for a, c in zip(art_a["queries"], art_c["queries"]))
    assert differs, ("a different seed that does not change the order means the "
                     "shuffle is not seeded per query, so a candidate list could be "
                     "read as agreement in an order that never varied")


def test_entity_candidates_come_from_the_entity_name_table_by_token_overlap(tmp_path):
    with _sandbox_store(tmp_path / "kg.sqlite") as store:
        for name in ENTITIES:
            store.entities.register(name)
        names = lac.entity_name_table()
        assert set(names) == set(ENTITIES), (
            "the candidate namespace must be the store's own entity-name table, not "
            "a list assembled here — a namespace assembled here is a second source "
            "of truth about what entities exist")
        picks = lac.entity_candidates("What does Mission Control show?", names,
                                      cap=3, query_id="mc")
        assert "Mission Control" in picks, (
            "narrowing by query-token overlap must keep the name the query names, "
            "or the gold is never offered and the labeler cannot agree with it")
        assert len(picks) == 3
        assert lac.entity_candidates("zzz nothing matches zzz", names, cap=20,
                                     query_id="x") == lac.entity_candidates(
            "zzz nothing matches zzz", names, cap=20, query_id="x"), (
            "a query with no overlap anywhere must still get a deterministic pool, "
            "not an empty one that would report agreement 0 for a non-reason")


def test_restoring_the_default_store_never_provisions_an_absent_database(tmp_path,
                                                                        monkeypatch):
    """The teardown shape, pinned because it broke the GATE, not a test.

    Stands in for a round home: the process default points at a database that is
    NOT there (`LLOYD_DATA` is the round's own data root on every gate rung —
    `scripts/automod/gate.py:_child_env`). Restoring must move the pointer. A
    `configure(saved)` restore creates the file, and a 0-row store in the data
    root is what made the corpus guard refuse inside every gate run of the
    previous round of this item, which the base probe then mis-filed as
    PRE-EXISTING because it re-ran in that same data root.
    """
    from app import kg_store
    absent_round_db = tmp_path / "home" / "lloyd-data" / "_pipeline" / "vault-derived" / "kg.sqlite"
    absent_round_db.parent.mkdir(parents=True)
    monkeypatch.setattr(kg_store, "_default_path", absent_round_db)
    with _sandbox_store(tmp_path / "kg.sqlite") as store:
        store.entities.register("Knowledge Graph")
        assert store.path == tmp_path / "kg.sqlite"
    assert kg_store._default_path == absent_round_db, (
        "the pointer has to come back where it was, or the next test on this "
        "worker labels against a namespace that no longer exists")
    assert not absent_round_db.exists(), (
        "restoring the KG default wrote a database where there was none: "
        "configure() provisions an absent path, so in a round home this is the "
        "0-row store tests/test_eval_corpus_guard.py refuses on — the `tests` "
        "rung failure that blocked round SM_20260924_063720 at heads e9a4be19, "
        "a7b8891e and eddd4d75")


def test_doc_candidates_come_from_vault_paths_only(tmp_path):
    vault = tmp_path / "vault"
    (vault / "knowledge").mkdir(parents=True)
    (vault / "knowledge" / "alpha.md").write_text("x", encoding="utf-8")
    (vault / "knowledge" / "beta.md").write_text("x", encoding="utf-8")
    (vault / ".git").mkdir()
    (vault / ".git" / "hook.md").write_text("x", encoding="utf-8")
    paths = lac.vault_markdown_paths(vault)
    assert paths == ["knowledge/alpha.md", "knowledge/beta.md"], (
        "the doc namespace is the vault's own .md paths, and .git must not join it: "
        "a counter that includes .git/** is this repo's catalogued false-verdict")
    with pytest.raises(SystemExit):
        lac.vault_markdown_paths(tmp_path / "nowhere")


# ── clause 2: agreement over the corpus's own counts, disagreements named ────

def test_agreement_uses_the_corpus_label_counts_not_a_constant(tmp_path):
    art, _, _ = _run(tmp_path)
    counts = art["corpus"]["counts"]
    assert counts == {"queries": 4, "entity_queries": 4, "entity_labels": 5,
                      "doc_queries": 4, "doc_labels": 4}, (
        "the denominators are read from the corpus on disk; the item is explicit "
        "that the figure is 94 entity labels across 66 queries and 139 doc labels, "
        "not the 43 an earlier draft of itself quoted")
    a = art["agreement"]
    assert a["entity_labels"] == counts["entity_labels"]
    assert a["doc_labels"] == counts["doc_labels"]
    # The stub picks candidate 1 of each list, so the rate is determined and small.
    assert a["entity_label_agreement"] == round(
        a["entity_labels_agreed"] / counts["entity_labels"], 4)
    assert a["doc_label_agreement"] == round(
        a["doc_labels_agreed"] / counts["doc_labels"], 4)


def test_every_disagreeing_label_is_named(tmp_path):
    art, queries, _ = _run(tmp_path)
    a = art["agreement"]
    disagreements = a["disagreements"]["entity"]
    assert len(disagreements) == counts_entity_disagree(art)
    for d in disagreements:
        assert {"id", "gold", "second_labels", "gold_offered"} <= set(d)
        assert d["gold"], "an unnamed disagreement is not a disagreement set"
    # Agreement + disagreement must account for every label, or the list is partial.
    assert len(disagreements) == (a["entity_labels"] - a["entity_labels_agreed"])
    assert len(a["disagreements"]["doc"]) == (a["doc_labels"]
                                             - a["doc_labels_agreed"])


def counts_entity_disagree(art):
    return art["agreement"]["entity_labels"] - art["agreement"]["entity_labels_agreed"]


def test_agreement_is_a_full_lower_bound_when_gold_was_never_offered(tmp_path):
    """A label the labeler was never SHOWN cannot be agreed with, and counting it as
    disagreement would read candidate narrowing as label noise — the direction that
    inflates `score / ceiling` and excuses a real defect."""
    art, *_ = _run(tmp_path)
    a, c = art["agreement"], art["ceiling"]
    rows = a["disagreements"]["entity"]
    hidden = [d for d in rows if not d["gold_offered"]]
    assert hidden, (
        "no disagreement row says a gold was never offered, so every assertion below "
        "this one runs over an empty set and the fixture stopped exercising the "
        "branch — widen the stub corpus's entity cap, do not delete this test")

    # The denominator is the offered labels, not the corpus's labels: an unmeasurable
    # gold must leave the bottom of the fraction, where it would dilute agreement, and
    # must not join the numerator either.
    assert a["entity_labels"] == 5 and a["entity_labels_offered"] == 3, (
        f"expected a fixture with two golds the candidate pool hid, got "
        f"{a['entity_labels']} labels / {a['entity_labels_offered']} offered")
    assert len(hidden) == a["entity_labels"] - a["entity_labels_offered"], (
        f"{len(hidden)} rows name a hidden gold but the denominator moved by "
        f"{a['entity_labels'] - a['entity_labels_offered']}: narrowing the candidate "
        "pool is being booked as label noise, which lowers the ceiling and inflates "
        "score/ceiling — the instrument manufacturing its own excuse")
    assert a["entity_label_agreement"] == pytest.approx(
        a["entity_labels_agreed"] / a["entity_labels_offered"]), (
        "the rate is divided by something other than the labels that could have been "
        "agreed with")
    # And the bookkeeping closes: of the labels that WERE offered, each is either
    # agreed or a disagreement — no third bucket where a label vanishes.
    scored = [d for d in rows if d["gold_offered"]]
    assert a["entity_labels_agreed"] + len(scored) == a["entity_labels_offered"], (
        f"{a['entity_labels_agreed']} agreed + {len(scored)} disagreed != "
        f"{a['entity_labels_offered']} offered, so an offered label is in neither "
        "bucket and the agreement rate is not a partition of what was measurable")

    # Query level: a query contributes to the ceiling only if at least one of its
    # golds was in the pool; one with none is named as unmeasurable, not dropped.
    zero = {r["id"] for r in art["queries"] if r["entity_labels_offered"] == 0}
    assert zero == set(c["excluded"]["entity"]), (
        f"queries with no offered gold {sorted(zero)} vs named-unmeasurable "
        f"{sorted(c['excluded']['entity'])}: a query silently leaving the ceiling's "
        "denominator is how a ceiling gets read over a corpus nobody labelled")
    assert set(c["excluded"]["entity"]) <= {r["id"] for r in art["queries"]}


def test_recomputing_from_the_stored_artifact_reproduces_both_numbers(tmp_path):
    art, _, _ = _run(tmp_path)
    out_dir = tmp_path / "baselines" / lac.ARTIFACT_DIR_NAME
    out_dir.mkdir(parents=True)
    path = out_dir / "label-agreement-20260924-000000.json"
    # Written without the derived blocks, so recomputation is the only source.
    stripped = {k: v for k, v in art.items() if k not in ("agreement", "ceiling")}
    path.write_text(json.dumps(stripped), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    again = lac.agreement(loaded)
    assert again["entity_label_agreement"] == art["agreement"]["entity_label_agreement"]
    assert again["doc_label_agreement"] == art["agreement"]["doc_label_agreement"]
    assert (len(again["disagreements"]["entity"])
            == len(art["agreement"]["disagreements"]["entity"]))
    assert (len(again["disagreements"]["doc"])
            == len(art["agreement"]["disagreements"]["doc"]))


def test_loader_refuses_a_stub_artifact_and_an_edited_one(tmp_path):
    art, _, _ = _run(tmp_path)
    out_dir = tmp_path / "baselines" / lac.ARTIFACT_DIR_NAME
    out_dir.mkdir(parents=True)
    path = out_dir / "label-agreement-20260924-000000.json"
    path.write_text(json.dumps(art), encoding="utf-8")
    with pytest.raises(lac.ArtifactRefused) as exc:
        lac.load_artifact(path)
    assert "stub" in str(exc.value).lower(), (
        "a stub's agreement is a property of the stub; a nightly must never inherit "
        "it as a measured ceiling")
    assert lac.load_artifact(path, allow_stub=True)["agreement"]
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["labeler"] = {"kind": "engine", "model": "someone-elses-run",
                         "endpoint": "http://nowhere"}
    edited["agreement"]["entity_label_agreement"] = 0.99
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(lac.ArtifactRefused) as exc:
        lac.load_artifact(path, allow_stub=True)
    assert "recompute" in str(exc.value).lower(), (
        "an artifact whose stored figure does not reproduce from its own stored "
        "labels is an edited artifact, and must be refused rather than reported")


def test_loader_refuses_a_ceiling_whose_gold_labels_moved(tmp_path):
    art, _, _ = _run(tmp_path)
    path = tmp_path / "label-agreement-20260924-000001.json"
    path.write_text(json.dumps(art), encoding="utf-8")
    good = lac.load_artifact(path, allow_stub=True,
                             expect_labels_sha256=art["corpus"]["labels_sha256"])
    assert good["_path"] == str(path)
    with pytest.raises(lac.ArtifactRefused) as exc:
        lac.load_artifact(path, allow_stub=True, expect_labels_sha256="0" * 16)
    assert "labels" in str(exc.value).lower(), (
        "commit 9b028e9 re-pointed 22 gold entity names under an unchanged query-id "
        "set; a ceiling measured against those labels describes labels that no "
        "longer exist")


# ── clause 3: the payload carries no retrieval output ───────────────────────

def test_prompt_carries_no_retrieval_output(tmp_path):
    """A labeler shown what retrieval returned agrees with retrieval, not with the
    gold. The clause is a literal string-absence check against a real baseline
    record's top-10 outputs, which is what the review can re-run."""
    # The live data root by name. `tests/conftest.py` deliberately points
    # `app.paths.DATA_ROOT` at a scratch root so a worktree cannot reach the running
    # system's store; a test that wants the live artifacts says so through
    # `PRODUCTION_DATA_ROOT`, the documented reader's route. Reading real records is
    # the point: the clause names `doc_paths_top10` and `fact_entities_top10` FROM A
    # BASELINE RECORD, so the check must run against values a real retrieval pass
    # produced rather than against an author-chosen lookalike.
    from app.data_root import PRODUCTION_DATA_ROOT
    srcs = sorted((PRODUCTION_DATA_ROOT / "eval" / "baselines")
                  .glob("nightly-*.json"))[-3:]
    assert srcs, (
        "no nightly baseline on this box to draw retrieval outputs from. An empty "
        "list would make every string-absence assertion below vacuously true, which "
        "is the false zero this repo has catalogued repeatedly — so this is a "
        "failure, not a skip")
    recs = []
    for src in srcs:
        blob = json.loads(src.read_text(encoding="utf-8"))
        recs += [r for r in (blob.get("records") or []) if not r.get("error")]
    # Exactly the two fields the clause names, at the path where a real record keeps
    # them (`result_summary`, since #1011 stopped writing the whole result).
    retrieval_values = set()
    for r in recs:
        summary = r.get("result_summary") or {}
        retrieval_values.update(str(v) for v in (summary.get("doc_paths_top10") or []))
        retrieval_values.update(str(v) for v in (summary.get("fact_entities_top10") or []))
    retrieval_values = {v for v in retrieval_values if len(v) > 4}
    assert len(retrieval_values) > 10, (
        "the positive control: the retrieval values must actually be present in the "
        "records, or a 0-hit prompt check is a missing-path 0")

    queries, corpus_path = _corpus(tmp_path)
    prompts = []
    seen_paths = []

    def spy(request):
        prompts.append(request["prompt"])
        seen_paths.append(request["entity_candidates"]
                          + request["doc_candidates"])
        return {"text": '{"entities": [], "docs": []}'}

    art = lac.label_corpus(labeler=spy,
                           labeler_identity={"kind": "stub", "model": "stub"},
                           queries=queries, entity_names=ENTITIES,
                           vault_paths=VAULT_DOCS, corpus_path=corpus_path)
    assert len(prompts) == len(queries)
    # Two exclusions, both honest, and both counted so the check cannot be vacuous.
    # A value that is one of the OFFERED CANDIDATES belongs in the prompt — the doc
    # namespace is the vault path namespace, which is where retrieval's paths come
    # from too, so a retrieved path can legitimately sit in the menu. A value that is
    # in the QUERY TEXT belongs there as well. What must never appear is a retrieval
    # value that is neither offered nor asked about: that is the run's result riding
    # along in the payload.
    checked = 0
    for row, text, offered in zip(queries, prompts, seen_paths):
        for value in retrieval_values:
            # Substring-aware on purpose: `backlog/363` is a retrieval path AND a
            # prefix of an offered candidate, and an exact-membership test would call
            # that a leak. A value that is contained in something the labeler was
            # shown anyway is in the prompt because of the menu, not because of the
            # retrieval result.
            if any(value in cand for cand in offered) or value in row["query"]:
                continue
            checked += 1
            assert value not in text, (
                f"a labeler prompt contains {value!r}, which a retrieval run "
                f"returned; the ceiling would then measure agreement with retrieval")
    assert checked >= 100, (
        f"only {checked} retrieval values were actually put to the test — below the "
        f"point where a passing check would mean anything")
    for text in prompts:
        for banned in ("doc_paths_top10", "fact_entities_top10", "scoring",
                       "gold", "expect_entities"):
            assert banned not in text
    # The gold itself never reaches the labeler either: knowing the answer is a
    # weaker version of the same contamination. Substring-aware for the same reason
    # as above — the corpus writes gold docs as path prefixes (`backlog/363`), and
    # such a prefix is legitimately a prefix of a candidate the labeler must be shown.
    for row, text in zip(art["queries"], prompts):
        menu = row["entity_candidates"] + row["doc_candidates"]
        for gold in row["primary_entities"] + row["primary_docs"]:
            if gold in row["query"] or any(gold in cand for cand in menu):
                continue
            assert gold not in text, (
                f"the labeler can see the gold label {gold!r} it is being asked to "
                f"independently recover, so its agreement is not a second opinion")


def test_build_prompt_signature_cannot_receive_a_retrieval_result():
    """The structural half of clause 3. `build_prompt` takes the query and two
    candidate lists — so there is no parameter through which a baseline record, a
    scored result or a gold label could reach the labeler, and a future caller
    cannot leak one by accident without changing the signature."""
    import inspect
    params = list(inspect.signature(lac.build_prompt).parameters)
    assert params == ["query", "entity_candidates", "doc_candidates"]
    prompt = lac.build_prompt("q", ["Alpha", "Beta"], ["one.md", "two.md"])
    assert "1. Alpha" in prompt and "2. two.md" in prompt
    assert "order means nothing" in prompt, (
        "the prompt must say the order is meaningless: option-order sensitivity is "
        "large enough to wash a measured result out to 50/50")


def test_reply_is_read_as_candidate_indices_not_free_text():
    """An out-of-range or unparsable selection is EMPTY and stamped, never guessed.
    Names instead of indices would let a 35B model invent an entity id the store
    does not hold and have it counted as a second opinion."""
    picks = lac.parse_reply('{"entities": [2, 9], "docs": [1]}', n_entities=3, n_docs=2)
    assert picks["entities"] == [1] and picks["docs"] == [0]
    assert picks["parse_failed"] is False
    junk = lac.parse_reply("I cannot answer that.", n_entities=3, n_docs=2)
    assert junk["entities"] == [] and junk["docs"] == []
    assert junk["parse_failed"] is True


def test_ceiling_is_the_surrogate_scored_by_the_scorers_own_rule(tmp_path):
    """The ceiling must be in the units of the metric it bounds — that is what makes
    `score / ceiling` a real position — and it must not gain points from seed
    extraction the labeler never performed."""
    spec = {"query": "What did the Backlog Item 363 audit decide?",
            "expect_entities": ["Backlog Item #363"],
            "expect_docs": ["backlog/363"]}
    hit = lac.surrogate_scoring(spec, ["Backlog Item #363"], ["backlog/363-x.md"])
    assert hit["entity_hit"] is True and hit["doc_hit"] is True
    assert hit["entity_recall"] == 1.0
    miss = lac.surrogate_scoring(spec, ["Mission Control"], ["nowhere.md"])
    assert miss["entity_hit"] is False and miss["doc_hit"] is False
    # `seeds=[]`: the query text contains "Backlog Item 363", so scoring with
    # seeds=None would re-extract that string and call it a hit the labeler never
    # chose, inflating the ceiling by an effect no labeler produced.
    assert miss["entity_recall"] == 0.0, (
        "a surrogate that re-extracts seeds from the query text scores the query, "
        "not the labeler")


def test_unmeasurable_leg_is_named_not_zeroed(tmp_path):
    art, *_ = _run(tmp_path)
    c = art["ceiling"]
    assert c["values"]["fact_entity_recall_avg"] is None
    assert "fact" in c["unmeasured"]["fact_entity_recall_avg"], (
        "the fact leg has no gold-side surrogate: the labeler labels entities and "
        "paths, not fact rows, so a ceiling there would be 0 by construction and "
        "would read as a measured floor")
    assert c["kind"] == lac.CEILING_KIND


def test_split_half_reports_the_queries_it_could_actually_run_on(tmp_path):
    art, *_ = _run(tmp_path)
    sh = lac.split_half(art, resamples=20)
    assert sh["entity"]["queries"] == 1, (
        "only the one two-label query can be split; reporting the figure without "
        "that count is how a resample of zeros becomes a headline ceiling")
    assert sh["doc"]["queries"] == 0 and sh["doc"]["reason"]


# ── the seams the review rung named unverified (round 2 of #654) ────────────
#
# The review refused the first round with three named gaps, all of the same shape:
# code that crosses a process boundary while every test injects a callable and so
# never reaches the boundary. Two of them are here, one in
# `tests/test_eval_ceiling_fields.py`:
#
#   1. `resolve_engine` -> `app.config` (the chat-completions URL plus the
#      `secondary_enabled` gate). Its first draft read `from app.config import config`,
#      which does not exist, inside a `try:`, so the ImportError was caught and the
#      engine was reported disabled whatever config.yaml said. No test could have
#      caught that, because none called it.
#   2. `http_labeler` -> POST <base>/v1/chat/completions. Now: one test through an
#      injected transport that records the payload, and one through a real loopback
#      socket on the default transport, so the injection is never the only proof.
#   3. `skills/retrieval-eval/SKILL.md` Step 2b -> `--print`, a documented command the
#      parser rejected with `unrecognized arguments`. Below, with the flag list read
#      out of the skill's own committed HEAD rather than a list I typed here.

import re  # noqa: E402
import shlex  # noqa: E402

import app.config as cfg_mod  # noqa: E402
import eval.run_eval as ev  # noqa: E402


# ── seam 1: resolve_engine -> app.config ─────────────────────────────────────

#: Shaped like the real `models:` table (`primary`/`secondary` keys carrying
#: `alias` and `base_url`) so that the two app.config helpers this function
#: delegates to — `_get_model_cfg`'s alias scan and `_resolve_model_name` — run
#: for real. They are not replaced: patching them would test a resolution this
#: module does not perform.
def _fake_config(monkeypatch, *, secondary_enabled: bool = False,
                 djev_base: str = "http://127.0.0.1:8010",
                 primary_base: str = "http://127.0.0.1:8096") -> None:
    models = {
        "primary": {"alias": "primary", "base_url": primary_base,
                    "env": {"ANTHROPIC_BASE_URL": primary_base}},
        "secondary": {"alias": "secondary", "base_url": "http://127.0.0.1:8091",
                      "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8091"}},
    }
    monkeypatch.setattr(cfg_mod, "MODEL_CONFIGS", models, raising=False)
    monkeypatch.setattr(cfg_mod, "CONFIG", {
        "secondary_enabled": secondary_enabled,
        "models": models,
        "djev": {"enabled": True, "base_url": djev_base},
    }, raising=False)


def test_resolve_engine_reads_the_real_config_mapping(monkeypatch):
    """The `secondary_enabled` gate must read the mapping config.yaml actually
    loads into (`app/config.py:288 CONFIG`), not a lowercase `config` that no
    version of that module has ever had.

    The falsifying case is the one the first round shipped: with
    `secondary_enabled: true` the labeler must NOT refuse. Reading a nonexistent
    attribute inside `try:` made it refuse anyway, so a person who paid the human
    cost of enabling the engine would still have been told it was off — a guard
    reporting a verdict from its own missing input.
    """
    _fake_config(monkeypatch, secondary_enabled=False)
    with pytest.raises(SystemExit) as off:
        lac.resolve_engine("secondary")
    assert "LABELER_ENGINE_DISABLED" in str(off.value)

    _fake_config(monkeypatch, secondary_enabled=True)
    url, model = lac.resolve_engine("secondary")
    assert url == "http://127.0.0.1:8091/v1/chat/completions"
    assert model == "secondary", (
        "the model name written into the artifact is what a later reader is told "
        "produced the ceiling; an unresolved alias would misname it")


def test_resolve_engine_reads_base_url_from_either_config_key(monkeypatch):
    """A model slot may carry its endpoint as `base_url` or only as the
    `ANTHROPIC_BASE_URL` env override. `app/secondary_models.py:_endpoint` reads
    both; reading one alone makes a live engine look unconfigured, which is a
    false outage of the same family as the finding this round answers."""
    _fake_config(monkeypatch)
    cfg_mod.MODEL_CONFIGS["primary"].pop("base_url")
    url, _ = lac.resolve_engine("primary")
    assert url == "http://127.0.0.1:8096/v1/chat/completions"


def test_resolve_engine_learns_the_served_djev_name_over_a_real_socket(monkeypatch):
    """Clause 3, and the reason #1655's ruling has been unrunnable since it was
    written: `config.yaml` carries `djev.base_url` and no `djev.model`, and adding one
    is a human edit to a file the loop may not write. The old code refused and told a
    person to go read `GET <base>/v1/models` and paste the answer into a flag — so the
    name WAS discoverable, from the endpoint that the run is about to ask, and the
    resolver asked nothing.

    This goes through `urllib` and a loopback server rather than an injected fetcher,
    because the thing that has to be true is that the GET is really sent to
    `/v1/models` on the configured base and its reply really decoded: an injected
    fetcher would let a resolver that never built the URL pass. The name it learns is
    the reply's `id` (`djev` on this box, per the live probe of 2026-09-28, whose
    `root` is the long `nvidia-diffusiongemma-26B-A4B-it-NVFP4` path — a name nobody
    should be copying into config).
    """
    import http.server
    import threading

    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            got["path"] = self.path
            raw = json.dumps({"object": "list", "data": [
                {"id": "djev", "object": "model",
                 "root": "/models/nvidia-diffusiongemma-26B-A4B-it-NVFP4"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        _fake_config(monkeypatch, djev_base=base)
        assert lac.resolve_engine("djev") == (
            f"{base}/v1/chat/completions", "djev"), (
            "the pair the artifact records as `model`/`endpoint` must be the chat URL "
            "and the name THAT endpoint answers to: vLLM refuses a name it is not "
            "serving, and the artifact's provenance is the name it was asked under")
        assert got["path"] == "/v1/models", (
            "the probe has to be a GET of /v1/models on the configured base, not a "
            "guess about it")
    finally:
        srv.shutdown()
        srv.server_close()


def test_resolve_engine_refuses_a_djev_run_only_when_the_endpoint_cannot_answer(monkeypatch):
    """The refusal survives, narrowed to the state that needs a person: the endpoint
    is down, so there is no served name to learn and no run to start. (The clause is
    asymmetric on purpose — resolve, or refuse with a reason that cannot be fixed by
    starting a service. A refusal that fires while the engine is up is what made this
    item's ruling unexecutable.)

    The second half is the other direction of the same rule: a configured
    `djev.model` is a person's choice and must win WITHOUT a probe, so the fetcher
    here raises if it is ever called."""
    _fake_config(monkeypatch, djev_base="http://127.0.0.1:9")
    with pytest.raises(SystemExit) as exc:
        lac.resolve_engine("djev")
    msg = str(exc.value)
    assert "LABELER_MODEL_UNKNOWN" in msg
    assert "http://127.0.0.1:9/v1/models" in msg, "must name the probe that failed"
    assert "--labeler-model" in msg and "--labeler-url" in msg, (
        "and the two flags that resolve it without a config edit")

    def no_probe(url):
        raise AssertionError(f"probed {url} although config.yaml names the model")

    _fake_config(monkeypatch)  # back to the live-shaped base, with a model set below
    monkeypatch.setitem(cfg_mod.CONFIG["djev"], "model", "diffusiongemma-26b-a4b")
    assert lac.resolve_engine("djev", models_fetcher=no_probe) == (
        "http://127.0.0.1:8010/v1/chat/completions", "diffusiongemma-26b-a4b")


@pytest.mark.parametrize("data,marker", [
    ([{"id": "djev-a"}, {"id": "djev-b"}], "LABELER_MODEL_AMBIGUOUS"),
    ([], "LABELER_MODEL_UNKNOWN"),
], ids=["two models", "empty list"])
def test_resolve_engine_refuses_when_the_models_list_cannot_name_one_model(
        monkeypatch, data, marker):
    """Two states where the endpoint is up and still cannot answer the question, each
    with the reason that fits it.

    More than one model is the only case here that is a real CHOICE — which model
    produces the ceiling changes what the number means — so it gets its own code and
    names both ids, rather than silently taking `data[0]`, which would decide the
    meaning of the artifact by list order. An empty list is not a choice: it names no
    model, so it is the unreachable case wearing a 200, and takes the UNKNOWN reason
    that tells a person to start or point the engine.
    """
    _fake_config(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        lac.resolve_engine("djev", models_fetcher=lambda url: {"data": data})
    msg = str(exc.value)
    assert marker in msg
    assert "http://127.0.0.1:8010/v1/models" in msg
    if marker == "LABELER_MODEL_AMBIGUOUS":
        assert "djev-a" in msg and "djev-b" in msg, (
            "a refusal to choose must show what it was refusing to choose between")


def test_resolve_engine_names_the_engines_it_will_accept(monkeypatch):
    _fake_config(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        lac.resolve_engine("gpt-9")
    assert "LABELER_ENGINE_UNKNOWN" in str(exc.value)
    for name in lac.ENGINE_NAMES:
        assert name in str(exc.value)


def test_resolve_engine_against_the_shipped_config():
    """No patch: the function must work against the config.yaml this repo ships,
    which is the only way a test notices that the key it reads has been renamed.
    Asserted against a value parsed from the file by a separate reader, so the
    test does not inherit the same lookup it is checking."""
    import yaml
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    base = raw["models"]["primary"]["base_url"].rstrip("/")
    assert lac.resolve_engine("primary") == (f"{base}/v1/chat/completions", "primary")
    # And the switch this box is actually on: secondary_enabled is false here, so
    # the labeler must refuse. If a person flips it, this assertion flips with the
    # file — which is the point of reading it rather than asserting it.
    if not raw.get("secondary_enabled"):
        with pytest.raises(SystemExit, match="LABELER_ENGINE_DISABLED"):
            lac.resolve_engine("secondary")


# ── seam 2: http_labeler -> POST /v1/chat/completions ────────────────────────

def test_http_labeler_builds_a_chat_completions_request_and_decodes_the_reply():
    """The payload shape, temperature 0, and the index clamp — through the real
    builder and decoder, with only the POST replaced. A stub *labeler* would have
    replaced all four at once, which is exactly what the review rung called out."""
    seen: list[tuple[str, dict]] = []

    def transport(url: str, payload: dict) -> dict:
        seen.append((url, payload))
        return {"choices": [{"message": {"content": '{"entities": [1, 99], "docs": [2]}'}}]}

    label = lac.http_labeler("http://127.0.0.1:9/v1/chat/completions",
                             "qwen-labelled-by-test", transport=transport)
    out = label({"id": "x", "prompt": "PROMPT-MARKER choose from the lists"})

    url, payload = seen[0]
    assert url == "http://127.0.0.1:9/v1/chat/completions"
    assert payload["model"] == "qwen-labelled-by-test"
    assert payload["temperature"] == 0.0, (
        "a labeler that samples is not a second rater; the ceiling would measure "
        "the sampler")
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["messages"][1]["content"] == "PROMPT-MARKER choose from the lists"
    assert set(payload) == {"model", "messages", "temperature", "max_tokens"}, (
        "a new payload key is a new channel into the labeler; clause 3 says only "
        "the prompt-derived text may reach it, so additions must be a decision")
    assert out["text"] == '{"entities": [1, 99], "docs": [2]}'

    picks = lac.parse_reply(out["text"], n_entities=2, n_docs=2)
    assert picks["entities"] == [0], "1-based candidate numbers become 0-based indices"
    assert picks["docs"] == [1]
    assert 99 not in picks["entities"], "an out-of-range index must be clamped away"


@pytest.mark.parametrize("body", [
    {"choices": [{"message": {"content": "   "}}]},
    {"choices": [{"message": {}}]},
    {"choices": []},
    {},
])
def test_http_labeler_calls_an_empty_completion_an_instrument_failure(body):
    """An engine that answers 200 with nothing is not disagreeing with the gold.
    Returned as `""`, every query becomes `parse_failed`, the artifact still
    writes, and a ceiling of nothing reads as a measured number over the whole
    corpus."""
    label = lac.http_labeler("http://127.0.0.1:9/v1/chat/completions", "m",
                             transport=lambda url, payload: body)
    with pytest.raises(RuntimeError, match="LABELER_EMPTY_RESPONSE"):
        label({"id": "x", "prompt": "PROMPT"})


def test_http_post_json_crosses_a_real_socket():
    """The injection point cannot be the only proof of the HTTP behaviour, or the
    test suite verifies a lambda and the nightly verifies nothing. This one goes
    through `urllib` and a loopback server, so the request encoding, the headers,
    the path and the decode are the code under test."""
    import http.server
    import threading

    reply = json.dumps({"choices": [{"message": {"content": "entities: 1\ndocs: 1"}}]})
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            got["path"] = self.path
            got["ctype"] = self.headers.get("Content-Type")
            got["body"] = json.loads(self.rfile.read(n).decode("utf-8"))
            raw = reply.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
    try:
        out = lac.http_labeler(url, "socket-served")({"id": "x", "prompt": "PROMPT"})
        assert out["text"] == "entities: 1\ndocs: 1"
        assert got["path"] == "/v1/chat/completions"
        assert got["ctype"] == "application/json"
        assert got["body"]["model"] == "socket-served"
        assert got["body"]["temperature"] == 0.0
        assert got["body"]["messages"][-1]["content"] == "PROMPT"
    finally:
        srv.shutdown()
        srv.server_close()


#: Copied byte for byte out of the live reply from `:8010/v1/chat/completions` on
#: 2026-09-28 (`temperature: 0` in the request, 400 out). Its seven parameter names
#: are what `_DIFFUSION_REJECTED_SAMPLING` lists, and it is kept verbatim rather than
#: paraphrased because the matcher reads the body for those words: a rewrite of this
#: string is a rewrite of the thing the endpoint actually says.
_SAMPLING_REFUSAL = (
    '{"error":{"message":"The temperature, min_p, seed, min_tokens, logit_bias, '
    'bad_words, and allowed_token_ids sampling parameters are not yet supported '
    'with diffusion models.","type":"BadRequestError","code":400}}')


def _post_that_refuses_then_accepts(replies: list[tuple[int, str]]):
    """A loopback chat-completions endpoint that answers each POST with the next
    (status, body) in `replies`, and records every request body it received. Real
    `urllib` on both sides: `HTTPError` is something `urlopen` raises, and whether the
    refused body is still READABLE off it (`exc.read()`) is exactly the part an
    injected transport cannot demonstrate."""
    import http.server
    import threading

    seen: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append(json.loads(self.rfile.read(n).decode("utf-8")))
            status, body = replies[min(len(seen) - 1, len(replies) - 1)]
            raw = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
    return srv, url, seen


def test_http_labeler_retries_once_without_the_sampling_parameters_a_400_names(
        capsys):
    """Clause 1, across the socket.

    What this refuses to let happen: the ruling on #1655 (`--engine djev`, chosen
    because it needs no pool pause) dying on query 1 of 86. The labeler sends
    `temperature: 0` because a labeler that samples is not a second rater, and the
    djev build rejects the word, not the value — so `urlopen` raised `HTTPError`, the
    exception escaped `http_labeler` before `LABELER_EMPTY_RESPONSE` could even be
    considered, and every query in the corpus failed rather than degraded. The fix is
    one retry with the named parameters dropped, and it has to keep the label: a
    labeler that returns nothing is the instrument failure the empty-reply check
    already exists to name.

    The refusal goes to stderr, not into the artifact, and that is asserted here
    because the alternative is the quieter bug: an artifact that recorded
    `temperature 0` for a request that was demonstrably sent without it would be a
    number describing a run that did not happen.
    """
    accepted = json.dumps(
        {"choices": [{"message": {"content": 'thought\n{"entities": [1], "docs": [2]}'}}]})
    srv, url, seen = _post_that_refuses_then_accepts([(400, _SAMPLING_REFUSAL),
                                                      (200, accepted)])
    try:
        out = lac.http_labeler(url, "djev")({"id": "x", "prompt": "PROMPT"})
    finally:
        srv.shutdown()
        srv.server_close()

    assert out["text"] == 'thought\n{"entities": [1], "docs": [2]}', (
        "the label survived the refusal — which is the whole clause")
    assert len(seen) == 2, f"expected one retry, saw {len(seen)} POSTs"
    assert seen[0]["temperature"] == 0.0, (
        "the first request is unchanged: the retry is a response to the endpoint's "
        "refusal, not a pre-emptive workaround")
    assert "temperature" not in seen[1], "the refused parameter must be gone"
    assert seen[1]["max_tokens"] == 300 and seen[1]["model"] == "djev", (
        "only what the 400 named is dropped; the keys this endpoint accepts are sent "
        "again untouched")
    assert lac.parse_reply(out["text"], n_entities=2, n_docs=2)["entities"] == [0], (
        "and the reply still parses through the `thought` prefix the diffusion build "
        "puts on its answers")
    err = capsys.readouterr().err
    assert "labeler-sampling-dropped" in err and "temperature" in err, (
        "a label produced without the temperature the module's docstring promises has "
        "to be visible where the run is watched")


@pytest.mark.parametrize("status,body,why", [
    (400, '{"error":{"message":"The model `djev-x` does not exist.","code":400}}',
     "a 400 about a name this payload sent nothing about"),
    (503, _SAMPLING_REFUSAL, "the sampling sentence arriving as a 503"),
], ids=["unrelated 400", "sampling text but not a 400"])
def test_http_labeler_does_not_retry_a_400_that_names_nothing_it_sent(
        status, body, why):
    """Clause 2's other half, and the half that makes the retry safe to have.

    A retry that fired on any 400 would turn a genuine request bug — a misnamed
    model, a malformed payload — into a second, stranger request and then report the
    failure of the SECOND one, which is how a guard eats the evidence of its own
    trigger. So the HTTPError propagates on the first POST, one request sent, no
    second chance. The 503 case pins the status test specifically: the same words from
    a different kind of failure are not the sampling refusal.
    """
    srv, url, seen = _post_that_refuses_then_accepts([(status, body)])
    try:
        with pytest.raises(urllib.error.HTTPError) as raised:
            lac.http_labeler(url, "djev")({"id": "x", "prompt": "PROMPT"})
    finally:
        srv.shutdown()
        srv.server_close()
    assert raised.value.code == status
    assert len(seen) == 1, f"retried anyway ({why}): {len(seen)} POSTs"
    assert seen[0]["temperature"] == 0.0


def test_http_labeler_sends_temperature_to_an_endpoint_that_takes_it():
    """Clause 2, the ordinary path: an endpoint that accepts sampling parameters is
    sent exactly the payload `http_labeler` has always built — one POST, `temperature`
    present at 0.0, and none of the seven names removed. The retry is invisible to a
    server that never refuses, which is what keeps this ceiling's `temperature 0`
    claim true for every engine that can honour it."""
    seen: list[dict] = []

    def transport(url: str, payload: dict) -> dict:
        seen.append(payload)
        return {"choices": [{"message": {"content": '{"entities": [1], "docs": []}'}}]}

    out = lac.http_labeler("http://127.0.0.1:9/v1/chat/completions", "qwen",
                           transport=transport)({"id": "x", "prompt": "PROMPT"})
    assert out["text"] == '{"entities": [1], "docs": []}'
    assert len(seen) == 1, "an endpoint that accepts sampling must not be retried"
    assert seen[0]["temperature"] == 0.0
    assert set(seen[0]) == {"model", "messages", "temperature", "max_tokens"}, (
        "same payload as before this round: the drop is a reaction to a refusal, not "
        "a new default shape")


# ── seam 3: the skill's documented command -> the parser ─────────────────────

VAULT = Path.home() / "obsidian"
SKILL_RELPATH = "skills/retrieval-eval/SKILL.md"

# Read the skill the way `tests/test_eval_trend_stats.py:810-826` does — from the
# vault's committed HEAD, via git, and FAIL when HEAD is unreadable rather than
# skip: a skipped acceptance check is a gate that reads green because nobody could
# look. Same residual limit, same admission: the vault is its own tree, so its HEAD
# advances independently of this commit.
@pytest.fixture(scope="module")
def skill_text() -> str:
    proc = subprocess.run(["git", "-C", str(VAULT), "show", f"HEAD:{SKILL_RELPATH}"],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        pytest.fail(
            f"clause 5 is unverifiable: `git -C {VAULT} show HEAD:{SKILL_RELPATH}` "
            f"failed ({proc.stderr.strip()[:200]})")
    return proc.stdout


_FLAG = re.compile(r"(?:^|\s)(--[\w-]+)")
_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# `$( )`, backticks and `${ }` are one shell WORD whatever they contain, and the
# shell resolves them before the interpreter sees argv. shlex.split does not know
# that: the skill's `--label nightly-$(date +%Y%m%d)` would split on the space
# inside the parentheses and hand `parse_args` a stray `+%Y%m%d)` token, reddening
# the suite over a command that runs fine. Blanking the substitution to one opaque
# token replays what the parser actually receives.
_SHELL_SUBST = re.compile(r"\$\([^)]*\)|`[^`]*`|\$\{[^}]*\}")


def _join_continuations(block: str) -> list[str]:
    r"""Merge backslash-continued lines: one shell command, however many lines it takes.

    The skill writes its `run_eval.py` command across three lines
    (`skills/retrieval-eval/SKILL.md:41-45`, `--label` and `--notes` on the
    continuations), and a line-by-line extractor sees only the script's own line,
    whose tail after the script name is ` \` — zero flags. `parse_args([])` accepts
    any parser, so the check that is supposed to catch a documented-but-unimplemented
    flag could not fail for that script at all. This is the review rung's finding on
    heads e9a4be19/a7b8891e/eddd4d75/028d5dfc, and the reason the non-vacuity control
    below now feeds it a continued command rather than only a one-line one.
    """
    out: list[str] = []
    pending = ""
    for raw in block.splitlines():
        line = raw.strip()
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        out.append((pending + line).strip())
        pending = ""
    if pending:
        out.append(pending.strip())
    return out


def _script_lines(skill: str, script: str) -> list[str]:
    """Comment-stripped text after `script` in each command of each code block.

    `_shell_subst` is applied here so a `$( … )` stays one argv token downstream
    (`.venvs/lloyd/bin/python eval/run_eval.py --label nightly-$(date +%Y%m%d)` is
    three shell words and FIVE tokens to `shlex.split`, and a `+%Y%m%d)` handed to
    argparse reddens the suite on a command that runs fine in a terminal). Flag
    extraction is unaffected: the pattern keys on the leading whitespace, and the
    placeholder is whitespace-free on both sides.
    """
    out: list[str] = []
    for block in _CODE_BLOCK.findall(skill):
        for line in _join_continuations(block):
            if script not in line or line.lstrip().startswith("#"):
                continue
            tail = _SHELL_SUBST.sub("SUBSTITUTED",
                                    line.split("#", 1)[0].split(script, 1)[1])
            out.append(tail.replace("\\", " "))
    return out


def documented_flags(skill: str, script: str) -> list[str]:
    """Every `--flag` the skill's code blocks pass to `script`.

    Extracted rather than listed, because a hand-copied list is a second source
    that can quietly stop agreeing with the skill — the exact defect the review
    rung found (`--print` documented, not implemented). The `<--summary` line in
    one of the skill's output blocks is why the pattern requires whitespace
    before the dashes: without it the extractor invents a `--summary` flag.
    """
    found: list[str] = []
    for tail in _script_lines(skill, script):
        found += _FLAG.findall(tail)
    return sorted(set(found))


def documented_invocations(skill: str, script: str) -> list[list[str]]:
    """The argument vectors of every command line the skill shows for `script`.

    Whole invocations, not just flags: a reporter copies the line, so an
    unfashionable positional or a value the parser refuses is the same failure as
    an unknown flag. `ENV=value` prefixes are dropped because the shell eats them
    before the parser sees them.
    """
    out: list[list[str]] = []
    for tail in _script_lines(skill, script):
        try:
            toks = shlex.split(tail)
        except ValueError:
            continue
        out.append([t for t in toks if not re.fullmatch(r"[A-Z_]+=\S+", t)])
    return out


def test_the_skill_command_extractor_itself_is_not_vacuous():
    """Deterministic control for the skill check below, with no vault read.

    Written against a string this file owns precisely because the real skill cannot
    be the control. Round 2 of this item first asserted `--print` must appear in the
    committed skill; `#1432` then rewrote that step (`~/obsidian` `d74211b9`) to
    describe the ceiling as unlanded instead of prescribing the command, and the
    assertion went red on a suite that had done nothing wrong — the red-at-main test
    class `#1396` exists to stop. So non-vacuity is proven here, on a synthetic
    block, and the vault-derived test below is allowed to find nothing without
    lying about what it checked.
    """
    block = ("text\n```bash\n"
             "python eval/label_agreement_ceiling.py --print --seed 7 --entity-cap 5\n"
             "```\n")
    flags = documented_flags(block, "eval/label_agreement_ceiling.py")
    assert flags == ["--entity-cap", "--print", "--seed"], (
        f"extracted {flags}: the whole point is that every flag of a documented "
        f"invocation is collected, so a partial extraction would let an "
        f"unimplemented flag through the check below")
    invs = documented_invocations(block, "eval/label_agreement_ceiling.py")
    assert invs == [["--print", "--seed", "7", "--entity-cap", "5"]], invs
    # A command for the OTHER script must not leak into this one's list: the two
    # paths share the substring `eval/`, so the match has to be on the file name.
    assert documented_invocations(block, "eval/run_eval.py") == []

    # The continuation case, which is the shape the real skill uses and the shape
    # this extractor used to silently empty out. A control over a one-line command
    # cannot see it — that control passed while the vault-derived check was vacuous.
    continued = ("```bash\n"
                 "cd ~/lloyd && .venvs/lloyd/bin/python eval/run_eval.py \\\n"
                 "  --label nightly-$(date +%Y%m%d) \\\n"
                 "  --notes \"nightly retrieval eval\"\n"
                 "```\n")
    cont_flags = documented_flags(continued, "eval/run_eval.py")
    assert cont_flags == ["--label", "--notes"], (
        f"extracted {cont_flags} from a backslash-continued command: an empty list "
        f"here is exactly the vacuity the review rung refused on — the invocation "
        f"reaches parse_args with no arguments and every parser on earth accepts that")
    cont_invs = documented_invocations(continued, "eval/run_eval.py")
    assert len(cont_invs) == 1, f"one command, three lines, got {cont_invs}"
    assert cont_invs[0] == ["--label", "nightly-SUBSTITUTED", "--notes",
                            "nightly retrieval eval"], (
        f"the shell substitution was not kept as one argv token: {cont_invs[0]}. "
        f"A shell hands `nightly-$(date +%Y%m%d)` over as ONE word; if the replay "
        f"splits it on the space inside the parentheses, the check fails on a "
        f"command that runs fine and the flag list grows a phantom positional")


@pytest.mark.parametrize("script", [
    "eval/label_agreement_ceiling.py",   # Step 2b's ceiling command
    "eval/run_eval.py",                  # the eval Step 2 runs, fields read by Step 3
])
def test_every_command_the_skill_documents_parses(skill_text, script):
    """Whatever the committed skill shows for these two scripts must be runnable.

    A command the skill documents and the parser rejects exits 2 with `unrecognized
    arguments`, which the reporter reads as "there is no ceiling" — the instrument's
    output never reaches the report and the gap looks like a null rather than a bug.
    That is exactly how the first round of this item was refused: `--print`
    documented, unimplemented.

    The vault is a separate tree and its prose is another item's to edit, so an empty
    extraction is a skip, not a failure — but only after the skill is proven to still
    name the script at all. A skill that stopped naming the file has stopped
    documenting the instrument, and that is the finding, not an absence of work:
    hence the two assertions rather than a bare skip.

    And when something IS extracted, every-empty-argv is refused below rather than
    replayed: `parse_args([])` is the one input that makes this check unable to fail,
    which is the failure mode the extractor used to have on the skill's own
    backslash-continued command.
    """
    build = lac.build_parser if "label_agreement" in script else ev.build_parser
    invs = documented_invocations(skill_text, script)
    if not invs:
        assert script in skill_text, (
            f"{script} appears nowhere in {SKILL_RELPATH}: the skill no longer "
            f"prescribes or describes the instrument at all, so the seam this test "
            f"covers has been removed rather than satisfied")
        pytest.skip(f"{SKILL_RELPATH} describes {script} without prescribing a "
                    f"command line (see ~/obsidian d74211b9, #1432); nothing to "
                    f"replay, and the CLI's own behaviour is pinned by "
                    f"test_the_print_flag_is_the_documented_no_write_report")
    for args in invs:
        try:
            build().parse_args(list(args))
        except SystemExit:
            pytest.fail(f"{script} {' '.join(args)} is documented in {SKILL_RELPATH} "
                        f"but its own parser rejects it, so the documented nightly "
                        f"command cannot run")


def _artifact_file(tmp_path, art):
    p = tmp_path / "label-agreement-20260924-000000.json"
    p.write_text(json.dumps(art), encoding="utf-8")
    return p


def test_the_print_flag_is_the_documented_no_write_report(tmp_path, monkeypatch, capsys):
    """`--print` is what the skill tells the nightly reporter to run, so it is
    graded as a command, not as a function: argv in, exit code and stdout out."""
    art, _, corpus_path = _run(tmp_path)
    art_file = _artifact_file(tmp_path, art)
    before = art_file.read_bytes()
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path / "artifacts"))

    argv = ["eval/label_agreement_ceiling.py", "--print",
            "--artifact", str(art_file), "--corpus", str(corpus_path),
            "--allow-stub-ceiling"]
    monkeypatch.setattr(sys, "argv", argv)
    assert lac.main() == 0, "the documented nightly command must exit 0, not 2"

    out = capsys.readouterr().out
    a, c = art["agreement"], art["ceiling"]
    assert str(a["entity_label_agreement"]) in out
    assert str(a["doc_label_agreement"]) in out
    assert c["kind"] in out, "a normalized score must say which ceiling it used"
    assert "stub-model-x" in out, "which model produced the ceiling travels with it"
    assert not (tmp_path / "artifacts").exists(), (
        "--print labels nothing and writes nothing: no engine, no new artifact")
    assert art_file.read_bytes() == before, "--print must not rewrite the artifact"


def test_print_names_the_disagreement_set_when_agreement_is_below_0_80(tmp_path,
                                                                       monkeypatch,
                                                                       capsys):
    """Clause 5's other half, at the CLI a person actually runs: an advisory that
    says "this is label noise" without naming the labels is the excuse-shaped
    instrument the item warns about, so the names and the 0.80 verdict are one
    print, not two optional ones."""
    art, _, corpus_path = _run(tmp_path, pick=lambda req: [])
    assert art["agreement"]["entity_label_agreement"] < 0.80
    assert art["agreement"]["disagreements"]["entity"], "fixture must disagree"
    art_file = _artifact_file(tmp_path, art)
    monkeypatch.setattr(sys, "argv", [
        "eval/label_agreement_ceiling.py", "--print",
        "--artifact", str(art_file), "--corpus", str(corpus_path),
        "--allow-stub-ceiling"])
    assert lac.main() == 0
    out = capsys.readouterr().out
    assert "BELOW 0.80" in out
    for d in art["agreement"]["disagreements"]["entity"]:
        assert d["gold"] in out, f"disagreeing label {d['gold']!r} not named"


def test_print_without_an_artifact_says_so_instead_of_crashing(tmp_path, monkeypatch,
                                                               capsys):
    """The reporter's other common state: the instrument has never run. Exit 2 with
    a reason naming how to produce one — not a traceback, and not a silent 0 that
    a nightly log reads as "reported"."""
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path / "nothing-here"))
    empty = tmp_path / "queries.yaml"
    empty.write_text("queries: []\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval/label_agreement_ceiling.py", "--print",
                                      "--corpus", str(empty)])
    assert lac.main() == 2
    out = capsys.readouterr().out
    assert "NO CEILING" in out
    assert "--engine" in out, "the reason must name the command that makes one"


def test_the_stub_artifact_is_still_refused_by_default_under_print(tmp_path, monkeypatch,
                                                                   capsys):
    """The CLI's own half of the same rule the loader enforces for run_eval: a
    stub's agreement is a property of the stub. `--print` reaching for the newest
    artifact must not be a way to read a smoke test as a measurement."""
    art, _, corpus_path = _run(tmp_path)
    # Deliberately not captured by name: `--print` with no `--artifact` is supposed to
    # find this file by itself, so writing it into the same directory OUT_DIR_ENV
    # points at is the setup, and reading it back would be the assertion the test is
    # trying not to make.
    _artifact_file(tmp_path, art)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["eval/label_agreement_ceiling.py", "--print",
                                      "--corpus", str(corpus_path)])
    assert lac.main() == 2
    out = capsys.readouterr().out
    assert "stub" in out and "NO CEILING" in out


def test_the_real_corpus_denominators_are_the_ones_this_file_quotes():
    """Clause 2's numbers, checked against the corpus file itself rather than against
    a count this file's own prose remembers.

    The item's clause names 94 entity labels across 66 queries and 139 doc labels,
    and rejects the 43 an earlier draft of the item quoted. Quoting them again here
    would be exactly the remembered-number bug the docstring at
    `label_agreement_ceiling.py:151` warns about, so the assertion is an independent
    recount of the file — `sum(...)` over the parsed YAML, not a call into
    `label_counts`, which is the function under test.

    The bounds are lower bounds, not equalities, on purpose: the corpus grows
    (#1319 took it 20 queries to 81), and an equality here would block an unrelated
    future round that adds a query. What must never come back is a denominator from
    the 20-query era, so each is floored above that era and the remembered 43 is
    excluded by name.
    """
    # `yaml` imported here rather than at module level, and the file parsed here
    # rather than through `lac.load_corpus`: the point is a second reader of the
    # corpus, so a filter added inside `load_corpus` could not hide a label from
    # the count and still agree with it.
    import yaml
    raw = yaml.safe_load((ROOT / "eval" / "vault_recall_queries.yaml").read_text(
        encoding="utf-8"))
    queries = raw["queries"]
    ent_labels = sum(len(q.get("expect_entities") or []) for q in queries
                     if q.get("expect_entities"))
    doc_labels = sum(len(q.get("expect_docs") or []) for q in queries
                     if q.get("expect_docs"))
    ent_queries = sum(1 for q in queries if q.get("expect_entities"))
    doc_queries = sum(1 for q in queries if q.get("expect_docs"))

    counts = lac.label_counts(lac.load_corpus(lac.CORPUS_PATH))
    assert counts == {"queries": len(queries), "entity_queries": ent_queries,
                      "entity_labels": ent_labels, "doc_queries": doc_queries,
                      "doc_labels": doc_labels}, (
        "label_counts disagrees with a direct recount of the file, so one of the two "
        "agreement denominators is computed wrong")
    assert ent_labels >= 90 and doc_labels >= 130 and ent_queries >= 60, (
        f"a 20-query-era denominator is back ({counts}); the corpus is at 94/66/139 "
        f"as of 2026-09-24 and an agreement rate over a stale denominator is a rate "
        f"about nothing")
    assert ent_labels != 43, "the number the item's own superseded draft quoted"


# ── the CLI seam: the whole labelling command, no engine, no live store ──────

def _run_main(args: list[str], *, home: Path, kg_db: Path,
              corpus: Path, out: Path) -> "subprocess.CompletedProcess":
    """Run `main()` as the operator runs it, with every ambient path pinned.

    HOME is a temp dir holding a two-file `obsidian/` so the doc leg has a
    namespace that does not depend on what the round's home happens to symlink, and
    LLOYD_KG_DB points at a store provisioned right here — `entity_name_table()`
    goes through `app.kg_store.store()`, which refuses an absent database (#1236)
    and a gate worktree has none.
    """
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["LLOYD_KG_DB"] = str(kg_db)
    # Set explicitly rather than inherited: the script's own sys.path[0] is `eval/`,
    # so `import app.paths` only resolves with the repo root on the path. The gate
    # exports PYTHONPATH and a bare `pytest` happens to work, but a seam test that
    # passes because of the caller's environment is a seam test that fails in the
    # one environment that matters.
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [str(PY), str(ROOT / "eval" / "label_agreement_ceiling.py"),
         "--corpus", str(corpus), "--out-dir", str(out), *args],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT), env=env)


def test_the_labelling_command_runs_end_to_end_with_a_stub_and_no_engine(tmp_path):
    """`main` from argv to printed report, with the stub labeler.

    Unit tests on `label_corpus` cannot see this seam: `main` is where the parser,
    the namespace loaders, the artifact write and the printer meet, and each was
    previously exercised only in pieces. That is how `--split-half` survived review:
    `main` read `args.split_half`, an option the parser never registered, so every
    successful labelling run wrote its artifact and then died with AttributeError on
    the print line — the number on disk, the report never reaching whoever ran it.
    The stub is what makes the whole path runnable in a gate: `secondary_enabled:
    false` means the engine the item names is a stopped process, and a command only
    exercisable with a person's GPU switch is a command nobody can test.
    """
    from app.kg_store import KGStore
    home = tmp_path / "home"
    vault = home / "obsidian"
    for rel in VAULT_DOCS:
        f = vault / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    db = tmp_path / "kg.sqlite"
    store = KGStore(db)
    for name in ENTITIES:
        store.entities.register(name)
    store.close()
    _queries, corpus_path = _corpus(tmp_path)
    out = tmp_path / "out"

    proc = _run_main(["--stub", "--split-half", "25"], home=home, kg_db=db,
                     corpus=corpus_path, out=out)
    assert proc.returncode == 0, f"{proc.stdout[-1200:]}\n{proc.stderr[-2500:]}"
    files = sorted(out.glob(lac.ARTIFACT_GLOB))
    assert len(files) == 1, f"expected one artifact, got {files}"
    art = json.loads(files[0].read_text(encoding="utf-8"))
    assert art["labeler"]["kind"] == "stub", (
        "the artifact must stamp that a stub produced it — the loader refuses one "
        "by default precisely so this run can never be read as a measured ceiling")
    assert art["corpus"]["counts"] == {"queries": 4, "entity_queries": 4,
                                       "entity_labels": 5, "doc_queries": 4,
                                       "doc_labels": 4}, (
        "the CLI labelled a different set than the corpus on disk, so the "
        "agreement denominators in the artifact are not the corpus's own")
    assert art["agreement"]["entity_label_agreement"] is not None
    assert art["ceiling"]["kind"] == lac.CEILING_KIND
    # The report reached the operator, not just the file: an exit code of 0 with the
    # numbers only on disk is the shape that hid the missing --split-half flag.
    assert "gold-side ceiling" in proc.stdout, proc.stdout
    assert "label agreement" in proc.stdout, proc.stdout
    assert "split-half:" in proc.stdout, (
        "--split-half was asked for and not printed: either the flag is unregistered "
        "again or its result is dropped between the run and the page\n"
        f"{proc.stdout[-1200:]}")

    # And the same command with NO optional flag — the shape that actually died:
    # `main` read `args.split_half` unconditionally, so the default labelling run was
    # the broken one and passing the flag would only have changed the error text.
    out_default = tmp_path / "out-default"
    proc2 = _run_main(["--stub"], home=home, kg_db=db, corpus=corpus_path,
                      out=out_default)
    assert proc2.returncode == 0, f"{proc2.stdout[-800:]}\n{proc2.stderr[-2500:]}"
    assert "split-half:" not in proc2.stdout, (
        "the resample ran when it was not asked for: it is opt-in because most entity "
        "queries have one gold label, and a headline built by averaging zeros into a "
        "ceiling is the failure its own docstring names")


def test_the_labelling_command_refuses_to_choose_an_engine_for_you(tmp_path):
    """No default labeler is a decision, not an oversight, so it has to be a refusal.

    Which model produced a ceiling changes what the number means — an artifact from
    the primary measures one model against a hand-authored gold, one from djev
    measures a 26B diffusion model against it. Falling back to "whatever is up"
    would print a ceiling whose subject is unknown, which is the thing clause 1's
    labeler identity exists to prevent; so the CLI must exit non-zero and say what
    it needs rather than pick.
    """
    home = tmp_path / "home"
    (home / "obsidian").mkdir(parents=True)
    out = tmp_path / "out"
    proc = _run_main([], home=home, kg_db=tmp_path / "absent.sqlite",
                     corpus=ROOT / "eval" / "vault_recall_queries.yaml", out=out)
    assert proc.returncode == 2, f"{proc.stdout}\n{proc.stderr}"
    assert "LABELER_REQUIRED" in proc.stdout, proc.stdout + proc.stderr
    assert not list(out.glob(lac.ARTIFACT_GLOB)), (
        "a run with no labeler wrote an artifact: that file is what a later reader "
        "loads as the ceiling")


# ===========================================================================
# #1637 — the gold-label fingerprint on every baseline, and the guard on a
# benchmark whose labels moved under an unchanged id set.
#
#   Seam 5: `eval/run_eval.py` -> baseline JSON -> `scripts/eval_trend_stats.py`.
#
# Commit `9b028e9` re-pointed 22 gold entity names while every query id stayed
# put. The trend audit joins two nights on `records[].id`, so it saw an unchanged
# benchmark and printed a delta anyway; the only fingerprint that existed,
# `load_artifact(expect_labels_sha256=...)`, guards the CEILING divisor, and with
# no artifact on disk (#1655 owns the first labelling run) it can never fire. So
# the fingerprint has to be written by the run itself, from the query file, and
# read by the audit from the artifact — one value crossing one process boundary,
# which is what the three tests below drive rather than mock.
# ===========================================================================

import scripts.eval_trend_stats as ts  # noqa: E402

#: Two fingerprints in the writer's own shape: `hexdigest()[:16]`.
_FP_A = "a1b2c3d4e5f60718"
_FP_B = "0f1e2d3c4b5a6978"


def _gold_run_baseline(tmp_path, monkeypatch, queries, label):
    """Run `eval/run_eval.py`'s `main()` for real and hand back the baseline it wrote.

    In-process, with the two injected seams `tests/test_eval_scorer.py` uses — the
    recall and the store — and the label-agreement directory pointed at an empty
    directory, so every test here runs in the state the item says production is in:
    no ceiling artifact anywhere. What the run is NOT is a hand-built dict: the
    clause is about the bytes `main()` writes, so `main()` writes them.
    """
    import sys as _sys

    qf = tmp_path / f"{label}-queries.yaml"
    lines = ["queries:"]
    for q in queries:
        lines.append(f"  - id: {q['id']}")
        lines.append(f"    query: {json.dumps(q['query'])}")
        lines.append("    category: entity")
        if q.get("expect_entities"):
            lines.append(f"    expect_entities: {json.dumps(q['expect_entities'])}")
        if q.get("expect_docs"):
            lines.append(f"    expect_docs: {json.dumps(q['expect_docs'])}")
    qf.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gold = (queries[0].get("expect_entities") or ["Gold Entity"])[0]

    def fake_recall(params, **kw):
        return {"entities": [], "facts": [{"entity": gold, "text": "t",
                                           "source": "memory/x.md"}],
                "documents": [], "graph_expanded_facts": [],
                "graph_neighbors_used": [],
                "fact_read_coverage": {"attempted": 1, "read": 1},
                "graph_expansion": {}}

    def fake_store():
        class _S:
            def stats(self):
                return {"entities_total": 1, "edges_total": 1, "edges_active": 1,
                        "aliases": 0, "facts": 1}
            def resolve(self, text):
                return text
            def active_edges(self):
                return []
            def neighbours(self, *a, **k):
                return {}
        return _S()

    out_dir = tmp_path / f"{label}-baselines"
    monkeypatch.setattr(ev, "_recall_seeds", lambda q, k: [gold])
    monkeypatch.setattr(ev, "_semantic_seed_k", lambda: 1)
    monkeypatch.setattr(ev, "_vault_recall", fake_recall)
    monkeypatch.setattr(ev, "store", fake_store)
    monkeypatch.setattr(ev, "EVAL_BASELINES_DIR", out_dir)
    monkeypatch.setattr(ev, "LLOYD_CODE_ROOT", tmp_path)
    # The state production is in: nothing has ever labelled the corpus.
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path / "no-artifacts-here"))
    monkeypatch.setattr(_sys, "argv", [
        "run_eval.py", "--queries", str(qf), "--label", label, "--limit", "9",
        "--no-counterfactual", "--allow-empty-corpus"])
    assert ev.main() == 0, "the run itself refused; the fingerprint never got written"
    return next(iter(sorted(out_dir.glob(f"{label}-*.json"))))


def _labelled_queries():
    return [
        {"id": "fp-alpha", "query": "what re-points a gold label",
         "category": "entity", "expect_entities": ["Backlog Item #1637"],
         "expect_docs": ["backlog/1637-gold-label-fingerprint.md"]},
        {"id": "fp-beta", "query": "which engine labels the corpus",
         "category": "entity", "expect_entities": ["Nightly Retrieval Eval"],
         "expect_docs": ["knowledge/software/nightly-retrieval-eval.md"]},
    ]


def _baseline_doc(label, day, ids, entity_bits, labels_sha256=_FP_A, corpus=True):
    """One baseline file in the shape `eval/run_eval.py` writes, gold stamp optional.

    `labels_sha256` is passed through verbatim, and `None` means the key is absent
    — which is what a legacy night looks like, and a key present-with-null is a
    different artifact that must not be silently confused with it.
    """
    doc = {
        "label": label,
        "ran_at": f"2026-01-{day:02d}T13:00:00+00:00",
        "records": [
            {"id": q, "query": f"query {q}",
             "scoring": {"entity_hit": bool(e), "doc_hit": bool(e),
                         "ndcg10": 1.0 if e else 0.0, "rr_doc": 1.0 if e else 0.0}}
            for q, e in zip(ids, entity_bits)
        ],
        "summary": {"overall": {"n_queries": len(ids)}},
        "corpus": {"facts": 200000, "edges_active": 4000, "entities": 23600},
    }
    if labels_sha256 is not None:
        doc["labels_sha256"] = labels_sha256
    return doc


def _write_baselines(tmp_path, docs):
    d = tmp_path / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    for name, doc in docs:
        (d / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
    return d


def test_a_scored_run_records_its_gold_labels_without_any_ceiling_artifact(tmp_path,
                                                                           monkeypatch,
                                                                           capsys):
    """Clause 1, at the file boundary: the bytes `run_eval` writes.

    Three things have to be true of the written artifact at once, and a test that
    only reads one of them would pass on a wrong implementation:

      1. `<GOLD_LABELS_KEY>` equals `lac.labels_sha256` over THIS run's scored
         queries — the clause names that function as the value, so it is the oracle;
      2. the same artifact still reports `ceiling.kind: null` with the
         no-artifact reason beside it, which is what makes the stamp independent of
         the ceiling rather than a copy of a field the ceiling block already has;
      3. re-pointing one gold name, ids untouched, moves the value — the `9b028e9`
         property, and the only reason the field exists.

    A corpus carrying no gold at all yields no fingerprint — the hash of an empty
    label set is the hash of nothing, equal for every such corpus, and a stamp that
    cannot disagree with anything is not an instrument. That boundary is pinned on
    the writer directly, for the reason in the last block below.
    """
    queries = _labelled_queries()
    path = _gold_run_baseline(tmp_path, monkeypatch, queries, "goldfp")
    blob = json.loads(path.read_text(encoding="utf-8"))
    capsys.readouterr()

    assert ev.GOLD_LABELS_KEY == "labels_sha256" == ts.GOLD_LABELS_KEY, (
        "writer and reader fell out over the key spelling; the audit would then "
        "read a legacy absence and stay silent forever")
    assert blob[ev.GOLD_LABELS_KEY] == lac.labels_sha256(
        sorted(queries, key=lambda q: q["id"])), (
        "the recorded fingerprint is not the one label_agreement_ceiling computes "
        "over the scored queries' labels")
    assert blob[ev.GOLD_LABELS_KEY] != lac.labels_sha256([]), (
        "a stamp over no labels reads as a measurement of the gold")

    overall = blob["summary"]["overall"]
    assert overall["ceiling"]["kind"] is None
    assert "no label-agreement artifact" in overall["ceiling"]["reason"], (
        "the fingerprint arrived only because a ceiling did, which is the coupling "
        "clause 1 exists to break")

    # The reader's half of the seam: the audit sees the same value off disk.
    assert ts.load_night(path).labels_sha256 == blob[ev.GOLD_LABELS_KEY]

    moved = [dict(queries[0], expect_entities=["Backlog Item #999"])] + queries[1:]
    moved_path = _gold_run_baseline(tmp_path, monkeypatch, moved, "goldfpmoved")
    moved_blob = json.loads(moved_path.read_text(encoding="utf-8"))
    assert moved_blob[ev.GOLD_LABELS_KEY] != blob[ev.GOLD_LABELS_KEY], (
        "re-pointing a gold entity name under an unchanged id left the stamp "
        "alone: this is exactly the `9b028e9` edit the field must make visible")

    # The no-gold boundary, pinned at the writer rather than through a whole run:
    # `print_report` has a pre-existing crash on a corpus with no gold at all
    # (`by_category`'s `entity_recall_avg` is None and the category table formats it
    # with `:.2f`, run_eval.py:1628-1633), which is a different defect on a path this
    # item does not touch — recorded on #1637 rather than fixed here.
    bare = [{"id": "fp-alpha", "query": "what re-points a gold label",
             "category": "entity"}]
    assert ev.gold_label_fingerprint(bare, ["fp-alpha"]) is None, (
        "an unlabelled corpus recorded the hash of an empty label set as though it "
        "were a gold fingerprint")
    assert ev.gold_label_fingerprint(queries, []) is None, (
        "a run that scored none of the labelled queries still stamped the labels it "
        "did not measure")


def test_the_audit_refuses_a_pair_whose_ids_match_but_whose_gold_moved(tmp_path,
                                                                      capsys):
    """Clause 2: ids equal + fingerprints differ -> refusal naming both, no delta.

    The scores are identical here on purpose: the pair a reader must never be given
    is one that looks quiet. Both nights report the same hit pattern, so the McNemar
    discordance is zero, p = 1.000, and the audit could print "no change" — while
    every one of those hits was judged against a different gold answer. "Nothing
    moved" would be the most misleading sentence in the file.
    """
    ids = tuple(f"q{i}" for i in range(1, 9))
    bits = [1, 1, 0, 1, 0, 1, 1, 0]
    d = _write_baselines(tmp_path, [
        ("nightly-20260101", _baseline_doc("nightly-20260101", 1, ids, bits, _FP_A)),
        ("nightly-20260102", _baseline_doc("nightly-20260102", 2, ids, bits, _FP_B)),
    ])
    prev, cur = ts.load_window(d)
    assert prev.ids == cur.ids, "fixture: the id set is the half that did NOT move"

    t = ts.audit_transition(prev, cur, reps=200)
    assert t.joinable, "the ids do join; this is not the unjoinable case"
    assert t.incomparable, "a moved gold label set must not come back comparable"
    assert _FP_A in t.incomparable and _FP_B in t.incomparable, t.incomparable
    assert t.legs == {}, "no test may run over two different benchmarks"
    assert t.rejected == [] and t.withheld and not t.admissible

    assert ts.main(["--baselines", str(d), "--no-claims", "--reps", "200"]) == 0
    out = capsys.readouterr().out
    assert "INCOMPARABLE" in out and _FP_A in out and _FP_B in out, out
    assert "no delta printed" in out, out
    assert "verdict: WITHHELD" in out, out
    # The transition's own block only: past the first `=` rule sits the summary,
    # whose column labels mention McNemar and resampling for the window as a whole.
    body = out.split("TRANSITION nightly-20260101")[1].split("=" * 78)[0]
    assert not re.search(r"(entity_hit|doc_hit)\s+delta", body), (
        "the refused pair still printed per-leg deltas")
    assert "resampling approximation" not in body, "a bootstrap ran over the refused pair"
    assert "McNemar" not in body, "a McNemar test ran over the refused pair"
    assert "1 incomparable" in out, out


def test_the_audit_is_unchanged_when_the_fingerprints_agree_or_are_absent(tmp_path,
                                                                         capsys):
    """Clause 3: the guard fires for one condition only, so equality and silence
    must print exactly what they printed before it existed.

    Pinned by rendering, not by a flag: the whole transition block for a pair
    carrying equal stamps, and for each of the three absent shapes (both absent,
    earlier only, later only), is compared byte for byte against the same pair with
    no stamp at all. A guard that annotated a legacy pair, or that quietly stopped
    a matching pair, fails here even if it still reported a correct delta — and a
    key present with a null value is checked separately, because "recorded as null"
    is a fourth state the writer never produces and the reader must not invent.
    """
    ids = tuple(f"q{i}" for i in range(1, 9))
    bits = [1, 1, 0, 1, 0, 1, 1, 0]
    moved = [1, 1, 1, 1, 0, 1, 1, 0]

    slots = iter(str(i) for i in range(20))

    def render(docs):
        d = _write_baselines(tmp_path / f"pair-{next(slots)}", docs)
        capsys.readouterr()
        assert ts.main(["--baselines", str(d), "--no-claims", "--reps", "200"]) == 0
        return capsys.readouterr().out

    def block(out):
        body = out.split("TRANSITION nightly-20260101")[1]
        return body.split("TRANSITION")[0].split("=" * 78)[0]

    plain = render([("nightly-20260101",
                     _baseline_doc("nightly-20260101", 1, ids, bits, None)),
                    ("nightly-20260102",
                     _baseline_doc("nightly-20260102", 2, ids, moved, None))])
    assert "INCOMPARABLE" not in plain and "delta " in plain, plain

    assert block(render([
        ("nightly-20260101", _baseline_doc("nightly-20260101", 1, ids, bits, _FP_A)),
        ("nightly-20260102", _baseline_doc("nightly-20260102", 2, ids, moved, _FP_A)),
    ])) == block(plain), "equal fingerprints changed the report"

    # Three legacy shapes, named by the one that is missing the key: one-sided on
    # each side, and neither side carrying it.
    for where, fp_prev, fp_cur in (
            ("absent on the later night", _FP_A, None),
            ("absent on the earlier night", None, _FP_B),
            ("absent on either side", None, None)):
        got = render([
            ("nightly-20260101", _baseline_doc(
                "nightly-20260101", 1, ids, bits, fp_prev)),
            ("nightly-20260102", _baseline_doc(
                "nightly-20260102", 2, ids, moved, fp_cur)),
        ])
        assert block(got) == block(plain), (
            f"a pair whose fingerprint is {where} is no longer the report it used to be")

    nullish = _baseline_doc("nightly-20260102", 2, ids, moved, None)
    nullish["labels_sha256"] = None
    assert ts.load_night(_write_baselines(
        tmp_path / "nullish", [("nightly-20260102", nullish)]) / "nightly-20260102.json"
    ).labels_sha256 is None, "a recorded null must read as no fingerprint, not as ''"
