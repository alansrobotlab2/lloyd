"""Every quality aggregate is reported beside a gold-side ceiling and a ratio (#654).

Clause 4 of item #654. `entity_hit_rate` was a bare number: it read 0.50 on 2026-09-09
and 0.37 on 2026-09-23 over a corpus whose labels had been re-pointed in between, and
nothing in the artifact could say whether the labels or the retriever had moved. This
adds the ceiling that turns a rate into a position on a scale, and pins the four ways
such a field goes wrong:

  * it is DROPPED when unmeasured, so the next reader cannot tell "no ceiling exists
    for this metric" from "the instrument has never run" — to a tool reading this
    file, an absent key and a measured zero are the same reading, which is the failure
    the field exists to close;
  * it shadows or renames the seed-side ceiling (`anchorless_query_count`) that
    automod's armed regression already reads (#1164);
  * it changes the raw numbers it is supposed to divide, which would make every prior
    baseline incomparable with tonight's;
  * it silently inherits an artifact written by a plumbing stub, or one whose gold
    labels no longer exist.

The in-process tests build a small synthetic corpus so they run with no engine and no
live store — a gate on a worktree has neither, and a clause pinned only by tests that
skip there is not pinned. The subprocess tests are the serialisation seam (the nightly
is `run_eval.py → JSON file → automod_regression`, and a unit test on `summarize` alone
cannot show the field survives `json.dump` and lands under `summary.overall`); they need
a live engine and follow the same `LLOYD_CI` gate as `tests/test_eval_ci_reporting.py`.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_VENV = ROOT / ".venvs" / "lloyd" / "bin" / "python"
PY = _VENV if _VENV.exists() else Path(sys.executable)
SCRIPT = ROOT / "eval" / "run_eval.py"
SKILL = Path.home() / "obsidian" / "skills" / "retrieval-eval" / "SKILL.md"

sys.path.insert(0, str(ROOT))
# Imported at module level on purpose. `tests/conftest.py` points LLOYD_DATA at a
# scratch root before anything imports `app.paths`, so `BASELINES` here is the same
# scratch directory the child process writes into — the arrangement
# tests/test_eval_ci_reporting.py runs under.
from app.paths import EVAL_BASELINES_DIR as BASELINES  # noqa: E402
import eval.run_eval as ev  # noqa: E402
import eval.label_agreement_ceiling as lac  # noqa: E402

CI = os.environ.get("LLOYD_CI", "") == "1"
#: Said out loud at every skip because nothing on this box sets that variable: the
#: promotion pipeline's `tests` rung runs pytest with no LLOYD_CI, and
#: `grep -rn LLOYD_CI scripts/ workers/ agent-services/` is empty (measured
#: 2026-09-24), so these nodes skip in EVERY run here, not only in a gate worktree.
#: The gate report's "engine-absent condition (LLOYD_CI=1)" wording describes the
#: intent, not a job that exists. The written-JSON contract those nodes would have
#: pinned is therefore pinned hermetically by
#: `test_a_measured_ceiling_survives_the_json_round_trip_into_summary_overall`, which
#: needs no engine; what remains engine-only is the live retrieval QUALITY, and the
#: skip below names which node it stands in for.
CI_SKIP = ("retrieval eval needs a live engine and LLOYD_CI=1; nothing sets "
           "LLOYD_CI on this box (scripts/, workers/, agent-services/ all 0 hits), "
           "so this is a permanent skip here, not a transient one — the written-"
           "baseline contract is covered hermetically by "
           "test_a_measured_ceiling_survives_the_json_round_trip_into_summary_overall")

#: the seven metrics every run reports; each gets a normalized value and a kind
METRICS = ["entity_hit_rate", "doc_hit_rate", "entity_recall_avg",
           "doc_recall_avg", "mrr_doc", "ndcg10", "fact_entity_recall_avg"]

RAW_KEYS = ["n_queries", "entity_hit_rate", "doc_hit_rate", "entity_recall_avg",
            "doc_recall_avg", "mrr_doc", "ndcg10", "fact_entity_recall_avg",
            "anchorless_query_count", "anchorless_query_ids", "errors",
            "counterfactual_moved_rate", "counterfactual_pinned_rate"]


# ── the semantics, with no engine and no live store ──────────────────────────

def test_ceiling_context_names_the_absent_state():
    """`ceiling: null` arrives with the reason naming which state it is — absent,
    stub, stale labels, or an artifact that does not recompute. Every key is present
    as null: an absent key and a measured zero are the same reading to the next tool,
    which is the failure this field exists to close."""
    fields = ev._ceiling_absent_fields("no label-agreement artifact under /tmp/x")
    assert fields["ceiling"]["kind"] is None
    assert "artifact" in fields["ceiling"]["reason"]
    assert fields["label_agreement"]["entity_label_agreement"] is None
    for metric in METRICS:
        assert f"{metric}_normalized" in fields
        assert fields[f"{metric}_normalized"] is None
        assert fields[f"{metric}_ceiling_kind"] is None


def test_ceiling_context_without_an_artifact_on_disk():
    """The state the landing is in right now: the instrument exists, no labeler has
    been authorized, so no artifact is on disk. Null plus a reason, never a 0.0 and
    never a missing key."""
    fields = ev.ceiling_context(lac.load_corpus()[:2])
    assert fields["ceiling"]["kind"] is None
    assert fields["ceiling"]["reason"], "a null with no reason is a dead field"
    assert fields["label_agreement"]["labeler"] is None


def test_summarize_keeps_the_keys_when_no_ceiling_was_computed():
    """`summarize` is called by the promotion gate's baseline arm, the CI runner and
    the profile probes, none of which compute a ceiling. The keys must exist there
    too, or 'absent' is once again a missing key."""
    rec = {"id": "x", "category": "c", "latency_ms": 10, "error": None,
           "scoring": {"entity_hit": True, "doc_hit": False, "entity_recall": 1.0,
                       "doc_recall": 0.0, "rr_doc": 0.0, "ndcg10": 0.0,
                       "fact_entity_recall": None, "entities_matched": ["a"],
                       "counterfactual_moved_rate": None,
                       "counterfactual_pinned_rate": None}}
    o = ev.summarize([rec])["overall"]
    for metric in METRICS:
        assert f"{metric}_normalized" in o and o[f"{metric}_normalized"] is None
        assert f"{metric}_ceiling_kind" in o
    assert o["label_agreement"] is None and o["ceiling"] is None
    # The seed-side ceiling is untouched by any of this: it predates #654 and
    # automod's armed regression reads it (#1164).
    assert "anchorless_query_count" in o


def test_zero_ceiling_normalizes_to_null_with_a_note():
    """A ceiling of 0.0 means no answer at all could satisfy these gold labels. The
    ratio is 0/0, and a printed 0.0 would claim a scored position where there is a
    division by nothing."""
    fields = {"ceiling": {"kind": lac.CEILING_KIND, **{m: 0.0 for m in METRICS[:-1]},
                          "fact_entity_recall_avg": None}}
    for metric in METRICS:
        fields[f"{metric}_normalized"] = None
        fields[f"{metric}_ceiling_kind"] = lac.CEILING_KIND
    scores = {"entity_hit_rate": 0.5, "doc_hit_rate": 0.4, "entity_recall_avg": 0.3,
              "doc_recall_avg": 0.2, "mrr_doc": 0.1, "ndcg10": 0.1,
              "fact_entity_recall_avg": None}
    ev._normalize_against_ceiling(fields, scores)
    assert fields["entity_hit_rate_normalized"] is None
    assert "undefined" in fields["ceiling_notes"]["entity_hit_rate"]


def test_normalization_divides_and_names_the_kind():
    """`<metric>_normalized = score / ceiling`, and the number names which kind of
    ceiling it used. A metric with no ceiling must not acquire a ratio — silently
    borrowing another leg's ceiling is what makes a normalized number lie."""
    fields = {"ceiling": {"kind": lac.CEILING_KIND, "entity_hit_rate": 0.5,
                          "doc_hit_rate": None, "entity_recall_avg": 0.25,
                          "doc_recall_avg": None, "mrr_doc": None, "ndcg10": None,
                          "fact_entity_recall_avg": None}}
    for metric in METRICS:
        fields[f"{metric}_normalized"] = None
        fields[f"{metric}_ceiling_kind"] = None
    ev._normalize_against_ceiling(
        fields, {"entity_hit_rate": 0.35, "doc_hit_rate": 0.8,
                 "entity_recall_avg": 0.3, "doc_recall_avg": 0.5, "mrr_doc": 0.4,
                 "ndcg10": 0.5, "fact_entity_recall_avg": 0.2})
    assert fields["entity_hit_rate_normalized"] == pytest.approx(0.7)
    assert fields["entity_hit_rate_ceiling_kind"] == lac.CEILING_KIND
    assert fields["entity_recall_avg_normalized"] == pytest.approx(1.2)
    assert fields["doc_hit_rate_normalized"] is None
    assert "ceiling_notes" not in fields


# ── #2014: a ratio is never emitted from two different query sets ────────────

def _ceilinged(ceil_n: dict | None) -> dict:
    fields = {"ceiling": {"kind": lac.CEILING_KIND, "entity_hit_rate": 0.5094,
                          "doc_hit_rate": 0.6122, "entity_recall_avg": 0.0,
                          "doc_recall_avg": None, "mrr_doc": None, "ndcg10": None,
                          "fact_entity_recall_avg": None}}
    if ceil_n is not None:
        fields["ceiling"]["n"] = ceil_n
    for metric in METRICS:
        fields[f"{metric}_normalized"] = None
        fields[f"{metric}_ceiling_kind"] = None
    return fields


def _scored(entity_n, doc_n) -> dict:
    return {"entity_hit_rate": 0.652, "doc_hit_rate": 0.716, "entity_recall_avg": 0.3,
            "doc_recall_avg": 0.5, "mrr_doc": 0.4, "ndcg10": 0.5,
            "fact_entity_recall_avg": None,
            "ci95": {"entity_hit_rate": {"n": entity_n}, "doc_hit_rate": {"n": doc_n},
                     "entity_recall_avg": {"n": entity_n}}}


def test_a_ratio_is_withheld_when_the_two_halves_cover_different_query_sets():
    """The 2026-10-01 nightly, in miniature: entity_hit_rate 0.652 over 66 queries
    against a ceiling of 0.5094 over 53 printed 1.2799 — a population gap read as
    retrieval above its ceiling. The ratio is null and the note names both n's."""
    fields = _ceilinged({"entity_hit_rate": 53, "doc_hit_rate": 49, "entity_recall_avg": 38})
    ev._normalize_against_ceiling(fields, _scored(66, 74))
    for metric, (sn, cn) in {"entity_hit_rate": (66, 53), "doc_hit_rate": (74, 49)}.items():
        assert fields[f"{metric}_normalized"] is None, metric
        note = fields["ceiling_notes"][metric]
        assert note.startswith(ev.POPULATION_MISMATCH), note
        assert f"n={sn}" in note and f"n={cn}" in note, note
        assert fields[f"{metric}_ceiling_kind"] == lac.CEILING_KIND
    # A side that cannot say how many queries it covered is a mismatch, not a match.
    half = _ceilinged({"entity_hit_rate": 53})
    scores = _scored(53, 74)
    del scores["ci95"]
    ev._normalize_against_ceiling(half, scores)
    assert half["entity_hit_rate_normalized"] is None
    assert "n=None" in half["ceiling_notes"]["entity_hit_rate"]


def test_a_ratio_is_still_emitted_when_both_halves_cover_the_same_query_set():
    """The guard is a population guard, not a retirement of the field: equal n's
    divide, exactly as before."""
    fields = _ceilinged({"entity_hit_rate": 66, "doc_hit_rate": 74, "entity_recall_avg": 66})
    ev._normalize_against_ceiling(fields, _scored(66, 74))
    assert fields["entity_hit_rate_normalized"] == round(0.652 / 0.5094, 4)
    assert fields["doc_hit_rate_normalized"] == round(0.716 / 0.6122, 4)
    # The zero-ceiling leg answers first and keeps its own note, n's equal or not.
    assert fields["entity_recall_avg_normalized"] is None
    assert "undefined" in fields["ceiling_notes"]["entity_recall_avg"]
    assert set(fields["ceiling_notes"]) == {"entity_recall_avg"}
    # No ceiling for a metric: still null, still no ratio borrowed, no note invented.
    assert fields["doc_recall_avg_normalized"] is None


def test_the_page_prints_null_for_a_ratio_over_two_populations(capsys):
    """The printed page is what the nightly reporter copies: a mismatched metric
    shows `score/ceiling=null` with both n's on its own line, never a number."""
    fields = ev._ceiling_absent_fields("unused")
    fields["ceiling"] = {"kind": lac.CEILING_KIND, "entity_hit_rate": 0.8,
                         "doc_hit_rate": 0.9, "entity_recall_avg": 0.7,
                         "doc_recall_avg": 0.8, "mrr_doc": 0.6, "ndcg10": 0.7,
                         "fact_entity_recall_avg": None,
                         "n": {"entity_hit_rate": 7, "doc_hit_rate": 10,
                               "entity_recall_avg": 7, "doc_recall_avg": 10,
                               "mrr_doc": 10, "ndcg10": 10}}
    records, summary = _printed_summary(fields)
    assert summary["overall"]["ci95"]["entity_hit_rate"]["n"] == 10
    ev.print_table(records, summary)
    lines = _page_lines(capsys.readouterr().out)
    assert "score/ceiling=null (ceiling=0.8 kind=gold_label_surrogate)" in lines["entity_hit"]
    assert "n=10" in lines["entity_hit"] and "n=7" in lines["entity_hit"], lines["entity_hit"]
    assert "score/ceiling=0." not in lines["entity_hit"], lines["entity_hit"]
    # Control on the same page: doc_hit's halves agree (10 and 10), so it divides.
    assert "score/ceiling=1.1111 (ceiling=0.9" in lines["doc_hit"], lines["doc_hit"]


# ── the printed page: what the nightly reporter actually reads ───────────────

PRINTED = (("entity_hit", "entity_hit_rate"), ("doc_hit", "doc_hit_rate"),
           ("fact_entity_recall", "fact_entity_recall_avg"))


def _printed_records(n: int = 10, entity_hits: int = 4) -> list[dict]:
    """Records whose `entity_hit_rate` is exactly `entity_hits / n`."""
    return [{"id": f"p{i}", "category": "single", "latency_ms": 100.0, "error": None,
             "scoring": {"entity_hit": i < entity_hits, "doc_hit": True,
                         "entity_recall": 0.5, "doc_recall": 0.5, "rr_doc": 0.5,
                         "ndcg10": 0.5, "fact_entity_recall": 0.5,
                         "first_doc_rank": 1}} for i in range(n)]


def _printed_summary(overall_extra: dict, *, n: int = 10,
                     entity_hits: int = 4) -> tuple[list[dict], dict]:
    records = _printed_records(n, entity_hits)
    summary = ev.summarize(records)
    summary["overall"].update(overall_extra)
    ev._normalize_against_ceiling(overall_extra, summary["overall"])
    summary["overall"].update(overall_extra)
    return records, summary


def _page_lines(text: str) -> dict:
    return {label: next(ln for ln in text.splitlines()
                        if ln.strip().startswith(label + " "))
            for label, _ in PRINTED}


def test_the_reported_line_shows_the_ratio_not_the_divisor(capsys):
    """`score/ceiling=` must be followed by the RATIO, with the divisor behind it.

    The printed page is the seam: the nightly reporter reads this text, not the
    JSON, and an earlier draft printed the ceiling in the ratio's slot
    (`score/ceiling=0.8=0.5`) — two numbers one `=` apart, one of which is the
    position a reader is here to copy. The three candidate numbers are deliberately
    all different here (score 0.4, ceiling 0.8, ratio 0.5) so a swap is caught
    rather than absorbed. Clause 5's "prints score / ceiling beside
    `entity_hit_rate` with the ceiling kind named" is this line.
    """
    fields = ev._ceiling_absent_fields("unused")
    fields["ceiling"] = {"kind": lac.CEILING_KIND, "entity_hit_rate": 0.8,
                         "doc_hit_rate": 0.9, "entity_recall_avg": 0.7,
                         "doc_recall_avg": 0.8, "mrr_doc": 0.6, "ndcg10": 0.7,
                         "fact_entity_recall_avg": None}
    records, summary = _printed_summary(fields)
    ev.print_table(records, summary)
    lines = _page_lines(capsys.readouterr().out)
    assert "score/ceiling=0.5 (ceiling=0.8 kind=gold_label_surrogate)" \
        in lines["entity_hit"], lines["entity_hit"]
    assert "score/ceiling=0.8" not in lines["entity_hit"], (
        "the divisor printed in the ratio's slot: a reader copying the number "
        f"after score/ceiling= off this page copies {0.8}, not the position\n"
        f"{lines['entity_hit']}")
    # The leg with no gold-side ceiling says null rather than borrowing a number.
    assert "score/ceiling=null" in lines["fact_entity_recall"], lines["fact_entity_recall"]


def test_the_page_names_the_labeler_and_the_disagreement_set(capsys):
    """A sub-0.80 agreement on this page comes with the labels that caused it.

    Clause 5's second half, on the run_eval page (the labeler's own `--print` is
    pinned in tests/test_eval_label_agreement.py): a low ceiling printed alone is
    an excuse, and the same number printed with its disagreement set is a work
    list. The labeler identity rides along because a ceiling from an unnamed model
    is a number about nobody.
    """
    fields = ev._ceiling_absent_fields("unused")
    fields["ceiling"] = {"kind": lac.CEILING_KIND, "entity_hit_rate": 0.8,
                         "doc_hit_rate": 0.9, "entity_recall_avg": 0.7,
                         "doc_recall_avg": 0.8, "mrr_doc": 0.6, "ndcg10": 0.7,
                         "fact_entity_recall_avg": None}
    fields["label_agreement"] = {
        "entity_label_agreement": 0.72, "doc_label_agreement": 0.95,
        "labeler": {"model": "stub-model-x", "endpoint": "in-process"},
        "entity_disagreements": ["backlog-363 gold=Backlog Item #363",
                                 "qwen38-local-serving gold=Qwen3.8-27B"],
        "doc_disagreements": []}
    records, summary = _printed_summary(fields)
    ev.print_table(records, summary)
    out = capsys.readouterr().out
    assert "stub-model-x@in-process" in out, out
    assert "disagreement set" in out, out
    assert "backlog-363 gold=Backlog Item #363" in out, out
    assert "qwen38-local-serving gold=Qwen3.8-27B" in out, out
    assert "too ambiguous to call it a retrieval defect" in out, out


def test_the_page_says_why_there_is_no_ceiling(capsys):
    """The null state is reported, not hidden: an absent ceiling and an
    unmeasured ceiling are different facts, and neither is a bare blank."""
    fields = ev._ceiling_absent_fields("no label-agreement artifact on disk")
    records, summary = _printed_summary(fields)
    ev.print_table(records, summary)
    out = capsys.readouterr().out
    assert "gold_ceiling" in out and "no label-agreement artifact on disk" in out, out
    assert "score/ceiling=null (ceiling=null kind=null)" in out, out


# ── refusal states, on a synthetic corpus so no live store is needed ─────────

def _synth_corpus() -> list[dict]:
    """Five queries with hand-written gold, in the same shape the real corpus loads
    into. Synthetic so these tests run with no engine and no live store — a gate on a
    worktree has neither, and a clause pinned only by tests that skip there is not
    pinned. Ids and labels are invented, the SHAPE is the real one."""
    return [
        {"id": "alpha", "query": "which service hosts the wake word audio room",
         "category": "entity", "expect_entities": ["LiveKit"],
         "expect_docs": ["knowledge/software/livekit-node.md"]},
        {"id": "beta", "query": "what does the guardian watch after a landing",
         "category": "safety", "expect_entities": ["Guardian"],
         "expect_docs": ["knowledge/software/guardian.md"]},
        {"id": "gamma", "query": "where is the retrieval eval corpus described",
         "category": "retrieval", "expect_entities": ["Mission Control"],
         "expect_docs": ["knowledge/software/mc-dashboard.md"]},
        {"id": "delta", "query": "which model runs structured decisions on gpu two",
         "category": "eval", "expect_entities": ["djev"],
         "expect_docs": ["knowledge/software/djev-head.md"]},
        {"id": "epsilon", "query": "what consolidates memory overnight",
         "category": "memory", "expect_entities": ["Dream Consolidation"],
         "expect_docs": ["skills/dream-consolidation/SKILL.md"]},
    ]


def _synthetic_artifact(**mutate):
    """A real artifact over the synthetic corpus, built through the real labelling
    pass with an engine-shaped identity so the loader admits it."""
    corpus = _synth_corpus()
    names = [e["expect_entities"][0] for e in corpus] + [
        "Inner Voice", "Knowledge Graph", "Backlog Item #363", "vLLM", "Whisper"]
    paths = [d["expect_docs"][0] for d in corpus] + [
        "memory/2026-09-24.md", "knowledge/software/unrelated-a.md",
        "knowledge/software/unrelated-b.md", "projects/lloyd/x.md",
        "projects/lloyd/y.md"]
    art = lac.label_corpus(
        labeler=lac.stub_labeler(),
        labeler_identity={"kind": "engine", "model": "fixture-ceiling-model",
                          "endpoint": "fixture://local"},
        queries=corpus, entity_names=names, vault_paths=paths,
        entity_cap=len(names), doc_cap=len(paths))
    for key, value in mutate.items():
        if value is _DROP:
            art.pop(key, None)
        elif key == "entity_labels" and value == "moved":
            art["corpus"]["labels_sha256"] = "deadbeefdeadbeef"
        elif key == "kind" and value == "stub":
            art["labeler"]["kind"] = "stub"
        elif key == "recompute" and value == "broken":
            # Hand-edit the STORED ceiling so it stops following from the rows. The
            # loader re-derives the ceiling from the per-query surrogates and refuses
            # a stored value it cannot reproduce — this is the DIVISOR, so a number
            # nothing in the file supports must never divide tonight's score.
            art["ceiling"]["values"]["entity_hit_rate"] = 0.999
    return art


class _Drop:
    pass


_DROP = _Drop()


def _write(art: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "label-agreement-20260924-000000.json").write_text(
        json.dumps(art), encoding="utf-8")


# ── #1655: an ENGINE-labelled artifact all the way to a divided score ─────────

#: The identity `resolve_engine("djev")` yields after #1655's fix — the pair that
#: `test_resolve_engine_learns_the_served_djev_name_over_a_real_socket` (same dir,
#: `test_eval_label_agreement.py`) pins — recorded here rather than invented, because
#: the artifact's provenance is
#: what makes its number mean anything: `run_eval` writes `labeler.model` into every
#: baseline, and the item's own ruling note is that "which model produced the ceiling
#: changes what the number means".
DJEV_IDENTITY = {"kind": "engine", "model": "djev",
                 "endpoint": "http://127.0.0.1:8010/v1/chat/completions"}

#: Three queries, one gold name each, and a two-name pool so every gold is always
#: OFFERED: an unofferable gold is excluded from the ceiling rather than counted
#: against it, which would let the instrument's own cap move the number this file
#: exists to keep honest.
_CEIL_CORPUS = [
    {"id": "q-alive", "query": "what did the nightly reflection knowledge write decide",
     "expect_entities": ["Research"], "expect_docs": []},
    {"id": "q-disagree", "query": "which godnode index lists the research notes",
     "expect_entities": ["Research"], "expect_docs": []},
    {"id": "q-miss", "query": "where is the research note about the reflection write",
     "expect_entities": ["Research"], "expect_docs": []},
]


def _engine_artifact(directory: Path, *, disagree_on=()):
    """Run the real labelling pass over `_CEIL_CORPUS` with a second rater that agrees
    everywhere except the ids in `disagree_on`.

    Nothing here is hand-written JSON: the artifact comes out of `lac.label_corpus`,
    so its per-query rows, its `labels_sha256` and its stored ceiling are all things
    `load_artifact` can recompute. An artifact assembled by hand would clear the loader
    by construction and prove nothing about the file a real djev pass writes.

    `corpus_path` is a file this helper wrote into `directory`, not the shipped 86-query
    yaml: the artifact records `corpus.path` and its sha as its provenance, and a file
    that named the shipped corpus while carrying three rows would be a fixture lying
    about where it came from.
    """
    corpus_file = directory / "three-query-corpus.yaml"
    if not corpus_file.exists():
        corpus_file.write_text(
            "".join(f"- id: {q['id']}\n  query: {q['query']}\n"
                    f"  expect_entities: [{q['expect_entities'][0]}]\n"
                    for q in _CEIL_CORPUS), encoding="utf-8")

    def labeler(request: dict) -> dict:
        want = "Godnodes" if request["id"] in set(disagree_on) else "Research"
        cands = request["entity_candidates"]
        idx = cands.index(want) if want in cands else 0
        # 1-based, as an OpenAI-style reply is; no docs, so the doc leg stays
        # unmeasured rather than measured against an empty pool.
        return {"text": json.dumps({"entities": [idx + 1], "docs": []})}
    return lac.label_corpus(
        labeler=labeler, labeler_identity=dict(DJEV_IDENTITY),
        queries=[dict(q) for q in _CEIL_CORPUS],
        entity_names=["Research", "Godnodes"], vault_paths=[],
        entity_cap=2, doc_cap=0, corpus_path=corpus_file)


def test_an_engine_artifact_clears_all_four_refusals_and_divides_the_score(
        tmp_path, monkeypatch):
    """Clause 4: the file a real djev pass would write, admitted by the loader, and
    the score it produces divided by the ceiling it measured.

    Why this node exists at all: `--print` has exited 2 since the module landed, so
    every guard in `load_artifact` has only ever been exercised against a
    hand-assembled stub artifact from a test. The state the item is about — a real
    engine's file on disk — is the one state no node had seen, and a loader that
    refuses everything looks identical to one that refuses only what it should.

    Three queries, one gold name each; the second rater disagrees on `q-disagree`, so
    measured `entity_label_agreement` is 2/3 = 0.6667 and the ceiling it implies for
    `entity_hit_rate` is the same 0.6667 (the surrogate of a query whose gold is not
    reproducible is floored at half, so 1 + 0.5 + 1 over 3). Retrieval that hits the
    gold on two of the three then scores 0.6667 raw and 0.9999 of the ceiling — not
    1.0, because the stored ceiling is rounded to four places (0.6667) while the score
    is 2/3 exact, and the residual is that rounding, not a shortfall. The reading is
    the veto's whole point: a miss on a query two labelers cannot agree about is not a
    retrieval defect.

    The control is the same corpus with a labeler that agrees everywhere: ceiling
    1.0, normalized 0.6667. Same retrieval, same queries, same scorer — only the
    labels differ, which is what shows the divisor came from the labels and not from
    the seed blocks or the corpus size.
    """
    art = _engine_artifact(tmp_path, disagree_on={"q-disagree"})
    assert art["agreement"]["entity_label_agreement"] == 0.6667
    assert art["agreement"]["doc_label_agreement"] is None, (
        "no doc labels were offered, so the doc leg is unmeasured — not zero")
    assert not any(row["parse_failed"] for row in art["queries"]), (
        "0 of 3 replies failed to parse: a ceiling computed from rows the parser "
        "gave up on is an instrument reading, not a label disagreement, and #1655's "
        "owed ruling is to check exactly this number on the real pass")
    ids = [q["id"] for q in _CEIL_CORPUS]
    gold_queries = [dict(q) for q in _CEIL_CORPUS]
    gold = lac.labels_sha256(gold_queries)
    art_dir = tmp_path / "label-agreement"
    art_dir.mkdir()

    def load(artifact, *, labels_sha=gold):
        """Put ONE artifact on disk and load it. The unlink matters twice over: the
        loader takes the NEWEST file in the directory, so a previous step's mutant left
        behind would be what the next assertion reads — which is exactly how the
        gold-moved refusal here first reported a stub refusal instead."""
        for stale in art_dir.glob("label-agreement-*.json"):
            stale.unlink()
        (art_dir / "label-agreement-20260928-090000.json").write_text(
            json.dumps(artifact), encoding="utf-8")
        return lac.load_artifact(directory=art_dir, expect_labels_sha256=labels_sha)

    # Refusal 1, absent — the state production has been in since this module landed and
    # the reason `--print` exits 2. Named first because it is the state this item exists
    # to end, and because a loader that refused everything would be indistinguishable
    # from one that refuses only what it should unless something gets PAST it.
    with pytest.raises(lac.ArtifactRefused) as exc:
        lac.load_artifact(directory=tmp_path / "nothing-here",
                          expect_labels_sha256=gold)
    assert "no label-agreement artifact under" in str(exc.value)

    # …and the file a real pass writes gets through all four.
    good = load(art)
    assert good["labeler"]["model"] == "djev" and good["labeler"]["kind"] == "engine"
    assert good["ceiling"]["values"]["entity_hit_rate"] == 0.6667

    stub = json.loads(json.dumps(art))
    stub["labeler"]["kind"] = "stub"
    with pytest.raises(lac.ArtifactRefused) as exc:
        load(stub)
    assert "labelled by a stub" in str(exc.value), "refusal 2: a stub agrees with itself"

    with pytest.raises(lac.ArtifactRefused) as exc:
        load(art, labels_sha="f" * 64)
    assert "the gold labels moved" in str(exc.value), (
        "refusal 3: `9b028e9` re-pointed 22 gold names under an unchanged id set")

    edited = json.loads(json.dumps(art))
    edited["ceiling"]["values"]["entity_hit_rate"] = 0.999
    with pytest.raises(lac.ArtifactRefused) as exc:
        load(edited)
    assert "does not reproduce from the artifact" in str(exc.value), (
        "refusal 4: this is the DIVISOR, so a stored value the rows do not support "
        "must never divide tonight's score")
    stripped = json.loads(json.dumps(art))
    stripped["queries"] = []
    with pytest.raises(lac.ArtifactRefused) as exc:
        load(stripped)
    assert "does not reproduce from its own contents" in str(exc.value), (
        "refusal 4's other half: rows removed, so the agreement cannot recompute")

    # ── and the same file through the nightly's own route ─────────────────────
    # Re-write the good artifact first: the battery above left its last MUTANT on
    # disk, and the loader takes the newest file in the directory, so without this the
    # run below would be annotating a score with the rows-removed file's refusal.
    load(art)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(art_dir))
    fields = ev.ceiling_context(gold_queries, scored_ids=ids)
    assert fields["ceiling"]["kind"] == lac.CEILING_KIND, (
        "the ceiling is the gold-label surrogate, not the seed-side anchorless count")
    assert fields["ceiling"]["entity_hit_rate"] == 0.6667
    assert fields["ceiling"]["labeler"] == DJEV_IDENTITY, (
        "the baseline records WHICH ENGINE produced the divisor — #1655's own note is "
        "that the model changes what the number means — and `ceiling_context` carries "
        "kind, model AND endpoint straight out of the artifact")
    assert fields["label_agreement"]["entity_disagreements"] == [
        "q-disagree gold=Research"], (
        "a sub-1.0 agreement cannot be read without the labels that caused it")
    assert fields["entity_hit_rate_normalized"] is None, (
        "before there are scores there is nothing to divide: the field is present "
        "and null, which is the shape that keeps a null from reading as a zero")

    # The score's own denominator travels with it, as `summarize` writes it (#2014):
    # all three queries were offered their gold, so both halves are over the same three.
    assert fields["ceiling"]["n"]["entity_hit_rate"] == 3
    overall = {"entity_hit_rate": 2 / 3, "ci95": {"entity_hit_rate": {"n": 3}}}
    ev._normalize_against_ceiling(fields, overall)
    assert fields["entity_hit_rate_normalized"] == pytest.approx(1.0, abs=1e-3), (
        "0.6667 / 0.6667: at the ceiling, because the one query it missed is the one "
        "the two labelers disagree on")
    assert fields["entity_hit_rate_ceiling_kind"] == lac.CEILING_KIND

    ctl_dir = tmp_path / "control" / "label-agreement"
    ctl_dir.mkdir(parents=True)
    agreeing = _engine_artifact(ctl_dir)
    assert agreeing["agreement"]["entity_label_agreement"] == 1.0, (
        "same three queries, same gold, a labeler that never dissents")
    assert agreeing["ceiling"]["values"]["entity_hit_rate"] == 1.0
    (ctl_dir / "label-agreement-20260928-090001.json").write_text(
        json.dumps(agreeing), encoding="utf-8")
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(ctl_dir))
    ctl = ev.ceiling_context([dict(q) for q in _CEIL_CORPUS], scored_ids=ids)
    ev._normalize_against_ceiling(ctl, overall)
    assert ctl["entity_hit_rate_normalized"] == pytest.approx(0.6667, abs=1e-3), (
        "identical retrieval, labels that agree: 0.6667 of the ceiling, not 1.0 — the "
        "difference between these two numbers is the labels, and only the labels")


def test_a_present_artifact_produces_a_measured_ceiling(tmp_path, monkeypatch):
    art = _synthetic_artifact()
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    fields = ev.ceiling_context(_synth_corpus())
    assert fields["ceiling"]["kind"] == lac.CEILING_KIND
    assert fields["label_agreement"]["labeler"]["model"] == "fixture-ceiling-model", (
        "the run records WHICH labeler produced the ceiling: a ceiling from one model "
        "is a different claim from a ceiling from another, and the item's decision "
        "rule turns on what the number means")
    for metric in ("entity_hit_rate", "entity_recall_avg", "doc_hit_rate",
                   "doc_recall_avg", "mrr_doc", "ndcg10"):
        assert fields["ceiling"][metric] is not None, metric
        assert fields[f"{metric}_ceiling_kind"] == lac.CEILING_KIND
    assert fields["ceiling"]["fact_entity_recall_avg"] is None, (
        "metric values sit flat beside kind/n/excluded, reached by name")
    assert "fact" in fields["ceiling"]["unmeasured"]["fact_entity_recall_avg"]
    assert fields["label_agreement"]["artifact"], (
        "the run records which artifact the ceiling came from")


def test_a_measured_ceiling_survives_the_json_round_trip_into_summary_overall(
        tmp_path, monkeypatch):
    """The measured state, serialised — which is the state automod reads.

    `nightly-*.json → summary.overall` is the boundary the acceptance check is
    written against, and until now the only tests that crossed it were the
    subprocess pair behind `LLOYD_CI=1`, which a gate worktree skips (no live
    engine, no KG store). A clause pinned only by tests that skip where the change
    is verified is not pinned there, so this covers the measured half hermetically:
    a real artifact on disk, `ceiling_context` for real, `summarize` for real, then
    `json.dump` and a re-read. What it can refuse to let happen: a value that only
    exists in the in-process dict, a raw aggregate the division moved, and the
    seed-side `anchorless_query_count` disappearing beside the new keys.
    """
    art = _synthetic_artifact()
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    # Records whose ids ARE the artifact's ids: `ceiling_context` re-averages the
    # ceiling over `scored_ids`, so records named p0..p9 against an artifact about
    # alpha..epsilon would score an empty intersection and quietly assert on nothing.
    queries = _synth_corpus()
    records = [dict(r, id=q["id"]) for r, q in
               zip(_printed_records(len(queries), 2), queries)]
    summary = ev.summarize(records)
    raw_before = {m: summary["overall"][m] for m in METRICS}
    anchorless_before = (summary["overall"]["anchorless_query_count"],
                         list(summary["overall"]["anchorless_query_ids"]))
    fields = ev.ceiling_context(_synth_corpus(), scored_ids=[r["id"] for r in records])
    ev._normalize_against_ceiling(fields, summary["overall"])
    summary["overall"].update(fields)

    path = tmp_path / "nightly-roundtrip.json"
    path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    o = json.loads(path.read_text(encoding="utf-8"))["summary"]["overall"]

    assert o["ceiling"]["kind"] == lac.CEILING_KIND, (
        "a measured ceiling that does not survive `json.dump` is indistinguishable "
        "from an absent one to the next tool that reads this file")
    assert o["label_agreement"]["entity_label_agreement"] is not None
    assert o["label_agreement"]["labeler"]["model"] == "fixture-ceiling-model"
    # The entity leg is the clause's own metric, so IT specifically must carry a
    # ratio — a loop over all six whose `if cap and score` guard could otherwise
    # skip every metric and still report success. On this fixture the stub labeler
    # never picks a gold doc, so all four doc ceilings are 0.0 and their ratios are
    # null by the zero-cap rule; that half is asserted too rather than skipped,
    # because "0.0 ceiling" and "no ceiling" must stay distinguishable on disk.
    assert o["ceiling"]["entity_hit_rate"] == pytest.approx(0.2)
    assert o["entity_hit_rate"] == pytest.approx(0.4)
    assert o["entity_hit_rate_normalized"] == pytest.approx(2.0), (
        "the one number clause 4 names did not survive the write, or was never "
        "divided")
    assert o["entity_hit_rate_ceiling_kind"] == lac.CEILING_KIND
    assert o["entity_recall_avg_normalized"] == pytest.approx(2.5)
    for metric in ("doc_hit_rate", "doc_recall_avg", "mrr_doc", "ndcg10"):
        assert o["ceiling"][metric] == 0.0, metric
        assert o[f"{metric}_normalized"] is None, metric
        assert "undefined" in o["ceiling_notes"][metric], metric
    assert o["ceiling"]["fact_entity_recall_avg"] is None
    assert o["fact_entity_recall_avg_normalized"] is None
    assert {m: o[m] for m in METRICS} == raw_before, (
        "reporting a normalized score changed a raw score")
    assert (o["anchorless_query_count"], o["anchorless_query_ids"]) == anchorless_before, (
        "the seed-side ceiling is automod's armed-regression contract (#1164); the "
        "gold-side one coexists with it")


def test_stale_labels_are_refused(tmp_path, monkeypatch):
    """The corpus's labels moved under the ceiling — commit `9b028e9` re-pointed 22
    gold entity names with the query ids untouched, which a join on query id cannot
    see. A ceiling measured against labels that no longer exist must not divide
    tonight's score."""
    art = _synthetic_artifact(entity_labels="moved")
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    fields = ev.ceiling_context(_synth_corpus())
    assert fields["ceiling"]["kind"] is None
    assert "labels" in fields["ceiling"]["reason"].lower()


def test_a_stub_artifact_is_refused(tmp_path, monkeypatch):
    """The instrument's own guard at the reporter boundary: a nightly that inherited a
    stub artifact would print a fabricated ceiling as tonight's measurement."""
    art = _synthetic_artifact(kind="stub")
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    fields = ev.ceiling_context(_synth_corpus())
    assert fields["ceiling"]["kind"] is None
    assert "stub" in fields["ceiling"]["reason"].lower()


def test_an_artifact_that_does_not_recompute_is_refused(tmp_path, monkeypatch):
    """A ceiling number is only a ceiling if it is the number the scorer produces. An
    artifact whose stored surrogate cannot be recomputed from its rows — hand-edited,
    or written by a scorer of a different shape — is refused rather than divided by."""
    art = _synthetic_artifact(recompute="broken")
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    fields = ev.ceiling_context(_synth_corpus())
    assert fields["ceiling"]["kind"] is None
    assert "recomput" in fields["ceiling"]["reason"].lower()


def test_the_ceiling_is_re_averaged_over_the_queries_actually_scored(tmp_path,
                                                                     monkeypatch):
    """`--limit 3` scores three queries. Dividing a three-query rate by an
    eighty-one-query ceiling yields a number shaped like a position on a scale and
    nothing of the kind, so the divisor is recomputed over the scored ids."""
    art = _synthetic_artifact()
    whole = lac.ceiling(art)
    part = lac.ceiling(art, ids=["alpha", "beta", "gamma"])
    assert whole["n"]["entity_hit_rate"] == 5
    assert part["n"]["entity_hit_rate"] == 3
    assert part["values"] != whole["values"], (
        "restricting the ids changed nothing, so the restriction is not applied")
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    fields = ev.ceiling_context(_synth_corpus(),
                                scored_ids=["alpha", "beta", "gamma"])
    assert fields["ceiling"]["n"]["entity_hit_rate"] == 3


def test_the_two_ceilings_are_named_as_distinct_instruments():
    """Reading one as the other is the conflation this clause exists to prevent: a
    query the seeds cannot anchor cannot be hit by any retriever, while a query the
    two labelers disagree on can be hit and still be noise."""
    # The kind string has ONE owner — the module that measures the ceiling. run_eval
    # passes it through from the artifact rather than keeping its own copy, because a
    # second copy of the string is the drift bug this item is about.
    assert lac.CEILING_KIND == "gold_label_surrogate"
    assert set(ev.CEILING_UNMEASURED) == set(lac.UNMEASURED_METRICS)
    assert "anchorless" not in lac.CEILING_KIND
    assert ev._ceiling_absent_fields("x")["ceiling"]["kind"] is None


# ── the serialisation seam: a real run, the real artifact, real paths ────────

def _live_pools_or_skip(why: str):
    """Entity names and vault paths from the live store, or a skip naming its nodes.

    Read-only, and deliberately not routed through `app.kg_store.store()`: the point
    is the real entity-name table, and `configure` here is reached only when the
    production database already exists, so nothing is provisioned. Inside a gate
    worktree `LLOYD_DATA` points at the round's own data root, which has no KG store
    at all — provisioning one there is the exact defect
    `tests/test_eval_label_agreement.py::
    test_restoring_the_default_store_never_provisions_an_absent_database` now pins.
    """
    from app.data_root import PRODUCTION_DATA_ROOT
    kg = PRODUCTION_DATA_ROOT / "_pipeline" / "vault-derived" / "kg.sqlite"
    vault = Path.home() / "obsidian"
    if not kg.exists() or not vault.is_dir():
        pytest.skip(f"no live KG/vault to build a matching artifact from; stands in "
                    f"for {why}")
    import app.kg_store as kg_store
    store = kg_store.configure(kg)
    paths = sorted(str(p.relative_to(vault)) for p in vault.rglob("*.md")
                   if ".git" not in p.parts)[:4000]
    return store.entities.all(), paths


def _run(label: str, artifact_dir: Path) -> tuple[dict, str]:
    env = dict(os.environ)
    env["LLOYD_LABEL_AGREEMENT_DIR"] = str(artifact_dir)
    proc = subprocess.run([str(PY), str(SCRIPT), "--label", label, "--limit", "3"],
                          capture_output=True, text=True, timeout=900,
                          cwd=str(ROOT), env=env)
    assert proc.returncode == 0, proc.stderr[-2500:]
    files = sorted(BASELINES.glob(f"{label}-*.json"), key=lambda p: p.stat().st_mtime)
    assert files, "the run wrote no baseline record"
    return json.loads(files[-1].read_text(encoding="utf-8")), proc.stdout


@pytest.fixture(scope="module")
def with_artifact(tmp_path_factory) -> tuple[dict, str]:
    if not CI:
        pytest.skip(CI_SKIP + "; stands in for with_artifact, which 3 nodes take "
                    "(test_baseline_carries_ceiling_label_agreement_and_normalized, "
                    "test_seed_side_anchorless_fields_present_and_unmodified, "
                    "test_raw_values_unchanged_by_the_ceiling)")
    names, paths = _live_pools_or_skip(
        "the same 3 nodes: test_baseline_carries_ceiling_label_agreement_and_"
        "normalized, test_seed_side_anchorless_fields_present_and_unmodified, "
        "test_raw_values_unchanged_by_the_ceiling")
    art = lac.label_corpus(
        labeler=lac.stub_labeler(),
        labeler_identity={"kind": "engine", "model": "fixture-ceiling-model",
                          "endpoint": "fixture://local"},
        queries=lac.load_corpus(), entity_names=names, vault_paths=paths,
        entity_cap=len(names), doc_cap=len(paths))
    directory = tmp_path_factory.mktemp("ceiling-present")
    _write(art, directory)
    return _run("ceiling-present", directory)


@pytest.fixture(scope="module")
def without_artifact(tmp_path_factory) -> tuple[dict, str]:
    if not CI:
        pytest.skip(CI_SKIP + "; stands in for without_artifact -> "
                    "test_no_artifact_emits_ceiling_null_with_a_reason and its 2 "
                    "sibling nodes")
    return _run("ceiling-absent", tmp_path_factory.mktemp("ceiling-absent"))


def test_baseline_carries_ceiling_label_agreement_and_normalized(with_artifact):
    """The clause's own check, on the written file: the keys are present, the ratio
    is the division, and the ceiling kind is named."""
    out, stdout = with_artifact
    o = out["summary"]["overall"]
    assert o["label_agreement"], "an artifact was present, so agreement is measured"
    assert o["label_agreement"]["labeler"]["model"] == "fixture-ceiling-model"
    assert o["ceiling"]["kind"] == lac.CEILING_KIND
    for metric in METRICS:
        assert f"{metric}_normalized" in o, (
            f"{metric}: a dropped key is not a null")
        assert f"{metric}_ceiling_kind" in o
    cap = o["ceiling"]["entity_hit_rate"]
    assert cap is not None
    if o["entity_hit_rate"] is not None and cap:
        assert o["entity_hit_rate_normalized"] == pytest.approx(
            o["entity_hit_rate"] / cap, abs=1e-3)
        assert o["entity_hit_rate_ceiling_kind"] == lac.CEILING_KIND
    assert lac.CEILING_KIND in stdout, (
        "the printed line names the ceiling kind, not just the JSON")
    # The ceiling dict carries its per-metric values FLATTENED beside `kind`
    # (`ceiling.entity_hit_rate`), not nested under a `values` key — the shape the
    # artifact itself uses is nested, so this is the one place the two diverge and
    # the place a reader's `ceiling["values"]` KeyError comes from. Pinned in the
    # written file because that is where the nightly reads it.
    assert "values" not in o["ceiling"], (
        "the baseline's ceiling dict grew a nested `values` while the artifact keeps "
        f"the same numbers flat: readers now have two shapes for one field {list(o['ceiling'])}")
    assert o["ceiling"]["fact_entity_recall_avg"] is None
    assert o["fact_entity_recall_avg_normalized"] is None
    assert o["label_agreement"]["artifact"]


def test_no_artifact_emits_ceiling_null_with_a_reason(without_artifact):
    out, stdout = without_artifact
    o = out["summary"]["overall"]
    assert o["ceiling"]["kind"] is None
    assert o["ceiling"]["reason"]
    assert "reason" in o["label_agreement"]
    for metric in METRICS:
        assert o[f"{metric}_normalized"] is None
        assert o[f"{metric}_ceiling_kind"] is None
    assert "null" in stdout, "a null and a 0.0 must not print the same"


def test_seed_side_anchorless_fields_present_and_unmodified(without_artifact,
                                                            with_artifact):
    """`anchorless_query_count` is in automod's armed-regression contract (#1164). The
    gold-side ceiling coexists with it — it does not rename, reshape or shadow it — and
    it is identical with and without a ceiling artifact, because the two bound
    different failures and the reader needs both."""
    a = with_artifact[0]["summary"]["overall"]
    b = without_artifact[0]["summary"]["overall"]
    assert isinstance(b["anchorless_query_count"], int)
    assert isinstance(b["anchorless_query_ids"], list)
    assert len(b["anchorless_query_ids"]) == b["anchorless_query_count"]
    assert a["anchorless_query_ids"] == b["anchorless_query_ids"], (
        "the seed-side field moved once a gold-side ceiling was available")


def test_raw_values_unchanged_by_the_ceiling(with_artifact, without_artifact):
    """The ceiling is a divisor reported beside the number, never folded into it. The
    two runs are the same three queries over the same tree, so any movement here is
    the instrument changing what it measures."""
    a = with_artifact[0]["summary"]["overall"]
    b = without_artifact[0]["summary"]["overall"]
    for k in RAW_KEYS:
        assert a.get(k) == b.get(k), f"{k} moved once a ceiling was available"


def test_the_skill_names_the_ceiling_and_the_veto():
    """Clause 5. The report is where the next reader learns which ceiling is which and
    what a low agreement forbids, so it lives in the skill, not only in code. Skipped
    when the vault is not mounted here; the same arrangement as
    `tests/test_automod_skill_maps_the_radius`, which asserts on vault prose too."""
    if not SKILL.exists():
        pytest.skip("no vault skills tree here")
    text = SKILL.read_text(encoding="utf-8")
    assert lac.CEILING_KIND in text, "the skill names the gold-side ceiling"
    assert "anchorless" in text, "and names the seed-side ceiling beside it"
    assert "entity_label_agreement" in text
    assert "0.80" in text, (
        "below 0.80 the disagreement set must be named, so a low ceiling is never "
        "reported without the labels that caused it")


# ── #1823: `labels_unofferable`, the field the `_narrow` docstring promised ────
#
# `_narrow`'s docstring closed with "Raising the cap is the knob that converts exclusions
# back into measurements, and `labels_unofferable` beside the ceiling says how many the
# caps caused." `git grep -n "labels_unofferable"` before this round found that one
# docstring line and nothing else: the field was never written, and the 2026-09-29
# artifact's `ceiling` block carried only `kind`, `values`, `n`, `excluded`, `unmeasured`.
# So the one number that tells a person whether widening `ENTITY_CAP` is worth a re-run
# did not exist anywhere in the artifact.




def _labeled_row(qid: str, ent_gold: list[str], ent_menu: list[str],
                 ent_pick: list[str], doc_gold: list[str], doc_menu: list[str],
                 doc_pick: list[str], **extra) -> dict:
    """One row in the shape `label_corpus` writes, so `ceiling()` reads it as a real run.

    `label_corpus` is driven over in `tests/test_eval_label_agreement.py`; what this file
    owns is the artifact-as-consumed, and `ceiling()` is the function that assembles the
    block, so the field is pinned where it is written and where disk hands it back.
    """
    return {"id": qid, "query": f"query {qid}", "category": "project",
            "primary_entities": ent_gold, "primary_docs": doc_gold,
            "entity_candidates": ent_menu, "doc_candidates": doc_menu,
            "second_entities": ent_pick, "second_docs": doc_pick,
            "entity_labels_offered": sum(
                1 for g in ent_gold
                if any(lac.entity_label_satisfied(g, [c]) for c in ent_menu)),
            "doc_labels_offered": sum(
                1 for g in doc_gold
                if any(lac.doc_label_satisfied(g, [c]) for c in doc_menu)),
            **extra}


def test_ceiling_carries_labels_unofferable_per_leg_under_the_promised_name():
    """Clause 2: the name the docstring uses is the name the block carries, per leg.

    Both legs are present even when one has nothing to report, because a consumer reading
    `ceiling["labels_unofferable"][leg]` must not KeyError on a quiet leg, and the split
    beside the count is what makes the count actionable: 1 label excluded says little,
    1-of-1 outside the cap says a wider cap recovers it.
    """
    art = {"queries": [
        _labeled_row(
            "a", ["Robot", "Raspberry Pi 5"], ["Robot"], ["Robot"],
            ["memory/entities/robot.md"], ["memory/entities/robot.md"],
            ["memory/entities/robot.md"],
            entity_labels_unofferable=[
                {"gold": "Raspberry Pi 5", "kind": lac.OUTSIDE_CAP}],
            doc_labels_unofferable=[]),
        # Nothing unofferable on this query: the legs' counts must still be there, at 0.
        _labeled_row("b", ["Vision System"], ["Vision System"], ["Vision System"],
                     ["memory/entities/vision.md"], ["memory/entities/vision.md"],
                     ["memory/entities/vision.md"],
                     entity_labels_unofferable=[], doc_labels_unofferable=[]),
    ]}
    block = lac.ceiling(art)
    assert "labels_unofferable" in block, sorted(block)
    assert block["labels_unofferable"] == {"entity": 1, "doc": 0}, \
        block["labels_unofferable"]
    detail = block["labels_unofferable_detail"]["entity"]
    assert detail["total"] == 1 and detail["outside_cap"] == 1 \
        and detail["absent_from_namespace"] == 0 and detail["unclassified"] == 0, detail
    assert block["labels_unofferable_detail"]["doc"]["total"] == 0, \
        block["labels_unofferable_detail"]["doc"]


def test_labels_unofferable_survives_the_disk_round_trip_and_a_legacy_artifact():
    """The field must read the same after JSON, and stay honest when it was never written.

    A run writes the ceiling block once and every later reader — `--print`, the nightly
    reporter, the owed-check job that decides whether to widen the cap — reads it out of
    JSON, so a leg that recorded zero exclusions must not come back as missing and be
    mistaken for an old artifact. The genuinely old shape is different and must not be
    flattened into the same number: a pre-#1823 artifact carries no per-label field at
    all, and reads as `None` (unmeasured), not as the reassuring 0 that a run with no
    exclusions also produces. Collapsing those two is how an instrument reports a
    measurement it never took.
    """
    art = {"queries": [_labeled_row(
        "a", ["Robot"], ["Robot"], ["Robot"], ["memory/entities/robot.md"],
        ["memory/entities/robot.md"], ["memory/entities/robot.md"],
        entity_labels_unofferable=[], doc_labels_unofferable=[])]}
    art["ceiling"] = lac.ceiling(art)

    revived = json.loads(json.dumps(art))
    assert revived["ceiling"]["labels_unofferable"] == {"entity": 0, "doc": 0}
    assert lac.ceiling(revived)["labels_unofferable"] == {"entity": 0, "doc": 0}

    legacy = {"queries": [{k: v for k, v in row.items()
                           if "labels_unofferable" not in k}
                          for row in art["queries"]]}
    legacy["queries"][0]["primary_entities"] = ["Robot", "Nonexistent Entity"]
    assert all("entity_labels_unofferable" not in row for row in legacy["queries"])
    assert lac.ceiling(legacy)["labels_unofferable"] == {"entity": None, "doc": None}, \
        "a run that recorded no reasons must not report that there were none"
    assert lac.ceiling(legacy)["labels_unofferable_detail"] == {"entity": None,
                                                                "doc": None}


# ── #1938: the offered-only reading reaches the baseline block and the page ───

#: The four #1823 figures #1938 carries into `summary.overall.label_agreement`. The
#: all-gold rate stays the #654 veto figure; these two pairs are the reading that
#: tells a cap artefact from a labelling disagreement. The block has carried the
#: offered-only DENOMINATOR (`entity_labels_offered`) since #1823 without its rate or
#: its numerator, so nothing downstream of this JSON could say which reading any
#: printed rate was — and the veto verdict is read on this page, not in the labeler's
#: own `--print`.
OFFERED_KEYS = ("entity_label_agreement_when_offered",
                "entity_labels_agreed_when_offered",
                "doc_label_agreement_when_offered",
                "doc_labels_agreed_when_offered")

#: `summary.overall.label_agreement` exactly as the 2026-09-30 nightly wrote it: 13
#: keys, no `when_offered` among them. An extract of the vault witness
#: `backlog/data/nightly-20260930-20260930-060339.json` (`wc -l` 12790), committed here
#: because the gate's HOME has no `~/obsidian` and a node that read the vault copy would
#: skip rather than pin.
WITNESS = (Path(__file__).resolve().parent / "fixtures"
           / "nightly-20260930-label_agreement.json")


def _cap_narrowed(art: dict) -> dict:
    """The same artifact with every AGREED entity gold dropped from its offered menu,
    and the two derived blocks re-stamped from the mutated queries.

    This is the shape #1823 measured on the live store — 52 of the 53 unofferable
    entity gold names were this instrument's own `ENTITY_CAP`, not a name missing from
    the namespace — and it is the only fixture that separates the two readings. With
    the shipped synthetic menu every gold is offered, so `*_when_offered` equals the
    all-gold figure and a block that merely copied the all-gold rate into the
    offered-only slot would pass; narrowed this way, the entity leg reads 0.2 all-gold
    against 0.0 offered-only. Re-stamping `agreement` and `ceiling` is what keeps
    `load_artifact`'s two recompute guards admitting the file rather than refusing an
    artifact whose stored figure no longer follows from its own rows.
    """
    for q in art["queries"]:
        agreed = [g for g in (q.get("primary_entities") or [])
                  if lac.entity_label_satisfied(g, q.get("second_entities") or [])]
        if agreed:
            q["entity_candidates"] = [
                c for c in (q.get("entity_candidates") or [])
                if not any(lac.entity_label_satisfied(g, [c]) for g in agreed)]
    art["agreement"] = lac.agreement(art)
    art["ceiling"] = lac.ceiling(art)
    return art


def _veto_page_fields(entity_all_gold: float, entity_offered: float) -> dict:
    """A measured ceiling block for the printed page.

    The doc leg is the 2026-09-29 artifact's RECORDED doc leg, copied out of
    ``WITNESS``: all-gold 0.259 over 139 gold labels (36 agreed) against 0.5714 among
    the 63 that were offered. The two entity figures are arguments because clause 4
    needs rates on both sides of 0.80, and the callers pass that artifact's recorded
    pair (0.3404 all-gold = 32/94, 0.7273 offered-only = 32/44) wherever the page is
    being read rather than the veto being probed. Every figure here is therefore a
    recorded one except the two entity rates clause 4 deliberately moves.
    """
    fields = ev._ceiling_absent_fields("unused")
    fields["ceiling"] = {"kind": lac.CEILING_KIND, "entity_hit_rate": 0.8,
                         "doc_hit_rate": 0.9, "entity_recall_avg": 0.7,
                         "doc_recall_avg": 0.8, "mrr_doc": 0.6, "ndcg10": 0.7,
                         "fact_entity_recall_avg": None}
    fields["label_agreement"] = {
        "entity_label_agreement": entity_all_gold,
        "entity_label_agreement_when_offered": entity_offered,
        "doc_label_agreement": 0.259,
        "doc_label_agreement_when_offered": 0.5714,
        "entity_labels": 94, "entity_labels_agreed": 32,
        "entity_labels_agreed_when_offered": 32, "entity_labels_offered": 44,
        "doc_labels": 139, "doc_labels_agreed": 36,
        "doc_labels_agreed_when_offered": 36, "doc_labels_offered": 63,
        "labeler": {"model": DJEV_IDENTITY["model"],
                    "endpoint": DJEV_IDENTITY["endpoint"]},
        "entity_disagreements": [], "doc_disagreements": []}
    return fields


def _verdict_line(text: str) -> str:
    """The agreement verdict line of a printed page — both readings have to sit on
    that one line, so a test that searched the whole page could be satisfied by the
    offered-only figure turning up anywhere else on it."""
    hits = [ln for ln in text.splitlines()
            if ln.strip().startswith("entity_label_agreement=")]
    assert len(hits) == 1, f"expected exactly one agreement verdict line, got {hits}"
    return hits[0]


def test_the_emitted_block_carries_the_offered_only_rate_and_numerator(tmp_path,
                                                                       monkeypatch):
    """Clause 1: `summary.overall.label_agreement` carries both readings, per leg.

    Until now the offered-only rate existed only inside `label_agreement_ceiling
    --print`. The baseline JSON carried `entity_labels_offered` — 44 on the 2026-09-29
    artifact — with neither `entity_label_agreement_when_offered` (0.7273) nor its
    numerator (32), while the veto line printed the all-gold 0.3404 with nothing beside
    it. Each value is asserted against `lac.agreement()` over the very artifact the
    block was built from, so the block cannot invent a number, and against a menu
    narrowed by the cap, so it cannot copy the all-gold rate into the offered-only slot
    either.
    """
    art = _cap_narrowed(_synthetic_artifact())
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    fields = ev.ceiling_context(_synth_corpus())
    la = fields["label_agreement"]
    fresh = lac.agreement(lac.load_artifact(
        directory=tmp_path,
        expect_labels_sha256=lac.labels_sha256(_synth_corpus())))
    for key in OFFERED_KEYS:
        assert key in la, f"{key} absent from the block: to the next reader an " \
                          "absent key and a measured zero are the same reading"
        assert la[key] == fresh[key], (
            f"{key}: block says {la[key]!r}, the artifact's own rows compute "
            f"{fresh[key]!r}")
    assert (la["entity_label_agreement"],
            la["entity_label_agreement_when_offered"]) == (0.2, 0.0), (
        "this fixture only tests clause 1 while the two readings differ: 1 of 5 gold "
        "labels agreed, and that label's name was outside the narrowed menu")
    assert (la["entity_labels_agreed"], la["entity_labels_agreed_when_offered"],
            la["entity_labels_offered"]) == (1, 0, 4), (
        "the numerator that belongs to the offered-only denominator, not the "
        "all-gold one beside it")
    # And in the WRITTEN bytes, which is the artefact the acceptance names: the nightly
    # serialises this block with `json.dump` and the next reader opens the file, so a
    # key that exists only in the in-process dict never reached anyone.
    summary = ev.summarize(_printed_records(3, 1))
    summary["overall"].update(fields)
    written = tmp_path / "written-baseline.json"
    written.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    on_disk = json.loads(written.read_text(encoding="utf-8"))["summary"]["overall"]
    for key in OFFERED_KEYS:
        assert key in on_disk["label_agreement"], (
            f"{key} did not survive the write into summary.overall.label_agreement")
    assert on_disk["label_agreement"]["entity_label_agreement_when_offered"] == 0.0


def test_an_artifact_stored_without_them_gets_them_back_filled_not_nulled(tmp_path,
                                                                          monkeypatch):
    """Clause 2: the shape of every artifact on disk today.

    `label-agreement-20260929-005508.json` — the only one on this box — predates #1823
    and stores none of the four keys. So a copy written as a raw
    `art["agreement"][key]` raises KeyError on tonight's nightly, and one written as
    `.get(key)` emits a null for the very reading the verdict line now prints beside the
    veto figure. `lac._agreement_for_print` fills ONLY the absent keys, by deterministic
    recomputation over the stored rows with no engine call, which is the same
    measurement the labeler would have written rather than a new one.
    """
    art = _cap_narrowed(_synthetic_artifact())
    for key in OFFERED_KEYS:
        art["agreement"].pop(key)
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    fields = ev.ceiling_context(_synth_corpus())
    la = fields["label_agreement"]
    for key in OFFERED_KEYS:
        assert la.get(key) is not None, (
            f"{key} came back {la.get(key)!r}; the labeler's own --print has a number "
            "here, so a null in the baseline is the asymmetry this item closes")
    assert (la["entity_label_agreement_when_offered"],
            la["entity_labels_agreed_when_offered"]) == (0.0, 0), (
        "back-filled by recomputation and not copied across: the all-gold numerator is "
        "1 on this artifact and the offered-only one is 0")


def test_the_agreement_verdict_line_names_both_readings(capsys):
    """Clause 3: the all-gold rate and the offered-only rate on the verdict's own line.

    The page is the seam: the nightly reporter reads this text, and a reader who
    copies 0.3404 off it cannot otherwise tell a labelling disagreement from
    `ENTITY_CAP` narrowing a namespace that does hold the name. Both figures have to be
    on the agreement line itself, each named by which reading it is.
    """
    ev.print_table(*_printed_summary(_veto_page_fields(0.3404, 0.7273)))
    line = _verdict_line(capsys.readouterr().out)
    assert "0.3404" in line and "0.7273" in line, line
    assert "all-gold" in line and "offered" in line, (
        f"two rates with nothing naming which is which is the original defect:\n{line}")


def test_the_veto_still_consumes_the_all_gold_rate_alone(capsys):
    """Clause 4: the offered-only figure is diagnostic only, never the veto input.

    Pinned in both directions, because one direction alone cannot say which of the two
    printed figures the sub-0.80 comparison reads: all-gold 0.72 with offered-only 0.95
    must still call the labels too ambiguous, and all-gold 0.95 with offered-only 0.55
    must still let the entity hit rate stand. #654's veto asks whether the gold can be
    trusted at all, and a gold that was never offered cannot be; relaxing it on the
    offered-only reading is a person's call (#1823 owed entry 3), not a print change.
    """
    ev.print_table(*_printed_summary(_veto_page_fields(0.72, 0.95)))
    out = capsys.readouterr().out
    assert "labels too ambiguous to call it a retrieval defect" in out, out
    ev.print_table(*_printed_summary(_veto_page_fields(0.95, 0.55)))
    out2 = capsys.readouterr().out
    assert "labels hold; the entity hit rate stands as a retrieval result" in out2, out2
    assert "too ambiguous" not in out2, out2


def test_the_witness_block_re_derives_and_the_four_keys_are_pure_addition(tmp_path,
                                                                          monkeypatch):
    """Clause 5: the quoted figures have committed bytes behind them, and #1938 adds
    keys without moving one.

    The witness is a vault commit — `backlog/data/nightly-20260930-20260930-060339.json`,
    `wc -l` 12790, sha256 `b037d9751b243c4b…` — and no node in this repo can read it: the
    gate runs with HOME at the round home, where `~/obsidian` does not exist, so a node
    that opened the vault copy would skip, and a skipping node pins nothing. What is
    pinned here is the extract of that single block, committed in this repo and holding
    the identical values, for both halves of the clause.

      * every figure this item quotes re-derives out of the extract's own counts: entity
        32/94 = 0.3404 all-gold against 32/44 = 0.7273 offered-only, doc 36/139 = 0.259
        against 36/63 = 0.5714. That is the whole argument for keeping the all-gold rate
        as the veto figure and printing the other beside it — the doc leg moves by more
        than half again on the same 36 agreed labels;
      * the block #1938 emits is these keys plus exactly the four offered-only ones, so
        nothing a reader took off last night's baseline is missing from tonight's.
    """
    witness = json.loads(WITNESS.read_text(encoding="utf-8"))
    assert not [k for k in witness if "when_offered" in k], (
        "the extract has stopped being the pre-fix baseline it is quoted as")
    assert witness["entity_label_agreement"] == 0.3404
    assert witness["doc_label_agreement"] == 0.259
    assert round(witness["entity_labels_agreed"] / witness["entity_labels"], 4) == 0.3404
    assert round(witness["entity_labels_agreed"]
                 / witness["entity_labels_offered"], 4) == 0.7273
    assert round(witness["doc_labels_agreed"] / witness["doc_labels"], 4) == 0.259
    assert round(witness["doc_labels_agreed"]
                 / witness["doc_labels_offered"], 4) == 0.5714
    art = _synthetic_artifact()
    _write(art, tmp_path)
    monkeypatch.setenv(lac.OUT_DIR_ENV, str(tmp_path))
    emitted = ev.ceiling_context(_synth_corpus())["label_agreement"]
    assert set(witness) <= set(emitted), (
        f"a key the 2026-09-30 baseline wrote has stopped being emitted: "
        f"{sorted(set(witness) - set(emitted))}")
    assert set(emitted) - set(witness) == set(OFFERED_KEYS), (
        "the new block is not a superset of last night's — either a key went missing or "
        f"something besides the four offered-only figures arrived: "
        f"{sorted(set(emitted) - set(witness))}")
