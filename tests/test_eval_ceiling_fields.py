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
