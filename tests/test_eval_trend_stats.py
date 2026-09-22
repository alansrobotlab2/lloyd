#!/usr/bin/env python3
"""Tests for ``scripts/eval_trend_stats.py`` (backlog #608).

Two kinds of test live here and they are not interchangeable.

*Synthetic* tests build two baseline files whose McNemar p-values and bootstrap
behaviour are known before the script runs, so an assertion can be exact: one
flipped query gives ``b=1 c=0`` and p = 1.000, six one-sided flips out of eight
give p = 0.03125, and a dataset in which every query moves by exactly the same
amount must give a bootstrap interval of width zero — which is the only proof
that the interval comes from resampling queries rather than from a normal-theory
formula bolted onto the delta.

*Live* tests re-run the audit over the real ``eval/baselines/nightly-*.json``.
Those files are gitignored, so the audit resolves them through
``default_baselines_dir()``, which falls back to the live checkout exactly like
``app/uptake.py:224 lloyd_root()`` does for the logs it reads. The live tests pin
the two numbers backlog #608's acceptance check is written in: **11 of 12**
transitions withheld over 2026-09-04..2026-09-17, and the 09-09 "third
consecutive night of entity-side decline. Not noise." verdict labelled
**UNSUPPORTED**. Neither assertion skips: a skipped acceptance check is a gate
that reads green because nobody looked.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The document-corpus vocabulary is imported, not restated: a test that typed
# `"doc"` and `"doc_vectors"` as literals would keep passing if the writer and
# the reader came to disagree about either string, and which key MEANS what is
# the substance of backlog #1374.
from app.doc_corpus import DOC_DRIFT_KEY, DOC_KEY  # noqa: E402
from scripts.eval_trend_stats import (  # noqa: E402
    ALPHA,
    CONT_LEGS,
    CORPUS_KEYS,
    UnjoinableQueries,
    audit_transition,
    corpus_diff,
    default_baselines_dir,
    fmt_drift,
    join_ids,
    load_window,
    main,
    mcmemar_exact,
    paired_bootstrap,
    print_transition,
    REPORTING_CONTRACT,
    required_n,
    wilson_interval,
)

IDS = tuple(f"q{i}" for i in range(1, 9))


def _night(label: str, day: int, entity, doc, ndcg, rr, corpus):
    """One baseline document in the shape ``eval/run_eval.py`` actually writes.

    Per-query results live under ``records[]`` with a ``scoring`` block; the
    aggregate block is ``summary.overall`` and the drift term is ``corpus``. A
    writer that invented ``results`` or ``queries`` would pass a mock-based test
    and find nothing on real data, so the fixture keeps the real key names.
    """
    doc_out = {
        "label": label,
        "ran_at": f"2026-01-{day:02d}T13:00:00+00:00",
        "records": [
            {"id": q, "query": f"query {q}",
             "scoring": {"entity_hit": bool(e), "doc_hit": bool(d),
                         "entity_recall": 1.0 if e else 0.0,
                         "doc_recall": 1.0 if d else 0.0,
                         "ndcg10": n, "rr_doc": r, "first_doc_rank": 1,
                         "fact_entity_recall": 1.0 if e else 0.0}}
            for q, e, d, n, r in zip(IDS, entity, doc, ndcg, rr)
        ],
        "summary": {"overall": {"n_queries": len(IDS)}},
    }
    if corpus is not None:
        doc_out["corpus"] = dict(corpus)
    return doc_out


def _write(tmp_path: Path, name: str, doc: dict) -> Path:
    d = tmp_path / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return d


CORPUS_A = {"facts": 200000, "edges_active": 4000, "entities": 23600}
CORPUS_B = {"facts": 245906, "edges_active": 32340, "entities": 24290}


# ===========================================================================
# Clause 1 — the join, and both binary legs with an exact McNemar p
# ===========================================================================

def test_two_nights_join_on_records_id_and_print_both_hit_legs_with_exact_p(tmp_path, capsys):
    """One flipped query out of eight is reported as one flipped query.

    ``entity_hit`` q2 goes 1 -> 0 and nothing else moves, so ``b=1 c=0`` and the
    exact two-sided McNemar p is 1.000 (P(Bin(1, 0.5) <= 0) = 0.5, doubled).
    ``doc_hit`` does not move at all: ``b=0 c=0``, p = 1.000. Both legs print
    their delta, their discordant counts, their p, and a Wilson interval on the
    later night's own rate.
    """
    a = _night("nightly-20260101", 1,
               [1, 1, 0, 0, 1, 1, 0, 0], [1] * 8,
               [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
               [1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0], CORPUS_A)
    b = _night("nightly-20260102", 2,
               [1, 0, 0, 0, 1, 1, 0, 0], [1] * 8,
               [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
               [1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0], CORPUS_A)
    d = _write(tmp_path, "nightly-a.json", a)
    _write(tmp_path, "nightly-b.json", b)
    nights = load_window(d)
    t = audit_transition(nights[0], nights[1])
    print_transition(t)
    out = capsys.readouterr().out

    assert "entity_hit" in out and "doc_hit" in out
    leg = t.legs["entity_hit"]
    assert (leg["b"], leg["c"]) == (1, 0)
    assert leg["p"] == pytest.approx(1.0)
    assert leg["exact"] is True, "a binary leg must never be an approximation"
    assert leg["delta"] == pytest.approx(-0.125)
    assert leg["prev_rate"] == pytest.approx(0.5)
    assert leg["cur_rate"] == pytest.approx(0.375)
    assert t.legs["doc_hit"]["m"] == 0 and t.legs["doc_hit"]["p"] == pytest.approx(1.0)
    assert "b=1 c=0" in out
    assert "exact McNemar p=1.000" in out
    assert "delta -0.125" in out
    assert "Wilson 95%" in out


def test_exact_mcnemar_values_are_the_hand_computable_ones():
    """The p-values this file stands behind, checked against arithmetic.

    Under the null the ``m = b + c`` discordant pairs are coin flips, so
    ``p = 2 * P(Bin(m, 0.5) <= min(b, c))`` capped at 1:

    ======  ======  =========  =========================================
      b       c      p          why
    ======  ======  =========  =========================================
      0       0     1.000      no discordance, no evidence of anything
      1       0     1.000      2 * 0.5**1
      2       0     0.500      2 * 0.5**2
      3       0     0.250      2 * 0.5**3
      6       0     0.03125    2 * 0.5**6  -- the only case here that rejects
      3       7     0.2265625  2 * P(Bin(10, 0.5) <= 3)
    ======  ======  =========  =========================================
    """
    cases = [(0, 0, 1.0), (1, 0, 1.0), (2, 0, 0.5), (3, 0, 0.25), (6, 0, 0.03125)]
    for b, c, expected in cases:
        prev = [1] * b + [0] * c + [0] * 2 + [1] * 2
        cur = [0] * b + [1] * c + [0] * 2 + [1] * 2
        got = mcmemar_exact(prev, cur)
        assert (got["b"], got["c"]) == (b, c)
        assert got["p"] == pytest.approx(expected), f"b={b} c={c}"
    # A two-sided case: b=3 (1 -> 0), c=7 (0 -> 1) out of m=10 discordant pairs,
    # so p = 2 * P(Bin(10, 0.5) <= 3) = 2 * 176/1024 = 0.34375.
    prev = [1, 1, 1] + [0] * 7
    cur = [0, 0, 0] + [1] * 7
    got = mcmemar_exact(prev, cur)
    assert (got["b"], got["c"]) == (3, 7)
    assert got["p"] == pytest.approx(0.34375)
    assert got["significant"] is False


def test_six_one_sided_flips_out_of_eight_reject_and_are_not_called_noise():
    """The audit is not a machine that always says "withheld".

    Six of eight queries losing their entity hit is ``b=6 c=0``, exact
    p = 0.03125 < 0.05, so the leg rejects. An instrument that could never
    reject would be theatre; this one rejects at the size the arithmetic supports
    and no smaller.
    """
    prev = [1, 1, 1, 1, 1, 1, 0, 0]
    cur = [0, 0, 0, 0, 0, 0, 0, 0]
    got = mcmemar_exact(prev, cur)
    assert got["p"] == pytest.approx(2 * 0.5 ** 6)
    assert got["significant"] is True
    assert got["p"] < ALPHA


def test_mcmemar_refuses_unpaired_input_rather_than_assuming_alignment():
    with pytest.raises(ValueError, match="paired"):
        mcmemar_exact([1, 1, 0], [1, 0])


def test_a_query_id_present_in_only_one_night_errors_instead_of_printing_a_delta(tmp_path, capsys):
    """Clause 1's error path, and it fires on real data.

    ``qwen35-users`` was replaced by ``qwen38-local-serving`` between the 09-07
    and 09-08 baselines. A paired test over two different question sets is not a
    paired test, so ``join_ids`` raises, ``audit_transition`` records the failure,
    and the report prints **no delta at all** for that transition — naming the
    ids instead. The alternative, quietly dropping the unmatched query, turns a
    changed benchmark into a changed score.
    """
    a = _night("nightly-20260101", 1, [1] * 8, [1] * 8,
               [0.5] * 8, [0.5] * 8, CORPUS_A)
    b = _night("nightly-20260102", 2, [1] * 8, [1] * 8,
               [0.5] * 8, [0.5] * 8, CORPUS_A)
    b["records"][7]["id"] = "q-new"          # the benchmark changed questions
    d = _write(tmp_path, "nightly-a.json", a)
    _write(tmp_path, "nightly-b.json", b)
    nights = load_window(d)

    with pytest.raises(UnjoinableQueries) as exc:
        join_ids(nights[0], nights[1])
    assert exc.value.only_prev == ["q8"]
    assert exc.value.only_cur == ["q-new"]

    t = audit_transition(nights[0], nights[1])
    assert t.joinable is False and t.withheld is True
    print_transition(t)
    out = capsys.readouterr().out
    assert "ERROR" in out and "q8" in out and "q-new" in out
    after = out.split("ERROR", 1)[1]
    assert not re.search(r"delta [+-]\d", after), \
        "an unjoinable pair must print no number at all, not a delta over a partial join"
    assert "cannot evaluate" in out


def test_a_baseline_that_is_not_a_per_query_record_set_is_refused(tmp_path):
    """No ``records[]`` means there is nothing to pair, which is not a zero."""
    p = tmp_path / "baselines"
    p.mkdir(parents=True)
    (p / "nightly-20260101-20260101-060000.json").write_text(
        json.dumps({"label": "nightly-20260101", "summary": {"overall": {"ndcg10": 0.5}}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="records"):
        load_window(p)


# ===========================================================================
# Clause 2 — paired bootstrap for ndcg10 / mrr_doc, labelled as resampling
# ===========================================================================

def test_ndcg10_and_mrr_doc_get_a_paired_bootstrap_interval_over_queries(tmp_path, capsys):
    """Both continuous legs print a delta, a 95 % interval and the label.

    ``mrr_doc`` is read from ``scoring.rr_doc`` — that is the key
    ``eval/run_eval.py`` writes; there is no per-query ``mrr_doc`` key, so a
    loader that asked for one would raise on every real file.
    """
    prev_ndcg = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]        # mean 0.550
    cur_ndcg = [0.8, 0.75, 0.6, 0.55, 0.4, 0.35, 0.2, 0.15]     # mean 0.475
    prev_rr = [1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5]          # mean 0.625
    cur_rr = [0.8, 0.3, 0.45, 0.4, 0.85, 0.42, 0.3, 0.45]       # mean 0.49625
    a = _night("nightly-20260101", 1, [1] * 8, [1] * 8, prev_ndcg, prev_rr, CORPUS_A)
    b = _night("nightly-20260102", 2, [1] * 8, [1] * 8, cur_ndcg, cur_rr, CORPUS_A)
    d = _write(tmp_path, "nightly-a.json", a)
    _write(tmp_path, "nightly-b.json", b)
    n0, n1 = load_window(d)
    t = audit_transition(n0, n1)
    print_transition(t)
    out = capsys.readouterr().out

    assert [label for label, _ in CONT_LEGS] == ["ndcg10", "mrr_doc"]
    assert t.legs["ndcg10"]["delta"] == pytest.approx(-0.075)
    assert t.legs["mrr_doc"]["delta"] == pytest.approx(-0.12875)
    for label in ("ndcg10", "mrr_doc"):
        leg = t.legs[label]
        assert leg["exact"] is False
        # Every query moved down and the moves differ in size, so the interval
        # must sit entirely below zero and still have real width: the spread can
        # only come from resampling the 8 queries with replacement.
        assert leg["hi"] < 0.0 and leg["significant"] is True
        assert leg["hi"] - leg["lo"] > 0.0
        assert f"{label} " in out
    assert out.count("resampling approximation, not an exact test") == 2
    assert "seed" in out and "4000 replicates" in out
    assert "unit = query" in out


def test_the_bootstrap_interval_comes_from_resampling_not_from_a_formula():
    """Where every query moves by exactly the same amount, the resampling
    distribution is a point mass, so the interval must have width exactly zero.

    Any normal-theory interval (``delta +/- 1.96 * se``) would report a width
    here and call it uncertainty. This is the assertion that separates the two
    constructions, and it is why a *varying* dataset is the one that must be
    non-degenerate (next test).
    """
    prev = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    cur = [v - 0.1 for v in prev]           # every single query: -0.100
    leg = paired_bootstrap(prev, cur, reps=500, seed=20260919)
    assert leg["delta"] == pytest.approx(-0.1)
    assert leg["lo"] == pytest.approx(-0.1)
    assert leg["hi"] == pytest.approx(-0.1)
    assert leg["hi"] - leg["lo"] == pytest.approx(0.0, abs=1e-12)
    assert leg["significant"] is True, "a constant shift is a real shift"


def test_a_fixed_seed_gives_a_non_degenerate_and_reproducible_interval():
    prev = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    cur = [0.8, 0.75, 0.6, 0.55, 0.4, 0.35, 0.2, 0.15]
    one = paired_bootstrap(prev, cur, reps=2000, seed=20260919)
    two = paired_bootstrap(prev, cur, reps=2000, seed=20260919)
    assert (one["lo"], one["hi"]) == (two["lo"], two["hi"]), \
        "a fixed seed must reproduce the interval, or the report is not re-runnable"
    assert one["hi"] - one["lo"] > 0.02, "resampling with replacement must have spread"
    assert one["lo"] < one["delta"] < one["hi"]
    other = paired_bootstrap(prev, cur, reps=2000, seed=7)
    assert other["delta"] == pytest.approx(one["delta"]), \
        "the delta is the data, only the interval depends on the seed"


def test_the_bootstrap_refuses_unpaired_and_empty_input():
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="no paired queries"):
        paired_bootstrap([], [])


# ===========================================================================
# Clause 3 — the corpus-block diff is the drift term, and absent is not zero
# ===========================================================================

def test_every_transition_carries_the_corpus_diff_with_facts_edges_and_entities(tmp_path, capsys):
    a = _night("nightly-20260101", 1, [1] * 8, [1] * 8, [0.5] * 8, [0.5] * 8, CORPUS_A)
    b = _night("nightly-20260102", 2, [1] * 8, [1] * 8, [0.5] * 8, [0.5] * 8, CORPUS_B)
    d = _write(tmp_path, "nightly-a.json", a)
    _write(tmp_path, "nightly-b.json", b)
    n0, n1 = load_window(d)
    drift = corpus_diff(n0, n1)
    # Three fact terms and the document term. The document term is None HERE
    # because CORPUS_A/CORPUS_B are written in the pre-#1374 shape — a corpus
    # block with no `doc` key — and "no artifact recorded a vector count" is a
    # different answer from 0. #1374 is the item that split the two halves; the
    # fact terms keep their exact values so a change to the KG diff is still
    # caught by this test, and the fourth key is what stops the split being
    # silently reverted.
    assert drift == {"facts": 45906, "edges_active": 28340, "entities": 690,
                     DOC_DRIFT_KEY: None}
    t = audit_transition(n0, n1)
    print_transition(t)
    out = capsys.readouterr().out
    assert "corpus diff: facts +45906, edges_active +28340, entities +690" in out
    assert "corpus moved" in out
    # A drift term that can move is a drift term that can explain a delta: with
    # every leg concordant, the transition must be withheld on drift alone.
    assert t.rejected == [] and t.admissible is False
    assert "observation, not a verdict" in out


def test_a_missing_corpus_block_prints_drift_unknown_and_never_zero(tmp_path, capsys):
    """Three nights of the real series (09-04, 09-05, 09-06) predate the
    ``corpus`` block entirely. Reading them as zero drift would be the claim
    "the vault did not move" made from the absence of a measurement — the exact
    failure ``automod_regression.py:35-36`` forbids for a missing noise file
    ("cannot evaluate", never "no regression").
    """
    a = _night("nightly-20260101", 1, [1] * 8, [1] * 8, [0.5] * 8, [0.5] * 8, None)
    b = _night("nightly-20260102", 2, [1] * 8, [1] * 8, [0.5] * 8, [0.5] * 8, CORPUS_B)
    d = _write(tmp_path, "nightly-a.json", a)
    _write(tmp_path, "nightly-b.json", b)
    n0, n1 = load_window(d)
    assert corpus_diff(n0, n1) is None
    assert fmt_drift(None).startswith("drift unknown")
    t = audit_transition(n0, n1)
    assert t.drift_moved is None and t.admissible is False
    print_transition(t)
    drift_line = [ln for ln in capsys.readouterr().out.splitlines()
                  if "corpus diff" in ln][0]
    assert "drift unknown" in drift_line
    for key in ("facts", "edges_active", "entities"):
        assert key not in drift_line, f"{key} must not be reported as 0 when unmeasured"
    assert "+0" not in drift_line


def test_an_identical_corpus_is_reported_as_identical_not_as_a_missing_term():
    assert "corpus identical" in fmt_drift(
        {"facts": 0, "edges_active": 0, "entities": 0, DOC_DRIFT_KEY: 0})


# ===========================================================================
# #1374 — the DOCUMENT half of the corpus diff. `eval/run_eval.py` used to record
# only the fact half, so a pair could print `corpus identical` while the qmd
# daemon had re-embedded between the two runs and every `doc_hit` in the pair was
# scored against a different set of vectors. The four tests below are the four
# things that must now be true, in the order the acceptance clauses list them.
# ===========================================================================

#: Vector counts this section uses, from the item's own measurement: one daemon
#: process, up 54 860 s, no restart, 38 768 vectors at 14:11Z and 39 210 at
#: 14:41Z. The drift term for that pair is therefore +442, and the fact triple
#  behind it did not move at all.
VECTORS_THEN = 38_768
VECTORS_NOW = 39_210
VECTORS_DELTA = VECTORS_NOW - VECTORS_THEN


def _doc_corpus(vectors: int | None) -> dict:
    """A ``corpus`` block in the shape ``run_eval.py`` writes after #1374.

    ``vectors=None`` is the artifact's ``doc: null`` — the probe could not answer
    — which is a different shape from ``vectors=0`` (a daemon that answered and
    counted nothing) and a different shape again from the key being absent (an
    artifact written before the key existed). All three read as *unknown* or not,
    and that distinction is what these tests are for.
    """
    return {**CORPUS_A,
            DOC_KEY: None if vectors is None else {
                "vectors": vectors,
                "health_url": "http://localhost:8181/health",
                "code_root": str(ROOT)}}


def _rejecting_pair_with_vectors(tmp_path: Path, vectors_prev, vectors_cur):
    """Two nights whose ``entity_hit`` leg REJECTS at 95 % (six one-sided flips of
    eight, exact McNemar p = 0.03125) on a fact corpus that never moves.

    The leg has to reject or ``admissible`` is vacuously False and the test proves
    nothing: the clause is "cannot reach the ADMISSIBLE verdict", which is only a
    claim about a pair that would otherwise have reached it. Same scores both
    nights, same ``facts``/``edges_active``/``entities`` — only the recorded
    vector count differs.
    """
    a = _night("nightly-20260101", 1, [1] * 8, [1] * 8,
               [0.5] * 8, [0.5] * 8, _doc_corpus(vectors_prev))
    b = _night("nightly-20260102", 2, [1, 1, 0, 0, 0, 0, 0, 0], [1] * 8,
               [0.5] * 8, [0.5] * 8, _doc_corpus(vectors_cur))
    d = _write(tmp_path, "nightly-a.json", a)
    _write(tmp_path, "nightly-b.json", b)
    return load_window(d)


def test_vectors_moving_on_a_still_fact_triple_is_corpus_moved_and_never_admissible(tmp_path, capsys):
    """Clause 4: the pair the old code printed as `corpus identical`.

    Same 200,000 facts, same 4,000 active edges, same 23,600 entities — and 442
    more vectors in the index the document leg searched. Before #1374 that pair
    was `corpus identical`, and a rejecting leg on it was ADMISSIBLE.
    """
    n0, n1 = _rejecting_pair_with_vectors(tmp_path, VECTORS_THEN, VECTORS_NOW)
    drift = corpus_diff(n0, n1)
    assert {k: drift[k] for k in CORPUS_KEYS} == {
        "facts": 0, "edges_active": 0, "entities": 0}, "the fact half must stay still"
    assert drift[DOC_DRIFT_KEY] == VECTORS_DELTA

    t = audit_transition(n0, n1)
    assert t.rejected, "the fixture is only dispositive if a leg actually rejects"
    assert t.drift_moved is False, "the fact half is unchanged; that count is not the finding"
    assert t.doc_drift_moved is True
    assert t.admissible is False

    print_transition(t)
    out = capsys.readouterr().out
    assert f"{DOC_DRIFT_KEY} +{VECTORS_DELTA}" in out, out
    assert "corpus moved" in out and "corpus identical" not in out
    assert "WITHHELD" in out and "ADMISSIBLE" not in out


def test_a_rejection_on_a_pair_whose_both_halves_are_still_is_still_admissible(tmp_path):
    """The gate must tighten without breaking: same fixture, same recorded vector
    count on both nights, so nothing moved and the verdict is allowed.

    Without this the four clauses could be satisfied by a change that made
    `admissible` a constant False, which would be a stricter report and a worse
    instrument.
    """
    n0, n1 = _rejecting_pair_with_vectors(tmp_path, VECTORS_THEN, VECTORS_THEN)
    t = audit_transition(n0, n1)
    assert t.rejected and t.drift_moved is False and t.doc_drift_moved is False
    assert t.admissible is True


def test_an_unanswerable_doc_probe_prints_unknown_and_never_zero(tmp_path, capsys):
    """Clauses 3 and 5 together, at the reader's end.

    ``doc: null`` (a daemon that would not answer) and a key that is absent
    altogether (an artifact from before #1374) are the two ways a pair can have no
    document term. Both print `unknown`. Neither may print `+0`: on the shipped
    series 30 groups of baselines share an identical fact triple across hour-scale
    gaps, and a `doc_vectors +0` on those pairs would be the same fabricated
    "the document corpus did not move" that this item exists to stop.
    """
    # The artifact wrote `doc: null`: the probe ran and could not be answered.
    n0, n1 = _rejecting_pair_with_vectors(tmp_path, VECTORS_THEN, None)
    assert corpus_diff(n0, n1)[DOC_DRIFT_KEY] is None
    t = audit_transition(n0, n1)
    assert t.doc_drift_moved is None
    assert t.admissible is False, "an unrecorded doc half can account for the move"
    print_transition(t)
    out = capsys.readouterr().out
    assert f"{DOC_DRIFT_KEY} unknown" in out, out
    assert f"{DOC_DRIFT_KEY} +0" not in out, out
    assert "doc drift unknown, never read as 0" in out, out

    # A pre-#1374 artifact (no `doc` key at all) behaves the same way, and so does
    # a pair where BOTH nights are pre-#1374 shape — the case every historical
    # transition in `eval/baselines/` is.
    a = _night("nightly-20260101", 1, [1] * 8, [1] * 8, [0.5] * 8, [0.5] * 8, CORPUS_A)
    b = _night("nightly-20260102", 2, [1] * 8, [1] * 8, [0.5] * 8, [0.5] * 8, CORPUS_A)
    d = _write(tmp_path, "nightly-a.json", a)
    _write(tmp_path, "nightly-b.json", b)
    n0, n1 = load_window(d)
    drift_line = fmt_drift(corpus_diff(n0, n1))
    assert f"{DOC_DRIFT_KEY} unknown" in drift_line, drift_line
    assert f"{DOC_DRIFT_KEY} +0" not in drift_line, drift_line
    assert "doc side unknown" in drift_line, \
        "a still fact half must not be printed as a still corpus"




# ===========================================================================
# Clause 4 — the live series: 11 of 12 withheld, the 09-09 claim UNSUPPORTED,
#            and the required query count measured from observed discordance
# ===========================================================================

#: The window backlog #608's acceptance check is written against. Pinned by date
#: so the nightly run that lands tomorrow cannot move the number under the test.










#: A window in which every night shares the same 20 ``records[].id`` values, so
#: every one of its 4 transitions joins. The query set changed once in this
#: series — between 09-07 and 09-08 — and never again through 09-14.






# ── the process boundary the nightly job actually crosses ────────────────────
#
# Everything above calls ``main()`` in this process. That is the right shape for
# asserting statistics, and it is the wrong shape for the clause the nightly job
# depends on: the runner's Step 4 is a shell line, so what it sees is the
# interpreter's exit status, not main()'s return value. The bridge between them —
# ``if __name__ == "__main__": raise SystemExit(main())``, and argparse turning a
# ``--strict`` on the command line into ``args.strict`` — is a separate artifact,
# and dropping either one leaves the 30 in-process nodes green while the gate
# still cannot see a broken join. These two run the shipped module as the job
# runs it and assert on the process, not the function.













def test_the_sizing_moves_when_the_observed_discordance_moves():
    """A number that cannot respond to the data is decoration."""
    quiet = required_n(observed={"rate": 0.017})
    noisy = required_n(observed={"rate": 0.30})
    assert noisy["p_used"] == pytest.approx(0.30)
    assert quiet["p_used"] == pytest.approx(0.10), "the floor is the shift worth detecting"
    # Under the one-sided shape the script uses (q=1), the discordance rate IS the
    # effect size, so a bigger one needs fewer queries, and a smaller target needs
    # more of them.
    assert noisy["n_exact"] < quiet["n_exact"]
    assert required_n(detect=0.05)["n_exact"] > required_n(detect=0.10)["n_exact"]


# ── #1319 clause 4: the power line is computed from the WINDOW, not from 20 ───
#
# Until #1319 this block evaluated `power_exact(20, ...)` unconditionally and
# printed the literal `at n=20`, while the corpus it describes had been grown to
# 87 queries — so the audit's headline number described a window that no longer
# existed, and the Check "the power line for the current n is at or above 0.80"
# could not be satisfied by growing anything. `joined_paired_n(transitions)` is
# now threaded in, and the window has to be injectable for that to be testable:
# every real baseline on this box holds 20 records, because baselines with 78+
# records can only be written by a nightly that runs after the growth lands. A
# test that waited for live data would be asserting a post-landing observation,
# which is what the human half of this clause is for.

BIG_WINDOW_N = 90          # paired queries per transition in the injected window
BIG_WINDOW_POWER_FLOOR = 0.80


def _night_n(label: str, day: int, ids: tuple[str, ...], entity, doc,
             ndcg, rr, corpus) -> dict:
    """`_night` at an arbitrary query count; the record shape is identical.

    Exists because `IDS` has eight ids and clause 4 needs a window whose paired
    count clears the 78-query power floor.
    """
    doc_out = {
        "label": label,
        "ran_at": f"2026-01-{day:02d}T13:00:00+00:00",
        "records": [
            {"id": q, "query": f"query {q}",
             "scoring": {"entity_hit": bool(e), "doc_hit": bool(d),
                         "entity_recall": 1.0 if e else 0.0,
                         "doc_recall": 1.0 if d else 0.0,
                         "ndcg10": n, "rr_doc": r, "first_doc_rank": 1,
                         "fact_entity_recall": 1.0 if e else 0.0}}
            for q, e, d, n, r in zip(ids, entity, doc, ndcg, rr)
        ],
        "summary": {"overall": {"n_queries": len(ids)}},
    }
    if corpus is not None:
        doc_out["corpus"] = dict(corpus)
    return doc_out


def _write_paired_window(tmp_path: Path, n_ids: int) -> Path:
    """Three nights sharing `n_ids` query ids, with discordance kept rare.

    Three one-sided flips on night 2's entity leg and two on night 3's doc leg,
    so the pooled discordance over 2 transitions x 2 legs x ``n_ids`` records is
    5/360 at n=90 — below the 0.10 shift worth detecting, which is the shape the
    live series actually has and the reason the sizing uses p = max(observed,
    detect) = 0.10. Same ids in every night, because a transition that cannot
    join contributes no paired n and the median would then describe fewer nights
    than were written.
    """
    ids = tuple(f"g{i:03d}" for i in range(1, n_ids + 1))
    entity_1 = [1] * n_ids
    entity_2 = [0, 0, 0] + [1] * (n_ids - 3)
    doc_1 = [1] * n_ids
    doc_2 = [0, 0] + [1] * (n_ids - 2)
    d = tmp_path / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    nights = [("nightly-20260101", 1, entity_1, doc_1),
              ("nightly-20260102", 2, entity_2, doc_1),
              ("nightly-20260103", 3, entity_2, doc_2)]
    for label, day, entity, doc in nights:
        night = _night_n(label, day, ids, entity, doc,
                         [0.5] * n_ids, [0.5] * n_ids, CORPUS_A)
        (d / f"{label}.json").write_text(json.dumps(night), encoding="utf-8")
    return d


def _sizing_power_line(sizing: str) -> tuple[int, float]:
    """The (n, power) the block actually printed — parsed, not recomputed.

    Reading the number off the line is the point: asserting on
    ``power_exact(...)`` again in the test would pass even if the block printed a
    constant, which is the defect being closed.
    """
    m = re.search(r"power to detect a [\d.]+ paired change at n=(\d+): ([\d.]+)",
                  sizing)
    assert m, f"no power line in the sizing block:\n{sizing[:600]}"
    return int(m.group(1)), float(m.group(2))


def test_the_power_line_names_the_window_not_the_literal_20(tmp_path, capsys):
    """An injected 90-paired-query window reports power at 90, and it clears 0.80.

    Both directions are asserted, because the clause is "at the real n", not
    "≥ 0.80": an 8-id window in the same run must report n=8 and a power far
    below the floor. A block that printed a constant power would satisfy the
    ≥ 0.80 half on the big window and fail here.
    """
    big = _write_paired_window(tmp_path, BIG_WINDOW_N)
    assert main(["--baselines", str(big), "--reps", "200", "--no-claims"]) == 0
    sizing = capsys.readouterr().out.split("POWER / QUERY-COUNT SIZING")[1]
    n_big, power_big = _sizing_power_line(sizing)
    assert n_big == BIG_WINDOW_N, (
        f"the block printed n={n_big} for a window whose transitions each join "
        f"{BIG_WINDOW_N} query ids — the printed n must be the window's own "
        "paired count, not a constant lifted from the pre-#1319 corpus")
    assert power_big >= BIG_WINDOW_POWER_FLOOR, (
        f"power {power_big} at n={n_big}, under the {BIG_WINDOW_POWER_FLOOR} "
        "the sizing block itself says is required for a 0.10 paired change")
    assert "at n=20" not in sizing, "the pre-#1319 literal is back"
    assert "joined paired-query count" in sizing, (
        "the line must say which n it is quoting; an unqualified n is how the "
        "old line read as a corpus claim when it was a window claim")

    small = _write_paired_window(tmp_path / "small", len(IDS))
    assert main(["--baselines", str(small), "--reps", "200", "--no-claims"]) == 0
    sizing_small = capsys.readouterr().out.split("POWER / QUERY-COUNT SIZING")[1]
    n_small, power_small = _sizing_power_line(sizing_small)
    assert n_small == len(IDS), (
        f"the 8-id window printed n={n_small}; the block must follow the window "
        "down as well as up, or it is a constant with better arithmetic")
    assert power_small < 0.05, (
        f"power {power_small} at n={n_small} should be near zero — the 20-query "
        "series measured 0.011, which is the whole reason #1319 was filed")
    assert power_small < power_big, "power must respond to the window's size"


def test_a_window_that_pairs_nothing_reports_no_power_rather_than_inventing_n(tmp_path, capsys):
    """No joinable transition means no paired n, and the block must say so.

    ``required_n`` previously fell back to a literal 20 whenever there was no
    window to read; the replacement returns None. Printing `n=0` would be a
    number, and a number gets quoted into a nightly report.
    """
    d = tmp_path / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    for day, ids in ((1, tuple(f"a{i}" for i in range(8))),
                     (2, tuple(f"b{i}" for i in range(8)))):
        night = _night_n(f"nightly-2026010{day}", day, ids, [1] * 8, [1] * 8,
                         [0.5] * 8, [0.5] * 8, CORPUS_A)
        (d / f"nightly-2026010{day}.json").write_text(json.dumps(night),
                                                      encoding="utf-8")
    assert main(["--baselines", str(d), "--reps", "200", "--no-claims"]) == 0
    sizing = capsys.readouterr().out.split("POWER / QUERY-COUNT SIZING")[1]
    assert "no-verdict" in sizing, sizing[:600]
    assert "at n=0" not in sizing and "at n=20" not in sizing, sizing[:600]


def test_the_sizing_block_records_the_approved_re_base_point(tmp_path, capsys):
    """The closing line states the 2026-09-21 re-base, not "is a person's call".

    #608's third `human_clause` — whether to pay the re-basing cost — was approved
    on 2026-09-20 and discharged by the growth itself, so the sentence that told
    every reader the decision was still open became the thing that re-opens a
    closed decision. It is replaced by the boundary a later reader must not
    compare across. The corpus count quoted in that sentence is checked against
    the corpus file, because a number in prose that outlives the growth it
    describes is the same defect this item is about, one file over.
    """
    import yaml

    corpus = yaml.safe_load(
        (ROOT / "eval" / "vault_recall_queries.yaml").read_text())["queries"]
    d = _write_paired_window(tmp_path, BIG_WINDOW_N)
    assert main(["--baselines", str(d), "--reps", "200", "--no-claims"]) == 0
    sizing = capsys.readouterr().out.split("POWER / QUERY-COUNT SIZING")[1]
    assert "person's call" not in sizing, (
        "the pre-#1319 line is back: it tells the next reader the 80 %-power "
        "decision is unmade, which it is not")
    for required in ("#1319", "2026-09-21", "re-base"):
        assert required in sizing, f"{required!r} missing from the sizing block"
    assert f"{len(corpus)} gold queries" in sizing, (
        f"the line quotes a corpus size that is not the corpus on disk "
        f"({len(corpus)} queries)")




def test_the_audit_reads_the_live_tree_from_a_worktree(tmp_path, monkeypatch):
    """``eval/baselines/`` is gitignored, so a round worktree has an empty one.

    An audit that read its own checkout would find zero nights and report a
    flawless record. ``LLOYD_ROOT`` wins, else this checkout if it has the data,
    else the live checkout — the same resolution ``app/uptake.py:224`` uses.
    """
    monkeypatch.delenv("LLOYD_ROOT", raising=False)
    assert default_baselines_dir().is_dir()
    monkeypatch.setenv("LLOYD_ROOT", str(tmp_path))
    assert default_baselines_dir() == tmp_path / "eval" / "baselines"


# ===========================================================================
# Interval arithmetic that the report's prose quotes directly
# ===========================================================================

def test_wilson_interval_is_the_score_interval_not_the_normal_one():
    """At k=2, n=4 the naive Wald interval is 0.5 +/- 0.245 = [0.255, 0.745];
    Wilson's is [0.1499, 0.8501], and at k=0 the Wald interval is [0, 0] — an
    interval with zero width on a metric that has never been hit, which is the
    overclaim this item is about.
    """
    lo, hi = wilson_interval(2, 4)
    assert lo == pytest.approx(0.1499, abs=1e-3)
    assert hi == pytest.approx(0.8501, abs=1e-3)
    assert lo < 0.255 and hi > 0.745, "Wilson must be wider than Wald at n=4"
    lo0, hi0 = wilson_interval(0, 5)
    assert lo0 == 0.0
    assert 0.2 < hi0 < 0.7, "zero hits is consistent with a rate well above zero"
    nan_lo, nan_hi = wilson_interval(0, 0)
    assert math.isnan(nan_lo) and math.isnan(nan_hi), \
        "zero denominator is no-interval, not a widthless [0, 0]"




# ── the reporting contract the audit exists to enforce (clause 5) ─────────────
#
# The skill is the artifact the nightly runner actually loads, so a contract that
# lives only in the audit's docstring is a contract the runner never sees. That
# makes it the one clause here whose subject is not in this repo — and the two
# ways of getting at it are both wrong on their own:
#
#   * reading the working tree proves what the vault says *today*, which is not
#     what this commit shipped: a concurrent session's uncommitted edit moves the
#     target from under the assertion;
#   * skipping when it is absent is the failure this whole file argues against at
#     `test_the_real_series_...` above — a skipped acceptance check is a gate that
#     reads green because nobody looked.
#
# So: read the vault's committed `HEAD`, via git, and fail loudly when that is
# unreadable. Committed HEAD is the state a person can revert to and the state
# `automod_vault_land` wrote, so a pass means the contract is landed, not merely
# drafted. The residual limit — the vault is its own tree, so HEAD advances
# independently of this commit — is named in each failure message rather than
# implied away.

VAULT = Path.home() / "obsidian"
SKILL_RELPATH = "skills/retrieval-eval/SKILL.md"

# The exact instruction #608 retires. It is quoted here so that restoring it is
# caught, while the new text may still *discuss* 0.05 — it must, to explain why
# the threshold went away.
RETIRED_INSTRUCTION = re.compile(
    r"call out any metric that moved more than 0\.05", re.IGNORECASE)


@pytest.fixture(scope="module")
def skill_text() -> str:
    proc = subprocess.run(["git", "-C", str(VAULT), "show", f"HEAD:{SKILL_RELPATH}"],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        pytest.fail(
            f"clause 5 is unverifiable: `git -C {VAULT} show HEAD:{SKILL_RELPATH}` "
            f"failed ({proc.stderr.strip()[:200]}). The nightly reporting contract "
            f"is this item's fifth clause; it must not read green because the "
            f"artifact it grades could not be read.")
    return proc.stdout


def test_the_skill_points_the_nightly_runner_at_the_audit(skill_text):
    assert "scripts/eval_trend_stats.py" in skill_text, \
        "the nightly report step must run the paired test, not eyeball deltas"


def test_the_skill_forbids_a_verdict_the_paired_test_does_not_support(skill_text):
    """Clause 5: a decline/regression verdict needs the test to reject at 95 %
    AND the corpus diff not to account for the move; otherwise the interval plus
    the hedge sentence, with the move recorded as an observation.
    """
    for required in ("decline", "regression",
                     "not confident enough to decide",
                     "paired test rejects the null at 95",
                     "corpus diff does not account for the move"):
        assert required in skill_text, f"the contract clause {required!r} is missing"
    assert "observation" in skill_text, \
        "a move that fails the test is recorded as an observation, never a verdict"
    assert "Every cross-night delta in the report carries an interval" in skill_text


def test_the_skill_retires_the_bare_005_threshold_and_says_why(skill_text):
    """Pre-#608 Step 3 said "call out any metric that moved more than 0.05 in
    either direction and name the most likely cause" — an untested threshold one
    flipped query trips at n=20. Gone, with the reason stated where a future
    editor will read it rather than quietly deleted.
    """
    assert not RETIRED_INSTRUCTION.search(skill_text), \
        "the bare 0.05 call-out instruction is back; at n=20 it is the resolution limit"
    assert "resolution limit" in skill_text
    assert "Third night in a row" in skill_text, \
        "the 09-09 failure stacked three withheld observations into a verdict"


def test_the_skill_names_the_variance_source_it_must_not_cite(skill_text):
    """The 09-14 nightly justified a cross-night delta by citing
    ``automod_regression.py``'s repeat-run stdev — the smaller of two variance
    sources, exactly as triage recorded. The contract must name both numbers.
    """
    assert "automod_regression.py" in skill_text
    assert "stdev 0.0000" in skill_text
    assert "0.044" in skill_text, "must cite how much cross-night drift actually is"
    assert "smaller of the two variance sources" in skill_text


def test_the_skill_does_not_send_the_nightly_job_to_the_pinned_corpus_arm(skill_text):
    """``PinnedCorpus`` VACUUM-copies the qmd index and starts a second qmd daemon
    holding an embedding model on GPU 0 beside the primary engine. That is a
    person's call, so the skill must say the drift arm is NOT run by this job.
    """
    assert "evalpin" in skill_text
    assert "GPU 0" in skill_text
    assert "No pinned-corpus drift arm runs in this job" in skill_text




def test_the_audit_and_the_skill_hedge_in_the_same_words(skill_text):
    """One sentence, two places. If they drift, a report copied from the audit's
    ``verdict:`` line and a report written from the skill produce different hedges
    for the same withheld result, and no diff shows why.
    """
    contract = "\n".join(REPORTING_CONTRACT)
    hedge = "not confident enough to decide"
    assert hedge in contract and hedge in skill_text
    assert "paired test rejecting at {alpha}" in contract, \
        "the contract states its own threshold rather than assuming one"
    assert "record the move as an observation" in contract
    # markdown emphasis is not semantic, so compare on the stripped prose
    plain = skill_text.replace("**", "")
    assert "recorded as an observation, never a verdict" in plain, \
        "the skill's observation-not-verdict rule is the contract's, in other words"


# ===========================================================================
# The --strict contract, on nights this test builds.
#
# It used to be pinned only against the live 2026-09-04..09-17 series, which the
# 2026-09-22 deletion destroyed. The exit codes are a property of the CLI and of
# whether a window's records join, not of that series, so they are pinned here on
# a two-night fixture instead and keep their meaning on any machine.
# ===========================================================================

def _two_night_window(tmp_path: Path, *, joinable: bool) -> Path:
    """Two nights under a throwaway baselines dir, joinable by ids or not."""
    ids_a = ("q001", "q002", "q003")
    ids_b = ids_a if joinable else ("q004", "q005", "q006")
    d = tmp_path / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    for label, day, ids in (("nightly-20260101", 1, ids_a),
                            ("nightly-20260102", 2, ids_b)):
        night = _night_n(label, day, ids, [1, 1, 1], [1, 1, 1],
                         [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], CORPUS_A)
        (d / f"{label}.json").write_text(json.dumps(night), encoding="utf-8")
    return d


def test_strict_exits_non_zero_on_an_unjoinable_pair(tmp_path, capsys):
    """Without --strict an unjoinable pair reports and exits 0; with it, exit 1.

    Both arms run the same window, so the exit status is the only thing that moves
    — which is the whole contract: --strict changes what the caller sees, not what
    a human reads.
    """
    d = _two_night_window(tmp_path, joinable=False)
    argv = ["--baselines", str(d), "--no-claims"]
    assert main(argv) == 0, "without --strict an unjoinable pair still exits 0"
    capsys.readouterr()
    assert main(argv + ["--strict"]) == 1, "--strict must exit 1 on an unjoinable pair"


def test_strict_exits_zero_when_every_pair_in_the_window_joins(tmp_path, capsys):
    """The converse arm: --strict is not a blanket non-zero exit."""
    d = _two_night_window(tmp_path, joinable=True)
    assert main(["--baselines", str(d), "--no-claims", "--strict"]) == 0, (
        "an all-joinable window must exit 0 under --strict")
