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

from scripts.eval_trend_stats import (  # noqa: E402
    ALPHA,
    CONT_LEGS,
    UnjoinableQueries,
    audit_transition,
    corpus_diff,
    default_baselines_dir,
    fmt_drift,
    join_ids,
    load_window,
    main,
    mcmemar_exact,
    observed_discordance,
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
    assert drift == {"facts": 45906, "edges_active": 28340, "entities": 690}
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
    assert "corpus identical" in fmt_drift({"facts": 0, "edges_active": 0, "entities": 0})


# ===========================================================================
# Clause 4 — the live series: 11 of 12 withheld, the 09-09 claim UNSUPPORTED,
#            and the required query count measured from observed discordance
# ===========================================================================

#: The window backlog #608's acceptance check is written against. Pinned by date
#: so the nightly run that lands tomorrow cannot move the number under the test.
AUDIT_WINDOW = ("2026-09-04", "2026-09-17")


def _live_baselines_in_window() -> list:
    d = default_baselines_dir()
    if not (d / "nightly-20260904-20260904-060219.json").exists():
        pytest.fail(
            f"{d} no longer holds the 2026-09-04 baseline. The acceptance check for "
            "#608 is a claim about those thirteen files; regenerating them is not "
            "possible (they are snapshots of a vault that no longer exists), so "
            "this is a real failure and not a skip — a skipped acceptance check is "
            "a gate that reads green because nobody looked.")
    return load_window(d, AUDIT_WINDOW[0], AUDIT_WINDOW[1])


def test_the_live_baselines_yield_eleven_of_twelve_transitions_withheld(capsys):
    """The acceptance check itself, run through the shipped script.

    Thirteen nights in 2026-09-04..2026-09-17 -> 12 transitions. Not one written
    verdict survives: both binary legs never reject (the largest discordance on
    any leg in the whole series is two queries, exact p = 0.500), and the single
    ``ndcg10`` leg that clears 95 % (09-06 -> 09-07, delta +0.026) has a lower
    bound of +0.0005 and sits on the corpus boundary. So 11 transitions are
    withheld for want of a rejection and the twelfth is the unjoinable
    09-07 -> 09-08 pair, which cannot evaluate at all.
    """
    nights = _live_baselines_in_window()
    assert len(nights) == 13, f"expected 13 nights in the window, got {len(nights)}"
    assert main(["--since", AUDIT_WINDOW[0], "--until", AUDIT_WINDOW[1]]) == 0
    out = capsys.readouterr().out
    assert "transitions in window:      12" in out
    assert "withheld: 11 of 12 transitions" in out
    assert "binary legs rejecting (of 22): 0" in out
    assert "verdicts admissible under the drift-controlled contract: 0 of 12" in out
    assert "nightly-20260908 -> nightly-20260909" in out


def test_the_real_09_07_to_09_08_query_swap_is_reported_as_unjoinable(capsys):
    """Clause 1's error path is not hypothetical: it fires on the shipped series."""
    assert main(["--since", "2026-09-07", "--until", "2026-09-08"]) == 0
    out = capsys.readouterr().out
    assert "queries not joinable by records[].id" in out
    assert "qwen35-users" in out and "qwen38-local-serving" in out
    assert "cannot evaluate" in out


#: A window in which every night shares the same 20 ``records[].id`` values, so
#: every one of its 4 transitions joins. The query set changed once in this
#: series — between 09-07 and 09-08 — and never again through 09-14.
JOINY_WINDOW = ("2026-09-09", "2026-09-14")


def test_strict_makes_an_unjoinable_pair_a_non_zero_exit(capsys):
    """Clause 1's *error*, at the level a caller can see it.

    An unjoinable pair prints ``ERROR … cannot evaluate`` and, by default, still
    exits 0: an audit that reports bad news has succeeded (``main``'s docstring,
    the same rule as the withheld count). That is right for a human reading the
    report and wrong for a caller that must not treat "no delta printed" as
    "nothing wrong". ``--strict`` is the difference between exit 0 and exit 1 on
    exactly this window — 13 nights, 12 transitions, one unjoinable pair
    (09-07 -> 09-08) — and nothing else about the run changes.
    """
    argv = ["--since", AUDIT_WINDOW[0], "--until", AUDIT_WINDOW[1], "--no-claims"]
    assert main(argv) == 0, "without --strict the same window exits 0"
    capsys.readouterr()
    assert main(argv + ["--strict"]) == 1, (
        "with --strict the window holding one unjoinable pair must exit non-zero")
    out = capsys.readouterr().out
    assert "queries not joinable by records[].id" in out, \
        "the non-zero exit must carry the report, not replace it"
    assert "transitions in window:      12" in out
    assert "transitions audited:        11  (1 unjoinable by records[].id)" in out


def test_strict_exits_zero_when_every_transition_in_the_window_joins(capsys):
    """The non-zero exit is about the data, not a default that always fires.

    2026-09-09..2026-09-14 is 5 nights / 4 transitions, all joinable: the one
    query-set change in this series happened earlier, between 09-07 and 09-08.
    Without this the flag could be a constant 1 and no test here would notice.
    """
    argv = ["--since", JOINY_WINDOW[0], "--until", JOINY_WINDOW[1],
            "--no-claims", "--strict"]
    assert main(argv) == 0, "an all-joinable window must exit 0 under --strict"
    out = capsys.readouterr().out
    assert "transitions in window:      4" in out
    assert "transitions audited:        4" in out
    assert "ERROR" not in out


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

def _run_module(*args: str) -> subprocess.CompletedProcess:
    """Run the module the way the skill's Step 4 does: ``cd <repo> && python -m``.

    ``-m`` resolves the ``scripts`` package off the CWD, so the checkout root is
    both the import root and where ``default_baselines_dir()`` lands without
    ``LLOYD_ROOT`` — the same two facts the nightly job relies on. The interpreter
    is the one running pytest, not a hardcoded path, so the gate's candidate venv
    is the thing under test.
    """
    return subprocess.run([sys.executable, "-m", "scripts.eval_trend_stats", *args],
                          capture_output=True, text=True, cwd=ROOT, timeout=300)


def test_the_module_exits_1_on_an_unjoinable_pair_under_strict():
    """Clause 1's error, at the level the nightly gate reads it.

    A fresh interpreter, the real command line, and the real shipped baselines:
    the 2026-09-04..09-17 window carries the unjoinable 09-07 -> 09-08 pair, so
    ``--strict`` must exit non-zero and name the ids on stdout, while the same
    invocation without ``--strict`` reports and exits 0. Both arms get identical
    arguments apart from the flag, and their stdout must be byte-identical:
    ``--strict`` changes the exit status and nothing a human reads, so a change
    that also altered the report would be caught here rather than in the morning.
    """
    proc = _run_module("--since", AUDIT_WINDOW[0], "--until", AUDIT_WINDOW[1],
                       "--strict")
    assert proc.returncode == 1, \
        f"--strict over an unjoinable pair must exit 1; got {proc.returncode}\n{proc.stderr[-800:]}"
    assert "queries not joinable by records[].id" in proc.stdout
    assert "qwen35-users" in proc.stdout, "the named ids are the report's evidence"

    lenient = _run_module("--since", AUDIT_WINDOW[0], "--until", AUDIT_WINDOW[1])
    assert lenient.returncode == 0, \
        f"without --strict the audit reports and succeeds; got {lenient.returncode}\n{lenient.stderr[-800:]}"
    assert "cannot evaluate" in lenient.stdout
    assert lenient.stdout == proc.stdout, \
        "--strict changes the exit status, never what is printed"


def test_the_module_exits_zero_under_strict_when_every_pair_in_the_window_joins():
    """The other arm of the CLI contract above, at the same level.

    2026-09-11 → 09-12 shares its 20 query ids on both sides, so the identical
    command line minus the one-night gap must exit 0. Both arms asserted as
    subprocesses is the point: argparse turned the flag on, and the pair is what
    proves ``--strict`` discriminates. Either arm alone still passes with a
    ``--strict`` that always fails, or never fails.
    """
    proc = _run_module("--since", "2026-09-11", "--until", "2026-09-12", "--strict")
    assert proc.returncode == 0, (
        f"--strict must pass where every pair joins; got {proc.returncode}\n"
        f"stdout tail: {proc.stdout[-600:]}\nstderr tail: {proc.stderr[-600:]}")
    assert "transitions in window:      1" in proc.stdout, \
        "the window must really hold a transition, or a zero exit proves nothing"
    assert "queries not joinable by records[].id" not in proc.stdout


def test_the_module_fails_before_printing_a_report_on_a_bad_invocation():
    """Two wrong invocations, both caught before any interval reaches stdout.

    ``--since`` / ``--until`` / ``--strict`` are the command line the nightly skill
    tells the runner to pass, so their parsing is shipped interface too. An unknown
    flag is argparse's own contract: exit 2, usage on stderr. A malformed date is
    not argparse's — it reaches ``load_window`` and raises — so this pins the part
    that matters to a gate: the run fails, names the offending value, and prints
    no report, rather than exiting 0 on an empty window that reads like
    "nothing moved". The traceback shape itself is recorded as a finding on #608,
    not asserted here, so this node does not lock in the worst of the two shapes.
    """
    unknown = _run_module("--mcmemar-please")
    assert unknown.returncode == 2, f"argparse rejects an unknown flag with 2, got {unknown.returncode}"
    assert "unrecognized arguments" in unknown.stderr

    bad = _run_module("--since", "not-a-date")
    assert bad.returncode != 0, "a malformed window may never exit 0"
    assert "not-a-date" in bad.stderr, "the failure names the value it could not parse"
    assert "transitions in window" not in bad.stdout, \
        "a broken invocation prints no report; an empty window would, and reads as 'nothing moved'"


def test_the_third_consecutive_entity_decline_claim_is_labelled_unsupported(capsys):
    """Test case #1 of backlog #608, answered in writing.

    ``autonomy-runs/82/run_82_20260909_130032.md`` published "**Regression —
    third consecutive night of entity-side decline. Not noise.**" The three legs
    it rests on are, in the audit's own output: 09-06 -> 09-07 ``entity_hit``
    delta +0.000 (drift unknown, so un-adjudicable), 09-07 -> 09-08 unjoinable,
    and 09-08 -> 09-09 delta -0.050 with ``b=1 c=0`` and exact McNemar p = 1.000
    over a corpus that grew 690 entities and 45 906 facts. The verdict is
    UNSUPPORTED at 5 %.
    """
    assert main(["--since", AUDIT_WINDOW[0], "--until", AUDIT_WINDOW[1]]) == 0
    out = capsys.readouterr().out
    block = out.split('RE-SCORED CLAIM: "Regression')[1].split("RE-SCORED CLAIM:")[0]
    assert "third consecutive night of entity-side decline" in block
    assert "metric entity_hit" in block
    assert "UNSUPPORTED at 5%" in block
    assert "CLAIM VERDICT: UNSUPPORTED at 5%" in block
    assert "b=1 c=0" in block, "the -0.05 delta is one flipping query and must read that way"
    assert "exact McNemar p=1.000" in block
    assert "drift unknown" in block, "09-06/09-07 carry no corpus block: never read as 0"
    assert "observation, not a verdict" in out


def test_required_query_count_is_measured_from_observed_discordance(capsys):
    """The sizing is arithmetic on the observed flip rate, not a taste number.

    Pooled discordance across the 22 binary legs of the window is 9 discordant
    pairs in 440 = 0.020 per paired record, which is *below* the 0.10 shift worth
    detecting — and a 0.10
    net change cannot happen unless at least 0.10 of queries flip, so the sizing
    uses p = max(observed, detect) = 0.10 with the one-sided shape the data
    actually shows (every discordant pair in this series is ``b,c`` = 1,0 or 0,1,
    i.e. ``q = 1``). Under that most-favourable shape the exact power of this
    script's own McNemar test reaches 80 % at **n = 78**, and every discordant
    pair observed so far is one-sided, so 78 is a lower bound. The live run also
    prints the power at n = 20, which is 0.011.
    """
    assert main(["--since", AUDIT_WINDOW[0], "--until", AUDIT_WINDOW[1]]) == 0
    out = capsys.readouterr().out
    sizing = out.split("POWER / QUERY-COUNT SIZING")[1]
    assert "required queries for 0.10 at 80% power" in sizing
    assert "n = 78" in sizing
    assert "power to detect a 0.10 paired change at n=20" in sizing
    assert "LOWER BOUND" in sizing
    transitions = [audit_transition(p, c) for p, c in
                   zip(_live_baselines_in_window(), _live_baselines_in_window()[1:])]
    disc = observed_discordance(transitions)
    assert disc["legs"] == 22, "11 joinable transitions x 2 binary legs"
    assert disc["rate"] == pytest.approx(9 / 440, abs=1e-9), \
        "9 discordant of 11 joinable transitions x 20 queries x 2 legs = 440"
    assert disc["max_per_leg"] == 2, "the largest discordance anywhere in the series"
    measured = required_n(observed=disc)
    assert measured["n_exact"] == 78
    assert measured["p_used"] == pytest.approx(0.10)
    assert measured["power_at_n20"] < 0.02
    assert measured["normal_approximation"] == 79, "cross-check within one query of the exact"
    assert measured["n_exact"] > 20, "the whole point: n=20 is not enough"


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



def test_the_audit_does_not_touch_the_query_set(tmp_path):
    """Growing ``eval/vault_recall_queries.yaml`` re-bases every absolute value
    the 09-04 -> 09-17 series is compared on, which is a person's call. The audit
    states the needed n and edits nothing.
    """
    queries = ROOT / "eval" / "vault_recall_queries.yaml"
    before = queries.read_bytes()
    mtime = queries.stat().st_mtime_ns
    assert main(["--since", AUDIT_WINDOW[0], "--until", AUDIT_WINDOW[1]]) == 0
    assert queries.read_bytes() == before
    assert queries.stat().st_mtime_ns == mtime


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


def test_the_contract_is_printed_with_the_audit_that_enforces_it(capsys):
    """The sentence a nightly runner must obey travels with the numbers, so a
    report copied from this output cannot accidentally drop the rule.
    """
    main(["--since", AUDIT_WINDOW[0], "--until", AUDIT_WINDOW[1]])
    out = capsys.readouterr().out
    assert "not confident enough to decide" in out
    assert "paired test rejecting at 5%" in out
    assert "corpus diff that cannot account for the move" in out


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


def test_the_skill_quotes_drift_figures_the_baselines_support(skill_text):
    """Every drift number in the skill must be the number the data prints.

    The skill's Step 4 tells the nightly runner to put a concrete `corpus` diff in
    its sample sentence, and a worked example there is the one thing a runner
    copies verbatim. The first version of that text quoted `facts` going
    251,685 → 351,365 (+39.6 %) for 09-08 → 09-09 and a sample `drift facts
    +587`; neither number is in any baseline on disk (251,685 is the 09-09 value,
    the *later* night, and no transition in the series moves `facts` by 587). A
    prose number nobody can re-measure is how an invented drift term gets into a
    real report, so the figures are re-derived here from the two baseline files.
    """
    prev, cur = load_window(default_baselines_dir(), "2026-09-08", "2026-09-09")
    assert (prev.label, cur.label) == ("nightly-20260908", "nightly-20260909")
    facts_prev = prev.corpus["facts"]
    facts_cur = cur.corpus["facts"]
    edges_prev = prev.corpus["edges_active"]
    edges_cur = cur.corpus["edges_active"]
    delta = facts_cur - facts_prev
    assert (facts_prev, facts_cur) == (205779, 251685), \
        "the two baselines the skill quotes changed under this test"

    # thousands separators allowed, trailing punctuation not: [\d,]+ would eat the
    # comma of "32,373, so" and quietly int() it back to the right value anyway
    num = r"\d+(?:,\d{3})*"
    quoted = re.search(rf"`facts` went ({num}) → ({num}) "
                       rf"\(\+({num}), \+([\d.]+) %\)", skill_text)
    assert quoted, "Step 4 must quote the 09-08 -> 09-09 facts drift"
    assert [int(g.replace(",", "")) for g in quoted.groups()[:3]] == \
        [facts_prev, facts_cur, delta], \
        "the skill's facts pair and its delta must be the baselines' own numbers"
    assert float(quoted.group(4)) == pytest.approx(100.0 * delta / facts_prev, abs=0.05), \
        "the percent has to be the ratio of those two numbers, not a remembered one"

    edges = re.search(rf"`edges_active` ({num}) → ({num})", skill_text)
    assert edges, "Step 4 names the edge leg too, and it must carry real values"
    assert [int(g.replace(",", "")) for g in edges.groups()] == [edges_prev, edges_cur]

    sample = re.search(rf"drift facts \+({num})", skill_text)
    assert sample, "the sample sentence carries a drift figure"
    assert int(sample.group(1).replace(",", "")) == corpus_diff(prev, cur)["facts"], \
        ("the sentence a nightly runner copies must show the drift term the audit "
         "actually prints for the transition it is drawn from")


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
