"""eval/stats.py — the interval arithmetic #696 puts behind every eval number.

These are the numbers the item quoted, re-derived rather than restated: the
Wilson 95 % interval on a 20-query hit rate, what a one-query flip does to a
paired delta, and how much tighter the paired form is than the independent one.
Every assertion here is a claim a nightly report now makes about itself, so a
regression in this file is a wrong interval in someone's verdict.
"""
import ast
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import stats  # noqa: E402



# ── the module's own shape ───────────────────────────────────────────────────

def test_the_module_is_stdlib_only():
    """Clause 1 of #696. An install step between the eval and its interval is
    how a gate worktree ends up reporting a point estimate because the
    statistics package was not there."""
    tree = ast.parse((ROOT / "eval" / "stats.py").read_text())
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                tops.add(node.module.split(".")[0])
    assert tops <= {"math", "random", "__future__"}, tops


# ── Wilson: the interval on a rate ───────────────────────────────────────────

def test_wilson_ci_of_ten_hits_in_twenty_is_the_interval_the_item_quoted():
    lo, hi = stats.wilson_ci(10, 20)
    assert abs(lo - 0.299) < 0.002, lo
    assert abs(hi - 0.701) < 0.002, hi
    # 40 points of interval behind a point estimate that moves 5 points per
    # flipped query: this is the whole premise of the item, measured.
    assert hi - lo > 0.4


def test_wilson_ci_is_the_score_interval_and_not_the_wald_one():
    """Wald (p ± z·sd) is what a hand-rolled version produces, and it is the
    interval that reports a perfect score as a certainty."""
    lo, hi = stats.wilson_ci(10, 20)
    wald_lo, wald_hi = 0.280865, 0.719135          # 0.5 ± 1.96·sqrt(0.25/20)
    assert lo > wald_lo and hi < wald_hi, (lo, hi)
    perf_lo, perf_hi = stats.wilson_ci(20, 20)
    assert perf_hi == 1.0
    assert abs(perf_lo - 0.8389) < 0.002, perf_lo
    # Wald would hand back (1.0, 1.0) here — zero width on n=20.
    assert perf_lo < 0.9


def test_wilson_ci_of_an_empty_denominator_is_no_verdict():
    lo, hi = stats.wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_ci_refuses_counts_it_cannot_describe():
    with pytest.raises(ValueError):
        stats.wilson_ci(21, 20)
    with pytest.raises(ValueError):
        stats.wilson_ci(-1, 20)


# ── paired bootstrap: the interval on a delta ────────────────────────────────

def test_paired_bootstrap_ci_is_identical_across_two_calls_with_the_same_seed():
    a = [0.0, 0.5, 1.0, 0.25, 0.75] * 4
    b = [x + 0.05 for x in a]
    assert (stats.paired_bootstrap_ci(a, b, n_resamples=2000, seed=20260921)
            == stats.paired_bootstrap_ci(a, b, n_resamples=2000, seed=20260921))


def test_paired_bootstrap_ci_bounds_are_stable_across_seeds():
    """The item's own risk clause: the bootstrap must be checked against seed
    churn. On the one-flip case the bounds are seed-independent; only the
    resampling-approximation p moves, in the third decimal."""
    a = [0.5] * 20
    b = [0.5] * 19 + [0.6]
    runs = [stats.paired_bootstrap_ci(a, b, seed=s) for s in (20260921, 7, 1234, 99)]
    assert {round(r["lo"], 4) for r in runs} == {0.0}
    assert {round(r["hi"], 4) for r in runs} == {0.015}
    assert all(r["p"] > 0.5 for r in runs)


def test_paired_bootstrap_ci_returns_the_observed_mean_difference():
    """The point is the measured delta, not the bootstrap's resampled mean —
    a bootstrap that moved the estimate would not be an interval."""
    a = [0.0] * 20
    b = [0.2] * 20
    res = stats.paired_bootstrap_ci(a, b)
    assert res["diff"] == pytest.approx(0.2)
    assert res["n"] == 20
    assert res["n_resamples"] == 10000
    assert res["lo"] > 0.0 and res["significant"] is True


def test_a_difference_on_one_query_of_twenty_is_indistinguishable():
    """The worked case from the item's clause 1, and the shape of every nightly
    'regression' the 0.05 callout rule used to produce: 36 % of resamples never
    draw the query that moved, so the 2.5th percentile sits exactly at 0."""
    a = [0.5] * 20
    b = [0.5] * 19 + [0.6]
    res = stats.paired_bootstrap_ci(a, b)
    assert res["lo"] <= 0.0 <= res["hi"], (res["lo"], res["hi"])
    assert res["significant"] is False
    assert stats.ci_excludes_zero(res) is False
    assert res["diff"] == pytest.approx(0.005)


def test_the_paired_form_is_far_tighter_than_the_independent_one_on_shared_queries():
    """Why the gate may use the tight interval and the nightly report may not:
    pairing buys roughly an order of magnitude of width, and it buys it only
    when the two arms really share their records."""
    a = [0.0, 0.5, 1.0, 0.25, 0.75, 0.0, 0.5, 1.0, 0.5, 0.25,
         0.0, 0.5, 1.0, 0.75, 0.5, 0.0, 0.25, 0.5, 1.0, 0.5]
    b = [x + (0.1 if i % 3 == 0 else -0.05 if i % 3 == 1 else 0.0)
         for i, x in enumerate(a)]
    paired = stats.paired_bootstrap_ci(a, b)
    indep = stats.independent_bootstrap_ci(a, b)
    assert paired["diff"] == pytest.approx(indep["diff"])
    pw, iw = paired["hi"] - paired["lo"], indep["hi"] - indep["lo"]
    assert pw < iw
    assert iw > 2 * pw


def test_the_paired_bootstrap_refuses_vectors_it_cannot_pair():
    with pytest.raises(ValueError):
        stats.paired_bootstrap_ci([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        stats.paired_bootstrap_ci([], [])
    with pytest.raises(ValueError):
        stats.independent_bootstrap_ci([], [0.1])


# ── one-sample bootstrap: the non-binary metrics ─────────────────────────────

def test_bootstrap_mean_ci_brackets_a_rate_that_is_not_a_count_of_successes():
    vals = [0.0, 0.2, 0.4, 1.0] * 5
    res = stats.bootstrap_mean_ci(vals)
    assert res["point"] == pytest.approx(0.4)
    assert res["lo"] < res["point"] < res["hi"]
    assert res["lo"] > 0.0 and res["hi"] < 1.0


def test_bootstrap_mean_ci_reports_no_verdict_below_two_values():
    """A percentile bootstrap on one cluster reports the input back at zero
    width. That is false precision, and the caller prints 'no verdict'."""
    one = stats.bootstrap_mean_ci([0.5])
    assert one["point"] == 0.5
    assert one["lo"] is None and one["hi"] is None and one["n"] == 1
    none = stats.bootstrap_mean_ci([])
    assert none["point"] is None and none["n"] == 0


def test_bootstrap_mean_ci_is_seed_reproducible():
    vals = [0.1, 0.9, 0.5] * 7
    assert (stats.bootstrap_mean_ci(vals, n_resamples=500)
            == stats.bootstrap_mean_ci(vals, n_resamples=500))


# ── the decision predicate ───────────────────────────────────────────────────

def test_ci_excludes_zero_only_on_a_present_interval_wholly_off_zero():
    assert stats.ci_excludes_zero((0.05, 0.4)) is True
    assert stats.ci_excludes_zero((-0.4, -0.05)) is True
    assert stats.ci_excludes_zero((-0.05, 0.05)) is False
    assert stats.ci_excludes_zero((0.0, 0.05)) is False      # boundary, not a call
    assert stats.ci_excludes_zero(None) is False
    assert stats.ci_excludes_zero([None, None]) is False
    assert stats.ci_excludes_zero((float("nan"), float("nan"))) is False
    assert stats.ci_excludes_zero({"lo": None, "hi": None}) is False
    assert stats.ci_excludes_zero({"ci": [0.02, 0.3]}) is True
    assert stats.ci_excludes_zero({"ci": [None, None]}) is False
