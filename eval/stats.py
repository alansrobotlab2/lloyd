#!/usr/bin/env python3
"""The interval arithmetic every eval number is read against (#696).

Stdlib only, on purpose: this is the one eval module that must be importable
from a bare `python3` in a gate worktree, a nightly job, or the backtest, with
no store, no daemon and no third-party statistics package. A dependency here
would put the interval behind an install.

Why it exists as its own module rather than as more functions inside
`scripts/eval_trend_stats.py`: the interval is now needed by three callers with
different lifetimes — the nightly trend audit (which owns the exact tests), the
eval runner (which has to write the interval into the artifact the audit later
re-reads), and the backtest (which re-decides old verdicts offline). Two copies
of `wilson` and a bootstrap is how two sides come to disagree about whether a
number moved, which is the same failure the fact-leg guard is imported rather
than restated for (`eval/run_eval.py:106-112`).

The two forms are not interchangeable and the caller must pick one knowingly:

* **paired** — resamples *queries*, taking both arms' value on the same resampled
  index, so the query is the unit and the between-arm difference is what varies.
  Valid only when both arms measured the same records against the same system
  state; it is the tight interval, and on an unpaired comparison it is a lie.
* **independent** — resamples each arm's own indices separately. Wider, and the
  honest form for anything whose corpus moved underneath it, which includes
  every night-over-night retrieval comparison (`facts` moved under 9 of the 12
  transitions the 09-19 audit covered).

Every resampling function takes a `seed` and defaults it to a fixed constant:
an interval that cannot be re-produced is not evidence, it is a rumour.
"""
from __future__ import annotations

import math
import random

__all__ = [
    "Z_975",
    "SEED",
    "N_RESAMPLES",
    "MIN_BOOT_N",
    "wilson_ci",
    "paired_bootstrap_ci",
    "independent_bootstrap_ci",
    "bootstrap_mean_ci",
    "ci_excludes_zero",
]

#: two-sided 95 %
Z_975 = 1.959963984540054
#: Fixed so two runs of the same audit print the same interval. Not a secret and
#: not tuned — a bootstrap that moves between readings cannot be cited twice.
SEED = 20260921
#: 10,000 replicates: the item's own step 1, and cheap (well under a second for
#: a 20-query vector), so nothing here is trading accuracy for wall clock.
N_RESAMPLES = 10000
#: A percentile bootstrap on ONE cluster reports the input back as an interval
#: of zero width. That is not an interval, so n=1 gets no verdict instead of a
#: degenerate one; the caller prints "no verdict".
MIN_BOOT_N = 2


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Empirical CDF inverse at `q` over an already-sorted replicate vector."""
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(q * n)))
    return sorted_vals[idx]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion `k/n` at `z` (1.96 = 95 %).

    Chosen over the textbook Wald interval because Wald is wrong exactly where
    this eval lives: at ``k = 0`` or ``k = n`` it collapses to width zero and
    reports a perfect score as a certainty. Wilson on ``k = 20, n = 20`` is
    ``[0.8389, 1.0]``, which still says an unmeasured query could have failed.

    ``wilson_ci(10, 20)`` is ``(0.2993, 0.7007)`` — the 31-point width behind a
    20-query hit rate, i.e. one flipped query (5 points) is well inside the
    interval the same measurement carries.

    ``(nan, nan)`` when ``n <= 0``: the rate is undefined, so the interval is
    undefined, and reporting ``(0, 1)`` or ``(0, 0)`` would be an invented
    bound. Callers who serialise must map nan to a null first — `NaN` is not
    valid JSON and a baseline artifact has to stay parseable.
    """
    if k < 0 or n < 0 or k > n:
        raise ValueError(f"wilson_ci needs 0 <= k <= n, got k={k} n={n}")
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _check_pair(a: list[float], b: list[float], what: str) -> int:
    if len(a) != len(b):
        raise ValueError(f"{what} is paired: the two vectors must have the same records")
    if not a:
        raise ValueError(f"{what}: no records to resample")
    return len(a)


def paired_bootstrap_ci(a: list[float], b: list[float], *,
                        n_resamples: int = N_RESAMPLES,
                        seed: int = SEED,
                        alpha: float = 0.05) -> dict:
    """Paired percentile bootstrap CI for the mean of ``b - a`` over shared records.

    The resampling unit is the *record index*: one draw picks n indices and both
    arms are read at those same indices, which is what makes the design paired
    and what makes it far tighter than two independent intervals. Valid only for
    two arms that measured the same records under the same system state.

    Returns ``{"diff", "lo", "hi", "p", "n", "n_resamples", "seed",
    "significant", "marginal"}`` with `diff` the OBSERVED mean difference (the
    bootstrap estimates its interval, not the estimate itself).

    `p` is the sign-based bootstrap p-value from the same replicate vector, and
    lives here rather than in the caller for the module's whole reason to exist:
    the nightly audit needs a p and an interval off one resample draw, and a
    second implementation of either is how the two disagree about a verdict.

    A difference that shows up on ONE query of 20 lands in an interval that
    contains zero: about 36 % of resamples never draw that query at all, so the
    2.5th percentile sits at 0.0 and the honest reading is "indistinguishable".
    That is the result the item was written for, not a rounding artefact.

    `marginal` marks a bound sitting exactly on zero with a non-zero observed
    delta — a boundary, never a rejection, and never a nothing either.
    """
    n = _check_pair(a, b, "paired bootstrap")
    rng = random.Random(seed)
    diffs = sorted(
        sum((b[i] - a[i]) for i in (rng.randrange(n) for _ in range(n))) / n
        for _ in range(n_resamples)
    )
    obs = (sum(b) - sum(a)) / n
    lo = _percentile(diffs, alpha / 2)
    hi = _percentile(diffs, 1.0 - alpha / 2)
    neg = sum(1 for d in diffs if d <= 0)
    pos = sum(1 for d in diffs if d >= 0)
    p = min(1.0, 2.0 * min((neg + 1) / (n_resamples + 1), (pos + 1) / (n_resamples + 1)))
    significant = (lo > 0.0) or (hi < 0.0)
    return {"diff": obs, "lo": lo, "hi": hi, "p": p, "n": n,
            "n_resamples": n_resamples, "seed": seed,
            "significant": significant,
            "marginal": (not significant) and obs != 0.0 and (lo == 0.0 or hi == 0.0)}


def independent_bootstrap_ci(a: list[float], b: list[float], *,
                             n_resamples: int = N_RESAMPLES,
                             seed: int = SEED,
                             alpha: float = 0.05) -> dict:
    """Independent (unpaired) percentile bootstrap CI for the difference of means.

    Each arm's indices are resampled from its own vector, so nothing is shared
    between them. The right form whenever the two measurements did NOT share a
    system state — two nightlies of a retrieval eval included, since the vault
    under them moves (`facts` +45,906 between 09-08 and 09-09) and a paired
    interval would credit that movement to the code.

    Same vector lengths are not required; the arms may have different n. An empty
    arm raises, because a difference against nothing has no interval.
    """
    if not a or not b:
        raise ValueError("independent bootstrap needs two non-empty vectors")
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    # One rng, and both legs drawn per replicate, so the whole function is
    # reproducible off the single seed rather than off the call order.
    diffs = sorted(
        (sum(b[i] for i in (rng.randrange(nb) for _ in range(nb))) / nb)
        - (sum(a[i] for i in (rng.randrange(na) for _ in range(na))) / na)
        for _ in range(n_resamples)
    )
    obs = sum(b) / nb - sum(a) / na
    lo = _percentile(diffs, alpha / 2)
    hi = _percentile(diffs, 1.0 - alpha / 2)
    neg = sum(1 for d in diffs if d <= 0)
    pos = sum(1 for d in diffs if d >= 0)
    p = min(1.0, 2.0 * min((neg + 1) / (n_resamples + 1), (pos + 1) / (n_resamples + 1)))
    significant = (lo > 0.0) or (hi < 0.0)
    return {"diff": obs, "lo": lo, "hi": hi, "p": p, "n_a": na, "n_b": nb,
            "n_resamples": n_resamples, "seed": seed,
            "significant": significant,
            "marginal": (not significant) and obs != 0.0 and (lo == 0.0 or hi == 0.0)}


def bootstrap_mean_ci(values: list[float], *,
                      n_resamples: int = N_RESAMPLES,
                      seed: int = SEED,
                      alpha: float = 0.05) -> dict:
    """One-sample percentile bootstrap CI for the mean of `values`.

    For a metric that is not a hit/miss rate — `mrr_doc`, `ndcg10`, the three
    recalls — Wilson does not apply (it is a binomial interval, and an MRR is not
    a count of successes out of trials), so the run's own uncertainty comes from
    resampling its per-query values.

    Returns ``{"point", "lo", "hi", "n", ...}``. With fewer than `MIN_BOOT_N`
    values there is no spread to resample and `lo`/`hi` come back `None` rather
    than equal to `point`: a zero-width interval on one observation is false
    precision, and the caller prints "no verdict".
    """
    n = len(values)
    if n == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0,
                "n_resamples": n_resamples, "seed": seed}
    point = sum(values) / n
    if n < MIN_BOOT_N:
        return {"point": point, "lo": None, "hi": None, "n": n,
                "n_resamples": n_resamples, "seed": seed}
    rng = random.Random(seed)
    means = sorted(
        sum(values[i] for i in (rng.randrange(n) for _ in range(n))) / n
        for _ in range(n_resamples)
    )
    return {"point": point, "lo": _percentile(means, alpha / 2),
            "hi": _percentile(means, 1.0 - alpha / 2), "n": n,
            "n_resamples": n_resamples, "seed": seed}


def ci_excludes_zero(ci) -> bool:
    """True only when a two-element interval is present and wholly off zero.

    Accepts a ``(lo, hi)`` pair, a list, or a dict carrying `lo`/`hi`. A missing
    or half-open interval is NOT a rejection: "no interval" must never read as
    either "significant" or "indistinguishable" — it reads as no verdict, which
    is the same rule `automod_regression` applies to a missing noise file.
    """
    if ci is None:
        return False
    if isinstance(ci, dict):
        lo, hi = ci.get("lo"), ci.get("hi")
        if lo is None and isinstance(ci.get("ci"), (list, tuple)):
            pair = ci["ci"]
            lo, hi = (pair[0], pair[1]) if pair and len(pair) == 2 else (None, None)
    else:
        lo, hi = (ci[0], ci[1]) if ci and len(ci) == 2 else (None, None)
    if lo is None or hi is None:
        return False
    if any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)):
        return False
    return lo > 0.0 or hi < 0.0
