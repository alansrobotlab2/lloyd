#!/usr/bin/env python3
"""Paired-test and drift audit for the nightly retrieval eval's cross-night claims.

Backlog #608. ``eval/run_eval.py`` (autonomy task #82, ``skills/retrieval-eval``)
has been writing night-over-night verdicts on the strength of a one-query
difference out of twenty. The run summary for 2026-09-09
(``autonomy-runs/82/run_82_20260909_130032.md``) reads::

    **Regression — third consecutive night of entity-side decline. Not noise.**

Its own records say otherwise: ``entity_hit_rate`` moved 0.55 -> 0.50, which is
one flipping query, and the corpus underneath it grew by 690 entities, 28 340
active edges and 45 906 facts in the same window. The eval's determinism is not
in question — ``workers/sources/automod_regression.py:14-18`` measured stdev
0.0000 on every quality metric over five repeat runs — but that is the *smaller*
of the two variance sources, and the nightly comparison is dominated by the
larger one, which the same file refuses to compare across days
(``:20-24``, "would therefore measure how much the vault moved, not what the
code change did"). Reporting a confidence interval off repeat-run variance
while the comparison carries corpus drift is the overconfident interval the AI
Engineer talk this item cites (~17-20 % coverage against a nominal 95 %) in
miniature.

What this script does, from the ``records[]`` already stored in
``eval/baselines/nightly-*.json`` — no eval is re-run, no LLM, CPU only:

* joins two nights **by ``records[].id``** and refuses to print a number when a
  query id exists in only one of them (the corpus of *questions* moving is not a
  paired comparison). This fires on real data: ``qwen35-users`` was replaced by
  ``qwen38-local-serving`` between 09-07 and 09-08.
* ``entity_hit`` / ``doc_hit``: exact two-sided McNemar on the discordant pairs
  (an exact test, not an approximation) plus the discordant counts, and a Wilson
  interval on the current night's rate.
* ``ndcg10`` / ``mrr_doc`` (from ``scoring.rr_doc``): a **paired bootstrap over
  queries**, always printed with the words "resampling approximation" because it
  is not an exact test.
* the ``corpus`` block diff (``facts``, ``edges_active``, ``entities``) beside
  every transition, printing ``drift unknown`` — never ``0`` — when a baseline
  has no ``corpus`` key at all. A drift term that reads ``0`` when the truth is
  unmeasured is the failure mode this whole file exists to remove.
* how many of the transitions' written verdicts would have been **withheld**, and
  the query count needed to detect a 0.10 paired change at 80 % power, computed
  from the discordance actually observed rather than from taste.
* a written claim re-scored against all of the above. Two shipped claims are
  built in; ``--claim`` adds more.

Exit status is 0 for an audit that found things — an audit that reports "every
verdict would have been withheld" has succeeded. ``--strict`` makes an
unjoinable pair a non-zero exit instead, for a caller that wants to treat it as
a failure of the caller's own data.

Measured on the shipped baselines of 2026-09-19 (2026-09-04 .. 2026-09-17, 12
transitions), which is the headline of backlog #608:

* **11 of 12 transitions have their verdict withheld** — no paired test rejects
  at 95 %. One of the 12 (09-07 -> 09-08) cannot even be audited: its query set
  changed.
* No binary leg ever rejects. The largest discordance on any single leg across
  the whole series is **two queries** (``doc_hit`` 09-08 -> 09-09, ``b=2 c=0``,
  exact McNemar p = 0.500); every other leg carries discordance of one or zero
  flips, which is p = 1.000.
* ``ndcg10`` rejects on **1 of 12** transitions (09-06 -> 09-07, delta +0.026)
  and its lower bound is a bare +0.0005, which is a boundary and not a verdict.
  The 09-08 -> 09-09 ``ndcg10`` interval is [-0.168, +0.031]: the same data
  covers a 17-point collapse and a 3-point gain.
* Power to detect a 0.10 paired change at n = 20 is **0.011**; the query count
  needed for 0.10 at 80 % power, from observed discordance, is **n = 78** — a
  lower bound, because every discordant pair in the series is one-sided.
* The corpus block moved under **9** of the 12 transitions and was unmeasurable
  on 3, so even a transition that cleared its test would face a second bar.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Run as `python scripts/eval_trend_stats.py`, sys.path[0] is `scripts/`, so the
# repo root has to be named before `app.` resolves. Tests import this module as
# `scripts.eval_trend_stats` with the root already on the path, which the guard
# below leaves alone.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The document half of the corpus diff (#1374). Imported, never restated: what an
# absent `corpus.doc` key MEANS is the entire content of this item, and a reader
# that re-implements the writer's rule is the defect recurring one file over.
from app.doc_corpus import DOC_DRIFT_KEY, doc_drift  # noqa: E402

ALPHA = 0.05
Z_975 = 1.959963984540054          # two-sided 95 %
Z_80 = 0.8416212335729143           # one-sided 80 % power
DETECT = 0.10                      # the paired change we want to be able to detect
POWER_TARGET = 0.80
BOOT_REPS = 4000                   # same replicate count the triage audit used
SEED = 20260919                    # fixed: the interval must be reproducible

#: label -> key inside ``records[].scoring``
BINARY_LEGS = (("entity_hit", "entity_hit"), ("doc_hit", "doc_hit"))
CONT_LEGS = (("ndcg10", "ndcg10"), ("mrr_doc", "rr_doc"))

#: The contract the nightly report must obey, printed with every audit so the
#: sentence travels with the numbers it governs. ``{alpha}`` is filled in at
#: print time. ``tests/test_eval_trend_stats.py`` asserts this text and the same
#: clauses in ``skills/retrieval-eval/SKILL.md`` agree.
REPORTING_CONTRACT = (
    "CONTRACT: a written decline/regression verdict requires a paired test "
    "rejecting at {alpha}",
    "AND a corpus diff that cannot account for the move. Otherwise: print the "
    "interval and say",
    '"not confident enough to decide", and record the move as an observation.',
)
#: The FACT half of the corpus diff. The document half rides in the same dict
#: under ``DOC_DRIFT_KEY`` (#1374) and is deliberately not folded into this
#: tuple: its value can be ``None`` meaning *unknown*, and a tuple that can hold
#: a non-number stops being a list of things you can subtract. Everything here
#: that iterates these keys is a statement about the KG, and says nothing about
#: the vectors the document leg actually searched — which is the blind spot
#: ``corpus_diff`` used to print ``corpus identical`` through.
CORPUS_KEYS = ("facts", "edges_active", "entities")

#: Nightly claims written into a run summary, re-scored by default. Each is a
#: verdict that was published without an interval; the audit re-runs the test
#: the sentence implicitly claims to have passed.
CLAIMS = (
    {
        "text": "Regression — third consecutive night of entity-side decline. Not noise.",
        "source": "autonomy-runs/82/run_82_20260909_130032.md",
        "metric": "entity_hit",
        "direction": "decline",
        "legs": (("nightly-20260906", "nightly-20260907"),
                 ("nightly-20260907", "nightly-20260908"),
                 ("nightly-20260908", "nightly-20260909")),
    },
    {
        "text": "doc_hit / doc_recall up — #504 landed, and it did exactly what it "
                "claimed. […] this movement is signal, not noise.",
        "source": "autonomy-runs/82/run_82_20260914_130023.md",
        "metric": "doc_hit",
        "direction": "improvement",
        "legs": (("nightly-20260913", "nightly-20260914"),),
    },
)


class UnjoinableQueries(Exception):
    """A query id exists in one night and not the other.

    There is no paired test over different question sets, and reporting one
    anyway is how a swapped query becomes a "regression". The caller must not
    print a number: the exception carries the ids so the report can name them.
    """

    def __init__(self, only_prev: list[str], only_cur: list[str]) -> None:
        self.only_prev = only_prev
        self.only_cur = only_cur
        super().__init__(
            f"{len(only_prev)} id(s) only in the earlier night {only_prev}; "
            f"{len(only_cur)} id(s) only in the later night {only_cur}"
        )


@dataclass
class Night:
    """One baseline file, reduced to what a paired comparison needs."""

    label: str
    path: Path
    ran_at: datetime | None
    corpus: dict | None
    #: query id -> scoring dict
    scores: dict[str, dict] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return sorted(self.scores)


def _night_key(path: Path, doc: dict) -> tuple[datetime | None, str]:
    """(ran_at, label) for ordering. Filename first, because two files can share
    a label and only the run timestamp tells them apart."""
    ran_at = None
    raw = doc.get("ran_at")
    if isinstance(raw, str):
        try:
            ran_at = datetime.fromisoformat(raw)
        except ValueError:
            ran_at = None
    if ran_at is None:
        stamp = re.search(r"(\d{8})-\d{6}", path.name)
        if stamp:
            try:
                ran_at = datetime.strptime(stamp.group(1), "%Y%m%d")
            except ValueError:
                ran_at = None
    label = doc.get("label") or re.sub(r"-\d{6}\.json$", "", path.name)
    return ran_at, str(label)


def load_night(path: Path) -> Night:
    doc = json.loads(Path(path).read_text())
    ran_at, label = _night_key(Path(path), doc)
    records = doc.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path.name}: no records[] — this is not a per-query baseline")
    scores: dict[str, dict] = {}
    for rec in records:
        rid = rec.get("id")
        if rid is None:
            raise ValueError(f"{path.name}: a record carries no id; cannot join")
        if rid in scores:
            raise ValueError(f"{path.name}: duplicate record id {rid!r}; cannot join")
        scoring = rec.get("scoring")
        if not isinstance(scoring, dict):
            raise ValueError(f"{path.name}: record {rid!r} has no scoring block")
        scores[rid] = scoring
    corpus = doc.get("corpus") if isinstance(doc.get("corpus"), dict) else None
    return Night(label=label, path=Path(path), ran_at=ran_at, corpus=corpus, scores=scores)


def load_window(baselines_dir: Path, since: str | None = None,
                until: str | None = None, pattern: str = "nightly-*.json") -> list[Night]:
    """Baselines in ``baselines_dir``, ordered by run time, cut to ``[since, until]``.

    The window is inclusive on both ends and compared on the *date*, because the
    nightly run stamps 06:00 local and a UTC instant would slice a day.
    """
    baselines_dir = Path(baselines_dir)
    nights = [load_night(p) for p in sorted(baselines_dir.glob(pattern))]
    if not nights:
        raise FileNotFoundError(f"no {pattern} under {baselines_dir}")
    nights.sort(key=lambda n: (n.ran_at is None, n.ran_at or datetime.min, n.label))
    lo = datetime.fromisoformat(since).date() if since else None
    hi = datetime.fromisoformat(until).date() if until else None
    out = []
    for n in nights:
        if n.ran_at is None:
            out.append(n)
            continue
        day = n.ran_at.date()
        if lo and day < lo:
            continue
        if hi and day > hi:
            continue
        out.append(n)
    return out


def join_ids(prev: Night, cur: Night) -> list[str]:
    """Query ids present in both nights, or ``UnjoinableQueries``.

    Deliberately stricter than an inner join: dropping the unmatched query in
    silence would let a *changed benchmark* be reported as a *changed score*.
    """
    only_prev = sorted(set(prev.ids) - set(cur.ids))
    only_cur = sorted(set(cur.ids) - set(prev.ids))
    if only_prev or only_cur:
        raise UnjoinableQueries(only_prev, only_cur)
    return prev.ids


def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def mcmemar_exact(prev_bits: list[int], cur_bits: list[int]) -> dict:
    """Exact two-sided McNemar on matched binary records.

    ``b`` = pairs that went 1 -> 0 (a loss), ``c`` = pairs that went 0 -> 1 (a
    gain). Under the null the discordant pairs are coin flips, so the exact
    p-value is ``2 * P(Bin(b + c, 0.5) <= min(b, c))``, capped at 1. No normal
    approximation: at ``b + c = 1`` the approximation is meaningless and the
    exact answer is 1.0, which is the whole point of this file.
    """
    if len(prev_bits) != len(cur_bits):
        raise ValueError("McNemar is paired: the two nights must have the same records")
    b = sum(1 for p, c in zip(prev_bits, cur_bits) if p == 1 and c == 0)
    c = sum(1 for p, c in zip(prev_bits, cur_bits) if p == 0 and c == 1)
    m = b + c
    n = len(prev_bits)
    delta = (c - b) / n if n else 0.0
    if m == 0:
        p = 1.0
    else:
        lo = min(b, c)
        p = min(1.0, 2.0 * sum(_binom_pmf(k, m) for k in range(lo + 1)))
    return {"b": b, "c": c, "m": m, "n": n, "delta": delta, "p": p,
            "significant": p < ALPHA, "exact": True}


def wilson_interval(k: int, n: int, z: float = Z_975) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, the night's own rate."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(prev_vals: list[float], cur_vals: list[float],
                     reps: int = BOOT_REPS, seed: int = SEED,
                     alpha: float = ALPHA) -> dict:
    """Paired bootstrap over queries for a continuous metric's delta.

    Resamples *query indices* with replacement and takes the difference of the
    two night means on the same resampled index, which is what makes it paired:
    the query, not the night, is the resampling unit. Percentile interval, and
    labelled everywhere it appears as a **resampling approximation, not an exact
    test** — a percentile bootstrap on 20 clusters covers roughly what it covers.
    """
    if len(prev_vals) != len(cur_vals):
        raise ValueError("bootstrap is paired: the two nights must have the same records")
    n = len(prev_vals)
    if n == 0:
        raise ValueError("no paired queries to resample")
    rng = random.Random(seed)
    diffs = sorted(
        sum((cur_vals[i] - prev_vals[i]) for i in
            (rng.randrange(n) for _ in range(n))) / n
        for _ in range(reps)
    )
    obs = sum(cur_vals) / n - sum(prev_vals) / n

    def pct(q: float) -> float:
        idx = min(reps - 1, max(0, int(q * reps)))
        return diffs[idx]

    lo, hi = pct(alpha / 2), pct(1 - alpha / 2)
    neg = sum(1 for d in diffs if d <= 0)
    pos = sum(1 for d in diffs if d >= 0)
    p = min(1.0, 2.0 * min((neg + 1) / (reps + 1), (pos + 1) / (reps + 1)))
    significant = (lo > 0.0) or (hi < 0.0)
    # A percentile sitting exactly on zero is a boundary case, not a rejection:
    # report it as marginal so nobody reads it as either a verdict or a nothing.
    marginal = (not significant) and obs != 0.0 and (lo == 0.0 or hi == 0.0)
    return {"delta": obs, "lo": lo, "hi": hi, "p": p, "reps": reps, "seed": seed,
            "significant": significant, "marginal": marginal, "exact": False}


def corpus_diff(prev: Night, cur: Night) -> dict | None:
    """The ``corpus`` block diff, or ``None`` when either night lacks the block.

    ``None`` is what makes the report print ``drift unknown``. A missing block is
    never zero drift — the same rule ``automod_regression.py:35-36`` applies to a
    missing noise file ("cannot evaluate", never "no regression").

    The fact keys are the difference of two counts that are always present in a
    post-#1129 artifact, so they diff to an int. The document key does not
    deserve that treatment: an artifact written before ``corpus.doc`` existed
    records no vector count, and treating that as 0 would manufacture exactly the
    thing it cannot see — a pair whose vectors clearly moved printed as
    ``corpus identical`` (#1374). So ``DOC_DRIFT_KEY`` carries an int when both
    nights recorded one and ``None`` when either did not, and every reader is
    allowed to ask about it separately.
    """
    if prev.corpus is None or cur.corpus is None:
        return None
    drift = {k: (cur.corpus.get(k) or 0) - (prev.corpus.get(k) or 0)
             for k in CORPUS_KEYS}
    drift[DOC_DRIFT_KEY] = doc_drift(prev.corpus, cur.corpus)
    return drift


@dataclass
class Transition:
    prev: Night
    cur: Night
    n: int
    legs: dict[str, dict]
    drift: dict | None
    joinable: bool = True
    error: str | None = None

    @property
    def drift_moved(self) -> bool | None:
        """True/False when the corpus block could be diffed, None when unknown.

        The FACT half only (#1374). Folding the document term in here would look
        like the more thorough choice and would quietly re-label the nine
        transitions this file's acceptance check counts — "9 moved / 3
        unmeasurable" is a statement about the KG, and it is quoted in
        ``skills/retrieval-eval/SKILL.md``. The document half is its own term,
        ``doc_drift_moved``, and ``admissible`` requires both.
        """
        if self.drift is None:
            return None
        return any(self.drift[k] != 0 for k in CORPUS_KEYS)

    @property
    def doc_drift_moved(self) -> bool | None:
        """The DOCUMENT half: True moved, False identical, None unknown.

        None is the common case and the honest one — every artifact written
        before ``corpus.doc`` existed, and every one whose daemon would not
        answer, lands here. It is NOT False: "not recorded" and "recorded and
        equal" are different facts, and the whole defect this term exists for is
        that the second used to be printed for the first.
        """
        if self.drift is None:
            return None
        term = self.drift.get(DOC_DRIFT_KEY)
        return None if term is None else term != 0

    @property
    def rejected(self) -> list[str]:
        return [k for k, v in self.legs.items() if v.get("significant")]

    @property
    def marginal(self) -> list[str]:
        return [k for k, v in self.legs.items() if v.get("marginal")]

    @property
    def withheld(self) -> bool:
        """The item's own contract sentence, applied: a verdict is allowed only when
        a paired test rejects the null at 95 %, so **anything that does not reject is
        withheld** — including a transition whose bootstrap bounds merely touch zero,
        which is a boundary, not a rejection.

        The *drift* term is a second, independent bar and is reported separately as
        ``admissible``: a transition can clear the test and still be un-adjudicable
        because the corpus moved under it.
        """
        return not self.rejected

    @property
    def admissible(self) -> bool:
        """A verdict is admissible only if a test rejected *and* drift cannot explain it.

        "Drift" is now both halves (#1374). An unknown document term blocks the
        verdict rather than passing it: the contract this class enforces is "a
        corpus diff that cannot account for the move", and a corpus whose
        document half was never recorded is exactly a diff that could account for
        it — the vectors behind every `doc_hit` in the pair may have differed
        while every fact count stood still. Tightening this costs nothing that
        existed: the shipped series has never had an admissible transition
        (`0 of 12`, pinned by a test below), so the change only ever removes a
        verdict a reader was about to over-read.
        """
        return (bool(self.rejected) and self.drift_moved is False
                and self.doc_drift_moved is False)


def audit_transition(prev: Night, cur: Night, reps: int = BOOT_REPS,
                     seed: int = SEED) -> Transition:
    try:
        ids = join_ids(prev, cur)
    except UnjoinableQueries as exc:
        return Transition(prev=prev, cur=cur, n=0, legs={}, drift=corpus_diff(prev, cur),
                          joinable=False, error=str(exc))
    legs: dict[str, dict] = {}
    for label, key in BINARY_LEGS:
        prev_bits = [_bit(prev.scores[i], key) for i in ids]
        cur_bits = [_bit(cur.scores[i], key) for i in ids]
        leg = mcmemar_exact(prev_bits, cur_bits)
        k = sum(cur_bits)
        leg["prev_rate"], leg["cur_rate"] = _mean(prev_bits), _mean(cur_bits)
        leg["wilson"] = wilson_interval(k, len(ids))
        legs[label] = leg
    for label, key in CONT_LEGS:
        legs[label] = paired_bootstrap(
            [_num(prev.scores[i], key) for i in ids],
            [_num(cur.scores[i], key) for i in ids],
            reps=reps, seed=seed)
    return Transition(prev=prev, cur=cur, n=len(ids), legs=legs,
                      drift=corpus_diff(prev, cur))


def _bit(scoring: dict, key: str) -> int:
    val = scoring.get(key)
    if val is None:
        raise ValueError(f"scoring has no {key!r}")
    return 1 if val else 0


def _num(scoring: dict, key: str) -> float:
    val = scoring.get(key)
    if val is None:
        raise ValueError(f"scoring has no {key!r}")
    return float(val)


def _mean(xs: list[int]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def observed_discordance(transitions: list[Transition]) -> dict:
    """Pooled per-query discordance across every audited transition, both binary legs."""
    disc = tot = 0
    max_disc = 0
    for t in transitions:
        if not t.joinable:
            continue
        for label, _ in BINARY_LEGS:
            leg = t.legs[label]
            disc += leg["m"]
            tot += leg["n"]
            max_disc = max(max_disc, leg["m"])
    return {"discordant": disc, "pairs": tot,
            "rate": (disc / tot) if tot else float("nan"),
            "max_per_leg": max_disc, "legs": sum(1 for t in transitions if t.joinable) * len(BINARY_LEGS)}


def power_exact(n: int, p_discord: float, q: float, detect: float,
                alpha: float = ALPHA) -> float:
    """Exact power of the McNemar test this script actually runs.

    Conditions on the number of discordant pairs ``m ~ Binomial(n, p_discord)``
    and then on the split ``b ~ Binomial(m, q)``, where ``q`` is the probability
    that a discordant pair is a gain. The net paired change is
    ``p_discord * (2q - 1)``, so a caller that wants ``detect`` must pass a
    ``q >= 0.5`` that makes ``p_discord * (2q - 1) == detect``. Rejection uses
    the same exact two-sided test as ``mcmemar_exact``, which is why the
    required ``n`` below is not the normal-approximation number.
    """
    if p_discord <= 0 or p_discord > 1:
        return 0.0
    total = 0.0
    for m in range(0, n + 1):
        pmf_m = _binom_pmf(m, n, p_discord)
        if pmf_m < 1e-15:
            continue
        if m == 0:
            total += pmf_m * 0.0
            continue
        rej = 0.0
        for b in range(0, m + 1):
            lo = min(b, m - b)
            p = min(1.0, 2.0 * sum(_binom_pmf(k, m) for k in range(lo + 1)))
            if p < alpha:
                rej += _binom_pmf(b, m, q)
        total += pmf_m * rej
    return total


def joined_paired_n(transitions: list[Transition]) -> int | None:
    """The paired-query count of the audited window: the median joined id count.

    ``t.n`` is the number of query ids one transition's two baseline files share,
    which is what every McNemar verdict in that transition was computed over — so
    it, not the size of the query file, is the ``n`` a power claim must name. The
    median rather than the maximum because the block describes the window most
    verdicts were reached on, not the largest pair in it. None when no transition
    joined, which the caller must print as no-verdict rather than as n = 0.
    """
    ns = sorted(t.n for t in transitions if t.joinable and t.n)
    return ns[len(ns) // 2] if ns else None


def required_n(detect: float = DETECT, power: float = POWER_TARGET,
               alpha: float = ALPHA, observed: dict | None = None,
               max_n: int = 2000, paired_n: int | None = None) -> dict:
    """Queries needed to detect a ``detect`` paired change at ``power``.

    Derived from measured discordance, not taste. Two shapes are reported:

    ``measured_shape`` — the discordance probability the baselines actually
    show. Every discordant pair observed in the 09-04 -> 09-17 series is
    one-sided (``b``, ``c`` = 1,0 or 0,1), so the measured shape is ``q = 1``
    and the smallest discordance rate that can express a ``detect`` shift is
    ``p = detect`` itself. This is the most favourable case the data allows, so
    the ``n`` it yields is a **lower bound**.

    ``normal_approximation`` — the textbook McNemar size
    ``(z_975 + z_80)^2 * p / detect^2``, kept as a cross-check because the
    exact power search above is the number this script stands behind.

    ``paired_n`` is the joined paired-query count of the window being audited
    (``joined_paired_n(transitions)``); ``power_at_paired_n`` is this script's
    own exact power evaluated at it, and is None when the window pairs nothing.
    """
    obs_rate = (observed or {}).get("rate")
    p_measured = max(detect, obs_rate) if obs_rate and obs_rate > 0 else detect
    n_exact = next((n for n in range(1, max_n + 1)
                    if power_exact(n, p_measured, 1.0, detect, alpha) >= power), None)
    # Power is evaluated at the paired-query count of the window being audited —
    # `paired_n`, the joined id count of its transitions — and NOT at a literal
    # 20. The literal was accurate only while the corpus and the joined baseline
    # files both held 20 ids; once #1319 grew the corpus the line kept printing
    # `at n=20: 0.011` no matter how many queries were scored, so the Check
    # "the power line for the current n is at or above 0.80" could not be
    # satisfied by growing anything. `paired_n=None` means no joinable
    # transition exists, and then this reports None — no verdict — rather than
    # inventing a denominator.
    power_at_paired_n = (power_exact(paired_n, p_measured, 1.0, detect, alpha)
                         if paired_n else None)
    return {"detect": detect, "alpha": alpha, "power_target": power,
            "observed_rate": obs_rate, "p_used": p_measured,
            "n_exact": n_exact,
            "paired_n": paired_n,
            "power_at_paired_n": power_at_paired_n,
            "normal_approximation": math.ceil(
                (Z_975 + Z_80) ** 2 * p_measured / (detect ** 2)),
            "lower_bound": True}


def fmt_drift(drift: dict | None) -> str:
    if drift is None:
        return "drift unknown (a baseline carries no `corpus` block; never read as 0)"
    doc = drift.get(DOC_DRIFT_KEY)
    # `unknown`, spelled out, on the same line as the fact terms. The bug this
    # line is written against is a print statement: 30 pairs in
    # `eval/baselines/` share an identical fact triple across hour-scale gaps and
    # used to read `corpus identical` while the daemon re-embedded through all of
    # them. A reader must not be able to mistake "no artifact recorded a vector
    # count" for "the vector count did not move", so the absence is a word here
    # rather than a `+0` that looks like every other term.
    doc_part = f"{DOC_DRIFT_KEY} unknown" if doc is None else f"{DOC_DRIFT_KEY} {doc:+d}"
    fact_moved = any(drift[k] for k in CORPUS_KEYS)
    if fact_moved or (doc not in (None, 0)):
        tail = "  -> corpus moved"
    elif doc is None:
        tail = "  -> corpus identical on the fact half, doc side unknown"
    else:
        tail = "  -> corpus identical"
    return ", ".join(f"{k} {drift[k]:+d}" for k in CORPUS_KEYS) + f", {doc_part}{tail}"


def print_transition(t: Transition, alpha: float = ALPHA) -> None:
    head = f"TRANSITION {t.prev.label} -> {t.cur.label}"
    if not t.joinable:
        print(f"\n{head}")
        print(f"  corpus diff: {fmt_drift(t.drift)}")
        print(f"  ERROR: queries not joinable by records[].id -> {t.error}")
        print("  no delta printed: a benchmark that changed questions is not a score that changed")
        print("  verdict: WITHHELD (cannot evaluate; not 'no change')")
        return
    print(f"\n{head}  (n={t.n} paired queries)")
    print(f"  corpus diff: {fmt_drift(t.drift)}")
    for label, _ in BINARY_LEGS:
        leg = t.legs[label]
        lo, hi = leg["wilson"]
        call = "rejects H0" if leg["significant"] else "does not reject H0"
        print(f"  {label:<10} delta {leg['delta']:+.3f}  "
              f"rates {leg['prev_rate']:.3f} -> {leg['cur_rate']:.3f}  "
              f"discordant b={leg['b']} c={leg['c']}  "
              f"exact McNemar p={leg['p']:.3f} ({call})  "
              f"Wilson 95% on {t.cur.label}: [{lo:.3f}, {hi:.3f}]")
    for label, _ in CONT_LEGS:
        leg = t.legs[label]
        if leg["significant"]:
            near = min(abs(leg["lo"]), abs(leg["hi"]))
            call = "excludes 0"
            if near < 0.005:
                call += f" by {near:.4f} — a boundary case, do not read it as a verdict"
        elif leg["marginal"]:
            call = "MARGINAL: a bound sits exactly on 0, which is not a rejection"
        else:
            call = "covers 0"
        print(f"  {label:<10} delta {leg['delta']:+.3f}  "
              f"paired bootstrap 95% interval [{leg['lo']:+.4f}, {leg['hi']:+.4f}] "
              f"(resampling approximation, not an exact test; {leg['reps']} replicates, "
              f"seed {leg['seed']}, unit = query) p={leg['p']:.3f} ({call})")
    if t.admissible:
        verdict = ("ADMISSIBLE: a paired test rejected and neither corpus half — "
                   "facts/edges/entities, nor the recorded vector count — moved")
    elif t.drift_moved is None:
        verdict = "WITHHELD: drift unknown, so no rejection can be attributed to the code"
    elif t.drift_moved:
        verdict = ("WITHHELD: corpus block moved, so the corpus diff can account for the "
                   "move — recorded as an observation, not a verdict")
    elif t.doc_drift_moved is None:
        # Ordered after the fact-half branches on purpose: this says something
        # narrower than "drift unknown", namely that the KG provably did not move
        # and the only thing that could explain the delta is the document corpus
        # this pair never recorded (#1374). On the shipped series every pair with
        # a still fact half lands here, because no artifact on disk predating this
        # term has a vector count to compare.
        verdict = ("WITHHELD: the document corpus is unrecorded, so a still fact half "
                   "cannot make the pair comparable — doc drift unknown, never read as 0")
    else:
        verdict = "WITHHELD: no paired test rejected at 95%"
    print(f"  verdict: {verdict}")


def print_totals(transitions: list[Transition], alpha: float = ALPHA) -> dict:
    total = len(transitions)
    audited = [t for t in transitions if t.joinable]
    withheld = sum(1 for t in transitions if t.withheld)
    rejected = [t for t in audited if t.rejected]
    marginal_only = [t for t in audited if t.marginal and not t.rejected]
    binary_sig = sum(1 for t in audited for k, _ in BINARY_LEGS if t.legs[k]["significant"])
    cont_sig = sum(1 for t in audited for k, _ in CONT_LEGS if t.legs[k]["significant"])
    admissible = sum(1 for t in transitions if t.admissible)
    drift_unknown = sum(1 for t in transitions if t.drift_moved is None)
    drift_moved = sum(1 for t in transitions if t.drift_moved is True)
    # The document half counted separately, and its unknown count printed beside
    # it, because on the day this landed `doc unknown` is 12 of 12 and that is the
    # finding — not a zero to be smoothed past. Once artifacts record a vector
    # count these three numbers start moving independently of the fact half above.
    doc_moved = sum(1 for t in transitions if t.doc_drift_moved is True)
    doc_identical = sum(1 for t in transitions if t.doc_drift_moved is False)
    doc_unknown = sum(1 for t in transitions if t.doc_drift_moved is None)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print(f"  transitions in window:      {total}")
    print(f"  transitions audited:        {len(audited)}"
          + (f"  ({total - len(audited)} unjoinable by records[].id)"
             if total != len(audited) else ""))
    print(f"  withheld: {withheld} of {total} transitions"
          "  (no paired test rejected the null at the stated confidence)")
    if marginal_only:
        print(f"    of those, {len(marginal_only)} had a bootstrap bound sitting exactly on"
              " zero — a boundary, not a rejection, so still withheld:"
              f" {[(t.prev.label, t.cur.label, t.marginal) for t in marginal_only]}")
    print(f"  transitions with a rejecting leg: {len(rejected)}"
          + (f"  {[(t.prev.label, t.cur.label, t.rejected) for t in rejected]}" if rejected else ""))
    print(f"  binary legs rejecting (of {len(audited) * len(BINARY_LEGS)}): {binary_sig}"
          "   <- exact McNemar")
    print(f"  continuous legs rejecting (of {len(audited) * len(CONT_LEGS)}): {cont_sig}"
          "   <- paired bootstrap, resampling")
    print(f"  corpus block moved: {drift_moved}; drift unknown: {drift_unknown}"
          "   <- fact half (facts / edges_active / entities)")
    # A separate line, not a clause on the one above: folding the two halves into
    # one number is how the audit could print a fact-half verdict and have it read
    # as the whole corpus. Until artifacts carry `corpus.doc` the unknown count is
    # every transition, which is the honest answer and must be visible as such.
    print(f"  {DOC_DRIFT_KEY}: moved {doc_moved}; identical {doc_identical}; "
          f"unknown {doc_unknown}"
          "   <- document half (qmd vectors the doc leg searched)")
    print(f"  verdicts admissible under the drift-controlled contract: {admissible} of {total}")
    return {"total": total, "audited": len(audited), "withheld": withheld,
            "rejected": len(rejected), "marginal_only": len(marginal_only),
            "binary_sig": binary_sig, "cont_sig": cont_sig,
            "admissible": admissible, "drift_unknown": drift_unknown,
            "drift_moved": drift_moved,
            "doc_moved": doc_moved, "doc_identical": doc_identical,
            "doc_unknown": doc_unknown}


def rescore_claim(claim: dict, by_label: dict[str, Night], reps: int, seed: int,
                  alpha: float = ALPHA) -> str:
    """Re-score a published verdict with the test that verdict implicitly claimed."""
    print(f"\nRE-SCORED CLAIM: \"{claim['text']}\"")
    print(f"  source: {claim['source']}  |  metric {claim['metric']}  |  "
          f"asserted direction {claim['direction']}")
    metric = claim["metric"]
    decline = claim["direction"].lower().startswith("decl")
    all_support = True
    for prev_label, cur_label in claim["legs"]:
        prev = by_label.get(prev_label)
        cur = by_label.get(cur_label)
        if prev is None or cur is None:
            print(f"  {prev_label} -> {cur_label}: no baseline on disk -> cannot evaluate")
            all_support = False
            continue
        t = audit_transition(prev, cur, reps=reps, seed=seed)
        if not t.joinable:
            print(f"  {prev_label} -> {cur_label}: ERROR unjoinable ({t.error}) -> cannot evaluate")
            all_support = False
            continue
        leg = t.legs[metric]
        moving = leg["delta"] < 0 if decline else leg["delta"] > 0
        direction_ok = moving and leg["significant"]
        supported = direction_ok and not (t.drift_moved is not False)
        detail = (f"delta {leg['delta']:+.3f}")
        if leg["exact"]:
            detail += (f", exact McNemar p={leg['p']:.3f}, discordant b={leg['b']} c={leg['c']}, "
                       f"Wilson 95% [{leg['wilson'][0]:.3f}, {leg['wilson'][1]:.3f}]")
        else:
            detail += (f", paired bootstrap 95% [{leg['lo']:+.3f}, {leg['hi']:+.3f}] "
                       f"(resampling approximation) p={leg['p']:.3f}")
        print(f"  {prev_label} -> {cur_label}: {detail}; corpus diff "
              f"{fmt_drift(t.drift)}")
        print(f"      -> {'supported' if supported else 'UNSUPPORTED at ' + format(alpha, '.0%')}")
        all_support = all_support and supported
    verdict = "SUPPORTED at the stated confidence" if all_support else \
        f"UNSUPPORTED at {alpha:.0%}"
    print(f"  CLAIM VERDICT: {verdict}")
    if not all_support:
        print("  The claim survives as an observation; it does not survive as a verdict.")
    return verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baselines", default=None,
                    help="directory of nightly-*.json (default: the live checkout's "
                         "eval/baselines, which is gitignored and so absent from a worktree)")
    ap.add_argument("--since", default=None, help="inclusive YYYY-MM-DD, compared on the run date")
    ap.add_argument("--days", type=int, default=0,
                    help="shorthand for --since: the last N days including today, "
                         "which is the form the retrieval-eval skill invokes")
    ap.add_argument("--until", default=None, help="inclusive YYYY-MM-DD")
    ap.add_argument("--pattern", default="nightly-*.json")
    ap.add_argument("--reps", type=int, default=BOOT_REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--detect", type=float, default=DETECT)
    ap.add_argument("--power", type=float, default=POWER_TARGET)
    ap.add_argument("--claim", action="append", default=[],
                    help="extra claim to re-score: METRIC:DIRECTION:PREV:CUR, e.g. "
                         "entity_hit:decline:nightly-20260908:nightly-20260909")
    ap.add_argument("--no-claims", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any pair of nights is unjoinable")
    args = ap.parse_args(argv)

    if args.days:
        if args.since:
            raise SystemExit("--days and --since are two ways of saying the same thing")
        args.since = (date.today() - timedelta(days=args.days - 1)).isoformat()
    baselines = Path(args.baselines) if args.baselines else default_baselines_dir()
    try:
        nights = load_window(baselines, args.since, args.until, args.pattern)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if len(nights) < 2:
        print(f"ERROR: need two nights in the window, found {len(nights)} "
              f"({' '.join(n.label for n in nights) or 'none'})", file=sys.stderr)
        return 2

    print(f"# eval trend audit — {baselines}")
    print(f"  nights: {len(nights)}  {nights[0].label} .. {nights[-1].label}  "
          f"(alpha={args.alpha:.2f}, {args.reps} bootstrap replicates, seed {args.seed})")
    series = _rate_series(nights, "entity_hit")
    print("  entity_hit_rate series: "
          + "  ".join(f"{lab}={val:.2f}" for lab, val in series))

    transitions = [audit_transition(prev, cur, reps=args.reps, seed=args.seed)
                   for prev, cur in zip(nights, nights[1:])]
    for t in transitions:
        print_transition(t, args.alpha)

    totals = print_totals(transitions, args.alpha)

    disc = observed_discordance(transitions)
    need = required_n(args.detect, args.power, args.alpha, disc,
                      paired_n=joined_paired_n(transitions))
    print("\nPOWER / QUERY-COUNT SIZING")
    print(f"  observed pooled discordance: {disc['discordant']}/{disc['pairs']} "
          f"paired records = {disc['rate']:.3f} over {disc['legs']} binary legs; "
          f"largest discordance on any one leg = {disc['max_per_leg']} query")
    pn = need["paired_n"]
    if pn and need["power_at_paired_n"] is not None:
        print(f"  power to detect a {args.detect:.2f} paired change at n={pn}: "
              f"{need['power_at_paired_n']:.3f} — under the most favourable discordance "
              f"shape the data allows (every pair one-sided, q=1). n here is the "
              f"joined paired-query count of this window, not the size of the query "
              f"file, so it moves when the corpus the baselines were scored on moves.")
    else:
        print(f"  power to detect a {args.detect:.2f} paired change: no-verdict — no "
              f"transition in this window joined a query-id set, so there is no "
              f"paired n to evaluate power at and none is invented.")
    print(f"  observed discordance {disc['rate']:.3f} is below the {args.detect:.2f} shift "
          f"worth detecting, so sizing uses p = max(observed, detect) = "
          f"{need['p_used']:.3f}: a {args.detect:.2f} net change cannot occur unless at "
          f"least {args.detect:.2f} of queries flip.")
    print(f"  required queries for {args.detect:.2f} at {args.power:.0%} power, "
          f"alpha {args.alpha:.2f}: n = {need['n_exact']} "
          f"(exact power of this script's own McNemar test; normal approximation "
          f"{need['normal_approximation']} as a cross-check)")
    if need["lower_bound"]:
        print("  that n is a LOWER BOUND: every discordant pair observed in this "
              "series is one-sided (b,c = 1,0 or 0,1), the most favourable shape "
              "available. Any two-sided discordance needs more queries.")
    print("  The query file was grown under #1319 on 2026-09-21 (20 -> 87 gold "
          "queries, the original 20 ids first and byte-identical), and that growth "
          "IS the approved re-base point: absolute values measured from the first "
          "night scored on the grown corpus are not comparable with the "
          "2026-09-04..2026-09-17 series, so do not read a trend across that "
          "boundary. Later on 2026-09-21 a gold audit re-pointed 38 labels that "
          "did not answer their query and dropped 6 queries the vault cannot "
          "answer (87 -> 81 gold queries, the original 20 texts untouched): a "
          "second re-base point, for the same reason. On 2026-09-24 #1354 added "
          "5 queries whose answers are the lloyd checkout's own architecture "
          "docs (81 -> 86 gold queries), which no collection indexes yet, so "
          "they score as document misses until one does: a third re-base "
          "point. And a fourth re-base point, which is NOT a corpus change: "
          "#1486 (`dbfde750`) landed semantic seeding of entity queries at "
          "2026-09-25 16:57 PDT, "
          "between the 09-25 and 09-26 nights, so the 2026-09-26 baseline is the "
          "first scored on the new seed definition. It moved the entity leg alone "
          "— entity_hit_rate 0.337 -> 0.500, entity_recall_avg 0.384 -> 0.579, "
          "anchorless_query_count 25 -> 16 — while the document leg held "
          "(doc_hit_rate 0.640 -> 0.628, doc_recall_avg 0.572 -> 0.572), and both "
          "nights reported matches_production_defaults: true because the "
          "conjunction compares six parsed args against six RECALL_* constants "
          "and a config-sourced seeding knob cannot be one of them (#1547). So do "
          "not compare an ENTITY-side value from before 2026-09-26 with one after "
          "it; the document leg does cross the boundary. A night's own artifact is "
          "the answer to which seeding it used, as `semantic_seeding` "
          "{enabled, k} — recorded from #1547 on, and absent on every earlier "
          "night, which is why the older entity-side nights are pre-re-base by "
          "absence rather than by a recorded off. The 80%-power decision "
          "that used to sit behind this line is no longer open — see the n "
          "printed above.")

    by_label = {n.label: n for n in nights}
    claims = [] if args.no_claims else list(CLAIMS) + [_parse_claim(c) for c in args.claim]
    if claims:
        print("\n" + "=" * 78)
        print("RE-SCORED VERDICTS")
        for claim in claims:
            rescore_claim(claim, by_label, args.reps, args.seed, args.alpha)

    print("\n" + "=" * 78)
    print("\n".join(REPORTING_CONTRACT).format(alpha=f"{args.alpha:.0%}"))
    print(f"  Under that contract {totals['withheld']} of {totals['total']} transitions "
          f"in this window would not have received a verdict.")

    if args.strict and totals["total"] != totals["audited"]:
        return 1
    return 0


def _parse_claim(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) != 4:
        raise SystemExit(f"--claim expects METRIC:DIRECTION:PREV:CUR, got {spec!r}")
    metric, direction, prev, cur = parts
    return {"text": f"(ad hoc) {metric} {direction} {prev} -> {cur}",
            "source": "--claim", "metric": metric, "direction": direction,
            "legs": ((prev, cur),)}


def _rate_series(nights: list[Night], key: str) -> list[tuple[str, float]]:
    out = []
    for n in nights:
        bits = [_bit(s, key) for s in n.scores.values()]
        out.append((n.label, _mean(bits)))
    return out


def default_baselines_dir() -> Path:
    """The runtime ``eval/baselines`` this audit is about.

    Baselines live under the data root (``app.paths.EVAL_BASELINES_DIR``), and an
    automod worktree's data root has an empty one, so an audit that read its own
    root would cheerfully compare zero nights. ``LLOYD_ROOT`` wins (read as a data
    root: ``<LLOYD_ROOT>/eval/baselines``), else this process's data root if it
    actually has nightly files, else production's (``production_data_root()``).
    """
    from app.paths import EVAL_BASELINES_DIR, production_data_root
    override = _env_root()
    if override:
        return override / "eval" / "baselines"
    if any(EVAL_BASELINES_DIR.glob("nightly-*.json")):
        return EVAL_BASELINES_DIR
    # Off the passwd home, never `$HOME`: a gate's `HOME=<round>/home` would name
    # the round's own empty root that sent us here.
    return production_data_root() / "eval" / "baselines"


def _env_root() -> Path | None:
    import os
    raw = os.environ.get("LLOYD_ROOT")
    return Path(raw).expanduser() if raw else None


if __name__ == "__main__":
    raise SystemExit(main())
