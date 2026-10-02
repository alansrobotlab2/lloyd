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
* the ``labels_sha256`` gold-label fingerprint (#1637). Two nights whose
  ``records[].id`` sets are identical but whose fingerprints differ are reported
  **INCOMPARABLE**, with both values named and no delta, p-value or interval: the
  id join cannot see a gold answer being re-pointed, and ``9b028e9`` re-pointed 22
  of them while every id stayed put. Equal fingerprints, or either night carrying
  no fingerprint (every artifact written before #1637), report exactly as before.
* how many of the transitions' written verdicts would have been **withheld**, and
  the query count needed to detect a 0.10 paired change at 80 % power, computed
  from the discordance actually observed rather than from taste.
* a written claim re-scored against all of the above. Two shipped claims are
  built in; ``--claim`` adds more.

Exit status is 0 for an audit that found things — an audit that reports "every
verdict would have been withheld" has succeeded. ``--strict`` makes a pair this
audit could not adjudicate a non-zero exit instead, whether the ids did not join or
the gold labels moved under an id set that did (#1637), for a caller that wants to
treat it as a failure of the caller's own data.

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

#: Key of the gold-label fingerprint (#1637). The writer is `eval/run_eval.py`, where
#: the same string is `GOLD_LABELS_KEY`; the two modules share no import, because this
#: one has to run in a gate worktree with no retrieval stack, so
#: `tests/test_eval_label_agreement.py` asserts the two spellings are one string
#: instead of importing it. A key the writer wrote and the reader looked for under a
#: different name would read as a legacy artifact forever, silently.
GOLD_LABELS_KEY = "labels_sha256"

#: Key of the query-text witness (#1852), written by `eval/run_eval.py` as
#: `QUESTIONS_KEY`; same no-shared-import contract as the key above. The stamp is the
#: summary; the evidence this tool actually compares is each record's own `query`,
#: which every artifact on disk already carries, so the affected ids are derived from
#: the artifacts and no list of exempt ids exists anywhere.
QUESTIONS_KEY = "questions_sha256"

#: The five scored rates whose denominator #1663 re-defined on 2026-09-28, each keyed to
#: the leg whose gold-bearing subset that rate now divides over. `eval/run_eval.py:911`
#: (`GOLD_BEARING_LEGS`) is the same map at write time; it is restated here rather than
#: imported because a trend tool has to be re-runnable over artifacts the current
#: scorer no longer wrote, and a `run_eval` import drags the engine's whole config with
#: it. #1822.
GOLD_BEARING_LEGS = {
    "entity_hit_rate": "entities",
    "entity_hit_rate_retrieval_carried": "entities",
    "doc_hit_rate": "docs",
    "mrr_doc": "docs",
    "ndcg10": "docs",
}
#: The `records[].expected` keys that say whether one query carries gold for one leg.
#: A non-empty list is presence, which is what #1663 divides over — so the denominator
#: is a property of the record and never of a number the artifact publishes about it.
#: `summary.overall.ci95[<metric>].n` is deliberately NOT read: on every pre-re-base
#: artifact it is the all-records n, so it says "nothing moved" precisely where the
#: re-base happened.
GOLD_EXPECTED_KEYS = ("entities", "docs")

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
    #: The gold-label fingerprint this run recorded (#1637), or None for "this
    #: artifact predates the field". Absent and null are the same state here: the
    #: writer emits null only for a run that scored no gold at all, and either way
    #: there is nothing to compare, so the guard stays silent.
    labels_sha256: str | None = None
    #: leg -> query ids whose own `expected` block carries gold for that leg, or None
    #: when NO record in the artifact carried an `expected` block at all. Retained here
    #: rather than recomputed downstream because it is the only record-level evidence
    #: of what denominator a rate divided over, and `Night` used to keep `scores` alone
    #: and drop it at load time (#1822). None and a leg of size 0 are different states:
    #: the first is "this artifact says nothing about gold", the second is "no query
    #: carries gold", and only the second is a measurement.
    gold_ids: dict[str, set[str]] | None = None
    #: query id -> the question text that record was asked (#1852). Only ids whose
    #: record carried a string `query`; an id absent here says nothing about its text.
    questions: dict[str, str] = field(default_factory=dict)
    #: The run's own `questions_sha256` stamp, or None for an artifact predating it.
    questions_sha256: str | None = None

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
    #: leg -> query ids whose own `expected` block carries gold for that leg, or None
    #: until some record proves this artifact records gold at all. Built here because
    #: `scores` keeps the `scoring` block alone: once this loop ends, the per-record
    #: gold that decided each rate's denominator under #1663 is no longer reachable,
    #: and a trend tool that cannot see it cannot see the re-base at all (#1822).
    gold_ids: dict[str, set[str]] | None = None
    questions: dict[str, str] = {}
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
        if isinstance(rec.get("query"), str):
            questions[rid] = rec["query"]
        expected = rec.get("expected")
        if isinstance(expected, dict):
            if gold_ids is None:
                gold_ids = {leg: set() for leg in GOLD_EXPECTED_KEYS}
            for leg in GOLD_EXPECTED_KEYS:
                vals = expected.get(leg)
                if isinstance(vals, (list, tuple)) and len(vals) > 0:
                    gold_ids[leg].add(rid)
    corpus = doc.get("corpus") if isinstance(doc.get("corpus"), dict) else None
    # Absent, null and empty all mean "this artifact records no gold fingerprint";
    # only a non-empty string is a stamp, and a guard that read `""` as a value would
    # refuse every pair of artifacts written before #1637.
    gold = doc.get(GOLD_LABELS_KEY)
    return Night(label=label, path=Path(path), ran_at=ran_at, corpus=corpus,
                 scores=scores,
                 labels_sha256=gold if isinstance(gold, str) and gold else None,
                 gold_ids=gold_ids, questions=questions,
                 questions_sha256=(stamp if isinstance(stamp := doc.get(QUESTIONS_KEY), str)
                                   and stamp else None))


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
    #: Why this pair may not be compared *even though its ids join* (#1637): the two
    #: recorded gold-label fingerprints, both named. None is every pair the guard has
    #: no finding about — an equal pair, and a pair where either night carries no
    #: fingerprint at all, which is every artifact written before #1637.
    incomparable: str | None = None
    #: Which of the five scored rates divide over a different gold subset on each side
    #: of this pair (#1663, via `definition_break`), or None. An ANNOTATION and never a
    #: refusal: it is not read by `auditable`, the pair stays joinable, and the
    #: statistics print as they always did — #1663's owed-check ruled on 2026-09-29 that
    #: the pre-re-base nights stay in the published window annotated rather than
    #: dropped, which is the opposite of what `incomparable` above does.
    definition_break: str | None = None
    #: The query ids whose QUESTION TEXT differs between the two nights while the id
    #: set is unchanged (#1852), as a printed sentence, or None. Like
    #: `definition_break` it is an annotation and never a refusal — the pair joins,
    #: the statistics print — but unlike it, it bars `admissible`: a paired test
    #: across a re-worded question scores the edit, so no rejection on such a pair
    #: can be credited to the system.
    question_break: str | None = None

    @property
    def auditable(self) -> bool:
        """Ids join AND the gold did not move: the pairs a number may be printed for.

        ``joinable`` alone was sufficient before #1637 because a re-pointed gold
        answer left no trace an id join could see. It is not sufficient now, and
        every count of transitions this audit actually adjudicated reads this, so a
        gold-moved pair can never be tallied as an audited pair that merely found
        nothing.
        """
        return self.joinable and self.incomparable is None

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
                and self.doc_drift_moved is False
                and self.question_break is None)


def _gold_moved(prev: Night, cur: Night, ids: list[str]) -> str | None:
    """The #1637 guard: identical id set, differing gold fingerprints -> the sentence
    that refuses the pair, naming both fingerprints. None means compare.

    It fires on one condition only, and every other shape returns None so the pair is
    printed exactly as this script printed it before the guard existed:

      * either night carries no fingerprint. That is a legacy artifact — every file
        written before #1637 — or a run that scored no gold. Annotating every legacy
        pair would bury the one line per window that matters, and a missing stamp is
        not evidence that anything moved;
      * the fingerprints are equal, which is what two consecutive nights normally
        look like, since the corpus of questions moves far more often than a gold
        answer does;
      * the id sets differ, in which case ``join_ids`` has already refused the pair
        for a reason a reader can act on.

    ``ids`` is passed only to say how many queries the two different benchmarks
    share, which is the number a reader needs to judge how much of the window the
    refusal costs.
    """
    if not prev.labels_sha256 or not cur.labels_sha256:
        return None
    if prev.labels_sha256 == cur.labels_sha256:
        return None
    if sorted(prev.scores) != sorted(cur.scores):
        return None
    return (f"gold labels moved under an unchanged query-id set of {len(ids)}: "
            f"{GOLD_LABELS_KEY}={prev.labels_sha256} in {prev.label}, "
            f"{GOLD_LABELS_KEY}={cur.labels_sha256} in {cur.label}")


def definition_break(prev: Night, cur: Night) -> str | None:
    """The five scored rates whose gold-bearing subset differs between two nights.

    #1663 (`efcee660`) re-defined each of the five on 2026-09-28: a rate now divides
    over the queries carrying gold for its own leg, where before it divided over every
    record. Nothing in the id join can see that — an unchanged corpus keeps every id —
    so the only thing a pair can be audited against is each record's own gold, which is
    what the loader retains in ``Night.gold_ids``.

    Deliberately NOT ``summary.overall.ci95[<metric>].n``: on a pre-re-base artifact
    that field is the all-records n, so a pair straddling the re-base would show two
    numbers that never moved while the subset every rate divides over had. Absence of an
    ``expected`` block is silence, not a zero denominator, for the same reason the #1637
    guard treats a missing fingerprint as nothing to compare: annotating every legacy
    pair would spend the one line per window that means something.
    """
    if prev.gold_ids is None or cur.gold_ids is None:
        return None
    moved = []
    for metric, leg in GOLD_BEARING_LEGS.items():
        n_prev, n_cur = len(prev.gold_ids[leg]), len(cur.gold_ids[leg])
        if n_prev != n_cur:
            moved.append(f"{metric} {n_prev} -> {n_cur}")
    if not moved:
        return None
    return (", ".join(moved) + " — the queries carrying gold for the named leg(s) "
            "differ between these nights, so each rate above is a rate over a "
            "different subset (#1663 denominator re-base, annotated not dropped)")


def question_break(prev: Night, cur: Night) -> str | None:
    """Query text that moved under an unchanged query-id set (#1852), or None.

    The gold corpus is reserved by prose (`eval/counterfactual.py`), which nothing
    enforces, and every other witness misses a re-worded question: the id is
    unchanged so the join is silent, `labels_sha256` hashes the gold and not the
    question, and the corpus block counts facts. The evidence is each record's own
    `query`, compared id by id.

    Silent in every other shape, so a pair prints as it did before this existed:
    differing id sets (the join already refuses those, louder); an id whose record
    carries no text on either side — a missing witness is not evidence that anything
    moved (the #1637 rule), so a legacy artifact annotates nothing; and equal texts.
    When the per-record text cannot name the ids but both runs stamped
    `questions_sha256` and the stamps differ, the line says so without ids rather
    than staying quiet.
    """
    if sorted(prev.scores) != sorted(cur.scores):
        return None
    moved = sorted(i for i in prev.questions
                   if i in cur.questions and prev.questions[i] != cur.questions[i])
    if moved:
        return (f"query text moved under an unchanged query-id set: {', '.join(moved)} "
                f"({len(moved)} of {len(prev.scores)})")
    if (prev.questions_sha256 and cur.questions_sha256
            and prev.questions_sha256 != cur.questions_sha256
            and set(prev.questions) == set(cur.questions)):
        return ("query text moved under an unchanged query-id set: ids not derivable, "
                f"{QUESTIONS_KEY}={prev.questions_sha256} in {prev.label}, "
                f"{cur.questions_sha256} in {cur.label}")
    return None


def question_witness(prev: Night, cur: Night) -> int:
    """How many joined ids carry question text on BOTH sides — what the check above
    could actually compare. 0 means it was blind on this pair, not that it was clean."""
    return sum(1 for i in prev.questions if i in cur.questions)


def audit_transition(prev: Night, cur: Night, reps: int = BOOT_REPS,
                     seed: int = SEED) -> Transition:
    try:
        ids = join_ids(prev, cur)
    except UnjoinableQueries as exc:
        return Transition(prev=prev, cur=cur, n=0, legs={}, drift=corpus_diff(prev, cur),
                          joinable=False, error=str(exc))
    # Before any statistic: a pair whose gold answers differ is two benchmarks, and
    # a McNemar pair over two benchmarks is not a score that changed. Checked after
    # the id join because an unjoinable pair already says the louder thing.
    gold = _gold_moved(prev, cur, ids)
    if gold:
        return Transition(prev=prev, cur=cur, n=len(ids), legs={},
                          drift=corpus_diff(prev, cur), incomparable=gold)
    legs: dict[str, dict] = {}
    for label, key in BINARY_LEGS:
        prev_bits = [_bit(prev.scores[i], key) for i in ids]
        cur_bits = [_bit(cur.scores[i], key) for i in ids]
        leg = mcmemar_exact(prev_bits, cur_bits)
        # #2060: `rate_n` is the divisor read back off the very lists the rate and the
        # interval are computed from, so the count a print names beside them cannot
        # drift away from the arithmetic behind them. It is deliberately NOT the
        # artifact's headline denominator: `summary.overall.ci95[<metric>].n` divides
        # over the queries carrying gold for the leg (#1663), which on
        # nightly-20261002 is 43/66 = 0.652 published against 43/81 = 0.531 here — a
        # 0.12 regression that exists only in the difference between the two
        # denominators. Whether the audit's rate should move onto that gold subset is
        # the ruling #2060 reserves to a person, and taking it would leave `delta`,
        # which is `(c - b)/n` over the joined population, printing beside a rate on
        # another one; what is not open is a bare figure in the headline's column.
        leg["rate_n"] = len(cur_bits)
        leg["prev_rate"], leg["cur_rate"] = _mean(prev_bits), _mean(cur_bits)
        leg["wilson"] = wilson_interval(sum(cur_bits), leg["rate_n"])
        legs[label] = leg
    for label, key in CONT_LEGS:
        legs[label] = paired_bootstrap(
            [_num(prev.scores[i], key) for i in ids],
            [_num(cur.scores[i], key) for i in ids],
            reps=reps, seed=seed)
    return Transition(prev=prev, cur=cur, n=len(ids), legs=legs,
                      drift=corpus_diff(prev, cur),
                      definition_break=definition_break(prev, cur),
                      question_break=question_break(prev, cur))


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
    """Pooled per-query discordance across every audited transition, both binary legs.

    ``auditable``, not ``joinable``: a pair the gold-label guard refused (#1637) has
    no legs at all, so reading its ``legs[label]`` would be a KeyError, and its
    id count is not evidence about discordance in the first place.
    """
    disc = tot = 0
    max_disc = 0
    for t in transitions:
        if not t.auditable:
            continue
        for label, _ in BINARY_LEGS:
            leg = t.legs[label]
            disc += leg["m"]
            tot += leg["n"]
            max_disc = max(max_disc, leg["m"])
    return {"discordant": disc, "pairs": tot,
            "rate": (disc / tot) if tot else float("nan"),
            "max_per_leg": max_disc, "legs": sum(1 for t in transitions if t.auditable) * len(BINARY_LEGS)}


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

    ``auditable`` here too (#1637): a pair whose gold labels moved shares ids, but
    no verdict was computed over them, so its id count belongs in the denominator of
    a power claim exactly as much as an unjoinable pair's does — which is not at all.
    """
    ns = sorted(t.n for t in transitions if t.auditable and t.n)
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
    if t.incomparable:
        # Same shape as the unjoinable branch, because it is the same kind of fact: a
        # benchmark whose gold answers moved is not a score that moved. The numbers
        # are not printed at all rather than printed with a caveat, because the
        # failure this exists for is a quiet pair — identical hit patterns either side
        # of a re-label, p = 1.000, and "no change" is the one sentence that would be
        # wrong here. Both fingerprints are named so a reader can find the edit.
        print(f"\n{head}  (n={t.n} query ids shared, gold labels differ)")
        print(f"  corpus diff: {fmt_drift(t.drift)}")
        print(f"  INCOMPARABLE: {t.incomparable}")
        print("  no delta printed: two nights judged against different gold answers are two")
        print("  benchmarks; a paired test across them scores the re-label, not the system")
        print("  verdict: WITHHELD (cannot evaluate; not 'no change')")
        return
    print(f"\n{head}  (n={t.n} paired queries)")
    print(f"  corpus diff: {fmt_drift(t.drift)}")
    if t.definition_break:
        # Annotation, printed before the numbers and never instead of them. The join
        # cannot see a denominator change on an unchanged corpus, so without this line a
        # night scored under #1663's per-leg gold subset reads as a two-percent move.
        # The pair still counts as audited and its verdict still prints: #1663's
        # owed-check ruled that pre-re-base nights stay in the window annotated, not
        # dropped and not refused the way an incomparable pair is.
        print(f"  DEFINITION BREAK: {t.definition_break}")
    if t.question_break:
        # Same shape (#1852): printed before the numbers, never instead of them. A
        # re-worded question keeps its id, so nothing else on this page can show it.
        print(f"  QUESTION BREAK: {t.question_break}")
    for label, _ in BINARY_LEGS:
        leg = t.legs[label]
        lo, hi = leg["wilson"]
        call = "rejects H0" if leg["significant"] else "does not reject H0"
        # The divisor spelled out twice, once beside the rates and once beside the
        # interval, because the figure this line used to print — 0.531 for a night the
        # report publishes as 0.652 — sat in the same column position as the headline
        # and differed only by the population underneath it (#2060). Naming the count
        # is what turns 0.531 and 0.652 into two measurements instead of a regression.
        print(f"  {label:<10} delta {leg['delta']:+.3f}  "
              f"rates {leg['prev_rate']:.3f} -> {leg['cur_rate']:.3f} "
              f"(over {leg['rate_n']} paired queries)  "
              f"discordant b={leg['b']} c={leg['c']}  "
              f"exact McNemar p={leg['p']:.3f} ({call})  "
              f"Wilson 95% on {t.cur.label} over {leg['rate_n']} paired queries: "
              f"[{lo:.3f}, {hi:.3f}]")
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
    if t.question_break:
        verdict = ("WITHHELD: query text moved under an unchanged id set, so a paired "
                   "test across this pair scores the edit, not the system — a re-base "
                   "point, recorded as an observation")
    elif t.admissible:
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
    # `auditable`, not `joinable`: a pair the gold-label guard refused (#1637) joins
    # its ids and still gets no number, so counting it as audited would put it in the
    # denominator of the two "legs rejecting (of N)" lines below — where it has no
    # legs — and would report a window of refused pairs as a window that was measured.
    audited = [t for t in transitions if t.auditable]
    unjoinable = sum(1 for t in transitions if not t.joinable)
    gold_moved = sum(1 for t in transitions if t.incomparable)
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
          + (f"  ({unjoinable} unjoinable by records[].id"
             if unjoinable else "")
          + (f"{'; ' if unjoinable else ''}{gold_moved} incomparable: gold labels moved "
             "under an unchanged id set (#1637)" if gold_moved else "")
          + (")" if (unjoinable or gold_moved) else ""))
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
    # The witness count rides beside the moved count (#1852) so a 0 cannot read as a
    # clean bill from a tool that had no text to compare.
    question_moved = sum(1 for t in transitions if t.question_break)
    question_seen = sum(1 for t in transitions
                        if t.joinable and question_witness(t.prev, t.cur))
    print(f"  query text moved under an unchanged id set: {question_moved}"
          f"   <- compared on {question_seen} of {total} transitions "
          "(records[].query present on both nights)")
    print(f"  verdicts admissible under the drift-controlled contract: {admissible} of {total}")
    return {"total": total, "audited": len(audited), "withheld": withheld,
            "question_moved": question_moved, "question_seen": question_seen,
            "incomparable": gold_moved,
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
        if t.incomparable:
            # Named before the metric is read, because `t.legs` is empty for exactly
            # this reason: a claim re-scored across a gold re-label would otherwise
            # key a leg that was never computed, and the claim's own verdict is not
            # what the guard is for. The two fingerprints are printed so the reader
            # can find the edit that moved the gold.
            print(f"  {prev_label} -> {cur_label}: INCOMPARABLE ({t.incomparable}) "
                  "-> cannot evaluate")
            all_support = False
            continue
        leg = t.legs[metric]
        moving = leg["delta"] < 0 if decline else leg["delta"] > 0
        direction_ok = moving and leg["significant"]
        supported = direction_ok and not (t.drift_moved is not False)
        detail = (f"delta {leg['delta']:+.3f}")
        if leg["exact"]:
            # This is the line the nightly loop quotes when it rules a written
            # regression claim SUPPORTED or UNSUPPORTED, so an interval printed here
            # without its denominator is the one that reaches a verdict first (#2060).
            detail += (f", exact McNemar p={leg['p']:.3f}, discordant b={leg['b']} c={leg['c']}, "
                       f"Wilson 95% over {leg['rate_n']} paired queries "
                       f"[{leg['wilson'][0]:.3f}, {leg['wilson'][1]:.3f}]")
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
                    help="directory of nightly-*.json (default: app.paths."
                         "EVAL_BASELINES_DIR under this process's data root when it "
                         "holds them, else production_data_root()/eval/baselines; a "
                         "LLOYD_ROOT override wins over both. The repo copy is never "
                         "the default: it holds tracked non-nightly pins and no "
                         "nightly-*.json, so a run from a worktree falls back to the "
                         "production data root rather than auditing zero nights)")
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
                    help="exit non-zero if any pair of nights is unjoinable by "
                         "records[].id or refused as incomparable (gold labels moved "
                         "under an unchanged id set, #1637)")
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
    # Named as what it is: the mean of `scoring.entity_hit` over EVERY record of the
    # night, which is why it sits below the published `entity_hit_rate` of the same
    # night — #1663 divides that one over the queries carrying entity gold. The series
    # borrows the published metric's name, so its denominator has to travel with it
    # (#2060).
    print("  entity_hit_rate series (one mean per night over every record of that "
          "night, so below the published rate of the same name, which divides over "
          "the queries carrying entity gold): "
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
    #: The re-base chronology. One constraint on its own wording, found the hard way
    #: in this round: the #1663 node in `tests/test_eval_trend_stats.py`
    #: (`test_a_pair_differing_only_in_which_queries_carry_gold_is_annotated`) proves
    #: a moved gold subset did NOT break the id join by asserting that the join's
    #: error title is absent from this block's whole output — so the sixth point below
    #: has to describe that ERROR line rather than quote its title, or the paragraph
    #: trips the check that describes it.
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
          "docs (81 -> 86 gold queries): a third re-base point. It stays a point "
          "even though those five are gone again below, because the nights before "
          "this drop really were scored on 86; what was false in earlier printings "
          "of this line was that the five were waiting on a collection — no "
          "collection ever indexed them, which is why they were retired rather "
          "than re-based. And a fourth re-base point, which is NOT a corpus change: "
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
          "absence rather than by a recorded off. And a fifth re-base point, which is "
          "also NOT a corpus change: #1663 (`efcee660`) re-defined the DENOMINATOR of "
          "the five scored rates on 2026-09-28 — each now divides over the queries "
          "carrying gold for its own leg instead of over every record — so "
          "nightly-20260929 is the first night scored the new way and nightly-20260928 "
          "the last scored the old. The measured step across that boundary was "
          "entity_hit_rate 0.488 -> 0.636 (n 86 -> 66), doc_hit_rate 0.616 -> 0.671, "
          "mrr_doc 0.305 -> 0.332, ndcg10 0.360 -> 0.392: a rate taken over a smaller, "
          "gold-bearing denominator, not a system that got better. It moves the five "
          "SCORED rates only — the recall series divided over the gold-bearing subset "
          "already, so those cross the boundary and stay comparable — and the shift is "
          "invisible to this tool's id join, because an unchanged corpus keeps every "
          "id. So do not compare one of the five from before 2026-09-28 with one after "
          "it, and read the DEFINITION BREAK line a pair prints when its two nights' "
          "gold-bearing subsets differ as this boundary arriving inside your window: "
          "the line names the legs that moved and is an annotation, never a refusal — "
          "the pre-re-base nights stay in the published window annotated. The 80%-power "
          "decision "
          "that used to sit behind this line is no longer open — see the n "
          "printed above. And a sixth re-base point, which IS a corpus change and "
          "the undoing of the third: on 2026-09-28 #1662 (`e64ac4b3`) retired "
          "those same five #1354 queries (86 -> 81 gold queries), because the "
          "checkout documents they named are in no deployed qmd collection, so "
          "each of the five counted on the document leg and scored a miss there on "
          "every run that ever scored them. Their removal lifts doc_hit_rate, "
          "doc_recall_avg, mrr_doc and ndcg10 with retrieval standing exactly "
          "still — a mean moving because its population did — so do not compare "
          "absolute DOCUMENT-leg means across the 2026-09-29 night, the first "
          "scored on 81. No number is quoted for that step, deliberately: #1662's "
          "drop and #1663's denominator re-definition landed the same day, so the "
          "one published pair spanning them carries both moves and neither is "
          "attributable from it; the step above is #1663's alone, and a figure "
          "credited to #1662 would be a number nobody checked. What is verifiable "
          "rather than measured is the half this step does NOT touch: the entity "
          "and term legs do not move here, because all five carried "
          "`expect_entities: []` and no `expect_terms`, the block-present-names-no-"
          "label case `run_eval.py`'s `_counts_on_leg` had already dropped from "
          "those two denominators — which is the same fact that left the document "
          "leg as the only one they could drag. And unlike the fifth point, this "
          "one is not invisible to this tool: an id leaving the corpus breaks the "
          "join on purpose, so when this boundary falls inside a window it arrives "
          "as the ERROR line this tool prints for ids present in one night and not "
          "the other, naming the ids only in the earlier night, plus the "
          "unjoinable transition in the counts above — an announced re-base, not a "
          "silent one. And one CLASS of re-base point with no dated instance yet "
          "(#1852): a gold query whose TEXT is edited keeps its id and its gold, so "
          "neither the join, the labels fingerprint nor the corpus block can see it. "
          "This tool compares each record's own question text across a pair and "
          "prints a QUESTION BREAK line naming the ids — an annotation like the "
          "fifth point's, never a refusal, but a pair carrying it is never an "
          "admissible verdict; the SUMMARY line counts how many transitions it could "
          "compare, so a zero there is a measurement and not blindness.")

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
