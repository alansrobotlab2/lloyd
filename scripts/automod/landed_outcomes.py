"""Landed-change outcomes split by author: the loop's commits vs a person's.

The automod program asserts that agent-authored changes are safe enough to land
unattended. The only quality signal it had measures whether a *run* succeeded
(`run_outcomes.fail_rate`) or, at best, the aggregate undo rate
(`scripts/automod/scorecard.py` row 5), which pools both arms into one number
and so can never say whether the loop's landings are undone more often than the
person's. This splits those outcomes by author and puts a Wilson interval on
each, so the two arms can be compared on the overlap window instead of
asserted.

Read-only. It runs `git log`/`git show` over `~/lloyd` and reads the automod
ledger; it writes nothing inside the checkout (the guardian alerts hourly on
runtime data there) and makes no model call. `record()` appends one row to
`landed_outcomes.jsonl` beside `scorecard.jsonl`, in the automod state dir —
`scorecard_path()`'s directory — and the default run only prints.

The arms, and why not `git log --format=%an`: landings have credited Alan as the
author since 2026-09-26 (`9407ddd1`, "a landing credits Alan and Lloyd only,
never a model-written co-author"), so of the 30 newest `promoted` commits 25 are
authored `alansrobotlab` and 2 `Lloyd`. Author name alone would file nearly every
landing under "human". The key is therefore the ledger's landed sha, with the
pre-`9407ddd1` loop identities kept as a secondary test — the same rule
`scorecard.py`'s row 5 uses, generalised to both directions.

What each arm gets:

  commits            every commit on `main` in the window, one arm each
  later-undo         an arm's commit whose added lines a later commit of the
                     OTHER arm removed within 7 d — `scorecard.py` row 5's
                     definition, reused unchanged, so the two numbers compare
  revert commits     a commit whose subject is a revert, over the arm's commits
  guardian rollback  a `promoted` commit a `rollback_succeeded` row took off
                     `main`, over agent landings. The agent arm only: a person's
                     commit is never on the ledger as a promotion.
  rounds-to-land     `review` rows per landed round, attempts not rungs. The
                     agent arm only — a person's commit goes through no gate, so
                     the human row prints `n/a (no gate)`, never a zero.

What the loop PAYS per landing (#2378) is the other half of the page and the
half that did not exist: `cost` and `latency` ride in the same row, and change
the denominator from the activity performed to the outcome achieved. Every
compute figure this system shows elsewhere is per-run — `runs.duration_seconds`
summed into `gpu_hours` per task (`workers/queue.py:1353`) — so a three-hour
implementer turn that lands nothing was one more run inside a budget, and the
program could answer "were landed changes undone?" but never "what did a
landing cost, and how long from the signal to the landing?":

  gpu_hours_per_landing   wall-clock seconds/3600 — the autonomy page's own
                          proxy, under that name — summed over the ledger's rung
                          and landing-wait seconds of promoted rounds, per
                          landing, never printed without the next line
  never_landed_share      the seconds sitting in rounds that produced no
                          `promoted` row: cost-per-outcome is gameable by
                          attempting less, and this is the line that shows it
  run_join                `runs` rows name a round only in free-text `summary`,
                          never in a structured column, so the match rate prints
                          with the figure and the figure is withheld below
                          `JOIN_FLOOR` instead of printed low
  filing_to_first_round   p50/p90 with n, beside `first_round_to_landing`: the
  first_round_to_landing  triage wait and the implementation+gate wait, the
                          split no figure on this box had ever measured of its
                          own loop

`n` is what the window can actually observe: a commit with less than 7 d of
later history cannot yet have been undone, so it is excluded from the
denominator rather than counted as a clean landing, and `0 landings` renders as
`0/0 (no landings)` rather than `0%` — the zero-denominator rule. The window is
the ledger's own span, so the arms are only comparable from the first ledger row
on (`2026-09-06` as measured at file time). What lies before it is reported, not
folded in: `human_history_from` is the oldest human-arm commit of the whole of
`main` — measured at filing, `30646b61` at 2026-04-04 — and one line of the
report names how many commits predate `comparable_from` (358 then, every one of
them human-arm), so the reader sees the months no rate covers instead of
reading the window start as the human arm's whole reach (#1870).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.stats import wilson_ci

from scripts.automod import scorecard as SC
from scripts.automod.scorecard import HUMAN_TOUCH_DAYS, LOOP_AUTHORS

LIVE_ROOT = SC.LIVE_ROOT
AGENT, HUMAN = "agent", "human"
# A `promoted` row's commit is the landing itself. `arch_review` is the doc pass
# landing a vault commit and is the loop's work by the same right row 5 grants it.
LANDING_EVENTS = ("promoted", "arch_review")
# git writes a revert as `Revert "subject"`, and a hand-written one usually
# starts the same way or with a `Re:`/`fix:` style prefix. Anchored to the
# subject's start: a commit that merely mentions reverting in its body is not
# one, and a rate built from those would be a claim about intent, not evidence.
REVERT_SUBJECT = re.compile(r"^\s*(?:[A-Za-z][A-Za-z]*:\s*)?revert\b", re.I)
# A prefix shorter than a git short-sha cannot identify a commit. Without this
# a ledger row whose `commit` is "" or "a" would prefix-match all of history
# into the agent arm — `scorecard.py` only guards the empty string.
MIN_SHA_PREFIX = 7


# ── arm assignment ────────────────────────────────────────────────────────

def landing_shas(events: list[dict]) -> list[str]:
    """Every commit value the ledger records as landed by the loop."""
    return [str(e.get("commit") or "").strip() for e in events
            if e.get("event") in LANDING_EVENTS and str(e.get("commit") or "").strip()]


def is_agent(commit: dict, shas: list[str], *,
             loop_authors: frozenset[str] = LOOP_AUTHORS) -> bool:
    """The agent arm iff a landed sha prefix-matches this commit's, or its git
    author name is one the loop itself committed under."""
    if str(commit.get("author") or "").strip().lower() in loop_authors:
        return True
    sha = str(commit.get("sha") or "")
    if len(sha) < MIN_SHA_PREFIX:
        return False
    return any(len(s) >= MIN_SHA_PREFIX and (sha.startswith(s) or s.startswith(sha))
               for s in shas)


def assign_arms(commits: list[dict], shas: list[str]) -> list[dict]:
    """`commits` with an `"arm"` added, each in exactly one arm."""
    out = []
    for c in commits:
        out.append(dict(c, arm=(AGENT if is_agent(c, shas) else HUMAN)))
    return out


# ── the shape every number is printed in ──────────────────────────────────

def rate(k: int, n: int) -> dict[str, Any]:
    """One measurement: numerator, denominator, point rate, Wilson interval.

    `k` and `n` are separate fields, never folded into a string, and the CI is
    computed from the same `n` that is printed — an interval from a different
    denominator than the one on the page is a decoration. `wilson_ci` answers
    `n == 0` with `(nan, nan)`, and NaN is not valid JSON, so the bounds go to
    `None` and `rate_text` says what is missing instead of drawing a dash.
    """
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n,
            "rate": round(k / n, 4) if n else None,
            "ci_low": (round(lo, 4) if n else None),
            "ci_high": (round(hi, 4) if n else None)}


def rate_text(row: dict) -> str:
    """`k/n (pct %) [lo, hi]`, or the reason there is no rate."""
    k, n = row["k"], row["n"]
    if not n:
        return f"{k}/{n} (no landings)"
    return (f"{k}/{n} ({100 * k / n:.1f} %) "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]")


# ── the measures ──────────────────────────────────────────────────────────

def later_undo(commits: list[dict], *, repo: Path, arm: str,
               now: float, days: int = HUMAN_TOUCH_DAYS) -> dict[str, Any]:
    """Scorecard row 5's question, asked of one arm with the other arm as the
    remover: how often were this arm's added lines gone within 7 days.

    One function, both arms, so `agent` and `human` are the same measurement and
    the intervals mean what they look like they mean. Row 5 counted only a
    person undoing the loop; symmetric here, because the loop also edits what a
    person wrote and that is the same kind of event. The denominator is this
    arm's commits with at least `days` of history after them: a commit from
    yesterday cannot be observed undone yet, and counting it as clean would pull
    the rate down with the age of the window rather than the quality of the work.
    """
    mine, theirs = [c for c in commits if c["arm"] == arm], \
                   [c for c in commits if c["arm"] != arm]
    horizon = days * 86400
    eligible = [c for c in mine if now - c["ct"] >= horizon]
    k = 0
    for c in eligible:
        later = [t for t in theirs if horizon >= t["ct"] - c["ct"] > 0]
        if not later:
            continue
        if SC._undone_by_hand(repo, {"commit": c["sha"]}, later):
            k += 1
    return rate(k, len(eligible))


def revert_rate(commits: list[dict], *, arm: str) -> dict[str, Any]:
    """Revert subjects over the arm's own commits — the human-side analogue of a
    guardian rollback, and the only undo signal available for a commit that was
    never on the ledger as a promotion."""
    mine = [c for c in commits if c["arm"] == arm]
    return rate(sum(1 for c in mine if REVERT_SUBJECT.search(str(c.get("subject") or ""))),
                len(mine))


def rollback_rate(events: list[dict], *, since: float, until: float) -> dict[str, Any]:
    """Landings a `rollback_succeeded` row took back off `main`, over the
    landings in the window. The one definition of "took back" is
    `state.reverted_commits`, which already walks a `reset` route's whole parent
    chain — counting only the row's own `commit` under-reported it (#939/#763).

    `true_positives` stays `None`: whether a rollback caught a real defect is a
    human judgment and none has been recorded (scorecard row 10 keeps the same
    null rather than writing a claim)."""
    from scripts.automod import state as S
    lands = [e for e in events if e.get("event") == "promoted"
             and str(e.get("commit") or "") and since <= SC._ts(e) <= until]
    reverted = S.reverted_commits(events)
    return rate(sum(1 for e in lands if str(e["commit"]) in reverted), len(lands))


def rounds_to_land(events: list[dict], *, landed_rounds: set[str]) -> dict[str, Any]:
    """Attempts per landed round, counted over `review` rows keyed by `round_id`.

    Not `gate` rows: those are per rung, ~12 for one attempt, so a gate-row count
    would report the ladder's length as the loop's iteration cost. A round with
    no `review` row is left out of the denominator rather than counted as one
    attempt — nothing graded it, and inventing an attempt would bias the number
    toward the cheap rounds, which are exactly the ones the gate answers from its
    cache without a review.

    The rate-shaped part is the binomial — the share of graded landed rounds that
    needed a SECOND review attempt — because that is the one figure here with a
    `k/n` meaning and therefore a Wilson interval. Total reviews per round is a
    mean, not a proportion (3 reviews over 2 rounds is not a probability), so it
    rides beside it as `mean`/`median`/`distribution` and never inside `k`/`n`.
    """
    per_round: dict[str, int] = {}
    for e in events:
        if e.get("event") == "review" and e.get("round_id") in landed_rounds:
            per_round[str(e["round_id"])] = per_round.get(str(e["round_id"]), 0) + 1
    counts = sorted(per_round.values())
    row = rate(sum(1 for c in counts if c > 1), len(counts))
    row.update({
        "median": round(statistics.median(counts), 2) if counts else None,
        "mean": round(sum(counts) / len(counts), 2) if counts else None,
        "total_reviews": sum(counts),
        "distribution": {str(x): counts.count(x) for x in sorted(set(counts))},
    })
    return row


def rounds_text(row: dict) -> str:
    """`mean 1.5 reviews/round over 2 rounds; second attempt 1/2 (50.0 %) […]`,
    or `n/a (no gate)` — a zero would say the arm was graded once and passed,
    which is the opposite of not being gated at all."""
    if not row["n"]:
        return "n/a (no gate)"
    return (f"mean {row['mean']} reviews/round over {row['n']} rounds "
            f"(median {row['median']}); second attempt {rate_text(row)}")


# ── cost and latency per ACHIEVED landing (#2378) ─────────────────────────
#
# Everything above is a quality signal. None of it is an economic one, and every
# compute figure the system shows elsewhere is activity-denominated:
# `runs.duration_seconds` summed and divided by 3600 into `gpu_hours` per TASK
# (`workers/queue.py:1353`), and `app/autonomy.py` (`:5157` at filing) summing
# the same quotient per run per task. A task that burns three hours and lands
# nothing is therefore one more run inside its budget, and the loop can answer
# "were landed changes undone?" but never "what did a landing cost". Denominator
# = the achieved outcome, not the activity, is the whole of the change below.
#
# Cost-per-outcome is itself gameable by the worst move available — a loop that
# stops attempting hard items gets cheaper and looks better — so
# `never_landed_share` is emitted with the figure, never apart from it, and the
# joined-run term is withheld when it cannot see enough of the turns to bound
# the total. Report-only: nothing here is promoted, gated or auto-disabled on.

#: The ledger rows that are one round's spend, as `(event, field)` pairs,
#: counted only on a row that carries a `round_id` — an unattributed wait cannot
#: be charged to a landing. This is `scorecard.py` row 9's per-round `gate`
#: accumulation (`:968-971` at filing) widened to the other fields a round
#: spends, not a third accumulation of the same rows.
#:
#: Included, each a distinct span: `gate.seconds` (one rung; every rung of the
#: ladder sums), `review.seconds` (the grader's turn, timed from its own call),
#: `review.review_confirm_seconds` (#1903's second reader, which runs AFTER that
#: row stamps its `seconds`, so it is additive and not nested),
#: `land_wait_rounds.waited_s`, `pause_set.seconds`, `restart_flushed.waited_s`.
#:
#: Excluded, each on a stated reason: `errors_window_s` (the length of a
#: rate-observation window, not a spend), `review.rung_wait_s` (the wait for a
#: review graded CONCURRENTLY with the tests rung — adding it double-counts the
#: same wall clock), `review.waited_s` (the grader-retry sleep, nested inside
#: the same row's `seconds`), `restart_flushed.waited_rounds_s` /
#: `waited_settle_s` (phases of that row's own `waited_s`), and
#: `noise_refreshed.seconds` (no round to charge it to). As measured on
#: 2026-10-09, `pause_set` and `restart_flushed` rows carry no `round_id` at all
#: and so contribute nothing here; the entries stay so that a writer which
#: starts recording the round is counted rather than silently dropped.
ROUND_SPEND_FIELDS: tuple[tuple[str, str], ...] = (
    ("gate", "seconds"),
    ("review", "seconds"),
    ("review", "review_confirm_seconds"),
    ("land_wait_rounds", "waited_s"),
    ("pause_set", "seconds"),
    ("restart_flushed", "waited_s"),
)
#: `duration_seconds / 3600` is what the autonomy page prints as `gpu_hours`
#: (`workers/queue.py:1353`). It is wall-clock and it carries gate-rung
#: subprocess time inside the same run, so the column says that rather than
#: inventing a GPU-second precision the store does not have (#2378's own risk).
WALLCLOCK_PROXY = ("wall-clock seconds/3600 — the same proxy the autonomy page "
                   "prints as gpu_hours, not measured GPU time")
#: The item's own floor: below this share of promoted landings the run join
#: cannot bound a landing's turn time, so the cost figure is withheld instead of
#: printed low. 595/622 (95.7 %) clears it as measured on 2026-10-09.
JOIN_FLOOR = 0.5
JOIN_TOO_WEAK = "join too weak to interpret"
NO_LANDINGS = "0/0 (no landings)"
#: How a round id is spelled by the ledger and by the one `runs` column that
#: actually carries it — free-text `runs.summary` (`runs.response_json` in 10
#: rows, `queue.payload_json` nowhere usefully): the join is prose, which is
#: why it is measured and floored rather than trusted.
ROUND_ID_RE = re.compile(r"\bSM_\d{8}_\d{6}\b")


def _percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile of `values`, None on an empty population.

    The definition is `scripts/autoresearch/promotion_fp_rate.percentile` —
    imported, not restated, so the two measured p90s on this box cannot
    disagree about what `p90` means the way #1667's parallel scorecard nearly
    did about what a landing means.
    """
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    from scripts.autoresearch.promotion_fp_rate import percentile
    return percentile(xs, p)


def _pctl(values: list[float]) -> dict[str, Any]:
    """One latency population as `n`, p50 and p90, or None percentiles and n=0."""
    xs = [float(v) for v in values]
    return {"n": len(xs),
            "p50_s": (round(_percentile(xs, 0.5), 1) if xs else None),
            "p90_s": (round(_percentile(xs, 0.9), 1) if xs else None)}


def pctl_text(row: dict) -> str:
    """`p50 43200 s, p90 77760 s  (n=3)`, or `n/a` — never a dash or a 0, which
    would read as a latency of zero rather than no measurable latency."""
    if not row["n"]:
        return "n/a (no landing in the window has both a filing and a round stamp)"
    return f"p50 {row['p50_s']:.0f} s, p90 {row['p90_s']:.0f} s  (n={row['n']})"


def share_text(share: dict, *, unit: str = "s") -> str:
    """`k/n s (pct %)`, or `0/0 (no landings)`. A share of SECONDS is not a
    binomial proportion, so it carries no Wilson interval — `rate_text` is for
    rates over trials, this is for hours over hours."""
    k, n = share["k"], share["n"]
    if not n:
        return f"0/0 {unit} (no ledger-attributed seconds in the window)"
    return f"{k}/{n} {unit} ({100 * k / n:.1f} %)"


def join_text(join: dict) -> str:
    """The join's own match rate, printed beside the figure that leans on it."""
    k, n = join["k"], join["n"]
    head = f"{k}/{n} promoted landings attributed to at least one run row"
    if not n:
        return f"{head} ({NO_LANDINGS})"
    return (f"{head} ({100 * k / n:.1f} %), floor {join['floor']:.2f}"
            + ("" if join["strong"] else " — too weak to interpret a cost figure"))


def round_spend(events: list[dict], *, since: float, now: float) -> dict[str, float]:
    """Wall-clock seconds the ledger attributes to each `round_id` in the window.

    Windowed by the ROW's own stamp: a rung that ran in the window is spend the
    window pays. A round whose landing lies outside the window still lands in
    `never_landed` if no `promoted` row of its falls inside it, which is the
    honest reading at the window edge and why the window is printed on every
    figure.
    """
    wanted: dict[str, list[str]] = {}
    for event, field in ROUND_SPEND_FIELDS:
        wanted.setdefault(event, []).append(field)
    spend: dict[str, float] = {}
    for e in events:
        rid = str(e.get("round_id") or "")
        fields = wanted.get(str(e.get("event") or ""))
        if not rid or not fields:
            continue
        ts = SC._ts(e)
        if not (since <= ts <= now):
            continue
        for field in fields:
            val = e.get(field)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                spend[rid] = spend.get(rid, 0.0) + float(val)
    return spend


def _workers_db() -> Path:
    """The queue store this report reads: the LIVE one, always.

    `app.paths.WORKERS_DB` is the resolver a WRITER wants — rule 3 of the data
    home gives a round's worktree its own `.lloyd-data/`, so a report run from a
    worktree or the canary would open a store with no runs in it and print a 0 %
    join as though the loop had none. This module reports on the live loop
    wherever it is invoked (it reads the live ledger and `LIVE_ROOT`'s history
    the same way), so it reaches the live root through `production_data_root()`,
    the function whose docstring restricts it to readers that mean production on
    purpose. Nothing here ever opens the store for writing: the join is a
    `mode=ro` URI.
    """
    from app.data_root import production_data_root
    return production_data_root() / "workers.db"


def _empty_join(why: str) -> dict[str, Any]:
    """A join that was never run: `strong` false, so no cost figure prints."""
    return {"available": False, "why": why, "db": None, "k": 0, "n": 0,
            "share": None, "floor": JOIN_FLOOR, "strong": False,
            "seconds": {}, "runs": 0, "ambiguous_runs": 0,
            "runs_naming_rounds": 0, "rows_scanned": 0}


def run_round_join(round_ids: set[str], *, db_path: Path | None = None,
                   n_landings: int | None = None) -> dict[str, Any]:
    """Which of `round_ids` at least one `runs` row names, and their seconds.

    `round_id` is not stored structurally on a run row: the carrier is the
    free-text `runs.summary` (#1188's note in `workers/sources/autocode.py`
    writes it there). So the join is one scan of `runs`, a round-id regex over
    `summary`/`response_json`/`meta_json`, and then a floor on the resulting
    match rate — not a fuzzy time window.

    A run row naming TWO rounds is dropped from the seconds rather than
    credited to both: its wall clock cannot be split, and halving it would
    invent a split. `ambiguous_runs` says how many were dropped. `n_landings`
    is the denominator the caller means (every promoted landing in the window,
    attributed or not) — the match rate is over landings, not over the rounds
    that happened to be scanned.
    """
    wanted = {str(r) for r in round_ids if str(r)}
    out = _empty_join("no runs db read")
    out["n"] = (len(wanted) if n_landings is None else int(n_landings))
    if db_path is None:
        return out
    path = Path(db_path)
    out["db"] = str(path)
    if not path.exists():
        out["why"] = f"runs db not found at {path}"
        return out
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "select duration_seconds, summary, response_json, meta_json from runs").fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:               # a missing table is "no join", not a crash
        out["why"] = f"runs table unreadable at {path}: {str(exc)[:120]}"
        return out
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    ambiguous = named = 0
    for dur, summary, response_json, meta_json in rows:
        found: set[str] = set()
        for blob in (summary, response_json, meta_json):
            if blob:
                found.update(ROUND_ID_RE.findall(str(blob)))
        if not found:
            continue
        named += 1
        if len(found) > 1:
            ambiguous += 1
            continue
        rid = next(iter(found))
        if rid not in wanted:
            continue
        secs = float(dur or 0)
        seconds[rid] = seconds.get(rid, 0.0) + secs
        counts[rid] = counts.get(rid, 0) + 1
    out.update({"available": True, "why": None, "seconds": seconds,
                "runs": sum(counts.values()), "ambiguous_runs": ambiguous,
                "runs_naming_rounds": named, "rows_scanned": len(rows),
                "k": sum(1 for r in wanted if r in seconds)})
    out["share"] = (round(out["k"] / out["n"], 4) if out["n"] else None)
    out["strong"] = bool(out["share"] is not None and out["share"] >= JOIN_FLOOR)
    return out


def cost_per_landing(events: list[dict], *, since: float, now: float,
                     join: dict[str, Any],
                     median_full_gate_s: float | None = None) -> dict[str, Any]:
    """What one achieved landing cost, over the same window as the arms.

    `gpu_hours_per_landing` is the total ledger-attributed seconds inside
    promoted rounds divided by the landings that have any, then /3600 — the
    autonomy page's proxy under the autonomy page's name. `median` rides beside
    the mean because the mean over 611 rounds is dragged by the round that
    re-gated itself four times.

    The figure is withheld (`join_strong` false → `JOIN_TOO_WEAK` on the page)
    when the run join cannot see most of the landings. The ledger half needs no
    join, but it is only gate, review and landing-wait time; the implementer's
    turn exists only in `runs.duration_seconds`. Printing rung time alone as
    *the* cost of a landing while a third of the turns cannot be attributed
    would understate spend by a factor the report cannot see, which is a worse
    number than no number — so the joined turn term and the figure that leans
    on it go together.
    """
    landed: set[str] = set()
    for e in events:
        rid = str(e.get("round_id") or "")
        if rid and e.get("event") == "promoted" and since <= SC._ts(e) <= now:
            landed.add(rid)
    spend = round_spend(events, since=since, now=now)
    per_landing = sorted(spend[r] for r in landed if r in spend)
    landed_seconds = sum(per_landing)
    ledger_seconds = sum(spend.values())
    never_seconds = ledger_seconds - landed_seconds
    joined = [float(join["seconds"][r]) for r in landed if r in join.get("seconds")]
    row: dict[str, Any] = {
        "window_from": _iso(since), "window_to": _iso(now),
        "proxy": WALLCLOCK_PROXY,
        "landings": len(landed),
        "n": len(per_landing),
        "landings_without_seconds": len(landed) - len(per_landing),
        "landed_seconds": round(landed_seconds, 1),
        "ledger_seconds": round(ledger_seconds, 1),
        "never_landed": {"k": int(never_seconds), "n": int(ledger_seconds)},
        "gpu_hours_total": round(landed_seconds / 3600.0, 3),
        "gpu_hours_per_landing": (round(landed_seconds / len(per_landing) / 3600.0, 4)
                                  if per_landing else None),
        "median_seconds_per_landing": (round(statistics.median(per_landing), 1)
                                       if per_landing else None),
        "p90_seconds_per_landing": (round(_percentile(per_landing, 0.9), 1)
                                    if per_landing else None),
        "joined_runs": int(join.get("runs") or 0),
        "joined_seconds": round(sum(joined), 1),
        "joined_n": len(joined),
        "joined_gpu_hours_per_landing": (round(sum(joined) / len(joined) / 3600.0, 4)
                                         if joined else None),
        "median_full_gate_s": median_full_gate_s,
        "join": {"k": join["k"], "n": join["n"], "share": join["share"],
                 "floor": join["floor"], "strong": bool(join["strong"])},
        "join_available": bool(join["available"]),
        "join_why": join.get("why"),
        "ambiguous_runs": int(join.get("ambiguous_runs") or 0),
    }
    return row


def _cost_key_lines(cost: dict[str, Any]) -> list[str]:
    """The number-bearing cost lines: withheld together, printed together.

    Three states, in this order: no landing in the window (`0/0 (no landings)`),
    a join too weak to bound the total (`join too weak to interpret`), and a
    measured figure with its own `n` and the window on the same line.
    """
    window = f"{cost['window_from']} → {cost['window_to']}"
    per_landing = f"{'gpu_hours_per_landing':30s}"
    joined = f"{'joined_run_hours_per_landing':30s}"
    if not cost["n"]:
        return [f"{per_landing} {NO_LANDINGS}  {window}",
                f"{joined} {NO_LANDINGS}  {window}"]
    if not cost["join"]["strong"]:
        # The match rate itself is on the `run_join` line above; repeating the
        # whole clause here would put the same 0/622 on the page twice. What this
        # branch must not do is print a figure that looks interpretable.
        weak = (f"{JOIN_TOO_WEAK}  (run join {cost['join']['k']}/{cost['join']['n']}, "
                f"floor {cost['join']['floor']:.2f})")
        return [f"{per_landing} {weak}", f"{joined} {weak}  {window}"]
    joined_text = (f"{cost['joined_gpu_hours_per_landing']} h  n={cost['joined_n']}  {window}"
                   if cost["joined_gpu_hours_per_landing"] is not None
                   else f"n/a (no run row attributed to a landing)  {window}")
    return [
        f"{per_landing} {cost['gpu_hours_per_landing']:.3f} h"
        f" (total {cost['gpu_hours_total']} h, median "
        f"{cost['median_seconds_per_landing']:.0f} s,"
        f" p90 {cost['p90_seconds_per_landing']:.0f} s)  n={cost['n']}  {window}",
        f"{joined} {joined_text}",
    ]


def cost_lines(cost: dict[str, Any]) -> list[str]:
    """The cost block as printed lines. `never_landed_share` is never separable
    from the figure beside it: a cheaper loop that attempted less is the one way
    this metric gets gamed, and the share is what shows it."""
    out = [f"cost per achieved landing — {cost['window_from']} → {cost['window_to']} — "
           f"{cost['proxy']}",
           f"{'run_join':30s} {join_text(cost['join'])}"]
    out += _cost_key_lines(cost)
    never = share_text(cost["never_landed"])
    out.append(f"{'never_landed_share':30s} {never} of the ledger-attributed seconds sit "
               f"in rounds with no `promoted` row"
               + (f" ({cost['never_landed']['k'] / 3600:.1f} h of "
                  f"{cost['never_landed']['n'] / 3600:.1f} h)"
                  if cost["never_landed"]["n"] else ""))
    if cost["landings_without_seconds"]:
        out.append(f"{'landings_no_seconds':30s} {cost['landings_without_seconds']} of "
                   f"{cost['landings']} promoted landings in the window carry no ledger "
                   f"seconds at all, so they are in the landings count and not in n")
    if cost["ambiguous_runs"]:
        out.append(f"{'ambiguous_runs':30s} {cost['ambiguous_runs']} run rows name two or "
                   f"more round ids and are dropped from the joined seconds rather than "
                   f"split across them")
    if cost["median_full_gate_s"] is not None:
        out.append(f"{'median_full_gate_s':30s} {cost['median_full_gate_s']} s (one whole "
                   f"ladder run: `gate_duration_stats` through scorecard row 14's "
                   f"`_full_gate_median`, reused not re-derived) — the yardstick for the "
                   f"per-round figures above")
    if cost["join_why"]:
        out.append(f"{'join_unavailable':30s} {cost['join_why']}")
    return out


def cost_text(cost: dict[str, Any]) -> str:
    return "\n".join(cost_lines(cost))


def _filing_ts(item_id: int, *, backlog_dir: Path | None = None) -> float | None:
    """The item's own `created:` as an instant, or None if it cannot be read.

    Routed through `scripts.automod.backlog._iso_ts(..., legacy_local=True)`,
    which is `app.backlog_move.utc_instant` — the one place this box decided
    what a naive board stamp means (#1517). At and after
    `LOCAL_STAMP_CUTOVER` a naive stamp IS UTC; below it the stamp came off a
    surface that wrote the machine's local clock, and reading it as UTC would
    move the filing seven hours on this box and report a triage latency that
    never happened. "One naive-UTC clock" means the same INSTANT on both sides
    of the subtraction, which is what that function is for.
    """
    from scripts.automod import backlog as B
    root = Path(backlog_dir or B.BACKLOG_DIR)
    try:
        item_id = int(item_id)
    except (TypeError, ValueError):
        return None
    # The live board first, then the archive the monthly retention sweep moves a
    # closed item to (`retire_merged_board_items`' destination, the same board
    # root's `archived/` sibling): a landing is by definition a closed item, so a
    # reader who only ever globs the live board eventually loses its filing stamp
    # and the latency n decays as the board ages.
    for directory in (root, root.parent / "archived"):
        for path in sorted(Path(directory).glob(f"{item_id}-*.md")):
            ts = B._iso_ts(SC._frontmatter(path).get("created"), legacy_local=True)
            if ts:
                return ts
    return None


def filing_latency(events: list[dict], *, since: float, now: float,
                   ledger_from: float | None = None,
                   backlog_dir: Path | None = None) -> dict[str, Any]:
    """Signal-to-landing, split into the two waits that produce it.

    Three stamps per landed round, all read as one instant: the item's
    `created:` (filing), the round's EARLIEST `round_start` row (the loop first
    picked the item up), and its `promoted` row inside the window (the landing).
    The split is the point — the talk's "coding was never the bottleneck"
    predicts filing→first-round dominates, and no figure on this box had ever
    tested that of its own loop. Which arm dominates at p90 is a ruling on the
    first real row, not a claim this module makes, so `queue_wait_share_median`
    is descriptive and nothing more.

    Only rounds with all three stamps are in any n; a round whose item file is
    gone, or whose `created:` postdates its own first round, is counted out and
    named. Items filed before the ledger exists are not truncated away — they
    are reported as `filed_before_ledger`, because their filing stamp is real
    even where no ledger row is, and a silent drop is the #1667 trap the arms
    already fell into once.
    """
    first_start: dict[str, float] = {}
    start_item: dict[str, int] = {}
    item_of: dict[str, int] = {}
    for e in events:
        rid = str(e.get("round_id") or "")
        if not rid:
            continue
        iid = e.get("item_id")
        if isinstance(iid, (int, float)) and not isinstance(iid, bool):
            item_of.setdefault(rid, int(iid))
        if e.get("event") != "round_start":
            continue
        ts = SC._ts(e)
        if ts and (rid not in first_start or ts < first_start[rid]):
            first_start[rid] = ts
            if isinstance(iid, (int, float)) and not isinstance(iid, bool):
                start_item[rid] = int(iid)
    landed_at: dict[str, float] = {}
    for e in events:
        rid = str(e.get("round_id") or "")
        if rid and e.get("event") == "promoted" and since <= SC._ts(e) <= now:
            prev = landed_at.get(rid)
            if prev is None or SC._ts(e) > prev:
                landed_at[rid] = SC._ts(e)

    f2r: list[float] = []
    r2l: list[float] = []
    f2l: list[float] = []
    ratios: list[float] = []
    strata: dict[str, dict[str, list[float]]] = {
        "filed_in_ledger_window": {"f2r": [], "r2l": [], "f2l": []},
        "filed_before_ledger": {"f2r": [], "r2l": [], "f2l": []}}
    missing_item = missing_start = negative = 0
    for rid, land_ts in sorted(landed_at.items()):
        start_ts = first_start.get(rid)
        if start_ts is None:
            missing_start += 1
            continue
        item_id = start_item.get(rid) or item_of.get(rid)
        if item_id is None:
            missing_item += 1
            continue
        filed = _filing_ts(item_id, backlog_dir=backlog_dir)
        if filed is None:
            missing_item += 1
            continue
        filing_to_round, round_to_land = start_ts - filed, land_ts - start_ts
        if filing_to_round < 0 or round_to_land < 0:
            negative += 1
            continue
        f2r.append(filing_to_round)
        r2l.append(round_to_land)
        f2l.append(land_ts - filed)
        if land_ts - filed > 0:
            ratios.append((start_ts - filed) / (land_ts - filed))
        bucket = ("filed_before_ledger" if ledger_from is not None
                  and filed < ledger_from else "filed_in_ledger_window")
        strata[bucket]["f2r"].append(filing_to_round)
        strata[bucket]["r2l"].append(round_to_land)
        strata[bucket]["f2l"].append(land_ts - filed)
    return {
        "window_from": _iso(since), "window_to": _iso(now),
        "clock": ("the item's `created:`, the round's earliest `round_start` and its "
                  "`promoted` row all read as one instant through "
                  "`app.backlog_move.utc_instant` (#1517)"),
        "landings": len(landed_at),
        "filing_to_first_round": _pctl(f2r),
        "first_round_to_landing": _pctl(r2l),
        "filing_to_landing": _pctl(f2l),
        "queue_wait_share_median": (round(_percentile(ratios, 0.5), 4) if ratios else None),
        "missing_item_stamp": missing_item,
        "missing_round_start": missing_start,
        "negative_latency_excluded": negative,
        "strata": [{"cohort": name, "n": len(body["f2l"]),
                    "filing_to_first_round": _pctl(body["f2r"]),
                    "first_round_to_landing": _pctl(body["r2l"]),
                    "filing_to_landing": _pctl(body["f2l"])}
                   for name, body in strata.items()],
    }


def latency_lines(latency: dict[str, Any]) -> list[str]:
    out = [f"signal-to-landing latency — {latency['window_from']} → "
           f"{latency['window_to']} — over {latency['landings']} promoted "
           f"landing(s) in the window",
           f"{'filing_to_first_round':30s} {pctl_text(latency['filing_to_first_round'])}",
           f"{'first_round_to_landing':30s} {pctl_text(latency['first_round_to_landing'])}",
           f"{'filing_to_landing':30s} {pctl_text(latency['filing_to_landing'])}"]
    share = latency["queue_wait_share_median"]
    out.append(f"{'queue_wait_share_median':30s} "
               + ("n/a (no landing measurable in the window)" if share is None else
                  f"{share:.3f} of filing→landing is triage wait, per-round median — "
                  f"descriptive only; which side dominates at p90 is the owed-check "
                  f"ruling, not this line"))
    for st in latency["strata"]:
        if st["n"]:
            out.append(f"{st['cohort']:30s} n={st['n']}: filing→round "
                       f"{pctl_text(st['filing_to_first_round'])}; round→landing "
                       f"{pctl_text(st['first_round_to_landing'])}")
        else:
            out.append(f"{st['cohort']:30s} 0/0 (no landings)")
    skipped = (f"{latency['missing_item_stamp']} no readable item `created:`, "
               f"{latency['missing_round_start']} no `round_start` row, "
               f"{latency['negative_latency_excluded']} negative")
    out.append(f"{'excluded_from_n':30s} {skipped}")
    return out


def latency_text(latency: dict[str, Any]) -> str:
    return "\n".join(latency_lines(latency))


# ── the report ────────────────────────────────────────────────────────────

def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _week(ts: float) -> str:
    y, w, _d = datetime.fromtimestamp(ts, tz=timezone.utc).isocalendar()
    return f"{y}-W{w:02d}"


def _landed_rounds(events: list[dict]) -> set[str]:
    return {str(e.get("round_id") or "") for e in events
            if e.get("event") == "promoted" and e.get("round_id")}


def by_author(*, now: float | None = None, since_days: float | None = None,
              repo: Path | None = None, events: list[dict] | None = None,
              horizon_days: int = HUMAN_TOUCH_DAYS,
              ledger: Path | None = None, runs_db: Path | None = None,
              backlog_dir: Path | None = None) -> dict[str, Any]:
    """One author-split report over `events` (the ledger) and `main`'s history.

    `since_days=None` means the whole ledger span, which is the widest window in
    which both arms have a counterpart; the human arm's older months are labelled
    as having no ledger at all rather than folded into the comparison — labelled,
    that is, from unfiltered history: the whole of `main` is fetched once and the
    rates window it, while `human_history_from` and `pre_comparable_excluded`
    read the unfiltered list, so the reach line reports how far the human arm
    actually goes rather than where the window happens to start (#1870).

    `runs_db` is the queue's store the cost join reads: `None` (the default, and
    what every unit caller gets) means NO run rows are read and the cost figure
    is therefore withheld as `join too weak to interpret`, while `main()` passes
    `app.paths.WORKERS_DB` so the shipped report measures the join for real. A
    report that quietly reached into production data from a test would be both
    slow and non-reproducible; a withheld figure is honest. `ledger` is only for
    the `gate_duration_stats` yardstick beside the per-round figures, and
    `backlog_dir` resolves an item id to its `created:` stamp.
    """
    repo = Path(repo or LIVE_ROOT)
    now = now or datetime.now(timezone.utc).timestamp()
    events = [] if events is None else list(events)
    stamps = [SC._ts(e) for e in events if SC._ts(e)]
    led_from, led_to = (min(stamps), max(stamps)) if stamps else (now, now)
    since = led_from if since_days is None else max(led_from, now - since_days * 86400)

    shas = landing_shas(events)
    history = assign_arms(SC._git_log(repo, 0.0), shas)
    commits = [c for c in history if since <= c["ct"] <= now]
    human_reach = [c["ct"] for c in history if c["arm"] == HUMAN]
    # Commits older than the first ledger row are in no rate's k or n: before
    # the ledger exists there is nothing to pair them with.
    pre_comparable = ([c for c in history if c["ct"] < led_from] if stamps else [])
    pre_arms = sorted({c["arm"] for c in pre_comparable})
    agent_lands = [e for e in events if e.get("event") == "promoted"
                   and str(e.get("commit") or "") and since <= SC._ts(e) <= now]

    arms = {}
    for arm in (AGENT, HUMAN):
        mine = [c for c in commits if c["arm"] == arm]
        arms[arm] = {
            "arm": arm,
            "commits": len(mine),
            "first_commit": (_iso(min(c["ct"] for c in mine)) if mine else None),
            "last_commit": (_iso(max(c["ct"] for c in mine)) if mine else None),
            "landings": (len(agent_lands) if arm == AGENT else 0),
            "later_undo_7d": later_undo(commits, repo=repo, arm=arm,
                                         now=now, days=horizon_days),
            "revert_commits": revert_rate(commits, arm=arm),
            "guardian_rollbacks": (rollback_rate(events, since=since, until=now)
                                   if arm == AGENT else rate(0, 0)),
            "rounds_to_land": (rounds_to_land(events, landed_rounds=_landed_rounds(events))
                               if arm == AGENT else rate(0, 0)),
        }

    strata: dict[str, dict[str, int]] = {}
    for c in commits:
        w = strata.setdefault(_week(c["ct"]), {"agent": 0, "human": 0})
        w[c["arm"]] += 1
    rollback_triggers = sorted({str(e.get("trigger") or "?") for e in events
                                if e.get("event") == "rollback_succeeded"})

    landed_rounds = {str(e.get("round_id")) for e in events
                     if e.get("event") == "promoted" and e.get("round_id")
                     and since <= SC._ts(e) <= now}
    join = run_round_join(landed_rounds, db_path=runs_db, n_landings=len(landed_rounds))
    if runs_db is None:
        join = _empty_join("no runs db read (pass runs_db= to measure the join)")
        join["n"] = len(landed_rounds)
    # A yardstick, never a verdict: `ledger=None` means no yardstick, so a unit
    # caller never reaches the production ledger by omission, and `main()` names
    # the file it read. A ledger that will not parse costs the line, not the row.
    median_full_gate_s: float | None = None
    if ledger is not None:
        try:
            median_full_gate_s = SC._full_gate_median(Path(ledger))
        except Exception:        # noqa: BLE001 — a yardstick is never the report
            median_full_gate_s = None
    cost = cost_per_landing(events, since=since, now=now, join=join,
                            median_full_gate_s=median_full_gate_s)
    latency = filing_latency(events, since=since, now=now, ledger_from=led_from,
                             backlog_dir=backlog_dir)
    return {
        "generated_at": _iso(now),
        "window": {"from": _iso(since), "to": _iso(now),
                   "days": round((now - since) / 86400, 2),
                   "ledger_from": (_iso(led_from) if stamps else None),
                   "ledger_to": (_iso(led_to) if stamps else None),
                   "ledger_rows": len(events),
                   # The arms overlap only from the first ledger row: the human
                   # arm's history before it has no counterpart at all. The
                   # reach is the oldest HUMAN-arm commit of the WHOLE of
                   # `main`'s history — a window-filtered list, or a min over
                   # both arms, cannot report it (#1870).
                   "comparable_from": (_iso(led_from) if stamps else None),
                   "human_history_from": (_iso(min(human_reach))
                                          if human_reach else None),
                   "pre_comparable_excluded": len(pre_comparable),
                   "pre_comparable_single_arm": (pre_arms[0]
                                                 if len(pre_arms) == 1 else None)},
        "by_author": arms,
        "cost": cost,
        "latency": latency,
        "run_join": join_view(join),
        "week_strata": [{"week": k, **v} for k, v in sorted(strata.items())],
        "rollback_triggers": rollback_triggers,
        "true_positives": None,
        "notes": ["later_undo_7d is one definition run on both arms (scorecard row 5's: "
                  "added lines removed by a later commit of the other arm within 7 d), so "
                  "the two intervals compare",
                  "rounds_to_land characterises the agent arm only: a person's commit is "
                  "gated by nothing, so its row is n/a (no gate), not 0",
                  "true_positives is null on purpose: no rollback has been adjudicated "
                  "as a real catch, and 0/N would be a claim",
                  "gpu_hours_per_landing is wall-clock/3600 over ledger rung seconds only, "
                  "the same proxy the autonomy page prints; compute a landing CAUSED in a "
                  "later nightly job, and the human reading it, is outside the automod "
                  "source and out of every number here",
                  "cost-per-landing is gameable by attempting less: it is never printed "
                  "without never_landed_share beside it, and is withheld outright when the "
                  "run join sees under half of the landings",
                  "no model call, nothing written inside the checkout"],
    }


def join_view(join: dict[str, Any]) -> dict[str, Any]:
    """The join as it goes into the row: its rate, its floor, and what it could
    not attribute. `seconds` is dropped — per-round figures belong to `cost`."""
    return {"k": join["k"], "n": join["n"], "share": join["share"],
            "floor": join["floor"], "strong": bool(join["strong"]),
            "available": bool(join["available"]), "why": join.get("why"),
            "db": join.get("db"), "attributed_runs": int(join.get("runs") or 0),
            "ambiguous_runs": int(join.get("ambiguous_runs") or 0),
            "runs_naming_rounds": int(join.get("runs_naming_rounds") or 0),
            "rows_scanned": int(join.get("rows_scanned") or 0)}


def render(row: dict[str, Any]) -> str:
    w = row["window"]
    single = w["pre_comparable_single_arm"]
    excluded_line = (f"{w['pre_comparable_excluded']} commits on main before the "
                     f"first ledger row are excluded from every rate: "
                     + (f"single-arm ({single}), " if single else "")
                     + "no ledger counterpart to compare against")
    out = [f"landed-change outcomes by author — {w['from']} → {w['to']} "
           f"({w['days']} d, {w['ledger_rows']} ledger rows)",
           f"comparable from {w['comparable_from']} (first ledger row); "
           f"human arm history reaches {w['human_history_from']}",
           excluded_line,
           ""]
    out.append(f"{'measure':22s} {'agent':>34s}   {'human':>34s}")
    a, h = row["by_author"][AGENT], row["by_author"][HUMAN]
    out.append(f"{'commits':22s} {a['commits']:>34d}   {h['commits']:>34d}")
    out.append(f"{'landings':22s} {a['landings']:>34d}   {h['landings']:>34d}")
    out.append(f"{'later_undo_7d':22s} {rate_text(a['later_undo_7d']):>34s}   "
               f"{rate_text(h['later_undo_7d']):>34s}")
    out.append(f"{'revert_commits':22s} {rate_text(a['revert_commits']):>34s}   "
               f"{rate_text(h['revert_commits']):>34s}")
    out.append(f"{'guardian_rollbacks':22s} {rate_text(a['guardian_rollbacks']):>34s}   "
               f"{rate_text(h['guardian_rollbacks']):>34s}")
    out.append("")
    out.append(f"rounds_to_land  agent: {rounds_text(a['rounds_to_land'])}")
    out.append(f"                human: {rounds_text(h['rounds_to_land'])}")
    out.append("")
    out.extend(cost_lines(row["cost"]))
    out.append("")
    out.extend(latency_lines(row["latency"]))
    out.append("")
    out.append(f"rollback triggers (none adjudicated: true_positives null): "
               f"{', '.join(row['rollback_triggers']) or '—'}")
    out.append("per ISO week (agent/human commits):")
    for s in row["week_strata"]:
        out.append(f"  {s['week']}  agent {s['agent']:>5d}   human {s['human']:>5d}")
    for n in row["notes"]:
        out.append(f"- {n}")
    return "\n".join(out)


def landed_outcomes_path() -> Path:
    from scripts.automod import state as S
    # Beside scorecard.jsonl, in the automod state dir — never inside the
    # checkout, where the guardian alerts on runtime data hourly.
    return S.STATE_DIR / "landed_outcomes.jsonl"


def record(row: dict[str, Any], path: Path | None = None) -> Path:
    path = path or landed_outcomes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Landed-change outcomes split by author (read-only)")
    ap.add_argument("--since", default="all",
                    help="window: all (the ledger's span), 21d, 36h, 2w")
    ap.add_argument("--json", action="store_true", help="print the row as JSON")
    ap.add_argument("--record", action="store_true",
                    help=f"append the row to {landed_outcomes_path()}")
    args = ap.parse_args(argv)
    from scripts.automod import state as S
    days = None if args.since.strip().lower() in ("all", "0") else SC.parse_since(args.since)
    events = S.ledger_rows(S.LEDGER_PATH)
    row = by_author(events=events, since_days=days, ledger=S.LEDGER_PATH,
                    runs_db=_workers_db())
    print(json.dumps(row, indent=2, sort_keys=True) if args.json else render(row))
    if args.record:
        print(f"\nrecorded → {record(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
