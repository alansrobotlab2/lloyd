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

`n` is what the window can actually observe: a commit with less than 7 d of
later history cannot yet have been undone, so it is excluded from the
denominator rather than counted as a clean landing, and `0 landings` renders as
`0/0 (no landings)` rather than `0%` — the zero-denominator rule. The window is
the ledger's own span, so the arms are only comparable from the first ledger row
on (`2026-09-06` as measured at file time), with the human arm's older months
labelled as having no ledger counterpart.
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
              horizon_days: int = HUMAN_TOUCH_DAYS) -> dict[str, Any]:
    """One author-split report over `events` (the ledger) and `main`'s history.

    `since_days=None` means the whole ledger span, which is the widest window in
    which both arms have a counterpart; the human arm's older months are labelled
    as having no ledger at all rather than folded into the comparison.
    """
    repo = Path(repo or LIVE_ROOT)
    now = now or datetime.now(timezone.utc).timestamp()
    events = [] if events is None else list(events)
    stamps = [SC._ts(e) for e in events if SC._ts(e)]
    led_from, led_to = (min(stamps), max(stamps)) if stamps else (now, now)
    since = led_from if since_days is None else max(led_from, now - since_days * 86400)

    shas = landing_shas(events)
    commits = [c for c in assign_arms(SC._git_log(repo, since), shas)
               if since <= c["ct"] <= now]
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
    return {
        "generated_at": _iso(now),
        "window": {"from": _iso(since), "to": _iso(now),
                   "days": round((now - since) / 86400, 2),
                   "ledger_from": (_iso(led_from) if stamps else None),
                   "ledger_to": (_iso(led_to) if stamps else None),
                   "ledger_rows": len(events),
                   # The arms overlap only from the first ledger row: the human
                   # arm's history before it has no counterpart at all.
                   "comparable_from": (_iso(led_from) if stamps else None),
                   "human_history_from": (_iso(min(c["ct"] for c in commits))
                                          if commits else None)},
        "by_author": arms,
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
                  "no model call, nothing written inside the checkout"],
    }


def render(row: dict[str, Any]) -> str:
    w = row["window"]
    out = [f"landed-change outcomes by author — {w['from']} → {w['to']} "
           f"({w['days']} d, {w['ledger_rows']} ledger rows)",
           f"comparable from {w['comparable_from']} (first ledger row); "
           f"human arm history reaches {w['human_history_from']}",
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
    row = by_author(events=events, since_days=days)
    print(json.dumps(row, indent=2, sort_keys=True) if args.json else render(row))
    if args.record:
        print(f"\nrecorded → {record(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
