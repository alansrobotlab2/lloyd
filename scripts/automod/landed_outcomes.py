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
summed into `gpu_hours` per task by `run_rollup_by_source()` in `workers/queue.py` — so
a three-hour implementer turn that lands nothing was one more run inside a budget, and the
program could answer "were landed changes undone?" but never "what did a
landing cost, and how long from the signal to the landing?":

  gpu_hours_per_landing   wall-clock seconds/3600 — the autonomy page's own
                          proxy, under that name — summed over the ledger's rung
                          and landing-wait seconds of promoted rounds, per
                          landing, never printed without the next line
  never_landed_share      the seconds sitting in rounds that produced no
                          `promoted` row: cost-per-outcome is gameable by
                          attempting less, and this is the line that shows it
  run_join                a `runs` row is attributed by its structured
                          `meta_json.round_id` when it carries one, and by the
                          free-text `summary` scan otherwise (#2573) — most rows
                          predate the key, so the match rate prints with the
                          figure and the figure is withheld below `JOIN_FLOOR`
                          instead of printed low
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


# ── the ledger, read whole ────────────────────────────────────────────────
#
# `scripts/automod/state.py::ledger_rows` decodes ONE path (`read_text`, no gzip),
# and the groundskeeper retention sweep rolls old rows out of the live file into
# `promotions-archive-<YYYYMM>.jsonl.gz` beside it on its own schedule
# (`scripts/groundskeeper/retention-sweep.py:2293` removes what it rolled). So a
# reader naming one file does not read a smaller window: it reads a population that
# no longer contains the events it exists to count. Measured on the live state dir
# 2026-10-20 15:00Z: live 21,254 rows, archives 26,417 + 7,547, merged 55,218 — and
# ALL ELEVEN `rollback_succeeded` rows are in `promotions-archive-202609.jsonl.gz`,
# zero in the live file. The shipped report therefore printed `guardian_rollbacks
# 0/516 (0.0 %)` with `rollback triggers …: —` while the same functions over the
# merged corpus read 9/1,293 with triggers `crash, data_damage, error_rate,
# promote_failed, regression`, under a header that named the merged-looking span
# `2026-09-06T18:19:23Z → 2026-10-20T14:59:48Z (43.86 d, 21252 ledger rows)`. A
# swept ledger rendered as a quiet period. #2858 fixed exactly this for the
# gate-to-land lag reader (`scripts/maintenance/gate_to_land_lag.py:130`); this is
# the same reader for the same reason, and every ledger read in this module goes
# through it.
#
# The glob is the point: a reader that hard-coded one month, or the live file,
# silently redefines its own population. Dedupe is by canonical row, so a row the
# sweep rolled but did not finish removing from the live file counts once (0 such
# rows today — 55,218 merged vs 55,218 unique — so this guards a partial roll, not
# a live defect). Decodes are cached per file on (mtime_ns, size, inode) exactly as
# `state._decode_ledger` does, so one report run is one read of each file and
# `tests/test_backlog_ledger_cache.py`'s "1 read per cycle" property holds.
LEDGER_CACHE: dict[str, tuple[int, int, int, list[dict]]] = {}


def ledger_files(ledger: Path | None = None) -> list[Path]:
    """The live ledger plus every archive beside it, oldest month first.

    `ledger` defaults to the live state file; a fixture passes its own live path and
    the archives are globbed from ITS directory, which is what makes a hermetic
    test able to prove an archived-only row reaches the report.
    """
    from scripts.automod import state as S
    live = Path(ledger or S.LEDGER_PATH)
    return [live, *sorted(live.parent.glob("promotions-archive-*.jsonl.gz"))]


def _decode_one(path: Path) -> list[dict]:
    """Decode one ledger file, cached on its identity, gzip transparently.

    A plain JSONL file goes through `state.ledger_rows`, the cached single-flight
    reader, so the merged read keeps #1204's one-decode-per-cycle property that
    `state.ledger_read_count` instruments and `tests/test_backlog_ledger_cache.py`
    asserts; the gzip branch has to be ours because `state._decode_ledger` is
    `read_text` and cannot open one at all — which is exactly why the archives were
    invisible to this report (#3079).
    """
    import gzip

    from scripts.automod import state as S

    if not str(path).endswith(".gz"):
        return S.ledger_rows(path)

    try:
        st = path.stat()
        text = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
    except OSError:
        return []
    key = str(path)
    hit = LEDGER_CACHE.get(key)
    if hit and (hit[0], hit[1], hit[2]) == (st.st_mtime_ns, st.st_size, st.st_ino):
        return hit[3]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue        # a truncated final line is normal in an appended JSONL
        if isinstance(row, dict):
            rows.append(row)
    LEDGER_CACHE[key] = (st.st_mtime_ns, st.st_size, st.st_ino, rows)
    return rows


def read_ledger(ledger: Path | None = None) -> tuple[list[dict], dict[str, Any]]:
    """The whole promotions ledger as it exists now: live file + every archive.

    Returns `(rows, population)`. `population` is what the header and the recorded
    row must carry, because `len(rows)` alone cannot tell a reader whether the live
    file had been swept: `ledger_rows` (what the report is computed over),
    `live_rows`, `archived_rows`, `archive_files` and `archive_names`.
    """
    files = ledger_files(ledger)
    out: list[dict] = []
    seen: set[str] = set()
    live_rows = 0
    per_file: dict[str, int] = {}
    for i, f in enumerate(files):
        rows = _decode_one(f)
        kept = 0
        for row in rows:
            try:
                key = json.dumps(row, sort_keys=True, default=str)
            except (TypeError, ValueError):
                key = repr(sorted(row.items(), key=lambda kv: str(kv[0])))
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            kept += 1
        per_file[f.name] = kept
        if i == 0:
            live_rows = kept
    archive_names = [f.name for f in files[1:] if per_file.get(f.name)]
    return out, {
        "ledger_rows": len(out),
        "live_rows": live_rows,
        "archived_rows": len(out) - live_rows,
        "archive_files": len(archive_names),
        "archive_names": archive_names,
        "ledger_files": [f.name for f in files],
    }


def ledger_population(events: list[dict],
                      population: dict[str, Any] | None = None) -> dict[str, Any]:
    """The population a report was computed over, stated as honestly as its caller
    can state it.

    A caller that read the ledger itself passes what `read_ledger` returned and the
    split is exact. A unit caller that hands over a hand-built `events` list has no
    files behind it, and saying `archived_rows: 0` there would be the same lie in
    the other direction — so the split is reported as `None`, unread, while
    `ledger_rows` is always the rows actually counted.
    """
    if population is not None:
        return dict(population)
    return {"ledger_rows": len(events), "live_rows": None, "archived_rows": None,
            "archive_files": None, "archive_names": [],
            "ledger_files": ["caller-supplied rows"]}


def population_line(pop: dict[str, Any]) -> str:
    """`21,254 ledger rows` becomes `55,218 ledger rows (21,254 live + 33,964 in 2
    archive file(s))`. A pruned live file cannot then render as the whole ledger."""
    n = pop.get("ledger_rows")
    arch = pop.get("archived_rows")
    if not isinstance(arch, int):
        # Rows the caller handed over without saying where they came from: say so, in the
        # header, rather than printing a bare count that reads as "the whole ledger".
        return f"{n} ledger rows (archived-row split not recorded)"
    if arch == 0:
        return f"{n} ledger rows (no archive rows present)"
    return (f"{n} ledger rows ({pop.get('live_rows')} live + {arch} archived in "
            f"{pop.get('archive_files')} file(s))")


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
# `runs.duration_seconds` summed and divided by 3600 into `gpu_hours` per TASK by
# `run_rollup_by_source()` in `workers/queue.py`, and `app/autonomy.py` (`:5157` at filing)
# summing the same quotient per run per task. A task that burns three hours and lands
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
#: `duration_seconds / 3600` is what the autonomy page prints as `gpu_hours`, via
#: `run_rollup_by_source()` in `workers/queue.py`. It is wall-clock and it
#: carries gate-rung subprocess time inside the same run, so the column says that rather than
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


def _meta_round_id(meta_json: Any) -> str | None:
    """The round a run row records STRUCTURALLY, or None (#2573).

    `workers.pool.normalize_result` hoists a source's reported round into
    `meta_json.round_id`, so a run that opened a round states which one and this
    needs no prose. Returns None whenever the row does not say — absent key, a
    blank or non-string value, `meta_json` that is not a JSON object, or a row
    written before the key existed — and the caller then falls back to the
    regex scan. The fallback is the whole point: 1,189 autocode rows were recorded
    without the key, and a join that read only the new field would report this
    week's cost as near-zero and call it an improvement.
    """
    if not meta_json:
        return None
    try:
        meta = json.loads(str(meta_json))
    except (TypeError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    rid = meta.get("round_id")
    return rid.strip() if isinstance(rid, str) and rid.strip() else None


def run_round_join(round_ids: set[str], *, db_path: Path | None = None,
                   n_landings: int | None = None) -> dict[str, Any]:
    """Which of `round_ids` at least one `runs` row names, and their seconds.

    A row is attributed by its structured `meta_json.round_id` when it has one
    (#2573) and that id alone; otherwise the carrier is still the free-text
    `runs.summary` (#1188's note in `workers/sources/autocode.py` writes it
    there), so the join falls back to one scan of `runs`, a round-id regex over
    `summary`/`response_json`/`meta_json`. Then a floor on the resulting match
    rate — not a fuzzy time window.

    Structured first is not cosmetic: a run's own report often quotes another
    round — the one it was sent back from — and a row whose prose names two is
    dropped rather than split, which is what `ambiguous_runs` counts. Where the
    row states its round structurally, that key is what the source knew, so it
    wins and the row is attributed instead of dropped. Autocode is the only
    source that reports a round today, so the other sources' ambiguous rows stay
    ambiguous: this removes the ambiguity for the rows that say, not for the ones
    that merely mention.

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
        structured = _meta_round_id(meta_json)
        if structured is not None:
            found.add(structured)
        else:
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
                   backlog_dir: Path | None = None,
              human_input: dict[str, Any] | None = None) -> dict[str, Any]:
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
              backlog_dir: Path | None = None,
              population: dict[str, Any] | None = None,
              human_input: dict[str, Any] | None = None) -> dict[str, Any]:
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

    `population` is the second half of `read_ledger`'s answer — how many of the
    rows came from the live file and how many from `promotions-archive-*.jsonl.gz`.
    It rides the row and the header because `ledger_rows` alone cannot say whether
    the live file had been swept: a report that reads 21,254 of 55,218 rows and
    prints the number without the split is the report that called 11 archived
    rollbacks a zero (#3079). Absent from a unit caller, the split is recorded as
    unread rather than as zero.
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
                   # …and the population the window's rows were drawn from: live vs
                   # rolled-into-archive. `ledger_rows` is the merged count; the split
                   # is what stops a swept live file reading as a short ledger.
                   **ledger_population(events, population),
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
        # The other half of the bucket (#2661), nested rather than spread through
        # this row so no existing author-split key moves: a row re-read next month
        # has to line up with the ones already archived. `None` when no corpus was
        # read, so a caller that did not ask for the block pays nothing for it.
        "human_input": human_input,
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
           f"({w['days']} d, {population_line(w)})",
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
    out += render_human_input(row.get("human_input"))
    return "\n".join(out)




def _session_stamps(data_root: Path | None,
                    corpus: list[tuple[str, str | None, str, float | None]]
                    ) -> dict[str, float]:
    """Each session id in `corpus` → the epoch its OWN document is stamped at.

    A message's own `timestamp` is the primary stamp (59 of 59 live user messages
    sampled 2026-10-13 carry one, and none carries `ts`); this is the FALLBACK for a
    message that has none, and the session document's own `last_active` — read through
    `uptake._last_active_epoch`, the same key the retention sweep ages on — is the only
    time available then. Read only for the sessions the corpus names, so the walk is
    bounded by the corpus rather than by the store; a session with no readable stamp
    yields nothing, and its turns are reported as `undated_turns` rather than given an
    invented time.
    """
    root = Path(data_root) if data_root else uptake.lloyd_root()
    want = {sid for sid, _c, _x, _t in corpus}
    out: dict[str, float] = {}
    for path in uptake._session_transcripts(root):
        sid = uptake.session_id(path)
        if sid not in want:
            continue
        doc = uptake.read_session(path)
        if not isinstance(doc, dict):
            continue
        t = uptake._last_active_epoch(doc)
        if t is not None:
            out[sid] = t
    return out

# ── the human input beside the compute (2026-10-13, #2661) ───────────────────
#
# `by_author` above prices a landing in tokens, seconds and GPU-hours, and its own
# scoping note says the human reading a landing "is outside the automod source and
# out of every number here". That was the honest boundary at the time; this is the
# other half of the bucket. The move worth recreating is not a dollar split — dollars
# are unavailable HERE (`cost_usd` is zero-filled on every usage row, so no key in
# this file carries one and none is added) — it is putting the scarce unmeasured
# input in the same bucket as the machine cost and dividing by the achieved outcome.
# On this box that input is Alan's input, counted in turns and decisions because
# those are not commensurable with GPU-hours: two columns, never one summed "total
# cost", which would be a fabrication.
#
# Two rules this block keeps from the file it lives in. The denominator is an
# ACHIEVED outcome, and one landing writes two ledger rows (`promoted` carries
# `round_id` and no `item_id`; `item_landed` carries both), so the denominator
# dedupes on the round and prints both row counts rather than adding them — the live
# ledger is asymmetric (554 `promoted` against 1,089 `item_landed` rows measured
# 2026-10-13), so a sum is wrong in a different direction every day. And the window
# is the one the corpus COVERS, not the one asked for, the way `comparable_from`
# labels the ledger's own start (#2510's `MIN(completed_at)` lesson).

from app import uptake

#: The reader of record for a human turn. Bound, not copied: uptake filters a
#: transcript corpus already (#1848 is the price of the second reader that used to
#: live in `uptake_probe`), and #2661's step 1(b) is "reuse it, do not write a
#: parallel report". `human_turn_sources` is its unfiltered twin, so this block can
#: count what the filter EXCLUDES instead of silently dropping it.
HUMAN_TURNS_READER = uptake.human_turn_sources

#: Printed verbatim in place of a number whenever no human-event class reaches the
#: attribution floor. A made-up attention figure is worse than the honest absence it
#: replaces, so the text IS the answer, not an apology for one.
ATTENTION_WITHHELD = "attention budget not derivable from these joins"

#: The item's own kill switch, applied to the share of human turns that name an
#: achieved outcome.
ATTENTION_FLOOR = 0.5

#: The eight-to-twelve in `skills/backlog-triage/SKILL.md` Step 2, carried as a
#: quoted number so the derived read can be stated AGAINST it. The cap is a human
#: ruling: nothing here edits the skill, and no gate reads this constant.
PROMOTION_CAP_CARRIER = "skills/backlog-triage/SKILL.md"
PROMOTION_CAP_QUOTE = (8, 12)

#: The ledger classes an escalation count is printed for. `ambient` is NOT one of
#: them: it is delivery volume, and a MUTED injection persists no message at all, so
#: the store cannot see non-delivery — which is why it prints beside the decision
#: classes and is never added to them.
HUMAN_EVENT_CLASSES = ("alert", "hand_settle", "human_paths", "review_escalated")
ESCALATION_VOLUME_CLASS = "ambient"

#: The ledger events that DISPOSITION an item — the records the answered join reads.
#: Every name here is a live writer's own event with rows on the ledger (measured
#: 2026-10-13 over 22,007 rows: `status_moved` 253, `backlog_retriage` 138,
#: `item_landed` 1,090, `item_closed` 28, `hand_settle` 21, `settled` 539), and the
#: join reaches a real share of each class: 12 of 13 `review_escalated` rows, 21 of 24
#: `human_paths` rows. `escalation_resolved` is deliberately NOT here — it has 0 rows
#: in those 22,007 and no writer anywhere in the tracked tree
#: (`git grep -n escalation_resolved`), so an answer read through it is structurally
#: zero on live traffic and would read as "Alan never answers", which is a claim about
#: a person rather than a measurement (#2661's review, clause 3).
ANSWER_DISPOSITION_EVENTS = ("status_moved", "backlog_retriage", "item_landed",
                            "item_closed", "hand_settle", "settled")
#: The tighter join, for a path a round left owed to a person: the item has to have
#: REACHED an outcome, not merely moved.
ANSWER_OUTCOME_EVENTS = ("item_landed", "item_closed")
#: How long #2661's step 2 gives an escalation to be answered by a human turn:
#: "guardian 'needs a human' escalations and whether any human turn followed within a
#: window". A guardian `alert` row carries no `item_id` and no `round_id` (all 25 live
#: rows), so no ledger join is possible for it and this is the only join the stores
#: allow. Measured live 2026-10-13: 25 of 25 alerts were followed by a human turn
#: inside this window, which is exactly the fact #593 could not see.
ANSWER_WINDOW_S = 86_400
#: Where an answer came from, so a reader never mistakes a transcript join for a
#: ledger record, or a structurally-unanswerable class for an ignored one.
ANSWER_KINDS = ("ledger_record", "transcript_join", "is_the_answer", "volume_only")

#: What the two human counts are, spelled out so no reader mistakes them for one
#: unit, and so nobody adds them to a GPU-hour.
HUMAN_UNIT = "human-originated user turn + human decision event"


def _week_key(ts: float) -> str:
    """ISO `YYYY-Www`, the same week key `week_strata` prints."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%G-W%V")


def _in(t: float | None, lo: float | None, hi: float | None) -> bool:
    return t is not None and (lo is None or t >= lo) and (hi is None or t <= hi)


def _landing_key(e: dict) -> str:
    """One achieved outcome, whichever of the two rows is being looked at."""
    return str(e.get("round_id") or f"item:{e.get('item_id')}"
               if (e.get("round_id") or e.get("item_id")) else f"ts:{SC._ts(e):.0f}")


def _outcomes(events: list[dict], *, lo: float | None, hi: float | None,
              now: float) -> dict[str, Any]:
    """Achieved outcomes in the window, deduped on the landing, with both row counts.

    Counted as the UNION of `promoted` and `item_landed` keyed on the round: 554
    promoted rows against 1,089 item_landed rows means an item can land with no
    promotion row, so the union is the only denominator that is neither doubled nor
    short.
    """
    keys: dict[str, float] = {}
    promoted = landed = 0
    for e in events:
        ev = e.get("event")
        if ev not in ("promoted", "item_landed"):
            continue
        t = SC._ts(e)
        if not t or t > now:
            continue
        if ev == "promoted":
            promoted += 1 if _in(t, lo, hi) else 0
        else:
            landed += 1 if _in(t, lo, hi) else 0
        if _in(t, lo, hi):
            keys.setdefault(_landing_key(e), t)
    return {"outcomes": len(keys), "outcomes_promoted": promoted,
            "outcomes_item_landed": landed, "keys": keys,
            "stamps": sorted(keys.values())}


def _escalations(events: list[dict], *, lo: float | None, hi: float | None, now: float,
                 ambient_in_window: int,
                 human_stamps: list[float] | None = None) -> dict[str, Any]:
    """Human decision and escalation events, one line per class, never summed.

    Each class is a DIFFERENT act — a guardian alarm, a settlement someone wrote by
    hand against an item, the path a round refused and left owed to a person, a round
    a review sent to a decider — and a single "total" would add an alarm to a ruling to
    a delivery count and call it attention. That is also the number gameable in the
    worst direction (an agent that stops surfacing things lowers its own measured
    attention cost), so no line here prints a sum: `escalations` is the one
    measurement, and the flat `human_event_opened_by_class` is read back out of it
    rather than counted a second time.

    `answered` names the record it came from, per class, and `answer_kind` says where
    that record lives:

    * `review_escalated` / `human_paths` — `ledger_record`. Joined on `item_id`, not
      `round_id`, because 6 of the 24 live `human_paths` rows carry a null `round_id`
      and a round-keyed join silently loses that quarter of the class. The joining
      record must be AT OR AFTER the escalation, so a landing from before it cannot
      answer it.
    * `hand_settle` — `is_the_answer`. Its own `outcome` field IS the disposition
      (live vocabulary measured 2026-10-13 over its 21 rows: `close` 9, `reopen` 9,
      `ruling,reopen` 1, `settled,recheck,recheck` 1, `settled,close` 1), so there is
      no answer to look for and printing one would double-count the act. It carries
      `outcome`, never `decision`.
    * `alert` — `transcript_join`. Its rows carry no `item_id` and no `round_id`, so
      the only answer the stores can show is the item's own definition: did a
      human-originated turn follow within `ANSWER_WINDOW_S`. The complement is printed
      as `stalled_no_human_turn_24h`, which is the instrument for #593's finding
      (escalations going unanswered "with no human turn in any cycle").
    * `ambient` — `volume_only`, printed beside the classes and never added to one.

    A zero therefore always arrives with the record that would have made it nonzero:
    `answered 0/2 — a status_moved|… row naming the same item_id at or after it` is a
    measurement, while a bare `answered 0` reads as an accusation about a person.
    """
    human_stamps = sorted(human_stamps or [])

    def dedupe(rows: list[dict]) -> dict[str, dict]:
        got: dict[str, dict] = {}
        for e in rows:
            got.setdefault(_landing_key(e) if (e.get("round_id") or e.get("item_id"))
                           else f"row{SC._ts(e):.0f}", e)
        return got

    def later_record(item_id: object, ts: float, names: tuple[str, ...]) -> dict | None:
        """The first record of `names` that dispositions `item_id` at or after `ts`.

        `None` is an absence, not a zero: an escalation whose item has moved on carries
        no answer, and an escalation about an item that has not moved carries the same
        nothing — which is why every caller also prints the join it used.
        """
        if item_id is None:
            return None
        for e in events:
            if e.get("item_id") == item_id and e.get("event") in names \
                    and SC._ts(e) >= ts:
                return e
        return None

    esc: dict[str, dict[str, Any]] = {}
    outside: dict[str, int] = {}
    for name in HUMAN_EVENT_CLASSES:
        every = dedupe([e for e in events
                        if e.get("event") == name and SC._ts(e) <= now])
        inw = [e for _k, e in every.items() if _in(SC._ts(e), lo, hi)]
        outside[name] = len(every) - len(inw)
        row: dict[str, Any] = {"opened": len(inw), "answered": None}
        if name == "hand_settle":
            outcomes: dict[str, int] = {}
            for e in inw:
                got = str(e.get("outcome") or "(no outcome)")
                outcomes[got] = outcomes.get(got, 0) + 1
            row["answer_kind"] = "is_the_answer"
            row["answer_join"] = ("no answer sought: the row IS the settlement, whose own "
                                  "`outcome` field carries the disposition")
            row["outcomes"] = dict(sorted(outcomes.items()))
        elif name == "review_escalated":
            row["answered"] = sum(1 for e in inw
                                  if later_record(e.get("item_id"), SC._ts(e),
                                                  ANSWER_DISPOSITION_EVENTS))
            row["answer_kind"] = "ledger_record"
            row["answer_join"] = ("a " + "|".join(ANSWER_DISPOSITION_EVENTS)
                                  + " row naming the same item_id at or after it")
            row["routed_to_named_decider"] = sum(1 for e in inw if e.get("decider"))
        elif name == "human_paths":
            row["answered"] = sum(1 for e in inw
                                  if later_record(e.get("item_id"), SC._ts(e),
                                                  ANSWER_OUTCOME_EVENTS))
            row["answer_kind"] = "ledger_record"
            row["answer_join"] = ("an " + "|".join(ANSWER_OUTCOME_EVENTS)
                                  + " row naming the same item_id at or after it "
                                  "(joined on item_id: round_id is null on 6 of the 24 "
                                  "live rows)")
        else:  # alert — no item_id, no round_id, so no ledger join exists to make
            ans = sum(1 for e in inw
                      if any(SC._ts(e) <= t <= SC._ts(e) + ANSWER_WINDOW_S
                             for t in human_stamps))
            row["answered"] = ans
            row["answer_kind"] = "transcript_join"
            row["answer_join"] = ("no ledger record answers an alert (its rows carry no "
                                  "item_id or round_id), so the answer is a "
                                  f"human-originated turn within {ANSWER_WINDOW_S // 3600} h")
            row["stalled_no_human_turn"] = len(inw) - ans
        esc[name] = row
    esc[ESCALATION_VOLUME_CLASS] = {
        "opened": ambient_in_window, "answered": None,
        "answer_kind": "volume_only",
        "answer_join": ("not an escalation: delivered injection volume; a muted "
                        "injection persists no message, so the store cannot see "
                        "non-delivery")}
    outside[ESCALATION_VOLUME_CLASS] = 0
    return {"escalations": esc,
            "human_event_opened_by_class": {k: v["opened"] for k, v in esc.items()},
            "classes_outside_window": outside,
            "note": ("counts are per class and are never summed; `answer_kind` names "
                     "whether the answer is a ledger record, a transcript join, the "
                     "row itself, or nothing at all")}


def _attribution(human_rows: list[tuple[str, float]],
                 keys: dict[str, float]) -> dict[str, Any]:
    """Do the counted human turns NAME one of the counted achieved outcomes?

    Two denominators were wrong in the first cut of this function and both are fixed
    by construction here, so this records why they cannot come back:

    * the numerator and denominator now come from the SAME rows the block prints —
      `human_rows` is the counted human turns, not every user message with no inject
      marker. Reading the unfiltered corpus instead made the live line claim 3640/8414
      human turns while the headline above it said 139: the extra 8,275 are turns in
      sessions uptake already excludes (autonomy runs, benches), and an attribution
      denominator that disagrees with the printed turn count makes the floor a
      measurement of the wrong corpus.
    * `keys` is the WINDOW's landing keys, the same set `outcomes` counts. Joining
      against every key the ledger holds while dividing by a windowed denominator is
      how the same line came out `951/873 outcomes are named = 1.0893` — a share above
      1, which is not a share. `named` is drawn from `keys`, so both shares here are
      at most 1 and the test node pins it.

    The test is by identifier, on the human turns only: a turn containing `#2661` or
    `SM_20261013_195356` is about an outcome whatever its intent — ruling, dismissal or
    question. That is the most generous join the stores allow, because a transcript
    carries no role label: a share that fails here fails for want of identifiers, not
    for want of interpretation. Both readings print, and the stricter governs, because
    clearing only one of them is the ambiguity a made-up attention figure would hide
    behind. Measured live 2026-10-13: 0 of the 139 counted turns names an outcome,
    which is why the withheld text is the live answer.
    """
    round_re = re.compile(r"\b(SM_\d{8}_\d{6})\b")
    item_re = re.compile(r"#(\d{2,6})\b")
    #: `promoted` rows key on the round, `item_landed` rows on `item:NNNN` (see
    #: `_landing_key`); a human types `#2401`, not `item:2401`, so both spellings of a
    #: key are looked up while the answer stays the key itself.
    want = {(k[5:] if k.startswith("item:") else k): k for k in keys}
    matched = 0
    named: set[str] = set()
    for text, _t in human_rows:
        hits = ({*round_re.findall(text or ""), *item_re.findall(text or "")}
                & set(want))
        if hits:
            matched += 1
            named |= {want[h] for h in hits}
    human_n = len(human_rows)
    outcomes = len(keys)
    share = round(matched / human_n, 4) if human_n else None
    outcome_share = round(len(named) / outcomes, 4) if outcomes else None
    return {"attribution_share": share, "attributable_turns": matched,
            "n_human_turns": human_n, "attribution_outcome_share": outcome_share,
            "attribution_outcomes_named": len(named), "attribution_outcomes": outcomes,
            "attribution_floor": ATTENTION_FLOOR,
            "clears": bool(share is not None and outcome_share is not None
                           and share >= ATTENTION_FLOOR
                           and outcome_share >= ATTENTION_FLOOR)}


def human_input_block(*, now: float | None = None, days_requested: float = 30,
                      data_root: Path | None = None, ledger: Path | None = None,
                      corpus: list[tuple[str, str | None, str, float | None]] | None = None,
                      events: list[dict] | None = None) -> dict[str, Any]:
    """The human side of the bucket, over the window the corpus actually covers.

    Two reads of the transcripts, both uptake's, each used for what only it can give:

    * `uptake.human_turns` — the human TURN COUNT, and the window. Its rows also carry
      the turn's text, which is what the attribution join is run over, so the
      attribution denominator is the same set of turns the headline prints.
    * `corpus`, `HUMAN_TURNS_READER`'s output (`uptake.human_turn_sources`) —
      `(session_id, source, text, ts)` for every persisted user message, including the
      injected ones `human_turns` drops. Only the injected rows are read from it, as the
      volume each injector writes keyed by its message `source`; a `None` source there is
      uptake's "no inject marker", never added to a human count.

    `days_requested` is quoted, never applied to the transcripts: the window is the
    covered span of the human turns, which is shorter than any number asked for until the
    corpus ages that far. Nothing here reads the checkout or writes anything: it is a
    read of transcripts, the ledger and (through uptake) nothing else.
    """
    now = now if now is not None else datetime.now(timezone.utc).timestamp()
    # The merged read, same reason as `main()`: this block's ledger half counts
    # `backlog_item_filed` rows, and a row the sweep rolled out of the live file is
    # still a filing that happened (#3079).
    events = read_ledger(ledger)[0] if events is None else events
    if corpus is None:
        # `data_root` is the DATA root: uptake's own enumeration resolves
        # `<root>/sessions/*.json[.gz]`, so the root is handed over as-is and a
        # test points the whole block at a fixture tree.
        corpus = HUMAN_TURNS_READER(root=data_root, days=None)
    # The human count comes from `human_turns` itself — the authoritative reader,
    # which also drops non-interactive sessions and scheduler-prefixed turns. The
    # unfiltered corpus is used only for the volumes it cannot give: how much of
    # each injected class there was. Two counts, one reader each, never a second
    # filter that could drift from the first (#1848).
    # `days` is an mtime pre-filter, not a window: 100_000 days is every transcript
    # on the box, and the window below is the corpus's own covered span.
    turns = uptake.human_turns(root=data_root, days=100_000)
    session_stamp = _session_stamps(data_root, corpus)
    human = [(t.session, t.ts, session_stamp.get(t.session), t.user_text)
             for t in turns]
    dated = [(sid, (uptake._msg_epoch(ts) if ts else fallback), text)
             for sid, ts, fallback, text in human]
    undated = len(human) - sum(1 for _s, t, _x in dated if t is not None)
    #: The block's ONE human-turn measurement: every count, ratio, escalation answer
    #: and attribution share below is read out of these (epoch, text) rows, so the
    #: headline n and the attribution denominator cannot be two different corpora.
    in_window = [(t, text) for _s, t, text in dated if t is not None and t <= now]
    counted = [t for t, _x in in_window]
    lo, hi = (min(counted), max(counted)) if counted else (None, None)
    n_human = len(in_window)
    injected: dict[str, int] = {}
    for _sid, source, _x, _t in corpus:
        if source is not None:
            injected[source] = injected.get(source, 0) + 1
    ambient = injected.get(ESCALATION_VOLUME_CLASS, 0)
    out = _outcomes(events, lo=lo, hi=hi, now=now)
    # The alert join reads the SAME stamp list the counts above were made from, so the
    # answered arm and the turn count cannot disagree about when Alan was at the box.
    esc = _escalations(events, lo=lo, hi=hi, now=now, ambient_in_window=ambient,
                       human_stamps=counted)
    att = _attribution([(text, t) for t, text in in_window], out["keys"])
    weeks: dict[str, dict[str, int]] = {}
    for t, _x in in_window:
        weeks.setdefault(_week_key(t), {"human_turns": 0, "outcomes": 0})["human_turns"] += 1
    for t in out["stamps"]:
        weeks.setdefault(_week_key(t), {"human_turns": 0, "outcomes": 0})["outcomes"] += 1
    blk: dict[str, Any] = {
        "author_of_record": HUMAN,
        "human_unit": HUMAN_UNIT,
        "window_covered_start": _iso(lo) if lo else None,
        "window_covered_end": _iso(hi) if hi else None,
        "window_covered_days": (round((hi - lo) / 86_400, 2) if lo and hi else None),
        "window_requested_days": days_requested,
        "human_turns": n_human,
        "undated_turns": undated,
        "turn_stamp_source": ("message `timestamp`, else the session document's own "
                              "stamp; a turn with neither is an absence, never an "
                              "invented time"),
        "injected_turns_by_source": injected,
        "outcomes": out["outcomes"],
        "outcomes_promoted": out["outcomes_promoted"],
        "outcomes_item_landed": out["outcomes_item_landed"],
        "k_n": f"{n_human}/{out['outcomes']}",
        "touches_per_outcome": (round(n_human / out["outcomes"], 4)
                                if out["outcomes"] and n_human else None),
        "outcomes_per_touch": (round(out["outcomes"] / n_human, 4) if n_human else None),
        "weeks": [{"week": k, **v,
                   "touches_per_outcome": (round(v["human_turns"] / v["outcomes"], 3)
                                           if v["outcomes"] else None)}
                  for k, v in sorted(weeks.items())],
        "human_supply_turns_per_week": (round(n_human / max(1, len(weeks)), 2)
                                        if n_human else None),
        "attention_budget": None,
        "promotion_cap": {"quote": list(PROMOTION_CAP_QUOTE),
                          "carrier": PROMOTION_CAP_CARRIER,
                          "modified_by_this_code": False},
        **esc, **{k: v for k, v in att.items() if k != "clears"},
    }
    # The per-outcome ratio is the attribution-DEPENDENT one, so it is gated on the
    # join; `outcomes_per_touch` is not — both of its counts come straight off the
    # stores and are true with no attribution at all.
    blk["human_touches_per_outcome"] = (
        blk["touches_per_outcome"] if att["clears"] else None)
    if att["clears"]:
        blk["attention_budget"] = {
            "median_human_touches_per_promoted_item": blk["touches_per_outcome"],
            "measured_weekly_human_supply": blk["human_supply_turns_per_week"]}
    else:
        blk["attention_budget"] = ATTENTION_WITHHELD
        blk["withheld_reason"] = (
            "no human-event class reached the attribution floor on the attribution "
            "line above, so no per-outcome or attention figure is printed; the fleet "
            "counts stand")
    return blk


def attention_line(blk: dict[str, Any]) -> str:
    """The derived read on the promotion cap, or the literal that replaces it.

    Below the floor this returns ONLY the withheld text — no decimals, no "but
    roughly" — because the sentence's whole job is to be unquotable as a number.
    """
    if blk.get("attention_budget") == ATTENTION_WITHHELD or not isinstance(
            blk.get("attention_budget"), dict):
        return ATTENTION_WITHHELD
    per = blk["touches_per_outcome"]
    supply = blk["human_supply_turns_per_week"]
    lo, hi = PROMOTION_CAP_QUOTE
    return (f"{per} human turns per achieved outcome x the quoted cap {lo}-{hi} = "
            f"{round(per * lo, 1)}-{round(per * hi, 1)} turns/wk against a measured "
            f"supply of {supply} turns/wk; the cap itself is a human ruling in "
            f"{PROMOTION_CAP_CARRIER} and this code does not change it")


def attention_cap(*, up_next: int | None, cap: tuple[int, int] = PROMOTION_CAP_QUOTE,
                  n_human_turns: float | None = None) -> str:
    """The cap's own read, stated as a comparison and never as a target.

    The 2026-10-13 reading is `up_next` = 4 against a quoted 8-12, so the honest
    sentence is "cap not binding", not a recommendation to promote more.
    """
    n = "" if n_human_turns is None else f" n={n_human_turns}"
    if up_next is None:
        return f"cap {cap[0]}-{cap[1]}: up_next not read, so no comparison{n}"
    if up_next >= cap[0]:
        return (f"cap {cap[0]}-{cap[1]} reached: up_next={up_next}{n} — the queue is "
                f"at the cap, which is the only thing this instrument can say about it")
    return (f"cap {cap[0]}-{cap[1]}: cap not binding, up_next={up_next}{n} — the "
            f"measured queue is below the quoted number, so nothing here argues for "
            f"moving it")


def render_escalations(blk: dict[str, Any]) -> list[str]:
    """One line per class: opened, answered where a record answers it, and the join.

    The join is printed even when the answer is zero, because a bare `answered 0` is
    what reads as "Alan never answers" — the same figure with its join named is a
    statement about the ledger. Classes whose answer is not a lookup (`hand_settle`,
    whose own row is the disposition; `ambient`, which is volume) print their nature in
    place of a number rather than a zero that would misrepresent them.
    """
    lines: list[str] = []
    for cls, row in sorted((blk.get("escalations") or {}).items()):
        opened = row.get("opened")
        kind = row.get("answer_kind")
        ans = row.get("answered")
        bits = [f"  {cls} opened {opened}"]
        if isinstance(ans, int):
            bits.append(f" | answered {ans}/{opened}")
        elif kind == "is_the_answer":
            bits.append(" | the row IS the settlement")
        elif kind == "volume_only":
            bits.append(" | not an escalation")
        if kind:
            bits.append(f" ({kind})")
        if row.get("stalled_no_human_turn") is not None:
            bits.append(f" | stalled with no human turn in the window "
                        f"{row['stalled_no_human_turn']}")
        if row.get("routed_to_named_decider") is not None:
            bits.append(f" | routed to a named decider "
                        f"{row['routed_to_named_decider']}")
        if row.get("outcomes"):
            bits.append(" | outcomes " + ", ".join(f"{k}={v}"
                                                   for k, v in row["outcomes"].items()))
        bits.append(f" — join: {row.get('answer_join')}")
        lines.append("".join(bits))
    return lines


def render_human_input(blk: dict[str, Any] | None) -> list[str]:
    """The block's lines, appended under the author split. Empty without a block."""
    if not blk:
        return []
    out = ["", f"Human input (author of record: {blk.get('author_of_record')}, "
               f"n={blk.get('human_turns')})"]
    out.append(f"- window COVERED: {blk.get('window_covered_start')} → "
               f"{blk.get('window_covered_end')} = {blk.get('window_covered_days')} d "
               f"(asked for {blk.get('window_requested_days')} d; the corpus reach is "
               f"the window)")
    out.append(f"- human turns {blk.get('human_turns')} (n={blk.get('human_turns')}) | "
               f"human decision events by class (below) | achieved outcomes "
               f"promoted+item_landed {blk.get('outcomes')} (n={blk.get('outcomes')}) "
               f"[rows {blk.get('outcomes_promoted')} promoted / "
               f"{blk.get('outcomes_item_landed')} item_landed, deduped on the round]")
    out.append(f"- touches per outcome {blk.get('touches_per_outcome')} | outcomes per "
               f"touch {blk.get('outcomes_per_touch')} | human touches per outcome "
               f"{blk.get('human_touches_per_outcome')} (attribution-gated)")
    if blk.get("undated_turns"):
        out.append(f"- undated turns {blk.get('undated_turns')}: no message `ts` and no "
                   f"session stamp — counted as an absence, given no invented time")
    inj = ", ".join(f"{k}={v}" for k, v in sorted(blk.get("injected_turns_by_source", {}).items()))
    out.append(f"- injected turns (not a human act, never in the turn count): {inj or '(none)'}")
    out += render_escalations(blk)
    att = (f"- attribution: {blk.get('attributable_turns')}/{blk.get('n_human_turns')} "
           f"human turns name an achieved outcome = {blk.get('attribution_share')}, and "
           f"{blk.get('attribution_outcomes_named')}/{blk.get('attribution_outcomes')} "
           f"outcomes are named = {blk.get('attribution_outcome_share')}; floor "
           f"{blk.get('attribution_floor')}")
    out.append(att)
    out.append(f"- attention budget: {attention_line(blk)}")
    if blk.get("withheld_reason"):
        out.append(f"  ({blk['withheld_reason']})")
    cap = blk.get("promotion_cap") or {}
    out.append(f"- promotion cap {cap.get('quote')} quoted from "
               f"{cap.get('carrier')}: modified by this code = "
               f"{cap.get('modified_by_this_code')}")
    return out


def human_input_path() -> Path:
    from scripts.automod import state as S
    # Same rule as `landed_outcomes_path`: beside the outcome rows in the automod
    # state dir, never inside the checkout, where the guardian alerts hourly (#1667).
    return S.STATE_DIR / "human_input.jsonl"


def record_human_input(blk: dict[str, Any], path: Path | None = None) -> Path:
    path = path or human_input_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(blk, sort_keys=True) + "\n")
    return path


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
    ap.add_argument("--days", type=int, default=30,
                    help="transcript window for the human-input block; the block "
                         "prints the window that corpus actually COVERED, which is "
                         "shorter than any number asked for here (#2661)")
    ap.add_argument("--record", action="store_true",
                    help=f"append the row to {landed_outcomes_path()}")
    args = ap.parse_args(argv)
    from scripts.automod import state as S
    days = None if args.since.strip().lower() in ("all", "0") else SC.parse_since(args.since)
    # One read of the ledger as it actually exists — the live file plus every
    # `promotions-archive-*.jsonl.gz` beside it — shared by both blocks below, so
    # the rollback/cost/latency figures and the population in the header come from
    # the same rows. Reading only `promotions.jsonl` here is what printed
    # `guardian_rollbacks 0/516` with an empty trigger list while all 11
    # `rollback_succeeded` rows sat in the 202609 archive (#3079).
    events, population = read_ledger(S.LEDGER_PATH)
    now = datetime.now(timezone.utc).timestamp()
    # The block reads the transcript corpus, whose reach is shorter than any
    # `--days`; `human_input_block` prints the window it actually COVERED.
    blk = human_input_block(now=now, days_requested=float(args.days), events=events)
    row = by_author(now=now, events=events, since_days=days, ledger=S.LEDGER_PATH,
                    runs_db=_workers_db(), human_input=blk, population=population)
    print(json.dumps(row, indent=2, sort_keys=True) if args.json else render(row))
    if args.record:
        print(f"\nrecorded → {record(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
