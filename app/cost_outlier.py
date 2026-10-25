"""Token-denominated cost-outlier trigger for the worker fleet (#2510).

The mechanism Palafox describes for GitHub Copilot pipeline agents: instrument
every run, and when a run costs much more than its own source usually costs, fire
an investigation that says *where the tokens went* rather than leaving a human to
read dashboards. This module is the trigger half — the replay that shows what
would have fired, the arming rule, and the one-job-per-cooldown enqueue. The
investigation itself is `workers/sources/cost_investigate.py`.

**Tokens, never dollars.** `cost_usd` is `0.0` on every one of the ~9,700 rows in
the usage store, and `app/usage_store.py` says why in its own skill-breakdown
docstring: the write sites put `0.0` in `cost_usd` on every row, so a per-skill
cost "would be a column of zeros dressed up as a cost. Tokens are the only
magnitude this store actually carries." A `cost_usd > P90` trigger on this box has
P90 = 0 and fires exactly zero times forever. The item's dollar framing is false
here; the numbers below are tokens (input + output).

**The margin is the whole design, and it was measured before it was chosen.** A
hold-out replay over the live stores (60% of each source's history as the
baseline, the last 40% scored, sample floor 40, 10 sources armed, measured
2026-10-10 for the triage of #2510):

    tokens   1.0xP90 -> 197 fires / 1,969 runs (10.0%)
             1.5x    ->  10 (0.5%)
             2.0x    ->   0
    duration 1.0x    -> 242 / 2,213 (10.9%);  1.5x -> 35;  2.0x -> 12;  3.0x -> 1

"Exceeds its own P90" is not a rare event: it is one run in ten by construction,
which at the 2026-10-10 fleet rate is ~34 diagnostic runs a day — the loop would
spend far more than it can ever learn. 2.0x on tokens makes the trigger dead. 1.5x
is the measured middle, and it is the operating point the item's acceptance names;
whether it is the *right* volume is a ruling that needs live fire volume, so it
lives in config (`workers.sources.cost-investigate.margin`) rather than being
argued here.

**The recursion the source talk never addresses.** A `cost-investigate` run is the
run in this fleet most likely to be an outlier, so an unguarded hook turns one
expensive audit into a chain of them, and each investigation raises the P90 of the
source watching it. `EXCLUDED_SOURCES` keeps this source's runs out of every
distribution the trigger computes, and `should_arm` refuses to arm for it at all.

Everything here reads. The only write is one queue row plus its watermark.
"""

from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# `usage_store` owns the `compaction` column's shape (#1078) and already knows how
# to read it; this module is a consumer. The dependency runs one way —
# `usage_store` imports stdlib and `app.paths` only — so there is no cycle here,
# and a second parser for the same column would be the thing that drifts.
from app import usage_store

# ── Tuning ───────────────────────────────────────────────────────────────────

#: The source whose runs this trigger never scores. See the recursion note above.
SOURCE = "cost-investigate"
EXCLUDED_SOURCES: tuple[str, ...] = (SOURCE,)

#: Days of `runs` history the distribution is drawn from. Requesting more than the
#: store holds is allowed — `replay` reports the window it actually covered.
DEFAULT_WINDOW_DAYS = 30.0

#: Below this many completed runs in the window a source is not armable at all.
#: A P90 over 12 runs is not a distribution, and `bench-mine` (34 runs) and
#: `automod-regression` (39) sat under this line as the item was triaged.
DEFAULT_MIN_SAMPLES = 40

#: The multiple of the source's own window P90 a run must exceed. See the table.
DEFAULT_MARGIN = 1.5

#: Seconds between two firings for the SAME subject source. Per-subject, not
#: global: autocode and autotriage being expensive at once is two real signals.
DEFAULT_COOLDOWN_SECONDS = 6 * 3600

#: 92, below every other source in `workers/sources/*.py` (the highest existing
#: DEFAULT_PRIORITY is 80). An investigation is the least urgent thing in the
#: queue and must never delay the work it is auditing.
DEFAULT_PRIORITY = 92

#: Max turns for the diagnostic run. It reads a profile this module hands it and
#: writes one record; 12 turns is generous, and it is the bound that keeps the
#: loop cheaper than what it explains.
DEFAULT_MAX_TURNS = 12

_WM_LAST_FIRE = "last_fire"

#: A lower bound older than any row, used when the window filter belongs to the
#: RUNS table rather than to the usage rows themselves (see `_token_totals`).
_ALL_TS = "0000-01-01T00:00:00"

#: A candidate whose rule labels at least this share of the window's sessions says
#: nothing about the run it fired on, because it would say the same thing about every
#: other run in the window. 50% is #3280's bar and is deliberately untuned: a rule
#: true of half the fleet is a description of the fleet. The measurement that
#: justified it is in `cache_pct_of`'s docstring.
FALSE_POSITIVE_BASE_RATE = 50.0

#: Days the two-week report asks for (#3280). Deliberately NOT `DEFAULT_WINDOW_DAYS`
#: (30): that is the arming window the live trigger runs on, and this is #2510's
#: fortnightly review cadence. Both clamp against the store's own age and print
#: held-of-requested, so the two windows stay visible side by side instead of one
#: constant silently answering two different questions.
REPORT_WINDOW_DAYS = 14.0

#: The cut-off `prefix_cache` armed on until #3280 took it out. Kept as a name for
#: ONE purpose: a fire whose payload predates the fix carries no `cache_pct_p25`, and
#: the only rule that could have produced its candidate is this constant. Scoring such
#: a fire at any other cut-off grades it against a rule it was never armed by — which
#: is how a report ends up printing a false-positive share of 0.0% about four records
#: three of which are tautologies. Nothing that ARMS reads this value; `analyse`
#: compares against a supplied cut-off or stays silent.
LEGACY_PREFIX_CUT_OFF = 50.0

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── The percentile rule ──────────────────────────────────────────────────────

def percentile(values: "Sequence[float]", q: float) -> "Optional[float]":
    """Nearest-rank percentile over an unsorted list of numbers.

    Widened from ints to numbers for #3280, which asks for a percentile of a float
    population (per-session cache_pct). The return is the element the rank lands on,
    so a token list still yields an int and a fired record never prints `124030.4`.

    The same rule as the local `_pct` closure in `usage_store.ttft_summary`
    (#1746). That one is a closure inside a function and cannot be imported, so
    this is a second copy rather than a reuse — stated deliberately, because two
    percentile helpers that disagree by one index would make the replay's P90 and
    the dashboard's latency percentiles non-comparable. `tests/
    test_workers_cost_replay.py` pins the rule against that definition.
    """
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


# ── Reading the two stores ───────────────────────────────────────────────────

def _ro(db_path: Path) -> sqlite3.Connection:
    """Open a store strictly read-only.

    `workers.db` is the live queue the pool is writing to and `usage.db` is the
    engine's. A reader that opened either read-write would run schema migrations
    against a file another process holds — which is why `usage_store` grew
    `read_rows_readonly` (#2406) instead of `_conn()`. `mode=ro` also means a
    missing store is an error rather than a created-empty-file lie.
    """
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _run_rows(db_path: Path, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    """One read-only query over a store, through the same `mode=ro` URL as `_ro`.

    `fires`, `window_sessions` and `session_cache_pcts` all aggregate in SQL, where
    the module's existing reads go through `usage_store.read_rows_readonly`. They
    cannot use it: it selects columns and does not take a query, and the report's
    measurements are GROUP BYs over both stores. So this is the one door, and it
    enforces the same rule `_ro` states — a reader that opened either store read-write
    would run migrations against a file another process holds (#2406).
    """
    conn = _ro(Path(db_path))
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def session_ids_of(meta: Any) -> list[str]:
    """Session ids a run recorded, from its `runs.meta_json`.

    The key depends on the source: `autotriage` and `owed-check` fan out over
    sessions and record `session_ids`; a single-session source like
    `youtube-digest` records `session_id`. Both are read because the run→cost join
    is the only cost this fleet has, and a source's logging choice is not a
    licence to be invisible to it.
    """
    if meta is None:
        return []
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return []
    if not isinstance(meta, dict):
        return []
    out = [s for s in (meta.get("session_ids") or []) if isinstance(s, str)]
    one = meta.get("session_id")
    if isinstance(one, str) and one not in out:
        out.append(one)
    return out


def _token_totals(usage_db: Path, session_ids: Iterable[str]) -> dict[str, int]:
    """Total input+output tokens per session, in one pass over the usage store.

    Read as `SUM(input_tokens + output_tokens)` over the rows whose `session_id`
    is one of the run's. Two things this deliberately does not do:

    * It does not compare `usage.ts` to `runs.completed_at`. The two columns do
      not share a format — usage rows are written `2026-01-01 00:00:00` with a
      space and `+00:00` only on newer rows, run rows are `...T...+00:00` — and a
      string comparison across those two formats silently selects the wrong rows.
      The window belongs to the run; every usage row is fetched and grouped.
    * It does not zero-fill a session that is absent. A missing session has an
      UNKNOWN cost, which is not a cost of zero: triage measured 355 of the last
      400 successful runs attributable, so ~11% would have been scored as free
      runs and pulled every P90 down. Callers see a short dict and must handle
      the miss (`unattributed`).
    """
    ids = [s for s in session_ids if s]
    if not ids:
        return {}
    rows = usage_store.read_rows_readonly(
        Path(usage_db), ["session_id", "input_tokens", "output_tokens"],
        since_ts=_ALL_TS)
    want = set(ids)
    totals: dict[str, int] = {s: 0 for s in ids}
    for r in rows:
        sid = r["session_id"]
        if sid in want:
            totals[sid] += int(r["input_tokens"] or 0) + int(r["output_tokens"] or 0)
    return totals


def cache_pct_of(prompt: int, cache_read: int) -> float:
    """Cache hits as a share of prompt spend — the ONE definition of that figure.

    `analyse`'s `cache_pct` figure and the window cut-off `prefix_cache` now arms
    below (#3280) both go through here. Two copies of the divisor would let the
    cut-off and the fired record disagree about the same run, which is precisely the
    failure this change exists to close: a rule whose threshold and whose diagnosis
    are computed two steps apart is how a label becomes a tautology unnoticed.

    **The measurement that retired the constant.** Over the trailing 14 days on this
    box (2026-10-25), per-session cache_pct measured median 41.6, P25 34.1, max 48.6:
    the rule that armed on `< 50.0` labelled 4,564 of the 4,564 sessions in its own
    window, because no session in the store can reach 50.0. A threshold above a
    distribution's ceiling is not a filter.
    """
    denom = prompt + cache_read
    return round(100.0 * cache_read / denom, 1) if denom else 0.0


def session_cache_pcts(usage_db: Path, session_ids: Iterable[str]) -> dict[str, float]:
    """cache_pct per session, over the ids one window names.

    The sibling of `_token_totals`, written in its shape on purpose: same
    `read_rows_readonly` call, same Python-side grouping, same refusal to zero-fill.
    A session with no usage row is ABSENT from the result rather than 0.0 — it did not
    cache badly, it did not run — and `percentile` plus every share below read that
    absence as "not in the denominator". A second read idiom here (SQL GROUP BY in one
    place, `read_rows_readonly` in the other) is how the two token figures in this
    module would start disagreeing.

    `cache_read` and `cache_create` are what makes this possible at all:
    `_token_totals` reads only `input_tokens`/`output_tokens`, which is why the cut-off
    had nowhere to come from before #3280 and the rule fell back to a constant.
    """
    ids = [s for s in session_ids if s]
    if not ids:
        return {}
    rows = usage_store.read_rows_readonly(
        Path(usage_db), ["session_id", "input_tokens", "cache_create", "cache_read"],
        since_ts=_ALL_TS)
    want = set(ids)
    prompt: dict[str, int] = {}
    cached: dict[str, int] = {}
    for r in rows:
        sid = r["session_id"]
        if sid not in want:
            continue
        prompt[sid] = prompt.get(sid, 0) + int(r["input_tokens"] or 0) \
            + int(r["cache_create"] or 0)
        cached[sid] = cached.get(sid, 0) + int(r["cache_read"] or 0)
    return {sid: cache_pct_of(prompt[sid], cached.get(sid, 0)) for sid in prompt}


def tokens_for_sessions(usage_db: Path, session_ids: Iterable[str]) -> Optional[int]:
    """Combined tokens over a run's sessions, or None when none of them is in the store."""
    totals = _token_totals(usage_db, session_ids)
    if not totals:
        return None
    return sum(totals.values())


# ── The distribution, and the replay over it ─────────────────────────────────

def replay(workers_db: Path, usage_db: Path, *,
           window_days: float = DEFAULT_WINDOW_DAYS,
           min_samples: int = DEFAULT_MIN_SAMPLES,
           margin: float = DEFAULT_MARGIN,
           exclude: tuple[str, ...] = EXCLUDED_SOURCES) -> dict:
    """What would have fired, over the runs the store actually holds.

    Returns per source: `samples` (successful runs with an attributable cost in
    the window), `p90_tokens`, `fires` (runs above `margin` x that P90), `armed`
    (samples >= `min_samples`), and the fire ids. Plus, at the top:
    `window_days_requested`, `window_days_held`, `first_completed_at`, and
    `unattributed` — successful runs in the window whose sessions are not in the
    usage store, which are OUT of the distribution and counted in the open rather
    than scored as zero.

    `window_days_held` is the number a reader has to have. This box's `workers.db`
    has never held `window_days` (earliest `completed_at` was 2026-09-22 when
    #2510 was triaged, ~18 days), and `app/routers/workers.py:174` already
    documents the same blindness for the dashboard's per-source stats: "a source's
    fail_rate and gpu_hours printed under a 90-day label rest on however many
    hours of runs the store actually holds." Repeating the requested number back
    as if it were coverage would be that bug one layer up.
    """
    workers_db, usage_db = Path(workers_db), Path(usage_db)
    now = _now()
    with _ro(workers_db) as conn:
        oldest = conn.execute(
            "SELECT MIN(completed_at) AS t FROM runs WHERE completed_at IS NOT NULL"
        ).fetchone()["t"]
        rows = conn.execute(
            "SELECT run_id, source, completed_at, duration_seconds, meta_json "
            "FROM runs WHERE status='success' AND completed_at IS NOT NULL "
            "ORDER BY completed_at DESC LIMIT ?",
            (50_000,),
        ).fetchall()

    held = window_days
    if oldest:
        try:
            first = datetime.fromisoformat(str(oldest).replace(" ", "T"))
            if first.tzinfo is None:
                first = first.replace(tzinfo=timezone.utc)
            age_days = (now - first).total_seconds() / 86400.0
            held = min(window_days, max(0.0, age_days))
        except ValueError:
            pass
    cutoff = (now - timedelta(days=held)).isoformat()

    in_window = [r for r in rows if str(r["completed_at"]) >= _iso_cut(cutoff)]
    sessions: set[str] = set()
    per_run: dict[str, tuple[str, list[str]]] = {}
    for r in in_window:
        if r["source"] in exclude:
            continue
        ids = session_ids_of(r["meta_json"])
        per_run[r["run_id"]] = (r["source"], ids)
        sessions.update(ids)

    totals = _token_totals(usage_db, sessions)
    # The cut-off `prefix_cache` arms below (#3280): the window's OWN 25th percentile
    # of per-session cache_pct, taken over the same `sessions` list the token
    # percentiles come from — one population, so a run cannot be judged unusual for its
    # tokens by one sample and unusual for its caching by another. Worst-quartile by
    # construction arms on ~25% of sessions; the constant it replaced armed on 100% of
    # them. Same `percentile` as the P90, so the cut-off and the distribution beside it
    # in the printed report have one estimator and one bias.
    cache_pcts = session_cache_pcts(usage_db, sessions)
    cache_pct_p25 = percentile(sorted(cache_pcts.values()), 0.25)

    by_source: dict[str, list[int]] = {}
    unattributed = 0
    for run_id, (source, ids) in per_run.items():
        tok = sum(totals[s] for s in ids if s in totals)
        if not ids or tok == 0:
            unattributed += 1
            continue
        by_source.setdefault(source, []).append(tok)

    sources_out: dict[str, dict] = {}
    for source, toks in sorted(by_source.items()):
        p90 = percentile(toks, 0.9)
        armed = len(toks) >= min_samples
        fire_ids: list[str] = []
        if armed and p90:
            for run_id, (src, ids) in per_run.items():
                if src != source:
                    continue
                tok = sum(totals[s] for s in ids if s in totals)
                if tok and tok > margin * p90:
                    fire_ids.append(run_id)
        sources_out[source] = {
            "samples": len(toks),
            "p90_tokens": p90,
            "max_tokens": max(toks) if toks else None,
            "median_tokens": percentile(toks, 0.5),
            "armed": armed,
            "fires": len(fire_ids),
            "fire_run_ids": sorted(fire_ids),
        }

    return {
        "window_days_requested": float(window_days),
        "window_days_held": round(held, 1),
        "first_completed_at": oldest,
        "margin": float(margin),
        "min_samples": int(min_samples),
        "excluded_sources": list(exclude),
        "runs_in_window": len(in_window),
        "unattributed": unattributed,
        # The cut-off, and how many sessions it was measured over. No sessions means
        # None, and a caller has to be able to tell that (the rule must stay silent)
        # from a measured cut-off of 0.0 (the rule would fire on everything).
        "cache_pct_p25": cache_pct_p25,
        "cache_pct_sessions": len(cache_pcts),
        "sources": sources_out,
    }


def _iso_cut(ts: str) -> str:
    """Normalise a timestamp for string comparison against `runs.completed_at`.

    Every `completed_at` this box holds is UTC and written `+00:00` (measured:
    8,152 of 8,154 rows contain `+00:00`), so lexicographic order IS chronological
    order within the column. The one thing that breaks it is the trailing offset,
    so the cutoff is expressed the same way rather than as a bare `...Z`.
    """
    return ts.replace("+00:00", "").replace("Z", "") + "+00:00"


# ── The arming rule ──────────────────────────────────────────────────────────

def should_arm(*, tokens: Optional[int], samples: int, p90_tokens: Optional[int],
               min_samples: int = DEFAULT_MIN_SAMPLES,
               margin: float = DEFAULT_MARGIN,
               source: Optional[str] = None) -> bool:
    """Both bounds or nothing: enough history, then the multiple.

    Neither bound is decorative and each has its own test: with the floor removed
    a 12-run source fires on a single odd run, and with the margin at 1.0 the
    trigger is a 10%-of-all-runs firehose (measured above). A run with no
    attributable tokens is NOT armed on — unknown cost is a reason to say nothing,
    not a reason to compare against zero.
    """
    if source is not None and source in EXCLUDED_SOURCES:
        return False
    if tokens is None or samples < min_samples:
        return False
    if not p90_tokens:
        return False
    return tokens > margin * p90_tokens


def load_config() -> dict:
    """This source's config block, or `{}` when the pool runs without one.

    Read through `workers.sources.get_sources_config` rather than `app.config`
    directly because that is the seam the tests and the pool already use, and it
    keeps this module importable from a context where `workers.sources` pulls the
    whole registry in.
    """
    try:
        from workers.sources import get_sources_config
        conf = get_sources_config() or {}
        block = conf.get(SOURCE)
        return dict(block) if isinstance(block, dict) else {}
    except Exception:
        return {}


def cooldown_open(queue, subject_source: str, *, cfg: dict,
                  now: Optional[datetime] = None) -> bool:
    """Has this SUBJECT source been investigated recently? Per-subject on purpose.

    One global cooldown would let a cheap source suppress an expensive one's
    signal; the window is stored under `last_fire:<subject>`.
    """
    now = now or _now()
    seconds = float(cfg.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS))
    if seconds <= 0:
        return True
    # `wm_get` returns the VALUE as a plain string, not a row. Reading
    # `last["value"]` off it raises, and an earlier draft of this function caught
    # that and returned True — which read as "cooldown open" forever and made the
    # bound silently inert, the worst possible failure for a rule whose entire
    # job is to not fire twice. A malformed stamp still falls open: a missed
    # cooldown is one extra diagnostic, a stuck one is a trigger that never
    # fires again and never says so.
    try:
        last = queue.wm_get(SOURCE, f"{_WM_LAST_FIRE}:{subject_source}")
    except Exception:
        logger.exception("cost-outlier: could not read the cooldown watermark")
        return True
    if not last:
        return True
    try:
        then = datetime.fromisoformat(str(last).replace(" ", "T"))
    except ValueError:
        logger.warning("cost-outlier: cooldown stamp %r is unparseable, opening", last)
        return True
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).total_seconds() >= seconds


def arm_after_run(queue, *, run: dict, cfg: Optional[dict] = None,
                  workers_db: Optional[Path] = None,
                  usage_db: Optional[Path] = None,
                  now: Optional[datetime] = None) -> Optional[int]:
    """Called once per completed run: enqueue one diagnostic if this run is an outlier.

    `run` is the row the caller already has (`run_id`, `source`, `status`,
    `completed_at`, `meta_json`) — read from `workers/queue.list_runs` in a replay,
    or built from what `workers/pool.py` holds at completion. No re-read, because
    the row was written moments ago by the caller itself.

    Returns the queue id of the job it enqueued, else None. Enqueues at most one
    job per subject source per cooldown, and never for its own source.
    """
    cfg = cfg if cfg is not None else load_config()
    # The distribution comes from the queue this run completed in, not from a
    # global default: `queue.db_path` is the store whose `runs` table the row was
    # just written to. Defaulting to `paths.WORKERS_DB` instead would make an
    # offline or test run read a different fleet than the one it is judging.
    workers_db = Path(workers_db) if workers_db else Path(queue.db_path)
    usage_db = Path(usage_db) if usage_db else _default_usage_db()
    now = now or _now()

    source = run.get("source")
    if not source or source in EXCLUDED_SOURCES:
        return None
    if str(run.get("status")) != "success":
        # A failed or poisoned run's token count records an attempt dying, not a
        # job's cost. The distribution was measured over successful runs; mixing
        # the two would move the P90 the trigger is judged by.
        return None

    window = float(cfg.get("window_days", DEFAULT_WINDOW_DAYS))
    min_samples = int(cfg.get("min_samples", DEFAULT_MIN_SAMPLES))
    margin = float(cfg.get("margin", DEFAULT_MARGIN))

    rep = replay(workers_db, usage_db, window_days=window,
                 min_samples=min_samples, margin=margin)
    prior = rep["sources"].get(source)
    samples = int(prior["samples"]) if prior else 0
    p90 = prior["p90_tokens"] if prior else None

    tokens = tokens_for_sessions(usage_db, session_ids_of(run.get("meta_json")))
    if not should_arm(tokens=tokens, samples=samples, p90_tokens=p90,
                      min_samples=min_samples, margin=margin, source=source):
        return None
    if not cooldown_open(queue, source, cfg=cfg, now=now):
        return None

    payload = {
        "run_id": run.get("run_id"),
        "subject_source": source,
        "tokens": tokens,
        "p90_tokens": p90,
        "samples": samples,
        "margin": margin,
        "window_days": window,
        "window_days_held": rep["window_days_held"],
        # The distribution this run was judged against travels WITH the job, not just
        # the run's own numbers. A record gets read days later by someone who cannot
        # re-derive the window the arming pass saw, and `prefix_cache` now means
        # "below THIS window's P25" — so the alternative to shipping the cut-off is a
        # record stating a percentage with nothing beside it, which is the very defect
        # #2510's owed report had to measure its way out of.
        "cache_pct_p25": rep.get("cache_pct_p25"),
        "completed_at": run.get("completed_at"),
        "session_ids": session_ids_of(run.get("meta_json")),
    }
    queue_id = queue.enqueue(
        SOURCE, "cost-investigate", payload,
        priority=int(cfg.get("priority", DEFAULT_PRIORITY)),
        # The run is the key: the same run can be reported once. The cooldown
        # covers a DIFFERENT run of the same source; this covers the same run
        # arriving twice, which the pool can do across a restart.
        dedup_key=f"cost-investigate:{run.get('run_id')}")
    if queue_id:
        # Stamped only on a real enqueue: a coalesced attempt must not spend the
        # cooldown, or a queue that is full silently mutes the trigger for 6h.
        # Three positional args: `wm_set(source, key, value)` has no `note`
        # parameter, and passing one raised a TypeError that a bare `except: pass`
        # used to swallow — so the watermark was never written and no cooldown
        # ever closed. A failed stamp is logged: it means the NEXT outlier of
        # this source also fires, which is visible cost, not a silent mute.
        try:
            queue.wm_set(SOURCE, f"{_WM_LAST_FIRE}:{source}", now.isoformat())
        except Exception:
            logger.exception("cost-outlier: armed but could not stamp the cooldown "
                             "for %s — the next outlier of this source will fire too",
                             source)
    return queue_id


def _default_usage_db() -> Path:
    from app import paths
    return paths.USAGE_DB


def record_path(run_id: str, *, out_dir: Optional[Path] = None) -> Path:
    """Where one diagnostic record lands. One file per fired run, so a two-week
    count is a file count and the owed report can read the directory.

    `out_dir` overrides the directory for a caller that must not write into the live
    pipeline tree — a test, or a rehearsal of a fire. Production calls take the
    default, which is `report_path`'s own directory, so record and report cannot drift
    to two trees.
    """
    from app import paths
    base = Path(out_dir) if out_dir else (paths.PIPELINE_DIR / "cost-outliers")
    return base / f"{run_id}.md"


def report_path(day: str, *, out_dir: Optional[Path] = None) -> Path:
    """Where the dated report row for one day goes (#3280).

    The SAME directory as the per-fire records by default, because the row and the
    records it counts are one body of evidence and a reader who finds one should find
    the other. `out_dir` exists for the test and for the command's `--out-dir`.
    """
    from app import paths
    base = Path(out_dir) if out_dir else (paths.PIPELINE_DIR / "cost-outliers")
    return base / f"report-{day}.md"


# ── What the loop actually did, and what it was worth (#3280) ───────────────
#
# `replay()` answers "what WOULD have fired". This answers the other question, the one
# #2510's owed entry asked and never got: what DID fire, what did it cost, and how much
# of what it said was worth saying. Three numbers, and the middle one is the reason the
# report exists — a trigger that fires often and means it, and a trigger that fires
# often and restates the fleet's standing state, look identical from a fire count.

#: Which session ids belong to the loop itself rather than to what it investigated is
#: the one distinction that makes its cost line mean anything. A `cost-investigate`
#: PAYLOAD carries the SUBJECT's session ids (what `investigate` reads to write the
#: record); the loop's own sessions live in its RUN ROWS' `meta_json`. See `loop_cost`.


def candidate_from_summary(summary) -> Optional[str]:
    """The candidate rule one `cost-investigate` run named, out of its own summary.

    Read back from the run row rather than re-derived: `investigate` writes that
    trailing `; candidate <rule>` itself, so parsing it scores the fire on the rule it
    actually named. The three attempts of the 2026-10-13 quarantined fire carry a
    `ValueError` traceback in this column instead and yield None — which is why an
    unscored fire is a state this module prints rather than an exception it raises.
    """
    if not summary:
        return None
    head, sep, tail = str(summary).rpartition("candidate ")
    if not sep:
        return None
    name = tail.strip().split(" ")[0]
    known = {key for key, _tpl in _CANDIDATE_RULES} | {"spread"}
    return name if name in known else None


def fires(workers_db: Path) -> dict:
    """What the trigger actually did, counted the only way that count is true.

    **Arms come from `queue`, not from `runs`.** One arming is one `queue` row with
    `source='cost-investigate'`; `runs` holds one row per ATTEMPT, so a fire that
    failed twice before quarantining contributes three run rows and a run-row count
    reports 7 fires where the store holds 5 — measured on the live box: 5 queue rows
    against 7 run rows in 2026-10-13→10-20, the difference being the retries of one
    quarantined fire. Counting `runs` reports retries as discoveries, which is the
    error #3280 clause 2 exists to prevent.

    Per arm: its attempt rows, whether it wrote its record, whether it ended
    quarantined, and the rule it named. **A record is a run row carrying an
    `artifact_path`**, not a row with `status='success'`: `investigate` returns the file
    it wrote and the pool stores that column, so the artifact is the store's own witness
    that a record was written. All three attempts of the 2026-10-13 fire carry
    none, which is what makes "quarantined arm, 3 attempt rows, 0 records" a state the
    report prints instead of a contradiction it has to explain away.

    A quarantined arm is named by its SUBJECT run id and never re-fired here: this
    function reads stores. Re-running a diagnostic would have the report making
    evidence rather than recording it.
    """
    rows = _run_rows(
        workers_db,
        "SELECT id, state, attempts, enqueued_at, completed_at, error, payload_json "
        "FROM queue WHERE source=? ORDER BY enqueued_at ASC", (SOURCE,))
    run_rows = _run_rows(
        workers_db,
        "SELECT queue_id, run_id, status, duration_seconds, meta_json, summary, "
        "artifact_path FROM runs WHERE source=? ORDER BY completed_at ASC", (SOURCE,))
    by_queue: dict[Any, list[sqlite3.Row]] = {}
    for r in run_rows:
        by_queue.setdefault(r["queue_id"], []).append(r)

    out: list[dict] = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        mine = by_queue.get(r["id"], [])
        artifacts = [x["artifact_path"] for x in mine if x["artifact_path"]]
        candidate = None
        for x in mine:
            candidate = candidate_from_summary(x["summary"]) or candidate
        own: list[str] = []
        for x in mine:
            for s in session_ids_of(x["meta_json"]):
                if s not in own:
                    own.append(s)
        out.append({
            "queue_id": r["id"],
            "state": r["state"],
            "enqueued_at": r["enqueued_at"],
            "subject_run_id": payload.get("run_id"),
            "subject_source": payload.get("subject_source"),
            "tokens": payload.get("tokens"),
            "p90_tokens": payload.get("p90_tokens"),
            # None on every payload written before #3280, which is exactly what tells
            # the report to score that fire at the legacy constant.
            "cache_pct_p25": payload.get("cache_pct_p25"),
            "own_session_ids": own,
            "attempt_rows": len(mine),
            "records": len(artifacts),
            "record_paths": artifacts,
            "candidate": candidate,
            "quarantined": str(r["state"]) == "quarantined",
            "error": r["error"],
            "run_duration_seconds": sum(float(x["duration_seconds"] or 0)
                                        for x in mine),
        })

    # A run row whose `queue_id` is NULL, or points at a queue row this query did not
    # return, still belongs to the loop's own cost; dropping it would understate the
    # gpu-minutes the report exists to publish. `orphan_run_rows` says how many, so a
    # printed total is never silently smaller than the store's own.
    counted = {f["queue_id"] for f in out}
    orphans = [r for r in run_rows if r["queue_id"] not in counted]
    own_all: list[str] = []
    for r in run_rows:
        for s in session_ids_of(r["meta_json"]):
            if s not in own_all:
                own_all.append(s)
    return {
        "arms": len(out),
        "attempts": len(run_rows),
        "records": sum(f["records"] for f in out),
        "quarantined": sum(1 for f in out if f["quarantined"]),
        "fires": out,
        "own_session_ids": own_all,
        "orphan_run_rows": len(orphans),
        "orphan_seconds": round(sum(float(r["duration_seconds"] or 0)
                                    for r in orphans), 6),
        "run_seconds_total": round(sum(float(r["duration_seconds"] or 0)
                                       for r in run_rows), 6),
    }


def loop_cost(usage_db: Path, own_session_ids: list[str]) -> dict:
    """What the loop spent, from its OWN run rows — and 0 tokens is a measurement.

    **Why 0, and why it stays 0.** Every `cost-investigate` run on this box carries
    `"session_ids": []` in its `meta_json` (measured: 7 of 7 run rows), because
    `workers/sources/cost_investigate.py` deliberately spends no model turn — it reads
    the subject run's `usage` rows and applies a rule. So there is nothing to attribute,
    `tokens_for_sessions` over an empty id list is None, and the honest print is 0: not
    a dash, not an estimate. Should a future version spend a turn, its run's `meta_json`
    would carry the session, these two figures would become non-zero with no change to
    this function, and the report would start charging the loop for itself — the only
    way the number stays honest instead of flattering.

    The ids are the RUN ROWS' (`fires()`'s `own_session_ids`), never the payload's:
    those name the run being investigated, and a cost line that read them would print
    197,903 tokens as the price of noticing 197,903 tokens.

    gpu-minutes is `SUM(duration_seconds) / 60` — the same seconds rule
    `run_rollup_by_source()` in `workers/queue.py` computes
    (`gpu_hours = round(float(r["seconds"] or 0) / 3600.0, 2)`), in the unit a loop that
    has spent 12.345 seconds in total can actually be read in. The divide and the round
    are at the call site, exactly as they are at that method's own print, so one
    quantity is not divided by 60 here and by 3600 there.
    """
    ids = [s for s in (own_session_ids or []) if s]
    tokens = tokens_for_sessions(usage_db, ids) if ids else None
    turns = 0
    if ids:
        want = set(ids)
        for r in usage_store.read_rows_readonly(
                Path(usage_db), ["session_id", "num_turns"], since_ts=_ALL_TS):
            if r["session_id"] in want:
                turns += int(r["num_turns"] or 0)
    return {"tokens": int(tokens or 0), "model_turns": int(turns),
            "attributed_sessions": len(ids)}


def window_sessions(workers_db: Path, *, window_days: float,
                    exclude: tuple[str, ...] = EXCLUDED_SOURCES,
                    now: Optional[datetime] = None) -> dict:
    """The sessions a window holds, with the same held-of-requested clamp `replay` prints.

    A separate function rather than a call to `replay()`, for one reason: `replay()`
    clamps to `DEFAULT_WINDOW_DAYS` and takes no `window_days` argument, so calling it
    for a 14-day report would measure the base rates over 30 days while the printed row
    said 14 — a share computed over a window the row does not name. So the three steps
    are restated over the caller's own number: successful runs, this source excluded,
    `session_ids_of`. The population must be the one the trigger judged, or the base
    rate neither indicts nor acquits it.
    """
    now = now or _now()
    rows = _run_rows(
        workers_db,
        "SELECT run_id, source, completed_at, meta_json FROM runs "
        "WHERE status='success' AND completed_at IS NOT NULL "
        "ORDER BY completed_at DESC LIMIT ?", (50_000,))
    oldest = _run_rows(
        workers_db,
        "SELECT MIN(completed_at) AS t FROM runs WHERE completed_at IS NOT NULL"
    )[0]["t"]
    held = float(window_days)
    if oldest:
        try:
            first = datetime.fromisoformat(str(oldest).replace(" ", "T"))
            if first.tzinfo is None:
                first = first.replace(tzinfo=timezone.utc)
            held = min(held, max(0.0, (now - first).total_seconds() / 86400.0))
        except ValueError:
            pass
    cutoff = _iso_cut((now - timedelta(days=held)).isoformat())
    ids: list[str] = []
    runs_in_window = 0
    for r in rows:
        if r["source"] in exclude or str(r["completed_at"]) < cutoff:
            continue
        runs_in_window += 1
        for s in session_ids_of(r["meta_json"]):
            if s not in ids:
                ids.append(s)
    return {"window_days_requested": float(window_days),
            "window_days_held": round(held, 1),
            "first_completed_at": oldest,
            "window_start": cutoff,
            "runs_in_window": runs_in_window,
            "session_ids": ids}


def base_rates(usage_db: Path, session_ids: list[str], *,
               prefix_cut_offs: Iterable[float] = ()) -> dict:
    """What share of these sessions each shipped rule would have labelled.

    Computed by running the SHIPPED `analyse()` once per session — the same function that
    writes a fired record, over the window's rows grouped by `session_id`. Not a
    re-derivation of each rule in SQL and not a restatement of a hand-measured number:
    the only way this can catch a false positive is if the real rule produced it.

    `prefix_cut_offs` is why one rule gets special treatment. Its base rate is a
    property of the cut-off, not of the fleet: shipped today it labels ~25% of sessions
    by construction (it arms below the window's own P25), and under
    `LEGACY_PREFIX_CUT_OFF` it labelled every session in this store. So the share is
    reported once per cut-off asked for, in `prefix_cache_at`, and a fire is scored at
    the cut-off THAT FIRE carried. One number here would be a claim about a rule with no
    window attached to it — the shape of the mistake #3280 is about.

    A session with no usage row counts in `sessions` but never in a share's denominator:
    it was not labelled poorly, it did not run.
    """
    counts = {key: 0 for key, _tpl in _CANDIDATE_RULES}
    want = set(session_ids)
    by_session: dict[str, list[dict]] = {}
    for r in usage_store.read_rows_readonly(Path(usage_db), list(_USAGE_COLS),
                                            since_ts=_ALL_TS):
        if r["session_id"] in want:
            by_session.setdefault(r["session_id"], []).append(dict(r))
    cache_figs: list[float] = []
    for sid, group in by_session.items():
        verdict = analyse(group)
        checks = verdict["checks"]
        for key in counts:
            if checks.get(key):
                counts[key] += 1
        # The SAME `prompt > 0` precondition the shipped rule carries, so the
        # population this share is measured over is exactly the population the rule
        # could ever have fired on. It is `prompt_tokens` at the verdict's top level,
        # not a `checks[...]` entry and not inside `figures`: a `.get()` on a key that
        # does not exist returns None, which made the first version of this loop
        # measure nothing and print 0.0% for a rule that in fact labels every session
        # in the store — a guard reading its own missing input reports a clean verdict.
        if float(verdict["prompt_tokens"]) > 0:
            cache_figs.append(float(verdict["figures"]["cache_pct"]))
    n = len(by_session)
    at: dict[float, dict] = {}
    for c in {float(x) for x in prefix_cut_offs}:
        hit = sum(1 for x in cache_figs if x < c)
        at[c] = {"labelled": hit,
                 "share": (round(100.0 * hit / n, 1) if n else None)}
    return {
        "sessions": len(session_ids),
        "sessions_with_rows": n,
        "rules": {key: {"labelled": counts[key],
                        "share": (round(100.0 * counts[key] / n, 1) if n else None)}
                  for key in counts},
        "prefix_cache_at": at,
    }


def score_fires(fired: list[dict], rates: dict) -> dict:
    """Which fires said something about their run, and which restated the fleet.

    A fire is a false positive when the rule it named labels at least
    `FALSE_POSITIVE_BASE_RATE` of the window's sessions: a candidate true of half the
    fleet describes the fleet, not the run, and the record's whole value is gone. The
    shares come from `base_rates`, which runs the shipped `analyse()` over those
    sessions, so the number that condemns a fire comes from the same code that produced
    it.

    **Each `prefix_cache` fire is scored at the cut-off it was armed with**, which is the
    whole difference. A payload predating #3280 carries no `cache_pct_p25`, so the only
    rule that could have named that candidate is `LEGACY_PREFIX_CUT_OFF`, whose base rate
    here is every session in the store; a payload armed after it carries its window's
    P25, whose base rate is ~25% by construction and is therefore not a false positive.
    Grading both at one cut-off prints 0.0% about a directory whose records are three
    tautologies and one signal — a number computed honestly from the wrong question,
    which is worse than no number because it closes the report that would have caught it.

    A fire with no candidate is UNSCORED, not a false positive: the 2026-10-13
    quarantined arm never reached its write, so it named nothing and there is no claim
    here to grade. The report prints it as a record lost instead.
    """
    per_fire: list[dict] = []
    fp = scored = 0
    for f in fired:
        cand = f.get("candidate")
        cut = share = None
        if cand == "prefix_cache":
            cut = (float(f["cache_pct_p25"]) if f.get("cache_pct_p25") is not None
                   else LEGACY_PREFIX_CUT_OFF)
            share = (rates.get("prefix_cache_at", {}).get(cut) or {}).get("share")
        elif cand:
            share = (rates.get("rules", {}).get(cand) or {}).get("share")
        row = {"queue_id": f.get("queue_id"),
               "subject_run_id": f.get("subject_run_id"),
               "subject_source": f.get("subject_source"),
               "candidate": cand, "cut_off": cut, "base_rate": share,
               "false_positive": None}
        if cand and share is not None:
            scored += 1
            row["false_positive"] = share >= FALSE_POSITIVE_BASE_RATE
            fp += 1 if row["false_positive"] else 0
        per_fire.append(row)
    return {"scored": scored, "false_positives": fp,
            "unscored": sum(1 for r in per_fire if r["candidate"] is None),
            "share": (round(100.0 * fp / scored, 1) if scored else None),
            "bar": FALSE_POSITIVE_BASE_RATE,
            "per_fire": per_fire}


def report(workers_db: Path, usage_db: Path, *,
           window_days: float = REPORT_WINDOW_DAYS,
           exclude: tuple[str, ...] = EXCLUDED_SOURCES,
           now: Optional[datetime] = None) -> dict:
    """Everything one dated row carries, measured over one window (#3280).

    Four measurements, each with its own docstring: `fires` (what happened, arms from
    `queue`), `window_sessions` + `base_rates` (what the shipped rules would have said
    about the window's other runs), `loop_cost` (what the loop spent saying it), and the
    window's own cache_pct distribution — which is both the cut-off `prefix_cache` now
    arms on and the number that shows the change took.
    """
    workers_db, usage_db = Path(workers_db), Path(usage_db)
    win = window_sessions(workers_db, window_days=window_days, exclude=exclude,
                          now=now)
    fired = fires(workers_db)
    cost = loop_cost(usage_db, fired["own_session_ids"])
    pcts = session_cache_pcts(usage_db, win["session_ids"])
    ordered = sorted(pcts.values())
    p25 = percentile(ordered, 0.25)
    cuts = [LEGACY_PREFIX_CUT_OFF] + ([p25] if p25 is not None else [])
    rates = base_rates(usage_db, win["session_ids"], prefix_cut_offs=cuts)
    return {
        "window": win,
        "fires": fired,
        "cost": {**cost,
                 "run_seconds": fired["run_seconds_total"],
                 "gpu_minutes": round(fired["run_seconds_total"] / 60.0, 2)},
        "cache_pct": {
            "sessions_measured": len(pcts),
            "p25": p25,
            "median": percentile(ordered, 0.5),
            "max": (max(pcts.values()) if pcts else None),
            "labelled_at_legacy_constant": (
                rates["prefix_cache_at"].get(LEGACY_PREFIX_CUT_OFF) or {}),
            "labelled_at_window_p25": (
                rates["prefix_cache_at"].get(p25) or {} if p25 is not None else {}),
        },
        "base_rates": rates,
        "scored": score_fires(fired["fires"], rates),
    }


# ── The diagnostic ──────────────────────────────────────────────────────────
#
# Deterministic by decision. The item's own risk list says the investigation must
# cost less than what it explains, and an LLM turn re-reading an expensive run is
# a second expensive run — with no bound that the trigger could honour, since a
# model-written diagnosis cannot be checked against the row it diagnoses without
# another model. Everything below is a read of `usage` rows the run already wrote:
# the numbers are the run's own, the candidate is a rule with its figure beside
# it. If this turns out to be too crude to be useful, the fix is a bounded turn
# ADDED to these figures, never a turn that replaces them.

_USAGE_COLS = ("session_id", "ts", "model", "input_tokens", "output_tokens",
               "cache_create", "cache_read", "num_turns", "reprefill_tokens",
               "prefix_misses", "compaction", "stop_reason", "tool_calls",
               "tool_errors", "tool_errors_by_class", "skills")

#: Candidate rules, in the order they are worth naming. Each is (predicate,
#: template) where the template gets the figures substituted. One candidate is
#: reported, the first that matches, because a record that lists five tells a
#: reader nothing about which one to fix first.
_CANDIDATE_RULES: tuple[tuple[str, str], ...] = (
    ("turn_cap", "stop_reason `max_turns` on {capped} of {rows} rows: the run was "
                 "still working when it hit its turn cap, so the tokens went into "
                 "work that never finished — raise the cap only if the task is "
                 "really that long, otherwise split the job."),
    ("compaction", "{compact} row(s) rewrote the history and freed tokens, and "
                   "every rewrite re-prefills everything after the break: "
                   "{compact_pct}% of this run's tokens are what its context "
                   "policy removed, so the fix is to hand it a narrower input."),
    ("prefix_cache", "cache_read is {cache_pct}% of prompt spend, BELOW this window's "
                     "own P25 of {cache_p25}%: this run re-reads its prefix far more "
                     "than its peers in the same window do, so the fix is a stable "
                     "prompt prefix (skills, tools and system text unchanged across "
                     "turns) rather than a smaller model."),
    ("tool_errors", "{tool_errs} of {tool_calls} tool calls errored "
                    "({err_by_class}): the run spent its tokens retrying, so the "
                    "fix is the tool or its arguments, not the budget."),
    ("refill", "reprefill_tokens is {refill_pct}% of prompt spend: the context "
               "window is being refilled, which usually means a tool returned "
               "something enormous — cap that tool's output."),
)


def usage_rows(usage_db: Path, session_ids: list[str]) -> list[dict]:
    """Every usage row for these sessions, oldest first.

    Read with a read-only URI: this runs inside the pool's own worker thread
    against the live store, and a write here could not be justified by anything
    the diagnostic does.
    """
    if not session_ids:
        return []
    conn = sqlite3.connect(f"file:{usage_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        marks = ",".join("?" * len(session_ids))
        rows = conn.execute(
            f"SELECT {', '.join(_USAGE_COLS)} FROM usage "
            f"WHERE session_id IN ({marks}) ORDER BY ts ASC, id ASC",
            list(session_ids)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def analyse(rows: list[dict], *,
            cache_pct_p25: Optional[float] = None) -> dict:
    """Where the tokens went, plus the first rule that names a fix.

    Deterministic, and every figure in the candidate string is a number from
    these rows — a candidate with no figure beside it is a guess about a run that
    is already over.
    """
    total = sum(int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0)
                for r in rows)
    prompt = sum(int(r.get("input_tokens") or 0) + int(r.get("cache_create") or 0)
                 for r in rows)
    cache_read = sum(int(r.get("cache_read") or 0) for r in rows)
    refill = sum(int(r.get("reprefill_tokens") or 0) for r in rows)
    # `usage.compaction` is TEXT holding one run's freeing record (#1078), not a
    # count, so `int()` on it raises on every populated row — which is how every
    # real fire of this diagnostic died before reaching its write (#2738). A row
    # counts toward the rule only when its record says something WAS freed:
    # triage measured 8,384 populated rows against 1,145 that freed anything, so
    # counting presence would arm this rule on 86% of runs, ahead of
    # `prefix_cache`, whose threshold is the one that actually discriminates.
    compact = [f for f in (_freed_tokens(r) for r in rows) if f > 0]
    capped = [r for r in rows if str(r.get("stop_reason") or "") == "max_turns"]
    tool_calls = sum(int(r.get("tool_calls") or 0) for r in rows)
    tool_errs = sum(int(r.get("tool_errors") or 0) for r in rows)
    err_class: dict[str, int] = {}
    for r in rows:
        raw = r.get("tool_errors_by_class")
        if not raw:
            continue
        try:
            for k, v in (json.loads(raw) or {}).items():
                err_class[k] = err_class.get(k, 0) + int(v)
        except (TypeError, ValueError):
            continue
    biggest = max(rows, key=lambda r: int(r.get("input_tokens") or 0)
                  + int(r.get("output_tokens") or 0), default=None)
    pct = lambda part, whole: round(100.0 * part / whole, 1) if whole else 0.0

    figures = {
        "rows": len(rows),
        "capped": len(capped),
        "compact": len(compact),
        # A share of WHAT THE POLICY REMOVED, not a sum of records: the two are the
        # same number only if every record holds a count, which none does. Read as
        # freed tokens over the run's own spend, because the cost a rewrite
        # imposes is re-prefilling what it dropped, and a run that freed 10% of its
        # tokens through compaction is paying that 10% again on the next turn.
        "compact_pct": pct(sum(compact), total),
        "cache_pct": cache_pct_of(prompt, cache_read),
        # Beside the figure the rule is judged by, so a reader holding one record can
        # see the cut-off it was armed against without re-opening the store (#3280).
        # "none" is the honest shape of a payload that carried no window distribution.
        "cache_p25": ("none" if cache_pct_p25 is None
                      else f"{float(cache_pct_p25):g}"),
        "tool_errs": tool_errs,
        "tool_calls": tool_calls,
        "err_by_class": ", ".join(f"{k} x{v}" for k, v in
                                  sorted(err_class.items(), key=lambda kv: -kv[1])[:3])
                       or "none classified",
        "refill_pct": pct(refill, prompt),
    }
    checks = {
        "turn_cap": bool(capped),
        "compaction": bool(compact),
        # Arm only BELOW the window's own cut-off, supplied by the caller (#3280).
        # The constant this line carried since #2510 — `< 50.0`, commented "below 50%
        # cached is the interesting case on this box" — is true of every session the
        # usage store holds: per-session cache_pct measured median 41.6, P25 34.1, max
        # 48.6 over the trailing 14 days, so 4,564 of 4,564 window sessions labelled
        # and no run on this box could fail it. A rule true of everything identifies
        # nothing, and because this check won the precedence race it hid the run's own
        # cost story: 3 of the 4 records the loop ever wrote name `prefix_cache`,
        # against 1 real signal. With no cut-off supplied the rule stays silent — an
        # unmeasured window is not a window in which this run is unusual. (The retired
        # comment's premise was not even arithmetic: a run at 31% cached had already
        # re-read its prefix about once per turn, which is the standing state of a
        # session cache, not a finding about a run.)
        "prefix_cache": (cache_pct_p25 is not None and prompt > 0
                         and figures["cache_pct"] < float(cache_pct_p25)),
        "tool_errors": tool_errs > 0 and tool_calls > 0 and tool_errs * 4 >= tool_calls,
        "refill": figures["refill_pct"] > 10.0,
    }
    candidate = None
    for key, template in _CANDIDATE_RULES:
        if checks.get(key):
            candidate = {"rule": key, "text": template.format(**figures)}
            break
    if candidate is None:
        candidate = {"rule": "spread",
                     "text": f"no rule matched: the spend is spread across "
                             f"{len(rows)} rows with a largest single row of "
                             f"{pct(_row_tokens(biggest), total)}% of the run, so "
                             "the candidate is the task's size itself — a shorter "
                             "prompt or fewer turns, not a broken step"}
    return {"total_tokens": total, "prompt_tokens": prompt,
            "cache_read_tokens": cache_read, "refill_tokens": refill,
            "rows": len(rows), "models": sorted({str(r.get("model")) for r in rows}),
            "biggest_row_tokens": _row_tokens(biggest),
            "figures": figures, "checks": checks, "candidate": candidate}


def _freed_tokens(row) -> int:
    """Tokens one usage row's `compaction` record says its context policy freed.

    The one way this module is allowed to read that column. `usage.compaction` is
    TEXT holding a JSON record (#1078), not a count, so there is no cast that
    reads it correctly — all three sites this helper replaced raised on every
    populated row, which is why no fire of this diagnostic had ever written its
    record (#2738). The parse lives in `usage_store`, which owns the column, and
    it sums BOTH freeing paths: a reader that looks only under `turn_start` loses
    the 506 relief-only rows out of the 1,145 that freed anything on this box, the
    same mistake #2027's first draft made and which does not fail loudly — it
    prints a plausible weaker number (`_freeing_evidence`'s docstring).

    A row with no record, a record with no `turn_start` block, an unparseable
    record, and an integer all read 0. An integer is not a live shape (measured
    2026-10-15: 8,384 populated rows, all TEXT beginning `{`, none integer-shaped),
    so reading one as 0 frees nothing to cost and invents nothing either.
    """
    if not row:
        return 0
    return usage_store.compaction_freed_tokens(row.get("compaction"))


def _row_tokens(row) -> int:
    if not row:
        return 0
    return int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)


def render(payload: dict, rows: list[dict], verdict: dict) -> str:
    """The one markdown record for one fired run.

    Written so the two-week report is a file count over this directory and so a
    reader can check every figure against the store without opening code.
    """
    lines = [f"# Cost outlier — `{payload['run_id']}` ({payload['subject_source']})",
             "",
             f"Fired because this run's token spend is an outlier for its own "
             f"source: **{payload['tokens']:,} tokens against a window P90 of "
             f"{payload['p90_tokens']:,}** ({payload['margin']}x), from "
             f"{payload['samples']} samples.",
             "",
             f"- source: `{payload['subject_source']}` (this trigger never scores "
             f"itself: `{SOURCE}` runs are excluded from that distribution)",
             f"- window held: {payload['window_days_held']}d of "
             f"{payload['window_days']}d requested",
             # Printed on EVERY record, not only on the one whose candidate it was
             # (#3280): a record saying only "cache_read is 31.5%" cannot be checked or
             # argued with by a reader who cannot re-run the arming pass, and `none` is
             # the honest shape of a payload that carried no distribution — the rule did
             # not fire, and the record says why rather than going quiet about it.
             f"- prefix_cache cut-off: window P25 = "
             f"{verdict['figures']['cache_p25']}"
             f"{'' if verdict['figures']['cache_p25'] == 'none' else '%'} "
             f"(armed below it; no constant)",
             f"- completed: {payload['completed_at']}",
             f"- sessions: {', '.join(f'`{s}`' for s in payload['session_ids'])}",
             f"- usage rows read: {verdict['rows']}, models: "
             f"{', '.join(verdict['models']) or 'none'}",
             "",
             "## Where the tokens went",
             "",
             "| session | ts | in | out | cache_read | refill | turns | compaction "
             "freed | stop | tools (err) |",
             "|---|---|---|---|---|---|---|---|---|---|",
             ]
    for r in rows[:40]:
        lines.append(
            f"| `{str(r.get('session_id'))[-16:]}` | {r.get('ts')} | "
            f"{int(r.get('input_tokens') or 0):,} | "
            f"{int(r.get('output_tokens') or 0):,} | "
            f"{int(r.get('cache_read') or 0):,} | "
            f"{int(r.get('reprefill_tokens') or 0):,} | "
            f"{int(r.get('num_turns') or 0)} | {_freed_tokens(r):,} | "
            f"{r.get('stop_reason') or ''} | {int(r.get('tool_calls') or 0)} "
            f"({int(r.get('tool_errors') or 0)}) |")
    if len(rows) > 40:
        lines.append(f"| … | | | | | | | | | *{len(rows) - 40} more rows* |")
    lines += [
        "",
        f"Prompt spend {verdict['prompt_tokens']:,} tokens; cache_read covers "
        f"{verdict['figures']['cache_pct']}% of it; largest single row is "
        f"{verdict['biggest_row_tokens']:,} tokens.",
        "",
        "## One improvement candidate",
        "",
        f"**`{verdict['candidate']['rule']}`** — {verdict['candidate']['text']}",
        "",
        f"*Written by `{SOURCE}` from the run's own `usage` rows. No model turn "
        f"was spent to produce this record; the figures above are the whole "
        f"evidence base. Rules and their order live in `_CANDIDATE_RULES` in "
        f"`app/cost_outlier.py`.*",
        "",
    ]
    return "\n".join(lines)


def investigate(payload: dict, *, usage_db: Optional[Path] = None) -> dict:
    """Produce the one record for one fired run. Returns the queue's result dict."""
    usage_db = Path(usage_db) if usage_db else _default_usage_db()
    rows = usage_rows(usage_db, list(payload.get("session_ids") or []))
    # The cut-off comes from the payload — the window the arming pass measured — NOT
    # from a fresh read here. Re-measuring over a window that has moved on since the
    # fire would judge the run against a different distribution than the one that armed
    # it, and two records would stop being comparable to each other. A payload from
    # before #3280 carries no key, and `analyse` reads that as "no cut-off": the record
    # is written, and `prefix_cache` is simply not offered as its candidate.
    verdict = analyse(rows, cache_pct_p25=payload.get("cache_pct_p25"))
    dest = record_path(str(payload.get("run_id") or "unknown"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render(payload, rows, verdict), encoding="utf-8")
    # The cut-off rides in the summary, not only in the body. The report scores a fire
    # from its run row (`candidate_from_summary` reads the rule back out of this
    # string), and a fire's base rate means nothing without the threshold it was armed
    # at — the same rule is a tautology at 50.0 and a quartile test at a window P25.
    # The two numbers of one report line have to come from one string.
    cut = ("" if payload.get("cache_pct_p25") is None
           else f" (window P25 {float(payload['cache_pct_p25']):g}%)")
    return {"summary": f"cost outlier for {payload['subject_source']}: "
                       f"{payload['tokens']:,} tokens vs p90 "
                       f"{payload['p90_tokens']:,}; candidate "
                       f"{verdict['candidate']['rule']}{cut}",
            "artifact_path": str(dest)}
