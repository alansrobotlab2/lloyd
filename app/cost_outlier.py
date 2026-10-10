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
from typing import Any, Iterable, Optional

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

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── The percentile rule ──────────────────────────────────────────────────────

def percentile(values: list[int], q: float) -> Optional[int]:
    """Nearest-rank percentile over an unsorted list of ints.

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
    from app import usage_store

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


def record_path(run_id: str) -> Path:
    """Where one diagnostic record lands. One file per fired run, so a two-week
    count is a file count and the owed report can read the directory."""
    from app import paths
    return paths.PIPELINE_DIR / "cost-outliers" / f"{run_id}.md"


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
    ("compaction", "{compact} row(s) show compaction, which re-reads the history: "
                   "the run spent {compact_pct}% of its tokens on re-reading its "
                   "own context, so the fix is to hand it a narrower input."),
    ("prefix_cache", "cache_read is {cache_pct}% of prompt spend, so nearly every "
                     "turn re-reads an uncached prefix: the fix is a stable prompt "
                     "prefix (skills, tools and system text unchanged across turns) "
                     "rather than a smaller model."),
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


def analyse(rows: list[dict]) -> dict:
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
    compact = [r for r in rows if int(r.get("compaction") or 0)]
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
        "compact_pct": pct(sum(int(r.get("compaction") or 0) for r in compact), total),
        "cache_pct": pct(cache_read, prompt + cache_read),
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
        # Below 50% cached is the interesting case on this box: a run whose whole
        # prompt is re-sent every turn is paying the context over and over, which
        # is how a long run gets to 800k tokens without doing 800k of work.
        "prefix_cache": figures["cache_pct"] < 50.0 and prompt > 0,
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
             f"- completed: {payload['completed_at']}",
             f"- sessions: {', '.join(f'`{s}`' for s in payload['session_ids'])}",
             f"- usage rows read: {verdict['rows']}, models: "
             f"{', '.join(verdict['models']) or 'none'}",
             "",
             "## Where the tokens went",
             "",
             "| session | ts | in | out | cache_read | refill | turns | compaction | stop | tools (err) |",
             "|---|---|---|---|---|---|---|---|---|---|",
             ]
    for r in rows[:40]:
        lines.append(
            f"| `{str(r.get('session_id'))[-16:]}` | {r.get('ts')} | "
            f"{int(r.get('input_tokens') or 0):,} | "
            f"{int(r.get('output_tokens') or 0):,} | "
            f"{int(r.get('cache_read') or 0):,} | "
            f"{int(r.get('reprefill_tokens') or 0):,} | "
            f"{int(r.get('num_turns') or 0)} | {int(r.get('compaction') or 0)} | "
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
    verdict = analyse(rows)
    dest = record_path(str(payload.get("run_id") or "unknown"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render(payload, rows, verdict), encoding="utf-8")
    return {"summary": f"cost outlier for {payload['subject_source']}: "
                       f"{payload['tokens']:,} tokens vs p90 "
                       f"{payload['p90_tokens']:,}; candidate "
                       f"{verdict['candidate']['rule']}",
            "artifact_path": str(dest)}
