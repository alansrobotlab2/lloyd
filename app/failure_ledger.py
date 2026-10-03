r"""The failure-issue ledger: recurring failures as named, dated, countable issues.

#611 wanted this and its round `SM_20260919_145031` died at a 503 grader with no
ref, no stash and no worktree, so this is a rewrite rather than a re-land
(#2079). `git log --all -S'failure_issues'` and `-S'failure_events'` both return
0 commits over every ref (`git for-each-ref | wc -l` is 161 at this base,
`86ab880d`), and `git grep -l
"failure_issue" -- '*.py'` returns 0 files — the two positive controls that make
those zeros mean something are `58cca994` for `normalize_message` and
`agent-services/guardian/detect.py:1` for `def normalize_message`.

**The problem it solves.** A failure that repeats is currently only visible as
prose. The guardian's `Service down, but no promotion to revert` alert fired 18
times between 2026-09-06T18:34:06Z and 2026-09-29T23:14:16Z (measured over
`promotions-archive-202609.jsonl.gz` + `promotions.jsonl`) and no store could
answer "when did it start" or "how often per day". Onset and prevalence are the
two things that make a standing nuisance investigable, and neither exists until
something is a row.

# The one signature, borrowed

Every signature here comes from the guardian's own normalizer,
`agent-services/guardian/detect.py::signature`/`normalize_message`, loaded by
path for the same reason `app/daily_note.py` loads the daily-note builder by
path: `agent-services` has a hyphen in the directory name so it is not a
package. There is deliberately **no second normalizer and no extension of the
first**: `signature()` is what the guardian's error-rate rollback counts on
(`guardian.py:521`, `logtail.py:136,141`), so widening its regexes to suit a
report would silently change what the guardian considers a novel error. The two
guardian suites, `tests/test_guardian_logtail.py` and
`tests/test_guardian_predicates.py`, are the purpose check on that restraint and
pass untouched.

What this module decides is only **which text** to hand the normalizer:

* a promotions-ledger alert is keyed on its `title`, because the title IS the
  family and the body names the instance. Measured on the 18 real rows, keying
  on `title + body` splits that one family into **three** signatures (12 / 5 / 1)
  — `health probe failed <n> consecutive times`, `STOPPED without an intentional
  stop`, and `FATAL: can't find command '<path>'` — which is three issues where
  the operator saw one, and would break the 18-occurrence count the guardian
  family is supposed to carry. The body is preserved in `sample_summary`.
* any other record (a worker run's failure text, a hypothesis dump) is keyed on
  the message itself.

# What the borrowed normalizer actually collapses, measured

`normalize_message` applies `_PATH`=`(/[\w.\-]+){2,}`, then `_DUR`,
`_HEX`=`\b[0-9a-f]{7,}\b`, then `_NUM`=`\d+`. Measured against candidate message
shapes at this base commit:

* `… --fetch rc=2: usage: /home/…/bundle.py: can't fetch https://youtu.be/pAnLpiAG6Es`
  and the same line with `50IgNjRtwNE` → **one** signature, because `_PATH` eats
  `/youtu.be/<id>`.
* `run_30_20261002_211240` vs `run_77_20261005_091500` → **one** signature, via
  `_NUM`, so a run id and a `YYYYMMDD` need no new regex either.
* An alphanumeric id that sits bare in the message — `ai-engineer pAnLpiAG6Es:
  fetch crashed: --fetch=pAnLpiAG6Es rc=-15:`, the shape `workers/sources/
  youtube_digest.py:738` writes today — → **two** signatures. Chasing that with
  a wider `_NUM`/`_HEX` is the change this module refuses to make; it is recorded
  as a finding on #2079 instead.

# The store

`<DATA_ROOT>/failure-ledger.db` (`app.paths.FAILURE_LEDGER_DB`), which is
`/home/alansrobotlab/lloyd-data/failure-ledger.db` in production and a scratch
root inside a round or the suite — the rule that keeps a database out of the
code tree (`architecture/data-home.md`). Three derived tables over one
insert-only event log:

* `failure_events(signature, ts, run_id, source)` — append-only. `sweep()` never
  UPDATEs or DELETEs it; the UNIQUE index exists so re-ingesting the same file
  cannot double-count, which is the only way the sweep can be run twice.
* `failure_issues(…)` — what `sweep()` derives FROM those events. `first_seen` is
  written as `MIN(stored, derived)`, so onset can only ever move earlier: a
  pruned event can never rewrite the ledger's own baseline. `sample_summary` and
  `status` are set at first insert and never rewritten by a sweep.
* `failure_findings(id, …)` + `failure_dispatches(day, …)` — the detectors'
  output as rows rather than prose, and the one-dispatch-per-day cap as a store
  property instead of a caller's promise.

# Denominators, and what a week of this has to prove

`workers.db` `runs.status` is the denominator for "what did the ledger do?". Not
`queue.state`. The dashboard prints `failed` for both: `by_state` and
`depth_by_source` come from the `queue` table and are labelled queue DEPTH by
`app/routers/dashboard.py` itself, while `runs`/`run_outcomes` is the per-attempt
outcome table, and tests/test_dashboard_sections.py:491 pins that
`by_state["failed"] == 1` is a queue row. One dispatch that fails twice and is
then poisoned is ONE `runs` row and ONE dead queue row; a dispatch killed before
any run row was minted is a dead queue row with no run row at all. `reconcile()`
returns both counts side by side, so a report cannot quote one without naming
which store it came from.

Per #543's discipline this sweep carries its own delete-or-keep review: after
FOUR WEEKS of dispatching, at least one investigation it sent must have fixed a
real bug, or a person disables it. That clause is #2079's owed list, not code
here, and it is a real threat — an ledger that files named issues nobody acted on
is a noise generator with a schema.

What "working" means is measured on the two surfaces an operator actually reads:
the daily note, and the `[guardian]` board. NOT the morning brief, which is
parked. The claim to test after a real week is that identical unanswered
operational notices on those two surfaces fell by at least half while named,
dated open issues rose — and it needs a human ruling on what counts as "the same
notice", which no detector here can supply.

# What this module does NOT do

It does not run itself. Nothing here is scheduled, and `failure-ledger` is not
registered in `workers.sources.SOURCE_REGISTRY`, so a pool that claims one of
these rows does not leave it queued waiting: `workers/pool.py:1250-1255` looks the
row's `source` up in that registry, misses, logs `Unknown source … — marking
poisoned` and calls `mark_failed(item.id, "unknown source failure-ledger", 0)`.
Registering the source and writing its handler is the next round's, and is
recorded on #2079; until then the queue row is a written-down finding with no
reader, which is the honest state and is what the row's own `error` field says.
Nor does this module ingest anything in production: `ingest_promotions` and
`record_failure` are the primitives a caller must invoke, and the one caller that
does — a scheduled sweep over the live promotions ledger — is not wired here. It
also does not suppress anything the guardian files; that is #2080.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from app import paths

logger = logging.getLogger("lloyd-failure-ledger")

# ---------------------------------------------------------------------------
# The guardian's normalizer, loaded by path (see `app/daily_note.py`)
# ---------------------------------------------------------------------------

_GUARDIAN_DIR = paths.LLOYD_HOME / "agent-services" / "guardian"
_DETECT_PATH = _GUARDIAN_DIR / "detect.py"

_spec = importlib.util.spec_from_file_location("lloyd_guardian_detect", _DETECT_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - a missing file is louder
    raise ImportError(f"cannot load the guardian's normalizer from {_DETECT_PATH}")
detect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(detect)

#: The logger name alert titles are keyed under. `signature()` hashes
#: `logger|normalized`, so this prefix is what keeps an alert family separate
#: from a log line that happens to read the same.
ALERT_LOGGER = "guardian.alert"

#: Feed names, stored in `source` on both tables. A row's provenance is part of
#: its reading: 18 alerts from the automod ledger are not 18 worker runs.
SOURCE_LEDGER = "automod-ledger"
SOURCE_RECORD = "record"


#: The guardian's own function objects, BOUND and not wrapped (#1887's rule, and
#: the reason these are assignments rather than `def`s): `fl.signature is
#: detect.signature` is what makes "does the ledger agree with the guardian's
#: error-rate rollback?" a one-line answer instead of a diff review, and
#: `tests/test_failure_ledger.py::test_the_ledger_owns_no_normalizer_of_its_own`
#: asserts exactly that identity. A wrapper here would be the second normalizer
#: this item forbids, because it could drift without changing any call site.
normalize_message = detect.normalize_message
signature = detect.signature


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def parse_ts(value: str) -> datetime:
    """Parse both timestamp spellings the live stores carry.

    `promotions.jsonl` writes `2026-09-06T18:34:06Z`; `workers.db` writes
    `2026-09-30T15:09:11.330523+00:00`. The second is TEXT ISO, not epoch, so
    `datetime(completed_at,'unixepoch')` in sqlite returns NULL for every row —
    the reason the age arithmetic lives here and not in SQL.
    """
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iso_utc(dt: datetime) -> str:
    """One canonical spelling, so `first_seen` comparisons are string-safe."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_of(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS failure_events (
  signature TEXT NOT NULL,
  ts        TEXT NOT NULL,
  run_id    TEXT NOT NULL DEFAULT '',
  source    TEXT NOT NULL,
  -- Insert-only. This index is not a place to update anything: it exists so
  -- that ingesting the same ledger twice cannot count the same alert twice,
  -- which is the precondition for a sweep that can be run twice at all.
  UNIQUE (signature, ts, run_id, source)
);

CREATE TABLE IF NOT EXISTS failure_issues (
  signature      TEXT PRIMARY KEY,
  source         TEXT NOT NULL,
  first_seen     TEXT NOT NULL,
  last_seen      TEXT NOT NULL,
  occurrences    INTEGER NOT NULL,
  per_day        TEXT NOT NULL,
  sample_summary TEXT NOT NULL,
  status         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS failure_findings (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  signature       TEXT NOT NULL,
  kind            TEXT NOT NULL,
  detected_on     TEXT NOT NULL,
  occurrences     INTEGER NOT NULL,
  dispatched      INTEGER NOT NULL DEFAULT 0,
  queue_id        INTEGER,
  deferred_reason TEXT
);

CREATE TABLE IF NOT EXISTS failure_dispatches (
  day         TEXT PRIMARY KEY,
  queue_id    INTEGER NOT NULL,
  finding_ids TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_signature ON failure_events (signature, ts);
CREATE INDEX IF NOT EXISTS idx_findings_day ON failure_findings (detected_on, dispatched);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the ledger and make sure the schema exists. Idempotent."""
    path = Path(db_path) if db_path is not None else paths.FAILURE_LEDGER_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    # `Row`, not the default tuple, and it belongs on the connection rather than
    # at each call site: `_strongest`, `_latest_finding` and the dispatch source all
    # read columns BY NAME, so with the default factory `dispatch_findings` raises
    # `TypeError: tuple indices must be integers` on the first finding — on a fresh
    # store, in production, with every test green because every test set the factory
    # itself. Caught only when tests/test_failure_ledger_dispatch.py started driving
    # the module through the pool's path instead of the fixture's.
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Ingestion — insert-only
# ---------------------------------------------------------------------------


def record_event(conn: sqlite3.Connection, *, sig: str, ts: str, run_id: str,
                 source: str, sample: str | None = None) -> bool:
    """Append one failure event. Returns False when the row was already there.

    `sample` is written to the issue row only when this signature is brand new
    (`sweep` never overwrites it), so the ledger keeps the FIRST example it ever
    saw rather than the most recent one.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO failure_events (signature, ts, run_id, source) "
        "VALUES (?, ?, ?, ?)",
        (sig, ts, run_id or "", source),
    )
    if cur.rowcount:
        if sample is not None:
            conn.execute(
                "INSERT OR IGNORE INTO failure_issues "
                "(signature, source, first_seen, last_seen, occurrences, per_day, "
                " sample_summary, status) VALUES (?, ?, ?, ?, 0, '{}', ?, 'open')",
                (sig, source, ts, ts, sample[:400]),
            )
        return True
    return False


def record_failure(conn: sqlite3.Connection, *, logger_name: str, message: str,
                   ts: str, run_id: str = "", source: str = SOURCE_RECORD) -> str:
    """Record one free-text failure (a run's error, a dumped parser message).

    The public primitive for the feed `workers.db` would supply: `run_id` is
    where a worker run's identity belongs, so a ledger row can be walked back to
    the run that produced it.
    """
    sig = signature(logger_name, message)
    record_event(conn, sig=sig, ts=iso_utc(parse_ts(ts)), run_id=run_id,
                 source=source, sample=f"{logger_name}: {message}"[:400])
    return sig


def iter_jsonl(path: Path | str) -> Iterable[dict[str, Any]]:
    """Read one promotions ledger, rotated (`.jsonl.gz`) or live (`.jsonl`)."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


#: Only these alert levels are failures — a warn is the guardian thinking out
#: loud, and counting it would inflate every prevalence number. Measured at this
#: base over the two corpora the reader has to keep apart: the LIVE
#: `promotions.jsonl` holds 57 `alert` rows (55 `error`, 1 `warn`, 1 `critical`)
#: and with `promotions-archive-202609.jsonl.gz` the pair is 86 (79 `error`,
#: 5 `warn`, 2 `critical`). 5 of 86 and 1 of 57 are warns this filter drops.
ALERT_LEVELS = frozenset({"error", "critical"})


def ingest_promotions(conn: sqlite3.Connection, *, live: Path | str,
                      archives: Iterable[Path | str] = ()) -> dict[str, int]:
    """Ingest the automod/guardian promotions ledger: rotated archives AND live.

    Both, always. The live `promotions.jsonl` holds 2 rows of the guardian's
    `Service down, but no promotion to revert` family and the rotated
    `promotions-archive-202609.jsonl.gz` holds the other 16, so a reader that
    opens the live file alone reports that family as 2 occurrences starting
    2026-09-24 — a fabricated onset for a fault that began 2026-09-06. Rotation
    is the reason the archive argument is not optional.

    Keyed on `title` (see the module docstring), with `title: body` kept as the
    sample. Aged on `created_at`, not `ts`: `ts` is an epoch float and a JSONL
    date filter over it selects 0 rows and reads as "nothing happened".
    """
    inserted = 0
    considered = 0
    for path in [Path(live), *[Path(a) for a in archives]]:
        if not path.exists():
            logger.warning("failure-ledger: promotions ledger %s absent", path)
            continue
        for row in iter_jsonl(path):
            if row.get("event") != "alert" or row.get("level") not in ALERT_LEVELS:
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            considered += 1
            created = str(row.get("created_at") or "").strip()
            if not created:
                continue
            body = str(row.get("body") or "")
            if record_event(
                conn,
                sig=signature(ALERT_LOGGER, title),
                ts=iso_utc(parse_ts(created)),
                run_id=str(row.get("round_id") or ""),
                source=SOURCE_LEDGER,
                sample=f"{title}: {body}"[:400],
            ):
                inserted += 1
    conn.commit()
    return {"considered": considered, "inserted": inserted}


# ---------------------------------------------------------------------------
# The sweep — derive issues from events, never rewrite the baseline
# ---------------------------------------------------------------------------


def sweep(conn: sqlite3.Connection) -> dict[str, int]:
    """Re-derive every `failure_issues` row from `failure_events`.

    Pure in the events and monotone in the baseline, which is what makes it
    idempotent: running it twice over one event set leaves every `first_seen` and
    `occurrences` byte-identical, and deleting the earliest event of a signature
    and re-sweeping leaves `first_seen` where it was — onset can only move
    earlier, because a ledger that forgets its own onset has stopped being a
    ledger. `sample_summary` and a human's `status` are never rewritten.
    """
    agg: dict[str, dict[str, Any]] = {}
    for sig, source, ts, run_id in conn.execute(
        "SELECT signature, source, ts, run_id FROM failure_events "
        "ORDER BY ts, run_id"
    ).fetchall():
        a = agg.get(sig)
        if a is None:
            a = agg[sig] = {"source": source, "first": ts, "last": ts,
                            "n": 0, "per_day": {}}
        a["n"] += 1
        a["last"] = max(a["last"], ts)
        a["first"] = min(a["first"], ts)
        day = ts[:10]
        a["per_day"][day] = a["per_day"].get(day, 0) + 1

    for sig, a in agg.items():
        existing = conn.execute(
            "SELECT first_seen FROM failure_issues WHERE signature = ?", (sig,)
        ).fetchone()
        first_seen = min(existing[0], a["first"]) if existing else a["first"]
        conn.execute(
            "INSERT INTO failure_issues (signature, source, first_seen, last_seen, "
            " occurrences, per_day, sample_summary, status) "
            "VALUES (?, ?, ?, ?, ?, ?, '', 'open') "
            "ON CONFLICT(signature) DO UPDATE SET "
            "  source = excluded.source, "
            "  first_seen = excluded.first_seen, "
            "  last_seen = excluded.last_seen, "
            "  occurrences = excluded.occurrences, "
            "  per_day = excluded.per_day",
            (sig, a["source"], first_seen, a["last"], a["n"],
             json.dumps(a["per_day"], sort_keys=True, separators=(",", ":"))),
        )
    conn.commit()
    return {"signatures": len(agg),
            "events": sum(a["n"] for a in agg.values())}


def get_issue(conn: sqlite3.Connection, sig: str) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM failure_issues WHERE signature = ?",
                            (sig,)).fetchone()
    finally:
        conn.row_factory = None


# ---------------------------------------------------------------------------
# Detectors — findings are rows, never prose
# ---------------------------------------------------------------------------

#: A signature is new-onset inside this window, and needs this many occurrences
#: before it is worth a run rather than one bad minute.
NEW_ONSET_MAX_AGE_H = 24.0
NEW_ONSET_MIN_OCCURRENCES = 3

#: Growth is today's count against 2x the trailing-7-day median, over the days
#: that actually had an event. Including zero-days in that median is the bug:
#: an intermittent family's median over `[0,0,0,2,0,3,0]` is 0.0, which either
#: trips on nothing or, guarded by `median > 0`, can never trip at all. Only the
#: active days carry a rate, so `median_active` is the honest denominator and an
#: all-zero trailing window means "no rate to compare against" — never a trip.
GROWTH_MEDIAN_MULTIPLE = 2.0
GROWTH_MIN_COUNT = 3
GROWTH_WINDOW_DAYS = 7


def trailing_counts(conn: sqlite3.Connection, sig: str, day: str,
                    window: int = GROWTH_WINDOW_DAYS) -> dict[str, int]:
    """Counts per day for the `window` days BEFORE `day` (not including it)."""
    start = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=window)).strftime("%Y-%m-%d")
    out: dict[str, int] = {}
    for sig_, ts in conn.execute(
        "SELECT signature, ts FROM failure_events WHERE signature = ?", (sig,)
    ).fetchall():
        d = ts[:10]
        if start <= d < day:
            out[d] = out.get(d, 0) + 1
    return out


def median_active(counts: dict[str, int]) -> float | None:
    """Median over the days that recorded at least one event. None if none did.

    Returns None rather than 0.0 on purpose: 0.0 is a real median reading ("it
    was quiet on the days it was open") while None means the window holds no rate
    at all, and the caller must not turn that absence into a threshold of zero.
    """
    values = sorted(v for v in counts.values() if v > 0)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2.0


def _emit_finding(conn: sqlite3.Connection, *, sig: str, kind: str, day: str,
                  occurrences: int) -> int:
    """One finding per (signature, kind, day): a re-sweep is not a new finding."""
    existing = conn.execute(
        "SELECT id FROM failure_findings WHERE signature = ? AND kind = ? "
        "AND detected_on = ?", (sig, kind, day)).fetchone()
    if existing:
        return int(existing[0])
    cur = conn.execute(
        "INSERT INTO failure_findings (signature, kind, detected_on, occurrences) "
        "VALUES (?, ?, ?, ?)", (sig, kind, day, occurrences))
    return int(cur.lastrowid)


def detect_new_onset(conn: sqlite3.Connection, *, now: datetime,
                     day: str | None = None) -> list[int]:
    """Signatures whose `first_seen` is inside 24 h with >=3 occurrences."""
    day = day or day_of(now)
    cutoff = iso_utc(now - timedelta(hours=NEW_ONSET_MAX_AGE_H))
    ids: list[int] = []
    for sig, first_seen, occ in conn.execute(
        "SELECT signature, first_seen, occurrences FROM failure_issues "
        "WHERE first_seen >= ? AND occurrences >= ? ORDER BY first_seen",
        (cutoff, NEW_ONSET_MIN_OCCURRENCES),
    ).fetchall():
        ids.append(_emit_finding(conn, sig=sig, kind="new_onset", day=day,
                                 occurrences=int(occ)))
    conn.commit()
    return ids


def detect_growth(conn: sqlite3.Connection, *, day: str) -> list[int]:
    """Signatures whose count on `day` is >=2x their trailing-7-day active median."""
    ids: list[int] = []
    counts: dict[str, int] = {}
    for sig, ts in conn.execute("SELECT signature, ts FROM failure_events").fetchall():
        if ts[:10] == day:
            counts[sig] = counts.get(sig, 0) + 1
    for sig, count in sorted(counts.items()):
        if count < GROWTH_MIN_COUNT:
            continue
        median = median_active(trailing_counts(conn, sig, day))
        if median is None:
            continue
        if count >= GROWTH_MEDIAN_MULTIPLE * median:
            ids.append(_emit_finding(conn, sig=sig, kind="growth", day=day,
                                     occurrences=count))
    conn.commit()
    return ids


# ---------------------------------------------------------------------------
# Dispatch — at most one investigation run per day
# ---------------------------------------------------------------------------

SOURCE = "failure-ledger"
KIND = "failure-investigation"

#: The investigation's bounded prompt. #611's own framing: the run reads the
#: ledger row, pulls at most this many raw records, and writes a diagnosis — so
#: the cap is what keeps a detector that fires on noise from becoming a run that
#: reads the whole store.
#: One day's investigation goes in at this priority. The number used to live only
#: in `architecture/workers-jobs.md` §1 and in the worker source module's
#: `DEFAULT_PRIORITY`, while `dispatch_findings` passed nothing and the queue wrote
#: its own default of 50 — nine places ABOVE owed-check and bench-mine, which is the
#: opposite of what the roster promised. It lives here because this is the function
#: that writes the row; `workers/sources/failure_ledger.py` reads it back for the
#: scheduler, so the number in the doc, in the scheduler and on the row cannot be
#: three different things. Lower runs sooner: behind every stream that does work for
#: someone (autocode 40, research 70, owed-check 72), ahead of the instrument checks
#: (bench-mine and the frontend-probe canary, both 80).
DISPATCH_PRIORITY = 75

DISPATCH_MAX_RECORDS = 20

#: How far around `first_seen` to look for the change that caused it. Onset is
#: only useful as a question about a window, which is the whole point of dating
#: an issue in the first place.
DISPATCH_WINDOW_DAYS = 3


#: The queue states that mean "a worker ran this and the engine gave up", and the
#: ones that mean "an attempt is happening or is owed". `workers/queue.py:16-22`
#: spells the difference out — `poisoned` is attempts exhausted before a verdict,
#: `quarantined` is blocked pre-flight — and `app/routers/dashboard.py` puts both
#: under `by_state`/`depth_by_source`, which its own docstring calls queue DEPTH,
#: while `runs`/`run_outcomes` is the per-attempt outcome table. Two different
#: stores, and the dashboard prints the same word `failed` for both of them.
QUEUE_UNRECOVERED_STATES = ("poisoned", "quarantined")
QUEUE_OPEN_STATES = ("queued", "claimed", "running")


def reconcile(queue: Any) -> dict[str, int]:
    """Investigation counts from the three places a reader might look.

    The dashboard can print `failed: 2` (queue rows whose attempts ran out) beside
    `run outcomes failed: 1` for the SAME batch: one dispatch that failed twice and
    was then poisoned is one `runs` row and one dead queue row, and a dispatch
    killed before any run row was written is a dead queue row with NO run row at
    all. Returning the numbers together is the point — a report that quotes one of
    them alone is quoting a denominator it has not named, which is the ambiguity
    #611's README clause exists to kill. The denominator for "how much did the
    ledger actually do?" is `runs_total`: a queue row is an intention, a run row is
    a thing that happened.

    `queue` is anything with `depth_by_source()` and a `_connect()` that reaches a
    `runs` table, so a test can hand it a `WorkQueue` over a temp file.
    """
    depth = queue.depth_by_source().get(SOURCE, {})
    unrecovered = sum(int(depth.get(state, 0)) for state in QUEUE_UNRECOVERED_STATES)
    open_rows = sum(int(depth.get(state, 0)) for state in QUEUE_OPEN_STATES)
    finished = open_rows + unrecovered + int(depth.get("completed", 0))
    with queue._connect() as conn:
        runs: dict[str, int] = {}
        for status, n in conn.execute(
            "SELECT status, COUNT(*) FROM runs WHERE source = ? GROUP BY status",
            (SOURCE,),
        ).fetchall():
            runs[str(status)] = int(n)
    return {
        "queue_total": sum(int(v) for v in depth.values()),
        "queue_unrecovered": unrecovered,
        "queue_open": open_rows,
        "runs_total": sum(runs.values()),
        "runs_failed": runs.get("failed", 0),
        "reconcilable": finished,
    }


def window_around(first_seen: str, days: int = DISPATCH_WINDOW_DAYS) -> dict[str, str]:
    """`first_seen` +/- `days`, as dates, so the prompt names a real window."""
    centre = parse_ts(first_seen)
    return {
        "from": (centre - timedelta(days=days)).strftime("%Y-%m-%d"),
        "to": (centre + timedelta(days=days)).strftime("%Y-%m-%d"),
        "centre": iso_utc(centre),
    }


def build_investigation_prompt(*, day: str, records: list[dict[str, Any]],
                              diagnosis: str, window: dict[str, str]) -> str:
    """The bounded prompt for one investigation run.

    Carries the concrete record ids and the ±3-day window around `first_seen`.
    Never a list of the whole day's findings: 20 findings get one run whose
    prompt names at most `DISPATCH_MAX_RECORDS` of them.
    """
    ids = [str(r["finding_id"]) for r in records]
    lines = [
        f"Failure-issue ledger investigation for {day}.",
        "",
        f"Diagnosis to establish: {diagnosis}",
        "",
        f"Onset window: {window['from']} to {window['to']} "
        f"(first_seen {window['centre']}, +/-{DISPATCH_WINDOW_DAYS} days).",
        f"Findings to read ({len(ids)}): {', '.join(ids)}",
        "",
        "Read the ledger rows for these finding ids, pull at most "
        f"{DISPATCH_MAX_RECORDS} matching raw records, count commits in the onset "
        "window, and write a diagnosis naming what changed near first_seen.",
    ]
    return "\n".join(lines)


def dispatch_findings(conn: sqlite3.Connection, day: str, *, queue,
                      diagnosis: str, now: datetime | None = None,
                      priority: int = DISPATCH_PRIORITY) -> int | None:
    """Enqueue ONE investigation run for `day`; return its queue id or None.

    The cap is a row in `failure_dispatches`, not a variable: a second call for
    the same day returns None and writes nothing, whatever the detector found in
    between. The findings that did not anchor the run are recorded as
    undispatched (`dispatched = 0` with a reason) rather than dropped, so the
    ledger can say how much it deliberately left uninvestigated — which is the
    number that tells a person whether the cap is set in the right place.

    An empty `diagnosis` is rejected before anything is enqueued: a run whose
    prompt does not name what it is there to establish is a run that reads the
    store and reports the weather, and the 1/day cap would have been spent on it.
    """
    if not str(diagnosis).strip():
        raise ValueError(
            "failure-ledger: an investigation needs a non-empty diagnosis — "
            "what changed near first_seen that the run is being asked to confirm")

    row = conn.execute("SELECT queue_id FROM failure_dispatches WHERE day = ?",
                       (day,)).fetchone()
    if row:
        return None

    findings = conn.execute(
        "SELECT id, signature, kind, occurrences, dispatched FROM failure_findings "
        "WHERE detected_on = ? AND dispatched = 0 ORDER BY occurrences DESC, id",
        (day,)).fetchall()
    if not findings:
        return None

    chosen = findings[:DISPATCH_MAX_RECORDS]
    anchor_sig = str(chosen[0][1])
    issue = conn.execute("SELECT first_seen FROM failure_issues WHERE signature = ?",
                         (anchor_sig,)).fetchone()
    if issue is None:  # pragma: no cover - a finding without an issue row
        raise ValueError(f"failure-ledger: finding {chosen[0][0]} has no issue row")
    window = window_around(str(issue[0]))

    records = [{"finding_id": int(f[0]), "signature": str(f[1]), "kind": str(f[2]),
                "occurrences": int(f[3])} for f in chosen]
    prompt = build_investigation_prompt(day=day, records=records,
                                        diagnosis=diagnosis, window=window)

    dedup_key = f"{SOURCE}:{day}"
    queue_id = queue.enqueue(
        source=SOURCE, kind=KIND,
        payload={
            "day": day,
            "finding_ids": [r["finding_id"] for r in records],
            "signatures": [r["signature"] for r in records],
            "window": window,
            "diagnosis": diagnosis,
            "max_records": DISPATCH_MAX_RECORDS,
            "prompt": prompt,
        },
        # The parameter, not an omission: `WorkQueue.enqueue`'s own default is 50,
        # and with no argument here every row this ledger wrote sat nine places ABOVE
        # owed-check and bench-mine while the roster page and the source module both
        # said 75. The library is what writes the row, so the number it passes is the
        # only one that reaches the pool.
        priority=priority,
        dedup_key=dedup_key,
    )
    if queue_id is None:
        # A live queue row already owns this dedup key. That is only possible
        # when the ledger's own cap row is gone — the queue remembers a day this
        # store does not — so the honest repair is to adopt the row that exists,
        # named by the id the queue actually gave it, rather than invent a
        # sentinel that reads as a row and never resolves.
        existing = [item.id for item in queue.list_items(source=SOURCE, limit=200)
                    if item.dedup_key == dedup_key
                    and item.state in ("queued", "claimed", "running")]
        if not existing:
            raise RuntimeError(
                f"failure-ledger: enqueue coalesced {dedup_key} into a live row "
                "but no live row with that dedup_key is readable — the cap cannot "
                "be recorded without the row it defers to")
        logger.info("failure-ledger: %s already queued for %s as row %d",
                    SOURCE, day, existing[0])
        queue_id = int(existing[0])

    conn.execute(
        "INSERT OR REPLACE INTO failure_dispatches (day, queue_id, finding_ids) "
        "VALUES (?, ?, ?)",
        (day, int(queue_id), json.dumps([r["finding_id"] for r in records])))
    conn.execute(
        "UPDATE failure_findings SET dispatched = 1, queue_id = ? WHERE id = ?",
        (int(queue_id), records[0]["finding_id"]))
    for rec in records[1:]:
        conn.execute(
            "UPDATE failure_findings SET dispatched = 0, queue_id = NULL, "
            " deferred_reason = ? WHERE id = ?",
            (f"dispatch-cap: 1/day, carried in queue row {int(queue_id)}",
             rec["finding_id"]))
    conn.commit()
    return int(queue_id)
