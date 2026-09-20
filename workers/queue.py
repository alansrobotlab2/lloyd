"""SQLite-backed work queue for Lloyd.

See `architecture/workers.md`. WAL mode, so the backend (which owns the pool)
and the MCP aggregator (which enqueues autoresearch rounds) can share the file.
A claim is an `UPDATE ... WHERE state='queued'` guarded by a re-read, which is
what makes it safe across those two processes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("lloyd-workers.queue")


# Authority grants (#534) live in this DB, in this file, because this file owns
# the DB. `app/harness/policy.py` reads the DDL from here rather than keeping
# its own copy — two definitions of a table whose NOT-NULL column is the whole
# safety property is how one of them stops being true.
#
# `expires_at` is NOT NULL on purpose: a grant with no expiry is not a grant,
# and a schema that permits one will eventually contain one.
GRANT_DDL = """
CREATE TABLE IF NOT EXISTS authority_grants (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  scope          TEXT NOT NULL,
  tool_pattern   TEXT NOT NULL,
  arg_predicate  TEXT NOT NULL DEFAULT '',
  quota          INTEGER,
  consumed       INTEGER NOT NULL DEFAULT 0,
  issued_by      TEXT NOT NULL,
  minted_by      TEXT NOT NULL DEFAULT 'human',
  note           TEXT,
  issued_at      TEXT NOT NULL,
  expires_at     TEXT NOT NULL,
  revoked_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_grants_scope_expiry
  ON authority_grants(scope, expires_at) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS grant_dispatch (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  at        TEXT NOT NULL,
  scope     TEXT NOT NULL,
  tool      TEXT NOT NULL,
  grant_id  INTEGER,
  decision  TEXT NOT NULL,
  reason    TEXT
);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  source        TEXT NOT NULL,
  kind          TEXT NOT NULL,
  priority      INTEGER NOT NULL DEFAULT 50,
  payload_json  TEXT NOT NULL,
  dedup_key     TEXT UNIQUE,
  state         TEXT NOT NULL DEFAULT 'queued',
  attempts      INTEGER NOT NULL DEFAULT 0,
  enqueued_at   TEXT NOT NULL,
  claimed_at    TEXT,
  claimed_by    TEXT,
  completed_at  TEXT,
  error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_queue_state_prio
  ON queue(state, priority, enqueued_at);

CREATE TABLE IF NOT EXISTS runs (
  run_id            TEXT PRIMARY KEY,
  queue_id          INTEGER,
  source            TEXT NOT NULL,
  task_id           TEXT,
  status            TEXT NOT NULL,
  started_at        TEXT NOT NULL,
  completed_at      TEXT NOT NULL,
  duration_seconds  REAL,
  summary           TEXT,
  artifact_path     TEXT,
  response_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_source_time
  ON runs(source, completed_at DESC);

CREATE TABLE IF NOT EXISTS watermarks (
  source      TEXT NOT NULL,
  key         TEXT NOT NULL,
  value       TEXT,
  updated_at  TEXT NOT NULL,
  PRIMARY KEY (source, key)
);
""" + GRANT_DDL


# The retry ceiling this queue enforces when no caller has supplied one. The
# worker pool always supplies the configured `workers.max_attempts`; this
# constant exists so a bare `WorkQueue` — a test, a script, the maintenance
# sweep's own default — agrees with `mark_failed` instead of carrying a second
# copy of the number that can drift from the first.
DEFAULT_MAX_ATTEMPTS = 3

# How many lost claim races one `claim_next` call tolerates before it reports
# "nothing claimable". Counted separately from the cap sweep below: poisoning
# an exhausted row is progress, not a loss, and must not eat a race retry.
_CLAIM_RACE_RETRIES = 5

# Bound on rows one claim call will sweep past because they are at the cap.
# Every successful poison removes its row from the candidate set, so this only
# stops a pathological pile from spinning; rows left over are still capped at
# their next claim, they are not exempt.
_CLAIM_CAP_SCAN_LIMIT = 64


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_or_none(raw: Any) -> Optional[datetime]:
    """Parse one of this module's ISO stamps, or say it cannot be parsed.

    Split out of `_iso_seconds_between` because the double-count guard has to
    ORDER two stamps, and `runs.started_at` is a plain TEXT column: a stamp this
    module did not write is any shape the writer chose. Comparing those as bytes
    compares their SPELLING, not the instants — SQLite's own `CURRENT_TIMESTAMP`
    renders `2026-09-20 09:00:05.123456`, and the space at position 10 (0x20)
    sorts it below every `2026-09-20T…` string no matter what time it names, so a
    run that genuinely recorded itself reads as predating the claim and gets
    billed a second time. Stamps with no offset are read as UTC, the same
    assumption `_iso_seconds_between` already makes about them.
    """
    try:
        parsed = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# The one INSERT for the `runs` table. `record_run` and the crash-recovery sweep
# both write run rows, and the column list drifts (it gained `meta_json`, then
# `claims_json`, by ALTER TABLE): two copies of it is how one of them stops
# writing a column the other fills, and a row missing `completed_at` is invisible
# to every window query this table has.
_RUN_INSERT_SQL = """INSERT INTO runs (run_id, queue_id, source, task_id, status,
                                     started_at, completed_at, duration_seconds,
                                     summary, artifact_path, response_json, meta_json,
                                     claims_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def _insert_run(conn: sqlite3.Connection, *, run_id: str, queue_id: Optional[int],
                source: str, status: str, started_at: str, completed_at: str,
                duration_seconds: float, summary: str = "", artifact_path: str = "",
                response_json: str = "", task_id: Optional[str] = None,
                meta_json: str = "", claims_json: str = "") -> None:
    """Insert one run row on `conn`, without committing.

    No commit here on purpose: the crash-recovery sweep writes its rows inside
    the same transaction as the queue UPDATE that reclaims them, so a sweep
    cannot be half-applied — a run recorded while its queue row is still
    `running`, or a row reclaimed with no run behind it, are both the reading
    this item exists to prevent.
    """
    conn.execute(
        _RUN_INSERT_SQL,
        (
            run_id, queue_id, source, task_id, status,
            started_at, completed_at, float(duration_seconds),
            summary[:500], artifact_path, response_json[:50000],
            meta_json[:20000] if meta_json else None,
            claims_json[:20000] if claims_json else None,
        ),
    )


def _iso_seconds_between(start_iso: str, end_iso: str) -> float:
    """Elapsed seconds between two ISO timestamps, floored at 0.

    A wall-clock difference, not a monotonic one: the process that measured the
    run is gone, so the only two stamps left are the ones in the row. Timestamps
    that cannot be parsed return 0.0 rather than raising — a sweep that throws
    would leave every remaining row `running` and switch its source off for
    good, which is the #765 starvation bug wearing this fix's hat.
    """
    start = _iso_or_none(start_iso)
    end = _iso_or_none(end_iso)
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start).total_seconds())


def _loads_or_none(raw: Any) -> Optional[dict]:
    """Parse a JSON column, tolerating NULL and anything unparseable.

    A malformed triage blob must not make a queue row unreadable — the row is
    the work, the triage is a note about it.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class QueueItem:
    id: int
    source: str
    kind: str
    priority: int
    payload: dict
    dedup_key: Optional[str]
    state: str
    attempts: int
    enqueued_at: str
    claimed_at: Optional[str]
    claimed_by: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    not_before: Optional[str] = None
    triaged_at: Optional[str] = None
    triage: Optional[dict] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "QueueItem":
        return cls(
            id=row["id"],
            source=row["source"],
            kind=row["kind"],
            priority=row["priority"],
            payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
            dedup_key=row["dedup_key"],
            state=row["state"],
            attempts=row["attempts"],
            enqueued_at=row["enqueued_at"],
            claimed_at=row["claimed_at"],
            claimed_by=row["claimed_by"],
            completed_at=row["completed_at"],
            error=row["error"],
            not_before=(row["not_before"] if "not_before" in row.keys() else None),
            triaged_at=(row["triaged_at"] if "triaged_at" in row.keys() else None),
            triage=(_loads_or_none(row["triage_json"]) if "triage_json" in row.keys() else None),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "priority": self.priority,
            "payload": self.payload,
            "dedup_key": self.dedup_key,
            "state": self.state,
            "attempts": self.attempts,
            "enqueued_at": self.enqueued_at,
            "claimed_at": self.claimed_at,
            "claimed_by": self.claimed_by,
            "completed_at": self.completed_at,
            "error": self.error,
            "triaged_at": self.triaged_at,
            "triage": self.triage,
        }


class WorkQueue:
    """Thread-safe SQLite-backed work queue.

    Use a single instance per process — connections are short-lived and
    opened inside a _lock for writes. Reads use fresh connections.

    **The queue owns the retry ceiling.** `max_attempts` is enforced at every
    path that can raise `attempts` — the claim itself and the crash-recovery
    sweep — not only inside `mark_failed`. That distinction is the whole
    point: the counter is bumped at claim time, so a path that hands a row
    back to `queued` without failing it (a process that died mid-run, which
    `recover_claimed` exists to mop up) used to climb past the cap without
    ever meeting the code that tested it. Row 3946 reached `attempts=9`
    against a cap of 3 that way, and `deep_research` reads that count as a
    topic's real retry budget.
    """

    def __init__(self, db_path: str | Path,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.max_attempts = self._valid_cap(max_attempts)
        self._init_db()

    # ── The retry ceiling ─────────────────────────────────────────────────

    @staticmethod
    def _valid_cap(max_attempts: int) -> int:
        """Coerce a supplied cap to something the enforcement can use.

        A cap below 1 is not honoured: it would mean "nothing is ever
        claimable", and the one caller that genuinely wants a row terminal
        without a run (`mark_failed(item, err, 0)` for an unknown source)
        states it at the call site, once, rather than setting the whole queue
        to zero at boot and disabling the pool.
        """
        try:
            cap = int(max_attempts)
        except (TypeError, ValueError):
            logger.warning("max_attempts %r is not a number — using %d",
                           max_attempts, DEFAULT_MAX_ATTEMPTS)
            return DEFAULT_MAX_ATTEMPTS
        if cap < 1:
            logger.warning("max_attempts=%d is below 1 — clamped to 1", cap)
            return 1
        return cap

    def set_max_attempts(self, max_attempts: int) -> int:
        """Adopt the ceiling a caller owns and return the effective value.

        The worker pool calls this from its constructor with the
        `workers.max_attempts` it was started with, so the configured number
        reaches enforcement without the queue having to read config — the
        queue is constructed by `get_queue()` long before the pool exists, and
        the aggregator process that also claims from this file never builds a
        pool at all.
        """
        self.max_attempts = self._valid_cap(max_attempts)
        return self.max_attempts

    @staticmethod
    def _cap_error(attempts: int, cap: int, action: str) -> str:
        """The `error` text recorded when the ceiling stops a row.

        Names the cap, because the row is the only record a person reading
        the Background tab will have of *why* this work stopped, and
        "max_attempts=3" is the thing they need to be able to grep.
        """
        return (f"max_attempts={cap} reached: {action} refused because the row "
                f"already carries {attempts} attempt(s), which is the cap on "
                f"claims. Poisoned instead of persisting attempts above the cap.")

    def _poison_at_cap(self, conn: sqlite3.Connection, row: sqlite3.Row,
                       cap: int, action: str) -> None:
        """Terminate a row whose attempt count has met the cap.

        Mirrors `mark_failed`'s terminal branch exactly — `completed_at`, the
        bounded error, and the released `dedup_key` — so a row stopped by the
        ceiling looks the same as one stopped by failure and the poison sweep
        triages it like any other poisoned item.
        """
        attempts = int(row["attempts"])
        conn.execute(
            "UPDATE queue SET state='poisoned', completed_at=?, error=?, dedup_key=NULL "
            "WHERE id=? AND state=?",
            (_now_iso(), self._cap_error(attempts, cap, action)[:2000],
             row["id"], row["state"]),
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Additive migration: retry-backoff support. `not_before` gates when a
            # requeued item becomes claimable again, so a failing item (e.g. during
            # a vLLM wedge) backs off instead of being re-claimed instantly.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(queue)").fetchall()}
            if "not_before" not in cols:
                conn.execute("ALTER TABLE queue ADD COLUMN not_before TEXT")
            # Additive migration: per-run structured metadata (stop_reason, usage,
            # num_turns, timeout/empty flags). Without it a run's outcome could only
            # be guessed from the free-text summary.
            run_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
            if "meta_json" not in run_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN meta_json TEXT")
            # Additive migration (#525): the run's claims, each with the check
            # that was run against disk and the status it came back. `summary` is
            # what the model said; this is what was verified about it. NULL means
            # no bundle was produced — a source outside the pilot, or a run that
            # never reached its claims — which the health view reports as
            # `runs_without_bundle` rather than as a clean sheet.
            if "claims_json" not in run_cols:
                conn.execute("ALTER TABLE runs ADD COLUMN claims_json TEXT")
            # The `run_id` the pool minted for the attempt now holding the row
            # (#1137). The log names it the moment the attempt starts; this is
            # what lets the recovery sweep write the interrupted row under the
            # SAME id, so a run the log says started is a run the table can be
            # queried for. NULL on rows predating the column and on rows never
            # marked running.
            if "current_run_id" not in cols:
                conn.execute("ALTER TABLE queue ADD COLUMN current_run_id TEXT")
            # Additive migration: poison-sweep bookkeeping. `triage_json` carries
            # the revive budget across a re-poisoning — mark_failed rewrites
            # state and error but leaves these alone, which is what stops a
            # revived-then-failed item from being revived forever.
            if "triaged_at" not in cols:
                conn.execute("ALTER TABLE queue ADD COLUMN triaged_at TEXT")
            if "triage_json" not in cols:
                conn.execute("ALTER TABLE queue ADD COLUMN triage_json TEXT")
            conn.commit()
        logger.info("workers.db initialized at %s", self.db_path)

    # ── Enqueue ───────────────────────────────────────────────────────────

    def enqueue(
        self,
        source: str,
        kind: str,
        payload: dict | None = None,
        priority: int = 50,
        dedup_key: Optional[str] = None,
    ) -> Optional[int]:
        """Insert a new queued item. Returns id, or None if dedup collision.

        A dedup collision means an item with the same dedup_key already exists
        in state queued|claimed|running — the new enqueue is dropped (coalesced).
        If the existing item is completed/failed/poisoned, the new one supersedes
        it (old row keeps its dedup_key NULLed out).
        """
        payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)
        with self._lock, self._connect() as conn:
            if dedup_key:
                existing = conn.execute(
                    "SELECT id, state FROM queue WHERE dedup_key = ?",
                    (dedup_key,),
                ).fetchone()
                if existing:
                    if existing["state"] in ("queued", "claimed", "running"):
                        return None  # coalesce
                    # Old terminal row: null its dedup_key so the new one can own it.
                    conn.execute(
                        "UPDATE queue SET dedup_key = NULL WHERE id = ?",
                        (existing["id"],),
                    )

            cur = conn.execute(
                """INSERT INTO queue
                   (source, kind, priority, payload_json, dedup_key, state, enqueued_at)
                   VALUES (?, ?, ?, ?, ?, 'queued', ?)""",
                (source, kind, priority, payload_json, dedup_key, _now_iso()),
            )
            conn.commit()
            return cur.lastrowid

    def has_live(self, dedup_key: str) -> bool:
        """Whether a row holding `dedup_key` is queued, claimed or running —
        exactly the rows an `enqueue` with that key would coalesce against."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM queue WHERE dedup_key = ? AND state IN ('queued','claimed','running')",
                (dedup_key,),
            ).fetchone()
            return row is not None

    # ── Claim (atomic) ────────────────────────────────────────────────────

    def claim_next(
        self,
        worker_id: str,
        max_inflight_per_source: dict[str, int] | None = None,
        exclude_sources: list[str] | None = None,
    ) -> Optional[QueueItem]:
        """Claim the highest-priority claimable item. Atomic under the queue lock.

        If max_inflight_per_source is given, sources at or above their inflight
        quota (claimed + running) are excluded.

        **The quota is applied in SQL, not to a window of results.** This used
        to `SELECT ... LIMIT 50` and then skip over-quota rows in Python, which
        is a different thing whenever the candidate set is larger than the
        window: fifty queued rows belonging to one saturated source, all at a
        lower priority number, fill the window completely and the loop returns
        None with claimable work sitting behind them. The pool then reads that
        as an empty queue and sleeps. Filtering the saturated sources out of
        the query means the ordering picks the first *claimable* row, whatever
        the depth of what it skipped, and `LIMIT 1` is then enough.

        `exclude_sources` is the worker pool's KV gate
        (`WorkerPool._kv_gate_held`): sources held back while the primary's
        KV cache is over budget. They join the same `NOT IN` as the saturated
        sources, for the same reason — a held source's queued rows must not
        hide a claimable row behind them.

        **A row already at the ceiling is not handed to a worker; it is
        poisoned, with the cap named in its `error`.** The claim is where the
        ceiling has to live because this method's own UPDATE is what raises
        `attempts`. `mark_failed` can only see rows that reached it through a
        failure, and the paths that do not — the crash-recovery sweep, or a
        revive parked one short of the cap that then dies again — used to
        inflate the count with nothing behind it. A row left `queued` but
        unclaimable would be the starvation bug this file already documents
        for the inflight quota, so the sweep terminates it and moves on.
        """
        max_inflight_per_source = max_inflight_per_source or {}
        with self._lock, self._connect() as conn:
            inflight_counts = {
                row["source"]: row["n"]
                for row in conn.execute(
                    "SELECT source, COUNT(*) AS n FROM queue "
                    "WHERE state IN ('claimed','running') GROUP BY source"
                ).fetchall()
            }
            saturated = [
                src for src, quota in max_inflight_per_source.items()
                if quota is not None and inflight_counts.get(src, 0) >= int(quota)
            ]
            sql = ("SELECT * FROM queue WHERE state='queued' "
                   "AND (not_before IS NULL OR not_before <= ?)")
            args: list[Any] = [_now_iso()]
            excluded = sorted(set(saturated) | set(exclude_sources or ()))
            if excluded:
                sql += f" AND source NOT IN ({','.join('?' * len(excluded))})"
                args.extend(excluded)
            sql += " ORDER BY priority ASC, enqueued_at ASC LIMIT 1"

            cap = self.max_attempts
            # Re-read after the UPDATE rather than trusting rowcount: the lock
            # is per-process and the aggregator writes to the same file, so a
            # row can be claimed out from under this SELECT. A lost race is a
            # retry, not an empty queue — and it is counted apart from the cap
            # sweep below, because terminating an exhausted row is progress and
            # must not spend one of the race retries.
            losses = 0
            cap_sweeps = 0
            while losses < _CLAIM_RACE_RETRIES:
                row = conn.execute(sql, args).fetchone()
                if not row:
                    return None
                # The ceiling, enforced here rather than only in `mark_failed`,
                # because this UPDATE is what raises `attempts`: a row that got
                # back to `queued` without ever being failed — crash recovery,
                # or a revive that parked it one short of the cap and then died
                # again — would otherwise be handed out for attempt cap+1 and
                # every design that reads that number as a budget would be
                # reading a fiction. Terminate it and keep looking: leaving it
                # `queued` and unselectable is the starvation bug this file
                # already documents for the inflight quota.
                if int(row["attempts"]) >= cap:
                    if cap_sweeps >= _CLAIM_CAP_SCAN_LIMIT:
                        logger.warning(
                            "claim_next reached its %d-row cap-sweep bound; the rows "
                            "it passed are still refused at their next claim",
                            _CLAIM_CAP_SCAN_LIMIT)
                        return None
                    cap_sweeps += 1
                    self._poison_at_cap(conn, row, cap, "claim")
                    conn.commit()
                    continue
                # `current_run_id` is cleared here, not only by the sweep: the
                # column means "the attempt NOW holding this row", and a row
                # coming off a prior attempt still carries that attempt's id.
                # Nothing between this UPDATE and `mark_running` re-stamps it, so
                # a death in that window would leave the recovery sweep holding a
                # DEAD attempt's id — which already has a `runs` row, and
                # `runs.run_id` is a PRIMARY KEY: the sweep's INSERT would raise
                # and take the whole boot-time recovery with it. Minting the new
                # id at `mark_running` is the other half; this half makes the
                # gap between them carry no id at all, which is the truth.
                conn.execute(
                    "UPDATE queue SET state='claimed', claimed_at=?, claimed_by=?, attempts=attempts+1, "
                    "current_run_id=NULL "
                    "WHERE id=? AND state='queued'",
                    (_now_iso(), worker_id, row["id"]),
                )
                conn.commit()
                refreshed = conn.execute(
                    "SELECT * FROM queue WHERE id=?", (row["id"],)
                ).fetchone()
                if refreshed and refreshed["state"] == "claimed" \
                        and refreshed["claimed_by"] == worker_id:
                    return QueueItem.from_row(refreshed)
                losses += 1
        return None

    def mark_running(self, item_id: int, run_id: "str | None" = None) -> None:
        """Move the row to `running`, stamping the id this attempt was minted.

        `run_id` is optional so a caller with no id to record keeps working, but
        the pool always passes one: it is the only thing that survives an
        unhandled `CancelledError`, and therefore the only way the next boot's
        sweep can report the killed attempt under the id the log already names.

        A call with no id CLEARS the column rather than leaving whatever the
        previous attempt stamped. The column means "the attempt now holding this
        row", and the pool mints a fresh id per attempt, so a stale value is a
        lie: the sweep would write this death under an id that already has a
        recorded run, and the second attempt's wall clock would be billed to the
        first attempt's transcript.
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE queue SET state='running', current_run_id=? "
                "WHERE id=? AND state='claimed'",
                (run_id, item_id),
            )
            conn.commit()

    def mark_completed(
        self,
        item_id: int,
        dedup_release: bool = True,
    ) -> None:
        """Mark success. Clears `error` (prior retry errors become stale on success)
        and releases `dedup_key` so future enqueues of the same key succeed.
        `attempts` is preserved so 'recovered after N attempts' is still visible.
        """
        with self._lock, self._connect() as conn:
            if dedup_release:
                conn.execute(
                    "UPDATE queue SET state='completed', completed_at=?, error=NULL, dedup_key=NULL WHERE id=?",
                    (_now_iso(), item_id),
                )
            else:
                conn.execute(
                    "UPDATE queue SET state='completed', completed_at=?, error=NULL WHERE id=?",
                    (_now_iso(), item_id),
                )
            conn.commit()

    def mark_failed(
        self,
        item_id: int,
        error: str,
        max_attempts: Optional[int] = None,
    ) -> str:
        """Mark failed. If attempts >= the cap, state=poisoned; else requeue.

        `max_attempts` defaults to the ceiling this queue already owns instead
        of a second literal 3, so the failure branch and the claim-time and
        recovery-time enforcement cannot drift apart. A caller that means
        something else says so at the call site: the pool passes its configured
        value, and the unknown-source path passes 0 to make a row terminal
        without ever running it.

        Returns the new state.
        """
        cap = self.max_attempts if max_attempts is None else int(max_attempts)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT attempts, dedup_key FROM queue WHERE id=?", (item_id,)
            ).fetchone()
            if not row:
                return "missing"
            attempts = row["attempts"]
            if attempts >= cap:
                conn.execute(
                    "UPDATE queue SET state='poisoned', completed_at=?, error=?, dedup_key=NULL WHERE id=?",
                    (_now_iso(), error[:2000], item_id),
                )
                new_state = "poisoned"
            else:
                # Exponential backoff before the item is claimable again, so a
                # failing item (esp. during a vLLM wedge) doesn't get re-claimed
                # instantly in a tight failure loop. 30s, 60s, 120s, ... capped 10m.
                backoff = min(600, 30 * (2 ** max(0, attempts - 1)))
                not_before = (
                    datetime.now(timezone.utc) + timedelta(seconds=backoff)
                ).isoformat()
                conn.execute(
                    "UPDATE queue SET state='queued', claimed_at=NULL, claimed_by=NULL, "
                    "error=?, not_before=? WHERE id=?",
                    (error[:2000], not_before, item_id),
                )
                new_state = "queued"
            conn.commit()
            return new_state

    def recover_claimed(self, worker_ids: list[str] | None = None) -> int:
        """Reset claimed|running items back to queued (called on startup).

        If worker_ids given, only reset items claimed by those workers
        (useful when a single worker dies but others keep running).

        **The pool calls this with no ids**, and the difference matters.
        Worker ids are positional (`worker-0` … `worker-{slots-1}`), so
        recovering only the current slot names strands every row claimed by a
        slot that no longer exists the moment `workers.slots` is reduced.
        Those rows sit in `running` forever, and because `claim_next` counts
        `claimed|running` toward the inflight quota, a source with
        `max_inflight: 1` is then silently switched off for good. Nothing is
        in flight when a pool starts, so the unfiltered sweep is the correct
        one; the filtered form stays for a caller that really is retiring one
        live worker among others.

        **This sweep enforces the same ceiling `claim_next` does, in both
        directions.** It is the path that produced the `attempts=9` against a
        cap of 3 recorded in backlog item #765: a process dies mid-run, the row
        comes back with its count intact, the next boot claims it and raises the
        count again, and no `mark_failed` call was ever involved — so nothing
        that tests the cap ever ran. A row that met the cap while in flight is
        terminated here rather than rejoined to the queue, and is not counted in
        the returned number. A row *below* the cap is recovered exactly as
        before: that is what a crash is for, and poisoning a first attempt
        because the machine was rebooted would turn an outage into lost work.

        **Every row this sweeps is recorded as an interrupted run.** A run that
        ends because the process died or the pool cancelled it passes through
        neither of `WorkerPool._run_item`'s two recording arms — `CancelledError`
        is a `BaseException`, and a `SIGKILL` records nothing either — so before
        this the only trace of a killed run was a line in a 10 MB rotating log,
        and every timeout and wasted-hours metric that reads `runs` reported a
        clean fleet while items were being re-queued from scratch (#1137). The
        sweep is the writer that survives every kind of death, so it is the
        writer: one `status='interrupted'` row per swept row, sharing the
        requeue's transaction. See `_record_interrupted_run` for the row's shape
        and for the one case where a run is already recorded and is skipped.

        Two interactions with the ceiling above that the tests pin, because they
        are where a second writer meets the first. **A row poisoned here is
        recorded as well as poisoned** — its attempt died in flight like any
        other and spent wall clock doing it — which means `return n` (the number
        *requeued*) can be smaller than the number of `runs` rows written; the
        log line reports both so the difference is visible rather than
        mysterious. And **every read precedes every write**: the SELECT above is
        the single snapshot that decides what gets recorded, and the poison
        UPDATE, the requeue and the INSERTs all follow it, so the recorded set
        cannot shift underneath the loop that writes it.
        """
        with self._lock, self._connect() as conn:
            owner_sql = ""
            owner_args: list[Any] = []
            if worker_ids:
                placeholders = ",".join("?" * len(worker_ids))
                owner_sql = f" AND claimed_by IN ({placeholders})"
                owner_args = list(worker_ids)
            cap = self.max_attempts

            swept = conn.execute(
                f"SELECT id, state, attempts, source, claimed_at, enqueued_at, "
                f"payload_json, current_run_id FROM queue "
                f"WHERE state IN ('claimed','running'){owner_sql}",
                owner_args,
            ).fetchall()
            exhausted = [r for r in swept if int(r["attempts"]) >= cap]
            for row in exhausted:
                self._poison_at_cap(conn, row, cap, "crash recovery")

            result = conn.execute(
                f"UPDATE queue SET state='queued', claimed_at=NULL, claimed_by=NULL, "
                f"current_run_id=NULL "
                f"WHERE state IN ('claimed','running') AND attempts < ?{owner_sql}",
                [cap, *owner_args],
            )
            swept_at = _now_iso()
            recorded = 0
            for row in swept:
                recorded += self._record_interrupted_run(conn, row, swept_at)
            conn.commit()
            n = result.rowcount or 0
            if recorded:
                logger.info(
                    "Recorded %d interrupted run(s) for runs with no live owner",
                    recorded)
            if exhausted:
                logger.warning(
                    "Recovered %d claimed/running item(s); %d at max_attempts=%d "
                    "were poisoned rather than requeued",
                    n, len(exhausted), cap)
            elif n:
                logger.info("Recovered %d claimed/running items to queued", n)
            return n

    #: Summary on a run row the sweep writes for a run that had no live owner.
    #: Matched by `autonomy.compute_health` only through `status`, never by
    #: text — a metric that reads a sentence can be broken by editing prose.
    INTERRUPTED_SUMMARY = "recovered: no live owner at pool start"

    def _record_interrupted_run(self, conn: sqlite3.Connection,
                                row: sqlite3.Row, swept_at: str) -> int:
        """Write the one `runs` row for a swept row, or skip it. Returns 0/1.

        The shape is fixed by what the row has to answer afterwards: "how much
        wall clock did the fleet burn on runs that never finished".

        - `queue_id` is the queue row's id, so `list_runs_joined` can still name
          the task through its LEFT JOIN and its payload.
        - `source` is the queue row's own, so the per-source rollup attributes
          the interruption instead of dropping it.
        - `started_at` is the row's `claimed_at` — the instant the attempt began.
          A row left `claimed` with no stamp (a writer that died between the two
          UPDATEs) falls back to `enqueued_at`: `started_at` is NOT NULL, and a
          NULL would lose the run rather than date it early.
        - `completed_at` is the sweep instant, because `list_runs_joined` and
          `run_rollup_by_source` both window on it. A row carrying only
          `started_at` is invisible to the very health report this exists to
          correct.
        - `duration_seconds` is that span measured, not rounded up. A run that
          spans no measurable time is rare enough that reporting 0.0 for it is
          more truthful than inventing a floor.

        Skipped when the run already has a row: `record_run` happens BEFORE
        `mark_completed` in the pool, so a death in that gap leaves the queue row
        `running` with its run already recorded, and an unconditional sweep row
        would bill that run twice. The test is the pool's own `started_at`
        (taken just after the claim) against this row's `claimed_at`: any run
        this process could have recorded for this claim started at or after it,
        and a retry from an earlier boot always started before it, so a genuine
        second attempt still gets its row.

        That ordering is decided on parsed timestamps, deliberately not in SQL.
        `started_at` is an unconstrained TEXT column, so `started_at >= ?`
        compares a spelling rather than an instant and can rank a run that
        recorded itself BELOW the claim it belongs to — see `_iso_or_none`. A
        prior row whose stamp cannot be parsed is treated as not recorded: a
        corrupt legacy stamp must not silence a live one.
        """
        claimed_at = row["claimed_at"] or row["enqueued_at"]
        if not claimed_at:
            # No start at all: a row cannot be dated, and a fabricated timestamp
            # would put a duration into a metric that has to be believed.
            return 0
        claim = _iso_or_none(claimed_at)
        if claim is not None:
            prior = conn.execute(
                "SELECT started_at FROM runs WHERE queue_id=?", (row["id"],)
            ).fetchall()
            if any((st := _iso_or_none(r[0])) is not None and st >= claim
                   for r in prior):
                return 0
        payload = _loads_or_none(row["payload_json"]) or {}
        task_id = payload.get("task_id")
        stamped = row["current_run_id"]
        if stamped and conn.execute(
                "SELECT 1 FROM runs WHERE run_id=? LIMIT 1", (stamped,)
        ).fetchone():
            # A DEAD attempt's id left on the row — a build that predates the
            # claim-time clear, or a claim that died before `mark_running` after
            # an earlier attempt had already recorded itself. `runs.run_id` is a
            # PRIMARY KEY, so writing under it would raise IntegrityError inside
            # the sweep's transaction and abort boot-time recovery for EVERY
            # stranded row, not just this one. Minting instead costs the cross-
            # check one id it cannot resolve; failing the sweep costs the fleet
            # every run it was about to reclaim.
            stamped = None
        _insert_run(
            conn,
            # The id the pool minted and logged for this attempt when the row
            # carries one; a freshly minted one only when it does not — a row
            # stranded before `mark_running`, or one claimed by a build that
            # predates this column.
            run_id=stamped or new_run_id(str(row["source"])),
            queue_id=row["id"],
            source=str(row["source"]),
            task_id=None if task_id is None else str(task_id),
            status="interrupted",
            started_at=str(claimed_at),
            completed_at=swept_at,
            duration_seconds=_iso_seconds_between(str(claimed_at), swept_at),
            summary=self.INTERRUPTED_SUMMARY,
            meta_json=json.dumps({"interrupted": True}),
        )
        return 1

    # ── Poison triage (see workers/maintenance.py) ────────────────────────

    def revive(
        self,
        item_id: int,
        *,
        attempts: int,
        delay_seconds: float,
        triage: dict | None = None,
    ) -> bool:
        """Return a poisoned item to `queued` for a bounded retry.

        `attempts` is set explicitly rather than reset to 0: the sweep hands an
        item exactly one more claim, not a fresh budget of `max_attempts`. A
        job with a 600s cap given three fresh tries is half an hour of GPU
        spent on a guess about a failure nobody has diagnosed yet.

        The `dedup_key` is already gone (mark_failed NULLs it on poison) and
        cannot be restored, so a revive cannot coalesce against whatever the
        source enqueued in the meantime — callers must check `has_open_sibling`
        first or the sweep becomes a duplicate-work generator.
        """
        not_before = (
            datetime.now(timezone.utc) + timedelta(seconds=max(0.0, delay_seconds))
        ).isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE queue SET state='queued', attempts=?, claimed_at=NULL, "
                "claimed_by=NULL, completed_at=NULL, not_before=?, "
                "triaged_at=?, triage_json=? WHERE id=? AND state='poisoned'",
                (int(attempts), not_before, _now_iso(),
                 json.dumps(triage or {}, default=str), item_id),
            )
            conn.commit()
            return bool(cur.rowcount)

    def quarantine(self, item_id: int, triage: dict | None = None) -> bool:
        """Move a poisoned item to the terminal `quarantined` state.

        Quarantine is a distinct state rather than a flag on `poisoned` so that
        `poisoned` keeps meaning "needs attention". An alarm that also counts
        every row already looked at stops being an alarm.
        """
        now = _now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE queue SET state='quarantined', "
                "completed_at=COALESCE(completed_at, ?), dedup_key=NULL, "
                "triaged_at=?, triage_json=? WHERE id=? AND state='poisoned'",
                (now, now, json.dumps(triage or {}, default=str), item_id),
            )
            conn.commit()
            return bool(cur.rowcount)

    def has_open_sibling(
        self, source: str, kind: str, payload: dict, exclude_id: int
    ) -> bool:
        """True if an equivalent item is already queued, claimed or running."""
        payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM queue WHERE source=? AND kind=? AND payload_json=? "
                "AND id<>? AND state IN ('queued','claimed','running') LIMIT 1",
                (source, kind, payload_json, exclude_id),
            ).fetchone()
            return row is not None

    # ── Inspection ────────────────────────────────────────────────────────

    def get(self, item_id: int) -> Optional[QueueItem]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM queue WHERE id=?", (item_id,)).fetchone()
            return QueueItem.from_row(row) if row else None

    def list_items(
        self,
        state: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> list[QueueItem]:
        q = "SELECT * FROM queue WHERE 1=1"
        args: list[Any] = []
        if state:
            q += " AND state=?"
            args.append(state)
        if source:
            q += " AND source=?"
            args.append(source)
        q += " ORDER BY enqueued_at DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(q, args).fetchall()
            return [QueueItem.from_row(r) for r in rows]

    def depth_by_source(self) -> dict[str, dict[str, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source, state, COUNT(*) as n FROM queue GROUP BY source, state"
            ).fetchall()
            out: dict[str, dict[str, int]] = {}
            for r in rows:
                out.setdefault(r["source"], {})[r["state"]] = r["n"]
            return out

    # ── Runs ──────────────────────────────────────────────────────────────

    def record_run(
        self,
        run_id: str,
        queue_id: Optional[int],
        source: str,
        status: str,
        started_at: str,
        completed_at: str,
        duration_seconds: float,
        summary: str = "",
        artifact_path: str = "",
        response_json: str = "",
        task_id: Optional[str] = None,
        meta_json: str = "",
        claims_json: str = "",
    ) -> None:
        with self._lock, self._connect() as conn:
            _insert_run(
                conn, run_id=run_id, queue_id=queue_id, source=source,
                status=status, started_at=started_at, completed_at=completed_at,
                duration_seconds=duration_seconds, summary=summary,
                artifact_path=artifact_path, response_json=response_json,
                task_id=task_id, meta_json=meta_json, claims_json=claims_json,
            )
            conn.commit()

    def list_runs(
        self,
        source: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        q = "SELECT * FROM runs WHERE 1=1"
        args: list[Any] = []
        if source:
            q += " AND source=?"
            args.append(source)
        if task_id:
            q += " AND task_id=?"
            args.append(task_id)
        q += " ORDER BY completed_at DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, args).fetchall()]

    def run_rollup_by_source(self, since_iso: str) -> dict[str, dict]:
        """Per-source run counts and durations since `since_iso`.

        The rollup `/api/workers/status` never had. That endpoint reports
        configuration (enabled, interval, max_inflight) and queue depth, which
        together say what a source is *allowed* to do and how much is waiting
        — and nothing at all about whether it works. A source failing every
        run looked exactly like a source succeeding at every run.

        Aggregated in SQL rather than by walking rows in Python, because the
        Background tab polls this beside everything else and `runs` is the
        table that grows fastest.
        """
        # `interrupted` is billed as a failure here, not as a fourth bucket:
        # #1137's rows are runs whose process died, and a rollup that left them
        # outside ok/failed/skipped would report a source whose runs keep being
        # killed as though it never ran them at all — depressing `fail_rate`
        # exactly when the panel most needs it to rise.
        q = """SELECT source,
                      COUNT(*)                                   AS total,
                      SUM(status = 'success')                     AS ok,
                      SUM(status IN ('failed','interrupted'))     AS failed,
                      SUM(status = 'skipped')                     AS skipped,
                      SUM(COALESCE(duration_seconds, 0))          AS seconds,
                      MAX(completed_at)                           AS last_completed
               FROM runs WHERE completed_at >= ? GROUP BY source"""
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(q, (since_iso,)).fetchall()]
        out: dict[str, dict] = {}
        for r in rows:
            total = int(r["total"] or 0)
            failed = int(r["failed"] or 0)
            out[str(r["source"])] = {
                "total": total,
                "ok": int(r["ok"] or 0),
                "failed": failed,
                "skipped": int(r["skipped"] or 0),
                # A rate over zero runs is not 0.0, it is unknown — and
                # rendering "0% failing" for a source that has never run is
                # the reading this panel exists to prevent.
                "fail_rate": (failed / total) if total else None,
                "gpu_hours": round(float(r["seconds"] or 0) / 3600.0, 2),
                "last_completed": r["last_completed"],
            }
        return out

    def list_runs_joined(self, source: str, since_iso: str,
                         limit: int = 20000) -> list[dict]:
        """Runs since `since_iso`, joined to their queue row.

        The join recovers the task_id for historical rows written before the
        pool passed one through (237 such rows / 73.6 GPU-hours were otherwise
        unattributable): the payload always carried it.
        """
        q = """SELECT r.*, q.payload_json AS queue_payload_json
               FROM runs r LEFT JOIN queue q ON q.id = r.queue_id
               WHERE r.source = ? AND r.completed_at >= ?
               ORDER BY r.completed_at DESC LIMIT ?"""
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, (source, since_iso, limit)).fetchall()]

    # ── Watermarks ────────────────────────────────────────────────────────

    def wm_get(self, source: str, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM watermarks WHERE source=? AND key=?",
                (source, key),
            ).fetchone()
            return row["value"] if row else None

    def wm_keys(self, source: str) -> list[str]:
        """Every watermark key this source has recorded.

        Sources that must remember a *set* — session-distill's "already
        distilled" markers — read it in one query rather than one `wm_get`
        per candidate.
        """
        with self._connect() as conn:
            return [
                r["key"] for r in conn.execute(
                    "SELECT key FROM watermarks WHERE source=?", (source,)
                ).fetchall()
            ]

    def wm_all(self, source: str) -> dict[str, str]:
        """Every watermark under one source. Used by the poison sweep to walk
        and prune its per-signature tallies."""
        with self._connect() as conn:
            return {
                r["key"]: r["value"]
                for r in conn.execute(
                    "SELECT key, value FROM watermarks WHERE source=?", (source,)
                ).fetchall()
            }

    def wm_delete(self, source: str, key: str) -> None:
        """Drop one watermark. Used to retire a cursor a source has outgrown,
        and by the poison sweep to prune a tally."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM watermarks WHERE source=? AND key=?",
                         (source, key))
            conn.commit()

    def wm_set(self, source: str, key: str, value: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO watermarks (source, key, value, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(source, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (source, key, value, _now_iso()),
            )
            conn.commit()


# ── Module-level singleton ────────────────────────────────────────────────

_queue_instance: Optional[WorkQueue] = None


def configured_db_path() -> Path:
    """The `workers.db_path` the live config names, expanded.

    This lives here rather than in each caller because **the aggregator is a
    different process from the backend**. Only the backend runs
    `start_worker_pool`, so nothing initialises the singleton on the
    aggregator side and a bare `get_queue()` there raises — which is how the
    `autoresearch_run` MCP tool came to answer "work queue not available" for
    its entire life. Not one queue row in `workers.db` carries the `targets`
    payload that tool sends, because not one of its enqueues ever reached the
    database. A process that wants the shared queue asks for the configured
    path instead of inventing one, and WAL makes the sharing safe.
    """
    from app.config import CONFIG
    raw = (CONFIG.get("workers") or {}).get("db_path") or "~/lloyd/workers.db"
    return Path(str(raw)).expanduser()


def configured_max_attempts() -> int:
    """The `workers.max_attempts` the live config names, defaulting as the code always did.

    The ceiling arrives at the queue by two routes and both are needed. The
    backend's `WorkerPool.__init__` calls `set_max_attempts` with the value
    `start_worker_pool` was handed; this covers every *other* process, because
    the aggregator never builds a pool — the same reason `configured_db_path`
    exists. Without it the aggregator enforces this module's default while the
    backend enforces the config, on one shared file, which is the two-numbers
    problem #765 is about.
    """
    try:
        from app.config import CONFIG
        raw = (CONFIG.get("workers") or {}).get("max_attempts")
    except (ImportError, AttributeError, TypeError, ValueError):
        # A cap that cannot be read must not make the shared queue
        # unconstructible. The default is a ceiling, not a licence.
        return DEFAULT_MAX_ATTEMPTS
    return DEFAULT_MAX_ATTEMPTS if raw is None else max(1, int(raw))


def get_queue(db_path: str | Path | None = None) -> WorkQueue:
    """Get or create the process-wide WorkQueue singleton.

    Raising when no path has ever been supplied is deliberate and load-bearing:
    the routers use it to tell "workers are switched off" from "workers are
    running and empty", and auto-creating the database here would make a
    disabled pool report itself initialised. Callers outside the backend pass
    `configured_db_path()`.

    The singleton takes the configured ceiling with it. In the backend the pool
    overwrites it a moment later with the same number; in the aggregator, which
    never builds a pool and only ever reaches the queue this way, this is the
    only thing that makes its enforcement match the backend's on the shared
    file. See `configured_max_attempts`.
    """
    global _queue_instance
    if _queue_instance is None:
        if db_path is None:
            raise RuntimeError("First call to get_queue() must pass db_path")
        _queue_instance = WorkQueue(db_path, max_attempts=configured_max_attempts())
    return _queue_instance


def new_run_id(source: str) -> str:
    """Generate a run_id matching the existing `run_<source>_<ts>_<hex>` shape."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{source}_{ts}_{uuid.uuid4().hex[:6]}"
