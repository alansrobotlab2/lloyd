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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        }


class WorkQueue:
    """Thread-safe SQLite-backed work queue.

    Use a single instance per process — connections are short-lived and
    opened inside a _lock for writes. Reads use fresh connections.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

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

    # ── Claim (atomic) ────────────────────────────────────────────────────

    def claim_next(
        self,
        worker_id: str,
        max_inflight_per_source: dict[str, int] | None = None,
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
            if saturated:
                sql += f" AND source NOT IN ({','.join('?' * len(saturated))})"
                args.extend(saturated)
            sql += " ORDER BY priority ASC, enqueued_at ASC LIMIT 1"

            # Re-read after the UPDATE rather than trusting rowcount: the lock
            # is per-process and the aggregator writes to the same file, so a
            # row can be claimed out from under this SELECT. A lost race is a
            # retry, not an empty queue.
            for _ in range(5):
                row = conn.execute(sql, args).fetchone()
                if not row:
                    return None
                conn.execute(
                    "UPDATE queue SET state='claimed', claimed_at=?, claimed_by=?, attempts=attempts+1 "
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
        return None

    def mark_running(self, item_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE queue SET state='running' WHERE id=? AND state='claimed'",
                (item_id,),
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
        max_attempts: int = 3,
    ) -> str:
        """Mark failed. If attempts >= max_attempts, state=poisoned; else requeue.

        Returns the new state.
        """
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT attempts, dedup_key FROM queue WHERE id=?", (item_id,)
            ).fetchone()
            if not row:
                return "missing"
            attempts = row["attempts"]
            if attempts >= max_attempts:
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
        """
        with self._lock, self._connect() as conn:
            if worker_ids:
                placeholders = ",".join("?" * len(worker_ids))
                result = conn.execute(
                    f"UPDATE queue SET state='queued', claimed_at=NULL, claimed_by=NULL "
                    f"WHERE state IN ('claimed','running') AND claimed_by IN ({placeholders})",
                    worker_ids,
                )
            else:
                result = conn.execute(
                    "UPDATE queue SET state='queued', claimed_at=NULL, claimed_by=NULL "
                    "WHERE state IN ('claimed','running')"
                )
            conn.commit()
            n = result.rowcount or 0
            if n:
                logger.info("Recovered %d claimed/running items to queued", n)
            return n

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
            conn.execute(
                """INSERT INTO runs (run_id, queue_id, source, task_id, status,
                                     started_at, completed_at, duration_seconds,
                                     summary, artifact_path, response_json, meta_json,
                                     claims_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, queue_id, source, task_id, status,
                    started_at, completed_at, float(duration_seconds),
                    summary[:500], artifact_path, response_json[:50000],
                    meta_json[:20000] if meta_json else None,
                    claims_json[:20000] if claims_json else None,
                ),
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
        q = """SELECT source,
                      COUNT(*)                                   AS total,
                      SUM(status = 'success')                     AS ok,
                      SUM(status = 'failed')                      AS failed,
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

    def wm_delete(self, source: str, key: str) -> None:
        """Drop one watermark. Used to retire a cursor a source has outgrown."""
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


def get_queue(db_path: str | Path | None = None) -> WorkQueue:
    """Get or create the process-wide WorkQueue singleton.

    Raising when no path has ever been supplied is deliberate and load-bearing:
    the routers use it to tell "workers are switched off" from "workers are
    running and empty", and auto-creating the database here would make a
    disabled pool report itself initialised. Callers outside the backend pass
    `configured_db_path()`.
    """
    global _queue_instance
    if _queue_instance is None:
        if db_path is None:
            raise RuntimeError("First call to get_queue() must pass db_path")
        _queue_instance = WorkQueue(db_path)
    return _queue_instance


def new_run_id(source: str) -> str:
    """Generate a run_id matching the existing `run_<source>_<ts>_<hex>` shape."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{source}_{ts}_{uuid.uuid4().hex[:6]}"
