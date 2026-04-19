"""SQLite-backed work queue for Lloyd.

Schema defined in docs/21-unified-work-queue.md. WAL mode for concurrent readers.
Atomic claim via `UPDATE ... RETURNING` in a single transaction.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("lloyd-workers.queue")


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
"""


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
        """Claim the highest-priority queued item. Atomic under the queue lock.

        If max_inflight_per_source is given, skip items whose source is at
        or above its inflight quota (claimed + running).
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
            rows = conn.execute(
                "SELECT * FROM queue WHERE state='queued' "
                "ORDER BY priority ASC, enqueued_at ASC LIMIT 50"
            ).fetchall()
            for row in rows:
                src = row["source"]
                quota = max_inflight_per_source.get(src)
                if quota is not None and inflight_counts.get(src, 0) >= quota:
                    continue
                conn.execute(
                    "UPDATE queue SET state='claimed', claimed_at=?, claimed_by=?, attempts=attempts+1 "
                    "WHERE id=? AND state='queued'",
                    (_now_iso(), worker_id, row["id"]),
                )
                conn.commit()
                refreshed = conn.execute(
                    "SELECT * FROM queue WHERE id=?", (row["id"],)
                ).fetchone()
                if refreshed and refreshed["state"] == "claimed":
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
        with self._lock, self._connect() as conn:
            # Release dedup_key so future enqueues of the same key succeed.
            if dedup_release:
                conn.execute(
                    "UPDATE queue SET state='completed', completed_at=?, dedup_key=NULL WHERE id=?",
                    (_now_iso(), item_id),
                )
            else:
                conn.execute(
                    "UPDATE queue SET state='completed', completed_at=? WHERE id=?",
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
                conn.execute(
                    "UPDATE queue SET state='queued', claimed_at=NULL, claimed_by=NULL, error=? WHERE id=?",
                    (error[:2000], item_id),
                )
                new_state = "queued"
            conn.commit()
            return new_state

    def recover_claimed(self, worker_ids: list[str] | None = None) -> int:
        """Reset claimed|running items back to queued (called on startup).

        If worker_ids given, only reset items claimed by those workers
        (useful when a single worker dies but others keep running).
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
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO runs (run_id, queue_id, source, task_id, status,
                                     started_at, completed_at, duration_seconds,
                                     summary, artifact_path, response_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, queue_id, source, task_id, status,
                    started_at, completed_at, float(duration_seconds),
                    summary[:500], artifact_path, response_json[:50000],
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

    # ── Watermarks ────────────────────────────────────────────────────────

    def wm_get(self, source: str, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM watermarks WHERE source=? AND key=?",
                (source, key),
            ).fetchone()
            return row["value"] if row else None

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


def get_queue(db_path: str | Path | None = None) -> WorkQueue:
    """Get or create the process-wide WorkQueue singleton."""
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
