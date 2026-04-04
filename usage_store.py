"""
Token usage tracking — SQLite store for Anthropic API usage metrics.

Records per-request token counts and provides aggregation queries
for the Usage dashboard (4-hour window, 7-day window, time-series).
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "usage.db"

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Thread-local SQLite connection."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _init_schema(_local.conn)
    return _local.conn


def _init_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS usage (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
            session_id      TEXT,
            model           TEXT,
            input_tokens    INTEGER NOT NULL DEFAULT 0,
            output_tokens   INTEGER NOT NULL DEFAULT 0,
            cache_create    INTEGER NOT NULL DEFAULT 0,
            cache_read      INTEGER NOT NULL DEFAULT 0,
            cost_usd        REAL,
            duration_ms     INTEGER,
            duration_api_ms INTEGER,
            num_turns       INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_usage_ts    ON usage(ts);
        CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(model);
    """)


def record_usage(
    session_id: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_create: int = 0,
    cache_read: int = 0,
    cost_usd: Optional[float] = None,
    duration_ms: Optional[int] = None,
    duration_api_ms: Optional[int] = None,
    num_turns: Optional[int] = None,
):
    """Insert a single usage record."""
    conn = _conn()
    conn.execute(
        """INSERT INTO usage
           (session_id, model, input_tokens, output_tokens,
            cache_create, cache_read, cost_usd,
            duration_ms, duration_api_ms, num_turns)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, model, input_tokens, output_tokens,
         cache_create, cache_read, cost_usd,
         duration_ms, duration_api_ms, num_turns),
    )
    conn.commit()


def _since(hours: Optional[float] = None, days: Optional[float] = None) -> str:
    """ISO timestamp for N hours/days ago."""
    delta = timedelta(hours=hours or 0, days=days or 0)
    return (datetime.utcnow() - delta).strftime("%Y-%m-%dT%H:%M:%S")


def summary(
    hours: Optional[float] = None,
    days: Optional[float] = None,
    exclude_models: Optional[list[str]] = None,
) -> dict:
    """Aggregated totals for a time window. Optionally exclude specific models."""
    conn = _conn()
    where = []
    params: list = []
    if hours or days:
        where.append("ts >= ?")
        params.append(_since(hours=hours, days=days))
    if exclude_models:
        placeholders = ",".join("?" for _ in exclude_models)
        where.append(f"model NOT IN ({placeholders})")
        params.extend(exclude_models)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    row = conn.execute(
        f"""SELECT
             COUNT(*)              AS requests,
             COALESCE(SUM(input_tokens), 0)  AS input_tokens,
             COALESCE(SUM(output_tokens), 0) AS output_tokens,
             COALESCE(SUM(cache_create), 0)   AS cache_create,
             COALESCE(SUM(cache_read), 0)     AS cache_read,
             COALESCE(SUM(cost_usd), 0)       AS cost_usd,
             COALESCE(SUM(duration_ms), 0)    AS duration_ms,
             COALESCE(SUM(duration_api_ms), 0) AS duration_api_ms
           FROM usage{where_sql}""",
        params,
    ).fetchone()
    return dict(row) if row else {}


def _excl_clause(exclude_models: Optional[list[str]], existing_where: bool = False) -> tuple[str, list]:
    """Return (sql_fragment, params) for model exclusion."""
    if not exclude_models:
        return "", []
    placeholders = ",".join("?" for _ in exclude_models)
    prefix = " AND " if existing_where else " WHERE "
    return f"{prefix}model NOT IN ({placeholders})", list(exclude_models)


def history_buckets(
    hours: float,
    bucket_minutes: int = 15,
    exclude_models: Optional[list[str]] = None,
) -> list[dict]:
    """Time-series buckets for charting."""
    conn = _conn()
    since = _since(hours=hours)
    excl_sql, excl_params = _excl_clause(exclude_models, existing_where=True)
    rows = conn.execute(
        f"""SELECT
              strftime('%Y-%m-%dT%H:', ts) ||
                printf('%02d', (CAST(strftime('%M', ts) AS INTEGER) / {bucket_minutes}) * {bucket_minutes}) ||
                ':00' AS bucket,
              COUNT(*)                        AS requests,
              COALESCE(SUM(input_tokens), 0)  AS input_tokens,
              COALESCE(SUM(output_tokens), 0) AS output_tokens,
              COALESCE(SUM(cache_create), 0)  AS cache_create,
              COALESCE(SUM(cache_read), 0)    AS cache_read,
              COALESCE(SUM(cost_usd), 0)      AS cost_usd
            FROM usage
            WHERE ts >= ?{excl_sql}
            GROUP BY bucket
            ORDER BY bucket""",
        [since] + excl_params,
    ).fetchall()
    return [dict(r) for r in rows]


def history_daily(
    days: int = 7,
    exclude_models: Optional[list[str]] = None,
) -> list[dict]:
    """Daily buckets for 7-day view."""
    conn = _conn()
    since = _since(days=days)
    excl_sql, excl_params = _excl_clause(exclude_models, existing_where=True)
    rows = conn.execute(
        f"""SELECT
             strftime('%Y-%m-%d', ts) AS bucket,
             COUNT(*)                        AS requests,
             COALESCE(SUM(input_tokens), 0)  AS input_tokens,
             COALESCE(SUM(output_tokens), 0) AS output_tokens,
             COALESCE(SUM(cache_create), 0)  AS cache_create,
             COALESCE(SUM(cache_read), 0)    AS cache_read,
             COALESCE(SUM(cost_usd), 0)      AS cost_usd
           FROM usage
           WHERE ts >= ?{excl_sql}
           GROUP BY bucket
           ORDER BY bucket""",
        [since] + excl_params,
    ).fetchall()
    return [dict(r) for r in rows]


def model_breakdown(
    hours: Optional[float] = None,
    days: Optional[float] = None,
    exclude_models: Optional[list[str]] = None,
) -> list[dict]:
    """Per-model breakdown for a time window."""
    conn = _conn()
    where: list[str] = []
    params: list = []
    if hours or days:
        where.append("ts >= ?")
        params.append(_since(hours=hours, days=days))
    if exclude_models:
        placeholders = ",".join("?" for _ in exclude_models)
        where.append(f"model NOT IN ({placeholders})")
        params.extend(exclude_models)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(
        f"""SELECT
               model,
               COUNT(*)                        AS requests,
               COALESCE(SUM(input_tokens), 0)  AS input_tokens,
               COALESCE(SUM(output_tokens), 0) AS output_tokens,
               COALESCE(SUM(cache_create), 0)  AS cache_create,
               COALESCE(SUM(cache_read), 0)    AS cache_read,
               COALESCE(SUM(cost_usd), 0)      AS cost_usd
             FROM usage{where_sql}
             GROUP BY model ORDER BY cost_usd DESC""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def recent_requests(limit: int = 20) -> list[dict]:
    """Most recent usage records."""
    conn = _conn()
    rows = conn.execute(
        """SELECT id, ts, session_id, model, input_tokens, output_tokens,
                  cache_create, cache_read, cost_usd, duration_ms, num_turns
           FROM usage ORDER BY ts DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
