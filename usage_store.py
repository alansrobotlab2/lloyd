"""
Token usage tracking — SQLite store for Anthropic API usage metrics.

Records per-request token counts and provides aggregation queries
for the Usage dashboard (4-hour window, 7-day window, time-series).
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

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

        -- ── Inner Voice (thin observer) ──
        -- One table per observer decision. Each row captures one observation.
        -- v4 actions: noop | inject | cancel | ambient | clarify, plus
        -- noop_* variants for guarded/skipped decisions. Pre-v4 rows may
        -- also contain deny_tool | allow — these stay in the table for
        -- historical render but are never written by current code.
        -- Triggers: assistant_message | tool_call | tool_result | result | pretool.
        CREATE TABLE IF NOT EXISTS inner_voice_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            sequence_in_turn INTEGER NOT NULL,
            trigger TEXT NOT NULL,                -- assistant_message | tool_call | tool_result | result | pretool
            action TEXT NOT NULL,                 -- v4: noop | inject | cancel | ambient | clarify (+ noop_*)
            reason TEXT,
            content TEXT,                         -- inject text | ambient body | clarify question
            related_tool TEXT,                    -- for pretool / tool_call observations
            input_tokens INTEGER,
            output_tokens INTEGER,
            latency_ms INTEGER,
            model TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_iv_obs_session ON inner_voice_observations(session_id);
        CREATE INDEX IF NOT EXISTS idx_iv_obs_turn    ON inner_voice_observations(turn_id);
    """)
    # Drop the legacy stage-2-through-7 tables if they exist. The thin observer
    # has its own schema; old data is intentionally discarded.
    conn.executescript("""
        DROP TABLE IF EXISTS inner_voice_critiques;
        DROP TABLE IF EXISTS inner_voice_interventions;
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


# ---------------------------------------------------------------------------
# Inner Voice — observer observations
# ---------------------------------------------------------------------------

def record_inner_voice_observation(
    session_id: str,
    turn_id: str,
    sequence_in_turn: int,
    trigger: str,
    action: str,
    *,
    reason: Optional[str] = None,
    content: Optional[str] = None,
    related_tool: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    latency_ms: Optional[int] = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
) -> int:
    """Insert one Inner Voice observation row. Returns the new row's id."""
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO inner_voice_observations
           (session_id, turn_id, sequence_in_turn, trigger, action,
            reason, content, related_tool,
            input_tokens, output_tokens, latency_ms, model, error)
           VALUES (?, ?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?, ?, ?)""",
        (session_id, turn_id, sequence_in_turn, trigger, action,
         reason, content, related_tool,
         input_tokens, output_tokens, latency_ms, model, error),
    )
    conn.commit()
    return cur.lastrowid


def list_inner_voice_observations(
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """List observer observations, newest first. Filters by session/turn."""
    conn = _conn()
    where: list[str] = []
    params: list = []
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
    if turn_id:
        where.append("turn_id = ?")
        params.append(turn_id)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(
        f"""SELECT id, session_id, turn_id, sequence_in_turn, trigger, action,
                   reason, content, related_tool,
                   input_tokens, output_tokens, latency_ms, model, error,
                   created_at
            FROM inner_voice_observations{where_sql}
            ORDER BY id DESC LIMIT ?""",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def count_inner_voice_observations_by_action(
    session_id: str,
) -> dict[str, int]:
    """Return action → count for one session, used by /state."""
    conn = _conn()
    rows = conn.execute(
        """SELECT action, COUNT(*) AS n
             FROM inner_voice_observations
            WHERE session_id = ?
         GROUP BY action""",
        (session_id,),
    ).fetchall()
    return {r["action"]: int(r["n"]) for r in rows}
