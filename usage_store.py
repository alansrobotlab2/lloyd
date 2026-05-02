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

        -- ── Inner Voice (#345) ──
        -- Two tables. `inner_voice_critiques` captures "what Brain 2 thought"
        -- per persona invocation. `inner_voice_interventions` captures "what
        -- was done about it" — the actual injected steer/interrupt/continue.
        -- Linked via FK so we can query each independently and join when we
        -- want the causal chain. Forensic offsets point into event_log JSONL
        -- so the SQLite row stays small while the raw prompt/response live
        -- in the append-only log.
        CREATE TABLE IF NOT EXISTS inner_voice_critiques (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            persona TEXT NOT NULL,
            persona_version TEXT,
            model TEXT NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            latency_ms INTEGER,
            disagrees BOOLEAN,
            severity REAL,
            reason TEXT,
            suggested_action TEXT,                -- 'nudge' | 'veto' | 'escalate' | NULL
            action_taken TEXT,                    -- 'log_only' | 'steer' | 'interrupt' | 'continue' | 'escalate' | 'agreement'
            nudge_succeeded BOOLEAN,              -- backfilled by grading pass
            anchor_response_excerpt TEXT,
            event_log_offset INTEGER,             -- line number of persona_invoked event
            raw_response_offset INTEGER,          -- line number of persona_response_raw event
            prompt_hash TEXT,                     -- sha256 for "same-prompt" grouping
            parse_attempts INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_iv_critiques_session     ON inner_voice_critiques(session_id);
        CREATE INDEX IF NOT EXISTS idx_iv_critiques_turn        ON inner_voice_critiques(turn_id);
        CREATE INDEX IF NOT EXISTS idx_iv_critiques_prompt_hash ON inner_voice_critiques(prompt_hash);

        CREATE TABLE IF NOT EXISTS inner_voice_interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            triggered_by_critique_id INTEGER REFERENCES inner_voice_critiques(id),
            kind TEXT NOT NULL,                   -- 'steer' | 'interrupt' | 'continue' | 'escalate'
            target_turn_id TEXT NOT NULL,
            content TEXT NOT NULL,                -- the actual injected text
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            outcome_turn_id TEXT,                 -- backfilled by grading pass
            outcome_addressed BOOLEAN,
            outcome_summary TEXT,
            graded_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_iv_interventions_session  ON inner_voice_interventions(session_id);
        CREATE INDEX IF NOT EXISTS idx_iv_interventions_critique ON inner_voice_interventions(triggered_by_critique_id);
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
# Inner Voice (#345) — critiques + interventions
# ---------------------------------------------------------------------------

def record_inner_voice_critique(
    session_id: str,
    turn_id: str,
    persona: str,
    model: str,
    *,
    persona_version: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    latency_ms: Optional[int] = None,
    disagrees: Optional[bool] = None,
    severity: Optional[float] = None,
    reason: Optional[str] = None,
    suggested_action: Optional[str] = None,
    action_taken: Optional[str] = None,
    anchor_response_excerpt: Optional[str] = None,
    event_log_offset: Optional[int] = None,
    raw_response_offset: Optional[int] = None,
    prompt_hash: Optional[str] = None,
    parse_attempts: int = 1,
) -> int:
    """Insert an Inner Voice critique row. Returns the new row's id."""
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO inner_voice_critiques
           (session_id, turn_id, persona, persona_version, model,
            input_tokens, output_tokens, latency_ms,
            disagrees, severity, reason, suggested_action, action_taken,
            anchor_response_excerpt, event_log_offset, raw_response_offset,
            prompt_hash, parse_attempts)
           VALUES (?, ?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?, ?, ?,  ?, ?, ?,  ?, ?)""",
        (session_id, turn_id, persona, persona_version, model,
         input_tokens, output_tokens, latency_ms,
         disagrees, severity, reason, suggested_action, action_taken,
         anchor_response_excerpt, event_log_offset, raw_response_offset,
         prompt_hash, parse_attempts),
    )
    conn.commit()
    return cur.lastrowid


def record_inner_voice_intervention(
    session_id: str,
    kind: str,
    target_turn_id: str,
    content: str,
    *,
    triggered_by_critique_id: Optional[int] = None,
) -> int:
    """Insert an Inner Voice intervention row. Returns the new row's id."""
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO inner_voice_interventions
           (session_id, triggered_by_critique_id, kind, target_turn_id, content)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, triggered_by_critique_id, kind, target_turn_id, content),
    )
    conn.commit()
    return cur.lastrowid


def list_inner_voice_critiques(
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """List critiques, newest first. Filters by session_id and/or turn_id."""
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
        f"""SELECT id, session_id, turn_id, persona, persona_version, model,
                   input_tokens, output_tokens, latency_ms,
                   disagrees, severity, reason, suggested_action, action_taken,
                   nudge_succeeded, anchor_response_excerpt,
                   event_log_offset, raw_response_offset, prompt_hash,
                   parse_attempts, created_at
            FROM inner_voice_critiques{where_sql}
            ORDER BY id DESC LIMIT ?""",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def list_inner_voice_interventions(
    session_id: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """List interventions, newest first. Filters by session_id."""
    conn = _conn()
    where_sql = " WHERE session_id = ?" if session_id else ""
    params: list = [session_id] if session_id else []
    rows = conn.execute(
        f"""SELECT id, session_id, triggered_by_critique_id, kind, target_turn_id,
                   content, created_at, outcome_turn_id, outcome_addressed,
                   outcome_summary, graded_at
            FROM inner_voice_interventions{where_sql}
            ORDER BY id DESC LIMIT ?""",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


# ── Stage 5: grading pass helpers ─────────────────────────────────────────


def list_ungraded_interventions(
    session_id: str,
    *,
    exclude_target_turn_id: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Return interventions in `session_id` whose outcome hasn't been graded
    yet, oldest-first.

    The grading pass calls this from `_run_turn` after every ambient turn
    completes. The just-finished turn becomes the candidate outcome. Pass
    its turn_id as `exclude_target_turn_id` so we don't grade an
    intervention against the very turn that triggered it (which would be
    a tautology).
    """
    conn = _conn()
    where = ["session_id = ?", "outcome_turn_id IS NULL"]
    params: list = [session_id]
    if exclude_target_turn_id:
        where.append("target_turn_id != ?")
        params.append(exclude_target_turn_id)
    where_sql = " WHERE " + " AND ".join(where)
    rows = conn.execute(
        f"""SELECT id, session_id, triggered_by_critique_id, kind, target_turn_id,
                   content, created_at, outcome_turn_id, outcome_addressed,
                   outcome_summary, graded_at
            FROM inner_voice_interventions{where_sql}
            ORDER BY id ASC LIMIT ?""",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def update_intervention_outcome(
    intervention_id: int,
    *,
    outcome_turn_id: str,
    outcome_addressed: Optional[bool],
    outcome_summary: str,
) -> None:
    """Backfill the grading verdict on `inner_voice_interventions`.

    `outcome_addressed` is tri-valued: True / False / None (ambiguous).
    `graded_at` set to `CURRENT_TIMESTAMP` server-side.
    """
    conn = _conn()
    conn.execute(
        """UPDATE inner_voice_interventions
              SET outcome_turn_id = ?,
                  outcome_addressed = ?,
                  outcome_summary = ?,
                  graded_at = CURRENT_TIMESTAMP
            WHERE id = ?""",
        (outcome_turn_id, outcome_addressed, outcome_summary, intervention_id),
    )
    conn.commit()


def grading_progress_since(start_iso: str) -> dict[str, Any]:
    """Return grading-pass coverage stats since `start_iso`.

    Used by `/api/inner_voice/state` and the meta-review notebook to answer
    "what fraction of recent interventions got graded?". Schema:

      {
        "total":   <int>,    -- interventions created since start_iso
        "graded":  <int>,    -- with non-null outcome_addressed
        "ratio":   <float>,  -- graded / total (0.0 if total == 0)
      }
    """
    conn = _conn()
    row = conn.execute(
        """SELECT
              COUNT(*)                                     AS total,
              COUNT(outcome_addressed)                     AS graded
            FROM inner_voice_interventions
           WHERE created_at >= ?""",
        (start_iso,),
    ).fetchone()
    total = int(row["total"] or 0) if row else 0
    graded = int(row["graded"] or 0) if row else 0
    return {
        "total": total,
        "graded": graded,
        "ratio": (graded / total) if total > 0 else 0.0,
    }
