"""
Token usage tracking — SQLite store for Anthropic API usage metrics.

Records per-request token counts and provides aggregation queries
for the Usage dashboard (4-hour window, 7-day window, time-series).
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

from app.paths import USAGE_DB

DB_PATH = USAGE_DB

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Thread-local SQLite connection, reopened if `DB_PATH` has moved.

    Reopening on a moved path is what lets a test point the store at a
    scratch file: the cache is per thread, and a connection a worker thread
    opened earlier would otherwise go on writing wherever it first pointed.
    """
    path = str(DB_PATH)
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != path:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _init_schema(conn)
        _local.conn = conn
        _local.path = path
    return conn


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
            num_turns       INTEGER,
            -- Prefix-cache misses (app/prefix_miss.py). NULL = not measured:
            -- a row from before the counter, or a turn whose cache_read read
            -- zero on every counted iteration (the unread-field signature).
            reprefill_tokens INTEGER,
            prefix_misses    INTEGER
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
            cache_read INTEGER,
            cache_create INTEGER,
            latency_ms INTEGER,
            model TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            safeguard TEXT                        -- deterministic rule that decided; NULL = the model (#770)
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
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(inner_voice_observations)").fetchall()}
    if "cache_read" not in existing_cols:
        conn.execute("ALTER TABLE inner_voice_observations ADD COLUMN cache_read INTEGER")
    if "cache_create" not in existing_cols:
        conn.execute("ALTER TABLE inner_voice_observations ADD COLUMN cache_create INTEGER")
    # Forward-only (#770): rows written before it read NULL, and iv_grade falls
    # back to its prose regex for exactly those. A backfill is a human call.
    if "safeguard" not in existing_cols:
        conn.execute("ALTER TABLE inner_voice_observations ADD COLUMN safeguard TEXT")
    usage_cols = {row["name"] for row in conn.execute("PRAGMA table_info(usage)").fetchall()}
    for col in ("reprefill_tokens", "prefix_misses"):
        if col not in usage_cols:
            conn.execute(f"ALTER TABLE usage ADD COLUMN {col} INTEGER")
    # The skill dimension (#783): a JSON array of {"name", "route"} objects
    # naming the skill bodies that were in front of the model for this turn and
    # how each got there. Additive, on the same path as the prefix-miss pair, so
    # a live `usage.db` keeps every row and gains NULLs. The live store held
    # 3,338 rows of real traffic at #783's triage (2026-09-18), so a recreated
    # table would have thrown away the dashboard's whole history. NULL means "no
    # skill recorded for this turn", which covers both a turn that delivered none
    # and every row written before this column; `skill_breakdown` leaves those
    # rows out rather than reading them as a skill that cost nothing.
    if "skills" not in usage_cols:
        conn.execute("ALTER TABLE usage ADD COLUMN skills TEXT")


def _skills_column(
    skills: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[str]:
    """Normalise a turn's skill deliveries into the stored JSON, or None.

    Accepts the dicts `skill_dispatch.skill_deliveries` produces (or
    `(name, route)` pairs) and keeps only entries that name a skill. Nothing here
    raises: a row's skill dimension is accounting attached to a token count, and
    the turn writers wrap the insert in a `try` whose failure would drop the token
    row with it. An empty list is stored as NULL, so "no skill" and "not recorded"
    stay the same value and neither is a fake zero.

    A bare string is refused, not split: `str` iterates its own characters, so the
    likeliest caller mistake — handing over `injected_skill_names()`'s set of
    names — would otherwise store `{"name": "a", "route": "l"}` for a skill called
    "alpha". A nameless entry and an entry with no route are both kept as they
    are, because "a skill, route unknown" is a fact and not a corruption.
    """
    entries: list[dict[str, str]] = []
    for item in skills or ():
        if isinstance(item, Mapping):
            name = str(item.get("name") or "")
            route = str(item.get("route") or "")
        elif isinstance(item, (tuple, list)) and item and isinstance(item[0], str):
            name = item[0]
            route = str(item[1]) if len(item) > 1 else ""
        else:
            continue
        if not name:
            continue
        entries.append({"name": name, "route": route})
    if not entries:
        return None
    return json.dumps(entries, sort_keys=True)


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
    reprefill_tokens: Optional[int] = None,
    prefix_misses: Optional[int] = None,
    skills: Optional[Sequence[Mapping[str, Any]]] = None,
):
    """Insert a single usage record.

    `reprefill_tokens` / `prefix_misses` are the turn's prefix-cache misses
    (`app/prefix_miss.py`). None is stored as NULL and means unmeasured, so
    a zero can never stand in for "could not tell".

    `skills` is the turn's skill deliveries — `[{"name", "route"}, ...]`, the
    output of `app.harness.skill_dispatch.skill_deliveries` over the text the
    turn was prompted with (#783). Nothing here decides which skills were
    delivered; the store only records what the turn's own writer saw.
    """
    conn = _conn()
    conn.execute(
        """INSERT INTO usage
           (session_id, model, input_tokens, output_tokens,
            cache_create, cache_read, cost_usd,
            duration_ms, duration_api_ms, num_turns,
            reprefill_tokens, prefix_misses, skills)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, model, input_tokens, output_tokens,
         cache_create, cache_read, cost_usd,
         duration_ms, duration_api_ms, num_turns,
         reprefill_tokens, prefix_misses, _skills_column(skills)),
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


def prefix_miss_summary(hours: float = 24) -> dict:
    """Prefix-cache misses over a window, for the dashboard.

    `turns_measured` counts the rows that carry the measurement at all. Rows
    from before the counter and turns whose cache_read never read non-zero
    are NULL and excluded, so "0 misses" beside "0 measured turns" cannot
    pass for a clean bill of health.
    """
    conn = _conn()
    row = conn.execute(
        """SELECT
             COUNT(*)                                        AS turns,
             COUNT(reprefill_tokens)                         AS turns_measured,
             COALESCE(SUM(CASE WHEN prefix_misses > 0 THEN 1 ELSE 0 END), 0)
                                                             AS turns_with_misses,
             COALESCE(SUM(prefix_misses), 0)                 AS prefix_misses,
             COALESCE(SUM(reprefill_tokens), 0)              AS reprefill_tokens,
             COALESCE(MAX(reprefill_tokens), 0)              AS worst_turn_reprefill
           FROM usage WHERE ts >= ?""",
        (_since(hours=hours),),
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


def skill_breakdown(
    hours: Optional[float] = None,
    days: Optional[float] = None,
    exclude_models: Optional[list[str]] = None,
) -> list[dict]:
    """Per skill-and-route token breakdown for a time window (#783).

    `model_breakdown`'s shape over the same rows, the same window and the same
    exclusions, with the dimension flipped from the model to the skill body that
    was in front of it: one row per (skill, route) that at least one turn in the
    window named. A turn naming two skills gives its full token counts to both
    rows — this is attribution ("what was being spent while skill X was in the
    prompt"), not a partition of the window's total, and `requests` counts the
    turns that named the skill.

    No dollar figure, on purpose: the write sites put `0.0` in `cost_usd` on
    every row, so a per-skill cost would be a column of zeros dressed up as a
    cost. Tokens are the only magnitude this store actually carries.

    Rows whose `skills` is NULL are absent, not zeroed: "no skill delivered" and
    "written before the column" are the same stored value, and neither is
    evidence about a skill. The explosion runs in Python over the window's rows
    rather than through SQLite's `json_each`, which is a build option — and a
    window is hundreds of rows, not millions, so portability costs nothing.
    """
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
        f"""SELECT skills, input_tokens, output_tokens,
                   cache_create, cache_read
            FROM usage{where_sql}""",
        params,
    ).fetchall()

    totals: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        raw = row["skills"]
        if not raw:
            continue
        try:
            deliveries = json.loads(raw)
        except (ValueError, TypeError):
            continue                      # not written by `record_usage`
        if not isinstance(deliveries, list):
            continue
        counted: set[tuple[str, str]] = set()
        for delivery in deliveries:
            if not isinstance(delivery, Mapping):
                continue
            name = str(delivery.get("name") or "")
            route = str(delivery.get("route") or "")
            if not name or (name, route) in counted:
                continue                  # a doubled pair would double the turn
            counted.add((name, route))
            bucket = totals.setdefault((name, route), {
                "skill": name, "route": route, "requests": 0,
                "input_tokens": 0, "output_tokens": 0,
                "cache_create": 0, "cache_read": 0,
            })
            bucket["requests"] += 1
            for column in ("input_tokens", "output_tokens",
                           "cache_create", "cache_read"):
                bucket[column] += int(row[column] or 0)
    return sorted(totals.values(),
                  key=lambda r: (-r["input_tokens"], r["skill"], r["route"]))


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
    cache_read: Optional[int] = None,
    cache_create: Optional[int] = None,
    latency_ms: Optional[int] = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
    safeguard: Optional[str] = None,
) -> int:
    """Insert one Inner Voice observation row. Returns the new row's id."""
    # `created_at` is written explicitly as local-naive ISO (`datetime.now().isoformat()`)
    # to match the format used by primary message timestamps. The SQLite default
    # `CURRENT_TIMESTAMP` would store UTC-naive in `YYYY-MM-DD HH:MM:SS`, which
    # the frontend timeline merge would then mis-parse as local time and shift
    # observations into the future by the local TZ offset.
    conn = _conn()
    cur = conn.execute(
        """INSERT INTO inner_voice_observations
           (session_id, turn_id, sequence_in_turn, trigger, action,
            reason, content, related_tool,
            input_tokens, output_tokens, cache_read, cache_create,
            latency_ms, model, error, created_at, safeguard)
           VALUES (?, ?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?, ?,  ?, ?, ?, ?, ?)""",
        (session_id, turn_id, sequence_in_turn, trigger, action,
         reason, content, related_tool,
         input_tokens, output_tokens, cache_read, cache_create,
         latency_ms, model, error, datetime.now().isoformat(), safeguard),
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
                   input_tokens, output_tokens, cache_read, cache_create,
                   latency_ms, model, error,
                   created_at, safeguard
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
