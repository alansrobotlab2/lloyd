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
    # The context-policy dimension (#1078): a JSON object naming which
    # mechanism rewrote this turn's history and how much it freed. Additive on
    # the same path as the skill column for the same reason — the live store is
    # the dashboard's only history, so a recreated table would discard it.
    # NULL here means UNMEASURED: a row from before this column, or a turn
    # whose writer never reached either mechanism. It is never the value of "a
    # mechanism fired and removed nothing" — that is a record whose
    # `tokens_freed` is 0, which `app/compaction_record.py` guarantees by
    # returning a record for any pass that ran a rung.
    if "compaction" not in usage_cols:
        conn.execute("ALTER TABLE usage ADD COLUMN compaction TEXT")
    # Harness telemetry (review 2026-09-24, P11): how the turn ended, how long
    # the engine took to start answering, what its tools cost and how they
    # failed. Additive on the same path as the columns above, for the same
    # reason. NULL = unmeasured — every row before this landed — never zero.
    for col, kind in _TELEMETRY_COLUMNS:
        if col not in usage_cols:
            conn.execute(f"ALTER TABLE usage ADD COLUMN {col} {kind}")


# (column, SQL type), in insert order. `tool_errors_by_class` is a JSON object
# {class: count}; the classes are `app.harness.events.TOOL_ERROR_CLASSES`.
_TELEMETRY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("stop_reason", "TEXT"),
    ("reasoning_tokens", "INTEGER"),
    ("ttft_ms_first", "INTEGER"),
    ("ttft_ms_max", "INTEGER"),
    ("tool_calls", "INTEGER"),
    ("tool_errors", "INTEGER"),
    ("tool_errors_by_class", "TEXT"),
    ("tool_ms_total", "INTEGER"),
    ("stream_retries", "INTEGER"),
    ("overflow_recoveries", "INTEGER"),
    ("hook_raised", "INTEGER"),
    ("wrapped_up", "INTEGER"),
)


def _telemetry_values(values: Mapping[str, Any]) -> tuple:
    """The telemetry kwargs in column order, normalised, never raising.

    A count that is not an int is stored NULL rather than failing the insert:
    this sits inside the writers' `try`, and a raise there drops the whole
    token row with it.
    """
    out: list[Any] = []
    for col, kind in _TELEMETRY_COLUMNS:
        value = values.get(col)
        if value is None:
            out.append(None)
        elif col == "tool_errors_by_class":
            if isinstance(value, str):
                out.append(value)
            elif isinstance(value, Mapping):
                try:
                    out.append(json.dumps(
                        {str(k): int(v) for k, v in value.items()},
                        sort_keys=True))
                except (TypeError, ValueError):
                    out.append(None)
            else:
                out.append(None)
        elif kind == "TEXT":
            out.append(str(value))
        else:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                out.append(None)
    return tuple(out)


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


def _compaction_column(compaction: Any) -> Optional[str]:
    """Normalise a turn's context-policy record into the stored JSON, or None.

    Accepts what `app/compaction_record.py` hands over — a `TurnCompaction`
    (asked for its `to_record()`), a plain mapping, or None — and stores None
    as NULL, meaning *unmeasured*. An empty mapping is NULL too, because an
    empty record is what "measured nothing" looks like once built; a mechanism
    that fired and freed nothing is a non-empty record with a zero in it, which
    is stored, not dropped.

    Nothing here raises, for the reason `record_usage`'s callers give: the
    insert sits inside a `try` whose failure drops the token row with it, so
    accounting that can throw is accounting that costs a turn its usage. The
    one real failure mode — a value JSON cannot encode — is contained by
    `default=str`, which can only fail on an object whose own `str()` throws.
    """
    to_record = getattr(compaction, "to_record", None)
    if callable(to_record):
        try:
            compaction = to_record()
        except Exception:  # noqa: BLE001 — never lose the token row over this
            return None
    if not compaction:
        return None
    if isinstance(compaction, str):
        # A caller that already serialised its record still lands, rather than
        # being stored as a JSON string of a string no reader would parse. But
        # the column's contract is an OBJECT, so only an object passes: `"[]"`
        # is valid JSON and would otherwise be stored as a record that every
        # reader's `.get("mechanisms")` reads as empty — a shape that looks
        # measured and is not.
        try:
            parsed = json.loads(compaction)
        except ValueError:
            return None
        if not isinstance(parsed, dict) or not parsed:
            return None
        return compaction
    if not isinstance(compaction, Mapping):
        return None
    try:
        return json.dumps(dict(compaction), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


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
    compaction: Any = None,
    **telemetry: Any,
):
    """Insert a single usage record.

    `reprefill_tokens` / `prefix_misses` are the turn's prefix-cache misses
    (`app/prefix_miss.py`). None is stored as NULL and means unmeasured, so
    a zero can never stand in for "could not tell".

    `skills` is the turn's skill deliveries — `[{"name", "route"}, ...]`, the
    output of `app.harness.skill_dispatch.skill_deliveries` over the text the
    turn was prompted with (#783). Nothing here decides which skills were
    delivered; the store only records what the turn's own writer saw.

    `compaction` is the turn's context-policy record (#1078): which of Lloyd's
    mechanisms rewrote the history this turn was prompted with, and how many
    tokens each removed. Like the two dimensions above, nothing here decides
    what fired — it stores what the turn's own writer saw, and NULL means that
    writer measured nothing.

    `**telemetry` takes the P11 columns (`_TELEMETRY_COLUMNS`), which both
    writers produce with `app.turn_usage.TurnTelemetry.row()`. An unknown key
    is a caller's typo and raises, like any other bad keyword would.
    """
    unknown = set(telemetry) - {c for c, _ in _TELEMETRY_COLUMNS}
    if unknown:
        raise TypeError(f"record_usage: unknown telemetry column(s) {sorted(unknown)}")
    tel_cols = ", ".join(c for c, _ in _TELEMETRY_COLUMNS)
    tel_marks = ", ".join("?" for _ in _TELEMETRY_COLUMNS)
    conn = _conn()
    conn.execute(
        f"""INSERT INTO usage
           (session_id, model, input_tokens, output_tokens,
            cache_create, cache_read, cost_usd,
            duration_ms, duration_api_ms, num_turns,
            reprefill_tokens, prefix_misses, skills, compaction,
            {tel_cols})
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {tel_marks})""",
        (session_id, model, input_tokens, output_tokens,
         cache_create, cache_read, cost_usd,
         duration_ms, duration_api_ms, num_turns,
         reprefill_tokens, prefix_misses, _skills_column(skills),
         _compaction_column(compaction), *_telemetry_values(telemetry)),
    )
    conn.commit()


def read_rows_readonly(db_path, columns: list[str], since_ts: str) -> list[dict]:
    """`usage` rows at or after `since_ts`, opened read-only (`mode=ro`).

    For offline readers (eval/run_context_rot_eval.py's cost side) that must
    never write to — or migrate — a live usage.db, and that `eval/` may not
    open with sqlite3 itself (tests/test_counterfactual_eval.py). A column the
    file does not carry yet reads as None rather than failing the query.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        have = {r[1] for r in conn.execute("PRAGMA table_info(usage)")}
        sel = [c if c in have else f"NULL AS {c}" for c in columns]
        rows = conn.execute(f"SELECT {', '.join(sel)} FROM usage WHERE ts >= ?",
                            (since_ts,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


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


def stop_reason_breakdown(hours: float = 24) -> list[dict]:
    """How turns ended over a window, most common first (P11).

    Only rows that carry a `stop_reason` count: older rows are unmeasured,
    and folding them in as "unknown" would drown the window's real mix in
    history. `turns_measured` beside it is the same guard `prefix_miss_summary`
    keeps.
    """
    conn = _conn()
    rows = conn.execute(
        """SELECT stop_reason,
                  COUNT(*)                          AS turns,
                  COALESCE(SUM(wrapped_up), 0)      AS wrapped_up
             FROM usage
            WHERE ts >= ? AND stop_reason IS NOT NULL
         GROUP BY stop_reason
         ORDER BY turns DESC, stop_reason""",
        (_since(hours=hours),),
    ).fetchall()
    return [dict(r) for r in rows]


def tool_error_breakdown(hours: float = 24) -> dict:
    """Tool calls, failures and failures by class over a window (P11).

    The by-class tally is summed in Python over the JSON column for the
    reason `skill_breakdown` gives: `json_each` is a build option, and a day
    is hundreds of rows.
    """
    conn = _conn()
    rows = conn.execute(
        """SELECT tool_calls, tool_errors, tool_errors_by_class, tool_ms_total
             FROM usage
            WHERE ts >= ? AND tool_calls IS NOT NULL""",
        (_since(hours=hours),),
    ).fetchall()
    by_class: dict[str, int] = {}
    calls = errors = ms = 0
    for row in rows:
        calls += int(row["tool_calls"] or 0)
        errors += int(row["tool_errors"] or 0)
        ms += int(row["tool_ms_total"] or 0)
        raw = row["tool_errors_by_class"]
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        for cls, n in parsed.items():
            try:
                by_class[str(cls)] = by_class.get(str(cls), 0) + int(n)
            except (TypeError, ValueError):
                continue
    return {
        "turns_measured": len(rows),
        "tool_calls": calls,
        "tool_errors": errors,
        "tool_ms_total": ms,
        "by_class": dict(sorted(by_class.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def ttft_summary(hours: float = 24) -> dict:
    """Time to first token over a window (P11): the first iteration's TTFT
    per turn (the cold prefill of the turn's history) and the worst of any
    iteration. Percentiles in Python — SQLite has none built in."""
    conn = _conn()
    rows = conn.execute(
        """SELECT ttft_ms_first, ttft_ms_max, reasoning_tokens
             FROM usage
            WHERE ts >= ? AND ttft_ms_first IS NOT NULL""",
        (_since(hours=hours),),
    ).fetchall()
    firsts = sorted(int(r["ttft_ms_first"]) for r in rows)
    maxes = [int(r["ttft_ms_max"]) for r in rows if r["ttft_ms_max"] is not None]

    def _pct(values: list[int], q: float) -> Optional[int]:
        if not values:
            return None
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    return {
        "turns_measured": len(firsts),
        "first_p50_ms": _pct(firsts, 0.5),
        "first_p90_ms": _pct(firsts, 0.9),
        "max_ms": max(maxes) if maxes else None,
    }


def reasoning_tokens_summary(hours: float = 24) -> dict:
    """Reasoning tokens over a window, against the output they are part of."""
    conn = _conn()
    row = conn.execute(
        """SELECT COUNT(reasoning_tokens)            AS turns_measured,
                  COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                  COALESCE(SUM(CASE WHEN reasoning_tokens IS NOT NULL
                                    THEN output_tokens ELSE 0 END), 0)
                                                     AS output_tokens
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
