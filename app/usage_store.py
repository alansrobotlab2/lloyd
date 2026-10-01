"""
Token usage tracking — SQLite store for Anthropic API usage metrics.

Records per-request token counts and provides aggregation queries
for the Usage dashboard (4-hour window, 7-day window, time-series).
"""

import json
import os
import sqlite3
from pathlib import Path
import threading
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

from app.paths import USAGE_DB

DB_PATH = USAGE_DB

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Thread-local SQLite connection, reopened if the file behind `DB_PATH`
    is not the one this handle writes to.

    Comparing the PATH is not enough, and the half-truth cost two rounds of
    #1866: a same-path-but-replaced file keeps a live handle pointed at the
    old file, SQLite goes on writing to the unlinked inode without complaint,
    and `record_usage` returns success for a row that exists nowhere. pytest
    9 does exactly that to a test: it hands two param cases the SAME
    `tmp_path` directory name (`test_x__0` for both `""` and `"[]"`, brackets
    stripped) while giving the second a FRESH, EMPTY directory, so the
    fixture's `usage.db` is simply gone at the second case. The reopen-on-moved-
    path rule alone covers a path that changes; this covers the path that
    doesn't.

    The check is one `os.stat` per connection use. A file that has vanished
    (`FileNotFoundError`) counts as replaced: reconnecting recreates it, which
    is what a caller that asked to record a row expects.
    """
    path = str(DB_PATH)
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == path:
        try:
            same_file = os.stat(path).st_ino == getattr(_local, "ino", None)
        except OSError:
            same_file = False
        if same_file:
            return conn
        conn.close()
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_schema(conn)
    _local.conn = conn
    _local.path = path
    try:
        _local.ino = os.stat(path).st_ino
    except OSError:
        _local.ino = None      # unreadable: the next call reconnects and retries
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
            safeguard TEXT,                       -- deterministic rule that decided; NULL = the model (#770)
            verdict TEXT,                         -- a human's label: 'up' | 'down' | NULL (unlabelled)
            verdict_at TEXT
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
    # A human's thumbs on one intervention (IV plan R3). The proxies iv_grade
    # reports are the observer's own evidence; this is the one label nobody
    # else can write. `scripts/iv_label_export.py` copies labelled rows into
    # the tracked corpus under `eval/iv/`, because this table lives in
    # gitignored runtime data and the 2026-09-22 wipe took everything before it.
    if "verdict" not in existing_cols:
        conn.execute("ALTER TABLE inner_voice_observations ADD COLUMN verdict TEXT")
    if "verdict_at" not in existing_cols:
        conn.execute("ALTER TABLE inner_voice_observations ADD COLUMN verdict_at TEXT")
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


def read_session_rows_readonly(db_path, session_id: str, columns: list[str]) -> list[dict]:
    """`usage` rows of one session, opened read-only (`mode=ro`).

    For the autoresearch cost side (#2019), which prices a bench trial from the rows
    its recorded session wrote and must never write to, or migrate, a live usage.db.
    A column the file does not carry reads as None, as in `read_rows_readonly`.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        have = {r[1] for r in conn.execute("PRAGMA table_info(usage)")}
        sel = [c if c in have else f"NULL AS {c}" for c in columns]
        rows = conn.execute(f"SELECT {', '.join(sel)} FROM usage WHERE session_id = ? "
                            "ORDER BY id", (session_id,)).fetchall()
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


# ---------------------------------------------------------------------------
# Per-mechanism re-prefill attribution (#2027)
#
# `prefix_miss_summary` above says what a window's prefix misses cost. It cannot
# say which mechanism caused them, and the two of Lloyd's destructive layers
# that do cause them break the shared prefix at the FRONT, which is the expensive
# end: `app/compaction.py:505` makes the new conversation
# `[CS.summary_message(record)] + ...`, so position 0 changed and the longest
# common prefix of the old and new prompt is ~0, and
# `app/harness/microcompact.py:11` clears "oldest first, and stops" — restated at
# `:28`, "Clearing is oldest-first by design" — so the first cleared message is
# the break and the surviving suffix is nearly the whole conversation.
# arXiv:2609.37725v1 §4.3 prices a context edit as the re-prefill of everything
# AFTER its break point, so front-breaking edits are the costly kind, and
# `compaction` on each usage row (#1078) joined to `reprefill_tokens` (#1078's
# column, measured per turn by `app/prefix_miss.py`) is what lets that price be
# read per mechanism instead of per week.
# ---------------------------------------------------------------------------

#: The intra-turn relief ladder's rungs, in the order the ladder runs them.
#: `app/harness/loop.py:1708` states the order ("Four rungs, in cost order, each
#: running only while the meter still says...") and the appends that build
#: `report["rungs"]` follow it — images at :1795, tool_results at :1816,
#: reasoning at :1834, arguments at :1860, truncate at :1884. A stored rung name
#: carries its own count (`"reasoning:6"`); `_rung_base` strips it. Kept as a
#: constant rather than harvested from the window so a rung the window holds none
#: of still gets a printed row with `n=0`: a rung that stopped firing is a
#: result, and an absent row reads identically to a healthy machine.
RELIEF_RUNGS_IN_LADDER_ORDER: tuple[str, ...] = (
    "images", "tool_results", "reasoning", "arguments", "truncate",
)

#: A row whose relief passes freed tokens but whose first freeing pass named a
#: rung outside the ladder above. Expected `n=0`; a non-zero row here means the
#: ladder grew a rung this constant does not know, and
#: `reprefill_attribution()["unknown_relief_rungs"]` names it. Aggregating rather
#: than dropping is what keeps the printed Σ exhaustive.
RELIEF_OTHER = "relief:other"

#: A record that exists but names no mechanism and freed nothing on either path.
#: The live shape is `{"mechanisms": []}` with no `turn_start` block — 21 such rows
#: in the 2026-09-24..10-01 window. An entirely empty record never reaches here:
#: the writer's own sanitizer turns one into NULL (#1078), so it is
#: `NO_COMPACTION_RECORD`. This bucket is NOT the control:
#: `ran_noop` is a record that says the pass ran (`turn_start.ran`) and freed
#: nothing, which is why its 3.9% is a credible floor; a record that records no
#: mechanism at all is an absence of evidence and gets its own row so it can never
#: dilute the arm the 52x ratio is measured against.
NO_FREEING_EVIDENCE = "no_freeing_evidence"

#: A row with no `compaction` value at all: written before #1078's column, or by
#: a writer that recorded nothing. The item's arm A.
NO_COMPACTION_RECORD = "no_compaction_record"

#: A row carrying a record whose two freeing paths both freed nothing. This is
#: the CONTROL arm (the item's arm D): the pass was offered and did not edit, so
#: whatever re-prefill it paid is the floor every rung row is compared against.
RAN_NOOP = "ran_noop"

#: Printed order, and the tie-break for a row naming more than one mechanism:
#: front-most break first. Two mechanisms in one row is common (the top-reprefill
#: rows all carry `"mechanisms": ["microcompact", "relief:intra_turn"]`), and the
#: record holds ONE `reprefill_tokens` per row with no per-rung split —
#: `compaction.relief[].rungs` is name:count and `freed_tokens` is per whole pass
#: — so a row is assigned wholly to one bucket under this rule, which is what
#: makes the printed Σ exhaustive rather than a selection of interesting rungs.
#: Front-most is the defensible choice because §4.3's price is set by the
#: earliest break: everything after it re-prefills, so crediting a later edit
#: would price the cheap edit with the expensive one's bill. Within the
#: `turn_start:*` family the three front-position edits (a replaced position 0, a
#: fold of the summary that sits at the front, a truncation that drops the oldest
#: turns) all break at position ~0, so their relative order is a deterministic
#: tie-break, not a claim about which of them is dearer.
ATTRIBUTION_BUCKETS: tuple[str, ...] = (
    "turn_start:summarized",
    "turn_start:summary_folds",
    "turn_start:truncated",
    "turn_start:microcompact",
    "turn_start:other",
    *(f"relief:{rung}" for rung in RELIEF_RUNGS_IN_LADDER_ORDER),
    RELIEF_OTHER,
    RAN_NOOP,
    NO_FREEING_EVIDENCE,
    NO_COMPACTION_RECORD,
)


def _rung_base(rung: Any) -> str:
    """`"reasoning:6"` -> `"reasoning"`. The count suffix is the ladder's own
    tally of how many items that rung edited, not part of its name."""
    return str(rung).split(":", 1)[0].strip()


def _freeing_evidence(compaction: Any) -> dict:
    """What one row's `compaction` record says about freeing.

    Reads BOTH freeing paths, which is the whole instrument (#2027's own first
    draft read only `turn_start.*` and thereby dumped every relief-only turn into
    the control arm — the wrong predicate does not fail loudly, it prints a
    plausible weaker number): `turn_start.tokens_freed` for the turn-start pass
    and top-level `relief_tokens_freed` for the intra-turn ladder.

    A record that will not parse is read as a record that names no mechanism and
    freed nothing — the same reading `json_extract` gives it in SQL, which is what
    keeps the arms table reproducible from the shell. It is bucketed
    `NO_FREEING_EVIDENCE`, not `RAN_NOOP`, because nothing in it claims a pass ran.

    Returns `{"no_record": bool, "ran": bool, "turn_start_freed": int,
    "relief_freed": int, "turn_start_flags": [bucket, ...],
    "relief_rungs": [bucket, ...], "relief_rung_names": [rung, ...]}` where the
    two bucket lists are in front-most-first order and hold the buckets the row is
    *eligible* for, and `relief_rung_names` keeps the raw names so a rung the
    ladder constant does not know is still visible to the caller after its row is
    folded into `RELIEF_OTHER`. A NULL record is `no_record`: "the writer recorded
    nothing at all" is a different statement from "the record names nothing".
    """
    blank = {"no_record": False, "ran": False, "turn_start_freed": 0,
             "relief_freed": 0, "turn_start_flags": [], "relief_rungs": [],
             "relief_rung_names": []}
    if compaction is None:
        return dict(blank, no_record=True)
    if isinstance(compaction, str):
        try:
            compaction = json.loads(compaction)
        except (ValueError, TypeError):
            return blank
    if not isinstance(compaction, Mapping):
        return blank
    ts = compaction.get("turn_start")
    ts = ts if isinstance(ts, Mapping) else {}
    flags: list[str] = []
    if ts.get("summarized"):
        flags.append("turn_start:summarized")
    if _as_int(ts.get("summary_folds")) > 0:
        flags.append("turn_start:summary_folds")
    if ts.get("truncated"):
        flags.append("turn_start:truncated")
    if _as_int(ts.get("microcompacted")) > 0:
        flags.append("turn_start:microcompact")

    relief_freed = _as_int(compaction.get("relief_tokens_freed"))
    rungs: list[str] = []
    names: list[str] = []
    passes = compaction.get("relief")
    if isinstance(passes, list):
        for one in passes:
            if not isinstance(one, Mapping) or _as_int(one.get("freed_tokens")) <= 0:
                continue
            named = [_rung_base(r) for r in (one.get("rungs") or [])]
            names.extend(named)
            # The first rung of the pass that freed tokens, since the ladder
            # appends in run order (loop.py:1795-:1884) and so the earliest one
            # listed is the earliest one that edited. A name this file's ladder
            # constant does not know does not become an invisible row: the row is
            # folded into RELIEF_OTHER and the name is reported.
            known = [r for r in named if r in RELIEF_RUNGS_IN_LADDER_ORDER]
            rungs.append(f"relief:{known[0]}" if known else RELIEF_OTHER)

    if _as_int(ts.get("tokens_freed")) > 0 and not flags:
        flags.append("turn_start:other")
    return {
        "no_record": False,
        "ran": bool(ts.get("ran")),
        "turn_start_freed": _as_int(ts.get("tokens_freed")),
        "relief_freed": relief_freed,
        "turn_start_flags": flags,
        "relief_rungs": rungs,
        "relief_rung_names": names,
    }


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def attribution_bucket(compaction: Any) -> str:
    """The one bucket a turn's `compaction` record is attributed to.

    Precedence is `ATTRIBUTION_BUCKETS`: a row that freed via the turn-start pass
    is credited to the front-most turn-start flag that fired, a row whose only
    freeing signal is `relief_tokens_freed` is credited to the first rung of the
    first pass that freed tokens, and a row that freed by both paths is counted
    once — under the turn-start edit, which is nearer the front. A row with a
    record and no freeing on either path is `RAN_NOOP`, the control; a row with no
    record is `NO_COMPACTION_RECORD`.
    """
    ev = _freeing_evidence(compaction)
    if ev.get("no_record"):
        return NO_COMPACTION_RECORD
    if ev["turn_start_freed"] > 0:
        # Front-most turn-start flag that fired; `turn_start:other` only when the
        # pass freed tokens and none of the four named flags says how.
        for bucket in ev["turn_start_flags"]:
            return bucket
        return "turn_start:other"
    if ev["relief_freed"] > 0:
        return ev["relief_rungs"][0] if ev["relief_rungs"] else RELIEF_OTHER
    # Neither documented freeing path freed a token. A record whose turn-start
    # pass reports it ran is the control — arm D, the 3.9% every rung mean is
    # compared against, defined by "freed nothing" rather than by "ran", exactly
    # as the item's predicate defines it. A record that does not even claim a run
    # is the absence of evidence, and gets its own row.
    return RAN_NOOP if ev.get("ran") else NO_FREEING_EVIDENCE


def reprefill_attribution(hours: float = 168.0, *, since: Optional[str] = None,
                          until: Optional[str] = None) -> dict:
    """Price the window's re-prefill per mechanism, and show the split is exhaustive.

    The window is the last `hours` hours, or — when `since` is given — the absolute
    interval `since <= ts < until` (`until` optional). The absolute form exists for
    a figure that has already been published and must stay checkable after the
    clock moves: `replay_usage_extract` plus `REPREFILL_WITNESS_SINCE` re-answers
    the 2026-10-01 verdict over the same rows in a month's time. One row per bucket in
    `ATTRIBUTION_BUCKETS` — every rung, plus the control (`ran_noop`) and the
    unmeasured arm (`no_compaction_record`) — each carrying `n=` beside its mean,
    and a rung the window holds none of is printed with `n=0` rather than
    omitted. `report` is the text to print.

    **The total the table is exhaustive against is named in the header**: the
    row-level `COALESCE(SUM(reprefill_tokens), 0)` over EVERY usage row with
    `ts >= since`, with no validity filter. That choice is deliberate and stated
    because two figures are in play and mixing them silently is what this item's
    triage did — it took `prefix_miss_summary`'s `reprefill_tokens` for a
    validity-filtered turn-level total, when that SELECT carries no case logic on
    the column and is the SAME sum over the SAME window, equal to the named total
    byte for byte. The pair it measured was one expression over two WINDOWS: the
    `ts` column holds a literal 'T' and sqlite renders `datetime('now','-7 day')`
    with a space, and 'T' (0x54) sorts above the space (0x20), so the
    space-spelled predicate is wider. Both figures are printed, each labelled, and
    `share=` is against the named one.

    Every row in the window goes to exactly one bucket, so `bucket_sum` equals
    `window_total` exactly and `gap_pct` is 0.0 — a row that freed nothing is in
    the table too, which is the only way the sum can be exhaustive rather than a
    selection of interesting rungs. Means are over measured rows only
    (`measured=`), since a NULL `reprefill_tokens` is unmeasured, not zero.

    The `arms` block reproduces the three-arm table this item was filed with
    (`A_no_record` / `B_freed_something` / `D_ran_noop`) over measured rows, with
    both paths in the predicate, and `ratio_b_over_d_mean` beside
    `size_controlled`. It is False by construction here: arm B's mean
    `input_tokens` is roughly twice arm D's, so a mean-to-mean ratio is not
    size-controlled, and the within-session paired comparison that would control
    for it needs paired sessions this function does not attempt (#2027's owed
    clause). `MISS_RATIO_ABOVE_100_IS_NOT_A_BUG`: `reprefill_tokens` is summed
    over a turn's iterations while `input_tokens` is one request, so a share of
    prompt above 100% is arithmetic, not a mistake to "fix".
    """
    absolute = since is not None
    since = since or _since(hours=hours)
    conn = _conn()
    span = "ts >= ?" + (" AND ts < ?" if until is not None else "")
    span_args: list[Any] = [since] + ([until] if until is not None else [])
    # The same sum, asked with the separator sqlite's own datetime() emits. Printed
    # so a shell re-run reconciles instead of looking like a contradiction. Only
    # meaningful for an offset window, where a shell would spell `since` that way.
    sqlite_spelling_total = (None if absolute else _as_int(conn.execute(
        f"SELECT COALESCE(SUM(reprefill_tokens), 0) FROM usage WHERE {span}",
        [a.replace("T", " ") if isinstance(a, str) else a for a in span_args],
    ).fetchone()[0]))
    # The same `span` the total above is asked with: a row at or after `until` is
    # outside the window the header prints, so it belongs to neither the named
    # total nor any bucket. Dropping the bound here would leave the header and the
    # arms table claiming a closed interval while the rows behind them stayed open
    # — and an extract re-cut a day later than its `until` would quietly widen.
    rows = conn.execute(
        f"""SELECT compaction, reprefill_tokens, input_tokens
              FROM usage WHERE {span}""",
        span_args,
    ).fetchall()
    window_total = sum(_as_int(r["reprefill_tokens"]) for r in rows)

    counts: dict[str, dict[str, int]] = {}
    # Rows in which a relief rung edited something at all, whether or not this row
    # was ATTRIBUTED to it. A rung can hold `n=0` (no row is credited to it,
    # because an earlier rung in the same pass is nearer the front and takes the
    # row) while still firing on hundreds of rows, and printing the bare `n=0`
    # without this would read as "the truncate rung did not run this week".
    fired: dict[str, int] = {}

    def bump(bucket: str, reprefill: Optional[int], input_tokens: Any) -> None:
        cell = counts.setdefault(bucket, {"n": 0, "measured": 0, "sum": 0, "input": 0})
        cell["n"] += 1
        cell["input"] += _as_int(input_tokens)
        if reprefill is not None:
            cell["measured"] += 1
            cell["sum"] += _as_int(reprefill)

    arms: dict[str, dict[str, int]] = {}
    unknown_rungs: set[str] = set()
    for row in rows:
        raw = row["compaction"]
        ev = _freeing_evidence(raw)
        bucket = attribution_bucket(raw)
        bump(bucket, row["reprefill_tokens"], row["input_tokens"])
        for name in (ev.get("relief_rung_names") or []):
            bucket_name = f"relief:{name}"
            if bucket_name in ATTRIBUTION_BUCKETS:
                fired[bucket_name] = fired.get(bucket_name, 0) + 1
            else:
                unknown_rungs.add(bucket_name)
        if row["reprefill_tokens"] is None:
            continue        # unmeasured is not arm D; the arms table is measured-only
        # A literal transcription of the item's predicate, so a shell re-run of it
        # lands every row in the same arm: NULL record = A; either freeing path
        # >0 = B; a record that freed nothing = D.
        if ev.get("no_record"):
            arm = "A_no_record"
        elif ev["turn_start_freed"] > 0 or ev["relief_freed"] > 0:
            arm = "B_freed_something"
        else:
            arm = "D_ran_noop"
        cell = arms.setdefault(arm, {"n": 0, "input": 0, "reprefill": 0})
        cell["n"] += 1
        cell["input"] += _as_int(row["input_tokens"])
        cell["reprefill"] += _as_int(row["reprefill_tokens"])

    buckets = []
    for name in ATTRIBUTION_BUCKETS:
        cell = counts.get(name, {"n": 0, "measured": 0, "sum": 0, "input": 0})
        buckets.append({
            "bucket": name,
            "n": cell["n"],
            "measured": cell["measured"],
            "reprefill_tokens": cell["sum"],
            "mean_reprefill_tokens": (round(cell["sum"] / cell["measured"])
                                      if cell["measured"] else None),
            "mean_input_tokens": round(cell["input"] / cell["n"]) if cell["n"] else None,
            "share_pct": (round(100.0 * cell["sum"] / window_total, 1)
                          if window_total else 0.0),
            # Relief rungs only: rows whose record names this rung in a pass that
            # freed tokens, whether or not the row was CREDITED to it. Not
            # additive — one row can name three rungs — so it is never part of the
            # Σ, and it is what stops `n=0` reading as "this rung never ran".
            "fired_rows": (fired.get(name) if name.startswith("relief:") else None),
        })
    # No rows are appended for names outside `RELIEF_RUNGS_IN_LADDER_ORDER`:
    # their tokens are already inside `relief:other`, and a second row naming the
    # same tokens would double-count the Σ the header claims is exhaustive. The
    # name is reported instead, in `unknown_relief_rungs` and on the report's last
    # line, which is what tells a reader the ladder grew a rung.
    bucket_sum = sum(b["reprefill_tokens"] for b in buckets)
    arm_out = {}
    for arm in ("A_no_record", "B_freed_something", "D_ran_noop"):
        cell = arms.get(arm, {"n": 0, "input": 0, "reprefill": 0})
        arm_out[arm] = {
            "n": cell["n"],
            "mean_input_tokens": round(cell["input"] / cell["n"]) if cell["n"] else None,
            "mean_reprefill_tokens": (round(cell["reprefill"] / cell["n"])
                                      if cell["n"] else None),
            "reprefill_tokens": cell["reprefill"],
        }
    b_mean = arm_out["B_freed_something"]["mean_reprefill_tokens"]
    d_mean = arm_out["D_ran_noop"]["mean_reprefill_tokens"]
    ratio = (round(b_mean / d_mean, 1) if b_mean and d_mean else None)
    pms = ({} if absolute else prefix_miss_summary(hours=hours))

    out = {
        "window_hours": hours,
        "since": since,
        "rows": len(rows),
        "window_since": since,
        "window_until": until,
        "window_absolute": absolute,
        "window_total_reprefill_tokens": window_total,
        "window_total_named": (
            "row-level COALESCE(SUM(reprefill_tokens), 0) over every usage row "
            f"with {('ts >= ' + since + (' AND ts < ' + until) if absolute else 'ts >= ' + since)}"
            ", no validity filter"),
        # The one number the Σ is exhaustive against is `window_total_...` above.
        # These two are printed because a reader who checks this table with a shell
        # will meet both of them, and neither is that total.
        "sqlite_spelling_total_reprefill_tokens": sqlite_spelling_total,
        "window_spelling_note": (_WINDOW_SPELLING_NOTE if not absolute else
                                 "absolute window: quote this exact `since` (and "
                                 "`until`) — the spelling hazard below is why the "
                                 "interval is given literally"),
        "sqlite_spelling_total_named": (
            "COALESCE(SUM(reprefill_tokens),0) over ts >= "
            f"'{since.replace('T', ' ')}' — the same expression, the space-separated "
            "spelling sqlite's datetime() produces, which matches MORE rows than "
            "this window does. Not a different kind of total: a wider window"),
        "prefix_miss_summary_reprefill_tokens": pms.get("reprefill_tokens"),
        "prefix_miss_summary_note": (
            "prefix_miss_summary's reprefill_tokens field is the SAME "
            "COALESCE(SUM(reprefill_tokens),0) over the SAME window as the named "
            "total, so it is equal to it and is not a validity-filtered figure — "
            "only its turns/prefix_misses columns carry case logic. #2027's triage "
            "took the pair as row-level versus turn-level totals; the difference it "
            "measured was the window spelling above, and both of its numbers are "
            "correct sums over the windows they each named"),
        "bucket_sum_reprefill_tokens": bucket_sum,
        "gap_pct": (round(100.0 * abs(bucket_sum - window_total) / window_total, 2)
                    if window_total else 0.0),
        "buckets": buckets,
        "arms": arm_out,
        "ratio_b_over_d_mean": ratio,
        "size_controlled": False,
        "size_control_note": (
            "no — arm B's mean input_tokens is "
            f"{arm_out['B_freed_something']['mean_input_tokens']} against arm D's "
            f"{arm_out['D_ran_noop']['mean_input_tokens']}, so a mean-to-mean ratio "
            "carries the size difference; the within-session paired comparison is "
            "owed on #2027 and no ratio is quoted from it"),
        "unknown_relief_rungs": sorted(unknown_rungs),
    }
    out["report"] = _format_attribution(out)
    return out


_WINDOW_SPELLING_NOTE = (
    "quote this exact since string in any shell predicate: the ts column carries a "
    "literal 'T' between date and time, so a comparison written "
    "datetime('now','-7 day') — which sqlite renders with a SPACE — is "
    "lexicographically WIDER than this window ('T' 0x54 sorts above ' ' 0x20, so "
    "every row on the boundary day passes), and reaches back to 00:00 of that day. "
    "That is why #2027 saw '135,848,789 over 4101 rows' beside '112,908,568 over "
    "3868 rows' and read the pair as two kinds of total: both are "
    "COALESCE(SUM(reprefill_tokens),0), over two different windows.")

#: The vault extract the #2027 verdict is priced on: every `usage` row of the
#: 168 h window ending 2026-10-01T16:28:38, with `compaction` reduced to the
#: fields this module reads. A VAULT path, not a repo path — `~/obsidian/` plus
#: this — because a unit test reading the vault would be measuring whichever
#: machine ran it. What the repo carries is the reader below, so the extract is
#: re-derivable by anyone who has the vault commit.
REPREFILL_WITNESS = "backlog/data/2026-10-01.2027-reprefill-witness.jsonl"
REPREFILL_WITNESS_COMMIT = "93005612eeda3fdebd274432f8b47f0740cb42f4"
REPREFILL_WITNESS_ROWS = 3868
REPREFILL_WITNESS_TOTAL = 112_908_568
#: The `since` this extract's rows were cut on, verbatim: the extract is a
#: window, and #2027's own triage showed how one character of that window's
#: spelling moves its total by 22,940,221 tokens.
REPREFILL_WITNESS_SINCE = "2026-09-24T16:28:38"
REPREFILL_WITNESS_UNTIL = "2026-10-01T16:28:38"
_WITNESS_COLUMNS = ("ts", "session_id", "model", "input_tokens", "output_tokens",
                    "reprefill_tokens", "prefix_misses", "compaction")


def _encode_extract_value(column: str, value: Any) -> Any:
    """One extract value as it goes into the `usage` column.

    JSON `null` in the extract is SQL NULL, and that distinction is load-bearing,
    not cosmetic: a NULL `compaction` is `NO_COMPACTION_RECORD` (arm A), while the
    four-character string `'null'` parses to a non-mapping and lands in
    `NO_FREEING_EVIDENCE` instead. Dumping nulls through `json.dumps` moved 1,458
    rows out of arm A in a trial replay, took 7,195,735 tokens off that arm's mean
    and shifted the headline ratio from 51.8 to 52.3 — a wrong headline with every
    Σ still intact, which is the shape of error a total check does not catch.
    """
    if value is None:
        return None
    if column == "compaction" and not isinstance(value, str):
        return json.dumps(value, sort_keys=True)
    return value


def replay_usage_extract(extract: Path, *, db_path: Optional[Path] = None) -> Path:
    """Load a usage-row extract into a `usage` table; return the file holding it.

    The mechanism behind "the numbers are re-derivable from committed bytes":
    point `DB_PATH` at the result and `reprefill_attribution` answers over the
    extract exactly as it does over the live database.

        from app import usage_store
        db = usage_store.replay_usage_extract(
            Path.home() / "obsidian" / usage_store.REPREFILL_WITNESS, db_path=Path("/tmp/w.db"))
        usage_store.DB_PATH = db
        attr = usage_store.reprefill_attribution(
            since=usage_store.REPREFILL_WITNESS_SINCE, until=usage_store.REPREFILL_WITNESS_UNTIL)

    `record_usage` is deliberately NOT used: it takes no `ts`, so a historical row
    cannot be written through the public writer, and a replay that stamped rows
    "now" would answer a different window than the one priced. So this inserts
    directly, onto the schema `_init_schema` builds — the module's own, migrations
    included, never a hand-copied column list, which is the difference between a
    replay and a replay that dies on `no such column: exit_plan_mode`.

    A row carrying a key that is not a `usage` column raises rather than dropping
    silently: an extract that grew a field this reader does not load would
    otherwise reproduce the tables while quietly not containing the reason.
    """
    extract = Path(extract)
    db_path = Path(db_path) if db_path else extract.with_suffix(".replay.sqlite")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row      # _init_schema reads PRAGMA rows by name
    try:
        _init_schema(conn)
        rows, inserted = [], 0
        for lineno, line in enumerate(extract.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            unknown = sorted(k for k in row if k not in _WITNESS_COLUMNS)
            if unknown:
                raise ValueError(
                    f"{extract.name}:{lineno}: extract row carries column(s) "
                    f"{unknown} that `usage` does not have; refusing to drop them")
            rows.append(row)
            cols = [c for c in _WITNESS_COLUMNS if c in row]
            conn.execute(
                f"INSERT INTO usage ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [_encode_extract_value(c, row[c]) for c in cols])
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    if not inserted:
        raise ValueError(f"{extract}: no rows — an empty extract proves nothing")
    return db_path


def _format_attribution(attr: Mapping[str, Any]) -> str:
    """The printable table `reprefill_attribution` collects. One line per bucket,
    `n=` on every line, `n=0` for an empty rung, `mean=-` where there is no
    measured row to average."""
    lines = [
        (f"reprefill attribution — ABSOLUTE window: {attr['window_since']} <= ts "
         f"< {attr['window_until']} — a published figure, it does not move with "
         f"the clock — rows={attr['rows']}"
         if attr["window_absolute"] else
         f"reprefill attribution — window: last {attr['window_hours']} h "
         f"(since {attr['since']}), rows={attr['rows']}"),
        f"named total: {attr['window_total_named']}"
        f" = {attr['window_total_reprefill_tokens']:,}",
        *(["NOT the named total, and here for reconciliation only: "
           f"{attr['sqlite_spelling_total_named']}"
           f" = {attr['sqlite_spelling_total_reprefill_tokens']:,}"]
          if attr["sqlite_spelling_total_reprefill_tokens"] is not None else []),
        *([f"{attr['prefix_miss_summary_note']}"
           f" = {attr['prefix_miss_summary_reprefill_tokens']:,}"]
          if attr["prefix_miss_summary_reprefill_tokens"] is not None else []),
        (f"Σ over every bucket below = {attr['bucket_sum_reprefill_tokens']:,} "
         f"(gap to the named total: {attr['gap_pct']}%) — every row in the window "
         "is attributed to exactly one bucket"),
        "",
        "one row per bucket, printed whether or not the window holds a row for it;",
        "columns after the name: n= rows CREDITED to this bucket, measured= rows",
        "carrying a reprefill_tokens (a NULL is unmeasured, never 0, so every mean",
        "is taken over measured= and not over n=), Σ reprefill=, mean= over",
        "measured=, mean input= over n=, share= of the named total. Relief rows",
        "also carry fired_= rows whose record names that rung in a freeing pass:",
        "not additive (a row can name several rungs) and never part of the Σ, so",
        "n=0 with a fired_>0 says the rung ran but an earlier rung in the same",
        "pass is nearer the front and owns the row.",
        "",
    ]
    for row in attr["buckets"]:
        mean = ("-" if row["mean_reprefill_tokens"] is None
                else f"{row['mean_reprefill_tokens']:,}")
        min_ = ("-" if row["mean_input_tokens"] is None
                else f"{row['mean_input_tokens']:,}")
        line = (f"{row['bucket']:<26} n={row['n']:>6,} "
                f"measured={row['measured']:>6,} "
                f"Σ reprefill={row['reprefill_tokens']:>13,} mean={mean:>10} "
                f"mean input={min_:>9} share={row['share_pct']:>5}%")
        if row.get("fired_rows") is not None:
            line += f" fired_={row['fired_rows']:,}"
        lines.append(line)
    lines += ["", "arms (measured rows only, both freeing paths in the predicate):"]
    for arm in ("A_no_record", "B_freed_something", "D_ran_noop"):
        cell = attr["arms"][arm]
        lines.append(
            f"  {arm:<18} n={cell['n']:,} mean input="
            f"{'-' if cell['mean_input_tokens'] is None else format(cell['mean_input_tokens'], ',')} "
            f"mean reprefill="
            f"{'-' if cell['mean_reprefill_tokens'] is None else format(cell['mean_reprefill_tokens'], ',')} "
            f"Σ reprefill={cell['reprefill_tokens']:,}")
    lines.append(f"  B/D mean ratio = {attr['ratio_b_over_d_mean']}  "
                 f"size-controlled: {attr['size_controlled']} "
                 f"({attr['size_control_note']})")
    lines += ["", f"  {attr['window_spelling_note']}"]
    if attr["unknown_relief_rungs"]:
        lines.append(f"  unknown relief rungs folded into {RELIEF_OTHER}: "
                     f"{', '.join(attr['unknown_relief_rungs'])}")
    return "\n".join(lines)


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
                   created_at, safeguard, verdict, verdict_at
            FROM inner_voice_observations{where_sql}
            ORDER BY id DESC LIMIT ?""",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


IV_VERDICTS = ("up", "down")


def set_inner_voice_observation_verdict(obs_id: int, verdict: Optional[str]) -> bool:
    """Label one observation `up`/`down`, or clear it with None.

    Returns False when no row has that id. Anything else is refused by the
    caller before it gets here; a verdict outside the pair raises.
    """
    if verdict is not None and verdict not in IV_VERDICTS:
        raise ValueError(f"verdict must be one of {IV_VERDICTS} or None")
    conn = _conn()
    cur = conn.execute(
        "UPDATE inner_voice_observations SET verdict = ?, verdict_at = ? WHERE id = ?",
        (verdict, datetime.now().isoformat() if verdict else None, int(obs_id)),
    )
    conn.commit()
    return cur.rowcount > 0


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
