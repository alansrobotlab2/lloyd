"""#544 — exactly-once *effect*, not exactly-once scheduling.

An agent's first move on any failure is to retry. A timeout is therefore the
dangerous case, not the merely annoying one: the server-side effect may have
landed while the caller was handed an error. Munaf's line (AI Engineer, TikTok)
is that a timeout has never meant failure — it means **unknown** — and the only
deterministic answer is a key on every side-effecting call, a status lookup that
resolves "unknown" before a new effect is allowed, and a record of what already
fired so a retried run can tell.

Lloyd already had exactly-once *scheduling* and nothing below it. The work
queue's `dedup_key` prevents a duplicate *enqueue* and is NULLed on every
terminal state (`mark_completed` / `mark_failed` / `mark_poisoned`), so by
construction it cannot prevent a duplicated **effect**. Meanwhile the pool
cancels a job at `max_duration_seconds`, records the run failed, requeues it up
to `max_attempts`, and the next attempt re-runs the whole turn from zero — every
`email_send` / `backlog_write_task` / `vault_write` / `fact_add` the first
attempt already fired fires again.

Shape
-----
One row per (tool, canonical arguments, scope) in `tool_effects`, in the same
SQLite file as the work queue — same file because the scope *is* the queue item,
and because the backend reads the suppression counter off it (`/api/dashboard`)
while the aggregator is the process writing it. WAL, as everywhere else here.

    unknown → ok | error

The `unknown` row is written **before** dispatch. That pre-write is the whole
point: it is the only record that survives being cancelled mid-call, and a row
still `unknown` after a retry means "the effect may have landed and nobody
knows" — which is refused, not re-fired.

Rules that are not obvious
--------------------------
* **A row is claimed with `INSERT OR IGNORE`, and the winner dispatches.** Two
  concurrent identical calls cannot both fire; a `status='error'` row is taken
  over by a guarded `UPDATE ... WHERE status='error'` so exactly one retry
  inherits it.
* **A cancelled call leaves the row `unknown` on purpose.** `finish` runs on a
  returned result and on an `isError` result; `CancelledError` re-raises without
  touching it. Marking a cancelled call `error` (i.e. retriable) would be the
  same lie the queue's timeout branch tells today.
* **Read-only tools never touch this module at all** — not "cheaply", not
  "cached": the first thing `claim()` does after the scope check is a frozenset
  membership test, and classification comes from `agent_mcp.annotations`, the
  same table plan mode and the UI badges read.
* **Failure is fail-open.** If the ledger DB is locked or missing, the call
  dispatches and the incident is logged. A ledger that can brick every write in
  the fleet because one WAL checkpoint is behind is a worse outage than the
  duplicate email it exists to prevent.
* **Scope is a queue item, not a source.** `item:<source>:<id>` is stable across
  that item's attempts (which is the case being fixed) and never reused, so a
  tomorrow run of the same job is a fresh scope. Keying on the *grant* scope
  (`autonomy-task:39`) instead would suppress a legitimate second effect forever
  across every future run of that task — over-suppression is the failure that
  bites here, so the narrow key is the correct one.
* **An empty scope means no ledger.** A worker item is the case with measured
  re-fire traffic (63 items reached `attempts>=2` in 8 days) and unambiguous
  semantics. Interactive turns are deliberately unscoped in this first cut:
  within one turn, re-issuing an identical call is often the model retrying
  after fixing a cause, and a silent replay of a stored result there would be a
  stale answer delivered as a fresh one. See the `REPEAT_EXPECTED` note in
  `annotations.py`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_mcp import annotations as tool_annotations

logger = logging.getLogger("lloyd-tool-effects")

# `_meta` key carrying the effect scope. Must match
# app/harness.mcp_pool.META_EFFECT_SCOPE, which is asserted in
# tests/test_tool_effects.py for the same reason test_change_ledger.py pins the
# other keys: the two halves ship from one repo but speak over HTTP, so a
# renamed key fails silently — the ledger just stops seeing a scope and every
# effect stops being guarded.
META_EFFECT_SCOPE = "lloyd/effect_scope"

# A stored replay is capped, not truncated into nonsense: past this the row
# keeps its digest and reports that the body is unavailable rather than
# handing back half a JSON document as if it were the tool's answer.
RESULT_MAX_CHARS = 8000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_effects (
    effect_key      TEXT PRIMARY KEY,
    tool            TEXT NOT NULL,
    scope           TEXT NOT NULL,
    session_id      TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL,
    result_digest   TEXT NOT NULL DEFAULT '',
    result_text     TEXT NOT NULL DEFAULT '',
    result_truncated INTEGER NOT NULL DEFAULT 0,
    suppress_count  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tool_effects_scope_tool_idx
    ON tool_effects(scope, tool);
"""

_init_lock = threading.Lock()
_init_done: set[str] = set()


def db_path() -> Path:
    """The file the ledger lives in: the work queue's own database.

    `workers.queue.configured_db_path()` exists for exactly this — the
    aggregator is a different process from the backend, so it must ask for the
    configured path rather than invent one. `LLOYD_EFFECT_LEDGER_DB` overrides it
    for tests and rebuilds, the way `LLOYD_KG_DB` does for the graph store.
    """
    raw = os.environ.get("LLOYD_EFFECT_LEDGER_DB")
    if raw:
        return Path(raw).expanduser()
    from workers.queue import configured_db_path
    return configured_db_path()


def _connect(create: bool = True) -> sqlite3.Connection | None:
    path = db_path()
    if not create and not path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure(conn: sqlite3.Connection) -> None:
    """Create the table once per process per file.

    Keyed by resolved path, not a bare flag: a test that moves the ledger to a
    second tmp file must still get its schema.
    """
    key = str(db_path().resolve())
    if key in _init_done:
        return
    with _init_lock:
        if key in _init_done:
            return
        conn.executescript(_SCHEMA)
        _init_done.add(key)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_arguments(arguments: Any) -> str:
    """The argument form the key is computed from.

    Sorted keys and fixed separators so two attempts that disagree only about
    dict ordering still collide. The model's `summary` caption is not in here —
    it rides in `_meta`, which is precisely why a reworded caption cannot
    change a key. `_session_id` is stripped by the aggregator before this is
    called, and any field that legitimately varies per call must not be an
    argument at all.
    """
    try:
        return json.dumps(arguments if isinstance(arguments, dict) else {},
                          sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, default=str)
    except Exception:
        return repr(arguments)


def effect_key(tool: str, arguments: Any, scope: str) -> str:
    return hashlib.sha256(
        f"{tool}\x00{canonical_arguments(arguments)}\x00{scope}".encode("utf-8")
    ).hexdigest()


def ledgered(name: str) -> bool:
    """Does this tool's second identical call risk a second durable effect?

    Classification comes from `agent_mcp.annotations` — the one ordered table the
    rest of the system already trusts — rather than a second list maintained
    beside it. The read-only check lives inside `side_effecting`, and it is the
    first thing that runs: a frozenset membership test, which is what keeps the
    ledger off a read-only call's path entirely — no connect, no key, no SELECT.
    """
    return tool_annotations.side_effecting(name)


class Claim:
    """The guard's answer to "may this call fire?".

    `may_dispatch` False means one of two things, told apart by `unknown`:
    the effect is already recorded `ok` (so the stored result is replayed), or
    its state is genuinely unknown (so nothing fires and the caller is told to
    look). `key` is empty when the call was not ledgered at all.
    """

    __slots__ = ("key", "may_dispatch", "unknown", "replay_text",
                 "replay_truncated", "replay_digest")

    def __init__(self, *, key: str = "", may_dispatch: bool = True,
                 unknown: bool = False, replay_text: str = "",
                 replay_truncated: bool = False, replay_digest: str = ""):
        self.key = key
        self.may_dispatch = may_dispatch
        self.unknown = unknown
        self.replay_text = replay_text
        self.replay_truncated = replay_truncated
        self.replay_digest = replay_digest


_UNLEDGERED = Claim()          # shared: immutable in practice, never written back


def _bump(conn: sqlite3.Connection, key: str) -> None:
    """Count a refused duplicate. This counter is what `/api/dashboard` shows,
    and it has to be non-zero within a week of real traffic or the guard is
    decoration — so both refusal branches count, not just the clean replay."""
    conn.execute(
        "UPDATE tool_effects SET suppress_count = suppress_count + 1,"
        " updated_at=? WHERE effect_key=?", (_now(), key))


def _claim(name: str, arguments: Any, scope: str, session_id: str) -> Claim:
    key = effect_key(name, arguments, scope)
    conn = _connect()
    try:
        _ensure(conn)
        now = _now()
        conn.execute(
            "INSERT OR IGNORE INTO tool_effects"
            " (effect_key, tool, scope, session_id, status, created_at, updated_at)"
            " VALUES (?,?,?,?, 'unknown', ?, ?)",
            (key, name, scope, session_id or "", now, now),
        )
        if conn.execute("SELECT changes()").fetchone()[0] == 1:
            # We wrote the row: we dispatch. Status stays `unknown` until the
            # call actually returns — that is the record a cancel cannot erase.
            return Claim(key=key)

        row = conn.execute(
            "SELECT status, result_text, result_truncated, result_digest, suppress_count"
            " FROM tool_effects WHERE effect_key=?", (key,)
        ).fetchone()
        if row is None:
            # Deleted between the INSERT and the SELECT by nothing in this code;
            # treat it as ours rather than lose the call.
            return Claim(key=key)

        status = row["status"]
        if status == "ok":
            _bump(conn, key)
            return Claim(key=key, may_dispatch=False,
                         replay_text=row["result_text"] or "",
                         replay_truncated=bool(row["result_truncated"]),
                         replay_digest=row["result_digest"] or "")
        if status == "unknown":
            # A prior attempt was cancelled with this effect in flight. Refuse
            # rather than fire a second one; the caller does the status lookup.
            _bump(conn, key)
            return Claim(key=key, may_dispatch=False, unknown=True)

        # status == 'error': the tool answered that it failed. Only one caller
        # may inherit that row, so the takeover is a guarded update.
        cur = conn.execute(
            "UPDATE tool_effects SET status='unknown', updated_at=?"
            " WHERE effect_key=? AND status='error'", (_now(), key))
        if cur.rowcount == 1:
            return Claim(key=key)
        # Somebody else took it first. Re-read and answer for the state it is
        # in now rather than pretending this call may fire.
        row = conn.execute(
            "SELECT status, result_text, result_truncated, result_digest"
            " FROM tool_effects WHERE effect_key=?", (key,)).fetchone()
        if row is None or row["status"] == "error":
            # Vanished, or settled failed again while we looked: treat it as
            # unknown rather than start a third firing of the same effect.
            return Claim(key=key, may_dispatch=False, unknown=True)
        if row["status"] == "ok":
            _bump(conn, key)
            return Claim(key=key, may_dispatch=False,
                         replay_text=row["result_text"] or "",
                         replay_truncated=bool(row["result_truncated"]),
                         replay_digest=row["result_digest"] or "")
        return Claim(key=key, may_dispatch=False, unknown=True)
    finally:
        conn.close()


async def claim(name: str, arguments: Any, scope: str, session_id: str = "") -> Claim:
    """Decide whether an effect may fire, and reserve it if it may.

    Off the event loop: the pool's own docstring is explicit that one loop
    serves every HTTP request and streams every turn, so a WAL write here has
    to hop threads like the queue's writes do.
    """
    if not scope or not ledgered(name):
        return _UNLEDGERED
    try:
        return await asyncio.to_thread(_claim, name, arguments, scope, session_id)
    except Exception as exc:
        # Fail open, loudly. See the module docstring.
        logger.error("tool_effects: ledger unavailable for %s (%s); dispatching"
                     " unguarded", name, exc)
        return _UNLEDGERED


def _finish(key: str, text: str, is_error: bool) -> None:
    conn = _connect()
    try:
        _ensure(conn)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        stored = text[:RESULT_MAX_CHARS]
        conn.execute(
            "UPDATE tool_effects SET status=?, result_digest=?, result_text=?,"
            " result_truncated=?, updated_at=? WHERE effect_key=? AND status='unknown'",
            ("error" if is_error else "ok", digest, stored,
             int(len(text) > RESULT_MAX_CHARS), _now(), key))
    finally:
        conn.close()


async def finish(key: str, text: str, is_error: bool) -> None:
    """Settle a reserved effect. Never raises: the call already happened."""
    if not key:
        return
    try:
        await asyncio.to_thread(_finish, key, text or "", bool(is_error))
    except Exception as exc:
        logger.error("tool_effects: could not settle %s… (%s)", key[:12], exc)


def suppressed_total() -> int:
    """How many duplicate effects the guard has refused, since the ledger began.

    Reads without creating: workers may be switched off entirely, and a health
    view must not materialise a database to report zero.
    """
    try:
        conn = _connect(create=False)
        if conn is None:
            return 0
        try:
            _ensure(conn)
            row = conn.execute(
                "SELECT COALESCE(SUM(suppress_count),0) FROM tool_effects").fetchone()
            return int(row[0] or 0)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("tool_effects: suppression counter unreadable (%s)", exc)
        return 0
