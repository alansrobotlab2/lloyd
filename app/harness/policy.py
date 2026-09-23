"""Scope-bound expiring authority grants (#534).

A third state between allow and deny that the harness did not have: a
pre-authorized, expiring, quota-bound **grant**, minted by a human before
the action is known, checked mechanically at dispatch.

Why this exists. `app/harness/safety.py` gates catastrophic Bash and returns
`{}` for every other tool, and the non-interactive turn paths — worker
sources via `_worker_run_options`, and `autonomy.run_task` — ran with
`hooks=None` entirely. So `email_send`, `calendar_delete_event`,
`contacts_delete`, `email_empty_trash` and the rest of the durable-external
surface were ungated on exactly the turns where no human is watching. A
static policy has only two exits there: keep the denylist so narrow it never
bites, or deny and burn the run. A grant is the exit that lets the gate be
tight AND still unblock real work — authority issued ahead of time, by the
human, in reviewable batches, with a death date.

What a grant is: `{scope, tool_pattern, arg_predicate, quota, issued_by,
issued_at, expires_at, consumed, revoked_at}`. A worker that wants to mail
three named addresses this week runs against a row saying exactly that, and
the row dies on its date.

Deliberate limits, each of which is a decision rather than an omission:

* **No cryptographic layer.** No SD-JWT, no VC, no signing. There is no
  untrusted counterparty in a single-owner local system; a row plus a
  stdlib expiry check is the whole portable part of the source design.
* **No evidence layer.** That is #525/#527. This module records `grant_id`
  on a dispatch ledger row so the join exists and nothing more.
* **No renewal path.** Renewing means the human minting a new row (or
  editing the date in the task file). A `renew()` here would be a self-mint
  with better letterhead.
* **Expiry is evaluated at every dispatch, including mid-run.** A grant that
  dies while its task is running denies the next call. That is the honest
  reading of "a human authorized this until Thursday": Thursday ended. It
  also means a long-running task must be issued a grant that outlives it,
  which is visible at issuance time rather than discovered at 03:00.
* **Nothing here bounds judgment**, only scope. It proves a human said so in
  advance for a bounded thing; it cannot prove the thing was sensible.

Enforcement is not prompt-dependent, and that is testable rather than a
claim: this module imports no prompt builder, no session state, no inner
voice and no model client. `check_grants` is a pure function of (store,
scope, tool, args, now). `tests/unit/test_grant_policy.py` pins it.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.harness.hooks import HookRegistry

logger = logging.getLogger("lloyd-harness-policy")

# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

GRANT_TABLE = "authority_grants"
DISPATCH_TABLE = "grant_dispatch"


def _apply_grant_ddl(conn: sqlite3.Connection) -> None:
    """Create-or-migrate the grant tables, through `workers/queue.py` — the
    module that owns the DB — rather than a copy of its DDL kept here. Two
    definitions of a table whose NOT-NULL `expires_at` is the whole safety
    property is how one of them stops being true.

    The queue's migrator and not the DDL text: `CREATE TABLE IF NOT EXISTS` is
    not a migration, and every live database has held `authority_grants` since
    #534, so a store that ran the DDL text alone would keep the pre-#628 shape
    forever while every freshly-created database had the new one.

    Function-local import on purpose: `workers/__init__` pulls in the pool, so
    a module-level import would tie `app.harness` to the worker package at
    import time and open a cycle whichever side loads first.
    """
    from workers.queue import apply_grant_ddl
    apply_grant_ddl(conn)


# Default lifetime a deny suggests when it renders the grant shape, so the
# human's answer to a denial is a one-line edit rather than a decision.
SUGGESTED_TTL_DAYS = 7


class GrantError(ValueError):
    """A grant that cannot be understood is never written."""


def _utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _iso(value: dt.datetime) -> str:
    return _utc(value).isoformat()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_expiry(value: Any) -> dt.datetime:
    """Parse a grant's expiry. No default: a grant with no expiry is the
    defect this whole design exists to prevent, so absence is an error and
    never `datetime.max`."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise GrantError("expires_at is required — a grant with no expiry is not a grant")
    if isinstance(value, dt.datetime):
        return _utc(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            parsed = dt.datetime.fromisoformat(text + "T23:59:59+00:00")
        else:
            raise GrantError(f"expires_at {value!r} is not an ISO date or datetime")
    return _utc(parsed)


# ---------------------------------------------------------------------------
# Argument predicates — a two-form mini-language, never `eval`
#
#   len(field)<=N     length of the named argument
#   field<=N          numeric argument (falls back to length when non-numeric)
#
# with the operator in <= < >= > == . Parsed by regex, evaluated by a fixed
# interpreter. Anything else is rejected at mint, so a grant can never carry
# an executable expression.
# ---------------------------------------------------------------------------

_PREDICATE_RE = re.compile(
    r"^(?:(?P<fn>len)\(\s*)?(?P<field>[A-Za-z_][A-Za-z0-9_]*)(?P<closer>\s*\))?"
    r"\s*(?P<op><=|>=|==|<|>)\s*(?P<num>\d{1,9})$"
)


def validate_predicate(text: Any) -> str:
    """Return the normalized predicate, or raise. Empty string = any args."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise GrantError(f"arg_predicate must be a string, got {type(text).__name__}")
    text = text.strip()
    if not text:
        return ""
    m = _PREDICATE_RE.match(text)
    if not m:
        raise GrantError(
            f"unparsable predicate {text!r}; accepted forms are "
            "'len(field)<=N' and 'field<=N' with N an integer"
        )
    if (m.group("fn") == "len") != bool(m.group("closer")):
        raise GrantError(f"unbalanced predicate {text!r}")
    return f"{m.group('fn') or ''}({m.group('field')}){m.group('op')}{m.group('num')}" \
        if m.group("fn") else f"{m.group('field')}{m.group('op')}{m.group('num')}"


def _arg_value(tool_input: dict, field: str) -> Any:
    if not isinstance(tool_input, dict):
        return None
    if field in tool_input:
        return tool_input[field]
    # MCP wrappers nest the real arguments under one key (`{"args": {...}}`);
    # a predicate naming a business field means that field wherever it sits.
    for v in tool_input.values():
        if isinstance(v, dict) and field in v:
            return v[field]
    return None


def predicate_matches(predicate: str, tool_input: dict) -> bool:
    if not predicate:
        return True
    m = _PREDICATE_RE.match(predicate)
    if not m:  # unreachable: mint validates, and rows only arrive from mint
        return False
    field, op, num = m.group("field"), m.group("op"), int(m.group("num"))
    value = _arg_value(tool_input or {}, field)
    if m.group("fn") == "len":
        try:
            left = len(value) if value is not None else 0
        except TypeError:
            return False
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                left = len(value)
            except (TypeError, AttributeError):
                return False
        else:
            left = value
    return {
        "<=": left <= num, "<": left < num,
        ">=": left >= num, ">": left > num,
        "==": left == num,
    }[op]


# ---------------------------------------------------------------------------
# Reversibility tiers
# ---------------------------------------------------------------------------
#
# Tiering is by how hard the action is to undo, not by category. Tier 1 is
# the default and is deliberately the empty case: an unclassified tool is
# not gated, because a gate that guesses wrong denies real work and a worker
# that cannot write its output burns a run.
#
# Bash is not in either list. It is durable-external and hard-to-reverse in
# the same breath depending on the string, and command-level tiering is a
# different mechanism (`safety.py`'s pattern set) — so it stays outside this
# gate rather than being gated as a name, which would deny every worker.

TIER2_TOOLS = frozenset({
    # Durable-external: the counterparty is outside this machine, or the
    # effect is visible to someone else, but the record still exists.
    "email_send", "email_reply", "email_forward",
    "email_update", "email_move_folder",
    "email_create_folder", "email_rename_folder",
    "email_create_filter", "email_update_filter", "email_reorder_filters",
    "email_apply_filters",
    "calendar_create", "calendar_update_event",
    "contacts_create", "contacts_update",
    "tasks_create", "tasks_update",
    # Scheduler state (#724). Tiered by NAME, gated per call — see
    # `effective_tier`. The sibling that deletes a task is tier 3 above; this one
    # rewrites whether and when every task in the fleet runs, and was tier 1.
    "autonomy_write_task",
})

TIER3_TOOLS = frozenset({
    # Hard-to-reverse: after the call the thing is gone, or the bulk of it is.
    "email_delete", "email_empty_trash", "email_empty_junk",
    "email_delete_folder",
    "calendar_delete_event",
    "contacts_delete",
    "email_delete_filter",
    "autonomy_delete_task",
})


def normalize_tool_name(name: Any) -> str:
    """Strip the legacy `mcp__<server>__` prefix; the advertised form is bare
    but old config and old transcripts carry the qualified one."""
    text = str(name or "")
    if text.startswith("mcp__"):
        rest = text[len("mcp__"):]
        if "__" in rest:
            return rest.split("__", 1)[1]
    return text


# ---------------------------------------------------------------------------
# Scheduler-state writes (#724)
# ---------------------------------------------------------------------------
#
# `autonomy_write_task` is in TIER2_TOOLS above, but a NAME cannot express what
# needs gating here, and the difference is the whole design.
#
# The tool rewrites the fields the scheduler reads to decide WHETHER and WHEN a
# task runs. Before this change it was tier 1, so every unattended turn could
# arm, park or re-point any task in the fleet. #724 counted 17 dispatches from
# unattended sessions in the retained event window; this round re-ran that count
# and got 17 again, 7 of them moving a dispatch-affecting field, none from a
# turn whose contract authorised a scheduler change: a backlog-triage turn armed
# task #84, automod implement rounds parked and re-armed #85, and nightly task
# #40 armed #68 and #85. That is #534's forbidden state arriving by a different
# door — an unattended scope issuing itself authority, over the schedule instead
# of over a grant.
#
# But ten of those seventeen carried only `activity_note`/`activity`, the
# run-record habit every skill prescribes, mostly against the writer's OWN task.
# Gating the name outright would deny the fleet's ordinary bookkeeping on the
# first nightly while still naming the wrong predicate — the writes that moved
# dispatch were the cross-task ones. So `effective_tier` demotes a call that
# moves no dispatch-affecting field back to 1, and the gate asks one question of
# the calls that do: may this turn change what runs?
#
# `model`, `preferred_hours`, `next_run`/`last_run`, `max_retries`,
# `failure_count` and `runs_per_day` are deliberately NOT in the set. The first
# two change how a run behaves once dispatched, not whether it dispatches — a
# parked task stays parked with a new model, and a narrowed hour window cannot
# start a run its status has not authorised. The rest are scheduler OUTPUT; the
# scheduler writes them itself, and gating them would deny every completion
# stamp. There is no interactive fallback on the worker path, so an over-broad
# predicate here ends every nightly parked behind a human.
SCHEDULE_STATE_TOOL = "autonomy_write_task"

# The six fields the acceptance names, plus `preferred_hours`: the nightly
# chain's ORDER lives nowhere but in each task's own `preferred_hours` (the
# `pipeline:` key is read by display code and by no gate), so editing one
# window silently re-sequences #38 -> #42 -> #39 -> #40. Leaving it out would
# keep the hazard this item was filed about standing.
SCHEDULE_STATE_FIELDS = frozenset({
    "status", "scheduled_at", "depends_on", "auto_advance", "frequency",
    "skill_name", "preferred_hours",
})

#: The one status the queue will actually dispatch (`DISPATCHING_STATUS` mirrors
#: the comment "Only up_next dispatches" in `autonomy._is_task_due`).
#: `in_progress` and `failed` are in `RUNNABLE_STATUSES` for dependency lookups,
#: not because either can start a run.
DISPATCHING_STATUS = "up_next"


def schedule_fields_changed(tool_input: Any) -> list[str]:
    """Dispatch-affecting fields this call proposes to move, sorted.

    Two shapes, because a create and an update do not carry the same risk:

    * **Update** (`id` present): every non-empty dispatch-affecting field counts.
      A key alone is not enough — the measured benign case is an update that
      passes a *falsy* value for a field it does not mean to change
      (`status: ""`), and `_handle_write` (`agent_mcp/autonomy.py`) drops
      empty/None updates, so such a call changes nothing. What it cannot see is
      whether the value differs from the field's current value: a write that
      resends `up_next` to an already-armed task is denied. That is deliberate —
      the gate reads the call, not the disk, so a decision never depends on the
      vault being readable, and no measured unattended write does that
      (of 17, the 6 self-targeted ones carried only `activity_note`).
    * **Create** (`id` absent or 0): only `status: up_next` dispatches anything.
      A new task with `skill_name`/`frequency` and the default `draft` is inert
      until something arms it, so gating creates on those fields would deny the
      documented dispatch route (`skills/pipeline-dispatch`) to buy no safety.
    """
    if not isinstance(tool_input, dict):
        return []
    new_task = tool_input.get("id") in (None, "", 0)
    if new_task:
        if str(tool_input.get("status") or "").strip() == DISPATCHING_STATUS:
            return ["status"]
        return []
    return sorted(
        field for field in SCHEDULE_STATE_FIELDS
        if tool_input.get(field) not in (None, "", [])
    )


def changes_schedule_state(name: Any, tool_input: Any) -> bool:
    """True when this call would move a field the scheduler reads to dispatch."""
    if normalize_tool_name(name) != SCHEDULE_STATE_TOOL:
        return False
    return bool(schedule_fields_changed(tool_input))


def tool_tier(name: Any) -> int:
    tool = normalize_tool_name(name)
    if tool in TIER3_TOOLS:
        return 3
    if tool in TIER2_TOOLS:
        return 2
    return 1


def effective_tier(name: Any, tool_input: Any = None) -> int:
    """The tier that governs THIS call, not the tier of the tool's name.

    Every tool except `autonomy_write_task` answers with `tool_tier`. That one
    tool has two shapes with opposite risk — an update that appends a run record
    to its own task, and an update that re-arms another task — and one tier
    cannot cover both, so the benign shape is demoted. It reuses the existing
    tier machinery (the `check_grants` check, the denial ledger, the `grants:`
    materialisation) rather than opening a parallel authority system, which is
    why it is a tier function and not a second gate.
    """
    tier = tool_tier(name)
    if tier == 2 and normalize_tool_name(name) == SCHEDULE_STATE_TOOL:
        if not changes_schedule_state(name, tool_input):
            return 1
    return tier


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

GRANT_MINT_TOOL = "grant_create"
INTERACTIVE_SCOPE = "interactive"
NON_INTERACTIVE_PREFIXES = ("worker", "autonomy-task", "scheduled-task", "background")

#: Set by the worker pool around a job so the hook knows whose authority it is
#: checking. Defaults to the most restrictive reading, so a path that forgets
#: to set it is gated rather than waved through.
current_scope: contextvars.ContextVar[str] = contextvars.ContextVar(
    "lloyd_grant_scope", default="worker")

#: Set by the worker pool around a job for #544 and forwarded to the aggregator
#: in `_meta`. It names the thing whose effects must not happen twice — the queue
#: item — so a retried attempt is recognised as the same run rather than a new
#: one. It sits beside `current_scope` because the pool binds both, per job, for
#: the same reason (only the pool knows what it claimed), but they bound
#: different things and must never be conflated: `current_scope` bounds
#: AUTHORITY and is deliberately coarse (`autonomy-task:39`), which is right for
#: a grant and wrong here — keying an effect ledger on it would suppress a
#: legitimate second effect forever, across every future run of that task. An
#: empty scope means "no ledger", which is what an interactive turn gets.
current_effect_scope: contextvars.ContextVar[str] = contextvars.ContextVar(
    "lloyd_effect_scope", default="")


def is_interactive_scope(scope: Any) -> bool:
    text = str(scope or "").strip()
    if not text:
        return False
    if text == INTERACTIVE_SCOPE or text.startswith(INTERACTIVE_SCOPE + ":"):
        return True
    return text.split(":")[0] not in NON_INTERACTIVE_PREFIXES


def grant_scope_for_source(source: str, payload: dict | None = None) -> str:
    """The authority scope a queue row of this `source` runs under (#534).

    An autonomy task gets its own scope rather than sharing `worker:scheduled-task`
    with every other task, because that is the difference between a human
    granting `email_send` to the nightly mail job and granting it to whatever
    runs on that source next. Anything else is its source.

    This is the one mapping. `workers.pool.grant_scope_for` delegates here, and
    so does `agent_mcp.egress.scope_context` on the aggregator side — which
    cannot call the pool's version because it has no `QueueItem`, only the
    `item:<source>:<id>` effect scope that crosses the `_meta` seam. Two copies
    of "whose authority is this turn borrowing" is how a grant minted against
    one side stops being readable by the gate on the other.
    """
    payload = payload if isinstance(payload, dict) else {}
    if str(source or "") == "scheduled-task":
        task_id = payload.get("task_id")
        if task_id is not None:
            return f"autonomy-task:{task_id}"
    return f"worker:{source}"


def _is_worker_identity(text: str) -> bool:
    return str(text or "").strip().split(":")[0] in NON_INTERACTIVE_PREFIXES


# ---------------------------------------------------------------------------
# GrantStore
# ---------------------------------------------------------------------------


class GrantStore:
    """The grant rows. Lives in the worker job-queue SQLite DB — inspectable
    with one `sqlite3` command, no new service.

    Connections are opened per call, like the queue itself: the backend, the
    MCP aggregator and the pool are separate processes over one WAL file.
    """

    def __init__(self, db_path: str | Path):
        # Deliberately does not mkdir or connect: a store that cannot be
        # opened must fail at check time (closed), not at construction time,
        # because construction happens once at import on the dispatch path.
        self.db_path = Path(str(db_path)).expanduser()
        self._schema_applied = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        # Every path, reads included, once per store. Only the writers used to
        # apply it, so `live()`/`candidates()` against a database the pool had
        # not initialised raised "no such table" — which the egress guard reads
        # as a closed door, denying every unattended fetch on that box — and
        # against a pre-#628 table they would read rows with no `destination`.
        if not self._schema_applied:
            _apply_grant_ddl(conn)
            self._schema_applied = True
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            _apply_grant_ddl(conn)

    # ── mint ───────────────────────────────────────────────────────────────

    def mint(self, *, scope: str, tool_pattern: str,
             arg_predicate: str = "", quota: int | None = None,
             issued_by: str, expires_at: Any, note: str = "",
             minted_by: str = "human", now: dt.datetime | None = None,
             destination: str | None = None) -> dict:
        """Write one grant. The only write path that creates authority.

        Validates rather than defaults: expiry is mandatory, an issuer is
        mandatory, a predicate outside the mini-language is refused, and a
        `minted_by` that identifies a worker scope is refused outright — a
        turn subject to this gate must not be able to write its way out of it.

        `destination` (#628) is validated at mint and never defaulted: an empty
        string would authorize every host through `destination_covers`, so a
        caller that means "no destination" passes None and a caller that means
        "this host" passes the host. A malformed one raises like any other bad
        field rather than being stored and read loosely later.
        """
        scope = str(scope or "").strip()
        tool = normalize_tool_name(tool_pattern)
        issued_by = str(issued_by or "").strip()
        if not scope:
            raise GrantError("scope is required")
        if not tool:
            raise GrantError("tool is required")
        if tool == GRANT_MINT_TOOL:
            raise GrantError("a grant may not authorize minting a grant")
        if not issued_by:
            raise GrantError("issued_by is required — every grant names a human")
        if _is_worker_identity(issued_by) or _is_worker_identity(minted_by):
            raise GrantError(
                f"grant may not be minted by a non-interactive identity "
                f"(minted_by={minted_by!r}, issued_by={issued_by!r})")
        if quota is not None:
            quota = int(quota)
            if quota < 1:
                raise GrantError("quota must be >= 1 when set")
        predicate = validate_predicate(arg_predicate)
        expiry = _parse_expiry(expires_at)
        dest = (normalize_destination(destination)
                if destination is not None else None)
        issued = _iso(now or _now())
        with self._connect() as conn:
            _apply_grant_ddl(conn)
            cur = conn.execute(
                f"INSERT INTO {GRANT_TABLE} (scope, tool_pattern, arg_predicate,"
                " quota, issued_by, minted_by, note, issued_at, expires_at,"
                " destination)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (scope, tool, predicate, quota, issued_by,
                 str(minted_by or "human"), note or None, issued, _iso(expiry),
                 dest),
            )
            row = self._row(conn, cur.lastrowid)
        logger.info(
            "[grants] minted id=%s scope=%s tool=%s destination=%s predicate=%r "
            "quota=%s expires=%s issued_by=%s", row["id"], scope, tool, dest,
            predicate, quota, row["expires_at"], issued_by)
        return row

    # ── read ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row(conn: sqlite3.Connection, grant_id: int) -> dict:
        r = conn.execute(
            f"SELECT * FROM {GRANT_TABLE} WHERE id=?", (grant_id,)).fetchone()
        return dict(r) if r else {}

    @staticmethod
    def _as_dict(r: sqlite3.Row) -> dict:
        return dict(r)

    def get(self, grant_id: int) -> dict | None:
        with self._connect() as conn:
            r = conn.execute(
                f"SELECT * FROM {GRANT_TABLE} WHERE id=?", (grant_id,)).fetchone()
        return dict(r) if r else None

    def live(self, *, scope: str | None = None,
             now: dt.datetime | None = None) -> list[dict]:
        at = _iso(now or _now())
        sql = (f"SELECT * FROM {GRANT_TABLE} WHERE revoked_at IS NULL"
               " AND expires_at > ?")
        args: list[Any] = [at]
        if scope is not None:
            sql += " AND scope=?"
            args.append(scope)
        sql += " ORDER BY expires_at ASC"
        with self._connect() as conn:
            return [self._as_dict(r) for r in conn.execute(sql, args)]

    def candidates(self, *, scope: str, tool: str,
                   now: dt.datetime | None = None) -> list[dict]:
        """Rows for this (scope, tool) regardless of life — used to turn a
        bare deny into a reason that says *which* condition failed."""
        with self._connect() as conn:
            return [self._as_dict(r) for r in conn.execute(
                f"SELECT * FROM {GRANT_TABLE} WHERE scope=? AND tool_pattern=?"
                " ORDER BY expires_at DESC", (scope, tool))]

    # ── mutate ─────────────────────────────────────────────────────────────

    def consume(self, grant_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {GRANT_TABLE} SET consumed = consumed + 1 WHERE id=?",
                (grant_id,))

    def revoke(self, grant_id: int, *, now: dt.datetime | None = None) -> bool:
        """Revoke takes effect at the next dispatch, including a dispatch
        inside the run that already consumed this grant."""
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE {GRANT_TABLE} SET revoked_at=? "
                "WHERE id=? AND revoked_at IS NULL",
                (_iso(now or _now()), grant_id))
            return cur.rowcount > 0

    # ── dispatch ledger ────────────────────────────────────────────────────

    def record_dispatch(self, *, scope: str, tool: str, decision: str,
                        grant_id: int | None = None, reason: str = "",
                        now: dt.datetime | None = None) -> None:
        with self._connect() as conn:
            _apply_grant_ddl(conn)
            conn.execute(
                f"INSERT INTO {DISPATCH_TABLE} (at, scope, tool, grant_id,"
                " decision, reason) VALUES (?,?,?,?,?,?)",
                (_iso(now or _now()), scope, tool, grant_id, decision,
                 reason or None))

    def dispatch_rows(self, *, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            return [self._as_dict(r) for r in conn.execute(
                f"SELECT * FROM {DISPATCH_TABLE} ORDER BY id DESC LIMIT ?",
                (limit,))]

    def count_worker_minted(self) -> int:
        """Acceptance counter: grants minted by a worker scope must be zero,
        forever. Queryable rather than asserted."""
        n = 0
        with self._connect() as conn:
            for r in conn.execute(f"SELECT minted_by FROM {GRANT_TABLE}"):
                if _is_worker_identity(r["minted_by"]) or _is_worker_identity(r["issued_by"]):
                    n += 1
        return n


_STORE_CACHE: dict[str, GrantStore] = {}


def default_store() -> GrantStore:
    """The store the hook uses when none was handed to it.

    `LLOYD_GRANT_DB` overrides so a canary boot, or a test, can point the
    gate at a file without touching config.
    """
    raw = os.environ.get("LLOYD_GRANT_DB")
    if not raw:
        from workers.queue import configured_db_path
        raw = str(configured_db_path())
    key = str(raw)
    store = _STORE_CACHE.get(key)
    if store is None:
        store = GrantStore(key)
        _STORE_CACHE[key] = store
    return store


# ---------------------------------------------------------------------------
# The check — pure, stdlib, synchronous
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    allowed: bool
    grant_id: int | None = None
    reason: str = ""


def grant_shape(*, scope: str, tool: str, tool_input: dict,
                now: dt.datetime | None = None,
                destination: str | None = None) -> str:
    """Render the exact grant that would authorize this call.

    This is the issuance UI. The denial that interrupts a live run is the
    thing this design replaces; the denial that names the row to mint turns
    it into a batched renewal the human can action in one line.

    `destination` (#628) is rendered as the grant's own field rather than
    folded into the predicate, because `validate_predicate` accepts only
    `len(k)<=N` and `k==<integer>` — a hostname cannot be expressed there,
    which is exactly why this axis needed a column. The rendered row is what a
    human pastes, so it must be the row the gate actually reads.
    """
    predicate = _suggested_predicate(tool_input)
    expiry = _iso((_utc(now or _now())) + dt.timedelta(days=SUGGESTED_TTL_DAYS))
    bits = [f"scope='{scope}'", f"tool='{tool}'"]
    if destination:
        bits.append(f"destination='{destination}'")
    if predicate:
        bits.append(f"predicate='{predicate}'")
    bits += [f"expires_at='{expiry}'", "issued_by='alan'"]
    return f"{GRANT_MINT_TOOL}(" + ", ".join(bits) + ")"


def _suggested_predicate(tool_input: dict) -> str:
    """Bound the suggestion by the collection the call is actually carrying,
    so the smallest grant that clears this call is the one offered."""
    if not isinstance(tool_input, dict):
        return ""
    for key, value in tool_input.items():
        if isinstance(value, (list, tuple)) and value:
            return f"len({key})<={max(1, len(value))}"
    return ""


#: A registrable hostname, or an IP literal (checked separately in
#: `normalize_destination`). Labels may not be empty and may not start or end with
#: a hyphen, which is what stops `a..com` and `-evil.com` reading as hosts.
_HOSTNAME_RE = re.compile(r"^(?=.{1,253}$)[a-z0-9]([a-z0-9-]*[a-z0-9])?"
                          r"(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$")


def normalize_destination(text: Any) -> str:
    """Normalize one destination to a bare lowercase host, or raise `GrantError`.

    Deliberately narrow, and the narrowness is the security property: a host, not
    a URL, not a port, not a glob. The empty string and a wildcard are refused
    because `destination_covers` matches on a dot-anchored suffix, so an entry
    with no dot would authorize the world and `*` would be silently permissive.
    Validating at mint time is the only place a bad destination can be caught
    before it becomes authority.
    """
    from ipaddress import ip_address

    if text is None or (isinstance(text, str) and not text.strip()):
        raise GrantError("destination is required — an empty destination would "
                         "authorize every host")
    if not isinstance(text, str):
        raise GrantError(f"destination must be a string, got {type(text).__name__}")
    raw = text.strip().lower()
    try:
        # An IP literal first: an IPv6 address is all colons, and the URL/port
        # filter below would otherwise refuse `::1` as "not a bare host".
        return str(ip_address(raw))
    except ValueError:
        pass
    if any(ch in raw for ch in ("*", "/", ":", " ")) or raw.startswith("."):
        raise GrantError(f"destination {text!r} must be a bare host "
                         "(example.com), not a URL, a port, or a wildcard")
    if raw.rsplit(".", 1)[-1].isdigit():
        # `256.1.1.1` is not an address, and no real hostname ends in an
        # all-numeric label, so it must not fall through to the hostname grammar
        # (which it satisfies) and be stored as a host nothing can resolve.
        raise GrantError(f"destination {text!r} looks like an IPv4 address but "
                         "is not a valid one")
    if "." not in raw:
        raise GrantError(f"destination {text!r} has no dot; a single label would "
                         "authorize every domain under that suffix")
    if not _HOSTNAME_RE.match(raw):
        raise GrantError(f"destination {text!r} is not a valid hostname")
    return raw


def destination_covers(entry: Any, host: Any) -> bool:
    """Does the grant's destination `entry` authorize the host `host`?

    Exact, or a proper subdomain: `duckduckgo.com` covers `html.duckduckgo.com`
    and must NOT cover `evil-duckduckgo.com`, which is why the suffix test is
    anchored on a dot rather than `str.endswith`. An empty entry covers nothing:
    a grant minted before #628, or minted for the tool as a whole, is not a
    network licence. One definition lives here because `check_grants` must stay
    pure and stdlib, and `agent_mcp.egress` imports it from here.
    """
    entry = str(entry or "").strip().lower()
    host = str(host or "").strip().lower()
    if not entry or not host:
        return False
    return host == entry or host.endswith("." + entry)


def _explain_missing(scope: str, tool: str, rows: Iterable[dict],
                     now: dt.datetime, *,
                     destination: str | None = None) -> str:
    at = _utc(now)
    for row in rows:
        if destination is not None and not destination_covers(
                row.get("destination"), destination):
            continue
        if row.get("revoked_at"):
            return (f"grant #{row['id']} for '{tool}' from scope '{scope}' was "
                    f"revoked at {row['revoked_at']}")
        if str(row.get("expires_at") or "") <= _iso(at):
            return (f"grant #{row['id']} for '{tool}' from scope '{scope}' "
                    f"expired at {row['expires_at']}")
        quota, consumed = row.get("quota"), row.get("consumed") or 0
        if quota is not None and consumed >= quota:
            return (f"grant #{row['id']} for '{tool}' from scope '{scope}' is "
                    f"over quota ({consumed}/{quota})")
        if not predicate_matches(row.get("arg_predicate") or "", {}):
            continue
    return ""


def _decide_destination(store: GrantStore, *, scope: str, tool: str,
                        live: list[dict], destination: Any,
                        at: dt.datetime, now: dt.datetime | None,
                        record: bool) -> Decision:
    """Answer a call that names a destination, on the destination axis alone.

    Deliberately not folded into the `match` loop below, because the two
    questions differ in kind. Tier-2 matching asks "does this scope hold a
    licence for this tool, bounded by an argument predicate"; the destination
    answer is "may this scope contact *this host*", and no predicate in
    `validate_predicate`'s mini-language can say so (it accepts only
    `len(k)<=N` and `k==<integer>` — `url==example.com` raises, which is the
    fact that forced a column). Two consequences a reader should expect:

    * a grant with **no** destination never qualifies here — it predates the
      axis, and reading NULL as "any host" would make every pre-#628 grant a
      network licence for the tool it names;
    * a grant whose destination is a different host never qualifies — a grant
      for `api.example.com` must not pay for a fetch to `attacker.tld`. That
      non-transferability is the clause #628 exists to pin.

    Quota works as it does for tier 2: a destination grant with `quota=20`
    bounds how many calls it pays for, not just which host.
    """
    host = normalize_destination(destination)
    for row in live:
        if row["tool_pattern"] != tool:
            continue
        if not destination_covers(row.get("destination"), host):
            continue
        quota, consumed = row.get("quota"), row.get("consumed") or 0
        if quota is not None and consumed >= quota:
            continue
        store.consume(row["id"])
        if record:
            store.record_dispatch(scope=scope, tool=tool, decision="allow",
                                  grant_id=row["id"], now=at)
        return Decision(True, row["id"], "")

    why = _explain_missing(scope, tool,
                           store.candidates(scope=scope, tool=tool, now=at), at,
                           destination=host)
    reason = (f"grant: {'no grant' if not why else why} — '{tool}' to "
              f"'{host}' is a network egress to a destination scope '{scope}' "
              f"has not been given.")
    reason += " " + grant_shape(scope=scope, tool=tool, tool_input={}, now=at,
                                destination=host)
    if record:
        store.record_dispatch(scope=scope, tool=tool, decision="deny",
                              reason=reason, now=at)
    logger.warning("[grants] denied egress tool=%s scope=%s host=%s", tool,
                   scope, host)
    return Decision(False, None, reason)


def check_grants(store: GrantStore, *, scope: str, tool_name: Any,
                 tool_input: dict | None = None,
                 now: dt.datetime | None = None,
                 record: bool = True,
                 destination: str | None = None) -> Decision:
    """May this call proceed? The only question the dispatch hook asks.

    Tier 1 returns before the store is opened — that is what makes a dead
    store a denial for the tools that matter and a non-event for the rest.
    An allowed call consumes one unit of quota, so a grant bounds volume as
    well as reach.

    `destination` (#628) is the host a network call is about to contact. It is
    the one input that defeats the tier-1 early return, and deliberately so:
    the egress tools are tier 1 by the ladder's own definition (reading a page
    is not hard to reverse), so if the ladder short-circuited them the
    destination axis could never refuse an exfiltration. Pass a destination and
    only a destination-scoped grant covering that host can pay for the call;
    omit it and a destination-scoped grant cannot pay for anything (see
    `_decide_destination` and the `match` loop).
    """
    at = _utc(now or _now())
    tool = normalize_tool_name(tool_name)
    args = tool_input if isinstance(tool_input, dict) else {}
    scope = str(scope or "").strip() or "worker"

    if tool == GRANT_MINT_TOOL:
        if is_interactive_scope(scope):
            return Decision(True, None, "interactive scope may mint")
        return Decision(False, None,
                        f"grant: a non-interactive scope may not mint authority — "
                        f"'{GRANT_MINT_TOOL}' is not callable from scope '{scope}'. "
                        "Authority is issued ahead of time by a human: "
                        + grant_shape(scope=scope, tool="<the tool you need>",
                                      tool_input={}, now=at))

    tier = effective_tier(tool, args)
    if tier == 1 and destination is None:
        return Decision(True, None, "")

    if is_interactive_scope(scope):
        # A human is present on this turn; the pre-send review dialogs are
        # the control there. Gating chat would be per-write review friction,
        # which is the thing this design exists to avoid.
        return Decision(True, None, "")

    try:
        live = store.live(scope=scope, now=at)
    except Exception as exc:
        raise GrantError(f"grant store unreadable ({exc.__class__.__name__}: {exc})")

    if destination is not None:
        # #628: the destination axis, consulted even for a tier-1 tool. The
        # egress tools are tier 1 — reading a page the operator named is not
        # hard to reverse, which is exactly why the ladder cannot be what stops
        # an exfiltration POST — so a call that names a destination is answered
        # by the destination rows and nothing else.
        return _decide_destination(store, scope=scope, tool=tool, live=live,
                                   destination=destination, at=at, now=now,
                                   record=record)

    match = None
    for row in live:
        if row["tool_pattern"] != tool:
            continue
        quota, consumed = row.get("quota"), row.get("consumed") or 0
        if quota is not None and consumed >= quota:
            continue
        if not predicate_matches(row.get("arg_predicate") or "", args):
            continue
        # A grant minted *for a destination* is not a licence for every other
        # tier-2 call that tool can make: `email_send` to api.example.com does
        # not authorize a vault write under the same tool pattern.
        if str(row.get("destination") or "").strip():
            continue
        match = row
        break

    if match is not None:
        store.consume(match["id"])
        if record:
            store.record_dispatch(scope=scope, tool=tool, decision="allow",
                                  grant_id=match["id"], now=at)
        return Decision(True, match["id"], "")

    why = _explain_missing(scope, tool,
                           store.candidates(scope=scope, tool=tool, now=at), at,
                           destination=destination)
    reason = (f"grant: {'no grant' if not why else why} — '{tool}' is a "
              f"tier-{tier} (hard-to-reverse) action and scope '{scope}' is "
              f"unattended.")
    if tool == SCHEDULE_STATE_TOOL:
        # The scope and tool alone are not actionable here: the same tool call
        # either appends a run record or re-arms a nightly, and whoever has to
        # decide needs to see which one was proposed.
        changed = ", ".join(schedule_fields_changed(args))
        target = args.get("id") or "new (this call creates the task)"
        reason += (f" This call would change dispatch-affecting field(s) "
                   f"[{changed}] on target #{target}, which decides whether and "
                   f"when that task runs. To authorise it, a human either adds a "
                   f"`grants:` block to the calling task's frontmatter "
                   f"(scope `autonomy-task:*` only) or runs the line below.")
    reason += (" " + grant_shape(scope=scope, tool=tool, tool_input=args, now=at,
                                 destination=destination))
    if record:
        store.record_dispatch(scope=scope, tool=tool, decision="deny",
                              reason=reason, now=at)
    logger.warning("[grants] denied tool=%s scope=%s — %s", tool, scope, reason)
    return Decision(False, None, reason)


# ---------------------------------------------------------------------------
# Hook wiring
# ---------------------------------------------------------------------------


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def install_policy_hook(hooks: HookRegistry, *, store: GrantStore | None = None,
                        scope: str | None = None,
                        now: dt.datetime | None = None) -> None:
    """Gate tier-2/tier-3 tools on a live grant, for one turn.

    `scope` names whose authority this turn is borrowing (`worker:<source>`,
    `autonomy-task:<id>`); omit it and the hook reads `current_scope` when it
    fires, which the worker pool sets per job.

    `now` is the instant the expiry check runs against, and it is forwarded to
    `check_grants`, which has always taken one. Production omits it, and the
    default is `None` — *not* `_now()` — so the clock is read fresh inside
    `check_grants` on every call: this callback is installed once and fires for
    the life of the turn, so an instant captured at install time would let a
    long unattended turn keep spending a grant that expired mid-run. The seam
    exists for the test, and for the trap a missing one sets: a hook-path test
    that cannot name a time has to mint its fixtures against the wall clock to
    stay green, which is how `test_hook_allows_granted_call` came to be red
    forever from 2026-09-11T12:00Z and cost two automod rounds their review
    attempts (#848/#853, closed at the test layer only; #973 is this parameter).

    The callback denies rather than raises. `HookRegistry.fire_pre_tool_use`
    treats a raising callback as a pass — correct for observers, exactly wrong
    here, since a store that cannot be opened would otherwise open the gate.
    """

    async def _policy_pretool_cb(input_data: dict[str, Any],
                                _tool_use_id: str | None, _ctx: Any) -> dict:
        tool_name = input_data.get("tool_name", "")
        if tool_tier(tool_name) == 1 and tool_name != GRANT_MINT_TOOL:
            return {}
        active = store
        try:
            if active is None:
                active = default_store()
            decision = check_grants(
                active,
                scope=scope or current_scope.get(),
                tool_name=tool_name,
                tool_input=input_data.get("tool_input") or {},
                now=now,
            )
        except Exception as exc:
            reason = (f"grant: authority gate could not be evaluated "
                      f"({exc.__class__.__name__}: {exc}); '{tool_name}' from "
                      f"scope '{scope or current_scope.get()}' is denied while "
                      "the grant store is unreadable")
            logger.warning("[grants] %s", reason)
            return _deny(reason)
        if decision.allowed:
            return {}
        return _deny(decision.reason)

    hooks.add_pre_tool_use(None, _policy_pretool_cb)


# ---------------------------------------------------------------------------
# Autonomy task frontmatter — the second mint path
# ---------------------------------------------------------------------------
#
# `grants:` on a task file is a declarative mint: the human wrote it, so the
# human is the issuer. It is validated at LOAD, and a block that cannot be
# understood fails the task closed rather than being ignored — an unreadable
# authorization is not an absent one, and running anyway is fail-open with a
# log line attached.


def validate_task_grants(grants: Any) -> tuple[list[dict], list[str]]:
    """`(normalized specs, errors)`. Never partially accepts: any error means
    the caller must not run the task."""
    if grants is None:
        return [], []
    if not isinstance(grants, list):
        return [], [f"grants: must be a list of grant maps, got "
                    f"{type(grants).__name__}"]
    errors: list[str] = []
    specs: list[dict] = []
    for i, item in enumerate(grants):
        if not isinstance(item, dict):
            errors.append(f"grants[{i}]: must be a map, got {type(item).__name__}")
            continue
        unknown = set(item) - {"tool", "predicate", "quota", "expires_at",
                               "issued_by", "note"}
        if unknown:
            errors.append(f"grants[{i}]: unknown key(s) {sorted(unknown)}")
        tool = normalize_tool_name(item.get("tool"))
        if not tool:
            errors.append(f"grants[{i}]: 'tool' is required")
        elif tool == GRANT_MINT_TOOL:
            errors.append(f"grants[{i}]: a task may not grant itself the mint tool")
        try:
            predicate = validate_predicate(item.get("predicate") or "")
        except GrantError as exc:
            errors.append(f"grants[{i}]: {exc}")
            predicate = ""
        try:
            expiry = _parse_expiry(item.get("expires_at"))
        except GrantError as exc:
            errors.append(f"grants[{i}]: {exc}")
            expiry = None
        quota = item.get("quota")
        if quota is not None:
            try:
                quota = int(quota)
                if quota < 1:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"grants[{i}]: quota must be an integer >= 1")
                quota = None
        issued_by = str(item.get("issued_by") or "").strip()
        if not issued_by:
            errors.append(f"grants[{i}]: 'issued_by' is required — a grant "
                          "names the human who issued it")
        if not errors:
            specs.append({"tool": tool, "predicate": predicate, "quota": quota,
                          "expires_at": expiry, "issued_by": issued_by,
                          "note": str(item.get("note") or "")})
    return specs, errors


def sync_task_grants(store: GrantStore, *, task_id: Any, scope: str,
                     grants: Any, now: dt.datetime | None = None) -> int:
    """Materialize a task's declared `grants:` block into grant rows.

    Idempotent and never renewing: a row already covering this
    (scope, tool, predicate) is left exactly as it is, so a task that runs
    every night cannot extend its own expiry by running. Renewal is the human
    editing the file, which is the visible act the design asks for.
    """
    specs, errors = validate_task_grants(grants)
    if errors:
        raise GrantError("; ".join(errors))
    at = _utc(now or _now())
    minted = 0
    for spec in specs:
        expiry = spec["expires_at"]
        if expiry is None or expiry <= at:
            logger.warning(
                "[grants] task #%s declares a grant for '%s' that is not live "
                "at %s; not materialized — the task runs without it",
                task_id, spec["tool"], _iso(at))
            continue
        existing = [r for r in store.live(scope=scope, now=at)
                    if r["tool_pattern"] == spec["tool"]
                    and (r["arg_predicate"] or "") == spec["predicate"]]
        if existing:
            continue
        store.mint(scope=scope, tool_pattern=spec["tool"],
                   arg_predicate=spec["predicate"], quota=spec["quota"],
                   issued_by=spec["issued_by"], expires_at=expiry,
                   note=spec["note"] or f"autonomy task #{task_id}",
                   minted_by=f"frontmatter:{task_id}", now=at)
        minted += 1
    return minted
