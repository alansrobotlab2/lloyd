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

def _grant_ddl() -> str:
    """The grant tables' DDL, which lives in `workers/queue.py` — the module
    that owns the DB — and is imported here rather than duplicated. Two
    definitions of a table whose NOT-NULL `expires_at` is the whole safety
    property is how one of them stops being true.

    Function-local on purpose: `workers/__init__` pulls in the pool, so a
    module-level import would tie `app.harness` to the worker package at
    import time and open a cycle whichever side loads first.
    """
    from workers.queue import GRANT_DDL
    return GRANT_DDL


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


def tool_tier(name: Any) -> int:
    tool = normalize_tool_name(name)
    if tool in TIER3_TOOLS:
        return 3
    if tool in TIER2_TOOLS:
        return 2
    return 1


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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_grant_ddl())

    # ── mint ───────────────────────────────────────────────────────────────

    def mint(self, *, scope: str, tool_pattern: str,
             arg_predicate: str = "", quota: int | None = None,
             issued_by: str, expires_at: Any, note: str = "",
             minted_by: str = "human", now: dt.datetime | None = None) -> dict:
        """Write one grant. The only write path that creates authority.

        Validates rather than defaults: expiry is mandatory, an issuer is
        mandatory, a predicate outside the mini-language is refused, and a
        `minted_by` that identifies a worker scope is refused outright — a
        turn subject to this gate must not be able to write its way out of it.
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
        issued = _iso(now or _now())
        with self._connect() as conn:
            conn.executescript(_grant_ddl())
            cur = conn.execute(
                f"INSERT INTO {GRANT_TABLE} (scope, tool_pattern, arg_predicate,"
                " quota, issued_by, minted_by, note, issued_at, expires_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (scope, tool, predicate, quota, issued_by,
                 str(minted_by or "human"), note or None, issued, _iso(expiry)),
            )
            row = self._row(conn, cur.lastrowid)
        logger.info(
            "[grants] minted id=%s scope=%s tool=%s predicate=%r quota=%s "
            "expires=%s issued_by=%s", row["id"], scope, tool, predicate,
            quota, row["expires_at"], issued_by)
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
            conn.executescript(_grant_ddl())
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
                now: dt.datetime | None = None) -> str:
    """Render the exact grant that would authorize this call.

    This is the issuance UI. The denial that interrupts a live run is the
    thing this design replaces; the denial that names the row to mint turns
    it into a batched renewal the human can action in one line.
    """
    predicate = _suggested_predicate(tool_input)
    expiry = _iso((_utc(now or _now())) + dt.timedelta(days=SUGGESTED_TTL_DAYS))
    bits = [f"scope='{scope}'", f"tool='{tool}'", f"expires_at='{expiry}'",
            "issued_by='alan'"]
    if predicate:
        bits.insert(2, f"predicate='{predicate}'")
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


def _explain_missing(scope: str, tool: str, rows: Iterable[dict],
                     now: dt.datetime) -> str:
    at = _utc(now)
    for row in rows:
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


def check_grants(store: GrantStore, *, scope: str, tool_name: Any,
                 tool_input: dict | None = None,
                 now: dt.datetime | None = None,
                 record: bool = True) -> Decision:
    """May this call proceed? The only question the dispatch hook asks.

    Tier 1 returns before the store is opened — that is what makes a dead
    store a denial for the tools that matter and a non-event for the rest.
    An allowed call consumes one unit of quota, so a grant bounds volume as
    well as reach.
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

    tier = tool_tier(tool)
    if tier == 1:
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

    match = None
    for row in live:
        if row["tool_pattern"] != tool:
            continue
        quota, consumed = row.get("quota"), row.get("consumed") or 0
        if quota is not None and consumed >= quota:
            continue
        if not predicate_matches(row.get("arg_predicate") or "", args):
            continue
        match = row
        break

    if match is not None:
        store.consume(match["id"])
        if record:
            store.record_dispatch(scope=scope, tool=tool, decision="allow",
                                  grant_id=match["id"], now=at)
        return Decision(True, match["id"], "")

    why = _explain_missing(scope, tool, store.candidates(scope=scope, tool=tool, now=at), at)
    reason = (f"grant: {'no grant' if not why else why} — '{tool}' is a "
              f"tier-{tier} (hard-to-reverse) action and scope '{scope}' is "
              f"unattended. "
              + grant_shape(scope=scope, tool=tool, tool_input=args, now=at))
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
                        scope: str | None = None) -> None:
    """Gate tier-2/tier-3 tools on a live grant, for one turn.

    `scope` names whose authority this turn is borrowing (`worker:<source>`,
    `autonomy-task:<id>`); omit it and the hook reads `current_scope` when it
    fires, which the worker pool sets per job.

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
