"""`lloyd_rpc` inside the aggregator: the Bash env, the parent registry, admission.

The policy (what may be called, the deny list, the trailer's shape) is
`app/harness/rpc_policy.py`; this module is the state that makes it
server-authoritative (review 2026-09-24, P9).

The turn decides: the loop stamps `lloyd/rpc_deny` on a Bash call only while
`harness.rpc.enabled` (read in the process that owns the turn). When a stamped
Bash call arrives for a non-sandboxed session, `builtin_bash._bash` asks
`bash_env` for the child's environment; an unstamped one inherits the
aggregator's environment exactly as before. That call
**registers a parent** — the Bash call's id, its session, turn, effect scope,
surface, the deny list the loop stamped in `_meta`, and a deadline — and a
nested call is admitted (`admit`) only against a live parent and only against
the parent's recorded deny list. The copy in `LLOYD_RPC_DENY` is for the
client's own early refusal; editing it in a script changes nothing, because
the server never reads it back. The nested call is then dispatched under the
PARENT's session id, so every gate in `main.call_tool` that keys on the session
— the bench/eval sandbox, the desktop rule, the sessionless-write refusal, the
effect ledger, the read-before-edit record — sees the same session the Bash
call did.

What this is not: containment against code that already runs as this user.
The credential is the aggregator's own token file (`aggregator_auth`), which a
shell of this uid can read today whether or not rpc is on; that limit is
`aggregator_auth`'s docstring, unchanged. What rpc must not do is give the
honest path a way around a gate, and every nested call either passes the same
gates a direct call would or is one of the read-only tools none of the skipped
harness hooks ever denies.

A sandboxed session gets no env at all, and the sandbox unsets the credential
variables and covers the token file (`_tool_sandbox.bwrap_argv`), so a bench
trial cannot even read the secret — it has no network to use it on either.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import stat
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.harness import rpc_policy

logger = logging.getLogger("lloyd-mcp-rpc")

#: The deny list the loop stamped on THIS Bash call (`lloyd/rpc_deny`), bound
#: by `main.call_tool`. None when the call carried none — rpc is off for that
#: turn, and the shell gets no rpc env.
current_rpc_deny: contextvars.ContextVar[tuple[str, ...] | None] = \
    contextvars.ContextVar("lloyd_rpc_deny", default=None)

#: The foreground parent `_bash` registered, read back by `builtin_bash.call_tool`
#: to append the trailer and retire it.
current_parent: contextvars.ContextVar["Parent | None"] = \
    contextvars.ContextVar("lloyd_rpc_parent", default=None)

#: A background Bash has no timeout of its own; its nested calls are bounded
#: by the foreground ceiling (`builtin_bash.MAX_TIMEOUT_MS`).
BACKGROUND_DEADLINE_S = 600.0

_port: int = 8500


def set_server_port(port: int) -> None:
    """`main` tells this module where it listens (its `PORT`)."""
    global _port
    _port = int(port)


def server_url() -> str:
    return f"http://127.0.0.1:{_port}/mcp"


@dataclass
class Parent:
    """One Bash call whose shell may make nested calls."""
    parent_call_id: str
    session_id: str
    turn_id: str
    effect_scope: str
    surface: str
    deny: frozenset[str]
    deadline: float
    background: bool = False
    started: float = field(default_factory=time.monotonic)
    counts: Counter = field(default_factory=Counter)
    errors: int = 0
    seq: int = 0

    def child_meta(self, call_id: str) -> dict[str, Any]:
        """The `_meta` a nested call is dispatched with: the parent's, never
        the script's. No `lloyd/rpc_parent_call_id`, so the re-entry into
        `main.call_tool` is an ordinary call through every gate."""
        meta: dict[str, Any] = {"lloyd/session_id": self.session_id,
                                "lloyd/call_id": call_id}
        if self.turn_id:
            meta["lloyd/turn_id"] = self.turn_id
        if self.effect_scope:
            meta["lloyd/effect_scope"] = self.effect_scope
        if self.surface:
            meta["lloyd/surface"] = self.surface
        return meta


_lock = threading.Lock()
_parents: dict[str, Parent] = {}


def _prune(now: float) -> None:
    for key in [k for k, p in _parents.items() if p.deadline < now - 60.0]:
        _parents.pop(key, None)


def parent_for(parent_call_id: str) -> Parent | None:
    with _lock:
        return _parents.get(parent_call_id)


def reset_for_tests() -> None:
    with _lock:
        _parents.clear()


def _token_file_ok(path) -> bool:
    """The credential file exists, is ours, and nobody else can read it."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    return (stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid()
            and not (st.st_mode & 0o077))


def bash_env(*, sandboxed: bool, timeout_s: float, background: bool
             ) -> tuple[dict[str, str] | None, Parent | None]:
    """The child environment for a Bash call, and the parent it registered.

    `(None, None)` — inherit the aggregator's environment unchanged, today's
    behaviour — for a call the loop did not stamp (rpc off for that turn), for
    a sandboxed session, for a call with no session, and when the credential
    file is not private.
    """
    stamped = current_rpc_deny.get()
    if sandboxed or stamped is None:
        return None, None
    from agent_mcp import _task_registry, aggregator_auth, builtin_task
    from app.harness import policy as harness_policy

    sid = _task_registry.current_session_id.get() or ""
    if not sid:
        return None, None
    from agent_mcp import _tool_sandbox
    if _tool_sandbox.is_sandboxed_session(sid):  # belt: `sandboxed` is the verdict
        return None, None
    token_path = aggregator_auth.token_path()
    if not (os.environ.get(aggregator_auth.TOKEN_ENV) or _token_file_ok(token_path)):
        logger.warning("lloyd_rpc: credential file %s is missing or not 0600; "
                       "Bash runs without rpc env", token_path)
        return None, None

    call_id = _task_registry.current_call_id.get() or f"rpc-{uuid.uuid4().hex[:12]}"
    deny = frozenset(rpc_policy.bash_deny(stamped))
    surface = builtin_task.current_parent_surface.get() or ""
    if surface:
        try:
            from agent_mcp.annotations import hidden_on_surface
            deny = deny | hidden_on_surface(surface)
        except Exception:  # noqa: BLE001 — the loop's stamp already carries it
            pass
    budget = BACKGROUND_DEADLINE_S if background else float(timeout_s)
    wall_deadline = time.time() + max(budget - rpc_policy.DEADLINE_MARGIN_S, 1.0)
    parent = Parent(
        parent_call_id=call_id,
        session_id=sid,
        turn_id=_task_registry.current_turn_id.get() or "",
        effect_scope=harness_policy.current_effect_scope.get() or "",
        surface=surface,
        deny=deny,
        deadline=wall_deadline,
        background=background,
    )
    with _lock:
        _prune(time.time())
        _parents[call_id] = parent

    env = dict(os.environ)
    env.update({
        rpc_policy.ENV_URL: server_url(),
        rpc_policy.ENV_TOKEN_FILE: str(token_path),
        rpc_policy.ENV_SESSION_ID: sid,
        rpc_policy.ENV_TURN_ID: parent.turn_id,
        rpc_policy.ENV_PARENT_CALL_ID: call_id,
        rpc_policy.ENV_EFFECT_SCOPE: parent.effect_scope,
        rpc_policy.ENV_SURFACE: surface,
        rpc_policy.ENV_DENY: json.dumps(sorted(deny)),
        rpc_policy.ENV_DEADLINE: f"{wall_deadline:.3f}",
        rpc_policy.ENV_DEPTH: "1",
    })
    return env, parent


def finish(parent: Parent | None) -> str:
    """Retire a foreground parent; return its trailer ("" for no calls)."""
    if parent is None:
        return ""
    with _lock:
        _parents.pop(parent.parent_call_id, None)
    return rpc_policy.trailer(dict(parent.counts), parent.errors,
                              time.monotonic() - parent.started)


def admit(name: str, meta: dict) -> tuple[Parent | None, str | None]:
    """`(parent, None)` to dispatch a nested call, `(parent|None, reason)` to refuse."""
    parent_id = meta.get(rpc_policy.META_RPC_PARENT_CALL_ID)
    parent = parent_for(parent_id) if isinstance(parent_id, str) and parent_id else None
    if parent is None:
        return None, ("unknown or finished parent Bash call — lloyd_rpc works only "
                      "inside the Bash call whose environment it was given")
    if time.time() > parent.deadline:
        return parent, "the parent Bash call's rpc deadline has passed"
    depth = meta.get(rpc_policy.META_RPC_DEPTH, 1)
    if depth not in (1, "1"):
        return parent, "nested lloyd_rpc (depth > 1) is not allowed"
    from agent_mcp import _tool_sandbox
    from agent_mcp import annotations as tool_annotations
    if _tool_sandbox.is_sandboxed_session(parent.session_id):
        return parent, "a read-only (bench/eval) session has no lloyd_rpc"
    why = rpc_policy.refusal(name, parent.deny, tool_annotations.READ_ONLY)
    return parent, why


def next_call_id(parent: Parent) -> str:
    with _lock:
        parent.seq += 1
        return f"{parent.parent_call_id}.rpc{parent.seq}"


def record(parent: Parent | None, *, name: str, arguments: Any, ms: float,
           is_error: bool, refused: str | None = None) -> None:
    """Tally a nested call on its parent and write `harness.rpc_call`.

    Nothing enters the conversation: the event log is the record, keyed on
    `parent_call_id` so a timeline can hang each nested call under the Bash
    call that made it.
    """
    bare = rpc_policy.bare_name(name)
    if parent is not None:
        with _lock:
            parent.counts[bare] += 1
            if is_error:
                parent.errors += 1
    data: dict[str, Any] = {
        "parent_call_id": parent.parent_call_id if parent else "",
        "tool": bare,
        "args_digest": rpc_policy.args_digest(arguments),
        "ms": round(ms, 1),
        "is_error": bool(is_error),
    }
    if refused:
        data["refused"] = refused[:300]
    if parent is None:
        logger.warning("lloyd_rpc: refused %s with no live parent: %s", bare, refused)
        return
    try:
        from app.harness.telemetry import log_harness_event
        log_harness_event(parent.session_id, "harness.rpc_call", data,
                          turn_id=parent.turn_id or None)
    except Exception:  # noqa: BLE001 — diagnostics never fail the call
        logger.debug("lloyd_rpc: event log failed", exc_info=True)
