#!/usr/bin/env python3
"""
Lloyd unified MCP server — every tool in one Server("lloyd").

Run with:  python -m agent_mcp.main  (from ~/lloyd directory)
Endpoints: http://127.0.0.1:8500/mcp     MCP transport (Streamable HTTP)
           http://127.0.0.1:8500/health  discovery/liveness JSON

Protocol: MCP 2026-07-28, stateless. There is no initialize/initialized
handshake and no Mcp-Session-Id — every request carries its own context in
`_meta`, so any request can be served without prior state. The old
HTTP+SSE transport (a GET stream plus a separate /messages/ POST mount) is
deprecated upstream and gone here; it was also the source of the recurring
"Expected ASGI message 'http.response.body'" errors in logs/mcp.err,
which came from its `return Response()` teardown.

Module contract
---------------
Every entry in MODULES is a plain Python module exposing two coroutines::

    async def list_tools() -> list[Tool]
    async def call_tool(name: str, arguments: dict) -> list[TextContent]

and optionally::

    async def shutdown() -> None      # release long-lived resources

Modules do NOT create their own `mcp.server.Server`. They used to, and the
instances were dead weight: the SDK's `@server.list_tools()` decorator
registers a handler and returns the function unchanged, so every per-module
`Server` held a request-handler map that nothing ever dispatched. The
aggregator calls the module functions directly. See `_check_module`.
"""

import asyncio
import json
import logging
import os
import time as _time
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import uvicorn
from mcp.server.caching import CacheHint
from mcp.server.lowlevel import Server
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_mcp import (
    _change_ledger,
    _injection_probe,
    _subagent_registry,
    aggregator_auth,
    _task_registry,
    _tool_effects,
    _tool_sandbox,
    _tsc_runner,
    annotations as tool_annotations,
    ambient,
    autonomy,
    autoresearch,
    backlog,
    browser,
    builtin_bash,
    builtin_fs,
    builtin_goal,
    builtin_grants,
    builtin_plan,
    builtin_task,
    builtin_todo,
    code_graph,
    desktop,
    discord_bot,
    djev,
    facts,
    http_tools,
    ide,
    mission_control,
    mission_control_ui,
    research,
    automod,
    session,
    skills,
    thunderbird,
    vault,
)
# #544: the effect-scope contextvar the harness loop reads at dispatch. Bound
# here around each dispatch so a `Task` subagent's nested loop inherits it.
# `app_paths`, not `paths`: the `/changes/revert` route below binds a local named
# `paths` for the request body's list of files, and a module-level name that can be
# shadowed in the same file is a name a later reader gets wrong.
from app import paths as app_paths
from app.harness import policy as harness_policy
from app.harness.safety import check_bash_command, desktop_refusal

logger = logging.getLogger("lloyd-mcp")


def _resolve_port() -> int:
    """Bind port for the aggregator.

    Every *client* of this server resolves its URL through `services.lloyd_mcp`
    in config.yaml (`app/config.py::service_url`), but the server itself used
    to bind a hardcoded 8500 — so moving the port in config silently pointed
    every client at a server that was still on the old one. Resolve from the
    same registry the clients use, with an env override so a canary can take a
    second port without a config edit.
    """
    env = os.environ.get("LLOYD_MCP_PORT")
    if env:
        return int(env)
    try:
        from app.config import service_url
        parsed = urlparse(service_url("lloyd_mcp", "http://127.0.0.1:8500/mcp"))
        return parsed.port or 8500
    except Exception:
        return 8500


PORT = _resolve_port()

# memory.py was split into facts/vault/session in #340 PR 5. The legacy
# memory module remains as a backward-compat re-export shim for callers
# (prefetch.py, app/post_capture.py) but is NOT in MODULES — including it
# would double-register every tool.
MODULES = [
    # Built-in tool replicas (formerly provided by claude-agent-sdk)
    builtin_bash,
    builtin_fs,
    builtin_goal,
    builtin_grants,
    builtin_plan,
    builtin_task,
    builtin_todo,
    # Domain modules
    ambient,
    autonomy,
    autoresearch,
    backlog,
    browser,
    desktop,
    code_graph,
    discord_bot,
    djev,
    facts,
    vault,
    session,
    mission_control,
    mission_control_ui,
    ide,
    research,
    automod,
    skills,
    http_tools,
    thunderbird,
]


@runtime_checkable
class ToolModule(Protocol):
    """Structural contract every MODULES entry satisfies."""

    async def list_tools(self) -> list[Tool]: ...

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult | list[TextContent]: ...


def _check_module(mod: Any) -> None:
    """Fail at import time if a module doesn't meet the contract.

    Cheaper than discovering it on the first tools/list in production —
    a module missing `call_tool` would otherwise register its tools fine
    and then fail every dispatch.
    """
    for attr in ("list_tools", "call_tool"):
        fn = getattr(mod, attr, None)
        if not callable(fn):
            raise TypeError(f"{mod.__name__} is missing a callable {attr}()")
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(f"{mod.__name__}.{attr}() must be async")


for _mod in MODULES:
    _check_module(_mod)


# tool_name -> module, rebuilt on each list_tools call.
#
# Never mutated in place. `list_tools` builds a fresh dict and rebinds this
# name in one statement, which is atomic under the GIL — the previous map
# keeps serving concurrent call_tool dispatches until the instant it is
# replaced. The old code cleared this dict first and refilled it across 22
# `await` boundaries, so any tool call arriving from another MCP session in
# that window routed to nothing and the model was told "Unknown tool".
_dispatch: dict[str, Any] = {}

# Per-module discovery state, surfaced by /health.
_discovery_status: dict[str, dict[str, Any]] = {}

# `_meta` key carrying the harness session id. Namespaced per the MCP
# convention for implementation-specific metadata.
META_SESSION_ID = "lloyd/session_id"

# `_meta` keys carrying the CALLING turn's model and endpoint. Only the
# Task tool consumes them, so a subagent can inherit the model of the turn
# that spawned it rather than defaulting to `primary`. Must match
# app.harness.mcp_pool.META_MODEL / META_BASE_URL.
META_MODEL = "lloyd/model"
META_BASE_URL = "lloyd/base_url"

# `_meta` key carrying the model's caption for this call — the injected
# `summary` display parameter, which the harness pops before dispatch so it
# never reaches a tool's arguments (nothing validates those against the
# inputSchema; an unknown key would reach the handler). Consumed by the two tools
# that label a row someone reads later (background Bash, Task). Must match
# app.harness.mcp_pool.META_SUMMARY.
META_SUMMARY = "lloyd/summary"

# `_meta` keys identifying the chat turn and the individual tool call. The
# change ledger records a file write against (session, turn) so a chat can
# show what a turn changed and offer a revert. Must match
# app.harness.mcp_pool.META_TURN_ID / META_CALL_ID.
META_TURN_ID = "lloyd/turn_id"
META_CALL_ID = "lloyd/call_id"

# `_meta` key carrying the #544 effect scope: the queue item a worker turn is
# running (`item:<source>:<id>`), bound by the pool and forwarded by the
# harness. It is what makes a retry's second `email_send` recognisable as the
# same effect rather than a new one — an item's id is stable across its attempts
# and never reused, so the guard covers exactly the retry and nothing else.
# Absent for an interactive turn, which leaves that turn unguarded by design
# (see agent_mcp/_tool_effects.py). Must match
# app.harness.mcp_pool.META_EFFECT_SCOPE.
META_EFFECT_SCOPE = "lloyd/effect_scope"
# The calling turn's tool surface, inherited by a Task subagent. Must match
# app.harness.mcp_pool.META_SURFACE.
META_SURFACE = "lloyd/surface"
# The calling turn's #534 grant scope and its live deny list, inherited by a
# Task subagent (review 2026-09-24, D4). Must match
# app.harness.mcp_pool.META_GRANT_SCOPE / META_DISALLOWED.
META_GRANT_SCOPE = "lloyd/grant_scope"
META_DISALLOWED = "lloyd/disallowed_tools"

# OpenAI's spec caps tool names at 64 chars. Enforced here at registration
# so a bad name fails loudly on the first list_tools() instead of
# mid-conversation in the harness translator (tool_schema.py keeps its own
# check as a backstop for non-aggregator servers).
TOOL_NAME_MAX = 64

# How long a client may treat a tools/list result as fresh. Short enough
# that toggling a tool in the Tools page reaches a running harness.
TOOLS_LIST_TTL_MS = 60_000


async def list_tools() -> list[Tool]:
    """Aggregate every module's tools.

    A module that raises is logged, recorded in `_discovery_status` and
    skipped — one broken module must not take the whole tool surface with
    it. Before this guard, a single raising module failed tools/list, which
    failed the harness pool open, which left the agent with no tools at all.
    """
    global _dispatch
    new_dispatch: dict[str, Any] = {}
    new_status: dict[str, dict[str, Any]] = {}
    all_tools: list[Tool] = []

    for mod in MODULES:
        name = mod.__name__.rsplit(".", 1)[-1]
        try:
            tools = await mod.list_tools()
        except Exception as exc:
            logger.exception("list_tools: module %s failed discovery", name)
            new_status[name] = {"tools": 0, "ok": False, "error": str(exc)[:300]}
            continue

        kept = 0
        for tool in tools:
            if len(tool.name) > TOOL_NAME_MAX:
                logger.error(
                    "list_tools: dropping %r from %s — %d chars (max %d)",
                    tool.name, name, len(tool.name), TOOL_NAME_MAX,
                )
                continue
            prior = new_dispatch.get(tool.name)
            if prior is not None:
                # Silent shadowing was possible here: the last module to
                # claim a name won and the tool appeared twice in the
                # advertised list. MCPPool logs collisions; so do we.
                logger.error(
                    "list_tools: duplicate tool %r — %s keeps it, %s ignored",
                    tool.name, prior.__name__.rsplit(".", 1)[-1], name,
                )
                continue
            new_dispatch[tool.name] = mod
            all_tools.append(tool_annotations.annotate(tool))
            kept += 1
        new_status[name] = {"tools": kept, "ok": True, "error": None}

    _dispatch = new_dispatch
    _discovery_status.clear()
    _discovery_status.update(new_status)
    return all_tools


async def on_list_tools(ctx, params) -> ListToolsResult:
    """tools/list handler.

    `ttl_ms`/`cache_scope` let the client cache this result instead of
    re-asking on every connection — the 2026-07-28 replacement for both
    our hand-rolled discovery caches and `tools/list_changed`. Kept short
    enough that a tool toggled in the Tools page reaches a running harness
    within the window, rather than never (discovery used to be frozen for
    the life of a pool).
    """
    return ListToolsResult(
        tools=await list_tools(),
        ttl_ms=TOOLS_LIST_TTL_MS,
        cache_scope="private",
    )


def _bound_session_id(arguments: dict, meta: Any = None) -> str:
    """Resolve the harness session id for the in-flight tool call.

    Preferred source is the request's `_meta` — the field the MCP spec
    reserves for implementation metadata, and where the 2026-07-28 spec
    puts all per-request context. Falls back to the legacy `_session_id`
    argument so a harness and aggregator at different versions still
    correlate. `_meta` is strictly better than the argument form. Nothing
    validates `arguments` against the tool's inputSchema — the low-level
    server checks only the JSON-RPC params model, the harness pool coerces
    top-level primitives only, and unknown keys reach the handler — so an
    injected argument is not rejected; it lands in the handler's arguments
    as though the model had sent it, where per-parameter checks are the
    handler's own job.
    """
    if isinstance(meta, dict):
        sid = meta.get(META_SESSION_ID)
        if isinstance(sid, str) and sid:
            return sid
    if isinstance(arguments, dict):
        return arguments.get("_session_id", "") or ""
    return ""


def _result_is_error(result: Any) -> bool:
    """Read `isError` across the mcp 1.x/2.x field-name split.

    `app/harness/mcp_pool.py` documents this asymmetry at length: 2.x renamed
    the attribute to snake_case in Python while the wire format stayed
    camelCase, so `getattr(result, "isError", False)` returns False on a 2.x
    `CallToolResult` rather than raising. Here that failure is not cosmetic —
    every failed effect would be recorded `ok`, and `ok` is the state that
    licenses unlimited replays of a half-written effect.
    """
    for attr in ("is_error", "isError"):
        value = getattr(result, attr, None)
        if value is not None:
            return bool(value)
    return False


def _effect_text(result: Any) -> tuple[str, bool]:
    """`(joined text, is_error)` from either shape a module may return.

    Only the text half is stored for a replay, so a second attempt gets back
    what the first one's caller saw. A result carrying non-text blocks (an
    image) replays as its text plus a note naming the digest — the effect is
    already fire-safe either way, and losing an attachment on a replay is a
    smaller lie than firing the effect again to get it.
    """
    if isinstance(result, CallToolResult):
        blocks = list(result.content or [])
        return ("\n".join(b.text for b in blocks
                          if getattr(b, "type", "") == "text"),
                _result_is_error(result))
    blocks = list(result or [])
    return ("\n".join(b.text for b in blocks
                      if getattr(b, "type", "") == "text"), False)


def _effect_refused(name: str, effect: "_tool_effects.Claim",
                    arguments: Any = None) -> CallToolResult:
    """The payload for a call the ledger will not let fire twice.

    Both branches log at WARNING with the key AND the canonical arguments.
    Over-suppression is the failure that bites here — a silently dropped email
    is quieter and worse than the duplicate this exists to prevent — so a
    suppression has to be findable in the log with the arguments that produced
    it, not merely counted. The first cut logged a key prefix only, which
    names nothing a human can grep for.
    """
    short = (effect.key or "")[:16]
    args_text = _tool_effects.canonical_arguments(arguments)[:300]
    if effect.unknown:
        logger.warning(
            "tool_effects: refused re-fire of %s (effect %s…) args=%s. State "
            "unknown: a prior attempt was cancelled with this effect in flight. "
            "Do the status lookup before assuming it did not land.",
            name, short, args_text)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({
                "error": (
                    "#544 effect ledger: this call was already fired once in "
                    "this scope and its outcome was never recorded — the "
                    "attempt was cancelled with the effect in flight. A "
                    "timeout means unknown, not failure, so it is not "
                    "re-fired."
                ),
                "tool": name,
                "effect_key": short,
                "effect_state": "unknown",
                "do_not": "re-fire this call with these arguments",
                "next_step": (
                    "Read the state back to find out whether the effect landed "
                    "(backlog_tasks / email_recent / calendar_events / "
                    "vault_read / fact_get — whatever this tool writes), then "
                    "continue with the work that does not duplicate it."
                ),
            }))],
            isError=True,
        )
    logger.warning("tool_effects: suppressed duplicate %s (effect %s…) args=%s; "
                   "replaying the recorded result", name, short, args_text)
    if effect.replay_truncated or not effect.replay_text:
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({
                "effect_state": "already_applied",
                "tool": name,
                "effect_key": short,
                "note": (
                    "An identical call already succeeded in this scope. Its "
                    "result body was too large to replay; digest "
                    f"{effect.replay_digest[:16]}… . The effect did happen — "
                    "verify by reading the state it writes rather than by "
                    "firing it again."
                ),
            }))],
            isError=False,
        )
    return CallToolResult(
        content=[
            TextContent(type="text", text=effect.replay_text),
            TextContent(type="text", text=(
                f"[#544] Replayed result of an identical {name} call that "
                f"already fired in this scope (effect {short}…). Nothing ran "
                f"again."
            )),
        ],
        isError=False,
    )


def _safety_parent_of(session_id: str) -> str | None:
    """A `task:*` subagent's parent session, for the service-control rule:
    a subagent of a worker turn is a worker turn."""
    try:
        from agent_mcp import _subagent_registry
        parent = _subagent_registry.parent_scope(session_id)
        return parent[0] if parent else None
    except Exception:  # noqa: BLE001 — unresolvable means not refused
        return None


async def call_tool(name: str, arguments: dict, meta: Any = None):
    if not _dispatch:
        await list_tools()
    mod = _dispatch.get(name)
    if not mod:
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))],
            isError=True,
        )

    sid = _bound_session_id(arguments, meta)
    # Strip the legacy argument form before the per-tool handler validates,
    # so module schemas never have to advertise an internal field.
    if isinstance(arguments, dict) and "_session_id" in arguments:
        arguments = {k: v for k, v in arguments.items() if k != "_session_id"}

    # Refusals that do not depend on who installed which hook. Both sit above
    # the effect ledger's claim, for the reason given there: a call that was
    # never allowed to run must never be recorded as an `unknown` effect.
    #
    # 1. A bench or eval session may observe this machine and never change it
    #    (`_tool_sandbox`). The vault was deleted twice, on 2026-09-10 and
    #    2026-09-12, by a bench trial doing what its prompt said.
    sandboxed = _tool_sandbox.is_sandboxed_session(sid)
    if sandboxed:
        # Off the loop: the first call probes bwrap with a subprocess.
        why = await asyncio.to_thread(_tool_sandbox.refusal, name, arguments)
        if why:
            logger.warning("tool_sandbox: refused %s for read-only session %s: %s",
                           name, sid, why)
            return _refused_call(name, f"read-only session: {why}")
    # 2. The destructive-command check, for every session. The harness hook
    #    runs the same function, but only where a caller installed it; this is
    #    where the command actually executes, so nothing reaches a shell
    #    without passing it.
    if name == "Bash" and isinstance(arguments, dict):
        cwd = arguments.get("cwd")
        match = check_bash_command(arguments.get("command") or "",
                                   cwd if isinstance(cwd, str) and cwd else None,
                                   at_dispatch=True, session_id=sid,
                                   parent_of=_safety_parent_of)
        if match:
            label, excerpt = match
            logger.warning("safety: refused Bash for session %s: %s — %r",
                           sid, label, (arguments.get("command") or "")[:500])
            return _refused_call(name, f"harness safety: blocked {label!r} on {excerpt!r}")

    # 2b. Desktop computer use is for a person's chat only. The screen is
    #     Alan's private desktop and the input lands under Alan's hands, so a
    #     worker, an autonomy task, a bench or eval trial, or a subagent of
    #     one may neither look at it nor touch it — refused here, where every
    #     caller passes, not in a hook a caller may not have installed.
    if name.startswith("desktop_"):
        from app.harness.service_control import is_background_session
        if sandboxed or not sid or is_background_session(sid, parent_of=_safety_parent_of):
            logger.warning("desktop: refused %s for non-chat session %r", name, sid)
            return _refused_call(
                name, "desktop computer use is only available in a person's chat "
                      "session — never to background, worker, bench or sessionless "
                      "calls (the screen and input are Alan's)")

    # 2c. Nothing but Alan grants the desktop lease, and no tool call reads or
    #     feeds the retained frame mirror either (#1418). Bash is covered by
    #     `check_bash_command`; this covers every other tool that could reach a
    #     route (http_request allows loopback) or the file (Write/Edit).
    #     `desktop_refusal` matches only the argument fields that can actually
    #     arrive somewhere — a URL, a path — so a backlog item or vault note
    #     whose body merely describes the lease route is documentation and
    #     passes, where scanning the whole serialized blob denied it (#1418
    #     clause 4).
    if name != "Bash" and name not in tool_annotations.READ_ONLY \
            and isinstance(arguments, dict):
        why = desktop_refusal(arguments)
        if why:
            logger.warning("desktop: refused %s reaching a desktop route for %r",
                           name, sid)
            return _refused_call(name, why)

    # 3. A state-changing call that arrives with no session id is refused
    #    (#1053). `_tool_sandbox.is_sandboxed_session("")` is False by design —
    #    an empty id cannot be a bench id — so before this line the *absence* of
    #    a session id was itself a way to read as "not sandboxed": a bench trial
    #    that dropped its own id from `_meta` got the whole write surface back,
    #    and so did anything else that omitted it. Attributing a write to nothing
    #    is also what blinds the change ledger and the effect ledger, which both
    #    key on the session.
    #
    #    Every caller this can plausibly break names a session: `loop.py` mints
    #    one before its first dispatch (`run_query` falls back to a uuid when
    #    `options.session_id` is empty) and passes it as `session_id` in
    #    `call_kw`; the background-task drain at `app/routers/messages.py` passes
    #    one too. A caller that genuinely has no session and needs to write is
    #    rare enough that the honest answer is to give it one, not to reopen the
    #    hole. Read-only tools keep dispatching with no session id, so
    #    discovery, `tools/list` and every probe are untouched.
    if not sid and _tool_sandbox.state_changing_tool(name):
        logger.warning("sessionless write: refused %s with no session id", name)
        return _refused_call(
            name,
            "a call that can change state arrived with no session id — refusing "
            "rather than treating a missing session id as 'not sandboxed' "
            "(#1053); pass the session id in `lloyd/session_id` `_meta`")

    parent_model = meta.get(META_MODEL, "") if isinstance(meta, dict) else ""
    parent_base_url = meta.get(META_BASE_URL, "") if isinstance(meta, dict) else ""
    call_summary = meta.get(META_SUMMARY, "") if isinstance(meta, dict) else ""
    turn_id = meta.get(META_TURN_ID, "") if isinstance(meta, dict) else ""
    call_id = meta.get(META_CALL_ID, "") if isinstance(meta, dict) else ""
    effect_scope = (meta.get(META_EFFECT_SCOPE, "") if isinstance(meta, dict) else "")
    if not isinstance(effect_scope, str):
        effect_scope = ""

    token = _task_registry.current_session_id.set(sid)
    stok = _task_registry.current_call_summary.set(
        call_summary if isinstance(call_summary, str) else ""
    )
    mtok = builtin_task.current_parent_model.set(
        parent_model if isinstance(parent_model, str) else ""
    )
    btok = builtin_task.current_parent_base_url.set(
        parent_base_url if isinstance(parent_base_url, str) else ""
    )
    ttok = _task_registry.current_turn_id.set(
        turn_id if isinstance(turn_id, str) else ""
    )
    ctok = _task_registry.current_call_id.set(
        call_id if isinstance(call_id, str) else ""
    )
    parent_surface = meta.get(META_SURFACE, "") if isinstance(meta, dict) else ""
    surtok = builtin_task.current_parent_surface.set(
        parent_surface if isinstance(parent_surface, str) else ""
    )
    # #544: a `Task` subagent re-enters this function over loopback `/mcp`
    # from a fresh ASGI task, so only `_meta` crosses — and its own loop
    # stamps `_meta` from `policy.current_effect_scope` (loop.py). Binding the
    # incoming scope here, around the dispatch that runs the subagent, is
    # what lets a subagent's writes inside a worker item be ledgered under
    # that item. Inherited verbatim, not suffixed by task id: a retry mints a
    # new task id, and an identical write from parent and child is the same
    # effect.
    etok = harness_policy.current_effect_scope.set(effect_scope)
    # D4: the parent's grant scope and deny list, for a Task child. A child's
    # hooks and options are built inside this dispatch, so contextvars bound
    # here are what `builtin_task` reads. `current_scope` is bound too (only
    # when a scope came in — its default stays the restrictive "worker"), so
    # the child's outbound content gate reads the parent's scope rather than
    # the default.
    parent_grant_scope = meta.get(META_GRANT_SCOPE, "") if isinstance(meta, dict) else ""
    if not isinstance(parent_grant_scope, str):
        parent_grant_scope = ""
    parent_disallowed = meta.get(META_DISALLOWED, ()) if isinstance(meta, dict) else ()
    if not isinstance(parent_disallowed, (list, tuple)):
        parent_disallowed = ()
    gstok = builtin_task.current_parent_grant_scope.set(parent_grant_scope)
    dtok = builtin_task.current_parent_disallowed.set(
        tuple(str(n) for n in parent_disallowed if isinstance(n, str) and n))
    pstok = (harness_policy.current_scope.set(parent_grant_scope)
             if parent_grant_scope else None)
    sbtok = _tool_sandbox.current_sandboxed.set(sandboxed)
    try:
        # #544 — exactly-once EFFECT, not exactly-once scheduling. The retry
        # that makes this necessary is the pool's: a job cancelled at
        # `max_duration_seconds` is recorded failed, requeued, and re-runs its
        # whole turn, so every `email_send` / `backlog_write_task` / `vault_write`
        # the first attempt landed lands again. `effect_scope` is the queue item,
        # stable across that item's attempts and never reused, so this covers the
        # retry and nothing else. A read-only call returns from `claim()` on a
        # frozenset membership test — no connect, no key, no SELECT.
        #
        # Ordering is deliberate: everything that can REFUSE this call runs
        # upstream of it. The grant/authority gate is a PreToolUse hook in the
        # harness (`app.harness.policy.install_policy_hook`), so a call that was
        # never allowed to run never reaches this line and can never be written
        # to the ledger as an `unknown` effect. #521's deterministic policy gate
        # belongs on that same upstream side — or above this block, never below
        # it, for the same reason.
        effect = await _tool_effects.claim(name, arguments, effect_scope, sid)
        if not effect.may_dispatch:
            return _effect_refused(name, effect, arguments)
        try:
            result = await mod.call_tool(name, arguments)
        except BaseException as exc:
            # Leave the row `unknown`, which is the honest state: the handler may
            # have finished the effect before it died. `finish` is not called
            # here on purpose — recording `error` would license a retry to fire a
            # second, possibly-duplicate effect, which is exactly the lie the
            # pool's timeout branch tells today.
            if effect.key and not _tool_effects.is_shadow_scope(effect_scope):
                logger.warning(
                    "tool_effects: %s raised %s in scope %r; effect %s… stays "
                    "UNKNOWN — an identical call will be refused until a status "
                    "lookup resolves it",
                    name, type(exc).__name__, effect_scope, effect.key[:16])
            raise
        if effect.key:
            await _tool_effects.finish(effect.key, *_effect_text(result))
        # P10 seam 2: instruction-shaped text in what a background session
        # just read. After the effect ledger, so a replay stores the tool's own
        # result and never the probe's warning. `apply` never raises; the
        # guard around it is for the classification above it.
        if name in _injection_probe.PROBED_TOOLS and sid:
            try:
                from app.harness.service_control import is_background_session
                background = is_background_session(sid, parent_of=_safety_parent_of)
                parent = ((_safety_parent_of(sid) if sid.startswith("task:")
                           else None) or sid) if background else sid
            except Exception:  # noqa: BLE001
                background, parent = False, sid
            if background:
                result = await _injection_probe.apply(
                    name, result, session_id=sid, is_background=True,
                    call_id=call_id if isinstance(call_id, str) else "",
                    log_session=parent)
        return result
    finally:
        _task_registry.current_session_id.reset(token)
        _task_registry.current_call_summary.reset(stok)
        _task_registry.current_turn_id.reset(ttok)
        _task_registry.current_call_id.reset(ctok)
        builtin_task.current_parent_model.reset(mtok)
        builtin_task.current_parent_base_url.reset(btok)
        builtin_task.current_parent_surface.reset(surtok)
        harness_policy.current_effect_scope.reset(etok)
        builtin_task.current_parent_grant_scope.reset(gstok)
        builtin_task.current_parent_disallowed.reset(dtok)
        if pstok is not None:
            harness_policy.current_scope.reset(pstok)
        _tool_sandbox.current_sandboxed.reset(sbtok)


def _refused_call(name: str, reason: str) -> CallToolResult:
    """A call refused before dispatch. Worded like the harness's own hook
    deny ("Tool call denied: …") so a bench trace files it under
    `denied_calls` — the model tried, and the attempt is the measurement."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({
            "error": f"Tool call denied: {reason}",
            "tool": name,
        }))],
        isError=True,
    )


async def on_call_tool(ctx, params) -> CallToolResult:
    """tools/call handler.

    `ctx.meta` carries the caller's per-request context — on 2026-07-28
    that includes the protocol version and client info the old handshake
    used to negotiate once, plus our own `lloyd/session_id`.
    """
    result = await call_tool(params.name, params.arguments or {}, ctx.meta)
    if isinstance(result, CallToolResult):
        return result
    # A module that still returns a bare content list.
    return CallToolResult(content=list(result), isError=False)


# DNS-rebinding protection. The aggregator binds loopback with no auth, so
# the only thing standing between a page in the user's browser and this
# tool surface is that a cross-origin POST needs a preflight it won't get.
# That's an accident of content-type rules, not a control — the SDK ships
# the actual control, so use it.
#
# Hosts and origins are matched by NAME with a wildcard port. The threat
# this blocks is DNS rebinding — a page on an attacker's domain resolving
# that domain to 127.0.0.1 and talking to this server; such a request
# carries the attacker's hostname in Host/Origin, which is what gets
# rejected. The port is not part of that defence, and pinning it to PORT
# is actively harmful: the aggregator answers /health happily on any other
# port while every MCP request fails 421 "Invalid Host header" — a
# silent-partial-failure of exactly the kind this review set out to remove.
_LOOPBACK_HOSTS = ["127.0.0.1", "localhost", "[::1]"]
_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[f"{h}:*" for h in _LOOPBACK_HOSTS] + _LOOPBACK_HOSTS,
    allowed_origins=(
        [f"http://{h}:*" for h in _LOOPBACK_HOSTS]
        + [f"http://{h}" for h in _LOOPBACK_HOSTS]
    ),
)


async def health(request):
    """Discovery + liveness JSON.

    supervisord only knows whether the process is up. This says whether
    each module actually produced tools, which is the failure that
    matters: a degraded aggregator serves fewer tools and looks fine.
    """
    if not _discovery_status:
        await list_tools()
    degraded = sorted(n for n, s in _discovery_status.items() if not s["ok"])
    return JSONResponse({
        "status": "degraded" if degraded else "ok",
        "tools": len(_dispatch),
        "modules": len(MODULES),
        "degraded_modules": degraded,
        "discovery": _discovery_status,
    }, status_code=200 if not degraded else 503)


def _task_row(r) -> dict:
    """One background-bash row for the dashboard, running or finished.

    A running task's elapsed time is measured against now; a finished one
    against when it finished, or its duration keeps ticking up forever and
    a task that ran for two seconds reads as hours old by evening.
    """
    end = r.finished_at if r.finished_at is not None else _time.time()
    return {
        "task_id": r.task_id,
        "session_id": r.session_id,
        "description": r.description,
        "command": r.command[:200],
        "status": r.status,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
        "exit_code": r.exit_code,
        "elapsed_s": round(end - r.started_at, 1),
        "output_path": str(r.output_path),
    }


def _qmd_client_stats() -> dict:
    """What the qmd daemon has told this process about its own answers."""
    try:
        from app import qmd_health
        return qmd_health.stats()
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


async def state(request):
    """Live agent-side state for the Mission Control dashboard.

    Subagents and background bash tasks both run inside THIS process,
    not the backend — the aggregator owns the `Task` tool and spawns the
    `Bash(run_in_background=true)` children. The backend has no handle on
    either, so it reads them over loopback here rather than through a
    shared file, which would need a lock and would still be a snapshot of
    the past.
    """
    tasks = [_task_row(r) for r in _task_registry.list_active()]
    recent_tasks = [_task_row(r) for r in _task_registry.list_recent(10)]
    return JSONResponse({
        "subagents": _subagent_registry.snapshot(),
        "background_tasks": {
            "active": tasks,
            "active_count": len(tasks),
            # Without this a background task vanished the moment it exited,
            # so a failure three seconds in looked like it never ran.
            "recent": recent_tasks,
        },
        "tsc": _tsc_runner.stats(),
        "qmd": _qmd_client_stats(),
        "changes": _change_ledger.stats(),
        "tools": len(_dispatch),
        # Read by the bench runner before it starts a trial: a runner that
        # cannot see the sandbox enforced refuses to run.
        "tool_sandbox": await asyncio.to_thread(_tool_sandbox.status),
    })


async def loaded_paths(request):
    """`POST /loaded {paths}`: which of them THIS process has loaded as
    modules. The automod promoter asks before a landing, of the backend and of
    this process both — a commit neither has loaded needs no restart
    (`app/loaded_paths.py`). Read-only, and inside the credential like every
    route but `/health`."""
    from app.loaded_paths import answer
    try:
        data = await request.json()
    except Exception:
        data = {}
    body = answer((data or {}).get("paths"))
    return JSONResponse(body, status_code=400 if "error" in body else 200)


async def browser_navigate(request):
    """Mission Control's URL bar. `POST /browser/navigate {url}`.

    Playwright lives in this process, so the backend has no handle on the
    browser and proxies here — the same seam the dashboard crosses to read
    `/state`. It is a route rather than a tool because the user typing a URL
    is not the agent calling something, and it must not be logged as one.

    A navigation that fails is a 200 carrying `error`: DNS not resolving is
    an answer for the viewer to read, not a broken request. Only a malformed
    body is a 4xx.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be an object"}, status_code=400)
    url = body.get("url")
    if not isinstance(url, str) or not url.strip():
        return JSONResponse({"error": "url is required"}, status_code=400)

    try:
        result = await browser.navigate_from_ui(url)
    except Exception as e:
        logger.warning("browser/navigate failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=200)
    return JSONResponse(result)


async def subagent_control(request):
    """Stop or steer a running `Task` child. `POST /subagents/{task_id}/{verb}`.

    The children run in this process, so Mission Control's buttons are proxied
    here by the backend (`app/routers/subagents.py`). Authority is the
    registry's own `orchestrator-session` rule, applied to the session the
    request names — see `_subagent_registry.control_request`.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    status, out = _subagent_registry.control_request(
        request.path_params.get("verb", ""),
        request.path_params.get("task_id", ""), body)
    return JSONResponse(out, status_code=status)


async def changes(request):
    """What a turn wrote. `GET /changes?session=<sid>&turn=<turn_id>`."""
    session = request.query_params.get("session") or ""
    turn = request.query_params.get("turn") or ""
    if not session or not turn:
        return JSONResponse({"error": "session and turn are required"},
                            status_code=400)
    return JSONResponse({
        "session_id": session,
        "turn_id": turn,
        "files": _change_ledger.list_changes(session, turn),
    })


async def changes_revert(request):
    """Undo a turn's writes. `POST /changes/revert {session, turn, paths?}`.

    Per file, and never silently: a file something else has written since is
    refused by name rather than skipped, because restoring the pre-image
    there would destroy that other write — the exact damage this exists to
    prevent.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    session = str(body.get("session") or "")
    turn = str(body.get("turn") or "")
    if not session or not turn:
        return JSONResponse({"error": "session and turn are required"},
                            status_code=400)
    paths = body.get("paths")
    paths = [str(p) for p in paths] if isinstance(paths, list) else None
    return JSONResponse({
        "session_id": session,
        "turn_id": turn,
        "results": _change_ledger.revert(session, turn, paths),
    })


@asynccontextmanager
async def lifespan(app):
    # The runtime-state directories this process writes into (#712). The
    # aggregator is a separate process from the backend, so the backend's startup
    # hook cannot cover it, and `app.paths` no longer creates anything at import:
    # this is the moment the sessions dir and the change ledger exist. Ahead of
    # the ledger prune below for the same reason.
    app_paths.ensure_dirs()
    await discord_bot.start_bot_task()
    # Seed the tsc baseline a little after boot. Without a baseline the
    # first run attributes every pre-existing error in the tree to whoever
    # edited first, so the first `.tsx` edit after every restart would be the
    # one that gets a wrong answer. Detached and failure-tolerant: this must
    # never delay or block startup.
    warm = asyncio.create_task(_tsc_runner.warm_baseline(), name="tsc-warm-baseline")
    # Retention for the change ledger. On a thread because it stats every
    # snapshot under `sessions/*.changes/`, and this loop serves every tool
    # call in the process.
    try:
        await asyncio.to_thread(_change_ledger.prune)
    except Exception:
        logger.exception("lifespan: change-ledger prune failed")
    try:
        yield
    finally:
        warm.cancel()
        try:
            await _tsc_runner.shutdown()
        except Exception:
            logger.exception("lifespan: tsc runner shutdown failed")
        try:
            await discord_bot.stop_bot()
        finally:
            # Modules holding long-lived external resources (a Chromium
            # under Playwright, a Node bridge subprocess) get a chance to
            # release them. Previously only the Discord bot and the
            # background-task registry were torn down, so every restart
            # orphaned a browser and a node process.
            for mod in MODULES:
                fn = getattr(mod, "shutdown", None)
                if fn is None:
                    continue
                try:
                    await fn()
                except Exception:
                    logger.exception(
                        "lifespan: %s.shutdown() failed",
                        mod.__name__.rsplit(".", 1)[-1],
                    )
            await _task_registry.terminate_all()


combined = Server(
    "lloyd",
    version="2.0",
    instructions="Lloyd's unified tool surface: filesystem, shell, knowledge "
                 "graph, vault, mail, calendar, browser and automation.",
    lifespan=lifespan,
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
    # Protocol-level freshness hints. The client caches tools/list for this
    # long instead of re-querying on every connection.
    cache_hints={"tools/list": CacheHint(ttl_ms=TOOLS_LIST_TTL_MS, scope="private")},
)

# Streamable HTTP replaces the GET-stream + POST-mount pair. `stateless_http`
# matches the 2026-07-28 core: no session is pinned to this process, so a
# dropped connection costs a reconnect (~6ms) rather than a torn-down
# session shared by every in-flight turn.
# `json_response=True` returns each response as a plain JSON body instead
# of wrapping it in an SSE event.
#
# This is not a preference — it is required. Streamable HTTP's SSE framing
# runs through httpx2's parser, which enforces a 1 MiB
# DEFAULT_MAX_EVENT_SIZE_BYTES, and mcp's client constructs its
# `EventSource(response)` with no way to raise it. A tool result above
# 1 MiB (fact_get on a well-connected entity returns ~1.4 MB) therefore
# died as "SSE stream ended without a response" — with the real cause
# swallowed into a debug log. We use no progress notifications or partial
# streaming, so the SSE framing buys nothing and costs a size ceiling.
starlette_app = combined.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
    transport_security=_security,
    custom_starlette_routes=[
        Route("/health", health, methods=["GET"]),
        Route("/state", state, methods=["GET"]),
        Route("/loaded", loaded_paths, methods=["POST"]),
        Route("/browser/navigate", browser_navigate, methods=["POST"]),
        Route("/changes", changes, methods=["GET"]),
        Route("/changes/revert", changes_revert, methods=["POST"]),
        Route("/subagents/{task_id}/{verb}", subagent_control, methods=["POST"]),
    ],
)

# Everything above is the route table; this is the one control that applies to
# all of it at once (#1053). Until this line existed the aggregator's only
# request check was the SDK's Host/Origin guard, so any process on this box
# could POST a `tools/call` straight to `/mcp` and have it dispatched with none
# of the harness gates applied — the grant gate is a PreToolUse hook, plan mode
# only removes tools from `tools/list`, the effect ledger is fed from harness
# `_meta`, and the bench/eval sandbox is keyed on a session id an out-of-band
# caller simply omits. Wrapping the ASGI app is what makes the control
# un-forgettable: a new route is inside it on the day it is added, and the two
# mutating custom routes (`/changes/revert`, `/browser/navigate`) are covered by
# the same line as the MCP transport. `GET /health` is the sole open path —
# supervisord, `app/routers/health.py`, the promotion gate and the guardian all
# probe it with no credential, and refusing it would turn a healthy aggregator
# into a restart storm.
starlette_app = aggregator_auth.require_credential(starlette_app)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="127.0.0.1", port=PORT)
