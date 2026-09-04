#!/usr/bin/env python3
"""
Lloyd unified MCP server — every tool in one Server("lloyd") over SSE.

Run with:  python -m agent_mcp.main  (from ~/lloyd directory)
Endpoints: http://127.0.0.1:8500/sse     MCP transport
           http://127.0.0.1:8500/health  discovery/liveness JSON

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
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, Tool
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from agent_mcp import (
    _task_registry,
    annotations as tool_annotations,
    ambient,
    autonomy,
    autoresearch,
    backlog,
    browser,
    builtin_bash,
    builtin_fs,
    builtin_goal,
    builtin_plan,
    builtin_task,
    builtin_todo,
    discord_bot,
    facts,
    http_tools,
    ide,
    mission_control,
    mission_control_ui,
    session,
    skills,
    thunderbird,
    vault,
)

logger = logging.getLogger("lloyd-mcp")

PORT = 8500

# memory.py was split into facts/vault/session in #340 PR 5. The legacy
# memory module remains as a backward-compat re-export shim for callers
# (prefetch.py, app/post_capture.py) but is NOT in MODULES — including it
# would double-register every tool.
MODULES = [
    # Built-in tool replicas (formerly provided by claude-agent-sdk)
    builtin_bash,
    builtin_fs,
    builtin_goal,
    builtin_plan,
    builtin_task,
    builtin_todo,
    # Domain modules
    ambient,
    autonomy,
    autoresearch,
    backlog,
    browser,
    discord_bot,
    facts,
    vault,
    session,
    mission_control,
    mission_control_ui,
    ide,
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


combined = Server("lloyd")

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

# OpenAI's spec caps tool names at 64 chars. Enforced here at registration
# so a bad name fails loudly on the first list_tools() instead of
# mid-conversation in the harness translator (tool_schema.py keeps its own
# check as a backstop for non-aggregator servers).
TOOL_NAME_MAX = 64


@combined.list_tools()
async def list_tools():
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


def _bound_session_id(arguments: dict) -> str:
    """Resolve the harness session id for the in-flight tool call.

    Preferred source is the request's `_meta` — the field the MCP spec
    reserves for implementation metadata, and where the 2026-07-28 spec
    puts all per-request context. Falls back to the legacy `_session_id`
    argument so a harness and aggregator at different versions still
    correlate. `_meta` is strictly better than the argument form: the SDK
    validates `arguments` against the tool's inputSchema *before* the
    handler runs, so an injected argument is validated as if it were a
    real parameter and would be rejected outright by any schema setting
    `additionalProperties: false`.
    """
    try:
        meta = combined.request_context.meta
    except LookupError:
        meta = None  # called outside a request (tests, direct dispatch)
    if meta is not None:
        sid = getattr(meta, META_SESSION_ID, None)
        if isinstance(sid, str) and sid:
            return sid
    if isinstance(arguments, dict):
        return arguments.get("_session_id", "") or ""
    return ""


@combined.call_tool()
async def call_tool(name: str, arguments: dict):
    if not _dispatch:
        await list_tools()
    mod = _dispatch.get(name)
    if not mod:
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))],
            isError=True,
        )

    sid = _bound_session_id(arguments)
    # Strip the legacy argument form before the per-tool handler validates,
    # so module schemas never have to advertise an internal field.
    if isinstance(arguments, dict) and "_session_id" in arguments:
        arguments = {k: v for k, v in arguments.items() if k != "_session_id"}

    token = _task_registry.current_session_id.set(sid)
    try:
        return await mod.call_tool(name, arguments)
    finally:
        _task_registry.current_session_id.reset(token)


# DNS-rebinding protection. The aggregator binds loopback with no auth, so
# the only thing standing between a page in the user's browser and this
# tool surface is that a cross-origin POST to /messages/ needs a preflight
# it won't get. That's an accident of content-type rules, not a control —
# the SDK ships the actual control, so use it.
_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[f"127.0.0.1:{PORT}", f"localhost:{PORT}"],
    allowed_origins=[f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"],
)

transport = SseServerTransport("/messages/", security_settings=_security)


async def handle_sse(request):
    async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await combined.run(streams[0], streams[1], combined.create_initialization_options())
    return Response()


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


@asynccontextmanager
async def lifespan(app):
    await discord_bot.start_bot_task()
    try:
        yield
    finally:
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


starlette_app = Starlette(
    routes=[
        Route("/sse", handle_sse, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Mount("/messages/", app=transport.handle_post_message),
    ],
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="127.0.0.1", port=PORT)
