"""Persistent MCP client for the harness.

The lloyd-mcp aggregator at http://127.0.0.1:8500/sse owns every MCP
tool — built-ins (Bash, Read, Edit, Write, Grep, Glob, Task) plus the
existing domain modules. The harness `loop.py` reuses one process-wide
pool keyed by mcp_servers config (see `get_or_open_pool`); cleanup is
handled at FastAPI shutdown via `lifecycle.shutdown_cleanup`.

Only SSE / HTTP transports are implemented today — stdio is not wired
because every active config uses the consolidated aggregator. Adding
stdio means importing `mcp.client.stdio.stdio_client` and branching in
`_open_session`; nobody needs it yet, so it raises NotImplementedError.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from mcp import ClientSession, McpError
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.config import service_url
from app.harness.errors import ToolDispatchError

logger = logging.getLogger("lloyd-harness-mcp-pool")

# Default URL for the unified lloyd-mcp aggregator (agent_mcp/main.py).
DEFAULT_LLOYD_MCP_URL = service_url("lloyd_mcp", "http://127.0.0.1:8500/sse")

# `_meta` key carrying the harness session id. Must match
# agent_mcp.main.META_SESSION_ID.
META_SESSION_ID = "lloyd/session_id"

# Ceiling on a single tools/call round trip. Sits above the Bash tool's own
# 600s hard cap so a legitimately long command finishes on its own terms and
# this only fires when something is genuinely wedged. Without it a hung tool
# blocks the harness indefinitely — there is no default in the MCP client.
CALL_TIMEOUT_SECONDS = 660.0


class MCPPool:
    """One-process pool that holds open MCP client sessions keyed by
    server name. `aclose()` tears them all down.

    The pool is server-config-aware: pass the same shape that
    `app.mcp_discovery._get_mcp_servers()` returns — a dict of
    {server_name: {"type": "sse"|"stdio", "url"|"command"|"args": ...}}.

    All tools advertise to the model under bare MCP names. The pool
    resolves them by asking each server for its tools/list and building
    a bare-name → server map. On dispatch, bare names route to whichever
    server claims them.
    """

    def __init__(self, server_configs: dict[str, dict[str, Any]]):
        self._configs = server_configs
        self._sessions: dict[str, ClientSession] = {}
        self._tool_routes: dict[str, str] = {}  # bare_name → server_name
        self._schemas: dict[str, dict[str, Any]] = {}  # bare_name → inputSchema
        self._discovered: list[tuple[str, list[dict[str, Any]]]] = []
        self._opened = False
        self._open_lock = asyncio.Lock()
        # Owner-task pattern: a single dedicated task holds the AsyncExitStack
        # for the SSE clients and ClientSessions. All cleanup happens in that
        # same task, avoiding anyio's "cancel scope exited in a different task"
        # error that previously left the task group spinning in
        # _deliver_cancellation at 100 % CPU.
        self._owner_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._opened_event = asyncio.Event()
        self._open_error: BaseException | None = None
        self._poisoned = False

    async def open(self) -> None:
        """Open sessions to every configured server and discover tools.

        Idempotent — concurrent callers wait on the lock and see the
        already-opened pool.
        """
        async with self._open_lock:
            if self._opened:
                return
            self._owner_task = asyncio.create_task(
                self._owner_loop(), name="mcp_pool_owner"
            )
            await self._opened_event.wait()
            if self._open_error is not None:
                # Owner task already exited; surface the error and reset.
                err = self._open_error
                self._open_error = None
                raise err
            self._opened = True

    async def _owner_loop(self) -> None:
        """Hold every SSE/ClientSession context for the pool's lifetime.

        Opens all configured servers under one AsyncExitStack, signals the
        opener via `_opened_event`, then parks on `_shutdown_event`. When
        shutdown fires (or any context raises), the AsyncExitStack unwinds
        in this same task — keeping anyio's cancel scopes consistent.
        """
        try:
            async with AsyncExitStack() as stack:
                for server_name, cfg in self._configs.items():
                    try:
                        session = await self._open_session(stack, server_name, cfg)
                    except Exception as exc:
                        logger.warning(
                            "mcp_pool: failed to open %s: %s", server_name, exc
                        )
                        continue
                    self._sessions[server_name] = session
                    tools = await self._list_tools(server_name, session)
                    self._discovered.append((server_name, tools))
                    for tool in tools:
                        bare = tool["name"]
                        if bare in self._tool_routes:
                            logger.warning(
                                "mcp_pool: tool name collision on %r — %s wins over %s",
                                bare,
                                self._tool_routes[bare],
                                server_name,
                            )
                            continue
                        self._tool_routes[bare] = server_name
                        schema = tool.get("inputSchema")
                        if isinstance(schema, dict):
                            self._schemas[bare] = schema
                self._opened_event.set()
                await self._shutdown_event.wait()
        except BaseException as exc:
            # Open failed, or a child task in one of the SSE task groups
            # propagated. Surface to whoever's awaiting open(), then let
            # the AsyncExitStack finish unwinding in this task.
            self._open_error = exc
            self._opened_event.set()
            self._poisoned = True
            if not isinstance(exc, Exception):
                raise

    async def aclose(self) -> None:
        """Signal the owner task to tear down, then await it.

        Cleanup runs in the owner task — never in the caller's task — so
        anyio cancel scopes always exit in the task that entered them.
        """
        self._shutdown_event.set()
        if self._owner_task is not None:
            try:
                await self._owner_task
            except Exception as exc:
                logger.warning("mcp_pool: owner task exited with %s", exc)
            self._owner_task = None
        self._sessions.clear()
        self._tool_routes.clear()
        self._schemas.clear()
        self._discovered = []
        self._opened = False

    @property
    def discovered(self) -> list[tuple[str, list[dict[str, Any]]]]:
        """List of (server_name, mcp_tools_list) pairs ready for
        `app.harness.tool_schema.build_tool_list`.
        """
        return self._discovered

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str = "",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Dispatch a tool call to the right server.

        `name` is normally a bare tool name (`"Bash"`, `"email_recent"`).
        The legacy ``mcp__server__tool`` form is still accepted so old
        persisted session JSON replays cleanly. Returns
        ``{"content": str, "is_error": bool}``. Raises ToolDispatchError
        when routing or transport fails — caller maps to a tool_result
        with ``is_error=True``.

        `session_id` travels in the request's ``_meta``, not in ``args``.
        The MCP server validates ``args`` against the tool's inputSchema
        before its handler runs, so anything injected there is validated
        as a real parameter; ``_meta`` is the field the spec reserves for
        exactly this kind of implementation metadata.
        """
        if not self._opened:
            await self.open()

        # Resolve to (server_name, bare_tool_name).
        if name.startswith("mcp__"):
            rest = name[len("mcp__"):]
            sep = rest.find("__")
            if sep > 0:
                server_name = rest[:sep]
                bare = rest[sep + 2:]
            else:
                raise ToolDispatchError(name, "malformed namespaced tool name")
        else:
            bare = name
            server_name = self._tool_routes.get(bare, "")
            if not server_name:
                raise ToolDispatchError(
                    name, f"no server claims tool {bare!r}"
                )

        session = self._sessions.get(server_name)
        if session is None:
            raise ToolDispatchError(name, f"server {server_name!r} not open")

        coerced = _coerce_args(args, self._schemas.get(bare))
        meta = {META_SESSION_ID: session_id} if session_id else None
        budget = timeout_seconds if timeout_seconds is not None else CALL_TIMEOUT_SECONDS
        try:
            result = await session.call_tool(
                bare,
                coerced,
                read_timeout_seconds=timedelta(seconds=budget),
                meta=meta,
            )
        except McpError as exc:
            # A protocol-level error from a live session: the server was
            # reachable and answered. That is a failed *call*, not a failed
            # transport — surfacing it as a tool error keeps the session
            # (shared with every other in-flight turn) intact.
            logger.info("mcp_pool: %s on %s returned an MCP error: %s", bare, server_name, exc)
            return {"content": f"MCP error calling {bare}: {exc}", "is_error": True}
        except Exception as exc:
            # Transport-shaped failure. Mark poisoned, evict from cache,
            # and signal the owner task to tear the pool down in its own
            # context. The owner task closes the AsyncExitStack — that's
            # the task that entered the cancel scopes, so anyio is happy.
            #
            # This is deliberately blunt: the session is shared, so a dead
            # transport has to take it down for everyone. The 2026-07-28
            # stateless protocol removes the shared session and with it
            # this whole failure mode; until then, a timeout is what stops
            # one wedged tool from hanging every concurrent turn forever.
            self._poisoned = True
            _evict_pool(self)
            self._shutdown_event.set()
            logger.warning(
                "mcp_pool: %s on %s failed (%s); pool evicted from cache",
                bare, server_name, exc,
            )
            raise ToolDispatchError(name, f"transport error: {exc}") from exc

        # MCP CallToolResult.content is a list of TextContent /
        # ImageContent / EmbeddedResource. We flatten to text — the
        # built-in tools return only TextContent.
        text_parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                text_parts.append(text)
            else:
                text_parts.append(json.dumps({"type": getattr(item, "type", "?")}))
        is_error = bool(getattr(result, "isError", False))
        return {"content": "".join(text_parts), "is_error": is_error}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _open_session(
        self,
        stack: AsyncExitStack,
        server_name: str,
        cfg: dict[str, Any],
    ) -> ClientSession:
        """Enter the SSE client + ClientSession contexts on the supplied
        ``stack``. The stack belongs to the owner task, so cleanup runs in
        the same task that entered the contexts.
        """
        server_type = cfg.get("type", "stdio")
        if server_type in ("sse", "http"):
            url = cfg["url"]
            ctx = sse_client(url)
        elif server_type == "stdio":
            command = cfg.get("command")
            if not command:
                raise ValueError(f"stdio server {server_name!r} has no 'command'")
            env = cfg.get("env")
            ctx = stdio_client(
                StdioServerParameters(
                    command=command,
                    args=list(cfg.get("args") or []),
                    env=dict(env) if env else None,
                    cwd=cfg.get("cwd") or None,
                )
            )
        else:
            raise ValueError(
                f"mcp_pool: unknown transport {server_type!r} for {server_name!r}"
            )

        read_stream, write_stream = await stack.enter_async_context(ctx)
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        return session

    async def _list_tools(
        self, server_name: str, session: ClientSession
    ) -> list[dict[str, Any]]:
        result = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------


def _coerce_args(args: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce primitive args to match the tool's inputSchema.

    vLLM occasionally emits string-shaped scalars (`"5"` for an integer
    field, `"true"` for a boolean) and MCP's strict jsonschema validator
    rejects them. We coerce best-effort using the declared `type` of each
    top-level property; anything ambiguous is passed through untouched
    so the validator can still surface real errors.
    """
    if not isinstance(args, dict) or not isinstance(schema, dict):
        return args
    props = schema.get("properties")
    if not isinstance(props, dict):
        return args
    out = dict(args)
    for k, v in args.items():
        prop = props.get(k)
        if not isinstance(prop, dict):
            continue
        target = prop.get("type")
        if isinstance(target, list):
            target = next((t for t in target if t != "null"), None)
        out[k] = _coerce_one(v, target)
    return out


def _coerce_one(v: Any, target: Any) -> Any:
    if v is None or target is None:
        return v
    if target == "integer":
        if isinstance(v, bool) or isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s.lstrip("-").isdigit():
                try:
                    return int(s)
                except ValueError:
                    return v
        return v
    if target == "number":
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                return v
        return v
    if target == "boolean":
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "yes"):
                return True
            if s in ("false", "0", "no", ""):
                return False
        if isinstance(v, int):
            return bool(v)
        return v
    if target == "string":
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        return v
    return v


# ---------------------------------------------------------------------------
# Process-wide pool cache
# ---------------------------------------------------------------------------

_POOL_CACHE: dict[str, MCPPool] = {}
_POOL_CACHE_LOCK = asyncio.Lock()


def _config_key(server_configs: dict[str, dict[str, Any]]) -> str:
    """Stable hashable key for a server-config dict."""
    return json.dumps(server_configs, sort_keys=True, default=str)


async def get_or_open_pool(server_configs: dict[str, dict[str, Any]]) -> MCPPool:
    """Return a process-wide MCPPool for `server_configs`, opening on first use.

    Concurrent callers serialize on a lock so the SSE handshake +
    `tools/list` only runs once per unique config. Subsequent turns reuse
    the open sessions.

    Pools that hit transport failures are marked `_poisoned` and evicted
    via `_evict_pool` from `call_tool`. If a poisoned pool is somehow
    still in the cache when a new caller arrives (race: evicted but the
    same instance got re-cached), we treat it as missing and rebuild.
    """
    key = _config_key(server_configs)
    async with _POOL_CACHE_LOCK:
        pool = _POOL_CACHE.get(key)
        if pool is not None and not pool._poisoned:
            return pool
        pool = MCPPool(server_configs)
        await pool.open()
        _POOL_CACHE[key] = pool
        return pool


def _evict_pool(pool: MCPPool) -> None:
    """Drop a poisoned pool from the cache.

    Synchronous best-effort: drop the cache entry only. Closing the pool
    is handled by the owner task once `_shutdown_event` is set (see
    ``MCPPool.call_tool``); we never await across tasks here, so we don't
    need to do any async work in this function.

    Two concurrent evictions race harmlessly: dict ops are atomic under
    the GIL, and the second pop sees the entry already gone.
    """
    for k, p in list(_POOL_CACHE.items()):
        if p is pool:
            _POOL_CACHE.pop(k, None)
            return


async def close_all_pools() -> None:
    """Close every cached pool. Call from FastAPI shutdown."""
    async with _POOL_CACHE_LOCK:
        for pool in list(_POOL_CACHE.values()):
            try:
                await pool.aclose()
            except Exception as exc:
                logger.warning("close_all_pools: %s", exc)
        _POOL_CACHE.clear()
