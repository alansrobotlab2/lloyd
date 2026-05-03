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
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from app.harness.errors import ToolDispatchError

logger = logging.getLogger("lloyd-harness-mcp-pool")

# Default URL for the unified lloyd-mcp aggregator (agent_mcp/main.py).
DEFAULT_LLOYD_MCP_URL = "http://127.0.0.1:8500/sse"


class MCPPool:
    """One-process pool that holds open MCP client sessions keyed by
    server name. `aclose()` tears them all down.

    The pool is server-config-aware: pass the same shape that
    `app.mcp_discovery._get_mcp_servers()` returns — a dict of
    {server_name: {"type": "sse"|"stdio", "url"|"command"|"args": ...}}.

    Built-in tools (Bash/Read/Edit/...) advertise to the model under
    bare names (no `mcp__server__` prefix). The pool resolves them by
    asking each server for its tools/list and building a name → server
    map. On dispatch, bare names route to whichever server claims them.
    """

    def __init__(self, server_configs: dict[str, dict[str, Any]]):
        self._configs = server_configs
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tool_routes: dict[str, str] = {}  # bare_name → server_name
        self._schemas: dict[str, dict[str, Any]] = {}  # bare_name → inputSchema
        self._discovered: list[tuple[str, list[dict[str, Any]]]] = []
        self._opened = False
        self._open_lock = asyncio.Lock()

    async def open(self) -> None:
        """Open sessions to every configured server and discover tools.

        Idempotent — concurrent callers wait on the lock and see the
        already-opened pool.
        """
        async with self._open_lock:
            if self._opened:
                return
            for server_name, cfg in self._configs.items():
                try:
                    session = await self._open_session(server_name, cfg)
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
                        # First server wins; log the collision.
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
            self._opened = True

    async def aclose(self) -> None:
        """Close every open session."""
        await self._exit_stack.aclose()
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

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call to the right server.

        `name` may be either a bare tool name (`"Bash"`) or namespaced
        (`"mcp__lloyd-mcp__memory_add"`). Returns
        `{"content": str, "is_error": bool}`. Raises ToolDispatchError
        when routing or transport fails — caller maps to a tool_result
        with `is_error=True`.
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
        try:
            result = await session.call_tool(bare, coerced)
        except Exception as exc:
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
        self, server_name: str, cfg: dict[str, Any]
    ) -> ClientSession:
        server_type = cfg.get("type", "stdio")
        if server_type in ("sse", "http"):
            url = cfg["url"]
            ctx = sse_client(url)
            read_stream, write_stream = await self._exit_stack.enter_async_context(ctx)
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            return session
        # stdio fallback would land here; we don't currently need it,
        # so fail loudly so it gets implemented when we do.
        raise NotImplementedError(
            f"mcp_pool: stdio transport not yet wired ({server_name})"
        )

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
    """
    key = _config_key(server_configs)
    async with _POOL_CACHE_LOCK:
        pool = _POOL_CACHE.get(key)
        if pool is not None:
            return pool
        pool = MCPPool(server_configs)
        await pool.open()
        _POOL_CACHE[key] = pool
        return pool


async def close_all_pools() -> None:
    """Close every cached pool. Call from FastAPI shutdown."""
    async with _POOL_CACHE_LOCK:
        for pool in list(_POOL_CACHE.values()):
            try:
                await pool.aclose()
            except Exception as exc:
                logger.warning("close_all_pools: %s", exc)
        _POOL_CACHE.clear()
