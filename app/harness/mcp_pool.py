"""Persistent MCP client for the harness.

The lloyd-mcp aggregator at http://127.0.0.1:8500/sse owns every MCP
tool — built-ins (Bash, Read, Edit, Write, Grep, Glob, Task) plus the
existing 14 modules. We hold one long-lived SSE session for the lifetime
of the harness process; per-turn dispatchers borrow it.

Stdio fallback exists for any future external MCP server declared in
config.yaml (none today — the consolidated aggregator is the only entry).
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
            self._opened = True

    async def aclose(self) -> None:
        """Close every open session."""
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tool_routes.clear()
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

        try:
            result = await session.call_tool(bare, args)
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
