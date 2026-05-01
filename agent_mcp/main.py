#!/usr/bin/env python3
"""
Lloyd unified MCP server — all 51 tools in one Server("lloyd") over SSE.

Run with:  python -m agent_mcp.main  (from ~/lloyd directory)
Endpoint:  http://127.0.0.1:8500/sse
"""

import json
from contextlib import asynccontextmanager

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route

from agent_mcp import (
    ambient,
    autonomy,
    autoresearch,
    backlog,
    browser,
    discord_bot,
    facts,
    http_tools,
    mission_control,
    pipeline,
    session,
    skills,
    subliminal,
    thunderbird,
    vault,
)

PORT = 8500

# memory.py was split into facts/vault/session in #340 PR 5. The legacy
# memory module remains as a backward-compat re-export shim for callers
# (prefetch.py, app/post_capture.py) but is NOT in MODULES — including it
# would double-register every tool.
MODULES = [
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
    skills,
    subliminal,
    http_tools,
    thunderbird,
    pipeline,
]

combined = Server("lloyd")

# tool_name -> module, built on first list_tools call
_dispatch: dict[str, object] = {}


@combined.list_tools()
async def list_tools():
    _dispatch.clear()
    all_tools = []
    for mod in MODULES:
        tools = await mod.list_tools()
        for tool in tools:
            _dispatch[tool.name] = mod
        all_tools.extend(tools)
    return all_tools


@combined.call_tool()
async def call_tool(name: str, arguments: dict):
    if not _dispatch:
        await list_tools()
    mod = _dispatch.get(name)
    if mod:
        return await mod.call_tool(name, arguments)
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


transport = SseServerTransport("/messages/")


async def handle_sse(request):
    async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await combined.run(streams[0], streams[1], combined.create_initialization_options())
    return Response()


@asynccontextmanager
async def lifespan(app):
    await discord_bot.start_bot_task()
    yield
    await discord_bot.stop_bot()


starlette_app = Starlette(
    routes=[
        Route("/sse", handle_sse, methods=["GET"]),
        Mount("/messages/", app=transport.handle_post_message),
    ],
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="127.0.0.1", port=PORT)
