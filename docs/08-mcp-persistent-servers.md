# MCP Persistent Servers: Migration from Stdio to SSE

> **Date**: 2026-04-04  
> **Status**: Proposal  
> **Prerequisite for**: Browser control (doc 07)

---

## 1. Problem Statement

Lloyd's MCP servers are currently spawned as **subprocesses by the Claude Agent SDK**. Each server is defined by `command` + `args` in `config.yaml`, and the SDK launches them when a `query()` call starts. This means:

- Servers have no state persistence — they die when the SDK session ends
- Every new session pays subprocess startup cost for all servers
- Servers that need long-lived resources (browser instances, database connections, caches) lose them between sessions
- No way to share a server instance across multiple concurrent agents (e.g., autonomy scheduler + user conversation)

The fix: run MCP servers as **persistent processes** that stay alive independently, with the SDK connecting to them as a client.

---

## 2. Current Architecture

### How it works today

```
config.yaml:
  mcp_servers:
    autonomy:
      command: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python
      args: ["/home/alansrobotlab/lloyd/mcp-servers/autonomy.py"]

server.py _get_mcp_servers() → {"autonomy": {"command": ..., "args": ...}}
    ↓
ClaudeCodeOptions(mcp_servers=MCP_SERVERS)
    ↓
SDK passes --mcp-config JSON to claude CLI
    ↓
CLI spawns each server as subprocess (stdio transport)
    ↓
Server dies when query/session ends
```

### Key files

| File | Role |
|------|------|
| `server.py:82-94` | `_get_mcp_servers()` — builds server config dict from config.yaml |
| `server.py:137-189` | `_discover_mcp_tools()` — spawns subprocess, sends JSON-RPC initialize + tools/list for tool discovery |
| `server.py:288-298` | `ClaudeCodeOptions(mcp_servers=MCP_SERVERS)` — passes to SDK |
| `server.py:2161-2209` | `/api/tools` endpoint — calls `_discover_mcp_tools()` per server |
| `config.yaml:55-79` | MCP server definitions (command + args) |
| `mcp-servers/*.py` | 8 servers, all using `mcp.server.stdio.stdio_server` |
| `autonomy.py:352-390` | Autonomy scheduler also uses `MCP_SERVERS` for its own SDK queries |

### The 8 servers

| Server | Port (proposed) | Notes |
|--------|----------------|-------|
| autonomy | 8501 | Task scheduling, reads ~/obsidian/autonomy/ |
| backlog | 8502 | Kanban boards, reads ~/obsidian/backlog/ |
| memory | 8503 | Knowledge graph, reads ~/obsidian/memory/ |
| mission_control | 8504 | Session management, reads ~/lloyd/sessions/ |
| subliminal | 8505 | Identity/context injection |
| http_tools | 8506 | Web search/fetch via DuckDuckGo + httpx |
| thunderbird | 8507 | Email/calendar proxy (spawns mcp-bridge.cjs) |
| pipeline | 8508 | Multi-stage worker dispatch |

---

## 3. The SDK Already Supports This

The Claude Code SDK (`claude_code_sdk` v0.0.25) defines three MCP server config types in `claude_code_sdk/types.py` (lines 152-187):

```python
class McpStdioServerConfig(TypedDict):
    """Current approach — SDK spawns subprocess."""
    type: NotRequired[Literal["stdio"]]
    command: str
    args: NotRequired[list[str]]

class McpSSEServerConfig(TypedDict):
    """Connect to running server via SSE."""
    type: Literal["sse"]
    url: str
    headers: NotRequired[dict[str, str]]

class McpHttpServerConfig(TypedDict):
    """Connect to running server via Streamable HTTP."""
    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]
```

So the SDK already knows how to connect to long-running servers. We just need to:
1. Make our servers speak SSE (or HTTP) instead of stdio
2. Update the config format
3. Update server.py to pass the new format and handle tool discovery differently

---

## 4. The MCP Package Has the Transports

The `mcp` package (v1.27.0) ships four server transports:

| Transport | Module | Use Case |
|-----------|--------|----------|
| stdio | `mcp.server.stdio` | Subprocess (current) |
| **SSE** | `mcp.server.sse` | **Long-running HTTP + SSE streaming** |
| Streamable HTTP | `mcp.server.streamable_http` | HTTP POST with optional SSE responses, session resumability |
| WebSocket | `mcp.server.websocket` | Bidirectional real-time |

**Recommendation: SSE.** It's the simplest, well-supported by the Claude CLI, and sufficient for our needs. Streamable HTTP adds session resumability but is more complex than we need.

---

## 5. What Changes Per Server

The tool logic stays **identical**. Only the entrypoint changes.

### Before (stdio)

```python
from mcp.server.stdio import stdio_server

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### After (SSE)

```python
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response
from mcp.server.sse import SseServerTransport

PORT = 8501  # Unique per server

sse = SseServerTransport("/messages/")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())
    return Response()

starlette_app = Starlette(routes=[
    Route("/sse", handle_sse, methods=["GET"]),
    Mount("/messages/", app=sse.handle_post_message),
])

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="127.0.0.1", port=PORT)
```

The `@app.list_tools()` and `@app.call_tool()` decorators are unchanged.

---

## 6. What Changes in config.yaml

### Before

```yaml
mcp_servers:
  autonomy:
    command: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python
    args: ["/home/alansrobotlab/lloyd/mcp-servers/autonomy.py"]
    enabled: true
    disabled_tools: []
```

### After

```yaml
mcp_servers:
  autonomy:
    type: sse
    url: http://127.0.0.1:8501/sse
    enabled: true
    disabled_tools: []
    # Keep command/args for process management (starting the server)
    command: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python
    args: ["/home/alansrobotlab/lloyd/mcp-servers/autonomy.py"]
```

The `command`/`args` fields are retained so Lloyd's backend can start servers if needed, but the SDK receives `{"type": "sse", "url": "..."}` instead.

---

## 7. What Changes in server.py

### 7.1 `_get_mcp_servers()` (lines 82-94)

Currently builds `{"command": ..., "args": ...}` dicts. Needs to output `{"type": "sse", "url": "..."}` for SSE servers:

```python
def _get_mcp_servers() -> dict[str, dict]:
    servers = {}
    for name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        server_type = cfg.get("type", "stdio")
        if server_type == "sse":
            servers[name] = {"type": "sse", "url": cfg["url"]}
        elif server_type == "http":
            servers[name] = {"type": "http", "url": cfg["url"]}
        else:
            servers[name] = {
                "command": cfg.get("command", "python"),
                "args": cfg.get("args", []),
            }
    return servers
```

### 7.2 `_discover_mcp_tools()` (lines 137-189)

Currently spawns a subprocess and sends JSON-RPC over stdio. For SSE servers, needs to use the MCP client SSE transport instead:

```python
from mcp.client.sse import sse_client
from mcp import ClientSession

async def _discover_mcp_tools_sse(server_name: str, url: str) -> list[dict]:
    """Discover tools from a running SSE MCP server."""
    async with sse_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {"name": t.name, "description": t.description or ""}
                for t in result.tools
            ]
```

The existing `_discover_mcp_tools()` (subprocess-based) remains as fallback for any stdio servers.

### 7.3 Server health/startup

Optional: on backend startup, check if each SSE server is reachable. If not, start it using the retained `command`/`args` from config. This could be a simple HTTP health check to the `/sse` endpoint.

---

## 8. Process Management

### Option A: Supervisord (recommended)

Add each MCP server to the supervisor config as a separate program. They run alongside `lloyd-backend` and `lloyd-frontend`:

```ini
[program:lloyd-mcp-autonomy]
command=/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python /home/alansrobotlab/lloyd/mcp-servers/autonomy.py
autostart=true
autorestart=true
stdout_logfile=/home/alansrobotlab/lloyd/logs/mcp-autonomy.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/mcp-autonomy.err

[program:lloyd-mcp-memory]
command=/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python /home/alansrobotlab/lloyd/mcp-servers/memory.py
autostart=true
autorestart=true
stdout_logfile=/home/alansrobotlab/lloyd/logs/mcp-memory.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/mcp-memory.err

# ... repeat for each server
```

Group them under `lloyd-mc` (or a new `lloyd-mcp` group) for easy management:
```bash
supervisorctl restart lloyd-mcp:*
```

### Option B: server.py manages startup

`server.py` spawns MCP servers as background subprocesses on startup and monitors them. Simpler config but couples lifecycle to the backend.

### Recommendation

**Option A (supervisord)** — keeps servers independent, allows restarting one without affecting others, consistent with existing infra.

---

## 9. Migration Strategy

### Phase 1: Dual-mode support

Update `server.py` to handle both stdio and SSE configs. Convert one server (e.g., `http_tools`) to SSE as a proof of concept. All others remain stdio.

**Changes**:
- `mcp-servers/http_tools.py` — add SSE entrypoint
- `config.yaml` — update http_tools entry to `type: sse`
- `server.py` — update `_get_mcp_servers()` and `_discover_mcp_tools()` to handle both types
- Supervisor config — add `lloyd-mcp-http-tools` program

**Validates**: SDK connects to running SSE server, tool discovery works, tools execute correctly.

### Phase 2: Convert all servers

Migrate remaining 7 servers to SSE. Each is a small change (swap entrypoint, assign port, add supervisor entry).

### Phase 3: Simplify

Once all servers are SSE:
- Remove subprocess-based `_discover_mcp_tools()` code
- Consider a shared SSE wrapper module to reduce boilerplate
- Add health check endpoint to each server
- Add startup dependency ordering if needed (e.g., memory before autonomy)

---

## 10. Thunderbird Special Case

`thunderbird.py` is already a proxy — it spawns `mcp-bridge.cjs` as a subprocess. Converting to SSE means the Thunderbird MCP server runs persistently and keeps its bridge subprocess alive. This is actually an improvement: the bridge connection persists instead of being re-established every session.

No special handling needed — the SSE transport wraps the existing proxy logic.

---

## 11. Dependencies

Already installed in the venv:
- `mcp` (v1.27.0) — includes `mcp.server.sse` and `mcp.client.sse`

Need to add:
- `uvicorn` — ASGI server for running SSE endpoints
- `starlette` — ASGI framework (lightweight, used by MCP's SSE transport)

```bash
.venvs/lloyd/bin/pip install uvicorn starlette
```

Both are lightweight and have no heavy transitive dependencies.

---

## 12. Risks

| Risk | Mitigation |
|------|------------|
| Port conflicts | Fixed port allocation (8501-8508), check on startup |
| Server crashes silently | supervisord `autorestart=true` + health checks from server.py |
| Multiple clients contend | SSE transport handles concurrent connections natively |
| Tool discovery during server startup | Retry with backoff in `_discover_mcp_tools_sse()` |
| Breaking autonomy.py | `autonomy.py` also uses `MCP_SERVERS` — same config change applies automatically |
| Backward compatibility | Phase 1 dual-mode support ensures stdio still works during migration |

---

## 13. Benefits

- **Browser MCP server** can launch Playwright and keep the browser alive indefinitely
- **Thunderbird bridge** stays connected instead of re-initializing every session
- **Memory server** can maintain in-memory caches across sessions
- **Multiple agents** (user conversation + autonomy scheduler) share the same server instances
- **Faster session startup** — no subprocess spawning, just SSE connection
- **Independent restarts** — restart one MCP server without affecting the backend or other servers
