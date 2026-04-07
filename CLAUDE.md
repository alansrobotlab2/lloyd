# Lloyd — Claude Code Context

## Project Overview

Lloyd is a clean-slate AI agent built on the **Claude Agent SDK** (`claude-code-sdk`). It replaces the legacy `hermes-agent` system. The backend is FastAPI + SSE, the frontend is React (Vite), and all custom tools are exposed as MCP servers.

- **Backend**: `server.py` (FastAPI, port 8080)
- **Frontend**: `web/` (Vite dev server, proxied through backend)
- **Config**: `config.yaml`
- **MCP servers**: `mcp-servers/*.py`
- **Venv**: `.venvs/lloyd/bin/python`

## Service Management

Lloyd runs under **supervisord inside the `lloyd` distrobox container**. We are already inside the container — use supervisorctl directly:

```bash
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/agent-services/supervisor/supervisord.conf restart lloyd-mc:lloyd-backend
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/agent-services/supervisor/supervisord.conf restart lloyd-mc:lloyd-frontend
/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/agent-services/supervisor/supervisord.conf status
```

The process group is `lloyd-mc`, not `lloyd-backend` bare. Always use `lloyd-mc:lloyd-backend` and `lloyd-mc:lloyd-frontend`.

After editing `server.py`, restart `lloyd-mc:lloyd-backend` for changes to take effect.  
After editing frontend files, Vite HMR usually picks up changes automatically (no restart needed).

## Architecture

```
~/lloyd/
├── server.py            # FastAPI backend — all API endpoints + SSE bridge
├── config.yaml          # Model configs, MCP server list, agent settings
├── prompt_builder.py    # System prompt assembly (SOUL.md + memories + skills)
├── autonomy.py          # Task scheduler
├── usage_store.py       # SQLite usage tracking
│
├── mcp-servers/         # Custom tools as MCP servers (stdio JSON-RPC)
│   ├── autonomy.py
│   ├── backlog.py
│   ├── memory.py
│   ├── mission_control.py
│   ├── subliminal.py
│   ├── http_tools.py
│   ├── thunderbird.py   # Wraps mcp-bridge.cjs → Thunderbird HTTP extension
│   └── pipeline.py
│
├── web/src/
│   ├── api.ts           # All API calls + TypeScript types
│   └── components/pages/
│       └── ToolsPage.tsx
│
├── .venvs/lloyd/        # Python venv (use this python for all lloyd scripts)
└── logs/                # server.log, server.err, frontend.log
```

## Thunderbird MCP Server

The `mcp-servers/thunderbird.py` server is a proxy — it spawns `~/agent-services/services/thunderbird-mcp/mcp-bridge.cjs` as a subprocess, which forwards JSON-RPC over HTTP to the Thunderbird extension at `localhost:8765`.

**Known bug (fixed)**: The `_ensure_bridge()` function previously used `time.sleep(1)` after sending the MCP `initialize` message. During that sleep, the bridge wrote the init response to stdout. The subsequent `_discover_tools()` call would read that stale init response instead of the `tools/list` response, causing zero tools to be discovered. The fix: call `_bridge_receive()` inside `_ensure_bridge()` to consume the init response immediately.

## Tools Tab

`/api/tools` discovers tools from each MCP server by spawning it as a subprocess, sending `initialize` + `tools/list` JSON-RPC, and parsing the response. Results are cached for 5 minutes (`_tools_cache`, TTL=300s).

Tool enable/disable state is stored in `config.yaml`:
- Server-level: `mcp_servers.<name>.enabled: false`
- Tool-level: `mcp_servers.<name>.disabled_tools: [tool_name, ...]`
- Built-in: `tools.disabled_builtin: [ToolName, ...]`

Disabled tools are enforced via the SDK's `disallowed_tools` option (format: `mcp__<server>__<tool>` for MCP tools, plain name for built-ins).

## Config Structure (config.yaml)

```yaml
model:
  default: Qwen3.5-122B-A10B

models:
  Qwen3.5-122B-A10B:
    alias: 122b
    env:
      ANTHROPIC_BASE_URL: "http://127.0.0.1:8096"
      ANTHROPIC_API_KEY: "no-key-required"
      ANTHROPIC_CUSTOM_MODEL_OPTION: "Qwen3.5-122B-A10B"
      ANTHROPIC_CUSTOM_MODEL_OPTION_NAME: "Qwen 122B"

mcp_servers:
  thunderbird:
    command: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python
    args: ["/home/alansrobotlab/lloyd/mcp-servers/thunderbird.py"]
    enabled: true          # optional, defaults to true
    disabled_tools: []     # optional, list of tool names to block

tools:
  disabled_builtin: []     # Claude built-in tools to block (Bash, Read, Write, etc.)

agent:
  max_turns: 60
  permission_mode: bypassPermissions
```

## Development Notes

- The SDK's `query()` call spawns a `claude` CLI subprocess per session. MCP servers are passed via `ClaudeCodeOptions.mcp_servers`.
- Local models (Qwen) are selected by passing env vars: `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_MODEL_OPTION`, etc.
- Session continuity: the SDK session ID from `SystemMessage` is stored in `sessions/<id>.json` as `sdk_session_id`, then passed as `resume=` on subsequent turns.
- The `/api/message/stream` endpoint uses SSE. The frontend connects via `fetch` + `ReadableStream`, not `EventSource`, to allow `POST`.
