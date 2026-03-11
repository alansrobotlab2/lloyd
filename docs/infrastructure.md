---
tags:
  - lloyd
  - architecture
  - infrastructure
type: reference
segment: projects
---

# Infrastructure

## Runtime Environment

| Property | Value |
|----------|-------|
| Host OS | Arch Linux (kernel 6.18.9) |
| Container | Distrobox container named `lloyd` |
| GPU | NVIDIA (CUDA 12, used by [[voice-pipeline]] ONNX Runtime) |

## Systemd Services (User)

| Service | Process | Port | Description |
|---------|---------|------|-------------|
| `openclaw-gateway.service` | OpenClaw gateway (Node.js) | 18789 | Main gateway, API, WebSocket, serves Mission Control |
| `lloyd-tool-mcp.service` | tool_services.py (Python) | 8093 | [[mcp-tools|MCP tools server]] (27 tools) |
| `lloyd-voice-mcp.service` | voice_services.py (Python) | 8094 | [[voice-pipeline|Voice MCP server]] |

**Known issue:** `lloyd-tool-mcp.service` needs `KillMode=control-group` to avoid stale PIDs on restart.

## Other Services

| Service | Port | Notes |
|---------|------|-------|
| Voice TUI (voice_mode.py) | 8092 | Manual start (not yet a systemd service) |
| Local vLLM | 8091 | Qwen3.5-35B-A3B |
| PostgreSQL | -- | System service (used by legacy Backlog Rails app, may be deprecated) |

## Port Map

| Port | Service |
|------|---------|
| 8091 | vLLM (local LLM) |
| 8092 | Voice TUI HTTP API |
| 8093 | MCP Tools Server (SSE) |
| 8094 | Voice MCP Server (SSE) |
| 18789 | OpenClaw Gateway |

## Cron System

OpenClaw has a built-in cron system (`openclaw cron`). Jobs run as isolated sessions with configurable model, timeout, and timezone.

### Active Crons

| Job | ID | Schedule | Agent | Model | Timeout |
|-----|----|----------|-------|-------|---------|
| `periodic-memory-capture` | `06a1d2b2` | Every 15m | memory | local-llm | 120s |
| `nightly-reflection` | `9de0e564` | Daily 2am PST | memory | Opus | 300s |

### Disabled Crons

| Job | ID | Notes |
|-----|----|-------|
| `nightly-memory-consolidation` | `f53ad621` | Replaced by nightly reflection |

See [[memory-system]] for details on what these crons do.

## Extension System

Plugins live at `~/.openclaw/extensions/`.

### Active Extensions

| Extension | Purpose |
|-----------|---------|
| `mcp-tools` | [[mcp-tools|27 MCP tools + prefill + mode switching]] |
| `voice-tools` | [[voice-pipeline|Voice tools + TTS hook]] |
| `mission-control` | React dashboard |
| `timing-profiler` | Performance profiling |

### Disabled Extensions

| Extension | Notes |
|-----------|-------|
| `web-local.disabled` | Superseded by mcp-tools |
| `memory-graph.disabled` | Superseded by mcp-tools |
| `file-tools.disabled` | Superseded by mcp-tools |
| `run-bash.disabled` | Superseded by mcp-tools |

### Extension Structure

Each extension contains:
- `openclaw.plugin.json` — metadata
- `index.ts` — implementation

### Plugin API

`OpenClawPluginApi` provides:
- `registerTool` — expose tools to agents
- `registerCommand` — slash commands
- `on(hook)` — lifecycle hooks
- `logger` — structured logging

### Available Hooks

| Hook | Trigger |
|------|---------|
| `before_prompt_build` | Before system prompt assembly |
| `before_tool_call` | Before tool execution |
| `message_sending` | Before response delivery |
| `session_end` | Session cleanup |
| `subagent_delivery_target` | Subagent result routing |

## Mission Control

React dashboard served by the [[index|OpenClaw Gateway]].

| Property | Value |
|----------|-------|
| Source | `~/.openclaw/extensions/mission-control/web/` |
| Route | `/mc/` |
| Stack | Vite + React + Tailwind |
| Build | `npm run build` in `web/` directory |
| Backend | `index.ts` -- REST API at `/api/mc/*`, WebSocket connections, diagnostic event tracking |
| Chat panel | `web/src/components/ChatPanel.tsx` |

**Features:** Chat panel, token usage stats, API monitoring, services tab.

Code changes require: compile + gateway restart.

## Ops Reference

Full startup and troubleshooting guide: `~/Projects/lloyd/docs/OPS.md`

## Related Docs

- [[index]] — High-Level Architecture
- [[mcp-tools]] — MCP Tools Server
- [[voice-pipeline]] — Voice Pipeline
- [[memory-system]] — Memory System (cron jobs)
- [[backlog]] — Backlog System
