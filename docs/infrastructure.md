---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Infrastructure

## Runtime Environment

| Property | Value |
|----------|-------|
| Host OS | Arch Linux |
| Container | Distrobox container named `lloyd-services` |
| Model | Opus 4.6 |
| Config | `~/.openclaw/openclaw.json` |
| Gateway | Local + LAN, TLS, token auth |

## GPUs

| GPU | Hardware | VRAM | Assignment |
|-----|----------|------|------------|
| GPU0 | RTX 5090 | 32 GB | Local LLM via llama-server (Qwen3.5-35B-A3B) |
| GPU1 | RTX 3090 | 24 GB | TTS + voice pipeline |
| GPU2 | RTX 3090 | 24 GB | QMD search + OpenClaw gateway |

## Systemd Services (User)

Three systemd user services run on the host (outside the distrobox container).

| Service | Process | Description |
|---------|---------|-------------|
| `lloyd-llm.service` | llama-server | Local LLM -- Qwen3.5-35B-A3B on GPU0 |
| `lloyd-qmd-daemon.service` | QMD Search | HTTP MCP search endpoint on port 8181 |
| `lloyd-qmd-watcher.service` | QMD Vault Watcher | Auto-indexes vault changes for QMD |

## Supervisord Services (Distrobox `lloyd-services`)

Five services managed by supervisord inside the distrobox container.

| Service | Command | Port | Autostart |
|---------|---------|------|-----------|
| `lloyd-tool-mcp` | `uv run tool_services.py --transport sse --port 8093` | 8093 | Yes |
| `lloyd-tts` | `bin/start-qwen3-tts.sh` (CUDA device 1) | 8090 | Yes |
| `lloyd-voice-mcp` | `uv run voice_services.py --transport sse --port 8094` | 8094 | Yes |
| `lloyd-voice-mode` | `uv run python voice_mode.py --headless` | 8092 | Yes |
| `openclaw-gateway` | `bin/start-gateway.sh` (CUDA device 2) | 18790 | No (manual) |

## Port Map

| Port | Service | Protocol |
|------|---------|----------|
| 8090 | Qwen3-TTS | HTTP |
| 8091 | Local LLM (llama-server) | HTTP |
| 8092 | Voice Mode API (voice_mode.py) | HTTP |
| 8093 | Tool Services MCP (tool_services.py) | SSE |
| 8094 | Voice MCP (voice_services.py) | SSE |
| 8181 | QMD Search | HTTP |
| 18790 | OpenClaw Gateway + Mission Control | HTTPS |

## Nightly Crons

OpenClaw has a built-in cron system (`openclaw cron`). Jobs run as isolated sessions with configurable model, timeout, and timezone.

| Job | Schedule | Description |
|-----|----------|-------------|
| Vault maintenance | 2:00 AM PST | Nightly vault cleanup and organization |
| Skills management | 3:00 AM PST | Skill review, updates, and maintenance |
| Reflection + self-improvement | 4:00 AM PST | Nightly reflection and self-improvement loop |

## Extension System

Plugins live at `~/.openclaw/extensions/`.

### Active Extensions

| Extension | Purpose |
|-----------|---------|
| `mcp-tools` | [[tools|26 MCP tools + prefill + mode switching]] |
| `voice-tools` | [[voice|Voice tools + TTS hook]] |
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
- `openclaw.plugin.json` -- metadata
- `index.ts` -- implementation

### Plugin API

`OpenClawPluginApi` provides:
- `registerTool` -- expose tools to agents
- `registerCommand` -- slash commands
- `on(hook)` -- lifecycle hooks
- `logger` -- structured logging

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

## Configuration

- **Main config:** `~/.openclaw/openclaw.json`
- **Gateway:** Serves local and LAN connections over TLS with token authentication
- **Tool allowlists:** Per-agent tool access defined in `openclaw.json`
- **Voice config:** `voice_bridge_config.json` for input mode and pipeline settings

## Ops Reference

Full startup and troubleshooting guide: `~/Projects/lloyd/docs/OPS.md`

## Related Docs

- [[index]] -- High-Level Architecture
- [[tools]] -- MCP Tools Server
- [[voice]] -- Voice Pipeline
- [[memory]] -- Memory System
- [[backlog]] -- Backlog System
