---
segment: architecture
tags: [architecture,lloyd]
relations:
  related-to:
  - projects/lloyd/gpu-allocation.md
  - architecture/morning-briefing.md
  - architecture/nightly-reflection.md
  - architecture/nightly-skills-management.md
  - architecture/nightly-vault-maintenance.md
  - architecture/skills.md
  - architecture/tools.md
  - architecture/voice.md
  - architecture/memory.md
  - architecture/autonomy-system.md
  - architecture/evaluation-engine.md
  - projects/lloyd/plans/voice-async-protocol.md
  - projects/lloyd/plans/document-relations-retrieval.md
  - architecture/agents.md
  - architecture/backlog.md
  - architecture/index.md
  - architecture/infrastructure.md
tags: [architecture]
summary: 'Infrastructure reference: 3-GPU setup,systemd user services,port map,
  OpenClaw extensions,Mission Control,and nightly cron schedule.'
type: reference

---

















# Infrastructure

## Runtime Environment

| Property | Value |
|----------|-------|
| Host OS | Arch Linux |
| Services Repo | `~/agents/agent-services/` (systemd user services) |
| Model | Opus 4.6 |
| Config | `~/.openclaw/openclaw.json` |
| Gateway | Local + LAN,TLS,token auth |

## GPUs

| GPU | Hardware | VRAM | Assignment |
|-----|----------|------|------------|
| GPU0 | RTX 5090 | 32 GB | Local LLM via llama-server (Qwen3.5-35B-A3B) |
| GPU1 | RTX 3090 | 24 GB | TTS + voice pipeline (distrobox services) |
| GPU2 | RTX 3090 | 24 GB | QMD search + OpenClaw gateway |

## Services

All services are systemd user units defined at `~/agents/agent-services/systemd/`. Managed with `systemctl --user`.

### OpenClaw Gateways

| Service | Description | Port |
|---------|-------------|------|
| `openclaw-lloyd` | OpenClaw Gateway — Lloyd (distrobox) | 18789 |
| `openclaw-dee` | OpenClaw Gateway — DEE | 19789 |
| `openclaw-cert` | Certificate download page | 18790 |

### LLM

| Service | Description | Notes |
|---------|-------------|-------|
| `agent-llm` | Local LLM — Qwen3.5-35B-A3B (llama-server) | GPU0 |

### TTS Engines

| Service | Description | Notes |
|---------|-------------|-------|
| `agent-tts` | Qwen3-TTS Server (primary) | GPU1,port 8090,distrobox |
| `agent-qwen3-tts` | Qwen3 TTS Server (alternate) | GPU1,distrobox |
| `agent-fish-speech` | Fish Speech TTS Server | GPU1,distrobox |
| `agent-index-tts` | Index TTS Server | GPU1,distrobox |

### Voice Pipeline

| Service | Description | Notes |
|---------|-------------|-------|
| `agent-voice-mode` | Voice Mode (headless) | Distrobox |
| `agent-voice-mcp` | Voice MCP Server (SSE) | Port 8094,distrobox |
| `agent-discord-voice-bridge` | Discord Voice Bridge (Node.js) | |
| `agent-discord-voice-server` | Discord Voice Bridge Server (Python) | |

### MCP & Tools

| Service | Description | Notes |
|---------|-------------|-------|
| `agent-tool-mcp` | Tool Services MCP (SSE) | Port 8093,distrobox |
| `agent-qmd-daemon` | QMD Search Daemon (HTTP MCP) | Port 8181 |
| `agent-qmd-watcher` | QMD Vault Watcher (auto-index) | |

### Other

| Service | Description | Notes |
|---------|-------------|-------|
| `agent-distrobox` | Distrobox supervisord (legacy) | Wraps distrobox `agent-services` container |

## Port Map

| Port | Service | Protocol |
|------|---------|----------|
| 8090 | Qwen3-TTS | HTTP |
| 8091 | Local LLM (llama-server) | HTTP |
| 8092 | Voice Mode API (voice_mode.py) | HTTP |
| 8093 | Tool Services MCP (tool_services.py) | SSE |
| 8094 | Voice MCP (voice_services.py) | SSE |
| 8181 | QMD Search | HTTP |
| 18789 | OpenClaw Gateway — Lloyd | HTTPS |
| 18790 | OpenClaw Certificate Page | HTTP |
| 19789 | OpenClaw Gateway — DEE | HTTPS |

## Nightly Crons

OpenClaw has a built-in cron system (`openclaw cron`). Jobs run as isolated sessions with configurable model,timeout,and timezone.

| Job | Schedule | Description |
|-----|----------|-------------|
| Vault maintenance | 2:00 AM PST | Nightly vault cleanup and organization |
| Skills management | 3:00 AM PST | Skill review,updates,and maintenance |
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
- `on(hook)