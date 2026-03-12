---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Lloyd -- High-Level Architecture Overview

Lloyd is a voice-first AI assistant built on **OpenClaw**, a custom gateway/agent platform. The system combines a Node.js gateway, Python tool and voice services, a local LLM, and an Obsidian knowledge vault into a unified personal assistant.

**Model:** Claude Opus 4.6 (Anthropic Max plan, OAuth auth)
**Multi-session:** Per-channel-peer DM scoping, Mission Control session resume
**Process management:** Supervisord (inside distrobox) + systemd user services

## Core Components

| Component | Description | Detail Doc |
|-----------|-------------|------------|
| OpenClaw Gateway | Node.js/TypeScript gateway (port 18789). Agents, sessions, cron, extensions, WebSocket/HTTP API. Runs inside distrobox container `lloyd`. | [[infrastructure]] |
| Main Agent (Lloyd) | Primary conversational agent. Opus 4.6 via Anthropic. Agent ID: `main`. | [[agents]] |
| Memory Agent | Local Qwen3.5-35B-A3B for periodic capture. | [[agents]], [[memory]] |
| Discord Agent | Local LLM for non-DM Discord channels. Agent ID: `discord-lloyd`. | [[agents]] |
| MCP Tools Server | Python FastMCP server (port 8093). 26 tools across vault, web, filesystem, system, backlog, and skills categories. | [[tools]] |
| Voice Pipeline | Wake word + VAD + Whisper STT + speaker ID + Qwen3-TTS. WebSocket streaming to browser. | [[voice]] |
| Mission Control | React dashboard at `/mc/`. Chat, token usage, API monitoring, services tab. | [[infrastructure]] |
| Obsidian Vault | Knowledge base at `~/obsidian/`. 5 segments: agents, personal, work, projects, knowledge. BM25 FTS5 search. | [[memory]] |
| Memory System | 3-tier: periodic capture (15m), nightly reflection, real-time signal detection. Prefill context injection. | [[memory]] |
| Backlog System | SQLite kanban. 4 tools: boards, tasks, get_task, write_task. | [[backlog]] |
| Skill System | 34 skills in `~/obsidian/skills/`. Loaded on-demand via SKILL.md. ClawhHub catalog integration. | [[skills]] |
| Usage Tracking | Token and cost tracking across agents and models. | [[usage-tracking]] |

## GPU Allocation

| GPU | Hardware | Role |
|-----|----------|------|
| GPU0 | RTX 5090 32GB | LLM inference (vLLM, Qwen3.5-35B-A3B) |
| GPU1 | RTX 3090 24GB | TTS (Qwen3-TTS) + voice pipeline |
| GPU2 | RTX 3090 24GB | QMD daemon + gateway |

## Nightly Schedule (PST)

| Time | Job | Detail Doc |
|------|-----|------------|
| 2:00am | Vault maintenance -- tag hygiene, frontmatter, structure | [[nightly-vault-maintenance]] |
| 3:00am | Skills management -- extraction, evaluation, deduplication | [[nightly-skills-management]] |
| 4:00am | Reflection + self-improvement -- corrections, mental models, MEMORY.md consolidation | [[nightly-reflection]] |

Periodic memory capture runs every 15 minutes (local Qwen3.5-35B-A3B).

## Services

### Systemd (user services)

| Service | Description |
|---------|-------------|
| `lloyd-llm` | vLLM serving Qwen3.5-35B-A3B |
| `lloyd-qmd-daemon` | QMD daemon (port 8181) |
| `lloyd-qmd-watcher` | QMD file watcher |

### Supervisord (inside distrobox)

| Service | Port | Description |
|---------|------|-------------|
| `lloyd-tool-mcp` | 8093 | MCP tools server (SSE) |
| `lloyd-tts` | 8090 | Qwen3-TTS (OpenAI-compatible API) |
| `lloyd-voice-mcp` | 8094 | Voice MCP server (SSE) |
| `lloyd-voice-mode` | 8092 | Voice TUI + HTTP API |
| `openclaw-gateway` | 18789 | Gateway (manual start, not autostart) |

## Architecture Diagram

```
+------------------+       +-------------------------+
|   Anthropic API  |       |      Discord / Web      |
|  (Opus 4.6)     |       |      Mission Control     |
+--------+---------+       +------------+------------+
         |                              |
         v                              v
+----------------------------------------------------------+
|              OpenClaw Gateway :18789                      |
|  Agents | Sessions | Cron | Extensions | WebSocket API   |
+----+------------------+------------------+---------------+
     |                  |                  |
     v                  v                  v
+----------+    +-----------+    +------------------+
| MCP Tools|    | Voice MCP |    | Extensions       |
| :8093    |    | :8094     |    | mcp-tools        |
+----+-----+    +-----+-----+   | voice-tools      |
     |                |          | mission-control   |
     v                v          +------------------+
+---------+    +-----------+
| Vault   |    | Voice     |
| BM25    |    | Pipeline  |
| Backlog |    | :8092     |
| Web     |    +-----+-----+
+---------+          |
     |               v
     v          +---------+
+---------+    | TTS     |
| Local   |    | :8090   |
| LLM     |    +---------+
| :8091   |
+---------+
  GPU0          GPU1         GPU2
  RTX 5090      RTX 3090     RTX 3090
  (LLM)        (TTS+Voice)  (QMD+GW)
```

## Key Paths

| Path | Purpose |
|------|---------|
| `~/Projects/lloyd-services/` | Voice pipeline + MCP servers (Python) |
| `~/.openclaw/` | Gateway config, extensions, data, sessions |
| `~/.openclaw/openclaw.json` | Main configuration file |
| `~/.openclaw/extensions/` | Plugins: mcp-tools, voice-tools, mission-control |
| `~/obsidian/` | Knowledge vault |
| `~/obsidian/agents/lloyd/` | Lloyd's workspace (SOUL.md, AGENTS.md, MEMORY.md) |
| `~/obsidian/skills/` | 34 custom skills |

## Related Docs

- [[agents]] -- Agent System
- [[memory]] -- Memory System Architecture
- [[tools]] -- MCP Tools Server
- [[voice]] -- Voice Pipeline
- [[infrastructure]] -- Infrastructure and Services
- [[backlog]] -- Backlog System
- [[skills]] -- Skill System
- [[usage-tracking]] -- Usage Tracking
- [[nightly-vault-maintenance]] -- Nightly Vault Maintenance
- [[nightly-skills-management]] -- Nightly Skills Management
- [[nightly-reflection]] -- Nightly Reflection + Self-Improvement
