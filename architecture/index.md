---
segment: architecture
relations:
  related-to: []
tags: [architecture]
type: reference

---

















# Lloyd -- High-Level Architecture Overview

Lloyd is a voice-first AI assistant built on a **custom harness** with local vLLM serving Qwen3.5-35B-A3B-nvfp4 at localhost:8096. The system combines a Python MCP tools server,local LLMs,and an Obsidian knowledge vault into a unified personal assistant.

**Model:** Qwen3.5-35B-A3B-nvfp4 (local vLLM at localhost:8096)
**Multi-session:** Per-channel-peer DM scoping,Mission Control session resume
**Process management:** Supervisord (inside distrobox) + systemd user services

## Core Components

| Component | Description | Detail Doc |
|-----------|-------------|------------|
| OpenClaw Gateway | Node.js/TypeScript gateway (port 18789). Agents,sessions,cron,extensions,WebSocket/HTTP API. Runs inside distrobox container `lloyd`. | [[infrastructure]] |
| Main Agent (Lloyd) | Primary conversational agent. Opus 4.6 via Anthropic. Agent ID: `main`. | [[agents]] |
| Memory Agent | Local Qwen3.5-35B-A3B for periodic capture. | [[agents]],[[memory]] |
| Discord Agent | Local LLM for non-DM Discord channels. Agent ID: `discord-lloyd`. | [[agents]] |
| MCP Tools Server | Python FastMCP server (port 8093). 26+ tools across vault,web,filesystem,system,backlog,and skills categories. | [[tools]] |
| Voice Pipeline | Wake word + VAD + Whisper STT + speaker ID + Qwen3-TTS. WebSocket streaming to browser. | [[voice]] |
| Mission Control | React dashboard at `/mc/`. Chat,token usage,API monitoring,services tab. | [[infrastructure]] |
| Obsidian Vault | Knowledge base at `~/obsidian/`. 7 segments: agents,memory,personal,work,projects,knowledge,skills. Per-segment QMD collections. | [[memory]] |
| Memory System | 3-tier: periodic capture (15m),nightly reflection,real-time signal detection. Recall pipeline with intent classification,parallel fan-out,and local consolidation. | [[memory]] |
| Recall Pipeline | Intent-aware search → parallel segment queries → 2B consolidation model → structured results. | [[memory]] |
| Backlog System | SQLite kanban. 4 tools: boards,tasks,get_task,write_task. | [[backlog]] |
| Skill System | 34 skills in `~/obsidian/skills/`. Loaded on-demand via SKILL.md. ClawhHub catalog integration. | [[skills]] |
| Usage Tracking | Token and cost tracking across agents and models. | [[usage-tracking]] |
| Groundskeeper | Vault health scanner + enrichment. 11 scan categories,fix loop (every 15min),research loop (hourly). | [[groundskeeper]] |

## GPU Allocation

| GPU | Hardware | VRAM | Role |
|-----|----------|------|------|
| GPU 0 | RTX 5090 | 32GB | TTS (Qwen3-TTS,~10GB) + Consolidation model (Qwen3.5-2B,~2.5GB) |
| GPU 1 | RTX PRO 6000 Blackwell | 96GB | LLM inference (vLLM,Qwen3.5-122B-A10B) |

## Nightly Schedule (PST)

| Time | Job | Detail Doc |
|------|-----|------------|
| 1:30am | `reflection-synthesis` -- extract decisions,clean corrections | [[nightly-reflection]] |
| 2:00am | `reflection-vault` -- tag hygiene,frontmatter,structure | [[nightly-vault-maintenance]] (deprecated in favor of [[groundskeeper]]) |
| 3:00am | `reflection-skills` -- extraction,evaluation,deduplication | [[nightly-skills-management]] |
| 4:00am | `reflection-signals` -- signal detection & classification | [[nightly-reflection]] |
| 4:20am | `reflection-knowledge` -- mental models,MEMORY.md,pattern analysis | [[nightly-reflection]] |
| 4:40am | `reflection-audit` -- system prompt quality audit & drift detection | [[nightly-reflection]] |
| 4:55am | `reflection-test` -- synthetic behavior tests & regression suite | [[nightly-reflection]] |
| 5:15am | `reflection-config` -- apply fixes from signals + audit + test failures,git commits | [[nightly-reflection]] |
| 7:00am | `morning-briefing` -- overnight synthesis,calendar,backlog | [[morning-briefing]] |
| Sun 1:00am | `reflection-backlog` -- staleness,blocked items,deprioritization | [[nightly-reflection]] |

`periodic-memory-capture` runs every 15 minutes (local Qwen3.5-35B-A3B).

## Services

### Systemd (user services)

| Service | Description |
|---------|-------------|
| `lloyd-llm` | vLLM serving Qwen3.5-122B-A10B (GPU 1) |
| `lloyd-qmd-daemon` | QMD daemon (port 8181) — 7 per-segment collections |
| `lloyd-qmd-watcher` | QMD file watcher |

### Supervisord (inside distrobox)

| Service | Port | GPU | Description |
|---------|------|-----|-------------|
| `agent-tool-mcp` | 8093 | — | MCP tools server (SSE) — mem_search,mem_write,vault tools |
| `agent-consolidation` | 8097 | GPU 0 | Qwen3.5-2B consolidation model (llama-server) |
| `agent-tts` | 8090 | GPU 0 | Qwen3-TTS (OpenAI-compatible API) |
| `agent-voice-mcp` | 8094 | — | Voice MCP server (SSE) |
| `agent-voice-mode` | 8092 | — | Voice TUI + HTTP API |

## Architecture Diagram

```
+------------------+       +-------------------------+
|   Anthropic API  |       |      Discord / Web      |
|  (Opus 4.6)      |       |      Mission