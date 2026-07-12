---
segment: architecture
tags: [agents,architecture,lloyd]
summary: 'Agent roster and delegation flow: main agent,memory agent,discord agent,
  orchestrator subagents (coder/reviewer/tester/planner/auditor/researcher/operator/skills).'
type: reference
relations:
  related-to:
  - architecture/morning-briefing.md
  - architecture/nightly-reflection.md
  - architecture/nightly-skills-management.md
  - architecture/nightly-vault-maintenance.md
  - architecture/voice.md
  - architecture/autonomy-system.md
  - architecture/evaluation-engine.md

---




















# Agent System

Lloyd runs on the OpenClaw agent platform with Claude Agent SDK subagent dispatch for sustained work.

## Autonomy System (2026-03-22)

Four specialized agent types pull from a single autonomy backlog with GPU-gated LLM dispatch:

### Agent Types

| Agent | Role | Trigger | GPU Gating |
|-------|------|---------|------------|
| **Memory** | Periodic jobs: nightly tasks,fact extraction,index rebuild | Cron (2:00-5:15 AM PST) | Polls GPU before LLM dispatch |
| **Operator** | Backlog review: analysis,code changes,config updates | Backlog items (tag-based) | Polls GPU before LLM dispatch |
| ** Idler** | Deep maintenance: memory audit,relationship discovery,skill harvest | Backlog items (priority-based) | Polls GPU before LLM dispatch |
| **Researcher** | Content ingestion: GitHub,RSS,docs,research | Backlog items (content ingestion) | Polls GPU before LLM dispatch |

### GPU-Gated LLM Dispatch

- **Polling:** `nvidia-smi --query-gpu=utilization.gpu` every 250ms
- **Threshold:** GPU < 30% → dispatch LLM call; GPU ≥ 30% → backoff 250ms,retry
- **Scope:** Only LLM calls stall; file ops,web fetches,tool calls proceed normally
- **Architecture:** GPU gate lives in each agent,not OpenClaw core — independent,no contention

### Task Routing

- **Tag-based:** `tag="periodic"`,`tag="implementation"`,`tag="audit"`
- **Priority-based:** High-priority items routed to appropriate agent type

### Model Tier Hierarchy (2026-03-26)

Intelligence funnel: Opus (planning/management) → Sonnet (implementation) → 122B (synthesis) → 35B (mechanical)

| Tier | Model | Agents | Rationale |
|------|-------|--------|-----------|
| Opus 4.6 | `anthropic/claude-opus-4-6` | main,orchestrator | Planning,analysis,task prompt authoring,pipeline management |
| Sonnet 4.6 | `anthropic/claude-sonnet-4-6` | coder,reviewer,planner | Core implementation pipeline — quality-critical,off-GPU |
| 122B | `local-llm-120b/Qwen3.5-122B-A10B` | researcher,auditor,discord-lloyd | Smart synthesis/scanning,GPU-local |
| 35B | `local-llm-35b/Qwen3.5-35B-A3B` | memory,operator,tester,explorer,skillser | Mechanical/procedural tasks,light GPU footprint |

**GPU contention solved:** Core implementation pipeline (plan → code → review) is entirely off-GPU via Sonnet,so wiggam tasks don't block interactive sessions.

## Agents

### Main Agent (Lloyd-Prime)

| Property | Value |
|----------|-------|
| Agent ID | `main` |
| Model | Claude Opus 4.6 (Anthropic Max plan,OAuth) |
| Workspace | `~/obsidian/agents/lloyd/` |
| Role | Primary conversational assistant |

**Key workspace files:**

| File | Purpose |
|------|---------|
| SOUL.md | Personality definition |
| AGENTS.md | Behavior rules,delegation policy,social rules |
| MEMORY.md | Long-term curated memory |
| USER.md | User context |
| TOOLS.md | Tool usage guide |
| HEARTBEAT.md | Open threads and pending items |
| IDENTITY.md | Core identity |

Lloyd handles conversation,quick lookups,voice responses,and [[backlog]] queries. Delegates sustained work to the Orchestrator.

### Memory Agent

| Property | Value |
|----------|-------|
| Agent ID | `memory` |
| Model | Local Qwen3.5-35B-A3B (periodic capture),Opus 4.6 (nightly jobs) |
| Workspace | `~/obsidian/agents/memory/` |
| Role | Session transcript extraction and memory capture |

Runs periodic memory capture every 15 minutes using the local LLM. Nightly jobs (vault maintenance,skills management,reflection) use Opus 4.6.

See [[memory]] for the full memory architecture.

### Discord Agent

| Property | Value |
|----------|-------|
| Agent ID | `discord-lloyd` |
| Model | Local LLM |
| Role | Discord server and group channel conversations |

Handles non-DM Discord interactions. User DMs route to the main agent instead.

### Social Agent (REMOVED)

The social agent was removed. Social behavior (Discord friend conversations,tone,restrictions) is now governed by rules in `AGENTS.md` on the main agent. There is no separate social agent.

## Channel Bindings

| Channel | Routed To |
|---------|-----------|
| User DMs | Main agent (`main`) |
| Discord servers / group channels | Discord agent (`discord-lloyd`) |
| Mission Control | Main agent (`main`) |
| Voice | Main agent (`main`) |

## Multi-Session Support

Lloyd supports concurrent sessions with per-channel-peer DM scoping:

- Each DM conversation (identified by channel + peer) m