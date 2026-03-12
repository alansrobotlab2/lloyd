---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Agent System

Lloyd runs on the OpenClaw agent platform with Claude Agent SDK subagent dispatch for sustained work.

## Agents

### Main Agent (Lloyd-Prime)

| Property | Value |
|----------|-------|
| Agent ID | `main` |
| Model | Claude Opus 4.6 (Anthropic Max plan, OAuth) |
| Workspace | `~/obsidian/agents/lloyd/` |
| Role | Primary conversational assistant |

**Key workspace files:**

| File | Purpose |
|------|---------|
| SOUL.md | Personality definition |
| AGENTS.md | Behavior rules, delegation policy, social rules |
| MEMORY.md | Long-term curated memory |
| USER.md | User context |
| TOOLS.md | Tool usage guide |
| HEARTBEAT.md | Open threads and pending items |
| IDENTITY.md | Core identity |

Lloyd handles conversation, quick lookups, voice responses, and [[backlog]] queries. Delegates sustained work to the Orchestrator.

### Memory Agent

| Property | Value |
|----------|-------|
| Agent ID | `memory` |
| Model | Local Qwen3.5-35B-A3B (periodic capture), Opus 4.6 (nightly jobs) |
| Workspace | `~/obsidian/agents/memory/` |
| Role | Session transcript extraction and memory capture |

Runs periodic memory capture every 15 minutes using the local LLM. Nightly jobs (vault maintenance, skills management, reflection) use Opus 4.6.

See [[memory]] for the full memory architecture.

### Discord Agent

| Property | Value |
|----------|-------|
| Agent ID | `discord-lloyd` |
| Model | Local LLM |
| Role | Discord server and group channel conversations |

Handles non-DM Discord interactions. User DMs route to the main agent instead.

### Social Agent (REMOVED)

The social agent was removed. Social behavior (Discord friend conversations, tone, restrictions) is now governed by rules in `AGENTS.md` on the main agent. There is no separate social agent.

## Channel Bindings

| Channel | Routed To |
|---------|-----------|
| User DMs | Main agent (`main`) |
| Discord servers / group channels | Discord agent (`discord-lloyd`) |
| Mission Control | Main agent (`main`) |
| Voice | Main agent (`main`) |

## Multi-Session Support

Lloyd supports concurrent sessions with per-channel-peer DM scoping:

- Each DM conversation (identified by channel + peer) maintains its own session
- Mission Control sessions can be resumed across browser reloads
- Minecraft integration supports session resume
- Session state persisted in `~/.openclaw/agents/main/sessions/`

## Subagent System (Orchestrator)

Dispatched by Lloyd for all sustained work. Manages a pipeline of specialized subagents.

- Never spawned directly by Lloyd -- always through `sessions_spawn`
- Coordinates multi-step tasks across specialized subagents

### Subagent Roster

| Agent | Model | Capabilities | Purpose |
|-------|-------|-------------|---------|
| orchestrator | -- | Pipeline management | Coordinates multi-step tasks |
| coder | Opus | Read/Write/Edit/Bash + vault/backlog MCP | Implementation |
| reviewer | Opus | Read-only | Code review |
| tester | Sonnet | Read/Write/Bash | Testing |
| planner | Opus | Read-only + vault MCP | Task breakdown |
| auditor | Opus | Read-only | Security analysis |
| researcher | Sonnet | Read/Web/vault MCP | Research + knowledge capture |
| operator | Sonnet | Read/Write/Bash + backlog MCP | Git, services, CI/CD |
| skills | Haiku | Read + vault MCP | Vault knowledge management |

### Delegation Flow

```
User --> Lloyd (main, Opus 4.6)
           |
           +--> sessions_spawn --> Orchestrator
                                      |
                                      +--> Planner (Opus)
                                      +--> Coder (Opus)
                                      +--> Tester (Sonnet)
                                      +--> Reviewer (Opus)
                                      +--> Researcher (Sonnet)
                                      +--> Operator (Sonnet)
                                      +--> Auditor (Opus)
                                      +--> Skills (Haiku)
```

## Delegation Rules

**Lloyd handles directly:**
- Conversation and quick responses
- Quick file reads
- Memory lookups (vault search)
- [[backlog]] queries
- Daily notes

**Lloyd delegates to Orchestrator:**
- Any code changes
- Research tasks
- Multi-step operations
- Configuration changes

**Hard limits (Lloyd never does):**
- File-modifying bash commands
- Build commands
- Git state changes
- Writes outside `agents/lloyd/`

**Subagent constraints:**
- Subagents cannot be spawned directly -- always through the Orchestrator

## Agent Workspaces

- Each subagent has a workspace doc at `~/obsidian/agents/{agent-id}.md`
- Lloyd's workspace is the full `~/obsidian/agents/lloyd/` directory
- Memory agent's workspace: `~/obsidian/agents/memory/`

## Modes

The mode system affects vault search scope, daily notes path, and prefill context.

| Mode | Vault Scope |
|------|-------------|
| Work | work, knowledge, agents |
| Personal | personal, projects, knowledge, agents |
| General | all segments |

Mode switching is handled by the [[tools|mcp-tools extension]] via `/work`, `/personal`, `/general`, and `/mode` commands.

## Agent Config

All agent configuration lives in `~/.openclaw/openclaw.json`. This includes:
- Agent definitions (ID, model, workspace, tool allowlists)
- Channel bindings (which agent handles which channels)
- Model routing and aliases
- Orchestrator subagent definitions

## Related Docs

- [[index]] -- High-Level Architecture
- [[memory]] -- Memory System (memory agent details)
- [[tools]] -- MCP Tools Server (tool access)
- [[skills]] -- Skill System (skill-based procedures)
- [[infrastructure]] -- Infrastructure (services and cron)
