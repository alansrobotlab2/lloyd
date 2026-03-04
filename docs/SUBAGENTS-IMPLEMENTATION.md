# Multi-Agent Roster

**Status:** Implemented
**Last updated:** 2026-03-01

---

## Overview

OpenClaw runs a two-tier multi-agent architecture where **Lloyd** is the main orchestrator and dispatches to specialist agents for parallel work, isolated tasks, and quality-loop patterns. Specialists are defined as `agents.list` entries and spawned via `sessions_spawn`.

- **Tier 1 (Roster)** — 4 specialist agents scoped to tool domains. Persistent sessions.
- **Tier 2 (Slim)** — 4 lightweight task agents. Fire-and-forget.

Lloyd retains ALL tools. He gains the ability to delegate, not lose capabilities.

---

## Agent Roster

### Lloyd — Orchestrator (`id: "main"`, `name: "Lloyd"`)

| | |
|---|---|
| **Model** | `anthropic/claude-sonnet-4-6` (via model-router: local/sonnet/opus) |
| **Workspace** | `~/.openclaw/workspaces/lloyd` |
| **Tools** | All tools (unrestricted) |

Routes requests to specialists, handles direct conversation, communication, scheduling, voice, and multi-domain tasks.

---

### Tier 1: Roster Agents

#### Memory (`id: "memory"`)

| | |
|---|---|
| **Model** | `local-llm/Qwen3.5-35B-A3B` (primary), Sonnet (fallback) |
| **Workspace** | `~/.openclaw/workspaces/memory` |
| **Tools** | `tag_search`, `tag_explore`, `vault_overview`, `mem_search`, `mem_get`, `mem_write`, `read` |
| **Spawn mode** | `session` (persistent) |

**Why local**: Eval showed local Qwen scores 4.8/5 on tool selection (outperforms Sonnet) and is 3x faster. Vault lookups are its sweet spot.

**Dispatch triggers**: vault search, note creation, knowledge management, bulk vault ops.

---

#### Coder (`id: "coder"`)

| | |
|---|---|
| **Model** | `anthropic/claude-opus-4-6` |
| **Workspace** | `~/.openclaw/workspaces/coder` |
| **Tools** | `read`, `edit`, `write`, `exec`, `process`, `apply_patch`, `file_read`, `file_write`, `file_edit`, `file_glob`, `file_grep`, `run_bash` |
| **Spawn mode** | `session` (persistent) |

**Dispatch triggers**: multi-file code changes, feature implementation, refactoring, debugging, builds.

---

#### Researcher (`id: "researcher"`)

| | |
|---|---|
| **Model** | `anthropic/claude-opus-4-6` |
| **Workspace** | `~/.openclaw/workspaces/researcher` |
| **Tools** | `http_search`, `http_fetch`, `http_request`, `browser`, `mem_search`, `mem_get`, `read` |
| **Spawn mode** | `session` (persistent) |

**Dispatch triggers**: web research, doc lookup, info gathering, "what's the latest on..."

---

#### Operator (`id: "operator"`)

| | |
|---|---|
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Workspace** | `~/.openclaw/workspaces/operator` |
| **Tools** | `exec`, `run_bash`, `process`, `read`, `file_read`, `file_glob`, `file_grep`, `file_write`, `file_edit`, `backlog_boards`, `backlog_tasks`, `backlog_next_task`, `backlog_get_task`, `backlog_update_task`, `backlog_create_task` |
| **Spawn mode** | `session` (persistent) |

**Dispatch triggers**: git, services, CI/CD, deployments, task board, process management.

---

### Tier 2: Slim Agents

All slim agents: `mode: "run"`, `cleanup: "delete"`, single task, auto-terminate.

#### Tester (`id: "tester"`)

| | |
|---|---|
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Workspace** | `~/.openclaw/workspaces/tester` |
| **Tools** | `read`, `write`, `edit`, `exec`, `run_bash`, `file_read`, `file_glob`, `file_grep` |
| **Tags** | `testing`, `debugging` |

**Spawned when**: after code changes — write tests, run suites, validate.

---

#### Reviewer (`id: "reviewer"`)

| | |
|---|---|
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Workspace** | `~/.openclaw/workspaces/reviewer` |
| **Tools** | `read`, `file_read`, `file_glob`, `file_grep` (read-only) |
| **Tags** | `code_review`, `security` |

**Spawned when**: after code is written — review, check for bugs, adversarial loops.

---

#### Planner (`id: "planner"`)

| | |
|---|---|
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Workspace** | `~/.openclaw/workspaces/planner` |
| **Tools** | `read`, `file_read`, `file_glob`, `file_grep`, `mem_search`, `mem_get` |
| **Tags** | `planning`, `natural_language` |

**Spawned when**: "plan how to implement...", task breakdown, complex multi-step work.

---

#### Auditor (`id: "auditor"`)

| | |
|---|---|
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Workspace** | `~/.openclaw/workspaces/auditor` |
| **Tools** | `read`, `file_read`, `file_glob`, `file_grep` (read-only) |
| **Tags** | `code_review`, `security` |

**Spawned when**: security scanning, vulnerability assessment, red-team reviews.

---

## Model Summary

| Agent | Model | Rationale |
|-------|-------|-----------|
| **Lloyd** | Sonnet (model-router) | Orchestrator uses routing: local for simple, sonnet default, opus for deep |
| **Memory** | Local Qwen (primary) | 3x faster, 4.8/5 tool selection accuracy, free |
| **Coder** | Opus | Code quality benefits most from strongest model |
| **Researcher** | Opus | Deep research and synthesis needs strongest reasoning |
| **Operator** | Sonnet | System ops are mostly tool dispatch, sonnet is sufficient |
| **Tester** | Sonnet | Test writing is structured, sonnet handles well |
| **Reviewer** | Sonnet | Code review is pattern matching, sonnet is capable |
| **Planner** | Sonnet | Planning is reasoning-heavy but not code-generation |
| **Auditor** | Sonnet | Security scanning is checklist-driven |

---

## Dispatch Rules

### Handle Directly (no dispatch)

- Simple questions, greetings, conversation
- Quick one-liner code edits
- Tasks requiring Lloyd's full context (voice, scheduling, multi-domain)
- Back-and-forth conversation

### Dispatch to Specialist

| Signal | Agent | Mode |
|--------|-------|------|
| Vault/memory search, note management | **memory** | session |
| Multi-file code implementation, refactoring | **coder** | session |
| Web research, doc lookup | **researcher** | session |
| Git, deploy, CI/CD, task board | **operator** | session |
| "Write tests for...", "run the suite" | **tester** | run |
| "Review this code", "check for bugs" | **reviewer** | run |
| "Plan how to...", "break this down" | **planner** | run |
| "Audit for security", "scan for vulnerabilities" | **auditor** | run |

---

## Coordination Patterns

### Sequential
```
Lloyd → spawn Planner("design the auth system")
       ← plan
Lloyd → spawn Coder("implement: {plan}")
       ← code
```

### Parallel
```
Lloyd → spawn Researcher("find best practices for X")
      + spawn Coder("scaffold the X module")
      ← both return results → synthesize
```

### Adversarial
```
spawn Coder("implement X") → spawn Reviewer("review: {code}") → loop until clean
```

### Pipeline
```
Planner → Coder → Tester → Reviewer
  plan  →  code  → tests  → review
```

---

## Critical: Subagent Deny List Override

OpenClaw hardcodes `mem_search` and `mem_get` in `SUBAGENT_TOOL_DENY_ALWAYS`. Without the override, Memory, Researcher, and Planner agents silently fail to use vault tools.

```json
"tools": {
  "subagents": {
    "tools": {
      "alsoAllow": ["mem_search", "mem_get"]
    }
  }
}
```

---

## File Structure

```
~/.openclaw/
├── openclaw.json                    # Agent definitions, tool overrides, subagent defaults
├── workspaces/
│   ├── lloyd/                       # Full workspace (AGENTS.md, SOUL.md, MEMORY.md, HEARTBEAT.md, skills/, memory/)
│   ├── memory/                      # AGENTS.md (role prompt) + SOUL.md → lloyd + USER.md → lloyd
│   ├── coder/                       # Same pattern
│   ├── researcher/
│   ├── operator/
│   ├── tester/
│   ├── reviewer/
│   ├── planner/
│   └── auditor/
├── workspace → workspaces/lloyd/    # Backward-compat symlink
├── agents/
│   ├── lloyd/agent/                 # Runtime: sessions, auth, models, tools.json
│   ├── memory/agent/
│   ├── coder/agent/
│   ├── researcher/agent/
│   ├── operator/agent/
│   ├── tester/agent/
│   ├── reviewer/agent/
│   ├── planner/agent/
│   └── auditor/agent/
└── docs/
    ├── SUBAGENTS-IMPLEMENTATION.md  # This file
    └── TOOL-ARCHITECTURE.md         # Tool assembly pipeline reference
```

### Per-agent workspace contents

**Lloyd (full)**: AGENTS.md, SOUL.md, USER.md, TOOLS.md, IDENTITY.md, HEARTBEAT.md, MEMORY.md, skills/ → obsidian, memory/ → obsidian, avatars/

**Specialists (minimal)**: AGENTS.md (role-specific system prompt), SOUL.md → lloyd's, USER.md → lloyd's

### Per-agent runtime files (`agents/{id}/agent/`)

- `models.json` — shared provider config (copied from lloyd)
- `auth-profiles.json` — shared API keys (copied from lloyd)
- `tools.json` — disables built-in tool collisions: `{ "mem_search": false, "mem_get": false, "http_search": false, "http_fetch": false }`
- `system-prompt.md` — detailed role definition (also copied into workspace AGENTS.md)

---

## Architecture Diagram

```
                        ┌──────────────┐
                        │    Lloyd     │
                        │ (orchestrator│
                        │  id: main)   │
                        └──────┬───────┘
                               │ sessions_spawn
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────┴─────┐  ┌──────┴──────┐  ┌──────┴──────┐
     │  Tier 1       │  │  Tier 1      │  │  Tier 1      │  ...
     │  Roster       │  │  Roster      │  │  Roster      │
     │               │  │              │  │              │
     │  memory       │  │  coder       │  │  researcher  │
     │  (local)      │  │  (opus)      │  │  (opus)      │
     │               │  │              │  │              │
     │  operator     │  │  (session)   │  │  (session)   │
     │  (sonnet)     │  └──────────────┘  └──────────────┘
     │  (session)    │
     └──────────────┘
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────┴─────┐  ┌──────┴──────┐  ┌──────┴──────┐
     │  Tier 2       │  │  Tier 2      │  │  Tier 2      │  ...
     │  Slim         │  │  Slim        │  │  Slim        │
     │               │  │              │  │              │
     │  tester       │  │  reviewer    │  │  planner     │
     │  auditor      │  │              │  │              │
     │  (sonnet)     │  │  (sonnet)    │  │  (sonnet)    │
     │  (run)        │  │  (run)       │  │  (run)       │
     └──────────────┘  └──────────────┘  └──────────────┘

  Tier 1: persistent sessions, domain-scoped tools
  Tier 2: fire-and-forget runs, narrow tools, auto-cleanup
```

---

## Verification Checklist

1. **Config validation** — Restart gateway, check logs for parse errors
2. **Agent listing** — `agents_list` shows all 9 agents (lloyd + 8 specialists)
3. **Basic spawn** — "search the vault for notes about X" → spawns `memory` agent
4. **Tool isolation** — Coder can't call `http_search`; Reviewer can't call `exec`
5. **Memory override** — Memory/Researcher/Planner can use `mem_search`/`mem_get` (alsoAllow)
6. **Model assignment** — Coder and Researcher use Opus; Memory uses local Qwen
7. **Parallel spawn** — "research X and write code for Y" → spawns researcher + coder simultaneously
8. **Pipeline test** — "plan, implement, test, and review feature Z" → chains planner → coder → tester → reviewer
