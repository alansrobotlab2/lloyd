# Multi-Agent Roster Implementation

**Status:** Planning
**Last updated:** 2026-03-01

---

## Overview

OpenClaw gains a two-tier multi-agent architecture where **Lloyd** remains the main orchestrator and dispatches to specialist agents for parallel work, isolated tasks, and quality-loop patterns (review, audit, test). Specialists are defined as `agents.list` entries and spawned via `sessions_spawn`.

- **Tier 1 (Roster)** — 4 specialist agents scoped to tool domains. Spawned with `mode: "session"` for persistent context across tasks.
- **Tier 2 (Slim)** — 4 lightweight task agents. Spawned with `mode: "run"`, `cleanup: "delete"` for fire-and-forget execution.

Lloyd retains ALL current tools. He doesn't lose capabilities — he gains the ability to delegate.

---

## Agent Roster

### Lloyd — Orchestrator (`id: "main"`)

Lloyd stays as the fully-capable main agent. He owns all orchestration, communication, and scheduling tools:

| Category | Tools |
|----------|-------|
| Agent management | `agents_list`, `sessions_list`, `sessions_history`, `sessions_send`, `sessions_spawn`, `subagents`, `session_status` |
| Communication | `message`, `tts`, `voice_last_utterance`, `voice_enroll_speaker`, `voice_list_speakers` |
| Scheduling | `cron`, `gateway` |
| UI / Media | `browser`, `canvas`, `image`, `nodes` |
| Everything else | All base coding tools, MCP tools, ClawDeck tools (unchanged) |

Lloyd decides which specialist handles a request based on dispatch heuristics (see [Dispatch Rules](#dispatch-rules)).

---

### Tier 1: Roster Agents

#### Memory (`id: "memory"`)

Vault and knowledge operations — search, retrieve, create, organize notes.

| | |
|---|---|
| **Tools** | `tag_search`, `tag_explore`, `vault_overview`, `memory_search`, `memory_get`, `memory_write`, `read` |
| **Model** | `local-llm/Qwen3.5-35B-A3B` (primary), `anthropic/claude-sonnet-4-6` (fallback) |
| **Spawn mode** | `session` (persistent) |

**Dispatch triggers**: "search the vault for...", "find notes about...", "create a knowledge doc for...", "update the note on...", bulk vault maintenance.

**Why local model**: Eval showed local Qwen scores 4.8/5 on tool selection (outperforms Sonnet) and is 3× faster. Vault lookups are its sweet spot.

---

#### Coder (`id: "coder"`)

Code generation, editing, debugging, refactoring, execution.

| | |
|---|---|
| **Tools** | `read`, `edit`, `write`, `exec`, `process`, `apply_patch`, `file_read`, `file_write`, `file_edit`, `file_glob`, `file_grep`, `run_bash` |
| **Model** | `anthropic/claude-sonnet-4-6` (default); Lloyd can escalate to Opus via spawn `model` param |
| **Spawn mode** | `session` (persistent) |

**Dispatch triggers**: "implement this feature", "fix this bug", "refactor this module", multi-file code changes, build/compile tasks.

---

#### Researcher (`id: "researcher"`)

Web research, documentation lookup, information synthesis.

| | |
|---|---|
| **Tools** | `web_search`, `web_fetch`, `http_request`, `browser`, `memory_search`, `memory_get`, `read` |
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Spawn mode** | `session` (persistent) |

**Dispatch triggers**: "look up docs for...", "research how to...", "find examples of...", "what's the latest on...", web-heavy info gathering.

---

#### Operator (`id: "operator"`)

System administration, DevOps, git, process management, project tracking.

| | |
|---|---|
| **Tools** | `exec`, `run_bash`, `process`, `read`, `file_read`, `file_glob`, `file_grep`, `file_write`, `file_edit`, `clawdeck_boards`, `clawdeck_tasks`, `clawdeck_next_task`, `clawdeck_get_task`, `clawdeck_update_task`, `clawdeck_create_task` |
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Spawn mode** | `session` (persistent) |

**Dispatch triggers**: "check git status", "deploy this", "restart the service", "check CI", "what's my next task", "update the board".

---

### Tier 2: Slim Agents

All slim agents: `mode: "run"`, `cleanup: "delete"`, single task, auto-terminate.

#### Tester (`id: "tester"`)

Write and run tests, validate functionality.

| | |
|---|---|
| **Tools** | `read`, `write`, `edit`, `exec`, `run_bash`, `file_read`, `file_glob`, `file_grep` |
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Tags** | `testing`, `debugging` |

**Spawned when**: After code changes — "write tests for this", "run the test suite", "validate this works".

---

#### Reviewer (`id: "reviewer"`)

Code review — find bugs, style issues, security problems. **Read-only** — no write or exec.

| | |
|---|---|
| **Tools** | `read`, `file_read`, `file_glob`, `file_grep` |
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Tags** | `code_review`, `security` |

**Spawned when**: After code is written — "review this PR", "check for bugs", adversarial review loops.

---

#### Planner (`id: "planner"`)

Break down tasks, create implementation plans, coordinate work.

| | |
|---|---|
| **Tools** | `read`, `file_read`, `file_glob`, `file_grep`, `memory_search`, `memory_get` |
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Tags** | `planning`, `natural_language` |

**Spawned when**: "Plan how to implement...", "break this down into tasks", complex multi-step work.

---

#### Auditor (`id: "auditor"`)

Security scanning — find vulnerabilities in code and configs. **Read-only**.

| | |
|---|---|
| **Tools** | `read`, `file_read`, `file_glob`, `file_grep` |
| **Model** | `anthropic/claude-sonnet-4-6` |
| **Tags** | `code_review`, `security` |

**Spawned when**: "Audit for security issues", "check for vulnerabilities", red-team reviews.

---

## Dispatch Rules

Lloyd uses these heuristics to decide when to dispatch vs handle directly:

### Handle Directly (no dispatch)

- Simple questions, greetings, conversation
- Quick one-liner code edits
- Tasks requiring Lloyd's full context (voice, scheduling, multi-domain)
- Anything that needs orchestration tools (cron, message, sessions)

### Dispatch to Specialist

| Signal in User Request | Agent | Mode |
|------------------------|-------|------|
| Vault/memory search, note creation, knowledge management | **memory** | session |
| Multi-file code implementation, refactoring, debugging | **coder** | session |
| Web research, doc lookup, "what's the latest on" | **researcher** | session |
| Git, deploy, CI/CD, service management, task board | **operator** | session |
| "Write tests for...", "run the test suite" | **tester** | run |
| "Review this code", "check for bugs" | **reviewer** | run |
| "Plan how to...", "break this down" | **planner** | run |
| "Audit for security", "scan for vulnerabilities" | **auditor** | run |

### Escalation

Lloyd can override any specialist's model at spawn time:

```
sessions_spawn({ agentId: "coder", model: "claude-opus-4-6", task: "..." })
```

Use Opus escalation for: architecture decisions, complex reasoning, deep analysis.

---

## Coordination Patterns

These are orchestration strategies Lloyd follows. No new code — just dispatch logic.

### Sequential

Agents run one after another, each building on prior output.

```
Lloyd → spawn Planner("design the auth system")
       ← Planner returns plan
Lloyd → spawn Coder("implement based on this plan: {plan}")
       ← Coder returns implementation
```

### Parallel

Multiple agents work simultaneously on independent subtasks.

```
Lloyd → spawn Researcher("find best practices for auth")
      + spawn Coder("scaffold the auth module skeleton")
      ← both return results
Lloyd → synthesize and respond
```

### Adversarial

One agent generates, another critiques. Loop until quality threshold met.

```
Lloyd → spawn Coder("implement feature X")
       ← Coder returns code
Lloyd → spawn Reviewer("review this code for bugs and issues: {code}")
       ← Reviewer returns issues
Lloyd → spawn Coder("fix these issues: {issues}")
       ← ... iterate until clean
```

### Debate

Two agents argue positions. Lloyd synthesizes.

```
Lloyd → spawn Planner("argue for approach A: microservices")
      + spawn Planner("argue for approach B: monolith")
      ← both return arguments
Lloyd → synthesize best approach
```

### Pipeline

Chain of agents where output flows through stages.

```
Planner → Coder → Tester → Reviewer
   plan  →  code  → tests  → review
```

Each stage spawned sequentially, output forwarded to the next.

---

## Implementation: Config Changes

### `openclaw.json` — Agent List

Add 8 agents to `agents.list` alongside the existing `main` entry:

```json
{
  "agents": {
    "defaults": {
      "model": { "primary": "anthropic/claude-sonnet-4-6" },
      "bootstrapMaxChars": 20000,
      "bootstrapTotalMaxChars": 150000,
      "compaction": { "mode": "safeguard" },
      "maxConcurrent": 4,
      "subagents": {
        "maxConcurrent": 8,
        "maxSpawnDepth": 2,
        "maxChildrenPerAgent": 5,
        "archiveAfterMinutes": 60,
        "runTimeoutSeconds": 300
      }
    },
    "list": [
      {
        "id": "main",
        "identity": { "avatar": "avatars/lloyd_003.jpg" },
        "subagents": {
          "allowAgents": ["memory", "coder", "researcher", "operator", "tester", "reviewer", "planner", "auditor"]
        }
      },
      {
        "id": "memory",
        "name": "Memory",
        "model": { "primary": "local-llm/Qwen3.5-35B-A3B", "fallbacks": ["anthropic/claude-sonnet-4-6"] },
        "tools": {
          "allow": ["tag_search", "tag_explore", "vault_overview", "memory_search", "memory_get", "memory_write", "read"]
        }
      },
      {
        "id": "coder",
        "name": "Coder",
        "model": "anthropic/claude-sonnet-4-6",
        "tools": {
          "allow": ["read", "edit", "write", "exec", "process", "apply_patch", "file_read", "file_write", "file_edit", "file_glob", "file_grep", "run_bash"]
        }
      },
      {
        "id": "researcher",
        "name": "Researcher",
        "model": "anthropic/claude-sonnet-4-6",
        "tools": {
          "allow": ["web_search", "web_fetch", "http_request", "browser", "memory_search", "memory_get", "read"]
        }
      },
      {
        "id": "operator",
        "name": "Operator",
        "model": "anthropic/claude-sonnet-4-6",
        "tools": {
          "allow": ["exec", "run_bash", "process", "read", "file_read", "file_glob", "file_grep", "file_write", "file_edit",
                     "clawdeck_boards", "clawdeck_tasks", "clawdeck_next_task", "clawdeck_get_task", "clawdeck_update_task", "clawdeck_create_task"]
        }
      },
      {
        "id": "tester",
        "name": "Tester",
        "model": "anthropic/claude-sonnet-4-6",
        "tools": {
          "allow": ["read", "write", "edit", "exec", "run_bash", "file_read", "file_glob", "file_grep"]
        }
      },
      {
        "id": "reviewer",
        "name": "Reviewer",
        "model": "anthropic/claude-sonnet-4-6",
        "tools": {
          "allow": ["read", "file_read", "file_glob", "file_grep"]
        }
      },
      {
        "id": "planner",
        "name": "Planner",
        "model": "anthropic/claude-sonnet-4-6",
        "tools": {
          "allow": ["read", "file_read", "file_glob", "file_grep", "memory_search", "memory_get"]
        }
      },
      {
        "id": "auditor",
        "name": "Auditor",
        "model": "anthropic/claude-sonnet-4-6",
        "tools": {
          "allow": ["read", "file_read", "file_glob", "file_grep"]
        }
      }
    ]
  }
}
```

### `openclaw.json` — Subagent Tool Overrides

OpenClaw hardcodes `memory_search` and `memory_get` in `SUBAGENT_TOOL_DENY_ALWAYS`. Override this for agents that need vault access:

```json
{
  "tools": {
    "web": { "search": { "enabled": false }, "fetch": { "enabled": false } },
    "subagents": {
      "tools": {
        "alsoAllow": ["memory_search", "memory_get"]
      }
    }
  }
}
```

Without this, the Memory, Researcher, and Planner agents will silently fail to use vault tools.

---

## Implementation: File Structure

### New directories

```
agents/memory/agent/
agents/coder/agent/
agents/researcher/agent/
agents/operator/agent/
agents/tester/agent/
agents/reviewer/agent/
agents/planner/agent/
agents/auditor/agent/
```

### Per-agent files (in each `agents/{id}/agent/`)

**`tools.json`** — Disable built-in tools that collide with MCP equivalents:
```json
{ "memory_search": false, "memory_get": false, "web_search": false, "web_fetch": false }
```

**`models.json`** — Copy from `agents/main/agent/models.json` (shared provider config).

**`auth-profiles.json`** — Copy from `agents/main/agent/auth-profiles.json` (shared API keys).

### Per-agent system prompts

Create `agents/{id}/system-prompt.md` for each agent defining role, constraints, and output format.

**`agents/memory/system-prompt.md`**:
```markdown
# Memory Agent

You are a vault and knowledge specialist. You search, retrieve, create, and organize
notes in the Obsidian vault.

## Constraints
- Only operate on vault content — do not edit code, run commands, or browse the web
- Use tag_search for structured lookups, memory_search for content search
- When creating notes, follow the vault frontmatter conventions (type, tags, source, date, summary)

## Output
- Return concise results with file paths and relevant excerpts
- For write operations, confirm what was created/updated
```

**`agents/coder/system-prompt.md`**:
```markdown
# Coder Agent

You are a specialist code agent. Write, edit, debug, and refactor code.

## Constraints
- Only modify files relevant to your assigned task
- Run tests after changes when possible
- Do not make architectural decisions — flag them for the orchestrator
- Do not search the web or access memory — you only work with code

## Output
- Concise summary of what you changed and why
- List of all files modified
- Any issues or follow-up work needed
```

**`agents/researcher/system-prompt.md`**:
```markdown
# Researcher Agent

You are a web research and information specialist. Find, analyze, and synthesize information.

## Constraints
- Focus on finding accurate, current information
- Cross-reference multiple sources when possible
- Use memory_search to check if relevant notes already exist before creating new ones
- Do not modify code or run system commands

## Output
- Structured findings with sources cited
- Key takeaways and recommendations
- Links to primary sources
```

**`agents/operator/system-prompt.md`**:
```markdown
# Operator Agent

You are a system operations specialist. Handle git, shell, services, CI/CD, and project tracking.

## Constraints
- Prefer non-destructive operations (check status before modifying)
- For destructive commands (rm, reset, force-push), flag for orchestrator approval
- Use ClawDeck tools for project/task management

## Output
- Command outputs and their interpretation
- Status summaries
- Any warnings or required follow-up actions
```

**`agents/tester/system-prompt.md`**:
```markdown
# Tester Agent

You are a testing specialist. Write tests, run test suites, validate functionality.

## Constraints
- Focus exclusively on testing — do not refactor production code
- Report all failures with clear reproduction steps
- Write tests that are deterministic and isolated

## Output
- Test results (pass/fail counts)
- Details on any failures
- Coverage gaps identified
```

**`agents/reviewer/system-prompt.md`**:
```markdown
# Reviewer Agent

You are a code review specialist. Find bugs, style issues, and potential problems.
You have READ-ONLY access — you cannot modify files.

## Constraints
- Review only the files/changes specified in your task
- Categorize issues by severity: critical, warning, suggestion
- Be specific — include file paths, line numbers, and concrete fixes

## Output
- Itemized list of issues found, ordered by severity
- Suggested fixes for each issue
- Overall assessment (approve / request changes)
```

**`agents/planner/system-prompt.md`**:
```markdown
# Planner Agent

You are a planning and task breakdown specialist. Analyze requirements, design approaches,
and create actionable implementation plans.

## Constraints
- Explore the codebase before planning (read relevant files, search for patterns)
- Identify risks and dependencies
- Break work into concrete, independently-testable steps

## Output
- Step-by-step implementation plan
- Files that will need modification
- Dependencies and risks
- Estimated complexity per step
```

**`agents/auditor/system-prompt.md`**:
```markdown
# Auditor Agent

You are a security auditor. Scan code and configs for vulnerabilities.
You have READ-ONLY access — you cannot modify files.

## Constraints
- Check for OWASP Top 10 vulnerabilities
- Scan for hardcoded secrets, insecure defaults, injection vectors
- Review dependency versions for known CVEs where visible
- Be specific — include file paths, line numbers, and CWE IDs where applicable

## Output
- Itemized vulnerability report, ordered by severity (critical → low)
- Concrete remediation steps for each finding
- Overall security posture assessment
```

---

## Implementation: Lloyd's Dispatch Heuristics

Add the following to `workspace/AGENTS.md` so Lloyd knows when and how to use his roster:

```markdown
## Agent Dispatch

You have 8 specialist agents available. Use them for parallel work, isolated tasks,
or quality loops. You can always handle things directly for simple tasks.

### When to Dispatch

| If the task involves... | Dispatch to | Spawn mode |
|-------------------------|-------------|------------|
| Vault search, note management, knowledge organization | `memory` | session |
| Multi-file code changes, feature implementation, debugging | `coder` | session |
| Web research, doc lookup, information gathering | `researcher` | session |
| Git, services, CI/CD, deployments, task board | `operator` | session |
| Writing/running tests after code changes | `tester` | run |
| Code review, bug hunting | `reviewer` | run |
| Task breakdown, implementation planning | `planner` | run |
| Security scanning, vulnerability assessment | `auditor` | run |

### When NOT to Dispatch

- Simple questions, greetings, conversation (just answer)
- Quick one-liner edits (faster to do directly)
- Tasks requiring your full context (voice, scheduling, multi-domain)
- When the user is having a back-and-forth conversation (stay present)

### Coordination Patterns

**Parallel** — For independent subtasks:
  Spawn researcher + coder simultaneously when tasks don't depend on each other.

**Pipeline** — For end-to-end feature work:
  planner → coder → tester → reviewer (each stage feeds the next).

**Adversarial** — For quality assurance:
  Spawn coder, then reviewer/auditor to critique. Loop until clean.

### Escalation

Override any agent's model at spawn time for hard problems:
  sessions_spawn({ agentId: "coder", model: "claude-opus-4-6", task: "..." })
```

---

## Verification Checklist

1. **Config validation** — Restart gateway, check logs for parse errors
2. **Agent listing** — `agents_list` should show all 9 agents (main + 8 specialists)
3. **Basic spawn** — "search the vault for notes about X" → spawns `memory` agent
4. **Tool isolation** — Coder can't call `web_search`; Reviewer can't call `exec`
5. **Memory override** — Memory/Researcher/Planner can use `memory_search`/`memory_get` (not blocked by deny list) thanks to `alsoAllow`
6. **Parallel spawn** — "research X and write code for Y at the same time" → spawns researcher + coder simultaneously
7. **Pipeline test** — "plan, implement, test, and review feature Z" → chains planner → coder → tester → reviewer
8. **Adversarial test** — "write code for X and review it" → spawns coder then reviewer with critique loop

---

## Architecture Diagram

```
                        ┌──────────────┐
                        │    Lloyd     │
                        │ (orchestrator│
                        │   id: main)  │
                        └──────┬───────┘
                               │ sessions_spawn
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────┴─────┐  ┌──────┴──────┐  ┌──────┴──────┐
     │  Tier 1       │  │  Tier 1      │  │  Tier 1      │  ...
     │  Roster       │  │  Roster      │  │  Roster      │
     │               │  │              │  │              │
     │  memory       │  │  coder       │  │  researcher  │
     │  operator     │  │              │  │              │
     │               │  │  (session)   │  │  (session)   │
     │  (session)    │  └──────────────┘  └──────────────┘
     └──────────────┘

              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────┴─────┐  ┌──────┴──────┐  ┌──────┴──────┐
     │  Tier 2       │  │  Tier 2      │  │  Tier 2      │  ...
     │  Slim         │  │  Slim        │  │  Slim        │
     │               │  │              │  │              │
     │  tester       │  │  reviewer    │  │  planner     │
     │  auditor      │  │              │  │              │
     │               │  │  (run)       │  │  (run)       │
     │  (run)        │  └──────────────┘  └──────────────┘
     └──────────────┘

  Tier 1: persistent sessions, domain-scoped tools, full models
  Tier 2: fire-and-forget runs, narrow tools, auto-cleanup
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `openclaw.json` | Agent definitions, tool overrides, subagent defaults |
| `workspace/AGENTS.md` | Lloyd's dispatch heuristics and coordination patterns |
| `agents/{id}/agent/tools.json` | Per-agent built-in tool collision disabling |
| `agents/{id}/agent/models.json` | Per-agent provider config (copied from main) |
| `agents/{id}/agent/auth-profiles.json` | Per-agent API keys (copied from main) |
| `agents/{id}/system-prompt.md` | Per-agent role definition and constraints |
| `docs/TOOL-ARCHITECTURE.md` | Reference: tool assembly pipeline and policy filtering |
| `docs/multi-tier-model-routing.md` | Reference: model routing strategy (local/sonnet/opus) |
