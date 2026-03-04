# Agent SDK Migration: OpenClaw Subagents → Claude Code Instances

**Date:** 2026-03-04
**Status:** Approved, Phase 1 in progress
**Related:** [SUBAGENTS-IMPLEMENTATION.md](SUBAGENTS-IMPLEMENTATION.md), [agent-framework-gameplan.md](agent-framework-gameplan.md)

---

## Problem

OpenClaw's subagent system uses its own LLM loop for all 10 agents (main, orchestrator, memory, coder, researcher, operator, tester, reviewer, planner, auditor). This produces noticeably lower-quality results than Claude Code's agentic loop, which has adaptive thinking, better tool execution, session persistence, and context management. The coder agent already works around this by wrapping `claude --print` via `bg_exec` — a fragile tmux-based approach.

## Solution

Replace the subagent system with the **Claude Agent SDK** (`@anthropic-ai/claude-agent-sdk`), which exposes Claude Code's full capabilities as a programmatic `query()` API. Each agent becomes a Claude Code instance with a specialized persona, tool set, and model selection.

## Architecture

```
Lloyd (OpenClaw main agent — stays as-is)
  │
  ├── cc_orchestrate(task, pipeline, budget)
  │     └── Agent SDK query() — Orchestrator instance
  │           ├── Task("coder", ...)      ← Opus, full file tools + MCP
  │           ├── Task("reviewer", ...)   ← Opus, read-only (parallel with tester)
  │           ├── Task("tester", ...)     ← Sonnet, Bash + file tools
  │           └── Task("coder", ...)      ← fix cycle if needed
  │
  ├── cc_orchestrate(task2, pipeline2)    ← multiple orchestrators in parallel
  │     └── query() — Orchestrator #2
  │
  └── cc_spawn(task, agent: "coder")      ← direct single-agent, no orchestrator
        └── query() — standalone worker
```

**Three layers:**
1. **Lloyd** (Layer 1) — user-facing, stays on OpenClaw. Spawns orchestrators via plugin tools.
2. **Orchestrator** (Layer 2) — a Claude Code instance (top-level `query()`) that plans and coordinates workers via the Task tool.
3. **Workers** (Layer 3) — Agent SDK subagents (`AgentDefinition`) invoked by the orchestrator. Each has specialized prompt, tools, and model.

**Key constraint:** Agent SDK subagents cannot spawn their own subagents. The orchestrator must be the top-level `query()` so it can invoke workers via Task.

## Implementation Vehicle

New OpenClaw plugin: `extensions/agent-orchestrator/`

### Tools for Lloyd

| Tool | Behavior |
|------|----------|
| `cc_orchestrate` | Spawn orchestrator with pipeline template + all worker agents. Returns instance ID immediately (async). |
| `cc_spawn` | Spawn a single Claude Code instance for simple tasks. Returns instance ID immediately (async). |
| `cc_status` | Check status/cost/activity of running instances. |
| `cc_result` | Get full result text from completed instance. |
| `cc_abort` | Abort a running instance. |

### Lifecycle

Fire-and-forget with notification:
1. Tool returns instance ID immediately — Lloyd stays free for conversation
2. Plugin runs `query()` in background, tracking status/cost/messages
3. On completion, plugin fires OpenClaw system event notification
4. Lloyd can check progress anytime via `cc_status`
5. Multiple orchestrators can run simultaneously

### MCP Bridging

Workers access existing MCP tools (vault, web, backlog) via SSE connection to `tool_services.py` on `:8093`. No new backend needed. MCP tool names follow `mcp__openclaw-tools__<tool>` convention inside Agent SDK.

Workers also get Claude Code's built-in tools: Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch.

### Agent Definitions

Each agent is an `AgentDefinition` object with:
- `prompt` — migrated from SOUL.md (shared personality) + AGENTS.md (role-specific workflow)
- `tools` — mapped from current allowlists to Claude Code built-in names + MCP tool names
- `model` — Opus for coder/reviewer/auditor, Sonnet for orchestrator/researcher/tester/planner/operator
- `maxTurns` — per-agent safety limit (10-50 depending on complexity)

### Memory Agent Exception

The memory agent stays on local Qwen 3.5-35B via the existing `sessions_spawn` system. Simple vault operations (search, read, write) are handled by MCP tool calls directly from workers — no dedicated agent needed. Complex knowledge tasks route through the Qwen memory agent for zero API cost.

### Cost Control

- `maxBudgetUsd` at orchestrator level (default $5.00)
- `maxTurns` per worker agent
- Lloyd can abort any instance via `cc_abort`

## Mission Control Changes

- New "Claude Code Instances" section on Activity page
- Instance cards: task, status, elapsed time, cost, turns, abort button
- Instance log viewer: streaming messages per instance
- API endpoints: `/api/mc/cc-instances`, `/api/mc/cc-instance-log`, `/api/mc/cc-instance-abort`

## Phased Rollout

| Phase | Scope | Milestone |
|-------|-------|-----------|
| 1 | Core plugin + `cc_spawn` + reviewer agent POC | Single agent runs, MCP tools work |
| 2 | `cc_orchestrate` + all 8 worker agents + pipeline templates | Full code/research/security pipelines |
| 3 | Mission Control integration | Real-time instance visibility in browser |
| 4 | Agent-by-agent migration + dual-mode operation | Both systems run in parallel, quality comparison |
| 5 | Deprecation of legacy subagent system | Remove `sessions_spawn` routing, archive old configs |

Migration order (simplest first): reviewer → auditor → planner → researcher → memory config → tester → operator → coder

## Key Dependencies

- `@anthropic-ai/claude-agent-sdk` npm package
- `ANTHROPIC_API_KEY` environment variable
- `tool_services.py` running on `:8093` (existing, unchanged)
- OpenClaw plugin SDK (`api.registerTool`, `api.registerHttpRoute`)

## Files

### New (extensions/agent-orchestrator/)
- `index.ts` — plugin entry, tool registration, lifecycle, MC API endpoints
- `types.ts` — CcInstance, status types
- `orchestrator-prompt.ts` — orchestrator system prompt with pipeline templates
- `query-consumer.ts` — async generator consumer, status/cost tracking
- `agents/*.ts` — one file per agent definition + barrel export
- `package.json` — dependency on `@anthropic-ai/claude-agent-sdk`

### Modified
- `openclaw.json` — register plugin, add tools to Lloyd's allowlist
- `extensions/mission-control/web/src/api.ts` — CC instance API types
- `extensions/mission-control/web/src/components/pages/ActivityPage.tsx` — instance cards
- `~/obsidian/agents/lloyd/AGENTS.md` — update routing to prefer `cc_orchestrate`

## Comparison: Before vs After

| Aspect | Before (sessions_spawn) | After (Agent SDK) |
|--------|------------------------|-------------------|
| LLM quality | OpenClaw's basic loop | Claude Code's full agentic loop |
| Tool execution | MCP proxy only | Claude Code built-in + MCP |
| File editing | Fragile `claude --print` wrapper | Native Read/Write/Edit |
| Concurrency | Max 8 via OpenClaw scheduler | Multiple `query()` instances |
| Session persistence | OpenClaw JSONL sessions | Agent SDK session management |
| Context management | Manual prefill hooks | Claude Code's adaptive context |
| Cost visibility | Per-session token counting | `maxBudgetUsd` + real-time tracking |
| Nesting depth | 2 levels (main→tier1→tier2) | 2 levels (orchestrator→workers) |
| Memory agent | Local Qwen (free) | Stays on local Qwen (hybrid) |
