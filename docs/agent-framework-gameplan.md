# Agent Framework Gameplan: Delegation-First Architecture

**Date:** 2026-03-03
**Type:** Architecture
**Status:** Complete ✅

## Context

Lloyd currently operates as a "smart generalist" — 18 tools, handles web lookups, knowledge capture, file reads, and bash commands directly, then delegates larger tasks to 8 specialized sub-agents. But the delegation is ad-hoc: no standard pipelines, no feedback loops, sub-agents work in isolation, and several have tool confusion (built-in + MCP duplicates).

**Key problem:** When Lloyd orchestrates a multi-step pipeline (plan → code → test → review), he's blocked from conversation. A 20-minute code pipeline means 20 minutes of Lloyd being unavailable. Lloyd is a personal assistant — he should pass off long-horizon work and stay present.

**Solution:** Add an **Orchestrator** agent that manages multi-step pipelines autonomously. Lloyd dispatches complex tasks to Orchestrator and stays free to chat. Orchestrator chains specialists (planner → coder → tester → reviewer) at depth 2 using `maxSpawnDepth: 2`.

---

## Architecture: 10 Agents

### Lloyd (main) — UI + Personal Assistant
- Personality, voice, TTS, conversation
- Quick direct actions (single reads, bash checks, memory lookups, quick web search)
- Dispatches multi-step tasks to **Orchestrator**
- Can still spawn individual agents for simple single-agent tasks

### Orchestrator (NEW) — Pipeline Manager
- Manages multi-step workflows: Code Pipeline, Research Pipeline, Security Pipeline
- Spawns and chains specialists at depth 2
- Runs in background while Lloyd stays available
- Reports combined results back to Lloyd on completion

### 8 Specialists (unchanged roles)
- **Memory** — vault specialist (Qwen/Sonnet)
- **Coder** — code orchestrator → Claude Code (Opus)
- **Researcher** — web research + knowledge capture (Sonnet)
- **Operator** — system ops, git, services, ClawDeck (Sonnet)
- **Tester** — write + run tests (Sonnet, fire-and-forget)
- **Reviewer** — code review, read-only (Sonnet, fire-and-forget)
- **Planner** — task breakdown, read-only (Sonnet, fire-and-forget)
- **Auditor** — security scanning, read-only (Sonnet, fire-and-forget)

---

## Spawn Topology

```
Lloyd (depth 0)
├── Orchestrator (depth 1, persistent) — multi-step pipelines
│   ├── Planner (depth 2)
│   ├── Coder (depth 2)
│   ├── Tester (depth 2)
│   ├── Reviewer (depth 2)
│   ├── Auditor (depth 2)
│   ├── Researcher (depth 2)
│   └── Memory (depth 2)
├── Memory (depth 1) — quick vault ops
├── Researcher (depth 1) — quick research
├── Operator (depth 1) — quick sys ops
└── Coder (depth 1) — simple code tasks
```

Lloyd can spawn agents directly at depth 1 for simple single-agent tasks (no pipeline needed). For multi-step work, Orchestrator manages the pipeline at depth 1, spawning specialists at depth 2.

---

## Lloyd's Task Routing

### Handle Directly
- Conversation, greetings, voice responses
- Quick file reads (`file_read`), bash checks (`run_bash`)
- Memory lookups (`qmd_search`, `qmd_get`, `tag_search`)
- Daily note updates (`memory_write`)
- Quick web searches (`http_search`)
- Scheduling (`cron`)

**Rule: if it needs >2 tool calls in the same domain, delegate.**

### Dispatch to Orchestrator (multi-step)
- Code changes that need planning + coding + testing + review
- Research that needs web search + knowledge capture + synthesis
- Any task requiring multiple agents in sequence
- "Implement X", "research and document Y", "audit and fix Z"

### Dispatch to Single Agent (simple delegation)
- "What's in the vault about X?" → Memory
- "Restart the gateway" → Operator
- "Quick fix: change the button color in file.tsx" → Coder (single-phase)

---

## Orchestrator Pipelines

### Code Pipeline
```
1. Spawn planner("plan: {task}")          ← understand scope
2. Spawn coder("[Phase: Execute] {task} per plan: {plan}")  ← implement
3. Spawn tester + reviewer IN PARALLEL    ← validate
4. IF reviewer finds Critical issues:
   Spawn coder("[Phase: Fix] {issues}")   ← fix
5. Report combined results to Lloyd
```

### Research Pipeline
```
1. Spawn researcher("Research {question}. Save findings to vault.")
   ← researcher checks vault, searches web, saves to ~/obsidian/lloyd/knowledge/
2. Report findings to Lloyd
```

### Security Pipeline
```
1. Spawn auditor("security scan: {files}")
2. IF Critical findings:
   Spawn coder("[Phase: Fix] fix: {findings}")
3. Report to Lloyd
```

### Full Feature Pipeline (plan + code + test + review + audit)
```
1. Planner → 2. Coder → 3. Tester + Reviewer + Auditor (parallel) → 4. Fix loop → 5. Report
```

---

## Tool Changes

### Lloyd: re-enable `http_search`, keep lean
**Final tools (~18):**
- Orchestration: `sessions_spawn, subagents, agents_list, sessions_list, sessions_history, sessions_send`
- Conversation: `message, cron, tts, voice_last_utterance`
- Memory: `qmd_search, qmd_get, tag_search, memory_write`
- Quick actions: `file_read, file_glob, run_bash`
- Web (quick): `http_search`

### Orchestrator (NEW): pipeline management tools
**Tools:**
- Orchestration: `sessions_spawn, subagents, sessions_list, sessions_history, sessions_send`
- Context: `file_read, file_glob, run_bash, qmd_search, qmd_get`

### Coder: 22 → 14 tools (trim non-coding)
Remove: `http_search, http_fetch, http_request, tag_explore, vault_overview, memory_write, clawdeck_boards, file_patch`

### Researcher: gain `memory_write`
Add: `memory_write` (direct knowledge capture to vault)
Replace: built-in `read` → MCP `file_read`

### Memory: gain `file_read`, `file_glob`
Replace: built-in `read` → MCP `file_read, file_glob`

### Tester: standardize on MCP
Replace: `read, write, edit, exec` → `file_read, file_write, file_edit, run_bash`

### Operator: standardize on MCP
Remove built-in `exec, process, read` from allow list

### Reviewer, Planner, Auditor: remove built-in `read`

---

## Block Built-in Tools on ALL Sub-Agents

Add to every sub-agent's `tools.json`:
```json
{
  "read": false, "write": false, "edit": false, "exec": false,
  "process": false, "apply_patch": false, "image": false, "browser": false,
  "canvas": false, "nodes": false, "gateway": false, "session_status": false,
  "memory_search": false, "memory_get": false, "web_search": false, "web_fetch": false
}
```

---

## Model Assignments

| Agent | Model | Rationale |
|-------|-------|-----------|
| Lloyd | Sonnet | Conversation + routing |
| Orchestrator | Sonnet | Pipeline management |
| Memory | Qwen local / Sonnet fallback | Vault ops, cheap |
| Coder | **Opus** | Gameplans, code review, quality-sensitive |
| Researcher | Sonnet (was Opus) | Search + fetch orchestration |
| Operator | Sonnet | System ops |
| Tester | Sonnet | Test execution |
| Reviewer | Sonnet | Code review |
| Planner | Sonnet | Task breakdown |
| Auditor | Sonnet | Security scanning |

---

## Files to Modify

| File | Change |
|------|--------|
| `openclaw.json` | Add orchestrator agent; update tool allow lists; Researcher → Sonnet; disable websearch skill |
| `agents/main/agent/tools.json` | Re-enable `http_search` |
| `agents/orchestrator/agent/tools.json` | NEW — block all built-in tools |
| `agents/{8 sub-agents}/agent/tools.json` | Block remaining built-in tools |
| `~/obsidian/agents/orchestrator/SOUL.md` | NEW — lean personality |
| `~/obsidian/agents/orchestrator/AGENTS.md` | NEW — pipeline definitions |
| `~/obsidian/agents/lloyd/AGENTS.md` | Rewrite with Orchestrator dispatch |
| `~/obsidian/agents/coder/AGENTS.md` | Fix tool section to MCP-only |
| `~/obsidian/agents/researcher/AGENTS.md` | Add knowledge capture workflow |
| `~/obsidian/agents/memory/AGENTS.md` | Add file_read/file_glob |
| `~/obsidian/agents/{tester,operator}/AGENTS.md` | Update tool refs |

---

## Verification

1. **Gateway restart**: Clean startup, no errors
2. **Tool isolation**: Sub-agents show ZERO built-in tool calls
3. **Orchestrator pipeline**: "Implement feature X" → Lloyd spawns Orchestrator → Orchestrator chains planner → coder → tester + reviewer → reports back
4. **Lloyd stays available**: While Orchestrator runs, Lloyd can answer questions
5. **Quick actions**: "What's in ~/foo.txt?" → Lloyd handles directly, no spawn
6. **Quick web**: Simple fact question → Lloyd uses http_search
7. **Research pipeline via orchestrator**: Complex research → Orchestrator spawns researcher → researcher saves to vault

---

## Implementation Results (2026-03-03)

### What Was Built

All items in "Files to Modify" were completed. Key additions:

- **`agents/orchestrator/`** — new agent workspace with SOUL.md, AGENTS.md, tools.json
- **`docs/agent-framework-gameplan.md`** — this document
- All 9 sub-agent `tools.json` files updated to block built-in tools
- `openclaw.json` fully updated (orchestrator definition, all tool allow lists, Researcher model, websearch skill disabled)
- Lloyd's AGENTS.md fully rewritten with Orchestrator dispatch rules
- All sub-agent AGENTS.md files updated with correct MCP tool names

### Issues Found and Fixed

**Issue 1: Orchestrator "aborted" in first test**
- Root cause: one-shot CLI (`openclaw agent -m`) exits after Lloyd's first reply, cascading an abort to child sessions
- Not a real bug — in interactive gateway sessions Lloyd stays alive and the full pipeline completes
- Verified by reading deleted session JSONL: Planner completed and announced results; Orchestrator received them; main session showed "Test complete!" with all 10 agent directories listed

**Issue 2: Orchestrator did content work directly (first research test)**
- Root cause: `tools.allow` in openclaw.json is additive, not a whitelist — globally-enabled MCP tools (`http_search`, `memory_write`) still reach the Orchestrator even if not listed
- The original Constraints section said "Do NOT do implementation work yourself" but the model didn't classify research as implementation
- Fix: Rewrote Constraints section with explicit prohibitions — "NEVER call `http_search`, `memory_write`, etc. — those are for specialist agents. If you need research done, spawn `researcher`."
- After fix: Orchestrator immediately spawned Researcher on next research test (zero web calls of its own)

### Test Results

| Test | Pipeline | Outcome |
|------|----------|---------|
| Spawn chain validation | Lloyd → Orchestrator → Planner | ✅ All 3 sessions appeared; Planner listed 10 agent directories |
| Lloyd direct: memory | `qmd_search` — Alfie vault notes | ✅ 10 results returned, no delegation |
| Lloyd direct: file read | `file_read` — orchestrator SOUL.md | ✅ Direct read, no spawn |
| Lloyd direct: web search | `http_search` — Node.js version | ✅ Answered inline, no Orchestrator |
| Research Pipeline | Lloyd → Orchestrator → Researcher → vault | ✅ `hardware/rp2040.md` written (8.7 KB, proper frontmatter, 9 sources) |
| Code Pipeline | Lloyd → Orchestrator → Planner → Coder (Opus) | ✅ `TODO.md` created in orchestrator workspace |

### Observed Behavior

**Research Pipeline trace (verified in session JSONL):**
```
Lloyd:        sessions_spawn({ agentId: "orchestrator", task: "Research Pipeline: RP2040..." })
Orchestrator: [thinking: "Let me spawn a researcher agent to handle this task."]
Orchestrator: sessions_spawn({ agentId: "researcher", runtime: "subagent", task: "Research RP2040..." })
Researcher:   http_search × 3, memory_write → hardware/rp2040.md
Orchestrator: [receives completion] → formats Pipeline report → announces to Lloyd
Lloyd:        relays summary to user
```

**Code Pipeline trace (verified in sessions list):**
```
Lloyd:        sessions_spawn({ agentId: "orchestrator", task: "Code Pipeline: create TODO.md..." })
Orchestrator: sessions_spawn({ agentId: "planner", ... })     ← Planner at depth 2
Planner:      [completes plan, announces back]
Orchestrator: sessions_spawn({ agentId: "coder", model: "claude-opus-4-6", ... })  ← Coder at depth 2
Coder:        file_write → obsidian/agents/orchestrator/TODO.md
Orchestrator: [receives completion] → announces to Lloyd
```

**Coder confirmed on Opus** — session listing showed `claude-opus-4-6` for the coder session.

### Remaining Known Limitation

`http_fetch` and `http_request` are disabled via Mission Control globally — this affects the Researcher's ability to pull full page content. Researcher falls back to `http_search` snippets + model knowledge. This is a pre-existing Mission Control configuration, not introduced by this work. The Researcher's output noted this transparently in its completion message.
