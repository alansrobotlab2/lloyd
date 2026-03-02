# OpenClaw Context Optimization Gameplan

## Problem Statement

The main agent's system prompt is ~20-23k tokens with ~46 tools on every turn, regardless of task type. This leads to:

1. **Tool confusion** — the LLM sees overlapping tools (3 ways to read files, 2 ways to run bash) and domain-specific tools (voice, ClawDeck) even for simple conversations
2. **Context bloat** — 9 skill files (~8,200 tokens) injected every turn when ~0 are used; AGENTS.md alone is 3,200 tokens of heartbeat/group-chat/coordination docs
3. **Latency** — more input tokens = slower TTFT; vault prefill runs on every turn including greetings
4. **Focus drift** — the LLM has so many instructions and tools it can lose track of what actually matters for a given task

## Current State

### Token Budget

| Component | Tokens | % of Total |
|-----------|--------|------------|
| Framework boilerplate | ~2,800 | 13% |
| SOUL.md | ~621 | 3% |
| IDENTITY.md + USER.md | ~34 | <1% |
| TOOLS.md | ~805 | 4% |
| MEMORY.md | ~1,545 | 7% |
| AGENTS.md | ~3,202 | 15% |
| **9 skill files** | **~8,196** | **38%** |
| Tool JSON schemas (~46 tools) | ~1,250 | 6% |
| Prefill context (variable) | ~0-2,000 | 0-9% |
| **Total** | **~18,450-22,450** | |

### Tool Inventory (~46 active)

| Layer | Count | Tools |
|-------|-------|-------|
| Built-in (already disabled) | 10 | read, write, edit, exec, process, apply_patch, web_search, web_fetch, memory_get, memory_search |
| Built-in (active) | ~15 | browser, canvas, nodes, cron, message, tts, gateway, agents_list, sessions_list, sessions_history, sessions_send, sessions_spawn, subagents, session_status, image |
| MCP tools | 18 | tag_search, tag_explore, vault_overview, qmd_search, qmd_get, memory_write, http_search, http_fetch, http_request, file_read, file_write, file_edit, file_patch, file_glob, file_grep, run_bash, bg_exec, bg_process |
| Voice tools | 3 | voice_last_utterance, voice_enroll_speaker, voice_list_speakers |
| ClawDeck | 6 | clawdeck_boards/tasks/next_task/get_task/update_task/create_task |

### Key Architectural Facts

- **Sub-agents already exist** — 8 specialists (memory, coder, researcher, operator, tester, reviewer, planner, auditor) with scoped tool allowlists, non-blocking spawn, auto-announce on completion
- **Sub-agents get "minimal" system prompts** — framework strips heartbeats, silent replies, messaging, voice, docs sections automatically
- **Sub-agents get filtered tool lists** — only their `tools.allow` set appears in the system prompt
- **Sub-agent workspaces exist** but are generic — all share Lloyd's identical SOUL.md (including TTS `<summary>` tags they shouldn't produce) and reference stale tool names
- **Prefill hook fires for ALL sessions** — coder gets vault search on "implement the login form" (useless)
- **Lloyd does everything himself** instead of delegating to the specialist infrastructure that's already built

---

## Phase 1: Skill Pruning + Quick Config Wins

**Goal**: Biggest token reduction with zero code changes. Config only.

### 1a: Disable 6 Skill Files (~7,000 tokens saved)

Skills are 38% of the system prompt. Most are procedural recipes used on <1% of turns.

**Keep loaded (always relevant):**
- `voice-mode/SKILL.md` (2,934 chars) — defines `<summary>` TTS tag pattern
- `websearch/SKILL.md` (2,534 chars) — knowledge capture loop is a core behavior

**Disable via `openclaw.json` → `skills.entries`:**

| Skill | Size | Used When |
|-------|------|-----------|
| claude-code-subagent | 8,392 chars | Launching Claude Code in tmux |
| obsidian-vault-maintenance | 4,880 chars | Periodic vault audit/rebuild |
| research-agent | 3,826 chars | Spawning researcher sub-agent |
| autolink | 3,337 chars | Vault auto-linking pass |
| youtube-transcript | 2,514 chars | YouTube summary tasks |
| restart-openclaw | 1,634 chars | Restarting the gateway |

(`voice-clone-sample` already disabled.)

**Fallback**: Add an "On-Demand Skills" section to TOOLS.md with file paths. Agent reads via `file_read` when needed — one extra tool call on rare turns.

### 1b: Disable 5 Unused Built-in Tools (~150 tokens saved)

Add to `agents/main/agent/tools.json`:
- `canvas` — zero references in workspace docs
- `nodes` — zero references
- `image` — rarely needed; re-enable via Mission Control when needed
- `gateway` — restart procedure uses CLI, not this tool
- `session_status` — debugging-only

### 1c: Clean Up MEMORY.md (~500 tokens saved)

Remove stale, duplicate, and sensitive entries:
- "OpenClaw update overdue" — stale
- HuggingFace tokens — **move to vault** (security: tokens shouldn't be in system prompt)
- Claude Code launch rules — duplicates the skill we're deferring
- QMD implementation details — operational, not needed per-turn

**Phase 1 total: ~7,650 tokens saved, 5 tools removed. ~20 min effort. Zero risk.**

### Files

| File | Change |
|------|--------|
| `openclaw.json` | Add 6 skill entries with `enabled: false` |
| `agents/main/agent/tools.json` | Add 5 tool entries with `false` |
| `~/obsidian/agents/lloyd/TOOLS.md` | Add "On-Demand Skills" reference list (~100 tokens) |
| `~/obsidian/agents/lloyd/MEMORY.md` | Remove stale/duplicate/sensitive entries |

---

## Phase 2: Orchestrator Architecture

**Goal**: Transform Lloyd from "do-everything with 41 tools" to "thin orchestrator with 18 tools that delegates to focused sub-agents." This phase combines tool reduction, AGENTS.md restructuring, sub-agent workspace fixes, and prefill gating into one cohesive architecture change.

### 2a: Trim Lloyd's Tool Set (41 → ~18 tools)

**Lloyd keeps (~18 tools):**

| Category | Tools | Why |
|----------|-------|-----|
| Conversation | `message`, `cron`, `tts` | Core interaction loop |
| Orchestration | `sessions_spawn`, `subagents`, `agents_list`, `sessions_list`, `sessions_history`, `sessions_send` | Manage sub-agents |
| Memory | `qmd_search`, `qmd_get`, `tag_search`, `memory_write` | Context before routing, daily notes |
| Quick actions | `file_read`, `file_glob`, `run_bash` | Trivial reads/checks without spawning a sub-agent |

**Delegate to sub-agents:**

| Tools removed from Lloyd | Delegate to |
|--------------------------|-------------|
| `file_write`, `file_edit`, `file_patch`, `file_grep` | coder |
| `bg_exec`, `bg_process` | operator |
| `http_search`, `http_fetch`, `http_request`, `browser` | researcher |
| `tag_explore`, `vault_overview` | memory |
| `voice_enroll_speaker`, `voice_list_speakers` | (keep `voice_last_utterance` only) |
| `clawdeck_*` (all 6) | operator |

Implementation: add these as `false` entries in `agents/main/agent/tools.json`.

### 2b: Restructure Lloyd's AGENTS.md

Current AGENTS.md is ~3,200 tokens with sections that are either redundant, bloated, or belong elsewhere. Restructure it around Lloyd's new role as orchestrator.

**Remove or relocate:**

| Section | Tokens | Action |
|---------|--------|--------|
| Heartbeat procedures (lines 196-276) | ~800 | Extract to `heartbeat/SKILL.md` (disabled by default) |
| ClawDeck docs (lines 155-177) | ~250 | Remove — tool descriptions are self-documenting; operator handles this |
| Reactions guide (lines 138-152) | ~150 | Remove — common sense the model already knows |
| Coordination pattern examples (lines 319-338) | ~250 | Remove — keep roster table only |

**Add:**

```
## Task Routing
- Multi-file code changes → spawn coder
- Web research / "look up" / "search for" → spawn researcher
- Git, deploys, ClawDeck tasks → spawn operator
- Vault maintenance, bulk knowledge ops → spawn memory
- Quick reads, simple questions, conversation → handle directly
- Rule of thumb: if it needs >2 tool calls, delegate it
```

**Compress group chat section** from ~250 tokens to ~100 tokens (essentials only).

Target: AGENTS.md from ~3,200 → ~1,500 tokens.

### 2c: Fix Sub-Agent Workspace Files

**Problem 1: SOUL.md is identical across all 9 agents** (2,498 bytes each), including TTS `<summary>` tag instructions that sub-agents shouldn't produce. Sub-agents send results to Lloyd, not to the user.

**Fix**: Replace each sub-agent's SOUL.md with a lean version (~800 chars) that strips:
- TTS `<summary>` tag instructions
- "verbal AI assistant" framing
- Group chat / external action cautions

Keep: personality traits, resourcefulness, boundaries, honesty.

**Problem 2: Sub-agent AGENTS.md files reference stale tool names.**
- Researcher says `web_search`, `web_fetch`, `memory_search`, `memory_get`
- Actual MCP names: `http_search`, `http_fetch`, `qmd_search`, `qmd_get`

**Fix**: Update all 8 sub-agent AGENTS.md files with correct MCP tool names.

### 2d: Smart Prefill Gating

Two changes to `extensions/mcp-tools/index.ts`:

**Conversational skip** — don't vault-search for greetings and acknowledgments:
```typescript
const SKIP_PATTERNS = /^(yes|no|ok|sure|thanks|yep|nah|do it|go ahead|sounds good)/i;
if (prompt.length < 40 && SKIP_PATTERNS.test(prompt.trim())) return;
```

**Agent-aware skip** — don't vault-search for specialist sub-agents that don't need it:
```typescript
const agentId = ctx?.agentId ?? "main";
const skipAgents = new Set(["coder", "tester", "reviewer", "auditor", "operator"]);
if (skipAgents.has(agentId)) return;
```

### Phase 2 Impact

- Lloyd: 41 → ~18 tools (~56% tool reduction)
- AGENTS.md: ~3,200 → ~1,500 tokens (~1,700 saved)
- Sub-agents: correct tool names, lean SOUL.md, no wasted prefill
- Latency: ~200-500ms saved per sub-agent turn + conversational turns

**Phase 2 total: ~2,200 tokens saved on Lloyd, ~23 tools removed. ~2 hours effort. Medium risk.**

**Risk mitigation**: Lloyd keeps `file_read`, `file_glob`, `run_bash` for quick one-off actions. If the LLM struggles to delegate, we can add tools back incrementally.

### Files

| File | Change |
|------|--------|
| `agents/main/agent/tools.json` | Disable ~23 more tools (file/web/bg/voice/clawdeck) |
| `~/obsidian/agents/lloyd/AGENTS.md` | Restructure: compress, add routing table, extract heartbeat |
| `~/obsidian/agents/lloyd/skills/heartbeat/SKILL.md` | New file — extracted heartbeat procedures |
| `~/obsidian/agents/{coder,memory,researcher,operator,tester,reviewer,planner,auditor}/SOUL.md` | Lean version without TTS tags |
| `~/obsidian/agents/{coder,memory,researcher,operator,tester,reviewer,planner,auditor}/AGENTS.md` | Fix stale tool names |
| `extensions/mcp-tools/index.ts` | Add conversational skip + agent-aware skip to prefill hook |

---

## Phase 3: Context-Profile Router

**Goal**: Extend the model-router to classify a "context profile" alongside the model tier. The profile controls what the prefill hook injects.

**Why**: Even after Phases 1-2, the system prompt is still one-size-fits-all. The model-router already classifies intent — it just needs to output an additional signal.

### Context Profiles

| Profile | Trigger Signals | Prefill Behavior |
|---------|----------------|------------------|
| `chat` | Short prompt, greeting, yes/no, follow-up | Skip entirely |
| `memory` | "remember", "what did we", vault keywords | Full pipeline (tags + BM25 + GLM) |
| `code` | Code blocks, "implement", "debug", "fix" | Skip vault |
| `research` | "search for", "look up", "what is" | BM25 only (skip GLM for speed) |
| `ops` | "restart", "deploy", ClawDeck keywords | Skip vault |

### Implementation

**Step 1**: Add `profile` field to model-router's `RoutingDecision`. Classify using the same regex patterns that already determine tier.

**Step 2**: Export the current profile via a module-level variable or shared temp file that mcp-tools can read.

**Step 3**: Make the prefill hook profile-aware — skip or customize based on profile.

**Step 4 (future)**: Use `before_agent_start`'s `systemPrompt` override to compose profile-specific system prompts. Deferred — requires reconstructing framework boilerplate.

**Phase 3 total: ~0-2,000 tokens saved (relevance improvement). ~2-4 hours effort. Medium risk.**

### Files

| File | Change |
|------|--------|
| `extensions/model-router/index.ts` | Add context profile classification + export |
| `extensions/mcp-tools/index.ts` | Read profile, conditionally customize prefill |

---

## Future Considerations

### Local Model Prompt Size
When model-router routes to local Qwen (35B at ~160 tps), the system prompt is a bigger relative burden. Simple tasks routed to local could benefit from a further-stripped prompt via `systemPrompt` override. High risk (requires reconstructing boilerplate). Defer unless local latency becomes a bottleneck.

### Compaction Strategy
Current `compaction.mode: "safeguard"` waits until hitting context limits. Consider `"aggressive"` for long sessions. Trade-off: loses conversational nuance. Worth testing.

### Framework Boilerplate Audit
~2,800 tokens of framework-generated content includes ~250 tokens of rarely-used sections (CLI reference, model aliases, self-update). Can only be trimmed via `systemPrompt` override. Low priority.

---

## Summary

| Phase | Token Savings | Tool Reduction | Effort | Risk |
|-------|--------------|----------------|--------|------|
| 1: Skill pruning + config wins | ~7,650 | 5 | ~20 min | Low |
| 2: Orchestrator architecture | ~2,200 | ~23 | ~2 hours | Medium |
| 3: Context-profile router | ~0-2,000 (relevance) | 0 | ~2-4 hours | Medium |
| **Phases 1-2** | **~9,850** | **~28** | **~2.5 hours** | **Low-Medium** |
| **All phases** | **~9,850-11,850** | **~28** | **~5-7 hours** | **Medium** |

**Before**: ~20-23k tokens, ~46 tools
**After Phase 1**: ~11-14k tokens, ~41 tools
**After Phase 2**: ~9-12k tokens, ~18 tools on Lloyd, focused sub-agents
**After Phase 3**: ~9-12k tokens, ~18 tools, profile-aware context routing

---

## Verification

After each phase, restart gateway and test:

1. **Baseline**: Send greeting, code question, vault recall, YouTube transcript request
2. **Token check**: `logs/timing.jsonl` for reduced input token counts
3. **Delegation** (Phase 2): "implement a login form" → Lloyd spawns coder, doesn't try to file_edit itself
4. **Quick actions** (Phase 2): "what's in ~/foo.txt" → Lloyd reads directly, no sub-agent spawn
5. **Deferred skills** (Phase 1): "run vault maintenance" → agent reads skill file first
6. **Sub-agent hygiene** (Phase 2): coder logs show no vault prefill, no `<summary>` tags in output
7. **Tool list**: Mission Control `/mc/` shows correct tools per session

---

## Implementation Results (2026-03-02)

All 3 phases implemented and verified.

### Phase 1 Results
- 7 skills disabled (6 deferred + 1 heartbeat extracted from AGENTS.md)
- 5 built-in tools disabled (canvas, nodes, image, gateway, session_status)
- MEMORY.md cleaned: removed stale entries, moved HF tokens to vault
- TOOLS.md updated with on-demand skills reference list
- **Verified**: 38 enabled tools (down from ~46)

### Phase 2 Results
- 20 additional tools disabled in tools.json → **18 enabled tools** for Lloyd
- AGENTS.md compressed from 345 lines (~3,200 tokens) to ~105 lines (~1,200 tokens)
- Heartbeat section extracted to `skills/heartbeat/SKILL.md` (disabled)
- All 8 sub-agent SOUL.md files replaced with lean version (no TTS `<summary>` tags)
- All 8 sub-agent AGENTS.md files updated with correct MCP tool names
- Prefill gating added: agent-aware skip + conversational pattern skip
- **Verified**: Gateway restart clean, 18 tools confirmed via Mission Control API

### Phase 3 Results
- Context-profile router implemented directly in mcp-tools `before_prompt_build` hook
- 7 profiles: chat, memory, code, research, ops, voice, default
- Prefill skipped for: chat, code, ops, voice (no vault search needed)
- Prefill runs for: memory, research, default (vault context valuable)
- **Design note**: Placed in mcp-tools (not model-router) because `before_prompt_build` fires before `before_agent_start`
- **Verified**: Gateway restart clean, no compile errors

### Final State
- **Tools**: 18 enabled (was ~46) — 61% reduction
- **System prompt tokens**: ~11-12k estimated (was ~20-23k) — ~45% reduction
- **Prefill latency**: Eliminated for ~60% of turns (chat/code/ops/voice profiles + sub-agents)
- **AGENTS.md**: ~1,200 tokens (was ~3,200) — 63% reduction

### Files Modified
| File | Change |
|------|--------|
| `openclaw.json` | 7 skills disabled |
| `agents/main/agent/tools.json` | 35 tools disabled (15 prior + 20 new) |
| `~/obsidian/agents/lloyd/AGENTS.md` | Compressed, heartbeat extracted |
| `~/obsidian/agents/lloyd/TOOLS.md` | Added on-demand skills reference |
| `~/obsidian/agents/lloyd/MEMORY.md` | Cleaned stale entries |
| `~/obsidian/agents/lloyd/skills/heartbeat/SKILL.md` | New — extracted from AGENTS.md |
| `~/obsidian/lloyd/knowledge/software/huggingface-tokens.md` | New — moved tokens out of system prompt |
| `~/obsidian/agents/{8 sub-agents}/SOUL.md` | Lean version without TTS tags |
| `~/obsidian/agents/{8 sub-agents}/AGENTS.md` | Updated to correct MCP tool names |
| `extensions/mcp-tools/index.ts` | Agent-aware + profile-based prefill gating |
