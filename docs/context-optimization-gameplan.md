# OpenClaw Context Optimization Gameplan

## Problem Statement

The main agent's system prompt is ~20-23k tokens with ~46 tools on every turn, regardless of task type. This leads to:

1. **Tool confusion** — the LLM sees overlapping tools (3 ways to read files, 2 ways to run bash) and domain-specific tools (voice, Backlog) even for simple conversations
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
| MCP tools | 18 | tag_search, tag_explore, vault_overview, mem_search, mem_get, mem_write, http_search, http_fetch, http_request, file_read, file_write, file_edit, file_patch, file_glob, file_grep, run_bash, bg_exec, bg_process |
| Voice tools | 3 | voice_last_utterance, voice_enroll_speaker, voice_list_speakers |
| Backlog | 6 | backlog_boards/tasks/next_task/get_task/update_task/create_task |

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
| Memory | `mem_search`, `mem_get`, `tag_search`, `mem_write` | Context before routing, daily notes |
| Quick actions | `file_read`, `file_glob`, `run_bash` | Trivial reads/checks without spawning a sub-agent |

**Delegate to sub-agents:**

| Tools removed from Lloyd | Delegate to |
|--------------------------|-------------|
| `file_write`, `file_edit`, `file_patch`, `file_grep` | coder |
| `bg_exec`, `bg_process` | operator |
| `http_search`, `http_fetch`, `http_request`, `browser` | researcher |
| `tag_explore`, `vault_overview` | memory |
| `voice_enroll_speaker`, `voice_list_speakers` | (keep `voice_last_utterance` only) |
| `backlog_*` (all 6) | operator |

Implementation: add these as `false` entries in `agents/main/agent/tools.json`.

### 2b: Restructure Lloyd's AGENTS.md

Current AGENTS.md is ~3,200 tokens with sections that are either redundant, bloated, or belong elsewhere. Restructure it around Lloyd's new role as orchestrator.

**Remove or relocate:**

| Section | Tokens | Action |
|---------|--------|--------|
| Heartbeat procedures (lines 196-276) | ~800 | Extract to `heartbeat/SKILL.md` (disabled by default) |
| Backlog docs (lines 155-177) | ~250 | Remove — tool descriptions are self-documenting; operator handles this |
| Reactions guide (lines 138-152) | ~150 | Remove — common sense the model already knows |
| Coordination pattern examples (lines 319-338) | ~250 | Remove — keep roster table only |

**Add:**

```
## Task Routing
- Multi-file code changes → spawn coder
- Web research / "look up" / "search for" → spawn researcher
- Git, deploys, Backlog tasks → spawn operator
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
- Actual MCP names: `http_search`, `http_fetch`, `mem_search`, `mem_get`

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
| `agents/main/agent/tools.json` | Disable ~23 more tools (file/web/bg/voice/backlog) |
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
| `ops` | "restart", "deploy", Backlog keywords | Skip vault |

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

---

## Phase 4: Sub-Agent Context Audit (2026-03-03)

### Background

Phases 1-3 optimized Lloyd's context from ~46 tools / ~20-23k tokens down to 18 tools / ~11-12k tokens with profile-aware prefill gating. This phase audits what actually happened in production after those changes, focusing on sub-agents (especially the coder) which were touched in Phase 2 but not deeply profiled.

### Methodology

- Analyzed active coder session `7a591493` and 27 deleted coder sessions from the past ~4 hours
- Reviewed timing log entries in `logs/timing.jsonl`
- Inspected all 8 sub-agent `tools.json` files, workspace files, and skill snapshots
- Traced tool inheritance through `openclaw.json` config and built-in framework defaults
- Measured skill payload sizes from bundled and workspace skill directories
- Compared token usage across Sonnet, Opus, and local Qwen sessions

---

### Finding 1: Coder Tool Confusion (CRITICAL — active cost burn)

**Expected state after Phase 2**: Coder has 22 MCP tools in its `allow` list, built-in tools blocked.

**Actual state**: The coder's `tools.json` only blocks 4 old renamed tools:
```json
// agents/coder/agent/tools.json
{
  "memory_search": false,
  "memory_get": false,
  "web_search": false,
  "web_fetch": false
}
```

Built-in tools (`read`, `write`, `edit`, `exec`, `process`, `apply_patch`, `image`, `browser`, `canvas`, `nodes`, etc.) are NOT blocked. They inherit from the framework and appear alongside MCP equivalents in the tool list. The coder actually has **~30+ tools**, not the 22 in the allowlist.

**Proof from session `7a591493`** (coder on Opus 4.6 at $75/Mtok output):

1. Model calls `file_write` (MCP) → disabled via Mission Control
2. Model falls back to `write` (built-in) → disabled via Mission Control
3. Model falls back to `run_bash` with heredoc → succeeds

This triple-attempt pattern costs ~1,800 output tokens per file write at Opus rates.

**Evidence from deleted sessions** (tool call counts):

| Session | Size | run_bash | file_read | read | file_write | write | file_edit | edit | exec | image |
|---------|------|----------|-----------|------|------------|-------|-----------|------|------|-------|
| db969e5b | 179K | 45 | — | — | — | — | — | — | — | — |
| a4234718 | 127K | 19 | — | — | — | — | 1 | 1 | — | — |
| 5855256a | 78K | 4 | 1 | 1 | 1 | 1 | — | — | 1 | 1 |
| d7d9a4cd | 77K | 12 | 4 | 1 | 1 | 1 | 1 | — | — | 2 |
| 594d37e6 | 64K | 12 | 3 | — | 1 | 1 | — | — | 1 | — |

Tool confusion is pervasive — the model uses both built-in (`read`/`write`/`edit`/`exec`) and MCP (`file_read`/`file_write`/`file_edit`/`run_bash`) versions interchangeably.

**Root cause**: The coder's AGENTS.md (lines 188-191) explicitly lists BOTH sets:
```
- `run_bash`, `bg_exec`, `bg_process` — shell commands
- `read`, `edit`, `write` — base file operations
- `file_read`, `file_write`, `file_edit`, `file_patch`, `file_glob`, `file_grep` — MCP file operations
- `exec`, `process`, `apply_patch` — execution and patching
```

The model sees both documented and tries both. When one fails, it tries the other, then falls back to `run_bash`.

---

### Finding 2: Bundled Skills Injected into All Sub-Agents (~52KB / ~13k tokens)

**Expected state**: Sub-agents get lean prompts focused on their domain.

**Actual state**: The `skillsSnapshot` in `agents/coder/sessions/sessions.json` shows 6 bundled skills loaded into every coder session:

| Skill | Source | Approx Size | Relevance to Coder |
|-------|--------|-------------|-------------------|
| `acp-router` | openclaw-extra (acpx) | ~7KB | None |
| `coding-agent` | openclaw-bundled | ~11KB | **Contradicts** coder's AGENTS.md workflow |
| `healthcheck` | openclaw-bundled | ~11KB | None |
| `skill-creator` | openclaw-bundled | ~18KB | None |
| `tmux` | openclaw-bundled | ~3.5KB | Maybe (coder uses `claude --print`, not tmux) |
| `weather` | openclaw-bundled | ~2.3KB | None |

**Total: ~52KB (~13,000 tokens)** of skill content injected into every sub-agent session.

This is LARGER than Lloyd's entire optimized system prompt (~11-12k tokens). The coder's own workspace files total only ~10.7KB (AGENTS.md: 8,753 + SOUL.md: 840 + TOOLS.md: 860), meaning skills are **5x the size of the agent's own instructions**.

The `coding-agent` skill is particularly harmful because it describes a PTY/tmux-based coding workflow that conflicts with the coder's AGENTS.md `claude --print` two-phase protocol.

**Root cause**: `commands.nativeSkills: "auto"` in `openclaw.json` enables skill discovery for all sessions globally. Bundled skills from the openclaw package are loaded for every agent. The `skills.entries` config only disables workspace skills, not bundled ones.

---

### Finding 3: Coder on Opus 4.6 — Cost vs. Value

**Pricing comparison**:
- Coder: Claude Opus 4.6 — $15/$75 per Mtok (input/output)
- Sonnet: Claude Sonnet 4.6 — $3/$15 per Mtok
- Ratio: **5x cost** across the board

**Coder session `7a591493` breakdown**:

| Metric | Value |
|--------|-------|
| Total tokens | 20,448 |
| Cache read | 66,951 |
| Cache write | 10,250 |
| Output tokens | 1,942 |
| Task | Hello world ASCII art (trivial) |

The coder architecture delegates to Claude Code (`claude --print`) for actual implementation. The coder agent itself primarily:
1. Reads files (`file_read` / `run_bash`)
2. Writes gameplans (`file_write` / `run_bash`)
3. Launches Claude Code (`run_bash` with `claude --print`)
4. Reviews diffs (`run_bash` with `git diff`)

These are orchestration tasks — Sonnet-grade work. The actual coding intelligence comes from Claude Code's own model. Opus is paying 5x for file I/O and bash commands.

**Note**: Researcher is also on Opus 4.6. Similar analysis applies — web search and fetch orchestration doesn't require Opus reasoning.

---

### Finding 4: run_bash Dominance (60-80% of coder tool calls)

Across all observed coder sessions, `run_bash` accounts for the majority of tool calls. In extreme cases, 45 out of ~50 calls were `run_bash`.

**Why the model prefers bash**:
1. It always works — never blocked by Mission Control or tool allow/deny
2. It can do everything in one call (cat, mkdir, python, git, curl)
3. Structured file tools fail unpredictably due to the allow/deny confusion (Finding 1)

**Implication**: The coder's 22-tool allowlist may be overengineered. A minimal set of `run_bash` + `bg_exec` + `bg_process` + `file_read` + `file_glob` might be more reliable and produce less tool confusion.

---

### Finding 5: Lloyd Working as Intended

Lloyd's current session (72,718 tokens, 36 messages, Sonnet 4.6):

| Tool | Calls | Status |
|------|-------|--------|
| `run_bash` | 5 | All succeeded |
| `mem_get` | 3 | Startup memory reads |
| `file_read` | 3 | All succeeded |
| `sessions_spawn` | 1 | Delegated to coder |
| `read` | 1 | **Failed** (correctly blocked) |

The `read` failure confirms Lloyd's `tools.json` correctly blocks built-ins. The `file_read` calls work via MCP. Lloyd correctly delegates coding work via `sessions_spawn`. Orchestrator behavior is functioning as designed in Phase 2.

Prefill gating is also working — coder sessions skip vault prefill, profile-based routing skips for chat/code/ops/voice.

---

### Finding 6: Token Usage Patterns

**Per-agent session costs**:

| Session | Agent | Model | Total Tokens | Cache Read | Cache Write | Output |
|---------|-------|-------|-------------|------------|-------------|--------|
| 8164b26e (active) | Lloyd | Sonnet 4.6 | 72,718 | 0 | 72,715 | 62 |
| 7a591493 (active) | Coder | Opus 4.6 | 20,448 | 66,951 | 10,250 | 1,942 |
| 721f9030 | Lloyd | Qwen local | 192,933 | 0 | 0 | 718 |
| 7e19b1cb | Lloyd | Qwen local | 151,556 | 0 | 0 | 630 |
| cron:f53ad621 | Lloyd | Sonnet 4.6 | 28,280 | 170,088 | 28,279 | 2,860 |

**Observations**:
- Lloyd's Anthropic sessions use caching effectively (72k write on first run)
- Coder inherits cached context from parent (67k cache read)
- Local model sessions have ZERO caching and hit near the 200k context limit
- Cron sessions are cache-heavy (170k read) from accumulated context

---

### Finding 7: Session Cleanup Working

- 27 coder sessions deleted in ~4 hours (`archiveAfterMinutes: 60`)
- Only 1 active coder session remains
- Deletion sizes range from 4KB to 179KB — cleanup handles both short and long sessions
- Main agent sessions persist via manual reset — no orphan accumulation

---

### Finding 8: Lloyd Attempted Disabled Tool

In the main session, Lloyd called `read` (built-in) which failed because it's disabled in `tools.json`. Lloyd then correctly fell back to `file_read` (MCP). This is the SAME pattern the coder exhibits (Finding 1) but Lloyd's broader toolset blocks catch it earlier.

This suggests the model has a natural preference for shorter/simpler tool names (`read` over `file_read`, `write` over `file_write`). The MCP tool naming convention (`file_*` prefix) works against this tendency.

---

### Recommendations

#### 4a: Block Built-in Tools in Sub-Agent tools.json

**Priority**: HIGH | **Effort**: 15 min | **Risk**: Low

Add explicit blocks to each sub-agent's `tools.json`. All 8 sub-agents currently share the same insufficient 4-entry deny list.

**Coder example** (`agents/coder/agent/tools.json`):
```json
{
  "memory_search": false, "memory_get": false, "web_search": false, "web_fetch": false,
  "read": false, "write": false, "edit": false, "exec": false,
  "process": false, "apply_patch": false, "image": false, "browser": false,
  "canvas": false, "nodes": false, "gateway": false, "session_status": false
}
```

Apply equivalent blocks to all 8 sub-agents, preserving only what each agent needs from their `tools.allow` list.

**Files**: All 8 `agents/{agent}/agent/tools.json`
**Impact**: Eliminates tool confusion, saves ~$0.10-0.25 per coder session in failed tool call retries

#### 4b: Suppress Bundled Skills for Sub-Agents

**Priority**: HIGH | **Effort**: 30 min | **Risk**: Medium (needs testing)

Test whether the `skills` field in agent config suppresses bundled skill loading.

**In `openclaw.json`, add to each sub-agent entry:**
```json
{
  "id": "coder",
  "skills": [],
  ...
}
```

If `skills: []` works, this removes ~13,000 tokens from every sub-agent session.

If the `skills` field doesn't control bundled skills, investigate `nativeSkills: false` at the agent level, or the `disableModelInvocation` flag per-skill.

**Files**: `openclaw.json` (agents.list entries)
**Impact**: ~13,000 tokens saved per sub-agent session

#### 4c: Fix Coder AGENTS.md Tool References

**Priority**: MEDIUM | **Effort**: 30 min | **Risk**: Low

Remove all built-in tool references from coder's AGENTS.md lines 186-192. Replace with:

```markdown
## Tools

- `run_bash` — primary shell execution (file ops, git, Claude Code launch)
- `bg_exec`, `bg_process` — background process management
- `file_read`, `file_glob`, `file_grep` — read and search files
- `file_write`, `file_edit`, `file_patch` — write and modify files
- `backlog_tasks`, `backlog_get_task`, `backlog_create_task`, `backlog_update_task` — task tracking
- `mem_search`, `mem_get` — vault search (for context only)
```

Do the same for all other sub-agent AGENTS.md files.

**Files**: All 8 `~/obsidian/agents/{agent}/AGENTS.md`
**Impact**: Prevents model from attempting disabled tools

#### 4d: Sonnet Default for Coder and Researcher

**Priority**: MEDIUM | **Effort**: 5 min | **Risk**: Low (reversible per-task)

Change coder and researcher defaults from Opus to Sonnet:

```json
{ "id": "coder", "model": "anthropic/claude-sonnet-4-6", ... }
{ "id": "researcher", "model": "anthropic/claude-sonnet-4-6", ... }
```

Lloyd can override for complex tasks via `sessions_spawn({ model: "claude-opus-4-6" })`.

**Files**: `openclaw.json`
**Impact**: ~80% cost reduction on coder/researcher sessions

#### 4e: Slim Coder Tool Allow List (22 → ~12)

**Priority**: LOW | **Effort**: 15 min | **Risk**: Low

Given run_bash dominance, reduce the allow list. Remove:
- `file_patch` — rarely used, `run_bash` + `git apply` preferred
- `http_search`, `http_fetch`, `http_request` — coder shouldn't do web research (delegate to researcher)
- `tag_explore`, `vault_overview` — exploratory memory tools, not needed for coding
- `mem_write` — coder shouldn't write to vault
- `backlog_boards` — unnecessary for task-level operations

**Files**: `openclaw.json` (agents.list.coder.tools.allow)
**Impact**: ~10 fewer tool schemas in system prompt

---

### Priority Matrix

| Rec | Priority | Effort | Token Savings | Cost Savings | Risk |
|-----|----------|--------|--------------|-------------|------|
| 4a: Block built-ins | HIGH | 15 min | ~200/session | $0.10-0.25/session | Low |
| 4b: Suppress skills | HIGH | 30 min | ~13,000/session | ~$1.00/session (Opus cache) | Medium |
| 4c: Fix AGENTS.md | MEDIUM | 30 min | 0 (behavioral) | Prevents wasted calls | Low |
| 4d: Sonnet default | MEDIUM | 5 min | 0 | ~80% cost reduction | Low |
| 4e: Slim tool list | LOW | 15 min | ~500/session | minor | Low |

**Phase 4 total (4a-4b): ~13,200 tokens saved per sub-agent session. ~$1.10/session at Opus pricing. ~1 hour effort.**

---

### Current State Summary (Post Phase 3, Pre Phase 4)

| Component | Lloyd (main) | Coder | Other Sub-agents |
|-----------|-------------|-------|-----------------|
| Model | Sonnet 4.6 | **Opus 4.6** | Sonnet 4.6 (memory: local) |
| Config tools | 18 allowed, 36 blocked | 22 allowed, 4 blocked | 4-17 allowed, 4 blocked |
| **Actual tools** | ~18 | **~30+** (inherits built-ins) | ~15-25+ (inherits) |
| Bundled skills | 8 (6 bundled + 2 workspace) | **6 bundled (~52KB)** | **6 bundled (~52KB)** |
| Prefill gating | Profile-based (working) | Skipped (correct) | Skipped (correct) |
| tools.json blocks | 36 entries | **4 entries (insufficient)** | **4 entries (insufficient)** |

### Verification (after implementing 4a-4e)

1. Restart gateway: `kill $(lsof -ti :18789); sleep 2; systemctl --user start openclaw-gateway.service`
2. Spawn a coder task: "Create a hello world file at ~/test.txt"
3. Check coder session — should use ONLY MCP tools, no `write`/`read`/`exec` calls
4. Check `skillsSnapshot` in sessions.json — should be empty or absent for sub-agents
5. Check `timing.jsonl` — no `tool_call` entries for disabled built-in tools
6. Verify cost — coder session should be ~$0.05 (Sonnet) vs ~$0.35 (Opus) for trivial tasks
