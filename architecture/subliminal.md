---
segment: architecture
tags: [architecture,subliminal]
- architecture
- memory
- subliminal
- prefetch
type: architecture
updated: 2026-04-14
---



# Context Injection & Prefetch System

## Overview

Automatic context retrieval injected before every LLM call. The agent sees relevant skills,facts,and vault documents without explicit tool calls. System prompt stays static for vLLM prefix cache hits; dynamic context is prepended to the user message.

Three layers of context injection operate independently:

1. **System prompt** (per-session,static): SOUL.md + MEMORY.md + USER.md + skills index
2. **Prefetch** (per-message,dynamic): skills + entity facts + vault search + conversation focus
3. **Agent tools** (on-demand): vault_recall,vault_search,fact_get,memory_read

## Architecture

```
User message arrives at server.py
    │
    ├─→ build_system_prompt()                    [per-session,static]
    │   ├─ SOUL.md (identity + operating contract)
    │   ├─ MEMORY.md + USER.md (persistent memory)
    │   ├─ Available skills index (names only)
    │   └─ Platform hint
    │
    ├─→ prefetch_context(text,session_id)       [per-message,~224ms warm]
    │   │
    │   ├─→ SessionFocus.update(text)            [0ms,in-memory]
    │   │   Accumulate weighted keywords,decay old by 0.75/turn
    │   │
    │   └─→ ThreadPoolExecutor(max_workers=3)    [parallel,~224ms total]
    │       ├─ Worker 1: _search_skills()        Keyword overlap scoring
    │       ├─ Worker 2: _search_facts()         Entity extraction → fact lookup
    │       └─ Worker 3: _search_vault(focus)    QMD hybrid search,focus-enriched
    │
    ├─→ Memory nudge injection                   [every 20 turns]
    │   <system-reminder> nudge to capture undocumented decisions
    │
    └─→ SDK query(text=<context>+message)
        │
        ↓
    LLM sees: [system prompt] [history] [<context>block + user msg]
              ←── prefix-cached ──→     ←── uncached (~2-8KB) ──→
```

## Prefetch Workers

### Worker 1: Skill Search
- **Source:** `~/obsidian/skills/` + `~/lloyd/skills/`
- **Method:** Token overlap scoring (name×3,desc×2,tags×1.5,body×1.0)
- **Output:** First skill full body (≤6000 chars),second skill excerpt (≤500 chars)
- **Thresholds:** Score ≥3.0 for first,≥4.0 for second
- **Cache:** 5-minute TTL skill list cache in memory

### Worker 2: Entity Fact Search
- **Source:** `~/obsidian/facts/{entity}/` fact files
- **Method:** Extract entity names from query → match against fact directory names → fetch top facts by confidence
- **Output:** Top 2 entities × 3 facts each = ≤6 fact lines
- **No LLM:** Pure regex entity matching + file reads

### Worker 3: Vault Search (QMD)
- **Source:** QMD daemon (port 8181) — hybrid BM25 + vector search
- **Collections:** memory,knowledge,projects,lloyd,work
- **Method:** Dual lex+vec queries sent to QMD,results filtered by score
- **Output:** Top 5 results with ≤500 char snippets,minimum score 0.5
- **Focus enrichment:** Query is augmented with conversation focus keywords (see below)
- **Gating:** Skipped when effective query < 25 chars (prevents noise on vague messages)
- **Latency:** ~211ms warm,~4.6s cold start (first query after restart loads embedding model)

## Conversation Focus Tracking

Per-session keyword accumulator that maintains awareness of what the conversation is about. Solves the "stateless prefetch" problem — without it,a message like "what about the PID gains?" produces garbage vault results; with it,the query inherits "alfie servo shoulder joint" from earlier turns.

### Tier 1: Keyword Accumulator (Zero Cost)

In-memory `SessionFocus` object per session:

```python
class SessionFocus:
    keyword_weights: dict[str,float]  # word → cumulative weight
    turn_count: int
    topics: list[str]                  # 35B-extracted topic phrases
```

- **Update:** Each message adds keywords with weight 1.0,all existing weights decay by 0.75
- **Cleanup:** Keywords below 0.1 weight are evicted (prevents unbounded growth)
- **Output:** `enrich_query()` appends top-6 focus keywords + 35B topics to vault search query
- **Noise filtering:** 120+ stopwords stripped before accumulation
- **LRU eviction:** Max 50 tracked sessions; oldest evicted on overflow
- **State:** Lost on restart (acceptable — focus rebuilds naturally within 2-3 turns)

### Tier 2: 35B Topic Extraction (Background,Async)

Every 5 turns (after ≥3 turns of context),a background call to Qwen3.5-35B extracts 3-5 topic phrases from the last 10 messages:

```
User messages → 35B model → ["servo PID tuning","Alfie shoulder joint","motor temperature"]
```

- **Endpoint:** localhost:8091 (llama.cpp,Qwen3.5-35B-A3B)
- **Timing:** Background `asyncio.ensure_future()` after SDK query completes — zero user-facing latency
- **One-turn delay:** Results available on the NEXT message,not the current one
- **Prompt:** "Extract 3-5 short topic phrases (2-4 words each),one per line"
- **Temperature:** 0.0 (deterministic)
- **Timeout:** 15s (non-fatal failure)

### Focus Example

```
Turn 1: "lets look at the alfie servo configuration"
  Focus: [alfie,servo,configuration]

Turn 3: "try increasing the D term to reduce oscillation"
  Focus: [pid,gains,shoulder,oscillation,alfie,servo]

Turn 5: "what about the PID gains?"  ← only 25 chars
  Enriched query: "what about the PID gains? pid gains motor temperature alfie servo"
  Vault finds: Alfie Shoulder GPIO Setup,Closed-Loop Stepper Control
```

## Context Block Format

```xml
<context>
<skill name="autonomy-error-handling" score="14.5">
[full skill body,≤6000 chars]
</skill>
<skill name="upstream-bug-triage" score="13.5" excerpt="true">
[excerpt,≤500 chars]
</skill>
<facts>
- [Lloyd] Vault search added to prefetch pipeline (confidence: 0.75)
- [QMD] QMD is installed at version 1.0.8 (confidence: 1.0)
</facts>
<vault-context>
- **Alfie Servo Settings** (score: 0.93): PID gains shoulder=0.8/0.1/0.05...
- **2026-04-13 Daily Notes** (score: 0.56): Discussed servo tuning approach...
</vault-context>
</context>

[original user message]
```

## Post-Session Enrichment

Two background processes fire after each SDK query completes:

### Inline Fact Extraction
- **Trigger:** Session ends with ≥3 user messages,non-trivial content
- **Method:** 35B model extracts 3-5 durable facts from session transcript
- **Output:** Facts written to `~/obsidian/facts/{entity}/{entity}-session-extracted.md`
- **Confidence:** 0.75,provenance: EXTRACTED,linked to source session
- **Benefit:** Reduces fact extraction latency from 11 hours (nightly) to seconds

### Memory Preservation Nudge
- **Trigger:** Every 20 user turns within a session
- **Method:** `<system-reminder>` prepended to user message
- **Content:** Reminds agent to call `memory_add` or `fact_add` for any undocumented decisions
- **Purpose:** Prevents context loss during SDK context compaction

## Tuning Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `SKILL_THRESHOLD_FIRST` | 3.0 | Min score to inject first skill |
| `SKILL_THRESHOLD_SECOND` | 4.0 | Min score to inject second skill (excerpt) |
| `SKILL_BODY_MAX` | 6000 chars | Max first skill body |
| `SKILL_EXCERPT_MAX` | 500 chars | Max second skill excerpt |
| `FACT_MAX_ENTITIES` | 2 | Max entities to look up |
| `FACT_MAX_PER_ENTITY` | 3 | Max facts per entity |
| `VAULT_MAX_RESULTS` | 5 | Max vault search results |
| `VAULT_SNIPPET_MAX` | 500 chars | Max chars per vault snippet |
| `VAULT_MIN_SCORE` | 0.5 | Min vault result relevance score |
| `VAULT_MIN_QUERY_LEN` | 25 chars | Skip vault for short queries (after focus enrichment) |
| `FOCUS_DECAY` | 0.75 | Per-turn keyword weight decay |
| `FOCUS_TOP_K` | 6 | Focus keywords used to enrich vault query |
| `FOCUS_EXTRACT_INTERVAL` | 5 turns | 35B topic extraction frequency |
| `MIN_MESSAGE_LEN` | 10 chars | Skip all prefetch for very short messages |

## Files

| File | Purpose |
|------|---------|
| `~/lloyd/prefetch.py` | Prefetch layer: 3-worker parallel search,SessionFocus tracker,context formatting |
| `~/lloyd/prompt_builder.py` | System prompt assembly (SOUL.md + MEMORY.md + USER.md + skills index) |
| `~/lloyd/server.py` | Integration: passes session_id to prefetch,35B focus extraction,memory nudge,post-session fact extraction |
| `~/lloyd/agent_mcp/subliminal.py` | Legacy subliminal_recall tool (keyword extraction + SOUL.md) — rarely used |
| `~/lloyd/agent_mcp/memory.py` | Vault/fact tools: _qmd_daemon_search,_extract_entities_from_query,_fact_add |
| `~/lloyd/agent_mcp/skills.py` | Skill loading/scoring: _iter_skills,_score_skill,_tokenize |

## Infrastructure

| Service | Port | GPU | Purpose |
|---------|------|-----|---------|
| QMD daemon | 8181 | GPU 0 (RTX 5090,1.2GB VRAM) | Hybrid BM25+vector search,embedding model |
| Qwen 35B | 8091 | GPU 0 (llama.cpp) | Post-session capture,fact extraction,focus extraction |
| vLLM | 8096 | GPU 1 (RTX PRO 6000) | Main agent model (prefix-cached system prompt) |

## History

- **2026-03-24:** Phase 3 deployed via `appendSystemContext` in OpenClaw — cache invalidation discovered
- **2026-03-27:** Cache problem analyzed,design doc for context event migration
- **2026-03-28:** Phase 3.5 — `context` plugin hook added to OpenClaw core,subliminal migrated to ephemeral injection
- **2026-04-05:** Lloyd migration — system moved from OpenClaw to Claude Agent SDK. Subliminal replaced by `prefetch.py`
- **2026-04-14:** Active Memory (#289) — vault search added to prefetch (3rd parallel worker),conversation focus tracking (keyword accumulator + 35B topic extraction),inline fact extraction,memory preservation nudge

