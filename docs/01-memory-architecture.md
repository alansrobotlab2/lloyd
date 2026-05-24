# Memory Architecture

## Overview

Hermes has two memory layers that work together:

1. **Core memory** — flat-file system (`MEMORY.md` / `USER.md`) injected as a frozen snapshot into the system prompt at session start. 2200 / 1375 char limits. Acts as a **cache layer** — a compact summary of what matters most right now.
2. **Structured fact store** — YAML-frontmatter fact files in `~/obsidian/memory/_pipeline/facts/{entity}/{entity}-{category}.md`. Unlimited storage, confidence-scored, grouped by entity and category. Contradiction detection built in.
3. **Vault tools** — BM25+vector search across 7 obsidian segments (`memory`, `knowledge`, `projects`, `agents`, `personal`, `work`, `skills`) via QMD daemon. Agent-callable mid-conversation.

All three are implemented in a single plugin: `~/.hermes/plugins/next-gen-memory/__init__.py`. Zero hermes-agent source changes — integration is entirely via the plugin hook system.

---

## Part 1: Core Memory (hermes-agent built-in)

### Storage

Two markdown files, split by `§` (section sign) as an entry delimiter:

- `~/.hermes/memories/MEMORY.md` — agent's personal notes (2200 char limit)
- `~/.hermes/memories/USER.md` — user profile and preferences (1375 char limit)

### The MemoryStore Class

`tools/memory_tool.py` implements `MemoryStore`:

```
MemoryStore:
  memory_entries: List[str]           # Lines from MEMORY.md
  user_entries: List[str]             # Lines from USER.md
  _system_prompt_snapshot: Dict       # Frozen at load time — never mutated mid-session
  memory_char_limit: 2200
  user_char_limit: 1375
```

**Frozen Snapshot Pattern:** On session start, files are loaded once and a snapshot is captured. Mid-session writes go to disk immediately but the system prompt never changes within a session. This keeps the prompt identical across all turns for stable prefix cache hits.

### Memory Tool Actions

The `memory` tool exposes three actions:
- `add` — append new entry, enforces char limit
- `replace` — update an existing entry by matching old text
- `remove` — delete an entry by matching text

Entries are scanned for injection safety before being accepted:
- Prompt injection patterns (`ignore previous instructions`)
- Exfiltration patterns (`curl $API_KEY`, reading `.env`)
- Invisible Unicode (U+200B, U+202E)
- SSH backdoor references (`authorized_keys`)

### Session Lifecycle

```
Session Start
│
├── MemoryStore.load_from_disk()
│     → Read MEMORY.md + USER.md
│     → Deduplicate entries
│     → Capture frozen snapshot
│
├── Plugin: on_session_start hook
│     → Sync MEMORY.md / USER.md into fact store (catches compression flush writes)
│
├── _build_system_prompt()
│     → Inject frozen snapshot into system prompt
│     → Cache prompt (never rebuilt mid-session)
│
├── Plugin: pre_llm_call hook (first turn only)
│     → Inject structured fact profile into ephemeral system prompt
│     → Returns None on turns 2+ (prefix cache preserved)
│
└── Main Loop
      │
      ├── Every turn: increment _turns_since_memory
      │
      ├── Every 10 turns (nudge_interval):
      │     → Set _should_review_memory = True
      │
      ├── After response delivered:
      │     → Spawn background review thread (daemon)
      │     → Isolated AIAgent runs against conversation snapshot
      │     → Calls memory tool → writes to disk
      │     → Plugin: post_tool_call hook mirrors write to fact store
      │     → Changes visible in NEXT session's system prompt
      │
      └── If context too large → compression:
            → Synchronous flush: API call with ONLY memory tool
            → Agent decides what's worth keeping before history is lost
            → (Flush bypasses dispatch — caught by on_session_start in next session)
```

### What the Agent Sees in the System Prompt

```
════════════════════════════════════════════════
MEMORY (your personal notes) [45% — 990/2,200 chars]
════════════════════════════════════════════════
[entry 1]
§
[entry 2]

════════════════════════════════════════════════
USER PROFILE (who the user is) [60% — 825/1,375 chars]
════════════════════════════════════════════════
[profile entry]

[USER FACTS — 23 facts]                  ← injected by pre_llm_call on first turn
preferences:
  • Prefers terse responses  (conf=0.95)
  • Dark mode in all editors  (conf=0.90)
corrections:
  • Don't mock the database in integration tests  (conf=0.92)
...

[AGENT FACTS — 15 facts]
environment:
  • mc_server.py runs on port 8080  (conf=0.85)
...
```

### Session Persistence

| Storage | Location | Content |
|---|---|---|
| SQLite | `~/.hermes/state.db` (WAL, FTS5) | Full message history, tool calls, reasoning, token costs |
| JSON logs | `~/.hermes/logs/session_[id].json` | Human-readable session snapshots |
| System prompt | Stored in `sessions` table | Full assembled prompt for session replay |

On **context compression**, a new session is created with `parent_session_id` pointing back, maintaining lineage. Before compression fires, a synchronous memory flush runs — the agent is given only the memory tool and asked to save anything important before history is discarded.

### Background Review (Nudge)

Every 10 turns, after the response is delivered, a daemon thread spawns an isolated `AIAgent`:
- Same model + config as main agent
- Shared `_memory_store` reference
- `nudge_interval = 0` (prevents recursive nudges)
- Full conversation snapshot as history

It runs a review prompt asking whether anything is worth saving (user preferences, corrections, behavioral patterns, non-trivial approaches). If so, it calls the memory tool. Output prints `💾 [summary]` to the terminal — never visible as part of a user response.

The nudge subagent goes through normal `model_tools.dispatch`, so the `post_tool_call` hook fires and mirrors writes to the fact store automatically.

Three prompt variants:
- **Memory only** — focus on user persona, preferences, behavioral expectations
- **Skills only** — focus on non-obvious approaches, trial-and-error findings
- **Combined** — both, used when both nudges fire together

### Key Design Properties

1. **Frozen snapshot** — prefix cache stability over freshness within a session
2. **File-backed** — survives process restarts, no in-memory-only state
3. **Background nudges** — never competes with the current user task
4. **Compression-safe flush** — synchronous save before history is discarded
5. **Character limits, not token limits** — model-agnostic, simple to enforce
6. **Injection scanning** — adversarial content blocked at write time

---

## Part 2: Structured Fact Store (plugin-native)

Implemented in `~/.hermes/plugins/next-gen-memory/__init__.py`. No external service dependency — reads/writes YAML files directly.

### Storage

YAML frontmatter files at:
`~/obsidian/memory/_pipeline/facts/{entity}/{entity}-{category}.md`

**Bi-temporal fact data model:**
```yaml
---
type: facts
entity: user
category: preferences
facts:
  - fact: "Prefers terse responses with no trailing summaries"
    confidence: 0.95
    event_date: null
    category: preferences
    id: "fact_abc123"
    # Bi-temporal fields (added 2026-04-02)
    created_at: "2026-04-02T10:00:00+00:00"   # when data entered the system
    valid_at: "2026-04-01T00:00:00Z"           # when fact became true (null = always)
    invalid_at: null                            # when fact stopped being true
    expired_at: null                            # when superseded by contradiction
    source_doc: "memory/2026-04-01.md"          # provenance back to source
last_updated: "2026-04-02T10:00:00"
---
```

Facts are never deleted — when superseded, they get `invalid_at` and `expired_at` timestamps. `_get_facts_sync` filters expired facts by default; pass `include_expired=True` for full history.

### Entity Resolution (Three-Tier)

All entity lookups go through `_resolve_entity()`:

1. **Exact match** — case-insensitive directory lookup
2. **Alias table** — `entity-aliases.json` maps lowercase names to canonical entities (537+ entries)
3. **Fuzzy match** — Levenshtein distance + token overlap (Jaccard) + substring containment

Examples: `"alan"` → `"Alan"`, `"agency swarm"` → `"Agency Swarm"`, `"User"` → `"user"`

Alias table is auto-populated and can be rebuilt via `fact_aliases action=rebuild`.

### Tools

| Tool | Signature | Purpose |
|---|---|---|
| `fact_get` | `(entity, category)` | Retrieve facts (auto-resolves entity, filters expired) |
| `fact_add` | `(entity, category, fact, confidence, valid_at, source_doc)` | Add fact with temporal fields + entity resolution |
| `fact_profile` | `(entity, include_summary)` | Synthesized profile grouped by category |
| `fact_check` | `(entity, category)` | Enhanced contradiction detection (opposing terms + negation + semantic overlap) |
| `fact_resolve` | `(entity, auto_resolve)` | Temporally invalidate lower-confidence contradicting facts |
| `fact_aliases` | `(action, alias, canonical, name)` | Manage entity alias table (list, rebuild, add, resolve) |
| `vault_graph_search` | `(entities, max_hops)` | Graph traversal from seed entities via relations-index |

### Contradiction Detection (Enhanced)

`fact_check` uses three detection methods:

1. **Opposing term pairs** — enabled/disabled, yes/no, true/false, active/inactive, etc.
2. **Negation patterns** — "never"/"always", "not"/"is", "no longer", "stopped", "changed from"
3. **High overlap detection** — facts with >60% token overlap are flagged as potential updates

Resolution via `fact_resolve` sets `expired_at` + `invalid_at` on the lower-confidence fact rather than deleting it, preserving full provenance history.

### How Facts Get Written

Three paths, all converging on `_fact_add`:

| Source | Mechanism | Confidence | Notes |
|---|---|---|---|
| Agent calls `memory` tool (add/replace) | `post_tool_call` hook | 0.80 | Includes background nudge writes |
| Agent calls `fact_add` directly | Normal tool dispatch | Per-call | Agent chooses entity/category/confidence |
| Session-start sync from flat files | `on_session_start` hook | 0.75 | Catches compression flush writes |

### Category Inference

When mirroring from `memory` tool writes, the category is inferred by keyword scan (`_infer_category`):

| Category | Keywords |
|---|---|
| `preferences` | prefer, don't like, style, tone, concise, verbose |
| `corrections` | correction, wrong, actually, don't do, stop doing |
| `capabilities` | can, able to, expert, proficient |
| `environment` | path, config, port, url, server, installed, directory |
| `workflow` | workflow, process, always, typically, convention |
| `general` | (fallback) |

---

## Part 3: Vault Tools (plugin-native, QMD-dependent)

Unstructured markdown documents stored across 7 segments in `~/obsidian/`:
`memory`, `knowledge`, `projects`, `agents`, `personal`, `work`, `skills`

### Hybrid Retrieval with RRF Fusion

`vault_search` combines three retrieval pillars fused via **Reciprocal Rank Fusion (RRF)**:

```
QMD vector search (semantic) ──────┐
QMD BM25 keyword search (exact) ──┼── RRF Fusion (k=60) ── Top-K results
Graph traversal (structural) ──────┘
```

1. **QMD daemon** — BM25+vector parallel search across vault segments
2. **Graph traversal** — BFS from extracted entities through relations-index, up to 2 hops
3. **RRF scoring** — `score(doc) = Σ 1/(k + rank_i)` across all sources. Documents appearing in multiple sources get boosted.

**Dual-keyword extraction** splits queries into:
- **Local keywords** — proper nouns, specific entities → drive entity graph lookup
- **Global keywords** — themes, concepts → drive segment routing

Intent-aware routing still applies:
- Temporal queries → `memory`, `personal` segments
- Factual queries → `knowledge`, `projects`, `agents` segments
- Conceptual queries → broader parallel search

Falls back to `qmd` subprocess if daemon is unavailable. Graph traversal is additive — if no graph results, falls back to QMD-only.

### Tools

| Tool | Signature | Purpose |
|---|---|---|
| `vault_search` | `(query, max_results, min_score, scope, consolidate)` | RRF-fused hybrid search (QMD + graph) |
| `vault_recall` | `(query, limit, include_facts)` | Parallel vault search + fact retrieval |
| `vault_graph_search` | `(entities, max_hops)` | Graph-only traversal from seed entities |
| `vault_read` | `(path, start_line, num_lines)` | Read vault file by path |
| `vault_write` | `(path, content)` | Write vault file with audit logging |
| `vault_overview` | `(detail)` | Vault statistics (summary/hubs) |

### Naming Convention

Three clear namespaces:

- `memory` — hermes-agent flat-file core (built-in, unchanged)
- `fact_*` — structured fact operations (plugin-native)
- `vault_*` — obsidian vault operations (plugin-native, QMD for search)

---

## Part 4: Plugin Hook Integration

All integration between core memory and the fact/vault systems is done via three plugin hooks. Zero hermes-agent source changes.

### `post_tool_call` → Live memory mirroring

Fires in `model_tools.dispatch` after every tool call — including background nudge subagents. When `tool_name == "memory"` and `action` is `add`/`replace` and result confirms `success: true`:
- Maps target to entity: `"user"` target → `entity="user"`, `"memory"` target → `entity="self"`
- Infers category via keyword scan
- Calls `_fact_add` with `confidence=0.8`
- Failure is swallowed (debug log only) — never blocks a memory write

### `on_session_start` → Compression flush sync

The compression flush calls `memory_tool` **directly** (bypassing `model_tools.dispatch`), so `post_tool_call` never fires for it. To catch flush-written entries:
- Reads `MEMORY.md` and `USER.md` at session start
- Fetches existing fact texts via `_get_facts_sync`
- Mirrors any entries not already in the fact store with `confidence=0.75`

### `pre_llm_call` → Dynamic fact injection

Fires once per `run_conversation` call before the tool loop. Returns `None` on all turns except the first (`is_first_turn=True`) — turns 2+ use the unmodified cached system prompt and fully hit the prefix cache.

On first turn:
- Calls `_get_facts_sync` for both `"user"` and `"self"`
- Groups by category, sorts by confidence descending
- Assembles compact block capped at 1400 chars
- Returns `{"context": "..."}` → appended to ephemeral system prompt

This is *supplemental* to the flat-file snapshot. The fact profile adds the full accumulated history — facts that didn't survive the char limit — in structured form.

---

## Design Decisions

### What's NOT covered (by design)

- `remove` actions on `memory` tool don't delete facts. Facts are a history; the flat file is the live cache.
- `replace` writes both old and new content to the fact store. Contradiction detection (`fact_check`) handles divergence.
- Gateway sessions fire `on_session_start` too — flush sync works since it reads from disk.

### Why plugin hooks instead of hermes-agent patches

- No fork maintenance — upstream hermes-agent updates don't break anything
- Single-file implementation — all integration logic in one place
- Hook failures are swallowed — the core agent loop is never affected
- Easy to disable — remove `next-gen-memory` from `config.yaml` platform_toolsets

### Confidence levels

| Source | Confidence | Rationale |
|---|---|---|
| Agent calls `fact_add` directly | Per-call (typically 0.9) | Agent explicitly chose to record this |
| `post_tool_call` mirror | 0.80 | Live write, but category is inferred |
| `on_session_start` sync | 0.75 | From flat file, may be stale |

### Prefix cache strategy

The frozen snapshot pattern is preserved exactly:
- `_cached_system_prompt` is never modified mid-session
- `pre_llm_call` only injects on `is_first_turn=True` via the ephemeral system prompt
- Turns 2+ see the identical system prompt → prefix cache hits

---

## Part 5: Knowledge Graph (NetworkX + mc_server.py)

The entity graph is backed by a **NetworkX DiGraph** built from `relations-index.json` + `facts-index.json`. Cached with 60s TTL, rebuilt lazily on first access.

### Graph Structure

- **Document nodes** — vault documents referenced in relations-index
- **Entity nodes** — entities from the structured fact store (`entity::{name}`)
- **Edges** — typed relationships with weight scores

### Typed Relationship Vocabulary

```
related-to      General semantic connection (default)
causes          Causal relationship (A causes B)
depends-on      Dependency (A requires B)
supersedes      A replaces/updates B
contradicts     A conflicts with B
part-of         Hierarchical containment
mentions        Document references entity
co-occurs       Entities appear together frequently
temporal-next   Sequential ordering
wiki-link       Existing wiki-link reference (legacy)
has-facts       Entity-to-fact-file link (auto-generated)
```

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/entity-graph` | GET | Full graph (nodes + edges) for visualization |
| `/api/entity-graph/traverse` | GET | BFS subgraph from seed entities (`?entities=X,Y&max_hops=2`) |
| `/api/entity-graph/stats` | GET | Node/edge counts, density, degree distribution, type breakdown |

### Current Stats (as of 2026-04-02)

- 649 nodes (94 documents + 555 entities)
- 804 edges
- Density: 0.001912
- Top connected: `projects/lloyd/architecture/index.md` (degree 44)

---

## Skipped / Future

| Item | Status | Notes |
|---|---|---|
| `tag_search` / `tag_explore` | Deprecated | Excluded from migration |
| Query-aware first-turn vault recall | Future | Could use `vault_recall` in `pre_llm_call` to weight facts toward the opening message. Requires QMD daemon to be running; current approach works without it. |
| `MEMORY.md` as rendered fact summary | Future | Instead of the agent managing the flat file directly, auto-render `MEMORY.md` from the top-N facts by confidence. Would remove the char-limit management burden from the agent entirely. |
| Community detection (Leiden) | Future | Would enable automatic topic clustering and corpus-wide synthesis queries. NetworkX supports it but needs additional tuning for our graph density. |
| Cross-encoder reranking | Future | After RRF fusion, pass top-K candidates through a reranker for precision. Could use existing local Qwen model. |
