---
title: "Document Relations & Retrieval Pipeline"
type: reference
tags: [lloyd, architecture, memory, retrieval, relations, vault]
segment: projects
date: 2026-03-10
---

# Document Relations & Retrieval Pipeline

Investigation into adding typed document relations to the vault and optimizing Lloyd's retrieval cycle time. Covers the full picture: relation schema, storage, discovery, indexing, and a new `context_bundle` tool that replaces Lloyd's current multi-call retrieval pattern.

## Problem Statement

### Tags Don't Express Relationships

The vault has 483 documents and 249 tags. Tags express **topical clustering** — "this doc is about `ai` and `tts`." They're great for discovery ("show me everything about TTS") but can't express:

- **Temporal ordering** — this design doc led to this implementation note, which was superseded by this updated approach
- **Dependency** — you can't understand doc B without reading doc A first
- **Conflict** — these two docs disagree (one is stale, or they represent different approaches)
- **Derivation** — this skill was extracted from this session log, this plan produced these backlog items
- **Hierarchy beyond folders** — a project's architecture doc, its phase summaries, and its individual session notes form a tree that folders only partially capture

The autolink skill inserts wikilinks into body text, but those are *mentions*, not *typed relationships*. Clicking a wikilink tells you nothing about *why* these docs are connected.

### Retrieval Is Slow and Expensive

When Lloyd needs vault context, the current flow takes 3-5 tool calls with full Opus reasoning between each:

```
Lloyd decides it needs context           ~0ms
  → mem_search API call                  ~200ms (BM25 + network)
  → LLM processes results, picks doc     ~3-5s  (Opus thinking)
  → mem_get API call                     ~100ms
  → LLM reads doc, decides needs more    ~3-5s  (Opus thinking)
  → tag_search or another mem_get        ~200ms
  → LLM synthesizes answer               ~5-8s  (Opus thinking)
                                         ─────
Total:                                   ~15-20s, 3-5 tool calls, ~$0.05-0.10
```

The tools themselves are fast (sub-second). The latency is in Opus reasoning between each call — deciding what to read next, whether it has enough context, what's missing. Each tool call is a full LLM round-trip.

### What's Available But Hidden

| Step | What Lloyd Gets | What Exists But Isn't Surfaced |
|------|----------------|-------------------------------|
| `mem_search` results | path, snippet, score | Tags, frontmatter, wikilinks — all in the doc but not in results |
| `mem_get` content | Full doc text | Outbound links point to related docs Lloyd hasn't read |
| `tag_search` results | Matching docs | Co-occurrence patterns, graph neighborhood |

The information exists. It's spread across multiple tool calls with no way to get it in one shot.

## Proposed Solution

### Relation Types

Six types, kept small so they're consistently applied:

| Relation | Inverse | Meaning |
|----------|---------|---------|
| `implements` | `designed-by` | Doc implements a plan/design described in the target |
| `supersedes` | `superseded-by` | Doc replaces/updates the target (target is stale) |
| `depends-on` | `required-by` | Must read the target to understand this doc |
| `derived-from` | `produces` | Doc was extracted/generated from the target |
| `related-to` | `related-to` | Symmetric catch-all for loose connections |
| `conflicts-with` | `conflicts-with` | Docs disagree or represent alternative approaches |

### Storage: Frontmatter + Generated Index

**Primary (source of truth):** Relations stored in document frontmatter:

```yaml
---
tags: [alfie, robotics, groot]
relations:
  implements: ["projects/alfie/gr00t/gr00t.md"]
  supersedes: ["projects/alfie/gr00t/phase1-summary.md"]
  depends-on: ["knowledge/ai/imitation-learning.md"]
---
```

**Generated cache:** `relations-index.json` built from frontmatter for fast graph traversal:

```json
{
  "edges": [
    {
      "source": "projects/alfie/gr00t/phase2-summary.md",
      "type": "supersedes",
      "target": "projects/alfie/gr00t/phase1-summary.md",
      "origin": "manual"
    }
  ],
  "stale": ["projects/alfie/gr00t/phase1-summary.md"],
  "built_at": "2026-03-10T09:30:00Z"
}
```

The index is a cache, not a source of truth — can be rebuilt anytime from frontmatter. Supports fast lookups: "what does this doc relate to?" and "what's stale?" without scanning 483 files.

### Relation Discovery — Three Sources

**1. Manual (Alan or Lloyd says so)**
- "This replaces the old phase 1 doc"
- Lloyd updates frontmatter on both sides (relation + inverse)

**2. Automated discovery (nightly job)**
- Scans recently modified docs (last 24h)
- Heuristics:
  - Existing wikilinks in body text → candidate `related-to`
  - Sequential naming (`phase1` → `phase2`) → `supersedes`
  - Content similarity via BM25 cross-search → `related-to`
  - Derivation signals (skill mentioning a session date) → `derived-from`
- High-confidence (>0.8) → written directly to frontmatter
- Medium-confidence (0.5-0.8) → queued in `memory/relation-review.md`
- Full vault scan runs weekly for cross-doc connections

**3. On-write indexing**
- When `mem_write` updates a document, the relations index rebuilds incrementally
- Parse written frontmatter for relations, update inverse side, rebuild index entry
- Pure Python, <100ms — no LLM needed

## Retrieval Architecture: `context_bundle` Tool

A new MCP tool in `tool_services.py` that replaces Lloyd's multi-call retrieval pattern with a single call. Two modes:

### Shallow Mode (Python only, <500ms)

Mechanical graph traversal — no LLM reasoning needed:

```python
context_bundle(query="GR00T status")
```

Internal pipeline:
1. BM25 search → top results
2. Read frontmatter of each result → get tags + relations
3. Filter stale docs (`superseded-by` exists → skip, note why)
4. Follow `depends-on` relations one hop → include as prerequisites
5. Follow `implements` / `derived-from` → include as context
6. Return primary doc full text + related doc summaries + skip list

Response:
```json
{
  "primary": {
    "path": "projects/alfie/gr00t/phase3-summary.md",
    "content": "...(full text)...",
    "tags": ["alfie", "groot"],
    "relations": { "supersedes": ["phase2-summary.md"], "implements": ["gr00t.md"] }
  },
  "related": [
    {
      "path": "projects/alfie/gr00t/gr00t.md",
      "relation": "designed-by",
      "summary": "...(first 200 chars or frontmatter summary)..."
    }
  ],
  "excluded": [
    { "path": "phase2-summary.md", "reason": "superseded by phase3-summary.md" }
  ]
}
```

**Latency:** <500ms. **Cost:** $0. **Handles:** 90% of retrieval needs.

### Deep Mode (Local LLM inline, 3-8s)

For ambiguous or cross-topic queries where mechanical traversal isn't enough:

```python
context_bundle(query="voice cloning across all projects", mode="deep", max_seconds=10)
```

Internal pipeline:
1. Run shallow mode first
2. Send results + original query to local Qwen via direct vLLM call (`localhost:8091`)
3. Qwen identifies gaps: "should also check knowledge/ai/tts/ and projects/alfie/ for voice-related docs"
4. Run additional targeted searches
5. Qwen synthesizes a context summary from all found docs
6. Return structured bundle with summary + source paths

This does NOT create an agent session — it makes a direct inference call to the local vLLM endpoint from within the Python MCP tool. No session bootstrap, no orchestrator, no workspace loading. Just fast LLM reasoning on free tokens.

**Latency:** 3-8s. **Cost:** $0 (local model). **Handles:** cross-topic, ambiguous, "find everything about X" queries.

### Comparison

| Approach | Latency | Cost | Tool Calls | When |
|----------|---------|------|------------|------|
| **Shallow** (Python graph traversal) | <500ms | $0 | 1 | Simple lookups, known topics |
| **Deep** (local LLM inline) | 3-8s | $0 | 1 | Cross-topic, ambiguous queries |
| **Current** (Opus multi-call) | 15-20s | $0.05-0.10 | 3-5 | ← being replaced |

### Impact on Lloyd's Tool Surface

Lloyd's retrieval surface area shrinks from 6 tools to 1 for most cases:

| Current | With `context_bundle` |
|---------|----------------------|
| `mem_search` → reasoning → `mem_get` → reasoning → `tag_search` → reasoning → `mem_get` | `context_bundle(query)` → answer |

Lloyd still has `mem_search`, `mem_get`, `tag_search`, `tag_explore`, `vault_overview` for specific/targeted lookups. But the default retrieval pattern becomes: **call `context_bundle`, answer from the result.**

### Enriched `mem_search` (Bonus)

Independent of `context_bundle`, `mem_search` results should include metadata already available in the index:

```json
{
  "path": "projects/alfie/gr00t/phase2-summary.md",
  "score": 0.89,
  "snippet": "Phase 2 complete...",
  "tags": ["alfie", "groot", "robotics"],
  "relations": {
    "superseded-by": ["phase3-summary.md"]
  },
  "stale": true
}
```

This lets Lloyd skip stale docs without a second call, even when using `mem_search` directly.

## Memory Agent Role

The existing memory agent (local Qwen, `agents/memory/`) continues its current jobs:
- Periodic capture (every 15 min)
- Nightly vault maintenance
- Nightly skills management

Its allowlist needs `mem_search`, `tag_search`, `tag_explore`, `vault_overview` added — it currently can't search the vault it manages.

For live retrieval, the memory agent is NOT in the hot path. `context_bundle` handles that directly. The memory agent remains a background worker for maintenance and batch processing.

## Implementation Phases

| Phase | What | Effort | Depends On |
|-------|------|--------|------------|
| **1. Schema** | Define frontmatter format, relation types, validation | Design only | — |
| **2. Relations index builder** | Python script: scan all frontmatter → `relations-index.json` | Medium | Phase 1 |
| **3. `context_bundle` shallow** | New MCP tool with BM25 + graph traversal | Medium | Phase 2 |
| **4. Enriched `mem_search`** | Return tags + relations in search results | Small | Phase 2 |
| **5. On-write indexing** | Incremental index update on `mem_write` | Small | Phase 2 |
| **6. `context_bundle` deep** | Add local LLM inline call for complex queries | Medium | Phase 3 |
| **7. Manual relation management** | Lloyd adds/removes relations via frontmatter updates | Convention | Phase 1 |
| **8. Automated discovery** | Nightly heuristic relation detection | Medium-High | Phase 2 |
| **9. Memory agent search tools** | Add `mem_search`, `tag_search`, `tag_explore`, `vault_overview` to allowlist | Config edit | — |

Critical path: **Phase 1 → 2 → 3** gets the core retrieval improvement live. Everything else layers on top.

## Open Questions

1. **Granularity** — whole documents or sections within documents? Whole docs is simpler and sufficient for 483 docs.
2. **Confidence decay** — should auto-discovered relations lose confidence over time if never confirmed?
3. **Obsidian Dataview** — should frontmatter be Dataview-compatible for graph view rendering?
4. **Relation density target** — every doc gets at least 1 relation? Or only where meaningful?
5. **Deep mode timeout** — hard cap at 10s? What if the local model is slow on a complex query?

## Related Docs

- [[memory-system]] — Memory System Architecture (current 3-tier design)
- [[mcp-tools]] — MCP Tools Server (where `context_bundle` would live)
- [[agent-system]] — Agent System (memory agent config)
