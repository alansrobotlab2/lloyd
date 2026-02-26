# Evaluation: Context Fill Path Before First LLM Call

**Date:** 2026-02-24

## Context
Two plugins fire `before_prompt_build` hooks to inject context before the LLM ever sees the prompt. This analysis maps the full pipeline, identifies redundancies and gaps, and suggests improvements.

## The Pipeline (sequential)

```
User prompt arrives
  │
  ├─ 1. memory-prefetch  (priority 0)
  │     ├─ Extract user query from envelope
  │     ├─ Simplify to keywords (filler removal, max 6 words)
  │     ├─ PARALLEL:
  │     │   ├─ memory_search(keywords, maxResults:5) via HTTP gateway
  │     │   └─ GLM keyword extraction (Qwen3, if query ≥40 chars)
  │     ├─ memory_get for top 3 results with score ≥ 0.7
  │     ├─ Extra memory_search ×2 from GLM keywords (maxResults:3 each)
  │     └─ Returns: <memory_prefetch> block (≤8000 chars)
  │         Contains: file contents (1500 chars each) + snippets
  │
  ├─ 2. memory-graph  (priority 200)
  │     ├─ Extract topic keywords (filler removal + bigrams)
  │     ├─ Match keywords → vault tags (case/hyphen insensitive)
  │     ├─ For each matched tag (max 3 topics):
  │     │   └─ Top 4 docs sorted by importance score
  │     └─ Returns: <vault_context> block (no size cap)
  │         Contains: path, type, title, summary, tags per doc
  │
  └─ LLM receives: <memory_prefetch> + <vault_context> + system prompt + user prompt
```

## Findings

### 1. Complementary Strengths (Good)
- **memory-prefetch** does **semantic vector search** — finds docs even without exact tag matches
- **memory-graph** does **structural tag matching** — finds docs by explicit topic classification
- These aren't redundant approaches — they catch different things

### 2. Overlapping Documents (Problem)
Both plugins can surface the **same document** from different angles. A doc about "alfie" with tag `alfie` will likely appear in both blocks. The LLM sees it twice — once with 1500 chars of content (prefetch), once with just metadata (graph). This wastes context window without adding information.

### 3. memory-graph Has No Size Cap (Risk)
memory-prefetch caps at 8000 chars. memory-graph has **no limit**. With 3 topics x 4 docs each = 12 doc entries, the `<vault_context>` block can grow unchecked. Combined with prefetch's 8000 chars, the pre-call injection could be 10-12k+ chars.

### 4. Duplicate Keyword Extraction (Waste)
Both plugins:
- Strip the same OpenClaw metadata envelope with the same regex
- Remove filler words using nearly identical word sets (76 shared words)
- memory-graph has 7 extra: `use, using, search, memory, related, connected, tag`

This work happens twice sequentially. memory-graph doesn't see memory-prefetch's results or extracted keywords.

### 5. HTTP Gateway Round-Trip (Latency)
memory-prefetch invokes `memory_search` and `memory_get` via HTTP POST to `localhost:18789`. These tools are already loaded in the same process via memory-core. The HTTP serialization/deserialization adds unnecessary latency inside a 2-second budget.

### 6. memory-graph Doesn't Provide Content (Gap)
memory-graph only injects metadata (path, title, summary, tags). If the LLM wants to answer from a tag-matched doc, it still needs to call `memory_get` or `read`. Meanwhile, prefetch already fetches full content for its matches. If a doc appears only in graph's results (not prefetch's), the LLM has a pointer but no content.

### 7. GLM Call May Be Wasted (Efficiency)
When GLM keywords happen to match existing vault tags, the tag-based path would have found those docs anyway (faster, from the in-memory index). The GLM extraction is most valuable for semantic variations that tags wouldn't capture.

## Potential Improvements

### A. Deduplicate at the graph hook
memory-graph (priority 200) runs after prefetch. It could read the already-injected `prependContext` from the event/context and skip docs that prefetch already surfaced. This eliminates duplicate entries.

### B. Add a size cap to memory-graph
Cap `<vault_context>` at ~3000 chars (keeping combined total under ~11k). Or share a budget between both hooks.

### C. Merge into a single hook
Instead of two independent plugins, a coordinator could:
1. Run tag matching (instant, in-memory) and vector search (async) in parallel
2. Merge results, deduplicate by path
3. Fetch full content for top N unique results
4. Produce one combined context block

This eliminates duplicate keyword extraction, deduplicates results, and gives a single size budget.

### D. Skip gateway HTTP for memory_search
If the memory-core tool can be invoked directly (same process), avoid the HTTP round-trip. Would need to check if the plugin API exposes a way to call sibling plugin tools directly.

### E. Use tag matches to guide vector search
Instead of running independently, use matched tags as a signal to boost or filter vector results. E.g., if "alfie" matches a tag, the vector search could deprioritize docs already covered by that tag.

## Recommendation

The least disruptive improvement is **A + B**: have memory-graph check what prefetch already injected and skip duplicates, plus add a size cap. This preserves the two-plugin architecture while fixing the main issues (duplicate docs, unbounded size).

The most impactful improvement is **C**: merging into a single coordinated hook that runs both search strategies in parallel, deduplicates, and produces one context block under a shared budget. This is more work but eliminates all redundancy.
