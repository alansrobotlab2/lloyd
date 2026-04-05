# Plan: Automatic Context Prefetch Layer

## Context

Lloyd has 107+ skills, ~15K facts, and a searchable vault — but none of this is proactively surfaced. The agent must remember to call `skills_search` as its first action (SOUL.md mandates this), but LLMs unreliably follow the instruction. The result: the agent either wastes a turn searching, or skips the search entirely and operates without relevant skill knowledge.

The fix: a server-side **context prefetch** that runs before every `query()` call, automatically injecting relevant skill content and facts into the user message.

## Approach

Create `prefetch.py` — a new module that extracts keywords from the user message, searches skills and facts in parallel, and prepends a `<context>` block to the prompt. This mirrors what `autonomy.py` already does for scheduled tasks (loading skill content into the prompt), extended to interactive messages.

### Why user message, not system prompt?

The system prompt is deliberately kept stable for vLLM prefix cache efficiency (see `prompt_builder.py:44` comment). The user message changes every turn anyway — zero cache penalty.

## Files to Create/Modify

### 1. Create `prefetch.py` (~100 lines)

```python
prefetch_context(text: str) -> str
```

- **Skip check**: Messages under 10 chars or all noise words → return original text
- **Skill search**: Reuse `_iter_skills()`, `_score_skill()`, `_tokenize()` from `mcp_server/skills.py`. Cache parsed skills for 5 min. Return top 1-2 skills above score threshold (≥3.0 for first, ≥4.0 for second)
- **Fact search**: Reuse `_extract_entities_from_query()`, `_get_facts_sync()` from `mcp_server/memory.py`. Top 2 entities, top 3 facts each by confidence
- **Parallel execution**: `concurrent.futures.ThreadPoolExecutor(max_workers=2)` — skills and facts run concurrently
- **Format output**: Prepend `<context>` block with:
  - Top skill: full SKILL.md body, truncated to 6000 chars
  - Second skill: 500-char excerpt only
  - Facts: bullet list with entity name and confidence
- **Budget**: ~1,800 tokens max injected context

### 2. Modify `server.py` (~8 lines)

- Add `from prefetch import prefetch_context` import
- **Streaming endpoint** (`post_message_stream`, line ~335): After `system_prompt = build_system_prompt()`, add:
  ```python
  prefetched_text = prefetch_context(text)
  ```
  Change `query(prompt=text, ...)` → `query(prompt=prefetched_text, ...)`  
  Keep `text` (not prefetched) in the user message persistence (line ~407) so chat history stays clean
- **Sync endpoint** (`post_message`, line ~697): Same pattern — prefetch before `query()`, use original text for persistence
- Add `t_prefetch` timing to existing TIMING log

### 3. Modify `SOUL.md` (lines 41-46)

Replace "Skill Check (MANDATORY)" section with:

```markdown
### Skills

Relevant skill content is automatically injected into your prompt as `<context>` when matched.
- If the injected skill is correct, follow it — no need to call skills_search first
- If you need a different skill, or no context was injected, call skills_search
- After completing a novel workflow not covered by existing skills, write a new skill
```

### 4. Optional: Modify `prompt_builder.py` (1 line)

Update the `<available_skills>` note to mention auto-injection, so the agent understands the context block it sees.

## Output Format

```xml
<context>
<skill name="research-agent" score="7.5">
[full SKILL.md body, ≤6000 chars]
</skill>
<skill name="web-search" score="4.5" excerpt="true">
[500-char excerpt]
</skill>
<facts>
- [lloyd] MCP servers run via unified SSE on port 8500 (confidence: 0.9)
- [thunderbird] Extension at localhost:8765 (confidence: 0.85)
</facts>
</context>

[original user message]
```

When nothing matches, the original text is returned unchanged.

## Performance

| Operation | Latency |
|---|---|
| Keyword extraction | <1ms |
| Skill scoring (182 skills, warm cache) | ~38ms |
| Fact retrieval (2 entities) | ~60-100ms |
| **Total (parallel)** | **~100-140ms** |
| Cold start (first call, loads all skills) | ~170ms |

Well within the 200ms budget. QMD vault search (~3.4s) is intentionally excluded — the agent can still call `vault_recall` manually.

## What This Does NOT Change

- System prompt structure (prefix cache safe)
- `skills_search`/`skills_read` MCP tools (still available for manual use)
- Autonomy task flow (already has its own skill injection)
- Vault/QMD search (too slow for synchronous prefetch)

## Verification

1. Restart backend: `distrobox enter lloyd -- supervisorctl -c ... restart lloyd-mc:lloyd-backend`
2. Send a message mentioning a known skill topic (e.g., "compose an email") → verify `<context>` block appears in the prompt (visible in server logs via TIMING line)
3. Send a short message ("ok") → verify no context block is injected
4. Check latency in logs stays under 200ms
5. Verify the agent follows injected skill instructions without calling `skills_search` first
