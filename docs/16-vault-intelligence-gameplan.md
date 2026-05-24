# 16 — Vault Intelligence: Conversation-Derived Relations + Proactive Research

> Gameplan for two interconnected features that make the vault smarter:
> mining conversation history for document relationships, and proactively
> filling knowledge gaps during live sessions.

## Context

Lloyd's vault has a mature relation system (6 typed relations in frontmatter, index builder, semantic heuristics) and a rich session history pipeline (trajectories, nightly reflection, memory capture). Two gaps exist:

1. **Relations are derived from static document analysis only** — shared tags, entity overlap, wiki-links. When Lloyd accesses docs A and B together in conversation to answer a question, that co-access signal is never captured as a relationship.
2. **Knowledge gaps are detected in batch only** (nightly groundskeeper scan). When Lloyd can't answer a question from vault content during a live conversation, it doesn't proactively research the gap.

This plan adds both capabilities as natural extensions of existing infrastructure.

---

## Feature 1: Conversation-Derived Relation Linking

### Approach

Two-stage pipeline running as an autonomy task (depends on trajectory extraction #45):

- **Stage 1 (deterministic):** Mine trajectory JSONL for document co-access pairs. Extract vault paths from tool calls (`vault_read`, `vault_search`, `file_read`, etc.), compute weighted co-access scores based on proximity (same call > adjacent > same session > same day), aggregate across sessions.
- **Stage 2 (LLM-assisted):** For high-confidence pairs (aggregate weight >= 0.8), extract surrounding conversation context from raw session JSON, use 122B to classify relationship type and extract a one-sentence reason.

### Files to Create

1. **`scripts/memory/conversation_relations.py`** — Core extraction engine
   - `normalize_vault_path(raw_path)` — Strips `~/obsidian/`, validates segment membership, returns vault-relative path
   - `extract_vault_docs_from_trajectory(entry)` — Maps tool names to path extraction rules (vault_read->path, skills_get->constructed path, etc.)
   - `extract_co_access_pairs(trajectory_dir, since_date)` — Stage 1: reads JSONL, builds (doc_a, doc_b) pairs with weights
   - `aggregate_pairs(pairs)` — Dampened sum across sessions: `sum(w_i) * (1 - 0.3^n)`
   - `extract_conversation_context(session_path, doc_a, doc_b)` — Finds message window around co-access in raw session JSON
   - `classify_relationship(doc_a, doc_b, context)` — Stage 2: LLM call to 122B at `localhost:8096`, returns `{type, reason, confidence}`
   - `deduplicate_against_index(proposals)` — Checks against existing `relations-index.json`
   - `auto_approve_strong(proposals, threshold=0.85)` — Auto-approve high-confidence proposals after 48h
   - CLI: `--incremental` (Stage 1), `--classify` (Stage 2), `--full`, `--stats`, `--approve-strong`

2. **`~/obsidian/autonomy/51-conversation-relation-linking.md`** — Autonomy task
   - `depends_on: 45` (trajectory extraction), `runs_per_day: 2`, `preferred_hours: [4, 5]`, model: 122B
   - Priority: medium, timeout: 900s

3. **`~/obsidian/skills/conversation-relation-linking/SKILL.md`** — Skill for autonomy agent
   - Run Stage 1, conditionally run Stage 2 if new high-weight pairs found, report top proposals

### Output Format

Proposals written to `~/obsidian/memory/_pipeline/conversation-relation-proposals.json`:

```json
{
  "watermark": { "last_trajectory_date": "2026-04-04", "sessions_processed": 181 },
  "proposals": [{
    "source": "projects/lloyd/phase-3.md",
    "target": "knowledge/ai/document-relations.md",
    "type": "related-to",
    "reason": "Both accessed when reviewing document relations architecture",
    "confidence": 0.85,
    "evidence": { "co_access_count": 3, "sessions": ["..."], "aggregate_weight": 1.6 },
    "status": "pending",
    "proposed_at": "2026-04-05T05:00:00Z"
  }]
}
```

### Integration with Existing Relations Index

- Proposals file is separate from `relations-index.json` (which is rebuilt from scratch by the data pipeline)
- Extend the data pipeline rebuild step to merge `status: "approved"` proposals into the index
- Auto-approval: proposals with confidence >= 0.85 and signal_strength "strong" get approved after 48h

### Weighting Scheme

| Signal | Weight | Example |
|--------|--------|---------|
| Same tool call (multi-doc) | 1.0 | vault_recall returns doc A + B |
| Adjacent calls (distance <= 2) | 0.8 | vault_read(A) then vault_read(B) |
| Same session (distance > 2) | 0.4 | Both accessed during same conversation |
| Same day, different sessions | 0.15 | Weak, only useful if reinforced |

Minimum aggregate weight to propose: 0.5. Minimum for LLM classification: 0.8.

### Filters

- Drop self-references, pipeline internals, daily-note-to-daily-note pairs
- Down-weight autonomy sessions (maintenance, not meaningful user work)
- Skip paths outside vault segments

---

## Feature 2: Proactive Research via Prompt + Existing Tools

### Approach

No new MCP module. Lloyd already has `pipeline_dispatch` (with a `research` stage), `vault_recall`, `vault_search`, `fact_add`, and web tools. The missing piece is **prompt instructions** that teach the agent to recognize knowledge gaps and use `pipeline_dispatch` to fill them asynchronously.

This is a prompt-first design: the agent's judgment drives when to research, the existing pipeline handles execution, and the existing notification system reports results back.

### What Changes

1. **`~/obsidian/lloyd/SOUL.md` or `MEMORY.md`** — Add a "Proactive Research" behavioral block with two-tier guidance
2. **`~/obsidian/skills/quick-research/SKILL.md`** (new) — Fast inline research: agent does it within the current turn
3. **`~/obsidian/skills/deep-research/SKILL.md`** (existing or new) — Async pipeline research for bigger topics

No new Python code. No new MCP tools. No queue infrastructure.

### Two-Tier Research Model

| | Quick Research | Deep Research |
|---|---|---|
| **When** | Small gap, user waiting, answer needed now | Big topic, breadth needed, user not blocked |
| **How** | Agent does it inline: 1-2 web searches, extract facts, continue | `pipeline_dispatch` with research stage, runs async |
| **Duration** | 30-60s within current turn | 5-15min in background |
| **Output** | Direct `fact_add` + answer the user with findings | Knowledge doc + facts, pipeline notifies on completion |
| **Blocking** | Yes — user waits briefly but gets a complete answer | No — user continues, results arrive later |
| **Skill** | `quick-research` | `deep-research` |

### Prompt Instructions (SOUL.md / MEMORY.md)

```markdown
## Proactive Research

When vault_recall returns thin results (<2 relevant hits) for a factual/technical
topic, choose a research tier:

**Quick research** (default) — do it now, inline:
- 1-2 targeted web searches on the specific question
- Read the top results, extract the key facts
- fact_add the findings immediately so the vault has them next time
- Answer the user with what you found — no need to mention "research"
- Use when: the gap is narrow, the user needs an answer now

**Deep research** — dispatch to background pipeline:
- pipeline_dispatch(task="Research: [topic]...", stages=["research"], model="122b")
- Tell the user: "I've kicked off deeper research on [topic]."
- Continue the conversation — don't block
- Use when: the topic is broad, multiple angles needed, or the user
  explicitly asks for a thorough investigation

When NOT to research:
- Personal/private topics (family, schedule, preferences)
- Things the user just told you (they're the source)
- Trivial or conversational topics
- Topics you've already researched in this session
```

### Skill: Quick Research

`~/obsidian/skills/quick-research/SKILL.md` — fast inline gap-filling:

```markdown
---
name: quick-research
description: Fast inline research to fill small vault knowledge gaps during conversation
category: memory
tags: [research, knowledge, gaps, inline, fast]
---

# Quick Research

Lightweight, inline research for narrow knowledge gaps. You do this within
your current turn — no pipeline dispatch, no background process.

## When to Use
- vault_recall returned <2 results for a specific factual question
- The gap is narrow and well-defined (not "tell me everything about X")
- The user is waiting for an answer

## Procedure
1. Identify the specific question the vault can't answer
2. 1-2 targeted web searches (http_fetch or web search tools)
3. Read the most relevant result(s)
4. Extract 2-5 key facts via fact_add(entity, category, fact, confidence, source_doc)
5. Answer the user directly with findings — weave it into your response naturally
6. Do NOT mention "research" or "gap" — just answer better than you could before

## Quality Bar
- Every fact_add must include source_doc (the URL you found it from)
- Confidence scores: 0.7-0.8 for web sources (not as reliable as primary docs)
- If you can't find anything useful in 1-2 searches, say so and move on —
  don't burn 5+ searches on a dead end

## What This Is NOT
- Not a deep dive — if the topic needs 5+ sources or multiple angles, use
  deep-research (pipeline_dispatch) instead
- Not for opinions or subjective topics
- Not a substitute for asking the user when they likely know the answer
```

### Skill: Deep Research

`~/obsidian/skills/deep-research/SKILL.md` — async pipeline research for bigger topics:

```markdown
---
name: deep-research
description: Dispatch thorough background research via pipeline for broad vault gaps
category: memory
tags: [research, knowledge, gaps, pipeline, async, deep]
---

# Deep Research

Background research for broad topics that need multiple sources and synthesis.
Dispatches via pipeline_dispatch — runs async, notifies you when done.

## When to Use
- Topic is broad ("what's the state of X?", "compare approaches to Y")
- User explicitly asks to "research", "look into", "investigate"
- Quick research wouldn't be sufficient (needs 5+ sources, multiple angles)
- User isn't blocked waiting for this specific answer

## How to Dispatch

pipeline_dispatch(
  task="Research: {topic}\n\nContext: {why it came up}\n\nInstructions:\n
    1. vault_recall first — check what we already have\n
    2. Web search with 3-5 varied queries from different angles\n
    3. Fetch and read 3-5 primary sources\n
    4. Extract structured facts via fact_add(entity, category, fact, confidence)\n
    5. Write knowledge doc to knowledge/{domain}/{slug}.md if >3 findings\n
    6. Cross-reference existing vault content to avoid contradictions",
  stages=["research"],
  model="122b"
)

## After Dispatch
- Tell the user research is running in the background
- Continue the conversation — don't block on it
- Pipeline notification delivers results when done
- If user asks for status, use pipeline_status to check

## What This Is NOT
- Not for quick factual lookups — use quick-research instead
- Not for personal/private topics
- Not for things the user just told you
```

### How It Works End-to-End

**Quick research (inline):**
1. User asks about topic X
2. Lloyd calls `vault_recall("topic X")` — thin results
3. Lloyd does 1-2 web searches inline, reads results
4. Lloyd calls `fact_add` to persist findings to vault
5. Lloyd answers the user directly with the findings
6. Next time topic X comes up, vault_recall finds the facts

**Deep research (async):**
1. User asks about broad topic Y, or says "research Y for me"
2. Lloyd calls `vault_recall("topic Y")` — thin results, topic is broad
3. Lloyd calls `pipeline_dispatch(task="Research: Y...", stages=["research"], model="122b")`
4. Pipeline spawns daemon thread running research stage with full tool access
5. Research agent: web searches, reads sources, calls `fact_add`, writes knowledge doc
6. Pipeline completes, `_notify_requester_session` sends summary back to session
7. Next time topic Y comes up, vault_recall finds the knowledge doc and facts

### Why This Works Without Dedicated Infrastructure

- **Dedup:** The agent sees its own conversation history — it won't research the same topic twice in one session. Across sessions, the research output itself deduplicates (vault_recall finds it next time, so the gap trigger won't fire).
- **Budget:** Quick research costs 1-2 web fetches (trivial). Deep research uses the local 122B (free). Pipeline's `stage_timeout` caps runaway agents.
- **Status tracking:** `pipeline_status` already exists for deep research. Quick research completes inline — no tracking needed.
- **Notification:** Pipeline's `_notify_requester_session` handles deep research callbacks. Quick research answers the user immediately.

---

## Implementation Order

| Phase | Work | Files |
|-------|------|-------|
| 1 | Co-access extraction (Stage 1) | `scripts/memory/conversation_relations.py` |
| 2 | LLM classification (Stage 2) | `scripts/memory/conversation_relations.py` |
| 3 | Autonomy task + skill for relations | `~/obsidian/autonomy/51-*.md`, `~/obsidian/skills/conversation-relation-linking/` |
| 4 | Quick research skill | `~/obsidian/skills/quick-research/SKILL.md` |
| 5 | Deep research skill | `~/obsidian/skills/deep-research/SKILL.md` |
| 6 | Research prompt instructions | `~/obsidian/lloyd/SOUL.md` or `MEMORY.md` |
| 7 | Merge approved proposals into index | Data pipeline rebuild step |

## Verification

- **Relations:** Run `conversation_relations.py --full --stats` against existing trajectory data, verify proposals are generated with reasonable types/confidence. Check that `relations-index.json` grows after approval.
- **Research:** In a live session, ask about an obscure topic. Verify Lloyd notices thin vault results, dispatches a `pipeline_dispatch` research run, and tells the user. Check that facts appear in the vault after pipeline completes. Ask about the same topic in a new session — verify vault_recall now returns the research output.
- **Integration:** Run full nightly pipeline (trajectory extraction -> conversation relations -> relation index rebuild) and verify end-to-end flow.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Trajectory JSONL as primary input for relations (not raw sessions) | Already scrubbed, structured, date-bucketed. Raw sessions only consulted in Stage 2 for causal context. |
| Separate proposals file (not direct relations-index writes) | `relations-index.json` is rebuilt from scratch by data pipeline. Direct writes would be blown away. |
| Prompt-based research (not dedicated MCP module) | Lloyd already has `pipeline_dispatch` + research stage + vault tools + web tools. Adding a queue/dedup/budget MCP server is infrastructure for infrastructure's sake. The agent's own judgment handles dedup (conversation history), and pipeline handles execution/notification. |
| Two-tier research (quick + deep) | Quick inline research (1-2 searches, fact_add, answer immediately) handles 80% of gaps without pipeline overhead. Deep research via pipeline_dispatch handles broad topics async. The agent picks the tier based on gap breadth and user context. |
| Two separate skill files | Keeps guidance specific to each tier. Agent discovers the right one via `skills_search` based on the situation. SOUL.md just has the high-level decision framework. |
| No automatic vault_search hooks | Many empty searches aren't research-worthy (personal, contextual). The agent must decide if the gap is researchable. Prompt-driven judgment > mechanical triggers. |
