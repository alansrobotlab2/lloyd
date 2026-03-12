---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Memory System Architecture

Lloyd uses a 3-tier memory architecture that combines automated capture, nightly analysis, and real-time behavioral signal detection. The system writes to and reads from the [[index|Obsidian Vault]].

## Tier 1: Periodic Capture

Automated session transcript extraction running every 15 minutes.

- **Cron job:** `periodic-memory-capture` (ID: `06a1d2b2`), every 15 minutes
- **Agent:** OpenClaw `memory` agent backed by local Qwen3.5-35B-A3B
- **Agent workspace:** `~/obsidian/agents/memory/`
- **Script:** `~/obsidian/agents/memory/scripts/extract-transcript.py`
- **Input:** JSONL session logs from `~/.openclaw/agents/main/sessions/`
- **Processing:** Filters to user/assistant text, strips thinking blocks and tool calls
- **State watermark:** `~/obsidian/agents/memory/state.json` (tracks last-processed position)
- **Output:** `~/obsidian/agents/lloyd/memory/YYYY-MM-DD.md` (daily notes with Mission Control session deep-links)

## Tier 2: Nightly Jobs

Three nightly jobs run in sequence, all using Opus 4.6 via Anthropic API.

| Time (PST) | Job | Purpose |
|------------|-----|---------|
| 2:00am | Vault maintenance | Tag hygiene, frontmatter validation, structure optimization |
| 3:00am | Skills management | Session log extraction, skill evaluation, deduplication |
| 4:00am | Reflection + self-improvement | Corrections review, mental models, MEMORY.md consolidation |

### Vault Maintenance (2am)

Structural upkeep of the Obsidian vault. Tag deduplication, frontmatter consistency, recently-modified doc review. See [[nightly-vault-maintenance]] for the full procedure.

### Skills Management (3am)

Three-stage pipeline: extract session logs, evaluate for skill-worthy patterns, create/update/deduplicate skills. See [[nightly-skills-management]] for the full procedure.

### Reflection + Self-Improvement (4am)

Deep analysis of behavioral signals, expanded into a full recursive self-improvement loop. See [[nightly-reflection]] for the procedure and full architecture.

**Seven phases:**

1. **Pre-Flight Snapshots** -- Git snapshot of `~/obsidian` and `~/.openclaw` before changes
2. **Self-Improvement Review** -- Process `corrections.md` signals, apply SOUL.md / AGENTS.md fixes
3. **Mental Models** -- Update `memory/mental-models.md` with Alan's reasoning patterns
4. **MEMORY.md Consolidation** -- Distill daily notes into long-term memory, keep under 200 lines
5. **Config Improvements** -- Propose and apply `.openclaw` config changes from behavioral signals
6. **Post-Improvement Snapshots** -- Git commit all changes
7. **Summary Report** -- Changelog to `memory/learnings/YYYY-MM-DD.md`, announce summary

## Tier 3: Real-time Signal Detection

Lloyd detects corrections during live conversation.

- **Explicit signals:** "bad lloyd" / "good lloyd"
- **Implicit signals:** "actually...", "perfect", "no that's wrong", etc.
- **Output:** Logged to `memory/corrections.md` with signal type (positive/negative), category, and status
- **Processing:** Accumulated signals are processed during nightly reflection (Tier 2, 4am)

## Memory Flow

```
Tier 3: Real-time
  Conversation --> corrections.md (good/bad lloyd, implicit signals)

Tier 1: Periodic (every 15m)
  Session JSONL logs --> extract-transcript.py --> memory/YYYY-MM-DD.md

Tier 2: Nightly
  2am  Vault maintenance --> tag/frontmatter fixes --> vault-maintenance/YYYY-MM-DD.md
  3am  Skills management --> session logs --> skill creation/updates
  4am  Reflection:
         corrections.md ----> SOUL.md / AGENTS.md updates
         corrections.md ----> .openclaw config updates
         daily notes -------> mental-models.md
         daily notes -------> MEMORY.md consolidation (~200 lines)
         all changes -------> learnings/YYYY-MM-DD.md
         git commit both repos
```

## Key Files

| File | Purpose |
|------|---------|
| `agents/lloyd/MEMORY.md` | Long-term curated memory (~200 lines max) |
| `agents/lloyd/memory/YYYY-MM-DD.md` | Daily notes (periodic capture output) |
| `agents/lloyd/memory/personal/YYYY-MM-DD.md` | Personal mode daily notes |
| `agents/lloyd/memory/work/YYYY-MM-DD.md` | Work mode daily notes |
| `agents/lloyd/memory/corrections.md` | Behavioral signal log |
| `agents/lloyd/memory/mental-models.md` | Alan's reasoning patterns |
| `agents/lloyd/memory/learnings/YYYY-MM-DD.md` | SOUL.md/AGENTS.md change log |
| `agents/lloyd/memory/heartbeat-state.json` | Heartbeat timestamps |
| `agents/memory/state.json` | Periodic capture watermark |
| `memory/vault-maintenance/YYYY-MM-DD.md` | Vault maintenance run reports |

## Vault Search

The vault is searchable through multiple mechanisms:

- **BM25 FTS5 full-text search** via `mem_search` tool (no vector/semantic search -- pure BM25)
- **Tag-based search** via memory-graph plugin: `tag_search`, `tag_explore`, `vault_overview`
- **5 vault segments** with scope filtering: agents, personal, work, projects, knowledge
- **Mode system** (work/personal/general) auto-scopes searches

## Prefill Pipeline (Context Injection)

Automated context injection via the `before_prompt_build` hook in the [[tools|mcp-tools extension]].

- **Turn 1:** Injects yesterday's + today's daily notes + active mode tag
- **Turn 2:** Semantic prefill using turn-1 query as search input via `prefill_context`
- **Turn 3+:** No prefill (conversation history carries context)
- **Context profiles:** chat, memory, code, research, ops, voice, heartbeat
- **Automated prompts** (heartbeat, cron) skip prefill entirely
- **`prefill_context` tool:** Uses BM25 + tag matching + GLM keywords

## Related Docs

- [[index]] -- High-Level Architecture
- [[nightly-reflection]] -- Nightly Reflection procedure (includes Tier 2 self-improvement)
- [[nightly-skills-management]] -- Nightly Skills Management procedure
- [[nightly-vault-maintenance]] -- Nightly Vault Maintenance procedure
- [[tools]] -- MCP Tools Server (tool implementations)
- [[agents]] -- Agent System (memory agent details)
- [[infrastructure]] -- Infrastructure (cron system)
