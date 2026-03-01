# Tools

## Memory & Vault Search
- memory_search — Semantic vector search across Obsidian vault, MEMORY.md, and daily-notes. Good for natural-language queries where you don't know exact tags.
  - After memory_search returns results, immediately READ the top 2-3 result files in the same round trip — don't issue a second memory_search first.
  - Use `read` (not `memory_get`) for project and vault files. Only use `memory_get` for MEMORY.md and memory/YYYY-MM-DD.md.
  - Vault paths are all lowercase with hyphens (e.g. `~/obsidian/projects/alfie/phase2-summary.md`). QMD search results map directly to `~/obsidian/<path>`.

## Tag Tools (memory-graph plugin)
- tag_search — Search vault by tags. Returns document title, summary, and all tags for each match. Faster and more precise than memory_search when you know the topic area.
  - `tags`: array of tags (no # prefix), e.g. ["alfie"], ["ai", "rag"]
  - `mode`: "or" (any tag, default) or "and" (all tags — use for intersection queries)
  - `type`: filter by doc type (hub, notes, project-notes, work-notes, talk)
  - Use tag_search when the user asks about a known project, topic, or domain. Use memory_search for open-ended or natural-language queries.
- tag_explore — Discover tag relationships. Shows co-occurring tags for a given tag, and optionally finds documents bridging two tags.
  - Use when exploring connections between topics or finding what's related to a concept.
- vault_overview — Vault statistics: doc/tag counts, type distribution, hub pages, tag frequencies.
  - Use when you need to understand what's in the vault or list available tags.

## When to Use Which
- **Known topic/project** → tag_search (e.g. "what do we have on alfie?" → tag_search(["alfie"]))
- **Natural-language question** → memory_search (e.g. "how did we set up the arm controller?")
- **Exploring connections** → tag_explore (e.g. "what's related to robotics?" → tag_explore("robotics"))
- The before_prompt_build hook auto-injects relevant vault docs when your query matches tags, so context is often already available before you call any tool.

## Knowledge Lookup Flow

When a question comes in, follow this order:
1. **Memory context** (pre-injected) — check first, often sufficient
2. **tag_search / memory_search** — if memory context doesn't cover it
3. **web_search + web_fetch** — always fair game, even if a knowledge doc exists

After any web lookup, **create or update** `~/obsidian/lloyd/knowledge/<domain>/<slug>.md`.
- New topic → create the doc
- Existing doc → update with new info, bump the date, add sources
See AGENTS.md → "Web Lookup Capture" for format and domains.
This keeps the vault growing as a living, up-to-date reference library.

## Response Format
- Responses are spoken via TTS. Keep replies under 3 sentences. Plain text only, no markdown.

## Skills
Operational knowledge lives in `skills/` — check there before improvising a complex procedure.
- `skills/websearch.md` — Web research workflow: search → answer with source links → save/update knowledge vault doc
- `skills/claude-code-subagent.md` — How to launch Claude Code in tmux + Kitty window
- `skills/obsidian-vault-maintenance.md` — Periodic QMD memory backend maintenance: audit, enrich, rebuild index
