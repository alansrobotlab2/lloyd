# Memory Segmentation Model

**Date:** 2026-03-03
**Status:** Active

---

## Motivation

Before segmentation, the Obsidian vault was a flat collection of directories with no structural boundaries between agent-generated content, personal notes, work materials, and reference knowledge. As the agent framework grew to 10 agents — each capable of writing to the vault — three problems emerged:

1. **Search noise**: A scoped lookup (e.g. "what do I know about alfie?") would surface irrelevant agent notes, work meeting logs, and reference papers alongside project content.
2. **Write ambiguity**: No agent had a clearly defined write target. Notes ended up wherever seemed convenient, making the vault increasingly incoherent.
3. **No enable/disable axis**: There was no way to say "this agent should never see personal content" or "this prefill context should skip agent workspace files."

Segmentation addresses all three by imposing a canonical directory hierarchy, tagging every document with its segment, and wiring the memory tools to filter by segment on demand.

---

## Vault Structure

Five canonical top-level segments. Every document in the vault belongs to exactly one.

```
~/obsidian/
├── agents/           # Agent-generated content
│   ├── lloyd/        # Lloyd main agent workspace
│   ├── orchestrator/ # Orchestrator workspace
│   ├── coder/        # Coder workspace
│   ├── memory/       # Memory agent workspace
│   ├── researcher/   # Researcher workspace
│   ├── operator/     # Operator workspace
│   ├── tester/
│   ├── reviewer/
│   ├── planner/
│   ├── auditor/
│   └── shared/       # Cross-agent notes (e.g. heartbeat state)
├── personal/         # Personal life, reflections
│   └── dreams/
├── work/             # Professional work
│   ├── aveva/
│   └── rssc/
├── projects/         # Personal / hobby projects
│   ├── alfie/
│   ├── arch/
│   ├── brayden/
│   ├── stompy/
│   ├── freecad/
│   ├── uncle-ron/
│   └── home-improvements/
├── knowledge/        # Reference material
│   ├── ai/
│   ├── hardware/
│   └── papers/
└── templates/        # Vault templates (not searchable)
```

**Segment ownership:**

| Segment | Who writes here |
|---------|----------------|
| `agents` | Lloyd (daily notes, MEMORY.md), Memory agent (shared notes), any agent writing self-notes |
| `personal` | Lloyd only (personal reflections, dreams log) |
| `work` | Researcher (work-domain findings), Lloyd (meeting notes) |
| `projects` | Coder (gameplan docs, writeups), Researcher (project research), Planner |
| `knowledge` | Researcher (reference articles, paper summaries, web research), websearch skill |

---

## Frontmatter Convention

Every vault document carries a `segment:` field in its YAML frontmatter. This is the machine-readable hook used by the tag index and tool filtering.

```yaml
---
title: "Alfie"
type: hub
segment: projects
tags: [alfie, robotics, ros2]
summary: "Alfie is Alan's humanoid robot..."
---
```

Valid values: `agents`, `personal`, `work`, `projects`, `knowledge`

The field was batch-added to all 391 existing vault documents on 2026-03-03 using a Python script that derives the segment from the path prefix. New documents should include `segment:` in their frontmatter — the researcher and websearch skill templates already do this.

---

## Memory Tools and the `scope` Parameter

Three memory tools accept an optional `scope` parameter that restricts results to one or more segments.

### `mem_search`

```python
mem_search(query: str, max_results: int = 10, min_score: float = 0.0, scope: str = "")
```

Calls `qmd query` (hybrid BM25 + vector search) against the full vault, then post-filters the JSON results by path prefix. Returns only documents whose vault-relative path starts with one of the resolved segment prefixes.

```
mem_search("alfie robot arm", scope="projects")
# → only returns results from projects/
```

### `tag_search`

```python
tag_search(tags: list[str], mode: str = "or", type: str = "any", limit: int = 10, scope: str = "")
```

Filters the in-memory TagIndex at query time. The tag-to-doc index contains all documents; `scope` applies a path prefix filter on the matched document set before ranking.

```
tag_search(["ros2", "embedded"], scope="projects,knowledge")
# → documents in projects/ or knowledge/ with those tags
```

### `prefill_context`

```python
prefill_context(prompt: str, session_id: str = "", scope: str = "")
```

The full prefill pipeline (tag match → BM25 → GLM keyword extraction → merge → rank → fetch) now threads `scope` through all internal calls:
- The tag match post-filters `tag_docs` by scope prefix before scoring
- The `mem_search` calls (initial + GLM extra) pass `scope` through
- The cache key includes the scope value, so `scope="projects"` and `scope=""` cache independently

### Scope value format

The `scope` parameter accepts:
- Empty string or omitted → no filtering (all segments)
- Single segment: `"projects"`
- Multiple segments (comma-separated): `"projects,knowledge"`
- Sub-path (passthrough): `"projects/alfie"` → only `projects/alfie/` subtree

Internally, `_resolve_scope_prefixes()` normalizes all forms to a list of path prefixes with trailing slash.

---

## Profile-Based Scope Routing

The `before_prompt_build` hook in `extensions/mcp-tools/index.ts` classifies each incoming user turn into a context profile and maps that to a search scope before calling `prefill_context`.

```typescript
const PROFILE_SCOPE: Record<ContextProfile, string> = {
  memory:   "",                          // all segments — explicit recall turn
  research: "knowledge,projects,work",   // no personal/agent noise
  default:  "",                          // all segments
  chat:     "",   // skipped (SKIP_PREFILL_PROFILES)
  code:     "",   // skipped
  ops:      "",   // skipped
  voice:    "",   // skipped
};
```

**Profile detection** (regex-based, in order):
1. `chat` — short greeting/ack patterns (< 50 chars) → skip prefill entirely
2. `voice` — TTS/narrate keywords → skip
3. `code` — code blocks or implement/debug/refactor → skip
4. `memory` — "remember", "what did we", "recall", "last session" → prefill, all segments
5. `ops` — restart/deploy/git push/Backlog → skip
6. `research` — "search for", "look up", "what is" → prefill, knowledge+projects+work
7. `default` — everything else → prefill, all segments

The research profile is the most impactful: it prevents agent workspace files (SOUL.md, AGENTS.md, daily notes) and personal content from polluting research context. A question like "what is the latest on GR00T N1?" only draws from `knowledge/`, `projects/`, and `work/`.

---

## End-to-End Flow

### Example: "What do I know about the alfie robot arm?"

1. User prompt arrives at `before_prompt_build`.
2. Profile classified as `default` (doesn't match chat/voice/code/memory/ops/research patterns).
3. `PROFILE_SCOPE["default"]` → `""` (all segments).
4. `prefill_context(prompt, scope="")` called.
5. Tag match finds `alfie` in `projects/alfie/alfie.md` and related docs.
6. `mem_search("alfie robot arm", scope="")` returns results from across the vault.
7. Results merged, ranked, top-1 fetched (`projects/alfie/alfie.md`).
8. `<memory_context>` block prepended to system prompt.

### Example: "Search for the latest transformer efficiency papers"

1. Profile classified as `research` (matches RESEARCH_RE).
2. `PROFILE_SCOPE["research"]` → `"knowledge,projects,work"`.
3. `prefill_context(prompt, scope="knowledge,projects,work")` called.
4. Tag match filtered to docs in `knowledge/`, `projects/`, `work/` only.
5. `mem_search("transformer efficiency papers", scope="knowledge,projects,work")` — results post-filtered.
6. Agent's SOUL.md, daily notes, personal dream logs are never returned.

### Example: Agent writes research findings

Researcher agent calls:
```
mem_write("knowledge/ai/transformer-efficiency-2026.md", content)
```
The path starts with `knowledge/` — correct segment. The frontmatter template in the researcher's AGENTS.md includes `segment: knowledge` so the doc is tagged on creation.

---

## Agent Default Scopes (Reference)

| Agent | Default Read Scope | Default Write Scope | Notes |
|-------|-------------------|---------------------|-------|
| Lloyd | all | `agents/lloyd/` | Daily notes, MEMORY.md |
| Memory | all | `agents/shared/` | Cross-agent memory |
| Researcher | `knowledge,projects,work` | `knowledge/` | Via AGENTS.md guidance |
| Coder | `projects,agents` | `agents/coder/` | Gameplan, writeup docs |
| Operator | `agents,projects` | `agents/shared/` | Operational notes |
| Orchestrator | `agents` | — (read-only) | Coordination only |
| Planner | `projects,knowledge` | — (read-only) | |
| Tester | `projects` | — (read-only) | |
| Reviewer | `projects` | — (read-only) | |
| Auditor | all | — (read-only) | Security-sensitive, needs all |

These are defaults encoded in agent AGENTS.md files. Agents can override scope per-call when the task requires it (e.g. a researcher writing work-domain notes to `work/` instead of `knowledge/`).

---

## Implementation Files

| File | Role |
|------|------|
| `~/Projects/lloyd-services/tool_services.py` | `_resolve_scope_prefixes()`, scope params on `mem_search`, `tag_search`, `prefill_context`; `TagIndex.search_by_tags()` scope filtering |
| `extensions/mcp-tools/index.ts` | `PROFILE_SCOPE` map, scope passed to `prefill_context` in `before_prompt_build` hook |
| `~/obsidian/agents/lloyd/AGENTS.md` | Memory Segments table for Lloyd |
| `~/obsidian/agents/researcher/AGENTS.md` | Segment table + scope usage examples, correct `knowledge/` write paths |
| `~/obsidian/agents/lloyd/skills/websearch/SKILL.md` | Updated knowledge path (`knowledge/` not `lloyd/knowledge/`) |
| `~/obsidian/agents/lloyd/skills/research-agent/SKILL.md` | Updated paths and tool names |
| `~/obsidian/agents/lloyd/skills/autolink/autolink.py` | Updated `REGISTRY_EXCLUDE`, `SKIP_HUB_DIRS`, `EXISTING_HUBS` for new structure |

---

## Maintenance Notes

- **Adding new documents**: always include `segment:` in frontmatter. Use the write path that matches the segment.
- **Adding new agents**: define their read/write scope in their AGENTS.md. Default write target should be `agents/{agentId}/`.
- **New top-level vault directories**: should not be created outside the 5 canonical segments. If a genuinely new category emerges, update `VALID_SEGMENTS` in `tool_services.py`, add to `PROFILE_SCOPE` entries, and re-batch-tag affected docs.
- **qmd reindex**: after any bulk file moves, run `qmd update` from inside the lloyd container. The index stores file paths; stale paths return no results.
- **Scope cache isolation**: the prefill cache key includes the scope string, so changing a profile's scope mapping invalidates only that profile's cache entries.
