---
segment: architecture
tags: [architecture,lloyd]
relations:
  related-to:
  - architecture/autonomy-system.md
  - architecture/evaluation-engine.md
  - projects/lloyd/plans/discord-voice-integration.md
  - projects/lloyd/plans/voice-async-protocol.md
  - projects/lloyd/plans/document-relations-retrieval.md
  - architecture/agents.md
  - architecture/backlog.md
  - architecture/index.md
  - architecture/infrastructure.md
  - architecture/morning-briefing.md
  - architecture/nightly-reflection.md
  - architecture/nightly-skills-management.md
  - architecture/nightly-vault-maintenance.md
  - architecture/skills.md
  - architecture/tools.md
  - architecture/voice.md
  - architecture/memory.md
tags: [architecture]
summary: 'Three-layer memory architecture: source documents (Layer 1),extracted fact files (Layer 2),derived indexes and profiles (Layer 3). Includes next-gen Facts/Relations/Profiles system,subliminal context injection,13-collection QMD search,and 3-tier capture/nightly/real-time pipeline.'
type: reference
updated: 2026-04-15

---

# Memory System Architecture

Lloyd uses a **three-layer memory architecture** that combines automated capture,structured fact extraction,semantic search,and nightly analysis. The system writes to and reads from the [[index|Obsidian Vault]] and is built on QMD v2.0.1 for hybrid BM25+vector search.

## Three-Layer Model

```
Layer 1: Source Documents
  ┌──────────────────────────────────────────────────────────────────────┐
  │  daily notes  │  knowledge/  │  projects/  │  sessions/  │  people/ │
  │  (human-written,captured by periodic cron or manually authored)    │
  └──────────────────────────────────────────────────────────────────────┘
           │ nightly extraction + add_fact calls
           ▼
Layer 2: Fact Files + Entity Relationship Graph
  ┌──────────────────────────────────────────────────────────────────────┐
  │  _pipeline/vault-derived/facts/{entity}/{entity}-{category}.md       │
  │  (23,571 entity dirs, 61,394 fact files — YAML frontmatter)          │
  │  _pipeline/vault-derived/kg.sqlite  (edges, aliases, entities,       │
  │   facts_idx — behind app.kg_store; see [[knowledge-graph]])          │
  └──────────────────────────────────────────────────────────────────────┘
           │ rebuild_index
           ▼
Layer 3: Derived Indexes
  ┌──────────────────────────────────────────────────────────────────────┐
  │  ~/lloyd/_pipeline/relations-index.json  (12,879 relationships)      │
  │  people/{name}/profile.md               (30 synthesized profiles)   │
  └──────────────────────────────────────────────────────────────────────┘
```

Layer 3 is a **rebuildable cache** — it can be regenerated at any time from Layer 1+2 via `rebuild_index`. This design ensures no permanent data loss if derived indexes become stale.

### Knowledge Wiki Layer

`~/obsidian/knowledge/` is Lloyd's **persistent,compounding wiki** — 172 LLM-maintained synthesis pages across 26 domain directories. Follows the [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). All pages carry `source_type: primary|synthesized|captured` for provenance tracking.

**Three core operations** (detailed in [[knowledge-graph]]):

1. **Ingest** — source-driven: user drops a URL/path → immutable source summary → update 3-10 existing wiki pages → extract facts → log. Skill: `~/obsidian/skills/ingest/SKILL.md`.

2. **Query → Write-back** — the compounding mechanism: every research answer filed back as a knowledge page. Enforced in `quick-research` (when synthesis > 200 words or ≥3 facts),`medium-research` (always),`deep-research` (always). Nightly Phase 6 catches unfiled research.

3. **Lint** — automated health: Knowledge Health Report (daily 03:30),Unfiled Research Scan (Phase 6),Knowledge Gap Harvest (Phase 7,dispatches research for recurring gaps),Entity Synthesis Generation (Phase 8,auto-generates overview pages from source docs for entities with >5 facts).

**Five ingestion pipelines** feed the wiki: ingest skill (real-time,source-driven),research skills (question-driven),nightly entity synthesis (automated),daily note extraction,and project status updates.

**Schema:** `knowledge/KNOWLEDGE_SCHEMA.md` — 8 page types,required frontmatter,naming conventions.
**Operations log:** `knowledge/_log.md` — append-only,grep-parseable.

## Vault Structure

The vault has 16 top-level directories. Most are QMD-indexed; some are utility-only.

| Directory | Purpose | QMD Collection | File Count |
|-----------|---------|----------------|------------|
| `memory/` | Agent experience — daily notes,reflections,pipeline data | `memory` (ignores `_pipeline/**`) | 1,231 |
| `knowledge/` | Compounding wiki — research,synthesis,docs,how-tos (schema: `KNOWLEDGE_SCHEMA.md`) | `knowledge` | 172 |
| `projects/` | Project notes — plans,architecture,backlogs | `projects` | 185 |
| `agents/` | Agent identity — SOUL,TOOLS,MEMORY,HEARTBEAT | `agents` | 151 |
| `personal/` | Alan's personal notes | `personal` | 30 |
| `work/` | Alan's work