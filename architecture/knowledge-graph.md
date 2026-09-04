---
relations:
  related-to:
  - architecture/memory.md
  - architecture/next-gen-memory-subsystem/next-gen-memory-system.md
segment: architecture
summary: 'Knowledge graph architecture: three-layer design (markdown source,fact files
  + entity relationship graph,derived indexes),v4 edge classifier,temporal queries,binary-quantized
  vector search,graph-driven wiki compilation,MCP tool API.'
tags:
- architecture
- knowledge
- memory
type: reference
updated: 2026-04-22---

# Knowledge Graph

*Formerly "Next-Gen Memory Subsystem." Renamed 2026-04-22 to reflect what it actually is: a typed entity relationship graph with fact provenance,temporal grounding,and now a compiled wiki layer on top.*

Supermemory-class capabilities (atomic facts,temporal grounding,relationship tracking,hybrid search,profile building,compiled wiki views) in human-readable markdown browsable in Obsidian.

**Core principle:** *Markdown is the source of truth. Everything else is derived and rebuildable.*

---

## Three-Layer Architecture

```
Layer 1: Source Documents
  daily notes,knowledge/,projects/,sessions/,people/
  Human-written. Enhanced frontmatter with temporal data and typed relations.

Layer 2: Fact Files + Entity Relationship Graph
  ~/obsidian/facts/{entity}/{entity}-{category}.md
  Atomic facts in YAML frontmatter. Auto-generated readable body.
  Linked back to source. Browsable in Obsidian.
  ~/obsidian/facts/_relationships.json  — typed entity edges (v4 classifier output)
  ~/obsidian/facts/_aliases.json        — fuzzy entity resolution

Layer 3: Derived Indexes  [rebuildable cache]
  ~/lloyd/_pipeline/relations-index.json    — document-level graph traversal
  ~/lloyd/_pipeline/content-hashes.json     — SHA256 hashes for incremental processing
  people/{name}/profile.md                  — synthesized profiles
  ~/obsidian/compiled/                      — graph-driven wiki pages (NEW,Task #321)
  QMD collections (binary-quantized bit[768]) — vector search over the vault
```

**Key invariant:** Layer 3 can be deleted and rebuilt from Layers 1+2 at any time. All Layer 3 artifacts are *derived views* — never authoritative.

### Current scale (2026-04-22)

| Metric | Count |
|---|---|
| Entity directories | 1,392 |
| Fact files | 4,160 |
| Total edges | 4,627 (active: 2,406 / expired: 2,221) |
| Entity aliases | 23 (consolidated from 500+ via entity-resolution sweep) |
| Edge type vocabulary | 32 active types (up from 7 in v1) |

---

## Layer 1: Source Document Schema

Enhanced frontmatter added to all vault documents:

```yaml
---
title: "Document Title"
type: [notes|reference|project-notes|work-notes|talk|facts|hub]
tags: [tag1,tag2]
segment: [projects|knowledge|agents|memory]
document_date: 2026-03-20        # when this doc was written
event_dates:
  - date: 2026-03-15
    context: "Event context"     # when described events occurred
relations:
  implements: ["path/to/doc.md"]
  supersedes: ["path/to/doc.md"]
  depends-on: ["path/to/doc.md"]
  derived-from: ["path/to/doc.md"]
  related-to: ["path/to/doc.md"]
  conflicts-with: ["path/to/doc.md"]
facts_extracted: true
facts_extracted_at: 2026-03-20T18:30:00Z
status: [draft|active|archived|stale]
---
```

### Document relation types (6 core,all bidirectional)

| Type | Inverse | Meaning |
|------|---------|---------|
| `implements` | `designed-by` | Implements a plan/design |
| `supersedes` | `superseded-by` | Replaces/updates target |
| `depends-on` | `required-by` | Must read target to understand this |
| `derived-from` | `produces` | Extracted/generated from target |
| `related-to` | `related-to` | Symmetric connection (catch-all) |
| `conflicts-with` | `conflicts-with` | Disagrees with target |

---

## Layer 2: Fact File Schema

```yaml
---
type: facts
entity: Alan
category: preferences
last_updated: 2026-04-14T18:30:00Z
facts:
  - id: pref-001
    fact: "Fact statement"
    confidence: 0.95
    category: preferences
    created_at: 2026-02-28T00:00:00Z  # when fact entered system (immutable)
    valid_at: 2026-02-28              # when fact became true (nullable)
    invalid_at: null                  # when fact was wrong (set by fact_resolve)
    expired_at: null                  # when fact became outdated (set by fact_invalidate)
    provenance: STATED                # STATED | EXTRACTED | INFERRED | AMBIGUOUS
    source_doc: "path/to/source.md"
---
```

### Temporal fields

| Field | Set by | Purpose | Filtering |
|-------|--------|---------|-----------|
| `created_at` | `fact_add` on write | When fact entered system | Passive |
| `valid_at` | Caller of `fact_add` | When fact became true | Used by `as_of` queries |
| `invalid_at` | `fact_resolve` on conflict | Fact was **wrong** | Active filter (excluded by default) |
| `expired_at` | `fact_invalidate` | Fact was **true but outdated** | Active filter (excluded by default) |

**Distinction:** `invalid_at` = fact was wrong (contradiction resolved). `expired_at` = fact was true but no longer current (manually retired).

### Provenance labels

| Label | Meaning | Typical source |
|-------|---------|----------------|
| `STATED` | Directly stated by user or agent | Inline `fact_add` calls during conversation |
| `EXTRACTED` | LLM-extracted from documents | Nightly extraction pipeline (122B model) |
| `INFERRED` | Derived from other facts | Low-confidence extraction (confidence < 0.7) |
| `AMBIGUOUS` | Uncertain classification | Conflicting sources,needs review |

### Fact ID scheme

`{prefix}-{sequence}` per fact file (e.g. `pref-001`,`fact-002`). IDs reset per file; citation layer disambiguates by prefixing `<file-stem>#<id>` (see "Graph-driven wiki compilation" below).

---

## Entity Relationship Graph

Typed edges between entities stored in `~/obsidian/facts/_relationships.json`:

```json
{
  "edges": [
    {"source": "Lloyd","target": "OpenClaw","type": "uses",
     "confidence": 0.95,"provenance": "STATED",
     "created_at": "ISO","expired_at": null,"source_doc": null}
  ],
  "schema_version": 1
}
```

### v4 edge classifier (2026-04-21)

Full-graph classification ran 2026-04-21: **723 edge changes (28% of active edges)**,95% spot-check accuracy,96% direction-check trustworthiness. Structural guardrails:

1. **Entity-type priors** — classifier uses entity category hints (system vs. person vs. project) to bias edge-type selection.
2. **Direction verification** — for asymmetric types (`uses`,`depends_on`,`created_by`),the classifier validates direction against context rather than trusting edge-level ordering.
3. **Hallucination gate** — edges with confidence below a context-dependent floor are softened to `related_to` rather than committed to a strong type.

**Apply rules** (from `apply_v4_fullgraph.py`):
- Safe downgrades (new type ∈ {`mentions`,`related_to`}) at confidence ≥ 0.45.
- Non-safe changes require `adj="none"` and confidence ≥ 0.60.
- `downgraded_unclear` with non-safe type is always skipped.

**Type distribution shift (v1 → v4):**

| Type | v1 count | v4 count |
|---|---|---|
| `uses` | 836 | 496 |
| `part_of` | 448 | 227 |
| `related_to` | 187 | 498 |
| `mentions` | 760 | 990 |

Strong-typed edge share dropped ~60% → ~40% — the graph is now more honest about what it knows vs. what it only loosely associates.

### Active edge type vocabulary (2026-04-22)

| Type | Count | Notes |
|---|---|---|
| `mentions` | 990 | Weak co-reference |
| `related_to` | 498 | Symmetric,catch-all |
| `uses` | 496 | A uses B (asymmetric) |
| `part_of` | 227 | Composition (asymmetric) |
| `discusses` | 74 | Document-level mention of entity |
| `competes_with` | 26 | |
| `depends_on` | 26 | Runtime dependency (asymmetric) |
| `implements` | 22 | |
| `created_by` | 10 | |
| `supersedes` | 10 | |
| (25 other low-frequency types) | 1–4 each | Long-tail from v4 classifier |

### Graph tools

- `fact_path` — BFS shortest path between two entities (default max_hops=3)
- `fact_neighbors` — N-hop neighborhood subgraph with min_confidence filter
- `fact_relationships` — all edges for an entity (inbound,outbound,or both)
- `vault_recall(expand_graph=true)` — 1-hop graph expansion on search results

### Entity resolution (rewritten 2026-09-03)

Two layers, in this order:

**At extraction.** `app.entity_naming.known_entities_in_text` finds the known
canonical names present in a chunk (proper-noun shapes only) and the extractor
lists them in its prompt with an instruction to reuse them verbatim. Until
2026-09-03 the extractor never saw a known name — every caller passed an empty
context — so it coined `Intel Pipeline System` beside an existing
`Intel Pipeline`; 303 of 442 near-duplicate clusters were born a day or more
after their canonical. Live test after the change: 18 of 18 entities reused a
known name.

**The sweep** (`scripts/memory/entity-resolution-sweep.py`, task #48 daily
dry-run). Name shape *clusters* candidates; only CASE and PUNCT variants merge
on shape. SUFFIX clusters (`X` vs `X System/Agent/SDK/Service/Pipeline/App`)
pass through `entity_semantic_gate.py`: both entities' definitions go to every
judge model (primary + secondary) and the pair merges only on unanimous SAME.
A missing definition is refused, not judged. Verdicts are cached by definition
hash. `--apply` refuses below 50% of the largest active-edge count ever
recorded, writes aliases before moving files, aliases every approved merge,
retags moved facts to the canonical with `merged_from`, and stamps an
invocation ledger into its report. `revert-suffix-merges.py` inverts any
apply report by tier.

Why: on 2026-09-03 an unattributed apply against the empty post-incident
graph merged 151 suffix pairs on shape alone — a news scanner into Intel, a
fact store into a robotics tokenizer, a robot's training pipeline into the
robot. All were reverted; the rules above are what stop the next one.

**Hygiene** is measured, not assumed: `scripts/memory/kg_hygiene.py` and the
daily knowledge-health report track cross-entity contamination (must be 0),
near-duplicate clusters, and duplicate regrowth dated by fact `created_at`.

State at 2026-09-03 close: 3,865 active edges, node coverage 13.7%, alias
coverage 84.4%, contamination 0, 301 near-duplicate clusters awaiting
definitions or review.

---

## Graph-Driven Wiki Compilation (Task #321,2026-04-22)

New Layer 3 artifact: **compiled wiki pages** at `~/obsidian/compiled/`.

The graph is authoritative; wiki pages are a *derived narrative view* of the graph. Never edited by hand. Regenerated from scratch on demand.

### Pipeline (`~/lloyd/_pipeline/memory-graph/`)

| Script | Purpose |
|---|---|
| `compile_wiki.py` | Take N source entities → read their facts + internal edges → synthesize a narrative markdown page via LLM. Writes to `~/obsidian/compiled/<slug>.md`. |
| `verify_faithfulness.py` | Post-compile mechanical verifier. Validates every `[<stem>#<fact-id>]` and `[<src>→<tgt>: <type>,conf=<X.XX>]` citation against the input graph context. |
| `sweep_triangles.py` | Batch runner across N triangles. Emits JSON summary with per-triangle counts + problem lists. |

### Citation format (strict; verifier-enforced)

**Facts** — `[<file-stem>#fact-<id>]` verbatim from the graph context. Multi-fact shorthand (same stem) allowed: `[Lloyd-fact-store#fact-007,#fact-008]`. Range syntax (`fact-007-013`) forbidden.

**Edges** — canonical form only: `[<source>→<target>: <type>,conf=<X.XX>]`. Example: `[Lloyd→OpenClaw: uses,conf=0.95]`. Legacy form `[type,conf=0.95]` deprecated — it's ambiguous about direction and enables misattribution hallucinations.

### Faithfulness verifier

Catches four hallucination classes:

1. **Fabricated edges** — full `(src,tgt,type,conf)` tuple absent from the graph. Includes misattribution (right type+conf but wrong source/target).
2. **Unknown fact stems** — cited file stem doesn't exist for any source entity.
3. **Missing fact IDs** — stem exists but `fact-<id>` is not among its facts.
4. **Malformed shorthand** — multi-stem shorthand in a single bracket group.

Exit code 1 on any problem. `--annotate` appends the report to the compiled file.

### Autonomy integration

**Task #69** — "Nightly Wiki Compile Sweep." Weekly cadence,local model (`primary`),`memory` agent. Skill at `~/obsidian/skills/wiki-compile-sweep/SKILL.md`. Runs `sweep_triangles.py --summary ~/lloyd/autonomy-runs/wiki-sweep-latest.json` across an 8-triangle slate.

### Current performance

- **8/8 triangles compile** successfully on local Qwen
- **~0.22% hallucination rate** (1 problem / 447 citations on last live run)
- **Legacy-form citation rate**: near zero after prompt tightening (occasional Qwen backslide,~1% of runs)
- **Misattribution hallucinations**: eliminated by canonical edge format requirement

### Principles

- The graph is authoritative; compiled pages are derived,never authoritative.
- Surface contradictions explicitly — if two facts disagree,*show the disagreement*,don't smooth it away.
- Freshness stamped per entity in frontmatter. Stale data gets a warning.
- Single writer (the script),atomic rename. No concurrent writes.
- Graph-hygiene issues (wrong-direction edges,circular `part_of`,self-edges) surface as "confident but wrong" content — that's the compiler working correctly. Fix the source graph,not the page. See Task #329.

---

## Vector Search (Binary Quantized)

Per-segment QMD collections power fast semantic recall across the vault.

### Binary quantization (Task #304,shipped 2026-04-21)

- Backend: `bit[768]` — 768-dim binary embeddings (1 bit per dim,96 bytes per vector)
- **p95 latency: 260ms → 126ms** on qmd vector search
- **File-recall@10: 39% → 57%**
- Reranker removed by default (500ms overhead,0% top-1 changes observed)

### End-to-end impact

| Stage | Before | After |
|---|---|---|
| Vault query (end-to-end) | ~2,500ms | ~145ms |
| qmd recall @10 | 39% | 57% |

### Query-time fixes

- **Fact retrieval** (Task #322,2026-04-22): god-node cap (10-fact limit) + query-aware ranking via keyword-overlap scoring. **fact_hit_rate 0.067 → 0.219 (+227%)**.
- **Doc retrieval** (Task #325,2026-04-22): hyphen-as-negation HTTP 500 bug in qmd. `_qmd_sanitize` now replaces hyphens with spaces. **recall@10 0.367 → 0.467** (+27%); 1-hop 0.571 → 0.857 (+50%).

---

## ClaudeSDKClient Integration (Task #307,shipped 2026-04-21)

Lloyd's backend migrated from per-turn `claude` CLI subprocess spawn to a persistent `ClaudeSDKClient` with idle TTL.

| Phase | Change |
|---|---|
| Phase 1 | Persistent client shared across turns |
| Phase 2 | 10-minute idle TTL; client retired after idleness |
| Phase 3 | Direct interrupt support (no subprocess kill) |

**Results (burst test):**
- 1 client connection served 2 turns
- `cache_create` tokens: **3,125 → 730**
- Turn duration: **9.3s → 2.2s**

This lowers the marginal cost of every tool call that touches the LLM — including memory MCP calls (`fact_add`,`vault_recall`,`fact_path`).

---

## Knowledge Wiki (Compounding Layer)

`~/obsidian/knowledge/` is Lloyd's persistent,compounding wiki — 172+ LLM-maintained synthesis pages across 26 domain directories. Follows the [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): raw sources are immutable,the wiki is LLM-maintained,and every research answer gets filed back as a page.

**Not to be confused with `~/obsidian/compiled/`:**
- `knowledge/` — human-curated and research-driven. Write-back on ingest/research.
- `compiled/` — graph-driven. Regenerated from Layer 2 on demand. Never hand-edited.

**Schema:** `knowledge/KNOWLEDGE_SCHEMA.md` defines 8 page types (`entity-overview`,`concept-synthesis`,`how-to`,`comparison`,`source-summary`,`quick-research`,`medium-research`,`deep-research`),required frontmatter,naming conventions,and cross-reference rules.

**Operations log:** `knowledge/_log.md` — append-only,grep-parseable.

### Three core operations

| Operation | Trigger | Effect |
|---|---|---|
| **Ingest** | User shares URL/doc | Source summary → update related wiki pages → fact_add |
| **Query → Write-back** | Any research during conversation | Knowledge page → fact_add → log entry |
| **Lint** | Scheduled (daily/nightly) | Knowledge Health Report,Unfiled Research Scan,Gap Harvest |

**Safety net:** Nightly reflection Phase 6 (Unfiled Research Scan) backfills research answers that weren't filed. Phase 7 (Knowledge Gap Harvest) dispatches medium-research for recurring gaps (max 3/night). Phase 8 (Entity Synthesis Generation) auto-generates entity-overview pages for rich-fact orphans (max 3/night).

### Knowledge Health Report

Nightly autonomy task (#59,03:30 daily):

| Metric | What it detects |
|---|---|
| **God entities** | >20 facts — may need splitting |
| **Thin entities** | <2 facts and 0 relationships — knowledge gaps |
| **Orphan entities** | Facts but zero relationships — need wiring |
| **Stale facts** | >60 days old with no `valid_at` update |
| **Relationship distribution** | Edge type counts across the graph |

Output: `~/lloyd/_pipeline/reflection/knowledge-health-YYYY-MM-DD.md`. Feeds the morning briefing.

---

## MCP Tool API

All tools live in `~/lloyd/agent_mcp/memory.py`,exposed by the memory MCP server via stdio JSON-RPC.

### Fact tools

| Tool | Description |
|------|-------------|
| `fact_get` | Retrieve facts for entity/category. Supports `as_of` (temporal query) and `include_expired` (historical view). |
| `fact_add` | Add atomic fact with confidence,provenance,valid_at,source_doc. Auto-generates IDs. |
| `fact_profile` | Synthesized entity profile — all facts grouped by category,top 3 per category. |
| `fact_check` | Detect contradictions via opposing term pairs and token overlap (>60%). |
| `fact_resolve` | Auto-resolve contradictions by keeping higher-confidence fact. Sets `invalid_at` on loser. |
| `fact_invalidate` | Expire facts that are no longer current. Sets `expired_at` on matching facts by entity/category/substring. |

### Relationship tools

| Tool | Description |
|------|-------------|
| `fact_relate` | Add typed edge between entities. Deduplicates against existing active edges. |
| `fact_relationships` | Get all edges for an entity (inbound,outbound,or both). Filter by type. |
| `fact_path` | BFS shortest path between two entities. Configurable max_hops (default 3). |
| `fact_neighbors` | N-hop neighborhood subgraph. Returns nodes + edges with min_confidence filter. |
| `fact_resolve` (entity) | Resolve a name/alias to a canonical entity. |

### Vault tools

| Tool | Description |
|------|-------------|
| `vault_get` | Read file from vault by path. |
| `vault_write` | Write file to vault. Audit-logged. |
| `vault_search` | Hybrid BM25+vector search across vault segments. Optional LLM consolidation. |
| `vault_recall` | Unified retrieval — vault search + entity facts in parallel. `expand_graph=true` adds 1-hop neighbor facts. |
| `vault_overview` | High-level summary of vault state (counts,recent changes). |

### Agent access control

| Agent | Access |
|-------|--------|
| `lloyd`,`memory` | Full read + write |
| `researcher`,`orchestrator` | Read-only or limited write |
| Subagents | Read-only |

---

## Implementation

**MCP Server:** `~/lloyd/agent_mcp/memory.py` — all fact,vault,and memory tools in one module. Registered in `~/lloyd/config.yaml` under `mcp_servers.memory`.

**Pipeline scripts:** `~/lloyd/_pipeline/memory-graph/`

| File | Purpose |
|------|---------|
| `classify-relationships-v4.py` | v4 edge classifier (entity-type priors,direction verification,hallucination gate) |
| `apply_v4_fullgraph.py` | Full-graph apply with safe-downgrade rules and backup discipline |
| `compile_wiki.py` | Graph-driven wiki compilation (v0.5) |
| `verify_faithfulness.py` | Post-compile faithfulness verifier |
| `sweep_triangles.py` | Batch triangle compile + verify driver |
| `entity-resolution-sweep.py` | Tiered entity resolution / merge candidate generation |
| `merge_alias_duplicates.py` | Alias consolidation |

**Extraction scripts:** `~/lloyd/scripts/memory/`

| File | Purpose |
|------|---------|
| `classify-relationships.py` | Legacy v1 classifier (retained for comparison) |
| `classify-relationships-v4.py` | Current classifier |
| `content_hasher.py` | SHA256 change detection for incremental processing |
| `extract-session-log.py` | Session trajectory extraction |
| `knowledge-health-report.py` | Nightly Knowledge Health Report generator |

**Key data files:**

| Path | Contents |
|------|----------|
| `~/obsidian/facts/{entity}/{entity}-{category}.md` | Fact files (1,392 entities,4,160 files) |
| `~/obsidian/facts/_relationships.json` | Entity relationship graph (4,627 total,2,406 active edges) |
| `~/obsidian/facts/_aliases.json` | Fuzzy entity resolution (23 mappings post-consolidation) |
| `~/obsidian/compiled/` | Graph-driven compiled wiki pages (Task #321) |
| `~/lloyd/_pipeline/relations-index.json` | Document-level relations graph |
| `~/lloyd/_pipeline/content-hashes.json` | SHA256 hashes for incremental processing |
| `~/lloyd/_pipeline/reflection/knowledge-health-YYYY-MM-DD.md` | Nightly health report |
| `~/lloyd/autonomy-runs/wiki-sweep-latest.json` | Latest weekly compile-sweep summary |

---

## Recent changes (week of 2026-04-15 → 2026-04-22)

| Date | Task | Change |
|---|---|---|
| 2026-04-21 | #307 | ClaudeSDKClient migration: persistent client,idle TTL,direct interrupt. cache_create 3125→730,turn duration 9.3s→2.2s. |
| 2026-04-21 | #304 | Binary-quantized vector search (bit[768]). p95 260ms→126ms,recall@10 39%→57%. End-to-end vault queries ~2500ms→~145ms. |
| 2026-04-21 | — | v4 classifier full-graph apply: 723 edge changes (28%),95% spot-check accuracy. Type distribution shift: strong-typed edges 62%→40%. |
| 2026-04-21 | — | Semantic entity resolution merged 81 entities at conf ≥ 0.90. Tier 1-3 entity count 679→653. Task normalization produced 45 canonical `Task #N` dirs. |
| 2026-04-22 | #322 | Fact retrieval fix: god-node cap + query-aware ranking. fact_hit_rate 0.067→0.219. |
| 2026-04-22 | #325 | Doc retrieval fix: qmd hyphen-as-negation bug. recall@10 0.367→0.467. |
| 2026-04-22 | #321 | Graph-driven wiki compilation shipped. Autonomy task #69 "Nightly Wiki Compile Sweep" running weekly. 0.22% hallucination rate on live run. |
| 2026-04-22 | #329 | Graph-hygiene cleanup filed — wrong-direction edges,circular `part_of`,self-edges surfaced by wiki compiler. |

---

## Related docs

- [[memory]] — Full memory system architecture (QMD,subliminal injection,recall pipeline)
- [[nightly-reflection]] — Nightly reflection and signal processing
- [[agents]] — Agent system (memory agent details)

