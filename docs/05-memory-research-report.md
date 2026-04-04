# Next-Gen Memory Subsystem: Deep Research Report

**Date:** 2026-04-02
**Scope:** Evaluate relationship-centric memory architectures, LightRAG, and identify improvements for Hermes

---

## Table of Contents

1. [Current Hermes Memory Architecture](#1-current-hermes-memory-architecture)
2. [LightRAG Deep Dive](#2-lightrag-deep-dive)
3. [State of the Art: Relationship-Centric Memory](#3-state-of-the-art-relationship-centric-memory)
4. [Gap Analysis: Hermes vs. Best Practices](#4-gap-analysis-hermes-vs-best-practices)
5. [Recommended Improvements](#5-recommended-improvements)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [References](#7-references)

---

## 1. Current Hermes Memory Architecture

### 1.1 Three-Layer Design

Hermes uses a pragmatic three-layer memory system:

| Layer | Storage | Purpose | Characteristics |
|-------|---------|---------|-----------------|
| **Core Memory** | `~/.hermes/memories/MEMORY.md` + `USER.md` | Session-level agent context | 2200/1375 char limits, flushed during compression, `§`-delimited entries |
| **Structured Fact Store** | `~/obsidian/memory/_pipeline/facts/{entity}/{category}.md` | Discrete facts per entity | YAML frontmatter, confidence scores, 150+ entities indexed |
| **Vault Search** | QMD daemon (BM25 + vector) across 7 Obsidian segments | Full knowledge base retrieval | Intent classification routes to segments, Qwen3.5-35B consolidation model |

### 1.2 Relationship Tracking (Current State)

**relations-index.json** (436KB): Tracks document-to-document relationships via wiki-link style edges with type, reason, and score. Visualized in the MC Web entity graph.

**facts-index.json** (171KB): Indexes all fact files by entity name, category, and count. Powers entity lookups and the knowledge graph UI.

**Entity Graph Construction** (`mc_server.py:1938-1990`):
- Document nodes + entity nodes
- Wiki-link edges (from relations-index) + has-facts edges (entity → fact file)
- Bidirectional deduplication, 60s cache TTL
- **Limitation:** Flat graph, no community detection, no typed relationship semantics beyond wiki-link/has-facts

### 1.3 Autonomy & Pipeline Systems

**Autonomy Plugin** (582 lines): Task management with YAML-frontmatter markdown files in `~/obsidian/autonomy/`. Supports scheduling, dependencies, priority, and run history. `autonomy_run_task` is still a stub — dispatch logic not yet migrated.

**Pipeline Plugin** (678 lines): Multi-stage delegation with worker isolation, signal detection (`SIGNAL:STAGE_COMPLETE|TASK_COMPLETE|BLOCKED`), and skill auto-injection. Completes linked autonomy tasks on finish.

### 1.4 Key Limitations of Current System

| Area | Current State | Gap |
|------|--------------|-----|
| **Entity Resolution** | Name-exact matching only | No fuzzy/semantic dedup — "Alan" vs "alan" vs "Alan Timm" are separate |
| **Relationship Types** | Only `wiki-link` and `has-facts` | No typed relationships (causes, depends-on, contradicts, supersedes) |
| **Temporal Tracking** | `last_updated` timestamp on facts | No valid_at/invalid_at — can't track when facts *were* true |
| **Cross-Document Linking** | Wiki-link based (relations-index.json) | No entity-mediated cross-document discovery |
| **Contradiction Handling** | `fact_check` uses opposing term pairs | Brittle pattern matching — misses semantic contradictions |
| **Multi-Hop Retrieval** | Not supported | QMD returns flat results — no graph traversal at query time |
| **Graph Storage** | JSON files | No graph query language, no traversal algorithms |
| **Retrieval Fusion** | Intent classification → segment routing | No RRF, no re-ranking, no hybrid graph+vector fusion |

---

## 2. LightRAG Deep Dive

### 2.1 Architecture

LightRAG (EMNLP 2025, HKU Data Science Lab) replaces flat chunk retrieval with a lightweight knowledge graph built during ingestion.

**Indexing Pipeline:**
1. Token-based chunking (1200 tokens, 100 overlap)
2. LLM-powered entity/relationship extraction as structured tuples
3. Deduplication and description merging via LLM summarization
4. Embedding generation for entities, relationships, and chunks
5. Graph construction: entities → nodes, relationships → edges, with chunk-entity bidirectional mapping

**Query Pipeline:**
1. Extract local keywords (specific entities) and global keywords (themes) from query
2. Match against entity and relationship vector DBs
3. One-hop graph traversal for context enrichment
4. Assemble context from entities + relationships + source chunks
5. LLM generates answer from combined context

### 2.2 Four Storage Layers

| Layer | Purpose | Default | Production Options |
|-------|---------|---------|-------------------|
| KV Storage | Documents/chunks | JSON files | PostgreSQL, Redis, MongoDB |
| Vector Storage | Embeddings | NanoVectorDB | PGVector, Milvus, Chroma, Qdrant |
| Graph Storage | Entity graph | NetworkX (in-memory) | Neo4j, PostgreSQL+AGE |
| Doc Status | Processing state | JSON files | PostgreSQL, MongoDB |

### 2.3 Retrieval Modes

- **Naive**: Pure vector similarity on chunks (baseline)
- **Local**: Entity vector DB → matched entities + their chunks (detail-oriented)
- **Global**: Relationship vector DB → relationships + their chunks (theme-oriented)
- **Hybrid**: Local + Global combined (recommended default)
- **Mix**: Naive + Local + Global with optional cross-encoder reranking
- **Bypass**: LLM-only, no retrieval

The dual-level (local + global) retrieval is LightRAG's core innovation. Local handles "Who diagnosed patient X?" while global handles "What themes dominate cardiology research?"

### 2.4 Entity Extraction Details

Extraction produces structured tuples:
```
entity | entity_name | entity_type | entity_description
relation | source_entity | target_entity | keywords | description
```

Built-in types: Person, Organization, Location, Event, Concept, Method, Content, Data, Artifact, NaturalObject. N-ary relationships are decomposed into binary pairs. Relationships are undirected by default.

**Gleaning**: Optional iterative re-extraction (re-prompts LLM to catch missed entities). Improves recall at cost of additional LLM calls.

### 2.5 Strengths

| Dimension | LightRAG | Microsoft GraphRAG |
|-----------|----------|-------------------|
| Indexing cost (500 pages) | ~$0.50 | $50-200 |
| Indexing time | ~3 min | ~45 min |
| Query latency | ~80ms | ~120ms |
| Tokens per query | <100 | ~610,000 |
| Incremental updates | Native (graph union) | Full rebuild |
| Quality vs. GraphRAG | 70-90% | 100% (baseline) |

### 2.6 Weaknesses Relevant to Hermes

1. **Entity resolution is name-exact only** — "Microsoft Corp" and "Microsoft" remain separate nodes. No fuzzy matching, no coreference resolution.
2. **Flat graph** — No community detection or hierarchy. Misses corpus-wide patterns that Leiden clustering captures.
3. **Requires 32B+ parameter models** — Smaller models frequently produce empty graphs. Extraction quality is highly model-dependent.
4. **Hallucinated entities** — LLM may fabricate entities/relationships not in source text.
5. **One-hop traversal limit** — May miss complex multi-hop relationships.
6. **NetworkX default doesn't scale** — In-memory graph; needs Neo4j/PostgreSQL for production.

### 2.7 What Hermes Could Adopt from LightRAG

1. **Dual-level retrieval (local + global)** — Currently Hermes routes by intent classification to vault segments. Adding entity-level and relationship-level vector search would enable both precise and thematic retrieval.
2. **Chunk-entity bidirectional mapping** — Hermes facts link to entities but not back to source documents systematically. Provenance chains would improve trust and debugging.
3. **Incremental graph updates via union** — Hermes already does incremental fact storage, but the relations-index is statically computed. Adopting LightRAG's incremental union approach for the entity graph would keep it current.
4. **Keyword-based query routing** — LightRAG's low-level/high-level keyword extraction could complement or replace Hermes' intent classification.

---

## 3. State of the Art: Relationship-Centric Memory

### 3.1 Knowledge Graph Construction

**LLM-Based vs. NLP Pipelines:**
- Few-shot GPT-4/Claude: 70-80% extraction accuracy, zero training data
- Fine-tuned Mistral-7B (domain): 91.3% F1 entities, 86.7% relationships
- Dependency-parsing classical NLP: 94% of LLM performance at fraction of cost
- **Cost threshold:** Below 1,500 docs, prompt-based is cheapest. Above 10,000, fine-tuning breaks even.

**KGGen (2025 SOTA):** Three-stage pipeline (generate → aggregate → cluster) using DSPy. Achieves 66.07% on MINE benchmark vs. 47.80% for GraphRAG, 29.84% for OpenIE. Iterative LLM-based entity clustering resolves the sparse-graph problem.

**Schema approach:** Start with predefined schema (3-7 node types, 5-15 relationship types) but support evolution. AutoSchemaKG achieves 95% alignment with human schemas across 900M+ nodes with zero manual intervention.

### 3.2 Temporal Knowledge Graphs

**Graphiti/Zep's bi-temporal model** (the current gold standard):

| Timestamp | Type | Meaning |
|-----------|------|---------|
| `created_at` | System time | When data entered the database |
| `valid_at` | Event time | When the fact became true in reality |
| `invalid_at` | Event time | When the fact stopped being true |
| `expired_at` | System time | When a contradiction invalidated it |

Facts are invalidated, not deleted, preserving full history. Enables point-in-time queries ("What did we know about X as of March 15?").

### 3.3 Entity Resolution (Three-Tier)

Production systems use cascading resolution:
1. **Exact match** — Same normalized string
2. **Fuzzy similarity** — Levenshtein distance, token overlap
3. **LLM semantic reasoning** — For ambiguous cases ("MSFT" vs "Microsoft Corporation")

Without aggressive entity resolution, graphs become sparse and unusable. This is the single biggest quality lever.

### 3.4 Hybrid Retrieval with RRF Fusion

The consensus architecture combines three retrieval pillars fused via **Reciprocal Rank Fusion**:

```
Vector similarity (semantic) ──┐
BM25 / keyword (exact terms) ──┼── RRF Fusion ── Cross-encoder reranking ── Top-K
Graph traversal (structural) ──┘
```

- Vector: Best for semantic matching when terminology is unknown (~10-50ms)
- BM25: Best for proper nouns, codes, IDs, exact terms
- Graph: Best for multi-hop, connected context that is semantically distant but structurally related (~50-150ms)

**HybridRAG results** (NVIDIA/BlackRock): 85%+ accuracy vs. 70% for vector-only. GraphRAG achieves 72-83% win rates on comprehensiveness, 75-82% on diversity.

### 3.5 Multi-Hop Retrieval

**StepChain GraphRAG (2025 SOTA):** Decomposes complex queries into sub-questions, uses BFS-based graph traversal along relevant edges, assembles explicit evidence chains. State-of-the-art on MuSiQue, 2WikiMultiHopQA, HotpotQA.

### 3.6 Agent Memory Frameworks (Benchmarked)

| Framework | Architecture | LongMemEval Accuracy | Key Differentiator |
|-----------|-------------|---------------------|-------------------|
| **Hindsight** | Multi-strategy hybrid (4 parallel) | 91.4% | Highest retrieval accuracy, "Reflect" LLM synthesis |
| **SuperMemory** | Memory graph + integrated RAG | 81.6% | Automatic contradiction resolution |
| **Zep/Graphiti** | Temporal knowledge graph | 63.8% | Best temporal reasoning, <200ms latency |
| **Mem0** | Vector + Graph + KV | 49.0% | Largest community, framework-agnostic |
| **Letta** | OS-inspired tiered | N/A | Agent self-manages memory, unlimited context |
| **Cognee** | KG from unstructured data | N/A | Multimodal, fully local, 30+ connectors |

**Hindsight** is notable: four parallel retrieval strategies (semantic, BM25, entity graph, temporal filtering) with cross-encoder reranking, all in embedded PostgreSQL + pgvector. Zero external dependencies.

### 3.7 Memory Consolidation

Drawing from neuroscience, modern systems treat forgetting as a feature:
- **Decay functions**: Exponential decline based on recency and access frequency
- **Relevance-based retention**: Prioritize memories aligned with agent goals
- **Three-level consolidation**: Local (within conversation) → cluster (across sessions) → global (system-wide synthesis)
- **Cognee's "Dreamify"**: Background process that adjusts and consolidates post-deployment

### 3.8 Fact Provenance and Confidence

Every fact should carry:
- **Confidence score** (0-1): Belief in truth, factored by source reliability
- **Provenance chain**: Episode → extraction → entity/edge, traceable to source
- **Temporal window**: When the fact was valid, not just when it was recorded
- **RDF-star style metadata**: Quoted triples enabling source attribution, confidence, and audit trails directly on individual facts (saves ~75% of triples vs legacy reification)

---

## 4. Gap Analysis: Hermes vs. Best Practices

### Critical Gaps (High Impact, Feasible)

| # | Gap | Current | Target | Impact |
|---|-----|---------|--------|--------|
| 1 | **Entity resolution** | Name-exact only | Three-tier (exact → fuzzy → LLM semantic) | Eliminates duplicate entities, makes graph usable |
| 2 | **Typed relationships** | wiki-link / has-facts only | Semantic types (causes, supersedes, depends-on, contradicts, related-to) | Enables reasoning about *how* things connect |
| 3 | **Temporal fact tracking** | `last_updated` only | Bi-temporal (valid_at, invalid_at, created_at, expired_at) | Tracks fact evolution, enables point-in-time queries |
| 4 | **Retrieval fusion** | Intent classification → segment routing | RRF fusion of vector + BM25 + graph traversal | 15-30% retrieval quality improvement based on benchmarks |
| 5 | **Contradiction resolution** | Opposing term pairs (brittle) | Temporal overlap + confidence-weighted LLM comparison | Catches semantic contradictions, not just lexical |

### Important Gaps (Medium Impact)

| # | Gap | Current | Target |
|---|-----|---------|--------|
| 6 | **Cross-document entity linking** | Wiki-links in relations-index | Entity-mediated discovery (shared entities → related documents) |
| 7 | **Fact provenance** | Entity + category only | Full chain: source document → chunk → extraction → fact |
| 8 | **Graph query capability** | JSON files, no traversal | Graph DB or in-memory graph with traversal algorithms |
| 9 | **Multi-hop retrieval** | Not supported | At minimum 2-hop traversal from seed entities |
| 10 | **Re-ranking** | None (QMD returns flat results) | Cross-encoder reranking of top-K candidates |

### Nice-to-Have Gaps (Lower Priority)

| # | Gap | Current | Target |
|---|-----|---------|--------|
| 11 | Community detection | None | Leiden algorithm for automatic topic clustering |
| 12 | Memory consolidation | Manual flush | Automated background consolidation with decay |
| 13 | Schema evolution | Implicit categories | Predefined schema with auto-discovery for new types |
| 14 | Dual-level keyword extraction | Intent classification | LightRAG-style local/global keyword split |

---

## 5. Recommended Improvements

### 5.1 Phase 1: Foundation (Entity Resolution + Temporal Facts)

**Goal:** Make the existing graph usable and facts trustworthy.

#### A. Three-Tier Entity Resolution

Add to the `next-gen-memory` plugin's fact ingestion pipeline:

```
New fact arrives for entity "X"
  ├── Exact match in facts-index.json? → Merge
  ├── Fuzzy match (Levenshtein ≤ 2, or token overlap > 0.7)? → Merge with canonical name
  └── No match → LLM prompt: "Are these the same entity? {X} vs {candidates}" → Merge or create new
```

Store an **alias table** mapping variant names to canonical entity IDs. This is the single highest-impact improvement — without it, the graph remains fragmented.

#### B. Bi-Temporal Fact Model

Extend fact YAML frontmatter:

```yaml
facts:
  - fact: "Alan prefers dark mode"
    confidence: 0.9
    id: "fact_abc1"
    created_at: "2026-04-02T10:00:00"   # when we learned it
    valid_at: "2026-04-01T00:00:00"      # when it became true (or null if always)
    invalid_at: null                      # when it stopped being true (null if still valid)
    expired_at: null                      # when superseded by contradiction
    source_doc: "memory/2026-04-01.md"   # provenance
```

When `fact_add` detects a contradiction (via improved `fact_check`), set `invalid_at` on the old fact and `expired_at` to current time, rather than deleting it.

### 5.2 Phase 2: Graph Upgrade (Typed Relationships + Traversal)

**Goal:** Enable meaningful graph queries and relationship reasoning.

#### A. Typed Relationship Schema

Define a core relationship type vocabulary:

```
related-to      — General semantic connection (default)
causes          — Causal relationship (A causes B)
depends-on      — Dependency (A requires B)
supersedes      — A replaces/updates B
contradicts     — A conflicts with B
part-of         — Hierarchical containment
mentions        — Document references entity
co-occurs       — Entities appear together frequently
temporal-next   — Sequential ordering
```

Extend `relations-index.json` entries with typed relationships and confidence scores.

#### B. Graph Storage Upgrade

Two viable paths:

**Option A — NetworkX in-memory (quick win):**
- Already a Python library, zero infrastructure
- Load relations-index.json + facts-index.json into a NetworkX DiGraph at server startup
- Enables BFS/DFS traversal, shortest path, connected components
- Cache with TTL, rebuild on index change
- **Limitation:** Doesn't persist, scales to ~100K nodes

**Option B — Kuzu embedded graph DB (production path):**
- Used by Cognee and supported by Graphiti
- Embedded (no separate server), Cypher-compatible queries
- Persistent on disk, handles millions of nodes
- Python bindings available

**Recommendation:** Start with NetworkX (Phase 2), migrate to Kuzu (Phase 3) if scale demands it.

#### C. Graph-Aware Retrieval

Add a `vault_graph_search` tool or enhance `vault_search`:

```python
def graph_search(query_entities: list[str], hops: int = 2) -> dict:
    """
    1. Resolve query entities via alias table
    2. Find matching nodes in graph
    3. BFS traversal up to N hops
    4. Collect connected entities + their fact profiles + source docs
    5. Return structured subgraph context
    """
```

### 5.3 Phase 3: Hybrid Retrieval Fusion

**Goal:** Combine all retrieval methods for best-in-class results.

#### A. RRF Fusion Pipeline

```
User query
  ├── QMD vector search (existing) ──────────────────┐
  ├── QMD BM25 keyword search (existing) ────────────┼── RRF Fusion ── Top-K
  └── Graph traversal from extracted entities (new) ──┘
```

**Reciprocal Rank Fusion** formula:
```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))  for each retrieval method i
```
where k = 60 (standard constant). Simple, parameter-free, proven effective.

#### B. Optional Cross-Encoder Reranking

After RRF fusion produces top-20 candidates, optionally pass through a cross-encoder model for precision reranking to top-5. Could use the existing local Qwen model or a dedicated reranker (bge-reranker-v2, etc.).

#### C. LightRAG-Style Keyword Extraction

Replace or augment intent classification with dual-keyword extraction:

```python
def extract_query_keywords(query: str) -> dict:
    """LLM extracts:
    - low_level_keywords: specific entities, proper nouns (→ entity search)
    - high_level_keywords: themes, concepts (→ relationship search)
    """
```

This routes the query to both local (entity) and global (relationship) retrieval simultaneously, rather than choosing one path via intent classification.

### 5.4 Phase 4: Autonomy Integration

**Goal:** Automated memory maintenance via autonomy tasks.

#### A. Entity Resolution Task (daily)

An autonomy task that:
1. Scans facts-index.json for potential duplicate entities
2. Runs fuzzy matching across all entity names
3. Presents candidates for merge (or auto-merges high-confidence matches)
4. Updates alias table and relations-index

#### B. Fact Consolidation Task (weekly)

An autonomy task that:
1. Groups related facts across entities
2. Identifies redundant or outdated facts (expired_at set, low confidence, old)
3. Summarizes fact clusters into consolidated entries
4. Prunes superseded facts (mark as archived, don't delete)

#### C. Relationship Discovery Task (daily)

An autonomy task that:
1. Scans recently added/modified documents
2. Extracts entities and relationships (LightRAG-style or simpler LLM extraction)
3. Updates the entity graph with new typed relationships
4. Rebuilds community clusters if using community detection

#### D. Graph Quality Audit Task (weekly)

An autonomy task that:
1. Identifies orphan nodes (entities with no relationships)
2. Finds disconnected subgraphs
3. Checks for stale relationships (source documents deleted/changed)
4. Reports graph health metrics to the autonomy dashboard

---

## 6. Implementation Roadmap

### Phase 1: Foundation (1-2 weeks)
- [ ] Implement three-tier entity resolution in `next-gen-memory` plugin
- [ ] Add alias table (`entity-aliases.json`) alongside facts-index
- [ ] Extend fact YAML schema with bi-temporal fields
- [ ] Update `fact_add` to populate temporal fields
- [ ] Improve `fact_check` to use temporal overlap + LLM comparison

### Phase 2: Graph Upgrade (2-3 weeks)
- [ ] Define typed relationship schema (8-10 core types)
- [ ] Integrate NetworkX DiGraph into `mc_server.py` entity graph
- [ ] Add graph traversal endpoint (`/api/entity-graph/traverse`)
- [ ] Create `vault_graph_search` tool in next-gen-memory plugin
- [ ] Update MC Web entity graph UI to display relationship types and support traversal

### Phase 3: Retrieval Fusion (2-3 weeks)
- [ ] Implement RRF fusion in `vault_search`
- [ ] Add dual-keyword extraction (local/global) to query pipeline
- [ ] Optional: integrate cross-encoder reranking
- [ ] Benchmark against current intent-classification approach

### Phase 4: Autonomy Tasks (1-2 weeks)
- [ ] Implement `autonomy_run_task` dispatch (currently a stub)
- [ ] Create entity resolution autonomy task
- [ ] Create fact consolidation autonomy task
- [ ] Create relationship discovery autonomy task
- [ ] Create graph quality audit autonomy task

### Total Estimated Scope: 6-10 weeks progressive rollout

Each phase delivers standalone value. Phase 1 alone would significantly improve fact quality. Phases can be parallelized (e.g., Phase 2 and Phase 4 are largely independent).

---

## 7. References

### LightRAG
- [LightRAG Paper (EMNLP 2025)](https://arxiv.org/html/2410.05779v1)
- [LightRAG Repository](https://github.com/HKUDS/LightRAG)
- [LightRAG DeepWiki Architecture](https://deepwiki.com/HKUDS/LightRAG)
- [Neo4j: Under the Covers with LightRAG Extraction](https://neo4j.com/blog/developer/under-the-covers-with-lightrag-extraction/)
- [GraphRAG vs LightRAG Comparative Analysis](https://www.maargasystems.com/2025/05/12/understanding-graphrag-vs-lightrag-a-comparative-analysis-for-enhanced-knowledge-retrieval/)

### Knowledge Graph Construction
- [KGGen: Extracting Knowledge Graphs from Plain Text](https://arxiv.org/html/2502.09956v1)
- [LLM-empowered KG Construction Survey](https://arxiv.org/abs/2510.20345)
- [From LLMs to Knowledge Graphs: Production-Ready Systems in 2025](https://medium.com/@claudiubranzan/from-llms-to-knowledge-graphs-building-production-ready-graph-systems-in-2025-2b4aff1ec99a)
- [Practical GraphRAG at Scale](https://arxiv.org/abs/2507.03226)

### Temporal Knowledge Graphs
- [Graphiti/Zep Temporal KG Architecture](https://arxiv.org/abs/2501.13956)
- [Graphiti DeepWiki](https://deepwiki.com/getzep/graphiti)
- [Temporal Agents with KGs (OpenAI Cookbook)](https://cookbook.openai.com/examples/partners/temporal_agents_with_knowledge_graphs/temporal_agents)

### Hybrid Retrieval
- [HybridRAG: Integrating KGs and Vector RAG](https://arxiv.org/abs/2408.04948)
- [StepChain GraphRAG](https://arxiv.org/html/2510.02827v1)
- [Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking](https://dev.to/lucash_ribeiro_dev/graph-augmented-hybrid-retrieval-and-multi-stage-re-ranking-a-framework-for-high-fidelity-chunk-50ca)
- [NexusRAG (hybrid vector+KG+reranking)](https://github.com/LeDat98/NexusRAG)

### Agent Memory Frameworks
- [MemGPT / Letta Docs](https://docs.letta.com/concepts/memgpt/)
- [Memory in the Age of AI Agents Survey](https://arxiv.org/abs/2512.13564)
- [Reflexion: Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Top AI Agent Memory Frameworks 2026](https://dev.to/nebulagg/top-6-ai-agent-memory-frameworks-for-devs-2026-1fef)
- [Benchmarking AI Agent Memory (Letta)](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- [AI Agent Memory Systems Compared (Dev Genius)](https://blog.devgenius.io/ai-agent-memory-systems-in-2026-mem0-zep-hindsight-memvid-and-everything-in-between-compared-96e35b818da8)

### Graph Databases & Storage
- [Apache AGE vs Neo4j](https://dev.to/pawnsapprentice/apache-age-vs-neo4j-battle-of-the-graph-databases-2m4)
- [Real-Time KGs with CocoIndex + Kuzu](https://dev.to/cocoindex/build-real-time-knowledge-graphs-from-documents-using-cocoindex-kuzu-with-llms-live-updates-n1b)
- [Chunking Strategies for KG RAG](https://medium.com/@visrow/knowledge-graph-chunking-for-rag-tbox-abox-and-advanced-strategies-b922ea286a6c)

### Production Guides
- [Graph RAG in 2026: What Works in Production](https://www.paperclipped.de/en/blog/graph-rag-production/)
- [Awesome-GraphRAG Resource List](https://github.com/DEEP-PolyU/Awesome-GraphRAG)
- [Enterprise RAG Guide 2025](https://datanucleus.dev/rag-and-agentic-ai/what-is-rag-enterprise-guide-2025)
