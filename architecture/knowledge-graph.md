---
relations:
  related-to:
  - architecture/memory.md
  - architecture/autonomy-system.md
segment: architecture
summary: 'Knowledge graph: markdown fact files as the fact layer, one SQLite store
  (app.kg_store) for edges, aliases, the entity registry and the fact index, the
  extraction chain that writes them, and the rules that came out of the 2026-08-22
  wipe and the 2026-09-03 merge incident.'
tags:
- architecture
- knowledge
- memory
type: reference
updated: 2026-09-04
---

# Knowledge Graph

Atomic facts with provenance and temporal grounding, a typed entity edge graph,
and the retrieval built on top. Facts stay human-readable markdown, browsable
in Obsidian. Everything structural lives in one SQLite file.

**Two principles, and they are not the same one:**

1. *The markdown fact files are the fact layer.* They can be read, edited and
   diffed by a person, and the store's index is derived from them.
2. *The store is the only write path for structure.* Edges, aliases and the
   entity registry are not derivable from the markdown, and nothing outside
   `app.kg_store` opens the database.

---

## Layout

```
_pipeline/vault-derived/
├── facts/<Entity>/<Entity>-<category>.md   # THE FACT LAYER (61,392 files)
│                                            # YAML frontmatter + generated body
└── kg.sqlite                                # THE STORE (76 MB, WAL)
    ├── entities   name, kind, definition, source_hash, timestamps
    ├── aliases    surface -> canonical, with kind and origin
    ├── edges      typed, with provenance, evidence, supersede chain
    ├── facts_idx  derived from the markdown; `kg reindex` rebuilds it
    └── meta       schema_version, last_reindex
```

Derived and rebuildable at any time: `facts_idx`, `_pipeline/relations-index.json`
(document co-occurrence), `_pipeline/content-hashes.json`, entity overview files,
QMD vector collections.

Not derivable, and therefore backed up: the edge graph, the alias table, merge
history and hand-review state. Fact *content* can be re-extracted from the vault;
the fact that two entities are related, and who decided so, cannot.

### Scale (2026-09-04)

| Metric | Count |
|---|---|
| Entity directories | 23,494 (975 of them junk-named; see below) |
| Fact files | 61,392 |
| Indexed facts | 205,573 |
| Edges | 6,703 total, 4,029 active |
| Nodes with ≥1 edge | 3,304 — 14% coverage |
| Aliases | 3,874 (2,541 case, 1,271 punct, 61 suffix, 1 semantic) |
| Entity kinds | 16,678 system, 4,171 unclassified, 951 concept, 703 skill, 631 doc, 425 task, 1 person |

Two of these numbers are the work still outstanding: **provenance coverage is
0.37%** and **15,825 fact files carry duplicate fact IDs**. Both are artefacts
of the pre-2026-09-04 extractor and both are fixed by re-extraction, not by
repair — see *The rebuild* below.

---

## The store

`app/kg_store.py`. One SQLite file in WAL mode, one module, one lock.

Before 2026-09, edges lived in `_relationships.json` and aliases in
`entity-aliases.json`, and six programs across three processes rewrote them
whole with no lock between them. Two incidents came directly out of that shape:

- **2026-08-22** — `nightly_extraction.clean_facts_directory` deleted the fact
  tree and took `_relationships.json` (12,131 edges) and the memory-graph
  working directory with it. There was no backup of any kind.
- **2026-09-03** — a sweep `--apply` ran against a 2-edge graph. Every entity
  looked disconnected, so every suffix pair passed the "a variant has zero
  degree" shortcut, and 151 distinct entities were merged: `Intel Pipeline
  System` into `Intel`, `Triage Agent` into `TRIAGE`.

What the store changes:

| Property | How |
|---|---|
| A write is all-or-nothing | `store.transaction()` — `BEGIN IMMEDIATE`, nestable. A sweep's alias writes and every edge rewrite commit together |
| Two processes cannot lose a write | WAL + `busy_timeout=30000`; tested with two concurrent writer processes and a `kill -9` mid-write |
| An unreadable store is not an empty one | `StoreUnavailable` is raised, never the empty schema. Writers abort; ranking-only readers degrade |
| Nothing is silently overwritten | A merge *expires* each edge and re-adds it, returning `(old_id, new_id)` so a revert is exact. `retype` sets `superseded_edge_id` |
| Every row says where it came from | `edges.origin` ∈ extractor, sweep, classifier, conversation, fact_relate, seed, migration; `aliases.origin` and `aliases.kind` likewise |
| Caches cannot go stale | Adjacency, degree and the alias map memoise on `PRAGMA data_version`, which moves when *any* process commits |
| A backup is a valid database | `sqlite3.Connection.backup()`, not `cp` — a plain copy of a WAL file taken mid-commit is not restorable |

### The API

Nothing outside this module opens the database.

```python
from app.kg_store import store
s = store()

s.edges.add(edge, origin=...)          # dedupes on the active (src,tgt,type) index
s.edges.expire(id, reason)             # never DELETE; history is the audit trail
s.edges.retype(old_id, new, origin=…)  # expire + add + supersede, one transaction
s.edges.rewrite_endpoint(old, new, …)  # merge helper -> [(old_id, new_id)]
s.edges.revert_rewrites(pairs, …)      # the exact inverse
s.edges.adjacency() / .degree()        # cached on data_version

s.aliases.resolve(name) / .set(surface, canonical, kind=…, origin=…)
s.entities.register(name) / .lookup(name) / .kinds()
s.facts_idx.for_entity(name, …) / .reindex(paths=None)

s.export_json(dir)    # legacy shape, for backups and external readers
s.backup(path)        # consistent under writers
```

---

## The write chain

```
vault documents                    pipeline_config.yaml sources.paths
  knowledge/ projects/ people/       -- an ALLOW-LIST. 2,829 documents.
  personal/ work/ memory/            Edit the config, not the script.
        │
        ▼
nightly_extraction.py  (#24, 6x/day)
        │  content-hash gate; a FAILED extraction is not hashed, so it retries
        ▼
fact_extractor.write_fact_file
        │  atomic_write_text inside locked_file(<file>.lock)
        ├──▶ <Entity>/<Entity>-<category>.md      the fact layer
        ├──▶ facts_idx.update_file()              the index
        └──▶ edges.add(type="mentions", …)        THE GROWTH PATH
                 one per other known entity the fact names,
                 carrying source_doc and the fact text as evidence
        │
        ▼
classify-v4-batch.py + apply-classifications-v4.py  (#74, daily)
        │  mentions -> uses / part_of / depends_on / … via edges.retype
        ▼
entity-resolution-sweep.py  (#48, daily, DRY RUN ONLY)
        │  clusters near-duplicates, runs the semantic gate, prints a plan
        ▼
   a human reads the plan and runs --apply
```

Two other writers:

- **`fact_relate` / `fact_add`** (MCP tools) — a fact or edge stated in a chat
  turn. `provenance: STATED`, `origin: fact_relate`.
- **`conversation_relations.py`** (#51) — co-access pairs from session
  trajectories become `co_accessed` edges, `provenance: INFERRED`, with the
  trajectory as `source_doc`.

`seed_relationship_edges.py` still exists but is a **backfill tool for trees
extracted before the extractor emitted edges**, not part of the nightly chain.

---

## What a fact carries

```yaml
- id: stat-014                       # <prefix>-NNN, continues from the file's max
  fact: Lloyd serves models through vLLM on port 8096
  category: state                    # one of 13 in CATEGORY_VOCAB
  confidence: 0.9
  provenance: EXTRACTED              # STATED | EXTRACTED | INFERRED | AMBIGUOUS
  created_at: 2026-09-04T03:12:00Z   # when we learned it
  valid_at: null                     # when it became true, if the fact says
  source_doc: knowledge/lloyd.md     # which document
  source_hash: 8f4343…               # the bytes that were read
  expired_at: null                   # was true, no longer is
  invalid_at: null                   # should not have been recorded
```

`expired_at` and `invalid_at` are different claims and are set by different
things. `fact_invalidate` expires; `fact_resolve --auto_resolve` invalidates.
Retrieval filters both by default; `as_of` reconstructs a past state.

---

## Retrieval

`vault_recall` runs a document search and a fact lookup in parallel, then
optionally expands through the graph.

| Knob | Default | Why |
|---|---|---|
| `graph_rerank` | **False** | Measured 2026-09-04: off scores MRR 0.500 / NDCG@10 0.601; on scores 0.386–0.419 at every alpha tried, and is slower. Default-on since 2026-05-12 on a measurement never repeated after the graph was rebuilt |
| `rerank_alpha` | 0.3 | Only consulted when rerank is explicitly on |
| `graph_top_k` / `graph_hops` | 5 / 1 | Historic |
| `demote_daily_logs` | True | Daily notes match everything |

`eval/run_eval.py` imports these constants, so a bare run measures what
production serves and each record carries `matches_production_defaults`. The
nightly eval used to run a different configuration than production and could
not have detected a regression in the real one.

**God-node handling.** An entity above `FACT_GODNODE_THRESHOLD` (50) facts
needs a query-token match before any of its facts are returned; graph expansion
divides each neighbour's weight by `log(degree + e)`; `fact_profile` caps each
category at 10. `Lloyd` has 5,489 facts — without these it answers every
question.

---

## Rules that came out of the incidents

1. **A corrupt store is not an empty store.** Any reader that returns "no data"
   on a read failure will eventually let a writer persist that emptiness.
2. **Applies are attended.** #48 and #67 propose; a human runs `--apply`. Both
   incidents were unattended applies.
3. **Refuse on a degraded graph.** `--apply` and `backup-graph.sh` both compare
   active edges against `graph-baseline.json` and refuse below 50%. A backup
   taken after a wipe is worse than no backup: it rotates the last good
   snapshot out of the window.
4. **Expire, never delete.** The pre-merge graph must stay readable, and a
   revert needs the id trail.
5. **Every destructive tool stamps `_invocation.invocation_ledger()`.** On
   2026-09-03 nothing recorded what invoked the sweep — not the run records,
   not any session, not shell history.
6. **The corpus is an allow-list.** A deny-list means every new directory is
   ingested by default, which is how half the fact tree became re-extracted
   pipeline exhaust.

---

## The rebuild

Two defects cannot be repaired in place, only re-extracted:

- **0.37% provenance coverage.** Facts written before 2026-09-04 have no
  `created_at` and no `source_doc`, so they cannot be dated, attributed or
  selectively reverted.
- **15,825 fact files with duplicate fact IDs.** The extractor restarted its
  numbering each run.

`scripts/memory/kg_rebuild.py` extracts the corpus into
`_pipeline/vault-derived/facts-rebuild/` (via `LLOYD_FACTS_ROOT`), carries over
what re-extraction cannot reproduce — STATED facts, judged aliases, semantic
verdicts, `facts/Experiments/**` — and refuses to swap unless the gate passes:
contamination 0, no junk-named directories, 100% provenance, every fact ID
unique within its file, node coverage ≥ 30%, and eval MRR/NDCG at or above the
pre-rebuild numbers with no category regressing more than 0.05.

Also outstanding: 977 junk-named entity directories
(`_pipeline/memory-graph/junk-entities-review.json`) holding 6,898 facts. The
extractor rejects these names before registration now, so the set cannot grow;
removing the existing ones moves fact files and drops edges, which is a
merge-class operation and goes through review. The rebuild produces a tree
without them.

---

## Tools

`fact_get`, `fact_add`, `fact_profile`, `fact_check`, `fact_resolve`,
`fact_invalidate`, `fact_relate`, `fact_relationships`, `fact_path`,
`fact_neighbors` (`agent_mcp/facts.py`), and `vault_recall`
(`agent_mcp/vault.py`).

`fact_resolve` reports by default. It defaulted to `auto_resolve=True`, so a
call that reads like a query silently expired facts — and its contradiction
detector fires on token overlap above 0.6, which is two facts phrased
similarly, not two facts that disagree.
