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
├── facts/<Entity>/<Entity>-<category>.md   # THE FACT LAYER (61,394 files)
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
| Entity directories | 23,571 (977 of them junk-named; see below) |
| Fact files | 61,394 |
| Indexed facts | 205,573 across 23,567 registered entities |
| Edges | 6,703 total, 4,029 active |
| Nodes with ≥1 edge | 3,304 — 14% coverage |
| Aliases | 3,874 (2,541 case, 1,271 punct, 61 suffix, 1 semantic) |
| Entity kinds | 16,678 system, 4,171 unclassified, 951 concept, 703 skill, 631 doc, 425 task, 1 person |

Two of these numbers are the work still outstanding: **provenance coverage is
0.37%** (755 of 205,573 facts) and **15,825 fact files carry duplicate fact
IDs**. Both are artefacts of the pre-2026-09-04 extractor and both are fixed by
re-extraction, not by repair — see *The rebuild* below, which is mid-flight.

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
  knowledge/ projects/ people/       -- an ALLOW-LIST. 2,837 documents.
  personal/ work/ memory/            Edit the config, not the script.
        │
        ▼
nightly_extraction.py  (#24, 6x/day)
        │  content-hash gate; a FAILED extraction is not hashed, so it retries
        ▼
fact_extractor.write_fact_file
        │  atomic_write_text inside locked_file(<file>.lock)
        ├──▶ <Entity>/<Entity>-<category>.md      the fact layer
        ├──▶ facts_idx.update_file()              the index (see below)
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

**The index is not optional.** The markdown is what a person reads; `facts_idx`
is what the router, `fact_profile`, the Memory page and the health report
actually read. Every writer of a fact file updates it in the same breath —
`fact_add` through `facts_idx.update_file`, `fact_resolve` and
`fact_invalidate` through `facts.py:_reindex_files()`, in the same call, as
soon as the markdown write returns. `fact_invalidate` did neither until
2026-09-04 — it wrote `expired_at` to the file and stopped, and it was missing
the `locked_file` the other two had — so the file said expired while the Memory
page went on serving the fact as current, and the two disagreed until the next
full reindex. A `StoreUnavailable` here is not
fatal — the markdown is written and `kg reindex` rebuilds the index — but a
silent skip is.

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
question. The same threshold refuses the pairwise contradiction scan behind
`fact_check` and `fact_resolve`; see *Tools*.

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
  numbering each run, so anything that addresses a fact by ID acts on whichever
  copy it finds first.

Repairing either in place means inventing provenance for facts whose source is
unknown, which is worse than not having it. Re-extraction is honest: every fact
in the new tree came from a named document at a known time.

`scripts/memory/kg_rebuild.py` builds a second tree beside the live one and
swaps it in only if a gate passes. The extraction writes to `facts-rebuild/`
and `kg-rebuild.sqlite` through `LLOYD_FACTS_ROOT` / `LLOYD_KG_DB`, so the live
system keeps serving throughout. Every step appends to `rebuild-state.json`;
`status` prints where a run got to.

| Step | What it does |
|---|---|
| `freeze` | Pauses #24, #48 and #74, sets `knowledge_graph.write_enabled: false`, backs up the store with `Connection.backup()`, runs the eval and records the baseline the gate is measured against. `--keep-writes` leaves fact writes on |
| `export` | Collects what re-extraction cannot reproduce: STATED/INFERRED/AMBIGUOUS facts and anything sourced from a session, semantic/suffix/manual aliases, `facts/Experiments/**` verbatim, judge verdicts and merge history, and stated edges |
| `extract` | Runs `nightly_extraction --full` against the rebuild tree with a content-hash index of its own, or it would skip every file the live tree already extracted |
| `import` | Replays the carry-over into the new tree through `fact_add`, so it gets the new ID scheme |
| `gate` | Every check, as JSON. Exit 0 only if all pass |
| `swap` | Two renames and a reindex. The old tree becomes `facts-quarantine-<ts>`, the old store `kg-quarantine-<ts>.sqlite`; re-enables writes |
| `rollback` | Reverses both renames |

`extract` is resumable and multi-pass. A failed document is deliberately never
content-hashed, so a re-run retries exactly the failures; `--passes` (default 3)
repeats until a pass gains nothing. That matters because documents time out
individually under concurrency — 3 of the first 533 at 8 workers, all 120s LLM
timeouts on long documents — and each one is a document the 98% coverage floor
would otherwise block the swap on. Sweeping them up is mechanical, so it is not
manual.

`import` runs as a **subprocess** under the rebuild env rather than reloading
modules in place: `app.paths` reads `LLOYD_FACTS_ROOT` at import time and a
dozen modules capture its constants into their own globals, so an in-process
reload would leave some of them still pointed at the tree about to be
quarantined. A junk entity name is a legitimate refusal; any other failure
writes `dropped-facts.json` and exits 4. These are facts that came from a
conversation — losing one silently to a line in a stats dict is the failure the
step exists to prevent.

### The gate

| Check | Threshold |
|---|---|
| `provenance_pct` | 100 — every fact says where it came from and when |
| `duplicate_id_files` | 0 |
| `contamination_dirs` | 0 — a directory holding facts about another entity means a merge went wrong |
| `junk_entity_dirs` | 0 |
| `corpus_coverage_pct` | ≥ 98 of the allow-list |
| `node_coverage_pct` | ≥ 30 (the live tree is at 14%) |
| `carryover_facts` | every exported fact present in the new tree, junk-named ones excepted |
| `eval_mrr_doc` / `eval_ndcg10` | at or above the frozen baseline |
| `eval_category_regression` | no category down more than 0.05 MRR |

`corpus_coverage_pct` is the only check that measures the new tree against
something outside itself. Every other structural check is a *ratio*, and a
rebuild that stopped at 60% of the vault looks exactly as clean as one that
finished. Its denominator is read from the extractor's own `_eligible_files()`,
not a second copy of the allow-list, so the two cannot drift; its numerator is
the rebuild's hash index, which counts successes only.

`carryover_facts` is belt-and-braces: `import` already fails loudly on a dropped
fact, but a carry-over that was never run at all would otherwise sail through
every other check.

`swap` refuses a second way. The system stays usable while the rebuild runs, so
a fact stated in a chat turn tonight lands in the tree `swap` is about to
quarantine. `_facts_written_since_export` queries the live index for hand-stated
facts newer than the export timestamp and refuses if it finds any — re-run
`export` and `import`, then swap. `--force` exists and is documented as
something you should not use.

### Where the run stands (2026-09-04)

Frozen 04:46Z against a baseline of MRR 0.496 / NDCG@10 0.579. Carry-over
exported: 444 facts, 62 aliases, 46 experiment files, 7 review files, 0 stated
edges. Extraction complete at 2,837/2,837 documents; the rebuild tree holds
7,649 entity directories, 12,372 fact files, 75,081 indexed facts and 20,323
edges over 5,663 nodes — 74% node coverage against the live tree's 14%.

The gate has been run once and failed, on a snapshot that predates the finished
extraction (corpus coverage 62%) and on a carry-over not yet imported (0/444).
Provenance, duplicate IDs, contamination and junk names all passed. `import`
then `gate` are the next steps. #24, #48 and #74 stay paused; fact writes were
turned back on deliberately — the rebuild runs for hours and the system stays
usable — which is what `swap`'s second refusal exists to cover.

Also outstanding: 977 junk-named entity directories
(`_pipeline/memory-graph/junk-entities-review.json`) holding 6,898 facts in the
live tree. The extractor rejects these names before registration now, so the set
cannot grow; removing the existing ones moves fact files and drops edges, which
is a merge-class operation and goes through review. The rebuild tree has none,
which is why the gate can demand zero.

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

`fact_check` and `fact_resolve` share that detector, and it is pairwise —
O(n²) in the entity's fact count. It is refused above `FACT_GODNODE_THRESHOLD`
inside `_detect_contradictions_sync`, so both tools inherit the refusal; the
guard used to sit on `auto_resolve` alone, which left the reporting path
running the scan. Against `Lloyd` that was 15 million comparisons: 113 seconds
through MCP, returning 32,857 "contradictions" that were almost entirely the
overlap heuristic firing on two facts phrased alike. Both tools now take
`category` to scan a slice — 113s → 4ms.
