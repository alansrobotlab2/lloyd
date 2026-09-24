---
relations:
  related-to:
  - architecture/memory.md
  - architecture/autonomy.md
segment: architecture
summary: 'Knowledge graph: markdown fact files as the fact layer, one SQLite store
  (app.kg_store) for edges, aliases, the entity registry and the fact index, the
  extraction chain that writes them, the declared-identity gate in front of it,
  and the rules that came out of the 2026-08-22 wipe and the 2026-09-03 merge
  incident.'
tags:
- architecture
- knowledge
- memory
type: reference
updated: 2026-09-18
status: implemented
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
├── facts/<Entity>/<Entity>-<category>.md   # THE FACT LAYER (71,868 files, 2026-09-18)
│                                            # YAML frontmatter + generated body
└── kg.sqlite                                # THE STORE (192 MB, WAL)
    ├── entities   name, kind, definition, source_hash, timestamps
    ├── aliases    surface -> canonical, with kind and origin
    ├── edges      typed, with provenance, evidence, supersede chain
    ├── facts_idx  derived from the markdown; `facts_idx.reindex()` rebuilds it
    └── meta       schema_version, last_reindex, migrated_at
```

There is no `kg` command. The index is rebuilt by calling
`store().facts_idx.reindex(root=…)` — what `kg_rebuild.py`, `repair_fact_ids.py`,
`entity-resolution-sweep.py` and `kg_migrate_to_sqlite.py` each do at the end of
a pass. Two comments in `agent_mcp/facts.py` still tell a reader to run
`kg reindex`; they mean that call.

Derived and rebuildable at any time: `facts_idx`, `_pipeline/relations-index.json`
(document co-occurrence — written by `scripts/memory/rebuild_index.py` and by nothing
else, #1148), `_pipeline/relations-index-typed.json` (typed frontmatter relations —
written and read by `scripts/memory/next-gen-memory/relations_index.py`),
`_pipeline/content-hashes.json`, entity overview files, QMD vector collections.
The two relation indexes are separate files on purpose: until 2026-09-21 both
writers wrote the same path with different schemas inside scheduled task #24, and
the second overwrote the first every cycle.

Not derivable, and therefore backed up: the edge graph, the alias table, merge
history and hand-review state. Fact *content* can be re-extracted from the vault;
the fact that two entities are related, and who decided so, cannot.

### Scale (table: 2026-09-04 → 2026-09-11)

| Metric | Count | 2026-09-08 | 2026-09-04 |
|---|---|---|---|
| Entity directories | 25,155 | 23,812 | 23,571 |
| Fact files | 68,834 | 62,410 | 61,394 |
| Indexed facts | 284,146 across 25,159 registered entities | 222,532 / 23,816 | 205,573 / 23,567 |
| Edges | 59,061 total, 46,114 active | 18,753 / 14,049 | 6,703 / 4,029 |
| Nodes with ≥1 edge | 12,825 — 51% coverage | 6,564 — 28% | 3,304 — 14% |
| Aliases | 3,919 (2,541 case, 1,268 punct, 61 suffix, 49 semantic) | 3,874 (… 1 semantic) | unchanged |
| Entity kinds | 16,732 system, 4,204 entity, 1,245 unclassified, 1,057 concept, 709 skill, 659 doc, 448 task, 59 person, 18 project, 17 subsystem, 14 pipeline, 1 model | 16,678 / 4,171 / 951 / 703 / 631 / 425 / 256 / 1 | |
| Fact files with duplicate IDs | 0 | 0 | 15,825 |
| Provenance coverage | 27.9% (79,293) | 7.9% (17,687) | 0.37% (755) |

Read the columns as a rate, not a snapshot: the pipeline runs 6x/day and these
numbers move between two queries in the same session.

Re-measured 2026-09-18T04:30Z, same queries: **26,110 entity dirs · 71,868 fact
files · 314,613 indexed rows over 27,528 indexed entities · 78,606 edges with
53,002 active · 14,086 nodes with ≥1 edge (54% of the registry) · 4,028 aliases
(2,541 case, 1,378 punct, 61 suffix, 48 semantic) · kinds: 16,953 system, 4,258
entity, 1,331 concept, 1,271 unclassified, 744 doc, 711 skill, 518 task, 164
person, 87 project, 61 subsystem, 23 pipeline, 1 model · 0 fact files with a
duplicate ID inside them.** Two things in that re-measure are not growth:
`provenance_pct` is now effectively 100% (314,566 of 314,613 rows) because the
gate's own metric counts the column, while the columns it meant to protect —
`created_at` and `source_doc` — are NULL on 204,733 and 204,768 rows (65%), so
two-thirds of the store still cannot be dated or attributed; and a re-run of the
junk gate's own predicate (`looks_like_junk_entity`, `source_doc` unresolved) over
the live tree names **977** entity dirs that
`looks_like_junk_entity` rejects, against a gate that demands 0 — the gate that
was never re-run after the junk was injected (2026-09-09), not a new defect.

The 09-04 column is not history for its own sake: everything between it and
09-08 was **one day of the pipeline running again**. #24 had been paused since
2026-09-04 by a `kg_rebuild.py freeze` whose rebuild never passed its gate, and
nothing un-paused it, so the graph simply stopped growing and the only record
of that was a `status:` line in a vault file. Edges nearly tripled and node
coverage doubled on the first day back. Treat a flat edge count as an outage
signal, not a quiet week. The three days to 09-11 are what that looks like
sustained: edges tripled again and node coverage passed half the registry.

Provenance is still the outstanding one, and it only improves going forward —
the 0.37% floor was every fact written before the extractor started recording
`created_at` and `source_doc`, and those cannot be dated now. What the column
measures is therefore dilution, not repair — and as of 2026-09-18 the column has
outrun its own metric: the gate's `provenance_pct` reads ~100% (314,566 of
314,613 rows carry a `provenance`) while `created_at` is NULL on 204,733 rows and
`source_doc` on 204,768, so two-thirds of the store still cannot be dated or
reverted. A one-word column is not the same fact as a date and a source. Duplicate fact IDs used to sit beside it
here; they were repaired in place on 2026-09-08 and the tree has held at zero
since, see *The rebuild*. The semantic-alias jump from 1 to 49 is the one row
that is not growth — see *The declared-identity gate*.

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
| An unreadable store is not an empty one | `StoreUnavailable` is raised, never the empty schema. Writers abort; ranking-only readers degrade. Since #1236 that covers an *absent* file too: `_open()` mkdirs and `sqlite3.connect`s, which CREATES a database whose every count answers 0, so the module-level reader `store()` calls `_require_database()` first and refuses a path holding no database, naming the resolved path. `KGStore(path)` and `configure(path)` remain the named routes that provision one |
| Nothing is silently overwritten | A merge *expires* each edge and re-adds it, returning `(old_id, new_id)` so a revert is exact. `retype` sets `superseded_edge_id` |
| Every row says where it came from | `kg_store.ORIGINS` — extractor, sweep, semantic, fact_add, fact_relate, seed, classifier, conversation, revert, migration, manual, legacy — are the *known* writers, not a constraint: the column is free-form, and `schema` (the identity gate) is a real origin that predates nothing in that tuple. `aliases.origin` and `aliases.kind` likewise |
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

s.export_json(dir)    # legacy shape; called only by the one-shot migration and
                      # the pre-rebuild freeze, into timestamped dirs under
                      # _pipeline/backups/ — never in the tree it describes (#474)
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
entity_naming.gate_entity_name          THE DECLARED-IDENTITY GATE (#537)
        │  schema / alias / typed_new / candidate / junk — see below
        ▼
fact_extractor.write_fact_file
        │  atomic_write_text inside locked_file(<file>.lock)
        ├──▶ <Entity>/<Entity>-<category>.md      the fact layer
        ├──▶ facts_idx.update_file()              the index (see below)
        └──▶ edges.add(type="mentions", …)        THE GROWTH PATH
                 one per other known entity the fact names,
                 carrying source_doc and the fact text as evidence —
                 unless the pair already holds an active typed edge,
                 which keeps its verdict (#1246; the skip count is in
                 the nightly log, and apply counts a record for such
                 a pair as `already_typed` instead of re-typing it)
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

Three other writers:

- **`fact_relate` / `fact_add`** (MCP tools) — a fact or edge stated in a chat
  turn. `provenance: STATED`, `origin: fact_relate`. Its `type` must be in
  `app.kg_store.EDGE_TYPES`, the closed edge vocabulary (#546); anything else
  is refused with the list, because a free-form type had been minting a
  count-1 type per novel word. The health report's `Edge-type cardinality:
  PASS|FAIL` line watches the result (report output only, not an alarm). A new
  entity `fact_add` names is minted through the declared-identity gate, with a
  kind (below). The `remember` and
  `forget` verbs that routed onto `fact_add` and `fact_invalidate` from
  2026-09-09 were retired on 2026-09-23; see `architecture/memory.md`.
- **`conversation_relations.py`** (#51) — co-access pairs from session
  trajectories become `co_accessed` edges, `provenance: INFERRED`, with the
  trajectory as `source_doc`. It has produced **143 active edges, 0 expired**,
  and it is writing daily now: 2 on 09-10, 31 on 09-12, 53 on 09-15, 28 on 09-16,
  15 on 09-17, 14 on 09-18. The "2 edges" figure this section carried was true on
  2026-09-11 and is seven days stale; the 08-22 precedent says an expired edge is
  worse than none, and nothing has expired one.
- **`entity_naming.register_schema_keys`** — writes the declared aliases, and it
  is not the only producer of `origin: schema` rows: `gate_entity_name` mints one
  whenever a raw name matches a declared key spelled differently from the
  canonical (entity_naming.py:871). That is why the live count was 47 against the
  41 aliases the schema file declares. It used to INSTALL only, so **a withdrawn
  declaration was permanent policy**: six `2026-09-10` rows survived the move from
  `extraction_schema.yaml` to JSON — including `Task` → `Entity Resolution Sweep`,
  which the landed schema's own `why` field says was deliberately not declared —
  and `gate_entity_name` answering `alias` before `typed_new` meant a generic word
  routed into the entity-resolution pipeline. It now reconciles as well
  (`_retract_undeclared_schema_rows`, entity_naming.py:686): a row goes when
  `_schema_key` — the same fold the gate uses, not a literal pair — does not
  resolve it to the same canonical, or when the surface is one the declaration
  path would refuse (`_brackets_balanced`, entity_naming.py:600, which is why
  `Entity Resolution Sweep (Task` goes while `Knowledge Graph (KG)` stays). Only
  `origin='schema'` with a NULL `report_path` is retractable; migration/sweep/
  test/triage rows and #475's apply-provenance rows are not its to touch, and
  entities are never unregistered here. Pinned by
  `tests/test_entity_identity_schema.py::test_a_withdrawn_declaration_is_retracted_by_the_next_register`
  and its four siblings — fold-not-literal, other-provenance, bracket-shape, and
  the extractor's own write path. So the six rows are gone the first time a
  gated write runs against the live store — the nightly extractor does it by
  itself. What is
  still a person's decision, and keeps #1234 open: whether `Knowledge Graphs`
  (22 facts) and `Periodic Memory Capture` (1 fact) merge into their old alias
  targets now that nothing routes them there, and which side is intended for
  `periodic memory capture`, where `aliases.resolve` answers `Memory Capture` and
  `aliases.all_lower()` answers `Periodic Memory Capture`.

Measured by origin on 2026-09-18, active edges: extractor 33,091, classifier
16,614, migration 2,898, sweep 248, conversation 143, `fact_relate` 8. (The
`semantic` origin carries 0 rows: #67 proposes, nothing has applied.) The extractor is the
graph, and the classifier is what makes it typed; everything else is rounding.

`seed_relationship_edges.py` still exists but is a **backfill tool for trees
extracted before the extractor emitted edges**, not part of the nightly chain.

### The declared-identity gate

Added 2026-09-09 (#537). Every name the extractor wants to file under now
passes `app/entity_naming.py::gate_entity_name` first, which returns
`(entity, verdict)` and answers in a fixed order:

| verdict | what happens |
|---|---|
| `schema` | the name is a declared key or declared alias → the canonical |
| `alias` | the store already routes it → the canonical |
| `typed_new` | unclaimed, but the model declared a type → a new typed entity |
| `candidate` | unclaimed and untyped → the sidecar, **nothing created** |
| `register` | unclaimed and untyped, `enforce=False` → the pre-#537 behaviour |
| `junk` | the existing junk predicate rejects it |

**The order is the design.** The declaration is asked *before* the store, so a
human saying two names are one thing outranks whatever a previous extraction
inferred. Nothing in the chain resembles a string metric — the gate exists
because minting on every alias miss produced sibling families (`Knowledge
Graph` plus five variants with one edge among them; `Autonomy Data Pipeline`
plus three), and a similarity rule is what would fuse the pairs that are
genuinely distinct. `tests/test_entity_identity_schema.py` pins that it never
consults one.

**`fact_add` mints through the same gate** (#758, 2026-09-24). It registered a
new name with no kind, and its reindex registered the directory a second time,
also untyped — the source of nearly every `kind IS NULL` row. A new or declared
name now goes through `gate_entity_name` with the caller's optional
`entity_type`, or `app.entity_kind.derive_kind` when none is given: a caller
naming a new thing is declaring it, so the tool gets `typed_new` where the
extractor would get `candidate`. A type the schema does not carry is refused
before anything is written, and the post-write `update_file` passes
`register_entities=False`. Attach is still exact + alias + declaration, never
fuzzy (#340). Whether an unattended caller should be refused an undeclared name
outright is Alan's call and not implemented. `tests/test_fact_add_entity_kind.py`.

`scripts/memory/entity_identity_schema.json` is the declaration: human-edited,
extractor-read-only, and validated on load — a type outside
`app.entity_kind.KINDS` (plus `pipeline`/`subsystem`, the two spellings
`entities.kind` already carries) is refused, as is one surface declared an
alias of two canonicals, because the answer would then depend on file order.
Declared aliases are written `kind='semantic'`, `origin='schema'`, which is the
whole of the scale table's 1 → 49 semantic-alias jump; `register_schema_keys`
is idempotent, so the nightly run installs them without churning `created_at`.

**A refusal is recorded, not swallowed.** `candidate` appends to
`_pipeline/vault-derived/entity-candidates.jsonl` — 15 names so far — for
weekly human review. Refusing to mint is easy and quietly stops remembering
things; the sidecar is the pressure valve that keeps the gate honest.

**Reads opt out, writes cannot.** `enforce=False` is passed only where a name
is being resolved to pull an entity's existing facts into the prompt
(`get_existing_facts`); refusing there would withhold context and degrade
extraction rather than protect the graph. Writes leave it at the default,
because a gate you have to remember to enable is a gate that gets forgotten.

### The index is not optional

The markdown is what a person reads; `facts_idx` is what the router,
`fact_get`, the Memory page and the health report actually read. Every
writer of a fact file updates it in the same breath —
`fact_add` through `facts_idx.update_file`, `fact_resolve` and
`fact_invalidate` through `facts.py:_reindex_files()`, in the same call, as
soon as the markdown write returns. `fact_invalidate` did neither until
2026-09-04 — it wrote `expired_at` to the file and stopped, and it was missing
the `locked_file` the other two had — so the file said expired while the Memory
page went on serving the fact as current, and the two disagreed until the next
full reindex. A `StoreUnavailable` here is not fatal — the markdown is written
and `facts_idx.reindex()` rebuilds the index — but a silent skip is.

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

`valid_at` is filled from the model's `event_date` when the fact dates itself,
and `created_at` stays the extraction time either way — the two answer "when
was this true" and "when did we learn it", and a fact about last April written
this morning needs both. The file's own frontmatter carries `entity`,
`category`, `last_extracted` and `last_updated` around the `facts:` list.

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
divides each neighbour's weight by `log(degree + e)`; `fact_get` caps each
category at 10 (it took the cap from `fact_profile`, retired 2026-09-23). `Lloyd` had 5,489 facts when this was written and has 6,642 on
2026-09-11 — without these it answers every question. The same threshold
refuses the pairwise contradiction scan behind `fact_resolve` and `fact_resolve_apply`;
see *Tools*.

---

## Rules that came out of the incidents

1. **A corrupt store is not an empty store.** Any reader that returns "no data"
   on a read failure will eventually let a writer persist that emptiness.
2. **Applies are attended.** #48 and #67 propose; a human runs `--apply`. Both
   incidents were unattended applies.
3. **Refuse on a degraded graph.** `--apply` and `backup-graph.sh` both compare
   active edges against `graph-baseline.json` and refuse below 50%. The backup
   half no longer fails open on its own missing input: a baseline that is
   missing, unparseable, or carries no positive `active_edges` is now a refusal
   naming the path, before the tarball is written or the window rotated, and
   every successful run prints the reference's `recorded_at` beside the counts
   (`tests/test_backup_graph.py`). `--apply` still fails open there:
   `load_baseline` returns 0 on any exception and `degraded_reason` answers
   "not degraded" for a baseline of 0 (entity-resolution-sweep.py:1162-1191),
   and the sweep then ratchets the reference up to whatever the damaged store
   reads. The baseline is also raised by the very program it guards
   (`update_baseline`, :1170-1179), so a slow leak never trips it — which is
   also what self-heals a refused backup within one nightly sweep. A backup
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
7. **A gate with no input fails loudly.** `IdentitySchemaUnavailable` is raised
   when the declared-identity schema is missing or unparseable, rather than
   read as "nothing is declared" — a gate whose input is absent reports success
   on every name that passes through it, which is rule 1 one level up. Its own
   docstring names the three times this repo has been bitten by that shape:
   `graph-baseline.json` rewriting itself, `_is_dependency_met` returning True
   for an unfindable upstream, and dream-consolidation gated on a lock file
   that never existed.

---

## The rebuild

One defect cannot be repaired in place, only re-extracted:

- **0.37% provenance coverage.** Facts written before 2026-09-04 have no
  `created_at` and no `source_doc`, so they cannot be dated, attributed or
  selectively reverted. Repairing this in place means inventing provenance for
  facts whose source is unknown, which is worse than not having it.
  Re-extraction is honest: every fact in the new tree came from a named
  document at a known time.

**Duplicate fact IDs were the second one, and they were repaired in place on
2026-09-08.** The claim that they could not be is what kept them: an ID is a
handle, so `assign_ids` refuses to renumber one that exists, and that reads
like "unrepairable" until you notice it only has to renumber the *later*
holder of a collision. The first holder keeps the ID — it is what outside
records name, and what a reader scanning the file already lands on, so
resolving the ambiguity that way agrees with every reference already made.

`scripts/memory/repair_fact_ids.py` did 130,614 IDs across 15,841 of 62,410
files, taking the live tree from 29,315 colliding `(file, fact_id)` pairs to
zero with no fact text changed. It re-reads each file inside `locked_file`
before writing, so it runs against a live system; extraction added 3,560 facts
while it worked. `vLLM/vLLM-state.md` alone held 4,098 collisions in 4,447
facts.

The extractor's restarting numbering was only the original source, fixed
2026-09-03 by `app.fact_ids`. The **merge** path kept minting new ones for
another five days: `_merge_fact_file_into` and both of `revert-suffix-merges.py`'s
merge branches concatenate two independently-numbered `facts:` lists, and the
dedup beside them is by fact *text* and cannot see an ID. They call
`dedupe_ids` now. Without that this repair would have been undone by the next
entity merge — which is exactly what happened to the rebuild tree on 2026-09-08
when a punct sweep put 356 collisions into a tree that had none.

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
| `import` | Replays the carry-over into the new tree through `fact_add`, so it gets the new ID scheme. Skips a fact already present, so the re-run its own failure message tells you to do adds nothing twice |
| `gate` | Every check, as JSON. Exit 0 only if all pass. `--skip-eval` runs the structural half and deliberately does **not** record a verdict `swap` will accept |
| `swap` | Two renames and a reindex. The old tree becomes `facts-quarantine-<ts>`, the old store `kg-quarantine-<ts>.sqlite`; re-enables writes and un-pauses what `freeze` paused |
| `rollback` | Reverses both renames, and un-pauses what `freeze` paused |
| `unfreeze` | Restores those tasks without swapping or rolling back — the way out of an abandoned rebuild |

**A freeze has to be able to end by itself.** `freeze` records what it paused
and, until 2026-09-08, nothing read that list back: `swap` printed
`Then un-pause #24, #48, #74.` — hardcoded to one run's ids — and `rollback`
said nothing at all. So the only exit ran through a line printed on the one
outcome that does not happen when a rebuild goes wrong. The 2026-09-03
rebuild's gate failed, the swap never ran, and #24 and #74 stayed paused for
four days with no reason recorded in their own task files. Fact extraction is
downstream of #24, so the whole graph stopped growing and the only thing that
said so was a `status:` line in a vault file nobody had cause to read.

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

### The 2026-09-03 run: attempted, abandoned 2026-09-08

Kept here because the machinery still works and someone will be tempted again.

It froze at 04:46Z against a baseline of MRR 0.496 / NDCG@10 0.579, extracted
the full 2,837-document corpus, and produced a tree that was cleaner than live
on every structural axis: 100% provenance against 7.9%, zero junk-named entity
directories against 977, and 20,314 active edges against 14,049. Every one of
the seven structural gate checks passed.

It never swapped, because it lost on the only measure that decides what Lloyd
finds: MRR 0.482 against the 0.496 baseline, and a fuzzy category down 0.250.
NDCG@10 was actually better (0.584 vs 0.579), and it won on multi-hop (+0.074)
and technical (+0.114). It lost on `single` and `fuzzy`.

Read that fuzzy number carefully before repeating the exercise. The category
holds **two** queries. One was unchanged; `autonomy-pipeline` went from rank 1
to rank 2 on one document. That single rank slip is the entire -0.250, and a
per-category threshold of -0.05 cannot distinguish it from a systemic
regression at n=2. The gate is right to refuse on a signal it cannot read, but
the signal is not what the number looks like.

**What actually killed it was the reasoning underneath, not the score.** Three
things justified rebuilding rather than repairing, and by 2026-09-08 two were
gone:

- *Duplicate fact IDs* were repaired in place, 130,614 of them (above).
- *Self-ingestion* — roughly half of live's facts came from the pipeline
  re-reading its own output — was fixed by the allow-list corpus in
  `pipeline_config.yaml`, which protects the live tree just as well.
- *Provenance* is the one that remains, and only a rebuild can fix it
  retroactively.

Meanwhile the live tree tripled its edges in the single day #24 spent unpaused,
and the rebuild aged: frozen at a 09-04 snapshot, its corpus coverage drifted
98.88% → 98.78% against a 98% floor purely because the vault kept growing. A
parked rebuild is a wasting asset, and it needs a fresh `export`/`import` for
every fact stated since its freeze.

So the trade was 222,532 facts of which 92% can never be dated, against 75,560
that all can — and worse retrieval today. Abandoned. The tree, its store, its
content hashes and its state file are deleted; `freeze.json`, `gate.json` and
the eval runs are kept under `_pipeline/backups/rebuild-20260903T214511Z/`
alongside `kg-before.sqlite`, a 2026-09-03 snapshot of the live store.

If provenance ever becomes the binding constraint, the tool is `kg_rebuild.py`
and this section is the record of what it costs. Do not restart one without
first deciding what a two-query category is allowed to veto.

Also outstanding, and unrelated to the rebuild: 977 junk-named entity
directories (`_pipeline/memory-graph/junk-entities-review.json`) holding 6,898
facts in the live tree. The extractor rejects these names before registration
now, so the set cannot grow; removing the existing ones moves fact files and
drops edges, which is a merge-class operation and goes through review. Still
exactly 977 when re-measured on 2026-09-11 against the same predicate the gate
uses (`looks_like_junk_entity`), across 25,155 directories — the containment
holds, and the count is a stock, not a leak.

---

## Tools

`fact_get`, `fact_add`, `fact_resolve`, `fact_resolve_apply`,
`fact_invalidate`, `fact_relate`, `fact_relationships` (`agent_mcp/facts.py`),
and `vault_recall` (`agent_mcp/vault.py`).

#376's four verbs (`remember`, `recall`, `forget`, `improve`) sat over these
from 2026-09-09 and were retired on 2026-09-23, with `fact_check` (a second
name for `fact_resolve`) and `fact_profile` (whose cap and `query` ranking
`fact_get` took). `fact_path` and `fact_neighbors` were then deleted on
2026-09-24 (#1077): no session stored on this box has ever called either, while
the writer `fact_add` has hundreds, so the server was advertising an
N-hop expansion no caller used — and its description told callers to work
around a truncation only that dead walk could reach. `fact_relationships`, the
edge reader that does carry traffic, took the confidence floor the walk used
to hold (`min_confidence`, default 0.0 = keep every edge).
`architecture/memory.md` has why, and the `improve` loop (#84, nightly, plan
mode), which runs from its script.

Three of those address a fact by `<file, id>` -- `fact_resolve`,
`fact_invalidate` and every revert report -- so the ID has to name exactly one
fact. `scripts/memory/repair_fact_ids.py` is the repair when it does not; see
**The rebuild** for what it fixed and why the merge path kept re-creating the
problem.

`fact_resolve` reports by default. It defaulted to `auto_resolve=True`, so a
call that reads like a query silently expired facts — and its contradiction
detector fires on token overlap above 0.6, which is two facts phrased
similarly, not two facts that disagree.

`fact_resolve` and `fact_resolve_apply` share that detector, and it is pairwise —
O(n²) in the entity's fact count. It is refused above `FACT_GODNODE_THRESHOLD`
inside `_detect_contradictions_sync`, so both tools inherit the refusal; the
guard used to sit on `auto_resolve` alone, which left the reporting path
running the scan. Against `Lloyd` that was 15 million comparisons: 113 seconds
through MCP, returning 32,857 "contradictions" that were almost entirely the
overlap heuristic firing on two facts phrased alike. Both tools now take
`category` to scan a slice — 113s → 4ms.

## Review log

- **2026-09-18 — #1236 fixed: an absent store now refuses instead of answering
  0.** The module-level reader `store()` calls `_require_database()`
  (`app/kg_store.py`) before it opens anything, so a path with no database
  raises `StoreUnavailable` naming the resolved path and leaves no file behind —
  where it previously mkdir'd, `sqlite3.connect`'d into existence and answered
  `facts_idx.count() == 0`, which is what a job booting from a self-mod worktree
  hit, because `_pipeline/` is gitignored. `KGStore(path)` and `configure(path)`
  still provision, so the rebuild and every test fixture keep creating; the
  rebuild's `extract` phase now creates `kg-rebuild.sqlite` by name, because the
  extractor only warns on `StoreUnavailable` and would have extracted a corpus
  with no edges. Pinned by
  `tests/test_kg_store.py::test_default_store_refuses_an_absent_database_and_creates_no_file`,
  `::test_default_store_refuses_without_making_the_missing_directory`,
  `::test_a_reader_process_refuses_an_absent_store_and_leaves_no_file` and
  `::test_the_provisioning_routes_still_create_an_absent_database`. Counts still
  have to name the path they read: the live tree holds ~314k indexed rows and a
  worktree has no store at all, so the same query is now either a real number or
  a refusal, never a zero.
- **2026-09-18 — stale.** The mechanisms all still exist and the nine rules are
  all still enforced in code, but the document had stopped describing the live
  store: re-measured the Scale section (26,110 dirs / 71,868 files / 314,613
  indexed rows / 53,002 active edges / 54% node coverage), corrected the
  provenance claim (the gate's `provenance_pct` now reads ~100% while
  `created_at`/`source_doc` are NULL on 65% of rows — the metric moved, the
  coverage did not), corrected the `conversation` edge count from 2 to 143 and
  the origin table, dropped two numbers no query supports (the 11 junk dirs and
  the 30-snapshot daily-retention claim; junk is 977 dirs under the gate's own
  predicate, and `_pipeline/vault-derived/store-backups/` is a different,
  unrotated series written by apply scripts). Three architecture findings filed
  against the tree, not the prose: **#1234** (a withdrawn schema declaration is
  never retracted — `Task` still routes to `Entity Resolution Sweep`), **#917**
  (both degraded-graph refusals fail open on a missing `graph-baseline.json`),
  **#1236** (an absent store is auto-created empty, so `count()` answers 0 rather
  than raising — the rule-7 invariant held only for a file that exists and will
  not open; closed by the entry above, same day); the duplicate-fact-stock
  finding merged into the open **#1144**.
