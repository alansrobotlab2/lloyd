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
updated: 2026-10-18
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
├── facts/<Entity>/<Entity>-<category>.md   # THE FACT LAYER (41,557 files, 2026-10-18)
│                                            # YAML frontmatter + generated body
└── kg.sqlite                                # THE STORE (149 MB, WAL)
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

**Both of those blocks measure a tree that no longer exists.** The live store
was wiped on 2026-09-22 and re-extracted whole by a completed rebuild that
swapped in on 2026-09-23 (*The rebuild*); the 977 junk dirs, the 4,028
mechanical aliases and the undated two-thirds went out with the old tree.
Re-measured 2026-10-18 on the current one: **15,164 entity dirs · 41,557 fact
files · 161,168 indexed rows · 85,416 edges with 50,558 active · 15,200
registered entities · 239 aliases · `created_at` and `source_doc` NULL on 0
rows.** The provenance gap this section called outstanding is closed — that is
exactly what the rebuild bought. What the swap did not re-grow is the
mechanical alias table (the sweep rebuilds it tier by tier) and the
conversation-origin edges (#51 below).

The 09-04 column is not history for its own sake: everything between it and
09-08 was **one day of the pipeline running again**. #24 had been paused since
2026-09-04 by a `kg_rebuild.py freeze` whose rebuild never passed its gate, and
nothing un-paused it, so the graph simply stopped growing and the only record
of that was a `status:` line in a vault file. Edges nearly tripled and node
coverage doubled on the first day back. Treat a flat edge count as an outage
signal, not a quiet week. The three days to 09-11 are what that looks like
sustained: edges tripled again and node coverage passed half the registry.

Provenance *was* the outstanding one until the 2026-09-23 swap: the rebuilt
tree carries `created_at` and `source_doc` on every indexed row (0 NULL on
2026-10-18), and the 0.37% floor — every fact written before the extractor
started recording those columns, which could never be dated after the fact —
went out with the deleted tree, so the rebuild was the repair, not more
dilution. Until then the column had outrun its own metric: as of 2026-09-18
`provenance_pct` read ~100% (314,566 of 314,613 rows carry a `provenance`)
while `created_at` was NULL on 204,733 rows and `source_doc` on 204,768, so
two-thirds of that store could not be dated or reverted. A one-word column is
not the same fact as a date and a source. Duplicate fact IDs used to sit beside it
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
s.facts_idx.for_session(sid) / .count_by_session(sid)   # who wrote it (#1709)

s.export_json(dir)    # legacy shape; called only by the one-shot migration and
                      # the pre-rebuild freeze, into timestamped dirs under
                      # _pipeline/backups/ — never in the tree it describes (#474)
s.backup(path)        # consistent under writers
```

---

## The write chain

```
vault documents                    pipeline_config.yaml sources.paths
  knowledge/ projects/ people/       -- an ALLOW-LIST. 3,106 docs (2026-09).
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
entity-resolution-sweep.py  (#48, daily, applies CASE,PUNCT itself)
        │  clusters near-duplicates, runs the semantic gate, prints a plan, then
        │  ends its scheduled run with `--apply --tiers CASE,PUNCT` unattended,
        │  on Alan's ruling on #990 (2026-09-16)
        ▼
   the two mechanical shape tiers land nightly; a SUFFIX merge waits for a human
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
  trajectories land as edges **typed by the Stage-2 classifier**, with
  `provenance: INFERRED` and the trajectory as `source_doc`:
  `land_approved_edges()` in `scripts/memory/conversation_relations.py` writes
  `type = p.get("type") or p.get("relation_type") or "co_accessed"`, so
  `co_accessed` is only the fallback for a pair nothing classified (#420,
  2026-09-12) — measured 2026-10-26 through `app.kg_store`, **0** of the 562
  `conversation`-origin edges in the store are `co_accessed`. On the
  pre-rebuild tree it had produced **143
  active edges, 0 expired** and was writing daily (2 on 09-10, 31 on 09-12, 53
  on 09-15, 28 on 09-16, 15 on 09-17, 14 on 09-18). Those edges did not survive
  the 2026-09-23 swap: the carry-over set's `CARRY_EDGE_ORIGINS` is
  `fact_relate`/`manual`, and #1161 measured `origin='conversation'` at 0 rows
  on the new store. The linker writes into the new one daily, and nothing has
  expired an edge.
- **`entity_naming.register_schema_keys`** — writes the declared aliases, and it
  is not the only producer of `origin: schema` rows: `gate_entity_name` mints one
  whenever a raw name matches a declared key spelled differently from the
  canonical (`gate_entity_name` in `app/entity_naming.py`). That is why the live count was 47 against the
  41 aliases the schema file declares. It used to INSTALL only, so **a withdrawn
  declaration was permanent policy**: six `2026-09-10` rows survived the move from
  `extraction_schema.yaml` to JSON — including `Task` → `Entity Resolution Sweep`,
  which the landed schema's own `why` field says was deliberately not declared —
  and `gate_entity_name` answering `alias` before `typed_new` meant a generic word
  routed into the entity-resolution pipeline. It now reconciles as well
  (`_retract_undeclared_schema_rows` in `app/entity_naming.py`): a row goes when
  `_schema_key` — the same fold the gate uses, not a literal pair — does not
  resolve it to the same canonical, or when the surface is one the declaration
  path would refuse (`_brackets_balanced`, which is why
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
`semantic` origin carries 0 rows: #67 proposes, nothing has applied.) Measured
again 2026-10-18 on the rebuilt tree: classifier 34,185, extractor 15,733,
conversation 444, sweep 137, `fact_relate` 47, manual 12, `semantic` still 0 —
the classifier's retypes now outnumber raw mentions, where pre-rebuild they
were half. The extractor is still where edges are born, and the classifier is
what makes the graph typed.

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
`_pipeline/vault-derived/entity-candidates.jsonl` — 432 names on 2026-10-18
(15 when this line was dated 2026-09-18) — for
weekly human review. Refusing to mint is easy and quietly stops remembering
things; the sidecar is the pressure valve that keeps the gate honest.

**A refused subject has no fallback entity (#1999).** The extractor used to
return `primary_entity or "general"` and the nightly fan-out repeated the
`or "general"`, so a document whose primary the gate refused filed every fact
without an entity of its own under an entity named `general`: 365 facts by
2026-10-01, none about anything called general, reachable by no query.
`general` is a category term. Now a refused primary goes to the sidecar (a
`junk` one too, which the gate itself does not record), the result's `entity`
is `""`, a fact whose own entity is `junk` is dropped in the extractor as a
`candidate` one already was, and the fan-out holds back any fact left with no
entity or with the literal `general`, counting it on the document's stdout
line and in the run summary (`N facts held back`). A fact that names no entity
still files under a primary the gate accepted. The 365 already filed were not
re-filed into an entity; `scripts/memory/quarantine_general_facts.py`
(#2303/#2443) has since moved `general` residue out of the live fact tree into
a `facts-quarantine-<ts>` directory, which the retention sweep bounds as store
18.

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

### The duplicate trend carries two denominators (#1942)

`facts_idx.exact_duplicate_stats()` is the #499 trend instrument and the
nightly knowledge-health report prints it. Its exact-twin numbers are keyed on
`text_hash` and read 0 on a store where most active rows re-assert, in other
words, something their own source document already said about that entity.
The same call therefore also returns `active_rows`,
`same_source_paraphrase_groups` and `same_source_redundant_rows`: over active
rows only, the `(entity, source_doc)` groups with more than one row and the
rows in them beyond the first (20,078 groups, 80,990 of 122,556 active rows =
66.1%, read-only on 2026-10-01). The report prints that count over
`active_rows`, never over `rows` (which includes retired ones), and calls it a
population at risk: one document supports many distinct claims, and the share
that is one claim twice was measured on recent writes only (#1487, about a
third), not on this row set.

### The paraphrase gate (#1487, armed in `noop`)

#499 refuses a byte-identical re-statement; a paraphrase walks through it, and
~a third of the writes after the 09-23 rebuild restated a fact the entity
already held in that category. `agent_mcp/fact_write_gate.py` sits after the
verbatim guard in `_fact_add` (so the tool and post-capture) and in the
extractor's `write_fact_file`: the lexically closest active fact in the same
file (token Jaccard ≥ 0.3) is put to djev as one three-way `choice`, and the
write is **NOOP** at P(restated) ≥ 0.8 (nothing written, success returned),
**UPDATE** when P(superseded) ≥ 0.3 *and* every word and number of the old fact
appears in the new one (new appended, old stamped `expired_at` in the same
atomic write), else **ADD**. Every djev failure is ADD; the result carries
`verdict`. `knowledge_graph.write_gate.mode`: `off` | `shadow` | `noop` | `on`
(`LLOYD_FACT_WRITE_GATE` overrides; a caller may pass `write_gate: "off"` —
the rebuild's carry-over does — never on). Production has run **`noop`** since
the 2026-09-25 ship call (`config.yaml`): NOOPs apply, an UPDATE verdict is
logged as ADD and expires nothing, and `on` stays reserved for an explicit
decision plus a measured LloydMemEval `knowledge_update` gain — the 402 logged
`verdict: "update"` rows do not authorise arming it. Decisions log to
`fact-write-gate.jsonl` beside `kg.sqlite`. Held out on 400 writes: 76/77 NOOPs
right, 3/3 UPDATEs safe, djev p50 341 ms on the 37% of writes it is asked
about — `eval/measurements/fact-write-gate-2026-09-25.md`.

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
things. `fact_invalidate` expires; `fact_resolve_apply` invalidates —
`fact_resolve`'s `auto_resolve` parameter was removed outright by #1326.
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
2. **A human applies anything beyond the two mechanical shape tiers.** #48 ends
   its scheduled run with `--apply --tiers CASE,PUNCT` unattended on Alan's ruling
   on #990 (2026-09-16), because a case or punctuation variant of one name is a
   mechanical merge and coverage decayed for weeks while that apply waited on a
   person. What makes those two tiers safe to leave unattended is not the absence
   of risk — the 2026-08-22 wipe and the 2026-09-03 151-merge incident were both
   unattended applies of exactly this shape — but the shape of the apply: the
   degraded-graph refusal (rule 3), the high-value exclusion, one store
   transaction, and an apply report that journals each dir outcome so a kill in
   the gap between the transaction and the fact-file half is finished by
   `--resume <that report>`. A SUFFIX merge needs the semantic gate's unanimous
   verdict *and* a human, and #67 only proposes: a suffix is a claim about meaning,
   so it is never applied on schedule, which is why the tiers are named on the
   command line rather than inferred.
3. **Refuse on a degraded graph.** `--apply` and `backup-graph.sh` both compare
   active edges against `graph-baseline.json` and refuse below 50%. The backup
   half no longer fails open on its own missing input: a baseline that is
   missing, unparseable, or carries no positive `active_edges` is now a refusal
   naming the path, before the tarball is written or the window rotated, and
   every successful run prints the reference's `recorded_at` beside the counts
   (`tests/test_backup_graph.py`). The `--apply` half fails closed the same way now
   (**#2968**): a floor that is absent, unparseable, or carries no positive
   `active_edges` refuses an apply over a store that holds edges — exit 3, the path
   named, the reason printed as *the guard could not run* rather than as a
   degradation, nothing applied. `load_baseline` still answers 0 on any exception; what
   changed is that the guard no longer reads that 0 as "no opinion". The empty-store
   half has refused since #1557 and still does, and a measured floor of 0 still applies
   labelled `unmeasured` — a recorded zero is a statement, an absent one is a missing
   comparison, and the two are separate classes in `guard_vacuity.py` so a defaulted
   `measured` argument cannot quietly re-widen the hole. **The floor moves only on a run
   that got past this decision** (#3221): on the apply path, once a refusal has already
   returned, `update_baseline` ratchets it up to that run's *pre-apply* active-edge
   count, and an unmeasured apply bootstraps the file at that same place. A dry-run
   bootstraps a floor it could not read — the fresh-install route — and never raises one
   it read, because a read-only run must not move the reference the apply is then
   measured against; `scripts/maintenance/corpus_shape.py` names that self-reference as
   #543's failure in its own module docstring, "a baseline the run it judges rewrites".
   Either way a night that refuses leaves the floor as it found it
   instead of recording the damaged count as the new normal. Why the ratchet's location
   is stated as a rule: #2968 moved the bootstrap past the decision inside
   `if not baseline_measured:`, which quietly took the ratchet out of every run that
   *read* a floor — every nightly #48 apply — and the file froze at its first bootstrap
   (50,152, stamped 2026-10-17) while six applies grew the store to 52,878, each one
   reporting `baseline_active_edges: 50152`. A reference that stops being rewritten is a
   trip point drifting below the graph it guards, one edge per edge added. What still does not refuse, named so nobody rediscovers it:
   (a) `--allow-degraded` on one of those runs both applies and seeds the floor from the
   run's own count, re-arming the guard from a possibly-damaged store — deliberate, since
   a permanently absent file strands every following night (#1395); (b) `update_baseline`
   is max-only, so the ratchet can only ever *raise* the reference and a slow leak never
   trips it. What self-heals a refused backup within one nightly sweep is emphatically
   not max-only: a backup refuses on an *absent* reference, which only a bootstrap can
   answer, and the bootstrap that answers it is the **#48 dry-run's** — the job's skill
   runs the dry-run half before the apply every night, so the absent file is re-seeded
   by a run that cannot be refused; an unattended `--apply` alone would meet #1395's
   permanent refusal, because #2968 made an absent floor a refusal and a refusal writes
   nothing. (c) the floor is written by a job in the family that
   reads it, which is Deliverable 3's independent-reference ask. A backup
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
| `node_coverage_pct` | ≥ 30 (the pre-rebuild tree was at 14% when this was set; the 2026-09-23 run measured 77.6% at its own gate) |
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
facts newer than the export timestamp, `_edges_written_since_export` (#1664)
does the same for carryable edges — the facts-only guard once let a swap
destroy an edge stated after the export while every check reported clean —
and `swap` refuses if either finds any: re-run `export` and `import`, then
swap. `--force` exists and is documented as
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
the eval runs were kept under `_pipeline/backups/rebuild-20260903T214511Z/`
alongside `kg-before.sqlite`, a 2026-09-03 snapshot of the live store — that
directory has itself since been swept, and what stands under
`_pipeline/backups/` now are the two `rebuild-20260922T*` run dirs below.

If provenance ever becomes the binding constraint, the tool is `kg_rebuild.py`
and this section is the record of what it costs. Do not restart one without
first deciding what a two-query category is allowed to veto.

### The 2026-09-22 run: completed, swapped 2026-09-23 on a forced gate

The exercise was repeated twelve days later, and the veto question got its
answer. The live store was wiped on 2026-09-22 — what remained measured 30
entities and no edges (`a18dd985`, #1395) — and run `rebuild-20260922T212318Z`
re-extracted the whole allow-list (3,106 documents, 100% coverage): 98,174
facts, 11,561 entities, 32,808 active edges, `provenance_pct` 100, zero junk
dirs, zero contamination. Its frozen eval baseline scored the *empty* graph —
0.363 MRR / 0.421 NDCG@10, deliberate per `a18dd985`: after a wipe the rebuild
must at least beat having no graph — so this run's retrieval half is a
comparison against nothing, not against the 0.496 the 09-03 run was held to.
The gate passed nine checks and failed one: `eval_category_regression`, hard
−0.051 against the −0.05 limit — the same check class that abandoned 09-03.
`swap` ran with `--force`, after an ABBA re-measure under the regression lock
(run spread 0.013, every category delta inside it) and a rule stated *before*
the result, on Alan's call — recorded in `FORCE-SWAP-REASON.md` beside that
run's `gate.json`. But `rebuild-state.json` keeps only `gate: false` beside
`swapped_at`: the override itself leaves no trace in the state every later
reader reads (**#2967**). The live store today is that rebuild tree — see the
2026-10-18 re-measure in *Scale*.

And the 977 junk-named entity directories went out with the wiped tree rather
than through review: the rebuild's own gate counted 0 junk dirs before
swapping, the extractor rejects those names before registration, and
`_pipeline/memory-graph/junk-entities-review.json` no longer exists. The stock
was disposed of, not dispositioned.

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

Three of those address a fact by `<file, id>` -- `fact_resolve_apply`,
`fact_invalidate` and every revert report -- so the ID has to name exactly one
fact. `scripts/memory/repair_fact_ids.py` is the repair when it does not; see
**The rebuild** for what it fixed and why the merge path kept re-creating the
problem.

`fact_resolve` reports, and marks nothing. It used to default to
`auto_resolve=True`, so a call that reads like a query silently marked facts;
#1326 removed the parameter — a call that still passes it gets the report and
a pointer to the verb that acts, and the old id-keyed marking, which aimed at
an id that is a per-file counter and invalidated 25 facts from 2 planned
actions on live `Assistant`, was replaced by `(file, id)` marks that refuse a
mark whose file is unknown. Its contradiction detector fires on token overlap
above 0.6, which is two facts phrased similarly, not two facts that disagree.

`fact_resolve` and `fact_resolve_apply` share that detector, and it is pairwise —
O(n²) in the entity's fact count. It is refused above `FACT_GODNODE_THRESHOLD`
inside `_detect_contradictions_sync`, so both tools inherit the refusal; the
guard used to sit on `auto_resolve` alone, which left the reporting path
running the scan. Against `Lloyd` that was 15 million comparisons: 113 seconds
through MCP, returning 32,857 "contradictions" that were almost entirely the
overlap heuristic firing on two facts phrased alike. Both tools now take
`category` to scan a slice — 113s → 4ms.

## Review log

- **2026-10-26 — #3354 fixed: the #51 writer bullet still typed its own edges.**
  §The write chain said co-access pairs from session trajectories landed as plain
  co-access edges, which stopped being true on 2026-09-12 (#420, `e4f6370f`):
  `land_approved_edges()` has written the type the Stage-2 classifier returned and
  kept `co_accessed` only as the fallback for a pair nothing classified. Measured
  2026-10-26 through `app.kg_store`: **0** edges of that fallback type among the
  562 `conversation`-origin edges, and none anywhere in the store, while the job
  is live (newest such edge `created_at` 2026-10-25T06:08:03Z). The same sentence
  stood in `architecture/autonomy.md`'s Knowledge-Graph Chain table and in the
  vault's `architecture/knowledge-graph.md`; all three now read classifier-typed
  with the fallback, and the vault note's write-path diagram lost its dry-run label
  and its human-runs-`--apply` arrow too — the same pair corrected below on
  2026-10-13 — with rule 2 of its own incidents list narrowed to match, so the note
  no longer says a person must run `--apply` two lines under a diagram that says the
  nightly run does it unattended. Pinned by
  `tests/test_kg_doc_claims.py::test_the_51_writer_bullet_says_the_edge_is_classifier_typed`,
  `::test_the_kg_chain_table_row_says_the_edge_is_classifier_typed`,
  `::test_no_architecture_doc_says_pairs_simply_become_co_accessed_edges`,
  `::test_the_51_claims_reject_the_pre_420_wording_when_restored`,
  `::test_the_vault_note_says_the_edge_is_classifier_typed`, and
  `::test_the_vault_note_diagram_shows_48_applying_the_mechanical_tiers`, with
  `::test_the_quoted_fallback_chain_is_the_chain_the_code_runs` parsing
  `land_approved_edges()` so the sentence is checked against the writer and not only
  against itself.
- **2026-10-24 — #3221 fixed: rule 3's floor had stopped moving.** The rule said a
  floor is "bootstrapped only by a run that went on to apply", and after #2968 that
  was true in a way nobody intended: the move put `update_baseline` below the guard
  decision inside `if not baseline_measured:`, so a run that *read* a floor never wrote
  one and the max-ratchet was unreachable on every nightly #48 apply. Measured on disk:
  `graph-baseline.json` frozen at 50,152 (`recorded_at` 2026-10-17T15:19:37Z) across six
  applies whose own `safety.active_edges` climbed to 52,878, each reporting
  `baseline_active_edges: 50152`. Three consumers read that one number — the sweep's
  guard, `backup-graph.sh`'s `THRESHOLD=$((BASE / 2))` and
  `knowledge-health-report.py`'s `baseline * 0.5` — so a 27,802-edge loss (53% of the
  store) passed all three. The apply path now ratchets the floor after the decision, on
  the count it read before applying, max-only as ever; the dry-run stays bootstrap-only.
  Rule 3's (b) also credited `update_baseline` being max-only with self-healing a
  refused backup within one nightly sweep — true, but by the #48 dry-run's bootstrap of
  an absent floor, not by max-only, and the sentence now says which is which. Pinned by
  `tests/test_entity_sweep.py::test_a_measured_apply_above_its_floor_ratchets_the_floor_to_its_pre_apply_count`,
  `::test_a_measured_apply_below_its_floor_never_lowers_the_floor`,
  `::test_a_refused_apply_leaves_the_floor_it_compared_byte_identical`,
  `::test_a_dry_run_bootstraps_an_absent_floor_and_never_ratchets_one_it_read`, and
  `tests/test_kg_doc_claims.py::test_rule_3_names_where_the_ratchet_lives_and_what_re_heals_a_refused_backup`.
- **2026-10-18 — stale.** All the machinery is live and re-verified against the
  tree (store API, gate order, write chain, tools, the doc-claim tests), but the
  document's *measured state* was three weeks behind an event it never
  mentioned: the live store was wiped on 2026-09-22 and a completed rebuild
  swapped in on 2026-09-23 with `--force` after `eval_category_regression`
  failed by 0.001 against a baseline scored on the empty graph. Added *The
  2026-09-22 run*; re-measured Scale on the current tree (15,164 dirs / 41,557
  files / 161,168 rows / 50,558 active edges / 239 aliases / 0 NULL
  `created_at`/`source_doc` — the provenance gap called "still outstanding"
  closed with the swap, and the 977 junk dirs and 4,028 mechanical aliases went
  out with the old tree, which also carries `conversation`-origin edges, 0 in
  the new store until #51 re-grew them); corrected the paraphrase gate from
  *ships off* to its live `noop` mode; narrowed the sweep's `--apply` fail-open
  claim to what #1557 left open; replaced `fact_resolve --auto_resolve`, a verb
  #1326 removed, with `fact_resolve_apply`; re-dated the candidate sidecar
  (432 names); dropped three drifted `file:line` citations; recorded that the
  09-03 run's kept artifacts dir is itself gone. Filed **#2967** (a forced
  `kg_rebuild swap` leaves no trace in `rebuild-state.json`) and **#2968** (a
  degraded-but-nonempty store with a missing baseline still applies unrefused).
- **2026-10-13 — #2641 fixed: the apply rule now shows its narrowing.** Two sites
  in this file still said #48 only ever proposes and that a person is the one who
  applies: rule 2 in §Rules that came out of the incidents, and the write-path
  diagram, which labelled the daily run dry-run-only and ended with a human
  reading the plan and applying it. Neither has been true since Alan's ruling on
  #990 (2026-09-16) — step 3 of the job's own skill prompt ends every scheduled run
  with `--apply --tiers CASE,PUNCT`, the apply reports land daily, and the run
  records leave the gate-approved SUFFIX clusters for a person to apply — so a
  reader who obeyed this doc was stopping the nightly write the job is authorised
  to make. Rule 2 and the diagram now name the two mechanical tiers as unattended
  and keep every apply beyond them, a SUFFIX merge, a human act; the narrowing, not
  a repeal. `architecture/autonomy.md` already said this in its job roster and the
  paragraph under it — this file was simply never brought along. Pinned by
  `tests/test_kg_doc_claims.py::test_rule_2_says_48_applies_the_mechanical_tiers_unattended`,
  `::test_the_write_path_diagram_shows_48_applying_the_mechanical_tiers`,
  `::test_the_rule_names_the_narrowing_not_a_repeal` and
  `::test_review_log_names_2641_and_both_corrected_sites`.
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
