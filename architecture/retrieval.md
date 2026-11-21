---
title: Vault recall — the retrieval pipeline and what was measured
status: implemented
date: 2026-11-21
---

# Vault recall — the retrieval pipeline and what was measured

`vault_recall` (`agent_mcp/vault.py::_vault_recall`) is how Lloyd finds notes. It
runs a document leg (the qmd daemon), a grep over Lloyd's code and an entity-facts
leg in parallel, with a fourth entity-graph expansion leg available to the caller —
`RECALL_EXPAND_GRAPH` is false at the defaults, so a production recall does not
spend it — and hands back the top documents plus facts. This doc is about the **document leg**: how its candidate pool is
built, who orders it, and every change to it that was measured on 2026-09-21,
kept or rejected. CLAUDE.md carries a line per decision; this is the long version.

## 1. The pipeline today

```
query ──► qmd /query (daemon :8181, fork ~/lloyd/qmd)
            lex leg:  BM25 over the 11 vault collections + lloyd-architecture
                      (+ `sessions` as the episodic floor, #1485), terms ORed
            vec leg:  Qwen3-Embedding-0.6B (embeddinggemma-300M until 2026-09-21)
            fusion:   global — one ranking across collections by score, each leg
                      fused 100 deep (`QMD_FUSION_DEPTH`, §3.6)
            pool:     fused head 18 + floors of 2 for autonomy/architecture/skills
                      and 1 for `sessions`: 18 + (2+2+2+1) x 2 = the ≤32 rows one
                      canvas holds (cross-encoder OFF). Head 20 with the floor off.
       ──► djev (GPU 2, :8011) orders the ≤32 rows          (RECALL_RERANKER "djev")
            160-char candidates, one read, max_n = `recall_djev_pool()` — derived
            from the served canvas (128/4 = 32), not a constant, 4 s timeout
       ──► top `limit` documents
fallback: djev does not answer ──► the whole recall re-runs on the cross-encoder
          path (global 40 + floors 5, plus the `sessions` floor while the episodic
          one is on, qmd rerank on), counted and announced by
          app/qmd_health.py::note_ranker
```

- **One definition of the request.** `recall_doc_leg_shape()` is read by the doc
  leg, by `vault_search`'s recall rows (`_vault_search_recall_rows`) and by
  `recall_doc_pool()` (the `djev.rank` `request_shape` report), and by
  `scripts/automod/evalpin.production_payload`, so the regression pin warms exactly
  what production sends and a change to the request lands once. The pool is the
  served djev canvas read at call time through `recall_djev_pool()` →
  `djev.canvas_rows()` (128 rows ÷ 4 candidates per question = 32), never a copied
  constant; the head and the floors are measured numbers and must not become
  derived — `tests/test_episodic_recall.py` checks their arithmetic against the
  function.
- **Kill switches.** `RECALL_RERANKER = "qmd"` restores the cross-encoder path;
  `RECALL_LEX_MODE = "and"` restores the AND keyword leg (and sends no key);
  `RECALL_QMD_FUSION = "collection"` restores #504's 240-row per-collection request;
  `retrieval.recall.episodic_floors: false` in config.yaml — or the
  `LLOYD_RECALL_EPISODIC=0` that `tests/conftest.py` pins, so a development box
  cannot score a floor it never asked for — drops the `sessions` collection and
  returns the head to 20.
- **Paths arrive decoded.** qmd percent-encodes every path segment; `qmd_file()`
  decodes at `_qmd_post` (and again in the Mission Control memory search), so
  `people/Ali Behrouz.md` is citable. 18 indexed notes carry a space, `+`, `#`
  or `&`. Every caller now goes through `qmd_query` (#1498): the MC memory
  search (`app/routers/memory.py`) and write-time backlog dedupe
  (`agent_mcp/backlog_similar.py`) included, so their rerank fallbacks are
  counted by `qmd_health` like the recall's. See `architecture/qmd.md`.
- **Which collections.** `VAULT_SEGMENTS` is the eleven vault segments, and
  `vault_search.architecture` (config.yaml, read at aggregator import by
  `apply_architecture_collection`) decides the architecture docs: `vault` is the
  vault's `architecture` copies, `repo` swaps in `lloyd-architecture` (this
  checkout's `architecture/*.md`), `both` (deployed 2026-11-10) asks for the two,
  with the floor staying on the vault copy. A `lloyd-architecture/<doc>.md` hit
  opens through `vault_read`, which serves it from the checkout read-only;
  `vault_write` refuses it. The counters (`vault_overview`, MC stats) walk
  `VAULT_DIR_SEGMENTS`, the vault's own directories. The prefetch leg honours the
  same switch now: `app/prefetch.py` keeps its own narrower list (a
  scope-restricted subset, fused with no fold and no global-fusion payload) with an
  `architecture` **slot** that `_derive_vault_collections()` replaces with
  `architecture_collections()` at import — so under `both` prefetch names both
  names too. It used to carry a second `== "repo"` ternary here, which honoured one
  mode only and left the leg searching the vault's stale clone of these docs
  (#4392, fixed `fa5298e0` 2026-11-20). `eval/measurements/architecture-collection-2026-11-10.md`.
- **Session hits are ids, not paths.** A `sessions` hit (the episodic floor, or
  prefetch's recent-sessions leg) carries `session_id` and `open_with:
  session_read`; `session_read` (`agent_mcp/session.py`) returns that chat's
  user and assistant text by turn range, size-capped. The export is re-written
  whenever the chat grows (`app/post_capture.py`), not only at turn 1.
- **`vault_search` has its own pipeline key.** `vault_search.pipeline: legacy`
  (AND keywords, cross-encoder, the tool's long-standing shape) or `recall`
  (this doc-leg's request and djev order) — deployed `recall`, with
  `vault_search.consolidate_default: true`, which is whether the tool spends a
  primary-model call summarising its hits when the caller does not say. Both are
  measured by `eval/run_vault_search_eval.py --compare` (it runs both arms,
  alternating per query); the defaults are what that measurement chose.
- **Tests never reach the live djev.** `tests/conftest.py` pins the recall to
  `"qmd"`; `tests/test_recall_djev_ranker.py` opts in with djev stubbed.
- **The prefetch hybrid leg lands on the same turn** (2026-09-24, #1482
  `d9ac3e59`): it costs ~55 ms after the lex leg now, not 1.1–3.0 s, so
  `_prefetch_run` waits for it and fuses it with the lex hits; carry-over to the
  next turn is only the busy-daemon fallback. The whole recall stack was
  reviewed that day against the frontier — `architecture/recall-research-2026-09-24.md`,
  items #1480–#1495 — a note `4a6cdd54` deleted on 2026-09-27 when research stopped
  being an architecture doc; read it back with
  `git show 4a6cdd54^:architecture/recall-research-2026-09-24.md`, and
  `architecture/index.md` names it under Retired.

## 2. How it is measured

`eval/run_eval.py` over `eval/vault_recall_queries.yaml`, the gold set, against a
**pinned qmd snapshot** (`scripts/automod/evalpin.PinnedCorpus` with its own
index name and port, so it never collides with the regression runner's pin), and
a per-arm wrapper that changes one thing at `_qmd_post` or a module constant.
**The pin is the arms that ask for it, not `run_eval` itself:** `PinnedCorpus` is
constructed twice in `workers/sources/automod_regression.py` (the noise arm and the
dev/holdout arm) and nowhere in `eval/run_eval.py`, which reaches the daemon through
`agent_mcp/vault.py`'s `QMD_DAEMON_URL` — so the nightly baseline scores the live
index, and what it scored with is recorded beside it. The `corpus` block its
artifact records (`eval/run_eval.py:128`, whose `label_agreement_ceiling` import the
block pins) is **no longer the KG half only**: #1374 gave it a `doc` half, owned by
`app/doc_corpus.py` — `vectors` read off the serving daemon's `/health`, `corpus_mode`
(`pinned` when a `LLOYD_CONFIG_OVERLAY` is serving the run, `live` otherwise, derived
never asserted), and the five identity keys `index_path` / `index_mtime` /
`index_size_bytes` / `content_vectors` / `index_reason`, written under
`doc_corpus.DOC_KEY` at `eval/run_eval.py:172` and answered by `doc_corpus.doc_ok` —
and `eval_trend_stats.fmt_drift` refuses to read a missing vector count as zero drift:
a baseline from before the `doc` half prints `doc_vectors unknown`, and a run whose
fact half matches prints "corpus identical on the fact half, doc side unknown", never
the bare "corpus identical" that used to print through a re-embed. A probe that cannot
answer writes `corpus["doc"] = null` rather than `vectors: 0`, which is what forces
every later reader to say *unknown*. Two more fields of the same artifact carry the rest of the
provenance a trend line needs: `observed_ranker`, the ranker the daemon reported for
that run rather than the knob that was requested (#3675), and the report-only
`episodic_floor_leg` that scores the generated past-chat proxy set against the floor
state the run actually served (#4404).
Every comparison is **paired per query** with a 10,000-draw bootstrap 95%
interval; two arms on different snapshots are only compared through a control arm
that matches the earlier pin query for query.

- **The rule** (Alan, 2026-09-21): equal accuracy with more throughput is a win; a
  slower change has to show a gain clear of the interval.
- **Count what the arm did, not only what it scored.** Every arm dumps
  `djev.stats()` and `qmd_health.stats()`. A djev call the client refuses (a
  `max_n` past the canvas, a split canvas) silently falls back to the
  cross-encoder, and an arm that looks like "djev plus X" is then the fallback —
  one arm on 2026-09-21 was exactly that until its doubled doc-leg count gave it
  away.
- **Check the pool before the ranker.** A reranker can only reorder what fusion
  hands it. Rerank-off pool ceilings (does the pool contain ANY expected doc)
  are cheap and bound every reranker arm.
- **A ranker that becomes the decision stops being observable.** The djev
  `rerank` shadow seam is on an `elif` behind `if ranker == "djev"` in
  `_vault_recall` — cited here as a line range until this round's own change
  moved the dispatch by ~120 lines, which is the lesson, not a footnote — so
  since #1336 it records nothing:
  `~/.local/state/lloyd-djev/shadow.jsonl` held 30 rerank rows on 09-20, 4 on
  09-21, none on 09-22, while the `dedupe` and `entity` seams kept flowing, and
  the same dispatch ignores a caller's `djev_rerank: true` (#1372). Volume, not
  an error, is how this failure shows. Fixed rather than only noted: a recall
  that was sent an inert knob names it in its own result
  (`RECALL_UNUSED_KNOBS_KEY`), and `djev_status` reports `shadow.seam_log` per
  seam plus `shadow.structurally_dark` for the seam the dispatch itself cannot
  reach, so the dark read no longer needs this bullet to survive.

### 2.1 The gold set was repaired on 2026-09-21

When #1319 grew the set from 20 to 87 queries, labels that matched no path were
"re-pointed" to whatever file matched the old string. An audit of the 68
(query, label) rows no retrieval leg could surface found 59 of them to be label
problems: 35 pointing at the wrong file, 24 weak, 4 questions the vault cannot
answer at all; 5 labels sat under `skills/.archived/`, which qmd never indexes.
38 labels were re-pointed to documents that answer the query (each checked
against the indexed `documents` table: 139 labels over 81 queries, every one
resolves) and 6 unanswerable queries were dropped (87 → 81). The original 20
query texts are unchanged (their digest is pinned); their labels moved where the
old ones were wrong, so absolute numbers re-base here. The disk guard in
`tests/test_eval_corpus_guard.py` now skips dot-directories, as qmd's indexer does.

## 3. What changed, with the numbers

### 3.1 djev orders the pool (#1336, `e7bb4280`)

On one pinned snapshot, the same ≤32 rows ordered by:

| ordered by | doc_hit | MRR | NDCG@10 | recall p50 |
|---|---|---|---|---|
| qmd cross-encoder | 0.425 | 0.193 | 0.214 | 1.04 s |
| **djev, 160 chars, one read** | **0.471** | **0.249** | **0.274** | **0.51 s** |
| djev, full candidates | 0.437 | 0.226 | 0.251 | 0.87 s |
| fusion order | 0.414 | 0.155 | 0.179 | 0.15 s |

End to end against the cross-encoder path: MRR +0.072 [+0.009, +0.135], p50 0.51 s
against 1.2–2.2 s. `samples: "auto"` was 200 ms slower and ranked worse; one read
is deterministic.

### 3.2 The keyword leg ORs its terms (qmd fork `lexMode`)

qmd joined every non-stopword term with AND, so a question matched only
documents holding every word. Keyword leg alone, 152 expected documents:

| | within top 10 | top 32 | top 240 |
|---|---|---|---|
| AND | 0.09 | 0.11 | 0.13 |
| **OR** | **0.25** | **0.36** | **0.54** |

End to end on one pin (djev ranking), OR against AND: doc_hit +0.011, doc_recall
+0.033, MRR +0.028, NDCG +0.042 [−0.005, +0.094] at +16 ms; on the cross-encoder
path doc_hit +0.046, doc_recall +0.050. `lexWeight` 0.5 and 2 sat inside the
noise of 1. Every other qmd caller keeps AND.

### 3.3 Qwen3-Embedding-0.6B for the vector leg

A side index with every searched collection re-embedded (26,672 chunks),
served with the same model for queries, against embeddinggemma-300M. Because the
side index's snapshot was 8–50 minutes older than the controls, it was compared
with THREE embeddinggemma controls from different moments; they agreed with each
other in aggregate (MRR 0.337–0.349), and Qwen3 beat every one:

| on the repaired gold set (81) | doc_hit | doc_recall | MRR | NDCG@10 | p50 |
|---|---|---|---|---|---|
| embeddinggemma-300M (production) | 0.691 | 0.615 | 0.349 | 0.381 | 497 ms |
| **Qwen3-Embedding-0.6B** | **0.741** | **0.652** | **0.399** | **0.451** | 508 ms |

Paired against the three controls: MRR +0.049 to +0.062 (each interval above
zero), NDCG +0.065 to +0.076 (each above zero), doc_hit +0.049 to +0.062. The
model is set in one place, `models.embed` in `~/.config/qmd/index.yml`, which
beats `QMD_EMBED_MODEL` in `resolveEmbedModel`; `evalpin.yml` must name the same
model, since the regression pin serves a copy of production's vectors. Changing
it is a full re-embed — every chunk vector in `content_vectors`, for every
collection `~/.config/qmd/index.yml` names (15 on 2026-11-21, `sessions` and
`lloyd-architecture` among them) — so read the size before costing it rather than
quoting this line: `GET :8181/health` → `vecIndex.vectors` read 38,796 on
2026-09-22 and 97,398 on 2026-11-21, and `architecture/qmd.md` deliberately pins no
count at all (it too carries dated readings only). Cost the re-embed from
that number and from the measured chunks/s.

### 3.4 Floors, not depth (#1335, `77a30cbb`)

Global fusion outscores small collections wholesale. Floors of 5 (cross-encoder
path) or 2 (djev path) on autonomy, architecture and skills recovered most of the
hits a 40-row global pool lost; a wider pool barely helped.

### 3.5 djev ranks document text, not qmd's diff header (#1467)

`_djev_doc_text` builds each row as title + snippet, cut at `RECALL_DJEV_CHARS`
(160). qmd returns the snippet as a diff hunk (`2: @@ -1,4 @@ (0 before, 153
after)` plus `NN:` on every line), and on a live 25-row pool that left ~40
characters of document text per 156. The row now goes through
`strip_qmd_snippet`, the helper `vault_search` and the prefetch already used.
One pin, 86 queries, paired against the unstripped row:

| row | MRR | ΔMRR [95%] | better/worse | NDCG@10 | ΔNDCG [95%] | latency |
|---|---|---|---|---|---|---|
| unstripped, 160 | 0.319 | — | — | 0.357 | — | 663 ms |
| unstripped, repeat | 0.328 | +0.009 | 2/1 | 0.363 | +0.006 | 662 ms |
| **stripped, 160** | **0.364** | **+0.044 [−0.003, +0.093]** | **16/7** | **0.393** | **+0.036 [+0.001, +0.075]** | **559 ms** |
| stripped, 240 | 0.330 | +0.010 | 15/9 | 0.365 | +0.008 | 655 ms |
| stripped, 320 | 0.323 | +0.003 | 12/14 | 0.356 | −0.000 | 737 ms |

doc_hit is unchanged (a pure reorder); wider rows give the gain back, as #1336
found for full candidates, so the width stays 160. A second pin the same
evening agreed in direction with a smaller step: MRR +0.022 [−0.024, +0.070],
NDCG@10 +0.017 [−0.015, +0.053], 12 better / 10 worse, 658 → 555 ms. Adopted on
equal-or-better accuracy plus the ~100 ms.

### 3.6 Each fusion leg fused 100 deep (#1475, fork `079c9c9`)

Global fusion cut both legs to the candidate limit (20) before RRF: the lexical
leg fetched 200 hits and fused 20, so a document at lexical rank 21 got nothing
from that leg however high the vector leg put it. `QMD_FUSION_DEPTH=100` in the
daemon's conf fuses both legs 100 deep. The pool djev ranks is unchanged in
size; only which rows reach it moves. Paired against depth 20, production recall:

| pin | doc_hit | MRR [95%] | NDCG@10 | latency |
|---|---|---|---|---|
| 1 | +0.012 | +0.034 [−0.023, +0.093], 26/22 | +0.031 | 572 → 572 ms |
| 2 (fresh snapshot) | +0.012 | +0.039 [−0.018, +0.096], 25/21 | +0.030 | 566 → 572 ms |

Same direction on every metric on both pins at flat latency; depth 200 was no
better (MRR +0.027). Adopted on the same bar as §3.5.

### 3.7 Entity seeds: alias surfaces and semantic seeds (#1486)

The entity leg's ceiling was its seeds: 25 of the 66 gold queries with expected
entities had no expected entity among them (`anchorless_query_count`), because
those queries paraphrase their subject. `retrieval.entity_seeding` adds two
sources to `extract_entities_from_query`, both unioned AFTER the lexical top
`RECALL_SEED_TOP_K` so the first ten seeds never change: alias surfaces matched as
names (scoring their canonical), and `k=3` Qwen3-Embedding-0.6B nearest entities
over `name (kind): top facts` (`app/entity_linker.py`, index built by
`scripts/memory/build_entity_vectors.py`, query embedded in-process by
`app/qwen3_embed.py`). That builder had **no runner at all** when #3907 read it: no queue
kind, no autonomy task and no unit in this tree ran it, so the artifact was hand-built and
its age was bounded only by `MAX_INDEX_AGE_DAYS` in `app/entity_linker.py` and the reading
`scripts/memory/build_entity_vectors.py --status` (rows, age, live-entity coverage,
provenance; exit 1 stale, 2 absent, 3 store unreadable). #4021 armed autonomy job **#111**
on 2026-11-09 to rebuild it weekly (`skills/entity-vectors-rebuild/SKILL.md`,
`timeout_seconds: 7200`), so the bound now has a cadence behind it. Read again at
2026-11-09T21:59Z, before that job's first run, the same check still exits 1 — 11,990 rows
against 16,964 live entities, **44.90 days** old, 70.4% coverage (5,020 entities unseedable,
46 index names no longer live), `build.json` **ABSENT**
(#3907 read it at 42.9 days with `build.json` absent) —
and `_load_index` now logs the same line once per cold load instead of
serving an unbounded-age index in silence. **#111 has run since.** Read again at
2026-11-21T20:47Z the same check exits 0: 17,485 x 1024 rows, built 2026-11-17,
4.32 days old against the 10-day bound, 98.4% live coverage (276 entities unseedable,
13 index names no longer live), `build.json` present naming its store — so the stale
reading above is the state before the cadence existed, not the state now, and job #111
sits `up_next` with `last_run` 2026-11-17 and `next_run` 2026-11-24. Note what the
bound is: `index_status()` says outright that nothing consults `MAX_INDEX_AGE_DAYS` to
change behaviour and no caller may use `stale` to drop a seed, so the age is a reported
comparison and the only thing that actually bounds it is #111's weekly cadence plus
`--status`'s exit codes. Miss two weekly runs and the semantic leg keeps serving an
index the bound has already called stale, quietly and by design. `recall_seeds` is the one list `vault_recall` and the eval
both use; the doc leg starts before the seeds when the semantic half is on.
Paired on one pin, 86 queries: entity_hit 0.337 → 0.500 (+0.163 [0.093, 0.244]),
fact_entity_recall +0.172 [0.088, 0.265], anchorless 25 → 16 (strict 38 → 27),
every doc metric identical, +71 ms mean latency; the same seed count taken
lexically (width 13) moves nothing. Rule-regenerated aliases
(`scripts/memory/regenerate_aliases.py`, 2,100 rows on a copy) moved nothing on
the gold set. `eval/measurements/entity-anchoring-2026-09-25.md`.

### 3.8 Topic phrases fused into the fact leg of `vault_recall` (#1456)

**Measured and left off** (`vault_recall.topics_merge: off`; the code default is `off` too). The
focus extractor drafts up to 3 topic phrases beside the raw recall, each phrase
runs the fact leg only, and the fact lists are RRF-fused (k=60) to one recall's
length; documents are untouched and prefetch never sees it. Pinned corpus,
86-query dev set, djev replayed: fact_entity_recall **+0.083 [+0.030, +0.152]**
(6/0, n=66), entity_recall +0.073 [+0.015, +0.136]; holdout 2 wins, 0 losses on
every entity metric; +43 ms p50 / +346 ms p95. The `full` shape (documents fused
too) gained doc_hit +0.093 [+0.023, +0.163] at ~+1.05 s a recall. **On top of §3.7's
entity seeding the "facts" gain is gone**: fact_entity_recall −0.030 [−0.083, +0.008]
(1/3), entity_hit 0.000, at +492 ms p50 — seeding already reaches the entities the
phrases added, and each topic recall pays the seeding embedder. Off; revisit only with
facts-only topic recalls that skip semantic seeds. `eval/measurements/recall-topics-merge-2026-09-25.md`.

## 4. What was measured and not kept

| idea | result | why it stays out |
|---|---|---|
| djev on top of the cross-encoder (#1335) | +0.054 MRR, [−0.002, +0.112], +0.5 s | slower, gain not clear |
| wider djev canvas, deeper pools (#1345) | 44 rows +0.011 hit (noise), 92 rows −0.115 | djev's listwise order degrades with N |
| djev tournament over 64 rows | hit −0.046, MRR −0.035, 1.4 s | worse and 2.7× slower |
| djev collection routing (3 variants) | all within ±0.023, +50 ms | slower, no gain; the static floors already cover it |
| vector leg fed the raw question | MRR −0.015, NDCG −0.015 | no gain |
| query-type routing | per-category oracle, fitted after the fact, only 0.28 → 0.32 MRR on 11–20 queries a cell | cannot beat noise; not built |
| front-matter-aware titles (qmd fork, branch `title-fix`) | 223 notes retitled (17 of 36 autonomy tasks were "Activity Log"); on the repaired gold set doc_hit −0.012, the rest within ±0.008 | no measured gain once OR landed; the extractor is kept on the branch |
| activity logs stripped from backlog/autonomy (1,272 docs, −20% bytes) | doc_hit −0.037, NDCG +0.024, both inside the interval | mixed |
| qmd's own query expansion (1.7B) | doc_hit −0.037, doc_recall −0.061, MRR +0.027; generation 0.74 s p50, 4.7 s p90 | slower, trades hits for order |
| near-duplicate collapse before ranking (48-row pool) | doc_hit −0.012, MRR +0.015, +78 ms | neutral, slower |
| `~/lloyd/architecture` as a qmd collection, config entry only (2026-09-21; #3076 2026-10-20) | doc_recall −0.025 (−0.059 with the floor moved to it); #3076: 0.000 on both legs | superseded 2026-11-10: a collection the recall does not name is never asked for. Deployed as the two-change option (`vault_search.architecture: both`), measured over the restored five: mrr_doc +0.061 [+0.009, +0.123] over 79 doc-gold queries, three low-ranked hits lost on the other 74 |
| binary relevance rows instead of the 4-level scale (#1468) | `noul` MRR −0.063 [−0.114, −0.018], 9/17; 2-level −0.010 | the scale's expected value carries the gradation |
| FIRST-style top-1 `choice` row, with and without CapCal (#1469) | −0.074 / −0.064 MRR, both clear of zero; letter labels alone −0.036 | the row's pick disagreed on 55/86 and lost |
| djev orders the facts leg (#1470) | fact_entity_recall +0.008 (1 query), +280 ms | no gain |
| pools past one canvas: 48/64 rows, anchored, naive or server-`sequential` chunks (#1474) | 64 rows doc_hit +0.058–0.070 and doc_recall +0.06–0.09, but MRR −0.064 to −0.067 (clear of zero) and +0.5 s; `sequential` MRR −0.137 | coverage bought with head order; slower |
| residual-quantized entity codes, shared-prefix-depth seed prior (#1497) | codes are finer than folders and carry edges (d2 edge lift 21.6 vs folders 3.8) but plain cosine at the same selectivity carries 62.2; as a prior on the semantic seeds, α 0.02–0.2 → 20–23 anchorless vs 19 at α 0 | quantized cosine, strictly less than the cosine it quantizes (`eval/measurements/rq-entity-codes-2026-09-25.md`) |
| PPR over the KG + entity→doc mentions, fused into the pool (#1492, scoped) | multi-hop+hard n=31: recovers 0/9 (lexical seeds) and 1/9 (semantic seeds) of the doc misses; equal RRF doc_hit −0.13 [−0.26, −0.03] / −0.10, weighted RRF Δ 0, MRR +0.01 [−0.09, +0.12]; +0.1 s | the entity leg's ceiling was the seeds, not propagation (`eval/measurements/ppr-graph-leg-2026-09-25.md`) |
| RRF `lexWeight` 0.5–2.0, floors 1/1/1 or none (#1475) | 1.5–2.0: doc_recall +0.053 [+0.015, +0.099] with MRR −0.035; 0.5–0.75: doc_hit −0.081 | mixed or worse |

## 5. What the gold repair did to the numbers

The same production configuration scores doc_hit ~0.52 on the old 87-query set
and ~0.69 on the repaired 81: most of the "misses" were labels that named the
wrong file. Absolute numbers re-base at the repair (`scripts/eval_trend_stats.py`
says so); paired comparisons made before it are still valid, just less powerful,
because a broken label misses in every arm. The set is **86** queries since
2026-11-10 restored five `lloyd-architecture` queries (81 → 86, another re-base
point, §4), and `tests/test_eval_corpus_guard.py` re-counts the file against its own
`GOLD SET SIZE:` header and against that number together — a growth is allowed, but
it has to move the header and the pin in the same change, so a silently larger set
cannot quietly move a trend line.

## 6. Files

- `agent_mcp/vault.py` — `RECALL_*` constants, `recall_doc_leg_shape`,
  `_qmd_daemon_search`, `_qmd_post`, `qmd_file`, `_djev_rank_recall`, `_vault_recall`
- `app/djev.py` — `rank(chars=, samples=, max_n=)`
- `app/qmd_health.py` — rerank and ranker fallbacks, counted and announced
- `scripts/automod/evalpin.py` — `production_payload` mirrors the doc leg,
  including #1485's episodic floor (`retrieval.recall.episodic_floors`: the
  `sessions` collection joins the request's collections and `collectionFloor`);
  `PinnedCorpus` refuses a snapshot holding no document of a collection that
  request names (`snapshot_collections`, `missing_collections`)
- `qmd/src/store.ts` (fork) — `buildFTS5Query(query, mode)`, `searchFTSAcross`, `structuredSearch` options
- `app/doc_corpus.py` — the document half of an eval run's corpus identity (#1374):
  `collect()` for the daemon-side vector count and `corpus_mode`, `index_identity()`
  for the five identity keys, `doc_ok()` for the artifact's verdict, and `None` (never
  a zero) when the probe cannot answer
- `app/entity_linker.py`, `app/qwen3_embed.py`, `agent_mcp/retrieval.py::recall_seeds` — entity seeding (#1486)
- `eval/vault_recall_queries.yaml` — the gold set; `tests/test_eval_corpus_guard.py` guards it
- tests: `test_recall_first_stage.py`, `test_recall_djev_ranker.py`, `test_recall_global_fusion.py`, `test_doc_candidate_pool.py`,
  and `test_episodic_recall.py` for #1485's floor — the only file that checks the
  head-plus-floors arithmetic against `recall_djev_pool()` and that
  `production_payload` is the doc leg's request

## Review log

- **2026-09-22** — `current`. §1's diagram, the request shape (`limit` 32 /
  `candidateLimit` 20 / floors `{2,2,2}` / `lexMode "or"` / cross-encoder off),
  `RECALL_DJEV_*` (160 chars, 1 sample, pool 32, 4 s), the three kill switches,
  the 11 vault collections, the 81-query gold set with its 139 labels, the 18
  encoded paths and `eval/stats.N_RESAMPLES = 10000` all re-measure true at
  `9d4c9ad`; `pytest` on the four recall/corpus test files named in §6 passes.
  Corrected four things: `_qmd_post` is not the only reader of a qmd reply
  (`app/routers/memory.py`, `agent_mcp/backlog_similar.py` open their own
  sockets, so their rerank fallbacks are uncounted); §2's **pinned qmd snapshot**
  describes the automod arms only — the nightly scores the live index and its
  artifact records no document-corpus identity, which is what makes the corpus
  gate blind to a re-embed (#1374); the chunk count that priced a re-embed was
  stale, so §3.3 now says where to read it; and §2 carries the new measurement
  trap that djev-as-ranker switches the `rerank` shadow seam off (#1372). Filed,
  not fixed here: the nightly task's prescribed `run_eval.py` command does not
  parse, and the baseline it produces is scored graph-on against its own
  procedure (#1373).
- **2026-11-10** — `lloyd-architecture` deployed beside the vault's
  `architecture` (§1 "Which collections", §4); the five checkout-doc queries
  restored (81 → 86, a re-base point); session hits name `session_read`; the
  `vault_search` pipeline and consolidate keys documented. Corrected: the MC memory
  search and backlog dedupe no longer open their own sockets — both call
  `qmd_query` since #1498.
- **2026-11-21** — `stale`. Four claims had described what ran rather than what runs,
  all of them sentences rather than mechanisms: §2 still said the nightly artifact's
  `corpus` block is the KG half only and that `corpus_diff` prints "corpus identical"
  through a re-embed, but #1374's fix landed in `a0b1b6b9` and `app/doc_corpus.py` now
  ships the `doc` half (`vectors`, `corpus_mode`, five identity keys, `doc_ok`) with
  `fmt_drift` printing `doc_vectors unknown` / "…doc side unknown" instead; §3.7's
  "nothing has rebuilt it since" is falsified by job #111, which has run — re-read
  2026-11-21T20:47Z, `--status` exits 0 at 17,485 rows, 4.32 days, 98.4% coverage,
  `build.json` present — and the 10-day figure is a reported comparison `entity_linker`
  forbids any caller to act on; §1's pool omitted the deployed episodic floor (head 18
  + a `sessions` floor of 1, which `tests/conftest.py` pins off so a dev box cannot
  score it); and §1 still claimed the prefetch block keeps the vault's architecture
  list, which #4392 refuted on 2026-11-20 (`fa5298e0`) — prefetch splices
  `architecture_collections()` now. Also corrected: two dead line citations
  (`workers/sources/automod_regression.py`, `agent_mcp/vault.py`), "all 14 collections" (the index names
  15), `architecture/recall-research-2026-09-24.md` cited as a live note when
  `4a6cdd54` deleted it, and §5's 81 carried without the 86 it grew to. Added: the
  episodic kill switch beside the other three, `observed_ranker` (#3675) and
  `episodic_floor_leg` (#4404) as artifact fields, `app/doc_corpus.py` and
  `tests/test_episodic_recall.py` in §6. Re-measured true and left alone: the
  pool-derived-from-canvas derivation (`canvas_rows()` 128/4 = 32), `note_ranker`'s two
  call sites and `qmd_query` as the one door that folds every reply's `meta`
  (`backlog_similar` and `memory.py` build their own payloads — the doc never claimed
  otherwise), `PREFETCH_BUDGET_MS = 300`, `topics_merge: off` in both code and config,
  `N_RESAMPLES = 10000`, the fork's `buildFTS5Query`/`searchFTSAcross`/`lexMode`,
  `QMD_FUSION_DEPTH`, `djev.enabled: true`, and `vecIndex.vectors` 97,398. Filed
  nothing: every mechanism the doc names resolves in the tree, and each of the
  mechanism-level doubts raised on the way (a shape that follows the engine's liveness,
  an index served past its bound) is already instrumented (`observed_ranker`) or is a
  shipped ruling rather than drift.
