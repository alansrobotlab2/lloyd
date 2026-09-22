---
title: Vault recall — the retrieval pipeline and what was measured
status: implemented
date: 2026-09-22
---

# Vault recall — the retrieval pipeline and what was measured

`vault_recall` (`agent_mcp/vault.py::_vault_recall`) is how Lloyd finds notes. It
runs four legs in parallel — documents from the qmd daemon, a grep over Lloyd's
code, an entity-graph lookup, and entity facts — and hands back the top documents
plus facts. This doc is about the **document leg**: how its candidate pool is
built, who orders it, and every change to it that was measured on 2026-09-21,
kept or rejected. CLAUDE.md carries a line per decision; this is the long version.

## 1. The pipeline today

```
query ──► qmd /query (daemon :8181, fork ~/lloyd/qmd)
            lex leg:  BM25 over the 11 vault collections, terms ORed   (lexMode "or")
            vec leg:  Qwen3-Embedding-0.6B (embeddinggemma-300M until 2026-09-21)
            fusion:   global — one ranking across collections by score
            pool:     fused head 20 + floors of 2 for autonomy/architecture/skills
                      (≤32 rows; cross-encoder OFF)
       ──► djev (GPU 2, :8011) orders the ≤32 rows          (RECALL_RERANKER "djev")
            160-char candidates, one read, max_n 32, 4 s timeout
       ──► top `limit` documents
fallback: djev does not answer ──► the whole recall re-runs on the cross-encoder
          path (global 40 + floors 5, qmd rerank on), counted and announced by
          app/qmd_health.py::note_ranker
```

- **One definition of the request.** `recall_doc_leg_shape()` is read by the doc
  leg and by `scripts/automod/evalpin.production_payload`, so the regression pin
  warms exactly what production sends.
- **Kill switches.** `RECALL_RERANKER = "qmd"` restores the cross-encoder path;
  `RECALL_LEX_MODE = "and"` restores the AND keyword leg (and sends no key);
  `RECALL_QMD_FUSION = "collection"` restores #504's 240-row per-collection request.
- **Paths arrive decoded.** qmd percent-encodes every path segment; `qmd_file()`
  decodes at `_qmd_post` (and again in the Mission Control memory search), so
  `people/Ali Behrouz.md` is citable. 18 indexed notes carry a space, `+`, `#`
  or `&`. Two callers open their own sockets and are not doors: the MC memory
  search reads `results` straight (`app/routers/memory.py:112-117`, it does apply
  `qmd_file`) and write-time backlog dedupe reads them raw
  (`agent_mcp/backlog_similar.py:176-190`), so neither's rerank fallback reaches
  `qmd_health`. See `architecture/qmd.md`.
- **Tests never reach the live djev.** `tests/conftest.py` pins the recall to
  `"qmd"`; `tests/test_recall_djev_ranker.py` opts in with djev stubbed.

## 2. How it is measured

`eval/run_eval.py` over `eval/vault_recall_queries.yaml`, the gold set, against a
**pinned qmd snapshot** (`scripts/automod/evalpin.PinnedCorpus` with its own
index name and port, so it never collides with the regression runner's pin), and
a per-arm wrapper that changes one thing at `_qmd_post` or a module constant.
**The pin is the arms that ask for it, not `run_eval` itself:** `PinnedCorpus` is
constructed in `workers/sources/automod_regression.py:803` and nowhere in
`eval/run_eval.py`, which reaches the daemon through
`agent_mcp/vault.py:70 QMD_DAEMON_URL` — so the nightly baseline scores the live
index, and the `corpus` block its artifact records (`eval/run_eval.py:101`) is the
KG half only: no document or vector count, so `corpus_ok` cannot go false however
the document corpus re-embeds, and `eval_trend_stats.corpus_diff` prints
"corpus identical" across a drift that moves every doc-side number (#1374).
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
  `rerank` shadow seam is on an `elif` behind `if ranker == "djev"`
  (`agent_mcp/vault.py:1889-1903`), so since #1336 it records nothing:
  `~/.local/state/lloyd-djev/shadow.jsonl` held 30 rerank rows on 09-20, 4 on
  09-21, none on 09-22, while the `dedupe` and `entity` seams kept flowing, and
  the same dispatch ignores a caller's `djev_rerank: true` (#1372). Volume, not
  an error, is how this failure shows.

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
it is a full re-embed — every chunk vector in `content_vectors`, all 14
collections — so read the size before costing it rather than quoting this line:
`GET :8181/health` → `vecIndex.vectors` read 38,796 on 2026-09-22, and
`architecture/qmd.md` deliberately pins no count at all. Cost the re-embed from
that number and from the measured chunks/s.

### 3.4 Floors, not depth (#1335, `77a30cbb`)

Global fusion outscores small collections wholesale. Floors of 5 (cross-encoder
path) or 2 (djev path) on autonomy, architecture and skills recovered most of the
hits a 40-row global pool lost; a wider pool barely helped.

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
| `~/lloyd/architecture` as a qmd collection | doc_recall −0.025 (−0.059 with the floor moved to it) | NOT a verdict: no gold label points into those docs, so the eval can only see them take pool slots — the current architecture docs are unreachable by recall today (backlog follow-up) |

## 5. What the gold repair did to the numbers

The same production configuration scores doc_hit ~0.52 on the old 87-query set
and ~0.69 on the repaired 81: most of the "misses" were labels that named the
wrong file. Absolute numbers re-base at the repair (`scripts/eval_trend_stats.py`
says so); paired comparisons made before it are still valid, just less powerful,
because a broken label misses in every arm.

## 6. Files

- `agent_mcp/vault.py` — `RECALL_*` constants, `recall_doc_leg_shape`,
  `_qmd_daemon_search`, `_qmd_post`, `qmd_file`, `_djev_rank_recall`, `_vault_recall`
- `app/djev.py` — `rank(chars=, samples=, max_n=)`
- `app/qmd_health.py` — rerank and ranker fallbacks, counted and announced
- `scripts/automod/evalpin.py` — `production_payload` mirrors the doc leg
- `qmd/src/store.ts` (fork) — `buildFTS5Query(query, mode)`, `searchFTSAcross`, `structuredSearch` options
- `eval/vault_recall_queries.yaml` — the gold set; `tests/test_eval_corpus_guard.py` guards it
- tests: `test_recall_first_stage.py`, `test_recall_djev_ranker.py`, `test_recall_global_fusion.py`, `test_doc_candidate_pool.py`

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
