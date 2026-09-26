# #1456: topic phrases merged into `vault_recall` — measured, deployed as "facts"

**Verdict: deploy the `facts` shape** (`vault_recall.topics_merge: facts` in
config.yaml; the code default stays `off`). It reproduces #1164's entity gain with
a tighter interval, touches no document, and costs +43 ms p50 / +346 ms p95. The
`full` shape (documents fused too) also gains doc_hit clear of zero, but at about
+1.05 s a recall, which puts the paired check's 1,600 ms average budget at its
edge, so it stays off and is recorded here for a person to decide.

## What was built

`agent_mcp/vault.py`: `_vault_recall` is now a thin wrapper over the old body
(`_vault_recall_base`). With the mode off it is the base call, call for call. With
a mode on, the focus extractor production already runs for prefetch
(`app.secondary_models._sync_secondary_focus_extraction`, on the primary since
`secondary_enabled: false`) drafts up to 3 topic phrases **beside** the raw recall
(one thread), then each phrase is recalled and the lists are fused by reciprocal
rank (k=60), cut to one recall's length — #1164's arm, as production code.

- `facts`: each phrase runs the fact leg only (`facts_only=True`: no qmd request,
  no djev read), and facts, graph facts and neighbours are fused. Documents are the
  raw query's exactly.
- `full`: each phrase is a whole recall; documents are fused too. Every phrase
  costs a qmd request and a djev read, and djev serves one request at a time.

The mode is read from config at call time and never from the tool's arguments;
the djev fallback recursion never drafts twice; a drafter failure costs the
merge, never the recall. Prefetch is untouched. `tests/test_recall_topics_merge.py`
pins all of it (16 tests), and `tests/conftest.py` pins the mode off for the rest
of the suite so no test drafts on the live primary.

## Setup

- Instrument: `eval/recall_topics_merge.py`. Pinned document corpus (the
  regression runner's `PinnedCorpus`, a VACUUM INTO of the live qmd index served
  on :8183 with production's daemon settings), fact tree and store pinned as copies
  (`~/lloyd-data/eval/pin-2026-09-25/`). Real `_vault_recall`, mode forced per arm.
  Nightly knobs (limit 20, expand_graph on, production seed width), nightly scorer
  (`eval.run_eval._score`) with the raw query's seeds on every arm.
- Arms per query, interleaved: `control` (off = production), `facts`, `full`,
  `control_b` (off again).
- **Quality pass** under djev replay anchored on `control`: every raw recall inside
  an arm got the very djev answer control got (86/86 replayed per arm; `full`'s 242
  topic recalls were fresh draws), so `control_b` is identical by construction and
  every delta below is the merge alone.
- **Latency pass**: same arms, replay off, primary.lock held exclusively.
- Paired percentile bootstrap (`eval/stats.py`), 2000 resamples, 95%.
- Holdout leg (#1412): the 27 reserved questions, aggregates only (the artifact
  carries no id and no per-query row).

## Results — dev (n=86; entity recall metrics over the 66 with expected entities)

| metric | control | facts Δ [95% CI] (W/L) | full Δ [95% CI] (W/L) |
|---|---|---|---|
| doc_hit_rate | 0.640 | 0.000 (0/0) | **+0.093 [+0.023, +0.163]** (9/1) |
| mrr_doc | 0.289 | 0.000 | +0.037 [−0.024, +0.099] (34/17) |
| ndcg10 | 0.354 | 0.000 | +0.032 [−0.020, +0.081] (31/16) |
| doc_recall_avg (n=79) | 0.572 | 0.000 | **+0.072 [+0.006, +0.141]** (15/4) |
| entity_hit_rate | 0.337 | +0.047 [0.000, +0.105] (5/1) | +0.035 [−0.012, +0.081] (4/1) |
| entity_recall_avg | 0.384 | **+0.073 [+0.015, +0.136]** (5/1) | **+0.058 [+0.009, +0.118]** (4/1) |
| fact_entity_recall_avg | 0.374 | **+0.083 [+0.030, +0.152]** (6/0) | **+0.068 [+0.015, +0.129]** (5/0) |

`control_b` moved nothing (0.000 on every metric, replay). The drafter returned
2.77 phrases per query on average, none empty.

## Results — holdout (n=27; entity recall n=18), aggregates only

| metric | facts Δ [95% CI] (W/L) | full Δ [95% CI] (W/L) |
|---|---|---|
| doc metrics | 0.000 | doc_hit +0.037, doc_recall +0.074 [0.000, +0.167] (3/0), MRR +0.025, NDCG +0.027 |
| entity_hit_rate | +0.074 [0.000, +0.185] (2/0) | +0.074 (2/0) |
| entity_recall_avg | +0.111 [0.000, +0.278] (2/0) | +0.111 (2/0) |
| fact_entity_recall_avg | +0.111 [0.000, +0.278] (2/0) | +0.111 (2/0) |

No holdout metric lost a single question in either arm, so `overfit_suspected`
(a dev gain past the dev floor with a holdout loss past the holdout floor) is
**false** on every armed metric. The holdout gains are not significant at n=18,
and point the same way as dev.

## Latency (ms per recall, latency pass, n=86)

| | mean | p50 | p95 |
|---|---|---|---|
| control (first call of the query, cold caches) | 551 | 486 | 849 |
| control_b (fourth call, warm) | 238 | 197 | 401 |
| facts | 341 | 259 | 665 |
| full | 1,285 | 1,321 | 1,616 |
| **facts − control_b (warm vs warm)** | **+103** | **+43** | **+346** |
| full − control_b | +1,047 | +1,080 | +1,305 |

The first call of each query pays qmd's and djev's cold caches, so
arm-minus-control reads negative for `facts`; the warm-to-warm difference against
`control_b` is the honest added cost, and it is what is quoted. The draft (~0.2 s
on the primary) overlaps the raw recall's ~0.5 s; what shows is the topic fact
legs and the tail of drafts that finish late.

**Budget.** The recall tool has no budget of its own; the paired check's
absolute ceiling (`LATENCY_BUDGET_MS["paired_check"]` = 1,600 ms average) is the
nearest one, and the nightly's is 4,800 ms. `facts` puts the average at
~550 + 103 ≈ 650 ms (p95 ≈ 1.2 s): it fits. `full` puts it at ~1.6 s — at the
paired-check ceiling — and djev's single sequence means its three topic reads
queue behind any other recall: not flipped.

## What deploying it changes

- Every `vault_recall` makes one primary call (thinking off, ≤100 tokens). When the
  primary is down the draft returns nothing and the recall is today's.
- The nightly (task #82) and the post-landing paired check both call
  `_vault_recall`, so the nightly's fact-entity baseline steps up by about +0.08
  on the first run after this lands, and both now need the primary. The step is
  this change, not drift. Greedy drafts can still differ between two runs under
  engine batching; the paired check's entity floors (resolution-governed at
  n=66) absorb a one-query flip, but a person reading a small entity wobble in a
  later check should know the drafter is in the loop.
- Needs `round restart` (aggregator and backend) to take effect.

Artifacts: `~/lloyd-data/eval/1456/run.{quality.dev,quality.holdout,latency.dev}.json`
(the holdout file carries aggregates only).

## Re-measured on top of #1486 (2026-09-25 evening) — verdict: OFF

#1486's entity seeding went live while this was measured, and it lifts the same
metric. Paired, pinned corpus + fresh fact pin (`~/lloyd-data/eval/pin-2026-09-25b`),
djev replayed (86/86 in every arm), arms interleaved A (seeding, merge off) / B
(seeding, `facts`) / A again (A/A moved 0.000):

- dev n=86 (entity metrics n=66): fact_entity_recall −0.030 [−0.083, +0.008], 1 win /
  3 losses; entity_recall +0.012 [−0.009, +0.046]; entity_hit 0.000; documents 0.000.
- holdout n=27 (entity n=18), aggregates: fact_entity_recall +0.083 [0.000, +0.222];
  no loss, overfit flag false — moot without a dev gain.
- latency (B vs warm A): +492 ms p50, +625 ms mean, +1,476 ms p95 — each topic recall
  now runs #1486's CPU query embedding.

The earlier +0.083 overlapped #1486 almost entirely. `topics_merge` ships `off`; code,
flag and tests stay. Artifacts: `~/lloyd-data/eval/1456-seeding/`.
