# #1164 self-question retrieval arm: measured, rejected (2026-09-24)

**Verdict: REJECT.** Retrieving against questions the LLM drafts adds about
1 s per recall even with the question recalls run in parallel (about 2 s run
one after another). The quality gains are small, and no gain on the headline
metric beats a paired interval. The expansion production already ships (the
focus extractor's topic phrases) did better on every entity metric in the
same run, at lower cost.

## Setup

- Instrument: `eval/self_question.py` (tests: `tests/test_self_question_arm.py`).
  Offline only, and it changes no production path.
- Gold set: `eval/vault_recall_queries.yaml`, all **86** queries, all scored.
  Scored with `eval.run_eval._score`, the nightly's own scorer, at the
  nightly's knobs (limit 20, expand_graph on, production seed width). Each arm
  keeps the raw query's seeds, so an arm gets credit only for what it
  *retrieved*.
- Corpus: a snapshot of `kg.sqlite` and the facts tree taken at 13:35 and
  pinned through `LLOYD_KG_DB` / `LLOYD_FACTS_ROOT`. Live qmd daemon and djev
  (production ranker).
- Arms, interleaved per query in this order:
  1. `control`: the raw query.
  2. `self_question`: the primary (the `secondary` alias, since
     `secondary_enabled: false`) drafts up to 3 questions, thinking off,
     greedy. There is one recall per question, fused with the raw list by
     reciprocal-rank fusion (RRF, k=60).
  3. `topics`: the same fusion over `_sync_secondary_focus_extraction`
     topic phrases, the expansion prefetch already uses.
  4. `control_b`: the raw query again, which gives the run's noise floor.
- Fusion is **budget-equal**: each fused list is cut to the length of one
  recall (20 docs, 10 facts). Without the cut, the union would win recall on
  length alone. The cost is that the fused list displaces 7.35 raw docs per
  query on average. That is recorded in the artifact and is why the item's
  "never lose a raw hit" clause was not adopted as written.
- Paired percentile bootstrap (`eval/stats.py`), 2000 resamples, 95 % CI.
- Held under exclusive `primary.lock` and `retrieval.lock`; it ran 18 minutes.

## Results (n=86, delta = arm − control, paired)

| metric | control | self_question Δ [95% CI] (W/L) | topics Δ [95% CI] (W/L) | noise floor (control_b Δ) |
|---|---|---|---|---|
| doc_hit_rate | 0.651 | +0.035 [−0.035, +0.105] (6/3) | +0.035 [−0.023, +0.093] (5/2) | 0.000 |
| mrr_doc | 0.332 | +0.026 [−0.051, +0.105] | +0.015 [−0.064, +0.092] | +0.005 |
| ndcg10 | 0.364 | +0.043 [−0.025, +0.111] | +0.032 [−0.036, +0.097] | +0.006 |
| doc_recall_avg | 0.564 | +0.068 [0.000, +0.140] (16/4) | +0.055 [−0.010, +0.123] | −0.006 |
| entity_hit_rate | 0.337 | +0.023 [0.000, +0.058] (2/0) | +0.035 [−0.012, +0.081] | 0.000 |
| entity_recall_avg | 0.384 | +0.023 [0.000, +0.061] | **+0.058 [+0.009, +0.118]** (4/1) | 0.000 |
| fact_entity_recall_avg | 0.374 | +0.020 [−0.020, +0.066] (3/1) | **+0.068 [+0.015, +0.129]** (5/0) | 0.000 |

The noise floor is small. djev reorders the ranking metrics by ≤0.006
between two identical recalls and moves nothing on the entity metrics, so
the deltas above are not noise from the ranker.

The item's own bar was fact_entity_recall_avg +0.10 with doc_hit not
worse. Self-question reached **+0.020**, with a CI that includes zero. Its
only interval that excludes zero is doc_recall_avg, and that lower bound sits
exactly at 0.000 (p=0.049), which is marginal.

## Latency (per query, ms)

| | mean | p50 | p95 |
|---|---|---|---|
| control recall | 577 | 506 | 841 |
| self_question: draft (primary) | 436 | 438 | 575 |
| self_question: added, sequential | 1999 | 1932 | 2396 |
| self_question: added, parallel lower bound | 1058 | 977 | 1412 |
| topics: draft | 189 | 190 | 252 |
| topics: added, parallel lower bound | 709 | 695 | 846 |

Judged against its own run context (`paired_check`, 1600 ms), the arm runs at
2576 ms per query, 1.61 times that budget. The prefetch budget is 300 ms,
which the draft call alone (436 ms) already exceeds.

## Reading

- This test is less favourable to the idea than the source setting. The gold
  queries carry no conversation context, while Tolan drafts from a live
  conversation. That is the fair caveat. However, prefetch sees the same
  single message this eval sees, and the drafts were on-topic (for example
  "What is the title and description of backlog item 363?"). They mostly
  restate the query, so their recalls land on what the raw query already
  found.
- The drafted questions lose to terse topic phrases on entity recall. With
  OR-lex and Qwen3 embeddings, short noun phrases reach entity facts better
  than full questions do.
- The topics finding is a separate result, not part of this item. RRF-fusing
  per-topic recalls, rather than prepending topics to the query as prefetch
  does today, measured fact_entity_recall +0.068 [+0.015, +0.129] (5 wins,
  0 losses). It costs ≈0.7 s parallel, so it cannot fit a 300 ms prefetch
  either. It would only be worth pursuing as a straggler arm or on the
  nightly/recall tool path. That decision belongs to a follow-up proposal,
  not this item.

Artifact: `eval/measurements/self-question-2026-09-24.json` (per-query
drafts, scores and latencies for every arm).
