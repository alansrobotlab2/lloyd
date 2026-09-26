# #1493: Qwen3-Embedding-4B side index — measured on a sub-corpus, rejected

**Verdict: REJECT.** On an identical sub-corpus, the 4B embedder ties the
production 0.6B on doc_hit (5 wins, 5 losses) and trails it slightly on
ordering. It costs 2.5× the re-embed time, 2× the query-embed time, and about
4.3 GB more VRAM on the card that already holds TTS and production qmd. djev
ranks the pool, so the embedder only decides which ~32 rows reach djev, and the
larger embedder does not change that set for the better.

## Why a sub-corpus, and what it is

A full re-embed at the 4B's measured rate (8.5 chunks/s on the shared 3090)
would take about 57 min for the 29,261 chunks of the searched collections, and
about 68 min for all 34,566. That is GPU 0 at ~94% the whole time, beside TTS
and production qmd, with regression.lock held against five other agents.
`eval/embed_side_index.py prepare` instead builds one sub-corpus that both arms
share exactly:

- Every document satisfying a narrow dev-set `expect_docs` label (≤50 matches:
  112 of 115 labels, 241 documents). The three family labels (`backlog/`,
  `robot`, `robotics`) are served by the sample.
- A seeded 25% sample (seed 1493) of the rest of the 11 searched collections.
- Everything else, including the non-searched collections, is deactivated in
  the copy.

The result is 1,564 documents and 8,909 chunks, about 28% of the searched
corpus. The copies are VACUUM INTO of the live index. `index.yml` and
`evalpin.yml` are untouched: each arm has its own `~/.config/qmd/<name>.yml`
naming its embed model. Both copies were re-embedded from scratch with
`qmd --index <name> embed -f`, so the only difference between them is the
model. The model is `Qwen3-Embedding-4B-Q8_0.gguf` (4.28 GB, now in
`~/.cache/qmd/models/`), and the queries use the same Qwen3 instruct prefix.

| | 0.6B (production) | 4B |
|---|---|---|
| re-embed, 8,909 chunks | 426 s (**20.9 chunks/s**) | 1,046 s (**8.5 chunks/s**) |
| GPU 0 during the build | — | 21.1 of 24.6 GB used, 94% util |

## Eval

Both side indexes were served at once with production's daemon settings (:8184
0.6B, :8185 4B). The runs used the real `_vault_recall`, djev ranking, the
pinned fact tree and store, the 86-query dev set and the nightly scorer. Arms
per query: `control` (0.6B), `candidate` (4B), `control_b` (0.6B again, the
noise), with fresh djev reads. The dev labels chose the sub-corpus, and no
holdout id was read.

| metric | 0.6B | 4B Δ [95% CI] (W/L) | noise (control_b Δ) |
|---|---|---|---|
| doc_hit_rate | 0.744 | 0.000 [−0.070, +0.070] (5/5) | +0.012 |
| mrr_doc | 0.401 | −0.012 [−0.071, +0.046] (22/24) | +0.004 |
| ndcg10 | 0.447 | −0.014 [−0.077, +0.050] (26/30) | +0.012 |
| doc_recall_avg (n=79) | 0.697 | −0.001 [−0.066, +0.070] (7/11) | +0.006 |

Entity metrics are identical, because the embedder does not reach the fact leg.
The sub-corpus is easier than production: doc_hit is 0.744 here against 0.64 on
the full index. Both arms get that ease equally, so the paired delta is the
comparison that counts, and it is zero.

**Latency, per qmd request (the vec leg's own phases):**

| | query embed p50 / p95 | vec search p50 | whole request p50 |
|---|---|---|---|
| 0.6B (warm repeat) | 10 / 11 ms | 14 ms | 44 ms |
| 4B | 15 / 26 ms | 32 ms | 70 ms |

The vector search is 2.5× wider per row (2,560 dims against 1,024). End to end,
the recall is dominated by djev (~0.3–0.5 s), so the 4B's extra ~25 ms is
invisible. It is still slower, though, with no quality gained.

## What would change the verdict

Only a larger gold set, P0 (#1480), showing a pool-membership gain that this
86-query set cannot see. The sub-corpus caveat cuts both ways: with 72% fewer
distractors, a better embedder has less room to show separation. If the fourth
3090 (#1495) arrives, a full-corpus re-run is one command per step, with no
contention on GPU 0:
`embed_side_index.py prepare --sample 1.0 && … embed --name sub06 && … embed --name sub4b && … eval`.

Artifacts: `~/lloyd-data/eval/1493/` (prepare.json, embed-*.json, eval.json).
