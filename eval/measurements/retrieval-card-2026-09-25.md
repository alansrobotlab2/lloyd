# #1495: does retrieval need a fourth 3090? What GPU-0 contention costs today

**Verdict: needs-human (hardware is Alan's).** The measurement does not make the
case for a retrieval card. Since djev took over ranking (#1336), production qmd's
only GPU work is a ~10 ms query embedding. The eval pin now shares GPU 0 for 3.2%
of wall time, down from 42% on 2026-09-21. When a pin does overlap a production
request, that request gets +7 ms slower at the median and about +94 ms slower at
p95, which is roughly 1 ms per recall averaged over the day. The three models
proposed for the card were each measured in this sweep, and none earned a place
on it (below). **Recommendation: do not install the card *for retrieval*.** If it
goes in for another reason, move the eval pin onto it, since that is free.

## 1. Contention on GPU 0, from the logs (no boot, no load added)

`eval/qmd_contention.py` joins production's daemon log (every `/query` with its
phases) to the regression runner's pin windows. The span covers
2026-09-22 20:03Z to 2026-09-25 22:47Z, 74.7 h, with 70 pinned checks (median
121 s each).

| production recall doc-leg request (lex+vec) | n | total p50 | total p95 | embed p50 | embed p95 |
|---|---|---|---|---|---|
| no pin running | 13,413 | 51 ms | 199 ms | 7 ms | 16 ms |
| pin running | 552 | 59 ms | 292 ms | 8 ms | 159 ms |
| Δ (unpaired bootstrap, 1000×) | | **+7 ms [+5, +10]** | **+94 ms [+28, +120]** | | |

- **The pin runs 3.2% of the time.** On 2026-09-21 it was 42% of the time, with a
  1.8× rerank slowdown, measured while the cross-encoder still ranked every recall
  and pins took ~5 s per recall.
- The cross-encoder path is still used by the recall's fallback, by
  `vault_search` and by backlog dedupe. It saw 507 requests in the span, and only
  13 of those overlapped a pin (p50 814 ms against 871 ms without).
- The expected cost per production recall is ≈ 0.032 × ~21 ms mean ≈ **<1 ms**.
  A whole recall takes ~500 ms, and djev on GPU 2 accounts for most of it.
- Caveat: pin windows can coincide with bursts of production traffic, which is
  a confound in the unhelpful direction. The TTS server on the same card logs
  no request stream to join, so voice latency under contention is not measured
  here.

## 2. A heavier tenant than the pin, measured live

The #1493 4B re-embed ran GPU 0 at 94% utilisation, with 21.1 of 24.6 GB in use,
for 17 minutes (2026-09-25 23:50–00:10Z). Production's recall requests in that
window: n=396, p50 85 ms, p95 182 ms, embed p50 8 ms, embed p95 25 ms. In the
hour before it: p50 105 ms, p95 163 ms, embed p95 15 ms. **A saturating job on
GPU 0 cost production qmd nothing measurable.** Its query embedding is too small
to queue behind anything for long.

## 3. What the card would hold, as measured in this sweep

| candidate tenant | item | result |
|---|---|---|
| Qwen3-Embedding-4B (~4.3 GB Q8) | #1493 | **rejected**: tied the 0.6B on doc_hit (5/5), MRR −0.012 [−0.071, +0.046]. It fits beside production on GPU 0 anyway (built there today). |
| Provence / learned pruner (≤0.4 GB) | #1491 | **not justified**: the lexical version moved MRR +0.03 on two pins, inside the interval. |
| Qwen3-Reranker-4B | — | not proposed on measurement: djev beat the 0.6B cross-encoder outright (#1336), and the cross-encoder is only the fallback now. |
| the eval pin | #1412/#1495 | 3.2% of wall time at <1 ms average cost (§1). Moving it is tidy, not needed. |
| contextual chunk prefixes | #1494 | no GPU at query time; the one-off generation is primary/djev time. |

GPU 0 has ~8.5 GB free at idle beside TTS (7.8 GB) and production qmd (4.7 GB),
which is enough for any one of the above.

## 4. When to revisit

- If P0 (#1480) shows an embedder or reranker gain that the 86-query set could
  not resolve, the full-corpus 4B build becomes worth running. It takes ~57 min
  on GPU 0, or the same on a new card with no contention.
- If voice latency (TTS on GPU 0) is ever shown to suffer during pins or
  re-embeds, that is the card's real case. It is not measured here.
- The djev slot on GPU 2 and its candidate replacements (`f8f154db`, decider
  bake-off) are a separate question that this item does not cover.

Artifacts: `~/lloyd-data/eval/1495/contention.json`. Re-run with
`.venvs/lloyd/bin/python eval/qmd_contention.py`.
