# `vault_search` on the recall document pipeline (2026-11-10)

**Adopted.** `vault_search.pipeline: recall`.

`vault_search` used to run its own document request:
- AND keywords, so every term had to match;
- qmd's cross-encoder to order the results;
- no djev.

Meanwhile the tool descriptions sent prose questions to `vault_search`, while
`vault_recall` ran the document leg that `architecture/retrieval.md` measured. `recall`
gives `vault_search` that same leg:
- OR keywords;
- djev ordering;
- collection floors;
- no entity facts.

## Measurement

Run with `eval/run_vault_search_eval.py --compare` (`--limit 10`) over the gold set's 79
doc-gold queries. Both pipelines hit the live qmd daemon and djev under `regression.lock`,
and each query was scored once per pipeline. The artifact is
`vault-search-compare-20261110.json`, kept in the session scratchpad and not in the tree.

| metric | legacy | recall | paired Δ (95% CI, 10k bootstrap) | W/L |
|---|---|---|---|---|
| doc_hit | 0.544 | 0.620 | +0.076 [−0.025, +0.177] | 11/5 |
| MRR | 0.230 | 0.354 | **+0.125 [+0.038, +0.214]** | 33/16 |
| nDCG@10 | 0.303 | 0.406 | **+0.104 [+0.027, +0.182]** | 33/20 |
| doc_recall | 0.414 | 0.460 | +0.046 [−0.044, +0.134] | 18/10 |
| latency p50 / p90 | 1006 / 1147 ms | **397 / 444 ms** | | |

The plan adopted `recall` only if doc_hit and MRR did not drop and latency improved. MRR
and nDCG gained with CIs clear of zero, doc_hit and recall rose with CIs spanning zero,
and both latency percentiles fell by more than half. All three conditions hold.

## Not measured

- **`consolidate_default` stays `true`.** It controls the extra primary-model summary
  written beside the hits. This eval scores ranking and cannot judge the summary, so the
  setting keeps its old value. The tool schema states the default.
- **If djev is down**, `recall` falls back to the cross-encoder path the same way
  `vault_recall` does (`app/qmd_health.py` counts and announces it).
