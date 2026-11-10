# Which architecture docs recall reaches — `vault_search.architecture` (2026-11-10)

**Adopted: `both`.** The recall asks qmd for the vault's `architecture` collection
(`~/obsidian/architecture`, April-era copies and retired docs) **and** the new
`lloyd-architecture` collection (this checkout's `architecture/*.md`). This is #1662's
option (a): Alan's ruling read "file (a) as its own proposal, deployed only if it measurably
helps Lloyd's code-work recall". It is also the two-change option that
`~/obsidian/eval/measurements/lloyd-architecture-ab-2026-10-20.md` (#3076) found to be the
only one with any effect: a collection in qmd's config that is missing from
`VAULT_SEGMENTS` is never asked for.

## Setup

- **Index.** `lloyd-architecture` was added to the live `~/.config/qmd/index.yml` and to
  the template synced from it (`agent-services/conf/qmd-index.yml`), with pattern `*.md`
  and top-level docs only. `qmd update` and `qmd embed` ran under `regression.lock`:
  1,180 chunks from 41 docs. The embed ran on GPU 4 (`CUDA_VISIBLE_DEVICES=4`,
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`) because GPU 0 had ~2 GB free and the watcher's own embed
  had failed there ("Failed to create any embedding context").
- **Arms.** One process per arm. `vault.apply_architecture_collection(mode)` ran before
  `eval/run_eval.py`. The script is `arch_ab.py <mode> <queries> <label>`, and it was run
  against the production daemon and a read-only snapshot of the KG store.
  - `vault`: the 11 segments as before.
  - `repo`: `architecture` swapped for `lloyd-architecture`, which takes over its floors.
  - `both`: `lloyd-architecture` appended to the 11, with no floor of its own.
- **Runs.** Two runs per arm, interleaved (vault, repo, vault, repo, both, both), on the
  81-query gold set plus the five #1354 queries (86 in all; 79 carry doc gold). Artifacts:
  `arch-{vault,repo,both}-*-20261110-*.json`.
- **Statistics.** Each query's two runs were averaged; deltas are paired over query ids
  with a 10,000-rep bootstrap 95% CI.

## Results

Paired against `vault`. "Original" is the 74 doc-gold queries that predate the five.

| arm | subset | n | doc_hit | doc_recall | mrr_doc | ndcg10 |
|---|---|---|---|---|---|---|
| vault | all | 79 | 0.684 | 0.541 | 0.301 | 0.335 |
| both | all | 79 | 0.696 (Δ +0.013 [−0.051, +0.076]) | 0.562 (Δ +0.021 [−0.040, +0.084]) | 0.362 (**Δ +0.061 [+0.009, +0.123]**) | 0.405 (**Δ +0.070 [+0.018, +0.131]**) |
| both | original | 74 | 0.689 (Δ −0.041 [−0.095, +0.000], W/L 0/3) | 0.545 (Δ −0.032 [−0.076, +0.003]) | 0.332 (Δ +0.011 [−0.019, +0.047]) | 0.379 (Δ +0.021 [−0.010, +0.054]) |
| both | the five | 5 | 0.800 (from 0.000) | 0.800 | 0.800 | 0.800 |
| repo | all | 79 | 0.671 (Δ −0.013) | 0.551 | 0.332 (Δ +0.031 [−0.034, +0.099]) | 0.374 |
| repo | original | 74 | 0.662 (Δ −0.068 [−0.135, +0.000], W/L 1/6) | 0.535 | 0.300 (Δ −0.021) | 0.345 |

Mean latency per query: vault 884 ms, repo 905 ms, both 754 ms. Latency was not a
deciding factor; the daemon was shared with production traffic.

## What it costs

`both` lost the doc hit on three queries, the same three in both runs. In each case the
gold doc had sat low in the vault arm (ranks 5, 13 and 14), and the new collection's
candidates pushed it out of the returned pool:

- `entity-guard-to-alias-table`: ranked 5 in the vault arm.
- `git-signals-in-async-code`: ranked 13.
- `primary-ram-floor-changed`: ranked 14.

It gained none on the original 74. MRR and nDCG on those 74 still rose (+0.011 and +0.021,
CIs spanning zero), because more gold docs moved up than fell out.

`repo` was rejected. Dropping the vault's `architecture` copies cost six original queries
their hit, and the gain on the five did not make up for it.

## Why `both` anyway

The five checkout-doc queries go from unanswerable (0.0 by construction: no collection
the recall asked for held them) to rank 1 in four of five. Those are the docs that
describe the code as it is now, and the vault copies are what Lloyd was finding
instead. The cost is three low-ranked hits on unrelated topics; the CI on that cost
touches zero. The prefetch `<context>` block is deliberately unchanged in `both` mode,
because its 300 ms budget and five-hit cap are not where this was measured.

## Follow-ups

- **Watch the three losers on the nightly eval.** If they stay lost, the candidate quota
  per collection (`RECALL_COLLECTION_FLOOR`) is the knob to look at; it was not changed
  here.
- **Re-base point.** The gold set is 81 → 86 from the first night after landing. This is
  booked in `eval/vault_recall_queries.yaml`'s header and in
  `scripts/eval_trend_stats.py`'s chronology (the seventh point).
- **Watcher.** `agent-services/scripts/qmd-watcher.sh` now watches
  `~/lloyd/architecture`, so a doc edit re-indexes. The watcher's embed runs on GPU 0 and
  fails while GPU 0 is full, which is a pre-existing problem that this change does not
  fix.
