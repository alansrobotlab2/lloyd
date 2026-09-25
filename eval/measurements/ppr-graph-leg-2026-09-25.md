# Graph leg in recall: PPR over the KG + entity→doc mentions (#1492) — 2026-09-25, scoped

**Verdict: rejected (time-boxed, scoped measurement).** Personalized PageRank over
the existing graph recovers almost none of the multi-hop/hard queries' document
misses (0 of 9 from today's seeds, 1 of 9 from #1486's), costs ~0.1 s, and fused
at equal weight it takes documents out of the top 10. The entity leg's ceiling was
its seeds (#1486 moved entity_hit +0.163 with no graph at all), not propagation.
The full build the item proposes (entity→chunk mention edges over ~39k qmd chunks,
a new edge type, pool fusion in `vault.py`) was not built.

## What was measured

- **Graph**: the 2026-09-25 store copy (`~/lloyd-data/eval/1486/kg-base.sqlite`):
  35,848 active entity–entity edges weighted `EDGE_TYPE_WEIGHTS × confidence`,
  plus entity→document edges from `facts_idx.source_doc` (a fact extracted from a
  document is a mention of its entity there) — HippoRAG 2's entity→passage edges
  at document granularity, built with no new indexing. 14,978 nodes, 69,518 edges.
- **PPR**: networkx, α 0.5, personalization on the query's recorded seeds; top 10
  document nodes and top 10 non-seed entity nodes.
- **Queries**: the `multi-hop` (19) and `hard` (12) categories only, n = 31, as the
  item specifies. Base documents: the top 10 of the #1486 pinned arm A record
  (lexical seeds) and arm D record (lexical + alias + semantic seeds), same pin.
- **Fusion**: RRF k=60 of base top 10 with PPR top 10, equal weight and PPR
  weights 0.5/0.3/0.15; scored with the eval's own doc-match rule at 10.
- Scripts and rows: `~/lloyd-data/eval/1492/` (`ppr_scoped.py`, `ppr-scoped.json`).

## Results (paired bootstrap, n = 31)

| seeds | base doc_hit | PPR docs alone | base misses PPR recovers | equal-RRF doc_hit Δ | weighted RRF (≤0.5) doc_hit Δ | weighted MRR Δ |
|---|---|---|---|---|---|---|
| lexical (arm A) | 0.710 | 0.097 | **0 / 9** | −0.129 [−0.258, −0.032] | 0 | +0.008 [−0.048, +0.073] |
| + alias + semantic (arm D) | 0.710 | 0.226 | **1 / 9** | −0.097 [−0.226, +0.032] | 0 | +0.011 [−0.086, +0.118] |

Entity side (n = 28 with gold entities): seeds + PPR's top 5 entities against
seeds + the 1-hop neighbours production's eval already expands (same width):
0.500 vs 0.536 (Δ −0.036 [−0.107, 0.0]) from lexical seeds, 0.679 vs 0.679 from
#1486's seeds. PPR's top 10 reads higher (0.607 / 0.750) only by being twice as wide.

PPR costs 0.11 s median per query on this graph (networkx, CPU).

## Why stop here

The item's gate was "judge on multi-hop and hard only; a whole-set gain is not
expected", and depends on P4 (#1486) for seeds. With #1486's seeds in hand the
graph walk still finds one of nine missing documents, and every fusion weight that
does not hurt doc_hit leaves it unchanged. The chunk-level build could raise the
PPR-alone hit rate (documents are coarse targets), but it cannot reach documents
the mention edges do not point at, and the doc-level walk that already has those
edges recovers 1 of 9. Not worth a 39k-chunk index on this evidence; reopen if a
multi-hop gold set larger than 31 shows a different ceiling.
