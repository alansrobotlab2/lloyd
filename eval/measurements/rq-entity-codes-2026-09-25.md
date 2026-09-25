# Residual-quantized entity codes as a seed prior (#1497) — 2026-09-25

**Verdict: rejected.** The codes are not a folder copy, but they are a lossy copy
of the cosine they quantize: at equal selectivity plain cosine carries three times
the KG-edge structure, and as a prior on the semantic seeds the prefix depth only
costs anchors. Nothing wired; no code change.

## Setup

- Vectors: the #1486 entity index — Qwen3-Embedding-0.6B over
  `name (kind): top-3 facts` for all 11,990 store entities (store copy of
  2026-09-25 15:48). No new encoder, as the item asked.
- RQ-KMeans, 3 levels × 32 clusters (sklearn KMeans, `n_init=4`, seeds 0/1/2),
  residuals carried level to level.
- Folder label per entity: the majority two-component vault folder over its facts'
  `source_doc` (11,943 of 11,990 have one; 156 folders).
- Relational check: 35,848 active KG edges between embedded entities, against
  200,000 random pairs.
- Artifacts and scripts: `~/lloyd-data/eval/1497/` (`gate-f3.json`,
  `step3-offline.json`, `rq_gate.py`, `rq_step3.py`).

## Step 2 — the kill gate

| grouping | occupied families | NMI vs folder | homogeneity | KG-edge lift (same group, edge vs random) |
|---|---|---|---|---|
| folder | 156 | — | — | 3.8 |
| prefix depth 1 | 32 | 0.242 | 0.259 | 6.3 |
| prefix depth 2 | 949 | 0.308 | 0.495 | 21.6 |
| prefix depth 3 | 7,660 | 0.444 | 0.877 | 116.8 |
| **cosine ≥ threshold at depth-2's selectivity** | — | — | — | **62.2** |
| cosine at depth-1's selectivity | — | — | — | 11.2 |

- **Not a folder copy.** Depth-2 NMI with folders is 0.31; an entity's depth-2
  family averages ~20 members (entity-weighted) against ~1,470 for its folder.
  The pre-registered rule as written ("more families AND a smaller *median*
  family") read `finer: false` — the folder median is 8 because 156 folders are
  mostly tiny while `knowledge/youtube` alone holds 3,552 entities — so that clause
  is a flaw in the rule, not a finding, and the gate was judged on substance.
- **But not more than cosine.** Codes carry relational structure (lift 21.6 vs
  3.8 for folders), yet a plain cosine threshold admitting the same fraction of
  random pairs carries 62.2. Prefix sharing is quantized cosine and strictly
  weaker than the similarity it was quantized from — the item's own open question
  ("the prefix prior must add something beyond nearby in the same space"),
  answered no.
- **Reverse audit.** 20 sampled depth-2 families have median folder purity 0.40
  and near-zero edge density; they read as mixed bags (`MEMORY.md`, `MemoryHigh`,
  `cgroup OOM`, `CRDT`; `Hootsuite`, `Helen Powell`, `Project Hail Mary`), not as
  misfile signals. No misfile candidate worth a human's time came out of it.

## Step 3, offline and scoped (the retrieval claim itself)

Query vectors encoded through the same codebooks; semantic candidates scored
`cos + α × shared_prefix_depth(query, entity)`, top 3 unioned after the lexical
top 10 exactly as #1486 ships, anchoring by the eval's `anchorless_queries`:

| α | 0 (= #1486) | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| anchorless of 66 | **19** | 20 | 21 | 20 | 23 |

Every non-zero prior loses anchors. A paired pinned run was not spent on a
variant that is worse offline on the measure the pinned run's entity metrics are
built from.

## Close

Null at vault scale (thousands of entities, not billions): the family structure
DoorDash reports does exist here, but the flat cosine it is derived from already
carries more of it, and #1486's semantic seeds use that cosine directly.
