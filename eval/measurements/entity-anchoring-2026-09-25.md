# Entity anchoring: alias surfaces, alias regeneration, semantic seeds (#1486) — 2026-09-25

**Verdict: deploy.** `retrieval.entity_seeding` on (alias surfaces + 3 semantic
seeds). Every entity metric moves clear of its paired 95% interval, no document
metric moves at all, no query loses a hit, and the cost is +71 ms mean recall
latency. The alias-regeneration script is shipped but measured **no gain** on the
gold set; applying it to the live store is optional and owed to a human.

## What was built

- **Alias surfaces as match candidates** (`retrieval.entity_seeding.alias_surfaces`).
  `extract_entities_from_query`'s alias fold only ran on a name that was itself an
  entity directory, so an alias whose surface has no directory — `Relationship
  Graph` → `Knowledge Graph`, `The Graph` → `Knowledge Graph` — could never be
  matched. With the switch on such a surface is matched like a name and scores its
  canonical (never itself).
- **Semantic seeds** (`retrieval.entity_seeding.semantic`, `app/entity_linker.py`).
  Qwen3-Embedding-0.6B vectors of `name (kind): top-3 facts by confidence` for
  every store entity (the store's `definition` column is empty on all 11,990 rows,
  so facts stand in for one); the query is embedded with the Qwen3 query
  instruction and the top `k=3` rankable entities not already in the lexical head
  are **appended after the lexical top `RECALL_SEED_TOP_K`** — a union, never a
  substitution: the first 10 seeds are byte-identical to HEAD. `recall_seeds` is
  the one definition `vault_recall` and `eval/run_eval.py` both use.
- **The query embedder is plain torch** (`app/qwen3_embed.py`): the production venv
  has torch + `tokenizers` but not `transformers`/`safetensors`, and qmd exposes no
  embedding endpoint. It is the Qwen3 forward pass written out over weights read
  from `model.safetensors`; cosine against transformers' implementation **1.00000**
  on 10 of 10 queries. CPU, ~0.2 s a query, 2.4 GB fp32 resident in the aggregator
  once loaded (only loaded when the switch is on and the index exists).
- **The document leg starts before the seeds** when semantic seeding is on, so the
  embedding overlaps qmd + djev instead of preceding them.
- **Alias regeneration** (`scripts/memory/regenerate_aliases.py`): rule aliases an
  entity's own name spells — parenthetical acronym (`Causal Agent Replay (CAR)` →
  `CAR`), parenthetical strip (never a citation year), `-`/`_` → space. Written only
  through `app.kg_store`'s alias writer with `origin=regen:<rule>` and `created_at`;
  never a surface that is itself an entity, already an alias, generic, numeric, or
  produced by two canonicals. On a copy of the 2026-09-25 store: **2,100 rows**
  (2,054 punct, 33 paren-strip, 13 paren-acronym), 1 collision dropped, 169 → 2,269
  alias rows.

## How it was measured

- Frozen inputs: a `sqlite .backup` copy of the live store and a `cp -a` of the
  fact tree taken 2026-09-25 15:48, `LLOYD_KG_DB`/`LLOYD_FACTS_ROOT` for every arm;
  the regenerated store is a second copy. Raw artifacts: `~/lloyd-data/eval/1486/`.
- `eval/run_eval.py` (the nightly harness, 86 queries, production defaults) against
  ONE `PinnedCorpus` qmd snapshot per run with ONE djev replay file anchored on arm
  A, under `regression.lock`. Arms differ only by store copy + config overlay.
- Paired bootstrap from `eval/stats.py` (10,000 resamples, seed 20260921).
- Anchoring is reported both ways (#1502): the eval's canonical
  `anchorless_query_count` (a seed and a gold entity contain one another) and a
  strict count (a gold entity IS a seed through `store().resolve`).

## Results (run1, arm A = HEAD; A2 = A repeated at the end, identical on every metric)

| arm | what | anchorless (eval) | strict | entity_hit | entity_recall | fact_entity_recall | doc metrics |
|---|---|---|---|---|---|---|---|
| A | HEAD | 25/66 | 38/66 | 0.337 | 0.384 | 0.374 | — |
| B | alias surfaces | 22 | 34 | +0.047 [0.012, 0.093] | +0.045 [0.008, 0.099] | +0.045 [0.008, 0.099] | Δ 0 |
| C | semantic k=3 | 19 | 31 | +0.116 [0.058, 0.186] | +0.149 [0.071, 0.235] | +0.126 [0.051, 0.212] | Δ 0 |
| **D** | **regen + alias + semantic** | **16** | **27** | **+0.163 [0.093, 0.244]** → 0.500 | **+0.194 [0.106, 0.288]** → 0.579 | **+0.172 [0.088, 0.265]** → 0.545 | Δ 0 |
| E | regen alone | 25 | — | 0 | 0 | 0 | Δ 0 |

n = 86 queries (entity_hit), 66 with gold entities (the recall metrics). doc_hit
0.663, MRR 0.304, NDCG@10 0.352 in every arm, exactly: the doc leg never reads the
seeds (graph re-rank is off), so none can move.

Controls (run2, own pin + replay file): **F = HEAD at seed width 13** (the same
number of seeds D carries, all lexical) moves **nothing** — every metric Δ 0, 25
anchorless — so the gain is not width. **G = regen + alias surfaces** equals B on
every metric: the 2,100 rule aliases anchor no gold query the existing 169 do not.

D vs A per query: 14 queries gain an entity hit (recent 4, fuzzy 3, single 3,
multi-hop 2, hard 2), **0 lose one**; fact_entity_recall rises on 13, falls on 0.
Audited: 11 of the 14 name the gold entity exactly (`Knowledge Graph` ×4,
`Nightly Retrieval Eval` ×4, `Model Routing`, `Memory Capture`,
`AmbientPrefetchEntry`, `Autonomy Data Pipeline`); three pass on the eval's
lenient substring rule through a close sibling (`Qwen3-TTS-streaming` for
`Qwen3-TTS`, `GPU hardware` for `GPU`, `Fact Store Evolution` for `Fact Store`),
which is why the strict count is reported beside it (38 → 27).

Offline, through the real `recall_seeds` path (no qmd): lexical at widths 13/15/20
anchors nothing new; semantic k=1/3/5 appended → 21/19/18 anchorless; putting
semantic seeds INTO the lexical head (lex 7 + sem 3) reaches 16 alone but changes
the first ten seeds, which the union rule forbids.

## Latency

`_vault_recall` on production qmd/djev, one process, arms interleaved ABBA per
query, 86 queries × 2 × 2, under `regression.lock`, embedder warm: off p50 221 ms /
p90 574 ms, on p50 315 ms / p90 672 ms; paired mean **+71 ms [0.1, 128]**. Without
the early doc-leg start the embedding (~0.2 s) would sit in series.

## Owed by a human

1. Merge + `round restart` (aggregator and backend both import the changed code).
2. Build the index: `.venvs/lloyd/bin/python scripts/memory/build_entity_vectors.py`
   (~40 min CPU for ~12k entities) → `~/lloyd-data/_pipeline/vault-derived/entity-vectors/`.
   Until it exists the semantic half is a no-op (alias surfaces still apply). The
   vectors measured here are at `~/lloyd-data/eval/1486/vec-f3/` (same model,
   transformers-built, same text) and can be copied there instead. Rebuild after a
   KG rebuild; entities created since are simply not semantic candidates.
3. Optional, no measured gain: `scripts/memory/regenerate_aliases.py --report <path>`
   (dry run), then `--apply` against the live store. Every row is `origin=regen:*`
   and removable with `aliases.remove_exact`.
4. Expect the nightly's entity metrics to step up once 1–2 land; the trend audit
   will read it as a change, which it is.
5. The model weights live in the HF cache (`Qwen/Qwen3-Embedding-0.6B`,
   `~/.cache/huggingface/hub`), downloaded 2026-09-25; a rebuild needs them
   (`retrieval.entity_seeding.semantic.model_dir` overrides the path).
6. #1502's anchoring-definition dispute: both counts are above; they agree on the
   direction (25 → 16 lenient, 38 → 27 strict).

Kill switches: `retrieval.entity_seeding.{alias_surfaces,semantic.enabled}` or
`LLOYD_ENTITY_SEEDING=0` in the aggregator's environment.
