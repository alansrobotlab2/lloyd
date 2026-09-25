---
segment: architecture
title: Recall research pass 2026-09-24 — the whole recall stack, the frontier, and sixteen proposals
tags: [architecture, retrieval, memory, subliminal, skills, research]
type: reference
status: implemented
date: 2026-09-24
---

# Recall research pass, 2026-09-24

Alan asked for a review of the whole recall stack (KG, qmd, djev, vault, memory,
skills) and a pass over open-source frontier techniques that would improve what
Lloyd recalls *and* how well he uses it. Every proposal had to run on this box
(GPU 0 3090 shared by TTS and qmd, GPU 1 PRO 6000 with the primary, GPU 2 3090
with djev, 251 GiB host RAM) or on one more RTX 3090 that Alan owns.

This doc is the record: the map as measured that day, what the literature says,
the proposals filed as backlog items **#1480–#1495**, and what was deliberately
not proposed. One of the proposals (P2) was measured and landed the same day;
§4 has the numbers. [[retrieval]] is the document leg's long version,
[[subliminal]] the prefetch path's, [[memory]] and [[knowledge-graph]] the fact
layer's, [[skills]] skill delivery's.

The house rules every proposal carries: an item is a research/eval proposal and
deploys only on a measured gain (equal accuracy with more throughput counts; a
slower change needs a gain clear of the paired-bootstrap interval); open source
may be adopted, commercial products only recreated; the doc gold set is n=81/86
with an MDE of about +0.09 MRR / +0.12 doc_hit, so most gains proposed here
(+0.02–0.06) are unmeasurable until P0 (#1480) or #1471 grows the evals; a round
must not edit `qmd/**`.

## 1. The map, as measured on 2026-09-24

**Document leg** (`agent_mcp/vault.py::_vault_recall`). qmd fork daemon :8181 on
GPU 0: FTS5 BM25 (OR mode, 11 collections) plus Qwen3-Embedding-0.6B Q8 (39,428
chunk vectors, 900-token chunks, 15% overlap), RRF k=60, head 20 plus floors of
2×3 (≤32 rows), then djev listwise over 160-char candidates in ~0.5 s, top 20
documents plus up to 10 facts. The cross-encoder is the fallback. The pool is the
ceiling. Nightly, live index, n=81: doc_hit 0.704, doc_recall 0.647, MRR 0.355,
NDCG@10 0.407, **entity_hit 0.358**, entity_recall 0.391, **fact_entity_recall
0.374**, p50 487 ms; 25 of 81 queries anchorless.

**Who calls recall.** About 24 times a week, all workers, never chat. Chat gets
memory through `prefetch.py`'s `<context>` prefix on the user message (300 ms
budget) and through whatever the model chooses to Read or Grep.

**The subliminal channel** carries every automatic recall: a byte-stable system
prompt (prefix-cached), the ephemeral `<context>` prefix rebuilt each turn from
the message and `SessionFocus` (2–8 KB, uncached, persisted as a `subliminal` row
nobody replays), and tools on demand. Ambient producers drain first, ≤3 per turn.
Direct-path runs (`run_prompt_on_primary`, `autonomy.run_task`) are prefetched by
nothing. A hard `/compact` discarded the subliminal rows until D11 (2026-09-24); it no
longer rewrites the messages.

**Sessions.** qmd's `sessions` collection is searched by prefetch but not by
`vault_recall`; `session_recall` is token overlap over seven days, five results,
cut at 5,000 chars (#1090).

**Facts and KG.** ~61k fact files, ~274k active facts, and `kg.sqlite` rebuilt
after the 09-22 wipe: 11,932 entities, 109,030 fact rows, 35,582 active edges
(30k `mentions`), **166 aliases (4,028 before the wipe)**. Graph lookup at recall
time is off in production (`RECALL_EXPAND_GRAPH` / `RECALL_GRAPH_RERANK`, stripped
at the MCP boundary) while the nightly eval runs `expand_graph=True`; graph-read
tools had zero calls in 1,540 sessions (#1077). The facts leg ranks by token
overlap; prefetch takes top 2 entities × 3 facts **sorted by confidence, not
relevance**. The improve loop found 133/133 detector pairs were near-duplicates
from auto-capture, and post-capture's per-session "durable facts" have produced
5 `session-extracted` rows: conversations barely reach the fact layer.

**Memory in the prompt.** SOUL.md 7 KB + MEMORY.md 73 KB (at its 73,728-byte
ceiling) + USER.md 16 KB ride whole into every turn, ≈96 KB, above the 80 KB
`PROMPT_BUDGET_CHARS` tripwire, which only logs.

**Compaction.** Microcompact spills each cleared result and leaves a marker naming
the file; LLM summarisation has never fired (0 of 802 clears, #1078). The #600
planted-fact eval: production 19/20 distinctive and 20/20 ambiguous, the aggressive
`tool_clear` arm 12/20 and 12/20, production paying +3.6–4.7 s TTFT per cleared
turn (`eval/measurements/compaction-recall-2026-09-24.md`).

**Skills.** 189 live skills; the system prompt lists names only; prefetch scores
the message lexically and injects the top skill at ≥3.0. The modal score sits on
that threshold because scores scale with corpus frequency, not query specificity.
Semantic matching (#557) lost to lexical (recall@5 0.686 vs 0.745).

## 2. What the frontier says, filtered to what runs here

1. **The memory substrate matters more than the model.** Across 8 backbones
   LongMemEval moves 3.4 points while cost moves 30× (Agent Zero Memory). Winners
   share hybrid search, an intent gate, a source router, provenance, citation lock.
2. **Update-on-write fact stores beat embedding retrieval on changed facts**
   (0.70–1.00 vs 0.30–0.95), and agents act on a correctly retrieved value only
   55% of the time (When Does Memory Help?).
3. **Sleep-time compute** cuts test-time compute ~5× and lifts accuracy 13–18%;
   ACE's delta curation beats rewrites (+10.6%).
4. **Instruction files grow without bound**; a rationale per line lets a curator
   delete 99.3% of excess. ReasoningBank distils strategies from failures too.
5. **Lossless compaction beats summarisation**: ARC's addressable observation log
   (99.4% vs 88.1% NIAH on Qwen3); CliffCompaction truncates and never re-compacts.
6. **Graphs help only on multi-hop/associative queries** (HippoRAG 2, PPR over
   entity + passage nodes).
7. **Retrieval is a decision**: gating on uncertainty cuts retrieval 70–90% at equal
   quality (TARG); store routing beats uniform retrieval on accuracy and tokens.
8. **Skill routing needs the body** (hiding it costs 37–44 points Hit@1);
   pseudo-queries per skill fix misalignment; one disclosure tier helps, two never.
9. **Context pruning at rerank time is free** (Provence).
10. **Open models that fit a 3090**: Qwen3-Embedding-4B/8B, Qwen3-Reranker-4B/8B,
    BGE-M3, jina-reranker-v3, GLiNER2, Provence, PyLate ColBERT.
11. **Navigational retrieval** (PageIndex, Corpus2Skill) wins on single-domain
    corpora with a taxonomy; the vault is open-domain and Lloyd already navigates
    with Read/Grep.
12. **RL-trained retrieval policies** are strongest and need a training loop; out
    of scope.

## 3. The proposals, as filed

Priority `high` goes to P0, P1, P2, P4 and P5 so the loop takes them next; the
rest are `medium`. Every item carries its technique and source, files, hardware,
measurement and do-not-duplicate pointers.

| # | item | what | measured by |
|---|---|---|---|
| P0 | #1480 | LloydMemEval: single/multi-session, knowledge update, temporal, preference, plus a "use" metric; ≥300 questions | is the measurement |
| P1 | #1481 | Addressable observation stubs + `recall_observation(id)`; never re-compact a stub; persist in-turn clears | #600 `tool_clear` 12/20 |
| P2 | #1482 | Hybrid vault leg lands in-turn (**landed**, §4); riders: facts by relevance, the 25-char floor | prefetch eval |
| P2b | #1483 | One djev `decide` per turn: intent gate + store router for `<context>` | injected chars, "use" |
| P2c | #1484 | Mid-turn ephemeral recall from the tool stream, appended via `state_anchor` | long-turn benches |
| P3 | #1485 | `sessions`/`autonomy-runs` as recall floors; `session_recall` as a qmd query | P0 multi-session/temporal |
| P4 | #1486 | Alias regeneration + semantic entity retrieval (§4) | entity_hit 0.358 |
| P5 | #1487 | ADD/UPDATE/NOOP gate at `fact_add`; conversations reach the fact layer | P0 knowledge update |
| P6 | #1488 | Rationale-line curation of MEMORY.md; core + retrieved archive; sleep-time notes via the ambient queue | behaviour bench, cache |
| P7 | #1489 | ReasoningBank for autocode rounds | scorecard rows |
| P8 | #1490 | Skill pseudo-queries, rerank over bodies, one-line descriptions | skill-match eval 0.745 |
| P9 | #1491 | Answer-bearing sentence pruning of the best-chunk window | "use", MRR (after #1467) |
| P10 | #1492 | Entity→chunk mentions + PPR fused into the pool | multi-hop + hard only |
| P11 | #1493 | Qwen3-Embedding-4B side index | pinned-corpus paired |
| P12 | #1494 | human-only: LLM-written contextual chunk prefixes in the qmd fork | pinned-corpus paired |
| P13 | #1495 | Fourth 3090 as the retrieval card | rerank p50, card sharing |

Filed before this pass and not duplicated: #1456 (prefetch phrases into recall),
#1461 (drift signal), #1467–#1479 (the djev pass: snippet strip, binary rows,
FIRST row, djev facts leg, grow the gold set, KG mention backlog, compaction
droppability, anchored chunks, fork first-stage, djev raw generation, overlay
rebuild, pair-judge, calibration ladder).

## 4. Measured the same day

**P2, the subliminal vector leg (#1482, landed).** `prefetch.py` sequenced the
lex+vec leg after the lex leg, expected it to take 1.1–3.0 s and carried its hits
to the next turn. On the 86-query `eval/run_prefetch_eval.py`, run twice, it takes
**52 ms p50 / 57 ms p90** after the lex leg: the fork's in-memory `VecIndex`
replaced the per-collection exact scans. Lex alone doc_hit 0.140, hybrid alone
0.151, the two merged 0.221, so neither leg is a superset. Replaying the real
`_prefetch_run` over the gold set as a first turn, old code against new, two runs
each (identical): the hybrid leg landed on 86 of 86 turns; doc_hit 0.151 → 0.221;
MRR 0.093 → 0.114 (7 queries better, 1 worse); turn latency p50 27 → 62 ms and p90
301 → 89 ms, because the old path pinned 14 of 86 turns at the budget wall. The
landing gate set beforehand (hybrid in budget on ≥85% of turns, merged doc_hit not
below today's) was met. [[subliminal]] §Worker 3 has the mechanism.

**P4, entity anchoring (#1486).** A scratch GLiNER2 run (`fastino/gliner2-base-v1`,
CPU, scratch venv) over the 66 gold queries with expected entities: today's
extractor has a seed naming an expected entity on 28; GLiNER2 spans linked to the
registry by normalized n-gram match, 27; the union, 30. All 36 queries still
unanchored have their expected entity *in the registry*: they paraphrase it rather
than name it. Span detection is not the bottleneck, so the item was re-weighted
toward alias regeneration and semantic entity retrieval over `name + definition`.

**P1, compaction (#1481).** The number to move is the #600 `tool_clear` arm, 12/20
on both facts, against production's 19/20 and 20/20 at +3.6–4.7 s TTFT.

## 5. Not proposed, and why

- **Mem0 / Zep / Letta / MemOS / Cognee as frameworks.** Lloyd already has their
  parts; the rule is recreate aspects (P5, P6 do).
- **PageIndex / Corpus2Skill navigation.** Open-domain vault; Read/Grep already
  navigate.
- **ColBERT late interaction as the ranker.** djev already beat the cross-encoder;
  PyLate stays an option if GPU 2 is ever freed.
- **Learned sparse (SPLADE, BGE-M3 sparse) as the lex leg.** A fork FTS rewrite for
  an uncertain gain over OR BM25; revisit after P12.
- **Host-tier KV (KVMem, LMCache).** Adjacent to prefix misses, not recall, and the
  host RAM already carries the 95 GiB n-gram table.
- **RL-trained retrieval policies; parametric memory / fine-tuning the primary.**
- **Already rejected, not re-proposed:** self-question arm (#1164), multi
  chunk-size RRF (#1192), wider canvas (#1345), age decay (#472), the raw-transcript
  episodic *doc* arm (#675 — P3 uses its entity signal instead), qmd query
  expansion, near-dup collapse, collection and query-type routing, front-matter
  titles, activity-log strip, USER.md trim (#1425), skill embedding arm (#557), late
  chunking, per-turn tool shortlisting (breaks the KV prefix).

## 6. Sources

Memory systems and benchmarks: Agent Zero Memory (arXiv 2608.29606); When Does
Memory Help? (2609.05441); MRAgent (2606.06036); SwiftMem (2601.08160); NapMem
(2607.05794); MemPro (2606.00619); Omni-SimpleMem (2604.01007); HiGram
(2608.05095); Dual-Layer Agentic Memory (2608.22215); Zep/Graphiti (2501.13956);
MemOS; Mem0 state of memory 2026; survey 2603.07670.
Sleep-time and curation: Sleep-time Compute (2504.13171); ACE (2510.04618);
ReasoningBank (2509.25140); Letta Context Repositories; Catastrophic Remembering in
CLAUDE.md (2608.11095).
Compaction: ARC (2607.25066); CliffCompaction (2609.26779); CompactionRL
(2607.05378).
Retrieval: HippoRAG 2 (2502.14802); When to use graphs (2506.05690); Anthropic
contextual retrieval; chunking evaluation (2504.19754); Provence (2501.16214); TARG
(2511.09803); store routing (2603.15658); Rank1 / ReasonRank / Rank-K; PLAID /
MUVERA; Qwen3-Embedding (2506.05176); Qwen3-Reranker model cards; GLiNER2.
Skills: SkillRouter (2603.22455); Skill Is Not Document (2606.03565); SkillDreamer
(2609.01642); Skill2Query (2608.16071); Is Progressive Disclosure All You Need
(2607.17598); Corpus2Skill (2604.14572).

## Review log

- **2026-09-24:** written by the recall research pass; items #1480–#1495 filed; P2
  landed the same day.
