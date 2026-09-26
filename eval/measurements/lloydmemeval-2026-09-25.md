# LloydMemEval v1 — the set, the baseline, and #1482 rider 1 (2026-09-25)

Items: **#1480** (the eval) and **#1482 rider 1** (relevance-ordered `<facts>`).
Raw artifacts: `~/lloyd-data/eval/1480/` — `runs/lloydmemeval-baseline-2026-09-25.json`
(every dev answer, both judges' readings, the prefetch facts lines per question),
`episodes.jsonl` / `drafts.jsonl` (the build), `corpus/` (the facts snapshot every
prefetch arm read).

## 1. The set

The retained chat corpus cannot supply a conversational-memory eval (13
mission-control sessions, 6 with ≥3 user turns, nothing before the 2026-09-22
wipe), so v1 is **synthetic and grounded** — Alan delegated that scope call:

- `eval/memory_eval_build.py sample` picks **360 episodes** from a snapshot of the
  live fact tree (11,959 entity dirs; entities with 3–400 active facts, facts
  40–240 chars carrying a number, code-ish token or proper noun, nothing
  email/phone/secret-shaped, entity reachable by prefetch's own entity extractor,
  ≤2 episodes per entity). Deterministic in seed 1480.
- `generate`: the **primary** (Qwen3.8-Flash-Next-nvfp4, thinking off, JSON
  grammar) writes the dated past sessions and the later question. A mechanical
  validator refuses a draft whose gold value is not in its fact, not in its
  evidence session, is in the question, or is a sentence (>6 words); a knowledge
  update must state an invented old value in s1 that the real fact does not
  contain. Two passes (the second re-drew refusals and every temporal-order
  item after the first pass showed order golds that did not match the labels the
  question used): **333 accepted of 360**.
- `freeze --version v1`: `eval/memory_eval/v1/` — 267 dev + 66 holdout (20%,
  stratified, ordered by `sha256("lloydmemeval-holdout-v1:" + id)`), one JSON per
  episode, `manifest.json` with a sha256 per file, the reserved ids and split
  hash, and `generator_model`. `run_memory_eval.py verify` recomputes all of it.

| category | dev | holdout | what it asks |
|---|---|---|---|
| single_session | 57 | 14 | one fact said in passing mid-session among unrelated topics |
| multi_session | 48 | 12 | two facts from two sessions; both gold values required |
| knowledge_update | 53 | 13 | old (invented) value in s1, real value in s2; action-shaped probe; old value = anti |
| temporal | 57 | 14 | which came first / days between two sessions / what was said on a named date |
| preference | 52 | 13 | a real preference fact stated by Alan; a request the preference should shape |

**Gold audit (35 dev items, 7 per category, by hand — `v1/AUDIT.md`):** 24 clean,
8 brittle (right label, long or generic gold string), **3 defective** (8.6%,
Wilson 95% [3.0%, 22.4%]): mu-060, pr-044, pr-042. Preference is the weak
category (2 of 7). This is a spot audit, not the full human review #1471's
closure asks for before a synthetic set is called gold.

## 2. The runner

`eval/run_memory_eval.py run` asks the primary each question (thinking on,
T=0.6, top_p 0.95, seed = sha256(id), max 4096 tokens) under four arms:
`closed_book`; `history` (the episode's sessions + 4 distractor episodes from
the same leg with no shared entity, dated, in the system prompt); `prefetch`
(`prefetch.prefetch_context(question)` over the facts snapshot and the live qmd
daemon, no history; the `<facts>` section spliced from
`_search_facts(rank="confidence")`); `prefetch_rel` (the same block, `<facts>`
from `rank="relevance"`). A prompt identical across arms is answered once.

Judging: the deterministic **rules** judge (fold + whole-token alias match)
settles `correct` / `stale` / `abstain`; `mixed` (gold and anti both named) and
`none`/`partial` go to **djev** (`choice`, negative options first, argmax).
The generator (the primary) is refused as judge in code (`SelfJudgeRefused`).

**Judge audit (hand, on this baseline):** djev's reading of *which value a
mixed answer acted on* was right **12 of 12**; its paraphrase upgrades
(`none` → `gold`) were right **11 of 20** (8 wrong — placeholder commands,
generic titles, an abstention — 1 ambiguous). On rules-settled answers djev
agreed 369/401 (0.920). So **`correct_strict`** (rules; djev only for mixed) is
the headline; `correct` (djev paraphrase upgrades too) is reported beside it
and over-counts about as much as strict under-counts (the audit's 8 brittle
golds).

## 3. Baseline on today's stack (dev n=267; accuracy run, shared primary lock)

`correct_strict`, Wilson 95%, per-category planning MDE (α=0.05, power 0.8):

| arm | single | multi | know.upd | temporal | preference | **all** |
|---|---|---|---|---|---|---|
| closed_book | 0.105 | 0.000 | 0.000 | 0.035 | 0.096 | **0.049** [0.029, 0.082] |
| history | 0.912 | 0.875 | 0.830 | 0.895 | 0.673 | **0.839** [0.790, 0.878] |
| prefetch (today) | 0.544 | 0.188 | 0.396 | 0.210 | 0.115 | **0.296** [0.244, 0.353] |
| prefetch_rel | 0.632 | 0.417 | 0.623 | 0.210 | 0.192 | **0.416** [0.358, 0.476] |
| MDE80, one category | ~0.16–0.26 | | | | | **~0.07–0.12** overall |

Retrieval (`evidence_in_context`: every gold value in what the model was
shown): prefetch 0.532 [0.472, 0.591], prefetch_rel 0.648; by category
prefetch single 0.579 / multi 0.312 / k.upd 0.566 / temporal 0.860 / pref 0.288.
Use (`correct / mentioned`): history 1.000 (n=224), prefetch 0.878 (n=90).

Holdout (aggregates only, n=66): closed_book 0.045, history 0.848, prefetch
0.242, prefetch_rel 0.439 — same ordering and same gaps as dev.

Paired (dev, `correct_strict`, paired bootstrap 10k): history − closed_book
**+0.790** [+0.742, +0.839]; prefetch − closed_book **+0.247** [+0.191, +0.303].

What the baseline says:

1. **The questions need memory** (closed book 0.049) and **are answerable from
   the sessions** (history 0.839; the misses include the brittle golds of §1 and 4
   answers cut at 4096 tokens). The eval measures something.
2. **Today's memory stack surfaces the fact for about half the questions and
   answers 30% of them.** Temporal is structural: the gold facts are in the
   block 86% of the time, but prefetch renders no session dates, so a
   when-question cannot be answered (0.21 either way) — P3/#1485's territory.
   Multi-session and preference are retrieval-bound (31% / 29% of golds reach
   the block).
3. Knowledge-update `stale` answers: 2/53 (prefetch) — the stack's facts carry
   only the current value here by construction (the old value exists only in
   the synthetic session), so this category measures retrieval + use for the
   stack arms and supersession only for `history`; #622's shadow-store harness
   (`run_stale_fact_eval.py`) remains the supersession measurement for stores.

Caveats: (a) synthetic set, 8.6% defective in the audit; (b) the primary
generated the questions and answers them — a closed-book advantage for
paraphrase-y facts is possible (0.049 says small); (c) 50 of 1,068 dev answers
hit the 4096-token cap (thinking loops) and score wrong; (d) the prefetch vault
leg ran on the live daemon under load (prefetch p50 110 ms), so the `<vault>`
half of the block is what production would have rendered at that moment, while
the `<facts>` half is deterministic.

## 4. #1482 rider 1: `<facts>` ordered by query relevance

`prefetch._search_facts` sorted each entity's facts by confidence and kept 3.
Now (`prefetch.facts.rank: relevance`) it ranks by `retrieval.fact_score` —
the fraction of the query's tokens in the fact, the scorer `fact_get`'s query
path already uses — with confidence as the tie-break, over the facts already
in hand. No usable query token falls back to the confidence order.

Paired, dev n=267, identical blocks except the `<facts>` lines:

| metric | prefetch | prefetch_rel | diff [95% CI] | MDE80 |
|---|---|---|---|---|
| **answer correct (strict)** | 0.296 | 0.416 | **+0.120 [+0.071, +0.172]** | 0.073 |
| gold in the block | 0.532 | 0.648 | +0.116 [+0.064, +0.169] | 0.074 |
| answer correct (lenient) | 0.521 | 0.637 | +0.116 [+0.064, +0.169] | 0.075 |
| no-model retrieval A/B (`prefetch-retrieval`, facts lines only) | 0.371 | 0.517 | +0.146 [+0.090, +0.202] | — |

By category (strict): multi +0.229 [+0.083, +0.375], knowledge_update +0.226
[+0.113, +0.340], preference +0.077 [+0.019, +0.154], single +0.088 [−0.035,
+0.210], temporal +0.000 [−0.088, +0.088]. No category moved down. Holdout
(aggregates): 16/66 → 29/66.

Cost: zero reads (pinned by `tests/test_prefetch.py::test_facts_relevance_adds_no_read`).
CPU: `_rank_entity_facts` is 0.3 → 6.6 ms on a hub entity (Lloyd 3,509 facts,
vLLM 3,060), 0.04–0.13 ms on ordinary ones; `_search_facts` median 1.48 → 1.65
ms on a mixed query. It runs in the parallel facts worker inside the 300 ms
budget, beside a vault leg that takes ~50–110 ms.

**Caveat that matters:** the questions were written from the facts, so their
wording overlaps the gold fact's more than a cold user turn would — which is
exactly what a token-overlap ranker rewards. The size of the gain is likely
overstated for real traffic; its sign is not in doubt at this n, and the change
costs no read.

**Verdict: deploy on** — `prefetch.facts.rank: relevance` in config.yaml, in
the #1482 commit. A backend restart picks it up.

**Rider 2 (the 25-char hybrid floor): owed.** v1 has **0 of 267** dev prompts
under 25 characters (shortest 36; median 126) — the set cannot measure it. It
needs short real turns or a v2 slice of deliberately terse follow-ups.

## 5. Reproduce

```bash
.venvs/lloyd/bin/python eval/run_memory_eval.py verify
flock -s ~/.local/state/lloyd-automod/primary.lock \
  .venvs/lloyd/bin/python eval/run_memory_eval.py run \
  --arms closed_book,history,prefetch,prefetch_rel --holdout --label <label>
.venvs/lloyd/bin/python eval/run_memory_eval.py prefetch-retrieval   # no model
```

The prefetch arms read the facts snapshot at `~/lloyd-data/eval/1480/corpus/`
(`--corpus`); a run against today's live tree is a different corpus and must be
compared unpaired. Answering took 31 min (1,872 s) for 1,275 distinct prompts at
concurrency 8 on a contended engine (djev judging and prefetch rendering not included). Set sha `9cf67d045c38a8c3…`.
