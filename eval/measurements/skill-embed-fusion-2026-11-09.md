# Skill matcher: embedding re-order of the lexical offers, 2026-11-09

**Verdict: ship for chat sessions (`skills.embed_fusion.enabled: true`, weight 2).**
The cosine between the user's turn and each skill's `name: description` re-orders the
lexical top-16 offers; scores, the injection gate and the offer pool stay lexical.

Runner: `eval/run_skill_match_eval.py --embed-fusion <weights> --embed-pools
rerank,admit`, `eval/skill_match_queries.yaml` (51 labelled turns, 24 expected-empty),
208 live skills, embedder `app.qwen3_embed` (Qwen3-Embedding-0.6B, CPU). No model call,
no GPU. Paired bootstrap CIs from `eval/stats.py`, arm minus lexical, n beside each.

## Results

| arm | hit@1 | recall@5 | MRR@5 | expected-empty named |
|---|---|---|---|---|
| lexical (production) | 0.510 | 0.677 | 0.602 | 15/24 |
| rerank w0.5 | 0.569 | 0.716 | — | 15/24 |
| rerank w1 | 0.588 | 0.735 | 0.662 | 15/24 |
| **rerank w2** | **0.686** | **0.755** | **0.722** | **15/24** |
| rerank w3 | 0.706 | 0.745 | 0.735 | 15/24 |
| admit w1, floor 0.60 | 0.628 | 0.814 | 0.716 | **23/24** |
| admit w1, floor 0.70 | 0.628 | 0.814 | 0.721 | 18/24 |
| admit w1, floor 0.75 | 0.628 | 0.794 | 0.711 | 15/24 |

rerank w2 vs lexical: hit@1 +0.176 (95% CI +0.078…+0.294, n=51), recall@5 +0.078
(+0.020…+0.157), expected-empty named +0.000 (n=24).

hit@1 is the number that matters for injection: the first injectable skill gets its
full body, the second only an excerpt at a score ≥ 4.0.

## Why this is believed

The 2026-09-25 pseudo-query arm looked like a win on one generation and did not
reproduce on a second, so three checks were run before reading the table:

- **Instruction wording.** Three query instructions (the shipped one, "Find the
  procedure document relevant to this message", and none) give hit@1 0.667–0.706 at
  w2–w3 and 0.588 at w1 for all three: the gain is not a prompt artifact.
- **Split half.** The weight chosen on the even-index labelled turns (w2) gains +0.160
  hit@1 and +0.120 recall@5 on the odd half (n=25); chosen on the odd half (w3) it gains
  +0.192 hit@1 on the even half (n=26).
- **Production path.** `prefetch._search_skills_for_turn` with the config on, scored by
  the same harness, reproduces rerank w2 to the digit (0.6863 / 0.7549 / 15 of 24).

## Why rerank and not admit

`admit` lets a skill in on cosine alone. At floor 0.60 it names a skill on 23 of 24
expected-empty turns — the #311 failure and the 2026-09-14 "geometric" arm's — and the
floor that restores 15/24 (0.75) gives the same hit@1 as rerank w1 at a lower one than
rerank w2. `rerank` cannot admit by construction, so the precision control cannot move.

## Costs

- **Latency:** query embed p50 133 ms, p90 164 ms (CPU, 450 calls). Bounded by
  `wait_ms` (200) inside the skills leg; a late embed answers in lexical order.
- **Memory:** ~2.5 GB RSS for fp32 weights in the backend, loaded on the first chat
  turn. Host had 62 GB available when measured.
- **Vector build:** 66.6 s for 208 skills, once, in the background; then only edited
  skills, cached by text hash in `_pipeline/skill-embeddings.json`.
- **Scope:** chat sessions only. Worker prompts are boilerplate the gold set does not
  cover, and they are 97% of prefetch traffic.

## Gold-set caveats

The three disputable labels the 2026-09-25 audit named (`smq029`, `smq043`, `smq060`)
are in the set as labelled. The lexical baseline has drifted from the recorded 0.6961
(2026-09-24) to 0.6765 as the library grew from 189 to 208 skills.
