# `skills_search` gets the embedding re-order (2026-11-10)

**Adopted.** `skills.search_embed_rerank: true`.

`skills.embed_fusion` re-orders the chat prefetch's matched skills using lexical score
plus weight × 10 × cosine. It was adopted on 2026-11-09
(`skill-embed-fusion-2026-11-09.md`), but it never reached the `skills_search` tool,
which kept lexical order. The tool now uses the same function
(`app/skill_embed_rank.rerank`) with the same weight (2.0) and k (16). Its wait budget is
1,500 ms instead of 200 ms, because a tool call is not bound by prefetch's 300 ms. A cold
model still answers in lexical order.

## Measurement

`eval/run_skill_match_eval.py --tool-arm`: 78 records against 208 active skills, 54 of
them with an expected skill. Run under `regression.lock`.

| arm | recall@5 | hit@5 | hit@1 | MRR@5 |
|---|---|---|---|---|
| prefetch matcher, lexical | 0.639 | 0.685 | 0.481 | 0.569 |
| `skills_search`, lexical (before) | 0.694 | 0.741 | 0.481 | 0.589 |
| `skills_search`, re-ordered (after) | **0.806** | **0.833** | **0.722** | **0.765** |

Against the prefetch lexical arm, paired over n=54:
- hit@1 +0.241 [+0.130, +0.352].
- recall@5 +0.167 [+0.074, +0.269].
- MRR +0.197 [+0.115, +0.289].

The tool's own lexical arm has the same hit@1 as the prefetch lexical arm (0.481), so the
hit@1 gain over the tool's previous behaviour is the same +0.24.

## Cost

- **Memory.** About 2.5 GB RSS of fp32 Qwen3-Embedding weights in the aggregator
  (`lloyd-mcp`), loaded on the first `skills_search` call. That call answers in lexical
  order while the library is embedded in a background thread. The backend holds its own
  copy for the prefetch leg, so the host carries two. MemAvailable was 52 GiB at the
  time.
- **CPU.** About 150 ms per query on one dedicated thread.
- **Tests.** `tests/conftest.py` holds the setting off in every test process, so xdist
  workers never load the weights.
