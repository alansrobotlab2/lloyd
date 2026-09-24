# #562: what the prefetched `<context>` block costs downstream (2026-09-24)

Instrument: `eval/run_prefetch_cost_eval.py`. Artifact: `prefetch-cost-562-merged.json`
(compact: numbers only. The rendered blocks and answers are in the raw per-run
artifacts, which were kept off the tree). Section map: `prefetch.injected_section_sizes`
/ `drop_sections` / `split_injected`. The `PROMPT_BUDGET` log line now carries
`sections=`.

## Design

- **Queries.** The legacy 20-query nightly set, which is the first 20 of
  `vault_recall_queries.yaml`. Each query is rendered through `prefetch_context`
  once, after a discarded warm pass (a cold process drops the whole `<skill>`
  section at the 300 ms budget). Every arm is then run end to end on the primary
  with the production system prompt, the tool surface (`surface=chat`), the safety
  hook, `max_turns` 30 and a 900 s timeout.
- **Sandboxing.** Sessions are `pt-eval-562-*`, so the aggregator runs their Bash
  read-only, and the run refuses to start unless `/state.tool_sandbox` says the
  sandbox is enforced.
- **Arms.** Four, each paired per query:
  - `injected`: the rendered block as a chat turn would send it.
  - `suppressed`: the bare query.
  - `no_skill`: the rendered block with `<skill>` removed.
  - `injected_aa`: `injected` run a second time, which measures the noise floor.
- **Cost metric.** `prefill_tokens` counts what each request added past the one
  before it, and for the first request everything past the 44,843-token
  system+tools prefix. It does not depend on the prefix cache. The engine's own
  `uncached_prompt_tokens` is recorded too, but it cannot be used across arms: it
  credits the A/A arm with its twin's cache (−35k on average against an identical
  prompt).
- **Quality.** The primary, with thinking off and blind to the arm, grades the
  answer against the gold note(s): 0 wrong, 1 partial, 2 right.
- **Data.** The fact tree and KG were copies of production's, pointed to through
  `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB`. The session index is empty in a worktree, so
  `<recent-sessions>` never rendered. It fires only on temporal queries.
- **Engine and lock.** The engine was shared with other evals the whole time
  (8 running / 6–13 waiting), under the shared primary lock. Token counts and
  grades do not depend on load. Wall seconds do, and are not compared.
  - The `no_skill` arm ran later on a busier engine, and its process hit an outer
    timeout before writing its artifact. 19 of its 20 arms were recovered from the
    run log. It has prefill, iterations, tools and judge, but no uncached or
    output tokens. The runner now checkpoints after every arm.

## Results (n = 20, paired 95 % bootstrap CIs)

| arm | injected tok | prefill tok | output tok | iterations | judge (0–2) | judge = 0 |
|---|---|---|---|---|---|---|
| injected | 1,638 | 28,159 | 5,472 | 10.7 | 1.85 | 1/20 |
| injected_aa | 1,638 | 28,646 | 5,764 | 10.3 | 1.65 | 3/20 |
| suppressed | 0 | 24,765 | 4,490 | 10.3 | 1.47 | 5/19 |
| no_skill | 318 | 27,230 | — | 10.8 | 1.88 | 1/17 |

- **Noise floor (A/A).** The median |Δprefill| between two runs of the same prompt
  is 4,221 tokens, and the p90 is 15,369. The mean A/A difference is +487
  (CI −3,074 … +4,338). One prompt's two runs disagree by about 2.6× the whole
  block.
- **suppressed − injected.**
  - prefill −2,977 (CI −7,895 … +1,768, p 0.23). The −2,977 includes the block
    itself (1,638).
  - output tokens −858 (CI −1,617 … −207, p 0.006). This is the one significant
    cost effect: with the block the model writes about 19 % more.
  - iterations −0.47, n.s.
  - judge −0.37 (CI −0.84 … +0.11, p 0.13).
- **marginal_value_per_injected_token = −1.82.**
  - The block does not remove downstream prefill: the point estimate is that it
    adds ~0.8 tokens past its own size per injected token.
  - The CI on the underlying difference (−4.8 … +1.1 per token) crosses zero, so
    the sign is not established.
- **negative_marginal_hit_count.**
  - 5 of 6 doc-hit queries, by clause 4's definition.
  - `strict` count: 4. This also requires the no-block turn to be graded at least
    as good.
  - Beyond the A/A p90: 1, `tgs-rag-state` (+28.6k). Its suppressed arm was graded
    0, so the block bought that answer.
  - No hit is net-negative beyond noise.
- **no_skill − injected.** The `<skill>` section averages 1,320 tokens: 81 % of the
  block here and 47 % of production's injected characters.
  - prefill −86 (CI −3,765 … +3,438).
  - iterations +0.9, n.s.
  - judge +0.06, n.s.
  - The skill body neither saves nor costs anything this eval can see.

## Verdict

**The prefetch block's marginal value per injected token is not shown positive on
prefill.** Its point estimate is negative (−1.8), inside a noise floor about 2.6×
the block. What it does show is quality: 1 zero-graded answer in 20 with the
block, against 5 in 19 without it (judge +0.37, p 0.13). It also costs about 860
more output tokens (p 0.006).

**No section is net-negative beyond noise**, so no trim knob ships and the
defaults are unchanged. `<skill>` is the only section big enough to matter. Taking
it out moved neither cost nor quality measurably. Separating its ~1.3k-token
direct cost from a zero downstream effect would need roughly 10× the queries: at
these σ, n ≈ 200 to resolve ±1.3k.

Measured null, `rejected` as a tuning proposal. The instrument and the section map
stay.

## Clause mapping and re-scopes

- **Clauses 1–2, re-scoped.** The "nightly retrieval artifact" carries
  `injected_tokens: null` by design (`run_prefetch_eval.py` header, #875). Like the
  tool-choice artifact (`run_tool_choice_eval.py`), which renders the block and
  spends the turn, this artifact carries `injected_tokens`, `iterations` and
  `uncached_prompt_tokens`, non-null for all 20 queries. It also carries a
  suppressed arm for every query. The tool-choice set itself stops at the first
  tool call and has no gold docs, so it cannot carry a downstream cost or a
  doc_hit.
- **Clause 3, re-scoped** to this eval's `summarize()`, for the same reason:
  `run_prefetch_eval.py` never builds a block.
- **Clause 4.** `negative_marginal_hit_count` is in the artifact, with both arms'
  per-query numbers.
- **Clause 5.** `prefetch.injected_section_sizes`, `context_sections` on
  `log_turn_prompt_budget` and the `sections=` log field.
  `tests/test_prefetch_cost_eval.py` pins it.
- **Clause 6.** No retrieval code changed, and `_format_context` output is
  byte-identical. No tuning step shipped, so the 0.85 / 0.536 floors
  (`TUNING_FLOORS`) are recorded but not exercised.
- **"Three consecutive nights".** Not done. This is a 60+ min end-to-end run on
  the primary, so wiring it into nightly task #82 is a scheduling decision, left
  for the coordinator.
