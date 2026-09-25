# ReasoningBank for autocode (#1489), offline replay — 2026-09-25

**Verdict: landed OFF. Retrieval by item similarity is rejected; a static
"most frequent refusal causes" paragraph did better offline, but whether ANY
lesson block makes rounds land more often needs a live A/B, which is a human's
call.** `workers.sources.autocode.reasoning_bank: false` ships off, with
`reasoning_bank_mode: prior` as the mode that A/B should try first.

## What was built

- `scripts/automod/reasoning_bank.py`: a deterministic distiller over the
  automod ledger. Every clause a blocking `review` graded `partial` / `unmet` /
  `unsatisfiable`, and every `blocking` test-honesty finding, becomes a strategy
  item: a refusal cause (13 regex classes + `other`), that cause's lesson, the
  grader's own words, the round id, item id and date. A clause refused and later
  graded `met` in the same round becomes a `worked` item carrying the accepted
  fix, dated at the fix's review. At most 3 items per round, distinct causes.
  No model call: the review rows already are the structured findings, so the
  LLM distiller the item allowed for was not needed (and the result below says
  the distillation form is not the bottleneck anyway).
- Bank file `app.paths.REASONING_BANK_PATH` (`~/lloyd-data/automod/reasoning_bank.jsonl`),
  a cache rebuilt whole from the ledger at most hourly; pruned at 30 days, and a
  `failed` item superseded by a later `worked` one on the same (item, clause).
- Two retrieval modes: `similar` (TF-IDF over the item's title + clauses, top 3
  distinct causes) and `prior` (the 3 most frequent causes, newest example each).
  Both never return the item's own rounds (the re-offer banner carries those).
- `workers/sources/autocode.py::_strategy_block`, appended to the `{reoffer}`
  slot of the implement prompt, behind the default-off key.

Corpus: 161 strategy items from 2026-09-11..25 (other 26% of refusals after the
taxonomy; 239 informative refusals in 692 review rows).

## Method

- **Held-out set**: every round starting on/after 2026-09-17T00:00Z whose first
  blocking review refused on an informative clause note or a blocking test
  finding. n = **64** rounds. (77 rounds were refused after 09-21 alone; 37 of
  those were refused only on unverified seams / a "not graded" downgrade — a
  policy retired 2026-09-24 (`seams_block: never`), so they are not refusals the
  current loop would make and are excluded.)
- **No leakage**: each round sees only bank items dated strictly before its own
  `round_start`, pruned as of that instant (date filter first, so a later fix
  cannot supersede an earlier refusal from the future); a `worked` item is dated
  at the review that accepted the fix. The item's own earlier rounds are
  excluded in the headline arms (`sim_same` allows them).
- **Query**: the round's goal line + the item's acceptance clauses from the last
  triage before the round (no item body: the body on disk now carries findings
  written after the round).
- **Arms** (k = 3 each): `sim` (similarity), `prior` (frequency), `random`
  (3 random distinct-cause items), `sim_same`.
- **Relevance**: LLM judge on the primary (Qwen3.8-Flash-Next, thinking off,
  temperature 0, one YES/NO per (refusal, item) pair, 577 pairs, cached in
  `~/lloyd-data/eval/1489/judge_cache.jsonl`), under
  `flock -s primary.lock`. A deterministic proxy (item cause == a refusal's
  classified cause, `other` never matching) is reported beside it.
- **Precision** = relevant injected items / injected items (pooled; items are
  clustered 3 per round, so the Wilson interval is somewhat optimistic).
  **Recall** = rounds with ≥1 relevant injected item / rounds. Arm differences
  on recall: paired bootstrap over rounds (`eval/stats.py`, 10k resamples).

## Results (cutoff 2026-09-17, n = 64 rounds, 192 injected items per arm)

| arm | precision (judge) | recall (judge) | precision (tag) | recall (tag) |
|---|---|---|---|---|
| `sim` | 62/192 = 0.32 [0.26, 0.39] | 40/64 = 0.62 [0.50, 0.73] | 19/192 = 0.10 [0.06, 0.15] | 19/64 = 0.30 [0.20, 0.42] |
| `prior` | 84/192 = **0.44** [0.37, 0.51] | 50/64 = **0.78** [0.67, 0.87] | 33/192 = 0.17 [0.13, 0.23] | 30/64 = 0.47 [0.35, 0.59] |
| `random` | 45/192 = 0.23 [0.18, 0.30] | 34/64 = 0.53 [0.41, 0.65] | 20/192 = 0.10 [0.07, 0.16] | 19/64 = 0.30 [0.20, 0.42] |
| `sim_same` | 67/192 = 0.35 [0.29, 0.42] | 42/64 = 0.66 [0.53, 0.76] | 19/192 = 0.10 | 19/64 = 0.30 |

Paired recall differences (bootstrap 95%):

- `sim − prior`: judge **−0.16 [−0.28, −0.03]**, p = 0.018; tag −0.17 [−0.30, −0.05], p = 0.010.
- `sim − random`: judge +0.09 [−0.05, +0.23], p = 0.21; tag 0.00 [−0.14, +0.14].

Sensitivity, cutoff 2026-09-21 (n = 40): `sim` recall 22/40 = 0.55 [0.40, 0.69],
`prior` 31/40 = 0.78 [0.63, 0.88], `random` 20/40; `sim − prior` −0.23
[−0.40, −0.05]. Same ordering.

**Token cost** (primary tokenizer, rendered block): `sim` mean 532 tokens
(median 526, max 672, n = 64), `prior` mean 521 (median 511, max 638). About
0.2% of the 262k window, once per round.

`prior` injected the same three causes for all 64 rounds — `unpinned`,
`half_missing`, `unfalsifiable_test` — i.e. it is a static paragraph.

## Judge audit (hand, 40 random judged pairs from `sim` and `prior`)

- Agreement 32/40 = 0.80 [0.65, 0.90].
- The judge is lenient: of its 13 YES, I agree with 7 (0.54 [0.29, 0.77]); of
  its 27 NO, 25 (0.93 [0.77, 0.98]). Hand-judged relevance in the sample is
  9/40 = 0.23 [0.12, 0.38] against the judge's 13/40 — absolute precision and
  recall above are overstated by roughly a third. The disagreements are
  not arm-specific in kind (a generic "half missing" / "unpinned" lesson credited
  against an unsatisfiable clause or a logic bug), so the arm ordering stands;
  the tag proxy, which has no judge at all, gives the same ordering and a
  significant `sim − prior` gap.

## Reading

1. **Item similarity does not predict why a round will be refused.** `sim` is
   indistinguishable from random on both relevance measures and significantly
   worse than the frequency prior. Refusal causes are properties of how the
   loop writes tests and meets clauses (a test that cannot fail, a half left
   undone, a clause pinned by nothing), not of the item's subject matter — two
   recall items fail for different reasons, a recall item and a voice item for
   the same one. A better similarity model (qmd embeddings, an LLM distiller)
   would be optimising the wrong signal.
2. **The recurrent causes are few and stable.** Three causes cover a large share
   of refusals, and naming them named a held-out refusal's cause in ~78% of
   rounds by the lenient judge (~0.5 after the audit's correction). That is a
   static paragraph, and a static paragraph belongs in the implement template
   itself, where it is part of the cached prefix and needs no bank, rather than
   in a per-round injection. The prompt already carries rules close to all
   three; that the rounds still fail on them is itself evidence that telling
   the model again may not move the landing rate.
3. **What offline cannot answer**: whether a round shown the lessons avoids the
   refusal. "The lesson names the cause" is necessary, not sufficient.

## What a live A/B needs (owed by a human)

- Turn on `workers.sources.autocode.reasoning_bank` with
  `reasoning_bank_mode: prior`, alternating per round (e.g. by round parity or
  item id parity) for a week — ~100 rounds at the recent rate; the refusal-rate
  difference detectable at that n is roughly ±0.15, so a smaller effect needs
  longer.
- Outcomes from scorecard rows already tracked: first-review refusal rate,
  landings per round, items resolved per round-hour (rows 9/14), and the
  refusal-cause mix per arm (this module's `classify_cause` over the review
  rows) — the specific prediction is fewer `unpinned` / `half_missing` /
  `unfalsifiable_test` refusals in the on arm.
- The ledger needs an arm marker on `backlog_implement started` (not written
  today) so the analysis does not have to re-derive the assignment.

## Reproduce

    flock -s -w 7200 ~/.local/state/lloyd-automod/primary.lock \
        .venvs/lloyd/bin/python eval/run_reasoningbank_eval.py --judge --audit 40
    .venvs/lloyd/bin/python eval/run_reasoningbank_eval.py --cutoff 2026-09-21T00:00:00Z --judge

Artifacts: `~/lloyd-data/eval/1489/` — `summary-0917.json`, `summary-0921.json`,
`pairs-0917.jsonl`, `judge_cache.jsonl`, `audit_sample.json` (the hand verdicts
are in this file's audit section; disagreements were sample indices 10, 16, 21,
23, 25, 28, 32, 35). Tests: `tests/test_reasoning_bank.py`.
