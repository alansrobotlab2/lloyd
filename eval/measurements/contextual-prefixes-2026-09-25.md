# #1494: LLM-written situating context — measured offline on the full corpus; rejected

**Verdict: rejected — not worth the fork work.** The document-level proxy was
first measured on the 1,564-doc #1493 sub-corpus (MRR +0.054 [−0.007, +0.115],
p=0.087; section "Sub-corpus run" below). Re-run on the **full searched corpus**
(every active document of the eleven collections `_vault_recall` searches, 5,513
distinct documents, ~29.3k chunks) the gain shrinks to MRR **+0.011
[−0.043, +0.064]**, p=0.70 on the 86-query dev set, and every document metric
**falls** on the 27-query holdout (MRR −0.069, NDCG@10 −0.088, doc_hit −0.074).
Nothing clears its interval on either leg. The sub-corpus result was the
distractor-thin corpus flattering the change, not a signal the fork work would
buy. The real (per-chunk) technique was not built — the fork is human-landed —
but the cheap document-level half, which the item and the sub-corpus write-up
proposed trying first, gives no measurable gain at full scale.

## Full-corpus run (2026-09-25 evening)

`eval/contextual_full_corpus.py`, stock fork build (`~/lloyd/qmd/dist/cli/qmd.js`
as production runs it), nothing in the fork, the live index, `index.yml` or
`evalpin.yml` touched:

1. **generate** — the primary wrote a ≤40-word context per distinct document
   (same prompt, 6,000-char cap, greedy, thinking off, as `contextual_titles.py`),
   under `primary.lock` shared, 6 workers. 1,558 contexts reused from the
   sub-corpus run (same content hash), 3,962 new in 836 s (1.27 s/call), 0 errors,
   mean 26.1 words. All 5,513 documents covered.
2. **prepare** — the live index `VACUUM INTO` two scratch files under
   `~/lloyd-data/eval/1494/full/idx/` (read-only on the source); both deactivate
   every collection the recall does not search (`subliminal`, `sessions`,
   `autonomy-runs`), leaving 5,546 active rows / 5,513 distinct documents. Arm
   **ctx** rewrites each document's first `#`/`##` heading as `<heading> —
   <context>` (5,510 of 5,513; 3 hashes changed on the live index between the two
   steps) and re-fires the FTS trigger, so every chunk's embed title and the lex
   body carry it — identical to the sub-corpus proxy.
3. **embed** — both arms re-embedded from scratch with production's
   Qwen3-Embedding-0.6B (`INDEX_PATH` + `QMD_CONFIG_DIR` pointed at the scratch
   file and a scratch copy of evalpin's models block), under `regression.lock`:
   plain 29,283 chunks in 1,395 s (21.0/s), ctx 29,647 chunks in 1,468 s (20.2/s;
   the context adds 364 chunks).
4. **score** — each arm served by its own daemon (production's command line and
   environment, `:8187` / `:8188`), warmed with production's recall request, and
   scored by the unmodified `eval/run_eval.py` through a `services.qmd` config
   overlay, pinned fact tree (`pin-2026-09-25`), one code root,
   `--no-counterfactual`. djev under one replay file per leg anchored on
   `plain`; a third arm re-runs `plain` replaying the anchor (control_b). Holdout
   leg via `LLOYD_EVAL_HOLDOUT_LEG=1`; its per-query run records and replay file
   were deleted once aggregated (#1412's reserve rule).

### Results — dev set (n=86; doc_recall n=79), paired bootstrap 95% CI

| metric | plain | ctx | Δ [95% CI] | p | W/L |
|---|---|---|---|---|---|
| mrr_doc | 0.310 | 0.320 | **+0.011 [−0.043, +0.064]** | 0.70 | 22/17 |
| ndcg10 | 0.369 | 0.375 | +0.006 [−0.047, +0.058] | 0.81 | 23/19 |
| doc_hit_rate | 0.663 | 0.651 | −0.012 [−0.070, +0.047] | 0.84 | 3/4 |
| doc_recall_avg | 0.589 | 0.610 | +0.021 [−0.042, +0.089] | 0.54 | 8/6 |

control_b (plain replayed) reproduced plain exactly on every metric (Δ 0.000),
so the scratch daemons and the pipeline are deterministic under replay.

### Results — holdout (n=27), aggregates only

| metric | Δ ctx − plain [95% CI] | W/L |
|---|---|---|
| mrr_doc | −0.069 [−0.206, +0.066] | 6/8 |
| ndcg10 | −0.088 [−0.202, +0.022] | 5/10 |
| doc_hit_rate | −0.074 [−0.185, 0.000] | 0/2 |
| doc_recall_avg | −0.037 [−0.167, +0.056] | 2/2 |

Overfit flag (#549's strict form — dev gain clear of its interval AND holdout
loss clear of its own): **not raised**, because no dev gain clears its interval.
The signs disagree on MRR, NDCG@10 and doc_recall (dev up, holdout down), which
is the direction that argues against the change, not for it.

**The ranker.** djev re-ranks every recall, and the context changes the 160-char
candidates it reads for most rows, so all 86 dev and 27 holdout rank requests in
the ctx arm were fresh djev draws (none replayed from the anchor). djev's own
run-to-run variance is therefore inside the ctx-vs-plain comparison; it can only
widen these intervals, not rescue a gain. Latency: recall mean 583 ms vs 582 ms
(dev), 697 vs 694 ms (holdout) — no query-path cost, as before.

### Why the sub-corpus read higher

The sub-corpus kept every narrow gold document plus a 25% sample of the rest, so
each query had about a quarter of the real distractors. Its control MRR was 0.394
against 0.310 here. The likely reading (not separately tested): a context
sentence that separates a gold note from a few hundred neighbours does much less
against a few thousand that got equally templated summaries — 395 of the 1,446
backlog contexts open with the same words, "Backlog item for Lloyd, Alan's local
AI agent, …". The paired delta fell from +0.054 to +0.011 on the same 86 queries.

### Cost of this run

Primary: 836 s of generation (plus 518 s in the sub-corpus run whose contexts
were reused). GPU 0: 2,863 s of embedding plus ~3 min of two scratch daemons
while scoring, all under `regression.lock`. Disk: 1.02 GB of scratch indexes,
deleted after scoring; 5.3 MB kept under `~/lloyd-data/eval/1494/`
(`contexts-full/contexts.json` — the 5,513 contexts, keyed by content hash —
`full/score.json`, `full/prepare.json`, `full/embed-*.json`, run logs).

## Sub-corpus run (earlier the same day)

### The proxy (no fork edit)

The fork already prefixes every chunk's embedding text with the document's title,
and takes that title from the first `#`/`##` heading (`extractTitle`). So, on a
scratch copy of the #1493 sub-corpus (1,564 documents, 8,909 chunks; see
`embedding-4b-side-index-2026-09-25.md`):

1. `eval/contextual_titles.py generate` asked the primary, once per document, for
   a ≤40-word context situating it in the vault (Anthropic's prompt, adapted;
   document text capped at 6,000 chars, thinking off, greedy).
2. `prepare` rewrote each document's first heading as `<heading> — <context>` (or
   prepended one) in the scratch index's `content` table and re-fired the FTS
   trigger, so both the vector leg (every chunk) and the lex leg see it.
   `documents.title` was left alone, so djev's row keeps today's title.
3. Re-embedded with production's 0.6B (`embed_side_index.py embed --name subctx`),
   compared against the same sub-corpus without contexts (`sub06`).

Differences from the technique, stated: the context is per **document**, not per
chunk (every chunk of a note gets the same sentence); and a snippet cut near the
top of a note can now include the context line, so a few djev rows saw it.

### Results (86-query dev set, pinned facts, real `_vault_recall`, djev ranking)

| metric | control (sub06) | context Δ [95% CI] (W/L) | noise (control_b Δ) |
|---|---|---|---|
| mrr_doc | 0.394 | **+0.054 [−0.007, +0.115]** p=0.087 (28/16) | −0.012 |
| ndcg10 | 0.450 | +0.040 [−0.016, +0.094] (32/22) | −0.016 |
| doc_hit_rate | 0.756 | +0.023 [−0.035, +0.081] (4/2) | −0.012 |
| doc_recall_avg (n=79) | 0.710 | +0.017 [−0.038, +0.078] (5/8) | −0.023 |

The query path costs nothing: query embed p50 is 8 ms against 7 ms, and the
qmd request p50 is 52 ms against 60 ms. The context rides in the index, not in
the query.

### Cost estimate made then (full searched corpus: 5,511 documents, 29,261 chunks, 53.4 M chars)

| step | measured basis | estimate |
|---|---|---|
| document-level contexts (the proxy) | 1,558 docs in 518 s, 6 workers on the primary, 2.0 s/call, 26 words each | **~31 min** of primary time |
| per-chunk contexts (the technique) | same call cost, one per chunk | **~2.7 h** at 6 workers (djev raw generation, ~140 tok/s, ≈ the item's 10 h for 39k) |
| re-embed with 0.6B | 20.9 chunks/s on the shared 3090 (this sweep) | ~23 min |
| fork work (human-landed) | a `context` column beside the chunk, prefixed in `formatDocForEmbedding` and the FTS body, and an idle-time job to fill new chunks | build + fork suite + commit on `lloyd` |

### What it asked a human (answered by the full-corpus run above)

1. Whether to try the cheap version first: the document-level context needs no
   new column, since it can live in front matter or a sidecar the fork's
   `extractTitle` reads. It measured +0.054 MRR here.
2. Whether to re-measure on the full corpus and on P0 (#1480) before building.
   86 queries cannot resolve +0.05 (MDE ≈ +0.09).

Artifacts: `~/lloyd-data/eval/1494/` (contexts.json, generate.json, eval.json).
`contextual_titles.py prepare` imports `eval/embed_side_index.py` (#1493's
harness, on the `hand/sweep-0925c` branch); `contextual_full_corpus.py` uses only
`contextual_titles`' prompt, cap and heading rewrite and runs on its own.

## What is left

Only the per-**chunk** variant is untested: a context written for each chunk
rather than one sentence shared by every chunk of a note. It is ~2.7 h of
primary time plus the fork column and is still human-landed. With the
document-level half flat on dev and negative on holdout at full scale, this
measurement does not justify asking for that fork work. Re-running it is
`contextual_full_corpus.py generate|prepare|embed|score|clean` (the contexts are
kept, so only `prepare` onward costs anything).
