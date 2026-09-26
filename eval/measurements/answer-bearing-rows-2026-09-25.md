# #1491: answer-bearing sentences as djev's ranked row — measured, rejected

**Verdict: REJECT, for now.** Two pinned runs both point up (MRR +0.026 and
+0.035, NDCG +0.020 and +0.034), but neither interval clears zero, doc_hit does
not move, and the change is not faster (it adds a disk read per pool row). By the
house rule a change that is not faster needs a gain clear of the paired interval,
and this is below the gold set's MDE (~+0.09 MRR). Worth re-measuring on P0
(LloydMemEval, #1480) once that set exists. The "use" half of the item (what the
model is handed) needs P0 too and was only sized here.

## What was measured

Production since #1467: djev ranks `title + strip_qmd_snippet(snippet)`, cut at
160 characters. qmd's snippet is a ~4-line window placed where the match sits.

The arm (`eval/answer_bearing_rows.py`, prototyped by swapping
`vault._djev_doc_text` in the eval process; no production code changed) reads
the ~1,200-char window around that match from the note on disk (qmd's own rerank
window width), splits it into sentences, keeps the sentences carrying the most
distinct query terms, and hands djev `title + those sentences in document order`
at the **same 160 characters**. It is a lexical, Provence-style pruner with no model.
If lexical selection does not move the ranking, a learned pruner would not be worth
a GPU-0 slot either.

Pinned corpus (VACUUM INTO of the live qmd index on :8183, production daemon
settings), pinned fact tree and store, 86-query dev set, nightly scorer. Arms per
query, interleaved: `control`, `answer_bearing`, `control_b` (production again).
djev replay is OFF, because the arm changes djev's input, so every row is a fresh
read and `control_b` measures the ranker's own noise. The pool, documents and
facts are identical across arms, so doc_hit and doc_recall can move only through
what falls off the 20-row cut.

## Results (n=86, doc_recall n=79; Δ vs control, paired bootstrap 95%)

| metric | run 1 Δ [CI] (W/L) | run 2 Δ [CI] (W/L) | noise (control_b Δ), run 1 / run 2 |
|---|---|---|---|
| mrr_doc | +0.026 [−0.026, +0.081] (23/19) | +0.035 [−0.022, +0.094] (24/21) | −0.005 / +0.008 |
| ndcg10 | +0.020 [−0.021, +0.062] (25/18) | +0.034 [−0.009, +0.077] (26/20) | +0.001 / +0.013 |
| doc_hit_rate | +0.012 [−0.035, +0.070] (3/2) | 0.000 (1/1) | −0.012 / +0.012 |
| doc_recall_avg | +0.025 [−0.027, +0.082] | −0.002 [−0.044, +0.042] | +0.002 / +0.002 |

Control MRR was 0.273 and 0.277. Run 1 had the #1456 topics merge on (config.yaml
had just turned it on), which moves only the fact leg: entity metrics differ
between the runs, and the document metrics do not. Run 2 has the merge off.

## Rows (run 1; 2,316 rows ranked)

- 2,259 of 2,316 rows (97.5%) resolved to a file on disk. The rest fell back to
  the snippet (qmd-normalised filenames).
- Today's stripped snippet: mean 242 chars, p50 294.
- The window read: mean 1,312 chars.
- The window pruned to answer-bearing sentences with a 600-char budget: mean 365,
  p50 361. This is the size a model-facing "pruned snippet" would be, about 1.5×
  today's snippet, chosen by query terms rather than position. Whether the model
  *uses* it better is P0's "use" metric and was not measured.

Latency was not separable in this design: the three arms run in order, and each
later arm reads qmd's and djev's warm caches. Reading about 32 small windows per
recall costs single-digit milliseconds, and nothing else changes.

## Reading

The direction matches #1467 (strip the header, and djev ranks better when the
row carries document text), but the lexical selection adds at most about 0.03
MRR on top of the strip. A learned pruner (Provence on GPU 0) would add a model
to the card #1495 is about, to chase a gain this set cannot resolve. Measure it
on P0 first.

Artifacts: `~/lloyd-data/eval/1491/run.json`, `run2.json`.
