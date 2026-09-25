# Summary arms: legacy vs persisted (D2) and memory flush (P3), 2026-09-25

**Verdicts: keep `compaction.persist_summary: false`; keep `compaction.memory_flush.enabled: false`.**

Runner: `eval/run_compaction_recall_eval.py` (conversation shape, D2e), late
probe, `--sizes 186000,192000 --depths 0.1,0.3,0.5 --sessions 6`, 12 paired
sessions per arm, primary (Qwen3.8-Flash-Next, FP8 KV) with the worker pool
paused. All 36 rows valid: the summarize layer fired on every one, no drops,
no errors. Raw rows: `eval/baselines/compaction-summary-arms-2026-09-25.json`.

| arm | distinctive | ambiguous | summary keeps codename / port now / old port | median TTFT (probe) | turn-start wall vs legacy | tool calls |
|---|---|---|---|---|---|---|
| summary_legacy | 11/12 | 10/12 | 11 / 10 / 7 | 4.1 s | — | 4.3 |
| summary_persisted | 8/12 | 8/12 | 9 / 9 / 9 | 11.7 s | +42 s (CI 30–53) | 5.7 |
| memory_flush (legacy + flush) | 12/12 | 11/12 | 12 / 11 / 8 | 4.2 s | ≈0 | 4.2 |

Paired differences vs legacy (bootstrap 95% CI):
- persisted: distinctive −0.25 (−0.58, +0.08), ambiguous −0.17 (−0.58, +0.25) —
  not significant, but the direction is a loss; wall +15 s and summed TTFT
  +19 s are significant (three ~55k folds on the first compaction).
- memory_flush: +0.08 on both (not significant); the flush turn saw the
  planted fact in 8 of 12 sessions and wrote it to memory in only 1
  (distinctive) / 3 (ambiguous) of those, so the small gain is not the flush.

## Reading

D2's persisted incremental summary is not non-inferior on recall on its
first compaction and costs ~40 s more at turn start. Its intended win —
reusing one stable summary on every later turn instead of re-summarising the
whole older block (the 120 s / cache-busting loop the review found) — is a
cross-turn effect this single-probe eval does not exercise. The record, the
folds, `/compact` (D11, which applies a manual record regardless of the
flag) and the telemetry stay; the default stays off until a multi-turn arm
(two consecutive over-threshold turns: second-turn TTFT and recall) shows
the reuse win without the recall loss. Worth trying there: fold into the
legacy 9-section format rather than the 5-section one, since the section
format is the obvious difference in what the summary kept.

P3's flush adds nothing measurable here and rarely persisted the planted
facts; it stays off.
