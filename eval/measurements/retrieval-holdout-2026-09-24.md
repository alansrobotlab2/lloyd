# Retrieval holdout leg — first reading (#1412, 2026-09-24)

One run of each leg, back to back, against today's live stack (qmd daemon :8181,
djev ranking the recall, production knobs: `matches_production_defaults: true`,
`corpus_ok: true` on both), under `retrieval.lock`, from the #1412 worktree with
`LLOYD_KG_DB` / `LLOYD_FACTS_ROOT` on the live store. The worker pool was paused.
Only aggregates were read: per-query output went to scratch files nobody opened,
and no holdout id appears below or anywhere outside the holdout file and its
manifest.

| metric | dev (n=86) | 95% CI | holdout (n=27) | 95% CI |
|---|---|---|---|---|
| entity_hit_rate | 0.337 | 0.246–0.442 | 0.259 | 0.132–0.447 |
| entity_recall_avg | 0.384 | 0.276–0.498 | 0.361 | 0.167–0.583 |
| fact_entity_recall_avg | 0.374 | 0.268–0.485 | 0.306 | 0.111–0.528 |
| doc_hit_rate | 0.651 | 0.546–0.744 | **0.963** | 0.817–0.993 |
| doc_recall_avg | 0.547 | 0.452–0.638 | **0.926** | 0.833–1.000 |
| mrr_doc | 0.331 | 0.248–0.418 | **0.647** | 0.503–0.785 |
| ndcg10 | 0.369 | 0.288–0.452 | **0.709** | 0.583–0.826 |
| latency_ms_avg | 583 | — | 752 | — |
| errors / empty-doc queries | 0 / 0 | | 0 / 0 | |

Wall clock: dev 102 s, holdout 22 s (process start included).

## What it says

- **The two legs are not the same difficulty, and that is fine for what the leg
  is for.** The entity leg reads the same within the CIs. The document leg is far
  easier on holdout: 26 of 27 questions put an expected note in the returned set.
  The holdout questions each target one distinctive note, where the dev set carries
  multi-hop and "hard" questions whose labels span several notes. The labels were
  fixed before this run and were not changed after it (the reserve rule); a harder
  tranche is the next tranche's job, not an edit to this one.
- **Consequence for the predicate.** `overfit_suspected` needs a dev GAIN and a
  holdout LOSS. A document-leg loss has plenty of room on holdout (from 0.963 down),
  so the leg can see the thing it is for; what it cannot show well is a holdout
  document *gain*, so `transfer_gap` on doc metrics will read positive whenever a
  change improves dev documents — read the gap next to `delta_holdout`, never alone.
- **Floors.** At n=27 the resolution term of `effective_floor` is 1/27 + 0.001 =
  0.038, so a holdout loss must be at least about one question to count; at n=86
  the dev term is 0.013. The published noise floor (`eval-noise.json`) has σ = 0 on
  every armed metric under replay, so resolution governs both legs there; under a
  fresh ranker 3σ ≈ 0.031 on `ndcg10`/`mrr_doc`, which governs dev but still not
  holdout.
- **Cost.** Two holdout arms add ~45 s to a paired check that already runs two to
  three 86-question arms, and the holdout arm's latency is report-only (the paired
  ceiling reads the dev arm).

## Decisions on the item's human clauses (delegated for this sweep)

1. **14-day outcome** — not observable in a round; it needs ~14 days of real
   promotions with the leg live. Step 6's retrospective re-score of past landings
   was NOT run: it needs a pinned second qmd daemon per promotion, which this
   sweep may not boot. Until the leg has caught or cleared one real promotion the
   threshold stays the check's own floors and is not tuned.
2. **Nightly vault edits** — decided: none. The nightly writes per-query rows into
   `~/lloyd-data/eval/baselines/`, which its skill and the trend audit read, so a
   nightly holdout run is precisely the leak the reserve rule forbids; and the
   transfer gap is a property of a change, which only the paired check measures.
   `run_eval.py` refuses a `nightly*` label under the holdout corpus to keep it so.
3. **Latency budget** — decided: exempt by construction. `latency_over_budget`
   reads the dev arm's average only; the holdout arm's average is recorded on the
   check's `holdout` block as a report. `LATENCY_BUDGET_MS["paired_check"]` needs
   no re-derivation for the holdout leg.
4. **delta_dev / delta_holdout on the ledger** — implemented and pinned with
   stubbed arms (`tests/test_automod_regression.py`); the first live record comes
   with the first promotion after this lands.
