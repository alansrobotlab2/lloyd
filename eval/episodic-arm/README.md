# Episodic raw-transcript arm (#675): verdict

**Reject routing episodic hits into recall or prefetch.** The instrument
(`eval/episodic_arm.py` + `eval/episodic_arm_search.mjs`) and this measurement
land. Nothing in production changes.

`results.md` / `results.json` hold the run: 81 queries, the baseline's scored
set from `nightly-20260924` (re-derived today; the item's nightly-20260909
artifact was lost in the 2026-09-22 wipe, so its overall figures are quoted
only). The corpus is 662 chat transcripts from 2026-08-22 to 2026-09-24. The
first 656 are the pre-wipe export, recovered from
`~/.cache/qmd/index.sqlite.bak-gemma-20260921`. They were re-embedded in a
scratch index with the daemon's own Qwen3 embed and rerank models and searched
lex+vec with lexMode or and rerank on.

## What it says

1. **The raw episodes carry nothing the vault docs' own text does not.** The
   scorer is text containment in both arms. With that, the episodic arm's
   entity_hit is 0.694 at every k [0.58, 0.79]. Recall's own top-10 documents,
   cut into the same windows at the same budget, score 0.736 / 0.792 / 0.806 at
   k = 0 / 2 / 5. The baseline misses its entity on 52 queries. The arm finds
   that entity on 21 of them (0.40, CI [0.28, 0.54]) once episodes that are
   talk about the eval itself are dropped, but the vault docs already hold it
   for 18 of those 21. That leaves **3 of 52** answered only by a transcript
   (autonomy-task-to-skill, cheaper-model-every-turn,
   numbers-differ-after-rebuild). At 4096 tokens per episode it is 2, and
   prose-only it is 1 or 0. That is under the item's 10-point bar.
2. **So the fact path's misses are not an extraction loss that raw logs
   repair.** On most baseline entity misses the answer string is already in
   the documents recall returns. The loss happens between those documents and
   the entity leg, which is the seed/fact scorer's problem (#1260, #969), not
   the corpus's.
3. **Eval-talk contamination is real.** Chats about the retrieval eval itself
   name every label. Dropping episodes that carry eval markers takes the arm
   from 0.694 to 0.597. Any future session-corpus eval has to control for it.
4. **The adjacency window only matters on prose.** With tool output included,
   a 1024-token episode fits about 1.7 turns, so k=2 and k=5 change nothing
   (0.694 at every k). Prose-only (user and lloyd lines), the window is the
   whole effect: entity_hit goes 0.45 → 0.55 → 0.60 at k = 0 / 2 / 5 (n=60
   eligible; 21 queries are zero-gold there). Even at k=5 it stays below the
   control's 0.82.
5. **Cost.** Searching the 662-document collection takes p50 824 ms and p95
   961 ms. The baseline's full recall averages 541 ms. Ten episodes cost about
   8.7k tokens at k=2, against 5.8k for the same windows over vault docs.

| variant | zero-gold | arm k=0 / 2 / 5 entity_hit | clean k=2 | control k=2 | miss-subset adds not in control |
|---|---|---|---|---|---|
| all kinds, 1024 tok (committed) | 9 | 0.694 / 0.694 / 0.694 | 0.597 | 0.792 | 3 |
| prose only, 1024 tok | 21 | 0.450 / 0.550 / 0.600 | 0.483 | 0.800 | 1 |
| all kinds, 4096 tok | 9 | 0.750 / 0.764 / 0.764 | 0.667 | 0.806 | 2 |
| prose only, 4096 tok | 21 | 0.450 / 0.550 / 0.600 | 0.483 | 0.817 | 0 |

The sensitivity rows re-score the committed search output with `--kinds user
lloyd` and/or `--budget 4096`; retrieval is identical.

## Found on the way (for a person)

- **The live `sessions` qmd collection holds 6 documents.** It held 656 before
  the 2026-09-22 data-home wipe, and prefetch's `sessions` leg has searched the
  6 since. The user-session JSONs before 09-22 are gone too; 9 remain in
  `~/lloyd-data/sessions`. The last copy of those conversations is the
  `content` table of `~/.cache/qmd/index.sqlite.bak-gemma-20260921` (and
  `…bak-20260919-203417`). Do not delete those backups. Restoring means writing
  the 656 documents back to `~/lloyd-data/_pipeline/vault-derived/sessions/`
  for the watcher to re-embed. That is a production data write, left to a
  person.
- `VAULT_SEGMENTS` still omits `sessions`. On this evidence, leave it out: the
  corpus adds no answers the recall pool lacks, it costs tokens, and
  contamination risk comes with it.

## Re-running

    .venvs/lloyd/bin/python eval/episodic_arm.py \
        --baseline ~/lloyd-data/eval/baselines/nightly-<date>.json \
        --corpus-dir <exported session markdown> --work-dir <scratch> \
        [--search-json <scratch>/search_out.json] [--kinds user lloyd] [--budget N]

Run the search half under `retrieval.lock`. It loads the embed and rerank
models on GPU 0 in its own process, and the index and embed step took about 3
minutes.
