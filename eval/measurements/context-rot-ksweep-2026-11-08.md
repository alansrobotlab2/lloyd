# Context rot: the k axis on the live primary, 2026-11-08

Artifact (the bytes this doc reports, and their only two homes): `/home/alansrobotlab/lloyd-data/ksweep-2026-11-08/context-rot-ksweep-2026-11-08.json`, sha256 `e463040b404ed7b743dc55e3595feb9a0a053656d776e5f89f1cb2767980459f`, archived copy for history at `~/obsidian/backlog/data/2026-11-09.3923-context-rot-ksweep-2026-11-08.json`; provenance commit `2d2aaba3` (head of `automod/SM_20261108_050722`, not an ancestor of `main` — that commit also edited `.gitignore`, so only this doc's text was taken from it). `meta.expect_model` `Qwen3.8-Flash-Next`, `complete: True`. **No copy of the JSON is ever added to this repo, and none is owed here**: the repo's ignore rules cover both the raw-artifact directory and `*.json` at large, and `.gitignore` is a path no automod round may write, so the runtime bytes named first above and the vault copy are the artifact's whole history; whether the raw file should get repo history anyway is a ruling owed on #3989, not a step this doc skipped.

`decision.action: inconclusive` and `decision.partial_grid: true` in the appendix are **by construction for any k-only run**, not a finding of this one: `recommend()` sets `"partial_grid": not full_grid` (`eval/run_context_rot_eval.py:1239`) and answers `action="inconclusive"` as soon as any shape's `L_star` is `None` (`:1241`), and the k arms never populate the length grid that prefix rule reads. So `L* (repo) = None` and `first fail: null` below are placeholders recording "there was no length grid to decide on", and the measurement this run produced is `summary[shape].k_curve` and nothing else.

Grid actually run: shapes ['repo', 'session'], arms ['distractK', 'dissimilarK'], lengths [50000, 150000], depths [0.5], ks [0,1,2,4], seeds [1, 2, 3] -> 96 cells. The k axis in canonical compact form, which is how `eval/check_doc_matches_json.py` matches it against `meta.grid.ks`: [0,1,2,4].
**The curve is TRUNCATED** -- this is the decision tranche, not the full k sweep: depths [0.5] and ks [0, 1, 2, 4] only, so the k axis stops at k=4 and no depth other than the sweep's own is covered.

## What the instrument was built to answer

#3801 planted k look-alike values and asked how accuracy moves as k grows at a FIXED context length, where the 2026-09-24 grid only ever varied length. The two arms are `distractK` (k look-alikes planted, depth fixed) and `dissimilarK` (k look-alikes that are NOT near-duplicates, as the control). The curve is `summary[shape]["k_curve"][arm][length]["curve"]`, one row per k, four outcome classes per row.

## Headline per arm and length

`largest k with correct >= 0.9` and `first k where hedged > 0.1`, read straight out of `k_curve` -- the two figures the item asked for. `None` means no k in the range reached it, which is a measurement of this grid, not an absence of measurement.

| shape | arm | length | largest k, correct >= 0.9 | first k, hedged > 0.1 |
|---|---|---|---|---|
| repo | distractK | 50,000 | 4 | None |
| repo | distractK | 150,000 | 4 | None |
| repo | dissimilarK | 50,000 | 4 | None |
| repo | dissimilarK | 150,000 | 4 | None |
| session | distractK | 50,000 | 4 | None |
| session | distractK | 150,000 | 4 | None |
| session | dissimilarK | 50,000 | 4 | None |
| session | dissimilarK | 150,000 | 4 | None |

The per-k rows behind every line above are in the appendix, written by the runner's own `render_md` from this same JSON, so they cannot drift from it. `eval/check_doc_matches_json.py` (run from the repo root, see How to re-run) recomputes each figure in this table and each appendix row from the JSON and fails on any that disagree.

---

## Why this run's accuracy is NOT comparable with `context-rot-2026-09-24`

**Not comparable. This is a rule, not a hint to read the numbers side by side** -- and the reason is the grader, not the engine:

1. **A third grade class the pin's grid never had.** `72341888` added `hedged`, so `correct` here is a 3-way split where the pin's is 2-way. the grids are different objects: the pin ran shapes ['repo', 'session'], conditions ['single', 'multi3', 'distract4'], lengths [50000, 100000, 150000, 200000, 240000], depths [0.1, 0.3, 0.5, 0.7, 0.9] for 450 cells and its `summary` carries no `k_curve` and no `hedged` column at all (checked by reading it), so there is no hedge in it to subtract.
2. **A different answer budget.** The k arms run under `K_ARM_MAX_TOKENS = 128`; the pin recorded `max_tokens = 64` and the module default `MAX_TOKENS` is 64. 72341888 raised the k-arm budget so a hedge is "a decision and not a truncation" rather than a cut-off sentence, and its own message measures what the old budget hid -- hedged rates of 0.36-0.40 by truncation alone -- so an arm's `correct` here and the pin's `correct` there are not the same experiment.
3. **Engine: unchanged.** the served model is **the same engine**: `Qwen3.8-Flash-Next-nvfp4`, root `Inferact-Qwen3.8-Flash-Next-NVFP4`, `max_model_len 262144` -- all three equal to the pin's `meta.engine`, so the non-comparability below is the grader's, not the model's. No engine change ever came; a reader waiting for one is waiting for nothing.

A `hedged` reply counts as **not-a-hit**, exactly like `wrong` and `miss` -- 72341888's stated reason for choosing it is that "hedged and wrong are both not-a-hit, so `correct` stays the same measure across the hedge change". So the DIRECTION of the difference is knowable even where the size is not: a hedge the pin's grid filed under `wrong` or `miss` is filed under `hedged` here, and neither counts. What is not knowable is how much of any gap between the two runs that accounts for.

## The hedge rule, stated so a later reader can re-decide it

Per fact, code only (`grade()`, `eval/run_context_rot_eval.py:705-749`):

- `correct` -- the answer names the accepted value and no wrong value;
- `wrong` -- the answer names a wrong value;
- **`hedged` -- the answer names the accepted value AND at least one planted look-alike** (`run_context_rot_eval.py:742-744`). It has not answered; it is the model reporting the conflict rather than resolving it, and it is counted as not-a-hit;
- `miss` -- anything else, including silence.

In a `distractK` cell the k planted look-alikes are among the candidate wrong values, so the `hedged` rate IS the k response: naming the accepted value plus one of k planted values. `first_k_hedged_over_0.1` is the smallest k whose hedged rate exceeds `HEDGE_FLAG = 0.1`; `largest_k_correct_at_least_0.9` is the largest k whose correct rate reaches `CORRECT_KEEP = 0.9`. Both constants are the runner's (`:1119-1120`). If a later reader wants a different cliff -- hedged over 0.2, or hedged counted as half a hit -- these rows are recomputable from the per-k counts without re-running anything.

## The window this ran in

Pool status read before: `{'running': True, 'paused': False, 'paused_by': [], 'paused_since': None, 'in_flight_count': 1, 'holder_kind': None}`

Pool status read after: `{'running': True, 'paused': False, 'paused_by': [], 'paused_since': None, 'in_flight_count': 1, 'holder_kind': None}`

Pause lines the run recorded into `meta.pool`: `pool paused by this run; lease refreshing every 60 s: while this process is alive the hold stands across a backend restart, and once it is not the next pool to start reconciles the halt (#2498); pool resumed`

**The hold was not clean, and the log says so.** The first launch of this window was killed with its process group roughly 11 minutes in, before the runner reached `write_doc` -- the runner persists only at the end (`eval/run_context_rot_eval.py:1667`), so a killed run leaves no artifact, and because its `PoolPause.__aexit__` never ran the **machine hold stayed held after the process was gone** (`paused: true, paused_by: ['operator'], holder_kind: 'machine'`, pool stalled with nothing running). It was released by hand with `curl -X POST localhost:8080/api/workers/pause -d '{"paused": false}'`, the command the runner's own timeout branch prints for exactly this case. The second launch is the run this artifact records, and its BEFORE reading below is taken with the pool genuinely unpaused.

Both readings are printed rather than an asserted transition, because the transition is not always available: `PoolPause.__aenter__` takes no hold at all when the pool is already paused by someone else, and `__aexit__` deliberately REFUSES to resume when `paused_by` still names `automod`, leaving the pool halted with an operator instruction (`eval/run_context_rot_eval.py:931-933, :950-972`). A launch inside an automod landing window would therefore show "already paused" and no release; that is the case this section exists to make visible instead of quietly passing.

## Real turns beside the curve, and the cliff

The field here is NOT named `k`: `eval/prefetch_conflict_count.py` reports the per-turn count of injected items as `items_histogram` (and `values_histogram`, distinct values of one entity per turn), and those two are the k analogue. Naming the field is the point -- a reader who searches the report for `k` finds nothing and may conclude the measurement was never taken.

| sample | turns | resolution_coverage | items_histogram | max | turns with conflict |
|---|---|---|---|---|---|
| newest 120 session files | 107 | 0.0 | {0: 107} | 0 | 0 |
| whole 14-day store | 4562 | 0.0054 | {0: 4546, 1: 6, 3: 10} | 3 | 10 |
| `iv*` sessions, 2026-09-22..2026-11-06 | 115 | 0.2701 | {0: 81, 1: 6, 3: 28} | 3 | 28 |

The newest-120 row is the coverage floor the item cites, re-measured the same day: `resolution_coverage 0.0` -- 0 resolved of 279 injected items, items_by_section `{"vault-context": 279}`. Its `{0: 107}` is therefore a floor, not a rate: every one of those turns scored zero conflicts because no injected value could be matched back to a fact, not because turns are conflict-free.

The wide row is the same counter over every session in the 14-day window (11974 files read). It is better than a newest-N sample but still not interactive: coverage 0.0054 because that store is autonomy/autocode/review traffic, whose injected `vault-context` prose carries no `<facts>` line to resolve against. Its max of 3 is a weak upper bound, not the interactive distribution.

The interactive row is the one that answers the question. There is no `_chat` session family on this machine (checked: 0 files match `_chat`/`_mc_`/`voice`/`_cli`); the turns that carry `<facts>` lines are the `iv*` (inner-voice) transcripts, 119 of them spanning 2026-09-22 to 2026-11-06, sampled by pointing the counter at a directory of symlinks to exactly those files (`--sessions-dir`, so the newest-N ordering cannot re-dilute it). Coverage 0.2701 -- 158 resolved of 585 items, items_by_section `{"facts": 158, "vault-context": 427}` -- and the distribution is `{0: 81, 1: 6, 3: 28}` with 28 of 115 turns holding a conflict.

**Cliff verdict, one line:** no hedged cliff was measured: no arm's hedged rate exceeded 0.1 anywhere in the k range that ran ([0, 1, 2, 4]), while real interactive turns reach k = 3 -- so on this evidence the disambiguation-or-collapse edit is not indicated, and the reading is censored two ways (the k axis stops at 4, and an interactive max is never the tail).

## Engine block this run read, and how to re-run it

`meta.engine` as recorded by the run: `{"served": "Qwen3.8-Flash-Next-nvfp4", "root": "/home/alansrobotlab/lloyd/agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4", "max_model_len": 262144, "cache_config": {"cache_dtype": "fp8", "kv_cache_size_tokens": 844969, "block_size": 3200}}`; `meta.expect_model` `Qwen3.8-Flash-Next`. The pin's block for comparison: `{"served": "Qwen3.8-Flash-Next-nvfp4", "root": "/home/alansrobotlab/lloyd/agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4", "max_model_len": 262144, "cache_config": {"cache_dtype": "fp8", "kv_cache_size_tokens": 844969, "block_size": 3200}}`.

The window, in one command (it holds the machine lease itself via the runner's `PoolPause`, and takes NO hold if the pool is already paused by another holder — launch it when no automod round is landing). `--no-warm` matters and is not in the item's command: warm replay is ON by default, so without it the same 168 cells issue 336 requests, and the 30-45 minute sizing in the item is for 168. The counter runs separately and needs the real data root, since launching anything from a worktree silently anchors `app.paths` at the worktree's empty data dir:

```
cd ~/lloyd && .venvs/lloyd/bin/python eval/run_context_rot_eval.py \
    --k-sweep --no-warm --out <artifact>.json          # 168 cells, cold only
# the item's fallback tranche needs --depths 0.5 or it builds 480 cells, not 48:
cd ~/lloyd && .venvs/lloyd/bin/python eval/run_context_rot_eval.py \
    --conditions distractK,dissimilarK --ks 0,1,2,4 --lengths 50k,150k --depths 0.5 \
    --no-warm --dry-run                                 # grid: 48 cells
LLOYD_DATA=/home/alansrobotlab/lloyd-data \
    .venvs/lloyd/bin/python eval/prefetch_conflict_count.py --days 14 --json
# interactive sample: the counter has no --filter for session family, so point it at one
ln -s ~/lloyd-data/sessions/2026*iv*.json /tmp/iv/ && \
  LLOYD_DATA=/home/alansrobotlab/lloyd-data .venvs/lloyd/bin/python \
    eval/prefetch_conflict_count.py --sessions-dir /tmp/iv --days 30 --json
```

Re-checking this doc against its artifact — `eval/check_doc_matches_json.py` recomputes the headline table and every appendix row from the JSON and exits non-zero on any disagreement, so "the doc equals the JSON" is a command rather than a claim. The checker takes the artifact and this doc as its two arguments and resolves its third input — the conflict-count sample whose interactive maximum the cliff line compares against — under `$LLOYD_DATA`, which is why the command exports it; run from the repo root, where `eval/check_doc_matches_json.py` and `eval/measurements/…` both resolve:

```
cd ~/lloyd && LLOYD_DATA=/home/alansrobotlab/lloyd-data .venvs/lloyd/bin/python eval/check_doc_matches_json.py /home/alansrobotlab/lloyd-data/ksweep-2026-11-08/context-rot-ksweep-2026-11-08.json eval/measurements/context-rot-ksweep-2026-11-08.md
```

A run whose `$LLOYD_DATA` no longer holds `ksweep-2026-11-08/conflicts-iv.json` exits non-zero naming the path it looked for rather than reporting a disagreement: the cliff line's compared figure comes from those bytes, so without them the checker says it cannot verify it instead of guessing. The archived copies named in the artifact line above are where those bytes outlive the runtime directory.

---

## Appendix: the runner's own report, written from this JSON by `render_md`

Reproduced verbatim below — the grid, the decision block, the per-shape accuracy and the per-k curve tables — generated from the same `doc` dict this file reads, so the figures above and the rows here cannot disagree unless the JSON changes under both.

# Context rot: Qwen3.8-Flash-Next, 2026-11-07 (P7)

**Verdict: inconclusive.** no length passes the rule in at least one shape; do not move the trigger on this run

_Partial grid: this run is not the measurement the rule was written for._

Runner: `eval/run_context_rot_eval.py`. Commit `21d1ae0020`. Engine: {"served": "Qwen3.8-Flash-Next-nvfp4", "root": "/home/alansrobotlab/lloyd/agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4", "max_model_len": 262144, "cache_config": {"cache_dtype": "fp8", "kv_cache_size_tokens": 844969, "block_size": 3200}}. Pool: pool paused by this run; lease refreshing every 60 s: while this process is alive the hold stands across a backend restart, and once it is not the next pool to start reconciles the halt (#2498); pool resumed.

## repo

| length |  | cold TTFT p50 | warm TTFT p50 | prompt p50 |
|---|---|---|---|
| 50,000 |  | 3.973 | None | 50078.0 |
| 150,000 |  | 12.452 | None | 149870.5 |

### repo distractK at 50k — look-alike count at fixed length (#3801). NOT comparable with the 2026-09-24 grid: different conditions, and `hedged` is a new outcome class.

| k | correct | hedged | wrong | miss | worst pair J | n |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |

Largest k with correct >= 0.9: 4; first k with hedged > 0.1: None

### repo distractK at 150k — look-alike count at fixed length (#3801). NOT comparable with the 2026-09-24 grid: different conditions, and `hedged` is a new outcome class.

| k | correct | hedged | wrong | miss | worst pair J | n |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |

Largest k with correct >= 0.9: 4; first k with hedged > 0.1: None

### repo dissimilarK at 50k — look-alike count at fixed length (#3801). NOT comparable with the 2026-09-24 grid: different conditions, and `hedged` is a new outcome class.

| k | correct | hedged | wrong | miss | worst pair J | n |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.143 | 3 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.143 | 3 |

Largest k with correct >= 0.9: 4; first k with hedged > 0.1: None

### repo dissimilarK at 150k — look-alike count at fixed length (#3801). NOT comparable with the 2026-09-24 grid: different conditions, and `hedged` is a new outcome class.

| k | correct | hedged | wrong | miss | worst pair J | n |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.143 | 3 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.143 | 3 |

Largest k with correct >= 0.9: 4; first k with hedged > 0.1: None

L* (repo) = None; base A(50000) = None; first fail: null

## session

| length |  | cold TTFT p50 | warm TTFT p50 | prompt p50 |
|---|---|---|---|
| 50,000 |  | 4.054 | None | 50769.0 |
| 150,000 |  | 12.61 | None | 152086.5 |

### session distractK at 50k — look-alike count at fixed length (#3801). NOT comparable with the 2026-09-24 grid: different conditions, and `hedged` is a new outcome class.

| k | correct | hedged | wrong | miss | worst pair J | n |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |

Largest k with correct >= 0.9: 4; first k with hedged > 0.1: None

### session distractK at 150k — look-alike count at fixed length (#3801). NOT comparable with the 2026-09-24 grid: different conditions, and `hedged` is a new outcome class.

| k | correct | hedged | wrong | miss | worst pair J | n |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |

Largest k with correct >= 0.9: 4; first k with hedged > 0.1: None

### session dissimilarK at 50k — look-alike count at fixed length (#3801). NOT comparable with the 2026-09-24 grid: different conditions, and `hedged` is a new outcome class.

| k | correct | hedged | wrong | miss | worst pair J | n |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.143 | 3 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.143 | 3 |

Largest k with correct >= 0.9: 4; first k with hedged > 0.1: None

### session dissimilarK at 150k — look-alike count at fixed length (#3801). NOT comparable with the 2026-09-24 grid: different conditions, and `hedged` is a new outcome class.

| k | correct | hedged | wrong | miss | worst pair J | n |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3 |
| 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.143 | 3 |
| 4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.143 | 3 |

Largest k with correct >= 0.9: 4; first k with hedged > 0.1: None

L* (session) = None; base A(50000) = None; first fail: null

## Cost side (usage.db, last 14 days, read-only)

```
{
  "window_days": 14,
  "turns": 5640,
  "busy_seconds": 1025443,
  "busy_seconds_per_day": 73246,
  "peak_bands": {
    "0k-50k": 967,
    "50k-100k": 3086,
    "100k-150k": 1083,
    "150k-200k": 445,
    "200k-240k": 48,
    "240k-infk": 11
  },
  "current_trigger_tokens": 151303,
  "soak_baseline": {
    "days": 3,
    "turns": 1261,
    "prefix_misses": 316,
    "reprefill_tokens": 37538208,
    "turns_with_misses": 132
  },
  "extra_compactions": null,
  "verdict": "not applicable: the trigger does not move"
}
```

Re-run whenever `models.primary.expect_model` changes.

