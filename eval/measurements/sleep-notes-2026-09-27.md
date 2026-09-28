# Sleep-time notes, measured: the #1516 channel against both shipped rankings (sleep-notes-2026-09-27, #1631)

**What was measured.** Whether #1516's next-session channel is worth wiring a
producer for. The `sleep_notes` arm takes the exact facts the `prefetch` arm
retrieves for a question and carries them through `app.next_session_notes` —
written, then drained the way a morning turn drains them — so they arrive
inside `<next-session-notes>` instead of `<facts>`. Same retrieval, different
transport, position and cap, which is what makes it a measurement of the
channel and not of retrieval. This is the comparison #1516 left owed: before
this run the arm had never been answered at all, and the live store had never
been written to (`~/lloyd-data/next-session-notes.json` absent, only its `.lock`).

## What the numbers say

- **Against the arm the channel's own docstring names, it is a tie.**
  `multi_session` Δ=-0.021 [-0.062, 0.000], n=48 — does not clear 0;
  `knowledge_update` Δ=+0.038 [-0.094, 0.170], n=53 — does not clear 0.
  Neither clears 0, so the pair #1516 registered on its own would have
  reported the channel as free.
  The `multi_session` interval's upper bound sits exactly on `0.000` with a
  non-zero Δ — a boundary, never a rejection, and never a nothing either.
- **Against the ranking production ships, the channel loses, and it loses in
  both named categories.**
  `multi_session` Δ=+0.188 [0.021, 0.354], n=48;
  `knowledge_update` Δ=+0.226 [0.113, 0.358], n=53;
  whole dev slice Δ=+0.112 [0.056, 0.169], n=267.
- **The retrieval half of the gap is larger than the answer half.** On
  `evidence_in_context` the same pair runs
  Δ=+0.247 [0.183, 0.311], n=267 — the shape the
  channel's content cap predicts, and the reason a parity reading against
  `prefetch` does not mean the channel is free.

The ship/no-ship ruling is **not made here**: it is a deployment decision and
goes to the owed-check job with these numbers (see *What this does not decide*).

## Reproduction

```bash
flock -s ~/.local/state/lloyd-automod/primary.lock \
  .venvs/lloyd/bin/python eval/run_memory_eval.py run \
  --arms prefetch,prefetch_rel,sleep_notes \
  --label sleep-notes-2026-09-27 \
  --out-dir /home/alansrobotlab/lloyd-data/eval/1631/runs
```

The shared primary lock is the whole of the concurrency protection the runner
implements — there is no autonomy-pool check in it — so the run is schedulable
alongside the pool, and it answers on the shared engine.

sleep_notes_store: /home/alansrobotlab/lloyd-data/eval/1631/runs/sleep-notes-store-1691149.json

The arm's store is a file of the run's own, inside its out dir
(`/home/alansrobotlab/lloyd-data/eval/1631/runs`); it is **not** the live channel (`/home/alansrobotlab/lloyd-data/next-session-notes.json`),
which is still absent after this run — the run drained every note it wrote.

| provenance | |
|---|---|
| label | sleep-notes-2026-09-27 |
| created (UTC) | 2026-09-28T03:02:26+00:00 |
| arms answered | prefetch, prefetch_rel, sleep_notes |
| dev questions | 267 |
| set | v1, manifest hash 9cf6…4eca (sha256 over the frozen set's files — a content hash, not a git object) |
| corpus (facts snapshot) | /home/alansrobotlab/lloyd-data/eval/1480/corpus/facts |
| answerer | Qwen3.8-Flash-Next-nvfp4 @ http://127.0.0.1:8096, thinking on, T=0.6, top_p 0.95, max_tokens 4096 |
| judge | djev:DiffusionGemma-26B-A4B-NVFP4, rules first (mixed / unmatched answers); rules/djev agreement 0.9123 on 285 answers the rules settled alone |
| wall, answering only | 1693.6 s |
| min_category_n | 20 |
| artifact | /home/alansrobotlab/lloyd-data/eval/1631/runs/lloydmemeval-sleep-notes-2026-09-27.json |

## The paired-bootstrap CIs

Paired per question, percentile bootstrap, 95%. `correct_strict` is the
headline metric (`eval/measurements/lloydmemeval-2026-09-25.md`);
`evidence_in_context` is the retrieval leg — was a gold value anywhere the
model could read it. `compare()` prints a CI for a category only at
n ≥ 20; every row below is a number, never `insufficient`.

### sleep_notes vs prefetch — correct_strict (dev slice)

Δ is what `compare()` computes: `prefetch − sleep_notes`, in points of `correct_strict` over the 267 questions both arms answered. A **negative Δ means the next-session channel is ahead** of the arm it is compared with.

| category | n | a (sleep_notes) | b (prefetch) | Δ (b − a) | 95% CI | clears 0 |
|---|---|---|---|---|---|---|
| multi_session | 48 | 0.229 | 0.208 | -0.021 | [-0.062, 0.000] | no |
| knowledge_update | 53 | 0.396 | 0.434 | +0.038 | [-0.094, 0.170] | no |
| all | 267 | 0.292 | 0.300 | +0.007 | [-0.034, 0.049] | no |

- multi_session (n=48): CI [-0.062, 0.000] does not clear 0
- knowledge_update (n=53): CI [-0.094, 0.170] does not clear 0

### sleep_notes vs prefetch — evidence_in_context (dev slice)

Δ is what `compare()` computes: `prefetch − sleep_notes`, in points of `evidence_in_context` over the 267 questions both arms answered. A **negative Δ means the next-session channel is ahead** of the arm it is compared with.

| category | n | a (sleep_notes) | b (prefetch) | Δ (b − a) | 95% CI | clears 0 |
|---|---|---|---|---|---|---|
| multi_session | 48 | 0.271 | 0.312 | +0.042 | [0.000, 0.104] | no |
| knowledge_update | 53 | 0.491 | 0.566 | +0.075 | [0.019, 0.151] | yes |
| all | 267 | 0.393 | 0.528 | +0.135 | [0.094, 0.176] | yes |

- multi_session (n=48): CI [0.000, 0.104] does not clear 0
- knowledge_update (n=53): CI [0.019, 0.151] clears 0

### sleep_notes vs prefetch_rel — correct_strict (dev slice)

Δ is what `compare()` computes: `prefetch_rel − sleep_notes`, in points of `correct_strict` over the 267 questions both arms answered. A **negative Δ means the next-session channel is ahead** of the arm it is compared with.

| category | n | a (sleep_notes) | b (prefetch_rel) | Δ (b − a) | 95% CI | clears 0 |
|---|---|---|---|---|---|---|
| multi_session | 48 | 0.229 | 0.417 | +0.188 | [0.021, 0.354] | yes |
| knowledge_update | 53 | 0.396 | 0.623 | +0.226 | [0.113, 0.358] | yes |
| all | 267 | 0.292 | 0.405 | +0.112 | [0.056, 0.169] | yes |

- multi_session (n=48): CI [0.021, 0.354] clears 0
- knowledge_update (n=53): CI [0.113, 0.358] clears 0

### sleep_notes vs prefetch_rel — evidence_in_context (dev slice)

Δ is what `compare()` computes: `prefetch_rel − sleep_notes`, in points of `evidence_in_context` over the 267 questions both arms answered. A **negative Δ means the next-session channel is ahead** of the arm it is compared with.

| category | n | a (sleep_notes) | b (prefetch_rel) | Δ (b − a) | 95% CI | clears 0 |
|---|---|---|---|---|---|---|
| multi_session | 48 | 0.271 | 0.500 | +0.229 | [0.062, 0.396] | yes |
| knowledge_update | 53 | 0.491 | 0.717 | +0.226 | [0.113, 0.358] | yes |
| all | 267 | 0.393 | 0.640 | +0.247 | [0.183, 0.311] | yes |

- multi_session (n=48): CI [0.062, 0.396] clears 0
- knowledge_update (n=53): CI [0.113, 0.358] clears 0

## Why two comparisons, not one

`prefetch` is the arm the channel's docstring names and the pair the runner
has always printed. It is **not** what production ships: `config.yaml` orders
`<facts>` by relevance, so a CI against the confidence-ordered arm alone
prices the channel against a block no live turn renders — and on this run it
returns parity. The `prefetch_rel` rows are the channel measured against the
ranking that actually runs, and they are the rows a ship/no-ship ruling has
to read. That pair is now in `DEV_COMPARISON_PAIRS` for every future run.

## What this does not decide

1. **The ruling itself.** Keep or delete the channel, given the CIs above,
   is a deployment decision and goes to the owed-check job.
2. **Who writes the notes.** No scheduled producer exists — nothing under
   `~/obsidian/skills/` or `~/obsidian/autonomy/` calls the channel — so the
   live store has never held a note. Which nightly job would own writing them
   is a scope call outside this round.
3. **A real morning.** Whether a producer-written note reaches the first chat
   turn is only measurable once (2) exists; and per #1516's drain-eats-notes
   exposure that check has to write a fresh note first, not read an empty
   channel as a failure.

## What these tables cannot see

The channel's one designed cost is `NEXT_SESSION_CONTENT_MAX` in
`app/next_session_notes.py`: material the `<facts>` block would have carried
whole can be truncated on the way through a note, and that truncation lands
in `evidence_in_context`. The artifact stores the rendered `<facts>` block
for the prefetch arms but not the drained note for this one, so **no row here
separates a truncation cost from a retrieval miss**: a `sleep_notes` row below
its comparator says the channel lost something, not which. A ruling that
hinged on that difference would need the cap instrumented first — this run
shows the gap is large enough (the `evidence_in_context` rows above) that
instrumenting it is worth an item, and it is not this one.

## What pins this file

`tests/test_memory_eval.py`:

- `test_the_committed_report_carries_the_ship_no_ship_ci_for_both_named_categories`
- `test_the_committed_report_names_the_command_and_a_store_that_is_not_the_live_channel`
- `test_the_channel_is_priced_against_the_ranking_production_ships`

Every `n` is checked against the frozen dev slice, every Δ against the two
rates printed beside it, every CI against the Δ it covers, and every
`clears 0` line against the interval it states.

