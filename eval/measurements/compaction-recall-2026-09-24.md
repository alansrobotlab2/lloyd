# Compaction recall: planted-fact eval, 2026-09-24 (backlog #600)

**Verdict: reaffirm.** Leave `compaction.microcompact` at 0.72 / 0.52 and
`harness.context_relief` as it is. Production's clearing costs no measurable
recall. The only alternative that beats it on cost (fire the in-turn pass
later) can't be expressed as a safe `compaction.*` edit, because the same key
also moves the turn-start trigger. See "What would win" below.

Runner: `eval/run_compaction_recall_eval.py`. Baselines:
`eval/baselines/compaction-recall-2026-09-24.json` (late probe, the result) and
`eval/baselines/compaction-recall-early-pilot-2026-09-24.json` (early probe,
the pilot). Engine: the primary, Qwen3.8-Flash-Next on vLLM :8096, FP8 KV, run
under the exclusive primary lock with the worker pool paused.

## Design in one paragraph

Each synthetic session is real session JSON: tool call and result filler cut
from this tree's own files, with one planted Read result at depth 0.1, 0.5 or
0.85. That result holds a distinctive fact (a release codename) and an
ambiguous one (the billing-east relay's current port; the old port and sibling
relays' ports are digit permutations sprinkled through the filler). Each
probe turn goes through the real `load_and_compact_session` and the real
`run_query`, using the production `RunOptions` and all 145 live tool schemas
(stub handlers; Read also serves the spill files). Before the probe, the
uncompacted prompt is sent with `max_tokens: 1`, so the cache state is that of
a session arriving at its next turn. In the late probe, the question sits
inside a file the turn must Read first, so the need for the old fact arises
only after the in-turn pass has had its chance to clear. Which mechanism fired
comes from the #1078 record; a run whose arm did not exercise its own feature
is dropped and counted. The grading is a code check; no judge was needed
(undecided: 2 in `tool_clear`, 1 in `raised`, counted as misses).

## The finding that shapes everything: which pass actually fires

The two passes share one fraction but measure different things:

- **Turn start** compares 0.72 × 210,144 = 151k against an estimate of the
  history alone.
- **In turn** compares the same 151k against the engine's reported prompt,
  which includes the system prompt and tool schemas. Measured here, reported
  ≈ 43k + 1.375 × history estimate: 90k history → 167k reported, 130k →
  222k.

So the in-turn pass fires after the first tool call on any session past
roughly 80k of history. The turn-start pass would need a history estimate of
151k, which is already ≈ 251k reported, at the 262k wall. Every production
run in this baseline was cleared in turn (60–119k tokens freed) and none at
turn start. This matches #1078's log census (802 in-turn clears, 0
summarizations). The item's 30k–150k band is therefore the band where only
the in-turn pass acts, and at 30–80k nothing acts at all.

## Results: late probe, 20 sessions × 3 arms, plus 2 candidate arms

Sizes are the history estimate (reported prompt in brackets). Hits are
distinctive / ambiguous. The 95% CIs are Wilson intervals, per fact, never
blended.

| arm | size | n | distinctive | ambiguous | TTFT after clear, median | tool calls, mean | run cache hit | peak KV |
|---|---|---|---|---|---|---|---|---|
| none | 90k (167k) | 10 | 10/10 [.72,1] | 10/10 | 0.84 s | 1.3 | 0.97 | 0.215 |
| production | 90k | 10 | 10/10 [.72,1] | 10/10 | 4.35 s | 2.3 | 0.83 | 0.211 |
| tool_clear | 90k | 10 | 5/10 [.24,.76] | 5/10 | 2.25 s | 6.9 | 0.79 | 0.159 |
| none | 130k (222k) | 10 | 10/10 [.72,1] | 10/10 | 0.99 s | 1.1 | 0.98 | 0.267 |
| production | 130k | 10 | 9/10 [.60,.98] | 10/10 | 3.63 s | 2.8 | 0.87 | 0.263 |
| tool_clear | 130k | 10 | 7/10 [.40,.89] | 7/10 | 3.0 s | 8.8 | 0.81 | 0.166 |
| raised 0.9/0.7 | 130k | 10 | 10/10 | 8/10 [.49,.94] | 7.46 s | 2.0 | 0.79 | 0.263 |
| trigger90 0.9/0.52 | 130k | 10 | 10/10 | 10/10 | 3.62 s | 2.5 | 0.86 | 0.263 |

Across both sizes (n = 20 per arm): none 20/20 and 20/20; production 19/20
[0.76, 0.99] and 20/20; tool_clear 12/20 [0.39, 0.78] and 12/20. `raised` and
`trigger90` dropped their 10 runs at 90k: at a 189k trigger they cannot fire
on a 167k prompt, so the validity gate removed them before any engine time was
spent, which is the gate working. Dropped counts: 0 for none, production and
tool_clear. Preemptions: 0 in every run (single-tenant; see the clauses
below).

The paired comparison against production is over the same sessions, using a
10,000-resample bootstrap. The ttft figure is the sum over the turn's
iterations.

| vs production | size | distinctive Δ | ambiguous Δ | ttft Δ | tool calls Δ | wall Δ |
|---|---|---|---|---|---|---|
| none | 90k | 0 (identical) | 0 (identical) | **−4.7 s [−5.5, −4.1]** | −1.0 [−2, 0] | −6.6 s [−14.8, +0.7] |
| none | 130k | +0.1 [0, .3] | 0 | **−3.6 s [−4.2, −3.0]** | **−1.7 [−2.8, −0.8]** | **−8.9 s [−15.4, −2.8]** |
| tool_clear | 90k | **−0.5 [−.8, −.2]** | **−0.5 [−.8, −.2]** | **+2.9 s** | **+4.6** | **+11.6 s** |
| raised | 130k | +0.1 | −0.2 [−.5, 0] | **+3.5 s [2.8, 4.2]** | −0.8 | +10.4 s* |
| trigger90 | 130k | +0.1 | 0 | +0.2 s [−0.5, 1.0] | −0.3 | +7.7 s* |

\* `raised` and `trigger90` ran in a later batch, not interleaved with the
first three arms. Their wall-time deltas include the engine's state drift
between batches, and trigger90's +7.7 s against an identical mechanism
(ttft Δ ≈ 0) is that drift. Use their ttft and tool-call columns, not the wall
column.

What the numbers say:

- **Production's clearing costs no measurable recall.** The model re-reads
  the spilled or original file. At 90k, 7 of 10 production runs got the facts
  back through a tool result; at 130k, 9 of 10. The one miss (130k, depth 0.5)
  answered without re-reading.
- **It does cost latency.** A cleared prompt re-prefills from the first
  cleared result, which takes 3.6–4.7 s of TTFT per turn and 1–1.7 extra tool
  calls, and cuts the run's cache hit from 0.97 to 0.83–0.87. It does not
  lower peak KV within the turn, because the first iteration, sent before the
  clear, is the peak. Its KV benefit falls on the rest of a long turn, which a
  single probe turn does not measure.
- **Aggressive clearing (the video's preset) is a real recall loss:** 12/20 on
  both facts (paired vs production over 20: −0.35 [−0.55, −0.15] and −0.40
  [−0.60, −0.20]). It also re-clears what the model re-reads, and most of its failed runs ended
  with the model saying its reads "keep getting swept". The video's result
  holds, in its blunt form, for this model.
- **Raising the target with the trigger (0.9/0.7) is worse.** It re-prefills
  147k instead of 109k after the clear, which adds about 3.5 s of TTFT per
  turn, with no recall gain.

### The early probe (pilot, 6 sessions × 3 arms)

With the question in the user message, production answered 6/6 and 6/6 even
though 5 of 6 runs had the planted result cleared off the wire. The model
read the question at iteration 1, answered it in that iteration's reasoning,
and preserved thinking carried the answer past the clear. So a question the
turn already holds is immune to the in-turn pass. The late probe is the
harder, decision-relevant case. tool_clear scored 4/6 there too.

## What would win, and why it is not a config change

`trigger90` (in-turn trigger 189k, target unchanged at 109k) matches
production where it fires (130k: 20/20 facts, ttft Δ +0.2 s, ns). Where it
does not fire, it behaves exactly like `none`, and in the 151–189k
reported-prompt band that is **equal recall for ~4.7 s less TTFT and ~1 fewer
tool call per turn** (the 90k row above). Every turn of a session in that band
pays that cost again, because an in-turn clear is not persisted: the next turn
rebuilds the full history and clears it again.

It cannot ship as `compaction.microcompact.trigger_fraction: 0.9`. That one
key also moves the turn-start trigger to a history estimate of 189k, which is
≈ 303k reported and past the 262k window, so iteration 1 of such a turn would
overflow into the 413 recovery path. Delivering the win needs a code change:
a separate in-turn trigger key (for example
`harness.context_relief.intra_turn_trigger_fraction`, default 0.72 = today),
or one pass converted to the other's units. That is follow-up work, and
wanting it is a judgement call. The engine has 845k of KV pool, but a 189k
resident prompt per session is the preemption risk the item's third human
clause is about.

## Human clauses (decided)

- **Config edit**: none recommended. Nothing for a human to apply.
- **Nightly GPU budget**: decided **not to schedule** this runner nightly. A
  full run takes about 35 min of exclusive primary time (60 probe turns plus
  warm-ups), and the policy only changes when the compaction or relief code,
  the thresholds or the model change. Re-run it on those events, as the item's
  own risk section asks ("re-run after any prefix-breaking change"). The
  "fits the nightly budget" observation therefore does not arise.
- **Preemption guard**: no trigger is widened, so there is nothing to confirm
  under live load. For the record: `vllm:num_preemptions_total` did not move
  in any of the 98 probe turns (80 late + 18 pilot). These ran single-tenant, so that is not the
  concurrent-load evidence the clause asks for, and it would be required before
  any trigger90-style change.

## Limits

- Synthetic sessions. The filler is this tree's real text in real tool shapes,
  but the facts are planted in one Read result. A fact the assistant restated
  in prose would survive every clearing mechanism.
- n = 10 per arm per size. The no-difference result for none vs production
  bounds a recall loss at roughly ≤ 25 points per fact per size (Wilson lower
  bound 0.72 at 10/10). Across both sizes the bound is about 0.76. That is
  enough to say production's clearing is not a large recall tax, but not
  enough to rule out a small one.
- The summarize layer never fired: the history estimate never reached 210k
  before the window. It is effectively dead configuration at this window size,
  as #1078 suspected. The 30–80k band, where no mechanism acts, was not run,
  since every arm there is `none` by construction.
