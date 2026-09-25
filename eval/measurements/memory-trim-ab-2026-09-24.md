# #1425 — behavioural A/B of the 2026-09-14 USER.md trim

**Verdict: the trim lost no behaviour that matters. Restore nothing.** Every
cut rule probed was still acted on correctly without its line: the restored
arm passed 20/20, the canonical arm 18/20 (19/20 on human review), and a
second canonical run 20/20. Canonical-vs-restored discordance (2/20) equals the
canonical-vs-canonical noise floor (2/20). What restoring a line buys is
speed on the one turn that needs it: ~3.7 fewer tool calls and ~1 minute less
wall time. Paying for that would mean adding ~79 KB (~20k tokens) to every
user turn.

Driver: `eval/run_memory_trim_ab.py`. Probes: `eval/memory_trim_ab_probes.yaml`.
Tests: `tests/test_memory_trim_ab_probe.py`. Raw traces are not committed,
because they quote the vault's personal memory files.

## Design

- **Arms.** SOUL.md, MEMORY.md and USER.md were copied once into a frozen
  `canonical` overlay. Each probe has a `restored` overlay: the same copy with
  ONE archive line put back verbatim at the end of its `## ` section. Before
  any trial, `verify_arms` builds both prompts through the real
  `prompt_builder.build_system_prompt` and refuses unless removing the one
  line from the restored prompt gives the canonical prompt byte for byte
  (canonical prompt 102,000 chars). This removes the MEMORY.md-growth confound
  (MEMORY.md grew from 21.9 KB to 72 KB after the trim).
- **Noise floor.** Every probe also ran the canonical arm a second time
  (`canonical_rep`).
- **Trials.** Real harness turns through `bench_runner_sdk.run_bench_sdk` on
  the primary, sandboxed `bench` session ids, all tools live and read-only, a
  16-iteration cap and a 600 s timeout per trial. Total: 60 turns plus retries,
  run under the shared `primary.lock` on a busy engine. An earlier pilot at an
  8-iteration cap was discarded: 5 of its 30 trials were cut off mid-lookup
  and graded FAIL for running out of iterations rather than for what they knew.
- **Probes.** 20, each bound at load to one ledger `move` row. The line had to
  be verbatim in `lloyd/reviews/2026-09-14-user-md-trim-archive.md` and absent
  from live USER.md, MEMORY.md and SOUL.md. The loader refuses condense rows
  and lines that are already loaded. Three of the item's four seed probes were
  invalid:
  - The max-id rule post-dates the trim, so it is not in the archive.
  - The restart-split rule is in MEMORY.md, which the trim never touched.
  - The two-daily-note-paths rule is a condense row.

  The fourth seed probe, `ss`-is-blind, is loaded today from MEMORY.md. The 20
  probes split into 6 fact-recall probes and 14 rule probes; 4 are marked
  `gist_in_live` because a condensed form of their general rule still loads.
- **Grader.** The primary at temperature 0 with thinking off. It saw one
  answer at a time, blind to the arm, with the probe's explicit `criterion`
  and the tool captions, and returned PASS or FAIL. The pair verdict is
  derived: `lost` means restored PASS and canonical FAIL.

## Results (n = 20 move probes, 60 graded answers)

| | divergent | rate | Wilson 95% | lost / gained | sign test |
|---|---|---|---|---|---|
| canonical vs restored (effect) | 2 | 0.10 | [0.03, 0.30] | 2 / 0 | p = 0.50 |
| canonical vs canonical_rep (noise) | 2 | 0.10 | [0.03, 0.30] | 2 / 0 | p = 0.50 |

Pass rates: canonical 0.90, canonical_rep 1.00, restored 1.00.

Cost, paired bootstrap (10k resamples) against the mean of the two canonical
runs:

| | canonical | canonical_rep | restored | restored − canonical mean |
|---|---|---|---|---|
| tool calls per turn | 9.3 | 9.95 | 5.9 | −3.7 [−5.7, −1.7] |
| seconds per turn | 239 | 263 | 192 | −59 s [−115, −11] |

canonical_rep − canonical is −0.65 tool calls [−2.65, 1.3], so the saving is
not noise. The engine was shared with other runs, so the seconds are noisy.
Iteration-cap hits: 0 in every arm.

**Condense population: n = 0.** It was not run. Several original lines merged
into one condensed line, and the ledger does not identify the swap target, so
the loader refuses condense probes instead of guessing. **Whether word overlap
orders divergence therefore cannot be determined here.** The reporter already
computes it per overlap decile once condense arms exist.

## Per-pair verdicts

| probe | kind | canonical | rep | restored | pair |
|---|---|---|---|---|---|
| p01 base_model_key | rule | PASS | PASS | PASS | same |
| p02 wsola_not_phase_vocoder | rule, gist | PASS | PASS | PASS | same |
| p03 chrome_manifest | rule | PASS | PASS | PASS | same |
| p04 qwen35_dense | fact | PASS | PASS | PASS | same |
| p05 pr34223 | fact, gist | PASS | PASS | PASS | same |
| p06 opencode_plan_agent | rule, gist | FAIL* | PASS | PASS | lost* |
| p07 bernie_drive | fact | PASS | PASS | PASS | same |
| p08 freecad_internal_lines | fact | PASS | PASS | PASS | same |
| p09 10_20_70 | fact | PASS | PASS | PASS | same |
| p10 autodesign_gate | fact | PASS | PASS | PASS | same |
| p11 ytdlp_wrong_tool | rule | PASS | PASS | PASS | same |
| p12 transcript_fallback_order | rule | FAIL† | PASS | PASS | lost† |
| p13 stale_bypass_dependent | rule | PASS | PASS | PASS | same |
| p14 fence_blind_grep | rule | PASS | PASS | PASS | same |
| p15 inner_voice_pilot_null | rule | PASS | PASS | PASS | same |
| p16 cited_path_at_head | rule | PASS | PASS | PASS | same |
| p17 progress_denominator | rule | PASS | PASS | PASS | same |
| p18 trend_two_points | rule | PASS | PASS | PASS | same |
| p19 backlog_time_bases | rule | PASS | PASS | PASS | same |
| p20 autonomy_health_gap | rule | PASS | PASS | PASS | same |

Both `lost` pairs pass on `canonical_rep`, so neither reproduces in the arm
without the line.

## Human spot-check (delegated; all 60 answers read)

I agree with the grader on 59 of 60 answers. I overturn one; a second is right
under the rule but is not a loss.

- **\* p06 canonical: overturned, FAIL → PASS on substance.** The answer
  refused to call the setup safe. It found from the installed binary that
  `plan` sets edit and bash to `ask` (not `deny`), and that a headless
  `serve` either hangs or auto-approves. It then recommended explicit `deny`
  rules and a disposable clone. It did not repeat the archived wording ("the
  binary ignores `agent: plan`"), which is why the literal rule failed it.
  The protective behaviour was there.
- **† p12 canonical: the grader is right under the rule, but this is not a
  loss.** The answer put yt-dlp at rung 2. It did so because it found that the
  archived premise ("no JS runtime") is now false: `node` v26.10.0 is
  installed. It also found that `page.transcriptExtractor` does not exist in
  this harness. The restored line is stale, and the canonical arm's deviation
  came from fresh evidence.
- **Several restored lines are stale today.** The restored arms said so
  themselves. p16 restored: "the loaded-memory claim that it's absent is
  stale". p19 restored: "loaded memory still carries … eleven days stale".
  In p13 and p20 the defect the line describes has since been fixed. Putting
  these lines back would load false premises.

With the overturn, canonical passes 19/20, and the one difference (p12) is a
case where the restored line is wrong.

## Scope guard

`~/obsidian/lloyd/USER.md` was 16,368 B before and after the run and was never
written; arms were built from copies. No config change is recommended. What
the result licenses (a lower ceiling, re-adding lines) is Alan's ruling. The
data say re-adding is unwarranted.

## Limits

- n = 20 rules out a large loss, not a small one. The Wilson upper bound on
  the loss rate is 0.30.
- Each probe asks about its own line directly. The measurement is whether the
  model gets the behaviour without the line, not whether an unrelated turn
  would have benefited from it.
- The turns ran through the bench harness, so production's skill prefetch and
  memory recall were absent. Production has more ways to recover a cut fact,
  not fewer.
