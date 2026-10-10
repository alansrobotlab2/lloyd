# LloydMemEval v3 — the anti-overlap repair (2026-10-10)

v3 is v2 with exactly one thing changed: the 26 dev items whose `accept.none_of`
value shared a content word with their own `accept.all_of` gold — the `pr-044`
class `label_audit.py` counts as `none_of_shares_content_word` (#2353, #2508).
`26 / 264` on v2's dev leg; **`0 / 264`** here. Every other byte of the set is
v2's: `derived_from: v2`, the same 264 dev items and 66 holdout items, the same
generator, the same facts snapshot, v2's three absent items still absent.

    v2 set_sha 16ec1ae14c046c614428c14d87918734ce3e0cc8530b7d13f8f885120651a079
      -> v3 set_sha abe162a0de484b04e9cacd6b768bb81035f18d07c49bb453a3c7234fde849af6
    verify --set eval/memory_eval/v2   still verifies v2, label_status=pilot
    verify --set eval/memory_eval/v3   verifies v3 at the digest above, label_status=pilot

    (Digests are quoted in full: `verify` prints only the first 16 characters, and
    a 16-character digest of CONTENT is not a git object — the gate's sha rail
    voids a review over one it took for a commit, which happened to this round's
    first gate. The full values are the manifests' own `set_sha`.)
    label_audit.py --set v3            none_of_shares_content_word  0 / 264 (dev)

The `version:` line in each leg's `questions.yaml` is `v3` — `load_set` refuses a
copy that disagrees with its directory — and nothing else in either leg moved.
This file, `anti_repairs.yaml` and `label_verdicts.yaml` sit at the **set root**,
never under `dev/`, for the reason `../v2/AUDIT.md:16-23` records:
`write_manifest` hashes what `_leg_files` rglobs *inside* a leg, so a record
file added under `dev/` would move `set_sha` and `load_set` would then refuse the
set with `manifest edited after freezing`.

## The 26 repairs

`anti_repairs.yaml` is the machine-readable form of this table (id, the v2 value,
the v3 value, the token it shared, and why the new value is the wrong answer this
probe is about). The repair re-states the anti as the **generic alternative** —
the same wrong answer phrased so that a correct answer cannot collide with it —
and never deletes it: an item with no anti value stops punishing the failure it
was written for. Gold is untouched in all 26, and in every one of the other 238.

| id | v2 anti (shared word) | v3 anti |
|---|---|---|
| lme-kn-002 | Email marketing (*marketing*) | the operating-system metaphor |
| lme-kn-013 | 50 million (*million*) | half a billion pages |
| lme-kn-014 | Backlog item #45 (*backlog, item*) | #45 |
| lme-kn-017 | 4-GPU acceleration (*acceleration*) | CPU-only inference |
| lme-kn-018 | 4 hours (*hours*) | 240 minutes |
| lme-kn-021 | 2 hours (*hours*) | one full day |
| lme-kn-029 | Nature in 2024 (*nature*) | a 2024 workshop paper |
| lme-kn-036 | 32 cores (*cores*) | 32 vCPUs |
| lme-kn-040 | WebRTC AEC2 (*webrtc*) | SpeexDSP |
| lme-kn-045 | 20 KB (*kb*) | 20 megabytes |
| lme-kn-047 | handle out-of-date training data Cypher generation (*cypher, solution, stale*) | a benchmark for query latency |
| lme-kn-048 | April 2023 (*2023*) | April 2022 |
| lme-kn-049 | --install flag only (*install*) | the bare script with no arguments |
| lme-kn-050 | 5.0-8.0ms per request (*request*) | 5.0-8.0ms on each call |
| lme-kn-054 | Flux 1 (*flux*) | Stable Video Diffusion |
| lme-kn-067 | 0, 1, 2 (*0, 1*) | a 1-to-5 Likert scale |
| lme-kn-069 | Day 1 (*day*) | the opening session |
| lme-pr-005 | navigate only (*navigate, snapshot*) | fetching the page HTML with a plain HTTP GET |
| lme-pr-023 | light read (*read*) | a passing mention |
| lme-pr-029 | decentralised knowledge (*knowledge*) | federated per-user silos |
| lme-pr-030 | apply default (*apply*) | pass the flag that writes changes |
| lme-pr-045 | rank 8 (*rank*) | lora_alpha=8 |
| lme-pr-064 | allow synchronous cc_orchestrate calls (*synchronous, calls, cc_orchestrate*) | allow blocking dispatch from the realtime path |
| lme-te-024 | Alan and Lisa (*and*) | Alan plus Lisa |
| lme-te-033 | Read-only guarded SELECT queries (*read-only, guarded, select, query*) | a write transaction against kg.sqlite |
| lme-te-072 | VLA models (*models*) | a vision-only perception dataset |

Two of these are worth reading before trusting the rest. `lme-te-072`'s v2 anti
was `VLA models` — the gold's own expansion is `Vision-Language-Action Models`,
so a correct answer that named the acronym the corpus itself uses could be scored
`mixed` and never `clean`: the anti was the failure mode, not the guard.
`lme-te-033`'s was the gold sentence restated with one word changed, which is why
`../v2/AUDIT.md` called it "the same defect as pr-044". `lme-te-024`'s collision
is on the word **and**, which the audit's frozen 134-word function-word list
(`label_audit.py:59-75`) does not contain, and the gold list is punctuated with
`and` — an item can collide on a word that is not a word as long as the same rule
counts it for both sides.

## Judged verdicts: recomputed over the repaired gold

`label_verdicts.yaml` (set root) is the census the manifest cites: **264 of 264
dev items**, 190 clean / 63 brittle / 11 defective — defect rate 4.2%, Wilson
95% [2.3%, 7.3%], upper bound under the 10% ceiling, so `label_quality(v3)`
reports `gold_eligible: true`. That is a computation over these counts, and v2's
row said 32 / 24 / 8 / 0 with an upper bound of 10.7% — it said `false` because
32 items cannot bound a rate at 10% however good they are.

Honest about its own provenance, because #2486's own record is what a reader has:

- 238 rows are **carried** from the #2486 census of 2026-10-09 unchanged; each is
  stamped `read: carried from the v2 census of 2026-10-09, dev leg`.
- The 26 repaired items were **re-read** against their restated anti; each is
  stamped `read: re-read 2026-10-10 against v3's restated accept.none_of` and
  carries `anti_repaired:` naming the value it replaced. 20 of the 26 rewrote their
  reason — 19 of those changed nothing else — because v2's reason quoted an anti
  value that no longer
  exists — a reason that cites a removed value checks nothing, and v2's own
  provenance sweep (`../v2/AUDIT.md`) caught that exact class of stale quote.
- **One verdict moved**: `lme-te-072` defective → brittle. Its defect was the anti
  being the gold's abbreviation; with the anti gone the remaining problem is that
  gold is the expansion only while the user's own June-14 turn says `VLA models`,
  so an answer echoing the corpus wording misses every alias — which biases
  `correct_strict` DOWN. That is brittle, not defective.
- The blind second read that disagreed with the first reader 6 times in 32 rows
  was run against **v2's** labels, not v3's. It is cited, not claimed.
- **`label_status` stays `pilot`.** `gold_eligible` is arithmetic on this row;
  publishing v3 at `gold` and dropping `derived_from` is a ruling about what an
  agent census plus a 32-row blind control is worth as a label record, and #2508
  left it to owed-check. `load_set`'s gold gate (`../run_memory_eval.py:584-596`)
  would accept v3 — the audit passes it — and does not get to decide that alone.

## What this repair does not fix, and cannot

- **A mechanical `0` is not "no anti defects".** The pass catches a token shared
  with the gold. Of v2's 12 defects, four were anti-shape defects and only one of
  them (`lme-te-072`) is visible to the rule. The other three are defective for
  reasons the tokeniser cannot see — `lme-pr-002` (the only right reply, "avoid
  yt-dlp", must *utter* the anti), `lme-pr-017` (anti `all`/`every` are bare
  function words, and "that covers them all" matches), `lme-pr-043` (the prompt
  itself states the anti `trial`, so a correct answer repeats it) — and
  `lme-pr-006` (brittle) pairs gold `cost per capability per second` with an anti
  whose ask names `pass@1`. None of the four is in the 26, `--list` cannot name
  them, and clause 3 of #2508 forbids touching an item outside the 26: they stay
  exactly as v2 wrote them, and `lme-pr-002`/`pr-017`/`pr-043` stay `defective`
  in this census. That is the honest count, not a repair that flattered itself.
- **`lme-kn-018` is still defective.** Its anti was one of the 26 (`4 hours`
  shared `hours`), and repairing it changes nothing about its defect: the prompt
  asks for a percentage increase and gold demands an absolute deployment time, so
  the only correct answer fails. Wrong-slot defects are not anti defects.
- **The 70 long-gold dev items are untouched**, still `70 / 264`. v1's fix
  recommendation was to shorten them, #2486 declined on evidence that the judge's
  `MINW` splits gold on whitespace and so punishes a long *value* far more often
  than a long *item* (#2353's control), and #2508 part 6 keeps that ruling: the
  shortening waits on a paired arm-vs-arm score comparison between long-gold and
  short-gold dev items, which needs a model-backed scored run of two arms over
  v3's dev leg and can only be measured after v3 exists.
- **The holdout leg is unopened and its 4 flagged items stay flagged**
  (`none_of_shares_content_word 4 / 66`). `--list` prints `## leg: holdout — not
  opened` and names no ids — the manifest's `reserve_rule` — so no tuning-view
  round can repair them without deciding to open the reserved split, which is a
  re-freeze decision and not this one.
- **v2 is byte-identical.** `git diff --stat HEAD -- eval/memory_eval/v2` prints
  nothing, its `set_sha` is still the full digest quoted at the top of this file, and `../v2/AUDIT.md` keeps
  its own `26 / 264` line and its own sentence saying v1's fix recommendation was
  never applied: both were true when written, and v3 is where the change landed.
