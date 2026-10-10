# LloydMemEval v2 — label audit (mechanical gold-shape pass, 2026-10-07)

What this file is, and what it pointedly is not. Until today it was a
byte-identical copy of `../v1/AUDIT.md` — same md5, first line still naming v1,
its table still describing **35 of v1's 333 items**. This is a v2 document now:
the mechanical gold-shape pass over **all 330 of v2's items**, both legs, plus
the honest state of the judgement those counts cannot substitute for.

## Judged verdicts: all 264 dev items, read 2026-10-09

v1's audit covered 35 sampled items — 7 per category, `random.Random(25).sample` per category,
and that 25 is its seed, not its size (`../v1/AUDIT.md:3`, and its own total row's `24 / 35`).
This is a census: every one of the 264 dev items was read
once — its prompt, its `accept.all_of` gold, its `none_of` anti-value, its grounding lines and
the cited session text — against the same three verdicts (`v1/AUDIT.md:9-13`). The per-item
verdicts and the one-line reason for each live in `label_verdicts.yaml`, at the SET ROOT and not
under `dev/`: `write_manifest` (`../run_memory_eval.py:355-386`) hashes what `_leg_files`
rglobs *inside* each leg, so a file added under `dev/` would move `set_sha` and `load_set`
would then refuse the set with `manifest edited after freezing`. The census is a record, not an
input to `set_sha`, and the frozen set still verifies unchanged (`## Mechanical pass` below, and
a node in `tests/test_label_audit.py` that fails if either half drifts).

| category | n | clean | brittle | defective | defective | + brittle |
|---|---|---|---|---|---|---|
| knowledge_update | 53 | 37 | 13 | 3 | 5.7% | 30.2% |
| multi_session | 47 | 34 | 11 | 2 | 4.3% | 27.7% |
| preference | 50 | 24 | 20 | 6 | 12.0% | 52.0% |
| single_session | 57 | 46 | 11 | 0 | 0.0% | 19.3% |
| temporal | 57 | 49 | 7 | 1 | 1.8% | 14.0% |
| **all dev** | **264** | **190** | **62** | **12** | **4.5%** | **28.0%** |

Defect rate **4.5%** (12/264), Wilson 95% **[2.6%, 7.8%]** — `stats.wilson_ci(12, 264)`. Two
figures sit beside this one and neither is it. The inherited record v2 still carries,
`manifest.label_audit`'s `audited: 32` with `derived_from: v1`, is 0 defective in 32 with
`[0.0%, 10.7%]` — 0 because #2170 cut v2 to exclude v1's 3 defective items, not because anyone
re-read them. v1's own reading of its 35 items was **3 of 35 = 8.6%, [3.0%, 22.4%]**
(`../v1/AUDIT.md:26`). This census sits between the two: the same order of magnitude as the rate
v1 measured on this rubric, an interval a quarter of v1's width because it rests on 264 items
rather than 35, and the first number describing v2's own gold rather than a sample inherited by
subtraction. It does not overwrite the manifest; it is the measurement that ruling has to
answer.

Counting the brittle labels in as well — the v1 audit's own convention, since a brittle label
under-reports a good answer — gives **74/264 = 28.0%, Wilson [23.0%, 33.7%]**
(`stats.wilson_ci(74, 264)`). v1's equivalent — its 3 defective plus its 8 brittle, out of the
same 35 — was 11/35 = 31.4% `[18.6%, 48.0%]`, so the census finds a slightly smaller share of
labels that under-report a good answer, on an order of magnitude more items, and pins it: the two
intervals overlap and v1's spanned 29 points where this one spans 11. It also says the true rate
is *not* zero, which the inherited 0-of-32 record could not.

Produced by reading every dev item; the counts recompute from the data file with

```
.venvs/lloyd/bin/python -c "
import yaml, collections, sys; sys.path.insert(0, 'eval'); import stats
d = yaml.safe_load(open('eval/memory_eval/v2/label_verdicts.yaml'))
c = collections.Counter((x['category'], x['verdict']) for x in d)
n = collections.Counter(x['category'] for x in d)
for k in sorted(n):
    print(k, n[k], c[(k,'clean')], c[(k,'brittle')], c[(k,'defective')])
t = collections.Counter(x['verdict'] for x in d); de = t['defective']
print('all dev', len(d), t['clean'], t['brittle'], de, stats.wilson_ci(de, len(d)))
print('defective+brittle', de + t['brittle'], stats.wilson_ci(de + t['brittle'], len(d)))"
```

The order this file set out before the census — judge the 264, publish the counts with their
bounds, have a second reader re-judge ≥30, and only then rewrite the manifest's `label_audit`
and drop `derived_from: v1` — is unchanged, and the third step is now done. The fourth is not:
`manifest.json` still records `audited: 32`, `clean: 24`, `brittle: 8`, `defective: 0`,
`derived_from: "v1"`, `verify --set eval/memory_eval/v2` still prints
`set_sha=16ec1ae14c046c61 label_status=pilot`, and `label_quality` still reports coverage 0.097
with `gold_eligible: False`. That is deliberate: rewriting it moves the 10% pilot/gold ceiling
through `run_memory_eval.label_quality`, which is a ruling, not a side effect of this census —
and a reading of 264 items by an agent, controlled by one blind re-read of 32, is a stronger
record than v1's 35-item sample but not a human sign-off (#2486's owed entry says exactly which
spot-check is still owed).

Date of the reading: 2026-10-09. The categories' own shapes say where the defects are: the
preference leg is the worst at 6 defective in 50 (12.0%) and 52.0% with brittle counted — gold
that answers a different slot than the prompt asks (a "which film" question whose only gold is
the phrase "the right choice"), and anti-values that a correct answer must say to explain what
it corrected. The two legs that were re-derived from real dates rather than preferences are the
cleanest: single_session has no defect at all and temporal one.

### Second reader

One agent reading 264 items once is a census with no control on it, and the rubric's `clean`
is the verdict a lazy reader defaults to. So a second reader, in a fresh session that never saw
the verdicts above, re-judged a seeded sample blind:

```
.venvs/lloyd/bin/python -c "
import random, yaml
ids = [q['id'] for q in yaml.safe_load(open('eval/memory_eval/v2/dev/questions.yaml'))['questions']]
print(random.Random(20261009).sample(sorted(ids), 32))"
```

Seed **20261009**, sample **32** dev ids, in the order the draw produced them:

```
lme-pr-005 lme-te-041 lme-mu-025 lme-si-027 lme-kn-059 lme-mu-068 lme-te-072 lme-pr-025
lme-kn-067 lme-te-066 lme-te-044 lme-si-031 lme-te-056 lme-si-066 lme-si-026 lme-mu-019
lme-te-059 lme-mu-053 lme-mu-034 lme-te-067 lme-pr-054 lme-mu-072 lme-te-048 lme-pr-028
lme-te-009 lme-si-019 lme-te-030 lme-kn-012 lme-te-024 lme-pr-051 lme-pr-027 lme-pr-006
```

That is 3 kn / 6 mu / 7 pr / 5 si / 11 te — the draw's own shape, not a chosen mix. The reader
was given the 32 item records, the rubric and the shipped matcher's facts, and none of the first
verdicts. It disagreed on **6 of 32 = 18.8%**:
`lme-mu-053`, `lme-pr-006`, `lme-pr-028`, `lme-pr-054`, `lme-te-048`, `lme-te-072`. Every one
was re-adjudicated against the item record, and the verdict changed in all six:

| id | first | second | settled | why |
|---|---|---|---|---|
| `lme-mu-053` | defective | brittle | **brittle** | the second reader's objection was that the gold sits only in the assistant's turns — true of group 1 (`calibrate LLM judges`, `prevent drift`), false of group 2, whose `JSONL` the user says himself. Both groups match their grounding lines and nothing mis-scores a correct recall, so the half-wrong slot is a needle-shape weakness, not a defect. Not clean either. |
| `lme-pr-006` | clean | brittle | **brittle** | it found what the first read missed: one user turn pairs the gold `cost per capability per second` with the anti `single-step accuracy`, and the ask itself names the anti `pass@1`, so explaining the switch scores mixed — exactly v1 `pref-005`'s mode. (The sentence it quoted to make the point is its own paraphrase and no session says it; the pairing is real and is re-derived from the turns here.) |
| `lme-pr-028` | clean | defective | **defective** | the prompt asks for a fix plan; the only gold is the verdict phrase "red flag". A plan that fixes the cycle without uttering that phrase scores zero — a gold in the wrong slot, the kn-018 class. |
| `lme-pr-054` | clean | brittle | **brittle** | "anxious"/"concerned" are one-word generic sentiment tokens and the grounding line is Berman describing himself that way, so an answer says the word without recalling the item. |
| `lme-te-048` | clean | brittle | **brittle** | all three aliases are punctuation-locked (`generated: by/at`, `MINW=3`); the ordinary way to say it — "a generated block with by and at fields" — misses the fold. |
| `lme-te-072` | brittle | defective | **defective** | the anti `VLA models` is the gold expansion's own abbreviation, and the item's second grounding line asserts it inside the same claim: quoting "Vision-Language-Action Models" and naming the acronym scores mixed, and no answer can avoid the acronym. |

The disagreements moved *toward* the defect in five of the six, and four of the six were the
first read calling an item clean when a closer look at the same fields found a collision or a
wrong slot; the sixth (`lme-mu-053`) moved one step back, `defective` to `brittle`, on re-read.
That is the direction a 18.8% disagreement rate should be read in: `clean` here is an optimistic
floor, so the true rate of the 264 is at least this census's number rather than at most. What
this cannot establish is that the 232 items nobody re-read are right; the owed human spot-check
is against this same sample, re-drawable from the seed above.

A third pass then swept the census's own `reason` text against the record each row is supposed
to be a reading of: every single-quoted value in all 264 reasons was looked for in that item's
own prompt, gold, anti, grounding lines and session text. Five rows failed it — `lme-mu-002` and
`lme-mu-034` quoted a numeric '128'/'64' item that exists nowhere in the dev set, `lme-mu-053`
and `lme-pr-006` quoted sentences no session says, and `lme-si-039` illustrated with invented
wrong answers. All five verdicts survive re-derivation from the real record, so no count in the
table above moved; every reason is now quoted from the item it judges. There is no test node for
this sweep, and deliberately so: a reason legitimately quotes a *hypothetical* answer wording
(`lme-pr-020`'s '85% of VRAM'), so no substring rule separates a fabrication from a hypothesis
without a reading — which puts the sweep in the same owed human spot-check as the verdicts.

## Mechanical pass: both legs, run 2026-10-07

Produced by (repo root, read-only — it edits no gold and writes nothing under
`eval/memory_eval/`):

```
.venvs/lloyd/bin/python eval/label_audit.py --set v2 --holdout
```

| defect | dev | holdout |
|---|---|---|
| a gold value longer than the 3-word cap | **70 / 264** | **19 / 66** |
| the item's only gold value is a single common English word | **0 / 264** | **0 / 66** |
| a `none_of` value shares a content word with any `all_of` gold | **26 / 264** | **4 / 66** |

Denominators are the legs' own item counts: 264 dev, 66 holdout, 330 total. The
holdout leg was **counted and never opened** — `--holdout` prints the three
figures above and no holdout item id, gold string or prompt, per the manifest's
`reserve_rule` ("it reports aggregates, never per-question rows or ids").

The rules behind those numbers are printed with them by the script, and the
report is what to read for them, not this table. That is a deliberate clause of
#2353: three passes over this same frozen YAML have produced three different sets
of figures — the #2170 filing's **61 / 2 / 24** (dev) and **16 / 0 / 5**
(holdout), triage's independent pass **73 / 0 / 35** and **24 / 0 / 9**, and this
run's **70 / 0 / 26** and **19 / 0 / 4** — not because anybody's arithmetic
differed, but because "a gold value" was undefined where an `all_of` group
carries two aliases and "common English word" was undefined as a word list. So:
a gold value is **every alias string of every group** and an item is flagged if
any of them trips a rule; word count is `len(value.split())`; the 134-word list
treated as common English (and, as its complement, "content word") is printed in
full in the report. A reader who disagrees with an entry can name it and re-run.
The 3-item and 9-item dev spreads between this pass and triage's are the cost of
not writing those definitions down, and why the counts above are quoted with
their rules attached.

## Reading the counts

70 of 264 dev items carry at least one gold over the 3-word cap — the brittle
shape v1 flagged by hand ("the gold string is long… the rules judge will miss a
correct paraphrase"), the most common failure mode here, and consistent with v1's
note that its build validator capped gold at 6 words and "many of these sit at
5-6".

26 of 264 carry an anti value that overlaps the right answer's own content words
— the pr-044 class, where `none_of` can punish a correct answer. v1 recommended
re-stating those probes so the anti value is the generic alternative and never a
component of the right answer; 26 items in v2 still have that shape. The two
classes are largely independent — 7 items carry both, so **89 of the 264 dev
items carry at least one of the three shapes** (22 of 66 on the holdout leg),
and `--list` names the dev items behind each count.

0 items on either leg rest on a single common English word as their only gold —
v1's pr-042 class. That is the one number that reads as a success rather than a
debt: pr-042, pr-044 and mu-060 are exactly the three items v2 was built by
dropping, and the mechanical pass finds no surviving instance of pr-042's shape.
Read it as what it is: v2 contains no item whose *only* gold is one of the 134
listed words. It does not say v2's golds are well-chosen — nothing here does.

## What is owed against these numbers

Shortening the 70 long-gold items and re-stating the 26 overlapping anti values
— item counts over the same 264 dev items as the table, under the definitions
this file states above, and `--list` names the dev items behind each — are the
repairs these counts point at, and both move the frozen set's `set_sha` off
`16ec1ae14c046c61`, so they need their own re-freeze decision rather than
happening as a side effect of an audit. That decision, the 264 judgements and
the second reader are what #2353 leaves open; the counts above are the shape of
the job, measured over every item, with the rules stated.
