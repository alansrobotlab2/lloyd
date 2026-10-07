# LloydMemEval v2 — label audit (mechanical gold-shape pass, 2026-10-07)

What this file is, and what it pointedly is not. Until today it was a
byte-identical copy of `../v1/AUDIT.md` — same md5, first line still naming v1,
its table still describing **35 of v1's 333 items**. This is a v2 document now:
the mechanical gold-shape pass over **all 330 of v2's items**, both legs, plus
the honest state of the judgement those counts cannot substitute for.

## Judged verdicts: none of the 264

**The per-item clean / brittle / defective verdicts for v2's 264 dev items have
not been made.** Not for one item. v1's audit read 35 items by hand (24 clean, 8
brittle, 3 defective — `../v1/AUDIT.md`); v2 inherited 32 of those audited items
by subtraction, and the other 298 remain unaudited. `manifest.json`'s
`label_audit` still records exactly that — `audited: 32`, `clean: 24`,
`brittle: 8`, `defective: 0`, `derived_from: "v1"` — and this pass deliberately
left it alone: `verify --set eval/memory_eval/v2` still prints
`set_sha=16ec1ae14c046c61 label_status=pilot` and
`label_quality` still reports coverage 0.097 with `gold_eligible: False`. A
mechanical count measures shape. It cannot say whether a gold value is the
*right* value, which is what a defect rate is a rate *of*, so it does not touch
the audited record and nothing here moves v2 off `pilot`.

What still has to happen, in order: a person judges all 264 dev items and the
per-category clean/brittle/defective counts land in this file with
`eval/stats.py`'s `wilson_ci` bounds; a second independent reader re-judges ≥30
of them and the disagreements are printed, so the record carries its own error
rate; only then does the manifest's `label_audit` get rewritten and
`derived_from: v1` dropped — a ruling that also moves the 10% pilot/gold ceiling
through `run_memory_eval.label_quality`, which is why it is not a diff.

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
