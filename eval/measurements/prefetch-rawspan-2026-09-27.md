# `prefetch_rawspan` against `prefetch_rel`: the distilled line loses, and half of why is the char budget

**Date of measurement:** 2026-09-27 (the 1200-character leg) and 2026-10-04 (the
2400-character repeat). **Set:** LloydMemEval `v1`, dev leg, n=267.
**Model calls:** none, in either leg. This is a retrieval-half measurement: for each
question it asks whether the gold value is present in the facts block an arm would
render, with the selection held fixed, so the two arms differ only in what they put
in front of a model that is not being asked anything.
**Command:** `eval/run_memory_eval.py prefetch-retrieval --arms
prefetch_rel,prefetch_rawspan [--char-budget N]` (#1556 built the arm, #2201 added
the budget flag; the flag's absence is what the 2026-09-27 artifact was written at).

## Why both budgets are in one file

At the shipped 1200-character ceiling the rawspan arm did not merely score lower than
the distilled arm — it had **452 of the 1,421 facts it selected cut out of its own
block before anything could read them** (`n_budget_cut` / `n_selected`). So the 1200
number answers two questions at once and separates neither: is the span rule choosing
the wrong text, or is the text it chose being thrown away? Doubling the ceiling and
re-rendering the *same* selection is the only no-GPU way to ask the second question,
and it needed a flag that did not exist until #2201: the budget was a module constant
at `eval/run_memory_eval.py:161`.

The answer is that the shortfall is **both**: doubling the budget recovers 0.0974 of a
0.2959 gap — 32.9% of it — and none of the sign.

## Artifacts

| Budget | Path | `generated_at` | `char_budget` | `window_chars` |
| --- | --- | --- | --- | --- |
| 1200 | `~/lloyd-data/eval/1480/runs/prefetch-retrieval-prefetch_rel-prefetch_rawspan.json` | 2026-09-27T00:03:47 | 1200 | 200 |
| 2400 | `~/lloyd-data/eval/1480/runs/prefetch-retrieval-prefetch_rel-prefetch_rawspan-budget2400.json` | 2026-10-04T15:50:00 | 2400 | 400 |

A run at any budget other than the shipped 1200 appends `-budget<N>` to the artifact
name (#2201), which is why the 2026-09-27 file is still where every existing reference
points to it. Both read the same facts snapshot —
`facts_root: ~/lloyd-data/eval/1480/corpus/facts` in each — and the same 267 questions:
the 2400 artifact's `set.set_sha` is
`9cf67d045c38a8c3721151f9002ccce2a9419bfe039b950da55cc29990854eca`, the `set_sha`
recorded in `eval/memory_eval/v1/manifest.json`.

**They are not byte-comparable artifacts, and only one number is quoted from each
being the same.** #2170 (`92efe238`, 2026-10-04) landed between them, so the 2400 file
carries a `set` block with `label_status: pilot` and a `label_quality` header that the
2026-09-27 file does not have (`set: null`, no `label_quality`). The comparison stays
sound on what both files hold: `n`, the split, `char_budget`, `window_chars`,
`by_category`, `rawspan_counts` and `comparison`. The rates are also unaffected by the
label gate in principle — the no-model path publishes retrieval-half rates and is
exempt from #2170's refusal, which withholds `correct` rates, not these.

## The paired comparison, at each budget

`diff` is `prefetch_rawspan - prefetch_rel`: the rate of gold values present in the
rendered block, per question, paired on the question id. `ci` is a 95% paired-bootstrap
interval over the 1000 resamples `paired_bootstrap_ci` draws. Both arms' `rel` column is
the distilled arm's rate and is **identical at both budgets in every category** — the
block it renders is short enough that 1200 characters was never cutting it, which is
exactly the asymmetry the repeat was for.

### 1200 characters (window 200)

| Category | n | rel | rawspan | diff | 95% paired CI |
| --- | --- | --- | --- | --- | --- |
| `single_session` | 57 | 0.7193 | 0.2807 | -0.4386 | [-0.5789, -0.2982] |
| `multi_session` | 48 | 0.4792 | 0.1458 | -0.3333 | [-0.4792, -0.1875] |
| `knowledge_update` | 53 | 0.6415 | 0.2453 | -0.3962 | [-0.5472, -0.2453] |
| `temporal` | 57 | 0.4912 | 0.2982 | -0.1930 | [-0.3333, -0.0526] |
| `preference` | 52 | 0.2308 | 0.1154 | -0.1154 | [-0.2115, -0.0192] |
| **all rows** | **267** | **0.5169** | **0.2210** | **-0.296** | **[-0.360, -0.236]** |

All five category intervals exclude 0, all in the same direction.

### 2400 characters (window 400)

| Category | n | rel | rawspan | diff | 95% paired CI |
| --- | --- | --- | --- | --- | --- |
| `single_session` | 57 | 0.7193 | 0.4211 | -0.2982 | [-0.4211, -0.1754] |
| `multi_session` | 48 | 0.4792 | 0.2500 | -0.2292 | [-0.3958, -0.0625] |
| `knowledge_update` | 53 | 0.6415 | 0.4151 | -0.2264 | [-0.3585, -0.0755] |
| `temporal` | 57 | 0.4912 | 0.3684 | -0.1228 | [-0.2456, +0.0000] |
| `preference` | 52 | 0.2308 | 0.1154 | -0.1154 | [-0.2115, -0.0192] |
| **all rows** | **267** | **0.5169** | **0.3184** | **-0.1985** | **[-0.2584, -0.1386]** |

Four of five category intervals still exclude 0; `temporal`'s upper bound sits at
exactly 0.0, so at 2400 it is the one category this measurement can no longer call a
loss. `preference` is unchanged at both rates in both runs (0.2308 / 0.1154) even
though its budget cuts halved from 78 to 35: for that category the extra room bought
nothing, because the gold is not inside the windows the rule selected at all.

### The five counters, per budget

| Counter | 1200 | 2400 |
| --- | --- | --- |
| `n_selected` | 1421 | 1421 |
| `n_unresolved_source` | 18 | 18 |
| `n_no_span` | 0 | 0 |
| `n_budget_cut` | 452 | 229 |
| `n_rendered` | 951 | 1174 |

Per-category counts, same order (`n_selected` / `n_unresolved_source` / `n_no_span` /
`n_budget_cut` / `n_rendered`): at 1200 — `single_session` 300/4/0/93/203,
`multi_session` 260/1/0/86/173, `knowledge_update` 293/8/0/89/196, `temporal`
308/2/0/106/200, `preference` 260/3/0/78/179. At 2400 — `single_session`
300/4/0/47/249, `multi_session` 260/1/0/44/215, `knowledge_update` 293/8/0/46/239,
`temporal` 308/2/0/57/249, `preference` 260/3/0/35/222.

## What the pair of runs says

1. **About a third of the 1200-character deficit is the budget, not the span rule.**
   The rawspan rate rises 0.2210 → 0.3184, which is 0.0974 of the 0.2959 gap at 1200 —
   32.9% of it — while `prefetch_rel` does not move at all in any category. The wider
   ceiling restored 223 records (452 cuts down to 229) and bought that third of the gap. A distilled line is short, so the ceiling was only ever constraining the raw
   arm; the gap measured at the shipped budget is partly a rendering artefact.
2. **The sign survives the double, so the loss is not just truncation.** All-row diff
   is still -0.1985 with a CI of [-0.2584, -0.1386] that excludes 0, and 4 of 5
   categories still exclude 0. The span rule finds the wrong text about one time in
   five even when it is allowed all the room the arm gets.
3. **`n_no_span` is 0 in both runs and `n_unresolved_source` is 18 in both.** Neither
   result is a provenance gap and neither is the windowing rule failing to find
   anything to cut: every selected record resolved to a source and yielded a span. The
   deficit is the rule choosing spans that do not contain the gold value, which is a
   property of the rule and not of the corpus.

## The GPU leg is retired, not owed

Retired by owed-check ruling `20261004_141754_owedcheck_07e8` on #1556
(`~/obsidian/backlog/1556-add-a-prefetch-rawspan-arm-so-representation-is-se.md`, the
ledger entry under `## Activity` at 2026-10-04T21:20:09, ruling text at that file's
`ruling:` field):

> The primary/GPU leg is retired, not owed: gold_in_block is an upper bound on what a
> model can answer from the block, the -0.296 deficit has every category CI excluding 0,
> and the mechanism is counted (452/1421 spans trimmed by the budget before any model
> saw them), so ~534 primary-model calls can only reconfirm a sign and cannot flip a
> decision under the shipping budget — contrary to 'deploy only on measured gain'.

The 2400 repeat does not reopen that decision: `gold_in_block` is still the ceiling on
what a model could answer from the block, and the all-row CI still excludes 0. Note
what it *does* change about the ruling's arithmetic — the "452/1421 spans trimmed"
mechanism the ruling cites is 229/1421 at a 2400 budget, so a wider arm would be
defending a smaller budget story and a larger rule story.

**The condition, stated in the direction that would matter:** if a wider-budget repeat
had flipped the sign — made rawspan's rate meet or beat `prefetch_rel`'s — then
**"distilled >= raw" must not be quoted**, and the GPU question would reopen on the
grounds that the 1200-character measurement had been measuring a budget rather than a
representation. **It did not flip**, at any budget measured, in any of the five
categories, or across all 267 rows. The claim may be quoted at the shipped budget, with
the 32.9%-is-budget share above attached to it. The repeat that would trigger the
condition is cheap and has not been run beyond 2400; nobody should quote the claim
against a wider arm until either that leg exists or the arm's ceiling is documented as
1200.

## What this does not say

- Not that rawspan is useless, nor that it is a low-value knob. `prefetch_rawspan` is
  unbudgeted and off by default (`app/prefetch.py`); these two runs measure a
  representation at one pair of ceilings, and the wider-budget gain is real even
  though the sign held.
- Not that a model answers worse from rawspan blocks by the 0.1985-0.296 margins. Those
  are gold-in-block rates, an upper bound on what any model could answer from the text,
  measured on a pilot set.
- Not the `#1480` rider-1 comparison. That is `prefetch` against `prefetch_rel` —
  confidence order against relevance order over the same facts, a different arm pair
  from the one measured here, whose own artifact and reading live with #1480.
- Not a rerun of the 2026-09-27 artifact. That file was not rewritten and remains the
  1200-char witness, byte-identical; the flag's default is what made that possible.

## Re-derive

From the committed witness (the 1200 artifact's bytes, copied into the vault by #2201 —
see `~/obsidian/backlog/data/prefetch-retrieval-*.witness.md`):

```
cd ~/lloyd-data/eval/1480/runs
python3 -c "import json;d=json.load(open('prefetch-retrieval-prefetch_rel-prefetch_rawspan.json'));print(d['n'],d['char_budget'],d['window_chars'],d['rawspan_counts'])"
# -> 267 1200 200 {'n_selected': 1421, 'n_unresolved_source': 18, 'n_no_span': 0, 'n_budget_cut': 452, 'n_rendered': 951}
python3 -c "import json;d=json.load(open('prefetch-retrieval-prefetch_rel-prefetch_rawspan-budget2400.json'));print(d['n'],d['char_budget'],d['window_chars'],d['comparison']['by_category']['all'],d['rawspan_counts'])"
```

Re-running the 2400 leg costs one no-model pass over 267 questions against the facts
snapshot and no model call; it rewrites the `-budget2400` artifact in place. `--limit`
is not offered on this command, and `--char-budget` is only on `prefetch-retrieval` —
the modelled `run` command still renders at 1200 (#2201 deliberately did not widen a
shipped-path surface for a number it did not need).
