# Durable-write judge: the induced-rubric arm — backlog #4093

**Status: offline experiment, one induction pass and one judging pass per arm. Nothing in the live write path reads a judge score.** This is PMR's rubric induction only (arXiv:2611.14017); its GRPO credit assignment is not reproduced.

## Split and engines

- split: `induce_split.json` — seq parity: odd seq -> train (induction input); even seq -> held_out (judged)
- held-out subset every arm is scored on: **25** samples (10 bad / 15 good); training half: 25 (induction input only)
- Judge A (hand-authored rubric, 5 random labelled examples): served model `Qwen3.8-Flash-Next-nvfp4`; 25 held-out rows read, 0 training-half rows ignored
- Judge B (hand-authored rubric, 5 nearest-retrieved examples, k=5): served model `Qwen3.8-Flash-Next-nvfp4`; 25 held-out rows read, 0 training-half rows ignored
- Judge C (induced rubric, Judge A's 5 random labelled examples): served model `Qwen3.8-Flash-Next-nvfp4`; 25 held-out rows read, 0 training-half rows ignored

## Results (held-out subset, equal n)

Exact counts beside every percentage: with ~10 held-out bad samples one sample is ~10 points.

| judge | judged bad/good | recall-on-bad % (tp/judged bad) | false-positive-rate % (fp/judged good) | silent-pass % (fn/accepted) |
|---|---|---|---|---|
| Judge A (hand-authored rubric, 5 random labelled examples) | 10/15 (unjudged 0) | **30.0 (3/10)** | 0.0 (0/15) | 31.8 (7/22) UNRATIFIED-DENOMINATOR |
| Judge B (hand-authored rubric, 5 nearest-retrieved examples, k=5) | 10/14 (unjudged 1) | **30.0 (3/10)** | 0.0 (0/14) | 33.3 (7/21) UNRATIFIED-DENOMINATOR |
| Judge C (induced rubric, Judge A's 5 random labelled examples) | 10/15 (unjudged 0) | **40.0 (4/10)** | 0.0 (0/15) | 28.6 (6/21) UNRATIFIED-DENOMINATOR |

silent-pass is printed UNRATIFIED-DENOMINATOR: its denominator holds the `good` labels refused as gold, so recall-on-bad is the only quotable headline.

## Per-class recall (held-out; every class, including 0)

| defect class | n bad held out | A caught/judged (recall %) | B caught/judged (recall %) | C caught/judged (recall %) |
|---|---|---|---|---|
| `false_cutoff_claim` | 1 | 0/1 (0.0) | 0/1 (0.0) | 0/1 (0.0) |
| `invented_url` | 5 | 2/5 (40.0) | 2/5 (40.0) | 3/5 (60.0) |
| `missing_section` | 2 | 0/2 (0.0) | 0/2 (0.0) | 1/2 (50.0) |
| `omitted_numerical_data` | 2 | 2/2 (100.0) | 2/2 (100.0) | 1/2 (50.0) |
| `wrong_metadata` | 2 | 1/2 (50.0) | 1/2 (50.0) | 1/2 (50.0) |

## Leakage check (per induced criterion)

`training hits`: a training-half defect word (class label, short evidence string) or 8-word sample/evidence run appearing verbatim. `held-out hits`: a held-out id, defect word or 8-word run the training half does not contain — i.e. the criterion could have been written from the held-out labels. Either flags the criterion.

| # | criterion | training hits | held-out hits | flagged |
|---|---|---|---|---|
| 1 | When editing the body text, distinguish between URLs explicitly named in the source transcript and those inferred from the speaker's affiliation or project name. | - | - | no |
| 2 | When editing front matter, distinguish between the source identifier present in the raw metadata and the identifier reconstructed from the video title or context. | - | - | no |
| 3 | When editing the summary, distinguish between a complete narrative digest and a raw transcript dump or a placeholder indicating processing failure. | - | - | no |
| 4 | When editing the summary, distinguish between a truncated source recording and a complete source recording that simply ends abruptly. | - | - | no |
| 5 | When editing the summary, distinguish between qualitative descriptions of performance and the specific quantitative metrics provided by the speaker. | - | - | no |
| 6 | When editing the summary, distinguish between the speaker's explicit caveats regarding scope or duration and a generalized statement of capability. | - | - | no |

## Class coverage

FINDING: 1 induced criterion(s) not expressible as any of the five corpus classes: #6

`maps_to` is the induction pass's own reading of each criterion; read the criteria verbatim in `induced_rubric.md` before quoting this line.

## Success margin and stop rule

Judge C vs Judge A on the held-out subset: recall-on-bad 40.0 (4/10) vs 30.0 (3/10) = **10.0 points**, FPR 0.0 vs 0.0 (not worse). Bar (>= +10.0 points at <= A's FPR): **met**.

Stop rule (#4093 step 6): one induction pass, no re-tuning, no second pass. A margin is quoted only after Judge C's reasons are checked against each sample's `correction_evidence` — #580 found judges right for the wrong URL.
