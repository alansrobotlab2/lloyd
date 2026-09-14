# Durable-write judge calibration — backlog #580

**Status: offline experiment. Nothing in the live write path reads a judge score; `grep -n durable_write_judge workers/ app/ scripts/ autonomy/ skills/` returns nothing.** Labels are rule-derived from repair diffs and the repairer's own recorded words; **one rater (Lloyd), no inter-rater check, pending Alan's ratification** — the silent-pass number is not trusted until that exists.

## Corpus

Artifact: `eval/durable_write_judge/corpus.jsonl`, rebuilt with `python eval/durable_write_judge/build_corpus.py` (window 2026-09-05..2026-09-12, `--bad-cap 20 --good-cap 30 --class-quota 4`).

- bad: **20** / good: **30** (targets >=15 bad, >=30 good)
- every bad sample carries its defect classes, its pre-repair text, and the `git show <repair>^:<path>` object it came from

| defect class | n in corpus |
|---|---|
| `false_cutoff_claim` | 2 |
| `invented_url` | 13 |
| `missing_section` | 4 |
| `omitted_numerical_data` | 4 |
| `wrong_metadata` | 4 |

Each class is the repairer's own word for the defect, cited in `build_corpus.DEFECT_CLASSES`; vault commit messages carry none of them (repairs land inside multi-hundred-file batch commits).

## Method

Both judges run the **same model and the same rubric preamble**, over the same 50 samples at temperature 0 on the secondary engine (`http://127.0.0.1:8091/v1/chat/completions`, llama.cpp `Qwen3.6-35B-A3B-UD-Q3_K_XL`), 100 calls in total. They differ in one thing: **which five labelled examples they are shown** — Judge A a seeded random draw, Judge B the five nearest by BM25 over the corpus (k=5). `head` and `tail` of the prompt are byte-identical across judges (`test_the_two_judges_differ_only_in_example_selection`).

Retrieval is BM25 over the labelled corpus alone, not qmd: qmd indexes the vault, where every bad sample's file now holds its **repaired** text, so the live index would return the answer that defines the label. Neither judge's model ever sees the repaired text.

## Results

| judge | judged bad/good | recall-on-bad % | false-positive-rate on good % | **silent-pass rate %** |
|---|---|---|---|---|
| Judge A (static rubric, 5 random labelled examples) | 20/30 (unjudged 0) | **45.0** | 23.3 | **32.4** |
| Judge B (static rubric, 5 nearest-retrieved examples, k=5) | 20/30 (unjudged 0) | **20.0** | 13.3 | **38.1** |

silent-pass rate = accepted-and-labelled-bad / all accepted.
- Judge A (static rubric, 5 random labelled examples): of 34 accepted samples, 11 were labelled bad — **32.4%** silent pass; recall 45.0% (9 of 20 bad flagged).
- Judge B (static rubric, 5 nearest-retrieved examples, k=5): of 42 accepted samples, 16 were labelled bad — **38.1%** silent pass; recall 20.0% (4 of 20 bad flagged).

Fox's reference point is one in five clean passes hiding a serious error.

## Success margin and stop rule

Acceptance: Judge B must raise recall-on-bad by >=10 points absolute at <= Judge A's false-positive rate. Measured margin: **-25.0 points** at B FPR 13.3 vs A FPR 23.3 (not worse). Success criterion: **NOT met**.

The item's stop rule: if B does not beat A at equal false-positive rate, close with the negative result and the per-class breakdown. No step-7 shadow run is started, and `k` and the index are not re-tuned for a second pass. The go/no-go on a shadow run stays the item's own decision, made by a person on this number.

## Per-class recall (every class present, including 0)

| defect class | n bad | A caught | A recall % | B caught | B recall % |
|---|---|---|---|---|---|
| `false_cutoff_claim` | 2 | 1 | 50.0 | 0 | 0.0 |
| `invented_url` | 13 | 6 | 46.2 | 3 | 23.1 |
| `missing_section` | 4 | 1 | 25.0 | 0 | 0.0 |
| `omitted_numerical_data` | 4 | 3 | 75.0 | 3 | 75.0 |
| `wrong_metadata` | 4 | 3 | 75.0 | 1 | 25.0 |

A class at 0 is the deliverable, not a failure: it names something the judge cannot see.

## Selection quality (reported separately from judgement quality)

Fox's open question is whether the retrieved-case leg picks the wrong shape of case and makes the judge confidently wrong in a new way. That needs its own number, computed here from the retrieval leg and the gold labels only.

- bad samples where at least one same-defect-class case arrived in the retrieved five: **75.0%** of 20
- bad samples whose same-file twin (the repaired version of the same note) was excluded from retrieval: leak guard held for **100.0%** (0.0% leaked, and a leak would be the answer, not a hint)

| sample | classes | same-class retrieved |
|---|---|---|
| `bad-001` | invented_url | yes |
| `bad-002` | omitted_numerical_data | yes |
| `bad-003` | invented_url | yes |
| `bad-004` | missing_section | no |
| `bad-005` | invented_url, wrong_metadata | yes |
| `bad-006` | invented_url | yes |
| `bad-007` | invented_url | yes |
| `bad-008` | invented_url | yes |
| `bad-009` | invented_url, omitted_numerical_data | yes |
| `bad-010` | invented_url | yes |
| `bad-011` | invented_url, omitted_numerical_data | yes |
| `bad-012` | invented_url, omitted_numerical_data | yes |
| `bad-013` | wrong_metadata | yes |
| `bad-014` | false_cutoff_claim | no |
| `bad-015` | missing_section | no |
| `bad-016` | missing_section | no |
| `bad-017` | invented_url, missing_section | yes |
| `bad-018` | invented_url, wrong_metadata | yes |
| `bad-019` | false_cutoff_claim, invented_url | yes |
| `bad-020` | wrong_metadata | no |

## Which cases moved the score

| sample | label | A | B | classes | same-class retrieved | path |
|---|---|---|---|---|---|---|
| `bad-001` | bad | flag | accept | invented_url | yes | `knowledge/youtube/AI_Engineer/20260711-develop-at-idea-velocity-jeffrey-lee-chan-snapchat.md` |
| `bad-004` | bad | flag | accept | missing_section | no | `knowledge/youtube/AI_Engineer/20260711-state-of-the-union-why-local-why-now-nvidia-osmantic-roboflow-exo-labs-matthew_.md` |
| `bad-005` | bad | flag | accept | invented_url, wrong_metadata | yes | `knowledge/youtube/AI_Engineer/20260711-stop-ai-agent-hallucinations-5-techniques-production-patterns-elizabeth-fuent.md` |
| `bad-019` | bad | flag | accept | false_cutoff_claim, invented_url | yes | `knowledge/youtube/AI_Engineer/20260829-the-signal-layer-what-to-build-when-anything-can-be-built-lena-hall-akamai.md` |
| `bad-020` | bad | flag | accept | wrong_metadata | no | `knowledge/youtube/Automated_Podcast/20260910-openmind-ai-brain-for-humanoid-robots.md` |
| `good-005` | good | flag | accept | - | n/a | `knowledge/youtube/AI_Engineer/20260909-500-skills-zero-fine-tuning-linkedins-playbook-for-ai-agents-ajay-prakash-linke.md` |
| `good-007` | good | flag | accept | - | n/a | `knowledge/youtube/AI_Engineer/20260909-how-long-can-your-skills-be-before-your-agent-forgets-what-you-told-it-laurie-v.md` |
| `good-018` | good | flag | accept | - | n/a | `knowledge/youtube/AI_Engineer/20260910-the-spatial-harness-bringing-agents-to-the-canvas-max-drake-tldraw.md` |
| `good-020` | good | flag | accept | - | n/a | `knowledge/youtube/AI_Engineer/20260911-building-ambitious-software-jonathan-kelley-dioxus-labs-cognition-2026-09-11-131.md` |
| `good-028` | good | accept | flag | - | n/a | `knowledge/youtube/Discover_AI/20260719-6-dim-harness-w-memory-via-textual-gradients.md` |

## Boundaries kept

- #525's discipline holds: anything a stdlib check can settle (file exists, count equals) belongs to that verifier, never to a judge. This judge's scope is the residue — significance, omission, wrong-level — and that boundary is the first paragraph of `eval/durable_write_judge/README.md`.
- The judging pass is offline and one-shot. It costs secondary-engine capacity only (100 calls, `requests_running: 0` on the primary), and adds no prompt tokens to the interactive loop, so #520's prefix-cache pressure is untouched.

## Caveats

- **One rater.** Every label derives from Lloyd's own recorded repair; Alan is the domain expert who would have to ratify them, and inter-rater reliability is unavailable. Correction evidence travels with each sample (`correction_evidence`) so a rater can re-judge the label, not just the text.
- **Rule-derived labels are conservative and partial.** A class is assigned only when its signature is visible in the repair diff; a repair that reworded prose without removing a URL or adding a heading is unlabelled here. Meaning-reversals and misattributed speakers — both recorded in `memory/2026-09-09.md` — are classes the detector has **no signature for**, so the corpus under-represents them; that is a limitation of the corpus, not evidence they are rare.
- **`good` means no class signature fired**, not verified-clean. A judge flagging a `good` sample may be right, which inflates the measured false-positive rate and makes the reported silent-pass rate optimistic.
- **Small n.** 20 bad across 5 classes; two classes have n<2. Percentages move by ~5 points per sample.
- **The model cannot check the claim against the transcript.** It is judging internal plausibility, which is exactly the residue a rubric can reach; invented-but-plausible URLs are only detectable here against the retrieved examples, which is the hypothesis under test.

## Reproduce

```bash
cd ~/lloyd
PY=.venvs/lloyd/bin/python
$PY eval/durable_write_judge/build_corpus.py            # corpus.jsonl (reads vault git, writes nothing)
$PY eval/durable_write_judge/judge.py --judge both      # 100 offline calls on the secondary
$PY eval/durable_write_judge/score.py                   # this report
$PY -m pytest tests/test_durable_write_judge.py -q      # the pinned clauses
```
