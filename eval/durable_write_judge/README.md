# Durable-write judge calibration (#580)

Offline experiment, one pass, **nothing in the live write path reads a judge
score**. It measures whether a durable write Lloyd has already made hides a
defect, and at what rate a judge accepts one that does — the **silent-pass rate**,
which the item body calls "the Lloyd version of one in five clean passes".

Result: [`report.md`](report.md). Reproduce at the bottom.

## The boundary against #525, written down before either ships

#525 makes worker claims machine-checkable by a **stdlib verifier** and keeps the
verifier *never LLM-judged*. That division is not a preference, it is the only
thing that stops this harness becoming a second silent failure sitting on top of
the first:

| the question | who answers it |
|---|---|
| does the file exist, does the count equal N, is the URL in the transcript | `#525`'s stdlib verifier. Deterministic, cheap, never a judge. |
| is the omission significant, is this learning wrong-level, did the note claim what the transcript never said | the residue — a judge. And only a judge. |

Anything a stdlib check can settle must go to that verifier even when a judge
would happily opine on it. This harness is scoped to the residue, and its scope
is enforced structurally, not by convention: `test_the_judging_harness_is_not_wired_into_any_write_path`
greps `workers/`, `app/`, `scripts/`, the entrypoints, and the vault's `autonomy/`
and `skills/` for any reference to these modules, and fails if one appears.
`judge_one` returns a string and never touches a write.

## Why it is offline and stays offline

Per-sample retrieval adds prompt tokens to a harness that is prefix-cache
sensitive (`prefix_cache_hit_rate` 58.9 % since boot / 77.3 % recent; #520 is
open on that). One-shot judging over 50 samples costs the **secondary** engine
(llama.cpp Qwen3.6-35B-A3B, `http://127.0.0.1:8091`) 100 calls and zero primary
capacity. Nothing in the interactive loop pays for it.

Step 7 of the item — a two-week shadow run over live writes — is **not started
here**. The item's stop rule makes that a decision on the measured margin, and a
person's call.

## Files

| file | what it does |
|---|---|
| `build_corpus.py` | Rebuilds `corpus.jsonl` from vault git: bad = pre-repair text of a digest note that was later repaired, with the defect classes detected from the diff; good = same-shape notes accepted and never edited. Read-only over the vault. |
| `corpus.jsonl` | The labelled corpus. One JSON object per line: `label`, `defect_classes`, `pre_repair_text`/`accepted_text`, `correction_evidence`, `repaired_diffstat`, `recovered_from` (the `git show <repair>^:<path>` object). |
| `retrieve.py` | The retrieval leg for Judge B, plus the leak guard. BM25 over the labelled corpus. |
| `judge.py` | Judge A and Judge B. Same model, same rubric preamble, same temperature, same output contract — five labelled examples each, A's drawn at random, B's nearest-retrieved. |
| `score.py` | Recall-on-bad, false-positive-rate, silent-pass rate, per-class recall, selection quality, moved cases → `report.md`. |
| `judge_raw_{a,b}.jsonl` | Every call's raw verdict and the example ids shown, so a person can re-derive both metrics from the shipped artifacts. |
| `report.md` | The result, the caveats, and the stop-rule verdict. |

## Why the labels live in the diff, not the commit

The triage correction that changed the recipe: the recorded repairs have **no
matching vault commit messages** — repairs land inside multi-hundred-file batch
commits (`f31de999` modifies 174 digest notes, removing 456 URL lines), and zero
commit messages match `fabricat*|invented url|hallucinat*`. So a class is
assigned **only when its signature is visible in the before/after diff of the
note**:

- `invented_url` — a URL the accepted version does not keep, and the daily note
  calls fabricated / invented / non-existent.
- `false_cutoff_claim` — the pre-repair note claimed a cut-off; the repair
  dropped it and added content.
- `omitted_numerical_data` — the repair added ≥3 numeric tokens the note lacked.
- `missing_section` — the repair added a `## ` heading the note lacked.
- `wrong_metadata` — a front-matter `title` / `url` / `speaker` value changed.

Each class cites the repairer's own word in `build_corpus.DEFECT_CLASSES`.
No class is authored from the whiteboard: the set is the signature set, and
`omitted_numerical_data` / `missing_section` are the two that came out of the
corpus rather than from the item's list. Classes that are **recorded but have no
diff signature** — meaning-reversals, misattributed speakers — are absent from
the corpus, and `report.md` says so rather than implying they are rare.

Labels are rule-derived, one rater (Lloyd), and **unratified by Alan** — the
silent-pass number is not trustworthy until that ratification exists. Every
sample carries its `correction_evidence` so a rater can re-judge the label
itself.

## Retriever choice, and the leak that had to be closed

Retrieval is BM25 over the labelled corpus, **not** qmd, and the reason is a leak
rather than a preference: qmd indexes the vault, where every bad sample's file
now holds the **repaired** text — its `## Q&A` section, its corrected URL, its
restored numbers. Retrieving from qmd would hand Judge B the very thing that
defines the label, and its win would measure the answer. Neither judge's model
ever sees the repaired text: `pre_repair_text` is what is shown and judged.

The remaining leak path is the same-file twin (a note repaired twice appears as
two samples whose files are identical before the second repair), and
`retrieve.excluded_ids` drops the judged sample **and every sample sharing its
`vault_path`** before ranking. That exclusion is what makes
`selection_quality`'s same-class rate a real number instead of a tautology — see
`test_selection_quality_reads_the_labels_not_the_verdict`.

## Reproduce

```bash
cd ~/lloyd
PY=.venvs/lloyd/bin/python
$PY eval/durable_write_judge/build_corpus.py --counts   # counts only, no git walk
$PY eval/durable_write_judge/build_corpus.py            # rebuild corpus.jsonl from vault git
$PY eval/durable_write_judge/judge.py --judge both      # 100 offline calls, secondary engine
$PY eval/durable_write_judge/score.py                   # rewrite report.md from the raws
$PY -m pytest tests/test_durable_write_judge.py -q      # the pinned clauses
```

Determinism: the corpus builder is `--since`/`--until`-bounded and path-ordered
with a per-class quota; Judge A's draw is seeded (`SEED = 580`); Judge B's is
inherited from the corpus order; both judge at `temperature=0`. Re-running
`judge.py` reproduces the same `example_ids`, and
`test_no_judged_sample_was_its_own_retrieved_example_in_the_shipped_run` proves
the shipped run is reconstructible from the shipped corpus.

## Known limits, stated before the number

- The judge cannot check a claim against the transcript — it judges internal
  plausibility, which is precisely the residue a static rubric can reach.
- `good` means *no signature fired*, not verified-clean. A judge flagging one may
  be right, which inflates the measured false-positive rate and makes the
  reported silent-pass rate **optimistic**.
- n is small: 20 bad across 5 classes, the smallest class at n = 2. One sample
  moves a class row by up to 50 points and an overall row by 5.

## Measured (2026-09-14, `report.md`)

Judge A (static rubric, 5 random labelled examples): recall-on-bad **45.0 %**,
false-positive-rate on good **23.3 %**, silent-pass rate **32.4 %**.
Judge B (same model, same preamble, 5 nearest-retrieved examples): recall **20.0 %**,
FPR **13.3 %**, silent pass **38.1 %**. Margin **−25.0 points** — the item's stop
rule fires: retrieval-conditioned judging did not transfer to this corpus, and no
step-7 shadow run was started. Two further findings, both in the report: the
retrieval leg's class-match rate is **36.8 %** with **68.4 % twin leakage**, so
selection quality is a real defect rather than a footnote; and every
`invented_url` miss carries a confident reason naming a *different* missing URL
than the one the repair removed — a judge can flag the right class for a reason
that does not match the record.
