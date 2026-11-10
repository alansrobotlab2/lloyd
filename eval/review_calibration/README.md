# Review-rung calibration

The `review` rung (`scripts/automod/review.py`, `architecture/automod.md` §4.5)
blocks landings from day one, so its judgment is checked against diffs whose
verdict is already known before its numbers are trusted. Each `*.json` here is
one case: a landed round (item, parent, commit, changed paths) and what the
grader must say about it.

```bash
.venvs/lloyd/bin/python -m scripts.automod.review_tools calibrate          # all cases, exit 1 on disagreement
.venvs/lloyd/bin/python -m scripts.automod.review_tools calibrate --only 544-first-cut
.venvs/lloyd/bin/python -m scripts.automod.review_tools fixture --name … --round SM_… --expect-kind retry --expect-unmet 4
.venvs/lloyd/bin/python -m scripts.automod.review_tools backfill --limit 5    # grade real landings → review_backfill.jsonl
```

Each case checks out the commit into a detached worktree under
`~/lloyd-work/review_cal_<name>/`, runs the grader exactly as the rung does
(same prompt, schema, live backend, `run_tests.sh`), and compares. Nothing
here touches a round, a landing or `promotions.jsonl`.

| case | what it is | expected |
|---|---|---|
| `544-first-cut` | `0f019f9`, the #544 landing that motivated the rung: the loopback seam untested, `or True` in a test; its clause (d) charge is refuted by the code in that same commit (see the case note) | `retry`; a test-honesty finding |
| `544-stripped-tests` | the same commit with its test file removed (`strip_tests`) | `retry`; a test-honesty/no-test finding |
| `392-small-met` | `ce713bf3`, a one-file landing whose author said `met`, and whose `expect` is provisional because live gradings of `parent..commit` answer `retry` against it | `retry` — the `pass` confirmed by a diff read on 2026-10-16 is the disagreement, not the verdict |
| `447-small-met` | `01219641`, a two-file landing whose author said `met` | `pass` — confirmed by a diff read on 2026-10-16 |

`provisional: true` marks an expectation nobody has confirmed by reading the
diff. Confirm or correct it, then drop the flag. A grader that agrees on all
four — including answering #544's clause (d) `met`, against the 2026-09-10
human review's charge, and answering #392's landing `retry`, against a diff
read that called it `pass` — is one whose backfill numbers (`scorecard`
metric 2, the audit delta) mean something.

Four validated cases validate the grader; they do not license scorecard metric
2 (the audit delta) as a headline rate. Metric 2 is computed over real landed
rounds from `promotions.jsonl`, never over these fixtures, and agreement on
four diffs — two of them the same #544 landing — says the grader can be run,
not that its number is a rate.

A case models a **range**, so a re-read of one is `git diff --name-only
<parent> <commit>`, never `git show <commit>`: `grade_commit` grades
`parent..commit` and hands the grader `base=parent`. `447-small-met` is the
case that catches the wrong read — its range is the round's two cherry-picks
(`bd033a19` + `01219641`) and prints both declared paths, while the tip commit
alone prints one. `tests/test_automod_review_tools.py` pins every shipped case's
`changed_paths` against that range for exactly this reason.

## The cause corpus (`cause_corpus.jsonl`) — #4026

Alongside the four graded cases sits a **cause-labelled disagreement corpus**: every
grader-versus-reality dispute that is already written down, one JSON object per line,
with exactly the keys `case_id`, `source`, `surface`, `verdict_given`,
`contrary_evidence`, `cause_class`, `resolvable_now`. `cause_class` is one of
`grader_defect`, `spec_ambiguity`, `graded_wrong_input`, `real_unmet_work`,
`grader_wrong_on_approved_work`, `unknown` — the point of the fixed set is that a change
to grader behaviour has to name the class it fixes, instead of the class of the last
incident being guessed at.

Two of the six fields carry the machine proof: `contrary_evidence` ends in
`graded_commit=… graded_parent=… cites=…`, and `scripts/automod/review_cause.py`
re-checks those citations against the git objects they name. `grader_defect` and
`graded_wrong_input` are legal **only** when a cited `evidence_path`/`evidence_line`/
`test_node_id` fails to resolve at the commit that was graded, or the graded range is not
the surface the clause named; a label whose citation resolves is refused by the loader.
That is what makes those two labels evidence rather than opinion, and why a row whose
source does not resolve to a real backlog item or promotion-ledger round cannot load.

`calibrate` prints one `cause <class>: <ok>/<n> agree` line per class present **before**
the aggregate, which stays the last line and still counts every case:
`scripts/automod/review_tools.py build-cause-corpus` rebuilds the corpus from the ledgers
and publishes the dated distribution to `~/lloyd-data/eval/review-calibration/`. Growing
the graded cases themselves to >=12 labelled fixtures is a second round.
