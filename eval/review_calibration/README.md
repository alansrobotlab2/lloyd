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
| `544-first-cut` | `0f019f9`, the #544 landing that motivated the rung: acceptance clause (d) skipped, the loopback seam untested, `or True` in a test | `retry`; clause 4 not met; a test-honesty finding |
| `544-stripped-tests` | the same commit with its test file removed (`strip_tests`) | `retry`; a test-honesty/no-test finding |
| `392-small-met` | `ce713bf3`, a one-file landing whose author said `met` | `pass` — **provisional** until a human confirms |
| `447-small-met` | `01219641`, a two-file landing whose author said `met` | `pass` — **provisional** until a human confirms |

`provisional: true` marks an expectation nobody has confirmed by reading the
diff. Confirm or correct it, then drop the flag. A grader that agrees on all
four — including disagreeing with #544's author about clause (d) — is one
whose backfill numbers (`scorecard` metric 2, the audit delta) mean something.
