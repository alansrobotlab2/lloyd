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

## The independently labelled set (`independent/`) — 2026-11-22

The four cases above were confirmed, where they were confirmed at all, by the
same loop that runs the grader, and after the 2026-10 runs disagreed two of them
were re-marked toward what the grader had answered. A set like that cannot say
how often the grader is wrong. `independent/` is the start of one that can: each
case is a real landed round whose label comes from a reader that is **not** the
grader (a different model, given the item's clauses and `git diff parent..commit`
and no prior review), and each `note` says who read it, when, and how sure.

```bash
.venvs/lloyd/bin/python -m scripts.automod.review_tools calibrate --set independent
```

`calibrate` prints the two errors apart — **false met** (a `pass` on a case
labelled `retry`: a bad change lands) and **false refusal** (a `retry` on a case
labelled `pass`: good work is sent back) — because one agreement figure hides
which of the two a grader makes.

What it holds, and what it cannot yet measure:

- 12 cases, all labelled `pass`: six rounds whose first review was a refusal and
  six that passed first time, drawn at random (seed 20261122) from the 95 landed
  rounds of the previous 14 days with 40–450 inserted lines and at least one
  code file. Three are `provisional` (a clause a reader could not settle
  without running something).
- **On its own it measures false refusal only.** It holds no `retry` case, so
  alone it reports false met — the costly error — `unmeasured`, not zero. The
  ledger cannot supply one: of 41 owed-check reopens, one follows a landing, and
  every rollback on record was a false positive. The `retry` cases are made, and
  live in `negative/` (next section).
- **A label here is never re-marked to match a grader.** It changes only on a
  named re-read, and since 2026-11-22 the lock below enforces that.

## The known-bad set (`negative/`) — 2026-11-22

21 cases whose correct verdict is `retry`, none labelled by asking the grader.
`scripts/automod/calibration_cases.py` is the generator and the reader; each
fixture says which of two routes its label came by (`label_source`).

**`constructed` (15) — true by construction.** A landed round with one thing a
clause demands taken back out. All three transforms are the same operation,
"restore these paths to the parent's version", and differ in what is restored:

| `transform.kind` | what is restored | cases |
|---|---|---|
| `strip_tests` | the round's test files | 6 |
| `revert_source` | one changed source file, tests and claims kept | 5 |
| `drop_doc` | the doc a clause names | 4 |

Each fixture records the clause it breaks with that clause's words at the time
(`violates`), a `proof` — a fixed string that must or must not be in one file of
the built tree, checkable with `git show` and no model — and `tests_rung`: what
the round's own test files did on the defect, measured by running them
(2026-11-22). **10 are `red` and 5 are `green`.** A red case cannot land on a
false `met`, because the `tests` rung stops it first; a green one can, so
`calibrate` counts those five apart (`tests rung green, review alone: k/n`).
Every red failure was read and is a node of the round's own that the removed
change was there to satisfy.

A constructed commit exists nowhere until it is asked for. `materialize` rebuilds
it with git plumbing — a private index, `write-tree`, `commit-tree` under a fixed
identity and date — so its sha is a pure function of `(source_commit, parent,
transform)`, the fixture pins it, and a rebuild that hashes to anything else is
refused rather than graded. No ref is written and nothing is checked out; the
object is loose in the repo `calibrate` runs in. This is why a case here carries
`strip_tests: false` where `544-stripped-tests` carries `true`: that one applies
its transform at grading time, these have it in the commit.

**`refused_then_fixed` (6) — a refused head a reader confirmed.** A round whose
earlier head the rung refused and whose later head landed, kept only where a
reader who is not the grader read `git diff <refused head> <landed head>` and
found the refusal named a real defect that the later head repaired (`refusal`,
`evidence`, `fixed_by`). 200 such rounds have both heads still in the repo; eight
were read, six kept, and two set aside because the diff alone did not settle
them (one needs the live vault, one a mutation run). The rest were not read.
They are the only cases whose commits are not on `main`: each is reachable from
`refs/automod/rounds/<round>` in `~/lloyd` and nowhere else.

**These six cannot give an unbiased false-met rate**: each was selected *because*
the grader refused it once, so re-grading it measures run-to-run consistency on
defects it has already caught. `calibrate` prints false met per label source for
that reason; the constructed 15 are the rate, the six are a floor on agreement.

One case per round, and no round is also in `independent/`. Add one with

```bash
.venvs/lloyd/bin/python -m scripts.automod.review_tools defect --name 4999-dropped-doc \
    --round SM_… --kind drop_doc --path architecture/x.md --violates 3 --why "…"
.venvs/lloyd/bin/python -m scripts.automod.review_tools lock-labels
```

## The lock (`labels.lock`)

One line per case of `independent/` and `negative/`: its label and a hash of
`(name, label, parent, commit, source_commit, transform)`.
`tests/test_calibration_cases.py::test_the_shipped_labels_are_the_locked_ones`
fails when a fixture stops matching its line. To change a label, add to the
fixture

```json
"relabel": {"from": "pass", "to": "retry", "by": "<who re-read the diff>",
            "on": "<date>", "reason": "<what the re-read found>"}
```

and run `lock-labels`, which refuses the change without that record, refuses a
`by` that names the grader, refuses a new diff under an old name (that is a new
case), and refuses to drop a locked case (retire one by editing the lock by hand
in a reviewed commit). **The grader's answer is never a valid reason** — these
cases exist to measure it. The four top-level cases are not locked: two of them
are the re-marks this rule was written against.

## Both rates in one run, and the trend

```bash
.venvs/lloyd/bin/python -m scripts.automod.review_tools calibrate \
    --set negative --set independent --record
```

`--set` repeats; the run ends with one combined line,

    false met k/N (95% Wilson lo–hi); false refusal k/N (95% Wilson lo–hi) [false met by label source: constructed k/n, refused_then_fixed k/n] [tests rung green, review alone: k/n]

(intervals from `eval/stats.py::wilson_ci`). `--record` appends one dated row to
`review_calibration.jsonl` in the automod state dir, so the two rates become a
series; an `--only` run is written `partial` and stays out of it.
`calibration_cases.scorecard_reading()` returns the newest full run and the one
before it for a scorecard row (`render_scorecard_reading` formats it), and says
`measured: False` when there is none. Run it from `~/lloyd` only — it needs the
live backend, the engine, and for six cases the round refs. 33 cases at the
ledger's review durations (median 233 s, p90 481 s over the last 14 days) is
about 2.1 h of engine time, 4.4 h if every case ran at p90; a case is cut at 600 s.

What is measured is the rung, not the bare model: `calibrate` runs the grader and
then the same deterministic rails and `decide` the gate runs, so a `met` the
rails downgrade counts as the refusal it would have been. No precheck fires on a
`strip_tests` case (there is no changed test file for one to read), so those are
decided by the grader's clause verdicts and the evidence rail alone.

A constructed case is graded with `tests_passed=True` — as if the `tests` rung
were green — whether or not it would be. That is the condition under which the
review is the only thing between the defect and a landing, and it is the lenient
direction: it can only make a false `met` easier.

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
