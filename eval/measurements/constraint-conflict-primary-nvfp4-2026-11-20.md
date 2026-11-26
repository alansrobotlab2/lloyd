# Constraint-conflict cc11 ask rate — `primary-nvfp4`, fired 2026-11-19 23:39, finished 2026-11-20 00:24

**Question.** When a conflict task *withholds an argument* the hatched tool would
need — a write with no destination, a send with no recipient — does the model
**ask** for the missing argument, **invent** one, or use the control's legal
equivalent anyway? Item #4128 added the class (`cc11_conflict_*` plus its
matched control) because the violation rate on its own cannot tell a restrained
model from a blocked one: it scores "did not attempt the hatch", which the model
that stopped to ask a question and the model that quietly filled the argument in
with a plausible guess both earn.

**Answer.** On the NVFP4 checkpoint the model asked for the missing argument in
**1/3 = 0.3333** of its scored missing-argument conflict trials (95% CI
0.0615-0.7923), invented one in **0/3 = 0.0000**, and used the control's tool in
**0/3 = 0.0000**. The class denominator is stated with the number, always, and
the reason is in *The denominator this class owns* below. On the headline
measure the same firing put the conflict violation rate at **0.1818** against
the shipped baseline's **0.1667** — a first reading of a re-cut suite, not a
regression.

## Figure ledger

Every `k/n`, rate and CI below is transcribed from
`eval/measurements/primary-nvfp4_20261120.json`, which is committed beside this
file, and `tests/test_constraint_conflict_nvfp4_baseline_doc.py` re-parses that
JSON and fails on any cell that stops matching. Rates are the driver's, rounded
to 4 dp by the scorer; the CI is Wilson 95%.

| metric | k/n | rate | 95% CI |
|---|---|---|---|
| `violation_rate` | 6/33 | 0.1818 | 0.0861-0.3439 |
| `violation_rate_uncensored` | 6/29 | 0.2069 | 0.0985-0.3839 |
| `control_acted_rate` | 25/33 | 0.7576 | 0.5898-0.8717 |
| `missing_arg_ask_rate` | 1/3 | 0.3333 | 0.0615-0.7923 |
| `missing_arg_invention_rate` | 0/3 | 0.0000 | 0.0000-0.5615 |
| `missing_arg_control_used_rate` | 0/3 | 0.0000 | 0.0000-0.5615 |

`missing_arg_*` are the cc11 class only. `violation_rate` and
`control_acted_rate` are the #678 rates over the whole suite and share nothing
with them but the run.

## The denominator this class owns

The scorer prints the cc11 block on its own labelled line rather than appending
it to the rate list above, precisely so a reader cannot borrow the wrong `n`:

```
missing-argument class: 3 conflict / 3 control trial(s) scored, 0 unmeasurable
```

(`scripts/autoresearch/constraint_conflict.py:1011-1013` builds that line, and it
omits the `, 0 unmeasurable` tail when the count is zero — the console output for
this firing therefore ended at `scored`. `0 unmeasurable` is the JSON's own
`missing_arg_trials.unmeasurable`, and it is spelled out here because a `0/0`
in this class means *no task tested it*, not *nothing happened*.)

So: **quoting the 1/3 against `n=66`, or against the shipped baseline's `5/30`,
is the misread the comment at
`scripts/autoresearch/constraint_conflict.py:1003-1009` warns about** — "a reader
who thinks the two are the same denominator reads a 2/3 as a 2/66". The cc11
denominator is the trials of the scenarios that withhold an argument (3 conflict
and 3 control here, out of 11 conflict and 11 control tasks at 3 trials per
case), so the 1/3 and the 0.1818 are answers to different questions about
different sets of trials.

## Against the shipped baseline — rates, never k/n

The shipped baseline `eval/constraint_conflict/baselines/primary_20260924.json`
records a conflict violation rate of **0.1667**; this firing records **0.1818**.
Compared as rates, which is the only honest way to compare them: the two suites
have different denominators by construction (the 2026-09-24 suite is 10 conflict
tasks, this one is 11), so putting one firing's k/n beside the other's `5/30`
would be reading a suite change as a behaviour change.

The shipped baseline has **no ask-rate field at all** — it predates cc11, and its
own `suite` block is 10 conflict / 10 control. The 1/3 above is therefore the
**first ask-rate reading that has ever existed**, not a delta against anything.
Its CI is 0.0615-0.7923 because three trials is three trials: this is a floor
under the number, published so a later firing with a bigger denominator can be
compared to it, and it is not yet a baseline in the sense the 2026-09-24
violation rate is. Growing that denominator is #4243's three-visibility-arm work.

## Run identity — and what `primary-nvfp4` is

| field | value |
|---|---|
| `arm_label` / `model` | `primary-nvfp4` |
| `engine.model_id` | `primary` |
| `engine.base_url` | `http://127.0.0.1:8096` (`base_url_source: explicit --base-url`) |
| `engine.checkpoint_root` | `/home/alansrobotlab/lloyd/agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4` |
| `git_sha` | `9353ad43224c897a98954711a42b082dd9a18f3f` |
| `suite` | 11 conflict / 11 control, `trials_per_case` 3 |
| `visibility` | `described` |
| `measures` | `restraint` |
| `witness` | 66/66 trials carry a witness (`trials_with_witness` 66 of `trials` 66) |
| `excluded_trials` | 0 |
| `censored_trials` | 4 conflict / 1 control |
| started → finished | `2026-11-19T23:39:41-0800` → `2026-11-20T00:24:07-0800` |

**`primary-nvfp4` names a weight format, not a second engine.** The arm label is
this repo's way of saying "the primary, serving the NVFP4-quantised checkpoint at
the `checkpoint_root` above": `engine.model_id` is `primary`, the base URL is the
loopback port the primary already listens on, and `engine.alias` is null because
the URL was named on the command line rather than resolved from `config.yaml`.
Nothing in the artifact records a second serving process, a new GPU, or a rebuilt
interpreter — the driver is a client, and its only engine-side request is the
read-only `GET /v1/models` at `scripts/autoresearch/constraint_conflict.py:1162`
that reads the identity fields above. What this doc claims about the checkpoint
is only what the endpoint reported as its `root`; how the endpoint came to serve
that checkpoint on 2026-11-19 is a serving-side fact outside this artifact.

## The null firing of 2026-11-21, and the `/v1` rule

The same command fired again the next morning and measured nothing. Recorded in
`eval/measurements/primary-nvfp4_20261121.json`, committed beside this file for
the reason given in *Provenance*:

| field | value |
|---|---|
| started → finished | `2026-11-21T04:01:19-0800` → `2026-11-21T04:01:23-0800` (4 s) |
| `git_sha` | `5b59f98d314f4b4911b46e83a4d66baf69ad0a63` |
| `measures` | `not_measured` |
| `excluded_trials` | 66 — every trial in the suite |
| `engine.base_url` | `http://127.0.0.1:8096/v1` |
| `engine.checkpoint_root` | `""`, with `checkpoint_root_error: "'data'"` |

**Attribute a `git_sha` per firing:** the measured firing above ran at
`9353ad43…` and this null one at `5b59f98d…`, four commits' worth of tree apart.

The cause is one slash. `scripts/autoresearch/constraint_conflict.py:1162` is
`httpx.get(ident["base_url"].rstrip("/") + "/v1/models", timeout=5).json()["data"][0]`
— it appends `/v1/models` itself — so passing the API's version prefix in the
flag builds `http://127.0.0.1:8096/v1/v1/models`, that body has no `data` key,
and the subscript raises `KeyError('data')`, which is exactly the recorded error
string `'data'`. **Rule: `--base-url http://127.0.0.1:8096`, without the `/v1`
suffix.** The failure is loud in the artifact and silent on the console: 4
seconds, 66 excluded trials, and a JSON whose rates are all `null`, which is why
a firing like this has to be written down rather than diffed.

## Provenance — why the figures are inline, and what is not committed

Both firing summaries — `primary-nvfp4_20261120.json` and
`primary-nvfp4_20261121.json` — existed, before this file, in exactly that one
place each, and `.gitignore:42` (`*.json`) ignores both of them. That is why
every figure above is copied **into** this document rather than only cited by
path, and why both summaries are `git add -f`-tracked in the same commit as this
file: a document whose source is untracked is a document whose numbers nobody can
check, which is precisely the hazard a doc-only clause invites. This follows the
repo's existing convention for a measurement artifact a document depends on
(precedent: `eval/measurements/autoresearch-judge-compare-2026-09-24.json`,
tracked despite the same rule). `.gitignore` is untouched; `git check-ignore`
still matches both JSONs, and `git ls-files` still returns them — which is the
pair of facts `tests/test_constraint_conflict_nvfp4_baseline_doc.py` asserts, so
a later edit cannot quietly move the rule instead of the file.

`tests/test_constraint_conflict_nvfp4_baseline_doc.py` is what stops any of the
above from being an invented number: it parses the committed JSON and asserts
each cell of the ledger, each denominator, both `git_sha`s and the null firing's
error string. It skips the JSON-comparison nodes only if the JSON is not on disk
at all — which cannot happen in a checkout after this commit — and it fails
loudly, never skips, if this document is missing.

**Trials files.** Three untracked `.trials.jsonl` sit beside the JSON:
`primary-nvfp4_20261119.trials.jsonl` (100,421 B, with **no summary JSON**),
`primary-nvfp4_20261120.trials.jsonl` (100,421 B) and
`primary-nvfp4_20261121.trials.jsonl`. The two 100,421-byte files are *not* the
same bytes. The run driver names a trials file by **start** date and its summary
by **finish** date, and this run crossed midnight (23:39 → 00:24), which is the
likeliest reading of the orphan. This doc does **not** claim which trials file
the committed figures were scored from: settling that needs a `--rescore` pass
over the trials, and until then **the summary JSON is the authority** and the
`_20261119` file is an orphan of unknown relation.

**No engine call and no service restart** were made to write this document or
this test: nothing here opens a connection to `:8096`, and the invariant a reader
can re-check is that `ls eval/measurements/primary-nvfp4_*` still lists exactly
the files over the `_20261119` / `_20261120` / `_20261121` prefixes and no new
date prefix appears.
