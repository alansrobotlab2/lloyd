# #623 pilot: autonomy runs graded against declared acceptance (2026-09-24)

`c85e8c92` made a task's `acceptance:` block gradeable; no task declared one.
This is the pilot: eight tasks given a block, and their history graded offline.

## The pilot tasks

Chosen for an objectively checkable ending that the judge's existing check
types can express (tool dispatch + the report's own shape). Each block was
written from the task's SKILL.md and the script's print statements **before**
any history was graded, so the history is not the fit set.

| task | checks |
|---|---|
| #24 Data Pipeline | `tool_called: Bash`; the gate outcome named (`files_processed` / `status=ran\|noop\|locked` / `PIPELINE_RESULT`) |
| #30 Intelligence Pipeline | `tool_called: Bash`; feed coverage `N of M`; held count; written count |
| #76 Queue Health Check | fleet line: `N runs`; failures |
| #79 Retention Sweep | `tool_called: Bash`; `workers.db` line reported; `tool_not_called: Edit`, `Write` |
| #80 OKF Conformance | `tool_called: Bash`; `max_tool_calls: 6`; violations count; ends on a statement, not `?` |
| #81 QMD Index Maintenance | `tool_called: Bash`; daemon-health line; orphan counts |
| #86 IV Metrics Series | `tool_called: Bash`; verdict line verbatim (or the exit-3 case); `tool_not_called: Edit` |
| #90 Corpus Shape Trend | `tool_called: Bash`; `max_tool_calls: 4`; one of the three report shapes; no Edit/Write |

Skipped on purpose (the #443 lesson — no prose criteria): #48, whose skill says
both "apply CASE/PUNCT" and "do not apply", so there is no one acceptance to
write; #39/#40/#38, whose done-state is a file on disk the judge's check types
cannot see (a `file_exists` type would be a judge change, not a pilot).

Validated with `scripts/autonomy/validate_tasks.py` (36 parse, 0 structural
warnings — it now also lints `acceptance:` blocks) and with the scheduler's own
`autonomy._parse_task_file` + `run_acceptance.parse_acceptance` (0 invalid).

## Historical readout (offline)

`scripts/autonomy/regrade_runs.py` joins each `status='success'`
`scheduled-task` row in `workers.db` to its session transcript and rebuilds the
dispatch trace `run_task` collects live (`trace_from_transcript`; parity with
the live trace is pinned by `tests/test_run_acceptance_regrade.py`). History
only reaches back to 2026-09-22 (the data-home wipe).

    python scripts/autonomy/regrade_runs.py --days 30 --tasks 24,30,76,79,80,81,86,90 \
        --workers-db ~/lloyd-data/workers.db --sessions-dir ~/lloyd-data/sessions --show-runs

| task | success runs graded | graded_fail | false-completion rate | 95% Wilson |
|---|---|---|---|---|
| #24 | 9 | 0 | 0.00 | 0–0.30 |
| #30 | 5 | 0 | 0.00 | 0–0.43 |
| #76 | 2 | 0 | 0.00 | 0–0.66 |
| #79, #80, #81, #86 | 1 each | 0 | 0.00 | 0–0.79 |
| #90 | 0 (new task) | — | None | — |
| **pooled** | **20** | **0** | **0.00** | **0–0.161** |

Window 2026-09-22T20:26Z → 2026-09-24T17:04Z. No success run lacked a transcript.

### Is the grader right? (hand label)

All 20 final reports read by hand against the skill: 20/20 did the job and
reported it → 20/20 agreement with `graded_pass`. Two regex hits matched the
right report for a slightly wrong span (#30 "5 `Skipping (already written`",
the written count was also present; #24 "no `status=locked` path", the
`PIPELINE_RESULT` line was also present) — correct verdicts, noted as the
regexes' known looseness.

The item's 10 pass / 10 fail hand-label cannot be filled: **there is no real
fail in the history.** In its place, a discrimination check: every pilot block
applied to the 46 success runs of *other* tasks (which do not do that job):

| block | own runs pass | other tasks' runs fail |
|---|---|---|
| #24 | 9/9 | 36/37 |
| #30 | 5/5 | 41/41 |
| #76 | 2/2 | 42/44 |
| #79 | 1/1 | 44/45 |
| #80, #81, #86 | 1/1 | 45/45 |
| #90 | — | 46/46 |

Cross-task false pass 4/348 (1.1%): the blocks are not vacuous — they fail runs
that did not do the job — so a 0/20 on the pilots' own history is a reading,
not a grader that cannot fail.

## Decisions

- **Bounded retry: not enabled.** 0 of 20 graded successes failed their
  acceptance (upper 95% bound 16%), so a retry has nothing measured to convert,
  and every retry costs a full run on a 2-slot pool. #544's effect ledger is
  live (`harness.effect_ledger.enabled: true`), so idempotency is no longer the
  blocker — yield is. Revisit only if the 7-day live readout shows a
  graded_fail rate above ~10% with hand-confirmed fails.
- **Pilot does not widen yet.** The grade is record-only and costs no engine
  call; keep the eight through the 7-day live readout, then widen to tasks
  whose done-state a text/dispatch check can see.

## What remains

The 7-day readout over live pilot traffic (`false_completion_rate(conn,
by="task_id")` against `runs.meta_json`) needs seven days of the pool running;
the worker pool is paused as of this writing.
