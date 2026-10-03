---
name: step-conformance-replay
segment: skills
description: Run the read-only step-conformance replay over the last week of real runs
  and report its numbers verbatim. Use for the step conformance replay autonomy task
  (task 93, backlog 2104). Prints a flag rate; files at most one backlog draft; never
  alerts anybody.
tags:
- monitoring
- autonomy
- memory
status: active
category: autonomy
type: skill
timestamp: '2026-10-03T00:00:00'
---
# Step conformance replay

One read-only command, and then a report. Nothing else.

## Do

1. `cd ~/lloyd && python3 scripts/step_conformance.py replay --days 7 --support 1.0 --escalate-writes`
   (default `--grace-hours 12`; pass `--grace-hours` only if the report's NOTE line says the
   grace window is swallowing recent runs).
2. Paste the printed lines **verbatim** into your report. The two that matter, in this order:
   `runs_flagged=N/M (x%) runs_pending=… deviations=… pending=…` and
   `escalations_filed=N/M …`. Quote them exactly — a paraphrased flag rate is the failure this
   task exists to prevent, and a number quoted without its denominator is not a number.
3. If the script prints `escalations_filed=1/1` (or more), name the draft line it printed too:
   `- task <id> <run_id> missing 'write:…' -> <path>`. The script wrote that file itself; you
   write nothing.
4. Exit code 0 with output is success. Report it and stop.

## Do not

- Do not write, edit, move or delete any file. This task has no file-writing step: the script
  opens the ledger and the trajectories store read-only, and the single write it may make is the
  backlog draft `--escalate-writes` files for a run that never wrote an artifact every other run
  of its task wrote.
- Do not run `learn`, do not pass `learn --out`, and do not pass `replay --json`. Publishing a
  baseline and writing a report file are both writes, and neither belongs to this task.
- Do not adjudicate a flag. A deviation is measured over unlabelled real traffic, so its
  false-alarm rate is UNMEASURED — the labelled rate comes from the detector's `validate`
  subcommand, not from you. A filed item is a `draft` a human reads at their desk, on purpose.
- Do not file, promote, close, re-triage or comment on any backlog item yourself; the script
  files, and only one shape. Do not open a self-modification round, and do not fix the task
  whose stage went missing — that is the reader's call.
- Do not send anything to anybody. No message, no injected context, no elevated-priority
  notification, no dashboard post. The guardian's record is that every rollback so far was a
  false positive, which is exactly why this task's answer to a flag is a draft and not a page.

## If it fails

- The script errors (no trajectories, unreadable ledger, empty window): paste the traceback
  tail verbatim, say which store it named, and stop. Do not create, copy or repair a store to
  make it run.
- Zero trajectories in the window, or `tasks_without_baseline` is every task: report that
  verbatim as the finding. An empty window is not a pass — say so in those words.
