---
acceptance:
  objective_checks:
  - type: tool_called
    value: Bash
  - type: max_tool_calls
    value: 6
  - type: regex
    value: 'runs_flagged=\d+/\d+'
  - type: tool_not_called
    value: Write
  - type: tool_not_called
    value: Edit
  rubric:
  - accuracy
agent_id: memory
auto_advance: false
category: technical
created: '2026-10-03T07:15:00Z'
description: 'Read-only nightly step-conformance replay (#2104), per skill step-conformance-replay:
  `cd ~/lloyd && python3 scripts/step_conformance.py replay --days 7 --support 1.0 --escalate-writes`,
  then paste its printed lines VERBATIM — above all the `runs_flagged=N/M (x%) runs_pending=...`
  line and the `escalations_filed=N/M ...` line, each with its denominator. Read-only: never pass
  `learn --out` or `replay --json`, never open the ledger or the trajectories store for writing,
  never write or edit or delete a file yourself, and never fix or adjudicate the task whose stage
  went missing. The only write allowed in this whole job is the one the script makes for you: a
  backlog DRAFT for a flagged run that never wrote an artifact every other run of its task wrote.
  Do not file, promote, close or re-triage any backlog item, do not inject anything into anybody''s
  session or send any message or notification of any kind, and do not open a self-modification
  round. An empty window or zero baselines is reported as the finding, never as a pass.'
expected_error_patterns: []
failure_count: 0
frequency: daily
id: 93
infra_failure_count: 0
max_retries: 3
model: primary
name: Step conformance replay
next_run: '2026-10-03T21:10:00+00:00'
notify_on_complete: false
preemptible: true
preferred_hours:
- 4
priority: low
runs_per_day: 1
scheduled_at: ''
segment: autonomy
skill_name: step-conformance-replay
status: up_next
tags:
- autonomy
- monitoring
- memory
timeout_seconds: 600
title: Step conformance replay
type: autonomy
updated: '2026-10-03T07:15:00+00:00'
---

> **This body is documentation and the machine-written activity log — not an instruction channel.** It is not delivered to the worker: `_build_task_prompt` (`~/lloyd/app/autonomy.py`) renders only the `skill_name` SKILL.md and the front-matter `description`, so a step that lives only below this line reaches no run.

# Step conformance replay

Backlog #2104. The detector from #673 shipped read-only and had never been run by anything: 0 of
the 37 task files here named `step_conformance`, the job queue carried 0 payloads naming it, and
no file under `~/lloyd-data/autonomy-runs/` contained a `runs_flagged` line. This task is the
missing caller — the report half. It deliberately has **no alert surface**: `notify_on_complete:
false`, no injected context of any kind, and the one escalation it can produce is a backlog
`draft` (see `skills/step-conformance-replay/SKILL.md` for why a draft and not a page — the
guardian's record is that every rollback so far was a false positive).

Why the command carries `--escalate-writes` while the item's clause 5 names the bare replay: the
read-only default is what a human gets when they run it by hand (pinned by
`tests/test_step_conformance_escalation.py::test_replay_without_the_flag_files_nothing`), and
clause 4's filer needs a caller or it is dead code that never files. The flag adds exactly one
behaviour — file a draft for a missing *written* artifact — and adds no surface: still no message,
still no queue promotion.

`--days 7` is the window, `--support 1.0` is strict support and already the script's default; both
are spelled out because a baseline whose support was quietly relaxed would print a flag rate that
means something else. `--grace-hours 12` (the default) keeps yesterday's in-flight run out of the
count.

## Owed after landing (owed-check adjudicates, not this task)

- After one nightly cycle: a run record under `~/lloyd-data/autonomy-runs/93/` carrying the
  `runs_flagged=` line.
- `python3 scripts/step_conformance.py replay --days 14` over live traces emits no step name
  containing `$(` or an unbalanced `(` — the #2104 clause 5 grep goes 4 → 0.
- On the 2026-09-24 task-39 evidence the escalation files exactly one draft; on a `tool:TodoWrite`
  deviation, zero.
- Owed-check ruling: is a 0.29% flag rate a tolerable draft-filing rate for this surface, or should
  the write-only escalation narrow further.

## Activity Log
