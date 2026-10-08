---
acceptance:
  objective_checks:
  - type: tool_called
    value: Bash
  - type: max_tool_calls
    value: 4
  - type: regex
    value: 'mitigation drill: session_cancel state=\S+ n=(?:\d+|-) median=(?:[\d.]+|-) age=(?:\S+) \| pool_pause state=\S+ n=(?:\d+|-) median=(?:[\d.]+|-) age=(?:\S+)|\w+(?:Error|Exception)\b'
  - type: tool_not_called
    value: Edit
  - type: tool_not_called
    value: Write
  rubric:
  - accuracy
agent_id: worker
auto_advance: false
category: technical
created: '2026-10-04T03:30:00Z'
description: 'Daily mitigation-drill report (#703, armed by #2153; made a reader by
  #2432): `curl -s http://127.0.0.1:8080/api/workers/status` — one Bash call, and read its
  `.mitigation` block. Do NOT fire the drill: `python -m scripts.mitigation_drill` is run by
  the worker pool''s maintenance seat and is the only writer of
  `~/lloyd-data/mitigation_drill.json`, and a run of your own is refused while a
  self-modification round holds the pool, which is why every day this task has so far
  produced a refusal and no reading. Report the JSON VERBATIM (all of it), then exactly one
  line: `mitigation drill: session_cancel state=<classification> n=<N> median=<S> age=<A> |
  pool_pause state=<classification> n=<N> median=<S> age=<A>` — `<S>` is `median_seconds` or
  `-` when it is null (the dispatch-only `pool_pause` has no stop-time, so print `-`, never
  `0`) and `<A>` is the age of that surface''s newest `at`, like `4h` or `45m`. A surface
  whose latest `classification` is `no-op` is a CONTROL REGRESSION: name the surface and say
  the control no longer stops what it is supposed to stop. `{"state": "never-run"}` (with or
  without an `error`) or a stale newest `at` is a report, not a failure, not a finding and
  not a retry — keep the line''s shape, print `-` in the fields the readings do not carry,
  and stop. `n` counts the readings the state file still stores; it is not a cadence and
  proves no schedule, so never call it "the last <X> hours". Never call Edit or Write, never
  run the drill or re-run the GET to confirm a `no-op` or refresh a stale reading, never
  edit the script, the state file or a threshold, never open a self-modification round: a
  person decides what a regression means.'
expected_error_patterns: []
failure_count: 0
frequency: daily
id: 95
infra_failure_count: 0
last_attempt: '2026-10-08T13:03:52.962917+00:00'
last_run: '2026-10-08T13:03:52.962917+00:00'
max_retries: 3
model: primary
name: Mitigation drill
next_run: '2026-10-09T13:00:00+00:00'
notify_on_complete: false
preemptible: true
preferred_hours:
- 6
- 7
priority: low
runs_per_day: 1
scheduled_at: ''
segment: autonomy
skill_name: mitigation-drill
status: up_next
tags:
- autonomy
- workers
- safety
timeout_seconds: 300
title: Mitigation drill
type: autonomy
updated: '2026-10-08T13:03:52.962917+00:00'
---

> **This body is documentation and the machine-written activity log — not an instruction channel.** It is not delivered to the worker: `_build_task_prompt` (`~/lloyd/app/autonomy.py`) renders only the `skill_name` SKILL.md and the front-matter `description`, so a step that lives only below this line reaches no run.

# Mitigation drill

Backlog #2153, armed from #703's owed entry; #2432 rewrote the run from a firer into a
reader. `scripts/mitigation_drill.py` has existed since `d59cdd09`/`fded0a6b`. #2153 added
the bounded per-surface history (20 readings, oldest dropped) and the `median_seconds`/`n`
the route now publishes, which is what makes a reading series exist at all. #2333 then moved
the trigger itself onto the pool's maintenance seat — `workers/pool.py::_maybe_mitigation_drill`
calls `workers/maintenance.py::maybe_run_mitigation_drill`, which spawns what
`mitigation_drill_argv` builds — because this task, the once-a-day caller, was refused by the
round hold on every day it tried and wrote no reading. The reporting half is what is left for
a scheduled task, and this file plus `skills/mitigation-drill/SKILL.md` are the whole of it:
one GET, one reading report. The renderer that decides what a run receives is
`_build_task_prompt` (`app/autonomy.py`), which embeds the skill and the front-matter
`description` — which is why a step has to be in one of those two, never only here.

## Why the reporter reads instead of firing

The drill reads `round_hold` from `/api/workers/status` and refuses to fire a control while
a self-modification round holds the pool, so it never adds load while a round or the landing
behind it wants the box. On a box that runs rounds as often as this one, a task that fires
the drill is told "refused" most weeks, and a red the task itself says to ignore is how the
one red that matters gets ignored too. Reading the published block cannot be refused: the
seat has already paid for the measurement, `GET /api/workers/status` answers with whatever
was last written, and the report is the same shape on a busy box and an idle one.
`round_hold` stays the reason the drill sometimes writes nothing, which is why `never-run`
and a stale newest `at` are spelled out as reports rather than failures.

## Verification after landing

Two scheduled runs of #95 from now, `runs.summary` for `task_id='95'` in
`~/lloyd-data/workers.db` must carry the per-surface `n`/`median`/`age` line rather than a
`round_hold` refusal — owed-check's to measure, and nothing in this file can produce it
early. The series itself belongs to the pool's seat: `curl -s localhost:8080/api/workers/status`
shows `mitigation.session_cancel.n` and its `median_seconds` moving whether or not this task
ran that day, and `pool_pause.median_seconds` stays null by design because a pause stops
claims, never the run in flight.
