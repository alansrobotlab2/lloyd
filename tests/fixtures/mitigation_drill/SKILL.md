---
name: mitigation-drill
segment: skills
description: 'Report the mitigation drill''s published readings — do not fire the
  drill. One Bash call: `curl -s http://127.0.0.1:8080/api/workers/status`, read its
  `.mitigation` block, report the JSON verbatim and one summary line giving each
  surface''s `n`, `median_seconds` (`-` when null) and the age of its newest `at`. A
  surface whose latest `classification` is `no-op` is a control regression to name
  plainly; `{"state": "never-run"}` or a stale newest reading is a report, not a
  failure. Use for the mitigation-drill autonomy task (#95, backlog 2432).'
tags:
- autonomy
- workers
- safety
status: active
category: autonomy
type: skill
timestamp: '2026-10-04T00:00:00'
---
# Mitigation drill

Two stop controls — `session_cancel` and `pool_pause` — are fired at a synthetic run and
judged by what they actually stop, not by what they are called. The drill appends each
reading to `~/lloyd-data/mitigation_drill.json`, and `GET /api/workers/status` publishes
each surface's median stop-time over the most recent 20 readings beside its latest one.

**You read that series; you do not produce it.** Since #2333 the drill is fired by the
worker pool's own maintenance seat — `workers/pool.py::_maybe_mitigation_drill` calls
`workers/maintenance.py::maybe_run_mitigation_drill`, which spawns what
`mitigation_drill_argv` builds — and `python -m scripts.mitigation_drill` is the writer of
every number you will report. Reading is what is left for a scheduled task: firing the
drill from a run is refused while a self-modification round holds the pool, which is
precisely why every day this task was armed produced a refusal and no reading, while the
seat went on taking readings the whole time.

## Do

1. `curl -s http://127.0.0.1:8080/api/workers/status` — one Bash call, and read its
   `.mitigation` block. It is either `{"session_cancel": {...}, "pool_pause": {...}}`,
   each surface carrying `classification`, `seconds`, `at`, `median_seconds` and `n`, or
   the no-readings marker described in step 4. Run nothing else: producing these numbers is
   the pool's job, and one GET is the whole of your tool budget.
2. Report the JSON **verbatim, all of it**, then exactly one summary line:
   `mitigation drill: session_cancel state=<classification> n=<N> median=<S> age=<A> | pool_pause state=<classification> n=<N> median=<S> age=<A>`
   where `<classification>` is that surface's `classification`, `<N>` its `n`, `<S>` its
   `median_seconds` in seconds, and `<A>` the age of its newest `at` — whole hours or
   minutes since that timestamp, like `4h` or `45m`. Print `-`, never `0`, for
   `median_seconds` when it is null: the dispatch-only `pool_pause` stops claims, never the
   run in flight, so it has no stop-time to report and a null median there is the honest
   answer rather than a missing measurement. Do not paraphrase, round, reorder or truncate
   the JSON: those numbers are the record.
3. A surface whose latest `classification` is `no-op` is a **control regression**. Name the
   surface and say plainly that the control no longer stops what it is supposed to stop.
   That is the finding this task exists to catch, and it is the only reason to expect a bad
   day from this report.
4. A `.mitigation` block reading `{"state": "never-run"}` — with or without an `error` key
   — or a newest `at` that is many hours or days old is **a report, not a failure**. Keep
   the line's shape, print `-` in every field the readings do not carry (so
   `state=- n=- median=- age=-` for a surface with no reading at all), say the drill has
   never fired or has not fired recently, and stop. Do not re-run anything to make a number
   fresher; a stale reading is the fact of the day, not a flake to be refreshed away.
5. `n` counts the readings the state file still stores — the writer keeps at most 20 per
   surface and drops the oldest. It is **not a cadence and proves no schedule**: the same 20
   readings can be stamped days apart or all within one second of each other. Never report
   `n` as "the last <X> hours" or as evidence the drill runs hourly; the ages in the line
   above are the only thing here that says whether the series is current.
6. A traceback, an unreadable response, or a response with no `.mitigation` block at all:
   report the last line of what you got and stop.

## Never

- Call Edit or Write. This task runs one read-only command and reports; the drill is the
  only writer of `~/lloyd-data/mitigation_drill.json`.
- Fire the drill yourself, or run anything else — no `python -m scripts.mitigation_drill`
  with or without `--wait-free-window`, no second GET to "confirm" or "refresh" a reading.
  A `no-op` is a measurement, not a flake, and a re-run buries the reading it would have to
  be compared against.
- Fix anything: no editing `scripts/mitigation_drill.py`, `app/mitigation_state.py`, a
  threshold or a control, no self-modification round, no backlog item. A `no-op` says a
  stop control regressed in production; a person decides what that means.
- Skip the report because the readings look old or empty. A never-run day is a report, and
  the median series is only trustworthy if the empty days are reported too.
