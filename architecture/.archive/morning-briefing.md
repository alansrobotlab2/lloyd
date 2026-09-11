---
segment: architecture
tags: [architecture,lloyd]
relations:
  related-to:
  - autonomy/38-nightly-reflection-signals.md
  - architecture/nightly-skills-management.md
  - architecture/skills.md
  - architecture/tools.md
  - architecture/memory.md
  - architecture/autonomy-system.md
  - architecture/index.md
  - architecture/infrastructure.md
  - architecture/nightly-reflection.md
  - architecture/background-runs.md
  - autonomy/68-morning-brief-triage.md
type: reference

---














# Morning Briefing Architecture

**Live — but not the job this document was written for.** `morning-briefing`
names two things. One runs every fifteen minutes and triages mail and calendar
into whatever chat session Alan is actually sitting in. The other — the 6 am
"read the overnight reports and announce a synthesis" job the sections below
were written about — no longer runs at all, and no autonomy task binds it. The
name outlived the job.

## What runs today

Autonomy task **#68**, `~/obsidian/autonomy/68-morning-brief-triage.md`, whose
`name:` is *Email & Calendar Triage*, running the vault skill
[`morning-brief-and-triage`](../../skills/morning-brief-and-triage/SKILL.md).
It scans today's calendar and unread mail, classifies each surviving item
`act-now` or `fyi`, and injects the result into the user's session. Nothing
about it synthesises the nightly pipeline.

| | |
|---|---|
| **Cadence** | `scheduled_at: '*/15 * * * *'`, `frequency: every-15min` — and **no `preferred_hours` at all**, so it runs around the clock rather than at a morning hour |
| **Status** | `up_next` |
| **Model** | `model: secondary` — Qwen3.6-35B-A3B on llama.cpp at `:8091` |
| **Budget** | `timeout_seconds: 600` in the task file, under the `scheduled-task` source's `max_duration_seconds: 3600`. There is no dollar cost |
| **Runner** | the worker pool's `scheduled-task` source (`workers/sources/scheduled_task.py`) → `autonomy.run_task` |
| **Delivery** | `session_inject_context` at `ambient` or `notable` — see [Delivery](#delivery-the-seam-that-broke-on-2026-09-07) |
| **Other frontmatter** | `agent_id: memory`, `priority: low`, `preemptible: true`, `notify_on_complete: true` |

Four claims this section used to make were true of nothing that has run
recently, and are recorded here rather than deleted. The schedule was given as
**7:00 AM PST daily** — even the skill it linked to said 6:00 AM, so the two
never agreed. The model was given as **Claude Sonnet 4.6** and the budget as
**<$5 per run**; this stack runs local engines, and as `nightly-reflection.md`
puts it, there is no dollar budget, the cost is GPU-hours. **`agent: memory
(isolated session)`** is the half-right one: `agent_id: memory` is still real
frontmatter, read at `app/routers/autonomy.py:91`, but the isolation is now a
*recorded* background session — `run_task` mints one with
`new_background_session_id("autonomy")` (`autonomy.py:855`) on platform
`autonomy`, with Inner Voice off unless the task asks for it.

## What the retired job was

The nightly pipeline's final job: read the overnight reports and announce a
summary before Alan's day started. **Nothing schedules it.** Its skill is
archived at `~/obsidian/skills/.archived/morning-briefing/`, and an unbound
copy survives as `~/obsidian/skills/nightly-morning-briefing/` — last touched
2026-09-08, bound to no autonomy task, still carrying "Runs at 6:00 AM PST
daily" and "Use announce mode to deliver to chat channel" in its own text.

That last phrase is where this document's **announce mode** came from. There is
no announce delivery mode for autonomy tasks anywhere in the code; the only
`announce()` in the tree is the guardian's alert fan-out, which is a different
mechanism for a different purpose. Task `autonomy/6-morning-briefing.md`, which
`architecture/nightly-skills-management.md` still lists as a relation, does not
exist in any form. The predecessor #72 *Inbox morning summary* sits archived at
`status: draft`, past its one-shot `2026-05-02T09:00:00` schedule, written off
in `autonomy/meta-analysis-2026-06-03.md` as "stuck in `draft`, past one-shot
schedule, no skill, superseded by #68".

## How it decides what to surface

`~/obsidian/skills/morning-brief-and-triage/SKILL.md` is the whole procedure.
Two files beside it hold the state: `references/prefs.yaml` (`watch_senders`,
`watch_domains`, `mailing_lists`, `block_senders`, `block_domains`,
`priority_keywords`, `ignore_subjects`, `watch_calendars`) and
`scripts/state.json` (`notified_emails`, `notified_calendars`, `last_run` —
26 KB of already-surfaced ids, pruned to 24 h each cycle).

Calendar comes from `calendar_events` and mail from `email_recent`, both
Thunderbird bridge tools (`agent_mcp/thunderbird.py:75,93`), advertised only
while Thunderbird and its bridge are up. Filters run blocklist first, then
`ignore_subjects`, then the allowlist and `priority_keywords`, then per-item
dedup, then a soft reject of anything newsletter-shaped; at most 3 calendar
items and 5 emails survive. Each survivor is classed `act-now` (an event inside
~3 hours, a deadline inside 48 hours, a personally addressed ask) or `fyi`, and
**a single `act-now` item changes the routing of the whole signal**.

Step 1 is read-only on purpose. It used to read the id lists, keep every one of
them, and write them back — a no-op round trip whose only observable effect was
a new way to fail: on `run_68_20260906_225158` the model reproduced the heredoc
without `os.path.expanduser` and the step died on a literal `~`, counted as a
tool error against a run that had produced the brief correctly.

## Delivery: the seam that broke on 2026-09-07

The skill calls `session_inject_context` (`agent_mcp/ambient.py:178`) with
`source="autonomy:morning-brief-and-triage"` and an hourly `dedup_key`, and the
`priority` it picks decides which of two mechanisms carries it:

- **`ambient`** (nothing is act-now) → `POST /api/sessions/{id}/inject-prefetch`
  (`app/routers/sessions.py:931`). No turn, no tokens, no transcript noise: the
  entry waits in a queue and is drained into the `<context>` block of the
  user's *next* turn.
- **`notable`** (anything is act-now) → `POST /api/sessions/{id}/inject`
  (`app/routers/sessions.py:868`), a real ambient turn. The skill is told never
  to escalate to `urgent` — that is reserved for safety and security.

**Which session counts as "the user's" is the whole problem.** The producer
passes no session id, so `_resolve_session_id` (`agent_mcp/ambient.py:164`)
asks `GET /api/sessions/active` (`app/routers/sessions.py:985`), which is
`sessions_io.get_active_session_id()` (`app/sessions_io.py:228`): the last
session to receive a user turn, else the most recent user session by mtime
inside 24 hours.

On **2026-09-07** that answered with a machine. The brief — a MockBOT meeting
that night — was injected into a backlog-triage **worker** session that had
finished 77 seconds earlier, and was answered there, to nobody. Worker turns go
through the chat path so that they can have Inner Voice, which means they set
`_last_user_session_id` exactly as a human typing does. The fix is one
definition of "a human reads this": `NON_USER_PLATFORMS = frozenset({"autonomy",
"worker"})` (`app/sessions_io.py:94`), applied through `is_user_session` to
**both** resolution rules. `worker` joined the set that day; `autonomy` had been
excluded from the start, which is also why a task can never inject into its own
run session.

It is a **deny-list on purpose**: a platform this code has never heard of stays
eligible, because silently losing a client's briefs is worse than delivering one
to an unexpected session. `tests/test_active_session_resolution.py` pins all of
it, that case included.

Two details that are easy to get backwards:

- **`/inject` answers 409, not a 200 "skipped"** (`app/routers/sessions.py:906`),
  so `session_inject_context` reports `ok: false` and **no producer records the
  notification as delivered**. A 200 would make a lost brief look like a sent
  one, which is the failure being closed.
- **`/inject-prefetch` carries no such gate** — it only 404s on a session that
  does not exist. The ambient path, which is this skill's common case, is
  protected at *resolution* rather than at the endpoint. That is sufficient only
  because the producer lets the server resolve the session; a caller passing an
  explicit worker session id would still be accepted there.

## The second channel: Discord, and `[SILENT]`

Injection is not the only way a run reaches a human. #68 carries
`notify_on_complete: true`, so `scheduled_task.py:297-305` posts the run's
response preview to the Discord home channel through
`app/discord_notify.py::_discord_notify_task_complete`.

`[SILENT]` is what keeps a 15-minute job from being a 15-minute notification.
`autonomy.py:578` tells every task to answer with exactly `[SILENT]` and
nothing else when there is nothing new, and `autonomy.py:1177` records the run
as silent on `final_response.strip() == "[SILENT]"`.

Three checks, two semantics, and the split is worth knowing before editing any
of them: the run record (`autonomy.py:1177`) and `discord_notify.py:28` both
test for an **exact** `[SILENT]`, while the call site above the latter
(`scheduled_task.py:297`) skips the notification when `[SILENT]` appears
**anywhere** in the preview. So a response that merely mentions the token is
still recorded as a real run while its Discord post is dropped.

The skill's own last line is written against the equality reading, because the
loose one cost a delivery: `[SILENT]` inside a longer response suppresses
nothing, so a paragraph ending in it was delivered to Alan in full, as noise
(observed 2026-09-03). Hence the rule as stated — *your entire response must be
exactly `[SILENT]`*.

## The retired job's inputs and output, corrected

Kept as the record. The purpose was real enough: the nightly pipeline produced
several reports, and without synthesis Alan would have had to read five files to
learn what happened overnight. The paths, however, were wrong in a way worth
naming — this table said `memory/reflection/…`, and that directory does not
exist in the vault. The reflection chain writes to `~/lloyd/_pipeline/reflection/`,
and three of the six documented sources were never written by anything.

| Source, as documented | Reality |
|---|---|
| `memory/reflection/signals-latest.md` | real, at `_pipeline/reflection/signals-latest.md` |
| `memory/reflection/test-results-latest.md` | never existed — there is no behaviour-test job |
| `memory/reflection/prompt-audit-latest.md` | never existed — there is no prompt-audit job |
| `memory/reflection/day-synthesis-latest.md` | absent; the `day-end-synthesis` skill is archived |
| `memory/learnings/YYYY-MM-DD.md` | real, written by #40 (config) |
| Backlog API | real — `~/obsidian/backlog/` and the `backlog_*` tools |

[[nightly-reflection]] is the authority there and says it plainly: "Jobs 3 and 4
do not exist… No autonomy task has ever written any of them."

The documented output, `memory/reflection/morning-briefing-latest.md`, exists
nowhere on disk. **Today's run writes no report file at all.** Its output is the
injected signal plus the updated `scripts/state.json`; the durable record is the
run record under `autonomy-runs/68/` and the recorded session transcript. The
six documented sections (Overnight Pipeline / Calendar / Email / Backlog / Open
Threads / Today's Focus) describe the retired job. The live signal has three,
each omitted when empty, and the run exits without injecting when nothing
survives filtering:

    Brief + Triage — YYYY-MM-DD HH:MM PST

    🔔 Act now:
    - [item, with why it is time-sensitive]

    📬 FYI:
    - [sender] — [subject] (1-line context)

    📅 Today: [event at time], …

## Nightly automation sequence

The table this section used to carry listed eleven jobs at fixed clock times,
under names — `reflection-synthesis`, `reflection-vault`, `reflection-skills`,
`reflection-audit`, `reflection-test`, `reflection-backlog`,
`periodic-memory-capture` — that match no autonomy task on disk, and pointed at
three skills that are now archived (`day-end-synthesis`,
`nightly-reflection-knowledge`, `morning-briefing`). Two of the skills it named
do still exist and are bound to nothing: `nightly-prompt-audit` and
`nightly-behavior-test` are the "Jobs 3 and 4" that [[nightly-reflection]]
records as never having run.

The fixed clock times were the deeper error. Autonomy jobs are gated by
`preferred_hours` windows in machine-local time plus `depends_on`, not by a
cron minute — so a table of exact times reads as a schedule that is being kept
when it is really a schedule nothing enforces.

What actually runs, read from the task files:

| Task | Window | Skill |
|---|---|---|
| #38 Nightly Reflection — Signals | `preferred_hours` 22–04 | `nightly-reflection-signals` |
| #42 Nightly Reflection — Knowledge Analysis | 22–04 | `nightly-reflection-knowledge-analysis` |
| #39 Nightly Reflection — Knowledge Write | 01–04 | `nightly-reflection-knowledge-write` |
| #40 Nightly Reflection — Config | 02–04 | `nightly-reflection-config` |
| #60 Knowledge Health Report | hour 04 (`scheduled_at: 04:30:00 PST`) | `knowledge-health-report` |
| #58 Nightly Skill Consolidation | `preferred_hours: []` | `nightly-skill-consolidation` |
| #83 Nightly Skills Management | no window | `nightly-skills-management` |
| #77 Weekly Backlog Hygiene | `0 4 * * 0` | `weekly-backlog-hygiene` |
| **#68 Email & Calendar Triage** | **none — every 15 min, all day** | `morning-brief-and-triage` |

The 15-minute memory-capture job is archived too
(`autonomy/_archived/25-memory-capture.md`), so the only quarter-hourly job
left in this list is #68 itself. [[nightly-reflection]] is the authority on the
chain and its dependencies; the table above exists only so the one job this
document is about can be located among them.

## Known defect

Backlog **#481** is open against the live skill. Step 2 tells the agent to skip
a missing calendar source **silently** ("do not retry, do not treat it as an
error"), which contradicts the degraded-run attribution guardrail added to the
bottom of the same file on 2026-08-30 — a run whose source died must name it.
Both instructions are live in one document about 150 lines apart, and Step 3
has no missing-tool branch at all. The 2026-08-29 "succeeding but blind" run
(`0 emails` against a normal 15–21, after a ~24 h IMAP/Thunderbird bridge gap)
is the shape of failure that then reproduces by instruction rather than by
accident.

## Related docs

- [[nightly-reflection]] -- the nightly chain, rewritten from audited reality
- [[autonomy-system]] -- the scheduled task fleet and the five gates on "due"
- [[background-runs]] -- why every background run is recorded, and what makes a session a worker's
- [[groundskeeper]] -- vault maintenance
- [[nightly-skills-management]] -- skills management
- [[memory]] -- memory system architecture

