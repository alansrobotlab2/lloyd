---
segment: architecture
tags: [architecture, lloyd, autonomy]
type: reference
status: implemented
date: 2026-10-12
---

# The autonomy jobs

What each scheduled task is *for*. [[autonomy]] is the mechanism — the five
due-gates, the failure ladder, the deadline anchor, the run records; this
document is the fleet it dispatches, one entry per job, grouped by what the job
is trying to accomplish.

**Artifact paths.** Every bare `_pipeline/` in this document means `~/lloyd-data/_pipeline/`,
not a path under `~/lloyd`: runtime data left the code tree in `6426668b`, and
[[data-home]] owns the layout it moved to.

**The skill is the job.** A task file is frontmatter: a schedule, a set of
gates, and a `skill_name`. The procedure — the phases, the output contract, the
incident history, the standing non-findings — lives in the vault at
`~/obsidian/skills/<skill_name>/SKILL.md`, which `autonomy.run_task` loads as
the prompt. So changing *what a job does* is a vault edit that takes effect on
the next run with no deploy, and changing *what the scheduler will let it do* is
a code change in `~/lloyd`. The vault is a live shared tree with no PR path and
no gate: a saved file is a deploy, and the run after the save is the one that
sees it.

**This document deliberately does not restate the schedule.** Status, `last_run`,
`next_run`, `preferred_hours` and `failure_count` are on disk in the task file
and change nightly; a copy of them here would be wrong within a week, which is
exactly how [[autonomy]]'s own hand-maintained inventory of 19 tasks came to
describe four retired jobs. `GET /api/autonomy/tasks` is the live view, the
Autonomy tab renders it with each row's `blocked` reason, and
`python scripts/autonomy/validate_tasks.py` is the structural check. What is
written down here is the half that does not change on a nightly clock: what the
job is for, what it reads and writes, what it must not do, and what has already
gone wrong with it.

Jobs are numbered by their task id and that numbering is historical, not
ordered: ids run 24 through 103 or beyond, with gaps where tasks were retired
(2026-10-19). #103, the code rules census weekly, is the newest and was filed
2026-10-19; #102, the find-all arms run, was filed the day before it. #101, the
gate-to-land lag daily record, was the newest when this note last moved, filed on
2026-10-16 six hours after #100. Which ids are live is
`GET /api/autonomy/tasks`'s business, not this document's. Typing a ceiling here
has been wrong twice inside a week: the bound stopped at 85 while 86 was already
running nightly, #1102 corrected it to 86, and #87–#90 overtook that correction
inside three days; #91–#97 overtook the correction this paragraph carries between
2026-09-29 and 2026-10-11, and #98 and #99 overtook the one before it inside two
days — the second time the directory ran ahead of the sentence, and the second
time a node was the thing that said so. #100 and #101 overtook the correction
that pass typed within hours of it landing, which is the third time the directory
has run ahead of this sentence and the third time the membership node rather than
a reader was what said so. So the
range above is a floor, and the id it names carries the date it was the newest.

**A filed task and an armed task are two different things, and the two kinds of
statement below follow different ones.** The membership tables take the task
*directory* as their source of truth, because the directory is what dispatches: #98
is filed, so it already has its row in both tables and the counts beside those rows
include it. The dependency arithmetic follows the fleet that actually runs instead —
#98 declares `depends_on: 24` while it sits `status: draft`, so the wiring diagram
still draws the ten arrows #1520 verified and §The wiring keeps its own dated count.
Arming the job is what adds the eleventh arrow and moves that sentence one, and
sending a filed task to `up_next` is a dispatch-affecting edit an unattended round
may not make, which is why that step is a person's. Which state an id is in is
`GET /api/autonomy/tasks`'s business, as above.

**Membership below is complete as of 2026-10-17, in both directions, and since
2026-10-13 that completeness is checked against the directory instead of
promised.** The seven task files #91 nightly
thinking-fidelity prevalence series, #92 vLLM prefix-miss daily, #93
step-conformance replay, #94 referential-integrity ledger, #95 mitigation drill,
#96 djev name-prior probe and #97 landed-outcomes daily record have their
entries, in [Watch the machinery](#watch-the-machinery-91-92-93-94-95-96-97-99-100-101-102-103).
#99 supply-chain inventory regen has one in that same group — a watcher whose
subject is the evidence the self-modification loop is judged on, and one that can
file a backlog item, which is why it sits beside #92–#94 rather than with the six
that only report. #100 eval instrument index publish and #101 gate-to-land lag
daily record join that group on the reporting side: each appends one dated row to
one JSONL outside this repo and neither files anything, and both are
`status: draft`, which is #98's precedent that a filed task earns its row before
an armed one.
#98 retag untagged facts has its row, and the reason a repaired-the-record job
belongs beside #48 rather than beside the watchers, under
[Canonicalize](#canonicalize-48-67-84-98).
The three paper digests #87 #88 #89 are gone from the tables below: each read
one named arXiv paper into exactly one knowledge note, was dispatched
interactively on 2026-09-24, and leaves no `87-*.md`, `88-*.md` or `89-*.md`
behind in `~/obsidian/autonomy/` or in its `_archived/` — a run record outliving
its task file does not make the job a fleet member. What *had* to change is the
reason both errors were possible: the two tables below are pinned to *each
other*, never to the directory, so a task file appearing in
`~/obsidian/autonomy/` failed nothing and a task file vanishing from it failed
nothing either. Backlog **#2631** closed that gap: two nodes in
`tests/test_autonomy_jobs_doc_claims.py` —
`test_every_id_the_membership_tables_name_has_a_task_file_or_a_retired_row` and
`test_every_live_task_file_is_named_once_in_each_membership_table` — now read
the task-file directory as the source of truth and fail on either direction, an
id the tables name with no `NN-*.md` behind it (unless that id's own row says it
is retired, which is how #36 stays listed with its file in `_archived/`) and a
live task file the tables omit. #2630 owns the content of this paragraph, which
is the same failure #1102 caught for #86, recurring for seven jobs, then for
#99 on the day #99 landed, and, in the
other direction, for three.

## The functions

| Function | Jobs | What it is for |
|---|---|---|
| [Ingest](#ingest-68-30-53) | #68 #30 #53 | turn outside signal into internal state |
| [Distil](#distil-38-42-39-40-47--56-57-58-83--54) | #38 #42 #39 #40 #47 · #56 #57 #58 #83 · #54 | read what we already have, write something more useful than the sum of it |
| [Build the graph](#build-the-graph-24-51-74) | #24 #51 #74 | turn documents and trajectories into facts and typed edges |
| [Canonicalize](#canonicalize-48-67-84-98) | #48 #67 #84 #98 | make what is already stored agree with itself |
| [Queue the work](#queue-the-work-65-35-77) | #65 #35 #77 | decide what gets attention next |
| [Bound entropy](#bound-entropy-79-81-78-80-90) | #79 #81 #78 #80 · #90 | stop the stores rotting or growing without limit |
| [Measure](#measure-60-82-70-36-76-86) | #60 #82 #70 #36 #76 #86 | say whether any of it is working |
| [Watch the machinery](#watch-the-machinery-91-92-93-94-95-96-97-99-100-101-102-103) | #91 #92 #93 #94 #95 #96 #97 #99 #100 #101 #102 #103 | read the machinery the rest of the fleet runs on — session traces, the engine's log window, run records, loaded memory, the mitigations, the decision engine, the loop's own landings, the instrument index, the retrieval arms run, the shipped code rules — and report what it looks like |

### Write authority is the fault line

Function is what a job means to do; **write authority is what it is permitted to
do when nobody is watching**, and it is the axis worth checking first before
changing any of these. Five tiers, covering every job the table above names. They
do not follow the function groups, and the top row is the least uniform of them:
writing a nightly reflection report, consolidating the skills library and
rewriting the entity store are all "durable state", and they sit side by side —
the row answers the permission question, never the size question.

**The unit of account, because "durable state" is what the rows are built out of
and the phrase defined nothing until #3335 put it here.** The permission above is
descriptive, not permissive: the rows record what a run leaves behind when nobody
is watching, not what some lane allows it to touch, and no row here is a grant or
the lack of one. `FILE_NOTE` = `{Write, vault_write}`
(`app/harness/capabilities.py:91`) is in every scheduled task's base set beside
`Bash` and `Edit` (`app/harness/capabilities.py:346`), and
`PROTECTED_WRITE_ROOTS` (`app/harness/protected_paths.py:151`) holds one vault
path, `lloyd/SOUL.md`, while `PROTECTED_SHELL_RO_ROOTS`
(`app/harness/protected_paths.py:222`) adds `lloyd/USER.md` and
`lloyd/MEMORY.md` on the Bash lane — so every other vault path is
reachable by every job in this table, and a tier is something to read off a run,
not something a tool enforces. A write **is durable state** when it lands in a
store the fleet or a human reads back as the system's record: prose under
`knowledge/`, `memory/` or `backlog/`, a `SKILL.md` in the skills library, the
derived fact tree, or an appended row of a dated series under `~/lloyd-data/`. It
is **not** durable state when it is a job's own scratch — a file under
`~/obsidian/skills/<skill>/scripts/` that the run rewrites for its own
bookkeeping, prunes on a timer, and reads back itself, consumed by nothing
downstream. **#68 is that case**, and it is the reason the definition is written
down rather than implied: Step 5 rewrites
`~/obsidian/skills/morning-brief-and-triage/scripts/state.json` — the ids already
surfaced, pruned to 24 h each cycle — on every cycle that surfaces or suppresses
an item, under the vault tree, and that write still earns it no row in the top
tier. The unit is the reader, not the path: #47 rewriting a `SKILL.md` on that
same tree *is* the top tier, because every later run reads what it wrote. The
five rows, each with the count beside it:

| Tier | Jobs | |
|---|---|---|
| Writes durable state unattended | #30 #53 · #38 #42 #39 #40 #47 #56 #57 #58 #83 #54 · #24 #51 #74 · #65 #35 #77 · #79 #81 · #48 #98 · #92 #93 #94 #99 #103 | 27 |
| Injects expiring context unattended | #68 | 1 |
| Proposes; an operator applies | #67 #84 | 2 |
| Reports only | #60 #82 #70 #78 #80 #36 #86 #90 · #91 #95 #96 #97 · #100 #101 #102 | 15 |
| Acts on the fleet itself | #76 | 1 |

The rows answer "may it write while nobody is watching", not "how much may it
write". **#48 is in the top tier because it does write** — aliases, edge
rewrites and moved fact files, every night — and what holds that to name-shape
variants is the guard rail layer described under
[Canonicalize](#canonicalize-48-67-84-98), not the tier. Its group mates split
the same way the heading's own table says: #67 and #84 stay in the propose row
because they stop at a file of proposals, and #98 joins #48 at the top because
it rewrites fact files on the same tree unattended the moment anyone arms it.

Two consequences the function grouping makes visible and the chain grouping did
not:

- **The most heavily verified group is the only one whose verification is what
  stands between a scheduled run and a write, and the group with the most write
  authority has the least of it.** Canonicalize carries the semantic gates, the
  unanimous-verdict rule and the guard rails that downgrade a questionable merge
  to alias-only — and #48's scheduled run *applies*: nightly, unattended, the
  mechanical CASE/PUNCT tiers, under the #990 ruling. So the rails are the
  boundary there rather than a plan-mode default, and the 2026-08-22 wipe
  (12,131 edges) and the 2026-09-03 151-merge incident — both unattended applies
  — are why the rails are strict rather than advisory. #67 and #84 propose into
  a file and stop. Distil writes
  `lloyd/USER.md`, `config.yaml` and the skills library with none of that
  apparatus. `workers/evidence.py` is the correction and the scheduler module
  `app/autonomy.py` is where its pilot set lives (`EVIDENCE_PILOT_TASK_IDS =
  frozenset({24, 38, 39, 40, 42, 53, 56})` at `app/autonomy.py:2898` in that module, pinned by
  `tests/test_worker_evidence.py`) — 7 of the 23 jobs that write durable state
  unattended. #53 Documentation Digester joined on 2026-10-09 (#2482), the first
  widening since #902 scoped the pilot to the reflection chain: that job's run of
  2026-10-09 filed a `priority: high` backlog item (#2478) over an audit database, a
  table and a row count that have never existed on this box, and emitted no claims
  block at all — so there was nothing for the verifier to refute, which is exactly the
  hole the widening closes. #24 Data Pipeline and #56 Nightly Trajectory Extraction
  joined on 2026-10-14 (#2675), the two writers #902 had named as the same shape one
  step out: #24 is 6x-daily and its own activity line is a per-run count
  (`files_processed=133 facts=4969 failed=1048` on 2026-10-13) that had never been
  bound to disk, and #56 writes one dated JSONL per day of session tool-call
  trajectories and states how many sessions it wrote every run. #81 QMD Index
  Maintenance is held out by that same ruling rather than overlooked: what it would
  claim is the qmd embedding index's own state, which none of the four check kinds
  reads, so scoring it would be a false negative and not a gap.
  The pilot has verdicts now: since
  #945 copied each run's `claims` key through `workers/sources/scheduled_task.py`
  into the pool, `workers/pool.py` grades the claims as the run record is
  written. On 2026-09-26 `GET /api/autonomy/health?days=7` reported 22 claims
  checked, 22 verified, 0 refuted and 0 insufficient, from 4 runs that carried a
  bundle: #38 2 runs 11 claims, #39 1 run 6 claims, #40 1 run 5 claims. All of
  them verified and none refuted is the state to write down, and it is the weaker
  claim: a pilot that has never rejected a claim has not yet shown that it can.
- **Nothing watches the deciders.** #65, #35 and #77 all direct future effort —
  what gets researched, what reaches `up_next`, and how the board reads when it is
  counted (#77 touches no file: see §Queue the work) — and no job
  in Measure covers decision quality. #76 watches task *health*, which is a
  different question.

### The wiring, which this layout no longer carries

Grouping by chain had one advantage this grouping gives up: the layout was the
dependency graph. Ten `depends_on` edges exist and **three of them now cross a
function boundary**, so they are written down here instead:

```
#38 ─► #42 ─► #39 ─► #40 ─► #47     distil, wholly inside
#56 ─► #57 ─► #58 ─► #83            distil, wholly inside
#56 ─► #51                          distil ──► build
#24 ─► #48 ─► #74                   build ──► canonicalize ──► build
```

The last line is the interesting one, and it is a genuine finding rather than an
artifact of the regrouping: **the graph pipeline alternates writing with
canonicalizing**, which is exactly why #74 declares `depends_on: 48` — it must
re-type edges only after the names under them have settled. Under the chain
grouping those three jobs sat in one list and the alternation was invisible.

29 of the 39 tasks (2026-10-12) declare no dependency at all: ten jobs name a
parent, one each, and those ten are exactly the arrows drawn above. None of
#91–#97 declares a parent, so the arrows are the same ten #1520 verified and the
pair moved by seven together. Neither #100, #101, #102 nor #103 declares a parent
either, so naming #102 (filed 2026-10-18) and #103 (filed 2026-10-19) in the
membership tables moves no arrow and no number in this paragraph — the 29 and the
39 are the fleet the route parsed on the day
the sentence was written, and a job entering these tables is not a job entering
that count. #103 is the newest file and #102 sits beside it; #101 was the newest
when that sentence was written, and #99 before it, and #98 before that — and #98 is still the only one
that declares a parent without being drawn: `depends_on: 24`, filed `status: draft`,
so it is not in the fleet the diagram counts — arming it adds an eleventh arrow
and takes one off the 29, which is the armed-state move the fleet-bound
paragraph above already carries.

### Two holes in the wiring

Both were measured against the reflection chain, both are properties of the
mechanism and apply to every edge above:

- **An upstream that resolves to nothing used to satisfy `depends_on`
  vacuously — closed 2026-09-19 by #558.** `_is_dependency_met` returned True
  when the id found no task, and until item #870 landed (2026-09-11) that
  swallowed far more than a typo: dispatch resolved against
  `_all_runnable_tasks()`, which kept only `up_next`, `in_progress` and
  `failed`, so a `paused` or `draft` upstream was *invisible* and unblocked
  everything below it — while the Mission Control board, resolving against every
  parsed task, printed `waiting on #N` for that same task in the same second.
  `dependency_resolution_set()` is the one set both surfaces read now, and an
  upstream that exists on disk blocks whatever its status. The narrow case #870
  deliberately left open — an id with no file behind it, which has no
  `last_run` to measure — is now held too: it is read as NOT met and logged, on
  Alan's ruling of 2026-09-13, with `stale_bypass_hours` as the escape. The one
  status that still passes is `failed`, whose last SUCCESS is a real artifact.
  So a chain whose upstream id is a typo stops rather than silently consuming
  nothing, which is what inverted the nightly chain on 2026-09-08.
- **An empty-but-successful run is indistinguishable from a dead one.** A run's
  status is decided by its terminal text, and the chains gate on status. Measured
  2026-09-11: `run_38_20260911_050055` wrote a complete signal report, was
  recorded `empty: true, status: failed` after 248 s with `stop_reason: stop` and
  19 turns, `last_run` did not advance, and only the 05:15 retry unblocked #42.
  Open as backlog **#832**.

---

## Ingest: #68, #30, #53

What the outside world did while nobody was looking. #30 and #53 write into
`knowledge/`; #68 writes no knowledge note — what it produces for a human is a
*conversation* (an inject that expires) rather than a file, which is why it has
a write tier of its own. That tier answers the durable-state question defined
under [Write authority is the fault line](#write-authority-is-the-fault-line),
not the "does it touch a file" one: Step 5 rewrites `scripts/state.json` inside
its own skill directory, which is under the vault tree (#3334) — the write is
real, and `scripts/state.json` is #68's own dedup scratch, pruned to 24 h each
cycle and read back by nothing but its next run, so it is not durable state and
earns no second row. It was also the only task file that
named the **secondary** engine, the slot retired 2026-09-20: that line read
`model: secondary` until #2632 set it to `primary` on 2026-10-13 — the value the
job already ran at, because `resolve_model_alias` rewrites it (§#68). The park is
Alan's ruling of 2026-09-17, not that line.

| ID | Freq | Role |
|----|------|---|
| #68 | every 15 min | Calendar + unread mail, filtered and classified, injected as ambient context |
| #30 | daily | GitHub releases/issues and YouTube RSS → keyword pre-filter → scorer → high-scoring items to the vault |
| #53 | daily | Changelogs and release notes for the stack; compare each source's latest version against last-checked; only process actual changes, then write `knowledge/stack-updates/YYYY-MM-DD-<source>.md` |

**The three paper digests held rows here until 2026-10-12.** #87 #88 #89 were
`research-agent` jobs that each read one named arXiv paper — 2609.26758,
2609.26532, 2609.26550 — into exactly one `knowledge/` note, and were dispatched
interactively on 2026-09-24. No `87-*.md`, `88-*.md` or `89-*.md` is left in
`~/obsidian/autonomy/` or in its `_archived/`, so the rows here described jobs
with nothing to dispatch them; #2630 took the trio out of the
function row, the write-authority row and this heading. The run records they
left under `~/lloyd-data/autonomy-runs/` are history, and history is not
membership.

### #68 — the name outlived the job

`morning-briefing` names two things. One is #68, whose fifteen-minute cadence is
made to triage mail and calendar into whatever chat session Alan is actually
sitting in. **The job is parked**: `status: draft` since Alan's ruling of
2026-09-17, last run 2026-09-16. Its `model:` line named the slot `551e9044`
disabled on 2026-09-20 until #2632 set it to `primary` on 2026-10-13 — the value
`resolve_model_alias` was already routing every run to, which is the confusion
backlog **#1555** recorded: a task claiming a retired slot's engine and receiving
the *primary*, invisible to both slot guards, because `slot_arm_block` and the
#1577 lint read `requires_slot:` and this task declares none — so arming it
succeeds, which `tests/test_autonomy_slot_arm_guard.py` pins for id 68, and
nothing ever asked the `model:` value about the flag. That blind spot is what
#2632 closed: `scripts/autonomy/validate_tasks.py` now reports a `model:` naming
a switched-off slot in the same one-line shape it uses for `requires_slot:`, and
under `--strict` that warning fails the run. Measured 2026-10-17, no task file
names the retired slot: `grep -h '^model:' ~/obsidian/autonomy/*.md` answers
`primary` at least 43 times or beyond — a typed **minimum**, measured the same day
and not a ceiling: exactly one `model:` line per task file that declares an engine,
all of them `primary`, and the fleet only grows (#101 `gate-to-land-lag-daily-record`
joined 2026-10-16), so a floor cannot lag it the way the exact count did three days
running. It stays pinned: `tests/test_validate_tasks_linter.py` parses the minimum out
and re-measures the fleet, going red only if the fleet drops below the floor. All of
this describes the mechanism as
designed and as it behaved while live. The other — the 6 am "read the
overnight reports and announce a synthesis" job — no longer runs at all and no
autonomy task binds it. Its skill is archived; an unbound copy survives as
`~/obsidian/skills/nightly-morning-briefing/` still carrying "Runs at 6:00 AM PST
daily" and "Use announce mode", and **there is no announce delivery mode for
autonomy tasks anywhere in the code** — the `announce()`s that do exist (the
guardian's alert fan-out, `scripts/automod/promote.py:676`, `announce_breach` in
`scripts/iv_metrics_record.py`) are notification helpers for other purposes. Its
documented output, `memory/reflection/morning-briefing-latest.md`, exists nowhere
on disk. Today's run writes no report file at all: its output is the injected
signal plus its own `scripts/state.json`, and the durable record is the run record
and the session transcript.

**How it decides what to surface.** `references/prefs.yaml` holds `watch_senders`,
`watch_domains`, `mailing_lists`, `block_senders`, `block_domains`,
`priority_keywords`, `ignore_subjects`, `watch_calendars`; `scripts/state.json`
holds already-surfaced ids, pruned to 24 h each cycle. Calendar comes from
`calendar_events` and mail from `email_recent`, both Thunderbird bridge tools
advertised only while Thunderbird and its bridge are up. Filters run blocklist
first, then `ignore_subjects`, then the allowlist and `priority_keywords`, then
per-item dedup, then a soft reject of anything newsletter-shaped; at most 3
calendar items and 5 emails survive. The classing has no numeric deadline rule in
it, whatever a paraphrase once said: the only calendar `act-now` trigger is an
event starting within ~3 hours, and a cancellation outranks even that (3.5g); an
email reaches `act-now` only if this same run called `email_read` on it (3.5a), it
carries no broadcast transport evidence (3.5b), it is addressed to Alan beyond the
`To:` field (3.5c), **and** its 🔔 line names the body phrase proving a date,
request, question or obligation. An allowlist or `priority_keywords` hit raises
salience *inside* `fyi` and never crosses the line (3.5d). **A single `act-now`
changes the routing of the whole signal**:

- **`ambient`** (nothing is act-now) → `POST /api/sessions/{id}/inject-prefetch`.
  No turn, no tokens, no transcript noise: the entry waits in a queue and is
  drained into the `<context>` block of the user's *next* turn.
- **`notable`** (anything is act-now) → `POST /api/sessions/{id}/inject`, a real
  ambient turn. The skill is told never to escalate to `urgent` — that is
  reserved for safety and security.

**Which session counts as "the user's" is the whole problem.** The producer passes
no session id, so resolution asks for the last session to receive a user turn,
else the most recent user session by mtime inside 24 hours. On **2026-09-07** that
answered with a machine: the brief — a meeting that night — was injected into a
backlog-triage **worker** session that had finished 77 seconds earlier, and was
answered there, to nobody. Worker turns go through the chat path so they can have
Inner Voice, which means they set `_last_user_session_id` exactly as a human
typing does. The fix is one definition of "a human reads this":
`NON_USER_PLATFORMS = frozenset({"autonomy", "worker"})`, applied through
`is_user_session` to **both** resolution rules. It is a **deny-list on purpose** —
a platform this code has never heard of stays eligible, because silently losing a
brief is worse than delivering one to an unexpected session. Two details that are
easy to get backwards: `/inject` answers **409, not a 200 "skipped"**, so no
producer records a lost brief as delivered; and since `af5f9082` (2026-09-20)
`/inject-prefetch` answers **the same 409** when `is_user_session` says no
(`app/routers/sessions.py:1415`), so the ambient path is gated at
*resolution* **and** at the endpoint. The endpoint half is what catches a producer
that passes an explicit worker id instead of leaving the server to resolve it —
before that commit such a producer got a 200 and an entry queued in a session that
takes no user turn, which nothing reclaimed until expiry ran inside the drain.

**`[SILENT]` is what keeps a 15-minute job from being a 15-minute notification.**
Every task is told to answer with exactly `[SILENT]` and nothing else when there
is nothing new. It used to be read two ways — the run record and the Discord
helper by **exact** match, the worker adapter and the `/api/autonomy/health`
rollup by **substring** — so a report that merely *mentioned* the token was
recorded as a real run, lost its Discord post, and was counted as a declined run
in `silent_rate`. #1507 (`9147db78`, 2026-09-25) collapsed the four readers onto
one predicate, `is_silent_response` in `app/silent_sentinel.py` — exact match on
the stripped text, which `app/discord_notify.py:66` now calls — with
`run_is_silent` deferring to the run record's own `meta.silent` so no downstream
text slice can re-decide a verdict made on the terminal block. Exact is canonical
because it is what the prompt states, and the skill's own last line is written
against equality for the reason that started the whole thing: a paragraph ending
in `[SILENT]` was delivered to Alan in full, as noise (2026-09-03).

**Backlog #481 is closed, and it closed the contradiction rather than only the
heading.** Step 2 used to tell the agent to skip a missing calendar source
**silently** ("do not retry, do not treat it as an error"), which contradicted the
degraded-run attribution guardrail at the foot of the same file — a run whose
source died must name it. #1504 renamed that heading to "may be unavailable —
record the absence, never skip silently" and gave Step 2 a `CALENDAR_SOURCE`
branch, and Step 3 gained the matching `EMAIL_SOURCE` branch it never had: a call
returning 0 messages is stamped `email: 0 returned — suspect (baseline ~15-21)`
instead of being read as a quiet morning. Step 5 writes both to
`state.json → sources` and Step 4 prints any non-`ok` one, so the 2026-08-29
"succeeding but blind" run (0 emails against a normal 15–21, after a ~24 h
IMAP/Thunderbird bridge gap) is now the shape the skill refuses by instruction
rather than the shape it reproduced by it — and a zero-item run that finds a source
*newly* degraded answers with those source lines instead of `[SILENT]`.

---

## Distil: #38 #42 #39 #40 #47 · #56 #57 #58 #83 · #54

Ten of the fleet's 39 jobs (2026-10-12), and the largest concentration of
unattended write authority in it. Two chains and one straggler: reflection turns
the day's *signals* into memory and config, trace2skill turns the *tool-call
record* into skills, and #54 turns the vault's existing notes into new ones.

### The reflection chain: #38 → #42 → #39 → #40 (+#47)

The chain turns the day's signals into durable changes: what Alan corrected,
what the system learned, and what should therefore change in memory and config.
All four run on the local `primary`. There is no dollar budget; the cost is
GPU-hours.

```
#38 signals ──► #42 analysis ──► #39 knowledge write ──► #40 config
                                                              │
                                                              └─► #47 dream (weekly)
```

| ID | Skill | Reads | Writes |
|----|-------|-------|--------|
| #38 | `nightly-reflection-signals` | `memory/corrections.md`, last 3 daily notes, that range's session data | `_pipeline/reflection/signals-latest.md` |
| #42 | `nightly-reflection-knowledge-analysis` | signals-latest, last 7 daily notes, `memory/mental-models.md`, `lloyd/MEMORY.md`, `lloyd/USER.md` | `_pipeline/reflection/knowledge-handoff-<date>.md`, validated by `scripts/validate_handoff.py` |
| #39 | `nightly-reflection-knowledge-write` | the handoff, newest by mtime, refused over 24 h old | `lloyd/USER.md`, `lloyd/MEMORY.md`, `memory/mental-models.md`, `people/alan/profile.md`, tool/conversation pattern summaries, vault commit |
| #40 | `nightly-reflection-config` | signals-latest, `lloyd/MEMORY.md` after #39 | `~/lloyd/config.yaml`, vault surfaces (SOUL/skills/autonomy frontmatter), `memory/learnings/<date>.md`, commits on `main` |
| #47 | `dream-consolidation` | the memory files #38–#40 wrote — never raw sessions | merged topics, resolved contradictions, a tighter MEMORY.md index |

**Analysis is split from write on purpose.** #42 derives every update and writes
a structured handoff; #39 opens a fresh context window holding only that handoff
and executes the writes. A single job doing both spent its whole budget reading
and died before writing.

**#39's first target is `lloyd/USER.md`** — the user-facts file the runtime loads
into every system prompt, and the write with the largest blast radius in the
chain. Documentation that named only `mental-models.md` and the profile hid it.
On 2026-09-11 that job re-applied 44 curated lines to `lloyd/MEMORY.md` that an
unrelated job's pre-flight commit had clobbered the night before.

**Every job claims its output file in its first turns**, enriches it in place,
and flips a `status:` field to `complete` as its last act. This is not stylistic.
These jobs used to investigate exhaustively and write at the very end, so a run
that hit `max_turns` produced *nothing* while still reporting success: on
2026-09-03 a 1429-second #38 run left `signals-latest.md` untouched from two days
earlier, and #42, #39 and #40 all consumed that stale file believing it current.
#39 separately did its real writes and then died during bookkeeping, so
`mental-models.md` was genuinely updated while the record said nothing had
happened.

The recurring failure is looping on "one last verification pass" until the turn
limit kills the run. #42 is the reference implementation: a machine-checkable
output contract plus a validator it runs against itself, and — since 2026-09-18 —
a **skeleton written before the first deep read**, amended in place, so a run
killed anywhere after it leaves a consumable handoff rather than nothing. The
"write the handoff by a fixed turn number" rule that stood there from 2026-09-08
was **deleted** on that date because a run cannot see its own turn count, so it
was uncheckable prose and the only real backstop was the cap plus an auto-retry:
#42 was killed anyway on 2026-09-14 (`run_42_20260914_050447`) with
`stop_reason: max_turns`, `num_turns: 61`, the artifact landing 39 ms before
termination and the cycle still recorded FAILED — the same 61 turns and 2.5 M
tokens spent on `tool_errors: 0` voluntary work that motivated the rule. What
replaced it is three facts the run can read, not a number it cannot. #38 carries
the same budget in **minutes**, because its turn limit was never what killed it:
Phase 2 dies at 1809–1826 s against a 1800 s cap while healthy runs take
181–639 s. Raising the ceiling is the wrong lever — the 600→1800 s raise on
2026-09-01 enlarged the burn rather than making overruns rarer.

**The pairwise offset has since been undone, so the clock orders nothing.** All
four once sat on 22–04 against `workers.slots: 2`, so #39 could be claimed in the
hours #42 was still writing the handoff it was about to read; the 2026-09-09 fix
narrowed #38 and #42 to 22–01. Read off the task files again on 2026-10-26, both
of them declare 22, 23 and 0–4 again, #39 declares 1–4 and #40 2–4 — every pair
that consumes another's output now shares that output's whole window — and
`workers.slots` is 6 (since 2026-09-18), not the 2 the windows were sized
against. What orders the chain is the freshness half of `depends_on`, not the
clock — which is the correct division of labour, and the reason the two holes
above matter more than any window edit. The jobs are minutes (measured 2026-10-26
over the 7-day window `/api/workers/health` holds unclamped: #38 avg 508 s with
its slowest run at 683 s against its 1800 s cap, #42 730 s, #39 774 s, #40 513 s,
against `timeout_seconds` 1800 / 1800 / 2400 / 1800), so whatever span the chain
shows end to end is scheduling latency, not work.

Two tool-shaped rules, both learned expensively. `Write` refuses to overwrite a
file it has not read this session, so a Phase 0 skeleton write is *refused on
every run* — `signals-latest.md` always exists from the night before. And the
vault tools reject anything under `~/lloyd/` with `PATH_ESCAPE`, so every
`_pipeline/` artifact in this chain is written with `Write` at an absolute path,
never `vault_write`.

**A `-latest` overwrite keeps a dated copy, and the directory is checked for it.**
`scripts/reflection_archive.py` refuses a governed skill that drops its archive
`cp` (skill text, at `automod_vault_land`), and since #1227 `copy_gaps` reads the
directory itself: a report that names the copy it made (`Archive copied before
this write: …`, or any `<stem>-latest-<stamp>.md`) whose file is absent is a lost
cycle, listed under `## Reflection Retention` in the daily Knowledge Health
Report. The expectation comes from that pointer, never from the calendar — a
report naming no copy is not a gap.

**The claims a run makes are re-checked against disk.** The chain's most-repeated
defect was never a crash, it was a confident number: a handoff reporting the
graph "restored to 12,131 relationships" against a same-night health report
reading zero; counts of 96/21 where disk held 121/64; a 13,503-byte file called
"307KB". `workers/evidence.py` is the structural fix and its pilot set is exactly
this chain plus #53, which joined 2026-10-09 (#2482), plus #24 Data Pipeline and
#56 Nightly Trajectory Extraction, which joined 2026-10-14 (#2675) as the two
writers #902 named and left for a later ruling — #24 is 6x-daily and its activity
line is a per-run count that had never been bound to disk, #56 reports how many
sessions it wrote into one dated JSONL per day. #81 QMD Index Maintenance is held
out by that same ruling: what it would claim is the qmd embedding index's own
state, which none of the four check kinds can read —
`EVIDENCE_PILOT_TASK_IDS = frozenset({24, 38, 39, 40, 42, 53, 56})`
(`app/autonomy.py:2898`, not inside the verifier's own file, and pinned there by
`tests/test_worker_evidence.py:686`), a literal frozenset so
widening it is a change someone reads. The verifier is **stdlib-only and never
LLM-judged**, because a model grading its own claims is the narration this
replaces one layer up; and **a claim that cannot be evaluated is `insufficient`,
never `verified`**. Whatever failed to verify is carried into the next run of
that task through the queue's watermarks, not written back into the
human-edited task file.

**The bundle was severed one hop short of the verifier; #945 closed that.**
`run_task` attaches the parsed claims block to its result for a piloted task, but
`workers/sources/scheduled_task.py` rebuilt the dict it hands the pool as a
hand-written whitelist with no `claims` key on it, so `workers/pool.py` read an
out-of-scope source every time: 0 bundles across the 22 piloted runs in the window
to 2026-09-12, and `claims_checked: 0` over 713 fleet runs — which is also why
#714's prediction of a first window full of `insufficient` gaps could not come
true. The key is copied now when the model emitted one, and the pilot reports:
measured 2026-10-26 over the 7-day window the route holds unclamped, all seven
piloted tasks together, **892 claims checked, 850 verified, 17 refuted, 25
insufficient, over 77 runs that carried a bundle** out of the source's 253. That
supersedes the 2026-09-26 reading (22 checked, 22 verified, 0 refuted, 0
insufficient, across 4 bundled runs — #38 11, #39 6, #40 5): both `insufficient`
and `refuted` are now observed in volume, and the chain's own analyst is its
largest refuter — read off the runs table for the same window, #42 emitted 69
claims of which 6 were refuted and 1 insufficient, against #39's 4 refuted and
#38's 2. What is left is coverage, not plumbing — 176 of those 253 runs carry no
bundle at all, because only the piloted tasks emit a claims block. #902
(coverage: 4 of the fleet's unattended writers) and #945 (this severance) are
both closed. **Widened 2026-10-09 (#2482):** #53
Documentation Digester joined, so five tasks emitted a claims block on the day of that
measurement, and its first live refutation is what the widening was for — the digester
had filed a high-priority item over an audit store with no row in it anywhere on this
box. **Widened again 2026-10-14 (#2675):** #24 Data Pipeline and #56 Nightly Trajectory
Extraction joined, so seven tasks emit one now.

**#47 Dream Consolidation** is the weekly synthesizer on the tail. It does not
re-read raw sessions — that is the chain's job — it merges near-duplicate topics
across the files reflection wrote, resolves contradictions, prunes stale entries
and keeps the MEMORY.md index tight. Gate-checked on ≥24 h since its last run and
≥3 new sessions, and lock-file protected.

#### Jobs 3 and 4 do not exist

Earlier documentation, and #40's own data-load, referenced a prompt audit and a
behavior test writing `prompt-audit-issues.md`, `prompt-audit-latest.md`,
`test-failures.md` and `test-results-latest.md`. **No autonomy task has ever
written any of them.** `skills/nightly-prompt-audit/` and
`skills/nightly-behavior-test/` sit on disk at `status: archived` (2026-09-19,
#900), so neither is retrievable, and no task file's `skill_name` names either.
That skill has since been cleaned up: #40's data load now lists all four files,
both skills and `propagation-log.md` under "Three things earlier versions of this
skill told you to read do not exist", and tells the run not to read, wait for, or
report a gap about them. The `### Prompt Audit & Behavior Test Issues` heading and
its Phase 3 changelog asks are gone, so a run can no longer produce an empty
section for jobs that were never built.

The numbering inside the skills never converged and is not worth trusting: the
four self-label "Job 1 of 3", "Job 2a of 4", "Job 2b of 4" and "Job 5 of 5".
There are four jobs.

### trace2skill: #56 → #57 → #58 → #83

Every interaction leaves a record, and that record contains procedural knowledge
— troubleshooting steps, corrections, behavioral rules, gotchas — that would
otherwise be lost between sessions. This chain surfaces it, packages it as
skills, and maintains the library.

| ID | Skill | What it does |
|----|-------|---|
| #56 | `trajectory-extraction` | tool-call trajectories from every session (worker + main) into one JSONL per day at `_pipeline/trajectories/<date>.jsonl`; watermark-gated and incremental |
| #57 | `trajectory-skill-mining` | 7-day window over those trajectories for error and success patterns, grouped into skill candidates with metadata. Installs **new** skill dirs only; defers existing ones to #83 |
| #58 | `nightly-skill-consolidation` | groups dated candidate snapshots, drops any pattern with a terminal verdict in the ledger, filters to `sessions >= 3`, proposes patches to existing skills and flags new candidates. It **applies nothing itself** since 2026-09-19 (#719): a proposal lands in `proposed/` with `applied: false` and #83 Stage 6 is the only task allowed to land it, so the confidence ≥ 0.85 AND sessions ≥ 5 floor is #83's decision rule, quoted by words in both files |
| #83 | `nightly-skills-management` | the lifecycle pass: evaluate, create/update, review drafts, dedup the library, regenerate `memory/skills-index.md` |

#56 is also the input to #51 — the one edge that leaves this function group. The
trajectories it writes are the same artifact the graph chain mines for co-access
pairs.

**#83's pre-flight is a git snapshot, because there is no other undo.** The pass
opens by running `scripts/util/vault-commit.sh "skills-mgmt: pre-run snapshot
(unattributed dirty state)"`. Nightly jobs commit straight to the vault's `main` — no
worktree, no gate, no guardian, unlike [[automod]] — so the commit taken *before* the run
is the whole of the rollback story. That sweep is why the message says what it does: it
takes the whole tree, so it commits work this job did not write, and since #1070 the
wrapper says so in the commit body and on stderr (`unattributed dirty state:` plus the
path list) instead of letting a job's name stand in for authorship. The five nightly
final commits go the other way and name their paths (`-- memory/ knowledge/ …`), which
is what makes *their* subjects mean what they say; see
[[guardian-data-damage-false-trip]] for the sibling class of a check that reads the
wrong thing.

**What it mines is the human half of the day, by construction.** Post-session
capture appends the daily-note summary only for a user session and routes
background exports to `_pipeline/vault-derived/sessions-background/`, so the
background sessions a day this machine runs for itself are not in the daily notes
the pass falls back to — and the corpus is three times the size this paragraph
said: 845, 903 and 728 session files on 2026-10-23, -24 and -25 (measured
2026-10-26), the 10-25 population headed by `review` 212, `autotriage` 147,
`owedcheck` 121 and `autocode` 68. The trajectory JSONL is the wider corpus and
does carry them.

#### Evidence integrity: three signals weaker than they look

Added to the skill 2026-09-06 after measurement, and the most load-bearing thing
in it. A mining pass that trusts its inputs manufactures skills from incidents
that never happened, and a wrong skill is worse than no skill: it is retrieved,
obeyed, and wrong.

- **`has_errors` / `error_tools` is a text heuristic, not an error flag.** The
  extractor matches `Error:`, `Traceback`, `No such file or directory` against the
  *result text*, so reading source code that mentions an exception counts as a
  failure. On the 2026-09-05 trajectories: 3 of 3 sessions flagged and 30
  `error_tools` entries against 6 real failures in `stats.is_error` — roughly two
  thirds of them `Read`/`Grep` output quoting code.
- **`[INNER VOICE]` messages are machine output, not Alan's corrections.** The
  repetition guard asserts "the result has not changed" from argument similarity
  alone; its signature has no result field at all. On 2026-09-05 it fired 5 times
  while a result-level comparison found no unchanged-result repeat behind 4 of
  them. A false fire logged on the corrections axis becomes a false skill — which
  is why it is pinned from the other side in tracked code:
  `tests/integration/test_iv_repetition_wording.py` bans the unobservable claim
  in the inject wording and names #83 Stage 2 as the cost.
- **`stats.is_error` over-counts too.** On the 2026-09-07 trajectories (39
  sessions) the heuristic flagged 128 tool calls, the session JSONs marked 54
  messages, and reading the payloads left **13** genuine failures. Of the 54, 32
  were only a non-zero exit code and 9 carried no error text at all — each
  truncated at exactly 2014 characters, so whatever tripped the marker sits past
  the stored cap and cannot be checked. Those are unverifiable and get dropped,
  not counted. A useful failure is one the tool itself framed as an error, or a
  deliberate block or cancellation; never merely "the shell returned non-zero".

#### Authoring rules: why the generator was off for 68 days

The predecessor task #37 was paused 2026-06-27 for one recorded reason: the
generator "authored/grew dense bash-runbook skills that the primary model echoed
instead of executing". The primary treats fenced shell blocks in a retrieved
skill as few-shot "print this" demonstrations, so a command-dense skill
manufactures that failure on every retrieval. It was archived 2026-09-03 after 68
days in which its own `depends_on: 58` was also paused and the chain was dead,
and re-enabled as #83 only once the skill could state the rules that prevent it:

1. **No fenced executable blocks** in a generated skill body — prose and inline
   `code` spans only.
2. **Scoped "When to Use", never a bare tool name** — trigger on a specific error
   signature, since "when using the Bash tool" matches every routine request.
3. **Bounded length** — target under 80 lines; consolidate rather than stacking
   past ~120.
4. **Only real tool names**, verified before promotion by
   `tests/test_skill_tool_names.py`. A skill naming a tool that does not exist is
   actively harmful: the model calls it, gets an unknown-tool error, and takes
   whatever fallback the skill listed. That is how `web_search`/`WebSearch` —
   names Lloyd has never had — taught it to shell out to `curl` for every web
   lookup on 2026-09-04. The web tools are `http_search`, `http_fetch`,
   `http_request`.
5. **Never restate a tool's parameter contract.** Ranges, defaults and enum
   values belong to the advertised schema; a copy drifts silently. Eleven skills
   were archived 2026-09-04 for documenting an `extract_mode` retry chain, a
   `max_chars` floor and a default that were all wrong.
6. **No tool-failure skills from pre-2026-09-04 evidence.** Before that date no
   tool set `isError`, so every failure arrived as a success whose text contained
   `{"error": ...}`. Patterns mined from older sessions describe a signalling bug
   that has since been fixed.

**Two linters enforce rule 4, and each carries a seven-name exemption list of
which three entries are this chain.**
`tests/test_skill_tool_names.py:150` (`ALLOWED_TO_MENTION`) and
`scripts/skill_lint.py:281` (`PHANTOM_EXEMPT`) let `nightly-skills-management`,
`trajectory-skill-mining` and `nightly-skill-consolidation` write the phantom
names down, because their job is to say those names are not real. The same two
lists now also carry `web-search-and-fetch`, `create-hermes-plugin`,
`autonomy-task-diagnosis` and `pipeline-dispatch`, each exempt for a reason
unrelated to this chain — and they are hand-maintained in two different files,
with nothing pinning them equal. The test's
`KNOWN_UNFIXED` debt ledger is **empty** — the 91 skills that carried a phantom
name have been rewritten or archived — and a new entry there is a regression, not
a grandfathering.

#### Two traps in the review stage

- **Draft skills are selected on parsed frontmatter, never a body grep.** The
  `nightly-skills-management` skill's own body contains the literal string
  `status: draft` inside its instructions, so a body grep reports it as a draft on
  every single run: measured 2026-09-08, that grep returned exactly one file —
  itself — and there were zero drafts. Parsing also catches the quoted
  `status: "draft"` variant a string grep misses.
- **Flagging is not withdrawal.** A low-quality skill marked `status: needs-review`
  stays in circulation. `_QUARANTINE_STATUSES` is
  `{inactive, archived, disabled, retired, quarantined}` (`agent_mcp/skills.py:87`),
  and since #1294 `prompt_builder._is_quarantined_skill` is a plain alias of that
  module's `is_quarantined_skill_file` — one spelling of the rule, where
  `prompt_builder` used to re-scan the front matter against its own copy of the
  set and an equality assertion in one test was the only thing holding the two in
  step. So a needs-review skill
  keeps appearing in `<available_skills>` and keeps being retrievable until
  somebody acts on it.

**The other half of "wrong skill" is measured, not written.** When the body is
fine and the *description* is not, retrieval fires the skill on the wrong request
or fails to fire it on the right one; that is #70's `MISSING_DESC` and `DRIFT`,
advisory only, and it lives in [Measure](#measure-60-82-70-36-76-86). When the
procedure itself has rotted, that is #83's Stage 3, which appends a correction
rather than rewriting.

**#83 runs unobserved, and the thing that used to stop it landing anyway is
gone.** It sets no `inner_voice:` and the fleet default is `false`
(`autonomy.inner_voice` in `config.yaml`, the row `inner_voice: false`). On
2026-09-11 the pass tried to land
a vault change through `automod_vault_land` and was refused, because the tool
then gated on Inner Voice being attached and the observer attaches at turn start
— switching it on mid-turn cannot cover the turn. **That refusal can no longer
happen to a job in this group:** #901 put a platform exemption ahead of both the
refusal marker and the flag — `agent_mcp/automod.py:203` returns `None` for any
session whose `platform` is in `NON_USER_PLATFORMS` (`app/sessions_io.py:101`,
`{autonomy, worker}`) — and `automod.require_inner_voice` additionally defaults
false at `agent_mcp/automod.py:50`, absent from `config.yaml`. #2670 then made
`automod_vault_land` the single standing worker ban an `autonomy-task:<id>` scope
may be licensed around (`app/harness/capabilities.py:182`,
`SCOPE_BAN_EXCEPTIONS`), and #38, #47, #83 and #54 each hold such a row now,
minted under #2927 as "vault-land outputs only", expiring 2027-01-31. The
whole-tree pre-flight commit above is still the path all four actually take;
unobserved-and-licensed is nonetheless the new standing state, and the observer
that the 2026-09-11 refusal was enforcing is now exempt by platform.

### #54 — the straggler

| ID | Freq | Role |
|----|------|---|
| #54 | weekly | Take 2–3 knowledge areas and find non-obvious connections grounded in actual vault content — not brainstorming; read existing notes, identify gaps or bridges, write to `knowledge/synthesis/` |

#54 sat under inbound signal while this document was grouped by chain, and that
was the wrong shelf: nothing about it is inbound. Its inputs are notes already on
disk and its output is a new note derived from them, which is the same shape as
the two chains above — read what we already have, write something more useful
than the sum of it. It differs only in the corpus: reflection reads the day's
signals, trace2skill reads the tool-call record, and #54 reads the vault itself.

---

## Build the graph: #24, #51, #74

What each of these writes is the grouping, not whether it writes: #24 produces
entities and `mentions` edges; #51 adds co-access edges from the trajectory
record; #74 re-types what #24 left untyped, after
[Canonicalize](#canonicalize-48-67-84-98) has settled the names underneath. These
are the only jobs that **create** edges and entities. #48 writes the same store
unattended every night too — it rewrites edges onto the surviving name, writes
aliases and moves fact files — but it creates no entity and no relation, only a
decision about which of two names an existing pair always should have shared.

| ID | Freq | Depends | Role |
|----|------|---------|---|
| #24 | 6×/day | — | Content-hash-gated fact extraction over the vault corpus **allow-listed** in `scripts/memory/next-gen-memory/pipeline_config.yaml` — `knowledge`, `projects`, `people`, `personal`, `work`, and `memory` only once a daily note has settled — then index rebuild. The `kg_health` snapshot is **Step 0, launched ahead of every gate**, so a run that exits early still writes one; it is not a tail step. Emits `mentions` edges as it writes facts; no-ops when no source document changed |
| #51 | daily | #56 | Session co-access pairs from trajectories → edges **typed by the Stage-2 classifier**, `co_accessed` only as the fallback when nothing classified the pair (#420, 2026-09-12), provenance `INFERRED`, the trajectory session as `source_doc`. Measured 2026-10-26: 562 conversation-origin edges in the store, **none** of them `co_accessed` (321 `related_to`, 129 `implements`, 84 `depends_on`, 23 `derived_from`, 5 other) |
| #74 | daily | #48 | Re-types active `mentions` edges with the v4 classifier and applies confident upgrades through `app.kg_store`. Runs after #48 so canonical names are settled first |

**#24 is the single most expensive job in the fleet** — 10.4 GPU-h over the seven
days to 2026-10-26 (window unclamped), against 36 runs: 37% of the autonomy
fleet's 27.9 h, and more than the next four tasks put together. The task this
sentence used to pair it with, #68, has been parked by Alan since 2026-09-17 and
has logged nothing since, so the pairing is gone; the second-heaviest task is this
group's own #74 at 2.0 h (7%).

**The `Depends` column is a freshness gate, not a clock order.** Measured over the
seven days to 2026-10-26, #51 dispatched at ~06:00Z and its upstream #56 at
~08:00Z **every night** — the dependent runs two hours ahead of the job it waits
for and reads the *previous* night's trajectory file, released by the bound in its
own row (`stale_bypass_hours: 12`) rather than by the ordering. #74 (~04:30Z)
likewise precedes #48 (~17:50Z) on the same clock day, so the apply whose
canonical names it trusts is the previous day's. #24 is what sits in that gap:
running 6×/day, it writes `mentions` edges twice between #48's apply and #74's
next run, and those are typed before #48 has ever seen their names.

**#51 produced no graph at all until 2026-09-04**, when approval only flipped a
status field in a JSON file nothing downstream read. It also ran every 15 minutes
until it was moved to nightly, where 95 of 96 daily runs were no-ops — the input
is written once a night by #56.

**One store, one writer, and two read-only readers.** Edges, aliases, the entity
registry and the fact index live in `_pipeline/vault-derived/kg.sqlite` behind
`app.kg_store`, and nothing else **writes** it. Two openers besides the owner read
it, both for a row count: the guardian's data-damage tripwire
(`count_kg_rows`, `agent-services/guardian/guardian.py`) and the promoter's
baseline (`scripts/automod/promote.py`), which is the number the watchdog is
weighed against — the two still resolve the path by different rules, tree for one
and production root for the other, the unification #1525 recorded rather than did.
That watchdog runs system python on a staged snapshot, so it can import neither `app.kg_store` nor
`app.paths`, and the path it counts comes from the stdlib-only
`app/data_root.py` — the one spelling both readers share (#1525). See
[[knowledge-graph]].

---

## Canonicalize: #48, #67, #84, #98

Make what is already stored agree with itself — one name per entity, no
contradictory claims. **What this group may change is a name, never a fact**, and
within that boundary its write authority is not uniform: #67 and #84 propose into
a file and stop, while **#48 applies the mechanical tiers unattended every
night**, by Alan's ruling on #990 (2026-09-16, reaffirmed 2026-09-24). That
asymmetry is the group's defining property, not an accident of scheduling.

| ID | Freq | Depends | Role |
|----|------|---------|---|
| #48 | daily | #24 | Name-shape clustering; **applies** the `CASE`/`PUNCT` shape merges unattended (`--apply --tiers CASE,PUNCT`). The apply is **one store transaction** — the aliases and every edge rewrite — followed by a fact-file half that runs outside that transaction and is **resumable**: each variant's dir outcome is journalled into the apply report as it completes, and a kill in the gap is finished by `--resume <the apply report>`, which replays only the pending `dir_operations`. Suffix pairs never apply on schedule: they need a unanimous definition-based verdict from the semantic gate *and* a human to pass the tier |
| #67 | weekly | — | LLM-judges the pairs string rules cannot — differently-spelled names, abbreviations, path variants. ~40 min/run. Writes `semantic-proposals-latest.jsonl` for #48's review list; strict guard rails downgrade questionable merges to alias-only |
| #84 | daily | — | Fact-quality pass: reads the corrections log and recent-write drift, pairs contradictions, reports which claims an independent reason condemns |
| #98 | daily | #24 | Repair the one fact-tag defect the rail can count: records stored with an empty `entity`, which silently drop out of entity recall while looking healthy in the file. Snapshots the facts tree with `cp -al` **first**, then `retag_untagged_facts.py --apply` — **entity tags only**, never a text, date or provenance edit — then reads the rail's own count back and prunes snapshots keeping the newest 7. **Filed and not yet armed:** `status: draft`, so no run exists and none of its numbers is a measurement. `preferred_hours: [3]` — one hour ahead of #60's `[4]`, both read as machine-local hours, which is the frame `_is_preferred_hour` compares — is what orders it against the report; its `depends_on: 24` is an ordering *statement* only, since #24 runs six times a day |

**#48 moves fact files unattended, and only for the tiers that are mechanical.**
Step 3 of `skills/entity-resolution-sweep/SKILL.md` — the prompt the scheduler
loads for this job — ends the run with `--apply --tiers CASE,PUNCT`, on the
reasoning that a case or punctuation variant of one name is a mechanical merge
rather than a claim about meaning. One transaction writes the aliases and every
edge rewrite, then moves the fact files, behind the degraded-graph refusal
(`--allow-degraded` is never passed), the high-value-name exclusion, a store
backup and an apply report. The 2026-09-25 run record reads "Applied —
CASE,PUNCT only (3 merges) … 4 edges rewritten, 3 aliases written, 3 dirs moved,
Entities 11,992 → 11,989", and the four records before it end the same way — 30
merges and 705 fact files moved, then 111, then 6 — each one naming the suffix
queue it deliberately left alone. What stays a human act is every merge that
*claims a meaning*:
`SUFFIX_SAFE` is never passed to `--apply` on schedule, so the gate-approved
suffix queue waits on a person — 17 clusters on 2026-09-24, 19 on 2026-09-25,
growing while nobody runs them. #67 proposes into a JSONL and stops; its own
`--apply` was retired 2026-09-04. #84 reports and writes nothing; applying is an
operator act. The rails are the whole point of the arrangement, since the
2026-08-22 wipe (12,131 edges) and the 2026-09-03 151-merge incident were both
unattended applies — and the symbol-variant names (`C` / `C++` / `C#`) that the
first automatic apply merged, and a revert then undid, are why the line sits at
*mechanical* rather than at *whatever the judge approves*.

**#98 applies a repair, and the ruling that says the loop may run it is not this
page's.** The defect it fixes is the one the measurement rail can count and no
job repaired: fact records stored with an empty `entity`, which fall out of
entity recall while the file still looks healthy. `scripts/memory/retag_untagged_facts.py`
has been able to repair them since 2026-10-13, and #98 has never run as of
2026-10-16 (`select count(*) from runs where task_id=98` over
`~/lloyd-data/workers.db` was 0 that day), because its contract named an operator
for `~/lloyd-data`; the owed-check ruling that closed
#2686 on 2026-10-14 decided the remaining work was a scheduled run on this box
rather than a human task, which is the same answer #990 gave #48. So the write
joins the top tier, and it is bounded the way #48's is — **entity tags only**,
never a text, a date or a provenance field. Two duties are new to it and both
live in `skills/retag-untagged-facts/SKILL.md` rather than here. It snapshots the
facts tree with `cp -al` **before** the apply, because unlike #48 it has no
per-item journal and no `--resume`: a hardlink tree is the whole undo, costs links
rather than bytes, and holds its pre-apply bytes through the repair only because
the repair renames a new inode over each file it touches — the property
`test_a_hardlink_snapshot_still_holds_the_pre_apply_bytes_after_the_apply` pins,
since a snapshot that silently tracked the write would be worse than none. And it
prunes what it creates, keeping the newest 7, because no other mechanism bounds
that root — `scripts/groundskeeper/retention-sweep.py` has no pattern for
`facts-snapshot-*`, so an unpruned nightly snapshot of a 40,411-file tree is an
unbounded store opened on the very night a count was scheduled to come down. Its
report reads the count back through the rail (#60's scan) and not through its own
plan, because the plan is a strict superset of what the rail counts — 3,209
records in 831 files against 3,206 in 830 on the 2026-10-14 dry run — and any
check written as an equality between the two fails on arithmetic instead of on
the repair.

**#84 exits 2 rather than report zero** when the graph cannot be read — the
monitor rule it shares with #60, stated under
[Measure](#measure-60-82-70-36-76-86).

---

## Queue the work: #65, #35, #77

Three jobs that decide what future effort goes to — what gets researched, what
reaches `up_next` on the kanban at `~/obsidian/backlog/`, and how the board reads
when it is counted. Nothing in the fleet measures how well they decide. The board-side half is
measured since #904 — `scripts/automod/board_decisions.py`, surfaced through
`board_health.decisions` (dashboard backlog panel, the steward's
`<board_health>` block) and `round board-decisions`: every promotion into
`up_next` joined to the ledger event that made it and to its terminal state,
per-day counts, and retirements later reversed. It sees #35 only when a round
picks its promotion up (`unrecorded`, since the MCP store writes no ledger
row); #65's registry conversion is still unmeasured.

| ID | Freq | Role |
|----|------|---|
| #65 | daily | Propose 5–8 research topics into `research.db` via `research_propose`, from the knowledge health report, session-distill gaps, open backlog and the last two daily notes |
| #35 | daily | Promote 2–4 high-priority `draft` items into `up_next`, holding the queue to a **human attention budget** of eight to twelve promoted items and skipping when that is full unless critical/high blockers exist. The budget is what one person can review in a sitting, not the loop's intake limit: the depth gate is `implement_pool_bound`, derived as `max(implement_pool_floor, items landed in the trailing 7 days)` and counted over `ready_confirmed` items — read `implement_pool.bound` / `.ready` from `board_health`, and see [[automod]] for the bound's derivation |
| #77 | weekly | Report-only census of the board — tally by status, flag stale drafts (>30 days untouched), duplicate ids, duplicate titles. Touches no file: archives nothing, edits nothing, deletes nothing, files nothing; every candidate it names is the user's act |

**#65 feeds the `deep-research` worker and no longer reads or writes any queue
file** — [[research-pipeline]] is the registry it proposes into and the source
that drains it. The checklist it replaced held 2,839 items of which 314 were
unique and cost 1.02M tokens a night to read.

**#35 and #77 are the *human* backlog's maintenance.** The self-modification
loop's own triage and implement passes are worker sources, not autonomy tasks,
and run under [[automod]]'s gates — see [[backlog]] for the board itself and the
quarantine rule that keeps the loop from eating what it files.

**No job retires a closed item off the board, and that gap has an owner.** #77's
census on 2026-09-26 counted 1,488 `status: done` files of 1,507 in
`~/obsidian/backlog/` (98.7% terminal, no `archived/` directory of any kind), with
665 of them completed on or after 2026-09-20 — roughly 110 a day arriving. Nothing
moves one out: `scripts/groundskeeper/retention-sweep.py` prunes logs, sessions and
run records, never a backlog file; `app/backlog_move.py` is a status-move *recorder*
("One definition of 'record a backlog status move'") and never touches a path; the
only writer in the tree that removes a backlog file is `backlog_task_delete`
(`app/routers/backlog.py:849`), which deletes rather than archives and is wired to
no job. The board hides the pile after `DONE_WINDOW_DAYS = 7`
(`web/src/components/pages/BacklogPage.tsx:572`), so nothing downstream feels it
either. Archiving ~1,488 files is a bulk human-gated move in a protected path, and
naming a job that may thereafter move board files is a scope call, so **it is not an
autonomy task's to make: backlog #1566 is where that decision lives**, and a run
that reports the pile cites it (`skills/weekly-backlog-hygiene/SKILL.md`, the prompt
#77 runs, says the same). The older reading of this section — that the archiving
happens and that #77 does it — is not true, and these are the sentences that
replaced it.

---

## Bound entropy: #79, #81, #78, #80, #90

Stop the stores rotting or growing without limit. The group splits on write
authority: #79 and #81 delete and rebuild, #78, #80 and #90 only report what they
find.

| ID | Freq | Writes? | Role |
|----|------|---------|---|
| #81 | daily | yes | Prune orphaned qmd embedding chunks, backfill pending embeddings |
| #79 | daily | yes | Retention: delete `_pipeline/tasks` background-bash logs over 30 days, gzip session transcripts inactive 90+ days |
| #78 | weekly | no | Broken backlinks, unreferenced notes, stale skill references, broken cross-links → a cleanup report |
| #80 | weekly | no | OKF v0.1 conformance — `validate_okf.py` catches pages with no non-empty `type` or unparseable frontmatter |
| #90 | daily | no | Shape of four nightly-written corpora — daily notes, `USER.md`/`MEMORY.md`, `SKILL.md` bodies, trajectory rows: counts, length p95, distinct-key ratio, duplicate and self-reference rate — written to `~/lloyd-data/_pipeline/metrics/corpus-shape-*.json`, one row per UTC day, and diffed against the newest row from an earlier day |

**#81 is the one with a measured payoff**: orphaned chunks had grown the index to
24 GB and vec queries to 700 ms, and orphans displace real results, so it buys
both speed and accuracy. **#79** bounds the two unbounded-growth stores
identified in the 2026-06-11 architecture review; its gzipped archives stay
recoverable, but live consumers glob `*.json`, so archived sessions drop out of
listings and recall by design.
`scripts/groundskeeper/retention-sweep.py` is #79, pinned by
`tests/test_retention_sweep.py`; its directory now holds this script alone —
the sentence that used to read "three scripts, not two" named
`groundskeeper-survey.py` and `groundskeeper-weekly-summary.py`, and both were
deleted with #1012 (§#36) on 2026-09-23.

**#78 and #80 look like the fleet's worst jobs and are not** — see
[What the fleet actually costs](#what-the-fleet-actually-costs). Both are
report-only by design: neither has ever been asked to fix what it names, and the
reports have no scheduled consumer, which puts them in the same position as two
of the six jobs in Measure.

**#90 bounds growth the stores themselves cannot see.** #79 and #81 free bytes,
and a memory file or a skills library can grow past its usefulness while every
file on disk stays small. So #90 measures shape rather than size — daily notes,
`USER.md`/`MEMORY.md`, `SKILL.md` bodies and trajectory rows, each by count,
length p95, distinct-key ratio, duplicate rate and self-reference rate — writes
one dated row per UTC day under `~/lloyd-data/_pipeline/metrics/`, keyed on the
run's resolved UTC time rather than its second, and diffs it against the newest
row from an earlier day: a second run inside one day writes nothing, so a retry
cannot become the following night's diff base, and the base is always read before
the day's own row exists, so no run gates on a file it wrote itself (#1576). One
row per day bounds duplicates, not gaps — a day #90 did not run leaves the next
run's base older than one day. It exits 2 only when a metric crosses a bound
recalibrated from day-over-day deltas over the first 8 clean UTC dates (2026-09-27 to
2026-10-04, #2200), a sentence newly recurs across items, or a line of one of the
ten `knowledge/*/youtube-digest.md` digests is flagged by the intel pipeline's own
interest-profile guard — counted per digest on every run, by that guard rather than
by a phrase list, because the hand-written grep #2011 verified the corpus with caught
10 of the 27 sentences published (#2039). Exit 0 prints
`corpus shape: no finding`; either way it changes nothing outside its own series,
which is why it sits in
this group and in the Reports-only tier rather than in Measure, where the other
trend recorders are. `scripts/maintenance/corpus_shape.py` is #90.

---

## Measure: #60, #82, #70, #36, #76, #86

Say whether any of it is working. Five of the six change nothing; #76 is the only
monitor in the fleet with authority to act on what it finds.

| ID | Freq | Watches | Role |
|----|------|---------|---|
| #60 | daily | the graph | Knowledge Health Report: god entities (>20 facts), thin (<2), orphans, stale facts, contamination, near-duplicates, provenance coverage |
| #82 | daily | retrieval | 20-query retrieval eval against production's recall defaults; records the trend |
| #70 | weekly | the skills library | `skill-lint`, advisory — `DEAD`, `MISSING_DESC`, `DRIFT`, `DUPLICATE`, `STALE`. Read-only, writes `skill-lint-report.{md,json}` for a human |
| #36 | daily | the vault | **Retired 2026-09-23 (#1012)** — the row and the id stay because the group tables pin each other; its task file is in `_archived/` and nothing dispatches it. Read the groundskeeper queue and report counts, health score, and whether the timer ran |
| #76 | daily | the fleet | Queue health: per-task failure rate, timeouts, empty runs, `[SILENT]` rate, GPU-hours; reports the poisoned and quarantined rows the queue sweep has already triaged, and pauses failing tasks |
| #86 | daily | the Inner Voice observer | Nightly grader run appended to a metrics series; flags a sustained `dropped_rate` breach |

**Two of the six have no consumer.** #36's queue holds 31,291 items and
nothing reads it back (below). #70's findings are advisory and reach #83 only if
#83 happens to look — there is no `depends_on` between them. Under the chain
grouping those were two unrelated facts in two different sections; grouped by
function they are two of six, which is a pattern rather than two incidents.

**A monitor that reports success when it cannot see the thing it monitors is
worse than none.** #60 exits 2 and alerts when the store cannot be read, when
active edges fall below half the baseline, when any directory holds facts about
another entity, or when any fact file carries duplicate fact IDs; #84 exits 2
when the graph could not be read. It is the same rule
`_is_dependency_met`'s `if not dep_task: return True` gets wrong and the evidence
verifier's `insufficient` verdict is written against.

**#82 must pass no flags.** Until 2026-09-04 it ran `graph_rerank=False/alpha=0.5`
against a production serving `True/0.3` and scored a configuration nobody used.
Every record now carries `matches_production_defaults`, and a `false` there means
the run is not comparable to the others. It is ordered after #81's index
maintenance by window only — it declares no `depends_on`, so a late #81 does not
hold it. If a night's writes hurt recall, this is where it shows up.

**#70 owns half of "wrong skill".** Its `MISSING_DESC` and `DRIFT` cover the case
where a skill's body is fine and its *description* is not, so retrieval fires it
on the wrong request or fails to fire it on the right one. The other half — a
procedure that has rotted — belongs to #83 Stage 3 in
[Distil](#distil-38-42-39-40-47--56-57-58-83--54), which appends a correction
rather than rewriting.

**#76 reads `/api/autonomy/health`, which reads `workers.db` and not the
per-task run records.** 237 pool-timeout rows
carried a NULL `task_id` and were unreachable from any per-task view by
construction — a task that timed out on every single run was indistinguishable
from a healthy one. Its poison step is read-only: the deterministic sweep in
`workers/maintenance.py` triages that pile — one bounded revive for a transient
cause, `quarantined` for a structural one — and #76 reports those rows and the
sweep's own escalation tally instead of overwriting them, because until #1578 its
write named no `source` and reached rows owned by other sources. It pauses any
task with 3+ consecutive failures and a >50% fail rate.

**A value that resolves to nothing is shaped like an outage.** A field whose
*value* resolves to nothing parses clean here and fails at dispatch, which is the
class backlog **#811** added these checks for. A `model:` no engine serves is the
sharpest case: `resolve_model_alias` rewrites only `secondary` → `primary`,
`_get_model_env` returns `{}`, and the run then goes to the *primary's* endpoint
(the hardcoded `base_url` fallback) under a name it does not serve, where the
engine answers `404 The model 'eco' does not exist.` A 404 is shaped like an
engine being down, which is the class `_record_failure(kind="infra")` deliberately
keeps off the retry budget — so a permanently misconfigured task can look like a
transient outage indefinitely. So the linter checks that each value *resolves*, not
merely that the field is present: a `depends_on` whose upstream parses but is
parked at any status other than `up_next`/`in_progress`; a `model:` that is neither
a key of `models:` nor one of its declared `alias` values; a `skill_name`/
`skill_path` with no `SKILL.md` (or, for a path-valued reference, no file) behind
it; and since backlog **#1577** a `requires_slot:` naming a program whose enabled
flag is false. That last one is invisible to name resolution, because `secondary`
is still a key of `models:` after the slot behind it was retired. Since **#1555**
the two writers that arm a task refuse a disabled slot outright
(`app/autonomy.py::slot_arm_block`); this linter is what says the same thing before
anyone arms anything.

**#86 measures the observer, not the model.** It runs `scripts/iv_grade.py
--json` — which opens `usage.db` with `mode=ro` and writes nothing in it — and
pipes the result into `scripts/iv_metrics_record.py`, which appends **one row
per night** to `_pipeline/reflection/iv-metrics.jsonl`; it has run nightly since
2026-09-15. What it watches is whether the observer's verdicts arrive at all:
`dropped_rate` is dropped verdicts over `llm_calls`, the share of the observer's
judgment calls that returned nothing because the deadline expired or the call
errored, so it turns "the observer seems quiet today" into a trend with a bound
on it. Where that bound sits is a ruling and not an implementation detail, so it
lives in `DEFAULT_THRESHOLD` in the recorder and is deliberately not copied here.
This is the group's rarest shape — a report that has a consumer, and the consumer
is its own recorder: the median over the last `DEFAULT_WINDOW_ROWS` rows decides a
sustained breach and `announce`s it once, so the series is read back by the thing
that alerts on it. Two prohibitions are contract rather than taste. **Nothing in
this job writes `usage.db`** — `scripts/iv_grade.py` keeps its read-only open, and
`tests/test_iv_metrics_series.py::test_run_leaves_usage_db_byte_untouched`
compares the store's SHA-256, mtime and `sqlite_master` dump across a full run.
**A night whose grader emitted no JSON is never hand-written into the series**:
exit 3 means nothing was appended, and an invented row is an outage dressed as a
datapoint that also moves the next median. A missing night is supposed to stay
visible as a gap.

### #36 and the groundskeeper queue

> **Retired 2026-09-23 (#1012, Alan's ruling).** The nightly survey, its queue
> writer (`queue_io.py`), the weekly summary and both consumers
> (`process-groundskeeper-queue.py`, `batch-process-orphans.py`) are deleted,
> the `lloyd-groundskeeper-survey` timer and service are removed, and #36 is
> archived. In the queue's whole history, 30,990 of 30,990 recorded actions
> were skips of one false-positive class, and nothing repaired an item. The
> section below is the record of why; git history has the code.

The survey is the only vault-health number this machine produces, and it is
**a measurement with no consumer**.

`scripts/groundskeeper/groundskeeper-survey.py` (deleted by `201901b4`, #1012, on
2026-09-23) walked the vault plus the
~63,000-file fact tree across 11 categories and wrote
`_pipeline/groundskeeper-queue.json`. It ran at 02:30 under the
`lloyd-groundskeeper-survey.timer` systemd user unit and took about 40 minutes.
Three roots, and the distinction matters for reading any item it emits:
`VAULT_ROOT` is `~/obsidian`, `FACTS_DIR` is
`~/lloyd-data/_pipeline/vault-derived/facts` — **outside the vault** — and `MEMORY_MD`
is `~/obsidian/lloyd/MEMORY.md`.

**The scan belongs on a timer and not in a turn, and that is an incident.** Until
2026-09-03 #36 ran the survey inline. A 40-minute scan is longer than any agent
tool call can wait: the Bash timeout killed it at 2–5 minutes, the agent
backgrounded it and burned the rest of its budget polling, the task timed out
roughly **70 times a week (~12 GPU-hours)**, and each dead run left an orphaned
scan still running and competing to overwrite the queue file. #36 was
restructured that day into read-and-report — `timeout_seconds` 900 → **180**,
daily, skill `groundskeeper-survey` — and runs now take 9–70 seconds. Nothing in
the survey needs an LLM. The task file and the skill both say *do not run the
survey here*, and the reason is a comment in
`lloyd-groundskeeper-survey.service` so it survives the next person who wonders
why a timer runs a script an agent could call.

**The queue's consumers are gone.** Autonomy tasks #33 (Groundskeeper Fix Loop)
and #34 (Groundskeeper Research) drained it; neither exists in
`~/obsidian/autonomy/` nor in its `_archived/`, no live task names the
`groundskeeper-loop` or `groundskeeper-research` skills, and
`autonomy-runs/33/` and `/34/` are empty directories. As of the 2026-09-11 03:19
run the queue holds 31,291 items — 27,399 pending, 3,892 skipped — at a health
score of 58.4, and nothing reads it back.

Four properties worth knowing before touching any of it:

- **A verdict is permanent.** Idempotency is preserved by *dropping*, not by
  re-emitting: each check looks its item's stable id up in the previous queue and
  skips it when the status is `done` or `skipped`. So a resolved item never comes
  back, anything re-emitted is always `pending`, and once something marks an item
  `skipped` no later survey raises it again however the vault changes.
- **The write is atomic with an item-count check before the rename, and every
  writer shares it.** Since 2026-09-19 (#899) the guard is one function,
  `scripts/groundskeeper/queue_io.write_queue_atomic`: serialise to a `.tmp`,
  re-read it, compare the re-read item count against what was intended, and
  rename only when they match — on mismatch the temp goes, the live queue is
  left byte-identical, and `QueueWriteError` is raised. The survey,
  `scripts/memory/process-groundskeeper-queue.py` and
  `scripts/memory/batch-process-orphans.py` all called it until `201901b4`
  (#1012, 2026-09-23) retired all three; the survey used to keep
  the check inline, which is exactly how the two consumers came to re-dump a
  9.3 MB queue nightly from outside it. Each write also appends one row to
  `_pipeline/groundskeeper-writes.jsonl` naming the writing script, its pid, the
  item count before and after, how many items that run stamped, and the
  `generated_at` it read. All of this exists because of the **2026-03-30
  incident**: the fix loop processed four `BROKEN_LINK` items and wrote *only
  those four* back over a queue holding ~2,700, destroying the accumulated
  `done`/`skipped` status of everything else — which by the rule above is
  unrecoverable. The guard lives in code rather than in the skill because a rule
  an LLM has to remember is not a rule.
- **79% of the queue names a path that does not exist.** `check_thin_profiles`
  writes `source_file: memory/facts/<entity>/`, vault-relative;
  `~/obsidian/memory/facts/` is gone and the fact tree moved to
  `~/lloyd-data/_pipeline/vault-derived/facts/` (`VAULT_FACTS_ROOT`,
  `app/paths.py:278`). That is 15,299 `THIN_PROFILE` plus
  9,559 `ENRICH_THIN_PROFILE` items each naming a directory nothing can open.
  `STALE_FACT` is wrong more loudly: `relpath` against `VAULT_ROOT` on a path
  outside it yields a `../lloyd/...` escape.
- **20% of the composite score is structurally zero.** Three of the eleven
  denominators are the literal `100`, marked `# Approximate` in the source, and
  the per-dimension score is `max(0, (1 - count/total) * 100)`. `large_doc`
  scores 0 whenever the vault holds more than 100 large documents — it holds 148
  — and carries 10% of the weight. `orphan_files` scores 0 because numerator and
  denominator count different populations: the denominator excludes `memory/`,
  `agents/` and `skills/` while the scan does not, so orphans found under
  `memory/` land in the numerator and not the denominator — 3,892 against 3,756.
  Another 10%. The headline 58.4 is that much below what its own dimensions
  describe; the fix is making the denominators real, since raising the constants
  only moves the cliff.

Two defects that are live and unowned:

- **The orphan skipper still runs; it is no longer anonymous.** 253 of the
  queue's 278 skipped `ORPHAN_FILE` items read reason `hub-page-linked-survey-bug`
  and 25 read `legitimate organized project file in folder hierarchy` — the two
  strings belong to two different scripts
  (`scripts/memory/process-groundskeeper-queue.py` and
  `scripts/memory/batch-process-orphans.py`), and only the first wrote a log row,
  so one of the two writers was invisible to `_pipeline/groundskeeper-log.jsonl`
  entirely. The population is still growing: 30,990 log rows on 2026-09-19, one
  batch on every one of the 15 days 09-04→09-18, at run times drifting between
  10:21 and 14:23 UTC — not a timer, consistent with an agent turn. **What runs
  them is still unidentified** (no timer, no autonomy task, no worker source, no
  skill names either script), and the scripts were deliberately left in place at
  the time because deleting an unattributed nightly job turns a silent write into
  a silent failure. That decision did not survive the week: `201901b4` (#1012,
  2026-09-23) deleted both, with the survey and its queue writer. What changed on
  2026-09-19 (#899) is that the next pass is nameable
  from disk: both consumers now stamp a log row per item they touch — so the
  second writer's reason finally appears in the log — and every queue write lands
  a row in `_pipeline/groundskeeper-writes.jsonl` naming its script and pid.
  Reading the caller off that file after two nightly passes is the open
  needs-human step on #899. The compounding effect is unchanged and still needs
  Alan's call on whether the auto-skip verdict should exist at all: a `skipped`
  item is never re-raised, so a category worth 10% of the health score is
  zeroed a little further every night by a repair for a bug that was fixed.
- **Frontmatter is read in a 2,000-byte prefix.** `build_relation_inbound` and
  `check_stale_relations` both `read(2000)` and regex the front matter out of
  that prefix, so a `relations:` block starting past it is invisible twice over:
  its targets get no inbound credit and can be reported as `ORPHAN_FILE`, and its
  broken paths are never reported at all. This is the shape of the bug the
  dashboard's `_frontmatter` reader had — a flat byte prefix hides precisely the
  files with the longest front matter, which are the files with the most
  relations. Unfixed; recorded because it fails silently in both directions.

**Two orphaned skills are still loadable.** `groundskeeper-loop` and
`groundskeeper-research` remain under `~/obsidian/skills/` at `status: active`
with no task naming either, and both still describe running the 40-minute survey
inline — `groundskeeper-research` Step 1 is literally
`Bash("python3 ~/lloyd/scripts/groundskeeper/groundskeeper-survey.py")`, which is
the thing that cost ~12 GPU-hours a week until 2026-09-03. `groundskeeper-research`
also instructs the agent to append to `groundskeeper-research-log.jsonl`, which
has never existed on disk. An orphaned skill is not inert: it is retrievable by
name.

---

## Watch the machinery: #91 #92 #93 #94 #95 #96 #97 #99 #100 #101 #102 #103

Measure asks whether the fleet is working. These twelve ask a different question —
whether the *machinery the fleet runs on* still looks like what its own
documentation says — and they read it from outside: the engine's reasoning traces,
the serving process's cache and KV window, other jobs' run records, the memory
surface a prompt is assembled from, the mitigations that stop a run, the decision
engine on GPU 2, and the self-modification loop's own landings. None of them is
about a job's output, which is why they are not scattered through the six groups
above and why they arrived as a block: seven of the twelve were created between
2026-09-29 and 2026-10-11 and none of those seven had an entry here until #2630;
the eighth, #99, joined the directory and this table both on 2026-10-16 (vault
commit `5394a06e`, 01:35Z; its own `created:` is `2026-10-16T00:00:00Z`); the
ninth and tenth, #100 eval instrument index publish and #101 gate-to-land lag
daily record, landed in the directory on 2026-10-16 (their own `created:` stamps
are `2026-10-16T13:00:00Z` and `2026-10-16T20:50:00Z`) and entered these tables
two days later, on 2026-10-17, because the membership node named the omission and
a person had not read the directory. Neither job has ever run: `select count(*) from runs
where task_id=100` over `~/lloyd-data/workers.db` returned 0 rows as of 2026-10-17, and so
did the same read for 101 — so their two rows here are stated from the script and its skill,
which is the caveat #98 carries at §Canonicalize for the same reason. Both append one dated
row to one journal outside this repo — #100 by running `python -m scripts.eval_instrument_index`
with nothing added to it, #101 by `scripts/maintenance/gate_to_land_lag.py --record`, where
the flag is the only thing that makes the run append rather than merely print. That is the
recorder shape #97 (#2572) arrived with, and it is what settled #100's group: the owed-check
pass ruled on 2026-10-17 (#2881's owed list) that #100 belongs here beside #97 — it appends
one dated row to one journal outside the repo and files nothing — and that §Measure stays
closed to it, because §Measure is the group whose jobs score the fleet: being about an
instrument is not the same as measuring it. Re-opening the placement needs a measurement of
§Measure's counts moving, not a subject-matter preference.

The eleventh and twelfth, #102 find-all arms run and #103 code rules census weekly,
landed in the directory on 2026-10-18 and 2026-10-19 (their own `created:` stamps
are `2026-10-18T07:42:00Z` and `2026-10-19T01:40:00Z`) and entered these tables on
2026-10-19 because the nightly skills-management Pre-Flight named the omission — the
fourth time the directory ran ahead of the two tables, and the moment #2895's
filing-time report was built to speak at. Both are `status: draft`, so neither is
armed; the rows record what a filed task file names, which is how #98 and #100 are
already listed here. Their tiers split, and the split is the tier table's own
question — may it write while nobody is watching — not the size of what it leaves:
#102 publishes its own measurement table and trial rows under `eval/measurements/`
and files nothing, so its write *is* its report and it sits with the reporters,
while #103 drives `scripts/maintenance/code_rules_census.py`, and that script opens
a backlog item itself whenever the census has hits, which is a board file another
job or a person acts on — the #92/#93/#99 shape, so #103 is in the unattended-writer
tier. #3012 is where that reading was taken and where both rows were written.

| ID | Freq | Job | Reads | Writes | Must not |
|----|------|-----|-------|--------|----------|
| #91 | daily | Nightly thinking-fidelity prevalence series (`thinking-fidelity-prevalence-series`, #1770) | every session file under `~/lloyd-data/sessions`, walked once by `scripts.thinking_fidelity_scan --record --per-day` (#2819: the day roll-up and the whole-store row were two walks over the 10,082 `*.json` of 2026-10-16, and the whole-store row now carries the walk's own `scan_seconds`) | one row per calendar day plus a final `scope: whole_store` row, appended to `~/lloyd-data/_pipeline/metrics/thinking-fidelity-prevalence.jsonl` | `config.yaml` or `preserve_thinking_iterations`: switching the capture to `false` makes the marker rate fall without one fabricated trace going away, which is the failure #1510 closed |
| #92 | daily | vLLM prefix-miss daily (`vllm-prefix-miss-daily`, #2006) | `usage.db`, `event_logs/*.events.jsonl`, and the primary engine's status lines in `~/lloyd-data/logs/services/agent-llm-primary.log*`, through `scripts.vllm_prefix_miss_window` | `~/lloyd-data/vllm-prefix-miss/vllm_prefix_miss_<counted-day>.json`, plus a backlog item when a §10 trigger moved or a day is about to rotate out of the ~11-file log window uncounted | stage or commit anything: `workers/sources/arch_review.py` and an automod round are the two things that commit onto live main, and a nightly reader that staged its own fixture would be a third |
| #93 | daily | Step conformance replay (`step-conformance-replay`, #2104) | the last 7 days of run traces under `~/lloyd-data/autonomy-runs`, replayed against each task's learned `expected_steps` ledger by `scripts/step_conformance.py replay --days 7 --support 1.0 --escalate-writes` | its printed lines verbatim into its own run record, and the one escalation: a backlog draft for a flagged run that wrote no artefact every other run of its task wrote | `learn --out` and `replay --json`; the ledger and the trajectory store stay closed for writing, and the job never adjudicates, promotes, closes or re-triages the draft it filed, never injects into a session, never opens a round |
| #94 | daily | Referential integrity ledger (`referential-integrity-ledger`, #2151 from #882) | every backticked path and `path:N` cite in `lloyd/MEMORY.md`, `lloyd/USER.md` and `memory/mental-models.md`, resolved against the repo, vault, data root and the citing file's directory | `~/obsidian/autonomy/referential-integrity-latest.md` and a dated copy under `~/obsidian/autonomy/referential-integrity/` — the only two paths it may touch, both through Bash; exit 1 is the expected result every night, because baseline dangling cites exist | edit a cited memory file or any other file, fix a dangling cite, pass `--expect-dangling` to move a number, file an item, or read an unchanged count as a problem to escalate |
| #95 | daily | Mitigation drill, read (`mitigation-drill`, #2432 over #703/#2153) | `GET /api/workers/status`'s `.mitigation` block — one Bash call — which publishes what `~/lloyd-data/mitigation_drill.json` holds | nothing at all: `python -m scripts.mitigation_drill` from the pool's maintenance seat is the only writer of that state file, and a drill run of its own is refused while a self-modification round holds the pool | fire the drill, re-run the GET to refresh a stale reading or to confirm a `no-op`, edit the script, the state file or a threshold. A latest classification of `no-op` is a control regression to be named in one line; `{"state": "never-run"}` is a report, not a failure |
| #96 | weekly | djev name-prior probe (`djev-name-prior-probe`, #2215 from #1452) | the pinned corpus `eval/djev/name_prior_corpus.jsonl` — 38 prompts on 2026-10-12, 10 dedupe / 10 entity / 10 edges / 8 rank, whose `sha256` every report records — one `choice` question per prompt against the decision engine | `~/lloyd-data/eval/djev/name_prior_<report date>.json`, the probe's default `--out-dir` since #2455, so a run leaves no file inside `~/lloyd` for anybody to `git add`; the date key is the run's **UTC** day (`build_report`'s `_now()` instant converted with `timezone.utc`), the same calendar the `runs` row that records this probe is stamped in, so the weekly series joins to `runs` on date — it used to be the probe's *local* date, which is why `name_prior_2026-10-11.json` on disk is the report whose `runs` row reads `2026-10-12T06:30Z` and why an artefact can read as missing on the day it ran (#2820); `generated_at` beside it keeps that zone's offset | add a flag — `--samples`, `--url`, `--out-dir`, `--date` and `--corpus` are all absent from the command the skill gives — re-run the probe, or edit it; exit 3 means the engine was unreachable, no report was written and no flip rate was measured |
| #97 | daily | automod landed-outcomes daily record (`landed-outcomes-daily-record`, #2572 from #2378) | `git log`/`git show` over `~/lloyd` plus the automod ledger, split by author and put through a Wilson interval by `scripts/automod/landed_outcomes.py`; it makes no model call | exactly one JSON row appended to `~/.local/state/lloyd-automod/landed_outcomes.jsonl` — `--record` is what makes it append, and without the flag the module prints, writes nothing and still exits 0 | add `--since`, which moves the window every rate in the row is measured on; edit, truncate, re-sort, compact or dedupe the journal; run it twice in one turn; hand-write a row |
| #99 | weekly | supply-chain inventory regen (`supply-chain-inventory-regen`, #2796 from #2593): enumerate every distribution installed in the venv roots this box's launch scripts name, scan the set offline with `app/harness/supply_chain_inventory.py`, and compare the two headline numbers with the previous dated report, falling back to the values quoted in its own `description` (the committed artifact's 698/150 of 2026-10-11, the 2026-10-16 reading of 906/150) | every `~/lloyd/.venvs/<name>/lib/python3*/site-packages` a launcher under `agent-services/` names, plus the account user-site `~/.local/lib/python3.*/site-packages` — the roots `site_roots()` walks; no `~/lloyd-data/venvs` exists and no run opens `eval/supply-chain/baseline.yaml`, and the scan opens no network | the synthesised per-site requirements and the merged scan input under `~/lloyd-data/supply-chain/observed/`, one dated JSON report beside that directory, and a backlog item only when the *missed-advisory* delta moves or the committed verdict's meaning changes | calling `Write` or `Edit`; any `git` or `rm` command |
| #100 | daily | eval instrument index publish (`eval-instrument-index-daily`, #2797): run the publisher once, unflagged, read the journal back with `wc -l` for the row count, and report its one headline line | the seven instruments' own artifacts — the retrieval baselines under `~/lloyd-data/eval/baselines`, `~/lloyd-data/_pipeline/reflection/iv-metrics.jsonl`, `~/.local/state/lloyd-automod/scorecard.jsonl`, `~/.local/state/lloyd-automod/landed_outcomes.jsonl`, `~/lloyd-data/eval/tool-quality`, `~/lloyd-data/eval/funnel` and `eval/uptake` — each one's newest dated row, tiered by the maturity rubric and trended by Theil–Sen in `scripts/eval_instrument_index.py`; plain Python over files already on disk (0.09 s on 2026-10-16), no model call, no engine touched | exactly one JSON row appended to `~/lloyd-data/eval/instrument-index/instrument-index.jsonl` by `append_row`, the module's only write, which creates nothing outside that directory | never add a flag: `--no-record` appends nothing and still exits 0, `--now` stamps a row whose `date` is not today, `--out-dir` grows a series nothing reads; never rewrite, re-sort, compact or dedupe the journal, run it twice in one turn, hand-write or patch a row to keep the streak when exit 1 refuses the totals, file a backlog item or open a round — exit 1 is the designed output |
| #101 | daily | gate-to-land lag daily record (`gate-to-land-lag-daily-record`, #2858): join the newest passing `gate` row to the `promoted` row of the same `round_id` and append the day's lag distribution | `~/.local/state/lloyd-automod/promotions.jsonl` *and every* `promotions-archive-YYYYMM.jsonl.gz` beside it — 26,798 of the 47,767 ledger rows are archived (measured 2026-10-15), so a reader taking only the live file described under 60 % of the corpus and reported the shorter lag — through `scripts/maintenance/gate_to_land_lag.py --record`, and `--record` is the only flag the skill permits (`--state-dir` moves both sides off the live state dir, `--json` deletes the report lines a later reader checks) | exactly one JSON row appended to `gate_to_land_lag.jsonl` inside `~/.local/state/lloyd-automod/` — a file that does not exist yet, because the job has never run — whose receipt is the printed `recorded → <path>` line because `main()` returns 0 with or without the flag; every p50 travels with the `n_rounds` it is computed over, so a thin week cannot read as a tight one | never edit, truncate, re-sort or re-stamp the promotion ledger or this journal, never re-run the day to move a number, never a second copy of the command in one turn; `identity_refuse_count` stays `null` because this reader joins only `gate`/`promoted` rows, and a moved-tree trend belongs to a human ruling on the #2859 chase |
| #102 | weekly | Find-all arms run (`find-all-arms-run`, #2962): drive the retrieval-phrasing A/B to completion and publish its table | the engine lock `~/.local/state/lloyd-automod/primary.lock`, waited on with a bounded timeout instead of `flock -n`'s silent exit, plus the four gold counts it re-derives before spending any engine time | `eval/measurements/find-all-<UTC-date>/trials.jsonl` and the published `eval/measurements/find-all-<UTC-date>.md` — the only artefact in this group that lands inside the checkout, left uncommitted as every measurement under `eval/measurements/` is | report the run's result from the turn that launched it: the driver runs 1.5-2.5 h against the pool's 3570 s cap and is started with `setsid nohup` precisely so it outlives the session, so the table is read on the next fire and never promised by this one |
| #103 | weekly | Code rules census (`code-rules-census-weekly`, #2995 from #2810): one whole-tree pass of the shipped content rules, through the one command the skill gives with nothing added to it | every tracked `.py` the census scans (1397 scanned of 1397 tracked on the 2026-10-19 dry run) against the rules in `scripts/automod/code_rules.toml`, with the allowlist, the suppression markers and each rule's own `covered X of Y` counted beside the hits — the denominators are the only thing a month-old row cannot reconstruct | one dated row `~/lloyd-data/_pipeline/code-rules-census/code-rules-census-<UTC-date>.json`, at most one per UTC day, **and one backlog item whenever the census has hits** — filing runs only after `write_row`'s `O_EXCL` claim is taken, so the day that writes the row is the day that files, and a run that cannot claim it files nothing and prints the filing the existing row already carries (#3257, which fixed the reverse order: `file_hits` used to run first and a same-day retry opened a second item for one signature) | pass `--no-file`, `--out`, `--tree`, `--rules-path` or `--backlog-dir`, each of which moves the measurement or the filing somewhere the next reader will not look; a second run in one UTC day is harmless now but never useful — it writes no row, so a signature the tree gained after the first run lands in no row and waits for the next pass |

Nine of the twelve carry an `acceptance.objective_checks` block naming `Write` and
`Edit` in its `tool_not_called` list, so their read-only bound is checked by the
runner and not by the prose. #91, #92 and #102 carry no such block: for those three the
prohibition lives only in the `description` field a run is handed, which is why
`preserve_thinking_iterations`, the staging rule and the launch-then-return rule are
spelled out there in full.

**Five of the twelve are not report-only, whatever the report looks like.** The
write-authority table asks one question — may it write while nobody is watching —
and #92 files a backlog item, #93 files a backlog draft and #94 overwrites a
latest-path ledger in the vault plus a dated copy beside it, and #99 files a
backlog item when the missed-advisory count moves, and #103's census script files a
backlog item whenever the census has hits. Those are files another
job or a person will act on, so the five sit in the unattended-writer
tier beside #30 and #53. #91, #95, #96, #97, #100, #101 and #102 are the report-only seven: one dated
artefact each under the data root, the automod state dir or `eval/measurements/` —
#95 leaves not even that, since it reads a state file another process owns, and
#102 is the one in this set whose artefact sits inside the checkout rather than
under a data root — and none of the seven files anything in the vault or the queue.
#2630's triage read all seven of its own arrivals as report-only from the artefact
paths; the three of those that file are where reading the artefact path alone was
wrong, and the tier is where that difference has to be visible.

The group has one shape, and it is the shape #86 arrived with: read something the
rest of the system produces, append or overwrite exactly one dated artefact, print
a line whose numbers a later reader can re-derive, and hold every judgement for a
person. Four readings follow from that and are worth having written down, because
each one looks like an outage from the run record alone:

- **#94 exits 1 every night.** Twelve baseline dangling cites existed on 2026-10-04
  (`200 cites, 12 dangling, 4 exempt`), so exit 1 is the ledger working — but 12 is
  not a standing baseline to compare a night against: run 10 read `342 cites,
  23 dangling, 8 exempt, 2 newly dangling` on 2026-10-14 and run 11 read `361 cites,
  23 dangling, 8 exempt, 0 newly dangling` on 2026-10-15, and no open backlog item
  names the 23 (#2821). Exit ≥2 or a traceback is the failure, and the
  `; echo "EXIT=$?"` on the command line is there so the expected 1 is not recorded
  as a tool error.
- **#95 reporting `-` is not a gap in #95.** A `never-run` state or a stale newest
  `at` means the maintenance seat has not run the drill; the job's whole job is to
  say so in the shape of one line and stop.
- **#91's day rows are snapshots until their day closes.** 2026-09-28 read
  45/10005 while the day was open and 45/10324 after, so a falling marker rate is
  only ever judged across closed days — and the lever that would legitimately move
  it is the prompt, not the capture setting.
- **#91's day rows and its whole-store row are one walk now, and `scan_seconds` is
  the only field on that row that is not a measurement of the store.** Until
  #2819 the day rows came from one walk of the store and the whole-store row from a
  second, so the two could disagree about a session that was still being written
  (the two `files` values beside each other in the file were the tell), and a run
  killed partway wrote nothing at all — losing every day row, which is the deliverable.
  Now each file is read once and folded into both tallies, and a day's row is
  appended the moment that day's slice finishes, so a killed run keeps the days it
  covered and the whole-store row that never arrives is the only loss. Read the
  consequence the other way too: a series with a whole-store row missing at the end
  is a run that died mid-walk, and the day rows above it are complete for their own
  days. `scan_seconds` is new from #2819 and absent on every earlier row — it is the
  walk's wall clock (47.0 s over the 10,105 files of 2026-10-16, where the two-walk
  script took 106.3 s), and its absence on an old row means nobody timed that run,
  never that the run was fast. Task #91 declares no `acceptance.objective_checks`,
  so that field is currently the only thing in the fleet watching its margin.
- **#99 belongs to this group by its authority, not by its report.** Filed
  2026-10-16 and still `status: draft` with zero rows in `runs` — #2797 closed
  `done` on 2026-10-16 listing the one-line `status` flip as *still owed*, so the
  live owner of that flip is #2796's owed list and not a closed item; it is the same
  standing human decision that holds #98 (`draft`, also zero runs) — it re-runs the
  observed
  supply-chain inventory offline and can file a backlog item when the
  missed-advisory count moves,
  which is #94's reason for being here, and the evidence it maintains is a file
  nothing else in the fleet reads, so its going stale is invisible, which is #92's.
  That makes it an unattended writer even though every run ends as a report, and its
  own contract prohibits the two tools and the two commands that would let a run
  write into the tree instead of beside it. It is also the recurrence: still absent
  from both membership tables when the nightly Pre-Flight ran the membership node on
  2026-10-16, and named here only because that node went red.
- **#93 reporting an empty window is a finding.** Zero baselines and zero flags is
  reported as the absence it is, never as a pass; the detector has been read-only
  for most of its life and an empty window is how that keeps looking fine.

**#97's row is a record now.** `select count(*) from runs where task_id=97` over
`~/lloyd-data/workers.db` returned 4 runs as of 2026-10-16T10:00Z, all
`status=success`, the first completed 2026-10-13T08:06:34Z and the newest
2026-10-16T08:01:48Z, and
`~/.local/state/lloyd-automod/landed_outcomes.jsonl` holds one dated row per run —
4 rows, newest `generated_at` 2026-10-16T08:01:08Z; that run's own record reported
`landed-outcomes: rows=4 latest=2026-10-16T08:01:08Z cost=ok latency=ok`. So the
reads and writes in the row above are confirmed against an artefact and not stated
from `skills/landed-outcomes-daily-record/SKILL.md` and the module's `record()`
docstring — the state §Review log records as the one held on 2026-10-12. What even
today's
reading still understates: the newest row's `cost.join` share
(`k=489 n=516 share=0.9477`, the 2026-10-16 row) is a read-only query over that same
`runs` table, so
the job reads the store the fleet's run records live in as well as the automod
ledger — said here rather than in the table cell, which is #2631's reconciler's to
edit. And the count above is dated on purpose: #97 fires daily at ~08:0xZ, so a
bare number is false by the next morning and only the re-derivable one stays true.

Two of these ten are the reason the other eight are trusted at all: #93 is the
only thing that notices when a run stopped doing a step every other run of its task
did, and #94 is the only thing that notices when loaded memory starts pointing at
files that are not there. Both are deliberately silent about what they find —
`notify_on_complete: false`, no injected context, and one backlog draft as the
entire alert surface — because the record behind that choice is the guardian's:
every rollback so far was a false positive, so a page is not this group's currency.
A group whose findings arrive as drafts and dated rows is easy to mistake for an
inert one. It is not: #93 exists because the detector shipped read-only in #673 and
nothing called it for six weeks, and #94 exists because `skill_lint` covers SKILL.md
only and the groundskeeper covers wikilinks, so nothing at all resolved the
backticked cites that every prompt in this system leans on.

---

## What the fleet actually costs

A snapshot, not a fact — the seven days to 2026-09-11, from `workers.db` where
`source = 'scheduled-task'`. Reproduce it rather than trusting the numbers; they
are here for the *shape* of the load, which is lopsided in a way no per-task view
reveals.

| Job | Function | Runs | GPU-h | Note |
|---|---|---|---|---|
| #68 triage | ingest | 407 | 7.3 | the quarter-hourly job — the only run count in the hundreds |
| #24 data pipeline | build | 28 | 8.5 | the single most expensive job in the fleet |
| #74 mention classifier | build | 9 | 2.2 | cold-start backlog amortizes over days |
| #39, #42 | distil | 7 each | 2.0 each | the reflection chain's two heavy halves |
| everything else | — | 1–20 each | ≤1.8 | the weeklies land at 0.0–0.6 |

**Two jobs carry 40% of the fleet's ~40 GPU-hours**, and they sit in the two
functions that touch the outside world and the graph — ingest and build. Distil
is the next, through #39 and #42. Canonicalize, Queue the work, Bound entropy and
Measure are nearly free between them. If a change has to make the fleet cheaper,
#24 and #68 are the only two places where it can matter.

**Aggregate a week of this fleet and you describe a fleet that no longer exists.**
Failures over the same window, by day: **35, 7, 15, 9, 15, 10, 2, 1**. The rate
collapsed across the window, so any seven-day failure count is dominated by its
own first half. Read a day, not a week:

- **#24** shows 9 failures in 28 runs, and its last six runs are all green — the
  one recent failure was `max_turns` at 61 turns on 2026-09-10.
- **#78** (1 success in 7) and **#80** (1 in 5) look like the fleet's worst jobs
  and are not. Every one of those failures is timestamped 2026-09-08 or 09 —
  `timed out after 300s`, `timed out after 600s` — and both are *weekly*, so each
  has had exactly one run since and both succeeded. #80 runs `validate_okf.py`, a
  2.4-second script, on a 300 s budget: it had the answer in the first minute and
  kept investigating until it was killed without ever being asked to write down
  what it found. That is the failure the deadline anchor in [[autonomy]] was built
  for, and this is what it looks like from the other side of the fix.

A run row also outlives its task. Rows for **#52** (deep-dive research, last run
2026-09-07, archived) and **#75** (the AI Engineer channel monitor, last run
2026-09-09, superseded by the `youtube-digest` worker source) fall inside the
window with no task file behind them.

## Related

- [[autonomy]] — the mechanism: due-gates, failure ladder, deadline anchor, run records, fleet health
- [[workers]] — the one queue every one of these is dispatched through
- [[background-runs]] — why every run is recorded and what makes Inner Voice opt-in
- [[skills]] — the skill loader and the quarantine set the trace2skill chain writes against
- [[knowledge-graph]] — the store the graph chain writes to
- [[research-pipeline]] — the registry #65 proposes into

`nightly-reflection`, `nightly-skills-management`, `morning-briefing` and
`groundskeeper` were four separate docs until 2026-09-11. They covered 8 of the
fleet's 32 jobs between them and left 24 undescribed, so they were folded in here
and taken out of the tree. There is no architecture/.archive/ in this checkout —
the copies git last carried were untracked by `f80c9d00` — so their text is
recoverable only from history, at
`git show f80c9d00^:architecture/.archive/<slug>.md` for any of the four.
Their incident history is carried above;
what was dropped was the retired two-loop groundskeeper architecture, the
per-line source citations, and every restatement of a schedule that lives in the
task file.

This document was grouped by chain — which pipeline a job's output feeds — until
later the same day, and is now grouped by what each job is for. The chains are
real and still gate execution, so they are drawn in
[The wiring](#the-wiring-which-this-layout-no-longer-carries) rather than being
the layout. Three jobs moved shelf and the move is the point: #54 was under
inbound signal and reads nothing inbound; #36 was under vault hygiene and repairs
nothing; #65 was under inbound signal and belongs with the other two jobs that
decide what gets worked on next.

## Review log

- **2026-09-12 — `current`.** Every path, symbol, config key, task id, cadence and
  measured count re-verified against the tree and the live routes at
  `28abecf073ee`: the 32 live tasks and their `depends_on` set (ten edges, three
  crossing a function boundary), the 74-row function grouping, the quarantine and
  `NON_USER_PLATFORMS` literals, the inject 409 versus the unguarded prefetch, the
  three `[SILENT]` call sites, the script thresholds (30/90-day retention, god >20
  / thin <2, #36's 180 s), `validate_tasks.py`'s coverage and its
  `model:` blind spot, `workers/evidence.py`'s three-state contract, the
  groundskeeper queue (31,291 / 27,399 / 58.4 / 18,176 log rows), and the empty
  `autonomy-runs/33` and `/34`. Five statements were wrong and are corrected: the
  pilot frozenset is in `app/autonomy.py:710` and not `workers/evidence.py`; the
  evidence pilot has verified **nothing** (0 bundles across 22 piloted runs,
  `claims_checked: 0` over 713) because `scheduled_task.py` drops the `claims` key
  before `pool.py` reads it (#902); the reflection windows are offset pairwise, not
  disjoint — hour 01 is still offered to both #42 and #39; `scripts/groundskeeper/`
  holds a third script; and the orphan skip is 3,867 + 25 with two reason strings,
  not 3,892 with one. Also appended to #899 and #900 — the unbound-skill hazard is
  five skills, all five advertised in the index today.

- **2026-09-27 — `retired`.** #85 (`Secondary Routing Eval`) leaves this doc's
  membership tables, the write-authority table and the Measure anchor. Alan ruled
  it retired, not re-scoped (backlog **#1577**): `config.yaml` has carried
  `secondary_enabled: false` since `551e9044` (2026-09-20, "Retire the secondary
  slot and put DiffusionGemma (djev) on GPU 2"), so there is no second arm for a
  paired routing eval to measure — and djev is not a chat slot and nothing routes a
  turn to it, so re-scoping the comparison onto it would measure nothing real. The
  sentence this replaces, "#85 is the fleet's only `draft` task", was false the day
  it was written: #68 has carried the same status throughout. What the retirement
  is: the numbered task file was deleted from `~/obsidian/autonomy/` rather than
  moved to `_archived/`, and what survives is the skill it drove —
  `~/obsidian/skills/secondary-routing-eval/SKILL.md` — beside
  `eval/secondary_routing_eval.py`, both left on `main`, because a task file is
  cheap to re-add if a second chat engine ever returns. The blind spot that let the
  declaration sit for a week is closed the same day — `validate_tasks.py` now
  reports a `requires_slot:` whose enabled flag is false, answering from
  `app.llm_slots` rather than a table of its own, and `--strict` exits 2 on it. The
  dated fleet-size counts elsewhere in this doc are measurements of the day each
  was taken and are not restated here; a retirement moves the fleet, not them.

- **2026-10-12 — `stale`.** Re-verified against `3b438c45` and the live
  `~/obsidian/autonomy/`. The wiring holds: exactly the ten `depends_on` edges
  the diagram draws exist, and `EVIDENCE_PILOT_TASK_IDS` really is
  `app/autonomy.py:2898` pinned at `tests/test_worker_evidence.py:686` (both numbers
  retyped on 2026-10-14, the drift-in-place this file's own 2026-10-12 pass does to a
  stale pointer: #2675 widened the roster to seven and wrote its per-id reasons above
  the definition, which moved it from 2818, and moved the pin's assert from 663).
  Membership does not: seven live tasks (#91–#97) have no entry here at all,
  #87–#89 are named as live members with no task file behind them, and #68 is
  parked (`draft`, last run 2026-09-16) with a `model: secondary` that names a
  slot disabled since 2026-09-20 — invisible to `slot_arm_block`, which reads
  only `requires_slot:`. Corrected in place: four drifted `file:line` citations
  (`test_skill_tool_names.py` 137→150, `skill_lint.py` 276→281,
  `app/routers/backlog.py` 754→804, `BacklogPage.tsx` 571→572), the
  groundskeeper "three scripts" sentence (the directory holds one since
  #1012), #24's corpus spelled with its full path, #36's Measure row marked
  retired, and both of #68's "runs around the clock on the secondary engine"
  sentences. Filed: #2630 (the seven missing entries), #2631 (no check
  reconciles the doc against the task-file directory), #2632 (#68's stale
  `model:` and the guard blind spot), #2633 (pin the citation form so line
  drift goes red).

- **2026-10-12 — `current` for the fleet's membership; `stale` for one row.**
  #2630 wrote the seven entries and took out the three that had no task file.
  #91–#97 are a group of their own, [Watch the machinery](#watch-the-machinery-91-92-93-94-95-96-97)
  (the slug as it stood that day; #2813's row moved it — see 2026-10-16),
  because what they read is the machinery the rest of the fleet runs on rather than
  the outside world (Ingest) or a job's own output (Measure). Their write authority
  split three/four against the triage's read that all seven were report-only: #92
  files a backlog item, #93 files a backlog draft and #94 overwrites a vault ledger,
  so those three sit with the unattended writers and #91 #95 #96 #97 are the
  report-only half. The tier row counts did not move: the trio left the writer row
  as the three arrived, and the report-only row went 8 → 12. #87–#89 are gone from
  the function row, the tier row, the §Ingest heading and their own three rows; their
  run records stay under `~/lloyd-data/autonomy-runs/`, and a record is not a
  member. Two fleet counts were corrected rather than restated: §The wiring's
  no-dependency sentence and §Distil's opening now read 29 of 39 and 10 of 39 on
  `GET /api/autonomy/tasks` (the ten `depends_on` edges are unchanged — none of the
  seven declares a parent), and the numbering floor moved to 97 with its date, which
  is what the sentence asks a correcting pass to do. What is still owed on this
  document: #2631's reconciler against `~/obsidian/autonomy/` (both directions of
  this failure are still one `ls` away from being caught), #2632 (#68's stale
  `model:`) and #2633 (the citation form).

- **2026-10-16 — `current` for #97, and one owed item settles.** What §Watch the
  machinery said until this morning, kept here because it was a true measurement of
  its own day: #97 had never run — `select count(*) from runs where task_id=97`
  over `~/lloyd-data/workers.db` was 0 on 2026-10-12, the module was created at
  19:52Z on 2026-10-11, and `~/.local/state/lloyd-automod/landed_outcomes.jsonl`
  did not exist on disk yet — so its reads and writes were stated from
  `skills/landed-outcomes-daily-record/SKILL.md` and the module's `record()`
  docstring, and the paragraph said so plainly rather than hiding it. Measured
  2026-10-16 instead: 3 runs, all `status=success`, the first at
  2026-10-13T08:06:10Z, one dated row per run in that journal (newest
  `generated_at` 2026-10-15T08:01:30Z), and the run's own report line
  `landed-outcomes: rows=3 latest=2026-10-15T08:01:30Z cost=ok latency=ok`. The live
  paragraph is that count beside the command which re-derives it, and dated — the
  job fires daily, so a bare number rots overnight, which is the same reason the
  relative "the day this was written" the 2026-10-12 entry used for this count is
  re-dated to 2026-10-12 here instead of being left to read as current. Three
  consequences, in order: the fourth item that owed list carried ("one row above
  that cannot be confirmed until its job has run") comes out of its sentence, the
  journal being the confirmation it waited on; #98's parallel claim at §Canonicalize
  keeps its words and gains what it lacked, since `count(*) where task_id=98` really
  is 0 on 2026-10-16 and the sentence now carries that date and its command; and the
  store #97 queries to produce its `cost.join` share is still missing from its Reads
  cell, which is #2631's reconciliation to make and not a table edit from a doc
  pass.

- **2026-10-16 — `current` again, one row late, and a node is what caught it.** #99
  supply-chain inventory regen (#2796) arrived in `~/obsidian/autonomy/` with its
  own `created: 2026-10-16T00:00:00Z` stamp and neither membership table named it: the same failure #2630 closed,
  one id and one day after #98 did it. Read-only through the node's own helpers it
  reported `live 41` and `([], ['#99'], [])` for both tables — no ghost, no
  duplicate, exactly one unlisted id. It is named in both now, in §Watch the
  machinery and in `Writes durable state unattended`, because it can file a backlog
  item (#94's reason) over evidence nothing else in the fleet reads (#92's), and
  because #98 — also `status: draft`, also in both tables — is the precedent that
  arming is not what earns a row. The hand-typed floor moved to `24 through 99 or
  beyond` and the group's own figures moved with its heading: seven members became
  eight, the `objective_checks` count went five to six (measured across the eight
  task files: #91 and #92 stay the only two with no block), and the unattended-writer
  split three to four. Two figures were deliberately left alone — §The wiring's
  no-dependency sentence and §Distil's opening count the fleet that actually runs,
  and a `draft` job is not in it, so moving either would red the node that checks
  those two against the arrows this document draws. What this entry cannot fix is why
  the drift was ever possible: both membership nodes carry `@live_vault` and the gate
  runs `-m "not live_vault"`, so nothing between a task file landing and the nightly
  Pre-Flight reads the directory against these tables.

- **2026-10-17 — `current` again, one day late, third time the same class.** #100
  eval instrument index publish and #101 gate-to-land lag daily record landed in
  `~/obsidian/autonomy/` on 2026-10-16 (`created:` 13:00Z and 20:50Z) and neither
  membership table named them: #2813's failure one day on, and the third instance of
  the class after #2630. Through the node's own helpers it reported `live 43` and
  `([], ['#100', '#101'], [])` for both tables — no ghost, no duplicate, exactly two
  unlisted ids. Both are named now, in §Watch the machinery and in `Reports only`:
  each appends one dated row to one journal outside this repo and neither files a
  backlog item or opens a round, which is #97's position and the opposite of
  #92–#94's and #99's; their two task files carry `tool_not_called: Write` and
  `tool_not_called: Edit`, so the tier reading is the runner's, not this doc's
  inference. The owed-check pass ruled on 2026-10-17 (#2881's owed list) that the
  placement written here is the settled one: #100 sits beside #97 in §Watch the
  machinery and §Measure stays closed to it, because being about an instrument is not
  the same as measuring the fleet. This entry moved no membership, so §Measure's id
  table and its own counts stayed as they were. The hand-typed floor moved to
  `24 through 101 or beyond`
  and every figure the membership moves with it: seven members became ten, the
  `objective_checks` count six to eight (both new files carry a block, so #91 and #92
  stay the group's only two without one), the report-only half four became six, and
  the tier's trailing count went 12 to 14 while `Writes durable state unattended`
  stayed at 26. §The wiring's no-dependency sentence and §Distil's opening count are
  again left alone for the reason #2813 states — they count the fleet that actually
  runs, and neither job is armed or has ever run (`select count(*) from runs where
  task_id=100` over `~/lloyd-data/workers.db`: 0 rows as of 2026-10-17) — and #2882
  is the open item that counts the census the two `model: primary` lines belong to.
  The gap this entry exists to record is unchanged: both membership nodes carry
  `@live_vault`, the gate runs `-m "not live_vault"`, and `validate_tasks.py` never
  opens this document (`git grep -c autonomy-jobs scripts/autonomy/validate_tasks.py`
  → 0 on 2026-10-17), so a task file can still land unseen between a directory write
  and the nightly Pre-Flight. That durable fix is deliberately not this round's; the
  owed-check pass is asked to split it into its own code item.
- **2026-10-19 — `current` again, one day late, fourth time the same class.** #102
  (`find-all-arms-run`, its own `created:` is `2026-10-18T07:42:00Z`) and #103
  (`code-rules-census-weekly`, `2026-10-19T01:40:00Z`) reached the directory with no
  row in either membership table, and the nightly skills-management Pre-Flight's
  `live_vault` rung went red on 2026-10-19 for exactly that: 179 collected / 176
  passed / 3 failed, each failing node reproducing alone, and `validate_tasks.py`
  printing four `DOC_MEMBERSHIP` warnings — one per table per id. #3012 named both
  ids in both tables. What moved with them: the group heading and §The functions'
  anchor grew to twelve ids, `Writes durable state unattended` went 26 → 27 taking
  #103, `Reports only` 14 → 15 taking #102, the numbering note's floor moved to
  `24 through 103 or beyond`, and the §Watch counts recomputed — ten became twelve,
  the `objective_checks` carriers eight became nine (#102 is the third file with no
  block, beside #91 and #92), the report-only six became seven. §The wiring's
  newest-file sentence moved off #101 onto #103 while its 29 and 39 stayed, for the
  reason #2813 and #2881 each give: that sentence counts the fleet that actually
  runs, and both new files sit `status: draft`. The tier split was taken here, at
  the rows, on the table's own question: #102 publishes a measurement table of its
  own under `eval/measurements/` and files nothing, so its write is its report;
  #103 drives `scripts/maintenance/code_rules_census.py`, which opens a backlog item
  itself whenever the census has hits, so it is the #92/#93/#99 shape and not a
  reporter's. The class this entry belongs to is unchanged by it: both membership
  nodes carry `@live_vault`, the gate runs `-m "not live_vault"`, and #2895's
  filing-time report is written never to refuse a task file —
  `tests/test_autonomy_task_membership_at_filing.py` pins that bound — so a filing
  whose report was read and not acted on redds these same three nodes again at the
  next task. Whether the report should ever withhold a filing is a decision no round
  can take for itself, and #3012 hands it to the owed-check pass with the next
  filing as the trigger.
