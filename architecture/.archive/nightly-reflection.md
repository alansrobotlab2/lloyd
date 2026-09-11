---
segment: architecture
relations:
  related-to: []
tags: [architecture]
type: reference

---

# Nightly Reflection Architecture

**Last updated:** 2026-09-11 (re-audited against the tree; the windows, the
`stale_bypass_hours` placement and #39's write list had all moved.) Rewritten
2026-09-03 from audited reality — the version before that was truncated after
Job 1, described five jobs at fixed clock times, budgeted "<$25/night" against
Claude Opus, and listed handoff files that have never existed.

The nightly chain turns the day's signals into durable changes: what Alan
corrected, what the system learned, and what should therefore change in memory
and config. It is **live** — all four jobs ran on 2026-09-11.

## Where the chain lives, and why it is in two halves

Each job is an autonomy task file (`~/obsidian/autonomy/<id>-<name>.md`) whose
frontmatter holds the schedule and whose `skill_name:` names a vault skill at
`~/obsidian/skills/<skill_name>/SKILL.md`. **That skill is the job** — the
phases, the output contract, the incident history, the standing non-findings.
It is rewritable by the chain itself, which is the point: #40 edits skills and
task frontmatter nightly, and those edits take effect on the next run with no
deploy.

The tracked half in `~/lloyd` is everything a prompt cannot enforce:
`autonomy.py` (the due/dependency/backoff gates, and the prompt assembly),
`workers/sources/scheduled_task.py` (which enqueues, delegating due-ness to
`autonomy._is_task_due`), `scripts/validate_handoff.py` (the handoff's
structural validator) and `workers/evidence.py` (the claim verifier — see
Guardrails). Changing *what a job does* is a vault edit; changing *what the
scheduler will let it do* is a code change.

Nightly jobs commit straight to live `main` in both trees. There are no
`nightly-improvement-*` branches and none are left behind — see Guardrails.

## The chain as it actually runs

```
#38 signals ──► #42 analysis ──► #39 knowledge write ──► #40 config
     22-04         22-04              01-04                 02-04

#56 trajectory extraction (01-02, independent)
     ├─► #57 mining (23-04)
     └─► #51 relation linking (23-04)
```

All four reflection jobs run on the **local** model (`primary`), not Opus. There
is no dollar budget; the cost is GPU-hours.

| ID | Skill | Reads | Writes |
|----|-----|-------|--------|
| #38 | `nightly-reflection-signals` | `memory/corrections.md`, last 3 daily notes, that date range's session data | `_pipeline/reflection/signals-latest.md` |
| #42 | `nightly-reflection-knowledge-analysis` | signals-latest.md, last 7 daily notes, `memory/mental-models.md`, `lloyd/MEMORY.md`, `lloyd/USER.md` | `_pipeline/reflection/knowledge-handoff-<date>.md` (validated by `scripts/validate_handoff.py`) |
| #39 | `nightly-reflection-knowledge-write` | the handoff (newest by mtime, refused if older than 24 h) | `lloyd/USER.md`, `lloyd/MEMORY.md`, `memory/mental-models.md`, `people/alan/profile.md`, `tool-patterns-latest.md`, `conversation-patterns-latest.md`, completion note, vault commit |
| #40 | `nightly-reflection-config` | signals-latest.md, `lloyd/MEMORY.md` after #39 | `~/lloyd/config.yaml`, vault surfaces (SOUL/skills/autonomy frontmatter), `memory/learnings/<date>.md`, commits on `main` |

**#39's first target is `lloyd/USER.md`**, the user-facts file the runtime loads
into every system prompt — not `agents/`, retired 2026-07-14. Naming only
`mental-models.md` and the profile, as this table did until 2026-09-11, hides
the write with the largest blast radius in the chain: on 2026-09-11 that job
re-applied 44 curated lines to `lloyd/MEMORY.md` that an unrelated job's
pre-flight commit had clobbered the night before.

Two artifacts named by the skills do not exist and should not be waited on:
`_pipeline/reflection/propagation-log.md` (#40's data load calls it "Job 2's
propagation log", nothing has ever written it) and, since 2026-09-11, the
retired single Job 2 — `skills/nightly-reflection-knowledge`, the umbrella that
#42 and #39 replaced when analysis was split from write, now under
`skills/.archived/`.

**Jobs 3 and 4 do not exist.** Earlier docs and #40's own data-load referenced a
prompt audit and a behavior test writing `prompt-audit-issues.md`,
`prompt-audit-latest.md`, `test-failures.md` and `test-results-latest.md`. No
autonomy task has ever written any of them: `skills/nightly-prompt-audit/` and
`skills/nightly-behavior-test/` are on disk, and no task file's `skill_name`
names either. #40's **data load** no longer reads them and says so inline, which
stopped every run rediscovering their absence and narrating it — but the rest of
that skill was never cleaned up, so `### Prompt Audit & Behavior Test Issues`
still instructs the job to treat their findings as signals, and Phase 3 still
asks for `### Prompt audit findings` and `### Behavior test results` in the
changelog. Expect those two headings to come back empty; that is the skill
talking about jobs that were never built, not a gap.

The numbering in the skills themselves never converged and is not worth
trusting: the four self-label as "Job 1 of 3", "Job 2a of 4", "Job 2b of 4" and
"Job 5 of 5". There are four jobs.

## Scheduling

Windows are `preferred_hours` in machine-local time (PDT), not fixed clock
times, and each job also waits on `depends_on`. Before 2026-09-03 only #38 and
#42 had windows, so any upstream slip could push #39/#40 into the middle of the
next day.

**The windows are staggered, not shared, since 2026-09-09.** All four sat on
23-04 against `workers.slots: 2`, so #39 could be claimed in the hours #42 was
still writing the handoff it was about to read. #40 narrowed #39 to 01-04 and
itself to 02-04 that night, and the shape of the chain is now a queue in wall
clock as well as in `depends_on`. That is also why it is no longer a three-hour
chain: 2026-09-10 ran 05:31Z → 09:05Z and 2026-09-11 ran 05:00Z → 09:06Z, about
four hours, most of it the deliberate gap between #42 finishing around 05:30Z
and #39's window opening at 08:00Z. The jobs themselves are minutes — 181 s,
307 s, 584 s, 291 s on 09-10.

A window only binds if it is written down: `_effective_preferred_hours`
(`autonomy.py:400`) falls back to the hour in `scheduled_at`, and a task with
neither runs whenever else it is due. `_is_task_due` also grants up to an hour
of slack against the frequency interval when a window is in force, because
`last_run` is a *completion* time and the due moment otherwise drifts later by
each run's own duration until it steps past a narrow window and skips a day.

Two timeout caps apply and the smaller wins: the task's own `timeout_seconds`
(1800 s for #38/#42/#40, 2400 s for #39) and the source cap,
`workers.sources.scheduled-task.max_duration_seconds: 3600`. `run_task` derives
its own deadline as the source cap minus 30 s so its handler always wins the
race — when the two were equal the pool's timer cancelled `run_task` before it
could write anything, and 237 such runs left no record at all.

## The output contract (why this chain kept producing nothing)

Every job in this chain **claims its output file in its first turns**, then
enriches it in place, and flips a `status:` field to `complete` as its last act.

This is not stylistic. These jobs used to investigate exhaustively and write at
the very end, so a run that hit `max_turns` produced *nothing* while still
reporting success. On 2026-09-03 a 1429-second #38 run left `signals-latest.md`
untouched from two days earlier, and #42, #39 and #40 all consumed that stale
file believing it was current. #39 separately did its real writes and then died
during bookkeeping, so `mental-models.md` was genuinely updated while the record
said nothing had happened.

The recurring failure mode is looping on "one last verification pass" until the
turn limit kills the run. #42 is the reference implementation: a machine-checkable
output contract plus a validator it runs against itself, and a hard rule to write
the handoff **by turn 40** and treat having written it as being done — added
2026-09-08 after three consecutive failures, one of which burned 61 turns and
2.5 M tokens with `tool_errors: 0`, doing entirely voluntary work.

#38 carries the same budget in **minutes**, because its turn limit was never what
killed it: Phase 2 is where the run dies, at 1809–1826 s against the 1800 s cap,
while the six most recent healthy runs took 181–639 s. So it stops opening new
investigation threads at minute 12 and files the rest as `Unverified`. Raising the
ceiling is the wrong lever — the 600→1800 s raise on 2026-09-01 enlarged the burn
rather than making overruns rarer.

Two tool-shaped rules sit in the same contract, and both were learned the
expensive way. `Write` refuses to overwrite a file it has not read this session,
so the Phase 0 skeleton write was *refused on every run that followed it
literally* — `signals-latest.md` always exists from the night before. And the
vault tools reject anything under `~/lloyd/` with `PATH_ESCAPE`, so every
`_pipeline/` artifact in this chain is written with `Write` at an absolute path,
never `vault_write`; the runs that survived the old wording did so because the
model noticed and switched tools on its own.

A related trap: `compaction.microcompact` clears the oldest tool results mid-turn
once the conversation passes `trigger_fraction` (0.72) of the truncation
threshold, spilling each to disk so the marker can name the file. A job that
re-reads what it already read can loop indefinitely against that. If a reflection
job reports "my earlier reads were cleared from context", that is this.

## Guardrails

- **Commit on `main`.** #40 commits config changes directly; it must start and
  end on `main` and merges any leftover `nightly-improvement-*` branch first.
  An earlier guardrail said the opposite and stranded 8+ commits on an unmerged
  branch.
- **Empty output is a failure**, recorded as one, with backoff. It used to be
  recorded as success and unblock the rest of the chain — that is how ~180
  phantom runs passed during the 2026-09-01 empty window. Backoff is
  exponential from 10 minutes, doubling per consecutive failure, capped at the
  task's own interval or 6 h; the cooldown is read as "`last_attempt` is newer
  than `last_run`".
- **A failed upstream does not satisfy `depends_on`** — the gate reads
  `last_run`, which only successes set. A *paused* upstream is the hole:
  `_all_runnable_tasks` (`autonomy.py:258`) keeps only `up_next`,
  `in_progress` and `failed`, and `_is_dependency_met` returns True when the
  dependency is not found, so a paused or `draft` upstream vacuously unblocks
  everything below it. `failed` is in that set deliberately, for exactly this
  reason; `paused` is not, and the code half is open as backlog #558. The
  config half shipped as the 09-09 stagger above.
- **Yesterday's upstream run does not count as today's.** `_is_dependency_met`
  applies an `interval / 2` freshness test (`autonomy.py:384`) rather than
  "has it run since I last ran", which was satisfied by a day-old artifact and
  let the chain settle into a stable inverted order — observed June 2026 as
  39→38/40→42, and 57 before 56.
- `stale_bypass_hours` lets a downstream job run on stale input rather than
  block the chain forever, and is honoured only when the upstream is not
  currently `in_progress`, so a merely-late upstream is still waited for.
  **It is read on the dependent, not the upstream** (`_dependency_bypassed`,
  `autonomy.py:336`), which is a one-line hole this doc helped keep open: only
  #38 and #40 carried it, and #38 is the chain's root with no `depends_on`, so
  the value sat where nothing consulted it while the two edges that actually
  decide the chain — #38→#42 and #42→#39 — had `bypass_hours = 0` and could
  only answer False. #42 and #39 were given 36 on 2026-09-11. All four carry
  it now.
- **A run's status is decided by its terminal text, and the chain gates on
  status.** An empty final message is recorded `failed` however much the run
  actually did, `last_run` does not advance, and the dependent sees an upstream
  that worked as one that died. Measured on 2026-09-11:
  `run_38_20260911_050055` wrote a complete signal report, was recorded
  `empty: true, status: failed` after 248 s with `stop_reason: stop` and 19
  turns, and only the 05:15 retry unblocked #42. Open as backlog #832; until it
  closes, an empty-but-successful run is indistinguishable from a dead one.

### The claims a run makes are re-checked against disk

The chain's most-repeated defect was never a crash, it was a confident number:
the 08-24 handoff reporting the entity graph "restored to 12,131 relationships"
against a same-night health report reading zero, the 09-03 counts of 96/21 where
disk held 121/64, the 09-04 `signals-latest.md` called "307KB" when it was 13,503
bytes. The fix was procedural for months — a skill line saying *verify on disk
before claiming* — which binds only a willing model.

`workers/evidence.py` is the structural version, and its pilot set is exactly
this chain: `EVIDENCE_PILOT_TASK_IDS = {38, 42, 39, 40}`, a literal frozenset in
`autonomy.py` rather than a config key, so widening it is a change someone reads.
A pilot run's prompt gets a claims instruction appended at the **tail** (the
prefix stays cached), the run ends with `{claim, check}` pairs in an `evidence`
fence, and `pool.py` verifies the bundle against the filesystem at the moment the
ledger row is written. Two rules carry it: the verifier is **stdlib-only and
never LLM-judged** — a model grading its own claims is the narration this
replaces, one layer up — and **a claim that cannot be evaluated is
`insufficient`, never `verified`**, because a gate that reads its own missing
input as a pass is the failure recorded three times over in this tree, including
`_is_dependency_met`'s own `if not dep_task: return True` two bullets above.
Whatever failed to verify is carried into the next run of that same task, held in
the queue's watermarks rather than written back into the human-edited task file.
`tests/test_worker_evidence.py` pins it.

## Verification

Retrieval quality is measured, not assumed: task #82 runs the fixed 20-query eval
(`eval/vault_recall_queries.yaml`, via `eval/run_eval.py`) in the 06:00 hour and
records the trend, after #81's index maintenance in the 05:00 hour. The ordering
is by window only — #82 declares no `depends_on`, so a late #81 does not hold it.
If a night's writes hurt recall, that is where it shows up.

Pass no flags: until 2026-09-04 that eval ran `graph_rerank=False/alpha=0.5`
against a production serving `True/0.3` and scored a configuration nobody used.
Every record now carries `matches_production_defaults`, and a `false` there means
the run is not comparable to the others.
