---
segment: architecture
tags: [architecture, lloyd, autonomy]
type: reference
status: implemented
date: 2026-09-11
---

# The autonomy jobs

What each scheduled task is *for*. [[autonomy]] is the mechanism — the five
due-gates, the failure ladder, the deadline anchor, the run records; this
document is the fleet it dispatches, one entry per job, grouped by the chain it
belongs to.

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
ordered: the fleet runs 24 through 85 with gaps where tasks were retired.

## The families

| Family | Jobs | What it produces |
|---|---|---|
| [Nightly reflection](#nightly-reflection-38--42--39--40-47) | #38 #42 #39 #40 #47 | durable changes to memory and config from the day's signals |
| [trace2skill](#trace2skill-56--57--58--83-70) | #56 #57 #58 #83 #70 | the skills library, mined from Lloyd's own session record |
| [Knowledge graph](#the-knowledge-graph-chain-24-48-67-74-51-60-84-82) | #24 #48 #67 #74 #51 #60 #84 #82 | facts, edges, and the measurements that say whether they got better |
| [Vault hygiene](#vault-hygiene-36-78-79-80-81) | #36 #78 #79 #80 #81 | structural health of the vault and its indexes |
| [Backlog](#backlog-35-77) | #35 #77 | the kanban at `~/obsidian/backlog/`, kept drained and tidy |
| [Inbound signal](#inbound-signal-68-30-53-54-65) | #68 #30 #53 #54 #65 | what the outside world did while nobody was looking |
| [Fleet self-watch](#fleet-self-watch-76-85) | #76 #85 | whether the fleet and the routing it depends on are working |

**Exactly one `depends_on` edge crosses a family boundary**: #51 (knowledge
graph) waits on #56 (trace2skill), because the trajectories #56 writes are the
same artifact #51 mines for co-access pairs. Every other edge is inside its own
chain, and 22 of the 32 tasks declare no dependency at all.

---

## Nightly reflection: #38 → #42 → #39 → #40 (+#47)

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
output contract plus a validator it runs against itself, and a hard rule to write
the handoff **by turn 40** and treat having written it as being done — added
2026-09-08 after three consecutive failures, one of which burned 61 turns and
2.5 M tokens with `tool_errors: 0`, doing entirely voluntary work. #38 carries
the same budget in **minutes**, because its turn limit was never what killed it:
Phase 2 dies at 1809–1826 s against a 1800 s cap while healthy runs take
181–639 s. Raising the ceiling is the wrong lever — the 600→1800 s raise on
2026-09-01 enlarged the burn rather than making overruns rarer.

**The windows are staggered, not shared, since 2026-09-09.** All four sat on
22–04 against `workers.slots: 2`, so #39 could be claimed in the hours #42 was
still writing the handoff it was about to read. The chain is now a queue in wall
clock as well as in `depends_on`, which is why it spans about four hours end to
end while the jobs themselves are minutes.

Two tool-shaped rules, both learned expensively. `Write` refuses to overwrite a
file it has not read this session, so a Phase 0 skeleton write is *refused on
every run* — `signals-latest.md` always exists from the night before. And the
vault tools reject anything under `~/lloyd/` with `PATH_ESCAPE`, so every
`_pipeline/` artifact in this chain is written with `Write` at an absolute path,
never `vault_write`.

**The claims a run makes are re-checked against disk.** The chain's most-repeated
defect was never a crash, it was a confident number: a handoff reporting the
graph "restored to 12,131 relationships" against a same-night health report
reading zero; counts of 96/21 where disk held 121/64; a 13,503-byte file called
"307KB". `workers/evidence.py` is the structural fix and its pilot set is exactly
this chain — `EVIDENCE_PILOT_TASK_IDS = {38, 42, 39, 40}`, a literal frozenset so
widening it is a change someone reads. The verifier is **stdlib-only and never
LLM-judged**, because a model grading its own claims is the narration this
replaces one layer up; and **a claim that cannot be evaluated is `insufficient`,
never `verified`**. Whatever failed to verify is carried into the next run of
that task through the queue's watermarks, not written back into the
human-edited task file.

**#47 Dream Consolidation** is the weekly synthesizer on the tail. It does not
re-read raw sessions — that is the chain's job — it merges near-duplicate topics
across the files reflection wrote, resolves contradictions, prunes stale entries
and keeps the MEMORY.md index tight. Gate-checked on ≥24 h since its last run and
≥3 new sessions, and lock-file protected.

### Two known holes

- **A paused upstream vacuously satisfies `depends_on`.** `_all_runnable_tasks`
  keeps `up_next`, `in_progress` and `failed`, and `_is_dependency_met` returns
  True when the dependency is not found — `failed` is in that set deliberately so
  a disabled upstream still blocks, but `paused` and `draft` are not, so either
  unblocks everything below it. Open as backlog **#558**.
- **An empty-but-successful run is indistinguishable from a dead one.** A run's
  status is decided by its terminal text, and the chain gates on status. Measured
  2026-09-11: `run_38_20260911_050055` wrote a complete signal report, was
  recorded `empty: true, status: failed` after 248 s with `stop_reason: stop` and
  19 turns, `last_run` did not advance, and only the 05:15 retry unblocked #42.
  Open as backlog **#832**.

### Jobs 3 and 4 do not exist

Earlier documentation, and #40's own data-load, referenced a prompt audit and a
behavior test writing `prompt-audit-issues.md`, `prompt-audit-latest.md`,
`test-failures.md` and `test-results-latest.md`. **No autonomy task has ever
written any of them.** `skills/nightly-prompt-audit/` and
`skills/nightly-behavior-test/` are on disk and no task file's `skill_name` names
either. #40's data load no longer reads them and says so inline, which stopped
every run rediscovering their absence and narrating it — but the rest of that
skill was never cleaned up, so `### Prompt Audit & Behavior Test Issues` still
instructs the job to treat their findings as signals and Phase 3 still asks for
those headings in the changelog. Expect them empty; that is the skill talking
about jobs that were never built, not a gap. `_pipeline/reflection/propagation-log.md`
is the same shape — #40's data load calls it "Job 2's propagation log" and
nothing has ever written it.

The numbering inside the skills never converged and is not worth trusting: the
four self-label "Job 1 of 3", "Job 2a of 4", "Job 2b of 4" and "Job 5 of 5".
There are four jobs.

---

## trace2skill: #56 → #57 → #58 → #83 (+#70)

Every interaction leaves a record, and that record contains procedural knowledge
— troubleshooting steps, corrections, behavioral rules, gotchas — that would
otherwise be lost between sessions. This chain surfaces it, packages it as
skills, and maintains the library.

| ID | Skill | What it does |
|----|-------|---|
| #56 | `trajectory-extraction` | tool-call trajectories from every session (worker + main) into one JSONL per day at `_pipeline/trajectories/<date>.jsonl`; watermark-gated and incremental |
| #57 | `trajectory-skill-mining` | 7-day window over those trajectories for error and success patterns, grouped into skill candidates with metadata. Installs **new** skill dirs only; defers existing ones to #83 |
| #58 | `nightly-skill-consolidation` | groups dated candidate snapshots, proposes patches to existing skills, flags new candidates; auto-applies at confidence ≥ 0.85 and sessions ≥ 5 |
| #83 | `nightly-skills-management` | the lifecycle pass: evaluate, create/update, review drafts, dedup the library, regenerate `memory/skills-index.md` |
| #70 | `skill-lint` | weekly advisory lint — `DEAD`, `MISSING_DESC`, `DRIFT`, `DUPLICATE`, `STALE`. Read-only, writes `skill-lint-report.{md,json}` for a human |

**#83's pre-flight is a git snapshot, because there is no other undo.** The pass
opens by running `scripts/util/vault-commit.sh "skills-mgmt: pre-run snapshot"`.
Nightly jobs commit straight to the vault's `main` — no worktree, no gate, no
guardian, unlike [[automod]] — so the commit taken *before* the run is the whole
of the rollback story.

**What it mines is the human half of the day, by construction.** Post-session
capture appends the daily-note summary only for a user session and routes
background exports to `_pipeline/vault-derived/sessions-background/`, so the ~240
background sessions a day this machine runs for itself are not in the daily notes
the pass falls back to. The trajectory JSONL is the wider corpus and does carry
them.

### Evidence integrity: three signals weaker than they look

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

### Authoring rules: why the generator was off for 68 days

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

**Two linters enforce rule 4 and both exempt this chain by name.**
`tests/test_skill_tool_names.py` (`ALLOWED_TO_MENTION`) and
`scripts/skill_lint.py` (`PHANTOM_EXEMPT`) let `nightly-skills-management`,
`trajectory-skill-mining` and `nightly-skill-consolidation` write the phantom
names down, because their job is to say those names are not real. The test's
`KNOWN_UNFIXED` debt ledger is **empty** — the 91 skills that carried a phantom
name have been rewritten or archived — and a new entry there is a regression, not
a grandfathering.

### Two traps in the review stage

- **Draft skills are selected on parsed frontmatter, never a body grep.** The
  `nightly-skills-management` skill's own body contains the literal string
  `status: draft` inside its instructions, so a body grep reports it as a draft on
  every single run: measured 2026-09-08, that grep returned exactly one file —
  itself — and there were zero drafts. Parsing also catches the quoted
  `status: "draft"` variant a string grep misses.
- **Flagging is not withdrawal.** A low-quality skill marked `status: needs-review`
  stays in circulation. `_QUARANTINE_STATUSES` is
  `{inactive, archived, disabled, retired, quarantined}` (`agent_mcp/skills.py`),
  and `prompt_builder._is_quarantined_skill` imports that same set precisely so
  the advertised index and the readable set cannot drift — so a needs-review skill
  keeps appearing in `<available_skills>` and keeps being retrievable until
  somebody acts on it.

**#70 owns the other half of "wrong skill".** When the body is fine and the
*description* is not, retrieval fires the skill on the wrong request or fails to
fire it on the right one; that is #70's `MISSING_DESC` and `DRIFT`, advisory only.
When the procedure itself has rotted, that is #83's Stage 3, which appends a
correction rather than rewriting.

**#83 runs unobserved, and that has a cost.** It sets no `inner_voice:` and the
fleet default is `false`. On 2026-09-11 the pass tried to land a vault change
through `automod_vault_land` and was refused, because that tool gates on Inner
Voice being attached and the observer attaches at turn start — switching it on
mid-turn cannot cover the turn.

---

## The knowledge graph chain: #24, #48, #67, #74, #51, #60, #84, #82

Ordered because each step's input is the previous step's output. #24 produces
entities and `mentions` edges; #48 and #67 settle canonical names; #74 types the
edges; #60, #84 and #82 measure the result.

| ID | Freq | Depends | Applies? | Role |
|----|------|---------|----------|------|
| #24 | 6×/day | — | yes | Content-hash-gated fact extraction over the `pipeline_config.yaml` corpus, then index rebuild and a `kg_health` snapshot. Emits graph edges as it writes facts; no-ops when no source document changed |
| #51 | daily | #56 | yes | Session co-access pairs from trajectories → `co_accessed` edges, provenance `INFERRED`, the trajectory as `source_doc` |
| #48 | daily | #24 | **no** | Name-shape clustering; `CASE`/`PUNCT` merge on shape, suffix pairs only on a unanimous definition-based verdict from the semantic gate. Reports a plan |
| #67 | weekly | — | **no** | LLM-judges the pairs string rules cannot — differently-spelled names, abbreviations, path variants. ~40 min/run. Writes `semantic-proposals-latest.jsonl` for #48's review list; strict guard rails downgrade questionable merges to alias-only |
| #74 | daily | #48 | yes | Re-types active `mentions` edges with the v4 classifier and applies confident upgrades through `app.kg_store`. Runs after #48 so canonical names are settled first |
| #60 | daily | — | no | Knowledge Health Report: god entities (>20 facts), thin (<2), orphans, stale facts, contamination, near-duplicates, provenance coverage |
| #84 | daily | — | **no** | Fact-quality pass: reads the corrections log and recent-write drift, pairs contradictions, reports which claims an independent reason condemns |
| #82 | daily | — | no | 20-query retrieval eval against production's recall defaults; records the trend |

**Nothing in this chain moves fact files unattended.** #48 is the one that could
— it is the sweep that merges — but it never passes `--apply`, so its scheduled
form reports a plan and stops. #67 proposes into a JSONL and stops too; its own
`--apply` was retired 2026-09-04. #84 runs in plan mode and writes nothing;
applying is an operator act. Only #51 and #74 write to the edge store outside
extraction. That split is deliberate: the 2026-08-22 wipe (12,131 edges) and the
2026-09-03 151-merge incident were both unattended applies.

**Two of these monitors exit 2 rather than report zero.** #60 exits 2 and alerts
when the store cannot be read, when active edges fall below half the baseline,
when any directory holds facts about another entity, or when any fact file
carries duplicate fact IDs. #84 exits 2 when the graph could not be read. The
rule both encode: *a monitor that reports success when it cannot see the thing it
monitors is worse than none* — the same failure `_is_dependency_met`'s
`if not dep_task: return True` and the evidence verifier's `insufficient` verdict
are each written against.

**#82 must pass no flags.** Until 2026-09-04 it ran `graph_rerank=False/alpha=0.5`
against a production serving `True/0.3` and scored a configuration nobody used.
Every record now carries `matches_production_defaults`, and a `false` there means
the run is not comparable to the others. It is ordered after #81's index
maintenance by window only — it declares no `depends_on`, so a late #81 does not
hold it. If a night's writes hurt recall, this is where it shows up.

**#51 produced no graph at all until 2026-09-04**, when approval only flipped a
status field in a JSON file nothing downstream read. It also ran every 15 minutes
until it was moved to nightly, where 95 of 96 daily runs were no-ops — the input
is written once a night by #56.

**One store.** Edges, aliases, the entity registry and the fact index live in
`_pipeline/vault-derived/kg.sqlite` behind `app.kg_store`, and nothing else opens
it. See [[knowledge-graph]].

---

## Vault hygiene: #36, #78, #79, #80, #81

| ID | Freq | Role |
|----|------|---|
| #36 | daily | Read the groundskeeper queue and report counts, health score, and whether the timer ran |
| #78 | weekly | Broken backlinks, unreferenced notes, stale skill references, broken cross-links → a cleanup report |
| #79 | weekly | Retention: delete `_pipeline/tasks` background-bash logs over 30 days, gzip session transcripts inactive 90+ days |
| #80 | weekly | OKF v0.1 conformance — `validate_okf.py` catches pages with no non-empty `type` or unparseable frontmatter |
| #81 | daily | Prune orphaned qmd embedding chunks, backfill pending embeddings |

**#81 is the one with a measured payoff**: orphaned chunks had grown the index to
24 GB and vec queries to 700 ms, and orphans displace real results, so it buys
both speed and accuracy. **#79** bounds the two unbounded-growth stores
identified in the 2026-06-11 architecture review; its gzipped archives stay
recoverable, but live consumers glob `*.json`, so archived sessions drop out of
listings and recall by design.

### #36 and the groundskeeper queue

The survey is the only vault-health number this machine produces, and it is
**a measurement with no consumer**.

`scripts/groundskeeper/groundskeeper-survey.py` walks the vault plus the
~63,000-file fact tree across 11 categories and writes
`_pipeline/groundskeeper-queue.json`. It runs at 02:30 under the
`lloyd-groundskeeper-survey.timer` systemd user unit and takes about 40 minutes.
Three roots, and the distinction matters for reading any item it emits:
`VAULT_ROOT` is `~/obsidian`, `FACTS_DIR` is
`~/lloyd/_pipeline/vault-derived/facts` — **outside the vault** — and `MEMORY_MD`
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
- **The write is atomic with a size check before the rename.** The queue goes to
  a `.tmp`, is read back, and its item count compared against what was intended;
  a mismatch deletes the temp file, leaves the previous queue in place, and
  raises `Queue corruption detected: size mismatch`. This exists because of the
  **2026-03-30 incident**: the fix loop processed four `BROKEN_LINK` items and
  wrote *only those four* back over a queue holding ~2,700, destroying the
  accumulated `done`/`skipped` status of everything else — which by the rule
  above is unrecoverable. The guard lives in the scanner rather than in the skill
  because a rule an LLM has to remember is not a rule.
- **79% of the queue names a path that does not exist.** `check_thin_profiles`
  writes `source_file: memory/facts/<entity>/`, vault-relative;
  `~/obsidian/memory/facts/` is gone and the fact tree moved to
  `~/lloyd/_pipeline/vault-derived/facts/`. That is 15,299 `THIN_PROFILE` plus
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

- **The orphan skipper is an unattributed daily write.** Every `ORPHAN_FILE` item
  in the queue — all 3,892 — reads `status: skipped`, reason
  `hub-page-linked-survey-bug`, stamped 2026-09-11T10:37:04Z, eighteen minutes
  after that night's survey finished. `_pipeline/groundskeeper-log.jsonl` has
  grown to 18,176 rows, every one an `ORPHAN_FILE` skip. That is
  `scripts/memory/process-groundskeeper-queue.py`, a one-off written when the
  survey had no hub-page awareness and every organised project file read as an
  orphan. The survey has had hub-page detection since; the premise is gone and
  the script is not. **What runs it nightly is unidentified** — no timer, no
  autonomy task, no worker source, no skill, and no session transcript contains
  the reason string. Until that is found the effect stands, and the idempotency
  rule makes it compound: a category worth 10% of the health score is permanently
  zeroed every night by a repair for a bug that was fixed.
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

`scripts/groundskeeper/retention-sweep.py` shares that directory and nothing
else — it is #79, a separate live job pinned by `tests/test_retention_sweep.py`.

---

## Backlog: #35, #77

| ID | Freq | Role |
|----|------|---|
| #35 | daily | Promote 2–4 high-priority inbox items into `up_next`; target queue size 8–12; skip when `up_next` is at capacity unless critical/high blockers exist |
| #77 | weekly | Archive stale and done tasks, clear draft clutter, reprioritize what remains |

These are the *human* backlog's maintenance. The self-modification loop's own
triage and implement passes are worker sources, not autonomy tasks, and run under
[[automod]]'s gates — see [[backlog]] for the board itself and the quarantine rule
that keeps the loop from eating what it files.

---

## Inbound signal: #68, #30, #53, #54, #65

What the outside world did while nobody was looking.

| ID | Freq | Role |
|----|------|---|
| #68 | every 15 min | Calendar + unread mail, filtered and classified, injected as ambient context |
| #30 | 3×/day | GitHub releases/issues and YouTube RSS → keyword pre-filter → scorer → high-scoring items to the vault |
| #53 | daily | Changelogs and release notes for the stack; compare each source's latest version against last-checked; only process actual changes |
| #54 | weekly | Take 2–3 knowledge areas and find non-obvious connections grounded in actual vault content — not brainstorming; read existing notes, identify gaps or bridges, write to `knowledge/synthesis/` |
| #65 | daily | Propose 5–8 research topics into `research.db` via `research_propose`, from the knowledge health report, session-distill gaps, open backlog and the last two daily notes |

**#65 feeds the `deep-research` worker and no longer reads or writes any queue
file** — [[research-pipeline]] is the registry it proposes into and the source
that drains it. The checklist it replaced held 2,839 items of which 314 were
unique and cost 1.02M tokens a night to read.

### #68 — the name outlived the job

`morning-briefing` names two things. One is #68, which runs every fifteen minutes
around the clock on the **secondary** engine and triages mail and calendar into
whatever chat session Alan is actually sitting in. The other — the 6 am "read the
overnight reports and announce a synthesis" job — no longer runs at all and no
autonomy task binds it. Its skill is archived; an unbound copy survives as
`~/obsidian/skills/nightly-morning-briefing/` still carrying "Runs at 6:00 AM PST
daily" and "Use announce mode", and **there is no announce delivery mode for
autonomy tasks anywhere in the code** — the only `announce()` in the tree is the
guardian's alert fan-out, a different mechanism for a different purpose. Its
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
calendar items and 5 emails survive. Each is classed `act-now` (an event inside
~3 hours, a deadline inside 48 hours, a personally addressed ask) or `fyi`, and
**a single `act-now` changes the routing of the whole signal**:

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
producer records a lost brief as delivered; and `/inject-prefetch` carries no such
gate, so the ambient path is protected at *resolution* rather than at the
endpoint, which is sufficient only because the producer lets the server resolve
the session.

**`[SILENT]` is what keeps a 15-minute job from being a 15-minute notification.**
Every task is told to answer with exactly `[SILENT]` and nothing else when there
is nothing new. Three checks, two semantics: the run record and
`discord_notify.py` both test for an **exact** `[SILENT]`, while the call site
above the latter skips the notification when `[SILENT]` appears **anywhere** in
the preview. So a response that merely mentions the token is still recorded as a
real run while its Discord post is dropped. The skill's own last line is written
against the equality reading, because the loose one cost a delivery: a paragraph
ending in `[SILENT]` was delivered to Alan in full, as noise (2026-09-03).

**Known defect — backlog #481.** Step 2 tells the agent to skip a missing calendar
source **silently** ("do not retry, do not treat it as an error"), which
contradicts the degraded-run attribution guardrail at the bottom of the same file
— a run whose source died must name it. Both instructions are live in one document
about 150 lines apart, and Step 3 has no missing-tool branch at all. The
2026-08-29 "succeeding but blind" run (0 emails against a normal 15–21, after a
~24 h IMAP/Thunderbird bridge gap) is the shape of failure that then reproduces by
instruction rather than by accident.

---

## Fleet self-watch: #76, #85

**#76 Queue Health Check** is the fleet's watchdog. It reads
`/api/autonomy/health` for per-task failure rate, timeouts, empty runs, `[SILENT]`
rate and GPU-hours; clears poisoned queue items after recording why; and pauses
any task with 3+ consecutive failures and a >50% fail rate. That endpoint reads
**`workers.db`**, not the per-task run records, because 237 pool-timeout rows with
a NULL `task_id` were unreachable from any per-task view by construction — a task
that timed out on every single run was indistinguishable from a healthy one.

**#85 Nightly Secondary Routing Eval** runs the paired primary-vs-secondary eval
over the five generation jobs routed to the secondary engine and records the
per-job routing decision, turning a routing choice made on throughput into one
checked nightly against a stated margin. It is the fleet's only `draft` task and
**it cannot run**: it pins `model: eco`, and `models:` in config.yaml defines only
`primary` and `secondary`.

---

## What the fleet actually costs

A snapshot, not a fact — the seven days to 2026-09-11, from `workers.db` where
`source = 'scheduled-task'`. Reproduce it rather than trusting the numbers; they
are here for the *shape* of the load, which is lopsided in a way no per-task view
reveals.

| Job | Runs | GPU-h | Note |
|---|---|---|---|
| #68 triage | 407 | 7.3 | the quarter-hourly job — the only run count in the hundreds |
| #24 data pipeline | 28 | 8.5 | the single most expensive job in the fleet |
| #74 mention classifier | 9 | 2.2 | cold-start backlog amortizes over days |
| #39, #42 | 7 each | 2.0 each | the reflection chain's two heavy halves |
| everything else | 1–20 each | ≤1.8 | the weeklies land at 0.0–0.6 |

**Two jobs carry 40% of the fleet's ~40 GPU-hours.** Three families — reflection,
trace2skill and the graph chain — account for most of the rest, and the weekly
hygiene jobs are nearly free. If a change has to make the fleet cheaper, #24 and
#68 are the only two places where it can matter.

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
and retired to `architecture/.archive/`. Their incident history is carried above;
what was dropped was the retired two-loop groundskeeper architecture, the
per-line source citations, and every restatement of a schedule that lives in the
task file.
