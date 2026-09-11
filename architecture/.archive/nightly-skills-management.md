---
segment: architecture
relations:
  related-to:
  - tools.md
  - architecture/voice.md
  - memory.md
  - architecture/autonomy-system.md
  - architecture/index.md
  - architecture/infrastructure.md
  - architecture/morning-briefing.md
  - architecture/nightly-reflection.md
  - skills.md
tags: [architecture]
type: reference

---















# Nightly Skills Management Architecture

An automated pipeline that mines Lloyd's own session record for reusable procedural
knowledge, creates and maintains a skills library, and self-corrects incorrect or
stale skills. It is **live**, as autonomy task **#83**.

**The durable half of this job is not in this repo.** The mechanism here is three
files of tracked code — the scheduler, the skill loader and the two linters — and
everything about *what the pass does* lives in the vault at
`~/obsidian/skills/nightly-skills-management/SKILL.md`, which the task loads as its
prompt. Fixing this job means editing that skill, which is a live shared tree with
no PR path: a saved file is a deploy, and the run before the next one is the one
that sees it.

## Schedule

- **Task:** `#83 Nightly Skills Management`, `~/obsidian/autonomy/83-nightly-skills-management.md`,
  `skill_name: nightly-skills-management`. `autonomy._load_skill_content` resolves a
  bare slug to `~/obsidian/skills/<slug>/SKILL.md` (`autonomy.py:568`) and refuses the
  run when it does not resolve.
- **Cadence:** `frequency: daily`, which is an **interval and not a clock time** —
  `_frequency_interval_seconds` maps it to 86400 s measured from the last *successful*
  run (`autonomy.py:319`). `scheduled_at` is empty and the task sets no
  `preferred_hours`, so the hour slides with whatever the chain upstream did: the
  2026-09-11 pass started 09:49 UTC (02:49 PDT).
- **Order:** `depends_on: 58` — it is the tail of the trace2skill chain,
  #56 extraction → #57 mining → #58 consolidation → #83. The gate wants the upstream
  fresh *within half this task's interval* rather than merely "since my last run"
  (`autonomy.py:382`); without that freshness rule the nightly chain settles into a
  stable inverted order in which every job consumes yesterday's artifact, which is
  what was observed in June 2026 when 57 ran before 56.
- **Model:** `model: primary` — the local Qwen3.8-Flash-Next on `:8096`
  (`config.yaml:472`). There is no dollar budget; the cost is GPU-hours.
- **Budget:** `timeout_seconds: 900`, enforced as `asyncio.timeout` around the turn and
  clamped to stay strictly under the caller's cap by `_POOL_TIMEOUT_MARGIN`
  (30 s, `autonomy.py:202`) so the pool timer cannot cancel the coroutine before its
  own handler writes a run record. The model is told the clock: `run_task` passes
  `state_anchor=_build_deadline_anchor(timeout)` (`autonomy.py:1009`), which fires once
  at 70% and once at 90%.
- **Skill file:** [`nightly-skills-management/SKILL.md`](../../obsidian/skills/nightly-skills-management/SKILL.md)
  (`~/obsidian/skills/nightly-skills-management/SKILL.md`)
- **Run records:** `~/lloyd/autonomy-runs/83/run_83_<stamp>.md` — `AUTONOMY_RUNS_DIR`
  is anchored to `LLOYD_HOME` (`app/paths.py:22`), so state follows the code. The
  activity-log line the task appends still spells that path vault-relative
  (`autonomy.py:783`), where it no longer resolves.

**Until 2026-09-03 this was a different task, and the header above said so.** The job
described here as running at 3:00 AM PST on the `memory` agent, on Claude Sonnet 4.6,
under a <$15 budget, was task **#37**. None of those four facts survives: the fleet
has run on the local primary with no dollar cost since the move off hosted models,
`agent_id: memory` persists as frontmatter that only `app/routers/autonomy.py` reads
back, and the run is an ordinary recorded autonomy session
(`create_session(..., platform="autonomy")`, `autonomy.py:918`) rather than an
isolated agent. #37 was **paused on 2026-06-27** because "generator authored/grew
dense bash-runbook skills that the primary model echoed instead of executing" — the
failure the authoring rules below exist to prevent — and **archived on 2026-09-03**
after 68 days in which its own `depends_on: 58` was also paused and the chain was
dead. It survives at `~/obsidian/autonomy/_archived/37-nightly-skills-management.md`.
The sequence this doc used to quote — `reflection-vault` at 2am, `reflection-synthesis`
at 1:30am, `nightly-reflection` at 4am — named tasks that no longer exist anywhere in
the fleet; the reflection chain is #38/#42/#39/#40 and is windowed by `preferred_hours`,
not by clock times (see [[nightly-reflection]]).

## Overview

Every interaction between Alan and Lloyd leaves a record, and that record contains
procedural knowledge -- troubleshooting steps, corrections, behavioral rules, gotchas
-- that would otherwise be lost between sessions. The skills management pipeline
surfaces this knowledge, packages it as reusable skills, and maintains the library
over time.

**The record is no longer JSONL session files.** A session is one JSON document at
`~/lloyd/sessions/<session_key>.json`; the JSONL in this pipeline is the *trajectory*
format, one file per day at `~/lloyd/_pipeline/trajectories/<date>.jsonl`, written by
#56 from those sessions. The pass reads the trajectory for the day when it exists and
the vault's daily note (`~/obsidian/memory/<date>.md`) otherwise.

**What it mines is the human half of the day, by construction.** Post-session capture
appends the daily-note summary only for a user session (`is_user_session`,
`app/post_capture.py:347`) and routes background exports to
`_pipeline/vault-derived/sessions-background/`, so the ~240 background sessions a day
this machine runs for itself are not in the daily notes the pass falls back to. The
trajectory JSONL is the wider corpus and does carry them.

This is a key component of Lloyd's self-improvement architecture, alongside [[nightly-reflection]] (mental models, MEMORY.md consolidation, config improvements) and [[groundskeeper]] (structural hygiene).

## Five stages and a pre-flight

```
Trajectories / daily notes            Skills Library
~/lloyd/_pipeline/trajectories/       ~/obsidian/skills/
~/obsidian/memory/<date>.md                  ^
         |                                   |
         v                                   |
+---------------------+          +-----------+----------+
|  Stage 1: Read      |          |  Stage 5: Dedup &    |
|  yesterday's        |          |  Health Check        |
|  trajectory JSONL,  |          |  - list every skill  |
|  else the daily     |          |  - archive to        |
|  note               |          |    .archived/        |
+--------+------------+          |  - rewrite           |
         |                       |    skills-index.md   |
         v                       +-----------^----------+
+---------------------+                      |
|  Stage 2: Evaluate  |          +-----------+----------+
|  signal detection,  |--------->|  Stage 3: Create /   |
|  errors first,      |          |  Update Skills       |
|  evidence re-checked|          |  append, never       |
|  before it counts   |          |  overwrite           |
+---------------------+          +-----------+----------+
                                             |
                                             v
                                 +----------------------+
                                 |  Stage 4: Review     |
                                 |  draft skills        |
                                 |  - promote to active |
                                 |  - merge duplicates  |
                                 |  - needs-review      |
                                 +----------------------+
```

**Pre-flight: a git snapshot, because there is no other undo.** The pass opens by
running `~/lloyd/scripts/util/vault-commit.sh "skills-mgmt: pre-run snapshot"`. Nightly
jobs commit straight to the vault's `main` — there is no worktree, no gate and no
guardian on this path, unlike [[automod]] — so the commit taken *before* the run is
the whole of the rollback story. The 2026-09-11 run recorded both hashes in its report
(`vault 9ce57dc0`, pre-flight `d0692ce4`).

### Data Flow

1. **Source**: the day's trajectory JSONL at `~/lloyd/_pipeline/trajectories/<date>.jsonl`
   (written nightly by #56), falling back to the vault daily note
   `~/obsidian/memory/<date>.md`. Each trajectory row carries `session_key`,
   `tool_count`, `error_count`, `has_errors` and a per-tool `params_summary`; the
   underlying transcript is `~/lloyd/sessions/<session_key>.json`.
2. **Evaluation**: each session is assessed against the signal detection criteria below,
   error trajectories first, with every error claim re-checked against the session JSON
   before it is allowed to count.
3. **Authoring**: new or updated skills at `~/obsidian/skills/<name>/SKILL.md` — appended
   to, never overwritten.
4. **Output**: `~/obsidian/memory/skills-index.md` (regenerated: 189 skills at the
   2026-09-10 pass, against 191 directories on disk) and the append-only usage log
   `~/obsidian/memory/skills-usage.jsonl`. Archived skills move to
   `~/obsidian/skills/.archived/`, 193 of them today, and are not indexed.

**Until 2026-09-03 this section described a pipeline that no longer has a single live
part.** The source was `~/.openclaw/agents/main/sessions/`,
`~/.openclaw/agents/memory/sessions/` and `~/.openclaw/logs/cc-instances/`; none of
those paths exists. The extractor was `extract-session-log.py --hours 24`, deleted —
`app/post_capture.py:191` now describes its own writer as matching "the old Hermes
extract-session-log.py output for consistency", which is the only trace of it left. It
wrote one `.log` per session into `~/obsidian/memory/skill-maintenance/<date>/` with a
`report.md` beside it; that directory has never existed under the current vault. And
`~/obsidian/sessions/`, which the live skill's Stage 1 says was removed, is still
present as an empty directory dated 2026-08-22 — empty, so the skill's conclusion
holds even though its wording does not. Stage 3 was "Dedup & Consolidate" and Stage 4
was "Effectiveness Tracking", writing effectiveness metrics; the live skill's Stage 4
is a draft review and its Stage 5 is the dedup, and nothing in this pass writes an
effectiveness metric today.

## Signal Detection Criteria

The evaluation stage looks for six categories of skill-worthy patterns. The list below
is this doc's original wording; the live skill groups the same ground slightly
differently -- it folds *behavioral rules* and *troubleshooting playbooks* back into
failure-to-fix chains, and it carries two categories this doc never had: **recurring
multi-step workflows** (the same sequence of actions repeated across sessions) and
**tool usage patterns** (diagnostic sequences that consistently work). It also treats
**error pattern clustering** -- the same error type across several sessions -- as the
highest-value candidate of all, which is what the priority ordering below serves.

### 1. Corrections
Alan corrected Lloyd's behavior or approach. Extract as a guardrail -- update an existing skill with a warning, or create a new skill with the correct procedure.

### 2. Remember Requests
Alan explicitly asked to preserve a procedure ("from now on","always do X"). Create a skill capturing the procedure exactly.

### 3. Failure-to-Fix Chains
Something broke, investigation found a non-obvious root cause after multiple steps. Extract the diagnostic path and fix as a troubleshooting skill.

### 4. Behavioral Rules
Corrections that establish ongoing patterns ("never do X","always Y"). Extract as a guardrail skill or update an existing skill's constraints.

### 5. Troubleshooting Playbooks
Failure-to-fix chains where the root cause was not obvious from the symptom -- even one-time incidents. Extract as: symptom, investigation steps, root cause, fix.

### 6. Stale or Incorrect Skill Steps
Evidence that an existing skill's instructions are wrong or outdated. Two failure modes:
- **(A) Wrong skill selected**: the body is fine and the *description* is not, so
  retrieval fires the skill on the wrong request or fails to fire it on the right one.
  This half now belongs to task **#70 Skill Lint Sweep** (`skill_name: skill-lint`,
  weekly, advisory only), which flags `MISSING_DESC` and `DRIFT` -- a description that
  describes output instead of trigger conditions -- and writes
  `skill-lint-report.{md,json}` into `~/obsidian/autonomy/` for a human to read. It
  applies no fixes.
- **(B) Right skill, wrong steps**: the procedure itself has rotted. That is this
  pass's Stage 3, which appends a correction to the existing skill rather than
  rewriting it.

*(The sentence above was cut off mid-word -- "Fix descrip" -- in the original import of
this doc, in the very first commit that carried it. The completion is reconstructed
from where the live system puts the two halves, not from the lost text.)*

## Evidence integrity: three signals that are weaker than they look

Added to the skill on 2026-09-06 after measurement, and the most load-bearing thing in
it. A mining pass that trusts its inputs manufactures skills from incidents that never
happened, and a skill is worse than no skill: it is retrieved, obeyed, and wrong.

- **`has_errors` / `error_tools` is a text heuristic, not an error flag.** The
  extractor matches `Error:`, `Traceback`, `No such file or directory` against the
  *result text*, so reading source code that mentions an exception counts as a failure.
  On the 2026-09-05 trajectories: 3 of 3 sessions flagged and 30 `error_tools` entries,
  against 6 failures in the `stats.is_error` field of the underlying session JSON --
  roughly two thirds of them `Read`/`Grep` output quoting code.
- **`[INNER VOICE]` messages are machine output, not Alan's corrections.** The
  repetition guard asserts "the result has not changed" from argument similarity alone;
  its signature carries the tool name, the normalised arguments and the identifiers,
  and there is no result field on it at all. On 2026-09-05 it fired 5 times while a
  result-level comparison found no unchanged-result repeat in the two sessions behind 4
  of those injections. A false fire logged on the corrections axis becomes a false
  skill, which is why this is pinned from the *other* side in tracked code:
  `tests/integration/test_iv_repetition_wording.py` bans the unobservable claim in the
  inject wording and names #83 Stage 2 as the cost.
- **`stats.is_error` over-counts too.** Measured on the 2026-09-07 trajectories (39
  sessions): the heuristic flagged 128 tool calls, the session JSONs marked 54 messages,
  and reading the payloads left **13** genuine failures. Of the 54, 32 were only a
  non-zero exit code and 9 carried no error text at all -- each truncated at exactly
  2014 characters, so whatever tripped the marker sits past the stored cap and cannot be
  checked. Those are unverifiable and get dropped, not counted. A useful failure is one
  the tool itself framed as an error, or a deliberate block or cancellation; never
  merely "the shell returned non-zero".

## Authoring rules: why the generator was switched off for 68 days

Task #37 was paused on 2026-06-27 for one reason, recorded in its own `paused_reason`:
the generator "authored/grew dense bash-runbook skills that the primary model echoed
instead of executing". The primary treats fenced shell blocks in a retrieved skill as
few-shot "print this" demonstrations, so a command-dense skill manufactures that
failure on every retrieval. The pass was not re-enabled until the skill could state the
rules that prevent it, which it now does as **Authoring Output Rules (CRITICAL)**:

1. **No fenced executable blocks** in a generated skill body -- prose and inline `code`
   spans only.
2. **Scoped "When to Use", never a bare tool name** -- trigger on a specific error
   signature, since "when using the Bash tool" matches every routine request.
3. **Bounded length**, target under 80 lines; consolidate old examples rather than
   stacking new ones past ~120.
4. **Only real tool names**, verified before promotion by running
   `tests/test_skill_tool_names.py`. A skill naming a tool that does not exist is
   actively harmful: the model calls it, gets an unknown-tool error, and takes whatever
   fallback the skill listed. That is how `web_search`/`WebSearch` -- names Lloyd has
   never had -- taught it to shell out to `curl` for every web lookup on 2026-09-04. The
   web tools are `http_search`, `http_fetch`, `http_request`.
5. **Never restate a tool's parameter contract.** Ranges, defaults and enum values
   belong to the advertised schema; a copy drifts silently. Eleven skills were archived
   on 2026-09-04 for documenting an `extract_mode` retry chain, a `max_chars` floor and
   a default that were all wrong.
6. **No tool-failure skills from pre-2026-09-04 evidence.** Before that date no tool set
   `isError`, so every failure arrived as a success whose text contained
   `{"error": ...}`. Patterns mined from older sessions describe a signalling bug that
   has since been fixed.

**Two linters in tracked code enforce rule 4, and both exempt this skill by name.**
`tests/test_skill_tool_names.py` (`ALLOWED_TO_MENTION`) and `scripts/skill_lint.py`
(`PHANTOM_EXEMPT`) let `nightly-skills-management`, `trajectory-skill-mining` and
`nightly-skill-consolidation` write the phantom names down, because their job is to say
those names are not real. The test's `KNOWN_UNFIXED` debt ledger is **empty** -- the 91
skills that carried a phantom name have been rewritten or archived -- and a new entry
there is a regression, not a grandfathering.

**The Opus routing paragraph is a documented no-op.** The skill says authoring should
be dispatched to an Opus-tier subagent, then says to check `subagents:` in config.yaml
for one: it holds only `general-purpose` and `read-only`, both inheriting the calling
turn's model, so there is no Opus target and the executing agent authors the skill
itself. The instruction is to record that substitution in the usage log
(`authoring_route`) rather than drop the work or claim it was routed -- which is the
right shape for a capability that does not exist yet, and the reason the paragraph is
kept rather than deleted.

## Stage 4: the grep that always found itself

Draft skills are selected on the **parsed frontmatter** `status` key, not by grepping
the file. This skill's own body contains the literal string `status: draft` inside its
instructions, so a body grep reports it as a draft on every single run: measured on
2026-09-08, that grep returned exactly one file -- this one -- and there were zero
drafts. Parsing also catches the quoted `status: "draft"` variant a string grep misses.
A hit that turns out to be body text is a false positive to be logged, never a skill to
be promoted.

A draft that passes review becomes `status: active`; a duplicate is merged and deleted;
a low-quality one is marked `status: needs-review` and left for a human. **That last
state does not take the skill out of circulation.** `_QUARANTINE_STATUSES` is
`{inactive, archived, disabled, retired, quarantined}` (`agent_mcp/skills.py:38`), and
`prompt_builder._is_quarantined_skill` imports that same set precisely so the advertised
index and the readable set cannot drift apart -- so a skill this pass flags as
needs-review keeps appearing in `<available_skills>` and keeps being retrievable until
somebody acts on it. Flagging is not withdrawal.

## What it costs, and what watches it

- **It runs unobserved.** #83 sets no `inner_voice:` frontmatter and the fleet default
  `autonomy.inner_voice` is `false`, so there is no observer on the turn. That is a
  deliberate fleet-wide default -- the observer runs on the primary at priority 1 in
  front of every chat turn -- but it has a cost here: on 2026-09-11 the pass tried to
  land a vault change through `automod_vault_land` and was refused, because that tool
  gates on Inner Voice being attached (`agent_mcp/automod.py:109`, called at `:584`)
  and it attaches at turn start, so switching it on mid-turn cannot cover the turn.
- **The budget is real and has killed runs.** Against the 900 s cap, successes land
  between 402 s and 556 s. Three runs died at `max_turns` (61 turns) recording an empty
  response -- 2026-09-06 at 544 s, and twice on 2026-09-08 at 502 s and 568 s. Every
  successful run in the last week also carried 1-5 tool errors, which the run record
  reproduces verbatim.
- **Every run leaves a transcript.** The turn runs in a real recorded session
  (`platform="autonomy"`), so the pass is readable after the fact in the Background tab
  rather than only as a summary line -- see [[background-runs]].
