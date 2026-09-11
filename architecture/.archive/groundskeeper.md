---
segment: architecture
tags: [architecture]
type: reference
relations:
  related-to:
  - architecture/index.md
  - architecture/autonomy-system.md
  - architecture/memory.md

---

# Groundskeeper System Architecture

**Created:** 2026-03-25  
**Status:** the scanner is live; both consuming loops are retired.

**Read this first.** The half of the system that *measures* runs every night
and is healthy. The half that *acted* on the measurements is gone. The scanner
still walks the vault at 02:30 under `lloyd-groundskeeper-survey.timer` and
still writes `~/lloyd/_pipeline/groundskeeper-queue.json` — 31,291 items and a
health score of 58.4 as of the 2026-09-11 03:19 run. Autonomy tasks **#33
(Groundskeeper Fix Loop)** and **#34 (Groundskeeper Research)**, which drained
that queue, no longer exist: there is no `33-*.md` or `34-*.md` in
`~/obsidian/autonomy/` nor in its `_archived/`, no live task names the
`groundskeeper-loop` or `groundskeeper-research` skills, and
`~/lloyd/autonomy-runs/33/` and `/34/` are empty directories. So the queue is
now a **measurement with no consumer** — a standing report on vault health
that nothing reads back. The body below is kept as the record of what the
system was and of what the scanner still does, because the scanner's output is
still the only vault-health number this machine produces.

## Overview

The Groundskeeper is a vault health and enrichment system that scans the
Obsidian knowledge vault for issues and enrichment opportunities. It evolved
from the "Ralph Wiggum" survey scanner and absorbed the Vault Maintenance task
(#32) into a unified two-loop architecture:

1. **Groundskeeper Fix Loop** — Automated mechanical repairs (autonomy task #33) — *retired*
2. **Groundskeeper Research** — Deep web research for enrichment (autonomy task #34) — *retired*

Both loops shared a single scanner script and queue file, with the autonomy
system dispatching them on different schedules: #33 every 15 minutes taking up
to 25 pending items by priority, #34 hourly taking exactly one enrichment item.
(The two numbers in the diagram below said 5 and 1; the surviving skill files
say 25 and 1 — `skills/groundskeeper-loop/SKILL.md` slices `items[:25]`.)

**Until 2026-09-03 the survey ran inside autonomy task #36**, and that is the
incident the present shape exists because of. The scan walks the whole vault
plus a ~63,000-file facts tree and takes about 40 minutes, which is longer than
any agent tool call can wait: the Bash timeout killed it at 2–5 minutes, the
agent backgrounded it and burned the rest of its budget polling, and the task
timed out roughly **70 times a week (~12 GPU-hours)** — while each dead run
left an orphaned 40-minute scan still running and competing to overwrite the
queue file. Task #36 was restructured that day into *read the queue and
report*: `timeout_seconds` 900 → **180**, `frequency: daily`, skill
`groundskeeper-survey`, and runs now take 9–70 seconds. Nothing in the survey
needs an LLM, which is why it belongs on a timer and not in a turn. The task
file and the skill both say **do not run the survey here**, and the reason is
written into `lloyd-groundskeeper-survey.service` as a comment so it survives
the next person who wonders why a timer runs a script an agent could call.

## Architecture Diagram

Live today (solid), retired (marked):

```
       lloyd-groundskeeper-survey.timer  (02:30 nightly, ~40 min)
                              |
                              v
                    +---------+---------+
                    |  Groundskeeper    |
                    |  Survey Script    |
                    |  (groundskeeper-  |
                    |   survey.py)      |
                    +---------+---------+
                              |
                              v
                    +---------+---------+          +--------------------+
                    |  Queue JSON File  |<-------->|  Task #36          |
                    |  _pipeline/       |  reads   |  read-and-report   |
                    |  groundskeeper-   |  only    |  daily, 180s       |
                    |  queue.json       |          +--------------------+
                    +---------+---------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
+-------------+-------------+     +-----------+-----------+
|   Fix Loop      RETIRED   |     |  Research Loop RETIRED|
|  (Task #33)               |     |  (Task #34)           |
|  Every 15 minutes         |     |  Hourly               |
|  Memory agent             |     |  Researcher agent     |
|  25 items per run         |     |  1 item per run       |
+-------------+-------------+     +-----------+-----------+
              |                               |
              v                               v
+-------------+-------------+     +-----------+-----------+
|  Audit Log                |     |  Audit Log            |
|  groundskeeper-log.jsonl  |     |  groundskeeper-       |
|  (still growing — see     |     |  research-log.jsonl   |
|   "The orphan skipper")   |     |  NEVER EXISTED        |
+---------------------------+     +-----------------------+

              |
              v
    +---------+---------+
    | Weekly Summary    |   nothing schedules it — see below
    | (groundskeeper-   |
    |  weekly-summary.py)|
    +-------------------+
```

Three notes on that picture, each of which was wrong in the original:

- **`groundskeeper-research-log.jsonl` has never existed on disk.** The
  `groundskeeper-research` skill still instructs the agent to append to it
  (`Step 6`), but the task that ran the skill is gone and the file is absent
  from `_pipeline/`. It is drawn here because the skill still asks for it, not
  because anything ever wrote it.
- **Nothing schedules `groundskeeper-weekly-summary.py`.** It is not in any
  timer, any autonomy task, or any skill — the only references to it in the
  tree are the script itself and this document. Its output
  (`_pipeline/groundskeeper-weekly-summary.md`) is nonetheless dated
  2026-09-10, so something runs it by hand or as part of a turn; see
  "The orphan skipper" for the same unresolved question about the log.
- **The queue is a file, not a service.** No code in `app/`, `workers/` or
  `agent_mcp/` mentions the groundskeeper at all; there is no worker source and
  no config key. The only programmatic consumers are the three vault skills,
  two of which no task invokes.

## Scanner (`scripts/groundskeeper/groundskeeper-survey.py`)

The scanner performs a comprehensive scan across 11 categories in two groups.
**Three roots, not one**, and the distinction matters for reading any item it
emits: `VAULT_ROOT` is `/home/alansrobotlab/obsidian` (`:32`), `FACTS_DIR` is
`app.paths.VAULT_FACTS_ROOT` — `~/lloyd/_pipeline/vault-derived/facts`, 25,155
entity directories, **outside the vault** (`:34`) — and `MEMORY_MD` is
`~/obsidian/lloyd/MEMORY.md` (`:35`). Output is
`~/lloyd/_pipeline/groundskeeper-queue.json` (`:33`).

### Fix Categories (Handled by Fix Loop)

| Category | Description |
|----------|-------------|
| BROKEN_LINK | Wiki-links that don't resolve to existing files |
| STALE_FACT | Facts with `last_updated > 30 days` old |
| STALE_RELATION | Broken relation paths in frontmatter `relations:` blocks |
| MEMORY_HYGIENE | Missing file references in `MEMORY.md` |
| ORPHAN_FILE | Files with zero inbound wiki-links or relations |
| THIN_PROFILE | Entities with < 3 facts (no knowledge/projects references) |
| MISSING_FRONTMATTER | Files missing frontmatter or incorrect segment/type fields |
| LARGE_DOC | Documents over 300 lines that may need splitting |

### Enrichment Categories (Handled by Research Loop)

| Category | Description |
|----------|-------------|
| ENRICH_THIN_PROFILE | Entities with < 3 facts BUT 2+ references in knowledge/projects |
| ENRICH_STALE_TOPIC | Knowledge files not updated in > 7 days with 2+ inbound links |
| ENRICH_STUB | Files in knowledge/projects with < 200 characters of content |

### Scanner Workflow

1. **File Index Building** — Scans all `.md` files, building:
   - Basename index: `{lowercase_name_without_ext: [full_paths]}`
   - Relative path index: `{lowercase_rel_path_without_ext: full_path}`
   - Used for wiki-link resolution (handles various wiki-link formats)

2. **Relation Inbound Counting** — Parses frontmatter `relations:` blocks:
   - Extracts `related-to:` and `references:` arrays
   - Builds map of which files have inbound relations
   - Used for orphan file detection and enrichment prioritization

3. **Health Score Computation** — Weighted composite across 11 dimensions:
   ```
   Overall = Σ(score_i × weight_i)
   
   Weights:
   - BROKEN_LINK: 15%
   - STALE_FACT: 15%
   - MEMORY_HYGIENE: 10%
   - ORPHAN_FILE: 10%
   - THIN_PROFILE: 5%
   - STALE_RELATION: 10%
   - ENRICH_THIN_PROFILE: 5%
   - ENRICH_STALE_TOPIC: 5%
   - ENRICH_STUB: 5%
   - MISSING_FRONTMATTER: 10%
   - LARGE_DOC: 10%
   ```

4. **Queue Idempotency** — Preserves status across runs by *dropping*, not by
   re-emitting. Each `check_*` function looks its item's stable id up in the
   previous queue and `continue`s when the status is `done` or `skipped` — the
   same three-line guard at eleven sites (`:179`, `:224`, `:276`, `:514`,
   `:589`, `:667`, `:705`, `:757`, `:812`, `:849`, `:892`). So a resolved item never
   comes back, and anything that *is* re-emitted is always `pending`. The
   consequence is that **a verdict is permanent**: once something marks an item
   `skipped`, no later survey raises it again however the vault changes. That is
   what makes "The orphan skipper" below a durable suppression rather than a
   daily annoyance.

5. **Atomic write, with a size check before the rename** (`:1083`–`:1102`). The
   queue goes to `QUEUE_OUTPUT + '.tmp'`, is read back, and its item count
   compared against what was intended; a mismatch deletes the temp file, leaves
   the previous queue in place, and raises `Queue corruption detected: size
   mismatch` (`:1099`). Only then does `os.rename` run (`:1102`).

   This exists because of the **2026-03-30 queue corruption incident**. The fix
   loop processed four `BROKEN_LINK` items and wrote *only those four* back over
   a queue holding ~2,700 — destroying the accumulated `done`/`skipped` status
   of everything else, which by the rule above is unrecoverable: the survey
   regenerates the items, but every verdict ever made on them is gone.
   `~/obsidian/memory/learnings/2026-03-30-groundskeeper-queue-corruption.md` is
   the write-up, and its most useful line is the admission that the skill had
   warned in bold not to do exactly that. The guard lives in the scanner rather
   than in the skill because a rule an LLM has to remember is not a rule.

## What the categories actually scan

The two tables above group by *who was supposed to fix it*. Grouping by *what
gets walked* is what you need to read an item:

- **Three categories never touch the vault.** STALE_FACT, THIN_PROFILE and
  ENRICH_THIN_PROFILE walk `FACTS_DIR`, which is outside it.
- **Their `source_file` names a path that no longer exists.**
  `check_thin_profiles` writes `source_file: memory/facts/<entity>/` (`:599`,
  `:610`) — vault-relative. `~/obsidian/memory/facts/` is gone; the fact tree
  moved to `~/lloyd/_pipeline/vault-derived/facts/`. That is 15,299
  THIN_PROFILE plus 9,559 ENRICH_THIN_PROFILE items — **79% of the queue** —
  each naming a directory nothing can open. STALE_FACT's is wrong in a louder
  way: `os.path.relpath(fact_file, VAULT_ROOT)` (`:231`) on a path outside the
  vault yields a `../lloyd/...` escape.
- **STALE_FACT currently emits nothing, and that is measured.** Across a
  4,000-entity sample, 6,805 fact files carry a quoted `last_updated:` and
  **zero** are older than 30 days. The regex requires the quotes (`:213`), and
  every file in the sample that has the field has it quoted — so the check is
  working and the facts are simply being re-dated by the extraction pipeline.
  The dimension scores 100 honestly.
- **MISSING_FRONTMATTER is scoped to four directories** — `knowledge/`,
  `projects/`, `personal/`, `work/` — and excludes `memory/`, `agents/`,
  `skills/` (`:785`–`:786`).
- **ENRICH_STALE_TOPIC uses filesystem mtime, not frontmatter** (`:740`),
  deliberately: enrichment rewrites a file without touching its dates, so a
  frontmatter rule reported the files it had just enriched.
- **BROKEN_LINK strips fenced and inline code before extracting `[[...]]`**
  (`:130`–`:131`). Without it, JSON and code examples in documents — including
  this one — read as wiki-links. Eleven false positives sourced to
  `groundskeeper.md` itself were hand-skipped on 2026-03-25, before the
  stripping landed.

## Two health dimensions are pinned at zero by their denominators

The weights are exact. The denominators are not: three of the eleven are the
literal `100`, marked `# Approximate` in the source (`:982`, `:983`, `:985`),
and the per-dimension score is `max(0, (1 - count / total) * 100)` (`:991`).

- **`large_doc` scores 0 whenever the vault holds more than 100 large
  documents.** It holds 148. The dimension has been 0 for as long as that has
  been true and carries 10% of the weight.
- **`orphan_files` scores 0 because numerator and denominator count different
  populations.** The denominator is `eligible_files`, excluding `memory/`,
  `agents/` and `skills/` (`:945`); the scan excludes only `sessions/`,
  `*/SKILL.md`, and files whose frontmatter says `type: facts` (`:481`). Orphans
  found under `memory/` therefore land in the numerator and not the denominator
  — 3,892 against 3,756. Another 10%.

So **20% of the composite is structurally zero regardless of the vault's
state**, and the headline 58.4 is that much below what its own dimensions
describe. The fix is making the denominators real; raising the constants would
only move the cliff.

## Frontmatter is read in a 2,000-byte prefix

`build_relation_inbound` (`:328`) and `check_stale_relations` (`:626`) both
`f.read(2000)` and regex the front matter out of that prefix. A `relations:`
block starting past it is invisible twice over: to the first, so its targets get
no inbound credit and can be reported as ORPHAN_FILE; to the second, so its
broken paths are never reported at all. This is the shape of the bug the
dashboard's `_frontmatter` reader had — a flat byte prefix hides precisely the
files with the longest front matter, which are the files with the most
relations. Unfixed; recorded because it fails silently in both directions.

## `find_hub_pages` is dead code

Defined at `:379`, with a docstring explaining hub-page candidates, and called
from nowhere. `check_orphan_files` builds its own hub set inline by a different
rule — a file is a hub when its basename equals its parent directory's name
(`:432`). Reading the function to understand orphan detection will mislead you;
the inline block is the live rule.

## The orphan skipper — an unattributed daily write

The one thing here still moving that should not be. Every ORPHAN_FILE item in
the queue — all 3,892 — reads `status: skipped`, `reason:
hub-page-linked-survey-bug`, stamped `2026-09-11T10:37:04Z`, eighteen minutes
after that night's survey finished. `_pipeline/groundskeeper-log.jsonl` has
grown to 18,176 rows, every one an ORPHAN_FILE skip, all dated September:
roughly 3,400–3,900 on each of 09-04, 09-05, 09-07, 09-09 and 09-11. The queue
also carries a top-level `items_processed` key that exactly one program writes.

That program is `scripts/memory/process-groundskeeper-queue.py`: a one-off that
marks every pending ORPHAN_FILE `skipped` with that exact reason string, written
back when the survey had no hub-page awareness and every organised project file
read as an orphan. The survey has had hub-page detection since. The premise is
gone; the script is not.

**What runs it nightly is unidentified.** It is in no timer (`systemctl --user
list-timers` shows four, one of them the survey), no autonomy task, no worker
source, no skill, and no session transcript in `sessions/` contains the reason
string. Until that is found the effect stands, and the idempotency rule makes it
compound: a category worth 10% of the health score is permanently zeroed every
night by a repair for a bug that was fixed. Twenty-five items carry a
hand-written reason instead — "legitimate organized project file in folder
hierarchy" — which is the fix loop's own voice from when it still ran.

## The skills outlived their tasks

Three skills remain under `~/obsidian/skills/`, all `status: active`:

| Skill | State |
|---|---|
| `groundskeeper-survey` | **live** — task #36 uses it; reads the queue and reports, never scans |
| `groundskeeper-loop` | orphaned — no task names it; still documents the 25-item fix pass |
| `groundskeeper-research` | orphaned — no task names it; still tells the agent to append to a log that has never existed |

An orphaned skill is not inert: it is loadable by name, and both still describe
running the 40-minute survey inline (`groundskeeper-research` Step 1 is
`Bash("python3 ~/lloyd/scripts/groundskeeper/groundskeeper-survey.py")`), which
is the thing that cost ~12 GPU-hours a week until 2026-09-03.

`scripts/groundskeeper/retention-sweep.py` shares the directory and nothing
else: it is a separate, live job (autonomy task #79, weekly, skill
`retention-sweep`, pinned by `tests/test_retention_sweep.py`) that bounds
unbounded-growth stores. It is not part of this system.
