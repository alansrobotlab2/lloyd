---
segment: architecture
tags: [architecture, lloyd, workers, automod]
type: reference
status: implemented
date: 2026-09-11
---

# arch-review — the pass that keeps these docs honest

`architecture/` was hand-curated on 2026-09-11: 22 top-level docs, 17 retired
into the gitignored `.archive/`, an [[index]] that lists what is left. Nothing
kept it honest from there. Three `tests/test_*_doc_claims.py` pin numbers in
three of the 22; `agent_mcp/memory_ops.py:12` cited a doc that no longer
existed; the measured tables in [[autonomy-jobs]] and [[workers-jobs]] are
snapshots of one afternoon. A doc that has drifted is worse than no doc,
because it is read as current.

`workers/sources/arch_review.py` is the pass. One unit per real session: check
the claims against the tree, review the code the unit names, **edit that one
doc**, and file everything else as backlog drafts.

It is deliberately one fifth of the machinery an earlier plan called for. That
plan built a registry parsed from the docs' own tables, bundles, churn-driven
cadence and three revert nets, and almost all of it protected against one
thing: *the model editing a production doc*. This version keeps the edit and
replaces the nets with a `git status` diff taken either side of the turn, two
numbers from `git diff --numstat`, and a commit the source makes rather than
the model.

## 1. The picklist is the scheduler

Two kinds of unit:

| kind | id | what it is |
|---|---|---|
| doc | `doc:<slug>` | one top-level `architecture/*.md`, whole |
| group | `group:<slug>:<name>` | one `## ` section of [[autonomy-jobs]] or [[workers-jobs]] |

Docs are read from disk at call time: every top-level `architecture/*.md`, and
never `.archive/` or a subdirectory. The exclusion is **by path**, not by
gitignore — five archived docs were still tracked when this landed, and a doc
retired *for being wrong* is the one review with no possible value.

The groups are hand-kept in `workers.sources.arch-review.groups`, one string
per functional group (11 today), because the two jobs docs group the unattended
fleet by what its members are *for*. A function is a review unit of its own:
who may write, which consumer does not exist, which edges cross groups, which
defect three members share. One session over an 824-line doc answers none of
those; one session over "Distil" answers all four.

A group's section is the `## ` heading whose text — after stripping a leading
`N. ` and everything from the first `:` or ` — ` on — casefold-equals the
configured name, running to the line before the next `## ` or EOF. Both jobs
docs decorate their headings with what the section contains (`## Distil: #38
#42 …`, `## 6. Mining — Lloyd's own exhaust back out`) and those suffixes change
whenever a member does, so matching the whole line would make every regroup a
`section_missing`. A `###` subheading and a `---` rule inside a section belong
to it; only a level-2 heading ends it.

**Cadence is age and nothing else.** Never-reviewed first, then oldest
`last_reviewed_at`, docs before groups on a tie. A unit rests
`review_interval_days` (30) after a review; `max_attempts` (3) consecutive
failures park it for `retry_spacing_seconds` (6 h). There is no churn trigger
and no backoff — 33 units at `daily_max: 4` is a first pass in about eight
days, and a review a month per unit after that. `daily_max` is counted from
`arch_review` ledger events rather than from the state file, because the ledger
is what survives the state file being deleted.

State is `~/.local/state/lloyd-automod/arch_review.json`, one row per unit:
`last_reviewed_at`, `reviewed_commit`, `verdict`, `filed`, `attempts`,
`last_attempt_at`, `last_error`, `pending_commit`.

## 2. Every review edits production, and four rails decide what survives

`~/lloyd` is the running tree: a saved file is a deploy. The turn is given
`Edit`, `Bash`, `Read`/`Grep`/`Glob`, the `graph_*` readers and
`backlog_write_task`, and denied `Write` (the doc exists — a new file is either
a finding to file or a stray write), `Task` (a subagent's writes land on the
parent's turn, after the diff was measured), `vault_write`, the automod tools
and every queue writer.

1. **A `git status` diff, not a snapshot.** Baselines are taken in `~/lloyd`
   and in the vault's `skills/` and `autonomy/` *before* the turn, with
   `--untracked-files=all` so an untracked directory is never one entry to
   delete wholesale. Afterwards, every path that appeared and is not the doc is
   reverted: tracked back to HEAD, untracked unlinked. A snapshot would revert
   a human's open editor buffer; a diff cannot. `backlog/` is deliberately
   outside the sweep, because filing is the job.
2. **The doc's own diff is bounded.** More than `max_delta_lines` (400)
   changed lines is a rewrite, not a correction. More than `max_shrink_pct`
   (30%) deleted is the doc gutted — waived for `superseded` and
   `aspirational`, where a banner over a body nobody should trust is the point.
   A front matter block that no longer opens the file is refused outright: the
   vault indexes on it.
3. **A group edits only its own section.** `git diff -U0` gives old-side hunks;
   a pure insertion `@@ -L,0` means "after old line L" and is inside when
   `start <= L <= end`, while a change `@@ -L,n` covers `L..L+n-1` and is inside
   when `start+1 <= L` and `L+n-1 <= end` — so the heading line is never
   editable. Zero context is load-bearing: with context a one-line edit reports
   a hunk reaching three lines into the neighbour, and seven groups sharing
   [[autonomy-jobs]] would reject each other constantly.
4. **The source commits, the model never runs `git`.** One `git commit -m … --
   <doc>` with a pathspec and no `git add`, so it commits that file's
   working-tree content and touches neither another file nor a human's
   partially staged index. It runs under `scripts.automod.state.Lock`
   (non-blocking) and only when `app.routers.automod.drain_active()` is False —
   a commit inside the promoter's idle window moves HEAD under a round that is
   mid-merge. Either refusal leaves the doc dirty with `pending_commit` on its
   state row, and the next tick pays it; the automod loop tolerates dirt
   disjoint from a round's own diff, so a waiting doc stalls nothing.

A rejected doc edit does **not** unfile the findings. The two halves of a
review are independent, and losing four real findings because the fifth
paragraph was too long would be the worse trade.

## 3. The verdict

`ARCH_REVIEW_SCHEMA` through the harness finalizer, with a trailing
`DOC_STATUS:` block as the fallback — last block wins, so a model that states
an outcome, reconsiders and restates has its final verdict read. `source` is
recorded on the event for the reason `autotriage` records it: a finalizer that
quietly stopped working looks exactly like one that is working, and the only
tell is the fallback rate.

| field | values |
|---|---|
| `doc_status` | `current`, `stale`, `superseded`, `aspirational` |
| `grouping` | `holds`, `split`, `merge`, `move` — groups only |
| `doc_updated`, `summary`, `filed`, `appended_to` | |

A **group** clamps `superseded`/`aspirational` to `stale`: a section cannot be
retired on its own, the doc it lives in holds that verdict, and a section
describing something gone is exactly what `stale` means. A **doc** never
reports a grouping.

`superseded` and `aspirational` do not trigger an archive path. The turn adds
one paragraph under the H1 and files a `needs-human` item — "retire
`architecture/<slug>.md` → `.archive/`". Moving a doc is a human's commit, or
an autocode round's.

Every `filed` id is verified on disk before the ledger records it: the file
must exist, carry the `arch-review` tag, and open with this unit's provenance
line (`Found reviewing architecture/<slug>.md`, plus ` §<heading>` for a
group). An id at or below the pre-turn `max_item_id()` is a **merge**, not a
spawn — `backlog_write_task` answers `merged_into: #n` when an open item
already covers the finding. An id with no file at all is `filed_unverified`.

## 4. The tag asymmetry

`spawned-by-review` is read by three things and they disagree on purpose:

| reader | keys on | so a review finding is |
|---|---|---|
| write-time merge (`agent_mcp/backlog.py`) | the `spawned-by-` **prefix** | merged into an item that already covers it |
| quarantine (`backlog.is_self_spawned`) | `SPAWN_TAGS`, the fixed four | **not** held out of the triage pool |
| expiry and the scorecard gauge | `LOOP_SPAWN_TAGS`, the union | closed at 30 d if nothing picked it up |

Quarantine asks whether an item can answer the staleness question. One triage
filed cannot — it was written from a check that had just run, so re-asking is
re-running that check. A review finding can: it describes the tree as of a
commit up to a month old, and deciding whether it still holds is precisely what
single triage is for. Expiry asks the other question — did anything ever act on
this? — and the answer is no for both, so the bound applies to both.

Inflow is bounded three ways regardless: `spawn_cap` (5) per run,
`max_open_items` (25) open `arch-review` drafts across both kinds before a tick
skips entirely, and the 30-day expiry. The middle one is the R > 1 lesson
applied before it can happen — a pass that files faster than the board closes
does not need better verdicts, it needs an edge cut.

## 5. Where it shows up

`scripts/automod/scorecard.py` row 12: units reviewed by kind, doc edits
committed against edits rejected, ids filed/merged/appended, stray writes
reverted, and the verdict histogram. `rejected` is the number to watch — it
counts turns whose doc edit was thrown away for breaking a bound, which is the
only way this job spends a whole session and produces nothing, and a rate that
stops being near zero means a bound is wrong rather than that the model is.

The ledger event is `arch_review` in `promotions.jsonl`, one per completed
review.

## 6. What it does not do

- **It does not move files.** No archive path; retirement is a `needs-human`
  item.
- **It does not fix what it finds** outside the one doc. A phantom tool name in
  a SKILL.md, an unbounded autonomy step, a dead consumer: all filed. This is
  the rule most likely to look like waste and is the reason the sweep in §2.1
  exists — the model is told it, *and* the tree enforces it.
- **It does not stage an artifact.** Nothing under `pending-research/`, so no
  `_DEFAULT_DEST` entry and no row in the Review tab. Its output is a commit
  and a set of drafts, both already durable.
- **It does not parse the docs' tables.** The group list is hand-kept, which is
  the cost of dropping the parser: one config line per regroup. The two tells
  that Alan regrouped and config did not follow are a `section_missing` failure
  and the `<groups_config>` block a jobs doc's *own* review is shown and told
  to file a finding about.

## Related

[[workers]], [[workers-jobs]], [[automod]], [[backlog]], [[index]]
