---
segment: architecture
tags: [architecture, lloyd, workers, automod]
type: reference
status: implemented
date: 2026-10-12
---

# arch-review — the pass that keeps these docs honest

`architecture/` was hand-curated on 2026-09-11: 22 top-level docs then, **36
now** (measured 2026-10-12; the glob is the authority, and
`tests/test_architecture_coverage_doc_claims.py` pins it as
`ARCH_DOC_SURFACE`) — this doc joined the picklist the day it landed, so the
pass reviews itself — and 17 docs retired, **none of them readable in this
checkout**: architecture/.archive/ does not exist here. Five retired copies were
tracked in git at all, and `f80c9d00` is the commit that untracked all five; the
rest were deleted from `architecture/` with no tracked copy made at any point.
So what survives of each is the commit's parent — for those five,
`git show f80c9d00^:architecture/.archive/<slug>.md`; for the others,
`git show b94be171^:architecture/<slug>.md`. An [[index]] lists what is left,
and its Retired section names every one of them. Nothing
kept it honest from there. Three `tests/test_*_doc_claims.py` pinned numbers in
three of them (23 such files now, over more than twice as many docs);
`agent_mcp/memory_ops.py:12` cited a doc that no longer
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
replaces the nets with a scratch checkout the turn edits in (since #1462), a
`git status` diff taken either side of the turn, two numbers from
`git diff --numstat`, and a commit the source makes rather than the model.

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
and no backoff — the picklist is read from disk at call time, and at
`daily_max: 4` a board this size is a first pass in its unit-count divided by
four, and a review a month per unit after that. No count of
units is restated in prose for the same reason: the first one that was went
stale the same day. `daily_max` is counted from
`arch_review` ledger events rather than from the state file, because the ledger
is what survives the state file being deleted.

State is `~/.local/state/lloyd-automod/arch_review.json`, one row per unit:
`last_reviewed_at`, `reviewed_commit`, `verdict`, `filed`, `attempts`,
`last_attempt_at`, `last_error`, `pending_commit`, `commit_conflict`.

## 2. A review edits a scratch checkout, and four rails decide what reaches live

`~/lloyd` is the running tree: a saved file is a deploy. Until 2026-09-25 the
turn edited it in place — on 09-24 `workers-jobs.md` sat dirty on live `main`
for 20+ minutes mid-review (21 edits): embedded by qmd's watcher, recallable,
dirt in the landing path, stranded by any backend death, and a human who opened
the doc mid-turn would have had their edits committed under `arch-review:`
(#1462). Now `execute` cuts a **detached worktree at live HEAD** under
`~/.local/state/lloyd-automod/arch-review-worktrees/` and hands the turn that
path as `Repository:`; production is named in the prompt as read-only.
Reviewing code at HEAD in a checkout is reviewing live, and the health routes
are the live backend's either way. Not under `~/lloyd-work`: autocode counts
every worktree there as an open round. The checkout is removed in a
`finally` (success, drain, timeout, exception), and one a killed backend left
behind is swept at the top of the next tick — `_ACTIVE_WORKTREES` is an
in-memory set, so a restart empties it together with the turns it described.

What the turn may not
use is a **deny-list over the whole chat toolbox**, not a grant list:
`workers/sources/_common.py:1043` says outright that a worker
session "is handed exactly the toolbox a chat gets", and `DISALLOWED`
subtracts from that. It names `Write` (the doc exists — a new file is either
a finding to file or a stray write), `Task` (a subagent's writes land on the
parent's turn, after the diff was measured), `vault_write`, the automod tools,
the autonomy writers, `http_request`, `graph_refresh` and the mutating half of
the browser. The job needs `Edit`, `Bash`, `Read`/`Grep`/`Glob`, the `graph_*`
readers and `backlog_write_task`. Since #709 landed the memory and fact writers
are named too — `memory_add`/`memory_remove`/`memory_replace`, and
`fact_add`/`fact_relate`/`fact_invalidate`/`fact_resolve`/`fact_resolve_apply` —
and that half is load-bearing rather than belt-and-braces: `_pipeline/` is
gitignored, `git status` does not report an ignored path, so §2.1's sweep is
structurally incapable of seeing a fact write and the deny-list is the only
defence there, not the second one.

1. **A `git status` diff, not a snapshot.** Baselines are taken in the
   checkout, on live and across the vault except `backlog/` *before* the turn,
   with `--untracked-files=all` so an untracked directory is never one entry to
   delete wholesale. Afterwards, every path that appeared in the checkout or
   the vault and is not the doc is reverted: tracked back to HEAD, untracked
   unlinked (in the checkout that is moot — it is deleted — but the record is
   the point). A snapshot would revert a human's open editor buffer; a diff
   cannot. A path already dirty *before* the turn is fingerprinted and a
   rewrite of it is reported, never reverted (#915); a vault `autonomy/*.md`
   task file is the scheduler's and is kept (#1296). **On live, the turn's
   own writes are reverted precisely, everything else is reported.** A
   `git status` diff of live cannot tell a model's write into `~/lloyd` by
   absolute path from a human's, a landing's or a nightly's made in the same
   minutes, so live is not swept. The per-turn change ledger can:
   `revert_live_ledger_writes` walks every turn of the review's own session
   under `sessions/<sid>.changes/` (the session is minted for this review, and
   the turn id never returns to the caller), takes each entry whose realpath is
   under live and not under the checkout, and reverts it through
   `_change_ledger.revert` — pre-image restored, a create removed
   (`reverted via change ledger (restored|deleted)`). The ledger refuses a file
   whose content moved since the turn wrote it, and a refusal is reported by
   name (`refused by change ledger: …; not reverted`), never forced. A live
   path that changed but is in no ledger entry — a human's edit, a Bash write,
   which the ledger does not see — is reported
   (`changed on live during the turn; not reverted`). `backlog/` is
   deliberately outside the sweep, because filing is the job.
2. **The doc's own diff is bounded.** More than `max_delta_lines` (400)
   changed lines is a rewrite, not a correction — that cap is `git diff
   --numstat` over the whole file for both unit kinds. More than
   `max_shrink_pct` (30%) deleted is the doc gutted — waived for `superseded`
   and `aspirational`, where a banner over a body nobody should trust is the
   point. The denominator of the percentage is the **unit**, not the file: a
   group's deletions are a share of its own section (#913 — 43 lines of an
   824-line doc, where the whole file as denominator let one turn delete its
   section entire and score a few percent; the section rail of §2.3 cannot
   catch that case, since a hunk removing the whole section is by construction
   inside it). What is still wrong is the sentence the turn is shown: the
   shared `PROMPT` tells a group "more than 30% of the doc deleted" while the
   rail it will be judged by says *of your section* (#2603).
   A front matter block that no longer opens the file is refused outright: the
   vault indexes on it.
3. **A group edits only its own section.** `git diff -U0` gives old-side hunks;
   a pure insertion `@@ -L,0` means "after old line L" and is inside when
   `start <= L <= end`, while a change `@@ -L,n` covers `L..L+n-1` and is inside
   when `start+1 <= L` and `L+n-1 <= end` — so the heading line is never
   editable. Zero context is load-bearing: with context a one-line edit reports
   a hunk reaching three lines into the neighbour, and seven groups sharing
   [[autonomy-jobs]] would reject each other constantly.
4. **The source commits onto live, the model never runs `git` — one half by
   blob check, the other by `app/harness/live_git_writes.py::check_live_git_writes`
   (#2604).** Both halves used to be instructions with one rail underneath: the
   commit arrived through `commit_reviewed`, and the no-`git` rule lived only in
   the prompt, where `Bash` is kept on purpose and `git commit` sits at tier 1
   beside the `git log` the prompt orders. A commit was also invisible — the live
   sweep baselines twice with `git status --porcelain` and reports the difference,
   and a commit cleans the tree it dirtied, while `revert_live_ledger_writes`
   walks the `Edit`/`Write` change ledger and a Bash write leaves no entry there.
   So the lane now refuses a mutating git subcommand (`commit`, `add`, `apply`,
   `reset`, `rebase`, `checkout`, `restore`, `switch`, `merge`, `revert`,
   `cherry-pick`, `amend`, a non-read `stash`) whose working directory is
   `app.paths.LIVE_CHECKOUT` **or any directory inside it** — git resolves a
   repository by walking up, so `cd <live>/architecture` is the same `HEAD` — for
   any session `attended_session` cannot place with
   a person — and the run event's `live_head_moved` names base and observed HEAD
   when HEAD moved anyway and `commit` was not this source's. Reads are
   untouched (`log`, `status`, `diff`, `show`, `grep`, `merge-base`, `stash list`),
   as is any other checkout: a round's worktree, a `/tmp` clone and the vault all
   still commit. The bound
   checks run on the checkout's diff; what survives is carried to live by
   `commit_reviewed`. Under `scripts.automod.state.Lock` (non-blocking) and
   only when `app.routers.automod.drain_active()` is False — a commit inside
   the promoter's idle window moves HEAD under a round that is mid-merge — it
   requires the live doc's blob in HEAD, the index **and** the working tree to
   still equal the review's base blob, writes the reviewed content, and runs
   one `git commit -m … -- <doc>` with a pathspec and no `git add`. Live HEAD
   having moved for other files is fine; the doc having moved is a
   `commit_conflict`: never an overwrite, recorded on the event and the state
   row, and the reviewed content kept under `arch-review-conflicts/` for a
   human to read. A drain or the lock **defers**: the content goes to
   `arch-review-pending/<unit>.md` with `pending_commit {rel, message,
   content_path, base_blob}`, live stays clean, and the next tick commits it
   under the same conflict rule. A pending row from before #1462 (only
   `rel`, doc dirty on live) is still paid by `commit_doc`. A turn that timed
   out never has its half edit committed.

**What still rests on the prompt, after the lane.** Rail 4's second clause is
"instruct **only** to run `git` reads", and the lane enforces one property of that
sentence — no mutating `git` in the production checkout — not the sentence. A turn
can still dirty live with `sed -i`, a redirected `cp`, or `rm`; nothing here
promises production stays clean, and the sweep is what reports it when it does
(`revert_live_ledger_writes` for the `Edit`/`Write` half, the `live_strays` report
for the rest). The rail is that a turn's judgement can no longer become a
**permanent object on `main`** by a command the source did not run: a tree write is
revertable and visible, a commit was neither.

A rejected doc edit does **not** unfile the findings. The two halves of a
review are independent, and losing four real findings because the fifth
paragraph was too long would be the worse trade.

## 3. The verdict

`ARCH_REVIEW_SCHEMA` through the harness finalizer, with a trailing
`DOC_STATUS:` block as the fallback — last block wins, so a model that states
an outcome, reconsiders and restates has its final verdict read.
`verdict_source` is
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
one paragraph under the H1 and files an item — "retire
`architecture/<slug>.md` → `.archive/`". Moving a doc is a human's commit, or
an autocode round's.

Every `filed` id is verified on disk before the ledger records it: the file
must exist, carry the `arch-review` tag, and carry this unit's provenance line
(`Found reviewing architecture/<slug>.md`, plus ` §<name>` for a group — the
**configured** name from `groups:`, not the heading text it resolved to, so a
regroup that decorates a heading does not orphan what was already filed for it).
Since #914 the provenance is matched as a whole line (`^…\s*$`, multiline)
anywhere in the item's first 20k characters, so a doc unit can no longer accept
one of its own groups' items as proof; the tag half is still a substring test
over the same 20k, so an item that merely mentions `arch-review` in its body
satisfies it. An id at or below the pre-turn `max_item_id()` is a
**merge**, not a spawn — `backlog_write_task` answers `merged_into: #n` when an
open item already covers the finding. An id with no file at all is
`filed_unverified`.

## 4. The tag asymmetry

`spawned-by-review` is read by three things and they disagree on purpose:

| reader | keys on | so a review finding is |
|---|---|---|
| write-time merge (`agent_mcp/backlog.py`) | the `spawned-by-` **prefix** | merged into an item that already covers it |
| quarantine (`backlog.is_self_spawned`) | `quarantine_tags()`: `SPAWN_TAGS`, the fixed four, plus `youtube-eval` | **not** held out of the triage pool |
| expiry and the scorecard gauge | `loop_spawn_tags()` **and** the `spawned-by-` prefix | closed at 30 d if nothing picked it up |

The prefix half of the last row is #1160: `backlog_write_task` stamps
`spawned-by-<whatever minted it>`, so an enumeration of the names anyone
thought of in advance undercounted the loop's own open output by 10% while
expiry read those items as a human's. Quarantine keeps the exact set on
purpose — widening it would hold every `spawned-by-review` item out of the
single triage that exists to re-check it.

Quarantine asks whether an item can answer the staleness question. One triage
filed cannot — it was written from a check that had just run, so re-asking is
re-running that check. A review finding can: it describes the tree as of a
commit up to a month old, and deciding whether it still holds is precisely what
single triage is for. Expiry asks the other question — did anything ever act on
this? — and the answer is no for both, so the bound applies to both.

Inflow is bounded at three points, but only two of them bind. `max_open_items`
(25) open `arch-review` drafts across both kinds stops the whole tick, and
expiry (`spawn_expiry_days()` — `expire_spawns_after_days: 30`, raised from 7 on
2026-09-15, because expiry was then the only exit a quarantined draft had and it
was closing items nothing had read) closes anything untouched; `spawn_cap` (5) per run is a prompt
instruction plus the `spawned_over_cap` field on the ledger event — nothing
truncates a turn that files a sixth finding. The middle one is the R > 1 lesson
applied before it can happen — a pass that files faster than the board closes
does not need better verdicts, it needs an edge cut, and the only edge cut here
is the one that stops the tick.

## 5. Where it shows up

`scripts/automod/scorecard.py` row 12: units reviewed by kind, doc edits
committed against edits rejected, ids filed/merged/appended, stray writes
reverted, and the verdict histogram. `rejected` is the number to watch — it
counts turns whose doc edit was thrown away for breaking a bound, which is the
only way this job spends a whole session and produces nothing, and a rate that
stops being near zero means a bound is wrong rather than that the model is.

The ledger event is `arch_review` in `promotions.jsonl`, one per completed
review.

Kill switch: `workers.sources.arch-review.enabled`.

## 6. What it does not do

- **It does not move files.** No archive path; retirement is a filed
  item.
- **It does not fix what it finds** outside the one doc. A phantom tool name in
  a SKILL.md, an unbounded autonomy step, a dead consumer: all filed. This is
  the rule most likely to look like waste and is the reason the sweep in §2.1
  exists — the model is told it, *and* the tree enforces it. The sweep covers
  the checkout whole-tree and the whole vault bar `backlog/`, and watches live;
  what it cannot cover
  is a **gitignored** path, since `git status` does not report one, so the
  tools that write under `_pipeline/` (`fact_*`) or to `lloyd/MEMORY.md`
  (`memory_*`) are denied instead (#709).
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

## Review log

- 2026-09-12 — **stale**. The mechanism is all live and verified against
  `workers/sources/arch_review.py` at `2de2150`, but four counts and two
  claims about enforcement were wrong: the picklist is 23 docs / 34 units and
  `.archive/` held 12 at the time (not 22 / 33 / 17) — those five were lost to
  a `git rm --cached` whose deletion a merge then applied to the working tree,
  and they were restored the same evening, so it holds 17 again; tool policy is
  a deny-list over the
  chat toolbox rather than the grant list §2 described, so the memory and fact
  writers are live and outside the sweep (#709); `spawn_cap` is a prompt plus a
  gauge, not a bound; and two rails are looser than their own sentences — the
  shrink percentage is whole-file for groups (#913) and the filing check matches
  provenance as a substring (#914). Filed #709, #913, #914, #915, #916.
- 2026-10-12 — **stale**. The mechanism is unchanged and every rail in §2 still
  runs, but four sentences described defects that have since been fixed, and
  §4 disagreed with itself across thirty lines. `memory_add` and the `fact_*`
  writers are now in `DISALLOWED` (#709 landed), so §2's "still live" clause
  contradicted §6's "denied instead"; the shrink percentage takes the **unit**
  as its denominator rather than the whole file (#913 landed), which §2.2 now
  says, though the shared `PROMPT` still tells a group the bound is a share of
  the doc (#2603); the filing check matches the provenance as a whole line and
  names the configured group name (#914 landed); quarantine is the fixed four
  plus `youtube-eval`, and expiry and the gauge take the `spawned-by-` prefix
  as well as the enumeration (#1160). §4's own table said 30 d while its prose
  said 7 d — the prose was the false side, and the bound has been 30 since
  2026-09-15. Corpus counts: 36 top-level docs, 23 `tests/test_*_doc_claims.py`
  files, and §1's "a first pass in about a week" is replaced by the division it
  was an estimate of, since the picklist is 47 units and not 33.
  `_common.py:592-600` is `:1043`. Filed #2603 (the prompt/rail denominator
  mismatch, with the `group:workers-jobs:Dispatch` rejection it has already
  cost), #2604 (rail 4's "the model never runs git" is a prompt rule, not a
  rail — `Bash` is kept and `git commit` is tier 1), #2605 (the module's own
  prose still says five retired docs are tracked, and #916 closed without
  fixing it). #2604 landed the same day, so "the mechanism is unchanged" held
  only for that pass: §2 rail 4 is a rail now
  (`app/harness/live_git_writes.py`, the sixth lane of `check_bash_command`) and
  the run event carries `live_head_moved`, which is what makes the sentence true
  of the property it was written for rather than of the prompt.
