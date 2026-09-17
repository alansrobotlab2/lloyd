"""Background triage of the Lloyd backlog: re-check premises, retire the stale.

The backlog holds ~53 open items going back to February. Many were written
against a system that has since changed. This source works through them oldest
first and asks one question per item: **is this still true?**

It deliberately does **not** implement anything. Triage is read-only, and its
output is a verdict plus the evidence for it. Confirmed items accumulate as
*verified* work for the self-modification loop to pick up later, through the
normal gate. Splitting it this way is the whole point: the failure mode worth
avoiding is a confident, tested, gated change that solves a problem nobody has,
and that failure mode is only reachable if implementation can start from an
unverified premise.

Retiring an item is a success. For a backlog this age it is the *expected*
outcome, and a pipeline that only counts code as progress would quietly turn a
stale backlog into a pile of unnecessary changes.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from workers.queue import WorkQueue, QueueItem
from workers.sources._common import WORKER_AUTOMOD_BAN

logger = logging.getLogger("lloyd-workers.autotriage")

NAME = "autotriage"
#: A long-lived re-admitter: hand-driven triages took 45-76 iterations at
#: 120k+. Held by the pool's KV gate while the primary is over budget.
LONG_LIVED = True
# `priority ASC` — lower runs sooner. Below the research/distill stream (70),
# above the implement round (40): see autocode.DEFAULT_PRIORITY.
DEFAULT_PRIORITY = 55
DEDUP_KEY = "autotriage:triage"

# How many NEW items one triage may file, and only when the item it read is
# closing. A triage that keeps its item appends its findings to it instead —
# the rule the implement prompt has carried since 2026-09-11.
#
# 3 → 1 on 2026-09-13. The quarantine stopped the queue feeding itself, but
# the pass still filed 0.989 items per item it closed: 80 of the first 100
# `confirmed` single triages filed at least one item (0/1/2/3/4 filed in
# 20/34/29/15/2 runs), and not one `backlog_triage` row carried an append. The
# prompt was blind to where an item came from and told the model the item was
# "about to be closed" when a confirmed item was not. Recorded, not enforced:
# the items exist on disk before SPAWNED is parsed. A finding that lives only
# in EVIDENCE is still lost — see #229 — so the answer to more findings than
# the cap is `## Findings` on an item, never fewer findings.
SPAWN_CAP = 1

# Group mode: one turn over a cluster of related items from clusters.json
# (scripts/automod/cluster.py). The question is consolidation, not
# staleness, so quarantine does not apply and the fan-out is one item.
GROUP_SPAWN_CAP = 1
DEFAULT_GROUP_MIN_ITEMS = 2
# 8 -> 4 on 2026-09-14. An umbrella carries a clause or two per member, and
# umbrellas of 8-12 clauses were 56 of 92 `up_next` items while 5 of 68
# formed had landed. Four members is a contract a round can meet.
DEFAULT_GROUP_MAX_ITEMS = 4
DEFAULT_GROUP_MAX_TURNS = 120
PER_ITEM_MIN_CHARS = 2500

# Sweep mode (2026-09-15): a batch of items per turn, quarantine lifted,
# nothing filed, each item retired or ranked. Takes precedence over group
# and single triage while `backlog.sweep_pool` is non-empty, and autocode
# yields to it (`yield_to_sweep`). Ships off: `workers.sources.autotriage.sweep`.
DEFAULT_SWEEP_BATCH = 8
DEFAULT_SWEEP_MAX_TURNS = 60
# Read-only by construction, not by instruction: a sweep may not write the
# board, the tree or the vault, and may not open a round.
SWEEP_DISALLOWED: tuple[str, ...] = (
    *WORKER_AUTOMOD_BAN,
    "Edit", "Write", "Task", "backlog_write_task", "vault_write",
    "autonomy_write_task", "autonomy_delete_task", "autonomy_run_task",
    "research_propose", "research_next", "research_complete",
)

SWEEP_PROMPT = """\
You are SWEEPING {n} items from Lloyd's own backlog in one pass. The board holds \
hundreds of open items and the loop lands about ten a day, so most of these will \
wait weeks whatever you decide. Your job is to decide, cheaply, which are dead \
and how the rest should be ordered — not to fix anything, not to write a \
contract, and not to file or append anything.

<batch id="{batch_id}">
{items}
</batch>

For each item, in this order:

1. **Is it dead?** Read the item. If you can see from its text, or with ONE \
cheap check (a Grep, a Read of the file it names, a `git log -S`), that its \
premise no longer describes the system, the verdict is `stale`; if a commit \
already fixed it, `already_done` — name the commit. If another OPEN item, in \
this batch or elsewhere on the board, is the same finding, `duplicate_of #<id>` \
with the better handoff as the survivor (a survivor must itself be open and \
kept). Spend at most a few tool calls per item. If it would take a real \
investigation to know, it is not dead: keep it.
2. **If it lives, rank it** — `keep`, with two labels:
   - worth: `high` = a bug, a safety or data-integrity hole, or a measured \
correctness or throughput problem in the loop, the harness, the vault \
protection or the workers; or something Alan asked for by name. `medium` = a \
real improvement with evidence it is needed, not urgent. `low` = a nice-to-have, \
a speculative idea, a talk's suggestion with no evidence it fits this system, \
cosmetic, or about a subsystem that has since been retired. Low is NOT closed: \
it is parked, visible, and a person can promote it.
   - size: `small` = one or two files and one test; a round lands it in an \
hour. `medium` = a few files, a day of a person's work. `large` = cross-cutting, \
a new subsystem, or needs a design decision first.

Rules: you are read-only. Do not edit, write or commit anything, do not call \
backlog_write_task, do not append findings, do not start an automod round. A \
wrong `stale` or `duplicate_of` deletes a real finding, so cite what decides it; \
when unsure, `keep` and rank. `keep` for all {n} is a legitimate answer. Every \
item must be listed exactly once, with worth and size on every line, retiring \
verdicts included.

Finish with exactly this block and nothing after it:

SWEEP_VERDICTS:
#<id>: <stale | already_done | duplicate_of #<id> | keep> worth=<high|medium|low> size=<small|medium|large> — <one line of evidence>
(one line per item; every item listed)
"""

# Appended to GROUP_PROMPT when `form_umbrellas` is off: the cluster is
# still judged for duplicates and staleness, but nothing is consolidated.
NO_UMBRELLA_RULE = """

UMBRELLAS ARE OFF for this run. `fold` is not available: judge every item \
`duplicate_of`, `stale`, `already_done` or `keep`, file no umbrella, and \
write `UMBRELLA: none`. Items that would have been folded are `keep`.
"""

GROUP_PROMPT = """\
You are triaging {n} items from Lloyd's own backlog TOGETHER. A nightly pass \
found them related ({reason}); most were filed by earlier automod runs and \
several are probably the same finding written more than once. Your job is to \
consolidate them — not to fix anything, and not to re-ask whether each one is \
stale on its own.

<cluster id="{cluster_id}" anchor_paths="{anchor_paths}" parent="{parent}">
{items}
</cluster>

Work in this order:

1. **Read every item.** For each pair that says the same thing, decide which is \
the better handoff (more specific paths, the check that reproduces); the others \
are `duplicate_of` it. A duplicate's target must itself be `fold` or `keep`, \
never another duplicate.
2. **Check the shared premise ONCE** against the live tree (Read, Grep, Glob, \
Bash — read-only on the code, no edits, no automod round). If the premise is \
gone, those items are `stale`; if a commit fixed it, `already_done` — name the \
commit.
3. **Of what survives, decide what is ONE piece of work.** Items that would be \
implemented together are `fold`; an item that is genuinely separate work is \
`keep` (it goes back to the normal pool, unchanged).
4. **If two or more items are `fold`, file ONE umbrella** with \
`backlog_write_task` (board `lloyd`, no `task_id`, tags `umbrella` and \
`spawned-by-triage`). Its description is the merged handoff: first line \
"Umbrella for #a #b #c, formed by automod group triage of {cluster_id} on <date>"; \
then the claim, the current state with file paths and line numbers, the check \
that shows it, and the acceptance clauses — at most {max_clauses}, each one \
thing a single test can pin, each ending with the test file that pins it \
(`— tests/test_<area>.py`, an existing file or the one the round should \
create; for a `vault` umbrella, the vault path that shows it instead, never a \
test). The tool returns the id; that is UMBRELLA. Do not \
file an umbrella for a single item: one `fold` is a `keep`.
5. **File at most {spawn_cap} further item**, only for a finding none of these \
covers (tag `spawned-by-triage`). If the tool answers `merged_into: N`, list N.
6. Finish with exactly this block and nothing after it:

GROUP_VERDICTS:
#<id>: <duplicate_of #<id> | stale | already_done | fold | keep> — <one line of evidence>
(one line per item; every item listed)
UMBRELLA: <#id or none>
UMBRELLA_MEMBERS: <the ids marked fold, or none>
SURFACE: <one of code|frontend|vault|mixed|external>
CHECK: <the one check you ran, one line>
EVIDENCE: <2-4 sentences>
ACCEPTANCE: <the umbrella's contract, or none>
ACCEPTANCE_CLAUSES: <numbered, one per line, each ending "— tests/<file>.py", or none>
SPAWNED: <ids from step 5, or none>

Rules: closing a duplicate is a success and folding is a success; `keep` for \
all {n} is a legitimate answer. A wrong `duplicate_of` deletes a real finding, \
so cite what makes two items the same. Quarantine does not apply here: an item \
filed yesterday can be a duplicate of one filed last week. The umbrella's \
clauses are graded one by one at the gate by a reviewer who sees only the \
umbrella, its clauses and the diff, so name observable behaviour, not mechanism.
"""

PROMPT = """\
You are triaging one item from Lloyd's own backlog. It was written {age} days \
ago, and the system has changed since. Your job is to find out whether it is \
**still true** — not to fix it.

<item id="{item_id}" status="{status}" priority="{priority}">
# {name}

{body}
</item>

{origin}

Work in this order:

1. **State the premise.** In one sentence, what does this item assert is true \
about the system? If it asserts nothing checkable (it is an idea, a research \
prompt, or a wish), the verdict is `unverifiable`.
2. **Decide the surface.** Which part of the system would a fix touch? \
`code` (Python under `~/lloyd`), `frontend` (`web/src`), `vault` (skills, \
SOUL.md, memories, autonomy tasks and notes under `~/obsidian`), `mixed`, or \
`external`. Hardware, robots and third-party services are `external`: the \
verdict is `not_code`, do not investigate further. **Vault content is in \
scope** — the loop lands vault changes through its own route — so an item \
about a skill, a task definition or an identity file gets a real verdict.
3. **Design a check.** A command, a file to read, a grep, a metric to query — \
something that would come out differently depending on whether the premise \
holds. Write it down before running it.
4. **Run the check.** Use Read, Grep, Glob and Bash. You are read-only **on the \
code**: do not edit, write, or commit anything under the repo, and do not start \
an automod round. The backlog is the one thing you write to — step 6 requires it.
5. **Reach a verdict** from the evidence:
   - `confirmed` — the premise still holds; the problem is real today
   - `already_done` — it was real, and something has since fixed it
   - `stale` — the premise no longer describes this system
   - `unverifiable` — the item states no claim that can be checked
   - `not_code` — not about Lloyd's own code
6. **Put every finding where it will be read.** A triage nearly always turns \
up more than its headline premise: a narrower claim that still holds, a bug you \
noticed on the way, a newer premise the old one has become. Where each goes \
depends on the verdict, and you write it **now**, before the verdict block:

   - **`confirmed`, `unverifiable` or `not_code` — this item lives on.** Every \
finding goes **onto this item**, once, as a section: \
`backlog_write_task(task_id={item_id}, description_mode="append", \
description="## Findings (triage <date>)\\n\\n- <what is wrong, where \
(file:line), how to verify>")`. One bullet per finding. File no new item; \
SPAWNED is `none`. The implementer reads this item; a sibling item is one it \
never sees, and 80 of the first 100 confirmed triages filed one.
   - **`stale` or `already_done` — this item is about to be closed**, so a \
finding that survives it needs another home, in this order. An open item that \
already covers it — one a create's `similar` names, or one you found — gets \
it appended (`task_id=N, description_mode="append"`), and N goes under \
SPAWNED. Else this item's parent from `<origin>`, if still open, the same \
way. Only when neither exists, file **at most {spawn_cap} new item** with \
`backlog_write_task` (board `lloyd`, no `task_id`, tag `spawned-by-triage`), \
first line "Split from #{item_id} during automod triage on <date>", written as \
a handoff a fresh session can execute alone: the claim, the current state with \
file paths and line numbers, the check that shows it. More real findings than \
that go under `## Findings` on the item you filed. That is a cap on fan-out, \
not on honesty: nothing is dropped.

   The tool checks the board for you: when a create answers `merged_into: N`, \
an open item already covered the finding and your text was appended to it — \
list N under SPAWNED as you would a new id (the ledger tells the two apart). \
If a merge is wrong, re-file with `force: true` and say why in EVIDENCE. \
`<origin>` names what earlier triages of this item already filed or appended \
to; append to those rather than filing the same finding again. **A finding \
that lives only in EVIDENCE is lost**: nobody reads this transcript for \
to-dos. Filing nothing is fine when there is nothing — say `none` — but \
"those belong in two new items" with nothing written anywhere is the one \
outcome this step exists to prevent.

What good triage looks like — learned from the three items closed on \
2026-09-07, each of which turned on one of these:

- **Check the item's own numbers against the live system before anything \
else.** Spring items quote entity and edge counts that are now 5-30x off. An \
item whose headline figures no longer describe the system is usually `stale` \
before you read its proposal.
- **`git log -S'<symbol>'` and `git log --oneline -- <path>`** are how you tell \
`already_done` (a commit fixed it) from `stale` (the area was rewritten or \
removed). Name the commit.
- **Ask whether the surface the item targets has any traffic.** Grep \
`sessions/*.json` for the tool it improves. Work aimed at a surface with zero \
calls is `stale` whatever its premise says.
- **If it proposes a retrieval or graph change, name the metric in \
`eval/run_eval.py` that would move.** Edge-only changes move nothing there \
(measured); such an item is `unverifiable` until a harness exists, and you \
should say which harness.
- **Check for a newer item that already covers it.** Superseded is `stale`, \
and the evidence is the newer item's number.
- **An item making several claims gets a verdict per claim.** The verdict \
block judges the item's headline premise; every claim that survives it is \
written where step 6 says — under `## Findings` on this item, or for a \
closing item onto the item that covers it — not left in EVIDENCE.

Rules that matter:

- **`stale` and `already_done` are good outcomes.** Most of a backlog this old \
should reach them. Do not strain to confirm an item so it looks productive.
- **Never guess.** If you could not run a conclusive check, say `unverifiable` \
and explain what you would need. A wrong `confirmed` sends the automod loop \
after a problem that does not exist.
- **Quote your evidence.** File paths with line numbers, command output, commit \
SHAs. A verdict without evidence is unusable, because the point of this pass is \
that a human can audit it later.
- If you find the premise confirmed, also state **how the fix would be \
verified** — the check you just ran should fail to reproduce afterwards.
- **Some paths the loop may never touch**: `config.yaml`, `data/**`, `.env*`, \
`pytest.ini`, `.gitignore`, and under `web/` the build inputs (`package.json`, \
the lockfile, `vite.config.*`, `tsconfig*.json`). If the fix needs one of \
them the item is still `confirmed`, but a human has to land it: begin \
ACCEPTANCE with `human-only:` and name the path. The implementer skips those \
instead of spending a round finding out.
- **A condition only a person can satisfy is not an acceptance clause.** Ten \
items audited by Alan, a sign-off, a scope decision, a number that needs a week \
of real traffic: put it under HUMAN_CLAUSES, never under ACCEPTANCE_CLAUSES. \
The implementer is not asked to fake it, the reviewer does not grade it, and \
the item stays open tagged `needs-human` after the code lands until a person \
does it. #578 spent its round on a clause asking for ten human-audited items. \
The same rule for **a check that can only run after landing** — a day of \
traffic, a nightly run, a script over live data: the pre-landing mechanism \
and its test go in ACCEPTANCE_CLAUSES, the post-landing check goes in \
HUMAN_CLAUSES. #859 was refused twice on "needs a day of post-change \
traffic" with its mechanism complete.
- **Clauses must be jointly satisfiable.** Never write one clause forbidding \
a change beside another requiring its effect. #875 carried a clause fixing `n` \
and a clause requiring a floor that only a different `n` reaches, so no diff \
could satisfy both and the round could only be refused. A trade-off is ONE \
clause with the number in it ("the floor is 12,000 tokens"), not two clauses \
pulling opposite ways.
- **A clause pins the change, never an invariant the tree does not already \
hold.** Before writing a "no regression" or "unchanged" clause, check what the \
code does today and word the clause against that. #1199 asked for a smaller \
list payload; its contract said a priority-only save "leaves the file \
byte-identical", which the writer had never done (it round-trips YAML), and two \
rounds spent 300 iterations building raw-frontmatter preservation nobody asked \
for. "A priority-only update changes no body text" was the clause. If the item \
does not ask for the stronger property, do not require it.
- **When an item weakens a gate, a threshold or a check, add a purpose \
clause.** Name what that gate exists to catch and the test that shows it still \
catches it. Relaxing a check is the change most likely to pass every rung and \
be wrong, because the thing it stops catching leaves no trace.
- **A re-triaged item** — `<origin>` says so, and names the round that was \
refused, the review's findings and the grader's verdict per clause — was \
confirmed once and its round could not meet the contract. Judge the premise \
again from scratch. If it is still `confirmed`, write a NEW contract: drop \
every clause the grader found `unmet` or `unsatisfiable` on two reviews (its \
substance goes under `## Findings` on this item, or under HUMAN_CLAUSES when \
only a person can settle it), keep at most {max_clauses}, and never restate \
a clause the refusal shows no round can meet. If what survives is not worth a \
round, retire it (`stale`) and say why. This item gets no third automatic \
chance: a second refused attempt goes to a human.

Finish with exactly this block and nothing after it:

VERDICT: <one of confirmed|already_done|stale|unverifiable|not_code>
SURFACE: <one of code|frontend|vault|mixed|external>
CHECK: <the command or method you ran, one line>
EVIDENCE: <2-4 sentences citing what you actually observed>
ACCEPTANCE: <if confirmed: what must become true for this to be done; otherwise the word none>
ACCEPTANCE_CLAUSES: <if confirmed: the same contract as separately checkable clauses, \
one per line, each numbered "1." "2." …, each one thing a single test can pin, and each \
ending with the test file that pins it, e.g. "— tests/test_workers_pool.py"; otherwise the \
word none>
HUMAN_CLAUSES: <if confirmed and any: the conditions only a person can satisfy, one per \
line, numbered; otherwise the word none>
SPAWNED: <for stale/already_done, the ids you filed or appended to in step 6, e.g. #401 #402; otherwise the word none>

The clauses are graded one by one at the gate by a reviewer who sees only the \
item, the clauses and the diff — so a clause has to name the observable \
behaviour, not the mechanism ("a retried worker item fires `email_send` once", \
not "add a ledger"). **End each clause with the test file that pins it** \
(`— tests/<file>.py`: an existing file whose area it is, or the new file the \
round should create). The grader downgrades a `met` it cannot tie to a test \
node to `partial`, and naming the file removes the round's guesswork about \
where that node belongs. A `vault` surface is the exception: its review reads \
the vault and wants no test, so end each clause with the vault path that shows \
it (`— skills/<name>/SKILL.md`) — a test file named there sends the round to \
write one. **At most {max_clauses} clauses**: every clause is graded \
on its own and one unmet clause refuses the round, so six pass together less \
than half the time and twelve one time in five. If the work needs more, \
confirm the part one small change can finish and append the rest to this item \
under `## Findings` (step 6) — clauses past the cap are dropped, not graded.
"""

def _when(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(float(ts)))
    except (TypeError, ValueError):
        return "?"


def _origin_block(candidate, ledger) -> str:
    """Where this item came from and what has already been done with it.

    The prompt used to carry six fields — id, status, priority, name, body,
    age — and nothing saying "the loop filed this from a live check two days
    ago", "group triage judged it `keep` last night" or "an earlier triage of
    this item already filed #601". A model shown none of that re-derives it,
    and files it again.
    """
    from scripts.automod import backlog as B
    lines: list[str] = []
    origin = B.spawn_origin(ledger, candidate.id)
    parent_id = candidate.parent
    if origin:
        by, par = origin.get("by"), origin.get("parent")
        what = f" ({origin['verdict']})" if origin.get("verdict") else ""
        if by == "arch-review":
            lines.append(f"filed by the architecture review of {par}{what} on {_when(origin.get('ts'))}")
        else:
            lines.append(f"filed by {by} of #{par}{what} on {_when(origin.get('ts'))}")
            try:
                parent_id = int(par)
            except (TypeError, ValueError):
                pass
    elif B.is_loop_spawned(candidate):
        lines.append("tagged as filed by the loop, with no ledger row naming it")
    else:
        lines.append("no ledger record of the loop filing it: a human's item, or a writer outside the loop")
    keep = B.group_keep_note(ledger, candidate.id)
    if keep:
        lines.append(f"group triage {keep['cluster_id']} judged it `keep` on {_when(keep.get('ts'))}: "
                     f"distinct from its cluster, back in the single-item pool")
    prior = B.prior_triage_spawned(ledger, candidate.id)
    if prior:
        lines.append("earlier triages of this item filed or appended to: "
                     + " ".join(f"#{i}" for i in prior))
    lines.append(f"priority: {candidate.priority} (set on the item; high runs before medium before low "
                 "in every pool)")
    if candidate.worth or candidate.size:
        lines.append(f"a sweep read it and ranked it worth={candidate.worth or '?'} "
                     f"size={candidate.size or '?'}; your contract should fit that size")
    sections = B.findings_sections(candidate.body)
    if sections:
        lines.append(f"{sections} Findings section(s) already on this item")
    retriage = B.last_retriage(ledger, candidate.id)
    if retriage:
        lines.append(_retriage_line(retriage))
    if B.BLOCKER_TAG in candidate.tags:
        blocked = B.blocked_item_of(candidate, B.blocker_targets(ledger))
        target = B.item_by_id(blocked) if blocked is not None else None
        if blocked is None:
            lines.append("a BLOCKER, naming no item it blocks: judge whether it still stops anything")
        else:
            lines.append(f"a BLOCKER of #{blocked} ({target.status if target else 'not found'}): an "
                         f"implement round deferred a clause of #{blocked} to this item. Judge it as a "
                         f"handoff — is the obstacle still real on today's tree? A confirmed blocker "
                         f"skips the depth gate and is taken early")
    parent_attr = ""
    if parent_id:
        parent = B.item_by_id(parent_id)
        parent_attr = f' parent="#{parent_id} ({parent.status if parent else "not found"})"'
    return (f'<origin tags="{",".join(candidate.tags)}" created="{(candidate.created or "")[:10]}"'
            f'{parent_attr}>\n' + "\n".join(lines) + "\n</origin>")


def _live_blocker_waiting() -> bool:
    """Is a live blocker among the untriaged candidates right now?"""
    from scripts.automod import backlog as B, state as S
    candidates, _held = B.triage_pool(S.LEDGER_PATH)
    return bool(B.live_blockers(S.LEDGER_PATH, items=candidates))


def _retriage_line(ev: dict) -> str:
    """The refusal a re-triaged item carries into its second triage: which
    round, what the review found, and what the grader said per clause."""
    rid = ev.get("round_id") or "no round"
    parts = [f"RE-TRIAGED on {_when(ev.get('ts'))}: its implement attempt was spent ({rid})"]
    if ev.get("outcome_detail"):
        parts.append(f"outcome: {str(ev['outcome_detail'])[:300]}")
    if ev.get("findings"):
        parts.append(f"the review's findings: {str(ev['findings'])[:800]}")
    per_clause = "; ".join(
        f"clause {c.get('clause')} {c.get('verdict')}"
        + (f": {str(c.get('note'))[:160]}" if c.get("note") and c.get("verdict") != "met" else "")
        for c in (ev.get("clauses") or []) if isinstance(c, dict))
    if per_clause:
        parts.append(f"the grader per clause: {per_clause}")
    previous = [str(c) for c in (ev.get("previous_clauses") or [])]
    if ev.get("unmet_twice"):
        named = "; ".join(f"{n}. {previous[n - 1][:200]}" if 0 < int(n) <= len(previous) else str(n)
                          for n in ev["unmet_twice"])
        parts.append("graded unmet or unsatisfiable on two reviews, so drop them: clause(s) " + named)
    if previous:
        parts.append("the refused contract was: "
                     + " | ".join(f"{i}. {c[:200]}" for i, c in enumerate(previous, 1)))
    if ev.get("round_id"):
        parts.append(f"its work is kept on branch `automod/{ev['round_id']}`")
    return " — ".join(parts)


def _single_max_clauses() -> int:
    from scripts.automod.backlog import SINGLE_MAX_CLAUSES
    return SINGLE_MAX_CLAUSES


def render_prompt(candidate, *, ledger, body_chars: int = 30_000, spawn_cap: int = SPAWN_CAP) -> str:
    """The single-item prompt for one candidate."""
    return PROMPT.format(
        item_id=candidate.id, status=candidate.status, priority=candidate.priority,
        name=candidate.name, body=candidate.body[:body_chars], age=candidate.age_days,
        origin=_origin_block(candidate, ledger), spawn_cap=spawn_cap,
        max_clauses=_single_max_clauses(),
    )


def _acceptance_text(value: str) -> str:
    from scripts.automod.backlog import acceptance_text
    return acceptance_text(value)


def _parse_spawned(value: str) -> list[int]:
    from scripts.automod.backlog import parse_spawned
    return parse_spawned(value)


_FIELD = re.compile(
    r"^(VERDICT|SURFACE|CHECK|EVIDENCE|ACCEPTANCE_CLAUSES|HUMAN_CLAUSES|ACCEPTANCE|SPAWNED):\s*(.*)$",
    re.I)


def _clause_list(value, limit: int) -> list[str]:
    from scripts.automod.backlog import clean_clauses, split_clause_lines
    if isinstance(value, list):
        return clean_clauses(value, limit=limit)
    return split_clause_lines(str(value or ""), limit=limit)


def _clauses(value) -> list[str]:
    """Human clauses: not graded, so the read bound."""
    from scripts.automod.backlog import READ_MAX_CLAUSES
    return _clause_list(value, READ_MAX_CLAUSES)


def _contract_clauses(value) -> tuple[list[str], list[str]]:
    """`(acceptance clauses, dropped)`: the contract a round will be graded
    against, capped at `MAX_CLAUSES` on both parse paths, and the clauses that
    fell past the cap — as text, for the verdict row and the item."""
    from scripts.automod.backlog import MAX_CLAUSES
    every = _clause_list(value, 10_000)
    return every[:MAX_CLAUSES], every[MAX_CLAUSES:]


def parse_verdict(text: str, structured: dict | None = None) -> dict | None:
    """The turn's verdict, from the finalizer's object or from the text.

    Parsed from the LAST `VERDICT:` onward, not by one regex over the whole
    tail. A model that states a verdict, reconsiders, and restates would
    otherwise have the first verdict paired with the last evidence — a
    silently wrong record, which for this pipeline means a `confirmed` that
    nobody actually concluded.

    `structured` is the harness finalizer's output (a second completion under
    a JSON schema — see app/harness/finalizer.py). It wins when it carries a
    known verdict, and the text block stays in the prompt regardless: the
    finalizer is skipped whenever the turn did not end of its own accord, it
    can fail, and a verdict pipeline with no fallback would turn a transient
    engine error into a lost triage. The returned dict records which path
    produced it as `source`, so the ledger can show the fallback rate rather
    than the two being indistinguishable.
    """
    from scripts.automod.backlog import VERDICTS, SURFACES

    if isinstance(structured, dict):
        parsed = _from_structured(structured, VERDICTS, SURFACES)
        if parsed is not None:
            return parsed

    lines = text[-6000:].splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VERDICT:"):
            start = i
    if start is None:
        return None

    fields: dict[str, list[str]] = {}
    current = None
    for line in lines[start:]:
        m = _FIELD.match(line.strip())
        if m:
            current = m.group(1).upper()
            fields[current] = [m.group(2)]
        elif current:
            fields[current].append(line)

    verdict = " ".join(fields.get("VERDICT", [])).strip().lower()
    if verdict not in VERDICTS:
        return None
    surface = " ".join(fields.get("SURFACE", [])).strip().lower().strip("`'\"")
    if surface not in SURFACES:
        surface = "external" if verdict == "not_code" else "code"
    def joined(key: str, limit: int) -> str:
        return "\n".join(fields.get(key, [])).strip()[:limit]
    clauses, dropped = _contract_clauses(joined("ACCEPTANCE_CLAUSES", 8000))
    return {
        "verdict": verdict,
        "surface": surface,
        "check": " ".join(joined("CHECK", 4000).split())[:400],
        "evidence": joined("EVIDENCE", 2000),
        # The implementer's contract. 600 cut #278's mid-way through its
        # regression guards; a contract is not the field to save bytes on.
        "acceptance": _acceptance_text(joined("ACCEPTANCE", 4000))[:3000],
        # Its own field, not `- ` bullets under ACCEPTANCE: `ACCEPTANCE: -` is
        # the placeholder the text path already reads as "none", and a bullet
        # would collide with it.
        "acceptance_clauses": clauses,
        "clauses_dropped": len(dropped),
        "clauses_dropped_text": dropped,
        "human_clauses": _clauses(joined("HUMAN_CLAUSES", 4000)),
        "spawned": _parse_spawned(joined("SPAWNED", 400)),
        "source": "regex",
    }


def _from_structured(obj: dict, verdicts, surfaces) -> dict | None:
    """The finalizer's object, clamped the same way the text path clamps.

    The clamps live here rather than in the schema on purpose: a `maxLength`
    is enforced by the guided decoder, so the model would stop mid-sentence
    at the limit instead of writing something shorter. Truncating afterwards
    costs a cut sentence; constraining the grammar costs the thought.
    """
    verdict = str(obj.get("verdict") or "").strip().lower()
    if verdict not in verdicts:
        return None
    surface = str(obj.get("surface") or "").strip().lower()
    if surface not in surfaces:
        surface = "external" if verdict == "not_code" else "code"
    spawned = obj.get("spawned")
    if isinstance(spawned, list):
        # Reuse the text parser per entry rather than coercing: it already
        # knows that `#401` and `401` are the same id, and a second private
        # notion of "an item id" is how the two paths would come to disagree
        # about what the model filed.
        ids: list[int] = []
        for entry in spawned:
            ids.extend(_parse_spawned(str(entry)))
        spawned = ids
    else:
        spawned = _parse_spawned(str(spawned or ""))
    clauses, dropped = _contract_clauses(obj.get("acceptance_clauses"))
    return {
        "verdict": verdict,
        "surface": surface,
        "check": " ".join(str(obj.get("check") or "").split())[:400],
        "evidence": str(obj.get("evidence") or "").strip()[:2000],
        "acceptance": _acceptance_text(str(obj.get("acceptance") or ""))[:3000],
        "acceptance_clauses": clauses,
        "clauses_dropped": len(dropped),
        "clauses_dropped_text": dropped,
        "human_clauses": _clauses(obj.get("human_clauses")),
        "spawned": spawned,
        "source": "structured",
    }


DEFAULT_MAX_TURNS = 90
DEFAULT_BODY_CHARS = 30_000
# The depth gate's floor: single-item triage pauses while at least
# max(floor, items landed in 7 d) confirmed items are ready in up_next.
DEFAULT_IMPLEMENT_POOL_FLOOR = 20


_GROUP_LINE = re.compile(
    r"^\W*#?(\d+)\s*(?:->|=>|[:→—–-])+\s*(duplicate[ _]of|dup(?:licate)?|stale|already[ _]done|fold|keep)\b"
    r"\s*(?:#?(\d+))?\s*(?:[—–-]+\s*(.*))?$", re.I)
_GROUP_FIELD = re.compile(
    r"^(UMBRELLA_MEMBERS|UMBRELLA|SURFACE|CHECK|EVIDENCE|ACCEPTANCE_CLAUSES|ACCEPTANCE|SPAWNED):\s*(.*)$",
    re.I)


def _norm_group_verdict(word: str) -> str:
    w = word.strip().lower().replace(" ", "_")
    if w in ("dup", "duplicate", "duplicate_of"):
        return "duplicate_of"
    return w


def parse_group_verdict(text: str, structured: dict | None, member_ids: list[int]) -> dict | None:
    """`{items: {id: {verdict, duplicate_of, evidence}}, umbrella: {...}, spawned}`
    from the finalizer's object, else from the last GROUP_VERDICTS block.

    Unknown ids are dropped; unjudged members become `keep` and are listed
    under `unjudged`; anything unparsed degrades to `keep`, never to a close.
    """
    from scripts.automod.backlog import GROUP_VERDICTS, SURFACES
    members = {int(i) for i in member_ids}
    items: dict[int, dict] = {}
    umbrella: dict = {}
    spawned: list[int] = []
    source = "none"
    if isinstance(structured, dict) and isinstance(structured.get("items"), list):
        for raw in structured["items"]:
            if not isinstance(raw, dict):
                continue
            try:
                iid = int(raw.get("item_id"))
            except (TypeError, ValueError):
                continue
            v = str(raw.get("verdict") or "").strip().lower()
            if iid not in members or v not in GROUP_VERDICTS:
                continue
            items[iid] = {"verdict": v, "duplicate_of": int(raw.get("duplicate_of") or 0),
                          "evidence": " ".join(str(raw.get("evidence") or "").split())[:600]}
        u = structured.get("umbrella") or {}
        if isinstance(u, dict):
            umbrella = {"item_id": int(u.get("item_id") or 0),
                        "members": _parse_spawned(u.get("members")) if isinstance(u.get("members"), (str, list)) else [],
                        "surface": u.get("surface") if u.get("surface") in SURFACES else "code",
                        "check": str(u.get("check") or ""),
                        "evidence": str(u.get("evidence") or ""),
                        "acceptance": _acceptance_text(u.get("acceptance"))}
            umbrella["acceptance_clauses"], umbrella["clauses_dropped_text"] = \
                _contract_clauses(u.get("acceptance_clauses"))
        spawned = _parse_spawned(structured.get("spawned")) if isinstance(structured.get("spawned"), (str, list)) else []
        if items:
            source = "structured"
    if not items:
        tail = (text or "")[-12000:]
        idx = tail.rfind("GROUP_VERDICTS:")
        if idx < 0:
            return None
        block = tail[idx + len("GROUP_VERDICTS:"):]
        fields: dict[str, str] = {}
        current = ""
        for line in block.splitlines():
            fm = _GROUP_FIELD.match(line.strip())
            if fm:
                current = fm.group(1).upper()
                fields[current] = fm.group(2).strip()
                continue
            lm = _GROUP_LINE.match(line)
            if lm and not current:
                iid = int(lm.group(1))
                if iid in members:
                    items[iid] = {"verdict": _norm_group_verdict(lm.group(2)),
                                  "duplicate_of": int(lm.group(3) or 0),
                                  "evidence": (lm.group(4) or "").strip()[:600]}
                continue
            if current and line.strip():
                fields[current] = (fields[current] + "\n" + line.strip()).strip()
        if not items:
            return None
        source = "regex"
        umbrella = {"item_id": (_parse_spawned(fields.get("UMBRELLA")) or [0])[0],
                    "members": _parse_spawned(fields.get("UMBRELLA_MEMBERS")),
                    "surface": (fields.get("SURFACE") or "code").strip().lower(),
                    "check": fields.get("CHECK", ""), "evidence": fields.get("EVIDENCE", ""),
                    "acceptance": _acceptance_text(fields.get("ACCEPTANCE", ""))}
        umbrella["acceptance_clauses"], umbrella["clauses_dropped_text"] = \
            _contract_clauses(fields.get("ACCEPTANCE_CLAUSES", ""))
        if umbrella["surface"] not in SURFACES:
            umbrella["surface"] = "code"
        spawned = _parse_spawned(fields.get("SPAWNED"))
    unjudged = sorted(members - set(items))
    for iid in unjudged:
        items[iid] = {"verdict": "keep", "duplicate_of": 0, "evidence": "not judged by the turn"}
    return {"items": items, "umbrella": umbrella, "spawned": spawned,
            "unjudged": unjudged, "source": source}


_SWEEP_LINE = re.compile(
    r"^\W*#?(\d+)\s*(?:->|=>|[:→—–-])+\s*(duplicate[ _]of|dup(?:licate)?|stale|already[ _]done|keep)\b"
    r"\s*(?:#?(\d+))?(?P<rest>.*)$", re.I)
_SWEEP_WORTH = re.compile(r"\bworth\s*[=:]\s*(high|medium|low)\b", re.I)
_SWEEP_SIZE = re.compile(r"\bsize\s*[=:]\s*(small|medium|large)\b", re.I)


def parse_sweep_verdict(text: str, structured: dict | None, member_ids: list[int]) -> dict | None:
    """`{items: {id: {verdict, duplicate_of, worth, size, evidence}}, unjudged, source}`
    from the finalizer's object, else from the last SWEEP_VERDICTS block.

    Unknown ids are dropped. An unjudged member is NOT filled in as `keep`:
    it stays unswept and is offered to the next batch, because a rank the
    turn never wrote is not a rank.
    """
    from scripts.automod.backlog import SIZE_LEVELS, SWEEP_VERDICTS, WORTH_LEVELS
    members = {int(i) for i in member_ids}
    items: dict[int, dict] = {}
    source = "none"
    if isinstance(structured, dict) and isinstance(structured.get("items"), list):
        for raw in structured["items"]:
            if not isinstance(raw, dict):
                continue
            try:
                iid = int(raw.get("item_id"))
            except (TypeError, ValueError):
                continue
            v = str(raw.get("verdict") or "").strip().lower()
            if iid not in members or v not in SWEEP_VERDICTS:
                continue
            w = str(raw.get("worth") or "").strip().lower()
            s = str(raw.get("size") or "").strip().lower()
            items[iid] = {"verdict": v, "duplicate_of": int(raw.get("duplicate_of") or 0),
                          "worth": w if w in WORTH_LEVELS else "",
                          "size": s if s in SIZE_LEVELS else "",
                          "evidence": " ".join(str(raw.get("evidence") or "").split())[:600]}
        if items:
            source = "structured"
    if not items:
        tail = (text or "")[-16000:]
        idx = tail.rfind("SWEEP_VERDICTS:")
        if idx < 0:
            return None
        for line in tail[idx + len("SWEEP_VERDICTS:"):].splitlines():
            lm = _SWEEP_LINE.match(line)
            if not lm:
                continue
            iid = int(lm.group(1))
            if iid not in members:
                continue
            rest = lm.group("rest") or ""
            wm, sm = _SWEEP_WORTH.search(rest), _SWEEP_SIZE.search(rest)
            evidence = _SWEEP_SIZE.sub("", _SWEEP_WORTH.sub("", rest)).strip(" \t—–-:")
            items[iid] = {"verdict": _norm_group_verdict(lm.group(2)),
                          "duplicate_of": int(lm.group(3) or 0),
                          "worth": wm.group(1).lower() if wm else "",
                          "size": sm.group(1).lower() if sm else "",
                          "evidence": " ".join(evidence.split())[:600]}
        if not items:
            return None
        source = "regex"
    return {"items": items, "unjudged": sorted(members - set(items)), "source": source}


def _render_cluster(members, per_item_chars: int) -> str:
    parts = []
    for m in members:
        parts.append(f'<item id="{m.id}" status="{m.status}" age="{m.age_days}" '
                     f'tags="{",".join(m.tags)}">\n# {m.name}\n\n{m.body[:per_item_chars]}\n</item>')
    return "\n\n".join(parts)


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    # Budgets ride in the payload so `execute` sees the config that was live
    # when the item was queued, not whatever it is by the time it runs.
    new_id = queue.enqueue(
        source=NAME, kind="triage",
        payload={"max_turns": int(src_cfg.get("max_turns", DEFAULT_MAX_TURNS)),
                 "body_chars": int(src_cfg.get("body_chars", DEFAULT_BODY_CHARS)),
                 "structured_verdict": bool(src_cfg.get("structured_verdict", True)),
                 "spawn_cap": int(src_cfg.get("spawn_cap", SPAWN_CAP)),
                 "implement_pool_floor": int(src_cfg.get("implement_pool_floor",
                                                         DEFAULT_IMPLEMENT_POOL_FLOOR)),
                 # Kill switch for holding: off, a full pool pauses single
                 # triage outright (the 2026-09-13 behaviour) and anything
                 # already held is released.
                 "hold_confirmations": bool(src_cfg.get("hold_confirmations", True)),
                 # Group mode, carried like the budgets. Off: this source runs
                 # exactly as before and clusters.json is ignored.
                 "group_triage": bool(src_cfg.get("group_triage", True)),
                 "group_min_items": int(src_cfg.get("group_min_items", DEFAULT_GROUP_MIN_ITEMS)),
                 "group_max_items": int(src_cfg.get("group_max_items", DEFAULT_GROUP_MAX_ITEMS)),
                 "group_max_turns": int(src_cfg.get("group_max_turns", DEFAULT_GROUP_MAX_TURNS)),
                 # Off: a group triage still closes duplicates and retires
                 # the stale, but folds nothing and files no umbrella.
                 "form_umbrellas": bool(src_cfg.get("form_umbrellas", True)),
                 # Sweep mode, carried like the budgets. Off: never runs.
                 "sweep": bool(src_cfg.get("sweep", False)),
                 "sweep_batch": int(src_cfg.get("sweep_batch", DEFAULT_SWEEP_BATCH)),
                 "sweep_max_turns": int(src_cfg.get("sweep_max_turns", DEFAULT_SWEEP_MAX_TURNS))},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued backlog triage id=%d", new_id)


async def execute(item: QueueItem) -> dict[str, Any]:
    """Triage one item in a real session, and never mistake running out of
    room for a conclusion.

    Session-backed (see `run_prompt_in_session`) because a verdict on Lloyd's
    own code is something a human has to be able to review afterwards, and the
    direct `run_query` path leaves no transcript and attaches no observer.

    The budget check is the important part. A turn that hits `max_turns`
    stops cleanly with whatever text it had, which for a triage means no
    verdict block. That used to be recorded as `unverifiable` — a TERMINAL
    verdict — so the hardest items on the board were retired for good on first
    contact, for a reason indistinguishable from "states no checkable claim".
    Now it is recorded as `incomplete`, the item comes back, and only a second
    exhaustion retires it, with evidence that says exactly that.
    """
    from scripts.automod import backlog as B, state as S
    from workers.sources._common import DrainActive, TurnTimeout, run_prompt_in_session

    payload = item.payload or {}
    floor = int(payload.get("implement_pool_floor") or DEFAULT_IMPLEMENT_POOL_FLOOR)
    hold_on = bool(payload.get("hold_confirmations", True))
    # Room first: a confirmation held on an earlier run enters the pool
    # before this run adds to the queue behind it.
    try:
        released = await asyncio.to_thread(B.release_held_confirmations, S.LEDGER_PATH,
                                           floor=floor, enabled=hold_on)
        if released:
            logger.info("released %d held confirmation(s): %s", len(released),
                        [r["item_id"] for r in released])
    except Exception as exc:  # noqa: BLE001 — a release that fails costs a poll, not the run
        logger.warning("release_held_confirmations failed: %s", exc)

    # A `high` draft is picked up next (2026-09-16, Alan's ask): ahead of a
    # sweep batch, ahead of any cluster, ahead of the depth gate's pause, and
    # its confirmation is never held. The tag is a person saying "this one
    # now"; every other rule here is about the pile.
    urgent = await asyncio.to_thread(B.select_urgent, S.LEDGER_PATH)
    if urgent is not None:
        logger.info("backlog #%d is priority high: triaging it before the sweep and the clusters",
                    urgent.id)

    if urgent is None and bool(payload.get("sweep", False)):
        # The sweep goes first: while any open item is unread, reading eight
        # of them beats confirming one. Group and single triage resume on
        # their own once the pool is empty.
        batch = await asyncio.to_thread(B.select_sweep_batch, S.LEDGER_PATH,
                                        int(payload.get("sweep_batch") or DEFAULT_SWEEP_BATCH))
        if batch:
            return await _execute_sweep(item, batch)
        logger.info("sweep: every open item has been read; falling through to group/single triage")

    group_pick = None
    if urgent is None and bool(payload.get("group_triage", True)):
        from scripts.automod import cluster as CL
        pick = B.select_cluster(S.LEDGER_PATH, CL.load_clusters(),
                                min_size=int(payload.get("group_min_items") or DEFAULT_GROUP_MIN_ITEMS),
                                max_size=int(payload.get("group_max_items") or DEFAULT_GROUP_MAX_ITEMS))
        if pick is not None:
            # A qualifying cluster wins over the single pool: consolidation
            # is finite, the single pool is not — unless the single pool's
            # best candidate outranks every member on the human's
            # `priority` (2026-09-16). The tag is honoured across the two
            # pools, not only inside each; a `high` draft does not wait a
            # day of nightly clusters. The cluster is kept as the fallback
            # for the case the depth gate pauses single triage below.
            single = await asyncio.to_thread(B.select_candidate, S.LEDGER_PATH)
            if single is None or not B.priority_beats(single, pick[1]):
                return await _execute_group(item, pick[0], pick[1])
            group_pick = pick
            logger.info("single #%d (priority %s) outranks cluster %s; taking it first",
                        single.id, single.priority, pick[0].get("id"))

    # The depth gate. Triage confirmed 59 items on 2026-09-12 against a drain
    # of ~6 a day, and nothing read the depth: 78 of 89 up_next items had
    # never been attempted. A confirmation beyond what rounds will take within
    # a week is inventory that ages until a re-triage finds it stale. Counted
    # as `ready_confirmed`, what autocode would actually take.
    #
    # It gates the CONFIRMATION, not the turn. Its first cut returned here,
    # which also stopped triage's retirements — the loop's largest closer
    # (23 of 103 verdicts the day before) — and left it filing faster than it
    # closed. A full pool now holds a `confirmed` verdict in `draft` (see
    # `B.held_confirmations`); only with holding switched off does it pause.
    gate = await asyncio.to_thread(B.implement_pool_full, S.LEDGER_PATH, floor=floor)
    # A live blocker is never held back by the gate, holding or not: the pause
    # below would otherwise leave it untriaged (select_candidate takes it first).
    if gate["full"] and not hold_on and urgent is None \
            and not await asyncio.to_thread(_live_blocker_waiting):
        if group_pick is not None:
            return await _execute_group(item, group_pick[0], group_pick[1])
        return {"status": "skipped",
                "summary": (f"single-item triage paused: {gate['ready']} ready in up_next ≥ bound "
                            f"{gate['bound']} ({gate['landed_items_7d']} items landed in 7 d, "
                            f"floor {gate['floor']}); group triage still runs")}

    candidates, held = B.triage_pool(S.LEDGER_PATH)
    candidate = B.select_candidate(S.LEDGER_PATH)
    if candidate is None:
        # Say which empty this is. A queue drained of real work and a queue
        # holding 106 of this loop's own drafts look identical from here, and
        # only one of them means the pass is finished.
        summary = "every open backlog item has been triaged"
        if held:
            summary = (f"{summary}; {held} self-filed item(s) held — they are never "
                       f"triaged one by one, and expire unclustered at "
                       f"{B.spawn_expiry_days()} days")
        return {"status": "skipped", "summary": summary}
    if held:
        logger.info("triage pool: %d candidate(s), %d self-filed item(s) quarantined",
                    len(candidates), held)

    budget = int((item.payload or {}).get("max_turns") or DEFAULT_MAX_TURNS)
    body_chars = int((item.payload or {}).get("body_chars") or DEFAULT_BODY_CHARS)
    # Kill switch, in the payload like the budgets so a queued item runs under
    # the config that was live when it was enqueued. With it off the turn runs
    # identically and only the regex path reads the result — which is what
    # makes flipping it a real rollback rather than a code path nobody has
    # exercised. Items queued before this landed default to on.
    want_structured = bool((item.payload or {}).get("structured_verdict", True))
    logger.info("triaging backlog #%s (%s days old, budget %d): %s",
                candidate.id, candidate.age_days, budget, candidate.name[:70])

    spawn_cap = int(payload.get("spawn_cap") or SPAWN_CAP)
    # Off the event loop: the origin block reads the ledger four times.
    prompt = await asyncio.to_thread(render_prompt, candidate, ledger=S.LEDGER_PATH,
                                     body_chars=body_chars, spawn_cap=spawn_cap)
    # Taken BEFORE the turn: an id the turn claims that is at or below this
    # already existed, so it is a merge (or a citation), not a spawn.
    id_floor = B.max_item_id()
    try:
        run = await run_prompt_in_session(
            prompt, title=f"backlog triage #{candidate.id}: {candidate.name[:48]}",
            source=NAME, max_turns=budget, priority=1,
            final_schema=B.TRIAGE_VERDICT_SCHEMA if want_structured else None,
            final_schema_prompt=(
                "Restate the verdict block above as a single JSON object "
                "matching the schema. Same verdict, same acceptance check, "
                "same filed ids — this is a transcription, not a re-decision."
            ))
    except DrainActive as exc:
        # A landing owns the backend right now. Not a result; try next tick.
        return {"status": "skipped", "summary": f"landing in progress: {exc}"}
    except TurnTimeout as exc:
        # In-band, so the queue does not retry. Raising would send the item
        # back through the retry path, and since no verdict was recorded the
        # retry re-selects this same item — a second full triage of the item
        # that has already proved it does not finish in the time allowed.
        S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                        "verdict": B.INCOMPLETE, "reason": "turn timeout",
                        "budget": budget, "auto": True})
        logger.warning("backlog #%s: %s", candidate.id, exc)
        return {"status": "failed", "item_id": candidate.id,
                "summary": f"#{candidate.id}: {exc}"}

    session_id = run["session_id"]
    stop_reason = run.get("stop_reason")
    structured = run.get("structured") if want_structured else None
    structured_error = str(run.get("structured_error") or "")
    parsed = parse_verdict(run["text"], structured)

    if not parsed:
        if stop_reason == "max_turns":
            attempt = B.incomplete_counts(S.LEDGER_PATH).get(candidate.id, 0) + 1
            if attempt < B.MAX_INCOMPLETE_ATTEMPTS:
                S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                                "verdict": B.INCOMPLETE, "attempt": attempt,
                                "budget": budget, "session_id": session_id,
                                "num_turns": run.get("num_turns")})
                logger.warning("backlog #%s: out of budget (%d) on attempt %d — will retry",
                               candidate.id, budget, attempt)
                return {"status": "skipped", "item_id": candidate.id,
                        "verdict": B.INCOMPLETE, "attempt": attempt,
                        "session_id": session_id,
                        "summary": f"#{candidate.id} ran out of budget ({budget}); retrying later"}
            evidence = (f"ran out of iteration budget ({budget}) on {attempt} consecutive "
                        f"attempts without reaching a verdict; transcript in session "
                        f"{session_id}")
        else:
            evidence = (f"triage turn ended ({stop_reason}) with no parseable verdict "
                        f"block; transcript in session {session_id}")
        # Terminal, ledger only: the item's status is not touched, but the
        # reason is honest and the transcript is named.
        S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                        "verdict": "unverifiable", "check": "", "evidence": evidence,
                        "auto": True, "session_id": session_id,
                        "verdict_source": "none",
                        "structured_error": structured_error,
                        "finalizer_tokens": run.get("finalizer_tokens"),
                        "stop_reason": stop_reason})
        logger.warning("backlog #%s: %s", candidate.id, evidence)
        return {"status": "success", "item_id": candidate.id, "verdict": "unverifiable",
                "session_id": session_id, "summary": evidence[:200]}

    # Retiring verdicts close the item; everything else only annotates it.
    # `confirmed` deliberately does NOT open a round — implementation is the
    # `autocode` source's job, and it is gated separately.
    close = parsed["verdict"] in B.RETIRING

    # What the model says it filed is a claim; the file on disk is the fact.
    # An id with no file behind it is recorded as unverified, never as a link.
    spawned, merged = B.split_claimed(parsed["spawned"], id_floor=id_floor,
                                      self_id=candidate.id)
    unverified = [i for i in parsed["spawned"]
                  if i not in spawned and i not in merged and i != candidate.id]
    if unverified:
        logger.warning("backlog #%s: SPAWNED names %s but no such item exists",
                       candidate.id, unverified)
    # Counted off the file, and before `record_verdict` rewrites it: a turn
    # that says it appended three findings and appended none reads as none.
    after = B.load_item(candidate.path)
    findings_appended = B.count_findings(candidate.body, after.body if after else candidate.body)
    # Asked again now, not at the start: a triage turn runs for many minutes,
    # and rounds land and take items in the meantime.
    hold = False
    if hold_on and parsed["verdict"] == "confirmed" and not B.is_human_only(parsed["acceptance"]):
        gate = await asyncio.to_thread(B.implement_pool_full, S.LEDGER_PATH, floor=floor)
        # A live blocker is never held: it is the precondition of a clause a
        # round already deferred, not more inventory for a full pool. Nor is
        # a `high` item: held, it would wait for the pool to drain under its
        # bound, which on 2026-09-16 was 73 ready against a bound of 73 —
        # "picked up next" would have meant never.
        live = await asyncio.to_thread(B.live_blockers, S.LEDGER_PATH, items=[candidate])
        hold = bool(gate["full"]) and candidate.id not in live and not B.is_high(candidate)
    B.record_verdict(candidate, parsed["verdict"], parsed["evidence"],
                     check=parsed["check"], close=close, spawned=spawned, merged=merged,
                     acceptance=parsed["acceptance"],
                     acceptance_clauses=parsed.get("acceptance_clauses") or (),
                     human_clauses=parsed.get("human_clauses") or (),
                     dropped_clauses=parsed.get("clauses_dropped_text") or (),
                     hold=hold)

    # The cap is a prompt instruction, and the items exist on disk by the time
    # we read SPAWNED — unfiling them would destroy real findings. So it is
    # recorded rather than enforced: a number that can be watched, on the one
    # metric that told us the pass had inverted.
    # No `+1` any more: the overflow item it allowed for is gone, so a second
    # filing is an overshoot and is logged as one.
    over_cap = max(0, len(spawned) - spawn_cap)
    if over_cap:
        logger.warning("backlog #%s filed %d item(s) over the cap of %d",
                       candidate.id, over_cap, spawn_cap)

    S.append_event({"event": "backlog_triage", "item_id": candidate.id,
                    "name": candidate.name[:200], "age_days": candidate.age_days,
                    "spawn_cap": spawn_cap, "spawned_over_cap": over_cap,
                    "findings_appended": findings_appended,
                    "verdict": parsed["verdict"], "surface": parsed["surface"],
                    "held": hold,
                    "check": parsed["check"],
                    "evidence": parsed["evidence"][:1000],
                    "acceptance": parsed["acceptance"],
                    "acceptance_clauses": parsed.get("acceptance_clauses") or [],
                    # Real clauses the verdict wrote past MAX_CLAUSES and the
                    # parser dropped. Non-zero says the prompt's budget was
                    # ignored, which is worth watching rather than inferring.
                    "clauses_dropped": int(parsed.get("clauses_dropped") or 0),
                    "clauses_dropped_text": parsed.get("clauses_dropped_text") or [],
                    "human_clauses": parsed.get("human_clauses") or [],
                    "closed": close,
                    "spawned": spawned, "merged": merged, "id_floor": id_floor,
                    "spawned_unverified": unverified,
                    # Which parser produced this verdict, and why the
                    # structured one did not when it did not. Without both,
                    # a finalizer that silently stopped working looks exactly
                    # like one that is working.
                    "verdict_source": parsed.get("source", "regex"),
                    "structured_error": structured_error,
                    "finalizer_tokens": run.get("finalizer_tokens"),
                    "session_id": session_id, "stop_reason": stop_reason,
                    "num_turns": run.get("num_turns"), "budget": budget})

    logger.info("backlog #%s → %s%s%s (session %s)", candidate.id, parsed["verdict"],
                " (closed)" if close else "", " (held: pool full)" if hold else "", session_id)
    return {"status": "success", "item_id": candidate.id, "name": candidate.name,
            "verdict": parsed["verdict"], "closed": close, "held": hold,
            "session_id": session_id,
            "summary": f"#{candidate.id} → {parsed['verdict']}"
                       f"{' (closed)' if close else ''}"
                       f"{' (held: implement pool full)' if hold else ''}"}



async def _execute_group(item: QueueItem, cluster: dict, members: list) -> dict[str, Any]:
    """One turn over a cluster: per-item verdicts, one umbrella at most."""
    from scripts.automod import backlog as B, state as S
    from workers.sources._common import DrainActive, TurnTimeout, run_prompt_in_session

    payload = item.payload or {}
    budget = int(payload.get("group_max_turns") or DEFAULT_GROUP_MAX_TURNS)
    body_chars = int(payload.get("body_chars") or DEFAULT_BODY_CHARS)
    want_structured = bool(payload.get("structured_verdict", True))
    cid = str(cluster.get("id") or "")
    per_item = max(PER_ITEM_MIN_CHARS, body_chars // max(1, len(members)))
    member_ids = [m.id for m in members]
    logger.info("group triage %s: %d items %s (budget %d)", cid, len(members), member_ids, budget)

    prompt = GROUP_PROMPT.format(
        n=len(members), reason=str(cluster.get("reason") or "similar text and shared files"),
        cluster_id=cid, anchor_paths=",".join(cluster.get("anchor_paths") or []) or "none",
        parent=f"#{cluster['parent']}" if cluster.get("parent") else "none",
        items=_render_cluster(members, per_item), max_clauses=B.MAX_CLAUSES,
        spawn_cap=GROUP_SPAWN_CAP)
    form_umbrellas = bool(payload.get("form_umbrellas", True))
    if not form_umbrellas:
        prompt += NO_UMBRELLA_RULE
    id_floor = B.max_item_id()
    try:
        run = await run_prompt_in_session(
            prompt, title=f"backlog group triage {cid}: {len(members)} items",
            source=NAME, max_turns=budget, priority=1,
            final_schema=B.GROUP_TRIAGE_SCHEMA if want_structured else None,
            final_schema_prompt=(
                "Restate the GROUP_VERDICTS block above as a single JSON object matching "
                "the schema: one entry per item with its verdict, the umbrella you filed "
                "(item_id 0 if none) with its clauses, and the ids you filed or were merged "
                "into. A transcription, not a re-decision."))
    except DrainActive as exc:
        return {"status": "skipped", "summary": f"landing in progress: {exc}"}
    except TurnTimeout as exc:
        S.append_event({"event": "backlog_group_triage", "cluster_id": cid, "item_ids": member_ids,
                        "verdict": B.INCOMPLETE, "reason": "turn timeout", "budget": budget,
                        "judged": {}}, path=S.LEDGER_PATH)
        return {"status": "failed", "summary": f"group triage {cid}: {exc}"}

    session_id = run["session_id"]
    stop_reason = run.get("stop_reason")
    structured = run.get("structured") if want_structured else None
    parsed = parse_group_verdict(run.get("text") or "", structured, member_ids)
    if not parsed:
        attempts = sum(1 for d in B._ledger_events(S.LEDGER_PATH, "backlog_group_triage", require_item=False)
                       if d.get("cluster_id") == cid and d.get("verdict") == B.INCOMPLETE) + 1
        outcome = B.INCOMPLETE if attempts < B.MAX_INCOMPLETE_ATTEMPTS else "abandoned"
        S.append_event({"event": "backlog_group_triage", "cluster_id": cid, "item_ids": member_ids,
                        "verdict": outcome, "attempt": attempts, "judged": {},
                        "session_id": session_id, "stop_reason": stop_reason,
                        "num_turns": run.get("num_turns"), "budget": budget}, path=S.LEDGER_PATH)
        logger.warning("group triage %s: no parseable verdict block (%s); %s", cid, stop_reason, outcome)
        return {"status": "skipped" if outcome == B.INCOMPLETE else "success",
                "summary": f"group triage {cid}: no verdict ({stop_reason}); {outcome}"}

    if not form_umbrellas:
        # A fold the model wrote anyway is a keep, and an umbrella it filed
        # anyway is not confirmed (it stays an ordinary self-spawned draft).
        for v in parsed["items"].values():
            if v.get("verdict") == "fold":
                v["verdict"] = "keep"
                v["evidence"] = (v.get("evidence") or "") + " (umbrellas are off; kept)"
        parsed["umbrella"] = {**parsed["umbrella"], "item_id": 0}
    spawned, merged = B.split_claimed(parsed["spawned"], id_floor=id_floor, self_id=0)
    umbrella = None
    uid = int(parsed["umbrella"].get("item_id") or 0)
    if uid and uid > id_floor:
        umbrella = next((i for i in B.open_items(None) if i.id == uid), None)
    fold_ids = [i for i, v in parsed["items"].items() if v["verdict"] == "fold"]
    if umbrella is not None and len(fold_ids) < 2:
        # One fold is a keep, and an umbrella over one item is just a copy.
        B.note_item(uid, f"filed by group triage {cid} over fewer than two folds; not confirmed")
        umbrella = None
    spawned = [i for i in spawned if i != uid]
    # The umbrella is a confirmation like any other, so it waits for room like
    # any other. Its folds and retirements still apply now.
    hold = False
    if umbrella is not None and bool(payload.get("hold_confirmations", True)):
        floor = int(payload.get("implement_pool_floor") or DEFAULT_IMPLEMENT_POOL_FLOOR)
        gate = await asyncio.to_thread(B.implement_pool_full, S.LEDGER_PATH, floor=floor)
        hold = bool(gate["full"])
    result = B.record_group_verdict(cluster, members, parsed["items"], umbrella, parsed["umbrella"],
                                    session_id=session_id, spawned=spawned, merged=merged, hold=hold,
                                    extra={"verdict_source": parsed["source"],
                                           "structured_error": str(run.get("structured_error") or ""),
                                           "stop_reason": stop_reason,
                                           "num_turns": run.get("num_turns"), "budget": budget,
                                           "unjudged": parsed["unjudged"],
                                           # A run that could not fold binds nothing
                                           # for the clusterer (`group_triaged_ids`).
                                           "form_umbrellas": form_umbrellas})
    summary = (f"group triage {cid}: {result['duplicates']} duplicate(s) closed, "
               f"{result['retired']} retired, {result['folded']} folded"
               + (f" into #{result['umbrella_id']}" if result.get("umbrella_id") else "")
               + f", {result['kept']} kept")
    logger.info(summary)
    return {"status": "success", "cluster_id": cid, "session_id": session_id, "summary": summary,
            **{k: result[k] for k in ("duplicates", "retired", "folded", "kept", "umbrella_id")}}


async def _execute_sweep(item: QueueItem, members: list) -> dict[str, Any]:
    """One turn over a batch: each item retired or ranked, nothing filed."""
    from scripts.automod import backlog as B
    from workers.sources._common import DrainActive, TurnTimeout, run_prompt_in_session

    payload = item.payload or {}
    budget = int(payload.get("sweep_max_turns") or DEFAULT_SWEEP_MAX_TURNS)
    body_chars = int(payload.get("body_chars") or DEFAULT_BODY_CHARS)
    want_structured = bool(payload.get("structured_verdict", True))
    member_ids = [m.id for m in members]
    batch_id = B.sweep_batch_id(member_ids)
    per_item = max(PER_ITEM_MIN_CHARS, body_chars // max(1, len(members)))
    logger.info("sweep %s: %d items %s (budget %d)", batch_id, len(members), member_ids, budget)

    prompt = SWEEP_PROMPT.format(n=len(members), batch_id=batch_id,
                                 items=_render_cluster(members, per_item))
    try:
        run = await run_prompt_in_session(
            prompt, title=f"backlog sweep {batch_id}: {len(members)} items",
            source=NAME, max_turns=budget, priority=1,
            extra_disallowed=list(SWEEP_DISALLOWED),
            final_schema=B.SWEEP_SCHEMA if want_structured else None,
            final_schema_prompt=(
                "Restate the SWEEP_VERDICTS block above as a single JSON object matching "
                "the schema: one entry per item with its verdict, duplicate_of (0 unless "
                "duplicate_of), worth, size and evidence. A transcription, not a re-decision."))
    except DrainActive as exc:
        return {"status": "skipped", "summary": f"landing in progress: {exc}"}
    except TurnTimeout as exc:
        return _sweep_out_of_budget(batch_id, member_ids, budget, session_id="",
                                    stop_reason="turn_timeout", detail=str(exc))

    session_id = run["session_id"]
    stop_reason = run.get("stop_reason")
    structured = run.get("structured") if want_structured else None
    parsed = parse_sweep_verdict(run.get("text") or "", structured, member_ids)
    if not parsed:
        return _sweep_out_of_budget(batch_id, member_ids, budget, session_id=session_id,
                                    stop_reason=str(stop_reason), num_turns=run.get("num_turns"))
    result = B.record_sweep_verdicts(batch_id, members, parsed["items"], session_id=session_id,
                                     extra={"verdict_source": parsed["source"],
                                            "structured_error": str(run.get("structured_error") or ""),
                                            "stop_reason": stop_reason,
                                            "num_turns": run.get("num_turns"), "budget": budget,
                                            "unjudged": parsed["unjudged"]})
    summary = (f"sweep {batch_id}: {result['retired']} retired, {result['duplicates']} duplicate(s) "
               f"closed, {result['kept']} ranked ({result['parked']} parked)"
               + (f", {len(parsed['unjudged'])} unjudged" if parsed["unjudged"] else ""))
    logger.info(summary)
    return {"status": "success", "batch_id": batch_id, "session_id": session_id, "summary": summary,
            **{k: result[k] for k in ("retired", "duplicates", "kept", "parked")}}


def _sweep_out_of_budget(batch_id: str, member_ids: list[int], budget: int, *, session_id: str,
                         stop_reason: str, detail: str = "", num_turns=None) -> dict[str, Any]:
    """A batch that reached no verdict block: `incomplete` once (the same
    batch is offered again, since its items are still unswept), `abandoned`
    the second time — its items are then left to the ordinary passes rather
    than retried forever. Nothing is written on the items either way."""
    from scripts.automod import backlog as B, state as S
    attempts = B.sweep_incomplete_attempts(S.LEDGER_PATH, batch_id) + 1
    outcome = B.INCOMPLETE if attempts < B.MAX_INCOMPLETE_ATTEMPTS else "abandoned"
    S.append_event({"event": "backlog_sweep", "batch_id": batch_id, "item_ids": member_ids,
                    "verdict": outcome, "attempt": attempts, "judged": {}, "ranked": {},
                    "session_id": session_id, "stop_reason": stop_reason,
                    "num_turns": num_turns, "budget": budget, "detail": detail[:300]},
                   path=S.LEDGER_PATH)
    logger.warning("sweep %s: no parseable verdict block (%s); %s", batch_id, stop_reason, outcome)
    return {"status": "skipped" if outcome == B.INCOMPLETE else "success",
            "batch_id": batch_id,
            "summary": f"sweep {batch_id}: no verdict ({stop_reason}); {outcome}"}
