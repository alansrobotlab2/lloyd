"""Treat a backlog item as a hypothesis, not an instruction.

The Lloyd backlog holds 50-odd open items going back to February. Many were
written against a system that has since changed: the bug was fixed, the module
was rewritten, the upstream issue closed, the approach was superseded. Acting
on those blind produces the worst possible outcome — a confident, tested,
gated change that solves a problem nobody has.

So nothing here implements anything until the item's *premise* has been
re-checked against the system as it is today. Four verdicts:

  confirmed     the premise still holds; there is real work here
  already_done  the premise held once and something has since fixed it
  stale         the premise no longer describes this system
  unverifiable  the item states no claim that can be checked

**`stale` and `already_done` are successes.** For a backlog this age, retiring
items with evidence is worth more than implementing them, and it is the
outcome most items should reach. A pipeline that "succeeds" only by writing
code will quietly convert a stale backlog into a pile of unnecessary changes.

The premise check does double duty. When it confirms a problem, that same
check becomes the acceptance test: after the fix lands it must now fail to
reproduce. An item whose premise cannot be turned into a check is not ready to
be implemented automatically — that is what `unverifiable` means, and it is a
request for a human to sharpen the item, not a failure.
"""

from __future__ import annotations

import json
import time
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

# `app/kg_store.py:48` shape; the reason and the 15.1x measurement are written
# out at `app/routers/dashboard.py`. `_split_frontmatter` is the hot one of the
# three readers: `all_items`, `load_item` and every save path funnel through it,
# and `board_health` walks the whole board with it. `yaml.safe_load` names the
# pure-Python `SafeLoader` in its own signature, so the swap has to happen at
# the call site — `tests/test_dashboard_yaml_loader.py` proves it by counting
# loader constructions, not by reading this attribute.
try:  # ~15x faster than the pure-Python loader; all input is our own files
    from yaml import CSafeLoader as _YamlLoader  # type: ignore
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _YamlLoader  # type: ignore

from app.backlog_status import (
    CLOSED_ALIASES,
    CLOSED_STATUSES,
    OPEN_STATUSES,
    PIPELINE_STATUSES,
    canonical_status,
    is_off_vocabulary,
)
from app.backlog_tags import is_spawn_tag, normalize_tags

BACKLOG_DIR = Path.home() / "obsidian" / "backlog"

# Tags `backlog_write_task` puts on items this loop files for itself —
# `spawned-by-triage` from a verdict turn, `spawned-by-autocode` from an
# implement round. They are how a self-filed item is told from a human one,
# and `draft` alone cannot do it: that is the status of half the board.
# Both spellings: ~100 items on the board carry the pre-rename tag, and
# quarantine that stopped recognising them would re-admit every one of them
# to the triage pool at once. New items get the new tag.
# Two renames of the implement source (selfmod → autoimplement on 2026-09-09,
# autoimplement → autocode the same day) left their tags on the board. Read all
# three; write the newest. Quarantine that stopped recognising an old tag would
# re-admit every item carrying it to the triage pool at once.
SPAWN_TAGS = frozenset({"spawned-by-triage", "spawned-by-autocode",
                        "spawned-by-autoimplement", "spawned-by-selfmod"})

# The architecture reviewer (`workers/sources/arch_review.py`) files findings
# the same way, and they are loop output by every measure that matters for the
# board's size — so expiry bounds them and the scorecard gauge counts them.
# They are NOT quarantined, and that asymmetry is the point of keeping two
# names rather than widening one.
REVIEW_SPAWN_TAGS = frozenset({"spawned-by-review"})

# The YouTube channel digest (`workers/sources/youtube_digest.py`) files its
# evaluations tagged `youtube-eval` and no `spawned-by-*` tag, so until
# 2026-09-14 they read as a human's items: never quarantined, never expired,
# never merged at write time. A one-day backfill filed 125 in the week to
# 09-14 and 101 were still open. Alan's ruling that day: they are loop output.
# Quarantined as well as expired — a confirmed eval item becomes
# `confirmed-held`, which expiry exempts, so bounding only expiry would leave
# the pass free to confirm them into a pile nothing drains. Kill switch
# `workers.sources.youtube-digest.loop_spawned` (see `eval_items_loop_spawned`).
EVAL_SPAWN_TAGS = frozenset({"youtube-eval"})

# Three readers, three different rules, and they disagree on purpose:
#
#   merge at write time   `spawned-by-` prefix   (agent_mcp/backlog.py)
#                         + `youtube-eval`
#   quarantine            QUARANTINE_TAGS        (is_quarantined, below)
#   expiry and the gauge  LOOP_SPAWN_TAGS        (this union)
#                         + the `spawned-by-` prefix
#
# Quarantine asks "can this item answer the staleness question?" — an item
# triage filed from a check it just ran cannot, by construction, and neither
# can an evaluation written against today's profile of the system. A review
# finding can: it describes the tree as of one commit a month ago, and single
# triage is exactly the pass that should decide whether it still holds. Expiry
# asks the other question — "did anything ever pick this up?" — and the answer
# is no for all three, so the bound applies to all three. Both unions are the
# shape with the eval switch on; readers go through `quarantine_tags()` and
# `loop_spawn_tags()`, which honour it.
#
# The prefix belongs to the expiry/gauge rule and not to quarantine, and the
# asymmetry is the point: `backlog_write_task` stamps `spawned-by-<whatever the
# session was>`, so the exact set here was only ever the subset of mints someone
# thought of in advance. Measured through the production loader on 2026-09-17,
# 43 of 412 open loop-tagged items were rejected by the enumeration — 23
# distinct minters — and the scorecard gauge read 369 against a true 412, a 10 %
# undercount of the loop's own output. #1160 was among them, which is why every
# triage of it was told "a human's item, or a writer outside the loop".
# Quarantine stays exact on purpose: widening it would hold out of single-item
# triage every `spawned-by-review` item whose comment says it should be
# re-checked, and would re-admit nothing the enumeration had already held.
LOOP_SPAWN_TAGS = SPAWN_TAGS | REVIEW_SPAWN_TAGS | EVAL_SPAWN_TAGS
QUARANTINE_TAGS = SPAWN_TAGS | EVAL_SPAWN_TAGS


def eval_items_loop_spawned() -> bool:
    """`workers.sources.youtube-digest.loop_spawned`, default on. Lazy, and
    fail-open to the ruling, so the automod CLI does not need a config."""
    try:
        from app.config import CONFIG
        cfg = (((CONFIG or {}).get("workers") or {}).get("sources") or {}).get("youtube-digest") or {}
        return bool(cfg.get("loop_spawned", True))
    except Exception:  # noqa: BLE001
        return True


def quarantine_tags() -> frozenset[str]:
    """The tags that hold an item out of single-item triage, right now."""
    return SPAWN_TAGS | (EVAL_SPAWN_TAGS if eval_items_loop_spawned() else frozenset())


def loop_spawn_tags() -> frozenset[str]:
    """The *enumerated* tags expiry bounds and the scorecard gauge counts, right
    now — the exact half of the rule. A mint named after its own session is loop
    output too, so nothing asks this directly: go through `loop_spawn_tag()`,
    which adds the `spawned-by-` prefix this returns alongside.
    """
    return (SPAWN_TAGS | REVIEW_SPAWN_TAGS
            | (EVAL_SPAWN_TAGS if eval_items_loop_spawned() else frozenset()))


def loop_spawn_tag(tags) -> str:
    """Which tag says a loop session wrote this item — or "" if none does.

    The one test for "is this the board's own output?", used by expiry, by the
    scorecard's open self-spawned gauge, and by `board_health`. It answers in
    two ways, because the board is written two ways: an exact hit on the
    enumerated set (which is also how the `youtube-eval` kill switch
    `workers.sources.youtube-digest.loop_spawned` takes effect), or the
    `spawned-by-` prefix that `backlog_write_task` actually stamps — the same
    test the writer applies when it decides a create may be merged into an
    existing item (`app.backlog_tags.is_spawn_tag`). Before that second half
    existed, a mint named after its own session (`spawned-by-task-24`,
    `spawned-by-data-pipeline`, 23 distinct ones on the board on 2026-09-17) was
    merged at write as loop output and read as a human's item ever after, so it
    was neither bounded by expiry nor counted by the gauge (#1160).

    Returns the tag rather than a bool so the expiry ledger row can name which
    minter produced the item it closed.
    """
    spawn_tags = loop_spawn_tags()
    for tag in tags:
        if tag in spawn_tags or is_spawn_tag(tag):
            return tag
    return ""

# How long a self-filed draft may sit untouched before `expire_stale_spawns`
# closes it. The constant is the fallback; the live value is
# `spawn_expiry_days()`, which reads
# `workers.sources.autocode.expire_spawns_after_days`. Every reader — expiry,
# autotriage's skip summary, the scorecard gauge — goes through that function,
# or `over_bound` reports against one bound while expiry runs at another.
# Expiry had never fired by 2026-09-13: the oldest self-spawn was six days old
# against a 30-day bound while 400 open self-spawned items accrued.
SPAWN_EXPIRY_DAYS = 30
# The pre-knob name, from when the number was a quarantine age. Alias only;
# read `spawn_expiry_days()`.
SPAWN_TRIAGE_MIN_AGE_DAYS = SPAWN_EXPIRY_DAYS


def spawn_expiry_days() -> int:
    """The expiry bound in days: config over `SPAWN_EXPIRY_DAYS`. Lazy, and
    fail-open to the constant, so the automod CLI does not need a config."""
    try:
        from app.config import CONFIG
        cfg = (((CONFIG or {}).get("workers") or {}).get("sources") or {}).get("autocode") or {}
        v = int(cfg.get("expire_spawns_after_days") or 0)
        return v if v > 0 else SPAWN_EXPIRY_DAYS
    except Exception:  # noqa: BLE001
        return SPAWN_EXPIRY_DAYS

# Only Lloyd's own board. The backlog is shared: of 52 open items, 3 are Alfie
# (robot firmware) and 1 is on an Architecture board. Those are legitimately
# out of scope for a self-modification pass, and the `board` field says so for
# free — filtering here rather than spending an LLM turn per item to rediscover
# it. Verified against #38 "Alfie — Fix mecanum wheels behavior": triage burned
# a full turn to correctly conclude `not_code`, which the board already knew.
DEFAULT_BOARDS = ("lloyd",)

VERDICTS = ("confirmed", "already_done", "stale", "unverifiable", "not_code")

# Which part of the system a fix would touch. Decides the implementer's route:
# `code` and `frontend` go through a worktree round and the gate; `vault`
# goes through `vault_round` (validate → commit only those paths → revert on
# failure), because the vault is a live tree with no worktree; `mixed` does
# the vault half first. `external` is hardware, robots and third-party
# services — the only things `not_code` still means.
SURFACES = ("code", "frontend", "vault", "mixed", "external")

# An acceptance that opens with this is a contract only a human can execute:
# the fix needs a path the loop may never touch (config.yaml, data/**, .env*,
# pytest.ini, .gitignore, the frontend's build inputs). `select_confirmed`
# skips it rather than spending an implement round discovering it — which is
# exactly what #278 spent nine iterations on before web/src was allowed.
HUMAN_ONLY_PREFIX = "human-only:"


def is_human_only(acceptance) -> bool:
    return str(acceptance or "").strip().lower().startswith(HUMAN_ONLY_PREFIX)

# Verdicts that retire an item rather than producing work. Both are wins.
RETIRING = {"already_done", "stale"}

# Not a verdict: the triage turn ran out of iteration budget before reaching
# one. Recorded so the attempt is visible, but it is NOT terminal — the item
# comes back for another pass with more room. Before this existed, running out
# of budget produced no verdict block, which was recorded as `unverifiable`,
# and `unverifiable` is terminal: the hardest items on the board were being
# retired permanently on first contact, for a reason indistinguishable from
# "states no checkable claim". The three triages driven by hand used 45, 65 and
# 76 iterations against the worker's budget of 30; all three would have been.
INCOMPLETE = "incomplete"
MAX_INCOMPLETE_ATTEMPTS = 2


# The same verdict, as a machine contract. Built from VERDICTS/SURFACES rather
# than restated, so a new verdict cannot be added in one place and forgotten in
# the other — the grammar and the validator have to be the same list.
#
# Length clamps stay in Python (`parse_verdict`): a `maxLength` in the schema
# is enforced by the decoder, which would make the model stop mid-sentence at
# the limit rather than write a shorter one. INCOMPLETE is deliberately absent:
# it is not a verdict, it is the record of a turn that ran out of budget, and
# the finalizer never runs on such a turn anyway.
# What an implement round says about the acceptance check when its turn ends.
# `met` closes the item once the promotion settles; `not_met` and `deferred`
# leave it open and say why; `unnecessary` and `rejected` close it with no
# landing. Built from this tuple, not restated, for the reason the triage
# schema is: one list, or a new value lands in the grammar and not in the
# validator.
#
# `rejected` (2026-09-16, Alan's rule): every backlog item is a proposal for
# research and eval, and deployment happens only when the measurement says it
# improves things. Before this the loop had no honest exit for "built it,
# measured it, no gain" — a round either forced a landing or spent its
# attempts as `not_met`, was re-triaged, and finally parked for a human. A
# rejection with evidence closes the item cleanly; the loop is judged on items
# *resolved*, not items landed.
ACCEPTANCE_OUTCOMES = ("met", "not_met", "deferred", "unnecessary", "rejected")
# Verdicts on the ITEM, kept as stated whatever the clauses say.
ITEM_VERDICT_OUTCOMES = ("unnecessary", "rejected")
# Per clause. No `unnecessary`/`rejected`: those are verdicts on the item.
CLAUSE_OUTCOMES = ("met", "not_met", "deferred")

IMPLEMENT_OUTCOME_SCHEMA: dict = {
    "type": "object",
    "title": "backlog_implement_outcome",
    "properties": {
        "landed": {"type": "boolean",
                   "description": ("Did this turn call automod_land (or automod_vault_land) "
                                   "on a change that passed the gate?")},
        "acceptance": {"type": "string", "enum": list(ACCEPTANCE_OUTCOMES),
                       "description": ("met: the acceptance check recorded at triage is now "
                                       "true. not_met: it is not, and this round did not "
                                       "make it so. deferred: it cannot be judged until "
                                       "something else happens — name it in deferred_to. "
                                       "unnecessary: the work is not needed after all (the "
                                       "premise no longer holds, or it is already true) and "
                                       "the item should close without a landing. rejected: "
                                       "you built or measured it and the evidence says it does "
                                       "not improve things (an eval no better, cost above the "
                                       "gain, a design that does not fit) — the item closes as "
                                       "tried-and-rejected; put the measurement in summary. "
                                       "When clause_outcomes is non-empty and this is neither "
                                       "unnecessary nor rejected, it is derived from them.")},
        "clause_outcomes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "clause": {"type": "integer", "description": "1-based index into the acceptance clauses."},
                "outcome": {"type": "string", "enum": list(CLAUSE_OUTCOMES)},
                "evidence": {"type": "string",
                             "description": "The test node id or file:line that shows it, one line."},
                "deferred_to": {"type": "array", "items": {"type": "integer"},
                                "description": "Ids this clause waits on. Empty unless deferred."},
            },
            "required": ["clause", "outcome", "evidence", "deferred_to"],
            "additionalProperties": False,
        }, "description": ("One entry per acceptance clause, in order. Empty only when "
                           "acceptance is unnecessary.")},
        "deferred_to": {"type": "array", "items": {"type": "integer"},
                        "description": ("Backlog ids that must close before the acceptance "
                                        "can be judged. Empty unless acceptance is deferred.")},
        "summary": {"type": "string",
                    "description": "One sentence: what landed, or why nothing did."},
        "spawned": {"type": "array", "items": {"type": "integer"},
                    "description": "Backlog ids filed during this round."},
        "human_paths": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path you could not change."},
                "reason": {"type": "string", "description": "One sentence: what needed to change there, and why."},
            },
            "required": ["path", "reason"],
            "additionalProperties": False,
        }, "description": ("Paths this change needed but the loop may never write "
                           "(denied or outside the writable set). Leaving one out and "
                           "landing the rest is correct; hiding it is not. A `met` "
                           "landing still closes, and the item carries needs-human so "
                           "the path is not lost (#1210). Empty when there are none.")},
    },
    "required": ["landed", "acceptance", "clause_outcomes", "deferred_to", "summary", "spawned"],
    "additionalProperties": False,
}


def _ints(v) -> list[int]:
    out: list[int] = []
    for x in (v or []):
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def parse_outcome(structured) -> dict | None:
    """The finalizer's object, validated and clamped, or None if unusable.

    Clamped here rather than in the grammar for the reason the triage schema
    carries no `maxLength`: a guided decoder stops mid-sentence at a limit
    rather than writing something shorter.

    The overall `acceptance` is DERIVED when clauses are present — all met →
    `met`; any not_met → `not_met`; else `deferred` — and **a deferral that
    names nothing is `not_met`**. #544 declared `deferred` with
    `deferred_to: []`; `close_settled_items` wrote "deferred to an unnamed
    follow-up; close this when that closes" and `desired_statuses` parked it
    `in_progress` forever, because nothing ever re-read an empty list.
    `unnecessary` is a verdict on the item and is kept as stated.
    """
    if not isinstance(structured, dict):
        return None
    acceptance = str(structured.get("acceptance") or "")
    if acceptance not in ACCEPTANCE_OUTCOMES:
        return None

    clauses: list[dict] = []
    for raw in (structured.get("clause_outcomes") or []):
        if not isinstance(raw, dict):
            continue
        oc = str(raw.get("outcome") or "").strip().lower()
        if oc not in CLAUSE_OUTCOMES:
            continue
        try:
            idx = int(raw.get("clause") or (len(clauses) + 1))
        except (TypeError, ValueError):
            idx = len(clauses) + 1
        clauses.append({"clause": idx, "outcome": oc,
                        "evidence": " ".join(str(raw.get("evidence") or "").split())[:300],
                        "deferred_to": _ints(raw.get("deferred_to"))})
    deferred_to = _ints(structured.get("deferred_to"))
    for c in clauses:
        deferred_to += [i for i in c["deferred_to"] if i not in deferred_to]

    if acceptance not in ITEM_VERDICT_OUTCOMES and clauses:
        outcomes = {c["outcome"] for c in clauses}
        if "not_met" in outcomes:
            acceptance = "not_met"
        elif outcomes == {"met"}:
            acceptance = "met"
        else:
            acceptance = "deferred"
    if acceptance == "deferred" and not deferred_to:
        acceptance = "not_met"

    human_paths = []
    for raw in (structured.get("human_paths") or []):
        if not isinstance(raw, dict):
            continue
        path = " ".join(str(raw.get("path") or "").split())[:300]
        if not path:
            continue
        human_paths.append({
            "path": path,
            "reason": " ".join(str(raw.get("reason") or "").split())[:300],
        })

    return {"landed": bool(structured.get("landed")), "acceptance": acceptance,
            "clause_outcomes": clauses,
            "deferred_to": deferred_to,
            "summary": " ".join(str(structured.get("summary") or "").split())[:400],
            "spawned": _ints(structured.get("spawned")),
            "human_paths": human_paths}


def apply_post_landing(outcome: dict | None, post_landing: list[int]) -> dict | None:
    """Re-read an outcome with the review's `post_landing` clauses honoured.

    The review rung, not the implementer, decides that a clause is observable
    only after landing — the implementer is inside the round and cannot know.
    So the implementer honestly reports `deferred` or `not_met` for such a
    clause and this re-reads those indices as met, tagging the outcome so
    `close_settled_items` knows to hold the item open for a human rather than
    closing it.

    Deliberately NOT applied to a clause the implementer called `met`: that
    claim stands on its own and needs no rescue.
    """
    if not outcome or not post_landing:
        return outcome
    idx = {int(i) for i in post_landing}
    changed = []
    clauses = []
    for c in (outcome.get("clause_outcomes") or []):
        if int(c.get("clause") or 0) in idx and c.get("outcome") in ("deferred", "not_met"):
            c = {**c, "outcome": "met", "post_landing": True}
            changed.append(int(c["clause"]))
        clauses.append(c)
    if not changed:
        return outcome
    out = {**outcome, "clause_outcomes": clauses, "post_landing_clauses": changed}
    outcomes = {c["outcome"] for c in clauses}
    if "not_met" in outcomes:
        out["acceptance"] = "not_met"
    elif outcomes == {"met"}:
        out["acceptance"] = "met"
    else:
        out["acceptance"] = "deferred"
    return out


def unmet_clauses(outcome: dict | None) -> list[int]:
    """Clause indices an outcome reports as not met."""
    return [int(c["clause"]) for c in ((outcome or {}).get("clause_outcomes") or [])
            if c.get("outcome") == "not_met"]


TRIAGE_VERDICT_SCHEMA: dict = {
    "type": "object",
    "title": "backlog_triage_verdict",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "surface": {"type": "string", "enum": list(SURFACES)},
        "check": {"type": "string",
                  "description": "One concrete, runnable check that decides it."},
        "evidence": {"type": "string",
                     "description": "What was measured, with paths and line numbers."},
        "acceptance": {"type": "string",
                       "description": ("For `confirmed`: the contract the implementer "
                                       "is held to. Prefix with 'human-only:' when the "
                                       "fix needs a path the loop may never touch. "
                                       "Empty otherwise.")},
        "acceptance_clauses": {"type": "array", "items": {"type": "string"},
                               "description": ("For `confirmed`: the same contract split into "
                                               "separately checkable clauses, each one thing a "
                                               "test can pin and ending with the test file that "
                                               "pins it (`— tests/<file>.py`; for a `vault` "
                                               "surface the vault path that shows it, never a "
                                               "test), in order. The "
                                               "implementer reports "
                                               "per clause and the review rung grades per "
                                               "clause. Empty otherwise.")},
        "human_clauses": {"type": "array", "items": {"type": "string"},
                          "description": ("For `confirmed`: conditions only a person can "
                                          "satisfy — an audit, a sign-off, a decision, a "
                                          "measurement that needs real traffic. Never in "
                                          "acceptance_clauses: the implementer is not asked "
                                          "to fake them and the reviewer does not grade them; "
                                          "the item waits on a human for them after the code "
                                          "lands. Empty when none.")},
        "spawned": {"type": "array", "items": {"type": "integer"},
                    "description": "Backlog ids filed during this triage."},
    },
    "required": ["verdict", "surface", "check", "evidence", "acceptance",
                 "acceptance_clauses", "human_clauses", "spawned"],
    "additionalProperties": False,
}

# Group triage: one turn over a cluster of related items. Per-item verdicts
# are the retiring ones plus three that only make sense with siblings in
# view — `duplicate_of` (closed, pointing at the survivor), `fold` (into the
# umbrella this turn files) and `keep` (distinct work; back to the single
# pool). Built from RETIRING, not restated, for the reason the single schema
# is built from VERDICTS.
GROUP_VERDICTS = ("fold", "duplicate_of", "keep") + tuple(sorted(RETIRING))
# The ledger verdict a folded member gets. Outside VERDICTS on purpose, so
# `triaged_ids` does not count it as judged — like INCOMPLETE, it is a state
# of the item, not a conclusion about its premise.
FOLDED = "folded"

GROUP_TRIAGE_SCHEMA: dict = {
    "type": "object",
    "title": "backlog_group_triage_verdict",
    "properties": {
        "items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "item_id": {"type": "integer"},
                "verdict": {"type": "string", "enum": list(GROUP_VERDICTS)},
                "duplicate_of": {"type": "integer",
                                 "description": "The surviving item's id; 0 unless verdict is duplicate_of."},
                "evidence": {"type": "string",
                             "description": "One or two sentences with the path/line or commit that decides it."},
            },
            "required": ["item_id", "verdict", "duplicate_of", "evidence"],
            "additionalProperties": False,
        }, "description": "One entry per item in the cluster, every item listed."},
        "umbrella": {"type": "object", "properties": {
            "item_id": {"type": "integer",
                        "description": "Id backlog_write_task returned for the umbrella; 0 when nothing was folded."},
            "members": {"type": "array", "items": {"type": "integer"}},
            "surface": {"type": "string", "enum": list(SURFACES)},
            "check": {"type": "string"},
            "evidence": {"type": "string"},
            "acceptance": {"type": "string"},
            "acceptance_clauses": {"type": "array", "items": {"type": "string"}},
        }, "required": ["item_id", "members", "surface", "check", "evidence",
                        "acceptance", "acceptance_clauses"],
            "additionalProperties": False},
        "spawned": {"type": "array", "items": {"type": "integer"},
                    "description": "Backlog ids filed or merged into during this triage, umbrella excluded."},
    },
    "required": ["items", "umbrella", "spawned"],
    "additionalProperties": False,
}

# The clause budget of a contract triage writes. 12 -> 6 on 2026-09-14: the
# review rung refused 109 of 147 reviews, and at a per-clause not-met rate of
# 13% six clauses pass together ~43% of the time and twelve ~19%. First
# reviews by clause count: <=4 promoted 3/16, 5-8 21/52, 9+ 3/11 — and 62
# confirmations in the week carried 9-12. The single-item prompt asks for at
# most SINGLE_MAX_CLAUSES and the group prompt for MAX_CLAUSES; both parse
# paths and `record_verdict` cap at MAX_CLAUSES and the verdict row records
# `clauses_dropped`. The review rung assumes no count.
MAX_CLAUSES = 6
SINGLE_MAX_CLAUSES = 5
# What a READER of a contract already on disk keeps. Not MAX_CLAUSES: 56
# umbrellas confirmed before the cut carry 8-12, and truncating them on read
# would grade a round against half its contract, close the item on the half
# it met — and `amend_clause` would write the truncated half back.
READ_MAX_CLAUSES = 12
CLAUSE_MAX_CHARS = 600


def clean_clauses(values, *, limit: int = READ_MAX_CLAUSES) -> list[str]:
    """Clauses as a bounded list of non-placeholder strings. `limit` is
    `MAX_CLAUSES` where a new contract is written, else the read bound."""
    out: list[str] = []
    for v in (values or []):
        s = acceptance_text(v)
        if s and s not in out:
            out.append(s[:CLAUSE_MAX_CHARS])
        if len(out) >= limit:
            break
    return out


def cap_new_clauses(values) -> tuple[list[str], list[str]]:
    """`(clauses, dropped)` for a contract being written: cleaned, capped at
    `MAX_CLAUSES`, and the real clauses that fell past the cap — kept as text,
    because the prose ACCEPTANCE still states them and a count alone would
    lose what they said."""
    every = clean_clauses(values, limit=10_000)
    return every[:MAX_CLAUSES], every[MAX_CLAUSES:]


# A clause that can only be OBSERVED once the change is live. The prompt tells
# triage to put these under HUMAN_CLAUSES, and this is the backstop for when it
# does not: #859 was refused twice on "needs a day of post-change traffic" with
# its mechanism complete on both commits, and parked.
#
# Deliberately narrow. A clause mentioning "traffic" in passing is not one of
# these; the shapes here all say the evidence arrives with TIME, which is
# exactly what a pre-landing gate cannot wait for. A false positive moves a
# gradeable clause out of the graded contract, which is worse than a false
# negative — the review rung's own `post_landing` verdict catches what this
# misses.
POST_LANDING_RX = re.compile(
    r"(?:"
    r"after (?:it|the change|this) (?:has )?land(?:s|ed|ing)"
    r"|post[- ]landing"
    r"|once (?:it|the change|this) is live"
    r"|(?:a |one )?(?:day|week|24 hours|48 hours) of (?:real )?traffic"
    r"|over (?:a|the) (?:next )?(?:day|week|month)"
    r"|(?:the )?next nightly run"
    r"|in production over"
    r")",
    re.IGNORECASE,
)


def split_post_landing_clauses(clauses) -> tuple[list[str], list[str]]:
    """`(graded, post_landing)` — clauses a gate can judge, and the rest.

    Applied at `record_verdict`, so a triage verdict that put a
    can-only-be-seen-later clause in ACCEPTANCE_CLAUSES has it moved to
    `human_clauses` before any round is held to it.
    """
    graded: list[str] = []
    later: list[str] = []
    for c in (clauses or []):
        (later if POST_LANDING_RX.search(str(c or "")) else graded).append(c)
    return graded, later


_CLAUSE_LINE = re.compile(r"^\s*(\d{1,2})[.)]\s+(.*\S)\s*$")


def split_clause_lines(text: str, *, limit: int = READ_MAX_CLAUSES) -> list[str]:
    """Numbered lines (`1. …`, `2) …`) into clauses; unnumbered prose is one
    clause. A `-` or `none` placeholder is no clause at all."""
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    numbered = [m.group(2) for m in (_CLAUSE_LINE.match(ln) for ln in lines) if m]
    if numbered:
        return clean_clauses(numbered, limit=limit)
    return clean_clauses([" ".join(str(text or "").split())], limit=limit)


def acceptance_clauses_of(event: dict | None, frontmatter: dict | None = None) -> list[str]:
    """The clauses an implementer is held to, from wherever they were recorded.

    Front matter first (written by `record_verdict` since clauses existed),
    then the triage event, then the prose acceptance as a single clause — an
    item confirmed before clauses existed has prose only, and refusing it
    would block the whole current pool.
    """
    fm = frontmatter or {}
    ev = event or {}
    for source in (fm.get("acceptance_clauses"), ev.get("acceptance_clauses")):
        if isinstance(source, list):
            cleaned = clean_clauses(source)
            if cleaned:
                return cleaned
    prose = acceptance_text(ev.get("acceptance"))
    if not prose:
        return []
    if is_human_only(prose):
        return [prose]
    return split_inline_lettered(prose) or [prose]


def human_clauses_of(event: dict | None, frontmatter: dict | None = None) -> list[str]:
    """Conditions only a person can satisfy, from the item's front matter or
    the triage event. No prose fallback: absent means none."""
    fm = frontmatter or {}
    ev = event or {}
    for source in (fm.get("human_clauses"), ev.get("human_clauses")):
        if isinstance(source, list):
            cleaned = clean_clauses(source)
            if cleaned:
                return cleaned
    return []


def human_clauses_for_item(path: Path | None, event: dict | None) -> list[str]:
    """`human_clauses_of` read off the item file, for callers holding an
    `Item` rather than its front matter."""
    fm: dict = {}
    if path is not None and Path(path).exists():
        try:
            fm, _ = _split_frontmatter(Path(path).read_text(encoding="utf-8"))
        except OSError:
            fm = {}
    return human_clauses_of(event, fm)


# ── Clause amendments ───────────────────────────────────────────────────────
#
# The review rung may judge a clause `unsatisfiable`: no diff can meet it as
# written. The author's only moves used to be a retry that could not succeed
# or an abort; now it may amend exactly that clause, and the amendment holds
# only if the NEXT review ratifies it. The second reader's judgment, not the
# author's — a loop that rewrote its own acceptance unchecked would be one
# that could declare anything met. Records live on the item, so a human
# reading the board sees what the contract was and what it became.
AMENDMENTS_KEY = "clause_amendments"


def _item_path(item_id: int) -> Path | None:
    paths = sorted(BACKLOG_DIR.glob(f"{int(item_id)}-*.md"))
    return paths[0] if paths else None


def pending_amendments(frontmatter: dict | None, round_id: str | None = None) -> list[dict]:
    """Unratified amendments on the item; with `round_id`, only that round's.

    An amendment is a move inside ONE round: the round's review judged the
    clause unsatisfiable, the round amended it, the round's next review
    ratifies it. Read item-wide, a pending record from a round that ended
    before its review ran reopened the question for every later round of the
    item — and a round holding one skips the same-head check, patch-id reuse
    and the two-attempt cap. #860 bought a third graded attempt ("3/2") that
    way on an amendment made in SM_20260911_183542, two days earlier.
    """
    return [dict(a) for a in ((frontmatter or {}).get(AMENDMENTS_KEY) or [])
            if isinstance(a, dict) and a.get("state") == "pending"
            and (round_id is None or str(a.get("round_id") or "") == str(round_id))]


def orphan_stale_amendments(item_id: int, round_id: str) -> list[int]:
    """Close every pending amendment another round left on the item.

    `state: orphaned`, and the clause text goes back to `was`: an amendment
    no review ratified is not part of the contract — the same reading a
    refusal gets. The activity line names the old round and quotes the
    amended text, so the round now open can re-amend it in one call if its
    own review calls the clause unsatisfiable again. Idempotent; returns
    the clause indices orphaned.
    """
    path = _item_path(item_id)
    if path is None:
        return []
    fm, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    records = fm.get(AMENDMENTS_KEY) or []
    clauses = list(fm.get("acceptance_clauses") or [])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    orphaned: list[int] = []
    quoted: list[str] = []
    for a in records:
        if not isinstance(a, dict) or a.get("state") != "pending":
            continue
        if str(a.get("round_id") or "") == str(round_id):
            continue
        a["state"] = "orphaned"
        a["settled_at"] = stamp
        a["note"] = f"its round ended unratified; orphaned when round {round_id} was reviewed"
        i = int(a.get("clause") or 0) - 1
        if 0 <= i < len(clauses) and a.get("was"):
            clauses[i] = a["was"]
        orphaned.append(i + 1)
        quoted.append(f"clause {i + 1} from round {a.get('round_id')}: {a.get('now')}")
    if not orphaned:
        return []
    fm["acceptance_clauses"] = clauses
    log = list(fm.get("activity_log") or [])
    log.append(f"**{stamp}** — automod: {len(orphaned)} unratified amendment(s) from an earlier "
               f"round orphaned and the clause text restored before round {round_id} was "
               f"reviewed; amend again if the review still calls it unsatisfiable. "
               + " — ".join(quoted))
    fm["activity_log"] = log
    fm["updated"] = stamp
    _write_item(path, fm, body)
    return orphaned


def last_graded_review(ledger: Path, round_id: str) -> dict | None:
    rows = [d for d in _ledger_events(ledger, "review", require_item=False)
            if str(d.get("round_id") or "") == str(round_id) and d.get("ok")]
    return rows[-1] if rows else None


def _write_item(path: Path, fm: dict, body: str) -> None:
    path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body}", encoding="utf-8")


def amend_clause(item_id: int, clause: int, text: str, reason: str, *,
                 round_id: str, ledger: Path | None = None) -> dict:
    """Replace one acceptance clause the review rung judged unsatisfiable.

    Refuses (ValueError) for a clause the last graded review of `round_id`
    did not mark `unsatisfiable`, an index off the list, empty text or
    reason, or a clause already amended and awaiting ratification. Returns
    the amendment record, state `pending`.
    """
    from scripts.automod import state as S
    ledger = ledger or S.LEDGER_PATH
    path = _item_path(item_id)
    if path is None:
        raise ValueError(f"no backlog item #{item_id} on disk")
    fm, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    clauses = clean_clauses(fm.get("acceptance_clauses") or [])
    if not clauses:
        raise ValueError(f"item #{item_id} has no acceptance_clauses on disk to amend")
    try:
        idx = int(clause)
    except (TypeError, ValueError):
        raise ValueError(f"clause must be an integer, got {clause!r}")
    if not 1 <= idx <= len(clauses):
        raise ValueError(f"clause {idx} is off the list (item #{item_id} has {len(clauses)})")
    text = " ".join(str(text or "").split())[:CLAUSE_MAX_CHARS]
    reason = " ".join(str(reason or "").split())[:CLAUSE_MAX_CHARS]
    if not text or not reason:
        raise ValueError("both the amended clause text and a reason are required")
    review = last_graded_review(ledger, round_id)
    verdicts = {int(c.get("clause") or 0): str(c.get("verdict") or "")
                for c in ((review or {}).get("clauses") or []) if isinstance(c, dict)}
    if verdicts.get(idx) != "unsatisfiable":
        raise ValueError(f"the review rung has not judged clause {idx} unsatisfiable in round "
                         f"{round_id}; only a clause the grader marked unsatisfiable may be amended")
    if any(int(a.get("clause") or 0) == idx for a in pending_amendments(fm)):
        raise ValueError(f"clause {idx} is already amended and awaiting the next review")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    old = clauses[idx - 1]
    clauses[idx - 1] = text
    fm["acceptance_clauses"] = clauses
    rec = {"clause": idx, "was": old, "now": text, "reason": reason,
           "round_id": str(round_id), "at": stamp, "state": "pending"}
    fm[AMENDMENTS_KEY] = list(fm.get(AMENDMENTS_KEY) or []) + [rec]
    log = list(fm.get("activity_log") or [])
    log.append(f"**{stamp}** — automod round {round_id} amended clause {idx} (review judged it "
               f"unsatisfiable; awaiting ratification by the next review). Was: {old} — Now: "
               f"{text} — Reason: {reason}")
    fm["activity_log"] = log
    fm["updated"] = stamp
    _write_item(path, fm, body)
    return rec


def settle_amendments(item_id: int, round_id: str, *, ratified: bool,
                      note: str = "") -> list[int]:
    """Mark this round's pending amendments ratified or refused. A refusal
    restores the clause text. Returns the clause indices settled."""
    path = _item_path(item_id)
    if path is None:
        return []
    fm, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    clauses = list(fm.get("acceptance_clauses") or [])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    settled: list[int] = []
    for a in fm.get(AMENDMENTS_KEY) or []:
        if not isinstance(a, dict) or a.get("state") != "pending":
            continue
        if str(a.get("round_id") or "") != str(round_id):
            continue
        a["state"] = "ratified" if ratified else "refused"
        a["settled_at"] = stamp
        if note:
            a["note"] = note[:600]
        i = int(a.get("clause") or 0) - 1
        if not ratified and 0 <= i < len(clauses):
            clauses[i] = a.get("was") or clauses[i]
        settled.append(i + 1)
    if not settled:
        return []
    fm["acceptance_clauses"] = clauses
    log = list(fm.get("activity_log") or [])
    verb = "ratified" if ratified else "refused (clause text restored)"
    log.append(f"**{stamp}** — automod review {verb} the amendment of clause(s) "
               f"{', '.join(map(str, settled))} from round {round_id}"
               + (f": {note[:300]}" if note else ""))
    fm["activity_log"] = log
    fm["updated"] = stamp
    _write_item(path, fm, body)
    return settled


_LETTERED = re.compile(r"\(([a-h]|\d{1,2})\)\s+")


def split_inline_lettered(text: str) -> list[str]:
    """`(a) … ; (b) … ; (c) …` written as one paragraph, into clauses.

    Eleven of the first forty confirmed acceptances enumerated sub-clauses
    inline this way — #544's `(a)`–`(e)` among them — before a clauses field
    existed. Fewer than two markers is prose, not a list, and comes back
    empty so the caller keeps the whole text as one clause.
    """
    marks = list(_LETTERED.finditer(str(text or "")))
    if len(marks) < 2:
        return []
    parts: list[str] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        seg = text[m.end():end].strip().rstrip(";,").strip()
        seg = re.sub(r"\s+(and|plus)$", "", seg)
        if seg:
            parts.append(seg)
    return clean_clauses(parts)


@dataclass
class Item:
    path: Path
    id: int
    name: str
    status: str
    priority: str
    created: str
    body: str
    board: str = ""
    tags: list[str] = field(default_factory=list)
    # Relations. `parent` is the item this one was split from or found while
    # implementing (persisted from the prose first line by the clustering
    # pass); `group` is the umbrella a member was folded into; `members` are
    # the items an umbrella consolidates. None of these is a status.
    parent: int | None = None
    group: int | None = None
    members: list[int] = field(default_factory=list)
    # The sweep's rank (`record_sweep_verdicts`): `worth` in WORTH_LEVELS,
    # `size` in SIZE_LEVELS, both "" until a sweep has read the item. Read by
    # `rank_key`, which orders every pool. `clause_count` is the contract on
    # disk, so the implement order can prefer a small one without re-reading
    # the file.
    worth: str = ""
    size: str = ""
    clause_count: int = 0

    @property
    def age_days(self) -> int:
        try:
            created = datetime.fromisoformat(self.created.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - created).days
        except (ValueError, AttributeError):
            return 0


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file into (front matter, body).

    Parses through `_YamlLoader` — libyaml's `CSafeLoader` where libyaml is
    importable, `SafeLoader` where it is not (see the import at the top of the
    file). `yaml.YAMLError` still catches both: the C scanner raises
    subclasses of it too, which `test_malformed_front_matter_behaves_the_same`
    pins, because a loader swap that turned a malformed item into an
    exception instead of an empty dict would take the whole board walk down.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.load(parts[1], Loader=_YamlLoader) or {}
    except yaml.YAMLError:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), parts[2]


def load_item(path: Path) -> Item | None:
    try:
        fm, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    m = re.match(r"^(\d+)[-_]", path.name)
    if not m:
        return None
    name = ""
    for line in body.splitlines():
        if line.startswith("# "):
            name = line[2:].strip()
            break
    return Item(
        path=path, id=int(m.group(1)), name=name or path.stem,
        status=str(fm.get("status", "draft")),
        priority=_level(fm.get("priority"), PRIORITY_LEVELS) or DEFAULT_PRIORITY,
        created=str(fm.get("created", "")), body=body,
        board=str(fm.get("board", "") or ""),
        tags=normalize_tags(fm.get("tags")),
        parent=_int_or_none(fm.get("parent")),
        group=_int_or_none(fm.get("group")),
        members=_ints(fm.get("members")),
        worth=_level(fm.get("worth"), WORTH_LEVELS),
        size=_level(fm.get("size"), SIZE_LEVELS),
        clause_count=len(fm["acceptance_clauses"]) if isinstance(fm.get("acceptance_clauses"), list) else 0,
    )


def _level(v, levels: tuple[str, ...]) -> str:
    s = str(v or "").strip().lower()
    return s if s in levels else ""


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


def update_frontmatter(path: Path, updates: dict, *, activity: str = "",
                       add_tags: tuple[str, ...] = (), remove_tags: tuple[str, ...] = ()) -> bool:
    """Set frontmatter keys on an item without moving its status.

    The one writer for the relation keys (`parent`, `group`, `members`,
    `duplicate_of`). Refuses a file whose YAML did not parse: the other
    writers here rewrite the file from the parsed dict, and a dict that came
    back empty because the YAML was broken would be written back as a file
    with its frontmatter destroyed — the MCP writer guards the same case
    with `_yaml_broken`. A key set to None is removed.
    """
    text = path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)
    if text.startswith("---") and not fm:
        return False
    changed = False
    for k, v in (updates or {}).items():
        if v is None:
            if k in fm:
                del fm[k]
                changed = True
        elif fm.get(k) != v:
            fm[k] = v
            changed = True
    raw_tags = fm.get("tags")
    tags = normalize_tags(raw_tags)
    new_tags = [t for t in tags if t not in remove_tags] + [t for t in add_tags if t not in tags]
    if new_tags != tags or (add_tags or remove_tags) and not isinstance(raw_tags, list):
        fm["tags"] = new_tags
        changed = True
    if not changed and not activity:
        return False
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    if activity:
        log = list(fm.get("activity_log") or [])
        log.append(f"**{stamp}** — {activity}")
        fm["activity_log"] = log
    fm["updated"] = stamp
    path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body}", encoding="utf-8")
    return True


def all_items(boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
              backlog_dir: Path | None = None) -> list[Item]:
    """Every item on `boards`, whatever its status — including a status
    outside `PIPELINE_STATUSES`, which is the one thing `open_items` cannot
    show you and the reason the rescue below needs its own walk.

    `backlog_dir` defaults to the module's `BACKLOG_DIR`, which is bound at
    import; a reader that resolves the vault at call time (the dashboard)
    passes its own."""
    wanted = {b.lower() for b in boards} if boards else None
    out = []
    for path in sorted((backlog_dir or BACKLOG_DIR).glob("*.md")):
        item = load_item(path)
        if not item:
            continue
        if wanted is not None and item.board.lower() not in wanted:
            continue
        out.append(item)
    return out


def open_items(boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
               backlog_dir: Path | None = None) -> list[Item]:
    """Open items, restricted to `boards` unless it is None."""
    return [i for i in all_items(boards, backlog_dir=backlog_dir) if i.status in OPEN_STATUSES]


def _ledger_rows(ledger: Path) -> list[dict]:
    """Every row of the promotions ledger, decoded at most once per change.

    Delegates to `scripts.automod.state.ledger_rows`, which caches the decoded
    file against its (mtime_ns, size, inode). The reason: one cold
    `_backlog()` + `_automod()` read the file **57 times** and ran
    `json.loads` 331,569 times over a 6,317,811-byte ledger (measured
    2026-09-17 04:44Z; the item's filed numbers at triage were 55 calls,
    3.79 s cumulative and 318,670 loads over 6,286,192 bytes — the same
    defect, a 45 KB younger file). `board_health`, `board_flow` and
    `scorecard.compute` ask ~55 filtered questions per cycle (triage verdicts,
    gate findings, review records, spawn events, sweep verdicts, grouping), and
    each question used to decode all 4,808 lines again. Post-fix that cycle is
    1 read; `tests/test_backlog_ledger_cache.py` asserts it.

    The rows are shared read-only objects: filter and copy, do not mutate.
    """
    from scripts.automod import state as S

    return S.ledger_rows(Path(ledger))


def _ledger_read_count(ledger: Path) -> int:
    """Times this process asked the ledger for its rows. The instrument
    `tests/test_backlog_ledger_cache.py` measures; see `state.ledger_read_count`.
    """
    from scripts.automod import state as S

    return S.ledger_read_count(Path(ledger))


def _ledger_cache_clear() -> None:
    """Forget the decoded ledger and the read counters. For tests, and for a
    ledger replaced out-of-band by something that did not go through
    `state.append_event`."""
    from scripts.automod import state as S

    S.ledger_rows_reset()


def _ledger_events(ledger: Path, event: str, *, require_item: bool = True) -> list[dict]:
    """Rows of one event type. `require_item=False` for the event types that
    are about a round rather than an item — `gate` carries a `round_id` and no
    `item_id`, and the default filter drops it silently.

    Filters `_ledger_rows`, so the file is read and decoded at most once per
    change no matter how many event types ask. The predicates are exactly the
    ones this function applied when it decoded the file itself: a missing ledger
    is `[]`, a malformed line is skipped, not raised. One deliberate hardening:
    a line that parses as JSON but is not an object used to be returned and then
    crash its caller on `.get`; it is now dropped by the shared decoder.
    """
    out: list[dict] = []
    for d in _ledger_rows(ledger):
        if d.get("event") != event:
            continue
        if require_item and d.get("item_id") is None:
            continue
        out.append(d)
    return out


_ACCEPTANCE_PLACEHOLDERS = {"none", "n/a", "na", "not applicable", "null", "nil", "no"}


def acceptance_text(value) -> str:
    """The acceptance check as recorded, or "" when the model wrote a placeholder.

    The first cut of the verdict prompt spelled the not-confirmed case as
    `else: ->`, and the model copied the template's own closing bracket
    verbatim: #229 was recorded with acceptance `->`. The guard in
    `select_confirmed` did `.strip("-")`, which leaves `>` — truthy — so a
    `confirmed` verdict written the same way would have handed the
    implementer `>` as its contract. A check is only a check if it has words
    in it, and a lone placeholder word is not a check either.
    """
    s = " ".join(str(value or "").split()).strip()
    if not re.search(r"[A-Za-z0-9]", s):
        return ""
    if s.strip(" -<>()[].:'\"").lower() in _ACCEPTANCE_PLACEHOLDERS:
        return ""
    return s


_ID_RE = re.compile(r"#?(\d{1,6})\b")


def parse_spawned(value) -> list[int]:
    """Item ids from a `SPAWNED:` value — `#401, #402`, `401 402`, `none`."""
    s = " ".join(str(value or "").split()).strip()
    if not s or s.strip(" -<>()[].:'\"").lower() in _ACCEPTANCE_PLACEHOLDERS:
        return []
    out: list[int] = []
    for m in _ID_RE.finditer(s):
        i = int(m.group(1))
        if i not in out:
            out.append(i)
    return out


def parse_spawned_line(text: str) -> list[int]:
    """The LAST `SPAWNED:` line in a turn's text, for turns with no verdict
    block (the implementer's report)."""
    found = ""
    for line in (text or "")[-8000:].splitlines():
        if line.strip().upper().startswith("SPAWNED:"):
            found = line.strip()[8:]
    return parse_spawned(found)


def existing_ids(ids) -> list[int]:
    """The subset of `ids` that exist on disk, any status. What the model says
    it filed is a claim; the file is the fact. An id the model invented — or
    meant to file and ran out of room before doing — is dropped, and the
    caller records it as unverified rather than as a link to nothing."""
    out: list[int] = []
    for i in ids:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if any(BACKLOG_DIR.glob(f"{i}-*.md")):
            out.append(i)
    return out


def item_by_id(item_id: int) -> Item | None:
    """The item with this id, any status and board, or None."""
    for path in sorted(BACKLOG_DIR.glob(f"{int(item_id)}-*.md")):
        item = load_item(path)
        if item is not None:
            return item
    return None


def max_item_id() -> int:
    """The highest id on disk right now. Taken BEFORE a worker turn so the
    ids it claims afterwards can be split into what it created and what it
    merged into (see `split_claimed`)."""
    top = 0
    for path in BACKLOG_DIR.glob("*.md"):
        m = re.match(r"^(\d+)[-_]", path.name)
        if m:
            top = max(top, int(m.group(1)))
    return top


def split_claimed(claimed, *, id_floor: int, self_id: int) -> tuple[list[int], list[int]]:
    """`(spawned, merged)` from the ids a turn claims under SPAWNED.

    Mechanical, not self-reported: an id above the floor recorded before the
    turn is an item this turn created; an id at or below it is an item that
    already existed — a write-time merge (`backlog_write_task` answered
    `merged_into`), or the model citing an item it found. Both are filed
    findings, neither is a spawn. #370's finished row listed itself and a
    pre-existing #221 as spawns, which is what this separates.
    """
    present = existing_ids(claimed)
    spawned = [i for i in present if i > int(id_floor)]
    merged = [i for i in present if i <= int(id_floor) and i != int(self_id)]
    return spawned, merged


# ── One automatic second life ───────────────────────────────────────────────
#
# A spent attempt parked the item `draft` + `needs-human` and a person swept
# the pile by hand: ~35 items a week, and 42 of the 67 that were reopened
# later landed. The mark is a `backlog_retriage` row. Every reader of an
# item's triage or implement history ignores rows at or before its latest
# mark, so a re-triaged item is untriaged and unattempted again — it goes
# back through triage, gets a new contract (or retires), and has a fresh
# attempt count. `RETRIAGE_CAP` bounds it: the second spend is a human's, as
# it always was. See `retriage_spent_items`.
RETRIAGE_TAG = "re-triage"
RETRIAGE_CAP = 1


def retriage_marks(ledger: Path) -> dict[int, float]:
    """`{item_id: ts}` of each item's latest `backlog_retriage` row."""
    marks: dict[int, float] = {}
    for d in _ledger_events(ledger, "backlog_retriage"):
        i = int(d["item_id"])
        marks[i] = max(marks.get(i, 0.0), float(d.get("ts") or 0))
    return marks


def _after_mark(d: dict, marks: dict[int, float]) -> bool:
    """Whether a row counts: no mark for its item, or written after it."""
    m = marks.get(int(d["item_id"]))
    return m is None or float(d.get("ts") or 0) > m


def implement_history(ledger: Path, marks: dict[int, float] | None = None) -> dict[int, list[dict]]:
    """`{item_id: [backlog_implement rows]}` in ledger order, counting from the
    item's last re-triage mark, with every attempt a landing drain refused
    removed.

    `_run_and_record` writes `started` and then, when `DrainActive` refuses
    the turn before it runs, `skipped`. Read raw, the item's latest row is
    `skipped` — no round, no stop reason — and `implement_outcomes` fell
    through to `spent` for a turn that never ran. That was a quiet loss while
    a spend parked the item; with re-triage a spend discards its contract. So
    a `skipped` row takes its `started` with it.
    """
    marks = retriage_marks(ledger) if marks is None else marks
    out: dict[int, list[dict]] = {}
    for d in _ledger_events(ledger, "backlog_implement"):
        if not _after_mark(d, marks):
            continue
        rows = out.setdefault(int(d["item_id"]), [])
        if str(d.get("phase") or "") == "skipped":
            for i in range(len(rows) - 1, -1, -1):
                if str(rows[i].get("phase") or "") == "started":
                    del rows[i:]
                    break
            continue
        rows.append(d)
    return {iid: rows for iid, rows in out.items() if rows}


def _live_promoted_rounds(ledger: Path) -> set[str]:
    reverted = {str(d.get("commit") or "") for d in
                _ledger_events(ledger, "rollback_succeeded", require_item=False)}
    return {str(d.get("round_id") or "") for d in
            _ledger_events(ledger, "promoted", require_item=False)
            if d.get("round_id") and str(d.get("commit") or "") not in reverted}


def items_with_unfinished_rounds(ledger: Path, *,
                                 history: dict[int, list[dict]] | None = None) -> set[int]:
    """Items whose attempt is not over, whatever `implement_outcomes` reads.

    `implement_outcomes` answers "is another attempt owed"; it reads `spent`
    for a turn in flight (`started` with nothing after), for a round whose
    turn has ended while its detached landing waits for idle, re-gates or —
    with the chamber — waits for a settle, and for a promotion not yet swept.
    None of those may be unfolded or re-triaged: the umbrella would close
    under its own landing, the item would be re-judged under a change about to
    go live. So, an item is mid-round when its latest row is `started`, when
    any of its rounds promoted and was not reverted (the settle sweep or a
    person owns it), or when its latest round is named by `current.json`,
    still has a worktree, or has a live gate or land marker. Fails closed.
    """
    from scripts.automod import state as S, worktree as W
    history = implement_history(ledger) if history is None else history
    live_promoted = _live_promoted_rounds(ledger)
    try:
        current_rid = str((S.read_current() or {}).get("round_id") or "")
    except Exception:  # noqa: BLE001
        current_rid = ""
    out: set[int] = set()
    for iid, rows in history.items():
        if str(rows[-1].get("phase") or "") == "started":
            out.add(iid)
            continue
        rids = [str(r["round_id"]) for r in rows if r.get("round_id")]
        if not rids:
            continue
        if any(r in live_promoted for r in rids) or rids[-1] == current_rid:
            out.add(iid)
            continue
        try:
            if (W.worktree_path(rids[-1]).exists() or S.gate_in_progress(rids[-1])
                    or S.land_in_progress(rids[-1])):
                out.add(iid)
        except Exception:  # noqa: BLE001 — unsure means unfinished
            out.add(iid)
    return out


def triaged_ids(ledger: Path) -> dict[int, str]:
    """{item_id: verdict} for items with a TERMINAL verdict.

    `incomplete` is deliberately not one: an item whose triage ran out of
    budget is not triaged, it is waiting for a bigger budget. Rows before a
    re-triage mark do not count.
    """
    marks = retriage_marks(ledger)
    seen: dict[int, str] = {}
    for d in _ledger_events(ledger, "backlog_triage"):
        verdict = d.get("verdict", "")
        if verdict in VERDICTS and _after_mark(d, marks):
            seen[int(d["item_id"])] = verdict
    return seen


def incomplete_counts(ledger: Path) -> dict[int, int]:
    """How many times each item's triage has run out of budget, since its
    last re-triage mark."""
    marks = retriage_marks(ledger)
    counts: dict[int, int] = {}
    for d in _ledger_events(ledger, "backlog_triage"):
        if d.get("verdict") == INCOMPLETE and _after_mark(d, marks):
            i = int(d["item_id"])
            counts[i] = counts.get(i, 0) + 1
    return counts


def confirmed_verdicts(ledger: Path) -> dict[int, dict]:
    """{item_id: latest `confirmed` triage event}. The event carries the
    ACCEPTANCE the implementer is held to. A confirmation before a re-triage
    mark is a contract the item has already failed once; it does not count."""
    marks = retriage_marks(ledger)
    out: dict[int, dict] = {}
    for d in _ledger_events(ledger, "backlog_triage"):
        if d.get("verdict") == "confirmed" and _after_mark(d, marks):
            out[int(d["item_id"])] = d
    return out


def human_only_ids(ledger: Path) -> dict[int, str]:
    """{item_id: acceptance} for confirmed items only a human can land."""
    return {i: ev.get("acceptance", "") for i, ev in confirmed_verdicts(ledger).items()
            if is_human_only(ev.get("acceptance"))}


# One attempt per item is the rule. What the rule was missing is that some
# rounds never reach a verdict at all — and a round that was refused at the
# door, ran out of clock, or never started is not a judgment on the item.
# Triage has recorded budget exhaustion as `incomplete` since #229 ("the item
# comes back once"); implement never got the same rule, and paid for it six
# times in seventeen attempts.
#
# Each re-offer is capped, because `select_confirmed` takes the OLDEST ready
# item: an uncapped re-offer is re-picked every round for as long as the cause
# persists, starving everything behind it.
EXTERNAL_RETRY_CAP = 3       # a red tree surviving four rounds is an incident
INCOMPLETE_RETRY_CAP = 1     # triage's rule: it comes back once
ROLLED_BACK_RETRY_CAP = 1    # the work is gone with the branch; one redo
# The review rung found the premise sound and the implementation or tests
# short. Two re-offers: the third round is the one where author and grader
# have disagreed twice, and another run will not resolve that — a human will.
REVIEW_RETRY_CAP = 2
# A round LANDED and its own finalizer said a clause was not met. One more
# go, told which clauses; the branch is gone with the landing.
PARTIAL_RETRY_CAP = 1

# Stop reasons that mean the turn ran out of room rather than reaching a
# conclusion. #446 committed 757 lines and was killed by the wall clock
# fourteen seconds before its `automod_gate` call.
INCOMPLETE_STOP_REASONS = {"turn_timeout", "max_turns"}


def _last_gate_per_round(ledger: Path) -> dict[str, dict]:
    """The event that last JUDGED each round: the rung that ended its most
    recent gate run, or a landing that failed after the gate had passed.

    The ladder short-circuits, so the last gate event of a failed run is the
    failing rung and of a passing run is `drill`. A `land_failed` comes later
    than either and supersedes it — a round can pass every rung and still be
    refused at landing because `main` moved again underneath it, and that
    refusal is what decides whether the item's attempt was spent. Ordered by
    `ts`, because the two event types are filtered out of the ledger
    separately and file order is lost in the merge.
    """
    rows = (_ledger_events(ledger, "gate", require_item=False)
            + _ledger_events(ledger, "land_failed", require_item=False))
    rows.sort(key=lambda d: float(d.get("ts") or 0))
    last: dict[str, dict] = {}
    for d in rows:
        rid = str(d.get("round_id") or "")
        if rid:
            last[rid] = d
    return last


def gate_passed_unlanded_rounds(ledger: Path) -> set[str]:
    """Rounds whose last gate run passed every rung and that were never
    promoted and never recorded a `land_failed`: the landing did not happen,
    which is not a judgment on the change.

    Every way a landing can FAIL writes a `land_failed` row. What writes
    nothing is a landing that never ran or was killed outright — #1179's was
    ended by `timeout 120` around a foreground `round land`, the reaper
    closed the round a second after the turn, and the item read as `spent`
    with nine green rungs and a kept branch. `drill` is the ladder's last
    rung, so an ok `drill` as the round's last gate event is a full pass.
    """
    promoted = {str(d.get("round_id") or "") for d in
                _ledger_events(ledger, "promoted", require_item=False)}
    return {rid for rid, ev in _last_gate_per_round(ledger).items()
            if ev.get("event") == "gate" and ev.get("ok") and str(ev.get("rung")) == "drill"
            and rid not in promoted}


def externally_blocked_rounds(ledger: Path) -> set[str]:
    """Rounds whose final gate attempt failed on a condition they did not cause.

    Two rungs set the flag, and both are about the state of the *live tree*
    rather than the diff: `tests` when every failure reproduces at the round's
    base, `preflight` when the live tree is dirty or HEAD has moved under it.
    An empty diff ("no changes to promote") is a preflight failure that IS the
    round's own, and deliberately does not carry the flag.
    """
    return {rid for rid, ev in _last_gate_per_round(ledger).items()
            if not ev.get("ok") and ev.get("external_blocker")}


def _external_budget_left(ledger: Path, item_id: int, round_id: str, attempts: int) -> bool:
    """Whether an externally blocked round may still be re-offered.

    A red tree is capped on the item's attempts, as it always was. A landing
    that failed AFTER the gate passed is capped on its own count instead:
    #1204 (2026-09-17) passed nine rungs on its fifth attempt — three earlier
    ones refused by a skip-marker precheck, one lost to a backend restart —
    and the promoter then gave up waiting for a 37-minute scheduled task. At
    `attempts > EXTERNAL_RETRY_CAP` that read as `spent`, and a finished,
    graded change went back to triage. Attempts other causes consumed say
    nothing about whether the landing will work next time.
    """
    last = _last_gate_per_round(ledger).get(round_id) or {}
    if last.get("event") != "land_failed":
        return attempts <= EXTERNAL_RETRY_CAP
    rounds = {str(d.get("round_id") or "") for d in _ledger_events(ledger, "backlog_implement")
              if int(d.get("item_id") or 0) == int(item_id)}
    failed = sum(1 for d in _ledger_events(ledger, "land_failed", require_item=False)
                 if d.get("external_blocker") and str(d.get("round_id") or "") in rounds)
    return failed <= EXTERNAL_RETRY_CAP


def review_retry_rounds(ledger: Path) -> dict[str, dict]:
    """`{round_id: gate event}` for rounds whose last gate attempt was refused
    by the review rung with the premise judged sound — the implementation or
    its tests fell short. The findings ride the event (`review_findings`)
    because `gate.json` dies with the worktree."""
    return {rid: ev for rid, ev in _last_gate_per_round(ledger).items()
            if not ev.get("ok") and ev.get("review_retry")}


def review_unsound_rounds(ledger: Path) -> dict[str, dict]:
    """Rounds the review rung refused because the ITEM's premise is unsound.
    No retry helps; the item goes to a human with the grader's summary."""
    return {rid: ev for rid, ev in _last_gate_per_round(ledger).items()
            if not ev.get("ok") and ev.get("review_premise_unsound")}


def review_events_for_item(ledger: Path, item_id: int) -> list[dict]:
    """Every `review` event for rounds this item's implement turns opened,
    oldest first — since its last re-triage mark, whose new contract numbers
    its clauses afresh (a clause 2 refused before it is a different clause)."""
    mark = retriage_marks(ledger).get(int(item_id), float("-inf"))
    rids = {str(d.get("round_id") or "") for d in _ledger_events(ledger, "backlog_implement")
            if int(d["item_id"]) == int(item_id) and d.get("round_id")
            and float(d.get("ts") or 0) > mark}
    rids.discard("")
    rows = [d for d in _ledger_events(ledger, "review", require_item=False)
            if str(d.get("round_id") or "") in rids and float(d.get("ts") or 0) > mark]
    rows.sort(key=lambda d: float(d.get("ts") or 0))
    return rows


def review_disagreement(ledger: Path, item_id: int) -> int | None:
    """The clause index two consecutive blocking reviews both flagged, or None.

    The early exit from the author/grader loop: when the same clause comes
    back unmet on two successive reviews, the author has twice believed it
    satisfied and the grader has twice disagreed. A third round re-runs the
    same disagreement; a human resolves it.
    """
    blocking = [d for d in review_events_for_item(ledger, item_id) if d.get("blocking")]
    if len(blocking) < 2:
        return None
    prev, last = blocking[-2], blocking[-1]
    # Two refusals of the SAME commit are one refusal delivered twice — the
    # duplicated gate of 2026-09-11, which parked #578 as a "disagreement"
    # the author had never been shown. Rows without a head predate the field
    # and keep the old reading.
    if prev.get("head") and last.get("head") and prev["head"] == last["head"]:
        return None
    def flagged(ev: dict) -> set[int]:
        # A `partial` the grader wrote as `met` and Python downgraded for
        # missing evidence is not the grader disagreeing with the author —
        # it is the grader agreeing without receipts. #860's clause 8 ("the
        # suite exits 0") was downgraded on both reviews, once for the test
        # node and once for the path, and would have parked the item as a
        # disagreement nobody had.
        return {int(c.get("clause") or 0) for c in (ev.get("clauses") or [])
                if c.get("verdict") in ("unmet", "partial", "unsatisfiable")
                and not c.get("downgraded")}
    both = flagged(last) & flagged(prev)
    both.discard(0)
    return min(both) if both else None


def rolled_back_rounds(ledger: Path) -> set[str]:
    """Rounds that promoted and were then reverted by the guardian.

    Nothing else joins these two facts: the promotion carries the round id and
    the rollback carries only the commit, so an unattended round that landed
    and was reverted spent its item and left no trace on it. Every rollback
    this loop has performed has been a false positive, which is the argument
    for re-offering rather than against it — and the landing deletes the
    branch, so the redo really is a redo. The guardian tag holds the tree.
    """
    reverted = {str(d.get("commit") or "") for d in
                _ledger_events(ledger, "rollback_succeeded", require_item=False)}
    reverted.discard("")
    return {str(d.get("round_id") or "") for d in
            _ledger_events(ledger, "promoted", require_item=False)
            if str(d.get("commit") or "") in reverted and d.get("round_id")}


def _never_ran(ev: dict) -> bool:
    """A `finished` event from a turn that never reported completion.

    `run_prompt_in_session` returns `stop_reason=None` when the stream closed
    without a `done` frame — the backend was down, the connection dropped.
    #392's session holds one user message and nothing else; it was recorded as
    that item's one attempt one second after it started. Newer runs record
    `infra_failed` outright, but this reads the same fact off the old shape so
    history heals without a backfill.
    """
    # The key must be PRESENT and null, not merely absent. `execute` always
    # records `stop_reason`, so an explicit null is the producer saying the
    # stream closed without a `done` frame; a missing key is some other writer
    # whose shape we should not be interpreting. Absence falls through to
    # `spent`, which is the status quo and the safe direction.
    return ("stop_reason" in ev and ev["stop_reason"] is None
            and not ev.get("num_turns"))


def implement_outcomes(ledger: Path) -> dict[int, tuple[str, str]]:
    """`{item_id: (verdict, detail)}` for every item an implement turn touched.

    `verdict` is one of `spent`, `reopened`, `external`, `incomplete`,
    `infra`, `rolled_back`, `review_retry`, `partial` — where everything but
    `spent` means the item is offered again. Exposed rather than folded into
    `implemented_ids` because the reason is worth putting in front of the
    next round: a re-offer whose branch still exists, or whose work was
    reverted, is not a fresh start.

    History restarts twice. Rows at or before a re-triage mark are ignored,
    so a re-triaged item is not in this table until it is attempted again.
    And a human `reopen_item` resets the attempt count: it used to reset only
    the latest row, so an item reopened after four external blocks read
    `n = 5` on its first new round and every cap below was already spent.
    A turn a landing drain refused before it ran is not an attempt at all
    (see `implement_history`).
    """
    history = implement_history(ledger)
    latest: dict[int, dict] = {}
    attempts: dict[int, int] = {}
    # Budget deaths counted on their own. The incomplete rule used to read the
    # shared attempt count as `n <= 1 + INCOMPLETE_RETRY_CAP`, which re-offered
    # a second budget death as well — three rounds for "it comes back once".
    # #575 died at 151 iterations, was re-offered, and its second round was
    # heading for the same wall on the same test file. Counting incompletes
    # alone keeps "once" true without spending it on an unrelated earlier
    # verdict (an external block, say).
    incompletes: dict[int, int] = {}
    for iid, rows in history.items():
        for d in rows:
            latest[iid] = d
            phase = str(d.get("phase") or "")
            if phase == "reopened":
                attempts[iid] = 0
                incompletes[iid] = 0
            elif phase in ("finished", "infra_failed"):
                attempts[iid] = attempts.get(iid, 0) + 1
                if (phase == "finished"
                        and str(d.get("stop_reason") or "") in INCOMPLETE_STOP_REASONS):
                    incompletes[iid] = incompletes.get(iid, 0) + 1

    blocked = externally_blocked_rounds(ledger)
    unlanded = gate_passed_unlanded_rounds(ledger)
    # Capped on its own count, like an external landing failure: how many of
    # this item's rounds ended that way, not how many attempts it has used.
    unlanded_counts: dict[int, int] = {}
    for d in _ledger_events(ledger, "backlog_implement"):
        if d.get("phase") == "finished" and str(d.get("round_id") or "") in unlanded:
            unlanded_counts[int(d["item_id"])] = unlanded_counts.get(int(d["item_id"]), 0) + 1
    reverted = rolled_back_rounds(ledger)
    review_retry = review_retry_rounds(ledger)
    promoted_ev = {str(d.get("round_id") or ""): d for d in
                   _ledger_events(ledger, "promoted", require_item=False)}
    promoted = set(promoted_ev)
    out: dict[int, tuple[str, str]] = {}
    for iid, ev in latest.items():
        phase = str(ev.get("phase") or "")
        rid = str(ev.get("round_id") or "")
        n = attempts.get(iid, 0)
        if phase == "reopened":
            out[iid] = ("reopened", str(ev.get("reason") or "reopened by a human"))
            continue
        # A promotion is a verdict however the turn ended. #278 died at
        # `max_turns` and the observer's ambient follow-up gated and landed it
        # anyway; without this the incomplete rule below would re-offer a
        # change that is already in `main` and the next round would redo it.
        # Only a rollback reopens a landed item, and that is the next branch.
        if rid and rid in promoted and rid not in reverted:
            # ...unless the round's own finalizer said a clause was not met.
            # That is a landed change that did not finish the job, and the
            # honest move is one more round told which clauses, not a closed
            # item and not a park.
            unmet = unmet_clauses(ev.get("outcome"))
            if unmet and n <= 1 + PARTIAL_RETRY_CAP:
                sha = str(promoted_ev[rid].get("commit") or "")[:8]
                out[iid] = ("partial",
                            f"round {rid} landed as {sha or '?'} but its own outcome reported "
                            f"clause(s) {unmet} not met; the branch is gone with the landing, "
                            f"so start from live main")
                continue
            out[iid] = ("spent", "")
            continue
        if rid and rid in reverted and n <= ROLLED_BACK_RETRY_CAP:
            out[iid] = ("rolled_back",
                        f"round {rid} landed and the guardian reverted it; the branch "
                        f"was deleted at landing, so the tree is in the guardian tag")
            continue
        if phase == "infra_failed" or (phase == "finished" and _never_ran(ev)):
            if n <= 1 + INCOMPLETE_RETRY_CAP:
                errs = "; ".join(str(e)[:120] for e in (ev.get("errors") or [])[:2])
                out[iid] = ("infra", f"the turn never reported completion{': ' + errs if errs else ''}")
                continue
        if (phase == "finished" and str(ev.get("stop_reason") or "") in INCOMPLETE_STOP_REASONS
                and incompletes.get(iid, 0) <= INCOMPLETE_RETRY_CAP):
            out[iid] = ("incomplete",
                        f"the turn ran out of {'clock' if ev.get('stop_reason') == 'turn_timeout' else 'iterations'} "
                        f"before reaching a verdict"
                        + (f"; its work is on branch `automod/{rid}`" if rid else ""))
            continue
        # After `incomplete`, before `external`: a review-refused round whose
        # turn then died at its budget is still incomplete, and both verdicts
        # read the last gate event, so they are exclusive with `external`.
        if rid and rid in review_retry:
            clause = review_disagreement(ledger, iid)
            if clause is not None:
                out[iid] = ("spent",
                            f"review disagreement: clause {clause} came back unmet on two "
                            f"consecutive reviews of #{iid}; a human decides")
                continue
            if n <= 1 + REVIEW_RETRY_CAP:
                findings = str(review_retry[rid].get("review_findings") or
                               review_retry[rid].get("detail") or "")[:1200]
                out[iid] = ("review_retry",
                            f"the review rung found the premise sound but the implementation "
                            f"or its tests short — {findings}; its work is on branch "
                            f"`automod/{rid}` (pass it as from_branch)")
                continue
        if rid and rid in unlanded and unlanded_counts.get(iid, 0) <= EXTERNAL_RETRY_CAP:
            out[iid] = ("external",
                        f"round {rid} passed every gate rung and its landing never completed "
                        f"(no promotion, no land_failed); the change is gated — resume branch "
                        f"`automod/{rid}` (pass it as from_branch), gate, and call automod_land, "
                        f"never `round land` from Bash")
            continue
        if rid and rid in blocked and _external_budget_left(ledger, iid, rid, n):
            g = _last_gate_per_round(ledger).get(rid) or {}
            out[iid] = ("external",
                        f"round {rid} was blocked at the `{g.get('rung')}` rung by a condition "
                        f"it did not cause; its work is on branch `automod/{rid}`")
            continue
        out[iid] = ("spent", "")
    return out


def implemented_ids(ledger: Path) -> set[int]:
    """Items whose one unattended attempt is spent.

    An attempt is spent when a round reached a verdict on the change — it
    landed, or the gate judged the diff and refused it. Everything else is not
    a verdict and is offered again, bounded: see `implement_outcomes`.

    On 2026-09-08 three rounds aborted on pre-existing test failures (#361,
    #370, #376), one was refused because an unrelated uncommitted edit sat in
    the production tree (#447), one was killed by the wall clock fourteen
    seconds before it would have gated (#446), and one was recorded as an
    attempt a second after starting because the backend was down (#392). Six
    of seventeen attempts, none of them a judgment on the item, all six
    unreachable afterwards.
    """
    return {iid for iid, (verdict, _) in implement_outcomes(ledger).items()
            if verdict == "spent"}


def reoffer_reason(ledger: Path, item_id: int) -> str:
    """Why this item is being offered again, for the round that gets it.

    A re-offer is not a fresh start: the branch may still hold the work, or a
    landing may have been reverted. A round told nothing re-derives it, or
    worse, redoes it.
    """
    verdict, detail = implement_outcomes(ledger).get(int(item_id), ("", ""))
    return "" if verdict in ("", "spent") else f"{verdict}: {detail}"


def prior_rounds(ledger: Path, item_id: int) -> list[dict]:
    """Every finished implement turn for the item: what it filed, what it
    merged into, how many findings it appended. Old rows lack the newer keys."""
    out: list[dict] = []
    for d in _ledger_events(ledger, "backlog_implement"):
        if int(d["item_id"]) != int(item_id) or d.get("phase") != "finished":
            continue
        out.append({"round_id": str(d.get("round_id") or ""),
                    "spawned": _ints(d.get("spawned")),
                    "merged": _ints(d.get("merged")),
                    "findings_appended": int(d.get("findings_appended") or 0),
                    "ts": d.get("ts")})
    return out


def prior_spawned(ledger: Path, item_id: int) -> list[int]:
    """Ids earlier rounds of this item filed or merged into, in ledger order.

    A re-offered round told nothing re-derives the same peripheral findings
    and files them again: #549 ran four times in 110 minutes and filed ten
    children, three of them one finding. This is what the next round is
    shown so it appends instead.
    """
    seen: list[int] = []
    for r in prior_rounds(ledger, item_id):
        for i in r["spawned"] + r["merged"]:
            if i not in seen:
                seen.append(i)
    return seen


_FINDINGS_HEADING = re.compile(r"^##\s+Findings\b", re.M)
_BULLET = re.compile(r"^\s*[-*]\s+\S")


def _findings_bullets(body: str) -> int:
    """Bullets under every `## Findings…` heading, up to the next `## `."""
    count = 0
    lines = (body or "").splitlines()
    inside = False
    for line in lines:
        if line.startswith("## "):
            inside = bool(_FINDINGS_HEADING.match(line))
            continue
        if inside and _BULLET.match(line):
            count += 1
    return count


def count_findings(body_before: str, body_after: str) -> int:
    """How many findings a round appended to its item — counted off the file,
    not the report, so a round that says it appended three and appended none
    reads as none."""
    return max(0, _findings_bullets(body_after) - _findings_bullets(body_before))


def findings_sections(body: str) -> int:
    """How many `## Findings…` sections an item already carries — what earlier
    triage and implement turns appended to it."""
    return len(_FINDINGS_HEADING.findall(body or ""))


def prior_triage_spawned(ledger: Path, item_id: int) -> list[int]:
    """Ids earlier triage runs of this item filed or merged into, in ledger
    order. `prior_spawned`'s twin over `backlog_triage` rows — including
    `incomplete` ones, which may have filed before the budget ran out."""
    seen: list[int] = []
    for d in _ledger_events(ledger, "backlog_triage"):
        if int(d["item_id"]) != int(item_id):
            continue
        for i in _ints(d.get("spawned")) + _ints(d.get("merged")):
            if i not in seen and i != int(item_id):
                seen.append(i)
    return seen


def spawn_origin(ledger: Path, item_id: int) -> dict | None:
    """Which unattended pass filed this item: `{by, parent, ts, verdict}`.

    Read off the ledger rows that name it as filed — `backlog_triage.spawned`,
    `backlog_implement.spawned`, `arch_review.filed` — because the `parent`
    frontmatter key is set only by the cluster pass and the prose first line
    ("Split from #N") is the model's own spelling. The first row wins: a later
    row naming the same id is a re-citation, not a second filing.
    """
    iid = int(item_id)
    rows = ([("triage", d, "spawned") for d in _ledger_events(ledger, "backlog_triage")]
            + [("autocode", d, "spawned") for d in _ledger_events(ledger, "backlog_implement")]
            + [("arch-review", d, "filed")
               for d in _ledger_events(ledger, "arch_review", require_item=False)])
    rows.sort(key=lambda r: float(r[1].get("ts") or 0))
    for by, d, key in rows:
        if iid in _ints(d.get(key)):
            parent = d.get("item_id") if by != "arch-review" else d.get("unit")
            return {"by": by, "parent": parent, "ts": d.get("ts"),
                    "verdict": str(d.get("verdict") or d.get("phase") or "")}
    return None


# Every landed item stayed open. Nine promotions settled in the loop's first
# three days and not one item was closed: `promote` writes the commit,
# the guardian writes `settled`, `execute` writes `finished`, and nothing
# joined the three back to the item's status. `implemented_ids` kept them
# from being re-picked, so they sat on the board as `up_next` and `draft` —
# the loop's own finished work, counted as its backlog.
LANDED_MARKER = "automod_landed"
# Items landed before the rename carry the old marker. Read both; write the new.
_LEGACY_LANDED_MARKERS = ("autoimplement_landed", "selfmod_landed")


def settled_landings(ledger: Path) -> list[dict]:
    """Every item whose round landed and stayed landed.

    A code round counts once its promotion has `settled` — the guardian
    watched the window and did not revert. A vault round counts on its own
    `vault_land`: it is validated and committed in one step and has no window
    to survive. A reverted promotion is not a landing.

    **A `vault` item's landing counts even when its turn also opened a code
    round.** The implement prompt told every surface to `automod_start` and
    pin each clause with a test, so 8 of the first 15 vault turns cut a round
    beside their vault commits — and a finished row with a `round_id` was
    read only as a code landing. #575's fix landed at 21:25Z; its tests-only
    round never promoted, so the landed fix was invisible here and the item
    was re-offered. Such a landing counts only on the vault review's own
    verdict (`vault_review_outcome`). While a promotion of the round exists
    and has not settled, nothing counts yet; once it settles, the code landing
    is the one recorded. A landing still waiting for idle has no `promoted`
    row, so the vault landing is recorded first and the later promotion is
    not processed again — which is right for a `vault` contract, whose
    clauses the vault review graded against the vault, not the round.

    **The vault review's verdict stands in for a missing outcome** on a
    `vault` item, whichever landing is recorded. A turn that dies at
    `max_turns` or its wall clock gets no finalizer, so its outcome is None
    and the landing used to wait for a human. When the newest vault landing's
    review graded every clause `met`, that is a second reader's verdict on
    the whole contract and it is used. An outcome the turn did report is never
    overridden, and a landing beside a round closes on a reported `met` only
    when that review agrees.
    """
    settled = {str(d.get("commit") or ""): d
               for d in _ledger_events(ledger, "settled", require_item=False)}
    settled.pop("", None)
    reverted = {str(d.get("commit") or "") for d in
                _ledger_events(ledger, "rollback_succeeded", require_item=False)}
    promoted_any = {str(d.get("round_id") or "")
                    for d in _ledger_events(ledger, "promoted", require_item=False)}
    promoted = {str(d.get("round_id") or ""): d
                for d in _ledger_events(ledger, "promoted", require_item=False)
                if str(d.get("commit") or "") in settled and d.get("round_id")}
    # Built once, on the first row that needs them: a whole-ledger read per
    # row is ~0.08 s each on a 4.4 MB ledger, and this runs at turn end.
    memo: dict[str, object] = {}

    def surface_of(d: dict) -> str:
        if d.get("surface"):
            return str(d["surface"])
        if "confirmed" not in memo:
            memo["confirmed"] = confirmed_verdicts(ledger)
        return str((memo["confirmed"].get(int(d["item_id"])) or {}).get("surface") or "")

    def graded_for(d: dict, vault: list[str]) -> dict | None:
        if not vault or surface_of(d) != "vault":
            return None
        if "lands" not in memo:
            memo["lands"] = {str(e.get("commit") or ""): e for e in
                             _ledger_events(ledger, "vault_land", require_item=False)
                             if e.get("ok")}
            memo["vault_reverted"] = {str(e.get("reverted") or "") for e in
                                      _ledger_events(ledger, "vault_revert", require_item=False)}
        return vault_review_outcome(ledger, vault, lands=memo["lands"],
                                    reverted=memo["vault_reverted"])

    out: list[dict] = []
    for d in _ledger_events(ledger, "backlog_implement"):
        if d.get("phase") != "finished":
            continue
        rid = str(d.get("round_id") or "")
        vault = [str(c) for c in (d.get("vault_commits") or []) if c]
        outcome = d.get("outcome")
        reported = outcome.get("acceptance") if isinstance(outcome, dict) else None
        if rid in promoted and promoted[rid]["commit"] not in reverted:
            p = promoted[rid]
            out.append({"item_id": int(d["item_id"]), "round_id": rid, "commit": p["commit"],
                        "settled_at": settled[p["commit"]].get("created_at"),
                        "outcome": outcome if reported else (graded_for(d, vault) or outcome),
                        "vault": False})
            continue
        if not vault:
            continue
        if not rid:
            out.append({"item_id": int(d["item_id"]), "round_id": "", "commit": vault[-1],
                        "settled_at": d.get("created_at"),
                        "outcome": outcome if reported else (graded_for(d, vault) or outcome),
                        "vault": True})
            continue
        if rid in promoted_any or reported not in (None, "", "met"):
            continue
        graded = graded_for(d, vault)
        if graded is None:
            continue
        out.append({"item_id": int(d["item_id"]), "round_id": "", "commit": vault[-1],
                    "settled_at": d.get("created_at"),
                    "outcome": outcome if reported else graded, "vault": True})
    return out


def vault_review_outcome(ledger: Path, vault_commits: list[str], *,
                         lands: dict | None = None, reverted: set | None = None) -> dict | None:
    """A `met` outcome built from the vault review of a turn's landings, or None.

    Reads the NEWEST of `vault_commits` only. Its `vault_land` event must carry
    the grader's `review_clauses` — one row per clause of the contract as it
    stood at grading (`review.grade_vault`) — and every row must be `met`:
    `partial`, `not_met`, `post_landing`, `ungraded` or a gap all answer None,
    which leaves the landing exactly where it stood. So does a newest landing
    the grader did not pass (skipped during a drain, say): an earlier review's
    verdict describes a tree a later ungraded commit may have changed. None too
    when any of the commits was reverted. Landings recorded before
    `review_clauses` existed carry none and answer None.

    `lands` / `reverted` are the `vault_land` map and reverted-sha set, passed
    by a caller that already built them.
    """
    commits = [str(c) for c in vault_commits if c]
    if not commits:
        return None
    if reverted is None:
        reverted = {str(d.get("reverted") or "")
                    for d in _ledger_events(ledger, "vault_revert", require_item=False)}
    if reverted & set(commits):
        return None
    if lands is None:
        lands = {str(d.get("commit") or ""): d
                 for d in _ledger_events(ledger, "vault_land", require_item=False) if d.get("ok")}
    sha = commits[-1]
    ev = lands.get(sha) or {}
    graded = ev.get("review_clauses")
    if ev.get("review") != "pass" or not graded:
        return None
    verdicts: dict[int, str] = {}
    for c in graded:
        try:
            verdicts[int(c["clause"])] = str(c["verdict"])
        except (KeyError, TypeError, ValueError):
            return None
    n = len(verdicts)
    if set(verdicts) != set(range(1, n + 1)) or any(v != "met" for v in verdicts.values()):
        return None
    return {"acceptance": "met", "landed": True, "source": "vault_review",
            "deferred_to": [], "spawned": [],
            "clause_outcomes": [{"clause": i, "outcome": "met", "deferred_to": [],
                                 "evidence": f"vault review of {sha[:8]}"}
                                for i in sorted(verdicts)],
            "summary": f"the vault review of {sha[:8]} graded all {n} clause(s) met"}


def close_landed(item: Item, *, commit: str, round_id: str, settled_at: str,
                 close: bool, why: str, tags: tuple[str, ...] = ()) -> Path:
    """Record the landing on the item; close it when `close`.

    The marker is written either way, so a landing is processed once. A
    human can still close an item the loop left open; the loop will not
    reopen one a human closed.
    """
    fm, body = _split_frontmatter(item.path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    where = f"round {round_id}" if round_id else "a vault round"
    entry = (f"**{stamp}** — automod landed as `{commit[:8]}` ({where}, "
             f"{'settled' if round_id else 'committed'} {settled_at}). "
             + (f"Closed: {why}" if close else f"Left open: {why}"))
    log = list(fm.get("activity_log") or [])
    log.append(entry)
    fm["activity_log"] = log
    fm["updated"] = stamp
    fm[LANDED_MARKER] = commit
    if tags:
        have = [str(t) for t in (fm.get("tags") or [])]
        fm["tags"] = have + [t for t in tags if t not in have]
    if close:
        fm["status"] = "done"
        fm["completed"] = stamp
    section = (f"\n\n## Automod landed — {stamp[:10]}\n\n`{commit[:8]}`, {where}, "
               f"{'settled' if round_id else 'committed'} {settled_at}.\n\n"
               + ("**Closed.** " if close else "**Left open.** ") + why + "\n")
    item.path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body.rstrip()}{section}", encoding="utf-8")
    return item.path


_CLOSE_SWEEP_LOCK = threading.Lock()


def close_settled_items(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                        enabled: bool = True, close_members: bool = True) -> list[dict]:
    """The sweep: note every settled landing on its item, close the ones whose
    round said the acceptance check was met.

    Only `met` closes. The round is the only party that judged the acceptance
    and it is asked in a structured finalizer, not read out of prose; anything
    else — `deferred` with the ids it waits on, `not_met`, or no outcome at all
    because the round predates the finalizer — is noted and left for a human.
    A closed item is never re-triaged, which is why the default is to leave
    it open rather than guess.
    """
    if not enabled:
        return []
    # One sweep at a time. Housekeeping runs it in a worker thread and
    # `autocode.execute` runs it at the end of a vault-landing turn; the
    # landed-marker check and the write are not atomic, so two overlapping
    # sweeps could both record the same landing.
    with _CLOSE_SWEEP_LOCK:
        return _close_settled_items(ledger, boards, close_members=close_members)


def _close_settled_items(ledger: Path, boards: tuple[str, ...] | None, *,
                         close_members: bool) -> list[dict]:
    from scripts.automod import state as S
    by_id = {i.id: i for i in open_items(boards)}
    done: list[dict] = []
    for landing in settled_landings(ledger):
        item = by_id.get(landing["item_id"])
        if item is None:
            continue
        fm, _ = _split_frontmatter(item.path.read_text(encoding="utf-8"))
        if fm.get(LANDED_MARKER) or any(fm.get(m) for m in _LEGACY_LANDED_MARKERS):
            continue
        outcome = landing.get("outcome") or {}
        # The review rung's own `post_landing` verdicts, honoured here rather
        # than by the implementer — which is inside the round and cannot know
        # that a clause needs live traffic. A clause so marked reads as met
        # for the purpose of "did the round do its job" and holds the item
        # open for the purpose of "has anyone confirmed it".
        outcome = apply_post_landing(outcome, post_landing_clause_ids(item.path)) or outcome
        acc = outcome.get("acceptance")
        human = list(human_clauses_of(None, fm))
        # A path the loop may never write is the same shape of debt as a
        # human clause: the round did what it could, and a person owes the
        # rest. Reported rather than hidden, which is what `git add -f` was.
        for hp in (outcome.get("human_paths") or []):
            human.append(f"apply `{hp.get('path')}` by hand: {hp.get('reason')}")
        for idx in (outcome.get("post_landing_clauses") or []):
            human.append(f"confirm clause {idx} now that the change is live")
        tags: tuple[str, ...] = ()
        if acc == "met" and human:
            # #1210: the loop's half is done, so the item closes. It used to be
            # left open in `draft` carrying `needs-human`, and `draft` is the
            # pool single-item triage reads — six landed items sat in it on
            # 2026-09-17, #1199 among them. `done` carrying the tag states both
            # halves in one place: the code is live, and a person still owes a
            # check. It stamps `completed`, because `close_landed` stamps it for
            # every close and a close is a close whatever its reason.
            close, tags = True, (NEEDS_HUMAN_TAG,)
            why = ("the round reported every acceptance clause met, so the item closes; a person "
                   "still owes: " + "; ".join(human))
        elif acc == "met":
            close = True
            why = ("the vault review graded every acceptance clause met; the turn ended "
                   "without reporting an outcome" if outcome.get("source") == "vault_review"
                   else "the round reported the acceptance check met") + (
                f" — {outcome['summary']}" if outcome.get("summary") else "")
        elif acc == "deferred":
            ids = ", ".join(f"#{i}" for i in outcome.get("deferred_to") or []) or "an unnamed follow-up"
            close, why = False, (f"the round deferred the acceptance check to {ids}; "
                                 f"close this when that closes")
        elif acc == "not_met":
            unmet = unmet_clauses(outcome)
            close, why = False, ("the round landed but reported the acceptance check not met"
                                 + (f" (clause(s) {unmet}); offered again for those" if unmet
                                    else ""))
        else:
            close, why = False, ("the round recorded no structured outcome (it predates the "
                                 "finalizer); a human decides")
        close_landed(item, commit=landing["commit"], round_id=landing["round_id"],
                     settled_at=str(landing.get("settled_at") or ""), close=close, why=why,
                     tags=tags)
        S.append_event({"event": "item_landed", "item_id": item.id,
                        "round_id": landing["round_id"], "commit": landing["commit"],
                        "vault": landing["vault"], "closed": close,
                        "acceptance": acc, "reason": why[:300],
                        "acceptance_source": outcome.get("source") or "round",
                        "human_clauses": human}, path=ledger)
        done.append({"item_id": item.id, "closed": close, "acceptance": acc})
        # An umbrella that closed `met` closes the members it consolidated.
        # `not_met`, `deferred` and no-outcome leave them folded: the
        # umbrella's own note says why, and a human can `unfold_umbrella`.
        if close and close_members and item.members:
            by_all = {i.id: i for i in open_items(None)}
            for mid in item.members:
                member = by_all.get(int(mid))
                if member is None:
                    continue
                mfm, _ = _split_frontmatter(member.path.read_text(encoding="utf-8"))
                if mfm.get(LANDED_MARKER):
                    continue
                close_landed(member, commit=landing["commit"], round_id=landing["round_id"],
                             settled_at=str(landing.get("settled_at") or ""), close=True,
                             why=f"landed via umbrella #{item.id} as {landing['commit'][:8]}")
                S.append_event({"event": "item_closed", "item_id": member.id, "by": "umbrella",
                                "umbrella_id": item.id, "round_id": landing["round_id"],
                                "commit": landing["commit"]}, path=ledger)
                done.append({"item_id": member.id, "closed": True, "acceptance": "met",
                             "via": item.id})
    return done


def post_landing_clause_ids(path: Path | None) -> list[int]:
    """Clause indices the review rung marked `post_landing` on this item."""
    if path is None or not Path(path).exists():
        return []
    fm, _ = _split_frontmatter(Path(path).read_text(encoding="utf-8"))
    out: list[int] = []
    for raw in (fm.get("post_landing_clauses") or []):
        try:
            n = int(raw)
        except (TypeError, ValueError):
            continue
        if n > 0 and n not in out:
            out.append(n)
    return sorted(out)


def mark_clause_post_landing(item_id: int, clause: int, note: str = "",
                             round_id: str = "") -> bool:
    """Record that clause N of item `item_id` can only be observed live.

    Written by `gate.rung_review` on a PASS, so the mark is a fact the
    grader established about a change that is about to land — not a claim
    the implementer made about its own work. Idempotent: a clause marked
    twice by two rounds is marked once.
    """
    paths = sorted(BACKLOG_DIR.glob(f"{int(item_id)}-*.md"))
    if not paths:
        return False
    path = paths[0]
    fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
    existing = [int(x) for x in (fm.get("post_landing_clauses") or [])
                if str(x).lstrip("-").isdigit()]
    if int(clause) in existing:
        return False
    return update_frontmatter(
        path, {"post_landing_clauses": sorted(existing + [int(clause)])},
        activity=(f"review marked acceptance clause {int(clause)} as observable "
                  f"only after landing"
                  + (f" ({round_id})" if round_id else "")
                  + (f": {note}" if note else "")))


def note_review_advisories(item_id: int, round_id: str, seams: list[str],
                           findings: list[str]) -> bool:
    """Record what a PASSING review still said, on the item.

    The review prompt has always promised the grader that an untestable seam
    is "recorded for the item as a post-landing check", and nothing did it:
    on a pass `rung_review` kept the clauses and dropped the advisory text,
    so the only record was a findings string in `gate.json`, deleted with the
    round. Seam texts accumulate under `post_landing_seams` (deduplicated);
    advisory findings go on one activity line with the seams.

    Recorded, not enforced: the item is NOT held open `needs-human` for a
    seam. 23 of the grader era's first 30 seams were untestable by the
    grader's own word, and holding on them would park nearly every landing.
    """
    seams = [" ".join(str(s).split())[:300] for s in (seams or []) if str(s).strip()]
    findings = [" ".join(str(f).split())[:300] for f in (findings or []) if str(f).strip()]
    if not seams and not findings:
        return False
    path = _item_path(item_id)
    if path is None:
        return False
    fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
    existing = [str(s) for s in (fm.get("post_landing_seams") or [])]
    merged = existing + [s for s in seams if s not in existing]
    parts = []
    if seams:
        parts.append("post-landing seams: " + " | ".join(seams))
    if findings:
        parts.append("advisory findings: " + " | ".join(findings))
    return update_frontmatter(
        path, {"post_landing_seams": merged} if seams else {},
        activity=(f"automod review passed round {round_id} with advisories — "
                  + "; ".join(parts))[:2000])


def record_human_paths(item_id: int, human_paths: list[dict],
                       round_id: str = "") -> list[str]:
    """Record paths a round needed and the loop may never write.

    `close_settled_items` treats these exactly like a human clause, which since
    #1210 means: a `met` landing closes and the item carries `needs-human` with
    the path named in its activity log. The debt is what survives the closure,
    not an open row in the triage pool. Returns the paths recorded.
    """
    if not human_paths:
        return []
    paths = sorted(BACKLOG_DIR.glob(f"{int(item_id)}-*.md"))
    if not paths:
        return []
    path = paths[0]
    fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
    existing = list(fm.get("human_paths") or [])
    seen = {str(e.get("path") if isinstance(e, dict) else e) for e in existing}
    added: list[str] = []
    for hp in human_paths:
        if not isinstance(hp, dict):
            continue
        rel = str(hp.get("path") or "").strip()
        if not rel or rel in seen:
            continue
        existing.append({"path": rel, "reason": str(hp.get("reason") or "")[:300]})
        seen.add(rel)
        added.append(rel)
    if not added:
        return []
    update_frontmatter(
        path, {"human_paths": existing}, add_tags=(NEEDS_HUMAN_TAG,),
        activity=(f"round {round_id or '?'} needed paths the loop may not write, "
                  f"left for a person: " + ", ".join(f"`{x}`" for x in added)))
    return added


def work_title_for_round(ledger: Path, round_id: str) -> str:
    """What a round was FOR, in words a person recognises.

    "Promoted SM_20260909_160856" was the toast, and the spoken version of it
    is a date read out one digit at a time. The round id is bookkeeping and
    lives in the ledger row; the thing worth hearing is the item's name. Three
    sources, cheapest first: the implement turn's own `started` event carries
    the name; failing that the triage event does; failing that the round's
    goal, which for an unattended round names the item in its first sentence.
    Empty when nothing is known — the caller says something generic rather
    than falling back to the id.
    """
    rid = str(round_id or "")
    item_id = None
    for d in _ledger_events(ledger, "backlog_implement"):
        if d.get("phase") == "finished" and str(d.get("round_id") or "") == rid:
            item_id = int(d["item_id"])
    if item_id is not None:
        for kind in ("backlog_implement", "backlog_triage"):
            for d in reversed(_ledger_events(ledger, kind)):
                if int(d["item_id"]) == item_id and d.get("name"):
                    return str(d["name"]).strip()[:120]
        for item in open_items(None):
            if item.id == item_id and item.name:
                return item.name.strip()[:120]
    for d in reversed(_ledger_events(ledger, "round_start", require_item=False)):
        if str(d.get("round_id") or "") == rid and d.get("goal"):
            goal = " ".join(str(d["goal"]).split())
            head = goal.split(". ")[0]
            return (head if len(head) <= 120 else head[:117].rsplit(" ", 1)[0] + "…")
    return ""


# ── Status is the pipeline's state machine ──────────────────────────────────
#
# Until 2026-09-09 the loop wrote `status` in exactly two places, both `done`.
# A `confirmed` verdict left the item wherever it was; a round never set
# `in_progress`; #353 landed while still `draft`. The ledger was the state
# machine and the board showed none of it. Now:
#
#   draft ──autotriage confirms──▶ up_next ──round opens──▶ in_progress
#                 │                                               │
#                 └──already_done / stale──▶ done ◀── landed & met, or unnecessary
#
# and back to `up_next` when an attempt ends without a verdict (external,
# incomplete, infra, rolled back, reopened). `done` is terminal for this
# writer. A human may set any status by hand; the loop only rewrites a status
# it has a ledger opinion about.
# `PIPELINE_STATUSES` / `OPEN_STATUSES` come from `app.backlog_status`, which
# five readers share; see that module for why a status outside it is the
# failure mode rather than disagreement about what is inside it.
TRIAGE_POOL_STATUS = "draft"
IMPLEMENT_POOL_STATUS = "up_next"


# A spent attempt goes to `draft`, where a human looks for things that need a
# judgment — but `draft` is also 250 items deep, and an item that needs a
# decision looks exactly like one nobody has read yet. The tag is the
# difference. It rides the status move both ways: on when the item goes to
# draft as spent, off when a reopen takes it back into the pool.
NEEDS_HUMAN_TAG = "needs-human"

# A confirmation that arrived while the implement pool was full: triaged,
# judged real, parked in `draft` until a slot opens. Visible on the item
# the way `needs-human` is; the ledger (`held: true` on the verdict, then a
# `backlog_confirm_released`) is the source of truth. See
# `held_confirmations`.
HELD_TAG = "confirmed-held"

# ── The sweep's rank ────────────────────────────────────────────────────────
#
# A sweep (`select_sweep_batch` / `record_sweep_verdicts`, driven by
# autotriage's sweep mode) reads a batch of items in one turn and either
# retires each one or ranks it: `worth` (how much the system gains) and
# `size` (what a round would cost). The rank is written to front matter and
# tagged `swept`; a `low` draft is also tagged `parked`. Parked is the
# replacement for expiry as the exit for an item nobody asked for: still
# open, visible, out of every pool, never expired, and a person promotes it
# by removing the tag. `rank_key` orders the triage pool, the implement pool
# and the held-confirmation release, so the loop works the best item it
# knows about rather than the oldest.
SWEPT_TAG = "swept"
PARKED_TAG = "parked"
WORTH_LEVELS = ("high", "medium", "low")
SIZE_LEVELS = ("small", "medium", "large")
# Unranked sorts between medium and low: the sweep has not read it yet, and
# an item the sweep called medium was judged worth more than a guess.
_WORTH_ORDER = {"high": 0, "medium": 1, "": 2, "low": 3}
_SIZE_ORDER = {"small": 0, "medium": 1, "": 1, "large": 2}

# ── The human's priority ────────────────────────────────────────────────────
#
# `priority` in front matter is the one field on an item a person sets to
# say how much they want it, and until 2026-09-16 no pool read it: the MCP
# writer stamped `medium` on everything it filed, the HTTP route stamped
# `none`, and the loop ordered on the sweep's `worth`. Alan's rule: the pace
# is tolerable if the tag is honoured, so `priority_key` sorts ahead of
# `rank_key` in every pool — triage, sweep, implement, the held release —
# and a single candidate that outranks every member of the cluster group
# triage would take goes first (`priority_beats`). The default is `low`, on
# both writers and on read: a value nobody chose must not outrank one
# somebody did. `none`, an absent key or an unknown word all read as `low`;
# `backfill_priority` writes the word onto the files that had none.
PRIORITY_LEVELS = ("high", "medium", "low")
DEFAULT_PRIORITY = "low"
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def priority_key(item: Item) -> int:
    """Sort key: high before medium before low. Anything else is low."""
    return _PRIORITY_ORDER.get(str(item.priority or "").strip().lower(),
                               _PRIORITY_ORDER[DEFAULT_PRIORITY])


def is_high(item: Item) -> bool:
    """A `high` item is the one a person asked for next: it goes ahead of a
    sweep batch, a cluster and a live blocker in triage, its confirmation is
    never held by the depth gate, and it is first in the implement order."""
    return priority_key(item) == _PRIORITY_ORDER["high"]


def _created_ts(item: Item) -> float | None:
    try:
        c = datetime.fromisoformat(str(item.created).replace("Z", "+00:00"))
        if c.tzinfo is None:
            c = c.replace(tzinfo=timezone.utc)
        return c.timestamp()
    except (ValueError, AttributeError, TypeError):
        return None


def recency_key(item: Item) -> tuple[float, int]:
    """Age as a sort key: oldest first — except within `high`, where the
    NEWEST goes first. A high item is a person asking for it next, and the
    board carried 87 open highs on 2026-09-16 (17 ready, 9 held), most of
    them months old; oldest-first would have put the one just raised
    behind all of them. An unknown `created` sorts last either way."""
    ts = _created_ts(item)
    if is_high(item):
        return (float("inf") if ts is None else -ts, item.id)
    return (float("inf") if ts is None else ts, item.id)


def priority_beats(item: Item, others) -> bool:
    """True when `item` is strictly higher priority than every one of
    `others` — the single-pool candidate against a cluster's members."""
    others = list(others)
    return bool(others) and priority_key(item) < min(priority_key(o) for o in others)


def backfill_priority(boards: tuple[str, ...] | None = None, *,
                      default: str = DEFAULT_PRIORITY, dry_run: bool = False,
                      reset_open: bool = False,
                      backlog_dir: Path | None = None) -> list[dict]:
    """Write `default` onto every item — any board, any status — whose
    `priority` is absent, `none`, or not one of `PRIORITY_LEVELS`.

    `reset_open` is the stronger reading of "default low for existing
    items": every OPEN item is written `default` whatever it carries, so the
    high tier starts empty and a person raises what they want next. Never
    run by anything but a human's `round priority-backfill --reset-open`;
    the value it overwrites is kept in the activity line.

    One activity line per file so the change is on the item's own record. A
    file whose YAML does not parse is reported `skipped`, never rewritten
    (`update_frontmatter` refuses it). Returns one row per item touched or
    skipped; `dry_run` reports without writing.
    """
    out: list[dict] = []
    for item in all_items(boards, backlog_dir=backlog_dir):
        fm, _ = _split_frontmatter(item.path.read_text(encoding="utf-8"))
        raw = fm.get("priority")
        valid = _level(raw, PRIORITY_LEVELS)
        if valid and not (reset_open and item.status in OPEN_STATUSES and valid != default):
            continue
        row = {"item_id": item.id, "status": item.status, "was": raw, "now": default}
        if dry_run:
            out.append({**row, "written": False})
            continue
        why = (f"it was {raw!r} and no pool read that" if not valid
               else f"reset from {raw!r} by hand so the high tier starts empty")
        ok = update_frontmatter(
            item.path, {"priority": default},
            activity=f"automod: priority set to {default} — the board default since 2026-09-16; {why}")
        out.append({**row, "written": ok, **({} if ok else {"skipped": "frontmatter did not parse"})})
    return out


def rank_key(item: Item) -> tuple[int, int, int]:
    """Sort key: worth first, then size, then defects before proposals. Lower
    is better.

    The third element (2026-09-16): within one worth/size a research proposal
    filed by the YouTube digest sorts after a defect. Measured over the week
    to 2026-09-16, proposal rounds landed 2 of 29 against 30 of 96 for
    defects, and 38 of the 74 `up_next` items were proposals — so an equal
    rank was handing half the implement pool to the shape that resolves
    slowest. A proposal still runs; it runs after the bug beside it.
    """
    return (_WORTH_ORDER.get(item.worth, 2), _SIZE_ORDER.get(item.size, 1),
            1 if any(t in EVAL_SPAWN_TAGS for t in item.tags) else 0)


def is_swept(item: Item) -> bool:
    return SWEPT_TAG in item.tags


def is_parked(item: Item) -> bool:
    return PARKED_TAG in item.tags


def set_status(item_id: int, status: str, why: str, *,
               add_tags: tuple[str, ...] = (), remove_tags: tuple[str, ...] = ()) -> bool:
    """Move an open item, once, with the reason in its log. False if it is
    not open, already there, or `done` (terminal for this writer)."""
    if status not in PIPELINE_STATUSES:
        raise ValueError(f"unknown status {status!r}")
    for item in open_items(None):
        if item.id == int(item_id):
            return _apply_status(item.path, status, why,
                                 add_tags=add_tags, remove_tags=remove_tags)
    return False


def _apply_status(path: Path, status: str, why: str, *,
                  add_tags: tuple[str, ...] = (), remove_tags: tuple[str, ...] = ()) -> bool:
    """Write one status move onto an item file, reason in its activity log.

    The single writer. `set_status` reaches it by id through `open_items`;
    the off-vocabulary rescue reaches it by path, because the item it is
    fixing is by definition not in `open_items`. A second definition of
    "record a status move" is how the two would come to disagree about the
    log line, the `updated` stamp, or which moves are refused.
    """
    fm, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    if fm.get("status") == status or fm.get("status") == "done":
        return False
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    log = list(fm.get("activity_log") or [])
    log.append(f"**{stamp}** — {fm.get('status')} → {status}: {why}")
    fm["activity_log"] = log
    fm["status"] = status
    # `normalize_tags`, never a bare iteration: a `tags` string that looks like
    # a list (the eval digest writes them) would come back one tag per character.
    tags = normalize_tags(fm.get("tags"))
    tags = [t for t in tags if t not in remove_tags] + [t for t in add_tags if t not in tags]
    fm["tags"] = tags
    fm["updated"] = stamp
    if status == "done":
        fm["completed"] = stamp
    path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body}", encoding="utf-8")
    return True


def desired_statuses(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS,
                     *, open_round_items: set[int] = frozenset()) -> dict[int, tuple]:
    """`{item_id: (status, why[, needs_human])}` — what the ledger says each
    open item's status should be. Only items the loop has an opinion about
    appear. The optional third element marks a spent attempt: the tag goes on
    with the move to draft and comes off with any move back into the pool.

    Pure, so the migration and the per-poll reconcile are the same function
    run against the same table, and so a test can read the table without a
    board. `open_round_items` is the set with a round in flight right now —
    the reconciler passes it, the migration passes what it can see.
    """
    confirmed = confirmed_verdicts(ledger)
    held = held_confirmations(ledger)
    verdicts = triaged_ids(ledger)
    outcomes = implement_outcomes(ledger)
    marks = retriage_marks(ledger)
    # Re-triaged and not yet confirmed again: whatever the implement history
    # after the mark says (a landing that raced the mark, a human reopen before
    # the second triage), `up_next` would strand it — no contract, so neither
    # a round nor triage would take it.
    def _awaits_triage(iid: int) -> bool:
        return iid in marks and iid not in confirmed
    # A turn in flight is a `started` with nothing after it. `implement_outcomes`
    # reads that shape as `spent` (it is not a verdict either way), so the
    # reconciler decides in-flight for itself rather than trusting a caller
    # who may be running an hour after the turn began.
    latest_phase: dict[int, str] = {}
    for d in _ledger_events(ledger, "backlog_implement"):
        latest_phase[int(d["item_id"])] = str(d.get("phase") or "")
    in_flight = {i for i, ph in latest_phase.items() if ph == "started"} | set(open_round_items)
    reverted = {str(d.get("commit") or "") for d in _ledger_events(ledger, "rollback_succeeded", require_item=False)}
    live_promoted = {str(d.get("round_id") or "") for d in _ledger_events(ledger, "promoted", require_item=False)
                     if str(d.get("commit") or "") not in reverted}
    observing: set[int] = set()
    for d in _ledger_events(ledger, "backlog_implement"):
        if d.get("phase") == "finished" and str(d.get("round_id") or "") in live_promoted:
            observing.add(int(d["item_id"]))
    # A landing in flight is a round in flight (2026-09-17). `automod_land`
    # returns at once and the turn ends; the detached promoter writes
    # `promoted` minutes later. The reconcile that runs at turn end saw a
    # finished row with no promotion and read it as spent: #1197 twice and
    # #1199 were parked `draft` + needs-human for one poll while their
    # landing was mid-drain, and came back at the next. The landing marker
    # (`land.running`, alive pid) is the record that closes the gap.
    from scripts.automod import state as S
    for d in _ledger_events(ledger, "backlog_implement"):
        if d.get("phase") != "finished":
            continue
        rid = str(d.get("round_id") or "")
        if rid and rid not in live_promoted and S.land_in_progress(rid):
            in_flight.add(int(d["item_id"]))
    # The landing's own outcome, last per item. `in_progress` means a round
    # is running on the item and nothing else (Alan's ruling, 2026-09-13):
    # a landing that settled and left the item open is not running — it is
    # waiting on a person, on other items, or on another attempt — and each
    # of those has a status that says so. Seventeen items sat `in_progress`
    # with nothing running and the dashboard counted them as active work.
    landed_outcome: dict[int, dict] = {}
    for d in _ledger_events(ledger, "item_landed"):
        landed_outcome[int(d["item_id"])] = d
    out: dict[int, tuple[str, str]] = {}
    for item in open_items(boards):
        iid = item.id
        fm, _ = _split_frontmatter(item.path.read_text(encoding="utf-8"))
        landed = fm.get(LANDED_MARKER) or any(fm.get(m) for m in _LEGACY_LANDED_MARKERS)
        # A landing whose own outcome said a clause was not met is offered
        # again once the sweep has marked it (the marker means settled) — it
        # must not sit in the `landed` park below. Until the sweep runs it is
        # still under observation and stays `in_progress`.
        partial = outcomes.get(iid, ("", ""))[0] == "partial"
        if is_grouped(item):
            # Folded into an umbrella: it closes when that lands `met`, and
            # nothing else may lift it into a pool.
            out[iid] = ("draft", f"folded into umbrella #{item.group}; it closes when that lands")
        elif iid in in_flight:
            out[iid] = ("in_progress", "an automod round is in flight for it")
        elif landed and not partial and landed_outcome.get(iid, {}).get("acceptance") == "not_met":
            # Documented rule: "a landed round with a not_met clause is offered
            # once more for exactly those clauses". Fall through to the
            # `outcomes` branch below, which knows whether that attempt is
            # still owed (`up_next`) or spent (`draft`, needs-human). This
            # branch used to park it `in_progress` and contradict the rule.
            verdict, detail = outcomes.get(iid, ("", ""))
            if verdict == "spent":
                out[iid] = ("draft", "landed with a clause not met and its one unattended attempt "
                                     "is spent; a human decides (reopen_item to grant another)", True)
            elif _awaits_triage(iid):
                out[iid] = ("draft", "re-triaged; waiting for its second triage to confirm a contract")
            else:
                out[iid] = ("up_next", "landed with a clause not met; offered once more for "
                                       "exactly those clauses")
        elif landed and not partial:
            ev = landed_outcome.get(iid, {})
            acc = ev.get("acceptance")
            if acc == "met":
                # #1210: a `met` landing is CLOSED. This branch proposed
                # `draft` + needs-human, and that is the `status_moved` row
                # which wrote #1199 back to `draft` 43 seconds after its own
                # `item_landed`. An item the sweep closed is off the board and
                # never reaches `open_items`, so arriving here means a met
                # landing sits in `draft` — the state the old rule wrote — and
                # the move is to close it, naming the landing commit so a reader
                # can tell that a landing decided this and not a triage.
                # The commit goes LAST, and the sweep reason is budgeted around
                # it: `reconcile_statuses` stores this string through
                # `[:200]`, so a fixed 160-char reason plus a 40-hex sha
                # (275 chars) loses exactly the part that names who decided.
                head = "landed with every clause met; closed, a person still owes it: "
                sha = str(fm.get(LANDED_MARKER) or "")
                tail = f" (landed as {sha})" if sha else ""
                room = max(0, 200 - len(head) - len(tail))
                out[iid] = ("done", head + (ev.get("reason") or "")[:room] + tail, True)
            elif acc == "deferred":
                # The landing's reason already names the ids it waits on.
                out[iid] = ("draft", "landed; not running, waiting on other items — "
                                     + (ev.get("reason") or "deferred")[:160])
            else:
                out[iid] = ("draft", "landed with no structured outcome recorded (predates the "
                                     "finalizer); a human decides", True)
        elif iid in observing and not (landed and partial):
            # Promoted, not yet settled: the guardian is watching it and the
            # sweep has not run. Neither back in the pool nor done.
            out[iid] = ("in_progress", "landed; the promotion is under observation")
        elif iid in outcomes:
            verdict, detail = outcomes[iid]
            if verdict == "spent":
                # Not `up_next`: that pool means "implement will take this", and
                # it will not — the one unattended attempt is used, and a second
                # is a human's call (`reopen_item`). `draft` is where a human
                # looks for things that need a judgment.
                out[iid] = ("draft", "its one unattended attempt is spent; a human decides "
                                     "(reopen_item to grant another)", True)
            elif _awaits_triage(iid):
                out[iid] = ("draft", "re-triaged; waiting for its second triage to confirm a contract")
            else:
                out[iid] = ("up_next", f"offered again — {verdict}: {detail[:120]}")
        elif iid in confirmed and not is_human_only(confirmed[iid].get("acceptance")):
            if iid in held:
                out[iid] = ("draft", "triage confirmed it; held until the implement pool has room")
            else:
                out[iid] = ("up_next", "triage confirmed it with an acceptance check")
        elif iid in verdicts:
            # unverifiable / not_code / incomplete / human-only: triaged, not for
            # the loop. Stays where triage found it.
            out[iid] = ("draft", f"triaged {verdicts[iid]}; not for the unattended loop")
        elif item.status == IMPLEMENT_POOL_STATUS:
            # Untriaged but sitting in the implement pool: nothing can pull it
            # from there. Back to where triage looks.
            out[iid] = ("draft", "never triaged; autotriage reads draft")
    return out


def rescue_off_vocabulary(ledger: Path,
                          boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> list[dict]:
    """Bring items whose status is outside `PIPELINE_STATUSES` back onto the
    board, mapping each onto the word it meant.

    This cannot be part of `reconcile_statuses`' own pass, because that pass
    reads `open_items` and `open_items` filters *on status* — so the one
    defect it can never see is a status that is wrong in this particular way.
    Such an item is stranded in the gap between the two halves of the system:
    `dashboard._BACKLOG_CLOSED` counts `review` as open work, while
    `OPEN_STATUSES` cannot see it at all, so it is shown to a human forever
    and is invisible to every machine that would move it. #287 (`review`) and
    #304 (`closed`) sat there from April 2026 until 2026-09-09.

    Runs before the reconcile rather than after it, so the rescued item is in
    `open_items` by the time the ledger's opinions are applied and gets a
    real verdict in the same pass instead of waiting for the next one.

    Board-filtered like everything else here: the backlog is shared, and an
    Alfie or Architecture item with an unusual status is not this loop's to
    rewrite.
    """
    from scripts.automod import state as S
    moved: list[dict] = []
    for item in all_items(boards):
        if not is_off_vocabulary(item.status):
            continue
        target = canonical_status(item.status)
        why = (f"{item.status!r} is not one of {', '.join(PIPELINE_STATUSES)} — "
               f"the board counted it "
               f"{'closed' if item.status.strip().lower() in CLOSED_ALIASES else 'open'} "
               f"and the loop could not see it at all")
        if _apply_status(item.path, target, why):
            S.append_event({"event": "status_moved", "item_id": item.id,
                            "from": item.status, "to": target,
                            "reason": why[:200], "off_vocabulary": True}, path=ledger)
            moved.append({"item_id": item.id, "from": item.status, "to": target})
    return moved


def reconcile_statuses(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                       open_round_items: set[int] = frozenset(), enabled: bool = True) -> list[dict]:
    """Write the desired statuses that differ. Idempotent; returns what moved."""
    if not enabled:
        return []
    from scripts.automod import state as S
    # First, anything the pass below is structurally unable to see. `current`
    # is read after it so the rescued items are judged in this same pass.
    moved: list[dict] = rescue_off_vocabulary(ledger, boards)
    # A hand reopen of an expired item is a status move the loop did not
    # make; the tag comes off here the way `needs-human` rides its move.
    for iid in clear_expired_on_reopen(ledger, boards):
        moved.append({"item_id": iid, "from": "done", "to": TRIAGE_POOL_STATUS})
    # Likewise a held confirmation a human moved into the pool by hand: that
    # is the release, and the pass below must not move it back.
    release_held_confirmations(ledger, boards, hand_moves_only=True)
    current = {i.id: i.status for i in open_items(boards)}
    for iid, want in desired_statuses(ledger, boards, open_round_items=open_round_items).items():
        status, why = want[0], want[1]
        needs_human = bool(want[2]) if len(want) > 2 else False
        if current.get(iid) == status:
            continue
        if set_status(iid, status, why,
                      add_tags=(NEEDS_HUMAN_TAG,) if needs_human else (),
                      remove_tags=() if needs_human else (NEEDS_HUMAN_TAG,)):
            S.append_event({"event": "status_moved", "item_id": iid, "from": current.get(iid),
                            "to": status, "reason": why[:200]}, path=ledger)
            moved.append({"item_id": iid, "from": current.get(iid), "to": status})
    return moved


def reopen_item(item_id: int, reason: str, *, ledger: Path | None = None) -> dict:
    """Grant an item another unattended implement attempt. Records why, in the
    ledger and on the item, so the second attempt is auditable as a decision
    rather than a retry loop."""
    reason = " ".join(str(reason or "").split()).strip()
    if not reason:
        raise ValueError("a reason is required — a reopen is a decision, and decisions are recorded")
    ledger = ledger or LEDGER_DEFAULT()
    if int(item_id) not in {int(d["item_id"]) for d in _ledger_events(ledger, "backlog_implement")}:
        raise ValueError(f"#{item_id} has no implement attempt on record; nothing to reopen")
    from scripts.automod import state as S
    S.append_event({"event": "backlog_implement", "item_id": int(item_id), "phase": "reopened",
                    "reason": reason}, path=ledger)
    note_item(item_id, f"reopened for a second automod attempt: {reason}")
    return {"item_id": int(item_id), "reopened": True, "reason": reason}


def note_item(item_id: int, text: str) -> bool:
    """Append one activity-log line to an open item. False if not found."""
    for item in open_items(None):
        if item.id == int(item_id):
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
            fm, body = _split_frontmatter(item.path.read_text(encoding="utf-8"))
            log = list(fm.get("activity_log") or [])
            log.append(f"**{stamp}** — {text}")
            fm["activity_log"] = log
            fm["updated"] = stamp
            item.path.write_text(
                f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
                f"---\n{body}", encoding="utf-8")
            return True
    return False


def LEDGER_DEFAULT() -> Path:
    from scripts.automod import state as S
    return S.LEDGER_PATH


def is_umbrella(item: Item) -> bool:
    return "umbrella" in item.tags or bool(item.members)


def is_grouped(item: Item) -> bool:
    return item.group is not None


# When `form_umbrellas` was switched off for the backlog sweep (config commit
# 52958ff). A group triage run with umbrellas off records every would-be fold
# as `keep`, and a `keep` is what `clusterable_items` reads as "the sameness
# question was answered" — so the sweep's runs silently removed their items
# from clustering for good. Rows carry `form_umbrellas` since 2026-09-16; a
# row from before that carries nothing, and this stamp says which of those
# ran with folding off.
UMBRELLAS_OFF_SINCE = "2026-09-15T21:39:07"


def group_triaged_ids(ledger: Path, *, binding_only: bool = False) -> set[int]:
    """Every id a group triage judged, whatever the verdict.

    `binding_only` drops the runs that could not have folded anything —
    recorded with `form_umbrellas: false`, or undated-by-flag and after
    `UMBRELLAS_OFF_SINCE` — so the clusterer may offer those items again once
    umbrellas are back on. Every other reader wants the full set.
    """
    out: set[int] = set()
    for d in _ledger_events(ledger, "backlog_group_triage", require_item=False):
        if binding_only:
            flag = d.get("form_umbrellas")
            stamp = str(d.get("created_at") or d.get("ts") or "")
            if flag is False or (flag is None and stamp[:19] >= UMBRELLAS_OFF_SINCE):
                continue
        judged = d.get("judged") or {}
        if isinstance(judged, dict):
            for k in judged:
                try:
                    out.add(int(k))
                except (TypeError, ValueError):
                    continue
    return out


def is_self_spawned(item: Item) -> bool:
    """Did the triage/implement loop — or the eval digest — write this item?
    The quarantine test.

    Deliberately narrower than `is_loop_spawned`: see `LOOP_SPAWN_TAGS`.
    """
    tags = quarantine_tags()
    return any(t in tags for t in item.tags)


def is_loop_spawned(item: Item) -> bool:
    """Did any unattended pass write this item — triage, implement, review or
    the eval digest?

    The bound-the-board test. Everything under it is subject to expiry and
    shows on the scorecard's open self-spawned gauge, whether or not it is
    held out of the triage pool.

    Delegates to `loop_spawn_tag`, which recognises the `spawned-by-` prefix the
    mint really uses and not only the enumerated names; see the asymmetry note
    on `LOOP_SPAWN_TAGS`.
    """
    return bool(loop_spawn_tag(item.tags))


# A self-filed item that nothing picked up. Closed, not deleted: the file
# stays on disk with the tag, and a human setting its status back to `draft`
# reopens it (the reconciler strips the tag and releases it into the pool).
EXPIRED_TAG = "expired"
# A closed proposal the loop tried and found wanting (`rejected` outcome).
# On the item so a later triage of the same idea can see it was measured.
REJECTED_TAG = "rejected"
# Never expired: a member's fate is its umbrella's, an umbrella is confirmed
# work, a `needs-human` item is waiting on a decision, and an expired item
# has already been judged once. A swept or parked item has been read and
# ranked by the sweep, which is the exit expiry used to be.
EXPIRY_EXEMPT_TAGS = frozenset({"grouped", "umbrella", NEEDS_HUMAN_TAG, EXPIRED_TAG, HELD_TAG,
                                SWEPT_TAG, PARKED_TAG})


def expired_ids(ledger: Path) -> set[int]:
    return {int(d["item_id"]) for d in _ledger_events(ledger, "backlog_expired")}


def group_kept_ids(ledger: Path) -> set[int]:
    """Items a group triage judged `keep` — distinct work, back in the single
    pool. Written by the cluster half of the loop; read here so a kept item
    is released from quarantine the moment it is judged."""
    out: set[int] = set()
    for d in _ledger_events(ledger, "backlog_group_triage", require_item=False):
        judged = d.get("judged") or {}
        if isinstance(judged, dict):
            for k, v in judged.items():
                if str(v) == "keep":
                    try:
                        out.add(int(k))
                    except (TypeError, ValueError):
                        continue
    return out


def group_keep_note(ledger: Path, item_id: int) -> dict | None:
    """The latest group triage that judged this item `keep`: `{cluster_id, ts}`.
    `group_kept_ids` answers whether; this keeps the when, which the single
    triage of a kept item is shown."""
    found: dict | None = None
    for d in _ledger_events(ledger, "backlog_group_triage", require_item=False):
        judged = d.get("judged") or {}
        if isinstance(judged, dict) and str(judged.get(str(int(item_id)))) == "keep":
            found = {"cluster_id": str(d.get("cluster_id") or ""), "ts": d.get("ts")}
    return found


def swept_ids(ledger: Path) -> set[int]:
    """Every id a sweep judged, whatever the verdict (`backlog_sweep` rows)."""
    out: set[int] = set()
    for d in _ledger_events(ledger, "backlog_sweep", require_item=False):
        judged = d.get("judged") or {}
        if isinstance(judged, dict):
            for k in judged:
                try:
                    out.add(int(k))
                except (TypeError, ValueError):
                    continue
    return out


def released_ids(ledger: Path) -> set[int]:
    """Self-filed items the pass may triage after all: expired and reopened,
    judged `keep` by a group triage, ranked by a sweep, or re-triaged after a
    spent attempt."""
    return (expired_ids(ledger) | group_kept_ids(ledger) | swept_ids(ledger)
            | set(retriage_marks(ledger)))


# ── Blockers ────────────────────────────────────────────────────────────────
#
# The one item an implement round may still file: a finding that stops one of
# its clauses from becoming true, tagged `spawned-by-autocode` + `blocker`,
# first line "Blocks #N", written as a handoff a fresh session can execute
# alone. The round defers the clause to it. Until 2026-09-14 the only reader
# of the tag was write-time dedupe, so a blocker was an ordinary self-spawn:
# quarantined, so no triage ever gave it a contract, so it never reached
# `up_next` — and then expired at seven days, orphaning the clause that waited
# on it. On that day 13 of the 101 quarantined drafts were blockers, four of
# them in front of items already in the implement pool.
#
# A blocker is *live* while the item it blocks is open. A live blocker is
# triaged (first), never held behind the depth gate, taken early by autocode
# and never expired. Once the blocked item closes it is an ordinary self-spawn
# again — quarantined and bound by expiry — rather than closed: the finding
# may be real without the item that surfaced it (#987, a bench that gave a
# destructive prompt live Bash, was filed as a blocker of a retrieval item).
BLOCKER_TAG = "blocker"
_BLOCKS_RE = re.compile(r"^\W*blocks\s+#(\d+)", re.IGNORECASE)


def blocker_targets(ledger: Path | None = None, *, events: list[dict] | None = None) -> dict[int, int]:
    """`{blocker_id: blocked_item_id}` off implement rows' `spawned` — the ids
    an implement round filed, which by its prompt are blockers only. Latest
    row wins. `events` is a caller's own read of the ledger (the scorecard)."""
    rows = ([e for e in events if e.get("event") == "backlog_implement" and e.get("item_id") is not None]
            if events is not None else _ledger_events(ledger, "backlog_implement"))
    out: dict[int, int] = {}
    for d in rows:
        for s in d.get("spawned") or []:
            try:
                out[int(s)] = int(d["item_id"])
            except (TypeError, ValueError):
                continue
    return out


def blocked_item_of(item: Item, targets: dict[int, int] | None = None) -> int | None:
    """The id a blocker blocks: its own "Blocks #N" (title, then first body
    line — what the prompt asks for), else the round that filed it. None when
    neither says."""
    first = next((l for l in item.body.splitlines()
                  if l.strip() and not l.startswith("# ")), "")
    for text in (item.name, first):
        m = _BLOCKS_RE.match(text or "")
        if m:
            return int(m.group(1))
    return (targets or {}).get(item.id)


def live_blockers(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                  items: list[Item] | None = None,
                  backlog_dir: Path | None = None) -> dict[int, int | None]:
    """`{blocker_id: blocked_id}` for open `blocker` items whose blocked item is
    still open. A blocker whose target cannot be named, or names no item on
    disk, counts as live: triage is the pass that should find that out, and
    a blocker judged once is out of expiry's reach anyway.

    Targets are looked up by id on every board — a blocker's target need not
    share its board, and a board-filtered read would call a closed target
    missing, and so live. `backlog_dir` is for a caller that resolves the
    vault at call time (the dashboard).
    """
    root = backlog_dir or BACKLOG_DIR

    def status_of(iid: int) -> str | None:
        for path in sorted(root.glob(f"{int(iid)}-*.md")):
            target = load_item(path)
            if target is not None:
                return target.status
        return None

    candidates = [i for i in (items if items is not None else open_items(boards, backlog_dir=backlog_dir))
                  if BLOCKER_TAG in i.tags and i.status in OPEN_STATUSES and not is_grouped(i)]
    if not candidates:
        return {}
    targets = blocker_targets(ledger)
    out: dict[int, int | None] = {}
    for item in candidates:
        live, blocked = blocker_liveness(item, targets, status_of)
        if live:
            out[item.id] = blocked
    return out


def blocker_liveness(item: Item, targets: dict[int, int], status_of) -> tuple[bool, int | None]:
    """`(live, blocked_id)` for one item — the single rule `live_blockers` and
    the scorecard's over-bound gauge both apply. `status_of(id)` returns the
    blocked item's status, or None when there is no such item.

    A member folded under an umbrella is not live: its fate is the umbrella's,
    single triage never reaches it, and counting it would keep
    `board_health`'s untriaged blockers from ever draining."""
    if BLOCKER_TAG not in item.tags or item.status not in OPEN_STATUSES or is_grouped(item):
        return False, None
    blocked = blocked_item_of(item, targets)
    if blocked is None:
        return True, None
    status = status_of(blocked)
    return (status is None or status in OPEN_STATUSES), blocked


def is_quarantined(item: Item, *, released: frozenset[int] | set[int] = frozenset(),
                   live: frozenset[int] | set[int] | dict = frozenset()) -> bool:
    """A self-filed item is not a single-item triage candidate.

    Triage asks one question: does this old claim still describe the system?
    An item this loop filed, from a check it ran against live code with file
    paths and line numbers, cannot answer it — it is not stale, by
    construction. Re-asking costs a 90-turn session to re-confirm what the
    previous session proved.

    That waste is not the reason for this gate, though. The reason is that
    `select_candidate` reads open items and `OPEN_STATUSES` includes `draft`,
    which is the status `backlog_write_task` writes — so every item triage
    filed re-entered the queue it came out of. Measured over the loop's first
    48 hours: 40 triage runs closed 28 items and filed 78, a reproduction
    number of 1.95. Each run replaced itself with two, and at a 30-minute
    cadence that is +46 open items a day, diverging regardless of how long it
    runs or how good the verdicts are. The open board went 19 -> 122 and 110
    of those 122 were the loop's own output. No cap on spawns per run fixes
    that shape; only cutting the edge does.

    Age does NOT release an item any more. The first cut let a spawned item
    back into the pool at 30 days, and by 2026-09-11 that was 291 items due
    to re-enter triage in October, each spawning ~2 more. The exits now are
    the ones that do not re-enter the queue they came out of: the nightly
    clustering pass (which ignores quarantine — its question is consolidation,
    not staleness), a group triage that judges the item `keep`, expiry
    (`expire_stale_spawns`), and a human reopening an expired item. Those are
    `released`.

    A live blocker (`live`, from `live_blockers`) is not quarantined either:
    the edge this gate cuts is triage feeding itself, and a blocker is capped
    at one per implement round and gates work already in the pool.
    """
    return (is_self_spawned(item) and int(item.id) not in released
            and int(item.id) not in live)


# ── triage claims: two triage turns at once ─────────────────────────────
#
# A triage turn takes minutes and writes nothing on its item until it ends, so
# with `workers.sources.autotriage.max_inflight` > 1 the second run's
# selection would pick exactly what the first is reading — the same single
# item, the same cluster, the same sweep batch — and the board would get two
# verdicts, or two umbrellas over one theme. Implement has `in_progress` for
# this; triage has no status of its own, and inventing one would put a fifth
# word into a vocabulary five lists agree on. The claim is in memory instead:
# both runs live in the backend process, and a restart that loses the set
# also kills the turns that held it.

_TRIAGE_CLAIMS: dict[str, set[int]] = {}
_TRIAGE_CLAIMS_LOCK = threading.Lock()


def claim_for_triage(token: str, ids) -> None:
    with _TRIAGE_CLAIMS_LOCK:
        _TRIAGE_CLAIMS.setdefault(str(token), set()).update(int(i) for i in ids)


def release_triage_claim(token: str) -> None:
    with _TRIAGE_CLAIMS_LOCK:
        _TRIAGE_CLAIMS.pop(str(token), None)


def triage_claimed_ids() -> set[int]:
    """Ids some triage turn in this process is reading right now."""
    with _TRIAGE_CLAIMS_LOCK:
        return set().union(*_TRIAGE_CLAIMS.values()) if _TRIAGE_CLAIMS else set()


def triage_pool(ledger: Path,
                boards: tuple[str, ...] | None = DEFAULT_BOARDS
                ) -> tuple[list[Item], int]:
    """Untriaged open items, and how many were held back by quarantine.

    The count is returned rather than logged so the caller can say *why* it
    has nothing to do. "Every open backlog item has been triaged" and "there
    are 106 items I filed myself and may not re-triage yet" are different
    states, and a pass that reports the first while in the second is how you
    stop noticing that the board is growing.
    """
    seen = triaged_ids(ledger)
    released = released_ids(ledger)
    items = open_items(boards)
    live = live_blockers(ledger, boards, items=items)
    # `draft` only: that is where an item waits to be judged. `up_next` is the
    # implement pool and `in_progress` is a round in flight; an untriaged item
    # in either is a dead state the reconciler moves back here.
    # A parked item was read by a sweep and judged not worth a round: out of
    # every pool until a person removes the tag.
    claimed = triage_claimed_ids()
    untriaged = [i for i in items
                 if i.id not in seen and i.id not in claimed
                 and i.status == TRIAGE_POOL_STATUS
                 and not is_grouped(i) and not is_parked(i)]
    fresh = [i for i in untriaged if not is_quarantined(i, released=released, live=live)]
    return fresh, len(untriaged) - len(fresh)


def expire_stale_spawns(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                        enabled: bool = True,
                        max_age_days: int | None = None) -> list[dict]:
    """Close self-filed drafts that nothing picked up in `max_age_days`
    (default `spawn_expiry_days()`).

    The hard bound on the board. Everything that could have taken the item
    has had its window: clustering, group triage, a human. Closing it is the
    honest record that nothing did — the text stays, tagged, and a hand
    reopen brings it back (see `clear_expired_on_reopen`). Human-authored
    items are never touched: the tag test is `is_self_spawned`, not `draft`.
    """
    if not enabled:
        return []
    if max_age_days is None:
        max_age_days = spawn_expiry_days()
    from scripts.automod import state as S
    seen = triaged_ids(ledger)
    judged = set(seen) | expired_ids(ledger) | group_kept_ids(ledger)
    judged |= {int(d["item_id"]) for d in _ledger_events(ledger, "backlog_implement")}
    items = open_items(boards)
    # A clause is deferred to a live blocker; closing it orphans the clause.
    live = live_blockers(ledger, boards, items=items)
    out: list[dict] = []
    for item in items:
        if not is_loop_spawned(item) or item.status != TRIAGE_POOL_STATUS:
            continue
        if item.id in judged or item.id in live or (set(item.tags) & EXPIRY_EXEMPT_TAGS):
            continue
        if item.age_days < int(max_age_days):
            continue
        fm, _ = _split_frontmatter(item.path.read_text(encoding="utf-8"))
        if fm.get(LANDED_MARKER) or any(fm.get(m) for m in _LEGACY_LANDED_MARKERS):
            continue
        why = (f"expired: self-filed {item.age_days} d ago and never triaged, clustered "
               f"or picked up; reopen by setting status back to draft")
        if not _apply_status(item.path, "done", why, add_tags=(EXPIRED_TAG,)):
            continue
        spawned_by = loop_spawn_tag(item.tags)
        S.append_event({"event": "backlog_expired", "item_id": item.id,
                        "age_days": item.age_days, "name": item.name[:200],
                        "spawned_by": spawned_by}, path=ledger)
        out.append({"item_id": item.id, "age_days": item.age_days})
    return out


def clear_expired_on_reopen(ledger: Path,
                            boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> list[int]:
    """An OPEN item still carrying `expired` was reopened by hand. Strip the
    tag and say so; `expired_ids` keeps it released, so it is now a triage
    candidate and is never expired twice — a reopen is a decision."""
    from scripts.automod import state as S
    out: list[int] = []
    for item in open_items(boards):
        if EXPIRED_TAG not in item.tags:
            continue
        if tag_item(item.id, remove=(EXPIRED_TAG,)):
            note_item(item.id, "reopened by hand after expiry; back in the triage pool")
            S.append_event({"event": "backlog_unexpired", "item_id": item.id}, path=ledger)
            out.append(item.id)
    return out


def select_candidate(ledger: Path,
                     boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> Item | None:
    """Oldest untriaged open item first, skipping this loop's own fresh output.

    Oldest-first deliberately: age is the best available proxy for staleness,
    and the point of this pipeline is to find out which old items are still
    real. Priority ordering would front-load the items most likely to be
    genuine, which is exactly backwards for a first pass over a stale backlog.

    Oldest-first also means the quarantine in `is_quarantined` is not merely
    a throttle. Self-filed items sort to the back, so without it the pass
    works the real backlog first and only then starts eating its own tail —
    which reads as healthy right up to the moment there is nothing else left.
    On 2026-09-08 that moment was three hours away: 6 of the 112 untriaged
    open items predated the loop.

    A live blocker goes ahead of the oldest: it is not a stale claim but the
    precondition of a clause some round already deferred.

    Since the sweep (2026-09-15), a ranked item goes ahead of an unranked one
    and a `high` ahead of a `medium`: the sweep has read the whole board
    once, so oldest-first is no longer the only signal, and a single triage
    spends a 90-turn session writing a contract — that session belongs to
    the item most worth landing. Age still orders items of one rank.

    Since 2026-09-16 the human's `priority` sorts above the sweep's rank: a
    `high` item unread by any sweep goes before a `low` one the sweep called
    worth `high`. A `high` item goes before a live blocker too — it is the
    one a person asked for next; the blocker still goes before everything
    medium or low, being a dependency rather than a preference.
    """
    candidates, _held = triage_pool(ledger, boards)
    if not candidates:
        return None
    live = live_blockers(ledger, boards, items=candidates)
    # Within `high` the sweep's rank does not apply and the newest goes
    # first (`recency_key`): the tag is the ranking.
    return sorted(candidates, key=lambda i: (not is_high(i), i.id not in live, priority_key(i),
                                             rank_key(i) if not is_high(i) else (0, 0, 0),
                                             recency_key(i)))[0]


def select_urgent(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> Item | None:
    """The `high` draft single triage takes before a sweep batch or a
    cluster, or None. `select_candidate`'s pick when that pick is high."""
    pick = select_candidate(ledger, boards)
    return pick if pick is not None and is_high(pick) else None


def select_confirmed(ledger: Path,
                     boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> tuple[Item, dict] | None:
    """Oldest still-open `confirmed` item that no implementation turn has run for.

    Returns (item, triage_event) — the event, not just the id, because the
    ACCEPTANCE recorded at triage is the contract the implementer is held to.
    An item confirmed with no acceptance check is not ready to implement
    unattended; it is skipped here rather than guessed at.
    """
    outcomes = implement_outcomes(ledger)
    ready = ready_confirmed(ledger, boards, outcomes=outcomes)
    if not ready:
        return None
    # Nearest to landing first: a re-offer whose last graded review met
    # every clause has one small task left (a test across a seam, a clause
    # amendment to ratify) and lands in one gate; a fresh item costs an hour
    # and two review attempts. On 2026-09-11 five such re-offers sat behind
    # fresh umbrellas that each took the hour and aborted. Then fresh
    # confirmations before other re-offers — oldest-first alone let a
    # sent-back item be re-picked on the very next round for as long as its
    # cap allowed, monopolising the loop while the rest of the pool waited.
    # Cut 3 of senses-not-supervision: when the board steward is APPLYING,
    # its pick is the order. It read the same ledger and the same board and
    # was asked the same question, and one judgment with reasons beats three
    # sort keys. Only a pick that is in `ready` counts — the pick is advice,
    # the readiness rules are facts.
    picked = steward_pick()
    if picked is not None:
        for item, ev in ready:
            if item.id == picked:
                return item, ev
    near = set(last_review_all_met(ledger))
    # A FIRST re-offer whose branch still holds the work belongs in the same
    # tier, for the same reason: it is a fix cycle, not an hour.
    near |= first_reoffer_with_a_branch(ledger, outcomes)
    # Within each tier, a live blocker first: landing it un-defers a clause of
    # an item some round already carried most of the way. Inside the tiers,
    # not above them — a sent-back blocker ahead of fresh confirmations would
    # be re-picked every round until its cap, the monopoly they exist to stop.
    live = live_blockers(ledger, boards, items=[i for i, _ in ready])
    # Then the sweep's rank (2026-09-15): a `high`/`small` item before an
    # unranked one before a `low`/`large` one, and within a rank the shorter
    # contract first — a 12-clause umbrella lands one time in five, a
    # 4-clause single one in three. The near tier stays above it: one fix
    # cycle is cheaper than any fresh round whatever its rank.
    # The human's `priority` sits above all of that (2026-09-16): a fresh
    # `high` runs before a `low` that is one fix cycle from landing, because
    # the tag is the one thing on the board a person set on purpose. Within
    # medium and low the tiers above still hold; within `high` only the near
    # tier precedes recency: the sweep's rank, the contract length and
    # fresh-before-re-offer do not apply. The last one is deliberate
    # (2026-09-17): with 26 never-attempted highs queued, it put the one
    # item a person had just raised — re-offered with its branch — 27th.
    # A high re-offer cannot monopolise the loop the way that key guards
    # against elsewhere: its review-retry and incomplete caps bound it.
    def _key(pair):
        item, ev = pair
        high = is_high(item)
        return (priority_key(item),
                item.id not in near,
                rank_key(item) if not high else (0, 0, 0),
                (item.clause_count or len(acceptance_clauses_of(ev))) if not high else 0,
                item.id in outcomes if not high else False,
                item.id not in live,
                recency_key(item))
    return sorted(ready, key=_key)[0]


def ready_confirmed(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                    items: list[Item] | None = None,
                    outcomes: dict | None = None) -> list[tuple[Item, dict]]:
    """Every item autocode would actually take, unordered: `(item, triage_event)`.

    `select_confirmed` orders this; the triage depth gate and `board_health`
    count it. One definition, because a gate that counted raw `up_next` would
    count grouped members, human-only items, items with no acceptance and
    spent attempts — none of which a round will ever start.
    """
    confirmed = confirmed_verdicts(ledger)
    if outcomes is None:
        outcomes = implement_outcomes(ledger)
    done = {iid for iid, (verdict, _) in outcomes.items() if verdict == "spent"}
    ready = []
    for item in (items if items is not None else open_items(boards)):
        ev = confirmed.get(item.id)
        if not ev or item.id in done:
            continue
        # The board is the state machine now: a confirmed item the loop may
        # take sits in `up_next`, and only there. A human parks one anywhere
        # else to keep it out of the loop's hands.
        if item.status != IMPLEMENT_POOL_STATUS:
            continue
        # A member is never implemented on its own: its umbrella carries the
        # contract and closes it.
        if is_grouped(item):
            continue
        if not acceptance_text(ev.get("acceptance")):
            continue
        if is_human_only(ev.get("acceptance")):
            continue
        ready.append((item, ev))
    return ready


def _iso_ts(value, *, naive_local: bool = False) -> float | None:
    """An ISO stamp as epoch seconds. A naive stamp is UTC unless
    `naive_local` says its writer used the machine's local clock — the board
    has both: `agent_mcp/backlog.py` and the Mission Control router write
    `created`/`updated` with `datetime.now()`, and every closer in this module
    writes `completed`/`updated` in UTC."""
    s = str(value or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.astimezone() if naive_local else dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def landed_items_trailing(ledger: Path, days: float = 7, *, now: float | None = None) -> int:
    """Distinct items that reached a landing in the last `days`: a settled code
    promotion, or an `ok` `vault_land`.

    Items, not rounds. `vault_land` and `promoted` rows overcount the drain by
    about three times, because a re-offered item lands again — #487 landed
    several times in three days. The depth gate sizes the implement pool by
    this, and a pool sized by rounds would be three times too deep.
    """
    now = time.time() if now is None else now
    since = now - float(days) * 86400
    ids: set[int] = set()
    for landing in settled_landings(ledger):
        t = _iso_ts(landing.get("settled_at"))
        if t is not None and since <= t <= now:
            ids.add(int(landing["item_id"]))
    for d in _ledger_events(ledger, "vault_land"):
        if d.get("ok") and since <= float(d.get("ts") or 0) <= now:
            ids.add(int(d["item_id"]))
    return len(ids)


# The implement pool is not deeper than this many ready items plus the week's
# drain. Below it, a triage confirmation is work a round will take within about
# a week; above it, it is inventory that ages until a re-triage finds it stale.
IMPLEMENT_POOL_FLOOR = 20


def implement_pool_bound(ledger: Path, *, floor: int = IMPLEMENT_POOL_FLOOR,
                         now: float | None = None) -> dict:
    """`{bound, floor, landed_items_7d}` — the single-item triage depth gate's
    bound, derived from the trailing landing rate (Alan, 2026-09-13)."""
    landed = landed_items_trailing(ledger, 7, now=now)
    return {"bound": max(int(floor), landed), "floor": int(floor), "landed_items_7d": landed}


# ── Held confirmations ──────────────────────────────────────────────────────
#
# The depth gate's first cut (2026-09-13) paused single-item triage outright
# while the pool was full. Triage does two things, though: it confirms, which
# is what fills the pool, and it retires — `stale` and `already_done` were 23
# of its 103 verdicts the day before the gate, the largest closer the loop
# had. Pausing the pass stopped both. In the first ten hours under the gate
# the loop filed 6 items and closed 3, and single triage stayed off because
# ready (84, then 103) sat far above a bound (57) that drains at about eight
# landings a day. The gate now holds a confirmation instead of refusing the
# turn: the verdict is recorded, the item stays in `draft`, and it enters
# `up_next` oldest-first as room opens.

def held_confirmations(ledger: Path) -> dict[int, float]:
    """`{item_id: ts}` of confirmations still waiting for room, oldest first
    by `ts`. Held means the item's latest `confirmed` verdict carries
    `held: true` and no `backlog_confirm_released` row has followed it."""
    latest = confirmed_verdicts(ledger)
    released: dict[int, float] = {}
    for d in _ledger_events(ledger, "backlog_confirm_released"):
        released[int(d["item_id"])] = float(d.get("ts") or 0)
    out: dict[int, float] = {}
    for iid, ev in latest.items():
        if not ev.get("held"):
            continue
        ts = float(ev.get("ts") or 0)
        if released.get(iid, float("-inf")) >= ts:
            continue
        out[iid] = ts
    return out


def implement_pool_full(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                        floor: int = IMPLEMENT_POOL_FLOOR, now: float | None = None) -> dict:
    """`{full, ready, bound, floor, landed_items_7d}` — the depth gate's one
    question, asked the same way by the triage pass and the releaser."""
    pool = implement_pool_bound(ledger, floor=floor, now=now)
    ready = len(ready_confirmed(ledger, boards))
    return {"full": ready >= pool["bound"], "ready": ready, **pool}


def release_held_confirmations(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                               floor: int = IMPLEMENT_POOL_FLOOR, enabled: bool = True,
                               hand_moves_only: bool = False,
                               now: float | None = None) -> list[dict]:
    """Move held confirmations into `up_next`, oldest first, while the pool
    has room. Returns one `{item_id, reason, moved}` per release.

    A held item a human already moved out of `draft` is released where it
    stands — the move was the decision, and the reconciler must not undo it
    (`hand_moves_only` does just that part, for `reconcile_statuses`).
    `enabled=False` is the kill switch: everything held is released at once,
    so switching holding off never strands an item.
    """
    from scripts.automod import state as S
    held = held_confirmations(ledger)
    if not held:
        return []
    open_by_id = {i.id: i for i in open_items(boards)}
    out: list[dict] = []

    def _release(iid: int, why: str, move: bool) -> None:
        S.append_event({"event": "backlog_confirm_released", "item_id": iid,
                        "reason": why[:200], "moved": move}, path=ledger)
        if move:
            if set_status(iid, IMPLEMENT_POOL_STATUS, f"released from hold: {why}",
                          remove_tags=(HELD_TAG,)):
                S.append_event({"event": "status_moved", "item_id": iid,
                                "from": TRIAGE_POOL_STATUS, "to": IMPLEMENT_POOL_STATUS,
                                "reason": f"released from hold: {why}"[:200]}, path=ledger)
        else:
            tag_item(iid, remove=(HELD_TAG,))
        out.append({"item_id": iid, "reason": why, "moved": move})

    waiting: list[int] = []
    for iid, _ts in sorted(held.items(), key=lambda kv: kv[1]):
        item = open_by_id.get(iid)
        if item is None:
            continue
        if item.status != TRIAGE_POOL_STATUS:
            _release(iid, f"moved to {item.status} by hand", move=False)
            continue
        waiting.append(iid)
    if hand_moves_only or not waiting:
        return out
    # A live blocker does not wait for room: it is the precondition of work
    # the pool already holds, not more inventory for it. Nor does a `high`
    # item (2026-09-16): a person raising a held item to high is asking for
    # it next, and the next pass moves it.
    live = live_blockers(ledger, boards, items=[open_by_id[i] for i in waiting])
    for iid in [i for i in waiting if i in live]:
        of = f"#{live[iid]}" if live[iid] is not None else "an unnamed item"
        _release(iid, f"a live blocker of {of}; blockers are never held", move=True)
    waiting = [i for i in waiting if i not in live]
    for iid in [i for i in waiting if is_high(open_by_id[i])]:
        _release(iid, "priority high: never held by the depth gate", move=True)
    waiting = [i for i in waiting if not is_high(open_by_id[i])]
    if not waiting:
        return out
    # Best first, then oldest: room in the pool goes to the confirmation the
    # human prioritised highest, then to the one the sweep ranked highest,
    # not to whichever arrived first.
    waiting.sort(key=lambda i: (priority_key(open_by_id[i]), rank_key(open_by_id[i]), held[i]))
    if enabled:
        gate = implement_pool_full(ledger, boards, floor=floor, now=now)
        room = gate["bound"] - gate["ready"]
        why = f"the implement pool has room ({gate['ready']} ready < bound {gate['bound']})"
    else:
        room, why = len(waiting), "holding confirmations is switched off"
    for iid in waiting[:max(0, room)]:
        _release(iid, why, move=True)
    return out


STEWARD_PICK_MAX_AGE_S = 2 * 3600


def steward_pick(path: Path | None = None, *, now: float | None = None) -> int | None:
    """The board steward's `next_pick`, when it is applying and fresh.

    None unless `workers.sources.board-steward.apply` is on: in dry-run the
    steward records what it WOULD pick and the ordering below stays the
    order. Stale (older than two of its intervals) means the steward has
    stopped running, and a pick from a board that has since moved is worse
    than the sort.
    """
    try:
        from app.config import CONFIG
        cfg = ((CONFIG.get("workers") or {}).get("sources") or {}).get("board-steward") or {}
        if not cfg.get("apply", False):
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        from scripts.automod import state as S
        p = path or (S.STATE_DIR / "steward_pick.json")
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if (now or time.time()) - float(d.get("ts") or 0) > STEWARD_PICK_MAX_AGE_S:
            return None
        iid = int(d.get("item_id") or 0)
        return iid or None
    except Exception:  # noqa: BLE001
        return None


def rounds_for_item(ledger: Path) -> dict[int, list[str]]:
    """Round ids each item has had, oldest first.

    Off `backlog_implement` `started` rows, which carry both ids. Used for
    ordering only — a missing row costs a place in the queue, never a
    correctness property.
    """
    out: dict[int, list[str]] = {}
    for d in _ledger_events(ledger, "backlog_implement"):
        rid = str(d.get("round_id") or "")
        # Any phase that names the round. On the live ledger only `finished`
        # carries `round_id` — `started` has `budget`, `item_id`, `phase` —
        # so reading `started` alone returned nothing for every item and the
        # first-re-offer ordering (`first_reoffer_with_a_branch`) never fired.
        if not rid:
            continue
        try:
            iid = int(d["item_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if rid not in out.setdefault(iid, []):
            out[iid].append(rid)
    return out


def first_reoffer_with_a_branch(ledger: Path, outcomes: dict,
                                repo: Path | None = None) -> set[int]:
    """Items on their FIRST re-offer whose round branch still holds the work.

    `automod_start(from_branch=…)` resumes it, so the next round is a fix
    cycle rather than a fresh hour. Only the first: a second re-offer has
    already had its fix cycle, and letting it keep jumping the queue is the
    monopoly the oldest-first ordering exists to prevent.
    """
    from scripts.automod import worktree as W

    root = repo or W.LIVE_ROOT
    rounds = rounds_for_item(ledger)
    out: set[int] = set()
    for iid, (verdict, _detail) in (outcomes or {}).items():
        if verdict not in ("review_retry", "external", "incomplete", "infra"):
            continue
        ids = rounds.get(int(iid)) or []
        if len(ids) != 1:
            continue
        try:
            if W.branch_exists(root, f"automod/{ids[-1]}"):
                out.add(int(iid))
        except Exception:  # noqa: BLE001 — ordering is not correctness
            continue
    return out


def last_review_all_met(ledger: Path) -> set[int]:
    """Items whose most recent graded review found every clause `met`.

    The refusal, if any, was on a seam or a test-honesty finding — one
    change away from a pass. Read off the review events by item, newest
    graded row per item.
    """
    marks = retriage_marks(ledger)
    latest: dict[int, dict] = {}
    for d in _ledger_events(ledger, "review"):
        if d.get("ok") and d.get("clauses") and _after_mark(d, marks):
            latest[int(d["item_id"])] = d
    return {iid for iid, d in latest.items()
            if all(c.get("verdict") == "met" for c in d["clauses"])}


def tag_item(item_id: int, *, add: tuple[str, ...] = (), remove: tuple[str, ...] = ()) -> bool:
    """Add or remove tags on an open item without moving its status."""
    for item in open_items(None):
        if item.id == int(item_id):
            fm, body = _split_frontmatter(item.path.read_text(encoding="utf-8"))
            tags = normalize_tags(fm.get("tags"))
            new = [t for t in tags if t not in remove] + [t for t in add if t not in tags]
            if new == tags and isinstance(fm.get("tags"), list):
                return False
            fm["tags"] = new
            fm["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
            item.path.write_text(
                f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
                f"---\n{body}", encoding="utf-8")
            return True
    return False


def record_verdict(item: Item, verdict: str, evidence: str, *,
                   check: str = "", close: bool = False,
                   spawned: list[int] | tuple[int, ...] = (),
                   merged: list[int] | tuple[int, ...] = (),
                   acceptance: str = "",
                   acceptance_clauses: list[str] | tuple[str, ...] = (),
                   human_clauses: list[str] | tuple[str, ...] = (),
                   dropped_clauses: list[str] | tuple[str, ...] = (),
                   hold: bool = False) -> Path:
    """Append the verdict to the item's activity log, optionally closing it.

    Always writes the evidence, never just the conclusion. An item closed as
    stale with no stated reason is indistinguishable from one closed by
    mistake, and the whole value of this pass is that a human can audit it
    later.

    `hold` parks a `confirmed` item in `draft`, tagged `confirmed-held`,
    instead of moving it into the implement pool — the caller found the pool
    full. The caller's ledger row must carry `held: true`; that row, not the
    tag, is what `held_confirmations` reads.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    text = item.path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    entry = f"**{stamp}** — autotriage: **{verdict}**. {evidence.strip()}"
    if check:
        entry += f" Check: `{check}`"
    if spawned:
        entry += " Filed as new items: " + ", ".join(f"#{i}" for i in spawned) + "."
    if merged:
        entry += " Merged findings into: " + ", ".join(f"#{i}" for i in merged) + "."
    log = list(fm.get("activity_log") or [])
    log.append(entry)
    fm["activity_log"] = log
    fm["updated"] = stamp

    if close and verdict in RETIRING:
        fm["status"] = "done"
        # Every other closer stamps it (`_apply_status`, `close_landed`), and
        # `board_health`'s outflow reads it. This one did not, so a day's
        # triage retirements dated from whatever next touched the file.
        fm["completed"] = stamp
        fm["autotriage_retired"] = verdict
    elif verdict == "confirmed" and fm.get("status") != "done":
        if hold:
            # The pool is full: judged real, parked until a slot opens.
            tags = [str(t) for t in (fm.get("tags") or [])]
            fm["tags"] = tags + ([HELD_TAG] if HELD_TAG not in tags else [])
        else:
            # Into the implement pool. Before 2026-09-09 a confirmed item
            # stayed wherever it was, and #353 landed while still `draft`.
            fm["status"] = IMPLEMENT_POOL_STATUS
    clauses, past_cap = cap_new_clauses(acceptance_clauses)
    past_cap = list(dropped_clauses) + past_cap
    # Backstop for the rule the prompt states: a clause whose evidence only
    # arrives with time cannot be graded before landing, and holding a round
    # to one can only refuse it. #859 was refused twice on "needs a day of
    # post-change traffic" with its mechanism complete on both commits.
    clauses, moved_later = split_post_landing_clauses(clauses)
    human = clean_clauses(list(human_clauses) + moved_later)
    if verdict == "confirmed" and clauses:
        # On the item, not only in the ledger: the review rung reads the
        # contract from disk, and a human editing the clauses here is editing
        # what the grader holds the next round to.
        fm["acceptance_clauses"] = clauses
    if verdict == "confirmed" and human:
        # What a person must do before this is done. Kept apart from the
        # clauses so no round is asked to fake an audit (#578's clause 5).
        fm["human_clauses"] = human

    section = (f"\n\n## Automod triage — {stamp[:10]}\n\n"
               f"**Verdict:** {verdict}\n\n{evidence.strip()}\n")
    if verdict == "confirmed" and hold:
        section += ("\n**Held:** the implement pool was full when this was confirmed. It stays "
                    "in `draft` and enters `up_next`, oldest first, when a slot opens.\n")
    if moved_later:
        section += ("\n**Moved to human clauses** (observable only after "
                    "landing, so no gate can judge them): "
                    + "; ".join(moved_later) + "\n")
    if check:
        section += f"\n**Premise check:**\n```\n{check.strip()}\n```\n"
    if spawned:
        section += "\n**Filed as new items:** " + ", ".join(f"#{i}" for i in spawned) + "\n"
    if merged:
        section += "\n**Merged findings into:** " + ", ".join(f"#{i}" for i in merged) + "\n"
    if acceptance.strip():
        # The item is the handoff. A contract that lives only in the ledger
        # and a transcript is one the next reader of the item never sees.
        section += f"\n**Acceptance — what must become true:**\n{acceptance.strip()}\n"
    if verdict == "confirmed" and human:
        section += ("\n**Needs a person before this closes:**\n"
                    + "\n".join(f"- {c}" for c in human) + "\n")
    if verdict == "confirmed" and clauses:
        section += "\n**Acceptance clauses** (graded one by one at the gate):\n" + "\n".join(
            f"{i}. {c}" for i, c in enumerate(clauses, 1)) + "\n"
    if verdict == "confirmed" and past_cap:
        section += (f"\n**Past the {MAX_CLAUSES}-clause budget — not graded, not part of the "
                    f"contract:**\n" + "\n".join(f"- {c}" for c in past_cap) + "\n")

    item.path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body.rstrip()}{section}",
        encoding="utf-8")
    return item.path


def select_cluster(ledger: Path, clusters: dict, *, min_size: int = 3, max_size: int = 4,
                   boards: tuple[str, ...] | None = DEFAULT_BOARDS
                   ) -> tuple[dict, list[Item]] | None:
    """The cluster a group triage should take: re-validated against disk.

    `clusters.json` is a night old by the time it is read, so every member
    is checked to still be an untriaged, ungrouped open draft; ids a group
    triage already judged are dropped (a cluster of `keep`s is never
    retaken); a cluster whose last group event was not `incomplete` is
    skipped. The survivors are trimmed to `max_size` keeping the `duplicates`
    pairs together, then highest priority, then oldest. The cluster whose
    best member carries the highest `priority` wins; the largest among
    equals (2026-09-16 — it was largest-first alone before).
    """
    seen = triaged_ids(ledger)
    judged = group_triaged_ids(ledger)
    by_id = {i.id: i for i in open_items(boards)
             if i.status == TRIAGE_POOL_STATUS and i.id not in seen
             and not is_grouped(i) and not is_umbrella(i) and not is_parked(i)
             and NEEDS_HUMAN_TAG not in i.tags and EXPIRED_TAG not in i.tags}
    last_group: dict[str, str] = {}
    for d in _ledger_events(ledger, "backlog_group_triage", require_item=False):
        last_group[str(d.get("cluster_id") or "")] = str(d.get("verdict") or d.get("outcome") or "done")
    best: tuple[dict, list[Item]] | None = None
    best_score: tuple[int, int] = (0, 0)
    claimed = triage_claimed_ids()
    for c in (clusters or {}).get("clusters") or []:
        cid = str(c.get("id") or "")
        if cid in last_group and last_group[cid] != INCOMPLETE:
            continue
        # A cluster another triage turn is reading any member of is skipped
        # whole: what is left of it would form a second group over the theme
        # the first is about to file an umbrella for.
        if claimed and any(int(i) in claimed for i in (c.get("item_ids") or [])):
            continue
        ids = [int(i) for i in (c.get("item_ids") or []) if int(i) in by_id and int(i) not in judged]
        if len(ids) < int(min_size):
            continue
        dup_ids: list[int] = []
        for pair in c.get("duplicates") or []:
            for i in pair:
                if int(i) in ids and int(i) not in dup_ids:
                    dup_ids.append(int(i))
        rest = sorted((i for i in ids if i not in dup_ids),
                      key=lambda i: (priority_key(by_id[i]), by_id[i].created or "9999", i))
        chosen = (dup_ids + rest)[:int(max_size)]
        members = [by_id[i] for i in sorted(chosen)]
        score = (-min(priority_key(m) for m in members), len(members))
        if best is None or score > best_score:
            best, best_score = (c, members), score
    return best


def _resolve_duplicates(verdicts: dict[int, dict], members: set[int]) -> dict[int, dict]:
    """Chains resolve to the terminal survivor (a→b→c ⇒ a→c); a cycle or a
    target outside the cluster becomes `keep`. A duplicate's target must
    itself be `fold` or `keep`."""
    # Resolved against the verdicts as given, never against the rewrites in
    # progress: with 4→5→4, rewriting 4 to `keep` first would let 5 resolve
    # to "duplicate of a keep" instead of to the cycle it is.
    orig = {i: dict(v) for i, v in verdicts.items()}
    out = {i: dict(v) for i, v in verdicts.items()}
    for i, v in orig.items():
        if v.get("verdict") != "duplicate_of":
            continue
        seen = {i}
        t = int(v.get("duplicate_of") or 0)
        while t in orig and orig[t].get("verdict") == "duplicate_of" and t not in seen:
            seen.add(t)
            t = int(orig[t].get("duplicate_of") or 0)
        if t not in members or t in seen or t == i or orig.get(t, {}).get("verdict") == "duplicate_of":
            out[i]["verdict"] = "keep"
            out[i]["duplicate_of"] = 0
            out[i]["evidence"] = (v.get("evidence") or "") + " (duplicate target unresolved; kept)"
        else:
            out[i]["duplicate_of"] = t
    return out


def record_group_verdict(cluster: dict, members: list[Item], verdicts: dict[int, dict],
                         umbrella: Item | None, umbrella_fields: dict, *,
                         session_id: str = "", spawned=(), merged=(),
                         extra: dict | None = None, hold: bool = False) -> dict:
    """Write a group triage's verdicts onto the items and the ledger.

    One `backlog_triage` row per member so every existing reader — the
    scorecard, `triaged_ids`, the status pipeline — sees ordinary verdicts,
    plus one `backlog_group_triage` summary keyed on the cluster.
    """
    from scripts.automod import state as S
    cid = str(cluster.get("id") or "")
    member_ids = {m.id for m in members}
    by_id = {m.id: m for m in members}
    verdicts = _resolve_duplicates(verdicts, member_ids)
    fold_ids = [i for i, v in verdicts.items() if v.get("verdict") == "fold"]
    umbrella_missing = bool(fold_ids) and umbrella is None
    if umbrella_missing:
        for i in fold_ids:
            verdicts[i]["verdict"] = "keep"
            verdicts[i]["evidence"] = (verdicts[i].get("evidence") or "") + " (no umbrella on disk; kept)"
        fold_ids = []
    counts = {"duplicates": 0, "retired": 0, "folded": 0, "kept": 0}
    judged: dict[str, str] = {}
    for i, v in verdicts.items():
        item = by_id.get(i)
        if item is None:
            continue
        verdict = str(v.get("verdict") or "keep")
        evidence = str(v.get("evidence") or "").strip()
        judged[str(i)] = verdict
        if verdict == "duplicate_of":
            t = int(v["duplicate_of"])
            record_verdict(item, "stale", f"duplicate of #{t}: {evidence}", close=True)
            update_frontmatter(item.path, {"duplicate_of": t})
            S.append_event({"event": "backlog_triage", "item_id": i, "verdict": "stale",
                            "closed": True, "duplicate_of": t, "group_cluster": cid,
                            "group_verdict": verdict, "evidence": evidence[:1000],
                            "session_id": session_id, "spawned": [], "auto": True}, path=_ledger_or(extra))
            counts["duplicates"] += 1
        elif verdict in RETIRING:
            record_verdict(item, verdict, evidence, close=True)
            S.append_event({"event": "backlog_triage", "item_id": i, "verdict": verdict,
                            "closed": True, "group_cluster": cid, "group_verdict": verdict,
                            "evidence": evidence[:1000], "session_id": session_id,
                            "spawned": [], "auto": True}, path=_ledger_or(extra))
            counts["retired"] += 1
        elif verdict == "fold":
            update_frontmatter(item.path, {"group": umbrella.id},
                               activity=f"folded into umbrella #{umbrella.id} by automod group "
                                        f"triage ({cid}): {evidence}"[:600],
                               add_tags=("grouped",))
            S.append_event({"event": "backlog_triage", "item_id": i, "verdict": FOLDED,
                            "closed": False, "group": umbrella.id, "group_cluster": cid,
                            "group_verdict": verdict, "evidence": evidence[:1000],
                            "session_id": session_id}, path=_ledger_or(extra))
            counts["folded"] += 1
        else:  # keep — still untriaged, on purpose; released from quarantine by the summary
            note_item(i, f"group triage {cid}: distinct from the others; stays in the single-item pool"
                         + (f" — {evidence}" if evidence else ""))
            counts["kept"] += 1
    if umbrella is not None and fold_ids:
        clauses, dropped = cap_new_clauses(umbrella_fields.get("acceptance_clauses") or ())
        dropped = list(umbrella_fields.get("clauses_dropped_text") or []) + dropped
        record_verdict(umbrella, "confirmed", str(umbrella_fields.get("evidence") or ""),
                       check=str(umbrella_fields.get("check") or ""),
                       acceptance=str(umbrella_fields.get("acceptance") or ""),
                       acceptance_clauses=clauses, dropped_clauses=dropped, hold=hold)
        update_frontmatter(umbrella.path, {"members": sorted(fold_ids)}, add_tags=("umbrella",))
        S.append_event({"event": "backlog_triage", "item_id": umbrella.id, "verdict": "confirmed",
                        "held": bool(hold), "clauses_dropped": len(dropped),
                        "clauses_dropped_text": dropped,
                        "surface": str(umbrella_fields.get("surface") or "code"),
                        "check": str(umbrella_fields.get("check") or ""),
                        "evidence": str(umbrella_fields.get("evidence") or "")[:1000],
                        "acceptance": str(umbrella_fields.get("acceptance") or ""),
                        "acceptance_clauses": clauses, "spawned": [], "closed": False,
                        "umbrella": True, "members": sorted(fold_ids), "group_cluster": cid,
                        "session_id": session_id, "verdict_source": (extra or {}).get("verdict_source", "structured")},
                       path=_ledger_or(extra))
    summary = {"event": "backlog_group_triage", "cluster_id": cid,
               "item_ids": sorted(member_ids), "judged": judged, **counts,
               "umbrella_id": umbrella.id if (umbrella is not None and fold_ids) else None,
               "umbrella_missing": umbrella_missing,
               "spawned": list(spawned), "merged": list(merged), "session_id": session_id,
               **(extra or {})}
    S.append_event(summary, path=_ledger_or(extra))
    return {**counts, "umbrella_id": summary["umbrella_id"], "umbrella_missing": umbrella_missing,
            "judged": judged}


def _ledger_or(extra: dict | None) -> Path:
    from scripts.automod import state as S
    return S.LEDGER_PATH


def unfold_umbrella(umbrella_id: int, reason: str, *, ledger: Path | None = None) -> dict:
    """The human escape hatch: release an umbrella's members back to the
    single pool and clear the umbrella's member list. Recorded, because a
    fold was a judgement and undoing it is one too."""
    from scripts.automod import state as S
    reason = " ".join(str(reason or "").split()).strip()
    if not reason:
        raise ValueError("a reason is required")
    ledger = ledger or LEDGER_DEFAULT()
    umbrella = next((i for i in all_items(None) if i.id == int(umbrella_id)), None)
    if umbrella is None:
        raise ValueError(f"#{umbrella_id} not found")
    released: list[int] = []
    for item in all_items(None):
        if item.group == int(umbrella_id):
            update_frontmatter(item.path, {"group": None},
                               activity=f"released from umbrella #{umbrella_id}: {reason}",
                               remove_tags=("grouped",))
            released.append(item.id)
    update_frontmatter(umbrella.path, {"members": []},
                       activity=f"unfolded ({len(released)} member(s) released): {reason}")
    S.append_event({"event": "backlog_group_unfold", "item_id": int(umbrella_id),
                    "released": released, "reason": reason}, path=ledger)
    return {"umbrella_id": int(umbrella_id), "released": released, "reason": reason}


UNFOLDED_TAG = "unfolded"


def unfold_spent_umbrellas(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                           enabled: bool = True) -> list[dict]:
    """Unfold every open umbrella whose one unattended attempt is spent, and
    close it. Returns one `{umbrella_id, released, reason}` per umbrella.

    A spent umbrella parked `draft` + `needs-human` and held its members
    with it: `grouped` is expiry-exempt, so 156 members on 2026-09-14 could
    be neither triaged nor expired, and of 68 umbrellas formed 5 had landed.
    Alan's ruling that day: a spent umbrella unfolds and its members expire.
    So the members drop `group`/`grouped` and are otherwise left alone — a
    self-filed one stays quarantined and closes under the self-spawn expiry;
    none re-forms its cluster, because a group triage already judged it
    (`group_triaged_ids`) — and the umbrella closes `done`, tagged `unfolded`,
    text kept. A landed umbrella (the landed marker) is never touched: that
    one is waiting on a person, not spent. `grouped` stays in
    `EXPIRY_EXEMPT_TAGS`, because a member of a still-live umbrella closed by
    expiry would be misattributed by `close_settled_items` when it lands.
    """
    if not enabled:
        return []
    from scripts.automod import state as S
    history = implement_history(ledger)
    outcomes = implement_outcomes(ledger)
    unfinished = items_with_unfinished_rounds(ledger, history=history)
    out: list[dict] = []
    for item in open_items(boards):
        if not is_umbrella(item):
            continue
        verdict, detail = outcomes.get(item.id, ("", ""))
        if verdict != "spent" or item.id in unfinished:
            continue
        fm, _ = _split_frontmatter(item.path.read_text(encoding="utf-8"))
        if fm.get(LANDED_MARKER) or any(fm.get(m) for m in _LEGACY_LANDED_MARKERS):
            continue
        reason = ("its one unattended implement attempt is spent"
                  + (f" ({detail[:200]})" if detail else "")
                  + "; members released to be triaged or expire on their own")
        res = unfold_umbrella(item.id, reason, ledger=ledger)
        set_status(item.id, "done", f"unfolded: {reason}",
                   add_tags=(UNFOLDED_TAG,), remove_tags=(NEEDS_HUMAN_TAG,))
        S.append_event({"event": "backlog_umbrella_unfolded", "item_id": item.id,
                        "released": res["released"], "reason": reason[:400],
                        "auto": True}, path=ledger)
        out.append({"umbrella_id": item.id, "released": res["released"], "reason": reason})
    return out


def retriage_counts(ledger: Path) -> dict[int, int]:
    """How many times each item has been sent back through triage."""
    counts: dict[int, int] = {}
    for d in _ledger_events(ledger, "backlog_retriage"):
        i = int(d["item_id"])
        counts[i] = counts.get(i, 0) + 1
    return counts


def last_retriage(ledger: Path, item_id: int) -> dict | None:
    """The item's latest `backlog_retriage` row — what its second triage is
    told about the attempt that was refused — or None."""
    found = None
    for d in _ledger_events(ledger, "backlog_retriage"):
        if int(d["item_id"]) == int(item_id):
            found = d
    return found


def _refusal_for_retriage(ledger: Path, item_id: int, round_id: str, detail: str) -> dict:
    """What triage should be told about the attempt that was spent: the
    grader's findings, its last per-clause verdicts, and the clauses it found
    unmet or unsatisfiable on two reviews. Read before the mark is written,
    while these reviews still count."""
    graded = [e for e in review_events_for_item(ledger, item_id) if e.get("ok") and e.get("clauses")]
    last = graded[-1] if graded else {}
    findings = str(last.get("findings") or "")
    if not findings and round_id:
        gate = _last_gate_per_round(ledger).get(round_id) or {}
        findings = str(gate.get("review_findings") or gate.get("review_summary")
                       or gate.get("detail") or "")
    flagged: dict[int, int] = {}
    for ev in graded:
        for c in ev.get("clauses") or []:
            if isinstance(c, dict) and c.get("verdict") in ("unmet", "unsatisfiable"):
                n = int(c.get("clause") or 0)
                if n:
                    flagged[n] = flagged.get(n, 0) + 1
    return {"findings": (findings or detail)[:800],
            "clauses": [{"clause": c.get("clause"), "verdict": c.get("verdict"),
                         "note": str(c.get("note") or "")[:200]}
                        for c in (last.get("clauses") or []) if isinstance(c, dict)],
            "unmet_twice": sorted(n for n, k in flagged.items() if k >= 2)}


def retriage_spent_items(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                         enabled: bool = True) -> list[dict]:
    """Send a spent item back through triage once, with its refusal attached.

    Its one unattended attempt used to end the loop's interest in it: `draft`,
    `needs-human`, and a person sweeping the pile. ~35 items a week went
    there, and of the 67 a human had reopened by 2026-09-14, 42 later landed
    — a second offer lands most of them. The second offer goes through
    TRIAGE rather than straight back to a round, because what failed was
    usually the contract (twelve clauses, one of them unmeetable), and triage
    is the pass that writes contracts. It sees the refused round's findings
    and per-clause verdicts in `<origin>`.

    Eligible: implement outcome `spent`, never re-triaged (`RETRIAGE_CAP`),
    not an umbrella or a member (`unfold_spent_umbrellas` owns those), and not
    mid-round (`items_with_unfinished_rounds`: in flight, landing, observed or
    promoted — a promotion and the landed marker are the settle sweep's or a
    person's, a landed `met` item owing human clauses among them). The item moves to `draft` tagged `re-triage`, drops
    `needs-human` and `review-disagreement`, and a `backlog_retriage` row is
    the mark every history reader starts from. The second spend parks it for
    a human, as before.
    """
    if not enabled:
        return []
    from scripts.automod import state as S
    history = implement_history(ledger)
    outcomes = implement_outcomes(ledger)
    counts = retriage_counts(ledger)
    unfinished = items_with_unfinished_rounds(ledger, history=history)
    last_round = {iid: next((str(r["round_id"]) for r in reversed(rows) if r.get("round_id")), "")
                  for iid, rows in history.items()}
    out: list[dict] = []
    for item in open_items(boards):
        verdict, detail = outcomes.get(item.id, ("", ""))
        if verdict != "spent" or counts.get(item.id, 0) >= RETRIAGE_CAP:
            continue
        if is_umbrella(item) or is_grouped(item):
            continue
        # In flight, landing, observed, or promoted: not over, whatever
        # `spent` says.
        if item.id in unfinished:
            continue
        fm, _ = _split_frontmatter(item.path.read_text(encoding="utf-8"))
        if fm.get(LANDED_MARKER) or any(fm.get(m) for m in _LEGACY_LANDED_MARKERS):
            continue
        rid = last_round.get(item.id, "")
        refusal = _refusal_for_retriage(ledger, item.id, rid, detail)
        why = ("its implement attempt was spent" + (f" in round {rid}" if rid else "")
               + "; sent back through triage once, with the refusal, for a contract a round "
                 "can meet or a retirement")
        tag_item(item.id, add=(RETRIAGE_TAG,), remove=(NEEDS_HUMAN_TAG, "review-disagreement"))
        # The old contract comes off the item: `acceptance_clauses_of` prefers
        # front matter, so a second triage that confirms in prose alone would
        # otherwise hand the next round the contract that was just refused.
        # It stays on the ledger row, and in the item's triage section.
        update_frontmatter(item.path, {"acceptance_clauses": None, "human_clauses": None,
                                       AMENDMENTS_KEY: None})
        if item.status != TRIAGE_POOL_STATUS:
            set_status(item.id, TRIAGE_POOL_STATUS, why)
        note_item(item.id, why)
        S.append_event({"event": "backlog_retriage", "item_id": item.id, "round_id": rid,
                        "outcome_detail": detail[:600],
                        "previous_clauses": list(fm.get("acceptance_clauses") or []),
                        **refusal}, path=ledger)
        out.append({"item_id": item.id, "round_id": rid, "reason": why})
    return out


def summarize(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> dict:
    seen = triaged_ids(ledger)
    counts: dict[str, int] = {}
    for verdict in seen.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    items = open_items(boards)
    total_open = len(items)
    return {
        "boards": list(boards) if boards else "all",
        "open_items": total_open,
        "triaged": len(seen),
        "untriaged": max(0, total_open - sum(1 for i in items if i.id in seen)),
        "verdicts": counts,
        "retired": sum(counts.get(v, 0) for v in RETIRING),
        "confirmed": counts.get("confirmed", 0),
    }


def board_flow(boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
               backlog_dir: Path | None = None, items: list[Item] | None = None,
               now: float | None = None) -> dict:
    """Items created and closed in the last 24 h and 7 d: `{"24h": {created,
    closed, net}, "7d": {...}}`. Board files only, no ledger — cheap enough
    for the scorecard to call on its own.

    Inflow is `created`. Outflow is `completed`, else `updated`, on a closed
    item; both are stamped by the writer that closed it, so a file untouched
    for a week cannot have closed inside the week and is skipped on mtime
    without being parsed.

    Each stamp is read in its writer's clock. `created` comes from the MCP
    store or the Mission Control router, in naive local time; `completed` only
    ever from this module's closers, in naive UTC; a closed item with no
    `completed` was closed by one of the local-time writers, so its `updated`
    is local too. Until 2026-09-14 all three were read as UTC, which on this
    box put every creation seven hours before every close — the 24-hour net
    compared two different days.
    """
    now = time.time() if now is None else now
    everything = items if items is not None else all_items(boards, backlog_dir=backlog_dir)
    widest = now - 7 * 86400
    created_ts = [t for t in (_iso_ts(i.created, naive_local=True) for i in everything)
                  if t is not None]
    closed_ts: list[float] = []
    for i in everything:
        if i.status in OPEN_STATUSES:
            continue
        try:
            if i.path.stat().st_mtime < widest:
                continue
            fm, _ = _split_frontmatter(i.path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if fm.get("completed"):
            t = _iso_ts(fm.get("completed"))
        else:
            t = _iso_ts(fm.get("updated"), naive_local=True)
        if t is not None:
            closed_ts.append(t)

    def _window(seconds: float) -> dict:
        since = now - seconds
        created = sum(1 for t in created_ts if since <= t <= now)
        closed = sum(1 for t in closed_ts if since <= t <= now)
        return {"created": created, "closed": closed, "net": created - closed}

    return {"24h": _window(86400), "7d": _window(7 * 86400)}


def board_health(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                 backlog_dir: Path | None = None, now: float | None = None,
                 floor: int = IMPLEMENT_POOL_FLOOR) -> dict:
    """What the board is made of and which way it is moving. One definition
    for three readers: the dashboard's backlog panel, the scorecard's net-flow
    row and the board steward's prompt.

    The raw status count hides the thing that matters. On 2026-09-13 the board
    read 480 `draft`, of which 139 were members folded under an umbrella
    already in `up_next`, 93 quarantined self-spawns and 32 waiting on a
    person — 200 were actually triageable. And `up_next` 88 was 42 ready.

    `draft` is a partition, each item counted once, by precedence: grouped >
    needs-human > held > triaged > quarantined > pool. `held` is a confirmation
    waiting for room in the implement pool (see `held_confirmations`).

    Cost, so it is not re-derived: it walks every item file and asks the ledger
    many filtered questions, all answered from one decode since #1204
    (`scripts.automod.state.ledger_rows`). 5.63 s standalone measured
    2026-09-17 over 1,141 item files before that work; 0.40 s after. Never call
    it per item — the walk is per board, and the dashboard reaches it once a
    minute at `_SCORECARD_TTL_S`.
    """
    now = time.time() if now is None else now
    everything = all_items(boards, backlog_dir=backlog_dir)
    items = [i for i in everything if i.status in OPEN_STATUSES]
    seen = triaged_ids(ledger)
    released = released_ids(ledger)
    live = live_blockers(ledger, boards, items=items, backlog_dir=backlog_dir)
    held = held_confirmations(ledger)
    outcomes = implement_outcomes(ledger)
    attempted = {int(d["item_id"]) for d in _ledger_events(ledger, "backlog_implement")}

    open_counts: dict[str, int] = {}
    for i in items:
        open_counts[i.status] = open_counts.get(i.status, 0) + 1

    draft = {"pool": 0, "quarantined": 0, "grouped": 0, "needs_human": 0, "held": 0,
             "parked": 0, "triaged": 0}
    for i in items:
        if i.status != TRIAGE_POOL_STATUS:
            continue
        if is_grouped(i):
            draft["grouped"] += 1
        elif NEEDS_HUMAN_TAG in i.tags:
            draft["needs_human"] += 1
        elif i.id in held:
            draft["held"] += 1
        elif is_parked(i):
            draft["parked"] += 1
        elif i.id in seen:
            draft["triaged"] += 1
        elif is_quarantined(i, released=released, live=live):
            draft["quarantined"] += 1
        else:
            draft["pool"] += 1
    draft["total"] = sum(draft.values())

    # The sweep's coverage: how much of the open board it has read and how
    # it ranked what lives. `unswept` should drain to 0 while a sweep is on.
    sweepable = sweep_pool(ledger, boards, items=items)
    worth = {w: 0 for w in WORTH_LEVELS}
    for i in items:
        if i.worth in worth:
            worth[i.worth] += 1
    sweep = {"unswept": len(sweepable), "swept": sum(1 for i in items if is_swept(i)),
             "parked": sum(1 for i in items if is_parked(i)), "worth": worth}

    up = [i for i in items if i.status == IMPLEMENT_POOL_STATUS]
    ready = ready_confirmed(ledger, boards, items=up, outcomes=outcomes)
    umbrellas = sum(1 for i in up if is_umbrella(i))
    up_next = {"total": len(up), "umbrellas": umbrellas, "singles": len(up) - umbrellas,
               "never_attempted": sum(1 for i in up if i.id not in attempted),
               "ready": len(ready), "unready": len(up) - len(ready)}

    # #1210: `draft.needs_human` was the only figure on the board speaking for
    # "a person owes this", and it could only see drafts. A landed-met item now
    # closes carrying the tag, so the closed side needs its own count — without
    # it the steward sees the needs-human number fall while the work a person
    # owes has not moved at all.
    closed_needs_human = sum(1 for i in everything
                             if i.status in CLOSED_STATUSES and NEEDS_HUMAN_TAG in i.tags)

    pool = implement_pool_bound(ledger, floor=floor, now=now)
    return {
        "open": open_counts,
        "draft": draft,
        "closed_needs_human": closed_needs_human,
        "up_next": up_next,
        "flow": board_flow(boards, items=everything, now=now),
        "self_spawned_open": sum(1 for i in items if is_loop_spawned(i)),
        # Open blockers whose blocked item is still open, and how many of
        # those no triage has judged yet — the second should drain to 0.
        "live_blockers": {"open": len(live),
                          "untriaged": sum(1 for b in live if b not in seen)},
        "landed_items_7d": pool["landed_items_7d"],
        "implement_pool": {"ready": len(ready), "bound": pool["bound"], "floor": pool["floor"]},
        "sweep": sweep,
    }


# ── The sweep: every open item read once, retired or ranked ─────────────────
#
# Single triage judges one item per 90-turn session and confirms 73% of what
# it reads; the implement loop lands about ten items a day. On 2026-09-15 that
# left 156 confirmed items queued, 82 quarantined drafts nothing had ever read
# (27 of them due to expire unread within a day), 158 members folded under
# umbrellas that land one time in five, and a board of 560. The sweep is the
# gear change: a batch of `sweep_batch` items per turn, quarantine lifted,
# nothing filed and nothing appended, each item either retired (`stale`,
# `already_done`, `duplicate_of`) or kept with a rank. The rank is what the
# rest of the loop then orders on (`rank_key`), and `parked` (a `low` draft)
# is the exit that replaced expiry: read by someone, still open, out of the
# pools, never closed for being old.

SWEEP_VERDICTS = ("keep", "duplicate_of") + tuple(sorted(RETIRING))
SWEEP_SCHEMA: dict = {
    "type": "object",
    "title": "backlog_sweep_verdict",
    "properties": {
        "items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "item_id": {"type": "integer"},
                "verdict": {"type": "string", "enum": list(SWEEP_VERDICTS)},
                "duplicate_of": {"type": "integer",
                                 "description": "The surviving open item's id; 0 unless verdict is duplicate_of."},
                "worth": {"type": "string", "enum": list(WORTH_LEVELS)},
                "size": {"type": "string", "enum": list(SIZE_LEVELS)},
                "evidence": {"type": "string",
                             "description": "One sentence: the path, commit or check that decides it."},
            },
            "required": ["item_id", "verdict", "duplicate_of", "worth", "size", "evidence"],
            "additionalProperties": False,
        }, "description": "One entry per item in the batch, every item listed."},
    },
    "required": ["items"],
    "additionalProperties": False,
}
SWEEP_POOL_STATUSES = (TRIAGE_POOL_STATUS, IMPLEMENT_POOL_STATUS)


def sweep_enabled() -> bool:
    """`workers.sources.autotriage.sweep`, default off. Lazy and fail-closed:
    with no config there is no sweep, and nothing yields to one."""
    try:
        from app.config import CONFIG
        cfg = (((CONFIG or {}).get("workers") or {}).get("sources") or {}).get("autotriage") or {}
        return bool(cfg.get("sweep", False))
    except Exception:  # noqa: BLE001
        return False


def sweep_abandoned_ids(ledger: Path) -> set[int]:
    """Items in a batch that ran out of budget twice. Left unswept for the
    ordinary passes rather than retried forever."""
    out: set[int] = set()
    for d in _ledger_events(ledger, "backlog_sweep", require_item=False):
        if d.get("verdict") == "abandoned":
            out |= {int(i) for i in (d.get("item_ids") or [])}
    return out


def sweep_pool(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
               items: list[Item] | None = None) -> list[Item]:
    """Open items the sweep has not read, never-judged first, then by the
    human's priority, then oldest. A `high` item is not the sweep's to read:
    single triage takes it next, whatever else is unread.

    Quarantine does not apply — reading a self-filed item once is the whole
    point. `draft` and `up_next` both: a confirmed item is ranked so the
    implement order can prefer it, and can still be retired if its premise
    has gone. Out: grouped members (their umbrella is the unit), umbrellas
    (confirmed work with a contract), `needs-human`, `expired`, anything
    already swept or parked, and a batch that ran out of budget twice.
    """
    done = swept_ids(ledger) | sweep_abandoned_ids(ledger)
    seen = triaged_ids(ledger)
    claimed = triage_claimed_ids()
    pool = [i for i in (items if items is not None else open_items(boards))
            if i.status in SWEEP_POOL_STATUSES and i.id not in done
            and i.id not in claimed
            and not is_swept(i) and not is_parked(i) and not is_high(i)
            and not is_grouped(i) and not is_umbrella(i)
            and NEEDS_HUMAN_TAG not in i.tags and EXPIRED_TAG not in i.tags]
    return sorted(pool, key=lambda i: (i.id in seen, priority_key(i), i.created or "9999", i.id))


def select_sweep_batch(ledger: Path, n: int,
                       boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> list[Item]:
    return sweep_pool(ledger, boards)[:max(1, int(n))]


def sweep_pending(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> int:
    """How many items a sweep still has to read; 0 when the sweep is off.
    Autocode yields to a positive answer (`yield_to_sweep`)."""
    if not sweep_enabled():
        return 0
    return len(sweep_pool(ledger, boards))


def sweep_batch_id(item_ids) -> str:
    import hashlib
    ids = sorted(int(i) for i in item_ids)
    return "sw-" + hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:10]


def sweep_incomplete_attempts(ledger: Path, batch_id: str) -> int:
    return sum(1 for d in _ledger_events(ledger, "backlog_sweep", require_item=False)
               if d.get("batch_id") == batch_id and d.get("verdict") == INCOMPLETE)


def record_sweep_verdicts(batch_id: str, members: list[Item], verdicts: dict[int, dict], *,
                          session_id: str = "", extra: dict | None = None,
                          ledger: Path | None = None) -> dict:
    """Write a sweep's verdicts onto the items and the ledger.

    Retiring verdicts go through `record_verdict` and get a `backlog_triage`
    row, so the status pipeline, the scorecard and `triaged_ids` see an
    ordinary close. `duplicate_of` may point at any OPEN item, in the batch
    or not; a target that is closed, missing, or itself retired here becomes
    `keep`. A `keep` writes `worth`/`size` and the `swept` tag; a `low`
    draft is `parked` too. Only judged ids are recorded — an item the turn
    did not list stays unswept and is offered again.
    """
    from scripts.automod import state as S
    ledger = ledger or LEDGER_DEFAULT()
    by_id = {m.id: m for m in members}
    member_ids = set(by_id)
    verdicts = {int(i): dict(v) for i, v in verdicts.items() if int(i) in member_ids}
    # A target outside the batch counts as a member for chain resolution
    # when it is open on disk; anything else `_resolve_duplicates` turns
    # into a keep.
    external = set()
    for i, v in verdicts.items():
        if v.get("verdict") != "duplicate_of":
            continue
        t = int(v.get("duplicate_of") or 0)
        if t and t not in member_ids:
            target = item_by_id(t)
            if target is not None and target.status in OPEN_STATUSES and not is_parked(target):
                external.add(t)
    verdicts = _resolve_duplicates(verdicts, member_ids | external)
    counts = {"retired": 0, "duplicates": 0, "kept": 0, "parked": 0}
    judged: dict[str, str] = {}
    ranked: dict[str, str] = {}
    for i, v in verdicts.items():
        item = by_id[i]
        verdict = str(v.get("verdict") or "keep")
        evidence = " ".join(str(v.get("evidence") or "").split())[:600]
        worth = _level(v.get("worth"), WORTH_LEVELS)
        size = _level(v.get("size"), SIZE_LEVELS)
        judged[str(i)] = verdict
        if verdict == "duplicate_of":
            t = int(v["duplicate_of"])
            record_verdict(item, "stale", f"duplicate of #{t}: {evidence}", close=True)
            update_frontmatter(item.path, {"duplicate_of": t})
            S.append_event({"event": "backlog_triage", "item_id": i, "verdict": "stale",
                            "closed": True, "duplicate_of": t, "sweep_batch": batch_id,
                            "evidence": evidence[:1000], "session_id": session_id,
                            "spawned": [], "auto": True}, path=ledger)
            counts["duplicates"] += 1
        elif verdict in RETIRING:
            record_verdict(item, verdict, evidence, close=True)
            S.append_event({"event": "backlog_triage", "item_id": i, "verdict": verdict,
                            "closed": True, "sweep_batch": batch_id,
                            "evidence": evidence[:1000], "session_id": session_id,
                            "spawned": [], "auto": True}, path=ledger)
            counts["retired"] += 1
        else:
            park = worth == "low" and item.status == TRIAGE_POOL_STATUS
            tags = (SWEPT_TAG,) + ((PARKED_TAG,) if park else ())
            update_frontmatter(item.path, {"worth": worth or None, "size": size or None},
                               activity=(f"swept ({batch_id}): worth={worth or '?'} "
                                         f"size={size or '?'}" + (" — parked: out of every pool "
                                         "until a person removes the tag" if park else "")
                                         + (f" — {evidence}" if evidence else ""))[:700],
                               add_tags=tags)
            ranked[str(i)] = f"{worth or '?'}/{size or '?'}"
            counts["kept"] += 1
            counts["parked"] += int(park)
    summary = {"event": "backlog_sweep", "batch_id": batch_id,
               "item_ids": sorted(member_ids), "judged": judged, "ranked": ranked, **counts,
               "session_id": session_id, **(extra or {})}
    S.append_event(summary, path=ledger)
    return {**counts, "judged": judged, "ranked": ranked}


def unfold_oversized_umbrellas(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                               min_clauses: int = 8, dry_run: bool = False) -> list[dict]:
    """Unfold every never-attempted open umbrella whose contract carries at
    least `min_clauses` clauses, and close it `done` tagged `unfolded`.

    56 umbrellas sat in `up_next` on 2026-09-15, every one with 8–12
    clauses from the days before `group_max_items` fell to 4, and that shape
    landed 5 of 27 rounds against 52 of 151 for singles. Their 158 members
    were folded out of every pool. A member goes back to `draft` untriaged
    (its `folded` row is not a verdict), where the sweep reads and ranks it;
    it does not re-cluster (`group_triaged_ids`). An umbrella a round has
    already attempted, or that is landing or landed, is not touched — that
    is `unfold_spent_umbrellas`' or a person's call.
    """
    from scripts.automod import state as S
    history = implement_history(ledger)
    unfinished = items_with_unfinished_rounds(ledger, history=history)
    confirmed = confirmed_verdicts(ledger)
    out: list[dict] = []
    for item in open_items(boards):
        if not is_umbrella(item) or item.id in history or item.id in unfinished:
            continue
        fm, _ = _split_frontmatter(item.path.read_text(encoding="utf-8"))
        if fm.get(LANDED_MARKER) or any(fm.get(m) for m in _LEGACY_LANDED_MARKERS):
            continue
        n = len(acceptance_clauses_of(confirmed.get(item.id), fm))
        if n < int(min_clauses):
            continue
        reason = (f"oversized: {n} clauses against a cap of {MAX_CLAUSES} and never attempted; "
                  f"members released to be swept and ranked on their own")
        row = {"umbrella_id": item.id, "clauses": n, "members": list(item.members), "reason": reason}
        if dry_run:
            out.append(row)
            continue
        res = unfold_umbrella(item.id, reason, ledger=ledger)
        set_status(item.id, "done", f"unfolded: {reason}",
                   add_tags=(UNFOLDED_TAG,), remove_tags=(NEEDS_HUMAN_TAG, HELD_TAG))
        S.append_event({"event": "backlog_umbrella_unfolded", "item_id": item.id,
                        "released": res["released"], "reason": reason[:400],
                        "oversized": True, "clauses": n, "auto": True}, path=ledger)
        out.append({**row, "released": res["released"]})
    return out
