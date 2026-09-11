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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.backlog_status import (
    CLOSED_ALIASES,
    OPEN_STATUSES,
    PIPELINE_STATUSES,
    canonical_status,
    is_off_vocabulary,
)
from app.backlog_tags import normalize_tags

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

# How old a self-filed item must be before triage may judge it. Long enough
# that 'is this still true?' is a real question about a claim nobody acted
# on, rather than a re-run of the check that produced it.
SPAWN_TRIAGE_MIN_AGE_DAYS = 30

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
# `met` closes the item once the promotion settles; the other two leave it
# open and say why. Built from this tuple, not restated, for the reason the
# triage schema is: one list, or a new value lands in the grammar and not in
# the validator.
ACCEPTANCE_OUTCOMES = ("met", "not_met", "deferred", "unnecessary")
# Per clause. No `unnecessary`: that is a verdict on the item, not on a clause.
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
                                       "the item should close without a landing. When "
                                       "clause_outcomes is non-empty this is derived from it.")},
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

    if acceptance != "unnecessary" and clauses:
        outcomes = {c["outcome"] for c in clauses}
        if "not_met" in outcomes:
            acceptance = "not_met"
        elif outcomes == {"met"}:
            acceptance = "met"
        else:
            acceptance = "deferred"
    if acceptance == "deferred" and not deferred_to:
        acceptance = "not_met"

    return {"landed": bool(structured.get("landed")), "acceptance": acceptance,
            "clause_outcomes": clauses,
            "deferred_to": deferred_to,
            "summary": " ".join(str(structured.get("summary") or "").split())[:400],
            "spawned": _ints(structured.get("spawned"))}


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
                                               "test can pin, in order. The implementer reports "
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

MAX_CLAUSES = 12
CLAUSE_MAX_CHARS = 600


def clean_clauses(values) -> list[str]:
    """Clauses as a bounded list of non-placeholder strings."""
    out: list[str] = []
    for v in (values or []):
        s = acceptance_text(v)
        if s and s not in out:
            out.append(s[:CLAUSE_MAX_CHARS])
        if len(out) >= MAX_CLAUSES:
            break
    return out


_CLAUSE_LINE = re.compile(r"^\s*(\d{1,2})[.)]\s+(.*\S)\s*$")


def split_clause_lines(text: str) -> list[str]:
    """Numbered lines (`1. …`, `2) …`) into clauses; unnumbered prose is one
    clause. A `-` or `none` placeholder is no clause at all."""
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    numbered = [m.group(2) for m in (_CLAUSE_LINE.match(ln) for ln in lines) if m]
    if numbered:
        return clean_clauses(numbered)
    return clean_clauses([" ".join(str(text or "").split())])


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


def pending_amendments(frontmatter: dict | None) -> list[dict]:
    return [dict(a) for a in ((frontmatter or {}).get(AMENDMENTS_KEY) or [])
            if isinstance(a, dict) and a.get("state") == "pending"]


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
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.safe_load(parts[1]) or {}
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
        status=str(fm.get("status", "draft")), priority=str(fm.get("priority", "medium")),
        created=str(fm.get("created", "")), body=body,
        board=str(fm.get("board", "") or ""),
        tags=normalize_tags(fm.get("tags")),
        parent=_int_or_none(fm.get("parent")),
        group=_int_or_none(fm.get("group")),
        members=_ints(fm.get("members")),
    )


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
    tags = [str(t) for t in (fm.get("tags") or [])]
    new_tags = [t for t in tags if t not in remove_tags] + [t for t in add_tags if t not in tags]
    if new_tags != tags:
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


def all_items(boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> list[Item]:
    """Every item on `boards`, whatever its status — including a status
    outside `PIPELINE_STATUSES`, which is the one thing `open_items` cannot
    show you and the reason the rescue below needs its own walk."""
    wanted = {b.lower() for b in boards} if boards else None
    out = []
    for path in sorted(BACKLOG_DIR.glob("*.md")):
        item = load_item(path)
        if not item:
            continue
        if wanted is not None and item.board.lower() not in wanted:
            continue
        out.append(item)
    return out


def open_items(boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> list[Item]:
    """Open items, restricted to `boards` unless it is None."""
    return [i for i in all_items(boards) if i.status in OPEN_STATUSES]


def _ledger_events(ledger: Path, event: str, *, require_item: bool = True) -> list[dict]:
    """Rows of one event type. `require_item=False` for the event types that
    are about a round rather than an item — `gate` carries a `round_id` and no
    `item_id`, and the default filter drops it silently."""
    out: list[dict] = []
    if not ledger.exists():
        return out
    try:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") != event:
                continue
            if require_item and d.get("item_id") is None:
                continue
            out.append(d)
    except OSError:
        pass
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


def triaged_ids(ledger: Path) -> dict[int, str]:
    """{item_id: verdict} for items with a TERMINAL verdict.

    `incomplete` is deliberately not one: an item whose triage ran out of
    budget is not triaged, it is waiting for a bigger budget.
    """
    seen: dict[int, str] = {}
    for d in _ledger_events(ledger, "backlog_triage"):
        verdict = d.get("verdict", "")
        if verdict in VERDICTS:
            seen[int(d["item_id"])] = verdict
    return seen


def incomplete_counts(ledger: Path) -> dict[int, int]:
    """How many times each item's triage has run out of budget."""
    counts: dict[int, int] = {}
    for d in _ledger_events(ledger, "backlog_triage"):
        if d.get("verdict") == INCOMPLETE:
            i = int(d["item_id"])
            counts[i] = counts.get(i, 0) + 1
    return counts


def confirmed_verdicts(ledger: Path) -> dict[int, dict]:
    """{item_id: latest `confirmed` triage event}. The event carries the
    ACCEPTANCE the implementer is held to."""
    out: dict[int, dict] = {}
    for d in _ledger_events(ledger, "backlog_triage"):
        if d.get("verdict") == "confirmed":
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
    oldest first."""
    rids = {str(d.get("round_id") or "") for d in _ledger_events(ledger, "backlog_implement")
            if int(d["item_id"]) == int(item_id) and d.get("round_id")}
    rids.discard("")
    rows = [d for d in _ledger_events(ledger, "review", require_item=False)
            if str(d.get("round_id") or "") in rids]
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
    """
    latest: dict[int, dict] = {}
    attempts: dict[int, int] = {}
    for d in _ledger_events(ledger, "backlog_implement"):
        iid = int(d["item_id"])
        latest[iid] = d
        if str(d.get("phase") or "") in ("finished", "infra_failed"):
            attempts[iid] = attempts.get(iid, 0) + 1

    blocked = externally_blocked_rounds(ledger)
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
                and n <= 1 + INCOMPLETE_RETRY_CAP):
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
        if rid and rid in blocked and n <= EXTERNAL_RETRY_CAP:
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
    """
    settled = {str(d.get("commit") or ""): d
               for d in _ledger_events(ledger, "settled", require_item=False)}
    settled.pop("", None)
    reverted = {str(d.get("commit") or "") for d in
                _ledger_events(ledger, "rollback_succeeded", require_item=False)}
    promoted = {str(d.get("round_id") or ""): d
                for d in _ledger_events(ledger, "promoted", require_item=False)
                if str(d.get("commit") or "") in settled and d.get("round_id")}
    out: list[dict] = []
    for d in _ledger_events(ledger, "backlog_implement"):
        if d.get("phase") != "finished":
            continue
        rid = str(d.get("round_id") or "")
        vault = [str(c) for c in (d.get("vault_commits") or []) if c]
        if rid in promoted and promoted[rid]["commit"] not in reverted:
            p = promoted[rid]
            out.append({"item_id": int(d["item_id"]), "round_id": rid, "commit": p["commit"],
                        "settled_at": settled[p["commit"]].get("created_at"),
                        "outcome": d.get("outcome"), "vault": False})
        elif vault and not rid:
            out.append({"item_id": int(d["item_id"]), "round_id": "", "commit": vault[-1],
                        "settled_at": d.get("created_at"),
                        "outcome": d.get("outcome"), "vault": True})
    return out


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
        acc = outcome.get("acceptance")
        human = human_clauses_of(None, fm)
        tags: tuple[str, ...] = ()
        if acc == "met" and human:
            # The loop's half is done; the item is not. Left open, tagged,
            # and the note names what a person still owes it.
            close, tags = False, (NEEDS_HUMAN_TAG,)
            why = ("the round reported every acceptance clause met; still waiting on a person for: "
                   + "; ".join(human))
        elif acc == "met":
            close = True
            why = "the round reported the acceptance check met" + (
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
    tags = [str(t) for t in (fm.get("tags") or [])]
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
    verdicts = triaged_ids(ledger)
    outcomes = implement_outcomes(ledger)
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
        elif landed and not partial:
            out[iid] = ("in_progress", "landed, awaiting the acceptance check or a human close")
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
            else:
                out[iid] = ("up_next", f"offered again — {verdict}: {detail[:120]}")
        elif iid in confirmed and not is_human_only(confirmed[iid].get("acceptance")):
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


def group_triaged_ids(ledger: Path) -> set[int]:
    """Every id a group triage judged, whatever the verdict."""
    out: set[int] = set()
    for d in _ledger_events(ledger, "backlog_group_triage", require_item=False):
        judged = d.get("judged") or {}
        if isinstance(judged, dict):
            for k in judged:
                try:
                    out.add(int(k))
                except (TypeError, ValueError):
                    continue
    return out


def is_self_spawned(item: Item) -> bool:
    """Did this loop write this item? `backlog_write_task` tags what it files."""
    return any(t in SPAWN_TAGS for t in item.tags)


# A self-filed item that nothing picked up. Closed, not deleted: the file
# stays on disk with the tag, and a human setting its status back to `draft`
# reopens it (the reconciler strips the tag and releases it into the pool).
EXPIRED_TAG = "expired"
# Never expired: a member's fate is its umbrella's, an umbrella is confirmed
# work, a `needs-human` item is waiting on a decision, and an expired item
# has already been judged once.
EXPIRY_EXEMPT_TAGS = frozenset({"grouped", "umbrella", NEEDS_HUMAN_TAG, EXPIRED_TAG})


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


def released_ids(ledger: Path) -> set[int]:
    """Self-filed items the pass may triage after all."""
    return expired_ids(ledger) | group_kept_ids(ledger)


def is_quarantined(item: Item, *, released: frozenset[int] | set[int] = frozenset()) -> bool:
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
    """
    return is_self_spawned(item) and int(item.id) not in released


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
    # `draft` only: that is where an item waits to be judged. `up_next` is the
    # implement pool and `in_progress` is a round in flight; an untriaged item
    # in either is a dead state the reconciler moves back here.
    untriaged = [i for i in open_items(boards)
                 if i.id not in seen and i.status == TRIAGE_POOL_STATUS
                 and not is_grouped(i)]
    fresh = [i for i in untriaged if not is_quarantined(i, released=released)]
    return fresh, len(untriaged) - len(fresh)


def expire_stale_spawns(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                        enabled: bool = True,
                        max_age_days: int = SPAWN_TRIAGE_MIN_AGE_DAYS) -> list[dict]:
    """Close self-filed drafts that nothing picked up in `max_age_days`.

    The hard bound on the board. Everything that could have taken the item
    has had a month: clustering, group triage, a human. Closing it is the
    honest record that nothing did — the text stays, tagged, and a hand
    reopen brings it back (see `clear_expired_on_reopen`). Human-authored
    items are never touched: the tag test is `is_self_spawned`, not `draft`.
    """
    if not enabled:
        return []
    from scripts.automod import state as S
    seen = triaged_ids(ledger)
    judged = set(seen) | expired_ids(ledger) | group_kept_ids(ledger)
    judged |= {int(d["item_id"]) for d in _ledger_events(ledger, "backlog_implement")}
    out: list[dict] = []
    for item in open_items(boards):
        if not is_self_spawned(item) or item.status != TRIAGE_POOL_STATUS:
            continue
        if item.id in judged or (set(item.tags) & EXPIRY_EXEMPT_TAGS):
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
        spawned_by = next((t for t in item.tags if t in SPAWN_TAGS), "")
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
    """
    candidates, _held = triage_pool(ledger, boards)
    if not candidates:
        return None
    return sorted(candidates, key=lambda i: (i.created or "9999", i.id))[0]


def select_confirmed(ledger: Path,
                     boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> tuple[Item, dict] | None:
    """Oldest still-open `confirmed` item that no implementation turn has run for.

    Returns (item, triage_event) — the event, not just the id, because the
    ACCEPTANCE recorded at triage is the contract the implementer is held to.
    An item confirmed with no acceptance check is not ready to implement
    unattended; it is skipped here rather than guessed at.
    """
    confirmed = confirmed_verdicts(ledger)
    outcomes = implement_outcomes(ledger)
    done = {iid for iid, (verdict, _) in outcomes.items() if verdict == "spent"}
    ready = []
    for item in open_items(boards):
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
    near = last_review_all_met(ledger)
    return sorted(ready, key=lambda pair: (pair[0].id not in near,
                                           pair[0].id in outcomes,
                                           pair[0].created or "9999", pair[0].id))[0]


def last_review_all_met(ledger: Path) -> set[int]:
    """Items whose most recent graded review found every clause `met`.

    The refusal, if any, was on a seam or a test-honesty finding — one
    change away from a pass. Read off the review events by item, newest
    graded row per item.
    """
    latest: dict[int, dict] = {}
    for d in _ledger_events(ledger, "review"):
        if d.get("ok") and d.get("clauses"):
            latest[int(d["item_id"])] = d
    return {iid for iid, d in latest.items()
            if all(c.get("verdict") == "met" for c in d["clauses"])}


def tag_item(item_id: int, *, add: tuple[str, ...] = (), remove: tuple[str, ...] = ()) -> bool:
    """Add or remove tags on an open item without moving its status."""
    for item in open_items(None):
        if item.id == int(item_id):
            fm, body = _split_frontmatter(item.path.read_text(encoding="utf-8"))
            tags = [str(t) for t in (fm.get("tags") or [])]
            new = [t for t in tags if t not in remove] + [t for t in add if t not in tags]
            if new == tags:
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
                   human_clauses: list[str] | tuple[str, ...] = ()) -> Path:
    """Append the verdict to the item's activity log, optionally closing it.

    Always writes the evidence, never just the conclusion. An item closed as
    stale with no stated reason is indistinguishable from one closed by
    mistake, and the whole value of this pass is that a human can audit it
    later.
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
        fm["autotriage_retired"] = verdict
    elif verdict == "confirmed" and fm.get("status") != "done":
        # Into the implement pool. Before 2026-09-09 a confirmed item stayed
        # wherever it was, and #353 landed while still `draft`.
        fm["status"] = IMPLEMENT_POOL_STATUS
    clauses = clean_clauses(acceptance_clauses)
    if verdict == "confirmed" and clauses:
        # On the item, not only in the ledger: the review rung reads the
        # contract from disk, and a human editing the clauses here is editing
        # what the grader holds the next round to.
        fm["acceptance_clauses"] = clauses
    human = clean_clauses(human_clauses)
    if verdict == "confirmed" and human:
        # What a person must do before this is done. Kept apart from the
        # clauses so no round is asked to fake an audit (#578's clause 5).
        fm["human_clauses"] = human

    section = (f"\n\n## Automod triage — {stamp[:10]}\n\n"
               f"**Verdict:** {verdict}\n\n{evidence.strip()}\n")
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

    item.path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body.rstrip()}{section}",
        encoding="utf-8")
    return item.path


def select_cluster(ledger: Path, clusters: dict, *, min_size: int = 3, max_size: int = 8,
                   boards: tuple[str, ...] | None = DEFAULT_BOARDS
                   ) -> tuple[dict, list[Item]] | None:
    """The cluster a group triage should take: re-validated against disk.

    `clusters.json` is a night old by the time it is read, so every member
    is checked to still be an untriaged, ungrouped open draft; ids a group
    triage already judged are dropped (a cluster of `keep`s is never
    retaken); a cluster whose last group event was not `incomplete` is
    skipped. The survivors are trimmed to `max_size` keeping the `duplicates`
    pairs together, then oldest first. Largest surviving cluster wins.
    """
    seen = triaged_ids(ledger)
    judged = group_triaged_ids(ledger)
    by_id = {i.id: i for i in open_items(boards)
             if i.status == TRIAGE_POOL_STATUS and i.id not in seen
             and not is_grouped(i) and not is_umbrella(i)
             and NEEDS_HUMAN_TAG not in i.tags and EXPIRED_TAG not in i.tags}
    last_group: dict[str, str] = {}
    for d in _ledger_events(ledger, "backlog_group_triage", require_item=False):
        last_group[str(d.get("cluster_id") or "")] = str(d.get("verdict") or d.get("outcome") or "done")
    best: tuple[dict, list[Item]] | None = None
    for c in (clusters or {}).get("clusters") or []:
        cid = str(c.get("id") or "")
        if cid in last_group and last_group[cid] != INCOMPLETE:
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
                      key=lambda i: (by_id[i].created or "9999", i))
        chosen = (dup_ids + rest)[:int(max_size)]
        members = [by_id[i] for i in sorted(chosen)]
        if best is None or len(members) > len(best[1]):
            best = (c, members)
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
                         extra: dict | None = None) -> dict:
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
        clauses = clean_clauses(umbrella_fields.get("acceptance_clauses") or ())
        record_verdict(umbrella, "confirmed", str(umbrella_fields.get("evidence") or ""),
                       check=str(umbrella_fields.get("check") or ""),
                       acceptance=str(umbrella_fields.get("acceptance") or ""),
                       acceptance_clauses=clauses)
        update_frontmatter(umbrella.path, {"members": sorted(fold_ids)}, add_tags=("umbrella",))
        S.append_event({"event": "backlog_triage", "item_id": umbrella.id, "verdict": "confirmed",
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
