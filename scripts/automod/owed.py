"""What an item still owes after the loop is done with it — settled by Lloyd.

Until 2026-09-27 this was the `needs-human` tag. A landing that met every
clause but left a post-landing check, a path the loop may not write, or a spent
attempt closed (or parked) carrying the tag, and nothing ever came back for it:
257 items piled up. A hand sweep that day measured every one — 95 were already
done, 30 moot, 20 waiting on a date, 68 policy calls, 63 leftover work, and 10
needed Alan's hands (sudo, secrets, deletions) — and Alan ruled that nothing
parks on him any more: "lloyd can approve his own choices now."

So the tag is gone as a destination. Each thing owed is an entry in the item's
`owed` list, and `workers/sources/owed_check.py` settles it in a visible
session: evidence (settled), a date (recheck), a ruling Lloyd makes under the
delegation, a follow-up item, another implement attempt, or a close. The one
class no software can act on — something only Alan's hands can do on the host —
moves to `owed_outside`, which Mission Control lists; it is never a tag and
never blocks the board.

Every writer here goes through `backlog.update_frontmatter` (or the item
writers beside it), so the unparsed-YAML guard and the activity log hold.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OWED_KEY = "owed"
SETTLED_KEY = "owed_settled"
OUTSIDE_KEY = "owed_outside"

# Where an owed entry came from. Only for the reader and the prompt: every
# kind is settled the same way.
KINDS = ("check", "path", "decide")

# What the owed-check session may answer for one entry.
OUTCOMES = ("settled", "recheck", "ruling", "work", "reopen", "close", "outside")

# A recheck is never further out than this: a date past it is clamped, so an
# entry cannot be parked for good by a model writing 2027.
MAX_RECHECK_DAYS = 30

# An entry rechecked this many times is ruled on at the next pass, not
# rescheduled again: the session is told so, and `recheck` is refused.
MAX_RECHECKS = 4


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp() -> str:
    from app.backlog_move import now_stamp
    return now_stamp()


def _text(value: Any, limit: int = 600) -> str:
    return " ".join(str(value or "").split())[:limit]


# The bound on a settled record's `evidence` and `ruling` (#1955). It was a
# literal 500 at both call sites, cut with a bare slice: measured 2026-10-01,
# 494 of 576 `owed_settled` fields on the board were exactly 500 chars, none
# marked, and #1644's 951-char ruling lost its clause (c), the reopen bound.
# 1200 holds the longest ruling observed with room; the cap stays as a guard
# against a runaway answer. `workers.sources.owed_check.parse_answer` must not
# cut below it, or the effective bound is silently the lower one.
SETTLED_TEXT_LIMIT = 1200
TRUNCATION_MARKER = " … [truncated]"
_SENTENCE_END = re.compile(r"[.!?](?=\s)")


def _bounded(value: Any, limit: int = SETTLED_TEXT_LIMIT) -> tuple[str, bool]:
    """`value` flattened to one line and, when it exceeds `limit`, cut at the
    last sentence end (else the last whitespace) with `TRUNCATION_MARKER`
    appended. Returns (text, was_cut): a cut is always visible on the board."""
    flat = " ".join(str(value or "").split())
    if len(flat) <= limit:
        return flat, False
    head = flat[:limit]
    ends = [m.end() for m in _SENTENCE_END.finditer(flat[:limit + 1])]
    if ends:
        head = head[:ends[-1]]
    elif not flat[limit].isspace() and " " in head:
        head = head[:head.rindex(" ")]
    return head.rstrip() + TRUNCATION_MARKER, True


def entries_of(fm: dict) -> list[dict]:
    """The item's owed entries, each normalised to a dict. Tolerates the
    plain-string shape a hand edit would write."""
    out: list[dict] = []
    for raw in (fm.get(OWED_KEY) or []):
        if isinstance(raw, dict) and _text(raw.get("what")):
            out.append({"what": _text(raw.get("what")),
                        "kind": raw.get("kind") if raw.get("kind") in KINDS else "check",
                        "since": str(raw.get("since") or ""),
                        "recheck_after": str(raw.get("recheck_after") or ""),
                        "rechecks": int(raw.get("rechecks") or 0)})
        elif isinstance(raw, str) and _text(raw):
            out.append({"what": _text(raw), "kind": "check", "since": "",
                        "recheck_after": "", "rechecks": 0})
    return out


def add_owed(path: Path, whats: list[str], *, kind: str = "check", activity: str = "") -> bool:
    """Append entries (deduped by text) to the item's `owed` list."""
    from scripts.automod import backlog as B
    fm, _ = B._split_frontmatter(path.read_text(encoding="utf-8"))
    have = entries_of(fm)
    seen = {e["what"] for e in have}
    stamp = _stamp()
    new: list[dict] = []
    for w in whats:
        what = _text(w)
        if what and what not in seen:
            seen.add(what)
            new.append({"what": what, "kind": kind if kind in KINDS else "check", "since": stamp})
    if not new:
        return False
    return B.update_frontmatter(path, {OWED_KEY: [_compact(e) for e in have] + new},
                                activity=activity or ("owed: " + "; ".join(e["what"] for e in new))[:600])


def _compact(e: dict) -> dict:
    return {k: v for k, v in e.items() if v not in ("", None, 0)}


def _due(entry: dict, now: datetime) -> bool:
    when = entry.get("recheck_after") or ""
    if not when:
        return True
    try:
        at = datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError:
        return True
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at <= now


@dataclass
class Owing:
    item: Any
    fm: dict
    entries: list[dict]
    due: list[int]          # indexes into `entries`


def owing_items(boards: tuple[str, ...] | None = None, *, now: datetime | None = None,
                due_only: bool = True) -> list[Owing]:
    """Items (open or closed) with owed entries, oldest owed first."""
    from scripts.automod import backlog as B
    now = now or _now()
    out: list[Owing] = []
    for item in B.all_items(boards if boards is not None else B.DEFAULT_BOARDS):
        try:
            fm, _ = B._split_frontmatter(item.path.read_text(encoding="utf-8"))
        except OSError:
            continue
        entries = entries_of(fm)
        if not entries:
            continue
        due = [i for i, e in enumerate(entries) if _due(e, now)]
        if due_only and not due:
            continue
        out.append(Owing(item=item, fm=fm, entries=entries, due=due))
    out.sort(key=lambda o: (min((o.entries[i]["since"] for i in o.due), default="~") or "~",
                            o.item.id))
    return out


def recheck_date(value: str, now: datetime | None = None) -> str:
    """An ISO date the session named, clamped to (now, now + MAX_RECHECK_DAYS]."""
    from datetime import timedelta
    now = now or _now()
    try:
        at = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
    except ValueError:
        at = now + timedelta(days=1)
    at = max(at, now + timedelta(hours=1))
    at = min(at, now + timedelta(days=MAX_RECHECK_DAYS))
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


def close_item(path: Path, why: str) -> bool:
    """Close an open item on Lloyd's own ruling. The status move goes through
    the shared recorder, so `completed` and the activity line are the ones
    every other closer writes."""
    import yaml
    from app.backlog_move import record_status_move
    from scripts.automod import backlog as B
    text = path.read_text(encoding="utf-8")
    fm, body = B._split_frontmatter(text)
    if B._unparsed_guard(path, text, fm, "owed.close_item"):
        return False
    if str(fm.get("status") or "") in ("done",):
        return False
    if not record_status_move(fm, "done", f"closed by owed-check: {why}"[:600],
                              remove_tags=(B.NEEDS_HUMAN_TAG,)):
        return False
    fm["closed_by"] = "owed-check"
    path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body}", encoding="utf-8")
    return True


# The cheap first test, NOT the guard: the words one incident produced (#1772 on
# 2026-09-28, `9b22b6c2` "a placeholder follow-up is never filed"), kept because
# it is free and already right about those words. A denylist cannot close this
# property — #1932 arrived 2026-09-30 named `ignore` with the body `ignore`, and
# no list of stand-in words ever finishes, because the corpus is whatever the
# model next types into an optional field. What closes it is shape, below.
_PLACEHOLDER_NAMES = frozenset({"placeholder", "todo", "tbd", "follow-up", "follow up",
                                "untitled", "none", "n/a", "name"})

# What ends the leading token of a name: a colon, or a dash set off by spaces.
# The leads that mean "no follow-up". Narrower than `_PLACEHOLDER_NAMES` on
# purpose: "Follow-up: fix the cursor" leads with a word from that set and is work.
_NO_FOLLOW_UP_LEADS = frozenset({"none", "n/a", "nothing"})
_LEAD_SEPARATOR = re.compile(r"\s*:\s+|\s+[-–—]\s+")

# A body shorter than this with no sentence terminator in it is a token standing
# in for an instruction, not one. The margin is measured, not guessed. The job's
# own prompt (`workers/sources/owed_check.py:117-119`) asks for a body that "says
# what to change, why, and how to check it (at most six checkable clauses)", and
# the whole live population was replayed through this predicate on 2026-10-01: of
# the 142 backlog files carrying a `Filed by owed-check from #N's owed entry:`
# trailer, each replayed with its own title and the body as the model wrote it
# (trailer and `# title` heading stripped), 140 are kept and exactly 2 are
# refused — `#1772 Placeholder` (16 characters) and `#1932 ignore` (6) — while the
# shortest KEPT body is 53 characters (`#1915`, which also carries a full stop, so
# both halves of the conjunction pass it). The rule stays a conjunction because
# that is what buys the margin in both directions: a length-only floor set high
# enough to catch a lazy one-liner would have eaten #1915's 53-character real
# follow-up, and a terminator-only rule would refuse any real instruction the
# model happens to write as a bare clause.
MIN_INSTRUCTION_CHARS = 40
_SENTENCE_TERMINATORS = ".!?"
# Surrounding punctuation is decoration on an answer, so the same-token test
# strips it before comparing: `" IGNORE "` and `"Ignore."` are the same word.
_EDGE_PUNCT = string.punctuation + "\u2014\u2013\u2026\u2018\u2019\u201c\u201d"


def _bare(value: Any) -> str:
    """A name or body reduced to its answer: lower-cased, whitespace collapsed,
    stripped of the punctuation wrapped around it."""
    return " ".join(str(value or "").split()).lower().strip(_EDGE_PUNCT)


def is_real_follow_up(follow: dict) -> bool:
    """A follow-up worth filing: an instruction, not a stand-in token.

    The schema's `follow_up` object invites the model to fill it even for a
    ruling that needs none; on 2026-09-28 that filed #1772, named "Placeholder"
    with the body "Placeholder body", and on 2026-09-30 it filed #1932 named
    "ignore". Three tests, cheapest first, and the last two are structural:

    1. the denylist above — right about the words it names, open-ended as a
       mechanism, which is why #1933 stopped relying on it;
    2. the same-token test — name and body that reduce (`_bare`) to the SAME
       SINGLE WORD mean the model answered one blank twice. "ignore" / "Ignore."
       is that case, and it is refused here rather than by test 3: the body
       carries a full stop, so its shape looks like a sentence and only its
       equality with the name gives it away;
    3. the fragment test — a body under MIN_INSTRUCTION_CHARS with no sentence
       terminator anywhere in it ("nothing to do", "placeholder body") is a
       fragment. Its subject is the shape of a non-sentence and nothing else: a
       short body that does carry a terminator ("Do it.") passes this test on its
       own, and neither test consults a vocabulary, so "Ignore the 3 stale rows."
       files without anyone adding "ignore" to a list.
    """
    name = _text(follow.get("name")).strip().strip(".").lower()
    body = str(follow.get("body") or "").strip().lower()
    if not name or name in _PLACEHOLDER_NAMES or "placeholder" in name:
        return False
    # A name that LEADS with a stand-in token and then explains itself — "None —
    # ruling closes the entry" (#1998, #2002, #2020, all 2026-10-01) — is the
    # model saying there is no follow-up, in the blank where one goes.
    if _bare(_LEAD_SEPARATOR.split(name, maxsplit=1)[0]) in _NO_FOLLOW_UP_LEADS:
        return False
    if not body or body.startswith("placeholder") or body in _PLACEHOLDER_NAMES:
        return False
    bare_name = _bare(name)
    if bare_name and " " not in bare_name and bare_name == _bare(body):
        return False
    if len(body) < MIN_INSTRUCTION_CHARS and not any(c in body for c in _SENTENCE_TERMINATORS):
        return False
    return True


def apply_verdict(path: Path, entries: list[dict], answers: list[dict], *, item_id: int,
                  session_id: str = "", spawn_cap: int = 3, now: datetime | None = None) -> dict:
    """Write one owed-check answer onto the item. Returns what was done.

    `answers` is the parsed list: {"n", "outcome", "evidence", "ruling",
    "recheck_after", "follow_up": {"name", "body"}, "outside"}. An entry with
    no answer stays owed. Item-level moves (`reopen`, `close`) run after the
    entries are recorded, at most one of them, `reopen` winning.
    """
    from scripts.automod import backlog as B
    now = now or _now()
    by_n = {int(a["n"]): a for a in answers if str(a.get("n", "")).isdigit()}
    fm, _ = B._split_frontmatter(path.read_text(encoding="utf-8"))
    settled = list(fm.get(SETTLED_KEY) or [])
    outside = list(fm.get(OUTSIDE_KEY) or [])
    keep: list[dict] = []
    notes: list[str] = []
    filed: list[int] = []
    item_move = ""
    move_why = ""
    stamp = _stamp()
    for n, e in enumerate(entries, 1):
        a = by_n.get(n)
        if a is None:
            keep.append(_compact(e))
            continue
        out = a["outcome"]
        evidence, cut = _bounded(a.get("evidence"))
        if out == "recheck" and e.get("rechecks", 0) >= MAX_RECHECKS:
            out = "ruling"
            a = {**a, "ruling": a.get("ruling") or
                 f"rechecked {MAX_RECHECKS} times without an answer; closed as unobservable: "
                 f"{evidence or 'no evidence arrived'}"}
        if out == "recheck":
            when = recheck_date(a.get("recheck_after") or "", now)
            keep.append(_compact({**e, "recheck_after": when, "rechecks": e.get("rechecks", 0) + 1}))
            notes.append(f"#{n} recheck after {when[:10]}: {evidence}")
            continue
        if out == "outside":
            outside.append({"what": e["what"], "needs": _text(a.get("outside") or evidence, 400),
                            "since": stamp})
            notes.append(f"#{n} needs Alan's hands: {_text(a.get('outside') or evidence, 200)}")
            continue
        record = {"what": e["what"], "outcome": out, "at": stamp}
        if evidence:
            record["evidence"] = evidence
        if out in ("ruling", "work", "reopen", "close") and _text(a.get("ruling")):
            record["ruling"], ruling_cut = _bounded(a.get("ruling"))
            cut = cut or ruling_cut
        if cut:
            # A cut field names where the whole text lives: the artifact the
            # session gave, else the session that wrote it.
            artifact = _text(a.get("artifact"), 300) or session_id
            if artifact:
                record["artifact"] = artifact
        follow = a.get("follow_up") or {}
        if out in ("work", "ruling") and is_real_follow_up(follow) and len(filed) < spawn_cap:
            new = B.new_item(_text(follow.get("name"), 140),
                             f"{str(follow.get('body') or '').strip()}\n\n"
                             f"Filed by owed-check from #{item_id}'s owed entry: {e['what']}",
                             status="draft")
            filed.append(new.id)
            record["follow_up"] = new.id
        elif out == "work":
            # No follow-up could be filed (cap reached, none named, or a
            # placeholder): still owed.
            keep.append(_compact(e))
            notes.append(f"#{n} work owed but not filed this pass")
            continue
        if out == "reopen":
            item_move, move_why = "reopen", record.get("ruling") or evidence
        elif out == "close" and item_move != "reopen":
            item_move, move_why = "close", record.get("ruling") or evidence
        settled.append(record)
        notes.append(f"#{n} {out}: " + (record.get("ruling") or evidence)[:200]
                     + (f" -> #{record['follow_up']}" if record.get("follow_up") else ""))
    updates = {OWED_KEY: keep or None, SETTLED_KEY: settled or None, OUTSIDE_KEY: outside or None}
    from app.backlog_tags import normalize_tags
    legacy = B.NEEDS_HUMAN_TAG in normalize_tags(fm.get("tags"))
    B.update_frontmatter(path, updates,
                         activity=(f"owed-check ({session_id or 'session'}): " + "; ".join(notes))[:1500],
                         # Only when the legacy tag is there: a removal with
                         # nothing to remove would write `tags: []`.
                         remove_tags=(B.NEEDS_HUMAN_TAG,) if legacy else ())
    moved = ""
    is_open = str(fm.get("status") or "") != "done"
    if item_move == "reopen" and is_open:
        try:
            B.reopen_item(item_id, f"owed-check: {move_why or 'another attempt granted'}"[:400])
            moved = "reopened"
        except ValueError as exc:
            # Never attempted: nothing to reopen. Back into the implement pool instead.
            moved = "up_next" if B.set_status(item_id, "up_next", f"owed-check: {move_why}"[:300]) \
                else f"reopen refused: {exc}"[:200]
    elif item_move == "close" and is_open:
        moved = "closed" if close_item(path, move_why) else ""
    elif is_open and settled and not keep and not outside:
        # #1909: the sweep that empties the list is the sweep that closes. An open
        # item with nothing owed was a hole, not a neutral state — #1751 sat in it
        # until a hand sweep on 2026-09-30 found its entries already settled, and
        # a `decide` entry on a draft (what a `human-only:` guard leaves behind)
        # ruled settled left the same hole. The job visits only items that owe
        # something, so nothing else was ever going to look at it again, and the
        # draft it stayed in is re-offered to triage for ever.
        #
        # `not keep` alone would not do: an entry ruled `outside` leaves `keep`
        # empty and is still a debt, and one sent for recheck or left as unfiled
        # `work` stays in `keep`. This is the only place the emptiness is acted on,
        # so it is stated once, at the write.
        last = settled[-1]
        moved = "closed" if close_item(
            path, "every owed entry is settled — the last "
                  f"({last['outcome']}) at {str(last['at'])[:10]}: "
                  f"{last.get('evidence') or last.get('ruling') or 'no detail given'}"[:400],
        ) else ""
    return {"remaining": len(keep), "settled": len(settled), "outside": len(outside),
            "filed": filed, "moved": moved, "notes": notes}


def outside_list(boards: tuple[str, ...] | None = None) -> list[dict]:
    """Everything on the board that only Alan's hands can do, for Mission Control."""
    from scripts.automod import backlog as B
    out: list[dict] = []
    for item in B.all_items(boards if boards is not None else B.DEFAULT_BOARDS):
        try:
            fm, _ = B._split_frontmatter(item.path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for o in (fm.get(OUTSIDE_KEY) or []):
            if isinstance(o, dict) and not o.get("done"):
                out.append({"item_id": item.id, "name": item.name,
                            "what": _text(o.get("what"), 300), "needs": _text(o.get("needs"), 300),
                            "since": str(o.get("since") or "")})
    return out


def defer(path: Path, entries: list[dict], due: list[int], *, hours: float = 24,
          why: str = "") -> bool:
    """Push due entries back without spending a recheck — for a pass that got
    no answer (the session failed), which says nothing about the entry."""
    from datetime import timedelta
    from scripts.automod import backlog as B
    when = (_now() + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    kept = [_compact({**e, "recheck_after": when}) if i in due else _compact(e)
            for i, e in enumerate(entries)]
    return B.update_frontmatter(path, {OWED_KEY: kept},
                                activity=f"owed-check deferred {len(due)} entr(ies) to {when[:16]}: {why}"[:600])
