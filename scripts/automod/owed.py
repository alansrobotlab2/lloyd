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

# The origin an entry carries when it was never written into `owed`: derived at
# read time from the item's `human_clauses`. It is what lets a ruling tell a
# stranded clause from one the landing route recorded (#2055).
STRANDED_ORIGIN = "human_clauses"

# How many items whose ONLY entries are derived one owed-check run may claim.
# The scan on 2026-10-02 found 454 `status: done` items in #1999's state — 72 of
# them closed on or after 2026-09-27, when the "nothing parks on Alan" route
# landed in aa68ea74 — and `owing_items` sorts oldest-first and returns all of
# them. An unbounded derivation therefore hands the next tick 454 items and every
# genuinely recorded entry behind them waits behind historical debt (#2055).
MAX_STRANDED_PER_RUN = 5


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


def _recorded(fm: dict) -> list[dict]:
    """The entries the item itself carries in `owed`, each normalised to a dict.
    Tolerates the plain-string shape a hand edit would write.

    This is the writer's view: `add_owed` and `apply_verdict` must materialise
    only what is on the record, never a read-time derivation, or a derivation
    would become front matter by side effect of some unrelated write."""
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


def _since(fm: dict) -> str:
    """When a derived clause started being owed: the day the item stopped having
    an owner, which for a closed item is `completed:`, falling back through the
    last touch and the filing date so the entry never claims a date it cannot
    name."""
    for key in ("completed", "updated", "created"):
        value = str(fm.get(key) or "").strip()
        if value:
            return value
    return ""


def derived_entries(fm: dict) -> list[dict]:
    """`human_clauses` strings that appear in none of `owed`, `owed_settled` and
    `owed_outside`.

    Only a CLOSED item is derived from (the boundary `tests/test_backlog_unattended.py`
    already encodes at :4595, where an open item's `entries_of` must be empty until
    the landing mints its row). An open item's `human_clauses` are its round's
    contract: `split_post_landing_clauses` shows the landing route owes only the
    POST-LANDING half of them and grades the landing-time half through the review
    rung, so deriving an open item's clauses would hand the owed job clauses its own
    round is still working, and it could `reopen` or duplicate a round in flight. The
    hole is a close that skipped the route, so the derivation stops there.

    The hole (#2055): the conversion of `human_clauses` into owed entries
    (`add_owed(human_clauses_of(...))`) is called only from the landing writers in
    `scripts/automod/backlog.py`. An item closed any other way keeps its clauses
    in front matter and gets no `owed:` key, and `owing_items` read `owed` alone —
    so #1999, closed on 2026-10-01 by a hand sweep with three clauses and no
    landing section, was invisible to the job whose whole job is ruling on clauses
    a diff cannot settle. Deriving at read time needs no migration and no write:
    the clause is on the record already, the reader just was not looking there.

    Dedupe is by the flattened text `_text` of each side — the same normalisation
    `owed` entries already pass through on the way in and out, so a clause the
    landing route did record matches its own recorded copy exactly and yields
    nothing here. `owed_settled` counts as recorded: a clause that has already
    been ruled on is not stranded, whatever its text went on to become.
    """
    from app.backlog_status import CLOSED_STATUSES
    from scripts.automod import backlog as B
    if str(fm.get("status") or "").strip().lower() not in CLOSED_STATUSES:
        return []
    clauses = B.human_clauses_of(None, fm)
    if not clauses:
        return []
    seen = {_text(e.get("what")) for e in _recorded(fm)}
    seen |= {_text(r.get("what")) for r in (fm.get(SETTLED_KEY) or []) if isinstance(r, dict)}
    # `owed_outside` is a ruling too. Left out, a derived clause ruled `outside`
    # was derived again on the next tick, ruled again and announced again: #538
    # was filed on Alan's list six times in 28 minutes on 2026-10-02, a toast and
    # a spoken alert each, until a seventh session happened to answer `settled`.
    seen |= {_text(r.get("what")) for r in (fm.get(OUTSIDE_KEY) or []) if isinstance(r, dict)}
    seen.discard("")
    since = _since(fm)
    out: list[dict] = []
    for clause in clauses:
        what = _text(clause)
        if not what or what in seen:
            continue
        seen.add(what)
        out.append({"what": what, "kind": "check", "since": since,
                    "recheck_after": "", "rechecks": 0, "origin": STRANDED_ORIGIN})
    return out


def entries_of(fm: dict, *, derived: bool = True) -> list[dict]:
    """The item's owed entries: what `owed` records, plus — since #2055 — one due
    `check` entry per `human_clauses` string recorded in neither `owed` nor
    `owed_settled`. `derived=False` is the writer's view (`_recorded` only)."""
    out = _recorded(fm)
    if derived:
        out.extend(derived_entries(fm))
    return out


def add_owed(path: Path, whats: list[str], *, kind: str = "check", activity: str = "") -> bool:
    """Append entries (deduped by text) to the item's `owed` list."""
    from scripts.automod import backlog as B
    fm, _ = B._split_frontmatter(path.read_text(encoding="utf-8"))
    # The writer's view: a read-time derivation must never be materialised into
    # `owed` as a side effect of an unrelated append (#2055).
    have = _recorded(fm)
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
                due_only: bool = True,
                stranded_cap: int | None = MAX_STRANDED_PER_RUN) -> list[Owing]:
    """Items (open or closed) with owed entries, oldest owed first.

    An item with nothing but derived entries is a STRANDED one, and at most
    `stranded_cap` of them are returned (#2055 clause 4): the 454 on the board on
    2026-10-02 cannot be claimed in one run, and the bound is applied AFTER the
    oldest-first sort so which ones get ruled on first is the wait they have
    actually done, not the board's directory order. An item with at least one
    recorded entry is never held back by it — the bound is on the historical
    back-fill, not on anything a route filed on purpose. `stranded_cap=None` asks
    for the whole population, which is what a caller that is COUNTING rather than
    claiming wants.
    """
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
    if stranded_cap is not None:
        kept: list[Owing] = []
        stranded = 0
        for o in out:
            if all(e.get("origin") == STRANDED_ORIGIN for e in o.entries):
                if stranded >= stranded_cap:
                    continue
                stranded += 1
            kept.append(o)
        out = kept
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


# ── The child copy of an owed line (#2013) ──────────────────────────────────
#
# A filed follow-up quotes the parent's owed line in its trailer. Copied
# verbatim it carried the parent's `file:line` range with no root and no pin,
# and the child is what a later round opens: measured 2026-10-01, 62 of 192
# children held such a citation and #1932's child pointed `SKILL.md:240-245`
# at a paragraph that had moved to `:300-317`. The copy drops the range, keeps
# the path, says which tree the path resolves in, and names the revision it
# was read at. The parent's own entry is never rewritten.

#: label -> tree. None means "the live vault and this checkout", resolved on
#: first use; a test repoints it at temp dirs.
CITE_ROOTS: dict[str, Path] | None = None
UNPINNED = "cited at an unpinned revision"
_CITED_PATH = re.compile(
    r"(?<![\w/.~-])(?P<path>~?[\w./-]*\w\.[A-Za-z][A-Za-z0-9]{0,5}):\d+(?:-\d+)?(?![\w])")


def _cite_roots() -> dict[str, Path]:
    if CITE_ROOTS is not None:
        return CITE_ROOTS
    from app import paths
    return {"vault": paths.VAULT_ROOT, "repo": paths.LLOYD_HOME}


def _cite_label(token: str) -> str:
    """Which tree resolves `token`: `<name>-relative`, or `root unresolved`
    when none does — never a guessed root."""
    roots = _cite_roots()
    expanded = Path(token).expanduser()
    for name, root in roots.items():
        try:
            if expanded.is_absolute():
                if expanded.is_file() and expanded.resolve().is_relative_to(root.resolve()):
                    return f"{name}-relative"
            elif (root / token).is_file():
                return f"{name}-relative"
        except OSError:
            continue
    return "root unresolved"


def _cite_revision() -> str:
    """The short HEAD of each cite root that is a git checkout, e.g.
    `repo a03a7300, vault 29302c7f`; empty when none can be read."""
    import subprocess
    parts = []
    for name, root in _cite_roots().items():
        try:
            r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0 and r.stdout.strip():
            parts.append(f"{name} {r.stdout.strip()}")
    return ", ".join(parts)


def cite_for_child(what: str, revision: str | None = None) -> str:
    """The owed line as a filed child may quote it: every `path:start-end`
    loses its range and gains a root label, and the whole carries a revision
    marker (`UNPINNED` when no revision is known)."""
    def _sub(m: re.Match) -> str:
        token = m.group("path")
        return f"{token} ({_cite_label(token)})"
    reduced = _CITED_PATH.sub(_sub, what)
    if revision is None:
        try:
            revision = _cite_revision()
        except Exception:  # noqa: BLE001 — a pin is a courtesy, never a failed filing
            revision = ""
    return f"{reduced} [{'cited at ' + revision if revision else UNPINNED}]"


def apply_verdict(path: Path, entries: list[dict], answers: list[dict], *, item_id: int,
                  session_id: str = "", spawn_cap: int = 3, now: datetime | None = None,
                  revision: str | None = None) -> dict:
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
            # One entry per clause: a second `outside` ruling on the same clause
            # restates the ask in place and keeps the date it has waited since.
            ask = {"what": e["what"], "needs": _text(a.get("outside") or evidence, 400),
                   "since": stamp}
            same = [i for i, o in enumerate(outside) if isinstance(o, dict)
                    and not o.get("done") and _text(o.get("what")) == _text(e["what"])]
            if same:
                ask["since"] = str(outside[same[0]].get("since") or stamp)
                outside = [o for i, o in enumerate(outside) if i not in same[1:]]
                outside[same[0]] = ask
            else:
                outside.append(ask)
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
                             f"Filed by owed-check from #{item_id}'s owed entry: "
                             f"{cite_for_child(e['what'], revision)}",
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
