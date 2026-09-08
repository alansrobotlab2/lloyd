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

BACKLOG_DIR = Path.home() / "obsidian" / "backlog"
OPEN_STATUSES = {"up_next", "draft", "in_progress"}

# Tags `backlog_write_task` puts on items this loop files for itself —
# `spawned-by-triage` from a verdict turn, `spawned-by-selfmod` from an
# implement round. They are how a self-filed item is told from a human one,
# and `draft` alone cannot do it: that is the status of half the board.
SPAWN_TAGS = frozenset({"spawned-by-triage", "spawned-by-selfmod"})

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
        "spawned": {"type": "array", "items": {"type": "integer"},
                    "description": "Backlog ids filed during this triage."},
    },
    "required": ["verdict", "surface", "check", "evidence", "acceptance", "spawned"],
    "additionalProperties": False,
}


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
        tags=[str(t) for t in (fm.get("tags") or [])],
    )


def open_items(boards: tuple[str, ...] | None = DEFAULT_BOARDS) -> list[Item]:
    """Open items, restricted to `boards` unless it is None."""
    wanted = {b.lower() for b in boards} if boards else None
    out = []
    for path in sorted(BACKLOG_DIR.glob("*.md")):
        item = load_item(path)
        if not item or item.status not in OPEN_STATUSES:
            continue
        if wanted is not None and item.board.lower() not in wanted:
            continue
        out.append(item)
    return out


def _ledger_events(ledger: Path, event: str) -> list[dict]:
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
            if d.get("event") == event and d.get("item_id") is not None:
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


def implemented_ids(ledger: Path) -> set[int]:
    """Items an implementation turn has already been run for, whatever it did —
    unless a human has since reopened them (`reopen_item`), in which case the
    latest event is `reopened` and the item is eligible for exactly one more
    attempt. One attempt per item is the rule; a second is a human's call, and
    this is how the human makes it."""
    latest: dict[int, str] = {}
    for d in _ledger_events(ledger, "backlog_implement"):
        latest[int(d["item_id"])] = str(d.get("phase") or "")
    return {i for i, phase in latest.items() if phase != "reopened"}


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
    from scripts.selfmod import state as S
    S.append_event({"event": "backlog_implement", "item_id": int(item_id), "phase": "reopened",
                    "reason": reason}, path=ledger)
    note_item(item_id, f"reopened for a second selfmod implement attempt: {reason}")
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
    from scripts.selfmod import state as S
    return S.LEDGER_PATH


def is_self_spawned(item: Item) -> bool:
    """Did this loop write this item? `backlog_write_task` tags what it files."""
    return any(t in SPAWN_TAGS for t in item.tags)


def is_quarantined(item: Item) -> bool:
    """A freshly self-filed item is not a triage candidate.

    Triage asks one question: does this old claim still describe the system?
    An item this loop filed minutes ago, from a check it just ran against
    live code with file paths and line numbers, cannot answer it — it is not
    stale, by construction. Re-asking costs a 90-turn session to re-confirm
    what the previous session proved.

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

    Quarantine, not exclusion: a spawned item that nobody implements really
    can go stale, so it becomes a candidate again once it is old enough for
    that to be a real question. Until then the pass has nothing to tell us
    about it.
    """
    return is_self_spawned(item) and item.age_days < SPAWN_TRIAGE_MIN_AGE_DAYS


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
    untriaged = [i for i in open_items(boards) if i.id not in seen]
    fresh = [i for i in untriaged if not is_quarantined(i)]
    return fresh, len(untriaged) - len(fresh)


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
    done = implemented_ids(ledger)
    ready = []
    for item in open_items(boards):
        ev = confirmed.get(item.id)
        if not ev or item.id in done:
            continue
        if not acceptance_text(ev.get("acceptance")):
            continue
        if is_human_only(ev.get("acceptance")):
            continue
        ready.append((item, ev))
    if not ready:
        return None
    return sorted(ready, key=lambda pair: (pair[0].created or "9999", pair[0].id))[0]


def record_verdict(item: Item, verdict: str, evidence: str, *,
                   check: str = "", close: bool = False,
                   spawned: list[int] | tuple[int, ...] = (),
                   acceptance: str = "") -> Path:
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
    entry = f"**{stamp}** — selfmod triage: **{verdict}**. {evidence.strip()}"
    if check:
        entry += f" Check: `{check}`"
    if spawned:
        entry += " Filed as new items: " + ", ".join(f"#{i}" for i in spawned) + "."
    log = list(fm.get("activity_log") or [])
    log.append(entry)
    fm["activity_log"] = log
    fm["updated"] = stamp

    if close and verdict in RETIRING:
        fm["status"] = "done"
        fm["selfmod_retired"] = verdict

    section = (f"\n\n## Selfmod triage — {stamp[:10]}\n\n"
               f"**Verdict:** {verdict}\n\n{evidence.strip()}\n")
    if check:
        section += f"\n**Premise check:**\n```\n{check.strip()}\n```\n"
    if spawned:
        section += "\n**Filed as new items:** " + ", ".join(f"#{i}" for i in spawned) + "\n"
    if acceptance.strip():
        # The item is the handoff. A contract that lives only in the ledger
        # and a transcript is one the next reader of the item never sees.
        section += f"\n**Acceptance — what must become true:**\n{acceptance.strip()}\n"

    item.path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body.rstrip()}{section}",
        encoding="utf-8")
    return item.path


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
