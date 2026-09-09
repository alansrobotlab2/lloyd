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

from app.backlog_tags import normalize_tags

BACKLOG_DIR = Path.home() / "obsidian" / "backlog"
OPEN_STATUSES = {"up_next", "draft", "in_progress"}

# Tags `backlog_write_task` puts on items this loop files for itself —
# `spawned-by-triage` from a verdict turn, `spawned-by-autoimplement` from an
# implement round. They are how a self-filed item is told from a human one,
# and `draft` alone cannot do it: that is the status of half the board.
# Both spellings: ~100 items on the board carry the pre-rename tag, and
# quarantine that stopped recognising them would re-admit every one of them
# to the triage pool at once. New items get the new tag.
SPAWN_TAGS = frozenset({"spawned-by-triage", "spawned-by-autoimplement", "spawned-by-selfmod"})

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

IMPLEMENT_OUTCOME_SCHEMA: dict = {
    "type": "object",
    "title": "backlog_implement_outcome",
    "properties": {
        "landed": {"type": "boolean",
                   "description": ("Did this turn call autoimplement_land (or autoimplement_vault_land) "
                                   "on a change that passed the gate?")},
        "acceptance": {"type": "string", "enum": list(ACCEPTANCE_OUTCOMES),
                       "description": ("met: the acceptance check recorded at triage is now "
                                       "true. not_met: it is not, and this round did not "
                                       "make it so. deferred: it cannot be judged until "
                                       "something else happens — name it in deferred_to. "
                                       "unnecessary: the work is not needed after all (the "
                                       "premise no longer holds, or it is already true) and "
                                       "the item should close without a landing.")},
        "deferred_to": {"type": "array", "items": {"type": "integer"},
                        "description": ("Backlog ids that must close before the acceptance "
                                        "can be judged. Empty unless acceptance is deferred.")},
        "summary": {"type": "string",
                    "description": "One sentence: what landed, or why nothing did."},
        "spawned": {"type": "array", "items": {"type": "integer"},
                    "description": "Backlog ids filed during this round."},
    },
    "required": ["landed", "acceptance", "deferred_to", "summary", "spawned"],
    "additionalProperties": False,
}


def parse_outcome(structured) -> dict | None:
    """The finalizer's object, validated and clamped, or None if unusable.

    Clamped here rather than in the grammar for the reason the triage schema
    carries no `maxLength`: a guided decoder stops mid-sentence at a limit
    rather than writing something shorter.
    """
    if not isinstance(structured, dict):
        return None
    acceptance = str(structured.get("acceptance") or "")
    if acceptance not in ACCEPTANCE_OUTCOMES:
        return None

    def ints(v) -> list[int]:
        out: list[int] = []
        for x in (v or []):
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    return {"landed": bool(structured.get("landed")), "acceptance": acceptance,
            "deferred_to": ints(structured.get("deferred_to")),
            "summary": " ".join(str(structured.get("summary") or "").split())[:400],
            "spawned": ints(structured.get("spawned"))}


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
        tags=normalize_tags(fm.get("tags")),
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

# Stop reasons that mean the turn ran out of room rather than reaching a
# conclusion. #446 committed 757 lines and was killed by the wall clock
# fourteen seconds before its `autoimplement_gate` call.
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
    `infra`, `rolled_back` — where everything but `spent` means the item is
    offered again. Exposed rather than folded into `implemented_ids` because
    the reason is worth putting in front of the next round: a re-offer whose
    branch still exists, or whose work was reverted, is not a fresh start.
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
    promoted = {str(d.get("round_id") or "") for d in
                _ledger_events(ledger, "promoted", require_item=False)}
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
                        + (f"; its work is on branch `autoimplement/{rid}`" if rid else ""))
            continue
        if rid and rid in blocked and n <= EXTERNAL_RETRY_CAP:
            g = _last_gate_per_round(ledger).get(rid) or {}
            out[iid] = ("external",
                        f"round {rid} was blocked at the `{g.get('rung')}` rung by a condition "
                        f"it did not cause; its work is on branch `autoimplement/{rid}`")
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


# Every landed item stayed open. Nine promotions settled in the loop's first
# three days and not one item was closed: `promote` writes the commit,
# the guardian writes `settled`, `execute` writes `finished`, and nothing
# joined the three back to the item's status. `implemented_ids` kept them
# from being re-picked, so they sat on the board as `up_next` and `draft` —
# the loop's own finished work, counted as its backlog.
LANDED_MARKER = "autoimplement_landed"
# Items landed before the rename carry the old marker. Read both; write the new.
_LEGACY_LANDED_MARKERS = ("selfmod_landed",)


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
                 close: bool, why: str) -> Path:
    """Record the landing on the item; close it when `close`.

    The marker is written either way, so a landing is processed once. A
    human can still close an item the loop left open; the loop will not
    reopen one a human closed.
    """
    fm, body = _split_frontmatter(item.path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    where = f"round {round_id}" if round_id else "a vault round"
    entry = (f"**{stamp}** — autoimplement landed as `{commit[:8]}` ({where}, "
             f"{'settled' if round_id else 'committed'} {settled_at}). "
             + (f"Closed: {why}" if close else f"Left open: {why}"))
    log = list(fm.get("activity_log") or [])
    log.append(entry)
    fm["activity_log"] = log
    fm["updated"] = stamp
    fm[LANDED_MARKER] = commit
    if close:
        fm["status"] = "done"
        fm["completed"] = stamp
    section = (f"\n\n## Autoimplement landed — {stamp[:10]}\n\n`{commit[:8]}`, {where}, "
               f"{'settled' if round_id else 'committed'} {settled_at}.\n\n"
               + ("**Closed.** " if close else "**Left open.** ") + why + "\n")
    item.path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
        f"---\n{body.rstrip()}{section}", encoding="utf-8")
    return item.path


def close_settled_items(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                        enabled: bool = True) -> list[dict]:
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
    from scripts.autoimplement import state as S
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
        if acc == "met":
            close = True
            why = "the round reported the acceptance check met" + (
                f" — {outcome['summary']}" if outcome.get("summary") else "")
        elif acc == "deferred":
            ids = ", ".join(f"#{i}" for i in outcome.get("deferred_to") or []) or "an unnamed follow-up"
            close, why = False, (f"the round deferred the acceptance check to {ids}; "
                                 f"close this when that closes")
        elif acc == "not_met":
            close, why = False, "the round landed but reported the acceptance check not met"
        else:
            close, why = False, ("the round recorded no structured outcome (it predates the "
                                 "finalizer); a human decides")
        close_landed(item, commit=landing["commit"], round_id=landing["round_id"],
                     settled_at=str(landing.get("settled_at") or ""), close=close, why=why)
        S.append_event({"event": "item_landed", "item_id": item.id,
                        "round_id": landing["round_id"], "commit": landing["commit"],
                        "vault": landing["vault"], "closed": close,
                        "acceptance": acc, "reason": why[:300]}, path=ledger)
        done.append({"item_id": item.id, "closed": close, "acceptance": acc})
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
PIPELINE_STATUSES = ("draft", "up_next", "in_progress", "done")
TRIAGE_POOL_STATUS = "draft"
IMPLEMENT_POOL_STATUS = "up_next"


def set_status(item_id: int, status: str, why: str) -> bool:
    """Move an open item, once, with the reason in its log. False if it is
    not open, already there, or `done` (terminal for this writer)."""
    if status not in PIPELINE_STATUSES:
        raise ValueError(f"unknown status {status!r}")
    for item in open_items(None):
        if item.id != int(item_id):
            continue
        fm, body = _split_frontmatter(item.path.read_text(encoding="utf-8"))
        if fm.get("status") == status or fm.get("status") == "done":
            return False
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        log = list(fm.get("activity_log") or [])
        log.append(f"**{stamp}** — {fm.get('status')} → {status}: {why}")
        fm["activity_log"] = log
        fm["status"] = status
        fm["updated"] = stamp
        if status == "done":
            fm["completed"] = stamp
        item.path.write_text(
            f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)}"
            f"---\n{body}", encoding="utf-8")
        return True
    return False


def desired_statuses(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS,
                     *, open_round_items: set[int] = frozenset()) -> dict[int, tuple[str, str]]:
    """`{item_id: (status, why)}` — what the ledger says each open item's
    status should be. Only items the loop has an opinion about appear.

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
        if iid in in_flight:
            out[iid] = ("in_progress", "an autoimplement round is in flight for it")
        elif landed:
            out[iid] = ("in_progress", "landed, awaiting the acceptance check or a human close")
        elif iid in observing:
            # Promoted, not yet settled: the guardian is watching it and the
            # sweep has not run. Neither back in the pool nor done.
            out[iid] = ("in_progress", "landed; the promotion is under observation")
        elif iid in outcomes:
            verdict, detail = outcomes[iid]
            if verdict == "spent":
                out[iid] = ("up_next", "its one unattended attempt is spent; a human decides "
                                       "(reopen_item to grant another)")
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


def reconcile_statuses(ledger: Path, boards: tuple[str, ...] | None = DEFAULT_BOARDS, *,
                       open_round_items: set[int] = frozenset(), enabled: bool = True) -> list[dict]:
    """Write the desired statuses that differ. Idempotent; returns what moved."""
    if not enabled:
        return []
    from scripts.autoimplement import state as S
    moved: list[dict] = []
    current = {i.id: i.status for i in open_items(boards)}
    for iid, (status, why) in desired_statuses(ledger, boards, open_round_items=open_round_items).items():
        if current.get(iid) == status:
            continue
        if set_status(iid, status, why):
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
    from scripts.autoimplement import state as S
    S.append_event({"event": "backlog_implement", "item_id": int(item_id), "phase": "reopened",
                    "reason": reason}, path=ledger)
    note_item(item_id, f"reopened for a second autoimplement attempt: {reason}")
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
    from scripts.autoimplement import state as S
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
    # `draft` only: that is where an item waits to be judged. `up_next` is the
    # implement pool and `in_progress` is a round in flight; an untriaged item
    # in either is a dead state the reconciler moves back here.
    untriaged = [i for i in open_items(boards)
                 if i.id not in seen and i.status == TRIAGE_POOL_STATUS]
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
        # The board is the state machine now: a confirmed item the loop may
        # take sits in `up_next`, and only there. A human parks one anywhere
        # else to keep it out of the loop's hands.
        if item.status != IMPLEMENT_POOL_STATUS:
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
    entry = f"**{stamp}** — autoimplement triage: **{verdict}**. {evidence.strip()}"
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
        fm["autotriage_retired"] = verdict
    elif verdict == "confirmed" and fm.get("status") != "done":
        # Into the implement pool. Before 2026-09-09 a confirmed item stayed
        # wherever it was, and #353 landed while still `draft`.
        fm["status"] = IMPLEMENT_POOL_STATUS

    section = (f"\n\n## Autoimplement triage — {stamp[:10]}\n\n"
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
