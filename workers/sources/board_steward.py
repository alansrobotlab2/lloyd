"""One steward turn instead of a state machine — cut 3 of senses-not-supervision.

`scripts/automod/backlog.py` grew to 2,491 lines and 90 functions, most of
them deciding what should happen to an item: `desired_statuses`,
`reconcile_statuses`, `rescue_off_vocabulary`, `implement_outcomes` with six
outcome classes and a cap per class, quarantine, expiry, `select_confirmed`'s
three-tier ordering. Every one is a rule the model was not trusted to apply,
written after its verdict was captured by a regex and came out wrong. The
finalizer fixed the capture; the rules never came out.

This source is the replacement, shipped **beside** the state machine rather
than instead of it. Every run it reads the ledger events since its last run
and the open board, and answers with one structured object: a list of moves
and the item autocode should take next. With `apply: false` (the shipped
default) it records how far its answer agrees with what the state machine
would have done and writes nothing — that record is what decides whether
`apply` flips. With `apply: true` one mechanical writer applies the moves and
`select_confirmed` takes its pick.

**The one rule that stays hard: the steward may not set `done`.** Closing is
gated on a fact — a settled promotion whose outcome said `met` — and stays in
`close_settled_items`. A closed item is never re-triaged, so this is the one
move the loop cannot recover from, and it is the one move no judgment makes.

It runs on the secondary by default: a judgment over frontmatter and ledger
rows does not need the primary, and the primary is what a round is using.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.board-steward")

NAME = "board-steward"
# Behind research (70) — a steward pass can wait for a research job — and
# never held by the KV gate or the round hold: it runs on the secondary and
# is what tells the pool which round to run next.
DEFAULT_PRIORITY = 68
LONG_LIVED = False
DEDUP_KEY = "board-steward:tick"
# Primary, not secondary, since the first live tick. Five dry-runs and one
# live tick on the secondary: two runs with ordering errors (an older event
# read over the latest), and the live tick spent all eight iterations on
# tool reads and died at `max_turns` with no structured answer. The primary
# made no ordering errors and answered in 90-170 s. A steward pass is a
# two-minute judgment every fifteen minutes; the primary can afford it, and
# the round hold keeps it off the engine while a round is running.
DEFAULT_MODEL = "primary"
# Headroom, not a target: the prompt says decide from what is shown, and a
# pass that needs sixteen tool calls is one that ignored the prompt.
DEFAULT_MAX_TURNS = 16
DEFAULT_MAX_EVENTS = 300
DEFAULT_MAX_ITEMS = 80
BODY_CHARS = 280

# What the steward may set. `done` is deliberately absent — see the module
# docstring — and the schema is built from this so the grammar and the
# validator cannot disagree.
STEWARD_STATUSES = ("draft", "up_next", "in_progress")

# Ledger events worth showing. Everything else is bookkeeping the steward
# does not need to see to decide an item's status.
_SHOWN_EVENTS = {
    "backlog_triage", "backlog_group_triage", "backlog_implement", "item_landed",
    "item_closed", "promoted", "settled", "rollback_succeeded", "round_abort",
    "land_failed", "status_moved", "review_escalated", "human_paths",
    "amend_clause", "reopen",
}

STEWARD_SCHEMA: dict = {
    "type": "object",
    "title": "board_steward",
    "properties": {
        "moves": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "item_id": {"type": "integer"},
                "status": {"type": "string", "enum": list(STEWARD_STATUSES),
                           "description": ("draft: triaged and not for the loop, or not yet "
                                           "triaged. up_next: confirmed and the loop may take "
                                           "it. in_progress: a round is running on it now.")},
                "tags_add": {"type": "array", "items": {"type": "string"}},
                "tags_remove": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string", "description": "One sentence, for the item's activity log."},
            },
            "required": ["item_id", "status", "tags_add", "tags_remove", "note"],
            "additionalProperties": False,
        }, "description": ("Only items whose status or tags should CHANGE. An item already "
                           "where it belongs is not listed.")},
        "next_pick": {"type": "integer",
                      "description": ("The up_next item autocode should implement next, or 0 "
                                      "when none is ready. Prefer a first re-offer whose branch "
                                      "still holds the work, then the oldest confirmed item.")},
        "next_pick_reason": {"type": "string"},
        "summary": {"type": "string", "description": "One sentence on the board's state."},
    },
    "required": ["moves", "next_pick", "next_pick_reason", "summary"],
    "additionalProperties": False,
}

PROMPT = """\
You are the steward of Lloyd's own backlog. Read the ledger events since your \
last pass and the open board below, and decide what should move.

The vocabulary, and it is the whole of what you may set:
- `draft` — triaged and not for the loop, or not yet triaged. Where an item \
waits for a human, and where a spent item goes back to.
- `up_next` — triage confirmed it and the loop may implement it. Only here.
- `in_progress` — a round is running on it right now.
- `done` is NOT yours. Closing is gated on a settled promotion whose outcome \
said `met`; you never propose it. An item that looks finished but has no such \
landing stays where it is with a note.

The rules the state machine applied, which you now apply with judgment:
- **A triage verdict moves the item.** `backlog_triage` with `confirmed` \
puts it in `up_next` — whatever it is now, unless a round is in flight or a \
human moved it since. `already_done` and `stale` close it, which is not \
yours to do. Any other verdict leaves it `draft`. An item that is `draft` \
with a `confirmed` verdict and no later event is the commonest move on this \
board, and the first dry-run missed exactly that (#898).
- An `umbrella` is an ordinary confirmed item: it goes to `up_next` and the \
loop implements it. Its `members` (tagged `grouped`) are the ones that stay \
`draft` — the umbrella carries them. Do not confuse the two.
- A `backlog_implement` `started` with nothing after it is a round in flight: \
`in_progress`. A round that ended with a promotion that `settled` and an \
outcome of `met` closes by itself; `not_met` / `deferred` leave it `up_next` \
for one more attempt on exactly the unmet clauses.
- A round that aborted for a reason outside its diff (`external`, a grader \
that was down, a pre-existing red test, a rebase conflict) has NOT spent the \
item: `up_next`. A round that spent its one attempt on a judgment of the \
change — a review refusal twice, `spent` — goes back to `draft` with the tag \
`needs-human`; the tag comes off with any move back into the pool.
- **A landed item that is waiting on a person stays where it is.** An \
`item_landed` with `closed=false` and a reason naming what a person still \
owes it (a post-landing check, a human clause, a `human_paths` entry) is \
held deliberately — the code is live and the item is open for someone to \
confirm. It is not "stuck in_progress"; propose no move and say so in your \
summary if you think the vocabulary is wrong for it.
- `spent` is spent. An item whose one attempt was consumed by a verdict on \
the change (`spent` in an outcome, "a human decides" in its status reason) \
goes to `draft` with `needs-human` and stays there until a human reopens it. \
Do not send it back to `up_next` because the outcome looks recoverable.
- A `rollback_succeeded` naming a landing's commit reopens its item: `up_next`.
- An item tagged `grouped` (folded into an umbrella) stays `draft` whatever \
its own history; the umbrella carries it.
- A human moved it by hand if the board disagrees with the ledger and the \
last event is older than the item's `updated`. Honour the human.
- Never move an item you have no event for. Silence is not evidence.

`next_pick`: the `up_next` item the loop should implement next. A first \
re-offer whose branch `automod/<round>` still exists is one fix cycle, not an \
hour — prefer it. Then the oldest confirmed item. Never an item with a round \
in flight, never a `grouped` member, never one whose acceptance begins \
`human-only:`.

<ledger since="{since}" events="{n_events}">
{events}
</ledger>

<board items="{n_items}" of="{n_open}">
{board}
</board>

Decide with the evidence above only. **Do not open items or run tools** — \
every item you may move is shown above with its history, and a pass that \
reads its way through the board runs out of budget before it answers (the \
first live pass did exactly that). Write your decision directly; when you \
finish you will be asked to restate it as one JSON object.
"""


# ── state ────────────────────────────────────────────────────────────────

def _state_path() -> Path:
    from scripts.automod import state as S
    return S.STATE_DIR / "board_steward.json"


def _pick_path() -> Path:
    from scripts.automod import state as S
    return S.STATE_DIR / "steward_pick.json"


def _read_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(d: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


# ── input ────────────────────────────────────────────────────────────────

def events_since(ledger: Path, since_ts: float, *, limit: int = DEFAULT_MAX_EVENTS,
                 for_items: set[int] = frozenset(), per_item: int = 6) -> list[dict]:
    """Shown events newer than `since_ts`, oldest first, bounded from the end
    — plus, for each id in `for_items`, its last `per_item` events whatever
    their age. An item the steward is shown with none of its history is one it
    can only guess about."""
    recent: list[dict] = []
    history: dict[int, list[dict]] = {i: [] for i in for_items}
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return recent
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("event") not in _SHOWN_EVENTS:
            continue
        iid = d.get("item_id")
        if isinstance(iid, int) and iid in history:
            history[iid].append(d)
        if float(d.get("ts") or 0) > since_ts:
            recent.append(d)
    out = recent[-limit:]
    have = {id(d) for d in out}
    for rows in history.values():
        for d in rows[-per_item:]:
            if id(d) not in have:
                out.append(d)
                have.add(id(d))
    out.sort(key=lambda d: float(d.get("ts") or 0))
    return out


def _event_line(d: dict) -> str:
    keep = ("event", "item_id", "round_id", "phase", "verdict", "acceptance", "commit",
            "reason", "detail", "closed", "from", "to", "by", "outcome")
    parts = []
    for k in keep:
        if k in d and d[k] not in (None, "", [], {}):
            v = d[k]
            if isinstance(v, dict):
                v = {kk: v[kk] for kk in ("acceptance", "landed") if kk in v}
            parts.append(f"{k}={str(v)[:120]}")
    ts = time.strftime("%m-%d %H:%M", time.gmtime(float(d.get("ts") or 0)))
    return f"{ts} " + " ".join(parts)


def board_view(items: list[Any], touched: set[int], *, max_items: int = DEFAULT_MAX_ITEMS,
               pending: set[int] = frozenset()) -> list[Any]:
    """Which open items the steward sees: everything in the pool, every item
    the state machine has a pending move for, then drafts an event touched or
    a human flagged. Bounded, in that order.

    `pending` is the fix for the first dry-run's one miss: #898 was a
    confirmed item parked in `draft` that the machine wanted in `up_next`,
    and the steward never proposed it because its triage event was older
    than the event window and nothing else had touched it. The steward
    cannot move what it is not shown, and a comparison against a machine
    that sees the whole board has to show it at least what the machine is
    about to act on.
    """
    # Pending FIRST. The pool alone is bigger than `max_items` on this board
    # (91 items in up_next/in_progress on 2026-09-12), so with the pool ahead
    # of it the one item the machine was about to move was truncated off the
    # end three dry-runs running — the steward saw its event and no board
    # line, and correctly refused to move an item it had not been shown.
    due = [i for i in items if i.id in pending]
    seen = {i.id for i in due}
    pool = [i for i in items if i.status in ("up_next", "in_progress") and i.id not in seen]
    seen |= {i.id for i in pool}
    rest = [i for i in items if i.id not in seen
            and (i.id in touched or "needs-human" in (i.tags or []))]
    return (due + pool + rest)[:max_items]


def _item_line(i: Any) -> str:
    body = " ".join((i.body or "").split())[:BODY_CHARS]
    tags = ",".join(i.tags or [])
    rel = []
    if i.group:
        rel.append(f"group=#{i.group}")
    if i.members:
        rel.append(f"members={len(i.members)}")
    if i.parent:
        rel.append(f"parent=#{i.parent}")
    return (f"#{i.id} [{i.status}] {i.name[:90]} | created={i.created[:10]} "
            f"tags={tags or '-'} {' '.join(rel)}\n    {body}")


def build_prompt(*, events: list[dict], items: list[Any], n_open: int, since_ts: float) -> str:
    since = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(since_ts)) if since_ts else "the beginning"
    return PROMPT.format(
        since=since, n_events=len(events),
        events="\n".join(_event_line(d) for d in events) or "(none)",
        n_items=len(items), n_open=n_open,
        board="\n".join(_item_line(i) for i in items) or "(empty)",
    )


# ── output ───────────────────────────────────────────────────────────────

def parse_steward(structured: Any) -> dict | None:
    """Validated and clamped, or None. `done` can never come out of here."""
    if not isinstance(structured, dict):
        return None
    moves: list[dict] = []
    for raw in (structured.get("moves") or []):
        if not isinstance(raw, dict):
            continue
        try:
            iid = int(raw.get("item_id") or 0)
        except (TypeError, ValueError):
            continue
        status = str(raw.get("status") or "").strip()
        if iid <= 0 or status not in STEWARD_STATUSES:
            continue
        moves.append({
            "item_id": iid, "status": status,
            "tags_add": [str(x) for x in (raw.get("tags_add") or []) if x][:6],
            "tags_remove": [str(x) for x in (raw.get("tags_remove") or []) if x][:6],
            "note": " ".join(str(raw.get("note") or "").split())[:300],
        })
    try:
        pick = int(structured.get("next_pick") or 0)
    except (TypeError, ValueError):
        pick = 0
    return {"moves": moves, "next_pick": max(0, pick),
            "next_pick_reason": " ".join(str(structured.get("next_pick_reason") or "").split())[:300],
            "summary": " ".join(str(structured.get("summary") or "").split())[:400]}


def agreement(moves: list[dict], expected: dict[int, tuple], current: dict[int, str]) -> dict:
    """How far the steward's moves match the state machine's table.

    `expected` is `desired_statuses(...)`: `{item_id: (status, why[, needs_human])}`
    for every item the machine has an opinion about. Three counts, kept
    separate because they mean different things: moves that agree, moves the
    machine would not make (the steward's own opinion — a human reads these),
    and machine moves the steward did not propose (misses).
    """
    proposed = {m["item_id"]: m["status"] for m in moves}
    agree = [i for i, s in proposed.items() if i in expected and expected[i][0] == s]
    # Two different things the first dry-run counted as one. `disagree` is
    # the machine saying A and the steward B — a real conflict. `no_opinion`
    # is the machine deliberately leaving an item where it is (the stranded
    # landings it parks `in_progress` for a human) and the steward having a
    # view; that is not the steward being wrong, it is the machine abstaining.
    disagree = [i for i, s in proposed.items() if i in expected and expected[i][0] != s]
    no_opinion = [i for i, s in proposed.items() if i not in expected]
    missed = [i for i, want in expected.items()
              if current.get(i) != want[0] and proposed.get(i) != want[0]]
    machine_moves = [i for i, want in expected.items() if current.get(i) != want[0]]
    # Rate over the decisions BOTH sides made: the machine's pending moves and
    # the steward's moves on items the machine has an opinion about.
    judged = set(machine_moves) | {i for i in proposed if i in expected}
    return {"agree": sorted(agree), "disagree": sorted(disagree),
            "no_opinion": sorted(no_opinion), "missed": sorted(missed),
            "machine_moves": len(machine_moves), "proposed": len(proposed),
            "rate": (len(agree) / len(judged)) if judged else 1.0}


def apply_moves(moves: list[dict], *, round_id: str = "") -> list[dict]:
    """Write the moves through the one status writer. Never `done`."""
    from scripts.automod import backlog as B
    done: list[dict] = []
    for m in moves:
        if m["status"] not in STEWARD_STATUSES:
            continue      # belt and braces over the schema
        why = f"board steward: {m['note'] or 'no note'}"
        moved = B.set_status(m["item_id"], m["status"], why,
                             add_tags=tuple(m["tags_add"]), remove_tags=tuple(m["tags_remove"]))
        if not moved and (m["tags_add"] or m["tags_remove"]):
            moved = B.tag_item(m["item_id"], add=tuple(m["tags_add"]),
                               remove=tuple(m["tags_remove"]))
        done.append({**m, "applied": bool(moved)})
    return done


# ── the source ───────────────────────────────────────────────────────────

async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    new_id = queue.enqueue(
        source=NAME, kind="tick",
        payload={"apply": bool(src_cfg.get("apply", False)),
                 "model": str(src_cfg.get("model", DEFAULT_MODEL)),
                 "max_turns": int(src_cfg.get("max_turns", DEFAULT_MAX_TURNS)),
                 "max_events": int(src_cfg.get("max_events", DEFAULT_MAX_EVENTS)),
                 "max_items": int(src_cfg.get("max_items", DEFAULT_MAX_ITEMS))},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued board steward tick id=%d", new_id)


async def execute(item: QueueItem) -> dict[str, Any]:
    from scripts.automod import backlog as B, state as S
    from workers.sources._common import DrainActive, TurnTimeout, run_prompt_in_session

    p = item.payload or {}
    apply = bool(p.get("apply", False))
    state = _read_state()
    since_ts = float(state.get("last_ts") or 0.0)
    ledger = S.LEDGER_PATH

    open_items = await asyncio.to_thread(B.open_items, None)
    expected = await asyncio.to_thread(B.desired_statuses, ledger)
    current = {i.id: i.status for i in open_items}
    pending = {i for i, want in expected.items() if current.get(i) != want[0]}
    recent = await asyncio.to_thread(events_since, ledger, since_ts,
                                     limit=int(p.get("max_events", DEFAULT_MAX_EVENTS)))
    touched = {int(d["item_id"]) for d in recent if str(d.get("item_id") or "").isdigit()}
    shown = board_view(open_items, touched, max_items=int(p.get("max_items", DEFAULT_MAX_ITEMS)),
                       pending=pending)
    events = await asyncio.to_thread(events_since, ledger, since_ts,
                                     limit=int(p.get("max_events", DEFAULT_MAX_EVENTS)),
                                     for_items={i.id for i in shown})
    if not events and not any(i.status in ("up_next", "in_progress") for i in shown):
        return {"status": "skipped", "summary": "nothing happened since the last pass"}

    prompt = build_prompt(events=events, items=shown, n_open=len(open_items), since_ts=since_ts)
    try:
        run = await run_prompt_in_session(
            prompt, title=f"board steward ({len(events)} events)", source=NAME,
            max_turns=int(p.get("max_turns", DEFAULT_MAX_TURNS)), priority=2,
            model=str(p.get("model", DEFAULT_MODEL)),
            # A judgment, not an action: the steward reads and answers.
            extra_disallowed=["Bash", "Edit", "Write", "Task", "backlog_write_task",
                              "automod_start", "automod_gate", "automod_land",
                              "automod_abort", "automod_vault_land"],
            final_schema=STEWARD_SCHEMA,
            final_schema_prompt=("Restate your decision as one JSON object: the moves, the "
                                 "next pick, one-sentence summary. `done` is not a status "
                                 "you may set."),
        )
    except DrainActive as exc:
        return {"status": "skipped", "summary": str(exc)}
    except TurnTimeout as exc:
        return {"status": "failed", "summary": f"steward turn timed out: {exc}"}

    parsed = parse_steward(run.get("structured"))
    if parsed is None:
        return {"status": "failed", "summary": f"no structured answer: {run.get('structured_error')}",
                "meta": {"session_id": run.get("session_id")}}

    agree = agreement(parsed["moves"], expected, current)

    applied: list[dict] = []
    if apply:
        applied = await asyncio.to_thread(apply_moves, parsed["moves"])
        if parsed["next_pick"]:
            pp = _pick_path()
            pp.parent.mkdir(parents=True, exist_ok=True)
            pp.write_text(json.dumps({"item_id": parsed["next_pick"],
                                      "reason": parsed["next_pick_reason"],
                                      "ts": time.time()}), encoding="utf-8")

    newest = max((float(d.get("ts") or 0) for d in events), default=since_ts)
    _write_state({"last_ts": newest, "last_run": time.time(),
                  "last_session": run.get("session_id")})
    S.append_event({"event": "board_steward", "apply": apply, "events": len(events),
                    "items_shown": len(shown), "moves": parsed["moves"],
                    "next_pick": parsed["next_pick"],
                    "next_pick_reason": parsed["next_pick_reason"],
                    "agreement": agree, "applied": applied,
                    "session_id": run.get("session_id"), "summary": parsed["summary"]})
    verb = "applied" if apply else "proposed (dry run)"
    return {"status": "success",
            "summary": (f"{len(parsed['moves'])} move(s) {verb}, agreement "
                        f"{agree['rate']:.0%} ({len(agree['agree'])} agree, "
                        f"{len(agree['disagree'])} disagree, {len(agree['no_opinion'])} "
                        f"where the machine abstains, {len(agree['missed'])} missed); "
                        f"next pick #{parsed['next_pick'] or '—'}: {parsed['summary']}"),
            "meta": {"session_id": run.get("session_id"), "agreement": agree}}

