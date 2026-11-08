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

It runs on the primary (see `DEFAULT_MODEL` for why the secondary lost), at
priority 2, every fifteen minutes. It is also the one agent-side reader of the
board's shape: `<board_health>` gives it the counts and the flow that the
eighty items it is shown cannot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem
# #3519's reader. A sibling source module, and the one import of it this gets:
# `board_disagreement` imports no host module at its own top level, so this line
# adds nothing to the `app` closure the import-surface test guards.
from workers.board_disagreement import _file_quietly

logger = logging.getLogger("lloyd-workers.board-steward")

NAME = "board-steward"
# Behind research (70) — a steward pass can wait for a research job. Not
# LONG_LIVED, so the KV gate does not hold it; it is what tells the pool which
# round to run next.
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
    "amend_clause", "reopen", "backlog_confirm_released",
}

#: The ceilings one tick's answer may carry, read ONCE by both the grammar and
#: the parser. `parse_steward` cut notes to 300 characters and summaries to 400
#: long before the schema said anything about length, so these are not new
#: limits: they name the limits the source already had, in one place, so the two
#: surfaces cannot drift (#2197). A grammar cap ABOVE a parse ceiling would be a
#: cap the parser silently contradicts — the model writes 450 characters, the
#: grammar permits it and `parse_steward` throws 50 away — so every string cap
#: here is exactly the ceiling the parse path imposes, and a reply at the caps
#: survives it unchanged.
#:
#: The numbers, from the 378 `board_steward` rows in the promotion ledger: the
#: largest legitimate reply anyone has produced is 27 moves, notes and reasons
#: saturating at 300 characters, summaries at 400, and at most 2 tags in one
#: direction in one move. `MOVES_MAX` is the one cap above a saturating
#: observation rather than at it: 40 is 13 moves over the largest real tick and
#: half the 80 items a tick is shown, and at ~350 bytes a move the worst case it
#: admits is ~14 KB — inside the 8192 tokens these caps exist to stay inside.
#: `TAG_MAX` is the one cap the item did not ask for, and it is load-bearing:
#: `_node_is_bounded` asks an array only whether its ITEMS are bounded and never
#: consults `maxItems`, so `tags_add` with an uncapped tag string left the schema
#: open-ended with its count capped (its docstring names that case:
#: `TRIAGE_VERDICT_SCHEMA`'s `acceptance_clauses`). 40 is above the longest tag
#: any steward move has ever carried — 19 characters, `review-disagreement`, over
#: the 12 distinct tags in the ledger's 378 rows.
MOVES_MAX = 40
TAGS_MAX = 6
TAG_MAX = 40
NOTE_MAX = 300
REASON_MAX = 300
SUMMARY_MAX = 400

# Every node in this schema now carries a finite cap (#2197), which is the whole
# reason for the change and not a style rule. `app.harness.finalizer` reads an
# open string or an open array as "this completion could go on forever", and so
# blames a truncated one on the budget: 18 board-steward runs between 10-01 and
# 10-04 failed with `output truncated at 8192 tokens — raise
# harness.finalizer.max_tokens` (20 failed vs 134 success over the window), and
# every one of those partial outputs is `{"moves":[],"next_pick":` followed by
# whitespace — the generation diverged inside the grammar, which no budget knob
# touches. With every node capped, `_schema_is_bounded` is True and the same
# completion is reported for what it is: `…not a budget`. This is the shape
# #1706 shipped for deep-research (`workers/sources/deep_research.py`,
# `RESULT_SCHEMA`), whose comment already records that a capped schema can still
# run to 8192 — capping makes the runaway UNGRAMMATICAL and makes the failure
# honest; it does not promise the runaway stops. That is the owed check's job.
STEWARD_SCHEMA: dict = {
    "type": "object",
    "title": "board_steward",
    "properties": {
        "moves": {"type": "array", "maxItems": MOVES_MAX, "items": {
            "type": "object",
            "properties": {
                "item_id": {"type": "integer"},
                "status": {"type": "string", "enum": list(STEWARD_STATUSES),
                           "description": ("draft: triaged and not for the loop, or not yet "
                                           "triaged. up_next: confirmed and the loop may take "
                                           "it. in_progress: a round is running on it now.")},
                "tags_add": {"type": "array", "maxItems": TAGS_MAX,
                             "items": {"type": "string", "maxLength": TAG_MAX}},
                "tags_remove": {"type": "array", "maxItems": TAGS_MAX,
                                "items": {"type": "string", "maxLength": TAG_MAX}},
                "note": {"type": "string", "maxLength": NOTE_MAX,
                         "description": "One sentence, for the item's activity log."},
            },
            "required": ["item_id", "status", "tags_add", "tags_remove", "note"],
            "additionalProperties": False,
        }, "description": (f"Only items whose status or tags should CHANGE. An item already "
                           f"where it belongs is not listed. At most {MOVES_MAX} moves.")},
        "next_pick": {"type": "integer",
                      "description": ("The up_next item autocode should implement next, or 0 "
                                      "when none is ready. Prefer a first re-offer whose branch "
                                      "still holds the work, then the oldest confirmed item.")},
        "next_pick_reason": {"type": "string", "maxLength": REASON_MAX},
        "summary": {"type": "string", "maxLength": SUMMARY_MAX,
                    "description": "One sentence on the board's state."},
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
board, and the first dry-run missed exactly that (#898). **Except a held \
confirmation:** a `confirmed` row carrying `held: true` arrived while the \
implement pool was full, and the item stays `draft` (tagged `confirmed-held`) \
until a `backlog_confirm_released` row follows it — the loop releases held \
items oldest first as rounds drain the pool. Do not move a held item to \
`up_next` yourself.
- An `umbrella` is an ordinary confirmed item: it goes to `up_next` and the \
loop implements it. Its `members` (tagged `grouped`) are the ones that stay \
`draft` — the umbrella carries them. Do not confuse the two.
- A `backlog_implement` `started` with nothing after it is a round in flight: \
`in_progress`. **A `finished` row is not the end of a round** — it is the end of \
one TURN inside it. A round is over only when its landing has settled, so while \
any state below holds the item stays `in_progress` and is NEVER proposed \
`up_next`, however final the `finished` row looks and however long ago it was: \
the common case is a turn that finished minutes ago in a round that is still \
running. A board line carrying `round_state=` is the machine telling you which \
state holds, and it is read from the board and the process table, not inferred \
from the events: `promoted` (its landing commit has neither a `settled` row nor \
a `rollback_succeeded`, so the guardian has not ruled it landed or rolled it \
back yet), `closing` (its landing commit HAS settled and this item has no \
`item_landed` row yet — the landing is done, its outcome is not yet recorded; see \
the bullet on that state below), `worktree` (the round's working tree under \
`~/lloyd-work/` is still \
open — a gate or a review retry is running in it), `gate` (a gate process of \
that round is alive), `landing` (a `land.running` marker with a live pid — the \
`promoted` row has not been written yet). A line with a `finished` row and NO \
`round_state` is a round that really is over: read its outcome. A round that \
ended with a promotion that `settled` and an outcome of `met` closes by itself; \
`not_met` / `deferred` leave it `up_next` for one more attempt on exactly the \
unmet clauses.
- **`round_flag=` is on EVERY board line, and it is the gate.** Three values, no \
others: `round_in_flight` (a `worktree`, `gate` or `landing` state holds), \
`observation_live` (`promoted` or `closing` holds — landed, and either the guardian \
has not ruled or the ruling is not yet on the item), `none` (no round of this item is \
live, so none of the holds below apply to it). While the flag is anything but `none` \
you may propose **`in_progress` and nothing else** for that item: a `up_next` or \
`draft` proposal, or a proposal to leave it where it sits, is refused by the machine \
before it reaches the board AND is still scored against you in this tick's agreement \
record, so proposing it costs the agreement and changes nothing on the board. The \
flag is projected from the same rows the \
five states above are read from, and it names no status the machine wants — only \
whether a round is live.
- **`triage=` is on EVERY board line, and it is the only thing that decides the \
`up_next` -> `draft` rule.** Four values, no others: `confirmed` (a triage turn \
wrote a `confirmed` verdict for this item and nobody has re-triaged it since, and \
the loop MAY implement it), `confirmed-held` (confirmed, and the machine is HOLDING \
it because a landed round's own clause verdict came back not `met` — it stays where \
it is until that ruling is cleared, which takes a landed round or a re-triage, not a \
board move), `confirmed-human-only` (confirmed, and either its acceptance names a \
decision the gate would really refuse, or its `hand_run:` receipt does not pass — an \
unexecuted hand-run is a command only an attended session may run, so the dispatcher \
holds the item whatever its verdict says; either way the finding stands, but the loop \
may not be the one that acts on it), `none` (no confirmed verdict the state machine \
counts: never \
triaged, or re-triaged since the confirmation). **On a `confirmed-human-only` line \
the loop may not implement the item, so `up_next` is never the proposal for it — \
propose `draft`, or propose nothing; never `done`, which would record the opposite \
of what the ruling says.** The fourth value is not a demotion and not a status: it \
is the one case where a confirmation and a possible next move coexist, and the field \
that used to call it `confirmed` was telling you to propose the move that was \
already refused. It is derived from the same ledger rows the \
machine reads, so a line and a machine that disagree about a VERDICT mean one of \
them is broken, and on a `triage=confirmed` line a suspected mis-rendering earns a \
note in `summary` or `next_pick_reason` and never a status move: for such a line the \
proposal is `up_next` or nothing, whatever its acceptance or its body says — the \
line shows a truncated body and no acceptance, and a verdict the machine read from \
the full ledger outranks what the excerpt suggests. The verdict is \
an OBSERVATION and not an instruction: `triage=confirmed` does not make an item your \
next pick, and no line ever names the status the machine wants, because this tick is \
scored against that table and printing it would be handing you the answer sheet.
- **`round_state=closing`: the landing is over, its outcome is not yet on the \
item, and the item stays `in_progress`.** The commit `settled`, the sweep has not \
yet written this item's `item_landed` row, and `desired_statuses` is holding it \
`in_progress` for exactly that interval — so propose NO move: not `up_next`, \
not `draft`. You are not the one holding it; the machine is, until `item_landed`. \
Within the window the round's outcome is still the `finished` row, and a \
`backlog_implement` turn-timeout/abort row is NOT the round's outcome: the \
worker's TURN died, which is why a `land_rescued` row exists ("gate passed at … \
and no landing; the loop started one"). #3450 at 2026-10-30T21:51:37Z did exactly \
this — its line carried no `round_state` at all (the settlement had already \
withdrawn `promoted`), its note read "Board line carries NO round_state=, so the \
round is over", and it proposed `up_next` for an item the sweep closed `done` at \
21:56:27.
- A round that aborted for a reason outside its diff (`external`, a grader \
that was down, a pre-existing red test, a rebase conflict) has NOT spent the \
item: `up_next` — but only while NO line of this item names `promoted` or \
`closing`. Either one means a LATER promotion has already overwritten the abort \
as the round's state, and then the bullet above is the rule that applies, not this \
one.
- **A review refusal on the contract spends nothing either.** A round sent \
back because the grader marked a clause `unsatisfiable` (or `post_landing`) \
was refused for a defect in the item's clauses, not in the diff; the item is \
offered again with its branch kept and the finding attached, and the next \
round amends the clause. That is `review_retry` → `up_next`. A `status_moved` \
to `draft` "spent" followed two minutes later by `up_next` "offered again — \
review_retry" is the machine correcting its own first pass, not a human \
reopen you should undo (#860, 2026-09-11 19:04 → 19:06). A round that spent its one attempt on a judgment of the \
change — a review refusal twice, `spent` — goes back to `draft` with the tag \
`needs-human`; the tag comes off with any move back into the pool.
- **`in_progress` means a round is running on it right now, and nothing \
else** (Alan's ruling, 2026-09-13). A landing that settled and left the item \
open is not running. `item_landed` with `closed=false`: acceptance `met` with \
a person still owed a check → `done` carrying `needs-human` — it is CLOSED, so \
move it nowhere; `not_met` → `up_next` (offered once more for exactly those \
clauses) unless its attempt is `spent`, then `draft` + `needs-human`; \
`deferred` to other items → `draft`, no tag; no recorded outcome → `draft` + \
`needs-human`. A promotion still under \
observation (promoted, no `settled` row and no rollback yet) is the one landed \
state that stays `in_progress` — the round is not over until the guardian says \
so, and its line says so too: `round_state=promoted`. Settling changes nothing \
about the promotion, so it also ends the marker; `item_landed` is the row to \
read once it is gone.
- `spent` is spent. An item whose one attempt was consumed by a verdict on \
the change (`spent` in an outcome; "owed-check decides", or "a human decides" \
on rows before 2026-10-05, in its status reason) goes to `draft` and stays \
there until the owed-check job reopens or closes it. \
Do not send it back to `up_next` because the outcome looks recoverable.
- A `rollback_succeeded` naming a landing's commit reopens its item: `up_next`.
- An item tagged `grouped` (folded into an umbrella) stays `draft` whatever \
its own history; the umbrella carries it.
- A human moved it by hand if the board disagrees with the ledger and the \
last event is older than the item's `updated`. Honour the human.
- **An `up_next` item whose line reads `triage=none` goes back to `draft`.** \
`up_next` means the loop may implement it, and the loop only takes items triage \
confirmed; nothing can pull an untriaged item out of there, and autotriage only \
reads `draft`. A writer that files straight into `up_next` (the first tick under \
the shared rule missed #1024 and #1025, both filed by a triage turn that way) has \
parked it where nothing will look. **`triage=` is the ONLY thing that decides this \
rule, and it is on every board line**: `triage=confirmed` or `triage=confirmed-held` \
means a triage turn confirmed this item and nobody has re-triaged it since, so \
`draft` is the WRONG move for it however parked it looks — say so in `reason` and \
propose `up_next`, or propose nothing. Do not infer a verdict from the body: the \
line prints 280 characters and a confirmed item's verdict lives in its acceptance, \
below the cut. And `triage=none` still does not order you to move it — an item whose \
tags or recent events say a human is mid-sentence stays where it is, with a note.
- Otherwise, never move an item you have no event for. Silence is not \
evidence — an `up_next` line reading `triage=none` is the one silence that is.

`next_pick`: the `up_next` item the loop should implement next. A first \
re-offer whose branch `automod/<round>` still exists is one fix cycle, not an \
hour — prefer it. Then the oldest confirmed item. Never an item with a round \
in flight, never a `grouped` member, never a line reading \
`triage=confirmed-human-only`: that value is the machine's own answer to \
whether the loop may implement the item, read from the same rows the scoring \
table is built from, and no other reading of an item — its body, its wording, \
its history — substitutes for it in either direction.

<ledger since="{since}" events="{n_events}">
{events}
</ledger>

<board_health>
{health}
</board_health>

<board items="{n_items}" of="{n_open}">
{board}
</board>

Decide with the evidence above only. **Do not open items or run tools** — \
every item you may move is shown above with its history, and a pass that \
reads its way through the board runs out of budget before it answers (the \
first live pass did exactly that). Write your decision directly; when you \
finish you will be asked to restate it as one JSON object. Lead your `summary` \
with the 24-hour net flow from `<board_health>` and name the bucket that grew.
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


#: How much of the finalizer's own `structured_error` a divergence row keeps. The
#: live `runs.summary` column is capped at 500 characters and the message ends in
#: up to 200 characters of captured completion (`app/harness/finalizer.py` files a
#: divergence with `content[:200]!r`), so 400 keeps the whole diagnosis — the
#: `diverged at 8192 tokens … not a budget` phrase and the `attempts=2` count that
#: #2370 puts in front of it — and drops only the excerpt's tail.
DIVERGENCE_ERROR_MAX = 400


def _record_divergence(*, state: dict, apply: bool, since_ts: float, newest: float,
                       events: int, items_shown: int, session_id: str | None,
                       err_text: str) -> None:
    """Publish a tick the finalizer never answered, and leave its window alone.

    #3088. `execute` used to return the failed record two lines before
    `_write_state` and `S.append_event`, so a diverged tick left nothing at all in
    the promotion ledger: the 19 divergence failures over 511 board-steward ticks
    from 2026-10-05 to 2026-10-20 produced no `board_steward` row between them —
    four sampled runs, zero rows inside any one's [started_at, completed_at] — so
    the ledger could not even show that a window had been tried. The 2026-10-07
    pair is what that blindness hid: the tick at 05:25:50 failed, the next fired
    one 900 s interval later at 05:41:24 with only 6 new events in, and its window
    was the 11 events the first tick had already been shown plus those 6. That is
    a re-presentation proven by arithmetic, and invisible to every later reader.

    The window is deliberately NOT advanced. `tests/test_board_steward.py` records
    that 6 of the 10 divergences of 2026-10-04→10-07 were recovered on the NEXT
    scheduled tick, which is exactly what re-reading an un-advanced window buys;
    moving `last_ts` here would trade a visible retry for a window the steward
    never judges at all. What the failure path gains is the record: one state
    write and one ledger row per failed tick, with `last_ts` passed through
    untouched. `since_ts` is the row's join key — two failures over one window
    carry the same value, which is what 'this window was retried' looks like.

    `S.append_event` RAISES on failure (`scripts/automod/state.py:176-177`, on
    purpose, unlike autoresearch's best-effort append), so both writes are
    swallowed here: a tick that could not be judged stays a `failed` tick whatever
    the ledger does about it, which is the guarantee #3088 clause 4 pins.
    """
    from scripts.automod import state as S
    # A streak is per window, not global: a success replaces the state dict whole
    # (`_write_state` at the foot of `execute`), so the counter only ever grows
    # while the same `since_ts` is still the one being retried.
    streak = (int(state.get("divergent_streak") or 0) + 1
              if float(state.get("divergent_since_ts") or 0.0) == since_ts else 1)
    try:
        _write_state({**state,
                      "last_run": time.time(),
                      "last_session": session_id,
                      "divergent_since_ts": since_ts,
                      "divergent_newest_ts": newest,
                      "divergent_ts": time.time(),
                      "divergent_streak": streak})
    except Exception as exc:  # noqa: BLE001 — the record is extra, the verdict is not
        logger.warning("board steward: diverged state write failed: %s", exc)
    try:
        S.append_event({"event": "board_steward", "apply": apply,
                        "diverged": True, "streak": streak,
                        "structured_error": err_text[:DIVERGENCE_ERROR_MAX],
                        "since_ts": since_ts, "newest_ts": newest,
                        "events": events, "items_shown": items_shown,
                        "moves": [], "next_pick": 0, "next_pick_reason": "",
                        "applied": [], "session_id": session_id,
                        "summary": f"tick diverged, no structured answer "
                                   f"({events} events shown)"})
    except Exception as exc:  # noqa: BLE001 — same reason: never replace the record
        logger.warning("board steward: diverged ledger append failed: %s", exc)


def _record_death(*, state: dict, apply: bool, since_ts: float, events: int,
                  items_shown: int, exc: BaseException) -> None:
    """Publish a tick whose turn never survived, and leave its window alone.

    #3321. `execute` caught `DrainActive` and `TurnTimeout` and nothing else, and
    `workers/sources/_common.py` opens its stream with `httpx.AsyncClient` /
    `client.stream` and no `except httpx.*` anywhere in that module, so a connect
    failure left `execute` entirely and was recorded only by the pool's generic arm
    (`workers/pool.py:2043`). Triage counted the cost on 2026-10-25: 13
    board-steward runs in the trailing 30 days died that way (11 `ConnectTimeout` +
    2 `ConnectError`), and ZERO `board_steward` rows fall inside any of those 13
    [started_at, completed_at] windows — while `_record_divergence` above
    demonstrably does fire on the parse path (4 `diverged: True` rows on 10-20,
    10-23 and 10-25×2). The ledger could show a tick the finalizer never answered
    and could not show a tick the network never delivered.

    This is deliberately NOT `_record_divergence` under a new label. That row
    asserts `diverged: True` with the `tick diverged, no structured answer`
    summary — a statement about the FINALIZER, which a transport death never
    reached; publishing one as the other would send the next reader to the wrong
    component. Its counters are separate for the same reason: `death_streak` and
    `divergent_streak` measure different failures over one window, and a death must
    not lengthen a divergence streak.

    `last_ts` is passed through untouched, exactly as in the divergence record.
    Here the retry is the pool's, not the next tick's: 11 of the 13 live rows are
    `attempts=2` and their second run SUCCEEDED, usually inside a minute, so
    moving the watermark would hand that retry a window the dead tick was never
    shown. #3322 is the same rule on the success path, from the other side.

    `session_id` is None because there is no `run` dict to read one out of: the
    stream never opened. Both writes are swallowed as #3088 swallows them
    (`S.append_event` RAISES on failure by design), because the caller RE-RAISES
    and a recorder's own failure must never become the exception the pool
    classifies.
    """
    from scripts.automod import state as S
    cls = type(exc).__name__
    # Per window, like the divergence streak: two deaths over one `since_ts` read
    # 1 then 2, and a death over a different window restarts at 1.
    streak = (int(state.get("death_streak") or 0) + 1
              if float(state.get("death_since_ts") or 0.0) == since_ts else 1)
    try:
        _write_state({**state,
                      "last_run": time.time(),
                      "death_since_ts": since_ts,
                      "death_ts": time.time(),
                      "death_class": cls,
                      "death_streak": streak})
    except Exception as rec_exc:  # noqa: BLE001 — the record is extra, the death is not
        logger.warning("board steward: death state write failed: %s", rec_exc)
    try:
        S.append_event({"event": "board_steward", "apply": apply,
                        "died": True, "error_class": cls,
                        # Same 400-character budget as a divergence row: one
                        # column, one cap, and the class name is in front of it.
                        "error": f"{cls}: {exc}"[:DIVERGENCE_ERROR_MAX],
                        "streak": streak,
                        "since_ts": since_ts,
                        "events": events, "items_shown": items_shown,
                        "moves": [], "next_pick": 0, "next_pick_reason": "",
                        "applied": [], "session_id": None,
                        "summary": f"tick died: {cls} ({events} events shown)"})
    except Exception as rec_exc:  # noqa: BLE001 — same reason: never mask the death
        logger.warning("board steward: death ledger append failed: %s", rec_exc)


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


#: The states that make a round unfinished: "a gate, landing, worktree or
#: observation is still live" (`items_with_unfinished_rounds`,
#: scripts/automod/backlog.py), plus `closing`, which that phrase does not name and
#: which #3602 adds (see `_settled_promotions`). A round in any of them is NOT over
#: whatever its latest `backlog_implement` row says, and that distinction is what
#: #3546 was filed on: at the 2026-10-29T17:27:13Z tick the steward read #3518's
#: `finished` row at 17:11 as the end of its round while the round's `review
#: kind=retry` fired at 17:36:16, and proposed `up_next` for #3524 while its
#: promotion (16:47:56Z, commit 752cca1a…) had not settled. Neither fact was
#: renderable from what it was shown.
#:
#: `closing` is where that same class survived #3546, because it is the one window
#: in which NONE of the four is live: the landing has already SETTLED, so no
#: observation is "still live" and `_unsettled_promotions` drops the round, while
#: the sweep has not yet recorded the landing on the item. #3450 at
#: 2026-10-30T21:51:37Z is that window — `promoted` 21:38:06 (commit 9e5324b7…),
#: `settled` 21:43:07, the steward's tick 21:51:37, `item_landed` 21:56:27 — and
#: the move note that tick wrote says the only thing the line could say: "Board
#: line carries NO round_state=, so the round is over".
ROUND_STATES = ("promoted", "closing", "worktree", "gate", "landing")

#: Ledger rows that are about a ROUND rather than an item, so they carry no
#: `item_id` and `events_since` — which joins a row to an item through the row's
#: own `item_id` (`:447-449`) — can never attach them to one. Measured over the
#: live ledger on 2026-10-29: 0 of the 400 `promoted` rows carry an `item_id`, and
#: 0 of the 381 `settled` rows carry a `round_id`, so a promotion the prompt asks
#: the steward to reason about arrived unjoinable.
ROUND_ONLY_EVENTS = ("promoted", "settled")


def _unsettled_promotions(ledger: Path) -> set[str]:
    """Rounds promoted whose landing commit has settled **neither** way.

    "Under observation" in the literal the prompt uses it: promoted, and no
    guardian verdict yet. `reverted` is the machine's own set (a rollback took the
    commit off `main`), `settled` is the guardian's verdict that the window
    closed. Both are read through the host's ledger decoder so the grounding and
    the state machine cannot disagree about what a revert is.

    The machine's `observing` set (`desired_statuses`) subtracts only the revert,
    and for a status decision it need not subtract the settlement: the sweep that
    settles a promotion closes the item in the same pass, and the branch above
    matches `item_landed` first. A board line has no such ordering to lean on —
    a `settled` row does not withdraw its `promoted` row — so a grounding that
    copied the machine's arithmetic would render every item that ever landed as
    unfinished forever, and would then be telling the steward never to offer it
    again. That is the opposite failure from #3546 and a worse one.
    """
    from scripts.automod import backlog as B
    gone = set(B._reverted_commits(ledger))
    gone |= {str(d.get("commit") or "") for d in
             B._ledger_events(ledger, "settled", require_item=False)}
    return {str(d.get("round_id") or "") for d in
            B._ledger_events(ledger, "promoted", require_item=False)
            if d.get("round_id") and str(d.get("commit") or "") not in gone}


def _settled_promotions(ledger: Path) -> dict[str, float]:
    """`{round_id: promoted_ts}` for rounds whose landing commit SETTLED and stayed.

    The settled-side complement of `_unsettled_promotions`, and this is exactly
    where #3450 fell through: at its 2026-10-30T21:51:37Z tick the promotion
    (21:38:06, commit 9e5324b7…) had settled eight minutes earlier at 21:43:07, so
    `_unsettled_promotions` had dropped the round and its `item_landed` row was
    five minutes in the future at 21:56:27. The line the steward was shown named no
    state, and its own note drew the only conclusion available.

    A rolled-back commit is deliberately NOT a closing window: the prompt's own rule
    sends a `rollback_succeeded` to `up_next`, and marking that item `closing`
    would hold it in place against a rule that is right. The revert is subtracted
    for the same reason `_unsettled_promotions` subtracts it, and the state is keyed
    on the settlement rather than on the machine's `observing` set because
    `_unsettled_promotions`' docstring rules out copying that arithmetic here.
    """
    from scripts.automod import backlog as B
    reverted = set(B._reverted_commits(ledger))
    settled = {str(d.get("commit") or "") for d in
               B._ledger_events(ledger, "settled", require_item=False)}
    out: dict[str, float] = {}
    for d in B._ledger_events(ledger, "promoted", require_item=False):
        rid, commit = str(d.get("round_id") or ""), str(d.get("commit") or "")
        if not rid or commit in reverted or commit not in settled:
            continue
        out[rid] = max(float(d.get("ts") or 0.0), 0.0)
    return out


def _item_landed_rows(ledger: Path) -> tuple[set[str], dict[int, float]]:
    """Every landing already recorded on an item, joined BOTH ways it can be found.

    `(round_ids_named, {item_id: latest_landing_ts})`. Both arms are needed, and the
    reason is the failure `_unsettled_promotions`' docstring rules out — an item
    rendered as unfinished forever after it landed, which then reads as "never
    offer this again":

    * `round_id` is the exact join. The landing sweep builds its `item_landed` row
      with `round_id` beside `item_id` and the commit (`item_landed` in
      `scripts/automod/backlog.py`'s landing branch), so a row naming this round
      settles the question for this round and no other.
    * The per-item ts is the belt for a row that lost the field. An item's OLDER
      landing must not hide a NEWER round's window — which is what a bare
      "has any `item_landed` row" test would do — so it closes the window only when
      that landing is not older than this round's promotion.

    Either arm firing closes the window, so the state always decays: it cannot
    outlive the sweep, and an item that landed is never shown as unfinished.
    """
    from scripts.automod import backlog as B
    rounds: set[str] = set()
    latest: dict[int, float] = {}
    for d in B._ledger_events(ledger, "item_landed"):
        rid = str(d.get("round_id") or "")
        if rid:
            rounds.add(rid)
        try:
            iid = int(d.get("item_id"))
        except (TypeError, ValueError):
            continue
        latest[iid] = max(latest.get(iid, 0.0), max(float(d.get("ts") or 0.0), 0.0))
    return rounds, latest


def unfinished_rounds(ledger: Path, for_items: set[int]) -> dict[int, tuple]:
    """`{item_id: (round_id, states)}` for the shown items whose round is not over.

    `states` is a tuple drawn from `ROUND_STATES`, in that order, and an item is
    in the map only if at least one holds: the line names a state that was read,
    never one that was guessed. This is the round-level fact the prompt's rule
    reasons about and which `_item_line` used to carry no field for.

    The round named is the LATEST one, except when its unfinished state belongs to
    an older round: #3524 at the filing tick was exactly that case —
    `SM_20261029_150548` promoted at 16:47:56Z, then a second round
    `SM_20261029_161851` abandoned eight seconds before the tick, which took its
    worktree with it (`abort` keeps the branch, not the tree). Naming the newest
    round with no state there would have hidden the live observation, so the
    older round is named instead.

    `closing` is the arm #3546 could not see, because it is the window in which its
    own grounding goes silent: `_unsettled_promotions` subtracts every `settled`
    commit, so a landing that settled cleanly takes the round's state away while the
    sweep that writes `item_landed` has still to run. For those minutes the item is
    `in_progress` to the state machine and the round's outcome is the landing — NOT
    the `backlog_implement` abort that ended the worker's TURN, which is what
    `land_rescued` exists for ("gate passed at … and no landing; the loop started
    one"). #3450 is that case exactly: abort 2026-10-30T21:36:40Z, `land_rescued`
    21:36:41, `promoted` 21:38:06, `settled` 21:43:07, tick 21:51:37, `item_landed`
    21:56:27.
    """
    from scripts.automod import backlog as B, state as ST
    from scripts.automod import worktree as WT
    history = B.implement_history(ledger)
    unsettled = _unsettled_promotions(ledger)
    settled_promos = _settled_promotions(ledger)
    landed_rounds, landed_ts = _item_landed_rows(ledger)

    def closing(rid: str, iid: int) -> bool:
        # Open from the settlement until the sweep records the landing on the item.
        # Either arm of `_item_landed_rows` closes it, so the state cannot outlive
        # the sweep and no landed item is ever rendered unfinished.
        if not rid or rid not in settled_promos or rid in landed_rounds:
            return False
        return landed_ts.get(iid, 0.0) < settled_promos[rid]

    out: dict[int, tuple] = {}
    for iid in for_items:
        rids = [str(r["round_id"]) for r in (history.get(iid) or []) if r.get("round_id")]
        if not rids:
            continue
        latest = rids[-1]
        try:
            wt_open = bool(WT.worktree_path(latest).exists())
        except OSError:
            wt_open = False
        states = tuple(s for s, holds in (("promoted", latest in unsettled),
                                          ("closing", closing(latest, iid)),
                                          ("worktree", wt_open),
                                          ("gate", bool(ST.gate_in_progress(latest))),
                                          ("landing", bool(ST.land_in_progress(latest))))
                       if holds)
        if not states:
            # The newest round is over. An older one may still hold a promotion
            # nobody settled, or a landing that settled and has yet to be recorded.
            older = [r for r in rids if r in unsettled]
            if not older:
                older = [r for r in rids if closing(r, iid)]
                if not older:
                    continue
                latest, states = older[-1], ("closing",)
            else:
                latest, states = older[-1], ("promoted",)
        out[iid] = (latest, states)
    return out


#: #3547: the ONE value the move gate reads. #3546 put the five states on the board
#: line for the prompt's per-state bullets, but a rule that refuses a write needs a
#: single named value, and re-deriving the five states inside the gate would be a
#: second reader of the same ledger — the two would eventually disagree about
#: whether a round is live, and only one of them is on the line the steward reads.
#:
#: The two live values are #3546's states grouped along the same seam
#: `desired_statuses` decides on: `observation_live` is its `observing` arm
#: (`scripts/automod/backlog.py`: a `backlog_implement` `phase=finished` whose
#: `round_id` has a `promoted` row no `rollback_succeeded` reverted — the guardian
#: has not ruled it), which is `promoted` plus `closing` here; `round_in_flight` is
#: its `in_flight` arm, which is `worktree` plus `gate` plus `landing`. `none` is
#: neither, and it is the only value that licenses a demotion.
#:
#: No value names a status. The flag says whether a round is live and nothing else:
#: printing what `desired_statuses` wants would move `judgement.rate`, which is the
#: measurement that decides whether `apply` flips (#3546's clause 1 rails the same
#: thing, and `test_the_recorded_side_is_not_rendered_into_the_prompt_or_the_board_line`
#: holds both).
ROUND_FLAG_VALUES = ("observation_live", "round_in_flight", "none")
_OBSERVATION_LIVE_STATES = frozenset(("promoted", "closing"))
_IN_FLIGHT_STATES = frozenset(("worktree", "gate", "landing"))


def round_flag(round_state: tuple | None) -> str:
    """One of `ROUND_FLAG_VALUES` for an item's `unfinished_rounds` entry.

    The input is the ENTRY — `(round_id, states)` or None — and not its states
    tuple: the one shape the caller already holds is the one shape this takes, so a
    caller cannot hand it the round id and read back `round_in_flight` for a
    promotion the guardian is watching.

    A projection of what is already rendered, never a second read of the ledger: the
    flag and the `round_state=` beside it are the same fact at two resolutions, so a
    gate that consulted its own copy could refuse a move the line invites.

    A state in neither named set answers `round_in_flight`, not `none`. The direction
    is deliberate: `none` is the value that licenses `up_next` and `draft`, so an
    unmapped state — a sixth one `ROUND_STATES` may later grow — must fail toward
    holding the item, never toward releasing it. An absent entry is the only honest
    `none`, because that is what `unfinished_rounds` leaves out for an item it
    checked and found finished.

    `observation_live` loses to `round_in_flight` when both hold (a promoted,
    unsettled round whose worktree is still open). Both prohibit the same moves, so
    the choice is about which one the steward is told, and a round with a live
    process is the more specific fact.
    """
    if not round_state:
        return "none"
    held = set(round_state[1])
    if held & _IN_FLIGHT_STATES or held - _OBSERVATION_LIVE_STATES:
        return "round_in_flight"
    return "observation_live"


#: The four values a board line's `triage=` field takes. None of them is a status,
#: which is the rail this field has to keep to: #3329's no-answer-sheet rule says the
#: line may carry an OBSERVATION the model can check its proposal against, and may
#: never carry the status `desired_statuses` wants. `confirmed` is the observation
#: "#3433 has a confirmed verdict on the ledger that is newer than any re-triage mark
#: on it", which is a fact about the ledger and not a destination.
#:
#: #3784 added the fourth because three were one too few: `confirmed` stood for two
#: machine branches at once. `desired_statuses` asks a confirmed verdict a second
#: question — `elif iid in confirmed and not is_human_only(...)`
#: (`scripts/automod/backlog.py:6576`) — and a confirmation whose acceptance names a
#: decision the gate would really refuse falls past that branch to
#: `("draft", "triaged confirmed; not for the unattended loop")`
#: (`scripts/automod/backlog.py:6584`), while a plain
#: one goes to `up_next`. One word covering both meant the prompt's confirmed rule
#: (`workers/sources/board_steward.py:305-308`) ordered the steward to propose
#: `up_next` for an item the machine had
#: already refused — a disagreement manufactured by the board line itself. Measured at
#: triage: 10 open items take that parked branch and 7 of them (#3119 #3359 #3446
#: #3466 #3477 #3645 #3782) read bare `triage=confirmed`, so seven more disagreement
#: rows were pre-booked against a floor the board is already under. `confirmed-held`
#: exists for the same reason — a verdict the ledger holds is a different observation
#: from a verdict with no round — and a human-only confirmation is a third.
TRIAGE_VALUES = ("confirmed", "confirmed-held", "confirmed-human-only", "none")


def _triage_map(ledger) -> dict[int, str]:
    """item_id -> the value its `triage=` field carries, for every item with a verdict.

    The point of this function is what it does NOT read. It calls the same
    `backlog.confirmed_verdicts` `desired_statuses` calls, and `held_confirmations`
    too, so the field and the machine's own branch — `elif iid in confirmed and not
    is_human_only(...)` at `scripts/automod/backlog.py:6576` — answer from
    one measurement: a verdict that `confirmed_verdicts` drops as older than the
    item's newest `backlog_retriage` mark is absent HERE as well, and the line says
    `triage=none` rather than showing a verdict the machine no longer counts. That
    agreement is the whole #3766 fix — four items (#3433 #3537 #3656 #3704) were
    moved to `draft` by a model that had never been told they were confirmed, in a
    tick where `confirmed_verdicts` had a live row for each of them.

    An item whose newest confirmed row is HELD (a landing froze its acceptance) reads
    `confirmed-held`, because that is a different machine branch: held items go to
    `draft` and are never re-triaged, so a line that said merely `confirmed` would
    invite the model to keep such an item in the implement pool — the opposite error.
    An item held but with no live confirmed row is absent: `desired_statuses` only
    consults `held` behind `iid in confirmed`
    (`scripts/automod/backlog.py:6576`), so showing held for
    it would be a second, better-informed reader — #3677's exact defect.

    A third branch reads `confirmed-human-only` (#3784), for the confirmed item whose
    acceptance names a decision the gate would really refuse — #3782's opens
    `human-only: ~/obsidian/autonomy/105-check-health-census-weekly.md:14`. That is the
    case `desired_statuses` parks at `draft` "not for the unattended loop"
    (`scripts/automod/backlog.py:6584`), so a line saying merely `confirmed` sent the
    model to the prompt's
    confirmed rule and the rule said `up_next`: the tick then scored the steward against
    a table the board line had already made unreachable. The predicate is
    `B.is_human_only` — the same call the parked branch makes, over the same
    `acceptance` text — so a marker naming only runtime data (`~/lloyd-data/**`,
    discarded via `_under_runtime_data_root`
    (`scripts/automod/backlog.py:403-411`, #3027) stays `confirmed` and
    the item remains implementable. Deliberately NOT a looser test such as "the
    acceptance mentions `human-only`": that would read a landable item as parked, and a
    false positive here costs work the loop could have done, which is the exact failure
    the 2026-10-05 ruling removed for `unverifiable`.

    A row whose `item_id` is not an integer is skipped by `confirmed_verdicts` itself;
    nothing here re-widens that.

    #3791 adds the OTHER route to that third value, because a confirmed verdict is
    refused by two questions and the field had only ever asked one. A carrier — a
    confirmed item whose `hand_run:` receipt does not pass — is refused by
    `ready_confirmed`'s own skip (#3289), and `owed.hand_run_pending` is the single
    predicate that decides it, the same one the four closers read. Before this, the
    four live carriers (#3433 #3537 #3656 #3704) each read `triage=confirmed` on a
    line the dispatcher had already refused, and the prompt's definition of that word
    ("the loop MAY implement it") is false for them: #3766 put the verdict on every
    line and the disagreement reproduced on the next two scored ticks, which is the
    proof that showing `triage=` is not sufficient unless the value itself carries
    the refusal. So the carrier reads `confirmed-human-only` too — one word for the
    one observation the steward actually needs ("the loop may not be the one that
    acts on this"), not a fifth value with a fifth rule.

    It asks the receipt question and never the item's paths, exactly as the dispatch
    skip does (#3027): an item that merely NAMES `~/lloyd-data/**` carries no block,
    is not pending, and stays `confirmed`. And the question is asked only of ids that
    are already bare `confirmed` — a held confirmation keeps `confirmed-held`, the
    branch `desired_statuses` reaches before any carrier check, so the loop's two
    surfaces still answer one question at a time and agree on the answer.
    """
    from scripts.automod import backlog as B
    from scripts.automod import owed as O
    confirmed = B.confirmed_verdicts(ledger)
    held = B.held_confirmations(ledger)
    out = {int(iid): ("confirmed-human-only"
                      if B.is_human_only((rec or {}).get("acceptance"))
                      else "confirmed")
           for iid, rec in confirmed.items()}
    # The carrier question, asked of the bare `confirmed` ids only. A `hand_run:`
    # block lives on the ITEM, not the ledger, so this is the one place the field
    # reaches outside `confirmed_verdicts`; it is a single walk of the open board,
    # and `owed._receipt_passes` runs its subprocess only for an item that really
    # carries a block, exactly as `ready_confirmed`'s skip does.
    paths = ({i.id: i.path for i in B.open_items(None)}
             if any(v == "confirmed" for v in out.values()) else {})
    for iid, value in list(out.items()):
        path = paths.get(iid) if value == "confirmed" else None
        if path is None:
            continue
        try:
            fm, _ = B._split_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if O.hand_run_pending(fm)[0]:
            out[iid] = "confirmed-human-only"
    for iid in held:
        try:
            key = int(iid)
        except (TypeError, ValueError):
            continue
        if key in out:
            out[key] = "confirmed-held"
    return out


def move_contradicts_flag(move: dict, flag: str) -> bool:
    """Whether writing `move` would contradict what the machine sees on this item.

    A non-`none` flag licenses exactly one destination: `in_progress`. Anything else
    for that item is the contradiction — a move off `in_progress` while the round is
    live, a move `-> draft`, and also a proposal that leaves the item at the status it
    is already at while the machine holds it for a promotion nobody has settled. That
    last case is what #3546 recorded for #3518 and #3524: an `up_next` proposed while
    the promotion was still under observation.

    Refusing it cannot lose a disagreement. The filer is called as
    `_file_quietly(agree, parsed["moves"], expected)` — the scored agreement and the
    PROPOSED moves — never from this partition, so the suppressed move reaches the
    ledger with the tick's own numbers whether or not it was written.

    The reverse direction is deliberately NOT gated. `-> in_progress` on a `none`
    item is the class the item's own summary names ("claiming `in_progress` with no
    round running"), and it is worth refusing, but not here: the flag is derived from
    ledger rows, so a round that has opened and not yet written its `backlog_implement`
    row reads `none`, and suppressing on the absence of a row would refuse the
    legitimate claim that the state machine itself is about to make.
    """
    if flag == "none":
        return False
    return move.get("status") != "in_progress"


def split_writable(moves: list[dict], flags: dict[int, str]) -> tuple[list[dict], list[dict]]:
    """`(writable, suppressed)` — the same move dicts, partitioned, order kept.

    An item the flag map does not name is treated as `none`: an item the machine
    never looked at is not evidence of a live round, and the map's absence is the
    same as the function answering "no round". A non-integer `item_id` cannot reach
    here — `parse_steward` coerces it and drops the move if the coercion fails.
    """
    writable: list[dict] = []
    suppressed: list[dict] = []
    for m in moves:
        bucket = suppressed if move_contradicts_flag(
            m, flags.get(int(m["item_id"]), "none")) else writable
        bucket.append(m)
    return writable, suppressed


def item_refs(ledger: Path) -> dict[str, int]:
    """Every ref a round-bearing row may name a round by → the item it belongs to.

    Keys are round ids and promoted commits, and the two namespaces cannot
    collide: a round id is `SM_` plus a UTC timestamp, a commit is 40 hex
    characters. Both are keyed because a `promoted` row names its round while a
    `settled` row names only the commit, and only a `backlog_implement` row names
    an item at all (698 of the live ledger's rows carry both a round and an item)
    — which is what makes the join possible without changing a writer.
    """
    from scripts.automod import backlog as B
    out: dict[str, int] = {}
    for d in B._ledger_events(ledger, "backlog_implement"):
        rid, iid = str(d.get("round_id") or ""), d.get("item_id")
        if rid and isinstance(iid, int):
            out[rid] = iid          # one round implements one item; any row of it says so
    for d in B._ledger_events(ledger, "promoted", require_item=False):
        iid = out.get(str(d.get("round_id") or ""))
        commit = str(d.get("commit") or "")
        if iid is not None and commit:
            out[commit] = iid
    return out


def _event_line(d: dict, *, item_refs: dict[str, int] | None = None) -> str:
    keep = ("event", "item_id", "round_id", "phase", "verdict", "acceptance", "commit",
            "reason", "detail", "closed", "from", "to", "by", "outcome")
    vals = {k: d[k] for k in keep if k in d and d[k] not in (None, "", [], {})}
    if (item_refs and str(d.get("event") or "") in ROUND_ONLY_EVENTS
            and "item_id" not in vals):
        iid = (item_refs.get(str(d.get("round_id") or ""))
               or item_refs.get(str(d.get("commit") or "")))
        if iid is not None:
            # Rendered under the same key every other row carries it under, so the
            # steward joins a promotion to a board line the way it already joins
            # every other event. It is resolved here, not read from the row: see
            # `item_refs` for why the row cannot carry it.
            vals["item_id"] = iid
    parts = []
    for k in keep:
        if k in vals:
            v = vals[k]
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


def _item_line(i: Any, *, round_state: tuple | None = None,
               triage: str = "none") -> str:
    """One board line. The bracketed status is the item's OWN status on disk and
    nothing else: what `desired_statuses` wants is deliberately not rendered here,
    because the agreement metric scores this prompt's proposals against that
    table, and printing the answer on the line would be grading the steward off
    the answer sheet. The round state is a fact about the board, not a verdict.

    `round_state` is that item's entry from `unfinished_rounds`, or None.

    #3547 changed what a None costs: `round_flag=` is on EVERY line, including an item
    with no live round, because the flag is the gate's input and a rule that reads "the
    line said nothing" is a rule that fires on a truncation. The cost is 16 characters
    on such a line (`round_flag=none` plus its separator), 28 on a line whose flag is
    `observation_live`, on at most `max_items` lines per tick — `test_a_shown_item_with_no_live_round_still_asks_its_flag`
    prints both figures rather than leaving them to this sentence. What they buy is the
    machine's own hold on the moves this item was filed over: 28 disagreeing ids in the
    seven days to 2026-10-29T17:54Z, whatever their destinations were. The
    fine-grained `round=… round_state=…` stays exactly where #3546 puts it — only on a
    line whose round is unfinished.

    #3766 gives the triage verdict the same treatment: `triage=` is on EVERY line,
    `triage=none` included, from `_triage_map` over the same `confirmed_verdicts` /
    `held_confirmations` calls `desired_statuses` reads. Before it the prompt's rule
    "an item with no triage verdict that sits in `up_next` goes back to `draft`" had no
    field to evaluate — a line carried status, name, created, tags, `round_flag=` and
    `body[:280]`, and #3433's first 280 characters carry no verdict — so the rule fired
    on four confirmed items in one tick (#3433 #3537 #3656 #3704) and trailing-7-day
    agreement fell to 0.8788 against owed-check's 0.90 floor. It names the VERDICT and
    never a status: the no-answer-sheet rail (#3329) forbids printing what
    `desired_statuses` WANTS, and "confirmed" is an observation about the ledger, not a
    destination.
    """
    body = " ".join((i.body or "").split())[:BODY_CHARS]
    tags = ",".join(i.tags or [])
    rel = [f"round_flag={round_flag(round_state)}", f"triage={triage}"]
    if i.group:
        rel.append(f"group=#{i.group}")
    if i.members:
        rel.append(f"members={len(i.members)}")
    if i.parent:
        rel.append(f"parent=#{i.parent}")
    if round_state:
        rid, states = round_state
        rel.append(f"round={rid} round_state={'+'.join(states)}")
    return (f"#{i.id} [{i.status}] {i.name[:90]} | created={i.created[:10]} "
            f"tags={tags or '-'} {' '.join(rel)}\n    {body}")


def _health_lines(h: dict | None) -> str:
    """`backlog.board_health` as the ten lines the steward reads. The board
    below is at most eighty items of ~560; these are the counts it cannot see.

    The ninth (#1210): closed items carrying `needs-human`. A landed `met` item
    now closes with that tag, so without this line the steward would watch
    `draft.needs_human` fall and conclude nobody owes anything. The tenth
    (#904): what the queue decisions produced — promotions by decider and
    their outcome, per-day counts, and retirements later reversed."""
    if not h:
        return "(unavailable)"
    d, u, f = h.get("draft") or {}, h.get("up_next") or {}, h.get("flow") or {}
    pool = h.get("implement_pool") or {}
    def fl(key: str) -> str:
        w = f.get(key) or {}
        return f"{w.get('created', 0)} created, {w.get('closed', 0)} closed, net {w.get('net', 0):+d}"
    return "\n".join([
        "open: " + ", ".join(f"{k} {v}" for k, v in sorted((h.get("open") or {}).items())),
        f"closed items carrying needs-human: {h.get('closed_needs_human', 0)}",
        (f"draft {d.get('total', 0)}: {d.get('pool', 0)} triageable, {d.get('quarantined', 0)} "
         f"quarantined self-spawns, {d.get('grouped', 0)} folded under an umbrella, "
         f"{d.get('needs_human', 0)} needs-human, {d.get('held', 0)} confirmed and held for "
         f"implement-pool room, {d.get('triaged', 0)} triaged and parked"),
        (f"up_next {u.get('total', 0)}: {u.get('umbrellas', 0)} umbrellas, {u.get('singles', 0)} "
         f"singles, {u.get('never_attempted', 0)} never attempted, {u.get('ready', 0)} ready, "
         f"{u.get('unready', 0)} not takeable"),
        f"flow 24h: {fl('24h')}",
        f"flow 7d: {fl('7d')}",
        f"loop-filed items open: {h.get('self_spawned_open', 0)}",
        f"items landed in 7 d: {h.get('landed_items_7d', 0)}",
        (f"implement pool: {pool.get('ready', 0)} ready against a bound of {pool.get('bound', 0)} "
         f"(floor {pool.get('floor', 0)}); single-item triage pauses at or above it"),
        _decisions_line(h.get("decisions")),
    ])


def _decisions_line(d: dict | None) -> str:
    """#904: what the queue decisions produced — the one line of
    `board_decisions.summary` the steward reads."""
    from scripts.automod import board_decisions as BD
    return BD.health_line(d)


def build_prompt(*, events: list[dict], items: list[Any], n_open: int, since_ts: float,
                 health: dict | None = None, round_state: dict[int, tuple] | None = None,
                 item_refs: dict[str, int] | None = None,
                 triage: dict[int, str] | None = None) -> str:
    """`round_state` / `item_refs` are `unfinished_rounds` / `item_refs` for the
    items in play (#3546); both default to nothing, which renders the prompt as it
    read before that item. They are arguments rather than reads because the cheap
    test path passes no ledger, and because the caller already holds the path.

    `triage` is #3766's map, and it defaults to a DERIVATION rather than to nothing:
    this prompt re-renders the board from `items`, so defaulting to "no map" would
    print `triage=none` beside a confirmed item on the very path `execute` takes — the
    defect the round exists to close. A caller passes a map to reuse one ledger read;
    `None` is how a cheap test path still gets a correct line."""
    since = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(since_ts)) if since_ts else "the beginning"
    rs = round_state or {}
    if triage is None:
        from scripts.automod import state as ST
        tm = _triage_map(ST.LEDGER_PATH) if items else {}
    else:
        tm = triage
    return PROMPT.format(
        since=since, n_events=len(events),
        events="\n".join(_event_line(d, item_refs=item_refs) for d in events) or "(none)",
        health=_health_lines(health),
        n_items=len(items), n_open=n_open,
        board="\n".join(_item_line(i, round_state=rs.get(i.id),
                                   triage=tm.get(i.id, "none"))
                        for i in items) or "(empty)",
    )


# ── output ───────────────────────────────────────────────────────────────

def parse_steward(structured: Any) -> dict | None:
    """Validated and clamped, or None. `done` can never come out of here.

    Every clamp is one of `STEWARD_SCHEMA`'s caps (#2197): the grammar stops a
    field at the number and so makes a truncation here something the model was
    never asked to produce, and the clamp stays because guided decoding is a
    request, not a guarantee. The two have to be one number: an answer that is
    legal at the caps has to come out of here unchanged, or the record of what
    the steward decided is not the answer it gave.
    """
    if not isinstance(structured, dict):
        return None
    moves: list[dict] = []
    for raw in (structured.get("moves") or [])[:MOVES_MAX]:
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
            "tags_add": [str(x)[:TAG_MAX] for x in (raw.get("tags_add") or []) if x][:TAGS_MAX],
            "tags_remove": [str(x)[:TAG_MAX] for x in (raw.get("tags_remove") or [])
                            if x][:TAGS_MAX],
            "note": " ".join(str(raw.get("note") or "").split())[:NOTE_MAX],
        })
    try:
        pick = int(structured.get("next_pick") or 0)
    except (TypeError, ValueError):
        pick = 0
    return {"moves": moves, "next_pick": max(0, pick),
            "next_pick_reason": " ".join(
                str(structured.get("next_pick_reason") or "").split())[:REASON_MAX],
            "summary": " ".join(str(structured.get("summary") or "").split())[:SUMMARY_MAX]}


def _as_instant(raw: Any) -> datetime | None:
    """A ledger stamp as an aware UTC datetime, or None when it is not one.

    The ledger carries two shapes and both are live: `created_at`, which
    `state.append_event` writes as an ISO-8601 string, and `ts`, which the
    writers set as an epoch float. A row whose stamp cannot be read is returned
    as None and the caller treats it as UNSTAMPED — it can neither prove nor
    disprove that the board moved — rather than guessed at: an unparsable string
    compared as a string sorts against an ISO cutoff by luck of the date prefix,
    and luck is what #3752 is here to take out of this comparison.
    """
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    # A naive stamp in this store has always meant local wall clock written as
    # UTC (#3752's own triage compares `created_at` strings against a UTC cutoff),
    # so a missing offset is read as UTC rather than shifted by the machine's.
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp


def moved_item_ids(ledger: Path | None, table_read_at: Any) -> set[int]:
    """Every item id with a ledger row stamped STRICTLY AFTER `table_read_at`.

    The set the abstention is decided from (#3752). `table_read_at` is the
    instant `execute` read `desired_statuses`, so an id here is an item whose own
    history moved between that read and now — the round that started, landed or
    was abandoned while the steward was thinking. Those are the events
    `desired_statuses` reads its answer from (its `in_flight` arm is a ledger
    read), so a row in this set is the proof that the table being scored against
    is no longer the table the steward reasoned over.

    Whole-file, like every other reader of this ledger: `state.read_events(path,
    0)` returns what is there, and a bounded read could miss the one row that
    matters and call a stale disagreement real — the failure this function
    exists to catch. A ledger that cannot be read returns the empty set, which
    leaves the tick scoring exactly as it scored before this function existed:
    the guard removes disagreements and never manufactures one, so failing open
    here cannot lift the rate, and the warning is what makes the gap visible.
    """
    cutoff = _as_instant(table_read_at)
    if ledger is None or cutoff is None:
        if table_read_at is not None and ledger is None:
            logger.warning("board-steward: no ledger path to check staleness against "
                           "(table read at %r): every disagreement scores as same-instant",
                           table_read_at)
        return set()
    try:
        from scripts.automod import state as S   # deferred, as every ledger read here is
        events = S.read_events(ledger, 0)
    except Exception as exc:  # noqa: BLE001 — see the docstring: fail open, and say so
        logger.warning("board-steward: staleness scan of %s failed (%s): every "
                       "disagreement scores as same-instant", ledger, exc)
        return set()
    moved: set[int] = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        stamp = _as_instant(ev.get("created_at")) or _as_instant(ev.get("ts"))
        if stamp is None or stamp <= cutoff:
            continue
        try:
            iid = int(ev.get("item_id"))
        except (TypeError, ValueError):
            continue            # a ledger row naming no item moved no item
        moved.add(iid)
    return moved


def agreement(moves: list[dict], expected: dict[int, tuple], current: dict[int, str],
              *, table_read_at: Any = None, ledger: Path | None = None) -> dict:
    """How far the steward's moves match the state machine's table.

    `table_read_at` and `ledger` together are the #3752 freshness guard, and they
    are keyword-only because every caller in the fleet but `execute` has no
    instant to give: the two arguments mean "this `expected` was read at
    `table_read_at`, and here is the ledger to check it against". With both
    present, an id whose own ledger rows moved after that instant is counted in
    the new `stale` list instead of `disagree` — see `moved_item_ids`. With
    either absent the counts are exactly what they were before the guard, which
    is what every existing caller and test depends on.

    `expected` is `desired_statuses(...)`: `{item_id: (status, why[, needs_human])}`
    for every item the machine has an opinion about. Three counts, kept
    separate because they mean different things: moves that agree, moves the
    machine would not make (the steward's own opinion — a human reads these),
    and machine moves the steward did not propose (misses).

    `disagree_sides` records, for each id in `disagree`, the status `expected` wanted
    and the reason it wanted it, so a row written later can name the machine's side as
    the disagreeing tick held it (#3677). It is a REPORT field and stays one: built
    after `judged` and `rate`, feeding neither, and the machine's target status stays out
    of the prompt and the board line — the rate IS the measurement of the steward's
    judgement, and rendering the answer into what the model reads would move the
    instrument that decides the flip. #3547 rules the same rail for the board line;
    `test_the_recorded_side_is_not_rendered_into_the_prompt_or_the_board_line` is what
    holds the record to it now that this block carries the answer at all.
    """
    proposed = {m["item_id"]: m["status"] for m in moves}
    agree = [i for i, s in proposed.items() if i in expected and expected[i][0] == s]
    # Two different things the first dry-run counted as one. `disagree` is
    # the machine saying A and the steward B — a real conflict. `no_opinion`
    # is an item the ledger has no view on at all — never triaged, never
    # landed, not parked by the `up_next`→`draft` rule, so `desired_statuses`
    # returns no row for it — and the steward having a view; that is not the
    # steward being wrong, it is the machine abstaining. (It used to describe
    # stranded landings parked `in_progress`; 55789b6 removed that branch —
    # `in_progress` now means a round is in flight or under observation.)
    disagree = [i for i, s in proposed.items() if i in expected and expected[i][0] != s]
    # #3752: a disagreement the board moved under is not the steward disagreeing.
    # `expected` is a snapshot with an instant (`table_read_at`), and the tick that
    # filed this item ran 146.6 s — during which #3570 got a `backlog_implement
    # phase=started` row 14 s in. Scoring that reply against the 14-seconds-stale
    # table booked it as a disagreement, and re-running `desired_statuses` on the
    # same ledger answers `in_progress` for the item, which is the steward's own
    # answer. So an id with a row of its own after the read abstains: it leaves
    # `disagree`, leaves `judged` and leaves `disagree_sides`, and lands in `stale`
    # beside them. The guard can only ever REMOVE a disagreement — an id in `agree`
    # stays in `agree` — and it moves the stale id out of `judged`, the
    # denominator, which is what an abstention is: #1688 removed an unmeasured
    # tick from that denominator the same way. It gates the INSTRUMENT and not the
    # write — with `apply` on, a stale move is still written, because the steward
    # judged the board it saw and `round_flag` is the guard on the write.
    stale = sorted(set(disagree) & moved_item_ids(ledger, table_read_at))
    disagree = [i for i in disagree if i not in set(stale)]
    # The machine's side of each disagreement, AS THIS TICK SCORED IT, into the block that
    # goes to the `board_steward` ledger event and to the run's `meta_json`. `disagree` is
    # a list of ids, and an id is only half a disagreement: the filer that turns a row into
    # something a person can rule needs the status `expected` wanted and the reason it
    # wanted it, and `expected` is alive only for the length of this call —
    # `desired_statuses` reads the board's LIVE state, so re-running it when the report is
    # written answers about that later instant and not about the tick. #3677: 12 of the 22
    # rows on that item printed `(the machine abstains)`, a sentence the two lines above
    # make impossible for an id they put in `disagree`, and 2 more printed the machine
    # agreeing with the steward.
    #
    # A list of dicts, not `{id: (status, why)}`: this block is `json.dumps`-ed into the
    # ledger line and read back, and a JSON object keyed by an int id comes back keyed by
    # the STRING "3456" — a lookup that misses silently and reads as "no record". The id is
    # therefore a VALUE here. The record is keyed by `sorted(disagree)`, the same order the
    # id list is published in, so the two line up row for row in the ledger and in
    # `meta_json` — a reader comparing them index by index compares like with like.
    disagree_sides = [{"item_id": i, "status": str(expected[i][0]),
                       "why": str(expected[i][1]) if len(expected[i]) > 1 else ""}
                      for i in sorted(disagree)]
    no_opinion = [i for i, s in proposed.items() if i not in expected]
    missed = [i for i, want in expected.items()
              if current.get(i) != want[0] and proposed.get(i) != want[0]]
    machine_moves = [i for i, want in expected.items() if current.get(i) != want[0]]
    # Rate over the decisions BOTH sides made: the machine's pending moves and
    # the steward's moves on items the machine has an opinion about.
    judged = (set(machine_moves) | {i for i in proposed if i in expected}) - set(stale)
    # No rate over an empty judged set (#1688). It used to read 1.0, and 76 of
    # 93 live ticks judged nothing, so the record the `apply` flip is judged
    # from read "agreement 100%" four ticks in five. `judged` is the count the
    # flip is gated on; the set is not recoverable from the lists above (an
    # `agree` on an item already at its wanted status is not a machine move).
    # A `stale` id is out of it too (#3752): the tick judged a board the table had
    # already stopped describing, so that decision is unmeasured rather than wrong,
    # and `judged` is the leg that says how much of the window was measured.
    return {"agree": sorted(agree), "disagree": sorted(disagree),
            "no_opinion": sorted(no_opinion), "missed": sorted(missed),
            # #3752: the abstentions, published beside the disagreements rather than
            # folded into them. A reader that sees `disagree` fall must be able to
            # tell an abstention from a steward that stopped disagreeing, so the
            # count rides in the same block the aggregate reads.
            "stale": stale,
            # beside the id lists, so the machine's half of a disagreement reaches the
            # `board_steward` event and `meta_json` with the tick's own numbers (#3677)
            "disagree_sides": disagree_sides,
            "machine_moves": len(machine_moves), "proposed": len(proposed),
            "judged": len(judged),
            "rate": (len(agree) / len(judged)) if judged else None}


def apply_moves(moves: list[dict], *, round_id: str = "", tick: str = "") -> list[dict]:
    """Write the moves through the one status writer. Never `done`.

    Every move that actually changed an item also appends one `steward_move` ledger
    row — `item_id`, the status it moved from, the status it moved to, the note and
    the `tick` that wrote it (#3467). That row is the precondition of `apply: true`,
    and not for want of a trace: the item's own activity line (`**ts** — up_next →
    draft: board steward: <note>`) records the same thing, but nothing reads front
    matter, so `desired_statuses` could not tell a move the steward WROTE from one it
    merely suggested and `reconcile_statuses` reverted the park on its next autocode
    poll,
    booking the revert as its own `status_moved` row. The tick's aggregate
    `board_steward` row cannot answer it either: its `moves` list carries PROPOSED
    moves on any tick and its `applied` list is empty while `apply: false`.

    Three deliberate edges:

    * Its own event name. `status_moved` means "the machine moved it" to
      `scripts/automod/board_decisions.py` and `scripts/automod/landed_outcomes.py`,
      which count that name into the dashboard's decision figures — a steward move
      booked there would flatter the machine on the dashboard the flip's own bar is
      read from.
    * Only a row the writer accepted. A refused move (`set_status` says the item is
      not open, or gone) and a move that changed nothing leave no row, or the reader
      would protect a status nobody wrote.
    * A STATUS claim. When the item already carries the status and only the tags go
      on (`tag_item`, the branch below), no row is written: the steward endorsed a
      status rather than moving one, and the abstention the row buys holds until a
      machine event lands — which would freeze the item at whatever status it happened
      to be at while a tag went on.

    `tick` is the session id of the steward run that wrote the move, passed by
    `execute`; it is what ties a row to the tick's own `board_steward` row and run
    record. A ledger append that raises propagates, exactly as it does for the tick
    row: a move on the board with no row is the state this change exists to prevent.
    """
    from scripts.automod import backlog as B, state as S
    done: list[dict] = []
    for m in moves:
        if m["status"] not in STEWARD_STATUSES:
            continue      # belt and braces over the schema
        before = B.item_by_id(m["item_id"])
        was = before.status if before is not None else ""
        why = f"board steward: {m['note'] or 'no note'}"
        moved_status = B.set_status(
            m["item_id"], m["status"], why,
            add_tags=tuple(m["tags_add"]), remove_tags=tuple(m["tags_remove"]))
        moved_tags = False
        if not moved_status and (m["tags_add"] or m["tags_remove"]):
            moved_tags = B.tag_item(m["item_id"], add=tuple(m["tags_add"]),
                                    remove=tuple(m["tags_remove"]))
        done.append({**m, "applied": bool(moved_status or moved_tags)})
        if moved_status:
            S.append_event({"event": "steward_move", "item_id": int(m["item_id"]),
                            "from": was, "to": m["status"],
                            "note": m["note"] or "no note", "tick": tick})
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

    # The loop's boards only (`DEFAULT_BOARDS`), which is the machine's own
    # scope: an Alfie hardware item or a purchase decision on another board
    # is not this loop's to move, and the steward was proposing exactly those
    # (#38, #40, #626, #756 — filed under `no_opinion` every tick overnight).
    open_items = await asyncio.to_thread(B.open_items, B.DEFAULT_BOARDS)
    # #3752: the instant the table was read, kept with it. `desired_statuses` is a
    # snapshot of a board that keeps moving — the pool opens a round continuously —
    # and a score needs both the answer and the instant the question was asked. A
    # live board-steward tick measured 146.6 s from `started_at` to `completed_at`,
    # and #3570's `backlog_implement phase=started` row landed 14 s into it: the
    # reply was scored against a table 14 s (and one round-start) out of date.
    table_read_at = S.now_iso()
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
    try:
        health = await asyncio.to_thread(B.board_health, ledger)
    except Exception as exc:  # noqa: BLE001 — the counts are context, not the job
        logger.warning("board_health failed: %s", exc)
        health = None
    if not events and not any(i.status in ("up_next", "in_progress") for i in shown):
        return {"status": "skipped", "summary": "nothing happened since the last pass"}

    # #3546: the round state the prompt reasons about has to be IN the prompt.
    # `expected` is computed here and thrown away — it was never shown — and both
    # disagreements the item records are items whose live round was real: legible
    # to the machine, invisible on the board line.
    round_state = await asyncio.to_thread(unfinished_rounds, ledger,
                                          {i.id for i in shown})
    refs = await asyncio.to_thread(item_refs, ledger)
    # `triage=` (#3766): derived from the same ledger path `desired_statuses` was just
    # called with, so the line the steward reads and the machine's own branch cannot
    # answer differently about the same item.
    prompt = build_prompt(events=events, items=shown, n_open=len(open_items), since_ts=since_ts,
                          health=health, round_state=round_state, item_refs=refs,
                          triage=_triage_map(ledger))
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
    except Exception as exc:  # noqa: BLE001 — record it, then let it kill the run
        # #3321: everything the two named branches do not cover — in the live
        # store always an httpx transport error out of `_common.py`'s
        # `client.stream` (11 `ConnectTimeout` + 2 `ConnectError` in the trailing
        # 30 days as triage counted). Record that the window was tried, then
        # RE-RAISE: returning a record instead would route the item to
        # `mark_completed` with no retry and no `meta.failure_kind`, when it is the
        # pool's retry that recovers these deaths (11 of the 13 live rows are
        # `attempts=2` with the second run succeeded). `asyncio.CancelledError`
        # derives from `BaseException`, so a shutdown cancel is not a death.
        _record_death(state=state, apply=apply, since_ts=since_ts,
                      events=len(events), items_shown=len(shown), exc=exc)
        raise

    parsed = parse_steward(run.get("structured"))
    if parsed is None:
        # Publish the failure and keep the window: `last_ts` is passed through
        # inside `_record_divergence`, so the next tick re-reads exactly this
        # window — the re-presentation that recovered 6 of the 10 divergences of
        # 2026-10-04→10-07 — and the ledger now shows that it is a retry (#3088).
        err = run.get("structured_error")
        _record_divergence(state=state, apply=apply, since_ts=since_ts,
                           newest=max((float(d.get("ts") or 0) for d in events),
                                      default=since_ts),
                           events=len(events), items_shown=len(shown),
                           session_id=run.get("session_id"),
                           err_text=str(err) if err is not None else "")
        return {"status": "failed", "summary": f"no structured answer: {err}",
                "meta": {"session_id": run.get("session_id")}}

    # Threaded because the guard reads the ledger to decide it (#3752): the same
    # reason the tick's every other ledger read is threaded — blocking this event
    # loop is how a steward tick starves the pool.
    agree = await asyncio.to_thread(agreement, parsed["moves"], expected, current,
                                    table_read_at=table_read_at, ledger=ledger)

    # #3547: the flag gates the WRITE and never the SCORE. `agree` above is built
    # from `parsed["moves"]` — the steward's own answer — and its `rate` is one of
    # the two legs of the `apply` flip (judged >= 50 AND rate >= 0.90). Filtering
    # before that call takes a suppressed move out of the judged set on every item
    # the machine holds without a pending move, so the same answer scores higher on a
    # smaller denominator: `tests/test_board_steward.py
    # ::test_a_move_the_flag_refuses_is_never_written_yet_still_scored` measures 1/2
    # where the steward earned 1/3, on one tick. That is #3546's triage's "manufacture
    # the flip bar's own number". So the filter is strictly after `agreement`, and a
    # suppressed move stays in the tick's disagreement exactly as proposed.
    flags = {i.id: round_flag(round_state.get(i.id)) for i in shown}
    writable, suppressed = split_writable(parsed["moves"], flags)

    applied: list[dict] = []
    if apply:
        # `tick` rides onto each move's `steward_move` row (#3467), so a reader can
        # tie an applied status to the tick that wrote it — the tick's own
        # `board_steward` row names this session too (`session_id` below).
        applied = await asyncio.to_thread(apply_moves, writable,
                                          tick=str(run.get("session_id") or ""))
        if parsed["next_pick"]:
            pp = _pick_path()
            pp.parent.mkdir(parents=True, exist_ok=True)
            pp.write_text(json.dumps({"item_id": parsed["next_pick"],
                                      "reason": parsed["next_pick_reason"],
                                      "ts": time.time()}), encoding="utf-8")

    # #3322: `events` is the `for_items`-widened list from :641-643, so it can
    # carry a shown item's history of any age while the window itself is empty. A
    # plain max over it is then BELOW `since_ts`, and this write — the only
    # `last_ts` write in the file — moves the watermark backwards: the live tick at
    # 2026-10-25T16:41:54Z left it at 03:49:23Z where it had been 16:36:32Z, which
    # #3322 measures as a 12 h 49 m rewind that re-exposed 172 already-consumed
    # events. `default=since_ts` is not a clamp: it fires only on an EMPTY list,
    # and the quiet tick's list is not empty — which is exactly the case the
    # `not events` skip above does not catch. The floor is the clamp. The
    # divergence path's `newest` at :681 is deliberately NOT clamped: it records
    # what a failed tick was shown and never writes a watermark (#3088).
    max_ts = max((float(d.get("ts") or 0) for d in events), default=since_ts)
    newest = max(max_ts, since_ts)
    _write_state({"last_ts": newest, "last_run": time.time(),
                  "last_session": run.get("session_id")})
    S.append_event({"event": "board_steward", "apply": apply, "events": len(events),
                    "items_shown": len(shown), "moves": parsed["moves"],
                    # #3547: what the flag refused, alongside the moves that were
                    # proposed anyway. The row already carries the full `moves` list
                    # and the agreement scored from it, so a reader can tell a
                    # suppressed move from an unmade one only if this is here — and
                    # in a dry run (`apply: false`, what ships today) nothing was
                    # written either way, so without it the refusal is invisible.
                    "suppressed": suppressed,
                    "next_pick": parsed["next_pick"],
                    "next_pick_reason": parsed["next_pick_reason"],
                    "agreement": agree, "applied": applied, "board_health": health,
                    "session_id": run.get("session_id"), "summary": parsed["summary"]})
    # #3519: the disagreement signal gets its reader. Deliberately the LAST thing
    # the tick does — after the watermark (`_write_state`) and after this tick's
    # ledger row, so the tick is fully recorded whatever the filing does — and its
    # return value is discarded for the same reason: it cannot reach `status`,
    # `summary` or `meta` below, so it cannot change the outcome the agreement
    # record is built from. `apply` was read at the top of the tick and is not
    # re-read here, which is why filing a question cannot widen what the steward
    # writes. Runs in a thread: it reads the queue and may write a backlog item,
    # and blocking this event loop is how a steward tick starves the pool.
    await asyncio.to_thread(_file_quietly, agree, parsed["moves"], expected)
    verb = "applied" if apply else "proposed (dry run)"
    # The steward's own sentence leads, and the counts follow it (#1606). This
    # string is capped at 500 characters on its way into `runs.summary`, the
    # sentence is up to 400 of them (`parse_steward`) and the counts are ~110,
    # so half of all ticks overflow — and what fell off the end was the tail of
    # the board state, the only outcome in the record: 77 of the 154
    # board-steward rows stop at exactly 500 characters, 63 of them mid-word
    # ("…remain under guardian observati"). The counts are the corroborating
    # detail and the machine-derived half; read the judgment first and let the
    # counts be the part that truncates.
    # A tick that judged nothing says so rather than printing a percentage
    # (#1688): `rate` is None there, and "100%" over zero decisions is what
    # made an empty run of ticks read as agreement.
    rate = (f"{agree['rate']:.0%} of {agree['judged']} judged"
            if agree["rate"] is not None else "not measured, 0 judged")
    # The move the steward PROPOSED is the count, so it stays `len(parsed["moves"])`
    # and never `len(writable)` — the sentence describes the answer it graded, and
    # the refusals are named beside it rather than subtracted out of it. ~30 extra
    # characters, and only on a tick that refused something, which is the case whose
    # outcome would otherwise be lost to the 500-character cap this string carries.
    refused = (f", {len(suppressed)} refused by round_flag" if suppressed else "")
    # Only on a tick that abstained (#3752): the count costs ~45 characters on the
    # one tick in which it means anything, and this string is capped at 500 on its
    # way into `runs.summary` with 77 of 154 rows already stopping at exactly 500.
    abstained = (f", {len(agree['stale'])} abstained (the board moved mid-tick)"
                 if agree["stale"] else "")
    counts = (f"{len(parsed['moves'])} move(s) {verb}{refused}{abstained}, agreement "
              f"{rate} ({len(agree['agree'])} agree, "
              f"{len(agree['disagree'])} disagree, {len(agree['no_opinion'])} "
              f"where the machine abstains, {len(agree['missed'])} missed); "
              f"next pick #{parsed['next_pick'] or '—'}")
    return {"status": "success",
            "summary": (f"{parsed['summary']} — {counts}"
                        if parsed["summary"] else counts),
            "meta": {"session_id": run.get("session_id"), "agreement": agree}}

