"""One definition of "record a backlog status move".

Two writers move a backlog item's `status`, and they disagreed about what a move
*costs*. The loop's (`scripts/automod/backlog.py::_apply_status`) appended an
`activity_log` line reading `old → new: why`, stamped `updated` and `completed`,
and handled the tags that ride the move. The Mission Control route
(`app/routers/backlog.py::backlog_task_update`) assigned `fm["status"]` and did
none of those. `architecture/backlog.md` states — in the activity-log section and
again in the Access rules — that every status move is attributed with its reason,
so the sentence was false for the one writer a person uses: closing a card in the
browser left no trace, no `completed:`, and left `needs-human` on an item whose
decision a human had just made.

The missing `completed:` is not cosmetic. `GET /api/backlog/tasks?done_since=`
judges a done row by `completed:` first, so a route-closed item's place in the
7-day window is whatever date last touched it — and a later priority edit bumps
`updated` and re-admits an item closed months ago, indefinitely. 135 items closed
since 2026-09-01 carry no `completed:` and this route generated them for as long
as it existed.

What is shared here is the recording: the log line, the stamps, the tag
add/remove. What is *not* is each writer's policy about which moves it accepts —
`_apply_status` refuses an item that is already there or already `done` because
`done` is terminal *for the loop*, and a human reopening a closed card from the
board must be allowed. A helper that refused both would have fixed the audit trail
by taking the board's reopen away, which is the other half of what the route is
for.

It lives in `app/` importing only the standard library (plus `app/backlog_tags`)
for the same reason `backlog_status.py` and `frontmatter.py` do:
`scripts/automod/backlog.py` is the light module the automod CLI loads, and it
must not pull `mcp` and `httpx` in behind a thirty-line helper.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from app.backlog_tags import normalize_tags

#: The stamp every backlog writer writes, and the one shape `completed:` and
#: `updated:` already appear in on the board. Naive UTC with microseconds: naive
#: because `_fm_date` compares calendar dates in the value's own zone and a mixed
#: population of offset-bearing and bare stamps is the thing it cannot judge
#: consistently, UTC because that is what the loop has always written.
_STAMP = "%Y-%m-%dT%H:%M:%S.%f"

#: The instant from which that sentence is true of *every* backlog writer (#1517).
#:
#: Until it, the board had two clocks. `app/routers/backlog.py` (both its save
#: paths), `agent_mcp/backlog.py` (`add_activity` and `_handle_write` — the
#: `backlog_write_task` path, the busiest writer on the board) and
#: `scripts/automod/backlog.py::new_item` stamped `created`/`updated` with
#: `datetime.now()`: naive **local** time. This module and the automod closers
#: stamped naive UTC. On this box (America/Los_Angeles, PDT, -0700) the two differ
#: by seven hours, so one file touched forty seconds apart can carry
#: `**2026-09-26T02:48:18**` above `**2026-09-25T19:49:05.757813**` and read as
#: though the earlier stamp came second; measured at triage, 985 of the 1,249 board
#: files with an `activity_log` hold a stamp more than an hour ahead of their own
#: `st_mtime`.
#:
#: A naive stamp below this value is therefore read in the machine's local clock
#: and a stamp at or above it in UTC — which lets the readers date a row in the
#: clock its writer used without rewriting the ~1,000 files the retired writers
#: left (backfill vs grandfather is a person's call, on #1517). The value is the
#: instant this change was authored, in naive UTC, and the direction of the old
#: error is what makes that safe: a naive-local stamp's numerals always run
#: *behind* the UTC numerals for the same instant this far west of Greenwich, so
#: every legacy row falls below the cut-off permanently and nothing the new code
#: writes can. The one case the cut-off cannot judge is a legacy writer still
#: running between this instant and this change's promotion: its rows are read as
#: UTC and look seven hours older than they are, inside a 24-hour window, until the
#: next day sweeps them past it.
LOCAL_STAMP_CUTOVER = datetime(2026, 9, 26, 4, 0)

#: The status that means the item is finished, and therefore dated.
DONE = "done"


def now_stamp() -> str:
    """The one stamp every backlog writer writes: `created:`, `updated:`,
    `completed:` and an `activity_log` line all come from here.

    Naive UTC, `LOCAL_STAMP_CUTOVER` onward. One call per write, so a move and the
    log line that narrates it cannot disagree about when they happened.
    """
    return datetime.now(timezone.utc).strftime(_STAMP)


def utc_instant(value: datetime, *, legacy_local: bool = False) -> datetime:
    """The instant a board stamp means, as an aware UTC `datetime` (#1517).

    Every reader that judges a stamp against a UTC-derived clock goes through here,
    because the store's stamps are naive and a naive stamp has no zone to read: an
    aware value is converted from its own zone; a naive one is UTC, except under
    `legacy_local` for a value *below* `LOCAL_STAMP_CUTOVER`, which was written by a
    surface that stamped the machine's local clock and is read in that zone. At or
    above the cut-off a naive stamp is UTC already and comes back with its instant
    unchanged — post-change rows are never shifted, which is the whole point of the
    cut-off.

    Callers spell the result however their comparison needs it: `.timestamp()` for
    an instant (`board_flow`'s 24-hour net) or `.replace(tzinfo=None)` for calendar
    dates in UTC (`?done_since=`). Both are right only because this returns an aware
    value — `.timestamp()` on a *naive* one would silently re-assume the local zone.

    `completed:` must not come through with `legacy_local=True`: it was written in
    UTC on both sides of the cut-off (this module, and the closers in
    `scripts/automod/backlog.py`), so reading it as local would move every closed
    item the other way instead of fixing anything.
    """
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc)
    if legacy_local and value < LOCAL_STAMP_CUTOVER:
        # `astimezone()` on a naive datetime assumes the machine's local zone —
        # exactly the reading the retired writers earned.
        return value.astimezone(timezone.utc)
    return value.replace(tzinfo=timezone.utc)


def record_status_move(
    fm: dict,
    new_status: str,
    why: str,
    *,
    add_tags: Sequence[str] = (),
    remove_tags: Sequence[str] = (),
    stamp: str | None = None,
) -> bool:
    """Write one status move onto parsed front matter. False if nothing moved.

    Mutates `fm` in place — `activity_log`, `status`, `tags`, `updated`, and
    `completed` when the move lands on `done` — and leaves the caller to persist
    it, because the two writers persist differently (the loop re-reads and
    re-writes a whole file, the route rebuilds one it already parsed).

    Returns False, having changed nothing, when `fm` already carries
    `new_status`. That is not a refusal, it is the absence of a move, and the
    distinction is load-bearing on the board: `TaskModal` posts the whole form on
    every save, status included, so a recorder that logged any posted status
    would append a `up_next → up_next` line on an ordinary title edit — and on a
    done card, re-stamp `completed` and pull an item closed months ago back into
    the board's 7-day done window.

    `completed:` is written when and only when the move lands on `done`. A reopen
    deliberately leaves the earlier one alone: the row is not `done` any more so
    no reader consults it, and clearing it would discard the only record of when
    the item was first closed.
    """
    current = fm.get("status")
    if current == new_status:
        return False

    when = stamp or now_stamp()
    log = list(fm.get("activity_log") or [])
    log.append(f"**{when}** — {current} → {new_status}: {why}")
    fm["activity_log"] = log
    fm["status"] = new_status

    # `normalize_tags`, never a bare iteration — the rule `app/backlog_tags.py`
    # exists to enforce: a `tags` field YAML handed back as a *string* that looks
    # like a list iterates one tag per character, and a writer that iterates it
    # and re-dumps the result shreds the item's real tags.
    raw: Any = fm.get("tags")
    stored = normalize_tags(raw)
    kept = [t for t in stored if t not in remove_tags]
    kept += [t for t in add_tags if t not in kept]
    # Write the field only when this move changed it or repaired its shape —
    # `update_frontmatter`'s and `close_landed`'s rule, for the same reason: an
    # item with no `tags` key must not gain an empty one from a writer that
    # merely moved it.
    if kept != stored or (raw is not None and not isinstance(raw, list)):
        fm["tags"] = kept

    fm["updated"] = when
    if new_status == DONE:
        fm["completed"] = when
    return True
