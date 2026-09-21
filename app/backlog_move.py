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

#: The status that means the item is finished, and therefore dated.
DONE = "done"


def now_stamp() -> str:
    """The timestamp this module writes into `updated:` and `completed:`."""
    return datetime.now(timezone.utc).strftime(_STAMP)


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
