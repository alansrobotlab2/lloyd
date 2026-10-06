"""Where the per-spoken-turn latency rows live, and how one gets appended.

The kept half of #2273. A spoken turn used to be measured and then dropped:
`agent-services/voice/timeline.py::TurnTimeline` builds the twelve-mark breakdown,
`livekit_worker.py` logged it once as a `[latency]` line, and the only consumer of that
line was `scripts/voice/e2e_voice.py` grepping it out of a synthetic rig's log. So every
latency number this project has quoted was rig-sampled, and production regressions came
to us as incidents. One JSON object per spoken reply, appended when the reply has
finished playing, is what turns the same measurement into a population a percentile can
be taken over — read by `scripts/maintenance/voice_turn_trend.py`, and bounded by store
15 of `scripts/groundskeeper/retention-sweep.py`.

This module owns the LOCATION, the ROW's identity half and the APPEND. The measurement
half is the timeline's own `TurnTimeline.as_dict()` (`agent-services/voice/timeline.py`),
which `turn_row()` embeds beside the fields a timeline cannot know — which turn it was,
which room, and what happened to it. `retention-sweep.py` imports the path out of here
rather than restating it. One owner for a store and its location is the shape
`app/ww_diag.py` already has for the wake-miss corpus, and #1444 is what happens when a
store does not have one: that corpus's directory moved to a home-relative path outside
the data root while its readers and its bounds stayed here, so the store escaped both
the delete guard and the snapshot timer.

The row (schema `v`, `SCHEMA_VERSION`):

    v                                   schema version, so a later reader can tell
    turn_id, room                       the join key. `turn_id` is the backend's own
                                        (`app/routers/voice.py` mints it into the
                                        `voice_turn` head, which is what the session's
                                        turn record carries beside it), `room` the
                                        LiveKit room the turn was spoken in
    at, epoch                           the SAME instant twice: an ISO string ending `Z`
                                        for a human, an epoch float for arithmetic and
                                        sorting. The monotonic marks inside a timeline
                                        cannot be joined to a session record — see
                                        `timeline.py`'s own #2213 note on two clocks —
                                        so the row carries wall clock, and a naive local
                                        timestamp would be read as UTC by every later
                                        machine reader (#2213), hence the explicit `Z`
    label                               `voice` for a microphone turn, `typed:user` for
                                        a typed one spoken aloud. They have different
                                        stage sets, so the trend splits on it
    interrupted, queued_behind, tools_ran
                                        the three flags a turn's shape turns on: this
                                        reply was cut short, it ran behind another, a
                                        tool ran during it
    spoken_chars, capped                how much was spoken, and whether it hit the
                                        length cap
    stages / eos_to_audio / max_gap_s   the timeline's own `as_dict()`; `at` and `epoch`
                                        come back inside it, since it is the half that
                                        knows the instant

`append_turn` never raises: a row is a measurement OF a turn, never a participant in
one, and the worker's contract is that a spoken turn finishes being spoken whatever the
disk does. Nothing propagates to the caller, and nothing is raised into an asyncio task
nobody is awaiting either — that becomes an unretrieved exception, which is a failure
that is visible only in a log nobody reads.

Not here, on purpose: the trend (that is a script, run nightly and read-only), the
prune (that is the retention sweep's, and it reaches the file through `turns_path()`
rather than naming it), and any per-turn `TurnTimeline` caching — the row is one line
built at the end of a turn, and a reader gets the whole file by opening it.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

# Same stdlib-only import route as `app/ww_diag.py`: the LiveKit unit imports this
# module from its own process with the repo root on `sys.path`, and `app.paths` would
# drag in the whole package for one constant.
_TREE = Path(__file__).resolve().parents[1]
if str(_TREE) not in sys.path:
    sys.path.insert(0, str(_TREE))
try:
    from app.data_root import resolve_data_root_for_tree
except ModuleNotFoundError:  # pragma: no cover - same fallback as `ww_diag`
    from data_root import resolve_data_root_for_tree

LOG = logging.getLogger("lloyd-voice-turns")

#: Where the rows live, relative to the data root. Under `voice/` beside the audio
#: diagnostics, never under the checkout: this is runtime data (#1444 and the standing
#: data-root rule), and the retention sweep bounds it by this exact path.
TURNS_RELATIVE = Path("voice") / "turns.jsonl"

#: Bump on a layout change rather reinterpreting an old file. `v` rides on each row.
SCHEMA_VERSION = 1


def data_root() -> Path:
    """The one data root, resolved by `app.data_root`'s three rules."""
    return resolve_data_root_for_tree(_TREE)


def turns_path(root: Optional[Path] = None) -> Path:
    """`<data root>/voice/turns.jsonl`. A test passes `root`; production never does."""
    return (root if root is not None else data_root()) / TURNS_RELATIVE


def turn_row(timeline, *, turn_id: str = "", room: str = "", interrupted: bool = False,
             queued_behind: bool = False, tools_ran: bool = False,
             spoken_chars: int = 0, capped: bool = False,
             epoch: Optional[float] = None) -> dict:
    """The whole row for one completed spoken turn: `timeline.as_dict()` plus the identity.

    The measurement half is the timeline's own record — this function adds nothing to it
    and renames nothing in it, which is what lets `tests/test_voice_duplex.py` pin the
    row against the real class rather than against a copy of its arithmetic. The identity
    fields come from the worker's own state: the `turn_id` from the backend's
    `voice_turn` head, the room, and the three flags. `TurnTimeline` stays ignorant of
    them — it knows stages, and not which turn it was.
    """
    return {"v": SCHEMA_VERSION, "turn_id": turn_id, "room": room,
            "interrupted": bool(interrupted), "queued_behind": bool(queued_behind),
            "tools_ran": bool(tools_ran), "spoken_chars": int(spoken_chars),
            "capped": bool(capped), **timeline.as_dict(epoch=epoch)}


def append_turn(row: dict, *, path: Optional[Path] = None) -> bool:
    """Append `row` as one JSON line. True on success, False on any failure.

    Every part of this swallow is the point of the function. A read-only mount, a full
    disk, a `voice` that is somehow a file and not a directory: none of them may reach
    the worker's reply path, where the consequence would be a turn that stops being
    spoken because a log was inconvenient. The WARNING is the observability half — the
    row is lost, and one line saying so is what distinguishes "the store is empty" from
    "nothing has been able to write to it since Tuesday".
    """
    target = path or turns_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        return True
    except (OSError, TypeError, ValueError) as exc:
        LOG.warning("voice turn row not recorded at %s: %s", target, exc)
        return False


def is_row(value: object) -> bool:
    """Whether a parsed JSON value has the shape this store writes.

    The gate the readers need: `[1,2,3]` and `"hello"` are valid JSON and would otherwise
    raise `AttributeError`/`TypeError` a line later, from inside the loop that is
    supposed to be counting them. `epoch` is not required — a line truncated mid-append
    is exactly the shape that has lost its tail, and the age guard has an ISO fallback.
    """
    return isinstance(value, dict) and "label" in value and "stages" in value


def read_rows(path: Optional[Path] = None) -> Iterator[tuple[int, Optional[dict]]]:
    """Yield `(line_number, row_or_None)` for every line in the store.

    A malformed line yields `(n, None)` — it is counted, not skipped silently. With the
    store filling at a fraction of a row a day the malformed count is normally 0, and the
    thing worth knowing is when it isn't; a reader that crashed on one bad line would
    take the whole nightly trend down for it.
    """
    target = path or turns_path()
    if not target.is_file():
        return
    with open(target, encoding="utf-8", errors="replace") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield number, None
                continue
            yield number, value if is_row(value) else None
