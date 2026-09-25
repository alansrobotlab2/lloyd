"""Which context policy rewrote this turn's history, and by how much.

Three mechanisms can rewrite a conversation between "what was persisted" and
"what the engine was sent": the turn-start stack
(`app.compaction.load_and_compact_session` — microcompact pre-pass → LLM
summarize → truncate fallback), the intra-turn pass, and the four-rung relief
ladder (`app.harness.loop._relieve_context`). Before this module, the only
record that any of them left was a `logger.info` line in `logs/server.err*`,
which rotates every ~10 MB — and the one mechanism that actually fires at
>1,000/day printed no session id at all, so the line could not be attributed
to a turn even while the log was retained. That made three questions
permanently unanswerable (#1078): which mechanism acted on a given turn, how
much it removed, and whether `compaction.mode: summarize` is dead
configuration or merely quiet. This is the per-turn record that answers them.

Two halves, deliberately separated, because they are written by different code
at different moments:

* **The rewrite itself** is booked where it happens. The relief ladder books a
  pass through `note_relief` (it is inside the harness loop, which owns
  `options.session_id`), and the turn-start decision travels on the dict
  `load_and_compact_session` already returns. Each of those sites also emits
  one `event_log` event — that is the record with a session id on it, and it is
  the only one that survives a turn which never reaches its usage row.
* **The turn's usage row** is written at the END of the turn by whoever books
  the tokens (`app/routers/messages.py` for both chat paths,
  `app/run_recorder.py` for a direct background run). None of those three
  callers owns the loop, so the ladder's passes are handed over through the
  session-keyed registry below rather than through a return value that would
  have to be threaded across three call sites that do not call each other.

The registry is per turn and per session: `start_turn` registers, the loop
finds the open turn by `options.session_id`, and the usage writer reads it back
when it inserts its row.
Turns are serialised per session by `SessionQueue`, and a second `start_turn`
for the same session replaces the first rather than merging — so the most a
crashed turn can do is leave one small object registered until that session's
next turn, which overwrites it.

NULL is load-bearing. `to_record()` returns None when this turn measured
nothing at all, and `usage_store` stores None as NULL, meaning *unmeasured*.
A mechanism that ran and removed nothing is NOT None — it is a record whose
`tokens_freed` is 0 — because a turn-start pass that ran and declined to
rewrite is the exact datum that decides the dead-configuration question, and
it must not be storable as "we have no idea".
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

#: The turn-start mechanisms, in the order the stack runs them. Kept as a
#: tuple so a projection can emit them in stack order rather than in whatever
#: order the flags happened to be true.
TURN_START_MECHANISMS = ("microcompact", "summarize", "truncate")

#: Integer fields copied off a relief report when the report carries them.
#: Absent keys stay absent: `_relieve_context` omits `passes`/`rearm` for a
#: caller that passed no latch, and a 0 written there would read as "armed at
#: zero" rather than "no latch existed".
_RELIEF_INT_FIELDS = (
    "used_before", "used_after", "target", "passes", "rearm",
    "argument_chars_freed", "truncated_chars_freed",
)


def turn_start_record(comp: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project one `load_and_compact_session` result onto the stored record.

    A missing or empty result has no record, not a record saying it declined:
    there is nothing here that measured anything, and inventing a decline is
    what makes NULL mean unmeasured. The live way to get one is a caller whose
    own `except` arm swallows a raise from this stack and continues on an
    uncompacted history — `messages.py` has no such arm today, so the value is
    that the writer's guard is there before the shape that needs it.

    `ran` is a literal `True`: this function is called only with a real result,
    and its presence in the stored record is what separates "the turn-start
    stack ran and declined" from "the turn never reached it" (no record at
    all). `mechanisms` names only the layers that changed the history; the
    summarize layer's *decision* is separate (`summarize_outcome`), because a
    turn that wanted to summarize and could not is a different fact from one
    that had nothing to summarize.
    """
    if not comp:
        return None
    before = int(comp.get("tokens_before") or 0)
    after = int(comp.get("tokens_after") or 0)
    fired = {
        "microcompact": int(comp.get("microcompacted") or 0) > 0,
        "summarize": bool(comp.get("summarized")),
        "truncate": bool(comp.get("truncated")),
    }
    return {
        "ran": True,
        "mechanisms": [m for m in TURN_START_MECHANISMS if fired[m]],
        "microcompacted": int(comp.get("microcompacted") or 0),
        "summarized": bool(comp.get("summarized")),
        "summarize_attempted": bool(comp.get("summarize_attempted")),
        "summarize_outcome": str(comp.get("summarize_outcome") or ""),
        "truncated": bool(comp.get("truncated")),
        "restored_files": int(comp.get("restored_files") or 0),
        "tokens_before": before,
        "tokens_after": after,
        "tokens_freed": max(0, before - after),
        "threshold": int(comp.get("threshold") or 0),
        "context_window": int(comp.get("context_window") or 0),
        # D2: the persisted summary. `summary_reused` is a stored record
        # applied as is (the prefix-cache win); `summary_folds` the folds this
        # turn added to it; `summary_covered_rows` how many rows it stands in
        # for. All three read false/0 with `compaction.persist_summary` off.
        "summary_reused": bool(comp.get("summary_reused")),
        "summary_folds": int(comp.get("summary_folds") or 0),
        "summary_covered_rows": int(comp.get("summary_covered_rows") or 0),
    }


def relief_record(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project one `_relieve_context` report onto the stored record.

    Only the rungs that RAN are in `report["rungs"]`, so an empty list means
    this pass was a no-op — a latched pass, or a disabled ladder — and is not
    booked at all. A pass that ran rungs and freed nothing IS booked, with
    `freed_tokens: 0`, because "the ladder fired and could not reach its
    target" is the finding; NULL is reserved for "did not look".
    """
    record: dict[str, Any] = {
        "reason": str(report.get("reason") or ""),
        "rungs": [str(r) for r in (report.get("rungs") or ())],
        "freed_tokens": int(report.get("freed_tokens") or 0),
    }
    for key in _RELIEF_INT_FIELDS:
        value = report.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            record[key] = int(value)
    return record


@dataclass
class TurnCompaction:
    """One turn's context-policy record, accumulated as the turn runs.

    Owned by the writer that books the usage row, found by the harness loop
    through `current()`. Nothing here raises: this is accounting bolted onto
    the path that answers a user, and a record that cannot be built costs a
    record, never a turn.
    """

    session_id: str = ""
    turn_id: str = ""
    turn_start: dict[str, Any] | None = None
    relief: list[dict[str, Any]] = field(default_factory=list)

    def note_turn_start(self, comp: Mapping[str, Any] | None) -> None:
        """Book the turn-start stack's decision. `None` means not reached."""
        if not comp:
            return
        try:
            self.turn_start = turn_start_record(comp)
        except Exception:  # noqa: BLE001 — accounting never breaks a turn
            self.turn_start = None

    def note_relief(self, report: Mapping[str, Any] | None) -> bool:
        """Book one relief pass; False when the pass ran no rung."""
        if not report:
            return False
        try:
            record = relief_record(report)
        except Exception:  # noqa: BLE001
            return False
        if not record["rungs"]:
            return False
        self.relief.append(record)
        return True

    @property
    def relief_tokens_freed(self) -> int:
        return int(sum(int(r.get("freed_tokens") or 0) for r in self.relief))

    @property
    def mechanisms(self) -> list[str]:
        """Every mechanism this turn saw act, turn-start order then relief.

        Relief passes are named `relief:<reason>` once per distinct reason, so
        a census of "how often does the ladder fire per reason" is a count of
        rows whose `mechanisms` contains the name, not a sum of the entries.
        """
        names: list[str] = list((self.turn_start or {}).get("mechanisms") or ())
        for report in self.relief:
            name = f"relief:{report.get('reason') or 'unknown'}"
            if name not in names:
                names.append(name)
        return names

    def to_record(self) -> dict[str, Any] | None:
        """The stored JSON body, or None when nothing was measured.

        None is not "nothing fired" — that is a record with `mechanisms: []`
        and a `turn_start` that says it ran. None means this turn's writers
        never saw either mechanism, which is what a NULL column must mean.
        """
        if self.turn_start is None and not self.relief:
            return None
        record: dict[str, Any] = {"mechanisms": self.mechanisms}
        if self.turn_start is not None:
            record["turn_start"] = self.turn_start
        if self.relief:
            record["relief"] = [dict(r) for r in self.relief]
            record["relief_passes"] = len(self.relief)
            record["relief_tokens_freed"] = self.relief_tokens_freed
        return record


# ---------------------------------------------------------------------------
# Session-keyed registry: how a pass booked inside the loop reaches the usage
# row written outside it.
# ---------------------------------------------------------------------------

#: Entries the registry may hold at once. The live count is the number of
#: sessions with a turn in flight — `SessionQueue` serialises a session's own
#: turns, and the next turn's `start_turn` replaces the previous entry — so an
#: entry outlives its turn only when that turn never wrote a usage row. The cap
#: is what bounds that leak without requiring every writer to remember a
#: teardown call: exceeding it evicts the oldest entries, and what an evicted
#: turn loses is its column, never its event.
_MAX_OPEN = 4096

_open: dict[str, TurnCompaction] = {}
_lock = threading.Lock()


def start_turn(session_id: str, turn_id: str = "") -> TurnCompaction:
    """Register the turn a usage row is about to be booked for.

    Always returns an object — a writer with no session id still gets a
    working accumulator it can hand to `record_usage` — but only a named
    session is registered, because the loop finds turns by
    `options.session_id` and an anonymous key would be claimed by every other
    anonymous turn in the process.
    """
    turn = TurnCompaction(session_id=session_id or "", turn_id=turn_id or "")
    if not turn.session_id:
        return turn
    with _lock:
        _open[turn.session_id] = turn
        while len(_open) > _MAX_OPEN:
            # Insertion order is the age order, and the oldest live entry is
            # the one whose turn is furthest behind on writing its row.
            _open.pop(next(iter(_open)))
    return turn


def current(session_id: str) -> TurnCompaction | None:
    """The open turn for this session, or None when nobody is booking one."""
    if not session_id:
        return None
    with _lock:
        return _open.get(session_id)


def note_relief(session_id: str, report: Mapping[str, Any] | None) -> bool:
    """Book a relief pass on this session's open turn. False if not booked.

    Returns False both when no turn is open (a `run_query` caller that books no
    usage row — an eval run, `post_session_capture`) and when the pass ran no
    rung. The caller has already emitted the event, which is the record that
    does not need a turn to be open; this half only feeds the usage row.
    """
    turn = current(session_id)
    if turn is None:
        return False
    return turn.note_relief(report)


__all__ = [
    "TURN_START_MECHANISMS",
    "TurnCompaction",
    "turn_start_record",
    "relief_record",
    "start_turn",
    "current",
    "note_relief",
]
