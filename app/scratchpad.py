"""A session-scoped, agent-authored scratchpad for long turns (backlog #1554).

What this is
------------
One file per session under the runtime data root that the model may append to and
read back, holding the prose it would otherwise only have in the KV-cached
history: what it tried, what it ruled out, what it decided and why, what it means
to do next. Prime Intellect's research harness calls the same affordance a scratch
pad and "the active memory of the model". The population it is for here is the one
that dies without ever seeing its own earlier decisions — 71 of 689 turns in the
2026-09-26 24h window ended on `max_turns`, every one of them `wrapped_up` by the
iteration anchor rather than by the model finishing, and 66 prefix misses in the
same window re-prefilled 7,648,814 tokens.

Why it is not the todo list and not `RunState`
----------------------------------------------
`TodoWrite` holds a status enum, and `app/harness/run_state.py` (#529) holds a
schema-validated Σ whose patches the harness may refuse — `RunState` is built to
*reject* free-form prose, and `tests/test_run_state.py:110-111` pins exactly that
with a `{"scratchpad": "everything I saw"}` patch. This module is the other half:
the unvalidated, uncategorised note the model keeps for itself. It validates one
thing only — that a session id cannot name a path — because a store whose
contents are graded stops being a scratchpad.

Why it never enters position 0
------------------------------
`RunOptions.system_prompt` is assembled once per turn and the loop keeps position
0 byte-identical for exactly as long as it can, because rewriting it re-prefills
the whole context (`app/harness/loop.py`, and the 0.796 prefix hit rate that rides
on it). Scratchpad bytes therefore reach the model only the way the budget anchor
does: appended as an extra message by `state_anchor`, at the iterations where the
turn is under pressure. `build_scratchpad_anchor` is that path, and
`MAX_INJECT_BYTES` is the ceiling that stops it becoming the thing it was added to
avoid — over the ceiling the oldest entries are dropped, never the newest.

Scope, and what this deliberately does not do
---------------------------------------------
Session-scoped and non-durable. A scratchpad is not memory: nothing promotes its
content into `MEMORY.md`, the vault or the fact store; the file lives with the
run's other ephemera under `DATA_ROOT`, and promotion is a separate decision
(#1554 lists it as out of scope). Each finished run reports what was written into
the sessions it used (`summarize`, called by `workers/pool.py` on the session ids
it already records), which is what turns step 1 of #1554 — does write rate per
active hour predict `success`/`failed`/`max_turns`? — into a `SELECT` over
`workers.db` instead of a new instrument. Whether the affordance earns its schema
cost is a different question, and answering it is the compute-matched A/B this
item leaves to a person with the pool paused: #731's replay is the cautionary
number, 40.498 s of prefill against 15.855 s on a state arm that also died at step
2, while prefix hit rate moved the "good" way and settled nothing.
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any

from app.deadline_anchor import ANCHOR_TAG
from app.paths import DATA_ROOT

SCRATCHPAD_DIRNAME = "scratchpad"

#: The declared ceiling on injected scratchpad bytes, in bytes, per injection.
#: A constant rather than a config key so the number that bounds the prompt has
#: exactly one home that the code and the test can both cite. Over the ceiling the
#: OLDEST entries are dropped (`_plan`), because the newest entry is the one that
#: says what the model is about to do. Sized against the pool's pressure: the 5
#: minute primary window already sits at p50 0.562 / p90 0.619 against a `kv_gate`
#: that engages at 0.6, so the injected block has to be small and rare.
MAX_INJECT_BYTES = 4096

#: One header per append, so the file is self-describing: the write count and the
#: byte total are recoverable from the file alone, which is what lets the pool
#: report a run's totals from disk after a write happened in another process.
ENTRY_HEADER = "[[scratchpad entry bytes={bytes} ts={ts}]]"
_ENTRY_RE = re.compile(
    r"^\[\[scratchpad entry bytes=(\d+) ts=([^\]\n]+)\]\]$", re.M)

#: A session id that can name a file and nothing else. `/` and `\\` are excluded
#: outright, so a caller cannot address another session's file by spelling a path,
#: and `..` has no separator to travel through. Colons stay allowed because they
#: are what the real ids are made of: `new_background_session_id`
#: (`workers/sources/_common.py:93`) mints `worker:<source>:<6hex>` for every
#: worker turn, `builtin_task` mints `task:*`, and chat ids look like
#: `20260926_154815_autocode_1e2a`. A colon is not a separator here, and rejecting
#: it would reject the population this file exists for.
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ScratchpadError(ValueError):
    """A caller asked for something the store refuses: an unnameable session."""


def scratchpad_dir() -> Path:
    """Where scratchpads live: `DATA_ROOT/scratchpad`, never inside the code tree.

    Read through `app.paths` at call time, not captured at import, so a test that
    repoints the data root and the gate that sets `LLOYD_DATA` both take effect —
    `app.paths` anchors to the git worktree it was imported from, and a module-level
    constant would freeze whichever tree happened to import it first.
    """
    return Path(DATA_ROOT) / SCRATCHPAD_DIRNAME


def safe_session_name(session_id: str) -> str:
    """The id, if it can only ever name one file of its own; else refuse."""
    if not isinstance(session_id, str) or not _SESSION_RE.match(session_id):
        raise ScratchpadError(
            f"session id {session_id!r} cannot name a scratchpad file")
    return session_id


def scratchpad_path(session_id: str) -> Path:
    return scratchpad_dir() / (safe_session_name(session_id) + ".md")


def _now() -> str:
    return datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append(session_id: str, text: str) -> dict[str, Any]:
    """Append one entry. Additive always: this module never truncates or rewrites.

    Returns the session's totals after the write, so the tool result can say how
    much is stored without a second read.
    """
    if not isinstance(text, str):
        raise ScratchpadError("scratchpad content must be text")
    safe_session_name(session_id)
    path = scratchpad_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A note that begins a line with our own header must not count as an extra
    # write: `writes` is the number step 1 of #1554 correlates against outcomes, so
    # a model-authored string that could inflate it is an accounting hole, not a
    # formatting quirk. One inserted space keeps the text readable and breaks the
    # match.
    text = text.replace("[[scratchpad entry", "[[ scratchpad entry")
    payload = text.encode("utf-8")
    header = ENTRY_HEADER.format(bytes=len(payload), ts=_now()).encode("utf-8")
    # `ab` is the whole guarantee against clobbering: O_APPEND puts the write at
    # the current end, and two writers on one session can never each start at
    # byte 0. There is deliberately no write mode that replaces the file.
    with open(path, "ab") as fh:
        fh.write(b"\n" + header + b"\n" + payload + b"\n")
    return {"session_id": session_id, "path": str(path), **stats(session_id)}


def read(session_id: str) -> str:
    """The whole file, or "" when this session has written nothing yet."""
    path = scratchpad_path(session_id)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def entries(session_id: str) -> list[tuple[int, str]]:
    """(declared_bytes, body) per entry, oldest first — headers stripped."""
    raw = read(session_id)
    if not raw:
        return []
    marks = list(_ENTRY_RE.finditer(raw))
    out: list[tuple[int, str]] = []
    for i, m in enumerate(marks):
        start = m.end() + 1
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        out.append((int(m.group(1)), raw[start:end].strip("\n")))
    return out


def stats(session_id: str) -> dict[str, int]:
    """Writes and content bytes for one session, read from the file itself.

    `bytes` sums the per-entry counts the headers declare — the content the model
    wrote, not the framing around it, which is the number worth correlating.
    """
    ent = entries(session_id)
    return {"writes": len(ent), "bytes": sum(b for b, _ in ent)}


def summarize(session_ids: list[str] | tuple[str, ...]) -> dict[str, int]:
    """Totals across the sessions one run used, as the `meta_json.scratchpad` object.

    `sessions` counts the sessions that actually have notes, so a run whose model
    never opened the tool reads `{writes: 0, bytes: 0, sessions: 0}` — a real zero,
    not a null a later reader has to guess at.

    Called by `workers/pool.py` beside the `session_ids` it already records, on
    every terminal branch — success, `pool_timeout`, exception. The session id is
    the whole protocol: the pool reads the file that id names, so a write made by
    the tool handler in the MCP process is counted exactly, with no signalling
    between the two processes and no new store to keep in step.

    Returns zeros rather than raising on a session whose file does not exist: a
    run that never used the scratchpad is a real and common observation, and step
    1 needs it distinguishable from a broken tally.
    """
    writes = 0
    bytes_ = 0
    with_a_file = 0
    for sid in session_ids:
        try:
            st = stats(sid)
        except ScratchpadError:
            continue
        writes += st["writes"]
        bytes_ += st["bytes"]
        with_a_file += 1 if st["writes"] else 0
    return {"writes": writes, "bytes": bytes_, "sessions": with_a_file}


def _plan(session_id: str, ceiling_bytes: int) -> tuple[str, int, int]:
    """Choose what fits under `ceiling_bytes`, walking back from the newest.

    Returns (text in chronological order, entries dropped, entries kept). Two
    rules, both deliberate:

    * The result is the newest **contiguous** suffix that fits. An older entry
      that would squeeze into a gap left by a bigger one is dropped anyway:
      splicing entries around a hole cuts the middle out of a decision log, and
      the middle of "I ruled out X because Y, so I am doing Z" is exactly the part
      that must not vanish.
    * Entries are kept whole, except when the newest entry alone busts the
      ceiling: then its tail is injected, truncated at the ceiling. A bound that a
      single oversized write could only ever exceed is a bound in name, and the
      tail is the part that says what comes next.

    A stopped walk means dropped entries, so `dropped` is a count the anchor can
    tell the model about rather than a shrug.
    """
    if ceiling_bytes <= 0:
        raise ScratchpadError(f"ceiling must be positive, got {ceiling_bytes}")
    ent = entries(session_id)
    if not ent:
        return "", 0, 0
    kept: list[str] = []
    used = 0
    for body in (b for _, b in reversed(ent)):
        chunk = body.encode("utf-8")
        if used + len(chunk) <= ceiling_bytes:
            kept.append(body)
            used += len(chunk)
            continue
        if not kept:                    # newest entry alone busts the ceiling
            tail = chunk[-ceiling_bytes:]
            return tail.decode("utf-8", errors="ignore"), len(ent) - 1, 1
        break                           # everything older is dropped
    return "\n".join(reversed(kept)), len(ent) - len(kept), len(kept)


def digest(session_id: str, ceiling_bytes: int = MAX_INJECT_BYTES) -> str:
    """The newest content that fits the ceiling, oldest-first; "" if nothing yet."""
    return _plan(session_id, ceiling_bytes)[0]


def dropped_entries(session_id: str, ceiling_bytes: int = MAX_INJECT_BYTES) -> int:
    """How many entries an injection at this ceiling would leave out."""
    return _plan(session_id, ceiling_bytes)[1]


def build_scratchpad_anchor(
    session_id: str,
    *,
    max_turns: int,
    warn_percentages: tuple[int, ...] = (50, 80),
    ceiling_bytes: int = MAX_INJECT_BYTES,
):
    """A `state_anchor` that injects the scratchpad only when the turn is short.

    The one path scratchpad bytes take into a prompt. It is an *appended* message
    on the iterations where the iteration clock says the budget is running out —
    same hook, same mechanism, same position in the transcript as the budget anchor
    from #1406 — and never a rewrite of position 0, which would re-prefill the
    context to deliver a note.

    Each level in `warn_percentages` fires at most once per turn, because a
    re-injected digest is the context churn this whole design is built to avoid.
    Composed with the wall-clock anchor by `compose_state_anchors`
    (`app/deadline_anchor.py`), which is how one hook carries two clocks.

    Returns `None` when there is no positive iteration cap — the same "no clock, no
    anchor" contract `build_iteration_anchor` keeps — and otherwise an anchor that
    returns `[]` on any iteration it has nothing to say on, including an empty
    scratchpad. Each emitted message carries `ANCHOR_TAG`, so the harness knows it
    wrote it and the chat transcript does not render it as a user line.
    """
    if not max_turns or max_turns <= 0:
        return None

    fired: set[int] = set()

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pct in warn_percentages:
            if pct in fired or iteration * 100 < pct * max_turns:
                continue
            fired.add(pct)                   # one injection per level, ever
            body = digest(session_id, ceiling_bytes)
            if not body:
                continue
            gone = dropped_entries(session_id, ceiling_bytes)
            note = (f"{gone} older entr{'y' if gone == 1 else 'ies'} dropped to "
                    f"hold this under {ceiling_bytes} bytes" if gone else
                    "the whole scratchpad fits under the ceiling")
            out.append({"role": "user", "content": (
                "<scratchpad>\n" + body + "\n</scratchpad>\n"
                f"<scratchpad-meta turns={iteration} of={max_turns} "
                f"level={pct}>{note}</scratchpad-meta>"),
                ANCHOR_TAG: {"kind": "scratchpad", "level": pct}})
        return out

    return anchor
