"""Session-store input for Inner Voice and the IV grader, minus untrustworthy sessions.

Backlog #1510 follow-up (#1656). The primary sometimes records a reasoning
block describing a request that was never made — the fabricated *"reproduce my
complete previous thinking"* traces `app/thinking_fidelity.py` names. #1510
withheld those blocks from what the ENGINE sees (`app/harness/loop.py` builds
the replayed assistant message with `reasoning = ""`), and compaction needed no
filter because `app/routers/_messages_harness_adapter.py` strips the
`reasoning` field off the history it hands the harness. The leg neither of them
covers is the one this module closes: what Inner Voice and the IV grader score.
Both read the session store directly, so a flagged session's recorded
`role="thinking"` rows reached their prompts unfiltered — an invented "the user
asked me to audit my thinking" trace becomes evidence about a turn the observer
never took.

So the rule is applied where the input is assembled: **a session with at least
one flagged reasoning block contributes no messages to IV/judge input.** The
whole session, not just the flagged block, because the block is not a bad row
in an otherwise good transcript — it is a run whose recorded thinking cannot be
trusted to mean what it says, and a grader reading the other rows against it is
reading them in a frame that was never there.

The verdict is never re-derived here. Every decision calls
`app.thinking_fidelity` (`scan_file` / `scan_messages`), so the meta-session
exemption comes along for free: a session whose own user turn names the marker
words (the distiller that triaged #1510, or this item's own triage turn) is
exempt — its blocks land in `FileScan.exempt`, not `.flagged` — and it stays in
IV input. Exempting those is `scan_messages`' decision; restating the marker
regex here would create a second definition of "flagged" able to drift from the
first, which is the failure `matched_marker`'s docstring exists to prevent.

One JSON parse per call, because these reads sit on the turn's critical path
(`_session_iv_flags`' docstring is the record of what repeated parses of a
multi-megabyte session file cost).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from app import thinking_fidelity

logger = logging.getLogger("lloyd-server")


def is_flagged_session(path: Path) -> bool:
    """True when a session file holds at least one flagged reasoning block.

    The answer is `app.thinking_fidelity`'s, not a local match: `scan_file`
    applies the marker set AND the meta-session exemption, so a session that is
    *about* the defect stays in IV input and one that merely emitted a
    fabricated trace does not. An unreadable file has no flagged block to
    exclude it for, so it reads as not flagged — which is how `scan_store`
    reports it (`FileScan.unreadable`) rather than folding it into the count.
    """
    return thinking_fidelity.scan_file(path).flagged > 0


def messages_are_trustworthy(messages: list, path: Path | None = None) -> bool:
    """True when an already-parsed message list has no flagged reasoning block.

    The in-hand half of the same verdict, for a caller that parsed the file a
    moment ago and must not parse it again to ask the question.
    """
    return thinking_fidelity.scan_messages(
        path or Path("<in-memory>"), messages).flagged == 0


def read_session_field(path: Path, field: str, default: Any = None) -> Any:
    """One top-level field of a session file, as IV/judge input.

    `default` when the file is missing or malformed — what every caller here
    already returned on those — and `default` when the session is excluded,
    which is why `read_session_messages` returning ``[]`` cannot by itself tell
    "no messages" from "excluded": `is_flagged_session` answers that separately.
    """
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError):
        return default
    if not isinstance(data, dict):
        return default
    messages = data.get("messages")
    msgs = messages if isinstance(messages, list) else []
    if not messages_are_trustworthy(msgs, path):
        logger.info(
            "[iv.session] excluded session from IV input: %s "
            "(a fabricated reasoning trace is recorded in it)", path.name)
        return default
    return data.get(field, default)


def read_session_messages(path: Path) -> list[dict]:
    """One session's message list, or ``[]`` when it is not trustworthy input."""
    msgs = read_session_field(path, "messages", [])
    return msgs if isinstance(msgs, list) else []


def assemble_session_input(paths: Iterable[Path]) -> dict[str, list[dict]]:
    """Assemble IV/judge input over several session files.

    Keyed by session id (the file stem), and a session excluded for a flagged
    reasoning block contributes no entry at all — not an empty list — so a
    caller counting keys counts the sessions it actually scored.
    """
    out: dict[str, list[dict]] = {}
    for path in paths:
        messages = read_session_messages(path)
        if messages:
            out[path.stem] = messages
    return out
