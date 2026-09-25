"""A session's memory files, frozen at its first turn (P1).

MEMORY.md and USER.md are rendered into the system prompt near its head, and
until this existed they were re-read on every turn. So a `memory_add` in any
session — or the nightly jobs that rewrite the vault — changed the system
prompt of every open session, and each one's next turn re-prefilled its whole
conversation from the memory block down: the prefix cache holds exactly one
thing per session, and it was the thing this invalidated.

With `harness.prompt_layout.freeze_memory` on, the first turn of a session
writes the memory text it rendered to
`<SESSIONS_DIR>/<sid>.tool-results/_memory_snapshot.md` (the session's own
side-data directory, beside its spilled tool results, so an aggregator or
backend restart does not re-freeze), and every later turn renders that file
instead. What changed since is not lost: `memory_delta_note` names it in a
short `<memory_delta>` block the caller appends to the user message — the one
place a per-turn difference costs nothing to the cached prefix.

The snapshot is the same text `prompt_builder._load_memories` produces for the
session's platform (post-#464 duplicate drop, a worker's USER.md omitted), so
a frozen session's first turn is byte-identical to an unfrozen one.

Never raises: any failure falls back to the live text and no note, which is
exactly today's behaviour.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

logger = logging.getLogger("lloyd-server")

SNAPSHOT_NAME = "_memory_snapshot.md"
#: Cap on the delta note. A rewrite of the whole file is still one short
#: block; the model can Read the file for the rest.
DELTA_MAX_CHARS = 2000


def freeze_enabled() -> bool:
    try:
        from app.config import CONFIG

        return bool(((CONFIG.get("harness") or {}).get("prompt_layout") or {})
                    .get("freeze_memory", False))
    except Exception:  # noqa: BLE001
        return False


def snapshot_path(session_id: str) -> Path:
    # Read off the module at call time so a test that points SESSIONS_DIR at
    # a tmp dir is honoured.
    from app import paths

    return paths.SESSIONS_DIR / f"{session_id}.tool-results" / SNAPSHOT_NAME


def live_memories(platform: str = "") -> str:
    """What `build_system_prompt` would render as the memory body right now."""
    import prompt_builder as pb

    overlay = pb._resolve_overlay(None)
    return pb._load_memories(
        overlay, soul=pb._load_soul(overlay), files=pb._memory_files_for(platform),
    ) or ""


def memories_for_session(session_id: str, platform: str = "") -> str:
    """The session's frozen memory text; the first call takes the snapshot."""
    path = snapshot_path(session_id)
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_snapshot: unreadable %s (%s); using live", path, exc)
        return live_memories(platform)
    text = live_memories(platform)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_snapshot: could not write %s (%s)", path, exc)
    return text


def memory_delta_note(snapshot: str, live: str, max_chars: int = DELTA_MAX_CHARS) -> str:
    """A short `<memory_delta>` block naming what changed, or "" when nothing did."""
    if snapshot == live:
        return ""
    added: list[str] = []
    removed = 0
    for line in difflib.ndiff(snapshot.splitlines(), live.splitlines()):
        if line.startswith("+ "):
            added.append(line[2:])
        elif line.startswith("- "):
            removed += 1
    if not added and not removed:
        return ""
    head = (f"The memory files changed since this session started "
            f"(+{len(added)} lines, -{removed} lines). The system prompt still "
            f"shows them as they were at the session's first turn; these are "
            f"the added lines:")
    body = "\n".join(line for line in added if line.strip())
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n… (truncated; Read the memory file for the rest)"
    return "<memory_delta>\n" + head + ("\n" + body if body else "") + "\n</memory_delta>"


def frozen_memories(session_id: str, platform: str = "") -> tuple[str | None, str]:
    """`(memories_text, delta_note)` for a turn.

    `(None, "")` when freezing is off or there is no session: the caller then
    passes nothing and `build_system_prompt` reads the live files as before.
    """
    if not session_id or not freeze_enabled():
        return None, ""
    try:
        snapshot = memories_for_session(session_id, platform)
        return snapshot, memory_delta_note(snapshot, live_memories(platform))
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_snapshot: %s: %s; rendering live memory", session_id, exc)
        return None, ""
