#!/usr/bin/env python3
"""Lloyd MCP Server: recall_observation (#1481).

When `compaction.microcompact.observation_stubs` is on, a tool result the
harness clears from context leaves an *observation stub*: an id, the call
that produced it, and a bounded verbatim head. This tool is the other half —
``recall_observation(id)`` returns the full text the stub stands for.

It is lossless by construction rather than by care: the content was written
to ``sessions/<session_id>.tool-results/<id>.{txt,json}`` before the stub
replaced it (`app.harness.tool_result_spill.persist_for_compaction`, or the
live spill / transcript pointer), and the id *is* that file's stem.

Two rules make it safe to hand a model:

* **Only the calling session's own spill directory.** The session is the one
  the aggregator bound for this call (`_shared.get_bound_session`), never an
  argument, so an id from another session — or a path dressed up as an id —
  resolves to nothing and is refused. The resolved file must sit directly in
  ``<SESSIONS_DIR>/<session_id>.tool-results/``.
* **Read-only.** Listed in `annotations.READ_ONLY`; it opens one file and
  returns its text.

The harness advertises it only to a turn whose relief writes stubs
(`loop._open_turn`), so with the switch off the catalog is unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from mcp.types import Tool

from agent_mcp._shared import get_bound_session, text_result
from app.harness.tool_result_spill import (
    RECALL_OBSERVATION_TOOL,
    session_record_paths,
)

logger = logging.getLogger("lloyd-recall-observation")

#: Upper bound on one answer, in chars. A larger file is returned in pages
#: (`offset`); the harness spills anything over 50k anyway, so this only
#: bounds what a single call can put on the wire.
MAX_CHARS = 200_000

_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")


def _refuse(msg: str) -> str:
    return json.dumps({"error": msg})


def resolve(obs_id: str, session_id: str) -> Path | None:
    """The file behind ``obs_id`` in ``session_id``'s spill dir, or None.

    An id is a spill file's stem. Anything that is not a plain stem — a
    separator, ``..``, an empty string — is not an id and resolves to None,
    as does a stem with no file behind it in this session's directory.
    """
    if not session_id or not isinstance(obs_id, str):
        return None
    if not _ID_RE.match(obs_id) or obs_id in (".", "..") or "/" in obs_id:
        return None
    spill = session_record_paths(session_id)[1]
    try:
        root = spill.resolve()
    except OSError:
        return None
    for ext in ("txt", "json"):
        cand = spill / f"{obs_id}.{ext}"
        try:
            real = cand.resolve()
        except OSError:
            continue
        if real.parent != root or not real.is_file():
            continue
        return real
    return None


def recall(obs_id: str, session_id: str, offset: int = 0) -> str:
    """The text for ``obs_id`` in ``session_id``, or a JSON refusal."""
    if not session_id:
        return _refuse("recall_observation called outside a session context")
    path = resolve(obs_id, session_id)
    if path is None:
        return _refuse(
            f"no observation {obs_id!r} in this session — an id is the one a "
            "cleared-result stub of THIS session names")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return _refuse(f"observation {obs_id!r} could not be read: {e}")
    off = max(0, int(offset or 0))
    chunk = text[off:off + MAX_CHARS]
    if off + MAX_CHARS < len(text):
        chunk += (f"\n…[{len(text) - off - MAX_CHARS:,} more chars; call again "
                  f"with offset={off + MAX_CHARS}]")
    return chunk


_DESC = """Return the full text of a tool result that was cleared from context.

When a long turn runs short of room, older tool results are replaced by an observation stub:

    [observation <id> — <tool> <args> — N chars cleared from context. Its first K chars, verbatim: ...]

Call this with that `<id>` to get the whole result back, exactly as the tool returned it. Only ids from this session's own stubs resolve. Read-only."""


def tool() -> Tool:
    """The one Tool this module serves (sync, so an eval can take its schema)."""
    return Tool(
        name=RECALL_OBSERVATION_TOOL,
        description=_DESC,
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The id an observation stub names.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset for a result over "
                                   f"{MAX_CHARS:,} chars (default 0).",
                },
            },
            "required": ["id"],
        },
    )


async def list_tools():
    return [tool()]


async def call_tool(name: str, arguments: dict[str, Any]):
    if name != RECALL_OBSERVATION_TOOL:
        return text_result(_refuse(f"Unknown tool: {name}"))
    args = arguments or {}
    sid = get_bound_session() or ""
    obs_id = str(args.get("id") or "")
    out = recall(obs_id, sid, int(args.get("offset") or 0))
    # Explicit: a recalled result may itself be a JSON error object, and the
    # sniffing default would report a successful recall as a failed call.
    refused = resolve(obs_id, sid) is None
    return text_result(out, is_error=refused)
