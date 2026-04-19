"""session-distill source — mines ~/lloyd/sessions/*.json for patterns.

Uses a watermark (source='session-distill', key='last_mtime') to only process
session files modified since the last run. Each eligible session → one
enqueue item with payload.session_path. The handler asks the primary model
to identify repeated failures, unresolved questions, or gap signals, and
writes findings to pending-research/distill/.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem
from workers.sources._common import write_staging_note, run_prompt_on_primary

logger = logging.getLogger("lloyd-workers.session_distill")

NAME = "session-distill"
DEFAULT_PRIORITY = 70
SESSIONS_DIR = Path.home() / "lloyd" / "sessions"

_MAX_ENQUEUE_PER_TICK = 3


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    if not SESSIONS_DIR.exists():
        return
    last_mtime_str = queue.wm_get(NAME, "last_mtime")
    last_mtime = float(last_mtime_str) if last_mtime_str else 0.0

    candidates: list[tuple[float, Path]] = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > last_mtime:
            candidates.append((m, p))

    candidates.sort()  # oldest first
    enqueued = 0
    highwater = last_mtime
    for m, p in candidates[:_MAX_ENQUEUE_PER_TICK]:
        dedup_key = f"session-distill:{p.name}"
        new_id = queue.enqueue(
            source=NAME,
            kind="distill",
            payload={"session_path": str(p), "mtime": m},
            priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
            dedup_key=dedup_key,
        )
        if new_id is not None:
            enqueued += 1
        if m > highwater:
            highwater = m

    if highwater > last_mtime:
        queue.wm_set(NAME, "last_mtime", f"{highwater:.6f}")
    if enqueued:
        logger.info("Enqueued %d session-distill items", enqueued)


async def execute(item: QueueItem) -> dict[str, Any]:
    session_path = item.payload.get("session_path", "")
    session_name = Path(session_path).stem

    prompt = (
        f"You are analyzing a saved Lloyd session transcript to distill what we can "
        f"learn from it. The session file is at: {session_path}\n\n"
        f"Read the file using the Read tool, then identify:\n"
        f"1. Any repeated user struggles or failed assistant attempts\n"
        f"2. Knowledge gaps — questions Lloyd couldn't confidently answer\n"
        f"3. Patterns that could become a skill (if the same sequence of tools is "
        f"invoked in a reliable order)\n"
        f"4. Durable facts about the user that should be captured\n\n"
        f"Return in this structure:\n"
        f"## Struggles\n- ...\n\n## Gaps\n- ...\n\n## Skill Candidates\n- ...\n\n"
        f"## Durable Facts\n- ...\n\n## Confidence\n<0.0-1.0>: <justification>\n"
    )
    response = await run_prompt_on_primary(prompt, max_turns=15)
    if not response:
        response = "(no response)"

    conf = _parse_confidence(response)
    path = write_staging_note(
        source=NAME,
        slug=session_name[:40],
        body=response,
        confidence=conf,
        rationale=f"distilled from {session_name}",
        source_refs=[session_path],
    )
    return {
        "summary": f"distilled {session_name}",
        "response": response,
        "artifact_path": str(path),
    }


def _parse_confidence(response: str) -> float:
    m = re.search(r"confidence[^0-9]*([0-1](?:\.\d+)?)", response, re.IGNORECASE)
    if not m:
        return 0.5
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.5
