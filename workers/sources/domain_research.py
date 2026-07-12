"""domain-research source — reads a user-curated research queue.

Input: ~/obsidian/lloyd/research-queue.md — a markdown list of topics/URLs.
Each unchecked line becomes one queue item. When the handler completes,
it writes a structured note to pending-research/domain/ and marks the
line as checked in-place (so the same topic isn't re-enqueued).
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem
from workers.sources._common import write_staging_note, run_prompt_on_primary

logger = logging.getLogger("lloyd-workers.domain_research")

NAME = "domain-research"
DEFAULT_PRIORITY = 70
QUEUE_FILE = Path.home() / "obsidian" / "lloyd" / "research-queue.md"

_MAX_ENQUEUE_PER_TICK = 3
_ITEM_RE = re.compile(r"^\s*-\s+\[ \]\s+(.+?)\s*$", re.MULTILINE)


def _scan_queue_file() -> list[str]:
    """Read the research-queue file and extract unchecked topics."""
    if not QUEUE_FILE.exists():
        return []
    content = QUEUE_FILE.read_text(encoding="utf-8")
    return _ITEM_RE.findall(content)


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    # File read + regex — small today, but keep it off the event loop on
    # principle so the scheduler tick can never block HTTP/UI.
    items = await asyncio.to_thread(_scan_queue_file)
    if not items:
        return
    enqueued = 0
    for topic in items[:_MAX_ENQUEUE_PER_TICK]:
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:50].strip("-")
        dedup_key = f"domain-research:{slug}"
        new_id = queue.enqueue(
            source=NAME,
            kind="research",
            payload={"topic": topic, "slug": slug},
            priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
            dedup_key=dedup_key,
        )
        if new_id is not None:
            enqueued += 1
    if enqueued:
        logger.info("Enqueued %d domain-research items", enqueued)


async def execute(item: QueueItem) -> dict[str, Any]:
    topic = item.payload.get("topic", "")
    slug = item.payload.get("slug", "topic")

    prompt = (
        f"Research this topic and produce a structured knowledge note:\n\n"
        f"Topic: {topic}\n\n"
        f"Use vault_recall first to check if we already have relevant notes. "
        f"If the topic includes a URL, WebFetch it. Synthesize into:\n\n"
        f"## Summary\n<3-5 sentences>\n\n## Key Facts\n- fact 1\n- fact 2\n\n"
        f"## Related (vault entities)\n- ...\n\n"
        f"## Open Questions\n- ...\n\n"
        f"## Sources\n- ref 1\n- ref 2\n\n"
        f"## Confidence\n<0.0-1.0>: <justification>\n"
    )
    response = await run_prompt_on_primary(prompt, max_turns=20)
    if not response:
        response = "(no response)"

    conf = _parse_confidence(response)
    path = write_staging_note(
        source=NAME,
        slug=slug,
        body=response,
        confidence=conf,
        rationale=topic[:200],
        source_refs=[],
    )
    _mark_done_in_queue_file(topic)
    return {
        "summary": f"researched {topic[:60]}",
        "response": response,
        "artifact_path": str(path),
    }


def _mark_done_in_queue_file(topic: str) -> None:
    """Flip `- [ ] topic` to `- [x] topic` so we don't reprocess."""
    if not QUEUE_FILE.exists():
        return
    try:
        content = QUEUE_FILE.read_text(encoding="utf-8")
        escaped = re.escape(topic)
        new_content = re.sub(
            rf"^(\s*-\s+)\[ \](\s+{escaped}\s*)$",
            r"\1[x]\2",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if new_content != content:
            QUEUE_FILE.write_text(new_content, encoding="utf-8")
    except Exception as e:
        logger.warning("Could not mark topic done in queue file: %s", e)


def _parse_confidence(response: str) -> float:
    m = re.search(r"confidence[^0-9]*([0-1](?:\.\d+)?)", response, re.IGNORECASE)
    if not m:
        return 0.5
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.5
