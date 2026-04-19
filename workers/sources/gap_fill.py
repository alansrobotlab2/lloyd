"""gap-fill source — resolves `label: gap` facts discovered in live sessions.

Scans ~/obsidian/facts/**/* for facts whose frontmatter contains
`label: gap` or `provenance: GAP`, dedups by fact id, and enqueues one
research item per gap. Handler asks the primary model to research the
topic (via vault tools + web if available), writes a resolution note
to pending-research/gaps/, and (at high confidence) updates the fact.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from workers.queue import WorkQueue, QueueItem
from workers.sources._common import write_staging_note, run_prompt_on_primary

logger = logging.getLogger("lloyd-workers.gap_fill")

NAME = "gap-fill"
DEFAULT_PRIORITY = 50
FACTS_ROOT = Path.home() / "obsidian" / "facts"

_MAX_ENQUEUE_PER_TICK = 5  # don't flood the queue with every gap at once


def _scan_gap_facts() -> list[dict]:
    """Find facts tagged as gaps. Returns a list of {entity, category, id, text}."""
    results: list[dict] = []
    if not FACTS_ROOT.exists():
        return results
    for entity_dir in FACTS_ROOT.iterdir():
        if not entity_dir.is_dir() or entity_dir.name.startswith("."):
            continue
        for md in entity_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8")
                parts = content.split("---\n", 2)
                if len(parts) < 3:
                    continue
                fm = yaml.safe_load(parts[1])
                if not isinstance(fm, dict):
                    continue
                for fact in fm.get("facts", []) or []:
                    if not isinstance(fact, dict):
                        continue
                    is_gap = (
                        str(fact.get("label", "")).lower() == "gap"
                        or str(fact.get("provenance", "")).upper() == "GAP"
                    )
                    if not is_gap:
                        continue
                    if fact.get("resolved_at"):
                        continue
                    results.append({
                        "entity": fm.get("entity") or entity_dir.name,
                        "category": fm.get("category", "unknown"),
                        "fact_id": fact.get("id") or "",
                        "text": fact.get("fact") or fact.get("text") or "",
                        "source_file": str(md),
                    })
            except Exception as e:
                logger.debug("Skipping %s: %s", md, e)
    return results


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    gaps = _scan_gap_facts()
    if not gaps:
        return
    enqueued = 0
    for gap in gaps[:_MAX_ENQUEUE_PER_TICK]:
        fact_id = gap["fact_id"] or re.sub(r"[^a-z0-9]+", "-", gap["text"].lower())[:40]
        dedup_key = f"gap-fill:{gap['entity']}:{fact_id}"
        new_id = queue.enqueue(
            source=NAME,
            kind="resolve_gap",
            payload=gap,
            priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
            dedup_key=dedup_key,
        )
        if new_id is not None:
            enqueued += 1
    if enqueued:
        logger.info("Enqueued %d gap-fill items (scanned %d total)", enqueued, len(gaps))


async def execute(item: QueueItem) -> dict[str, Any]:
    gap = item.payload
    entity = gap.get("entity", "unknown")
    text = gap.get("text", "")
    prompt = (
        f"You are researching a knowledge gap about entity \"{entity}\".\n\n"
        f"Gap: {text}\n\n"
        f"Use vault_recall and other available tools to find authoritative "
        f"information that resolves this gap. If you find 2+ independent sources, "
        f"summarize. If you cannot resolve, report the gap's current state and "
        f"what kind of new information would resolve it.\n\n"
        f"Return your findings in this structure:\n"
        f"## Resolution\n<1-3 paragraph summary>\n\n"
        f"## Sources\n- ref 1\n- ref 2\n\n"
        f"## Confidence\n<0.0-1.0>: <one-line justification>\n"
    )
    response = await run_prompt_on_primary(prompt, max_turns=12)
    if not response:
        response = "(no response)"

    slug = re.sub(r"[^a-z0-9]+", "-", f"{entity}-{gap.get('fact_id', '')}".lower())[:50].strip("-")
    conf = _parse_confidence(response)
    path = write_staging_note(
        source=NAME,
        slug=slug or "gap",
        body=response,
        confidence=conf,
        rationale=text[:200],
        source_refs=[gap.get("source_file", "")],
    )
    return {
        "summary": f"gap-fill {entity}: conf={conf:.2f}",
        "response": response,
        "artifact_path": str(path),
    }


def _parse_confidence(response: str) -> float:
    m = re.search(r"confidence[^0-9]*([0-1](?:\.\d+)?)", response, re.IGNORECASE)
    if not m:
        return 0.5
    try:
        val = float(m.group(1))
        return max(0.0, min(1.0, val))
    except ValueError:
        return 0.5
