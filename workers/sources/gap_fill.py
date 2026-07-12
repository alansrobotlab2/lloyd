"""gap-fill source — resolves `label: gap` facts discovered in live sessions.

Scans the live facts tree (resolved via app.paths.VAULT_FACTS_ROOT) for
facts whose frontmatter contains `label: gap` or `provenance: GAP`, dedups
by fact id, and enqueues one research item per gap. Handler asks the
primary model to research the topic (via vault tools + web if available),
writes a resolution note to pending-research/gaps/, and (at high
confidence) updates the fact.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import yaml

# Prefer the libyaml-backed C loader — ~10-20x faster than the pure-Python
# parser. The gap scan parses tens of thousands of frontmatter blocks, so
# this is the difference between a multi-minute scan and a few seconds.
try:
    from yaml import CSafeLoader as _FastYamlLoader
except ImportError:  # pragma: no cover - libyaml not built
    from yaml import SafeLoader as _FastYamlLoader

from workers.queue import WorkQueue, QueueItem
from workers.sources._common import write_staging_note, run_prompt_on_primary
from app.paths import VAULT_FACTS_ROOT as FACTS_ROOT

logger = logging.getLogger("lloyd-workers.gap_fill")

NAME = "gap-fill"
DEFAULT_PRIORITY = 50

def _scan_gap_facts(since_mtime: float = 0.0) -> tuple[list[dict], float]:
    """Find unresolved gap facts. Returns ``(gaps, highwater_mtime)``.

    Only files whose mtime is newer than ``since_mtime`` are read + YAML-parsed;
    everything else is skipped after a cheap ``stat()``. ``highwater_mtime`` is
    the newest mtime seen across the whole tree so the caller can advance its
    watermark. On the facts tree (tens of thousands of files) this turns a
    ~17s full parse into a ~0.8s stat-only walk on the common tick where
    nothing changed.

    Why the watermark is correct: a gap fact is caught on the first scan after
    its file is written (its mtime crosses the watermark), enqueued once
    (dedup_key), and when it's later resolved the file is rewritten — its mtime
    advances again, so the scan re-reads it, sees ``resolved_at``/no gap, and
    stops re-enqueuing. A file that hasn't changed since the last scan cannot
    have gained a new gap, so skipping its parse loses nothing.
    """
    results: list[dict] = []
    highwater = since_mtime
    if not FACTS_ROOT.exists():
        return results, highwater
    for entity_dir in FACTS_ROOT.iterdir():
        if not entity_dir.is_dir() or entity_dir.name.startswith("."):
            continue
        for md in entity_dir.glob("*.md"):
            try:
                m = md.stat().st_mtime
            except OSError:
                continue
            if m > highwater:
                highwater = m
            if m <= since_mtime:
                continue  # unchanged since last scan — no new gaps possible
            try:
                content = md.read_text(encoding="utf-8")
                parts = content.split("---\n", 2)
                if len(parts) < 3:
                    continue
                fm = yaml.load(parts[1], Loader=_FastYamlLoader)
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
    return results, highwater


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    # The scan walks the entire facts tree (tens of thousands of files). Even
    # the stat-only watermark pass is blocking I/O, and changed files add
    # blocking YAML parse — so run it in a worker thread and never on the
    # shared asyncio event loop (which serves all HTTP/UI). See
    # [[project_gap_fill_event_loop_freeze]]: this scan, run synchronously on
    # the loop every 5 minutes, froze the whole server for minutes.
    last_mtime = float(queue.wm_get(NAME, "last_scan_mtime") or 0.0)
    gaps, highwater = await asyncio.to_thread(_scan_gap_facts, last_mtime)

    # Enqueue every gap found in the changed files. We intentionally do NOT
    # cap per tick here: with the watermark, each tick only surfaces gaps from
    # files changed since the last scan (naturally a handful), and the queue's
    # dedup_key makes any repeat a no-op. Capping would strand gaps in already-
    # scanned files once the watermark advanced past them.
    enqueued = 0
    for gap in gaps:
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

    # Advance the watermark only after a successful scan+enqueue, so a crash
    # mid-scan re-does the work rather than skipping files.
    if highwater > last_mtime:
        queue.wm_set(NAME, "last_scan_mtime", f"{highwater:.6f}")
    if enqueued:
        logger.info("Enqueued %d gap-fill items (found %d in changed files)",
                    enqueued, len(gaps))


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
