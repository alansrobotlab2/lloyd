"""bench-mine source — generates new bench tasks from failure signal.

Inputs:
- Ledger losers at ~/lloyd/_pipeline/research/ledger.jsonl — bench tasks
  that baseline scored poorly on (room to grow).
- Recent failed autonomy runs under ~/lloyd/autonomy-runs/**/run_*.md
  with status=failed.

Produces: YAML-frontmatter bench task files under
~/obsidian/pending-research/bench/{yyyy-mm-dd}/. A human review step
moves the best ones into ~/obsidian/lloyd/bench/ for actual evaluation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem
from workers.sources._common import write_staging_note, run_prompt_on_primary

logger = logging.getLogger("lloyd-workers.bench_mine")

NAME = "bench-mine"
DEFAULT_PRIORITY = 80

LEDGER_PATH = Path.home() / "lloyd" / "_pipeline" / "research" / "ledger.jsonl"
AUTONOMY_RUNS_DIR = Path.home() / "lloyd" / "autonomy-runs"


def _recent_ledger_losers(days: int = 7, limit: int = 5) -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    try:
        with LEDGER_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                created = row.get("created_at", "")
                try:
                    if created.endswith("Z"):
                        created = created[:-1] + "+00:00"
                    dt = datetime.fromisoformat(created)
                except Exception:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    continue
                if not row.get("variant_id", "").startswith("baseline"):
                    continue
                if (row.get("composite_score") or 1.0) < 0.6:
                    rows.append(row)
    except Exception as e:
        logger.warning("Ledger read error: %s", e)
    rows.sort(key=lambda r: r.get("composite_score") or 1.0)
    return rows[:limit]


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    losers = _recent_ledger_losers()
    if not losers:
        return
    enqueued = 0
    for row in losers:
        task_id = row.get("task_id", "")
        dedup_key = f"bench-mine:{task_id}:{row.get('round_id','')}"
        new_id = queue.enqueue(
            source=NAME,
            kind="mine",
            payload={"loser_task_id": task_id, "composite_score": row.get("composite_score"),
                     "round_id": row.get("round_id")},
            priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
            dedup_key=dedup_key,
        )
        if new_id is not None:
            enqueued += 1
    if enqueued:
        logger.info("Enqueued %d bench-mine items", enqueued)


async def execute(item: QueueItem) -> dict[str, Any]:
    payload = item.payload
    loser = payload.get("loser_task_id", "unknown")
    score = payload.get("composite_score")

    prompt = (
        f"You are designing a new evaluation task for Lloyd's bench. "
        f"Baseline scored {score} on task `{loser}`, which indicates a weak spot.\n\n"
        f"Read the existing task at ~/obsidian/lloyd/bench/{loser}.md to understand "
        f"the category and objective. Then design ONE new related bench task that "
        f"stresses the same weakness from a slightly different angle.\n\n"
        f"Output ONLY the full markdown file content with YAML frontmatter:\n\n"
        f"```\n"
        f"---\n"
        f"id: bench_XXX_<slug>\n"
        f"category: <replay|synthetic|adversarial|safety>\n"
        f"objective: <one-line>\n"
        f"max_tool_calls: <int>\n"
        f"rubric:\n"
        f"  - <criterion 1>\n"
        f"  - <criterion 2>\n"
        f"---\n\n"
        f"## Prompt\n<the prompt to send to Lloyd>\n\n"
        f"## Expected Behavior\n<what success looks like>\n"
        f"```\n"
    )
    response = await run_prompt_on_primary(prompt, max_turns=8)
    if not response:
        response = "(no response)"

    slug = re.sub(r"[^a-z0-9]+", "-", f"mined-from-{loser}".lower())[:50].strip("-")
    path = write_staging_note(
        source=NAME,
        slug=slug,
        body=response,
        confidence=0.5,
        rationale=f"derived from baseline loss on {loser} (score={score})",
        source_refs=[f"~/obsidian/lloyd/bench/{loser}.md"],
    )
    return {
        "summary": f"bench-mine from {loser}",
        "response": response,
        "artifact_path": str(path),
    }
