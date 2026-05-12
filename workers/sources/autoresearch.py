"""autoresearch source — wraps scripts/autoresearch/run_round.py.

One round per interval_seconds. Dedup key `autoresearch:round` prevents
overlapping rounds (rounds take 30-60 min). Payload can override targets,
budget, bench_limit, max_parallel.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.autoresearch")

NAME = "autoresearch"
DEFAULT_PRIORITY = 60
DEDUP_KEY = "autoresearch:round"


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    new_id = queue.enqueue(
        source=NAME,
        kind="round",
        payload={"budget_minutes": src_cfg.get("budget_minutes", 60)},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued autoresearch round id=%d", new_id)


async def execute(item: QueueItem) -> dict[str, Any]:
    # Lazy import — avoid loading claude_agent_sdk at boot.
    from scripts.autoresearch.run_round import run as run_round

    payload = item.payload or {}
    targets = payload.get("targets")
    budget = int(payload.get("budget_minutes", 60))
    max_variants = payload.get("max_variants")
    bench_limit = payload.get("bench_limit")
    max_parallel = int(payload.get("max_parallel", 3))
    dry_run = bool(payload.get("dry_run", False))
    model = payload.get("model")

    result = await run_round(
        targets=targets,
        budget_minutes=budget,
        max_variants=max_variants,
        dry_run=dry_run,
        model=model,
        bench_limit=bench_limit,
        max_parallel=max_parallel,
    )

    rid = result.get("round_id", "unknown")
    winner = result.get("winner")
    decisions = result.get("decisions", [])
    promoted = sum(1 for d in decisions if d.get("should_promote"))
    summary = (
        f"round {rid}: {len(decisions)} variants evaluated, "
        f"{promoted} promotable, winner={winner.get('variant_id') if winner else 'none'}"
    )

    return {
        "summary": summary,
        "response": json.dumps(result, default=str)[:50000],
        "artifact_path": f"_pipeline/research/rounds/{rid}.md",
    }
