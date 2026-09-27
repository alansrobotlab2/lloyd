"""autoresearch source — wraps scripts/autoresearch/run_round.py.

One round per interval_seconds. Dedup key `autoresearch:round` prevents
overlapping rounds. Payload can override targets, budget, bench_limit,
max_variants, max_parallel; the source's own config block supplies the last
three (and an optional `budget_minutes`) when the payload does not.

The round's budget is derived from the pool cap that kills it (#1546). Until
2026-09-26 the payload carried a hard-coded 60 minutes against a 1800 s cap, and
the round read neither: every round from 2026-09-24 to 09-26 was cancelled
mid-matrix, 3-4 times each, and wrote nothing but its run_spec.yaml. The round
now stops starting trials before its budget runs out and writes its report, so
the budget has to be one the pool will actually allow it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.autoresearch")

NAME = "autoresearch"
DEFAULT_PRIORITY = 60
DEDUP_KEY = "autoresearch:round"
# workers/pool.py `_DEFAULT_MAX_DURATION_SECONDS`, restated for a config block
# that names no cap: the budget must never outlive the cap the pool applies.
POOL_DEFAULT_CAP_SECONDS = 900
# Sizing knobs the config block may set for every round.
SIZING_KEYS = ("max_variants", "bench_limit", "max_parallel")


def budget_minutes_for(src_cfg: dict, requested: Any = None) -> int:
    """The round's budget: the pool cap in whole minutes, or less when the
    payload or the config block asks for less — never more."""
    cap = int(src_cfg.get("max_duration_seconds") or POOL_DEFAULT_CAP_SECONDS) // 60
    asked = [int(v) for v in (requested, src_cfg.get("budget_minutes"))
             if v is not None and str(v).strip().isdigit()]
    return max(1, min([cap, *asked]))


def _live_src_cfg() -> dict:
    """This source's config block as the pool reads it, `{}` when unreadable."""
    try:
        from workers.sources import get_sources_config
        return dict(get_sources_config().get(NAME) or {})
    except Exception:  # noqa: BLE001
        return {}


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    payload: dict[str, Any] = {"budget_minutes": budget_minutes_for(src_cfg)}
    payload.update({k: src_cfg[k] for k in SIZING_KEYS if src_cfg.get(k) is not None})
    new_id = queue.enqueue(
        source=NAME,
        kind="round",
        payload=payload,
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued autoresearch round id=%d", new_id)


async def execute(item: QueueItem) -> dict[str, Any]:
    # Lazy import — avoid loading the round orchestrator and its harness at boot.
    from scripts.autoresearch.run_round import run as run_round

    payload = item.payload or {}
    src_cfg = _live_src_cfg()
    # Clamped again here, not only at enqueue: a row queued before a config
    # change (or before this code — every row until 2026-09-26 carried 60)
    # would otherwise hand the round a budget the pool no longer allows.
    budget = budget_minutes_for(src_cfg, payload.get("budget_minutes"))
    sizing = {k: payload.get(k, src_cfg.get(k)) for k in SIZING_KEYS}
    targets = payload.get("targets")
    dry_run = bool(payload.get("dry_run", False))
    model = payload.get("model")

    result = await run_round(
        targets=targets,
        budget_minutes=budget,
        max_variants=sizing["max_variants"],
        dry_run=dry_run,
        model=model,
        bench_limit=sizing["bench_limit"],
        max_parallel=int(sizing["max_parallel"] or 3),
    )

    rid = result.get("round_id", "unknown")
    winner = result.get("winner")
    decisions = result.get("decisions", [])
    promoted = sum(1 for d in decisions if d.get("should_promote"))
    summary = (
        f"round {rid}: {len(decisions)} variants evaluated, "
        f"{promoted} promotable, winner={winner.get('variant_id') if winner else 'none'}"
    )
    if result.get("deadline_stopped"):
        summary += (f"; stopped at its {budget} min budget, "
                    f"{len(result.get('tasks_not_reached') or [])} task(s) not reached")

    return {
        "summary": summary,
        "response": json.dumps(result, default=str)[:50000],
        "artifact_path": f"_pipeline/research/rounds/{rid}.md",
    }
