"""Nightly clustering of the open backlog — the deterministic half of group triage.

`scripts/automod/cluster.py` does the work; this source is the schedule.
It runs off the event loop (a numpy pass over qmd's stored vectors plus a
few pair-judge calls to the secondary) and writes `clusters.json` in the
automod state dir, which `autotriage`'s group mode consumes. No session: a
clustering pass is arithmetic, not a judgement anyone needs to review.

"Nightly" is expressed as the age of the last output rather than a wall-clock
hour: the pool polls on `interval_seconds` from process start, so an hourly
poll that runs only when `clusters.json` is older than `min_age_seconds`
lands once a day whatever the restart cadence, and never twice.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.backlog-cluster")

NAME = "backlog-cluster"
# Below the research stream (70) and above nothing: a night's clustering
# can wait behind a research job, and it is not held by the KV gate.
DEFAULT_PRIORITY = 65
LONG_LIVED = False
DEDUP_KEY = "backlog-cluster:nightly"
DEFAULT_MIN_AGE_SECONDS = 20 * 3600
# A used-up `clusters.json` is rebuilt early, but no more often than this.
DEFAULT_EXHAUSTED_MIN_AGE_SECONDS = 2 * 3600


def _clusters_age_seconds() -> float | None:
    from scripts.automod import cluster as CL
    try:
        return time.time() - CL.CLUSTERS_PATH.stat().st_mtime
    except OSError:
        return None


def _clusters_exhausted(min_size: int, max_size: int) -> bool:
    """True when group triage has nothing left to take from `clusters.json`:
    every cluster in it is judged, closed, or below the minimum. Reads the
    board and the ledger, so the caller runs it off the event loop."""
    from scripts.automod import backlog as B, cluster as CL, state as S
    return B.select_cluster(S.LEDGER_PATH, CL.load_clusters(),
                            min_size=min_size, max_size=max_size) is None


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    """Nightly by age — and sooner, once the last pass is used up.

    Group triage takes its clusters from this file and judges each once. On
    2026-09-13 the night's 31 clusters were consumed by mid-morning, and group
    triage then had nothing to do for the rest of the day. So a file younger
    than `min_age_seconds` is still rebuilt when it is at least
    `exhausted_min_age_seconds` old and `select_cluster` finds nothing in it.
    The pass is offline arithmetic plus pair-judge calls cached by body hash,
    so a rebuild over an unchanged board is cheap and simply finds nothing.
    """
    from scripts.automod import cluster as CL
    min_age = float(src_cfg.get("min_age_seconds", DEFAULT_MIN_AGE_SECONDS))
    age = _clusters_age_seconds()
    trigger = "nightly"
    if age is not None and age < min_age:
        if age < float(src_cfg.get("exhausted_min_age_seconds", DEFAULT_EXHAUSTED_MIN_AGE_SECONDS)):
            return
        from workers.sources import autotriage as T
        exhausted = await asyncio.to_thread(
            _clusters_exhausted,
            int(src_cfg.get("group_min_items", T.DEFAULT_GROUP_MIN_ITEMS)),
            int(src_cfg.get("group_max_items", T.DEFAULT_GROUP_MAX_ITEMS)))
        if not exhausted:
            return
        trigger = "exhausted"
    new_id = queue.enqueue(
        source=NAME, kind="cluster",
        payload={"trigger": trigger,
                 "threshold": float(src_cfg.get("threshold", CL.DEFAULT_THRESHOLD)),
                 "judge": bool(src_cfg.get("judge", True)),
                 "max_pairs_judged": int(src_cfg.get("max_pairs_judged", 200)),
                 "persist_parents": bool(src_cfg.get("persist_parents", True))},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued backlog clustering id=%d (%s)", new_id, trigger)


async def execute(item: QueueItem) -> dict[str, Any]:
    from scripts.automod import cluster as CL, state as S
    p = item.payload or {}
    data = await asyncio.to_thread(
        CL.build_clusters,
        threshold=float(p.get("threshold", CL.DEFAULT_THRESHOLD)),
        judge=bool(p.get("judge", True)),
        max_pairs_judged=int(p.get("max_pairs_judged", 200)),
        persist_parents_=bool(p.get("persist_parents", True)))
    path = await asyncio.to_thread(CL.write_clusters, data)
    n_items = sum(len(c["item_ids"]) for c in data["clusters"])
    S.append_event({"event": "backlog_cluster", "trigger": str(p.get("trigger") or "nightly"),
                    "clusters": len(data["clusters"]),
                    "items": n_items, "items_considered": data["items_considered"],
                    "vectors_found": data["vectors_found"],
                    "pairs_candidate": data["pairs_candidate"],
                    "pairs_judged": data["pairs_judged"], "judge_cached": data["judge_cached"],
                    "judge_errors": data["judge_errors"],
                    "parents_persisted": data["parents_persisted"],
                    "threshold": data["threshold"],
                    "cluster_ids": [c["id"] for c in data["clusters"]],
                    "path": str(path)})
    summary = (f"{len(data['clusters'])} clusters over {n_items} items "
               f"({data['pairs_judged']} pairs judged, {data['judge_cached']} cached); "
               f"{data['parents_persisted']} parents persisted")
    if not data["vectors_found"]:
        # Still a success: paths and parents are a real signal on their own.
        summary += " — no vectors available, clustered on paths and parents only"
    return {"status": "success", "summary": summary, "artifact_path": str(path)}
