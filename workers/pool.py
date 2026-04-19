"""Worker pool — N asyncio tasks draining the WorkQueue.

Sources register into SOURCE_REGISTRY (see workers/sources/__init__.py).
Each source provides:
  - NAME: str
  - async enqueue_if_due(queue, config) -> None
  - async execute(item) -> dict  (result with `summary`, `artifact_path`, `response`)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from workers.queue import WorkQueue, QueueItem, get_queue, new_run_id

logger = logging.getLogger("lloyd-workers.pool")


class WorkerPool:
    def __init__(
        self,
        queue: WorkQueue,
        slots: int = 8,
        max_attempts: int = 3,
        poll_idle_seconds: float = 2.0,
    ):
        self.queue = queue
        self.slots = slots
        self.max_attempts = max_attempts
        self.poll_idle_seconds = poll_idle_seconds

        self._running = False
        self._paused = False
        self._workers: list[asyncio.Task] = []
        self._scheduler_task: Optional[asyncio.Task] = None
        self._in_flight: dict[int, dict[str, Any]] = {}

    @property
    def worker_ids(self) -> list[str]:
        return [f"worker-{i}" for i in range(self.slots)]

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        # Recover any items stuck in claimed|running from a prior crash.
        self.queue.recover_claimed(self.worker_ids)

        for i in range(self.slots):
            self._workers.append(asyncio.create_task(
                self._worker_loop(f"worker-{i}"),
                name=f"lloyd-worker-{i}",
            ))
        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(), name="lloyd-worker-scheduler"
        )
        logger.info("Worker pool started with %d slots", self.slots)

    async def stop(self) -> None:
        self._running = False
        for task in self._workers:
            task.cancel()
        if self._scheduler_task:
            self._scheduler_task.cancel()
        # Drain cancellations
        for task in [*self._workers, self._scheduler_task]:
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._workers = []
        self._scheduler_task = None
        logger.info("Worker pool stopped")

    def pause(self, paused: bool = True) -> None:
        self._paused = paused
        logger.info("Worker pool %s", "paused" if paused else "resumed")

    @property
    def paused(self) -> bool:
        return self._paused

    def status(self) -> dict:
        return {
            "running": self._running,
            "paused": self._paused,
            "slots": self.slots,
            "in_flight": {
                str(k): {
                    "source": v.get("source"),
                    "kind": v.get("kind"),
                    "started_at": v.get("started_at"),
                }
                for k, v in self._in_flight.items()
            },
            "in_flight_count": len(self._in_flight),
        }

    # ── Scheduler loop — drives source.enqueue_if_due() ───────────────────

    async def _scheduler_loop(self) -> None:
        from workers.sources import SOURCE_REGISTRY, get_sources_config

        while self._running:
            try:
                cfg = get_sources_config()
                interval = 60
                for name, source in SOURCE_REGISTRY.items():
                    src_cfg = cfg.get(name, {})
                    if not src_cfg.get("enabled", False):
                        continue
                    wait = int(src_cfg.get("interval_seconds", 3600))
                    last_at = self.queue.wm_get(name, "last_enqueue_check")
                    if last_at:
                        last_dt = datetime.fromisoformat(last_at)
                        elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                        if elapsed < wait:
                            continue
                    try:
                        await source.enqueue_if_due(self.queue, src_cfg)
                    except Exception as e:
                        logger.error("Source %s enqueue_if_due failed: %s", name, e, exc_info=True)
                    self.queue.wm_set(name, "last_enqueue_check",
                                      datetime.now(timezone.utc).isoformat())
            except Exception as e:
                logger.error("Scheduler loop error: %s", e, exc_info=True)
            await asyncio.sleep(interval)

    # ── Worker loop — claims items and runs them ──────────────────────────

    async def _worker_loop(self, worker_id: str) -> None:
        from workers.sources import SOURCE_REGISTRY, get_sources_config

        while self._running:
            if self._paused:
                await asyncio.sleep(self.poll_idle_seconds)
                continue

            cfg = get_sources_config()
            max_inflight = {
                name: int(src.get("max_inflight", 999))
                for name, src in cfg.items()
                if src.get("max_inflight") is not None
            }

            item = await asyncio.get_event_loop().run_in_executor(
                None, self.queue.claim_next, worker_id, max_inflight
            )
            if not item:
                await asyncio.sleep(self.poll_idle_seconds)
                continue

            source = SOURCE_REGISTRY.get(item.source)
            if not source:
                logger.error("Unknown source %s for item %d — marking poisoned", item.source, item.id)
                self.queue.mark_failed(item.id, f"unknown source {item.source}", max_attempts=0)
                continue

            self.queue.mark_running(item.id)
            started_at_iso = datetime.now(timezone.utc).isoformat()
            started_perf = time.monotonic()
            self._in_flight[item.id] = {
                "source": item.source,
                "kind": item.kind,
                "started_at": started_at_iso,
                "worker": worker_id,
            }

            run_id = new_run_id(item.source)
            logger.info("[%s] running %s/%s (id=%d) run_id=%s",
                        worker_id, item.source, item.kind, item.id, run_id)

            try:
                result = await source.execute(item)
                duration = time.monotonic() - started_perf
                completed_at = datetime.now(timezone.utc).isoformat()
                self.queue.record_run(
                    run_id=run_id,
                    queue_id=item.id,
                    source=item.source,
                    status="success",
                    started_at=started_at_iso,
                    completed_at=completed_at,
                    duration_seconds=duration,
                    summary=(result.get("summary") or "")[:500],
                    artifact_path=result.get("artifact_path") or "",
                    response_json=(result.get("response") or "")[:50000],
                    task_id=str(result.get("task_id")) if result.get("task_id") is not None else None,
                )
                self.queue.mark_completed(item.id)
                logger.info("[%s] completed %s/%s in %.1fs",
                            worker_id, item.source, item.kind, duration)
            except Exception as e:
                duration = time.monotonic() - started_perf
                completed_at = datetime.now(timezone.utc).isoformat()
                error_msg = f"{type(e).__name__}: {e}"
                self.queue.record_run(
                    run_id=run_id,
                    queue_id=item.id,
                    source=item.source,
                    status="failed",
                    started_at=started_at_iso,
                    completed_at=completed_at,
                    duration_seconds=duration,
                    summary=error_msg[:500],
                )
                new_state = self.queue.mark_failed(item.id, error_msg, self.max_attempts)
                logger.error("[%s] failed %s/%s: %s → %s",
                             worker_id, item.source, item.kind, error_msg, new_state)
            finally:
                self._in_flight.pop(item.id, None)


# ── Module-level singleton ────────────────────────────────────────────────

_pool_instance: Optional[WorkerPool] = None


def get_pool() -> Optional[WorkerPool]:
    return _pool_instance


async def start_pool(queue: WorkQueue, slots: int, max_attempts: int = 3) -> WorkerPool:
    """Create the singleton pool (if missing) and start it."""
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = WorkerPool(queue, slots=slots, max_attempts=max_attempts)
    await _pool_instance.start()
    return _pool_instance
