"""Backend lifecycle handlers — graceful shutdown for in-flight turns."""

import asyncio
import logging

from app.sessions_io import _session_queues

logger = logging.getLogger("lloyd-server.lifecycle")


async def shutdown_cleanup(grace_seconds: float = 2.0) -> None:
    """Cancel in-flight turns and wait briefly for clean teardown."""
    consumer_tasks = []
    for session_id, q in list(_session_queues.items()):
        try:
            q.cancel_event.set()
        except Exception:
            pass
        if q.consumer_task and not q.consumer_task.done():
            consumer_tasks.append(q.consumer_task)

    if consumer_tasks:
        logger.info(f"Shutdown: waiting up to {grace_seconds}s for {len(consumer_tasks)} consumer task(s)")
        for t in consumer_tasks:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*consumer_tasks, return_exceptions=True),
                timeout=grace_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Shutdown: consumer tasks did not finish within grace period")
