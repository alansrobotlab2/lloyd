"""Backend lifecycle handlers — graceful shutdown for in-flight turns."""

import asyncio
import logging

from app.harness.mcp_pool import close_all_pools
from app.inner_voice.observer import aclose_clients as _iv_aclose_clients
from app.sessions_io import _session_queues

logger = logging.getLogger("lloyd-server.lifecycle")


async def shutdown_cleanup(grace_seconds: float = 2.0) -> None:
    """Cancel in-flight turns, then close shared MCP pools."""
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

    try:
        await close_all_pools()
    except Exception as exc:
        logger.warning("Shutdown: close_all_pools raised: %s", exc)

    # The Inner Voice observer keeps one pooled httpx.AsyncClient per event
    # loop for its vLLM calls. It had no production call site, so the client
    # and its keep-alive connections were never closed on shutdown.
    try:
        await _iv_aclose_clients()
    except Exception as exc:
        logger.warning("Shutdown: inner-voice client close raised: %s", exc)
