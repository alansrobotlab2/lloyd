"""Lloyd unified work queue + worker pool.

Replaces the time-based autonomy scheduler, the vault-change KG pipeline,
and the standalone autoresearch loop with a single persistent queue drained
by N concurrent asyncio workers (see docs/21-unified-work-queue.md).
"""

from workers.queue import WorkQueue, QueueItem
from workers.pool import WorkerPool, get_pool, start_pool

__all__ = ["WorkQueue", "QueueItem", "WorkerPool", "get_pool", "start_pool"]
