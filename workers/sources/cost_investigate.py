"""The diagnostic that a cost outlier arms (#2510).

**Nothing here is on a schedule.** Every other source in this directory is asked
"anything due?" by `WorkerPool._scheduler_pass` on its `interval_seconds`. This one
is different: its work exists only because a run just finished that cost more than
1.5x its own source's window P90, and `workers/pool.py` enqueues it at that moment
(`app.cost_outlier.arm_after_run`). A source that also polled would be inventing
work that the trigger already made, so `enqueue_if_due` declines, always, and says
why. The `interval_seconds` in its config block exists to satisfy the pool's
expectation that a source has a schedule, not to create one.

**No model turn, and that is the point.** The item's own risk list says the
investigation must cost less than what it explains, and it must not feed its own
trigger. An LLM turn re-reading an 800k-token run is a second 800k-token run: it
would be an outlier itself, and the recursion guard in `app/cost_outlier.py`
`EXCLUDED_SOURCES` would have to hide it from a distribution it legitimately
dirty. So this executes as a read of the run's own `usage` rows and a rule with a
figure beside it — deterministic, zero GPU, and checkable by anyone who opens the
store. If these records turn out too crude to act on, the fix is a bounded turn
added to the figures, never one replacing them.

**The recursion guard, said plainly:** `cost-investigate` runs are excluded from
the token distribution that arms this trigger. Without that, one expensive audit
raises its own source's P90 and hides the next one — and the audit of an audit is
the chain the source talk never mentions.
"""

from __future__ import annotations

import asyncio
import logging

from app import cost_outlier as CO
from workers.queue import QueueItem
from workers.sources import DECLINED

logger = logging.getLogger("lloyd-workers.cost-investigate")

NAME = CO.SOURCE
#: Lowest-priority band in the fleet, and deliberately so: this job explains a
#: cost that has already been paid, and it must never be the reason a triage or a
#: code round waits. `arm_after_run` reads the same constant for the row it writes.
DEFAULT_PRIORITY = CO.DEFAULT_PRIORITY

MAX_DURATION_SECONDS = 120.0


async def enqueue_if_due(queue, src_cfg: dict):
    """Always DECLINED: this source's work is created by a completion hook.

    Returning ENQUEUED from here would double the trigger — once from the hook,
    once from the tick — and a duplicate record of a run that already happened is
    exactly the noise the cooldown exists to prevent.
    """
    return DECLINED


async def execute(item: QueueItem) -> dict:
    """Write the one markdown record for the run named in the payload."""
    payload = dict(item.payload or {})
    if not payload.get("run_id"):
        # No requeue: a row without a run to investigate can never be
        # investigable, and requeuing it would spend a slot on a fixed error.
        return {"status": "failed",
                "summary": f"payload has no run_id: {sorted(payload)}"}
    result = await asyncio.to_thread(CO.investigate, payload)
    logger.info("[%s] wrote %s", NAME, result.get("artifact_path"))
    return result

