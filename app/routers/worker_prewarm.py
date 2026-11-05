"""``POST /api/worker/prewarm`` — the trigger seam for #3771.

A process that finished rewriting the prompt surface has to be able to say so:
the nightly consolidation, an ``automod_vault_land``, and the backend itself are
three different writers, and the only thing they share is that none of them is
this process. So they POST here and `app.worker_prewarm.prewarm` decides — the
admission rules, the ledger row and the sha register all live there, so no caller
can talk its way past a refusal.

A refusal is a 200 with ``outcome: "refused"`` and the reason, not a 4xx: every
caller here is a fire-and-forget notifier, and a 4xx on a warm that was correctly
declined reads as a broken caller in whatever log is watching. An unknown trigger
is a 400, because that is a caller bug and nothing downstream can tell it apart.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger("lloyd-automod")  # same file as every other automod POST


@router.post("/api/worker/prewarm")
async def worker_prewarm(
    body: dict = Body(default_factory=dict),
) -> JSONResponse:
    """Warm the job prompt head now, if admission allows it.

    Body: ``{"trigger": "prompt-surface-write" | "engine-ready" |
    "head-sha-changed", "source": "autotriage"}``, ``source`` optional. Replies
    with the ledger row, so a caller that wants the verdict can read it.
    """
    from app import worker_prewarm as wp

    trigger = str(body.get("trigger") or "")
    if trigger not in wp.PREWARM_TRIGGERS:
        return JSONResponse(
            status_code=400,
            content={"ok": False,
                     "error": f"unknown trigger {trigger!r}; "
                              f"expected one of {list(wp.PREWARM_TRIGGERS)}"})
    src = str(body.get("source") or "") or None
    row = await wp.prewarm(trigger, src=src)
    return JSONResponse(status_code=200,
                        content={"ok": row.get("outcome") == "sent", **row})
