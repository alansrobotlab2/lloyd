"""session-distill source — mines ~/lloyd/sessions/*.json for patterns.

Each eligible session becomes one queue item; the handler asks the primary
model to identify repeated failures, unresolved questions, or gap signals, and
writes findings to pending-research/distill/.

**A session is distilled once, after it goes quiet.** The three gates below
each exist because the original selection had none of them, and the cost was
not subtle: one chat was distilled 44 times, another 22, and the eight worst
account for 138 of this source's runs.

  * *Once.* Selection keyed on an `mtime > last_mtime` watermark, and a
    session's mtime advances with every message it receives. So an active chat
    crossed the watermark again on the very next tick, forever, and the queue's
    `dedup_key` could not help — `mark_completed` releases it by design, since
    most sources do want to re-run the same key later. Per-session `done:`
    markers replace the moving watermark, and they also fix its mirror-image
    bug: a session that was skipped while busy used to be stranded below the
    advanced watermark and never looked at again.
  * *Quiet.* A chat still being typed into is not a transcript to learn from.
  * *A user session.* `platform: worker` and `platform: autonomy` sessions are
    this system talking to itself. Distilling them fed Lloyd's own triage
    transcripts back in as observations about the user, and 12 of them had
    already been mined that way.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem
from workers.sources._common import (
    parse_confidence, run_prompt_on_primary, write_staging_note,
)

logger = logging.getLogger("lloyd-workers.session_distill")

NAME = "session-distill"
DEFAULT_PRIORITY = 70
from app.paths import SESSIONS_DIR  # anchored to LLOYD_HOME, not $HOME/lloyd

_MAX_ENQUEUE_PER_TICK = 3

# How long a session must have sat untouched before it is worth distilling.
_QUIET_SECONDS = 30 * 60

# Only the head of a session file is read to find its platform. The field is
# written by every producer within the first few keys (`sessions_io` and
# `_common.new_worker_session` both put it 5th), while the file itself carries
# the whole transcript and runs to megabytes — and this check happens per
# candidate, per tick. A prefix that does not contain the key reads as a user
# session, which is the same default `is_user_session` applies to a session
# with no platform at all.
_PLATFORM_PREFIX_BYTES = 8192
_PLATFORM_RE = re.compile(r'"platform"\s*:\s*"([^"]*)"')


def _done_key(name: str) -> str:
    return f"done:{name}"


def _session_platform(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            head = f.read(_PLATFORM_PREFIX_BYTES)
    except OSError:
        return ""
    m = _PLATFORM_RE.search(head)
    return m.group(1) if m else ""


def _eligible(path: Path, mtime: float, now: float) -> tuple[bool, str]:
    """Whether this session file is worth distilling, and why not if it isn't."""
    from app.sessions_io import is_session_active, is_user_session

    if now - mtime < _QUIET_SECONDS:
        return False, "still active"
    if is_session_active(path.stem):
        return False, "a turn is running"
    if not is_user_session({"platform": _session_platform(path)}):
        return False, "not a user session"
    return True, ""


def _scan(now: float, already_done: set[str]) -> list[tuple[float, Path]]:
    """Oldest-first eligible sessions. Blocking; called via a thread."""
    out: list[tuple[float, Path]] = []
    for p in SESSIONS_DIR.glob("*.json"):
        if p.name in already_done:
            continue
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        ok, _why = _eligible(p, m, now)
        if ok:
            out.append((m, p))
    out.sort()
    return out


_LEGACY_CURSOR = "last_mtime"


def _migrate_legacy_cursor(queue: WorkQueue) -> int:
    """Convert the old `last_mtime` cursor into per-session markers, once.

    Without this the fix is a regression on its own history: the old cursor
    was the only record that a session had been considered, so dropping it
    makes every session below it eligible again — 143 files here, most of
    them already distilled, re-offered three per tick.

    A session at or below the cursor was enqueued at some point, because the
    cursor only ever advanced over items that were. The key is deleted after,
    so the markers are the sole authority from here on and no cursor is left
    for a session to be stranded behind.
    """
    cursor = queue.wm_get(NAME, _LEGACY_CURSOR)
    if cursor is None:
        return 0
    try:
        cutoff = float(cursor)
    except ValueError:
        cutoff = 0.0
    seeded = 0
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            if p.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        if queue.wm_get(NAME, _done_key(p.name)) is None:
            queue.wm_set(NAME, _done_key(p.name),
                         json.dumps({"why": "migrated from last_mtime cursor",
                                     "at": time.time()}))
            seeded += 1
    queue.wm_delete(NAME, _LEGACY_CURSOR)
    logger.info("session-distill: migrated the last_mtime cursor into %d markers", seeded)
    return seeded


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    import asyncio

    if not SESSIONS_DIR.exists():
        return

    await asyncio.to_thread(_migrate_legacy_cursor, queue)

    # One watermark row per session already handled. Read whole rather than
    # queried per candidate: it is a few hundred rows against one connection.
    done = {k[len("done:"):] for k in queue.wm_keys(NAME) if k.startswith("done:")}
    candidates = await asyncio.to_thread(_scan, time.time(), done)

    enqueued = 0
    for m, p in candidates[:_MAX_ENQUEUE_PER_TICK]:
        new_id = queue.enqueue(
            source=NAME,
            kind="distill",
            payload={"session_path": str(p), "mtime": m},
            priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
            dedup_key=f"session-distill:{p.name}",
        )
        if new_id is not None:
            enqueued += 1
    if enqueued:
        logger.info("Enqueued %d session-distill items (%d eligible, %d already done)",
                    enqueued, len(candidates), len(done))


async def execute(item: QueueItem) -> dict[str, Any]:
    session_path = item.payload.get("session_path", "")
    session_name = Path(session_path).stem

    prompt = (
        f"You are analyzing a saved Lloyd session transcript to distill what we can "
        f"learn from it. The session file is at: {session_path}\n\n"
        f"Read the file using the Read tool, then identify:\n"
        f"1. Any repeated user struggles or failed assistant attempts\n"
        f"2. Knowledge gaps — questions Lloyd couldn't confidently answer\n"
        f"3. Patterns that could become a skill (if the same sequence of tools is "
        f"invoked in a reliable order)\n"
        f"4. Durable facts about the user that should be captured\n\n"
        f"CRITICAL GUARDRAIL: If the session is trivial (health check, empty, routine "
        f"maintenance, <5 messages, no substantive content), return sections with "
        f"'- none' or '- N/A' for each category. Do NOT create entities, facts, or "
        f"observations about the session itself (e.g., 'the session was short', "
        f"'had no unresolved threads', 'involved no external papers'). Only extract "
        f"facts about actual topics discussed, not metadata about the session's "
        f"characteristics. Sessions themselves are never entities. When calling "
        f"fact_add, ONLY create entities for real-world topics (people, companies, "
        f"technologies, concepts) — never for the session, the conversation, or "
        f"metadata about the interaction. If the session has no substantive topics, "
        f"return the distillation sections with '- none' and do NOT call fact_add.\n\n"
        f"Return in this structure:\n"
        f"## Struggles\n- ...\n\n## Gaps\n- ...\n\n## Skill Candidates\n- ...\n\n"
        f"## Durable Facts\n- ...\n\n## Confidence\n<0.0-1.0>: <justification>\n"
    )
    turn = await run_prompt_on_primary(
        prompt, max_turns=15, source=NAME,
        title=f"distill {session_name}")
    if not turn.ok:
        # 135 of this source's 356 notes have the body "(no response)". An
        # empty turn is a failed run, and the retry is this source's own: the
        # pool completes an in-band failure rather than requeueing it, so the
        # next scan is what tries again, and `_mark_done_if_exhausted` is what
        # stops that being a cycle.
        logger.warning("session-distill %s: %s", session_name, turn.failure_summary())
        _mark_done_if_exhausted(item)
        return {"status": "failed",
                "summary": f"{session_name}: {turn.failure_summary()}",
                "meta": {"empty_response": True, "stop_reason": turn.stop_reason,
                         "num_turns": turn.num_turns}}

    conf = parse_confidence(turn.text)
    path = write_staging_note(
        source=NAME,
        slug=session_name[:40],
        body=turn.text,
        confidence=conf,
        rationale=f"distilled from {session_name}",
        source_refs=[session_path],
    )
    _mark_done(Path(session_path).name, "distilled")
    _clear_failures(Path(session_path).name)
    return {
        "status": "success",
        "summary": f"distilled {session_name}",
        "response": turn.text,
        "artifact_path": str(path),
        "meta": {"stop_reason": turn.stop_reason, "num_turns": turn.num_turns},
    }


def _clear_failures(file_name: str) -> None:
    """Forget earlier failed attempts once a session finally distils."""
    try:
        from workers.queue import get_queue
        get_queue().wm_delete(NAME, _fail_key(file_name))
    except Exception:
        pass


def _mark_done(file_name: str, why: str) -> None:
    """Record that this session will not be offered again."""
    try:
        from workers.queue import get_queue
        get_queue().wm_set(NAME, _done_key(file_name),
                           json.dumps({"why": why, "at": time.time()}))
    except Exception as exc:
        logger.warning("could not record session-distill marker for %s: %s",
                       file_name, exc)


def _fail_key(name: str) -> str:
    return f"fail:{name}"


def _mark_done_if_exhausted(item: QueueItem) -> None:
    """Count a failed attempt, and give up once there have been enough.

    **The count cannot come from the queue item.** This reads `item.attempts`
    in its first cut, on the belief that a returned `{"status": "failed"}`
    goes back through the queue's retry path and increments it. It does not:
    `workers/pool.py` records the failed run and then calls `mark_completed`
    on the item regardless, so only a *raised* exception ever reaches
    `mark_failed`. `item.attempts` is therefore 1 on every failed distill, the
    give-up branch never fired, and the session came back on the next tick
    with no marker to stop it — 39 failed runs in the eight hours after that
    shipped, one session enqueued 21 times.

    So the attempt count is a watermark of this source's own, beside the
    `done:` markers, and it is cleared when a session finally succeeds.
    """
    from app.config import CONFIG

    name = Path(item.payload.get("session_path", "")).name
    if not name:
        return
    max_attempts = int((CONFIG.get("workers") or {}).get("max_attempts", 3))
    try:
        from workers.queue import get_queue
        queue = get_queue()
        attempts = int(queue.wm_get(NAME, _fail_key(name)) or 0) + 1
        if attempts >= max_attempts:
            _mark_done(name, f"abandoned after {attempts} failed attempts")
            queue.wm_delete(NAME, _fail_key(name))
        else:
            queue.wm_set(NAME, _fail_key(name), str(attempts))
    except Exception as exc:
        logger.warning("could not record a session-distill attempt for %s: %s",
                       name, exc)
