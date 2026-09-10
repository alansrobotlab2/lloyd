"""Worker pool — N asyncio tasks draining the WorkQueue.

Sources register into SOURCE_REGISTRY (see workers/sources/__init__.py).
Each source provides:
  - NAME: str
  - async enqueue_if_due(queue, config) -> None
  - async execute(item) -> dict  (see `normalize_result` for the contract)

**Everything here runs on the backend's one event loop.** That loop also
serves every HTTP request and streams every chat turn, so a source that
blocks it stops the whole machine, and the queue's own SQLite writes are
hopped onto a thread for the same reason. `architecture/workers.md` has the
long version.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from functools import partial
from typing import Any, Optional

from app import engine_pressure
from app.harness.policy import current_effect_scope, current_scope
from app.sessions_io import current_run_sessions
from workers.evidence import gaps_key, verify_bundle
from workers.queue import WorkQueue, QueueItem, get_queue, new_run_id

logger = logging.getLogger("lloyd-workers.pool")

# Fallback when a source has no max_duration_seconds in config.
_DEFAULT_MAX_DURATION_SECONDS = 900

# The statuses a run record may carry. `skipped` is not a failure and not a
# success: it means the source looked and there was nothing to do.
RUN_STATUSES = ("success", "failed", "skipped")

# ── KV budget gate ────────────────────────────────────────────────────
#
# The 09-09 stall was long-lived agent loops evicting each other's prefixes.
# Every iteration re-submits a 100-200k context, and between iterations a
# turn's prefix sits only in the engine's free pool, where the next
# allocation can take it. Three or four of those resident at once on a 398k
# pool and one of them came back cold on almost every iteration
# (architecture/vllm-throughput-mitigation.md). Short-lived jobs never came
# back to miss, which is why seven youtube digests at 81.5% KV ran clean the
# night before.
#
# So the gate decides *who* starts, not how many: a source that declares
# `LONG_LIVED = True` is not claimed while the primary's KV usage is above
# `workers.kv_gate.max_kv_usage`, and every other source claims as before. A
# hold is not a failure — the item stays queued with its attempt intact — and
# it reads the background sampler, never the engine, so a claim does not wait
# on an HTTP call. No reading (sampler off, engine down, stale sample) means
# the gate is open: a pressure signal that failed closed would turn an
# unreachable /metrics into a stopped pool.
#
# It judges the MEDIAN over the last minute, not the last sample. Measured on
# the FP8 build on 2026-09-10 (bench-admission-stall.py): a cold 200k prefill
# drives `kv_cache_usage_perc` from 0.20 to 0.96 over its 21 s and it falls
# to 0.50 the moment the prompt is in — while a prompt is being built this
# hybrid model references ~2.5x its resident footprint. On the last sample
# the gate would engage on every cold prefill and hold for its duration; on
# a one-minute median it sees what it is for, the residents.

DEFAULT_KV_GATE_MAX = 0.60
DEFAULT_KV_GATE_WINDOW_S = 60.0


def kv_gate_config() -> dict[str, Any]:
    try:
        from app.config import CONFIG
        return dict((CONFIG.get("workers") or {}).get("kv_gate") or {})
    except Exception:
        return {}


def long_lived_sources(registry: dict[str, Any]) -> list[str]:
    """Sources that declare themselves long-lived re-admitters.

    Static, by design: a module attribute is a claim a reviewer can read,
    where a runtime measurement of context size would be one more guess.
    """
    return sorted(name for name, src in registry.items()
                  if getattr(src, "LONG_LIVED", False))


def normalize_result(item: QueueItem, result: Any) -> dict[str, Any]:
    """Coerce whatever a source returned into the run-record contract.

    The contract is `{status, summary, artifact_path, response, task_id, meta,
    claims}`, and every field is optional except in the sense that its absence
    has to mean something defensible. Absent `status` means success, because
    most sources finish by returning their artifact and never think about it.

    The interesting case is the one that was silently wrong. `automod-
    regression` reports "I could not measure anything" by returning
    `{"skipped": "<reason>"}` — a key, not a status — so all 22 of its runs
    were recorded as successes with an empty summary. A check that never ran
    is indistinguishable, on the dashboard and in the runs table, from a
    check that ran and found nothing wrong, which is precisely the failure
    that source's own docstring is written to prevent ("a missing noise file
    means cannot evaluate, never no regression"). So a bare `skipped` key is
    honoured as a status, and a result that carries neither a summary nor an
    artifact is logged — an unreadable run record is a bug in the source, and
    it should be visible as one rather than as a blank cell.
    """
    if not isinstance(result, dict):
        if result is not None:
            logger.warning("source %s returned %s, not a dict — recording as success",
                           item.source, type(result).__name__)
        result = {}

    status = result.get("status")
    if status not in RUN_STATUSES:
        if status is not None:
            logger.warning("source %s returned unknown status %r — recording as success",
                           item.source, status)
        # `{"skipped": reason}` — a key where a status belongs.
        status = "skipped" if result.get("skipped") else "success"

    summary = str(result.get("summary") or "")
    if not summary and status == "skipped":
        summary = str(result.get("skipped") or "")
    if not summary and not result.get("artifact_path"):
        logger.warning("source %s produced a run record with no summary and no "
                       "artifact — the run is unreadable after the fact", item.source)

    # `claims` is a list of `{claim, check}` pairs the source wants recorded as
    # this run's evidence (#525). Presence of the key is the pilot's scope
    # mechanism: a source that does not emit claims gets no bundle at all, and
    # the health view counts that as `runs_without_bundle` instead of scoring it
    # as a clean check. An empty list is different and deliberate — the source
    # is in the pilot and its model emitted nothing, which is a gap.
    claims = result.get("claims") if isinstance(result.get("claims"), list) else None

    return {
        "status": status,
        "summary": summary[:500],
        "artifact_path": str(result.get("artifact_path") or ""),
        "response": str(result.get("response") or "")[:50000],
        "task_id": _task_id_of(item, result),
        "meta": result.get("meta") if isinstance(result.get("meta"), dict) else {},
        "claims": claims,
    }



def grant_scope_for(item: QueueItem) -> str:
    """The authority scope a claimed item runs under (#534).

    An autonomy task gets its own scope rather than sharing `worker:scheduled-task`
    with every other task, because that is the difference between a human
    granting `email_send` to the nightly mail job and granting it to whatever
    runs on that source next. Anything else is its source.
    """
    if item.source == "scheduled-task":
        task_id = item.payload.get("task_id")
        if task_id is not None:
            return f"autonomy-task:{task_id}"
    return f"worker:{item.source}"

def effect_scope_for(item: QueueItem) -> str:
    """The effect scope a claimed item runs under (#544).

    The queue item id, because that is the unit a retry re-runs: `run_id` is
    minted per attempt, so keying on it would let attempt 2 fire a fresh effect
    — which is the bug. The source is in the string because item ids are only
    unique within this database file, and a name makes a ledger row readable
    without a join back to `queue`.

    Deliberately NOT the grant scope: `autonomy-task:39` is stable across every
    run that task ever has, which is right for a permission and wrong for
    idempotency — it would suppress a legitimate second effect forever.
    """
    return f"item:{item.source}:{item.id}"


def _task_id_of(item: QueueItem, result: Any = None) -> Optional[str]:
    """Task id for a run record: prefer the handler's result, fall back to the
    queue payload. Timeout/exception branches have no result, and omitting the
    id there is what left 237 runs / 73.6 GPU-hours unattributable in the runs
    table — invisible to every per-task view."""
    tid = result.get("task_id") if isinstance(result, dict) else None
    if tid is None:
        tid = item.payload.get("task_id")
    return None if tid is None else str(tid)


class WorkerPool:
    def __init__(
        self,
        queue: WorkQueue,
        slots: int = 4,
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
        # KV gate state, reported by `status()`. See `_kv_gate_held`.
        self._kv_gate: dict[str, Any] = {
            "engaged": False,
            "engaged_since": None,
            "engagements": 0,
            "kv_usage": None,
            "held_sources": [],
        }

    @property
    def worker_ids(self) -> list[str]:
        return [f"worker-{i}" for i in range(self.slots)]

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        # Recover every item stuck in claimed|running from a prior crash —
        # not just the ones belonging to the slot names this pool is about to
        # use. Nothing is in flight when a pool starts, and filtering by the
        # current `worker_ids` strands rows claimed by a slot that no longer
        # exists as soon as `workers.slots` is lowered. See
        # `WorkQueue.recover_claimed`.
        await asyncio.to_thread(self.queue.recover_claimed)

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
            "kv_gate": self.kv_gate_status(),
        }

    def kv_gate_status(self) -> dict[str, Any]:
        cfg = kv_gate_config()
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "max_kv_usage": float(cfg.get("max_kv_usage", DEFAULT_KV_GATE_MAX)),
            "window_seconds": float(cfg.get("window_seconds", DEFAULT_KV_GATE_WINDOW_S)),
            **self._kv_gate,
            "held_sources": list(self._kv_gate["held_sources"]),
        }

    def _kv_gate_held(self, registry: dict[str, Any]) -> list[str]:
        """Sources this claim must skip because the primary's KV is over budget.

        Called before every claim. Logs only on a transition, so a gate that
        stays engaged for an hour is two lines, not 1,800.
        """
        cfg = kv_gate_config()
        enabled = bool(cfg.get("enabled", True))
        limit = float(cfg.get("max_kv_usage", DEFAULT_KV_GATE_MAX))
        window = float(cfg.get("window_seconds", DEFAULT_KV_GATE_WINDOW_S))
        # A fresh sample is the precondition; the median is the reading.
        kv = None
        if enabled and engine_pressure.latest() is not None:
            kv = engine_pressure.kv_percentile(0.5, window=window)
        engaged = enabled and kv is not None and kv > limit
        held = long_lived_sources(registry) if engaged else []
        gate = self._kv_gate
        gate["kv_usage"] = kv
        if engaged and not gate["engaged"]:
            gate.update(engaged=True, engagements=gate["engagements"] + 1,
                        engaged_since=datetime.now(timezone.utc).isoformat(),
                        held_sources=held)
            logger.info("KV gate engaged: primary KV %.0f%% > %.0f%% — holding %s",
                        100 * kv, 100 * limit, ", ".join(held) or "nothing")
        elif not engaged and gate["engaged"]:
            logger.info("KV gate released: primary KV %s",
                        "unknown" if kv is None else f"{100 * kv:.0f}%")
            gate.update(engaged=False, engaged_since=None, held_sources=[])
        return held

    def _carry_gaps(self, item: QueueItem, task_id: Optional[str],
                    bundle: dict) -> None:
        """Store one task's unverified claims for its next run's prompt (#525).

        The run record is where a refutation is *recorded*; this is where it
        becomes load-bearing. HOH's E-state keeps the gap set as a first-class
        part of the next iteration, because a gap that lives only in the prose
        of the last run is a gap nobody reads.

        Writing an empty list on a clean run is deliberate: a resolved gap must
        stop being re-litigated, or the carry-forward turns into permanent
        fine-print and gets skimmed like every other standing warning.
        """
        if not task_id:
            return
        try:
            self.queue.wm_set(item.source, gaps_key(task_id),
                              json.dumps(bundle.get("gap") or [])[:4000])
        except Exception as e:
            logger.warning("could not carry the evidence gap list for %s/%s: %s",
                           item.source, task_id, e)

    # ── Scheduler loop — drives source.enqueue_if_due() ───────────────────

    async def _scheduler_loop(self) -> None:
        from workers.sources import SOURCE_REGISTRY, get_sources_config

        # Defined outside the try, because the `sleep` at the bottom is
        # outside it too: if the first `get_sources_config()` raised, the
        # except branch fell through to `sleep(interval)` and died on a
        # NameError, taking the only thing that enqueues work with it.
        interval = 60
        while self._running:
            try:
                cfg = get_sources_config()
                for name, source in SOURCE_REGISTRY.items():
                    src_cfg = cfg.get(name, {})
                    if not src_cfg.get("enabled", False):
                        continue
                    wait = int(src_cfg.get("interval_seconds", 3600))
                    last_at = await asyncio.to_thread(
                        self.queue.wm_get, name, "last_enqueue_check")
                    if last_at:
                        last_dt = datetime.fromisoformat(last_at)
                        elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                        if elapsed < wait:
                            continue
                    try:
                        await source.enqueue_if_due(self.queue, src_cfg)
                    except Exception as e:
                        logger.error("Source %s enqueue_if_due failed: %s", name, e, exc_info=True)
                    await asyncio.to_thread(
                        self.queue.wm_set, name, "last_enqueue_check",
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

            held = self._kv_gate_held(SOURCE_REGISTRY)
            item = await asyncio.to_thread(
                self.queue.claim_next, worker_id, max_inflight, held)
            if not item:
                await asyncio.sleep(self.poll_idle_seconds)
                continue

            source = SOURCE_REGISTRY.get(item.source)
            if not source:
                logger.error("Unknown source %s for item %d — marking poisoned", item.source, item.id)
                await asyncio.to_thread(self.queue.mark_failed, item.id,
                                        f"unknown source {item.source}", 0)
                continue

            await asyncio.to_thread(self.queue.mark_running, item.id)
            started_at_iso = datetime.now(timezone.utc).isoformat()
            started_perf = time.monotonic()
            self._in_flight[item.id] = {
                "source": item.source,
                "kind": item.kind,
                "started_at": started_at_iso,
                "worker": worker_id,
            }

            run_id = new_run_id(item.source)
            cfg_all = get_sources_config()
            src_cfg_item = cfg_all.get(item.source, {}) if isinstance(cfg_all, dict) else {}
            max_duration = int(src_cfg_item.get("max_duration_seconds", _DEFAULT_MAX_DURATION_SECONDS))
            logger.info("[%s] running %s/%s (id=%d) run_id=%s timeout=%ds",
                        worker_id, item.source, item.kind, item.id, run_id, max_duration)

            # #534: whose authority is this turn borrowing. The grant gate
            # reads `current_scope` when it fires, because by then it is three
            # frames away inside a source's own call and only the pool knows
            # what it claimed. An autonomy task is its own scope — that is what
            # lets a human grant `email_send` to task #39 and nobody else.
            scope_token = current_scope.set(grant_scope_for(item))
            # #544: the same job, from the other side. `current_scope` bounds
            # whose authority this turn borrows; `current_effect_scope` bounds
            # which run's effects must not happen twice. The item id — not the
            # run id, which changes per attempt — is what survives a retry.
            effect_token = current_effect_scope.set(effect_scope_for(item))
            # Every background run is recorded now, so every run row can name
            # its transcript. Collected here rather than returned by each
            # source: a source that forgets is a run nobody can review, and
            # "the handler remembered to pass it back" is not a property worth
            # depending on eleven times.
            sessions_token = current_run_sessions.set([])
            try:
                result = await asyncio.wait_for(source.execute(item), timeout=max_duration)
                duration = time.monotonic() - started_perf
                completed_at = datetime.now(timezone.utc).isoformat()
                # A handler may report a task-level failure in-band rather than
                # raising. Raising would send the item back through the queue's
                # retry path, and for autonomy tasks that means re-running a
                # whole timed-out run up to max_attempts times before the
                # scheduler's own cooldown is ever consulted.
                norm = normalize_result(item, result)
                norm["meta"] = {**norm["meta"],
                                "session_ids": list(current_run_sessions.get() or [])}
                run_status = norm["status"]
                # #525 — verify the run's claims at the moment its record is
                # written, and only for a source that emitted any (`claims`
                # present is the pilot's scope switch). Off the event loop,
                # because a check reads files. The verifier is stdlib-only by
                # design: an LLM grading a model's own claims would be the
                # narration this replaces, one layer up.
                bundle = None
                if norm["claims"] is not None:
                    bundle = await asyncio.to_thread(verify_bundle, norm["claims"])
                    await asyncio.to_thread(
                        self._carry_gaps, item, norm["task_id"], bundle)
                await asyncio.to_thread(
                    partial(
                        self.queue.record_run,
                        run_id=run_id,
                        queue_id=item.id,
                        source=item.source,
                        status=run_status,
                        started_at=started_at_iso,
                        completed_at=completed_at,
                        duration_seconds=duration,
                        summary=norm["summary"],
                        artifact_path=norm["artifact_path"],
                        response_json=norm["response"],
                        task_id=norm["task_id"],
                        meta_json=json.dumps(norm["meta"], default=str),
                        claims_json=(json.dumps(bundle, default=str)
                                     if bundle is not None else ""),
                    )
                )
                await asyncio.to_thread(self.queue.mark_completed, item.id)
                logger.log(
                    logging.WARNING if run_status == "failed" else logging.INFO,
                    "[%s] %s %s/%s in %.1fs%s", worker_id,
                    {"failed": "FAILED", "skipped": "skipped"}.get(run_status, "completed"),
                    item.source, item.kind, duration,
                    f" — {norm['summary'][:120]}" if norm["summary"] else "")
                if bundle is not None:
                    unverified = (bundle["counts"]["refuted"]
                                  + bundle["counts"]["insufficient"])
                    if unverified:
                        logger.warning(
                            "[%s] %s/%s task %s: %d of %d evidence claims did NOT "
                            "verify against disk — %s; gap carried to the next run",
                            worker_id, item.source, item.kind, norm["task_id"],
                            unverified, bundle["counts"]["total"],
                            bundle["gap"][0][:160])
            except asyncio.TimeoutError:
                duration = time.monotonic() - started_perf
                completed_at = datetime.now(timezone.utc).isoformat()
                error_msg = f"TimeoutError: exceeded max_duration_seconds={max_duration}"
                await asyncio.to_thread(
                    partial(
                        self.queue.record_run,
                        run_id=run_id,
                        queue_id=item.id,
                        source=item.source,
                        status="failed",
                        started_at=started_at_iso,
                        completed_at=completed_at,
                        duration_seconds=duration,
                        summary=error_msg[:500],
                        task_id=_task_id_of(item),
                        # A timed-out run is the one most worth reading, so
                        # its transcript is named here too — not only on the
                        # success path.
                        meta_json=json.dumps({
                            "pool_timeout": True,
                            "max_duration_seconds": max_duration,
                            "session_ids": list(current_run_sessions.get() or [])}),
                    )
                )
                new_state = await asyncio.to_thread(
                    self.queue.mark_failed, item.id, error_msg, self.max_attempts)
                logger.error("[%s] timed out %s/%s after %.1fs → %s",
                             worker_id, item.source, item.kind, duration, new_state)
            except Exception as e:
                duration = time.monotonic() - started_perf
                completed_at = datetime.now(timezone.utc).isoformat()
                error_msg = f"{type(e).__name__}: {e}"
                await asyncio.to_thread(
                    partial(
                        self.queue.record_run,
                        run_id=run_id,
                        queue_id=item.id,
                        source=item.source,
                        status="failed",
                        started_at=started_at_iso,
                        completed_at=completed_at,
                        duration_seconds=duration,
                        summary=error_msg[:500],
                        task_id=_task_id_of(item),
                        meta_json=json.dumps({
                            "exception": type(e).__name__,
                            "session_ids": list(current_run_sessions.get() or [])}),
                    )
                )
                new_state = await asyncio.to_thread(
                    self.queue.mark_failed, item.id, error_msg, self.max_attempts)
                logger.error("[%s] failed %s/%s: %s → %s",
                             worker_id, item.source, item.kind, error_msg, new_state)
            finally:
                current_scope.reset(scope_token)
                current_effect_scope.reset(effect_token)
                current_run_sessions.reset(sessions_token)
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
