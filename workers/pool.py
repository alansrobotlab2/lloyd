"""Worker pool — N asyncio tasks draining the WorkQueue.

Sources register into SOURCE_REGISTRY (see workers/sources/__init__.py).
Each source provides:
  - NAME: str
  - async enqueue_if_due(queue, config) -> None | str  (see `_scheduler_pass`)
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
from datetime import datetime, timedelta, timezone
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
# What a source is allowed to declare. `interrupted` is deliberately absent: it
# is the one status the recovery sweep writes (#1137), and a status that names
# "the process holding this died" cannot be self-declared by the code whose run
# died — the whitelist exists so the terminal verdict of a run stays forgeable
# only by the side that actually observed the death.
RUN_STATUSES = ("success", "failed", "skipped")

# ── KV budget gate ────────────────────────────────────────────────────
#
# The 09-09 stall was long-lived agent loops evicting each other's prefixes.
# Every iteration re-submits a 100-200k context, and between iterations a
# turn's prefix sits only in the engine's free pool, where the next
# allocation can take it. Three or four of those resident at once on a 398k
# pool and one of them came back cold on almost every iteration
# (architecture/vllm.md). Short-lived jobs never came
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


# Round hold — see `WorkerPool._round_hold_held`.
#
# The source whose in-flight job holds the rest of the pool back, and the
# sources exempt from it. `scheduled-task` is exempt because those jobs are
# time-sensitive and short, and a user-visible schedule slipping by an hour
# is its own failure.
ROUND_SOURCE = "autocode"
# How stale the pool's answer to "is a landing running" may be. Claims come
# every second or two from every idle slot; a landing lasts minutes.
_LANDING_PROBE_SECONDS = 5.0
DEFAULT_ROUND_HOLD_EXEMPT: tuple[str, ...] = ("scheduled-task",)


def round_hold_config() -> dict[str, Any]:
    try:
        from app.config import CONFIG
        return dict((CONFIG.get("workers") or {}).get("round_hold") or {})
    except Exception:
        return {}


# Primary reachability hold — see `WorkerPool._primary_hold_held`.
#
# The two gates above ask how the engine is DOING. This one asks whether it is
# answering at all, and it exists because of 2026-09-23 19:35: the stack
# restarted, the backend came back first, and the pool spent the following
# minutes claiming autocode rounds against a primary that was still loading its
# 95.37 GiB host-RAM n-gram table. Every claim died with `All connection attempts
# failed` and each one was booked as an attempt against the item it claimed
# (#1220 and #654 went to draft inside two minutes; #1151 the same way on 09-18
# with no round ever opened). vLLM does not open its port until the model is
# resident, so "the engine is not answering /health" is not "the engine is slow"
# — it is "a turn started now cannot run", and the pool can learn that for one
# request, the same request a round is comfortably worth paying for.
#
# Claims come every second or two from every idle slot and a landing's restart
# lasts minutes, so the answer is cached and refreshed off the event loop: a
# claim must not wait on an HTTP call, which is also why the KV gate reads the
# background sampler instead of the engine.
#
# The three values below are what production runs on. `workers.primary_hold` in
# config.yaml (`enabled`, `sources`, `probe_seconds`, `probe_timeout_s`) is read
# over them when a person puts it there, and it is deliberately absent from the
# checked-in file: the gate that protects an item's attempt budget from this
# loop's own claims should not be a thing this loop can edit to disarm it. So
# `enabled: false` is an operator's kill switch, and adding the block is a human
# path (#1430's landing reports it).
_PRIMARY_PROBE_SECONDS = 10.0
# A local /health answers in milliseconds when the engine is up. The timeout
# bounds what a wedged listener can cost a claim — and a timeout counts as NOT
# answering, which holds the round rather than spending anything, so the cost of
# a low number here is a round that starts late, never an item that loses an
# attempt.
_PRIMARY_PROBE_TIMEOUT_S = 2.0
DEFAULT_PRIMARY_HOLD_SOURCES: tuple[str, ...] = (ROUND_SOURCE,)


def primary_hold_config() -> dict[str, Any]:
    try:
        from app.config import CONFIG
        return dict((CONFIG.get("workers") or {}).get("primary_hold") or {})
    except Exception:
        return {}


def primary_engine_answering(timeout: float = _PRIMARY_PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """Ask the primary engine's own `/health`, and report what came back.

    `(answering, detail)`. Blocking — call it through `asyncio.to_thread`. Any
    answer that is not HTTP 200 counts as not answering, including no answer:
    `promote._get` yields `(None, None)` for a refused connection or a timeout,
    which is the load-bearing case here, since an engine still loading its
    weights is unreachable rather than merely unhappy.

    The one answer that fails OPEN is not being able to ask the question at all.
    A broken probe must not become a stalled pool — the same rule the KV gate
    states for a missing reading ("a pressure signal that failed closed would
    turn an unreachable /metrics into a stopped worker pool"). An engine that
    answers 200 and then drops every stream is a different failure, and the
    ledger's own per-outage cap (`INFRA_RETRY_CAP`) is what bounds that one.
    """
    try:
        from scripts.automod.promote import PRIMARY_HEALTH, _get
    except Exception as exc:  # a missing probe is not evidence the engine is down
        return True, f"probe unavailable ({type(exc).__name__}: {exc})"
    status, _body = _get(PRIMARY_HEALTH, timeout=timeout)
    if status == 200:
        return True, "HTTP 200"
    if status is None:
        return False, "no answer (connection refused or timed out)"
    return False, f"HTTP {status}"


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


def _positive_int(value: Any) -> Optional[int]:
    """A config number that must be > 0, or None. A typo in `retry_seconds`
    costs the fast retry, never the scheduler pass."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# Where an operator pause is kept: the queue's own watermark table, so it lives
# in the same database as the rows it holds back.
PAUSE_WM_SOURCE = "_pool"
PAUSE_WM_KEY = "operator_paused"


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
        # Hand the ceiling to the queue, which is what enforces it. `claim_next`
        # and `recover_claimed` are where a row actually gets stopped, and both
        # live on the queue — which `get_queue()` builds before this pool exists,
        # and which the aggregator process builds without ever building a pool.
        # Without this line the configured `workers.max_attempts` would reach
        # `mark_failed` only and the crash-recovery path would enforce a default
        # rather than the number in the config.
        queue.set_max_attempts(max_attempts)
        self.poll_idle_seconds = poll_idle_seconds

        self._running = False
        # Two pauses, because they end differently. An OPERATOR pause (a person,
        # Mission Control, the guardian's vault trip) is persisted in the queue's
        # own database and survives a backend restart: it lived only in this
        # process until 2026-09-22, and three times a restart brought the pool
        # back running and it claimed 3-6 jobs, 4 of them autocode rounds, before
        # anyone could re-pause it. An AUTOMOD pause (the promoter's idle wait,
        # `round restart`) stays in memory, because the promoter relies on a
        # landing's own restart to clear it (`promote._POOL_PAUSED_BY_US`); made
        # durable, every landing would leave the pool paused for good.
        self._paused_operator = self._load_operator_pause()
        self._paused_automod = False
        self._workers: list[asyncio.Task] = []
        self._scheduler_task: Optional[asyncio.Task] = None
        self._in_flight: dict[int, dict[str, Any]] = {}
        self._landing_probe: tuple[float, bool] = (0.0, False)
        self._service_probe = None  # workers.service_probe.ServiceProbe, lazily
        # KV gate state, reported by `status()`. See `_kv_gate_held`.
        self._kv_gate: dict[str, Any] = {
            "engaged": False,
            "engaged_since": None,
            "engagements": 0,
            "kv_usage": None,
            "held_sources": [],
        }
        # Round hold. See `_round_hold_held`.
        self._round_hold: dict[str, Any] = {
            "engaged": False,
            "engaged_since": None,
            "engagements": 0,
            "held_sources": [],
        }
        # Primary reachability hold. See `_primary_hold_held`. `answering` is the
        # last probe's verdict (None until the first claim asks), and `last_probe`
        # is wall-clock so the cache age reads the same in a status dump as
        # `engaged_since` does.
        self._primary_hold: dict[str, Any] = {
            "engaged": False,
            "engaged_since": None,
            "engagements": 0,
            "held_sources": [],
            "answering": None,
            "last_detail": None,
            "last_probe": 0.0,
            "probing": False,
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
        # A crash also leaves poisoned rows nobody will ever look at again.
        # Triage them now rather than waiting out the first sweep interval —
        # boot is when the pile is most likely to be non-empty.
        await self._maybe_sweep_poisoned(force=True)

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

    def _load_operator_pause(self) -> bool:
        try:
            return self.queue.wm_get(PAUSE_WM_SOURCE, PAUSE_WM_KEY) == "1"
        except Exception:
            logger.exception("Worker pool: could not read the persisted pause; starting unpaused")
            return False

    def pause(self, paused: bool = True, owner: str = "operator") -> None:
        """Pause or resume. `owner` is `operator` (persisted) or `automod` (not).

        An automod resume lifts only its own pause, so a person's pause taken
        during a landing outlives the landing. An operator resume lifts both, as
        a resume always has: a person resuming the pool means it.
        """
        if owner == "automod":
            self._paused_automod = paused
        else:
            self._paused_operator = paused
            if not paused:
                self._paused_automod = False
            try:
                self.queue.wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY, "1" if paused else "0")
            except Exception:
                # The pause still holds for this process; only its survival of
                # a restart is lost, and that is worth a loud line, not a refusal.
                logger.exception("Worker pool: pause not persisted; a restart will clear it")
        logger.info("Worker pool %s by %s (operator=%s automod=%s)",
                    "paused" if paused else "resumed", owner,
                    self._paused_operator, self._paused_automod)

    @property
    def _paused(self) -> bool:
        return self._paused_operator or self._paused_automod

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def paused_by(self) -> list[str]:
        return [o for o, on in (("operator", self._paused_operator),
                                ("automod", self._paused_automod)) if on]

    def status(self) -> dict:
        return {
            "running": self._running,
            "paused": self._paused,
            "paused_by": self.paused_by,
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
            "round_hold": self.round_hold_status(),
            "primary_hold": self.primary_hold_status(),
        }

    def kv_gate_status(self) -> dict[str, Any]:
        cfg = kv_gate_config()
        window = float(cfg.get("window_seconds", DEFAULT_KV_GATE_WINDOW_S))
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "max_kv_usage": float(cfg.get("max_kv_usage", DEFAULT_KV_GATE_MAX)),
            "window_seconds": window,
            **self._kv_gate,
            # The reading now, not the one the last claim saw. The gate only
            # evaluates when a slot tries to claim, so with every slot busy
            # that value is as old as the longest job — on the first boot
            # after this landed it read the idle engine's 0% beside a card
            # showing 26%. `engaged` is still the last decision, which is
            # what the next claim will act on.
            "kv_usage": self._gate_reading(window),
            "held_sources": list(self._kv_gate["held_sources"]),
        }

    @staticmethod
    def _gate_reading(window: float) -> float | None:
        """The gate's input: the median over `window`, given a fresh sample."""
        if engine_pressure.latest() is None:
            return None
        return engine_pressure.kv_percentile(0.5, window=window)

    def round_hold_status(self) -> dict[str, Any]:
        cfg = round_hold_config()
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "exempt": list(cfg.get("exempt", DEFAULT_ROUND_HOLD_EXEMPT)),
            **self._round_hold,
            "held_sources": list(self._round_hold["held_sources"]),
        }

    def _landing_in_flight(self) -> bool:
        """Whether a landing (`S.rounds_landing`) or a land-train flush
        (`S.flush_in_progress`) is running, asked at most every
        `_LANDING_PROBE_SECONDS`.

        The hold used to end with the last round's turn — which is exactly
        when a waiting landing proceeds. On 2026-09-19 #608's landing waited
        430 s for a sibling turn; when it ended, `youtube-digest`,
        `board-steward` and `bench-mine` were released at 17:00:23-25, the
        landing paused the pool at 17:00:48, and then spent fourteen minutes
        waiting out jobs that had started seconds before it. What was held for
        the round stays held for the landing the round is waiting behind.
        Fails open: an unreadable state dir is not a landing.
        """
        now = time.monotonic()
        checked, value = getattr(self, "_landing_probe", (0.0, False))
        if now - checked < _LANDING_PROBE_SECONDS:
            return value
        try:
            from scripts.automod import state as S
            # A land-train flush is the restart a landing used to be: what
            # was held for it stays held until it is done.
            value = bool(S.rounds_landing()) or bool(S.flush_in_progress())
        except Exception:  # noqa: BLE001
            value = False
        self._landing_probe = (now, value)
        return value

    def _round_hold_held(self, registry: dict[str, Any]) -> list[str]:
        """Sources this claim must skip because an autocode round is running.

        A round is the rarest and most expensive thing this pool does: one
        worktree, a nine-rung gate, a landing that restarts the backend. It
        is also the job most damaged by sharing the primary, because it runs
        a 100k-200k-token context for an hour and re-submits all of it every
        iteration — so anything else claiming a slot evicts its prefix. On
        2026-09-11 every autocode session showed cold re-prefills (0.14-0.85M
        tokens per session), and in one 21-hour window 14 scheduled-task and
        15 autotriage runs shared the engine with the rounds.

        The KV gate beside this asks "is the engine full *now*"; this asks
        "is the one job worth protecting running at all". They are different
        questions and both are cheap.

        **No lease and no TTL**, deliberately. `_in_flight` is exact — it is
        written when a slot claims and popped on every exit path, including
        the timeout and exception branches — and it is bounded by the pool's
        own `wait_for`. A lease would add a second source of truth about the
        same fact, which is how a hold outlives the thing it was holding for.
        `autocode._loop_is_free` already guarantees at most one round.

        `scheduled-task` is exempt by default: those are time-sensitive, short,
        and a user-visible schedule slipping by an hour is its own failure.
        """
        cfg = round_hold_config()
        if not bool(cfg.get("enabled", True)):
            self._release_round_hold()
            return []
        running = {
            str(v.get("source") or "") for v in self._in_flight.values()
        }
        if ROUND_SOURCE not in running and not self._landing_in_flight():
            self._release_round_hold()
            return []
        exempt = set(cfg.get("exempt", DEFAULT_ROUND_HOLD_EXEMPT)) | {ROUND_SOURCE}
        held = sorted(name for name in registry if name not in exempt)
        hold = self._round_hold
        if not hold["engaged"]:
            hold.update(
                engaged=True, engagements=hold["engagements"] + 1,
                engaged_since=datetime.now(timezone.utc).isoformat(),
                held_sources=held,
            )
            logger.info(
                "Round hold engaged: an %s round is in flight — holding %s",
                ROUND_SOURCE, ", ".join(held) or "nothing",
            )
        else:
            hold["held_sources"] = held
        return held

    def _release_round_hold(self) -> None:
        hold = self._round_hold
        if hold["engaged"]:
            logger.info("Round hold released")
            hold.update(engaged=False, engaged_since=None, held_sources=[])

    def primary_hold_status(self) -> dict[str, Any]:
        """What the primary reachability hold reports, and what a claim costs.

        A pure read of the last decision, like `kv_gate_status` and
        `round_hold_status`: the pool polls claims every second or two, so this
        runs far more often than the question changes meaning, and a status call
        that probed the engine itself would make the dashboard the reason a claim
        waited on HTTP. `last_detail` is the probe's own words — the answer that
        decided the hold, not a restatement of it.
        """
        cfg = primary_hold_config()
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "sources": sorted(cfg.get("sources", DEFAULT_PRIMARY_HOLD_SOURCES)),
            "probe_seconds": float(cfg.get("probe_seconds", _PRIMARY_PROBE_SECONDS)),
            "probe_timeout_s": float(cfg.get("probe_timeout_s", _PRIMARY_PROBE_TIMEOUT_S)),
            **self._primary_hold,
            "held_sources": list(self._primary_hold["held_sources"]),
        }

    async def _primary_hold_held(self, registry: dict[str, Any]) -> list[str]:
        """Sources this claim must skip because the primary engine is not answering.

        The third gate. The KV gate needs a /metrics reading to exist and the
        round hold needs a round already in flight, so while the engine is still
        loading both are silent and every claim goes out and dies: on
        2026-09-23 19:35 that was three autocode claims inside 65 seconds against
        a primary that was not listening, and the ledger booked each one as an
        attempt on the item it claimed. `LONG_LIVED` is what decides, because a
        re-admitting loop is the job an outage cannot repair by retrying — and a
        source that is never claimed keeps its attempt intact, which is the point:
        this gate protects the attempt ledger, not the engine.

        The probe is cached for `probe_seconds` and refreshed off the event loop,
        so a claim never waits on HTTP. A probe already in flight is not joined:
        the previous answer stands while it runs, which keeps one slow health
        endpoint from parking every idle slot at once.

        Fails open on a broken probe — see `primary_engine_answering` — and logs
        only on a transition, so an engine down for an hour is two lines.
        """
        cfg = primary_hold_config()
        st = self._primary_hold
        if bool(cfg.get("enabled", True)) and not st["probing"] and \
                time.time() - float(st["last_probe"] or 0.0) >= \
                float(cfg.get("probe_seconds", _PRIMARY_PROBE_SECONDS)):
            st["probing"] = True
            try:
                answering, detail = await asyncio.to_thread(
                    primary_engine_answering,
                    float(cfg.get("probe_timeout_s", _PRIMARY_PROBE_TIMEOUT_S)))
            except Exception as exc:  # noqa: BLE001 — a crashed probe is not engine-down
                answering, detail = True, f"probe raised {type(exc).__name__}: {exc}"
            finally:
                st["probing"] = False
                # Stamped after the wait, so the next probe is `probe_seconds`
                # from the ANSWER and not from the request that started it: a
                # health endpoint that takes its full timeout would otherwise be
                # re-asked by every idle slot on every poll.
                st["last_probe"] = time.time()
                st["answering"], st["last_detail"] = answering, detail

        engaged = bool(cfg.get("enabled", True)) and st["answering"] is False
        held = sorted(set(long_lived_sources(registry))
                      & set(cfg.get("sources", DEFAULT_PRIMARY_HOLD_SOURCES))) \
            if engaged else []
        if engaged:
            st["held_sources"] = held
            if not st["engaged"]:
                st.update(engaged=True, engagements=int(st["engagements"]) + 1,
                          engaged_since=datetime.now(timezone.utc).isoformat())
                logger.warning("Primary hold engaged: the engine is not answering (%s) — holding %s",
                               st["last_detail"], ", ".join(held) or "nothing")
        elif st["engaged"]:
            logger.info("Primary hold released: the engine answers again (%s)", st["last_detail"])
            st.update(engaged=False, engaged_since=None, held_sources=[])
        return held

    async def _claim_holds(self, registry: dict[str, Any]) -> list[str]:
        """Every reason this claim must skip a source, unioned.

        Three independent gates, and the union is what the claim query takes.
        Kept as one call site so a third gate has an obvious home — which is the
        home the primary reachability hold just took.

        Async because the third gate asks the engine a question. The other two
        read a background sampler and a cached marker precisely so a claim does
        not wait on the network, and `_primary_hold_held` keeps that promise by
        serving a cached verdict and refreshing it on a thread.
        """
        held = set(await self._primary_hold_held(registry))
        held |= set(self._kv_gate_held(registry))
        held |= set(self._round_hold_held(registry))
        return sorted(held)

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
        kv = self._gate_reading(window) if enabled else None
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
        # Defined outside the try, because the `sleep` at the bottom is
        # outside it too: if the first `get_sources_config()` raised, the
        # except branch fell through to `sleep(interval)` and died on a
        # NameError, taking the only thing that enqueues work with it.
        interval = 60
        while self._running:
            await self._maybe_sweep_poisoned()
            await self._probe_services()

            try:
                await self._scheduler_pass()
            except Exception as e:
                logger.error("Scheduler loop error: %s", e, exc_info=True)
            await asyncio.sleep(interval)

    async def _scheduler_pass(self) -> None:
        """One pass over the sources: call each one that is due, stamp it.

        **A declined check is not a spent interval.** The watermark used to
        be stamped after every call, so a source that looked and could not
        act yet — autocode with a round still under observation — waited its
        whole `interval_seconds` (900 s) before looking again. The loop's
        median gap from a round ending to the next one starting was 9.6–14.5
        min, ~12 h of a free loop over fifty rounds. A source that returns
        `DECLINED` and has `retry_seconds` configured is stamped back-dated
        so it is due again in `retry_seconds`; everything else is stamped
        now, as before.
        """
        from workers.sources import DECLINED, SOURCE_REGISTRY, get_sources_config

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
            outcome = None
            try:
                outcome = await source.enqueue_if_due(self.queue, src_cfg)
            except Exception as e:
                logger.error("Source %s enqueue_if_due failed: %s", name, e, exc_info=True)
            stamp = datetime.now(timezone.utc)
            retry = _positive_int(src_cfg.get("retry_seconds"))
            if outcome == DECLINED and retry and retry < wait:
                stamp -= timedelta(seconds=wait - retry)
            await asyncio.to_thread(
                self.queue.wm_set, name, "last_enqueue_check", stamp.isoformat())

    # ── Queue maintenance — poison sweep ─────────────────────────────────

    async def _maybe_sweep_poisoned(self, force: bool = False) -> None:
        """Triage poisoned items on the maintenance interval.

        Deliberately driven from the scheduler loop rather than registered as a
        work source: a source that repairs the queue must be claimed by a free
        worker slot, so it would be starved exactly when the queue is backed
        up. See workers/maintenance.py for what a sweep decides.

        Never raises — a failed sweep must not take the scheduler loop (and
        with it every source's enqueue tick) down with it.
        """
        try:
            from app.config import CONFIG
            from workers import maintenance

            cfg = (CONFIG.get("workers") or {}).get("maintenance") or {}
            if not cfg.get("enabled", maintenance.DEFAULTS["enabled"]):
                return

            if not force:
                interval = int(cfg.get("interval_seconds",
                                       maintenance.DEFAULTS["interval_seconds"]))
                last = self.queue.wm_get(maintenance.SOURCE, "last_sweep_at")
                if last:
                    try:
                        elapsed = (datetime.now(timezone.utc)
                                   - datetime.fromisoformat(last)).total_seconds()
                    except ValueError:
                        elapsed = interval  # unparseable watermark: sweep and rewrite it
                    if elapsed < interval:
                        return

            await asyncio.to_thread(
                maintenance.run_sweep, self.queue, cfg, self.max_attempts
            )
        except Exception as e:
            logger.error("Poison sweep failed: %s", e, exc_info=True)

    async def _probe_services(self) -> None:
        """Announce a supervised program whose port never opens (#1359).

        Same seat as the poison sweep and for the same reason: it must run
        whether or not a worker slot is free. Never raises.
        """
        try:
            from workers import service_probe

            if self._service_probe is None:
                self._service_probe = service_probe.ServiceProbe(
                    announce=service_probe.guardian_announce)
            await asyncio.to_thread(service_probe.run_probe, self._service_probe)
        except Exception as e:
            logger.error("Service probe failed: %s", e, exc_info=True)

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

            # Every gate that decides what may start, unioned — including the
            # one that asks the engine whether it is answering at all, which is
            # what stops an autocode round being claimed against a primary that
            # is still loading and then booked as an attempt it never used.
            held = await self._claim_holds(SOURCE_REGISTRY)
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

            # The id is minted BEFORE the claim is stamped running, and stamped
            # onto the row with it, so it outlives this attempt: when a cancel
            # or a SIGKILL leaves no `runs` row, the recovery sweep writes the row
            # under THIS id (#1137) — the same one the `[worker-N] running …
            # run_id=…` line below already put in the log, so the run the log
            # names is a run the table can be queried for.
            run_id = new_run_id(item.source)
            await asyncio.to_thread(self.queue.mark_running, item.id, run_id)
            started_at_iso = datetime.now(timezone.utc).isoformat()
            started_perf = time.monotonic()
            self._in_flight[item.id] = {
                "source": item.source,
                "kind": item.kind,
                "started_at": started_at_iso,
                "worker": worker_id,
            }

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
                await self._repoll_on_complete(source)

    async def _repoll_on_complete(self, source) -> None:
        """Make a source due at the next scheduler pass once its run ends.

        Nothing woke a source when its own job finished: the next look was
        whenever its watermark said, up to `interval_seconds` later. For
        autocode that is the gap between one round ending and the next being
        queued — a `skipped` run that took milliseconds, or a round shorter
        than the interval, left a free loop idle for most of 15 minutes. A
        source opts in with `REPOLL_ON_COMPLETE = True`; the watermark is
        back-dated to the epoch, so the scheduler (a 60 s pass) asks it again.
        Failing to write it costs the early look, never the run's record.
        """
        if not getattr(source, "REPOLL_ON_COMPLETE", False):
            return
        name = getattr(source, "NAME", None)
        if not name:
            return
        try:
            await asyncio.to_thread(self.queue.wm_set, name, "last_enqueue_check",
                                    datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat())
        except Exception as e:  # noqa: BLE001
            logger.warning("could not re-arm %s after its run: %s", name, e)


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
