"""Is each dispatch source still ASKING? (#1681)

The pool calls every source's ``enqueue_if_due`` on its own interval and stamps
``last_enqueue_check`` **whether or not that call returned** — ``workers/pool.py``
catches every exception from it (one bad source must not stop the fleet) and then
advances the stamp as if the check had succeeded. The attempt stamp has to keep
advancing: the interval arithmetic that decides who is due reads it, and a source
that raises every tick must not be retried every second. But that leaves it unable
to say anything about whether dispatch *works*. A source raising on every tick
reads fresher than one merely between its intervals.

The blind spot is not cosmetic, though #1682 narrowed it. Three of the four
alarms that used to live *inside* the one call whose failure is being swallowed
— ``_grossly_overdue``, ``_next_run_stalled`` and
``_scan_unparseable_task_files`` — now run from
``workers/fleet_watchdog.py`` on the scheduler loop's own seat, before any
source is consulted, so a source that raises every tick no longer takes the
fleet's stall alarms down with it. ``_note_vllm_outage`` is still called from
``scheduled_task.enqueue_if_due`` and is still swallowed with everything after
the gate (#1683 owns that alarm). What this module watches is therefore the
half that has no other signal: dispatch itself, whose silence looks exactly
like an idle fleet. Nor does anything read
the stamp: outside this module the key is written in two places in ``pool.py``,
cited in tests as interval arithmetic, and named in architecture prose — nowhere
does a reader judge its age. And the health route cannot cover it either:
``app/routers/workers.py`` builds each source's row from the ``runs`` table, so a
source that has stopped enqueueing entirely keeps showing the shape of its last
run. A stale-watermark check is the only signal that survives the class of failure
where the alarm code itself is what throws.

So the pool stamps a second key, ``last_enqueue_ok``, only on a clean return, and
this module is the reader. It runs from the scheduler loop *outside* the try that
swallows a source's exception, beside the poison sweep and the service probe, so
it keeps running when every source raises at once. Judgement lives in ``verdict``
— one pure function — so the loop's announcement and the health route's row are
the same reading of the same two stamps.

The delivery is ``DispatchWatch``: announce-once on a crossing, announce again on
recovery, no ledger row, no auto-created task. It is ``workers/service_probe.py``
(#1359) applied to the scheduler instead of to a port, including the part that
makes it livable: a grace window in which a source that recovers is never
announced at all. Here the window is a multiple of the source's own
``interval_seconds``, which is why it is a multiple and not a constant —
``scheduled-task`` ticks every 60 s and ``autocode`` every 900 s, so a fixed
threshold would be far too twitchy for one and far too slow for the other.

Two false positives the threshold must not create, both pinned by tests in
``tests/test_workers_pool.py``: a source whose class declares
``REPOLL_ON_COMPLETE`` has its *attempt* stamp deliberately re-armed to the epoch
after every finished run (``pool._repoll_on_complete``), which makes that stamp
permanently stale on a source that is working, so the clean stamp is the only one
this can read; and a source disabled in config is skipped before the pool reads
any stamp, so its stamps go stale forever by design and are never judged.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Watermark keys. ``last_enqueue_check`` is the pool's attempt stamp and is what
#: the interval arithmetic reads; ``last_enqueue_ok`` is written only when
#: ``enqueue_if_due`` returned, and is the only one either of them may trust about
#: whether dispatch works.
ATTEMPT_WM_KEY = "last_enqueue_check"
OK_WM_KEY = "last_enqueue_ok"

#: A source is stalled when its clean stamp is older than this many of its own
#: dispatch intervals. Three is the smallest multiple that cannot fire on one
#: skipped interval plus one slow tick, and a multiple rather than a duration
#: because the 60-second source and the 900-second one have to be judged on their
#: own clocks: 180 s for ``scheduled-task``, 2700 s for ``autocode``.
STALL_FACTOR = 3.0

#: The interval a source is judged on when its config carries no
#: ``interval_seconds``. The same 3600 the pool applies at ``workers/pool.py``
#: when it decides who is due — written here rather than imported because the
#: pool reads its own literal, and the two are pinned equal by
#: ``tests/test_workers_pool.py`` so a change to one fails a test.
DEFAULT_INTERVAL_S = 3600

#: The states this reports. ``stalled`` and ``never_succeeded`` are the two that
#: set ``stalled: True``. The rest are each a reason a source is NOT called
#: stalled, distinct from one another so a reader can tell "healthy" from "not
#: judgeable": ``pending`` (asked, not yet answered cleanly, inside the window),
#: ``disabled`` (config says don't run it), ``unmeasured`` (no stamp at all, or a
#: stamp that will not parse, or a watermark read that failed). Every state here
#: is returned by some path.
STATE_OK = "ok"
STATE_STALLED = "stalled"
STATE_NEVER_SUCCEEDED = "never_succeeded"
STATE_PENDING = "pending"
STATE_DISABLED = "disabled"
STATE_UNMEASURED = "unmeasured"


def _parse(value: Optional[str]) -> Optional[datetime]:
    """Parse a watermark stamp, or None if there is none or it is not an ISO one.

    Both stamps are written ``datetime.now(timezone.utc).isoformat()`` by the pool.
    A value that does not parse is reported as ``unmeasured`` rather than passed
    over in silence: the failure this module exists to catch is a stamp that stops
    meaning anything, and treating a corrupt one as "no stamp" would let the same
    blind spot back in through the side door.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def verdict(*, ok_stamp: Optional[str], attempt_stamp: Optional[str],
            interval_seconds: Any, enabled: Any, now_epoch: float,
            repoll_on_complete: bool = False,
            factor: float = STALL_FACTOR) -> dict[str, Any]:
    """Judge one source from its two stamps. Pure: nothing here reads a clock.

    A source whose interval is not a usable number is ``unmeasured`` rather than
    judged against a guessed threshold, and a source with no stamps is
    ``unmeasured`` too — a pool that has never reached a source has not stalled,
    it has not started, and those read the same until one of them has a stamp.
    """
    base: dict[str, Any] = {
        "stalled": False, "age_seconds": None, "threshold_seconds": None,
        "interval_seconds": None, "repoll_on_complete": repoll_on_complete,
    }
    if not enabled:
        # The pool skips a disabled source before it reads any stamp, so its
        # stamps go stale forever. Not a stall, and a distinct state so a reader
        # cannot read "disabled" as "healthy".
        return {**base, "state": STATE_DISABLED,
                "detail": "source disabled in config; its stamps are not judged"}

    try:
        interval = float(interval_seconds)
    except (TypeError, ValueError):
        interval = 0.0
    if interval <= 0:
        return {**base, "state": STATE_UNMEASURED,
                "detail": f"interval_seconds={interval_seconds!r} is not a positive "
                          f"number, so there is no threshold to judge it against"}
    base["interval_seconds"] = interval
    threshold = interval * factor
    base["threshold_seconds"] = threshold

    if repoll_on_complete:
        # A source that re-arms its attempt stamp to the epoch after every run
        # cannot be judged by that stamp at all; the clean stamp below is the only
        # one that means anything for it. Recorded so a reader can see WHY the
        # attempt stamp reads ancient on a source that is working.
        base["repoll_on_complete"] = True

    ok_dt = _parse(ok_stamp)
    if ok_dt is None:
        if ok_stamp is not None:
            return {**base, "state": STATE_UNMEASURED,
                    "detail": f"{OK_WM_KEY}={ok_stamp!r} is not an ISO timestamp"}
        if repoll_on_complete:
            # For this kind of source the attempt stamp reads the epoch BOTH
            # before its first-ever dispatch and after every completed run, so
            # with no clean stamp there is nothing here to read either way —
            # calling it `never_succeeded` and stalled would be a claim built on
            # a stamp that carries no information.
            return {**base, "state": STATE_UNMEASURED,
                    "detail": "no clean stamp yet, and this source re-arms "
                              f"{ATTEMPT_WM_KEY} to the epoch, so the attempt "
                              "stamp carries no age to judge"}
        if attempt_stamp is None:
            return {**base, "state": STATE_UNMEASURED,
                    "detail": "no dispatch recorded for this source yet"}
        att_dt = _parse(attempt_stamp)
        if att_dt is None:
            return {**base, "state": STATE_UNMEASURED,
                    "detail": f"{ATTEMPT_WM_KEY}={attempt_stamp!r} is not an ISO "
                              f"timestamp"}
        age = max(0.0, now_epoch - att_dt.timestamp())
        if age > threshold:
            return {**base, "age_seconds": age, "state": STATE_NEVER_SUCCEEDED,
                    "stalled": True,
                    "detail": (f"every enqueue attempt in the last {age:.0f}s raised; "
                               f"no clean return since this pool started")}
        return {**base, "age_seconds": age, "state": STATE_PENDING,
                "detail": "no clean return yet, inside the window"}

    age = max(0.0, now_epoch - ok_dt.timestamp())
    if age > threshold:
        return {**base, "age_seconds": age, "state": STATE_STALLED, "stalled": True,
                "detail": (f"no clean enqueue in {age:.0f}s, over the {threshold:.0f}s "
                           f"window ({factor:g} x {interval:.0f}s interval)")}
    return {**base, "age_seconds": age, "state": STATE_OK, "stalled": False,
            "detail": "clean enqueue inside the window"}


class DispatchWatch:
    """Announce-once over ``verdict``, for the scheduler loop to tick.

    Built like ``ServiceProbe`` and for the same reason: a person sees the moment
    dispatch stops and sees it recover, and nothing is written to the ledger. The
    threshold is a multiple of the source's own interval, which is itself the
    grace window; a source that recovers inside it was never announced at all.

    ``sources`` may be a callable returning the per-source config mapping, which
    is how the pool passes it — ``get_sources_config()`` re-reads CONFIG per call
    so that ``/api/workers/enable`` and the Tools page take effect without a
    restart. A snapshot taken here would freeze ``enabled``, and ``enabled`` is one
    of the two inputs that decide whether a source is judged at all. A plain
    mapping is accepted too, for a caller that has one in hand.

    ``repoll`` is the set of source names whose class declares
    ``REPOLL_ON_COMPLETE``, the one input here that cannot come from config. Left
    ``None`` it is read from ``SOURCE_REGISTRY`` on first use, through a
    function-local import: ``workers.sources`` imports ``workers.pool``, which
    imports this module, so a module-level import of the registry here would be a
    cycle. A caller with the set in hand — a test running fake sources — passes it
    and is not at the mercy of what the real registry holds.
    """

    def __init__(self, *, queue: Any,
                 sources: "Callable[[], dict[str, dict]] | dict[str, dict]",
                 now_fn: Optional[Callable[[], float]] = None,
                 factor: float = STALL_FACTOR,
                 announce: Optional[Callable[[str, str, str], Any]] = None,
                 repoll: Optional[set[str]] = None) -> None:
        self._queue = queue
        self._sources = sources if callable(sources) else (lambda: sources)
        self._now = now_fn
        self._factor = factor
        self._announce = announce
        self._stalled: set[str] = set()
        self._repoll = repoll

    def _repoll_sources(self) -> set[str]:
        """Names whose declared re-arm makes the attempt stamp meaningless.

        Cached on first use: a source is a registered class, so its declaration
        cannot change without an edit to the tree and a restart.
        """
        if self._repoll is None:
            from workers.sources import SOURCE_REGISTRY
            self._repoll = {n for n, s in SOURCE_REGISTRY.items()
                            if getattr(s, "REPOLL_ON_COMPLETE", False)}
        return self._repoll

    def _source_map(self) -> dict[str, dict]:
        """The per-source config, or ``{}`` if it cannot be read.

        ``{}`` rather than an exception: the caller is the health route, and a
        config read that fails has already lost the interval for every source, so
        there is nothing left to judge and no row worth asserting.
        """
        try:
            return self._sources() or {}
        except Exception as e:  # noqa: BLE001 - a config read is not the watch's job
            logger.warning("dispatch_watch: source config unreadable: %s", e)
            return {}

    def rows(self) -> dict[str, dict[str, Any]]:
        """Every configured source's verdict, for the health route and the tick.

        Never raises for one source's sake: a watermark read that fails on one
        source must not cost the operator the other seven, so that source alone
        becomes ``unmeasured`` with the error in its detail. That is the same state
        ``verdict`` returns for a stamp that will not parse, and shared on purpose
        — both say *this watch cannot judge this source*, which is deliberately not
        the same as healthy.
        """
        ts = self._now() if self._now is not None else time.time()
        repoll = self._repoll_sources()
        out: dict[str, dict[str, Any]] = {}
        for name, cfg in self._source_map().items():
            cfg = cfg if isinstance(cfg, dict) else {}
            try:
                ok_stamp = self._queue.wm_get(name, OK_WM_KEY)
                attempt_stamp = self._queue.wm_get(name, ATTEMPT_WM_KEY)
            except Exception as e:  # noqa: BLE001 - the operator keeps the other rows
                out[name] = {"name": name, "state": STATE_UNMEASURED, "stalled": False,
                             "age_seconds": None, "threshold_seconds": None,
                             "interval_seconds": None,
                             "repoll_on_complete": name in repoll,
                             "detail": f"watermark read failed: {e}"}
                continue
            row = verdict(ok_stamp=ok_stamp, attempt_stamp=attempt_stamp,
                          interval_seconds=cfg.get("interval_seconds",
                                                   DEFAULT_INTERVAL_S),
                          # Default False, which is the pool's own at
                # `workers/pool.py:944` (`if not src_cfg.get("enabled", False):
                # continue`) and the health route's at
                # `app/routers/workers.py`. A source that does not say it is on is
                # not dispatching, so it must not be judged for not dispatching:
                # defaulting this True would invent stalls for whatever config
                # omits the key.
                enabled=bool(cfg.get("enabled", False)),
                          now_epoch=ts, repoll_on_complete=name in repoll,
                          factor=self._factor)
            row["name"] = name
            out[name] = row
        return out

    def tick(self) -> list[str]:
        """Judge every source and announce the crossings. Returns event lines.

        The crossing rule is what keeps a steady stall from becoming a recurring
        alarm: the first pass that measures a stall announces it, and every later
        one is silent until the source returns cleanly. A source with no stamps is
        silent, so booting the pool is not an alarm; a source already measured
        stalled IS announced on the first pass after a restart, because the clean
        stamp outlives the pool and a restart must not hide a stall that was
        already running.
        """
        events: list[str] = []
        for name, row in self.rows().items():
            if row["state"] == STATE_UNMEASURED:
                # Not silent: this is the one state where the watch cannot do its
                # job, and a silent unmeasurable source is the blind spot this
                # module exists to close. A warning per source, not a per-source-
                # per-tick announce, because it will repeat every 60 s.
                logger.warning("dispatch_watch: cannot judge %s: %s",
                               name, row["detail"])
                continue
            if row["stalled"] == (name in self._stalled):
                continue  # no crossing: a steady stall stays quiet
            if row["stalled"]:
                self._stalled.add(name)
                text = f"dispatch stalled: {name} — {row['detail']}"
            else:
                self._stalled.discard(name)
                text = f"dispatch recovered: {name} — {row['detail']}"
            events.append(text)
            # warning, not error, matching the announce level below: this is an
            # announcement about a stalled dispatcher, and the log level is what a
            # person grepping for a *fault* would read.
            (logger.warning if row["stalled"] else logger.info)(
                "dispatch_watch: %s", text)
            if self._announce is not None:
                try:
                    self._announce(
                        text,
                        f"Threshold {row['threshold_seconds']:.0f}s = "
                        f"{self._factor:g} x this source's own interval_seconds of "
                        f"{row['interval_seconds']:.0f}. Announcement, not alert: no "
                        f"ledger row, and never a rollback trigger — the alarms that "
                        f"would judge a run live inside the very call this watch "
                        f"exists to watch (#1681).",
                        # `warn`, never `critical`: notify.py's gate at :195 makes
                        # a critical level the ledger-and-ALERT-and-auto-backlog
                        # path, which is exactly the routing this module's announce
                        # choice (#1681 owed 2) is not taking.
                        "warn" if row["stalled"] else "info")
                except Exception as e:  # noqa: BLE001 - a bell is not the watch
                    logger.warning("dispatch_watch: announce %s failed: %s", name, e)
        return events


def guardian_announce(title: str, body: str, level: str) -> Any:
    """The guardian's one fan-out, borrowed from ``workers.service_probe``.

    Re-exported rather than re-implemented: that module already resolves the
    guardian's ``notify.py`` path, tolerates its absence, and returns the channel
    receipt. Two copies of a shim that finds a program's alert script is two
    copies that can drift on where the alerts go.
    """
    from workers.service_probe import guardian_announce as _announce

    return _announce(title, body, level)


def dispatch_health(queue: Any, sources: "dict[str, dict] | None" = None,
                    now: float | None = None) -> dict[str, dict]:
    """One judged row per source, for ``GET /api/workers/health`` (#1681 clause 5).

    The route's per-source ``depth`` and ``health`` come off the ``runs`` table, so
    a source that has stopped enqueueing keeps showing the shape of its last run —
    and the fleet watchdog that would notice is itself dispatched by the source it
    would catch. This is the row that answers "is this source still ASKING", and it
    is delegated to ``DispatchWatch.rows()``: the same ``verdict``, the same
    ``REPOLL_ON_COMPLETE`` set, the same disabled skip, so the dashboard a person
    reads and the announcement a person gets cannot disagree about one source.

    Cheap enough to compute inside a request: one watermark read per source, no
    lock, no write. An unreachable database raises out of ``wm_get``, and the route
    on the other side of that already has a failure path.
    """
    def _sources() -> dict[str, dict]:
        if sources is not None:
            return sources
        from workers.sources import get_sources_config
        return get_sources_config()

    return DispatchWatch(queue=queue, sources=_sources,
                         now_fn=(None if now is None else (lambda: float(now)))).rows()
