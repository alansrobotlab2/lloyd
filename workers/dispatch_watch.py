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
announced at all. The threshold is a multiple of the source's own
``interval_seconds``, which is why it is a multiple and not a constant —
``scheduled-task`` ticks every 60 s and ``autocode`` every 900 s, so a fixed
threshold would be far too twitchy for one and far too slow for the other.

On top of that window sits a per-process rail (#3061), and it is the same one
``ServiceProbe`` holds at ``workers/service_probe.py:165-168``: the FIRST measurement
a watch takes in a process announces nothing, because the stamps it reads on a boot
were written by the process that just died and their age measures that one's lifetime,
not whether dispatch works. ``_scheduler_loop`` ticks this watch before
``_scheduler_pass`` has stamped anything, so that first tick is the only kind a restart
produces. Over the retained logs from 2026-10-16 05:55 to 2026-10-19 23:07 that put 28
of 34 stall events within 60 s of a ``Worker pool started`` line with no
``enqueue_if_due failed`` anywhere in the window: a signal that fires on every deploy
is one a person learns to dismiss. A stall still standing at the SECOND measurement is
announced there — one scheduler interval late, never lost — and a stall that was never
announced cannot later report its own recovery. A pair of stamps that have BOTH aged
past the window is the remaining downtime case and is reported as ``downtime``, not as
a stall: a source whose every enqueue raises keeps its attempt stamp fresh, so nothing
asking at all means the pool was not running.

False positives the reading must not create, each pinned by a node in
``tests/test_workers_pool.py``: a source whose class declares
``REPOLL_ON_COMPLETE`` has its *attempt* stamp deliberately re-armed to the epoch
after every finished run (``pool._repoll_on_complete``), which makes that stamp
permanently stale on a source that is working, so the clean stamp is the only one
this can read — which is also why the downtime test below has to skip those sources
rather than merely borrow their exemption; and a source disabled in config is skipped
before the pool reads any stamp, so its stamps go stale forever by design and are
never judged.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Watermark keys. ``last_enqueue_check`` is the pool's attempt stamp and is what
#: the interval arithmetic reads; ``last_enqueue_ok`` is written only when
#: ``enqueue_if_due`` returned, and is the only one either of them may trust about
#: whether dispatch works.
ATTEMPT_WM_KEY = "last_enqueue_check"
OK_WM_KEY = "last_enqueue_ok"

#: A source is stalled when its clean stamp is older than this many of its own
#: dispatch intervals, and a multiple rather than a duration because the 60-second
#: source and the 900-second one have to be judged on their own clocks: 180 s for
#: ``scheduled-task``, 2700 s for ``autocode``.
#:
#: Three is held BY RULING as much as by derivation, and its derivation is
#: falsified on this box: 3 x 60 s does fire on one skipped interval plus one slow
#: tick. Six events between 2026-10-17 and 2026-10-19 sat at 186-209 s with no boot
#: and no failed enqueue behind them, each recovering on the next pass — a
#: whole-process block of ~190 s is enough. #3061's first-measurement rail keeps
#: those transient ones off the page, and owed-check
#: `20261025_060328_owedcheck_b7fe` ruled on 2026-10-25 that the rest of them get NO
#: alarm: the streak rail in ``DispatchWatch.tick`` forgets a source whose age reads
#: clean at the next measurement (`self._since.pop(name)`), so only a block measured
#: twice past the window can announce. Raising the factor to 4 would
#: push the 60 s source's real-stall delay from 180 s to 240 s to buy silence the
#: grace already provides. A block long enough to be measured twice past the window
#: (>=240 s here) is a different failure — the loop did not run at all — and belongs
#: to a watchdog outside the loop, not to this threshold. Prospective rather than
#: measured: 0 log gaps >=150 s in the 4.2 post-landing days to 2026-10-25.
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
#: stamp that will not parse, or a watermark read that failed), and ``downtime`` —
#: added for #3061 — where BOTH stamps are past the window, so nothing has asked
#: either and the pool itself was not running. Not a stall, and not silent-by-
#: accident: it is the answer to the question the stale stamp was being misread as.
#: Every state here is returned by some path.
STATE_OK = "ok"
STATE_STALLED = "stalled"
STATE_NEVER_SUCCEEDED = "never_succeeded"
STATE_PENDING = "pending"
STATE_DISABLED = "disabled"
STATE_UNMEASURED = "unmeasured"
STATE_DOWNTIME = "downtime"


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

    A stale clean stamp is read on its own; the attempt stamp is consulted only to
    decide whether the missing clean return is a fault at all. It is not, when that
    stamp has aged equally far: nothing has asked, which is ``downtime`` and silent,
    and it is ``stalled`` the moment the pool is asking again without getting an
    answer.
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
        # Both stamps past the window is a different event from one of them, and the
        # two need different words (#3061). ``pool.py`` advances the attempt stamp on
        # EVERY outcome, the raise included (``workers/pool.py:1635-1636``, sitting
        # after the ``except`` at :1628-1630), so a source that is being asked and
        # failing to answer keeps that stamp fresh — while a pool that was stopped,
        # halted or simply restarting asked nothing and let both stamps age together.
        # The DECLINED back-date at :1633-1634 cannot fake the second shape: it moves
        # the attempt stamp by ``interval - retry``, never the ``factor`` the window
        # needs. Sources that re-arm to the epoch are excluded outright, because for
        # them this test would be permanently true and would swallow their real
        # stalls — see the ``REPOLL_ON_COMPLETE`` paragraph in this module's docstring.
        att_dt = _parse(attempt_stamp)
        if not repoll_on_complete and att_dt is not None:
            att_age = max(0.0, now_epoch - att_dt.timestamp())
            if att_age > threshold:
                return {**base, "age_seconds": age, "state": STATE_DOWNTIME,
                        "stalled": False,
                        "detail": (f"downtime, not a stall: nothing was even "
                                   f"attempted in {att_age:.0f}s and no clean return "
                                   f"in {age:.0f}s, both past the {threshold:.0f}s "
                                   "window — the pool was not asking this source, so "
                                   "its silence is the pool's, not the source's")}
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

    Two rails keep the announcements worth reading, and the second is the one the
    first cannot do alone. The threshold judges each source on its own clock; this
    class additionally withholds an announcement until a stall has been MEASURED twice
    (#3061) — the tick that opens a streak only records its start, and the recovery is
    announced only for a stall that was announced. That covers the boot, where the
    first tick is judging stamps the previous process left, and the whole-process blocks
    this box also produces at ~190 s, which are already gone on the next measurement.
    A stall's streak record is per source and lives only while the stall stands, so a
    fault that persists is announced on its second measurement, one scheduler interval
    after it was found, and never lost. Both dicts here are bounded by the number of
    configured source names.

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
                 repoll: Optional[set[str]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._queue = queue
        self._sources = sources if callable(sources) else (lambda: sources)
        self._now = now_fn
        self._factor = factor
        self._announce = announce
        self._stalled: set[str] = set()
        self._repoll = repoll
        # Monotonic, on purpose and separately from `now_fn`. `now_fn` answers "how old
        # is that stamp", so it has to be wall-clock: it is subtracted from an ISO
        # timestamp written by another process. `clock` answers "how long has THIS
        # process been watching the fault", and a wall clock cannot answer that — an NTP
        # step or a boot from a wrong clock would move a stamp's age by hours, and the
        # rail built on it would either announce on the first measurement or never.
        # Same split as `ServiceProbe.clock` (workers/service_probe.py:138), the
        # precedent #3061 cites. `_since` is keyed by configured source name and loses
        # an entry the moment that source stops being stalled, so it cannot grow.
        self._clock = clock
        self._since: dict[str, float] = {}

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
        alarm: the pass that measures a stall announces it, and every later one is
        silent until the source returns cleanly.

        One measurement of that rule is spent saying nothing (#3061): the pass that
        OPENS a stall streak only records when it started, and the announcement waits
        for a pass that still finds the source past its window. This replaces the
        earlier rule, which announced a stall on the first pass after a restart on the
        grounds that the clean stamp outlives the pool — true, and the reason the rule
        was noise. The stamps outlive the pool precisely so a restart cannot hide a
        fault, but the tick that reads them first cannot tell that fault from a
        restart, and `_scheduler_loop` runs this before `_scheduler_pass` has stamped
        anything, so every boot produced one. Over 2026-10-16 05:55 → 2026-10-19 23:07:
        28 of 34 stall events within 60 s of a pool start, none of them behind a failed
        enqueue. The same streak rail covers this box's ~190 s whole-process blocks,
        which are gone by the next measurement. What a stall still standing at the
        second measurement pays is one scheduler interval of delay, which is the price
        #1359 already accepted for the same trade at `workers/service_probe.py:165-171`.

        The streak clock is per source and is dropped the moment the source is not
        stalled, so it cannot leak and cannot carry an old stall's patience into a new
        one. A source that was never announced gets no recovery line either: reporting
        the end of a fault nobody was told began is the other half of the noise.
        """
        now = self._clock()
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
            if not row["stalled"]:
                self._since.pop(name, None)
                if name not in self._stalled:
                    # Healthy all along, or a stall found inside the grace that has
                    # since cleared — no announcement either way.
                    continue
                self._stalled.discard(name)
                text = f"dispatch recovered: {name} — {row['detail']}"
            elif name in self._stalled:
                continue  # no crossing: a steady stall stays quiet
            elif name not in self._since:
                self._since[name] = now   # the grace pass: found, not yet announced
                continue
            else:
                held = now - self._since[name]
                self._stalled.add(name)
                text = (f"dispatch stalled: {name} — {row['detail']} "
                        f"(confirmed in this process {held:.0f}s after first "
                        "measuring it)")
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


#: The only source whose judgement this gates, named here rather than imported from
#: `workers.sources.board_steward` for the closure reason stated above. A typo in it
#: selects 0 rows and the aggregate reads an empty week, so the health node asserts
#: the string against the source's own `NAME`.
BOARD_STEWARD_SOURCE = "board-steward"

# ── the board-steward judgement gate, read not written ───────────────────────────
#
# #1688 made a steward tick's judgement recordable and nothing read it, so the
# figure the `apply` flip is gated on lived only in a hand-run sqlite query.
# The reader lives HERE rather than in `workers/sources/board_steward.py`, next
# to the writer it reads, for one reason: `app.routers.workers` is reachable from
# the backend agent's import closure, and importing a module that also contains a
# WRITER (`apply_moves`, which writes to the live backlog) would put that writer
# inside a tool agent's reach as a bare import. `workers.dispatch_watch` is
# already in that closure and already imported by this route as the read-only
# watcher it is, so hosting the aggregate here adds no edge to
# tests/fixtures/worker_co_presence_baseline.json. The cost of the move is that
# the reader no longer sits beside the code that emits the block, and
# `tests/test_board_steward.py::test_the_judgement_reader_names_the_keys_the_writer_emits`
# is what pays that cost back: it asserts the keys read here are the keys
# `agreement()` writes.
#: Rolling window `judgement_aggregate` reads, in days. Seven is the window #1688's
#: owed entry named for the `apply` flip gate, so the figure the health route shows
#: and the thresholds written in `config.yaml`'s board-steward comment are ONE
#: measurement rather than two that happen to agree.
JUDGEMENT_WINDOW_DAYS = 7

#: Rows one aggregate scan reads at most, and the reason it cannot be "no limit":
#: `WorkQueue.list_runs_joined` puts its `limit` into `LIMIT ?`, and SQLite answers a
#: NULL there with `datatype mismatch`, so there is no unbounded call to make. The
#: number is set above board-steward's heaviest realistic week (~271 run rows in the
#: live 7-day window on 2026-10-20), and `truncated` below is what stops the ceiling
#: from being a silent wrong answer: a window that overflows it reports a FLOOR, not
#: a total.
JUDGEMENT_SCAN_LIMIT = 5000


def judgement_aggregate(queue: Any, window_days: int = JUDGEMENT_WINDOW_DAYS,
                        now: Optional[datetime] = None) -> dict:
    """Add up #1688's per-tick `agreement` over a rolling window: the flip gate, readable.

    #1688 made each tick's own record gateable — `agreement()` carries `judged`, and
    `rate` is None over an empty judged set — but nothing downstream added the ticks
    up, so the trailing-7-day figures the `apply` flip waits on could only be had by
    hand-running sqlite over `workers.db` (measured that way on 2026-10-20: judged
    86, agree 57, disagree 29, rate 0.663). This is that query where the fleet's own
    health route already reads the store, and `GET /api/workers/health` puts its
    result on the board-steward row.

    Read-only, and it decides nothing. It does not touch `apply`. The numbers it adds
    up against — summed `judged` >= 50 and `rate >= 0.90` over this window — are
    settled, ruled by owed-check on 2026-10-25 (#3089 asked, #3329 landed the
    ruling), and re-ruling the floor as cadence moves stays that job's to do from a
    fresh recompute; a function that only reads has no part in either. It reports,
    and traffic or a person flips.

    What a row must look like to count: only a row whose `meta_json.agreement`
    carries an INTEGER `judged` contributes, and it contributes to BOTH sides.
      - a `failed`/`skipped`/`interrupted` tick carries no `agreement` block at all
        (175 such rows since #1688 landed) and adds nothing to either side;
      - a pre-#1688 row carries an `agreement` block with NO `judged` key (119 such
        rows, 13 of them with a non-empty `agree` list). Taking the numerator from
        those and the denominator only from keyed rows would inflate the rate — the
        same false-100% #1688 exists to stop — so they are out of both sides.
    A `judged: 0` tick IS counted as read and adds 0 to both sums, which is why no
    run of zero-judged ticks lifts the rate: summed `judged` 0 gives rate None, never
    1.0.

    The window is a string comparison, deliberately. `runs.completed_at` is an
    ISO-8601 TEXT column with a `+00:00` offset, so `list_runs_joined`'s
    `completed_at >= since` filters lexicographically and the cutoff below is built
    in that same shape; a numeric cutoff against that column selects 0 rows and
    reads as "no traffic" instead of "not old enough".

    `ticks_in_window` and `ticks_with_judged` are the pair that makes an empty window
    legible rather than reassuring: 0 rows read is a quiet or newly-created store,
    40 rows read with 0 keyed is a source that has stopped judging, and those are
    different facts about different problems.
    """
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    since = (ref - timedelta(days=int(window_days))).isoformat(timespec="seconds")
    agg = {"window_days": int(window_days), "window_since": since,
           "ticks_in_window": 0, "ticks_with_judged": 0,
           "judged": 0, "agree": 0, "disagree": 0, "rate": None,
           "scanned_limit": JUDGEMENT_SCAN_LIMIT, "truncated": False}
    rows = queue.list_runs_joined(source=BOARD_STEWARD_SOURCE, since_iso=since,
                                  limit=JUDGEMENT_SCAN_LIMIT)
    # Filling the SELECT's own ceiling is the only signal that more rows exist, and
    # it is stated rather than assumed: without it `ticks_in_window` would equal the
    # limit and read as a measurement of the window. SQLite has no unbounded
    # `LIMIT ?` — a NULL there is a `datatype mismatch` — so the ceiling is real and
    # has to be visible.
    agg["truncated"] = len(rows) >= JUDGEMENT_SCAN_LIMIT
    for row in rows:
        agg["ticks_in_window"] += 1
        meta = row.get("meta_json")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta or "{}")
            except ValueError:
                continue        # an unreadable row is not a judged tick
        block = (meta or {}).get("agreement")
        if not isinstance(block, dict):
            continue            # failed / skipped / interrupted: no tick happened
        judged = block.get("judged")
        if not isinstance(judged, int) or isinstance(judged, bool):
            continue            # pre-#1688 shape: no denominator to weigh against
        agg["ticks_with_judged"] += 1
        agg["judged"] += judged
        agg["agree"] += len(block.get("agree") or [])
        agg["disagree"] += len(block.get("disagree") or [])
    if agg["judged"]:
        agg["rate"] = round(agg["agree"] / agg["judged"], 4)
    return agg


def _agreement_block_keys() -> tuple[str, ...]:
    """The keys of an `agreement()` block this reader depends on, for the coupling test."""
    return ("judged", "agree", "disagree")


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
