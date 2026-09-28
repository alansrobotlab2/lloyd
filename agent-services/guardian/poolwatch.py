"""Out-of-band watch: the worker pool's alarms are inside the pool, so the pool
cannot report being switched off, and nothing else looked either (#1747).

Every alarm the fleet has lives in `WorkerPool._scheduler_loop`
(`workers/pool.py:960`, awaiting `_watch_dispatch` and `_watch_fleet`). Two
supported routes stop that loop, and both leave the system looking healthy from
every surface the backend itself publishes:

* `workers.enabled: false` at boot — `app/routers/workers.py:765-770` returns
  before `WorkerPool` is constructed, and `/api/workers/status` then answers
  HTTP 200 `{"initialized": false}` with no `pool` key, because `get_queue()`
  raises on purpose to tell "switched off" from "running and empty";
* `POST /api/workers/enable {"enabled": false}` — `app/routers/workers.py:409`
  awaits `pool.stop()`, which clears `_running` and cancels `_scheduler_task`,
  and the route keeps answering `{"pool": {"running": false}}`.

`/api/health` publishes `workers.enabled` (`app/routers/health.py:107,132`) and no
reader anywhere alarms on it; every autonomy job that might have noticed the
silence is itself a row in the queue the stopped pool is not draining. So the
reading happens here, in the guardian, which is its own process and survives both
the backend restarting and the pool not existing.

What it deliberately does *not* do:

* **Alert on a pause.** `_paused` (`workers/pool.py:579`) is derived from
  `_paused_operator` / `_paused_automod`, which `WorkerPool.pause` (`:555`) sets and
  which never touches `_running` — so the loop, and with it every alarm, stays alive.
  The guardian's own `_pause_workers` (`guardian.py:1058`) drives that during a vault
  tripwire, which is why this watch reads the pool above `tick()`'s `paused` return
  rather than after it. Out of scope by ruling.
* **Alert while the backend is unreachable.** `/health` not ok is the existing
  `Service down, but no promotion to revert` alert's territory; reading the pool
  in the same tick would page twice for one outage. The status route is not even
  asked in that case, and the grace streak does not move — otherwise a backend
  that was down for five minutes would come back to a pool alert on its first
  healthy tick.
* **Page on a boot race.** While the backend is still starting, `/health` answers
  503 (`app/routers/health.py`: `status = "starting"` → 503), so `probes.probe`
  reports `http_error` and this watch never sees the `initialized: false` body the
  startup hook has not reached yet — the same reason `cc7fe133` gives for retrying
  a failed self-test instead of paging.
"""

from __future__ import annotations

import time
from pathlib import Path

import gstate
import policy
import probes

#: Where the alert instant goes, in the guardian's own state dir. On disk rather
#: than in `Guardian._alert_seen` because that dict is per-process: a guardian that
#: restarts mid-silence has forgotten it alerted, and the silence has not.
STATE_NAME = "pool-silence.json"

#: The title, and therefore the daily-note heading. One constant so the writer and
#: any later retraction agree on it byte for byte.
ALERT_TITLE = "Worker pool is not running — its in-pool alarms are silent"

_STATUS_PATH = "/api/workers/status"


def pool_not_running(body) -> bool | None:
    """`True` / `False` for not-running / running, `None` when the body says nothing.

    `None` matters as much as the two booleans: it is what keeps a proxy's HTML
    error page, a 503 with no JSON, or a body that lost its `initialized` flag from
    reading as a stopped pool.

    The order is the whole point. A positively-reported `pool.running: true` wins
    even without the `initialized` flag, because that is the shape a running pool
    answers in and a false `True` here would page forever. Everything else that is a
    well-formed status body — `initialized` a bool — is not-running, which covers
    both `{"initialized": false}` (no `pool` key at all: the boot route) and
    `{"pool": {"running": false}}` (the enable route). A check that read only
    `pool.running` would be silent on the first of those, which is the route this
    was filed for.
    """
    if not isinstance(body, dict):
        return None
    pool = body.get("pool")
    if isinstance(pool, dict) and pool.get("running") is True:
        return False
    if not isinstance(body.get("initialized"), bool):
        return None
    return True


class PoolWatch:
    """One tick's read of `/api/workers/status`, grace-streaked and cooldown-bounded.

    Constructed by `Guardian.__init__` against the same backend the liveness probes
    use, so there is no second endpoint to keep in sync and no config key to forget.
    """

    def __init__(self, state_dir: Path, *, base_url: str,
                 timeout: float | None = None,
                 streak_threshold: int | None = None,
                 cooldown_seconds: float | None = None):
        # Resolved here rather than as default-argument values, which Python
        # freezes at def time: a test (or a later operator override) that moves
        # `policy.POOL_SILENT_STREAK` has to move the watch too, or the node
        # counting ticks and the watch counting ticks disagree by construction.
        self.dir = Path(state_dir)
        self.url = str(base_url).rstrip("/") + _STATUS_PATH
        self.timeout = policy.PROBE_TIMEOUT_SECONDS if timeout is None else timeout
        self.streak_threshold = (policy.POOL_SILENT_STREAK if streak_threshold is None
                                 else streak_threshold)
        self.cooldown_seconds = (policy.ALERT_REPEAT_SECONDS if cooldown_seconds is None
                                 else cooldown_seconds)
        self.streak = 0
        self.silent_since: float | None = None

    @property
    def state_path(self) -> Path:
        return self.dir / STATE_NAME

    def tick(self, health: dict | None, now: float | None = None,
             probe=None) -> dict:
        """Judge one tick. Returns a report; `report["alert"]` is an `alert()`
        payload or None, so the caller owns the fan-out and the title stays here.

        `health` is the backend's own `/health` verdict from this tick's snapshot —
        read from the same measurement the liveness predicate used, not a second
        probe of the same URL a moment later.
        """
        now = time.time() if now is None else float(now)
        probe = probe or probes.probe
        if not (health or {}).get("ok"):
            # Liveness is the other alarm's. Contributes nothing here, including
            # to the streak: see the module's docstring and the clause-4 test.
            return self._report("backend-not-ok", alert=None,
                                backend_kind=(health or {}).get("kind"))

        verdict = probe(self.url, self.timeout) or {}
        not_running = pool_not_running(verdict.get("body")) if verdict.get("ok") else None
        if not_running is None:
            return self._report("unreadable", alert=None,
                                status=verdict.get("status"),
                                kind=verdict.get("kind"))

        if not_running is False:
            self.streak = 0
            self.silent_since = None
            return self._report("running", alert=None)

        self.streak += 1
        if self.silent_since is None:
            self.silent_since = now
        if self.streak < self.streak_threshold:
            return self._report("silence-waiting", alert=None)

        last = float((gstate.read_json(self.state_path) or {}).get("last_alert_ts") or 0.0)
        if now - last < self.cooldown_seconds:
            # Still one incident: the window has not turned over. This is the branch
            # a restarted guardian takes and `_alert_seen` could not have.
            return self._report("silence-cooled", alert=None, last_alert_ts=last)

        alert = self._alert(now)
        self._write(now)
        return self._report("silent", alert=alert, alert_ts=now)

    # ── internals ────────────────────────────────────────────────────────
    def _report(self, reason: str, *, alert, alert_ts=None, **extra) -> dict:
        out = {"reason": reason, "streak": self.streak, "alert": alert,
               "alert_ts": alert_ts}
        out.update(extra)
        return out

    def _write(self, now: float) -> None:
        gstate.write_json_atomic(self.state_path, {
            "schema": 1,
            "last_alert_ts": now,
            "last_alert": gstate.now_iso(),
            "streak_at_alert": self.streak,
            "silent_since": self.silent_since,
            "url": self.url,
        })

    def _alert(self, now: float) -> dict:
        waited = self.streak * policy.TICK_SECONDS
        since = self.silent_since or now
        return {
            "level": "error",
            "title": ALERT_TITLE,
            "body": (
                f"`{self.url}` has reported no running pool for {self.streak} "
                f"consecutive guardian ticks (~{waited:.0f}s), while `/health` "
                f"answers ok.\n\n"
                "Every fleet alarm — the dispatch stall and the fleet watchdog "
                "from #1682 — runs inside `WorkerPool._scheduler_loop`, so while "
                "this holds none of them can fire, and the queue is not being "
                "drained. Two routes get here:\n"
                "  * `workers.enabled: false` in config (or its overlay) — the "
                "pool is never constructed at boot; status answers "
                "`{\"initialized\": false}`.\n"
                "  * `POST /api/workers/enable {\"enabled\": false}` — "
                "`pool.stop()` cancelled the scheduler task; status answers "
                "`{\"pool\": {\"running\": false}}`.\n\n"
                "Re-arm it with `POST /api/workers/enable {\"enabled\": true}` "
                "or set `workers.enabled: true`, then check what the queue "
                "accumulated while nothing was running.\n"
                "A pause (`running: true, paused: true`) is deliberately NOT "
                "reported here: a paused pool still has its alarms."),
            "evidence": (
                f"- streak {self.streak} of threshold {self.streak_threshold} "
                f"(tick {policy.TICK_SECONDS:g}s)\n"
                f"- silent since {gstate.now_iso() if self.silent_since is None else time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(since))}\n"
                f"- backend health probe ok; status url {self.url}\n"
                f"- watermark: {self.state_path}"),
        }
