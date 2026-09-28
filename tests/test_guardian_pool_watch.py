"""The out-of-band pool-silence watch: the alarms inside the worker pool cannot
report their own death, so this one sits outside the pool entirely.

Since #1682 every in-pool alarm runs in `WorkerPool._scheduler_loop`
(`workers/pool.py:960`, awaiting `_watch_dispatch` at :973 and `_watch_fleet` at
:979). Two supported routes stop that loop and neither can say so:

* `workers.enabled: false` at boot — `start_worker_pool()`
  (`app/routers/workers.py:765-770`) returns before `WorkerPool` is constructed,
  and `/api/workers/status` answers HTTP 200 `{"initialized": false}` with **no
  `pool` key at all**, because `get_queue()` raises deliberately to tell "workers
  are switched off" from "running and empty";
* `POST /api/workers/enable {"enabled": false}` — `await pool.stop()`
  (`app/routers/workers.py:409`) clears `_running` and cancels `_scheduler_task`,
  and the status route answers `{"pool": {"running": false}}`, because the `_pool`
  global is still set.

The two routes therefore report different shapes, which is why the predicate below
treats an absent `pool` key as not-running rather than as no information: a check
keyed on `pool.running` is silent on exactly the boot-time route this was filed
for. `/api/health` publishes `workers.enabled` (`app/routers/health.py:107,132`)
and nothing reads it, and every autonomy job that might have noticed is itself a
worker row the dead pool was supposed to run.

Out of scope by ruling: the pause. `WorkerPool.pause` (`workers/pool.py:555`) sets
`_paused_operator` / `_paused_automod` and never touches `_running`, so the loop and
its alarms stay alive; `Guardian._pause_workers` (`agent-services/guardian/guardian.py:1058`)
drives it during a vault tripwire, which is exactly why this watch runs above
`tick()`'s `paused` return. `paused: true` beside `running: true` is a working system
doing something deliberate and must not page. It is asserted as such below.

Everything here runs against fakes at the one boundary that matters — `probes.probe`
speaking to a backend that is not there.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services" / "guardian"))

import guardian as G          # noqa: E402
import policy                 # noqa: E402
import poolwatch as PW        # noqa: E402
import probes                 # noqa: E402

BACKEND = "http://127.0.0.1:8080"
WORKERS_URL = BACKEND + "/api/workers/status"
HEALTH_URL = BACKEND + "/health"

#: One healthy `/health` answer, in `probes.probe`'s own verdict shape.
HEALTH_OK = {"ok": True, "status": 200, "body": {"status": "ok"}, "error": None,
             "latency_ms": 2.0, "kind": "ok"}

# `/api/workers/status` bodies, each shaped by the route's own return:
# `{"initialized": False}` is the boot route, and `{"pool": {"running": False}}` is
# what the enable route answers — the `pool` key is `pool.status()` when a pool
# object exists and that same one-key dict when it does not.
DISABLED_AT_BOOT = {"initialized": False}
STOPPED = {"initialized": True, "workers_enabled": True, "pool": {"running": False}}
RUNNING = {"initialized": True, "workers_enabled": True,
           "pool": {"running": True, "paused": False, "in_flight_count": 2}}
RUNNING_PAUSED = {"initialized": True, "workers_enabled": True,
                  "pool": {"running": True, "paused": True,
                           "paused_by": "vault-tripwire", "in_flight_count": 0}}


class _Unreadable:
    """Marker for a status-route probe that answered with no usable body."""

    def __init__(self, kind: str):
        self.kind = kind


def _verdict(body, *, ok=True, kind="ok", status=200, error=None):
    return {"ok": ok, "status": status, "body": body, "error": error,
            "latency_ms": 1.0, "kind": kind}


class _Responder:
    """A `probes.probe` stand-in. `/health` always answers ok; the workers status
    route replays `bodies`, repeating the last one. Every URL asked is recorded, so a
    node can name the request the watch made rather than infer it from a verdict."""

    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float) -> dict:
        self.calls.append(url)
        if url == HEALTH_URL:
            return HEALTH_OK
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        if isinstance(body, _Unreadable):
            return _verdict(None, ok=False, status=503 if body.kind == "http_error" else None,
                            kind=body.kind, error=f"simulated {body.kind}")
        return _verdict(body)


def _tick(watch, body, *, health=HEALTH_OK, now=None):
    return watch.tick(health, now=now, probe=_Responder(body))


# ── clause 1: one predicate over both report shapes ────────────────────────

@pytest.mark.parametrize("body,expect_not_running", [
    (DISABLED_AT_BOOT, True),         # boot route: no `pool` key at all
    (STOPPED, True),                  # enable route: `pool.running` false
    ({"initialized": True}, True),    # no pool object: route answers {"running": false}
    ({"initialized": True, "pool": {"running": False}}, True),
    ({"initialized": True, "pool": None}, True),  # no pool object: nothing is running
    ({"initialized": True, "pool": {"running": True, "paused": True}}, False),
    ({"initialized": True, "pool": {"running": True}}, False),
    (RUNNING, False),
])
def test_the_predicate_names_not_running_for_both_report_shapes(body, expect_not_running):
    """The two disarmament routes report DIFFERENT shapes, and the out-of-pool reader
    has to call both of them not-running while a running pool — paused or not — is
    called running.

    `{"initialized": false}` carries no `pool` key, so a check keyed on
    `pool.running` reads it as "nothing to say" and stays silent on the boot route
    the item was filed for. `running: true, paused: true` is the guardian's own
    pause, or an operator's, with the scheduler loop alive: out of scope by ruling.
    """
    assert PW.pool_not_running(body) is expect_not_running, body


@pytest.mark.parametrize("body", [None, {}, "not-a-dict", {"pool": "wrong-type"},
                                  {"pool": {"running": "yes"}}, {"initialized": "true"}])
def test_the_predicate_calls_an_unreadable_body_unknown_not_not_running(body):
    """A body that says nothing is not evidence of a stopped pool.

    `probes.probe` answers `body: None` when the response is not JSON, so the
    no-information case arrives routinely; reading it as not-running would page on
    a proxy returning an HTML error page. A `running` that is not a bool is the same
    case — the watch does not get to guess which way a truthy string points."""
    assert PW.pool_not_running(body) is None, body


# ── clause 2: grace, then exactly one alert ────────────────────────────────

def test_one_alert_once_the_silence_holds_three_ticks(tmp_path):
    """Two silent ticks are the grace; the third is an alert; the window after it is
    quiet, because a stop that lasts 15 minutes must not be 180 notices."""
    watch = PW.PoolWatch(tmp_path, base_url=BACKEND)
    grace = [_tick(watch, STOPPED) for _ in range(policy.POOL_SILENT_STREAK - 1)]
    assert [r["alert"] for r in grace] == [None, None], (
        f"alerted inside the grace streak: {[r['reason'] for r in grace]}")
    assert [r["streak"] for r in grace] == [1, 2]
    fired = _tick(watch, STOPPED)
    assert fired["alert"] is not None, "the streak never reached an alert"
    assert fired["streak"] == policy.POOL_SILENT_STREAK == 3
    assert fired["alert"]["level"] == "error"
    assert fired["alert"]["title"] == PW.ALERT_TITLE
    # The alert has to name both routes, since nothing downstream can tell them apart.
    for needle in ("workers.enabled", "api/workers/enable"):
        assert needle in fired["alert"]["body"], fired["alert"]["body"]
    assert _tick(watch, STOPPED)["alert"] is None, "the same window alerted twice"


def test_the_status_url_is_the_configured_backend_not_a_hardcoded_port(tmp_path):
    """The watch asks the backend it was given, the way the liveness probes do:
    `Guardian.__init__` derives the base from `--backend-url`, so a guardian pointed
    at another port cannot go reading a port it was never pointed at."""
    watch = PW.PoolWatch(tmp_path, base_url="http://127.0.0.1:9999")
    probe = _Responder(STOPPED)
    watch.tick(HEALTH_OK, probe=probe)
    assert probe.calls == ["http://127.0.0.1:9999/api/workers/status"], probe.calls


def test_a_status_route_that_answers_nothing_advances_no_streak(tmp_path):
    """Health is fine but the workers route does not answer usefully (refused,
    timeout, 503): that is not a stopped pool, so nothing alerts and the grace
    streak is not spent either."""
    watch = PW.PoolWatch(tmp_path, base_url=BACKEND)
    for kind in ("refused", "timeout", "http_error"):
        report = _tick(watch, _Unreadable(kind))
        assert report["reason"] == "unreadable", report
        assert report["alert"] is None, report
    assert watch.streak == 0, "an unreadable route was counted as silence"
    # The grace still starts from here: three healthy silent ticks after it.
    assert _tick(watch, STOPPED)["alert"] is None
    assert _tick(watch, STOPPED)["alert"] is None
    assert _tick(watch, STOPPED)["alert"] is not None


# ── clause 3: a running pool never pages, paused or not ────────────────────

@pytest.mark.parametrize("body", [RUNNING, RUNNING_PAUSED])
def test_a_running_pool_produces_no_alert_however_long_it_runs(tmp_path, body):
    """Twelve ticks of a healthy pool: zero alerts, and the streak never opens.

    One of the two parametrised bodies is `paused: true` — the shape the guardian's
    own `_pause_workers` produces during a vault tripwire. The pause does not stop
    `_scheduler_loop`, so its alarms are live and alerting here would page for a
    system doing exactly what it was told."""
    watch = PW.PoolWatch(tmp_path, base_url=BACKEND)
    reports = [_tick(watch, body) for _ in range(12)]
    assert [r["alert"] for r in reports] == [None] * 12
    assert [r["reason"] for r in reports] == ["running"] * 12
    assert watch.streak == 0


def test_the_streak_needs_consecutive_silence_not_merely_many_ticks(tmp_path):
    """A pool that comes back in between resets the grace: a restarting pool is the
    pool working, not an alarm that has been switched off."""
    watch = PW.PoolWatch(tmp_path, base_url=BACKEND)
    assert _tick(watch, STOPPED)["alert"] is None
    assert _tick(watch, STOPPED)["alert"] is None
    assert _tick(watch, RUNNING)["reason"] == "running"
    assert watch.streak == 0
    assert _tick(watch, STOPPED)["alert"] is None
    assert _tick(watch, STOPPED)["alert"] is None
    assert _tick(watch, STOPPED)["alert"] is not None


# ── clause 4: the backend's own outage is the other alarm's business ───────

@pytest.mark.parametrize("kind", ["refused", "timeout", "http_error"])
def test_an_unreachable_backend_alerts_nothing_and_advances_no_streak(tmp_path, kind):
    """While `/health` is not ok the status route is not even asked, so a backend
    outage costs this watch nothing: liveness belongs to the existing
    `Service down…` alert, and two notices for one outage is the double alert this
    clause exists to prevent.

    The streak half is the load-bearing one. After five ticks of outage the first
    healthy silent tick must still be followed by two more before anything fires —
    which is only true if the outage contributed nothing."""
    watch = PW.PoolWatch(tmp_path, base_url=BACKEND)
    down = _verdict(DISABLED_AT_BOOT, ok=False,
                    status=503 if kind == "http_error" else None, kind=kind,
                    error=f"simulated {kind}")
    for _ in range(5):
        report = _tick(watch, DISABLED_AT_BOOT, health=down)
        assert report["reason"] == "backend-not-ok", report
        assert report["alert"] is None, report
    assert watch.streak == 0

    probe = _Responder(DISABLED_AT_BOOT)
    assert watch.tick(HEALTH_OK, probe=probe)["alert"] is None
    assert probe.calls == [WORKERS_URL], "the status route was probed while health was down"
    assert _tick(watch, DISABLED_AT_BOOT)["alert"] is None
    assert _tick(watch, DISABLED_AT_BOOT)["alert"] is not None


# ── clause 5: the watermark is on disk, because the dedupe is not ──────────

def test_the_cooldown_survives_a_guardian_restart(tmp_path):
    """`Guardian.alert`'s repeat-suppression is a dict in memory (`_alert_seen`,
    guardian.py:129), so a guardian that restarts mid-silence has forgotten that it
    alerted. The instant goes into the guardian state dir instead, and a fresh watch
    over that dir reads it back."""
    first = PW.PoolWatch(tmp_path, base_url=BACKEND)
    for _ in range(policy.POOL_SILENT_STREAK - 1):
        assert _tick(first, STOPPED)["alert"] is None
    fired = _tick(first, STOPPED)
    assert fired["alert"] is not None

    stamp = json.loads((tmp_path / PW.STATE_NAME).read_text(encoding="utf-8"))
    assert stamp["last_alert_ts"] == pytest.approx(fired["alert_ts"], abs=1e-6), stamp
    assert stamp["last_alert"], stamp

    second = PW.PoolWatch(tmp_path, base_url=BACKEND)      # a restarted process
    for _ in range(policy.POOL_SILENT_STREAK * 3):
        assert _tick(second, STOPPED)["alert"] is None, (
            "a restart re-alerted inside the cooldown window")

    # One alert per window: the same silence, outliving the window, speaks again.
    later = time.time() + policy.ALERT_REPEAT_SECONDS + 1.0
    assert _tick(second, STOPPED, now=later)["alert"] is not None
    after = json.loads((tmp_path / PW.STATE_NAME).read_text(encoding="utf-8"))
    assert after["last_alert_ts"] == pytest.approx(later, abs=2.0), after


def test_the_watermark_file_is_what_holds_the_second_alert_back(tmp_path):
    """Positive control for the node above: delete the watermark and a restarted
    watch alerts again. Without this the previous node would pass just as well on a
    watch that never alerts at all, and the cooldown would be unmeasured.

    The warm-up runs on ONE watch, not a fresh one per tick: the grace streak lives
    on the instance, so a watch constructed inside the loop never gets past streak
    1, never alerts, and never writes the file this node then deletes — which is how
    the first version of this node failed, on an `unlink()` of a path that no run had
    ever created."""
    warm = PW.PoolWatch(tmp_path, base_url=BACKEND)
    for _ in range(policy.POOL_SILENT_STREAK - 1):
        assert _tick(warm, STOPPED)["alert"] is None
    assert _tick(warm, STOPPED)["alert"] is not None, (
        "the warm-up never alerted, so there was no watermark to remove")
    assert (tmp_path / PW.STATE_NAME).exists(), "the alert wrote no watermark"
    (tmp_path / PW.STATE_NAME).unlink()

    restarted = PW.PoolWatch(tmp_path, base_url=BACKEND)
    for _ in range(policy.POOL_SILENT_STREAK - 1):
        assert _tick(restarted, STOPPED)["alert"] is None
    assert _tick(restarted, STOPPED)["alert"] is not None, (
        "deleting the watermark changed nothing: no node was measuring it")


# ── the seat: the watch runs inside the real tick, in every state ──────────

@pytest.fixture
def guardian(tmp_path, monkeypatch):
    """A real `Guardian` over temp state, driven through `tick()` and `collect()`.

    Only the boundaries are faked: the supervisor client, the HTTP probes, and the
    sibling watches whose alert-producing paths this node is not about (they return
    nothing on a healthy system, so stubbing them changes no control flow — the
    early returns the node is testing are `tick()`'s own).

    Placement is the property. `tick()` returns early for `infra_down`, `broken` and
    `paused`, and a silence check seated below those returns would itself be a
    one-surface guard — the same mistake, one level up, that made the in-pool alarms
    blindable in the first place.
    """
    monkeypatch.setattr(policy, "LOG_FILES", ())
    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url=HEALTH_URL, mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-backend", interval=5.0, no_external_alerts=True,
    )
    g = G.Guardian(args)
    alerts: list[tuple] = []
    monkeypatch.setattr(g, "alert", lambda *a, **k: alerts.append(a))
    monkeypatch.setattr(g, "check_vault", lambda: None)
    monkeypatch.setattr(g, "check_data", lambda: None)
    monkeypatch.setattr(g, "ensure_chronic", lambda: None)
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(g, "maybe_settle", lambda current: None)
    monkeypatch.setattr(g, "maybe_selftest", lambda: None)
    monkeypatch.setattr(g, "evaluate_liveness", lambda s: (False, "ok"))

    # `systemctl` is intercepted for EVERY node that gets this fixture, not just the
    # one that needs it: `tick()` runs a real
    # `subprocess.run(["systemctl", "--user", "restart", policy.SUPERVISORD_UNIT])`
    # once its own streak reaches `SUPERVISORD_DOWN_STREAK` (guardian.py:1233-1237),
    # and the nodes below drive five-plus ticks in the `supervisord_unreachable`
    # state. Left unstubbed, a red node of that shape restarts production's
    # supervisor from inside the gate's pytest run — the same reason
    # tests/test_guardian_predicates.py:1063-1079 gives for the same interception.
    restarts: list[list[str]] = []

    def _fake_run(argv, **kw):
        restarts.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(G.subprocess, "run", _fake_run)
    g._test_alerts = alerts
    g._test_restarts = restarts
    return g


def _pool_alerts(alerts: list[tuple]) -> list[tuple]:
    """Only the alerts this watch is allowed to raise.

    Every tick-level node filters through here. `tick()` has watches of its own that
    alert legitimately in the states these nodes construct — the supervisord branch
    pages once its streak reaches `policy.SUPERVISORD_DOWN_STREAK` — and clauses 2
    and 4 are about the pool-silence title, not about the whole alert surface. An
    assertion on "no alerts at all" is wrong by construction in those states, which
    is exactly how the first version of the tick-level clause-4 node failed: its own
    captured stdout was `supervisord unreachable — restarting the unit`.
    """
    return [a for a in alerts if len(a) > 1 and a[1] == PW.ALERT_TITLE]


def _drive_tick(g, monkeypatch, *, condition: str, body):
    """Run one real `Guardian.tick()` over a fake supervisor and fake HTTP. Returns
    `(state_returned_by_tick, urls_probed)`."""
    probe = _Responder(body)
    monkeypatch.setattr(probes, "probe", probe)
    if condition == "supervisord_unreachable":
        def procs():
            raise G.SupervisordUnreachable("simulated: no supervisor socket")
    else:
        def procs():
            return {}
    monkeypatch.setattr(g.sup, "all_process_info", procs)
    monkeypatch.setattr(g.state, "pause_remaining",
                        lambda cap: 600.0 if condition == "paused" else 0.0)
    monkeypatch.setattr(g.state, "is_broken", lambda: condition == "broken")
    monkeypatch.setattr(g.state, "current", lambda: None)
    return g.tick(), probe.calls


@pytest.mark.parametrize("condition", ["armed", "supervisord_unreachable", "broken", "paused"])
def test_the_pool_is_read_whatever_state_the_tick_returns_from(guardian, monkeypatch, condition):
    """Every tick asks the workers status route exactly once — including the ticks
    that return before the rollback logic — and a running pool alerts from none.

    The expected probe list is the whole claim in one line: `/health` once (the
    snapshot's own reading, reused rather than asked twice), `/api/workers/status`
    once, and nothing else. A check seated below the early returns would make the
    `broken` and `paused` rows come back without the status URL; a watch that
    probed health for itself as well would show two health hits.
    """
    state, calls = _drive_tick(guardian, monkeypatch, condition=condition, body=RUNNING)
    assert calls == [HEALTH_URL, WORKERS_URL], (condition, state, calls)
    # Pool-titled only, through the filter: a running pool must cost this watch
    # nothing, and whatever else a tick in `supervisord_unreachable` has to say is
    # that watch's business, not evidence about this one.
    assert _pool_alerts(guardian._test_alerts) == [], (condition, guardian._test_alerts)


def test_the_silence_reaches_the_one_alert_surface_from_inside_a_real_tick(guardian, monkeypatch):
    """Clause 2 across the only process boundary this watch crosses: five ticks whose
    status route answers `{"initialized": false}` produce exactly one `Guardian.alert`
    bearing the pool-silence title, and it is the THIRD tick that produces it — the
    first two are grace. Two more silent ticks follow and add nothing, because the
    watermark that bounds them is the file on disk: no second notifier, no new level,
    no new file format."""
    states, calls, per_tick = [], [], []
    for _ in range(policy.POOL_SILENT_STREAK + 2):
        before = len(_pool_alerts(guardian._test_alerts))
        state, tick_calls = _drive_tick(guardian, monkeypatch, condition="armed",
                                        body=DISABLED_AT_BOOT)
        states.append(state)
        calls.append(tick_calls)
        per_tick.append(len(_pool_alerts(guardian._test_alerts)) - before)
    assert all(c == [HEALTH_URL, WORKERS_URL] for c in calls), calls

    # The grace is the shape of this list, not just its total: nothing before tick 3,
    # exactly one on it, and nothing after — a stop that lasts minutes is one notice.
    assert per_tick == [0, 0, 1, 0, 0], (states, per_tick)

    alerts = _pool_alerts(guardian._test_alerts)
    assert len(alerts) == 1, (states, alerts)
    level, title, body = alerts[0][:3]
    assert level == "error", alerts
    assert title == PW.ALERT_TITLE, alerts
    for needle in ("workers.enabled", "api/workers/enable"):
        assert needle in body, body
    stamp = json.loads((Path(guardian.gdir) / PW.STATE_NAME).read_text(encoding="utf-8"))
    assert stamp["last_alert_ts"] > 0, stamp


def test_an_unreachable_backend_costs_the_pool_watch_nothing_from_inside_a_tick(guardian, monkeypatch):
    """The tick-level half of clause 4: while the snapshot's `/health` verdict is not
    ok, the status route is never asked, so an outage produces one alert (liveness)
    and not two, and the grace streak does not move.

    `supervisord_unreachable` is the condition used because it is the interesting
    one: `collect()` returns early there, and the health verdict still has to reach
    the watch, which is why the snapshot publishes it on that path too.

    The ticks here go past `SUPERVISORD_DOWN_STREAK` on purpose, and that branch
    legitimately alerts and restarts the unit — so every assertion below is on the
    pool-silence title only. What the outage IS allowed to produce is asserted
    positively rather than assumed away: exactly one alert, and it is the supervisor's,
    which is the double alert this clause prevents, measured rather than described.
    """
    down = _verdict(None, ok=False, kind="refused", error="simulated refused")

    def probe(url, timeout):
        probe.calls.append(url)
        return down if url == HEALTH_URL else _verdict(DISABLED_AT_BOOT)
    probe.calls = []
    monkeypatch.setattr(probes, "probe", probe)
    monkeypatch.setattr(guardian.sup, "all_process_info",
                        lambda: (_ for _ in ()).throw(G.SupervisordUnreachable("simulated")))
    monkeypatch.setattr(guardian.state, "is_broken", lambda: False)
    monkeypatch.setattr(guardian.state, "pause_remaining", lambda cap: 0.0)
    monkeypatch.setattr(guardian.state, "current", lambda: None)

    for _ in range(policy.SUPERVISORD_DOWN_STREAK + 2):
        assert guardian.tick() == "infra_down"
    assert probe.calls.count(WORKERS_URL) == 0, \
        "the workers status route was probed while the backend was unreachable"
    assert guardian.pool.streak == 0, "the outage advanced the pool-silence streak"
    assert _pool_alerts(guardian._test_alerts) == [], guardian._test_alerts

    # And positively: the outage did speak, once, on the liveness title — the one
    # notice for one outage. Reaching this branch is also what makes the fixture's
    # `systemctl` interception matter, so the restart is asserted as intercepted
    # rather than left to happen to production.
    others = [a for a in guardian._test_alerts if len(a) > 1 and a[1] != PW.ALERT_TITLE]
    assert [a[1] for a in others] == ["supervisord was unreachable"], guardian._test_alerts
    assert guardian._test_restarts == [
        ["systemctl", "--user", "restart", policy.SUPERVISORD_UNIT]], guardian._test_restarts


def test_collect_publishes_one_backend_health_verdict_for_the_watch_to_trust(guardian, monkeypatch):
    """The seam the seat depends on: the verdict `check_pool` reads is the one
    `collect()` took for the program probe of the same URL, not a second reading a
    moment later.

    One `/health` request per tick is the assertion. Two would mean the liveness
    predicate and the pool watch could act on different answers to the same question
    within one tick, which is the failure the snapshot exists to prevent."""
    probe = _Responder(RUNNING)
    monkeypatch.setattr(probes, "probe", probe)
    monkeypatch.setattr(guardian.sup, "all_process_info", lambda: {})

    snap = guardian.collect()

    assert probe.calls == [HEALTH_URL], probe.calls
    assert snap["probes"]["lloyd-backend"] is snap["backend_health"], snap
    assert snap["backend_health"]["kind"] == "ok"

