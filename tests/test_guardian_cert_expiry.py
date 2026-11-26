"""The tailnet TLS leaf gets a deadline, because nothing reads the unit that renews it.

`goliath.taile37041.ts.net.crt` is a 90-day Let's Encrypt leaf handed out by
`tailscale cert`, and `web/vite.config.ts` selects it for the Mission Control
frontend by `fs.existsSync` — by EXISTENCE, not by validity — so an expired pair
left standing in `agent-services/cert/` is served tailnet-wide right up to the
second it stops validating. `scripts/renew-tailnet-cert.sh` owns it on a daily
timer. #1727 ruled that the unit's non-zero exit is the ONLY signal of a refused
mint, "systemctl --user --failed and the journal". Both of those are readable, by
nobody: at triage on 2026-11-25,

* `git grep -nE 'is-failed|--failed|OnFailure' -- scripts workers app agent-services`
  answers twice, and both hits are prose comments —
  `agent-services/systemd/lloyd-cert-renew.service:23` and
  `scripts/renew-tailnet-cert.sh:33`. No code path, no `OnFailure=` unit.
* the one health route that does look at units cannot see a service either:
  `_UNITS_FAIL_RE` in `scripts/service_health_check.py` matches `.timer` names, off
  `scripts/maintenance/check-unit-enabledness.sh`, which asserts `is-enabled` and
  arming. A `lloyd-cert-renew.service` that exits 1 every day keeps every published
  row green.

Live state on the day it was filed agrees that this is a live leaf and not a
hypothetical one: `openssl x509 -enddate -noout -in
agent-services/cert/goliath.taile37041.ts.net.crt` reports
`notAfter=Dec 19 01:16:29 2026 GMT` — 23.5 days out at 2026-11-25 12:17 UTC — with
the timer `enabled` and last run `Tue 2026-11-24 04:18:44 PST`. So from ~2026-12-05
the window is open, and a mint refused on every run from then leaves no red
anywhere until TLS breaks on 2026-12-19.

The floor is 10 days, deliberately NOT the renew window's 14
(`CERT_RENEW_WINDOW_DAYS:-14`, `scripts/renew-tailnet-cert.sh`): at 14 this page
would fire on the first morning the window opens, before the daily run has had its
chance, and a healthy box would hear from it once a year for no reason. Ten days in,
at least one run has already failed to produce a replacement, which is the condition
being reported. Clause 1 pins that gap against the script's own text rather than
against a second copy of the number, so lowering the window below the floor cannot
pass silently.

What is proven here is by FIXTURE, not by log inspection, because the alternative is
not available: the leaf is gitignored (`.gitignore:99`, `/agent-services/cert/`), so
no cert of a known age can be committed beside this file, and the real one is 23 days
from its date — the only exercisable behaviour on the box today is the no-op. Every
leaf below is real, minted by the same `openssl` the script uses, with
`-not_after` set to the instant under test; the dates the alert is checked against
are read back out of the bytes with `openssl x509 -enddate`, never assembled in this
file. #1416 is the precedent for the shape: a guardian poller with a named threshold,
not a new timer.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services" / "guardian"))

import guardian as G          # noqa: E402
import policy                 # noqa: E402

DAY = 86400.0
# The two names the frontend itself hardcodes — `CERT_DIR` is
# `resolve(__dirname, '../agent-services/cert')` and `TS_HOST` is the hostname, both in
# `web/vite.config.ts`. Written literally here rather than read off `policy`, so that a
# guardian whose constants pointed elsewhere fails per node (it would go looking for a
# leaf that is not there) instead of agreeing with itself; the equality between this
# pair and the guardian's constants is pinned by its own node below.
CERT_DIR = "agent-services/cert"
HOST = "goliath.taile37041.ts.net"
RENEW_UNIT = "lloyd-cert-renew.service"
#: The read-out the alert must point a reader at, spelled as a person would run it.
JOURNAL_CMD = f"journalctl --user -u {RENEW_UNIT}"
#: The published title, written literally so this file can say what the alarm is
#: CALLED. A test that imported the constant could never notice it had been renamed to
#: something a reader would not act on; `G.CERT_LEAF_ALERT_TITLE` is pinned to this
#: string by its own node below instead.
TITLE = "The tailnet TLS leaf is expired, missing or unreadable"
RENEW_SCRIPT = ROOT / "scripts" / "renew-tailnet-cert.sh"


# ── real leaves, minted the way the renewer mints them ─────────────────────

@pytest.fixture(scope="module")
def key_pem(tmp_path_factory):
    """One RSA key for every leaf this module mints. `openssl req -x509 -key`
    costs milliseconds; `-newkey` would pay a keygen per leaf."""
    key = tmp_path_factory.mktemp("keys") / "leaf.key"
    subprocess.run(["openssl", "genrsa", "-out", str(key), "2048"], check=True,
                   capture_output=True)
    return key


def _stamp(ts: float) -> str:
    return time.strftime("%Y%m%d%H%M%SZ", time.gmtime(ts))


def _mint(dir_: Path, key_pem: Path, not_after_days: float) -> Path:
    """A real self-signed pair in `dir_` whose notAfter is `not_after_days` away.

    Written with openssl's own `-not_after`, so the string `openssl x509 -enddate`
    prints for it is the production string rather than one assembled here — and the
    `.key` beside it is written too, because the pair is what Vite looks for.
    """
    dir_.mkdir(parents=True, exist_ok=True)
    crt, key = dir_ / f"{HOST}.crt", dir_ / f"{HOST}.key"
    now = time.time()
    subprocess.run(["openssl", "req", "-x509", "-key", str(key_pem), "-out", str(crt),
                    "-subj", f"/CN={HOST}",
                    "-not_before", _stamp(now - DAY),
                    "-not_after", _stamp(now + not_after_days * DAY)],
                   check=True, capture_output=True)
    key.write_bytes(key_pem.read_bytes())
    return crt


def _enddate_independent(crt: Path) -> tuple[str, str]:
    """The pair's own `(raw GMT stamp, ISO date)`, read with the same command the
    check under test uses, run from HERE rather than from inside `guardian.py`. An
    assertion against this is an assertion about the certificate, not about a string
    this file built."""
    raw = subprocess.run(["openssl", "x509", "-enddate", "-noout", "-in", str(crt)],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert raw.startswith("notAfter="), raw
    stamp = raw[len("notAfter="):]
    gm = datetime.strptime(stamp, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
    return stamp, gm.date().isoformat()


def _cert_dir(repo: Path) -> Path:
    return repo / CERT_DIR


class Notifier:
    """The fan-out surface, captured. `alert` and `resolve` are the only two calls
    this watch makes, and both are recorded with every argument."""

    def __init__(self) -> None:
        self.alerts: list[tuple] = []
        self.resolves: list[tuple] = []

    def alert(self, *a, **kw) -> None:
        self.alerts.append((a, kw))

    def resolve(self, *a, **kw) -> None:
        self.resolves.append((a, kw))

    def titles(self, calls=None) -> list[str]:
        return [c[0][1] for c in (calls if calls is not None else self.alerts)]


@pytest.fixture
def watchdog(tmp_path):
    """A `Guardian` with just enough state for one watch: a repo, a notifier, and
    the alert-suppression dict `Guardian.alert` owns.

    Built with `__new__` and no `__init__`, exactly like the other single-watch
    guardian tests, so nothing else in the tick can alert and every call on the
    notifier below is attributable to `check_cert_expiry` alone. `alert` is the REAL
    method, not a stub: the repeat bound this file asserts lives inside it.
    """
    g = G.Guardian.__new__(G.Guardian)
    g.repo = str(tmp_path)
    g.notifier = Notifier()
    g._alert_seen = {}
    g._cert_leaf_alarm = None
    g.last_alert = ""
    return g


def _alerts(watchdog, title: str = TITLE) -> list[tuple]:
    """Only the alerts this watch is allowed to raise. `alert` is real here, so any
    other title on the notifier is a different watch's business — there should not be
    one, and the nodes that care say so by asserting the total is zero as well."""
    return [a for a in watchdog.notifier.alerts if a[0][1] == title]


def _body(alert: tuple) -> str:
    """The alert's body, and the level+title asserted separately."""
    return alert[0][2]


# ── clause 1: the floor is below the renew window, on purpose ───────────────

def test_the_floor_is_ten_days_and_strictly_below_the_renew_window():
    """`TAILNET_LEAF_MIN_DAYS_LEFT` is 10, and 10 is below the window the renewer
    opens — read out of the script's own `CERT_RENEW_WINDOW_DAYS:-N`, not restated
    here. Two copies of 14 would let either side move and keep this green; the
    assertion that matters is the ORDER, because a floor at or above the window is a
    page on the first morning of every renewal cycle, before the mint has run."""
    assert policy.TAILNET_LEAF_MIN_DAYS_LEFT == 10
    text = RENEW_SCRIPT.read_text(encoding="utf-8")
    declared = re.search(r'CERT_RENEW_WINDOW_DAYS:-(\d+)', text)
    # Bound and asserted, not chained: if the script ever stops carrying its default
    # this says "the window is no longer declared here" — the fact that actually
    # invalidates the comparison — instead of an AttributeError on None that reads as
    # a broken test.
    assert declared is not None, (
        "renew-tailnet-cert.sh no longer declares CERT_RENEW_WINDOW_DAYS:-N, so the "
        "floor below has nothing to be strictly below any more")
    window = int(declared.group(1))
    assert window == 14, text
    assert policy.TAILNET_LEAF_MIN_DAYS_LEFT < window


def test_the_pair_it_reads_is_the_pair_the_frontend_selects_by_existence(tmp_path):
    """The watch and the thing it protects must name one file. Both names above are
    the frontend's own — `web/vite.config.ts` carries the directory literal and the
    hostname literal, and its `haveTs` is an `fs.existsSync` pair — so this asserts the
    guardian's constants against the .ts text rather than against this file, and pins
    the alert's published title to the string every assertion below uses.

    A name that drifts is the quiet failure mode of any cert watcher: the alert keeps
    firing healthily about a path that is not the one being served. There is no second
    checkout of a live pair to discover that in — `agent-services/cert/` is gitignored
    (`.gitignore:99`), so it exists only in the running tree.
    """
    assert policy.TAILNET_CERT_DIR == CERT_DIR
    assert policy.TAILNET_CERT_HOST == HOST
    assert G.CERT_LEAF_ALERT_TITLE == TITLE

    vite = (ROOT / "web" / "vite.config.ts").read_text(encoding="utf-8")
    assert f'"../{CERT_DIR}"' in vite, vite
    assert f'const tsHost = "{HOST}"' in vite, vite
    # And the selection really is by existence of the PAIR, which is what makes a
    # half-written pair and an absent one the same fact to the server.
    assert "fs.existsSync(tsCert) && fs.existsSync(tsKey)" in vite, vite
    assert '"lloyd.crt"' in vite, vite


def test_a_leaf_sitting_on_the_floor_alerts_and_one_past_the_floor_does_not(tmp_path,
                                                                           key_pem,
                                                                           watchdog):
    """The condition is `days-left <= 10`, pinned from both sides of 10 and not just
    from one: a leaf whose notAfter is exactly ten days away is already inside the
    floor, and one ten and a half days away is not. The half day is the margin that
    keeps the boundary real — a leaf minted at exactly 10 days reads as 9.99 by the
    time the check runs, which is the inclusive case, while a check written `<`
    instead of `<=` would pass a 10-day mint and a 10.0-day mint alike."""
    soon = _mint(_cert_dir(tmp_path / "floor"), key_pem, 10.0)
    watchdog.check_cert_expiry(soon)
    assert len(_alerts(watchdog)) == 1, watchdog.notifier.alerts

    outside = G.Guardian.__new__(G.Guardian)
    outside.repo, outside.notifier = watchdog.repo, Notifier()
    outside._alert_seen, outside._cert_leaf_alarm, outside.last_alert = {}, None, ""
    outside.check_cert_expiry(_mint(_cert_dir(tmp_path / "above"), key_pem, 10.5))
    assert outside.notifier.alerts == []


# ── clause 2: where the date comes from, and the healthy case ───────────────

def test_the_date_is_read_with_openssl_enddate_from_the_path_it_is_handed(
        tmp_path, key_pem, watchdog, monkeypatch):
    """Clause 2's mechanism, not just its outcome: the argv that reaches the OS is
    `openssl x509 -enddate -noout -in <the path passed in>`, delegated to for real so
    the alert on the other side of it is the real one. A Python PEM parse would
    satisfy "knows the date" and still leave the guardian and the renewer holding two
    readings of one file, which is the thing this watch must never do."""
    crt = _mint(_cert_dir(tmp_path), key_pem, 3.0)

    calls: list[list[str]] = []
    real = subprocess.run

    def spy(argv, *a, **kw):
        calls.append(list(argv))
        return real(argv, *a, **kw)

    monkeypatch.setattr(G.subprocess, "run", spy)   # only the check's own spawns
    watchdog.check_cert_expiry(crt)

    assert calls == [["openssl", "x509", "-enddate", "-noout", "-in", str(crt)]], calls
    assert len(_alerts(watchdog)) == 1


def test_a_leaf_forty_days_out_costs_no_call_on_the_notifier_at_all(tmp_path, key_pem,
                                                                    watchdog):
    """A healthy leaf is silent in both directions. Not just no `alert`: no
    `resolve` either, because a watch that retracts something it never raised turns
    an all-clear into a write on every healthy tick, and 40 days is where a freshly
    minted 90-day leaf sits the morning after a renewal — the most common state by a
    wide margin."""
    crt = _mint(_cert_dir(tmp_path), key_pem, 40.0)
    watchdog.check_cert_expiry(crt)
    assert watchdog.notifier.alerts == []
    assert watchdog.notifier.resolves == []
    assert watchdog._cert_leaf_alarm is None


# ── clause 3: one page, and it says where to go read ────────────────────────

def test_a_leaf_three_days_out_pages_once_naming_its_date_and_the_renew_unit(
        tmp_path, key_pem, watchdog):
    """The alarm itself, against the certificate rather than a string assembled
    here: the ISO date it must carry is the one read back out of the minted bytes
    with `openssl x509 -enddate` from this file's own subprocess, and the pointer must
    be the unit that owns the mint. Then two more checks in the same state, because
    the condition lasts for days at a 5-second tick — one page, not a stream.

    Two bounds hold that, and only one of them is `Guardian.alert`'s own window:
    `_alert_seen` suppresses a same-title re-alert for
    `policy.ALERT_REPEAT_SECONDS` (900 s), which for a condition designed to last ten
    days is 96 fan-outs a day — and the backlog duplicate guard that would absorb
    those fails OPEN when the board's cold parse cache overruns its timeout (#3551
    filed #3558/#3559/#3566 that way). So the watch also latches on the verdict it
    reported, the `_tick_error_alerted` idiom at the top of this file's own class.
    Three ticks here are seconds apart and cannot tell the two apart; what they DO
    pin is that the pair of them is closed, and `test_a_replaced_leaf_...` below pins
    the latch's release."""
    crt = _mint(_cert_dir(tmp_path), key_pem, 3.0)
    stamp, iso = _enddate_independent(crt)

    watchdog.check_cert_expiry(crt)
    watchdog.check_cert_expiry(crt)
    watchdog.check_cert_expiry(crt)

    fired = _alerts(watchdog)
    assert len(fired) == 1, watchdog.notifier.alerts
    level, title = fired[0][0][0], fired[0][0][1]
    assert (level, title) == ("error", TITLE)
    body = _body(fired[0])
    assert iso in body, body
    assert stamp in body, body
    assert JOURNAL_CMD in body, body


def test_a_replaced_leaf_retracts_the_page_once_and_then_writes_nothing(tmp_path, key_pem,
                                                                        watchdog):
    """The latch opens on the other side, and only on the transition. A renewal that
    succeeds must close the section the alarm opened (#1536's rule: retract on the
    surface the alarm used) — exactly once, because a retraction on every subsequent
    healthy tick is a write per 5 seconds forever. Absent a prior alarm, `ok` writes
    nothing at all, which is the case `test_a_leaf_forty_days_out...` pins."""
    near = _mint(_cert_dir(tmp_path / "renew"), key_pem, 3.0)
    watchdog.check_cert_expiry(near)
    assert len(_alerts(watchdog)) == 1
    assert watchdog.notifier.resolves == []

    d = _cert_dir(tmp_path / "renew")
    _mint(d, key_pem, 90.0)                   # overwrites the pair in place
    crt = d / f"{HOST}.crt"
    watchdog.check_cert_expiry(crt)
    assert len(watchdog.notifier.resolves) == 1, watchdog.notifier.resolves
    assert watchdog._cert_leaf_alarm is None
    watchdog.check_cert_expiry(crt)
    watchdog.check_cert_expiry(crt)
    assert len(watchdog.notifier.resolves) == 1, watchdog.notifier.resolves
    assert len(_alerts(watchdog)) == 1


# ── clause 4: the two readings that are not a date, and no raising ──────────

def test_a_missing_pair_pages_and_names_both_causes_so_the_fallback_is_not_a_fault(
        tmp_path, watchdog):
    """Nothing at that path at all. The text has to carry BOTH causes or the alarm
    misreads #2209's own design as a new break: `renew-tailnet-cert.sh` step 5 MOVES a
    pair it could not renew into `.expired-<host>/` within 24 h of its notAfter and
    restarts the frontend, precisely so Vite's existence check falls back to the
    still-valid `lloyd.crt`. That is a working box with a missing tailnet pair, and
    the reader who gets only "never minted" goes looking for a mint that was never
    meant to happen."""
    crt = _cert_dir(tmp_path) / f"{HOST}.crt"     # deliberately never written
    watchdog.check_cert_expiry(crt)

    fired = _alerts(watchdog)
    assert len(fired) == 1, watchdog.notifier.alerts
    body = _body(fired[0])
    assert "NEVER MINTED" in body and "QUARANTINED" in body, body
    assert "lloyd.crt" in body, body
    assert JOURNAL_CMD in body, body


def test_half_a_pair_is_an_absent_pair(tmp_path, key_pem, watchdog):
    """A `.crt` with no `.key` alerts exactly like no pair at all, because that is
    how it reads to the thing being served: `haveTs` in `web/vite.config.ts` is
    `fs.existsSync(cert) && fs.existsSync(key)`, so Vite falls back to `lloyd.crt`
    and a check that looked only at the certificate file would call this healthy."""
    d = _cert_dir(tmp_path)
    _mint(d, key_pem, 40.0)
    (d / f"{HOST}.key").unlink()
    watchdog.check_cert_expiry(d / f"{HOST}.crt")

    fired = _alerts(watchdog)
    assert len(fired) == 1, watchdog.notifier.alerts
    assert "NEVER MINTED" in _body(fired[0]) and "QUARANTINED" in _body(fired[0])


def test_an_unparseable_cert_pages_too_and_says_a_file_is_actually_there(tmp_path,
                                                                         key_pem,
                                                                         watchdog):
    """Bytes that are not a leaf: openssl exits non-zero, or prints no `notAfter=`,
    and the check must not fall through to "fine". The same two causes are named, as
    the contract requires, and the body then says outright that neither fits a path
    that HAS a file at it — the honest reading is a pair published half-written — so
    the paragraph is not a guess a reader has to triage."""
    d = _cert_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{HOST}.crt").write_bytes(b"-----BEGIN CERTIFICATE-----\nnot a leaf\n")
    (d / f"{HOST}.key").write_bytes(b"-----BEGIN PRIVATE KEY-----\nnothing\n")
    watchdog.check_cert_expiry(d / f"{HOST}.crt")

    fired = _alerts(watchdog)
    assert len(fired) == 1, watchdog.notifier.alerts
    body = _body(fired[0])
    assert "NEVER MINTED" in body and "QUARANTINED" in body, body
    assert "A file IS present" in body, body
    assert JOURNAL_CMD in body, body


@pytest.mark.parametrize("breakage", ["no_openssl", "empty_answer"])
def test_no_state_of_the_cert_tree_raises_out_of_the_check(tmp_path, key_pem,
                                                           watchdog, monkeypatch,
                                                           breakage):
    """The check runs on every tick, above every early return, so a raise out of here
    is a tick lost — and past `policy.TICK_ERROR_ALERT_AFTER` consecutive ones the
    tick-error alarm is saying the whole rollback path is being skipped. Two ways the
    read can go wrong are fired in, one from each side of the boundary: openssl
    missing from the box entirely (`FileNotFoundError` out of the spawn, which is NOT
    the same fact as an unreadable certificate and must not escape as one), and an
    openssl that comes back `rc=0` with nothing in stdout — which is precisely what a
    blanket `subprocess.run` interception would leave behind, and the reason this
    watch treats an empty answer as the fault rather than as health.

    Both arms assert an alert AND no exception: the containment is not allowed to be
    bought by saying nothing."""
    crt = _mint(_cert_dir(tmp_path), key_pem, 3.0)
    if breakage == "no_openssl":
        def raiser(argv, *a, **kw):
            raise FileNotFoundError(2, "No such file or directory: 'openssl'")
        monkeypatch.setattr(G.subprocess, "run", raiser)
    else:
        def empty(argv, *a, **kw):
            return subprocess.CompletedProcess(list(argv), 0, stdout=b"", stderr=b"")
        monkeypatch.setattr(G.subprocess, "run", empty)

    watchdog.check_cert_expiry(crt)          # must not raise

    fired = _alerts(watchdog)
    assert len(fired) == 1, watchdog.notifier.alerts
    assert "A file IS present" in _body(fired[0]), _body(fired[0])
    assert JOURNAL_CMD in _body(fired[0])
    assert watchdog.notifier.resolves == []


# ── clause 5: the seat — it runs in the states nothing else runs in ─────────

@pytest.fixture
def live_guardian(tmp_path, monkeypatch):
    """A real `Guardian` over temp state, driven through `tick()`.

    Only the boundaries are faked: the supervisor client, the HTTP probes, the
    sibling watches whose alert paths this node is not about, and `systemctl` — that
    last one intercepted ONLY for argv that starts with `systemctl`, because a tick
    now shells out for something else too and a blanket interception would turn every
    openssl read below into an empty stdout and silently move every node in this file
    onto the unparseable branch. `repo` is a temp dir, so the leaf under test is the
    one this fixture's caller mints, at the default path, with no injection.
    """
    monkeypatch.setattr(policy, "LOG_FILES", ())
    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:8080/health", mcp_url="http://127.0.0.1:2/health",
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

    real = subprocess.run

    def _fake_run(argv, *a, **kw):
        argv = list(argv)
        if argv and argv[0] == "systemctl":
            # `tick()` restarts the supervisor unit once its own streak reaches
            # `SUPERVISORD_DOWN_STREAK`, and the parametrised states below include
            # the one that gets there. Intercepted for that argv alone.
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")
        return real(argv, *a, **kw)

    monkeypatch.setattr(G.subprocess, "run", _fake_run)
    g._test_alerts = alerts
    return g


def _drive_tick(live_guardian, monkeypatch, *, condition: str) -> str:
    """One real `Guardian.tick()` over a fake supervisor and fake HTTP."""
    def probe(url, *a, **kw):
        return {"ok": True, "status": 200, "body": {"status": "ok"}, "error": None,
                "latency_ms": 2.0, "kind": "ok"}

    monkeypatch.setattr(G.probes, "probe", probe)
    # One def, not one per branch: two `def procs` in an if/else is a name the reader
    # has to hold both halves of, and pyright calls it a shadowed declaration.
    def procs():
        if condition == "supervisord_unreachable":
            raise G.SupervisordUnreachable("simulated: no supervisor socket")
        return {}

    monkeypatch.setattr(live_guardian.sup, "all_process_info", procs)
    monkeypatch.setattr(live_guardian.state, "pause_remaining",
                        lambda cap: 600.0 if condition == "paused" else 0.0)
    monkeypatch.setattr(live_guardian.state, "is_broken", lambda: condition == "broken")
    monkeypatch.setattr(live_guardian.state, "current", lambda: None)
    return live_guardian.tick()


@pytest.mark.parametrize("condition",
                         ["armed", "supervisord_unreachable", "broken", "paused"])
def test_the_leaf_is_read_on_a_tick_that_returns_before_the_rollback_logic(
        tmp_path, key_pem, live_guardian, monkeypatch, condition):
    """Clause 5, and the reason it is a clause: the state that hides a certificate is
    a pause. A leaf's `notAfter` advances against the wall clock while the guardian is
    `paused` — a vault tripwire pauses the world, and the pool is stopped by
    `workers.enabled: false` while it stays paused — so a check seated under
    `tick()`'s `infra_down`/`broken`/`paused` returns stops counting down during
    exactly the outage that can outlast the cert.

    One tick in each state, with a real 3-day leaf at the DEFAULT path under a temp
    repo: no injected argument, no patched method, no recorder on the call. The only
    way this returns one page in all four rows is for `tick()` itself to make it."""
    crt = _mint(_cert_dir(Path(live_guardian.repo)), key_pem, 3.0)
    state = _drive_tick(live_guardian, monkeypatch, condition=condition)
    fired = [a for a in live_guardian._test_alerts if len(a) > 1 and a[1] == TITLE]
    assert len(fired) == 1, (condition, state, live_guardian._test_alerts)
    assert JOURNAL_CMD in fired[0][2], fired[0][2]
    assert crt.is_file()

# ── #4593: the confinement is in the sink, so `absent` still pages ────────────

def test_the_absent_leaf_still_pages_under_the_confined_sink(tmp_path, monkeypatch):
    """#4593 clause 4: confining where an alarm may WRITE did not mute WHAT pages.

    The temptation the fix invites is to quiet the page by making
    `check_cert_expiry` bail when `self.repo` has no cert directory — but a leaf
    that was NEVER MINTED on this box also reads `absent`
    (`guardian.py:1898-1902`), and that is one of the two causes the body itself
    names, so bailing on an empty cert dir is the exact false negative the watch
    exists to catch. So this node puts a cert directory that exists and holds no
    pair in a tmp repo — the same shape a pytest trial tree has — and asks whether
    the page still happens while `tests/conftest.py`'s seam has the fan-out aimed
    at a scratch vault.

    Two things are asserted together, and neither is enough alone: the alert is
    raised, naming the never-minted cause; and it is the REAL fan-out that raised
    it, which the confined sink received. A `check_cert_expiry` that had learned to
    skip a missing cert dir would fail the first; a sink confinement that muted the
    alert instead of redirecting it would fail the second.
    """
    import guardian as G
    import notify as notify_module
    import policy

    cert_dir = _cert_dir(tmp_path)
    cert_dir.mkdir(parents=True)
    (cert_dir / ".gitkeep").write_text("", encoding="utf-8")   # a dir, no pair

    scratch = tmp_path / "scratch-vault"
    (scratch / "memory").mkdir(parents=True)
    monkeypatch.setattr(policy, "VAULT_ROOT", str(scratch))

    g = G.Guardian.__new__(G.Guardian)
    g.repo = str(tmp_path)
    g.notifier = notify_module.Notifier(
        ledger=scratch / "ledger.jsonl", state_dir=scratch,
        vault_root=str(scratch))
    g._alert_seen = {}
    g._cert_leaf_alarm = None
    g.last_alert = ""

    g.check_cert_expiry()

    note = scratch / "memory" / (
        datetime.now().date().isoformat() + ".md")
    assert note.is_file(), (
        f"`check_cert_expiry` on a repo whose cert directory holds no pair wrote "
        f"nothing to the vault it was confined to ({scratch}) — a check that skips "
        f"a missing cert dir is the false negative this item forbids: an "
        f"unminted leaf reads the same way")
    text = note.read_text(encoding="utf-8")
    assert f"## Self-mod guardian: {TITLE}" in text, text[:800]
    assert "NEVER MINTED" in text, (
        "the page lost the cause that makes an unminted leaf distinguishable from a "
        f"quarantined one:\n{text[:800]}")
    assert str(cert_dir) in text, (
        "the page does not name the directory it looked in, so a reader cannot tell "
        f"a trial tree from this machine:\n{text[:800]}")
