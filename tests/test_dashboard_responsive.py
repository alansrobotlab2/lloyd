"""#1685 — dashboard panels must fit their grid track at every viewport.

The defect: two sections rendered about twice as wide as their own column on a
phone (651 px inside 288 px at a 320 px viewport). A grid item's `min-width` is
`auto`, so each card was floored at its content and one unbreakable line — the
worker stat row's four columns — widened the track and every card sharing it.

This measures the REAL page in headless chromium at the widths the review rung
named, by calling the same function the maintenance probe calls
(`scripts/maintenance/dashboard_mobile_probe.py`). One measurement, two
consumers: no second copy of the geometry JS that can drift from the instrument.

`test_dashboard_mobile_sizing.py` pins the two class declarations that carry the
fix; this pins the geometry they produce. Neither is redundant: a class string
can be present while the page still overflows, and green geometry says nothing
about which rule is holding it.

What it skips, and why: `web/node_modules` is untracked and chromium may be
missing, and a box where nobody ran `npm install` must skip and SAY so rather
than fail every frontend round for someone else's absent install (the
`_vitest_run` precedent in scripts/automod/gate.py). Every path that does run
asserts a section-count floor before trusting a zero-overflow number.

What a skip may NOT be, since #1691: silent. Six pins skipping on an absent vite
printed `ssssss` and exited 0, which is the same report a green run gives — and
the gate could not see it, because its skip budget is a suite total
(`scripts/automod/gate.py:77-80`, live runs at 31-32 of 40) and the partial-run
branch (`gate.py:1518-1529`) applies no floor at all. So the file now keeps a
ledger (`tests/dashboard_pins.py`) of which pins actually measured something and
reports the ones that did not: undeclared dependency -> exit 0 plus a named
`DASHBOARD_PINS_NOT_EXECUTED` finding that the gate prints in its rung detail;
declared dependency (the gate set it where it made `node_modules` reachable) ->
a failure naming the pins. A page that fails to LOAD is a failure too, never a
skip: only an absent chromium stays a skip. And the desktop pin re-measures once
before it fails, so its known xdist flake is tolerated by the test that owns it
rather than by the gate's serial retry.

The nodes that pin those rules fake the thing that would need a frontend and drive the
real code that consumes it: `test_a_page_that_will_not_load_is_a_failure` and
`test_an_absent_chromium_still_skips` hand `_measure` a probe object instead of a
browser, and `test_a_declared_teardown_fails_AND_stops_the_child` and
`test_an_undeclared_teardown_stops_the_child_and_stays_green` drive the module
fixture's teardown through a stub dev server. None of them needs vite, chromium or
`node_modules` — which is what lets the review grader execute them in a detached
snapshot that has none, the gap its verdict on #1685's clause 1 had to work around.
"""
from __future__ import annotations

import importlib.util
import socket
import ssl
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashboard_pins as dp  # noqa: E402

pytest.importorskip("playwright.sync_api", reason="playwright is not installed")

from playwright.sync_api import sync_playwright  # noqa: E402

PINS = dp.PinLedger(Path(__file__))

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PROBE_PATH = ROOT / "scripts" / "maintenance" / "dashboard_mobile_probe.py"

PHONE_WIDTHS = [320, 360, 390, 414]
DESKTOP_WIDTH = 1280

# Clause 1's floor. The dashboard has seven sections; demanding six means a page
# that rendered partially cannot hand back "no section overflows" over a short
# list and read as a pass — the exact vacuous verdict a cold vite compile once
# produced at 414 px.
MIN_SECTIONS = 6

# Clause 2's desktop tracks: measured on the fixed tree and confirmed by the
# review rung's own probe pass. Keyed by heading prefix, valued by the set of
# card widths that section's grid legitimately produces — so a section whose
# cards are data-dependent (System renders both four-column tiles and
# three-column meters) is pinned to widths it actually uses rather than to a
# panel count that changes with the fleet.
#
# Every entry here MUST have a non-empty measured list at 1280 px, and the test
# below asserts exactly that before comparing a single width. The first version
# of this pin listed "Services": [176] while the probe only collected
# `.rounded-lg` children — and Services' children are HealthPills, which are
# `.rounded-md`. So that entry compared an empty list against [176] and passed
# forever, whatever happened to the layout: a check with a zero denominator is
# not a check, and it took a reviewer to notice a green line that asserted
# nothing.
DESKTOP_TRACKS: dict[str, list[int]] = {
    "vLLM engines": [358],
    "Lloyd agent": [358],
    "Subagents": [358],
    "System": [173, 235],
    "Services": [176],
    "Automation & work": [235, 728],
    "Tokens": [728],
}
TOLERANCE_PX = 1


def _port_is_tls(port: int) -> bool:
    """True if the port answers a TLS handshake (vite served its cert)."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as sock:
            with ctx.wrap_socket(sock, server_hostname="localhost"):
                return True
    except (ssl.SSLError, OSError):
        return False


def _load_probe():
    spec = importlib.util.spec_from_file_location("dmm_probe", PROBE_PATH)
    assert spec and spec.loader, f"probe missing at {PROBE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(autouse=True)
def _pin_identity(request):
    """Which pin is running, so `_measure` can credit it without threading the
    node id through six call sites (#1691)."""
    PINS.begin(request.node.name)
    yield


def _vite_binary() -> Path:
    """The vite this tree's frontend ships (a path that may not exist)."""
    return WEB / "node_modules" / ".bin" / "vite"


def _stop_vite(proc) -> None:
    """Stop the dev server this module started, and wait for it to have gone.

    Its own function because the ORDER of these calls is the bug this fixes: the
    previous round reported the pin accounting first and terminated second, and in a
    declared run the accounting raises, so this code was never reached and the vite
    server started for the module sat holding its port for the rest of the gate run
    (the review rung's advisory on round SM_20260928_000124).
    """
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _wait_for_vite(proc, port: int, deadline_s: float = 60.0):
    """Poll a freshly-spawned vite until it answers.

    Returns `(url, None)` once the port is listening, or `(None, reason)` when the
    process died or never bound — exactly one half is ever set, so a caller that
    takes the URL still knows which failure it did NOT have.

    The scheme is a property of the tree, not of this test: `web/vite.config.ts`
    serves TLS when it finds a cert under `agent-services/cert` and plain HTTP when
    it does not, so the port is asked which one it is. (The first version hardcoded
    `http://` while its comment claimed otherwise, which the reviewer caught — a
    worktree DOES have that cert directory.)
    """
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return None, f"vite exited early (rc={proc.returncode}) on port {port}"
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                scheme = "https" if _port_is_tls(port) else "http"
                return f"{scheme}://127.0.0.1:{port}/", None
        except OSError:
            pass
        time.sleep(0.5)
    return None, f"vite never listened on port {port}"


def dashboard_serving(session, ledger=None):
    """The generator behind the `dashboard_url` fixture, split out so the teardown
    has an in-round executor that needs no vite, no chromium and no `node_modules`.

    Everything that stops the pins runs through the ledger first (#1691): `report`
    prints the named no-execution finding, or raises when the environment declared
    the dependencies reachable. The skip itself stays a skip — a box where nobody
    ran `npm install` must not go red — it may just no longer be silent.

    Two details are the whole point of the split:

    * the child is stopped BEFORE the accounting is done, because in a declared run
      the accounting raises and anything written after it would not run;
    * `ledger` is a parameter for the same reason. A node that drove this with the
      module's own ledger would flip its once-only `_reported` flag and silence the
      real finding for the rest of the session.
    """
    pins = PINS if ledger is None else ledger
    vite = _vite_binary()
    if not vite.exists():
        reason = ("web/node_modules has no vite — run npm install in web/ "
                  "(the gate symlinks it only for rounds that touch web/)")
        pins.note(reason)
        # Declared environments get a FAILURE from here; undeclared ones get the
        # named finding and the skip that has always been correct for them.
        pins.report(session)
        pytest.skip(reason)

    port = _free_port()
    proc = subprocess.Popen(
        [str(vite), "--port", str(port), "--host", "127.0.0.1", "--strictPort"],
        cwd=str(WEB),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    served = False
    try:
        url, reason = _wait_for_vite(proc, port)
        if url is None:
            # No accounting call here on purpose: the `finally` below stops the
            # child and reports, so a declared run gets its failure from one place
            # and the child is stopped on the way out of it either way.
            pins.note(reason)
            pytest.skip(reason)
        served = True
        yield url
    finally:
        _stop_vite(proc)
        # The last accounting point, and it can only come second: every pin that
        # did not measure did so for a reason `_measure` has already noted, and
        # `report` decides from it whether a declared environment turns this into a
        # failure or prints the finding and lets the skip stand (#1691).
        pins.report(session, None if served else pins.reason)


@pytest.fixture(scope="module")
def dashboard_url(request):
    """Serve THIS tree's frontend, not whatever dev server happens to be up.

    Probing the long-running :5173 would grade the live checkout instead of the
    tree under test — the mistake recorded on #1601, and the whole difference
    between a test that pins a diff and one that measures someone else's tree.

    The body lives in `dashboard_serving` so that the declared-mode teardown — the
    branch that raises, and the branch that used to orphan the vite child — is
    executable by a node in this file (#1691 review advisory).
    """
    yield from dashboard_serving(request.session)


def _measure(url, width):
    """One real browser measurement of one real page.

    Two outcomes used to be indistinguishable from a pass (#1691):

    * A page that failed to LOAD was a SKIP. That is the defect this item filed:
      the same catch-all that once swallowed `ERR_UNSAFE_PORT` now swallows
      whatever else goes wrong — a 500 from the dev server, a React error that
      blanks the root node, a timeout — and the suite reports a skip, which reads
      as green from here and as "not applicable" from the gate. A load failure is
      a FAILURE, with the browser's own words in the message.
    * Nothing credited the pin that measured, so the file could not tell a run
      that asserted geometry from one that skipped everything.

    An absent chromium is still a skip — that is the one dependency a box is
    allowed not to have, and the reason the finding names chromium rather than
    saying only that nothing ran.
    """
    probe = _load_probe()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            # The reason is BUILT from the helper's constant, not typed to match it:
            # `declared_covers` classifies a stop as one the gate's declaration never
            # promised by reading this prefix, and a word drifting here would silently
            # turn a legitimate skip into a red gate on every declared round.
            why = f"{dp.BROWSER_MISSING}: {str(exc)[:120]}"
            PINS.note(why)
            pytest.skip(why)
        try:
            try:
                res = probe.measure_page(browser, url, width)
            except Exception as exc:
                pytest.fail(f"dashboard did not load: {str(exc)[:160]}")
            PINS.measured()
            return res
        finally:
            browser.close()


def _assert_verdict_available(res, width):
    """Refuse to grade an unrendered page. `overflow: 0` over an empty list of
    sections is the same number a healthy dashboard produces."""
    assert "error" not in res, f"{width}px: probe error {res['error']}"
    n = len(res["sections"])
    assert n >= MIN_SECTIONS, (
        f"{width}px: only {n} dashboard sections measured (floor {MIN_SECTIONS}) — "
        "no verdict available, which is not the same thing as a pass"
    )


# --- clause 1 -------------------------------------------------------------

@pytest.mark.parametrize("width", PHONE_WIDTHS)
def test_no_section_overflows_its_box_on_a_phone(dashboard_url, width):
    res = _measure(dashboard_url, width)
    _assert_verdict_available(res, width)
    bad = [f"'{s['head']}' {s['sw']}px inside {s['cw']}px"
           for s in res["sections"] if s["overflow"] > 0]
    assert not bad, f"at {width}px these sections overflow: {bad}"
    assert res["pageOverflow"] <= 0, (
        f"at {width}px the page itself scrolls sideways by {res['pageOverflow']}px"
    )


# --- clause 2 -------------------------------------------------------------

def _desktop_complaints(res) -> list[str]:
    """Every way one 1280 px measurement disagrees with the pinned tracks.

    Returns complaints instead of asserting so the caller can take a second
    measurement (#1691); an empty list is the only verdict that means "agree".
    """
    complaints: list[str] = []
    try:
        _assert_verdict_available(res, DESKTOP_WIDTH)
    except AssertionError as exc:
        # The section floor tripped: no verdict exists, which is not a pass and
        # is exactly the transient a second measurement is meant to settle.
        return [str(exc)]
    if res["pageOverflow"] > 0:
        complaints.append(f"the page scrolls sideways by {res['pageOverflow']}px")

    checked = 0
    for sec in res["sections"]:
        allowed = next((w for head, w in DESKTOP_TRACKS.items()
                        if sec["head"].startswith(head)), None)
        if allowed is None:
            complaints.append(f"section '{sec['head']}' is not in DESKTOP_TRACKS")
            continue
        # A pinned section with NOTHING measured under it is a complaint, not a
        # skip. `continue` on an empty list here is how the first version of
        # this test came to assert nothing at all about Services: its children
        # are `.rounded-md` pills, the probe collected only `.rounded-lg`, and
        # the empty list sailed past every check below.
        cards = list(sec["panels"]) + list(sec.get("pills") or [])
        if not cards:
            complaints.append(
                f"section '{sec['head']}' is pinned to {allowed} but the probe "
                "measured no cards under it — the pin is vacuous, fix the selector"
            )
            continue
        checked += len(cards)
        for panel in cards:
            if not any(abs(panel["w"] - a) <= TOLERANCE_PX for a in allowed):
                complaints.append(
                    f"'{sec['head']}' card is {panel['w']}px, allowed {allowed}"
                )
            if panel["sw"] > panel["w"] + TOLERANCE_PX:
                complaints.append(
                    f"'{sec['head']}' card clips its content at {panel['w']}px"
                )
    if checked < 10:
        complaints.append(
            f"only {checked} desktop cards measured; too few for a verdict")
    return complaints


def desktop_track_verdict(measure) -> dp.ReMeasured:
    """The 1280 px pin's flake rule, as a function so it can be tested without a
    browser (#1691).

    This pin is the one node in the suite the gate's ledger records as
    "failed only under parallel load and passed serially" (`promotions.jsonl`,
    rung `tests`, 2026-09-27T23:29:44Z): under xdist-8 it fails, and the gate's
    serial retry (`gate.py:1443-1467`) turns that back into a pass without the
    pin ever saying so. Retrying here makes the tolerance explicit where the flake
    lives — ONE disagreeing measurement is load, a SECOND is a layout defect — and
    caps the cost at two page loads.
    """
    return dp.grade_once_then_twice(measure, _desktop_complaints)


def test_desktop_cards_keep_their_track_widths(dashboard_url):
    result = desktop_track_verdict(lambda: _measure(dashboard_url, DESKTOP_WIDTH))
    assert result.clean, (
        f"the 1280 px desktop pin disagrees with its pinned tracks: "
        f"{result.complaints} ({result.detail()}; first measurement said "
        f"{list(result.first)})"
    )


# --- clause 3 -------------------------------------------------------------

def test_worker_stat_row_fits_two_columns_below_sm(dashboard_url):
    """The row that held the whole track open: at 320 px it must compute to two
    columns, and no label may reach past its panel's right edge."""
    res = _measure(dashboard_url, 320)
    row_section = next((s for s in res["sections"]
                        if s["head"].startswith("Automation")), None)
    assert row_section, "the Automation & work section is absent"

    probe = _load_probe()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 320, "height": 844},
                                  is_mobile=True, has_touch=True)
        try:
            page = ctx.new_page()
            page.goto(dashboard_url, wait_until="load", timeout=45000)
            page.wait_for_timeout(2500)
            probe.open_dashboard(page)
            page.wait_for_timeout(3000)
            out = page.evaluate(
                """() => {
                  const lab = [...document.querySelectorAll('div')]
                      .find(d => d.textContent.trim() === 'In flight');
                  if (!lab) return {found: false};
                  const row = lab.parentElement.parentElement;
                  const card = row.closest('.rounded-lg');
                  const tracks = getComputedStyle(row).gridTemplateColumns
                      .split(' ').filter(Boolean).length;
                  const cardRight = card.getBoundingClientRect().right;
                  const worst = Math.max(...[...row.children]
                      .map(c => c.getBoundingClientRect().right));
                  return {found: true, tracks, cardRight, worst,
                          labels: [...row.children].map(c =>
                              c.textContent.trim().slice(0, 14))};
                }"""
            )
        finally:
            ctx.close()
            browser.close()

    assert out.get("found"), "the worker stat row ('In flight' tile) never rendered"
    assert out["tracks"] <= 2, (
        f"stat row computes to {out['tracks']} columns at 320px: {out['labels']}"
    )
    assert out["worst"] <= out["cardRight"] + 1, (
        f"stat labels reach {out['worst']:.0f}px, past the panel's right edge at "
        f"{out['cardRight']:.0f}px"
    )


# --- #1691 clause 3: a page that will not load is a FAILURE, not a SKIP ---------
#
# These two nodes drive `_measure` with the browser itself faked, so they run on a
# machine with no chromium and no `node_modules` — which is the whole point: the
# rule they pin is about how a failure is REPORTED, and a node that skipped on the
# box where the rule is being reviewed would leave the defect exactly where it was.

def _outcome(fn):
    """Classify one `_measure` call as `fail`, `skip` or `ok`, with its message.

    Written this way on purpose, and it is the whole of why these nodes are not
    themselves the silent-green defect: `pytest.raises(pytest.fail.Exception)` cannot
    catch a regression to `pytest.skip`, because a `Skipped` raised inside the `raises`
    block escapes and pytest records the WHOLE NODE as skipped — green. A test that
    reports "skipped" when the code under test wrongly skips is exactly the bug #1691
    is about, so the outcome is classified and then asserted: a load failure that comes
    back as a skip is a FAILURE of this test, not a pass.
    """
    try:
        fn()
    except pytest.fail.Exception as exc:
        return "fail", str(exc)
    except pytest.skip.Exception as exc:      # noqa: B013 - the point is to catch it
        return "skip", str(exc)
    return "ok", None


class _FakeBrowser:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakePlaywright:
    """Stands in for `sync_playwright()`. `launch_raises` is what an absent
    chromium looks like from here: an exception out of `launch()`."""

    def __init__(self, launch_raises: Exception | None = None):
        self.browser = _FakeBrowser()
        self._launch_raises = launch_raises

    def __enter__(self):
        outer = self

        class _Chromium:
            def launch(self_inner):
                if outer._launch_raises:
                    raise outer._launch_raises
                return outer.browser

        self.chromium = _Chromium()
        return self

    def __exit__(self, *exc):
        return False


def test_a_page_that_will_not_load_is_a_failure(monkeypatch):
    """`probe.measure_page` raising must FAIL the pin, with the browser's own
    words in the message.

    Until #1691 the catch-all here called `pytest.skip`, so a page that returned a
    500, blanked its root node or timed out was reported the same way an absent
    dependency was — and the same catch had already swallowed one real load error
    (`Page.goto: net::ERR_UNSAFE_PORT at https://127.0.0.1:1/`) while the suite
    went green. A dashboard that cannot load is precisely what these pins exist to
    catch.
    """
    boom = RuntimeError("Page.goto: net::ERR_UNSAFE_PORT at https://127.0.0.1:1/")

    class _BoomProbe:
        def measure_page(self, browser, url, width):
            raise boom

    monkeypatch.setattr(sys.modules[__name__], "_load_probe", lambda: _BoomProbe())
    monkeypatch.setattr(sys.modules[__name__], "sync_playwright",
                        lambda: _FakePlaywright())

    kind, message = _outcome(lambda: _measure("https://127.0.0.1:1/", 320))
    assert kind == "fail", (
        f"a page that will not load must FAIL the pin; it reported {kind!r}"
        + (" — a skip here is the defect this item filed, reported green again"
           if kind == "skip" else ""))
    assert "ERR_UNSAFE_PORT" in message, (
        f"the failure did not carry the load error's text: {message!r}")
    assert "dashboard did not load" in message, message


def test_an_absent_chromium_still_skips(monkeypatch):
    """The other half of the rule, and the reason the finding names chromium: a box
    that never installed the browser is not a box with a broken dashboard, and the
    gate must not fail every non-frontend round for someone else's absent install.
    """
    class _UnusedProbe:
        def measure_page(self, browser, url, width):  # pragma: no cover
            raise AssertionError("must not be reached without a browser")

    monkeypatch.setattr(sys.modules[__name__], "_load_probe", lambda: _UnusedProbe())
    monkeypatch.setattr(
        sys.modules[__name__], "sync_playwright",
        lambda: _FakePlaywright(
            launch_raises=RuntimeError(
                "Executable doesn't exist at ~/.cache/ms-playwright/chromium")))

    noted: list[str] = []
    monkeypatch.setattr(PINS, "note", noted.append)

    kind, message = _outcome(lambda: _measure("http://127.0.0.1:1/", 320))
    assert kind == "skip", (
        f"an absent chromium must still SKIP — a box that never installed the browser "
        f"is not a box with a broken dashboard; it reported {kind!r}")
    assert "chromium" in message.lower(), message
    assert noted and "chromium" in noted[0].lower(), (
        "the skip recorded no reason, so the no-execution finding would have to "
        f"say only that nothing ran: {noted}")


def test_a_page_that_will_not_load_is_credited_as_never_having_run(monkeypatch):
    """The accounting half of clause 3: a failed load must not count as a
    measurement. If it did, the ledger would report the pin as run and the finding
    #1691 exists to print would go quiet on exactly the run that needs it — the
    same shape as a count decremented on one path and not the other (#1541), and
    invisible in the report unless a node says so.
    """
    # A name no real pin can have credit for: the ledger is module-scoped, so by the
    # time this node runs on a box WITH a frontend the real 320 px pin has already
    # measured, and asserting on a name the session already credited would test
    # nothing. The rule under test is about the CURRENT pin, not its identity.
    pin = "test_a_page_that_will_not_load_is_credited_as_never_having_run"

    class _BoomProbe:
        def measure_page(self, browser, url, width):
            raise RuntimeError("net::ERR_CONNECTION_REFUSED")

    monkeypatch.setattr(sys.modules[__name__], "_load_probe", lambda: _BoomProbe())
    monkeypatch.setattr(sys.modules[__name__], "sync_playwright",
                        lambda: _FakePlaywright())
    monkeypatch.setattr(PINS, "note", lambda reason: None)
    monkeypatch.setattr(PINS, "_current", pin)
    assert not PINS.did_measure(pin), "the fixture itself pre-credited the pin"

    kind, _message = _outcome(lambda: _measure("http://127.0.0.1:1/", 320))
    assert kind == "fail", (
        f"a failed load must fail the pin before anything is credited; it reported "
        f"{kind!r}")
    assert not PINS.did_measure(pin), (
        "a pin whose page never loaded was credited with a measurement, so the "
        "no-execution finding would stay silent on a broken dashboard")


# ── #1691 clause 4: the 1280 px pin tolerates one load-tripped measurement ──────
#
# The ledger of record says this pin flinches: `12757 passed, 1 xfailed, 31 skipped
# (8 workers); 2 failed only under parallel load and passed serially:
# tests/test_dashboard_cold_render.py::test_one_cold_dashboard_cycle_beats_the_budget,
# tests/test_dashboard_responsive.py::test_desktop_cards_keep_their_track_widths`
# (`~/.local/state/lloyd-automod/promotions.jsonl`, rung `tests`, 2026-09-27T23:29:44Z).
# The gate's serial re-run (`scripts/automod/gate.py:1443-1467`) turned that failure
# into a pass without the pin ever saying anything, so a real regression could hide
# behind the same retry. These nodes run with NO browser: they script `measure`, so
# they pin the rule itself rather than a machine's load.

def _cards(widths: dict[str, int], n: int = 6):
    """A synthetic 1280 px measurement: `n` sections with one card each.

    Built to satisfy the section floor the pin itself imposes, so a scripted
    measurement is a page the pin is willing to grade.
    """
    sections = []
    for head, allowed in list(DESKTOP_TRACKS.items())[:n]:
        w = widths.get(head, allowed[0])
        sections.append({
            # `sw` is the card's CONTENT width: a content width above the box is the
            # clipped-content complaint, so a clean synthetic page has to sit inside
            # its own box or the fixture would be reporting a real defect.
            "head": head, "overflow": 0, "sw": w - 10, "cw": w,
            "panels": [{"w": w, "sw": w - 10} for _ in range(2)],
            "pills": [],
        })
    return {"sections": sections, "pageOverflow": 0}


def _good_desktop():
    return _cards({})


def test_a_clean_first_desktop_measurement_is_not_measured_twice():
    """The cost of the rule stays bounded: one agreeing measurement ends the pin, so
    a healthy dashboard does not pay a second page load and a healthy round does not
    look twice as patient as it is.
    """
    calls: list[int] = []

    def measure():
        calls.append(1)
        return _good_desktop()

    result = desktop_track_verdict(measure)
    assert result.clean, result.detail()
    assert len(calls) == 1, f"a healthy measurement was repeated: {len(calls)}"
    assert not result.retried, result.detail()


def test_one_disagreeing_desktop_measurement_is_re_measured_and_a_second_pass_clears_it():
    """The flake half. Under xdist-8 this pin sometimes fails and passes serially;
    one measurement that trips the section floor or reports an off-track card is now
    re-measured once, and the pin passes on the agreement. That moves the retry from
    the gate's ledger, where nobody reads it, to the test, where the reason is named.
    """
    readings = [_cards({"Services": 999}), _good_desktop()]   # first is off-track
    calls: list[int] = []

    def measure():
        calls.append(1)
        return readings[len(calls) - 1]

    result = desktop_track_verdict(measure)
    assert len(calls) == 2, (
        f"a disagreeing first measurement was not followed by exactly one "
        f"re-measurement (took {len(calls)})")
    assert result.retried, (
        "the result does not admit it re-measured, so the flake would be invisible "
        "in the failure output when it does fail")
    assert result.clean, f"the agreeing second measurement did not clear it: {result.detail()}"
    assert result.first, (
        "the first measurement's complaint was dropped, so a reviewer cannot tell a "
        "load flinch from a layout defect")


def test_the_section_floor_tripping_once_is_re_measured_too():
    """The other way the flake shows up: a page that had not painted its six sections
    when the probe looked. `_assert_verdict_available` is the floor, and the pin must
    treat tripping it as a disagreement to re-measure — not as a pass (an empty
    section list reports zero overflows) and not as a terminal failure.
    """
    thin = {"sections": [{"head": "Tasks", "overflow": 0, "sw": 100, "cw": 200,
                          "panels": [{"w": 100, "sw": 90}], "pills": []}],
            "pageOverflow": 0}
    readings = [thin, _good_desktop()]
    calls: list[int] = []

    def measure():
        calls.append(1)
        return readings[len(calls) - 1]

    result = desktop_track_verdict(measure)
    assert len(calls) == 2, f"the floor tripped and nothing was re-measured: {calls}"
    assert result.clean, result.detail()


def test_two_disagreeing_desktop_measurements_fail_and_say_what_both_saw():
    """The failure half, and the cap. A layout defect is a defect in both
    measurements, so the pin fails — after exactly two measurements, never a third —
    and the message carries the first and second complaint so the reader can see the
    disagreement rather than only the last one.
    """
    readings = [_cards({"Services": 999}), _cards({"Services": 997})]
    calls: list[int] = []

    def measure():
        calls.append(1)
        return readings[min(len(calls) - 1, len(readings) - 1)]

    result = desktop_track_verdict(measure)
    assert len(calls) == 2, (
        f"a second disagreeing measurement must not trigger a third: {len(calls)}")
    assert not result.clean, "two disagreeing measurements were reported as a pass"
    assert result.complaints and result.first, (
        "the failure kept neither measurement's complaint: "
        f"{result.detail()}")


def _pin_with_readings(monkeypatch, readings):
    """Point the 1280 px PIN BODY at scripted readings and return its call log.

    This drives `test_desktop_cards_keep_their_track_widths` (:340-346) itself — the
    pin's own `desktop_track_verdict(lambda: _measure(...))` and its own `assert
    result.clean` — rather than the helper, which is the difference between pinning the
    clause and pinning a reimplementation of it. A pin that delegated to the shared
    mechanism while handing it a grade that never complained would keep every
    helper-level node green and be back to trusting one reading of a flaky page.
    """
    calls: list[int] = []

    def fake_measure(url, width):
        calls.append(width)
        return readings[min(len(calls) - 1, len(readings) - 1)]

    monkeypatch.setattr(sys.modules[__name__], "_measure", fake_measure)
    return calls


def test_the_desktop_pin_itself_re_measures_a_flinch_and_passes_on_the_second_reading(monkeypatch):
    """Clause 4 through the pin: one off-track reading at 1280 px is re-measured once,
    and the agreeing second reading lets the PIN pass — the retry that used to belong
    to the gate now belongs to this test, and the pin body's assert is what decides.
    """
    calls = _pin_with_readings(monkeypatch, [_cards({"Services": 999}), _good_desktop()])
    test_desktop_cards_keep_their_track_widths("https://127.0.0.1:5173/")
    assert calls == [DESKTOP_WIDTH, DESKTOP_WIDTH], (
        f"the pin was expected to measure 1280 px twice (one disagreeing reading, then "
        f"the re-measure); it measured {calls}")


# ── #1691: the declared-mode teardown, executed in-round ───────────────────────────
#
# The review rung's second advisory on round SM_20260928_000124: every declared node
# written so far died in fixture SETUP, because neither the reviewer's detached
# snapshot nor a plain checkout has `web/node_modules`, so the one branch that RAISES
# — vite served, pins did not measure, the environment declared the dependencies —
# had no executor at all. That is also the branch that used to orphan the vite child:
# `report()` was the first statement of the `finally` and `terminate()` was the second.
#
# These nodes fake the served dashboard, so they run with no vite, no chromium and no
# `node_modules` — which is the point. The rule they pin is about what a teardown DOES
# when it fails, not about a machine that can serve a page.

class _StubItem:
    """Just enough of a collected item for the ledger's `pins()`."""

    def __init__(self, module_path: Path, name: str):
        self.name = name
        self.fixturenames = (dp.PIN_FIXTURE,)
        # Must be THIS file's path: the ledger matches pins by resolved path, and a
        # made-up one gives a denominator of zero, which is not a check (#1614).
        self.fspath = Path(module_path)


class _StubSession:
    def __init__(self, module_path: Path, names: list[str]):
        self.items = [_StubItem(module_path, n) for n in names]


class _FakeChild:
    """The vite subprocess as the teardown sees it: records the stop, never dies."""

    def __init__(self):
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self):
        self.killed = True


def _served_fixture(monkeypatch, ledger: dp.PinLedger):
    """Drive `dashboard_serving` through a served dashboard and a pin-less module.

    Returns `(generator, child)`: pull the URL with `next()`, run the teardown with
    `.close()`. Everything that would need a frontend — the vite binary, the free
    port, the process, the readiness wait — is faked, so the only thing left under
    test is the teardown's own two statements and their order.
    """
    child = _FakeChild()
    monkeypatch.setattr(sys.modules[__name__], "_vite_binary", lambda: Path(sys.executable))
    monkeypatch.setattr(sys.modules[__name__], "_free_port", lambda: 5199)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(
        sys.modules[__name__], "_wait_for_vite",
        lambda proc, port, deadline_s=60.0: (f"https://127.0.0.1:{port}/", None))
    session = _StubSession(Path(__file__), ["test_a_pin_that_never_measured[1]",
                                            "test_a_pin_that_never_measured[2]"])
    return dashboard_serving(session, ledger), child


def test_a_declared_teardown_fails_AND_stops_the_child(monkeypatch):
    """The order the advisory asked for, pinned in both directions: the declared
    module whose pins measured nothing must FAIL (#1691 clause 2's teardown half), and
    the dev server it started must still be terminated — a failure that leaks a
    process holding a port is a new defect paid for an old one.

    `close()` is how a generator fixture is torn down, and a `Failed` raised inside a
    generator being closed propagates out of the call, so this asserts the real
    behaviour rather than a paraphrase of it.
    """
    ledger = dp.PinLedger(Path(__file__))
    gen, child = _served_fixture(monkeypatch, ledger)
    assert next(gen) == "https://127.0.0.1:5199/", "the fixture never served a URL"
    monkeypatch.setenv(dp.DECLARED_ENV, "1")

    kind, message = _outcome(gen.close)
    assert kind == "fail", (
        "a declared module that served a dashboard and measured nothing must fail at "
        f"teardown — otherwise the declaration is decorative; it reported {kind!r}")
    assert dp.FINDING in message, (
        f"the declared teardown failed without naming the finding: {message!r}")
    assert child.terminated and child.waited, (
        "the accounting raised before the vite child was terminated, so the dev "
        f"server this module started is still holding its port: {vars(child)}")
    assert ledger.reported, (
        "the teardown never reached the ledger, so the finding was never computed")


def test_an_undeclared_teardown_stops_the_child_and_stays_green(monkeypatch):
    """The same teardown with nobody having promised anything: the child is stopped,
    the finding is SPOKEN, and NOTHING raises — the fix is visibility, and a box with
    no `npm install` still has to go green.

    `pytest.warns` rather than a plain call, for two reasons: it asserts the finding is
    really emitted, and it CONSUMES the warning. A fabricated stub finding left in this
    file's output would be scooped up by the gate's own `_pin_findings` scan and printed
    in the rung detail of every future round as though the dashboard pins had not run —
    a false alarm from a test, which is the kind of noise that gets a real signal
    ignored.
    """
    ledger = dp.PinLedger(Path(__file__))
    gen, child = _served_fixture(monkeypatch, ledger)
    assert next(gen) == "https://127.0.0.1:5199/"
    monkeypatch.delenv(dp.DECLARED_ENV, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kind, _message = _outcome(gen.close)
    assert kind == "ok", (
        f"an undeclared teardown must not fail the run; it reported {kind!r}")
    assert child.terminated, "the vite child outlived an ordinary green module"
    assert any(dp.FINDING in str(w.message) for w in caught), (
        "the undeclared teardown stayed silent instead of naming the pins it did not "
        f"measure: {[str(w.message) for w in caught]}")


def test_the_desktop_pin_itself_fails_after_two_disagreeing_readings(monkeypatch):
    """And the retry is not a veto: two off-track measurements make the PIN BODY
    itself fail, naming the card width that missed its track. This is the node that
    dies if the pin hands `grade_once_then_twice` a grade that never complains — with
    such a grade the pin would report a clean verdict and this call would return
    quietly instead of raising.
    """
    calls = _pin_with_readings(monkeypatch, [_cards({"Services": 999})])
    with pytest.raises(AssertionError) as caught:
        test_desktop_cards_keep_their_track_widths("https://127.0.0.1:5173/")
    message = str(caught.value)
    assert "999" in message, (
        f"the pin's failure does not name the offending card width: {message!r}")
    assert calls == [DESKTOP_WIDTH, DESKTOP_WIDTH], (
        f"a persistent defect must be measured exactly twice, never once and never "
        f"a third time: {calls}")
