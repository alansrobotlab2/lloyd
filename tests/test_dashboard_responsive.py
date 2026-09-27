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
"""
from __future__ import annotations

import importlib.util
import socket
import ssl
import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright is not installed")

from playwright.sync_api import sync_playwright  # noqa: E402

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


@pytest.fixture(scope="module")
def dashboard_url():
    """Serve THIS tree's frontend, not whatever dev server happens to be up.

    Probing the long-running :5173 would grade the live checkout instead of the
    tree under test — the mistake recorded on #1601, and the whole difference
    between a test that pins a diff and one that measures someone else's tree.
    """
    vite = WEB / "node_modules" / ".bin" / "vite"
    if not vite.exists():
        pytest.skip("web/node_modules has no vite — run npm install in web/")

    port = _free_port()
    proc = subprocess.Popen(
        [str(vite), "--port", str(port), "--host", "127.0.0.1", "--strictPort"],
        cwd=str(WEB),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = None
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.skip(f"vite exited early (rc={proc.returncode}) on port {port}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    # web/vite.config.ts serves TLS when it finds a cert under
                    # agent-services/cert and plain HTTP when it does not, so
                    # the scheme is a property of the tree, not of this test:
                    # ask the port which one it is rather than assuming. (The
                    # first version hardcoded http:// and its comment claimed
                    # otherwise, which the reviewer caught — a worktree DOES
                    # have that cert directory.)
                    scheme = "https" if _port_is_tls(port) else "http"
                    url = f"{scheme}://127.0.0.1:{port}/"
                    break
            except OSError:
                pass
            time.sleep(0.5)
        if not url:
            pytest.skip(f"vite never listened on port {port}")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _measure(url, width):
    probe = _load_probe()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"chromium is not available: {str(exc)[:120]}")
        try:
            try:
                return probe.measure_page(browser, url, width)
            except Exception as exc:
                pytest.skip(f"dashboard did not load: {str(exc)[:160]}")
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

def test_desktop_cards_keep_their_track_widths(dashboard_url):
    res = _measure(dashboard_url, DESKTOP_WIDTH)
    _assert_verdict_available(res, DESKTOP_WIDTH)
    assert res["pageOverflow"] <= 0, "the page scrolls sideways at 1280px"

    offenders: list[str] = []
    checked = 0
    for sec in res["sections"]:
        allowed = next((w for head, w in DESKTOP_TRACKS.items()
                        if sec["head"].startswith(head)), None)
        if allowed is None:
            offenders.append(f"section '{sec['head']}' is not in DESKTOP_TRACKS")
            continue
        # A pinned section with NOTHING measured under it is a failure, not a
        # skip. `continue` on an empty list here is how the first version of
        # this test came to assert nothing at all about Services: its children
        # are `.rounded-md` pills, the probe collected only `.rounded-lg`, and
        # the empty list sailed past every check below.
        cards = list(sec["panels"]) + list(sec.get("pills") or [])
        assert cards, (
            f"section '{sec['head']}' is pinned to {allowed} but the probe "
            "measured no cards under it — the pin is vacuous, fix the selector"
        )
        checked += len(cards)
        for panel in cards:
            if not any(abs(panel["w"] - a) <= TOLERANCE_PX for a in allowed):
                offenders.append(
                    f"'{sec['head']}' card is {panel['w']}px, allowed {allowed}"
                )
            if panel["sw"] > panel["w"] + TOLERANCE_PX:
                offenders.append(
                    f"'{sec['head']}' card clips its content at {panel['w']}px"
                )
    assert checked >= 10, (
        f"only {checked} desktop cards measured; too few for a verdict"
    )
    assert not offenders, offenders


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
