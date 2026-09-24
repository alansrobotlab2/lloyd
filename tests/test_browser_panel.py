"""Backlog #278 — MC Browser Panel contract.

Pins exactly what the item's acceptance check names, plus the two regression
guards it names:

  * Chromium's headless flag is env/config-driven and defaults to True
    (it used to be the literal ``headless=False``, which made the whole
    feature depend on a live display).
  * A browser-state frame (screenshot + annotated accessibility snapshot)
    is pushed after every browser tool call.
  * A route under ``/api/browser`` serves ``text/event-stream``, mirroring
    the one SSE channel MC already had (``/api/mc/events``).
  * Mission Control has a Browser page wired into the ``Page`` union and
    the Layout render path.

  * Guard: still 14 ``browser_*`` tools, no signature changes.
  * Guard: the SSRF host check still refuses loopback/private targets.

The SSE route is exercised by driving the streaming generator directly
rather than through ``httpx.ASGITransport`` — that transport buffers the
whole response body before returning, so an endless stream would hang the
suite instead of proving anything.
"""
import asyncio
import base64
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import server  # noqa: E402
from app.routers import browser as browser_router  # noqa: E402
from agent_mcp import browser as browser_module  # noqa: E402

WEB = ROOT / "web" / "src"


def _sidebar_page_union() -> str:
    src = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    m = re.search(r"export type Page\s*=\s*([^\n]+)", src)
    assert m, "no `export type Page` union in Sidebar.tsx"
    return m.group(1)


# ── Headless default ──────────────────────────────────────────────────────────

def test_headless_defaults_to_true_without_env_or_config(monkeypatch):
    monkeypatch.delenv("LLOYD_BROWSER_HEADLESS", raising=False)
    monkeypatch.setattr(browser_module, "CONFIG", {}, raising=False)
    assert browser_module._resolve_headless() is True


def test_headless_env_var_overrides_in_both_directions(monkeypatch):
    monkeypatch.setattr(browser_module, "CONFIG", {}, raising=False)
    for raw, want in (("1", True), ("true", True), ("0", False), ("false", False)):
        monkeypatch.setenv("LLOYD_BROWSER_HEADLESS", raw)
        assert browser_module._resolve_headless() is want, raw


def test_headless_reads_config_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("LLOYD_BROWSER_HEADLESS", raising=False)
    monkeypatch.setattr(browser_module, "CONFIG", {"browser": {"headless": False}},
                        raising=False)
    assert browser_module._resolve_headless() is False
    monkeypatch.setattr(browser_module, "CONFIG", {"browser": {"headless": True}},
                        raising=False)
    assert browser_module._resolve_headless() is True


def test_launch_no_longer_carries_a_hardcoded_headless_literal():
    src = (ROOT / "agent_mcp" / "browser.py").read_text(encoding="utf-8")
    m = re.search(r"chromium\.launch\((.*?)\n    \)", src, re.DOTALL)
    assert m, "could not find the chromium.launch( call"
    launch_args = m.group(1)
    assert "headless=_resolve_headless()" in launch_args, \
        "chromium.launch must resolve headless from env/config"
    assert "headless=False" not in launch_args


# ── Certificate verification stays on (backlog #1241) ──────────────────────────
#
# #1241's fix is provisioning the Lloyd CA into ~/.pki/nssdb (scripts/install-ca.sh,
# pinned by tests/test_gen_cert_ca_install.py). What it must NOT do is make the
# symptom go away by switching verification off: `ignore_https_errors=True` on the
# context would silence the failure for the MC frontend AND for every public-web
# navigation, which is exactly why #1089 refused that shape. These two tests are what
# keeps that refusal true after someone fixes the trust problem.

def test_browser_source_never_disables_https_verification():
    src = (ROOT / "agent_mcp" / "browser.py").read_text(encoding="utf-8")
    assert "ignore_https_errors" not in src, \
        "trusting the Lloyd CA is scripts/install-ca.sh's job, not a flag's"


async def test_launch_and_context_args_are_unchanged_so_verification_stays_on(monkeypatch):
    """Across the playwright seam: what `_launch_browser_locked` actually hands
    `chromium.launch` / `new_context`, not what the source says it does. The four
    sandbox/automation flags and the viewport/UA context are the whole call — no
    trust override appears in either, so an untrusted certificate still ends the
    navigation in net::ERR_CERT_AUTHORITY_INVALID (the negative arm of
    tests/test_gen_cert_ca_install.py proves Chromium really does say that here)."""
    captured: dict = {}

    class _FakeContext:
        def __init__(self):
            self.pages = []

        async def route(self, pattern, handler):
            captured["route"] = (pattern, handler)

    class _FakeBrowser:
        def is_connected(self):
            return True

        async def new_context(self, **kwargs):
            captured["context_kwargs"] = kwargs
            return _FakeContext()

    class _FakeChromium:
        async def launch(self, **kwargs):
            captured["launch_kwargs"] = kwargs
            return _FakeBrowser()

    class _FakeDriver:
        chromium = _FakeChromium()

        async def start(self):
            return self

        async def stop(self):
            pass

    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: _FakeDriver())
    # Reset the module's cached instance so the launch path runs for real, and so
    # these fakes are what gets put back afterwards.
    monkeypatch.setattr(browser_module, "_pw", None, raising=False)
    monkeypatch.setattr(browser_module, "_browser", None, raising=False)
    monkeypatch.setattr(browser_module, "_context", None, raising=False)

    context = await browser_module._launch_browser_locked()

    launch = captured["launch_kwargs"]
    assert launch["executable_path"] == "/usr/bin/chromium"
    assert launch["args"] == [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
    ], "chromium.launch args changed; #1241 requires them unchanged"
    assert "ignore_https_errors" not in launch, launch
    assert "ignore_https_errors" not in captured["context_kwargs"], captured["context_kwargs"]
    assert context is not None and captured["route"][0] == "**/*"


# ── State frame ───────────────────────────────────────────────────────────────

class _FakePage:
    url = "https://example.test/page"

    class _Locator:
        async def aria_snapshot(self):
            return '- webpage "Example"\n  - button "Submit" [e1]'

    def locator(self, _selector):
        return _FakePage._Locator()

    async def title(self):
        return "Example Title"

    async def screenshot(self, **kwargs):
        assert kwargs.get("type") == "jpeg", "streamed frames must not be PNG"
        return b"\xff\xd8jpeg-bytes\xff\xd9"


def test_existing_page_never_launches_a_browser(monkeypatch):
    """A cookies call with no page open must not spin up Chromium."""
    launched = []

    async def boom():
        launched.append(1)
        raise AssertionError("_capture_state must not launch a browser")

    monkeypatch.setattr(browser_module, "_get_page", boom, raising=False)
    monkeypatch.setattr(browser_module, "_context", None, raising=False)
    monkeypatch.setattr(browser_module, "_active_page", None, raising=False)

    assert browser_module._existing_page() is None
    assert launched == []


async def test_capture_state_builds_a_screenshot_plus_snapshot_frame(monkeypatch):
    monkeypatch.setattr(browser_module, "_existing_page", lambda: _FakePage(), raising=False)
    monkeypatch.setattr(browser_module, "_ref_map",
                        {"e1": {"role": "button", "name": "Submit", "occurrence": 0}},
                        raising=False)

    frame = await browser_module._capture_state("browser_navigate")

    assert frame["tool"] == "browser_navigate"
    assert frame["url"] == "https://example.test/page"
    assert frame["title"] == "Example Title"
    assert 'button "Submit" [e1]' in frame["snapshot"]
    assert frame["refs"] == [{"ref": "e1", "role": "button", "name": "Submit"}]
    # Screenshot travels base64-encoded so the SSE frame stays text-safe.
    assert base64.b64decode(frame["screenshot_b64"]) == b"\xff\xd8jpeg-bytes\xff\xd9"
    assert frame["mime"] == "image/jpeg"


async def test_capture_state_returns_none_without_a_page(monkeypatch):
    monkeypatch.setattr(browser_module, "_existing_page", lambda: None, raising=False)
    assert await browser_module._capture_state("browser_navigate") is None


async def test_state_push_fires_after_every_tool_call(monkeypatch):
    """The dispatcher, not just the helper: a tool call must emit a push.

    The push is fire-and-forget (the tool result is already computed and the
    agent's turn should not pay for the frontend's round-trip), so the loop
    is yielded a few times before asserting rather than checking immediately.
    """
    pushed = []

    def fake_schedule(tool_name):          # sync, like the real one
        pushed.append(tool_name)

    async def fake_snapshot(_full):
        return json.dumps({"ok": True})

    monkeypatch.setattr(browser_module, "_schedule_state_push", fake_schedule, raising=False)
    monkeypatch.setattr(browser_module, "_browser_snapshot", fake_snapshot, raising=False)

    await browser_module.call_tool("browser_snapshot", {})

    assert pushed == ["browser_snapshot"]


async def test_schedule_state_push_survives_garbage_collection(monkeypatch):
    """create_task alone is not enough — asyncio weak-references tasks."""
    started = []

    async def fake_push(tool_name):
        started.append(tool_name)

    monkeypatch.setattr(browser_module, "_push_browser_state", fake_push, raising=False)
    monkeypatch.setattr(browser_module, "_pending_pushes", set(), raising=False)

    browser_module._schedule_state_push("browser_navigate")

    assert browser_module._pending_pushes, "the task handle must be retained"
    await asyncio.gather(*browser_module._pending_pushes)
    assert started == ["browser_navigate"]


async def test_push_sends_the_frame_to_the_backend_state_route(monkeypatch):
    sent = {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            sent["url"] = url
            sent["json"] = json

            class _R:
                status_code = 200

            return _R()

    async def fake_capture(tool_name):
        return {"tool": tool_name, "url": "https://example.test"}

    import httpx

    monkeypatch.setattr(browser_module, "_capture_state", fake_capture, raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(browser_module, "LLOYD_API", "http://127.0.0.1:8080", raising=False)

    await browser_module._push_browser_state("browser_navigate")

    assert sent["url"].endswith("/api/browser/state")
    assert sent["json"]["url"] == "https://example.test"


async def test_push_failures_never_break_the_tool_call(monkeypatch):
    """Mission Control being down is not a reason for browser_navigate to fail."""
    async def fake_capture(tool_name):
        raise RuntimeError("no page")

    monkeypatch.setattr(browser_module, "_capture_state", fake_capture, raising=False)
    await browser_module._push_browser_state("browser_navigate")  # must not raise

    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise OSError("backend down")

        async def __aexit__(self, *a):
            return False

    async def fake_capture_ok(tool_name):
        return {"tool": tool_name}

    import httpx

    monkeypatch.setattr(browser_module, "_capture_state", fake_capture_ok, raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Boom)
    monkeypatch.setattr(browser_module, "LLOYD_API", "http://127.0.0.1:8080", raising=False)
    await browser_module._push_browser_state("browser_navigate")  # must not raise


# ── Backend route ─────────────────────────────────────────────────────────────

class _FakeRequest:
    async def is_disconnected(self):
        return False


@pytest.fixture
def _clean_state(monkeypatch):
    monkeypatch.setattr(browser_router, "_latest", None, raising=False)


async def test_post_state_route_accepts_a_frame(_clean_state):
    import httpx

    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 9999))
    async with httpx.AsyncClient(transport=transport, base_url="http://lloyd-test") as client:
        r = await client.post("/api/browser/state", json={"url": "https://a.test", "tool": "browser_navigate"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        r2 = await client.post("/api/browser/state", json={"url": "https://b.test"})
        assert r2.status_code == 200


async def test_get_state_route_streams_pushed_frames_to_subscribers(_clean_state):
    resp = await browser_router.get_browser_state(_FakeRequest())
    assert isinstance(resp, browser_router.StreamingResponse)
    assert resp.media_type == "text/event-stream"

    gen = resp.body_iterator
    hello = await asyncio.wait_for(gen.__anext__(), 5)
    await browser_router.post_browser_state(
        {"url": "https://a.test", "tool": "browser_navigate", "snapshot": "- webpage"}
    )
    state = await asyncio.wait_for(gen.__anext__(), 5)
    await gen.aclose()

    body = "".join(x.decode() if isinstance(x, bytes) else x for x in (hello, state))
    assert "event: hello" in body
    assert "event: state" in body
    assert "https://a.test" in body


async def test_pushed_frame_fans_out_to_a_subscribed_sse_client(_clean_state):
    q = browser_router.subscribe()
    try:
        await browser_router.post_browser_state({"url": "https://fan.test"})
        evt = await asyncio.wait_for(q.get(), 5)
        assert evt["url"] == "https://fan.test"
    finally:
        browser_router.unsubscribe(q)


# ── Frontend wiring ───────────────────────────────────────────────────────────

def test_browser_page_component_exists():
    hits = list(WEB.rglob("*[Bb]rowser*"))
    assert hits, "no browser page component under web/src"
    assert any(h.name == "BrowserPage.tsx" for h in hits)


def test_browser_is_a_member_of_the_page_union():
    assert "'browser'" in _sidebar_page_union()


def test_browser_page_is_imported_and_rendered_by_layout():
    src = (WEB / "components" / "Layout.tsx").read_text(encoding="utf-8")
    assert re.search(r"import\s+BrowserPage|BrowserPage\s*=", src), \
        "Layout.tsx must import the Browser page"
    assert re.search(r"^\s*browser:\s*", src, re.MULTILINE), \
        "Layout.tsx must map the 'browser' page to its component"


def test_browser_page_subscribes_to_the_state_stream():
    page = next(WEB.rglob("BrowserPage.tsx"))
    src = page.read_text(encoding="utf-8")
    assert "/api/browser/state" in src
    assert "EventSource" in src
    assert "<img" in src, "the frame must render as an image"


# ── Regression guards named by the item ───────────────────────────────────────

# Thirteen since 2026-09-23: `browser_type` became `browser_fill(keystrokes=true)`.

def test_still_thirteen_browser_tools_with_unchanged_names():
    src = (ROOT / "agent_mcp" / "browser.py").read_text(encoding="utf-8")
    assert src.count('Tool(name="browser_') == 13


async def test_tool_schemas_still_expose_the_same_thirteen_tools():
    tools = await browser_module.list_tools()
    assert len(tools) == 13
    assert {t.name for t in tools} == {
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_scroll", "browser_press", "browser_tabs", "browser_screenshot",
        "browser_evaluate", "browser_fill", "browser_wait", "browser_select",
        "browser_drag", "browser_cookies",
    }


# ── The SSRF guard ────────────────────────────────────────────────────────────
#
# The test that used to live here asserted `_is_private_host` returned the
# right booleans. It did, for five months, while being called from nowhere.
# Backlog #278's acceptance asked that "the SSRF host check ... is intact" and
# this test answered that question about a symbol rather than about a
# behaviour. So the predicate test stays, but it is no longer the point: what
# follows pins that every entry point INVOKES the guard, which is the property
# that was actually missing.


def test_the_predicate_still_classifies_hosts(monkeypatch):
    p = browser_module._is_private_host
    assert p("localhost") is True
    assert p("127.0.0.1") is True
    assert p("10.1.2.3") is True
    assert p("192.168.0.7") is True
    assert p("169.254.1.1") is True
    assert p("::1") is True
    assert p("") is True
    # Public: resolution stubbed so the suite never depends on DNS.
    browser_module._resolve_addrs.cache_clear()
    monkeypatch.setattr(browser_module, "_resolve_addrs", lambda h: ("93.184.216.34",))
    assert browser_module._is_private_host("example.com") is False


def test_the_guard_is_actually_wired_into_every_url_entry_point():
    """The regression that mattered: defined, tested, and never called.

    A source-level check because it is the only one that fails for the right
    reason. A behavioural test can be satisfied by a second, private check
    added beside the guard, which is how the two copies of this function came
    to exist in the first place.
    """
    src = (ROOT / "agent_mcp" / "browser.py").read_text(encoding="utf-8")
    body = src.split("async def _browser_navigate", 1)[1]
    assert "_host_block_reason" in body.split("async def ", 1)[0], \
        "browser_navigate must consult the guard"
    tabs = src.split("async def _browser_tabs", 1)[1].split("\nasync def ", 1)[0]
    assert "_host_block_reason" in tabs, "browser_tabs(new, url=...) must consult the guard"
    ui = src.split("async def navigate_from_ui", 1)[1]
    # The URL bar inherits it through _browser_navigate rather than repeating
    # the check; assert the delegation exists so it cannot quietly stop.
    assert "_browser_navigate" in ui.split("\nasync def ", 1)[0]
    assert 'await _context.route("**/*", _guard_route)' in src, \
        "the context interceptor covers clicks and subresources"
    # And the layer that covers what interception cannot: see the redirect
    # test below for why these two are not the same guard.
    for fn in ("_browser_snapshot", "_browser_evaluate"):
        body = src.split(f"async def {fn}", 1)[1].split("\nasync def ", 1)[0]
        assert "_enforce_landing" in body, f"{fn} must check where the page landed"
    cap = src.split("async def _capture_state", 1)[1].split("\nasync def ", 1)[0]
    assert "_host_block_reason" in cap, \
        "the MC frame must not carry a screenshot of a private host"
    # DNS is a blocking syscall and these run on the aggregator's loop, the
    # one that dispatches every MCP tool call.
    assert "await asyncio.to_thread(_resolve_addrs" in src, \
        "host resolution must not block the event loop"
    for fn in ("_guard_route", "_enforce_landing", "_capture_state"):
        body = src.split(f"async def {fn}", 1)[1].split("\nasync def ", 1)[0]
        assert "_host_block_reason_async" in body, f"{fn} must use the async check"


async def test_navigate_refuses_a_lan_address_without_opening_a_browser(monkeypatch):
    """Refused before `_get_page`, so a blocked URL never launches Chromium."""
    async def boom():
        raise AssertionError("must not reach the browser")
    monkeypatch.setattr(browser_module, "_get_page", boom)

    for host in ("192.168.1.1", "10.0.0.5", "172.16.4.4", "169.254.169.254"):
        out = json.loads(await browser_module._browser_navigate(f"http://{host}/"))
        assert "error" in out and "private/internal" in out["error"], (host, out)


async def test_navigate_allows_the_machine_itself(monkeypatch):
    """Loopback stays reachable, by IP and by name.

    Mirrors `http_tools.http_request`. The agent browses Lloyd's own dashboard,
    and it already holds Bash and the MCP surface — refusing the browser here
    removes no authority an injected prompt could not reach more directly.
    """
    reached = []

    class FakePage:
        url = "http://127.0.0.1:8080/x"
        async def goto(self, url, **kw):
            reached.append(url)
            return type("R", (), {"status": 200})()
        async def title(self):
            return "ok"

    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(FakePage()))
    for url in ("http://127.0.0.1:8080/x", "http://localhost:8080/x", "http://[::1]:8080/x"):
        out = json.loads(await browser_module._browser_navigate(url))
        assert out.get("ok") is True, (url, out)
    assert len(reached) == 3


async def test_an_encoded_lan_address_is_blocked_and_encoded_loopback_is_not():
    """What the old regex list could not do.

    `3232235777` is 192.168.1.1 and `2130706433` is 127.0.0.1; a prefix match
    on the hostname string classifies neither. Checking resolved addresses
    gets both right, because the platform resolver normalises exactly the way
    Chromium does.
    """
    assert browser_module._is_private_host("3232235777") is True
    assert browser_module._is_loopback_host("3232235777") is False
    assert browser_module._is_loopback_host("2130706433") is True
    assert browser_module._is_loopback_host("0x7f000001") is True
    assert browser_module._host_block_reason("http://3232235777/") is not None
    assert browser_module._host_block_reason("http://2130706433:8080/") is None


async def test_tabs_refuses_a_lan_url_before_opening_the_tab(monkeypatch):
    """A check placed after `new_page()` strands an empty tab against MAX_TABS."""
    async def boom():
        raise AssertionError("must not open a context")
    monkeypatch.setattr(browser_module, "_ensure_browser", boom)
    out = json.loads(await browser_module._browser_tabs("new", url="http://192.168.1.1/"))
    assert "private/internal" in out.get("error", "")


def test_a_blocked_redirect_reports_the_reason_not_a_timeout():
    """Aborting a redirect leaves `goto` waiting for a load that never comes.

    The raw failure is a bare 30s timeout, which sends the agent looking for a
    slow site rather than telling it the hop was refused.
    """
    before = browser_module._block_log["seq"]
    assert browser_module._block_since(before) is None
    browser_module._record_block("http://192.168.1.1/admin", 'Blocked — private/internal host "192.168.1.1"')
    msg = browser_module._block_since(before)
    assert msg and "192.168.1.1" in msg and "redirected to" in msg


class _LandingPage:
    """Minimal page: a url that can change, and a goto that records.

    Named apart from the module's other `_FakePage`, which is a fixed-content
    stub for the frame test and has no settable url.
    """
    def __init__(self, url):
        self.url = url
        self.goto_calls = []
    async def goto(self, url, **kw):
        self.goto_calls.append(url)
        self.url = url
        return type("R", (), {"status": 200})()
    async def title(self):
        return "t"


async def test_a_redirect_to_a_private_host_is_caught_after_the_fact():
    """Route interception does not see redirects, and that is measured.

    `route.continue_()` hands the request to Chromium, which follows a 3xx
    internally without re-entering interception. Against a loopback server
    that 302s to this box's own LAN address, the handler is called once, for
    the first hop, while the redirected request arrives only as an event. So
    the landing check is not belt-and-braces here, it is the only thing
    standing in that lane.
    """
    page = _LandingPage("http://192.168.50.108:8080/admin")
    reason = await browser_module._enforce_landing(page)
    assert reason and "private/internal" in reason
    # Blanked, or the next snapshot reads the page we just refused and the
    # state mirror pushes a screenshot of it to Mission Control.
    assert page.goto_calls == ["about:blank"]
    assert page.url == "about:blank"


async def test_the_landing_check_leaves_an_allowed_page_alone():
    page = _LandingPage("https://example.com/")
    assert await browser_module._enforce_landing(page) is None
    assert page.goto_calls == []


async def test_snapshot_refuses_to_read_a_private_page(monkeypatch):
    """The chokepoint: however the browser got there, content stops here."""
    # A fresh page per call: the first check blanks the one it refuses, which
    # is the point, so reusing it would test the blank page instead.
    page = _LandingPage("http://10.1.2.3/secrets")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    out = json.loads(await browser_module._browser_snapshot())
    assert "private/internal" in out.get("error", "")

    page2 = _LandingPage("http://10.1.2.3/secrets")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page2))
    out = json.loads(await browser_module._browser_evaluate("1+1"))
    assert "private/internal" in out.get("error", "")


async def test_the_mc_frame_carries_nothing_from_a_private_host(monkeypatch):
    """A frame is a screenshot plus an a11y tree, and a human reads the tab."""
    page = _LandingPage("http://192.168.0.9/cam")
    monkeypatch.setattr(browser_module, "_existing_page", lambda: page)
    assert await browser_module._capture_state("browser_navigate") is None


def test_a_blank_page_is_not_a_private_host():
    """The bug this guard nearly shipped with.

    `about:blank` has no hostname, and an empty host counts as private — so
    the blank page `_enforce_landing` navigates to was itself refused. One
    refusal then poisoned every later snapshot with `private/internal host ""`,
    and a freshly launched browser could not be snapshotted before its first
    navigation. Only http(s) names a host on the network.
    """
    assert browser_module._host_block_reason("about:blank") is None
    assert browser_module._host_block_reason("data:text/html,hi") is None
    # Still refused when the scheme really does reach out with no host.
    assert browser_module._host_block_reason("http:///x") is not None


async def test_the_landing_check_is_idempotent_on_the_page_it_blanks():
    """Blank once, not forever: the second call must be a no-op."""
    page = _LandingPage("http://192.168.1.9/x")
    assert await browser_module._enforce_landing(page) is not None
    assert page.url == "about:blank"
    assert await browser_module._enforce_landing(page) is None
    assert page.goto_calls == ["about:blank"]


def test_the_guard_has_a_kill_switch(monkeypatch):
    monkeypatch.setenv("LLOYD_BROWSER_BLOCK_PRIVATE", "0")
    assert browser_module._resolve_block_private() is False
    assert browser_module._host_block_reason("http://192.168.1.1/") is None
    monkeypatch.setenv("LLOYD_BROWSER_BLOCK_PRIVATE", "1")
    assert browser_module._resolve_block_private() is True
    assert browser_module._host_block_reason("http://192.168.1.1/") is not None


def _async(value):
    async def _c():
        return value
    return _c()


# ── The URL bar ───────────────────────────────────────────────────────────────
#
# The panel shipped read-only. A URL bar is the one control it has, and it
# deliberately does not go through MCP: the user typing a URL is not the agent
# calling a tool, and dispatching it as one would write a `browser_navigate`
# into the transcript that the model never made. So it is a plain route on the
# aggregator, proxied by the backend, and `navigate_from_ui` must stay off the
# tool list — which `test_still_fourteen_browser_tools_with_unchanged_names`
# above already enforces.


async def test_url_bar_completes_a_scheme_less_host(monkeypatch):
    """A human types `news.ycombinator.com`; the tool would reject it.

    `browser_navigate` stays strict on purpose — an agent omitting the scheme
    has made a mistake worth surfacing. A person has saved eight keystrokes.
    """
    seen: list[str] = []

    async def fake_navigate(url, wait_until="domcontentloaded"):
        seen.append(url)
        return json.dumps({"ok": True, "url": url, "title": "t", "status": 200})

    async def fake_push(tool_name):
        seen.append(f"push:{tool_name}")

    monkeypatch.setattr(browser_module, "_browser_navigate", fake_navigate)
    monkeypatch.setattr(browser_module, "_push_browser_state", fake_push)

    for typed, want in (
        ("news.ycombinator.com", "https://news.ycombinator.com"),
        ("example.com/a/b?q=1", "https://example.com/a/b?q=1"),
        ("//example.com", "https://example.com"),
        ("  example.com  ", "https://example.com"),
        # A port is not a scheme. Detecting one by the bare colon reads the
        # whole of "localhost:8080" as a scheme and leaves it uncompleted,
        # which `_browser_navigate` then rejects as not-http — and a box
        # serving this much on loopback hits that case first.
        ("localhost:8080", "https://localhost:8080"),
        ("example.com:8080/x?q=1", "https://example.com:8080/x?q=1"),
        ("127.0.0.1:8500/health", "https://127.0.0.1:8500/health"),
        # An explicit scheme is never rewritten, http included.
        ("http://example.com", "http://example.com"),
        ("https://example.com", "https://example.com"),
    ):
        seen.clear()
        result = await browser_module.navigate_from_ui(typed)
        assert seen[0] == want, f"{typed!r} -> {seen[0]!r}, wanted {want!r}"
        assert result.get("ok") is True


async def test_url_bar_falls_back_to_http_for_a_scheme_it_added(monkeypatch):
    """Local services split both ways, so guessing once is not enough.

    On this box the frontend serves TLS on 5173 while the backend, the
    aggregator and both engines are plain HTTP. `127.0.0.1:8080` completed to
    https dies on ERR_SSL_PROTOCOL_ERROR, and a URL bar that cannot open the
    service next to it is not much of a URL bar.
    """
    tried = []

    async def fake_navigate(url, wait_until="domcontentloaded"):
        tried.append(url)
        if url.startswith("https://"):
            return json.dumps({"error": "net::ERR_SSL_PROTOCOL_ERROR at " + url})
        return json.dumps({"ok": True, "url": url, "title": "t", "status": 200})

    monkeypatch.setattr(browser_module, "_browser_navigate", fake_navigate)
    monkeypatch.setattr(browser_module, "_push_browser_state", lambda *_: _async(None))

    out = await browser_module.navigate_from_ui("127.0.0.1:8080/api/mc/state")
    assert out.get("ok") is True
    assert tried == ["https://127.0.0.1:8080/api/mc/state",
                     "http://127.0.0.1:8080/api/mc/state"]


async def test_url_bar_retries_once_when_the_fallback_loses_a_navigation_race():
    """The failed https attempt settles into Chromium's error page.

    An immediate retry collides with it: "Navigation to http://... is
    interrupted by another navigation to chrome-error://chromewebdata/". That
    interruption says nothing about whether the scheme was right, so giving up
    there reports an SSL error for a URL that works.
    """
    calls = []

    async def fake_navigate(url, wait_until="domcontentloaded"):
        calls.append(url)
        if url.startswith("https://"):
            return json.dumps({"error": "net::ERR_SSL_PROTOCOL_ERROR at " + url})
        if len([c for c in calls if c.startswith("http://")]) == 1:
            return json.dumps({"error": 'Navigation to "%s" is interrupted by '
                                        'another navigation to "chrome-error://chromewebdata/"' % url})
        return json.dumps({"ok": True, "url": url, "title": "t", "status": 200})

    import agent_mcp.browser as bm
    orig_nav, orig_push, orig_page = bm._browser_navigate, bm._push_browser_state, bm._existing_page
    bm._browser_navigate = fake_navigate
    bm._push_browser_state = lambda *_: _async(None)
    bm._existing_page = lambda: None
    try:
        out = await bm.navigate_from_ui("127.0.0.1:8080/x")
        assert out.get("ok") is True, out
        assert calls == ["https://127.0.0.1:8080/x",
                         "http://127.0.0.1:8080/x",
                         "http://127.0.0.1:8080/x"]
    finally:
        bm._browser_navigate, bm._push_browser_state, bm._existing_page = orig_nav, orig_push, orig_page


async def test_url_bar_never_downgrades_a_scheme_the_user_typed(monkeypatch):
    """The fallback is scoped to a completion we made.

    Silently retrying a user's explicit `https://` over plaintext is the one
    behaviour a URL bar must not have.
    """
    tried = []

    async def fake_navigate(url, wait_until="domcontentloaded"):
        tried.append(url)
        return json.dumps({"error": "net::ERR_SSL_PROTOCOL_ERROR at " + url})

    monkeypatch.setattr(browser_module, "_browser_navigate", fake_navigate)
    monkeypatch.setattr(browser_module, "_push_browser_state", lambda *_: _async(None))

    out = await browser_module.navigate_from_ui("https://example.com")
    assert "error" in out
    assert tried == ["https://example.com"], "must not retry over http"


async def test_url_bar_does_not_retry_an_ordinary_404_or_dns_failure(monkeypatch):
    """A page that is simply not there is an answer, not a scheme problem."""
    tried = []

    async def fake_navigate(url, wait_until="domcontentloaded"):
        tried.append(url)
        return json.dumps({"error": "net::ERR_NAME_NOT_RESOLVED"})

    monkeypatch.setattr(browser_module, "_browser_navigate", fake_navigate)
    monkeypatch.setattr(browser_module, "_push_browser_state", lambda *_: _async(None))

    await browser_module.navigate_from_ui("nope.invalid")
    assert len(tried) == 1


async def test_url_bar_pushes_a_frame_even_when_the_page_fails(monkeypatch):
    """A 404 or a timeout still changes the viewport.

    Returning without a push leaves the tab showing the page they navigated
    away from, which is the most confusing possible answer to a click.
    """
    pushed: list[str] = []

    async def fake_navigate(url, wait_until="domcontentloaded"):
        return json.dumps({"error": "net::ERR_NAME_NOT_RESOLVED"})

    async def fake_push(tool_name):
        pushed.append(tool_name)

    monkeypatch.setattr(browser_module, "_browser_navigate", fake_navigate)
    monkeypatch.setattr(browser_module, "_push_browser_state", fake_push)

    result = await browser_module.navigate_from_ui("nope.invalid")
    assert "error" in result
    # Tagged so the tab can say a human drove this frame, not the agent.
    assert pushed == ["url_bar"]


async def test_url_bar_rejects_an_empty_url(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("must not reach the browser")

    monkeypatch.setattr(browser_module, "_browser_navigate", boom)
    monkeypatch.setattr(browser_module, "_push_browser_state", boom)
    assert "error" in await browser_module.navigate_from_ui("   ")


def test_aggregator_serves_the_navigate_route():
    """Playwright lives in the lloyd-mcp process, so the control action has
    to cross that seam — the same one the dashboard crosses for /state."""
    from agent_mcp import main as mcp_main

    paths = {r.path for r in mcp_main.starlette_app.routes if hasattr(r, "path")}
    assert "/browser/navigate" in paths


def test_backend_proxies_the_url_bar_to_the_aggregator():
    src = (ROOT / "app" / "routers" / "browser.py").read_text(encoding="utf-8")
    assert "/api/browser/navigate" in src
    # Resolved, not grepped for a port: since #1053 the URL comes from
    # `app.aggregator_config.route("browser_navigate")`, which derives it from
    # `services.lloyd_mcp` — the same module that hands the proxy the request
    # credential, so the endpoint and the header cannot drift apart. What has to
    # stay true is the destination: the aggregator process that owns Playwright.
    assert "aggregator_route(" in src, "the proxy stopped deriving its target"
    from app.aggregator_config import route
    assert route("browser_navigate") == browser_router._MCP_NAVIGATE_URL
    assert route("browser_navigate").startswith("http://127.0.0.1:")
    assert route("browser_navigate").endswith("/browser/navigate")


def test_navigate_summary_never_carries_the_screenshot():
    """`mc_navigate(tab="browser")` puts this in the model's context.

    The frame is ~20 KB of base64 plus up to 8 KB of accessibility tree; the
    summary is meant to be a line saying what is on screen.
    """
    browser_router._latest = {
        "url": "https://example.com",
        "title": "Example",
        "ts": 1.0,
        "tool": "url_bar",
        "mime": "image/jpeg",
        "screenshot_b64": "A" * 5000,
        "snapshot": "B" * 5000,
        "refs": [{"ref": "e1", "role": "link", "name": "x"}],
    }
    try:
        summary = browser_router.latest_frame_summary()
        assert summary["url"] == "https://example.com"
        assert summary["driven_by"] == "url_bar"
        assert "screenshot_b64" not in summary
        assert "snapshot" not in summary
        assert "refs" not in summary
        assert len(json.dumps(summary)) < 500
    finally:
        browser_router._latest = None

    assert browser_router.latest_frame_summary() == {"active": False}


def test_url_bar_does_not_bind_the_input_to_the_live_frame():
    """A frame lands after every browser_* tool call.

    Binding the input straight to `frame.url` would erase whatever the user
    is halfway through typing the moment the agent navigates, so the re-seed
    is gated on the field being clean.
    """
    page = next(WEB.rglob("BrowserPage.tsx"))
    src = page.read_text(encoding="utf-8")
    assert "browserNavigate" in src, "the URL bar must call the navigate API"
    assert "onSubmit" in src, "Enter must submit"
    assert re.search(r"if\s*\(!urlDirty\)\s*setUrlInput", src), \
        "the frame must only re-seed the input while it is clean"
    assert re.search(r"value=\{urlInput\}", src), \
        "the input must be driven by its own state, not frame.url"


# ── Backlog #1087: an error document is not a successful fetch ─────────────────
#
# `browser_navigate` returned `{"ok": true, ..., "status": 404}` for a document
# that was not there, and `browser_snapshot` then handed back that 404 page's
# accessibility tree — banner, nav links, "Skip to content" — as the page's
# content. Both are stored tool results in
# `sessions/20260910_020748_deeprese_bba8.json`, twice, in a session researching
# moved Nav2 documentation. The four clauses: an error document reads as not-ok
# with the code and title named in a non-error field; a 2xx/3xx or a goto that
# hands back no response handle still reads as ok; a snapshot of an error
# document says which status it is; and the human's URL bar still shows the page
# rather than a red banner.

_ERR_TITLE = "Page moved · Nav2 documentation"


class _ErrorDocPage:
    """A page whose `goto` hands back the document status the test asks for.

    Faked at the same boundary the rest of this file fakes Playwright — the
    response handle. `status=None` is the case the module guards today with
    `resp.status if resp else 0`: Playwright returns no handle at all.
    """

    def __init__(self, status=404, title=_ERR_TITLE,
                 url="http://127.0.0.1:45547/gone"):
        self.url = url
        self._title = title
        self._status = status
        self.goto_calls: list[str] = []

    async def goto(self, url, **kw):
        self.goto_calls.append(url)
        self.url = url
        if self._status is None:
            return None
        return type("R", (), {"status": self._status})()

    async def title(self):
        return self._title

    def locator(self, selector):
        return self

    async def aria_snapshot(self):
        if self._status is not None and self._status >= 400:
            return '- heading "404 Not Found" [level=1]\n- banner: Site chrome'
        return '- paragraph: the real document'


class _FakePost:
    """The aggregator route reads only `await request.json()`."""

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture(autouse=True)
def _no_inherited_doc_status():
    """Which document the last navigation landed on is module state.

    No test in this file may read a status another one recorded.
    """
    browser_module._forget_doc_status()
    yield
    browser_module._forget_doc_status()


async def test_an_error_document_is_not_reported_as_a_successful_fetch(monkeypatch):
    """Clause 1: ok is False, and a non-error field names code and title."""
    page = _ErrorDocPage(status=404)
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))

    out = json.loads(await browser_module._browser_navigate(page.url))

    assert out["ok"] is False, out
    assert "error" not in out, \
        "an `error` key turns BrowserPage.tsx:82 into a red banner over a page " \
        "the human asked to see — the verdict belongs in `warning`"
    assert out["status"] == 404, out
    assert out["title"] == _ERR_TITLE, out
    assert out["url"] == page.url, out
    warning = out["warning"]
    assert warning.startswith("HTTP 404 — "), warning
    assert _ERR_TITLE in warning, warning


async def test_a_normal_status_or_a_missing_response_handle_still_reads_as_ok(monkeypatch):
    """Clause 2: nothing about a good navigation changes.

    The `status=None` case is the trap in the current line
    (`resp.status if resp else 0`): a branch written as a bare
    `status >= 400` would report every handle-less navigation as a failure.
    """
    for status in (200, 204, 301, 302, 399):
        page = _ErrorDocPage(status=status, title="Fine",
                             url=f"http://127.0.0.1:45547/ok/{status}")
        monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
        out = json.loads(await browser_module._browser_navigate(page.url))
        assert out.get("ok") is True, (status, out)
        assert out["status"] == status, (status, out)
        assert out["title"] == "Fine" and out["url"] == page.url, (status, out)
        assert "warning" not in out, (status, out)

    page = _ErrorDocPage(status=None, title="Fine", url="http://127.0.0.1:45547/same-doc")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    out = json.loads(await browser_module._browser_navigate(page.url))
    assert out.get("ok") is True, out
    assert out["status"] == 0, out
    assert "warning" not in out, out


async def test_a_snapshot_of_an_error_page_names_the_status(monkeypatch):
    """Clause 3: the tree says it is an error page, and is still delivered."""
    page = _ErrorDocPage(status=404)
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    await browser_module._browser_navigate(page.url)

    snap = json.loads(await browser_module._browser_snapshot())
    assert "error" not in snap, snap
    assert "[HTTP 404" in snap["snapshot"], snap["snapshot"][:200]
    # The label annotates the tree rather than replacing it: which error page
    # this is stays readable, and a caller that already knows can still parse it.
    assert 'heading "404 Not Found"' in snap["snapshot"], snap["snapshot"][:200]


async def test_a_snapshot_of_a_normal_page_carries_no_error_label(monkeypatch):
    """Clause 3, other half: the label is not a permanent prefix."""
    page = _ErrorDocPage(status=200, title="Real page",
                         url="http://127.0.0.1:45547/here")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    await browser_module._browser_navigate(page.url)

    snap = json.loads(await browser_module._browser_snapshot())
    assert "[HTTP " not in snap["snapshot"], snap["snapshot"][:200]
    assert snap["snapshot"].startswith(
        "[Page] Real page — http://127.0.0.1:45547/here"), snap["snapshot"][:200]


async def test_the_error_label_does_not_outlive_the_document_it_describes(monkeypatch):
    """The label is a claim about one document, so it is keyed to that document.

    A click that follows a link replaces the document without going through
    `_browser_navigate`, and a tab switch moves the focus to a page that was
    never navigated to this way. Either one has to lose the label, or a healthy
    page starts reading as an error page — the false half of the same defect.
    """
    page = _ErrorDocPage(status=404)
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    await browser_module._browser_navigate(page.url)
    assert "[HTTP 404" in json.loads(
        await browser_module._browser_snapshot())["snapshot"]

    page.url = "http://127.0.0.1:45547/moved-to"   # a link was followed
    snap = json.loads(await browser_module._browser_snapshot())
    assert "[HTTP " not in snap["snapshot"], snap["snapshot"][:200]

    other = _ErrorDocPage(status=200, title="Real page",
                          url="http://127.0.0.1:45547/other")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(other))
    await browser_module._browser_navigate("http://127.0.0.1:45547/other")
    page.url = "http://127.0.0.1:45547/gone"       # switched back to tab one
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    assert "[HTTP " not in json.loads(
        await browser_module._browser_snapshot())["snapshot"]


async def test_the_url_bar_shows_an_error_page_instead_of_a_red_banner(monkeypatch):
    """Clause 4: the document facts reach the viewer, and no `error` key does.

    `BrowserPage.tsx:82` is `if (res.error) setNavError(res.error)`, so a 404
    reported through `error` would blank a page the human chose to look at.
    The scheme fallback keys on `error` too, so this also pins that a 404 does
    not spend a second goto over http.
    """
    page = _ErrorDocPage(status=404)
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    monkeypatch.setattr(browser_module, "_push_browser_state", lambda *_: _async(None))

    res = await browser_module.navigate_from_ui("127.0.0.1:45547/gone")

    assert "error" not in res, res
    assert res["status"] == 404, res
    assert res["title"] == _ERR_TITLE, res
    assert res["url"] == "https://127.0.0.1:45547/gone", res
    assert page.goto_calls == ["https://127.0.0.1:45547/gone"], page.goto_calls


async def test_the_url_bar_route_hands_the_error_document_through_unchanged(monkeypatch):
    """The same drive, across the aggregator's route.

    `POST /browser/navigate` is the seam the Browser tab crosses into the
    process that owns Playwright (`agent_mcp/main.py:757`), and its docstring
    promises that a failed navigation is a 200 carrying the page's own facts.
    This runs the real `navigate_from_ui` behind that route, so a route that
    started minting an `error` for `ok: false` fails here rather than in the UI.
    """
    from agent_mcp import main as mcp_main

    page = _ErrorDocPage(status=404)
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    monkeypatch.setattr(browser_module, "_push_browser_state", lambda *_: _async(None))

    resp = await mcp_main.browser_navigate(_FakePost({"url": "127.0.0.1:45547/gone"}))
    body = json.loads(resp.body)

    assert resp.status_code == 200, resp.status_code
    assert "error" not in body, body
    assert body["status"] == 404, body
    assert body["title"] == _ERR_TITLE, body
    assert body["warning"].startswith("HTTP 404 — "), body


async def test_the_backend_proxy_forwards_the_error_document_without_minting_one(monkeypatch):
    """One hop further out: the backend's proxy into the aggregator process.

    `app/routers/browser.py:140` mints an `error` when the aggregator answers
    4xx, so a proxy that learned to read `ok: false` as a broken request would
    put the red banner back for a page that only the *site* failed to find. The
    aggregator answers 200 with the document's own facts, and this pins that
    they arrive at the browser with nothing added.
    """
    sent: list[str] = []
    payload = {"ok": False, "url": "http://127.0.0.1:45547/gone",
               "title": _ERR_TITLE, "status": 404,
               "warning": "HTTP 404 — " + _ERR_TITLE}

    class _FakeResponse:
        status_code = 200
        text = ""
        def json(self):
            return payload

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, json=None, headers=None):
            sent.append(url)
            return _FakeResponse()

    monkeypatch.setattr(browser_router.httpx, "AsyncClient", _FakeClient)
    resp = await browser_router.post_browser_navigate(
        {"url": "http://127.0.0.1:45547/gone"})

    body = json.loads(resp.body)
    assert sent == [browser_router._MCP_NAVIGATE_URL]
    assert "error" not in body, body
    assert body["status"] == 404 and body["title"] == _ERR_TITLE, body
    assert body["warning"].startswith("HTTP 404 — "), body


async def test_the_error_verdict_crosses_the_mcp_seam(monkeypatch):
    """What the model actually reads: `call_tool`'s wrapped payload.

    `text_result` sniffs a top-level `error` key into `is_error`, which the MCP
    wire spells `isError`. So the choice of `warning` over `error` is also the
    choice that keeps an ordinary 404 out of the harness's tool-failure list —
    the transport worked and a document was fetched. The verdict has to be
    readable in the JSON instead, which is what `ok: false` and the warning are
    for.
    """
    page = _ErrorDocPage(status=503, title="503 Bad Gateway",
                         url="http://127.0.0.1:45547/down")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    monkeypatch.setattr(browser_module, "_schedule_state_push", lambda name: None)

    res = await browser_module.call_tool(
        "browser_navigate", {"url": "http://127.0.0.1:45547/down"})

    payload = json.loads(res.content[0].text)
    assert payload["ok"] is False, payload
    assert payload["status"] == 503, payload
    assert "503 Bad Gateway" in payload["warning"], payload
    assert res.is_error is False, \
        "a fetched error document is not a failed tool call"
    # And on the wire, which is the form the harness's error bookkeeping reads.
    assert res.model_dump(by_alias=True, exclude_none=True)["isError"] is False


# ── Backlog #1088: `networkidle` on a page that holds a live socket ───────────
#
# Playwright's `networkidle` waits for ~500 ms with no network connections at all.
# A page holding a websocket, an EventSource, Vite's HMR socket or a long-poll
# never grants that silence, so the wait dies at the tool's own 30 s deadline on a
# document that is fully rendered — and the tool handed the raw
# `Page.goto: Timeout 30000ms exceeded.` back as its error, which reads exactly
# like a dead site. Re-measured 2026-09-24 at base `b72a2ba` (also seen at
# `55edf4f1` on 2026-09-23) by driving the real `_browser_navigate`/`_browser_wait`
# against a fixture that holds an EventSource open (`http://127.0.0.1:5199/`). In
# that run, in that call order: `wait_until="networkidle"` returned
# `{"error": "Page.goto: Timeout 30000ms exceeded.…waiting until \"networkidle\""}`
# after 31.3 s of wall clock; `wait_until="load"` then answered
# `{"ok": true, "title": "SSE fixture", "status": 200}` immediately, as a
# same-document navigation to the URL the page already sat on; and
# `browser_wait("networkidle")` returned `Wait failed: Timeout 6000ms exceeded.`
# at 6.0 s — while the document was there the whole time. Loopback needs no env
# change, because both the host guard and the egress floor let the machine itself
# through at their default, so the fakes below exercise the same landing check
# against the same kind of host rather than patching it away.


class _TimeoutPage:
    """A page whose `goto` times out on one wait, with the document present.

    `land_on` is where the URL is left when the wait fails: normally the target
    (Chromium commits the document, then the network never goes quiet), the
    error-document URL or an empty string when nothing loaded at all, or another
    machine's URL when the document that came back belongs to somebody else.
    """

    def __init__(self, url="http://127.0.0.1:5199/stream", title="SSE fixture",
                 raises_for=("networkidle",), land_on=None):
        self.url = url
        self._home = url
        self._title = title
        self._raises_for = tuple(raises_for)
        self._land_on = land_on
        self.goto_calls: list[tuple[str, dict]] = []

    async def goto(self, url, **kw):
        self.goto_calls.append((url, kw))
        if kw.get("wait_until") in self._raises_for:
            self.url = self._home if self._land_on is None else self._land_on
            raise TimeoutError(
                f'Page.goto: Timeout {kw.get("timeout", 30000)}ms exceeded.\n'
                'Call log:\n'
                f'  - navigating to "{url}", waiting until "{kw.get("wait_until")}"\n'
            )
        self.url = url
        return type("R", (), {"status": 200})()

    async def title(self):
        return self._title

    async def wait_for_load_state(self, state, **kw):
        if state == "networkidle":
            raise TimeoutError(
                f'Page.wait_for_load_state: Timeout {kw.get("timeout", 5000)}ms exceeded.'
            )

    def locator(self, selector):
        return _FakePage._Locator()


async def test_a_networkidle_timeout_on_a_loaded_page_is_never_idle_not_failure(monkeypatch):
    """Clause 1: a socket-holding page is loaded, and must be reported loaded."""
    page = _TimeoutPage()
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))

    out = json.loads(await browser_module._browser_navigate(page.url, wait_until="networkidle"))

    assert out.get("ok") is True, out
    assert out["title"] == "SSE fixture", out
    assert out["network_idle"] is False, out
    assert "error" not in out, \
        "the raw goto timeout is what sent the agent looking for a dead site"
    # The wait the caller asked for is still the wait that was issued.
    assert page.goto_calls[0][1]["wait_until"] == "networkidle", page.goto_calls


async def test_a_networkidle_timeout_with_no_document_still_reports_the_timeout(monkeypatch):
    """Clause 2 (first of three nodes): nothing loaded is not "loaded but never idle".

    An empty URL and `chrome-error://chromewebdata/` are what Chromium leaves
    behind when the navigation itself produced no document, so the timeout is the
    whole truth there and the raw text still has to reach the caller.
    """
    for dead_url in ("", "chrome-error://chromewebdata/"):
        page = _TimeoutPage(land_on=dead_url)
        monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))

        out = json.loads(await browser_module._browser_navigate(
            "http://127.0.0.1:5199/stream", wait_until="networkidle"))

        assert "ok" not in out, (dead_url, out)
        assert "Timeout" in out.get("error", ""), (dead_url, out)
        assert "network_idle" not in out, (dead_url, out)


async def test_a_networkidle_timeout_on_another_host_is_not_the_loaded_page(monkeypatch):
    """Clause 2 (second node): the document must be the page that was asked for.

    `page.url` survives a navigation that never committed as the *previous*
    document's URL, so "there is an http URL in `page.url`" alone would let a dead
    host report itself loaded — that is the first half. The second half is the
    other direction: an apex-to-`www.` hop, which `_hosts_match` deliberately
    tolerates, must still count as the same page, or the guard would refuse the
    navigations it exists to rescue. Both directions are asserted because a
    comparison that always answered "different" would fail only the second one,
    and a comparison that always answered "same" only the first.
    """
    # Another machine's URL: not the page that was requested.
    page = _TimeoutPage(land_on="http://198.51.100.7:5199/stream")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))

    out = json.loads(await browser_module._browser_navigate(
        "http://127.0.0.1:5199/stream", wait_until="networkidle"))

    assert "ok" not in out, out
    assert "Timeout" in out.get("error", ""), out
    assert "network_idle" not in out, out

    # A `www.` hop on the requested host: the same page, so still rescued.
    hopped = _TimeoutPage(url="https://www.example.com/",
                          land_on="https://www.example.com/", title="Example")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(hopped))

    out = json.loads(await browser_module._browser_navigate(
        "https://example.com/", wait_until="networkidle"))

    assert out.get("ok") is True, out
    assert out["network_idle"] is False, out


async def test_a_load_or_domcontentloaded_wait_timeout_is_still_an_error(monkeypatch):
    """Clause 2 (third node): the fallback is scoped to `networkidle`, not to timeouts.

    Both waits the schema recommends for an SPA are covered, because both are the
    recommendation the new sentence makes and neither has the never-idle excuse:
    a `load` or `domcontentloaded` timeout means the document did not arrive.
    """
    for wait in ("load", "domcontentloaded"):
        page = _TimeoutPage(raises_for=(wait,))
        monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))

        out = json.loads(await browser_module._browser_navigate(
            "http://127.0.0.1:5199/stream", wait_until=wait))

        assert out.get("ok") is not True, (wait, out)
        assert "Timeout" in out.get("error", ""), (wait, out)
        assert "network_idle" not in out, (wait, out)


async def test_the_same_fixture_on_a_quiet_wait_still_reports_a_measured_status(monkeypatch):
    """Control for the fixture above, and the "load unchanged" half of the item.

    `_TimeoutPage` raises only for the wait it is told to, so this node is the
    only one that reaches its success branch: if the fake had stopped returning a
    response handle at all, the three error nodes above would still pass and the
    rescue would be the only thing left standing. What a normal wait must keep: a
    measured `status`, and no `network_idle` claim, because nothing here measured
    the network's idleness when the wait finished on its own terms.
    """
    page = _TimeoutPage()
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))

    out = json.loads(await browser_module._browser_navigate(
        "http://127.0.0.1:5199/stream", wait_until="domcontentloaded"))

    assert out.get("ok") is True, out
    assert out["status"] == 200, out
    assert "network_idle" not in out, out
    assert page.goto_calls[0][1]["wait_until"] == "domcontentloaded", page.goto_calls


async def test_the_served_schema_no_longer_sends_the_model_to_networkidle_for_spas():
    """Clause 4: the sentence is the defect's other half — it is served every turn.

    `browser_navigate`'s own `wait_until` description told the model to use the
    one wait that cannot finish on the page class it names, and the model
    obeyed: the exchange is quoted tool_call-and-all in backlog #1088 —
    `browser_navigate(url="http://127.0.0.1:5199/", wait_until="networkidle")`
    against a vite dev server, answered `Page.goto: Timeout 30000ms exceeded` on
    a page that was serving. The transcript itself (`20260907_021204_iv4acf.json`,
    2026-09-06) is gone from `~/lloyd-data/sessions/`, whose oldest file on
    2026-09-24 is dated 2026-09-09, so the item is the surviving record.
    """
    tools = await browser_module.list_tools()
    nav = next(t for t in tools if t.name == "browser_navigate")
    # Read the serialized form: `inputSchema` is the alias the MCP wire carries,
    # and this is the byte string the model is handed every turn.
    served = nav.model_dump(by_alias=True)
    desc = served["inputSchema"]["properties"]["wait_until"]["description"]

    assert "Use networkidle for SPAs" not in desc, desc
    assert "load" in desc, desc
    assert "browser_wait" in desc, desc
    assert "text" in desc and "selector" in desc, desc
    # The value stays accepted — an existing caller must not break — it is only
    # no longer recommended.
    assert "networkidle" in served["inputSchema"]["properties"]["wait_until"]["enum"]


async def test_browser_wait_networkidle_on_a_loaded_page_is_never_idle(monkeypatch):
    """Clause 3: the same trap in `browser_wait`, same answer.

    The `networkidle` *condition* reaches the same dead end as the navigate wait,
    so it gets the same verdict; the `about:blank` half of this node is the scope
    proof — with no http document there is nothing to call loaded, and the caller
    must still hear "Wait failed".
    """
    page = _TimeoutPage()
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))

    out = json.loads(await browser_module._browser_wait("networkidle", timeout=2000))

    assert out.get("ok") is True, out
    assert out["network_idle"] is False, out
    assert "error" not in out, out

    # And the same wait on a page with no http document is still a failure:
    # there is nothing there to call loaded.
    blank = _TimeoutPage(url="about:blank")
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(blank))
    out = json.loads(await browser_module._browser_wait("networkidle", timeout=2000))
    assert out.get("ok") is not True, out
    assert "Wait failed" in out.get("error", ""), out


async def test_the_never_idle_verdict_crosses_the_mcp_seam_without_isError(monkeypatch):
    """What the model reads: `call_tool`'s wrapped payload, over the wire.

    `text_result` sniffs a top-level `error` key into `isError`, so the old
    behaviour was not merely a confusing sentence — the MCP result for a live
    page was a tool failure.
    """
    page = _TimeoutPage()
    monkeypatch.setattr(browser_module, "_get_page", lambda: _async(page))
    monkeypatch.setattr(browser_module, "_schedule_state_push", lambda name: None)

    res = await browser_module.call_tool(
        "browser_navigate",
        {"url": "http://127.0.0.1:5199/stream", "wait_until": "networkidle"})

    payload = json.loads(res.content[0].text)
    assert payload["ok"] is True, payload
    assert payload["network_idle"] is False, payload
    assert payload["title"] == "SSE fixture", payload
    assert res.is_error is False, \
        "a page that loaded and never went idle is not a failed tool call"
    assert res.model_dump(by_alias=True, exclude_none=True)["isError"] is False


# ── Clause 5: the same bad advice outside the schema ──────────────────────────
#
# `browser_navigate`'s `wait_until` description is not the only place the model
# learns to ask for `networkidle`. `skills/browser-session-extract/SKILL.md` gave
# `"wait_until": "networkidle"` as the whole of its one Navigation example, and a
# live skill body is injected into context when the skill fires — so the setting
# that cannot finish on a page holding a socket was also being served as an
# instruction. The example was changed to `load` on the vault's main (commit
# `147e549a`, 2026-09-23T16:34−07:00); #409 then archived that skill (vault commit
# `244dd6b8`, 2026-09-23T20:06−07:00), which is why the scan below now measures an
# empty set where the clause expected one permitted file. What is pinned is
# therefore strictly stronger than the clause as written: no live skill shows that
# wait, and no live skill names it even in prose. The clause's named file is not a
# live skill any more — an archived skill is neither advertised nor injected, so its
# copies of the string reach nobody, and this file does not assert on a dead file.
#
# The roots are named outright rather than read from `prompt_builder`'s constants or
# `app.paths`' vault path, for the reason `tests/test_archived_skill_artifacts.py:56-63`
# records: inside an automod worktree those re-anchor to the round's tree and return
# nothing, and a guard that reads nothing passes. `ACCOUNT_HOME` is the passwd entry,
# so it survives the gate's `HOME=<round>/home` (`app/paths.py:11-16`), and
# `Path.home()` is listed beside it because that is what the sibling test scans.
# Two positive controls, because a clean scan is the one green result that can mean
# nothing: the floor assert proves the roots resolved, and
# `test_the_skill_scan_reports_who_names_the_broken_wait` proves the two matchers
# fire against a synthetic tree. A scan whose roots all failed to resolve would
# otherwise report a clean board — the one answer worse than the defect.

_LIVE_SKILL_ROOTS = [
    Path.home() / "obsidian" / "skills",
    ROOT / "skills",
]

def _skill_roots() -> list[Path]:
    """The skill roots, deduped at the call site by ``resolve()``.

    ``ACCOUNT_HOME`` is the passwd entry and ``Path.home()`` is what the gate
    re-points at the round's home, so on a round these name one vault twice.
    """
    from app.data_root import ACCOUNT_HOME

    return [ACCOUNT_HOME / "obsidian" / "skills", *_LIVE_SKILL_ROOTS]


def _live_skill_files(roots: list[Path] | None = None) -> list[Path]:
    """Every live ``SKILL.md`` under ``roots`` (default: :func:`_skill_roots`).

    ``rglob`` would find ``skills/.archived/<name>/SKILL.md`` too, so the archived
    tree is dropped by name rather than missed by an accident of depth: three
    archived skills still carry this exact string on 2026-09-24
    (``browser-navigate-handling``, ``browser-navigate-timeout``,
    ``browser-session-extract``), and they are out of scope because an archived
    skill is neither advertised nor readable through ``skills_read``.
    """
    found: set[Path] = set()
    for root in (_skill_roots() if roots is None else roots):
        if not root.is_dir():
            continue
        for path in root.rglob("SKILL.md"):
            if ".archived" in path.parts:
                continue
            # `resolve()` is not cosmetic: under the gate `Path.home()` is
            # `<round>/home`, whose `obsidian` is a symlink onto the real vault, so
            # the two vault roots here name the SAME file by two different path
            # strings. Deduped textually the scan reports every live skill twice,
            # and the assert on who names `networkidle` fails on its own
            # duplicate rather than on any drift.
            found.add(path.resolve())
    return sorted(found)


def test_no_live_skill_recommends_the_networkidle_wait():
    """Clause 5: the guidance the model copies must stop naming the broken wait.

    Three asserts, because the clause has three claims in it, and each fails for a
    different regression: the scan having measured nothing, some skill showing the
    bad value, and the surviving mention drifting from a prohibition back into an
    example.
    """
    files = _live_skill_files()
    # 191 live SKILL.md files under ~/obsidian/skills on 2026-09-24 (161 more under
    # skills/.archived/, which are excluded), counted by this same walk. The floor is
    # well below that and well above zero: it exists to turn "the roots did not
    # resolve" into a failure instead of a green run.
    assert len(files) >= 150, (
        f"the scan found only {len(files)} live SKILL.md files under "
        f"{[str(r) for r in _skill_roots()]} — it has measured nothing, so a "
        "clean result here would mean nothing")

    def _rel(p: Path) -> str:
        return f"{p.parent.name}/{p.name}"

    shown = sorted(_rel(p) for p in files
                   if '"wait_until": "networkidle"'
                   in p.read_text(encoding="utf-8", errors="replace"))
    assert not shown, (
        f"{shown} still shows a navigation example with the wait that never fires "
        "on a page holding a live socket; an agent copying it gets a 30 s timeout "
        "on a fully loaded page")

    naming = sorted(_rel(p) for p in files
                    if "networkidle" in p.read_text(encoding="utf-8", errors="replace"))
    # The clause permitted exactly one skill to name the word; that skill is
    # archived (see the header above), so the surviving reading is the stricter one:
    # nobody living. A future skill that has to discuss the wait — to tell the model
    # not to use it — reaches this line as a deliberate edit, not as a silent pass.
    assert naming == [], (
        f"{naming} names the wait that never fires on a page holding a live socket; "
        "no live skill is permitted to")


def test_the_skill_scan_reports_who_names_the_broken_wait(tmp_path):
    """The matcher the clean scan above rests on can fail; this tree makes it.

    An empty result over a corpus is only evidence if the same code reports the
    string when it is present. `tmp_path` is a synthetic skills root holding one
    skill that shows the bad wait in an example, one that names it in prose only,
    and one archived copy under the same name — so this node pins all three ways
    the real scan could be quietly vacuous: a matcher that matches nothing, an
    exclusion that excludes everything, and an exclusion that does not exclude the
    archive.
    """
    bodies = {
        "shows-the-wait": '{"name": "browser_navigate", "arguments": '
                          '{"url": "https://example.com", "wait_until": "networkidle"}}',
        "names-it-in-prose": "networkidle is the wait this page never reaches.",
    }
    for name, body in bodies.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text(body, encoding="utf-8")
    archived = tmp_path / ".archived" / "shows-the-wait"
    archived.mkdir(parents=True)
    (archived / "SKILL.md").write_text(bodies["shows-the-wait"], encoding="utf-8")

    files = _live_skill_files([tmp_path])
    assert [p.parent.name for p in files] == sorted(bodies), \
        "the archived copy shares a directory name with a live one, so a walk that " \
        "did not exclude `.archived` would show up here as a duplicate"

    def _who(substring: str) -> list[str]:
        return sorted(p.parent.name for p in files
                      if substring in p.read_text(encoding="utf-8"))

    assert _who('"wait_until": "networkidle"') == ["shows-the-wait"], \
        "the example matcher found nothing even with the example in front of it"
    assert _who("networkidle") == sorted(bodies), \
        "the prose matcher found nothing even with the word in front of it"

