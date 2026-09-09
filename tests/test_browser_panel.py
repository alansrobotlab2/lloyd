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

def test_still_fourteen_browser_tools_with_unchanged_names():
    src = (ROOT / "agent_mcp" / "browser.py").read_text(encoding="utf-8")
    assert src.count('Tool(name="browser_') == 14


async def test_tool_schemas_still_expose_the_same_fourteen_tools():
    tools = await browser_module.list_tools()
    assert len(tools) == 14
    assert {t.name for t in tools} == {
        "browser_navigate", "browser_snapshot", "browser_click", "browser_type",
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
    assert "8500/browser/navigate" in src, \
        "the proxy must target the aggregator, which is where the browser is"


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
