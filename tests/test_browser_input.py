"""The Browser tab's pointer and keyboard, and the mask that makes them safe.

The tab was a mirror with a URL bar, so a page that wanted a login could be
watched and never answered. `browser.input_from_ui` (aggregator `POST
/browser/input`, proxied by the backend's `POST /api/browser/input`) is a human's
click, text, key or wheel on the agent's Chromium.

What is typed there is a credential, and Chromium's accessibility tree carries a
password field's value in plain text — `- textbox "Password": hunter2`, measured
on /usr/bin/chromium on 2026-11-16, while the screenshot shows dots. That tree is
what `browser_snapshot` hands the model and what the frame hands Mission Control,
so `_aria_snapshot_masked` masks it on every read. The real-Chromium nodes below
are the ones that prove it, since a fake page would only prove the regex.
"""
import asyncio
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import browser as bm  # noqa: E402
from app.routers import browser as browser_router  # noqa: E402

SECRET = "hunter2-Sx9!q"
MASK = bm._PASSWORD_MASK


# ── The mask, on text ─────────────────────────────────────────────────────────

def test_a_password_value_is_masked_and_its_neighbours_are_not():
    raw = (
        '- textbox "Email": alan@example.com\n'
        f'- textbox "Password": {SECRET}\n'
        '- button "Sign in"'
    )
    out = bm._mask_password_values(raw, [SECRET])
    assert SECRET not in out
    assert f'- textbox "Password": {MASK}' in out
    assert '- textbox "Email": alan@example.com' in out
    assert '- button "Sign in"' in out


def test_the_mask_survives_a_field_that_changed_between_the_two_reads():
    """The tree and the DOM are two calls; a keystroke can land between them."""
    longer = bm._mask_password_values(f'- textbox "P": {SECRET}x', [SECRET])
    shorter = bm._mask_password_values(f'- textbox "P": {SECRET[:-1]}', [SECRET])
    assert SECRET not in longer and SECRET[:-1] not in shorter


def test_the_mask_handles_quoting_attrs_and_a_name_with_a_colon():
    raw = (
        f'  - textbox "Note: pw" [disabled]: "{SECRET}: tail"\n'
        '  - textbox "Search": cats'
    )
    out = bm._mask_password_values(raw, [SECRET])
    assert SECRET not in out
    assert f'  - textbox "Note: pw" [disabled]: {MASK}' in out
    assert '- textbox "Search": cats' in out


def test_not_knowing_which_fields_are_passwords_masks_them_all():
    raw = '- textbox "Email": alan@example.com\n- textbox "Code": 123456'
    out = bm._mask_password_values(raw, [], mask_all=True)
    assert "alan@example.com" not in out and "123456" not in out


async def test_a_failed_password_scan_fails_closed():
    class _Loc:
        async def aria_snapshot(self):
            return f'- textbox "Password": {SECRET}'

    class _Page:
        def locator(self, _sel):
            return _Loc()

        async def evaluate(self, _js):
            raise RuntimeError("Execution context was destroyed")

    assert SECRET not in await bm._aria_snapshot_masked(_Page())


def test_every_aria_read_goes_through_the_mask():
    """A fourth read added later would reopen the leak without failing a node."""
    src = (ROOT / "agent_mcp" / "browser.py").read_text(encoding="utf-8")
    reads = re.findall(r"await [\w.()\"]+\.aria_snapshot\(", src)
    assert reads == ['await target.locator("body").aria_snapshot('], (
        f"an aria read outside _aria_snapshot_masked: {reads}")
    assert src.count("await _aria_snapshot_masked(") == 3


# ── The event, refused before it touches anything ─────────────────────────────

@pytest.mark.parametrize("event, needle", [
    ("nope", "object"),
    ({"kind": "drag"}, "kind must be"),
    ({"kind": "type"}, "text is required"),
    ({"kind": "type", "text": ""}, "text is required"),
    ({"kind": "type", "text": "x" * (bm._UI_MAX_TEXT + 1)}, "limited"),
    ({"kind": "key", "key": "rm -rf"}, "unsupported key"),
    ({"kind": "key", "key": "Hyper+a"}, "unsupported key"),
])
async def test_malformed_events_are_refused(event, needle):
    out = await bm.input_from_ui(event)
    assert needle in out.get("error", ""), out


def test_key_names_are_playwrights_and_nothing_else():
    for good in ("Enter", "Tab", "Shift+Tab", "ControlOrMeta+a", "Control+Shift+z",
                 "a", "F5", "Alt+ArrowLeft", "Space"):
        assert bm._valid_ui_key(good), good
    for bad in ("", "Enterr", "Shift+", "Shift+Hyper", "ab", "x" * 60, 7, None,
                "\n"):
        assert not bm._valid_ui_key(bad), bad


async def test_input_with_no_page_never_launches_a_browser(monkeypatch):
    launched = []

    async def _ensure():
        launched.append(1)

    monkeypatch.setattr(bm, "_ensure_browser", _ensure)
    monkeypatch.setattr(bm, "_existing_page", lambda: None)
    out = await bm.input_from_ui({"kind": "click", "x": 1, "y": 1})
    assert "no page is open" in out["error"]
    assert launched == []


async def test_a_failure_while_typing_never_quotes_the_text(monkeypatch, caplog):
    """Playwright's errors echo their arguments; for `type` that is a password."""
    class _Keyboard:
        async def type(self, text):
            raise RuntimeError(f"Keyboard.type: Target closed while typing {text!r}")

    class _Page:
        keyboard = _Keyboard()
        viewport_size = {"width": 1280, "height": 800}

    monkeypatch.setattr(bm, "_existing_page", lambda: _Page())
    caplog.set_level(logging.DEBUG)
    out = await bm.input_from_ui({"kind": "type", "text": SECRET})
    assert out == {"error": "type failed (RuntimeError)"}
    assert SECRET not in caplog.text


# ── The two routes ────────────────────────────────────────────────────────────

def test_the_aggregator_serves_the_route_and_the_backend_proxies_it():
    from app.aggregator_config import route
    from agent_mcp import main as mcp_main

    assert route("browser_input").endswith("/browser/input")
    assert route("browser_input") == browser_router._MCP_INPUT_URL
    src = (ROOT / "agent_mcp" / "main.py").read_text(encoding="utf-8")
    assert 'Route("/browser/input", browser_input, methods=["POST"])' in src
    assert callable(mcp_main.browser_input)


async def test_the_backend_forwards_the_event_with_the_credential(monkeypatch, caplog):
    sent: list[tuple] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"ok": True, "kind": "type"}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            sent.append((url, json, headers))
            return _Resp()

    monkeypatch.setattr(browser_router.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(browser_router, "auth_headers_for", lambda url: {"X-Cred": "c"})
    caplog.set_level(logging.DEBUG)
    resp = await browser_router.post_browser_input({"kind": "type", "text": SECRET})
    assert json.loads(resp.body) == {"ok": True, "kind": "type"}
    assert sent == [(browser_router._MCP_INPUT_URL, {"kind": "type", "text": SECRET},
                     {"X-Cred": "c"})]
    assert SECRET not in caplog.text


async def test_an_unreachable_aggregator_is_named_without_the_body(monkeypatch, caplog):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            raise ConnectionError(f"refused while sending {json}")

    monkeypatch.setattr(browser_router.httpx, "AsyncClient", _Client)
    caplog.set_level(logging.DEBUG)
    resp = await browser_router.post_browser_input({"kind": "type", "text": SECRET})
    body = json.loads(resp.body)
    assert "unreachable" in body["error"] and SECRET not in body["error"]
    assert SECRET not in caplog.text


# ── End to end, on a real Chromium ────────────────────────────────────────────

CHROMIUM = "/usr/bin/chromium"
requires_chromium = pytest.mark.skipif(
    not (Path(CHROMIUM).is_file() and importlib.util.find_spec("playwright")),
    reason="needs playwright and /usr/bin/chromium",
)

LOGIN_DOC = (
    "<!doctype html><html><body style='margin:0'>"
    "<form id=f onsubmit=\"document.title='submitted:'+document.getElementById('u').value;"
    "return false\">"
    "<label>Email <input id=u style='display:block;width:300px;height:30px'></label>"
    "<label>Password <input id=p type=password style='display:block;width:300px;height:30px'></label>"
    "<button>Sign in</button></form>"
    "<p id=w>double click this word</p>"
    "<a id=pop href='about:blank' target=_blank>pop</a>"
    "<iframe width=400 height=120 srcdoc=\"<label>PIN <input id=q type=password></label>\">"
    "</iframe></body></html>"
)


@pytest.fixture
async def live_page(monkeypatch):
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(executable_path=CHROMIUM, headless=True,
                                       args=["--no-sandbox", "--disable-dev-shm-usage"])
    context = await browser.new_context(viewport={"width": 800, "height": 600})
    page = await context.new_page()
    await page.set_content(LOGIN_DOC, wait_until="domcontentloaded")
    frames: list[dict] = []

    async def _push(tool_name):
        frame = await bm._capture_state(tool_name)
        if frame is not None:
            frames.append(frame)

    async def _page():
        return bm._active_page

    monkeypatch.setattr(bm, "_context", context)
    monkeypatch.setattr(bm, "_active_page", page)
    monkeypatch.setattr(bm, "_get_page", _page)
    monkeypatch.setattr(bm, "_push_browser_state", _push)
    monkeypatch.setattr(bm, "_UI_TRAILING_PUSH_S", 0.05)
    monkeypatch.setattr(bm, "_UI_SETTLE_S", 0.02)
    monkeypatch.setattr(bm, "_ui_before", None)
    monkeypatch.setattr(bm, "_ui_first_frame", None)
    try:
        yield page, frames
    finally:
        if bm._ui_trailing_push is not None:
            bm._ui_trailing_push.cancel()
        await context.close()
        await browser.close()
        await pw.stop()


async def _center(page, selector, frame=None):
    box = await (frame or page).locator(selector).bounding_box()
    return {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}


@requires_chromium
async def test_a_human_logs_in_and_no_read_carries_the_password(live_page, caplog):
    page, frames = live_page
    caplog.set_level(logging.DEBUG)

    assert (await bm.input_from_ui({"kind": "click", **await _center(page, "#u")}))["ok"]
    assert (await bm.input_from_ui({"kind": "type", "text": "alan@example.com"}))["ok"]
    assert (await bm.input_from_ui({"kind": "key", "key": "Tab"}))["ok"]
    assert (await bm.input_from_ui({"kind": "type", "text": SECRET}))["ok"]
    child = page.frames[1]
    assert (await bm.input_from_ui({"kind": "click", **await _center(page, "iframe")}))["ok"]
    await child.locator("#q").focus()
    assert (await bm.input_from_ui({"kind": "type", "text": "8642"}))["ok"]

    # The input landed where a person's would have.
    assert await page.locator("#u").input_value() == "alan@example.com"
    assert await page.locator("#p").input_value() == SECRET
    assert await child.locator("#q").input_value() == "8642"

    # And no read hands the values on: not the frame Mission Control gets, not
    # the snapshot the model gets, not a log line.
    # An event returns once it is delivered; its frame follows off the request.
    await bm._ui_settled()
    assert frames, "every burst of events pushes a frame"
    last = frames[-1]
    assert last["tool"] == "user_input"
    assert "alan@example.com" in last["snapshot"]
    assert SECRET not in last["snapshot"] and "8642" not in last["snapshot"]
    assert MASK in last["snapshot"]
    snap = json.loads(await bm._browser_snapshot())["snapshot"]
    assert SECRET not in snap and "8642" not in snap
    assert snap.count(MASK) == 2
    assert SECRET not in caplog.text and "8642" not in caplog.text


@requires_chromium
async def test_enter_submits_and_a_double_click_is_one_double_click(live_page):
    page, _ = live_page
    await bm.input_from_ui({"kind": "click", **await _center(page, "#u")})
    await bm.input_from_ui({"kind": "type", "text": "a@b.c"})
    await bm.input_from_ui({"kind": "key", "key": "Enter"})
    assert await page.title() == "submitted:a@b.c"

    word = await _center(page, "#w")
    word["x"] = (await page.locator("#w").bounding_box())["x"] + 10
    await page.evaluate("""() => { window.downs = 0;
        document.getElementById('w').addEventListener('mousedown', () => window.downs++) }""")
    await bm.input_from_ui({"kind": "click", **word, "count": 1})
    await bm.input_from_ui({"kind": "click", **word, "count": 2})
    assert await page.evaluate("window.getSelection().toString()") == "double"
    assert await page.evaluate("window.downs") == 2


@requires_chromium
async def test_a_click_that_opens_a_tab_moves_the_view_to_it(live_page):
    page, frames = live_page
    await bm.input_from_ui({"kind": "click", **await _center(page, "#pop")})
    await bm._ui_settled()
    assert bm._active_page is not page
    assert len(bm._context.pages) == 2
    # The trailing frame lands after the burst, on the new tab.
    await asyncio.sleep(0.2)
    assert frames[-1]["url"] == "about:blank"


@requires_chromium
async def test_scroll_and_clamped_coordinates(live_page):
    page, _ = live_page
    await page.evaluate("document.body.style.height = '5000px'")
    out = await bm.input_from_ui({"kind": "scroll", "x": 9999, "y": -5, "dx": 0, "dy": 700})
    assert out["ok"]
    await page.wait_for_function("window.scrollY > 0", timeout=2000)


# ── Input does not wait for its frame, and the view is live ───────────────────

@requires_chromium
async def test_an_event_returns_before_its_frame_and_the_frame_still_arrives(live_page, monkeypatch):
    """Until 2026-11-17 every event awaited a settle, a landing check and a full
    frame under the input lock (~0.4 s a click, serialised), which is what made
    the tab feel like remote desktop over a modem."""
    page, frames = live_page
    monkeypatch.setattr(bm, "_UI_SETTLE_S", 0.3)
    out = await bm.input_from_ui({"kind": "click", **await _center(page, "#u")})
    assert out == {"ok": True, "kind": "click"}
    assert frames == [], "the event waited for its frame"
    # A burst is one settle, not one per key.
    for ch in "abc":
        await bm.input_from_ui({"kind": "key", "key": ch})
    monkeypatch.setattr(bm, "_UI_SETTLE_S", 0.02)
    await bm.input_from_ui({"kind": "key", "key": "d"})
    await bm._ui_settled()
    assert len(frames) == 1 and frames[0]["tool"] == "user_input"
    assert await page.locator("#u").input_value() == "abcd"


@requires_chromium
async def test_the_live_view_streams_a_page_that_changes_by_itself(live_page, monkeypatch):
    """No tool call and no input: the pixels still arrive, light, and stop being
    hauled when the backend says nobody is watching."""
    page, _ = live_page
    posts: list[dict] = []
    watchers = {"n": 1}

    class _Resp:
        def json(self):
            return {"ok": True, "subscribers": watchers["n"]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **k):
            posts.append(json)
            return _Resp()

    monkeypatch.setattr(bm.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(bm, "_LIVE_MIN_INTERVAL_S", 0.02)
    monkeypatch.setattr(bm, "_LIVE_IDLE_POLL_S", 0.2)
    monkeypatch.setenv("LLOYD_BROWSER_LIVE", "1")
    await page.evaluate("""() => { let n = 0; setInterval(() => {
        document.getElementById('w').textContent = 'tick ' + (n++) }, 50) }""")
    bm._ensure_live(page)
    try:
        await asyncio.sleep(1.0)
        assert len(posts) >= 3, f"a changing page produced {len(posts)} live frames"
        assert all(f["live"] is True and f["screenshot_b64"] for f in posts)
        assert all("snapshot" not in f and "refs" not in f for f in posts), \
            "a live frame carries pixels, never the accessibility tree"
        # Nobody watching: the stream drops to the idle poll.
        watchers["n"] = 0
        await asyncio.sleep(0.3)
        before = len(posts)
        await asyncio.sleep(0.7)
        assert len(posts) - before <= 5, "still streaming with no viewer"
    finally:
        bm._live["task"].cancel()
        await asyncio.gather(bm._live["task"], return_exceptions=True)


def test_a_live_frame_keeps_the_tree_for_late_readers_and_fans_out_light(monkeypatch):
    monkeypatch.setattr(browser_router, "_latest", None)
    monkeypatch.setattr(browser_router, "_subscribers", set())
    q = browser_router.subscribe()
    full = {"tool": "browser_click", "url": "https://a.test/", "title": "A", "ts": 1.0,
            "screenshot_b64": "OLD", "snapshot": "- button", "refs": [{"ref": "e1"}]}
    browser_router._publish(full)
    browser_router._publish({"live": True, "url": "https://a.test/", "ts": 2.0,
                             "mime": "image/jpeg", "screenshot_b64": "NEW"})
    kept = browser_router._latest
    assert kept["screenshot_b64"] == "NEW" and kept["snapshot"] == "- button"
    assert kept["tool"] == "browser_click" and kept["title"] == "A"
    assert browser_router.latest_frame_summary()["driven_by"] == "browser_click"
    q.get_nowait()
    sent = q.get_nowait()
    assert "snapshot" not in sent and sent["live"] is True, "the stream must stay light"
    # A full queue costs a slow viewer pixels, never its subscription.
    for i in range(40):
        browser_router._publish({"live": True, "screenshot_b64": str(i)})
    assert q in browser_router._subscribers
