#!/usr/bin/env python3
"""
Lloyd MCP Server: Browser — full browser control via Playwright.

Tools (Phase 1): browser_navigate, browser_snapshot, browser_click,
                 browser_type, browser_scroll, browser_press, browser_tabs
Tools (Phase 2): browser_screenshot, browser_evaluate, browser_fill, browser_wait
Tools (Phase 3): browser_select, browser_drag, browser_cookies
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
import urllib.parse
from pathlib import Path

import httpx
from mcp.types import Tool

from agent_mcp._shared import text_result

logger = logging.getLogger("lloyd-browser")

# ── Config ─────────────────────────────────────────────────────────────────────

CHROMIUM_EXECUTABLE = "/usr/bin/chromium"
from app.config import CONFIG, service_url  # noqa: E402
from app.paths import SCREENSHOTS_DIR  # noqa: E402  (was an absolute literal)
MAX_SNAPSHOT_CHARS = 8000
MAX_TABS = 10

# Where to publish post-call browser state (Mission Control's Browser tab).
LLOYD_API = os.environ.get("LLOYD_API_URL") or service_url("backend", "http://127.0.0.1:8080")

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _resolve_headless() -> bool:
    """Whether to launch Chromium without a UI. Defaults to headless.

    Headed mode needs a live display, and the agent-worker unit carries no
    DISPLAY of its own (backlog #402), so a literal headed default meant the
    whole browser surface came up only when someone happened to be logged in
    at the desktop. Opt back in for debugging with `browser.headless: false`
    in config.yaml, or LLOYD_BROWSER_HEADLESS=0 in the environment — the env
    wins, so a one-off debug doesn't need a config edit.
    """
    raw = os.environ.get("LLOYD_BROWSER_HEADLESS")
    if raw is None or not raw.strip():
        configured = (CONFIG.get("browser") or {}).get("headless")
        if configured is None:
            return True
        raw = str(configured)
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    logger.warning("browser: unparseable headless value %r — staying headless", raw)
    return True


# ── SSRF protection ────────────────────────────────────────────────────────────

_PRIVATE_IP_PATTERNS = [
    re.compile(r"^127\."),
    re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^0\."),
    re.compile(r"^169\.254\."),
    re.compile(r"^::1$"),
    re.compile(r"^fc00:", re.IGNORECASE),
    re.compile(r"^fd", re.IGNORECASE),
    re.compile(r"^fe80:", re.IGNORECASE),
]


def _is_private_host(hostname: str) -> bool:
    if not hostname or hostname.lower() == "localhost":
        return True
    return any(p.match(hostname) for p in _PRIVATE_IP_PATTERNS)


# ── Browser state (persistent for server lifetime) ─────────────────────────────

_pw = None        # Playwright instance
_browser = None   # Browser instance
_context = None   # BrowserContext
_active_page = None  # Currently focused Page
_ref_map: dict[str, dict] = {}  # "e1" -> {"role": ..., "name": ..., "occurrence": ...}
# Serializes launch/teardown: concurrent turns (user + ambient + autonomy)
# could otherwise interleave _ensure_browser() and launch two Chromiums,
# leaking one.
_launch_lock = asyncio.Lock()

_INTERACTIVE_ROLES = {
    "button", "link", "textbox", "searchbox", "checkbox",
    "combobox", "listbox", "menuitem", "menuitemcheckbox",
    "menuitemradio", "option", "radio", "slider", "spinbutton",
    "switch", "tab", "treeitem",
}

# aria_snapshot line pattern: "- role_name "name" [attrs]: inline_text"
_ARIA_LINE_RE = re.compile(
    r'^(?P<indent>\s*)'          # leading whitespace
    r'- '                        # list marker
    r'(?P<role>\S+)'             # role (no spaces)
    r'(?:\s+"(?P<name>[^"]*)")?' # optional "name"
    r'(?:\s+\[(?P<attrs>[^\]]*)\])?' # optional [attrs]
    r'(?::\s*(?P<text>.*))?$'    # optional ": inline text"
)


# ── Browser lifecycle ──────────────────────────────────────────────────────────

async def _ensure_browser():
    global _pw, _browser, _context
    if _browser and _browser.is_connected() and _context:
        return _context
    async with _launch_lock:
        return await _launch_browser_locked()


async def _launch_browser_locked():
    global _pw, _browser, _context
    # Re-check under the lock: a concurrent caller may have just launched.
    if _browser and _browser.is_connected() and _context:
        return _context
    # Clean up stale instance
    if _pw:
        try:
            await _pw.stop()
        except Exception:
            pass
        _pw = None
    from playwright.async_api import async_playwright
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(
        executable_path=CHROMIUM_EXECUTABLE,
        headless=_resolve_headless(),
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ],
    )
    _context = await _browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        ),
    )
    return _context


async def _get_page():
    global _active_page
    ctx = await _ensure_browser()
    pages = ctx.pages
    if not pages:
        _active_page = await ctx.new_page()
    elif _active_page is None or _active_page not in pages or _active_page.is_closed():
        _active_page = pages[-1]
    return _active_page


# ── Accessibility tree snapshot ────────────────────────────────────────────────

def _parse_aria_snapshot(text: str, ref_map: dict) -> tuple[str, int]:
    """
    Parse Playwright's aria_snapshot() YAML-like output.
    Injects ref IDs (e1, e2, ...) next to interactive elements.
    Returns (annotated_text, ref_count).
    """
    lines = text.split("\n")
    result_lines = []
    counter = 0
    role_name_counts: dict = {}

    for line in lines:
        m = _ARIA_LINE_RE.match(line)
        if not m:
            result_lines.append(line)
            continue

        indent = m.group("indent") or ""
        role = (m.group("role") or "").strip().rstrip(":")
        name = m.group("name") or ""
        attrs = m.group("attrs") or ""
        text_part = m.group("text") or ""

        ref_str = ""
        if role in _INTERACTIVE_ROLES and name:
            counter += 1
            ref_id = f"e{counter}"
            key = (role, name)
            occ = role_name_counts.get(key, 0)
            role_name_counts[key] = occ + 1
            ref_map[ref_id] = {"role": role, "name": name, "occurrence": occ}
            ref_str = f" [{ref_id}]"

        new_content = f"- {role}"
        if name:
            new_content += f' "{name}"'
        if ref_str:
            new_content += ref_str
        if attrs:
            new_content += f" [{attrs}]"
        if text_part:
            new_content += f": {text_part}"

        result_lines.append(indent + new_content)

    return "\n".join(result_lines), counter


async def _locate(page, ref_id: str):
    """Reconstruct a Playwright locator from a ref ID."""
    info = _ref_map.get(ref_id)
    if not info:
        raise ValueError(
            f"Unknown ref '{ref_id}'. Call browser_snapshot to get fresh refs."
        )
    return page.get_by_role(info["role"], name=info["name"], exact=True).nth(
        info["occurrence"]
    )


# ── Tool implementations ───────────────────────────────────────────────────────

async def _browser_navigate(url: str, wait_until: str = "domcontentloaded") -> str:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return json.dumps({"error": f"Invalid URL: {url}"})
    if parsed.scheme not in ("http", "https"):
        return json.dumps({"error": "Only http/https URLs are supported"})

    if wait_until not in ("load", "domcontentloaded", "networkidle", "commit"):
        wait_until = "domcontentloaded"

    page = await _get_page()
    _ref_map.clear()
    try:
        resp = await page.goto(url, wait_until=wait_until, timeout=30000)
        return json.dumps({
            "ok": True,
            "url": page.url,
            "title": await page.title(),
            "status": resp.status if resp else 0,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


async def _browser_snapshot(full: bool = False) -> str:
    global _ref_map
    page = await _get_page()
    _ref_map = {}
    try:
        raw = await page.locator("body").aria_snapshot()
    except Exception as exc:
        return json.dumps({"error": f"Accessibility snapshot failed: {exc}"})

    if not raw or not raw.strip():
        return json.dumps({"error": "Empty accessibility tree — page may not have loaded."})

    annotated, ref_count = _parse_aria_snapshot(raw, _ref_map)
    title = await page.title()
    full_text = f"[Page] {title} — {page.url}\n{annotated}"

    if len(full_text) > MAX_SNAPSHOT_CHARS:
        full_text = (
            full_text[:MAX_SNAPSHOT_CHARS]
            + f"\n\n[...truncated at {MAX_SNAPSHOT_CHARS} chars]"
        )

    return json.dumps({"snapshot": full_text, "refs": ref_count})


async def _browser_click(ref: str, button: str = "left") -> str:
    page = await _get_page()
    try:
        loc = await _locate(page, ref)
        btn = button if button in ("left", "right", "middle") else "left"
        await loc.click(button=btn, timeout=10000)
        return json.dumps({"ok": True, "ref": ref})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Click failed: {exc}"})


async def _browser_type(ref: str, text: str, clear: bool = False) -> str:
    page = await _get_page()
    try:
        loc = await _locate(page, ref)
        if clear:
            await loc.clear(timeout=5000)
        await loc.press_sequentially(text, delay=30)
        return json.dumps({"ok": True, "ref": ref})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Type failed: {exc}"})


async def _browser_scroll(direction: str = "down", amount: int = 300) -> str:
    page = await _get_page()
    dx, dy = 0, 0
    if direction == "down":
        dy = amount
    elif direction == "up":
        dy = -amount
    elif direction == "right":
        dx = amount
    elif direction == "left":
        dx = -amount
    try:
        await page.mouse.wheel(dx, dy)
        return json.dumps({"ok": True, "direction": direction, "amount": amount})
    except Exception as exc:
        return json.dumps({"error": f"Scroll failed: {exc}"})


async def _browser_press(key: str) -> str:
    page = await _get_page()
    try:
        await page.keyboard.press(key)
        return json.dumps({"ok": True, "key": key})
    except Exception as exc:
        return json.dumps({"error": f"Press failed: {exc}"})


async def _browser_tabs(action: str, page_id: int | None = None, url: str | None = None) -> str:
    global _active_page, _ref_map
    ctx = await _ensure_browser()
    pages = ctx.pages

    if action == "list":
        tabs = []
        for i, p in enumerate(pages):
            try:
                title = await p.title()
            except Exception:
                title = ""
            tabs.append({"id": i, "url": p.url, "title": title, "active": p == _active_page})
        return json.dumps({"tabs": tabs})

    elif action == "switch":
        if page_id is None or page_id < 0 or page_id >= len(pages):
            return json.dumps({"error": f"Invalid page_id {page_id}. Use browser_tabs(list) first."})
        _active_page = pages[page_id]
        _ref_map.clear()
        return json.dumps({"ok": True, "id": page_id, "url": _active_page.url})

    elif action == "close":
        if page_id is None or page_id < 0 or page_id >= len(pages):
            return json.dumps({"error": f"Invalid page_id {page_id}"})
        await pages[page_id].close()
        remaining = ctx.pages
        if not remaining:
            _active_page = await ctx.new_page()
        elif _active_page is None or _active_page.is_closed() or _active_page not in remaining:
            _active_page = remaining[-1]
        _ref_map.clear()
        return json.dumps({"ok": True, "closed_id": page_id})

    elif action == "new":
        if len(pages) >= MAX_TABS:
            await pages[0].close()
        new_page = await ctx.new_page()
        if url:
            try:
                parsed = urllib.parse.urlparse(url)
            except Exception:
                await new_page.close()
                return json.dumps({"error": f"Invalid URL: {url}"})
            if parsed.scheme not in ("http", "https"):
                await new_page.close()
                return json.dumps({"error": "Only http/https URLs supported"})
            await new_page.goto(url, wait_until="domcontentloaded", timeout=30000)
        _active_page = new_page
        _ref_map.clear()
        return json.dumps({"ok": True, "id": len(ctx.pages) - 1, "url": new_page.url})

    return json.dumps({"error": f"Unknown action '{action}'. Use: list, switch, close, new"})


async def _browser_screenshot() -> str:
    global SCREENSHOTS_DIR
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    page = await _get_page()
    try:
        ts = int(time.time())
        path = SCREENSHOTS_DIR / f"screenshot_{ts}.png"
        await page.screenshot(path=str(path), full_page=False)
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode()
        return json.dumps({"ok": True, "path": str(path), "size_bytes": len(data), "data_base64": b64})
    except Exception as exc:
        return json.dumps({"error": f"Screenshot failed: {exc}"})


# ── Live state for Mission Control (#278) ─────────────────────────────────────

def _existing_page():
    """The focused page, or None — never launches a browser to find one.

    `_get_page()` launches Chromium if the session is gone, and this runs
    after *every* tool call including `browser_cookies`, which is a perfectly
    reasonable thing to ask about with no page open. A state mirror must not
    be able to resurrect a browser the agent shut down.
    """
    if _context is None or _active_page is None:
        return None
    try:
        if _active_page.is_closed() or _active_page not in _context.pages:
            return None
    except Exception:
        return None
    return _active_page


async def _capture_state(tool_name: str) -> dict | None:
    """Build one Browser-tab frame: the current view plus its a11y refs.

    Returns None when no page exists — asking for state right after a tab
    was closed is normal, not an error worth pushing.

    JPEG rather than PNG because this travels on every single tool call: the
    same 1280x800 frame measured ~190 KB as PNG against ~15-25 KB here, and
    the frame is only ever displayed scaled down in the browser tab.
    """
    page = _existing_page()
    if page is None:
        return None
    try:
        shot = await page.screenshot(type="jpeg", quality=72, full_page=False)
    except Exception as exc:
        logger.debug("browser state: screenshot failed: %s", exc)
        return None
    try:
        title = await page.title()
    except Exception:
        title = ""
    snapshot = ""
    refs: dict[str, dict] = {}
    try:
        raw = await page.locator("body").aria_snapshot()
        snapshot, _ = _parse_aria_snapshot(raw, refs)
    except Exception as exc:
        logger.debug("browser state: aria_snapshot failed: %s", exc)
    return {
        "tool": tool_name,
        "url": page.url,
        "title": title,
        "ts": time.time(),
        "mime": "image/jpeg",
        "screenshot_b64": base64.b64encode(shot).decode(),
        "snapshot": snapshot[:MAX_SNAPSHOT_CHARS],
        "refs": [
            {"ref": rid, "role": info.get("role", ""), "name": info.get("name", "")}
            for rid, info in refs.items()
        ],
    }


async def _push_browser_state(tool_name: str) -> None:
    """Publish the current browser view to the backend's /api/browser/state.

    Deliberately cannot raise. The browser tools have exactly one caller —
    the agent turn — and Mission Control not listening (or being down entirely)
    must never turn a successful navigation into a failed tool call.
    """
    try:
        frame = await _capture_state(tool_name)
        if frame is None:
            return
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{LLOYD_API}/api/browser/state", json=frame)
    except Exception as exc:
        logger.debug("browser state push failed: %s", exc)


async def _browser_evaluate(script: str) -> str:
    page = await _get_page()
    try:
        result = await asyncio.wait_for(page.evaluate(script), timeout=10.0)
        result_json = json.dumps(result)
        if len(result_json) > 50000:
            result_json = result_json[:50000] + "...[truncated]"
            return json.dumps({"ok": True, "result": result_json, "truncated": True})
        return json.dumps({"ok": True, "result": result})
    except asyncio.TimeoutError:
        return json.dumps({"error": "Script timed out after 10 seconds"})
    except Exception as exc:
        return json.dumps({"error": f"Evaluate failed: {exc}"})


async def _browser_fill(ref: str, value: str) -> str:
    page = await _get_page()
    try:
        loc = await _locate(page, ref)
        await loc.fill(value, timeout=10000)
        return json.dumps({"ok": True, "ref": ref})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Fill failed: {exc}"})


async def _browser_wait(condition: str, value: str = "", timeout: int = 5000) -> str:
    page = await _get_page()
    timeout_ = min(max(timeout, 100), 30000)
    try:
        if condition == "selector":
            await page.wait_for_selector(value, timeout=timeout_)
        elif condition == "text":
            await page.wait_for_function(
                f"() => document.body.innerText.includes({json.dumps(value)})",
                timeout=timeout_,
            )
        elif condition == "navigation":
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_)
        elif condition == "networkidle":
            await page.wait_for_load_state("networkidle", timeout=timeout_)
        elif condition == "timeout":
            ms = int(value) if value.isdigit() else 1000
            await asyncio.sleep(min(ms, 10000) / 1000)
        else:
            return json.dumps({
                "error": f"Unknown condition '{condition}'. Use: selector, text, navigation, networkidle, timeout"
            })
        return json.dumps({"ok": True, "condition": condition})
    except Exception as exc:
        return json.dumps({"error": f"Wait failed: {exc}"})


async def _browser_select(ref: str, value: str = "", label: str = "") -> str:
    page = await _get_page()
    try:
        loc = await _locate(page, ref)
        if label:
            await loc.select_option(label=label, timeout=5000)
        elif value:
            await loc.select_option(value=value, timeout=5000)
        else:
            return json.dumps({"error": "Provide either value or label"})
        return json.dumps({"ok": True, "ref": ref})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Select failed: {exc}"})


async def _browser_drag(source_ref: str, target_ref: str) -> str:
    page = await _get_page()
    try:
        src = await _locate(page, source_ref)
        tgt = await _locate(page, target_ref)
        await src.drag_to(tgt, timeout=10000)
        return json.dumps({"ok": True, "from": source_ref, "to": target_ref})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Drag failed: {exc}"})


async def _browser_cookies(action: str, domain: str = "", cookies: list | None = None) -> str:
    ctx = await _ensure_browser()
    try:
        if action == "get":
            result = await ctx.cookies(urls=[domain] if domain else None)
            return json.dumps({"cookies": result})
        elif action == "set":
            if not cookies:
                return json.dumps({"error": "No cookies provided"})
            await ctx.add_cookies(cookies)
            return json.dumps({"ok": True, "added": len(cookies)})
        elif action == "clear":
            await ctx.clear_cookies()
            return json.dumps({"ok": True})
        else:
            return json.dumps({"error": f"Unknown action '{action}'. Use: get, set, clear"})
    except Exception as exc:
        return json.dumps({"error": f"Cookies operation failed: {exc}"})


# ── MCP interface ──────────────────────────────────────────────────────────────

async def list_tools():
    return [
        # ── Phase 1: Core browse-read-interact loop ────────────────────────────
        Tool(name="browser_navigate", description=(
            "Navigate the browser to a URL. Returns the page title and HTTP status. "
            "Always call browser_snapshot after navigating to see the page content."
        ), inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to (http/https only)"},
                "wait_until": {
                    "type": "string",
                    "enum": ["load", "domcontentloaded", "networkidle", "commit"],
                    "description": "When to consider navigation complete. Use networkidle for SPAs. Default: domcontentloaded",
                },
            },
            "required": ["url"],
        }),
        Tool(name="browser_snapshot", description=(
            "Get the accessibility tree of the current page as structured text. "
            "Interactive elements (links, buttons, form fields) are assigned ref IDs like e1, e2, e3. "
            "Use these refs with browser_click, browser_type, etc. "
            "Refs are invalidated after each new snapshot or navigation."
        ), inputSchema={
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "Include non-interactive/static elements. Default: false (compact view)",
                },
            },
        }),
        Tool(name="browser_click", description=(
            "Click an element by its ref ID from the last browser_snapshot. "
            "Example: browser_click(ref='e3') clicks the element with ref e3."
        ), inputSchema={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Element ref ID from browser_snapshot (e.g. 'e3')"},
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button. Default: left",
                },
            },
            "required": ["ref"],
        }),
        Tool(name="browser_type", description=(
            "Type text into a form field by ref ID. "
            "Use clear=true to clear the field first. "
            "For filling large amounts of text, prefer browser_fill which triggers change events properly."
        ), inputSchema={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Element ref ID from browser_snapshot"},
                "text": {"type": "string", "description": "Text to type"},
                "clear": {"type": "boolean", "description": "Clear the field first. Default: false"},
            },
            "required": ["ref", "text"],
        }),
        Tool(name="browser_scroll", description="Scroll the current page by a pixel amount in one direction. Use browser_snapshot afterwards to see what came into view.", inputSchema={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Scroll direction. Default: down",
                },
                "amount": {"type": "integer", "description": "Pixels to scroll. Default: 300"},
            },
        }),
        Tool(name="browser_press", description=(
            "Press a key or key combination on the current page. "
            "Examples: 'Enter', 'Tab', 'Escape', 'Ctrl+a', 'Ctrl+c', 'ArrowDown'."
        ), inputSchema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key or chord (e.g. 'Enter', 'Ctrl+a')"},
            },
            "required": ["key"],
        }),
        Tool(name="browser_tabs", description=(
            "Manage browser tabs. Actions: list (show all tabs), switch (activate a tab by id), "
            "close (close a tab by id), new (open a new tab)."
        ), inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "switch", "close", "new"],
                    "description": "Tab action to perform",
                },
                "page_id": {"type": "integer", "description": "Tab ID for switch/close actions"},
                "url": {"type": "string", "description": "URL to load in new tab (for action=new)"},
            },
            "required": ["action"],
        }),
        # ── Phase 2: Screenshots + JS ──────────────────────────────────────────
        Tool(name="browser_screenshot", description=(
            "Take a screenshot of the current page. Returns a base64-encoded PNG image "
            "and saves it to logs/screenshots/. Useful for visual verification."
        ), inputSchema={
            "type": "object",
            "properties": {},
        }),
        Tool(name="browser_evaluate", description=(
            "Execute JavaScript in the current page context and return the result. "
            "Useful for extracting data that isn't in the accessibility tree, "
            "checking JS state, or manipulating the DOM directly. "
            "Timeout: 10 seconds. Return value limited to 50KB."
        ), inputSchema={
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "JavaScript expression or function body to execute"},
            },
            "required": ["script"],
        }),
        Tool(name="browser_fill", description=(
            "Fill a form field with a value using Playwright's fill() API, which properly "
            "triggers input/change events. Use for text inputs, textareas, and contenteditable elements. "
            "Replaces the entire field value."
        ), inputSchema={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Element ref ID from browser_snapshot"},
                "value": {"type": "string", "description": "Value to fill"},
            },
            "required": ["ref", "value"],
        }),
        Tool(name="browser_wait", description=(
            "Wait for a condition before continuing. "
            "Conditions: selector (CSS selector appears), text (text appears on page), "
            "navigation (page load completes), networkidle (no network requests), "
            "timeout (wait N milliseconds)."
        ), inputSchema={
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "enum": ["selector", "text", "navigation", "networkidle", "timeout"],
                    "description": "Condition to wait for",
                },
                "value": {
                    "type": "string",
                    "description": "CSS selector (for selector), text content (for text), or ms (for timeout)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max wait time in milliseconds (100-30000). Default: 5000",
                },
            },
            "required": ["condition"],
        }),
        # ── Phase 3: Polish ────────────────────────────────────────────────────
        Tool(name="browser_select", description=(
            "Select an option from a <select> dropdown by element ref. "
            "Provide either value (the option's value attribute) or label (the option's visible text)."
        ), inputSchema={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Element ref ID from browser_snapshot"},
                "value": {"type": "string", "description": "Option value attribute"},
                "label": {"type": "string", "description": "Option visible label text"},
            },
            "required": ["ref"],
        }),
        Tool(name="browser_drag", description=(
            "Drag from one element to another by ref IDs. "
            "Useful for drag-and-drop interfaces."
        ), inputSchema={
            "type": "object",
            "properties": {
                "source_ref": {"type": "string", "description": "Ref ID of the element to drag from"},
                "target_ref": {"type": "string", "description": "Ref ID of the element to drag to"},
            },
            "required": ["source_ref", "target_ref"],
        }),
        Tool(name="browser_cookies", description=(
            "Get, set, or clear cookies. "
            "Actions: get (list cookies, optionally filtered by domain URL), "
            "set (add cookies, requires cookies list), clear (remove all cookies)."
        ), inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "clear"],
                    "description": "Cookie action",
                },
                "domain": {
                    "type": "string",
                    "description": "URL to filter cookies for (action=get)",
                },
                "cookies": {
                    "type": "array",
                    "description": "List of cookie objects {name, value, domain, path} (action=set)",
                    "items": {"type": "object"},
                },
            },
            "required": ["action"],
        }),
    ]


async def call_tool(name: str, arguments: dict):
    handlers = {
        "browser_navigate": lambda: _browser_navigate(
            arguments.get("url", ""),
            arguments.get("wait_until", "domcontentloaded"),
        ),
        "browser_snapshot": lambda: _browser_snapshot(
            arguments.get("full", False),
        ),
        "browser_click": lambda: _browser_click(
            arguments.get("ref", ""),
            arguments.get("button", "left"),
        ),
        "browser_type": lambda: _browser_type(
            arguments.get("ref", ""),
            arguments.get("text", ""),
            arguments.get("clear", False),
        ),
        "browser_scroll": lambda: _browser_scroll(
            arguments.get("direction", "down"),
            arguments.get("amount", 300),
        ),
        "browser_press": lambda: _browser_press(
            arguments.get("key", ""),
        ),
        "browser_tabs": lambda: _browser_tabs(
            arguments.get("action", "list"),
            arguments.get("page_id"),
            arguments.get("url"),
        ),
        "browser_screenshot": lambda: _browser_screenshot(),
        "browser_evaluate": lambda: _browser_evaluate(
            arguments.get("script", ""),
        ),
        "browser_fill": lambda: _browser_fill(
            arguments.get("ref", ""),
            arguments.get("value", ""),
        ),
        "browser_wait": lambda: _browser_wait(
            arguments.get("condition", "timeout"),
            arguments.get("value", ""),
            arguments.get("timeout", 5000),
        ),
        "browser_select": lambda: _browser_select(
            arguments.get("ref", ""),
            arguments.get("value", ""),
            arguments.get("label", ""),
        ),
        "browser_drag": lambda: _browser_drag(
            arguments.get("source_ref", ""),
            arguments.get("target_ref", ""),
        ),
        "browser_cookies": lambda: _browser_cookies(
            arguments.get("action", "get"),
            arguments.get("domain", ""),
            arguments.get("cookies"),
        ),
    }
    handler = handlers.get(name)
    if not handler:
        return text_result(json.dumps({"error": f"Unknown tool: {name}"}))
    result = await handler()
    # Mission Control's Browser tab mirrors what the agent is looking at, so
    # every call that can change the view republishes it. Fire-and-forget on
    # purpose: the tool result is already computed, and awaiting the push
    # would charge every browser call a loopback round-trip and make the
    # frontend's uptime everyone else's problem.
    _schedule_state_push(name)
    return text_result(result)


_pending_pushes: set[asyncio.Task] = set()


def _schedule_state_push(tool_name: str) -> None:
    """Fire a state push without making the tool call wait for it.

    The task handle is kept for its lifetime — asyncio only weak-references
    tasks, so a bare create_task can be garbage-collected before it runs.
    """
    task = asyncio.create_task(_push_browser_state(tool_name))
    _pending_pushes.add(task)
    task.add_done_callback(_pending_pushes.discard)


async def shutdown() -> None:
    """Close the browser context, browser and Playwright driver.

    Called from the aggregator's lifespan (see agent_mcp/main.py). Without
    it, every `supervisorctl restart lloyd-mcp` orphaned a Chromium and its
    Playwright node driver — the process tree grew one of each per restart.
    """
    global _pw, _browser, _context
    for label, obj, closer in (
        ("context", _context, "close"),
        ("browser", _browser, "close"),
        ("playwright", _pw, "stop"),
    ):
        if obj is None:
            continue
        try:
            await getattr(obj, closer)()
        except Exception as exc:
            logger.warning("browser: %s %s() failed during shutdown: %s", label, closer, exc)
    _context = _browser = _pw = None
