#!/usr/bin/env python3
"""
Lloyd MCP Server: Browser — full browser control via Playwright.

Tools (Phase 1): browser_navigate, browser_snapshot, browser_click,
                 browser_scroll, browser_press, browser_tabs
Tools (Phase 2): browser_screenshot, browser_evaluate, browser_fill, browser_wait
Tools (Phase 3): browser_select, browser_drag, browser_cookies; browser_upload_files (#4236)

`browser_type` was merged into `browser_fill(keystrokes=true)` on 2026-09-23.
"""

import asyncio
import base64
import ipaddress
import json
import logging
import os
import re
import time
import urllib.parse
from pathlib import Path

import httpx
from mcp.types import Tool

from agent_mcp._shared import image_result, text_result
from mcp.types import CallToolResult  # noqa: E402

logger = logging.getLogger("lloyd-browser")

# ── Config ─────────────────────────────────────────────────────────────────────

CHROMIUM_EXECUTABLE = "/usr/bin/chromium"
from app.config import CONFIG, service_url  # noqa: E402
from app.paths import SCREENSHOTS_DIR, VAULT_ROOT  # noqa: E402  (was an absolute literal)
MAX_SNAPSHOT_CHARS = 8000
# Every child frame's aria tree is read within a budget of its own rather than
# sharing the document's: `MAX_SNAPSHOT_CHARS` applied to the whole snapshot
# would start cutting frames after the first, and a frame that vanishes from the
# tree is the exact defect this replaces (#424). The sections still share a
# ceiling, because a page carrying twenty ad embeds must not multiply the tool
# result twenty-fold — whatever falls outside it keeps its header and says why
# its body is missing, so the budget is never the reason content goes quiet.
MAX_FRAME_CHARS_TOTAL = MAX_SNAPSHOT_CHARS * 2
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
#
# This guard existed as a function and nothing else from 2026-04-11 (`b20a8c4`)
# to 2026-09-09: `_is_private_host` was defined, tested, named in backlog #278's
# acceptance as "the SSRF host check ... is intact", and called from nowhere.
# Every stage confirmed the previous stage's reading of a symbol at a line
# number; none asked whether anything invoked it. The lesson is in the shape of
# the test below, which now asserts the *entry points call it* rather than that
# the predicate returns the right booleans.
#
# The policy mirrors `http_tools.http_request`, which is the other tool that
# drives things over HTTP: block the network the machine is on, allow the
# machine itself. Loopback stays reachable because the agent legitimately
# browses Lloyd's own dashboard, and because it already holds Bash and the MCP
# surface — blocking the browser there removes no authority an injected prompt
# could not get more directly. What it cannot reach is your LAN: the router,
# the NAS, the printer, every unauthenticated device on 192.168/10/172.16.
#
# The check is on RESOLVED addresses, not on the hostname string. That is what
# makes it more than a speed bump: the platform resolver normalises every
# encoding that beat the old regex list (`2130706433`, `0x7f000001`,
# `::ffff:127.0.0.1` are all 127.0.0.1 to getaddrinfo, exactly as they are to
# Chromium), and it catches a name like `router.local` that a string test can
# never classify.


def _resolve_block_private() -> bool:
    """Whether to refuse private/internal hosts. Defaults to on.

    `browser.block_private_hosts: false` in config.yaml, or
    LLOYD_BROWSER_BLOCK_PRIVATE=0, turns it off; the env wins, so a one-off
    debug against a LAN device does not need a config edit. Same shape as
    `_resolve_headless` above.
    """
    raw = os.environ.get("LLOYD_BROWSER_BLOCK_PRIVATE")
    if raw is None or not raw.strip():
        configured = (CONFIG.get("browser") or {}).get("block_private_hosts")
        if configured is None:
            return True
        raw = str(configured)
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    logger.warning("browser: unparseable block_private_hosts %r — staying on", raw)
    return True


# Both names now come from `agent_mcp/egress.py`, which is where #2584 put the one
# definition. Above this comment lived the resolving answer, while `http_tools.py`
# carried a prefix match over the hostname TEXT under the same name — on the lane
# that needs no browser lease and no Playwright session. So `localhost.localdomain`,
# which resolves 127.0.0.1 and ::1 on this box, was refused by `browser_navigate`
# and *fetched* by `http_fetch`.
# The call sites below go through the module, not through imported names, so the
# lane has exactly one lookup path: a test that pins the seam by rebinding
# `egress._resolve_addrs` reaches `_is_loopback_host` here AND the predicate that
# reads it, which imported names would not both do (`from x import y` binds the
# object, so the local name keeps pointing at the old one). Behaviour for this lane
# is unchanged, including the empty host — `_host_block_reason` asks the scheme
# first, so `about:blank` stays snapshot-able.
from agent_mcp import egress  # noqa: E402


def _is_loopback_host(hostname: str) -> bool:
    """The machine itself, by any of its names.

    NOT the same rule as `http_tools._is_loopback_host`, and #2584 is why this
    sentence says so out loud: that one matches the TEXT of the hostname and
    recognises exactly `localhost`, `127.*` and `::1`, while this one asks the
    resolver and exempts a name only when EVERY address it answers with is
    loopback. `localhost.localdomain` is therefore the machine here and is refused
    as private there. Both are exemptions, and the split is deliberate: this lane
    decides whether Chromium may open the box's own UI, whereas that one is
    `http_request`'s local-service *allowance*, and widening it to every name that
    resolves to loopback is a ruling owed on #2584 rather than something decided on
    the way through a browser-lane fix. `127.0.0.1` and `localhost` stay the same
    host under either reading — that half was never in dispute.
    """
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    addrs = egress._resolve_addrs(hostname)
    if not addrs:
        return False
    resolved = []
    for addr in addrs:
        try:
            resolved.append(ipaddress.ip_address(addr))
        except ValueError:
            return False
    # Every address must be loopback. A name that resolves to both loopback
    # and something else is not "the machine itself" in any useful sense, and
    # treating it as such is how a rebinding trick would buy the exemption.
    return bool(resolved) and all(ip.is_loopback for ip in resolved)


def _host_block_reason(url: str) -> str | None:
    """The one decision. Returns a message to refuse with, or None to allow.

    Every place that can put a URL in front of Chromium calls this, and so
    does the connection gate, which is what covers the three lanes an
    entry-point check cannot see: a redirect to a private address, a click on
    a link, and a subresource fetched by a page that is itself public.
    """
    if not _resolve_block_private():
        return None
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return f"Blocked — unparseable URL: {url}"
    # Only http(s) names a host on the network. `about:blank` has no hostname
    # at all, and treating an empty host as private made this guard block the
    # blank page `_enforce_landing` itself navigates to — so one refusal
    # poisoned every later snapshot with `private/internal host ""`, and a
    # freshly launched browser could not be snapshotted before its first
    # navigation. `data:` and `blob:` are the same shape. Non-http schemes are
    # out of scope here rather than allowed on purpose: navigation is already
    # restricted to http/https at each entry point.
    if parsed.scheme not in ("http", "https"):
        return None
    hostname = parsed.hostname or ""
    if _is_loopback_host(hostname):
        return None
    if egress._is_private_host(hostname):
        return f'Blocked — private/internal host "{hostname}"'
    return None


# The most recent refusal, so a navigation killed by the connection gate can say
# why. Aborting a redirect leaves `page.goto` waiting for a load that will
# never arrive, and it surfaces as a bare 30s timeout — a message that sends
# the agent looking for a slow site instead of telling it the hop was refused.
_block_log: dict = {"seq": 0, "url": "", "reason": "", "ts": 0.0}


def _record_block(url: str, reason: str) -> None:
    _block_log["seq"] += 1
    _block_log["url"] = url
    _block_log["reason"] = reason
    _block_log["ts"] = time.time()


def _block_since(seq: int) -> str | None:
    """The reason recorded since `seq`, if the guard fired during a call."""
    if _block_log["seq"] > seq and _block_log["reason"]:
        return f'{_block_log["reason"]} (redirected to {_block_log["url"][:120]})'
    return None


async def _prewarm_host(url: str) -> None:
    """Resolve `url`'s host off the event loop, into the cache the sync check reads.

    `socket.getaddrinfo` blocks, and the two hot callers are coroutines on the
    aggregator's loop — the same loop that dispatches every MCP tool call.
    A literal IP costs microseconds because it never leaves the resolver, but
    a cold hostname is a network round trip, and the connection gate sees
    one per unique host on every page. `egress._resolve_addrs` is lru_cached, so
    warming it here makes the `_host_block_reason` call that follows free.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return
        host = parsed.hostname or ""
        if host and host.lower() != "localhost":
            await asyncio.to_thread(egress._resolve_addrs, host)
    except Exception:
        # A failure here costs a blocking lookup in the check below, never
        # correctness — the sync path resolves for itself if the cache missed.
        pass


async def _host_block_reason_async(url: str) -> str | None:
    """`_host_block_reason` with the DNS moved off the loop."""
    if not _resolve_block_private():
        return None
    await _prewarm_host(url)
    return _host_block_reason(url)


def _guard_connect(host: str, port: int, url: str = "") -> tuple:
    """The connection gate. `_browser_proxy.GuardProxy` asks it before dialling.

    Returns `(reason, ())` to refuse, or `(None, addresses)` to allow — and the
    proxy then dials one of THOSE addresses, so the address judged is the one
    reached. Covers every connection Chromium makes: a navigation, a clicked
    link, a subresource, a websocket, and a redirect's next hop, which the
    route interceptor this replaced never re-entered for. `_enforce_landing`
    below still asks where the page ended up; it is now the second answer to
    that question instead of the only one.

    It was an interceptor (`context.route`) until 2026-11-17. Any interceptor
    makes Chromium ship every request BODY over the automation pipe before the
    request may leave: a 600 MiB upload fell from 771 MiB/s to 1-15 MiB/s, and
    a 3.48 GB YouTube upload was projected at 2.5 hours. Measurements are in
    `agent_mcp/_browser_proxy.py`. Never put a `context.route` back.

    Called on the proxy's executor thread, so the blocking resolver is fine
    here. A raise is the proxy's to handle: it fails OPEN with a warning, the
    interceptor's own trade — a bug here should cost depth, not the browser.
    An unresolvable name returns no addresses and the proxy answers 502.
    """
    bare = f"[{host}]" if ":" in host else host
    target = url or f"http://{bare}:{port}/"
    reason = _host_block_reason(f"http://{bare}:{port}/")
    if reason:
        logger.warning("browser: %s (%s)", reason, target[:120])
        _record_block(target, reason)
        return reason, ()
    if host.lower() == "localhost":
        return None, ("127.0.0.1", "::1")
    return None, egress._resolve_addrs(host)


# The word both fail-closed checks put at the front of their refusal. A caller
# can test it with `_is_guard_fault` instead of pattern-matching prose, which is
# what lets `browser_navigate` keep its "(redirected from …)" suffix for a real
# block and drop it for a fault.
_GUARD_FAULT = "Guard fault —"


def _guard_fault(which: str, url: str, exc: BaseException) -> str:
    """The refusal of a host check that raised instead of answering.

    `which` names the check (`landing`, `frame`), so the agent can tell whether
    the page or one document inside it is the thing nobody vouched for. Both the
    exception's type *and* its message are spelled out because `str(exc)` is
    routinely empty — the detached frame tree raises `OSError()`, whose message
    alone renders as `(OSError: )` and leaves an agent nothing to act on. The
    message is deliberately not a `Blocked —` sentence: those two words mean
    "this host is private, and the guard is working", and merging the two would
    make a broken guard indistinguishable from a doing-its-job one.
    """
    return (f"{_GUARD_FAULT} the {which} host check could not answer for "
            f"{url[:200]} ({type(exc).__name__}: {str(exc)[:200]}) — "
            f"refusing to read the page")


def _is_guard_fault(reason: str) -> bool:
    """Is this answer from `_enforce_landing` a fault rather than a block?"""
    return reason.startswith(_GUARD_FAULT)


async def _enforce_landing(page) -> str | None:
    """Where the page actually ended up, whatever route it took there.

    The last line, and the one that does not depend on enumerating lanes. A
    server-side redirect defeats the interceptor above; a meta-refresh, a
    `window.location` in a script and a form POST all do too. Rather than
    chase each one, every path that could put a private page in front of the
    model asks this, so content from a private host cannot be read, snapshotted
    or screenshotted no matter how the browser got there.

    Blanks the page on a hit. Leaving it parked on the LAN device would mean
    the very next `browser_snapshot` reads it, and the state mirror would push
    a screenshot of it to Mission Control.

    **Refuses when it cannot answer.** `None` means the check ran and said the
    host is fine, so a swallowed exception used to be read as exactly that — and
    `browser_snapshot` and `browser_evaluate` then served the page they had just
    failed to check. A fault now returns a `_GUARD_FAULT` string, so the three
    callers — which refuse on any non-None answer — return an error instead of
    content. It neither blanks nor records: the check never learned where the
    page is, so navigating it somewhere on the strength of that guess is a second
    guess, and recording it would report a redirect to a private host that was
    never observed. `_guard_connect` stays the only deliberate fail-open in this
    file, because it sits *behind* these checks; this one is the entry point.
    """
    url = "<unknown>"
    try:
        url = page.url
        reason = await _host_block_reason_async(url)
    except Exception as exc:
        logger.warning("browser: landing guard failed closed on %s: %s",
                       url[:120], f"{type(exc).__name__}: {exc}")
        return _guard_fault("landing", url, exc)
    if not reason:
        return None
    _record_block(page.url, reason)
    try:
        await page.goto("about:blank", wait_until="domcontentloaded", timeout=5000)
    except Exception:
        pass
    _ref_map.clear()
    return reason


# ── Browser state (persistent for server lifetime) ─────────────────────────────

_pw = None        # Playwright instance
_browser = None   # Browser instance
_context = None   # BrowserContext
_active_page = None  # Currently focused Page
_ref_map: dict[str, dict] = {}  # "e1" -> {"role": ..., "name": ..., "occurrence": ...}
# Which document the last tool-driven navigation actually landed on, and what
# the server said about it. `browser_snapshot` cannot otherwise tell a 404
# wearing normal site chrome from the page it was fetched for: the accessibility
# tree of both looks clean. Bound to the page *and* its URL rather than kept as
# a bare scalar, because both move under it — a tab switch changes the page, a
# clicked link changes the document — and a label that outlived the navigation
# that earned it would brand a healthy page an error page. One page is held at a
# time: every navigation overwrites the row, so a closed page is dropped at the
# next one rather than held open.
_nav_doc: dict = {"page": None, "url": "", "status": 0}
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
        proxy={"server": _guard_proxy_url()},
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
    # The private-host guard is the proxy above, not an interceptor: it is asked
    # per connection, so a tab opened by a target=_blank click is covered, and
    # the flag is read per connection, so a config change needs no relaunch.
    # No `context.route` here — see `_guard_connect` for what one costs.
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


def _remember_doc_status(page, status: int) -> None:
    """Record what the server said about the document `page` is showing now."""
    _nav_doc.update(page=page, url=page.url, status=int(status))


def _forget_doc_status() -> None:
    """Drop the record. Every path that does not end on a fetched document
    takes this, so a label can only ever describe a navigation that happened."""
    _nav_doc.update(page=None, url="", status=0)


def _doc_status_for(page) -> int:
    """The document status of what `page` is showing right now; 0 when unknown.

    Answers 0 — no label — whenever the page or its URL has moved since the
    navigation that recorded it, because the recorded status then describes a
    different document. 0 also covers the handle-less goto, where nothing was
    measured; guessing there would be the same invented verdict #1087 is about.
    """
    if _nav_doc["page"] is page and _nav_doc["url"] == page.url:
        return int(_nav_doc["status"] or 0)
    return 0


# ── Accessibility tree snapshot ────────────────────────────────────────────────

def _parse_aria_snapshot(
    text: str,
    ref_map: dict,
    frame_index: int | None = None,
    first_ref_number: int = 0,
) -> tuple[str, int]:
    """
    Parse Playwright's aria_snapshot() YAML-like output.
    Injects ref IDs (e1, e2, ...) next to interactive elements.
    Returns (annotated_text, last_ref_number).

    `frame_index` tags every ref this call issues with the frame its tree came
    from, so `_locate` can resolve it there instead of against the top frame —
    `page.get_by_role` is scoped to the main frame, so before #424 an in-frame
    control was unreachable however its ref was spelled. None means the main
    frame, which is the shape every ref carried before frames existed and the
    one the Mission Control frame's ref list still renders.

    `first_ref_number` continues the numbering across calls: one snapshot hands
    out one sequence of ids across all its frames, or the second frame's `e1`
    would silently overwrite the top frame's.
    """
    lines = text.split("\n")
    result_lines = []
    counter = first_ref_number
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
            entry = {"role": role, "name": name, "occurrence": occ}
            if frame_index is not None:
                entry["frame"] = frame_index
            ref_map[ref_id] = entry
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


def _page_frames(page) -> list:
    """The page's frames in the order `frame_index` addresses them.

    Playwright indexes the main frame into this list at 0, so the numbering is
    the same one the snapshot headers print and the one `browser_evaluate`
    accepts — three surfaces, one address space.
    """
    try:
        return list(page.frames)
    except Exception:
        return []


def _frame_at(page, frame_index):
    """The frame `frame_index` addresses.

    Raises ValueError naming the valid range: an off-by-one against a frame list
    one shorter than the agent assumed has to say what the range is, or the next
    call guesses again.
    """
    frames = _page_frames(page)
    try:
        idx = int(frame_index)
    except (TypeError, ValueError):
        raise ValueError(
            f"frame_index must be an integer, got {frame_index!r}"
        ) from None
    if not 0 <= idx < len(frames):
        valid = f"0 to {len(frames) - 1}" if frames else "none (no frames)"
        raise ValueError(
            f"frame_index {frame_index} is out of range: this page has "
            f"{len(frames)} frame(s), valid frame_index is {valid}. "
            "Omit frame_index to evaluate in the main frame."
        )
    return frames[idx]


async def _locate(page, ref_id: str):
    """Reconstruct a Playwright locator from a ref ID.

    A ref tagged with a frame resolves inside that frame; an untagged one keeps
    resolving against `page`, which is the main frame and every ref this tool
    handed out before frames were walked. The frame is re-resolved by index
    rather than cached because a frame can detach between the snapshot and the
    click — an iframe the page removed, a cross-origin swap — and then the index
    is stale, which has to be said instead of clicking the wrong document.
    """
    info = _ref_map.get(ref_id)
    if not info:
        raise ValueError(
            f"Unknown ref '{ref_id}'. Call browser_snapshot to get fresh refs."
        )
    scope = page
    frame_index = info.get("frame")
    if frame_index is not None:
        frames = _page_frames(page)
        if not 0 <= frame_index < len(frames):
            valid = f"0 to {len(frames) - 1}" if frames else "none (no frames)"
            raise ValueError(
                f"Ref '{ref_id}' pointed into frame {frame_index}, which is gone; "
                f"this page now has {len(frames)} frame(s), valid frame_index is "
                f"{valid}. Call browser_snapshot to get fresh refs."
            )
        scope = frames[frame_index]
    return scope.get_by_role(info["role"], name=info["name"], exact=True).nth(
        info["occurrence"]
    )


# ── Tool implementations ───────────────────────────────────────────────────────

# Playwright's `networkidle` wants ~500 ms with no network connections at all. A
# page holding a live socket — an EventSource, a websocket, Vite's HMR channel, a
# long-poll — never grants that silence, so the wait runs to this tool's own 30 s
# deadline on a document that is fully rendered. The raw
# `Page.goto: Timeout 30000ms exceeded.` then reads, to the model and to the
# harness (`text_result` turns a top-level `error` key into `isError`), exactly
# like a dead site. Backlog #1088: when a usable document is sitting there, the
# honest answer is the one the page gave, plus the network declared non-idle.
def _looks_like_timeout(exc: BaseException) -> bool:
    """Playwright's TimeoutError, or its text after a re-raise."""
    if "timeout" in type(exc).__name__.lower():
        return True
    text = str(exc).lower()
    return "timeout" in text and "exceeded" in text


def _hosts_match(requested: str, landed: str) -> bool:
    """Same machine, allowing the redirects that commonly change the URL.

    A page only counts as loaded if it is the page that was asked for. `page.url`
    survives a navigation that never committed as the *previous* document's URL,
    and calling that the target is how a dead host would read as a live one. An
    apex-to-`www.` hop and an http→https upgrade are the ordinary redirects a
    real navigation takes, so those still count as the same host.
    """
    try:
        want = urllib.parse.urlparse(requested).hostname or ""
        got = urllib.parse.urlparse(landed).hostname or ""
    except ValueError:
        return False
    if not want or not got:
        return False
    return want.removeprefix("www.") == got.removeprefix("www.")


async def _document_is_usable(page, requested: str = "") -> bool:
    """Is there a real document behind a wait that never finished?

    An empty URL, `chrome-error://chromewebdata/` and the blank page a freshly
    launched browser sits on are all Chromium's answer to "nothing loaded", and
    all three are free of an http scheme, which is the test. The title is read
    here because the caller is about to hand it back; a page that cannot answer
    that is not a page.
    """
    try:
        landed = str(page.url or "")
    except Exception:
        return False
    if not landed.startswith(("http://", "https://")):
        return False
    if requested and not _hosts_match(requested, landed):
        return False
    try:
        return isinstance(await page.title(), str)
    except Exception:
        return False


async def _browser_navigate(url: str, wait_until: str = "domcontentloaded") -> str:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return json.dumps({"error": f"Invalid URL: {url}"})
    if parsed.scheme not in ("http", "https"):
        return json.dumps({"error": "Only http/https URLs are supported"})

    if wait_until not in ("load", "domcontentloaded", "networkidle", "commit"):
        wait_until = "domcontentloaded"

    # #628: the destination decision, before `_get_page()` — so a refused
    # navigation never launches Chromium, and the refusal is the grant shape
    # rather than the 30s timeout a blocked route would have surfaced as. The
    # private-host checks below it (`_host_block_reason`, `_guard_connect`,
    # `_enforce_landing`) are untouched: this adds a destination axis, it does
    # not replace the address floor, and the floor runs first so an allow-list
    # entry naming a private address cannot reopen it.
    #
    # No `loopback_ok` here, deliberately: this lane has never had a loopback
    # allowance (`test_navigate_to_a_link_local_address_is_blocked` pins that
    # `http://169.254.1.1/` is refused, and the comment in
    # `_is_local_dev_loopback_host` explains why the browser is not given the
    # one `http_request` has), so the guard's floor refuses the same addresses it
    # always did. `floor_off` mirrors this module's own `block_private_hosts`
    # knob (`browser.block_private_hosts: false` exists for a local-only
    # browser); a guard that reinstated what the operator switched off would be
    # a behavior change wearing a security badge.
    from agent_mcp import egress
    # Off the loop: the guard writes a row and may read a session file, and
    # this coroutine runs on the aggregator's one event loop.
    verdict = await egress.aguard("browser_navigate", url,
                                  floor_off=not _resolve_block_private())
    if not verdict.allowed:
        return json.dumps({"error": verdict.reason})

    blocked = await _host_block_reason_async(url)
    if blocked:
        return json.dumps({"error": blocked})

    page = await _get_page()
    _ref_map.clear()
    # Until a document is fetched, nothing here knows what this page is.
    _forget_doc_status()
    seq = _block_log["seq"]
    never_idle = False
    try:
        resp = await page.goto(url, wait_until=wait_until, timeout=30000)
    except Exception as exc:
        # A redirect the guard refused arrives here as the same bare timeout —
        # the aborted hop leaves `goto` waiting for a load that never comes — so
        # the block reason is asked for first, and it wins. What remains, a
        # timeout on the `networkidle` wait over a document that is really
        # there, is not a failed navigation.
        blocked = _block_since(seq)
        loaded_never_idle = (
            not blocked
            and wait_until == "networkidle"
            and _looks_like_timeout(exc)
            # Last, because it costs a round trip to the page and only a real
            # `networkidle` timeout can make that question worth asking.
            and await _document_is_usable(page, url)
        )
        if not loaded_never_idle:
            return json.dumps({"error": blocked or str(exc)})
        resp, never_idle = None, True
    try:
        landed = await _enforce_landing(page)
        if landed:
            # The suffix is a claim about what happened, so it goes only with the
            # answer that earned it. A real block means the page ended up on a
            # private host, which does mean it was redirected; a fault means the
            # check could not tell us anything, and "(redirected from …)" would
            # send the agent hunting a redirect that was never observed.
            if _is_guard_fault(landed):
                return json.dumps({"error": landed})
            return json.dumps({"error": f"{landed} (redirected from {url[:100]})"})
        # `resp` is None when Playwright hands back no response handle — a
        # same-document navigation among them — so the status there is
        # *unmeasured*, not a failure. Testing the handle as well as the code is
        # what keeps those navigations reading as the success they are.
        status = resp.status if resp is not None else 0
        title = await page.title()
        result = {
            "ok": True,
            "url": page.url,
            "title": title,
            "status": status,
        }
        if resp is not None and status >= 400:
            # A 4xx/5xx document used to come back `ok: true` with the code in
            # a field next to it, leaving the model to notice and interpret it
            # while `browser_snapshot` served the error page as the content.
            result["ok"] = False
            # `warning`, never `error`: `BrowserPage.tsx` turns any `error`
            # field into a red banner over the URL bar, and `navigate_from_ui`
            # keys its https→http fallback on one. A human who asked to see a
            # 404 wants the 404 on screen; a fallback retry on a page that
            # answered is a second navigation bought with no new information.
            result["warning"] = (
                f"HTTP {status} — {title or '(untitled page)'}. The document that "
                "answered is an error page, not the target content: its title and "
                "url are the error page's own."
            )
        if never_idle:
            # The document is there; the wait for quiet never ended, and with it
            # the response handle — so `status` here is unmeasured, exactly as it
            # is for a same-document navigation, rather than evidence of anything.
            result["network_idle"] = False
            result["note"] = (
                "Loaded, but the network never went idle: a live websocket, "
                "EventSource or HMR connection holds it open, so the "
                "networkidle wait ran to its deadline. The content is usable; "
                "the HTTP status was not measured. Pass wait_until=\"load\" "
                "next time and wait for the content with browser_wait."
            )
        _remember_doc_status(page, status)
        return json.dumps(result)
    except Exception as exc:
        # Only on failure: a block during a navigation that still loaded was a
        # subresource, and reporting that as the navigation's error would turn
        # a working page into a refusal.
        return json.dumps({"error": _block_since(seq) or str(exc)})


# ── Child frames ───────────────────────────────────────────────────────────────

def _frame_entries(page) -> list[tuple[int, object]]:
    """Every frame on the page except the main one, each with its index.

    The main frame is the document the top-level tree came from, so printing it
    again would duplicate it; its index is skipped rather than the list being
    renumbered, because the number a header prints has to stay the number
    `browser_evaluate(frame_index=…)` takes for that same frame.
    """
    main = getattr(page, "main_frame", None)
    return [(i, f) for i, f in enumerate(_page_frames(page)) if f is not main]


def _frame_header(index: int, url: str) -> str:
    return f"### frame {index} {url or '(no url)'}"


async def _frame_sections(page, ref_map: dict, first_ref_number: int = 0) -> str:
    """Each child frame's aria tree under a `### frame <n> <url>` header.

    Returns "" when the page has no child frames, so a page without an iframe
    renders exactly as it did before this existed.

    Every frame gets a header whether or not its body arrives, and a body that did
    not arrive says which of the five reasons it did not: the host guard refused the
    frame's host, the guard could not finish and the read was skipped for that
    reason, the frame budget ran out, the aria read raised, or the frame reported
    nothing. That is the point of the whole function. Before #424 the only sign a
    frame existed was a bare `- iframe` node with no URL and no children, so
    content that lives in an embed — an editor pane, a payment widget, a video
    transcript — was missing from the snapshot with nothing marking the hole, and a
    partial extraction was reported as a complete one.

    Refs found inside a frame continue the numbering from `first_ref_number` and
    are tagged with the frame, so a later `browser_click` acts inside it.
    """
    entries = _frame_entries(page)
    if not entries:
        return ""
    out: list[str] = []
    next_ref = first_ref_number
    remaining = MAX_FRAME_CHARS_TOTAL
    for index, frame in entries:
        url = getattr(frame, "url", "") or ""
        header = _frame_header(index, url)
        # A frame is a document of its own and can name a host the page-level
        # guard refused. Same rule at the same place: content from a private
        # host stops here rather than entering the model's context.
        # Fail closed, the habit the page-level read has kept since #3294:
        # `_enforce_landing` and the per-frame check in `_browser_evaluate` refuse
        # on the same fault. A frame has nothing else to ask — the frame *is* the
        # document — so a host check that could not finish is a
        # reason not to read, not a reason to hope.
        try:
            blocked = await _host_block_reason_async(url)
        except Exception as exc:
            out.append(
                f"{header}\n[not read: the frame's host could not be checked "
                f"({type(exc).__name__})]"
            )
            continue
        if blocked:
            out.append(f"{header}\n[not read: {blocked}]")
            continue
        if remaining <= 0:
            out.append(
                f"{header}\n[not included: the {MAX_FRAME_CHARS_TOTAL}-character "
                "frame budget is spent on earlier frames]"
            )
            continue
        try:
            raw = await _aria_snapshot_masked(frame)
        except Exception as exc:
            out.append(f"{header}\n[unreadable: {exc}]")
            continue
        if not raw or not raw.strip():
            out.append(f"{header}\n[empty: the frame reported no accessible content]")
            continue
        budget = min(MAX_SNAPSHOT_CHARS, remaining)
        annotated, next_ref = _parse_aria_snapshot(
            raw, ref_map, frame_index=index, first_ref_number=next_ref)
        total = len(annotated)
        if total > budget:
            out.append(
                f"{header}\n{annotated[:budget]}\n"
                f"[partly included: truncated at {budget} of {total} chars, the "
                "rest of this frame is not shown]"
            )
        else:
            out.append(f"{header}\n{annotated}")
        remaining -= min(total, budget)
    return "\n\n" + "\n\n".join(out)


async def _browser_snapshot(full: bool = False) -> str:
    global _ref_map
    page = await _get_page()
    # Whatever route the browser took to a private host, its content stops
    # here rather than entering the model's context.
    landed = await _enforce_landing(page)
    if landed:
        return json.dumps({"error": landed})
    _ref_map = {}
    try:
        raw = await _aria_snapshot_masked(page)
    except Exception as exc:
        return json.dumps({"error": f"Accessibility snapshot failed: {exc}"})

    if raw and raw.strip():
        annotated, ref_count = _parse_aria_snapshot(raw, _ref_map)
    else:
        # An empty top frame is not a reason to report nothing: a page whose body
        # is an iframe host has all of its content one level down, and saying
        # "page may not have loaded" about it would be the wrong verdict.
        annotated, ref_count = "", 0
    frames_text = await _frame_sections(page, _ref_map, ref_count)
    if not annotated and not frames_text:
        return json.dumps({"error": "Empty accessibility tree — page may not have loaded."})

    title = await page.title()
    header = f"[Page] {title} — {page.url}"
    doc_status = _doc_status_for(page)
    if doc_status >= 400:
        # The tree cannot show that it is an error page — a moved-docs 404 comes
        # with the real site's banner and nav links, which snapshots completely
        # clean. It goes first so it survives the truncation below.
        header = (
            f"[HTTP {doc_status}] This document is an error page, "
            f"not the target content.\n{header}"
        )
    top_text = f"{header}\n{annotated}"

    if len(top_text) > MAX_SNAPSHOT_CHARS:
        # The budget belongs to the top frame, not to the whole document.
        # Applying it to the joined text would start cutting the frames below it
        # — the same silent omission this change exists to remove. A frame either
        # arrives inside its own budget or is named with the reason it did not, so
        # the result is bounded by MAX_SNAPSHOT_CHARS + MAX_FRAME_CHARS_TOTAL and
        # never needs a second cut.
        top_text = (
            top_text[:MAX_SNAPSHOT_CHARS]
            + f"\n\n[...truncated at {MAX_SNAPSHOT_CHARS} chars]"
        )
    full_text = top_text + frames_text

    return json.dumps({"snapshot": full_text, "refs": len(_ref_map)})


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
    # Validated before `_ensure_browser`, or a refused URL still launches
    # Chromium on its way to being told no.
    if url:
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            return json.dumps({"error": f"Invalid URL: {url}"})
        if parsed.scheme not in ("http", "https"):
            return json.dumps({"error": "Only http/https URLs supported"})
        blocked = await _host_block_reason_async(url)
        if blocked:
            return json.dumps({"error": blocked})

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
        # url was scheme- and host-checked at the top, before the launch.
        if len(pages) >= MAX_TABS:
            await pages[0].close()
        new_page = await ctx.new_page()
        if url:
            await new_page.goto(url, wait_until="domcontentloaded", timeout=30000)
        _active_page = new_page
        _ref_map.clear()
        return json.dumps({"ok": True, "id": len(ctx.pages) - 1, "url": new_page.url})

    return json.dumps({"error": f"Unknown action '{action}'. Use: list, switch, close, new"})


async def _browser_screenshot() -> "str | CallToolResult":
    """The page as an MCP image, plus a short JSON text block.

    It used to return the PNG as `data_base64` inside the JSON text, which the
    harness then fed to the model as ~100 KB of base64 characters and wrote
    into the session file. The harness now carries images as images
    (app/harness/tool_images.py) and decides per model whether it sees them.
    """
    global SCREENSHOTS_DIR
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    page = await _get_page()
    try:
        ts = int(time.time())
        path = SCREENSHOTS_DIR / f"screenshot_{ts}.png"
        await page.screenshot(path=str(path), full_page=False)
        data = path.read_bytes()
        vp = page.viewport_size or {}
        return image_result(json.dumps({
            "ok": True, "path": str(path), "size_bytes": len(data),
            "width": vp.get("width"), "height": vp.get("height"),
            "url": page.url,
        }), data)
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
    was closed is normal, not an error worth pushing. Returns None when the
    host check cannot answer either (#3043: a guard that raises drops the
    frame rather than pushing it — see the comment at the check).

    JPEG rather than PNG because this travels on every single tool call: the
    same 1280x800 frame measured ~190 KB as PNG against ~15-25 KB here, and
    the frame is only ever displayed scaled down in the browser tab.
    """
    page = _existing_page()
    if page is None:
        return None
    # A frame is a screenshot plus an accessibility tree; both are page
    # content, and the tab is a surface a human reads. Same rule as snapshot.
    #
    # Fails CLOSED on an internal error, which is the opposite of
    # `_guard_connect` above — and deliberately so. That layer is defence in
    # depth sitting behind the entry-point checks, so its failure mode should
    # be losing depth; this call *is* the entry-point check for the
    # screenshot path, the only thing between a LAN device's UI and the
    # Browser tab. `_frame_sections` decides the same question closed for a
    # frame of this same push. Losing one frame costs a stale thumbnail until
    # the next tool call; pushing one that was never checked cannot be
    # un-pushed, and would carry no mark saying the guard had not answered.
    url = "<unknown>"
    try:
        url = page.url
        blocked = await _host_block_reason_async(url)
    except Exception as exc:
        # `url` is still the sentinel when reading it was itself the raiser —
        # a page mid-teardown that cannot name its own host is exactly the
        # case this drops, so the sentinel is the honest thing to log. The
        # second field is the proxy's fail-open `%s` slot carrying the type as well
        # as `str(exc)`: a guard exception built with no message — `OSError()`
        # from a teardown, `KeyError` from a resolver bug — would otherwise
        # render a line naming nothing, and naming nothing is the gap this
        # record exists to close.
        logger.warning("browser: state guard failed closed on %s: %s",
                       url[:120], f"{type(exc).__name__}: {exc}")
        return None
    if blocked:
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
    top_refs = 0
    try:
        raw = await _aria_snapshot_masked(page)
        snapshot, top_refs = _parse_aria_snapshot(raw, refs)
    except Exception as exc:
        logger.debug("browser state: aria_snapshot failed: %s", exc)
    # The tab had the same blind spot the tool call had, and a human is the one
    # reading it: an embed's content never reached the tree at all. The frame
    # block carries its own reasons for anything it could not include, so a
    # partial view says so here too.
    try:
        frames_text = await _frame_sections(page, refs, top_refs)
    except Exception as exc:
        logger.debug("browser state: frame walk failed: %s", exc)
        frames_text = ""
    snapshot = snapshot[:MAX_SNAPSHOT_CHARS] + frames_text
    return {
        "tool": tool_name,
        "url": page.url,
        "title": title,
        "ts": time.time(),
        "mime": "image/jpeg",
        "screenshot_b64": base64.b64encode(shot).decode(),
        "snapshot": snapshot,
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
        _ensure_live(_existing_page())
    except Exception as exc:
        logger.debug("browser state push failed: %s", exc)


async def _browser_evaluate(script: str, frame_index: int | None = None) -> str:
    """Run `script` and return its value — in a named frame if asked.

    Omitting `frame_index` is unchanged: the script runs in the main frame's JS
    world, which is where page-level state lives. Passing one runs it inside
    `page.frames[frame_index]`, which is the capability that was missing —
    same-origin prose can be reached from the top frame through
    `contentDocument`, but a frame's own JS world and any cross-origin frame
    cannot be read that way at all, and until now nothing in this tool surface
    could address them.
    """
    page = await _get_page()
    landed = await _enforce_landing(page)
    if landed:
        return json.dumps({"error": landed})
    try:
        if frame_index is None:
            target = page
        else:
            frame = _frame_at(page, frame_index)
            # The frame is a document with a host of its own: the page passed the
            # landing guard, the frame has not necessarily.
            # Same rule as the landing check: an exception here used to leave
            # `blocked = None`, which is the value a *clean* host produces, so a
            # guard bug handed `target.evaluate` a frame nobody had checked —
            # and `_frame_sections`, one function over, was already refusing an
            # unchecked frame. An `about:blank` or empty url is not this case:
            # the predicate answers None for those, so a child frame the browser
            # created keeps working.
            url = "<unknown>"
            try:
                url = frame.url or ""
                blocked = await _host_block_reason_async(url)
            except Exception as exc:
                logger.warning("browser: frame guard failed closed on %s: %s",
                               url[:120], f"{type(exc).__name__}: {exc}")
                return json.dumps({"error": _guard_fault("frame", url, exc)})
            if blocked:
                return json.dumps({"error": blocked})
            target = frame
        result = await asyncio.wait_for(target.evaluate(script), timeout=10.0)
        result_json = json.dumps(result)
        if len(result_json) > 50000:
            result_json = result_json[:50000] + "...[truncated]"
            return json.dumps({"ok": True, "result": result_json, "truncated": True})
        return json.dumps({"ok": True, "result": result})
    except asyncio.TimeoutError:
        return json.dumps({"error": "Script timed out after 10 seconds"})
    except ValueError as exc:
        # A bad frame index or a bad type: the message already names the valid
        # range, which is the only thing that makes the next call right.
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Evaluate failed: {exc}"})


async def _browser_fill(ref: str, value: str, keystrokes: bool = False) -> str:
    """Set a field's value, in one write or key by key.

    `keystrokes` absorbed `browser_type` (2026-09-23). A one-shot `fill()` fires
    input/change but no per-key events, and a search-as-you-type box or an
    autocomplete widget listens for keys, so it clears the field and presses
    each character instead. Either way the field ends up holding `value` alone.
    """
    page = await _get_page()
    try:
        loc = await _locate(page, ref)
        if keystrokes:
            await loc.clear(timeout=5000)
            await loc.press_sequentially(value, delay=30)
        else:
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
            try:
                await page.wait_for_load_state("networkidle", timeout=timeout_)
            except Exception as exc:
                # The same trap as `browser_navigate`, reached the same way: a
                # page holding a live socket never goes idle, and the useful
                # answer is that the document is there and the network is not
                # quiet. Anything else — a page on `about:blank`, a load that
                # really did fail — stays the failure it reports as.
                if not (_looks_like_timeout(exc) and await _document_is_usable(page)):
                    return json.dumps({"error": f"Wait failed: {exc}"})
                return json.dumps({
                    "ok": True,
                    "condition": condition,
                    "network_idle": False,
                    "note": "Load state never reached networkidle: a live "
                            "websocket, EventSource or HMR connection holds the "
                            "network open. The page itself is loaded.",
                })
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


# ── File-attach policy (#4236) ────────────────────────────────────────────────
#
# `browser_upload_files` is the one browser verb that reads a file off this
# machine and hands its bytes to whatever page happens to be open. That makes
# it a file-read-and-exfiltrate primitive wearing a form-filling hat: every
# other browser tool can only put INTO a page bytes the caller already had,
# while this one can put a private key there. So the policy is default-deny —
# a path must fall inside `browser.upload_roots` to be attachable at all, and
# the roots are narrow by default rather than wide with exceptions.
#
# The defaults cover the two places a legitimate upload comes from: Lloyd's own
# runtime data (a generated artifact, a render, a recording awaiting publish)
# and the browser's Downloads folder. An EMPTY list means those defaults, never
# "everything" — clearing a list in a config file must not be the widest the
# policy has ever been.
#
# The hard refusals sit ABOVE the roots and are not configurable, because on
# those paths "the operator allowed it" is not a meaningful answer: the
# credential read-deny and write-deny sets (`app.harness.protected_paths`) and
# the vault. Dotfiles and dot-directories go with them — the credential layout
# of a Unix home is entirely dotted (`.ssh`, `.aws`, `.config/gh`,
# `.openclaw`), so one component rule covers a family no per-directory list
# would ever catch up to.
#
# Symlinks are resolved BEFORE any comparison, so a link inside an allowed root
# that points at `/etc/passwd` is judged as `/etc/passwd` — the roots answer
# where the bytes come from, not what the path is spelled like.
#
# And the whole evaluation is fail-CLOSED. That is the opposite of the effect
# ledger's fail-open rule, on purpose: a ledger that cannot write should not
# block a send, whereas a policy that cannot be evaluated has no idea where the
# bytes are going, and there is no retry that takes a private file back out of
# a web form.

#: Where an attach may read from when `browser.upload_roots` says nothing (or
#: says nothing). Directories, not globs; `~/Downloads/**` is accepted and
#: read as `~/Downloads`.
DEFAULT_UPLOAD_ROOTS: tuple[str, ...] = ("~/lloyd-data", "~/Downloads")

#: A 3.3 GB recording is the measured real use case (the 2026-11-16 RSSC
#: upload that filed this item), so the ceiling sits above that and below
#: "unbounded": it exists to stop an accidental multi-terabyte sweep of a
#: media directory, not to police a deliberate publish.
DEFAULT_UPLOAD_MAX_BYTES = 8 * 1024 ** 3


def _upload_config() -> tuple[list[Path], int]:
    """The approved roots and the byte ceiling, resolved.

    Raises ValueError when either cannot be evaluated, which the caller turns
    into a refusal for the whole call. A malformed root entry is refused rather
    than dropped: a list of five with one typo inside is not a policy that
    quietly becomes the other four.
    """
    browser_cfg = CONFIG.get("browser") or {}
    raw_roots = browser_cfg.get("upload_roots")
    if raw_roots is None or (isinstance(raw_roots, (list, tuple)) and not raw_roots):
        entries: list = list(DEFAULT_UPLOAD_ROOTS)
    elif isinstance(raw_roots, str):
        entries = [raw_roots]
    elif isinstance(raw_roots, (list, tuple)):
        entries = list(raw_roots)
    else:
        raise ValueError(
            "browser.upload_roots must be a list of directories, got "
            f"{type(raw_roots).__name__}"
        )

    roots: list[Path] = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"browser.upload_roots entry is not a path: {entry!r}")
        text = re.sub(r"/(?:\*\*?)+$", "", entry.strip()).rstrip("/") or "/"
        path = Path(os.path.expanduser(text))
        if not path.is_absolute():
            raise ValueError(
                f"browser.upload_roots entry is not absolute: {entry!r}"
            )
        roots.append(path.resolve())
    if not roots:
        raise ValueError("browser.upload_roots resolved to no directory")

    raw_max = browser_cfg.get("upload_max_bytes", DEFAULT_UPLOAD_MAX_BYTES)
    try:
        max_bytes = int(raw_max)
    except (TypeError, ValueError):
        raise ValueError(
            f"browser.upload_max_bytes is not a byte count: {raw_max!r}"
        ) from None
    if max_bytes <= 0:
        raise ValueError(f"browser.upload_max_bytes must be positive, got {max_bytes}")
    return roots, max_bytes


def _under_any(path: Path, roots: list[Path]) -> bool:
    """Is `path` one of `roots` or inside one? Whole components only."""
    return _covering_root(path, roots) is not None


def _covering_root(path: Path, roots: list[Path]) -> Path | None:
    """The deepest of `roots` that is `path` or contains it, or None.

    Deepest, so the dot rule is judged from the most specific root the operator
    named: with both `~` and `~/.config/app` listed, a file under the second is
    measured from it, not from `~` (where `.config` would refuse it).
    """
    covering = [root for root in roots if path == root or root in path.parents]
    return max(covering, key=lambda r: len(r.parts)) if covering else None


def _upload_hard_deny(real: str) -> str | None:
    """The label of the never-attachable root covering `real`, or None.

    Both existing deny-sets are consulted rather than a browser-specific copy of
    their contents: the credential trees and the write deny-set already decide
    which paths no tool may touch, and a second list here would be a second
    answer to that question — one that drifts the day either set grows.

    Imported here, not at module top, so the file's line anchors cited from
    architecture/*.md above the upload section stay where they were.
    """
    from app.harness import protected_paths
    reason = protected_paths.read_deny_reason(real)
    if reason:
        return reason
    reason = protected_paths.write_deny_reason(real)
    if reason:
        return reason
    if _under_any(Path(real), [_vault_root()]):
        return "the Obsidian vault is never attachable to a page"
    return None


def _vault_root() -> Path:
    return Path(str(VAULT_ROOT)).resolve()


def _upload_path_policy(raw: object, roots: list[Path],
                        max_bytes: int) -> tuple[Path | None, str]:
    """Approve one path for attaching, or say in one line why this one is refused.

    Returns `(resolved_path, "")` or `(None, reason)`. Every reason names the
    path as the caller spelled it, because that is the string the agent has to
    correct; a resolved path that differs is named alongside it.

    Order is not arbitrary: the hard-deny roots, the dot rule and the roots
    check all come before the existence check, so a refusal about a credential
    tree is a refusal about a credential tree whether or not the file is on
    disk — a caller cannot learn which files exist outside the approved roots
    by watching which refusals mention a missing file.

    The dot rule is judged on the components BELOW the approved root the path
    falls under, not on the root's own spelling. A root the operator listed is
    attachable whatever its parents are called (the gate runs every test under
    `~/lloyd-work/.t/<hex>`, and a root there is a root); what the rule exists
    to catch is a `.ssh` or `.config/gh` reached by walking down from a broad
    root, which the deny-sets do not enumerate.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, f"upload path {raw!r} is not a file path"
    text = raw.strip()
    expanded = os.path.expanduser(text)
    if not os.path.isabs(expanded):
        return None, (
            f"{raw}: upload paths must be absolute (or start with ~/); the "
            "server's working directory is not part of this tool's contract"
        )
    resolved = Path(expanded).resolve()
    real = str(resolved)

    hard = _upload_hard_deny(real)
    if hard:
        return None, f"{raw}: {hard}"
    root = _covering_root(resolved, roots)
    below = resolved.relative_to(root).parts if root is not None else resolved.parts
    dotted = any(part.startswith(".") for part in below if part not in ("/", ""))
    if root is None:
        return None, (
            f"{raw} resolves to {real}, outside browser.upload_roots "
            f"({', '.join(str(r) for r in roots)}); "
            + ("its path runs through a dot-directory or names a dotfile, so "
               "approve only the exact directory an upload is meant to come "
               "from, never a parent of it" if dotted else
               "add that directory to browser.upload_roots if this upload is "
               "meant to happen")
        )
    if dotted:
        return None, (
            f"{raw}: dotfiles and dot-directories are never attachable"
            + ("" if real == text else f" (resolves to {real})")
        )
    if not resolved.exists():
        return None, f"{raw} does not exist" + ("" if real == text else f" (resolves to {real})")
    if resolved.is_dir():
        return None, f"{raw} is a directory; name a file inside it"
    if not resolved.is_file():
        return None, f"{raw} is not a regular file"
    size = resolved.stat().st_size
    if size > max_bytes:
        return None, (
            f"{raw} is {size} bytes, over the {max_bytes}-byte "
            "browser.upload_max_bytes ceiling"
        )
    return resolved, ""


def _page_origin(page) -> str:
    """The scheme+host of what is open, for the audit line.

    The path is only half of what an attach discloses; the other half is the
    site that receives it. `page.url` survives a navigation that never
    committed as the previous document's URL, which is exactly the imprecision
    wanted here — it names the page the bytes were handed to as the browser
    understood it.
    """
    try:
        parts = urllib.parse.urlparse(page.url or "")
    except ValueError:
        return "?"
    if not parts.scheme:
        return (page.url or "?")[:120]
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else parts.scheme


async def _browser_upload_files(ref: str = "", paths: object = None,
                                selector: str = "",
                                frame_index: int | None = None) -> str:
    """Put local files into a page's `<input type=file>`.

    Two ways to name the input, because the real one is the awkward one. A page
    whose upload control is a styled button drives a `display:none` input that
    the accessibility tree never shows and so never gives a ref for — YouTube
    Studio's upload dialog among them. `selector` reaches that input directly,
    which is why the parameter exists and why a ref-only design would have been
    useless where this is needed. `ref` stays for a visible input.

    The bytes never travel through this process: Playwright hands Chromium the
    path and the browser reads the file, so a 3.3 GB recording costs the same
    one call as a 2 KB one. Every path is policy-checked before the input is
    touched, so a refusal — including one on the third path of a batch — leaves
    the page exactly as it was.
    """
    raws: list = list(paths) if isinstance(paths, (list, tuple)) else (
        [paths] if isinstance(paths, str) else [])
    if not raws:
        return json.dumps({"error": "paths is required: one absolute file path, or a list of them"})
    if bool(ref) == bool(selector):
        return json.dumps({
            "error": "pass exactly one of ref (a snapshot ref to a visible input) "
                     "or selector (a CSS selector, which is how a display:none "
                     "input is reached — it has no ref)"
        })
    if frame_index is not None and not selector:
        return json.dumps({
            "error": "frame_index goes with selector; a snapshot ref already names "
                     "the frame it was taken from"
        })

    try:
        roots, max_bytes = _upload_config()
    except ValueError as exc:
        logger.warning("browser_upload_files: refusing every path, upload policy "
                       "could not be evaluated: %s", exc)
        return json.dumps({
            "error": f"upload policy could not be evaluated, so nothing is "
                     f"attachable: {exc}"
        })

    approved: list[str] = []
    reasons: list[str] = []
    for raw in raws:
        path, reason = _upload_path_policy(raw, roots, max_bytes)
        if path is None:
            reasons.append(reason)
        else:
            approved.append(str(path))
    if reasons:
        # Nothing has touched the page yet, which is the whole point of checking
        # the batch up front rather than path by path as it is attached.
        logger.warning("browser_upload_files: refused %s", "; ".join(reasons))
        return json.dumps({"error": "; ".join(reasons),
                           "refused": reasons,
                           "not_attached": [str(r) for r in raws]})

    page = await _get_page()
    # Where the bytes go is half the question, so the same landing guard that
    # stands in front of every read stands in front of this write too. A page
    # that reached a private host by redirect has no business receiving a local
    # file, and `browser_upload_files` is the one browser verb where the
    # disclosure runs the other way: the guard on the read paths stops a LAN
    # device's content getting out, this stops this machine's content getting in.
    landed = await _enforce_landing(page)
    if landed:
        return json.dumps({"error": f"{landed} — nothing attached"})
    try:
        if selector:
            scope = _frame_at(page, frame_index) if frame_index is not None else page
            locator = scope.locator(selector)
            matched = await locator.count()
            if matched == 0:
                where = (f"frame {frame_index}" if frame_index is not None
                         else "the main frame")
                return json.dumps({
                    "error": f"no element matches selector '{selector}' in {where} "
                             f"of {_page_origin(page)}. Call browser_snapshot to see "
                             "the page, or pass a ref for a visible input."
                        })
            if matched > 1:
                return json.dumps({
                    "error": f"selector '{selector}' matches {matched} elements; "
                             "attaching to the first of several would be a guess "
                             "about which input receives the file. Narrow the "
                             "selector to one element."
                        })
            target = locator.first
        else:
            target = await _locate(page, ref)
        await target.set_input_files(approved, timeout=30000)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({
            "error": f"Upload failed ({type(exc).__name__}): {str(exc)[:300]}"
        })

    # The disclosure audit line: which local files, into which origin. Deliberately
    # INFO — this is the record an operator reads to answer "what did the agent
    # upload, and to whom", and a refusal above it is a warning.
    logger.info("browser_upload_files: attached %d file(s) to %s: %s",
                len(approved), _page_origin(page), ", ".join(approved))
    return json.dumps({
        "ok": True,
        "attached": len(approved),
        "paths": approved,
        "target_origin": _page_origin(page),
        "via": {"selector": selector} if selector else {"ref": ref},
    })


# ── MCP interface ──────────────────────────────────────────────────────────────

async def list_tools():
    return [
        # ── Phase 1: Core browse-read-interact loop ────────────────────────────
        Tool(name="browser_navigate", description=(
            "Navigate the browser to a URL. Returns the page title and HTTP status. "
            "A document that responds 4xx/5xx comes back with ok:false and a `warning` "
            "naming the status and the page title: the fetch reached a server, but what "
            "answered is an error page, not the target. "
            "Always call browser_snapshot after navigating to see the page content."
        ), inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to (http/https only)"},
                "wait_until": {
                    "type": "string",
                    "enum": ["load", "domcontentloaded", "networkidle", "commit"],
                    "description": (
                        "When to consider navigation complete. For an SPA use "
                        "\"load\", then `browser_wait` with condition \"text\" or "
                        "\"selector\" for the content you are waiting for: "
                        "\"networkidle\" wants ~500 ms of total network silence "
                        "and never fires on a page holding a live websocket, "
                        "EventSource or HMR connection, so it times out on pages "
                        "that are fully loaded. Default: domcontentloaded"
                    ),
                },
            },
            "required": ["url"],
        }),
        Tool(name="browser_snapshot", description=(
            "Get the accessibility tree of the current page as structured text. "
            "A document whose navigation returned an error status is labelled with that "
            "status on the first line, because its tree otherwise looks like the "
            "target's. "
            "Interactive elements (links, buttons, form fields) are assigned ref IDs like e1, e2, e3. "
            "Use these refs with browser_click, browser_fill, etc. "
            "Refs are invalidated after each new snapshot or navigation. "
            "Content inside an iframe is not folded into the top tree: each child "
            "frame gets its own `### frame <n> <url>` section, and a ref found in "
            "one acts inside that frame. A frame whose content could not be read "
            "is still listed, with the reason, so a partial extraction never reads "
            "as a complete one."
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
            "Click an element by its ref ID from the last browser_snapshot."
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
            "Take a screenshot of the current page. Returns the image (shown to "
            "models that can see) and saves it to logs/screenshots/. Useful for "
            "visual verification."
        ), inputSchema={
            "type": "object",
            "properties": {},
        }),
        Tool(name="browser_evaluate", description=(
            "Execute JavaScript in the current page context and return the result. "
            "Useful for extracting data that isn't in the accessibility tree, "
            "checking JS state, or manipulating the DOM directly. "
            "Timeout: 10 seconds. Return value limited to 50KB. "
            "Pass frame_index — the `<n>` from a `### frame <n> <url>` header in "
            "browser_snapshot — to run the script inside that frame instead of the "
            "top one. That is the only way to reach a cross-origin frame, and the "
            "only way to reach a frame's own JS world; same-origin prose is also "
            "readable from the top frame via contentDocument."
        ), inputSchema={
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "JavaScript expression or function body to execute"},
                "frame_index": {
                    "type": "integer",
                    "description": (
                        "Index into page.frames, as the snapshot's `### frame <n>` "
                        "headers number them; 0 is the main frame. Out of range is "
                        "an error."
                    ),
                },
            },
            "required": ["script"],
        }),
        Tool(name="browser_fill", description=(
            "Use to put text into a form field by ref; to press a single key use browser_press. "
            "Replaces the field's whole value in one write that fires input/change events. "
            "Pass keystrokes=true for a field that reacts to typing, such as search-as-you-type "
            "or autocomplete: it clears the field and types the value key by key."
        ), inputSchema={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Element ref ID from browser_snapshot"},
                "value": {"type": "string", "description": "Value the field should hold"},
                "keystrokes": {"type": "boolean", "description": "Type key by key instead of one write. Default: false"},
            },
            "required": ["ref", "value"],
        }),
        Tool(name="browser_wait", description=(
            "Wait for a condition before continuing: a selector or text to appear, "
            "navigation, network idle, or a fixed timeout in ms."
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
        Tool(name="browser_upload_files", description=(
            "Use this when a page needs a LOCAL FILE in its <input type=file>; pass "
            "selector='input[type=file]' instead of ref for a hidden input (most "
            "upload buttons). Any size is one call. Paths must sit inside "
            "browser.upload_roots; dotfiles, protected trees and the vault are refused."
        ), inputSchema={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Ref of a visible file input."},
                "paths": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Absolute paths; a call replaces the input's "
                                   "files, so pass them all at once.",
                },
                "selector": {"type": "string", "description": "CSS selector for one file input."},
                "frame_index": {"type": "integer",
                                "description": "With selector: search page.frames[n]."},
            },
            "required": ["paths"],
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
            arguments.get("frame_index"),
        ),
        "browser_fill": lambda: _browser_fill(
            arguments.get("ref", ""),
            arguments.get("value", ""),
            bool(arguments.get("keystrokes", False)),
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
        "browser_upload_files": lambda: _browser_upload_files(
            arguments.get("ref", ""),
            arguments.get("paths"),
            arguments.get("selector", ""),
            arguments.get("frame_index"),
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
    if isinstance(result, CallToolResult):
        return result
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
    if _ui_trailing_push is not None and not _ui_trailing_push.done():
        _ui_trailing_push.cancel()
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


# ── Mission Control's URL bar ──────────────────────────────────────────────────

# Failures that mean "right host, wrong scheme" rather than "no such page".
_WRONG_SCHEME_MARKERS = (
    "ERR_SSL_PROTOCOL_ERROR",
    "ERR_CONNECTION_CLOSED",
    "ERR_CONNECTION_RESET",
    "ERR_EMPTY_RESPONSE",
    "ERR_SSL_VERSION_OR_CIPHER_MISMATCH",
)


def _looks_like_wrong_scheme(error: str) -> bool:
    return any(m in error for m in _WRONG_SCHEME_MARKERS)


async def navigate_from_ui(url: str) -> dict:
    """Drive the shared browser from the Browser tab's URL bar.

    Reached over loopback from the backend, not through MCP: the URL bar is
    the user typing, and routing a human keystroke through the agent's tool
    surface would put it in the transcript as something the model did.

    Two deliberate differences from ``browser_navigate``:

    * A scheme-less host is completed to ``https://``. The tool stays strict
      because an agent that omits the scheme has made a mistake worth seeing;
      a human typing ``news.ycombinator.com`` has just saved eight keystrokes.
    * The state push is awaited rather than fired off. ``call_tool`` can spawn
      it because the agent's next move is its own tool call, but the URL bar's
      caller *is* the viewer — returning before the frame exists shows them
      the page they navigated away from until the next push happens to land.

    The frame is tagged ``url_bar`` rather than ``browser_navigate`` so the
    tab can say a human drove this one.
    """
    url = (url or "").strip()
    if not url:
        return {"error": "url is required"}
    # Detect a scheme by "://" rather than by the colon alone. A bare colon
    # is ambiguous with a port, and the port is the case this box hits first:
    # `[a-zA-Z][a-zA-Z0-9+.-]*:` happily reads the whole of "localhost:8080"
    # and "example.com:8080/x" as a scheme, leaving them uncompleted for
    # `_browser_navigate` to reject as not-http. A protocol-relative
    # "//example.com" has no scheme either and gets the same completion.
    completed = not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url)
    if completed:
        url = "https://" + url.lstrip("/")

    raw = await _browser_navigate(url)
    try:
        result = json.loads(raw)
    except Exception:
        result = {"error": raw}
    if not isinstance(result, dict):
        result = {"error": str(result)}

    # If the scheme was ours and https did not take, try http once. Local
    # services are the case that needs it and they are split both ways on this
    # box alone: the frontend serves TLS on 5173 while the backend, the
    # aggregator and both engines are plain HTTP. Guessing wrong either way
    # leaves a URL bar that cannot open the thing next to it. Browsers do the
    # same fallback; we scope it to a completion we made, so a URL the user
    # typed `https://` on themselves is never silently downgraded.
    if completed and result.get("error") and _looks_like_wrong_scheme(result["error"]):
        # A failed navigation settles into `chrome-error://chromewebdata/`
        # asynchronously, and an immediate second goto races that: "Navigation
        # to http://... is interrupted by another navigation to
        # chrome-error://chromewebdata/". So let the page settle, then retry
        # once more if we lost the race anyway — the interruption says nothing
        # about whether the scheme was right, and giving up there reports an
        # SSL error for a URL that works.
        retry = "http://" + url[len("https://"):]
        alt = None
        for attempt in range(2):
            page = _existing_page()
            if page is not None:
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
            try:
                alt = json.loads(await _browser_navigate(retry))
            except Exception:
                alt = None
            if not isinstance(alt, dict):
                alt = None
                break
            if not alt.get("error") or "interrupted by another navigation" not in alt["error"]:
                break
        if isinstance(alt, dict) and not alt.get("error"):
            result = alt

    # Push even on failure: a navigation that 404s or times out still leaves
    # the viewport showing something, and an unchanged tab after a click is
    # the most confusing possible answer.
    await _push_browser_state("url_bar")
    return result


# ── Mission Control's pointer and keyboard ─────────────────────────────────────
#
# The URL bar was the tab's one control, so a page that wanted a login could be
# watched and never answered. These are the human's hands on the same Chromium:
# a click at a pixel of the frame, text, a named key, a wheel. Same seam and the
# same reasoning as the URL bar — a route on this process, proxied by the
# backend, never an MCP call — and one more that matters more here: what a human
# types into a login form is a credential, and nothing on this path may write it
# down. No log line carries the text or a Playwright message about it (those
# echo their arguments), the frame masks password values
# (`_aria_snapshot_masked`), and the agent's transcript never sees the call.

_UI_INPUT_KINDS = ("click", "type", "key", "scroll")
_UI_MAX_TEXT = 4096
_UI_MAX_SCROLL = 5000
_UI_NAMED_KEYS = frozenset({
    "Enter", "Tab", "Backspace", "Delete", "Escape", "Space", "Insert",
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Home", "End", "PageUp", "PageDown",
    *(f"F{i}" for i in range(1, 13)),
})
_UI_MODIFIERS = frozenset({"Shift", "Control", "Alt", "Meta", "ControlOrMeta"})
# Serialises input from every open tab, so two viewers' keystrokes interleave
# per event rather than per character inside one `keyboard.type`.
_ui_input_lock = asyncio.Lock()
# What an event owes after it lands — the tab it may have opened, where the page
# ended up, the frame with its accessibility tree — is paid off the request, by
# one task per burst that a newer event supersedes. The pixels do not wait for
# it: the live view (`_live_loop`, below) is already streaming them. Until
# 2026-11-17 every event awaited all of this under the input lock, so a click
# cost ~0.4 s and a burst of keys queued behind each other.
_ui_trailing_push: asyncio.Task | None = None
_ui_before: list | None = None
_ui_first_frame: asyncio.Event | None = None
_UI_SETTLE_S = 0.15
_UI_TRAILING_PUSH_S = 1.0


def _valid_ui_key(key: object) -> bool:
    """`Shift+Tab`, `Control+a`, `Enter` — Playwright's names, nothing else."""
    if not isinstance(key, str) or not key or len(key) > 48:
        return False
    *mods, last = key.split("+") if key != "+" else ["+"]
    if any(m not in _UI_MODIFIERS for m in mods):
        return False
    return last in _UI_NAMED_KEYS or (len(last) == 1 and last.isprintable())


def _ui_number(value: object, lo: float, hi: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value:  # NaN
        return None
    return max(lo, min(hi, float(value)))


def _schedule_ui_settle(kind: str) -> None:
    global _ui_trailing_push, _ui_first_frame
    if _ui_trailing_push is not None and not _ui_trailing_push.done():
        _ui_trailing_push.cancel()
    superseded, _ui_first_frame = _ui_first_frame, asyncio.Event()
    if superseded is not None:
        superseded.set()  # release a waiter onto the burst that replaced it
    done = _ui_first_frame

    async def _later() -> None:
        global _active_page, _ui_before
        # A click or a key may have started a navigation or opened a tab. Give
        # it a moment to commit.
        await asyncio.sleep(_UI_SETTLE_S)
        before, _ui_before = _ui_before, None
        page = _existing_page()
        if _context is not None and before is not None:
            opened = [p for p in _context.pages if p not in before and not p.is_closed()]
            if opened:
                page = _active_page = opened[-1]
        if page is not None:
            await _enforce_landing(page)
        await _push_browser_state("user_input")
        done.set()
        if kind != "type":
            # A click's effect often lands late (a menu animating open, a SPA
            # route fetching); the tree wants one more read.
            await asyncio.sleep(_UI_TRAILING_PUSH_S)
            await _push_browser_state("user_input")

    _ui_trailing_push = asyncio.create_task(_later())


async def _ui_settled() -> None:
    """Wait until the newest burst has pushed its first frame. For tests."""
    while True:
        done = _ui_first_frame
        if done is None:
            return
        await done.wait()
        if done is _ui_first_frame:
            return


async def input_from_ui(event: dict) -> dict:
    """Apply one pointer or keyboard event from the Browser tab.

    `event` is one of::

        {"kind": "click",  "x": 412, "y": 233, "button": "left", "count": 1}
        {"kind": "type",   "text": "alan@example.com"}
        {"kind": "key",    "key": "Shift+Tab"}
        {"kind": "scroll", "x": 640, "y": 400, "dx": 0, "dy": 300}

    Coordinates are frame pixels, which are viewport CSS pixels — the frame is a
    viewport screenshot at device scale 1 — clamped to the viewport.

    Never launches a browser: input with no page open has nothing to land on,
    and reviving a browser the agent closed is the state mirror's rule too.
    Every navigation an event causes passes the same host checks a tool call's
    does — the connection gate for the request, `_enforce_landing` for
    wherever it ended up. A click that opens a tab moves the view to that tab,
    or the viewer would be left on the page they clicked away from. Returns as
    soon as the event is delivered: the landing check, the tab switch and the
    `user_input` frame follow in `_schedule_ui_settle`, and the live view shows
    the pixels meanwhile.

    An error never carries Playwright's message: for `type` that message would
    quote the text, and for the others the type name is enough to act on.
    """
    global _ui_before
    if not isinstance(event, dict):
        return {"error": "event must be an object"}
    kind = event.get("kind")
    if kind not in _UI_INPUT_KINDS:
        return {"error": f"kind must be one of {', '.join(_UI_INPUT_KINDS)}"}

    text = key = None
    if kind == "type":
        text = event.get("text")
        if not isinstance(text, str) or not text:
            return {"error": "text is required"}
        if len(text) > _UI_MAX_TEXT:
            return {"error": f"text is limited to {_UI_MAX_TEXT} characters per event"}
    elif kind == "key":
        key = event.get("key")
        if not _valid_ui_key(key):
            return {"error": "unsupported key"}

    async with _ui_input_lock:
        page = _existing_page()
        if page is None:
            return {"error": "no page is open — type a URL first"}
        viewport = page.viewport_size or {"width": 1280, "height": 800}
        vw, vh = viewport["width"] - 1, viewport["height"] - 1
        if _ui_before is None and _context is not None:
            _ui_before = list(_context.pages)
        try:
            if kind in ("click", "scroll"):
                x = _ui_number(event.get("x"), 0, vw)
                y = _ui_number(event.get("y"), 0, vh)
                if x is None or y is None:
                    return {"error": "x and y are required"}
            if kind == "click":
                button = event.get("button") if event.get("button") in ("left", "right", "middle") else "left"
                count = int(_ui_number(event.get("count", 1), 1, 3) or 1)
                if count == 1:
                    await page.mouse.click(x, y, button=button)
                else:
                    # The tab sends every click of a double-click as it happens,
                    # the second one carrying count 2 (the DOM's `detail`). One
                    # press at that count is what a real second click is;
                    # `mouse.click(click_count=2)` would press twice more.
                    await page.mouse.move(x, y)
                    await page.mouse.down(button=button, click_count=count)
                    await page.mouse.up(button=button, click_count=count)
            elif kind == "type":
                await page.keyboard.type(text)
            elif kind == "key":
                await page.keyboard.press(key)
            else:
                dx = _ui_number(event.get("dx", 0), -_UI_MAX_SCROLL, _UI_MAX_SCROLL) or 0.0
                dy = _ui_number(event.get("dy", 0), -_UI_MAX_SCROLL, _UI_MAX_SCROLL) or 0.0
                await page.mouse.move(x, y)
                await page.mouse.wheel(dx, dy)
        except Exception as exc:
            logger.warning("browser/input: %s failed: %s", kind, type(exc).__name__)
            return {"error": f"{kind} failed ({type(exc).__name__})"}

    _schedule_ui_settle(kind)
    return {"ok": True, "kind": kind}


# ── Password values in the accessibility tree ─────────────────────────────────
#
# Kept at the end of the module so the line anchors other docs cite above it
# do not move; every caller resolves it at call time.

# Chromium's accessibility tree carries a password field's value in plain text:
# `- textbox "Password": hunter2`, measured against /usr/bin/chromium on
# 2026-11-16. The screenshot shows dots, the tree does not. Every aria read in
# this module therefore goes through `_aria_snapshot_masked`, because the tree
# reaches two places a secret must not: the model's context (`browser_snapshot`,
# and from there the session transcript on disk) and the Mission Control frame.
# It mattered less while only the agent could fill a field — it already knew the
# value — and it matters now that the Browser tab lets a human type credentials.
_PASSWORD_MASK = "••••••••"
_PASSWORD_VALUES_JS = (
    "() => Array.from(document.querySelectorAll('input[type=password]'),"
    " e => e.value).filter(v => typeof v === 'string' && v.length > 0)"
)
# One textbox line of Playwright's aria YAML: role, optional quoted name (which
# may itself contain `: `), any number of `[attr]` groups, then `: value`.
_TEXTBOX_VALUE_RE = re.compile(
    r'^(?P<head>[ \t]*- textbox(?:[ \t]+"(?:[^"\\]|\\.)*")?(?:[ \t]+\[[^\]\n]*\])*):[ \t]'
    r'(?P<value>[^\n]+)$',
    re.M,
)


def _mask_password_values(raw: str, secrets: list[str], mask_all: bool = False) -> str:
    """Replace the value of every textbox line that is (part of) a password.

    Matched both ways round — the value holds a secret, or a secret holds the
    value — because the tree and the DOM are read in two calls and the field can
    change between them: a keystroke lengthens one, a backspace shortens it.
    YAML may quote a value with a colon in it, which the containment test
    survives as well. Only textbox values are touched, so a one-character
    password costs some over-masking of other fields, never a leak.

    `mask_all` is the fail-closed answer when the DOM could not be asked which
    fields are passwords: every textbox value goes.
    """
    if not raw or (not secrets and not mask_all):
        return raw

    def _sub(m: re.Match) -> str:
        value = m.group("value")
        bare = value.strip().strip('"')
        if mask_all or any(s in value or (bare and bare in s) for s in secrets):
            return f"{m.group('head')}: {_PASSWORD_MASK}"
        return m.group(0)

    return _TEXTBOX_VALUE_RE.sub(_sub, raw)


async def _aria_snapshot_masked(target) -> str:
    """`target.locator("body").aria_snapshot()` with password values masked.

    `target` is a Page or a Frame; both answer `locator` and `evaluate`. Raises
    what the aria read raises, so callers keep their own error wording.
    """
    raw = await target.locator("body").aria_snapshot()
    if not raw or "textbox" not in raw:
        return raw
    try:
        found = await target.evaluate(_PASSWORD_VALUES_JS)
        secrets = [s for s in (found or []) if isinstance(s, str) and s]
        mask_all = False
    except Exception as exc:
        # The tree was readable and the DOM was not: mid-navigation, most often.
        # Not knowing which fields are passwords is a reason to hide them all.
        logger.debug("browser: password scan failed, masking every textbox: %s",
                     type(exc).__name__)
        secrets, mask_all = [], True
    return _mask_password_values(raw, secrets, mask_all=mask_all)


# ── The connection gate's proxy ───────────────────────────────────────────────
#
# At the end of the module for the same reason as the block above. One proxy for
# the life of the process: a relaunched browser reuses it, and it holds nothing
# but a listening socket on 127.0.0.1.

_guard_proxy = None


def _guard_proxy_url() -> str:
    """Start the gate on first use and return its address.

    Raises when it cannot bind, and the launch fails with it: a browser with no
    gate in front of it is the one configuration this module must never run.
    """
    global _guard_proxy
    if _guard_proxy is None:
        from agent_mcp._browser_proxy import GuardProxy
        _guard_proxy = GuardProxy(
            lambda host, port, url: _guard_connect(host, port, url)).start()
    return _guard_proxy.url


# ── The live view ─────────────────────────────────────────────────────────────
#
# A frame used to exist only after a tool call or a viewer's own input, so a
# page that changed by itself — a progress bar, a slow load, a menu animating —
# was invisible until somebody poked it. Chromium's screencast pushes a frame
# whenever the pixels change; this loop forwards the newest one to the backend
# as a LIGHT frame (`live: true`, no accessibility tree), and the full frame
# with the tree still follows each tool call and each input burst.
#
# Three rules:
# - The ack is the throttle. Chromium sends the next frame only after the last
#   is acknowledged, so acking after the post caps the stream at
#   `_LIVE_MIN_INTERVAL_S` instead of hauling 60 fps of JPEG through the driver.
# - Nobody watching, nothing streaming. The backend's answer to every post
#   carries its subscriber count; at zero the screencast stops and the loop
#   falls back to one cheap screenshot every `_LIVE_IDLE_POLL_S`, which is also
#   how it learns a viewer arrived.
# - The same host check as `_capture_state`, failing closed the same way: a
#   frame of a host nobody could vouch for is dropped, not pushed.

_LIVE_MIN_INTERVAL_S = 0.12
_LIVE_IDLE_POLL_S = 3.0
_LIVE_QUIET_BEAT_S = 5.0
_LIVE_JPEG_QUALITY = 55
_live: dict = {"page": None, "task": None}


def _resolve_live_view() -> bool:
    """`browser.live_view` in config.yaml, LLOYD_BROWSER_LIVE in the env. On by default."""
    raw = os.environ.get("LLOYD_BROWSER_LIVE")
    if raw is None or not raw.strip():
        configured = (CONFIG.get("browser") or {}).get("live_view")
        if configured is None:
            return True
        raw = str(configured)
    return raw.strip().lower() not in _FALSY


def _ensure_live(page) -> None:
    """Stream `page` to the Browser tab, replacing a stream of any other page."""
    if page is None or not _resolve_live_view():
        return
    task = _live.get("task")
    if _live.get("page") is page and task is not None and not task.done():
        return
    if task is not None and not task.done():
        task.cancel()
    _live["page"] = page
    _live["task"] = asyncio.create_task(_live_loop(page))


async def _live_loop(page) -> None:
    cdp = None
    casting = False
    pending: dict = {"data": None, "session": None}
    wake = asyncio.Event()

    def _on_frame(params: dict) -> None:
        pending["data"] = params.get("data")
        pending["session"] = params.get("sessionId")
        wake.set()

    try:
        cdp = await page.context.new_cdp_session(page)
        cdp.on("Page.screencastFrame", _on_frame)
        viewers = 1  # optimistic: the push that started this loop had a reason
        full_url = page.url
        async with httpx.AsyncClient(timeout=5.0) as client:
            while not page.is_closed() and _active_page is page:
                if viewers > 0 and not casting:
                    vp = page.viewport_size or {"width": 1280, "height": 800}
                    await cdp.send("Page.startScreencast", {
                        "format": "jpeg", "quality": _LIVE_JPEG_QUALITY,
                        "maxWidth": vp["width"], "maxHeight": vp["height"],
                        "everyNthFrame": 1})
                    casting = True
                elif viewers == 0 and casting:
                    await cdp.send("Page.stopScreencast")
                    casting = False
                    pending["data"] = pending["session"] = None
                    wake.clear()

                if casting:
                    try:
                        await asyncio.wait_for(wake.wait(), _LIVE_QUIET_BEAT_S)
                    except asyncio.TimeoutError:
                        pass
                    wake.clear()
                else:
                    await asyncio.sleep(_LIVE_IDLE_POLL_S)
                data, session = pending["data"], pending["session"]
                pending["data"] = pending["session"] = None
                if data is None:
                    # A still page while watched, or the idle poll: one shot,
                    # mostly to ask the backend whether anyone is looking.
                    shot = await cdp.send("Page.captureScreenshot", {
                        "format": "jpeg", "quality": _LIVE_JPEG_QUALITY,
                        "optimizeForSpeed": True})
                    data = shot.get("data")

                url = page.url
                try:
                    blocked = await _host_block_reason_async(url)
                except Exception as exc:
                    logger.warning("browser: live guard failed closed on %s: %s",
                                   url[:120], f"{type(exc).__name__}: {exc}")
                    blocked = "unchecked"
                if not blocked and data:
                    if url != full_url:
                        # The page moved with nobody driving it; the tree and
                        # the title are stale until a full frame says otherwise.
                        full_url = url
                        _schedule_state_push("live")
                    try:
                        r = await client.post(f"{LLOYD_API}/api/browser/state", json={
                            "live": True, "url": url, "ts": time.time(),
                            "mime": "image/jpeg", "screenshot_b64": data})
                        viewers = int((r.json() or {}).get("subscribers") or 0)
                    except Exception as exc:
                        logger.debug("browser live: post failed: %s", type(exc).__name__)
                        viewers = 0
                await asyncio.sleep(_LIVE_MIN_INTERVAL_S)
                if session is not None and casting:
                    await cdp.send("Page.screencastFrameAck", {"sessionId": session})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # The page closed under the stream, most often. The next push restarts it.
        logger.debug("browser live: stream ended: %s", type(exc).__name__)
    finally:
        if cdp is not None:
            try:
                # Bounded: a detach on a browser that is already gone never answers.
                await asyncio.wait_for(cdp.detach(), 2)
            except Exception:
                pass
