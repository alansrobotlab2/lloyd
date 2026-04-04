# Browser Control: Analysis & Implementation Gameplan

> **Date**: 2026-04-04  
> **Status**: Proposal  
> **References**: `~/Projects/openclaw`, `~/Projects/hermes-agent`

---

## 1. Problem Statement

Lloyd currently has no browser automation capability. The `http_tools` MCP server can fetch URLs and extract text, but it cannot interact with dynamic web pages — no clicking, form filling, JavaScript execution, or visual inspection. Full browser control would enable Lloyd to:

- Navigate and interact with web applications
- Fill out forms, click buttons, follow multi-step workflows
- Read dynamic/JS-rendered page content
- Take screenshots for visual verification
- Extract structured data from pages that require interaction

Two reference implementations exist in our codebase. This document analyzes both and proposes an implementation path for Lloyd.

---

## 2. Reference Implementation Analysis

### 2.1 OpenClaw (`~/Projects/openclaw`)

**Stack**: Playwright Core (v1.58.2) + Chrome DevTools Protocol (CDP) + Express HTTP bridge server

**Architecture**:
```
Agent Tool Interface
    │
    ▼
Plugin Service (lazy-loaded)
    │
    ▼
Bridge Server (Express HTTP, per-session)
    │
    ├── Sandbox Mode: Docker container with Chromium + CDP
    └── Host Mode: Local Chrome via CDP WebSocket
```

**Key Design Choices**:

- **Bridge server pattern**: An Express HTTP server sits between the agent and the browser. Actions are HTTP POST requests (`/act`, `/snapshot`, `/screenshot`). This exists because the browser may run in a Docker sandbox, requiring network-level proxying of CDP.

- **AI-optimized snapshots**: Uses Playwright's undocumented `_snapshotForAI()` API for accessibility trees tuned for LLM consumption. Also supports standard ARIA snapshots and role-based references.

- **Element reference system**: Snapshots return refs like `e1`, `e2`, `e3` mapped to elements. Subsequent actions (`click`, `type`) use these refs. Two modes: `role` (Playwright role API) and `aria` (Playwright aria-ref tokens).

- **Rich action set**: click, type, press, hover, drag, select, fill (multi-field forms), resize, wait (multiple conditions), evaluate (JS execution), close, and **batch** (multiple actions in sequence).

- **Multi-profile support**: Separate browser profiles (`openclaw`, `user`, custom) with different drivers — `openclaw` for agent-managed browsers, `existing-session` for the user's logged-in Chrome via Chrome MCP.

- **Security**: SSRF policy enforcement, URL filtering, auth middleware (token + password), per-session bridge isolation.

- **Visual labels**: Optional overlay that injects numbered badges onto interactive elements in screenshots, so the agent can correlate visual position with ref IDs.

**Strengths**: Production-grade, comprehensive action set, sandbox isolation, multi-profile support.  
**Weaknesses**: Extremely complex (~50+ TypeScript files for browser alone), Docker dependency for sandbox mode, overkill for a single-agent system like Lloyd.

---

### 2.2 Hermes Agent (`~/Projects/hermes-agent`)

**Stack**: `agent-browser` CLI (Node.js npm package) + multi-provider abstraction (local, Browserbase cloud, BrowserUse cloud, Camofox anti-detect)

**Architecture**:
```
browser_tool.py (11 Python tool functions)
    │
    ▼
Cloud Provider Abstraction (browserbase.py / browser_use.py)
    │
    ▼
agent-browser CLI (subprocess per action)
    │
    ├── Local Chromium (headless)
    ├── Browserbase (cloud CDP)
    ├── BrowserUse (cloud CDP)
    └── Camofox (Firefox anti-detect, localhost:9377)
```

**Key Design Choices**:

- **CLI-per-action**: Every browser action spawns `agent-browser --session <name> --json <command>` as a subprocess. Session state persists in `agent-browser`'s daemon process between calls.

- **Text-first accessibility trees**: Primary page representation is a hierarchical text tree with `@eN` element references. Interactive elements get refs; static content is included for context. Auto-summarization via auxiliary LLM when tree exceeds ~8000 chars.

- **11 tools**: `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_press`, `browser_back`, `browser_close`, `browser_get_images`, `browser_vision`, `browser_console`.

- **Vision pipeline**: `browser_vision` takes a screenshot, encodes it as base64, sends it to a vision LLM with a natural language question. Supports `annotate=True` for numbered element overlays. This is a fallback for CAPTCHAs and complex visual layouts.

- **Session management**: Per-task session isolation, inactivity timeout (300s default), background cleanup thread, atexit handlers. Sessions stored in `_active_sessions` dict.

- **Bot detection**: Detects "access denied" / "cloudflare" / "bot detected" patterns in page titles. Returns actionable suggestions (add delays, enable stealth, try different pages).

- **Security**: SSRF protection (blocks private IPs pre- and post-redirect), secret exfiltration prevention (redacts API keys from URLs and snapshots).

**Strengths**: Simple Python integration, battle-tested tool set, vision fallback, bot detection, session management.  
**Weaknesses**: Node.js dependency (agent-browser CLI), subprocess-per-action overhead, multi-provider abstraction adds complexity Lloyd doesn't need.

---

### 2.3 Comparison Matrix

| Aspect | OpenClaw | Hermes | Lloyd Relevance |
|--------|----------|--------|-----------------|
| Browser library | Playwright Core (JS) | agent-browser CLI (Node.js) | Need Python-native |
| Page representation | AI snapshot + ARIA tree | Accessibility tree + vision | Text-first (local models) |
| Element refs | `e1`, `e2` (role/aria modes) | `@e1`, `@e2` | Essential |
| Actions | 12+ including batch, drag, fill | 11 including vision, console | 7 MVP, 14 full |
| Session persistence | Bridge server + Docker | CLI daemon process | SDK session keeps MCP alive |
| Screenshot handling | Sharp image processing | Base64 → vision LLM | Phase 2 |
| Sandbox/isolation | Docker container | Cloud provider | Not needed |
| Anti-bot | SSRF policy | Detection + mitigation | Basic SSRF |
| Complexity | ~50 TS files | ~1 Python file + providers | Target: 1 Python file |

---

## 3. Architecture Recommendation

### 3.1 Decision: Playwright Python Direct

**Use Playwright's Python bindings** to drive Chrome/Chromium directly via CDP. No Node.js bridge, no CLI wrapper, no cloud providers.

**Rationale**:

1. **No bridge needed.** OpenClaw's bridge server exists to proxy CDP from Docker containers. Lloyd runs browser and agent in the same environment — direct CDP connection suffices.

2. **No CLI wrapper needed.** Hermes uses `agent-browser` CLI because it predates MCP and needed a stable subprocess interface. Lloyd's MCP server IS the subprocess interface — Playwright calls go directly inside it.

3. **Simpler dependency chain.** One `pip install playwright` into the existing venv, plus `playwright install chromium`. No npm packages, no Node.js runtime dependency for this feature.

4. **Python accessibility API.** Playwright Python exposes `page.accessibility.snapshot()` for the text-based trees that local Qwen models need. The same core API that OpenClaw uses (via JS bindings) is available natively.

5. **First-class async.** Lloyd's MCP servers are async Python. Playwright's async API (`async_playwright`) integrates naturally without thread pools or subprocess overhead.

### 3.2 Session Lifecycle

With the persistent MCP server architecture (see [doc 08](08-mcp-persistent-servers.md)), MCP servers run as long-lived processes exposing an SSE endpoint. The SDK connects to them as a client — it does not spawn or kill them. This means the browser MCP server and its Chromium instance stay alive indefinitely.

```
supervisord
├── lloyd-backend (server.py)
├── lloyd-frontend (vite)
└── lloyd-mcp-browser (browser.py on port 8509)
    └── Playwright Chromium (launched once, stays alive)
        └── tabs, cookies, localStorage all persist across sessions

SDK query() connects via SSE → uses browser tools → disconnects
Next query() reconnects → same browser, same tabs, same state
```

**What persists across sessions**:
- Open tabs and their content
- Cookies, localStorage, sessionStorage
- In-page state (form data, JS variables, scroll position)
- Element ref map (reset per snapshot, but the pages stay)

**What resets**:
- Element refs (scoped to most recent `browser_snapshot` call)
- The SSE connection itself (SDK reconnects each session)

This is the simplest possible model — the browser is just always running. No state serialization, no reconnection logic, no profile recovery. If the server restarts (crash, deploy), Playwright relaunches Chromium with the same `user_data_dir` to recover cookies/localStorage, though open tabs and in-page state are lost.

---

## 4. Tool Design

### 4.1 Phase 1 — MVP (7 tools)

These cover the core browse-read-interact loop:

| Tool | Description | Key Arguments |
|------|-------------|---------------|
| `browser_navigate` | Navigate to a URL | `url`, `wait_until?` (load/domcontentloaded/networkidle) |
| `browser_snapshot` | Get accessibility tree of current page | `full?` (include non-interactive elements) |
| `browser_click` | Click an element by ref | `ref` (e.g. "e14"), `button?` (left/right/middle) |
| `browser_type` | Type text into an element | `ref`, `text`, `clear?` (clear field first) |
| `browser_scroll` | Scroll the page | `direction` (up/down), `amount?` (pixels) |
| `browser_press` | Press a key or key combo | `key` (e.g. "Enter", "Tab", "Ctrl+a") |
| `browser_tabs` | List, switch, open, or close tabs | `action` (list/switch/close/new), `page_id?` |

### 4.2 Phase 2 — Enhanced (+4 tools)

| Tool | Description |
|------|-------------|
| `browser_screenshot` | Take PNG screenshot, return as base64 image or file path |
| `browser_evaluate` | Execute JavaScript in page context, return result |
| `browser_fill` | Fill a form field using Playwright's `fill()` (triggers change events properly) |
| `browser_wait` | Wait for selector/navigation/text/timeout |

### 4.3 Phase 3 — Advanced (+3 tools)

| Tool | Description |
|------|-------------|
| `browser_select` | Select option(s) from a dropdown |
| `browser_drag` | Drag from one element to another |
| `browser_cookies` | Get/set/clear cookies for the current domain |

### 4.4 Element Reference System

Every `browser_snapshot` call assigns sequential ref IDs (`e1`, `e2`, `e3`...) to interactive elements. The ref map is maintained in-memory for the MCP server's lifetime (the full SDK session). Actions use these refs to target elements.

**Snapshot output format** (text-based, optimized for LLMs):
```
[Page] My Website - https://example.com
  [navigation] Main Menu
    [link e1] Home
    [link e2] Products
    [link e3] Contact
  [main]
    [heading] Welcome
    [paragraph] Some descriptive text about the page...
    [form]
      [textbox e4 "Email"] user@example.com
      [textbox e5 "Password"]
      [button e6] Sign In
    [list]
      [listitem] [link e7] First item
      [listitem] [link e8] Second item
  [footer]
    [link e9] Privacy Policy
```

**Ref lifecycle**:
- Refs are scoped to the most recent snapshot
- Calling `browser_snapshot` resets the ref map
- Page navigation invalidates refs — using a stale ref returns an error prompting a new snapshot
- Refs map to Playwright locators (not element handles) for stability

### 4.5 Interaction Workflow Example

```
1. browser_navigate("https://github.com/login")
2. browser_snapshot()
   → Returns tree with refs: e1=username, e2=password, e3=sign-in button
3. browser_type(ref="e1", text="user@example.com")
4. browser_type(ref="e2", text="password123")
5. browser_click(ref="e3")
6. browser_snapshot()
   → Returns new page (dashboard or error)
```

---

## 5. Implementation Details

### 5.1 New File: `mcp-servers/browser.py`

Follows the established MCP server pattern. Key structure:

```python
#!/usr/bin/env python3
"""Lloyd MCP Server: Browser — full browser control via Playwright.

Runs as a persistent SSE server (see doc 08). Playwright launches
Chromium once on first use; the browser stays alive across sessions.
"""

import asyncio, json, re
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

PORT = 8509
CHROME_PROFILE = "/home/alansrobotlab/lloyd/.chrome-profile"

app = Server("lloyd-browser")

# State persists for the server's lifetime (across all sessions)
_pw = None           # Playwright instance
_browser = None      # Browser instance (launched once)
_context = None      # BrowserContext (with persistent profile)
_active_page = None  # Currently focused Page
_ref_map = {}        # "e1" -> Locator
_ref_counter = 0     # Auto-incrementing ref ID

async def _ensure_browser():
    """Launch browser on first call. Relaunch if crashed."""
    global _pw, _browser, _context
    if _browser and _browser.is_connected():
        return _browser
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(headless=True)
    _context = await _browser.new_context()
    return _browser

async def _get_active_page() -> Page:
    """Get or discover the active page (tab)."""
    global _active_page
    await _ensure_browser()
    pages = _context.pages
    if not pages:
        _active_page = await _context.new_page()
    elif _active_page not in pages:
        _active_page = pages[-1]
    return _active_page

def _build_snapshot(node, depth=0) -> str:
    """Walk accessibility tree, assign refs to interactive elements."""
    global _ref_counter
    # ... recursive tree walk, format as indented text
    # Interactive roles (link, button, textbox, etc.) get ref IDs
    pass

@app.list_tools()
async def list_tools():
    return [
        Tool(name="browser_navigate", description="...", inputSchema={...}),
        Tool(name="browser_snapshot", description="...", inputSchema={...}),
        # ... all tools
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    # Dispatch to handler functions
    # All return [TextContent(type="text", text=json.dumps(result))]
    pass

# --- SSE transport (persistent server) ---
sse = SseServerTransport("/messages/")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())
    return Response()

starlette_app = Starlette(routes=[
    Route("/sse", handle_sse, methods=["GET"]),
    Mount("/messages/", app=sse.handle_post_message),
])

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="127.0.0.1", port=PORT)
```

### 5.2 Config Registration

Add to `config.yaml`:
```yaml
mcp_servers:
  browser:
    type: sse
    url: http://127.0.0.1:8509/sse
    enabled: true
    # command/args retained for process management
    command: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python
    args: ["/home/alansrobotlab/lloyd/mcp-servers/browser.py"]
```

Add to `_MCP_SERVER_META` in `server.py`:
```python
"browser": {"label": "Browser", "description": "Web browser control"},
```

### 5.3 Supervisord Entry

The browser MCP server runs as a persistent process under supervisord:

```ini
[program:lloyd-mcp-browser]
command=/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python
    /home/alansrobotlab/lloyd/mcp-servers/browser.py
autostart=true
autorestart=true
stdout_logfile=/home/alansrobotlab/lloyd/logs/mcp-browser.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/mcp-browser.err
```

Chromium is launched by Playwright inside the server process on first tool use. If the server restarts, Playwright relaunches Chromium automatically.

### 5.4 Dependencies

```bash
# In Lloyd's venv
.venvs/lloyd/bin/pip install playwright
.venvs/lloyd/bin/python -m playwright install chromium
```

Playwright bundles its own Chromium, but we can also point it at the system Chromium (`/usr/bin/chromium`) that's already installed. Using the system Chromium for the persistent instance and Playwright's CDP connector is the lightest path.

---

## 6. SSE Pipeline Changes (Phase 2)

Currently, `server.py` lines 374-380 extract tool results as text only:

```python
if isinstance(block.content, list):
    result_str = " ".join(
        getattr(c, "text", str(c)) for c in block.content
    )
```

For screenshot support, this needs to detect `ImageContent` blocks and forward them to the frontend. Options:

**Option A (recommended)**: Emit a separate SSE event for images:
```python
for c in block.content:
    if hasattr(c, "data") and hasattr(c, "mimeType"):
        yield f"event: tool_image\ndata: {json.dumps({'call_id': call_id, 'data': c.data, 'mimeType': c.mimeType})}\n\n"
```

**Option B**: Save screenshot to a static file path and include a URL in the text result. The frontend renders it as an `<img>` tag. Simpler, avoids large base64 in SSE stream.

**Frontend change**: Add image rendering to the tool result display in `ChatPanel.tsx` / tool results component.

**Note**: Phase 1 is text-only. Screenshots are a Phase 2 feature, primarily useful when Lloyd runs with Claude models (which have vision). Local Qwen models have limited vision capability, so text-based accessibility trees are the primary interface.

---

## 7. Security

### 7.1 SSRF Protection

Reuse the `_is_private_host()` pattern from `mcp-servers/http_tools.py`:

```python
_PRIVATE_IP_PATTERNS = [
    re.compile(r"^127\."), re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),
    re.compile(r"^192\.168\."), re.compile(r"^0\."),
    re.compile(r"^169\.254\."), re.compile(r"^::1$"),
    re.compile(r"^fc00:", re.IGNORECASE),
    re.compile(r"^fd", re.IGNORECASE),
    re.compile(r"^fe80:", re.IGNORECASE),
]
```

Apply in `browser_navigate` before loading the URL. Consider a config override (`browser.allow_private_urls`) for local development use cases.

### 7.2 JavaScript Execution

`browser_evaluate` (Phase 2) should:
- Timeout after 10 seconds
- Limit return value size (e.g., 50KB)
- Never execute in the context of chrome:// or devtools:// pages

### 7.3 Resource Limits

- Limit concurrent tabs (e.g., max 10) — close oldest if exceeded
- Playwright launch arg `--max-old-space-size=512` to cap Chrome memory
- `_ensure_browser()` re-launches if browser process died

### 7.4 Credential Safety

- Accessibility trees naturally mask password fields (Playwright returns `[textbox "Password"]` without the value)
- Consider redacting any detected API keys/tokens from snapshot output (follow Hermes pattern)

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Browser crashes / OOM | Open tabs and in-page state lost (cookies survive via user_data_dir) | `_ensure_browser()` relaunches; supervisord restarts the MCP server; `--max-old-space-size` flag caps memory |
| Stale element refs | Click/type targets wrong element | Clear ref map on navigation; return error "stale ref, take new snapshot" |
| Accessibility tree too large | Exceeds context window | Truncate at configurable char limit (e.g., 8000); prioritize interactive elements; offer `full=False` compact mode |
| Local model struggles with browser tools | Poor tool-calling accuracy | Detailed tool descriptions with usage examples in schema; limit tool count in Phase 1 |
| Page requires JavaScript rendering | Snapshot shows empty/incomplete page | Playwright waits for load by default; `wait_until=networkidle` option for SPAs |
| Bot detection / CAPTCHAs | Cannot interact with protected sites | Basic mitigations: realistic user agent, viewport size, webdriver flag removal. Advanced anti-bot is out of scope (would need cloud providers). |
| Distrobox display access for headed mode | Cannot see browser visually | Use `--headless=new` by default. If headed mode needed, distrobox can forward host X11/Wayland display. |

---

## 9. Phased Delivery

### Phase 0: Persistent MCP migration
**Prerequisite** — see [doc 08](08-mcp-persistent-servers.md). Migrate existing MCP servers from stdio to SSE transport so the SDK connects to persistent processes. This must land first.

### Phase 1: MVP
**Scope**: 7 tools, text-only output, persistent browser MCP server  
**Files**:
- `mcp-servers/browser.py` (new, ~300 lines, SSE transport)
- `config.yaml` (add browser server entry with `type: sse`)
- `server.py` (add `_MCP_SERVER_META` entry)
- Supervisor config (add `lloyd-mcp-browser` program)

**Validates**: Can Lloyd navigate, read, and interact with web pages using accessibility trees?

### Phase 2: Screenshots + JS
**Scope**: Add `browser_screenshot`, `browser_evaluate`, `browser_fill`, `browser_wait`  
**Files**:
- `mcp-servers/browser.py` (extend)
- `server.py` (SSE image content handling)
- `web/src/components/` (image rendering in tool results)

**Validates**: Can Lloyd take and display screenshots? Can it execute JS for data extraction?

### Phase 3: Polish
**Scope**: Add `browser_select`, `browser_drag`, `browser_cookies`. Session cleanup, bot detection, config options.  
**Files**:
- `mcp-servers/browser.py` (extend)
- `config.yaml` (browser-specific options)

**Validates**: Full browser control parity with Hermes reference implementation.

---

## 10. Decision Log

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| Browser library | Playwright Python | agent-browser CLI, raw CDP, Selenium | Native Python async, built-in a11y tree, no Node.js dep |
| Session persistence | Persistent MCP server (always running) | SDK-managed subprocess, state serialization | Browser stays alive across sessions; tabs, cookies, state all persist naturally |
| Page representation | Text accessibility tree | Screenshots-first, DOM HTML | Local Qwen models need text; vision is Phase 2 fallback |
| Element targeting | Ref IDs from snapshots | CSS selectors, XPath, pixel coords | Matches both reference impls; LLM-friendly |
| Deployment | Persistent MCP server via supervisord | Docker, SDK-spawned subprocess | Consistent with all other MCP servers; browser stays alive independently |
| Security model | SSRF block + headless | Full sandbox (Docker) | Proportionate to risk; Lloyd is a personal agent |
