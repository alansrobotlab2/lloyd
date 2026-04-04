# Browser Control: Analysis & Implementation Gameplan

> **Date**: 2026-04-04  
> **Status**: Implemented (all phases)  
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

3. **Simpler dependency chain.** One `pip install playwright` into the existing venv, using the system Chromium (`/usr/bin/chromium`) already installed in the lloyd distrobox. No npm packages, no Node.js runtime dependency, no Playwright-bundled browser download.

4. **Python accessibility API.** Playwright Python exposes `page.locator("body").aria_snapshot()` for YAML-like accessibility trees that local Qwen models need. (Note: `page.accessibility.snapshot()` was removed in Playwright 1.47+; `aria_snapshot()` lives on `Locator`, not `Page`.)

5. **First-class async.** Lloyd's MCP servers are async Python. Playwright's async API (`async_playwright`) integrates naturally without thread pools or subprocess overhead.

### 3.2 Session Lifecycle

With the persistent MCP server architecture (see [doc 08](08-mcp-persistent-servers.md)), MCP servers run as long-lived processes exposing an SSE endpoint. The SDK connects to them as a client — it does not spawn or kill them. This means the browser MCP server and its Chromium instance stay alive indefinitely.

```
supervisord
├── lloyd-backend (server.py)
├── lloyd-frontend (vite)
└── lloyd-mcp (mcp_server/main.py on port 8500)
    ├── all other tool modules (memory, backlog, http_tools, etc.)
    └── browser module → Playwright Chromium (launched on first use, stays alive)
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

**Snapshot output format** (Playwright `aria_snapshot()` YAML annotated with refs):
```
[Page] My Website — https://example.com
- navigation
  - link "Home" [e1]
  - link "Products" [e2]
  - link "Contact" [e3]
- main
  - heading "Welcome" [level=1]
  - paragraph: Some descriptive text about the page...
  - form
    - textbox "Email" [e4]: user@example.com
    - textbox "Password" [e5]
    - button "Sign In" [e6]
  - list
    - listitem
      - link "First item" [e7]
    - listitem
      - link "Second item" [e8]
- contentinfo
  - link "Privacy Policy" [e9]
```

**Ref lifecycle**:
- Refs are scoped to the most recent snapshot
- Calling `browser_snapshot` resets the ref map
- Page navigation clears the ref map — using a stale ref returns an error prompting a new snapshot
- Refs are reconstructed as Playwright locators via `page.get_by_role(role, name=name, exact=True).nth(occurrence)` — only named interactive elements get refs

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

### 5.1 Implemented: `mcp_server/browser.py`

The browser module lives inside the unified MCP server (`mcp_server/main.py`), not as a standalone SSE server. It follows the same pattern as `http_tools.py` and other modules — exporting `list_tools()` and `call_tool()` functions that `main.py` dispatches to.

Key implementation details:

- **System Chromium**: Uses `/usr/bin/chromium` (already installed in the lloyd distrobox) via `executable_path=` — no Playwright-bundled browser needed.
- **Headed mode**: Runs with `headless=False` so the browser is visible. The `lloyd-mcp` supervisord config passes `DISPLAY=:1` and `WAYLAND_DISPLAY=wayland-1` for host display forwarding.
- **Accessibility snapshots**: Uses `page.locator("body").aria_snapshot()` which returns a YAML-like string. A regex parser (`_ARIA_LINE_RE`) walks the output and injects `[eN]` ref IDs next to interactive elements.
- **Ref map**: `_ref_map` stores `{ref_id: {role, name, occurrence}}`. Locators are reconstructed via `page.get_by_role(role, name=name, exact=True).nth(occurrence)`.
- **No SSRF protection**: Removed since Lloyd is a personal agent that needs local network access.
- **No separate process**: The browser module runs inside the existing `lloyd-mcp` process (port 8500), not as a separate SSE server on port 8509 as originally proposed.

### 5.2 Config Registration

No separate `config.yaml` entry needed — the browser module runs inside the existing `lloyd` unified MCP server. The `_MCP_SERVER_META` description in `server.py` was updated to include "browser" in the tool list.

### 5.3 Supervisord Entry

No separate supervisord entry needed — the browser module runs inside `lloyd-mcp`. The existing config was updated with display env vars for headed mode:

```ini
[program:lloyd-mcp]
command=/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m mcp_server.main
directory=/home/alansrobotlab/lloyd
environment=DISPLAY=":1",WAYLAND_DISPLAY="wayland-1",XDG_RUNTIME_DIR="/run/user/1000"
autostart=true
autorestart=true
stdout_logfile=/home/alansrobotlab/lloyd/logs/mcp.log
stderr_logfile=/home/alansrobotlab/lloyd/logs/mcp.err
```

Chromium is launched by Playwright inside the server process on first tool use. If the server restarts, Playwright relaunches Chromium automatically.

### 5.4 Dependencies

```bash
# In Lloyd's venv — only pip install needed, no browser download
.venvs/lloyd/bin/python -m pip install playwright
```

Uses system Chromium at `/usr/bin/chromium` (v146, already in the lloyd distrobox) via Playwright's `executable_path=` parameter. No `playwright install chromium` needed.

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

**Decision: Removed.** Lloyd is a personal agent that needs local network access (e.g. navigating to `http://192.168.50.108:5173/` for Mission Control). The SSRF patterns from `http_tools.py` were initially included but removed after testing. Only the `http/https` scheme check remains.

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
| Distrobox display access for headed mode | Cannot see browser visually | Solved: supervisord config passes `DISPLAY=:1` and `WAYLAND_DISPLAY=wayland-1` env vars. Browser runs headed by default. |

---

## 9. Phased Delivery

### Phase 0: Persistent MCP migration
**Status**: Complete — see [doc 08](08-mcp-persistent-servers.md).

### Phase 1: MVP — COMPLETE
**Scope**: 7 tools, text-only output  
**Files changed**:
- `mcp_server/browser.py` (new, ~380 lines, integrated into unified server)
- `mcp_server/main.py` (added browser to imports and MODULES)
- `server.py` (updated `_MCP_SERVER_META` description)
- `requirements.txt` (added `playwright>=1.58.0`)
- Supervisor config (added display env vars to `lloyd-mcp`)

### Phase 2: Screenshots + JS — COMPLETE
**Scope**: `browser_screenshot`, `browser_evaluate`, `browser_fill`, `browser_wait`  
Implemented in same `mcp_server/browser.py`. Screenshots return base64 PNG and save to `logs/screenshots/`. SSE image content handling (server.py + frontend) deferred — not needed for local Qwen models which use text-based snapshots.

### Phase 3: Polish — COMPLETE
**Scope**: `browser_select`, `browser_drag`, `browser_cookies`  
Implemented in same `mcp_server/browser.py`. All 14 tools shipped in a single pass.

---

## 10. Decision Log

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| Browser library | Playwright Python 1.58 | agent-browser CLI, raw CDP, Selenium | Native Python async, built-in a11y tree, no Node.js dep |
| Browser binary | System Chromium v146 (`/usr/bin/chromium`) | Playwright-bundled, downloaded browser | Already installed in distrobox; no download needed |
| Session persistence | Unified MCP server (always running) | SDK-managed subprocess, state serialization | Browser stays alive across sessions; tabs, cookies, state all persist naturally |
| Page representation | `aria_snapshot()` YAML with injected refs | `page.accessibility.snapshot()` dict, screenshots-first | `page.accessibility` removed in PW 1.47+; `aria_snapshot()` on Locator is the current API |
| Element targeting | Ref IDs from snapshots | CSS selectors, XPath, pixel coords | Matches both reference impls; LLM-friendly |
| Deployment | Module inside unified `lloyd-mcp` server | Separate SSE server, Docker, SDK-spawned subprocess | Simpler than a separate process; browser shares the existing port 8500 |
| Display mode | Headed (`headless=False`) | Headless by default | Visible browser is useful for debugging and user verification |
| Security model | Scheme check only (no SSRF block) | Full sandbox (Docker) | Lloyd is a personal agent that needs local network access |
