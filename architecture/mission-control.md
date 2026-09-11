---
segment: architecture
tags: [architecture, lloyd, frontend, dashboard]
type: reference
status: implemented
date: 2026-09-11
---

# Mission Control

The React (Vite + Tailwind) frontend in `web/`, served by the `lloyd-frontend`
dev server on `:5173` — which proxies `/api` to the backend on `:8080`, that
direction and not the other one: nothing in `server.py` mounts or serves the
frontend, so a dead Vite is a dead UI against a perfectly healthy API. Sixteen
tabs in the `Page` union, fifteen of them buttons on the one sidebar, each
backed by a router under `app/routers/`.

| Tab | Backed by | Notes |
|---|---|---|
| `dashboard` | `GET /api/dashboard` (`app/routers/dashboard.py`) | one aggregated snapshot every 2 s; sections degrade independently |
| `chat` | `/api/message/stream` (SSE), `/api/sessions` | the primary conversation; thinking rows, tool bubbles with captions |
| `background` | `GET /api/background/sessions`, `GET /api/workers/health` | every session the machine ran for itself, apart from chat history |
| `inner_voice` | `app/routers/inner_voice.py` | the observer's timeline for any IV-enabled session |
| `workers` | `app/routers/workers.py` | queue depth, slots, runs, pause/enable, pending review |
| `autonomy` | `app/routers/autonomy.py` | the scheduled fleet, overdue vs held |
| `backlog` | `app/routers/backlog.py` | the markdown kanban in `~/obsidian/backlog/` |
| `memory`, `graph` | `app/routers/memory.py`, `entities.py` | facts, and the entity graph — which renders *inside* `memory`; the `graph` tab itself is still a stub page, with no sidebar button, reachable only by `mc_navigate` |
| `skills`, `tools` | `skills.py`, `tools.py` | vault skills; tool toggles persisted to `data/tool_overrides.yaml` |
| `services` | `services.py` → `app/supervisor_client.py` | supervisord status |
| `settings` | `system.py`, `models.py` | config view, model identity |
| `architecture` | `architecture.py` | a **source import-graph browser** for `.py`/`.ts`, not these docs |
| `ide` | `ide.py`, `lsp.py` | file view with LSP diagnostics |
| `browser` | `browser.py` | mirrors the agent's Chromium; the URL bar is the one control |

Four lists name the tabs and `tests/test_mc_tab_parity.py` keeps them equal:
`Page` in `web/src/components/Sidebar.tsx`, `mc_state.VALID_TABS`,
`mission_control_ui._VALID_TABS`, and `VALID_TABS` in
`useMcNavigationEvents.ts`. A tab missing from one of them fails in the
quietest direction: the state mirror keeps serving the previous tab and
`mc_get_state` answers wrongly. The same test pins a fifth list that fails
quieter still — `mc_ui._SUMMARIZERS`, the per-tab brief `mc_navigate` hands
back. A tab absent from it is not an error either: `_summarize_tab` returns
`{}` and the agent learns nothing about where it just sent the user, which is
why `architecture`, `settings` and `graph` register an explicit `lambda: {}`
rather than being left out.

## The dashboard

One endpoint rather than one per panel, because the page is open all day, and
`DashboardPage` polls it every `POLL_MS` (2 s). Sections and their sources are
tabled in `CLAUDE.md` § "Mission Control dashboard" — eleven of them today;
that table predates the `automod` scorecard section and still does not list
it. Rules that keep the endpoint honest:

- the expensive sections are cached and the live ones never are: the vault
  walks (`autonomy`, `backlog`) at `_VAULT_SCAN_TTL_S` 10 s, the
  session-directory scan at its own `_RECENT_TTL_S` 10 s — `recent` reads
  `sessions/`, not the vault — and the automod scorecard at
  `_SCORECARD_TTL_S` 60 s, since its numbers move per round, not per poll;
- the recent-chats scan opens only the newest files by mtime and stops at
  `_RECENT_KEPT` (8) user rows or `_RECENT_CEILING` (400) files opened —
  background sessions (four-part ids) are skipped unread, and `_RECENT_SHOWN`
  (2) survive the live filter, which is applied *outside* the cache so a chat
  that just started never reads as finished for ten seconds;
- "overdue" and "held" are different words: `autonomy.hold_reason` mirrors
  `_is_task_due`'s six gates in their order — skill, `up_next`, frequency,
  failure cooldown, `depends_on`, `preferred_hours` — and a past-due task is
  overdue only when nothing holds it. `classifier` reports `naive` when
  `autonomy` could not be imported and every past-due task is being called
  overdue, because a downgrade that looks like success is the failure this
  split exists to prevent;
- subagents and background bash live in the lloyd-mcp process, read over
  loopback from `GET :8500/state` — which also carries the tsc runner's and
  the change ledger's stats;
- a section can be *missing*, not merely failed, after a backend restart:
  test with `sectionOk(section)` and render `sectionError(section)` from
  `api.ts`, never `section.error`, which throws on an undefined section and
  blanks the very page this design exists to keep up.

## Sessions, titles, activity

`app/sessions_io.py` is the one writer for every session (`create_session`)
and the one definition of "is a human reading this" (`is_user_session`;
`NON_USER_PLATFORMS = {autonomy, worker}`). Titles come from the secondary
on a geometric schedule (`app/session_titles.py`) and every surface shares
the fallback chain in `web/src/lib/sessionLabel.ts`. The live activity line is
six kinds — `starting → prefill → thinking`/`responding` → `tool` →
`working`, worded for the screen by `ACTIVITY_TEXT` in the same file — stamped
by the turn runner through `sessions_io.set_turn_activity`. It is display
state the loop never reads back, so an unchanged state is dropped and the
streaming path can call it per token; it is read off the pure in-memory queue
snapshot, which is also the automod promoter's idle gate.

## The agent's view of the UI

`agent_mcp/mission_control_ui.py` lets a turn navigate the user
(`mc_navigate`) and read where they are (`mc_get_state`); the frontend
reports through `POST /api/mc/state`. `_summarize_browser` never carries the
screenshot or the accessibility tree — the summary goes into the model's
context on every move.

## The Browser tab, and the guard that was never called

The panel mirrors the agent's Chromium, and its **URL bar** is the one
control: `POST /api/browser/navigate`, which the backend proxies to the
aggregator's own `/browser/navigate` — Playwright runs in that process, the
same seam the dashboard crosses to read `/state`. Deliberately not an MCP
call, because the user typing a URL is not the agent using a tool, and
dispatching it as one would write a `browser_navigate` into the transcript
that the model never made. The frame it pushes is tagged `url_bar` so the tab
can say who drove it.

`_is_private_host` was defined on 2026-04-11 and called from nowhere until
2026-09-09 (`b7e2f69`, "wire the SSRF guard that was never called"). #278's
acceptance required that the check at `agent_mcp/browser.py:50` be "intact",
its triage verdict called that line the only network-shaped code in the
module, and the test written to satisfy both asserted the predicate returned
the right booleans. All three statements were true and none of them was about
a guard. **An acceptance criterion that names a symbol at a line number
certifies only that the symbol is still there.**

The policy mirrors `http_tools.http_request`: block the network the machine is
on, allow the machine itself. Loopback stays reachable because the agent
browses Lloyd's own dashboard, and an injected prompt that wants loopback
holds Bash already; the LAN is what this closes — the router, the NAS, the
printer. The check is on **resolved** addresses (`_resolve_addrs`,
`lru_cache`d, which also blunts DNS rebinding by holding the first answer), so
`2130706433` and `::ffff:127.0.0.1` are classified the way the resolver sees
them and a name like `router.local` is classified at all. Route interception
does not see redirects — `route.continue_()` hands the request to Chromium,
which follows a 3xx internally — so `_enforce_landing` asks where the page
actually ended up and blanks it on a hit, which needs no enumeration of lanes
and so also catches a meta-refresh. `_browser_snapshot` and `_browser_evaluate`
call it; `_capture_state` checks directly, or a screenshot of a LAN device
would be pushed to Mission Control. `browser.block_private_hosts: false` or
`LLOYD_BROWSER_BLOCK_PRIVATE=0` turns it off, the env winning.
`tests/test_browser_panel.py` asserts the entry points *invoke* it, which is
the property that was missing.

## Remote access

Vite serves HTTPS from `agent-services/cert/`, preferring the publicly trusted
Tailscale cert for `goliath.taile37041.ts.net` when it has been provisioned
and falling back to the private `lloyd.crt` signed by the CA that
`scripts/gen-cert.sh` makes once; `scripts/mint-client-cert.sh <device>`
enrolls a device into `clients.json`.

**mTLS was dropped on 2026-06-14** and Tailscale is the access boundary now:
iOS Chrome and the other third-party iOS browsers cannot present keychain
identities for mutual TLS — only Safari can — so Vite no longer requests a
client cert, and any browser on the tailnet works. The per-device allowlist
did not go away, it went optional. Vite still injects the peer cert's CN and
sha256 fingerprint as `x-client-cn` / `x-client-fingerprint` when a cert *is*
presented, and `server._require_client_cert` still refuses an unknown
fingerprint on `/api/*` so revocation keeps working for cert-bearing clients;
it simply no longer rejects a request that carries none. Loopback
(`127.0.0.1`, `::1`) bypasses it outright, because the LiveKit worker and the
autonomy ticker POST straight to `:8080` without crossing Vite's TLS layer at
all. See [[infrastructure]].

## Related

[[harness]], [[background-runs]], [[workers]], [[inner-voice]].
