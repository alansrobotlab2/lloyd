---
segment: architecture
tags: [architecture, lloyd, frontend, dashboard]
type: reference
status: implemented
date: 2026-09-19
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

Six lists name the tabs and one file keeps them equal:
`tests/test_mc_tab_parity.py`, over `Page` in
`web/src/components/Sidebar.tsx`, `mc_state.VALID_TABS`,
`mission_control_ui._VALID_TABS`, `VALID_TABS` in `useMcNavigationEvents.ts`,
and `mc_ui._SUMMARIZERS`, the per-tab brief `mc_navigate` hands back. A tab
missing from one of the first four fails in the quietest direction: the state
mirror keeps serving the previous tab and `mc_get_state` answers wrongly. A
tab absent from the summarizer registry is not an error either:
`_summarize_tab` returns `{}` and the agent learns nothing about where it just
sent the user, which is why `architecture`, `settings` and `graph` register an
explicit `lambda: {}` rather than being left out.
The sixth list is the one that decides whether the tab draws anything:
`const PAGES` in `web/src/components/Layout.tsx:79`, the map
`PageComponent = PAGES[page]` (`:315`) looks up and the `{PageComponent && …}`
guard (`:557`) renders. Its failure is the loudest-looking and the most
silent-actual: `mc_navigate` returns 200, the brief comes back, the frontend
switches `page`, and the pane stays empty — the same "nothing moves" symptom as
the hook list, one level deeper, and until #1274 `tests/test_mc_tab_parity.py`
parsed none of it. It parses the map now: the keys must satisfy
`PAGES ∪ STICKY_PAGES == Page` in both directions. The exemption is a declared
constant — `STICKY_PAGES = {chat, ide, memory}` — because those three tabs are
absent from the map on purpose: they are mounted elsewhere in `Layout.tsx` and
deliberately kept mounted there (`:391`, `:528`, `:549`) so Monaco, the LiveKit
room and the memory graph survive a tab switch. A fourth sticky tab has to be
declared in that constant *and* really mounted — the same test refuses a
declaration with no `page === '<name>'` guard behind it. Folding the three
into `PAGES` with their sticky wrappers, so the map alone is render truth and
the exception list can go away, is the follow-on this item leaves to a person.

## The dashboard

One endpoint rather than one per panel, because the page is open all day, and
`DashboardPage` polls it every `POLL_MS` (2 s). Sections and their sources are
tabled in `CLAUDE.md` § "Mission Control dashboard" — twelve of them (the
twelfth, `network`, is #628's egress destination inventory), and
`tests/test_dashboard_doc_claims.py` asserts that table's section names
against the `_gather(...)` call in `app/routers/dashboard.py`, so the count
here and the table are one fact a run can re-measure. Rules that keep the
endpoint honest:

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
  blanks the very page this design exists to keep up. Every `ErrorPanel` in
  `DashboardPage.tsx` derives its message that way, and
  `tests/test_dashboard_doc_claims.py` fails if a raw `error={x.error}`
  dereference or a `!== undefined` wrapper around a section comes back —
  the rule is held by that test, not by prose.
- the state the agent reads is now credentialed: `GET :8500/state` and
  `POST :8500/browser/navigate` refuse a request without the aggregator's
  boot credential, and both the URL and the header come from
  `app/aggregator_config.py` (`route` / `auth_headers_for`) rather than from
  a hand-written origin (#1053).

## Sessions, titles, activity

`app/sessions_io.py` is the writer for every session that goes through it
(`create_session`) and the one definition of "is a human reading this"
(`is_user_session`;
`NON_USER_PLATFORMS = {autonomy, worker}`). It is not quite the *only* writer:
`POST /api/sessions/create` mints its stub JSON itself (`app/routers/sessions.py:612`)
with a bare `write_text` and a field set missing `id`, `title` and `source`, so
a pre-created Inner Voice session is the third session shape on disk — filed as
#1275. Titles come from the secondary
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

**mTLS was dropped on 2026-06-14** because iOS Chrome and the other
third-party iOS browsers cannot present keychain identities for mutual TLS —
only Safari can — so Vite no longer requests a client cert and any browser on
the tailnet works. Tailscale was then documented as the access boundary while
nothing on the backend enforced it: `server._require_client_cert` checked the
per-device allowlist *only* when `x-client-fingerprint` was present, so a
request that carried nothing — the whole LAN, since the backend binds
`0.0.0.0:8080` — went straight through. Since 2026-09-20 it fails closed. The
middleware now refuses every `/api/*` request with 403 unless the ASGI peer is
loopback or inside `server.trusted_networks` (default Tailscale's CGNAT
`100.64.0.0/10`), and reads no header to decide, because uvicorn's
`proxy_headers` plus Vite's `xfwd: true` mean `X-Forwarded-For` rewrites the
peer rather than evidencing it. `/health` and `/health/deep` are the only
pre-auth paths, matched exactly. The allowlist still runs after that for a peer
already inside the gate, so an unknown or revoked fingerprint is still refused
and revocation still works; what a fingerprint can no longer do is buy a client
network reach it did not have. Loopback (`127.0.0.1`, `::1`) stays trusted —
the LiveKit worker, the autonomy ticker and the self-mod promoter POST straight
to `:8080` without crossing Vite's TLS layer at all — and the self-mod drain
keeps its own loopback-only guard on top of the network rule, so a tailnet
browser still cannot arm a drain. Binding the backend to `127.0.0.1` as well
(#683's open half) remains a person's edit: `config.yaml` is boot-read-only.
`ApiPeerGate` is a plain ASGI middleware rather than
`@app.middleware("http")`, so the IDE tab's language-server socket is covered
too: `@router.websocket("/api/lsp/{language}")` is an `/api/*` path an
HTTP-scope decorator cannot see at all, and a refused upgrade is closed before
it is accepted. See [[infrastructure]].

## Related

[[harness]], [[background-runs]], [[workers]], [[inner-voice]].

## Review log

- 2026-09-20 — **current for Remote access.** #683 closed the fail-open that
  section described, so its prose is rewritten rather than annotated. What was
  re-read rather than inherited: `clientCertHeaders()` still injects
  `x-client-cn`/`x-client-fingerprint` from the verified TLS peer and still
  proxies `/api` with `xfwd: true` (`web/vite.config.ts`); uvicorn 0.44.0
  defaults `proxy_headers=True` and `forwarded_allow_ips="127.0.0.1"`, which is
  the mechanism by which the rewrite reaches `request.client`; the drain's
  loopback-only guard is `app/routers/automod.py::_is_loopback`, asserted
  present by `tests/test_automod_hardening.py`. `config.yaml` still binds
  `0.0.0.0` — unchanged by this round, still Alan's to set. The 2026-09-19 pass
  below holds for every other section.
- 2026-09-19 — **current.** Checked every path, line number, route, count and
  cadence against `c55a501` and the live `:8080` routes; the tab lists, the
  dashboard TTLs and section set, the `hold_reason`/`_is_task_due` mirror and
  the whole SSRF section (including navigate's remaining landing hole) hold.
  Corrected three things the prose asserted that the tree does not: that
  `sessions_io` is the only session writer, that no sixth tab list exists, and
  that the `sectionOk` rule is in force. Filed #1273 #1274 #1275.
