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
| `chat` | `/api/message/stream` (SSE, read with `fetch` + `ReadableStream`, not `EventSource`), `/api/sessions` | the primary conversation; thinking rows, tool bubbles with captions |
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

The `dashboard` tab is first in the sidebar and the desktop landing tab.
One endpoint rather than one per panel, because the page is open all day —
eight requests per tick times however many tabs are open is real load on a box
whose job is holding a 262k-token KV cache steady — and `DashboardPage` polls
it every `POLL_MS` (2 s). Sections are gathered concurrently and **degrade
independently**: a wedged supervisord turns one panel into an error string and
leaves the rest live. A dashboard is most useful when something is broken, so
it must not be the second thing to break.

The sections and their sources — twelve of them (the twelfth, `network`, is
#628's egress destination inventory) — are the table below. CLAUDE.md
§ "Mission Control dashboard" carries a copy with shortened source cells, and
`tests/test_dashboard_doc_claims.py` asserts that copy's section names (and a
non-empty source for each, `_automod` in `dashboard.py` for `automod`) against
the `_gather(...)` call in `app/routers/dashboard.py`, so the count here and the
table are one fact a run can re-measure.

| Section | Source |
|---|---|
| `host` | `app/host_metrics.py` — psutil + `nvidia-smi` (2s cache) |
| `vllm` | `app/vllm_metrics.py` — scrapes `<base_url>/metrics` per configured model |
| `primary` | `sessions_io.active_sessions_snapshot()` + `session_titles` |
| `recent` | the last chats to stop talking — bounded scan of `sessions/` |
| `agents` | **the lloyd-mcp process**, over loopback (`GET :8500/state`) |
| `services` | `app/supervisor_client.py` |
| `workers` | `workers.queue` + `workers.pool` — pool slots, per-source depth, recent runs |
| `autonomy` | `~/obsidian/autonomy/*.md` frontmatter + the pool's in-flight `scheduled-task` jobs |
| `backlog` | `~/obsidian/backlog/*.md` frontmatter |
| `automod` | `app/routers/dashboard.py::_automod` — the loop's scorecard (`scripts/automod/scorecard.py`) over the last 7 days plus its live round state, cached at `_SCORECARD_TTL_S` |
| `network` | `agent_mcp/egress.py::network_report` — where `http_fetch`/`http_request`/`http_search`/`browser_navigate` went over 7 days, per destination and per scope (#628; the table is `egress_events` in `workers.db`) |
| `usage` | `usage_store` |

Rules that keep the endpoint honest:

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

### The traps behind those rules

Moved here from CLAUDE.md on 2026-09-25, when that file went back to being an
index; the wording is the incident record, kept whole.

**`recent` is the cached section with a trap.** A session JSON carries its
whole transcript (100 files, 7.5 MB when this was written), so the scan is
bounded twice: only the newest `_RECENT_CANDIDATES` files by **mtime** are
opened, and the parse is cached for 10 s. The mtime window is safe only because
mtime is never *earlier* than `last_active` — background writers (the titler,
post-session capture, TodoWrite) push a file's mtime later than its last real
message, so mtime can promote a stale chat but never demote a fresh one out of
the window. The rows are then sorted on `last_active`, which is what
`GET /api/sessions` sorts on too. The live filter — dropping sessions with a
running or queued turn, which the panel beside it already shows — is applied
outside that cache: cache the expensive scan, never the cheap freshness.

**Overdue is not "next up."** `_autonomy` splits scheduled tasks on `next_run`
vs now and returns them as separate lists. Sorting them together ascending and
labelling the head "next up" is how a fleet whose ticker is months behind
renders as a healthy schedule — the most overdue task lands exactly where the
soonest one belongs. Likewise `completed` is excluded from worker "open" counts
(`_OPEN_STATES`): it dominates the depth table and would bury the handful of
items actually waiting.

**And overdue is not "held."** `hold_reason` returns the first gate that bites
(`"paused"`, `"waiting on #42"`, `"outside hours 00-04,23"`, `"no skill"`) or
`None`, and both `_autonomy` and `GET /api/autonomy/tasks` call that one
function rather than restating the gates — a second private definition of
"due" is what this fixed. On 2026-09-06 the dashboard showed six overdue while
the scheduler considered none of them late: four nightly jobs outside their
window and two paused. A nightly task is past due for the eighteen hours a day
it is not allowed to run, so the counter was never zero and therefore said
nothing. The dependency gate resolves `depends_on` against whatever set it is
handed, and since #558 an id with no task file behind it, or an upstream
dispatch would not run, is *not met* — so it must always be handed the
**whole** board. Handed a status-filtered list
(`/api/autonomy/tasks?status=up_next`) the gate cannot see an upstream that is
`paused`, `in_progress` or `failed`, so every such row reads as `waiting on #N`
and the board invents a hold that does not exist.

**Front matter is bounded by its closing `---`, not by a byte count.**
`_frontmatter` reads in 4 KB chunks up to a 64 KB ceiling and stops at a
line-anchored `^---$`. The previous flat 3000-byte prefix silently dropped five
backlog items, and the selection was causal rather than random: an item grows
its `activity_log` precisely by being worked on, so the two it hid were the two
that were `in_progress` — the board reported zero. A cap that hides whatever is
most active is the worst possible reading of "bounded". Splitting on bare
`"---"` is the matching trap: it also fires inside quoted log prose and
truncates the block somewhere plausible. A block that parses to a list or a
string returns `{}`, since the caller's first move is `.get`.

**The aggregator owns the agent-side panels.** It owns the `Task` tool and
spawns `Bash(run_in_background=true)` children, so the backend has no handle on
either; `agent_mcp/main.py` exposes `GET :8500/state` beside `/health`, and
adding a new agent-side live panel means extending that route, not the backend.
`background_tasks` there carries `active` **and** `recent`. `list_active`
filters on `status == "running"`, so before that a background bash left the
dashboard the instant it exited — a task that died three seconds in was
indistinguishable from one that never started. `list_recent` is bounded by its
limit rather than by eviction: `_records` is kept whole so a later
`get(task_id)` can still hand the model an output path to Read. A finished
row's `elapsed_s` is measured against `finished_at`, not `now`, or a task that
ran for two seconds reads as hours old by evening.

**Workers are not in that panel.** The worker pool lives in the backend
(`workers.queue` + `workers.pool`, rendered by `WorkersPanel`). A worker job
whose prompt calls `Task` does put subagent rows in the agents panel — via
`workers/sources/_common.py::run_prompt_on_primary` — but anonymously: nothing
on the row says which worker source it came from.

**A subagent row opens before the run.** `agent_mcp/_subagent_registry.py`
opens it before the Task run loop starts — a `Task` blocks its caller for
minutes, so a row created on completion would only ever describe runs that no
longer need watching. Closing it is the subtle part: `finish` is idempotent and
first-writer-wins, so a blanket `finally: finish("cancelled")` runs *before* the
success path and silently stamps every completed run cancelled. Each exit path
closes the row with its own real status; `tests/test_task_registry_wiring.py`
pins that.

**Not every engine is vLLM.** A llama.cpp slot publishes `llamacpp:`
Prometheus names; `vllm_metrics._translate_llamacpp` renames them into the vLLM
vocabulary so one snapshot path and one dashboard card serve both, reports KV
occupancy and TTFT as `None` rather than `0`, treats a reachable server as
`awake`, and names the model from a `/props` probe cached per engine lifetime —
[[infrastructure]] § "The secondary is single-tenant by design" is the long
version.

**Counters vs. gauges.** vLLM exposes both. Gauges (`num_requests_running`,
`kv_cache_usage_perc`) are read straight. Counters (`prompt_tokens_total`,
`prefix_cache_hits_total`) are monotonic since engine boot and their absolute
value says nothing useful, so `vllm_metrics` keeps the previous scrape per
engine and reports a rate. A counter that goes backwards (engine restarted)
yields `None`, never a number — otherwise a restart renders as a one-second
spike of the engine's entire history. An unreachable engine drops its baseline
for the same reason.

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

The long version, moved from CLAUDE.md on 2026-09-25:

- **Every surface that names a session renders its title** — the chat history
  list, the chat header, the dashboard's agent panel, the Inner Voice picker —
  and the timestamp id survives as the element's `title=` tooltip.
  `app/session_titles.py` owns it end to end.
- **Titles are written off the primary's path.** `_sync_secondary_title`
  (`app/secondary_models.py`) is fired and forgotten off turn completion
  beside `_post_session_capture`. It was written for the single-tenant
  llama.cpp secondary (`--parallel 1`), where agent turns already queued;
  since `secondary_enabled: false` (2026-09-20) `secondary_models` routes it to
  the primary. Either way `should_title` re-titles on a **geometric** schedule
  — after the 1st real user message, then the 3rd, the 9th, the 27th —
  recorded in `title_at_count`, because a per-turn title call would put a model
  call in a shared queue for a label nobody asked to be refreshed.
- **`clean_title` is strict on purpose and `""` is a normal outcome.** A bad
  title is worse than none: the id at least identifies the row, while
  `Here is a title for the conversation` just looks like a bug. Consumers share
  one fallback chain — `web/src/lib/sessionLabel.ts`, title → preview → id — so
  a session never reads as two different sessions in two panels.
- **`title_for` caches on a TTL, not on mtime.** The session JSON is rewritten
  on every appended message, so an mtime-keyed cache would re-parse a
  multi-megabyte transcript on every 2-second dashboard poll — the exact cost
  the cache exists to avoid. `invalidate` closes the staleness window when a
  title is written.
- **Live activity** is `SessionTurn.activity` (`{kind, label, detail, at}`),
  surfaced through `active_sessions_snapshot`. "Busy" is equally true of a turn
  prefilling 160k tokens, one four minutes into a `Bash` build, and one wedged
  on a dead engine; this line is what tells them apart. Writing it is a no-op
  when nothing is running and again when the state is unchanged.
- **The snapshot stays pure in-memory queue state** — as the promoter's idle
  gate, a disk read there would put the filesystem in front of a restart
  decision. Titles are joined on in `_primary_state`, off the loop via
  `asyncio.to_thread`.
- **Each row on the agent panel opens the session in the Inner Voice tab**,
  through the same `setPendingFocus` + `setCurrentTab` pair the agent's
  `mc_navigate` uses, so `InnerVoicePage` never has to know who asked. Two
  things that panel taught:
  - *A page that applies incoming focus must not race its own list fetch.*
    `loadSessions` used to read `selectedSession` out of its closure to decide
    whether to default to the newest session. On mount that closure captures
    `null`, the fetch resolves *after* the focus has been applied, and the
    stale `null` overwrites it — so every row on the dashboard opened the same
    chat. Use the functional updater
    (`setSelectedSession(prev => prev ?? list[0].session_id)`) and keep the
    callback's deps empty; anything else reintroduces the race.
  - *The Inner Voice picker holds only IV-enabled sessions*, but focus can
    point anywhere. A `Select` whose value matches no option renders an empty
    trigger, so the picker carries an out-of-list selection in as its own
    option and names it from `/api/sessions/{id}/meta`.

## The agent's view of the UI

`agent_mcp/mission_control_ui.py` lets a turn navigate the user
(`mc_navigate`) and read where they are (`mc_get_state`); the frontend
reports through `POST /api/mc/state`. `_summarize_browser` never carries the
screenshot or the accessibility tree — the summary goes into the model's
context on every move. `screenshot_b64` and `snapshot` are ~124 KB of base64
plus 8 KB of accessibility tree; it reads
`browser_router.latest_frame_summary()`, which exists to make leaving them out
the default rather than a thing each caller remembers.

Why the tab lists are tested (the incident, from CLAUDE.md): until
`tests/test_mc_tab_parity.py` nothing made them agree, and drift is silent in
the worst direction. `browser` was in the `Page` union and in none of the other
three, so a user sitting on that tab made `POST /api/mc/state` return 400 — and
`useMcStateSync` swallows the failure *after* recording the payload as sent.
The mirror kept serving whichever tab they came from, so `mc_get_state`
answered confidently and **wrongly** for as long as they stayed there; not
"Lloyd doesn't know", which he could have said. `dashboard` was missing from
two of the three, the quieter half: the backend had carried a
`_summarize_dashboard` all along for a tab the agent was refused and the
frontend would have ignored.

## The Browser tab, and the guard that was never called

The panel mirrors the agent's Chromium (`app/routers/browser.py` + the frames
`agent_mcp/browser.py` pushes), and its **URL bar** is the one
control: `POST /api/browser/navigate`, which the backend proxies to the
aggregator's own `/browser/navigate` — Playwright runs in that process, the
same seam the dashboard crosses to read `/state`. Deliberately not an MCP
call, because the user typing a URL is not the agent using a tool, and
dispatching it as one would write a `browser_navigate` into the transcript
that the model never made. The frame it pushes is tagged `url_bar` so the tab
can say who drove it. `navigate_from_ui` completes a scheme-less host to
`https://`, detecting the scheme by `://` rather than by the bare colon — the
colon alone reads the whole of `localhost:8080` as a scheme, and on this box
that is the first thing anybody types. `browser_navigate` stays strict, because
an agent omitting the scheme has made a mistake worth seeing. Everything else
on the page stays read-only, and not for want of a route: the ref overlay has
nothing to send, since the tool surface has no "click pixel (x,y)" and the a11y
tree is gone by render time.

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
`2130706433`, `0x7f000001` and `::ffff:127.0.0.1` are classified the way
`getaddrinfo` — and Chromium — sees them (all 127.0.0.1), which is every
encoding that beat the old regex list, and a name like `router.local` is
classified at all. The policy mirrors `http_request` rather than `http_fetch`,
and the browser is not the weak link for loopback: an injected prompt that
wants it has Bash and the whole MCP surface already. Route interception
does not see redirects — `route.continue_()` hands the request to Chromium,
which follows a 3xx internally — and that is measured, not assumed: against a
loopback server that 302s to this box's own LAN address the handler fires
exactly once, for the first hop, while the redirected request arrives only as a
`request` event. So the interceptor covers a clicked link and a subresource,
and `_enforce_landing` asks where the page actually ended up and blanks it on a
hit, which needs no enumeration of lanes and so also catches a meta-refresh and
a `window.location`; leaving the page parked would let the next
`browser_snapshot` read it and the state mirror push a screenshot of it. `_browser_snapshot` and `_browser_evaluate`
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

[[harness]], [[background-runs]], [[workers]], [[inner-voice]], [[desktop]] (the Desktop tab and its lease), [[infrastructure]] (model slots, llama.cpp metrics).

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
