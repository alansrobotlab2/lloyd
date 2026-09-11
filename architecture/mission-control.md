---
segment: architecture
tags: [architecture, lloyd, frontend, dashboard]
type: reference
status: implemented
date: 2026-09-11
---

# Mission Control

The React (Vite + Tailwind) frontend in `web/`, served by the `lloyd-frontend`
dev server and proxied through the backend on `:8080`. One sidebar, sixteen
tabs, every one backed by a router under `app/routers/`.

| Tab | Backed by | Notes |
|---|---|---|
| `dashboard` | `GET /api/dashboard` (`app/routers/dashboard.py`) | one aggregated snapshot every 2 s; sections degrade independently |
| `chat` | `/api/message/stream` (SSE), `/api/sessions` | the primary conversation; thinking rows, tool bubbles with captions |
| `background` | `GET /api/background/sessions`, `GET /api/workers/health` | every session the machine ran for itself, apart from chat history |
| `inner_voice` | `app/routers/inner_voice.py` | the observer's timeline for any IV-enabled session |
| `workers` | `app/routers/workers.py` | queue depth, slots, runs, pause/enable, pending review |
| `autonomy` | `app/routers/autonomy.py` | the scheduled fleet, overdue vs held |
| `backlog` | `app/routers/backlog.py` | the markdown kanban in `~/obsidian/backlog/` |
| `memory`, `graph` | `app/routers/memory.py`, `entities.py` | facts and the knowledge graph |
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
`mc_get_state` answers wrongly.

## The dashboard

One endpoint rather than one per panel, because the page is open all day.
Sections and their sources are tabled in `CLAUDE.md` § "Mission Control
dashboard". Rules that keep it honest:

- vault-walking sections (`autonomy`, `backlog`, `recent`) are TTL-cached
  10 s; live sections never are;
- the recent-chats scan opens only the newest files by mtime and stops at
  `_RECENT_KEPT` user rows — background sessions (four-part ids) are skipped
  unread;
- "overdue" and "held" are different words: `autonomy.hold_reason` mirrors
  the scheduler's five gates, and a past-due task is overdue only when
  nothing holds it;
- subagents and background bash live in the lloyd-mcp process, read over
  loopback from `GET :8500/state`;
- a section can be *missing* after a backend restart; use
  `sectionError(section)` from `api.ts`, never `section.error`.

## Sessions, titles, activity

`app/sessions_io.py` is the one writer for every session (`create_session`)
and the one definition of "is a human reading this" (`is_user_session`;
`NON_USER_PLATFORMS = {autonomy, worker}`). Titles come from the secondary
on a geometric schedule (`app/session_titles.py`) and every surface shares
the fallback chain in `web/src/lib/sessionLabel.ts`. The live activity line
(`starting → prefill → thinking → tool → working`) is stamped by the turn
runner and read from the pure in-memory queue snapshot, which is also the
automod promoter's idle gate.

## The agent's view of the UI

`agent_mcp/mission_control_ui.py` lets a turn navigate the user
(`mc_navigate`) and read where they are (`mc_get_state`); the frontend
reports through `POST /api/mc/state`. `_summarize_browser` never carries the
screenshot or the accessibility tree — the summary goes into the model's
context on every move.

## Remote access

Vite terminates TLS with the CA in `agent-services/cert/` and presents client
certs to the backend, which enforces a per-device allowlist behind the `/api`
proxy. `scripts/gen-cert.sh` makes the CA once; `scripts/mint-client-cert.sh
<device>` enrolls a device. See [[infrastructure]].

## Related

[[harness]], [[background-runs]], [[workers]], [[inner-voice]].
