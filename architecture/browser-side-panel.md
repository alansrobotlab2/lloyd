---
segment: architecture
tags: [architecture, lloyd, browser, chrome-extension, sidepanel]
type: reference
status: implemented
date: 2026-09-28
---

# Lloyd — the browser side panel

An MV3 Chrome extension whose side panel mirrors Lloyd's chat, with **one Lloyd
session per browser tab and per page, created only when a person asks for it**.
Nothing about navigation spawns anything: the "Check it out, Lloyd" button at the
top of the panel is the only trigger, which is the whole design decision and the
reason this surface is small.

It had no doc before #1699. `architecture/background-runs.md` mentions the
extension once, in passing, as a producer that posts `platform: "browser"`
sessions; no doc owned the extension's own mechanism, so the area could not be a
review unit — `workers/sources/arch_review.py::doc_slugs` makes a unit out of a
doc that exists, and no doc did.

## The pieces

| Path | What it is |
|---|---|
| `chrome-extension/src/background/service-worker.ts` | 317 lines: tab/navigation/session orchestration, the manual-check guard, the kickoff prompts |
| `chrome-extension/src/background/url.ts` | 60 lines: `canonicalize()` and the YouTube tests |
| `chrome-extension/src/background/tab-session-map.ts` | 92 lines: the `chrome.storage.session` wrapper holding the tab→session mapping |
| `chrome-extension/src/background/lloyd-client.ts` | 74 lines: three backend calls, used by the service worker only |
| `web/sidepanel.html` | the panel's HTML entry, 12 lines |
| `web/src/sidepanel/main.tsx` | 10 lines: mount |
| `web/src/sidepanel/SidePanelApp.tsx` | 211 lines: the panel UI, reusing `web/src/components/ChatPanel.tsx` and the rest of `web/src` |
| `chrome-extension/dist/` | the built unpacked extension — what Chrome is pointed at |

Two clients, deliberately unequal. The panel uses the app's full
`web/src/api.ts`, session helpers included, because it renders a transcript. The
service worker has its own minimal fetcher because it needs three operations and
must not import the app's state.

## What one check does

Pressing "Check it out, Lloyd" on a tab, in order:

1. `POST /api/sessions/create` with `{platform: "browser", inner_voice: true}` —
   the tag is what makes these sessions findable (`app/routers/sessions.py`).
2. `PATCH /api/sessions/{id}/metadata` with `{url, title}`, so the page the
   session is about travels with it.
3. `POST /api/message/stream` with the kickoff — a page fetch and highlights, or
   a transcript on `youtube.com`/`youtu.be`. The response is abandoned; the
   backend keeps running on disconnect (`app/routers/messages.py`), which is the
   only reason a fire-and-forget POST is a sound kickoff.
4. The panel switches to that session.

Re-checking the **same canonical URL** re-focuses the existing session and does
not re-fire the kickoff — that is the first guard in `handleManualCheck`.
Navigating away drops the tab's mapping, so the next check on that tab is fresh.
Switching tabs re-focuses the panel on the matching session or on the
not-checked state. Closing a tab orphans the session, which stays in Lloyd's
session list: the panel's lifecycle is the tab's, the transcript's is not.
Closing the panel does not stop the run ([[background-runs]]).

## Why "same page" is a URL question

The re-focus-instead-of-respawn rule is only as good as its notion of "same", so
`canonicalize()` is origin plus pathname — no fragment, no trailing slash — with
the query string **kept**, because article IDs and search terms live there. The
one exception is a YouTube watch URL, normalised down to the video ID, so a `?t=`
timestamp does not mint a session per seek. A `URL` that fails to parse is
returned unchanged rather than dropped: an unparseable URL should still get a
session, and a canonicaliser that swallows its input on the error path would look
like "same page" for every malformed URL.

## Getting to the backend

`lloyd-client.ts` and the panel both talk to `http://127.0.0.1:8080/api` with no
client certificate. That is not an exception made for the extension: `server.py`
skips mTLS for loopback, so loopback is the only origin that reaches the API
without a cert, and an extension's service worker cannot present one. See
[[mission-control]] for the certificate story and [[authority-surfaces]] for why
that exemption is the surface it is.

## Build, and what is in git

```
cd web && VITE_API_BASE='http://127.0.0.1:8080/api' \
  npx vite build -c vite.chrome.config.ts --watch
```

Everything is emitted into `chrome-extension/dist/`, and that directory is
tracked —
`chrome-extension/dist/sidepanel.html` and `chrome-extension/dist/service-worker.js`
are committed files, so a rebuild shows up as a diff in a review. MV3 service
workers cannot hot-reload, so after a rebuild the extension's reload icon on the
extensions page is what picks it up; `--watch` only rewrites the bundle.

**The built output in git is not loadable on its own.** The MV3 manifest is not
tracked (#1701 owns the cause), so it is absent from a fresh checkout, from a
worktree, and from the tracked build output Chrome is pointed at. Loading
unpacked works on this machine because the manifest is present here. Nothing else
about this surface depends on the manifest's contents being described here, which
is why this doc says *manifest* and does not cite a path to one: a citation to a
file that exists only in one checkout would be a claim its own test could not
re-check.

## What this doc does not cover

- **The browser as a tool.** `agent_mcp/browser.py` — navigation, snapshots, the
  SSRF guard that refuses private and loopback targets — is the agent driving a
  browser, and is [[tools]]. This doc is the human driving Lloyd from a tab, and
  the two share only a backend.
- **Chat mechanics.** Sessions, titles, SSE and the `platform` tag are
  [[mission-control]] and [[harness]]; the panel is another client of them.
- **Background runs.** Why a run outlives its request, and how it is observed, is
  [[background-runs]].
- **Transport and certificates.** The certificate, the ports and the tailnet are
  [[infrastructure]].
- **Whether the panel is in use, or healthy.** There is no measurement of it: no
  arm in [[measurement]] scores browser sessions, and nothing here claims the
  extension is loaded on Alan's Chrome right now. A side panel that silently
  stopped creating sessions would be noticed by nobody but the person looking at
  it, which is the honest gap in this surface rather than a covered one.

## Review log

- **2026-09-28 — created (#1699).** Line counts and behaviour read from
  `chrome-extension/src/background/` and `web/src/sidepanel/` at `f7cf29f4`, and
  checked against `chrome-extension/README.md`, which is the older of the two and
  where the "zero mentions" claim in the item was already wrong —
  `architecture/background-runs.md` mentions the side panel, by two words rather
  than the compound, so the true claim was "no doc owns it". The manifest
  absence above was measured in this worktree, not inferred from the item.
