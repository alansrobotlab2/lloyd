# Lloyd Chrome Side Panel

A Chrome extension that opens a side panel mirroring Lloyd's chat UI, with **one Lloyd session per browser tab and per webpage**.

When you navigate to a new page (or open a new tab), the extension:

1. Creates a Lloyd session tagged `platform: "browser"`.
2. Fires a kickoff message asking Lloyd to fetch the page and produce highlights — YouTube transcripts on `youtube.com`/`youtu.be`, regular webpage content elsewhere.
3. Switches the side panel to that session.

Switching tabs re-focuses the panel on the matching session. Closing a tab orphans the session (it stays in Lloyd's main session list).

## Layout

```
chrome-extension/
  manifest.json                      # MV3 manifest
  sidepanel.html                     # (build output, copied from web/sidepanel.html)
  icons/{16,48,128}.png
  src/background/
    service-worker.ts                # tab/nav/session orchestration
    url.ts                           # canonicalize() + isYouTube()
    tab-session-map.ts               # chrome.storage.session wrapper
    lloyd-client.ts                  # fetch wrappers for the SW
  dist/                              # built unpacked extension (load this)
```

The React side panel lives in [web/src/sidepanel/](../web/src/sidepanel/) (it reuses [ChatPanel](../web/src/components/ChatPanel.tsx) and the rest of `web/src`).

## Build

```bash
cd web
VITE_API_BASE='http://127.0.0.1:8080/api' \
  npx vite build -c vite.chrome.config.ts --watch
```

This emits everything into `chrome-extension/dist/`. The `--watch` flag keeps rebuilding on source change.

## Load in Chrome

1. `chrome://extensions` → enable Developer mode.
2. **Load unpacked** → pick `chrome-extension/dist/`.
3. Click the extension icon to open the side panel.

After a code change, vite re-emits; click the extension's reload icon on the extensions page to pick it up. (MV3 service workers can't HMR.)

## Backend touchpoints

- `POST /api/sessions/create` with `{platform: "browser"}` — creates the session.
- `PATCH /api/sessions/{id}/metadata` with `{url, title}` — stashes page metadata.
- `POST /api/message/stream` — fires the kickoff (response is abandoned; backend keeps running on disconnect, see [messages.py:10](../app/routers/messages.py)).
- `GET /api/messages/{id}` — side panel uses this via [web/src/api.ts](../web/src/api.ts) to render the transcript.

The Lloyd backend's mTLS middleware skips loopback ([server.py:76-113](../server.py)), so the extension's calls to `http://127.0.0.1:8080` need no client cert.
