---
segment: architecture
tags: [architecture, lloyd, browser, chrome-extension, sidepanel]
type: reference
status: implemented
date: 2026-10-30
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
| `chrome-extension/src/background/service-worker.ts` | 339 lines: tab/navigation/session orchestration, the manual-check guard, the kickoff prompts |
| `chrome-extension/src/background/url.ts` | 68 lines: `canonicalize()`, the YouTube tests and `canCheckUrl` |
| `chrome-extension/src/background/tab-session-map.ts` | 92 lines: the `chrome.storage.session` wrapper holding the tab→session mapping |
| `chrome-extension/src/background/lloyd-client.ts` | 81 lines: three backend calls, used by the service worker only |
| `web/sidepanel.html` | the panel's HTML entry, 12 lines |
| `web/src/sidepanel/main.tsx` | 10 lines: mount |
| `web/src/sidepanel/SidePanelApp.tsx` | 211 lines: the panel UI, reusing `web/src/components/ChatPanel.tsx` and the rest of `web/src` |
| `chrome-extension/dist/` | the built unpacked extension — what Chrome is pointed at |

Two clients, deliberately unequal. The panel uses the app's full
`web/src/api.ts`, session helpers included, because it renders a transcript. The
service worker has its own minimal fetcher because it needs three operations and
must not import the app's state.

## What one check does

Pressing "Check it out, Lloyd" on a tab, in order. The button is the only entry
point, and it is enabled only when the tab's URL passes `canCheckUrl`
(`chrome-extension/src/background/url.ts`): http(s), and not refused by
`shouldSpawnSession`, which refuses `google.com` and its subdomains outright and
every YouTube page that is not a video. `handleManualCheck` re-reads the same
predicate before taking the tab lock, so the disabled button is a hint and the
service worker is the guard (#1765, `tests/test_sidepanel_spawn_guard.py`).

1. `POST /api/sessions/create` with `{platform: "browser", inner_voice: true}` —
   the tag is what makes these sessions findable (`app/routers/sessions.py`).
2. `PATCH /api/sessions/{id}/metadata` with `{url, title}`, so the page the
   session is about travels with it.
3. `POST /api/message/stream` with the kickoff — a page fetch and highlights, or
   a transcript on `youtube.com`/`youtu.be`. The response *body* is closed unread
   and the backend keeps running on disconnect (`app/routers/messages.py`), so
   the turn outlives the call; but the POST is **awaited**, and step 4 waits on
   it, because the handler enqueues the turn before it hands back its
   `StreamingResponse`. An answered POST is therefore the barrier that puts the
   panel's first `GET /status` inside the turn instead of racing it — fired and
   forgotten, the panel reached a session whose turn did not exist yet and showed
   no in-progress state for the entire kickoff. `spawnSession` has no way to
   skip this step: it takes no flag for it (#1767). The await is also the gate on
   the mint: `setMapping` and `setCanonical` run only once that POST has
   answered, so a refused one — a 503 out of the self-mod drain window, which
   `fireKickoff` throws on — commits nothing. The session it minted is left
   unreferenced, the same orphan a closed tab produces, and the tab keeps
   reporting not-checked with the button live, so the next press mints afresh and
   fires its own kickoff instead of the first guard below re-focusing a session
   with no turn (#3569, closed by #3645's contract, landed through #3854). The
   guard below reads the mapping, so the mapped session was what closed the URL
   to a retry, and `lastCommitted` was what made it stick: `handleNavigation`
   clears the mapping only when the tab's canonical changes, so a reload of the
   same page — its `getCanonical(tabId) === canonical` no-op — left both halves
   standing, which is precisely the reload a person reaches for when a button
   seems to do nothing. Committing only after the proof means a refusal writes
   neither half, so there is nothing to clear; `clearTab` is the only helper that
   drops both, and `clearMapping` keeps `lastCommitted` by design, which is why
   it could not have been the cleanup.
4. The panel switches to that session.

Re-checking the **same canonical URL** re-focuses the existing session and does
not re-fire the kickoff — that is the first guard in `handleManualCheck`.
Navigating away drops the tab's mapping, so the next check on that tab is fresh.
Switching tabs re-focuses the panel on the matching session or on the
not-checked state, through the same `pushFocus` the other four focus paths use —
one construction of `focus-session` in the file, so the switch carries the
`canCheck` the panel reads as `Boolean(msg.canCheck)`. It did not: the listener
built its own literal without the flag, which left the button disabled and the
panel saying "no checkable page here" of a checkable page until the next
navigation (#3568, fixed by #3741). Closing a tab
orphans the session, which stays in Lloyd's
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
client certificate, and none is asked of them. What admits the call is
`ApiPeerGate` in `server.py`, which decides on the peer address and nothing else
(`_is_trusted_peer`, never a `Host` or `X-Forwarded-For`): loopback is trusted, and
so is any network the backend is configured to trust. That is why the extension's
service worker gets through — its requests arrive from loopback — and it is why a
browser on the tailnet with no certificate gets through too. Neither is an
exception made for the panel, and the panel is not the only certless caller.
Which networks are trusted, and what certificate mechanism was retired and when,
belong to [[authority-surfaces]] (the Loopback entry) and to the transport history
in [[mission-control]] and [[infrastructure]]; this doc points at them rather than
copying them, so a change to the boundary has one place to be corrected. The
extension's own two write-ups say the same thing: `chrome-extension/README.md`
and the header of `chrome-extension/src/background/lloyd-client.ts` name
`ApiPeerGate` → `_is_trusted_peer` and say no client certificate is required
(#1722), and the header is in the corpus `tests/test_stale_mtls_comment_claims.py`
greps beside `web/src/api.ts`. `chrome-extension/` is inside automod's writable
set by name (#3646) — the service worker's TypeScript and the worker bundle it
builds are both round-writable — because the runner objection that made it
hand-landed is gone: `tests/test_sidepanel_focus_headless.py` drives the built
bytes in a node child, and the gate byte-compares them to a fresh build.

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

**The MV3 manifest is tracked, twice.** `chrome-extension/manifest.json` is the
hand-written source and `chrome-extension/dist/manifest.json` is the copy the
build writes beside the bundle; both need a negation in `.gitignore`, whose
blanket `*.json` rule kept them out of every tree until #1701 — a fresh
checkout got a built extension Chrome could not load.
`tests/test_chrome_extension_manifest_tracked.py` pins both as tracked and not
ignored. The build copies the manifest rather than generating it, so
`vite.chrome.config.ts` checks for `chrome-extension/manifest.json` when the
config loads and refuses with an error naming that path — before `emptyOutDir`
touches the output directory, instead of the pathless ENOENT it used to die
on after the output directory had already been emptied.

## What this doc does not cover

- **The browser as a tool.** `agent_mcp/browser.py` — navigation, snapshots, the
  SSRF guard that refuses private and internal hosts and lets loopback through on
  purpose (`_host_block_reason` in `agent_mcp/browser.py` returns allow for a
  hostname whose every resolved address is loopback, before `_is_private_host` is
  reached — cited by symbol, because the line number this clause carried had
  drifted fifteen lines into the function's unrelated empty-host comment) — is
  the agent driving a
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
- **2026-09-28 — corrected (#1759).** §Getting to the backend claimed
  `server.py` skips client-certificate auth for loopback, and that an extension
  could not present one anyway. Neither is true: the mechanism was retired for
  everyone on 2026-06-14 and `ApiPeerGate` has decided on the peer address since
  2026-09-20, so the sentence described a control that had not existed for
  fifteen months, in the one place a reader asks why plain HTTP to `:8080` is
  allowed. The copy in [[authority-surfaces]] was corrected by the review pass
  that filed this; the panel's copy survived it, which is that doc's own
  two-docs-one-guard failure. `tests/test_stale_mtls_comment_claims.py` now
  covers both files, and keeps the true history in the two docs that tell it.
- **2026-09-29 — corrected (arch-review, `a820d20b9e6f`).** §What one check does
  called the kickoff a fire-and-forget POST and made the abandoned response the
  reason that was sound. The POST has been **awaited**, deliberately, since
  `93733270` (2026-09-10) — eighteen days before this doc was written — because
  the enqueue lands before the `StreamingResponse` and the answered POST is what
  lets the panel's first `/status` see the turn; the body is still cancelled
  unread. The section now says what the await buys, and carries the one mechanism
  it never mentioned: `shouldSpawnSession`'s blocklist of `google.com` and
  non-video YouTube pages, which turns out to live in the button's `disabled`
  state rather than in `handleManualCheck` (#1765). Two claims fixed beside it —
  the `agent_mcp/browser.py` clause said its SSRF guard refuses loopback, and that
  guard in fact allows loopback and refuses private/internal hosts — and
  §Getting to the backend now warns that the extension's own README and client
  comment still point at the retired exemption (#1702/#1722), which is the same
  two-surfaces-one-rule failure one directory over. Re-measured and still true:
  all seven line counts in §The pieces, the three routes and their payloads, the
  tracked `chrome-extension/dist/` files, the build command, and
  `measurement`'s silence on browser sessions. **One item in that list was not
  re-measured:** "the manifest's absence from git" is the pre-#1701 state, which
  the hand-landing entry below retired the day before and the tree refutes now —
  `git ls-files` holds both `chrome-extension/manifest.json` and
  `chrome-extension/dist/manifest.json`, pinned by
  `tests/test_chrome_extension_manifest_tracked.py`, and §Build says they are
  tracked. The clause stays quoted because the pair is the finding: a retired
  claim and the entry that retired it sat eight days apart in one section of one
  doc, and only the body was read. Filed: the rotted `messages.py`
  citations in the two client comments (#1766) and `spawnSession`'s unreachable
  `kickoff` default, the last seam of the retired auto-spawn path (#1767).
- **2026-09-28 — hand-landed (#1701).** Both manifests are tracked through
  `.gitignore` negations, the build refuses a missing manifest by path before it
  empties the output directory, and §Build now says so; the "not loadable on its own"
  paragraph is gone with the cause it described.
- **2026-09-28 — hand-landed (#1722).** The service-worker client's header and
  the extension README describe the peer-address gate instead of the retired
  certificate exemption, and §Getting to the backend no longer warns that they
  do not.
- **2026-09-28 — hand-landed (#1766).** The two client comments that justify
  awaiting the kickoff POST cite `post_message_stream` / `enqueue_turn` and the
  module docstring of `app/routers/messages.py` instead of line ranges that had
  drifted onto the turn's error path; `tests/test_stale_mtls_comment_claims.py`
  now bans `messages.py:<digits>` in both client files as well.
- **2026-09-28 — hand-landed (#1767).** `spawnSession` lost its unreachable
  `kickoff = false` parameter and the `if` around the awaited kickoff; its one
  call passes four arguments, and §What one check does now says a session is
  never minted without its kickoff. `chrome-extension/dist/` rebuilt with the
  command in §Build.
- **2026-10-30 — stale (arch-review, `5348fadd`).** Stale for one claim, which is
  in this section rather than the body: the 2026-09-29 entry above lists "the
  manifest's absence from git" among things it calls re-measured, and that state
  died with #1701 the day before its own entry — both manifests are tracked.
  Quoted and corrected there rather than deleted, since the pair is the failure.
  Everything the body asserts was re-measured and holds: the seven line counts in
  §The pieces are exact, the three routes and their payloads match
  `chrome-extension/src/background/` against `app/routers/sessions.py` and
  `app/routers/messages.py` (where `enqueue_turn` at :2109 still precedes the
  `StreamingResponse` at :2113), `chrome-extension/dist/` is tracked with its
  manifest byte-identical to the source and every current string from
  `chrome-extension/src/` present in
  the bundles, `vite.chrome.config.ts` still refuses a missing manifest before
  `emptyOutDir`, `ApiPeerGate` still decides on the peer address alone, and
  `measurement` is still silent on browser sessions. Three sentences described
  intent rather than behaviour and are filed, not fixed here: the tab-activation
  push carries no `canCheck`, so the button this doc calls the only entry point
  arrives disabled on any already-loaded tab (#3568); the awaited kickoff does not
  gate the mint, so one refused POST closes that URL for the browser session
  (#3569); and nothing pins `chrome-extension/dist/` to `chrome-extension/src/`,
  which is load-bearing precisely
  because automod can neither edit `chrome-extension/` nor rebuild it (#3571).
  Replaced the one drifted pointer — `agent_mcp/browser.py:194` is a comment about
  empty hosts, not the loopback branch it was cited for — with the symbol, the way
  #2208 ruled for `authority-surfaces.md`; the same citation rots in eight places
  outside this doc (#3570).
- **2026-11-03 — the tab-switch focus push now carries `canCheck` (#3741, fixing #3568).**
  `chrome.tabs.onActivated` built its own `focus-session` literal, so a tab switch onto an
  already-loaded checkable tab pushed a message with no flag and the panel's
  `Boolean(msg.canCheck)` disabled the button and printed "No checkable page here". The
  listener body is now `await pushFocus(windowId, tabId)`, which already hydrates, reads
  the mapping and computes the flag — `type: "focus-session",` appears exactly once in
  `chrome-extension/src/background/service-worker.ts` (in `pushFocus`) and, after
  rebuilding through `web/vite.chrome.config.ts` into a temporary outDir and copying only
  `service-worker.js` back, exactly once in the tracked
  `chrome-extension/dist/service-worker.js` that Chrome loads. Not measurable by reading the
  text: `grep -c canCheck` on the bundle reads `1` before and after this change, because the
  one hit is `pushFocus`'s own send site in both trees, so the pins slice the bundle and
  then follow the awaited callee into the compiled function
  (`tests/test_automod_spec.py`, `tests/test_chrome_extension_manifest_tracked.py`). Those two
  can only ever say a string is present in the bytes Chrome loads; the flag itself is driven
  now, by the harness in the next bullet. The rebuild also renamed minifier-local identifiers
  across the whole bundle — which is why the pins below match a property by its shape and
  never by its identifier — and why `chrome-extension/dist/` is a tracked build output and not
  hand-edited.
- **2026-11-04 — the tracked bundle is driven headlessly, so the #3568 click-through is no
  longer owed to a person (#3772).** `tests/test_sidepanel_focus_headless.py` starts a node
  child, installs a stub `chrome` global over the ten `chrome.X.Y` members
  `chrome-extension/dist/service-worker.js` calls (measured: `grep -o
  "chrome\.[a-zA-Z]*\.[a-zA-Z]*"` over the bundle, sorted unique — `storage.session`,
  `tabs.get`/`onActivated`/`onUpdated`/`onRemoved`, `runtime.onConnect`/`onInstalled`,
  `webNavigation.onCommitted`/`onHistoryStateUpdated`, `sidePanel.setPanelBehavior`), imports
  the tracked bundle as an ES module, and runs the repro in order: a port named
  `lloyd-sidepanel` handed to the captured `runtime.onConnect` listener, a `panel-ready`
  message, and only then the captured `tabs.onActivated` listener. Both `webNavigation`
  registrations are in the stub because they run while the module body executes, so a stub
  written from the source rather than from the bundle's call sites fails the `import()`. What
  the test asserts is the object `port.postMessage` received: `type: "focus-session"` with
  `canCheck` true for `https://example.com/article` and false for
  `https://mail.google.com/mail/u/0/#inbox` from one process and one import, exactly one tab
  switch listener and one connect listener, `sessionKey`/`url`/`title` filled from a seeded
  `chrome.storage.session.get` and `sessionKey` null with the flag still computed when the
  store is empty, and zero `fetch` calls. A positive control drives a temporary copy whose one
  `canCheck:` send site was removed by shape and asserts the key is absent from both pushes —
  the leg that makes the two flag assertions evidence rather than a green-by-construction read,
  since #3568 is precisely an incident a text pin passed through. The ceiling this entry claimed
  there was false as written, and #3918 refutes it: `web/vitest.config.ts` `include` is
  `src/**/*.test.{ts,tsx}` with `web/` as its root, so it reaches `web/src/sidepanel/**` exactly
  as it reaches `web/src/components/**` — the include was never what kept the button's enabled
  state and the empty-state copy out of a test. `web/src/sidepanel/SidePanelView.test.tsx` now
  pins both under `environment: "node"` with `renderToStaticMarkup`, every render driven by
  `focusStateFromMessage`. What stays a browser concern, retired by nothing there, is what no
  vitest config reaches: reloading the shipped bytes into a real Chrome and pressing the button
  (#3854's leg, restated at the foot of the entry below).
- **2026-11-06 — a refused kickoff no longer maps the tab, so the button can retry the page
  (#3854, fixing #3569 on #3645's contract).** `spawnSession` wrote `setMapping` +
  `setCanonical` and only then awaited `POST /api/message/stream`, swallowing the rejection with
  a `console.error`, so one refused POST — a 503 from the self-mod drain window, the case
  `fireKickoff` throws on for exactly this reason — left the tab mapped to a session with no
  turn, and `handleManualCheck`'s first guard matched its canonical and re-focused that empty
  session on every later press: the button silently no-oped. §What one check does described that
  as the design until this entry. The commit now happens after the proof, inside the same
  `withTabLock` the check already holds, so a refusal writes neither half and there is nothing to
  unwind — the alternative in #3645's contract was a `clearTab` in the rejection handler (the
  only helper dropping `lastCommitted` as well as the mapping; `clearMapping` keeps it and would
  have stranded the canonical, reproducing the no-op one press later), and neither route reaches
  `clearMapping`. Two corrections to how this doc stated the blast radius: the strand lasted
  while the tab stayed on that canonical URL, not "for the rest of the browser session", because
  `handleNavigation` drops the mapping when the canonical changes — a **reload** is what did not
  recover, which is precisely when a person presses again; and a refused POST still leaves an
  orphan session in Lloyd's list, since the mint happens first and only the mapping waits. Not
  measurable by importing the TypeScript, and not measurable by reading the bytes either: the leg
  now drives `chrome-extension/dist/service-worker.js` under a stub `chrome` global with
  `/message/stream` answering 503 and reads out what the panel was handed (`sessionKey: null`,
  `canCheck` true), what `chrome.storage.session` holds (neither `tab:<id>` nor
  `lastCommitted:<id>`, which is why the store in that drive is a fake that keeps writes rather
  than the fixed answer the focus legs use), and that the second press attempted a second
  kickoff at all — the same drive with the POST answered mints once, shows the panel the session
  it minted without a second press, and logs nothing. Those legs are in
  `tests/test_sidepanel_focus_headless.py` beside the #3772 focus drive, with the shape they
  depend on pinned at source level by `tests/test_sidepanel_spawn_guard.py`, and the shipped
  `chrome-extension/dist/service-worker.js` is rebuild-verified by the gate's
  `chrome_bundle_provenance` rung rather than hand-authored. It is the second round to edit
  through #3646's grant — round `SM_20261103_223217` promoted `service-worker.ts` and its
  bundle on 2026-11-03, for #3741 — and the sentence in `## Getting to the backend` that said
  an edit under `chrome-extension/` is hand-landed is corrected here rather than left standing,
  because those two promotions are what refute it. What is still owed is the one thing a node
  child cannot do: reload these bytes into a real Chrome and
  press the button with the backend stopped, then again with it restored.
- **2026-11-08 — the panel's own render is now spec'd in vitest; the real-Chrome leg is the one
  left (#3918).** `web/src/sidepanel/SidePanelApp.tsx` exports `focusStateFromMessage`, the
  focus-push→state mapping lifted verbatim out of the `port.onMessage` listener (which now calls
  `setState(focusStateFromMessage(msg))`), and exports `CheckButton` / `EmptyState`. It is an
  export-only refactor: the `disabled` expression, the copy strings and the render tree are
  byte-identical, so the panel-bundle string pin in
  `tests/test_chrome_extension_manifest_tracked.py`, which harvests its literals from this source
  and looks them up in the shipped chunk, needed no rebuild — `chrome-extension/dist/**` is
  untouched by the commit and that file stays green with the rest. What the exports buy is
  `web/src/sidepanel/SidePanelView.test.tsx`: 7 tests in vitest's `node` environment rendering
  with `renderToStaticMarkup`, no DOM package and no config change (`web/vitest.config.ts` is
  neither in `ALLOWED_GLOBS` nor `DENIED_GLOBS` — `spec.classify` answers `unlisted` — so the
  unchanged-config half of the clause is enforced by the scope rail, not promised). The point of
  driving the render from the mapping instead of a hand-written prop is that the mapping is what
  broke: a `canCheck: true` push renders an enabled `Check it out, Lloyd` and `Ready when you
  are`, and the same push with the key absent — #3568's exact field — renders `disabled` and `No
  checkable page here`. Both halves are mutation-checked. Dropping `!canCheck` from
  `disabled={pending || !canCheck}` reddens exactly the omitted-canCheck render assertion (1 of 7)
  and nothing else, because with `pending` false only that one call has an argument for
  `disabled` to honour; a mapping that ignores `msg.canCheck` reddens the enabled-button and
  ready-copy assertions when it hardcodes `false` (2 of 7) and the omitted-canCheck assertions
  when it hardcodes `true` (3 of 7). The anchor matters for the second one: `canCheck: false,` is
  also `INITIAL`'s value, so an unanchored edit takes both, the module stops collecting, and
  vitest prints `Tests  no tests` — which is a compile error wearing a caught-mutation costume.
  What is still owed is unchanged by any of this: reload these bytes into a real Chrome, put the
  backend in the stopped/drain state, and press the button (#3854's leg). Clauses 1–4 pin the
  shipped mapping and the shipped copy, not Chrome's painting of them, and nothing in this repo
  retires that person-only leg.
  One thing the round learned the hard way: `tests/test_sidepanel_panel_view_contract.py` mirrors
  clauses 1–4 in Python over the same bytes, because the review rung's node rail reads a
  `test_node_id` only in a `.py` under a `pytest.ini` testpath
  (`scripts/automod/testpaths.py::is_test_file`), so a clause pinned only in a `.test.tsx` is
  recorded as untested — `tests/test_dashboard_label_contract.py` (#2202) and
  `tests/test_background_runs_panel_contract.py` (#2648) already carry that ruling, and this
  round spent a review attempt rediscovering it. The mirror is not a translation: it checks the
  mapping for purity, which a render cannot see, and it requires the copy the vitest file asserts
  with `toContain` to still be copy the panel ships, which is the only check that outlives this
  round.
