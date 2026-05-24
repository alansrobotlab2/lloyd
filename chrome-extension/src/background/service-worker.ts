// Lloyd Chrome side-panel service worker.
//
// One Lloyd session per browser tab + per committed top-frame nav, but
// **only when the side panel is actually open for that tab**. Background
// tabs and tabs in windows where the panel is closed don't spawn
// sessions. When the user later focuses such a tab (with the panel
// open), we backfill from the URL that's currently visible.
//
// Mapping is held in chrome.storage.session so it survives SW restarts
// within a browser session.

import {
  hydrate,
  getMapping,
  setMapping,
  updateMapping,
  clearTab,
  clearMapping,
  getCanonical,
  setCanonical,
} from "./tab-session-map"
import { canonicalize, isYouTube, shouldSpawnSession } from "./url"
import {
  createBrowserSession,
  patchBrowserMetadata,
  fireKickoff,
} from "./lloyd-client"

interface FocusMessage {
  type: "focus-session"
  windowId: number
  tabId: number
  sessionKey: string | null
  url?: string
  title?: string
}

interface PanelReadyMessage {
  type: "panel-ready"
  windowId: number
  tabId: number
}

// ── Port management ────────────────────────────────────────────────────
// Side panel is one-per-window. We keep at most one port per windowId
// and only push focus events to the matching window's panel.
const panelPorts = new Map<number, chrome.runtime.Port>()

function pushToPanel(windowId: number, msg: FocusMessage) {
  const port = panelPorts.get(windowId)
  if (!port) return
  try {
    port.postMessage(msg)
  } catch {
    panelPorts.delete(windowId)
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "lloyd-sidepanel") return
  let myWindowId: number | null = null

  port.onMessage.addListener(async (raw: unknown) => {
    if (!raw || typeof raw !== "object") return
    const msg = raw as PanelReadyMessage
    if (msg.type !== "panel-ready") return
    await hydrate()
    myWindowId = msg.windowId
    panelPorts.set(msg.windowId, port)

    // If the tab doesn't have a session yet (extension just installed,
    // browser just relaunched, or the user opened the panel on a tab
    // that hadn't navigated since), backfill one for the currently
    // visible URL.
    let entry = getMapping(msg.tabId)
    if (!entry) {
      try {
        const tab = await chrome.tabs.get(msg.tabId)
        if (
          tab.url &&
          /^https?:/i.test(tab.url) &&
          shouldSpawnSession(tab.url)
        ) {
          await withTabLock(msg.tabId, async () => {
            if (getMapping(msg.tabId)) return
            const canonical = canonicalize(tab.url!)
            if (getCanonical(msg.tabId) === canonical) return
            await spawnSession(
              msg.tabId,
              tab.url!,
              tab.title ?? "",
              canonical,
            )
          })
          entry = getMapping(msg.tabId)
        }
      } catch {
        /* tab closed between query and get */
      }
    }

    port.postMessage({
      type: "focus-session",
      windowId: msg.windowId,
      tabId: msg.tabId,
      sessionKey: entry?.sessionKey ?? null,
      url: entry?.url,
      title: entry?.title,
    } satisfies FocusMessage)
  })

  port.onDisconnect.addListener(() => {
    if (myWindowId !== null && panelPorts.get(myWindowId) === port) {
      panelPorts.delete(myWindowId)
    }
  })
})

// ── Per-tab serialization ──────────────────────────────────────────────
// A burst of nav events on the same tab (think: client redirects) could
// otherwise spawn duplicate sessions. We chain async work per tabId.
const tabLocks = new Map<number, Promise<unknown>>()
function withTabLock<T>(tabId: number, fn: () => Promise<T>): Promise<T> {
  const prev = tabLocks.get(tabId) ?? Promise.resolve()
  const next = prev.then(fn, fn)
  tabLocks.set(
    tabId,
    next.catch(() => undefined),
  )
  return next
}

// ── Setup ──────────────────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch(() => {
      /* ignore — older Chrome */
    })
})

// ── Navigation = the only place sessions are spawned ───────────────────
async function handleNavigation(details: {
  tabId: number
  url: string
  frameId: number
}) {
  if (details.frameId !== 0) return
  if (!/^https?:/i.test(details.url)) return
  await hydrate()
  const tabId = details.tabId
  await withTabLock(tabId, async () => {
      const canonical = canonicalize(details.url)
      if (getCanonical(tabId) === canonical) return // reload, no-op
      // Update lastCommitted unconditionally so the next nav comparison
      // still works even when this URL is excluded.
      await setCanonical(tabId, canonical)

      if (!shouldSpawnSession(details.url)) {
        // Excluded host — drop any existing mapping for this tab so the
        // panel falls back to the empty state. lastCommitted stays so a
        // later nav to a real page is correctly detected.
        await clearMapping(tabId)
        try {
          const tab = await chrome.tabs.get(tabId)
          if (typeof tab.windowId === "number") {
            pushToPanel(tab.windowId, {
              type: "focus-session",
              windowId: tab.windowId,
              tabId,
              sessionKey: null,
            })
          }
        } catch {
          /* tab closed */
        }
        return
      }

      // Only spawn if the user is actively engaging with this tab via
      // the side panel. Background tabs and tabs in windows with no
      // open panel are deferred — the spawn happens later via
      // tabs.onActivated or the panel-ready backfill.
      let tab: chrome.tabs.Tab | undefined
      try {
        tab = await chrome.tabs.get(tabId)
      } catch {
        return
      }
      const winId = tab.windowId
      if (
        !tab.active ||
        typeof winId !== "number" ||
        !panelPorts.has(winId)
      ) {
        // Old mapping is now stale (different URL). Drop it so we don't
        // serve the wrong session when the user does focus this tab.
        await clearMapping(tabId)
        return
      }

      await spawnSession(tabId, details.url, tab.title ?? "", canonical)
    })
}

chrome.webNavigation.onCommitted.addListener(handleNavigation, {
  url: [{ schemes: ["http", "https"] }],
})

// YouTube is a SPA — clicking a video from the home page triggers
// pushState, not a hard nav. Listen for history updates on YouTube
// hostnames only; canonicalize() collapses non-video YouTube pages so
// the per-tab canonical compare still works.
chrome.webNavigation.onHistoryStateUpdated.addListener(handleNavigation, {
  url: [
    { hostEquals: "www.youtube.com" },
    { hostEquals: "m.youtube.com" },
    { hostEquals: "youtube.com" },
    { hostEquals: "music.youtube.com" },
  ],
})

async function spawnSession(
  tabId: number,
  url: string,
  title: string,
  canonical: string,
) {
  let sessionKey: string
  try {
    sessionKey = await createBrowserSession()
  } catch (err) {
    console.error("[lloyd-sw] createBrowserSession failed:", err)
    return
  }

  await setMapping(tabId, { sessionKey, url, title })
  await setCanonical(tabId, canonical)

  // Best-effort metadata patch; title may still be hostname at this
  // point — the onUpdated listener will re-patch when the real title
  // arrives.
  patchBrowserMetadata(sessionKey, { url, title }).catch(() => undefined)

  const kickoff = isYouTube(url)
    ? `pull the youtube transcript and give me the highlights\n${url}`
    : `pull the webpage contents and give me the highlights\n${url}`

  fireKickoff(sessionKey, kickoff, `ext_tab_${tabId}`).catch((err) =>
    console.error("[lloyd-sw] fireKickoff failed:", err),
  )

  // Notify the panel (if open in the tab's window).
  try {
    const tab = await chrome.tabs.get(tabId)
    if (typeof tab.windowId === "number") {
      pushToPanel(tab.windowId, {
        type: "focus-session",
        windowId: tab.windowId,
        tabId,
        sessionKey,
        url,
        title,
      })
    }
  } catch {
    /* tab gone */
  }
}

// ── Tab switch = re-focus panel on the new tab's session ───────────────
// If the panel is open in this window and the newly focused tab doesn't
// yet have a session for its current URL, spawn one now (this is when
// the user is "actually engaging" with the tab through Lloyd).
chrome.tabs.onActivated.addListener(async ({ tabId, windowId }) => {
  await hydrate()
  let entry = getMapping(tabId)

  if (panelPorts.has(windowId)) {
    try {
      const tab = await chrome.tabs.get(tabId)
      if (
        tab.url &&
        /^https?:/i.test(tab.url) &&
        shouldSpawnSession(tab.url)
      ) {
        const canonical = canonicalize(tab.url)
        if (!entry || canonicalize(entry.url) !== canonical) {
          await withTabLock(tabId, async () => {
            const cur = getMapping(tabId)
            if (cur && canonicalize(cur.url) === canonical) return
            await spawnSession(tabId, tab.url!, tab.title ?? "", canonical)
          })
          entry = getMapping(tabId)
        }
      }
    } catch {
      /* tab closed */
    }
  }

  pushToPanel(windowId, {
    type: "focus-session",
    windowId,
    tabId,
    sessionKey: entry?.sessionKey ?? null,
    url: entry?.url,
    title: entry?.title,
  })
})

// ── Title updates land late; re-patch metadata when they do ────────────
chrome.tabs.onUpdated.addListener(
  async (tabId, changeInfo) => {
    if (!changeInfo.title) return
    await hydrate()
    const entry = getMapping(tabId)
    if (!entry) return
    await updateMapping(tabId, { title: changeInfo.title })
    patchBrowserMetadata(entry.sessionKey, {
      title: changeInfo.title,
    }).catch(() => undefined)
  },
)

// ── Tab close = drop mapping (session stays in Lloyd, orphaned) ────────
chrome.tabs.onRemoved.addListener(async (tabId) => {
  await hydrate()
  await clearTab(tabId)
})
