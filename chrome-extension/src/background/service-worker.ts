// Lloyd Chrome side-panel service worker.
//
// One Lloyd session per (tab, URL). Sessions are **never auto-spawned** —
// the "Check it out, Lloyd" button in the panel is the only trigger
// (`request-session` message). Navigation tracking still runs so the
// panel switches to the not-checked state when a tab moves to a new URL,
// and so re-checking the same URL after a reload doesn't fire a second
// kickoff (first guard in handleManualCheck).
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
  canCheck?: boolean
}

interface PanelReadyMessage {
  type: "panel-ready"
  windowId: number
  tabId: number
}

interface RequestSessionMessage {
  type: "request-session"
  windowId: number
  tabId: number
}

// Kickoff prompt for the "Check it out, Lloyd" check (YouTube variant
// when the URL is a YouTube video).
function buildKickoffMessage(url: string): string {
  return isYouTube(url)
    ? `pull the youtube transcript and give me the highlights\n${url}`
    : `pull the webpage contents and give me the highlights\n${url}`
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

// Push the tab's current focus state (session mapping + whether the tab's
// URL is checkable) to the panel in the given window. Replaces the
// inline focus-session pushes so the "Check it out, Lloyd" button always
// gets an up-to-date canCheck flag.
async function pushFocus(windowId: number, tabId: number) {
  await hydrate()
  const entry = getMapping(tabId)
  let canCheck = false
  try {
    const tab = await chrome.tabs.get(tabId)
    canCheck = Boolean(
      tab.url && /^https?:/i.test(tab.url) && shouldSpawnSession(tab.url),
    )
  } catch {
    /* tab gone — push what we have */
  }
  pushToPanel(windowId, {
    type: "focus-session",
    windowId,
    tabId,
    sessionKey: entry?.sessionKey ?? null,
    url: entry?.url,
    title: entry?.title,
    canCheck,
  })
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "lloyd-sidepanel") return
  let myWindowId: number | null = null

  port.onMessage.addListener(async (raw: unknown) => {
    if (!raw || typeof raw !== "object") return
    const msg = raw as PanelReadyMessage | RequestSessionMessage
    if (msg.type === "request-session") {
      await handleManualCheck(
        (msg as RequestSessionMessage).windowId,
        (msg as RequestSessionMessage).tabId,
      )
      return
    }
    if (msg.type !== "panel-ready") return
    await hydrate()
    myWindowId = msg.windowId
    panelPorts.set(msg.windowId, port)

    // Sessions are only created on explicit "Check it out, Lloyd" — so
    // opening the panel just reports the tab's current state (mapping,
    // if any, plus whether the visible URL is checkable).
    await pushFocus(msg.windowId, msg.tabId)
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

// ── Navigation = URL tracking only; sessions are never auto-spawned ────
// The "Check it out, Lloyd" button is the only way to create a session.
// Nav tracking still matters for two things:
//   1. lastCommitted canonical URLs (reload dedupe for the check button),
//   2. dropping the tab's session mapping when the URL changes, so the
//      panel switches to the "not checked" state instead of showing the
//      previous page's session for the new URL.
async function handleNavigation(details: {
  tabId: number
  url: string
  frameId: number
}) {
  if (details.frameId !== 0) return
  if (!/^https?:/i.test(details.url)) return
  await hydrate()
  const tabId = details.tabId
  const changed = await withTabLock(tabId, async () => {
    const canonical = canonicalize(details.url)
    if (getCanonical(tabId) === canonical) return false // reload, no-op
    await setCanonical(tabId, canonical)
    // The old mapping (if any) now points at a stale URL — drop it so
    // the next check against this tab's new URL spawns a fresh session.
    await clearMapping(tabId)
    return true
  })
  if (changed) {
    const tab = await chrome.tabs.get(tabId).catch(() => null)
    if (tab && typeof tab.windowId === "number") {
      await pushFocus(tab.windowId, tabId)
    }
  }
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
  kickoff = false,
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

  if (kickoff) {
    fireKickoff(sessionKey, buildKickoffMessage(url), `ext_tab_${tabId}`).catch(
      (err) => console.error("[lloyd-sw] fireKickoff failed:", err),
    )
  }

  // Notify the panel (if open in the tab's window).
  const tab = await chrome.tabs.get(tabId).catch(() => null)
  if (tab && typeof tab.windowId === "number") {
    await pushFocus(tab.windowId, tabId)
  }
}

// ── Manual check ("Check it out, Lloyd") ───────────────────────────────
// The panel asks the SW to spawn a session for the tab's current URL and
// kick it off. The canonical-URL dedupe still applies: if the tab already
// has a session for this URL we focus it instead of re-spawning.
async function handleManualCheck(windowId: number, tabId: number) {
  const tab = await chrome.tabs.get(tabId).catch(() => null)
  if (!tab || !tab.url || !/^https?:/i.test(tab.url)) {
    await pushFocus(windowId, tabId)
    return
  }

  const canonical = canonicalize(tab.url)

  await withTabLock(tabId, async () => {
    const cur = getMapping(tabId)
    if (cur && canonicalize(cur.url) === canonical) {
      // This tab's session is already for this exact URL — re-focus it
      // instead of spawning a duplicate or re-firing the kickoff.
      await pushFocus(windowId, tabId)
      return
    }
    // No session for this URL (first check, or navigated away and back)
    // — spawn + kickoff.
    await spawnSession(tab.id, tab.url, tab.title ?? "", canonical, true)
  })

  await pushFocus(windowId, tabId)
}

// ── Tab switch = re-focus panel on the new tab's session ───────────────
// Pure focus: no spawning. The button is the only way to create a
// session; switching tabs just shows the (possibly empty) state of the
// new tab and whether its current URL is checkable.
chrome.tabs.onActivated.addListener(async ({ tabId, windowId }) => {
  await hydrate()
  const entry = getMapping(tabId)
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
