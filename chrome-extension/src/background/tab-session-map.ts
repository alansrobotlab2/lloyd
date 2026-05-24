// Persistent tab → session mapping, backed by chrome.storage.session.
//
// MV3 service workers can be killed after ~30s idle; the in-memory Map
// dies with them. chrome.storage.session is cleared on browser restart
// but survives SW restarts within a session, which is exactly the
// lifetime we want for tab IDs (Chrome reissues tab IDs on restart too).
//
// Every consumer of getMapping()/getCanonical() must await the same
// hydrate() call once per handler entry; otherwise we'd read stale
// in-memory state right after a cold start.

export interface TabSession {
  sessionKey: string
  url: string
  title: string
}

const TAB_PREFIX = "tab:"
const COMMIT_PREFIX = "lastCommitted:"

const mapping = new Map<number, TabSession>()
const lastCommitted = new Map<number, string>()

let hydrated = false
let hydratePromise: Promise<void> | null = null

export function hydrate(): Promise<void> {
  if (hydrated) return Promise.resolve()
  if (hydratePromise) return hydratePromise
  hydratePromise = (async () => {
    const all = await chrome.storage.session.get(null)
    for (const [k, v] of Object.entries(all)) {
      if (k.startsWith(TAB_PREFIX) && v && typeof v === "object") {
        const id = Number(k.slice(TAB_PREFIX.length))
        if (!Number.isNaN(id)) mapping.set(id, v as TabSession)
      } else if (k.startsWith(COMMIT_PREFIX) && typeof v === "string") {
        const id = Number(k.slice(COMMIT_PREFIX.length))
        if (!Number.isNaN(id)) lastCommitted.set(id, v)
      }
    }
    hydrated = true
  })()
  return hydratePromise
}

export function getMapping(tabId: number): TabSession | undefined {
  return mapping.get(tabId)
}

export async function setMapping(tabId: number, entry: TabSession) {
  mapping.set(tabId, entry)
  await chrome.storage.session.set({ [`${TAB_PREFIX}${tabId}`]: entry })
}

export async function updateMapping(
  tabId: number,
  patch: Partial<TabSession>,
) {
  const existing = mapping.get(tabId)
  if (!existing) return
  const next = { ...existing, ...patch }
  mapping.set(tabId, next)
  await chrome.storage.session.set({ [`${TAB_PREFIX}${tabId}`]: next })
}

export async function clearTab(tabId: number) {
  mapping.delete(tabId)
  lastCommitted.delete(tabId)
  await chrome.storage.session.remove([
    `${TAB_PREFIX}${tabId}`,
    `${COMMIT_PREFIX}${tabId}`,
  ])
}

// Drop the session mapping but KEEP lastCommitted so subsequent
// navigations on this tab are still compared against the current URL.
// Used when navigating to an excluded host (google.com, youtube.com home).
export async function clearMapping(tabId: number) {
  mapping.delete(tabId)
  await chrome.storage.session.remove([`${TAB_PREFIX}${tabId}`])
}

export function getCanonical(tabId: number): string | undefined {
  return lastCommitted.get(tabId)
}

export async function setCanonical(tabId: number, canonical: string) {
  lastCommitted.set(tabId, canonical)
  await chrome.storage.session.set({
    [`${COMMIT_PREFIX}${tabId}`]: canonical,
  })
}
