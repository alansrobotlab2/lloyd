import { useEffect, useState } from 'react'
import { api, type SessionMeta } from '../api'

/**
 * Display metadata (title, preview, platform) for one open session.
 *
 * A session's title is written by the secondary model a few seconds after
 * a turn completes, so a brand-new chat has none and then acquires one.
 * A slow poll is how the header notices: there is no event for it, and
 * opening a socket to deliver a label would be more machinery than the
 * label is worth.
 *
 * Returns null while unresolved, when no session is open, or when the
 * fetch fails — every caller falls back to showing nothing.
 */
export function useSessionMeta(
  sessionKey: string | null,
  { pollMs = 20000, enabled = true }: { pollMs?: number; enabled?: boolean } = {},
): SessionMeta | null {
  const [meta, setMeta] = useState<SessionMeta | null>(null)

  useEffect(() => {
    if (!sessionKey || !enabled) {
      setMeta(null)
      return
    }
    // Clear immediately on a session switch — showing the previous
    // session's title over the new one's transcript is worse than blank.
    setMeta(null)

    let cancelled = false
    const load = () => {
      api.getSessionMeta(sessionKey)
        .then(m => { if (!cancelled) setMeta(m) })
        .catch(() => { /* session not persisted yet, or gone */ })
    }
    load()
    const id = setInterval(load, pollMs)
    return () => { cancelled = true; clearInterval(id) }
  }, [sessionKey, enabled, pollMs])

  return meta
}
