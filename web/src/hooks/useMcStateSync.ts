import { useEffect, useRef } from 'react'
import { useMcUi, type McFocus } from '../contexts/McUiContext'
import type { Page } from '../components/Sidebar'

// Debounced POST of the active tab + focus + IDE state to /api/mc/state.
// Mounted once at the top of the tree (Layout); fires on any change to
// the current tab, its focused item, or the IDE state mirror.
export function useMcStateSync() {
  const { currentTab, focusByTab, ideState } = useMcUi()
  const focus: McFocus | null = (focusByTab[currentTab] ?? null) as McFocus | null

  const timer = useRef<number | null>(null)
  const lastSent = useRef<string>('')

  useEffect(() => {
    // Only send the `ide` field when there's something to mirror — sending
    // null on every page-switch would clobber the backend state.
    const payload: Record<string, unknown> = { tab: currentTab as Page, focus }
    if (ideState) payload.ide = ideState
    const sig = JSON.stringify(payload)
    if (sig === lastSent.current) return

    if (timer.current !== null) {
      window.clearTimeout(timer.current)
    }
    timer.current = window.setTimeout(() => {
      lastSent.current = sig
      fetch('/api/mc/state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: sig,
      }).catch(() => {
        // Swallow — best-effort mirror.
      })
    }, 250)

    return () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current)
        timer.current = null
      }
    }
  }, [currentTab, focus, ideState])
}
