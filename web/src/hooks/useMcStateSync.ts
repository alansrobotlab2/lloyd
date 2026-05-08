import { useEffect, useRef } from 'react'
import { useMcUi, type McFocus } from '../contexts/McUiContext'
import type { Page } from '../components/Sidebar'

// Debounced POST of the active tab + focus to /api/mc/state.
// Mounted once at the top of the tree (Layout); fires on any change to
// the current tab or its focused item.
export function useMcStateSync() {
  const { currentTab, focusByTab } = useMcUi()
  const focus: McFocus | null = (focusByTab[currentTab] ?? null) as McFocus | null

  const timer = useRef<number | null>(null)
  const lastSent = useRef<string>('')

  useEffect(() => {
    const payload = { tab: currentTab as Page, focus }
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
  }, [currentTab, focus])
}
