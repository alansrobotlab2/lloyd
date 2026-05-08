import { useEffect } from 'react'
import { useMcUi } from '../contexts/McUiContext'
import type { Page } from '../components/Sidebar'

const VALID_TABS: ReadonlySet<string> = new Set([
  'inner_voice', 'chat', 'backlog', 'autonomy', 'workers',
  'memory', 'architecture', 'skills', 'tools', 'services',
  'settings', 'graph',
])

// Subscribes to /api/mc/events and applies navigate commands to the
// MC UI context. Single mount expected (Layout). Auto-reconnects when
// the EventSource closes — keeps the agent's reach alive across server
// restarts and transient network blips.
export function useMcNavigationEvents() {
  const { setCurrentTab, setPendingFocus } = useMcUi()

  useEffect(() => {
    let es: EventSource | null = null
    let closed = false
    let retryHandle: number | null = null

    const connect = () => {
      if (closed) return
      es = new EventSource('/api/mc/events')

      es.addEventListener('navigate', (ev: MessageEvent) => {
        try {
          const data = JSON.parse(ev.data)
          const tab = data?.tab
          const focusId = typeof data?.focus_id === 'string' && data.focus_id ? data.focus_id : null
          if (typeof tab === 'string' && VALID_TABS.has(tab)) {
            setCurrentTab(tab as Page)
            if (focusId) {
              setPendingFocus({ tab: tab as Page, focusId })
            }
          }
        } catch {
          // Ignore malformed payloads.
        }
      })

      es.onerror = () => {
        // EventSource auto-retries on transient failures; on hard close
        // (e.g., backend restart) we explicitly drop and reconnect.
        if (es) {
          es.close()
          es = null
        }
        if (!closed) {
          retryHandle = window.setTimeout(connect, 2000)
        }
      }
    }

    connect()

    return () => {
      closed = true
      if (retryHandle !== null) {
        window.clearTimeout(retryHandle)
        retryHandle = null
      }
      if (es) {
        es.close()
        es = null
      }
    }
  }, [setCurrentTab, setPendingFocus])
}
