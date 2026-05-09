import { useEffect, useRef } from 'react'
import { useMcUi, type IdeActionKind } from '../contexts/McUiContext'
import type { Page } from '../components/Sidebar'

const VALID_TABS: ReadonlySet<string> = new Set([
  'inner_voice', 'chat', 'backlog', 'autonomy', 'workers',
  'memory', 'architecture', 'skills', 'tools', 'services',
  'settings', 'graph', 'ide',
])

const VALID_IDE_ACTIONS: ReadonlySet<string> = new Set(['open_folder', 'close_tab'])

// Subscribes to /api/mc/events and applies navigate + ide_action commands
// to the MC UI context. Single mount expected (Layout). Auto-reconnects
// when the EventSource closes — keeps the agent's reach alive across
// server restarts and transient network blips.
export function useMcNavigationEvents() {
  const {
    setCurrentTab, setPendingFocus, setPendingIdeAction, setPendingFileChange,
    setPendingCloseModal,
  } = useMcUi()
  const ideSeq = useRef(0)
  const fileSeq = useRef(0)
  const closeSeq = useRef(0)

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

      es.addEventListener('ide_action', (ev: MessageEvent) => {
        try {
          const data = JSON.parse(ev.data)
          const kind = data?.kind
          const path = data?.path
          if (typeof kind === 'string' && VALID_IDE_ACTIONS.has(kind)
              && typeof path === 'string' && path) {
            ideSeq.current += 1
            setPendingIdeAction({
              kind: kind as IdeActionKind,
              path,
              seq: ideSeq.current,
            })
          }
        } catch {
          // Ignore malformed payloads.
        }
      })

      es.addEventListener('close_modal', (ev: MessageEvent) => {
        try {
          const data = JSON.parse(ev.data)
          const tab = data?.tab
          if (typeof tab === 'string' && VALID_TABS.has(tab)) {
            closeSeq.current += 1
            setPendingCloseModal({ tab: tab as Page, seq: closeSeq.current })
          }
        } catch {
          // Ignore malformed payloads.
        }
      })

      es.addEventListener('file_changed', (ev: MessageEvent) => {
        try {
          const data = JSON.parse(ev.data)
          const path = data?.path
          const deleted = !!data?.deleted
          if (typeof path === 'string' && path) {
            fileSeq.current += 1
            setPendingFileChange({
              path,
              deleted,
              seq: fileSeq.current,
            })
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
  }, [setCurrentTab, setPendingFocus, setPendingIdeAction, setPendingFileChange, setPendingCloseModal])
}
