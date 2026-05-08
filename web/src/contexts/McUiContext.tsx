import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { Page } from '../components/Sidebar'

export interface McFocus {
  kind: string
  id: string
  label?: string
}

interface PendingFocus {
  tab: Page
  focusId: string
}

interface McUiContextValue {
  currentTab: Page
  setCurrentTab: (tab: Page) => void

  // Per-tab focus, reported up by each page.
  focusByTab: Partial<Record<Page, McFocus | null>>
  reportFocus: (tab: Page, focus: McFocus | null) => void

  // Set by the navigate-events hook when the agent issues a navigate command.
  // Pages consume + clear it when they apply the requested focus.
  pendingFocus: PendingFocus | null
  setPendingFocus: (next: PendingFocus | null) => void
  consumePendingFocus: (tab: Page) => string | null
}

const McUiContext = createContext<McUiContextValue | null>(null)

interface ProviderProps {
  children: React.ReactNode
  initialTab?: Page
}

export function McUiProvider({ children, initialTab = 'inner_voice' }: ProviderProps) {
  const [currentTab, setCurrentTab] = useState<Page>(initialTab)
  const [focusByTab, setFocusByTab] = useState<Partial<Record<Page, McFocus | null>>>({})
  const [pendingFocus, setPendingFocus] = useState<PendingFocus | null>(null)

  // reportFocus is called from inside render via a useEffect in each page.
  // Skip state updates that wouldn't change anything, otherwise React loops.
  const reportFocus = useCallback((tab: Page, focus: McFocus | null) => {
    setFocusByTab(prev => {
      const existing = prev[tab] ?? null
      const same = (existing && focus &&
        existing.kind === focus.kind &&
        existing.id === focus.id &&
        (existing.label ?? null) === (focus.label ?? null)
      ) || (!existing && !focus)
      if (same) return prev
      return { ...prev, [tab]: focus }
    })
  }, [])

  const consumePendingFocus = useCallback((tab: Page) => {
    if (!pendingFocus || pendingFocus.tab !== tab) return null
    const id = pendingFocus.focusId
    setPendingFocus(null)
    return id
  }, [pendingFocus])

  const value = useMemo<McUiContextValue>(() => ({
    currentTab,
    setCurrentTab,
    focusByTab,
    reportFocus,
    pendingFocus,
    setPendingFocus,
    consumePendingFocus,
  }), [currentTab, focusByTab, reportFocus, pendingFocus, consumePendingFocus])

  return <McUiContext.Provider value={value}>{children}</McUiContext.Provider>
}

export function useMcUi(): McUiContextValue {
  const ctx = useContext(McUiContext)
  if (!ctx) throw new Error('useMcUi must be used inside <McUiProvider>')
  return ctx
}

// Each page calls this once with its tab id and current focused item.
// Pass null to clear.
export function useReportMcFocus(tab: Page, focus: McFocus | null) {
  const { reportFocus } = useMcUi()
  // Stable identity for focus to avoid pointless re-runs.
  const key = focus ? `${focus.kind}:${focus.id}:${focus.label ?? ''}` : ''
  const prevKey = useRef<string>('__init__')
  useEffect(() => {
    if (prevKey.current === key) return
    prevKey.current = key
    reportFocus(tab, focus)
    // We intentionally depend on the derived key, not the focus object,
    // so callers can pass a freshly-computed focus each render without
    // looping.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, tab])
}

// Pages that can apply an incoming focus call this to learn what the
// agent asked for. Returns the focus id once, then null after consumption.
export function usePendingFocusFor(tab: Page): string | null {
  const { pendingFocus, consumePendingFocus } = useMcUi()
  const [taken, setTaken] = useState<string | null>(null)
  useEffect(() => {
    if (pendingFocus && pendingFocus.tab === tab) {
      const id = consumePendingFocus(tab)
      setTaken(id)
    }
  }, [pendingFocus, tab, consumePendingFocus])
  // Clear the one-shot value on next render so callers don't re-apply.
  useEffect(() => {
    if (taken !== null) {
      const t = setTimeout(() => setTaken(null), 0)
      return () => clearTimeout(t)
    }
  }, [taken])
  return taken
}
