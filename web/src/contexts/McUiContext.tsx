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

export interface IdeState {
  open_folder?: string
  visible_file?: string
  open_tabs?: string[]
}

export type IdeActionKind = 'open_folder' | 'close_tab'

export interface PendingIdeAction {
  kind: IdeActionKind
  path: string
  // Bumped on each new action so consumers can re-fire on duplicates.
  seq: number
}

export interface PendingFileChange {
  path: string
  deleted: boolean
  // Bumped on each new event so consumers can re-fire on duplicates.
  seq: number
}

export interface PendingCloseModal {
  tab: Page
  // Bumped on each new event so consumers can re-fire on duplicates.
  seq: number
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

  // IDE tab state mirror — IdePage reports its open folder / visible file /
  // open tabs here, useMcStateSync sends them to the backend.
  ideState: IdeState | null
  reportIdeState: (next: IdeState | null) => void

  // Set by the navigate-events hook when the agent issues an ide_action.
  // IdePage reads + applies it, no explicit consume needed (sequence-gated).
  pendingIdeAction: PendingIdeAction | null
  setPendingIdeAction: (next: PendingIdeAction | null) => void

  // Set by the navigate-events hook when inotify fires for a file inside
  // the IDE's open folder. IdePage applies it (silent reload, animate,
  // or conflict banner depending on tab state).
  pendingFileChange: PendingFileChange | null
  setPendingFileChange: (next: PendingFileChange | null) => void

  // Set by the navigate-events hook when the agent issues a close_modal.
  // Pages that own modals read + apply it (sequence-gated).
  pendingCloseModal: PendingCloseModal | null
  setPendingCloseModal: (next: PendingCloseModal | null) => void
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
  const [ideState, setIdeState] = useState<IdeState | null>(null)
  const [pendingIdeAction, setPendingIdeAction] = useState<PendingIdeAction | null>(null)
  const [pendingFileChange, setPendingFileChange] = useState<PendingFileChange | null>(null)
  const [pendingCloseModal, setPendingCloseModal] = useState<PendingCloseModal | null>(null)

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

  const reportIdeState = useCallback((next: IdeState | null) => {
    setIdeState(prev => {
      const a = prev ?? {}
      const b = next ?? {}
      const sameTabs = (a.open_tabs ?? []).length === (b.open_tabs ?? []).length
        && (a.open_tabs ?? []).every((t, i) => t === (b.open_tabs ?? [])[i])
      const same = a.open_folder === b.open_folder
        && a.visible_file === b.visible_file
        && sameTabs
      if (same) return prev
      return next
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
    ideState,
    reportIdeState,
    pendingIdeAction,
    setPendingIdeAction,
    pendingFileChange,
    setPendingFileChange,
    pendingCloseModal,
    setPendingCloseModal,
  }), [currentTab, focusByTab, reportFocus, pendingFocus, consumePendingFocus,
       ideState, reportIdeState, pendingIdeAction, pendingFileChange,
       pendingCloseModal])

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
