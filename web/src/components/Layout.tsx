import { useState, useCallback, useEffect, useMemo } from 'react'
import Sidebar, { type Page } from './Sidebar'
import ChatPanel from './ChatPanel'
import SessionsPanel from './SessionsPanel'
import BacklogPage from './pages/BacklogPage'
import MemoryPage from './pages/MemoryPage'
import ServicesPage from './pages/ServicesPage'
import SkillsPage from './pages/SkillsPage'
import ToolsPage from './pages/ToolsPage'
import ArchitecturePageFull from './pages/ArchitecturePage'
import AutonomyPage from './pages/AutonomyPage'
import WorkersPage from './pages/WorkersPage'
import InnerVoicePage from './pages/InnerVoicePage'
import SettingsPage from './pages/SettingsPage'
import RightChatSidebar from './RightChatSidebar'
import { MessageCircle, PanelLeft, PanelLeftClose, Plus, ChevronDown, Bot, Menu } from 'lucide-react'
import { MessageProvider } from '../contexts/MessageContext'
import { useMcUi, useReportMcFocus, usePendingFocusFor } from '../contexts/McUiContext'
import { useMcStateSync } from '../hooks/useMcStateSync'
import { useMcNavigationEvents } from '../hooks/useMcNavigationEvents'
import { api, type ModelInfo } from '../api'
import { useIsMobile } from '../hooks/useIsMobile'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'

interface Slot {
  slotId: string
  sessionKey: string | null
  model: string
}

let slotCounter = 0
const nextSlotId = () => `slot_${Date.now()}_${++slotCounter}`

const GraphPage = () => (
  <div className="p-6">
    <h2 className="text-xl font-bold text-foreground">Graph</h2>
    <p className="text-muted-foreground mt-2">Coming soon...</p>
  </div>
)
const PAGES: Record<string, React.FC> = {
  services: ServicesPage,
  backlog: BacklogPage,
  memory: MemoryPage,
  graph: GraphPage,
  skills: SkillsPage,
  tools: ToolsPage,
  settings: SettingsPage,
  architecture: ArchitecturePageFull,
  autonomy: AutonomyPage,
  workers: WorkersPage,
  inner_voice: InnerVoicePage,  // Inner Voice (#345)
}

interface ModelMenuProps {
  models: ModelInfo[]
  currentModel: string
  visibleSlot: Slot | null
  visibleSlotId: string | null
  onSwitchModel: (slotId: string, model: string) => void
  triggerLabel: string
  triggerClassName?: string
}

function ModelMenu({
  models,
  currentModel,
  visibleSlot,
  visibleSlotId,
  onSwitchModel,
  triggerLabel,
  triggerClassName,
}: ModelMenuProps) {
  const handleSelect = async (model: string) => {
    if (visibleSlot?.sessionKey) {
      try {
        const result = await api.switchModel(model, visibleSlot.sessionKey)
        if (result.success && visibleSlotId) onSwitchModel(visibleSlotId, model)
      } catch (err) {
        console.error('Failed to switch model:', err)
      }
    } else if (visibleSlotId) {
      onSwitchModel(visibleSlotId, model)
    }
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className={cn('text-xs gap-1', triggerClassName)}>
          {triggerLabel}
          <ChevronDown className="w-3 h-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56 max-h-60 overflow-y-auto">
        {models.map(model => (
          <DropdownMenuItem
            key={model.name}
            onSelect={() => handleSelect(model.name)}
            className={cn(
              'flex-col items-start gap-0.5',
              model.name === currentModel && 'bg-primary/15 text-primary focus:bg-primary/20',
            )}
          >
            <span className="font-medium truncate w-full">{model.name}</span>
            <span className="text-xs text-muted-foreground truncate w-full">
              {model.alias} · {model.provider}
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default function Layout() {
  const isMobile = useIsMobile()
  const { currentTab: page, setCurrentTab: setPage } = useMcUi()
  // Mirror tab + focus to the backend; subscribe to navigate commands.
  useMcStateSync()
  useMcNavigationEvents()
  const [collapsed, setCollapsed] = useState(true)
  // Right chat sidebar — voice-driven dual-brain chat. Mounted on desktop;
  // owns its own collapse state + voice engagement (collapsing disengages
  // voice, expanding re-engages). Persisted via localStorage internally.
  const [slots, setSlots] = useState<Slot[]>([])
  const [visibleSlotId, setVisibleSlotId] = useState<string | null>(null)
  const [activeSessions, setActiveSessions] = useState<Set<string>>(new Set())
  const [showSessions, setShowSessions] = useState(true)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [showAgentDetails, setShowAgentDetails] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const visibleSlot = useMemo(() => slots.find(s => s.slotId === visibleSlotId) ?? null, [slots, visibleSlotId])
  const currentModel = visibleSlot?.model ?? ''

  // Mirror chat focus (visible slot's session) for the agent.
  useReportMcFocus(
    'chat',
    visibleSlot?.sessionKey
      ? { kind: 'session', id: visibleSlot.sessionKey }
      : null,
  )

  // Apply incoming chat focus from mc_navigate by opening the requested
  // session — same path the URL ?session= and SessionsPanel use.
  const chatPendingFocus = usePendingFocusFor('chat')
  const sessionsPanelRefreshTrigger = useMemo(() => slots.map(s => s.sessionKey ?? 'null').join(','), [slots])

  // Voice mode is owned by RightChatSidebar — on desktop it's always
  // mounted; on mobile it's mounted as the chat page itself.

  const handleNewSession = () => {
    const existingBlank = slots.find(s => s.sessionKey === null)
    if (existingBlank) {
      setVisibleSlotId(existingBlank.slotId)
      return
    }
    const id = nextSlotId()
    setSlots(prev => [...prev, { slotId: id, sessionKey: null, model: currentModel }])
    setVisibleSlotId(id)
  }

  // Auto-load session from URL query param ?session=<key>
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const sessionKey = params.get('session')
    if (sessionKey) {
      handleOpenSession(sessionKey)
    }
  }, [])

  const handleOpenSession = useCallback((sessionKey: string) => {
    setSlots(prev => {
      const existing = prev.find(s => s.sessionKey === sessionKey)
      if (existing) {
        setVisibleSlotId(existing.slotId)
        return prev
      }
      const id = nextSlotId()
      setVisibleSlotId(id)
      return [...prev, { slotId: id, sessionKey, model: '' }]
    })
    setPage('chat')
  }, [setPage])

  // mc_navigate tab=chat focus_id=<session-id> → open that session.
  useEffect(() => {
    if (chatPendingFocus) handleOpenSession(chatPendingFocus)
  }, [chatPendingFocus, handleOpenSession])

  const handleActiveSessionChange = useCallback((slotId: string, key: string | null) => {
    setSlots(prev => prev.map(s => s.slotId === slotId ? { ...s, sessionKey: key } : s))
  }, [])

  const handleThinkingChange = useCallback((sessionKey: string | null, thinking: boolean) => {
    if (!sessionKey) return
    setActiveSessions(prev => {
      const next = new Set(prev)
      if (thinking) next.add(sessionKey)
      else next.delete(sessionKey)
      return next
    })
  }, [])

  const handleSlotModelSwitch = useCallback((slotId: string, model: string) => {
    setSlots(prev => prev.map(s => s.slotId === slotId ? { ...s, model } : s))
  }, [])

  // Load models on mount
  useEffect(() => {
    api.getModels().then(result => {
      if (result.models) {
        setModels(result.models)
      }
    }).catch(err => {
      console.warn('Failed to load models:', err)
    })
  }, [])

  // Initialize first slot when on chat page with no slots yet.
  // Skipped on mobile — the chat page is RightChatSidebar there, which
  // owns its own session resolution and never reads the slot list.
  useEffect(() => {
    if (isMobile) return
    if (page !== 'chat' || slots.length > 0) return
    api.listSessions().then(result => {
      const mostRecent = result.sessions?.[0]
      const id = nextSlotId()
      setSlots([{ slotId: id, sessionKey: mostRecent?.session_key ?? null, model: '' }])
      setVisibleSlotId(id)
    }).catch(() => {
      const id = nextSlotId()
      setSlots([{ slotId: id, sessionKey: null, model: '' }])
      setVisibleSlotId(id)
    })
  }, [page, slots.length, isMobile])

  const PageComponent = PAGES[page]

  const handleMobileNavigate = (p: Page) => {
    setPage(p)
    setMobileMenuOpen(false)
  }

  const triggerModelLabel = currentModel
    ? (models.find(m => m.name === currentModel)?.name || currentModel)
    : 'Model'

  return (
    <MessageProvider>
      <div className={cn('h-screen flex bg-background text-foreground', isMobile && 'flex-col')}>
        {/* Desktop sidebar */}
        {!isMobile && (
          <Sidebar
            active={page}
            onNavigate={setPage}
            collapsed={collapsed}
            onToggleCollapse={() => setCollapsed(c => !c)}
          />
        )}

        {/* Mobile top bar */}
        {isMobile && (
          <div className="flex items-center justify-between px-3 py-2 bg-card border-b border-border flex-shrink-0">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setMobileMenuOpen(true)}
                aria-label="Open menu"
              >
                <Menu className="w-5 h-5" />
              </Button>
              <img src="/lloyd.jpg" alt="Lloyd" className="w-6 h-6 rounded-md object-cover" />
              <span className="text-sm font-bold text-foreground">LLOYD</span>
            </div>
            <div className="flex items-center gap-1">
              {/* Chat-specific controls are intentionally absent on mobile —
                  the chat page is RightChatSidebar, which owns its own
                  new-session, stop, and details controls. */}
            </div>
          </div>
        )}

        {/* Mobile menu sheet */}
        {isMobile && (
          <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
            <SheetContent side="left" className="p-0 w-64 flex flex-col">
              <SheetHeader className="px-4 py-3 border-b border-border">
                <div className="flex items-center gap-2">
                  <img src="/lloyd.jpg" alt="Lloyd" className="w-7 h-7 rounded-md object-cover" />
                  <div className="text-left">
                    <SheetTitle className="text-sm font-bold leading-tight">LLOYD</SheetTitle>
                    <div className="text-[10px] text-muted-foreground -mt-0.5">Mission Control</div>
                  </div>
                </div>
              </SheetHeader>
              <Sidebar
                active={page}
                onNavigate={handleMobileNavigate}
                collapsed={false}
                onToggleCollapse={() => {}}
                isMobile
              />
            </SheetContent>
          </Sheet>
        )}

        <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {/* Desktop chat header */}
          {!isMobile && page === 'chat' && (
            <div className="flex items-center justify-between flex-shrink-0 px-6 pt-6 pb-2">
              <div className="flex items-center gap-3">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-muted-foreground hover:text-primary"
                  onClick={() => setShowSessions(s => !s)}
                  aria-label={showSessions ? 'Hide sessions' : 'Show sessions'}
                >
                  {showSessions ? <PanelLeftClose className="w-5 h-5" /> : <PanelLeft className="w-5 h-5" />}
                </Button>
                <MessageCircle className="w-5 h-5 text-primary" />
                <h2 className="text-lg font-semibold text-foreground">Chat</h2>
              </div>
              <div className="flex items-center gap-1.5">
                {models.length > 0 && (
                  <ModelMenu
                    models={models}
                    currentModel={currentModel}
                    visibleSlot={visibleSlot}
                    visibleSlotId={visibleSlotId}
                    onSwitchModel={handleSlotModelSwitch}
                    triggerLabel={triggerModelLabel}
                    triggerClassName="text-muted-foreground hover:text-primary hover:bg-primary/10"
                  />
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowAgentDetails(v => !v)}
                  className={cn(
                    'text-xs gap-1.5',
                    showAgentDetails
                      ? 'text-primary bg-primary/15 hover:bg-primary/20'
                      : 'text-muted-foreground hover:text-primary hover:bg-primary/10',
                  )}
                >
                  <Bot className="w-3.5 h-3.5" />
                  Agent Details
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleNewSession}
                  className="text-xs gap-1.5 text-muted-foreground hover:text-primary hover:bg-primary/10"
                >
                  <Plus className="w-3.5 h-3.5" />
                  New Session
                </Button>
              </div>
            </div>
          )}

          {/* Chat area — always mounted so session state + messages survive tab switches.
              Hidden via display:none when another page is active.
              On mobile this slot-based chat is replaced by RightChatSidebar
              (rendered below), so it's only mounted on desktop. */}
          {!isMobile && (
            <div
              className={cn(
                'flex-1 flex min-h-0 overflow-hidden flex-shrink-0 mx-6 mb-6',
                page === 'chat' ? '' : 'hidden',
              )}
            >
              <div className="flex flex-1 flex-row h-full bg-card overflow-hidden border border-border rounded-xl">
                {showSessions && (
                  <div className="w-64 border-r border-border flex-shrink-0">
                    <SessionsPanel
                      onSwitchSession={(key) => handleOpenSession(key)}
                      currentSessionKey={visibleSlot?.sessionKey ?? null}
                      activeSessions={activeSessions}
                      refreshTrigger={sessionsPanelRefreshTrigger}
                    />
                  </div>
                )}

                {/* Chat panels — one per slot, only visible one shown */}
                <div className="flex-1 flex flex-col min-h-0 overflow-hidden relative">
                  {slots.map(slot => (
                    <div
                      key={slot.slotId}
                      className={cn(
                        'absolute inset-0 flex flex-col',
                        slot.slotId === visibleSlotId ? '' : 'hidden',
                      )}
                    >
                      <ChatPanel
                        requestedSessionKey={slot.sessionKey}
                        onSessionLoaded={() => {}}
                        onActiveSessionChange={(key) => handleActiveSessionChange(slot.slotId, key)}
                        onThinkingChange={(thinking, _toolName) => handleThinkingChange(slot.sessionKey, thinking)}
                        onModelSwitch={(model) => handleSlotModelSwitch(slot.slotId, model)}
                        currentSessionKey={slot.sessionKey}
                        showAgentDetails={showAgentDetails}
                        pendingModel={slot.model || models[0]?.name}
                        visible={page === 'chat' && slot.slotId === visibleSlotId}
                        isMobile={isMobile}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Mobile chat: RightChatSidebar full-width as the primary UI.
              Hidden via display:none when another page is active so its
              session/voice state survives tab switches. */}
          {isMobile && (
            <div
              className={cn(
                'flex-1 flex min-h-0 overflow-hidden flex-shrink-0',
                page === 'chat' ? '' : 'hidden',
              )}
            >
              <RightChatSidebar isMobile />
            </div>
          )}

          {/* Other pages */}
          {PageComponent && <PageComponent />}
        </main>

        {/* Right chat sidebar — voice-driven dual-brain conversation. Always
            mounted on desktop (per "not collapsible" decision); owns the
            LiveKit room for its session. */}
        {!isMobile && <RightChatSidebar />}
      </div>
    </MessageProvider>
  )
}
