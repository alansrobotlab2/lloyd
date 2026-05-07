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
import VoicePreviewPage from './pages/VoicePreviewPage'
import { MessageCircle, PanelLeft, PanelLeftClose, Plus, ChevronDown, Bot, Menu, X } from 'lucide-react'
import { MessageProvider } from '../contexts/MessageContext'
import { api, type ModelInfo } from '../api'
import { useIsMobile } from '../hooks/useIsMobile'

interface Slot {
  slotId: string
  sessionKey: string | null
  model: string
}

let slotCounter = 0
const nextSlotId = () => `slot_${Date.now()}_${++slotCounter}`

const GraphPage = () => <div className="p-6"><h2 className="text-xl font-bold">Graph</h2><p className="text-slate-400 mt-2">Coming soon...</p></div>
const SettingsPage = () => <div className="p-6"><h2 className="text-xl font-bold">Settings</h2><p className="text-slate-400 mt-2">Coming soon...</p></div>

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
  voice_preview: VoicePreviewPage,  // Phase 2A — aura visualizer with mock state
}

export default function Layout() {
  const isMobile = useIsMobile()
  const [page, setPage] = useState<Page>('chat')
  const [collapsed, setCollapsed] = useState(false)
  const [slots, setSlots] = useState<Slot[]>([])
  const [visibleSlotId, setVisibleSlotId] = useState<string | null>(null)
  const [activeSessions, setActiveSessions] = useState<Set<string>>(new Set())
  const [showSessions, setShowSessions] = useState(true)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [showModelDropdown, setShowModelDropdown] = useState(false)
  const [showAgentDetails, setShowAgentDetails] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [mobileSessionsOpen, setMobileSessionsOpen] = useState(false)

  const visibleSlot = useMemo(() => slots.find(s => s.slotId === visibleSlotId) ?? null, [slots, visibleSlotId])
  const currentModel = visibleSlot?.model ?? ''
  const sessionsPanelRefreshTrigger = useMemo(() => slots.map(s => s.sessionKey ?? 'null').join(','), [slots])

  // Route voice transcripts to whichever session is focused in the foreground tab.
  // Chat tab uses the visible slot; Inner Voice manages its own (in InnerVoicePage).
  // Other tabs leave the override alone so the previous owner's choice persists
  // until the user goes back to chat.
  useEffect(() => {
    if (page !== 'chat') return
    const sid = visibleSlot?.sessionKey ?? null
    api.voiceSetActiveSession(sid).catch(() => {})
  }, [page, visibleSlot?.sessionKey])

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
  }, [])

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

  // Initialize first slot when on chat page with no slots yet
  useEffect(() => {
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
  }, [page, slots.length])

  const PageComponent = PAGES[page]

  // Mobile navigation handler — close menu after navigating
  const handleMobileNavigate = (p: Page) => {
    setPage(p)
    setMobileMenuOpen(false)
  }

  return (
    <MessageProvider>
      <div className={`h-screen flex ${isMobile ? 'flex-col' : ''} bg-surface-0`}>
        {/* Desktop sidebar */}
        {!isMobile && (
          <Sidebar
            active={page}
            onNavigate={setPage}
            collapsed={collapsed}
            onToggleCollapse={() => setCollapsed((c) => !c)}
            sessionKey={visibleSlot?.sessionKey}
          />
        )}

        {/* Mobile top bar */}
        {isMobile && (
          <div className="flex items-center justify-between px-3 py-2 bg-surface-1 border-b border-surface-3/30 flex-shrink-0">
            <div className="flex items-center gap-2">
              <button onClick={() => setMobileMenuOpen(true)} className="text-slate-400 p-1">
                <Menu className="w-5 h-5" />
              </button>
              <img src="/lloyd.jpg" alt="Lloyd" className="w-6 h-6 rounded-lg object-cover" />
              <span className="text-sm font-bold text-slate-200">LLOYD</span>
            </div>
            <div className="flex items-center gap-1">
              {page === 'chat' && (
                <>
                  <button
                    onClick={() => setMobileSessionsOpen(true)}
                    className="text-slate-400 p-1.5"
                    title="Sessions"
                  >
                    <PanelLeft className="w-4 h-4" />
                  </button>
                  <button
                    onClick={handleNewSession}
                    className="text-slate-400 p-1.5"
                    title="New Session"
                  >
                    <Plus className="w-4 h-4" />
                  </button>
                </>
              )}
              {models.length > 0 && page === 'chat' && (
                <div className="relative">
                  <button
                    onClick={() => setShowModelDropdown((v) => !v)}
                    className="flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium text-slate-400"
                  >
                    {currentModel ? (models.find(m => m.name === currentModel)?.alias || currentModel) : 'Model'}
                    <ChevronDown className="w-3 h-3" />
                  </button>
                  {showModelDropdown && (
                    <div className="absolute top-full right-0 mt-1 w-48 bg-surface-1 border border-surface-3/50 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
                      {models.map((model) => (
                        <button
                          key={model.name}
                          onClick={async () => {
                            if (visibleSlot?.sessionKey) {
                              try {
                                const result = await api.switchModel(model.name, visibleSlot.sessionKey)
                                if (result.success && visibleSlotId) {
                                  handleSlotModelSwitch(visibleSlotId, model.name)
                                  setShowModelDropdown(false)
                                }
                              } catch (err) {
                                console.error('Failed to switch model:', err)
                              }
                            } else if (visibleSlotId) {
                              handleSlotModelSwitch(visibleSlotId, model.name)
                              setShowModelDropdown(false)
                            }
                          }}
                          className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                            model.name === currentModel
                              ? 'bg-brand-500/20 text-brand-400'
                              : 'hover:bg-surface-2 text-slate-200'
                          }`}
                        >
                          <div className="font-medium truncate">{model.name}</div>
                          <div className="text-xs text-slate-500 truncate">{model.alias} • {model.provider}</div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Mobile slide-out menu overlay */}
        {isMobile && mobileMenuOpen && (
          <div className="fixed inset-0 z-50 flex">
            <div className="absolute inset-0 bg-black/60" onClick={() => setMobileMenuOpen(false)} />
            <div className="relative w-64 bg-surface-1 h-full flex flex-col animate-slide-in-left">
              <div className="flex items-center justify-between px-4 py-3 border-b border-surface-3/30">
                <div className="flex items-center gap-2">
                  <img src="/lloyd.jpg" alt="Lloyd" className="w-7 h-7 rounded-lg object-cover" />
                  <div>
                    <div className="text-sm font-bold text-slate-200">LLOYD</div>
                    <div className="text-[10px] text-slate-500 -mt-0.5">Mission Control</div>
                  </div>
                </div>
                <button onClick={() => setMobileMenuOpen(false)} className="text-slate-400 p-1">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <Sidebar
                active={page}
                onNavigate={handleMobileNavigate}
                collapsed={false}
                onToggleCollapse={() => {}}
                sessionKey={visibleSlot?.sessionKey}
                isMobile
              />
            </div>
          </div>
        )}

        {/* Mobile sessions overlay */}
        {isMobile && mobileSessionsOpen && (
          <div className="fixed inset-0 z-50 flex">
            <div className="absolute inset-0 bg-black/60" onClick={() => setMobileSessionsOpen(false)} />
            <div className="relative w-72 bg-surface-1 h-full flex flex-col animate-slide-in-left">
              <div className="flex items-center justify-between px-4 py-3 border-b border-surface-3/30">
                <span className="text-sm font-semibold text-slate-200">Sessions</span>
                <button onClick={() => setMobileSessionsOpen(false)} className="text-slate-400 p-1">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto">
                <SessionsPanel
                  onSwitchSession={(key) => { handleOpenSession(key); setMobileSessionsOpen(false) }}
                  currentSessionKey={visibleSlot?.sessionKey ?? null}
                  activeSessions={activeSessions}
                  refreshTrigger={sessionsPanelRefreshTrigger}
                />
              </div>
            </div>
          </div>
        )}

        <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {/* Desktop page header */}
          {!isMobile && page === 'chat' && (
            <div className="flex items-center justify-between flex-shrink-0 px-6 pt-6 pb-2">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowSessions((s) => !s)}
                  title={showSessions ? "Hide sessions" : "Show sessions"}
                  className="text-slate-400 hover:text-brand-400 transition-colors"
                >
                  {showSessions ? <PanelLeftClose className="w-5 h-5" /> : <PanelLeft className="w-5 h-5" />}
                </button>
                <MessageCircle className="w-5 h-5 text-brand-400" />
                <h2 className="text-lg font-semibold text-slate-200">Chat</h2>
              </div>
              <div className="flex items-center gap-1.5">
                {models.length > 0 && (
                  <div className="relative">
                    <button
                      onClick={() => setShowModelDropdown((v) => !v)}
                      className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-brand-400 hover:bg-brand-500/10 transition-colors"
                    >
                      {currentModel ? models.find(m => m.name === currentModel)?.name || currentModel : 'Model'}
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    {showModelDropdown && (
                      <div className="absolute top-full right-0 mt-1 w-48 bg-surface-1 border border-surface-3/50 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
                        {models.map((model) => (
                          <button
                            key={model.name}
                            onClick={async () => {
                              if (visibleSlot?.sessionKey) {
                                try {
                                  const result = await api.switchModel(model.name, visibleSlot.sessionKey)
                                  if (result.success && visibleSlotId) {
                                    handleSlotModelSwitch(visibleSlotId, model.name)
                                    setShowModelDropdown(false)
                                  }
                                } catch (err) {
                                  console.error('Failed to switch model:', err)
                                }
                              } else if (visibleSlotId) {
                                handleSlotModelSwitch(visibleSlotId, model.name)
                                setShowModelDropdown(false)
                              }
                            }}
                            className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                              model.name === currentModel
                                ? 'bg-brand-500/20 text-brand-400'
                                : 'hover:bg-surface-2 text-slate-200'
                            }`}
                          >
                            <div className="font-medium truncate">{model.name}</div>
                            <div className="text-xs text-slate-500 truncate">{model.alias} • {model.provider}</div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                <button
                  onClick={() => setShowAgentDetails((v) => !v)}
                  title={showAgentDetails ? "Hide agent details" : "Show agent details"}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    showAgentDetails
                      ? 'text-brand-400 bg-brand-500/15'
                      : 'text-slate-400 hover:text-brand-400 hover:bg-brand-500/10'
                  }`}
                >
                  <Bot className="w-3.5 h-3.5" />
                  Agent Details
                </button>
                <button
                  onClick={handleNewSession}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-brand-400 hover:bg-brand-500/10 transition-colors"
                >
                  <Plus className="w-3.5 h-3.5" />
                  New Session
                </button>
              </div>
            </div>
          )}

          {/* Chat area — always mounted so session state + messages survive tab switches.
              Hidden via display:none when another page is active so no layout space is taken. */}
          <div className={`flex-1 flex min-h-0 overflow-hidden flex-shrink-0 ${isMobile ? 'm-0' : 'mx-6 mb-6'} ${page === 'chat' ? '' : 'hidden'}`}>
            <div className={`flex flex-1 flex-row h-full bg-surface-1 ${isMobile ? '' : 'border border-surface-3/50 rounded-xl'} overflow-hidden`}>
              {/* Sessions Panel - desktop only */}
              {!isMobile && showSessions && (
                <div className="w-64 border-r border-surface-3/30 flex-shrink-0">
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
                    className={`absolute inset-0 flex flex-col ${slot.slotId === visibleSlotId ? '' : 'hidden'}`}
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

            {/* Other pages */}
            {PageComponent && <PageComponent />}
          </main>
        </div>
      </MessageProvider>
  )
}
