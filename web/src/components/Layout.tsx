import { useState, useCallback, useEffect } from 'react'
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
import UsagePage from './pages/UsagePage'
import { MessageCircle, PanelLeft, PanelLeftClose, Plus, ChevronDown, Bot, Menu, X } from 'lucide-react'
import { MessageProvider } from '../contexts/MessageContext'
import { api, type ModelInfo } from '../api'
import { useIsMobile } from '../hooks/useIsMobile'

const DashboardPage = UsagePage
const GraphPage = () => <div className="p-6"><h2 className="text-xl font-bold">Graph</h2><p className="text-slate-400 mt-2">Coming soon...</p></div>
const SettingsPage = () => <div className="p-6"><h2 className="text-xl font-bold">Settings</h2><p className="text-slate-400 mt-2">Coming soon...</p></div>

const PAGES: Record<string, React.FC> = {
  services: ServicesPage,
  dashboard: DashboardPage,
  backlog: BacklogPage,
  memory: MemoryPage,
  graph: GraphPage,
  skills: SkillsPage,
  tools: ToolsPage,
  settings: SettingsPage,
  architecture: ArchitecturePageFull,
  autonomy: AutonomyPage,
}

export default function Layout() {
  const isMobile = useIsMobile()
  const [page, setPage] = useState<Page>('chat')
  const [collapsed, setCollapsed] = useState(false)
  const [chatSessionKey, setChatSessionKey] = useState<string | null>(null)
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null)
  const [showSessions, setShowSessions] = useState(true) // Show sessions panel by default
  const [models, setModels] = useState<ModelInfo[]>([])
  const [currentModel, setCurrentModel] = useState<string>('')
  const [showModelDropdown, setShowModelDropdown] = useState(false)
  const [showAgentDetails, setShowAgentDetails] = useState(false)
  const [sessionRefreshTrigger, setSessionRefreshTrigger] = useState(0)
  const [isNewSession, setIsNewSession] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [mobileSessionsOpen, setMobileSessionsOpen] = useState(false)

  const formatSessionKey = (key: string) => {
    const parts = key.split(':')
    const name = parts[parts.length - 1]
    if (name === 'main') return 'Main'
    return name.length > 12 ? name.slice(0, 12) + '...' : name
  }

  const handleNewSession = () => {
    setChatSessionKey(null)
    setActiveSessionKey(null)
    setIsNewSession(true)
    localStorage.removeItem('mc_session_id')
  }

  // Auto-load session from URL query param ?session=<key>
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const sessionKey = params.get('session')
    if (sessionKey) {
      setChatSessionKey(sessionKey)
      setPage('chat')
    }
  }, [])

  const handleOpenSession = useCallback((sessionKey: string) => {
    setChatSessionKey(sessionKey)
    setIsNewSession(false)
    setPage('chat')
  }, [])

  const handleSessionLoaded = useCallback(() => {
    setChatSessionKey(null)
  }, [])

  // Load models on mount
  useEffect(() => {
    api.getModels().then(result => {
      if (result.models) {
        setModels(result.models)
        // Try to get current model from config
        // For now, default to first model if available
        if (result.models.length > 0) {
          setCurrentModel(result.models[0].name)
        }
      }
    }).catch(err => {
      console.warn('Failed to load models:', err)
    })
  }, [])

  // Auto-load most recent session when on chat page with no session loaded
  // (skip if user explicitly clicked New Session)
  useEffect(() => {
    if (page === 'chat' && !chatSessionKey && !activeSessionKey && !isNewSession) {
      api.listSessions().then(result => {
        if (result.sessions && result.sessions.length > 0) {
          const mostRecent = result.sessions[0]
          if (mostRecent.session_key) {
            setChatSessionKey(mostRecent.session_key)
          }
        }
      }).catch(err => {
        console.warn('Failed to load sessions for auto-load:', err)
      })
    }
  }, [page, chatSessionKey, activeSessionKey, isNewSession])

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
            sessionKey={chatSessionKey || activeSessionKey}
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
                            const currentSession = chatSessionKey || activeSessionKey
                            if (currentSession) {
                              try {
                                const result = await api.switchModel(model.name, currentSession)
                                if (result.success) {
                                  setCurrentModel(model.name)
                                  setShowModelDropdown(false)
                                }
                              } catch (err) {
                                console.error('Failed to switch model:', err)
                              }
                            } else {
                              setCurrentModel(model.name)
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
                sessionKey={chatSessionKey || activeSessionKey}
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
                  currentSessionKey={chatSessionKey || activeSessionKey}
                  refreshTrigger={sessionRefreshTrigger}
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
                              const currentSession = chatSessionKey || activeSessionKey
                              if (currentSession) {
                                try {
                                  const result = await api.switchModel(model.name, currentSession)
                                  if (result.success) {
                                    setCurrentModel(model.name)
                                    setShowModelDropdown(false)
                                  }
                                } catch (err) {
                                  console.error('Failed to switch model:', err)
                                }
                              } else {
                                setCurrentModel(model.name)
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

          {/* Main content area */}
          {page === 'chat' && (
            <div className={`flex-1 flex min-h-0 overflow-hidden flex-shrink-0 ${isMobile ? 'm-0' : 'mx-6 mb-6'}`}>
              <div className={`flex flex-1 flex-row h-full bg-surface-1 ${isMobile ? '' : 'border border-surface-3/50 rounded-xl'} overflow-hidden`}>
                {/* Sessions Panel - desktop only */}
                {!isMobile && showSessions && (
                  <div className="w-64 border-r border-surface-3/30 flex-shrink-0">
                    <SessionsPanel
                      onSwitchSession={(key) => handleOpenSession(key)}
                      currentSessionKey={chatSessionKey || activeSessionKey}
                      refreshTrigger={sessionRefreshTrigger}
                    />
                  </div>
                )}

                {/* Chat panel */}
                <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
                  <ChatPanel
                    key={chatSessionKey || activeSessionKey || 'new'}
                    requestedSessionKey={chatSessionKey}
                    onSessionLoaded={handleSessionLoaded}
                    onActiveSessionChange={(key) => { setActiveSessionKey(key); setIsNewSession(false); setSessionRefreshTrigger(n => n + 1) }}
                    onModelSwitch={setCurrentModel}
                    currentSessionKey={chatSessionKey || activeSessionKey}
                    showAgentDetails={showAgentDetails}
                    pendingModel={currentModel}
                  />
                </div>
              </div>
            </div>
          )}

            {/* Other pages */}
            {PageComponent && <PageComponent />}
          </main>
        </div>
      </MessageProvider>
  )
}
