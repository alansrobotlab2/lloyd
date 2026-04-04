import { useState, useEffect, useCallback } from 'react'
import {
  ChartArea,
  LayoutList,
  Brain,
  Sparkles,
  Users,
  Wrench,
  LayoutGrid,
  Settings,
  Bot,
  MessageCircle,
  ChevronsLeft,
  ChevronsRight,
  Briefcase,
  Code2,
  Lightbulb,
  Mic,
  MicOff,
  Power,
  Radio,
  Volume2,
  VolumeX,
  Zap,
  ZapOff,
} from 'lucide-react'
import { api } from '../api'

export type Page = 'chat' | 'services' | 'dashboard' | 'backlog' | 'memory' | 'graph' | 'skills' | 'tools' | 'settings' | 'autonomy' | 'architecture'

interface NavItem {
  id: Page
  label: string
  icon: React.ComponentType<{ className?: string }>
}

const NAV_ITEMS: NavItem[] = [
  { id: 'chat', label: 'Chat', icon: MessageCircle },
  { id: 'dashboard', label: 'Usage', icon: ChartArea },
  { id: 'backlog', label: 'Backlog', icon: LayoutGrid },
  { id: 'autonomy', label: 'Autonomy', icon: Lightbulb },
  { id: 'memory', label: 'Memory', icon: Brain },
  { id: 'architecture', label: 'Architecture', icon: Code2 },
  { id: 'skills', label: 'Skills', icon: Sparkles },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'services', label: 'Services', icon: LayoutList },
]

const BOTTOM_ITEMS: NavItem[] = [
  { id: 'settings', label: 'Settings', icon: Settings },
]

interface SidebarProps {
  active: Page
  onNavigate: (page: Page) => void
  collapsed: boolean
  onToggleCollapse: () => void
  sessionKey?: string | null
}

export default function Sidebar({ active, onNavigate, collapsed, onToggleCollapse, sessionKey }: SidebarProps) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  void sessionKey
  const CollapseIcon = collapsed ? ChevronsRight : ChevronsLeft
  const [workMode, setWorkMode] = useState(false)
  // Voice state stubs - will be wired up later
  const [isListening, setIsListening] = useState(false)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const _wsAvailable = false
  const _statusLoaded = false
  const _pipelineState: 'LISTENING' | 'ACTIVE_LISTEN' | 'PROCESSING' | 'SPEAKING' = 'IDLE' as never

  // Voice toggle handlers (stub)
  const handleVoiceToggle = () => setVoiceEnabled(!voiceEnabled)
  const startMic = () => setIsListening(true)
  const stopMic = () => setIsListening(false)
  const handleTtsToggle = () => {} // stub

  // Autonomy scheduler state
  const [schedulerEnabled, setSchedulerEnabled] = useState(false)
  const [schedulerRunning, setSchedulerRunning] = useState(false)
  const [schedulerLoaded, setSchedulerLoaded] = useState(false)

  const pollSchedulerStatus = useCallback(() => {
    api.autonomySchedulerStatus()
      .then(s => {
        setSchedulerEnabled(s.enabled)
        setSchedulerRunning(s.running)
        setSchedulerLoaded(true)
      })
      .catch(() => setSchedulerLoaded(false))
  }, [])

  useEffect(() => {
    pollSchedulerStatus()
    const interval = setInterval(pollSchedulerStatus, 15000)
    return () => clearInterval(interval)
  }, [pollSchedulerStatus])

  const handleSchedulerToggle = async () => {
    try {
      if (schedulerEnabled) {
        const r = await api.autonomySchedulerDisable()
        setSchedulerEnabled(r.enabled ?? false)
      } else {
        const r = await api.autonomySchedulerEnable()
        setSchedulerEnabled(r.enabled ?? true)
      }
    } catch (e) {
      console.error('Failed to toggle scheduler:', e)
    }
  }

  const renderItem = (item: NavItem) => {
    const Icon = item.icon
    const isActive = active === item.id
    return (
      <button
        key={item.id}
        onClick={() => onNavigate(item.id)}
        title={collapsed ? item.label : undefined}
        className={`w-full flex items-center ${collapsed ? 'justify-center' : 'gap-2.5'} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
          isActive
            ? 'bg-brand-600/15 text-brand-400'
            : 'text-slate-400 hover:text-slate-200 hover:bg-surface-2'
        }`}
      >
        <Icon className="w-4 h-4 flex-shrink-0" />
        {!collapsed && <span className="truncate">{item.label}</span>}
      </button>
    )
  }

  // Compute state color/text based on pipeline state
  const stateColor = _pipelineState === 'LISTENING' || _pipelineState === 'ACTIVE_LISTEN' ? 'text-green-400 bg-green-600/10' : 'text-slate-400 bg-surface-2'
  const stateText = _pipelineState === 'IDLE' ? 'Idle' : _pipelineState

  return (
    <aside className={`${collapsed ? 'w-14' : 'w-48'} bg-surface-1 border-r border-surface-3/30 flex flex-col py-4 transition-all duration-200`}>
      {/* Brand */}
      <div className={`${collapsed ? 'px-2 justify-center' : 'px-4'} mb-6 flex items-center gap-2`}>
        <img src="/lloyd.jpg" alt="Lloyd" className="w-7 h-7 rounded-lg object-cover flex-shrink-0" />
        {!collapsed && (
          <div>
            <div className="text-sm font-bold tracking-wide text-slate-200">LLOYD</div>
            <div className="text-[10px] text-slate-500 -mt-0.5">Mission Control</div>
          </div>
        )}
      </div>

      {/* Main nav */}
      <nav className="flex-1 px-2 space-y-0.5">
        {NAV_ITEMS.map(renderItem)}
      </nav>

      {/* Bottom nav */}
      <div className="px-2 pt-2 border-t border-surface-3/30 space-y-0.5">
        {/* Voice status section */}
        <div className="space-y-0.5 mb-1">
          {!collapsed && (
            <div className="px-3 pt-1 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500">Voice Mode</div>
          )}
          {/* Row 1 — Last utterance (stub - disabled) */}
          {false && (
            <div
              title={collapsed ? 'No transcript' : undefined}
              className={`w-full flex items-center ${collapsed ? 'justify-center' : 'gap-2.5'} px-3 py-1.5 rounded-lg text-xs transition-opacity duration-1000 opacity-0 text-slate-400`}
            >
              {collapsed ? (
                <MessageCircle className="w-4 h-4 flex-shrink-0 text-slate-500" />
              ) : (
                <span className="truncate text-[11px] italic">""</span>
              )}
            </div>
          )}

          {/* Row 2 — Pipeline state */}
          {_wsAvailable && (
            <div
              title={collapsed ? stateText : undefined}
              className={`w-full flex items-center ${collapsed ? 'justify-center' : 'gap-2.5'} px-3 py-1.5 rounded-lg text-xs font-medium ${stateColor}`}
            >
              <Radio className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="truncate text-[11px] font-mono">{stateText}</span>}
            </div>
          )}

          {/* Row 3 — Mic toggle */}
          <button
            onClick={_wsAvailable && _statusLoaded ? () => (isListening ? stopMic() : startMic()) : undefined}
            disabled={!_wsAvailable || !_statusLoaded}
            title={collapsed ? (isListening ? 'Mic Active' : 'Mic Inactive') : undefined}
            className={`w-full flex items-center ${collapsed ? 'justify-center' : 'gap-2.5'} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              !_wsAvailable || !_statusLoaded
                ? 'text-slate-600 cursor-not-allowed opacity-50'
                : isListening
                  ? 'text-green-400 bg-green-600/10 hover:bg-green-600/20'
                  : 'text-red-400 bg-red-600/10 hover:bg-red-600/20'
            }`}
          >
            {isListening ? <Mic className={`w-4 h-4 flex-shrink-0 ${_pipelineState === 'LISTENING' || _pipelineState === 'ACTIVE_LISTEN' ? 'animate-pulse' : ''}`} /> : <MicOff className="w-4 h-4 flex-shrink-0" />}
            {!collapsed && <span className="truncate">{isListening ? 'Active' : 'Inactive'}</span>}
          </button>

          {/* Row 3.5 — Speaker toggle */}
          <button
            onClick={handleTtsToggle}
            title={collapsed ? (voiceEnabled ? 'Voice On' : 'Voice Off') : undefined}
            className={`w-full flex items-center ${collapsed ? 'justify-center' : 'gap-2.5'} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              voiceEnabled
                ? 'text-green-400 bg-green-600/10 hover:bg-green-600/20'
                : 'text-red-400 bg-red-600/10 hover:bg-red-600/20'
            }`}
          >
            {voiceEnabled ? <Volume2 className="w-4 h-4 flex-shrink-0" /> : <VolumeX className="w-4 h-4 flex-shrink-0" />}
            {!collapsed && <span className="truncate">{voiceEnabled ? 'Voice' : 'Voice'}</span>}
          </button>

          {/* Row 4 — Power toggle */}
          <button
            onClick={_statusLoaded ? handleVoiceToggle : undefined}
            disabled={!_statusLoaded}
            title={collapsed ? (voiceEnabled ? 'Voice Enabled' : 'Voice Disabled') : undefined}
            className={`w-full flex items-center ${collapsed ? 'justify-center' : 'gap-2.5'} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              !_statusLoaded
                ? 'text-slate-600 cursor-not-allowed opacity-50'
                : voiceEnabled
                  ? 'text-green-400 bg-green-600/10 hover:bg-green-600/20'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-surface-2'
            }`}
          >
            <Power className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span className="truncate">{voiceEnabled ? 'Enabled' : 'Disabled'}</span>}
          </button>
        </div>
        <div className="border-t border-surface-3/30 my-1" />

        {/* Autonomy Scheduler toggle */}
        <button
          onClick={schedulerLoaded ? handleSchedulerToggle : undefined}
          disabled={!schedulerLoaded}
          title={collapsed ? (schedulerEnabled ? 'Autonomy: On' : 'Autonomy: Off') : undefined}
          className={`w-full flex items-center ${collapsed ? 'justify-center' : 'gap-2.5'} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
            !schedulerLoaded
              ? 'text-slate-600 cursor-not-allowed opacity-50'
              : schedulerEnabled
                ? 'text-green-400 bg-green-600/10 hover:bg-green-600/20'
                : 'text-red-400 bg-red-600/10 hover:bg-red-600/20'
          }`}
        >
          {schedulerEnabled
            ? <Zap className={`w-4 h-4 flex-shrink-0 ${schedulerRunning ? 'animate-pulse' : ''}`} />
            : <ZapOff className="w-4 h-4 flex-shrink-0" />
          }
          {!collapsed && <span className="truncate">Autonomy</span>}
        </button>

        {/* Work Mode toggle */}
        <button
          onClick={() => setWorkMode(!workMode)}
          title={collapsed ? 'Work Mode' : undefined}
          className={`w-full flex items-center ${collapsed ? 'justify-center' : 'gap-2.5'} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
            workMode
              ? 'bg-purple-600 text-white hover:bg-purple-500'
              : 'text-slate-400 hover:text-slate-200 hover:bg-surface-2'
          }`}
        >
          <Briefcase className="w-4 h-4 flex-shrink-0" />
          {!collapsed && (
            <span className="truncate flex-1 text-left">Work Mode</span>
          )}
          {!collapsed && (
            <span className={`w-4 h-4 rounded border-2 flex-shrink-0 flex items-center justify-center transition-colors ${
              workMode ? 'bg-white border-white' : 'border-slate-500'
            }`}>
              {workMode && <span className="text-[10px] text-purple-600 font-bold leading-none">&#10003;</span>}
            </span>
          )}
        </button>
        <div className="border-t border-surface-3/30 my-1" />
        {BOTTOM_ITEMS.map(renderItem)}
        <div className="border-t border-surface-3/30 my-1" />
        <button
          onClick={onToggleCollapse}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="w-full flex items-center justify-center px-3 py-2 rounded-lg text-xs font-medium text-slate-500 hover:text-slate-300 hover:bg-surface-2 transition-colors mt-1"
        >
          <CollapseIcon className="w-4 h-4" />
        </button>
      </div>
    </aside>
  )
}
