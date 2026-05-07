import { useState, useEffect, useCallback } from 'react'
import {
  LayoutList,
  Brain,
  Sparkles,
  Wrench,
  LayoutGrid,
  Settings,
  MessageCircle,
  ChevronsLeft,
  ChevronsRight,
  Briefcase,
  Code2,
  Lightbulb,
  Mic,
  MicOff,
  Radio,
  Volume2,
  VolumeX,
  Workflow,
  BrainCircuit,
  Waves,
} from 'lucide-react'
import { api } from '../api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

export type Page = 'chat' | 'services' | 'backlog' | 'memory' | 'graph' | 'skills' | 'tools' | 'settings' | 'autonomy' | 'architecture' | 'workers' | 'inner_voice' | 'voice_preview'

interface NavItem {
  id: Page
  label: string
  icon: React.ComponentType<{ className?: string }>
}

const NAV_ITEMS: NavItem[] = [
  // Inner Voice — sibling to Chat. The critic ensemble runs only on
  // sessions opened in this tab; existing Chat tab behavior is unchanged.
  { id: 'inner_voice', label: 'Inner Voice', icon: BrainCircuit },
  { id: 'chat', label: 'Chat', icon: MessageCircle },
  { id: 'backlog', label: 'Backlog', icon: LayoutGrid },
  { id: 'autonomy', label: 'Autonomy', icon: Lightbulb },
  { id: 'workers', label: 'Workers', icon: Workflow },
  { id: 'memory', label: 'Memory', icon: Brain },
  { id: 'architecture', label: 'Architecture', icon: Code2 },
  { id: 'skills', label: 'Skills', icon: Sparkles },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'services', label: 'Services', icon: LayoutList },
  { id: 'voice_preview', label: 'Voice (preview)', icon: Waves },
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
  isMobile?: boolean
}

// Icon-button row with an optional tooltip that activates only when collapsed.
// Mirrors the old `title=` UX without polluting the markup.
function NavRow({
  collapsed,
  label,
  showTooltip,
  className,
  onClick,
  disabled,
  children,
}: {
  collapsed: boolean
  label: string
  showTooltip?: boolean
  className?: string
  onClick?: () => void
  disabled?: boolean
  children: React.ReactNode
}) {
  const button = (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'w-full flex items-center rounded-md text-xs font-medium transition-colors',
        collapsed ? 'justify-center px-3 py-2' : 'gap-2.5 px-3 py-2',
        className,
      )}
    >
      {children}
    </button>
  )
  if (collapsed && showTooltip !== false) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{button}</TooltipTrigger>
        <TooltipContent side="right">{label}</TooltipContent>
      </Tooltip>
    )
  }
  return button
}

export default function Sidebar({ active, onNavigate, collapsed, onToggleCollapse, sessionKey, isMobile }: SidebarProps) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  void sessionKey
  const CollapseIcon = collapsed ? ChevronsRight : ChevronsLeft
  const [workMode, setWorkMode] = useState(false)
  // Voice state — polls voice mode HTTP API via backend proxy.
  // Phase 6 will replace this with LiveKit room state.
  const [isListening, setIsListening] = useState(false)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [voiceOnline, setVoiceOnline] = useState(false)
  const [pipelineState, setPipelineState] = useState<string>('IDLE')
  const [ttsEnabled, setTtsEnabled] = useState(false)

  const pollVoiceStatus = useCallback(() => {
    api.voiceStatus()
      .then(s => {
        const online = s.state !== 'OFFLINE'
        setVoiceOnline(online)
        setVoiceEnabled(s.voice_enabled ?? false)
        setPipelineState(s.state ?? 'IDLE')
        setIsListening(s.state === 'LISTENING' || s.state === 'ACTIVE_LISTEN')
      })
      .catch(() => setVoiceOnline(false))
    api.voiceTtsStatus()
      .then(s => setTtsEnabled(s.tts_enabled ?? false))
      .catch(() => {})
  }, [])

  useEffect(() => {
    pollVoiceStatus()
    const interval = setInterval(pollVoiceStatus, 500)
    return () => clearInterval(interval)
  }, [pollVoiceStatus])

  const handleVoiceToggle = async () => {
    try {
      const r = await api.voiceToggle()
      setVoiceEnabled(r.voice_enabled ?? !voiceEnabled)
    } catch {
      // ignore
    }
  }
  const handleTtsToggle = async () => {
    try {
      const r = await api.voiceTtsToggle()
      setTtsEnabled(r.tts_enabled ?? !ttsEnabled)
    } catch {
      // ignore
    }
  }

  const renderItem = (item: NavItem) => {
    const Icon = item.icon
    const isActive = active === item.id
    return (
      <NavRow
        key={item.id}
        collapsed={collapsed}
        label={item.label}
        onClick={() => onNavigate(item.id)}
        className={
          isActive
            ? 'bg-primary/15 text-primary hover:bg-primary/20'
            : 'text-muted-foreground hover:text-foreground hover:bg-accent'
        }
      >
        <Icon className="w-4 h-4 flex-shrink-0" />
        {!collapsed && <span className="truncate">{item.label}</span>}
      </NavRow>
    )
  }

  const stateText = pipelineState === 'IDLE' ? 'Idle' : pipelineState
  const stateClass =
    pipelineState === 'LISTENING' || pipelineState === 'ACTIVE_LISTEN'
      ? 'text-emerald-400 bg-emerald-500/10'
      : 'text-muted-foreground bg-secondary'

  return (
    <TooltipProvider delayDuration={150}>
      <aside
        className={cn(
          'flex flex-col transition-all duration-200',
          isMobile ? 'w-full flex-1 py-2' : 'py-4 bg-card border-r border-border',
          isMobile ? '' : collapsed ? 'w-14' : 'w-48',
        )}
      >
        {/* Brand */}
        {!isMobile && (
          <div className={cn('mb-6 flex items-center gap-2', collapsed ? 'px-2 justify-center' : 'px-4')}>
            <img src="/lloyd.jpg" alt="Lloyd" className="w-7 h-7 rounded-lg object-cover flex-shrink-0" />
            {!collapsed && (
              <div>
                <div className="text-sm font-bold tracking-wide text-foreground">LLOYD</div>
                <div className="text-[10px] text-muted-foreground -mt-0.5">Mission Control</div>
              </div>
            )}
          </div>
        )}

        {/* Main nav */}
        <nav className="flex-1 px-2 space-y-0.5">
          {NAV_ITEMS.map(renderItem)}
        </nav>

        {/* Bottom nav */}
        <div className="px-2 pt-2 space-y-0.5 border-t border-border">
          {/* Voice status section */}
          <div className="space-y-0.5 mb-1">
            {!collapsed && (
              <div className="px-3 pt-1 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Voice Mode
              </div>
            )}

            {/* Pipeline state */}
            {voiceOnline && (
              <NavRow
                collapsed={collapsed}
                label={stateText}
                className={cn('text-xs font-medium pointer-events-none', stateClass)}
              >
                <Radio className="w-4 h-4 flex-shrink-0" />
                {!collapsed && <span className="truncate text-[11px] font-mono">{stateText}</span>}
              </NavRow>
            )}

            {/* Mic toggle */}
            <NavRow
              collapsed={collapsed}
              disabled={!voiceOnline}
              label={
                !voiceOnline ? 'Voice offline' :
                isListening ? 'Listening — click to disable' :
                voiceEnabled ? 'Mic on — click to disable' :
                'Mic off — click to enable'
              }
              onClick={voiceOnline ? handleVoiceToggle : undefined}
              className={cn(
                !voiceOnline
                  ? 'text-muted-foreground/50 cursor-not-allowed opacity-50'
                  : voiceEnabled
                    ? isListening
                      ? 'text-emerald-400 bg-emerald-500/20 hover:bg-emerald-500/30'
                      : 'text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent',
              )}
            >
              {voiceEnabled ? (
                <Mic className={cn('w-4 h-4 flex-shrink-0', isListening && 'animate-pulse')} />
              ) : (
                <MicOff className="w-4 h-4 flex-shrink-0" />
              )}
              {!collapsed && (
                <span className="truncate">
                  {!voiceOnline ? 'Offline' : isListening ? 'Listening' : voiceEnabled ? 'Mic On' : 'Mic Off'}
                </span>
              )}
            </NavRow>

            {/* TTS toggle */}
            <NavRow
              collapsed={collapsed}
              label={ttsEnabled ? 'Speak responses: ON' : 'Speak responses: OFF'}
              onClick={handleTtsToggle}
              className={
                ttsEnabled
                  ? 'text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent'
              }
            >
              {ttsEnabled ? <Volume2 className="w-4 h-4 flex-shrink-0" /> : <VolumeX className="w-4 h-4 flex-shrink-0" />}
              {!collapsed && <span className="truncate">{ttsEnabled ? 'Speak: On' : 'Speak: Off'}</span>}
            </NavRow>
          </div>

          <Separator className="my-1" />

          {/* Work Mode toggle */}
          <NavRow
            collapsed={collapsed}
            label="Work Mode"
            onClick={() => setWorkMode(!workMode)}
            className={cn(
              workMode
                ? 'bg-purple-600 text-white hover:bg-purple-500'
                : 'text-muted-foreground hover:text-foreground hover:bg-accent',
            )}
          >
            <Briefcase className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span className="truncate flex-1 text-left">Work Mode</span>}
            {!collapsed && (
              <span
                className={cn(
                  'w-4 h-4 rounded border-2 flex-shrink-0 flex items-center justify-center transition-colors',
                  workMode ? 'bg-white border-white' : 'border-muted-foreground',
                )}
              >
                {workMode && <span className="text-[10px] text-purple-600 font-bold leading-none">&#10003;</span>}
              </span>
            )}
          </NavRow>

          <Separator className="my-1" />

          {BOTTOM_ITEMS.map(renderItem)}

          {!isMobile && (
            <>
              <Separator className="my-1" />
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onToggleCollapse}
                    className="w-full justify-center text-muted-foreground hover:text-foreground"
                  >
                    <CollapseIcon className="w-4 h-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right">{collapsed ? 'Expand sidebar' : 'Collapse sidebar'}</TooltipContent>
              </Tooltip>
            </>
          )}
        </div>
      </aside>
    </TooltipProvider>
  )
}
