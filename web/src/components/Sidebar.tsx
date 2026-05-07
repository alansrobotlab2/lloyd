import { useState } from 'react'
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
  Volume2,
  Workflow,
  BrainCircuit,
  Waves,
  Power,
  Square,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useVoiceMode } from '../contexts/VoiceModeContext'
import { AgentAudioVisualizerAura } from './agents-ui/agent-audio-visualizer-aura'

export type Page = 'chat' | 'services' | 'backlog' | 'memory' | 'graph' | 'skills' | 'tools' | 'settings' | 'autonomy' | 'architecture' | 'workers' | 'inner_voice' | 'voice_preview'

interface NavItem {
  id: Page
  label: string
  icon: React.ComponentType<{ className?: string }>
}

const NAV_ITEMS: NavItem[] = [
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

// Voice block driven by VoiceModeContext (Phase 6). Replaces the legacy
// pollVoiceStatus loop that polled /api/voice/status every 500ms.
function VoiceModeBlock({ collapsed, sessionKey }: { collapsed: boolean; sessionKey?: string | null }) {
  const voice = useVoiceMode()
  const room = voice.room
  const status = room?.status ?? 'idle'
  const agentState = room?.agentState ?? 'idle'

  // Friendly state label for the pill.
  const stateLabel =
    status === 'connecting' ? 'connecting' :
    status === 'failed' ? 'offline' :
    !voice.enabled ? 'off' :
    agentState

  const stateClass = cn(
    'truncate text-[11px] font-mono',
    voice.enabled
      ? agentState === 'speaking'
        ? 'text-primary'
        : agentState === 'listening'
        ? 'text-emerald-400'
        : agentState === 'thinking' || agentState === 'connecting'
        ? 'text-amber-400'
        : 'text-muted-foreground'
      : 'text-muted-foreground',
  )

  const handleToggleVoice = () => {
    if (voice.enabled) {
      voice.disengage()
    } else if (sessionKey) {
      voice.engage(sessionKey)
    }
  }

  const handleToggleMic = () => {
    if (!room) return
    // The room manages publish state via setMicrophoneEnabled. Manual mute
    // here flips the publication; half-duplex auto-mute may immediately
    // reassert during agent speech.
    const r = room.room
    const enabled = !room.micMuted && room.micPublished
    r.localParticipant.setMicrophoneEnabled(!enabled).catch(() => {})
  }

  // ── Disabled / no session yet ─────────────────────────────────────
  if (!voice.enabled) {
    return (
      <div className="space-y-0.5 mb-1">
        {!collapsed && (
          <div className="px-3 pt-1 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Voice Mode
          </div>
        )}
        <NavRow
          collapsed={collapsed}
          label={sessionKey ? 'Enable voice for this chat' : 'Open a chat session first'}
          onClick={handleToggleVoice}
          disabled={!sessionKey}
          className={cn(
            sessionKey
              ? 'text-muted-foreground hover:text-foreground hover:bg-accent'
              : 'text-muted-foreground/50 cursor-not-allowed opacity-50',
          )}
        >
          <Power className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="truncate">Enable voice</span>}
        </NavRow>
      </div>
    )
  }

  // ── Enabled — render aura thumbnail + status + mic toggle ────────
  return (
    <div className="space-y-1 mb-1">
      {!collapsed && (
        <div className="flex items-center justify-between px-3 pt-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Voice Mode
          </span>
          <button
            onClick={handleToggleVoice}
            title="Disengage voice"
            className="text-[10px] text-muted-foreground hover:text-destructive underline-offset-2 hover:underline"
          >
            disengage
          </button>
        </div>
      )}

      {!collapsed && (
        <div className="flex items-center justify-center px-3 py-1">
          <AgentAudioVisualizerAura
            size="icon"
            state={agentState}
            audioTrack={room?.agentAudioTrack ?? room?.localAudioTrack}
            color="#1FD5F9"
            themeMode="dark"
            className="!h-12"
          />
        </div>
      )}

      {/* State pill */}
      <NavRow
        collapsed={collapsed}
        label={`Voice: ${stateLabel}`}
        className="pointer-events-none"
        showTooltip={false}
      >
        <Volume2 className={cn('w-4 h-4 flex-shrink-0',
          agentState === 'speaking' && 'text-primary animate-pulse',
        )} />
        {!collapsed && <span className={stateClass}>{stateLabel}</span>}
      </NavRow>

      {/* Mic mute toggle */}
      <NavRow
        collapsed={collapsed}
        label={room?.micMuted ? 'Mic muted (auto: half-duplex)' : 'Click to mute mic'}
        onClick={handleToggleMic}
        disabled={!room?.micPublished}
        className={cn(
          !room?.micPublished
            ? 'text-muted-foreground/50 cursor-not-allowed opacity-50'
            : room.micMuted
            ? 'text-amber-400 bg-amber-500/10 hover:bg-amber-500/20'
            : 'text-emerald-400 hover:text-emerald-300 hover:bg-emerald-500/10',
        )}
      >
        {room?.micMuted
          ? <MicOff className="w-4 h-4 flex-shrink-0" />
          : <Mic className="w-4 h-4 flex-shrink-0" />}
        {!collapsed && (
          <span className="truncate">
            {!room?.micPublished ? 'Mic offline' : room.micMuted ? 'Mic muted' : 'Mic on'}
          </span>
        )}
      </NavRow>

      {/* Interrupt — only visible while agent is actively speaking */}
      {room?.agentSpeaking && (
        <NavRow
          collapsed={collapsed}
          label="Interrupt — stop Lloyd's reply"
          onClick={() => { room.interrupt().catch(() => {}) }}
          className="text-destructive bg-destructive/10 hover:bg-destructive/20"
        >
          <Square className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="truncate">Interrupt</span>}
        </NavRow>
      )}

      {/* Error surface (mic permission denied, secure-context, etc.) */}
      {!collapsed && room?.error && (
        <div className="px-3 py-1 text-[10px] text-destructive break-words">
          {room.error}
        </div>
      )}
    </div>
  )
}

export default function Sidebar({ active, onNavigate, collapsed, onToggleCollapse, sessionKey, isMobile }: SidebarProps) {
  const CollapseIcon = collapsed ? ChevronsRight : ChevronsLeft
  const [workMode, setWorkMode] = useState(false)

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
          {/* Voice block — driven by VoiceModeContext (LiveKit room state) */}
          <VoiceModeBlock collapsed={collapsed} sessionKey={sessionKey} />

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
