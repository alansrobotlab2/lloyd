import { useCallback, useEffect, useRef, useState } from 'react'
import { Plus, Info, Loader2, Square, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { type AgentState, useTrackVolume } from '@livekit/components-react'
import { type LocalAudioTrack, type RemoteAudioTrack } from 'livekit-client'
import ChatPanel from './ChatPanel'
import { AgentAudioVisualizerAura } from './agents-ui/agent-audio-visualizer-aura'
import { WakeStatePill } from './agents-ui/wake-state-pill'
import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { api } from '../api'
import { useVoiceMode } from '../contexts/VoiceModeContext'

// Aura colorShift drives how much hue varies across the shader's iteration
// layers — the higher the value, the more rainbow-like and chromatic it
// looks. We modulate it from a calm floor when Lloyd is silent up to a
// vivid peak at full TTS volume, so the aura visibly "fans out" with the
// agent's voice intensity.
const AURA_COLORSHIFT_MIN = 0.25
const AURA_COLORSHIFT_MAX = 1.25
// Volume threshold for "agent is currently producing audio". Below this we
// treat the TTS track as silent. Matches the half-duplex threshold inside
// VoiceRoom so the speaking/listening flip lines up with mic muting.
const AGENT_SPEAKING_THRESHOLD = 0.02

const STORAGE_KEY = 'mc.rightChatSidebar.width'
const COLLAPSED_KEY = 'mc.rightChatSidebar.collapsed'
const MIN_WIDTH = 320
const MAX_WIDTH = 720
const DEFAULT_WIDTH = 448  // 28rem
const COLLAPSED_WIDTH = 56 // matches the left nav's collapsed width (w-14)

/** Compact dBFS meter for the local mic — mirrors the readout on the
 *  Voice tab so the sidebar shows mic activity at a glance. Reads off the
 *  local track only; agent TTS doesn't drive this bar. */
function MicDbMeter({
  audioTrack,
}: {
  audioTrack: LocalAudioTrack | RemoteAudioTrack | undefined
}) {
  const volume = useTrackVolume(audioTrack as LocalAudioTrack | undefined, {
    fftSize: 512,
    smoothingTimeConstant: 0.55,
  })
  const dbfs = volume > 0.0001 ? 20 * Math.log10(volume) : -120
  const dbDisplay = volume > 0 ? `${dbfs.toFixed(1)} dB` : '—'
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>mic</span>
        <span className="tabular-nums text-foreground">{dbDisplay}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
        <div
          className="h-full bg-primary transition-[width] duration-100"
          style={{ width: `${Math.min(100, volume * 200)}%` }}
        />
      </div>
    </div>
  )
}

/** Tiny mic dBFS bar for collapsed mode — same data as MicDbMeter, but
 *  rendered as a thin horizontal bar that fits inside the slender sidebar
 *  with no numeric readout. */
function MicDbMini({
  audioTrack,
}: {
  audioTrack: LocalAudioTrack | RemoteAudioTrack | undefined
}) {
  const volume = useTrackVolume(audioTrack as LocalAudioTrack | undefined, {
    fftSize: 512,
    smoothingTimeConstant: 0.55,
  })
  return (
    <div
      className="h-1 w-8 rounded-full bg-secondary overflow-hidden"
      title={volume > 0.0001 ? `${(20 * Math.log10(volume)).toFixed(1)} dB` : 'mic silent'}
    >
      <div
        className="h-full bg-primary transition-[width] duration-100"
        style={{ width: `${Math.min(100, volume * 200)}%` }}
      />
    </div>
  )
}

/** Single colored dot mirroring the wake-state pill's status colors —
 *  used in collapsed mode where the full pill won't fit. */
function StatusDot({
  status,
  agentSpeaking,
  agentThinking,
  wakeState,
}: {
  status: 'idle' | 'connecting' | 'connected' | 'failed'
  agentSpeaking: boolean
  agentThinking: boolean
  wakeState: 'idle' | 'listening'
}) {
  const cls =
    status !== 'connected' ? 'bg-muted-foreground' :
    agentThinking ? 'bg-amber-400 animate-pulse' :
    agentSpeaking ? 'bg-primary animate-pulse' :
    wakeState === 'listening' ? 'bg-emerald-400 animate-pulse' :
    'bg-muted-foreground/60'
  const label =
    status !== 'connected' ? 'Not connected' :
    agentThinking ? 'Thinking…' :
    agentSpeaking ? 'Speaking' :
    wakeState === 'listening' ? 'Listening' :
    "Say 'Lloyd'"
  return <span className={cn('inline-block h-2 w-2 rounded-full', cls)} title={label} />
}

function loadStoredWidth(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_WIDTH
    const n = parseInt(raw, 10)
    if (!Number.isFinite(n)) return DEFAULT_WIDTH
    return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, n))
  } catch {
    return DEFAULT_WIDTH
  }
}

function loadCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === '1'
  } catch {
    return false
  }
}


/**
 * Right-side voice-driven chat panel — always mounted on desktop.
 *
 * Renders ChatPanel against a dedicated mission-control session that has
 * Inner Voice (the dual-brain observer) enabled. On mount, picks the
 * most-recent IV-enabled session or creates one. The "+" button always
 * starts a fresh IV session.
 *
 * Voice engagement: this sidebar's session is what the LiveKit room is
 * bound to — voice auto-engages on mount, stays engaged for the lifetime
 * of the layout. The left-sidebar voice toggle is suppressed on desktop
 * (Layout's slot-tracking effect skips when the sidebar is mounted).
 */
export default function RightChatSidebar({ isMobile = false }: { isMobile?: boolean } = {}) {
  const voice = useVoiceMode()
  const [sessionKey, setSessionKey] = useState<string | null>(null)
  const [resolving, setResolving] = useState(false)
  const [agentDetails, setAgentDetails] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [width, setWidth] = useState<number>(loadStoredWidth)
  // On mobile the sidebar IS the chat UI — never collapsed, full width.
  const [collapsed, setCollapsed] = useState<boolean>(() => isMobile ? false : loadCollapsed())
  const dragStateRef = useRef<{ startX: number; startWidth: number } | null>(null)
  // ChatPanel publishes its "harness is thinking / streaming" state via
  // onThinkingChange. We mirror it here so the aura can flip to 'thinking'
  // mode (different animation than listening) while Lloyd is processing
  // a turn — i.e. before TTS starts but after the user finished speaking.
  const [chatThinking, setChatThinking] = useState(false)

  // Resolve a session on mount: most-recent inner-voice mc session if one
  // exists, otherwise create a fresh one. New session button below sets
  // force=true to skip the lookup.
  const resolveSession = useCallback(async (force = false) => {
    setResolving(true)
    setError(null)
    try {
      if (!force) {
        const list = await api.listSessions()
        const sessions = (list.sessions ?? []) as Array<{
          session_key: string
          platform?: string
          inner_voice?: boolean
        }>
        const recent = sessions.find(
          s => s.inner_voice && (s.platform ?? 'mission-control') === 'mission-control',
        )
        if (recent) {
          setSessionKey(recent.session_key)
          return
        }
      }
      const created = await api.createSession({
        platform: 'mission-control',
        inner_voice: true,
        inner_voice_evaluate_user_turns: true,
      })
      setSessionKey(created.session_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setResolving(false)
    }
  }, [])

  // Resolve once on mount. The new-session button calls resolveSession(true)
  // directly to skip the listing lookup.
  useEffect(() => {
    if (!sessionKey && !resolving) {
      void resolveSession(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Engage voice once we have a session AND the sidebar is expanded.
  // Collapsing the sidebar disengages voice (mic stops publishing, agent
  // track tears down) — re-expanding re-engages cleanly. Disengages on
  // unmount as well (Layout teardown).
  useEffect(() => {
    if (!sessionKey || collapsed) return
    voice.engage(sessionKey)
    return () => voice.disengage()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionKey, collapsed])

  const toggleCollapsed = useCallback(() => {
    setCollapsed(c => {
      const next = !c
      try { localStorage.setItem(COLLAPSED_KEY, next ? '1' : '0') } catch { /* ignore */ }
      return next
    })
  }, [])

  const handleNewSession = async () => {
    setSessionKey(null)
    await resolveSession(true)
  }

  // Stop: cancel the running turn AND drain queued ambient turns. The
  // backend cancel_event is shared with the IV observer (observer.py),
  // so this stops both primary and IV at the same time. Works even when
  // ChatPanel doesn't think it's busy — between IV-driven iterations the
  // harness may briefly settle to streaming=false, and the inline Stop
  // in ChatPanel disappears with it.
  const handleStop = useCallback(() => {
    if (!sessionKey) return
    api.cancelSession(sessionKey, { drainPending: true }).catch(() => { /* swallow */ })
  }, [sessionKey])

  // Resize: drag the left edge of the sidebar. mousemove + mouseup are
  // bound to window so the drag continues even if the cursor leaves the
  // narrow handle. Width is clamped + persisted on release.
  const onResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragStateRef.current = { startX: e.clientX, startWidth: width }
    const onMove = (ev: MouseEvent) => {
      const s = dragStateRef.current
      if (!s) return
      // Sidebar is on the right; dragging left grows the panel.
      const next = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, s.startWidth + (s.startX - ev.clientX)))
      setWidth(next)
    }
    const onUp = () => {
      const s = dragStateRef.current
      if (s) {
        try {
          // Read current width via the latest setState callback so we save
          // the actual value, not the captured-at-mousedown initial.
          setWidth(curr => {
            try { localStorage.setItem(STORAGE_KEY, String(curr)) } catch { /* ignore */ }
            return curr
          })
        } catch { /* ignore */ }
      }
      dragStateRef.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [width])

  const room = voice.room
  const status = room?.status ?? 'idle'
  const agentTrack = room?.agentAudioTrack
  const localTrack = room?.localAudioTrack

  // Aura follows whoever is the current audio source: agent track if
  // present (so the aura modulates with Lloyd's voice), else local mic.
  const visualizerTrack = agentTrack ?? localTrack

  // Track agent TTS volume (0..1) ourselves so we own both colorShift
  // modulation and the speaking/listening state flip without depending on
  // VoiceRoom's separately-derived agentState propagating through the
  // context bridge. Smoothing = 0.4 matches the half-duplex tracker.
  const agentVolume = useTrackVolume(agentTrack, {
    fftSize: 512,
    smoothingTimeConstant: 0.4,
  })
  const isSpeaking = agentVolume > AGENT_SPEAKING_THRESHOLD

  // Aura state machine, computed locally:
  //   - 'idle' when the room isn't connected
  //   - 'speaking' when TTS track has volume above threshold (overrides
  //     thinking even if both are true — TTS implies thinking is over)
  //   - 'thinking' while ChatPanel reports the harness is mid-turn
  //   - 'listening' otherwise (connected + waiting)
  const auraState: AgentState =
    status !== 'connected' ? 'idle' :
    isSpeaking ? 'speaking' :
    chatThinking ? 'thinking' :
    'listening'

  // Linear map [0..1] → [MIN..MAX]. Volume from useTrackVolume rarely
  // exceeds ~0.6 even on loud TTS, so we apply a gentle gain (×1.4) and
  // clamp to keep the dynamic range visible without saturating early.
  const dynamicColorShift =
    AURA_COLORSHIFT_MIN +
    Math.min(1, agentVolume * 1.4) * (AURA_COLORSHIFT_MAX - AURA_COLORSHIFT_MIN)

  // ── Collapsed layout ──────────────────────────────────────────────
  // Slender bar matching the left nav's collapsed width. Mini aura on
  // top, status dot, vertical icon buttons mirroring the right column,
  // dB indicator, and an expand button at the bottom. While collapsed
  // voice is fully disengaged (see useEffect above), so the aura's
  // visualizerTrack is undefined and it just shows the idle animation —
  // that's the desired "voice is off" cue.
  if (collapsed && !isMobile) {
    // Clicking anywhere on the bar expands it, except on the inner buttons —
    // those keep their own actions (new session, stop, details, expand).
    const handleBarClick = (e: React.MouseEvent) => {
      if ((e.target as HTMLElement).closest('button')) return
      toggleCollapsed()
    }
    return (
      <TooltipProvider delayDuration={150}>
        <aside
          style={{ width: `${COLLAPSED_WIDTH}px` }}
          onClick={handleBarClick}
          title="Expand chat — re-engages voice"
          className="shrink-0 h-full flex flex-col items-center gap-1.5 py-2 bg-card border-l border-border cursor-pointer"
        >
          <AgentAudioVisualizerAura
            size="icon"
            state={auraState}
            audioTrack={visualizerTrack}
            color="#A78BFA"
            colorShift={dynamicColorShift}
            themeMode="dark"
            className="!h-10 !w-10"
          />
          <StatusDot
            status={status}
            agentSpeaking={isSpeaking}
            agentThinking={chatThinking}
            wakeState={room?.wakeState ?? 'idle'}
          />
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={handleNewSession}
                disabled={resolving}
                className="h-8 w-8 shrink-0"
              >
                {resolving
                  ? <Loader2 className="w-4 h-4 animate-spin" />
                  : <Plus className="w-4 h-4" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">New chat session</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={chatThinking ? 'destructive' : 'outline'}
                size="icon"
                onClick={handleStop}
                disabled={!sessionKey}
                className="h-8 w-8 shrink-0"
              >
                <Square className="w-4 h-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">Stop turn (primary + IV)</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={agentDetails ? 'secondary' : 'outline'}
                size="icon"
                onClick={() => setAgentDetails(d => !d)}
                className="h-8 w-8 shrink-0"
              >
                <Info className="w-4 h-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">Toggle agent details</TooltipContent>
          </Tooltip>
          <MicDbMini audioTrack={localTrack} />
          <div className="flex-1" />
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleCollapsed}
                className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
              >
                <ChevronsLeft className="w-4 h-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">Expand chat — re-engages voice</TooltipContent>
          </Tooltip>
        </aside>
      </TooltipProvider>
    )
  }

  // ── Expanded layout ───────────────────────────────────────────────
  return (
    <aside
      style={isMobile ? undefined : { width: `${width}px` }}
      className={cn(
        'h-full flex flex-col bg-card relative',
        isMobile ? 'w-full flex-1 min-h-0' : 'shrink-0 border-l border-border',
      )}
    >
      {/* Resize handle + collapse chevron — desktop only. On mobile the
          sidebar IS the chat UI, so resizing/collapsing don't apply. */}
      {!isMobile && (
        <>
          <div
            onMouseDown={onResizeStart}
            className="absolute left-0 top-0 bottom-0 w-1.5 -ml-0.5 z-10 cursor-col-resize group"
            aria-label="Resize sidebar"
            role="separator"
          >
            <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-px bg-border group-hover:bg-primary/60 transition-colors" />
          </div>
          <button
            onClick={toggleCollapsed}
            title="Collapse chat — disengages voice"
            className="absolute top-1.5 left-1.5 z-20 h-6 w-6 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
            aria-label="Collapse chat sidebar"
          >
            <ChevronsRight className="w-4 h-4" />
          </button>
        </>
      )}

      {/* Header: aura centered (not stretched) on the left, stacked
          controls on the right. Right column owns its own height; aura
          is a fixed square that centers within whatever space is left. */}
      <div className="flex flex-col gap-2 p-3 border-b border-border">
        <div className="flex items-start gap-3">
          <div className="flex-1 min-w-0 flex items-center justify-center overflow-hidden">
            <AgentAudioVisualizerAura
              size="lg"
              state={auraState}
              audioTrack={visualizerTrack}
              color="#A78BFA"
              colorShift={dynamicColorShift}
              themeMode="dark"
              className="!h-[180px] !w-[180px] shrink-0"
            />
          </div>
          <div className="w-28 shrink-0 flex flex-col gap-1.5">
            <WakeStatePill
              compact
              status={status}
              wakeState={room?.wakeState ?? 'idle'}
              wakeRemainingS={room?.wakeRemainingS ?? 0}
              wakeContinuationS={room?.wakeContinuationS ?? 6}
              wakeSpeaker={room?.wakeSpeaker ?? null}
              agentSpeaking={isSpeaking}
              agentThinking={chatThinking}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={handleNewSession}
              disabled={resolving}
              className="h-7 text-xs w-full justify-start"
              title="Start a new dual-brain session"
            >
              {resolving
                ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                : <Plus className="w-3.5 h-3.5 mr-1.5" />}
              New
            </Button>
            <Button
              variant={chatThinking ? 'destructive' : 'outline'}
              size="sm"
              onClick={handleStop}
              disabled={!sessionKey}
              className="h-7 text-xs w-full justify-start"
              title="Stop the current turn (primary + Inner Voice) and drain queued ambient turns"
            >
              <Square className="w-3.5 h-3.5 mr-1.5" />
              Stop
            </Button>
            <Button
              variant={agentDetails ? 'secondary' : 'outline'}
              size="sm"
              onClick={() => setAgentDetails(d => !d)}
              className="h-7 text-xs w-full justify-start"
              title="Show / hide agent details (tool calls, thinking)"
            >
              <Info className="w-3.5 h-3.5 mr-1.5" />
              Details
            </Button>
            <MicDbMeter audioTrack={localTrack} />
          </div>
        </div>
        {error && (
          <div className="text-xs text-destructive break-words">{error}</div>
        )}
      </div>

      {/* Body: ChatPanel against the resolved session. Gets remounted when
          sessionKey changes, so a new-session click cleanly re-loads. */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {sessionKey ? (
          <ChatPanel
            key={sessionKey}
            requestedSessionKey={sessionKey}
            currentSessionKey={sessionKey}
            visible={true}
            showAgentDetails={agentDetails}
            compact
            isMobile={isMobile}
            onThinkingChange={(thinking) => setChatThinking(thinking)}
          />
        ) : (
          <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
            {resolving ? 'Resolving session…' : 'No session.'}
          </div>
        )}
      </div>
    </aside>
  )
}
