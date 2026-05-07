import { useCallback, useEffect, useRef, useState } from 'react'
import { Plus, Info, Loader2, Square, Brain } from 'lucide-react'
import { type AgentState, useTrackVolume } from '@livekit/components-react'
import ChatPanel from './ChatPanel'
import { AgentAudioVisualizerAura } from './agents-ui/agent-audio-visualizer-aura'
import { WakeStatePill } from './agents-ui/wake-state-pill'
import { Button } from '@/components/ui/button'
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
const MIN_WIDTH = 320
const MAX_WIDTH = 720
const DEFAULT_WIDTH = 448  // 28rem

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
export default function RightChatSidebar() {
  const voice = useVoiceMode()
  const [sessionKey, setSessionKey] = useState<string | null>(null)
  const [resolving, setResolving] = useState(false)
  const [agentDetails, setAgentDetails] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [width, setWidth] = useState<number>(loadStoredWidth)
  const dragStateRef = useRef<{ startX: number; startWidth: number } | null>(null)
  // ChatPanel publishes its "harness is thinking / streaming" state via
  // onThinkingChange. We mirror it here so the aura can flip to 'thinking'
  // mode (different animation than listening) while Lloyd is processing
  // a turn — i.e. before TTS starts but after the user finished speaking.
  const [chatThinking, setChatThinking] = useState(false)
  // IV is on when both flags are true. Tracked here so the toolbar
  // toggle reflects current state across reloads. We refresh it on
  // session change and after our own patches.
  const [ivEnabled, setIvEnabled] = useState<boolean>(true)
  const [ivBusy, setIvBusy] = useState<boolean>(false)

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

  // Engage voice once we have a session. Stays engaged for the lifetime of
  // the sidebar. Disengages on unmount (only happens on Layout teardown).
  useEffect(() => {
    if (sessionKey) {
      voice.engage(sessionKey)
      return () => voice.disengage()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionKey])

  const handleNewSession = async () => {
    setSessionKey(null)
    await resolveSession(true)
  }

  // Read IV state for the resolved session so the toggle reflects what's
  // actually enabled (across reloads, after patches from elsewhere).
  useEffect(() => {
    if (!sessionKey) return
    let cancelled = false
    api.innerVoiceState(sessionKey)
      .then(s => { if (!cancelled) setIvEnabled(!!s.inner_voice_enabled) })
      .catch(() => { /* leave default */ })
    return () => { cancelled = true }
  }, [sessionKey])

  // Stop: cancel the running turn AND drain queued ambient turns. Works
  // even when ChatPanel doesn't think it's busy — between IV-driven
  // iterations the harness may briefly settle to streaming=false, and
  // the inline Stop in ChatPanel disappears with it.
  const handleStop = useCallback(() => {
    if (!sessionKey) return
    api.cancelSession(sessionKey, { drainPending: true }).catch(() => { /* swallow */ })
  }, [sessionKey])

  // Pause / resume IV for this session. Toggling both flags off prevents
  // observation entirely; toggling on restores the dual-brain default.
  const handleToggleIv = useCallback(async () => {
    if (!sessionKey || ivBusy) return
    const next = !ivEnabled
    setIvBusy(true)
    try {
      const r = await api.patchSession(sessionKey, {
        inner_voice: next,
        inner_voice_evaluate_user_turns: next,
      })
      setIvEnabled(!!r.inner_voice)
    } catch {
      /* leave state */
    } finally {
      setIvBusy(false)
    }
  }, [sessionKey, ivBusy, ivEnabled])

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

  return (
    <aside
      style={{ width: `${width}px` }}
      className={cn(
        'shrink-0 h-full flex flex-col bg-card border-l border-border relative',
      )}
    >
      {/* Resize handle on the left edge. 6px wide hot zone with a subtle
          line that highlights on hover. */}
      <div
        onMouseDown={onResizeStart}
        className="absolute left-0 top-0 bottom-0 w-1.5 -ml-0.5 z-10 cursor-col-resize group"
        aria-label="Resize sidebar"
        role="separator"
      >
        <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-px bg-border group-hover:bg-primary/60 transition-colors" />
      </div>

      {/* Header: pill on top, aura below for breathing room, actions at the
          bottom. Aura is centered + sized to fill comfortably without
          dominating the sidebar's vertical real estate. */}
      <div className="flex flex-col gap-3 p-3 border-b border-border">
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
        <div className="flex items-center justify-center py-2">
          <AgentAudioVisualizerAura
            size="lg"
            state={auraState}
            audioTrack={visualizerTrack}
            color="#A78BFA"
            colorShift={dynamicColorShift}
            themeMode="dark"
            // 25% smaller than the size="lg" default (224 → 168px). The
            // size variants don't have an in-between, so we override
            // height + width directly with !important utility classes.
            className="!h-[168px] !w-[168px]"
          />
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={handleNewSession}
            disabled={resolving}
            className="h-7 text-xs"
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
            className="h-7 text-xs"
            title="Stop the current turn and drain queued ambient turns"
          >
            <Square className="w-3.5 h-3.5 mr-1.5" />
            Stop
          </Button>
          <Button
            variant={ivEnabled ? 'secondary' : 'outline'}
            size="sm"
            onClick={handleToggleIv}
            disabled={!sessionKey || ivBusy}
            className={cn(
              'h-7 text-xs',
              ivEnabled
                ? 'bg-purple-600/20 border-purple-500/30 text-purple-400 hover:bg-purple-600/30 hover:text-purple-300'
                : '',
            )}
            title={ivEnabled ? 'Pause Inner Voice for this session' : 'Resume Inner Voice for this session'}
          >
            {ivBusy
              ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              : <Brain className="w-3.5 h-3.5 mr-1.5" />}
            {ivEnabled ? 'IV on' : 'IV off'}
          </Button>
          <Button
            variant={agentDetails ? 'secondary' : 'outline'}
            size="sm"
            onClick={() => setAgentDetails(d => !d)}
            className="h-7 text-xs"
            title="Show / hide agent details (tool calls, thinking)"
          >
            <Info className="w-3.5 h-3.5 mr-1.5" />
            Details
          </Button>
          <span className="ml-auto text-[10px] text-muted-foreground font-mono truncate max-w-[10rem]" title={sessionKey ?? ''}>
            {sessionKey ?? (resolving ? 'creating…' : '—')}
          </span>
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
