import { useEffect, useRef, useState } from 'react'
import { type AgentState, useTrackVolume } from '@livekit/components-react'
import type { LocalAudioTrack } from 'livekit-client'
import { AgentAudioVisualizerAura } from '../agents-ui/agent-audio-visualizer-aura'
import VoiceRoom from '../VoiceRoom'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { api, type MessageEntry } from '../../api'

const STATES: AgentState[] = ['idle', 'listening', 'thinking', 'speaking']
const COLORS: Array<{ name: string; hex: `#${string}` }> = [
  { name: 'LiveKit cyan', hex: '#1FD5F9' },
  { name: 'Lloyd indigo', hex: '#818CF8' },
  { name: 'Mint',         hex: '#34D399' },
  { name: 'Amber',        hex: '#F59E0B' },
  { name: 'Magenta',      hex: '#E879F9' },
]

const VOICE_PREVIEW_SESSION = 'voice_preview'

/** Polling transcript log for the voice preview. Reads /api/messages/<id>
 *  every 500ms and renders the most recent user + assistant turns. Tool-
 *  call rounds (assistant text="" + a tool result) are collapsed into a
 *  single "tool calls (N)" line so they don't bury real transcripts. */
function TranscriptLog({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<MessageEntry[]>([])
  const [polledOnce, setPolledOnce] = useState(false)
  const [resetting, setResetting] = useState(false)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.loadMessages(sessionId, 200)
        if (!cancelled && r.success && Array.isArray(r.messages)) {
          setMessages(r.messages)
        }
      } catch {
        // session may not exist yet — that's fine, the worker creates it
        // on the first inject.
      } finally {
        if (!cancelled) setPolledOnce(true)
      }
    }
    poll()
    const id = setInterval(poll, 500)
    return () => { cancelled = true; clearInterval(id) }
  }, [sessionId])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages.length])

  const text = (m: MessageEntry) =>
    m.content?.map(c => (c.type === 'text' ? c.text : '')).join('') ?? ''

  // Build a display row list:
  // - Drop subliminal entries entirely (they're context injection).
  // - Drop assistant rows whose text is empty (those are tool-call frames).
  // - Collapse any consecutive run of tool messages into one summary row.
  type Row =
    | { kind: 'msg'; m: MessageEntry; text: string }
    | { kind: 'tools'; n: number; firstId: string }
  const rows: Row[] = []
  for (const m of messages) {
    if (m.role === 'subliminal') continue
    if (m.role === 'tool') {
      const last = rows[rows.length - 1]
      if (last && last.kind === 'tools') last.n += 1
      else rows.push({ kind: 'tools', n: 1, firstId: m.id })
      continue
    }
    const t = text(m).trim()
    if (m.role === 'assistant' && !t) continue
    rows.push({ kind: 'msg', m, text: t })
  }

  const handleReset = async () => {
    if (resetting) return
    if (!window.confirm(`Reset session ${sessionId}? This deletes the conversation history so the next utterance starts fresh.`)) return
    setResetting(true)
    try {
      await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
      setMessages([])
    } catch (e) {
      console.warn('reset failed:', e)
    } finally {
      setResetting(false)
    }
  }

  return (
    <div className="w-full max-w-2xl flex flex-col gap-1">
      <div className="text-xs text-muted-foreground flex items-center justify-between px-1">
        <span>Transcript — session <span className="font-mono text-foreground">{sessionId}</span></span>
        <div className="flex items-center gap-3">
          <span>{polledOnce ? `${rows.length} row${rows.length === 1 ? '' : 's'}` : 'loading…'}</span>
          <button
            onClick={handleReset}
            disabled={resetting}
            className="text-[11px] text-muted-foreground hover:text-destructive disabled:opacity-50 underline-offset-2 hover:underline"
          >
            {resetting ? 'resetting…' : 'reset session'}
          </button>
        </div>
      </div>
      <div
        ref={scrollRef}
        className="h-64 overflow-y-auto rounded-md border border-border bg-card p-3 space-y-2"
      >
        {rows.length === 0 && polledOnce && (
          <div className="text-xs text-muted-foreground text-center py-8">
            No messages yet. Speak to drop your first transcript here.
          </div>
        )}
        {rows.map((r, i) =>
          r.kind === 'tools' ? (
            <div key={`tools-${r.firstId}-${i}`} className="text-xs text-muted-foreground italic pl-22">
              <span className="inline-block w-20 mr-2 uppercase tracking-wider tabular-nums">tools</span>
              {r.n} tool {r.n === 1 ? 'call' : 'calls'} ↺
            </div>
          ) : (
            <div key={r.m.id} className="text-sm">
              <span
                className={cn(
                  'inline-block w-20 mr-2 text-xs uppercase tracking-wider tabular-nums',
                  r.m.role === 'user' ? 'text-primary' : 'text-emerald-400',
                )}
              >
                {r.m.role}
              </span>
              <span className="text-foreground whitespace-pre-wrap">{r.text}</span>
            </div>
          ),
        )}
      </div>
    </div>
  )
}

/** VU meter + state pill for the live preview. Reads volume off the local
 *  mic track and renders it as both a numeric dBFS readout and a horizontal
 *  bar — gives unambiguous feedback that audio is flowing even before the
 *  aura's modulation kicks in. */
function LiveStats({
  audioTrack,
  agentState,
}: {
  audioTrack: LocalAudioTrack | undefined
  agentState: AgentState
}) {
  // useTrackVolume returns 0..1 (analyser-derived); convert to dBFS for the
  // readout. Smoothing constant matches the visualizer hook so the bar and
  // the aura modulate in sync.
  const volume = useTrackVolume(audioTrack, { fftSize: 512, smoothingTimeConstant: 0.55 })
  const dbfs = volume > 0.0001 ? 20 * Math.log10(volume) : -120
  const dbDisplay = volume > 0 ? `${dbfs.toFixed(1)} dBFS` : '—'
  return (
    <div className="flex flex-col items-center gap-2 w-72">
      <div className="flex items-center justify-between w-full text-sm">
        <span className="text-muted-foreground">mic</span>
        <span className="tabular-nums text-foreground">{dbDisplay}</span>
        <span className="text-muted-foreground">{agentState}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-secondary overflow-hidden">
        <div
          className="h-full bg-primary transition-[width] duration-100"
          style={{ width: `${Math.min(100, volume * 200)}%` }}
        />
      </div>
    </div>
  )
}

export default function VoicePreviewPage() {
  const [color, setColor] = useState<`#${string}`>('#1FD5F9')
  const [mode, setMode] = useState<'live' | 'mock'>('live')
  const [mockState, setMockState] = useState<AgentState>('listening')
  const [autoCycle, setAutoCycle] = useState(true)

  useEffect(() => {
    if (mode !== 'mock' || !autoCycle) return
    const id = setInterval(() => {
      setMockState(prev => STATES[(STATES.indexOf(prev) + 1) % STATES.length])
    }, 2200)
    return () => clearInterval(id)
  }, [mode, autoCycle])

  const colorPicker = (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {COLORS.map(c => (
        <button
          key={c.hex}
          onClick={() => setColor(c.hex)}
          className={cn(
            'px-3 py-1.5 rounded-md text-sm font-medium border transition-colors flex items-center gap-2',
            color === c.hex
              ? 'border-foreground/40 bg-card'
              : 'border-border bg-card hover:bg-accent',
          )}
        >
          <span className="inline-block h-3 w-3 rounded-full ring-1 ring-border" style={{ background: c.hex }} />
          {c.name}
        </button>
      ))}
    </div>
  )

  const modeToggle = (
    <div className="flex items-center gap-2">
      <Button size="sm" variant={mode === 'live' ? 'default' : 'outline'} onClick={() => setMode('live')}>
        Live mic
      </Button>
      <Button size="sm" variant={mode === 'mock' ? 'default' : 'outline'} onClick={() => setMode('mock')}>
        Mock state
      </Button>
    </div>
  )

  if (mode === 'mock') {
    return (
      <div className="flex flex-col items-center justify-center w-full min-h-full p-6 gap-6 bg-background text-foreground">
        <div className="text-sm text-muted-foreground">
          Mock state — no audio source. Cycles agent state on a 2.2s timer.
        </div>
        {modeToggle}
        <AgentAudioVisualizerAura size="xl" state={mockState} color={color} themeMode="dark" />
        <div className="text-2xl font-medium tabular-nums text-primary">{mockState}</div>
        <div className="flex flex-wrap items-center justify-center gap-2">
          {STATES.map(s => (
            <Button
              key={s}
              size="sm"
              variant={mockState === s ? 'default' : 'outline'}
              onClick={() => { setAutoCycle(false); setMockState(s) }}
            >
              {s}
            </Button>
          ))}
          <span className="mx-2 h-5 w-px bg-border" />
          <Button size="sm" variant={autoCycle ? 'secondary' : 'outline'} onClick={() => setAutoCycle(c => !c)}>
            {autoCycle ? 'auto-cycling' : 'paused'}
          </Button>
        </div>
        {colorPicker}
      </div>
    )
  }

  return (
    <VoiceRoom sessionId={VOICE_PREVIEW_SESSION}>
      {({ status, agentState, localAudioTrack, error, reconnect }) => {
        // Until Phase 5A wires an actual agent audio track, repurpose
        // 'speaking' as "your voice is the source" so the aura's volume
        // modulation kicks in. Strict 'listening' would only run the
        // brightness pulse, with no reaction to mic level.
        const visualizerState: AgentState =
          status === 'connected' && localAudioTrack ? 'speaking' : agentState
        return (
          <div className="flex flex-col items-center justify-center w-full min-h-full p-6 gap-6 bg-background text-foreground">
            <div className="text-sm text-muted-foreground text-center max-w-xl">
              Live mic — your microphone publishes into a LiveKit room
              (<span className="font-mono text-foreground">lloyd-{VOICE_PREVIEW_SESSION}</span>).
              The aura modulates from your voice (preview repurposes the agent-speaking
              animation; Phase 5A swaps in the real agent audio track).
            </div>
            {modeToggle}
            <AgentAudioVisualizerAura
              size="xl"
              state={visualizerState}
              audioTrack={localAudioTrack}
              color={color}
              themeMode="dark"
            />
            <div className="flex items-center gap-3 text-sm">
              <span
                className={cn(
                  'inline-block h-2 w-2 rounded-full',
                  status === 'connected' && 'bg-emerald-400',
                  status === 'connecting' && 'bg-amber-400 animate-pulse',
                  status === 'idle' && 'bg-muted-foreground',
                  status === 'failed' && 'bg-destructive',
                )}
              />
              <span className="tabular-nums text-foreground">{status}</span>
              {status === 'failed' && (
                <Button size="sm" variant="outline" onClick={reconnect} className="ml-2">
                  Retry
                </Button>
              )}
            </div>
            {status === 'connected' && (
              <LiveStats audioTrack={localAudioTrack} agentState={visualizerState} />
            )}
            {error && (
              <div className="text-xs text-destructive max-w-md text-center break-words">
                {error}
              </div>
            )}
            <TranscriptLog sessionId={VOICE_PREVIEW_SESSION} />
            {colorPicker}
          </div>
        )
      }}
    </VoiceRoom>
  )
}
