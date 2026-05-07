import { useEffect, useState } from 'react'
import type { AgentState } from '@livekit/components-react'
import { AgentAudioVisualizerAura } from '../agents-ui/agent-audio-visualizer-aura'
import VoiceRoom from '../VoiceRoom'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const STATES: AgentState[] = ['idle', 'listening', 'thinking', 'speaking']
const COLORS: Array<{ name: string; hex: `#${string}` }> = [
  { name: 'LiveKit cyan', hex: '#1FD5F9' },
  { name: 'Lloyd indigo', hex: '#818CF8' },
  { name: 'Mint',         hex: '#34D399' },
  { name: 'Amber',        hex: '#F59E0B' },
  { name: 'Magenta',      hex: '#E879F9' },
]

// Stable session id so reloads land in the same room and the agent worker
// can recognise the connection.
const VOICE_PREVIEW_SESSION = 'voice_preview'

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
      <Button
        size="sm"
        variant={mode === 'live' ? 'default' : 'outline'}
        onClick={() => setMode('live')}
      >
        Live mic
      </Button>
      <Button
        size="sm"
        variant={mode === 'mock' ? 'default' : 'outline'}
        onClick={() => setMode('mock')}
      >
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
          <Button
            size="sm"
            variant={autoCycle ? 'secondary' : 'outline'}
            onClick={() => setAutoCycle(c => !c)}
          >
            {autoCycle ? 'auto-cycling' : 'paused'}
          </Button>
        </div>
        {colorPicker}
      </div>
    )
  }

  return (
    <VoiceRoom sessionId={VOICE_PREVIEW_SESSION}>
      {({ status, agentState, localAudioTrack, error, reconnect }) => (
        <div className="flex flex-col items-center justify-center w-full min-h-full p-6 gap-6 bg-background text-foreground">
          <div className="text-sm text-muted-foreground text-center max-w-xl">
            Live mic — your microphone publishes into a LiveKit room (
            <span className="font-mono text-foreground">lloyd-{VOICE_PREVIEW_SESSION}</span>
            ); the aura modulates from the actual audio levels.
          </div>
          {modeToggle}
          <AgentAudioVisualizerAura
            size="xl"
            state={agentState}
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
            <span className="text-muted-foreground">·</span>
            <span className="text-muted-foreground">agent: <span className="text-foreground tabular-nums">{agentState}</span></span>
            {status === 'failed' && (
              <Button size="sm" variant="outline" onClick={reconnect} className="ml-2">
                Retry
              </Button>
            )}
          </div>
          {error && (
            <div className="text-xs text-destructive max-w-md text-center break-words">
              {error}
            </div>
          )}
          {colorPicker}
        </div>
      )}
    </VoiceRoom>
  )
}
