import { useEffect, useState } from 'react'
import type { AgentState } from '@livekit/components-react'
import { AgentAudioVisualizerAura } from '../agents-ui/agent-audio-visualizer-aura'

const STATES: AgentState[] = ['idle', 'listening', 'thinking', 'speaking']
const COLORS: Array<{ name: string; hex: `#${string}` }> = [
  { name: 'LiveKit cyan', hex: '#1FD5F9' },
  { name: 'Lloyd indigo', hex: '#818CF8' },
  { name: 'Mint',         hex: '#34D399' },
  { name: 'Amber',        hex: '#F59E0B' },
  { name: 'Magenta',      hex: '#E879F9' },
]

export default function VoicePreviewPage() {
  const [state, setState] = useState<AgentState>('listening')
  const [color, setColor] = useState<`#${string}`>('#1FD5F9')
  const [autoCycle, setAutoCycle] = useState(true)

  useEffect(() => {
    if (!autoCycle) return
    const id = setInterval(() => {
      setState(prev => STATES[(STATES.indexOf(prev) + 1) % STATES.length])
    }, 2200)
    return () => clearInterval(id)
  }, [autoCycle])

  return (
    <div className="flex flex-col items-center justify-center w-full min-h-full p-6 gap-6 bg-background text-foreground">
      <div className="text-sm text-muted-foreground">
        Phase 2A preview — aura visualizer driven by mock agent state. No audio source yet.
      </div>

      <AgentAudioVisualizerAura
        size="xl"
        state={state}
        color={color}
        themeMode="dark"
      />

      <div className="text-2xl font-medium tabular-nums text-primary">
        {state}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        {STATES.map(s => (
          <button
            key={s}
            onClick={() => { setAutoCycle(false); setState(s) }}
            className={[
              'px-3 py-1.5 rounded-md text-sm font-medium border transition-colors',
              state === s
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-card text-foreground border-border hover:bg-accent'
            ].join(' ')}
          >
            {s}
          </button>
        ))}
        <span className="mx-2 h-5 w-px bg-border" />
        <button
          onClick={() => setAutoCycle(c => !c)}
          className={[
            'px-3 py-1.5 rounded-md text-sm font-medium border transition-colors',
            autoCycle
              ? 'bg-primary/15 text-primary border-primary/40'
              : 'bg-card text-foreground border-border hover:bg-accent'
          ].join(' ')}
        >
          {autoCycle ? 'auto-cycling' : 'paused'}
        </button>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        {COLORS.map(c => (
          <button
            key={c.hex}
            onClick={() => setColor(c.hex)}
            className={[
              'px-3 py-1.5 rounded-md text-sm font-medium border transition-colors flex items-center gap-2',
              color === c.hex
                ? 'border-foreground/40 bg-card'
                : 'border-border bg-card hover:bg-accent'
            ].join(' ')}
          >
            <span
              className="inline-block h-3 w-3 rounded-full ring-1 ring-border"
              style={{ background: c.hex }}
            />
            {c.name}
          </button>
        ))}
      </div>
    </div>
  )
}
