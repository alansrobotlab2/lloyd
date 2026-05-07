import { cn } from '@/lib/utils'

/** Visual indicator for the wake-word gate state. Five states top-to-bottom:
 *
 *   disconnected — gray dot, "not connected"
 *   thinking     — amber pulse, "Lloyd is thinking…"
 *   speaking     — primary pulse, "Lloyd is speaking" (continuation will
 *                  open after he finishes)
 *   listening    — emerald pulse, "Listening (Xs left)" + countdown bar
 *   idle         — muted, "Say 'Lloyd' to start"
 *
 * Order matters because the worker sets wake='listening' as soon as the
 * wake-word matches — but if the agent is currently speaking, the user
 * cares more about that than the still-open continuation window.
 */
export interface WakeStatePillProps {
  status: 'idle' | 'connecting' | 'connected' | 'failed'
  wakeState: 'idle' | 'listening'
  wakeRemainingS: number
  wakeContinuationS: number
  wakeSpeaker: string | null
  agentSpeaking: boolean
  agentThinking: boolean
  /** Compact mode: smaller padding/text for sidebar use. */
  compact?: boolean
}

export function WakeStatePill({
  status,
  wakeState,
  wakeRemainingS,
  wakeContinuationS,
  wakeSpeaker,
  agentSpeaking,
  agentThinking,
  compact,
}: WakeStatePillProps) {
  const padding = compact ? 'px-3 py-1.5 text-xs' : 'px-4 py-2 text-sm'
  const wWidth = compact ? 'w-full' : 'w-72'

  if (status !== 'connected') {
    return (
      <div className={cn(
        'flex items-center gap-2 rounded-full border border-border bg-card text-muted-foreground',
        padding,
      )}>
        <span className="inline-block h-2 w-2 rounded-full bg-muted-foreground" />
        Not connected
      </div>
    )
  }
  if (agentThinking) {
    return (
      <div className={cn(
        'flex items-center gap-2 rounded-full border border-amber-500/40 bg-amber-500/10 text-amber-300',
        padding,
      )}>
        <span className="inline-block h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
        Lloyd is thinking…
      </div>
    )
  }
  if (agentSpeaking) {
    return (
      <div className={cn(
        'flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 text-primary',
        padding,
      )}>
        <span className="inline-block h-2 w-2 rounded-full bg-primary animate-pulse" />
        Lloyd is speaking
      </div>
    )
  }
  if (wakeState === 'listening') {
    const pct = Math.max(0, Math.min(100, (wakeRemainingS / wakeContinuationS) * 100))
    return (
      <div className={cn(
        'flex flex-col items-stretch gap-1.5 rounded-2xl border border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
        padding, wWidth,
      )}>
        <div className="flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-medium">
            Listening{wakeSpeaker ? ` — ${wakeSpeaker}` : ''}
          </span>
          <span className="ml-auto tabular-nums text-emerald-200/80 text-xs">
            {wakeRemainingS.toFixed(1)}s
          </span>
        </div>
        <div className="h-1 w-full rounded-full bg-emerald-900/40 overflow-hidden">
          <div
            className="h-full bg-emerald-400 transition-[width] duration-100"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    )
  }
  return (
    <div className={cn(
      'flex items-center gap-2 rounded-full border border-border bg-card text-muted-foreground',
      padding,
    )}>
      <span className="inline-block h-2 w-2 rounded-full bg-muted-foreground/60" />
      Say <span className="font-mono text-foreground">'Lloyd'</span> to start
    </div>
  )
}
