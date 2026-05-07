import { useState } from 'react'
import type { InnerVoiceObservation, InnerVoiceObservationTrigger } from '../api'
import { actionStyle, ACTION_STYLES, TRIGGER_LABEL, parseObservationTime } from './innerVoiceStyles'
import { cn } from '@/lib/utils'

const timeStr = (iso: string) => {
  const ms = parseObservationTime(iso)
  if (!ms || isNaN(ms)) return ''
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// IV silently approving the turn end (`trigger=result` + `action=noop`) means
// the goal-completion check passed — render it green like an explicit
// acknowledge_complete instead of the muted noop styling.
function effectiveStyle(obs: InnerVoiceObservation) {
  if (obs.trigger === 'result' && obs.action === 'noop') {
    return { ...ACTION_STYLES.acknowledge_complete, label: 'approve: turn complete' }
  }
  return actionStyle(obs.action)
}

export default function ObservationBubble({ obs }: { obs: InnerVoiceObservation }) {
  const [expanded, setExpanded] = useState(false)
  const s = effectiveStyle(obs)
  const Icon = s.Icon
  const hasContent = !!(obs.content && obs.content.trim())
  const triggerLabel = TRIGGER_LABEL[obs.trigger as InnerVoiceObservationTrigger] || obs.trigger

  return (
    <div className="w-full">
      <div className={cn('rounded-xl border px-3 py-2', s.border, s.bg)}>
        <button
          type="button"
          onClick={() => hasContent && setExpanded(!expanded)}
          className={cn(
            'w-full flex items-center gap-2 text-left text-[11px]',
            hasContent ? 'cursor-pointer hover:opacity-90' : 'cursor-default',
          )}
        >
          <Icon className={cn('w-3.5 h-3.5 flex-shrink-0', s.color)} />
          <span className={cn('font-mono font-semibold', s.color)}>{s.label}</span>
          <span className="text-muted-foreground">·</span>
          <span className="text-muted-foreground">{triggerLabel}</span>
          {obs.related_tool && (
            <>
              <span className="text-muted-foreground">·</span>
              <span className="font-mono text-muted-foreground truncate">{obs.related_tool}</span>
            </>
          )}
          {obs.reason && (
            <span className="text-foreground/90 truncate ml-1">{obs.reason}</span>
          )}
        </button>
        {expanded && hasContent && (
          <div className={cn(
            'mt-2 px-2 py-1.5 rounded border bg-card border-border',
            'text-foreground/90 whitespace-pre-wrap font-mono text-[11px]',
          )}>
            {obs.content}
          </div>
        )}
        {obs.error && (
          <div className="mt-1 text-[10px] text-destructive font-mono">error: {obs.error}</div>
        )}
      </div>
      <div className="mt-1.5 text-[10px] text-muted-foreground/70 font-mono flex flex-wrap items-center gap-x-2.5 gap-y-0.5">
        <span>{timeStr(obs.created_at)}</span>
        {obs.input_tokens != null && obs.input_tokens > 0 && (() => {
          const pct = (obs.input_tokens! / 262144 * 100).toFixed(1)
          return <span>ctx: {obs.input_tokens!.toLocaleString()} ({pct}%)</span>
        })()}
        {obs.cache_read != null && obs.cache_read > 0 && (
          <span className="text-emerald-700">cache↑: {obs.cache_read.toLocaleString()}</span>
        )}
        {obs.cache_create != null && obs.cache_create > 0 && (
          <span className="text-amber-700">cache✎: {obs.cache_create.toLocaleString()}</span>
        )}
        {obs.output_tokens != null && obs.output_tokens > 0 && (
          <span>out: {obs.output_tokens.toLocaleString()}</span>
        )}
        {obs.latency_ms != null && (
          <span>time: {(obs.latency_ms / 1000).toFixed(1)}s</span>
        )}
        <span className="text-muted-foreground/60">#{obs.sequence_in_turn}</span>
      </div>
    </div>
  )
}
