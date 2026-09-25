import { useState, useEffect, useCallback } from 'react'
import { X, ChevronDown, ChevronRight, MessageSquare, Square, Bot } from 'lucide-react'
import { api, type ActiveProc, type SubagentRun } from '../api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

function useElapsed(createdAt: string): string {
  const [elapsed, setElapsed] = useState('')
  useEffect(() => {
    const update = () => {
      const ms = Date.now() - new Date(createdAt).getTime()
      const s = Math.floor(ms / 1000)
      if (s < 60) setElapsed(`${s}s`)
      else if (s < 3600) setElapsed(`${Math.floor(s / 60)}m ${s % 60}s`)
      else setElapsed(`${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`)
    }
    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [createdAt])
  return elapsed
}

function ActiveSessionCard({ proc, onKill }: { proc: ActiveProc; onKill: (sessionId: string) => void }) {
  const elapsed = useElapsed(proc.created_at ?? new Date().toISOString())
  const [killing, setKilling] = useState(false)

  const handleKill = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!proc.session_id) return
    if (!confirm(`Kill session subprocess (pid ${proc.pid})?`)) return
    setKilling(true)
    try {
      await onKill(proc.session_id)
    } finally {
      setKilling(false)
    }
  }

  const modelShort = proc.model?.split('-')[0] ?? 'unknown'

  return (
    <div className="bg-secondary/40 border border-border rounded-md p-2.5 space-y-1.5">
      <div className="flex items-start justify-between gap-1">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse flex-shrink-0 mt-0.5" />
          <span className="text-[10px] text-foreground truncate leading-tight">
            {proc.preview || proc.session_id || `pid ${proc.pid}`}
          </span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleKill}
          disabled={killing || !proc.session_id}
          title="Kill subprocess"
          className="h-5 w-5 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
        >
          <X className="w-3 h-3" />
        </Button>
      </div>
      <div className="flex items-center justify-between text-[10px]">
        <div className="flex items-center gap-2">
          <span className="font-medium text-primary">running</span>
          {!proc.streaming && (
            <span className="text-amber-400/80" title="SSE disconnected — orphaned subprocess">
              orphaned
            </span>
          )}
          <span className="text-muted-foreground/70 truncate max-w-[60px]">{modelShort}</span>
          <span className="text-muted-foreground/70 font-mono text-[9px]">pid {proc.pid}</span>
        </div>
        <span className="text-muted-foreground font-mono flex-shrink-0">{elapsed}</span>
      </div>
    </div>
  )
}

// One running Task child. The stop button asks the aggregator to set the
// child's cancel event (it stops at its next safe point and stays resumable by
// task_id). It acts FOR the session that spawned the child — the aggregator's
// orchestrator-session policy refuses anything else — so a row with no parent
// session has no button that could work, and shows none.
function SubagentCard({ run, onCancel }: {
  run: SubagentRun
  onCancel: (run: SubagentRun) => Promise<void>
}) {
  const [stopping, setStopping] = useState(false)
  const canStop = !!run.task_id && !!run.parent_session_id && run.cancellable !== false

  const handleStop = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!canStop) return
    if (!confirm(`Stop subagent "${run.description || run.subagent_type}"?`)) return
    setStopping(true)
    try {
      await onCancel(run)
    } finally {
      setStopping(false)
    }
  }

  return (
    <div className="bg-secondary/40 border border-border rounded-md p-2.5 space-y-1.5">
      <div className="flex items-start justify-between gap-1">
        <div className="flex items-center gap-1.5 min-w-0">
          <Bot className="w-3 h-3 text-primary flex-shrink-0" />
          <span className="text-[10px] text-foreground truncate leading-tight">
            {run.description || run.prompt_preview || run.task_id || run.run_id}
          </span>
        </div>
        {canStop && (
          <Button
            variant="ghost"
            size="icon"
            onClick={handleStop}
            disabled={stopping}
            title="Stop subagent (resumable by task_id)"
            className="h-5 w-5 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
          >
            <Square className="w-3 h-3" />
          </Button>
        )}
      </div>
      <div className="flex items-center justify-between text-[10px]">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-medium text-primary">{run.subagent_type}</span>
          <span className="text-muted-foreground/70 font-mono text-[9px]">
            {run.turns}/{run.max_turns}
          </span>
          {run.last_tool && (
            <span className="text-muted-foreground/70 truncate max-w-[70px]">{run.last_tool}</span>
          )}
        </div>
        <span className="text-muted-foreground font-mono flex-shrink-0">
          {Math.round(run.elapsed_s)}s
        </span>
      </div>
    </div>
  )
}

export default function RunningAgentsPanel() {
  const [procs, setProcs] = useState<ActiveProc[]>([])
  const [sessionsExpanded, setSessionsExpanded] = useState(true)
  const [subagents, setSubagents] = useState<SubagentRun[]>([])

  const loadProcs = useCallback(async () => {
    try {
      const result = await api.getActiveProcs()
      if (result.procs) setProcs(result.procs)
    } catch { /* silently ignore */ }
  }, [])

  const loadSubagents = useCallback(async () => {
    try {
      const result = await api.getSubagents()
      setSubagents(result.active ?? [])
    } catch { /* silently ignore */ }
  }, [])

  useEffect(() => {
    loadProcs()
    loadSubagents()
    const id = setInterval(() => { loadProcs(); loadSubagents() }, 3000)
    return () => clearInterval(id)
  }, [loadProcs, loadSubagents])

  const handleCancelSubagent = useCallback(async (run: SubagentRun) => {
    if (!run.task_id) return
    const res = await api.cancelSubagent(run.task_id, run.parent_session_id, 'stopped from Mission Control')
    if (res.error) alert(res.error)
    await loadSubagents()
  }, [loadSubagents])

  const handleKillProc = useCallback(async (sessionId: string) => {
    await api.killSessionProc(sessionId)
    await loadProcs()
  }, [loadProcs])

  if (procs.length === 0 && subagents.length === 0) return null

  const orphanedCount = procs.filter(p => !p.streaming).length

  return (
    <div className="border-t border-border flex-shrink-0">
      <button
        onClick={() => setSessionsExpanded(e => !e)}
        className={cn(
          'w-full flex items-center justify-between px-3 py-2 text-[11px] font-semibold transition-colors',
          'text-muted-foreground hover:text-foreground',
        )}
      >
        <div className="flex items-center gap-1.5">
          <MessageSquare className="w-3 h-3" />
          <span>Sessions</span>
          <Badge variant="secondary" className="px-1 py-0 text-[9px] font-mono bg-primary/15 text-primary">
            {procs.length + subagents.length}
          </Badge>
          {orphanedCount > 0 && (
            <Badge
              variant="secondary"
              className="px-1 py-0 text-[9px] bg-amber-500/15 text-amber-400 border-transparent"
            >
              orphaned
            </Badge>
          )}
        </div>
        {sessionsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>
      {sessionsExpanded && (
        <div className="px-2 pb-2 space-y-1.5 max-h-48 overflow-y-auto">
          {procs.map(proc => (
            <ActiveSessionCard key={proc.pid} proc={proc} onKill={handleKillProc} />
          ))}
          {subagents.map(run => (
            <SubagentCard key={run.run_id} run={run} onCancel={handleCancelSubagent} />
          ))}
        </div>
      )}
    </div>
  )
}
