import { useState, useEffect, useCallback } from 'react'
import { X, ChevronDown, ChevronRight, Cpu, MessageSquare } from 'lucide-react'
import { api, type PipelineRun, type ActiveProc } from '../api'
import PipelineModal from './PipelineModal'

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

function statusColor(status: string) {
  switch (status) {
    case 'running': return 'text-brand-400'
    case 'complete': return 'text-green-400'
    case 'aborted': return 'text-slate-500'
    case 'blocked': return 'text-amber-400'
    default: return 'text-slate-400'
  }
}

function AgentCard({ run, onAbort, onSelect }: { run: PipelineRun; onAbort: (id: number) => void; onSelect: (id: number) => void }) {
  const elapsed = useElapsed(run.created_at)
  const [aborting, setAborting] = useState(false)

  const handleAbort = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm(`Kill pipeline run #${run.run_id}?`)) return
    setAborting(true)
    try {
      await onAbort(run.run_id)
    } finally {
      setAborting(false)
    }
  }

  return (
    <div
      onClick={() => onSelect(run.run_id)}
      className="bg-surface-2 border border-surface-3/40 rounded-lg p-2.5 space-y-1.5 cursor-pointer hover:border-brand-500/30 hover:bg-surface-2/80 transition-colors"
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-1">
        <div className="flex items-center gap-1.5 min-w-0">
          {run.status === 'running' && (
            <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse flex-shrink-0 mt-0.5" />
          )}
          {run.status === 'blocked' && (
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 flex-shrink-0 mt-0.5" />
          )}
          <span className="text-[10px] text-slate-300 truncate leading-tight">
            {run.task_preview}
          </span>
        </div>
        {(run.status === 'running' || run.status === 'blocked') && (
          <button
            onClick={handleAbort}
            disabled={aborting}
            className="flex-shrink-0 p-0.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-40"
            title="Kill"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      {/* Status row */}
      <div className="flex items-center justify-between text-[10px]">
        <div className="flex items-center gap-2">
          <span className={`font-medium ${statusColor(run.status)}`}>
            {run.status}
          </span>
          {run.stage_count > 0 && (
            <span className="text-slate-500 font-mono">
              {run.current_stage} {run.stage_index + 1}/{run.stage_count}
            </span>
          )}
          {run.model && (
            <span className="text-slate-600 truncate max-w-[60px]">
              {run.model.split('-')[0]}
            </span>
          )}
        </div>
        <span className="text-slate-500 font-mono flex-shrink-0">{elapsed}</span>
      </div>

      {/* Blocked reason */}
      {run.status === 'blocked' && run.blocked_reason && (
        <div className="text-[10px] text-amber-400/80 truncate">
          ⚠ {run.blocked_reason}
        </div>
      )}
    </div>
  )
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
    <div className="bg-surface-2 border border-surface-3/40 rounded-lg p-2.5 space-y-1.5">
      <div className="flex items-start justify-between gap-1">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse flex-shrink-0 mt-0.5" />
          <span className="text-[10px] text-slate-300 truncate leading-tight">
            {proc.preview || proc.session_id || `pid ${proc.pid}`}
          </span>
        </div>
        <button
          onClick={handleKill}
          disabled={killing || !proc.session_id}
          className="flex-shrink-0 p-0.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-40"
          title="Kill subprocess"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
      <div className="flex items-center justify-between text-[10px]">
        <div className="flex items-center gap-2">
          <span className="font-medium text-brand-400">running</span>
          {!proc.streaming && (
            <span className="text-amber-400/80" title="SSE disconnected — orphaned subprocess">orphaned</span>
          )}
          <span className="text-slate-600 truncate max-w-[60px]">{modelShort}</span>
          <span className="text-slate-600 font-mono text-[9px]">pid {proc.pid}</span>
        </div>
        <span className="text-slate-500 font-mono flex-shrink-0">{elapsed}</span>
      </div>
    </div>
  )
}

export default function RunningAgentsPanel() {
  const [runs, setRuns] = useState<PipelineRun[]>([])
  const [procs, setProcs] = useState<ActiveProc[]>([])
  const [expanded, setExpanded] = useState(true)
  const [sessionsExpanded, setSessionsExpanded] = useState(true)
  const [showAll, setShowAll] = useState(false)
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)

  const loadRuns = useCallback(async () => {
    try {
      const result = await api.listPipelines()
      if (result.runs) setRuns(result.runs)
    } catch { /* silently ignore */ }
  }, [])

  const loadProcs = useCallback(async () => {
    try {
      const result = await api.getActiveProcs()
      if (result.procs) setProcs(result.procs)
    } catch { /* silently ignore */ }
  }, [])

  useEffect(() => {
    loadRuns()
    loadProcs()
    const id = setInterval(() => { loadRuns(); loadProcs() }, 3000)
    return () => clearInterval(id)
  }, [loadRuns, loadProcs])

  const handleAbort = useCallback(async (runId: number) => {
    await api.abortPipeline(runId)
    await loadRuns()
  }, [loadRuns])

  const handleKillProc = useCallback(async (sessionId: string) => {
    await api.killSessionProc(sessionId)
    await loadProcs()
  }, [loadProcs])

  const runningRuns = runs.filter(r => r.status === 'running')
  const otherRuns = runs.filter(r => r.status !== 'running').slice(0, 5)
  const displayed = showAll ? [...runningRuns, ...otherRuns] : runningRuns


  return (
    <>
    {selectedRunId !== null && (
      <PipelineModal
        runId={selectedRunId}
        onClose={() => setSelectedRunId(null)}
        onAbort={async (id) => { await handleAbort(id); }}
      />
    )}

    {/* Active Sessions */}
    {(procs.length > 0) && (
      <div className="border-t border-surface-3/30 flex-shrink-0">
        <button
          onClick={() => setSessionsExpanded(e => !e)}
          className="w-full flex items-center justify-between px-3 py-2 text-[11px] font-semibold text-slate-400 hover:text-slate-200 transition-colors"
        >
          <div className="flex items-center gap-1.5">
            <MessageSquare className="w-3 h-3" />
            <span>Sessions</span>
            <span className="bg-brand-500/20 text-brand-400 rounded px-1 py-0.5 text-[9px] font-mono">
              {procs.length}
            </span>
            {procs.some(p => !p.streaming) && (
              <span className="bg-amber-500/20 text-amber-400 rounded px-1 py-0.5 text-[9px]">orphaned</span>
            )}
          </div>
          {sessionsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        </button>
        {sessionsExpanded && (
          <div className="px-2 pb-2 space-y-1.5 max-h-48 overflow-y-auto">
            {procs.map(proc => (
              <ActiveSessionCard key={proc.pid} proc={proc} onKill={handleKillProc} />
            ))}
          </div>
        )}
      </div>
    )}

    {/* Pipelines */}
    <div className="border-t border-surface-3/30 flex-shrink-0">
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-center justify-between px-3 py-2 text-[11px] font-semibold text-slate-400 hover:text-slate-200 transition-colors"
      >
        <div className="flex items-center gap-1.5">
          <Cpu className="w-3 h-3" />
          <span>Pipelines</span>
          {runningRuns.length > 0 && (
            <span className="bg-brand-500/20 text-brand-400 rounded px-1 py-0.5 text-[9px] font-mono">
              {runningRuns.length}
            </span>
          )}
        </div>
        {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>

      {expanded && (
        <div className="px-2 pb-2 space-y-1.5 max-h-64 overflow-y-auto">
          {displayed.length === 0 ? (
            <div className="text-[10px] text-slate-600 text-center py-2">No recent pipelines</div>
          ) : (
            displayed.map(run => (
              <AgentCard key={run.run_id} run={run} onAbort={handleAbort} onSelect={setSelectedRunId} />
            ))
          )}
          {otherRuns.length > 0 && (
            <button
              onClick={() => setShowAll(s => !s)}
              className="w-full text-[10px] text-slate-600 hover:text-slate-400 transition-colors py-0.5"
            >
              {showAll ? '↑ Hide blocked/recent' : `↓ Show ${otherRuns.length} blocked/recent`}
            </button>
          )}
        </div>
      )}
    </div>
    </>
  )
}
