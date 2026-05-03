import { useState, useEffect, useCallback } from 'react'
import { X, ChevronDown, ChevronRight, MessageSquare } from 'lucide-react'
import { api, type ActiveProc } from '../api'

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
  const [procs, setProcs] = useState<ActiveProc[]>([])
  const [sessionsExpanded, setSessionsExpanded] = useState(true)

  const loadProcs = useCallback(async () => {
    try {
      const result = await api.getActiveProcs()
      if (result.procs) setProcs(result.procs)
    } catch { /* silently ignore */ }
  }, [])

  useEffect(() => {
    loadProcs()
    const id = setInterval(loadProcs, 3000)
    return () => clearInterval(id)
  }, [loadProcs])

  const handleKillProc = useCallback(async (sessionId: string) => {
    await api.killSessionProc(sessionId)
    await loadProcs()
  }, [loadProcs])

  if (procs.length === 0) return null

  return (
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
  )
}
