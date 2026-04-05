import { useState, useEffect, useRef, useCallback } from 'react'
import { X, Square, CheckCircle, AlertTriangle, Clock, Cpu } from 'lucide-react'
import { api, type PipelineRun } from '../api'

type FullRun = PipelineRun & {
  task: string
  stage_outputs: Record<string, string>
  skills: string[]
  live_log: string
}

function useElapsed(createdAt: string): string {
  const [elapsed, setElapsed] = useState('')
  useEffect(() => {
    const update = () => {
      const s = Math.floor((Date.now() - new Date(createdAt).getTime()) / 1000)
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

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'running': return <span className="w-2 h-2 rounded-full bg-brand-400 animate-pulse inline-block" />
    case 'complete': return <CheckCircle className="w-4 h-4 text-green-400" />
    case 'aborted': return <X className="w-4 h-4 text-slate-500" />
    case 'blocked': return <AlertTriangle className="w-4 h-4 text-amber-400" />
    default: return <Clock className="w-4 h-4 text-slate-500" />
  }
}

interface Props {
  runId: number
  onClose: () => void
  onAbort: (runId: number) => Promise<void>
}

export default function PipelineModal({ runId, onClose, onAbort }: Props) {
  const [run, setRun] = useState<FullRun | null>(null)
  const [aborting, setAborting] = useState(false)
  const logRef = useRef<HTMLPreElement>(null)
  const autoScrollRef = useRef(true)
  const elapsed = useElapsed(run?.created_at ?? new Date().toISOString())

  const load = useCallback(async () => {
    try {
      const data = await api.getPipeline(runId)
      setRun(data)
    } catch { /* ignore */ }
  }, [runId])

  // Initial load + poll every 2s while running
  useEffect(() => {
    load()
    const id = setInterval(() => {
      if (run?.status === 'running') load()
    }, 2000)
    return () => clearInterval(id)
  }, [load, run?.status])

  // Auto-scroll log to bottom when new content arrives
  useEffect(() => {
    if (autoScrollRef.current && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [run?.live_log])

  const handleAbort = async () => {
    if (!confirm(`Kill pipeline run #${runId}?`)) return
    setAborting(true)
    try {
      await onAbort(runId)
      await load()
    } finally {
      setAborting(false)
    }
  }

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />

      {/* Modal */}
      <div className="relative w-full max-w-3xl max-h-[85vh] flex flex-col bg-surface-1 border border-surface-3/50 rounded-xl shadow-2xl overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-surface-3/30 flex-shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <Cpu className="w-4 h-4 text-brand-400 flex-shrink-0" />
            <span className="text-sm font-semibold text-slate-200">Pipeline #{runId}</span>
            {run && (
              <>
                <div className="flex items-center gap-1.5">
                  <StatusIcon status={run.status} />
                  <span className="text-xs text-slate-400">{run.status}</span>
                </div>
                <span className="text-xs text-slate-500 font-mono">{elapsed}</span>
              </>
            )}
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {run?.status === 'running' && (
              <button
                onClick={handleAbort}
                disabled={aborting}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-red-600/20 hover:bg-red-600/30 border border-red-500/40 text-red-400 transition-colors disabled:opacity-40"
              >
                <Square className="w-3 h-3" />
                Kill
              </button>
            )}
            <button onClick={onClose} className="p-1 text-slate-500 hover:text-slate-200 transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {!run ? (
          <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">Loading...</div>
        ) : (
          <div className="flex-1 flex flex-col overflow-hidden">

            {/* Meta row */}
            <div className="px-4 py-2 border-b border-surface-3/20 flex-shrink-0 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
              {run.model && <span>Model: <span className="text-slate-300">{run.model}</span></span>}
              {run.stage_count > 0 && (
                <span>Stage: <span className="text-slate-300">{run.current_stage} ({run.stage_index + 1}/{run.stage_count})</span></span>
              )}
              {run.skills?.length > 0 && (
                <span>Skills: <span className="text-slate-300">{run.skills.join(', ')}</span></span>
              )}
              {run.created_at && (
                <span>Started: <span className="text-slate-300 font-mono">{new Date(run.created_at).toLocaleTimeString()}</span></span>
              )}
            </div>

            {/* Stage progress pills */}
            {run.stages.length > 0 && (
              <div className="px-4 py-2 border-b border-surface-3/20 flex-shrink-0 flex items-center gap-1.5 flex-wrap">
                {run.stages.map((stage, i) => {
                  const done = i < run.stage_index
                  const active = i === run.stage_index && run.status === 'running'
                  const blocked = i === run.stage_index && run.status === 'blocked'
                  return (
                    <div key={stage} className="flex items-center gap-1">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                        done ? 'bg-green-500/20 text-green-400' :
                        active ? 'bg-brand-500/20 text-brand-400' :
                        blocked ? 'bg-amber-500/20 text-amber-400' :
                        'bg-surface-3/30 text-slate-500'
                      }`}>
                        {active && <span className="inline-block w-1 h-1 rounded-full bg-brand-400 animate-pulse mr-1 mb-0.5" />}
                        {stage}
                      </span>
                      {i < run.stages.length - 1 && <span className="text-slate-600 text-[10px]">→</span>}
                    </div>
                  )
                })}
              </div>
            )}

            {/* Task */}
            <div className="px-4 py-2 border-b border-surface-3/20 flex-shrink-0 max-h-24 overflow-y-auto">
              <div className="text-[10px] text-slate-500 uppercase tracking-wide mb-1">Task</div>
              <pre className="text-[11px] text-slate-300 whitespace-pre-wrap font-mono leading-relaxed">
                {run.task}
              </pre>
            </div>

            {/* Blocked reason */}
            {run.status === 'blocked' && run.blocked_reason && (
              <div className="px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 flex-shrink-0">
                <span className="text-[11px] text-amber-400">⚠ {run.blocked_reason}</span>
              </div>
            )}

            {/* Live log */}
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              <div className="px-4 pt-2 pb-1 flex items-center justify-between flex-shrink-0">
                <span className="text-[10px] text-slate-500 uppercase tracking-wide">Live Output</span>
                <button
                  onClick={() => { autoScrollRef.current = !autoScrollRef.current }}
                  className="text-[10px] text-slate-600 hover:text-slate-400 transition-colors"
                  title="Toggle auto-scroll"
                >
                  {autoScrollRef.current ? '↓ auto-scroll on' : '↓ auto-scroll off'}
                </button>
              </div>
              <pre
                ref={logRef}
                onScroll={() => {
                  const el = logRef.current
                  if (!el) return
                  autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
                }}
                className="flex-1 overflow-y-auto px-4 pb-4 text-[11px] text-slate-300 font-mono whitespace-pre-wrap leading-relaxed min-h-0"
              >
                {run.live_log || (run.status === 'running' ? 'Waiting for output...' : 'No output captured.')}
              </pre>
            </div>

          </div>
        )}
      </div>
    </div>
  )
}
