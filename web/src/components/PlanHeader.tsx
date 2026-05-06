import { useEffect, useState } from 'react'
import { ScrollText, FileText, ChevronDown, ChevronRight, Square, Loader2 } from 'lucide-react'
import { api, type SessionPlan } from '../api'

interface PlanHeaderProps {
  sessionId: string | null
  refreshKey: number
  onExitPlanMode?: () => void
}

/**
 * Plan B — surfaces session plan state above the todo list.
 *
 * Two render modes:
 *   - plan_mode=true: an amber banner with an "Exit plan mode" button
 *     so the user can abandon a stuck plan-mode session without typing
 *     a slash command.
 *   - committed plan exists: a collapsible card showing the markdown
 *     body (loaded lazily on first expand to avoid pre-fetching every
 *     session's plan_md on mount).
 *
 * Hidden when neither applies — keeps the chat panel uncluttered for
 * sessions that never used /plan.
 */
export default function PlanHeader({ sessionId, refreshKey, onExitPlanMode }: PlanHeaderProps) {
  const [plan, setPlan] = useState<SessionPlan>({ plan_mode: false })
  const [planMd, setPlanMd] = useState<string>('')
  const [mdLoaded, setMdLoaded] = useState(false)
  const [mdLoading, setMdLoading] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [exiting, setExiting] = useState(false)

  useEffect(() => {
    if (!sessionId) {
      setPlan({ plan_mode: false })
      setPlanMd('')
      setMdLoaded(false)
      setExpanded(false)
      return
    }
    let cancelled = false
    api.getSessionPlan(sessionId).then(p => {
      if (!cancelled) setPlan(p || { plan_mode: false })
    })
    return () => { cancelled = true }
  }, [sessionId, refreshKey])

  // Reset md cache when plan changes (path different from what we loaded)
  useEffect(() => {
    setMdLoaded(false)
    setPlanMd('')
    setExpanded(false)
  }, [plan.plan_md_path, plan.committed_at])

  const handleExpand = async () => {
    const next = !expanded
    setExpanded(next)
    if (next && !mdLoaded && sessionId && plan.plan_md_path) {
      setMdLoading(true)
      try {
        const doc = await api.getSessionPlanDocument(sessionId)
        setPlanMd(doc.plan_md || '')
        setMdLoaded(true)
      } finally {
        setMdLoading(false)
      }
    }
  }

  const handleExit = async () => {
    if (!sessionId) return
    setExiting(true)
    try {
      await api.exitPlanMode(sessionId)
      // Optimistic local update; refreshKey from parent will re-fetch.
      setPlan({ ...plan, plan_mode: false, cancelled_at: new Date().toISOString() })
      onExitPlanMode?.()
    } finally {
      setExiting(false)
    }
  }

  // Plan mode active — render banner with exit button.
  if (plan.plan_mode) {
    return (
      <div className="border-t border-amber-500/30 bg-amber-500/10 px-4 py-2 flex items-center gap-3 text-xs">
        <ScrollText className="w-3.5 h-3.5 shrink-0 text-amber-400" />
        <span className="text-amber-300 font-semibold uppercase tracking-wide">Plan Mode</span>
        <span className="text-amber-200/70 truncate flex-1">
          Drafting plan — write tools blocked until commit
        </span>
        <button
          onClick={handleExit}
          disabled={exiting}
          className="px-2 py-0.5 text-[10px] font-mono uppercase tracking-wide text-amber-200 hover:text-amber-100 border border-amber-500/40 rounded hover:bg-amber-500/20 transition-colors disabled:opacity-50 flex items-center gap-1"
          title="Abandon plan mode without committing (/exit-plan)"
        >
          {exiting ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <Square className="w-3 h-3" />
          )}
          /exit-plan
        </button>
      </div>
    )
  }

  // Committed plan — render collapsible header.
  if (plan.plan_md_path && plan.committed_at) {
    const stages = plan.stages || []
    return (
      <details
        open={expanded}
        onToggle={(e) => {
          // Keep React state in sync with native <details> toggle so
          // expand triggers the lazy md fetch.
          const isOpen = (e.target as HTMLDetailsElement).open
          if (isOpen !== expanded) handleExpand()
        }}
        className="border-t border-surface-3/50 bg-surface-2/30"
      >
        <summary className="cursor-pointer list-none flex items-center gap-2 px-4 py-2 text-xs text-slate-400 hover:text-slate-300 transition-colors">
          <FileText className="w-3.5 h-3.5 shrink-0 text-brand-400" />
          <span className="font-semibold uppercase tracking-wide">Plan</span>
          {stages.length > 0 && (
            <span className="text-slate-500 font-mono">
              {stages.length} stage{stages.length === 1 ? '' : 's'}
            </span>
          )}
          {expanded ? (
            <ChevronDown className="w-3 h-3 ml-auto" />
          ) : (
            <ChevronRight className="w-3 h-3 ml-auto" />
          )}
        </summary>
        <div className="px-4 pb-2.5 max-h-64 overflow-y-auto">
          {mdLoading ? (
            <div className="flex items-center gap-2 text-xs text-slate-400 py-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading plan…
            </div>
          ) : planMd ? (
            <pre className="text-[11px] leading-snug text-slate-300 whitespace-pre-wrap font-mono">
              {planMd}
            </pre>
          ) : (
            <div className="text-xs text-slate-500 py-2">Plan markdown not available.</div>
          )}
          {stages.length > 0 && (
            <div className="mt-2 pt-2 border-t border-surface-3/40 space-y-0.5">
              {stages.map((s) => (
                <div key={s.n} className="text-[10px] text-slate-500 leading-snug">
                  <span className="text-slate-400 font-mono">{s.n}.</span>{' '}
                  <span className="text-slate-300">{s.title}</span>
                  {s.summary && (
                    <div className="ml-4 text-slate-500">{s.summary}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </details>
    )
  }

  return null
}
