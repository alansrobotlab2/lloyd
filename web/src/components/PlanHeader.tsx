import { useEffect, useState } from 'react'
import { ScrollText, FileText, ChevronDown, ChevronRight, Square, Loader2 } from 'lucide-react'
import { api, type SessionPlan } from '../api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Collapsible, CollapsibleContent, CollapsibleTrigger,
} from '@/components/ui/collapsible'

interface PlanHeaderProps {
  sessionId: string | null
  refreshKey: number
  onExitPlanMode?: () => void
}

/**
 * Plan B — surfaces session plan state above the todo list.
 *
 * Two render modes:
 *   - plan_mode=true: an amber banner with an "Exit plan mode" button.
 *   - committed plan exists: a collapsible card showing the markdown body
 *     (loaded lazily on first expand).
 *
 * Hidden when neither applies.
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

  useEffect(() => {
    setMdLoaded(false)
    setPlanMd('')
    setExpanded(false)
  }, [plan.plan_md_path, plan.committed_at])

  const ensureMdLoaded = async () => {
    if (mdLoaded || !sessionId || !plan.plan_md_path) return
    setMdLoading(true)
    try {
      const doc = await api.getSessionPlanDocument(sessionId)
      setPlanMd(doc.plan_md || '')
      setMdLoaded(true)
    } finally {
      setMdLoading(false)
    }
  }

  const handleOpenChange = (open: boolean) => {
    setExpanded(open)
    if (open) ensureMdLoaded()
  }

  const handleExit = async () => {
    if (!sessionId) return
    setExiting(true)
    try {
      await api.exitPlanMode(sessionId)
      setPlan({ ...plan, plan_mode: false, cancelled_at: new Date().toISOString() })
      onExitPlanMode?.()
    } finally {
      setExiting(false)
    }
  }

  // Plan mode active — render amber banner with exit button.
  if (plan.plan_mode) {
    return (
      <div className={cn(
        'border-t border-amber-500/30 bg-amber-500/10 px-4 py-2',
        'flex items-center gap-3 text-xs',
      )}>
        <ScrollText className="w-3.5 h-3.5 shrink-0 text-amber-400" />
        <span className="text-amber-300 font-semibold uppercase tracking-wide">Plan Mode</span>
        <span className="text-amber-200/70 truncate flex-1">
          Drafting plan — write tools blocked until commit
        </span>
        <Button
          variant="outline"
          size="sm"
          onClick={handleExit}
          disabled={exiting}
          title="Abandon plan mode without committing (/exit-plan)"
          className={cn(
            'h-6 px-2 text-[10px] font-mono uppercase tracking-wide gap-1',
            'border-amber-500/40 text-amber-200 hover:bg-amber-500/20 hover:text-amber-100',
          )}
        >
          {exiting ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <Square className="w-3 h-3" />
          )}
          /exit-plan
        </Button>
      </div>
    )
  }

  // Committed plan — render collapsible card.
  if (plan.plan_md_path && plan.committed_at) {
    const stages = plan.stages || []
    return (
      <Collapsible open={expanded} onOpenChange={handleOpenChange} className="border-t border-border bg-secondary/20">
        <CollapsibleTrigger className={cn(
          'w-full flex items-center gap-2 px-4 py-2 text-xs',
          'text-muted-foreground hover:text-foreground transition-colors',
        )}>
          <FileText className="w-3.5 h-3.5 shrink-0 text-primary" />
          <span className="font-semibold uppercase tracking-wide">Plan</span>
          {stages.length > 0 && (
            <Badge variant="secondary" className="font-mono text-[10px] px-1.5 py-0">
              {stages.length} stage{stages.length === 1 ? '' : 's'}
            </Badge>
          )}
          {expanded ? (
            <ChevronDown className="w-3 h-3 ml-auto" />
          ) : (
            <ChevronRight className="w-3 h-3 ml-auto" />
          )}
        </CollapsibleTrigger>
        <CollapsibleContent className="px-4 pb-2.5 max-h-64 overflow-y-auto">
          {mdLoading ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading plan…
            </div>
          ) : planMd ? (
            <pre className="text-[11px] leading-snug text-foreground/90 whitespace-pre-wrap font-mono">
              {planMd}
            </pre>
          ) : (
            <div className="text-xs text-muted-foreground py-2">Plan markdown not available.</div>
          )}
          {stages.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border space-y-0.5">
              {stages.map((s) => (
                <div key={s.n} className="text-[10px] text-muted-foreground leading-snug">
                  <span className="text-muted-foreground font-mono">{s.n}.</span>{' '}
                  <span className="text-foreground">{s.title}</span>
                  {s.summary && <div className="ml-4 text-muted-foreground">{s.summary}</div>}
                </div>
              ))}
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>
    )
  }

  return null
}
