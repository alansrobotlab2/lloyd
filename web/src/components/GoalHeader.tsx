import { useEffect, useState } from 'react'
import { Target, CheckCircle2, X, Loader2 } from 'lucide-react'
import { api, type SessionGoal } from '../api'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

interface GoalHeaderProps {
  sessionId: string | null
  refreshKey: number
  onCleared?: () => void
}

/**
 * Surfaces the session's persistent /goal above the plan + todo lists.
 *
 * Render modes:
 *   - active goal (no achieved_at): blue banner with goal text, attempt
 *     count, and a "Clear goal" button.
 *   - achieved: green banner showing the goal was met (dismissible).
 *   - none: returns null.
 */
export default function GoalHeader({ sessionId, refreshKey, onCleared }: GoalHeaderProps) {
  const [goal, setGoal] = useState<SessionGoal>({})
  const [clearing, setClearing] = useState(false)

  useEffect(() => {
    if (!sessionId) {
      setGoal({})
      return
    }
    let cancelled = false
    api.getSessionGoal(sessionId).then(g => {
      if (!cancelled) setGoal(g || {})
    })
    return () => { cancelled = true }
  }, [sessionId, refreshKey])

  const handleClear = async () => {
    if (!sessionId) return
    setClearing(true)
    try {
      await api.clearSessionGoal(sessionId)
      setGoal({})
      onCleared?.()
    } finally {
      setClearing(false)
    }
  }

  const text = (goal.text || '').trim()
  if (!text) return null

  const attempts = goal.attempts || 0
  const achieved = !!goal.achieved_at

  if (achieved) {
    return (
      <div className={cn(
        'border-t border-emerald-500/30 bg-emerald-500/10 px-4 py-2',
        'flex items-center gap-3 text-xs',
      )}>
        <CheckCircle2 className="w-3.5 h-3.5 shrink-0 text-emerald-400" />
        <span className="text-emerald-300 font-semibold uppercase tracking-wide">Goal Met</span>
        <span className="text-emerald-200/80 truncate flex-1" title={text}>
          {text}
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleClear}
          disabled={clearing}
          title="Dismiss the achieved goal banner"
          className="h-6 px-2 text-[10px] font-mono uppercase tracking-wide gap-1 text-emerald-300 hover:bg-emerald-500/20"
        >
          {clearing ? <Loader2 className="w-3 h-3 animate-spin" /> : <X className="w-3 h-3" />}
          dismiss
        </Button>
      </div>
    )
  }

  return (
    <div className={cn(
      'border-t border-sky-500/30 bg-sky-500/10 px-4 py-2',
      'flex items-center gap-3 text-xs',
    )}>
      <Target className="w-3.5 h-3.5 shrink-0 text-sky-400" />
      <span className="text-sky-300 font-semibold uppercase tracking-wide">Goal</span>
      {attempts > 0 && (
        <Badge variant="secondary" className="font-mono text-[10px] px-1.5 py-0">
          attempt {attempts}
        </Badge>
      )}
      <span className="text-sky-200/80 truncate flex-1" title={text}>
        {text}
      </span>
      <Button
        variant="outline"
        size="sm"
        onClick={handleClear}
        disabled={clearing}
        title="Clear the goal (/clear-goal)"
        className={cn(
          'h-6 px-2 text-[10px] font-mono uppercase tracking-wide gap-1',
          'border-sky-500/40 text-sky-200 hover:bg-sky-500/20 hover:text-sky-100',
        )}
      >
        {clearing ? <Loader2 className="w-3 h-3 animate-spin" /> : <X className="w-3 h-3" />}
        /clear-goal
      </Button>
    </div>
  )
}
