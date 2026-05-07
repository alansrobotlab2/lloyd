import { useEffect, useState } from 'react'
import { Circle, CheckCircle2, Loader2, ListChecks } from 'lucide-react'
import { api, type TodoItem } from '../api'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import {
  Collapsible, CollapsibleContent, CollapsibleTrigger,
} from '@/components/ui/collapsible'

interface TodoListProps {
  sessionId: string | null
  refreshKey: number
}

export default function TodoList({ sessionId, refreshKey }: TodoListProps) {
  const [todos, setTodos] = useState<TodoItem[]>([])

  useEffect(() => {
    if (!sessionId) {
      setTodos([])
      return
    }
    let cancelled = false
    api.getSessionTodos(sessionId).then(t => {
      if (!cancelled) setTodos(t)
    })
    return () => { cancelled = true }
  }, [sessionId, refreshKey])

  if (todos.length === 0) return null

  const counts = {
    completed: todos.filter(t => t.status === 'completed').length,
    in_progress: todos.filter(t => t.status === 'in_progress').length,
    pending: todos.filter(t => t.status === 'pending').length,
  }

  const completionPct = todos.length > 0 ? (counts.completed / todos.length) * 100 : 0

  return (
    <Collapsible defaultOpen className="border-t border-border bg-secondary/30">
      <CollapsibleTrigger className={cn(
        'w-full flex items-center gap-2 px-4 py-2 text-xs',
        'text-muted-foreground hover:text-foreground transition-colors',
      )}>
        <ListChecks className="w-3.5 h-3.5 shrink-0 text-primary" />
        <span className="font-semibold uppercase tracking-wide">Tasks</span>
        <Badge variant="secondary" className="font-mono text-[10px] px-1.5 py-0">
          {counts.completed}/{todos.length}
        </Badge>
        {counts.in_progress > 0 && (
          <span className="text-amber-400/80 font-mono">
            · {counts.in_progress} in progress
          </span>
        )}
        <Progress
          value={completionPct}
          className="ml-auto w-20 h-1 bg-muted"
          aria-label={`${counts.completed} of ${todos.length} completed`}
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="px-4 pb-2.5 space-y-1 max-h-48 overflow-y-auto">
          {todos.map((t, i) => (
            <li
              key={i}
              className={cn(
                'flex items-start gap-2 text-xs leading-snug',
                t.status === 'completed' ? 'text-muted-foreground' : 'text-foreground',
              )}
            >
              <span className="mt-0.5 shrink-0">
                {t.status === 'completed' ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500/80" />
                ) : t.status === 'in_progress' ? (
                  <Loader2 className="w-3.5 h-3.5 text-amber-400 animate-spin" />
                ) : (
                  <Circle className="w-3.5 h-3.5 text-muted-foreground/70" />
                )}
              </span>
              <span className={t.status === 'completed' ? 'line-through' : ''}>
                {t.status === 'in_progress' ? t.activeForm : t.content}
              </span>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  )
}
