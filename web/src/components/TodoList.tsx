import { useEffect, useState } from 'react'
import { Circle, CheckCircle2, Loader2, ListChecks } from 'lucide-react'
import { api, type TodoItem } from '../api'

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

  return (
    <details
      open
      className="border-t border-surface-3/50 bg-surface-2/40"
    >
      <summary className="cursor-pointer list-none flex items-center gap-2 px-4 py-2 text-xs text-slate-400 hover:text-slate-300 transition-colors">
        <ListChecks className="w-3.5 h-3.5 shrink-0 text-brand-400" />
        <span className="font-semibold uppercase tracking-wide">Tasks</span>
        <span className="text-slate-500 font-mono">
          {counts.completed}/{todos.length}
        </span>
        {counts.in_progress > 0 && (
          <span className="text-amber-400/80 font-mono">
            · {counts.in_progress} in progress
          </span>
        )}
      </summary>
      <ul className="px-4 pb-2.5 space-y-1 max-h-48 overflow-y-auto">
        {todos.map((t, i) => (
          <li
            key={i}
            className={`flex items-start gap-2 text-xs leading-snug ${
              t.status === 'completed' ? 'text-slate-500' : 'text-slate-300'
            }`}
          >
            <span className="mt-0.5 shrink-0">
              {t.status === 'completed' ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500/80" />
              ) : t.status === 'in_progress' ? (
                <Loader2 className="w-3.5 h-3.5 text-amber-400 animate-spin" />
              ) : (
                <Circle className="w-3.5 h-3.5 text-slate-600" />
              )}
            </span>
            <span className={t.status === 'completed' ? 'line-through' : ''}>
              {t.status === 'in_progress' ? t.activeForm : t.content}
            </span>
          </li>
        ))}
      </ul>
    </details>
  )
}
