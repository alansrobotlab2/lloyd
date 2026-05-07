import { ChevronRight, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ToolCall {
  id: string
  name: string
  status: 'pending' | 'success' | 'error'
  input?: string
  output?: string
  error?: string
}

interface ToolCallsPanelProps {
  toolCalls: ToolCall[]
  onToggle: (id: string) => void
  expanded: boolean
}

export function ToolCallsPanel({ toolCalls, onToggle, expanded }: ToolCallsPanelProps) {
  if (!expanded || toolCalls.length === 0) return null

  return (
    <div className="mt-2 space-y-1">
      {toolCalls.map((call) => (
        <div
          key={call.id}
          className="bg-secondary/50 rounded border border-border overflow-hidden"
        >
          <button
            onClick={() => onToggle(call.id)}
            className="w-full flex items-center justify-between p-2 hover:bg-accent transition-colors"
          >
            <div className="flex items-center gap-2">
              {call.status === 'pending' && (
                <Loader2 className="w-3 h-3 animate-spin text-primary" />
              )}
              {call.status === 'success' && (
                <div className="w-3 h-3 rounded-full bg-emerald-500" />
              )}
              {call.status === 'error' && (
                <div className="w-3 h-3 rounded-full bg-destructive" />
              )}
              <span className={cn('text-xs font-mono', 'text-foreground')}>{call.name}</span>
            </div>
            <ChevronRight className="w-3 h-3 text-muted-foreground" />
          </button>
        </div>
      ))}
    </div>
  )
}
