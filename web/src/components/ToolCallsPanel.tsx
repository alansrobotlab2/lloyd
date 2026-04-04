import { ChevronRight, Loader2 } from 'lucide-react'

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
        <div key={call.id} className="bg-surface-2/50 rounded border border-surface-3/20 overflow-hidden">
          <button
            onClick={() => onToggle(call.id)}
            className="w-full flex items-center justify-between p-2 hover:bg-surface-2/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              {call.status === 'pending' && <Loader2 className="w-3 h-3 animate-spin text-brand-400" />}
              {call.status === 'success' && <div className="w-3 h-3 rounded-full bg-green-500" />}
              {call.status === 'error' && <div className="w-3 h-3 rounded-full bg-red-500" />}
              <span className="text-xs font-mono text-slate-300">{call.name}</span>
            </div>
            <ChevronRight className="w-3 h-3 text-slate-500" />
          </button>
        </div>
      ))}
    </div>
  )
}
