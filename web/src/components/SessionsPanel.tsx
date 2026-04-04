import { useState, useEffect, useRef, useCallback } from 'react'
import { MessageCircle, Clock, Loader2 } from 'lucide-react'
import { api } from '../api'

interface Session {
  id: string
  session_key: string
  last_active: string
  preview: string
  platform?: string
  model?: string
}

interface SessionsPanelProps {
  onSwitchSession: (key: string) => void
  currentSessionKey: string | null
  refreshTrigger?: number
}

export default function SessionsPanel({ onSwitchSession, currentSessionKey, refreshTrigger }: SessionsPanelProps) {
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const sessionListRef = useRef<HTMLDivElement>(null)

  // Scroll to selected session when it changes
  useEffect(() => {
    if (currentSessionKey && sessionListRef.current) {
      const selectedEl = sessionListRef.current.querySelector('[data-selected="true"]')
      if (selectedEl) {
        selectedEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
      }
    }
  }, [currentSessionKey, sessions])

  const loadSessions = useCallback(async () => {
    try {
      const result = await api.listSessions()
      if (result.sessions) {
        setSessions(result.sessions as unknown as Session[])
      }
    } catch (err) {
      console.error('Failed to load sessions:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadSessions()
    const interval = setInterval(loadSessions, 30000)
    return () => clearInterval(interval)
  }, [loadSessions])

  // Refresh when trigger changes (new session created, message sent)
  useEffect(() => {
    if (refreshTrigger) {
      loadSessions()
    }
  }, [refreshTrigger, loadSessions])

  return (
    <div className="flex flex-col h-full bg-surface-1 border-r border-surface-3/30 w-64">
      {/* Header */}
      <div className="p-3 border-b border-surface-3/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageCircle className="w-4 h-4 text-brand-400" />
            <span className="text-sm font-semibold text-slate-200">Sessions</span>
          </div>
          <button
            onClick={loadSessions}
            className="p-1 text-slate-400 hover:text-slate-200 rounded transition-colors"
            title="Refresh"
          >
            <Loader2 className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-slate-500" />
          </div>
        ) : sessions.length === 0 ? (
          <div className="text-center text-slate-500 p-4">
            <p className="text-sm">No sessions found</p>
          </div>
        ) : (
          <div className="divide-y divide-surface-3/30" ref={sessionListRef}>
            {sessions.map((session) => (
              <button
                key={session.session_key || session.id}
                data-selected={currentSessionKey === (session.session_key || session.id) ? "true" : undefined}
                onClick={() => onSwitchSession(session.session_key || session.id)}
                className={`w-full text-left p-2 rounded-lg transition-colors ${
                  currentSessionKey === (session.session_key || session.id)
                    ? 'bg-brand-600/20'
                    : 'hover:bg-surface-2'
                }`}
              >
                <div className="text-[10px] text-slate-400 truncate mb-1">
                  {session.preview || 'No preview'}
                </div>
                <div className="flex items-center gap-1 text-[10px] text-slate-500">
                  <Clock className="w-2.5 h-2.5" />
                  <span>{session.last_active}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-2 border-t border-surface-3/30">
        <div className="text-[10px] text-slate-500 text-center">
          {sessions.length} session{sessions.length !== 1 ? 's' : ''}
        </div>
      </div>
    </div>
  )
}
