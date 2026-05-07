import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { MessageCircle, Clock, Loader2 } from 'lucide-react'
import { api } from '../api'
import { useSessionActivity } from '../hooks/useSessionActivity'
import RunningAgentsPanel from './RunningAgentsPanel'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'

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
  refreshTrigger?: number | string
  activeSessions?: Set<string>
}

export default function SessionsPanel({ onSwitchSession, currentSessionKey, refreshTrigger, activeSessions }: SessionsPanelProps) {
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const sessionListRef = useRef<HTMLDivElement>(null)

  const activitySessions = useMemo(
    () => sessions.map(s => ({ sessionKey: s.session_key || s.id, lastActivity: s.last_active })),
    [sessions],
  )
  const { hasActivity, markSeen } = useSessionActivity(activitySessions, true, currentSessionKey)

  useEffect(() => {
    if (currentSessionKey) {
      markSeen(currentSessionKey)
      if (sessionListRef.current) {
        const selectedEl = sessionListRef.current.querySelector('[data-selected="true"]')
        if (selectedEl) {
          selectedEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
        }
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

  useEffect(() => {
    if (refreshTrigger) loadSessions()
  }, [refreshTrigger, loadSessions])

  return (
    <div className="flex flex-col h-full bg-card text-card-foreground w-64">
      {/* Header */}
      <div className="p-3 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageCircle className="w-4 h-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">Sessions</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={loadSessions}
            title="Refresh"
            className="h-6 w-6 text-muted-foreground hover:text-foreground"
          >
            <Loader2 className={cn('w-3 h-3', loading && 'animate-spin')} />
          </Button>
        </div>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto p-2">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : sessions.length === 0 ? (
          <div className="text-center text-muted-foreground p-4">
            <p className="text-sm">No sessions found</p>
          </div>
        ) : (
          <div className="divide-y divide-border" ref={sessionListRef}>
            {sessions.map((session) => {
              const key = session.session_key || session.id
              const isCurrent = currentSessionKey === key
              const isActive = activeSessions?.has(key) ?? false
              const hasNewActivity = hasActivity(key)
              return (
                <button
                  key={key}
                  data-selected={isCurrent ? 'true' : undefined}
                  onClick={() => { markSeen(key); onSwitchSession(key) }}
                  className={cn(
                    'w-full text-left p-2 rounded-md transition-colors',
                    isCurrent
                      ? 'bg-primary/15 hover:bg-primary/20'
                      : 'hover:bg-accent',
                  )}
                >
                  <div className="text-[10px] text-muted-foreground truncate mb-1">
                    {session.preview || 'No preview'}
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1 text-[10px] text-muted-foreground/70">
                      <Clock className="w-2.5 h-2.5" />
                      <span>{session.last_active}</span>
                    </div>
                    {isActive && (
                      <span
                        className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse flex-shrink-0"
                        aria-label="streaming"
                      />
                    )}
                    {hasNewActivity && !isActive && (
                      <span
                        className="w-2 h-2 rounded-full bg-emerald-400 flex-shrink-0"
                        aria-label="unread activity"
                      />
                    )}
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      <Separator />
      <div className="p-2">
        <div className="text-[10px] text-muted-foreground/70 text-center">
          {sessions.length} session{sessions.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Running pipelines / subagents */}
      <RunningAgentsPanel />
    </div>
  )
}
