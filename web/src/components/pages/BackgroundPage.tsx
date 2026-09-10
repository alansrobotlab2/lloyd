/**
 * Background — every run nobody was watching.
 *
 * The other half of the history bifurcation. `/api/sessions` is
 * conversations; this page is autonomy tasks and worker jobs, which until
 * 2026-09-10 either left no record at all (autonomy, `run_prompt_on_primary`)
 * or sat in the user's chat history looking like something they had said
 * (session-backed workers).
 *
 * Two halves, because "what ran" and "is the thing that runs it healthy" are
 * different questions asked at different times:
 *
 *   • Runs — recorded transcripts, grouped by producer. A row opens in the
 *     Inner Voice reader through the same `setPendingFocus` + `setCurrentTab`
 *     pair the agent's `mc_navigate` uses, so that page never has to know who
 *     asked.
 *   • Sources — per-source config, queue depth, outcome rollup and recent
 *     runs. `/api/workers/status` reports only what a source is *allowed* to
 *     do; a source failing every run looked exactly like one succeeding at
 *     every run.
 *
 * Both degrade to an empty state on a failed fetch rather than throwing. A
 * page about what went wrong unattended must not be the second thing to
 * break.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Moon, RefreshCw, Bot, Clock, CheckCircle2, XCircle, MinusCircle,
  BrainCircuit, Activity,
} from 'lucide-react'
import { useMcUi, useReportMcFocus, usePendingFocusFor } from '../../contexts/McUiContext'
import { api, type BackgroundSession, type WorkerSourceHealth } from '../../api'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'

const POLL_MS = 10_000

function relative(iso: string): string {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (!Number.isFinite(then)) return ''
  const delta = (Date.now() - then) / 1000
  if (delta < 60) return 'just now'
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`
  return `${Math.floor(delta / 86400)}d ago`
}

/** A run's producer, as the tab groups them. The session's `source` is the
 *  fine-grained identity (`autonomy-task:39`); this is the family. */
function producerOf(s: BackgroundSession): string {
  if (s.platform === 'autonomy') return 'autonomy'
  return s.source || s.platform || 'unknown'
}

// ── Runs ────────────────────────────────────────────────────────────────

function SessionRow({
  session, onOpen,
}: { session: BackgroundSession; onOpen: (id: string) => void }) {
  return (
    <button
      onClick={() => onOpen(session.id)}
      title={session.id}
      className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left
                 hover:bg-accent/60"
    >
      <Bot className="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
      <span className="min-w-0 flex-1 truncate text-xs text-foreground">
        {session.title || session.preview || session.id}
      </span>
      {session.inner_voice && (
        <BrainCircuit
          className="h-3 w-3 flex-shrink-0 text-violet-400"
          // Recording is universal; observation is the per-job opt-in, and
          // "was anyone watching?" is the first question about a run that
          // went wrong.
          aria-label="Inner Voice observed this run"
        />
      )}
      <span className="flex-shrink-0 text-[10px] text-muted-foreground">
        {session.message_count} msg
      </span>
      <span className="w-16 flex-shrink-0 text-right text-[10px] text-muted-foreground">
        {relative(session.last_active)}
      </span>
    </button>
  )
}

function RunsPanel({ onOpen }: { onOpen: (id: string) => void }) {
  const [sessions, setSessions] = useState<BackgroundSession[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      const out = await api.listBackgroundSessions(150)
      setSessions(out.sessions || [])
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    const t = setInterval(() => { void load() }, POLL_MS)
    return () => clearInterval(t)
  }, [load])

  const grouped = useMemo(() => {
    const by = new Map<string, BackgroundSession[]>()
    for (const s of sessions) {
      const key = producerOf(s)
      const list = by.get(key)
      if (list) list.push(s)
      else by.set(key, [s])
    }
    return [...by.entries()].sort((a, b) => b[1].length - a[1].length)
  }, [sessions])

  if (error) {
    return (
      <div className="rounded border border-border p-4 text-xs text-muted-foreground">
        Could not load background runs — {error}
      </div>
    )
  }
  if (!loading && sessions.length === 0) {
    return (
      <div className="rounded border border-border p-4 text-xs text-muted-foreground">
        No background runs recorded yet. Autonomy tasks and worker jobs appear
        here as they run.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {grouped.map(([producer, rows]) => (
        <div key={producer} className="rounded border border-border">
          <div className="flex items-baseline gap-2 border-b border-border px-3 py-2">
            <span className="text-xs font-medium text-foreground">{producer}</span>
            <span className="text-[10px] text-muted-foreground">
              {rows.length} run{rows.length === 1 ? '' : 's'}
            </span>
          </div>
          <div className="p-1">
            {rows.map(s => (
              <SessionRow key={s.id} session={s} onOpen={onOpen} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Sources ─────────────────────────────────────────────────────────────

const STATUS_ICON: Record<string, React.ComponentType<{ className?: string }>> = {
  success: CheckCircle2,
  failed: XCircle,
  skipped: MinusCircle,
}

function SourceCard({ source }: { source: WorkerSourceHealth }) {
  const h = source.health
  const queued = Object.values(source.depth || {}).reduce((a, b) => a + b, 0)
  return (
    <div className="rounded border border-border">
      <div className="flex flex-wrap items-baseline gap-2 border-b border-border px-3 py-2">
        <span className="text-xs font-medium text-foreground">{source.name}</span>
        <Badge variant={source.enabled ? 'default' : 'secondary'} className="text-[10px]">
          {source.enabled ? 'enabled' : 'off'}
        </Badge>
        {source.inner_voice && (
          <Badge variant="outline" className="gap-1 text-[10px]">
            <BrainCircuit className="h-2.5 w-2.5" /> observed
          </Badge>
        )}
        {!source.configured && (
          <Badge variant="outline" className="text-[10px]">unconfigured</Badge>
        )}
        <span className="ml-auto text-[10px] text-muted-foreground">
          {source.interval_seconds ? `every ${source.interval_seconds}s` : 'on demand'}
          {source.priority != null && ` · priority ${source.priority}`}
          {queued > 0 && ` · ${queued} queued`}
        </span>
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1 px-3 py-2 text-[10px] text-muted-foreground">
        {h === null ? (
          // Not "0% failing" — a rate over zero runs is unknown, and reading
          // it as healthy is exactly the mistake this panel exists to stop.
          <span>no runs in the window</span>
        ) : (
          <>
            <span>{h.total} run{h.total === 1 ? '' : 's'}</span>
            <span className="text-emerald-400">{h.ok} ok</span>
            {h.failed > 0 && <span className="text-rose-400">{h.failed} failed</span>}
            {h.skipped > 0 && <span>{h.skipped} skipped</span>}
            <span className={cn(
              h.fail_rate !== null && h.fail_rate >= 0.5 && 'text-rose-400',
            )}>
              {h.fail_rate === null ? '—' : `${Math.round(h.fail_rate * 100)}% failing`}
            </span>
            <span>{h.gpu_hours}h</span>
            {h.last_completed && <span>last {relative(h.last_completed)}</span>}
          </>
        )}
      </div>

      {source.recent.length > 0 && (
        <div className="space-y-0.5 border-t border-border px-3 py-2">
          {source.recent.map(run => {
            const Icon = STATUS_ICON[run.status] ?? Clock
            return (
              <div key={run.run_id} className="flex items-center gap-2 text-[10px]">
                <Icon className={cn(
                  'h-3 w-3 flex-shrink-0',
                  run.status === 'success' && 'text-emerald-400',
                  run.status === 'failed' && 'text-rose-400',
                  run.status === 'skipped' && 'text-muted-foreground',
                )} />
                <span className="min-w-0 flex-1 truncate text-muted-foreground">
                  {run.summary || run.run_id}
                </span>
                <span className="flex-shrink-0 text-muted-foreground/70">
                  {Math.round(run.duration_seconds || 0)}s
                </span>
                <span className="w-16 flex-shrink-0 text-right text-muted-foreground/70">
                  {relative(run.completed_at)}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function SourcesPanel() {
  const [sources, setSources] = useState<WorkerSourceHealth[]>([])
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const out = await api.workersHealth(7, 5)
      setSources(out.sources || [])
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void load()
    const t = setInterval(() => { void load() }, POLL_MS)
    return () => clearInterval(t)
  }, [load])

  if (error) {
    return (
      <div className="rounded border border-border p-4 text-xs text-muted-foreground">
        Could not load worker health — {error}
      </div>
    )
  }
  return (
    <div className="space-y-3">
      {sources.map(s => <SourceCard key={s.name} source={s} />)}
      {sources.length === 0 && (
        <div className="rounded border border-border p-4 text-xs text-muted-foreground">
          No worker sources configured.
        </div>
      )}
    </div>
  )
}

// ── Page ────────────────────────────────────────────────────────────────

export default function BackgroundPage() {
  const [tab, setTab] = useState('runs')
  const [focused, setFocused] = useState<string | null>(null)
  const { setCurrentTab, setPendingFocus } = useMcUi()

  useReportMcFocus('background', focused ? { kind: 'session', id: focused } : null)

  const pendingFocus = usePendingFocusFor('background')
  useEffect(() => {
    if (pendingFocus) setFocused(pendingFocus)
  }, [pendingFocus])

  // A background transcript is read in the Inner Voice timeline, which
  // already renders every row shape the recorder writes — including the
  // `role="thinking"` ones. Handing it over through the same pair the agent's
  // `mc_navigate` uses means that page never has to know who asked.
  const openInReader = useCallback((sessionId: string) => {
    setFocused(sessionId)
    setPendingFocus({ tab: 'inner_voice', focusId: sessionId })
    setCurrentTab('inner_voice')
  }, [setPendingFocus, setCurrentTab])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center gap-2 border-b border-border px-4 py-2.5">
        <Moon className="h-4 w-4 text-muted-foreground" />
        <h2 className="text-sm font-semibold text-foreground">Background</h2>
        <span className="text-[10px] text-muted-foreground">
          runs nobody was watching
        </span>
        <Button
          variant="ghost" size="sm" className="ml-auto h-7 gap-1 text-xs"
          onClick={() => window.location.reload()}
        >
          <RefreshCw className="h-3 w-3" /> Refresh
        </Button>
      </div>

      <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col">
        <TabsList className="mx-4 mt-3 w-fit">
          <TabsTrigger value="runs" className="gap-1 text-xs">
            <Bot className="h-3 w-3" /> Runs
          </TabsTrigger>
          <TabsTrigger value="sources" className="gap-1 text-xs">
            <Activity className="h-3 w-3" /> Sources
          </TabsTrigger>
        </TabsList>
        <TabsContent value="runs" className="min-h-0 flex-1 overflow-auto p-4">
          <RunsPanel onOpen={openInReader} />
        </TabsContent>
        <TabsContent value="sources" className="min-h-0 flex-1 overflow-auto p-4">
          <SourcesPanel />
        </TabsContent>
      </Tabs>
    </div>
  )
}
