/**
 * Inner Voice — chat-driven UI (thin observer model).
 *
 * Single-column centerline timeline:
 *   • Left of the centerline: primary actions (user / assistant / tool / subliminal).
 *   • Right of the centerline: observer decisions (one per row).
 * Items merge chronologically by timestamp; ChatPanel owns the rendering.
 *
 * The "+ new chat" button creates a session with `inner_voice: true` and
 * `inner_voice_evaluate_user_turns: true` so the observer fires on user-typed
 * chat messages, not just ambient turns.
 *
 * Polling sources of truth:
 *   /api/inner_voice/state         → header counts (every 4s)
 *   /api/inner_voice/observations  → timeline (every 3s)
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  BrainCircuit,
  Activity,
  RefreshCw,
  Plus,
  MessageSquare,
  Bot,
} from 'lucide-react'
import {
  api,
  type InnerVoiceObservation,
  type InnerVoiceSession,
  type InnerVoiceState,
} from '../../api'
import ChatPanel from '../ChatPanel'
import { actionStyle } from '../innerVoiceStyles'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'

const DEFAULT_STATE: InnerVoiceState = {
  session_id: null,
  inner_voice_enabled: false,
  evaluate_user_turns: false,
  observations_count_by_action: {},
  last_observation_at: null,
  latest_goal_card: null,
  latest_user_request: null,
  latest_turn_id: null,
}

// ─────────────────────────────────────────────────────────────────────────
// Top-level page
// ─────────────────────────────────────────────────────────────────────────

export default function InnerVoicePage() {
  const [obsState, setObsState] = useState<InnerVoiceState>(DEFAULT_STATE)
  const [sessions, setSessions] = useState<InnerVoiceSession[]>([])
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [observations, setObservations] = useState<InnerVoiceObservation[]>([])
  const [refreshKey, setRefreshKey] = useState(0)
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const [showAgentDetails, setShowAgentDetails] = useState(false)

  // Sessions list
  const loadSessions = useCallback(async () => {
    try {
      const r = await api.innerVoiceSessions(50)
      setSessions(r.sessions || [])
      if (!selectedSession && r.sessions && r.sessions.length > 0) {
        setSelectedSession(r.sessions[0].session_id)
      }
    } catch {
      // best-effort
    }
  }, [selectedSession])

  useEffect(() => {
    loadSessions()
  }, [loadSessions, refreshKey])

  // State header poll
  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.innerVoiceState(selectedSession || undefined)
        if (!cancelled) setObsState({ ...DEFAULT_STATE, ...r })
      } catch {
        // best-effort
      }
    }
    poll()
    const t = setInterval(poll, 4000)
    return () => { cancelled = true; clearInterval(t) }
  }, [selectedSession, refreshKey])

  // Route voice transcripts to the inner-voice session this tab is showing.
  // Layout's effect skips its routing while page === 'inner_voice', so this
  // takes over until the user navigates away (then Layout re-asserts the chat
  // slot's session on the next page change).
  useEffect(() => {
    api.voiceSetActiveSession(selectedSession ?? null).catch(() => {})
  }, [selectedSession])

  // Observations poll. Sort ascending by id so chronological merge in ChatPanel
  // sees them in their natural order (the API returns newest-first).
  useEffect(() => {
    if (!selectedSession) { setObservations([]); return }
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.innerVoiceObservations(selectedSession, undefined, 200)
        if (!cancelled) {
          const list = (r.observations || []).slice().sort((a, b) => a.id - b.id)
          setObservations(list)
        }
      } catch { /* best-effort */ }
    }
    poll()
    const t = setInterval(poll, 3000)
    return () => { cancelled = true; clearInterval(t) }
  }, [selectedSession, refreshKey])

  // Create new Inner Voice session
  const handleCreateSession = useCallback(async () => {
    setCreateError(null)
    setCreating(true)
    try {
      const r = await api.createSession({
        model: 'primary',
        platform: 'mission-control',
        inner_voice: true,
        inner_voice_evaluate_user_turns: true,
      })
      setSelectedSession(r.session_id)
      setRefreshKey(k => k + 1)
    } catch (e) {
      setCreateError(e instanceof Error ? e.message : 'create failed')
    } finally {
      setCreating(false)
    }
  }, [])

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-border/30 flex-shrink-0">
        <BrainCircuit className="w-5 h-5 text-primary" />
        <h2 className="text-lg font-semibold text-foreground">Inner Voice</h2>

        <SessionPicker
          sessions={sessions}
          selectedSession={selectedSession}
          onSelect={setSelectedSession}
        />

        <div className="ml-auto flex items-center gap-2">
          {createError && (
            <span className="text-xs text-red-400 font-mono" title={createError}>
              create failed
            </span>
          )}
          <button
            onClick={() => setShowAgentDetails((v) => !v)}
            title={showAgentDetails ? 'Hide agent details' : 'Show agent details'}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              showAgentDetails
                ? 'text-primary bg-primary/15'
                : 'text-muted-foreground hover:text-primary hover:bg-primary/10'
            }`}
          >
            <Bot className="w-3.5 h-3.5" />
            Agent Details
          </button>
          <button
            onClick={handleCreateSession}
            disabled={creating}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-primary border border-primary text-white text-xs font-medium hover:bg-primary transition disabled:opacity-50"
            title="Create a new Inner Voice chat session (observer fires on user turns)"
          >
            <Plus className="w-3.5 h-3.5" />
            {creating ? 'creating…' : 'new chat'}
          </button>
          <button
            onClick={() => setRefreshKey(k => k + 1)}
            className="p-1.5 rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground transition"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Counts strip — observer state + per-action breakdown */}
      <CountsStrip obsState={obsState} />

      {/* Context row — most recent user input on the left, IV-captured intent on the right */}
      <ContextRow obsState={obsState} />

      {/* Single-column timeline (centerline rendered inside ChatPanel) */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden bg-background/50">
        {selectedSession ? (
          <ChatPanel
            key={selectedSession}
            requestedSessionKey={selectedSession}
            currentSessionKey={selectedSession}
            visible={true}
            showAgentDetails={showAgentDetails}
            onSessionLoaded={() => {}}
            onActiveSessionChange={() => {}}
            timelineRight={observations}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center max-w-md">
              <MessageSquare className="w-12 h-12 text-muted-foreground/70 mx-auto mb-3" />
              <div className="text-sm text-muted-foreground mb-2">No Inner Voice session selected</div>
              <div className="text-xs text-muted-foreground mb-4">
                Click <span className="font-mono text-primary">+ new chat</span> above
                to start a session where the observer watches every chat turn,
                or pick an existing session from the dropdown.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// Session picker (compact, in header)
// ─────────────────────────────────────────────────────────────────────────

function SessionPicker({
  sessions,
  selectedSession,
  onSelect,
}: {
  sessions: InnerVoiceSession[]
  selectedSession: string | null
  onSelect: (id: string | null) => void
}) {
  return (
    <div className="flex items-center gap-2 ml-2">
      <span className="text-xs text-muted-foreground">session:</span>
      <Select
        value={selectedSession ?? '__none__'}
        onValueChange={(v) => onSelect(v === '__none__' ? null : v)}
      >
        <SelectTrigger className="h-8 w-72 text-xs font-mono"><SelectValue /></SelectTrigger>
        <SelectContent>
          {sessions.length === 0 && <SelectItem value="__none__">— none —</SelectItem>}
          {sessions.map(s => (
            <SelectItem key={s.session_id} value={s.session_id} className="font-mono text-xs">
              {s.session_id}
              {s.evaluate_user_turns ? ' [chat]' : ''}
              {s.experiment_id ? ` · ${s.experiment_id}` : ''}
              {s.message_count ? ` · ${s.message_count}msg` : ''}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// Counts strip — observer state + per-action breakdown chips
// ─────────────────────────────────────────────────────────────────────────

function CountsStrip({ obsState }: { obsState: InnerVoiceState }) {
  const counts = obsState.observations_count_by_action || {}
  const totals = useMemo(() => {
    const total = Object.values(counts).reduce((a, b) => a + b, 0)
    const noop = (counts.noop || 0)
      + (counts.allow || 0)
      + (counts.noop_budget_exhausted || 0)
      + (counts.noop_empty_content || 0)
      + (counts.noop_no_ambient_channel || 0)
      + (counts.noop_ambient_failed || 0)
      + (counts.noop_pretool_after_cancel || 0)
    const interventions = (counts.inject || 0) + (counts.cancel || 0)
      + (counts.ambient || 0) + (counts.clarify || 0) + (counts.deny_tool || 0)
    return { total, noop, interventions }
  }, [counts])

  return (
    <div className="flex items-center flex-wrap gap-x-3 gap-y-1 px-6 py-2 border-b border-border/30 flex-shrink-0 text-[11px]">
      <div className="flex items-center gap-1.5">
        <Activity className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-foreground/90">observer</span>
        <span className={`font-mono ${obsState.inner_voice_enabled ? 'text-primary' : 'text-muted-foreground'}`}>
          {obsState.inner_voice_enabled ? 'enabled' : 'disabled'}
        </span>
        {obsState.evaluate_user_turns && (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-primary/10 text-primary border border-primary/20">
            chat
          </span>
        )}
      </div>
      <InlineStat label="rows" value={String(totals.total)} />
      <InlineStat label="interventions" value={String(totals.interventions)} warn={totals.interventions > 0} />
      <InlineStat label="noops" value={String(totals.noop)} />
      {totals.total > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          {Object.entries(counts)
            .sort((a, b) => b[1] - a[1])
            .map(([action, n]) => {
              const s = actionStyle(action)
              return (
                <span
                  key={action}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${s.color} ${s.bg} ${s.border}`}
                  title={`${action}: ${n}`}
                >
                  {s.label} · {n}
                </span>
              )
            })}
        </div>
      )}
      {obsState.last_observation_at && (
        <span className="ml-auto text-[10px] text-muted-foreground font-mono">
          last: {obsState.last_observation_at}
        </span>
      )}
    </div>
  )
}

function InlineStat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className={`flex items-baseline gap-1 px-2 py-0.5 rounded bg-secondary/60 border border-border/30 ${warn ? 'border-amber-500/30' : ''}`}>
      <span className="text-[9px] uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className={`font-mono text-[11px] ${warn ? 'text-amber-400' : 'text-foreground/90'}`}>{value}</span>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// Context row — last user input (left) | IV-captured intent (right)
// ─────────────────────────────────────────────────────────────────────────

function ContextRow({ obsState }: { obsState: InnerVoiceState }) {
  const userText = obsState.latest_user_request
  const goal = obsState.latest_goal_card
  const successCriteria = goal?.success_criteria || []
  const outOfScope = goal?.out_of_scope || []
  const completionSignals = goal?.completion_signals || []
  const hasGoal = successCriteria.length + outOfScope.length + completionSignals.length > 0

  if (!userText && !hasGoal) return null

  return (
    <div className="grid grid-cols-2 gap-3 px-6 py-2 border-b border-border/30 flex-shrink-0 text-[11px]">
      {/* Left — user input */}
      <div className="min-w-0">
        <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-1">user input</div>
        <div className="text-foreground leading-relaxed whitespace-pre-wrap break-words">
          {userText || <span className="text-muted-foreground italic">(no user request yet)</span>}
        </div>
      </div>
      {/* Right — IV-captured intent */}
      <div className="min-w-0">
        <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-1">iv intent</div>
        {!hasGoal ? (
          <div className="text-muted-foreground italic">(no actionable goal extracted)</div>
        ) : (
          <div className="space-y-1">
            {successCriteria.length > 0 && (
              <GoalList label="success" items={successCriteria} color="text-emerald-400" />
            )}
            {completionSignals.length > 0 && (
              <GoalList label="signals" items={completionSignals} color="text-primary" />
            )}
            {outOfScope.length > 0 && (
              <GoalList label="out of scope" items={outOfScope} color="text-amber-400" />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function GoalList({ label, items, color }: { label: string; items: string[]; color: string }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
      <span className="text-[9px] uppercase tracking-wider text-muted-foreground flex-shrink-0">{label}:</span>
      {items.map((item, i) => (
        <span key={i} className={`${color} text-[11px]`}>
          {item}{i < items.length - 1 && <span className="text-muted-foreground/70"> ·</span>}
        </span>
      ))}
    </div>
  )
}
