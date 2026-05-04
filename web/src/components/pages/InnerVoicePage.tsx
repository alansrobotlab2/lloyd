/**
 * Inner Voice — chat-driven UI (thin observer model).
 *
 * Two-pane layout:
 *   • Left:  ChatPanel bound to the selected Inner Voice session.
 *   • Right: observations timeline — one row per observer decision.
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
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Ban,
  Info,
  RefreshCw,
  Plus,
  MessageSquare,
  Bot,
  ChevronDown,
  ChevronRight,
  HelpCircle,
} from 'lucide-react'
import {
  api,
  type InnerVoiceObservation,
  type InnerVoiceObservationTrigger,
  type InnerVoiceSession,
  type InnerVoiceState,
} from '../../api'
import ChatPanel from '../ChatPanel'

const DEFAULT_STATE: InnerVoiceState = {
  session_id: null,
  inner_voice_enabled: false,
  evaluate_user_turns: false,
  observations_count_by_action: {},
  last_observation_at: null,
}

// ─────────────────────────────────────────────────────────────────────────
// Action / trigger styling
// ─────────────────────────────────────────────────────────────────────────

const ACTION_STYLES: Record<string, { color: string; bg: string; border: string; label: string; Icon: typeof CheckCircle2 }> = {
  noop:                       { color: 'text-slate-400', bg: 'bg-slate-600/10',  border: 'border-slate-500/20', label: 'noop',         Icon: CheckCircle2 },
  inject:                     { color: 'text-amber-400', bg: 'bg-amber-600/10',  border: 'border-amber-500/30', label: 'inject',       Icon: Activity },
  cancel:                     { color: 'text-red-400',   bg: 'bg-red-600/10',    border: 'border-red-500/30',   label: 'cancel',       Icon: XCircle },
  ambient:                    { color: 'text-blue-400',  bg: 'bg-blue-600/10',   border: 'border-blue-500/30',  label: 'ambient',      Icon: Activity },
  clarify:                    { color: 'text-purple-400',bg: 'bg-purple-600/10', border: 'border-purple-500/30',label: 'clarify',      Icon: HelpCircle },
  deny_tool:                  { color: 'text-red-400',   bg: 'bg-red-700/15',    border: 'border-red-500/30',   label: 'deny',         Icon: Ban },
  allow:                      { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', label: 'allow',        Icon: CheckCircle2 },
  noop_budget_exhausted:      { color: 'text-amber-500', bg: 'bg-amber-600/5',   border: 'border-amber-500/20', label: 'noop (budget)',Icon: AlertTriangle },
  noop_empty_content:         { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', label: 'noop (empty)', Icon: Info },
  noop_no_ambient_channel:    { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', label: 'noop (no ch)', Icon: Info },
  noop_ambient_failed:        { color: 'text-amber-500', bg: 'bg-amber-600/5',   border: 'border-amber-500/20', label: 'noop (fail)',  Icon: AlertTriangle },
  noop_no_clarify_channel:    { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', label: 'noop (no ch)', Icon: Info },
  noop_clarify_failed:        { color: 'text-amber-500', bg: 'bg-amber-600/5',   border: 'border-amber-500/20', label: 'noop (fail)',  Icon: AlertTriangle },
  noop_inject_on_result:      { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', label: 'noop (late)',  Icon: Info },
  noop_cancel_on_result:      { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', label: 'noop (late)',  Icon: Info },
  noop_clarify_on_result:     { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', label: 'noop (late)',  Icon: Info },
  noop_cancel_with_pending_tools:        { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', label: 'noop (mid-tool)', Icon: AlertTriangle },
  noop_pretool_after_cancel:             { color: 'text-slate-500', bg: 'bg-slate-600/5', border: 'border-slate-500/15', label: 'noop (cancelled)', Icon: Info },
  acknowledge_complete:       { color: 'text-emerald-400', bg: 'bg-emerald-600/10', border: 'border-emerald-500/30', label: 'agree: complete', Icon: CheckCircle2 },
}

const TRIGGER_LABEL: Record<InnerVoiceObservationTrigger, string> = {
  assistant_message: 'iter end',
  tool_call:         'tool call',
  tool_result:       'tool result',
  result:            'turn end',
  pretool:           'pre-tool',
}

function actionStyle(action: string) {
  return ACTION_STYLES[action] ?? ACTION_STYLES.noop
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

  // Observations poll
  useEffect(() => {
    if (!selectedSession) { setObservations([]); return }
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.innerVoiceObservations(selectedSession, undefined, 200)
        if (!cancelled) setObservations(r.observations || [])
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
      <div className="flex items-center gap-3 px-6 py-3 border-b border-surface-3/30 flex-shrink-0">
        <BrainCircuit className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold text-slate-200">Inner Voice</h2>

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
                ? 'text-brand-400 bg-brand-500/15'
                : 'text-slate-400 hover:text-brand-400 hover:bg-brand-500/10'
            }`}
          >
            <Bot className="w-3.5 h-3.5" />
            Agent Details
          </button>
          <button
            onClick={handleCreateSession}
            disabled={creating}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-brand-600 border border-brand-500 text-white text-xs font-medium hover:bg-brand-500 transition disabled:opacity-50"
            title="Create a new Inner Voice chat session (observer fires on user turns)"
          >
            <Plus className="w-3.5 h-3.5" />
            {creating ? 'creating…' : 'new chat'}
          </button>
          <button
            onClick={() => setRefreshKey(k => k + 1)}
            className="p-1.5 rounded-md hover:bg-surface-2 text-slate-400 hover:text-slate-200 transition"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Split pane */}
      <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden">
        {/* Left: chat */}
        <div className="flex-1 min-h-0 border-b lg:border-b-0 lg:border-r border-surface-3/30 flex flex-col overflow-hidden bg-surface-0/50">
          {selectedSession ? (
            <ChatPanel
              key={selectedSession}
              requestedSessionKey={selectedSession}
              currentSessionKey={selectedSession}
              visible={true}
              showAgentDetails={showAgentDetails}
              onSessionLoaded={() => {}}
              onActiveSessionChange={() => {}}
            />
          ) : (
            <div className="flex-1 flex items-center justify-center p-8">
              <div className="text-center max-w-md">
                <MessageSquare className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                <div className="text-sm text-slate-400 mb-2">No Inner Voice session selected</div>
                <div className="text-xs text-slate-500 mb-4">
                  Click <span className="font-mono text-brand-400">+ new chat</span> above
                  to start a session where the observer watches every chat turn,
                  or pick an existing session from the dropdown.
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right: observation panel */}
        <div className="w-full lg:w-[420px] xl:w-[480px] flex-shrink-0 flex flex-col min-h-0 overflow-hidden bg-surface-1/30">
          <ObservationPanel
            obsState={obsState}
            observations={observations}
            selectedSession={selectedSession}
          />
        </div>
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
      <span className="text-xs text-slate-500">session:</span>
      <select
        value={selectedSession ?? ''}
        onChange={e => onSelect(e.target.value || null)}
        className="bg-surface-2 border border-surface-3/40 rounded-md px-2 py-1 text-xs font-mono text-slate-200 max-w-[280px] focus:outline-none focus:border-brand-500/50"
      >
        {sessions.length === 0 && <option value="">— none —</option>}
        {sessions.map(s => (
          <option key={s.session_id} value={s.session_id}>
            {s.session_id}
            {s.evaluate_user_turns ? ' [chat]' : ''}
            {s.experiment_id ? ` · ${s.experiment_id}` : ''}
            {s.message_count ? ` · ${s.message_count}msg` : ''}
          </option>
        ))}
      </select>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// Right-side observation panel
// ─────────────────────────────────────────────────────────────────────────

function ObservationPanel({
  obsState,
  observations,
  selectedSession,
}: {
  obsState: InnerVoiceState
  observations: InnerVoiceObservation[]
  selectedSession: string | null
}) {
  // Pick out interesting count buckets for the header
  const counts = obsState.observations_count_by_action || {}
  const totalRows = Object.values(counts).reduce((a, b) => a + b, 0)
  const noopCount = (counts.noop || 0)
    + (counts.allow || 0)
    + (counts.noop_budget_exhausted || 0)
    + (counts.noop_empty_content || 0)
    + (counts.noop_no_ambient_channel || 0)
    + (counts.noop_ambient_failed || 0)
  const interventionCount = (counts.inject || 0) + (counts.cancel || 0) + (counts.ambient || 0) + (counts.clarify || 0) + (counts.deny_tool || 0)

  // Group observations by turn for compact display
  const grouped = useMemo(() => {
    const map = new Map<string, InnerVoiceObservation[]>()
    for (const o of observations) {
      const list = map.get(o.turn_id) || []
      list.push(o)
      map.set(o.turn_id, list)
    }
    return Array.from(map.entries()).map(([turn_id, rows]) => ({
      turn_id,
      rows: rows.sort((a, b) => b.sequence_in_turn - a.sequence_in_turn),
    }))
  }, [observations])

  return (
    <>
      {/* State header */}
      <div className="flex flex-col gap-2 px-4 py-3 border-b border-surface-3/30 flex-shrink-0">
        <div className="flex items-center gap-2 text-xs">
          <Activity className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-300">observer</span>
          <span className={`font-mono ${obsState.inner_voice_enabled ? 'text-brand-400' : 'text-slate-500'}`}>
            {obsState.inner_voice_enabled ? 'enabled' : 'disabled'}
          </span>
          {obsState.evaluate_user_turns && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-brand-500/10 text-brand-400 border border-brand-500/20">
              chat
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2 text-[11px] mt-1">
          <Stat label="rows" value={String(totalRows)} />
          <Stat
            label="interventions"
            value={String(interventionCount)}
            warn={interventionCount > 0}
          />
          <Stat label="noops" value={String(noopCount)} />
        </div>

        {/* Per-action breakdown chips */}
        {totalRows > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
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
          <div className="text-[10px] text-slate-500 font-mono">
            last observation: {obsState.last_observation_at}
          </div>
        )}
      </div>

      {/* Observation timeline */}
      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 px-1 mb-1">
          observations ({observations.length})
        </div>
        {!selectedSession && (
          <div className="text-xs text-slate-500 italic px-2 py-3">
            No session selected. Pick one above or create a new chat.
          </div>
        )}
        {selectedSession && observations.length === 0 && (
          <div className="text-xs text-slate-500 italic px-2 py-3">
            No observations yet. Send a chat message — the observer fires
            on every iteration boundary, tool result, and at turn end.
            A few seconds after the agent responds you'll see noop rows
            here (proving it watched and chose not to act).
          </div>
        )}
        {grouped.map(g => (
          <TurnGroup key={g.turn_id} turnId={g.turn_id} rows={g.rows} />
        ))}
      </div>
    </>
  )
}

function Stat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className={`px-2 py-1 rounded bg-surface-2/60 border border-surface-3/30 ${warn ? 'border-amber-500/30' : ''}`}>
      <div className="text-[9px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`font-mono text-xs ${warn ? 'text-amber-400' : 'text-slate-300'}`}>{value}</div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// Per-turn group
// ─────────────────────────────────────────────────────────────────────────

function TurnGroup({ turnId, rows }: { turnId: string; rows: InnerVoiceObservation[] }) {
  const [open, setOpen] = useState(true)
  const interventions = rows.filter(r =>
    r.action === 'inject' || r.action === 'cancel' || r.action === 'ambient' || r.action === 'clarify' || r.action === 'deny_tool'
  ).length
  return (
    <div className="rounded-md bg-surface-2/40 border border-surface-3/30">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-2/60 transition rounded-t-md"
      >
        {open ? <ChevronDown className="w-3.5 h-3.5 text-slate-500" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />}
        <span className="font-mono text-[11px] text-slate-400">{turnId}</span>
        <span className="ml-auto text-[10px] text-slate-500">
          {rows.length} obs
          {interventions > 0 && <span className="ml-2 text-amber-400">{interventions} intervened</span>}
        </span>
      </button>
      {open && (
        <div className="px-2 pb-2 space-y-1">
          {rows.map(r => <ObservationRow key={r.id} obs={r} />)}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// Single observation row
// ─────────────────────────────────────────────────────────────────────────

function ObservationRow({ obs }: { obs: InnerVoiceObservation }) {
  const [expanded, setExpanded] = useState(false)
  const s = actionStyle(obs.action)
  const Icon = s.Icon
  const hasContent = !!(obs.content && obs.content.trim())
  const triggerLabel = TRIGGER_LABEL[obs.trigger as InnerVoiceObservationTrigger] || obs.trigger

  return (
    <div className={`rounded border ${s.border} ${s.bg} text-[11px]`}>
      <button
        onClick={() => hasContent && setExpanded(!expanded)}
        className={`w-full flex items-center gap-2 px-2 py-1.5 text-left ${hasContent ? 'cursor-pointer hover:opacity-90' : 'cursor-default'}`}
      >
        <Icon className={`w-3.5 h-3.5 flex-shrink-0 ${s.color}`} />
        <span className={`font-mono font-semibold ${s.color}`}>{s.label}</span>
        <span className="text-slate-500">·</span>
        <span className="text-slate-400">{triggerLabel}</span>
        {obs.related_tool && (
          <>
            <span className="text-slate-500">·</span>
            <span className="font-mono text-slate-400">{obs.related_tool}</span>
          </>
        )}
        {obs.reason && (
          <span className="text-slate-300 truncate ml-1">{obs.reason}</span>
        )}
        <span className="ml-auto text-[10px] text-slate-500 font-mono flex-shrink-0">
          #{obs.sequence_in_turn}
          {obs.latency_ms != null && <span className="ml-1">· {obs.latency_ms}ms</span>}
        </span>
      </button>
      {expanded && hasContent && (
        <div className="px-2 pb-2 pt-0">
          <div className="mt-1 px-2 py-1.5 rounded bg-surface-1/80 border border-surface-3/30 text-slate-300 whitespace-pre-wrap font-mono text-[11px]">
            {obs.content}
          </div>
          {obs.error && (
            <div className="mt-1 text-[10px] text-red-400 font-mono">error: {obs.error}</div>
          )}
        </div>
      )}
    </div>
  )
}
