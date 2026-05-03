/**
 * Inner Voice (#345) — Stage 5 chat-driven UI.
 *
 * Two-pane layout:
 *   • Left:  ChatPanel (reused) bound to the selected Inner Voice session.
 *   • Right: observation panel — state header, critique cards, grading
 *     progress, recent interventions, click-to-detail.
 *
 * The "+ New IV session" button calls POST /api/sessions/create with
 * `inner_voice: true` AND `inner_voice_evaluate_user_turns: true` so
 * Brain 2 fires on user-typed chat messages, not just ambient turns.
 *
 * Polling sources of truth:
 *   /api/inner_voice/state            → header status (every 4s)
 *   /api/inner_voice/critiques        → annotation cards (every 3s)
 *   /api/inner_voice/interventions    → bottom strip (every 5s)
 *   /api/inner_voice/grading_summary  → addressed_rate (every 8s)
 *
 * SSE delivery is fire-and-forget post-loop; polling is the source of
 * truth. Future: long-lived inner-voice SSE channel.
 */

import { useCallback, useEffect, useState } from 'react'
import {
  Ear,
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  XCircle,
  X,
  RefreshCw,
  Plus,
  MessageSquare,
  Bot,
} from 'lucide-react'
import {
  api,
  type InnerVoiceCritique,
  type InnerVoiceIntervention,
  type InnerVoiceSession,
  type InnerVoiceState,
  type InnerVoiceGradingSummary,
} from '../../api'
import ChatPanel from '../ChatPanel'

const DEFAULT_STATE: InnerVoiceState = {
  session_id: null,
  state: 'idle',
  active_ensemble: null,
  personas: [],
  nudge_count: 0,
  stage: '0',
}

type ActionKind = 'steer' | 'interrupt' | 'continue' | 'agree' | 'log_only' | 'escalate'

const ACTION_STYLES: Record<ActionKind, { color: string; bg: string; border: string; label: string }> = {
  steer:     { color: 'text-amber-400', bg: 'bg-amber-600/10',  border: 'border-amber-500/20', label: 'steer' },
  interrupt: { color: 'text-red-400',   bg: 'bg-red-600/10',    border: 'border-red-500/20',   label: 'interrupt' },
  continue:  { color: 'text-blue-400',  bg: 'bg-blue-600/10',   border: 'border-blue-500/20',  label: 'continue' },
  agree:     { color: 'text-green-400', bg: 'bg-green-600/10',  border: 'border-green-500/20', label: 'agree' },
  log_only:  { color: 'text-slate-400', bg: 'bg-slate-600/10',  border: 'border-slate-500/20', label: 'log only' },
  escalate:  { color: 'text-red-400',   bg: 'bg-red-700/15',    border: 'border-red-500/30',   label: 'escalate' },
}

function actionFromCritique(c: InnerVoiceCritique): ActionKind {
  if (c.action_taken && c.action_taken !== 'log_only') {
    if (c.action_taken === 'agreement') return 'agree'
    return c.action_taken as ActionKind
  }
  if (!c.disagrees) return 'agree'
  return 'log_only'
}

// ─────────────────────────────────────────────────────────────────────────
// Top-level page
// ─────────────────────────────────────────────────────────────────────────

export default function InnerVoicePage() {
  const [obsState, setObsState] = useState<InnerVoiceState>(DEFAULT_STATE)
  const [sessions, setSessions] = useState<InnerVoiceSession[]>([])
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [critiques, setCritiques] = useState<InnerVoiceCritique[]>([])
  const [interventions, setInterventions] = useState<InnerVoiceIntervention[]>([])
  const [gradingSummary, setGradingSummary] = useState<InnerVoiceGradingSummary | null>(null)
  const [detail, setDetail] = useState<DetailState | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const [showAgentDetails, setShowAgentDetails] = useState(false)

  // ── Sessions list ──
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

  // ── State header poll ──
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

  // ── Critiques poll ──
  useEffect(() => {
    if (!selectedSession) { setCritiques([]); return }
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.innerVoiceCritiques(selectedSession)
        if (!cancelled) setCritiques(r.critiques || [])
      } catch { /* best-effort */ }
    }
    poll()
    const t = setInterval(poll, 3000)
    return () => { cancelled = true; clearInterval(t) }
  }, [selectedSession, refreshKey])

  // ── Interventions poll ──
  useEffect(() => {
    if (!selectedSession) { setInterventions([]); return }
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.innerVoiceInterventions(selectedSession)
        if (!cancelled) setInterventions(r.interventions || [])
      } catch { /* best-effort */ }
    }
    poll()
    const t = setInterval(poll, 5000)
    return () => { cancelled = true; clearInterval(t) }
  }, [selectedSession, refreshKey])

  // ── Grading summary poll ──
  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.innerVoiceGradingSummary(selectedSession || undefined, 168)
        if (!cancelled) setGradingSummary(r)
      } catch { /* best-effort */ }
    }
    poll()
    const t = setInterval(poll, 8000)
    return () => { cancelled = true; clearInterval(t) }
  }, [selectedSession, refreshKey])

  // ── Create new IV session ──
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

  // ── Render ──
  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* ── Header bar ── */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-surface-3/30 flex-shrink-0">
        <Ear className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold text-slate-200">Inner Voice</h2>
        <span className="text-xs text-slate-500 font-mono">stage {obsState.stage}</span>

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
            title="Create a new Inner Voice chat session (Brain 2 fires on user turns)"
          >
            <Plus className="w-3.5 h-3.5" />
            {creating ? 'creating…' : 'new IV chat'}
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

      {/* ── Split pane ── */}
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
                  Click <span className="font-mono text-brand-400">+ new IV chat</span> above
                  to start a session where Brain 2 fires on every chat turn,
                  or pick an existing IV session from the dropdown.
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right: observation panel */}
        <div className="w-full lg:w-[420px] xl:w-[480px] flex-shrink-0 flex flex-col min-h-0 overflow-hidden bg-surface-1/30">
          <ObservationPanel
            obsState={obsState}
            critiques={critiques}
            interventions={interventions}
            gradingSummary={gradingSummary}
            selectedSession={selectedSession}
            onCardClick={(c) => openDetail(c, selectedSession!, setDetail)}
          />
        </div>
      </div>

      {/* Detail panel modal */}
      {detail && <DetailPanel detail={detail} onClose={() => setDetail(null)} />}
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
  critiques,
  interventions,
  gradingSummary,
  selectedSession,
  onCardClick,
}: {
  obsState: InnerVoiceState
  critiques: InnerVoiceCritique[]
  interventions: InnerVoiceIntervention[]
  gradingSummary: InnerVoiceGradingSummary | null
  selectedSession: string | null
  onCardClick: (c: InnerVoiceCritique) => void
}) {
  const ensembleLabel = obsState.active_ensemble ?? 'autonomy_default'
  const personaChips = obsState.personas.length ? obsState.personas : ['completion_checker']
  const observingDot = obsState.state !== 'idle'
  const dotColor =
    obsState.state === 'intervening' ? 'bg-red-400' :
    obsState.state === 'critiquing'  ? 'bg-amber-400' :
    observingDot                     ? 'bg-green-400' :
    'bg-slate-600'
  const maxNudges = obsState.max_nudges_per_session ?? 2
  const gp = obsState.grading_progress

  return (
    <>
      {/* ── State header ── */}
      <div className="flex flex-col gap-2 px-4 py-3 border-b border-surface-3/30 flex-shrink-0">
        <div className="flex items-center gap-2 text-xs">
          <Activity className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-300">ensemble</span>
          <span className="font-mono text-brand-400">{ensembleLabel}</span>
          <span className="ml-auto flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${dotColor} ${observingDot ? 'animate-pulse' : ''}`} />
            <span className="text-slate-400">{obsState.state}</span>
          </span>
        </div>

        <div className="flex flex-wrap gap-1">
          {personaChips.map(p => (
            <span
              key={p}
              className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-surface-2 text-slate-400 border border-surface-3/30"
              title={`persona: ${p}`}
            >
              {p}
            </span>
          ))}
        </div>

        <div className="grid grid-cols-3 gap-2 text-[11px] mt-1">
          <Stat label="nudges" value={`${obsState.nudge_count}/${maxNudges}`} />
          <Stat
            label="vetoes"
            value={String(obsState.consecutive_vetoes ?? 0)}
            warn={(obsState.consecutive_vetoes ?? 0) >= 2}
          />
          <Stat
            label="esc"
            value={String(obsState.escalations_count ?? 0)}
            warn={(obsState.escalations_count ?? 0) > 0}
          />
        </div>

        {/* Stage 5 grading progress */}
        {gp && (gp.graded > 0 || gp.ungraded > 0) && (
          <div className="mt-1 px-2 py-1.5 rounded-md bg-surface-2/60 border border-surface-3/30">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">
              grading (this session)
            </div>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="text-slate-300">{gp.graded} graded</span>
              <span className="text-slate-500">/ {gp.ungraded} pending</span>
              <span className="ml-auto flex items-center gap-2">
                <span className="text-green-400" title="addressed=true">
                  ✓ {gp.addressed_true}
                </span>
                <span className="text-red-400" title="addressed=false">
                  ✗ {gp.addressed_false}
                </span>
                <span className="text-slate-500" title="ambiguous">
                  ? {gp.addressed_null}
                </span>
              </span>
            </div>
          </div>
        )}

        {/* Aggregate grading summary (last 7d, all sessions) */}
        {gradingSummary && gradingSummary.total_interventions > 0 && (
          <div className="text-[10px] text-slate-500 font-mono">
            7d global: {gradingSummary.graded}/{gradingSummary.total_interventions} graded
            {' · '}
            addressed_rate {(gradingSummary.addressed_rate * 100).toFixed(0)}%
          </div>
        )}
      </div>

      {/* ── Critique cards ── */}
      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 px-1 mb-1">
          critiques ({critiques.length})
        </div>
        {!selectedSession && (
          <div className="text-xs text-slate-500 italic px-2 py-3">
            No session selected. Pick one above or create a new IV chat.
          </div>
        )}
        {selectedSession && critiques.length === 0 && (
          <div className="text-xs text-slate-500 italic px-2 py-3">
            No critiques yet. Send a chat message — Brain 2 fires post-loop
            and lands a card here a few seconds after Brain 1's response
            finishes streaming.
          </div>
        )}
        {selectedSession && critiques.map(c => (
          <CritiqueCard key={c.id} critique={c} onClick={() => onCardClick(c)} />
        ))}
      </div>

      {/* ── Interventions strip ── */}
      <div className="px-4 py-2 border-t border-surface-3/30 flex-shrink-0 bg-surface-1/50">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
          interventions ({interventions.length})
        </div>
        {interventions.length === 0 ? (
          <div className="text-[11px] text-slate-500 italic">
            none yet — interventions fire when Brain 2 ensembles veto a SIGNAL:TASK_COMPLETE (ambient turns only)
          </div>
        ) : (
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {interventions.slice(0, 5).map(iv => (
              <InterventionRow key={iv.id} intervention={iv} />
            ))}
          </div>
        )}
        {obsState.last_critique_at && (
          <div className="text-[10px] text-slate-500 font-mono mt-1">
            last critique: {obsState.last_critique_at}
          </div>
        )}
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

function InterventionRow({ intervention }: { intervention: InnerVoiceIntervention }) {
  const kindColor = {
    continue:  'text-blue-400',
    steer:     'text-amber-400',
    interrupt: 'text-red-400',
    escalate:  'text-red-400',
  }[intervention.kind] ?? 'text-slate-400'

  let outcomeIcon = '○'
  let outcomeColor = 'text-slate-500'
  if (intervention.outcome_addressed === true) {
    outcomeIcon = '✓'
    outcomeColor = 'text-green-400'
  } else if (intervention.outcome_addressed === false) {
    outcomeIcon = '✗'
    outcomeColor = 'text-red-400'
  } else if (intervention.outcome_turn_id) {
    outcomeIcon = '?'
    outcomeColor = 'text-slate-400'
  }

  return (
    <div className="flex items-center gap-2 text-[11px] font-mono">
      <span className={kindColor}>{intervention.kind}</span>
      <span className={`${outcomeColor} text-sm leading-none`} title={
        intervention.outcome_addressed === true ? 'addressed' :
        intervention.outcome_addressed === false ? 'NOT addressed' :
        intervention.outcome_turn_id ? 'ambiguous' : 'awaiting outcome turn'
      }>
        {outcomeIcon}
      </span>
      <span className="text-slate-500 truncate flex-1" title={intervention.outcome_summary || ''}>
        {intervention.outcome_summary || '(no summary yet)'}
      </span>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// Critique card
// ─────────────────────────────────────────────────────────────────────────

function CritiqueCard({
  critique,
  onClick,
}: {
  critique: InnerVoiceCritique
  onClick: () => void
}) {
  const action = actionFromCritique(critique)
  const s = ACTION_STYLES[action]
  const sev = critique.severity ?? 0
  const Icon = action === 'agree' ? CheckCircle2
             : action === 'interrupt' || action === 'escalate' ? XCircle
             : action === 'log_only' ? AlertTriangle
             : ArrowRight

  return (
    <button
      onClick={onClick}
      className={`block w-full text-left rounded-lg px-3 py-2 text-sm border ${s.bg} ${s.border} hover:bg-opacity-20 transition`}
    >
      <div className={`flex items-center gap-1.5 ${s.color} text-xs font-medium mb-0.5`}>
        <Icon className="w-3 h-3" />
        <span className="font-mono truncate">{critique.persona}</span>
        <span className="text-slate-400">{sev.toFixed(2)}</span>
        <span className="text-slate-500 ml-auto text-[10px]">
          {critique.turn_id.slice(0, 6)}
        </span>
      </div>
      <div className="text-slate-300 text-[11px] italic line-clamp-2">
        "{critique.reason || '(no reason)'}"
      </div>
      <div className="flex items-center gap-2 mt-1 text-[10px] text-slate-500 font-mono">
        <span>{s.label}</span>
        {critique.latency_ms != null && <span>{critique.latency_ms}ms</span>}
        {critique.parse_attempts > 1 && (
          <span className="text-amber-400">retried {critique.parse_attempts - 1}x</span>
        )}
        <span className="ml-auto text-brand-400 text-[10px]">→ detail</span>
      </div>
    </button>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// Detail panel (full prompt + raw response)
// ─────────────────────────────────────────────────────────────────────────

interface DetailState {
  critique: InnerVoiceCritique
  sessionId: string
  loading: boolean
  systemPrompt: string | null
  userPrompt: string | null
  rawResponse: string | null
  error: string | null
}

type SetDetailFn = React.Dispatch<React.SetStateAction<DetailState | null>>

function openDetail(
  critique: InnerVoiceCritique,
  sessionId: string,
  setDetail: SetDetailFn,
) {
  setDetail({
    critique,
    sessionId,
    loading: true,
    systemPrompt: null,
    userPrompt: null,
    rawResponse: null,
    error: null,
  })

  ;(async () => {
    try {
      const offset = critique.event_log_offset ?? 0
      const r = await api.innerVoiceEventLog(sessionId, offset, 3, true)
      let systemPrompt: string | null = null
      let userPrompt: string | null = null
      let rawResponse: string | null = null
      for (const ev of r.events) {
        if (ev.event === 'inner_voice.persona_invoked') {
          const d = ev.data as { system_prompt?: string; user_prompt?: string }
          systemPrompt = d.system_prompt ?? null
          userPrompt = d.user_prompt ?? null
        } else if (ev.event === 'inner_voice.persona_response_raw') {
          const d = ev.data as { raw?: string }
          rawResponse = d.raw ?? null
        }
      }
      if (!rawResponse && critique.raw_response_offset != null) {
        const r2 = await api.innerVoiceEventLog(sessionId, critique.raw_response_offset, 1, true)
        const ev = r2.events[0]
        if (ev && ev.event === 'inner_voice.persona_response_raw') {
          const d = ev.data as { raw?: string }
          rawResponse = d.raw ?? null
        }
      }
      setDetail({
        critique,
        sessionId,
        loading: false,
        systemPrompt,
        userPrompt,
        rawResponse,
        error: null,
      })
    } catch (err) {
      setDetail({
        critique,
        sessionId,
        loading: false,
        systemPrompt: null,
        userPrompt: null,
        rawResponse: null,
        error: err instanceof Error ? err.message : 'load failed',
      })
    }
  })()
}

function DetailPanel({
  detail,
  onClose,
}: {
  detail: NonNullable<DetailState>
  onClose: () => void
}) {
  const { critique, loading, systemPrompt, userPrompt, rawResponse, error } = detail
  const sev = critique.severity ?? 0

  return (
    <div className="fixed inset-0 z-50 flex items-stretch justify-end bg-black/60" onClick={onClose}>
      <div
        className="w-full max-w-3xl bg-surface-1 border-l border-surface-3/50 flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-5 py-3 border-b border-surface-3/40">
          <div>
            <div className="text-sm font-semibold text-slate-200">
              <span className="font-mono text-brand-400">{critique.persona}</span>
              {critique.persona_version && (
                <span className="text-xs text-slate-500 font-mono ml-2">{critique.persona_version}</span>
              )}
            </div>
            <div className="text-xs text-slate-500 font-mono">
              turn {critique.turn_id} · severity {sev.toFixed(2)} ·{' '}
              {critique.disagrees ? 'disagrees' : 'agrees'} ·{' '}
              {critique.latency_ms ?? '?'}ms
            </div>
          </div>
          <button
            onClick={onClose}
            className="ml-auto p-1.5 rounded-md hover:bg-surface-2 text-slate-400 hover:text-slate-200 transition"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5 text-sm">
          <DetailRow label="Reason">
            <span className="italic text-slate-300">"{critique.reason || '(none)'}"</span>
          </DetailRow>
          <DetailRow label="Suggested action">
            <span className="font-mono">{critique.suggested_action || '(none)'}</span>
            <span className="text-xs text-slate-500 ml-2">
              taken: <span className="font-mono">{critique.action_taken || 'log_only'}</span>
            </span>
          </DetailRow>
          <DetailRow label="Forensic">
            <span className="text-xs text-slate-500 font-mono">
              event_log_offset {critique.event_log_offset ?? '?'} ·
              raw_offset {critique.raw_response_offset ?? '?'} ·
              parse_attempts {critique.parse_attempts}
              {critique.prompt_hash && (
                <> · hash {critique.prompt_hash.slice(0, 16)}…</>
              )}
            </span>
          </DetailRow>

          {loading && (
            <div className="text-xs text-slate-500 italic">loading prompt + raw from event log…</div>
          )}
          {error && (
            <div className="text-xs text-red-400 font-mono">load error: {error}</div>
          )}

          {systemPrompt && (
            <DetailRow label="Persona system prompt">
              <pre className="text-xs font-mono whitespace-pre-wrap bg-surface-2 border border-surface-3/40 rounded-md p-3 max-h-72 overflow-y-auto text-slate-300">
                {systemPrompt}
              </pre>
            </DetailRow>
          )}
          {userPrompt && (
            <DetailRow label="User prompt (assembled context)">
              <pre className="text-xs font-mono whitespace-pre-wrap bg-surface-2 border border-surface-3/40 rounded-md p-3 max-h-72 overflow-y-auto text-slate-300">
                {userPrompt}
              </pre>
            </DetailRow>
          )}
          {rawResponse && (
            <DetailRow label="Brain 2 raw response">
              <pre className="text-xs font-mono whitespace-pre-wrap bg-surface-2 border border-surface-3/40 rounded-md p-3 max-h-72 overflow-y-auto text-slate-300">
                {rawResponse}
              </pre>
            </DetailRow>
          )}
          {critique.anchor_response_excerpt && (
            <DetailRow label="Brain 1 response excerpt (first 500 chars)">
              <pre className="text-xs font-mono whitespace-pre-wrap bg-surface-2 border border-surface-3/40 rounded-md p-3 max-h-48 overflow-y-auto text-slate-300">
                {critique.anchor_response_excerpt}
              </pre>
            </DetailRow>
          )}
        </div>
      </div>
    </div>
  )
}

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-slate-500 mb-1.5">{label}</div>
      {children}
    </div>
  )
}
