/**
 * Inner Voice (#345) — Stage 2 live UI.
 *
 * Sibling tab to Chat. Shows Brain 2 critiques landing on Inner Voice
 * sessions. Stage 2 ships annotation cards from real `inner_voice_critiques`
 * rows + click-to-detail that loads the full Brain 2 prompt + raw response
 * from the per-session event log.
 *
 * Stage 2 wire shape:
 *   /api/inner_voice/sessions      → list of opted-in sessions
 *   /api/inner_voice/state         → header status (state, ensemble, nudges)
 *   /api/inner_voice/critiques     → annotation cards
 *   /api/inner_voice/event_log     → click-to-detail (raw + prompt)
 *
 * SSE delivery is treated as a fast-path hint only — the persona response
 * arrives a few seconds AFTER the SSE stream closes (Brain 2 is fire-and-
 * forget post-loop), so polling /critiques is the source of truth. Future
 * stages may add a long-lived inner-voice SSE channel that doesn't piggy-
 * back on the per-turn stream.
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
} from 'lucide-react'
import { api, type InnerVoiceCritique, type InnerVoiceSession, type InnerVoiceState } from '../../api'

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
  // Stage 2 always logs-only. Promote when later stages send a real action.
  if (c.action_taken && c.action_taken !== 'log_only') {
    if (c.action_taken === 'agreement') return 'agree'
    return c.action_taken as ActionKind
  }
  if (!c.disagrees) return 'agree'
  return 'log_only'
}

export default function InnerVoicePage() {
  const [obsState, setObsState] = useState<InnerVoiceState>(DEFAULT_STATE)
  const [sessions, setSessions] = useState<InnerVoiceSession[]>([])
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [critiques, setCritiques] = useState<InnerVoiceCritique[]>([])
  const [detail, setDetail] = useState<DetailState | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  // ── Session list ──
  const loadSessions = useCallback(async () => {
    try {
      const r = await api.innerVoiceSessions(50)
      setSessions(r.sessions || [])
      // Auto-select most recent if nothing chosen yet.
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
    const t = setInterval(poll, 5000)
    return () => { cancelled = true; clearInterval(t) }
  }, [selectedSession])

  // ── Critiques poll for selected session ──
  useEffect(() => {
    if (!selectedSession) { setCritiques([]); return }
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.innerVoiceCritiques(selectedSession)
        if (!cancelled) setCritiques(r.critiques || [])
      } catch {
        // best-effort
      }
    }
    poll()
    const t = setInterval(poll, 4000)
    return () => { cancelled = true; clearInterval(t) }
  }, [selectedSession, refreshKey])

  const ensembleLabel = obsState.active_ensemble ?? 'autonomy_default'
  const personaChips = obsState.personas.length ? obsState.personas : ['completion_checker']
  const observingDot = obsState.state !== 'idle'
  const dotColor =
    obsState.state === 'intervening' ? 'bg-red-400' :
    obsState.state === 'critiquing'  ? 'bg-amber-400' :
    observingDot                     ? 'bg-green-400' :
    'bg-slate-600'
  const maxNudges = obsState.max_nudges_per_session ?? 2

  // ── Render ──
  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden p-6 gap-4">
      {/* ── Header ── */}
      <div className="flex items-center gap-3">
        <Ear className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold text-slate-200">Inner Voice</h2>
        <span className="text-xs text-slate-500 font-mono">stage {obsState.stage}</span>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setRefreshKey(k => k + 1)}
            className="p-1.5 rounded-md hover:bg-surface-2 text-slate-400 hover:text-slate-200 transition"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ── Session picker ── */}
      <div className="flex items-center gap-3 px-4 py-2.5 bg-surface-1 border border-surface-3/40 rounded-xl text-sm">
        <span className="text-slate-400 text-xs">session:</span>
        <select
          value={selectedSession ?? ''}
          onChange={e => setSelectedSession(e.target.value || null)}
          className="bg-surface-2 border border-surface-3/40 rounded-md px-2 py-1 text-xs font-mono text-slate-200 flex-1 min-w-0 max-w-md focus:outline-none focus:border-brand-500/50"
        >
          {sessions.length === 0 && <option value="">— no Inner Voice sessions yet —</option>}
          {sessions.map(s => (
            <option key={s.session_id} value={s.session_id}>
              {s.session_id} {s.experiment_id ? `· ${s.experiment_id}` : ''}{s.title ? ` · ${s.title.slice(0, 30)}` : ''}
            </option>
          ))}
        </select>
        <span className="text-xs text-slate-500">
          {critiques.length} critique{critiques.length === 1 ? '' : 's'}
        </span>
      </div>

      {/* ── Status bar ── */}
      <div className="flex items-center gap-3 px-4 py-3 bg-surface-1 border border-surface-3/40 rounded-xl text-sm">
        <Activity className="w-4 h-4 text-slate-400" />
        <span className="text-slate-300">ensemble:</span>
        <span className="font-mono text-brand-400">{ensembleLabel}</span>
        <div className="flex gap-1.5 ml-2">
          {personaChips.map(p => (
            <span
              key={p}
              className="px-2 py-0.5 rounded-md text-[11px] font-mono bg-surface-2 text-slate-400 border border-surface-3/30"
              title={`persona: ${p}`}
            >
              {p}
            </span>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-3 text-xs text-slate-400">
          <span>nudges: {obsState.nudge_count}/{maxNudges}</span>
          <span className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${dotColor} ${observingDot ? 'animate-pulse' : ''}`} />
            {obsState.state}
          </span>
        </div>
      </div>

      {/* ── Spine: live critiques for selected session ── */}
      <div className="flex-1 min-h-0 overflow-y-auto bg-surface-1 border border-surface-3/40 rounded-xl p-4 space-y-3">
        {!selectedSession && (
          <div className="text-sm text-slate-500 italic px-2 py-3">
            Pick an Inner Voice session above. To create one, open a Chat session,
            then PATCH <code className="font-mono text-slate-400">/api/sessions/&lt;id&gt;</code>{' '}
            with <code className="font-mono text-slate-400">{`{"inner_voice":true}`}</code>.
          </div>
        )}
        {selectedSession && critiques.length === 0 && (
          <div className="text-sm text-slate-500 italic px-2 py-3">
            No critiques landed yet for this session. Brain 2 fires post-loop on
            ambient turns of opted-in sessions — give it an autonomy task or an
            ambient inject and refresh.
          </div>
        )}
        {selectedSession && critiques.map(c => (
          <CritiqueCard key={c.id} critique={c} onClick={() => openDetail(c, selectedSession, setDetail)} />
        ))}
      </div>

      {/* ── Footer: intervention log strip (Stage 2 has none yet) ── */}
      <div className="flex items-center gap-2 px-4 py-2 bg-surface-1 border border-surface-3/40 rounded-xl text-xs flex-shrink-0">
        <span className="text-slate-500 font-medium">Intervention log:</span>
        <span className="text-slate-500 italic">
          Stage 2 is observation-only — interventions land in Stage 3+.
        </span>
        {obsState.last_critique_at && (
          <span className="ml-auto text-slate-500">
            last critique: <span className="font-mono">{obsState.last_critique_at}</span>
          </span>
        )}
      </div>

      {/* ── Detail panel (full prompt + raw response) ── */}
      {detail && <DetailPanel detail={detail} onClose={() => setDetail(null)} />}
    </div>
  )
}

// ── Critique card ────────────────────────────────────────────────────────

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
      className={`block w-full text-left rounded-xl px-3.5 py-2.5 text-sm border ${s.bg} ${s.border} hover:bg-opacity-20 transition`}
    >
      <div className={`flex items-center gap-1.5 ${s.color} text-xs font-medium mb-1`}>
        <Icon className="w-3.5 h-3.5" />
        <span className="font-mono">{critique.persona}</span>
        {critique.persona_version && (
          <span className="text-[10px] text-slate-500 font-mono">{critique.persona_version}</span>
        )}
        <span className="text-slate-400">severity {sev.toFixed(2)}</span>
        <span className="text-slate-500 ml-auto">
          turn {critique.turn_id.slice(0, 8)}…
        </span>
      </div>
      <div className="text-slate-300 text-xs italic">"{critique.reason || '(no reason given)'}"</div>
      {critique.anchor_response_excerpt && (
        <div className="text-[11px] text-slate-500 mt-1 font-mono truncate">
          ↳ {critique.anchor_response_excerpt.slice(0, 200)}{critique.anchor_response_excerpt.length > 200 ? '…' : ''}
        </div>
      )}
      <div className="flex items-center gap-3 mt-1.5 text-[11px] text-slate-500 font-mono">
        <span>{s.label}</span>
        {critique.latency_ms != null && <span>{critique.latency_ms}ms</span>}
        {critique.input_tokens != null && critique.output_tokens != null && (
          <span>{critique.input_tokens}/{critique.output_tokens} tok</span>
        )}
        {critique.parse_attempts > 1 && (
          <span className="text-amber-400">retried {critique.parse_attempts - 1}x</span>
        )}
        <span className="ml-auto text-brand-400">click for detail →</span>
      </div>
    </button>
  )
}

// ── Detail panel ─────────────────────────────────────────────────────────

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
  // Initial state shows whatever we already have (anchor_excerpt, etc.) and
  // kicks off a load for the full prompt + raw via the event log.
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
      // Fetch the persona_invoked event AND persona_response_raw event by
      // their offsets. We use expand_blobs=true so the contents come back
      // inline — the events reference content-addressed blobs by SHA.
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
      // Fallback: if raw_response_offset points elsewhere, fetch it too.
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
