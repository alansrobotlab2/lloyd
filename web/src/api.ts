/** One file a turn wrote, as recorded by the aggregator's change ledger. */
export interface ChangedFile {
  path: string
  op: 'edit' | 'write' | 'create' | string
  /** Epoch seconds when this file was put back, or null. */
  reverted_at: number | null
}

export interface FilesChanged {
  turn_id: string
  files: ChangedFile[]
}

export interface RevertResult {
  path: string
  op: string
  status: 'restored' | 'deleted' | 'refused' | 'skipped' | string
  reason?: string
}

export interface TurnStats {
  input_tokens: number
  output_tokens: number
  cache_create: number
  cache_read: number
  cost_usd: number | null
  duration_ms: number | null
  num_turns: number | null
  model: string
  peak_input_tokens?: number
  // Tool-result messages reuse this `stats` slot with a different shape:
  // `{ result_chars, is_error }`. Keeping these optional avoids a union split.
  is_error?: boolean
  result_chars?: number
  /** Present when this turn wrote files. Persisted, so a reload keeps it. */
  files_changed?: FilesChanged
}

export interface QueueState {
  current: {
    turn_id: string
    source: 'user' | 'ambient' | 'system'
    started_at: string | null
  } | null
  pending_user: number
  pending_ambient: number
  depth: number
}

export interface MessageEntry {
  id: string
  role: 'user' | 'assistant' | 'tool' | 'subliminal' | 'thinking'
  content: Array<{ type: 'text'; text: string }>
  timestamp: string
  session_key?: string
  model?: string
  reasoning?: string
  /** Wall time the model spent reasoning, in milliseconds: first
   *  reasoning chunk to last. Absent on sessions that predate the field
   *  and on turns where the harness reported nothing. */
  reasoning_ms?: number
  /** Present only on `role: 'thinking'` rows — one reasoning phase of one
   *  agent-loop iteration, rendered on the timeline as "Thought for Xs".
   *  The text itself rides in `reasoning`; `content` is empty, which is
   *  what keeps these rows out of every transcript generated from the
   *  session logs. */
  thinking?: {
    chars: number
    iteration: number
    turn_id: string
    /** Set only on the provisional row the browser shows while a phase is
     *  still streaming, and replaced when `thinking_done` lands with the
     *  harness's own measurement. Never persisted. */
    live?: boolean
  }
  stats?: TurnStats
  context_tokens?: number
  tool_calls?: Array<{
    id: string
    call_id: string
    type: string
    function: {
      name: string
      arguments: string
    }
    // Model-written one-liner describing what this call is doing, rendered
    // beside the tool name in the collapsed tool bubble. Absent on every
    // session that predates the field, and on any call where the model
    // skipped it — render the tool name alone in that case.
    summary?: string
  }>
  tool_call_id?: string
  // #306: ephemeral context injection captured per turn (prefetch block,
  // ambient envelope, or 20-turn memory-preservation nudge). Only present
  // when role === 'subliminal'.
  subliminal?: {
    kind: 'prefetch' | 'ambient_envelope' | 'memory_nudge' | 'other'
    sources: string[]
    chars: number
    turn_id: string
  }
}

export interface Session {
  id: string
  session_key: string
  display_name?: string
  // Few-word label written by the secondary model after a turn completes.
  // Empty until a session has been titled — render `preview`, then the id.
  title?: string
  preview?: string
  last_active: string
  platform?: string
  // Inner Voice: A/B linkage tag and critic opt-in flag.
  experiment_id?: string | null
  inner_voice?: boolean
}

export interface SessionMeta {
  session_id: string
  title: string
  preview: string
  platform: string
  model: string
  message_count: number
  inner_voice: boolean
}

// ── Inner Voice types (thin observer) ───────────────────────────────

// One observation = one decision the observer made on one significant
// event in the primary's stream. action enumerates the lever pulled.
//
// v4 (current) levers: noop | inject | cancel | ambient | clarify, plus
// noop_* variants for guarded/skipped decisions. Pre-v4 rows may also
// contain `deny_tool` and `allow` — kept in the union for historical
// render fidelity. New rows never use them.
export type InnerVoiceObservationAction =
  | 'noop'
  | 'inject'
  | 'cancel'
  | 'ambient'
  | 'clarify'
  | 'noop_budget_exhausted'
  | 'noop_empty_content'
  | 'noop_no_ambient_channel'
  | 'noop_ambient_failed'
  | 'noop_no_clarify_channel'
  | 'noop_clarify_failed'
  | 'noop_inject_on_result'
  | 'noop_cancel_on_result'
  | 'noop_clarify_on_result'
  | 'noop_pretool_after_cancel'
  // historical (v3-only) — render with a v3 affix
  | 'deny_tool'
  | 'allow'

export type InnerVoiceObservationTrigger =
  | 'assistant_message'
  | 'tool_call'
  | 'tool_result'
  | 'result'
  | 'pretool'

export interface InnerVoiceObservation {
  id: number
  session_id: string
  turn_id: string
  sequence_in_turn: number
  trigger: InnerVoiceObservationTrigger
  action: InnerVoiceObservationAction
  reason: string | null
  content: string | null
  related_tool: string | null
  input_tokens: number | null
  output_tokens: number | null
  cache_read: number | null
  cache_create: number | null
  latency_ms: number | null
  model: string | null
  error: string | null
  created_at: string
}

export interface InnerVoiceGoalCard {
  success_criteria?: string[]
  out_of_scope?: string[]
  completion_signals?: string[]
}

export interface InnerVoiceState {
  session_id: string | null
  inner_voice_enabled: boolean
  evaluate_user_turns: boolean
  observations_count_by_action: Record<string, number>
  last_observation_at: string | null
  // Most recent goal-card extraction for this session (logged on each turn
  // start). Null when IV hasn't run yet, or when extraction failed. The UI
  // renders the user_request on the left and goal_card on the right.
  latest_goal_card: InnerVoiceGoalCard | null
  latest_user_request: string | null
  latest_turn_id: string | null
}

export interface InnerVoiceSession {
  session_id: string
  experiment_id: string | null
  title: string
  preview?: string
  created_at: string | null
  updated_at: string | null
  message_count: number
  evaluate_user_turns?: boolean
}

// A run nobody was watching: an autonomy task or a worker job. `title` is
// written when the session is created rather than by the LLM titler, so
// unlike a chat row it is never empty and never waited on the secondary.
export interface BackgroundSession {
  id: string
  session_key: string
  title: string
  preview: string
  platform: string
  source: string
  model: string
  inner_voice: boolean
  message_count: number
  last_active: string
}

export interface WorkerSourceHealth {
  name: string
  configured: boolean
  enabled: boolean
  // null = the source does not set it. For a source that never runs
  // through the chat path that is not "off", it is "not observable".
  inner_voice: boolean | null
  interval_seconds?: number | null
  max_inflight?: number | null
  priority?: number | null
  depth: Record<string, number>
  // null when the source has no runs in the window. A rate over zero runs is
  // unknown, not 0% — rendering "0% failing" for a source that has never run
  // is the reading this panel exists to prevent.
  health: {
    total: number
    ok: number
    failed: number
    skipped: number
    fail_rate: number | null
    gpu_hours: number
    last_completed: string | null
  } | null
  recent: Array<{
    run_id: string
    source: string
    status: string
    started_at: string
    completed_at: string
    duration_seconds: number
    summary: string
    task_id?: string | null
    meta_json?: string
  }>
}

export interface InnerVoiceEventLogEntry {
  ts: string
  session_id: string
  turn_id?: string
  event: string
  // `data` is event-specific; large fields may be `{$blob: <sha>}` references
  // when expand_blobs=false (the default).
  data: Record<string, unknown>
}

export interface ModelInfo {
  name: string
  alias: string
  provider: string
  base_url: string
  context_length: number
}

export interface ApiResponse {
  success: boolean
  response?: string
  session_id?: string
  detail?: string
  messages?: MessageEntry[]
  sessions?: Session[]
  model?: string
}

export interface SkillInfo {
  name: string
  description: string
  category?: string
  requires?: {
    bins?: string[]
    env?: string[]
    config?: string[]
    anyBins?: string[]
  }
  os?: string[]
  enabled: boolean
  configured: boolean
  location: string
}

export interface SkillsData {
  workspace: SkillInfo[]
  bundled: SkillInfo[]
}

export interface BacklogTask {
  id: number
  name: string
  description: string
  priority: string
  status: string
  blocked: boolean
  tags: string[]
  completed: boolean
  due_date: string | null
  position: number
  assigned_to_agent: boolean
  /** Board name — the stable identity. `board_id` is positional over the
   *  sorted board names and renumbers whenever a board appears or vanishes. */
  board: string
  board_id: number
  url: string
  created_at: string
  updated_at: string
}

export interface BacklogBoard {
  id: number
  name: string
  icon: string
  color: string
  tasks_count: number
}

export interface ServiceStatus {
  id: string
  name: string
  unit: string
  port: number
  systemdState: 'active' | 'inactive' | 'failed' | 'unknown'
  portHealthy: boolean
  health: 'healthy' | 'degraded' | 'stopped' | 'unknown'
}

export interface ServicesData {
  services: ServiceStatus[]
  timestamp: string
}

/**
 * One frame of the agent's live browser session (backlog #278).
 *
 * `refs` carries the accessibility refs the agent sees (`e1`, `e2`, …).
 * `x`/`y`/`w`/`h` are optional because the MCP server only knows geometry
 * when Playwright reported a bounding box; a frame with no geometry is still
 * a valid frame — the overlay just has nothing to draw.
 */
export interface BrowserRef {
  ref: string
  role: string
  name: string
  x?: number
  y?: number
  w?: number
  h?: number
}

export interface BrowserFrame {
  /** Frame JSON as delivered by GET /api/browser/frame carries `active`; the
   *  SSE `state` event does not. Both shapes arrive here. */
  active?: boolean
  tool?: string
  url?: string
  title?: string
  ts?: number
  mime?: string
  snapshot?: string
  screenshot_b64?: string
  refs?: BrowserRef[]
}

/** Result of driving the shared browser from the URL bar. `ok` and `error`
 *  are mutually exclusive; a 404 or a timeout arrives as `error`. */
export interface BrowserNavigateResult {
  ok?: boolean
  url?: string
  title?: string
  status?: number
  error?: string
}

export interface ServiceDetail {
  id: string
  name: string
  unit: string
  port: number
  pid: number | null
  memory: string | null
  cpu: string | null
  tasks: string | null
  activeSince: string | null
  logLines: string[]
  rawStatus: string
}

export interface LloydServiceUnit {
  id: string
  unit: string
  name: string
  activeState: 'active' | 'inactive' | 'failed' | 'unknown'
  subState: string
  port: number | null
  portHealthy: boolean | null
  uptime: string | null
  health: 'healthy' | 'degraded' | 'stopped' | 'unknown'
}

export interface LloydServicesData {
  services: LloydServiceUnit[]
  timestamp: string
}

export interface McpTool {
  name: string
  description: string
  enabled: boolean
  category?: string
}

export interface McpServer {
  name: string
  label: string
  description: string
  enabled: boolean
  tools: McpTool[]
  error?: string
}

export interface ToolDiscoverySettings {
  enabled: boolean
  threshold_tools: number
  baseline_tools: string[]
  max_results_default: number
  max_results_cap: number
  total_tools: number
  active: boolean
}

export interface ToolsData {
  servers: McpServer[]
}

export interface LloydServiceDetail {
  unit: string
  name: string
  pid: number | null
  memory: string | null
  cpu: string | null
  tasks: string | null
  activeSince: string | null
  logLines: string[]
  rawStatus: string
}

// ── Memory types ──────────────────────────────────────────────────────

export interface MemoryStats {
  docCount: number;
  tagCount: number;
  types: Record<string, number>;
  topTags: TagEntry[];
  lastRefresh: string;
}

export interface TagEntry {
  tag: string;
  count: number;
}

export interface MemorySearchResult {
  query: string;
  results: Array<{ path: string; title: string; score: number; snippet: string; summary: string }>;
}

export interface MemoryBrowseEntry {
  name: string;
  type: "file" | "dir";
  size?: number;
  title?: string;
  children?: number;
}

export interface MemoryBrowseResult {
  path: string;
  entries: MemoryBrowseEntry[];
}

export interface MemoryReadResult {
  path: string;
  frontmatter: Record<string, any>;
  content: string;
  lineCount: number;
}

/** One of app/entity_kind.py KINDS. Drives node colour and the legend. */
export type EntityKind =
  | "person" | "project" | "system" | "concept" | "skill" | "task" | "doc" | "entity";

export interface EntitySummary {
  name: string;
  factCount: number;
  kind: EntityKind;
  categories: string[];
}

export interface EntitiesListData {
  entities: EntitySummary[];
  /** Matches for the current query, not the page length. */
  total: number;
  offset: number;
  limit: number;
  returned: number;
  query: string | null;
}

export interface EntityFact {
  fact: string;
  confidence: number;
  category: string;
  event_date?: string | null;
  id?: string;
  created_at?: string | null;
  source_doc?: string | null;
  provenance?: string | null;
  expired_at?: string | null;
  invalid_at?: string | null;
}

export interface EntityRelationship {
  source: string;
  target: string;
  type: string;
  score: number;
  /** The endpoint that is NOT the entity being viewed. */
  other: string;
  provenance?: string | null;
  created_at?: string | null;
  source_doc?: string | null;
  evidence?: string | null;
}

export interface EntityAlias {
  surface: string;
  kind: string;
  origin: string;
  created_at: string;
  report_path?: string | null;
}

export interface EntityDetailData {
  name: string;
  kind: EntityKind;
  facts: EntityFact[];
  factCount: number;
  relationships: EntityRelationship[];
  outbound: EntityRelationship[];
  inbound: EntityRelationship[];
  aliases: EntityAlias[];
  definition?: string | null;
  summary?: string | null;
  includeExpired: boolean;
}

export interface EntityGraphNode {
  id: string;
  label: string;
  /** EntityKind. Was the entity's first fact category, which made a legend
   *  out of `state` and `goal` and called it a node type. */
  type: EntityKind;
  factCount?: number;
  /** Always null from /api/entity-graph — fetch via entityDetail on select. */
  definition?: string | null;
}

export interface EntityGraphEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
  bidirectional?: boolean;
  provenance?: string | null;
  created_at?: string | null;
}

export interface EntityGraphData {
  nodes: EntityGraphNode[];
  edges: EntityGraphEdge[];
  nodeCount: number;
  edgeCount: number;
  includeIsolated: boolean;
  minConfidence: number;
}

// ── Autonomy types ──────────────────────────────────────────────────────

export interface AutonomyTask {
  id: number;
  name: string;
  description: string;
  status: string;
  priority: string;
  scheduled_at: string | null;
  next_run: string | null;
  auto_advance: boolean;
  preemptible: boolean;
  pipeline_mode: boolean;
  notify_on_complete: boolean;
  created_at: string;
  updated_at: string;
  created?: string;
  updated?: string;
  runs_per_day: number | null;
  depends_on: number | null;
  pipeline: string | null;
  agent_id: string | null;
  skill_name: string | null;
  model: string | null;
  timeout_seconds: number | null;
  max_retries: number | null;
  preferred_hours: string | null;
  frequency: string | null;
  cron_id: string | null;
  last_run: string | null;
  last_attempt: string | null;
  failure_count: number | null;
  stale_bypass_hours: number | null;
  expected_error_patterns: string[] | null;
  tags?: string[];
  /** Why the scheduler will not dispatch this task right now, computed
   *  server-side by `autonomy.hold_reason` — "paused", "waiting on #42",
   *  "outside hours 00-04,23". Null means nothing is holding it. */
  blocked?: string | null;
}

export interface AutonomyHealthTask {
  task_id: string;
  name: string | null;
  status: string | null;
  frequency?: string | null;
  runs: number;
  successes: number;
  failures: number;
  timeouts: number;
  empty: number;
  silent: number;
  silent_indicator_runs?: number;
  max_turns_runs?: number;
  tool_error_runs?: number;
  fail_rate: number;
  silent_rate: number;
  gpu_hours: number;
  wasted_hours: number;
  avg_seconds: number;
  max_seconds: number;
  consecutive_failures: number;
  last_success: string | null;
  failure_count?: number;
}

export interface AutonomyHealth {
  days: number;
  generated_at: string;
  fleet: {
    runs: number;
    failures: number;
    fail_rate: number;
    gpu_hours: number;
    wasted_hours: number;
    empty_runs: number;
    timeout_runs: number;
    active_tasks: number;
    failed_tasks: string[];
    paused_tasks: string[];
  };
  tasks: AutonomyHealthTask[];
  idle_tasks: AutonomyHealthTask[];
}

export interface TodoItem {
  content: string
  status: 'pending' | 'in_progress' | 'completed'
  activeForm: string
  stage?: number
}

export interface PlanStage {
  n: number
  title: string
  summary?: string
}

export interface SessionPlan {
  plan_mode: boolean
  plan_md_path?: string
  stages?: PlanStage[]
  created_at?: string
  drafted_at?: string
  committed_at?: string
  cancelled_at?: string
}

export interface SessionGoal {
  text?: string
  set_at?: string
  achieved_at?: string | null
  attempts?: number
}

export interface ActiveProc {
  pid: number
  sdk_session_id: string | null
  session_id: string | null
  model: string | null
  preview: string
  created_at: string | null
  streaming: boolean
}

// The chrome side-panel build sets VITE_API_BASE='http://127.0.0.1:8080/api'
// (loopback bypasses the mTLS middleware at server.py:76-113). Main web app
// keeps the relative '/api' default so the Vite proxy injects client-cert
// headers.
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) || '/api'

export const api = {
  async sendMessage(text: string, clientId: string, sessionId?: string): Promise<ApiResponse> {
    const response = await fetch(`${API_BASE}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, client_id: clientId, session_id: sessionId }),
    })
    return response.json()
  },

  streamMessage(
    text: string,
    clientId: string,
    sessionId: string | undefined,
    callbacks: {
      onSession?: (sessionId: string) => void
      onToolStart?: (callId: string, name: string, args: string, contextTokens?: number, summary?: string) => void
      onToolComplete?: (callId: string, name: string, result: string) => void
      onToolProgress?: (name: string, preview: string) => void
      onTextDelta?: (text: string) => void
      onThinkingDelta?: (text: string) => void
      /** One reasoning phase ended. `seq` orders phases within the turn and
       *  `iteration` names the agent-loop iteration, matching the
       *  `role: 'thinking'` row the backend persists for the same phase. */
      onThinkingDone?: (fullText: string, durationMs?: number, seq?: number, iteration?: number) => void
      onDone?: (response: string, sessionId: string, stats?: TurnStats, reasoning?: string, cancelled?: boolean, reasoningMs?: number) => void
      onError?: (detail: string) => void
      onAborted?: () => void
      onQueueState?: (state: QueueState) => void
    },
    model?: string,
    think?: string,
  ): AbortController {
    const controller = new AbortController()
    fetch(`${API_BASE}/message/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, client_id: clientId, session_id: sessionId, ...(model ? { model } : {}), ...(think ? { think } : {}) }),
      signal: controller.signal,
    }).then(async (response) => {
      if (!response.ok || !response.body) {
        try {
          const errData = await response.json()
          callbacks.onError?.(errData.detail || `HTTP ${response.status}`)
        } catch {
          callbacks.onError?.(`HTTP ${response.status}`)
        }
        return
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // Parse SSE events from buffer
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          const lines = part.split('\n')
          let eventType = ''
          let data = ''
          for (const line of lines) {
            if (line.startsWith('event: ')) eventType = line.slice(7)
            else if (line.startsWith('data: ')) data = line.slice(6)
          }
          if (!eventType || !data) continue
          try {
            const payload = JSON.parse(data)
            switch (eventType) {
              case 'session': callbacks.onSession?.(payload.session_id); break
              case 'tool_start': callbacks.onToolStart?.(payload.call_id, payload.name, payload.args, payload.context_tokens, payload.summary); break
              case 'tool_complete': callbacks.onToolComplete?.(payload.call_id, payload.name, payload.result); break
              case 'tool_progress': callbacks.onToolProgress?.(payload.name, payload.preview); break
              case 'text_delta': callbacks.onTextDelta?.(payload.text); break
              case 'thinking_delta': callbacks.onThinkingDelta?.(payload.text); break
              case 'thinking_done': callbacks.onThinkingDone?.(payload.text, payload.duration_ms, payload.seq, payload.iteration); break
              case 'done': callbacks.onDone?.(payload.response, payload.session_id, payload.stats, payload.reasoning, payload.cancelled, payload.reasoning_ms); break
              case 'error': callbacks.onError?.(payload.detail); break
              case 'queue_state': callbacks.onQueueState?.(payload as QueueState); break
            }
          } catch { /* skip malformed */ }
        }
      }
    }).catch((err) => {
      if (err.name === 'AbortError') callbacks.onAborted?.()
      else callbacks.onError?.(err.message)
    })
    return controller
  },

  async loadMessages(sessionKey: string, limit = 50): Promise<ApiResponse> {
    // Try the new endpoint first
    try {
      const response = await fetch(`${API_BASE}/messages/${encodeURIComponent(sessionKey)}`)
      if (response.ok) {
        const data = await response.json()
        return { success: true, messages: data.messages, session_id: data.session_key, model: data.model }
      }
    } catch (err) {
      console.warn('Failed to load messages via new endpoint, trying old:', err)
    }
    // Fallback to old endpoint
    const response = await fetch(`${API_BASE}/messages?session_key=${encodeURIComponent(sessionKey)}&limit=${limit}`)
    return response.json()
  },

  async getSessionStatus(sessionId: string): Promise<{ streaming: boolean }> {
    const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/status`)
    return response.json()
  },

  async getSessionTodos(sessionId: string): Promise<TodoItem[]> {
    const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/todos`)
    if (!response.ok) return []
    const data = await response.json()
    return Array.isArray(data?.todos) ? data.todos : []
  },

  async getSessionPlan(sessionId: string): Promise<SessionPlan> {
    const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/plan`)
    if (!response.ok) return { plan_mode: false }
    const data = await response.json()
    return (data?.plan as SessionPlan) || { plan_mode: false }
  },

  async getSessionPlanDocument(sessionId: string): Promise<{ plan_md_path: string; plan_md: string }> {
    const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/plan/document`)
    if (!response.ok) return { plan_md_path: '', plan_md: '' }
    return response.json()
  },

  async enterPlanMode(sessionId: string): Promise<{ plan_mode: boolean; session_id: string }> {
    const response = await fetch(
      `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/plan_mode/enter`,
      { method: 'POST' },
    )
    return response.json()
  },

  async exitPlanMode(sessionId: string): Promise<{ plan_mode: boolean; session_id: string }> {
    const response = await fetch(
      `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/plan_mode/exit`,
      { method: 'POST' },
    )
    return response.json()
  },

  async getSessionGoal(sessionId: string): Promise<SessionGoal> {
    const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/goal`)
    if (!response.ok) return {}
    const data = await response.json()
    return (data?.goal as SessionGoal) || {}
  },

  async setSessionGoal(sessionId: string, text: string): Promise<{ goal: SessionGoal; session_id: string }> {
    const response = await fetch(
      `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/goal`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      },
    )
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }))
      throw new Error(err.detail || `HTTP ${response.status}`)
    }
    return response.json()
  },

  async clearSessionGoal(sessionId: string): Promise<{ session_id: string; cleared: boolean }> {
    const response = await fetch(
      `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/goal`,
      { method: 'DELETE' },
    )
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }))
      throw new Error(err.detail || `HTTP ${response.status}`)
    }
    return response.json()
  },

  async cancelSession(
    sessionId: string,
    options: { drainPending?: boolean } = {},
  ): Promise<{ cancelled: boolean; drained: number; detail?: string }> {
    const qs = options.drainPending ? '?drain_pending=true' : ''
    const response = await fetch(
      `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/cancel${qs}`,
      { method: 'POST' },
    )
    return response.json()
  },

  /** Put back the files a turn wrote. Omit `paths` for all of them. */
  async revertTurn(
    sessionId: string,
    turnId: string,
    paths?: string[],
  ): Promise<{ session_id: string; turn_id: string; results: RevertResult[] }> {
    const response = await fetch(
      `${API_BASE}/sessions/${encodeURIComponent(sessionId)}/turns/${encodeURIComponent(turnId)}/revert`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(paths ? { paths } : {}),
      },
    )
    if (!response.ok) {
      throw new Error(`revert failed: ${response.status} ${await response.text()}`)
    }
    return response.json()
  },

  async getSessionQueue(sessionId: string): Promise<QueueState> {
    const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/queue`)
    return response.json()
  },

  async injectAmbient(
    sessionId: string,
    text: string,
    dedupKey?: string,
  ): Promise<{ turn_id: string; source: string; preempted: boolean; dropped: string[]; deduped: boolean; queue: QueueState }> {
    const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/inject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, ...(dedupKey ? { dedup_key: dedupKey } : {}) }),
    })
    return response.json()
  },

  async listSessions(): Promise<ApiResponse> {
    const response = await fetch(`${API_BASE}/sessions`)
    return response.json()
  },

  // Every run that is not a conversation: autonomy tasks and worker jobs.
  // The other half of the history bifurcation — `listSessions` is chats and
  // excludes these, and the Background tab is the only page that reads this.
  listBackgroundSessions: (
    limit = 100,
  ): Promise<{ sessions: BackgroundSession[]; count: number; scanned?: number }> =>
    fetch(`${API_BASE}/background/sessions?limit=${limit}`).then(r => {
      if (!r.ok) throw new Error(`background sessions: ${r.status}`)
      return r.json()
    }),

  // Display metadata for one session, without pulling the whole list.
  // The chat header uses this to name the session it is showing.
  getSessionMeta: (sessionId: string): Promise<SessionMeta> =>
    fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/meta`).then(r => {
      if (!r.ok) throw new Error(`session meta ${sessionId}: ${r.status}`)
      return r.json()
    }),

  async clearSession(sessionKey: string): Promise<ApiResponse> {
    const response = await fetch(`${API_BASE}/clear`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_key: sessionKey }),
    })
    return response.json()
  },

  async getModels(): Promise<{ models: ModelInfo[] }> {
    const response = await fetch(`${API_BASE}/models`)
    return response.json()
  },

  async switchModel(model: string, sessionId?: string): Promise<{ success: boolean; model: string; session_id?: string }> {
    const response = await fetch(`${API_BASE}/model/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, session_id: sessionId }),
    })
    return response.json()
  },

  backlogBoards(): Promise<BacklogBoard[]> {
    return fetch(`${API_BASE}/backlog/boards`).then(r => r.json()).then(d => Array.isArray(d) ? d : [])
  },

  backlogTasks(params?: Record<string, string>): Promise<BacklogTask[]> {
    const qs = params ? '?' + new URLSearchParams(params).toString() : ''
    return fetch(`${API_BASE}/backlog/tasks${qs}`).then(r => r.json()).then(d => Array.isArray(d) ? d : [])
  },

  // These three threw away the response. `fetch` does not reject on 4xx, so a
  // refused write — a malformed-frontmatter 409, an invalid status, an
  // unresolvable board — resolved successfully and the task modal closed as
  // though it had saved. Raise so the caller can say what happened.
  async backlogUpdateTask(id: number, updates: Record<string, any>): Promise<void> {
    const r = await fetch(`${API_BASE}/backlog/task-update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, ...updates }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `task-update failed: ${r.status}`)
    }
  },

  async backlogDeleteTask(id: number): Promise<void> {
    const r = await fetch(`${API_BASE}/backlog/task-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `task-delete failed: ${r.status}`)
    }
  },

  skills(): Promise<SkillsData> {
    return fetch(`${API_BASE}/skills`).then(r => r.json()).then(d => ({
      workspace: Array.isArray(d?.workspace) ? d.workspace : [],
      bundled: Array.isArray(d?.bundled) ? d.bundled : [],
    }))
  },

  async skillToggle(skillName: string, enabled: boolean): Promise<void> {
    await fetch(`${API_BASE}/skill-toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skillName, enabled }),
    })
  },

  skillContent(name: string): Promise<{ content: string; location: string }> {
    return fetch(`${API_BASE}/skill-content?name=${encodeURIComponent(name)}`).then(r => r.json())
  },

  async skillContentSave(name: string, content: string): Promise<void> {
    await fetch(`${API_BASE}/skill-content`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skillName: name, content }),
    })
  },

  async skillsRefresh(): Promise<void> {
    await fetch(`${API_BASE}/skills/refresh`, { method: 'POST' })
  },

  tools(): Promise<ToolsData> {
    return fetch(`${API_BASE}/tools`).then(r => r.json()).then(d => ({
      servers: Array.isArray(d?.servers) ? d.servers : [],
    }))
  },

  async toolToggle(
    payload:
      | { type: 'server'; server: string; enabled: boolean }
      | { type: 'tool'; server: string; tool: string; enabled: boolean }
      | { type: 'baseline'; tool: string; enabled: boolean }
  ): Promise<void> {
    await fetch(`${API_BASE}/tool-toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },

  // Progressive tool discovery (config knobs under harness.tool_search).
  toolDiscovery(): Promise<ToolDiscoverySettings> {
    return fetch(`${API_BASE}/tool-discovery`).then(r => r.json())
  },

  async setToolDiscovery(
    patch: Partial<Pick<ToolDiscoverySettings,
      'enabled' | 'threshold_tools' | 'baseline_tools' |
      'max_results_default' | 'max_results_cap'>>,
  ): Promise<void> {
    await fetch(`${API_BASE}/tool-discovery`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    })
  },

  async backlogCreateTask(data: {
    name: string
    description?: string
    /** Board name. Preferred over `board_id`; see BacklogTask.board. */
    board?: string
    board_id?: number
    status?: string
    tags?: string[]
    priority?: string
  }): Promise<void> {
    const r = await fetch(`${API_BASE}/backlog/task-create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `task-create failed: ${r.status}`)
    }
  },

  services(): Promise<ServicesData> {
    return fetch(`${API_BASE}/services`).then(r => r.json())
  },

  serviceDetail(id: string): Promise<ServiceDetail> {
    return fetch(`${API_BASE}/services/detail?id=${encodeURIComponent(id)}`).then(r => r.json())
  },

  async serviceAction(serviceId: string, action: 'start' | 'stop' | 'restart'): Promise<void> {
    await fetch(`${API_BASE}/services/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ serviceId, action }),
    })
  },

  lloydServices(): Promise<LloydServicesData> {
    return fetch(`${API_BASE}/agent-services`).then(r => r.json())
  },

  lloydServiceDetail(unit: string): Promise<LloydServiceDetail> {
    return fetch(`${API_BASE}/agent-services/detail?unit=${encodeURIComponent(unit)}`).then(r => r.json())
  },

  async lloydServiceAction(serviceId: string, action: 'start' | 'stop' | 'restart'): Promise<void> {
    await fetch(`${API_BASE}/services/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ serviceId, action }),
    })
  },

  // Memory / Vault
  memoryStats: (): Promise<MemoryStats> =>
    fetch(`${API_BASE}/memory/stats`).then(r => r.json()),
  memorySearch: (q: string, limit = 10): Promise<MemorySearchResult> =>
    fetch(`${API_BASE}/memory/search?q=${encodeURIComponent(q)}&limit=${limit}`).then(r => r.json()),
  memoryBrowse: (path = ''): Promise<MemoryBrowseResult> =>
    fetch(`${API_BASE}/memory/browse?path=${encodeURIComponent(path)}`).then(r => r.json()),
  memoryRead: async (path: string): Promise<MemoryReadResult> => {
    const r = await fetch(`${API_BASE}/memory/read?path=${encodeURIComponent(path)}`)
    if (!r.ok) throw new Error(`memory_read failed (${r.status}): ${path}`)
    return r.json()
  },
  async memorySave(path: string, content: string, frontmatter?: Record<string, unknown>): Promise<{ ok: boolean }> {
    const res = await fetch(`${API_BASE}/memory/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content, frontmatter }),
    })
    return res.json()
  },

  // Entity / knowledge graph
  entityList: (limit = 200, opts: { q?: string; offset?: number } = {}): Promise<EntitiesListData> => {
    const p = new URLSearchParams({ limit: String(limit), offset: String(opts.offset ?? 0) });
    if (opts.q) p.set("q", opts.q);
    return fetch(`${API_BASE}/entities?${p}`).then(r => r.json());
  },
  entityDetail: (name: string, opts: { includeExpired?: boolean } = {}): Promise<EntityDetailData> => {
    const p = new URLSearchParams({ name });
    if (opts.includeExpired) p.set("include_expired", "1");
    return fetch(`${API_BASE}/entity?${p}`).then(r => r.json());
  },
  entityGraph: (opts: { includeIsolated?: boolean; limit?: number; minConfidence?: number } = {}): Promise<EntityGraphData> => {
    const p = new URLSearchParams();
    if (opts.includeIsolated) p.set("include_isolated", "1");
    if (opts.limit) p.set("limit", String(opts.limit));
    if (opts.minConfidence) p.set("min_confidence", String(opts.minConfidence));
    const qs = p.toString();
    return fetch(`${API_BASE}/entity-graph${qs ? "?" + qs : ""}`).then(r => r.json());
  },

  // Autonomy
  autonomyTasks: (): Promise<{ tasks: AutonomyTask[] }> =>
    fetch(`${API_BASE}/autonomy/tasks`).then(r => r.json()),
  autonomyHealth: (days = 7): Promise<AutonomyHealth> =>
    fetch(`${API_BASE}/autonomy/health?days=${days}`).then(r => r.json()),
  autonomyWriteTask: async (data: Record<string, any>): Promise<any> => {
    const res = await fetch(`${API_BASE}/autonomy/task-write`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    return res.json()
  },
  autonomyDeleteTask: async (id: number): Promise<any> => {
    const res = await fetch(`${API_BASE}/autonomy/task-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    })
    return res.json()
  },
  autonomyRuns: (taskId: number, limit = 20): Promise<{ runs: any[] }> =>
    fetch(`${API_BASE}/autonomy/runs?task_id=${taskId}&limit=${limit}`).then(r => r.json()),

  // Workers (unified work queue)
  workersStatus: (): Promise<{
    initialized: boolean
    workers_enabled?: boolean
    pool?: { running: boolean; paused: boolean; slots: number; in_flight: Record<string, any>; in_flight_count: number
             kv_gate?: KvGateState }
    depth?: Record<string, Record<string, number>>
    sources?: Array<{ name: string; enabled: boolean; interval_seconds?: number; max_inflight?: number; depth?: Record<string, number> }>
  }> => fetch(`${API_BASE}/workers/status`).then(r => r.json()),
  // Per-source health: config, queue depth, outcome rollup, recent runs.
  // `workersStatus` reports only what a source is allowed to do; this is
  // whether it works.
  workersHealth: (
    days = 7, runs = 10,
  ): Promise<{ initialized: boolean; days: number; sources: WorkerSourceHealth[] }> =>
    fetch(`${API_BASE}/workers/health?days=${days}&runs=${runs}`).then(r => {
      if (!r.ok) throw new Error(`workers health: ${r.status}`)
      return r.json()
    }),

  workersQueue: (opts: { state?: string; source?: string; limit?: number } = {}): Promise<{ items: any[] }> => {
    const q = new URLSearchParams()
    if (opts.state) q.set('state', opts.state)
    if (opts.source) q.set('source', opts.source)
    if (opts.limit) q.set('limit', String(opts.limit))
    return fetch(`${API_BASE}/workers/queue?${q.toString()}`).then(r => r.json())
  },
  workersRuns: (opts: { source?: string; task_id?: string; limit?: number } = {}): Promise<{ runs: any[] }> => {
    const q = new URLSearchParams()
    if (opts.source) q.set('source', opts.source)
    if (opts.task_id) q.set('task_id', opts.task_id)
    if (opts.limit) q.set('limit', String(opts.limit))
    return fetch(`${API_BASE}/workers/runs?${q.toString()}`).then(r => r.json())
  },
  workersEnqueue: (data: { source: string; kind: string; payload?: Record<string, unknown>; priority?: number; dedup_key?: string }): Promise<{ id?: number; coalesced?: boolean }> =>
    fetch(`${API_BASE}/workers/enqueue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }).then(r => r.json()),
  workersPause: (paused: boolean): Promise<{ paused: boolean }> =>
    fetch(`${API_BASE}/workers/pause`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paused }),
    }).then(r => r.json()),
  workersEnable: (enabled: boolean): Promise<{ enabled: boolean }> =>
    fetch(`${API_BASE}/workers/enable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    }).then(r => r.json()),
  workersPending: (source?: string, limit = 200): Promise<{
    items: Array<{
      path: string
      source: string
      date: string
      filename: string
      size_bytes: number
      mtime: string
      frontmatter: Record<string, any>
      preview: string
    }>
    sources: string[]
  }> => {
    const q = new URLSearchParams()
    if (source) q.set('source', source)
    q.set('limit', String(limit))
    return fetch(`${API_BASE}/workers/pending?${q.toString()}`).then(r => r.json())
  },
  workersPendingRead: (path: string): Promise<{
    path: string
    source: string | null
    frontmatter: Record<string, any>
    body: string
    raw: string
  }> => fetch(`${API_BASE}/workers/pending/read?path=${encodeURIComponent(path)}`).then(r => r.json()),
  workersPendingPromote: (data: { path: string; destination?: string; filename?: string }): Promise<{ promoted: boolean; from: string; to: string }> =>
    fetch(`${API_BASE}/workers/pending/promote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }).then(async r => {
      if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`)
      return r.json()
    }),
  workersPendingReject: (path: string): Promise<{ rejected: boolean; from: string; to: string }> =>
    fetch(`${API_BASE}/workers/pending/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }).then(r => r.json()),

  // Active SDK session subprocesses
  // Live browser session mirror (#278) — the frame that was current when the
  // agent last touched the browser. SSE /api/browser/state delivers the
  // updates; this is the cold-start read so a fresh tab isn't blank.
  getBrowserFrame: (): Promise<BrowserFrame> =>
    fetch(`${API_BASE}/browser/frame`).then(r => r.json()),

  // The Browser tab's URL bar. A failed navigation (bad host, timeout) comes
  // back as a 200 carrying `error` — it is an answer for the user to read,
  // not a broken request — so callers must check the body, not the status.
  browserNavigate: (url: string): Promise<BrowserNavigateResult> =>
    fetch(`${API_BASE}/browser/navigate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    }).then(r => r.json()),

  getActiveProcs: (): Promise<{ procs: ActiveProc[] }> =>
    fetch(`${API_BASE}/sessions/active-procs`).then(r => r.json()),
  killSessionProc: (sessionId: string): Promise<{ killed: boolean; session_id: string }> =>
    fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/kill-proc`, { method: 'POST' }).then(r => r.json()),

  // ── Inner Voice ──
  // Patch session metadata: experiment tag + critic opt-in flag +
  // user-turn evaluation flag. All optional — caller sends only what changed.
  patchSession: (
    sessionId: string,
    patch: {
      experiment_id?: string | null
      inner_voice?: boolean
      inner_voice_evaluate_user_turns?: boolean
    },
  ): Promise<{
    session_key: string
    experiment_id: string | null
    inner_voice: boolean
    inner_voice_evaluate_user_turns?: boolean
  }> =>
    fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    }).then(r => r.json()),

  // Pre-create a session with Inner Voice flags set so the critic fires
  // on turn 1. The Inner Voice tab uses this for the "+ new chat" button
  // — regular Chat sessions are still created lazily via the
  // streamMessage path.
  createSession: (
    body: {
      model?: string
      platform?: string
      inner_voice?: boolean
      inner_voice_evaluate_user_turns?: boolean
      experiment_id?: string | null
    } = {},
  ): Promise<{
    session_key: string
    session_id: string
    model: string
    platform: string
    inner_voice: boolean
    inner_voice_evaluate_user_turns: boolean
    experiment_id: string | null
  }> =>
    fetch(`${API_BASE}/sessions/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => {
      if (!r.ok) throw new Error(`createSession failed: ${r.status}`)
      return r.json()
    }),

  innerVoiceState: (sessionId?: string): Promise<InnerVoiceState> => {
    const params = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''
    return fetch(`${API_BASE}/inner_voice/state${params}`).then(r => r.json())
  },

  innerVoiceObservations: (
    sessionId?: string,
    turnId?: string,
    limit = 200,
  ): Promise<{ observations: InnerVoiceObservation[]; count: number }> => {
    const p = new URLSearchParams()
    if (sessionId) p.set('session_id', sessionId)
    if (turnId) p.set('turn_id', turnId)
    p.set('limit', String(limit))
    return fetch(`${API_BASE}/inner_voice/observations?${p}`).then(r => r.json())
  },

  innerVoiceEventLog: (
    sessionId: string,
    offset = 0,
    limit = 200,
    expandBlobs = false,
  ): Promise<{
    session_id: string
    events: InnerVoiceEventLogEntry[]
    offset: number
    limit: number
    returned: number
    total: number
  }> => {
    const p = new URLSearchParams({
      session_id: sessionId,
      offset: String(offset),
      limit: String(limit),
      expand_blobs: String(expandBlobs),
    })
    return fetch(`${API_BASE}/inner_voice/event_log?${p}`).then(r => r.json())
  },

  // List sessions opted into Inner Voice (sessions whose JSON has
  // `inner_voice: true`). Used by the InnerVoicePage session picker.
  innerVoiceSessions: (
    limit = 50,
  ): Promise<{ sessions: InnerVoiceSession[]; count: number }> =>
    fetch(`${API_BASE}/inner_voice/sessions?limit=${limit}`).then(r => r.json()),

  // Resolve a single blob hash to its content. Returns 404 on miss.
  innerVoiceEventLogBlob: (
    sha: string,
  ): Promise<{ sha: string; content: string; size: number }> =>
    fetch(`${API_BASE}/inner_voice/event_log/blob/${encodeURIComponent(sha)}`)
      .then(r => {
        if (!r.ok) throw new Error(`blob ${sha} not found (${r.status})`)
        return r.json()
      }),

  // ── LiveKit (Phase 3) ──────────────────────────────────────────────
  // Mint a room-scoped JWT for the browser client. The session_id maps
  // 1:1 to a room (`lloyd-${session_id}`); the agent-worker watches
  // RoomService and joins the same room once a participant connects.
  livekitToken: (
    sessionId: string,
    identity?: string,
  ): Promise<{ url: string; token: string; room: string; identity: string }> =>
    fetch(`${API_BASE}/livekit/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, identity }),
    }).then(r => {
      if (!r.ok) throw new Error(`livekit/token failed: ${r.status}`)
      return r.json()
    }),

  // ── Voiceprint enrollment ──────────────────────────────────────────
  voiceSpeakersList: (): Promise<{
    profiles: Array<{ name: string; embedding_dim: number; path: string }>
  }> =>
    fetch(`${API_BASE}/voice/speakers`).then(r => {
      if (!r.ok) throw new Error(`voice/speakers failed: ${r.status}`)
      return r.json()
    }),

  voiceSpeakersEnroll: async (
    name: string,
    audio: Blob,
  ): Promise<{ name: string; path: string; duration_s: number; sample_rate: number }> => {
    const fd = new FormData()
    fd.append('name', name)
    fd.append('audio', audio, `${name}.wav`)
    const r = await fetch(`${API_BASE}/voice/speakers/enroll`, { method: 'POST', body: fd })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `enroll failed: ${r.status}`)
    }
    return r.json()
  },

  voiceSpeakersDelete: async (name: string): Promise<{ deleted: string }> => {
    const r = await fetch(`${API_BASE}/voice/speakers/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `delete failed: ${r.status}`)
    }
    return r.json()
  },

  // ── LAN access / mTLS ────────────────────────────────────────────────
  getLanInfo: async (): Promise<{
    lan_ip: string | null
    hostname: string
    https_url: string | null
    ca_available: boolean
  }> => {
    const r = await fetch(`${API_BASE}/system/lan-info`)
    if (!r.ok) throw new Error(`lan-info failed: ${r.status}`)
    return r.json()
  },

  getIdentity: async (): Promise<{ name: string | null; fingerprint: string | null }> => {
    const r = await fetch(`${API_BASE}/system/identity`)
    if (!r.ok) throw new Error(`identity failed: ${r.status}`)
    return r.json()
  },

  getCABlob: async (): Promise<Blob> => {
    const r = await fetch(`${API_BASE}/system/cert/ca`)
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `CA download failed: ${r.status}`)
    }
    return r.blob()
  },

  listClients: async (): Promise<{
    clients: Array<{ name: string; fingerprint: string; issued_at: string }>
  }> => {
    const r = await fetch(`${API_BASE}/system/clients`)
    if (!r.ok) throw new Error(`list clients failed: ${r.status}`)
    return r.json()
  },

  mintClient: async (name: string, passphrase?: string): Promise<{
    name: string
    fingerprint: string
    issued_at: string
    passphrase: string
    p12_url: string
  }> => {
    const r = await fetch(`${API_BASE}/system/clients`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, ...(passphrase ? { passphrase } : {}) }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `mint failed: ${r.status}`)
    }
    return r.json()
  },

  downloadClientP12: async (name: string): Promise<Blob> => {
    const r = await fetch(`${API_BASE}/system/clients/${encodeURIComponent(name)}/p12`)
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `p12 download failed: ${r.status}`)
    }
    return r.blob()
  },

  revokeClient: async (name: string): Promise<void> => {
    const r = await fetch(`${API_BASE}/system/clients/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `revoke failed: ${r.status}`)
    }
  },

  // ── IDE tab ─────────────────────────────────────────────────────────

  ideList: async (path: string): Promise<IdeListResponse> => {
    const r = await fetch(`${API_BASE}/ide/list?path=${encodeURIComponent(path)}`)
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `ide list failed: ${r.status}`)
    }
    return r.json()
  },

  ideRead: async (path: string): Promise<IdeFileResponse> => {
    const r = await fetch(`${API_BASE}/ide/file?path=${encodeURIComponent(path)}`)
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `ide read failed: ${r.status}`)
    }
    return r.json()
  },

  ideWrite: async (
    path: string,
    content: string,
    expected_mtime?: number,
  ): Promise<IdeWriteResponse> => {
    const r = await fetch(`${API_BASE}/ide/file`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content, expected_mtime }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      const e = new Error(err?.detail || `ide write failed: ${r.status}`)
      ;(e as Error & { status?: number }).status = r.status
      throw e
    }
    return r.json()
  },

  ideCreate: async (path: string, type: 'file' | 'dir'): Promise<{ path: string; type: string }> => {
    const r = await fetch(`${API_BASE}/ide/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, type }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `ide create failed: ${r.status}`)
    }
    return r.json()
  },

  ideRename: async (from: string, to: string): Promise<{ from: string; to: string }> => {
    const r = await fetch(`${API_BASE}/ide/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from, to }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `ide rename failed: ${r.status}`)
    }
    return r.json()
  },

  ideDelete: async (path: string): Promise<{ path: string; deleted: boolean }> => {
    const r = await fetch(`${API_BASE}/ide/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `ide delete failed: ${r.status}`)
    }
    return r.json()
  },

  ideGlob: async (root: string, limit = 4000): Promise<{ root: string; files: string[]; truncated: boolean }> => {
    const r = await fetch(`${API_BASE}/ide/glob?root=${encodeURIComponent(root)}&limit=${limit}`)
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `ide glob failed: ${r.status}`)
    }
    return r.json()
  },

  ideGitDiff: async (path: string): Promise<{ path: string; hunks: GitHunk[] }> => {
    const r = await fetch(`${API_BASE}/ide/git/diff?path=${encodeURIComponent(path)}`)
    if (!r.ok) return { path, hunks: [] }
    return r.json()
  },

  ideAiHover: async (
    args: { path: string; code: string; symbol: string; line: number; language?: string },
  ): Promise<{ markdown: string }> => {
    const r = await fetch(`${API_BASE}/ide/ai/hover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args),
    })
    if (!r.ok) return { markdown: '' }
    return r.json()
  },

  ideAiAction: async (
    args: { path: string; code: string; range_code: string; action: string; language?: string },
  ): Promise<{ result?: string; edit?: string }> => {
    const r = await fetch(`${API_BASE}/ide/ai/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err?.detail || `ide ai action failed: ${r.status}`)
    }
    return r.json()
  },

  ideAiComplete: async (
    args: { prefix: string; suffix: string; language?: string },
    onChunk: (chunk: string) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const r = await fetch(`${API_BASE}/ide/ai/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args),
      signal,
    })
    if (!r.ok || !r.body) return
    const reader = r.body.getReader()
    const decoder = new TextDecoder()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      onChunk(decoder.decode(value, { stream: true }))
    }
  },
}

export interface GitHunk {
  type: 'add' | 'modify' | 'delete'
  new_start: number
  new_count: number
}

export interface IdeEntry {
  name: string
  isDir: boolean
  size: number
  mtime: number
}

export interface IdeListResponse {
  path: string
  entries: IdeEntry[]
  truncated: boolean
}

export interface IdeFileResponse {
  path: string
  size: number
  mtime: number
  binary: boolean
  too_large: boolean
  content: string | null
}

export interface IdeWriteResponse {
  path: string
  size: number
  mtime: number
}

// ── Dashboard ────────────────────────────────────────────────────────────
//
// Every section is gathered independently by the backend, so any one of
// them can come back as `{error}` while the rest hold real data. The
// `| SectionError` unions are load-bearing: the page renders a degraded
// panel rather than blanking when, say, supervisord is wedged.

export interface SectionError { error: string }

export interface VllmEngine {
  alias: string
  base_url: string
  reachable: boolean
  error?: string
  // Which server produced this snapshot. The secondary slot is llama.cpp
  // (Qwen3.6-35B-A3B GGUF); everything else is vLLM. llama.cpp reports no
  // KV-occupancy gauge and no TTFT, so those come back null there.
  engine?: 'vllm' | 'llama.cpp'
  model_name?: string
  awake?: boolean
  requests_running?: number | null
  requests_waiting?: number | null
  requests_waiting_by_reason?: Record<string, number>
  kv_cache_usage?: number | null
  prompt_tokens_per_s?: number | null
  generation_tokens_per_s?: number | null
  preemptions_per_s?: number | null
  ttft_s?: number | null
  itl_s?: number | null
  prompt_tokens_total?: number | null
  generation_tokens_total?: number | null
  preemptions_total?: number | null
  finished_by_reason?: Record<string, number>
  prefix_cache_hit_rate?: number | null
  prefix_cache_hit_rate_recent?: number | null
  spec_decode_hit_rate?: number | null
  spec_decode_hit_rate_recent?: number | null
  /** Primary only: KV pressure from the backend's background sampler
   *  (app/engine_pressure.py) — a p90 over the window, which one 2-second
   *  reading cannot show. Absent on other engines and on an older backend. */
  pressure?: EnginePressure
}

export interface EnginePressure {
  alias: string
  base_url: string
  sampling: boolean
  window_s: number
  samples: number
  kv_now: number | null
  /** The middle is residents; the tail is cold prefills, which reference
   *  ~2.5x their resident footprint while they build. */
  kv_p50: number | null
  kv_p90: number | null
  kv_max: number | null
  warn_line: number
  stale: boolean
  error: string | null
}

export interface GpuInfo {
  index: number
  name: string
  gpu_util: number | null
  mem_util: number | null
  memory_used_mb: number | null
  memory_total_mb: number | null
  memory_pct: number | null
  temperature_c: number | null
  power_draw_w: number | null
  power_limit_w: number | null
}

export interface HostMetrics {
  cpu: { percent: number; count: number; physical_count: number; load_average: number[] | null }
  memory: { used_bytes: number; total_bytes: number; percent: number }
  swap: { used_bytes: number; total_bytes: number; percent: number }
  disks: Array<{ path: string; used_bytes: number; total_bytes: number; percent: number }>
  gpus: GpuInfo[]
  uptime_seconds: number
  boot_time: number
}

// One live snapshot of what a running turn is doing. `kind` is the state
// machine; `label`/`detail` carry the tool name and its headline argument.
export interface TurnActivity {
  kind: 'starting' | 'prefill' | 'thinking' | 'responding' | 'tool' | 'working'
  label: string
  detail: string
  at: string
}

export interface PrimaryState {
  model: string
  base_url: string
  context_length: number | null
  max_turns: number | null
  permission_mode: string
  preserve_thinking_iterations: number | null
  sessions: Array<{
    session_id: string
    title?: string
    running: boolean
    turn_id: string | null
    source: string | null
    started_at: string | null
    enqueued_at: string | null
    preempted: boolean
    // What the turn is doing right now. Null before the first event of a
    // turn, and on every queued (not yet running) session.
    activity: TurnActivity | null
    pending_user: number
    pending_ambient: number
  }>
  running_count: number
  queued_count: number
  busy: boolean
}

/** A chat that has stopped talking — nothing running or queued on it. */
export interface RecentSession {
  session_id: string
  title: string
  preview: string
  /** ISO 8601 with an explicit `Z`; safe to hand straight to `new Date()`. */
  last_active: string
  message_count: number
  platform: string
  inner_voice: boolean
  model: string
  goal: string
  goal_achieved: boolean
  todo_counts: Record<string, number>
  /** Newest turn's tool-caption rate. `null` for a session that predates the
   *  counters — which must not render as 0/0, since that reads as a failure. */
  captions: { total: number; captioned: number } | null
}

export interface RecentSessions {
  sessions: RecentSession[]
}

export interface SubagentRun {
  run_id: string
  subagent_type: string
  description: string
  prompt_preview: string
  parent_session_id: string
  session_id: string
  model: string
  max_turns: number
  started_at: number
  finished_at: number | null
  elapsed_s: number
  status: string
  turns: number
  stop_reason: string
  error: string
  response_chars: number
  tool_call_count: number
  tool_counts: Record<string, number>
  last_tool: string
}

export interface BackgroundTask {
  task_id: string
  session_id: string
  description: string
  command: string
  // running | completed | failed | killed
  status: string
  started_at: number
  finished_at: number | null
  exit_code: number | null
  // Time since start while running; total duration once finished.
  elapsed_s: number
  output_path: string
}

export interface AgentState {
  subagents: { active: SubagentRun[]; active_count: number; recent: SubagentRun[] }
  background_tasks: {
    active: BackgroundTask[]
    active_count: number
    // Finished tasks, most recently finished first. Older builds of the
    // aggregator don't send this — treat it as optional.
    recent?: BackgroundTask[]
  }
  tools: number
}

export interface ServiceRow {
  id: string
  name: string
  group: 'infra' | 'lloyd'
  port: number | null
  state: string
  sub_state: string
  port_healthy: boolean | null
  health: 'healthy' | 'degraded' | 'stopped' | string
  uptime: string | null
}

export interface UsageBucket {
  bucket: string
  requests: number
  input_tokens: number
  output_tokens: number
  cache_create: number
  cache_read: number
  cost_usd: number
}

export interface DashboardUsage {
  last_hour: Record<string, number>
  last_24h: Record<string, number>
  last_7d: Record<string, number>
  daily: UsageBucket[]
  by_model_24h: Array<{ model: string } & Record<string, number>>
  /** Prefix-cache misses on long re-admissions (app/prefix_miss.py). Absent
   *  on a backend that predates them. */
  prefix_misses_1h?: PrefixMissSummary
  prefix_misses_24h?: PrefixMissSummary
}

export interface PrefixMissSummary {
  turns: number
  /** Rows that carry the measurement at all. Older rows, and turns whose
   *  cache_read never read non-zero, are NULL and not counted here. */
  turns_measured: number
  turns_with_misses: number
  prefix_misses: number
  reprefill_tokens: number
  worst_turn_reprefill: number
}

/** The worker pool's KV budget gate (workers/pool.py). */
export interface KvGateState {
  enabled: boolean
  max_kv_usage: number
  engaged: boolean
  engaged_since: string | null
  engagements: number
  kv_usage: number | null
  held_sources: string[]
}

export interface WorkersState {
  enabled: boolean
  pool: { running: boolean; paused?: boolean; slots?: number; in_flight_count?: number
          in_flight?: Record<string, { source: string; kind: string; started_at: string }>
          kv_gate?: KvGateState }
  depth_by_source: Record<string, Record<string, number>>
  by_state: Record<string, number>
  open_total: number
  poisoned_total: number
  sources: Array<{
    name: string; enabled: boolean; open: number; running: number
    completed: number; failed: number; poisoned: number
  }>
  recent_runs: Array<{
    run_id: string; source: string; status: string; started_at: string
    duration_seconds: number | null; summary: string
  }>
}

export interface AutonomyTaskRow {
  name: string
  status: string
  frequency: string
  next_run: string
  last_run: string
  /** Why the scheduler will not dispatch this task right now — "paused",
   *  "outside hours 00-04,23", "waiting on #42". Null means nothing is
   *  holding it, so a past-due task really is late. */
  blocked: string | null
}

export interface AutonomyState {
  total: number
  by_status: Record<string, number>
  /** Past due with nothing holding them back. These are the real misses. */
  overdue: AutonomyTaskRow[]
  overdue_count: number
  /** Past due, but deliberately held — paused, outside their hours, waiting
   *  on a dependency. Normal, and separated so it cannot drown `overdue`. */
  held: AutonomyTaskRow[]
  held_count: number
  upcoming: AutonomyTaskRow[]
  failing: AutonomyTaskRow[]
  running: Array<{ job_id: string; kind: string; started_at: string; elapsed_s: number | null }>
  running_count: number
  /** "autonomy" when the split used the scheduler's own predicates,
   *  "naive" when it could not be imported and everything past due is
   *  reported as overdue. A downgrade must never look like success. */
  classifier?: 'autonomy' | 'naive'
}

export interface BacklogState {
  total: number
  by_status: Record<string, number>
  by_board: Array<{ board: string; open: number; total: number }>
  open_total: number
  recent_open: Array<{ name: string; status: string; board: string; mtime: number }>
}

export interface DashboardSnapshot {
  host: HostMetrics | SectionError
  vllm: VllmEngine[] | SectionError
  primary: PrimaryState | SectionError
  recent: RecentSessions | SectionError
  agents: AgentState | SectionError
  services: { services: ServiceRow[]; unhealthy: string[]; total: number } | SectionError
  workers: WorkersState | SectionError
  autonomy: AutonomyState | SectionError
  backlog: BacklogState | SectionError
  usage: DashboardUsage | SectionError
  timestamp: number
}

export function sectionOk<T>(section: T | SectionError | undefined): section is T {
  return !!section && !(typeof section === 'object' && 'error' in (section as object))
}

/**
 * The message to render when `sectionOk` said no.
 *
 * A section can be missing outright, not just failed: a tab left open
 * across a backend restart polls the new build with the old snapshot
 * shape, and `section.error` on an undefined section throws inside
 * render — which blanks the whole page. A dashboard is most useful when
 * something is broken, so it must not be the second thing to break.
 */
export function sectionError(section: unknown): string {
  if (section && typeof section === 'object' && 'error' in section) {
    return String((section as SectionError).error)
  }
  return 'no data — is the backend running this build?'
}

export const dashboardApi = {
  get: (): Promise<DashboardSnapshot> =>
    fetch(`${API_BASE}/dashboard`).then(r => {
      if (!r.ok) throw new Error(`dashboard ${r.status}`)
      return r.json()
    }),
}
