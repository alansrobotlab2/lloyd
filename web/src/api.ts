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
  role: 'user' | 'assistant' | 'tool' | 'subliminal'
  content: Array<{ type: 'text'; text: string }>
  timestamp: string
  session_key?: string
  model?: string
  reasoning?: string
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
  preview?: string
  last_active: string
  platform?: string
  // Inner Voice: A/B linkage tag and critic opt-in flag.
  experiment_id?: string | null
  inner_voice?: boolean
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
  created_at: string | null
  updated_at: string | null
  message_count: number
  evaluate_user_turns?: boolean
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

export interface EntitySummary {
  name: string;
  factCount: number;
  categories: string[];
}

export interface EntitiesListData {
  entities: EntitySummary[];
  total: number;
}

export interface EntityFact {
  fact: string;
  confidence: number;
  category: string;
  event_date?: string | null;
  id?: string;
}

export interface EntityRelationship {
  target: string;
  type: string;
  score: number;
}

export interface EntityDetailData {
  name: string;
  facts: EntityFact[];
  relationships: EntityRelationship[];
  definition?: string | null;
  summary?: string | null;
}

export interface EntityGraphNode {
  id: string;
  label: string;
  type: string;
  factCount?: number;
  definition?: string | null;
}

export interface EntityGraphEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
  bidirectional?: boolean;
}

export interface EntityGraphData {
  nodes: EntityGraphNode[];
  edges: EntityGraphEdge[];
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
  tags: string[];
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

export interface ActiveProc {
  pid: number
  sdk_session_id: string | null
  session_id: string | null
  model: string | null
  preview: string
  created_at: string | null
  streaming: boolean
}

const API_BASE = '/api'

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
      onToolStart?: (callId: string, name: string, args: Record<string, unknown>, contextTokens?: number) => void
      onToolComplete?: (callId: string, name: string, result: string) => void
      onToolProgress?: (name: string, preview: string) => void
      onTextDelta?: (text: string) => void
      onThinkingDelta?: (text: string) => void
      onThinkingDone?: (fullText: string) => void
      onDone?: (response: string, sessionId: string, stats?: TurnStats, reasoning?: string, cancelled?: boolean) => void
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
              case 'tool_start': callbacks.onToolStart?.(payload.call_id, payload.name, payload.args, payload.context_tokens); break
              case 'tool_complete': callbacks.onToolComplete?.(payload.call_id, payload.name, payload.result); break
              case 'tool_progress': callbacks.onToolProgress?.(payload.name, payload.preview); break
              case 'text_delta': callbacks.onTextDelta?.(payload.text); break
              case 'thinking_delta': callbacks.onThinkingDelta?.(payload.text); break
              case 'thinking_done': callbacks.onThinkingDone?.(payload.text); break
              case 'done': callbacks.onDone?.(payload.response, payload.session_id, payload.stats, payload.reasoning, payload.cancelled); break
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

  async backlogUpdateTask(id: number, updates: Record<string, any>): Promise<void> {
    await fetch(`${API_BASE}/backlog/task-update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, ...updates }),
    })
  },

  async backlogDeleteTask(id: number): Promise<void> {
    await fetch(`${API_BASE}/backlog/task-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    })
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
    board_id?: number
    status?: string
    tags?: string[]
    priority?: string
  }): Promise<void> {
    await fetch(`${API_BASE}/backlog/task-create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
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
  memoryRead: (path: string): Promise<MemoryReadResult> =>
    fetch(`${API_BASE}/memory/read?path=${encodeURIComponent(path)}`).then(r => r.json()),
  async memorySave(path: string, content: string, frontmatter?: Record<string, unknown>): Promise<{ ok: boolean }> {
    const res = await fetch(`${API_BASE}/memory/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content, frontmatter }),
    })
    return res.json()
  },

  // Entity / knowledge graph
  entityList: (limit = 500): Promise<EntitiesListData> =>
    fetch(`${API_BASE}/entities?limit=${limit}`).then(r => r.json()),
  entityDetail: (name: string): Promise<EntityDetailData> =>
    fetch(`${API_BASE}/entity?name=${encodeURIComponent(name)}`).then(r => r.json()),
  entityGraph: (): Promise<EntityGraphData> =>
    fetch(`${API_BASE}/entity-graph`).then(r => r.json()),

  // Autonomy
  autonomyTasks: (): Promise<{ tasks: AutonomyTask[] }> =>
    fetch(`${API_BASE}/autonomy/tasks`).then(r => r.json()),
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
    pool?: { running: boolean; paused: boolean; slots: number; in_flight: Record<string, any>; in_flight_count: number }
    depth?: Record<string, Record<string, number>>
    sources?: Array<{ name: string; enabled: boolean; interval_seconds?: number; max_inflight?: number; depth?: Record<string, number> }>
  }> => fetch(`${API_BASE}/workers/status`).then(r => r.json()),
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
}
