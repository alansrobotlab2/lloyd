export interface TurnStats {
  input_tokens: number
  output_tokens: number
  cache_create: number
  cache_read: number
  cost_usd: number | null
  duration_ms: number | null
  num_turns: number | null
  model: string
}

export interface MessageEntry {
  id: string
  role: 'user' | 'assistant' | 'tool'
  content: Array<{ type: 'text'; text: string }>
  timestamp: string
  session_key?: string
  model?: string
  reasoning?: string
  stats?: TurnStats
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
}

export interface Session {
  id: string
  session_key: string
  display_name?: string
  preview?: string
  last_active: string
  platform?: string
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
  emoji?: string
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

export interface ToolEntry {
  name: string
  label: string
  description: string
  enabled: boolean
}

export interface McpTool {
  name: string
  description: string
  enabled: boolean
}

export interface McpServer {
  name: string
  label: string
  description: string
  enabled: boolean
  tools: McpTool[]
  error?: string
}

export interface ToolsData {
  builtin: ToolEntry[]
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
}

export interface EntityGraphNode {
  id: string;
  label: string;
  type: string;
  factCount?: number;
}

export interface EntityGraphEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

export interface EntityGraphData {
  nodes: EntityGraphNode[];
  edges: EntityGraphEdge[];
}

// ── Usage types ─────────────────────────────────────────────────────────

export interface UsageSummary {
  requests: number
  input_tokens: number
  output_tokens: number
  cache_create: number
  cache_read: number
  cost_usd: number
  duration_ms: number
  duration_api_ms: number
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

export interface UsageAllocation {
  tokens: number
  cost_usd: number
}

export interface UsageRateLimits {
  '5h-utilization'?: number
  '5h-status'?: string
  '5h-reset'?: number
  '7d-utilization'?: number
  '7d-status'?: string
  '7d-reset'?: number
  'fallback-percentage'?: number
  'overage-status'?: string
  [key: string]: string | number | undefined
}

export interface UsagePing {
  rate_limits: UsageRateLimits
  local_5h: UsageSummary
  local_7d: UsageSummary
  pinged_at: string
  error?: string
}

export interface UsageWindows {
  four_hour: UsageSummary
  seven_day: UsageSummary
  allocations: {
    '4h': UsageAllocation
    '7d': UsageAllocation
  }
}

export interface UsageModelBreakdown {
  model: string
  requests: number
  input_tokens: number
  output_tokens: number
  cache_create: number
  cache_read: number
  cost_usd: number
}

export interface UsageRecord {
  id: number
  ts: string
  session_id: string
  model: string
  input_tokens: number
  output_tokens: number
  cache_create: number
  cache_read: number
  cost_usd: number | null
  duration_ms: number | null
  num_turns: number | null
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
  skill_path: string | null;
  model: string | null;
  timeout_seconds: number | null;
  max_retries: number | null;
  preferred_hours: string | null;
  frequency: string | null;
  cron_id: string | null;
  last_run: string | null;
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
      onToolStart?: (callId: string, name: string, args: Record<string, unknown>) => void
      onToolComplete?: (callId: string, name: string, result: string) => void
      onToolProgress?: (name: string, preview: string) => void
      onTextDelta?: (text: string) => void
      onDone?: (response: string, sessionId: string, stats?: TurnStats) => void
      onError?: (detail: string) => void
    },
    model?: string,
  ): AbortController {
    const controller = new AbortController()
    fetch(`${API_BASE}/message/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, client_id: clientId, session_id: sessionId, ...(model ? { model } : {}) }),
      signal: controller.signal,
    }).then(async (response) => {
      if (!response.ok || !response.body) {
        callbacks.onError?.(`HTTP ${response.status}`)
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
              case 'tool_start': callbacks.onToolStart?.(payload.call_id, payload.name, payload.args); break
              case 'tool_complete': callbacks.onToolComplete?.(payload.call_id, payload.name, payload.result); break
              case 'tool_progress': callbacks.onToolProgress?.(payload.name, payload.preview); break
              case 'text_delta': callbacks.onTextDelta?.(payload.text); break
              case 'done': callbacks.onDone?.(payload.response, payload.session_id, payload.stats); break
              case 'error': callbacks.onError?.(payload.detail); break
            }
          } catch { /* skip malformed */ }
        }
      }
    }).catch((err) => {
      if (err.name !== 'AbortError') callbacks.onError?.(err.message)
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
      builtin: Array.isArray(d?.builtin) ? d.builtin : [],
      servers: Array.isArray(d?.servers) ? d.servers : [],
    }))
  },

  async toolToggle(
    payload:
      | { type: 'server'; server: string; enabled: boolean }
      | { type: 'tool'; server: string; tool: string; enabled: boolean }
      | { type: 'builtin'; tool: string; enabled: boolean }
  ): Promise<void> {
    await fetch(`${API_BASE}/tool-toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
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

  // Usage
  usagePing: (): Promise<UsagePing> =>
    fetch(`${API_BASE}/usage/ping`).then(r => r.json()),
  usageWindows: (): Promise<UsageWindows> =>
    fetch(`${API_BASE}/usage/windows`).then(r => r.json()),
  usageHistory: (period: '4h' | '24h' | '7d' | '30d' = '4h'): Promise<{ period: string; buckets: UsageBucket[] }> =>
    fetch(`${API_BASE}/usage/history?period=${period}`).then(r => r.json()),
  usageModels: (hours?: number, days?: number): Promise<{ models: UsageModelBreakdown[] }> => {
    const params = new URLSearchParams()
    if (hours) params.set('hours', String(hours))
    if (days) params.set('days', String(days))
    return fetch(`${API_BASE}/usage/models?${params}`).then(r => r.json())
  },
  usageRecent: (limit = 20): Promise<{ records: UsageRecord[] }> =>
    fetch(`${API_BASE}/usage/recent?limit=${limit}`).then(r => r.json()),

  // Autonomy Scheduler
  autonomySchedulerStatus: (): Promise<{ enabled: boolean; running: boolean; last_tick: number; current_task_id: number | null }> =>
    fetch(`${API_BASE}/autonomy/scheduler/status`).then(r => r.json()),
  autonomySchedulerEnable: (): Promise<{ enabled: boolean }> =>
    fetch(`${API_BASE}/autonomy/scheduler/enable`, { method: 'POST' }).then(r => r.json()),
  autonomySchedulerDisable: (): Promise<{ enabled: boolean }> =>
    fetch(`${API_BASE}/autonomy/scheduler/disable`, { method: 'POST' }).then(r => r.json()),
}
