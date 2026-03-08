const BASE = "/api/mc";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export interface Stats {
  totalInput: number;
  totalOutput: number;
  totalCacheRead: number;
  totalSessions: number;
}

export interface UsageChartData {
  range: string;
  bucketMs: number;
  data: Array<{ ts: number; input: number; output: number; cacheRead: number }>;
}

export interface RunEntry {
  ts: string;
  sessionId: string | null;
  model: string;
  totalMs: number;
  llmMs: number;
  toolMs: number;
  roundTrips: number;
  toolCallCount: number;
  success: boolean;
}

export interface ToolCallEntry {
  ts: string;
  sessionId: string | null;
  toolName: string;
  durationMs: number;
}

export interface ApiCallsData {
  runs: RunEntry[];
  toolCalls: ToolCallEntry[];
}

export interface SessionSummary {
  sessionKey: string;
  sessionId?: string;
  input?: number;
  output?: number;
  cacheRead?: number;
  messageCount?: number;
  lastActivity: string;
  model: string;
  summary?: string;
  source?: string;
  peer?: string;
}

export interface MessageEntry {
  id: string;
  timestamp: string;
  role: "user" | "assistant" | "toolResult";
  content: Array<{ type: string; text?: string; thinking?: string; id?: string; name?: string; arguments?: Record<string, unknown> }>;
  model?: string;
  usage?: { input: number; output: number; cacheRead: number; totalTokens: number };
  toolCallId?: string;
  toolName?: string;
  isError?: boolean;
  hasThinking?: boolean;
  toolCallCount?: number;
  durationMs?: number;
}

export interface HealthData {
  gateway: string;
  timestamp: string;
  auth: Record<string, { errorCount: number; lastUsed: string | null }>;
  cron: Array<{
    id: string;
    name: string;
    enabled: boolean;
    lastStatus: string;
    consecutiveErrors: number;
    nextRunAt: string | null;
  }>;
}

// ── Services types ─────────────────────────────────────────────────────

export interface ServiceStatus {
  id: string;
  name: string;
  unit: string;
  port: number;
  systemdState: "active" | "inactive" | "failed" | "unknown";
  portHealthy: boolean;
  health: "healthy" | "degraded" | "stopped" | "unknown";
}

export interface ServicesData {
  services: ServiceStatus[];
  timestamp: string;
}

export interface ServiceDetail {
  id: string;
  name: string;
  unit: string;
  port: number;
  pid: number | null;
  memory: string | null;
  cpu: string | null;
  tasks: string | null;
  activeSince: string | null;
  logLines: string[];
  rawStatus: string;
}

// ── Lloyd Services types ─────────────────────────────────────────────────

export interface LloydServiceUnit {
  id: string;
  unit: string;
  name: string;
  activeState: "active" | "inactive" | "failed" | "unknown";
  subState: string;
  port: number | null;
  portHealthy: boolean | null;
  uptime: string | null;
  health: "healthy" | "degraded" | "stopped" | "unknown";
}

export interface LloydServicesData {
  services: LloydServiceUnit[];
  timestamp: string;
}

export interface LloydServiceDetail {
  unit: string;
  name: string;
  pid: number | null;
  memory: string | null;
  cpu: string | null;
  tasks: string | null;
  activeSince: string | null;
  logLines: string[];
  rawStatus: string;
}

// ── Tools types ──────────────────────────────────────────────────────────

export interface ToolEntry {
  name: string;
  enabled: boolean;
}

export interface ToolGroupData {
  source: string;
  tools: ToolEntry[];
}

// ── Skills types ──────────────────────────────────────────────────────────

export interface SkillInfo {
  name: string;
  description: string;
  emoji?: string;
  requires?: { bins?: string[]; env?: string[]; config?: string[]; anyBins?: string[] };
  os?: string[];
  enabled: boolean;
  configured: boolean;
  location: string;
}

export interface SkillsData {
  workspace: SkillInfo[];
  bundled: SkillInfo[];
}

// ── Agents types ──────────────────────────────────────────────────────────

export interface WorkspaceFile {
  name: string;
  key: string;
  content: string | null;
}

export interface ToolGroupInfo {
  source: string;
  tools: string[];
}

export interface AgentInfo {
  id: string;
  name: string;
  avatar: string | null;
  identity: string | null;
  primaryModel: string | null;
  modelFallbacks: string[];
  sessions: { total: number; active: number };
  modelCount: number;
  enabledModels: number;
  disabledTools: number;
  toolsAllow: string[] | null;
  skills: string[] | null;
  maxConcurrent: number | null;
  subagentMaxConcurrent: number | null;
  workspace: Record<string, string | null>;
  workspaceFiles: WorkspaceFile[];
  workspacePath: string;
}

export interface AgentsData {
  agents: AgentInfo[];
  workspace: Record<string, string | null>;
  defaults: Record<string, any>;
  allToolGroups: ToolGroupInfo[];
  allSkillNames: string[];
}

// ── Agent Status types ────────────────────────────────────────────────────

export interface SubagentRunInfo {
  runId: string;
  childSessionKey: string;
  requesterSessionKey: string;
  task: string;
  label?: string;
  model?: string;
  spawnMode?: string;
  createdAt: number;
  startedAt?: number;
  endedAt?: number;
  outcome?: string;
  endedReason?: string;
  durationMs?: number;
}

export interface AgentStatusData {
  mainAgent: {
    state: "idle" | "processing" | "waiting";
    reason?: string;
    queueDepth: number;
    lastUpdated: number;
  };
  activity: {
    type: "tool_call" | "llm_thinking" | "idle";
    label: string;
    detail: string | null;
    elapsedMs: number;
  };
  heartbeat: { active: number; waiting: number; queued: number } | null;
  subagents: {
    active: SubagentRunInfo[];
    recentCompleted: SubagentRunInfo[];
  };
  timestamp: string;
}

// ── Backlog types ───────────────────────────────────────────────────────

export interface CallLogEntry {
  ts: string;
  type: "tool" | "llm";
  // Tool-specific
  toolName?: string;
  args?: Record<string, unknown>;
  isError?: boolean;
  resultPreview?: string;
  // LLM-specific
  model?: string;
  provider?: string;
  inputTokens?: number;
  outputTokens?: number;
  cost?: number;
  hasToolCalls?: boolean;
}

export interface BacklogBoard {
  id: number;
  name: string;
  icon: string;
  color: string;
  tasks_count: number;
}

export interface BacklogTask {
  id: number;
  name: string;
  description: string;
  priority: string;
  status: string;
  blocked: boolean;
  tags: string[];
  completed: boolean;
  due_date: string | null;
  position: number;
  assigned_to_agent: boolean;
  board_id: number;
  url: string;
  created_at: string;
  updated_at: string;
}

// ── Command types ─────────────────────────────────────────────────────

export interface CommandInfo {
  name: string;
  description: string;
  category: string;
  acceptsArgs: boolean;
  source: "built-in" | "plugin" | "skill";
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

// ── Vault Graph types ────────────────────────────────────────────────

export interface VaultGraphNode {
  id: string;
  label: string;
  type: string;
  tags: string[];
  folder: string;
}

export interface VaultGraphEdge {
  source: string;
  target: string;
  kind: string;
}

export interface VaultGraphData {
  nodes: VaultGraphNode[];
  edges: VaultGraphEdge[];
}

export interface TagGraphNode {
  id: string;
  label: string;
  count: number;
}

export interface TagGraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface TagGraphData {
  nodes: TagGraphNode[];
  edges: TagGraphEdge[];
}

export const api = {
  // Services
  services: () => fetchJson<ServicesData>("/services"),
  serviceDetail: (id: string) => fetchJson<ServiceDetail>(`/services/detail?id=${encodeURIComponent(id)}`),
  serviceAction: async (serviceId: string, action: "start" | "stop" | "restart") => {
    const res = await fetch(`${BASE}/services/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ serviceId, action }),
    });
    return res.json();
  },

  lloydServices: () => fetchJson<LloydServicesData>("/lloyd-services"),
  lloydServiceAction: async (serviceId: string, action: "start" | "stop" | "restart") => {
    const res = await fetch(`${BASE}/services/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ serviceId, action }),
    });
    return res.json();
  },
  lloydServiceDetail: (unit: string) => fetchJson<LloydServiceDetail>(`/lloyd-services/detail?unit=${encodeURIComponent(unit)}`),

  stats: () => fetchJson<Stats>("/stats"),
  usageChart: (range = "7d") => fetchJson<UsageChartData>(`/usage-chart?range=${range}`),
  apiCalls: () => fetchJson<ApiCallsData>("/api-calls"),
  sessions: () => fetchJson<{ sessions: SessionSummary[] }>("/sessions"),
  sessionMessages: (sessionKey: string, includeTools?: boolean) =>
    fetchJson<{ sessionKey: string; messages: MessageEntry[] }>(`/session-messages?sessionKey=${encodeURIComponent(sessionKey)}${includeTools ? "&tools=1" : ""}`),
  chat: async (message: string, sessionKey?: string) => {
    const res = await fetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, ...(sessionKey ? { sessionKey } : {}) }),
    });
    return res.json();
  },
  health: () => fetchJson<HealthData>("/health"),
  models: () => fetchJson<{ providers: Record<string, any> }>("/models"),
  modelToggle: async (provider: string, modelId: string, enabled: boolean) => {
    const res = await fetch(`${BASE}/model-toggle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, modelId, enabled }),
    });
    return res.json();
  },

  // Tools
  tools: () => fetchJson<{ groups: ToolGroupData[] }>("/tools"),
  toolToggle: async (toolName: string, enabled: boolean) => {
    const res = await fetch(`${BASE}/tool-toggle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ toolName, enabled }),
    });
    return res.json();
  },

  // Skills
  skills: () => fetchJson<SkillsData>("/skills"),
  skillToggle: async (skillName: string, enabled: boolean) => {
    const res = await fetch(`${BASE}/skill-toggle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skillName, enabled }),
    });
    return res.json();
  },
  skillContent: (name: string) => fetchJson<{ content: string; location: string }>(`/skill-content?name=${encodeURIComponent(name)}`),
  skillContentSave: async (name: string, content: string) => {
    const res = await fetch(`${BASE}/skill-content`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skillName: name, content }),
    });
    return res.json();
  },

  // Agents
  agents: () => fetchJson<AgentsData>("/agents"),
  agentStatus: () => fetchJson<AgentStatusData>("/agent-status"),
  agentToolsUpdate: async (agentId: string, tools: string[] | null) => {
    const res = await fetch(`${BASE}/agent-tools-update`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agentId, tools }),
    });
    return res.json();
  },
  agentSkillsUpdate: async (agentId: string, skills: string[] | null) => {
    const res = await fetch(`${BASE}/agent-skills-update`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agentId, skills }),
    });
    return res.json();
  },
  agentFileSave: async (agentId: string, fileName: string, content: string) => {
    const res = await fetch(`${BASE}/agent-file-save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agentId, fileName, content }),
    });
    return res.json();
  },

  // Memory / Vault
  memoryStats: () => fetchJson<MemoryStats>("/memory/stats"),
  memorySearch: (q: string, limit = 10) =>
    fetchJson<MemorySearchResult>(`/memory/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  memoryTags: (limit = 100) => fetchJson<{ tags: TagEntry[] }>(`/memory/tags?limit=${limit}`),
  memoryBrowse: (path = "") =>
    fetchJson<MemoryBrowseResult>(`/memory/browse?path=${encodeURIComponent(path)}`),
  memoryRead: (path: string) =>
    fetchJson<MemoryReadResult>(`/memory/read?path=${encodeURIComponent(path)}`),
  memorySave: async (path: string, content: string, frontmatter?: Record<string, unknown>): Promise<{ ok: boolean }> => {
    const res = await fetch(`${BASE}/memory/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, content, frontmatter }),
    });
    if (!res.ok) throw new Error(`Save failed: ${res.status}`);
    return res.json();
  },

  // Abort the currently running agent turn
  chatAbort: async (sessionKey?: string): Promise<{ ok: boolean }> => {
    const res = await fetch(`${BASE}/chat-abort`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sessionKey ? { sessionKey } : {}),
    });
    return res.json();
  },

  // Session management
  sessionNew: async (): Promise<{ ok: boolean; sessionKey: string }> => {
    const res = await fetch(`${BASE}/session-new`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!res.ok) throw new Error(`Session new failed: ${res.status}`);
    return res.json();
  },

  // Backlog
  backlogBoards: () => fetchJson<BacklogBoard[]>("/backlog/boards"),
  backlogTasks: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return fetchJson<BacklogTask[]>(`/backlog/tasks${qs}`);
  },
  backlogUpdateTask: async (id: number, updates: Record<string, any>) => {
    const res = await fetch(`${BASE}/backlog/task-update`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, ...updates }),
    });
    return res.json();
  },
  backlogDeleteTask: async (id: number) => {
    const res = await fetch(`${BASE}/backlog/task-delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    return res.json();
  },
  backlogCreateTask: async (data: {
    name: string;
    description?: string;
    board_id?: number;
    status?: string;
    tags?: string[];
    priority?: string;
  }) => {
    const res = await fetch(`${BASE}/backlog/task-create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  // Subagent abort
  subagentAbort: async (sessionKey: string) => {
    const res = await fetch(`${BASE}/subagent-abort`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionKey }),
    });
    return res.json();
  },

  agentCallLog: (agentId: string, limit = 30): Promise<{ entries: CallLogEntry[] }> =>
    fetchJson(`/agent-call-log?agentId=${encodeURIComponent(agentId)}&limit=${limit}`),

  // Vault graph
  vaultGraph: () => fetchJson<VaultGraphData>("/vault-graph"),
  tagGraph: () => fetchJson<TagGraphData>("/tag-graph"),

  // Commands (slash command picker)
  commands: () => fetchJson<{ commands: CommandInfo[] }>("/commands"),

  // Work mode
  mode: () => fetchJson<{ currentMode: string; lastSwitchedAt: string }>("/mode"),
  modeSet: async (mode: string) => {
    const res = await fetch(`${BASE}/mode-set`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    return res.json();
  },

  // Gateway sessions (subagent visibility)
  gatewaySessions: () => fetchJson<GatewaySessionsData>("/gateway-sessions"),

  // Claude Code instances & agents (agent-orchestrator plugin)
  ccAgents: () => fetchJson<SdkAgentsData>("/cc-agents"),
  ccInstances: () => fetchJson<{ instances: CcInstanceInfo[] }>("/cc-instances"),
  ccInstanceLog: (id: string, limit = 50) =>
    fetchJson<{ id: string; status: string; messages: CcInstanceMessage[] }>(
      `/cc-instance-log?id=${encodeURIComponent(id)}&limit=${limit}`
    ),
  ccInstanceAbort: async (id: string) => {
    const res = await fetch(`${BASE}/cc-instance-abort`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    return res.json();
  },
};

// ── Gateway Session Types ─────────────────────────────────────────────────

export interface GatewaySessionInfo {
  sessionKey: string;
  sessionId?: string;
  agentId: string;
  state: "idle" | "processing" | "waiting";
  source: "gateway";
  lastUpdated: number;
  elapsedMs: number;
}

export interface GatewaySessionsData {
  sessions: GatewaySessionInfo[];
  subagentRuns: SubagentRunInfo[];
  gwSubagents: any[];
  timestamp: string;
}

// ── Claude Code Instance Types ────────────────────────────────────────────

export interface CcInstanceInfo {
  id: string;
  type: "orchestrate" | "spawn";
  status: "running" | "complete" | "error" | "aborted";
  task: string;
  pipeline?: string;
  agent?: string;
  startedAt: number;
  endedAt?: number;
  elapsedMs: number;
  costUsd: number;
  turns: number;
  budgetUsd: number;
  activity?: string;
  resultPreview?: string;
  error?: string;
}

export interface CcInstanceMessage {
  ts: number;
  type: "text" | "tool_use" | "tool_result" | "subagent_start" | "subagent_end" | "error" | "task_progress";
  agent?: string;
  content: string;
}

// ── SDK Agent Types ────────────────────────────────────────────────────────

export interface SdkAgentInfo {
  id: string;
  model: string;
  description: string;
  maxTurns: number;
  tools: string[];
  mcpTools: string[];
  hasMcp: boolean;
  avatarUrl: string;
}

export interface SdkAgentsData {
  agents: SdkAgentInfo[];
  instanceCounts: Record<string, { active: number; recent: number }>;
}
