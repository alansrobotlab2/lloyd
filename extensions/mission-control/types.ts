/**
 * types.ts — Shared type definitions for Mission Control plugin modules
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { ServerResponse } from "node:http";

// ── Plugin Context (passed to all module registration functions) ─────

export interface PluginContext {
  api: OpenClawPluginApi;
  rootDir: string;
  configFile: string;
  sessionsDir: string;
  distWebDir: string;
  timingLog: string;
  routingLog: string;
  ccInstancesDir: string;
  modelsFile: string;
  toolsFile: string;
  authFile: string;
  cronJobsFile: string;
  summariesFile: string;
  agentSessionStates: Map<string, AgentSessionStatus>;
  currentActivity: { value: AgentActivity };
  lastHeartbeat: { value: { active: number; waiting: number; queued: number } | null };
  activityResetTimer: { value: ReturnType<typeof setTimeout> | null };
}

// ── Data types ──────────────────────────────────────────────────────

export interface CacheEntry<T> {
  data: T;
  ts: number;
}

export interface SessionMessage {
  type: string;
  id?: string;
  timestamp?: string;
  message?: {
    role: string;
    content: any[];
    usage?: {
      input: number;
      output: number;
      cacheRead: number;
      cacheWrite: number;
      totalTokens: number;
      cost?: { input: number; output: number; total: number };
    };
    model?: string;
    provider?: string;
  };
}

export interface TimingEntry {
  ts: string;
  event: string;
  runId: string | null;
  sessionId: string | null;
  toolName?: string;
  durationMs?: number;
  totalMs?: number;
  llmMs?: number;
  toolMs?: number;
  success?: boolean;
  error?: string | null;
  roundTrips?: number;
  toolCallCount?: number;
}

export interface RoutingEntry {
  ts: string;
  tier: string;
  reason: string;
  confidence: number;
  classifierUsed: boolean;
  latencyMs: number;
  promptLength: number;
  sessionDepth: number;
}

export interface AgentSessionStatus {
  sessionKey: string;
  sessionId?: string;
  state: "idle" | "processing" | "waiting";
  reason?: string;
  queueDepth: number;
  lastUpdated: number;
}

export interface AgentActivity {
  type: "tool_call" | "llm_thinking" | "idle";
  toolName?: string;
  model?: string;
  startedAt: number;
  sessionKey?: string;
}

export interface GwState {
  ws: any;
  ready: boolean;
  reqId: number;
  pending: Map<string, { resolve: (v: any) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }>;
  streamTextAccum: string;
  streamTtsInFlight: boolean;
}

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

export interface AgentCallLogEntry {
  ts: string;
  type: "tool" | "llm";
  toolName?: string;
  args?: Record<string, unknown>;
  isError?: boolean;
  resultPreview?: string;
  model?: string;
  provider?: string;
  inputTokens?: number;
  outputTokens?: number;
  cost?: number;
  hasToolCalls?: boolean;
}

export interface SubagentRunStatus {
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

export interface SupervisorEntry {
  name: string;
  state: string;
  pid: number | null;
  uptime: string | null;
}

export interface VaultDoc {
  path: string;
  title: string;
  type: string;
  tags: string[];
  summary: string;
  folder: string;
}

export interface CommandInfo {
  name: string;
  description: string;
  category: string;
  acceptsArgs: boolean;
  source: "built-in" | "plugin" | "skill";
}
