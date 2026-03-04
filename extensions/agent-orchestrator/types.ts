/**
 * types.ts — Shared types for the agent-orchestrator plugin.
 */

export type InstanceStatus = "running" | "complete" | "error" | "aborted";

export interface CcInstance {
  id: string;
  type: "orchestrate" | "spawn";
  status: InstanceStatus;
  task: string;
  pipeline?: string;
  agent?: string;
  startedAt: number;
  endedAt?: number;
  sessionId?: string;
  costUsd: number;
  turns: number;
  budgetUsd: number;
  maxTurns: number;
  /** Last activity description (e.g., "coder: editing auth.ts") */
  activity?: string;
  /** Final result text (populated on completion) */
  resultText?: string;
  /** Error message if status === "error" */
  error?: string;
  /** Recent message log for Mission Control (ring buffer, last N) */
  recentMessages: InstanceMessage[];
  /** AbortController for cancellation */
  _abort?: AbortController;
  /** The query handle for cleanup */
  _query?: any;
}

export interface InstanceMessage {
  ts: number;
  type: "text" | "tool_use" | "tool_result" | "subagent_start" | "subagent_end" | "error";
  agent?: string;
  content: string;
}

/** What cc_status returns to Lloyd */
export interface InstanceStatusResponse {
  id: string;
  type: string;
  status: InstanceStatus;
  task: string;
  pipeline?: string;
  agent?: string;
  elapsedMs: number;
  costUsd: number;
  turns: number;
  activity?: string;
  resultPreview?: string;
  error?: string;
}

/** What Mission Control gets from /api/mc/cc-instances */
export interface McInstanceInfo {
  id: string;
  type: string;
  status: InstanceStatus;
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
