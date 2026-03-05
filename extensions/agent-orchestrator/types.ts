/**
 * types.ts — Shared types for the agent-orchestrator plugin.
 */

export type InstanceStatus = "running" | "complete" | "error" | "aborted";

// ── Pending Questions (interactive mode) ────────────────────────────────

export type QuestionType = "permission" | "clarification" | "escalation";

export interface PendingQuestion {
  id: string;
  instanceId: string;
  type: QuestionType;
  agentId?: string;
  toolName?: string;
  toolInput?: Record<string, unknown>;
  question: string;
  options?: string[];
  createdAt: number;
  timeoutMs: number;
  status: "pending" | "answered" | "timeout" | "cancelled";
  answer?: string;
  _resolve?: (answer: QuestionAnswer) => void;
}

export interface QuestionAnswer {
  action: "allow" | "deny" | "answer";
  text?: string;
  updatedInput?: Record<string, unknown>;
}

/** Serializable pending question info (for cc_status / Mission Control) */
export interface PendingQuestionInfo {
  id: string;
  type: QuestionType;
  agentId?: string;
  toolName?: string;
  question: string;
  options?: string[];
  createdAt: number;
  elapsedMs: number;
  timeoutMs: number;
}

export interface CcInstance {
  id: string;
  type: "orchestrate" | "spawn";
  status: InstanceStatus;
  task: string;
  pipeline?: string;
  agent?: string;
  /** Plan-only mode — orchestrator returns plan without executing */
  planOnly?: boolean;
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
  /** Whether this instance uses interactive mode (approval gates + clarifications) */
  interactive?: boolean;
  /** Pending questions awaiting user response */
  pendingQuestions: PendingQuestion[];
  /** Recent message log for Mission Control (ring buffer, last N) */
  recentMessages: InstanceMessage[];
  /** AbortController for cancellation */
  _abort?: AbortController;
  /** The query handle for cleanup */
  _query?: any;
}

export interface InstanceMessage {
  ts: number;
  type: "text" | "tool_use" | "tool_result" | "subagent_start" | "subagent_end"
      | "error" | "task_progress" | "question_pending" | "question_answered";
  agent?: string;
  content: string;
  questionId?: string;
}

/** What cc_status returns to Lloyd */
export interface InstanceStatusResponse {
  id: string;
  type: string;
  status: InstanceStatus;
  task: string;
  pipeline?: string;
  agent?: string;
  interactive?: boolean;
  elapsedMs: number;
  costUsd: number;
  turns: number;
  activity?: string;
  resultPreview?: string;
  error?: string;
  pendingQuestions?: PendingQuestionInfo[];
}

/** What Mission Control gets from /api/mc/cc-instances */
export interface McInstanceInfo {
  id: string;
  type: string;
  status: InstanceStatus;
  task: string;
  pipeline?: string;
  agent?: string;
  interactive?: boolean;
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
