/**
 * pending-questions.ts — Manages the pending question store for interactive mode.
 *
 * When a subagent hits a canUseTool gate (permission or clarification), a
 * PendingQuestion is created with a blocking Promise. The promise resolves
 * when Lloyd calls cc_respond, unblocking the canUseTool callback so the
 * SDK can continue execution.
 *
 * Auto-timeout prevents indefinite hangs — questions auto-deny after timeoutMs.
 */

import { randomUUID } from "node:crypto";
import type { PendingQuestion, QuestionAnswer, QuestionType, PendingQuestionInfo } from "./types.js";

/** In-memory store of all pending questions across all instances */
const pendingQuestions = new Map<string, PendingQuestion>();

/** Timer handles for auto-timeout cleanup */
const timeoutHandles = new Map<string, ReturnType<typeof setTimeout>>();

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Create a pending question and return a promise that blocks until answered.
 * The caller (canUseTool callback) awaits this promise.
 */
export function createQuestion(opts: {
  instanceId: string;
  type: QuestionType;
  agentId?: string;
  toolName?: string;
  toolInput?: Record<string, unknown>;
  question: string;
  options?: string[];
  timeoutMs?: number;
}): { question: PendingQuestion; promise: Promise<QuestionAnswer> } {
  const id = randomUUID().slice(0, 12);
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  let resolve: (answer: QuestionAnswer) => void;
  const promise = new Promise<QuestionAnswer>((res) => {
    resolve = res;
  });

  const q: PendingQuestion = {
    id,
    instanceId: opts.instanceId,
    type: opts.type,
    agentId: opts.agentId,
    toolName: opts.toolName,
    toolInput: opts.toolInput,
    question: opts.question,
    options: opts.options,
    createdAt: Date.now(),
    timeoutMs,
    status: "pending",
    _resolve: resolve!,
  };

  pendingQuestions.set(id, q);

  // Auto-timeout — deny after timeoutMs
  const handle = setTimeout(() => {
    if (q.status === "pending") {
      q.status = "timeout";
      q._resolve?.({ action: "deny", text: "Timed out waiting for user response" });
      pendingQuestions.delete(id);
      timeoutHandles.delete(id);
    }
  }, timeoutMs);

  timeoutHandles.set(id, handle);

  return { question: q, promise };
}

/**
 * Resolve a pending question with the user's answer.
 * Called by cc_respond when Lloyd relays the user's response.
 * Returns true if the question was found and resolved.
 */
export function resolveQuestion(questionId: string, answer: QuestionAnswer): boolean {
  const q = pendingQuestions.get(questionId);
  if (!q || q.status !== "pending") return false;

  q.status = "answered";
  q.answer = answer.text;
  q._resolve?.(answer);

  // Clean up
  pendingQuestions.delete(questionId);
  const handle = timeoutHandles.get(questionId);
  if (handle) {
    clearTimeout(handle);
    timeoutHandles.delete(questionId);
  }

  return true;
}

/**
 * List pending questions, optionally filtered by instance ID.
 */
export function listPendingQuestions(instanceId?: string): PendingQuestionInfo[] {
  const now = Date.now();
  const all = Array.from(pendingQuestions.values());
  const filtered = instanceId ? all.filter((q) => q.instanceId === instanceId) : all;

  return filtered
    .filter((q) => q.status === "pending")
    .map((q) => ({
      id: q.id,
      type: q.type,
      agentId: q.agentId,
      toolName: q.toolName,
      question: q.question,
      options: q.options,
      createdAt: q.createdAt,
      elapsedMs: now - q.createdAt,
      timeoutMs: q.timeoutMs,
    }));
}

/**
 * Cancel all pending questions for an instance (on abort).
 * Resolves all promises with deny to prevent leaked/hanging promises.
 */
export function cancelAllForInstance(instanceId: string): number {
  let count = 0;
  for (const [id, q] of pendingQuestions) {
    if (q.instanceId === instanceId && q.status === "pending") {
      q.status = "cancelled";
      q._resolve?.({ action: "deny", text: "Instance cancelled" });
      pendingQuestions.delete(id);
      const handle = timeoutHandles.get(id);
      if (handle) {
        clearTimeout(handle);
        timeoutHandles.delete(id);
      }
      count++;
    }
  }
  return count;
}
