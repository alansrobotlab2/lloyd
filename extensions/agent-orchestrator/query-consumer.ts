/**
 * query-consumer.ts — Consumes Agent SDK query() async generators in the background.
 *
 * Tracks status, cost, messages, and fires completion notifications.
 * Persists a JSONL log per instance to logs/cc-instances/ and writes a
 * summary JSON on completion for post-mortem review.
 */

import type { CcInstance, InstanceMessage } from "./types.js";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** Max recent messages to keep per instance (ring buffer for Mission Control) */
const MAX_RECENT_MESSAGES = 100;

/** Directory for persistent instance logs */
const LOG_DIR = join(process.env.HOME || "/home/alansrobotlab", ".openclaw/logs/cc-instances");

// Ensure log directory exists at module load
try { mkdirSync(LOG_DIR, { recursive: true }); } catch {}

/** Logger interface (passed from plugin api) */
interface Logger {
  info?: (...args: any[]) => void;
  error?: (...args: any[]) => void;
}

/** Callback to fire system events when pipelines complete */
type NotifyFn = (instanceId: string, summary: string) => void;

/** Callback to push progress messages into Lloyd's session (interactive mode) */
type ProgressNotifyFn = (instanceId: string, message: string) => void;

/** Callback fired when a child sub-agent is started or finishes within an orchestrator query */
export type ChildInstanceCallback = (event: "start" | "end", info: {
  parentId: string;
  agent: string;
  taskDescription: string;
  taskId?: string;
  status?: string;
  summary?: string;
}) => void;

/**
 * Append a line to the instance's JSONL log file.
 */
function logToFile(instance: CcInstance, entry: Record<string, any>): void {
  try {
    const line = JSON.stringify({ ts: new Date().toISOString(), instanceId: instance.id, ...entry }) + "\n";
    appendFileSync(join(LOG_DIR, `${instance.id}.jsonl`), line);
  } catch {
    // Best-effort logging — don't crash the pipeline
  }
}

/**
 * Write a completion summary JSON for the instance.
 */
function writeSummary(instance: CcInstance): void {
  try {
    const summary = {
      id: instance.id,
      type: instance.type,
      status: instance.status,
      task: instance.task,
      pipeline: instance.pipeline,
      agent: instance.agent,
      startedAt: new Date(instance.startedAt).toISOString(),
      endedAt: instance.endedAt ? new Date(instance.endedAt).toISOString() : null,
      elapsedMs: (instance.endedAt || Date.now()) - instance.startedAt,
      costUsd: Math.round(instance.costUsd * 10000) / 10000,
      turns: instance.turns,
      budgetUsd: instance.budgetUsd,
      error: instance.error || null,
      resultPreview: instance.resultText ? instance.resultText.slice(0, 2000) : null,
    };
    writeFileSync(join(LOG_DIR, `${instance.id}.summary.json`), JSON.stringify(summary, null, 2) + "\n");
  } catch {
    // Best-effort
  }
}

/**
 * Consume an Agent SDK query() async generator in the background.
 * Updates the CcInstance in-place as messages arrive.
 */
export async function consumeQuery(
  instance: CcInstance,
  queryIter: AsyncGenerator<any, void>,
  logger: Logger,
  notify: NotifyFn,
  notifyProgress?: ProgressNotifyFn,
  onChildInstance?: ChildInstanceCallback,
): Promise<void> {
  // Track children started within this query to detect orphans on completion
  const activeChildren = new Map<string, { agent: string; taskDescription: string; taskId?: string }>();

  const wrappedChildCallback: ChildInstanceCallback | undefined = onChildInstance ? (event, info) => {
    if (event === "start") {
      // Track by agentKey (parentId:agent) since taskId correlation is unreliable
      const key = info.taskId || `${info.parentId}:${info.agent}`;
      activeChildren.set(key, { agent: info.agent, taskDescription: info.taskDescription, taskId: info.taskId });
    } else if (event === "end") {
      const key = info.taskId || `${info.parentId}:${info.agent}`;
      activeChildren.delete(key);
    }
    // Always forward to the real callback
    onChildInstance(event, info);
  } : undefined;

  // Log instance start
  logToFile(instance, {
    event: "start",
    type: instance.type,
    task: instance.task,
    pipeline: instance.pipeline,
    agent: instance.agent,
    budgetUsd: instance.budgetUsd,
    maxTurns: instance.maxTurns,
  });

  try {
    for await (const message of queryIter) {
      // Track session ID from init message
      if (message.type === "system" && message.subtype === "init" && message.session_id) {
        instance.sessionId = message.session_id;
        logToFile(instance, { event: "session_init", sessionId: message.session_id });
      }

      // Process based on message type
      if (message.type === "assistant" && message.message) {
        processAssistantMessage(instance, message, wrappedChildCallback);
      } else if (message.type === "result") {
        processResultMessage(instance, message);
      } else if (message.type === "error") {
        const errorContent = message.error?.message || message.message || "Unknown error";
        pushMessage(instance, { ts: Date.now(), type: "error", content: errorContent });
        logToFile(instance, { event: "error", content: errorContent });
      } else if (message.type === "system") {
        // Task lifecycle messages — subagent progress tracking
        processSystemMessage(instance, message, logger, notifyProgress, wrappedChildCallback);
      }
    }
  } catch (err: any) {
    instance.status = "error";
    instance.error = err.message || String(err);
    instance.endedAt = Date.now();
    logger.error?.(`agent-orchestrator: instance ${instance.id} error:`, err.message);
    pushMessage(instance, { ts: Date.now(), type: "error", content: `Fatal error: ${err.message}` });
    logToFile(instance, { event: "fatal_error", error: err.message });
  }

  // Complete any orphaned child instances that were started but never received an "end" event
  if (onChildInstance && activeChildren.size > 0) {
    for (const [key, child] of activeChildren) {
      onChildInstance("end", {
        parentId: instance.id,
        agent: child.agent,
        taskDescription: child.taskDescription,
        taskId: child.taskId,
        status: instance.status === "complete" ? "completed" : "failed",
        summary: instance.status === "complete"
          ? `Completed with parent orchestrator`
          : `Parent orchestrator failed: ${instance.error || "unknown"}`,
      });
    }
    activeChildren.clear();
  }

  // Write completion summary
  writeSummary(instance);

  // Log completion
  const elapsed = Math.round(((instance.endedAt || Date.now()) - instance.startedAt) / 1000);
  const costStr = instance.costUsd > 0 ? ` ($${instance.costUsd.toFixed(2)})` : "";
  const status = instance.status === "complete" ? "completed" : `failed (${instance.error || "unknown"})`;
  const summary = `Pipeline \`${instance.id}\` ${status} in ${elapsed}s${costStr}: ${truncate(instance.task, 80)}`;

  logToFile(instance, {
    event: "complete",
    status: instance.status,
    elapsedMs: (instance.endedAt || Date.now()) - instance.startedAt,
    costUsd: instance.costUsd,
    turns: instance.turns,
    resultPreview: instance.resultText ? instance.resultText.slice(0, 500) : null,
  });

  logger.info?.(`agent-orchestrator: ${summary}`);
  notify(instance.id, summary);
}

/**
 * Process an assistant message — extract tool use, text content, subagent activity.
 */
function processAssistantMessage(instance: CcInstance, message: any, onChildInstance?: ChildInstanceCallback): void {
  const content = message.message?.content;
  if (!Array.isArray(content)) return;

  for (const block of content) {
    if (block.type === "text" && block.text) {
      pushMessage(instance, {
        ts: Date.now(),
        type: "text",
        content: truncate(block.text, 500),
      });
      instance.activity = truncate(block.text, 100);
      logToFile(instance, { event: "text", content: truncate(block.text, 1000) });
    } else if (block.type === "tool_use") {
      const toolName = block.name || "unknown";
      const isSubagent = toolName === "Task" || toolName === "Agent";
      const agentType = isSubagent ? block.input?.subagent_type : undefined;

      if (isSubagent && agentType) {
        instance.activity = `spawning ${agentType} agent`;
        const promptPreview = truncate(block.input?.prompt || block.input?.description || agentType, 500);
        pushMessage(instance, {
          ts: Date.now(),
          type: "subagent_start",
          agent: agentType,
          content: truncate(promptPreview, 200),
        });
        logToFile(instance, {
          event: "subagent_start",
          agent: agentType,
          description: block.input?.description,
          prompt: truncate(block.input?.prompt || "", 2000),
        });

        // Notify parent about child instance creation
        if (onChildInstance) {
          onChildInstance("start", {
            parentId: instance.id,
            agent: agentType,
            taskDescription: block.input?.description || block.input?.prompt || agentType,
            taskId: block.id,
          });
        }
      } else {
        instance.activity = `using ${toolName}`;
        const paramSummary = `${toolName}(${summarizeParams(block.input)})`;
        pushMessage(instance, {
          ts: Date.now(),
          type: "tool_use",
          content: paramSummary,
        });
        logToFile(instance, {
          event: "tool_use",
          tool: toolName,
          params: summarizeParams(block.input),
        });
      }
    }
  }

  // Track usage/cost from message
  if (message.message?.usage) {
    const usage = message.message.usage;
    // Rough cost estimate: input $3/M, output $15/M (Sonnet 4.6 pricing)
    const inputCost = ((usage.input_tokens || 0) / 1_000_000) * 3;
    const outputCost = ((usage.output_tokens || 0) / 1_000_000) * 15;
    instance.costUsd += inputCost + outputCost;
  }

  instance.turns++;
}

/**
 * Process a result message — the query is done.
 */
function processResultMessage(instance: CcInstance, message: any): void {
  instance.endedAt = Date.now();
  instance.activity = undefined;

  if (message.subtype === "success") {
    instance.status = "complete";
    // Extract result text
    const resultText = message.result ||
      (Array.isArray(message.content)
        ? message.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n")
        : "");
    instance.resultText = resultText;
  } else {
    instance.status = "error";
    instance.error = message.error?.message || message.result || "Query failed";
    instance.resultText = message.result;
  }

  // Use SDK-reported cost if available
  if (message.total_cost_usd != null) {
    instance.costUsd = message.total_cost_usd;
  }
  if (message.num_turns != null) {
    instance.turns = message.num_turns;
  }

  pushMessage(instance, {
    ts: Date.now(),
    type: "text",
    content: `Pipeline ${instance.status}: ${truncate(instance.resultText || "", 300)}`,
  });

  // Log full result text to file (not truncated)
  logToFile(instance, {
    event: "result",
    status: instance.status,
    costUsd: instance.costUsd,
    turns: instance.turns,
    resultText: instance.resultText,
    error: instance.error,
  });
}

/** Push a message to the instance's ring buffer. */
function pushMessage(instance: CcInstance, msg: InstanceMessage): void {
  instance.recentMessages.push(msg);
  if (instance.recentMessages.length > MAX_RECENT_MESSAGES) {
    instance.recentMessages.shift();
  }
}

/** Truncate a string to maxLen characters. */
function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 3) + "...";
}

/** Summarize tool params for display (first 80 chars of key params). */
function summarizeParams(input: any): string {
  if (!input || typeof input !== "object") return "";
  const parts: string[] = [];
  for (const [k, v] of Object.entries(input)) {
    if (k === "prompt" || k === "description") continue; // Skip long text
    const val = typeof v === "string" ? truncate(v, 40) : JSON.stringify(v);
    parts.push(`${k}: ${val}`);
    if (parts.join(", ").length > 80) break;
  }
  return truncate(parts.join(", "), 80);
}

/**
 * Process SDK system messages — task lifecycle events for subagent tracking.
 * Handles task_started, task_progress, task_notification subtypes.
 */
function processSystemMessage(
  instance: CcInstance,
  message: any,
  logger: Logger,
  notifyProgress?: ProgressNotifyFn,
  onChildInstance?: ChildInstanceCallback,
): void {
  const subtype = message.subtype;

  if (subtype === "task_started") {
    const desc = message.description || "unknown task";
    const taskType = message.task_type || "";
    instance.activity = `task: ${desc}`;
    pushMessage(instance, {
      ts: Date.now(),
      type: "task_progress",
      agent: taskType,
      content: `Task started: ${desc}`,
    });
    logToFile(instance, {
      event: "task_started",
      taskId: message.task_id,
      taskType,
      description: desc,
    });
  } else if (subtype === "task_progress") {
    const desc = message.description || "";
    const lastTool = message.last_tool_name || "";
    instance.activity = desc || `using ${lastTool}`;
    pushMessage(instance, {
      ts: Date.now(),
      type: "task_progress",
      content: `Progress: ${desc}${lastTool ? ` (${lastTool})` : ""}`,
    });
    logToFile(instance, {
      event: "task_progress",
      taskId: message.task_id,
      description: desc,
      lastTool,
    });
  } else if (subtype === "task_notification") {
    const status = message.status || "unknown";
    const summary = message.summary || "";
    const isComplete = status === "completed";
    pushMessage(instance, {
      ts: Date.now(),
      type: isComplete ? "subagent_end" : "error",
      content: `Task ${status}: ${truncate(summary, 300)}`,
    });
    logToFile(instance, {
      event: "task_notification",
      taskId: message.task_id,
      status,
      summary: truncate(summary, 2000),
      usage: message.usage,
    });

    // Notify parent about child instance completion
    if (onChildInstance && (status === "completed" || status === "failed")) {
      onChildInstance("end", {
        parentId: instance.id,
        agent: message.agent || "",
        taskDescription: "",
        taskId: message.task_id,
        status,
        summary,
      });
    }

    // In interactive mode, push milestone notifications to Lloyd's session
    if (instance.interactive && notifyProgress) {
      const progressMsg = `**[Agent Progress]** \`${instance.id}\`: Task ${status} — ${truncate(summary, 200)}`;
      notifyProgress(instance.id, progressMsg);
    }
  }
}
