/**
 * query-consumer.ts — Consumes Agent SDK query() async generators in the background.
 *
 * Tracks status, cost, messages, and fires completion notifications.
 */

import type { CcInstance, InstanceMessage } from "./types.js";

/** Max recent messages to keep per instance (ring buffer for Mission Control) */
const MAX_RECENT_MESSAGES = 100;

/** Logger interface (passed from plugin api) */
interface Logger {
  info?: (...args: any[]) => void;
  error?: (...args: any[]) => void;
}

/** Callback to fire system events when pipelines complete */
type NotifyFn = (instanceId: string, summary: string) => void;

/**
 * Consume an Agent SDK query() async generator in the background.
 * Updates the CcInstance in-place as messages arrive.
 */
export async function consumeQuery(
  instance: CcInstance,
  queryIter: AsyncGenerator<any, void>,
  logger: Logger,
  notify: NotifyFn,
): Promise<void> {
  try {
    for await (const message of queryIter) {
      // Track session ID from init message
      if (message.type === "system" && message.subtype === "init" && message.session_id) {
        instance.sessionId = message.session_id;
      }

      // Process based on message type
      if (message.type === "assistant" && message.message) {
        processAssistantMessage(instance, message);
      } else if (message.type === "result") {
        processResultMessage(instance, message);
      } else if (message.type === "error") {
        pushMessage(instance, {
          ts: Date.now(),
          type: "error",
          content: message.error?.message || message.message || "Unknown error",
        });
      }
    }
  } catch (err: any) {
    instance.status = "error";
    instance.error = err.message || String(err);
    instance.endedAt = Date.now();
    logger.error?.(`agent-orchestrator: instance ${instance.id} error:`, err.message);
    pushMessage(instance, {
      ts: Date.now(),
      type: "error",
      content: `Fatal error: ${err.message}`,
    });
  }

  // Fire completion notification
  const elapsed = Math.round(((instance.endedAt || Date.now()) - instance.startedAt) / 1000);
  const costStr = instance.costUsd > 0 ? ` ($${instance.costUsd.toFixed(2)})` : "";
  const status = instance.status === "complete" ? "completed" : `failed (${instance.error || "unknown"})`;
  const summary = `Pipeline \`${instance.id}\` ${status} in ${elapsed}s${costStr}: ${truncate(instance.task, 80)}`;

  logger.info?.(`agent-orchestrator: ${summary}`);
  notify(instance.id, summary);
}

/**
 * Process an assistant message — extract tool use, text content, subagent activity.
 */
function processAssistantMessage(instance: CcInstance, message: any): void {
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
    } else if (block.type === "tool_use") {
      const toolName = block.name || "unknown";
      const isSubagent = toolName === "Task";
      const agentType = isSubagent ? block.input?.subagent_type : undefined;

      if (isSubagent && agentType) {
        instance.activity = `spawning ${agentType} agent`;
        pushMessage(instance, {
          ts: Date.now(),
          type: "subagent_start",
          agent: agentType,
          content: truncate(block.input?.prompt || block.input?.description || agentType, 200),
        });
      } else {
        instance.activity = `using ${toolName}`;
        pushMessage(instance, {
          ts: Date.now(),
          type: "tool_use",
          content: `${toolName}(${summarizeParams(block.input)})`,
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
