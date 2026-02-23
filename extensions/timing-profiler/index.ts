import { appendFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

const LOG_FILE = join(
  process.env.HOME ?? "/root",
  ".openclaw",
  "logs",
  "timing.jsonl",
);

function writeRecord(record: object) {
  try {
    mkdirSync(dirname(LOG_FILE), { recursive: true });
    appendFileSync(LOG_FILE, JSON.stringify(record) + "\n");
  } catch {
    // non-fatal
  }
}

// Per-run state
interface RunState {
  runId: string;
  sessionId: string;
  startMs: number;
  llmStartMs: number | null;
  llmRoundTrips: Array<{ durationMs: number; usage?: unknown }>;
  toolCallsLog: Array<{ toolName: string; durationMs: number }>;
  pendingTool: { toolName: string; toolCallId: string; startMs: number } | null;
}

const runs = new Map<string, RunState>();

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.("timing-profiler: active — writing to " + LOG_FILE);

  // LLM run started — fires once per full agent run
  api.on("llm_input", (event: any) => {
    const { runId, sessionId } = event ?? {};
    if (!runId) return;
    runs.set(runId, {
      runId,
      sessionId: sessionId ?? "unknown",
      startMs: Date.now(),
      llmStartMs: Date.now(),
      llmRoundTrips: [],
      toolCallsLog: [],
      pendingTool: null,
    });
  });

  // LLM run finished — fires once after all round trips complete
  api.on("llm_output", (event: any) => {
    const { runId, usage } = event ?? {};
    if (!runId) return;
    const run = runs.get(runId);
    if (!run || run.llmStartMs === null) return;
    // Record this as covering the full model time (we break it down via session JSONL)
    run.llmRoundTrips.push({
      durationMs: Date.now() - run.llmStartMs,
      usage,
    });
    run.llmStartMs = null;
  });

  // Tool call about to execute — before_tool_call is modifying; returning undefined = pass through
  api.on("before_tool_call", (event: any) => {
    const { toolName, toolCallId, runId } = event ?? {};
    if (!toolName) return;
    const run = runId ? runs.get(runId) : null;
    if (run) {
      run.pendingTool = { toolName, toolCallId: toolCallId ?? "?", startMs: Date.now() };
    } else {
      // No runId on before_tool_call in some versions — track globally via stack
      (register as any)._pendingTool = { toolName, toolCallId: toolCallId ?? "?", startMs: Date.now() };
    }
  });

  // Tool call completed
  api.on("after_tool_call", (event: any) => {
    const { toolName, runId } = event ?? {};
    const run = runId ? runs.get(runId) : null;
    const pending = run?.pendingTool ?? (register as any)._pendingTool;
    if (!pending) return;

    const durationMs = Date.now() - pending.startMs;

    writeRecord({
      ts: new Date().toISOString(),
      event: "tool_call",
      runId: runId ?? null,
      sessionId: run?.sessionId ?? null,
      toolName: toolName ?? pending.toolName,
      toolCallId: pending.toolCallId,
      durationMs,
    });

    if (run) {
      run.toolCallsLog.push({ toolName: toolName ?? pending.toolName, durationMs });
      run.pendingTool = null;
    } else {
      (register as any)._pendingTool = null;
    }
  });

  // Agent run ended
  api.on("agent_end", (event: any) => {
    const { runId, durationMs: hookDurationMs, success, error } = event ?? {};

    // Find the run state — agent_end may not have runId; fall back to most recent run
    let run: RunState | undefined;
    if (runId) {
      run = runs.get(runId);
    } else if (runs.size > 0) {
      // Take the last started run
      for (const r of runs.values()) run = r;
    }

    const totalMs = run ? Date.now() - run.startMs : (hookDurationMs ?? null);
    const llmMs = run?.llmRoundTrips.reduce((sum, r) => sum + r.durationMs, 0) ?? null;
    const toolMs = run?.toolCallsLog.reduce((sum, t) => sum + t.durationMs, 0) ?? null;

    writeRecord({
      ts: new Date().toISOString(),
      event: "run_end",
      runId: runId ?? run?.runId ?? null,
      sessionId: run?.sessionId ?? null,
      totalMs,
      llmMs,
      toolMs,
      overheadMs: (totalMs !== null && llmMs !== null && toolMs !== null)
        ? Math.max(0, totalMs - llmMs - toolMs)
        : null,
      roundTrips: run?.llmRoundTrips.length ?? null,
      toolCallCount: run?.toolCallsLog.length ?? null,
      toolCalls: run?.toolCallsLog ?? [],
      success: success ?? null,
      error: error ?? null,
    });

    if (run) runs.delete(run.runId);
  });
}

// Initialize fallback pending tool slot
(register as any)._pendingTool = null;
