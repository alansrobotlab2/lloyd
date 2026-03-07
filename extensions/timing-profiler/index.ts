import { mkdir, appendFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

const LOG_FILE = join(
  process.env.HOME ?? "/root",
  ".openclaw",
  "logs",
  "timing.jsonl",
);

let dirEnsured = false;

async function writeRecord(record: object) {
  try {
    if (!dirEnsured) {
      await mkdir(dirname(LOG_FILE), { recursive: true });
      dirEnsured = true;
    }
    await appendFile(LOG_FILE, JSON.stringify(record) + "\n");
  } catch {
    // non-fatal
  }
}

// Per-run state — keyed by ctx.sessionKey (the only ID available across all hooks)
interface RunState {
  sessionKey: string;
  runId: string | null;
  sessionId: string | null;
  startMs: number;
  lastBoundaryMs: number; // tracks time boundaries between LLM/tool phases
  llmSegments: number[];  // LLM thinking durations (gaps between tool calls)
  toolCallsLog: Array<{ toolName: string; durationMs: number }>;
  pendingTools: Map<string, number>; // toolName:timestamp -> startMs
}

const runs = new Map<string, RunState>();
let toolCallCounter = 0;

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.("timing-profiler: active — writing to " + LOG_FILE);

  // LLM run started — fires once per full agent run
  api.on("llm_input", (event: any, ctx: any) => {
    const key = ctx?.sessionKey;
    if (!key) return;
    runs.set(key, {
      sessionKey: key,
      runId: event?.runId ?? null,
      sessionId: event?.sessionId ?? ctx?.sessionId ?? null,
      startMs: Date.now(),
      lastBoundaryMs: Date.now(),
      llmSegments: [],
      toolCallsLog: [],
      pendingTools: new Map(),
    });
  });

  // LLM run finished — record final LLM segment
  api.on("llm_output", (event: any, ctx: any) => {
    const key = ctx?.sessionKey;
    if (!key) return;
    const run = runs.get(key);
    if (!run) return;
    const llmMs = Date.now() - run.lastBoundaryMs;
    if (llmMs > 10) run.llmSegments.push(llmMs);
    run.lastBoundaryMs = Date.now();
  });

  // Tool call about to execute — LLM just finished thinking, record that segment
  api.on("before_tool_call", (event: any, ctx: any) => {
    const key = ctx?.sessionKey;
    if (!key) return;
    const run = runs.get(key);
    if (run) {
      const llmMs = Date.now() - run.lastBoundaryMs;
      if (llmMs > 10) run.llmSegments.push(llmMs);
      run.pendingTools.set(event?.toolName + ":" + (++toolCallCounter), Date.now());
    }
  });

  // Tool call completed
  api.on("after_tool_call", (event: any, ctx: any) => {
    const key = ctx?.sessionKey;
    const toolName = event?.toolName ?? ctx?.toolName ?? "unknown";
    const durationMs = event?.durationMs ?? 0;

    const run = key ? runs.get(key) : null;

    writeRecord({
      ts: new Date().toISOString(),
      event: "tool_call",
      runId: run?.runId ?? null,
      sessionId: run?.sessionId ?? null,
      toolName,
      durationMs,
    });

    if (run) {
      run.toolCallsLog.push({ toolName, durationMs });
      run.lastBoundaryMs = Date.now();
      // Clean up oldest matching pending tool
      for (const [k] of run.pendingTools) {
        if (k.startsWith(toolName + ":")) {
          run.pendingTools.delete(k);
          break;
        }
      }
    }
  });

  // Agent run ended — compute final breakdown
  api.on("agent_end", (event: any, ctx: any) => {
    const key = ctx?.sessionKey;
    const run = key ? runs.get(key) : null;

    const totalMs = run
      ? Date.now() - run.startMs
      : (event?.durationMs ?? null);
    const llmMs = run
      ? run.llmSegments.reduce((s, v) => s + v, 0)
      : null;
    const toolMs = run
      ? run.toolCallsLog.reduce((s, t) => s + t.durationMs, 0)
      : null;

    writeRecord({
      ts: new Date().toISOString(),
      event: "run_end",
      runId: run?.runId ?? null,
      sessionId: run?.sessionId ?? null,
      totalMs,
      llmMs,
      toolMs,
      overheadMs:
        totalMs != null && llmMs != null && toolMs != null
          ? Math.max(0, totalMs - llmMs - toolMs)
          : null,
      roundTrips: run ? run.llmSegments.length : null,
      toolCallCount: run?.toolCallsLog.length ?? null,
      toolCalls: run?.toolCallsLog ?? [],
      success: event?.success ?? null,
      error: event?.error ?? null,
    });

    if (key) runs.delete(key);
  });
}
