/**
 * sessions.ts — Session listing, messages, stats, usage chart, chat endpoints
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import { join } from "path";
import { randomUUID } from "crypto";
import type { PluginContext, SessionMessage, TimingEntry, RoutingEntry, SubagentRunStatus, AgentSessionStatus } from "./types.js";
import { cached, HEAVY_CACHE_TTL, parseJsonl, jsonResponse, readBody, handleCorsOptions, requirePost } from "./helpers.js";
import { loadSummaries, parseSessionSource, stripInjectedContext, GREETING_PROMPT } from "./gateway.js";

export function registerSessionRoutes(
  ctx: PluginContext,
  deps: {
    gwWsSend: (method: string, params: Record<string, unknown>) => Promise<any>;
    triggerSummaryGeneration: (sessionKey: string) => void;
  },
) {
  const { api, sessionsDir, timingLog, routingLog, ccInstancesDir, summariesFile } = ctx;
  const { gwWsSend, triggerSummaryGeneration } = deps;

  function getSessionFiles(): string[] {
    if (!existsSync(sessionsDir)) return [];
    return readdirSync(sessionsDir)
      .filter((f) => f.endsWith(".jsonl") && !f.includes(".reset."))
      .map((f) => join(sessionsDir, f));
  }

  function aggregateTokenUsage() {
    return cached("token-usage", () => {
      let totalInput = 0, totalOutput = 0, totalCacheRead = 0;
      const bySession: any[] = [];
      const sessionFiles = getSessionFiles();

      for (const file of sessionFiles) {
        const lines = parseJsonl<SessionMessage>(file);
        let sInput = 0, sOutput = 0, sCacheRead = 0, msgCount = 0;
        let lastActivity = "", model = "", sessionId = "";

        for (const line of lines) {
          if (line.type === "session") sessionId = (line as any).id || "";
          if (line.type === "message" && line.message?.usage) {
            const u = line.message.usage;
            sInput += u.input || 0;
            sOutput += u.output || 0;
            sCacheRead += u.cacheRead || 0;
            msgCount++;
            if (line.timestamp) lastActivity = line.timestamp;
            if (line.message.model) model = line.message.model;
          }
        }

        totalInput += sInput;
        totalOutput += sOutput;
        totalCacheRead += sCacheRead;

        if (msgCount > 0) {
          bySession.push({ sessionId, input: sInput, output: sOutput, cacheRead: sCacheRead, messageCount: msgCount, lastActivity, model });
        }
      }

      bySession.sort((a, b) => new Date(b.lastActivity).getTime() - new Date(a.lastActivity).getTime());
      return { totalInput, totalOutput, totalCacheRead, totalSessions: bySession.length, bySession };
    }, HEAVY_CACHE_TTL);
  }

  function loadCcInstanceSummaries(): Array<{ id: string; costUsd: number; startedAt: string; status: string }> {
    return cached("cc-instance-summaries", () => {
      if (!existsSync(ccInstancesDir)) return [];
      const files = readdirSync(ccInstancesDir).filter(f => f.endsWith(".summary.json"));
      const results: Array<{ id: string; costUsd: number; startedAt: string; status: string }> = [];
      for (const f of files) {
        try {
          const data = JSON.parse(readFileSync(join(ccInstancesDir, f), "utf-8"));
          results.push({ id: data.id || f, costUsd: data.costUsd || 0, startedAt: data.startedAt || "", status: data.status || "unknown" });
        } catch { /* skip */ }
      }
      return results;
    }, HEAVY_CACHE_TTL);
  }

  // GET /api/mc/stats
  api.registerHttpRoute({
    path: "/api/mc/stats",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const usage = aggregateTokenUsage();
        const ccSummaries = loadCcInstanceSummaries();
        const totalSubagentCost = ccSummaries.reduce((sum, s) => sum + s.costUsd, 0);
        jsonResponse(res, {
          totalInput: usage.totalInput,
          totalOutput: usage.totalOutput,
          totalCacheRead: usage.totalCacheRead,
          totalSessions: usage.totalSessions,
          totalSubagentCost,
          totalSubagentInstances: ccSummaries.length,
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/usage-chart
  api.registerHttpRoute({
    path: "/api/mc/usage-chart",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "/", "http://localhost");
        const range = url.searchParams.get("range") || "7d";

        const result = cached(`usage-chart-${range}`, () => {
          const now = Date.now();
          const rangeMs: Record<string, number> = {
            "24h": 24 * 60 * 60 * 1000,
            "7d": 7 * 24 * 60 * 60 * 1000,
            "30d": 30 * 24 * 60 * 60 * 1000,
          };
          const cutoff = now - (rangeMs[range] || rangeMs["7d"]);
          const bucketMs: Record<string, number> = {
            "24h": 60 * 60 * 1000,
            "7d": 6 * 60 * 60 * 1000,
            "30d": 24 * 60 * 60 * 1000,
          };
          const bucket = bucketMs[range] || bucketMs["7d"];
          const buckets = new Map<number, { input: number; output: number; cacheRead: number; subagentCost: number }>();

          for (const file of getSessionFiles()) {
            for (const line of parseJsonl<SessionMessage>(file)) {
              if (line.type === "message" && line.message?.usage && line.timestamp) {
                const ts = new Date(line.timestamp).getTime();
                if (ts < cutoff) continue;
                const key = Math.floor(ts / bucket) * bucket;
                const existing = buckets.get(key) || { input: 0, output: 0, cacheRead: 0, subagentCost: 0 };
                existing.input += line.message.usage.input || 0;
                existing.output += line.message.usage.output || 0;
                existing.cacheRead += line.message.usage.cacheRead || 0;
                buckets.set(key, existing);
              }
            }
          }

          for (const summary of loadCcInstanceSummaries()) {
            if (!summary.startedAt) continue;
            const ts = new Date(summary.startedAt).getTime();
            if (ts < cutoff || isNaN(ts)) continue;
            const key = Math.floor(ts / bucket) * bucket;
            const existing = buckets.get(key) || { input: 0, output: 0, cacheRead: 0, subagentCost: 0 };
            existing.subagentCost += summary.costUsd;
            buckets.set(key, existing);
          }

          const data = Array.from(buckets.entries())
            .map(([ts, vals]) => ({ ts, ...vals }))
            .sort((a, b) => a.ts - b.ts);

          return { range, bucketMs: bucket, data };
        }, HEAVY_CACHE_TTL);

        jsonResponse(res, result);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/api-calls
  api.registerHttpRoute({
    path: "/api/mc/api-calls",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const data = cached("api-calls", () => {
          const timing = parseJsonl<TimingEntry>(timingLog, 200);
          const routing = parseJsonl<RoutingEntry>(routingLog, 200);
          const runs = timing
            .filter((e) => e.event === "run_end")
            .slice(-50)
            .map((r) => {
              const rTs = new Date(r.ts).getTime();
              const route = routing.find((rt) => Math.abs(new Date(rt.ts).getTime() - rTs) < 60000);
              return {
                ts: r.ts, sessionId: r.sessionId, model: route?.tier || "unknown",
                totalMs: r.totalMs, llmMs: r.llmMs, toolMs: r.toolMs,
                roundTrips: r.roundTrips, toolCallCount: r.toolCallCount, success: r.success,
              };
            });
          const toolCalls = timing
            .filter((e) => e.event === "tool_call")
            .slice(-50)
            .map((t) => ({ ts: t.ts, sessionId: t.sessionId, toolName: t.toolName, durationMs: t.durationMs }));
          return { runs: runs.reverse(), toolCalls: toolCalls.reverse() };
        }, HEAVY_CACHE_TTL);
        jsonResponse(res, data);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/sessions
  api.registerHttpRoute({
    path: "/api/mc/sessions",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const result = await gwWsSend("sessions.list", { agentId: "main" });
        const entries: any[] = result?.entries || result?.sessions || [];
        const sessions = entries
          .filter((e: any) => {
            const key = e.sessionKey || e.key || "";
            return !key.includes(":cron:") && !key.includes(":sub:");
          })
          .map((e: any) => {
            const key = e.sessionKey || e.key || "";
            const { source, peer } = parseSessionSource(key);
            return {
              sessionKey: key,
              sessionId: e.sessionId || e.id || undefined,
              lastActivity: e.lastActivity || e.updatedAt || e.createdAt || new Date().toISOString(),
              messageCount: e.messageCount || undefined,
              model: e.model || "",
              summary: e.summary || undefined,
              source, peer,
            };
          });

        const cachedSummaries = loadSummaries(summariesFile);
        for (const s of sessions) {
          if (cachedSummaries[s.sessionKey]) {
            s.summary = cachedSummaries[s.sessionKey];
          } else {
            triggerSummaryGeneration(s.sessionKey);
          }
        }

        jsonResponse(res, { sessions });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/session-messages
  api.registerHttpRoute({
    path: "/api/mc/session-messages",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "/", "http://localhost");
        const sessionKey = url.searchParams.get("sessionKey") || url.searchParams.get("id");
        if (!sessionKey) { jsonResponse(res, { error: "Missing sessionKey" }, 400); return; }
        const includeTools = url.searchParams.get("tools") === "1";

        let rawMessages: any[] = [];
        let fromJsonl = false;
        try {
          const sessResult = await gwWsSend("sessions.list", { agentId: "main" });
          const entries: any[] = sessResult?.entries || sessResult?.sessions || [];
          const match = entries.find((e: any) => (e.sessionKey || e.key) === sessionKey);
          const sessionId = match?.sessionId || match?.id;
          if (sessionId) {
            const jsonlPath = join(sessionsDir, sessionId + ".jsonl");
            if (existsSync(jsonlPath)) {
              const allLines = parseJsonl<any>(jsonlPath);
              rawMessages = allLines
                .filter((l: any) => l.type === "message" || l.type === "result")
                .map((l: any) => ({ id: l.id, timestamp: l.timestamp, ...l.message }));
              fromJsonl = true;
            }
          }
        } catch { /* fall through */ }

        if (!fromJsonl) {
          const result = await gwWsSend("chat.history", { sessionKey, limit: 200 });
          rawMessages = result?.messages || result?.history || [];
        }

        let lastUserTs: number | null = null;
        let turnHadThinking = false;
        const messages: Record<string, any>[] = [];

        for (const msg of rawMessages) {
          const role = msg.role || msg.message?.role;
          const content = msg.content || msg.message?.content || [];
          const timestamp = msg.timestamp || msg.ts || "";
          const id = msg.id || msg.messageId || `msg-${messages.length}`;
          const model = msg.model || msg.message?.model;
          const usage = msg.usage || msg.message?.usage;

          if (role === "user") {
            if (timestamp) lastUserTs = new Date(timestamp).getTime();
            turnHadThinking = false;
          } else if (role === "assistant") {
            if (content.some((c: any) => c.type === "thinking")) turnHadThinking = true;
            if (!includeTools && !content.some((c: any) => c.type === "text" && c.text)) continue;
          } else if (role === "toolResult") {
            if (!includeTools) continue;
          } else {
            continue;
          }

          const entry: Record<string, any> = {
            id, timestamp, role, content, model,
            usage: usage ? {
              input: usage.input || 0, output: usage.output || 0,
              cacheRead: usage.cacheRead || 0, cacheCreation: usage.cacheWrite || 0,
              totalTokens: usage.totalTokens || 0, cost: usage.cost?.total,
            } : undefined,
          };

          if (role === "assistant") {
            entry.hasThinking = (turnHadThinking || content.some((c: any) => c.type === "thinking")) || undefined;
            const tcc = content.filter((c: any) => c.type === "toolCall").length;
            if (tcc > 0) entry.toolCallCount = tcc;
            if (lastUserTs && timestamp) entry.durationMs = new Date(timestamp).getTime() - lastUserTs;
          }

          if (role === "toolResult") {
            entry.toolCallId = msg.toolCallId || msg.message?.toolCallId;
            entry.toolName = msg.toolName || msg.message?.toolName;
            entry.isError = msg.isError || msg.message?.isError || false;
            entry.content = (content || []).map((c: any) => ({
              type: c.type,
              text: typeof c.text === "string" ? c.text.slice(0, 300) : c.text,
            }));
          }

          messages.push(entry);
        }

        jsonResponse(res, { sessionKey, messages });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/chat
  api.registerHttpRoute({
    path: "/api/mc/chat",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
      try {
        const body = JSON.parse(await readBody(req));
        const message = body.message;
        if (!message || typeof message !== "string") { jsonResponse(res, { error: "Missing message" }, 400); return; }
        const idempotencyKey = randomUUID();
        const result = await gwWsSend("chat.send", {
          sessionKey: body.sessionKey || "agent:main:main",
          message,
          idempotencyKey,
        });
        jsonResponse(res, { ok: true, status: 200, data: result });
      } catch (err: any) {
        api.logger.error?.(`mission-control: chat.send failed: ${err.message}`);
        jsonResponse(res, { error: err.message }, 502);
      }
    },
  });

  // POST /api/mc/chat-abort
  api.registerHttpRoute({
    path: "/api/mc/chat-abort",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
      try {
        const body = JSON.parse(await readBody(req));
        await gwWsSend("chat.abort", { sessionKey: body.sessionKey || "agent:main:main" });
        jsonResponse(res, { ok: true });
      } catch (err: any) {
        api.logger.error?.(`mission-control: chat.abort failed: ${err.message}`);
        jsonResponse(res, { error: err.message }, 502);
      }
    },
  });

  // POST /api/mc/subagent-abort
  api.registerHttpRoute({
    path: "/api/mc/subagent-abort",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
      try {
        const body = await readBody(req);
        const { sessionKey } = JSON.parse(body);
        if (!sessionKey) { jsonResponse(res, { error: "sessionKey required" }, 400); return; }
        await gwWsSend("chat.abort", { sessionKey });
        jsonResponse(res, { ok: true });
      } catch (err: any) {
        api.logger.error?.(`mission-control: subagent-abort failed: ${err.message}`);
        jsonResponse(res, { error: err.message }, 502);
      }
    },
  });

  // POST /api/mc/session-new
  api.registerHttpRoute({
    path: "/api/mc/session-new",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
      try {
        const newKey = `agent:main:mc-${randomUUID()}`;
        await gwWsSend("chat.send", {
          sessionKey: newKey,
          message: GREETING_PROMPT,
          idempotencyKey: randomUUID(),
        });
        jsonResponse(res, { ok: true, sessionKey: newKey });
      } catch (err: any) {
        api.logger.error?.(`mission-control: session-new failed: ${err.message}`);
        jsonResponse(res, { error: err.message }, 502);
      }
    },
  });

  // GET /api/mc/agent-status
  api.registerHttpRoute({
    path: "/api/mc/agent-status",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const mainStates: AgentSessionStatus[] = [];
        for (const [key, status] of ctx.agentSessionStates) {
          if (key.startsWith("agent:main:")) mainStates.push(status);
        }
        mainStates.sort((a, b) => b.lastUpdated - a.lastUpdated);
        const mainState = mainStates[0] ?? {
          sessionKey: "agent:main:main",
          state: "idle" as const,
          queueDepth: 0,
          lastUpdated: Date.now(),
        };

        let activityLabel = "Idle";
        let activityDetail: string | null = null;
        const activity = ctx.currentActivity.value;
        if (activity.type === "tool_call") {
          activityLabel = "Running tool";
          activityDetail = activity.toolName ?? null;
        } else if (activity.type === "llm_thinking") {
          activityLabel = "Thinking";
          activityDetail = activity.model ?? null;
        }

        const subagentRuns = loadSubagentRuns(ctx);
        const activeRuns = subagentRuns.filter((r) => !r.endedAt);
        const recentCompleted = subagentRuns.filter((r) => !!r.endedAt).slice(0, 10);

        jsonResponse(res, {
          mainAgent: {
            state: mainState.state,
            reason: mainState.reason,
            queueDepth: mainState.queueDepth,
            lastUpdated: mainState.lastUpdated,
          },
          activity: {
            type: activity.type,
            label: activityLabel,
            detail: activityDetail,
            elapsedMs: Date.now() - activity.startedAt,
          },
          heartbeat: ctx.lastHeartbeat.value,
          subagents: { active: activeRuns, recentCompleted },
          timestamp: new Date().toISOString(),
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/gateway-sessions
  api.registerHttpRoute({
    path: "/api/mc/gateway-sessions",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const gatewaySessions: any[] = [];
        for (const [key, status] of ctx.agentSessionStates) {
          if (key.startsWith("agent:main:")) continue;
          const parts = key.split(":");
          const agentId = parts.length >= 2 && parts[0] === "agent" ? parts[1] : key;
          gatewaySessions.push({
            sessionKey: key, sessionId: status.sessionId, agentId,
            state: status.state, source: "gateway",
            lastUpdated: status.lastUpdated, elapsedMs: Date.now() - status.lastUpdated,
          });
        }
        gatewaySessions.sort((a, b) => {
          const aActive = a.state !== "idle" ? 1 : 0;
          const bActive = b.state !== "idle" ? 1 : 0;
          if (aActive !== bActive) return bActive - aActive;
          return b.lastUpdated - a.lastUpdated;
        });

        const subagentRuns = loadSubagentRuns(ctx);

        jsonResponse(res, {
          sessions: gatewaySessions,
          subagentRuns: subagentRuns.slice(0, 30),
          gwSubagents: [],
          timestamp: new Date().toISOString(),
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });
}

// ── Subagent runs helper (used by agent-status and gateway-sessions) ──

function loadSubagentRuns(ctx: PluginContext): SubagentRunStatus[] {
  const subagentRunsFile = join(ctx.rootDir, "subagents/runs.json");
  try {
    if (!existsSync(subagentRunsFile)) return [];
    const raw = JSON.parse(readFileSync(subagentRunsFile, "utf-8"));
    if (raw.version !== 2 || !raw.runs) return [];
    const runs: SubagentRunStatus[] = [];
    for (const [runId, record] of Object.entries<any>(raw.runs)) {
      runs.push({
        runId,
        childSessionKey: record.childSessionKey,
        requesterSessionKey: record.requesterSessionKey,
        task: record.task,
        label: record.label,
        model: record.model,
        spawnMode: record.spawnMode,
        createdAt: record.createdAt,
        startedAt: record.startedAt,
        endedAt: record.endedAt,
        outcome: typeof record.outcome === "object" && record.outcome !== null
          ? record.outcome.status : record.outcome,
        endedReason: record.endedReason,
        durationMs: record.endedAt && record.startedAt ? record.endedAt - record.startedAt : undefined,
      });
    }
    runs.sort((a, b) => {
      if (!a.endedAt && b.endedAt) return -1;
      if (a.endedAt && !b.endedAt) return 1;
      return b.createdAt - a.createdAt;
    });
    return runs;
  } catch {
    return [];
  }
}
