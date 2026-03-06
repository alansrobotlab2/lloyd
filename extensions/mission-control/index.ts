/**
 * index.ts — Mission Control Dashboard plugin
 *
 * Serves a React dashboard at /mc/ and exposes REST API endpoints at /api/mc/*
 * for token usage stats, API call monitoring, session chat, and system health.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { onDiagnosticEvent, type DiagnosticEventPayload } from "openclaw/plugin-sdk";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, writeFileSync, readdirSync, existsSync, statSync } from "fs";
import { join, extname, relative } from "path";
import { execSync } from "child_process";
import { connect as netConnect } from "net";
import { homedir } from "os";
import { randomUUID, createPrivateKey, createPublicKey, createHash, sign as cryptoSign } from "crypto";
// Use the ws module bundled with openclaw (can't use bare "ws" from plugin context)
const WS = require("/home/alansrobotlab/.npm-global/lib/node_modules/openclaw/node_modules/ws");

// ── Device identity helpers for WebSocket auth ──────────────────────────

const DEVICE_IDENTITY_PATH = join(homedir(), ".openclaw", "identity", "device.json");

function base64UrlEncode(buf: Buffer): string {
  return buf.toString("base64").replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/g, "");
}

const ED25519_SPKI_PREFIX = Buffer.from([0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00]);

function derivePublicKeyRaw(publicKeyPem: string): Buffer {
  const spki = createPublicKey(publicKeyPem).export({ type: "spki", format: "der" });
  if (spki.length === ED25519_SPKI_PREFIX.length + 32 && spki.subarray(0, ED25519_SPKI_PREFIX.length).equals(ED25519_SPKI_PREFIX))
    return spki.subarray(ED25519_SPKI_PREFIX.length);
  return spki;
}

function loadDeviceIdentity(): { deviceId: string; publicKeyPem: string; privateKeyPem: string } | null {
  try {
    const raw = readFileSync(DEVICE_IDENTITY_PATH, "utf8");
    const parsed = JSON.parse(raw);
    if (parsed?.version === 1 && typeof parsed.deviceId === "string" && typeof parsed.publicKeyPem === "string" && typeof parsed.privateKeyPem === "string") {
      return { deviceId: parsed.deviceId, publicKeyPem: parsed.publicKeyPem, privateKeyPem: parsed.privateKeyPem };
    }
  } catch {}
  return null;
}

function signDevicePayload(privateKeyPem: string, payload: string): string {
  const key = createPrivateKey(privateKeyPem);
  return base64UrlEncode(cryptoSign(null, Buffer.from(payload, "utf8"), key));
}

function buildDeviceAuthPayloadV3(params: {
  deviceId: string; clientId: string; clientMode: string; role: string;
  scopes: string[]; signedAtMs: number; token: string; nonce: string;
  platform: string; deviceFamily: string;
}): string {
  return ["v3", params.deviceId, params.clientId, params.clientMode, params.role,
    params.scopes.join(","), String(params.signedAtMs), params.token, params.nonce,
    params.platform, params.deviceFamily].join("|");
}

// ── Types ──────────────────────────────────────────────────────────────────

interface CacheEntry<T> {
  data: T;
  ts: number;
}

interface SessionMessage {
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

interface TimingEntry {
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

interface RoutingEntry {
  ts: string;
  tier: string;
  reason: string;
  confidence: number;
  classifierUsed: boolean;
  latencyMs: number;
  promptLength: number;
  sessionDepth: number;
}

// ── MIME types ──────────────────────────────────────────────────────────────

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
};

// ── Helpers ─────────────────────────────────────────────────────────────────

const CACHE_TTL = 5_000; // 5 seconds — short enough for interactive chat
const cache = new Map<string, CacheEntry<any>>();

function cached<T>(key: string, fn: () => T): T {
  const entry = cache.get(key);
  if (entry && Date.now() - entry.ts < CACHE_TTL) return entry.data;
  const data = fn();
  cache.set(key, { data, ts: Date.now() });
  return data;
}

function parseJsonl<T>(filePath: string, limit?: number): T[] {
  if (!existsSync(filePath)) return [];
  const content = readFileSync(filePath, "utf-8");
  const lines = content.trim().split("\n").filter(Boolean);
  const items: T[] = [];
  const start = limit ? Math.max(0, lines.length - limit) : 0;
  for (let i = start; i < lines.length; i++) {
    try {
      items.push(JSON.parse(lines[i]));
    } catch {}
  }
  return items;
}

function jsonResponse(res: ServerResponse, data: any, status = 200) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  });
  res.end(JSON.stringify(data));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

// ── Plugin Registration ────────────────────────────────────────────────────

export default function register(api: OpenClawPluginApi) {
  const rootDir = join(__dirname, "../..");
  const sessionsDir = join(rootDir, "agents/main/sessions");
  const timingLog = join(rootDir, "logs/timing.jsonl");
  const routingLog = join(rootDir, "logs/routing.jsonl");
  const modelsFile = join(rootDir, "agents/main/agent/models.json");
  const toolsFile = join(rootDir, "agents/main/agent/tools.json");
  const authFile = join(rootDir, "agents/main/agent/auth-profiles.json");
  const cronJobsFile = join(rootDir, "cron/jobs.json");
  const configFile = join(rootDir, "openclaw.json");
  const distWebDir = join(__dirname, "dist-web");

  api.logger.info?.("mission-control: loaded");

  // ── Agent Status Tracking ─────────────────────────────────────────────

  interface AgentSessionStatus {
    sessionKey: string;
    sessionId?: string;
    state: "idle" | "processing" | "waiting";
    reason?: string;
    queueDepth: number;
    lastUpdated: number;
  }

  interface AgentActivity {
    type: "tool_call" | "llm_thinking" | "idle";
    toolName?: string;
    model?: string;
    startedAt: number;
    sessionKey?: string;
  }

  const agentSessionStates = new Map<string, AgentSessionStatus>();
  let currentActivity: AgentActivity = { type: "idle", startedAt: Date.now() };
  let activityResetTimer: ReturnType<typeof setTimeout> | null = null;
  let lastHeartbeat: { active: number; waiting: number; queued: number } | null = null;

  onDiagnosticEvent((evt: DiagnosticEventPayload) => {
    if (evt.type === "session.state") {
      const key = evt.sessionKey ?? evt.sessionId ?? "unknown";
      agentSessionStates.set(key, {
        sessionKey: key,
        sessionId: evt.sessionId,
        state: evt.state,
        reason: evt.reason,
        queueDepth: evt.queueDepth ?? 0,
        lastUpdated: Date.now(),
      });
      if (evt.state === "idle" && key.startsWith("agent:main:")) {
        currentActivity = { type: "idle", startedAt: Date.now() };
      }
    }
    if (evt.type === "diagnostic.heartbeat") {
      lastHeartbeat = { active: evt.active, waiting: evt.waiting, queued: evt.queued };
    }
  });

  // ── Gateway WebSocket client ────────────────────────────────────────────
  // Uses the same protocol as the built-in control UI (chat.send) so
  // messages are injected into the active agent session rather than
  // creating isolated hook sessions.

  let gwWs: InstanceType<typeof WS> | null = null;
  let gwWsReady = false;
  let gwWsReqId = 0;
  const gwWsPending = new Map<string, { resolve: (v: any) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }>();

  const deviceIdentity = loadDeviceIdentity();

  function readGatewayToken(): string {
    try {
      const cfg = JSON.parse(readFileSync(configFile, "utf-8"));
      return cfg.gateway?.auth?.token || "";
    } catch { return ""; }
  }

  function gwWsConnect() {
    if (gwWs && (gwWs.readyState === WS.OPEN || gwWs.readyState === WS.CONNECTING)) return;
    if (!deviceIdentity) {
      api.logger.error?.("mission-control: no device identity found, cannot connect to gateway WS");
      return;
    }
    gwWsReady = false;
    try {
      const ws = new WS("ws://127.0.0.1:18789", {
        headers: { Origin: "http://127.0.0.1:18789" },
      });
      gwWs = ws;

      ws.on("message", (data: Buffer | string) => {
        try {
          const msg = JSON.parse(typeof data === "string" ? data : data.toString());

          // Handle the connect.challenge event — sign and respond with connect
          if (msg.type === "event" && msg.event === "connect.challenge" && msg.payload?.nonce) {
            const nonce: string = msg.payload.nonce;
            const signedAtMs = Date.now();
            const clientId = "webchat-ui";
            const clientMode = "webchat";
            const role = "operator";
            const scopes = ["operator.admin"];

            const payload = buildDeviceAuthPayloadV3({
              deviceId: deviceIdentity.deviceId,
              clientId,
              clientMode,
              role,
              scopes,
              signedAtMs,
              token: readGatewayToken(),
              nonce,
              platform: "linux",
              deviceFamily: "",
            });
            const signature = signDevicePayload(deviceIdentity.privateKeyPem, payload);
            const publicKeyBase64Url = base64UrlEncode(derivePublicKeyRaw(deviceIdentity.publicKeyPem));

            const connectId = `mc-connect-${++gwWsReqId}`;
            ws.send(JSON.stringify({
              type: "req",
              id: connectId,
              method: "connect",
              params: {
                minProtocol: 3,
                maxProtocol: 3,
                client: {
                  id: clientId,
                  displayName: "Mission Control",
                  version: "1.0.0",
                  platform: "linux",
                  mode: clientMode,
                },
                role,
                scopes,
                device: {
                  id: deviceIdentity.deviceId,
                  publicKey: publicKeyBase64Url,
                  signature,
                  signedAt: signedAtMs,
                  nonce,
                },
                ...(readGatewayToken() ? { auth: { token: readGatewayToken() } } : {}),
              },
            }));
            return;
          }

          if (msg.type === "res") {
            if (msg.payload?.type === "hello-ok") {
              gwWsReady = true;
              api.logger.info?.("mission-control: gateway WS connected (device auth OK)");
            }
            const entry = gwWsPending.get(msg.id);
            if (entry) {
              gwWsPending.delete(msg.id);
              clearTimeout(entry.timer);
              if (msg.ok) {
                entry.resolve(msg.payload);
              } else {
                entry.reject(new Error(msg.error?.message || "Gateway request failed"));
              }
            }
          }
        } catch {
          // Ignore malformed messages
        }
      });

      ws.on("close", () => {
        gwWsReady = false;
        gwWs = null;
        for (const [id, entry] of gwWsPending) {
          clearTimeout(entry.timer);
          entry.reject(new Error("WebSocket closed"));
          gwWsPending.delete(id);
        }
        setTimeout(gwWsConnect, 5000);
      });

      ws.on("error", (err: Error) => {
        api.logger.error?.(`mission-control: gateway WS error: ${err.message}`);
      });
    } catch (err: any) {
      api.logger.error?.(`mission-control: gateway WS connect failed: ${err.message}`);
      setTimeout(gwWsConnect, 5000);
    }
  }

  function gwWsSend(method: string, params: Record<string, unknown>): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!gwWs || !gwWsReady) {
        reject(new Error("Gateway WebSocket not connected"));
        return;
      }
      const id = `mc-${++gwWsReqId}`;
      const timer = setTimeout(() => {
        gwWsPending.delete(id);
        reject(new Error("Gateway request timeout"));
      }, 60_000);
      gwWsPending.set(id, { resolve, reject, timer });
      gwWs.send(JSON.stringify({ type: "req", id, method, params }));
    });
  }

  // Start the gateway WS connection after a brief delay to let the
  // gateway finish starting up (this plugin loads during gateway init).
  // Guard: only create one WS connection per process (gateway boot).
  // Subagent spawns re-load all plugins in the same process, which would
  // create zombie "Mission Control webchat" connections that cause bleed.
  const GW_WS_KEY = Symbol.for("mission-control-gw-ws-active");
  if (!(globalThis as any)[GW_WS_KEY]) {
    (globalThis as any)[GW_WS_KEY] = true;
    setTimeout(gwWsConnect, 2000);
  } else {
    api.logger.info?.("mission-control: skipping gateway WS (already active in this process)");
  }

  // ── Helper: aggregate token usage from session files ──────────────────

  function getSessionFiles(): string[] {
    if (!existsSync(sessionsDir)) return [];
    return readdirSync(sessionsDir)
      .filter((f) => f.endsWith(".jsonl") && !f.includes(".reset."))
      .map((f) => join(sessionsDir, f));
  }

  function aggregateTokenUsage(): {
    totalInput: number;
    totalOutput: number;
    totalCacheRead: number;
    totalSessions: number;
    bySession: Array<{
      sessionId: string;
      input: number;
      output: number;
      cacheRead: number;
      messageCount: number;
      lastActivity: string;
      model: string;
    }>;
  } {
    return cached("token-usage", () => {
      let totalInput = 0;
      let totalOutput = 0;
      let totalCacheRead = 0;
      const bySession: any[] = [];
      const sessionFiles = getSessionFiles();

      for (const file of sessionFiles) {
        const lines = parseJsonl<SessionMessage>(file);
        let sInput = 0;
        let sOutput = 0;
        let sCacheRead = 0;
        let msgCount = 0;
        let lastActivity = "";
        let model = "";
        let sessionId = "";

        for (const line of lines) {
          if (line.type === "session") {
            sessionId = (line as any).id || "";
          }
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
          bySession.push({
            sessionId,
            input: sInput,
            output: sOutput,
            cacheRead: sCacheRead,
            messageCount: msgCount,
            lastActivity,
            model,
          });
        }
      }

      bySession.sort(
        (a, b) =>
          new Date(b.lastActivity).getTime() -
          new Date(a.lastActivity).getTime(),
      );

      return {
        totalInput,
        totalOutput,
        totalCacheRead,
        totalSessions: bySession.length,
        bySession,
      };
    });
  }

  // ── Session summaries (generated lazily via local LLM) ────────────────

  const summariesFile = join(sessionsDir, "summaries.json");
  const pendingSummaries = new Set<string>();

  function loadSummaries(): Record<string, string> {
    try {
      if (existsSync(summariesFile)) return JSON.parse(readFileSync(summariesFile, "utf-8"));
    } catch {}
    return {};
  }

  function saveSummaries(summaries: Record<string, string>) {
    try { writeFileSync(summariesFile, JSON.stringify(summaries, null, 2)); } catch {}
  }

  /** Strip all gateway-injected context wrappers to find the real user text */
  function stripInjectedContext(text: string): string {
    return text
      .replace(/<daily_notes>[\s\S]*?<\/daily_notes>\s*/i, "")
      .replace(/<active_mode>\w*<\/active_mode>\s*/i, "")
      .replace(/<memory_context>[\s\S]*?<\/memory_context>\s*/i, "")
      .replace(/\[(?:[A-Z][a-z]{2} )?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?[^\]]*\]\s*/, "")
      .replace(/A new session was started via\b[\s\S]*/i, "")
      .trim();
  }

  function getFirstUserMessage(sessionId: string): string | null {
    const files = readdirSync(sessionsDir);
    const match = files.find((f) => f.startsWith(sessionId) && f.endsWith(".jsonl"));
    if (!match) return null;
    const lines = parseJsonl<SessionMessage>(join(sessionsDir, match));
    const userMsgs = lines.filter((l) => l.type === "message" && l.message?.role === "user");

    for (const msg of userMsgs) {
      if (!msg.message?.content) continue;
      let text = msg.message.content
        .filter((c: any) => c.type === "text" && c.text)
        .map((c: any) => c.text)
        .join("\n");
      if (!text) continue;

      // Strip all injected gateway context wrappers
      text = stripInjectedContext(text);
      if (!text) continue;

      // Strip cron hook wrapper: "[cron:... Hook] actual text\nCurrent time: ..."
      const cronMatch = text.match(/^\[cron:[^\]]+\]\s*/);
      if (cronMatch) text = text.slice(cronMatch[0].length);
      // Strip trailing "Current time: ..." injected by hooks
      text = text.replace(/\nCurrent time:[\s\S]*$/, "").trim();

      return text || null;
    }
    return null;
  }

  async function generateSummary(sessionId: string): Promise<string | null> {
    const text = getFirstUserMessage(sessionId);
    if (!text) return null;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);
    try {
      const resp = await fetch("http://127.0.0.1:8091/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer none" },
        body: JSON.stringify({
          model: "Qwen3.5-35B-A3B",
          messages: [
            { role: "system", content: "Summarize this conversation opener in 3-6 words. No quotes, no punctuation at the end. Just a brief title." },
            { role: "user", content: text.slice(0, 500) },
          ],
          max_tokens: 20,
          temperature: 0.3,
        }),
        signal: controller.signal,
      });
      clearTimeout(timer);
      const data = await resp.json() as any;
      const summary = data?.choices?.[0]?.message?.content?.trim();
      return summary || null;
    } catch {
      clearTimeout(timer);
      return null;
    }
  }

  function triggerSummaryGeneration(sessionId: string) {
    if (pendingSummaries.has(sessionId)) return;
    pendingSummaries.add(sessionId);
    generateSummary(sessionId).then((summary) => {
      pendingSummaries.delete(sessionId);
      if (summary) {
        const all = loadSummaries();
        all[sessionId] = summary;
        saveSummaries(all);
      }
    }).catch(() => pendingSummaries.delete(sessionId));
  }

  // ── API: /api/mc/stats ────────────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/stats",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const usage = aggregateTokenUsage();
        jsonResponse(res, {
          totalInput: usage.totalInput,
          totalOutput: usage.totalOutput,
          totalCacheRead: usage.totalCacheRead,
          totalSessions: usage.totalSessions,
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/usage-chart ──────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/usage-chart",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "/", "http://localhost");
        const range = url.searchParams.get("range") || "7d";

        const now = Date.now();
        const rangeMs: Record<string, number> = {
          "24h": 24 * 60 * 60 * 1000,
          "7d": 7 * 24 * 60 * 60 * 1000,
          "30d": 30 * 24 * 60 * 60 * 1000,
        };
        const cutoff = now - (rangeMs[range] || rangeMs["7d"]);

        // Bucket size: 1 hour for 24h, 6 hours for 7d, 1 day for 30d
        const bucketMs: Record<string, number> = {
          "24h": 60 * 60 * 1000,
          "7d": 6 * 60 * 60 * 1000,
          "30d": 24 * 60 * 60 * 1000,
        };
        const bucket = bucketMs[range] || bucketMs["7d"];

        const buckets = new Map<
          number,
          { input: number; output: number; cacheRead: number }
        >();

        const sessionFiles = getSessionFiles();
        for (const file of sessionFiles) {
          const lines = parseJsonl<SessionMessage>(file);
          for (const line of lines) {
            if (
              line.type === "message" &&
              line.message?.usage &&
              line.timestamp
            ) {
              const ts = new Date(line.timestamp).getTime();
              if (ts < cutoff) continue;
              const key = Math.floor(ts / bucket) * bucket;
              const existing = buckets.get(key) || {
                input: 0,
                output: 0,
                cacheRead: 0,
              };
              existing.input += line.message.usage.input || 0;
              existing.output += line.message.usage.output || 0;
              existing.cacheRead += line.message.usage.cacheRead || 0;
              buckets.set(key, existing);
            }
          }
        }

        const data = Array.from(buckets.entries())
          .map(([ts, vals]) => ({ ts, ...vals }))
          .sort((a, b) => a.ts - b.ts);

        jsonResponse(res, { range, bucketMs: bucket, data });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/api-calls ────────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/api-calls",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const data = cached("api-calls", () => {
          const timing = parseJsonl<TimingEntry>(timingLog, 200);
          const routing = parseJsonl<RoutingEntry>(routingLog, 200);

          // Get run_end events (completed API calls)
          const runs = timing
            .filter((e) => e.event === "run_end")
            .slice(-50)
            .map((r) => {
              // Find matching routing entry (closest timestamp)
              const rTs = new Date(r.ts).getTime();
              const route = routing.find(
                (rt) => Math.abs(new Date(rt.ts).getTime() - rTs) < 60000,
              );
              return {
                ts: r.ts,
                sessionId: r.sessionId,
                model: route?.tier || "unknown",
                totalMs: r.totalMs,
                llmMs: r.llmMs,
                toolMs: r.toolMs,
                roundTrips: r.roundTrips,
                toolCallCount: r.toolCallCount,
                success: r.success,
              };
            });

          // Also include recent tool calls
          const toolCalls = timing
            .filter((e) => e.event === "tool_call")
            .slice(-50)
            .map((t) => ({
              ts: t.ts,
              sessionId: t.sessionId,
              toolName: t.toolName,
              durationMs: t.durationMs,
            }));

          return { runs: runs.reverse(), toolCalls: toolCalls.reverse() };
        });

        jsonResponse(res, data);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/sessions ─────────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/sessions",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const usage = aggregateTokenUsage();
        const summaries = loadSummaries();
        const sessions = usage.bySession.slice(0, 100).map((s) => {
          const summary = summaries[s.sessionId];
          if (!summary) triggerSummaryGeneration(s.sessionId);
          return { ...s, summary: summary || undefined };
        });
        jsonResponse(res, { sessions });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/session-messages (query param: id) ───────────────────

  api.registerHttpRoute({
    path: "/api/mc/session-messages",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "/", "http://localhost");
        const sessionId = url.searchParams.get("id");
        if (!sessionId) {
          jsonResponse(res, { error: "Missing session id" }, 400);
          return;
        }

        // Find the session file
        const files = readdirSync(sessionsDir);
        const match = files.find(
          (f) => f.startsWith(sessionId) && f.endsWith(".jsonl"),
        );
        if (!match) {
          // No transcript file yet — session was just created but has no messages
          jsonResponse(res, { sessionId, messages: [] });
          return;
        }

        const includeTools = url.searchParams.get("tools") === "1";
        const lines = parseJsonl<SessionMessage>(join(sessionsDir, match));
        let lastUserTs: number | null = null;
        let turnHadThinking = false;
        const messages: Record<string, any>[] = [];
        for (const l of lines) {
          if (l.type !== "message" || !l.message) continue;
          const msg: any = l.message!;
          const role = msg.role;
          if (role === "user") {
            const ts = l.timestamp || msg.timestamp;
            if (ts) lastUserTs = new Date(ts).getTime();
            turnHadThinking = false;
          } else if (role === "assistant") {
            if (msg.content?.some((c: any) => c.type === "thinking")) turnHadThinking = true;
            if (!includeTools && !msg.content?.some((c: any) => c.type === "text" && c.text)) continue;
          } else if (role === "toolResult") {
            if (!includeTools) continue;
          } else {
            continue;
          }
          const entry: Record<string, any> = {
            id: l.id,
            timestamp: l.timestamp || msg.timestamp,
            role: msg.role,
            content: msg.content,
            model: msg.model,
            usage: msg.usage,
          };
          if (msg.role === "assistant") {
            const content: any[] = msg.content || [];
            entry.hasThinking = (turnHadThinking || content.some((c: any) => c.type === "thinking")) || undefined;
            const tcc = content.filter((c: any) => c.type === "toolCall").length;
            if (tcc > 0) entry.toolCallCount = tcc;
            if (lastUserTs) {
              const ts = l.timestamp || msg.timestamp;
              if (ts) entry.durationMs = new Date(ts).getTime() - lastUserTs;
            }
          }
          if (msg.role === "toolResult") {
            entry.toolCallId = msg.toolCallId;
            entry.toolName = msg.toolName;
            entry.isError = msg.isError || false;
            // Truncate tool result content to keep payloads small
            entry.content = (msg.content || []).map((c: any) => ({
              type: c.type,
              text: typeof c.text === "string" ? c.text.slice(0, 300) : c.text,
            }));
          }
          messages.push(entry);
        }

        jsonResponse(res, { sessionId, messages });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/chat ─────────────────────────────────────────────────
  // Uses the gateway's WebSocket chat.send method (same as the built-in
  // control UI) so messages go into the active agent session instead of
  // creating isolated hook sessions.

  api.registerHttpRoute({
    path: "/api/mc/chat",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        });
        res.end();
        return;
      }

      if (req.method !== "POST") {
        jsonResponse(res, { error: "Method not allowed" }, 405);
        return;
      }

      try {
        const body = JSON.parse(await readBody(req));
        const message = body.message;
        if (!message || typeof message !== "string") {
          jsonResponse(res, { error: "Missing message" }, 400);
          return;
        }

        const idempotencyKey = randomUUID();
        const result = await gwWsSend("chat.send", {
          sessionKey: "agent:main:main",
          message,
          idempotencyKey,
        });

        // Bust cache so sessions list picks up the new activity
        cache.delete("token-usage");

        jsonResponse(res, {
          ok: true,
          status: 200,
          data: result,
        });
      } catch (err: any) {
        api.logger.error?.(`mission-control: chat.send failed: ${err.message}`);
        jsonResponse(res, { error: err.message }, 502);
      }
    },
  });

  // ── API: /api/mc/chat-abort ──────────────────────────────────────────
  api.registerHttpRoute({
    path: "/api/mc/chat-abort",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        });
        res.end();
        return;
      }
      if (req.method !== "POST") {
        jsonResponse(res, { error: "Method not allowed" }, 405);
        return;
      }
      try {
        await gwWsSend("chat.abort", { sessionKey: "agent:main:main" });
        jsonResponse(res, { ok: true });
      } catch (err: any) {
        api.logger.error?.(`mission-control: chat.abort failed: ${err.message}`);
        jsonResponse(res, { error: err.message }, 502);
      }
    },
  });


  // ── API: /api/mc/subagent-abort ─────────────────────────────────────
  api.registerHttpRoute({
    path: "/api/mc/subagent-abort",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        });
        res.end();
        return;
      }
      if (req.method !== "POST") {
        jsonResponse(res, { error: "Method not allowed" }, 405);
        return;
      }
      try {
        const body = await readBody(req);
        const { sessionKey } = JSON.parse(body);
        if (!sessionKey) {
          jsonResponse(res, { error: "sessionKey required" }, 400);
          return;
        }
        await gwWsSend("chat.abort", { sessionKey });
        jsonResponse(res, { ok: true });
      } catch (err: any) {
        api.logger.error?.(`mission-control: subagent-abort failed: ${err.message}`);
        jsonResponse(res, { error: err.message }, 502);
      }
    },
  });

  // ── API: /api/mc/session-reset ────────────────────────────────────────
  // Calls the gateway's sessions.reset method (same as control UI /new).
  // Returns the new session entry so the frontend can switch to it immediately.

  api.registerHttpRoute({
    path: "/api/mc/session-reset",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        });
        res.end();
        return;
      }
      if (req.method !== "POST") {
        jsonResponse(res, { error: "Method not allowed" }, 405);
        return;
      }
      try {
        // 1. Reset the session to get the new sessionId
        const result = await gwWsSend("sessions.reset", {
          key: "agent:main:main",
          reason: "new",
        });
        cache.delete("token-usage");
        const sessionId = result?.entry?.sessionId ?? null;

        // 2. Send the startup greeting prompt (same as the built-in openclaw dashboard
        //    sends after /new). Fire-and-forget — the agent will process it async.
        const greetingPrompt =
          "A new session was started via /new or /reset. Execute your Session Startup " +
          "sequence now. Your daily notes have already been loaded into context above " +
          "— do NOT call mem_get for them. Greet the user in your configured persona, " +
          "if one is provided. Be yourself - use your defined voice, mannerisms, and " +
          "mood. Keep it to 1-3 sentences and ask what they want to do. If the runtime " +
          "model differs from default_model in the system prompt, mention the default " +
          "model. Do not mention internal steps, files, tools, or reasoning.";
        gwWsSend("chat.send", {
          sessionKey: "agent:main:main",
          message: greetingPrompt,
          idempotencyKey: randomUUID(),
        }).catch(() => {});

        jsonResponse(res, { ok: true, sessionId, data: result });
      } catch (err: any) {
        api.logger.error?.(`mission-control: sessions.reset failed: ${err.message}`);
        jsonResponse(res, { error: err.message }, 502);
      }
    },
  });

  // ── API: /api/mc/models ───────────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/models",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        if (!existsSync(modelsFile)) {
          jsonResponse(res, { providers: {} });
          return;
        }
        const models = JSON.parse(readFileSync(modelsFile, "utf-8"));
        // Strip API keys
        const safe: any = { providers: {} };
        for (const [name, provider] of Object.entries<any>(
          models.providers || {},
        )) {
          safe.providers[name] = {
            baseUrl: provider.baseUrl,
            api: provider.api,
            models: (provider.models || []).map((m: any) => ({
              id: m.id,
              name: m.name,
              contextWindow: m.contextWindow,
              maxTokens: m.maxTokens,
              reasoning: m.reasoning,
              enabled: m.enabled !== false,
              input: m.input || [],
              cost: m.cost || {},
            })),
          };
        }
        jsonResponse(res, safe);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/model-toggle ─────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/model-toggle",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") {
        jsonResponse(res, { error: "POST only" }, 405);
        return;
      }
      try {
        const body = JSON.parse(await readBody(req));
        const { provider: providerName, modelId, enabled } = body;
        if (!providerName || !modelId || typeof enabled !== "boolean") {
          jsonResponse(res, { error: "Missing provider, modelId, or enabled (boolean)" }, 400);
          return;
        }

        if (!existsSync(modelsFile)) {
          jsonResponse(res, { error: "models.json not found" }, 404);
          return;
        }

        const models = JSON.parse(readFileSync(modelsFile, "utf-8"));
        const provider = models.providers?.[providerName];
        if (!provider) {
          jsonResponse(res, { error: `Provider '${providerName}' not found` }, 404);
          return;
        }

        const model = (provider.models || []).find((m: any) => m.id === modelId);
        if (!model) {
          jsonResponse(res, { error: `Model '${modelId}' not found in provider '${providerName}'` }, 404);
          return;
        }

        model.enabled = enabled;
        writeFileSync(modelsFile, JSON.stringify(models, null, 2) + "\n");
        jsonResponse(res, { ok: true, provider: providerName, modelId, enabled });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── Tool enable/disable state ──────────────────────────────────────────

  // Tool groups definition (shared between API and hook)
  const TOOL_GROUPS = [
    {
      source: "openclaw — sessions & agents",
      tools: ["sessions_list", "sessions_history", "sessions_send", "sessions_spawn", "subagents", "session_status", "agents_list", "message"],
    },
    {
      source: "openclaw — files & runtime",
      tools: ["read", "write", "edit", "apply_patch", "exec", "process"],
    },
    {
      source: "openclaw — web & memory",
      tools: ["web_search", "web_fetch", "memory_search", "memory_get"],
    },
    {
      source: "openclaw — system & media",
      tools: ["cron", "gateway", "nodes", "browser", "canvas", "image", "tts"],
    },
    {
      source: "mcp-tools",
      tools: ["mem_search", "mem_get", "mem_write", "tag_search", "tag_explore", "vault_overview", "prefill_context", "http_search", "http_fetch", "http_request", "file_read", "file_write", "file_edit", "file_patch", "file_glob", "file_grep", "run_bash", "bg_exec", "bg_process"],
    },
    {
      source: "mcp-tools — backlog",
      tools: ["backlog_boards", "backlog_tasks", "backlog_next_task", "backlog_get_task", "backlog_update_task", "backlog_create_task"],
    },
    {
      source: "voice-tools",
      tools: ["voice_last_utterance", "voice_enroll_speaker", "voice_list_speakers"],
    },
    {
      source: "thunderbird-tools",
      tools: ["email_accounts", "email_folders", "email_search", "email_read", "email_recent", "calendar_list", "calendar_events", "contacts_search", "contacts_get"],
    },
  ];

  function loadToolsState(): Record<string, boolean> {
    try {
      if (existsSync(toolsFile)) return JSON.parse(readFileSync(toolsFile, "utf-8"));
    } catch {}
    return {};
  }

  function saveToolsState(state: Record<string, boolean>) {
    writeFileSync(toolsFile, JSON.stringify(state, null, 2) + "\n");
  }

  // ── Hooks: tool blocking + activity tracking ───────────────────────────

  api.on("before_tool_call", async (event: any, ctx: any) => {
    // Clear any pending llm_thinking reset so tool_call state shows immediately
    if (activityResetTimer) { clearTimeout(activityResetTimer); activityResetTimer = null; }
    // Track activity
    currentActivity = {
      type: "tool_call",
      toolName: event.toolName,
      startedAt: Date.now(),
      sessionKey: ctx?.sessionKey,
    };
    // Block disabled tools
    const state = loadToolsState();
    if (state[event.toolName] === false) {
      return { block: true, blockReason: `Tool "${event.toolName}" is disabled via Mission Control` };
    }
  });

  api.on("after_tool_call", async () => {
    activityResetTimer = setTimeout(() => {
      currentActivity = { type: "llm_thinking", startedAt: Date.now() };
      activityResetTimer = null;
    }, 600);
  });

  api.on("llm_input", async (event: any, ctx: any) => {
    currentActivity = {
      type: "llm_thinking",
      model: event?.model,
      startedAt: Date.now(),
      sessionKey: ctx?.sessionKey,
    };
  });

  api.on("agent_end", async () => {
    currentActivity = { type: "idle", startedAt: Date.now() };
  });

  // ── API: /api/mc/tools ─────────────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/tools",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const state = loadToolsState();
        const groups = TOOL_GROUPS.map((g) => ({
          source: g.source,
          tools: g.tools.map((name) => ({
            name,
            enabled: state[name] !== false,
          })),
        }));
        jsonResponse(res, { groups });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/tool-toggle ───────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/tool-toggle",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") {
        jsonResponse(res, { error: "POST only" }, 405);
        return;
      }
      try {
        const body = JSON.parse(await readBody(req));
        const { toolName, enabled } = body;
        if (!toolName || typeof enabled !== "boolean") {
          jsonResponse(res, { error: "Missing toolName or enabled (boolean)" }, 400);
          return;
        }
        const state = loadToolsState();
        if (enabled) {
          delete state[toolName]; // default is enabled, so remove explicit entry
        } else {
          state[toolName] = false;
        }
        saveToolsState(state);
        jsonResponse(res, { ok: true, toolName, enabled });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/skills ──────────────────────────────────────────────

  const YAML = require("/home/alansrobotlab/.npm-global/lib/node_modules/openclaw/node_modules/yaml");
  // Resolve main agent workspace from config (supports relocated workspaces)
  const mainAgentWorkspace = (() => {
    try {
      const cfg = JSON.parse(readFileSync(configFile, "utf-8"));
      const mainAgent = cfg.agents?.list?.find((a: any) => a.id === "main") || cfg.agents?.list?.[0];
      return mainAgent?.workspace?.replace(/^~/, homedir()) ?? null;
    } catch { return null; }
  })();
  const workspaceSkillsDir = mainAgentWorkspace
    ? join(mainAgentWorkspace, "skills")
    : join(homedir(), ".openclaw/workspaces/lloyd/skills");
  const bundledSkillsDir = join(homedir(), ".npm-global/lib/node_modules/openclaw/skills");

  interface SkillInfo {
    name: string;
    description: string;
    emoji?: string;
    requires?: { bins?: string[]; env?: string[]; config?: string[]; anyBins?: string[] };
    os?: string[];
    enabled: boolean;
    configured: boolean;
  }

  // Check if a binary exists on $PATH (mirrors SDK hasBinary logic)
  const binExistsCache = new Map<string, boolean>();
  function hasBinary(bin: string): boolean {
    if (binExistsCache.has(bin)) return binExistsCache.get(bin)!;
    const parts = (process.env.PATH ?? "").split(require("path").delimiter).filter(Boolean);
    for (const part of parts) {
      try {
        const candidate = join(part, bin);
        require("fs").accessSync(candidate, require("fs").constants.X_OK);
        binExistsCache.set(bin, true);
        return true;
      } catch {}
    }
    binExistsCache.set(bin, false);
    return false;
  }

  // Default config values for truthy checks (mirrors SDK defaults)
  const DEFAULT_CONFIG_VALUES: Record<string, any> = {
    "browser.enabled": true,
    "browser.evaluateEnabled": true,
  };

  function isConfigPathTruthy(config: any, pathStr: string): boolean {
    const parts = pathStr.split(".");
    let current = config;
    for (const part of parts) {
      if (current == null || typeof current !== "object") {
        // Fall back to defaults
        return Boolean(DEFAULT_CONFIG_VALUES[pathStr]);
      }
      current = current[part];
    }
    if (current == null) return Boolean(DEFAULT_CONFIG_VALUES[pathStr]);
    return Boolean(current);
  }

  // Evaluate whether all runtime requirements are met for a skill
  function checkSkillConfigured(
    requires: SkillInfo["requires"],
    os: string[] | undefined,
    config: any,
  ): boolean {
    // OS check
    if (os && os.length > 0 && !os.includes(process.platform)) return false;
    if (!requires) return true;
    // Required bins — all must exist
    for (const bin of requires.bins ?? []) {
      if (!hasBinary(bin)) return false;
    }
    // anyBins — at least one must exist
    const anyBins = requires.anyBins ?? [];
    if (anyBins.length > 0 && !anyBins.some((b) => hasBinary(b))) return false;
    // Environment variables
    for (const envName of requires.env ?? []) {
      if (!process.env[envName]) return false;
    }
    // Config paths
    for (const configPath of requires.config ?? []) {
      if (!isConfigPathTruthy(config, configPath)) return false;
    }
    return true;
  }

  function parseSkillDir(dir: string): SkillInfo[] {
    const skills: SkillInfo[] = [];
    if (!existsSync(dir)) return skills;
    // Read openclaw config for enabled/config-path checks
    let config: any = {};
    try {
      config = JSON.parse(readFileSync(configFile, "utf-8"));
    } catch {}
    const skillEntries = config?.skills?.entries ?? {};
    for (const entry of readdirSync(dir)) {
      const skillFile = join(dir, entry, "SKILL.md");
      if (!existsSync(skillFile)) continue;
      try {
        const raw = readFileSync(skillFile, "utf-8");
        const fmMatch = raw.match(/^---\n([\s\S]*?)\n---/);
        if (!fmMatch) continue;
        const fm = YAML.parse(fmMatch[1]);
        const oc = fm?.metadata?.openclaw ?? fm?.metadata?.["openclaw"] ?? {};
        const skillKey = fm.name || entry;
        const skillConfig = skillEntries[skillKey] ?? skillEntries[entry] ?? {};
        const enabled = skillConfig.enabled !== false;
        const configured = checkSkillConfigured(oc.requires, oc.os, config);
        skills.push({
          name: skillKey,
          description: fm.description || "",
          emoji: oc.emoji,
          requires: oc.requires,
          os: oc.os,
          enabled,
          configured,
        });
      } catch {}
    }
    return skills.sort((a, b) => a.name.localeCompare(b.name));
  }

  let skillsCache: { data: { workspace: SkillInfo[]; bundled: SkillInfo[] }; ts: number } | null = null;

  api.registerHttpRoute({
    path: "/api/mc/skills",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const now = Date.now();
        if (!skillsCache || now - skillsCache.ts > 30_000) {
          skillsCache = {
            data: {
              workspace: parseSkillDir(workspaceSkillsDir),
              bundled: parseSkillDir(bundledSkillsDir),
            },
            ts: now,
          };
        }
        jsonResponse(res, skillsCache.data);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/skill-toggle ─────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/skill-toggle",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") {
        jsonResponse(res, { error: "POST only" }, 405);
        return;
      }
      try {
        const body = JSON.parse(await readBody(req));
        const { skillName, enabled } = body;
        if (!skillName || typeof enabled !== "boolean") {
          jsonResponse(res, { error: "Missing skillName or enabled (boolean)" }, 400);
          return;
        }
        const config = JSON.parse(readFileSync(configFile, "utf-8"));
        if (!config.skills) config.skills = {};
        if (!config.skills.entries) config.skills.entries = {};
        if (enabled) {
          // Default is enabled — remove explicit entry (or the enabled key)
          if (config.skills.entries[skillName]) {
            delete config.skills.entries[skillName].enabled;
            // Clean up empty entry
            if (Object.keys(config.skills.entries[skillName]).length === 0) {
              delete config.skills.entries[skillName];
            }
          }
        } else {
          if (!config.skills.entries[skillName]) config.skills.entries[skillName] = {};
          config.skills.entries[skillName].enabled = false;
        }
        // Clean up empty skills.entries
        if (Object.keys(config.skills.entries).length === 0) delete config.skills.entries;
        if (Object.keys(config.skills).length === 0) delete config.skills;
        writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n");
        // Invalidate cache so next fetch reflects the change
        skillsCache = null;
        jsonResponse(res, { ok: true, skillName, enabled });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/agents ──────────────────────────────────────────────

  const workspaceDir = mainAgentWorkspace || join(rootDir, "workspaces/lloyd");

  function readFileOpt(p: string): string | null {
    try { return existsSync(p) ? readFileSync(p, "utf-8") : null; } catch { return null; }
  }

  function countSessions(agentId: string): { total: number; active: number } {
    const dir = join(rootDir, `agents/${agentId}/sessions`);
    if (!existsSync(dir)) return { total: 0, active: 0 };
    const files = readdirSync(dir);
    const active = files.filter((f) => f.endsWith(".jsonl") && !f.includes(".reset.")).length;
    const total = files.filter((f) => f.includes(".jsonl")).length;
    return { total, active };
  }

  api.registerHttpRoute({
    path: "/api/mc/agents",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const config = JSON.parse(readFileSync(configFile, "utf-8"));
        const agentList: any[] = config.agents?.list || [];
        const defaults = config.agents?.defaults || {};

        const agents = agentList.map((a: any) => {
          const id = a.id;
          const sessions = countSessions(id);
          const agentDir = join(rootDir, `agents/${id}/agent`);

          // Resolve per-agent model (per-agent field > global default)
          let primaryModel = defaults.model?.primary ?? null;
          if (a.model) {
            primaryModel = typeof a.model === "string" ? a.model : (a.model.primary ?? primaryModel);
          }

          // Read models for this agent
          const modelsPath = join(agentDir, "models.json");
          let modelCount = 0;
          let enabledModels = 0;
          if (existsSync(modelsPath)) {
            try {
              const m = JSON.parse(readFileSync(modelsPath, "utf-8"));
              for (const p of Object.values<any>(m.providers || {})) {
                for (const model of (p.models || [])) {
                  modelCount++;
                  if (model.enabled !== false) enabledModels++;
                }
              }
            } catch {}
          }

          // Read tools state
          const toolsPath = join(agentDir, "tools.json");
          let disabledTools = 0;
          if (existsSync(toolsPath)) {
            try {
              const t = JSON.parse(readFileSync(toolsPath, "utf-8"));
              disabledTools = Object.values(t).filter((v) => v === false).length;
            } catch {}
          }

          // Resolve per-agent workspace
          const agentWorkspaceDir = a.workspace
            ? a.workspace.replace(/^~/, homedir())
            : (a.default || id === agentList[0]?.id)
              ? join(rootDir, "workspaces/lloyd")
              : join(rootDir, `workspaces/${id}`);

          // Read per-agent workspace files (dynamic discovery)
          const agentWorkspace: Record<string, string | null> = {};
          const workspaceFiles: { name: string; key: string; content: string | null }[] = [];
          if (existsSync(agentWorkspaceDir)) {
            for (const entry of readdirSync(agentWorkspaceDir)) {
              if (!entry.endsWith(".md") || entry.startsWith(".")) continue;
              const fullPath = join(agentWorkspaceDir, entry);
              try { if (!statSync(fullPath).isFile()) continue; } catch { continue; }
              const key = entry.replace(/\.md$/, "").toLowerCase();
              const content = readFileOpt(fullPath);
              agentWorkspace[key] = content;
              workspaceFiles.push({ name: entry, key, content });
            }
            workspaceFiles.sort((a, b) => a.name.localeCompare(b.name));
          }

          // Extract one-liner from IDENTITY.md (line after the heading)
          const identityRaw = agentWorkspace.identity;
          let identity: string | null = null;
          if (identityRaw) {
            let lines = identityRaw.split("\n");
            // Strip YAML frontmatter (--- ... ---) if present
            if (lines[0]?.trim() === "---") {
              const end = lines.findIndex((l: string, i: number) => i > 0 && l.trim() === "---");
              if (end !== -1) lines = lines.slice(end + 1);
            }
            const oneLiner = lines.find((l: string) => l.trim() && !l.startsWith("#"));
            if (oneLiner) identity = oneLiner.trim();
          }

          return {
            id,
            name: a.name ?? id,
            avatar: discoverAvatar(id, a.name) ? `/api/mc/agent-avatar?id=${id}&name=${encodeURIComponent(a.name ?? id)}` : null,
            identity,
            primaryModel,
            modelFallbacks: (typeof a.model === "object" && a.model?.fallbacks) || null,
            sessions,
            modelCount,
            enabledModels,
            disabledTools,
            toolsAllow: a.tools?.allow ?? null,
            skills: a.skills ?? null,
            maxConcurrent: defaults.maxConcurrent ?? null,
            subagentMaxConcurrent: defaults.subagents?.maxConcurrent ?? null,
            workspace: agentWorkspace,
            workspaceFiles,
            workspacePath: agentWorkspaceDir,
          };
        });

        // Lloyd's workspace for backward compat
        const workspace: Record<string, string | null> = {
          soul: readFileOpt(join(workspaceDir, "SOUL.md")),
          identity: readFileOpt(join(workspaceDir, "IDENTITY.md")),
          agents: readFileOpt(join(workspaceDir, "AGENTS.md")),
          memory: readFileOpt(join(workspaceDir, "MEMORY.md")),
        };

        // Collect all available tool names and skill names for the UI
        const allToolGroups = TOOL_GROUPS.map((g) => ({ source: g.source, tools: g.tools }));
        const wsSkills = parseSkillDir(workspaceSkillsDir);
        const bdSkills = parseSkillDir(bundledSkillsDir);
        const allSkillNames = [...wsSkills, ...bdSkills].map((s) => s.name);

        jsonResponse(res, { agents, workspace, defaults, allToolGroups, allSkillNames });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/agent-tools-update ───────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/agent-tools-update",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { agentId, tools } = body;
        if (!agentId || typeof agentId !== "string") { jsonResponse(res, { error: "Missing agentId" }, 400); return; }
        if (tools !== null && !Array.isArray(tools)) { jsonResponse(res, { error: "tools must be string[] or null" }, 400); return; }

        const config = JSON.parse(readFileSync(configFile, "utf-8"));
        const agentList: any[] = config.agents?.list || [];
        const agent = agentList.find((a: any) => a.id === agentId);
        if (!agent) { jsonResponse(res, { error: `Agent '${agentId}' not found` }, 404); return; }

        if (tools === null) {
          if (agent.tools) { delete agent.tools.allow; if (Object.keys(agent.tools).length === 0) delete agent.tools; }
        } else {
          if (!agent.tools) agent.tools = {};
          agent.tools.allow = tools;
        }

        writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n");
        jsonResponse(res, { ok: true, agentId, tools });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/agent-skills-update ────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/agent-skills-update",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { agentId, skills } = body;
        if (!agentId || typeof agentId !== "string") { jsonResponse(res, { error: "Missing agentId" }, 400); return; }
        if (skills !== null && !Array.isArray(skills)) { jsonResponse(res, { error: "skills must be string[] or null" }, 400); return; }

        const config = JSON.parse(readFileSync(configFile, "utf-8"));
        const agentList: any[] = config.agents?.list || [];
        const agent = agentList.find((a: any) => a.id === agentId);
        if (!agent) { jsonResponse(res, { error: `Agent '${agentId}' not found` }, 404); return; }

        if (skills === null) {
          delete agent.skills;
        } else {
          agent.skills = skills;
        }

        writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n");
        jsonResponse(res, { ok: true, agentId, skills });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/agent-file-save ────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/agent-file-save",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { agentId, fileName, content } = body;
        if (!agentId || !fileName || typeof content !== "string") {
          jsonResponse(res, { error: "Missing agentId, fileName, or content" }, 400); return;
        }
        if (!fileName.endsWith(".md") || fileName.includes("/") || fileName.includes("\\") || fileName.includes("..")) {
          jsonResponse(res, { error: "Invalid fileName — must be a simple .md filename" }, 400); return;
        }

        const config = JSON.parse(readFileSync(configFile, "utf-8"));
        const agentList: any[] = config.agents?.list || [];
        const agent = agentList.find((a: any) => a.id === agentId);
        if (!agent) { jsonResponse(res, { error: `Agent '${agentId}' not found` }, 404); return; }

        const agentWorkspaceDir = agent.workspace
          ? agent.workspace.replace(/^~/, homedir())
          : join(rootDir, `workspaces/${agentId}`);

        const filePath = join(agentWorkspaceDir, fileName);
        if (!filePath.startsWith(agentWorkspaceDir)) {
          jsonResponse(res, { error: "Path traversal detected" }, 403); return;
        }

        writeFileSync(filePath, content);
        jsonResponse(res, { ok: true, agentId, fileName });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/agent-call-log ─────────────────────────────────────────

  interface AgentCallLogEntry {
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

  api.registerHttpRoute({
    path: "/api/mc/agent-call-log",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "/", "http://localhost");
        const agentId = url.searchParams.get("agentId");
        const limit = Math.min(100, parseInt(url.searchParams.get("limit") ?? "30", 10));

        if (!agentId) { jsonResponse(res, { error: "Missing agentId" }, 400); return; }
        if (!/^[a-zA-Z0-9_-]+$/.test(agentId)) { jsonResponse(res, { error: "Invalid agentId" }, 400); return; }

        const agentSessionsDir = join(rootDir, "agents", agentId, "sessions");
        if (!existsSync(agentSessionsDir)) { jsonResponse(res, { entries: [] }); return; }

        // Find the most recent session JSONL
        const sessionFiles = readdirSync(agentSessionsDir)
          .filter((f: string) => f.endsWith(".jsonl"))
          .map((f: string) => ({ name: f, mtime: statSync(join(agentSessionsDir, f)).mtimeMs }))
          .sort((a: { mtime: number }, b: { mtime: number }) => b.mtime - a.mtime);

        if (sessionFiles.length === 0) { jsonResponse(res, { entries: [] }); return; }

        const lines = parseJsonl<SessionMessage>(join(agentSessionsDir, sessionFiles[0].name));

        // Build toolCallId → result map for correlation
        const resultMap = new Map<string, { isError: boolean; preview: string }>();
        for (const line of lines) {
          if (line.type !== "message" || (line.message as any)?.role !== "toolResult") continue;
          const msg = line.message as any;
          const id = msg.toolCallId as string | undefined;
          if (!id) continue;
          const text = (msg.content ?? [])
            .filter((c: any) => c.type === "text")
            .map((c: any) => String(c.text))
            .join("")
            .slice(0, 120);
          resultMap.set(id, { isError: msg.isError ?? false, preview: text });
        }

        // Extract LLM calls and tool calls from assistant messages
        const entries: AgentCallLogEntry[] = [];
        for (const line of lines) {
          if (line.type !== "message" || line.message?.role !== "assistant") continue;
          const msg = line.message as any;
          const ts = line.timestamp ?? new Date().toISOString();
          const toolCallsInMsg = (msg.content ?? []).filter((c: any) => c.type === "toolCall");

          // Add LLM entry if usage data is present
          if (msg.usage) {
            entries.push({
              ts,
              type: "llm",
              model: msg.model ?? undefined,
              provider: msg.provider ?? undefined,
              inputTokens: msg.usage.input,
              outputTokens: msg.usage.output,
              cost: msg.usage.cost?.total ?? undefined,
              hasToolCalls: toolCallsInMsg.length > 0,
            });
          }

          // Add tool call entries
          for (const item of toolCallsInMsg) {
            const result = resultMap.get((item as any).id);
            entries.push({
              ts,
              type: "tool",
              toolName: (item as any).name,
              args: (item as any).arguments ?? {},
              isError: result?.isError ?? false,
              resultPreview: result?.preview ?? "",
            });
          }
        }

        jsonResponse(res, { entries: entries.slice(-limit) });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/agent-status ─────────────────────────────────────────

  const subagentRunsFile = join(rootDir, "subagents/runs.json");

  interface SubagentRunStatus {
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

  function loadSubagentRuns(): SubagentRunStatus[] {
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
            ? record.outcome.status
            : record.outcome,
          endedReason: record.endedReason,
          durationMs: record.endedAt && record.startedAt
            ? record.endedAt - record.startedAt
            : undefined,
        });
      }
      // Active first, then by createdAt descending
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

  api.registerHttpRoute({
    path: "/api/mc/agent-status",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        // Find main agent session state
        const mainStates: AgentSessionStatus[] = [];
        for (const [key, status] of agentSessionStates) {
          if (key.startsWith("agent:main:")) {
            mainStates.push(status);
          }
        }
        mainStates.sort((a, b) => b.lastUpdated - a.lastUpdated);
        const mainState = mainStates[0] ?? {
          sessionKey: "agent:main:main",
          state: "idle" as const,
          queueDepth: 0,
          lastUpdated: Date.now(),
        };

        // Activity description
        let activityLabel = "Idle";
        let activityDetail: string | null = null;
        if (currentActivity.type === "tool_call") {
          activityLabel = "Running tool";
          activityDetail = currentActivity.toolName ?? null;
        } else if (currentActivity.type === "llm_thinking") {
          activityLabel = "Thinking";
          activityDetail = currentActivity.model ?? null;
        }

        // Subagent runs
        const subagentRuns = loadSubagentRuns();
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
            type: currentActivity.type,
            label: activityLabel,
            detail: activityDetail,
            elapsedMs: Date.now() - currentActivity.startedAt,
          },
          heartbeat: lastHeartbeat,
          subagents: {
            active: activeRuns,
            recentCompleted,
          },
          timestamp: new Date().toISOString(),
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/gateway-sessions ────────────────────────────────────
  // Returns all gateway session states (including subagent sessions) for the
  // Activity tab to show multi-agent pipeline activity in real time.

  api.registerHttpRoute({
    path: "/api/mc/gateway-sessions",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        // 1. Collect all non-main gateway sessions from diagnostic events
        const gatewaySessions: Array<{
          sessionKey: string;
          sessionId?: string;
          agentId: string;
          state: string;
          source: "gateway";
          lastUpdated: number;
          elapsedMs: number;
        }> = [];

        for (const [key, status] of agentSessionStates) {
          // Skip main agent sessions — those are already shown
          if (key.startsWith("agent:main:")) continue;

          // Extract agentId from session key (format: agent:<agentId>:<sessionName>)
          const parts = key.split(":");
          const agentId = parts.length >= 2 && parts[0] === "agent" ? parts[1] : key;

          gatewaySessions.push({
            sessionKey: key,
            sessionId: status.sessionId,
            agentId,
            state: status.state,
            source: "gateway",
            lastUpdated: status.lastUpdated,
            elapsedMs: Date.now() - status.lastUpdated,
          });
        }

        // Sort: active first (processing/waiting), then by lastUpdated desc
        gatewaySessions.sort((a, b) => {
          const aActive = a.state !== "idle" ? 1 : 0;
          const bActive = b.state !== "idle" ? 1 : 0;
          if (aActive !== bActive) return bActive - aActive;
          return b.lastUpdated - a.lastUpdated;
        });

        // 2. Also try to fetch live subagent list from gateway via WS
        let gwSubagents: any[] = [];
        try {
          if (gwWsReady) {
            const result = await Promise.race([
              gwWsSend("subagents.list", {}),
              new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), 3000)),
            ]);
            if (result && Array.isArray((result as any).runs)) {
              gwSubagents = (result as any).runs;
            } else if (result && Array.isArray(result)) {
              gwSubagents = result as any[];
            }
          }
        } catch {
          // Gateway may not support subagents.list — that's fine, we still have
          // diagnostic-derived sessions and runs.json data
        }

        // 3. Merge runs.json data for richer lifecycle info
        const subagentRuns = loadSubagentRuns();

        jsonResponse(res, {
          sessions: gatewaySessions,
          subagentRuns: subagentRuns.slice(0, 30), // last 30 runs
          gwSubagents,
          timestamp: new Date().toISOString(),
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/health ───────────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/health",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const health: any = { gateway: "up", timestamp: new Date().toISOString() };

        // Auth profile health
        if (existsSync(authFile)) {
          const auth = JSON.parse(readFileSync(authFile, "utf-8"));
          health.auth = {};
          for (const [key, stats] of Object.entries<any>(
            auth.usageStats || {},
          )) {
            health.auth[key] = {
              errorCount: stats.errorCount,
              lastUsed: stats.lastUsed
                ? new Date(stats.lastUsed).toISOString()
                : null,
            };
          }
        }

        // Cron job health
        if (existsSync(cronJobsFile)) {
          const cron = JSON.parse(readFileSync(cronJobsFile, "utf-8"));
          health.cron = (cron.jobs || []).map((j: any) => ({
            id: j.id,
            name: j.name,
            enabled: j.enabled,
            lastStatus: j.state?.lastStatus,
            consecutiveErrors: j.state?.consecutiveErrors,
            nextRunAt: j.state?.nextRunAtMs
              ? new Date(j.state.nextRunAtMs).toISOString()
              : null,
          }));
        }

        jsonResponse(res, health);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── Service Management ──────────────────────────────────────────────

  const MANAGED_SERVICES = [
    { id: "gateway", name: "OpenClaw Gateway", unit: "openclaw-gateway.service", port: 18789 },
    { id: "llm", name: "LLM Service", unit: "lloyd-llm.service", port: 8091 },
    { id: "tts", name: "TTS Service", unit: "lloyd-tts.service", port: 8090 },
    { id: "voice-mode", name: "Voice Mode", unit: "lloyd-voice-mode.service", port: 8092 },
    { id: "tool-mcp", name: "Tool Services MCP", unit: "lloyd-tool-mcp.service", port: 8093 },
    { id: "voice-mcp", name: "Voice Services MCP", unit: "lloyd-voice-mcp.service", port: 8094 },
  ];

  function checkPort(port: number, timeoutMs = 2000): Promise<boolean> {
    return new Promise((resolve) => {
      const socket = netConnect({ host: "127.0.0.1", port });
      const timer = setTimeout(() => { socket.destroy(); resolve(false); }, timeoutMs);
      socket.on("connect", () => { clearTimeout(timer); socket.destroy(); resolve(true); });
      socket.on("error", () => { clearTimeout(timer); resolve(false); });
    });
  }

  // GET /api/mc/services
  api.registerHttpRoute({
    path: "/api/mc/services",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const services = await Promise.all(
          MANAGED_SERVICES.map(async (svc) => {
            let systemdState = "unknown";
            try {
              const result = execSync(
                `systemctl --user is-active ${svc.unit} 2>/dev/null`,
                { encoding: "utf-8", timeout: 3000 },
              ).trim();
              systemdState = result;
            } catch {
              systemdState = "inactive";
            }

            const portHealthy = await checkPort(svc.port);

            let health: string;
            if (systemdState === "active" && portHealthy) health = "healthy";
            else if (systemdState === "active" && !portHealthy) health = "degraded";
            else health = "stopped";

            return {
              id: svc.id,
              name: svc.name,
              unit: svc.unit,
              port: svc.port,
              systemdState,
              portHealthy,
              health,
            };
          }),
        );

        jsonResponse(res, { services, timestamp: new Date().toISOString() });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/services/action
  api.registerHttpRoute({
    path: "/api/mc/services/action",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        });
        res.end();
        return;
      }
      if (req.method !== "POST") {
        jsonResponse(res, { error: "POST only" }, 405);
        return;
      }
      try {
        const body = JSON.parse(await readBody(req));
        const { serviceId, action } = body;

        const svc = MANAGED_SERVICES.find((s) => s.id === serviceId);
        if (!svc) {
          jsonResponse(res, { error: `Unknown service: ${serviceId}` }, 400);
          return;
        }
        if (!["start", "stop", "restart"].includes(action)) {
          jsonResponse(res, { error: `Invalid action: ${action}` }, 400);
          return;
        }

        // For gateway restart, kill the port first to avoid conflicts
        if (svc.id === "gateway" && (action === "restart" || action === "start")) {
          try {
            execSync(`kill $(lsof -ti :18789) 2>/dev/null`, { timeout: 5000 });
          } catch {} // ok if nothing to kill
          await new Promise((r) => setTimeout(r, 2000));
        }

        execSync(`systemctl --user ${action} ${svc.unit}`, {
          encoding: "utf-8",
          timeout: 15000,
        });

        jsonResponse(res, { ok: true, serviceId, action });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/services/detail?id=<serviceId>
  api.registerHttpRoute({
    path: "/api/mc/services/detail",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const serviceId = url.searchParams.get("id");

        const svc = MANAGED_SERVICES.find((s) => s.id === serviceId);
        if (!svc) {
          jsonResponse(res, { error: `Unknown service: ${serviceId}` }, 400);
          return;
        }

        // Get full systemctl status output
        let statusOutput = "";
        try {
          statusOutput = execSync(
            `systemctl --user status ${svc.unit} 2>&1`,
            { encoding: "utf-8", timeout: 5000 },
          );
        } catch (e: any) {
          // systemctl status exits non-zero for inactive/failed services but still has output
          statusOutput = e.stdout || e.message || "Unable to get status";
        }

        // Parse useful fields from status
        const pidMatch = statusOutput.match(/Main PID:\s*(\d+)/);
        const memoryMatch = statusOutput.match(/Memory:\s*(\S+)/);
        const cpuMatch = statusOutput.match(/CPU:\s*(\S+)/);
        const activeMatch = statusOutput.match(/Active:\s*(.+)/);
        const tasksMatch = statusOutput.match(/Tasks:\s*(\S+)/);

        // Get recent journal logs
        let logs = "";
        try {
          logs = execSync(
            `journalctl --user -u ${svc.unit} -n 40 --no-pager -o short-iso 2>&1`,
            { encoding: "utf-8", timeout: 5000 },
          );
        } catch (e: any) {
          logs = e.stdout || "Unable to fetch logs";
        }

        const logLines = logs
          .split("\n")
          .filter((l: string) => l.trim() !== "")
          .slice(-40);

        jsonResponse(res, {
          id: svc.id,
          name: svc.name,
          unit: svc.unit,
          port: svc.port,
          pid: pidMatch ? parseInt(pidMatch[1], 10) : null,
          memory: memoryMatch ? memoryMatch[1] : null,
          cpu: cpuMatch ? cpuMatch[1] : null,
          tasks: tasksMatch ? tasksMatch[1] : null,
          activeSince: activeMatch ? activeMatch[1].trim() : null,
          logLines,
          rawStatus: statusOutput,
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── Memory / Vault endpoints ──────────────────────────────────────────

  const vaultRoot = join(homedir(), "obsidian");
  const qmdBin = join(homedir(), ".bun/bin/qmd");
  const VAULT_EXCLUDED = new Set([".obsidian", "templates", "images", ".trash"]);

  // ── Vault Index Cache ──────────────────────────────────────────────────

  interface VaultDoc {
    path: string;
    title: string;
    type: string;
    tags: string[];
    summary: string;
    folder: string;
  }

  let vaultIndex: VaultDoc[] = [];
  let vaultIndexTs = 0;
  const VAULT_CACHE_MS = 60_000;

  function parseFrontmatter(raw: string): Record<string, any> {
    const fm: Record<string, any> = {};
    if (!raw.startsWith("---")) return fm;
    const end = raw.indexOf("\n---", 3);
    if (end === -1) return fm;
    const block = raw.slice(4, end);
    for (const line of block.split("\n")) {
      const m = line.match(/^(\w[\w_-]*):\s*(.*)/);
      if (!m) continue;
      const [, key, val] = m;
      // Handle arrays like [tag1, tag2]
      if (val.startsWith("[") && val.endsWith("]")) {
        fm[key] = val.slice(1, -1).split(",").map((s: string) => s.trim().replace(/^["']|["']$/g, "")).filter(Boolean);
      } else {
        fm[key] = val.replace(/^["']|["']$/g, "").trim();
      }
    }
    return fm;
  }

  function walkVault(dir: string, rel: string, docs: VaultDoc[]) {
    let entries: string[];
    try { entries = readdirSync(dir); } catch { return; }
    for (const name of entries) {
      if (VAULT_EXCLUDED.has(name)) continue;
      const full = join(dir, name);
      const relPath = rel ? `${rel}/${name}` : name;
      let st;
      try { st = statSync(full); } catch { continue; }
      if (st.isDirectory()) {
        walkVault(full, relPath, docs);
      } else if (name.endsWith(".md") && st.size < 512_000) {
        try {
          const raw = readFileSync(full, "utf-8");
          const fm = parseFrontmatter(raw);
          docs.push({
            path: relPath,
            title: fm.title || name.replace(/\.md$/, ""),
            type: fm.type || "notes",
            tags: Array.isArray(fm.tags) ? fm.tags : [],
            summary: fm.summary || "",
            folder: rel || "",
          });
        } catch { /* skip unreadable */ }
      }
    }
  }

  function getVaultIndex(): VaultDoc[] {
    if (Date.now() - vaultIndexTs > VAULT_CACHE_MS) {
      const docs: VaultDoc[] = [];
      walkVault(vaultRoot, "", docs);
      vaultIndex = docs;
      vaultIndexTs = Date.now();
    }
    return vaultIndex;
  }

  // ── API: /api/mc/memory/stats ──────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/memory/stats",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const docs = getVaultIndex();
        const types: Record<string, number> = {};
        const tagCounts: Record<string, number> = {};
        for (const doc of docs) {
          types[doc.type] = (types[doc.type] || 0) + 1;
          for (const tag of doc.tags) {
            tagCounts[tag] = (tagCounts[tag] || 0) + 1;
          }
        }
        const topTags = Object.entries(tagCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 20)
          .map(([tag, count]) => ({ tag, count }));

        jsonResponse(res, {
          docCount: docs.length,
          tagCount: Object.keys(tagCounts).length,
          types,
          topTags,
          lastRefresh: new Date(vaultIndexTs).toISOString(),
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/vault-graph ───────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/vault-graph",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const docs = getVaultIndex();

        // Build node list
        const nodes = docs.map((doc) => ({
          id: doc.path,
          label: doc.title,
          type: doc.type,
          tags: doc.tags,
          folder: doc.folder,
        }));

        // Build edges from wikilinks by reading file contents
        const edges: { source: string; target: string; kind: string }[] = [];
        const pathSet = new Set(docs.map((d) => d.path));
        const pathByBasename = new Map<string, string>();
        for (const doc of docs) {
          const base = doc.path.replace(/\.md$/, "").split("/").pop()!.toLowerCase();
          pathByBasename.set(base, doc.path);
        }

        for (const doc of docs) {
          const fullPath = join(vaultRoot, doc.path);
          let raw: string;
          try { raw = readFileSync(fullPath, "utf-8"); } catch { continue; }

          // Extract [[wikilinks]]
          const wikilinkRe = /\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]/g;
          let m: RegExpExecArray | null;
          while ((m = wikilinkRe.exec(raw)) !== null) {
            const target = m[1].trim().toLowerCase();
            // Try exact path first, then basename lookup
            let targetPath: string | undefined;
            if (pathSet.has(target + ".md")) {
              targetPath = target + ".md";
            } else {
              targetPath = pathByBasename.get(target);
            }
            if (targetPath && targetPath !== doc.path) {
              edges.push({ source: doc.path, target: targetPath, kind: "wikilink" });
            }
          }
        }

        // Deduplicate edges
        const edgeKeys = new Set<string>();
        const uniqueEdges = edges.filter((e) => {
          const key = [e.source, e.target].sort().join("→");
          if (edgeKeys.has(key)) return false;
          edgeKeys.add(key);
          return true;
        });

        jsonResponse(res, { nodes, edges: uniqueEdges });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/tag-graph ────────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/tag-graph",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const docs = getVaultIndex();

        // Count tag occurrences and collect docs per tag
        const tagDocs = new Map<string, string[]>(); // tag → doc paths
        for (const doc of docs) {
          for (const tag of doc.tags) {
            let arr = tagDocs.get(tag);
            if (!arr) { arr = []; tagDocs.set(tag, arr); }
            arr.push(doc.path);
          }
        }

        // Build tag nodes
        const nodes = Array.from(tagDocs.entries()).map(([tag, docPaths]) => ({
          id: tag,
          label: tag,
          count: docPaths.length,
        }));

        // Build co-occurrence edges: two tags share an edge if they appear on the same doc
        const edges: { source: string; target: string; weight: number }[] = [];
        const edgeWeights = new Map<string, number>();
        for (const doc of docs) {
          const tags = doc.tags;
          for (let i = 0; i < tags.length; i++) {
            for (let j = i + 1; j < tags.length; j++) {
              const key = [tags[i], tags[j]].sort().join("→");
              edgeWeights.set(key, (edgeWeights.get(key) || 0) + 1);
            }
          }
        }

        for (const [key, weight] of edgeWeights) {
          const [source, target] = key.split("→");
          edges.push({ source, target, weight });
        }

        jsonResponse(res, { nodes, edges });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/memory/search ──────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/memory/search",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const query = url.searchParams.get("q") || "";
        const limit = Math.min(parseInt(url.searchParams.get("limit") || "10", 10), 30);

        if (!query || query.length < 2) {
          jsonResponse(res, { query, results: [] });
          return;
        }

        // Shell out to qmd search
        const safeQuery = query.replace(/"/g, '\\"').replace(/\$/g, '\\$');
        const cmd = `${qmdBin} search "${safeQuery}" -c obsidian -n ${limit} --json 2>/dev/null`;
        let raw: string;
        try {
          raw = execSync(cmd, { encoding: "utf-8", timeout: 5000, maxBuffer: 1024 * 1024 });
        } catch {
          jsonResponse(res, { query, results: [] });
          return;
        }

        const parsed = JSON.parse(raw);

        // Build path→summary lookup from vault index
        const docs = getVaultIndex();
        const summaryMap = new Map<string, string>();
        for (const d of docs) {
          if (d.summary) summaryMap.set(d.path, d.summary);
        }

        const results = (Array.isArray(parsed) ? parsed : []).map((r: any) => {
          const path = (r.file || "").replace(/^qmd:\/\/obsidian\//, "");
          return {
            path,
            title: r.title || "",
            score: r.score || 0,
            snippet: (r.snippet || "").replace(/@@ -\d+,?\d* @@[^\n]*\n?/g, "").trim().slice(0, 300),
            summary: summaryMap.get(path) || "",
          };
        });

        jsonResponse(res, { query, results });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/memory/tags ──────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/memory/tags",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const limit = Math.min(parseInt(url.searchParams.get("limit") || "100", 10), 200);
        const docs = getVaultIndex();
        const tagCounts: Record<string, number> = {};
        for (const doc of docs) {
          for (const tag of doc.tags) {
            tagCounts[tag] = (tagCounts[tag] || 0) + 1;
          }
        }
        const tags = Object.entries(tagCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, limit)
          .map(([tag, count]) => ({ tag, count }));

        jsonResponse(res, { tags });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/memory/browse ──────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/memory/browse",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const browsePath = (url.searchParams.get("path") || "").replace(/^\/+|\/+$/g, "");
        const fullPath = join(vaultRoot, browsePath);

        // Security: ensure path stays within vault
        if (!fullPath.startsWith(vaultRoot)) {
          jsonResponse(res, { error: "Invalid path" }, 400);
          return;
        }

        if (!existsSync(fullPath) || !statSync(fullPath).isDirectory()) {
          jsonResponse(res, { error: "Not a directory" }, 404);
          return;
        }

        const names = readdirSync(fullPath);
        const entries: any[] = [];
        for (const name of names) {
          if (VAULT_EXCLUDED.has(name) || name.startsWith(".")) continue;
          const fp = join(fullPath, name);
          let st;
          try { st = statSync(fp); } catch { continue; }

          if (st.isDirectory()) {
            // Count children
            let children = 0;
            try { children = readdirSync(fp).filter(n => !n.startsWith(".")).length; } catch { /* ok */ }
            entries.push({ name, type: "dir", children });
          } else if (name.endsWith(".md")) {
            // Quick title from frontmatter
            let title = name.replace(/\.md$/, "");
            try {
              const head = readFileSync(fp, "utf-8").slice(0, 500);
              const fm = parseFrontmatter(head);
              if (fm.title) title = fm.title;
            } catch { /* ok */ }
            entries.push({ name, type: "file", size: st.size, title });
          }
        }

        // Sort: dirs first, then files, alphabetical within each
        entries.sort((a, b) => {
          if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
          return a.name.localeCompare(b.name);
        });

        jsonResponse(res, { path: browsePath, entries });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/memory/read ──────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/memory/read",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const filePath = (url.searchParams.get("path") || "").replace(/^\/+/, "");
        const fullPath = join(vaultRoot, filePath);

        // Security: ensure path stays within vault
        if (!fullPath.startsWith(vaultRoot)) {
          jsonResponse(res, { error: "Invalid path" }, 400);
          return;
        }

        // Try exact path first; fall back to case-insensitive match on the filename,
        // then to a vault-wide basename search (handles stale index paths after moves).
        let resolvedPath = fullPath;
        if (!existsSync(resolvedPath) || statSync(resolvedPath).isDirectory()) {
          // Case-insensitive: check siblings in parent dir
          const dir = join(resolvedPath, "..");
          const base = resolvedPath.split("/").pop()!.toLowerCase();
          let found = false;
          if (existsSync(dir)) {
            for (const entry of readdirSync(dir)) {
              if (entry.toLowerCase() === base) {
                resolvedPath = join(dir, entry);
                found = true;
                break;
              }
            }
          }
          // Basename search across vault (handles moved files)
          if (!found) {
            try {
              const out = execSync(
                `find ${vaultRoot} -iname "${base}" -type f 2>/dev/null | head -1`,
                { encoding: "utf-8", timeout: 3000 },
              ).trim();
              if (out && out.startsWith(vaultRoot)) {
                resolvedPath = out;
                found = true;
              }
            } catch {}
          }
          if (!found || !existsSync(resolvedPath) || statSync(resolvedPath).isDirectory()) {
            jsonResponse(res, { error: "File not found" }, 404);
            return;
          }
        }

        const raw = readFileSync(resolvedPath, "utf-8");
        const fm = parseFrontmatter(raw);

        // Strip frontmatter from content
        let content = raw;
        if (raw.startsWith("---")) {
          const end = raw.indexOf("\n---", 3);
          if (end !== -1) content = raw.slice(end + 4).trimStart();
        }

        const actualPath = resolvedPath.startsWith(vaultRoot)
          ? resolvedPath.slice(vaultRoot.length + 1)
          : filePath;
        jsonResponse(res, {
          path: actualPath,
          frontmatter: fm,
          content,
          lineCount: raw.split("\n").length,
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/memory/save ──────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/memory/save",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") {
        jsonResponse(res, { error: "Method not allowed" }, 405);
        return;
      }
      try {
        const body = await readBody(req);
        const parsed = JSON.parse(body);
        const { path: filePath, content, frontmatter: fmEdits } = parsed;
        if (!filePath || typeof content !== "string") {
          jsonResponse(res, { error: "path and content required" }, 400);
          return;
        }

        const cleanPath = filePath.replace(/^\/+/, "");
        const fullPath = join(vaultRoot, cleanPath);

        // Security: ensure path stays within vault
        if (!fullPath.startsWith(vaultRoot)) {
          jsonResponse(res, { error: "Invalid path" }, 400);
          return;
        }

        if (!existsSync(fullPath)) {
          jsonResponse(res, { error: "File not found" }, 404);
          return;
        }

        // Read existing file and parse frontmatter
        const raw = readFileSync(fullPath, "utf-8");
        let fmBlock = "";
        if (raw.startsWith("---")) {
          const end = raw.indexOf("\n---", 3);
          if (end !== -1) fmBlock = raw.slice(0, end + 4) + "\n";
        }

        // If frontmatter edits are provided, merge them into the YAML block
        if (fmEdits && typeof fmEdits === "object" && fmBlock) {
          const fmContent = fmBlock.slice(4, fmBlock.lastIndexOf("\n---")); // between --- markers
          const lines = fmContent.split("\n");
          const updatedLines: string[] = [];
          const handled = new Set<string>();

          for (const line of lines) {
            const match = line.match(/^(\w[\w-]*):\s*/);
            if (match && match[1] in fmEdits) {
              const key = match[1];
              handled.add(key);
              const val = fmEdits[key];
              if (Array.isArray(val)) {
                updatedLines.push(`${key}: [${val.map((v: string) => v.includes(" ") || v.includes(",") ? `"${v}"` : v).join(", ")}]`);
              } else if (val !== undefined && val !== null) {
                const sv = String(val);
                updatedLines.push(`${key}: ${sv.includes(":") || sv.includes("#") ? `"${sv}"` : sv}`);
              }
              // skip if val is null/undefined (removes the field)
            } else {
              updatedLines.push(line);
            }
          }
          // Add new keys not already in frontmatter
          for (const [key, val] of Object.entries(fmEdits)) {
            if (handled.has(key) || val === undefined || val === null) continue;
            if (Array.isArray(val)) {
              updatedLines.push(`${key}: [${(val as string[]).map((v: string) => v.includes(" ") || v.includes(",") ? `"${v}"` : v).join(", ")}]`);
            } else {
              const sv = String(val);
              updatedLines.push(`${key}: ${sv.includes(":") || sv.includes("#") ? `"${sv}"` : sv}`);
            }
          }
          fmBlock = "---\n" + updatedLines.join("\n") + "\n---\n";
        }

        writeFileSync(fullPath, fmBlock + content, "utf-8");
        jsonResponse(res, { ok: true });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── Backlog (SQLite-backed) ───────────────────────────────────────────────
  // Direct SQLite access for the backlog kanban. Replaces the old ClawDeck
  // Rails proxy. Both this (Node) and tool_services.py (Python) access the
  // same SQLite file with WAL mode for safe concurrency.

  const { DatabaseSync } = require("node:sqlite") as typeof import("node:sqlite");
  const backlogDbPath = join(homedir(), ".openclaw", "data", "backlog.db");
  let backlogDb: InstanceType<typeof DatabaseSync> | null = null;

  function getBacklogDb() {
    if (!backlogDb) {
      backlogDb = new DatabaseSync(backlogDbPath);
      (backlogDb as any).exec("PRAGMA journal_mode=WAL");
      (backlogDb as any).exec("PRAGMA foreign_keys=ON");
    }
    return backlogDb as any;
  }

  function backlogTaskRow(row: any): any {
    return {
      ...row,
      blocked: !!row.blocked,
      completed: !!row.completed,
      assigned_to_agent: !!row.assigned_to_agent,
      tags: (() => { try { return JSON.parse(row.tags || "[]"); } catch { return []; } })(),
    };
  }

  // GET /api/mc/backlog/boards
  api.registerHttpRoute({
    path: "/api/mc/backlog/boards",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const db = getBacklogDb();
        const rows = db.prepare(`
          SELECT b.*, (SELECT COUNT(*) FROM tasks t WHERE t.board_id = b.id) AS tasks_count
          FROM boards b ORDER BY b.position, b.id
        `).all();
        jsonResponse(res, rows);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/backlog/tasks
  api.registerHttpRoute({
    path: "/api/mc/backlog/tasks",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const db = getBacklogDb();
        const url = new URL(req.url || "/", "http://localhost");
        const clauses: string[] = [];
        const params: any[] = [];

        const status = url.searchParams.get("status");
        if (status) { clauses.push("status = ?"); params.push(status); }
        const assigned = url.searchParams.get("assigned");
        if (assigned) { clauses.push("assigned_to_agent = ?"); params.push(assigned === "true" ? 1 : 0); }
        const blocked = url.searchParams.get("blocked");
        if (blocked) { clauses.push("blocked = ?"); params.push(blocked === "true" ? 1 : 0); }
        const boardId = url.searchParams.get("board_id");
        if (boardId) { clauses.push("board_id = ?"); params.push(Number(boardId)); }

        const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
        const rows = db.prepare(`SELECT * FROM tasks ${where} ORDER BY position, id`).all(...params);
        let tasks = rows.map(backlogTaskRow);

        const tag = url.searchParams.get("tag");
        if (tag) { tasks = tasks.filter((t: any) => t.tags?.includes(tag)); }

        jsonResponse(res, tasks);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/backlog/task-update (body: { id, status?, blocked?, name?, description?, priority?, tags?, position?, assigned_to_agent?, activity_note? })
  api.registerHttpRoute({
    path: "/api/mc/backlog/task-update",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type" });
        res.end();
        return;
      }
      if (req.method !== "POST") { jsonResponse(res, { error: "Method not allowed" }, 405); return; }
      try {
        const db = getBacklogDb();
        const body = JSON.parse(await readBody(req));
        const { id, activity_note, ...fields } = body;
        if (!id) { jsonResponse(res, { error: "Missing task id" }, 400); return; }

        const old = db.prepare("SELECT * FROM tasks WHERE id = ?").get(id);
        if (!old) { jsonResponse(res, { error: "Task not found" }, 404); return; }

        const TASK_FIELDS = ["status", "blocked", "name", "description", "priority", "tags", "due_date", "assigned_to_agent", "position"];
        const sets: string[] = [];
        const params: any[] = [];

        for (const key of TASK_FIELDS) {
          if (fields[key] !== undefined) {
            let val = fields[key];
            if (key === "tags" && Array.isArray(val)) val = JSON.stringify(val);
            if (key === "blocked" || key === "assigned_to_agent") val = val ? 1 : 0;
            sets.push(`${key} = ?`);
            params.push(val);

            // Record activity for status changes
            if (key === "status" && val !== old.status) {
              db.prepare("INSERT INTO task_activities (task_id, action, field_name, old_value, new_value, source) VALUES (?, 'moved', 'status', ?, ?, 'web')").run(id, old.status, val);
            }
          }
        }

        // Handle completed sync
        if (fields.status === "done") {
          sets.push("completed = 1", "completed_at = ?");
          params.push(new Date().toISOString());
        } else if (fields.status && old.status === "done") {
          sets.push("completed = 0", "completed_at = NULL");
        }

        if (activity_note) {
          db.prepare("INSERT INTO task_activities (task_id, action, note, source) VALUES (?, 'updated', ?, 'web')").run(id, activity_note);
        }

        if (sets.length) {
          sets.push("updated_at = ?");
          params.push(new Date().toISOString());
          params.push(id);
          db.prepare(`UPDATE tasks SET ${sets.join(", ")} WHERE id = ?`).run(...params);
        }

        const updated = db.prepare("SELECT * FROM tasks WHERE id = ?").get(id);
        jsonResponse(res, { task: backlogTaskRow(updated) });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/backlog/task-delete (body: { id })
  api.registerHttpRoute({
    path: "/api/mc/backlog/task-delete",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type" });
        res.end();
        return;
      }
      if (req.method !== "POST") { jsonResponse(res, { error: "Method not allowed" }, 405); return; }
      try {
        const db = getBacklogDb();
        const body = JSON.parse(await readBody(req));
        if (!body.id) { jsonResponse(res, { error: "Missing task id" }, 400); return; }
        db.prepare("DELETE FROM task_activities WHERE task_id = ?").run(body.id);
        db.prepare("DELETE FROM tasks WHERE id = ?").run(body.id);
        jsonResponse(res, { ok: true });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/backlog/task-create (body: { name, description?, board_id?, status?, tags?, priority? })
  api.registerHttpRoute({
    path: "/api/mc/backlog/task-create",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type" });
        res.end();
        return;
      }
      if (req.method !== "POST") { jsonResponse(res, { error: "Method not allowed" }, 405); return; }
      try {
        const db = getBacklogDb();
        const body = JSON.parse(await readBody(req));
        if (!body.name?.trim()) { jsonResponse(res, { error: "Missing task name" }, 400); return; }

        const boardId = body.board_id || db.prepare("SELECT id FROM boards ORDER BY position, id LIMIT 1").get()?.id;
        if (!boardId) { jsonResponse(res, { error: "No boards exist" }, 400); return; }

        const taskStatus = body.status || "inbox";
        const maxPos = db.prepare("SELECT COALESCE(MAX(position), 0) as m FROM tasks WHERE board_id = ? AND status = ?").get(boardId, taskStatus)?.m || 0;
        const tags = Array.isArray(body.tags) ? JSON.stringify(body.tags) : "[]";

        const result = db.prepare(
          "INSERT INTO tasks (name, description, board_id, status, priority, tags, position) VALUES (?, ?, ?, ?, ?, ?, ?)"
        ).run(body.name.trim(), body.description || "", boardId, taskStatus, body.priority || "none", tags, maxPos + 1);

        const taskId = Number(result.lastInsertRowid);
        db.prepare("INSERT INTO task_activities (task_id, action, source) VALUES (?, 'created', 'web')").run(taskId);

        const created = db.prepare("SELECT * FROM tasks WHERE id = ?").get(taskId);
        jsonResponse(res, { task: backlogTaskRow(created) });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/agent-avatar ───────────────────────────────────────────
  // Serves avatar images — checks ~/obsidian/lloyd/avatars/<id>.ext first (flat),
  // then legacy ~/obsidian/agents/<id>/avatar/ directory.

  const AVATAR_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];
  const AVATAR_MIME: Record<string, string> = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
  };
  const AVATARS_DIR = join(homedir(), "obsidian", "lloyd", "avatars");

  /** Check flat ~/obsidian/lloyd/avatars/<id>.{jpg,jpeg,png,webp} */
  function discoverAvatarFlat(agentId: string): string | null {
    if (!agentId || /[\/\.]/.test(agentId)) return null;
    for (const ext of AVATAR_EXTENSIONS) {
      const filePath = join(AVATARS_DIR, agentId + ext);
      if (existsSync(filePath)) return filePath;
    }
    return null;
  }

  /** Legacy: check ~/obsidian/agents/<id>/avatar/ directory */
  function discoverAvatarInDir(agentId: string): string | null {
    if (!agentId || /[\/\.]/.test(agentId)) return null;
    const avatarDir = join(homedir(), "obsidian", "agents", agentId, "avatar");
    if (!existsSync(avatarDir)) return null;
    let entries: string[];
    try { entries = readdirSync(avatarDir); } catch { return null; }
    for (const ext of AVATAR_EXTENSIONS) {
      const match = entries.find((e) => e.toLowerCase().endsWith(ext));
      if (match) return join(avatarDir, match);
    }
    return null;
  }

  /** Try flat avatars dir first, then legacy dir, then name fallback */
  function discoverAvatar(agentId: string, agentName?: string): string | null {
    return discoverAvatarFlat(agentId)
      || discoverAvatarInDir(agentId)
      || (agentName ? discoverAvatarFlat(agentName.toLowerCase()) : null)
      || (agentName ? discoverAvatarInDir(agentName.toLowerCase()) : null);
  }

  api.registerHttpRoute({
    path: "/api/mc/agent-avatar",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      const url = new URL(req.url || "/", "http://localhost");
      const id = url.searchParams.get("id");
      if (!id) { res.writeHead(400); res.end("Missing id"); return; }
      const name = url.searchParams.get("name") || undefined;
      const filePath = discoverAvatar(id, name);
      if (!filePath) { res.writeHead(404); res.end("No avatar found"); return; }
      try {
        const content = readFileSync(filePath);
        const ext = extname(filePath).toLowerCase();
        res.writeHead(200, {
          "Content-Type": AVATAR_MIME[ext] || "application/octet-stream",
          "Cache-Control": "public, max-age=3600",
        });
        res.end(content);
      } catch {
        res.writeHead(500); res.end("Error reading avatar");
      }
    },
  });

  // ── API: /api/mc/mode (GET) ──────────────────────────────────────────

  const MODE_STATE_PATH = join(rootDir, "extensions/mcp-tools/mode-state.json");

  function readModeState(): { currentMode: string; lastSwitchedAt: string } {
    try {
      return JSON.parse(readFileSync(MODE_STATE_PATH, "utf-8"));
    } catch {
      return { currentMode: "general", lastSwitchedAt: new Date().toISOString() };
    }
  }

  api.registerHttpRoute({
    path: "/api/mc/mode",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      jsonResponse(res, readModeState());
    },
  });

  // ── API: /api/mc/mode-set (POST) ───────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/mode-set",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "OPTIONS") {
        res.writeHead(200, {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
        });
        res.end();
        return;
      }
      if (req.method !== "POST") {
        jsonResponse(res, { error: "Method not allowed" }, 405);
        return;
      }
      try {
        const body = JSON.parse(await readBody(req));
        const mode = body.mode;
        if (!["work", "personal", "general"].includes(mode)) {
          jsonResponse(res, { error: "Invalid mode" }, 400);
          return;
        }
        const state = { currentMode: mode, lastSwitchedAt: new Date().toISOString() };
        writeFileSync(MODE_STATE_PATH, JSON.stringify(state, null, 2) + "\n");
        jsonResponse(res, state);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── API: /api/mc/commands ─────────────────────────────────────────────
  // Returns the full list of slash commands for the command picker UI.
  // Built-in commands are hardcoded (curated for MC relevance).
  // Plugin + skill commands are discovered dynamically.

  interface CommandInfo {
    name: string;
    description: string;
    category: string;
    acceptsArgs: boolean;
    source: "built-in" | "plugin" | "skill";
  }

  const BUILTIN_COMMANDS: CommandInfo[] = [
    { name: "help", description: "Show available commands.", category: "status", acceptsArgs: false, source: "built-in" },
    { name: "status", description: "Show current status.", category: "status", acceptsArgs: false, source: "built-in" },
    { name: "whoami", description: "Show your sender id.", category: "status", acceptsArgs: false, source: "built-in" },
    { name: "context", description: "Explain how context is built and used.", category: "status", acceptsArgs: true, source: "built-in" },
    { name: "export-session", description: "Export current session to HTML.", category: "status", acceptsArgs: true, source: "built-in" },
    { name: "new", description: "Start a new session.", category: "session", acceptsArgs: true, source: "built-in" },
    { name: "reset", description: "Reset the current session.", category: "session", acceptsArgs: true, source: "built-in" },
    { name: "stop", description: "Stop the current run.", category: "session", acceptsArgs: false, source: "built-in" },
    { name: "compact", description: "Compact the session context.", category: "session", acceptsArgs: true, source: "built-in" },
    { name: "session", description: "Manage session-level settings.", category: "session", acceptsArgs: true, source: "built-in" },
    { name: "model", description: "Show or set the model.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "models", description: "List model providers or provider models.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "think", description: "Set thinking level.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "verbose", description: "Toggle verbose mode.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "reasoning", description: "Toggle reasoning visibility.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "elevated", description: "Toggle elevated mode.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "exec", description: "Set exec defaults for this session.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "usage", description: "Usage footer or cost summary.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "queue", description: "Adjust queue settings.", category: "options", acceptsArgs: true, source: "built-in" },
    { name: "tts", description: "Control text-to-speech (TTS).", category: "media", acceptsArgs: true, source: "built-in" },
    { name: "restart", description: "Restart OpenClaw.", category: "tools", acceptsArgs: false, source: "built-in" },
    { name: "skill", description: "Run a skill by name.", category: "tools", acceptsArgs: true, source: "built-in" },
    { name: "config", description: "Show or set config values.", category: "management", acceptsArgs: true, source: "built-in" },
    { name: "debug", description: "Set runtime debug overrides.", category: "management", acceptsArgs: true, source: "built-in" },
    { name: "subagents", description: "Manage subagent runs.", category: "management", acceptsArgs: true, source: "built-in" },
    { name: "agents", description: "List thread-bound agents.", category: "management", acceptsArgs: false, source: "built-in" },
    { name: "kill", description: "Kill a running subagent (or all).", category: "management", acceptsArgs: true, source: "built-in" },
    { name: "steer", description: "Send guidance to a running subagent.", category: "management", acceptsArgs: true, source: "built-in" },
  ];

  let commandsCache: { data: CommandInfo[]; ts: number } | null = null;

  api.registerHttpRoute({
    path: "/api/mc/commands",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const now = Date.now();
        if (commandsCache && now - commandsCache.ts < 30_000) {
          jsonResponse(res, { commands: commandsCache.data });
          return;
        }

        const commands: CommandInfo[] = [...BUILTIN_COMMANDS];

        // Plugin commands — hardcoded (mcp-tools registers these via api.registerCommand)
        for (const cmd of [
          { name: "work", description: "Switch to work mode." },
          { name: "personal", description: "Switch to personal mode." },
          { name: "general", description: "Switch to general mode." },
          { name: "mode", description: "Show current mode." },
          { name: "local", description: "Switch to local model." },
          { name: "haiku", description: "Switch to Claude Haiku." },
          { name: "sonnet", description: "Switch to Claude Sonnet." },
          { name: "opus", description: "Switch to Claude Opus." },
        ] as const) {
          commands.push({ name: cmd.name, description: cmd.description, category: "plugin", acceptsArgs: false, source: "plugin" });
        }

        // Skill commands (from workspace skills)
        try {
          for (const skill of parseSkillDir(workspaceSkillsDir)) {
            if (skill.enabled) {
              commands.push({
                name: skill.name,
                description: skill.description || `Run ${skill.name} skill`,
                category: "skill",
                acceptsArgs: true,
                source: "skill",
              });
            }
          }
        } catch {}

        commandsCache = { data: commands, ts: now };
        jsonResponse(res, { commands });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── Static file serving for /mc/* ─────────────────────────────────────
  // Use registerHttpHandler to intercept all /mc paths (prefix-based)

  api.registerHttpHandler(async (req: IncomingMessage, res: ServerResponse) => {
    const url = new URL(req.url || "/", "http://localhost");
    const pathname = url.pathname;

    // Only handle /mc and /mc/* paths
    if (pathname !== "/mc" && !pathname.startsWith("/mc/")) return false;

    if (!existsSync(distWebDir)) {
      res.writeHead(503, { "Content-Type": "text/html" });
      res.end(
        "<h1>Mission Control</h1><p>Dashboard not built yet. Run <code>npm run build</code> in extensions/mission-control/web/</p>",
      );
      return true;
    }

    let filePath = pathname.replace(/^\/mc\/?/, "") || "index.html";

    // Security: prevent directory traversal
    if (filePath.includes("..")) {
      res.writeHead(400);
      res.end("Bad request");
      return true;
    }

    let fullPath = join(distWebDir, filePath);

    // SPA fallback: if file doesn't exist, serve index.html
    if (!existsSync(fullPath) || statSync(fullPath).isDirectory()) {
      fullPath = join(distWebDir, "index.html");
      filePath = "index.html";
    }

    try {
      const content = readFileSync(fullPath);
      const ext = extname(filePath);
      const mime = MIME_TYPES[ext] || "application/octet-stream";
      res.writeHead(200, {
        "Content-Type": mime,
        "Cache-Control":
          ext === ".html" ? "no-cache" : "public, max-age=31536000",
      });
      res.end(content);
    } catch {
      res.writeHead(404);
      res.end("Not found");
    }
    return true;
  });
}
