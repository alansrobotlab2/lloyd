/**
 * index.ts — Mission Control Dashboard plugin
 *
 * Serves a React dashboard at /mc/ and exposes REST API endpoints at /api/mc/*
 * for token usage stats, API call monitoring, session chat, and system health.
 *
 * This file is the thin entry point that wires together all domain modules.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, existsSync, writeFileSync, statSync } from "fs";
import { readFile as fsReadFile, access as fsAccess, stat as fsStat } from "fs/promises";
import { join, extname } from "path";

import type { PluginContext, AgentSessionStatus, AgentActivity, CommandInfo } from "./types.js";
import { MIME_TYPES, jsonResponse, readBody, handleCorsOptions, requirePost } from "./helpers.js";
import { registerVoiceRoutes } from "./voice.js";
import { setupGateway, getTriggerSummaryGeneration } from "./gateway.js";
import { registerSessionRoutes } from "./sessions.js";
import { registerServiceRoutes } from "./services.js";
import { registerMemoryRoutes } from "./memory.js";
import { resolveSkillDirs, parseSkillDir, registerSkillRoutes } from "./skills.js";
import { registerAgentRoutes } from "./agents.js";
import { registerBacklogRoutes } from "./backlog.js";
import { registerAutonomyRoutes } from "./autonomy.js";
import { registerArchitectureRoutes } from "./architecture.js";

// ── Stage definitions ────────────────────────────────────────────────
interface StageDefinition {
  name: string;
  default_model: string;
  signal: string;
  content: string;
}

const STAGES_DIR = join(require("os").homedir(), "obsidian", "agents", "worker", "stages");
let _stagesCache: { stages: StageDefinition[]; ts: number } | null = null;
const STAGES_CACHE_TTL = 60_000; // 1 min

function loadStages(): StageDefinition[] {
  const now = Date.now();
  if (_stagesCache && (now - _stagesCache.ts < STAGES_CACHE_TTL)) return _stagesCache.stages;

  const { readdirSync: _readdirSync, readFileSync: _readFileSync } = require("fs");
  const stages: StageDefinition[] = [];
  try {
    const files = _readdirSync(STAGES_DIR).filter((f: string) => f.endsWith(".md") && f !== "index.md");
    for (const file of files) {
      const raw = _readFileSync(join(STAGES_DIR, file), "utf-8");
      const fmMatch = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
      if (!fmMatch) continue;
      const fm = fmMatch[1];
      const content = fmMatch[2].trim();
      const name = fm.match(/^name:\s*(.+)$/m)?.[1]?.trim() ?? file.replace(".md", "");
      const default_model = fm.match(/^default_model:\s*(.+)$/m)?.[1]?.trim() ?? "anthropic/claude-sonnet-4-6";
      const signal = fm.match(/^signal:\s*(.+)$/m)?.[1]?.trim() ?? "STAGE_COMPLETE";
      stages.push({ name, default_model, signal, content });
    }
  } catch { /* stages dir doesn't exist yet */ }
  _stagesCache = { stages, ts: now };
  return stages;
}

// ── Pipeline resolution ───────────────────────────────────────────────
const DEFAULT_PIPELINE = ["plan", "implement", "review"];

function resolvePipeline(task: any): string[] | null {
  // Explicit pipeline field overrides default
  if (task.pipeline) {
    try {
      const parsed = JSON.parse(task.pipeline);
      if (Array.isArray(parsed) && parsed.every((s: any) => typeof s === "string")) return parsed;
    } catch { /* fall through */ }
  }
  // All pipeline_mode tasks get the default pipeline — planner figures out the rest
  if (task.pipeline_mode) return DEFAULT_PIPELINE;
  return null;
}

// ── Notification Store ───────────────────────────────────────────────
interface Notification {
  id: string;
  text: string;
  level: "info" | "success" | "warning" | "error";
  timestamp: number;
}

const NOTIFICATION_MAX = 50;
const NOTIFICATION_TTL_MS = 5 * 60 * 1000; // 5 minutes
const notificationStore: Notification[] = [];
const notificationSseClients = new Set<ServerResponse>();

function pruneNotifications(): void {
  const cutoff = Date.now() - NOTIFICATION_TTL_MS;
  let i = 0;
  while (i < notificationStore.length && notificationStore[i].timestamp < cutoff) i++;
  if (i > 0) notificationStore.splice(0, i);
  while (notificationStore.length > NOTIFICATION_MAX) notificationStore.shift();
}

function broadcastNotification(notification: Notification): void {
  const data = JSON.stringify(notification);
  for (const res of notificationSseClients) {
    try { res.write(`event: notification\ndata: ${data}\n\n`); } catch { /* client gone */ }
  }
}

export default function register(api: OpenClawPluginApi) {
  const rootDir = join(__dirname, "../..");
  const configFile = join(rootDir, "openclaw.json");
  const distWebDir = join(__dirname, "dist-web");

  // ── Build plugin context ────────────────────────────────────────────

  const agentSessionStates = new Map<string, AgentSessionStatus>();
  const currentActivity = { value: { type: "idle", startedAt: Date.now() } as AgentActivity };
  const lastHeartbeat = { value: null as { active: number; waiting: number; queued: number } | null };
  const activityResetTimer = { value: null as ReturnType<typeof setTimeout> | null };

  const ctx: PluginContext = {
    api,
    rootDir,
    configFile,
    sessionsDir: join(rootDir, "agents/main/sessions"),
    distWebDir,
    timingLog: join(rootDir, "logs/timing.jsonl"),
    routingLog: join(rootDir, "logs/routing.jsonl"),
    ccInstancesDir: join(rootDir, "logs/cc-instances"),
    modelsFile: join(rootDir, "agents/main/agent/models.json"),
    toolsFile: join(rootDir, "agents/main/agent/tools.json"),
    authFile: join(rootDir, "agents/main/agent/auth-profiles.json"),
    cronJobsFile: join(rootDir, "cron/jobs.json"),
    summariesFile: join(rootDir, "data/session-summaries.json"),
    agentSessionStates,
    currentActivity,
    lastHeartbeat,
    activityResetTimer,
  };

  api.logger.info?.("mission-control: loaded");

  // ── Diagnostic event tracking ─────────────────────────────────────

  try {
    const { onDiagnosticEvent } = require("openclaw/plugin-sdk");
    onDiagnosticEvent((evt: any) => {
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
          currentActivity.value = { type: "idle", startedAt: Date.now() };
        }
      }
      if (evt.type === "diagnostic.heartbeat") {
        lastHeartbeat.value = { active: evt.active, waiting: evt.waiting, queued: evt.queued };
      }
    });
  } catch {
    api.logger.info?.("mission-control: diagnostics-otel unavailable — session state tracking disabled");
  }

  // ── Idler task completion polling (removed) ──────────────────────
  // Completion detection now lives entirely in the idler daemon (main.py).
  // MC no longer polls for task completion or nudges stalled sessions.

  // ── Notification endpoints ────────────────────────────────────────

  // POST /api/mc/notify — store and broadcast a notification
  api.registerHttpRoute({
    path: "/api/mc/notify",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { text, level = "info" } = body;
        if (!text || typeof text !== "string") { jsonResponse(res, { error: "Missing text" }, 400); return; }
        const validLevels = ["info", "success", "warning", "error"];
        const safeLevel = validLevels.includes(level) ? level as Notification["level"] : "info";
        pruneNotifications();
        const notification: Notification = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          text,
          level: safeLevel,
          timestamp: Date.now(),
        };
        notificationStore.push(notification);
        broadcastNotification(notification);
        jsonResponse(res, { ok: true, id: notification.id });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/notifications — current non-expired notifications
  api.registerHttpRoute({
    path: "/api/mc/notifications",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      pruneNotifications();
      jsonResponse(res, { notifications: [...notificationStore] });
    },
  });

  // GET /api/mc/notification-stream — SSE stream for real-time notifications
  api.registerHttpRoute({
    path: "/api/mc/notification-stream",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "GET") { jsonResponse(res, { error: "GET only" }, 405); return; }
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "http://localhost:5173",
      });
      res.write(":\n\n"); // SSE comment to establish connection
      notificationSseClients.add(res);
      // Send any recent notifications on connect (last 60s worth)
      const recent = notificationStore.filter(n => Date.now() - n.timestamp < 60_000);
      for (const n of recent) {
        res.write(`event: notification\ndata: ${JSON.stringify(n)}\n\n`);
      }
      req.on("close", () => { notificationSseClients.delete(res); });
    },
  });

  // ── MC HTTP API for pipeline orchestration ────────────────────────

  // GET /api/mc/session-status?key=...
  // Returns session state from OTel diagnostic events (agentSessionStates map).
  // Used by idler pipeline.py to poll for stage completion.
  api.registerHttpRoute({
    path: "/api/mc/session-status",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      const url = new URL(req.url || "/", "http://localhost");
      const key = url.searchParams.get("key");
      if (!key) {
        jsonResponse(res, { error: "Missing key parameter" }, 400);
        return;
      }

      // Check OTel state — try exact match first
      let state = agentSessionStates.get(key);

      // If not found by exact key, try suffix match (session keys get prefixed by gateway)
      if (!state) {
        for (const [stateKey, stateVal] of agentSessionStates) {
          if (stateKey.endsWith(key) || key.endsWith(stateKey)) {
            state = stateVal;
            break;
          }
        }
      }

      if (state) {
        jsonResponse(res, {
          status: state.state,  // "idle" | "processing" | "waiting"
          lastUpdated: state.lastUpdated,
          queueDepth: state.queueDepth,
          sessionKey: state.sessionKey,  // resolved full key — useful for callers
        });
        return;
      }
      jsonResponse(res, { status: "unknown" });
    },
  });

  // GET /api/mc/session-history?key=...&limit=N
  // Proxies gateway's chat.history WebSocket command to HTTP.
  // Used by idler pipeline.py to extract stage output.
  api.registerHttpRoute({
    path: "/api/mc/session-history",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      const url = new URL(req.url || "/", "http://localhost");
      const key = url.searchParams.get("key");
      const limit = parseInt(url.searchParams.get("limit") || "10", 10);
      if (!key) {
        jsonResponse(res, { error: "Missing key parameter" }, 400);
        return;
      }

      // gwWsSend is defined later (after setupGateway) — but this handler
      // is only called at runtime, not at registration time, so it's fine.
      if (!gwWsSend) {
        jsonResponse(res, { error: "Gateway WebSocket not connected" }, 503);
        return;
      }

      // Resolve the full session key — gateway prefixes keys with "agent:{agentId}:"
      // Try exact match first, then suffix match against agentSessionStates
      let resolvedKey = key;
      if (!agentSessionStates.has(key)) {
        for (const stateKey of agentSessionStates.keys()) {
          if (stateKey.endsWith(key) || key.endsWith(stateKey)) {
            resolvedKey = stateKey;
            break;
          }
        }
      }

      try {
        const result = await gwWsSend("chat.history", { sessionKey: resolvedKey, limit });
        jsonResponse(res, result);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── Wire up domain modules ────────────────────────────────────────

  // Voice pipeline (must be before gateway — gateway imports broadcastSse)
  registerVoiceRoutes(ctx);

  // Gateway WebSocket client
  const { gwWsSend } = setupGateway(ctx);
  const triggerSummaryGeneration = getTriggerSummaryGeneration(gwWsSend);

  // Sessions, stats, chat
  registerSessionRoutes(ctx, { gwWsSend, triggerSummaryGeneration });

  // Service management
  registerServiceRoutes(ctx);

  // Memory / vault
  registerMemoryRoutes(ctx);

  // Skills
  const skillDirs = resolveSkillDirs(configFile);
  registerSkillRoutes(ctx, skillDirs);

  // Agents, tools, avatars
  registerAgentRoutes(ctx, skillDirs);

  // Backlog
  registerBacklogRoutes(ctx);

  // Autonomy
  registerAutonomyRoutes(ctx);

  // Architecture
  registerArchitectureRoutes(ctx);

  // ── Small endpoints (not worth a separate module) ─────────────────

  // Async file helpers for HTTP handlers
  async function fileExists(p: string): Promise<boolean> {
    try { await fsAccess(p); return true; } catch { return false; }
  }

  // GET /api/mc/models
  api.registerHttpRoute({
    path: "/api/mc/models",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        if (!await fileExists(configFile)) { jsonResponse(res, { providers: {} }); return; }
        const config = JSON.parse(await fsReadFile(configFile, "utf-8"));
        const providers = config.models?.providers || {};
        const safe: any = { providers: {} };
        for (const [name, provider] of Object.entries<any>(providers)) {
          safe.providers[name] = {
            baseUrl: provider.baseUrl,
            api: provider.api,
            models: (provider.models || []).map((m: any) => ({
              id: m.id, name: m.name, contextWindow: m.contextWindow,
              maxTokens: m.maxTokens, reasoning: m.reasoning,
              enabled: m.enabled !== false, input: m.input || [], cost: m.cost || {},
            })),
          };
        }
        jsonResponse(res, safe);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/model-toggle
  api.registerHttpRoute({
    path: "/api/mc/model-toggle",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { provider: providerName, modelId, enabled } = body;
        if (!providerName || !modelId || typeof enabled !== "boolean") {
          jsonResponse(res, { error: "Missing provider, modelId, or enabled (boolean)" }, 400); return;
        }
        if (!await fileExists(configFile)) { jsonResponse(res, { error: "openclaw.json not found" }, 404); return; }
        const config = JSON.parse(await fsReadFile(configFile, "utf-8"));
        if (!config.models) config.models = {};
        if (!config.models.providers) config.models.providers = {};
        const provider = config.models.providers[providerName];
        if (!provider) { jsonResponse(res, { error: `Provider '${providerName}' not found` }, 404); return; }
        const model = (provider.models || []).find((m: any) => m.id === modelId);
        if (!model) { jsonResponse(res, { error: `Model '${modelId}' not found in provider '${providerName}'` }, 404); return; }
        model.enabled = enabled;
        writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n");
        jsonResponse(res, { ok: true, provider: providerName, modelId, enabled });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/cron-jobs
  api.registerHttpRoute({
    path: "/api/mc/cron-jobs",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        let jobs: any[] = [];
        if (await fileExists(ctx.cronJobsFile)) {
          const data = JSON.parse(await fsReadFile(ctx.cronJobsFile, "utf-8"));
          jobs = data.jobs || [];
        }
        jsonResponse(res, { jobs });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/cron-runs
  api.registerHttpRoute({
    path: "/api/mc/cron-runs",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "/", "http://localhost");
        const jobId = url.searchParams.get("jobId");
        if (!jobId) { jsonResponse(res, { error: "Missing jobId parameter" }, 400); return; }
        const limit = parseInt(url.searchParams.get("limit") || "20", 10);
        const runFile = join(rootDir, "cron/runs", `${jobId}.jsonl`);
        let runs: any[] = [];
        if (await fileExists(runFile)) {
          const lines = (await fsReadFile(runFile, "utf-8")).trim().split("\n").filter(Boolean);
          const parsed: any[] = [];
          for (const line of lines) {
            try { const entry = JSON.parse(line); if (entry.action === "finished") parsed.push(entry); } catch { /* skip */ }
          }
          runs = parsed.slice(-limit).reverse();
        }
        jsonResponse(res, { jobId, runs });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/health
  api.registerHttpRoute({
    path: "/api/mc/health",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const health: any = { gateway: "up", timestamp: new Date().toISOString() };
        if (await fileExists(ctx.authFile)) {
          const auth = JSON.parse(await fsReadFile(ctx.authFile, "utf-8"));
          health.auth = {};
          for (const [key, stats] of Object.entries<any>(auth.usageStats || {})) {
            health.auth[key] = {
              errorCount: stats.errorCount,
              lastUsed: stats.lastUsed ? new Date(stats.lastUsed).toISOString() : null,
            };
          }
        }
        if (await fileExists(ctx.cronJobsFile)) {
          const cron = JSON.parse(await fsReadFile(ctx.cronJobsFile, "utf-8"));
          health.cron = (cron.jobs || []).map((j: any) => ({
            id: j.id, name: j.name, enabled: j.enabled,
            lastStatus: j.state?.lastStatus,
            consecutiveErrors: j.state?.consecutiveErrors,
            nextRunAt: j.state?.nextRunAtMs ? new Date(j.state.nextRunAtMs).toISOString() : null,
          }));
        }
        jsonResponse(res, health);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/mode
  const MODE_STATE_PATH = join(rootDir, "extensions/mcp-tools/mode-state.json");

  function readModeState(): { currentMode: string; lastSwitchedAt: string } {
    try { return JSON.parse(readFileSync(MODE_STATE_PATH, "utf-8")); }
    catch { return { currentMode: "general", lastSwitchedAt: new Date().toISOString() }; }
  }

  api.registerHttpRoute({
    path: "/api/mc/mode",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => { jsonResponse(res, readModeState()); },
  });

  // POST /api/mc/mode-set
  api.registerHttpRoute({
    path: "/api/mc/mode-set",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
      try {
        const body = JSON.parse(await readBody(req));
        const mode = body.mode;
        if (!["work", "personal", "general"].includes(mode)) { jsonResponse(res, { error: "Invalid mode" }, 400); return; }
        const state = { currentMode: mode, lastSwitchedAt: new Date().toISOString() };
        writeFileSync(MODE_STATE_PATH, JSON.stringify(state, null, 2) + "\n");
        jsonResponse(res, state);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/commands
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
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const now = Date.now();
        if (commandsCache && now - commandsCache.ts < 30_000) { jsonResponse(res, { commands: commandsCache.data }); return; }
        const commands: CommandInfo[] = [...BUILTIN_COMMANDS];
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
        try {
          for (const skill of skillDirs.workspaceSkillsDirs.flatMap(d => parseSkillDir(d, configFile, api.logger))) {
            if (skill.enabled) {
              commands.push({ name: skill.name, description: skill.description || `Run ${skill.name} skill`, category: "skill", acceptsArgs: true, source: "skill" });
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

  // GET /api/mc/cert
  api.registerHttpRoute({
    path: "/api/mc/cert",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const certPath = join(require("os").homedir(), ".openclaw", "certs", "mc.crt");
        if (!await fileExists(certPath)) { jsonResponse(res, { error: "Certificate not found" }, 404); return; }
        const cert = readFileSync(certPath); // binary read — keep sync
        res.writeHead(200, {
          "Content-Type": "application/x-x509-ca-cert",
          "Content-Disposition": 'attachment; filename="openclaw-mc.crt"',
          "Content-Length": cert.length,
        });
        res.end(cert);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── Static file serving for /mc/* (must be last) ──────────────────

  api.registerHttpRoute({
    path: "/mc",
    match: "prefix",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      const url = new URL(req.url || "/", "http://localhost");
      const pathname = url.pathname;

      if (!await fileExists(distWebDir)) {
        res.writeHead(503, { "Content-Type": "text/html" });
        res.end("<h1>Mission Control</h1><p>Dashboard not built yet. Run <code>npm run build</code> in extensions/mission-control/web/</p>");
        return true;
      }

      let filePath = pathname.replace(/^\/mc\/?/, "") || "index.html";
      if (filePath.includes("..")) { res.writeHead(400); res.end("Bad request"); return true; }

      let fullPath = join(distWebDir, filePath);
      try {
        const st = await fsStat(fullPath);
        if (st.isDirectory()) { fullPath = join(distWebDir, "index.html"); filePath = "index.html"; }
      } catch {
        fullPath = join(distWebDir, "index.html");
        filePath = "index.html";
      }

      try {
        const content = readFileSync(fullPath);
        const ext = extname(filePath);
        const mime = MIME_TYPES[ext] || "application/octet-stream";
        res.writeHead(200, {
          "Content-Type": mime,
          "Cache-Control": ext === ".html" ? "no-cache" : "public, max-age=31536000",
        });
        res.end(content);
      } catch {
        res.writeHead(404);
        res.end("Not found");
      }
      return true;
    },
  });
}
