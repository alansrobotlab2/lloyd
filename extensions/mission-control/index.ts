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

  // ── Idler task completion polling ───────────────────────────────────

  // Poll for completed idler tasks every 20 seconds.
  // Session keys are normalized by the gateway as: agent:{targetAgentId}:hook:idler:task-{taskId}
  // where targetAgentId comes from task routing (e.g. "memory", "orchestrator", "researcher").
  // We scan agentSessionStates for any key containing "hook:idler:task-{taskId}" to be robust
  // against routing changes.
  setInterval(async () => {
    try {
      const { getInProgressTasks, completeActiveRun, completeTask } = await import("./autonomy-service.js");
      const inProgressTasks = await getInProgressTasks();

      for (const task of inProgressTasks) {
        const taskId = task.id;
        const suffix = `hook:idler:task-${taskId}`;

        // Find matching session state by suffix — agent prefix varies by target agent
        let sessionState: AgentSessionStatus | undefined;
        for (const [key, state] of agentSessionStates) {
          if (key.endsWith(suffix)) {
            sessionState = state;
            break;
          }
        }

        if (!sessionState) continue; // No OTel event yet — still running or not started

        // Session states are: "idle" | "processing" | "waiting"
        // A completed cron session transitions to "idle" with queueDepth 0.
        if (sessionState.state === "idle" && (sessionState.queueDepth ?? 0) <= 0) {
          await completeActiveRun(taskId, "success", "Completed");
          await completeTask(taskId);
          api.logger.info?.(`[autonomy] Task #${taskId} completed via session state`);
          // Clean up tracked session state to avoid re-processing
          for (const [key] of agentSessionStates) {
            if (key.endsWith(suffix)) {
              agentSessionStates.delete(key);
              break;
            }
          }
        }
      }
    } catch (err: any) {
      api.logger.error?.(`[autonomy] Poll error: ${err.message}`);
    }
  }, 20000);

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

  // GET /api/mc/models
  api.registerHttpRoute({
    path: "/api/mc/models",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        if (!existsSync(configFile)) { jsonResponse(res, { providers: {} }); return; }
        const config = JSON.parse(readFileSync(configFile, "utf-8"));
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
        if (!existsSync(configFile)) { jsonResponse(res, { error: "openclaw.json not found" }, 404); return; }
        const config = JSON.parse(readFileSync(configFile, "utf-8"));
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
        if (existsSync(ctx.cronJobsFile)) {
          const data = JSON.parse(readFileSync(ctx.cronJobsFile, "utf-8"));
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
        if (existsSync(runFile)) {
          const lines = readFileSync(runFile, "utf-8").trim().split("\n").filter(Boolean);
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
        if (existsSync(ctx.authFile)) {
          const auth = JSON.parse(readFileSync(ctx.authFile, "utf-8"));
          health.auth = {};
          for (const [key, stats] of Object.entries<any>(auth.usageStats || {})) {
            health.auth[key] = {
              errorCount: stats.errorCount,
              lastUsed: stats.lastUsed ? new Date(stats.lastUsed).toISOString() : null,
            };
          }
        }
        if (existsSync(ctx.cronJobsFile)) {
          const cron = JSON.parse(readFileSync(ctx.cronJobsFile, "utf-8"));
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
        if (!existsSync(certPath)) { jsonResponse(res, { error: "Certificate not found" }, 404); return; }
        const cert = readFileSync(certPath);
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

      if (!existsSync(distWebDir)) {
        res.writeHead(503, { "Content-Type": "text/html" });
        res.end("<h1>Mission Control</h1><p>Dashboard not built yet. Run <code>npm run build</code> in extensions/mission-control/web/</p>");
        return true;
      }

      let filePath = pathname.replace(/^\/mc\/?/, "") || "index.html";
      if (filePath.includes("..")) { res.writeHead(400); res.end("Bad request"); return true; }

      let fullPath = join(distWebDir, filePath);
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
