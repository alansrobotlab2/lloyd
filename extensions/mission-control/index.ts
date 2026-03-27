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
  // All wiggam tasks get the default pipeline — planner figures out the rest
  if (task.wiggam) return DEFAULT_PIPELINE;
  return null;
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

  // ── Idler task completion polling ───────────────────────────────────

  // Wiggam loop: track nudge state per task
  const wiggamState = new Map<number, { nudgeCount: number; lastNudgeAt: number }>();
  const MAX_NUDGES = 20;

  // Pipeline stage runner: track per-task pipeline progress
  interface PipelineState {
    stages: string[];          // ordered stage names e.g. ["plan", "implement", "review"]
    currentStageIndex: number; // which stage we're on (0-based)
    nudgeCount: number;        // nudges within current stage
    lastNudgeAt: number;
  }
  const pipelineState = new Map<number, PipelineState>();
  const IDLE_BEFORE_NUDGE_MS = 30_000; // 30s idle before nudging

  // Poll for completed idler tasks every 5 seconds.
  // Session keys are normalized by the gateway as: agent:{targetAgentId}:hook:idler:task-{taskId}
  // where targetAgentId comes from task routing (e.g. "memory", "orchestrator", "researcher").
  // We scan agentSessionStates for any key containing "hook:idler:task-{taskId}" to be robust
  // against routing changes.
  // Guard: only start one polling loop across plugin re-loads
  const IDLER_POLL_KEY = Symbol.for("mission-control-idler-poll");
  let _autonomySvc: typeof import("./autonomy-service.js") | null = null;
  function getAutonomySvc() {
    if (!_autonomySvc) _autonomySvc = require("./autonomy-service.js");
    return _autonomySvc!;
  }
  if (!(globalThis as any)[IDLER_POLL_KEY]) {
    (globalThis as any)[IDLER_POLL_KEY] = true;
  // Persistent cache for idler poll's sessions.list — survives across poll cycles.
  // Child session status changes slowly; 5 min TTL avoids hammering the gateway
  // which takes 6-16s per sessions.list scan across ~1100 JSONL files.
  let _idlerSessionsCache: { data: any[]; ts: number } | null = null;
  const IDLER_SESSIONS_TTL = 300_000;

  // Adaptive poll: reschedule with setTimeout so interval can vary each cycle
  async function runIdlerPoll() {
    let nextInterval = 30_000; // default: nothing in progress → slow poll
    try {
      const getSessionsList = async (): Promise<any[]> => {
        const now = Date.now();
        if (_idlerSessionsCache && (now - _idlerSessionsCache.ts < IDLER_SESSIONS_TTL)) {
          return _idlerSessionsCache.data;
        }
        if (!gwWsSend) return [];
        try {
          const result = await gwWsSend("sessions.list", { activeMinutes: 30, limit: 100 });
          const sessions = result?.sessions || [];
          _idlerSessionsCache = { data: sessions, ts: now };
          return sessions;
        } catch {
          return _idlerSessionsCache?.data || [];
        }
      };

      const hasActiveChildren = async (sessionKey: string): Promise<{ hasChildren: boolean; allChildrenDone: boolean }> => {
        const sessions = await getSessionsList();
        const parentSession = sessions.find((s: any) => s.key === sessionKey);
        if (!parentSession?.childSessions?.length) {
          return { hasChildren: false, allChildrenDone: false };
        }
        const childKeys: string[] = parentSession.childSessions;
        let allDone = true;
        for (const childKey of childKeys) {
          const childSession = sessions.find((s: any) => s.key === childKey);
          // If child isn't in the list, it may be too old — treat as done
          if (childSession && childSession.status !== "done") {
            allDone = false;
          }
        }
        return { hasChildren: true, allChildrenDone: allDone };
      };

      const svc = getAutonomySvc();
      const inProgressTasks = svc.getInProgressTasks();
      // Clean up wiggam state for tasks no longer in progress
      for (const taskId of Array.from(wiggamState.keys())) {
        if (!inProgressTasks.some((t: any) => t.id === taskId)) {
          wiggamState.delete(taskId);
        }
      }
      // Clean up pipeline state for tasks no longer in progress
      for (const taskId of Array.from(pipelineState.keys())) {
        if (!inProgressTasks.some((t: any) => t.id === taskId)) {
          pipelineState.delete(taskId);
        }
      }

      if (inProgressTasks.length === 0) {
        // Nothing to poll — use slow interval
        nextInterval = 30_000;
      } else {
        // Tasks in progress — use fast interval
        nextInterval = 5_000;

        for (const task of inProgressTasks) {
          const taskId = task.id;
          const suffix = `hook:idler:task-${taskId}`;

          // Find matching session state by suffix — agent prefix varies by target agent
          let sessionState: AgentSessionStatus | undefined;
          let matchingSessionKey: string | null = null;
          for (const [key, state] of agentSessionStates) {
            if (key.endsWith(suffix)) {
              sessionState = state;
              matchingSessionKey = key;
              break;
            }
          }

          if (!sessionState) {
            continue; // No OTel event yet — still running or not started
          }

          // Session states are: "idle" | "processing" | "waiting"
          // A completed cron session transitions to "idle" with queueDepth 0.
          // Debounce: require idle for 30+ seconds before nudging
          const idleAgeMs = Date.now() - (sessionState.lastUpdated ?? 0);
          
          // Skip if not idle or idle for less than IDLE_BEFORE_NUDGE_MS
          if (sessionState.state !== "idle" || (sessionState.queueDepth ?? 0) > 0 || idleAgeMs < IDLE_BEFORE_NUDGE_MS) {
            continue;
          }

          // Call hasActiveChildren once — reused across both wiggam and non-wiggam branches
          const activeChildrenResult = matchingSessionKey
            ? await hasActiveChildren(matchingSessionKey)
            : { hasChildren: false, allChildrenDone: false };

          // Non-wiggam tasks: complete on idle (original behavior), but check for active children first
          const isWiggam = !!(task as any).wiggam;
          if (!isWiggam) {
            if (matchingSessionKey) {
              const { hasChildren, allChildrenDone } = activeChildrenResult;
              if (hasChildren && !allChildrenDone) {
                continue; // Children still working — don't complete prematurely
              }
              if (hasChildren && allChildrenDone) {
                // Children done but parent is idle — completion event was lost, nudge to re-inject
                if (!gwWsSend) continue;
                try {
                  await gwWsSend("chat.send", {
                    sessionKey: matchingSessionKey,
                    message: "Your subagent(s) have completed but the completion event was not delivered. Check the results by reviewing your spawned sessions and continue the pipeline. When ALL steps are complete, include TASK_COMPLETE in your final message.",
                  });
                  api.logger.info?.(`[autonomy] Task #${taskId} — children done, parent stalled, nudging`);
                } catch (err: any) {
                  api.logger.error?.(`[autonomy] Nudge failed for task #${taskId}: ${err.message}`);
                }
                continue; // Don't complete yet — let the nudge run
              }
            }
            // No children (or no session key) — original behavior: complete on idle
            await svc.completeActiveRun(taskId, "success", "Completed");
            await svc.completeTask(taskId);
            api.logger.info?.(`[autonomy] Task #${taskId} completed via session state`);
            if (matchingSessionKey) agentSessionStates.delete(matchingSessionKey);
            continue;
          }

          // Wiggam tasks: check for active children before nudging or completing
          if (matchingSessionKey) {
            const { hasChildren: wigHasChildren, allChildrenDone: wigAllChildrenDone } = activeChildrenResult;
            if (wigHasChildren && !wigAllChildrenDone) {
              continue; // Children still working — don't nudge or complete
            }
            if (wigHasChildren && wigAllChildrenDone) {
              // Children done but parent stalled — nudge with context about lost completion event
              if (!gwWsSend) continue;
              let wigEntry = wiggamState.get(taskId);
              if (!wigEntry) {
                wigEntry = { nudgeCount: 0, lastNudgeAt: 0 };
                wiggamState.set(taskId, wigEntry);
              }
              if (wigEntry.nudgeCount < MAX_NUDGES) {
                try {
                  await gwWsSend("chat.send", {
                    sessionKey: matchingSessionKey,
                    message: "Your subagent(s) have completed but the completion event was not delivered. Check the results by reviewing your spawned sessions and continue the pipeline. When ALL steps are complete, include TASK_COMPLETE in your final message.",
                  });
                  wigEntry.nudgeCount++;
                  wigEntry.lastNudgeAt = Date.now();
                  api.logger.info?.(`[wiggam] Task #${taskId} — children done, parent stalled, nudging (${wigEntry.nudgeCount}/${MAX_NUDGES})`);
                } catch (err: any) {
                  api.logger.error?.(`[wiggam] Children-done nudge failed for task #${taskId}: ${err.message}`);
                }
              }
              continue; // Don't complete yet
            }
            // No children — fall through to normal TASK_COMPLETE / nudge logic
          }

          // ── Pipeline stage runner ──────────────────────────────────────────
          // Initialize pipeline state on first encounter of a wiggam task with a pipeline
          if (!pipelineState.has(taskId) && isWiggam) {
            const pipeline = resolvePipeline(task);
            if (pipeline) {
              pipelineState.set(taskId, {
                stages: pipeline,
                currentStageIndex: 0,
                nudgeCount: 0,
                lastNudgeAt: 0,
              });
              // Send initial stage prompt (stage 0) — this kicks off the pipeline
              const stagesDefs = loadStages();
              const firstStage = stagesDefs.find(s => s.name === pipeline[0]);
              if (firstStage && matchingSessionKey && gwWsSend) {
                // Swap model for this stage
                try {
                  await gwWsSend("sessions.patch", { key: matchingSessionKey, model: firstStage.default_model });
                } catch (err: any) {
                  api.logger.warn?.(`[pipeline] Model swap failed for task #${taskId} stage ${firstStage.name}: ${err.message}`);
                }
                // Send stage prompt
                await gwWsSend("chat.send", {
                  sessionKey: matchingSessionKey,
                  message: `## Stage: ${firstStage.name.toUpperCase()}\n\n${firstStage.content}\n\nBegin this stage now. Signal ${firstStage.signal} when complete.`,
                });
                api.logger.info?.(`[pipeline] Task #${taskId} — started pipeline, stage 0: ${firstStage.name} (${firstStage.default_model})`);
              }
              continue; // Don't process further this cycle — let the stage run
            }
          }

          // Handle active pipeline state for staged tasks
          const pState = pipelineState.get(taskId);
          if (pState) {
            const stagesDefs = loadStages();
            const currentStageName = pState.stages[pState.currentStageIndex];
            const currentStage = stagesDefs.find(s => s.name === currentStageName);
            const expectedSignal = currentStage?.signal ?? "STAGE_COMPLETE";

            // Check for stage/task completion signal
            let stageSignalFound = false;
            let stageTaskCompleteFound = false;
            try {
              if (matchingSessionKey && gwWsSend) {
                const history = await gwWsSend("chat.history", { sessionKey: matchingSessionKey, limit: 5 });
                if (history && Array.isArray(history.messages)) {
                  for (const msg of history.messages) {
                    if (msg.role === "assistant" && msg.content) {
                      const contentArr = Array.isArray(msg.content) ? msg.content : [msg.content];
                      for (const block of contentArr) {
                        const text = typeof block === "string" ? block : block?.text;
                        if (typeof text === "string") {
                          if (text.includes("TASK_COMPLETE")) { stageTaskCompleteFound = true; stageSignalFound = true; break; }
                          if (text.includes("STAGE_COMPLETE")) { stageSignalFound = true; break; }
                        }
                      }
                      if (stageSignalFound) break;
                    }
                  }
                }
              }
            } catch (err: any) {
              api.logger.warn?.(`[pipeline] chat.history failed for task #${taskId}: ${err.message}`);
            }

            if (stageTaskCompleteFound || (stageSignalFound && pState.currentStageIndex >= pState.stages.length - 1)) {
              // Pipeline complete
              await svc.completeActiveRun(taskId, "success", `Pipeline complete (${pState.stages.join(" → ")})`);
              await svc.completeTask(taskId);
              pipelineState.delete(taskId);
              wiggamState.delete(taskId);
              if (matchingSessionKey) agentSessionStates.delete(matchingSessionKey);
              api.logger.info?.(`[pipeline] Task #${taskId} completed — full pipeline: ${pState.stages.join(" → ")}`);
              continue;
            }

            if (stageSignalFound) {
              // Advance to next stage
              pState.currentStageIndex++;
              pState.nudgeCount = 0;
              pState.lastNudgeAt = 0;
              const nextStageName = pState.stages[pState.currentStageIndex];
              const nextStage = stagesDefs.find(s => s.name === nextStageName);
              if (nextStage && matchingSessionKey && gwWsSend) {
                // Swap model
                try {
                  await gwWsSend("sessions.patch", { key: matchingSessionKey, model: nextStage.default_model });
                } catch (err: any) {
                  api.logger.warn?.(`[pipeline] Model swap failed for task #${taskId} stage ${nextStage.name}: ${err.message}`);
                }
                // Send next stage prompt
                await gwWsSend("chat.send", {
                  sessionKey: matchingSessionKey,
                  message: `## Stage: ${nextStage.name.toUpperCase()}\n\n${nextStage.content}\n\nBegin this stage now. Signal ${nextStage.signal} when complete.`,
                });
                api.logger.info?.(`[pipeline] Task #${taskId} — advanced to stage ${pState.currentStageIndex}: ${nextStage.name} (${nextStage.default_model})`);
              }
              continue; // Let the new stage run
            }

            // No signal found — nudge within current stage
            pState.nudgeCount++;
            if (pState.nudgeCount >= MAX_NUDGES) {
              // Max nudges in this stage — fail the task
              await svc.completeActiveRun(taskId, "timeout", `Pipeline stalled at stage ${currentStageName} (${pState.nudgeCount} nudges)`);
              await svc.completeTask(taskId);
              pipelineState.delete(taskId);
              wiggamState.delete(taskId);
              if (matchingSessionKey) agentSessionStates.delete(matchingSessionKey);
              api.logger.info?.(`[pipeline] Task #${taskId} failed — stalled at stage ${currentStageName}`);
              continue;
            }

            pState.lastNudgeAt = Date.now();
            if (matchingSessionKey && gwWsSend) {
              try {
                await gwWsSend("chat.send", {
                  sessionKey: matchingSessionKey,
                  message: `You appear to have stalled in the ${currentStageName} stage. Continue working. When this stage is complete, output ${expectedSignal} on its own line.`,
                });
                api.logger.info?.(`[pipeline] Task #${taskId} — nudge ${pState.nudgeCount}/${MAX_NUDGES} in stage ${currentStageName}`);
              } catch (err: any) {
                api.logger.error?.(`[pipeline] Nudge failed for task #${taskId}: ${err.message}`);
                pState.nudgeCount--; // Retry on failure
              }
            }
            continue; // Handled — skip the regular wiggam logic below
          }
          // ── End pipeline stage runner ──────────────────────────────────────

          // Wiggam tasks: check for TASK_COMPLETE signal
          let taskCompleteFound = false;
          try {
            if (matchingSessionKey && gwWsSend) {
              const history = await gwWsSend("chat.history", { sessionKey: matchingSessionKey, limit: 3 });
              if (history && Array.isArray(history.messages)) {
                for (const msg of history.messages) {
                  if (msg.role === "assistant" && msg.content) {
                    const contentArr = Array.isArray(msg.content) ? msg.content : [msg.content];
                    for (const block of contentArr) {
                      const text = typeof block === "string" ? block : block?.text;
                      if (typeof text === "string" && text.includes("TASK_COMPLETE")) {
                        taskCompleteFound = true;
                        break;
                      }
                    }
                    if (taskCompleteFound) break;
                  }
                }
              }
            }
          } catch (err: any) {
            // If chat.history fails (session closed, WS disconnected), fall back to "no TASK_COMPLETE found"
            api.logger.warn?.(`[wiggam] chat.history failed for session ${matchingSessionKey}: ${err.message}`);
          }

          if (taskCompleteFound) {
            // Agent signaled completion
            await svc.completeActiveRun(taskId, "success", "Completed via TASK_COMPLETE signal");
            await svc.completeTask(taskId);
            wiggamState.delete(taskId);
            if (matchingSessionKey) agentSessionStates.delete(matchingSessionKey);
            api.logger.info?.(`[wiggam] Task #${taskId} completed (agent signaled TASK_COMPLETE)`);
            continue;
          }

          // No TASK_COMPLETE found - nudge or give up
          let wiggamEntry = wiggamState.get(taskId);
          if (!wiggamEntry) {
            wiggamEntry = { nudgeCount: 0, lastNudgeAt: 0 };
            wiggamState.set(taskId, wiggamEntry);
          }

          if (wiggamEntry.nudgeCount >= MAX_NUDGES) {
            // Max nudges exceeded - fail the task
            await svc.completeActiveRun(taskId, "timeout", "Wiggam loop: max nudges exceeded");
            await svc.completeTask(taskId);
            wiggamState.delete(taskId);
            if (matchingSessionKey) agentSessionStates.delete(matchingSessionKey);
            api.logger.info?.(`[wiggam] Task #${taskId} failed — max nudges (${MAX_NUDGES}) exceeded`);
          } else {
            // Send nudge
            wiggamEntry.nudgeCount++;
            wiggamEntry.lastNudgeAt = Date.now();
            
            if (matchingSessionKey && gwWsSend) {
              try {
                await gwWsSend("chat.send", { 
                  sessionKey: matchingSessionKey, 
                  message: "You appear to have stalled. The task is not yet complete. Review where you are in the pipeline (plan → implement → review → test → commit → report) and continue to the next step. Delegate to subagents — do not do the work yourself. When ALL steps are complete and the build passes, include TASK_COMPLETE in your final message." 
                });
                api.logger.info?.(`[wiggam] Task #${taskId} nudge ${wiggamEntry.nudgeCount}/${MAX_NUDGES}`);
              } catch (err: any) {
                api.logger.error?.(`[wiggam] chat.send failed for task #${taskId}: ${err.message}`);
                // Decrement nudge count on failure so we can retry
                wiggamEntry.nudgeCount--;
              }
            }
          }
        }
      }
    } catch (err: any) {
      api.logger.error?.(`[autonomy] Poll error: ${err.message}`);
    } finally {
      // Reschedule next poll cycle (adaptive interval)
      setTimeout(runIdlerPoll, nextInterval);
    }
  }
  // Kick off first poll cycle
  setTimeout(runIdlerPoll, 5_000);
  } // end idler-poll guard

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
