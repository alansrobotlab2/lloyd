/**
 * pipeline-hooks.ts — Pipeline stage management via gateway hooks
 *
 * Registers context, llm_output, and agent_end hooks to:
 * - Inject stage-specific subliminal prompts into each LLM call (context hook)
 * - Detect signals and manage stage lifecycle in agent_end (via wiggam hook continuation)
 * - Update autonomy DB on completion/failure
 * - Expose POST /api/mc/pipeline-init for idler to initialize pipelines
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import os from "os";

import type { PluginContext } from "./types.js";
import { jsonResponse, readBody } from "./helpers.js";
import {
  completeTask,
  completeActiveRun,
  writeTask,
} from "./autonomy-service.js";

// ── Types ──────────────────────────────────────────────────────────────────

interface StageDefinition {
  name: string;
  default_model: string;
  signal: string;
  content: string;
}

interface PipelineState {
  taskId: number;
  stages: string[];
  currentStageIndex: number;
  stageDefinitions: Map<string, StageDefinition>;
  skills: string[];  // Array of skill content strings, loaded at init time
}

// ── In-memory pipeline registry ────────────────────────────────────────────
// Keyed by sessionKey (e.g. "hook:idler:task-123")
const activePipelines = new Map<string, PipelineState>();

// ── Key lookup helpers ─────────────────────────────────────────────────────
// The gateway prefixes session keys with "agent:{agentId}:" (e.g.
// "agent:worker:hook:idler:task-52") but pipeline-init stores them without the
// prefix (e.g. "hook:idler:task-52"). These helpers resolve either form.

function findPipeline(sessionKey: string): PipelineState | undefined {
  // Exact match first (handles already-prefixed keys if stored that way)
  const exact = activePipelines.get(sessionKey);
  if (exact) return exact;

  // Suffix match — gateway prefixes keys with "agent:{agentId}:"
  for (const [key, state] of activePipelines) {
    if (sessionKey.endsWith(key) || key.endsWith(sessionKey)) {
      return state;
    }
  }
  return undefined;
}

function findPipelineKey(sessionKey: string): string | undefined {
  if (activePipelines.has(sessionKey)) return sessionKey;
  for (const key of activePipelines.keys()) {
    if (sessionKey.endsWith(key) || key.endsWith(sessionKey)) {
      return key;
    }
  }
  return undefined;
}

// ── Skill loading ───────────────────────────────────────────────────────────

const SKILLS_DIR = join(os.homedir(), "obsidian", "skills");

function loadSkillContent(skillPath: string): string | null {
  // skillPath can be:
  // - A skill name like "pipeline-dispatch" → resolves to SKILLS_DIR/pipeline-dispatch/SKILL.md
  // - A relative path like "pipeline-dispatch/SKILL.md"
  // - An absolute path
  try {
    let fullPath: string;
    if (skillPath.startsWith("/")) {
      fullPath = skillPath;
    } else if (skillPath.includes("/")) {
      fullPath = join(SKILLS_DIR, skillPath);
    } else {
      fullPath = join(SKILLS_DIR, skillPath, "SKILL.md");
    }
    const raw = readFileSync(fullPath, "utf-8");
    // Strip frontmatter
    const fmMatch = raw.match(/^---\n[\s\S]*?\n---\n([\s\S]*)$/);
    return fmMatch ? fmMatch[1].trim() : raw.trim();
  } catch {
    return null;
  }
}

// ── Stage loading ───────────────────────────────────────────────────────────

const STAGES_DIR = join(os.homedir(), "obsidian", "agents", "worker", "stages");
let _stagesCache: { stages: StageDefinition[]; ts: number } | null = null;
const STAGES_CACHE_TTL = 60_000;

function loadStages(): StageDefinition[] {
  const now = Date.now();
  if (_stagesCache && now - _stagesCache.ts < STAGES_CACHE_TTL) return _stagesCache.stages;

  const stages: StageDefinition[] = [];
  try {
    const files = readdirSync(STAGES_DIR).filter(
      (f: string) => f.endsWith(".md") && f !== "index.md",
    );
    for (const file of files) {
      const raw = readFileSync(join(STAGES_DIR, file), "utf-8");
      const fmMatch = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
      if (!fmMatch) continue;
      const fm = fmMatch[1];
      const content = fmMatch[2].trim();
      const name = fm.match(/^name:\s*(.+)$/m)?.[1]?.trim() ?? file.replace(".md", "");
      const default_model =
        fm.match(/^default_model:\s*(.+)$/m)?.[1]?.trim() ?? "anthropic/claude-sonnet-4-6";
      const signal = fm.match(/^signal:\s*(.+)$/m)?.[1]?.trim() ?? "STAGE_COMPLETE";
      stages.push({ name, default_model, signal, content });
    }
  } catch {
    /* stages dir might not exist yet */
  }
  _stagesCache = { stages, ts: now };
  return stages;
}

// ── Gateway HTTP helpers ────────────────────────────────────────────────────

function getGatewayConfig(configFile: string): { url: string; token: string } {
  try {
    const cfg = JSON.parse(readFileSync(configFile, "utf-8"));
    const port = cfg.gateway?.port ?? 18789;
    const tls = cfg.gateway?.tls !== false;
    const url = `${tls ? "https" : "http"}://127.0.0.1:${port}`;
    const tok = cfg.gateway?.auth?.token;
    let token = "";
    if (typeof tok === "string") token = tok;
    else if (tok?.source === "env" && tok?.id) token = process.env[tok.id] || "";
    return { url, token };
  } catch {
    return { url: "https://127.0.0.1:18789", token: "" };
  }
}

async function gatewayPost(
  configFile: string,
  path: string,
  body: unknown,
): Promise<{ ok: boolean; data?: any; error?: string }> {
  const { url, token } = getGatewayConfig(configFile);
  const fullUrl = `${url}${path}`;
  try {
    const { Agent: HttpsAgent } = await import("node:https");
    const agent = new HttpsAgent({ rejectUnauthorized: false });
    const resp = await fetch(fullUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
      // @ts-ignore — undici/native fetch accepts agent option in Node ≥ 18
      dispatcher: undefined,
      agent,
    } as any);
    const data = await resp.json().catch(() => ({}));
    return { ok: resp.ok, data };
  } catch (err: any) {
    return { ok: false, error: err.message };
  }
}

// ── Helper: push notification ───────────────────────────────────────────────

async function pushNotification(
  configFile: string,
  text: string,
  level: "info" | "success" | "warning" | "error",
  logger: any,
): Promise<void> {
  const result = await gatewayPost(configFile, "/api/mc/notify", { text, level });
  if (!result.ok) {
    logger.warn?.(`pipeline-hooks: notification failed: ${result.error ?? "unknown"}`);
  }
}

async function announceToMain(
  configFile: string,
  text: string,
  logger: any,
): Promise<void> {
  // /hooks/wake uses hooks.token, not gateway.auth.token
  let hooksToken = "";
  try {
    const cfg = JSON.parse(readFileSync(configFile, "utf-8"));
    hooksToken = cfg.hooks?.token?.trim() ?? "";
  } catch {}

  const { url } = getGatewayConfig(configFile);
  try {
    const { Agent: HttpsAgent } = await import("node:https");
    const agent = new HttpsAgent({ rejectUnauthorized: false });
    const resp = await fetch(`${url}/hooks/wake`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(hooksToken ? { Authorization: `Bearer ${hooksToken}` } : {}),
      },
      body: JSON.stringify({ text, mode: "now" }),
      agent,
    } as any);
    if (!resp.ok) {
      logger.warn?.(`pipeline-hooks: announceToMain failed: ${resp.status}`);
    }
  } catch (err: any) {
    logger.warn?.(`pipeline-hooks: announceToMain error: ${err.message}`);
  }
}

// ── Skill harvesting ────────────────────────────────────────────────────────

interface SkillHarvestAction {
  action: "UPDATE" | "CREATE" | "NONE";
  skill?: string;
  changes?: string;
  reason?: string;
  description?: string;
  contentOutline?: string;
  checked?: string;
}

function parseSkillUpdates(text: string): SkillHarvestAction[] {
  const actions: SkillHarvestAction[] = [];

  // Find the ## Skill Updates section
  const skillSection = text.match(/## Skill Updates\s*\n([\s\S]*?)(?=\n## (?!#)|$)/);
  if (!skillSection) return actions;

  const content = skillSection[1];

  // Split by ### ACTION: lines
  const actionBlocks = content.split(/(?=### ACTION:\s*)/);

  for (const block of actionBlocks) {
    const actionMatch = block.match(/^### ACTION:\s*(UPDATE|CREATE|NONE)\s*\n([\s\S]*)/);
    if (!actionMatch) continue;

    const actionType = actionMatch[1] as "UPDATE" | "CREATE" | "NONE";
    const body = actionMatch[2].trim();

    if (actionType === "UPDATE") {
      const skill = body.match(/^Skill:\s*(.+)$/m)?.[1]?.trim();
      const changes = body.match(/^Changes:\s*(.+)$/m)?.[1]?.trim();
      const reason = body.match(/^Reason:\s*([\s\S]*?)(?=\n[A-Z]|\n###|$)/m)?.[1]?.trim();
      actions.push({ action: "UPDATE", skill, changes, reason });
    } else if (actionType === "CREATE") {
      const skill = body.match(/^Skill:\s*(.+)$/m)?.[1]?.trim();
      const description = body.match(/^Description:\s*(.+)$/m)?.[1]?.trim();
      const contentOutline = body.match(/^Content outline:\s*\n([\s\S]*?)(?=\n###|$)/m)?.[1]?.trim();
      actions.push({ action: "CREATE", skill, description, contentOutline });
    } else if (actionType === "NONE") {
      const checked = body.match(/^Checked:\s*(.+)$/m)?.[1]?.trim();
      const reason = body.match(/^Reason:\s*([\s\S]*?)(?=\n###|$)/m)?.[1]?.trim();
      actions.push({ action: "NONE", checked, reason });
    }
  }

  return actions;
}

// ── Lifecycle helpers ───────────────────────────────────────────────────────

async function handleTaskComplete(
  pipeline: PipelineState,
  sessionKey: string,
  configFile: string,
  logger: any,
  lastText?: string,
): Promise<void> {
  const { taskId } = pipeline;
  const stagesCompleted = pipeline.stages.join(" → ");
  try {
    completeActiveRun(taskId, "success", lastText?.slice(0, 500) ?? "Task completed via pipeline hooks");
    completeTask(taskId);
    logger.info?.(`pipeline-hooks: task #${taskId} completed (session: ${sessionKey})`);
    const summary = lastText?.slice(0, 1500) ?? "No output captured.";

    // Parse skill harvest from review output
    let skillHarvestSummary = "";
    if (lastText) {
      const harvestActions = parseSkillUpdates(lastText);
      if (harvestActions.length === 0) {
        logger.warn?.(`pipeline-hooks: task #${taskId} — no ## Skill Updates section found in review output`);
        skillHarvestSummary = "\n\n📚 **Skill Harvest:** No skill updates section found in review output.";
      } else {
        const parts: string[] = [];
        for (const ha of harvestActions) {
          if (ha.action === "UPDATE") {
            parts.push(`📝 UPDATE "${ha.skill}": ${ha.changes ?? "no details"}`);
            logger.info?.(`pipeline-hooks: skill harvest — UPDATE ${ha.skill}: ${ha.changes}`);
          } else if (ha.action === "CREATE") {
            parts.push(`✨ CREATE "${ha.skill}": ${ha.description ?? "no description"}`);
            logger.info?.(`pipeline-hooks: skill harvest — CREATE ${ha.skill}: ${ha.description}`);
          } else if (ha.action === "NONE") {
            parts.push(`✅ No updates needed (checked: ${ha.checked ?? "unspecified"})`);
          }
        }
        skillHarvestSummary = `\n\n📚 **Skill Harvest:**\n${parts.join("\n")}`;
      }
    }

    await pushNotification(configFile, `Task #${taskId} completed ✓ (${stagesCompleted})`, "success", logger);
    await announceToMain(
      configFile,
      `Pipeline task #${taskId} completed ✓ (${stagesCompleted})\n\n${summary}${skillHarvestSummary}`,
      logger,
    );
  } catch (err: any) {
    logger.error?.(`pipeline-hooks: handleTaskComplete error: ${err.message}`);
  } finally {
    const pipelineKey = findPipelineKey(sessionKey);
    if (pipelineKey) activePipelines.delete(pipelineKey);
  }
}

async function handleBlocked(
  pipeline: PipelineState,
  sessionKey: string,
  lastText: string,
  configFile: string,
  logger: any,
): Promise<void> {
  const { taskId } = pipeline;
  const reason = lastText.match(/SIGNAL:BLOCKED[:\s]*(.*)/i)?.[1]?.trim() ?? "Unknown reason";
  try {
    completeActiveRun(taskId, "failed", `Blocked: ${reason}`);
    await writeTask({ id: taskId, status: "up_next" });
    logger.info?.(`pipeline-hooks: task #${taskId} blocked: ${reason}`);
    await announceToMain(
      configFile,
      `Pipeline task #${taskId} blocked: ${reason}`,
      logger,
    );
  } catch (err: any) {
    logger.error?.(`pipeline-hooks: handleBlocked error: ${err.message}`);
  } finally {
    const pipelineKey = findPipelineKey(sessionKey);
    if (pipelineKey) activePipelines.delete(pipelineKey);
  }
}

async function handleTaskFailed(
  pipeline: PipelineState,
  sessionKey: string,
  reason: string,
  configFile: string,
  logger: any,
): Promise<void> {
  const { taskId } = pipeline;
  try {
    completeActiveRun(taskId, "failed", reason);
    await writeTask({ id: taskId, status: "up_next" });
    logger.warn?.(`pipeline-hooks: task #${taskId} failed: ${reason}`);
    await announceToMain(
      configFile,
      `Pipeline task #${taskId} failed: ${reason}`,
      logger,
    );
  } catch (err: any) {
    logger.error?.(`pipeline-hooks: handleTaskFailed error: ${err.message}`);
  } finally {
    const pipelineKey = findPipelineKey(sessionKey);
    if (pipelineKey) activePipelines.delete(pipelineKey);
  }
}

// ── Agent subliminal loading ────────────────────────────────────────────────

const AGENTS_DIR = join(os.homedir(), "obsidian", "agents");
let _agentSubliminalCache: { content: Map<string, string>; ts: number } | null = null;
const AGENT_SUBLIMINAL_CACHE_TTL = 60_000;

function loadAgentSubliminal(agentId: string): string | null {
  const now = Date.now();
  if (_agentSubliminalCache && (now - _agentSubliminalCache.ts < AGENT_SUBLIMINAL_CACHE_TTL)) {
    return _agentSubliminalCache.content.get(agentId) ?? null;
  }
  const content = new Map<string, string>();
  const filePath = join(AGENTS_DIR, agentId, "subliminal.md");
  try {
    const raw = readFileSync(filePath, "utf-8");
    const fmMatch = raw.match(/^---\n[\s\S]*?\n---\n([\s\S]*)$/);
    content.set(agentId, fmMatch ? fmMatch[1].trim() : raw.trim());
  } catch { /* file doesn't exist — that's fine */ }
  _agentSubliminalCache = { content, ts: now };
  return content.get(agentId) ?? null;
}

// ── Main registration ───────────────────────────────────────────────────────

export function registerPipelineHooks(ctx: PluginContext): void {
  const { api, configFile } = ctx;
  const logger = api.logger;

  // ── context hook ──────────────────────────────────────────────────────────
  // Fires before every LLM call. Returns modified messages array (ephemeral).
  api.on("context", async (event: any, hookCtx: any) => {
    const sessionKey: string | undefined = hookCtx?.sessionKey;
    if (!sessionKey) return undefined;

    const pipeline = findPipeline(sessionKey);
    if (!pipeline) return undefined;

    const stage = pipeline.stages[pipeline.currentStageIndex];
    if (!stage) return undefined;

    const def = pipeline.stageDefinitions.get(stage);
    if (!def) return undefined;

    // Prepend stage subliminal as a user message — LLM sees it, not persisted.
    const stageMessage: any = {
      role: "user",
      content: `[Stage: ${stage.toUpperCase()}]\n\n${def.content}`,
      timestamp: Date.now(),
    };

    // Build messages array in order: stage → skills → agent subliminal → history
    const messages: any[] = [stageMessage];

    // Skills injection (index 1)
    if (pipeline.skills.length > 0) {
      const skillContent = pipeline.skills.join("\n\n---\n\n");
      messages.push({
        role: "user",
        content: `## Injected Skills\n\nThe following skills contain domain-specific guidance for this task. Reference and follow them.\n\n${skillContent}`,
        timestamp: Date.now(),
      });
    }

    // Agent subliminal
    const agentSubliminal = loadAgentSubliminal(hookCtx.agentId ?? "lloyd");
    if (agentSubliminal) {
      messages.push({
        role: "user",
        content: agentSubliminal,
        timestamp: Date.now(),
      });
    }

    // Then append conversation history
    messages.push(...event.messages);

    return { messages };
  });

  // ── llm_output hook ───────────────────────────────────────────────────────
  // Passive debug/logging hook. All pipeline management is handled in agent_end.
  api.on("llm_output", async (event: any, hookCtx: any) => {
    const sessionKey: string | undefined = hookCtx?.sessionKey;
    if (!sessionKey) return;

    const pipeline = findPipeline(sessionKey);
    if (!pipeline) return;

    // Debug logging only — no stage management, no side effects
    const lastText: string = (event.assistantTexts as string[] | undefined)?.join("\n") ?? "";
    const signalLines = lastText.split("\n").map((l: string) => l.trim()).filter((l: string) =>
      l.startsWith("SIGNAL:")
    );
    if (signalLines.length > 0) {
      logger.info?.(
        `pipeline-hooks: [debug] llm_output signals detected for task #${pipeline.taskId}: ${signalLines.join(", ")}`,
      );
    }
  });

  // ── agent_end hook ────────────────────────────────────────────────────────
  // Fires when session ends. Handles ALL pipeline lifecycle:
  // - TASK_COMPLETE → finalize task
  // - BLOCKED → mark blocked, clean up
  // - STAGE_COMPLETE → advance stage, return continuation prompt
  // - Error/failure → mark failed
  // - No signal → nudge continuation
  // Returns { continue: true, prompt } to re-prompt via wiggam hook.
  api.on("agent_end", async (event: any, hookCtx: any) => {
    const sessionKey: string | undefined = hookCtx?.sessionKey;
    if (!sessionKey) return;

    // ── Shared signal checks ────────────────────────────────────────────────
    const signals: string[] = (event.signals as string[] | undefined) ?? [];
    const lastText: string = (event.lastAssistantText as string | undefined) ?? "";

    const pipeline = findPipeline(sessionKey);
    if (pipeline) {
      // ── Non-pipeline registered tasks (empty stages) ──────────────────────
      if (pipeline.stages.length === 0) {
        const success = event.success !== false;
        const hasTaskComplete = signals.includes("TASK_COMPLETE");

        if (hasTaskComplete) {
          try {
            completeActiveRun(pipeline.taskId, "success", lastText.slice(0, 500));
            completeTask(pipeline.taskId);
            logger.info?.(`pipeline-hooks: non-pipeline task #${pipeline.taskId} completed`);
          } catch (err: any) {
            logger.error?.(`pipeline-hooks: error completing non-pipeline task: ${err.message}`);
          }
          const pk = findPipelineKey(sessionKey);
          if (pk) activePipelines.delete(pk);
          return;
        }

        if (!success) {
          try {
            completeActiveRun(pipeline.taskId, "failed", "Session failed");
            writeTask({ id: pipeline.taskId, status: "up_next" });
            logger.warn?.(`pipeline-hooks: non-pipeline task #${pipeline.taskId} failed`);
          } catch (err: any) {
            logger.error?.(`pipeline-hooks: error failing non-pipeline task: ${err.message}`);
          }
          const pk = findPipelineKey(sessionKey);
          if (pk) activePipelines.delete(pk);
          return;
        }

        // Success but no TASK_COMPLETE — finalize as success.
        // Non-pipeline tasks don't need wiggam continuation; they run once and complete.
        // Only pipeline tasks (with stages) get continuation prompts.
        try {
          completeActiveRun(pipeline.taskId, "success", lastText.slice(0, 500));
          completeTask(pipeline.taskId);
        } catch (err: any) {
          logger.error?.(`pipeline-hooks: error completing non-pipeline task: ${err.message}`);
        }
        const pk = findPipelineKey(sessionKey);
        if (pk) activePipelines.delete(pk);
        return;
      }

      // ── Pipeline tasks (with stages) ──────────────────────────────────────
      const hasTaskComplete = signals.includes("TASK_COMPLETE") ||
        signals.some((s: string) => s.startsWith("TASK_COMPLETE"));
      const hasBlocked = signals.some((s: string) => s.startsWith("BLOCKED"));
      const hasStageComplete = signals.includes("STAGE_COMPLETE");

      // ── TASK_COMPLETE → finalize ──────────────────────────────────────────
      if (hasTaskComplete) {
        await handleTaskComplete(pipeline, sessionKey, configFile, logger, lastText);
        return;
      }

      // ── BLOCKED → mark blocked, clean up ──────────────────────────────────
      if (hasBlocked) {
        await handleBlocked(pipeline, sessionKey, lastText, configFile, logger);
        return;
      }

      // ── STAGE_COMPLETE → advance to next stage ────────────────────────────
      if (hasStageComplete) {
        pipeline.currentStageIndex++;

        if (pipeline.currentStageIndex >= pipeline.stages.length) {
          // All stages exhausted — treat as task complete
          logger.info?.(`pipeline-hooks: all stages complete for task #${pipeline.taskId}, finalizing`);
          await handleTaskComplete(pipeline, sessionKey, configFile, logger, lastText);
          return;
        }

        const nextStage = pipeline.stages[pipeline.currentStageIndex];
        const nextDef = pipeline.stageDefinitions.get(nextStage);
        if (!nextDef) {
          logger.warn?.(`pipeline-hooks: no definition for stage '${nextStage}', finalizing task`);
          await handleTaskComplete(pipeline, sessionKey, configFile, logger, lastText);
          return;
        }

        const prevStage = pipeline.stages[pipeline.currentStageIndex - 1] ?? "unknown";
        logger.info?.(
          `pipeline-hooks: advancing task #${pipeline.taskId} to stage '${nextStage}' via continuation`,
        );

        // Toast for stage transition (not a chat message)
        await pushNotification(
          configFile,
          `Task #${pipeline.taskId}: "${prevStage}" ✓ → "${nextStage}"`,
          "info",
          logger,
        );

        return {
          continue: true,
          prompt:
            `## Stage Transition: ${nextStage.toUpperCase()}\n\n` +
            `The previous stage is complete. You are now in the **${nextStage}** stage.\n\n` +
            nextDef.content,
        };
      }

      // ── Error/failure → mark failed ───────────────────────────────────────
      if (event.success === false || event.error) {
        const reason = event.error
          ? `Session error: ${event.error}`
          : "Session ended with failure";
        await handleTaskFailed(pipeline, sessionKey, reason, configFile, logger);
        return;
      }

      // ── No relevant signal — nudge continuation ───────────────────────────
      const currentStage = pipeline.stages[pipeline.currentStageIndex] ?? "unknown";
      logger.info?.(
        `pipeline-hooks: task #${pipeline.taskId} stopped without signal in stage '${currentStage}', nudging`,
      );
      return {
        continue: true,
        prompt: `You stopped without completing the task. You are in stage "${currentStage}". Continue your work — assess what's done, what's next, and take the next action. Do not repeat what you've already done.`,
      };
    }

    // ── Non-pipeline idler tasks (fallback) ─────────────────────────────────
    const idlerMatch = sessionKey.match(/hook:idler:task-(\d+)/);
    if (!idlerMatch) return;

    const taskId = parseInt(idlerMatch[1], 10);
    if (isNaN(taskId)) return;

    const success = event.success !== false;
    const hasComplete = signals.includes("TASK_COMPLETE");
    const hasBlocked = signals.some((s: string) => s.startsWith("BLOCKED"));

    try {
      if (hasComplete || (success && !hasBlocked)) {
        completeActiveRun(taskId, "success", lastText.slice(0, 500));
        completeTask(taskId);
        logger.info?.(`pipeline-hooks: idler task #${taskId} completed (non-pipeline, session: ${sessionKey})`);
      } else if (hasBlocked) {
        const reason = lastText.match(/BLOCKED[:\s]*(.*)/)?.[1]?.trim() ?? "Unknown";
        completeActiveRun(taskId, "blocked", reason);
        writeTask({ id: taskId, status: "inbox" });
        logger.warn?.(`pipeline-hooks: idler task #${taskId} blocked: ${reason}`);
      } else {
        completeActiveRun(taskId, "failed", "Session ended without completion signal");
        writeTask({ id: taskId, status: "up_next" });
        logger.warn?.(`pipeline-hooks: idler task #${taskId} failed (no signal, session: ${sessionKey})`);
      }
    } catch (err: any) {
      logger.error?.(`pipeline-hooks: error handling idler task #${taskId}: ${err.message}`);
    }
  });

  // ── POST /api/mc/pipeline-init ───────────────────────────────────────────
  api.registerHttpRoute({
    path: "/api/mc/pipeline-init",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") {
        jsonResponse(res, { error: "POST only" }, 405);
        return;
      }
      try {
        const body = JSON.parse(await readBody(req));
        const { sessionKey, taskId, stages, skillPaths } = body;

        if (!sessionKey || typeof sessionKey !== "string") {
          jsonResponse(res, { error: "Missing sessionKey" }, 400);
          return;
        }
        if (typeof taskId !== "number" && typeof taskId !== "string") {
          jsonResponse(res, { error: "Missing taskId" }, 400);
          return;
        }
        if (!Array.isArray(stages)) {
          jsonResponse(res, { error: "Missing stages (must be array)" }, 400);
          return;
        }

        const allStages = loadStages();
        const defMap = new Map<string, StageDefinition>(
          allStages.map((s) => [s.name, s]),
        );

        const numericTaskId = Number(taskId);

        // Load skills at init time
        const skills: string[] = [];
        if (Array.isArray(skillPaths)) {
          for (const sp of skillPaths) {
            const content = loadSkillContent(sp);
            if (content) skills.push(content);
          }
        }

        activePipelines.set(sessionKey, {
          taskId: numericTaskId,
          stages,
          currentStageIndex: 0,
          stageDefinitions: defMap,
          skills,
        });

        if (stages.length > 0) {
          logger.info?.(
            `pipeline-hooks: initialized pipeline for task #${numericTaskId} ` +
            `(${stages.join(" → ")}, ${skills.length} skills) session=${sessionKey}`,
          );
        } else {
          logger.info?.(
            `pipeline-hooks: registered non-pipeline task #${numericTaskId} for completion tracking session=${sessionKey}`,
          );
        }

        jsonResponse(res, { ok: true, taskId: numericTaskId, stages, sessionKey });
      } catch (err: any) {
        logger.error?.(`pipeline-hooks: pipeline-init error: ${err.message}`);
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── GET /api/mc/pipeline-status ──────────────────────────────────────────
  api.registerHttpRoute({
    path: "/api/mc/pipeline-status",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      const url = new URL(req.url || "/", "http://localhost");
      const sessionKey = url.searchParams.get("sessionKey");

      if (sessionKey) {
        const pipeline = activePipelines.get(sessionKey);
        if (!pipeline) {
          jsonResponse(res, { status: "not_found", sessionKey });
          return;
        }
        jsonResponse(res, {
          sessionKey,
          taskId: pipeline.taskId,
          stages: pipeline.stages,
          currentStageIndex: pipeline.currentStageIndex,
          currentStage: pipeline.stages[pipeline.currentStageIndex],
        });
        return;
      }

      const all: any[] = [];
      for (const [key, p] of activePipelines) {
        all.push({
          sessionKey: key,
          taskId: p.taskId,
          currentStage: p.stages[p.currentStageIndex],
          stages: p.stages,
          currentStageIndex: p.currentStageIndex,
        });
      }
      jsonResponse(res, { activePipelines: all, count: all.length });
    },
  });

  logger.info?.("pipeline-hooks: registered (context, llm_output, agent_end, pipeline-init)");
}
