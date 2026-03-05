/**
 * index.ts — agent-orchestrator: Claude Agent SDK-powered orchestration plugin.
 *
 * Spawns Claude Code instances as specialist agents with pipeline coordination.
 * Uses @anthropic-ai/claude-agent-sdk query() to run instances with full
 * Claude Code capabilities (Read, Write, Edit, Bash, etc.) + MCP tool access.
 *
 * Tools registered:
 *   cc_orchestrate — spawn orchestrator with pipeline + all worker agents
 *   cc_spawn       — spawn single Claude Code instance (no orchestrator)
 *   cc_status      — check status of running/completed instances
 *   cc_result      — get full result text from completed instance
 *   cc_abort       — abort a running instance
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { IncomingMessage, ServerResponse } from "node:http";
import { randomUUID, createPrivateKey, createPublicKey, sign as cryptoSign } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import type { CcInstance, InstanceStatusResponse, McInstanceInfo } from "./types.js";
import { consumeQuery } from "./query-consumer.js";
import { buildOrchestratorPrompt, buildDirectPrompt } from "./orchestrator-prompt.js";
import type { PipelineType } from "./orchestrator-prompt.js";
import {
  coderAgent,
  researcherAgent,
  reviewerAgent,
  testerAgent,
  plannerAgent,
  auditorAgent,
  operatorAgent,
  memoryAgent,
  clawhubAgent,
} from "./agents/index.js";

// ── Constants ──────────────────────────────────────────────────────────────

const MCP_SSE_URL = "http://127.0.0.1:8093/sse";
const DEFAULT_BUDGET_USD = 5.0;
const DEFAULT_MAX_TURNS = 30;
const CLAUDE_CODE_PATH = "/home/alansrobotlab/.npm-global/bin/claude";
const VAULT_AGENTS_DIR = join(process.env.HOME || "/home/alansrobotlab", "obsidian/agents");
const MODE_STATE_PATH = join(__dirname, "../mcp-tools/mode-state.json");

// ── Gateway Notification ─────────────────────────────────────────────────

const GATEWAY_WS_URL = "ws://127.0.0.1:18789";
const GATEWAY_SESSION_KEY = "agent:main:main";
const IDENTITY_PATH = join(process.env.HOME || "/home/alansrobotlab", ".openclaw/identity/device.json");
const DEVICE_AUTH_PATH = join(process.env.HOME || "/home/alansrobotlab", ".openclaw/identity/device-auth.json");
const PROTOCOL_VERSION = 3;
const ED25519_SPKI_PREFIX = Buffer.from("302a300506032b6570032100", "hex");

function base64UrlEncode(buf: Buffer): string {
  return buf.toString("base64").replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/g, "");
}

function derivePublicKeyRaw(publicKeyPem: string): Buffer {
  const spki = createPublicKey(publicKeyPem).export({ type: "spki", format: "der" });
  if (spki.length === ED25519_SPKI_PREFIX.length + 32 &&
      spki.subarray(0, ED25519_SPKI_PREFIX.length).equals(ED25519_SPKI_PREFIX)) {
    return spki.subarray(ED25519_SPKI_PREFIX.length);
  }
  return spki;
}

function publicKeyRawBase64Url(publicKeyPem: string): string {
  return base64UrlEncode(derivePublicKeyRaw(publicKeyPem));
}

function signPayload(privateKeyPem: string, payload: string): string {
  const key = createPrivateKey(privateKeyPem);
  return base64UrlEncode(cryptoSign(null, Buffer.from(payload, "utf8"), key));
}

interface DeviceIdentity {
  deviceId: string;
  publicKeyPem: string;
  privateKeyPem: string;
}

function loadDeviceIdentity(): DeviceIdentity | null {
  try {
    const raw = JSON.parse(readFileSync(IDENTITY_PATH, "utf-8"));
    if (raw?.version === 1 && raw.deviceId && raw.publicKeyPem && raw.privateKeyPem) {
      return { deviceId: raw.deviceId, publicKeyPem: raw.publicKeyPem, privateKeyPem: raw.privateKeyPem };
    }
  } catch {}
  return null;
}

function loadDeviceToken(): string | null {
  try {
    const raw = JSON.parse(readFileSync(DEVICE_AUTH_PATH, "utf-8"));
    return raw?.tokens?.operator?.token || null;
  } catch {}
  return null;
}

/**
 * Send a chat.send message to Lloyd's session via the OpenClaw gateway WebSocket.
 * Connects, authenticates with device identity, sends a user-side message that
 * triggers the agent pipeline, then disconnects.
 * Fully non-blocking and error-swallowing — never throws.
 */
async function injectGatewayMessage(message: string, logger: any): Promise<void> {
  const identity = loadDeviceIdentity();
  if (!identity) {
    logger.warn?.("agent-orchestrator: cannot inject — device identity not found");
    return;
  }
  const deviceToken = loadDeviceToken();

  // Dynamic import ws from openclaw's node_modules
  let WebSocket: any;
  const WS_PATHS = [
    "ws",
    "/home/alansrobotlab/.npm-global/lib/node_modules/openclaw/node_modules/ws/index.js",
  ];
  for (const wsPath of WS_PATHS) {
    try {
      const mod = await import(wsPath);
      WebSocket = mod.WebSocket || mod.default?.WebSocket || mod.default;
      if (WebSocket) break;
    } catch {}
  }
  if (!WebSocket) {
    logger.warn?.("agent-orchestrator: cannot inject — ws module not found");
    return;
  }

  return new Promise<void>((resolve) => {
    const timeout = setTimeout(() => {
      try { ws?.close(); } catch {}
      resolve();
    }, 8000);

    let ws: any;
    try {
      ws = new WebSocket(GATEWAY_WS_URL);
    } catch {
      clearTimeout(timeout);
      resolve();
      return;
    }

    let reqIdCounter = 0;
    const pending = new Map<string, (ok: boolean, payload: any) => void>();

    function sendReq(method: string, params: any): Promise<any> {
      return new Promise((res, rej) => {
        const id = `orch-notify-${++reqIdCounter}`;
        pending.set(id, (ok, payload) => {
          if (ok) res(payload);
          else rej(new Error(payload?.message || "gateway request failed"));
        });
        ws.send(JSON.stringify({ type: "req", id, method, params }));
      });
    }

    ws.on("open", () => {
      // Wait for connect.challenge event
    });

    ws.on("message", async (raw: any) => {
      try {
        const data = JSON.parse(typeof raw === "string" ? raw : raw.toString());

        // Handle connect.challenge event
        if (data.type === "event" && data.event === "connect.challenge") {
          const nonce = data.payload?.nonce;
          if (!nonce) {
            ws.close();
            clearTimeout(timeout);
            resolve();
            return;
          }

          const role = "operator";
          const scopes = ["operator.admin"];
          const signedAtMs = Date.now();
          const authToken = deviceToken || undefined;

          // Build v3 signed payload
          const payloadStr = [
            "v3",
            identity.deviceId,
            "cli",  // clientId
            "backend",          // clientMode
            role,
            scopes.join(","),
            String(signedAtMs),
            authToken ?? "",
            nonce,
            process.platform,
            "",                 // deviceFamily
          ].join("|");

          const signature = signPayload(identity.privateKeyPem, payloadStr);

          const connectParams = {
            minProtocol: PROTOCOL_VERSION,
            maxProtocol: PROTOCOL_VERSION,
            client: {
              id: "cli",
              displayName: "agent-orchestrator",
              version: "0.1.0",
              platform: process.platform,
              mode: "backend",
            },
            caps: [],
            auth: authToken ? { token: authToken, deviceToken: authToken } : undefined,
            role,
            scopes,
            device: {
              id: identity.deviceId,
              publicKey: publicKeyRawBase64Url(identity.publicKeyPem),
              signature,
              signedAt: signedAtMs,
              nonce,
            },
          };

          try {
            await sendReq("connect", connectParams);

            // Now send the message (triggers agent pipeline)
            await sendReq("chat.send", {
              sessionKey: GATEWAY_SESSION_KEY,
              message: `[System Message] ${message}`,
              idempotencyKey: randomUUID(),
            });

            logger.info?.("agent-orchestrator: notification injected into session");
          } catch (err: any) {
            logger.warn?.(`agent-orchestrator: gateway inject failed: ${err.message}`);
          }

          ws.close();
          clearTimeout(timeout);
          resolve();
          return;
        }

        // Handle response frames
        if (data.type === "res" && data.id && pending.has(data.id)) {
          const cb = pending.get(data.id)!;
          pending.delete(data.id);
          cb(data.ok !== false, data.payload);
        }
      } catch {}
    });

    ws.on("error", () => {
      clearTimeout(timeout);
      resolve();
    });

    ws.on("close", () => {
      clearTimeout(timeout);
      resolve();
    });
  });
}

/**
 * Format a completion notification message for injection into Lloyd's session.
 */
function formatCompletionNotification(instance: CcInstance): string {
  const typeLabel = instance.type === "orchestrate"
    ? `pipeline: ${instance.pipeline || "custom"}`
    : `agent: ${instance.agent || "unknown"}`;
  const costStr = `$${instance.costUsd.toFixed(2)}`;
  const taskStr = instance.task.length > 100
    ? instance.task.slice(0, 97) + "..."
    : instance.task;

  if (instance.status === "error") {
    const errorStr = instance.error || "unknown error";
    return [
      `**[Agent Error]** \`${instance.id}\` (${typeLabel})`,
      `Status: ✗ error | Cost: ${costStr} | Turns: ${instance.turns}`,
      `Task: ${taskStr}`,
      `Error: ${errorStr}`,
    ].join("\n");
  }

  const resultStr = instance.resultText
    ? (instance.resultText.length > 300
        ? instance.resultText.slice(0, 297) + "..."
        : instance.resultText)
    : "(no result)";

  return [
    `**[Agent Complete]** \`${instance.id}\` (${typeLabel})`,
    `Status: ✓ complete | Cost: ${costStr} | Turns: ${instance.turns}`,
    `Task: ${taskStr}`,
    `Result: ${resultStr}`,
  ].join("\n");
}

// ── Work Mode ────────────────────────────────────────────────────────────

type WorkMode = "work" | "personal" | "general";

const MODE_SCOPE: Record<WorkMode, string> = {
  work:     "work,knowledge,agents",
  personal: "personal,projects,knowledge,agents",
  general:  "",
};

/**
 * Read the current work mode from mcp-tools' persisted state.
 * Returns the vault scope string (e.g. "work,knowledge,agents") or "" for general.
 */
function getWorkModeScope(): { mode: WorkMode; scope: string } {
  try {
    const state = JSON.parse(readFileSync(MODE_STATE_PATH, "utf-8"));
    const mode = (state.currentMode || "general") as WorkMode;
    return { mode, scope: MODE_SCOPE[mode] || "" };
  } catch {
    return { mode: "general", scope: "" };
  }
}

// ── Vault Prompt Loading ──────────────────────────────────────────────────

/**
 * Read an agent's prompt from the vault at ~/obsidian/agents/{name}.md.
 * Strips YAML frontmatter (--- ... ---) if present.
 * Returns null if file doesn't exist — caller falls back to embedded prompt.
 */
function loadVaultPrompt(agentName: string): string | null {
  try {
    const filePath = join(VAULT_AGENTS_DIR, `${agentName}.md`);
    let content = readFileSync(filePath, "utf-8");
    // Strip YAML frontmatter
    if (content.startsWith("---")) {
      const endIdx = content.indexOf("---", 3);
      if (endIdx !== -1) {
        content = content.slice(endIdx + 3).trimStart();
      }
    }
    return content;
  } catch {
    return null;
  }
}

/** Static agent configs (model, tools, maxTurns, description, mcpServers) */
const AGENT_CONFIGS: Record<string, any> = {
  coder: coderAgent,
  researcher: researcherAgent,
  reviewer: reviewerAgent,
  tester: testerAgent,
  planner: plannerAgent,
  auditor: auditorAgent,
  memory: memoryAgent,
  operator: operatorAgent,
  clawhub: clawhubAgent,
};

/** Agents that use vault MCP tools and need scope injection */
const VAULT_USING_AGENTS = new Set(["memory", "researcher", "planner", "coder", "operator", "clawhub"]);

/**
 * Build fresh agent definitions by merging static configs with vault prompts.
 * Called at spawn time so edits in Obsidian take effect immediately.
 * If a vault scope is active, appends scope instructions to agents that use vault tools.
 */
function buildAgentDefs(logger: any, vaultScope?: { mode: string; scope: string }): Record<string, any> {
  const agents: Record<string, any> = {};
  const scopeSuffix = vaultScope?.scope
    ? `\n\n## Vault Scope\nThe user is in **${vaultScope.mode} mode**. When calling vault tools (mem_search, tag_search, tag_explore), always include the parameter \`scope: "${vaultScope.scope}"\`.`
    : "";

  for (const [name, config] of Object.entries(AGENT_CONFIGS)) {
    const vaultPrompt = loadVaultPrompt(name);
    const basePrompt = vaultPrompt || config.prompt;
    const needsScope = scopeSuffix && VAULT_USING_AGENTS.has(name);
    agents[name] = {
      ...config,
      prompt: needsScope ? basePrompt + scopeSuffix : basePrompt,
    };
    if (vaultPrompt) {
      logger.info?.(`agent-orchestrator: loaded ${name} prompt from vault`);
    }
  }
  return agents;
}

// ── Instance Store ─────────────────────────────────────────────────────────

const instances = new Map<string, CcInstance>();

/** Clean up old completed instances (keep last 50) */
function pruneInstances(): void {
  const completed = Array.from(instances.entries())
    .filter(([, inst]) => inst.status !== "running")
    .sort((a, b) => (b[1].endedAt || 0) - (a[1].endedAt || 0));

  if (completed.length > 50) {
    for (const [id] of completed.slice(50)) {
      instances.delete(id);
    }
  }
}

// ── Plugin Registration ────────────────────────────────────────────────────

export default function register(api: OpenClawPluginApi) {
  // Lazy-load the SDK to avoid startup crash if not installed yet.
  // Try multiple import paths since ESM resolution varies by runtime context.
  let sdkQuery: any = null;

  const SDK_PATHS = [
    "@anthropic-ai/claude-agent-sdk",
    "/home/alansrobotlab/.npm-global/lib/node_modules/@anthropic-ai/claude-agent-sdk/sdk.mjs",
    "/home/alansrobotlab/.openclaw/extensions/agent-orchestrator/node_modules/@anthropic-ai/claude-agent-sdk/sdk.mjs",
  ];

  async function getQuery() {
    if (!sdkQuery) {
      let lastErr: Error | null = null;
      for (const path of SDK_PATHS) {
        try {
          const sdk = await import(path);
          sdkQuery = sdk.query;
          api.logger.info?.(`agent-orchestrator: loaded SDK from ${path}`);
          break;
        } catch (err: any) {
          lastErr = err;
        }
      }
      if (!sdkQuery) {
        throw new Error(
          `Agent SDK not found. Run: npm install -g @anthropic-ai/claude-agent-sdk\n${lastErr?.message}`
        );
      }
    }
    return sdkQuery;
  }

  /** Notification callback — injects a formatted completion message into Lloyd's session */
  function notifyCompletion(instanceId: string, summary: string): void {
    api.logger.info?.(`agent-orchestrator: ${summary}`);

    const instance = instances.get(instanceId);
    if (!instance) return;

    const message = formatCompletionNotification(instance);
    // Fire-and-forget — non-blocking, never delays pipeline completion
    injectGatewayMessage(message, api.logger).catch(() => {});
  }

  // ── Tool: cc_orchestrate ───────────────────────────────────────────────

  api.registerTool({
    name: "cc_orchestrate",
    label: "Claude Code Orchestrate",
    description:
      "Hand off a project or task to an autonomous coordinator agent. " +
      "The coordinator analyzes the codebase, plans which specialist agents to dispatch " +
      "(coder, reviewer, tester, auditor, researcher, operator), executes them " +
      "sequentially or in parallel as needed, and delivers a structured report. " +
      "Returns an instance ID immediately — use cc_status to check progress, cc_result for output.",
    parameters: {
      type: "object",
      properties: {
        task: {
          type: "string",
          description:
            "The project or task to accomplish. Be specific: what needs to be done, which codebase/directory, " +
            "any constraints or requirements. The orchestrator will analyze scope and dispatch agents autonomously.",
        },
        context: {
          type: "string",
          description:
            "Additional context for the orchestrator: relevant file paths, prior decisions, " +
            "vault notes, user preferences, or constraints. Helps the orchestrator plan intelligently.",
        },
        pipeline: {
          type: "string",
          enum: ["code", "research", "security", "full", "custom"],
          description:
            "Suggested approach (the orchestrator may adapt). 'code' = implementation-focused. " +
            "'research' = information gathering. 'security' = audit-focused. 'full' = comprehensive. " +
            "'custom' = orchestrator decides everything (default).",
        },
        cwd: {
          type: "string",
          description: "Working directory for the project. Defaults to home directory.",
        },
        budget: {
          type: "number",
          description: `Max budget in USD (default ${DEFAULT_BUDGET_USD}).`,
        },
        maxTurns: {
          type: "integer",
          description: `Max orchestrator turns (default ${DEFAULT_MAX_TURNS}).`,
        },
        planOnly: {
          type: "boolean",
          description:
            "If true, the orchestrator analyzes and plans but does NOT execute. " +
            "Returns a detailed execution plan for user review. Call cc_result to retrieve the plan, " +
            "then call cc_orchestrate again with the approved plan in 'context' to execute.",
        },
      },
      required: ["task"],
    },
    async execute(_id: string, params: any) {
      try {
        const queryFn = await getQuery();
        const instanceId = randomUUID().slice(0, 8);
        const budget = params.budget ?? DEFAULT_BUDGET_USD;
        const maxTurns = params.maxTurns ?? DEFAULT_MAX_TURNS;
        const pipeline: PipelineType = params.pipeline ?? "custom";
        const planOnly: boolean = params.planOnly ?? false;
        const cwd = params.cwd || process.env.HOME || "/home/alansrobotlab";

        const abort = new AbortController();

        // Read current work mode for vault scope injection
        const workMode = getWorkModeScope();

        // Build agents fresh from vault prompts at spawn time (with scope)
        const agents = buildAgentDefs(api.logger, workMode.scope ? workMode : undefined);

        // Load orchestrator prompt from vault (falls back to built-in)
        const orchestratorPrompt = buildOrchestratorPrompt(
          params.task, pipeline, params.context, loadVaultPrompt("orchestrator"),
          workMode.scope ? workMode : null, planOnly,
        );

        const q = queryFn({
          prompt: orchestratorPrompt,
          options: {
            systemPrompt: {
              type: "preset" as const,
              preset: "claude_code" as const,
              append: planOnly
                ? "\nYou are a project coordinator in PLAN-ONLY mode. Analyze the task using Read/Glob/Grep and output a detailed execution plan. Do NOT dispatch agents or execute any work."
                : "\nYou are a project coordinator. Analyze the task using Read/Glob/Grep, then delegate ALL implementation work to specialist agents via the Task tool. You may read files for analysis but NEVER write, edit, or run commands yourself.",
            },
            cwd,
            model: "sonnet",
            maxBudgetUsd: planOnly ? 1.0 : budget,
            maxTurns: planOnly ? 15 : maxTurns,
            permissionMode: "bypassPermissions",
            allowDangerouslySkipPermissions: true,
            allowedTools: [
              // Analysis tools — orchestrator reads code to plan intelligently
              "Read", "Glob", "Grep",
              // Delegation tool — only available in execute mode
              ...(planOnly ? [] : ["Task"]),
              // Vault tools — context lookup without spawning an agent
              "mcp__openclaw-tools__mem_search",
              "mcp__openclaw-tools__mem_get",
              "mcp__openclaw-tools__tag_search",
              "mcp__openclaw-tools__tag_explore",
            ],
            mcpServers: {
              "openclaw-tools": {
                type: "sse" as const,
                url: MCP_SSE_URL,
              },
            },
            agents,
            abortController: abort,
            pathToClaudeCodeExecutable: CLAUDE_CODE_PATH,
          },
        });

        const instance: CcInstance = {
          id: instanceId,
          type: "orchestrate",
          status: "running",
          task: params.task,
          pipeline,
          planOnly,
          startedAt: Date.now(),
          costUsd: 0,
          turns: 0,
          budgetUsd: budget,
          maxTurns,
          recentMessages: [],
          _abort: abort,
          _query: q,
        };

        instances.set(instanceId, instance);
        pruneInstances();

        // Consume in background — don't await
        consumeQuery(instance, q, api.logger, notifyCompletion).catch((err) => {
          api.logger.error?.(`agent-orchestrator: consumeQuery error for ${instanceId}:`, err);
        });

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              instanceId,
              status: "started",
              type: "orchestrate",
              pipeline,
              planOnly,
              budget: planOnly ? 1.0 : budget,
              maxTurns: planOnly ? 15 : maxTurns,
              message: planOnly
                ? `Plan-only mode started. Use cc_result("${instanceId}") to retrieve the execution plan once complete.`
                : `Pipeline started. Use cc_status("${instanceId}") to check progress.`,
            }, null, 2),
          }],
        };
      } catch (err: any) {
        return {
          content: [{
            type: "text" as const,
            text: `cc_orchestrate error: ${err.message}`,
          }],
        };
      }
    },
  });

  // ── Tool: cc_spawn ─────────────────────────────────────────────────────

  api.registerTool({
    name: "cc_spawn",
    label: "Claude Code Spawn",
    description:
      "Spawn a single Claude Code instance for a specific task. " +
      "No orchestrator layer — the agent works directly on the task. " +
      "Returns an instance ID immediately. Use cc_status to check progress.",
    parameters: {
      type: "object",
      properties: {
        task: {
          type: "string",
          description: "The task for the agent to accomplish.",
        },
        agent: {
          type: "string",
          enum: Object.keys(AGENT_CONFIGS),
          description: "Which specialist agent to use. Each has different tools and model.",
        },
        cwd: {
          type: "string",
          description: "Working directory for file operations.",
        },
        budget: {
          type: "number",
          description: "Max budget in USD (default 2.0 for single agents).",
        },
      },
      required: ["task", "agent"],
    },
    async execute(_id: string, params: any) {
      try {
        const queryFn = await getQuery();
        const agentConfig = AGENT_CONFIGS[params.agent];
        if (!agentConfig) {
          return {
            content: [{
              type: "text" as const,
              text: `Unknown agent: ${params.agent}. Available: ${Object.keys(AGENT_CONFIGS).join(", ")}`,
            }],
          };
        }

        // Load fresh prompt from vault at spawn time, with work mode scope
        const vaultPrompt = loadVaultPrompt(params.agent);
        const workMode = getWorkModeScope();
        let agentPrompt = vaultPrompt || agentConfig.prompt;
        if (workMode.scope && VAULT_USING_AGENTS.has(params.agent)) {
          agentPrompt += `\n\n## Vault Scope\nThe user is in **${workMode.mode} mode**. When calling vault tools (mem_search, tag_search, tag_explore), always include the parameter \`scope: "${workMode.scope}"\`.`;
        }
        const agentDef = { ...agentConfig, prompt: agentPrompt };
        if (vaultPrompt) {
          api.logger.info?.(`agent-orchestrator: loaded ${params.agent} prompt from vault`);
        }

        const instanceId = randomUUID().slice(0, 8);
        const budget = params.budget ?? 2.0;
        const cwd = params.cwd || process.env.HOME || "/home/alansrobotlab";
        const abort = new AbortController();

        // For single agent spawn, the agent IS the top-level query (no subagents)
        const q = queryFn({
          prompt: buildDirectPrompt(params.task),
          options: {
            systemPrompt: agentDef.prompt,
            cwd,
            model: agentDef.model || "sonnet",
            maxBudgetUsd: budget,
            maxTurns: agentDef.maxTurns || 25,
            permissionMode: "bypassPermissions",
            allowDangerouslySkipPermissions: true,
            allowedTools: agentDef.tools || [],
            mcpServers: agentDef.mcpServers?.includes("openclaw-tools")
              ? { "openclaw-tools": { type: "sse" as const, url: MCP_SSE_URL } }
              : {},
            abortController: abort,
            pathToClaudeCodeExecutable: CLAUDE_CODE_PATH,
          },
        });

        const instance: CcInstance = {
          id: instanceId,
          type: "spawn",
          status: "running",
          task: params.task,
          agent: params.agent,
          startedAt: Date.now(),
          costUsd: 0,
          turns: 0,
          budgetUsd: budget,
          maxTurns: agentDef.maxTurns || 25,
          recentMessages: [],
          _abort: abort,
          _query: q,
        };

        instances.set(instanceId, instance);
        pruneInstances();

        consumeQuery(instance, q, api.logger, notifyCompletion).catch((err) => {
          api.logger.error?.(`agent-orchestrator: consumeQuery error for ${instanceId}:`, err);
        });

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              instanceId,
              status: "started",
              type: "spawn",
              agent: params.agent,
              model: agentDef.model || "sonnet",
              budget,
              message: `Agent '${params.agent}' started. Use cc_status("${instanceId}") to check progress.`,
            }, null, 2),
          }],
        };
      } catch (err: any) {
        return {
          content: [{
            type: "text" as const,
            text: `cc_spawn error: ${err.message}`,
          }],
        };
      }
    },
  });

  // ── Tool: cc_status ────────────────────────────────────────────────────

  api.registerTool({
    name: "cc_status",
    label: "Claude Code Status",
    description:
      "Check the status of Claude Code instances. " +
      "Pass an instanceId for a specific instance, or omit for all instances.",
    parameters: {
      type: "object",
      properties: {
        instanceId: {
          type: "string",
          description: "Instance ID to check. Omit to list all instances.",
        },
      },
    },
    async execute(_id: string, params: any) {
      if (params.instanceId) {
        const inst = instances.get(params.instanceId);
        if (!inst) {
          return {
            content: [{ type: "text" as const, text: `No instance found with ID: ${params.instanceId}` }],
          };
        }
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify(formatStatus(inst), null, 2),
          }],
        };
      }

      // List all instances
      const all = Array.from(instances.values())
        .sort((a, b) => b.startedAt - a.startedAt)
        .slice(0, 20)
        .map(formatStatus);

      return {
        content: [{
          type: "text" as const,
          text: all.length > 0
            ? JSON.stringify(all, null, 2)
            : "No Claude Code instances running or recently completed.",
        }],
      };
    },
  });

  // ── Tool: cc_result ────────────────────────────────────────────────────

  api.registerTool({
    name: "cc_result",
    label: "Claude Code Result",
    description:
      "Get the full result text from a completed Claude Code instance.",
    parameters: {
      type: "object",
      properties: {
        instanceId: {
          type: "string",
          description: "Instance ID to get results from.",
        },
      },
      required: ["instanceId"],
    },
    async execute(_id: string, params: any) {
      const inst = instances.get(params.instanceId);
      if (!inst) {
        return {
          content: [{ type: "text" as const, text: `No instance found with ID: ${params.instanceId}` }],
        };
      }

      if (inst.status === "running") {
        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              status: "running",
              activity: inst.activity,
              elapsedMs: Date.now() - inst.startedAt,
              message: "Instance still running. Use cc_status for progress.",
            }, null, 2),
          }],
        };
      }

      return {
        content: [{
          type: "text" as const,
          text: inst.resultText || inst.error || "(no result)",
        }],
      };
    },
  });

  // ── Tool: cc_abort ─────────────────────────────────────────────────────

  api.registerTool({
    name: "cc_abort",
    label: "Claude Code Abort",
    description: "Abort a running Claude Code instance.",
    parameters: {
      type: "object",
      properties: {
        instanceId: {
          type: "string",
          description: "Instance ID to abort.",
        },
      },
      required: ["instanceId"],
    },
    async execute(_id: string, params: any) {
      const inst = instances.get(params.instanceId);
      if (!inst) {
        return {
          content: [{ type: "text" as const, text: `No instance found with ID: ${params.instanceId}` }],
        };
      }

      if (inst.status !== "running") {
        return {
          content: [{
            type: "text" as const,
            text: `Instance ${params.instanceId} is already ${inst.status}.`,
          }],
        };
      }

      // Abort via AbortController
      try {
        inst._abort?.abort();
      } catch {}

      // Also try query.close() if available
      try {
        if (inst._query && typeof inst._query.close === "function") {
          inst._query.close();
        }
      } catch {}

      inst.status = "aborted";
      inst.endedAt = Date.now();
      inst.activity = undefined;

      return {
        content: [{
          type: "text" as const,
          text: `Instance ${params.instanceId} aborted. Ran for ${Math.round((inst.endedAt - inst.startedAt) / 1000)}s.`,
        }],
      };
    },
  });

  // ── Mission Control API Endpoints ──────────────────────────────────────

  function jsonResponse(res: ServerResponse, data: any, status = 200): void {
    res.writeHead(status, {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    });
    res.end(JSON.stringify(data));
  }

  function readBody(req: IncomingMessage): Promise<string> {
    return new Promise((resolve, reject) => {
      let body = "";
      req.on("data", (chunk: any) => { body += chunk; });
      req.on("end", () => resolve(body));
      req.on("error", reject);
    });
  }

  // GET /api/mc/cc-agents — list SDK agent definitions for Mission Control
  api.registerHttpRoute({
    path: "/api/mc/cc-agents",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      const agentList = Object.entries(AGENT_CONFIGS).map(([name, config]) => ({
        id: name,
        model: config.model || "sonnet",
        description: config.description || "",
        maxTurns: config.maxTurns || 25,
        tools: (config.tools || []).filter((t: string) => !t.startsWith("mcp__")),
        mcpTools: (config.tools || []).filter((t: string) => t.startsWith("mcp__")),
        hasMcp: config.mcpServers?.length > 0,
        avatarUrl: `/api/mc/agent-avatar?id=${name}`,
      }));

      // Add orchestrator as a virtual agent
      agentList.unshift({
        id: "orchestrator",
        model: "sonnet",
        description: "Autonomous project coordinator. Analyzes tasks, plans agent dispatch, manages execution, and delivers structured reports.",
        maxTurns: DEFAULT_MAX_TURNS,
        tools: ["Read", "Glob", "Grep", "Task"],
        mcpTools: ["mem_search", "mem_get", "tag_search", "tag_explore"],
        hasMcp: true,
        avatarUrl: "/api/mc/agent-avatar?id=orchestrator",
      });

      // Augment each agent with active/recent instance counts
      const instancesByAgent: Record<string, { active: number; recent: number }> = {};
      for (const inst of instances.values()) {
        const agentId = inst.type === "orchestrate" ? "orchestrator" : (inst.agent || "unknown");
        if (!instancesByAgent[agentId]) instancesByAgent[agentId] = { active: 0, recent: 0 };
        if (inst.status === "running") instancesByAgent[agentId].active++;
        else instancesByAgent[agentId].recent++;
      }

      jsonResponse(res, {
        agents: agentList,
        instanceCounts: instancesByAgent,
      });
    },
  });

  // GET /api/mc/cc-instances — list all instances
  api.registerHttpRoute({
    path: "/api/mc/cc-instances",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      const list: McInstanceInfo[] = Array.from(instances.values())
        .sort((a, b) => b.startedAt - a.startedAt)
        .slice(0, 30)
        .map((inst) => ({
          id: inst.id,
          type: inst.type,
          status: inst.status,
          task: inst.task,
          pipeline: inst.pipeline,
          agent: inst.agent,
          startedAt: inst.startedAt,
          endedAt: inst.endedAt,
          elapsedMs: (inst.endedAt || Date.now()) - inst.startedAt,
          costUsd: inst.costUsd,
          turns: inst.turns,
          budgetUsd: inst.budgetUsd,
          activity: inst.activity,
          resultPreview: inst.resultText ? inst.resultText.slice(0, 200) : undefined,
          error: inst.error,
        }));

      jsonResponse(res, { instances: list });
    },
  });

  // GET /api/mc/cc-instance-log?id=X&limit=50 — recent messages for an instance
  api.registerHttpRoute({
    path: "/api/mc/cc-instance-log",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      const url = new URL(req.url || "/", "http://localhost");
      const id = url.searchParams.get("id");
      const limit = parseInt(url.searchParams.get("limit") || "50", 10);

      if (!id) {
        jsonResponse(res, { error: "Missing id parameter" }, 400);
        return;
      }

      const inst = instances.get(id);
      if (!inst) {
        jsonResponse(res, { error: "Instance not found" }, 404);
        return;
      }

      const messages = inst.recentMessages.slice(-limit);
      jsonResponse(res, {
        id: inst.id,
        status: inst.status,
        messages,
      });
    },
  });

  // POST /api/mc/cc-instance-abort — abort an instance
  api.registerHttpRoute({
    path: "/api/mc/cc-instance-abort",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") {
        jsonResponse(res, { error: "Method not allowed" }, 405);
        return;
      }

      try {
        const body = JSON.parse(await readBody(req));
        const id = body.id;
        if (!id) {
          jsonResponse(res, { error: "Missing id" }, 400);
          return;
        }

        const inst = instances.get(id);
        if (!inst) {
          jsonResponse(res, { error: "Instance not found" }, 404);
          return;
        }

        if (inst.status !== "running") {
          jsonResponse(res, { ok: false, message: `Already ${inst.status}` });
          return;
        }

        try { inst._abort?.abort(); } catch {}
        try { inst._query?.close?.(); } catch {}
        inst.status = "aborted";
        inst.endedAt = Date.now();

        jsonResponse(res, { ok: true, message: "Aborted" });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // ── Helpers ────────────────────────────────────────────────────────────

  function formatStatus(inst: CcInstance): InstanceStatusResponse {
    return {
      id: inst.id,
      type: inst.type,
      status: inst.status,
      task: inst.task,
      pipeline: inst.pipeline,
      agent: inst.agent,
      elapsedMs: (inst.endedAt || Date.now()) - inst.startedAt,
      costUsd: Math.round(inst.costUsd * 1000) / 1000,
      turns: inst.turns,
      activity: inst.activity,
      resultPreview: inst.resultText ? inst.resultText.slice(0, 500) : undefined,
      error: inst.error,
    };
  }

  // ── Startup ────────────────────────────────────────────────────────────

  api.logger.info?.("agent-orchestrator: loaded — 5 tools (cc_orchestrate, cc_spawn, cc_status, cc_result, cc_abort)");
}
