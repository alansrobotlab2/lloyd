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
import { randomUUID } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

import type { CcInstance, InstanceStatusResponse, McInstanceInfo, PendingQuestionInfo, QuestionAnswer } from "./types.js";
import { consumeQuery } from "./query-consumer.js";
import { buildOrchestratorPrompt, buildDirectPrompt, buildInteractivePlanningPrompt } from "./orchestrator-prompt.js";
import type { PipelineType } from "./orchestrator-prompt.js";
import { createQuestion, resolveQuestion, listPendingQuestions, cancelAllForInstance } from "./pending-questions.js";
import {
  coderAgent,
  researcherAgent,
  reviewerAgent,
  testerAgent,
  plannerAgent,
  auditorAgent,
  operatorAgent,
  clawhubAgent,
} from "./agents/index.js";

// ── Constants ──────────────────────────────────────────────────────────────

const MCP_SSE_URL = "http://127.0.0.1:8093/sse";
const DEFAULT_BUDGET_USD = 5.0;
const DEFAULT_MAX_TURNS = 30;
const CLAUDE_CODE_PATH = "/home/alansrobotlab/.local/share/claude/versions/2.1.69";
const VAULT_AGENTS_DIR = join(process.env.HOME || "/home/alansrobotlab", "obsidian/agents");
const MODE_STATE_PATH = join(__dirname, "../mcp-tools/mode-state.json");
const CLI_JS_PATH = join(__dirname, "node_modules/@anthropic-ai/claude-agent-sdk/cli.js");

// Custom spawn function to bypass SDK's default native binary resolution,
// which fails with ENOENT inside the gateway process.
function expandTilde(p: string): string {
  return p.startsWith("~/") ? join(homedir(), p.slice(2)) : p;
}

function spawnClaudeCode(options: { command: string; args: string[]; cwd?: string; env: Record<string, string | undefined>; signal?: AbortSignal }): ChildProcess {
  let cmd = options.command;
  let args = options.args;

  // Expand ~ in command and cwd — Node's spawn doesn't do shell expansion
  cmd = expandTilde(cmd);
  const cwd = options.cwd ? expandTilde(options.cwd) : undefined;

  // If native binary not accessible, fall back to bundled cli.js via node
  if (!existsSync(cmd)) {
    console.error(`[agent-orchestrator] Native binary not found at ${cmd}, falling back to cli.js`);
    args = [CLI_JS_PATH, ...args];
    cmd = process.execPath;
  }

  // Clean env vars that prevent nested sessions or interfere with spawn
  const env = { ...options.env };
  delete env.CLAUDECODE;
  delete env.NODE_OPTIONS;

  console.error(`[agent-orchestrator] Spawning: ${cmd} (cwd: ${cwd || "inherit"})`);

  return spawn(cmd, args, {
    cwd,
    stdio: ["pipe", "pipe", "pipe"],
    env: env as NodeJS.ProcessEnv,
    signal: options.signal,
  });
}

// ── Hook-Based Notification ──────────────────────────────────────────────
//
// Push messages into Lloyd's session via the OpenClaw HTTP hooks endpoint.
// Much simpler than the old gateway WebSocket + Ed25519 auth approach.
// Same mechanism the voice pipeline uses for ASR transcript injection.

const HOOKS_URL = "http://127.0.0.1:18789/hooks/wake";
const HOOKS_SESSION_KEY = "agent:main:main";
const OPENCLAW_CONFIG_PATH = join(process.env.HOME || "/home/alansrobotlab", ".openclaw/openclaw.json");

let _hooksToken: string | null = null;

function loadHooksToken(): string | null {
  if (_hooksToken) return _hooksToken;
  try {
    const config = JSON.parse(readFileSync(OPENCLAW_CONFIG_PATH, "utf-8"));
    _hooksToken = config?.hooks?.token || null;
  } catch {}
  return _hooksToken;
}

/**
 * Inject a message into Lloyd's session via the HTTP hooks endpoint.
 * Non-blocking and error-swallowing — never throws.
 */
async function injectHookMessage(message: string, logger: any, sessionKey?: string): Promise<void> {
  const token = loadHooksToken();
  if (!token) {
    logger.warn?.("agent-orchestrator: cannot inject — hooks token not found in openclaw.json");
    return;
  }
  try {
    console.error(`[agent-orchestrator] injectHookMessage sessionKey=${sessionKey ?? "(none)"} url=${HOOKS_URL}`);
    const resp = await fetch(HOOKS_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: `[System Message] ${message}`,
        mode: "now",
      }),
    });
    console.error(`[agent-orchestrator] injectHookMessage response status=${resp.status}`);
  } catch (err: any) {
    logger.warn?.(`agent-orchestrator: hook inject failed: ${err.message}`);
  }
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
  operator: operatorAgent,
  clawhub: clawhubAgent,
};

/** Agents that use vault MCP tools and need scope injection */
const VAULT_USING_AGENTS = new Set(["researcher", "planner", "coder", "operator", "clawhub"]);

/**
 * Build fresh agent definitions by merging static configs with vault prompts.
 * Called at spawn time so edits in Obsidian take effect immediately.
 * If a vault scope is active, appends scope instructions to agents that use vault tools.
 */
/** Agents that should get AskUserQuestion in interactive mode */
const INTERACTIVE_AGENTS = new Set(["coder", "planner", "operator", "researcher"]);

function buildAgentDefs(logger: any, vaultScope?: { mode: string; scope: string }, interactive?: boolean): Record<string, any> {
  const agents: Record<string, any> = {};
  const scopeSuffix = vaultScope?.scope
    ? `\n\n## Vault Scope\nThe user is in **${vaultScope.mode} mode**. When calling vault tools (mem_search, tag_search, tag_explore), always include the parameter \`scope: "${vaultScope.scope}"\`.`
    : "";

  const interactiveSuffix = `\n\n## Interactive Mode\nYou are running in interactive mode. You may use AskUserQuestion to ask clarifying questions — your question will be relayed to the user and the answer returned. If AskUserQuestion is denied, the denial message contains the user's response — read and use it.`;

  for (const [name, config] of Object.entries(AGENT_CONFIGS)) {
    const vaultPrompt = loadVaultPrompt(name);
    const basePrompt = vaultPrompt || config.prompt;
    const needsScope = scopeSuffix && VAULT_USING_AGENTS.has(name);
    const needsInteractive = interactive && INTERACTIVE_AGENTS.has(name);

    let prompt = basePrompt;
    if (needsScope) prompt += scopeSuffix;
    if (needsInteractive) prompt += interactiveSuffix;

    // Add AskUserQuestion to tools in interactive mode
    const tools = [...(config.tools || [])];
    if (needsInteractive && !tools.includes("AskUserQuestion")) {
      tools.push("AskUserQuestion");
    }

    agents[name] = {
      ...config,
      prompt,
      tools,
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

    console.error(`[agent-orchestrator] notifyCompletion instance.sessionKey=${instance.sessionKey ?? "(none)"}`);
    const message = formatCompletionNotification(instance);
    // Fire-and-forget — non-blocking, never delays pipeline completion
    injectHookMessage(message, api.logger, instance.sessionKey).catch(() => {});
  }

  /** Progress callback — injects milestone notifications for interactive instances */
  function notifyProgress(instanceId: string, message: string): void {
    injectHookMessage(message, api.logger, instances.get(instanceId)?.sessionKey).catch(() => {});
  }

  /** Push a pending question notification into Lloyd's session */
  function notifyQuestion(instance: CcInstance, q: { id: string; type: string; agentId?: string; toolName?: string; question: string; options?: string[] }): void {
    const typeLabel = instance.type === "orchestrate"
      ? `pipeline: ${instance.pipeline || "custom"}`
      : `agent: ${instance.agent || "unknown"}`;

    const lines = [
      `**[Agent Question]** \`${instance.id}\` (${typeLabel})`,
      `Question ID: \`${q.id}\` | Type: ${q.type}${q.agentId ? ` | Agent: ${q.agentId}` : ""}`,
    ];

    if (q.type === "permission" && q.toolName) {
      lines.push(`Tool: \`${q.toolName}\``);
    }

    lines.push(`Question: ${q.question}`);

    if (q.options?.length) {
      lines.push(`Options: ${q.options.join(", ")}`);
    }

    lines.push(`→ Use \`cc_respond("${q.id}", action, text)\` to answer. Actions: "allow", "deny", or "answer".`);

    injectHookMessage(lines.join("\n"), api.logger, instance.sessionKey).catch(() => {});
  }

  // ── Interactive Mode: canUseTool Callback ───────────────────────────────

  /** Tools that require approval in interactive mode */
  const GATED_TOOLS = new Set(["Write", "Edit", "Bash", "NotebookEdit"]);

  /** Bash commands that are safe (read-only) and auto-allowed */
  const SAFE_BASH_PATTERNS = [
    /^(cat|ls|head|tail|wc|find|grep|rg|git\s+(status|log|diff|show|branch)|pwd|echo|which|type|file|stat)\b/,
    /^(node|python3?|ruby|go)\s+--version\b/,
    /^npm\s+(list|ls|outdated|view)\b/,
  ];

  /**
   * Build a canUseTool callback for interactive mode.
   * Intercepts destructive tools and AskUserQuestion, creating pending questions
   * that block until the user responds via cc_respond.
   */
  function buildCanUseTool(instance: CcInstance) {
    return async (toolName: string, input: Record<string, unknown>, options: any): Promise<any> => {
      // ── AskUserQuestion interception (Phase 3) ──
      // Agent is asking a clarification question — proxy it to the user.
      // We deny the tool with the user's answer in the denial message.
      if (toolName === "AskUserQuestion") {
        const questions = input.questions as any[];
        const questionText = questions?.map((q: any) => {
          const opts = q.options?.map((o: any) => o.label).join(", ");
          return opts ? `${q.question} (${opts})` : q.question;
        }).join("\n") || "Agent has a question";

        const questionOptions = questions?.[0]?.options?.map((o: any) => o.label);

        const { question: q, promise } = createQuestion({
          instanceId: instance.id,
          type: "clarification",
          agentId: options?.agentID,
          question: questionText,
          options: questionOptions,
          timeoutMs: 10 * 60 * 1000, // 10 min for clarifications
        });

        instance.pendingQuestions.push(q);
        notifyQuestion(instance, q);

        api.logger.info?.(`agent-orchestrator: clarification question ${q.id} from ${options?.agentID || "orchestrator"}`);

        const answer = await promise;

        if (answer.action === "deny") {
          return { behavior: "deny", message: answer.text || "User declined to answer" };
        }

        // Return user's answer as the denial message — agent reads it as tool output
        return { behavior: "deny", message: `[User Response] ${answer.text || "No answer provided"}` };
      }

      // ── Permission gating (Phase 2) ──
      // Auto-allow read-only and safe tools
      if (!GATED_TOOLS.has(toolName)) {
        return { behavior: "allow" };
      }

      // Auto-allow safe Bash patterns
      if (toolName === "Bash" && typeof input.command === "string") {
        const cmd = input.command.trim();
        if (SAFE_BASH_PATTERNS.some((p) => p.test(cmd))) {
          return { behavior: "allow" };
        }
      }

      // Gate destructive tools — create pending question and block
      const agentLabel = options?.agentID || "orchestrator";
      let question: string;
      if (toolName === "Write") {
        question = `Agent "${agentLabel}" wants to create/overwrite file: ${input.file_path}`;
      } else if (toolName === "Edit") {
        question = `Agent "${agentLabel}" wants to edit file: ${input.file_path}`;
      } else if (toolName === "NotebookEdit") {
        question = `Agent "${agentLabel}" wants to edit notebook: ${input.notebook_path}`;
      } else {
        question = `Agent "${agentLabel}" wants to run: ${String(input.command || "").slice(0, 200)}`;
      }

      const { question: q, promise } = createQuestion({
        instanceId: instance.id,
        type: "permission",
        agentId: agentLabel,
        toolName,
        toolInput: input,
        question,
        timeoutMs: 5 * 60 * 1000, // 5 min for permissions
      });

      instance.pendingQuestions.push(q);
      notifyQuestion(instance, q);

      api.logger.info?.(`agent-orchestrator: permission gate ${q.id} — ${toolName} from ${agentLabel}`);

      const answer = await promise;

      if (answer.action === "allow") {
        return { behavior: "allow", updatedInput: answer.updatedInput };
      }

      return {
        behavior: "deny",
        message: answer.text || "User denied this action",
        interrupt: false, // Don't kill the query, just deny this tool call
      };
    };
  }

  // ── Tool: cc_orchestrate ───────────────────────────────────────────────

  api.registerTool((ctx) => ({
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
        interactive: {
          type: "boolean",
          description:
            "Enable interactive mode: subagent file writes and commands require user approval, " +
            "and agents can ask clarifying questions mid-execution. Questions are pushed to your " +
            "session via hooks — answer them with cc_respond. Default false (fire-and-forget).",
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
        const interactive: boolean = params.interactive ?? false;
        const cwd = params.cwd || process.env.HOME || "/home/alansrobotlab";

        const abort = new AbortController();

        // Read current work mode for vault scope injection
        const workMode = getWorkModeScope();

        // Build agents fresh from vault prompts at spawn time (with scope)
        const agents = buildAgentDefs(api.logger, workMode.scope ? workMode : undefined, interactive);

        // Load orchestrator prompt from vault (falls back to built-in)
        const orchestratorPrompt = buildOrchestratorPrompt(
          params.task, pipeline, params.context, loadVaultPrompt("orchestrator"),
          workMode.scope ? workMode : null, planOnly,
        );

        // Build instance early so canUseTool can reference it
        const instance: CcInstance = {
          id: instanceId,
          type: "orchestrate",
          status: "running",
          task: params.task,
          pipeline,
          planOnly,
          interactive,
          startedAt: Date.now(),
          costUsd: 0,
          turns: 0,
          budgetUsd: budget,
          maxTurns,
          pendingQuestions: [],
          recentMessages: [],
          _abort: abort,
          sessionKey: ctx.sessionKey,
        };
        console.error(`[agent-orchestrator] spawn sessionKey: ${ctx.sessionKey ?? "(none)"}`);

        // Read-only tools that are always auto-allowed
        const readOnlyTools = [
          "Read", "Glob", "Grep",
          ...(planOnly ? [] : ["Task"]),
          "mcp__openclaw-tools__mem_search",
          "mcp__openclaw-tools__mem_get",
          "mcp__openclaw-tools__tag_search",
          "mcp__openclaw-tools__tag_explore",
        ];

        // In interactive mode, also allow AskUserQuestion (it gets intercepted by canUseTool)
        const allowedTools = interactive
          ? [...readOnlyTools, "AskUserQuestion"]
          : [
              ...readOnlyTools,
              // Non-interactive: allow all tools directly
              "Write", "Edit", "Bash", "NotebookEdit",
              "mcp__openclaw-tools__backlog_tasks",
              "mcp__openclaw-tools__backlog_get_task",
              "mcp__openclaw-tools__backlog_update_task",
            ];

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
            thinking: { type: "adaptive" as const },
            effort: "medium" as const,
            cwd,
            model: "sonnet",
            maxBudgetUsd: planOnly ? 1.0 : budget,
            maxTurns: planOnly ? 15 : maxTurns,
            permissionMode: interactive ? "default" : "bypassPermissions",
            allowDangerouslySkipPermissions: !interactive,
            canUseTool: interactive ? buildCanUseTool(instance) : undefined,
            allowedTools,
            mcpServers: {
              "openclaw-tools": {
                type: "sse" as const,
                url: MCP_SSE_URL,
              },
            },
            agents,
            abortController: abort,
            pathToClaudeCodeExecutable: CLAUDE_CODE_PATH,
            spawnClaudeCodeProcess: spawnClaudeCode,
          },
        });

        instance._query = q;

        instances.set(instanceId, instance);
        pruneInstances();

        // Consume in background — don't await
        consumeQuery(instance, q, api.logger, notifyCompletion, notifyProgress).catch((err) => {
          api.logger.error?.(`agent-orchestrator: consumeQuery error for ${instanceId}:`, err);
        });

        const interactiveNote = interactive
          ? " Interactive mode ON — questions will be pushed to your session. Answer with cc_respond."
          : "";

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              instanceId,
              status: "started",
              type: "orchestrate",
              pipeline,
              planOnly,
              interactive,
              budget: planOnly ? 1.0 : budget,
              maxTurns: planOnly ? 15 : maxTurns,
              message: planOnly
                ? `Plan-only mode started. Use cc_result("${instanceId}") to retrieve the execution plan once complete.`
                : `Pipeline started.${interactiveNote} Use cc_status("${instanceId}") to check progress.`,
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
  }));

  // ── Tool: cc_spawn ─────────────────────────────────────────────────────

  api.registerTool((ctx) => ({
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
        interactive: {
          type: "boolean",
          description:
            "Enable interactive mode: file writes and commands require user approval, " +
            "and the agent can ask clarifying questions. Default false.",
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

        const interactive: boolean = params.interactive ?? false;

        // Load fresh prompt from vault at spawn time, with work mode scope
        const vaultPrompt = loadVaultPrompt(params.agent);
        const workMode = getWorkModeScope();
        let agentPrompt = vaultPrompt || agentConfig.prompt;
        if (workMode.scope && VAULT_USING_AGENTS.has(params.agent)) {
          agentPrompt += `\n\n## Vault Scope\nThe user is in **${workMode.mode} mode**. When calling vault tools (mem_search, tag_search, tag_explore), always include the parameter \`scope: "${workMode.scope}"\`.`;
        }
        if (interactive) {
          agentPrompt += `\n\n## Interactive Mode\nYou are running in interactive mode. You may use AskUserQuestion to ask clarifying questions — your question will be relayed to the user and the answer returned. If AskUserQuestion is denied, the denial message contains the user's response — read and use it.`;
        }
        const agentDef = { ...agentConfig, prompt: agentPrompt };
        if (vaultPrompt) {
          api.logger.info?.(`agent-orchestrator: loaded ${params.agent} prompt from vault`);
        }

        const instanceId = randomUUID().slice(0, 8);
        const budget = params.budget ?? 2.0;
        const cwd = params.cwd || process.env.HOME || "/home/alansrobotlab";
        const abort = new AbortController();

        const instance: CcInstance = {
          id: instanceId,
          type: "spawn",
          status: "running",
          task: params.task,
          agent: params.agent,
          interactive,
          startedAt: Date.now(),
          costUsd: 0,
          turns: 0,
          budgetUsd: budget,
          maxTurns: agentDef.maxTurns || 25,
          pendingQuestions: [],
          recentMessages: [],
          _abort: abort,
          sessionKey: ctx.sessionKey,
        };
        console.error(`[agent-orchestrator] spawn sessionKey: ${ctx.sessionKey ?? "(none)"}`);

        // Build tool list: in interactive mode, add AskUserQuestion, canUseTool gates destructive tools
        const agentTools = [...(agentDef.tools || [])];
        if (interactive && !agentTools.includes("AskUserQuestion")) {
          agentTools.push("AskUserQuestion");
        }

        // For single agent spawn, the agent IS the top-level query (no subagents)
        const q = queryFn({
          prompt: buildDirectPrompt(params.task),
          options: {
            systemPrompt: agentDef.prompt,
            thinking: agentDef.thinking || { type: "adaptive" as const },
            effort: agentDef.effort,
            cwd,
            model: agentDef.model || "sonnet",
            maxBudgetUsd: budget,
            maxTurns: agentDef.maxTurns || 25,
            permissionMode: interactive ? "default" : "bypassPermissions",
            allowDangerouslySkipPermissions: !interactive,
            canUseTool: interactive ? buildCanUseTool(instance) : undefined,
            allowedTools: agentTools,
            mcpServers: agentDef.mcpServers?.includes("openclaw-tools")
              ? { "openclaw-tools": { type: "sse" as const, url: MCP_SSE_URL } }
              : {},
            abortController: abort,
            pathToClaudeCodeExecutable: CLAUDE_CODE_PATH,
            spawnClaudeCodeProcess: spawnClaudeCode,
          },
        });

        instance._query = q;

        instances.set(instanceId, instance);
        pruneInstances();

        consumeQuery(instance, q, api.logger, notifyCompletion, notifyProgress).catch((err) => {
          api.logger.error?.(`agent-orchestrator: consumeQuery error for ${instanceId}:`, err);
        });

        const interactiveNote = interactive
          ? " Interactive mode ON — questions will be pushed to your session."
          : "";

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              instanceId,
              status: "started",
              type: "spawn",
              agent: params.agent,
              model: agentDef.model || "sonnet",
              interactive,
              budget,
              message: `Agent '${params.agent}' started.${interactiveNote} Use cc_status("${instanceId}") to check progress.`,
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
  }));

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

      // Cancel any pending questions first (unblocks promises)
      const cancelledQuestions = cancelAllForInstance(params.instanceId);

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
          text: `Instance ${params.instanceId} aborted. Ran for ${Math.round((inst.endedAt - inst.startedAt) / 1000)}s.${cancelledQuestions > 0 ? ` Cancelled ${cancelledQuestions} pending question(s).` : ""}`,
        }],
      };
    },
  });

  // ── Tool: cc_respond ───────────────────────────────────────────────────

  api.registerTool({
    name: "cc_respond",
    label: "Claude Code Respond",
    description:
      "Answer a pending question from a running Claude Code instance. " +
      "When an interactive instance needs clarification or permission approval, " +
      "a question notification is pushed to your session. Use this tool to answer it. " +
      "The agent is blocked waiting — respond promptly.",
    parameters: {
      type: "object",
      properties: {
        questionId: {
          type: "string",
          description: "The question ID to respond to (from the [Agent Question] notification or cc_status output).",
        },
        action: {
          type: "string",
          enum: ["allow", "deny", "answer"],
          description: "'allow' to approve a permission request, 'deny' to reject, 'answer' for free-text response to a clarification.",
        },
        text: {
          type: "string",
          description: "Free-text response for 'answer' action, or reason for denial. Required for 'answer', optional for 'allow'/'deny'.",
        },
      },
      required: ["questionId", "action"],
    },
    async execute(_id: string, params: any) {
      const answer: QuestionAnswer = {
        action: params.action,
        text: params.text,
      };

      const resolved = resolveQuestion(params.questionId, answer);

      if (!resolved) {
        return {
          content: [{
            type: "text" as const,
            text: `Question "${params.questionId}" not found or already answered. Use cc_status to see current pending questions.`,
          }],
        };
      }

      return {
        content: [{
          type: "text" as const,
          text: `Response delivered to question "${params.questionId}" (action: ${params.action}). Agent will continue.`,
        }],
      };
    },
  });

  // ── Tool: cc_plan_interactive ─────────────────────────────────────────

  api.registerTool({
    name: "cc_plan_interactive",
    label: "Claude Code Interactive Planning",
    description:
      "Start an interactive planning session. A planner agent explores the codebase " +
      "and asks clarifying questions about requirements, scope, and preferences before " +
      "producing a detailed execution plan. Questions are pushed to your session via hooks — " +
      "answer them with cc_respond. When complete, use cc_result to retrieve the plan, " +
      "then execute it with cc_orchestrate.",
    parameters: {
      type: "object",
      properties: {
        task: {
          type: "string",
          description: "What to plan. Be specific about the goal but leave room for the planner to ask about details.",
        },
        context: {
          type: "string",
          description: "Additional context: relevant file paths, constraints, prior decisions.",
        },
        cwd: {
          type: "string",
          description: "Working directory for the project.",
        },
      },
      required: ["task"],
    },
    async execute(_id: string, params: any) {
      try {
        const queryFn = await getQuery();
        const instanceId = randomUUID().slice(0, 8);
        const cwd = params.cwd || process.env.HOME || "/home/alansrobotlab";
        const abort = new AbortController();

        const workMode = getWorkModeScope();
        const agents = buildAgentDefs(api.logger, workMode.scope ? workMode : undefined, true);

        const orchestratorPrompt = buildInteractivePlanningPrompt(
          params.task, params.context, loadVaultPrompt("orchestrator"),
          workMode.scope ? workMode : null,
        );

        const instance: CcInstance = {
          id: instanceId,
          type: "orchestrate",
          status: "running",
          task: params.task,
          pipeline: "interactive-plan",
          planOnly: true,
          interactive: true,
          startedAt: Date.now(),
          costUsd: 0,
          turns: 0,
          budgetUsd: 2.0,
          maxTurns: 20,
          pendingQuestions: [],
          recentMessages: [],
          _abort: abort,
        };

        const q = queryFn({
          prompt: orchestratorPrompt,
          options: {
            systemPrompt: {
              type: "preset" as const,
              preset: "claude_code" as const,
              append: "\nYou are a project coordinator in INTERACTIVE PLANNING mode. " +
                "Explore the codebase with Read/Glob/Grep, ask clarifying questions with AskUserQuestion, " +
                "then produce a detailed execution plan. Do NOT modify any files or dispatch agents.",
            },
            cwd,
            model: "opus",
            maxBudgetUsd: 2.0,
            maxTurns: 20,
            permissionMode: "default",
            allowDangerouslySkipPermissions: false,
            canUseTool: buildCanUseTool(instance),
            allowedTools: [
              "Read", "Glob", "Grep", "AskUserQuestion",
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
            spawnClaudeCodeProcess: spawnClaudeCode,
          },
        });

        instance._query = q;
        instances.set(instanceId, instance);
        pruneInstances();

        // Consume in background — questions are pushed via hooks
        consumeQuery(instance, q, api.logger, notifyCompletion, notifyProgress).catch((err) => {
          api.logger.error?.(`agent-orchestrator: consumeQuery error for ${instanceId}:`, err);
        });

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              instanceId,
              status: "started",
              type: "interactive-plan",
              message: `Interactive planning started. The planner will explore the codebase and push questions to your session. Answer with cc_respond. Use cc_result("${instanceId}") when complete.`,
            }, null, 2),
          }],
        };
      } catch (err: any) {
        return {
          content: [{
            type: "text" as const,
            text: `cc_plan_interactive error: ${err.message}`,
          }],
        };
      }
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
          interactive: inst.interactive,
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

        cancelAllForInstance(id);
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
    const pending = listPendingQuestions(inst.id);
    return {
      id: inst.id,
      type: inst.type,
      status: inst.status,
      task: inst.task,
      pipeline: inst.pipeline,
      agent: inst.agent,
      interactive: inst.interactive,
      elapsedMs: (inst.endedAt || Date.now()) - inst.startedAt,
      costUsd: Math.round(inst.costUsd * 1000) / 1000,
      turns: inst.turns,
      activity: inst.activity,
      resultPreview: inst.resultText ? inst.resultText.slice(0, 500) : undefined,
      error: inst.error,
      pendingQuestions: pending.length > 0 ? pending : undefined,
    };
  }

  // ── Startup ────────────────────────────────────────────────────────────

  api.logger.info?.("agent-orchestrator: loaded — 7 tools (cc_orchestrate, cc_spawn, cc_status, cc_result, cc_abort, cc_respond, cc_plan_interactive)");
}
