/**
 * index.ts — ClawDeck OpenClaw plugin
 *
 * Provides kanban task board tools via ClawDeck's REST API (localhost:3001).
 * Pure TypeScript — uses native fetch(), no MCP server subprocess needed.
 *
 * Also exposes a webhook endpoint at POST /webhook/clawdeck for real-time
 * task notifications from ClawDeck (create, update). Notifications are
 * injected as system events into the agent's next turn.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "fs";
import { join } from "path";

interface ClawDeckConfig {
  baseUrl: string;
  apiToken: string;
  agentName: string;
  agentEmoji: string;
}

interface ClawDeckTask {
  id: number;
  name: string;
  description?: string;
  status: string;
  position: number;
  blocked: boolean;
  assigned: boolean;
  tags: string[];
  board_id: number;
  priority?: string;
  created_at: string;
  updated_at: string;
}

interface ClawDeckBoard {
  id: number;
  name: string;
  icon: string;
  color: string;
  position: number;
  tasks_count?: number;
}

function loadConfig(): ClawDeckConfig {
  const raw = readFileSync(join(__dirname, "config.json"), "utf-8");
  return JSON.parse(raw);
}

function textResult(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

function formatTask(t: ClawDeckTask): string {
  const parts = [`#${t.id} — ${t.name}`];
  parts.push(`  Status: ${t.status}${t.blocked ? " [BLOCKED]" : ""}${t.assigned ? " [assigned]" : ""}`);
  if (t.priority && t.priority !== "none") parts.push(`  Priority: ${t.priority}`);
  if (t.tags?.length) parts.push(`  Tags: ${t.tags.join(", ")}`);
  if (t.description) {
    const desc = t.description.length > 200 ? t.description.slice(0, 200) + "..." : t.description;
    parts.push(`  Description: ${desc}`);
  }
  return parts.join("\n");
}

function formatBoard(b: ClawDeckBoard): string {
  return `${b.icon} ${b.name} (id: ${b.id}, ${b.tasks_count ?? "?"} tasks, color: ${b.color})`;
}

export default function register(api: OpenClawPluginApi) {
  let config: ClawDeckConfig;
  try {
    config = loadConfig();
  } catch (err: any) {
    api.logger.error?.(`clawdeck: failed to load config.json: ${err.message}`);
    return;
  }

  api.logger.info?.(
    `clawdeck: loaded — ${config.baseUrl}, agent: ${config.agentEmoji} ${config.agentName}`,
  );

  // ── Webhook endpoint for task notifications ─────────────────────────

  const MAIN_SESSION_KEY = "agent:main:main";

  function readBody(req: IncomingMessage): Promise<string> {
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
      req.on("error", reject);
    });
  }

  function formatWebhookEvent(event: string, task: any): string {
    const parts: string[] = [];
    const label = event === "task.created" ? "New task created" : "Task updated";
    parts.push(`[ClawDeck] ${label}: #${task.id} — ${task.name}`);
    if (task.status) parts.push(`Status: ${task.status}`);
    if (task.priority && task.priority !== "none") parts.push(`Priority: ${task.priority}`);
    if (task.board) parts.push(`Board: ${task.board}`);
    if (task.assigned) parts.push("Assigned to agent");
    if (task.changes && Object.keys(task.changes).length) {
      parts.push(`Changed: ${Object.keys(task.changes).join(", ")}`);
    }
    parts.push(
      "Use clawdeck_get_task to read full details. Triage: assess priority and complexity, then decide whether to start working or just acknowledge.",
    );
    return parts.join("\n");
  }

  api.registerHttpRoute({
    path: "/webhook/clawdeck",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") {
        res.writeHead(405, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Method not allowed" }));
        return;
      }

      // Validate token
      const authHeader = req.headers.authorization ?? "";
      const token = authHeader.replace(/^Bearer\s+/i, "");
      if (token !== config.apiToken) {
        res.writeHead(401, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Unauthorized" }));
        return;
      }

      try {
        const body = JSON.parse(await readBody(req));
        const event: string = body.event ?? "task.unknown";
        const task = body.task ?? {};

        api.logger.info?.(`clawdeck: webhook received — ${event} #${task.id ?? "?"}`);

        const notification = formatWebhookEvent(event, task);
        api.runtime.system.enqueueSystemEvent(notification, {
          sessionKey: MAIN_SESSION_KEY,
          contextKey: "clawdeck-webhook",
        });

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, event }));
      } catch (err: any) {
        api.logger.warn?.(`clawdeck: webhook parse error — ${err.message}`);
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Bad request" }));
      }
    },
  });

  // ── ClawDeck API helper ─────────────────────────────────────────────

  async function clawdeckFetch(
    method: string,
    path: string,
    body?: Record<string, unknown>,
  ): Promise<{ ok: boolean; status: number; data: any }> {
    const url = `${config.baseUrl}/api/v1${path}`;
    const headers: Record<string, string> = {
      Authorization: `Bearer ${config.apiToken}`,
      "X-Agent-Name": config.agentName,
      "X-Agent-Emoji": config.agentEmoji,
      Accept: "application/json",
    };
    if (body) headers["Content-Type"] = "application/json";

    const resp = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    const text = await resp.text();
    let data: any;
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
    return { ok: resp.ok, status: resp.status, data };
  }

  // ── clawdeck_boards ─────────────────────────────────────────────────────

  api.registerTool({
    name: "clawdeck_boards",
    label: "ClawDeck Boards",
    description: "List all boards on the ClawDeck kanban. Returns board names, icons, colors, and task counts.",
    parameters: {
      type: "object" as const,
      properties: {},
      required: [] as string[],
    },
    async execute() {
      try {
        const { ok, status, data } = await clawdeckFetch("GET", "/boards");
        if (!ok) return textResult(`ClawDeck API error (${status}): ${JSON.stringify(data)}`);
        const boards: ClawDeckBoard[] = data.boards ?? data;
        if (!boards.length) return textResult("No boards found.");
        return textResult(boards.map(formatBoard).join("\n"));
      } catch (err: any) {
        return textResult(`clawdeck_boards error: ${err.message}`);
      }
    },
  });

  // ── clawdeck_tasks ──────────────────────────────────────────────────────

  api.registerTool({
    name: "clawdeck_tasks",
    label: "ClawDeck Tasks",
    description:
      "List tasks from ClawDeck with optional filters. Use to check the backlog, see assigned work, or find blocked tasks.",
    parameters: {
      type: "object" as const,
      properties: {
        status: {
          type: "string" as const,
          enum: ["inbox", "up_next", "in_progress", "in_review", "done"],
          description: "Filter by task status",
        },
        assigned: {
          type: "boolean" as const,
          description: "Filter to assigned tasks only (true) or unassigned (false)",
        },
        blocked: {
          type: "boolean" as const,
          description: "Filter to blocked tasks (true) or unblocked (false)",
        },
        board_id: {
          type: "integer" as const,
          description: "Filter by board ID",
        },
        tag: {
          type: "string" as const,
          description: "Filter by tag name",
        },
      },
      required: [] as string[],
    },
    async execute(
      _id: string,
      params: { status?: string; assigned?: boolean; blocked?: boolean; board_id?: number; tag?: string },
    ) {
      try {
        const query = new URLSearchParams();
        if (params.status) query.set("status", params.status);
        if (params.assigned !== undefined) query.set("assigned", String(params.assigned));
        if (params.blocked !== undefined) query.set("blocked", String(params.blocked));
        if (params.board_id !== undefined) query.set("board_id", String(params.board_id));
        if (params.tag) query.set("tag", params.tag);

        const qs = query.toString();
        const path = `/tasks${qs ? `?${qs}` : ""}`;
        const { ok, status, data } = await clawdeckFetch("GET", path);
        if (!ok) return textResult(`ClawDeck API error (${status}): ${JSON.stringify(data)}`);
        const tasks: ClawDeckTask[] = data.tasks ?? data;
        if (!tasks.length) return textResult("No tasks match the filters.");
        return textResult(`${tasks.length} task(s):\n\n${tasks.map(formatTask).join("\n\n")}`);
      } catch (err: any) {
        return textResult(`clawdeck_tasks error: ${err.message}`);
      }
    },
  });

  // ── clawdeck_next_task ──────────────────────────────────────────────────

  api.registerTool({
    name: "clawdeck_next_task",
    label: "ClawDeck Next Task",
    description:
      "Get the next assigned task ready to work on. Returns the highest-priority assigned up_next task, or null if the queue is empty.",
    parameters: {
      type: "object" as const,
      properties: {},
      required: [] as string[],
    },
    async execute() {
      try {
        const { ok, status, data } = await clawdeckFetch("GET", "/tasks/next");
        if (!ok) {
          if (status === 404) return textResult("No tasks in the queue — nothing assigned.");
          return textResult(`ClawDeck API error (${status}): ${JSON.stringify(data)}`);
        }
        const task: ClawDeckTask = data.task ?? data;
        return textResult(`Next task:\n\n${formatTask(task)}`);
      } catch (err: any) {
        return textResult(`clawdeck_next_task error: ${err.message}`);
      }
    },
  });

  // ── clawdeck_get_task ───────────────────────────────────────────────────

  api.registerTool({
    name: "clawdeck_get_task",
    label: "ClawDeck Get Task",
    description:
      "Get full details for a single task by ID, including description, status, tags, and assignment state.",
    parameters: {
      type: "object" as const,
      properties: {
        id: {
          type: "integer" as const,
          description: "Task ID",
        },
      },
      required: ["id"] as string[],
    },
    async execute(_id: string, params: { id: number }) {
      try {
        const { ok, status, data } = await clawdeckFetch("GET", `/tasks/${params.id}`);
        if (!ok) return textResult(`ClawDeck API error (${status}): ${JSON.stringify(data)}`);
        const task: ClawDeckTask = data.task ?? data;
        return textResult(formatTask(task));
      } catch (err: any) {
        return textResult(`clawdeck_get_task error: ${err.message}`);
      }
    },
  });

  // ── clawdeck_update_task ────────────────────────────────────────────────

  api.registerTool({
    name: "clawdeck_update_task",
    label: "ClawDeck Update Task",
    description:
      "Update a task's status, blocked state, or add an activity note. Use to move tasks through the pipeline (up_next → in_progress → in_review) and communicate progress.",
    parameters: {
      type: "object" as const,
      properties: {
        id: {
          type: "integer" as const,
          description: "Task ID to update",
        },
        status: {
          type: "string" as const,
          enum: ["inbox", "up_next", "in_progress", "in_review", "done"],
          description: "New status for the task",
        },
        blocked: {
          type: "boolean" as const,
          description: "Set blocked state (true = blocked, false = unblocked)",
        },
        activity_note: {
          type: "string" as const,
          description: "Activity note to add (visible in task activity feed, attributed to the agent)",
        },
      },
      required: ["id"] as string[],
    },
    async execute(
      _id: string,
      params: { id: number; status?: string; blocked?: boolean; activity_note?: string },
    ) {
      try {
        const body: Record<string, unknown> = {};
        const task: Record<string, unknown> = {};
        if (params.status !== undefined) task.status = params.status;
        if (params.blocked !== undefined) task.blocked = params.blocked;
        if (Object.keys(task).length) body.task = task;
        if (params.activity_note) body.activity_note = params.activity_note;

        if (!Object.keys(body).length) return textResult("Nothing to update — provide status, blocked, or activity_note.");

        const { ok, status, data } = await clawdeckFetch("PATCH", `/tasks/${params.id}`, body);
        if (!ok) return textResult(`ClawDeck API error (${status}): ${JSON.stringify(data)}`);
        const updated: ClawDeckTask = data.task ?? data;
        return textResult(`Updated task:\n\n${formatTask(updated)}`);
      } catch (err: any) {
        return textResult(`clawdeck_update_task error: ${err.message}`);
      }
    },
  });

  // ── clawdeck_create_task ────────────────────────────────────────────────

  api.registerTool({
    name: "clawdeck_create_task",
    label: "ClawDeck Create Task",
    description:
      "Create a new task on a ClawDeck board. Use when you discover work that should be tracked (tech debt, follow-ups, new ideas).",
    parameters: {
      type: "object" as const,
      properties: {
        name: {
          type: "string" as const,
          description: "Task title (short, descriptive)",
        },
        description: {
          type: "string" as const,
          description: "Detailed task description",
        },
        board_id: {
          type: "integer" as const,
          description: "Board ID to create the task on",
        },
        status: {
          type: "string" as const,
          enum: ["inbox", "up_next"],
          description: "Initial status (default: inbox)",
        },
        tags: {
          type: "array" as const,
          items: { type: "string" as const },
          description: "Tags for the task (e.g. [\"bug\", \"auth\"])",
        },
        priority: {
          type: "string" as const,
          enum: ["none", "low", "medium", "high"],
          description: "Task priority (default: none)",
        },
      },
      required: ["name"] as string[],
    },
    async execute(
      _id: string,
      params: {
        name: string;
        description?: string;
        board_id?: number;
        status?: string;
        tags?: string[];
        priority?: string;
      },
    ) {
      try {
        const task: Record<string, unknown> = { name: params.name };
        if (params.description) task.description = params.description;
        if (params.board_id !== undefined) task.board_id = params.board_id;
        if (params.status) task.status = params.status;
        if (params.tags?.length) task.tags = params.tags;
        if (params.priority) task.priority = params.priority;

        const { ok, status, data } = await clawdeckFetch("POST", "/tasks", { task });
        if (!ok) return textResult(`ClawDeck API error (${status}): ${JSON.stringify(data)}`);
        const created: ClawDeckTask = data.task ?? data;
        return textResult(`Created task:\n\n${formatTask(created)}`);
      } catch (err: any) {
        return textResult(`clawdeck_create_task error: ${err.message}`);
      }
    },
  });
}
