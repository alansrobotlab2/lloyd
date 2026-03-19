/**
 * backlog.ts — SQLite CRUD for tasks and boards
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { join } from "path";
import { homedir } from "os";
import type { PluginContext } from "./types.js";
import { jsonResponse, readBody, handleCorsOptions, requirePost } from "./helpers.js";

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

export function registerBacklogRoutes(ctx: PluginContext) {
  const { api } = ctx;

  // GET /api/mc/backlog/boards
  api.registerHttpRoute({
    path: "/api/mc/backlog/boards",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const db = getBacklogDb();
        const rows = db.prepare(`
          SELECT b.*, (SELECT COUNT(*) FROM tasks t WHERE t.board_id = b.id AND t.status != 'done') AS tasks_count
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
    auth: "plugin",
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

  // POST /api/mc/backlog/task-update
  api.registerHttpRoute({
    path: "/api/mc/backlog/task-update",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
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
            if (key === "status" && val !== old.status) {
              db.prepare("INSERT INTO task_activities (task_id, action, field_name, old_value, new_value, source) VALUES (?, 'moved', 'status', ?, ?, 'web')").run(id, old.status, val);
            }
          }
        }

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

  // POST /api/mc/backlog/task-delete
  api.registerHttpRoute({
    path: "/api/mc/backlog/task-delete",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
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

  // POST /api/mc/backlog/task-create
  api.registerHttpRoute({
    path: "/api/mc/backlog/task-create",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
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
}
