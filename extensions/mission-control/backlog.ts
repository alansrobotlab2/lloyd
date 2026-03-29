/**
 * backlog.ts — Markdown file CRUD for tasks and boards
 * Replaces SQLite with vault markdown files at ~/obsidian/backlog/{id}-{slug}.md
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { join } from "path";
import { homedir } from "os";
import { readFileSync, writeFileSync, unlinkSync, existsSync, mkdirSync } from "fs";
import { globSync } from "fs";
import type { PluginContext } from "./types.js";
import { jsonResponse, readBody, handleCorsOptions, requirePost } from "./helpers.js";

const BACKLOG_DIR = join(homedir(), "obsidian", "backlog");

// Ensure backlog directory exists
if (!existsSync(BACKLOG_DIR)) {
  mkdirSync(BACKLOG_DIR, { recursive: true });
}

// Stable hash function for board names (string -> number)
function hashBoardName(name: string): number {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    const char = name.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

// Parse YAML frontmatter from markdown content
function parseFrontmatter(content: string): { frontmatter: Record<string, any>; body: string } {
  if (!content.startsWith("---")) {
    return { frontmatter: {}, body: content };
  }
  
  const endMarker = content.indexOf("\n---", 3);
  if (endMarker === -1) {
    return { frontmatter: {}, body: content };
  }
  
  const frontmatterStr = content.slice(4, endMarker);
  const body = content.slice(endMarker + 4);
  
  // Use yaml package for parsing
  const { parse } = require("yaml") as typeof import("yaml");
  let frontmatter: Record<string, any> = {};
  try {
    frontmatter = parse(frontmatterStr) || {};
  } catch {
    // Fall back to empty if parsing fails
    frontmatter = {};
  }
  
  return { frontmatter, body };
}

// Stringify frontmatter back to YAML
function stringifyFrontmatter(fm: Record<string, any>): string {
  const { stringify } = require("yaml") as typeof import("yaml");
  return stringify(fm).trim();
}

// Rebuild markdown content from frontmatter and body
function rebuildContent(frontmatter: Record<string, any>, body: string): string {
  const fmYaml = stringifyFrontmatter(frontmatter);
  return `---\n${fmYaml}\n---\n${body}`;
}

// Extract task name from markdown body (first heading)
function extractNameFromBody(body: string): string {
  const match = body.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : "";
}

// Extract description from body (between title and Activity Log)
function extractDescription(body: string): string {
  const lines = body.split("\n");
  const result: string[] = [];
  let inActivityLog = false;
  let started = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("# ")) {
      started = true;
      continue; // Skip the title line itself
    }
    if (line.startsWith("## Activity Log")) {
      inActivityLog = true;
      continue;
    }
    if (inActivityLog) {
      continue;
    }
    if (started) {
      result.push(line);
    }
  }
  
  return result.join("\n").trim();
}

// Find Activity Log section in body
function findActivityLogSection(body: string): { startIndex: number; endIndex: number } {
  const lines = body.split("\n");
  let activityStart = -1;
  let activityEnd = lines.length;
  
  // Find the LAST "## Activity Log" heading
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].startsWith("## Activity Log")) {
      activityStart = i;
      break;
    }
  }
  
  if (activityStart === -1) {
    // No Activity Log section found, append at end
    return { startIndex: lines.length, endIndex: lines.length };
  }
  
  // Find where Activity Log ends (next heading or end of file)
  for (let i = activityStart + 1; i < lines.length; i++) {
    if (lines[i].startsWith("#") && !lines[i].startsWith("## Activity Log")) {
      activityEnd = i;
      break;
    }
  }
  
  return { startIndex: activityStart + 1, endIndex: activityEnd };
}

// Append activity entry to body
function appendActivityEntry(body: string, entry: string): string {
  const { startIndex, endIndex } = findActivityLogSection(body);
  const lines = body.split("\n");
  
  // Ensure we have an Activity Log heading
  let activityHeadingIndex = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].startsWith("## Activity Log")) {
      activityHeadingIndex = i;
      break;
    }
  }
  
  if (activityHeadingIndex === -1) {
    // Add Activity Log section if it doesn't exist
    return `${body.trim()}\n\n## Activity Log\n\n${entry}\n`;
  }
  
  // Insert after the Activity Log heading
  const before = lines.slice(0, activityHeadingIndex + 1).join("\n");
  const after = lines.slice(activityHeadingIndex + 1).join("\n");
  
  // Trim trailing whitespace from after section
  const afterTrimmed = after.trim();
  
  if (afterTrimmed) {
    return `${before}\n\n${entry}\n${afterTrimmed}\n`;
  }
  
  return `${before}\n\n${entry}\n`;
}

// Generate slug from task name
function generateSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

// Find task file by ID
function findTaskFileById(id: number): string | null {
  const pattern = `${BACKLOG_DIR}/${id}-*.md`;
  const files = globSync(pattern);
  return files.length > 0 ? files[0] : null;
}

// Read and parse a task file
function readTaskFile(filePath: string): { frontmatter: Record<string, any>; body: string } | null {
  try {
    const content = readFileSync(filePath, "utf-8");
    return parseFrontmatter(content);
  } catch {
    return null;
  }
}

// Write a task file
function writeTaskFile(filePath: string, frontmatter: Record<string, any>, body: string): void {
  const content = rebuildContent(frontmatter, body);
  writeFileSync(filePath, content, "utf-8");
}

// Convert task file to BacklogTask response object
function taskFileToBacklogTask(filePath: string, frontmatter: Record<string, any>, body: string): any {
  const name = extractNameFromBody(body);
  const description = extractDescription(body);
  const tags = Array.isArray(frontmatter.tags) ? frontmatter.tags : (frontmatter.tags ? [frontmatter.tags] : []);
  const completed = frontmatter.status === "done";
  
  return {
    id: frontmatter.id,
    name,
    description,
    priority: frontmatter.priority || "none",
    status: frontmatter.status || "inbox",
    blocked: !!frontmatter.blocked,
    tags,
    completed,
    due_date: frontmatter.due_date || null,
    position: frontmatter.id, // Use id as position
    assigned_to_agent: !!frontmatter.assigned,
    board_id: hashBoardName(frontmatter.board || "default"),
    url: "",
    created_at: frontmatter.created || "",
    updated_at: frontmatter.updated || "",
  };
}

// Convert task file to BacklogBoard response object
function taskFileToBoardEntry(frontmatter: Record<string, any>, isActive: boolean): { board: string; active: boolean } {
  return {
    board: frontmatter.board || "default",
    active: isActive,
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
        const files = globSync(`${BACKLOG_DIR}/*.md`);
        const boardMap = new Map<string, { name: string; count: number }>();
        
        for (const file of files) {
          const result = readTaskFile(file);
          if (!result) continue;
          
          const { frontmatter } = result;
          const board = frontmatter.board || "default";
          const isActive = frontmatter.status !== "done";
          
          if (!boardMap.has(board)) {
            boardMap.set(board, { name: board, count: 0 });
          }
          
          if (isActive) {
            boardMap.get(board)!.count++;
          }
        }
        
        const boards = Array.from(boardMap.entries()).map(([name, data]) => ({
          id: hashBoardName(name),
          name: data.name,
          icon: "📋",
          color: "gray",
          tasks_count: data.count,
        }));
        
        jsonResponse(res, boards);
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
        const url = new URL(req.url || "/", "http://localhost");
        const files = globSync(`${BACKLOG_DIR}/*.md`);
        const tasks: any[] = [];
        
        for (const file of files) {
          const result = readTaskFile(file);
          if (!result) continue;
          
          const { frontmatter, body } = result;
          let include = true;
          
          // Filter by status
          const statusFilter = url.searchParams.get("status");
          if (statusFilter && frontmatter.status !== statusFilter) {
            include = false;
          }
          
          // Filter by assigned
          const assignedFilter = url.searchParams.get("assigned");
          if (assignedFilter !== null) {
            const isAssigned = !!frontmatter.assigned;
            if (assignedFilter === "true" && !isAssigned) include = false;
            if (assignedFilter === "false" && isAssigned) include = false;
          }
          
          // Filter by blocked
          const blockedFilter = url.searchParams.get("blocked");
          if (blockedFilter !== null) {
            const isBlocked = !!frontmatter.blocked;
            if (blockedFilter === "true" && !isBlocked) include = false;
            if (blockedFilter === "false" && isBlocked) include = false;
          }
          
          // Filter by board_id (hashed)
          const boardIdFilter = url.searchParams.get("board_id");
          if (boardIdFilter) {
            const fileBoardId = hashBoardName(frontmatter.board || "default");
            if (fileBoardId !== Number(boardIdFilter)) {
              include = false;
            }
          }
          
          if (include) {
            const task = taskFileToBacklogTask(file, frontmatter, body);
            tasks.push(task);
          }
        }
        
        // Filter by tag
        const tagFilter = url.searchParams.get("tag");
        if (tagFilter) {
          for (const task of tasks) {
            if (!task.tags?.includes(tagFilter)) {
              task._exclude = true;
            }
          }
          const filtered = tasks.filter((t: any) => !t._exclude);
          for (const t of filtered) {
            delete t._exclude;
          }
          filtered.sort((a: any, b: any) => a.id - b.id);
          jsonResponse(res, filtered);
          return;
        }
        
        // Sort by position (id)
        tasks.sort((a, b) => a.id - b.id);
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
        const body = JSON.parse(await readBody(req));
        const { id, activity_note, status, ...fields } = body;
        
        if (!id) {
          jsonResponse(res, { error: "Missing task id" }, 400);
          return;
        }
        
        const filePath = findTaskFileById(id);
        if (!filePath || !existsSync(filePath)) {
          jsonResponse(res, { error: "Task not found" }, 404);
          return;
        }
        
        const result = readTaskFile(filePath);
        if (!result) {
          jsonResponse(res, { error: "Failed to read task" }, 500);
          return;
        }
        
        const { frontmatter, body: taskBody } = result;
        const oldStatus = frontmatter.status;
        
        // Update frontmatter fields
        const now = new Date().toISOString();
        frontmatter.updated = now;
        
        // Update status-related fields
        if (status !== undefined) {
          frontmatter.status = status;
          
          // Handle completed timestamp
          if (status === "done" && oldStatus !== "done") {
            frontmatter.completed = now;
          } else if (status !== "done" && oldStatus === "done") {
            frontmatter.completed = null;
          }
        }
        
        // Update other fields
        for (const key of ["blocked", "priority", "assigned", "due_date"]) {
          if (fields[key] !== undefined) {
            frontmatter[key] = fields[key];
          }
        }
        
        // Update name/description if provided
        let updatedBody = taskBody;
        if (fields.name !== undefined || fields.description !== undefined) {
          const lines = taskBody.split("\n");
          const newLines: string[] = [];
          let inDescription = false;
          let titleWritten = false;
          
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (i === 0 && line.startsWith("# ")) {
              // Update title
              newLines.push(`# ${fields.name || extractNameFromBody(taskBody)}`);
              titleWritten = true;
              continue;
            }
            if (line.startsWith("## Activity Log")) {
              inDescription = false;
            }
            if (!titleWritten && line.startsWith("#")) {
              titleWritten = true;
              continue;
            }
            if (i > 0 && !inDescription) {
              newLines.push(line);
            }
          }
          
          if (fields.description !== undefined) {
            // Rebuild body with new description
            const name = fields.name || extractNameFromBody(taskBody);
            updatedBody = `# ${name}\n\n${fields.description}\n\n## Activity Log\n\n`;
          } else {
            updatedBody = newLines.join("\n");
          }
        }
        
        // Append activity entries
        if (status !== undefined && status !== oldStatus) {
          const timestamp = new Date().toISOString().slice(0, 16).replace("T", " ");
          const entry = `- **${timestamp}** — Moved to ${status} (web)`;
          updatedBody = appendActivityEntry(updatedBody, entry);
        }
        
        if (activity_note) {
          const timestamp = new Date().toISOString().slice(0, 16).replace("T", " ");
          const entry = `- **${timestamp}** — Updated (${activity_note})`;
          updatedBody = appendActivityEntry(updatedBody, entry);
        }
        
        // Write updated file
        writeTaskFile(filePath, frontmatter, updatedBody);
        
        // Read back and return
        const updated = readTaskFile(filePath);
        if (!updated) {
          jsonResponse(res, { error: "Failed to read updated task" }, 500);
          return;
        }
        
        jsonResponse(res, { task: taskFileToBacklogTask(filePath, updated.frontmatter, updated.body) });
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
        const body = JSON.parse(await readBody(req));
        if (!body.id) {
          jsonResponse(res, { error: "Missing task id" }, 400);
          return;
        }
        
        const filePath = findTaskFileById(body.id);
        if (filePath && existsSync(filePath)) {
          unlinkSync(filePath);
        }
        
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
        const body = JSON.parse(await readBody(req));
        if (!body.name?.trim()) {
          jsonResponse(res, { error: "Missing task name" }, 400);
          return;
        }
        
        // Find max ID
        const files = globSync(`${BACKLOG_DIR}/*.md`);
        let maxId = 0;
        for (const file of files) {
          const result = readTaskFile(file);
          if (result && result.frontmatter.id) {
            maxId = Math.max(maxId, result.frontmatter.id);
          }
        }
        const newId = maxId + 1;
        
        // Generate slug
        const slug = generateSlug(body.name);
        const filePath = `${BACKLOG_DIR}/${newId}-${slug}.md`;
        
        // Resolve board_id to board name by scanning existing boards
        let boardName = "default";
        if (body.board_id) {
          const boardFiles = globSync(`${BACKLOG_DIR}/*.md`);
          for (const bf of boardFiles) {
            const br = readTaskFile(bf);
            if (br && hashBoardName(br.frontmatter.board || "default") === body.board_id) {
              boardName = br.frontmatter.board || "default";
              break;
            }
          }
        }
        
        // Prepare frontmatter
        const now = new Date().toISOString();
        const frontmatter: Record<string, any> = {
          type: "backlog",
          id: newId,
          board: boardName,
          status: body.status || "inbox",
          priority: body.priority || "none",
          tags: Array.isArray(body.tags) ? body.tags : [],
          blocked: false,
          assigned: false,
          created: now,
          updated: now,
          completed: null,
        };
        
        // Prepare body
        const description = body.description || "";
        const activityLog = `- **${now.slice(0, 16).replace("T", " ")}** — Created (web)\n`;
        const taskBody = `# ${body.name.trim()}\n\n${description}\n\n## Activity Log\n\n${activityLog}`;
        
        // Write file
        writeTaskFile(filePath, frontmatter, taskBody);
        
        // Read back and return
        const created = readTaskFile(filePath);
        if (!created) {
          jsonResponse(res, { error: "Failed to read created task" }, 500);
          return;
        }
        
        jsonResponse(res, { task: taskFileToBacklogTask(filePath, created.frontmatter, created.body) });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });
}
