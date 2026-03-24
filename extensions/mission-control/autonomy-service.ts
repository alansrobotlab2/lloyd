import os from "node:os";
import { DatabaseSync } from "node:sqlite";

const AUTONOMY_DB = os.homedir() + "/.openclaw/autonomy.db";

let _db: DatabaseSync | null = null;

function getDb() {
  if (!_db) {
    _db = new DatabaseSync(AUTONOMY_DB);
    _db.exec("PRAGMA journal_mode=WAL");
    _db.exec("PRAGMA foreign_keys=ON");
    
    // Run reconciliation once on first DB access
    reconcileStuckTasks();
  }
  return _db;
}

export function reconcileStuckTasks(): void {
  const db = getDb();
  
  // Find and reset stuck tasks (status = 'in_progress')
  const tasksStmt = db.prepare("SELECT id FROM tasks WHERE status = 'in_progress'");
  const stuckTasks = tasksStmt.all() as any[];
  const taskCount = stuckTasks.length;
  
  if (taskCount > 0) {
    db.prepare("UPDATE tasks SET status = 'up_next' WHERE status = 'in_progress'").run();
  }
  
  // Find and mark orphaned runs (status = 'running')
  const runsStmt = db.prepare("SELECT id FROM runs WHERE status = 'running'");
  const orphanedRuns = runsStmt.all() as any[];
  const runCount = orphanedRuns.length;
  
  if (runCount > 0) {
    const now = new Date().toISOString();
    db.prepare(
      "UPDATE runs SET status = 'failed', completed = ?, summary = ? WHERE status = 'running'"
    ).run(now, 'Interrupted by gateway restart');
  }
  
  console.log(`[autonomy] reconciled ${taskCount} stuck tasks, ${runCount} orphaned runs`);
}

export async function getRuns(taskId: number, limit = 20): Promise<any[]> {
  const db = getDb();
  const stmt = db.prepare("SELECT * FROM runs WHERE task_id = ? ORDER BY id DESC LIMIT ?");
  return stmt.all(taskId, limit);
}

export async function getTasks(): Promise<any[]> {
  const db = getDb();
  const stmt = db.prepare("SELECT * FROM tasks ORDER BY CASE WHEN last_run IS NULL THEN 999999 ELSE (julianday('now') - julianday(last_run)) * 86400.0 / (86400.0 / COALESCE(runs_per_day, 1.0)) END DESC, priority, next_run, id");
  return stmt.all();
}

export async function getTask(id: number): Promise<any | null> {
  const db = getDb();
  const stmt = db.prepare("SELECT * FROM tasks WHERE id = ?");
  const row = stmt.get(id);
  return row || null;
}

export async function writeTask(data: any): Promise<any> {
  const db = getDb();
  const now = new Date().toISOString();
  
  if (data.id) {
    // UPDATE
    const sets: string[] = [];
    const params: any[] = [];
    
    if (data.name !== undefined) { sets.push("name = ?"); params.push(data.name); }
    if (data.description !== undefined) { sets.push("description = ?"); params.push(data.description); }
    if (data.status !== undefined) { sets.push("status = ?"); params.push(data.status); }
    if (data.priority !== undefined) { sets.push("priority = ?"); params.push(data.priority); }
    if (data.frequency !== undefined) { sets.push("frequency = ?"); params.push(data.frequency); }
    if (data.scheduled_at !== undefined) { sets.push("scheduled_at = ?"); params.push(data.scheduled_at); }
    if (data.next_run !== undefined) { sets.push("next_run = ?"); params.push(data.next_run); }
    if (data.auto_advance !== undefined) { sets.push("auto_advance = ?"); params.push(data.auto_advance ? 1 : 0); }
    if (data.tags !== undefined) { sets.push("tags = ?"); params.push(JSON.stringify(data.tags)); }
    if (data.runs_per_day !== undefined) { sets.push("runs_per_day = ?"); params.push(data.runs_per_day); }
    if (data.depends_on !== undefined) { sets.push("depends_on = ?"); params.push(data.depends_on); }
    if (data.pipeline !== undefined) { sets.push("pipeline = ?"); params.push(data.pipeline); }
    if (data.agent_id !== undefined) { sets.push("agent_id = ?"); params.push(data.agent_id); }
    if (data.skill_path !== undefined) { sets.push("skill_path = ?"); params.push(data.skill_path); }
    if (data.model !== undefined) { sets.push("model = ?"); params.push(data.model); }
    if (data.timeout_seconds !== undefined) { sets.push("timeout_seconds = ?"); params.push(data.timeout_seconds); }
    if (data.cron_id !== undefined) { sets.push("cron_id = ?"); params.push(data.cron_id); }
    if (data.last_run !== undefined) { sets.push("last_run = ?"); params.push(data.last_run); }
    
    if (data.activity_note !== undefined) {
      // Log activity note to runs table
      db.prepare(
        "INSERT INTO runs (task_id, started, completed, status, summary) VALUES (?, ?, ?, 'success', ?)"
      ).run(data.id, now, now, data.activity_note);
    }
    
    if (sets.length > 0) {
      sets.push("updated_at = ?");
      params.push(now);
      params.push(data.id);
      db.prepare(`UPDATE tasks SET ${sets.join(", ")} WHERE id = ?`).run(...params);
      const row = db.prepare("SELECT * FROM tasks WHERE id = ?").get(data.id);
      return row;
    } else {
      // Nothing to update, just return current
      return db.prepare("SELECT * FROM tasks WHERE id = ?").get(data.id);
    }
  } else {
    // INSERT
    const tags = data.tags ? JSON.stringify(data.tags) : "[]";
    const result = db.prepare(
      "INSERT INTO tasks (name, description, status, priority, frequency, scheduled_at, next_run, auto_advance, tags, created_at, updated_at, runs_per_day, depends_on, pipeline, agent_id, skill_path, model, timeout_seconds, cron_id, last_run) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    ).run(
      data.name || "", 
      data.description || "", 
      data.status || "inbox", 
      data.priority || "medium", 
      data.frequency || null, 
      data.scheduled_at || null, 
      data.next_run || null, 
      data.auto_advance ? 1 : 0, 
      tags, 
      now, 
      now,
      data.runs_per_day !== undefined ? data.runs_per_day : null,
      data.depends_on !== undefined ? data.depends_on : null,
      data.pipeline !== undefined ? data.pipeline : null,
      data.agent_id !== undefined ? data.agent_id : null,
      data.skill_path !== undefined ? data.skill_path : null,
      data.model !== undefined ? data.model : null,
      data.timeout_seconds !== undefined ? data.timeout_seconds : null,
      data.cron_id !== undefined ? data.cron_id : null,
      data.last_run !== undefined ? data.last_run : null
    );
    
    const row = db.prepare("SELECT * FROM tasks WHERE id = ?").get(result.lastInsertRowid);
    return row;
  }
}

export async function deleteTask(id: number): Promise<any> {
  const db = getDb();
  db.prepare("DELETE FROM runs WHERE task_id = ?").run(id);
  db.prepare("DELETE FROM tasks WHERE id = ?").run(id);
  return { success: true, id };
}

export async function runTask(id: number): Promise<any> {
  const db = getDb();
  const now = new Date().toISOString();
  
  // Get task
  const row = db.prepare("SELECT * FROM tasks WHERE id = ?").get(id);
  if (!row) throw new Error(`Task ${id} not found`);
  
  // Record run start
  db.prepare(
    "INSERT INTO runs (task_id, started, status) VALUES (?, ?, 'success')"
  ).run(id, now);
  
  return {
    run_id: `run-${id}-${Date.now()}`,
    task_id: id,
    status: "triggered",
    summary: `Task '${row.name}' triggered for execution`,
  };
}

export async function getConfig(key: string): Promise<string | null> {
  const db = getDb();
  const row = db.prepare("SELECT value FROM config WHERE key = ?").get(key);
  return row ? row.value : null;
}

export async function setConfig(key: string, value: string): Promise<void> {
  const db = getDb();
  db.prepare("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)").run(key, value);
}

export async function getAllConfig(): Promise<Record<string, string>> {
  const db = getDb();
  const rows = db.prepare("SELECT key, value FROM config").all();
  const result: Record<string, string> = {};
  rows.forEach((row: any) => { result[row.key] = row.value; });
  return result;
}

/** Get all tasks currently in_progress (used by idler polling in index.ts). */
export function getInProgressTasks(): any[] {
  const db = getDb();
  return db.prepare("SELECT * FROM tasks WHERE status = 'in_progress'").all();
}

/** Mark the active run for a task as completed. */
export function completeActiveRun(taskId: number, status: string = "success", summary?: string): void {
  const db = getDb();
  const now = new Date().toISOString();
  const run = db.prepare(
    "SELECT id, started FROM runs WHERE task_id = ? AND completed IS NULL ORDER BY id DESC LIMIT 1"
  ).get(taskId) as any;
  if (run) {
    const startedAt = run.started ? new Date(run.started).getTime() : Date.now();
    const durationSeconds = Math.round((Date.now() - startedAt) / 1000);
    db.prepare(
      "UPDATE runs SET status = ?, completed = ?, summary = ?, duration_seconds = ? WHERE id = ?"
    ).run(status, now, summary || null, durationSeconds, run.id);
  }
  db.prepare("UPDATE tasks SET last_run = ?, updated_at = ? WHERE id = ?").run(now, now, taskId);
}

/** Mark a task as complete — move from in_progress to done (or up_next for recurring). */
export function completeTask(taskId: number): void {
  const db = getDb();
  const now = new Date().toISOString();
  const task = db.prepare("SELECT frequency FROM tasks WHERE id = ?").get(taskId) as any;
  if (task?.frequency && task.frequency !== "one-time") {
    db.prepare("UPDATE tasks SET status = 'up_next', updated_at = ? WHERE id = ?").run(now, taskId);
  } else {
    db.prepare("UPDATE tasks SET status = 'done', updated_at = ? WHERE id = ?").run(now, taskId);
  }
}
