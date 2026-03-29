import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import yaml from 'yaml';

const HOME = process.env.HOME || os.homedir();
const AUTONOMY_DIR = path.join(HOME, 'obsidian', 'autonomy');
const TASKS_DIR = AUTONOMY_DIR;
const RUNS_DIR = path.join(AUTONOMY_DIR, 'runs');
const CONFIG_FILE = path.join(AUTONOMY_DIR, '_config.md');

// Ensure directories exist
function ensureDirs() {
  if (!fs.existsSync(AUTONOMY_DIR)) {
    fs.mkdirSync(AUTONOMY_DIR, { recursive: true });
  }
  if (!fs.existsSync(RUNS_DIR)) {
    fs.mkdirSync(RUNS_DIR, { recursive: true });
  }
}

// Helper: slugify name for filename
function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .substring(0, 50);
}

// Helper: parse YAML frontmatter from markdown file
function parseFrontmatter(content: string): { frontmatter: any; body: string } {
  const parts = content.split(/^---\s*$/m);
  if (parts.length < 3) {
    // No frontmatter, return empty object
    return { frontmatter: {}, body: content.trim() };
  }
  try {
    const fm = yaml.parse(parts[1]);
    const body = parts.slice(2).join('\n').trim();
    return { frontmatter: fm || {}, body };
  } catch (err) {
    console.error('[autonomy-service] frontmatter parse error:', err);
    return { frontmatter: {}, body: content };
  }
}

// Helper: write object as markdown with frontmatter
function stringifyWithFrontmatter(frontmatter: any, body?: string): string {
  const fm = yaml.stringify(frontmatter).trim();
  return `---\n${fm}\n---\n\n${body || ''}`;
}

// Helper: find task file by ID
function findTaskFile(id: number): string | null {
  ensureDirs();
  const pattern = `${id}-*.md`;
  const files = fs.readdirSync(TASKS_DIR).filter(f => f.startsWith(`${id}-`) && f.endsWith('.md'));
  if (files.length === 0) return null;
  return path.join(TASKS_DIR, files[0]);
}

// Helper: parse task file
function parseTaskFile(filePath: string): any {
  const content = fs.readFileSync(filePath, 'utf8');
  const { frontmatter, body } = parseFrontmatter(content);
  return { ...frontmatter, body };
}

// Helper: write task file
function writeTaskFile(task: any, body?: string): void {
  ensureDirs();
  const slug = slugify(task.name || 'task');
  const filePath = path.join(TASKS_DIR, `${task.id}-${slug}.md`);
  const content = stringifyWithFrontmatter(task, body || task.body || '');
  fs.writeFileSync(filePath, content, 'utf8');
}

// Helper: parse run file
function parseRunFile(filePath: string): any {
  const content = fs.readFileSync(filePath, 'utf8');
  const { frontmatter, body } = parseFrontmatter(content);
  return { ...frontmatter, body };
}

// Helper: write run file
function writeRunFile(run: any, body?: string): void {
  ensureDirs();
  const taskDir = path.join(RUNS_DIR, String(run.task_id));
  if (!fs.existsSync(taskDir)) {
    fs.mkdirSync(taskDir, { recursive: true });
  }
  const filePath = path.join(taskDir, `${run.run_id}.md`);
  const content = stringifyWithFrontmatter(run, body || run.body || '');
  fs.writeFileSync(filePath, content, 'utf8');
}

// Helper: get next available task ID
function getNextTaskId(): number {
  ensureDirs();
  const files = fs.readdirSync(TASKS_DIR).filter(f => f.endsWith('.md') && f !== '_config.md');
  let maxId = 0;
  for (const f of files) {
    const match = f.match(/^(\d+)-/);
    if (match) {
      const id = parseInt(match[1], 10);
      if (id > maxId) maxId = id;
    }
  }
  return maxId + 1;
}

// Helper: parse priority for sorting
function priorityValue(priority: string): number {
  switch (priority) {
    case 'critical': return 4;
    case 'high': return 3;
    case 'medium': return 2;
    case 'low': return 1;
    default: return 0;
  }
}

// Helper: calculate overdue score (replicates SQL logic)
function overdueScore(task: any): number {
  if (!task.last_run) return 999999;
  const now = Date.now();
  const lastRun = new Date(task.last_run).getTime();
  const runsPerDay = task.runs_per_day || 1.0;
  const secondsPerRun = 86400.0 / runsPerDay;
  const secondsSinceLastRun = (now - lastRun) / 1000;
  return secondsSinceLastRun / secondsPerRun;
}

// Exported Functions

export function reconcileStuckTasks(): void {
  ensureDirs();
  
  // Find and reset stuck tasks (status = 'in_progress')
  const files = fs.readdirSync(TASKS_DIR).filter(f => f.endsWith('.md') && f !== '_config.md');
  const stuckCount = files.filter(f => {
    const task = parseTaskFile(path.join(TASKS_DIR, f));
    return task.status === 'in_progress';
  }).length;
  
  if (stuckCount > 0) {
    for (const f of files) {
      const task = parseTaskFile(path.join(TASKS_DIR, f));
      if (task.status === 'in_progress') {
        task.status = 'up_next';
        task.updated = new Date().toISOString();
        writeTaskFile(task);
      }
    }
  }
  
  // Find and mark orphaned runs (status = 'running')
  let orphanedCount = 0;
  const runDirs = fs.readdirSync(RUNS_DIR);
  for (const taskDir of runDirs) {
    const taskRunDir = path.join(RUNS_DIR, taskDir);
    if (!fs.statSync(taskRunDir).isDirectory()) continue;
    const runFiles = fs.readdirSync(taskRunDir).filter(f => f.endsWith('.md'));
    for (const rf of runFiles) {
      const run = parseRunFile(path.join(taskRunDir, rf));
      if (run.status === 'running') {
        run.status = 'failed';
        run.completed = new Date().toISOString();
        run.body = run.body || 'Interrupted by gateway restart';
        writeRunFile(run);
        orphanedCount++;
      }
    }
  }
  
  console.log(`[autonomy] reconciled ${stuckCount} stuck tasks, ${orphanedCount} orphaned runs`);
}

export async function getRuns(taskId: number, limit = 20): Promise<any[]> {
  ensureDirs();
  const taskRunDir = path.join(RUNS_DIR, String(taskId));
  if (!fs.existsSync(taskRunDir)) return [];
  
  const files = fs.readdirSync(taskRunDir).filter(f => f.endsWith('.md'));
  const runs = files.map(f => parseRunFile(path.join(taskRunDir, f)));
  
  // Sort by started_at desc
  runs.sort((a, b) => {
    const aTime = a.started_at ? new Date(a.started_at).getTime() : 0;
    const bTime = b.started_at ? new Date(b.started_at).getTime() : 0;
    return bTime - aTime;
  });
  
  return runs.slice(0, limit);
}

export async function getTasks(): Promise<any[]> {
  ensureDirs();
  const files = fs.readdirSync(TASKS_DIR).filter(f => f.endsWith('.md') && f !== '_config.md');
  const tasks = files.map(f => parseTaskFile(path.join(TASKS_DIR, f)));
  
  // Sort by overdue priority (replicate SQL ORDER BY)
  tasks.sort((a, b) => {
    const overdueA = overdueScore(a);
    const overdueB = overdueScore(b);
    // Primary: overdue desc
    if (overdueA !== overdueB) return overdueB - overdueA;
    // Secondary: priority desc
    const priA = priorityValue(a.priority);
    const priB = priorityValue(b.priority);
    if (priA !== priB) return priB - priA;
    // Tertiary: next_run asc
    if (a.next_run && b.next_run) {
      const aTime = new Date(a.next_run).getTime();
      const bTime = new Date(b.next_run).getTime();
      if (aTime !== bTime) return aTime - bTime;
    }
    // Quaternary: id asc
    return (a.id || 0) - (b.id || 0);
  });
  
  return tasks;
}

export async function getTask(id: number): Promise<any | null> {
  const filePath = findTaskFile(id);
  if (!filePath) return null;
  return parseTaskFile(filePath);
}

export async function writeTask(data: any): Promise<any> {
  ensureDirs();
  const now = new Date().toISOString();
  
  if (data.id) {
    // UPDATE existing task
    const filePath = findTaskFile(data.id);
    if (!filePath) {
      throw new Error(`Task ${data.id} not found`);
    }
    
    const existing = parseTaskFile(filePath);
    // Merge updates
    for (const key of Object.keys(data)) {
      if (key !== 'id' && key !== 'body') {
        existing[key] = data[key];
      }
    }
    existing.updated = now;
    
    // Handle activity_note
    if (data.activity_note) {
      // Append to body's Activity Log section
      let body = existing.body || '';
      if (!body.includes('## Activity Log')) {
        body += '\n\n## Activity Log\n';
      }
      const logLine = `\n- **${now.slice(0, 16).replace('T', ' ')}** — ${data.activity_note}`;
      body += logLine;
      existing.body = body;
      
      // Also create a run file
      const run = {
        type: 'autonomy-run',
        task_id: data.id,
        run_id: Date.now(),
        status: 'success',
        duration_seconds: 0,
        started_at: now,
        completed_at: now,
        body: data.activity_note
      };
      writeRunFile(run);
    }
    
    writeTaskFile(existing);
    return existing;
  } else {
    // INSERT new task
    const newId = getNextTaskId();
    const task = {
      type: 'autonomy',
      id: newId,
      name: data.name || '',
      description: data.description || '',
      status: data.status || 'inbox',
      priority: data.priority || 'medium',
      frequency: data.frequency || null,
      scheduled_at: data.scheduled_at || null,
      next_run: data.next_run || null,
      auto_advance: data.auto_advance || false,
      tags: data.tags || [],
      runs_per_day: data.runs_per_day || null,
      depends_on: data.depends_on || null,
      pipeline: data.pipeline || null,
      agent_id: data.agent_id || null,
      skill_path: data.skill_path || null,
      model: data.model || null,
      timeout_seconds: data.timeout_seconds || null,
      cron_id: data.cron_id || null,
      last_run: data.last_run || null,
      pipeline_mode: data.pipeline_mode || false,
      notify_on_complete: data.notify_on_complete !== false,
      max_retries: data.max_retries || null,
      preferred_hours: data.preferred_hours || null,
      preemptible: data.preemptible || false,
      created: now,
      updated: now,
      body: data.description || ''
    };
    writeTaskFile(task);
    return task;
  }
}

export async function deleteTask(id: number): Promise<any> {
  ensureDirs();
  const filePath = findTaskFile(id);
  if (filePath) {
    fs.unlinkSync(filePath);
  }
  
  // Delete runs directory
  const taskRunDir = path.join(RUNS_DIR, String(id));
  if (fs.existsSync(taskRunDir)) {
    fs.rmSync(taskRunDir, { recursive: true });
  }
  
  return { success: true, id };
}

export async function runTask(id: number): Promise<any> {
  const task = await getTask(id);
  if (!task) throw new Error(`Task ${id} not found`);
  
  const now = new Date().toISOString();
  
  // Create run file with status='running'
  const run = {
    type: 'autonomy-run',
    task_id: id,
    run_id: Date.now(),
    status: 'running',
    duration_seconds: null,
    started_at: now,
    completed_at: null,
    body: ''
  };
  writeRunFile(run);
  
  return {
    run_id: run.run_id,
    task_id: id,
    status: 'triggered',
    summary: `Task '${task.name}' triggered for execution`,
  };
}

export async function getConfig(key: string): Promise<string | null> {
  ensureDirs();
  if (!fs.existsSync(CONFIG_FILE)) return null;
  
  const content = fs.readFileSync(CONFIG_FILE, 'utf8');
  const { frontmatter } = parseFrontmatter(content);
  return frontmatter[key] || null;
}

export async function setConfig(key: string, value: string): Promise<void> {
  ensureDirs();
  let frontmatter: any = {};
  
  if (fs.existsSync(CONFIG_FILE)) {
    const content = fs.readFileSync(CONFIG_FILE, 'utf8');
    const parsed = parseFrontmatter(content);
    frontmatter = parsed.frontmatter;
  }
  
  frontmatter[key] = value;
  fs.writeFileSync(CONFIG_FILE, stringifyWithFrontmatter(frontmatter, ''), 'utf8');
}

export async function getAllConfig(): Promise<Record<string, string>> {
  ensureDirs();
  if (!fs.existsSync(CONFIG_FILE)) return {};
  
  const content = fs.readFileSync(CONFIG_FILE, 'utf8');
  const { frontmatter } = parseFrontmatter(content);
  return frontmatter;
}

/** Get all tasks currently in_progress (used by idler polling in index.ts). */
export function getInProgressTasks(): any[] {
  ensureDirs();
  const files = fs.readdirSync(TASKS_DIR).filter(f => f.endsWith('.md') && f !== '_config.md');
  const tasks: any[] = [];
  for (const f of files) {
    const task = parseTaskFile(path.join(TASKS_DIR, f));
    if (task.status === 'in_progress') {
      tasks.push(task);
    }
  }
  return tasks;
}

/** Mark the active run for a task as completed. */
export function completeActiveRun(taskId: number, status: string = 'success', summary?: string): void {
  ensureDirs();
  const taskRunDir = path.join(RUNS_DIR, String(taskId));
  if (!fs.existsSync(taskRunDir)) return;
  
  // Find most recent run without completed_at
  const files = fs.readdirSync(taskRunDir).filter(f => f.endsWith('.md'));
  let activeRun: any | null = null;
  let activeRunFile: string | null = null;
  
  for (const f of files) {
    const run = parseRunFile(path.join(taskRunDir, f));
    if (!run.completed_at) {
      if (!activeRun || new Date(run.started_at || 0).getTime() > new Date(activeRun.started_at || 0).getTime()) {
        activeRun = run;
        activeRunFile = f;
      }
    }
  }
  
  if (!activeRun || !activeRunFile) return;
  
  const now = new Date().toISOString();
  const startedAt = activeRun.started_at ? new Date(activeRun.started_at).getTime() : Date.now();
  const durationSeconds = Math.round((Date.now() - startedAt) / 1000);
  
  activeRun.status = status;
  activeRun.completed_at = now;
  activeRun.duration_seconds = durationSeconds;
  if (summary) activeRun.body = summary;
  
  writeRunFile(activeRun);
  
  // Update task's last_run on success
  const task = getTask(taskId);
  if (task) {
    if (status === 'success') {
      task.last_run = now;
    }
    task.updated = now;
    writeTaskFile(task);
  }
}

/** Mark a task as complete — move from in_progress to done (or up_next for recurring). */
export function completeTask(taskId: number): void {
  const task = getTask(taskId);
  if (!task) return;
  
  const now = new Date().toISOString();
  
  if (task.frequency && task.frequency !== 'one-time') {
    task.status = 'up_next';
  } else if (task.pipeline_mode) {
    task.status = 'in_review';
  } else {
    task.status = 'done';
  }
  
  task.updated = now;
  writeTaskFile(task);
}
