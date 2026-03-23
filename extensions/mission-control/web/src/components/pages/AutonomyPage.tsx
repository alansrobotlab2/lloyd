import { useEffect, useState, useCallback, useRef, type DragEvent } from "react";
import {
  Lightbulb,
  AlertTriangle,
  Clock,
  Play,
  CheckCircle,
  X,
  Trash2,
  Save,
  Plus,
  Flag,
  Calendar,
  RefreshCw,
} from "lucide-react";
import { api, type AutonomyTask } from "../../api";

const STATUSES = ["inbox", "up_next", "in_progress", "in_review", "done"] as const;

const STATUS_LABELS: Record<string, string> = {
  inbox: "Inbox",
  up_next: "Up Next",
  in_progress: "In Progress",
  in_review: "In Review",
  done: "Done",
};

const STATUS_COLORS: Record<string, string> = {
  inbox: "bg-slate-500/20 text-slate-400 border-slate-500/30",
  up_next: "bg-sky-500/20 text-sky-400 border-sky-500/30",
  in_progress: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  in_review: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
  done: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
};

const PRIORITY_COLORS: Record<string, string> = {
  critical: "text-red-500 font-bold",
  high: "text-red-400",
  medium: "text-amber-400",
  low: "text-sky-400",
  background: "text-slate-500",
};

const PRIORITIES = ["background", "low", "medium", "high", "critical"];

// ── Helper Functions ────────────────────────────────────────────────────

function formatRunsPerDay(rpd: number | null): string {
  if (!rpd || rpd <= 0) return "manual";
  const intervalMin = 86400 / rpd / 60;
  if (intervalMin < 60) return `~${Math.round(intervalMin)}min`;
  if (intervalMin < 1440) return `~${Math.round(intervalMin / 60)}h`;
  return `~${Math.round(intervalMin / 1440)}d`;
}

function formatLastRun(lastRun: string | null): string {
  if (!lastRun) return "never";
  const diffMs = Date.now() - new Date(lastRun).getTime();
  const diffMin = diffMs / 60000;
  if (diffMin < 60) return `${Math.round(diffMin)}m ago`;
  if (diffMin < 1440) return `${Math.round(diffMin / 60)}h ago`;
  return `${Math.round(diffMin / 1440)}d ago`;
}

function calculateOverdueRatio(task: AutonomyTask): number | null {
  if (!task.runs_per_day || !task.last_run) return null;
  const now = Date.now();
  const lastRunMs = new Date(task.last_run).getTime();
  const intervalMs = (86400 * 1000) / task.runs_per_day;
  const elapsedMs = now - lastRunMs;
  return elapsedMs / intervalMs;
}



// ── Task Detail Modal ───────────────────────────────────────────────────

function TaskModal({
  task,
  onClose,
  onSave,
  onDelete,
  onCreate,
  allTasks = [],
}: {
  task: AutonomyTask | null;
  onClose: () => void;
  onSave: (id: number, updates: Record<string, any>) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
  onCreate: (data: Record<string, any>) => Promise<void>;
  allTasks?: AutonomyTask[];
}) {
  const isCreate = !task;
  const [name, setName] = useState(task?.name || "");
  const [description, setDescription] = useState(task?.description || "");
  const [status, setStatus] = useState(task?.status || "inbox");
  const [priority, setPriority] = useState(task?.priority || "medium");
  const [runsPerDay, setRunsPerDay] = useState(task?.runs_per_day?.toString() || "");
  const [pipeline, setPipeline] = useState(task?.pipeline || "");
  const [dependsOn, setDependsOn] = useState(task?.depends_on?.toString() || "");
  const [agentId, setAgentId] = useState(task?.agent_id || "");
  const [skillPath, setSkillPath] = useState(task?.skill_path || "");
  const [model, setModel] = useState(task?.model || "");
  const [timeoutSeconds, setTimeoutSeconds] = useState(task?.timeout_seconds?.toString() || "");
  const [scheduledAt, setScheduledAt] = useState(task?.scheduled_at || "");
  const [autoAdvance, setAutoAdvance] = useState(task?.auto_advance || false);
  const [runs, setRuns] = useState<Array<{id: number; started: string; completed: string | null; status: string; summary: string | null; activity_log: string | null; duration_seconds: number | null}>>([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const backdropRef = useRef<HTMLDivElement>(null);

  const loadRuns = useCallback(async () => {
    if (!task) return;
    setLogsLoading(true);
    try {
      const result = await api.autonomyRuns(task.id);
      const data = typeof result === "string" ? JSON.parse(result) : result;
      setRuns(data.runs || []);
    } catch (err) {
      console.error("Failed to load runs:", err);
    } finally {
      setLogsLoading(false);
    }
  }, [task]);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === backdropRef.current) onClose();
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const updates: Record<string, any> = {
        name, 
        description, 
        status, 
        priority, 
        scheduled_at: scheduledAt || null, 
        auto_advance: autoAdvance,
        runs_per_day: runsPerDay ? parseFloat(runsPerDay) : null,
        pipeline: pipeline || null,
        depends_on: dependsOn ? parseInt(dependsOn) : null,
        agent_id: agentId || null,
        skill_path: skillPath || null,
        model: model || null,
        timeout_seconds: timeoutSeconds ? parseInt(timeoutSeconds) : null,
      };
      if (isCreate) {
        await onCreate(updates);
      } else {
        await onSave(task.id, updates);
      }
      onClose();
    } catch (err) {
      console.error("Save failed:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!task) return;
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    setSaving(true);
    try {
      await onDelete(task.id);
      onClose();
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      ref={backdropRef}
      onClick={handleBackdropClick}
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
    >
      <div className="bg-surface-1 rounded-xl border border-surface-3/50 w-full max-w-[75vw] max-h-[85vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-surface-3/50">
          {isCreate ? (
            <span className="text-xs font-semibold text-brand-400">New Task</span>
          ) : (
            <>
              <span className="text-[10px] text-slate-500 font-mono">#{task.id}</span>
              <span
                className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded ${STATUS_COLORS[task.status]}`}
              >
                {STATUS_LABELS[task.status]}
              </span>
            </>
          )}
          <button
            onClick={onClose}
            className="ml-auto text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-hidden px-5 py-4">
          <div className="flex h-full gap-4">
            {/* Left column - Form fields */}
            <div className="flex-1 overflow-y-auto space-y-4" style={{ flex: '0 0 40%' }}>
          {/* Name */}
          <div>
            <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
              Title
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={isCreate ? "Task name" : undefined}
              autoFocus={isCreate}
              className="w-full bg-surface-2 text-sm text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
            />
          </div>

          {/* Description */}
          <div>
            <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              className="w-full bg-surface-2 text-sm text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50 resize-none"
            />
          </div>

          {/* Status + Priority row */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                Status
              </label>
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value)}
                className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
              >
                {STATUSES.map((s) => (
                  <option key={s} value={s} className="bg-surface-2">
                    {STATUS_LABELS[s]}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                Priority
              </label>
              <select
                value={priority}
                onChange={(e) => setPriority(e.target.value)}
                className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
              >
                {PRIORITIES.map((p) => (
                  <option key={p} value={p} className="bg-surface-2">
                    {p}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Scheduled At row (without Frequency) */}
          {!isCreate && (
            <div>
              <label className={`text-[10px] uppercase tracking-wider block mb-1 ${runsPerDay ? "text-slate-600" : "text-slate-500"}`}>
                Scheduled (PST) {runsPerDay ? "(using runs/day)" : ""}
              </label>
              <input
                type="text"
                value={runsPerDay ? "" : scheduledAt}
                onChange={(e) => setScheduledAt(e.target.value)}
                placeholder={runsPerDay ? "—" : "02:00:00"}
                disabled={!!runsPerDay}
                className={`w-full bg-surface-2 text-xs rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50 ${runsPerDay ? "text-slate-600 opacity-50 cursor-not-allowed" : "text-slate-200"}`}
              />
            </div>
          )}

          {/* New scheduler fields */}
          {!isCreate && (
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                  Runs per day
                </label>
                <input
                  type="number"
                  value={runsPerDay}
                  onChange={(e) => setRunsPerDay(e.target.value)}
                  placeholder="1.0"
                  className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
                />
              </div>
              <div>
                <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                  Timeout (sec)
                </label>
                <input
                  type="number"
                  value={timeoutSeconds}
                  onChange={(e) => setTimeoutSeconds(e.target.value)}
                  placeholder="1800"
                  className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
                />
              </div>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                Pipeline
              </label>
              <input
                type="text"
                value={pipeline}
                onChange={(e) => setPipeline(e.target.value)}
                placeholder="nightly"
                className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                Depends on
              </label>
              <select
                value={dependsOn}
                onChange={(e) => setDependsOn(e.target.value)}
                className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
              >
                <option value="">—</option>
                {allTasks.filter(t => t.id !== task?.id).map(t => (
                  <option key={t.id} value={t.id} className="bg-surface-2">
                    #{t.id} {t.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                Agent ID
              </label>
              <input
                type="text"
                value={agentId}
                onChange={(e) => setAgentId(e.target.value)}
                placeholder="memory"
                className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
                Skill path
              </label>
              <input
                type="text"
                value={skillPath}
                onChange={(e) => setSkillPath(e.target.value)}
                placeholder="skills/some-skill.ts"
                className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
              />
            </div>
          </div>
          <div>
            <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
              Model override
            </label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="llama3-70b"
              className="w-full bg-surface-2 text-xs text-slate-200 rounded-lg px-3 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50"
            />
          </div>

          {/* Auto-advance toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={autoAdvance}
              onChange={(e) => setAutoAdvance(e.target.checked)}
              className="rounded border-surface-3 bg-surface-2 text-brand-500 focus:ring-brand-500/30"
            />
            <span className="text-xs text-slate-300">Auto-advance to done (no approval needed)</span>
          </label>

            {/* Metadata */}
          {!isCreate && (
            <div className="grid grid-cols-2 gap-3 text-[10px] text-slate-500 pt-2 border-t border-surface-3/30">
              <div>
                Created: {new Date(task.created_at).toLocaleDateString()}
              </div>
              <div>
                Updated: {new Date(task.updated_at).toLocaleDateString()}
              </div>
              {task.next_run && (
                <div className="col-span-2 flex items-center gap-1 text-sky-400">
                  <Calendar className="w-3 h-3" />
                  Next run: {new Date(task.next_run).toLocaleString()}
                </div>
              )}
              {task.last_run && (
                <div className="col-span-2 flex items-center gap-1 text-slate-400">
                  <Clock className="w-3 h-3" />
                  Last run: {formatLastRun(task.last_run)}
                </div>
              )}
            </div>
          )}
            </div>

            {/* Right column - Logs panel */}
            {!isCreate && (
              <div className="flex-1 overflow-y-auto" style={{ flex: '0 0 60%' }}>
                <div className="h-full flex flex-col bg-surface-0 rounded-lg border border-surface-3/50">
                  <div className="flex items-center justify-between px-3 py-2 border-b border-surface-3/50">
                    <span className="text-xs font-semibold text-slate-300">Logs</span>
                    <button onClick={loadRuns} className="text-slate-400 hover:text-slate-200 transition-colors">
                      <RefreshCw className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto p-3">
                    {logsLoading ? (
                      <div className="text-[10px] text-slate-500 text-center py-4">Loading...</div>
                    ) : runs.length === 0 ? (
                      <div className="text-[10px] text-slate-500 text-center py-4">No runs yet.</div>
                    ) : (
                      <div className="space-y-2">
                        {runs.map((run) => {
                          const statusColor = run.status === 'success' ? 'text-emerald-400' :
                            run.status === 'failed' ? 'text-red-400' :
                            run.status === 'running' ? 'text-amber-400' :
                            run.status === 'timeout' ? 'text-orange-400' : 'text-slate-400';
                          return (
                            <div key={run.id} className="bg-black/30 rounded p-2 space-y-1">
                              <div className="flex items-center gap-2">
                                <span className={`text-[10px] font-semibold uppercase ${statusColor}`}>{run.status}</span>
                                <span className="text-[10px] text-slate-500">{new Date(run.started).toLocaleString()}</span>
                                {run.duration_seconds && (
                                  <span className="text-[10px] text-slate-600">{run.duration_seconds}s</span>
                                )}
                              </div>
                              {run.summary && (
                                <pre className="text-[10px] text-slate-300 font-mono whitespace-pre-wrap">{run.summary}</pre>
                              )}
                              {run.activity_log && (
                                <pre className="text-[10px] text-slate-400 font-mono whitespace-pre-wrap mt-1 border-t border-surface-3/30 pt-1">{run.activity_log}</pre>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center gap-2 px-5 py-3 border-t border-surface-3/50">
          {!isCreate && (
            <button
              onClick={handleDelete}
              disabled={saving}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-colors ${
                confirmDelete
                  ? "bg-red-600 text-white hover:bg-red-500"
                  : "text-red-400 hover:bg-red-400/10"
              }`}
            >
              <Trash2 className="w-3.5 h-3.5" />
              {confirmDelete ? "Confirm Delete" : "Delete"}
            </button>
          )}
          <div className="flex-1" />
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || !name.trim()}
            className="inline-flex items-center gap-1.5 px-4 py-1.5 text-xs bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white rounded-lg transition-colors"
          >
            {isCreate ? <Plus className="w-3.5 h-3.5" /> : <Save className="w-3.5 h-3.5" />}
            {saving ? (isCreate ? "Creating..." : "Saving...") : (isCreate ? "Create" : "Save")}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Task Card ───────────────────────────────────────────────────────────

function TaskCard({
  task,
  onClick,
  onDragOverCard,
  insertIndicator,
}: {
  task: AutonomyTask;
  onClick: (task: AutonomyTask) => void;
  onDragOverCard: (taskId: number, half: "top" | "bottom") => void;
  insertIndicator: "above" | "below" | null;
}) {
  const handleDragStart = (e: DragEvent) => {
    e.dataTransfer.setData("text/plain", String(task.id));
    e.dataTransfer.effectAllowed = "move";
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "move";
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const midY = rect.top + rect.height / 2;
    onDragOverCard(task.id, e.clientY < midY ? "top" : "bottom");
  };

  return (
    <div className="relative">
      {insertIndicator === "above" && (
        <div className="absolute -top-1.5 left-0 right-0 h-0.5 bg-brand-400 rounded-full z-10" />
      )}
      <div
        draggable
        onDragStart={handleDragStart}
        onDragOver={handleDragOver}
        onClick={() => onClick(task)}
        className="bg-surface-2 rounded-lg p-3 border border-surface-3/50 hover:border-brand-500/30 transition-colors cursor-pointer active:opacity-70"
      >
        <div className="flex items-start gap-2">
          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium text-slate-200 leading-snug">
              {task.name}
            </div>
            {task.description && (
              <p className="text-[10px] text-slate-500 mt-1 line-clamp-2">
                {task.description}
              </p>
            )}
          </div>
          <span className="text-[10px] text-slate-600 font-mono flex-shrink-0">
            #{task.id}
          </span>
        </div>

        <div className="flex items-center gap-1.5 mt-2 flex-wrap">
          {task.priority && task.priority !== "none" && (
            <span
              className={`inline-flex items-center gap-0.5 text-[10px] ${PRIORITY_COLORS[task.priority]}`}
            >
              <Flag className="w-2.5 h-2.5" />
              {task.priority}
            </span>
          )}
          {task.runs_per_day !== null && task.runs_per_day !== undefined && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-sky-400 bg-sky-400/10 px-1.5 py-0.5 rounded">
              <Clock className="w-2.5 h-2.5" />
              {formatRunsPerDay(task.runs_per_day)}
            </span>
          )}
          {task.pipeline && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-purple-400 bg-purple-400/10 px-1.5 py-0.5 rounded">
              {task.pipeline}
            </span>
          )}
          {task.depends_on && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded">
              → #{task.depends_on}
            </span>
          )}
          {task.agent_id && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-slate-400 bg-slate-400/10 px-1.5 py-0.5 rounded">
              {task.agent_id}
            </span>
          )}
          {task.scheduled_at && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded">
              <Calendar className="w-2.5 h-2.5" />
              {task.scheduled_at}
            </span>
          )}
          {task.status === "done" && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded">
              <CheckCircle className="w-2.5 h-2.5" />
              done
            </span>
          )}
          {task.status === "in_progress" && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded">
              <Play className="w-2.5 h-2.5" />
              running
            </span>
          )}
          {task.last_run && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-slate-500 bg-slate-500/10 px-1.5 py-0.5 rounded">
              <Clock className="w-2.5 h-2.5" />
              {formatLastRun(task.last_run)}
            </span>
          )}
          {task.status === "up_next" && (() => {
            const ratio = calculateOverdueRatio(task);
            if (ratio === null) return null;
            const color = ratio > 1 ? 'text-red-400 bg-red-400/10' : 'text-emerald-400 bg-emerald-400/10';
            return (
              <span className={`inline-flex items-center gap-0.5 text-[10px] ${color} px-1.5 py-0.5 rounded`}>
                {ratio > 1 ? '' : ''}{ratio.toFixed(1)}x {(ratio > 1 ? 'overdue' : 'ok')}
              </span>
            );
          })()}
        </div>
      </div>
      {insertIndicator === "below" && (
        <div className="absolute -bottom-1.5 left-0 right-0 h-0.5 bg-brand-400 rounded-full z-10" />
      )}
    </div>
  );
}

// ── Kanban Column ───────────────────────────────────────────────────────

function KanbanColumn({
  status,
  tasks,
  onDrop,
  onClickTask,
}: {
  status: string;
  tasks: AutonomyTask[];
  onDrop: (taskId: number, newStatus: string, insertIndex: number) => void;
  onClickTask: (task: AutonomyTask) => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [insertAt, setInsertAt] = useState<{ taskId: number; half: "top" | "bottom" } | null>(null);

  const handleDragOverColumn = (e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    setDragOver(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
      setDragOver(false);
      setInsertAt(null);
    }
  };

  const handleDragOverCard = (taskId: number, half: "top" | "bottom") => {
    setDragOver(true);
    setInsertAt({ taskId, half });
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const draggedId = parseInt(e.dataTransfer.getData("text/plain"), 10);
    if (isNaN(draggedId)) return;

    let insertIndex = tasks.length;
    if (insertAt) {
      const targetIdx = tasks.findIndex((t) => t.id === insertAt.taskId);
      if (targetIdx !== -1) {
        insertIndex = insertAt.half === "top" ? targetIdx : targetIdx + 1;
      }
    }

    setInsertAt(null);
    onDrop(draggedId, status, insertIndex);
  };

  const getIndicator = (taskId: number): "above" | "below" | null => {
    if (!insertAt) return null;
    if (insertAt.half === "top" && insertAt.taskId === taskId) return "above";
    if (insertAt.half === "bottom" && insertAt.taskId === taskId) return "below";
    return null;
  };

  return (
    <div
      onDragOver={handleDragOverColumn}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`flex flex-col min-w-[220px] flex-1 rounded-lg transition-colors ${
        dragOver ? "bg-brand-600/5 ring-1 ring-brand-500/30" : ""
      }`}
    >
      <div className="flex items-center gap-2 mb-2 px-1">
        <span
          className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded ${STATUS_COLORS[status]}`}
        >
          {STATUS_LABELS[status]}
        </span>
        <span className="text-[10px] text-slate-500">{tasks.length}</span>
      </div>

      <div className="space-y-2 overflow-y-auto flex-1 min-h-0 pr-1">
        {tasks.map((task) => (
          <TaskCard
            key={task.id}
            task={task}
            onClick={onClickTask}
            onDragOverCard={handleDragOverCard}
            insertIndicator={getIndicator(task.id)}
          />
        ))}
        {tasks.length === 0 && (
          <div
            className={`text-[10px] text-slate-600 text-center py-8 border border-dashed rounded-lg transition-colors ${
              dragOver ? "border-brand-500/40 text-brand-400" : "border-surface-3/30"
            }`}
          >
            {dragOver ? "Drop here" : "No tasks"}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────────────────────

export default function AutonomyPage() {
  const [tasks, setTasks] = useState<AutonomyTask[]>([]);
  const [allTasks, setAllTasks] = useState<AutonomyTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingTask, setEditingTask] = useState<AutonomyTask | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const result = await api.autonomyTasks();
      const data = typeof result === "string" ? JSON.parse(result) : result;
      const tasksList = data.tasks || [];
      setTasks(tasksList);
      setAllTasks(tasksList);
    } catch (err) {
      console.error("Autonomy load failed:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30_000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleDrop = async (taskId: number, newStatus: string, insertIndex: number) => {
    const task = tasks.find((t) => t.id === taskId);
    if (!task) return;

    // Optimistic update
    setTasks((prev) =>
      prev.map((t) =>
        t.id === taskId ? { ...t, status: newStatus } : t,
      ),
    );

    try {
      await api.autonomyWriteTask({ id: taskId, status: newStatus });
    } catch (err) {
      console.error("Move failed:", err);
      loadData();
    }
  };

  const handleSave = async (id: number, updates: Record<string, any>) => {
    await api.autonomyWriteTask({ id, ...updates });
    setTasks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...updates } : t)),
    );
  };

  const handleDelete = async (id: number) => {
    await api.autonomyDeleteTask(id);
    setTasks((prev) => prev.filter((t) => t.id !== id));
  };

  const handleCreate = async (data: Record<string, any>) => {
    await api.autonomyWriteTask(data);
    loadData();
  };

  const handleClickTask = (task: AutonomyTask) => {
    setEditingTask(task);
  };

  const tasksByStatus = STATUSES.reduce(
    (acc, status) => {
      acc[status] = tasks
        .filter((t) => t.status === status)
        .sort((a, b) => {
          // Sort by priority first, then by created_at
          const priorityOrder: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3, background: 4 };
          const priDiff = (priorityOrder[a.priority] ?? 2) - (priorityOrder[b.priority] ?? 2);
          if (priDiff !== 0) return priDiff;
          return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        });
      return acc;
    },
    {} as Record<string, AutonomyTask[]>,
  );

  return (
    <div className="p-6 flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4 flex-shrink-0">
        <Lightbulb className="w-5 h-5 text-amber-400" />
        <h2 className="text-lg font-semibold">Autonomy</h2>

        <button
          onClick={() => setShowCreateModal(true)}
          className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 text-xs bg-brand-600 hover:bg-brand-500 text-white rounded-lg transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          New Task
        </button>

        <span className="text-xs text-slate-500">
          {tasks.length} tasks — drag to move
        </span>
      </div>

      {/* Kanban board */}
      {loading ? (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
          Loading...
        </div>
      ) : (
        <div className="flex-1 flex gap-3 overflow-x-auto min-h-0">
          {STATUSES.map((status) => (
            <KanbanColumn
              key={status}
              status={status}
              tasks={tasksByStatus[status]}
              onDrop={handleDrop}
              onClickTask={handleClickTask}
            />
          ))}
        </div>
      )}

      {/* Task detail / create modal */}
      {(editingTask || showCreateModal) && (
        <TaskModal
          task={editingTask}
          onClose={() => { setEditingTask(null); setShowCreateModal(false); }}
          onSave={handleSave}
          onDelete={handleDelete}
          onCreate={handleCreate}
          allTasks={allTasks}
        />
      )}
    </div>
  );
}
