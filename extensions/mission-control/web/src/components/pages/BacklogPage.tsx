import { useEffect, useState, useCallback, useRef, type DragEvent } from "react";
import {
  LayoutGrid,
  AlertTriangle,
  Bot,
  Flag,
  X,
  Trash2,
  Save,
  Plus,
  Search,
} from "lucide-react";
import { api, type BacklogBoard, type BacklogTask } from "../../api";

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
  high: "text-red-400",
  medium: "text-amber-400",
  low: "text-sky-400",
  none: "text-slate-500",
};

const PRIORITIES = ["none", "low", "medium", "high"];

// ── Task Detail Modal ───────────────────────────────────────────────────

function TaskModal({
  task,
  onClose,
  onSave,
  onDelete,
  onCreate,
  defaultBoardId,
}: {
  task: BacklogTask | null;
  onClose: () => void;
  onSave: (id: number, updates: Record<string, any>) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
  onCreate: (data: Record<string, any>) => Promise<void>;
  defaultBoardId: number | null;
}) {
  const isCreate = !task;
  const [name, setName] = useState(task?.name || "");
  const [description, setDescription] = useState(task?.description || "");
  const [status, setStatus] = useState(task?.status || "inbox");
  const [priority, setPriority] = useState(task?.priority || "none");
  const [blocked, setBlocked] = useState(task?.blocked || false);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const backdropRef = useRef<HTMLDivElement>(null);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === backdropRef.current) onClose();
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      if (isCreate) {
        const data: Record<string, any> = { name, description, status, priority };
        if (defaultBoardId) data.board_id = defaultBoardId;
        await onCreate(data);
      } else {
        await onSave(task.id, { name, description, status, priority, blocked });
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
      <div className="bg-surface-1 rounded-xl border border-surface-3/50 w-full max-w-lg max-h-[80vh] flex flex-col shadow-2xl">
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
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Name */}
          <div>
            <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">
              Title
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={isCreate ? "What needs to be done?" : undefined}
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
              rows={4}
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
                    {p.charAt(0).toUpperCase() + p.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Blocked toggle */}
          {!isCreate && (
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={blocked}
                onChange={(e) => setBlocked(e.target.checked)}
                className="rounded border-surface-3 bg-surface-2 text-brand-500 focus:ring-brand-500/30"
              />
              <span className="text-xs text-slate-300">Blocked</span>
            </label>
          )}

          {/* Metadata */}
          {!isCreate && (
            <div className="grid grid-cols-2 gap-3 text-[10px] text-slate-500 pt-2 border-t border-surface-3/30">
              <div>
                Created: {new Date(task.created_at).toLocaleDateString()}
              </div>
              <div>
                Updated: {new Date(task.updated_at).toLocaleDateString()}
              </div>
              {task.assigned_to_agent && (
                <div className="col-span-2 flex items-center gap-1 text-brand-400">
                  <Bot className="w-3 h-3" />
                  Assigned to agent
                </div>
              )}
            </div>
          )}
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
  task: BacklogTask;
  onClick: (task: BacklogTask) => void;
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
          {task.blocked && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-red-400 bg-red-400/10 px-1.5 py-0.5 rounded">
              <AlertTriangle className="w-2.5 h-2.5" />
              blocked
            </span>
          )}
          {task.assigned_to_agent && (
            <span className="inline-flex items-center gap-0.5 text-[10px] text-brand-400 bg-brand-400/10 px-1.5 py-0.5 rounded">
              <Bot className="w-2.5 h-2.5" />
              assigned
            </span>
          )}
          {task.tags.map((tag) => (
            <span
              key={tag}
              className="text-[10px] text-slate-500 bg-surface-1 px-1.5 py-0.5 rounded"
            >
              {tag}
            </span>
          ))}
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
  tasks: BacklogTask[];
  onDrop: (taskId: number, newStatus: string, insertIndex: number) => void;
  onClickTask: (task: BacklogTask) => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [insertAt, setInsertAt] = useState<{ taskId: number; half: "top" | "bottom" } | null>(null);

  const handleDragOverColumn = (e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    setDragOver(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    // Only clear if leaving the column itself (not entering a child)
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

    // Calculate the insert index
    let insertIndex = tasks.length; // default: append at end
    if (insertAt) {
      const targetIdx = tasks.findIndex((t) => t.id === insertAt.taskId);
      if (targetIdx !== -1) {
        insertIndex = insertAt.half === "top" ? targetIdx : targetIdx + 1;
      }
    }

    setInsertAt(null);
    onDrop(draggedId, status, insertIndex);
  };

  // Figure out which card gets which indicator
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

export default function BacklogPage() {
  const [boards, setBoards] = useState<BacklogBoard[]>([]);
  const [tasks, setTasks] = useState<BacklogTask[]>([]);
  const [activeBoard, setActiveBoard] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [editingTask, setEditingTask] = useState<BacklogTask | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const loadData = useCallback(async () => {
    try {
      const [boardsData, tasksData] = await Promise.all([
        api.backlogBoards(),
        api.backlogTasks(activeBoard ? { board_id: String(activeBoard) } : undefined),
      ]);
      setBoards(boardsData);
      setTasks(tasksData);
      if (!activeBoard && boardsData.length > 0) {
        setActiveBoard(boardsData[0].id);
      }
    } catch (err) {
      console.error("Backlog load failed:", err);
    } finally {
      setLoading(false);
    }
  }, [activeBoard]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 15_000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleDrop = async (taskId: number, newStatus: string, insertIndex: number) => {
    const task = tasks.find((t) => t.id === taskId);
    if (!task) return;

    const sameColumn = task.status === newStatus;

    // Get current column tasks (excluding the dragged task)
    const columnTasks = filteredTasks
      .filter((t) => t.status === newStatus && t.id !== taskId)
      .sort((a, b) => a.position - b.position);

    // If same column and same position, skip
    if (sameColumn) {
      const currentIdx = filteredTasks
        .filter((t) => t.status === newStatus)
        .sort((a, b) => a.position - b.position)
        .findIndex((t) => t.id === taskId);
      if (currentIdx === insertIndex || currentIdx === insertIndex - 1) return;
    }

    // Calculate new position value
    // Position between surrounding tasks or at boundaries
    let newPosition: number;
    const adjustedIndex = sameColumn
      ? Math.min(insertIndex, columnTasks.length)
      : Math.min(insertIndex, columnTasks.length);

    if (columnTasks.length === 0) {
      newPosition = 1000;
    } else if (adjustedIndex === 0) {
      newPosition = columnTasks[0].position - 1000;
    } else if (adjustedIndex >= columnTasks.length) {
      newPosition = columnTasks[columnTasks.length - 1].position + 1000;
    } else {
      newPosition = Math.floor(
        (columnTasks[adjustedIndex - 1].position + columnTasks[adjustedIndex].position) / 2,
      );
    }

    // Optimistic update
    setTasks((prev) =>
      prev.map((t) =>
        t.id === taskId ? { ...t, status: newStatus, position: newPosition } : t,
      ),
    );

    try {
      const updates: Record<string, any> = { position: newPosition };
      if (!sameColumn) updates.status = newStatus;
      await api.backlogUpdateTask(taskId, updates);
    } catch (err) {
      console.error("Move failed:", err);
      loadData();
    }
  };

  const handleSave = async (id: number, updates: Record<string, any>) => {
    await api.backlogUpdateTask(id, updates);
    // Optimistic update
    setTasks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...updates } : t)),
    );
  };

  const handleDelete = async (id: number) => {
    await api.backlogDeleteTask(id);
    setTasks((prev) => prev.filter((t) => t.id !== id));
  };

  const handleCreate = async (data: Record<string, any>) => {
    await api.backlogCreateTask(data as any);
    loadData();
  };

  const handleClickTask = (task: BacklogTask) => {
    setEditingTask(task);
  };

  const filteredTasks = tasks
    .filter((t) => !activeBoard || t.board_id === activeBoard)
    .filter((t) => {
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      return (
        t.name.toLowerCase().includes(q) ||
        (t.description ?? "").toLowerCase().includes(q)
      );
    });

  const tasksByStatus = STATUSES.reduce(
    (acc, status) => {
      acc[status] = filteredTasks
        .filter((t) => t.status === status)
        .sort((a, b) => a.position - b.position);
      return acc;
    },
    {} as Record<string, BacklogTask[]>,
  );

  return (
    <div className="p-6 flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4 flex-shrink-0">
        <LayoutGrid className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Backlog</h2>

        {/* Board tabs */}
        <div className="flex gap-1 ml-4">
          {boards.map((board) => (
            <button
              key={board.id}
              onClick={() => setActiveBoard(board.id)}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                activeBoard === board.id
                  ? "bg-brand-600/15 text-brand-400 font-medium"
                  : "text-slate-400 hover:text-slate-200 hover:bg-surface-2"
              }`}
            >
              {board.icon} {board.name}
              <span className="ml-1.5 text-slate-500">{board.tasks_count}</span>
            </button>
          ))}
        </div>

        {/* Search input */}
        <div className="ml-auto relative flex items-center">
          <Search className="absolute left-2.5 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search tasks…"
            className="pl-8 pr-3 py-1.5 text-xs bg-surface-2 text-slate-200 rounded-lg border border-surface-3/50 outline-none focus:border-brand-500/50 w-48 placeholder:text-slate-500"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 text-slate-500 hover:text-slate-300 transition-colors"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>

        <button
          onClick={() => setShowCreateModal(true)}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs bg-brand-600 hover:bg-brand-500 text-white rounded-lg transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          New Task
        </button>

        <span className="text-xs text-slate-500">
          {filteredTasks.length} tasks — drag to move
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
          defaultBoardId={activeBoard}
        />
      )}
    </div>
  );
}
