import { useEffect, useState, useCallback, useRef, type DragEvent } from "react";
import { useReportMcFocus, usePendingFocusFor, useMcUi } from "../../contexts/McUiContext";
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
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

const STATUSES = ["draft", "up_next", "in_progress", "done"] as const;

const STATUS_LABELS: Record<string, string> = {
  draft: "Draft",
  up_next: "Up Next",
  in_progress: "In Progress",
  done: "Done",
};

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-slate-500/20 text-muted-foreground border-slate-500/30",
  up_next: "bg-sky-500/20 text-sky-400 border-sky-500/30",
  in_progress: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  done: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
};

const PRIORITY_COLORS: Record<string, string> = {
  high: "text-red-400",
  medium: "text-amber-400",
  low: "text-sky-400",
  none: "text-muted-foreground",
};

const PRIORITIES = ["none", "low", "medium", "high"];

// ── Task Detail Modal ───────────────────────────────────────────────────

function TaskModal({
  task,
  boards,
  onClose,
  onSave,
  onDelete,
  onCreate,
  defaultBoard,
}: {
  task: BacklogTask | null;
  boards: BacklogBoard[];
  onClose: () => void;
  onSave: (id: number, updates: Record<string, any>) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
  onCreate: (data: Record<string, any>) => Promise<void>;
  defaultBoard: string | null;
}) {
  const isCreate = !task;
  const [name, setName] = useState(task?.name || "");
  // The row's `description` is a 300-character snippet (`description_snippet` on
  // the wire, aliased to `description` for now). Seeding the editor from it and
  // posting it back would overwrite the item: `task-update` replaces the *whole*
  // body with whatever `description` it is sent, and bodies here run to a median
  // of 4,607 bytes. So the editor's text arrives from `GET /api/backlog/task/{id}`
  // below, and `bodyReady` says whether it has. Until then the textarea holds the
  // snippet as a placeholder and Save is disabled — see handleSave.
  const [description, setDescription] = useState(task?.description_snippet || "");
  const [bodyReady, setBodyReady] = useState(isCreate);
  const [status, setStatus] = useState(task?.status || "draft");
  // `low` is the board default (2026-09-16): the unattended loop sorts on
  // this field, so a new item starts where it cannot jump the queue.
  const [priority, setPriority] = useState(task?.priority || "low");
  const [board, setBoard] = useState(task?.board || defaultBoard || "");
  const [blocked, setBlocked] = useState(task?.blocked || false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const backdropRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isCreate || !task) return;
    let alive = true;
    setBodyReady(false);
    api
      .backlogTask(task.id)
      .then((full) => {
        if (!alive) return;
        // A body that is not a string means this response did not come from the
        // detail route. Editing from it would seed the box with `undefined`, and
        // posting that with `force_body_replace` would empty the item — so
        // nothing becomes ready and the error says why.
        if (typeof full.description !== "string") {
          setError(`task ${task.id} returned no body; refusing to edit from a row`);
          return;
        }
        setDescription(full.description);
        setBodyReady(true);
      })
      .catch((err) => {
        // Stay un-ready: the alternative is editing against the snippet, which
        // is the data loss. The error is shown, and Save stays disabled, so the
        // user can close and retry rather than save over an unseen body.
        console.error("Full item fetch failed:", err);
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      alive = false;
    };
  }, [task?.id, isCreate]);

  // A Select whose value matches no option renders a blank trigger, so an
  // out-of-list board is carried in as its own option rather than showing the
  // task as belonging to nothing. It happens for real: `board_id` is 0 for a
  // board name the map missed, and the board list is 10s-cached upstream.
  const boardOptions = !board || boards.some((b) => b.name === board)
    ? boards
    : [...boards, { id: -1, name: board, icon: "\u{1F4CB}", color: "", tasks_count: 0 }];

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === backdropRef.current) onClose();
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      if (isCreate) {
        const data: Record<string, any> = { name, description, status, priority };
        if (board) data.board = board;
        await onCreate(data);
      } else {
        const updates: Record<string, any> = { name, status, priority, blocked };
        // Only a body that came from the detail route may be posted, and once it
        // has, posting it is exactly what the user is editing. `force_body_replace`
        // is the server's guard (item #1199): a shorter `description` is otherwise
        // ignored, because a snippet-shaped one is indistinguishable from a
        // truncated body by length alone.
        if (bodyReady) {
          updates.description = description;
          updates.force_body_replace = true;
        }
        // Only on a real change: the parent reads `"board" in updates` as the
        // signal that this save moved the task and the whole board has to be
        // refetched, and an ordinary title edit should not pay for that.
        if (board && board !== task.board) updates.board = board;
        await onSave(task.id, updates);
      }
      onClose();
    } catch (err) {
      console.error("Save failed:", err);
      setError(err instanceof Error ? err.message : String(err));
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
    setError(null);
    try {
      await onDelete(task.id);
      onClose();
    } catch (err) {
      console.error("Delete failed:", err);
      setError(err instanceof Error ? err.message : String(err));
      setConfirmDelete(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      ref={backdropRef}
      onClick={handleBackdropClick}
      className="absolute inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
    >
      <div className="bg-card rounded-xl border border-border/50 w-[90vw] h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-border/50">
          {isCreate ? (
            <span className="text-xs font-semibold text-primary">New Task</span>
          ) : (
            <>
              <span className="text-[10px] text-muted-foreground font-mono">#{task.id}</span>
              <span
                className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded ${STATUS_COLORS[task.status]}`}
              >
                {STATUS_LABELS[task.status]}
              </span>
            </>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="ml-auto h-7 w-7 text-muted-foreground hover:text-foreground"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Body */}
        <div className="flex-1 flex flex-col overflow-y-auto px-5 py-4">
          {/* Name */}
          <div className="mb-4">
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider block mb-1">
              Title
            </label>
            <Input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={isCreate ? "What needs to be done?" : undefined}
              autoFocus={isCreate}
              className="w-full"
            />
          </div>

          {/* Description */}
          <div className="flex-1 min-h-0 flex flex-col">
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider block mb-1">
              Description{!bodyReady && !isCreate && " — loading full text…"}
            </label>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              readOnly={!bodyReady}
              className="w-full flex-1 resize-none"
            />
          </div>
        </div>

        {/* Additional fields section */}
        <div className="px-5 py-3 border-t border-border/50 space-y-3">
          {/* Board + Status + Priority row */}
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="text-[10px] text-muted-foreground uppercase tracking-wider block mb-1">
                Board
              </label>
              <Select value={board} onValueChange={setBoard} disabled={boardOptions.length === 0}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder={boardOptions.length === 0 ? "No boards" : "Pick a board"} />
                </SelectTrigger>
                <SelectContent>
                  {boardOptions.map((b) => (
                    <SelectItem key={b.name} value={b.name}>
                      {b.icon} {b.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground uppercase tracking-wider block mb-1">
                Status
              </label>
              <Select value={status} onValueChange={setStatus}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {STATUSES.map((s) => (
                    <SelectItem key={s} value={s}>{STATUS_LABELS[s]}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground uppercase tracking-wider block mb-1">
                Priority
              </label>
              <Select value={priority} onValueChange={setPriority}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {PRIORITIES.map((p) => (
                    <SelectItem key={p} value={p}>
                      {p.charAt(0).toUpperCase() + p.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Blocked toggle */}
          {!isCreate && (
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={blocked}
                onChange={(e) => setBlocked(e.target.checked)}
                className="rounded border-border bg-secondary text-primary focus:ring-primary/30"
              />
              <span className="text-xs text-foreground/90">Blocked</span>
            </label>
          )}

          {/* Metadata */}
          {!isCreate && (
            <div className="grid grid-cols-2 gap-3 text-[10px] text-muted-foreground">
              <div>
                Created: {new Date(task.created_at).toLocaleDateString()}
              </div>
              <div>
                Updated: {new Date(task.updated_at).toLocaleDateString()}
              </div>
              {task.assigned_to_agent && (
                <div className="col-span-2 flex items-center gap-1 text-primary">
                  <Bot className="w-3 h-3" />
                  Assigned to agent
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        {error && (
          <div className="px-5 pt-3 -mb-1 text-[11px] text-destructive flex items-start gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-px" />
            <span className="min-w-0 break-words">{error}</span>
          </div>
        )}
        <div className="flex items-center gap-2 px-5 py-3 border-t border-border/50">
          {!isCreate && (
            <Button
              variant={confirmDelete ? "destructive" : "ghost"}
              size="sm"
              onClick={handleDelete}
              disabled={saving}
              className={confirmDelete ? "" : "text-destructive hover:bg-destructive/10 hover:text-destructive"}
            >
              <Trash2 className="w-3.5 h-3.5" />
              {confirmDelete ? "Confirm Delete" : "Delete"}
            </Button>
          )}
          <div className="flex-1" />
          <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
          <Button size="sm" onClick={handleSave} disabled={saving || !name.trim() || !bodyReady}>
            {isCreate ? <Plus className="w-3.5 h-3.5" /> : <Save className="w-3.5 h-3.5" />}
            {saving ? (isCreate ? "Creating..." : "Saving...") : (isCreate ? "Create" : "Save")}
          </Button>
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
        <div className="absolute -top-1.5 left-0 right-0 h-0.5 bg-primary rounded-full z-10" />
      )}
      <div
        draggable
        onDragStart={handleDragStart}
        onDragOver={handleDragOver}
        onClick={() => onClick(task)}
        className="bg-secondary rounded-lg p-3 border border-border/50 hover:border-primary/30 transition-colors cursor-pointer active:opacity-70"
      >
        <div className="flex items-start gap-2">
          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium text-foreground leading-snug">
              {task.name}
            </div>
            {task.description_snippet && (
              <p className="text-[10px] text-muted-foreground mt-1 line-clamp-2">
                {task.description_snippet}
              </p>
            )}
          </div>
          <span className="text-[10px] text-muted-foreground/70 font-mono flex-shrink-0">
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
            <span className="inline-flex items-center gap-0.5 text-[10px] text-primary bg-primary/10 px-1.5 py-0.5 rounded">
              <Bot className="w-2.5 h-2.5" />
              assigned
            </span>
          )}
          {/* `tags` is normalized to an array by app/backlog_tags.py, but the
              board renders whatever the vault holds and a hand-edited file can
              still put a scalar there. One such row used to throw
              `task.tags.map is not a function` out of render and blank the
              entire board — the failure that motivated the normalizer. */}
          {Array.isArray(task.tags) && task.tags.map((tag) => (
            <span
              key={tag}
              className="text-[10px] text-muted-foreground bg-card px-1.5 py-0.5 rounded"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>
      {insertIndicator === "below" && (
        <div className="absolute -bottom-1.5 left-0 right-0 h-0.5 bg-primary rounded-full z-10" />
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
        dragOver ? "bg-primary/5 ring-1 ring-primary/30" : ""
      }`}
    >
      <div className="flex items-center gap-2 mb-2 px-1">
        <span
          className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded ${STATUS_COLORS[status]}`}
        >
          {STATUS_LABELS[status]}
        </span>
        <span className="text-[10px] text-muted-foreground">{tasks.length}</span>
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
            className={`text-[10px] text-muted-foreground/70 text-center py-8 border border-dashed rounded-lg transition-colors ${
              dragOver ? "border-primary/40 text-primary" : "border-border/30"
            }`}
          >
            {dragOver ? "Drop here" : "No tasks"}
          </div>
        )}
      </div>
    </div>
  );
}

/** How many days of finished items the board shows by default (item #1213).
 *
 * Alan's ruling, 2026-09-17. A legibility number, not a performance one: #1199
 * already cut the payload from 9.2 MB to ~920 KB by moving bodies off the list
 * route. What it could not touch was the row *count* — 679 of 1,152 rows
 * (59 % of the payload) are `status: done`, and the Done column had no window
 * at all, so every item ever closed was one scroll away from forever. Seven
 * days keeps roughly 400 of those 679 visible; the window is where to look if
 * Done still reads as crowded.
 */
const DONE_WINDOW_DAYS = 7;

/** The `done_since` value for *this* request: today − DONE_WINDOW_DAYS.
 *
 * Called inside `loadData`, never hoisted to mount or module scope. The page
 * refetches on a 15-second interval that is not gated on tab visibility, so a
 * page left open across midnight would otherwise keep asking for the same
 * seven days forever — the window would freeze on the day the tab was opened
 * and quietly show an ageing set until someone reloaded.
 *
 * Local date parts, deliberately: `toISOString()` is UTC, and this box is
 * UTC−7, so slicing it would name a day up to seven hours out of step with the
 * calendar the person reading the board is looking at.
 */
function doneSinceDate(): string {
  const d = new Date();
  d.setDate(d.getDate() - DONE_WINDOW_DAYS);
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${month}-${day}`;
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
  // The query the server was last asked about, debounced from the box. Search is
  // a `?q=` on the list route rather than a scan of what is already in memory,
  // because the row carries a 300-character snippet and mid-body text is no
  // longer in memory to scan (item #1199). Debounced so typing does not fetch
  // per keystroke, and one server search over a cached corpus is cheaper than
  // the 8.3 MB `toLowerCase().includes()` sweep this replaces.
  const [searchParam, setSearchParam] = useState("");
  // The 'include all done' checkbox: ticked, the request drops `done_since` and
  // the Done column goes back to every item ever closed. Default unticked —
  // i.e. the window is on — because the window is the default Alan asked for and
  // a first paint showing 679 done rows is the thing being fixed (item #1213).
  const [includeAllDone, setIncludeAllDone] = useState(false);
  // Whether this frame's request should carry `done_since`: the two omissions —
  // the checkbox, and an active search — named once, so the request and the
  // checkbox's disabled state cannot drift into disagreeing about what "windowed"
  // means.
  const windowActive = !includeAllDone && !searchParam;
  // The board list as `loadData` last saw it, so a refetch can tell which
  // board the active id *used* to name before the ids renumbered.
  const boardsRef = useRef<BacklogBoard[]>([]);
  const activeBoardName = boards.find((b) => b.id === activeBoard)?.name ?? null;

  // Mirror current focus for the agent. Editing-task wins; otherwise
  // active board.
  useReportMcFocus(
    "backlog",
    editingTask
      ? { kind: "task", id: String(editingTask.id), label: editingTask.name }
      : activeBoard != null
        ? { kind: "board", id: String(activeBoard) }
        : null,
  );

  // Apply incoming focus from mc_navigate. Numeric ids that match a
  // task open it; otherwise treat as board id.
  const pendingFocus = usePendingFocusFor("backlog");
  useEffect(() => {
    if (!pendingFocus) return;
    const asNum = Number(pendingFocus);
    if (!Number.isFinite(asNum)) return;
    const task = tasks.find((t) => t.id === asNum);
    if (task) {
      setEditingTask(task);
      return;
    }
    // Not in the payload. Either it names a board, or it names an item the
    // window did not load — and the second case is new with #1213: the agent's
    // `mc_navigate(tab=backlog, focus_id=<id>)` used to resolve against every
    // row that exists, so an item closed more than 7 days ago would now fall
    // through and be read as a board id. Board first (an id that *is* a board
    // must stay a board switch), then ask the detail route, which takes an id
    // and knows nothing about windows. Only a failed fetch is a board id.
    if (boardsRef.current.some((b) => b.id === asNum)) {
      setActiveBoard(asNum);
      return;
    }
    let stale = false;
    api
      .backlogTask(asNum)
      .then((loaded) => {
        if (!stale) setEditingTask(loaded);
      })
      .catch(() => {
        if (!stale) setActiveBoard(asNum);
      });
    return () => {
      stale = true;
    };
  }, [pendingFocus, tasks]);

  // Apply agent-issued close_modal for the backlog tab.
  const { pendingCloseModal } = useMcUi();
  const lastClosedSeq = useRef(0);
  useEffect(() => {
    if (!pendingCloseModal || pendingCloseModal.tab !== "backlog") return;
    if (pendingCloseModal.seq === lastClosedSeq.current) return;
    lastClosedSeq.current = pendingCloseModal.seq;
    setEditingTask(null);
    setShowCreateModal(false);
  }, [pendingCloseModal]);

  const loadData = useCallback(async () => {
    try {
      // The board the user is standing on, by name, taken before the refetch.
      const previousName = boardsRef.current.find((b) => b.id === activeBoard)?.name;
      // `done_since` is computed here, in the request, for two reasons. It has
      // to move with the calendar (see `doneSinceDate`), and the two omissions
      // below are only expressible at the place the query is assembled: the
      // checkbox drops the parameter outright, and a search drops it too, so a
      // query for an item closed last month can still come back — the server
      // matches `?q=` against whole bodies, and cutting the row by date first
      // would make old done items permanently unfindable.
      const [boardsData, tasksData] = await Promise.all([
        api.backlogBoards(),
        api.backlogTasks({
          ...(activeBoard ? { board_id: String(activeBoard) } : {}),
          ...(searchParam ? { q: searchParam } : {}),
          ...(windowActive ? { done_since: doneSinceDate() } : {}),
        }),
      ]);
      boardsRef.current = boardsData;
      setBoards(boardsData);
      setTasks(tasksData);
      if (!activeBoard && boardsData.length > 0) {
        setActiveBoard(boardsData[0].id);
      } else if (previousName) {
        // Board ids are positional over the sorted board names, so a board
        // appearing or vanishing renumbers every board after it — and one
        // vanishes exactly when its last task is moved off, which the modal
        // now does in one click. Following the id would leave the user on a
        // tab that is quietly a different board. Follow the name; the id
        // change re-runs this callback with the correct filter.
        const sameBoard = boardsData.find((b) => b.name === previousName);
        if (sameBoard) {
          if (sameBoard.id !== activeBoard) setActiveBoard(sameBoard.id);
        } else if (boardsData.length > 0) {
          setActiveBoard(boardsData[0].id);
        }
      }
    } catch (err) {
      console.error("Backlog load failed:", err);
    } finally {
      setLoading(false);
    }
  }, [activeBoard, searchParam, includeAllDone]);

  useEffect(() => {
    const timer = setTimeout(() => setSearchParam(searchQuery.trim()), 250);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 15_000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleDrop = async (taskId: number, newStatus: string, insertIndex: number) => {
    const task = tasks.find((t) => t.id === taskId);
    if (!task) return;

    const sameColumn = task.status === newStatus;

    const columnTasks = filteredTasks
      .filter((t) => t.status === newStatus && t.id !== taskId)
      .sort((a, b) => a.position - b.position);

    if (sameColumn) {
      const currentIdx = filteredTasks
        .filter((t) => t.status === newStatus)
        .sort((a, b) => a.position - b.position)
        .findIndex((t) => t.id === taskId);
      if (currentIdx === insertIndex || currentIdx === insertIndex - 1) return;
    }

    let newPosition: number;
    const adjustedIndex = Math.min(insertIndex, columnTasks.length);

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
    // A board move changes which tab the task belongs to, every board's
    // task_count, and possibly the board *ids* themselves — moving the last
    // task off a board deletes that board, and ids are positional over the
    // sorted names. None of that is derivable from `updates`, which carries
    // the board as a name and no board_id at all, so merging it locally would
    // leave the card sitting on the board it just left. Refetch instead.
    if ("board" in updates) {
      await loadData();
      return;
    }
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

  // Board only. The text predicate that used to live here is gone: `?q=` matched
  // server-side over the whole body, and re-filtering the returned rows against a
  // 300-character snippet would discard exactly the mid-body matches the server
  // just found (item #1199).
  const filteredTasks = activeBoard
    ? tasks.filter((t) => t.board_id === activeBoard)
    : tasks;

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
        <LayoutGrid className="w-5 h-5 text-primary" />
        <h2 className="text-lg font-semibold text-foreground">Backlog</h2>

        {/* Board tabs */}
        <div className="flex gap-1 ml-4">
          {boards.map((board) => (
            <button
              key={board.id}
              onClick={() => setActiveBoard(board.id)}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                activeBoard === board.id
                  ? "bg-primary/15 text-primary font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary"
              }`}
            >
              {board.icon} {board.name}
              <span className="ml-1.5 text-muted-foreground">{board.tasks_count}</span>
            </button>
          ))}
        </div>

        {/* The done window (item #1213). Unticked, the request carries
            `done_since` = today − DONE_WINDOW_DAYS: 679 of 1,152 rows were
            `status: done` at filing and the Done column held every item ever
            closed. Ticking it drops the parameter, and since the server only
            ever cuts done rows, ticking can only add rows back.

            Disabled during a search rather than silently inert: a search
            already omits `done_since` — see `loadData` — so that a query for an
            item closed last month can still match, and a control that looks
            live here while changing nothing is worse than no control. */}
        <label
          className="flex items-center gap-1.5 text-xs text-muted-foreground select-none cursor-pointer whitespace-nowrap"
          title={`Done column shows the last ${DONE_WINDOW_DAYS} days. Ticking this asks the server for every closed item.`}
        >
          <input
            type="checkbox"
            checked={includeAllDone}
            disabled={!!searchParam}
            onChange={(e) => setIncludeAllDone(e.target.checked)}
            className="w-3.5 h-3.5 accent-primary disabled:opacity-40"
          />
          include all done
        </label>

        {/* Search input */}
        <div className="ml-auto relative">
          <Search className="w-3.5 h-3.5 text-muted-foreground absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <Input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter tasks..."
            className="h-8 w-56 pl-7 pr-7 text-xs"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>

        <Button size="sm" onClick={() => setShowCreateModal(true)}>
          <Plus className="w-3.5 h-3.5" />
          New Task
        </Button>

        <span className="text-xs text-muted-foreground">
          {filteredTasks.length} tasks — drag to move
        </span>
      </div>

      {/* Kanban board */}
      {loading ? (
        <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
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
          boards={boards}
          onClose={() => { setEditingTask(null); setShowCreateModal(false); }}
          onSave={handleSave}
          onDelete={handleDelete}
          onCreate={handleCreate}
          defaultBoard={activeBoardName}
        />
      )}
    </div>
  );
}
