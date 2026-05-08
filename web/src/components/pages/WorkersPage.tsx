import { useEffect, useState, useCallback } from "react";
import { useReportMcFocus, usePendingFocusFor } from "../../contexts/McUiContext";
import {
  Workflow, Pause, Play, RefreshCw, CheckCircle2, XCircle, Clock, Loader2,
  ArrowUpCircle, Trash2, FileText, FolderOpen,
} from "lucide-react";
import { api } from "../../api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";

interface Status {
  initialized: boolean;
  workers_enabled?: boolean;
  pool?: {
    running: boolean;
    paused: boolean;
    slots: number;
    in_flight: Record<string, { source: string; kind: string; started_at: string }>;
    in_flight_count: number;
  };
  depth?: Record<string, Record<string, number>>;
  sources?: Array<{
    name: string;
    enabled: boolean;
    interval_seconds?: number;
    max_inflight?: number;
    depth?: Record<string, number>;
  }>;
}

interface QueueItem {
  id: number;
  source: string;
  kind: string;
  priority: number;
  state: string;
  attempts: number;
  enqueued_at: string;
  claimed_by?: string | null;
  error?: string | null;
}

interface RunRow {
  run_id: string;
  source: string;
  status: string;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
  summary: string;
  task_id?: string | null;
}

const STATE_COLORS: Record<string, string> = {
  queued: "bg-sky-500/20 text-sky-300 border-sky-500/30",
  claimed: "bg-violet-500/20 text-violet-300 border-violet-500/30",
  running: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  completed: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  failed: "bg-rose-500/20 text-rose-300 border-rose-500/30",
  poisoned: "bg-red-600/30 text-red-200 border-red-600/40",
};

export default function WorkersPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [items, setItems] = useState<QueueItem[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [tab, setTab] = useState<"queue" | "runs" | "sources" | "review">("queue");
  const [stateFilter, setStateFilter] = useState<string>("");
  const [sourceFilter, setSourceFilter] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useReportMcFocus(
    "workers",
    sourceFilter ? { kind: "source", id: sourceFilter } : null,
  );

  const pendingFocus = usePendingFocusFor("workers");
  useEffect(() => {
    if (pendingFocus) setSourceFilter(pendingFocus);
  }, [pendingFocus]);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [s, q, r] = await Promise.all([
        api.workersStatus(),
        api.workersQueue({ state: stateFilter || undefined, source: sourceFilter || undefined, limit: 100 }),
        api.workersRuns({ source: sourceFilter || undefined, limit: 50 }),
      ]);
      setStatus(s);
      setItems(q.items || []);
      setRuns(r.runs || []);
    } finally {
      setLoading(false);
    }
  }, [stateFilter, sourceFilter]);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  }, [refresh]);

  const totalDepth = status?.depth
    ? Object.values(status.depth).reduce(
        (sum, s) => sum + Object.values(s).reduce((a, b) => a + b, 0),
        0,
      )
    : 0;

  const pausedLabel = status?.pool?.paused ? "Paused" : status?.pool?.running ? "Running" : "Stopped";

  return (
    <div className="p-6 text-foreground">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Workflow className="w-6 h-6 text-sky-400" />
          <div>
            <h1 className="text-xl font-bold">Workers</h1>
            <p className="text-sm text-muted-foreground">
              Unified work queue — {pausedLabel} · {status?.pool?.slots ?? 0} slots ·
              {" "}{status?.pool?.in_flight_count ?? 0} in flight · {totalDepth} total in queue
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => api.workersPause(!status?.pool?.paused).then(refresh)}
          >
            {status?.pool?.paused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            {status?.pool?.paused ? "Resume" : "Pause"}
          </Button>
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* In-flight strip */}
      {status?.pool?.in_flight && Object.keys(status.pool.in_flight).length > 0 && (
        <div className="mb-4 p-3 rounded bg-secondary/60 border border-border">
          <div className="text-xs uppercase text-muted-foreground mb-2">In flight ({status.pool.in_flight_count})</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(status.pool.in_flight).map(([id, info]) => (
              <span key={id}
                    className="text-xs px-2 py-1 rounded border border-amber-500/30 bg-amber-500/10 text-amber-200 flex items-center gap-1.5">
                <Loader2 className="w-3 h-3 animate-spin" />
                {info.source}/{info.kind} (#{id})
              </span>
            ))}
          </div>
        </div>
      )}

      <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="space-y-3">
        <TabsList>
          <TabsTrigger value="queue">Queue</TabsTrigger>
          <TabsTrigger value="runs">Recent Runs</TabsTrigger>
          <TabsTrigger value="sources">Sources</TabsTrigger>
          <TabsTrigger value="review">Review</TabsTrigger>
        </TabsList>

        {/* Filters — only meaningful on Queue / Runs tabs */}
        {(tab === "queue" || tab === "runs") && (
          <div className="flex flex-wrap gap-2 items-center text-sm">
            {tab === "queue" && (
              <Select value={stateFilter || "__all__"} onValueChange={(v) => setStateFilter(v === "__all__" ? "" : v)}>
                <SelectTrigger className="h-8 w-40"><SelectValue placeholder="all states" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">all states</SelectItem>
                  {Object.keys(STATE_COLORS).map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}
                </SelectContent>
              </Select>
            )}
            <Select value={sourceFilter || "__all__"} onValueChange={(v) => setSourceFilter(v === "__all__" ? "" : v)}>
              <SelectTrigger className="h-8 w-48"><SelectValue placeholder="all sources" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">all sources</SelectItem>
                {status?.sources?.map((s) => <SelectItem key={s.name} value={s.name}>{s.name}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
        )}

        <TabsContent value="queue"><QueueTable items={items} /></TabsContent>
        <TabsContent value="runs"><RunsTable runs={runs} /></TabsContent>
        <TabsContent value="sources"><SourcesTable sources={status?.sources || []} /></TabsContent>
        <TabsContent value="review"><ReviewPanel /></TabsContent>
      </Tabs>
    </div>
  );
}

function QueueTable({ items }: { items: QueueItem[] }) {
  if (!items.length) return <div className="text-muted-foreground italic p-4">no items</div>;
  return (
    <div className="rounded-md border border-border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>id</TableHead>
            <TableHead>source / kind</TableHead>
            <TableHead>prio</TableHead>
            <TableHead>state</TableHead>
            <TableHead>attempts</TableHead>
            <TableHead>enqueued</TableHead>
            <TableHead>error</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((it) => (
            <TableRow key={it.id}>
              <TableCell className="font-mono text-muted-foreground">#{it.id}</TableCell>
              <TableCell>
                <span className="font-medium">{it.source}</span>
                <span className="text-muted-foreground"> / {it.kind}</span>
              </TableCell>
              <TableCell>{it.priority}</TableCell>
              <TableCell>
                <Badge variant="outline" className={`${STATE_COLORS[it.state] || ""}`}>
                  {it.state}
                </Badge>
              </TableCell>
              <TableCell>{it.attempts}</TableCell>
              <TableCell className="text-muted-foreground">{formatAgo(it.enqueued_at)}</TableCell>
              <TableCell className="max-w-[260px] truncate text-xs">
                {it.error && it.state !== "completed" && (
                  <span className="text-rose-400" title={it.error}>{it.error}</span>
                )}
                {it.error && it.state === "completed" && it.attempts > 1 && (
                  <span
                    className="text-muted-foreground italic"
                    title={`Succeeded on retry. Prior error: ${it.error}`}
                  >
                    (recovered after {it.attempts} attempts)
                  </span>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function RunsTable({ runs }: { runs: RunRow[] }) {
  if (!runs.length) return <div className="text-muted-foreground italic p-4">no runs yet</div>;
  return (
    <div className="rounded-md border border-border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>status</TableHead>
            <TableHead>source</TableHead>
            <TableHead>task</TableHead>
            <TableHead>completed</TableHead>
            <TableHead>duration</TableHead>
            <TableHead>summary</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((r) => (
            <TableRow key={r.run_id}>
              <TableCell>
                {r.status === "success"
                  ? <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  : <XCircle className="w-4 h-4 text-rose-400" />}
              </TableCell>
              <TableCell>{r.source}</TableCell>
              <TableCell className="font-mono text-muted-foreground">{r.task_id || "—"}</TableCell>
              <TableCell className="text-muted-foreground">{formatAgo(r.completed_at)}</TableCell>
              <TableCell className="tabular-nums">
                <Clock className="w-3 h-3 inline mr-1 text-muted-foreground" />
                {r.duration_seconds?.toFixed(1) ?? "—"}s
              </TableCell>
              <TableCell className="text-foreground/90 max-w-[480px] truncate" title={r.summary}>
                {r.summary}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function SourcesTable({ sources }: { sources: Array<{ name: string; enabled: boolean; interval_seconds?: number; max_inflight?: number; depth?: Record<string, number> }> }) {
  if (!sources.length) return <div className="text-muted-foreground italic p-4">no sources configured</div>;
  return (
    <div className="rounded-md border border-border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>source</TableHead>
            <TableHead>enabled</TableHead>
            <TableHead>interval</TableHead>
            <TableHead>max inflight</TableHead>
            <TableHead>queued</TableHead>
            <TableHead>running</TableHead>
            <TableHead>completed</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sources.map((s) => (
            <TableRow key={s.name}>
              <TableCell className="font-medium">{s.name}</TableCell>
              <TableCell>
                <Badge
                  variant="outline"
                  className={s.enabled ? "border-emerald-500/40 text-emerald-300 bg-emerald-500/10" : "text-muted-foreground"}
                >
                  {s.enabled ? "on" : "off"}
                </Badge>
              </TableCell>
              <TableCell className="text-muted-foreground">{s.interval_seconds}s</TableCell>
              <TableCell className="text-muted-foreground">{s.max_inflight ?? "—"}</TableCell>
              <TableCell>{s.depth?.queued ?? 0}</TableCell>
              <TableCell>{(s.depth?.claimed ?? 0) + (s.depth?.running ?? 0)}</TableCell>
              <TableCell className="text-muted-foreground">{s.depth?.completed ?? 0}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function formatAgo(iso: string): string {
  if (!iso) return "—";
  try {
    const now = Date.now();
    const then = new Date(iso).getTime();
    const delta = Math.max(0, now - then) / 1000;
    if (delta < 60) return `${Math.round(delta)}s ago`;
    if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
    return `${Math.round(delta / 86400)}d ago`;
  } catch {
    return iso;
  }
}

// ── Review tab ─────────────────────────────────────────────────────────────

interface PendingItem {
  path: string;
  source: string;
  date: string;
  filename: string;
  size_bytes: number;
  mtime: string;
  frontmatter: Record<string, any>;
  preview: string;
}

const DEFAULT_DEST: Record<string, string> = {
  "domain-research": "knowledge",
  "bench-mine": "lloyd/bench",
};

function ReviewPanel() {
  const [items, setItems] = useState<PendingItem[]>([]);
  const [sources, setSources] = useState<string[]>([]);
  const [sourceFilter, setSourceFilter] = useState<string>("");
  const [selected, setSelected] = useState<PendingItem | null>(null);
  const [detail, setDetail] = useState<{ frontmatter: Record<string, any>; body: string } | null>(null);
  const [destination, setDestination] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const d = await api.workersPending(sourceFilter || undefined, 300);
      setItems(d.items || []);
      setSources(d.sources || []);
    } finally {
      setLoading(false);
    }
  }, [sourceFilter]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    setDestination(DEFAULT_DEST[selected.source] || "");
    setError("");
    api.workersPendingRead(selected.path).then(d => {
      setDetail({ frontmatter: d.frontmatter || {}, body: d.body || "" });
    }).catch(e => setError(String(e)));
  }, [selected]);

  const promote = async () => {
    if (!selected) return;
    setBusy(true); setError("");
    try {
      await api.workersPendingPromote({
        path: selected.path,
        destination: destination || undefined,
      });
      setSelected(null);
      await refresh();
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const reject = async () => {
    if (!selected) return;
    if (!confirm(`Move to _rejected/?\n\n${selected.filename}`)) return;
    setBusy(true); setError("");
    try {
      await api.workersPendingReject(selected.path);
      setSelected(null);
      await refresh();
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const canPromoteWithoutDest = !!selected && !!DEFAULT_DEST[selected.source];

  return (
    <div className="flex gap-3" style={{ minHeight: "65vh" }}>
      {/* Left — item list */}
      <div className="w-96 flex-shrink-0 border border-border rounded-md bg-secondary/40 flex flex-col">
        <div className="p-2 border-b border-border flex gap-2 items-center">
          <Select
            value={sourceFilter || "__all__"}
            onValueChange={(v) => { setSourceFilter(v === "__all__" ? "" : v); setSelected(null); }}
          >
            <SelectTrigger className="h-8 flex-1"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">all sources ({items.length})</SelectItem>
              {sources.map(s => <SelectItem key={s} value={s}>{s}</SelectItem>)}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" onClick={refresh} disabled={loading} title="Refresh" className="h-8 w-8">
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
        <div className="overflow-y-auto flex-1">
          {items.length === 0 && (
            <div className="p-6 text-center text-muted-foreground italic text-sm">
              no pending artifacts
            </div>
          )}
          {items.map(it => {
            const active = selected?.path === it.path;
            const conf = it.frontmatter?.confidence;
            return (
              <button
                key={it.path}
                onClick={() => setSelected(it)}
                className={`w-full text-left px-3 py-2 border-b border-border hover:bg-secondary/60 ${active ? "bg-secondary/80" : ""}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-sky-400 font-medium">{it.source}</span>
                  <span className="text-xs text-muted-foreground">{formatAgo(it.mtime)}</span>
                </div>
                <div className="text-sm truncate mt-0.5" title={it.filename}>
                  {it.filename.replace(/\.md$/, "")}
                </div>
                <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                  {typeof conf === "number" && (
                    <span className={`px-1.5 py-0.5 rounded border ${
                      conf >= 0.7 ? "border-emerald-500/40 text-emerald-300" :
                      conf >= 0.4 ? "border-amber-500/40 text-amber-300" :
                      "border-rose-500/40 text-rose-300"
                    }`}>
                      {conf.toFixed(2)}
                    </span>
                  )}
                  <span className="truncate" title={it.preview}>{it.preview.slice(0, 60)}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Right — detail pane */}
      <div className="flex-1 border border-border rounded bg-secondary/40 flex flex-col min-w-0">
        {!selected && (
          <div className="flex-1 flex items-center justify-center text-muted-foreground italic">
            <div className="text-center">
              <FileText className="w-8 h-8 mx-auto mb-2 text-muted-foreground/70" />
              select an artifact to review
            </div>
          </div>
        )}
        {selected && (
          <>
            <div className="p-3 border-b border-border flex-shrink-0">
              <div className="flex items-center gap-2 mb-1">
                <FolderOpen className="w-4 h-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground truncate" title={selected.path}>
                  {selected.path.replace("/home/alansrobotlab/obsidian/", "~/obsidian/")}
                </span>
              </div>
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-medium truncate">{selected.filename}</h3>
                <div className="flex gap-2 items-center flex-shrink-0">
                  <Input
                    type="text"
                    placeholder={canPromoteWithoutDest ? `dest: ${DEFAULT_DEST[selected.source]}` : "destination required"}
                    value={destination}
                    onChange={(e) => setDestination(e.target.value)}
                    className="h-8 w-56 text-xs"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={promote}
                    disabled={busy || (!canPromoteWithoutDest && !destination)}
                    title="Promote to canonical vault location"
                    className="border-emerald-500/40 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 hover:text-emerald-200"
                  >
                    <ArrowUpCircle className="w-3.5 h-3.5" />
                    Promote
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={reject}
                    disabled={busy}
                    title="Move to _rejected/"
                    className="border-rose-500/40 bg-rose-500/10 text-rose-300 hover:bg-rose-500/20 hover:text-rose-200"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Reject
                  </Button>
                </div>
              </div>
              {error && <div className="mt-2 text-xs text-rose-400">{error}</div>}
            </div>
            <div className="overflow-y-auto flex-1 p-4 font-mono text-sm">
              {detail?.frontmatter && Object.keys(detail.frontmatter).length > 0 && (
                <div className="mb-4 p-3 rounded bg-secondary/80 border border-border">
                  <div className="text-xs text-muted-foreground uppercase mb-2">frontmatter</div>
                  {Object.entries(detail.frontmatter).map(([k, v]) => (
                    <div key={k} className="flex gap-2 text-xs">
                      <span className="text-muted-foreground w-28">{k}</span>
                      <span className="text-foreground break-all">{
                        typeof v === "object" ? JSON.stringify(v) : String(v)
                      }</span>
                    </div>
                  ))}
                </div>
              )}
              <pre className="whitespace-pre-wrap text-foreground">{detail?.body ?? "loading…"}</pre>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
