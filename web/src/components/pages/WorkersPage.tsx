import { useEffect, useState, useCallback } from "react";
import { Workflow, Pause, Play, RefreshCw, CheckCircle2, XCircle, Clock, Loader2 } from "lucide-react";
import { api } from "../../api";

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
  const [tab, setTab] = useState<"queue" | "runs" | "sources">("queue");
  const [stateFilter, setStateFilter] = useState<string>("");
  const [sourceFilter, setSourceFilter] = useState<string>("");
  const [loading, setLoading] = useState(false);

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
    <div className="p-6 text-slate-200">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Workflow className="w-6 h-6 text-sky-400" />
          <div>
            <h1 className="text-xl font-bold">Workers</h1>
            <p className="text-sm text-slate-400">
              Unified work queue — {pausedLabel} · {status?.pool?.slots ?? 0} slots ·
              {" "}{status?.pool?.in_flight_count ?? 0} in flight · {totalDepth} total in queue
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => api.workersPause(!status?.pool?.paused).then(refresh)}
            className="px-3 py-1.5 text-sm rounded border border-slate-600 hover:border-sky-500 hover:text-sky-300 flex items-center gap-1.5"
          >
            {status?.pool?.paused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            {status?.pool?.paused ? "Resume" : "Pause"}
          </button>
          <button
            onClick={refresh}
            disabled={loading}
            className="px-3 py-1.5 text-sm rounded border border-slate-600 hover:border-sky-500 hover:text-sky-300 flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* In-flight strip */}
      {status?.pool?.in_flight && Object.keys(status.pool.in_flight).length > 0 && (
        <div className="mb-4 p-3 rounded bg-slate-900/60 border border-slate-700">
          <div className="text-xs uppercase text-slate-400 mb-2">In flight ({status.pool.in_flight_count})</div>
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

      {/* Tab strip */}
      <div className="flex items-center gap-1 mb-3 border-b border-slate-700">
        {(["queue", "runs", "sources"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm border-b-2 ${tab === t ? "border-sky-400 text-sky-300" : "border-transparent text-slate-400 hover:text-slate-200"}`}
          >
            {t === "queue" ? "Queue" : t === "runs" ? "Recent Runs" : "Sources"}
          </button>
        ))}
      </div>

      {/* Filters */}
      {tab !== "sources" && (
        <div className="flex flex-wrap gap-3 mb-3 items-center text-sm">
          {tab === "queue" && (
            <select value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}
                    className="bg-slate-900 border border-slate-700 rounded px-2 py-1">
              <option value="">all states</option>
              {Object.keys(STATE_COLORS).map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          )}
          <select value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded px-2 py-1">
            <option value="">all sources</option>
            {status?.sources?.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
          </select>
        </div>
      )}

      {tab === "queue" && <QueueTable items={items} />}
      {tab === "runs" && <RunsTable runs={runs} />}
      {tab === "sources" && <SourcesTable sources={status?.sources || []} />}
    </div>
  );
}

function QueueTable({ items }: { items: QueueItem[] }) {
  if (!items.length) return <div className="text-slate-500 italic p-4">no items</div>;
  return (
    <div className="overflow-x-auto rounded border border-slate-700">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-900/80 text-slate-400 text-xs uppercase">
          <tr>
            <th className="px-3 py-2 text-left">id</th>
            <th className="px-3 py-2 text-left">source / kind</th>
            <th className="px-3 py-2 text-left">prio</th>
            <th className="px-3 py-2 text-left">state</th>
            <th className="px-3 py-2 text-left">attempts</th>
            <th className="px-3 py-2 text-left">enqueued</th>
            <th className="px-3 py-2 text-left">error</th>
          </tr>
        </thead>
        <tbody>
          {items.map((it) => (
            <tr key={it.id} className="border-t border-slate-800 hover:bg-slate-800/40">
              <td className="px-3 py-2 font-mono text-slate-400">#{it.id}</td>
              <td className="px-3 py-2">
                <span className="font-medium">{it.source}</span>
                <span className="text-slate-500"> / {it.kind}</span>
              </td>
              <td className="px-3 py-2">{it.priority}</td>
              <td className="px-3 py-2">
                <span className={`text-xs px-2 py-0.5 rounded border ${STATE_COLORS[it.state] || "border-slate-600"}`}>
                  {it.state}
                </span>
              </td>
              <td className="px-3 py-2">{it.attempts}</td>
              <td className="px-3 py-2 text-slate-400">{formatAgo(it.enqueued_at)}</td>
              <td className="px-3 py-2 text-rose-400 text-xs max-w-[260px] truncate" title={it.error || ""}>
                {it.error || ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RunsTable({ runs }: { runs: RunRow[] }) {
  if (!runs.length) return <div className="text-slate-500 italic p-4">no runs yet</div>;
  return (
    <div className="overflow-x-auto rounded border border-slate-700">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-900/80 text-slate-400 text-xs uppercase">
          <tr>
            <th className="px-3 py-2 text-left">status</th>
            <th className="px-3 py-2 text-left">source</th>
            <th className="px-3 py-2 text-left">task</th>
            <th className="px-3 py-2 text-left">completed</th>
            <th className="px-3 py-2 text-left">duration</th>
            <th className="px-3 py-2 text-left">summary</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id} className="border-t border-slate-800 hover:bg-slate-800/40">
              <td className="px-3 py-2">
                {r.status === "success" ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <XCircle className="w-4 h-4 text-rose-400" />
                )}
              </td>
              <td className="px-3 py-2">{r.source}</td>
              <td className="px-3 py-2 font-mono text-slate-400">{r.task_id || "—"}</td>
              <td className="px-3 py-2 text-slate-400">{formatAgo(r.completed_at)}</td>
              <td className="px-3 py-2 tabular-nums">
                <Clock className="w-3 h-3 inline mr-1 text-slate-500" />
                {r.duration_seconds?.toFixed(1) ?? "—"}s
              </td>
              <td className="px-3 py-2 text-slate-300 max-w-[480px] truncate" title={r.summary}>
                {r.summary}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SourcesTable({ sources }: { sources: Array<{ name: string; enabled: boolean; interval_seconds?: number; max_inflight?: number; depth?: Record<string, number> }> }) {
  if (!sources.length) return <div className="text-slate-500 italic p-4">no sources configured</div>;
  return (
    <div className="overflow-x-auto rounded border border-slate-700">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-900/80 text-slate-400 text-xs uppercase">
          <tr>
            <th className="px-3 py-2 text-left">source</th>
            <th className="px-3 py-2 text-left">enabled</th>
            <th className="px-3 py-2 text-left">interval</th>
            <th className="px-3 py-2 text-left">max inflight</th>
            <th className="px-3 py-2 text-left">queued</th>
            <th className="px-3 py-2 text-left">running</th>
            <th className="px-3 py-2 text-left">completed</th>
          </tr>
        </thead>
        <tbody>
          {sources.map((s) => (
            <tr key={s.name} className="border-t border-slate-800 hover:bg-slate-800/40">
              <td className="px-3 py-2 font-medium">{s.name}</td>
              <td className="px-3 py-2">
                <span className={`text-xs px-2 py-0.5 rounded border ${s.enabled ? "border-emerald-500/40 text-emerald-300 bg-emerald-500/10" : "border-slate-600 text-slate-500"}`}>
                  {s.enabled ? "on" : "off"}
                </span>
              </td>
              <td className="px-3 py-2 text-slate-400">{s.interval_seconds}s</td>
              <td className="px-3 py-2 text-slate-400">{s.max_inflight ?? "—"}</td>
              <td className="px-3 py-2">{s.depth?.queued ?? 0}</td>
              <td className="px-3 py-2">{(s.depth?.claimed ?? 0) + (s.depth?.running ?? 0)}</td>
              <td className="px-3 py-2 text-slate-400">{s.depth?.completed ?? 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
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
