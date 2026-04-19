import { useEffect, useState, useCallback } from "react";
import {
  Workflow, Pause, Play, RefreshCw, CheckCircle2, XCircle, Clock, Loader2,
  ArrowUpCircle, Trash2, FileText, FolderOpen,
} from "lucide-react";
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
  const [tab, setTab] = useState<"queue" | "runs" | "sources" | "review">("queue");
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
        {(["queue", "runs", "sources", "review"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm border-b-2 ${tab === t ? "border-sky-400 text-sky-300" : "border-transparent text-slate-400 hover:text-slate-200"}`}
          >
            {t === "queue" ? "Queue" : t === "runs" ? "Recent Runs" : t === "sources" ? "Sources" : "Review"}
          </button>
        ))}
      </div>

      {/* Filters — not on Sources or Review tabs */}
      {(tab === "queue" || tab === "runs") && (
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
      {tab === "review" && <ReviewPanel />}
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
              <td className="px-3 py-2 text-xs max-w-[260px] truncate">
                {it.error && it.state !== "completed" && (
                  <span className="text-rose-400" title={it.error}>{it.error}</span>
                )}
                {it.error && it.state === "completed" && it.attempts > 1 && (
                  <span
                    className="text-slate-500 italic"
                    title={`Succeeded on retry. Prior error: ${it.error}`}
                  >
                    (recovered after {it.attempts} attempts)
                  </span>
                )}
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
      <div className="w-96 flex-shrink-0 border border-slate-700 rounded bg-slate-900/40 flex flex-col">
        <div className="p-2 border-b border-slate-700 flex gap-2 items-center">
          <select
            value={sourceFilter}
            onChange={(e) => { setSourceFilter(e.target.value); setSelected(null); }}
            className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm flex-1"
          >
            <option value="">all sources ({items.length})</option>
            {sources.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <button
            onClick={refresh}
            disabled={loading}
            className="p-1.5 rounded border border-slate-600 hover:border-sky-500"
            title="Refresh"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
        <div className="overflow-y-auto flex-1">
          {items.length === 0 && (
            <div className="p-6 text-center text-slate-500 italic text-sm">
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
                className={`w-full text-left px-3 py-2 border-b border-slate-800 hover:bg-slate-800/60 ${active ? "bg-slate-800/80" : ""}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-sky-400 font-medium">{it.source}</span>
                  <span className="text-xs text-slate-500">{formatAgo(it.mtime)}</span>
                </div>
                <div className="text-sm truncate mt-0.5" title={it.filename}>
                  {it.filename.replace(/\.md$/, "")}
                </div>
                <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
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
      <div className="flex-1 border border-slate-700 rounded bg-slate-900/40 flex flex-col min-w-0">
        {!selected && (
          <div className="flex-1 flex items-center justify-center text-slate-500 italic">
            <div className="text-center">
              <FileText className="w-8 h-8 mx-auto mb-2 text-slate-600" />
              select an artifact to review
            </div>
          </div>
        )}
        {selected && (
          <>
            <div className="p-3 border-b border-slate-700 flex-shrink-0">
              <div className="flex items-center gap-2 mb-1">
                <FolderOpen className="w-4 h-4 text-slate-500" />
                <span className="text-xs text-slate-500 truncate" title={selected.path}>
                  {selected.path.replace("/home/alansrobotlab/obsidian/", "~/obsidian/")}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <h3 className="font-medium">{selected.filename}</h3>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder={canPromoteWithoutDest ? `dest: ${DEFAULT_DEST[selected.source]}` : "destination required"}
                    value={destination}
                    onChange={(e) => setDestination(e.target.value)}
                    className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs w-56"
                  />
                  <button
                    onClick={promote}
                    disabled={busy || (!canPromoteWithoutDest && !destination)}
                    className="px-3 py-1 text-sm rounded border border-emerald-500/40 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
                    title="Promote to canonical vault location"
                  >
                    <ArrowUpCircle className="w-3.5 h-3.5" />
                    Promote
                  </button>
                  <button
                    onClick={reject}
                    disabled={busy}
                    className="px-3 py-1 text-sm rounded border border-rose-500/40 bg-rose-500/10 text-rose-300 hover:bg-rose-500/20 disabled:opacity-40 flex items-center gap-1.5"
                    title="Move to _rejected/"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Reject
                  </button>
                </div>
              </div>
              {error && <div className="mt-2 text-xs text-rose-400">{error}</div>}
            </div>
            <div className="overflow-y-auto flex-1 p-4 font-mono text-sm">
              {detail?.frontmatter && Object.keys(detail.frontmatter).length > 0 && (
                <div className="mb-4 p-3 rounded bg-slate-950/80 border border-slate-800">
                  <div className="text-xs text-slate-500 uppercase mb-2">frontmatter</div>
                  {Object.entries(detail.frontmatter).map(([k, v]) => (
                    <div key={k} className="flex gap-2 text-xs">
                      <span className="text-slate-400 w-28">{k}</span>
                      <span className="text-slate-200 break-all">{
                        typeof v === "object" ? JSON.stringify(v) : String(v)
                      }</span>
                    </div>
                  ))}
                </div>
              )}
              <pre className="whitespace-pre-wrap text-slate-200">{detail?.body ?? "loading…"}</pre>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
