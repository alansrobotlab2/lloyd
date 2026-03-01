import { useEffect, useState } from "react";
import { CheckCircle, XCircle, Clock } from "lucide-react";
import { api, type RunEntry, type ToolCallEntry } from "../api";

type Tab = "runs" | "tools";

function timeAgo(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  if (diff < 60_000) return "just now";
  if (diff < 3600_000) return Math.floor(diff / 60_000) + "m ago";
  if (diff < 86400_000) return Math.floor(diff / 3600_000) + "h ago";
  return Math.floor(diff / 86400_000) + "d ago";
}

function formatMs(ms: number): string {
  if (ms >= 1000) return (ms / 1000).toFixed(1) + "s";
  return ms + "ms";
}

const TIER_COLORS: Record<string, string> = {
  local: "text-emerald-400 bg-emerald-400/10",
  haiku: "text-sky-400 bg-sky-400/10",
  sonnet: "text-indigo-400 bg-indigo-400/10",
  opus: "text-amber-400 bg-amber-400/10",
  unknown: "text-slate-400 bg-slate-400/10",
};

export default function ApiCallsTable() {
  const [tab, setTab] = useState<Tab>("runs");
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCallEntry[]>([]);

  useEffect(() => {
    const load = () =>
      api
        .apiCalls()
        .then((d) => {
          setRuns(d.runs);
          setToolCalls(d.toolCalls);
        })
        .catch(console.error);
    load();
    const interval = setInterval(load, 10_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-surface-1 rounded-xl border border-surface-3/50 flex flex-col min-h-0">
      <div className="flex items-center gap-4 px-5 pt-4 pb-3">
        <h3 className="text-sm font-medium text-slate-300">Recent API Calls</h3>
        <div className="flex gap-1 ml-auto">
          <button
            onClick={() => setTab("runs")}
            className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
              tab === "runs"
                ? "bg-brand-600 text-white"
                : "bg-surface-2 text-slate-400 hover:text-slate-200"
            }`}
          >
            Runs
          </button>
          <button
            onClick={() => setTab("tools")}
            className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
              tab === "tools"
                ? "bg-brand-600 text-white"
                : "bg-surface-2 text-slate-400 hover:text-slate-200"
            }`}
          >
            Tool Calls
          </button>
        </div>
      </div>

      <div className="overflow-auto flex-1 px-5 pb-4">
        {tab === "runs" ? (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-surface-3/50">
                <th className="text-left py-2 font-medium">Time</th>
                <th className="text-left py-2 font-medium">Model</th>
                <th className="text-right py-2 font-medium">Duration</th>
                <th className="text-right py-2 font-medium">LLM</th>
                <th className="text-right py-2 font-medium">Tools</th>
                <th className="text-right py-2 font-medium">Calls</th>
                <th className="text-center py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr
                  key={i}
                  className="border-b border-surface-3/30 hover:bg-surface-2/50 transition-colors"
                >
                  <td className="py-2 text-slate-400">{timeAgo(r.ts)}</td>
                  <td className="py-2">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${TIER_COLORS[r.model] || TIER_COLORS.unknown}`}
                    >
                      {r.model}
                    </span>
                  </td>
                  <td className="py-2 text-right text-slate-300 font-mono">
                    {formatMs(r.totalMs)}
                  </td>
                  <td className="py-2 text-right text-slate-400 font-mono">
                    {formatMs(r.llmMs)}
                  </td>
                  <td className="py-2 text-right text-slate-400 font-mono">
                    {formatMs(r.toolMs)}
                  </td>
                  <td className="py-2 text-right text-slate-400">
                    {r.toolCallCount}
                  </td>
                  <td className="py-2 text-center">
                    {r.success ? (
                      <CheckCircle className="w-3.5 h-3.5 text-emerald-400 inline" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5 text-red-400 inline" />
                    )}
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-500">
                    No runs recorded yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-surface-3/50">
                <th className="text-left py-2 font-medium">Time</th>
                <th className="text-left py-2 font-medium">Tool</th>
                <th className="text-right py-2 font-medium">Duration</th>
              </tr>
            </thead>
            <tbody>
              {toolCalls.map((t, i) => (
                <tr
                  key={i}
                  className="border-b border-surface-3/30 hover:bg-surface-2/50 transition-colors"
                >
                  <td className="py-2 text-slate-400">{timeAgo(t.ts)}</td>
                  <td className="py-2 text-slate-300 font-mono">{t.toolName}</td>
                  <td className="py-2 text-right text-slate-300 font-mono">
                    {formatMs(t.durationMs)}
                  </td>
                </tr>
              ))}
              {toolCalls.length === 0 && (
                <tr>
                  <td colSpan={3} className="py-8 text-center text-slate-500">
                    No tool calls recorded yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
