import { useEffect, useState } from "react";
import { Users, MessageSquare, Clock } from "lucide-react";
import { api, type SessionSummary } from "../../api";

function formatNum(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

function timeAgo(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  if (diff < 60_000) return "just now";
  if (diff < 3600_000) return Math.floor(diff / 60_000) + "m ago";
  if (diff < 86400_000) return Math.floor(diff / 3600_000) + "h ago";
  return Math.floor(diff / 86400_000) + "d ago";
}

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);

  useEffect(() => {
    api.sessions().then((d) => setSessions(d.sessions)).catch(console.error);
  }, []);

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Users className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Sessions</h2>
        <span className="text-xs text-slate-500">{sessions.length} active sessions</span>
      </div>

      <div className="bg-surface-1 rounded-xl border border-surface-3/50 overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-surface-3/50 bg-surface-2/50">
              <th className="text-left px-4 py-3 font-medium">Session ID</th>
              <th className="text-left px-4 py-3 font-medium">Model</th>
              <th className="text-right px-4 py-3 font-medium">Messages</th>
              <th className="text-right px-4 py-3 font-medium">Input</th>
              <th className="text-right px-4 py-3 font-medium">Output</th>
              <th className="text-right px-4 py-3 font-medium">Cache</th>
              <th className="text-right px-4 py-3 font-medium">Last Activity</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr
                key={s.sessionId}
                className="border-b border-surface-3/30 hover:bg-surface-2/50 transition-colors cursor-pointer"
              >
                <td className="px-4 py-3 font-mono text-slate-300">
                  {s.sessionId.slice(0, 12)}...
                </td>
                <td className="px-4 py-3">
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-brand-600/15 text-brand-400">
                    {s.model || "unknown"}
                  </span>
                </td>
                <td className="px-4 py-3 text-right text-slate-300">
                  <span className="inline-flex items-center gap-1">
                    <MessageSquare className="w-3 h-3 text-slate-500" />
                    {s.messageCount}
                  </span>
                </td>
                <td className="px-4 py-3 text-right text-slate-400 font-mono">{formatNum(s.input)}</td>
                <td className="px-4 py-3 text-right text-slate-400 font-mono">{formatNum(s.output)}</td>
                <td className="px-4 py-3 text-right text-slate-400 font-mono">{formatNum(s.cacheRead)}</td>
                <td className="px-4 py-3 text-right text-slate-500">
                  <span className="inline-flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {timeAgo(s.lastActivity)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
