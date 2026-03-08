import React, { useEffect, useState } from "react";
import { Users, MessageSquare, Clock, Globe, Hash, Send, Lock, Smartphone } from "lucide-react";
import { api, type SessionSummary } from "../../api";

function formatNum(n: number | undefined): string {
  if (n == null) return "-";
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

function formatSessionKey(key: string): string {
  const parts = key.split(":");
  const name = parts[parts.length - 1];
  if (name === "main") return "Main";
  return name.length > 16 ? name.slice(0, 16) + "..." : name;
}

interface SessionsPageProps {
  onOpenSession?: (sessionKey: string) => void;
}

export default function SessionsPage({ onOpenSession }: SessionsPageProps = {}) {
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
              <th className="text-left px-4 py-3 font-medium">Session</th>
              <th className="text-left px-4 py-3 font-medium">Source</th>
              <th className="text-left px-4 py-3 font-medium">Model</th>
              <th className="text-right px-4 py-3 font-medium">Messages</th>
              <th className="text-right px-4 py-3 font-medium">Last Activity</th>
            </tr>
          </thead>
          <tbody>
            {[...sessions].sort((a, b) => new Date(b.lastActivity).getTime() - new Date(a.lastActivity).getTime()).map((s) => (
              <tr
                key={s.sessionKey}
                onClick={() => onOpenSession?.(s.sessionKey)}
                className="border-b border-surface-3/30 hover:bg-surface-2/50 transition-colors cursor-pointer"
              >
                <td className="px-4 py-3">
                  {s.summary ? (
                    <div>
                      <span className="text-slate-200">{s.summary}</span>
                      <div className="text-[10px] font-mono text-slate-500 mt-0.5">{formatSessionKey(s.sessionKey)}</div>
                    </div>
                  ) : (
                    <span className="font-mono text-slate-300">{formatSessionKey(s.sessionKey)}</span>
                  )}
                </td>
                {(() => {
                  const sourceIcons: Record<string, React.ElementType> = { webchat: Globe, discord: Hash, telegram: Send, signal: Lock, whatsapp: Smartphone, other: MessageSquare };
                  const SourceIcon = sourceIcons[s.source || "other"] || MessageSquare;
                  const label = (() => {
                    if (!s.source || s.source === "webchat") return "alan";
                    if (s.peer === "912913439209443358") return "alan";
                    return s.peer || s.source;
                  })().toLowerCase();
                  return (
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center gap-1 text-slate-300">
                        <SourceIcon className="w-3.5 h-3.5 text-slate-500" />
                        {label}
                      </span>
                    </td>
                  );
                })()}
                <td className="px-4 py-3">
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-brand-600/15 text-brand-400">
                    {s.model || "unknown"}
                  </span>
                </td>
                <td className="px-4 py-3 text-right text-slate-300">
                  <span className="inline-flex items-center gap-1">
                    <MessageSquare className="w-3 h-3 text-slate-500" />
                    {s.messageCount ?? "-"}
                  </span>
                </td>
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
