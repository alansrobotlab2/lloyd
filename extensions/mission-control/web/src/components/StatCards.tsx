import { useEffect, useState } from "react";
import { ArrowUpRight, MessageSquare, Zap, Database, Layers } from "lucide-react";
import { api, type Stats } from "../api";

function formatNum(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

const CARDS = [
  { key: "totalInput" as const, label: "Input Tokens", icon: Zap, color: "text-indigo-400" },
  { key: "totalOutput" as const, label: "Output Tokens", icon: ArrowUpRight, color: "text-emerald-400" },
  { key: "totalCacheRead" as const, label: "Cache Read", icon: Database, color: "text-amber-400" },
  { key: "totalSessions" as const, label: "Sessions", icon: MessageSquare, color: "text-sky-400" },
];

export default function StatCards() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    const load = () => api.stats().then(setStats).catch(console.error);
    load();
    const interval = setInterval(load, 10_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="grid grid-cols-4 gap-4">
      {CARDS.map((card) => {
        const Icon = card.icon;
        const value = stats ? stats[card.key] : 0;
        return (
          <div
            key={card.key}
            className="bg-surface-1 rounded-xl p-5 border border-surface-3/50"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-slate-400">{card.label}</span>
              <Icon className={`w-4 h-4 ${card.color}`} />
            </div>
            <div className="text-2xl font-semibold tracking-tight">
              {stats ? formatNum(value) : "--"}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {card.key === "totalSessions" ? "active sessions" : "last 30 days"}
            </div>
          </div>
        );
      })}
    </div>
  );
}
