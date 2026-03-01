import { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { api, type UsageChartData } from "../api";

const RANGES = ["24h", "7d", "30d"] as const;

function formatTime(ts: number, range: string): string {
  const d = new Date(ts);
  if (range === "24h") return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (range === "7d") return d.toLocaleDateString([], { weekday: "short", hour: "2-digit" });
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

export default function UsageChart() {
  const [range, setRange] = useState<string>("7d");
  const [data, setData] = useState<UsageChartData | null>(null);

  useEffect(() => {
    api.usageChart(range).then(setData).catch(console.error);
  }, [range]);

  return (
    <div className="bg-surface-1 rounded-xl p-5 border border-surface-3/50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-slate-300">Token Usage</h3>
        <div className="flex gap-1">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                range === r
                  ? "bg-brand-600 text-white"
                  : "bg-surface-2 text-slate-400 hover:text-slate-200"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data?.data || []}>
            <defs>
              <linearGradient id="gradInput" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#818cf8" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradOutput" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#34d399" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="ts"
              tickFormatter={(ts) => formatTime(ts, range)}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={{ stroke: "#334155" }}
              tickLine={false}
            />
            <YAxis
              tickFormatter={formatTokens}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={50}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelFormatter={(ts) => new Date(ts).toLocaleString()}
              formatter={(value: number, name: string) => [
                formatTokens(value),
                name === "input" ? "Input" : name === "output" ? "Output" : "Cache",
              ]}
            />
            <Area
              type="monotone"
              dataKey="input"
              stroke="#818cf8"
              fill="url(#gradInput)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="output"
              stroke="#34d399"
              fill="url(#gradOutput)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
