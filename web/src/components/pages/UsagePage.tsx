import { useState, useEffect, useCallback } from 'react'
import { ChartArea, RefreshCw, Clock, Zap, Database, DollarSign } from 'lucide-react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { api, type UsagePing, type UsageBucket, type UsageModelBreakdown, type UsageRecord } from '../../api'

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function fmtCost(n: number): string {
  if (n < 0.01) return `$${n.toFixed(4)}`
  return `$${n.toFixed(2)}`
}

function bucketLabel(bucket: string, period: string): string {
  if (period === '7d' || period === '30d') {
    // "2026-04-03" — use noon UTC to avoid date shifting across timezones
    const d = new Date(bucket + 'T12:00:00Z')
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  }
  // "2026-04-03T14:30:00" stored as UTC — convert to local time
  const d = new Date(bucket + 'Z')
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false })
}

const MODEL_COLORS = [
  '#818cf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa',
  '#22d3ee', '#fb923c', '#e879f9', '#4ade80', '#f472b6',
]

// ── Allocation Meter ─────────────────────────────────────────────────────────

function AllocationMeter({
  label, utilization, status, resetTs, localTokens, sublabel,
}: {
  label: string
  utilization: number  // 0.0–1.0 from Anthropic
  status: string       // "allowed" | "throttled" etc.
  resetTs?: number     // unix epoch seconds
  localTokens?: number // local tracked tokens for context
  sublabel?: string
}) {
  const p = utilization * 100
  const color = p > 90 ? 'bg-red-500' : p > 70 ? 'bg-amber-500' : 'bg-brand-500'
  const textColor = p > 90 ? 'text-red-400' : p > 70 ? 'text-amber-400' : 'text-brand-400'
  const statusColor = status === 'allowed' ? 'text-green-400' : 'text-red-400'
  const resetCountdown = resetTs ? formatResetCountdown(resetTs) : ''
  const resetDate = resetTs ? formatResetDate(resetTs) : ''

  return (
    <div className="bg-surface-1 border border-surface-3/50 rounded-xl p-5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</span>
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-medium ${statusColor} uppercase`}>{status}</span>
          <span className={`text-sm font-bold ${textColor}`}>{p.toFixed(1)}%</span>
        </div>
      </div>
      <div className="text-sm text-slate-300 mb-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {localTokens != null && localTokens > 0 && (
            <span>
              <span className="font-mono font-semibold text-slate-100">{fmtTokens(localTokens)}</span>
              <span className="text-slate-500 text-xs ml-1">via Lloyd</span>
            </span>
          )}
          {sublabel && <span className="text-slate-500 text-xs">{sublabel}</span>}
        </div>
        {resetTs && (
          <div className="text-right">
            <div className="text-xs text-slate-400">
              resets <span className="font-mono font-semibold text-slate-300">{resetCountdown}</span>
            </div>
            <div className="text-[10px] text-slate-500 font-mono">{resetDate}</div>
          </div>
        )}
      </div>
      <div className="h-2.5 bg-surface-2 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${Math.min(100, p)}%` }}
        />
      </div>
    </div>
  )
}

function formatResetCountdown(epochSec: number): string {
  const diff = epochSec - Date.now() / 1000
  if (diff <= 0) return 'now'
  const h = Math.floor(diff / 3600)
  const m = Math.ceil((diff % 3600) / 60)
  if (h >= 24) {
    const d = Math.floor(h / 24)
    const rh = h % 24
    return `${d}d ${rh}h`
  }
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function formatResetDate(epochSec: number): string {
  const d = new Date(epochSec * 1000)
  return d.toLocaleDateString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric',
  }) + ' ' + d.toLocaleTimeString(undefined, {
    hour: 'numeric', minute: '2-digit',
  })
}

// ── Stat Card ────────────────────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, sub }: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  sub?: string
}) {
  return (
    <div className="bg-surface-1 border border-surface-3/50 rounded-xl p-4 flex items-start gap-3">
      <div className="p-2 rounded-lg bg-brand-500/10">
        <Icon className="w-4 h-4 text-brand-400" />
      </div>
      <div>
        <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">{label}</div>
        <div className="text-lg font-bold text-slate-100 font-mono">{value}</div>
        {sub && <div className="text-xs text-slate-500">{sub}</div>}
      </div>
    </div>
  )
}

// ── Custom Tooltip ───────────────────────────────────────────────────────────

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-surface-1 border border-surface-3/50 rounded-lg px-3 py-2 shadow-lg text-xs">
      <div className="font-semibold text-slate-300 mb-1">{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-400">{p.name}:</span>
          <span className="font-mono text-slate-200">{fmtTokens(p.value)}</span>
        </div>
      ))}
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────────────────────

type Period = '4h' | '24h' | '7d' | '30d'

export default function UsagePage() {
  const [ping, setPing] = useState<UsagePing | null>(null)
  const [history, setHistory] = useState<UsageBucket[]>([])
  const [period, setPeriod] = useState<Period>('4h')
  const [models, setModels] = useState<UsageModelBreakdown[]>([])
  const [recent, setRecent] = useState<UsageRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [pinging, setPinging] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    setPinging(true)
    try {
      const [p, h, m, r] = await Promise.all([
        api.usagePing(),
        api.usageHistory(period),
        api.usageModels(
          period === '4h' ? 4 : period === '24h' ? 24 : 0,
          period === '7d' ? 7 : period === '30d' ? 30 : 0,
        ),
        api.usageRecent(30),
      ])
      if (!p.error) setPing(p)
      setHistory(h.buckets)
      setModels(m.models)
      setRecent(r.records)
    } catch (err) {
      console.error('Failed to load usage data:', err)
    }
    setLoading(false)
    setPinging(false)
  }, [period])

  // Fetch on mount (tab selected) and when period changes
  useEffect(() => { refresh() }, [refresh])

  // Auto-refresh local stats every 30s (lightweight, no ping)
  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const [h, m, r] = await Promise.all([
          api.usageHistory(period),
          api.usageModels(
            period === '4h' ? 4 : period === '24h' ? 24 : 0,
            period === '7d' ? 7 : period === '30d' ? 30 : 0,
          ),
          api.usageRecent(30),
        ])
        setHistory(h.buckets)
        setModels(m.models)
        setRecent(r.records)
      } catch { /* keep stale */ }
    }, 30_000)
    return () => clearInterval(iv)
  }, [period])

  const chartData = history.map(b => ({
    ...b,
    label: bucketLabel(b.bucket, period),
    total: b.input_tokens + b.output_tokens,
  }))

  const rl = ping?.rate_limits
  const local5h = ping?.local_5h
  const local7d = ping?.local_7d

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <ChartArea className="w-5 h-5 text-brand-400" />
          <h2 className="text-lg font-semibold text-slate-200">Usage</h2>
          {ping?.pinged_at && (
            <span className="text-[10px] text-slate-500 font-mono">
              pinged {formatAgo(new Date(ping.pinged_at + 'Z'))}
            </span>
          )}
        </div>
        <button
          onClick={refresh}
          disabled={loading || pinging}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-brand-400 hover:bg-brand-500/10 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${pinging ? 'animate-spin' : ''}`} />
          {pinging ? 'Pinging...' : 'Refresh'}
        </button>
      </div>

      {/* Allocation Meters — from Anthropic rate-limit headers */}
      {rl && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {rl['5h-utilization'] != null && (
            <AllocationMeter
              label="5-Hour Window"
              utilization={rl['5h-utilization'] as number}
              status={(rl['5h-status'] as string) || 'unknown'}
              resetTs={rl['5h-reset'] as number | undefined}
              localTokens={local5h ? local5h.input_tokens + local5h.output_tokens : undefined}
            />
          )}
          {rl['7d-utilization'] != null && (
            <AllocationMeter
              label="7-Day Window"
              utilization={rl['7d-utilization'] as number}
              status={(rl['7d-status'] as string) || 'unknown'}
              resetTs={rl['7d-reset'] as number | undefined}
              localTokens={local7d ? local7d.input_tokens + local7d.output_tokens : undefined}
            />
          )}
        </div>
      )}

      {/* Summary Stats */}
      {local5h && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            icon={Zap}
            label="Requests (5h)"
            value={String(local5h.requests)}
            sub={local7d ? `${local7d.requests} this week` : undefined}
          />
          <StatCard
            icon={Database}
            label="Input Tokens (5h)"
            value={fmtTokens(local5h.input_tokens)}
            sub={local5h.cache_read > 0 ? `${fmtTokens(local5h.cache_read)} cached` : undefined}
          />
          <StatCard
            icon={Database}
            label="Output Tokens (5h)"
            value={fmtTokens(local5h.output_tokens)}
          />
          <StatCard
            icon={DollarSign}
            label="Cost (5h)"
            value={fmtCost(local5h.cost_usd)}
            sub={local7d ? `${fmtCost(local7d.cost_usd)} this week` : undefined}
          />
        </div>
      )}

      {/* Period Tabs + Token Area Chart */}
      <div className="bg-surface-1 border border-surface-3/50 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-300">Token Usage</h3>
          <div className="flex gap-1">
            {(['4h', '24h', '7d', '30d'] as Period[]).map(p => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  period === p
                    ? 'bg-brand-500/20 text-brand-400'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-surface-2'
                }`}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
        <div className="h-64">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="inputGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#818cf8" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="outputGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#34d399" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="cacheGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#fbbf24" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#64748b', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={fmtTokens}
                />
                <Tooltip content={<ChartTooltip />} />
                <Area
                  type="monotone"
                  dataKey="input_tokens"
                  name="Input"
                  stroke="#818cf8"
                  strokeWidth={2}
                  fill="url(#inputGrad)"
                  stackId="tokens"
                />
                <Area
                  type="monotone"
                  dataKey="output_tokens"
                  name="Output"
                  stroke="#34d399"
                  strokeWidth={2}
                  fill="url(#outputGrad)"
                  stackId="tokens"
                />
                <Area
                  type="monotone"
                  dataKey="cache_read"
                  name="Cache Read"
                  stroke="#fbbf24"
                  strokeWidth={1.5}
                  fill="url(#cacheGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-slate-500 text-sm">
              No usage data for this period
            </div>
          )}
        </div>
      </div>

      {/* Bottom Row: Model Breakdown + Recent Requests */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Model Breakdown Bar Chart */}
        <div className="bg-surface-1 border border-surface-3/50 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Model Breakdown</h3>
          {models.length > 0 ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={models.map(m => ({
                    ...m,
                    name: m.model?.split('/').pop()?.split('-').slice(0, 3).join('-') || m.model || 'unknown',
                    total: m.input_tokens + m.output_tokens,
                  }))}
                  layout="vertical"
                  margin={{ top: 0, right: 10, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={fmtTokens}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    width={120}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar dataKey="input_tokens" name="Input" stackId="a" radius={[0, 0, 0, 0]}>
                    {models.map((_, i) => (
                      <Cell key={i} fill={MODEL_COLORS[i % MODEL_COLORS.length]} fillOpacity={0.8} />
                    ))}
                  </Bar>
                  <Bar dataKey="output_tokens" name="Output" stackId="a" radius={[0, 4, 4, 0]}>
                    {models.map((_, i) => (
                      <Cell key={i} fill={MODEL_COLORS[i % MODEL_COLORS.length]} fillOpacity={0.5} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex items-center justify-center h-56 text-slate-500 text-sm">
              No model data
            </div>
          )}
        </div>

        {/* Recent Requests Table */}
        <div className="bg-surface-1 border border-surface-3/50 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Recent Requests</h3>
          <div className="overflow-y-auto max-h-56 -mx-1">
            {recent.length > 0 ? (
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-surface-3/30">
                    <th className="text-left py-1.5 px-2 font-semibold">Time</th>
                    <th className="text-left py-1.5 px-2 font-semibold">Model</th>
                    <th className="text-right py-1.5 px-2 font-semibold">In</th>
                    <th className="text-right py-1.5 px-2 font-semibold">Out</th>
                    <th className="text-right py-1.5 px-2 font-semibold">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map(r => {
                    const t = new Date(r.ts + 'Z')
                    const ago = formatAgo(t)
                    const shortModel = r.model?.split('/').pop()?.split('-').slice(0, 2).join('-') || r.model || '—'
                    return (
                      <tr key={r.id} className="border-b border-surface-3/20 hover:bg-surface-2/50">
                        <td className="py-1.5 px-2 text-slate-400 whitespace-nowrap" title={r.ts}>
                          <Clock className="w-3 h-3 inline mr-1 opacity-50" />{ago}
                        </td>
                        <td className="py-1.5 px-2 text-slate-300 font-mono truncate max-w-[120px]">{shortModel}</td>
                        <td className="py-1.5 px-2 text-right text-slate-300 font-mono">{fmtTokens(r.input_tokens)}</td>
                        <td className="py-1.5 px-2 text-right text-slate-300 font-mono">{fmtTokens(r.output_tokens)}</td>
                        <td className="py-1.5 px-2 text-right text-slate-400 font-mono">
                          {r.cost_usd != null ? fmtCost(r.cost_usd) : '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            ) : (
              <div className="flex items-center justify-center h-32 text-slate-500 text-sm">
                No requests recorded yet
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function formatAgo(date: Date): string {
  const sec = (Date.now() - date.getTime()) / 1000
  if (sec < 60) return 'just now'
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
  return `${Math.floor(sec / 86400)}d ago`
}
