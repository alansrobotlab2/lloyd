import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Activity, AlertTriangle, Bot, CalendarClock, CheckCircle2, Cpu, Gauge,
  Layers, MinusCircle, Server, Terminal, Workflow, XCircle, Zap,
} from 'lucide-react'
import {
  dashboardApi, sectionOk,
  type AutonomyState, type AutonomyTaskRow, type BacklogState,
  type DashboardSnapshot, type GpuInfo,
  type SubagentRun, type UsageBucket, type VllmEngine, type WorkersState,
} from '../../api'
import { useReportMcFocus } from '../../contexts/McUiContext'
import { cn } from '@/lib/utils'

// Poll cadence. Fast enough that a turn starting is visible almost
// immediately, slow enough that the aggregated endpoint (~25-100ms) is
// noise against a box whose real job is holding a KV cache steady.
const POLL_MS = 2000

// ── Status vocabulary ──────────────────────────────────────────────────
//
// Reserved for state, never reused as a series color. Every use pairs the
// color with an icon and a word, so the state is never carried by hue
// alone. Steps are the 400-level: on this surface they clear 3:1 contrast
// and hold ΔE 10.6 separation under protanopia.

type Tone = 'good' | 'warn' | 'crit' | 'accent' | 'idle'

const TONE_TEXT: Record<Tone, string> = {
  good: 'text-emerald-400',
  warn: 'text-amber-400',
  crit: 'text-rose-400',
  accent: 'text-violet-400',
  idle: 'text-muted-foreground',
}

const TONE_FILL: Record<Tone, string> = {
  good: 'bg-emerald-400',
  warn: 'bg-amber-400',
  crit: 'bg-rose-400',
  accent: 'bg-violet-400',
  idle: 'bg-muted-foreground/50',
}

// Unfilled track is a lighter step of the fill's own hue, so the meter
// reads as one object and severity carries across the whole bar.
const TONE_TRACK: Record<Tone, string> = {
  good: 'bg-emerald-400/15',
  warn: 'bg-amber-400/15',
  crit: 'bg-rose-400/15',
  accent: 'bg-violet-400/15',
  idle: 'bg-muted-foreground/15',
}

/** Utilisation → severity. Below 70% is unremarkable, not "good". */
function utilTone(pct: number | null | undefined): Tone {
  if (pct == null) return 'idle'
  if (pct >= 90) return 'crit'
  if (pct >= 70) return 'warn'
  return 'accent'
}

// ── Formatting ─────────────────────────────────────────────────────────

function compact(n: number | null | undefined, digits = 1): string {
  if (n == null || !isFinite(n)) return '—'
  const abs = Math.abs(n)
  if (abs >= 1e9) return `${(n / 1e9).toFixed(digits)}B`
  if (abs >= 1e6) return `${(n / 1e6).toFixed(digits)}M`
  if (abs >= 1e3) return `${(n / 1e3).toFixed(digits)}K`
  if (abs >= 100) return n.toFixed(0)
  if (abs >= 10) return n.toFixed(abs % 1 === 0 ? 0 : 1)
  return n.toFixed(abs % 1 === 0 ? 0 : digits)
}

function bytes(n: number | null | undefined): string {
  if (n == null) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

function pct(x: number | null | undefined): string {
  return x == null ? '—' : `${(x * 100).toFixed(0)}%`
}

function seconds(s: number | null | undefined): string {
  if (s == null) return '—'
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`
  return `${s.toFixed(2)}s`
}

function duration(totalSeconds: number | null | undefined): string {
  if (totalSeconds == null) return '—'
  const s = Math.floor(totalSeconds)
  const d = Math.floor(s / 86400)
  const h = Math.floor((s % 86400) / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (d) return `${d}d ${h}h`
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${s % 60}s`
  return `${s}s`
}

// ── Primitives ─────────────────────────────────────────────────────────

function Section({
  title, icon: Icon, right, children,
}: {
  title: string
  icon: React.ComponentType<{ className?: string }>
  right?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          <Icon className="h-3.5 w-3.5" />
          {title}
        </h2>
        {right}
      </div>
      {children}
    </section>
  )
}

function Panel({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className={cn('rounded-lg border border-border bg-card p-3', className)}>
      {children}
    </div>
  )
}

/** A single ratio against a limit. Value is always shown as text, so the
 *  meter is a second reading of the number rather than its only one. */
function Meter({
  label, value, display, tone, sub,
}: {
  label: string
  value: number | null | undefined   // 0..1
  display: string
  tone?: Tone
  sub?: string
}) {
  const t = tone ?? utilTone(value == null ? null : value * 100)
  const width = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-[11px] text-muted-foreground">{label}</span>
        <span className={cn('font-mono text-[11px] tabular-nums', TONE_TEXT[t])}>{display}</span>
      </div>
      <div className={cn('h-1.5 w-full overflow-hidden rounded-full', TONE_TRACK[t])}>
        <div
          className={cn('h-full rounded-full transition-[width] duration-500', TONE_FILL[t])}
          style={{ width: `${width}%` }}
        />
      </div>
      {sub && <div className="text-[10px] text-muted-foreground/70">{sub}</div>}
    </div>
  )
}

function StatTile({
  label, value, unit, tone = 'idle', sub,
}: {
  label: string
  value: string
  unit?: string
  tone?: Tone
  sub?: string
}) {
  return (
    <Panel className="min-w-0">
      <div className="truncate text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1">
        {/* Proportional figures: this is a standalone value, not a column. */}
        <span className={cn('text-xl font-semibold leading-none', TONE_TEXT[tone])}>{value}</span>
        {unit && <span className="text-[11px] text-muted-foreground">{unit}</span>}
      </div>
      {sub && <div className="mt-1 truncate text-[10px] text-muted-foreground/70">{sub}</div>}
    </Panel>
  )
}

const HEALTH_ICON = {
  healthy: CheckCircle2,
  degraded: AlertTriangle,
  stopped: MinusCircle,
} as const

const HEALTH_TONE: Record<string, Tone> = {
  healthy: 'good',
  degraded: 'warn',
  stopped: 'idle',
}

function HealthPill({ health, label }: { health: string; label: string }) {
  const Icon = HEALTH_ICON[health as keyof typeof HEALTH_ICON] ?? XCircle
  const tone = HEALTH_TONE[health] ?? 'crit'
  return (
    <div className="flex items-center gap-1.5 rounded-md border border-border bg-secondary/30 px-2 py-1.5">
      <Icon className={cn('h-3.5 w-3.5 flex-shrink-0', TONE_TEXT[tone])} />
      <span className="truncate text-[11px] text-foreground">{label}</span>
      {/* The state word, not just the icon colour. */}
      <span className="ml-auto text-[10px] capitalize text-muted-foreground">{health}</span>
    </div>
  )
}

function ErrorPanel({ what, error }: { what: string; error: string }) {
  return (
    <Panel className="border-amber-400/30">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-amber-400" />
        <div className="min-w-0">
          <div className="text-[11px] text-foreground">{what} unavailable</div>
          <div className="truncate font-mono text-[10px] text-muted-foreground">{error}</div>
        </div>
      </div>
    </Panel>
  )
}

// ── Engine card ────────────────────────────────────────────────────────

function EngineCard({ engine }: { engine: VllmEngine }) {
  if (!engine.reachable) {
    return (
      <Panel className="opacity-70">
        <div className="flex items-center gap-2">
          <MinusCircle className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-foreground">{engine.alias}</span>
          <span className="ml-auto text-[10px] text-muted-foreground">offline</span>
        </div>
        <div className="mt-1 truncate font-mono text-[10px] text-muted-foreground/70">
          {engine.base_url}
        </div>
      </Panel>
    )
  }

  const running = engine.requests_running ?? 0
  const waiting = engine.requests_waiting ?? 0
  const busy = running > 0 || waiting > 0
  const waitReasons = Object.entries(engine.requests_waiting_by_reason ?? {})
    .filter(([, n]) => n > 0)

  return (
    <Panel>
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'h-1.5 w-1.5 flex-shrink-0 rounded-full',
            busy ? 'animate-pulse bg-violet-400' : 'bg-emerald-400',
          )}
        />
        <span className="text-xs font-medium text-foreground">{engine.alias}</span>
        <span className="truncate font-mono text-[10px] text-muted-foreground">
          {engine.model_name}
        </span>
        <span className="ml-auto flex-shrink-0 text-[10px] text-muted-foreground">
          {engine.awake ? (busy ? 'serving' : 'idle') : 'asleep'}
        </span>
      </div>

      {/* Concurrency — the number the operator asked for first. */}
      <div className="mt-3 grid grid-cols-3 gap-3">
        <div>
          <div className="text-[10px] text-muted-foreground">Running</div>
          <div className={cn('font-mono text-lg leading-none', running > 0 ? 'text-violet-400' : 'text-muted-foreground')}>
            {running}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-muted-foreground">Waiting</div>
          <div className={cn('font-mono text-lg leading-none', waiting > 0 ? 'text-amber-400' : 'text-muted-foreground')}>
            {waiting}
          </div>
        </div>
        <div>
          {/* Lifetime counter, not a live gauge like the two beside it —
              say so, or it reads as "1K requests preempted right now". */}
          <div className="text-[10px] text-muted-foreground">Preemptions</div>
          <div className="font-mono text-lg leading-none text-muted-foreground">
            {compact(engine.preemptions_total, 0)}
          </div>
          <div className="text-[9px] text-muted-foreground/60">since boot</div>
        </div>
      </div>
      {waitReasons.length > 0 && (
        <div className="mt-1 text-[10px] text-amber-400/80">
          waiting: {waitReasons.map(([r, n]) => `${r} ${n}`).join(' · ')}
        </div>
      )}

      <div className="mt-3 space-y-2">
        <Meter
          label="KV cache"
          value={engine.kv_cache_usage}
          display={pct(engine.kv_cache_usage)}
        />
        <Meter
          label="Prefix cache hits"
          value={engine.prefix_cache_hit_rate}
          display={pct(engine.prefix_cache_hit_rate)}
          // A cache HIT rate is good when high — the opposite polarity to
          // a utilisation meter, so severity shading would be backwards.
          tone="accent"
          sub={
            engine.prefix_cache_hit_rate_recent != null
              ? `${pct(engine.prefix_cache_hit_rate_recent)} in the last poll`
              : 'lifetime'
          }
        />
        {engine.spec_decode_hit_rate != null && (
          <Meter
            label="Spec-decode acceptance"
            value={engine.spec_decode_hit_rate}
            display={pct(engine.spec_decode_hit_rate)}
            tone="accent"
          />
        )}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1.5 border-t border-border pt-2 text-[10px]">
        <Kv label="Prefill" value={`${compact(engine.prompt_tokens_per_s)} tok/s`} />
        <Kv label="Decode" value={`${compact(engine.generation_tokens_per_s)} tok/s`} />
        <Kv label="TTFT" value={seconds(engine.ttft_s)} />
        <Kv label="Inter-token" value={seconds(engine.itl_s)} />
      </div>
    </Panel>
  )
}

function Kv({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono tabular-nums text-foreground">{value}</span>
    </div>
  )
}

// ── GPU card ───────────────────────────────────────────────────────────

function GpuCard({ gpu }: { gpu: GpuInfo }) {
  return (
    <Panel>
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[10px] text-muted-foreground">GPU{gpu.index}</span>
        <span className="truncate text-[11px] text-foreground">
          {gpu.name.replace(/^NVIDIA\s+/, '')}
        </span>
      </div>
      <div className="mt-2 space-y-2">
        <Meter
          label="Compute"
          value={gpu.gpu_util == null ? null : gpu.gpu_util / 100}
          display={gpu.gpu_util == null ? '—' : `${gpu.gpu_util.toFixed(0)}%`}
        />
        <Meter
          label="VRAM"
          value={gpu.memory_pct == null ? null : gpu.memory_pct / 100}
          display={gpu.memory_pct == null ? '—' : `${gpu.memory_pct.toFixed(0)}%`}
          sub={`${bytes((gpu.memory_used_mb ?? 0) * 1024 ** 2)} / ${bytes((gpu.memory_total_mb ?? 0) * 1024 ** 2)}`}
        />
      </div>
      <div className="mt-2 flex items-baseline justify-between text-[10px] text-muted-foreground">
        <span>{gpu.temperature_c == null ? '—' : `${gpu.temperature_c.toFixed(0)}°C`}</span>
        <span className="font-mono tabular-nums">
          {gpu.power_draw_w == null ? '—' : `${gpu.power_draw_w.toFixed(0)}W`}
          {gpu.power_limit_w != null && (
            <span className="text-muted-foreground/60"> / {gpu.power_limit_w.toFixed(0)}W</span>
          )}
        </span>
      </div>
    </Panel>
  )
}

// ── Subagent row ───────────────────────────────────────────────────────

const RUN_TONE: Record<string, Tone> = {
  running: 'accent',
  completed: 'good',
  failed: 'crit',
  error: 'crit',
  cancelled: 'warn',
}

function SubagentRow({ run }: { run: SubagentRun }) {
  const tone = RUN_TONE[run.status] ?? 'idle'
  const tools = Object.entries(run.tool_counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
  return (
    <div className="rounded-md border border-border bg-secondary/30 px-2.5 py-2">
      <div className="flex items-center gap-2">
        {run.status === 'running' ? (
          <span className="h-1.5 w-1.5 flex-shrink-0 animate-pulse rounded-full bg-violet-400" />
        ) : (
          <span className={cn('h-1.5 w-1.5 flex-shrink-0 rounded-full', TONE_FILL[tone])} />
        )}
        <span className="truncate text-[11px] text-foreground">
          {run.description || run.prompt_preview.slice(0, 60) || run.run_id}
        </span>
        <span className={cn('ml-auto flex-shrink-0 text-[10px]', TONE_TEXT[tone])}>
          {run.status}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
        <span>{run.subagent_type}</span>
        <span className="font-mono tabular-nums">
          turn {run.turns}/{run.max_turns}
        </span>
        <span className="font-mono tabular-nums">{run.elapsed_s.toFixed(1)}s</span>
        {run.tool_call_count > 0 && (
          <span>
            {tools.map(([name, n]) => (n > 1 ? `${name}×${n}` : name)).join(' · ')}
          </span>
        )}
        {run.status === 'running' && run.last_tool && (
          <span className="text-violet-400/80">→ {run.last_tool}</span>
        )}
      </div>
      {run.error && (
        <div className="mt-1 truncate text-[10px] text-rose-400/90">{run.error}</div>
      )}
    </div>
  )
}

// ── Autonomy / workers / backlog ───────────────────────────────────────

/** Status counts as a labelled row of chips. Counts are the data; the
 *  colour is a secondary cue on the states that mean trouble. */
const STATUS_TONE: Record<string, Tone> = {
  running: 'accent',
  failed: 'crit',
  poisoned: 'crit',
  paused: 'warn',
  queued: 'warn',
  claimed: 'warn',
  review: 'warn',
  up_next: 'accent',
  done: 'good',
  completed: 'good',
  closed: 'idle',
  draft: 'idle',
}

function StatusChips({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts ?? {}).sort((a, b) => b[1] - a[1])
  if (entries.length === 0) {
    return <span className="text-[11px] text-muted-foreground">Nothing yet.</span>
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.map(([status, n]) => {
        const tone = STATUS_TONE[status] ?? 'idle'
        return (
          <span
            key={status}
            className="flex items-baseline gap-1 rounded-md border border-border bg-secondary/30 px-1.5 py-0.5"
          >
            <span className={cn('font-mono text-[11px] tabular-nums', TONE_TEXT[tone])}>{n}</span>
            <span className="text-[10px] text-muted-foreground">{status.replace(/_/g, ' ')}</span>
          </span>
        )
      })}
    </div>
  )
}

/** "in 3h", "2d ago", "—". Autonomy schedules are the one place on this
 *  page where the absolute timestamp is less useful than the offset. */
function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const t = new Date(iso).getTime()
  if (!isFinite(t)) return '—'
  const delta = (t - Date.now()) / 1000
  const abs = Math.abs(delta)
  const unit =
    abs < 60 ? `${Math.round(abs)}s`
      : abs < 3600 ? `${Math.round(abs / 60)}m`
      : abs < 86400 ? `${Math.round(abs / 3600)}h`
      : `${Math.round(abs / 86400)}d`
  return delta >= 0 ? `in ${unit}` : `${unit} ago`
}

function TaskLine({
  task, tone, when,
}: {
  task: AutonomyTaskRow
  tone: Tone
  when: string
}) {
  return (
    <div className="flex items-center gap-2 text-[10px]">
      <CalendarClock className={cn('h-3 w-3 flex-shrink-0', TONE_TEXT[tone])} />
      <span className="truncate text-foreground">{task.name}</span>
      {task.frequency && (
        <span className="flex-shrink-0 text-muted-foreground/60">{task.frequency}</span>
      )}
      <span className={cn('ml-auto flex-shrink-0 font-mono tabular-nums', TONE_TEXT[tone])}>
        {relativeTime(when)}
      </span>
    </div>
  )
}

function AutonomyPanel({ autonomy }: { autonomy: AutonomyState }) {
  return (
    <Panel>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
          Scheduled tasks
        </span>
        <span className="text-[10px] text-muted-foreground">{autonomy.total} defined</span>
      </div>
      <StatusChips counts={autonomy.by_status} />

      {autonomy.running.length > 0 && (
        <div className="mt-2.5 space-y-1 border-t border-border pt-2">
          <div className="text-[10px] uppercase tracking-wider text-violet-400">
            Running now ({autonomy.running.length})
          </div>
          {autonomy.running.map(r => (
            <div key={r.job_id} className="flex items-center gap-2 text-[10px]">
              <span className="h-1.5 w-1.5 flex-shrink-0 animate-pulse rounded-full bg-violet-400" />
              <span className="truncate text-foreground">{r.kind || r.job_id}</span>
              <span className="ml-auto flex-shrink-0 font-mono tabular-nums text-muted-foreground">
                {r.elapsed_s == null ? '—' : duration(r.elapsed_s)}
              </span>
            </div>
          ))}
        </div>
      )}

      {autonomy.overdue.length > 0 && (
        <div className="mt-2.5 space-y-1 border-t border-border pt-2">
          <div className="text-[10px] uppercase tracking-wider text-amber-400">
            Overdue ({autonomy.overdue_count})
          </div>
          {autonomy.overdue.map(t => (
            <TaskLine key={t.name} task={t} tone="warn" when={t.next_run} />
          ))}
        </div>
      )}

      {autonomy.upcoming.length > 0 && (
        <div className="mt-2.5 space-y-1 border-t border-border pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Next up</div>
          {autonomy.upcoming.map(t => (
            <TaskLine key={t.name} task={t} tone="idle" when={t.next_run} />
          ))}
        </div>
      )}

      {autonomy.failing.length > 0 && (
        <div className="mt-2.5 space-y-1 border-t border-border pt-2">
          <div className="text-[10px] uppercase tracking-wider text-rose-400">
            Failed ({autonomy.failing.length})
          </div>
          {autonomy.failing.map(t => (
            <div key={t.name} className="flex items-center gap-2 text-[10px]">
              <XCircle className="h-3 w-3 flex-shrink-0 text-rose-400" />
              <span className="truncate text-foreground">{t.name}</span>
              <span className="ml-auto flex-shrink-0 font-mono tabular-nums text-muted-foreground">
                {relativeTime(t.last_run)}
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}

const RUN_STATUS_TONE: Record<string, Tone> = {
  // The queue writes `success`; the subagent registry writes `completed`.
  // Both land in this table, so both are mapped.
  success: 'good',
  completed: 'good',
  running: 'accent',
  failed: 'crit',
  timeout: 'crit',
  poisoned: 'crit',
}

function WorkersPanel({ workers }: { workers: WorkersState }) {
  const pool = workers.pool ?? { running: false }
  const poolTone: Tone = !workers.enabled || !pool.running ? 'idle' : pool.paused ? 'warn' : 'good'
  const poolWord = !workers.enabled ? 'disabled' : !pool.running ? 'stopped' : pool.paused ? 'paused' : 'running'
  // Only sources with something to say — 20 idle rows is not a status.
  const busy = workers.sources.filter(s => s.open > 0 || s.running > 0 || s.poisoned > 0)

  return (
    <Panel>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
          Worker pool
        </span>
        <span className="flex items-center gap-1.5">
          <span className={cn('h-1.5 w-1.5 rounded-full', TONE_FILL[poolTone])} />
          <span className={cn('text-[10px]', TONE_TEXT[poolTone])}>{poolWord}</span>
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div>
          <div className="text-[10px] text-muted-foreground">In flight</div>
          <div className={cn('font-mono text-lg leading-none',
            (pool.in_flight_count ?? 0) > 0 ? 'text-violet-400' : 'text-muted-foreground')}>
            {pool.in_flight_count ?? 0}
            <span className="text-[11px] text-muted-foreground/60">/{pool.slots ?? '—'}</span>
          </div>
        </div>
        <div>
          <div className="text-[10px] text-muted-foreground">Open</div>
          <div className={cn('font-mono text-lg leading-none',
            workers.open_total > 0 ? 'text-amber-400' : 'text-muted-foreground')}>
            {workers.open_total}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-muted-foreground">Poisoned</div>
          <div className={cn('font-mono text-lg leading-none',
            workers.poisoned_total > 0 ? 'text-rose-400' : 'text-muted-foreground')}>
            {workers.poisoned_total}
          </div>
        </div>
      </div>

      {busy.length > 0 && (
        <div className="mt-2.5 space-y-1 border-t border-border pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Sources</div>
          {busy.map(src => (
            <div key={src.name} className="flex items-center gap-2 text-[10px]">
              <span className={cn('h-1.5 w-1.5 flex-shrink-0 rounded-full',
                src.running > 0 ? 'animate-pulse bg-violet-400'
                  : src.poisoned > 0 ? 'bg-rose-400' : 'bg-amber-400')} />
              <span className="truncate text-foreground">{src.name}</span>
              <span className="ml-auto flex-shrink-0 font-mono tabular-nums text-muted-foreground">
                {src.running > 0 && <span className="text-violet-400">{src.running} run </span>}
                {src.open > 0 && <span className="text-amber-400">{src.open} open </span>}
                {src.poisoned > 0 && <span className="text-rose-400">{src.poisoned} bad</span>}
              </span>
            </div>
          ))}
        </div>
      )}

      {workers.recent_runs.length > 0 && (
        <div className="mt-2.5 space-y-1 border-t border-border pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Recent runs
          </div>
          {workers.recent_runs.map(r => (
            <div key={r.run_id} className="flex items-center gap-2 text-[10px]">
              <span className={cn('h-1.5 w-1.5 flex-shrink-0 rounded-full',
                TONE_FILL[RUN_STATUS_TONE[r.status] ?? 'idle'])} />
              <span className="truncate text-muted-foreground">{r.source}</span>
              <span className="truncate text-foreground/80">{r.summary}</span>
              <span className="ml-auto flex-shrink-0 font-mono tabular-nums text-muted-foreground">
                {r.duration_seconds == null ? '—' : duration(r.duration_seconds)}
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}

function BacklogPanel({ backlog }: { backlog: BacklogState }) {
  const maxOpen = Math.max(...backlog.by_board.map(b => b.open), 1)
  return (
    <Panel>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Backlog</span>
        <span className="text-[10px] text-muted-foreground">
          {backlog.open_total} open of {backlog.total}
        </span>
      </div>
      <StatusChips counts={backlog.by_status} />

      {backlog.by_board.length > 0 && (
        <div className="mt-2.5 space-y-1 border-t border-border pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Open by board
          </div>
          {backlog.by_board.filter(b => b.open > 0).map(b => (
            <div key={b.board} className="flex items-center gap-2">
              <span className="w-20 flex-shrink-0 truncate text-[10px] text-muted-foreground">
                {b.board}
              </span>
              <div className="h-2 flex-1 overflow-hidden rounded-[4px] bg-violet-400/10">
                <div
                  className="h-full rounded-[4px] bg-violet-400/60"
                  style={{ width: `${(b.open / maxOpen) * 100}%` }}
                />
              </div>
              <span className="w-10 flex-shrink-0 text-right font-mono text-[10px] tabular-nums text-muted-foreground">
                {b.open}
              </span>
            </div>
          ))}
        </div>
      )}

      {backlog.recent_open.length > 0 && (
        <div className="mt-2.5 space-y-1 border-t border-border pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Recently touched
          </div>
          {backlog.recent_open.map(t => (
            <div key={`${t.board}/${t.name}`} className="flex items-center gap-2 text-[10px]">
              <span className={cn('h-1.5 w-1.5 flex-shrink-0 rounded-full',
                TONE_FILL[STATUS_TONE[t.status] ?? 'idle'])} />
              <span className="truncate text-foreground">{t.name}</span>
              <span className="ml-auto flex-shrink-0 text-muted-foreground">
                {t.status.replace(/_/g, ' ')}
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}


// ── Page ───────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [snap, setSnap] = useState<DashboardSnapshot | null>(null)
  const [error, setError] = useState<string>('')
  const [stale, setStale] = useState(false)
  const inflight = useRef(false)

  useReportMcFocus('dashboard', null)

  const load = useCallback(async () => {
    // Skip if the previous poll is still out — a slow backend must not
    // build a queue of overlapping requests.
    if (inflight.current) return
    inflight.current = true
    try {
      setSnap(await dashboardApi.get())
      setError('')
      setStale(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStale(true)
    } finally {
      inflight.current = false
    }
  }, [])

  useEffect(() => {
    load()
    let timer = window.setInterval(load, POLL_MS)
    // Polling a hidden tab burns backend cycles for nobody. Stop on
    // hide, resume (with an immediate read) on show.
    const onVisibility = () => {
      window.clearInterval(timer)
      if (document.visibilityState === 'visible') {
        load()
        timer = window.setInterval(load, POLL_MS)
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [load])

  if (!snap) {
    return (
      <div className="flex-1 min-h-0 overflow-y-auto p-6 text-sm text-muted-foreground">
        {error ? `Dashboard unavailable: ${error}` : 'Loading dashboard…'}
      </div>
    )
  }

  const { host, vllm, primary, focus, agents, services, workers, autonomy, backlog, usage } = snap

  const engines = sectionOk<VllmEngine[]>(vllm) ? vllm : []
  const primaryEngine = engines.find(e => e.alias === 'primary') ?? engines[0]
  const totalRunning = engines.reduce((n, e) => n + (e.requests_running ?? 0), 0)
  const totalWaiting = engines.reduce((n, e) => n + (e.requests_waiting ?? 0), 0)

  const activeSubagents = sectionOk(agents) ? agents.subagents.active : []
  const recentSubagents = sectionOk(agents) ? agents.subagents.recent : []
  const bgTasks = sectionOk(agents) ? agents.background_tasks.active : []

  const unhealthy = sectionOk(services) ? services.unhealthy : []
  const autonomyFailed = sectionOk(autonomy) ? (autonomy.by_status.failed ?? 0) : 0
  const busy = sectionOk(primary) ? primary.busy : false

  // The hero: decode throughput on the primary engine. On a box whose
  // whole job is running one model, this is the pulse — it is non-zero
  // exactly when Lloyd is thinking.
  const decode = primaryEngine?.generation_tokens_per_s ?? null

  return (
    <div
      className={cn(
        // `main` is a flex column with overflow-hidden, so a plain block
        // here gets clipped at the fold. Claim the leftover height
        // (flex-1 + min-h-0) and scroll within it.
        'flex-1 min-h-0 overflow-y-auto space-y-5 p-4 md:p-6',
        stale && 'opacity-60 transition-opacity',
      )}
    >
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h1 className="text-lg font-bold text-foreground">Mission Control</h1>
        <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <span
            className={cn(
              'h-1.5 w-1.5 rounded-full',
              stale ? 'bg-amber-400' : busy ? 'animate-pulse bg-violet-400' : 'bg-emerald-400',
            )}
          />
          {stale ? 'reconnecting' : busy ? 'agent running' : 'idle'}
        </span>
        {sectionOk(host) && (
          <span className="text-[11px] text-muted-foreground/70">
            up {duration(host.uptime_seconds)}
          </span>
        )}
        {error && <span className="text-[11px] text-amber-400">{error}</span>}
      </header>

      {/* Hero + headline tiles */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8">
        <Panel className="col-span-2 sm:col-span-3 lg:col-span-2">
          <div className="text-[11px] text-muted-foreground">Decode throughput</div>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span
              className={cn(
                'text-5xl font-semibold leading-none',
                decode ? 'text-violet-400' : 'text-muted-foreground/50',
              )}
            >
              {decode == null ? '—' : compact(decode, 0)}
            </span>
            <span className="text-xs text-muted-foreground">tok/s</span>
          </div>
          <div className="mt-1.5 truncate text-[10px] text-muted-foreground/70">
            {primaryEngine?.model_name ?? 'no engine'}
          </div>
        </Panel>

        <StatTile
          label="Requests running"
          value={String(totalRunning)}
          tone={totalRunning > 0 ? 'accent' : 'idle'}
          sub={totalWaiting > 0 ? `${totalWaiting} waiting` : 'none queued'}
        />
        <StatTile
          label="Subagents"
          value={String(activeSubagents.length)}
          tone={activeSubagents.length > 0 ? 'accent' : 'idle'}
          sub={bgTasks.length > 0 ? `${bgTasks.length} bg task(s)` : 'no bg tasks'}
        />
        <StatTile
          label="Agent turns"
          value={sectionOk(primary) ? String(primary.running_count) : '—'}
          tone={busy ? 'accent' : 'idle'}
          sub={sectionOk(primary) ? `${primary.queued_count} queued` : undefined}
        />
        <StatTile
          label="Services"
          value={sectionOk(services) ? `${services.total - unhealthy.length}/${services.total}` : '—'}
          tone={unhealthy.length === 0 ? 'good' : 'crit'}
          sub={unhealthy.length ? `${unhealthy.length} unhealthy` : 'all healthy'}
        />
        <StatTile
          label="Autonomy"
          value={sectionOk(autonomy) ? String(autonomy.running_count) : '—'}
          tone={autonomyFailed > 0 ? 'crit' : sectionOk(autonomy) && autonomy.running_count > 0 ? 'accent' : 'idle'}
          sub={
            !sectionOk(autonomy) ? undefined
              : autonomyFailed > 0 ? `${autonomyFailed} failed`
              : `${autonomy.total} scheduled`
          }
        />
        <StatTile
          label="Backlog open"
          value={sectionOk(backlog) ? String(backlog.open_total) : '—'}
          tone={sectionOk(backlog) && backlog.open_total > 0 ? 'accent' : 'idle'}
          sub={sectionOk(backlog) ? `${backlog.total} total` : undefined}
        />
      </div>

      {/* Engines */}
      <Section title="vLLM engines" icon={Zap}>
        {sectionOk<VllmEngine[]>(vllm) ? (
          <div className="grid gap-3 lg:grid-cols-2">
            {vllm.map(e => <EngineCard key={e.alias} engine={e} />)}
          </div>
        ) : (
          <ErrorPanel what="Engine telemetry" error={vllm.error} />
        )}
      </Section>

      {/* Agent */}
      <Section title="Lloyd agent" icon={Bot}>
        <div className="grid gap-3 lg:grid-cols-2">
          {sectionOk(primary) ? (
            <Panel>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[10px]">
                <Kv label="Model" value={primary.model} />
                <Kv label="Context" value={compact(primary.context_length, 0)} />
                <Kv label="Max turns" value={String(primary.max_turns ?? '—')} />
                <Kv label="Thinking window" value={String(primary.preserve_thinking_iterations ?? '—')} />
              </div>
              <div className="mt-2 border-t border-border pt-2">
                {primary.sessions.length === 0 ? (
                  <div className="text-[11px] text-muted-foreground">No active turns.</div>
                ) : (
                  <div className="space-y-1.5">
                    {primary.sessions.map(s => (
                      <div key={s.session_id} className="flex items-center gap-2 text-[10px]">
                        <span
                          className={cn(
                            'h-1.5 w-1.5 flex-shrink-0 rounded-full',
                            s.running ? 'animate-pulse bg-violet-400' : 'bg-amber-400',
                          )}
                        />
                        <span className="truncate font-mono text-foreground">{s.session_id}</span>
                        <span className="text-muted-foreground">{s.source ?? 'queued'}</span>
                        <span className="ml-auto flex-shrink-0 text-muted-foreground">
                          {s.pending_user + s.pending_ambient > 0
                            ? `+${s.pending_user + s.pending_ambient} queued`
                            : s.running ? 'running' : 'pending'}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </Panel>
          ) : (
            <ErrorPanel what="Primary state" error={primary.error} />
          )}

          {/* Goal / plan / todos for the session in view */}
          {sectionOk(focus) && focus.session_id ? (
            <Panel>
              <div className="truncate text-[11px] text-foreground">
                {focus.preview || focus.session_id}
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
                <span className="font-mono">{focus.session_id}</span>
                {focus.platform && <span>{focus.platform}</span>}
                {focus.inner_voice && <span className="text-violet-400/80">inner voice</span>}
                {focus.plan_mode && <span className="text-amber-400/80">plan mode</span>}
                {(focus.plan_stages ?? 0) > 0 && <span>{focus.plan_stages} stages</span>}
              </div>
              {focus.goal && (
                <div className="mt-2 rounded-md border border-violet-400/25 bg-violet-400/5 px-2 py-1.5">
                  <div className="text-[10px] text-violet-400">Goal</div>
                  <div className="text-[11px] text-foreground">{focus.goal}</div>
                </div>
              )}
              {(focus.todos?.length ?? 0) > 0 && (
                <div className="mt-2 space-y-1 border-t border-border pt-2">
                  {focus.todos!.slice(0, 6).map((t, i) => (
                    <div key={i} className="flex items-start gap-1.5 text-[10px]">
                      {t.status === 'completed' ? (
                        <CheckCircle2 className="mt-px h-3 w-3 flex-shrink-0 text-emerald-400" />
                      ) : t.status === 'in_progress' ? (
                        <Activity className="mt-px h-3 w-3 flex-shrink-0 animate-pulse text-violet-400" />
                      ) : (
                        <MinusCircle className="mt-px h-3 w-3 flex-shrink-0 text-muted-foreground/50" />
                      )}
                      <span
                        className={cn(
                          'truncate',
                          t.status === 'completed'
                            ? 'text-muted-foreground/60 line-through'
                            : 'text-foreground',
                        )}
                      >
                        {t.status === 'in_progress' && t.activeForm ? t.activeForm : t.content}
                      </span>
                    </div>
                  ))}
                  {focus.todos!.length > 6 && (
                    <div className="text-[10px] text-muted-foreground/70">
                      +{focus.todos!.length - 6} more
                    </div>
                  )}
                </div>
              )}
            </Panel>
          ) : (
            <Panel className="flex items-center justify-center">
              <span className="text-[11px] text-muted-foreground">No session in focus.</span>
            </Panel>
          )}
        </div>
      </Section>

      {/* Subagents */}
      <Section
        title="Subagents & background tasks"
        icon={Layers}
        right={
          <span className="text-[10px] text-muted-foreground">
            {sectionOk(agents) ? `${agents.tools} tools` : ''}
          </span>
        }
      >
        {!sectionOk(agents) ? (
          <ErrorPanel what="Agent registry" error={agents.error} />
        ) : (
          <div className="grid gap-3 lg:grid-cols-2">
            <Panel>
              <div className="mb-2 text-[10px] uppercase tracking-wider text-muted-foreground">
                Active ({activeSubagents.length})
              </div>
              {activeSubagents.length === 0 ? (
                <div className="text-[11px] text-muted-foreground">No subagents running.</div>
              ) : (
                <div className="space-y-1.5">
                  {activeSubagents.map(r => <SubagentRow key={r.run_id} run={r} />)}
                </div>
              )}
              {bgTasks.length > 0 && (
                <div className="mt-3 space-y-1.5 border-t border-border pt-2">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Background bash ({bgTasks.length})
                  </div>
                  {bgTasks.map(t => (
                    <div key={t.task_id} className="flex items-center gap-2 text-[10px]">
                      <Terminal className="h-3 w-3 flex-shrink-0 text-violet-400" />
                      <span className="truncate text-foreground">{t.description}</span>
                      <span className="ml-auto flex-shrink-0 font-mono tabular-nums text-muted-foreground">
                        {t.elapsed_s.toFixed(0)}s
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
            <Panel>
              <div className="mb-2 text-[10px] uppercase tracking-wider text-muted-foreground">
                Recent
              </div>
              {recentSubagents.length === 0 ? (
                <div className="text-[11px] text-muted-foreground">Nothing finished yet.</div>
              ) : (
                <div className="space-y-1.5">
                  {recentSubagents.slice(0, 5).map(r => <SubagentRow key={r.run_id} run={r} />)}
                </div>
              )}
            </Panel>
          </div>
        )}
      </Section>

      {/* System */}
      <Section title="System" icon={Cpu}>
        {!sectionOk(host) ? (
          <ErrorPanel what="Host metrics" error={host.error} />
        ) : (
          <div className="space-y-3">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Panel>
                <Meter
                  label={`CPU · ${host.cpu.count} threads`}
                  value={host.cpu.percent / 100}
                  display={`${host.cpu.percent.toFixed(0)}%`}
                  sub={
                    host.cpu.load_average
                      ? `load ${host.cpu.load_average.map(l => l.toFixed(2)).join('  ')}`
                      : undefined
                  }
                />
              </Panel>
              <Panel>
                <Meter
                  label="Memory"
                  value={host.memory.percent / 100}
                  display={`${host.memory.percent.toFixed(0)}%`}
                  sub={`${bytes(host.memory.used_bytes)} / ${bytes(host.memory.total_bytes)}`}
                />
              </Panel>
              <Panel>
                <Meter
                  label="Swap"
                  value={host.swap.total_bytes ? host.swap.percent / 100 : 0}
                  display={host.swap.total_bytes ? `${host.swap.percent.toFixed(0)}%` : 'none'}
                  sub={host.swap.total_bytes ? `${bytes(host.swap.used_bytes)} / ${bytes(host.swap.total_bytes)}` : undefined}
                />
              </Panel>
              {host.disks.map(d => (
                <Panel key={d.path}>
                  <Meter
                    label={`Disk ${d.path}`}
                    value={d.percent / 100}
                    display={`${d.percent.toFixed(0)}%`}
                    sub={`${bytes(d.used_bytes)} / ${bytes(d.total_bytes)}`}
                  />
                </Panel>
              ))}
            </div>
            {host.gpus.length > 0 && (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {host.gpus.map(g => <GpuCard key={g.index} gpu={g} />)}
              </div>
            )}
          </div>
        )}
      </Section>

      {/* Services */}
      <Section
        title="Services"
        icon={Server}
        right={
          unhealthy.length > 0 ? (
            <span className="text-[10px] text-rose-400">{unhealthy.join(', ')}</span>
          ) : undefined
        }
      >
        {!sectionOk(services) ? (
          <ErrorPanel what="Service health" error={services.error} />
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {services.services.map(s => (
              <HealthPill key={s.id} health={s.health} label={s.name} />
            ))}
          </div>
        )}
      </Section>

      {/* Automation — what runs without being asked, and the work waiting */}
      <Section title="Automation & work" icon={Workflow}>
        <div className="grid gap-3 lg:grid-cols-3">
          {sectionOk(autonomy)
            ? <AutonomyPanel autonomy={autonomy} />
            : <ErrorPanel what="Autonomy tasks" error={autonomy.error} />}
          {sectionOk(workers)
            ? <WorkersPanel workers={workers} />
            : <ErrorPanel what="Worker pool" error={workers.error} />}
          {sectionOk(backlog)
            ? <BacklogPanel backlog={backlog} />
            : <ErrorPanel what="Backlog" error={backlog.error} />}
        </div>
      </Section>

      {/* Local token accounting */}
      <Section title="Tokens" icon={Gauge}>
        {!sectionOk(usage) ? (
          <ErrorPanel what="Token usage" error={usage.error} />
        ) : (
          <Panel>
            <div className="mb-2 flex items-baseline justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Tokens per day
              </span>
              <span className="text-[10px] text-muted-foreground">
                {compact(usage.last_24h?.input_tokens)} in / {compact(usage.last_24h?.output_tokens)} out (24h)
              </span>
            </div>
            <DailyTokenBars daily={usage.daily} />
          </Panel>
        )}
      </Section>
    </div>
  )
}

/** Seven days of token volume. One series, one hue, light→dark by
 *  magnitude — the max is direct-labelled rather than every bar. */
function DailyTokenBars({ daily }: { daily: UsageBucket[] }) {
  const rows = daily ?? []
  if (rows.length === 0) {
    return <div className="text-[11px] text-muted-foreground">No usage recorded.</div>
  }
  const totals = rows.map(r => r.input_tokens + r.output_tokens)
  const max = Math.max(...totals, 1)
  return (
    <div className="space-y-1">
      {rows.map((r, i) => {
        const total = totals[i]
        const isMax = total === max
        // The 7-day window spans 8 calendar dates, so a bare weekday name
        // appears twice. Day-of-month disambiguates without much noise.
        const d = new Date(`${r.bucket}T00:00:00`)
        const day = `${d.toLocaleDateString(undefined, { weekday: 'short' })} ${d.getDate()}`
        return (
          <div key={r.bucket} className="flex items-center gap-2">
            <span className="w-12 flex-shrink-0 text-[10px] text-muted-foreground">{day}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-[4px] bg-violet-400/10">
              <div
                className={cn('h-full rounded-[4px]', isMax ? 'bg-violet-400' : 'bg-violet-400/50')}
                style={{ width: `${(total / max) * 100}%` }}
              />
            </div>
            <span
              className={cn(
                'w-12 flex-shrink-0 text-right font-mono text-[10px] tabular-nums',
                isMax ? 'text-violet-400' : 'text-muted-foreground',
              )}
            >
              {compact(total)}
            </span>
          </div>
        )
      })}
    </div>
  )
}
