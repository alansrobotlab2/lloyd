import { useEffect, useState, useCallback } from "react";
import { Activity, X } from "lucide-react";
import { api, AgentStatusData, SubagentRunInfo } from "../../api";
import AgentDeskRoom from "../AgentDeskRoom";

// ── Helpers ─────────────────────────────────────────────────────────────

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

const STATE_COLORS: Record<string, string> = {
  idle: "bg-slate-500",
  processing: "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]",
  waiting: "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.5)]",
};

const STATE_BADGES: Record<string, string> = {
  idle: "bg-slate-500/20 text-slate-400 border-slate-500/30",
  processing: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  waiting: "bg-amber-500/20 text-amber-400 border-amber-500/30",
};

function outcomeStr(outcome: unknown): string | undefined {
  if (typeof outcome === "string") return outcome;
  if (typeof outcome === "object" && outcome !== null && "status" in outcome)
    return String((outcome as { status: unknown }).status);
  return undefined;
}

function outcomeBadgeClass(outcome?: string): string {
  switch (outcome) {
    case "ok": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
    case "error": return "bg-red-500/20 text-red-400 border-red-500/30";
    case "timeout": return "bg-amber-500/20 text-amber-400 border-amber-500/30";
    default: return "bg-slate-500/20 text-slate-400 border-slate-500/30";
  }
}

// ── Agent Live Status Banner ─────────────────────────────────────────────

function AgentLiveStatus({ status }: { status: AgentStatusData | null }) {
  if (!status) {
    return (
      <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50 animate-pulse">
        <div className="h-4 bg-surface-2 rounded w-32" />
      </div>
    );
  }

  const { mainAgent, activity } = status;
  const stateColor = STATE_COLORS[mainAgent.state] ?? STATE_COLORS.idle;
  const stateBadge = STATE_BADGES[mainAgent.state] ?? STATE_BADGES.idle;

  return (
    <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50">
      <div className="flex items-center gap-3">
        <div className="relative flex-shrink-0">
          <div className={`w-3 h-3 rounded-full ${stateColor}`} />
          {mainAgent.state === "processing" && (
            <div className="absolute inset-0 w-3 h-3 rounded-full bg-emerald-400 animate-ping opacity-40" />
          )}
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${stateBadge}`}>
          {mainAgent.state}
        </span>
        <div className="flex-1 text-xs text-slate-400">
          {activity.label}
          {activity.detail && (
            <span className="text-slate-500 font-mono ml-1.5">{activity.detail}</span>
          )}
        </div>
        {activity.type !== "idle" && (
          <span className="text-[10px] text-slate-600 font-mono">
            {formatElapsed(activity.elapsedMs)}
          </span>
        )}
        {mainAgent.queueDepth > 0 && (
          <span className="text-[10px] text-amber-400 font-mono">
            {mainAgent.queueDepth} queued
          </span>
        )}
      </div>
      {status.heartbeat && (status.heartbeat.active > 0 || status.heartbeat.waiting > 0 || status.heartbeat.queued > 0) && (
        <div className="flex gap-4 mt-2 text-[10px] text-slate-500 font-mono">
          <span>{status.heartbeat.active} active</span>
          <span>{status.heartbeat.waiting} waiting</span>
          <span>{status.heartbeat.queued} queued</span>
        </div>
      )}
    </div>
  );
}

// ── Active Subagent Card (with kill button) ──────────────────────────────

function ActiveSubagentCard({
  run,
  onKill,
}: {
  run: SubagentRunInfo;
  onKill: (sessionKey: string) => void;
}) {
  const [killing, setKilling] = useState(false);
  const elapsed = Date.now() - (run.startedAt ?? run.createdAt);

  const handleKill = async () => {
    setKilling(true);
    try {
      await onKill(run.childSessionKey);
    } finally {
      setKilling(false);
    }
  };

  return (
    <div className="bg-surface-1 rounded-xl px-4 py-3 border border-emerald-500/30 transition-colors">
      <div className="flex items-start gap-3">
        <div className="mt-1 flex-shrink-0">
          <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs text-slate-300 line-clamp-2">{run.task}</div>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            {run.label && <span className="text-[10px] text-slate-500 font-mono">{run.label}</span>}
            {run.model && <span className="text-[10px] text-slate-500 font-mono">{run.model}</span>}
            {run.spawnMode && <span className="text-[10px] text-slate-600 font-mono">{run.spawnMode}</span>}
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="text-[10px] text-slate-500 font-mono">{formatElapsed(elapsed)}</div>
          <button
            onClick={handleKill}
            disabled={killing}
            title="Abort subagent"
            className="p-1 rounded-md text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Completed Run Card ───────────────────────────────────────────────────

function CompletedRunCard({ run }: { run: SubagentRunInfo }) {
  return (
    <div className="bg-surface-1 rounded-xl px-4 py-3 border border-surface-3/50">
      <div className="flex items-start gap-3">
        <div className="mt-1 flex-shrink-0">
          <div className={`w-2 h-2 rounded-full ${outcomeStr(run.outcome) === "ok" ? "bg-emerald-400" : "bg-red-400"}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs text-slate-300 line-clamp-2">{run.task}</div>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            {run.label && <span className="text-[10px] text-slate-500 font-mono">{run.label}</span>}
            {run.model && <span className="text-[10px] text-slate-500 font-mono">{run.model}</span>}
          </div>
        </div>
        <div className="text-right flex-shrink-0">
          {run.durationMs != null && (
            <div className="text-[10px] text-slate-500 font-mono">{formatElapsed(run.durationMs)}</div>
          )}
          {outcomeStr(run.outcome) && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono border mt-0.5 inline-block ${outcomeBadgeClass(outcomeStr(run.outcome))}`}>
              {outcomeStr(run.outcome)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Activity Feed (from recent completed) ───────────────────────────────

function ActivityFeed({ runs }: { runs: SubagentRunInfo[] }) {
  if (runs.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic py-4 text-center">No recent activity</div>
    );
  }

  return (
    <div className="space-y-2">
      {runs.map((run) => {
        const ago = run.endedAt ? Date.now() - run.endedAt : null;
        return (
          <div key={run.runId} className="flex items-start gap-3 py-2 border-b border-surface-3/20 last:border-0">
            <div className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${outcomeStr(run.outcome) === "ok" ? "bg-emerald-400" : "bg-red-400"}`} />
            <div className="flex-1 min-w-0">
              <div className="text-xs text-slate-300 line-clamp-1">{run.task}</div>
              {run.label && <div className="text-[10px] text-slate-500 font-mono">{run.label}</div>}
            </div>
            <div className="text-right flex-shrink-0 space-y-0.5">
              {ago != null && (
                <div className="text-[10px] text-slate-600 font-mono">{formatElapsed(ago)} ago</div>
              )}
              <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono border inline-block ${outcomeBadgeClass(outcomeStr(run.outcome))}`}>
                {outcomeStr(run.outcome) ?? "?"}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────

export default function ActivityPage() {
  const [status, setStatus] = useState<AgentStatusData | null>(null);

  const refresh = useCallback(() => {
    api.agentStatus().then(setStatus).catch(console.error);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Adaptive polling: 1.5s when active, 3s when idle
  useEffect(() => {
    const isActive =
      status?.mainAgent.state !== "idle" ||
      (status?.subagents.active.length ?? 0) > 0;

    const interval = setInterval(refresh, isActive ? 1500 : 3000);
    return () => clearInterval(interval);
  }, [status, refresh]);

  const handleKill = useCallback(async (sessionKey: string) => {
    try {
      await api.subagentAbort(sessionKey);
      setTimeout(refresh, 500);
    } catch (err) {
      console.error("Failed to abort subagent:", err);
    }
  }, [refresh]);

  const active = status?.subagents.active ?? [];
  const recentCompleted = status?.subagents.recentCompleted ?? [];

  return (
    <div className="p-6 space-y-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Activity className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Activity</h2>
        {active.length > 0 && (
          <span className="text-xs text-emerald-400 font-mono">
            {active.length} active
          </span>
        )}
      </div>

      {/* Agent Desk Room */}
      <AgentDeskRoom activeAgents={active} />

      {/* Main Agent Status */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Main Agent
        </div>
        <AgentLiveStatus status={status} />
      </div>

      {/* Active Subagents */}
      <div className="space-y-2">
        <div className="text-[10px] uppercase tracking-wider text-emerald-400 font-medium flex items-center gap-1.5">
          {active.length > 0 && (
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          )}
          Active Subagents ({active.length})
        </div>
        {active.length === 0 ? (
          <div className="text-xs text-slate-500 italic py-2 text-center">No active subagents</div>
        ) : (
          active.map((run) => (
            <ActiveSubagentCard key={run.runId} run={run} onKill={handleKill} />
          ))
        )}
      </div>

      {/* Recent Completed */}
      <div className="space-y-2">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Recent Completed ({recentCompleted.length})
        </div>
        {recentCompleted.length === 0 ? (
          <div className="text-xs text-slate-500 italic py-2 text-center">None yet</div>
        ) : (
          recentCompleted.map((run) => (
            <CompletedRunCard key={run.runId} run={run} />
          ))
        )}
      </div>

      {/* Activity Feed */}
      <div className="space-y-2">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Activity Feed
        </div>
        <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50">
          <ActivityFeed runs={recentCompleted} />
        </div>
      </div>
    </div>
  );
}
