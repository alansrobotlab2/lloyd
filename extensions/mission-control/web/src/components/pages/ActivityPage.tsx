/**
 * ActivityPage — Real-time activity dashboard showing agent status,
 * subagent runs, gateway sessions, and Claude Code instances.
 */
import { useEffect, useState, useCallback, useRef } from "react";
import { Activity, X, Cpu, Square, Radio, Filter, ChevronDown, ChevronRight } from "lucide-react";
import { api, AgentStatusData, SubagentRunInfo, CcInstanceInfo, CcInstanceMessage, GatewaySessionInfo } from "../../api";
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

/** Extract agent id from label or childSessionKey — mirrors AgentDeskRoom logic */
function agentIdFrom(run: SubagentRunInfo): string | null {
  if (run.label) {
    const id = run.label.split(":")[0].trim().toLowerCase();
    if (id) return id;
  }
  if (run.childSessionKey) {
    const parts = run.childSessionKey.split(":");
    if (parts.length >= 2 && parts[0] === "agent") return parts[1];
  }
  return null;
}

// ── Active Subagent Card (with kill button) ──────────────────────────────

function ActiveSubagentCard({
  run,
  onKill,
  onNavigate,
}: {
  run: SubagentRunInfo;
  onKill: (sessionKey: string) => void;
  onNavigate?: (agentId: string) => void;
}) {
  const [killing, setKilling] = useState(false);
  const elapsed = Date.now() - (run.startedAt ?? run.createdAt);
  const agentId = agentIdFrom(run);

  const handleKill = async () => {
    setKilling(true);
    try {
      await onKill(run.childSessionKey);
    } finally {
      setKilling(false);
    }
  };

  return (
    <div
      className={"bg-surface-1 rounded-xl px-4 py-3 border border-emerald-500/30 transition-colors" + (onNavigate && agentId ? " cursor-pointer hover:border-emerald-400/50" : "")}
      onClick={onNavigate && agentId ? () => onNavigate(agentId) : undefined}
    >
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
            onClick={(e) => { e.stopPropagation(); handleKill(); }}
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

// ── Claude Code Instance Card ────────────────────────────────────────────

const CC_STATUS_COLORS: Record<string, string> = {
  running: "bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.5)]",
  complete: "bg-emerald-400",
  error: "bg-red-400",
  aborted: "bg-slate-500",
};

const CC_STATUS_BADGES: Record<string, string> = {
  running: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  complete: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  error: "bg-red-500/20 text-red-400 border-red-500/30",
  aborted: "bg-slate-500/20 text-slate-400 border-slate-500/30",
};

function CcInstanceCard({
  instance,
  onAbort,
}: {
  instance: CcInstanceInfo;
  onAbort: (id: string) => void;
}) {
  const [aborting, setAborting] = useState(false);
  const statusColor = CC_STATUS_COLORS[instance.status] ?? CC_STATUS_COLORS.running;
  const statusBadge = CC_STATUS_BADGES[instance.status] ?? CC_STATUS_BADGES.running;

  const handleAbort = async () => {
    setAborting(true);
    try { await onAbort(instance.id); } finally { setAborting(false); }
  };

  const agentId = instance.type === "orchestrate" ? "orchestrator" : (instance.agent || "unknown");

  return (
    <div className="bg-surface-1 rounded-xl px-4 py-3 border border-cyan-500/30">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex-shrink-0 relative">
          <img
            src={`/api/mc/agent-avatar?id=${agentId}`}
            alt={agentId}
            className="w-8 h-8 rounded-lg object-cover"
            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
          />
          <div className={`absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border border-surface-1 ${statusColor}`} />
          {instance.status === "running" && (
            <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping opacity-30" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs text-slate-300 line-clamp-2">{instance.task}</div>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono border ${statusBadge}`}>
              {instance.type === "orchestrate" ? `pipeline:${instance.pipeline || "custom"}` : agentId}
            </span>
            <span className="text-[10px] text-slate-500 font-mono">
              {instance.turns} turns
            </span>
            {instance.costUsd > 0 && (
              <span className="text-[10px] text-slate-500 font-mono">
                ${instance.costUsd.toFixed(2)}
              </span>
            )}
          </div>
          {instance.activity && instance.status === "running" && (
            <div className="text-[10px] text-cyan-400/70 font-mono mt-1 truncate">
              {instance.activity}
            </div>
          )}
          {instance.error && (
            <div className="text-[10px] text-red-400/70 font-mono mt-1 truncate">
              {instance.error}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="text-[10px] text-slate-500 font-mono">
            {formatElapsed(instance.elapsedMs)}
          </div>
          {instance.status === "running" && (
            <button
              onClick={handleAbort}
              disabled={aborting}
              title="Abort instance"
              className="p-1 rounded-md text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
            >
              <Square className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Subagent Dispatch Info (parsed from log messages) ─────────────────

export interface SubagentDispatch {
  agent: string;
  startTs: number;
  endTs?: number;
  status: "running" | "complete";
}

export function parseSubagentDispatches(messages: CcInstanceMessage[]): SubagentDispatch[] {
  const dispatches: SubagentDispatch[] = [];
  let currentOpen: { agent: string; startTs: number } | null = null;

  for (const msg of messages) {
    if (msg.type === "subagent_start" && msg.agent) {
      // Close previous open dispatch (implicit end for sequential dispatches)
      if (currentOpen) {
        dispatches.push({
          agent: currentOpen.agent,
          startTs: currentOpen.startTs,
          endTs: msg.ts,
          status: "complete",
        });
      }
      currentOpen = { agent: msg.agent, startTs: msg.ts };
    } else if (msg.type === "subagent_end") {
      if (currentOpen) {
        dispatches.push({
          agent: currentOpen.agent,
          startTs: currentOpen.startTs,
          endTs: msg.ts,
          status: "complete",
        });
        currentOpen = null;
      }
    }
  }

  // Remaining open dispatch is still running
  if (currentOpen) {
    dispatches.push({
      agent: currentOpen.agent,
      startTs: currentOpen.startTs,
      status: "running",
    });
  }

  // Sort: running first, then by startTs desc
  dispatches.sort((a, b) => {
    if (a.status === "running" && b.status !== "running") return -1;
    if (b.status === "running" && a.status !== "running") return 1;
    return b.startTs - a.startTs;
  });

  return dispatches;
}

function SubagentDispatchRow({
  dispatch,
  messages,
}: {
  dispatch: SubagentDispatch;
  messages: CcInstanceMessage[];
}) {
  const [expanded, setExpanded] = useState(false);
  const duration = dispatch.endTs
    ? dispatch.endTs - dispatch.startTs
    : Date.now() - dispatch.startTs;

  // Filter messages belonging to this dispatch window
  const dispatchMessages = messages.filter(
    (m) =>
      m.ts >= dispatch.startTs &&
      (dispatch.endTs ? m.ts <= dispatch.endTs : true) &&
      (m.agent === dispatch.agent || m.type === "tool_use" || m.type === "tool_result" || m.type === "text") &&
      (m.agent === dispatch.agent || !m.agent)
  );

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left flex items-center gap-2 py-1.5 px-2 rounded-lg hover:bg-surface-2/30 transition-colors group"
      >
        <div className="flex-shrink-0 relative">
          <div
            className={`w-2 h-2 rounded-full ${
              dispatch.status === "running"
                ? "bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.5)]"
                : "bg-emerald-400"
            }`}
          />
          {dispatch.status === "running" && (
            <div className="absolute inset-0 w-2 h-2 rounded-full bg-cyan-400 animate-ping opacity-30" />
          )}
        </div>
        <img
          src={`/api/mc/agent-avatar?id=${dispatch.agent}`}
          alt={dispatch.agent}
          className="w-5 h-5 rounded-md object-cover flex-shrink-0"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
        <span
          className={`text-[10px] font-mono flex-shrink-0 ${
            dispatch.status === "running" ? "text-cyan-400" : "text-emerald-400"
          }`}
        >
          {dispatch.agent}
        </span>
        <span className="text-[10px] text-slate-600 font-mono flex-shrink-0">
          {formatElapsed(duration)}
        </span>
        <div className="flex-1" />
        {expanded ? (
          <ChevronDown className="w-3 h-3 text-slate-600 group-hover:text-slate-400" />
        ) : (
          <ChevronRight className="w-3 h-3 text-slate-600 group-hover:text-slate-400" />
        )}
      </button>

      {expanded && dispatchMessages.length > 0 && (
        <div className="ml-4 pl-3 border-l border-surface-3/30 mt-1 mb-2 space-y-0.5 max-h-48 overflow-y-auto">
          {dispatchMessages.map((msg, i) => (
            <div key={i} className="flex gap-2 py-0.5">
              <span
                className={`text-[10px] font-mono w-8 flex-shrink-0 ${
                  msg.type === "tool_use"
                    ? "text-blue-400"
                    : msg.type === "error"
                    ? "text-red-400"
                    : msg.type === "task_progress"
                    ? "text-amber-400"
                    : "text-slate-500"
                }`}
              >
                {msg.type === "tool_use"
                  ? "tool"
                  : msg.type === "error"
                  ? "err"
                  : msg.type === "task_progress"
                  ? "task"
                  : "txt"}
              </span>
              <span className="text-[10px] text-slate-400 truncate flex-1">
                {msg.content}
              </span>
              <span className="text-[10px] text-slate-600 font-mono flex-shrink-0">
                {new Date(msg.ts).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CcInstanceCardWithSubagents({
  instance,
  onAbort,
}: {
  instance: CcInstanceInfo;
  onAbort: (id: string) => void;
}) {
  const [messages, setMessages] = useState<CcInstanceMessage[]>([]);
  const [dispatches, setDispatches] = useState<SubagentDispatch[]>([]);
  const fetchedOnce = useRef(false);

  useEffect(() => {
    const load = () => {
      api
        .ccInstanceLog(instance.id, 100)
        .then((d) => {
          setMessages(d.messages);
          setDispatches(parseSubagentDispatches(d.messages));
          fetchedOnce.current = true;
        })
        .catch(() => {});
    };

    load();

    if (instance.status === "running") {
      const interval = setInterval(load, 2000);
      return () => clearInterval(interval);
    }
  }, [instance.id, instance.status]);

  return (
    <div>
      <CcInstanceCard instance={instance} onAbort={onAbort} />
      {dispatches.length > 0 && (
        <div className="ml-6 mt-1 pl-3 border-l-2 border-cyan-500/20 space-y-0.5">
          {dispatches.map((d, i) => (
            <SubagentDispatchRow
              key={`${d.agent}-${d.startTs}-${i}`}
              dispatch={d}
              messages={messages}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Gateway Session Card ──────────────────────────────────────────────

const GW_STATE_COLORS: Record<string, string> = {
  idle: "bg-slate-500",
  processing: "bg-violet-400 shadow-[0_0_6px_rgba(167,139,250,0.5)]",
  waiting: "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.5)]",
};

const GW_STATE_BADGES: Record<string, string> = {
  idle: "bg-slate-500/20 text-slate-400 border-slate-500/30",
  processing: "bg-violet-500/20 text-violet-400 border-violet-500/30",
  waiting: "bg-amber-500/20 text-amber-400 border-amber-500/30",
};

function GatewaySessionCard({
  session,
  onNavigate,
}: {
  session: GatewaySessionInfo;
  onNavigate?: (agentId: string) => void;
}) {
  const stateColor = GW_STATE_COLORS[session.state] ?? GW_STATE_COLORS.idle;
  const stateBadge = GW_STATE_BADGES[session.state] ?? GW_STATE_BADGES.idle;

  return (
    <div
      className={"bg-surface-1 rounded-xl px-4 py-3 border transition-colors" +
        (session.state !== "idle" ? " border-violet-500/30" : " border-surface-3/50") +
        (onNavigate ? " cursor-pointer hover:border-violet-400/50" : "")}
      onClick={onNavigate ? () => onNavigate(session.agentId) : undefined}
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex-shrink-0 relative">
          <img
            src={`/api/mc/agent-avatar?id=${session.agentId}`}
            alt={session.agentId}
            className="w-8 h-8 rounded-lg object-cover"
            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
          />
          <div className={`absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border border-surface-1 ${stateColor}`} />
          {session.state === "processing" && (
            <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-violet-400 animate-ping opacity-30" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-300 font-medium">{session.agentId}</span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono border ${stateBadge}`}>
              {session.state}
            </span>
          </div>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            <span className="text-[10px] text-slate-500 font-mono">{session.sessionKey}</span>
          </div>
        </div>
        <div className="text-right flex-shrink-0">
          <div className="text-[10px] text-slate-500 font-mono">{formatElapsed(session.elapsedMs)} ago</div>
          <span className="text-[10px] px-1.5 py-0.5 rounded-full font-mono border bg-violet-500/10 text-violet-400 border-violet-500/20 mt-0.5 inline-block">
            gateway
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Source Filter Toggle ─────────────────────────────────────────────

type SourceFilter = "all" | "board" | "gateway" | "sdk";

function SourceFilterBar({
  filter,
  onChange,
  counts,
}: {
  filter: SourceFilter;
  onChange: (f: SourceFilter) => void;
  counts: { board: number; gateway: number; sdk: number };
}) {
  const items: { id: SourceFilter; label: string; count: number; color: string }[] = [
    { id: "all", label: "All", count: counts.board + counts.gateway + counts.sdk, color: "text-slate-400 border-slate-500/30" },
    { id: "board", label: "Board", count: counts.board, color: "text-emerald-400 border-emerald-500/30" },
    { id: "gateway", label: "Gateway", count: counts.gateway, color: "text-violet-400 border-violet-500/30" },
    { id: "sdk", label: "SDK", count: counts.sdk, color: "text-cyan-400 border-cyan-500/30" },
  ];

  return (
    <div className="flex items-center gap-1.5">
      <Filter className="w-3 h-3 text-slate-500" />
      {items.map((item) => (
        <button
          key={item.id}
          onClick={() => onChange(item.id)}
          className={`text-[10px] px-2 py-0.5 rounded-full font-mono border transition-colors ${
            filter === item.id
              ? `${item.color} bg-white/5`
              : "text-slate-600 border-surface-3/30 hover:text-slate-400"
          }`}
        >
          {item.label} {item.count > 0 ? `(${item.count})` : ""}
        </button>
      ))}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────

export default function ActivityPage({ onNavigateToAgent }: { onNavigateToAgent?: (agentId: string) => void } = {}) {
  const [status, setStatus] = useState<AgentStatusData | null>(null);
  const [ccInstances, setCcInstances] = useState<CcInstanceInfo[]>([]);
  const [gatewaySessions, setGatewaySessions] = useState<GatewaySessionInfo[]>([]);
  const [gwSubagentRuns, setGwSubagentRuns] = useState<SubagentRunInfo[]>([]);
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [activeDispatches, setActiveDispatches] = useState<SubagentDispatch[]>([]);

  const refresh = useCallback(() => {
    api.agentStatus().then(setStatus).catch(console.error);
    api.ccInstances().then((d) => {
      setCcInstances(d.instances);
      // Collect running subagent dispatches from orchestrators
      const orchestrators = d.instances.filter(i => i.type === "orchestrate" && i.status === "running");
      if (orchestrators.length > 0) {
        Promise.all(orchestrators.map(inst =>
          api.ccInstanceLog(inst.id, 100).then(log => parseSubagentDispatches(log.messages)).catch(() => [])
        )).then(results => {
          setActiveDispatches(results.flat().filter(d => d.status === "running"));
        });
      } else {
        setActiveDispatches([]);
      }
    }).catch(() => { setCcInstances([]); setActiveDispatches([]); });
    api.gatewaySessions().then((d) => {
      setGatewaySessions(d.sessions);
      setGwSubagentRuns(d.subagentRuns);
    }).catch(() => {
      setGatewaySessions([]);
      setGwSubagentRuns([]);
    });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Adaptive polling: 3s when active, 10s when idle — only when tab is visible
  useEffect(() => {
    const hasCcRunning = ccInstances.some((i) => i.status === "running");
    const hasGwActive = gatewaySessions.some((s) => s.state !== "idle");
    const isActive =
      status?.mainAgent.state !== "idle" ||
      (status?.subagents.active.length ?? 0) > 0 ||
      hasCcRunning ||
      hasGwActive;

    let interval: ReturnType<typeof setInterval> | null = null;

    const start = () => {
      if (interval) return;
      interval = setInterval(refresh, isActive ? 3000 : 10_000);
    };
    const stop = () => {
      if (interval) { clearInterval(interval); interval = null; }
    };

    // Only poll when the page/tab is visible
    const onVisibility = () => {
      if (document.hidden) stop(); else { refresh(); start(); }
    };
    document.addEventListener("visibilitychange", onVisibility);

    if (!document.hidden) start();
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [status, ccInstances, gatewaySessions, refresh]);

  const handleKill = useCallback(async (sessionKey: string) => {
    try {
      await api.subagentAbort(sessionKey);
      setTimeout(refresh, 500);
    } catch (err) {
      console.error("Failed to abort subagent:", err);
    }
  }, [refresh]);

  const handleCcAbort = useCallback(async (id: string) => {
    try {
      await api.ccInstanceAbort(id);
      setTimeout(refresh, 500);
    } catch (err) {
      console.error("Failed to abort CC instance:", err);
    }
  }, [refresh]);

  const active = status?.subagents.active ?? [];
  const recentCompleted = status?.subagents.recentCompleted ?? [];
  const ccRunning = ccInstances.filter((i) => i.status === "running");
  const ccCompleted = ccInstances.filter((i) => i.status !== "running");
  const gwActive = gatewaySessions.filter((s) => s.state !== "idle");
  const gwIdle = gatewaySessions.filter((s) => s.state === "idle");

  // Counts for filter bar
  const filterCounts = {
    board: active.length + recentCompleted.length,
    gateway: gatewaySessions.length + gwSubagentRuns.length,
    sdk: ccInstances.length,
  };

  // Total active count across all sources
  const totalActive = active.length + gwActive.length + ccRunning.length;

  return (
    <div className="p-6 space-y-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-brand-400" />
          <h2 className="text-lg font-semibold">Activity</h2>
          {totalActive > 0 && (
            <span className="text-xs text-emerald-400 font-mono">
              {totalActive} active
            </span>
          )}
        </div>
        <SourceFilterBar filter={sourceFilter} onChange={setSourceFilter} counts={filterCounts} />
      </div>

      {/* Agent Desk Room — shows both legacy subagents and CC instances */}
      <AgentDeskRoom activeAgents={active} ccInstances={ccInstances} activeDispatches={activeDispatches} onAgentClick={onNavigateToAgent} />

      {/* Main Agent Status */}
      {(sourceFilter === "all" || sourceFilter === "board") && (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
            Main Agent
          </div>
          <AgentLiveStatus status={status} />
        </div>
      )}

      {/* Active Subagents (board-level) */}
      {(sourceFilter === "all" || sourceFilter === "board") && (
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
              <ActiveSubagentCard key={run.runId} run={run} onKill={handleKill} onNavigate={onNavigateToAgent} />
            ))
          )}
        </div>
      )}

      {/* Gateway Sessions */}
      {(sourceFilter === "all" || sourceFilter === "gateway") && (
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-wider text-violet-400 font-medium flex items-center gap-1.5">
            {gwActive.length > 0 && (
              <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
            )}
            <Radio className="w-3 h-3" />
            Gateway Sessions ({gwActive.length} active{gwIdle.length > 0 ? `, ${gwIdle.length} idle` : ""})
          </div>
          {gatewaySessions.length === 0 ? (
            <div className="text-xs text-slate-500 italic py-2 text-center">No gateway sessions</div>
          ) : (
            <>
              {gwActive.map((session) => (
                <GatewaySessionCard key={session.sessionKey} session={session} onNavigate={onNavigateToAgent} />
              ))}
              {gwIdle.slice(0, 5).map((session) => (
                <GatewaySessionCard key={session.sessionKey} session={session} onNavigate={onNavigateToAgent} />
              ))}
            </>
          )}

          {/* Gateway subagent runs (lifecycle history) */}
          {gwSubagentRuns.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-violet-400/60 font-medium">
                Gateway Run History ({gwSubagentRuns.length})
              </div>
              {gwSubagentRuns.filter((r) => !r.endedAt).map((run) => (
                <ActiveSubagentCard key={run.runId} run={run} onKill={handleKill} onNavigate={onNavigateToAgent} />
              ))}
              {gwSubagentRuns.filter((r) => !!r.endedAt).slice(0, 8).map((run) => (
                <CompletedRunCard key={run.runId} run={run} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Claude Code Instances */}
      {(sourceFilter === "all" || sourceFilter === "sdk") && ccInstances.length > 0 && (
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-wider text-cyan-400 font-medium flex items-center gap-1.5">
            {ccRunning.length > 0 && (
              <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            )}
            <Cpu className="w-3 h-3" />
            Claude Code Instances ({ccRunning.length} running{ccCompleted.length > 0 ? `, ${ccCompleted.length} done` : ""})
          </div>
          {ccRunning.map((inst) =>
            inst.type === "orchestrate" ? (
              <CcInstanceCardWithSubagents key={inst.id} instance={inst} onAbort={handleCcAbort} />
            ) : (
              <CcInstanceCard key={inst.id} instance={inst} onAbort={handleCcAbort} />
            )
          )}
          {ccCompleted.slice(0, 5).map((inst) =>
            inst.type === "orchestrate" ? (
              <CcInstanceCardWithSubagents key={inst.id} instance={inst} onAbort={handleCcAbort} />
            ) : (
              <CcInstanceCard key={inst.id} instance={inst} onAbort={handleCcAbort} />
            )
          )}
        </div>
      )}

      {/* Recent Completed (board-level) */}
      {(sourceFilter === "all" || sourceFilter === "board") && (
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
      )}

      {/* Activity Feed */}
      <div className="space-y-2">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Activity Feed
        </div>
        <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50">
          <ActivityFeed runs={[...recentCompleted, ...gwSubagentRuns.filter((r) => !!r.endedAt)].sort((a, b) => (b.endedAt ?? 0) - (a.endedAt ?? 0)).slice(0, 15)} />
        </div>
      </div>
    </div>
  );
}
