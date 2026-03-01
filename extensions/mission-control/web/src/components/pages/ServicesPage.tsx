import { useState, useEffect, useCallback, useRef } from "react";
import { Activity, Play, Square, RotateCcw, ChevronDown, Terminal, Cpu, HardDrive, Clock } from "lucide-react";
import { api, type ServiceStatus, type ServiceDetail } from "../../api";

export default function ServicesPage() {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [timestamp, setTimestamp] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ServiceDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await api.services();
      setServices(data.services);
      setTimestamp(data.timestamp);
    } catch {
      // keep stale data
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  // Fetch detail when a service is expanded
  const fetchDetail = useCallback(async (id: string) => {
    setDetailLoading(true);
    try {
      const data = await api.serviceDetail(id);
      setDetail(data);
    } catch {
      setDetail(null);
    }
    setDetailLoading(false);
  }, []);

  // Auto-refresh detail for expanded service
  useEffect(() => {
    if (!expandedId) return;
    fetchDetail(expandedId);
    const interval = setInterval(() => fetchDetail(expandedId), 5000);
    return () => clearInterval(interval);
  }, [expandedId, fetchDetail]);

  // Scroll log to bottom when logs update
  useEffect(() => {
    if (detail && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [detail?.logLines]);

  const toggleExpand = (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      setDetail(null);
    } else {
      setExpandedId(id);
    }
  };

  const handleAction = async (
    e: React.MouseEvent,
    serviceId: string,
    action: "start" | "stop" | "restart",
  ) => {
    e.stopPropagation();
    setActionLoading(`${serviceId}-${action}`);
    try {
      await api.serviceAction(serviceId, action);
      await new Promise((r) => setTimeout(r, 1500));
      await refresh();
      if (expandedId === serviceId) fetchDetail(serviceId);
    } catch {
      await refresh();
    }
    setActionLoading(null);
  };

  const healthDot = (health: string) => {
    switch (health) {
      case "healthy":
        return "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]";
      case "degraded":
        return "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.5)]";
      case "stopped":
        return "bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.4)]";
      default:
        return "bg-slate-500";
    }
  };

  const stateBadge = (state: string) => {
    switch (state) {
      case "active":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "failed":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      case "inactive":
        return "bg-slate-500/20 text-slate-400 border-slate-500/30";
      default:
        return "bg-slate-500/20 text-slate-400 border-slate-500/30";
    }
  };

  const healthyCount = services.filter((s) => s.health === "healthy").length;
  const totalCount = services.length;

  return (
    <div className="p-6 space-y-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-brand-400" />
          <h2 className="text-lg font-semibold">LLOYD Services</h2>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500">
            {healthyCount}/{totalCount} healthy
          </span>
          {timestamp && (
            <span className="text-[10px] text-slate-600 font-mono">
              {new Date(timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="text-sm text-slate-500 text-center py-8">
          Loading services...
        </div>
      )}

      {/* Service cards */}
      <div className="space-y-2">
        {services.map((svc) => {
          const isExpanded = expandedId === svc.id;
          return (
            <div key={svc.id} className="rounded-xl border border-surface-3/50 overflow-hidden">
              {/* Service row (clickable) */}
              <div
                onClick={() => toggleExpand(svc.id)}
                className={`bg-surface-1 px-5 py-4 flex items-center gap-4 cursor-pointer transition-colors ${
                  isExpanded
                    ? "border-b border-surface-3/50"
                    : "hover:border-surface-3/80"
                }`}
              >
                {/* Health indicator */}
                <div
                  className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${healthDot(svc.health)}`}
                />

                {/* Service info */}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-slate-200">
                    {svc.name}
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                    {svc.unit} &middot; :{svc.port}
                  </div>
                </div>

                {/* Status badge */}
                <span
                  className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${stateBadge(svc.systemdState)}`}
                >
                  {svc.systemdState}
                </span>

                {/* Port health */}
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                    svc.portHealthy
                      ? "text-emerald-400"
                      : "text-slate-600"
                  }`}
                  title={svc.portHealthy ? "Port responding" : "Port not responding"}
                >
                  :{svc.port}
                </span>

                {/* Action buttons */}
                <div className="flex gap-1">
                  <button
                    onClick={(e) => handleAction(e, svc.id, "start")}
                    disabled={actionLoading !== null}
                    title="Start"
                    className="p-1.5 rounded-lg text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-50"
                  >
                    <Play className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={(e) => handleAction(e, svc.id, "stop")}
                    disabled={actionLoading !== null}
                    title="Stop"
                    className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
                  >
                    <Square className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={(e) => handleAction(e, svc.id, "restart")}
                    disabled={actionLoading !== null}
                    title="Restart"
                    className={`p-1.5 rounded-lg text-slate-400 hover:text-amber-400 hover:bg-amber-500/10 transition-colors disabled:opacity-50 ${
                      actionLoading === `${svc.id}-restart` ? "animate-spin" : ""
                    }`}
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Expand chevron */}
                <ChevronDown
                  className={`w-4 h-4 text-slate-500 transition-transform ${
                    isExpanded ? "rotate-180" : ""
                  }`}
                />
              </div>

              {/* Expanded detail panel */}
              {isExpanded && (
                <div className="bg-surface-0 px-5 py-4 space-y-4">
                  {detailLoading && !detail ? (
                    <div className="text-xs text-slate-500 text-center py-4">
                      Loading service details...
                    </div>
                  ) : detail ? (
                    <>
                      {/* Stats row */}
                      <div className="flex flex-wrap gap-4">
                        {detail.pid && (
                          <StatChip icon={<Cpu className="w-3.5 h-3.5" />} label="PID" value={String(detail.pid)} />
                        )}
                        {detail.memory && (
                          <StatChip icon={<HardDrive className="w-3.5 h-3.5" />} label="Memory" value={detail.memory} />
                        )}
                        {detail.cpu && (
                          <StatChip icon={<Clock className="w-3.5 h-3.5" />} label="CPU" value={detail.cpu} />
                        )}
                        {detail.tasks && (
                          <StatChip icon={<Activity className="w-3.5 h-3.5" />} label="Tasks" value={detail.tasks} />
                        )}
                      </div>

                      {/* Active since */}
                      {detail.activeSince && (
                        <div className="text-[11px] text-slate-400 font-mono">
                          Active: {detail.activeSince}
                        </div>
                      )}

                      {/* Log lines */}
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Terminal className="w-3.5 h-3.5 text-slate-500" />
                          <span className="text-xs text-slate-400 font-medium">Recent Logs</span>
                        </div>
                        <div className="bg-black/40 rounded-lg p-3 max-h-72 overflow-y-auto font-mono text-[11px] leading-relaxed border border-surface-3/30">
                          {detail.logLines.length > 0 ? (
                            detail.logLines.map((line, i) => (
                              <div
                                key={i}
                                className={`whitespace-pre-wrap break-all ${
                                  line.match(/error|fail|panic|critical/i)
                                    ? "text-red-400"
                                    : line.match(/warn/i)
                                      ? "text-amber-400"
                                      : "text-slate-400"
                                }`}
                              >
                                {line}
                              </div>
                            ))
                          ) : (
                            <div className="text-slate-600 italic">No log lines available</div>
                          )}
                          <div ref={logEndRef} />
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="text-xs text-red-400 text-center py-4">
                      Failed to load service details
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StatChip({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 bg-surface-1 rounded-lg px-3 py-2 border border-surface-3/30">
      <span className="text-brand-400">{icon}</span>
      <div>
        <div className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</div>
        <div className="text-xs text-slate-200 font-mono">{value}</div>
      </div>
    </div>
  );
}
