import { useState, useEffect, useCallback, useRef } from "react";
import { Activity, Play, Square, RotateCcw, RefreshCw, ChevronDown, Terminal, Cpu, HardDrive, Clock, AlertTriangle } from "lucide-react";
import { api, type ServiceStatus, type ServiceDetail, type LloydServiceUnit, type LloydServiceDetail } from "../../api";

export default function ServicesPage() {
  // Gateway (managed) services state
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [timestamp, setTimestamp] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ServiceDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [gatewayRestarting, setGatewayRestarting] = useState(false);
  const [countdown, setCountdown] = useState(8);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Lloyd services state
  const [lloydServices, setLloydServices] = useState<LloydServiceUnit[]>([]);
  const [lloydTimestamp, setLloydTimestamp] = useState("");
  const [lloydLoading, setLloydLoading] = useState(true);
  const [lloydActionLoading, setLloydActionLoading] = useState<string | null>(null);
  const [lloydExpandedUnit, setLloydExpandedUnit] = useState<string | null>(null);
  const [lloydDetail, setLloydDetail] = useState<LloydServiceDetail | null>(null);
  const [lloydDetailLoading, setLloydDetailLoading] = useState(false);

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

  const refreshLloyd = useCallback(async () => {
    try {
      const data = await api.lloydServices();
      setLloydServices(data.services);
      setLloydTimestamp(data.timestamp);
    } catch {
      // keep stale data
    }
    setLloydLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    refreshLloyd();
    const interval = setInterval(refreshLloyd, 10000);
    return () => clearInterval(interval);
  }, [refreshLloyd]);

  // Fetch detail when a gateway service is expanded
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

  // Auto-refresh detail for expanded gateway service
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

  // Fetch detail when a lloyd service is expanded
  const fetchLloydDetail = useCallback(async (unit: string) => {
    setLloydDetailLoading(true);
    try {
      const data = await api.lloydServiceDetail(unit);
      setLloydDetail(data);
    } catch {
      setLloydDetail(null);
    }
    setLloydDetailLoading(false);
  }, []);

  useEffect(() => {
    if (!lloydExpandedUnit) return;
    fetchLloydDetail(lloydExpandedUnit);
    const interval = setInterval(() => fetchLloydDetail(lloydExpandedUnit), 5000);
    return () => clearInterval(interval);
  }, [lloydExpandedUnit, fetchLloydDetail]);

  const toggleExpand = (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      setDetail(null);
    } else {
      setExpandedId(id);
    }
  };

  const toggleLloydExpand = (unit: string) => {
    if (lloydExpandedUnit === unit) {
      setLloydExpandedUnit(null);
      setLloydDetail(null);
    } else {
      setLloydExpandedUnit(unit);
    }
  };

  const handleAction = async (
    e: React.MouseEvent,
    serviceId: string,
    action: "start" | "stop" | "restart",
  ) => {
    e.stopPropagation();

    if (serviceId === "gateway" && action === "restart") {
      setGatewayRestarting(true);
      setCountdown(8);
      api.serviceAction(serviceId, action).catch(() => {});
      const tick = setInterval(() => {
        setCountdown((c) => {
          if (c <= 1) {
            clearInterval(tick);
            window.location.reload();
            return 0;
          }
          return c - 1;
        });
      }, 1000);
      return;
    }

    if (serviceId === "gateway" && action === "stop") {
      const ok = window.confirm(
        "Stopping the gateway will disconnect Mission Control and all services will become unreachable. Continue?",
      );
      if (!ok) return;
      api.serviceAction(serviceId, action).catch(() => {});
      return;
    }

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

  const handleLloydAction = async (
    e: React.MouseEvent,
    serviceId: string,
    action: "start" | "stop" | "restart",
  ) => {
    e.stopPropagation();
    setLloydActionLoading(`${serviceId}-${action}`);
    try {
      await api.lloydServiceAction(serviceId, action);
      await new Promise((r) => setTimeout(r, 1500));
      await refreshLloyd();
      if (lloydExpandedUnit) fetchLloydDetail(lloydExpandedUnit);
    } catch {
      await refreshLloyd();
    }
    setLloydActionLoading(null);
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
  const lloydHealthy = lloydServices.filter((s) => s.health === "healthy").length;
  const totalHealthy = healthyCount + lloydHealthy;
  const totalServices = services.length + lloydServices.length;

  return (
    <div className="p-6 space-y-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-brand-400" />
          <h2 className="text-lg font-semibold">Services</h2>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500">
            {totalHealthy}/{totalServices} healthy
          </span>
          {timestamp && (
            <span className="text-[10px] text-slate-600 font-mono">
              {new Date(timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Gateway restart overlay */}
      {gatewayRestarting && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-surface-1 border border-surface-3/50 rounded-2xl px-10 py-8 text-center space-y-4 max-w-sm">
            <RefreshCw className="w-8 h-8 text-brand-400 animate-spin mx-auto" />
            <div className="text-base font-medium text-slate-200">Gateway Restarting</div>
            <div className="text-sm text-slate-400">
              Refreshing in <span className="font-mono text-brand-400">{countdown}s</span>
            </div>
            <div className="w-full bg-surface-3/30 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-brand-400 h-full rounded-full transition-all duration-1000 ease-linear"
                style={{ width: `${((8 - countdown) / 8) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && lloydLoading && (
        <div className="text-sm text-slate-500 text-center py-8">
          Loading services...
        </div>
      )}

      {/* Unified service cards */}
      <div className="space-y-2">
        {/* Gateway card(s) */}
        {services.map((svc) => {
          const isExpanded = expandedId === svc.id;
          return (
            <div key={svc.id} className="rounded-xl border border-surface-3/50 overflow-hidden">
              <div
                onClick={() => toggleExpand(svc.id)}
                className={`bg-surface-1 px-5 py-4 flex items-center gap-4 cursor-pointer transition-colors ${
                  isExpanded
                    ? "border-b border-surface-3/50"
                    : "hover:border-surface-3/80"
                }`}
              >
                <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${healthDot(svc.health)}`} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-slate-200">{svc.name}</div>
                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                    {svc.unit} &middot; :{svc.port}
                  </div>
                </div>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${stateBadge(svc.systemdState)}`}>
                  {svc.systemdState}
                </span>
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${svc.portHealthy ? "text-emerald-400" : "text-slate-600"}`}
                  title={svc.portHealthy ? "Port responding" : "Port not responding"}
                >
                  :{svc.port}
                </span>
                <div className="flex gap-1">
                  <button onClick={(e) => handleAction(e, svc.id, "start")} disabled={actionLoading !== null} title="Start" className="p-1.5 rounded-lg text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-50">
                    <Play className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={(e) => handleAction(e, svc.id, "stop")} disabled={actionLoading !== null} title="Stop" className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50">
                    <Square className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={(e) => handleAction(e, svc.id, "restart")} disabled={actionLoading !== null} title="Restart" className={`p-1.5 rounded-lg text-slate-400 hover:text-amber-400 hover:bg-amber-500/10 transition-colors disabled:opacity-50 ${actionLoading === `${svc.id}-restart` ? "animate-spin" : ""}`}>
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
              </div>

              {isExpanded && (
                <div className="bg-surface-0 px-5 py-4 space-y-4">
                  {detailLoading && !detail ? (
                    <div className="text-xs text-slate-500 text-center py-4">Loading service details...</div>
                  ) : detail ? (
                    <>
                      <div className="flex flex-wrap gap-4">
                        {detail.pid && <StatChip icon={<Cpu className="w-3.5 h-3.5" />} label="PID" value={String(detail.pid)} />}
                        {detail.memory && <StatChip icon={<HardDrive className="w-3.5 h-3.5" />} label="Memory" value={detail.memory} />}
                        {detail.cpu && <StatChip icon={<Clock className="w-3.5 h-3.5" />} label="CPU" value={detail.cpu} />}
                        {detail.tasks && <StatChip icon={<Activity className="w-3.5 h-3.5" />} label="Tasks" value={detail.tasks} />}
                      </div>
                      {detail.activeSince && (
                        <div className="text-[11px] text-slate-400 font-mono">Active: {detail.activeSince}</div>
                      )}
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Terminal className="w-3.5 h-3.5 text-slate-500" />
                          <span className="text-xs text-slate-400 font-medium">Recent Logs</span>
                        </div>
                        <div className="bg-black/40 rounded-lg p-3 max-h-72 overflow-y-auto font-mono text-[11px] leading-relaxed border border-surface-3/30">
                          {detail.logLines.length > 0 ? (
                            detail.logLines.map((line, i) => (
                              <div key={i} className={`whitespace-pre-wrap break-all ${line.match(/error|fail|panic|critical/i) ? "text-red-400" : line.match(/warn/i) ? "text-amber-400" : "text-slate-400"}`}>
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
                    <div className="text-xs text-red-400 text-center py-4">Failed to load service details</div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {/* Lloyd service cards */}
        {lloydServices.map((svc) => {
          const isExpanded = lloydExpandedUnit === svc.unit;
          return (
            <div key={svc.unit} className="rounded-xl border border-surface-3/50 overflow-hidden">
              <div
                onClick={() => toggleLloydExpand(svc.unit)}
                className={`bg-surface-1 px-5 py-4 flex items-center gap-4 cursor-pointer transition-colors ${
                  isExpanded ? "border-b border-surface-3/50" : "hover:border-surface-3/80"
                }`}
              >
                <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${healthDot(svc.health)}`} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-slate-200">{svc.name}</div>
                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                    {svc.unit}{svc.port ? ` \u00b7 :${svc.port}` : ""}
                  </div>
                </div>
                {svc.uptime && (
                  <span className="text-[10px] text-slate-500 font-mono flex-shrink-0">{svc.uptime}</span>
                )}
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${stateBadge(svc.activeState)}`}>
                  {svc.activeState}
                </span>
                {svc.port !== null && (
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${svc.portHealthy ? "text-emerald-400" : "text-slate-600"}`}
                    title={svc.portHealthy === true ? "Port responding" : svc.portHealthy === false ? "Port not responding" : "No port check"}
                  >
                    :{svc.port}
                  </span>
                )}
                <div className="flex gap-1">
                  <button onClick={(e) => handleLloydAction(e, svc.id, "start")} disabled={lloydActionLoading !== null} title="Start" className="p-1.5 rounded-lg text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-50">
                    <Play className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={(e) => handleLloydAction(e, svc.id, "stop")} disabled={lloydActionLoading !== null} title="Stop" className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50">
                    <Square className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={(e) => handleLloydAction(e, svc.id, "restart")} disabled={lloydActionLoading !== null} title="Restart" className={`p-1.5 rounded-lg text-slate-400 hover:text-amber-400 hover:bg-amber-500/10 transition-colors disabled:opacity-50 ${lloydActionLoading === `${svc.id}-restart` ? "animate-spin" : ""}`}>
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
              </div>

              {isExpanded && (
                <div className="bg-surface-0 px-5 py-4 space-y-4">
                  {lloydDetailLoading && !lloydDetail ? (
                    <div className="text-xs text-slate-500 text-center py-4">Loading service details...</div>
                  ) : lloydDetail && lloydDetail.unit === svc.unit ? (
                    <>
                      <div className="flex flex-wrap gap-4">
                        {lloydDetail.pid && <StatChip icon={<Cpu className="w-3.5 h-3.5" />} label="PID" value={String(lloydDetail.pid)} />}
                        {lloydDetail.memory && <StatChip icon={<HardDrive className="w-3.5 h-3.5" />} label="Memory" value={lloydDetail.memory} />}
                        {lloydDetail.cpu && <StatChip icon={<Clock className="w-3.5 h-3.5" />} label="CPU" value={lloydDetail.cpu} />}
                        {lloydDetail.tasks && <StatChip icon={<Activity className="w-3.5 h-3.5" />} label="Tasks" value={lloydDetail.tasks} />}
                      </div>
                      {lloydDetail.activeSince && (
                        <div className="text-[11px] text-slate-400 font-mono">Active: {lloydDetail.activeSince}</div>
                      )}
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Terminal className="w-3.5 h-3.5 text-slate-500" />
                          <span className="text-xs text-slate-400 font-medium">Recent Logs</span>
                        </div>
                        <div className="bg-black/40 rounded-lg p-3 max-h-72 overflow-y-auto font-mono text-[11px] leading-relaxed border border-surface-3/30">
                          {lloydDetail.logLines.length > 0 ? (
                            lloydDetail.logLines.map((line, i) => (
                              <div key={i} className={`whitespace-pre-wrap break-all ${line.match(/error|fail|panic|critical/i) ? "text-red-400" : line.match(/warn/i) ? "text-amber-400" : "text-slate-400"}`}>
                                {line}
                              </div>
                            ))
                          ) : (
                            <div className="text-slate-600 italic">No log lines available</div>
                          )}
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="text-xs text-red-400 text-center py-4">Failed to load service details</div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {!loading && !lloydLoading && services.length === 0 && lloydServices.length === 0 && (
          <div className="text-sm text-slate-600 text-center py-4 italic">
            No services found
          </div>
        )}
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
