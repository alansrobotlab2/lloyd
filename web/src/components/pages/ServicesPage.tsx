import { useState, useEffect, useCallback, useRef } from "react";
import { Activity, Play, Square, RotateCcw, RefreshCw, ChevronDown, Terminal, Cpu, HardDrive, Clock, AlertTriangle } from "lucide-react";
import { api, type ServiceStatus, type ServiceDetail, type LloydServiceUnit, type LloydServiceDetail } from "../../api";
import { useReportMcFocus, usePendingFocusFor } from "../../contexts/McUiContext";

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
  const [countdown, setCountdown] = useState(0);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Lloyd services state
  const [lloydServices, setLloydServices] = useState<LloydServiceUnit[]>([]);
  const [lloydTimestamp, setLloydTimestamp] = useState("");
  const [lloydLoading, setLloydLoading] = useState(true);
  const [lloydActionLoading, setLloydActionLoading] = useState<string | null>(null);
  const [lloydExpandedUnit, setLloydExpandedUnit] = useState<string | null>(null);
  const [lloydDetail, setLloydDetail] = useState<LloydServiceDetail | null>(null);
  const [lloydDetailLoading, setLloydDetailLoading] = useState(false);

  const focusedService = lloydExpandedUnit || expandedId;
  useReportMcFocus(
    "services",
    focusedService ? { kind: "service", id: focusedService } : null,
  );

  const pendingFocus = usePendingFocusFor("services");
  useEffect(() => {
    if (!pendingFocus) return;
    if (lloydServices.some((s) => s.unit === pendingFocus)) {
      setLloydExpandedUnit(pendingFocus);
    } else {
      setExpandedId(pendingFocus);
    }
  }, [pendingFocus, lloydServices]);

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

  useEffect(() => {
    if (!expandedId) return;
    fetchDetail(expandedId);
    const interval = setInterval(() => fetchDetail(expandedId), 5000);
    return () => clearInterval(interval);
  }, [expandedId, fetchDetail]);

  useEffect(() => {
    if (detail && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [detail?.logLines]);

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
      setCountdown(0);
      api.serviceAction(serviceId, action).catch(() => {});
      _pollForReconnect();
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

    if ((serviceId === "openclaw-gateway" || serviceId === "openclaw-lloyd" || serviceId === "hermes-mc") && action === "restart") {
      setGatewayRestarting(true);
      setCountdown(0);
      api.lloydServiceAction(serviceId, action).catch(() => {});
      _pollForReconnect();
      return;
    }

    if ((serviceId === "openclaw-gateway" || serviceId === "openclaw-lloyd" || serviceId === "hermes-mc") && action === "stop") {
      const ok = window.confirm(
        "Stopping the gateway will disconnect Mission Control and all services will become unreachable. Continue?",
      );
      if (!ok) return;
      api.lloydServiceAction(serviceId, action).catch(() => {});
      return;
    }

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

  const _pollForReconnect = () => {
    const startTime = Date.now();
    const maxWaitMs = 60000;
    const pollIntervalMs = 2000;
    const initialDelayMs = 3000;

    const showElapsed = setInterval(() => {
      setCountdown(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    setTimeout(() => {
      const poll = setInterval(async () => {
        try {
          const res = await fetch(`${window.location.origin}/api/services`, {
            signal: AbortSignal.timeout(2000),
          });
          if (res.ok) {
            clearInterval(poll);
            clearInterval(showElapsed);
            window.location.reload();
          }
        } catch {
          // still down, keep polling
        }
        if (Date.now() - startTime > maxWaitMs) {
          clearInterval(poll);
          clearInterval(showElapsed);
          window.location.reload();
        }
      }, pollIntervalMs);
    }, initialDelayMs);
  };

  const healthDot = (health: string) => {
    switch (health) {
      case "healthy":  return "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]";
      case "degraded": return "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.5)]";
      case "stopped":  return "bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.4)]";
      default:         return "bg-slate-500";
    }
  };

  const stateBadge = (state: string) => {
    switch (state) {
      case "active":   return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "failed":   return "bg-red-500/20 text-red-400 border-red-500/30";
      case "inactive": return "bg-secondary text-muted-foreground border-border";
      default:         return "bg-secondary text-muted-foreground border-border";
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
          <Activity className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Services</h2>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground">
            {totalHealthy}/{totalServices} healthy
          </span>
          {(timestamp || lloydTimestamp) && (
            <span className="text-[10px] text-muted-foreground/70 font-mono">
              {new Date(lloydTimestamp || timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Gateway restart overlay */}
      {gatewayRestarting && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-card border border-border/50 rounded-2xl px-10 py-8 text-center space-y-4 max-w-sm">
            <RefreshCw className="w-8 h-8 text-primary animate-spin mx-auto" />
            <div className="text-base font-medium text-foreground">Gateway Restarting</div>
            <div className="text-sm text-muted-foreground">
              Reconnecting... <span className="font-mono text-primary">{countdown}s</span>
            </div>
            <div className="w-full bg-muted/30 rounded-full h-1.5 overflow-hidden">
              <div className="bg-primary h-full rounded-full animate-pulse" />
            </div>
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && lloydLoading && (
        <div className="text-sm text-muted-foreground text-center py-8">
          Loading services...
        </div>
      )}

      {/* Service cards */}
      <div className="space-y-2">
        {/* Lloyd (agent) services — shown first */}
        {[...lloydServices].sort((a, b) => {
          const aGw = a.unit.startsWith("openclaw-") && !a.unit.includes("cert") ? 0 : 1;
          const bGw = b.unit.startsWith("openclaw-") && !b.unit.includes("cert") ? 0 : 1;
          return aGw - bGw;
        }).map((svc) => {
          const isExpanded = lloydExpandedUnit === svc.unit;
          return (
            <div key={svc.unit} className="rounded-xl border border-border/50 overflow-hidden">
              <div
                onClick={() => toggleLloydExpand(svc.unit)}
                className={`bg-card px-5 py-4 flex items-center gap-4 cursor-pointer transition-colors ${
                  isExpanded ? "border-b border-border/50" : "hover:border-border/80"
                }`}
              >
                <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${healthDot(svc.health)}`} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-foreground">{svc.name}</div>
                  <div className="text-[10px] text-muted-foreground font-mono mt-0.5">
                    {svc.unit}{svc.port ? ` · :${svc.port}` : ""}
                  </div>
                </div>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${stateBadge(svc.activeState)}`}>
                  {svc.activeState}
                </span>
                <span className={`text-[10px] w-14 text-right px-1.5 py-0.5 rounded font-mono tabular-nums ${svc.port && svc.portHealthy ? "text-emerald-400" : "text-muted-foreground/70"}`}>
                  :{svc.port ? String(svc.port).padStart(4, "0") : "0000"}
                </span>
                <div className="flex gap-1">
                  <button onClick={(e) => handleLloydAction(e, svc.id, "start")} disabled={lloydActionLoading !== null} title="Start" className="p-1.5 rounded-lg text-muted-foreground hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-50"><Play className="w-3.5 h-3.5" /></button>
                  <button onClick={(e) => handleLloydAction(e, svc.id, "stop")} disabled={lloydActionLoading !== null} title="Stop" className="p-1.5 rounded-lg text-muted-foreground hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"><Square className="w-3.5 h-3.5" /></button>
                  <button onClick={(e) => handleLloydAction(e, svc.id, "restart")} disabled={lloydActionLoading !== null} title="Restart" className={`p-1.5 rounded-lg text-muted-foreground hover:text-amber-400 hover:bg-amber-500/10 transition-colors disabled:opacity-50 ${lloydActionLoading === `${svc.id}-restart` ? "animate-spin" : ""}`}><RotateCcw className="w-3.5 h-3.5" /></button>
                </div>
                <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform ${isExpanded ? "rotate-180" : ""}`} />
              </div>
              {isExpanded && (
                <div className="bg-background px-5 py-4 space-y-4">
                  {lloydDetailLoading && !lloydDetail ? (
                    <div className="text-xs text-muted-foreground text-center py-4">Loading service details...</div>
                  ) : lloydDetail ? (
                    <LloydDetailView detail={lloydDetail} />
                  ) : (
                    <div className="text-xs text-red-400 text-center py-4">Failed to load service details</div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {/* Infrastructure services */}
        {services.map((svc) => {
          const isExpanded = expandedId === svc.id;
          return (
            <div key={svc.id} className="rounded-xl border border-border/50 overflow-hidden">
              <div
                onClick={() => toggleExpand(svc.id)}
                className={`bg-card px-5 py-4 flex items-center gap-4 cursor-pointer transition-colors ${
                  isExpanded ? "border-b border-border/50" : "hover:border-border/80"
                }`}
              >
                <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${healthDot(svc.health)}`} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-foreground">{svc.name}</div>
                  <div className="text-[10px] text-muted-foreground font-mono mt-0.5">
                    {svc.unit} &middot; :{svc.port}
                  </div>
                </div>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${stateBadge(svc.systemdState)}`}>
                  {svc.systemdState}
                </span>
                <span
                  className={`text-[10px] w-14 text-right px-1.5 py-0.5 rounded font-mono tabular-nums ${svc.portHealthy ? "text-emerald-400" : "text-muted-foreground/70"}`}
                  title={svc.portHealthy ? "Port responding" : svc.port ? "Port not responding" : "No port"}
                >
                  :{svc.port ? String(svc.port).padStart(4, "0") : "0000"}
                </span>
                <div className="flex gap-1">
                  <button onClick={(e) => handleAction(e, svc.id, "start")} disabled={actionLoading !== null} title="Start" className="p-1.5 rounded-lg text-muted-foreground hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-50">
                    <Play className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={(e) => handleAction(e, svc.id, "stop")} disabled={actionLoading !== null} title="Stop" className="p-1.5 rounded-lg text-muted-foreground hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50">
                    <Square className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={(e) => handleAction(e, svc.id, "restart")} disabled={actionLoading !== null} title="Restart" className={`p-1.5 rounded-lg text-muted-foreground hover:text-amber-400 hover:bg-amber-500/10 transition-colors disabled:opacity-50 ${actionLoading === `${svc.id}-restart` ? "animate-spin" : ""}`}>
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
                <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform ${isExpanded ? "rotate-180" : ""}`} />
              </div>

              {isExpanded && (
                <div className="bg-background px-5 py-4 space-y-4">
                  {detailLoading && !detail ? (
                    <div className="text-xs text-muted-foreground text-center py-4">Loading service details...</div>
                  ) : detail ? (
                    <ServiceDetailView detail={detail} logEndRef={logEndRef} />
                  ) : (
                    <div className="text-xs text-red-400 text-center py-4">Failed to load service details</div>
                  )}
                </div>
              )}
            </div>
          );
        })}


        {!loading && !lloydLoading && services.length === 0 && lloydServices.length === 0 && (
          <div className="text-sm text-muted-foreground/70 text-center py-4 italic">
            No services found
          </div>
        )}
      </div>
    </div>
  );
}

function ServiceDetailView({ detail, logEndRef }: { detail: ServiceDetail; logEndRef: React.RefObject<HTMLDivElement | null> }) {
  return (
    <>
      <div className="flex flex-wrap gap-4">
        {detail.pid && <StatChip icon={<Cpu className="w-3.5 h-3.5" />} label="PID" value={String(detail.pid)} />}
        {detail.memory && <StatChip icon={<HardDrive className="w-3.5 h-3.5" />} label="Memory" value={detail.memory} />}
        {detail.cpu && <StatChip icon={<Clock className="w-3.5 h-3.5" />} label="CPU" value={detail.cpu} />}
        {detail.tasks && <StatChip icon={<Activity className="w-3.5 h-3.5" />} label="Tasks" value={detail.tasks} />}
      </div>
      {detail.activeSince && (
        <div className="text-[11px] text-muted-foreground font-mono">Active: {detail.activeSince}</div>
      )}
      <LogView lines={detail.logLines} logEndRef={logEndRef} />
    </>
  );
}

function LloydDetailView({ detail }: { detail: LloydServiceDetail }) {
  return (
    <>
      <div className="flex flex-wrap gap-4">
        {detail.pid && <StatChip icon={<Cpu className="w-3.5 h-3.5" />} label="PID" value={String(detail.pid)} />}
        {detail.memory && <StatChip icon={<HardDrive className="w-3.5 h-3.5" />} label="Memory" value={detail.memory} />}
        {detail.cpu && <StatChip icon={<Clock className="w-3.5 h-3.5" />} label="CPU" value={detail.cpu} />}
        {detail.tasks && <StatChip icon={<Activity className="w-3.5 h-3.5" />} label="Tasks" value={detail.tasks} />}
      </div>
      {detail.activeSince && (
        <div className="text-[11px] text-muted-foreground font-mono">Active: {detail.activeSince}</div>
      )}
      <LogView lines={detail.logLines} />
    </>
  );
}

function LogView({ lines, logEndRef }: { lines: string[]; logEndRef?: React.RefObject<HTMLDivElement | null> }) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <Terminal className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs text-muted-foreground font-medium">Recent Logs</span>
      </div>
      <div className="bg-black/40 rounded-lg p-3 max-h-72 overflow-y-auto font-mono text-[11px] leading-relaxed border border-border/30">
        {lines.length > 0 ? (
          lines.map((line, i) => (
            <div key={i} className={`whitespace-pre-wrap break-all ${line.match(/error|fail|panic|critical/i) ? "text-red-400" : line.match(/warn/i) ? "text-amber-400" : "text-muted-foreground"}`}>
              {line}
            </div>
          ))
        ) : (
          <div className="text-muted-foreground/70 italic">No log lines available</div>
        )}
        {logEndRef && <div ref={logEndRef} />}
      </div>
    </div>
  );
}

function StatChip({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 bg-card rounded-lg px-3 py-2 border border-border/30">
      <span className="text-primary">{icon}</span>
      <div>
        <div className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</div>
        <div className="text-xs text-foreground font-mono">{value}</div>
      </div>
    </div>
  );
}
