import { useEffect, useState } from "react";
import {
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Play,
  ChevronDown,
  Timer,
  Settings,
  FileText,
  History,
} from "lucide-react";
import { api, type CronJob, type CronRunEntry } from "../../api";

function formatInterval(ms: number): string {
  if (ms >= 86400000) return `Every ${Math.round(ms / 86400000)}d`;
  if (ms >= 3600000) return `Every ${Math.round(ms / 3600000)}h`;
  if (ms >= 60000) return `Every ${Math.round(ms / 60000)} minutes`;
  return `Every ${Math.round(ms / 1000)}s`;
}

function formatDuration(ms: number): string {
  if (ms >= 60000) return `${(ms / 1000).toFixed(0)}s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${ms}ms`;
}

function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max) + "...";
}

export default function CronPage() {
  const [jobs, setJobs] = useState<CronJob[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [runs, setRuns] = useState<CronRunEntry[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);

  useEffect(() => {
    api.cronJobs().then((d) => setJobs(d.jobs)).catch(console.error);
  }, []);

  useEffect(() => {
    if (!expandedId) {
      setRuns([]);
      return;
    }
    setRunsLoading(true);
    api
      .cronRuns(expandedId)
      .then((d) => setRuns(d.runs))
      .catch(console.error)
      .finally(() => setRunsLoading(false));
  }, [expandedId]);

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Clock className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Crontab</h2>
        <span className="text-xs text-slate-500">
          {jobs.length} scheduled job{jobs.length !== 1 ? "s" : ""}
        </span>
      </div>

      <div className="space-y-3">
        {jobs.map((job) => {
          const isExpanded = expandedId === job.id;
          const lastStatus = job.state?.lastStatus || job.state?.lastRunStatus;

          return (
            <div
              key={job.id}
              className="bg-surface-1 rounded-xl border border-surface-3/50"
            >
              {/* Collapsed card header */}
              <div
                className="p-5 cursor-pointer select-none"
                onClick={() => setExpandedId(isExpanded ? null : job.id)}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div
                    className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                      lastStatus === "error"
                        ? "bg-red-400/10"
                        : job.enabled
                          ? "bg-emerald-400/10"
                          : "bg-slate-400/10"
                    }`}
                  >
                    {lastStatus === "error" ? (
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                    ) : (
                      <Play className="w-4 h-4 text-emerald-400" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium text-slate-200">
                      {job.name}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono flex items-center gap-2">
                      <span>{job.id}</span>
                      <span className="px-1.5 py-0.5 rounded bg-surface-2 text-slate-400">
                        {job.agentId}
                      </span>
                    </div>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                      job.enabled
                        ? "bg-emerald-400/10 text-emerald-400"
                        : "bg-slate-400/10 text-slate-400"
                    }`}
                  >
                    {job.enabled ? "enabled" : "disabled"}
                  </span>
                  <ChevronDown
                    className={`w-4 h-4 text-slate-500 transition-transform duration-200 ${
                      isExpanded ? "rotate-180" : ""
                    }`}
                  />
                </div>

                <div className="grid grid-cols-3 gap-4 text-xs">
                  <div>
                    <div className="text-slate-500 mb-1">Last Status</div>
                    <div className="flex items-center gap-1.5">
                      {lastStatus === "error" ? (
                        <XCircle className="w-3.5 h-3.5 text-red-400" />
                      ) : lastStatus === "ok" || lastStatus === "success" ? (
                        <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Clock className="w-3.5 h-3.5 text-slate-400" />
                      )}
                      <span className="text-slate-300">
                        {lastStatus || "never run"}
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500 mb-1">Consecutive Errors</div>
                    <div
                      className={`text-slate-300 ${
                        (job.state?.consecutiveErrors || 0) > 0
                          ? "text-red-400"
                          : ""
                      }`}
                    >
                      {job.state?.consecutiveErrors ?? 0}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500 mb-1">Next Run</div>
                    <div className="text-slate-300">
                      {job.state?.nextRunAtMs
                        ? new Date(job.state.nextRunAtMs).toLocaleString([], {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })
                        : "--"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Expanded section */}
              <div
                className={`overflow-hidden transition-all duration-300 ${
                  isExpanded ? "max-h-[2000px]" : "max-h-0"
                }`}
              >
                <div className="px-5 pb-5">
                  {/* Schedule */}
                  <div className="border-t border-surface-3/50 mt-4 pt-4">
                    <div className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-2">
                      <Timer className="w-3.5 h-3.5" />
                      Schedule
                    </div>
                    <div className="text-xs text-slate-400">
                      {job.schedule?.kind === "cron" ? (
                        <div className="flex items-center gap-3">
                          <span className="font-mono bg-surface-2 px-2 py-0.5 rounded text-slate-300">
                            {job.schedule.expr}
                          </span>
                          {job.schedule.tz && (
                            <span className="text-slate-500">
                              TZ: {job.schedule.tz}
                            </span>
                          )}
                        </div>
                      ) : job.schedule?.kind === "every" && job.schedule.everyMs ? (
                        <span>{formatInterval(job.schedule.everyMs)}</span>
                      ) : (
                        <span>Unknown schedule</span>
                      )}
                    </div>
                  </div>

                  {/* Payload */}
                  <div className="border-t border-surface-3/50 mt-4 pt-4">
                    <div className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-2">
                      <FileText className="w-3.5 h-3.5" />
                      Payload
                    </div>
                    <div className="flex items-center gap-2 mb-2">
                      {job.payload?.model && (
                        <span className="px-1.5 py-0.5 rounded bg-surface-2 text-[10px] text-slate-400 font-mono">
                          {job.payload.model}
                        </span>
                      )}
                      {job.payload?.timeoutSeconds && (
                        <span className="text-[10px] text-slate-500">
                          timeout: {job.payload.timeoutSeconds}s
                        </span>
                      )}
                    </div>
                    <pre className="bg-surface-2 rounded-lg p-3 text-xs text-slate-300 font-mono whitespace-pre-wrap max-h-64 overflow-y-auto">
                      {job.payload?.message || "(no message)"}
                    </pre>
                  </div>

                  {/* Configuration */}
                  <div className="border-t border-surface-3/50 mt-4 pt-4">
                    <div className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-2">
                      <Settings className="w-3.5 h-3.5" />
                      Configuration
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-xs">
                      <div>
                        <div className="text-slate-500 mb-0.5">
                          Session Target
                        </div>
                        <div className="text-slate-300 font-mono">
                          {job.sessionTarget || "--"}
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-500 mb-0.5">Wake Mode</div>
                        <div className="text-slate-300">{job.wakeMode || "--"}</div>
                      </div>
                      <div>
                        <div className="text-slate-500 mb-0.5">Delivery</div>
                        <div className="text-slate-300">
                          {job.delivery
                            ? `${job.delivery.mode} / ${job.delivery.channel}`
                            : "--"}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Timestamps */}
                  <div className="border-t border-surface-3/50 mt-4 pt-4">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                      <div>
                        <div className="text-slate-500 mb-0.5">Created</div>
                        <div className="text-slate-400">
                          {job.createdAtMs
                            ? new Date(job.createdAtMs).toLocaleString()
                            : "--"}
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-500 mb-0.5">Updated</div>
                        <div className="text-slate-400">
                          {job.updatedAtMs
                            ? new Date(job.updatedAtMs).toLocaleString()
                            : "--"}
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-500 mb-0.5">Last Run</div>
                        <div className="text-slate-400">
                          {job.state?.lastRunAtMs
                            ? new Date(job.state.lastRunAtMs).toLocaleString()
                            : "--"}
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-500 mb-0.5">Next Run</div>
                        <div className="text-slate-400">
                          {job.state?.nextRunAtMs
                            ? new Date(job.state.nextRunAtMs).toLocaleString()
                            : "--"}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Run History */}
                  <div className="border-t border-surface-3/50 mt-4 pt-4">
                    <div className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-3">
                      <History className="w-3.5 h-3.5" />
                      Run History
                      {runsLoading && (
                        <span className="text-slate-500 font-normal animate-pulse">
                          loading...
                        </span>
                      )}
                    </div>

                    {!runsLoading && runs.length === 0 && (
                      <div className="text-xs text-slate-500">No run history</div>
                    )}

                    {runs.length > 0 && (
                      <div className="space-y-2">
                        {runs.map((run, i) => (
                          <div
                            key={`${run.ts}-${i}`}
                            className="bg-surface-2 rounded-lg p-3 text-xs"
                          >
                            <div className="flex items-center gap-2 mb-1">
                              {run.status === "error" ? (
                                <XCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                              ) : (
                                <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                              )}
                              <span className="text-slate-300">
                                {new Date(run.ts).toLocaleString([], {
                                  month: "short",
                                  day: "numeric",
                                  hour: "2-digit",
                                  minute: "2-digit",
                                })}
                              </span>
                              {run.durationMs != null && (
                                <span className="text-slate-500">
                                  {formatDuration(run.durationMs)}
                                </span>
                              )}
                              {run.model && (
                                <span className="px-1.5 py-0.5 rounded bg-surface-1 text-[10px] text-slate-500 font-mono">
                                  {run.model}
                                </span>
                              )}
                            </div>
                            {run.summary && (
                              <div className="text-slate-400 mt-1">
                                {truncate(run.summary, 100)}
                              </div>
                            )}
                            {run.error && (
                              <div className="text-red-400 mt-1">{run.error}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}

        {jobs.length === 0 && (
          <div className="bg-surface-1 rounded-xl p-8 border border-surface-3/50 text-center">
            <Clock className="w-8 h-8 mx-auto mb-2 text-slate-500 opacity-30" />
            <p className="text-sm text-slate-500">No cron jobs configured</p>
          </div>
        )}
      </div>
    </div>
  );
}
