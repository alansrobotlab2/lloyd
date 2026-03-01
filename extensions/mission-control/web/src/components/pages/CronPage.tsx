import { useEffect, useState } from "react";
import { Clock, CheckCircle, XCircle, AlertTriangle, Play } from "lucide-react";
import { api, type HealthData } from "../../api";

export default function CronPage() {
  const [health, setHealth] = useState<HealthData | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(console.error);
  }, []);

  const jobs = health?.cron || [];

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Clock className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Crontab</h2>
        <span className="text-xs text-slate-500">{jobs.length} scheduled jobs</span>
      </div>

      <div className="space-y-3">
        {jobs.map((job) => (
          <div
            key={job.id}
            className="bg-surface-1 rounded-xl p-5 border border-surface-3/50"
          >
            <div className="flex items-center gap-3 mb-3">
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                  job.lastStatus === "error"
                    ? "bg-red-400/10"
                    : job.enabled
                      ? "bg-emerald-400/10"
                      : "bg-slate-400/10"
                }`}
              >
                {job.lastStatus === "error" ? (
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                ) : (
                  <Play className="w-4 h-4 text-emerald-400" />
                )}
              </div>
              <div className="flex-1">
                <div className="text-sm font-medium text-slate-200">{job.name}</div>
                <div className="text-[10px] text-slate-500 font-mono">{job.id}</div>
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
            </div>

            <div className="grid grid-cols-3 gap-4 text-xs">
              <div>
                <div className="text-slate-500 mb-1">Last Status</div>
                <div className="flex items-center gap-1.5">
                  {job.lastStatus === "error" ? (
                    <XCircle className="w-3.5 h-3.5 text-red-400" />
                  ) : job.lastStatus === "success" ? (
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                  )}
                  <span className="text-slate-300">{job.lastStatus || "never run"}</span>
                </div>
              </div>
              <div>
                <div className="text-slate-500 mb-1">Consecutive Errors</div>
                <div
                  className={`text-slate-300 ${job.consecutiveErrors > 0 ? "text-red-400" : ""}`}
                >
                  {job.consecutiveErrors}
                </div>
              </div>
              <div>
                <div className="text-slate-500 mb-1">Next Run</div>
                <div className="text-slate-300">
                  {job.nextRunAt
                    ? new Date(job.nextRunAt).toLocaleString([], {
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
        ))}

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
