/**
 * services.ts — systemctl, supervisor, port checks, and service management
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, existsSync } from "fs";
import { join } from "path";
import { execSync } from "child_process";
import { connect as netConnect } from "net";
import type { PluginContext, SupervisorEntry } from "./types.js";
import { jsonResponse, readBody, handleCorsOptions, requirePost } from "./helpers.js";

// ── Constants ───────────────────────────────────────────────────────

const MANAGED_SERVICES: { id: string; name: string; unit: string; port: number }[] = [];

const AGENT_DISPLAY_NAMES: Record<string, string> = {
  "agent-discord-voice-bridge.service": "Discord Voice Bridge",
  "agent-discord-voice-server.service": "Discord Voice Server",
  "agent-distrobox.service": "Distrobox (supervisord)",
  "agent-llm.service": "LLM 35B (Qwen3.5-35B-A3B)",
  "agent-qmd-daemon.service": "QMD Daemon",
  "agent-qmd-watcher.service": "QMD Watcher",
  "agent-tool-mcp.service": "Tool Services MCP",
  "agent-tts.service": "TTS Service",
  "agent-voice-mcp.service": "Voice Services MCP",
  "agent-voice-mode.service": "Voice Mode",
  "openclaw-cert.service": "Certificate Page",
  "openclaw-dee.service": "OpenClaw DEE Gateway",
  "openclaw-lloyd.service": "OpenClaw LLOYD Gateway",
  "openclaw-trey.service": "OpenClaw TREY Gateway",
  "openclaw-gateway.service": "OpenClaw Gateway",
};

export const AGENT_PORT_MAP: Record<string, number> = {
  "agent-llm.service": 8091,
  "agent-tts.service": 8090,
  "agent-voice-mode.service": 8092,
  "agent-tool-mcp.service": 8093,
  "agent-voice-mcp.service": 8094,
  "agent-qmd-daemon.service": 8181,
  "openclaw-cert.service": 18790,
  "openclaw-dee.service": 19789,
  "openclaw-lloyd.service": 18789,
  "openclaw-gateway.service": 18789,
};

const SUPERVISOR_LOGS_DIR = "/home/alansrobotlab/agents/agent-services-home/agent-services/logs";

// ── Shared helpers ──────────────────────────────────────────────────

function checkPort(port: number, timeoutMs = 2000): Promise<boolean> {
  return new Promise((resolve) => {
    // Try IPv4 first, fall back to IPv6 (some services bind to [::1] only)
    const tryConnect = (host: string, onFail: () => void) => {
      const socket = netConnect({ host, port });
      const timer = setTimeout(() => { socket.destroy(); onFail(); }, timeoutMs);
      socket.on("connect", () => { clearTimeout(timer); socket.destroy(); resolve(true); });
      socket.on("error", () => { clearTimeout(timer); onFail(); });
    };
    tryConnect("127.0.0.1", () => tryConnect("::1", () => resolve(false)));
  });
}

function getSupervisorStatus(): SupervisorEntry[] {
  try {
    const xmlBody = '<?xml version="1.0"?><methodCall><methodName>supervisor.getAllProcessInfo</methodName></methodCall>';
    const output = execSync(
      `curl --unix-socket /tmp/agent-supervisor.sock -s -X POST http://localhost/RPC2 -H "Content-Type: text/xml" -d '${xmlBody}'`,
      { encoding: "utf-8", timeout: 5000 },
    );
    const entries: SupervisorEntry[] = [];
    const procRegex = /<struct>([\s\S]*?)<\/struct>/g;
    let match;
    while ((match = procRegex.exec(output)) !== null) {
      const block = match[1];
      const getName = (field: string) => {
        const m = block.match(new RegExp(`<name>${field}</name>[\\s\\S]*?<(?:string|int)>([^<]*)</`));
        return m ? m[1] : null;
      };
      const name = getName("name");
      const statename = getName("statename");
      const pid = getName("pid");
      const description = getName("description");
      if (name && statename) {
        let uptime: string | null = null;
        if (description) {
          const um = description.match(/uptime\s+(\S+)/);
          if (um) uptime = um[1];
        }
        entries.push({ name, state: statename, pid: pid ? parseInt(pid, 10) : null, uptime });
      }
    }
    return entries;
  } catch {
    return [];
  }
}

function callSupervisorXmlRpc(method: string, processName: string): string {
  const body = `<?xml version="1.0"?><methodCall><methodName>${method}</methodName><params><param><value><string>${processName}</string></value></param></params></methodCall>`;
  return execSync(
    `curl --unix-socket /tmp/agent-supervisor.sock -s -X POST http://localhost/RPC2 -H "Content-Type: text/xml" -d '${body}'`,
    { encoding: "utf-8", timeout: 15000 },
  );
}

/** Parse systemctl status output and fetch journal logs. Shared between services/detail and agent-services/detail. */
function getSystemdServiceDetail(unit: string, logLines = 40): {
  pid: number | null; memory: string | null; cpu: string | null;
  tasks: string | null; activeSince: string | null; logLines: string[];
  rawStatus: string;
} {
  let statusOutput = "";
  try {
    statusOutput = execSync(`systemctl --user status ${unit} 2>&1`, { encoding: "utf-8", timeout: 5000 });
  } catch (e: any) {
    statusOutput = e.stdout || e.message || "Unable to get status";
  }

  const pidMatch = statusOutput.match(/Main PID:\s*(\d+)/);
  const memoryMatch = statusOutput.match(/Memory:\s*(\S+)/);
  const cpuMatch = statusOutput.match(/CPU:\s*(\S+)/);
  const activeMatch = statusOutput.match(/Active:\s*(.+)/);
  const tasksMatch = statusOutput.match(/Tasks:\s*(\S+)/);

  let logs = "";
  try {
    logs = execSync(`journalctl --user -u ${unit} -n ${logLines} --no-pager -o short-iso 2>&1`, { encoding: "utf-8", timeout: 5000 });
  } catch (e: any) {
    logs = e.stdout || "Unable to fetch logs";
  }

  return {
    pid: pidMatch ? parseInt(pidMatch[1], 10) : null,
    memory: memoryMatch ? memoryMatch[1] : null,
    cpu: cpuMatch ? cpuMatch[1] : null,
    tasks: tasksMatch ? tasksMatch[1] : null,
    activeSince: activeMatch ? activeMatch[1].trim() : null,
    logLines: logs.split("\n").filter((l: string) => l.trim() !== "").slice(-logLines),
    rawStatus: statusOutput,
  };
}

function getDisplayName(unit: string): string {
  return AGENT_DISPLAY_NAMES[unit] ?? unit.replace(/^agent-/, "").replace(/\.service$/, "").replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── Route registration ──────────────────────────────────────────────

export function registerServiceRoutes(ctx: PluginContext) {
  const { api } = ctx;

  // GET /api/mc/services
  api.registerHttpRoute({
    path: "/api/mc/services",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const services = await Promise.all(
          MANAGED_SERVICES.map(async (svc) => {
            let systemdState = "unknown";
            try {
              systemdState = execSync(`systemctl --user is-active ${svc.unit} 2>/dev/null`, { encoding: "utf-8", timeout: 3000 }).trim();
            } catch { systemdState = "inactive"; }
            const portHealthy = await checkPort(svc.port);
            let health: string;
            if (systemdState === "active" && portHealthy) health = "healthy";
            else if (systemdState === "active" && !portHealthy) health = "degraded";
            else health = "stopped";
            return { id: svc.id, name: svc.name, unit: svc.unit, port: svc.port, systemdState, portHealthy, health };
          }),
        );
        jsonResponse(res, { services, timestamp: new Date().toISOString() });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/services/action
  api.registerHttpRoute({
    path: "/api/mc/services/action",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (handleCorsOptions(req, res)) return;
      if (requirePost(req, res)) return;
      try {
        const body = JSON.parse(await readBody(req));
        const { serviceId, action } = body;

        const svc = MANAGED_SERVICES.find((s) => s.id === serviceId);
        // Hoist supervisor check so supervisord-managed services (e.g. idler-agent) aren't
        // rejected before we get a chance to route them through the supervisorctl path.
        const supEntries = getSupervisorStatus();
        const isSupervisorManaged = supEntries.some(e => e.name === serviceId);
        const agentUnit = !svc && (serviceId.startsWith("agent-") || serviceId.startsWith("openclaw-") || isSupervisorManaged) ? `${serviceId}.service` : null;
        if (!svc && !agentUnit) { jsonResponse(res, { error: `Unknown service: ${serviceId}` }, 400); return; }
        const unitName = svc ? svc.unit : agentUnit!;
        if (!["start", "stop", "restart"].includes(action)) { jsonResponse(res, { error: `Invalid action: ${action}` }, 400); return; }

        // Check if supervisor-managed
        const supervisorName = unitName.replace(".service", "");
        if (supEntries.some(e => e.name === supervisorName)) {
          // Detect self-restart: if restarting/stopping the gateway we're running inside,
          // send response first, then exit (supervisord autorestart will bring us back)
          const isSelfService = (serviceId === "openclaw-gateway" || serviceId === "openclaw-lloyd") && 
            (action === "restart" || action === "stop");
          
          if (isSelfService) {
            jsonResponse(res, { ok: true, serviceId, action, managedBy: "supervisor", selfRestart: true });
            setTimeout(() => process.exit(1), 500);
            return;
          }
          
          try {
            // Use supervisorctl CLI instead of XML-RPC to avoid stop/start race condition.
            // XML-RPC stopProcess returns before the process actually dies, so a subsequent
            // startProcess gets "already running" and the process ends up STOPPED after SIGKILL.
            execSync(
              `supervisorctl -c /home/alansrobotlab/agent-services/supervisor/supervisord.conf ${action} ${supervisorName}`,
              { encoding: "utf-8", timeout: 30000 },
            );
            jsonResponse(res, { ok: true, serviceId, action, managedBy: "supervisor" });
          } catch (err: any) {
            // supervisorctl restart reports "ERROR (not running)" on stop phase if already stopped, but still starts — treat as success
            if (action === "restart" && err.stdout?.includes("started")) {
              jsonResponse(res, { ok: true, serviceId, action, managedBy: "supervisor" });
            } else {
              jsonResponse(res, { error: err.message }, 500);
            }
          }
          return;
        }

        // For gateway restart/start, kill the port first
        if (unitName === "openclaw-lloyd.service" && (action === "restart" || action === "start")) {
          try { execSync(`kill $(lsof -ti :18789 -sTCP:LISTEN) 2>/dev/null`, { timeout: 5000 }); } catch { /* non-fatal */ }
          await new Promise((r) => setTimeout(r, 2000));
          execSync(`systemctl --user ${action} ${unitName}`, { encoding: "utf-8", timeout: 15000 });
        } else if ((action === "stop" || action === "restart") && (svc?.port || AGENT_PORT_MAP[unitName])) {
          const port = svc?.port ?? AGENT_PORT_MAP[unitName];
          execSync(`systemctl --user stop ${unitName}`, { encoding: "utf-8", timeout: 15000 });
          await new Promise((r) => setTimeout(r, 1000));
          try { execSync(`kill $(lsof -ti :${port} -sTCP:LISTEN) 2>/dev/null`, { timeout: 5000 }); } catch { /* non-fatal */ }
          await new Promise((r) => setTimeout(r, 1000));
          if (action === "restart") {
            execSync(`systemctl --user start ${unitName}`, { encoding: "utf-8", timeout: 15000 });
          }
        } else {
          execSync(`systemctl --user ${action} ${unitName}`, { encoding: "utf-8", timeout: 15000 });
        }

        jsonResponse(res, { ok: true, serviceId, action });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/services/detail
  api.registerHttpRoute({
    path: "/api/mc/services/detail",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const serviceId = url.searchParams.get("id");
        const svc = MANAGED_SERVICES.find((s) => s.id === serviceId);
        if (!svc) { jsonResponse(res, { error: `Unknown service: ${serviceId}` }, 400); return; }

        const detail = getSystemdServiceDetail(svc.unit);
        jsonResponse(res, { id: svc.id, name: svc.name, unit: svc.unit, port: svc.port, ...detail });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/agent-services/detail
  api.registerHttpRoute({
    path: "/api/mc/agent-services/detail",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const unit = url.searchParams.get("unit");
        if (!unit || ((!unit.startsWith("agent-") && !unit.startsWith("openclaw-")) || !unit.endsWith(".service"))) {
          jsonResponse(res, { error: `Invalid unit: ${unit}` }, 400);
          return;
        }

        const name = getDisplayName(unit);
        const supervisorName = unit.replace(".service", "");
        const supEntries = getSupervisorStatus();

        if (supEntries.some(e => e.name === supervisorName)) {
          const sup = supEntries.find(e => e.name === supervisorName);
          let logLines: string[] = [];
          try {
            const stdout = readFileSync(join(SUPERVISOR_LOGS_DIR, `${supervisorName}.log`), "utf-8");
            const stderr = readFileSync(join(SUPERVISOR_LOGS_DIR, `${supervisorName}.err`), "utf-8");
            logLines = [
              ...stdout.split("\n").filter((l: string) => l.trim()).slice(-30),
              ...stderr.split("\n").filter((l: string) => l.trim()).slice(-10).map((l: string) => `[stderr] ${l}`),
            ].slice(-40);
          } catch { /* logs may not exist */ }

          jsonResponse(res, {
            unit, name,
            pid: sup?.pid ?? null, memory: null, cpu: null, tasks: null,
            activeSince: sup?.uptime ? `uptime ${sup.uptime}` : null,
            logLines,
            rawStatus: sup ? `${sup.name} ${sup.state} pid ${sup.pid}, uptime ${sup.uptime}` : "unknown",
            managedBy: "supervisor",
          });
          return;
        }

        const detail = getSystemdServiceDetail(unit);
        jsonResponse(res, { unit, name, ...detail });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/agent-services
  api.registerHttpRoute({
    path: "/api/mc/agent-services",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        let units: any[] = [];
        try {
          const output = execSync(
            `systemctl --user list-units 'agent-*' 'openclaw-*' --all --output=json 2>/dev/null`,
            { encoding: "utf-8", timeout: 5000 },
          );
          units = JSON.parse(output).filter((u: any) => u.load !== "not-found");
        } catch (e: any) {
          jsonResponse(res, { error: "Failed to query systemctl: " + (e.message || String(e)) }, 500);
          return;
        }

        // Discover installed-but-unloaded unit files
        try {
          const ufOutput = execSync(
            `systemctl --user list-unit-files 'agent-*.service' 'openclaw-*.service' --output=json 2>/dev/null`,
            { encoding: "utf-8", timeout: 5000 },
          );
          const unitFiles: any[] = JSON.parse(ufOutput);
          const loadedUnits = new Set(units.map((u: any) => u.unit));
          for (const uf of unitFiles) {
            const unitName: string = uf.unit_file || uf["unit file"] || "";
            if (!unitName || loadedUnits.has(unitName)) continue;
            if (uf.state === "disabled") continue;
            try {
              const showOutput = execSync(
                `systemctl --user show ${unitName} --property=ActiveState,SubState,Description 2>/dev/null`,
                { encoding: "utf-8", timeout: 2000 },
              ).trim();
              const props: Record<string, string> = {};
              for (const line of showOutput.split("\n")) {
                const eq = line.indexOf("=");
                if (eq > 0) props[line.slice(0, eq)] = line.slice(eq + 1);
              }
              units.push({ unit: unitName, load: "loaded", active: props.ActiveState || "inactive", sub: props.SubState || "dead", description: props.Description || unitName });
            } catch {
              units.push({ unit: unitName, load: "loaded", active: "inactive", sub: "dead", description: unitName });
            }
          }
        } catch { /* non-fatal */ }

        const services = await Promise.all(
          units.map(async (u: any) => {
            const unit: string = u.unit || "";
            const id = unit.replace(".service", "");
            const name = getDisplayName(unit);
            const activeState: string = u.active || "unknown";
            const subState: string = u.sub || "unknown";
            const port: number | null = AGENT_PORT_MAP[unit] ?? null;
            let portHealthy: boolean | null = null;
            if (port !== null) portHealthy = await checkPort(port);

            let uptime: string | null = null;
            if (activeState === "active" && subState === "running") {
              try {
                const ts = execSync(
                  `systemctl --user show ${unit} --property=ActiveEnterTimestamp --value 2>/dev/null`,
                  { encoding: "utf-8", timeout: 2000 },
                ).trim();
                if (ts && ts !== "n/a" && ts !== "") {
                  const since = new Date(ts);
                  if (!isNaN(since.getTime())) {
                    const s = Math.floor((Date.now() - since.getTime()) / 1000);
                    if (s < 60) uptime = `${s}s`;
                    else if (s < 3600) uptime = `${Math.floor(s / 60)}m`;
                    else if (s < 86400) uptime = `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
                    else uptime = `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
                  }
                }
              } catch { /* non-fatal */ }
            }

            let health: string;
            if (activeState === "failed") health = "stopped";
            else if (activeState === "active" && subState === "running") health = portHealthy === false ? "degraded" : "healthy";
            else if (activeState === "active") health = "healthy";
            else health = "stopped";

            let category: string;
            if (unit.startsWith("openclaw-") && !unit.includes("cert")) category = "gateway";
            else if (/voice|tts|discord-voice/.test(unit)) category = "voice";
            else if (/tool-mcp|qmd/.test(unit)) category = "mcp";
            else if (unit.includes("llm")) category = "llm";
            else if (/distrobox|cert/.test(unit)) category = "infra";
            else category = "other";

            return { id, unit, name, activeState, subState, port, portHealthy, uptime, health, category };
          }),
        );

        // Overlay supervisor-managed services
        const supervisorEntries = getSupervisorStatus();
        for (const sup of supervisorEntries) {

          const unitName = `${sup.name}.service`;
          const existing = services.find((s: any) => s.unit === unitName);
          const activeState = sup.state === "RUNNING" ? "active" : sup.state === "FATAL" ? "failed" : "inactive";
          const subState = sup.state === "RUNNING" ? "running" : sup.state.toLowerCase();

          if (existing) {
            existing.activeState = activeState;
            existing.subState = subState;
            (existing as any).managedBy = "supervisor";
            if (sup.uptime) existing.uptime = sup.uptime;
            // Recompute health: trust port check over stale supervisor state
            if (existing.portHealthy === true) {
              existing.health = "healthy";
            } else if (activeState === "active") {
              existing.health = existing.portHealthy === false ? "degraded" : "healthy";
            } else {
              existing.health = "stopped";
            }
          } else {
            const port: number | null = AGENT_PORT_MAP[unitName] ?? null;
            let portHealthy: boolean | null = null;
            if (port !== null) portHealthy = await checkPort(port);
            let health: string;
            if (portHealthy === true) health = "healthy";
            else if (activeState === "active") health = portHealthy === false ? "degraded" : "healthy";
            else health = "stopped";

            let category: string;
            if (/voice|tts/.test(unitName)) category = "voice";
            else if (/tool-mcp/.test(unitName)) category = "mcp";
            else category = "other";

            const name = getDisplayName(unitName);
            services.push({
              id: sup.name, unit: unitName, name, activeState, subState,
              port, portHealthy, uptime: sup.uptime, health, category,
              managedBy: "supervisor" as any,
            });
          }
        }

        jsonResponse(res, { services, timestamp: new Date().toISOString() });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });
}
