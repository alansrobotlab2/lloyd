/**
 * index.ts — thunderbird-tools: OpenClaw plugin for Thunderbird email/calendar/contacts.
 *
 * Spawns the Thunderbird MCP bridge (mcp-bridge.cjs) as a stdio subprocess,
 * discovers tools via tools/list, and registers each as an OpenClaw tool
 * with a "tb_" prefix.
 *
 * Requires Thunderbird to be running with the MCP extension (localhost:8765).
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { spawn, ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";

const BRIDGE_PATH =
  "/home/alansrobotlab/Projects/lloyd-services/services/thunderbird-mcp/mcp-bridge.cjs";

const TOOL_NAMES: Record<string, string> = {
  listAccounts:      "email_accounts",
  listFolders:       "email_folders",
  searchMessages:    "email_search",
  getMessage:        "email_read",
  getRecentMessages: "email_recent",
  searchContacts:    "contacts_search",
  getContact:        "contacts_get",
  listCalendars:     "calendar_list",
  getEvents:         "calendar_events",
};
const DEFAULT_TIMEOUT_MS = 30_000;

// ── Stdio MCP client (adapted from mcp-tools/mcp-client.ts) ────────────

interface PendingCall {
  resolve: (value: any) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 30_000;
const CIRCUIT_BREAKER_THRESHOLD = 3;
const CIRCUIT_BREAKER_COOLDOWN_MS = 60_000;

class McpBridgeClient {
  private proc: ChildProcess | null = null;
  private pending = new Map<number, PendingCall>();
  private nextId = 2; // 1 reserved for initialize
  private ready: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((e: Error) => void) | null = null;
  private backoffMs = 0;
  private lastCrashTime = 0;
  private consecutiveCrashes = 0;
  private circuitOpenUntil = 0;

  start(): void {
    if (this.proc) return;

    this.ready = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });

    this.proc = spawn("node", [BRIDGE_PATH], {
      stdio: ["pipe", "pipe", "inherit"],
    });

    this.proc.on("error", (err) => {
      this.readyReject?.(err);
      this.readyResolve = null;
      this.readyReject = null;
    });

    this.proc.on("exit", (code) => {
      if (code !== 0 && code !== null) {
        this.readyReject?.(new Error(`Thunderbird bridge exited with code ${code}`));
        this.readyResolve = null;
        this.readyReject = null;
      }
      for (const [, p] of this.pending) {
        clearTimeout(p.timer);
        p.reject(new Error("Thunderbird bridge exited unexpectedly"));
      }
      this.pending.clear();
      this.proc = null;
      this.ready = null;

      const now = Date.now();
      if (now - this.lastCrashTime < 60_000) {
        this.backoffMs = Math.min(
          (this.backoffMs || INITIAL_BACKOFF_MS) * 2,
          MAX_BACKOFF_MS,
        );
        this.consecutiveCrashes++;
        if (this.consecutiveCrashes >= CIRCUIT_BREAKER_THRESHOLD) {
          this.circuitOpenUntil = now + CIRCUIT_BREAKER_COOLDOWN_MS;
        }
      } else {
        this.backoffMs = INITIAL_BACKOFF_MS;
        this.consecutiveCrashes = 0;
      }
      this.lastCrashTime = now;
    });

    const rl = createInterface({ input: this.proc.stdout! });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      let msg: any;
      try {
        msg = JSON.parse(line);
      } catch {
        return;
      }
      if (msg.id === 1) {
        this.sendRaw({
          jsonrpc: "2.0",
          method: "notifications/initialized",
          params: {},
        });
        this.readyResolve?.();
        this.readyResolve = null;
        this.readyReject = null;
        return;
      }
      const p = this.pending.get(msg.id);
      if (!p) return;
      clearTimeout(p.timer);
      this.pending.delete(msg.id);
      if (msg.error) {
        p.reject(new Error(`MCP error ${msg.error.code}: ${msg.error.message}`));
      } else {
        p.resolve(msg.result);
      }
    });

    this.sendRaw({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "thunderbird-tools-plugin", version: "1.0.0" },
      },
    });
  }

  private sendRaw(msg: object): void {
    try {
      this.proc?.stdin?.write(JSON.stringify(msg) + "\n");
    } catch {
      // EPIPE — subprocess is dead, will restart on next call
    }
  }

  async ensureReady(): Promise<void> {
    const now = Date.now();
    if (this.circuitOpenUntil > now) {
      throw new Error(
        `Thunderbird bridge circuit breaker open (${this.consecutiveCrashes} crashes, resets in ${Math.ceil((this.circuitOpenUntil - now) / 1000)}s)`,
      );
    }
    if (this.circuitOpenUntil > 0 && this.circuitOpenUntil <= now) {
      this.circuitOpenUntil = 0;
      this.consecutiveCrashes = 0;
      this.backoffMs = INITIAL_BACKOFF_MS;
    }
    if (!this.proc && this.backoffMs > 0) {
      await new Promise((r) => setTimeout(r, this.backoffMs));
    }
    if (!this.proc) this.start();
    await this.ready;
  }

  async callTool(
    name: string,
    args: Record<string, unknown>,
    timeoutMs: number = DEFAULT_TIMEOUT_MS,
  ): Promise<Array<{ type: string; text: string }>> {
    await this.ensureReady();

    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Thunderbird tool "${name}" timed out`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });
      this.sendRaw({
        jsonrpc: "2.0",
        id,
        method: "tools/call",
        params: { name, arguments: args },
      });
    }).then((result: any) => result?.content ?? []);
  }

  async listTools(): Promise<Array<{ name: string; description?: string; inputSchema?: any }>> {
    await this.ensureReady();

    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error("tools/list timed out"));
      }, 10_000);

      this.pending.set(id, { resolve, reject, timer });
      this.sendRaw({
        jsonrpc: "2.0",
        id,
        method: "tools/list",
        params: {},
      });
    }).then((result: any) => result?.tools ?? []);
  }

  destroy(): void {
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error("McpBridgeClient destroyed"));
    }
    this.pending.clear();
    this.proc?.kill();
    this.proc = null;
    this.ready = null;
  }
}

// ── Plugin registration ─────────────────────────────────────────────────

export default function register(api: OpenClawPluginApi) {
  const client = new McpBridgeClient();
  process.on("exit", () => client.destroy());

  // Discover tools from the bridge and register them
  client
    .listTools()
    .then((tools) => {
      for (const tool of tools) {
        const prefixedName = TOOL_NAMES[tool.name] ?? `tb_${tool.name}`;
        const params = tool.inputSchema ?? { type: "object", properties: {} };

        api.registerTool({
          name: prefixedName,
          label: `Thunderbird: ${tool.name}`,
          description: tool.description ?? `Thunderbird tool: ${tool.name}`,
          parameters: params,
          async execute(_id: string, callParams: any) {
            try {
              const content = await client.callTool(tool.name, callParams);
              const text =
                content.length > 0
                  ? content.map((c: any) => c.text).join("")
                  : "(no result)";
              return { content: [{ type: "text" as const, text }] };
            } catch (err: any) {
              return {
                content: [
                  { type: "text" as const, text: `${prefixedName} error: ${err.message}` },
                ],
              };
            }
          },
        });
      }
      api.logger.info?.(
        `thunderbird-tools: registered ${tools.length} tools (${tools.map((t) => TOOL_NAMES[t.name] ?? `tb_${t.name}`).join(", ")})`,
      );
    })
    .catch((err) => {
      api.logger.warn?.(
        `thunderbird-tools: failed to discover tools — ${err.message}. Is Thunderbird running?`,
      );
    });
}
