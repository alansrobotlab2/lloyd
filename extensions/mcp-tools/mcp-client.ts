/**
 * mcp-client.ts — MCP stdio client with crash resilience.
 *
 * Improvements over the original (extensions/memory-graph/mcp-client.ts):
 *  - Exponential restart backoff (prevents crash loops)
 *  - EPIPE handling in sendRaw (graceful pipe closure detection)
 *  - Single instance design (one server, all tools)
 */

import { spawn, ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";

interface PendingCall {
  resolve: (value: any) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 30_000;
const CIRCUIT_BREAKER_THRESHOLD = 3; // consecutive crashes before opening circuit
const CIRCUIT_BREAKER_COOLDOWN_MS = 60_000; // 60s cooldown before retrying

export class McpStdioClient {
  private proc: ChildProcess | null = null;
  private pending = new Map<number, PendingCall>();
  private nextId = 2; // 1 reserved for initialize
  private ready: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((e: Error) => void) | null = null;
  private serverPath: string;
  private backoffMs = 0;
  private lastCrashTime = 0;
  private consecutiveCrashes = 0;
  private circuitOpenUntil = 0; // timestamp when circuit breaker resets

  constructor(serverPath: string) {
    this.serverPath = serverPath;
  }

  private start(): void {
    if (this.proc) return;

    this.ready = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });

    this.proc = spawn("uv", ["run", this.serverPath], {
      stdio: ["pipe", "pipe", "inherit"],
    });

    this.proc.on("error", (err) => {
      this.readyReject?.(err);
      this.readyResolve = null;
      this.readyReject = null;
    });

    this.proc.on("exit", (code) => {
      if (code !== 0 && code !== null) {
        this.readyReject?.(new Error(`MCP server exited with code ${code}`));
        this.readyResolve = null;
        this.readyReject = null;
      }
      for (const [, p] of this.pending) {
        clearTimeout(p.timer);
        p.reject(new Error("MCP server subprocess exited unexpectedly"));
      }
      this.pending.clear();
      this.proc = null;
      this.ready = null;

      // Track crash timing for backoff + circuit breaker
      const now = Date.now();
      if (now - this.lastCrashTime < 60_000) {
        // Crashed again within 60s — increase backoff and crash count
        this.backoffMs = Math.min(
          (this.backoffMs || INITIAL_BACKOFF_MS) * 2,
          MAX_BACKOFF_MS,
        );
        this.consecutiveCrashes++;
        if (this.consecutiveCrashes >= CIRCUIT_BREAKER_THRESHOLD) {
          this.circuitOpenUntil = now + CIRCUIT_BREAKER_COOLDOWN_MS;
        }
      } else {
        // Been stable for >60s — reset backoff and crash count
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
        // Initialize response — complete handshake then signal ready
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

    // Send initialize
    this.sendRaw({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "mcp-tools-plugin", version: "1.0.0" },
      },
    });
  }

  private sendRaw(msg: object): void {
    try {
      this.proc?.stdin?.write(JSON.stringify(msg) + "\n");
    } catch {
      // EPIPE or write-after-close — subprocess is dead, will be restarted on next callTool
    }
  }

  /** Call a tool on the MCP server and return its content array. */
  async callTool(
    name: string,
    args: Record<string, unknown>,
    timeoutMs: number = 10_000,
  ): Promise<Array<{ type: string; text: string }>> {
    // Circuit breaker: reject immediately if open
    const now = Date.now();
    if (this.circuitOpenUntil > now) {
      throw new Error(
        `MCP circuit breaker open (${this.consecutiveCrashes} consecutive crashes, resets in ${Math.ceil((this.circuitOpenUntil - now) / 1000)}s)`,
      );
    }
    // If cooldown has passed, reset the circuit breaker
    if (this.circuitOpenUntil > 0 && this.circuitOpenUntil <= now) {
      this.circuitOpenUntil = 0;
      this.consecutiveCrashes = 0;
      this.backoffMs = INITIAL_BACKOFF_MS;
    }

    // Apply restart backoff if subprocess recently crashed
    if (!this.proc && this.backoffMs > 0) {
      await new Promise((r) => setTimeout(r, this.backoffMs));
    }

    if (!this.proc) this.start();
    await this.ready;

    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP tool call "${name}" timed out`));
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

  destroy(): void {
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error("McpStdioClient destroyed"));
    }
    this.pending.clear();
    this.proc?.kill();
    this.proc = null;
    this.ready = null;
  }
}
