/**
 * mcp-client.ts — Minimal MCP stdio client for OpenClaw plugin use.
 *
 * Spawns the openclaw MCP server subprocess and proxies tool calls via
 * JSON-RPC 2.0 over stdio. Zero external dependencies.
 */

import { spawn, ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const DEFAULT_SERVER_PATH = join(
  dirname(fileURLToPath(import.meta.url)),
  "../mcp-server/server.py",
);

interface PendingCall {
  resolve: (value: any) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class McpStdioClient {
  private proc: ChildProcess | null = null;
  private pending = new Map<number, PendingCall>();
  private nextId = 2; // 1 reserved for initialize
  private ready: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((e: Error) => void) | null = null;
  private serverPath: string;

  constructor(serverPath?: string) {
    this.serverPath = serverPath ?? DEFAULT_SERVER_PATH;
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
        clientInfo: { name: "memory-graph-plugin", version: "1.0.0" },
      },
    });
  }

  private sendRaw(msg: object): void {
    this.proc!.stdin!.write(JSON.stringify(msg) + "\n");
  }

  /** Call a tool on the MCP server and return its content array. */
  async callTool(
    name: string,
    args: Record<string, unknown>,
    timeoutMs: number = 10_000,
  ): Promise<Array<{ type: string; text: string }>> {
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
