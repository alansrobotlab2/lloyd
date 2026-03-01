/**
 * mcp-sse-client.ts — MCP SSE client with reconnection and circuit breaker.
 *
 * Connects to a standalone MCP server over SSE transport (instead of spawning
 * a subprocess over stdio). Same callTool() interface as McpStdioClient.
 *
 * Protocol:
 *  1. GET /sse → SSE stream
 *  2. First event "endpoint" → data contains POST URL for JSON-RPC messages
 *  3. Client sends initialize via POST, then tools/call for each invocation
 *  4. Server responds via SSE "message" events with JSON-RPC responses
 */

interface PendingCall {
  resolve: (value: any) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 30_000;
const CIRCUIT_BREAKER_THRESHOLD = 5;
const CIRCUIT_BREAKER_COOLDOWN_MS = 60_000;
const CONNECT_TIMEOUT_MS = 8_000; // max time to wait for SSE handshake

export class McpSseClient {
  private baseUrl: string;
  private postUrl: string | null = null;
  private abortController: AbortController | null = null;
  private pending = new Map<number, PendingCall>();
  private nextId = 2; // 1 reserved for initialize
  private ready: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((e: Error) => void) | null = null;
  private connected = false;
  private connecting = false;
  private backoffMs = 0;
  private lastFailTime = 0;
  private consecutiveFailures = 0;
  private circuitOpenUntil = 0;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  private async connect(): Promise<void> {
    if (this.connected || this.connecting) return;
    this.connecting = true;

    this.ready = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });

    try {
      this.abortController = new AbortController();
      const sseUrl = `${this.baseUrl}/sse`;

      const response = await fetch(sseUrl, {
        headers: { Accept: "text/event-stream" },
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`SSE connect failed: ${response.status} ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error("SSE response has no body");
      }

      // Read SSE stream in background
      this.readSseStream(response.body);
    } catch (err: any) {
      this.connecting = false;
      this.handleConnectionFailure();
      this.readyReject?.(err);
      this.readyResolve = null;
      this.readyReject = null;
      throw err;
    }
  }

  private async readSseStream(body: ReadableStream<Uint8Array>): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = "";
    let currentData = "";

    const processEvent = () => {
      if (!currentEvent && !currentData) return;

      if (currentEvent === "endpoint") {
        // The endpoint event data is the POST URL path (e.g., /messages/?session_id=xxx)
        const postPath = currentData.trim();
        this.postUrl = postPath.startsWith("http")
          ? postPath
          : `${this.baseUrl}${postPath}`;

        // Send initialize
        this.sendInitialize();
      } else if (currentEvent === "message") {
        try {
          const msg = JSON.parse(currentData);
          this.handleMessage(msg);
        } catch {
          // Ignore unparseable messages
        }
      }

      currentEvent = "";
      currentData = "";
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (let line of lines) {
          if (line.endsWith("\r")) line = line.slice(0, -1); // handle CRLF
          if (line === "") {
            processEvent();
          } else if (line.startsWith("event:")) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            currentData += (currentData ? "\n" : "") + line.slice(5).trim();
          }
        }
      }
    } catch (err: any) {
      if (err.name === "AbortError") return; // intentional disconnect
    } finally {
      this.connected = false;
      this.connecting = false;
      this.postUrl = null;

      // Reject all pending calls
      for (const [, p] of this.pending) {
        clearTimeout(p.timer);
        p.reject(new Error("SSE connection closed"));
      }
      this.pending.clear();

      this.handleConnectionFailure();
    }
  }

  private handleMessage(msg: any): void {
    if (msg.id === 1) {
      // Initialize response — complete handshake
      this.connected = true;
      this.connecting = false;
      this.backoffMs = 0;
      this.consecutiveFailures = 0;
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
  }

  private async sendInitialize(): Promise<void> {
    try {
      await this.postMessage({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "mcp-sse-plugin", version: "1.0.0" },
        },
      });
    } catch (err: any) {
      this.readyReject?.(err);
      this.readyResolve = null;
      this.readyReject = null;
    }
  }

  private async postMessage(msg: object): Promise<void> {
    if (!this.postUrl) {
      throw new Error("No POST endpoint available (SSE not connected)");
    }
    const resp = await fetch(this.postUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(msg),
    });
    if (!resp.ok) {
      throw new Error(`MCP POST failed: ${resp.status}`);
    }
  }

  private handleConnectionFailure(): void {
    const now = Date.now();
    if (now - this.lastFailTime < 60_000) {
      this.backoffMs = Math.min(
        (this.backoffMs || INITIAL_BACKOFF_MS) * 2,
        MAX_BACKOFF_MS,
      );
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
        this.circuitOpenUntil = now + CIRCUIT_BREAKER_COOLDOWN_MS;
      }
    } else {
      this.backoffMs = INITIAL_BACKOFF_MS;
      this.consecutiveFailures = 0;
    }
    this.lastFailTime = now;
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
        `MCP SSE circuit breaker open (${this.consecutiveFailures} consecutive failures, resets in ${Math.ceil((this.circuitOpenUntil - now) / 1000)}s)`,
      );
    }
    if (this.circuitOpenUntil > 0 && this.circuitOpenUntil <= now) {
      this.circuitOpenUntil = 0;
      this.consecutiveFailures = 0;
      this.backoffMs = INITIAL_BACKOFF_MS;
    }

    // Apply reconnection backoff
    if (!this.connected && this.backoffMs > 0) {
      await new Promise((r) => setTimeout(r, this.backoffMs));
    }

    if (!this.connected && !this.connecting) {
      await this.connect();
    }

    // Race the ready promise against a connection timeout so a hung
    // SSE handshake can never block callTool (and thus agent runs) forever.
    const connectDeadline = Math.min(timeoutMs, CONNECT_TIMEOUT_MS);
    try {
      await Promise.race([
        this.ready,
        new Promise<void>((_, reject) =>
          setTimeout(
            () => reject(new Error("MCP SSE handshake timed out")),
            connectDeadline,
          ),
        ),
      ]);
    } catch (err) {
      // If handshake timed out, tear down the in-progress connection so
      // subsequent callTool invocations start a fresh connect() instead
      // of racing the same stale ready promise forever.
      if (!this.connected) {
        this.connecting = false;
        this.abortController?.abort();
      }
      throw err;
    }

    const id = this.nextId++;
    return new Promise<any>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP SSE tool call "${name}" timed out`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });
      this.postMessage({
        jsonrpc: "2.0",
        id,
        method: "tools/call",
        params: { name, arguments: args },
      }).catch((err) => {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(err);
      });
    }).then((result: any) => result?.content ?? []);
  }

  destroy(): void {
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error("McpSseClient destroyed"));
    }
    this.pending.clear();
    this.abortController?.abort();
    this.abortController = null;
    this.connected = false;
    this.connecting = false;
    this.postUrl = null;
    this.ready = null;
  }
}
