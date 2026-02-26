/**
 * index.ts — web-local OpenClaw plugin (Phase 5 — MCP proxy)
 *
 * Proxies web_search, web_fetch, and http_request through the shared MCP server
 * subprocess (extensions/mcp-server/server.py) via McpStdioClient (JSON-RPC 2.0 stdio).
 *
 * Previously: inline implementation using execFile + linkedom + @mozilla/readability.
 * Now: thin proxy — all logic lives in server.py.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { McpStdioClient } from "../memory-graph/mcp-client.js";

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.(
    "web-local: registering web_search + web_fetch + http_request (proxied through MCP server)",
  );

  const mcpClient = new McpStdioClient();
  process.on("exit", () => mcpClient.destroy());

  // ── web_search ─────────────────────────────────────────────────────────────

  api.registerTool({
    name: "web_search",
    label: "Web Search",
    description:
      "Search the web using Google. Returns a list of results with title, URL, and snippet.",
    parameters: {
      type: "object" as const,
      properties: {
        query: { type: "string" as const, description: "Search query" },
        count: {
          type: "integer" as const,
          description: "Number of results to return (1–10, default 5)",
          minimum: 1,
          maximum: 10,
        },
      },
      required: ["query"] as string[],
    },
    async execute(_id: string, params: { query: string; count?: number }) {
      try {
        const content = await mcpClient.callTool("web_search", params as Record<string, unknown>);
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `web_search error: ${err.message}` }] };
      }
    },
  });

  // ── web_fetch ──────────────────────────────────────────────────────────────

  api.registerTool({
    name: "web_fetch",
    label: "Fetch Web Page",
    description:
      "Fetch a public web page and extract its readable content as clean markdown or text. " +
      "Uses readability to strip boilerplate and returns the article/main body — ideal for reading " +
      "documentation, blog posts, and news articles. GET only; public URLs only (no 127.0.0.1). " +
      "For REST APIs, POST requests, auth headers, or local services, use http_request instead.",
    parameters: {
      type: "object" as const,
      properties: {
        url: {
          type: "string" as const,
          description: "The URL to fetch (http or https)",
        },
        extractMode: {
          type: "string" as const,
          enum: ["markdown", "text"],
          description: 'Extraction mode: "markdown" or "text" (default "markdown")',
        },
        maxChars: {
          type: "integer" as const,
          description: "Maximum characters to return (default 50000)",
          minimum: 1000,
          maximum: 200000,
        },
      },
      required: ["url"] as string[],
    },
    async execute(
      _id: string,
      params: { url: string; extractMode?: "markdown" | "text"; maxChars?: number },
    ) {
      // Translate camelCase LLM params → snake_case Python params
      const mcpArgs: Record<string, unknown> = { url: params.url };
      if (params.extractMode !== undefined) mcpArgs.extract_mode = params.extractMode;
      if (params.maxChars !== undefined) mcpArgs.max_chars = params.maxChars;

      try {
        const content = await mcpClient.callTool("web_fetch", mcpArgs);
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `web_fetch error: ${err.message}` }] };
      }
    },
  });

  // ── http_request ───────────────────────────────────────────────────────────

  api.registerTool({
    name: "http_request",
    label: "HTTP Request",
    description:
      "Make a raw HTTP request (GET, POST, PUT, PATCH, DELETE, HEAD) and return the status code " +
      "and response body unprocessed. Use this for REST APIs, local services, and any endpoint " +
      "that needs custom headers, a request body, or non-GET methods. " +
      "127.0.0.1 (loopback) is allowed for local container services. " +
      "Other private/internal IPs are blocked. " +
      "For reading public web pages as readable text, use web_fetch instead.",
    parameters: {
      type: "object" as const,
      properties: {
        method: {
          type: "string" as const,
          enum: ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
          description: "HTTP method",
        },
        url: {
          type: "string" as const,
          description: "Full URL (http or https)",
        },
        headers: {
          type: "object" as const,
          description: "Optional request headers (e.g. {\"Authorization\": \"Bearer token\", \"Content-Type\": \"application/json\"})",
          additionalProperties: { type: "string" as const },
        },
        body: {
          type: "string" as const,
          description: "Optional request body string (for JSON, set Content-Type header yourself)",
        },
        timeout: {
          type: "integer" as const,
          description: "Max seconds to wait (default 30, max 120)",
          minimum: 1,
          maximum: 120,
        },
      },
      required: ["method", "url"] as string[],
    },
    async execute(
      _id: string,
      params: { method: string; url: string; headers?: Record<string, string>; body?: string; timeout?: number },
    ) {
      try {
        const content = await mcpClient.callTool("http_request", params as Record<string, unknown>);
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `http_request error: ${err.message}` }] };
      }
    },
  });
}
