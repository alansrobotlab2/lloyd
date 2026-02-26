/**
 * index.ts — web-local OpenClaw plugin (Phase 5 — MCP proxy)
 *
 * Proxies web_search and web_fetch through the shared MCP server subprocess
 * (extensions/mcp-server/server.py) via McpStdioClient (JSON-RPC 2.0 stdio).
 *
 * Previously: inline implementation using execFile + linkedom + @mozilla/readability.
 * Now: thin proxy — all logic lives in server.py.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { McpStdioClient } from "../memory-graph/mcp-client.js";

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.(
    "web-local: registering web_search + web_fetch (proxied through MCP server)",
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
      "Fetch a URL and extract its readable content. Returns the main text content of the page.",
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
}
