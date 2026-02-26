/**
 * index.ts — memory-graph OpenClaw plugin (v4 — Python-delegated prefill).
 *
 * Provides:
 *  1. Thin before_prompt_build hook — calls the Python MCP server's
 *     prefill_context tool; all pipeline logic lives in server.py.
 *  2. Explicit tools — tag_search, tag_explore, vault_overview,
 *     memory_search, memory_get — proxied through the openclaw MCP
 *     server subprocess via McpStdioClient.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { createPrefillHook } from "./prefill.js";
import { McpStdioClient } from "./mcp-client.js";

export default function register(api: OpenClawPluginApi) {
  // ── MCP client (shared by prefill hook and tool proxies) ──────────
  //
  // Spawns the MCP server subprocess on first use (lazy). Shared so that
  // the prefill hook and the tool proxies use the same running server.py process.

  const mcpClient = new McpStdioClient();
  process.on("exit", () => mcpClient.destroy());

  // ── Unified prefill hook — delegates to Python MCP server ─────────

  const prefillHandler = createPrefillHook(mcpClient, api.logger);
  api.on("before_prompt_build", prefillHandler);

  // ── Tool proxy via MCP server ──────────────────────────────────────
  //
  // All 5 memory tools are served by the openclaw MCP server subprocess.

  function proxyTool(
    name: string,
    description: string,
    parameters: object,
  ): void {
    api.registerTool({
      name,
      description,
      parameters,
      async execute(_id: string, params: any) {
        try {
          const content = await mcpClient.callTool(name, params);
          return { content: content.length > 0 ? content : [{ type: "text" as const, text: "(no result)" }] };
        } catch (err: any) {
          return { content: [{ type: "text" as const, text: `Error: ${err.message}` }] };
        }
      },
    });
  }

  proxyTool(
    "tag_search",
    "Search the Obsidian knowledge vault by tags. Returns documents matching the specified tag(s) " +
      "with their title, summary, tags, type, and status. Use AND mode to find documents at the " +
      "intersection of multiple topics, OR mode for broader searches. " +
      "Examples: tag_search([\"alfie\"]), tag_search([\"ai\", \"rag\"], \"and\"), tag_search([\"robotics\"], type=\"hub\").",
    {
      type: "object",
      properties: {
        tags: {
          type: "array",
          items: { type: "string" },
          description: "One or more tags to search for (without # prefix)",
        },
        mode: {
          type: "string",
          enum: ["and", "or"],
          description: "Match mode: 'and' = docs must have ALL tags, 'or' = docs with ANY tag (default: 'or')",
        },
        type: {
          type: "string",
          description: "Filter by document type: hub, notes, project-notes, work-notes, talk, or 'any' (default: 'any')",
        },
        limit: {
          type: "integer",
          description: "Max results to return (default: 10, max: 25)",
        },
      },
      required: ["tags"],
    },
  );

  proxyTool(
    "tag_explore",
    "Explore tag relationships in the vault. Given a tag, shows co-occurring tags ranked by " +
      "frequency. Optionally provide bridge_to to find documents that have BOTH tags. " +
      "Use this to discover connections between topics and navigate the vault's knowledge structure.",
    {
      type: "object",
      properties: {
        tag: {
          type: "string",
          description: "A tag to explore relationships for (without # prefix)",
        },
        bridge_to: {
          type: "string",
          description: "Optional second tag — shows documents that have BOTH tags (bridging documents)",
        },
        limit: {
          type: "integer",
          description: "Max related tags to show (default: 15)",
        },
      },
      required: ["tag"],
    },
  );

  proxyTool(
    "vault_overview",
    "Show vault statistics and structure: document counts by type, tags with frequencies, " +
      "hub pages (index pages), and type distribution. Use this to understand what's in " +
      "the vault before searching.",
    {
      type: "object",
      properties: {
        detail: {
          type: "string",
          enum: ["summary", "tags", "hubs", "types"],
          description: "What to show: 'summary' = overview, 'tags' = all tags with frequencies, 'hubs' = hub pages, 'types' = type breakdown (default: 'summary')",
        },
      },
    },
  );

  proxyTool(
    "memory_search",
    "BM25 full-text search across the Obsidian vault. Returns matching documents with paths, " +
      "scores, and snippets. Standalone — no gateway required.",
    {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        max_results: { type: "integer", description: "Max results to return (default: 10)" },
      },
      required: ["query"],
    },
  );

  proxyTool(
    "memory_get",
    "Read a specific file from the Obsidian vault by relative path. " +
      "path: relative from vault root, e.g. 'projects/alfie/alfie.md'. Supports optional line range.",
    {
      type: "object",
      properties: {
        path: { type: "string", description: "Vault-relative file path" },
        start_line: { type: "integer", description: "First line to return (1-indexed, 0 = beginning)" },
        end_line: { type: "integer", description: "Last line to return (0 = end)" },
      },
      required: ["path"],
    },
  );

  proxyTool(
    "memory_write",
    "Create or overwrite a file in the Obsidian vault. " +
      "path: vault-relative path, e.g. 'projects/alfie/notes.md'. Creates parent directories automatically.",
    {
      type: "object",
      properties: {
        path: { type: "string", description: "Vault-relative file path, e.g. 'projects/alfie/notes.md'" },
        content: { type: "string", description: "Text content to write" },
      },
      required: ["path", "content"],
    },
  );

  api.logger.info?.("memory-graph v4: registered (Python-delegated prefill + 6 tools via MCP)");
}
