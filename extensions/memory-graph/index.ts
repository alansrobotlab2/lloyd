/**
 * index.ts — memory-graph OpenClaw plugin (v3 — unified context prefill).
 *
 * Provides:
 *  1. Unified before_prompt_build hook — runs tag matching (instant,
 *     in-memory) and vector search (async) in parallel, merges and
 *     deduplicates results, returns a single <memory_context> block.
 *  2. Explicit tools — tag_search, tag_explore, vault_overview — proxied
 *     through the openclaw MCP server subprocess via McpStdioClient.
 */

import { existsSync } from "node:fs";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { scanVault } from "./scanner.js";
import { TagIndex } from "./tag-index.js";
import { createPrefillHook } from "./prefill.js";
import { McpStdioClient } from "./mcp-client.js";

// ── Configuration ─────────────────────────────────────────────────────

const VAULT_PATH = process.env.HOME
  ? `${process.env.HOME}/obsidian`
  : "/home/alansrobotlab/obsidian";
const REFRESH_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes

// ── State ─────────────────────────────────────────────────────────────

let tagIndex: TagIndex = new TagIndex();
let refreshLock = false;

// ── Index Build ───────────────────────────────────────────────────────

function buildIndex(logger?: { info?: (...args: any[]) => void; warn?: (...args: any[]) => void }): TagIndex {
  if (!existsSync(VAULT_PATH)) {
    logger?.warn?.(`memory-graph: vault not found at ${VAULT_PATH}`);
    return new TagIndex();
  }
  const scan = scanVault(VAULT_PATH);
  const idx = TagIndex.fromScan(scan);
  logger?.info?.(
    `memory-graph: indexed ${idx.docCount} docs, ${idx.tagCount} tags`,
  );
  return idx;
}

async function refreshIndex(logger?: any): Promise<void> {
  if (refreshLock) return;
  refreshLock = true;
  try {
    const newIndex = buildIndex(logger);
    tagIndex = newIndex; // atomic swap
  } finally {
    refreshLock = false;
  }
}

// ── Plugin Registration ───────────────────────────────────────────────

export default function register(api: OpenClawPluginApi) {
  // Build index on load
  tagIndex = buildIndex(api.logger);

  // Periodic refresh
  const timer = setInterval(() => refreshIndex(api.logger), REFRESH_INTERVAL_MS);
  if (timer.unref) timer.unref();

  // ── Unified prefill hook ───────────────────────────────────────────

  const prefillHandler = createPrefillHook(() => tagIndex, api.logger);
  api.on("before_prompt_build", prefillHandler);

  // ── Tool proxy via MCP server ──────────────────────────────────────
  //
  // tag_search, tag_explore, and vault_overview are now served by the
  // openclaw MCP server subprocess. The client spawns the server on the
  // first tool call and proxies subsequent calls over JSON-RPC 2.0 stdio.

  const mcpClient = new McpStdioClient();
  process.on("exit", () => mcpClient.destroy());

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

  api.logger.info?.("memory-graph v3: registered (unified prefill hook + 3 tools via MCP)");
}
