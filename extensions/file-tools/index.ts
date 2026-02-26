/**
 * index.ts — file-tools OpenClaw plugin (Phase 6 — MCP proxy)
 *
 * Proxies five filesystem tools through the shared MCP server subprocess
 * (extensions/mcp-server/server.py) via McpStdioClient (JSON-RPC 2.0 stdio).
 *
 *   file_read   — read any file within $HOME (with optional line range)
 *   file_write  — create or overwrite a file within $HOME
 *   file_edit   — replace exact text in a file (first occurrence)
 *   file_glob   — find files matching a glob pattern
 *   file_grep   — search file contents with a regex
 *
 * All path operations are sandboxed to $HOME inside the MCP server.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { McpStdioClient } from "../memory-graph/mcp-client.js";

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.(
    "file-tools: registering file_read, file_write, file_edit, file_glob, file_grep (proxied through MCP server)",
  );

  const mcpClient = new McpStdioClient();
  process.on("exit", () => mcpClient.destroy());

  // ── file_read ──────────────────────────────────────────────────────────────

  api.registerTool({
    name: "file_read",
    label: "Read File",
    description:
      "Read a file from the filesystem. path must be absolute or ~/relative (sandboxed to $HOME). " +
      "Supports optional line range via start_line / end_line (1-indexed).",
    parameters: {
      type: "object" as const,
      properties: {
        path: {
          type: "string" as const,
          description: "Absolute or ~/relative file path",
        },
        start_line: {
          type: "integer" as const,
          description: "First line to return (1-indexed, 0 = beginning)",
        },
        end_line: {
          type: "integer" as const,
          description: "Last line to return (0 = end of file)",
        },
      },
      required: ["path"] as string[],
    },
    async execute(
      _id: string,
      params: { path: string; start_line?: number; end_line?: number },
    ) {
      try {
        const content = await mcpClient.callTool("file_read", params as Record<string, unknown>);
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `file_read error: ${err.message}` }] };
      }
    },
  });

  // ── file_write ─────────────────────────────────────────────────────────────

  api.registerTool({
    name: "file_write",
    label: "Write File",
    description:
      "Create or overwrite a file. path must be absolute or ~/relative (sandboxed to $HOME). " +
      "Creates parent directories automatically.",
    parameters: {
      type: "object" as const,
      properties: {
        path: {
          type: "string" as const,
          description: "Absolute or ~/relative file path",
        },
        content: {
          type: "string" as const,
          description: "Text content to write",
        },
      },
      required: ["path", "content"] as string[],
    },
    async execute(_id: string, params: { path: string; content: string }) {
      try {
        const result = await mcpClient.callTool("file_write", params as Record<string, unknown>);
        const text = result.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `file_write error: ${err.message}` }] };
      }
    },
  });

  // ── file_edit ──────────────────────────────────────────────────────────────

  api.registerTool({
    name: "file_edit",
    label: "Edit File",
    description:
      "Replace an exact string in a file (first occurrence only). " +
      "old_text must appear exactly once — provide more surrounding context if it appears multiple times. " +
      "path must be absolute or ~/relative (sandboxed to $HOME).",
    parameters: {
      type: "object" as const,
      properties: {
        path: {
          type: "string" as const,
          description: "Absolute or ~/relative file path",
        },
        old_text: {
          type: "string" as const,
          description: "Exact text to find (must appear exactly once)",
        },
        new_text: {
          type: "string" as const,
          description: "Replacement text",
        },
      },
      required: ["path", "old_text", "new_text"] as string[],
    },
    async execute(
      _id: string,
      params: { path: string; old_text: string; new_text: string },
    ) {
      try {
        const result = await mcpClient.callTool("file_edit", params as Record<string, unknown>);
        const text = result.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `file_edit error: ${err.message}` }] };
      }
    },
  });

  // ── file_glob ──────────────────────────────────────────────────────────────

  api.registerTool({
    name: "file_glob",
    label: "Glob Files",
    description:
      "Find files matching a glob pattern. " +
      "Returns up to 200 matching paths relative to root. " +
      "Examples: file_glob(\"**/*.py\"), file_glob(\"*.md\", \"~/obsidian\")",
    parameters: {
      type: "object" as const,
      properties: {
        pattern: {
          type: "string" as const,
          description: 'Glob pattern, e.g. "**/*.py", "*.md", "src/**/*.ts"',
        },
        root: {
          type: "string" as const,
          description: "Directory to search from (default: $HOME); must be within $HOME",
        },
      },
      required: ["pattern"] as string[],
    },
    async execute(_id: string, params: { pattern: string; root?: string }) {
      try {
        const content = await mcpClient.callTool("file_glob", params as Record<string, unknown>);
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `file_glob error: ${err.message}` }] };
      }
    },
  });

  // ── file_grep ──────────────────────────────────────────────────────────────

  api.registerTool({
    name: "file_grep",
    label: "Grep Files",
    description:
      "Search file contents with a regular expression. " +
      "Returns matching lines with filename and line number. " +
      "path can be a directory (searched recursively) or a single file.",
    parameters: {
      type: "object" as const,
      properties: {
        pattern: {
          type: "string" as const,
          description: "Python regex pattern to search for",
        },
        path: {
          type: "string" as const,
          description: "File or directory to search (default: $HOME); must be within $HOME",
        },
        file_glob: {
          type: "string" as const,
          description: 'Glob pattern to filter files (default "**/*")',
        },
        max_results: {
          type: "integer" as const,
          description: "Maximum matching lines to return (default 50, max 200)",
        },
      },
      required: ["pattern"] as string[],
    },
    async execute(
      _id: string,
      params: { pattern: string; path?: string; file_glob?: string; max_results?: number },
    ) {
      try {
        const content = await mcpClient.callTool("file_grep", params as Record<string, unknown>);
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `file_grep error: ${err.message}` }] };
      }
    },
  });

  api.logger.info?.(
    "file-tools: registered (5 tools via MCP server)",
  );
}
