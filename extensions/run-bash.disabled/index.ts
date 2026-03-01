/**
 * index.ts — run-bash OpenClaw plugin
 *
 * Proxies the run_bash tool through the shared MCP server subprocess
 * (extensions/mcp-server/server.py) via McpStdioClient (JSON-RPC 2.0 stdio).
 *
 * Uses a 120s MCP call timeout to support long-running commands (builds,
 * tests, installs). The Python-side timeout parameter enforces the hard cap.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { McpStdioClient } from "../memory-graph/mcp-client.js";

const RUN_BASH_TIMEOUT_MS = 120_000;

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.("run-bash: registering run_bash (proxied through MCP server)");

  const mcpClient = new McpStdioClient();
  process.on("exit", () => mcpClient.destroy());

  api.registerTool({
    name: "run_bash",
    label: "Run Shell Command",
    description:
      "Execute a bash command and return combined stdout+stderr with exit code. " +
      "command is passed to bash -c. cwd must be within $HOME (default: $HOME). " +
      "timeout is max seconds to wait (default 30, max 120). " +
      "Examples: run_bash({command: \"git status\"}), " +
      "run_bash({command: \"npm test\", cwd: \"~/myproject\", timeout: 60})",
    parameters: {
      type: "object" as const,
      properties: {
        command: {
          type: "string" as const,
          description: "Bash command string (passed to bash -c)",
        },
        cwd: {
          type: "string" as const,
          description: "Working directory (absolute or ~/relative, must be within $HOME; default: $HOME)",
        },
        timeout: {
          type: "integer" as const,
          description: "Max seconds to wait (default 30, max 120)",
        },
      },
      required: ["command"] as string[],
    },
    async execute(
      _id: string,
      params: { command: string; cwd?: string; timeout?: number },
    ) {
      try {
        const content = await mcpClient.callTool(
          "run_bash",
          params as Record<string, unknown>,
          RUN_BASH_TIMEOUT_MS,
        );
        const text = content.map((c) => c.text).join("") || "(no output)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `run_bash error: ${err.message}` }] };
      }
    },
  });

  api.logger.info?.("run-bash: registered (1 tool via MCP server)");
}
