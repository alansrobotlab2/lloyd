/**
 * prefill.ts — Thin before_prompt_build hook.
 *
 * Delegates the full prefill pipeline to the Python MCP server's
 * prefill_context tool. All orchestration logic (tag match, BM25,
 * GLM keywords, merge/rank, format) lives in server.py.
 */

import type { McpStdioClient } from "./mcp-client.js";

const MIN_QUERY_LENGTH = 12;

export function createPrefillHook(
  mcpClient: McpStdioClient,
  logger: { info?: (...args: any[]) => void; warn?: (...args: any[]) => void } | undefined,
) {
  return async (event: any, ctx: any) => {
    const prompt: string = event.prompt ?? "";
    if (!prompt || prompt.length < MIN_QUERY_LENGTH) return;

    try {
      const content = await mcpClient.callTool("prefill_context", {
        prompt,
        session_id: ctx?.sessionId ?? "",
      });
      const text = content
        .filter((b: any) => b.type === "text")
        .map((b: any) => b.text)
        .join("")
        .trim();
      if (text) return { prependContext: text };
    } catch (err: any) {
      logger?.warn?.(`memory-prefill: ${err?.message}`);
    }
  };
}
