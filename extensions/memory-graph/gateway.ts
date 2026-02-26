/**
 * gateway.ts — HTTP gateway client for invoking OpenClaw tools.
 *
 * Plugins cannot invoke sibling tools directly; all tool calls go
 * through the gateway HTTP endpoint.
 */

const GATEWAY_URL = "http://127.0.0.1:18789/tools/invoke";

/** Invoke a tool via the OpenClaw HTTP gateway. */
export async function gatewayInvoke(
  tool: string,
  args: Record<string, unknown>,
  signal: AbortSignal,
): Promise<any> {
  const resp = await fetch(GATEWAY_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ tool, args }),
    signal,
  });
  if (!resp.ok) throw new Error(`gateway ${tool} failed: ${resp.status}`);
  const data = (await resp.json()) as any;
  if (!data.ok) throw new Error(`gateway ${tool} error: ${data.error}`);
  return data.result;
}
