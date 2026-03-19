/**
 * gateway.ts — Gateway WebSocket client, device auth, summary generation, TTS triggering
 */

import { readFileSync, existsSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { homedir } from "os";
import type { PluginContext, GwState } from "./types.js";
import { loadDeviceIdentity, buildDeviceAuthPayloadV3, signDevicePayload, base64UrlEncode, derivePublicKeyRaw } from "./device-auth.js";
import { broadcastSse } from "./voice.js";

const WS = require(join(homedir(), ".npm-global/lib/node_modules/openclaw/node_modules/ws"));

// ── Summary helpers ─────────────────────────────────────────────────

export function loadSummaries(summariesFile: string): Record<string, string> {
  try {
    if (existsSync(summariesFile)) return JSON.parse(readFileSync(summariesFile, "utf-8"));
  } catch { /* non-fatal */ }
  return {};
}

function saveSummaries(summariesFile: string, summaries: Record<string, string>) {
  try { mkdirSync(dirname(summariesFile), { recursive: true }); writeFileSync(summariesFile, JSON.stringify(summaries, null, 2)); } catch { /* non-fatal */ }
}

export function stripInjectedContext(text: string): string {
  return text
    .replace(/<daily_notes>[\s\S]*?<\/daily_notes>\s*/i, "")
    .replace(/<active_mode>\w*<\/active_mode>\s*/i, "")
    .replace(/<memory_context>[\s\S]*?<\/memory_context>\s*/i, "")
    .replace(/\[(?:[A-Z][a-z]{2} )?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?[^\]]*\]\s*/, "")
    .replace(/A new session was started via\b[\s\S]*/i, "")
    .replace(/Sender \(untrusted metadata\):[\s]*(?:```json\s*\{[\s\S]*?\}\s*```|\{[\s\S]*?\})\s*/i, "")
    .trim();
}

export function parseSessionSource(sessionKey: string): { source: string; peer: string | null } {
  const parts = sessionKey.split(":");
  const channel = parts[2] || "";
  const knownChannels = ["discord", "telegram", "signal", "whatsapp"];
  if (knownChannels.includes(channel)) {
    if (channel === "discord" && parts[3] === "channel") return { source: "discord", peer: null };
    if (parts[3] === "direct" || parts[3] === "dm") {
      return { source: channel, peer: parts[4] || null };
    }
    const peerId = parts.slice(3).join(":") || null;
    return { source: channel, peer: peerId };
  }
  return { source: "webchat", peer: null };
}

export const GREETING_PROMPT =
  "A new session was started via /new or /reset. Execute your Session Startup " +
  "sequence now. Your daily notes have already been loaded into context above " +
  "— do NOT call mem_get for them. Greet the user in your configured persona, " +
  "if one is provided. Be yourself - use your defined voice, mannerisms, and " +
  "mood. Keep it to 1-3 sentences and ask what they want to do. If the runtime " +
  "model differs from default_model in the system prompt, mention the default " +
  "model. Do not mention internal steps, files, tools, or reasoning.";

// ── Gateway WebSocket setup ─────────────────────────────────────────

export interface GatewayHandle {
  gwWsSend: (method: string, params: Record<string, unknown>) => Promise<any>;
  gwState: GwState;
}

export function setupGateway(ctx: PluginContext): GatewayHandle {
  const { api, configFile, summariesFile } = ctx;

  // Shared WS state across plugin re-loads in the same process
  const GW_STATE_KEY = Symbol.for("mission-control-gw-state");
  if (!(globalThis as any)[GW_STATE_KEY]) {
    (globalThis as any)[GW_STATE_KEY] = {
      ws: null,
      ready: false,
      reqId: 0,
      pending: new Map(),
      streamTextAccum: "",
      streamTtsInFlight: false,
    } as GwState;
  }
  const gwState: GwState = (globalThis as any)[GW_STATE_KEY];

  const deviceIdentity = loadDeviceIdentity();

  function readGatewayToken(): string {
    try {
      const cfg = JSON.parse(readFileSync(configFile, "utf-8"));
      const tok = cfg.gateway?.auth?.token;
      if (!tok) return "";
      if (typeof tok === "string") return tok;
      if (tok.source === "env" && tok.id) return process.env[tok.id] || "";
      return "";
    } catch (err: any) {
      api.logger.error?.(`mission-control: failed to read gateway token from ${configFile}: ${err.message}`);
      return "";
    }
  }

  const pendingSummaries = new Set<string>();
  const summaryFailures = new Map<string, number>();

  async function generateSummary(sessionKey: string): Promise<string | null> {
    try {
      const result = await gwWsSend("chat.history", { sessionKey, limit: 10 });
      const messages: any[] = result?.messages || result?.history || [];
      let text: string | null = null;
      for (const msg of messages) {
        const role = msg.role || msg.message?.role;
        if (role !== "user") continue;
        const content = msg.content || msg.message?.content || [];
        const parts = Array.isArray(content) ? content : [{ type: "text", text: String(content) }];
        for (const p of parts) {
          if (p.type === "text" && p.text) {
            const cleaned = stripInjectedContext(p.text);
            if (cleaned.length > 5) { text = cleaned; break; }
          }
        }
        if (text) break;
      }
      if (!text) return null;

      api.logger.info?.(`mc-summary-input: ${sessionKey} text="${text.slice(0, 100)}"`);
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 5000);
      try {
        const sysMsg = "Summarize this conversation opener in 3-6 words. No quotes, no punctuation at the end. Just a brief title.";
        const userMsg = text.slice(0, 500);
        const prompt = `<|im_start|>system\n${sysMsg}<|im_end|>\n<|im_start|>user\n${userMsg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
        const resp = await fetch("http://127.0.0.1:8091/completion", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            n_predict: 40,
            temperature: 0.3,
            stop: ["<|im_end|>"],
          }),
          signal: controller.signal,
        });
        clearTimeout(timer);
        const data = await resp.json() as any;
        return data?.content?.trim() || null;
      } catch {
        clearTimeout(timer);
        return null;
      }
    } catch (e) {
      api.logger.debug?.(`mission-control: generateSummary error: ${(e as Error).message}`);
      return null;
    }
  }

  function triggerSummaryGeneration(sessionKey: string) {
    if (pendingSummaries.has(sessionKey)) return;
    const failures = summaryFailures.get(sessionKey) || 0;
    if (failures >= 3) return;
    pendingSummaries.add(sessionKey);
    generateSummary(sessionKey).then((summary) => {
      api.logger.info?.(`mc-summary: ${sessionKey} -> "${summary}"`);
      pendingSummaries.delete(sessionKey);
      if (summary) {
        summaryFailures.delete(sessionKey);
        const summaries = loadSummaries(summariesFile);
        summaries[sessionKey] = summary;
        saveSummaries(summariesFile, summaries);
      } else {
        summaryFailures.set(sessionKey, failures + 1);
      }
    }).catch(() => {
      pendingSummaries.delete(sessionKey);
      summaryFailures.set(sessionKey, failures + 1);
    });
  }

  function triggerStreamTts(accum: string) {
    const match = accum.match(/<summary>([\s\S]*?)<\/summary>/);
    if (!match) return;
    const summaryText = match[1].trim();
    if (!summaryText) return;

    api.logger.info?.(`mc-stream-tts: firing TTS for summary (${summaryText.length} chars)`);

    (async () => {
      try {
        const ttsResp = await fetch("http://127.0.0.1:8090/v1/audio/speech", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-local",
          },
          body: JSON.stringify({
            input: summaryText,
            model: "tts-1",
            voice: "clone:cullen",
            response_format: "mp3",
          }),
        });
        if (!ttsResp.ok || !ttsResp.body) {
          api.logger.error?.("mc-stream-tts: TTS synthesis failed");
          return;
        }
        const chunks: Uint8Array[] = [];
        const reader = ttsResp.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) chunks.push(value);
        }
        const total = chunks.reduce((s, c) => s + c.length, 0);
        const merged = new Uint8Array(total);
        let offset = 0;
        for (const c of chunks) { merged.set(c, offset); offset += c.length; }
        const audio = Buffer.from(merged).toString("base64");
        broadcastSse("tts_mp3", JSON.stringify({ audio, mimeType: "audio/mpeg" }));
        api.logger.info?.("mc-stream-tts: broadcast tts_mp3 via SSE");
      } catch (err: any) {
        api.logger.error?.(`mc-stream-tts: error: ${err.message}`);
      }
    })();
  }

  function gwWsConnect() {
    if (gwState.ws && (gwState.ws.readyState === WS.OPEN || gwState.ws.readyState === WS.CONNECTING)) return;
    if (!deviceIdentity) {
      api.logger.error?.("mission-control: no device identity found, cannot connect to gateway WS");
      return;
    }
    gwState.ready = false;
    try {
      const gwPort = api.config?.gateway?.port ?? 18789;
      const useTls = !!api.config?.gateway?.tls?.enabled;
      const wsProto = useTls ? "wss" : "ws";
      const httpProto = useTls ? "https" : "http";
      const ws = new WS(`${wsProto}://127.0.0.1:${gwPort}`, {
        headers: { Origin: `${httpProto}://127.0.0.1:${gwPort}` },
      });
      gwState.ws = ws;

      ws.on("message", (data: Buffer | string) => {
        try {
          const msg = JSON.parse(typeof data === "string" ? data : data.toString());

          if (msg.type === "event" && msg.event === "connect.challenge" && msg.payload?.nonce) {
            const nonce: string = msg.payload.nonce;
            const signedAtMs = Date.now();
            const clientId = "webchat-ui";
            const clientMode = "webchat";
            const role = "operator";
            const scopes = ["operator.admin"];

            const payload = buildDeviceAuthPayloadV3({
              deviceId: deviceIdentity.deviceId,
              clientId,
              clientMode,
              role,
              scopes,
              signedAtMs,
              token: readGatewayToken(),
              nonce,
              platform: "linux",
              deviceFamily: "",
            });
            const signature = signDevicePayload(deviceIdentity.privateKeyPem, payload);
            const publicKeyBase64Url = base64UrlEncode(derivePublicKeyRaw(deviceIdentity.publicKeyPem));

            const connectId = `mc-connect-${++gwState.reqId}`;
            ws.send(JSON.stringify({
              type: "req",
              id: connectId,
              method: "connect",
              params: {
                minProtocol: 3,
                maxProtocol: 3,
                client: {
                  id: clientId,
                  displayName: "Mission Control",
                  version: "1.0.0",
                  platform: "linux",
                  mode: clientMode,
                },
                role,
                scopes,
                device: {
                  id: deviceIdentity.deviceId,
                  publicKey: publicKeyBase64Url,
                  signature,
                  signedAt: signedAtMs,
                  nonce,
                },
                ...(readGatewayToken() ? { auth: { token: readGatewayToken() } } : {}),
              },
            }));
            return;
          }

          // Accumulate streaming text for TTS detection
          if (msg.type === "event") {
            const payload = msg.payload || {};
            let delta: string | null = null;

            if (msg.event === "agent" && payload.stream === "assistant" && payload.data?.delta) {
              delta = payload.data.delta;
            }

            if (delta) {
              gwState.streamTextAccum += delta;
            }

            if (msg.event === "agent" && payload.stream === "lifecycle") {
              const phase = payload.data?.phase;
              if (phase === "end" || phase === "complete" || phase === "done" || phase === "stop") {
                if (!gwState.streamTtsInFlight) {
                  triggerStreamTts(gwState.streamTextAccum);
                }
                gwState.streamTextAccum = "";
                gwState.streamTtsInFlight = false;
              }
            }

            if (!gwState.streamTtsInFlight && gwState.streamTextAccum.includes("</summary>")) {
              triggerStreamTts(gwState.streamTextAccum);
              gwState.streamTtsInFlight = true;
            }
          }

          if (msg.type === "res") {
            if (msg.payload?.type === "hello-ok") {
              gwState.ready = true;
              api.logger.info?.("mission-control: gateway WS connected (device auth OK)");
            }
            const entry = gwState.pending.get(msg.id);
            if (entry) {
              gwState.pending.delete(msg.id);
              clearTimeout(entry.timer);
              if (msg.ok) {
                entry.resolve(msg.payload);
              } else {
                entry.reject(new Error(msg.error?.message || "Gateway request failed"));
              }
            }
          }
        } catch {
          // Ignore malformed messages
        }
      });

      ws.on("close", () => {
        gwState.ready = false;
        gwState.ws = null;
        for (const [id, entry] of gwState.pending) {
          clearTimeout(entry.timer);
          entry.reject(new Error("WebSocket closed"));
          gwState.pending.delete(id);
        }
        setTimeout(gwWsConnect, 5000);
      });

      ws.on("error", (err: Error) => {
        api.logger.error?.(`mission-control: gateway WS error: ${err.message}`);
      });
    } catch (err: any) {
      api.logger.error?.(`mission-control: gateway WS connect failed: ${err.message}`);
      setTimeout(gwWsConnect, 5000);
    }
  }

  function gwWsSend(method: string, params: Record<string, unknown>): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!gwState.ws || !gwState.ready) {
        reject(new Error("Gateway WebSocket not connected"));
        return;
      }
      const id = `mc-${++gwState.reqId}`;
      const timer = setTimeout(() => {
        gwState.pending.delete(id);
        reject(new Error("Gateway request timeout"));
      }, 60_000);
      gwState.pending.set(id, { resolve, reject, timer });
      gwState.ws.send(JSON.stringify({ type: "req", id, method, params }));
    });
  }

  // Start gateway WS — only one connection per process
  const GW_WS_KEY = Symbol.for("mission-control-gw-ws-active");
  if (!(globalThis as any)[GW_WS_KEY]) {
    (globalThis as any)[GW_WS_KEY] = true;
    setTimeout(gwWsConnect, 2000);
  } else {
    api.logger.info?.("mission-control: skipping gateway WS (already active in this process)");
  }

  // Expose the trigger for session routes
  (gwWsSend as any).triggerSummaryGeneration = triggerSummaryGeneration;

  return { gwWsSend, gwState };
}

/** Get the triggerSummaryGeneration function attached to gwWsSend */
export function getTriggerSummaryGeneration(gwWsSend: any): (sessionKey: string) => void {
  return gwWsSend.triggerSummaryGeneration;
}
