/**
 * voice.ts — Voice pipeline SSE bridge, TTS proxy, and voice commands
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import type { PluginContext } from "./types.js";
import { jsonResponse } from "./helpers.js";
import { join } from "path";
import { homedir } from "os";

const WS = require(join(homedir(), ".npm-global/lib/node_modules/openclaw/node_modules/ws"));

// ── Voice Pipeline SSE Bridge ───────────────────────────────────────

export const voiceSseClients = new Set<ServerResponse>();
let voicePipeWs: any = null;
let voicePipeWsReady = false;
let voicePipeConnecting = false;

export function broadcastSse(event: string, data: string) {
  for (const res of voiceSseClients) {
    res.write(`event: ${event}\ndata: ${data}\n\n`);
  }
}

export function voicePipeConnect() {
  if (voicePipeWs && voicePipeWs.readyState <= 1) return;
  if (voicePipeConnecting) return;
  voicePipeConnecting = true;
  try {
    const ws = new WS("ws://127.0.0.1:8095");
    ws.binaryType = "arraybuffer";
    voicePipeWs = ws;

    ws.on("open", () => {
      voicePipeConnecting = false;
      voicePipeWsReady = true;
      ws.send(JSON.stringify({ type: "hello" }));
      broadcastSse("connected", JSON.stringify({ connected: true }));
    });

    ws.on("close", () => {
      voicePipeWsReady = false;
      broadcastSse("connected", JSON.stringify({ connected: false }));
      voicePipeWs = null;
      setTimeout(() => voicePipeConnect(), 3000);
    });

    ws.on("error", () => {
      voicePipeConnecting = false;
    });

    ws.on("message", (rawData: any, isBinary: boolean) => {
      if (isBinary) {
        const buf = Buffer.isBuffer(rawData) ? rawData : Buffer.from(rawData);
        broadcastSse("tts_audio", buf.toString("base64"));
      } else {
        try {
          const msg = JSON.parse(typeof rawData === "string" ? rawData : rawData.toString("utf-8"));
          if (msg.type === "state") {
            broadcastSse("state", JSON.stringify({ state: msg.state }));
          } else if (msg.type === "transcript") {
            broadcastSse("transcript", JSON.stringify({ text: msg.text, speaker: msg.speaker, is_continuity: msg.is_continuity }));
          } else if (msg.type === "tts_start") {
            broadcastSse("tts_start", JSON.stringify({ sample_rate: msg.sample_rate || 24000 }));
          } else if (msg.type === "tts_end") {
            broadcastSse("tts_end", JSON.stringify({}));
          } else if (msg.type === "wakeword") {
            broadcastSse("wakeword", JSON.stringify({ detected: msg.detected }));
          }
        } catch { /* skip malformed voice WS message */ }
      }
    });
  } catch {
    voicePipeConnecting = false;
    voicePipeWs = null;
    voicePipeWsReady = false;
    setTimeout(() => voicePipeConnect(), 3000);
  }
}

// ── Voice HTTP routes ───────────────────────────────────────────────

export function registerVoiceRoutes(ctx: PluginContext) {
  const { api } = ctx;

  // GET /api/mc/voice-status
  api.registerHttpRoute({
    path: "/api/mc/voice-status",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const resp = await fetch("http://127.0.0.1:8092/v1/voice/ws-status");
        const data = await resp.json();
        jsonResponse(res, data);
      } catch {
        jsonResponse(res, { ws_active: false, ws_port: 8095, has_client: false, voice_enabled: false, state: "UNKNOWN" });
      }
    },
  });

  // POST /api/mc/voice-toggle
  api.registerHttpRoute({
    path: "/api/mc/voice-toggle",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const resp = await fetch("http://127.0.0.1:8092/v1/voice/toggle", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        const data = await resp.json();
        jsonResponse(res, data);
      } catch (err: any) {
        jsonResponse(res, { error: "Voice service unreachable", detail: err.message }, 502);
      }
    },
  });

  // POST /api/mc/voice-say
  api.registerHttpRoute({
    path: "/api/mc/voice-say",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { res.writeHead(405); res.end(); return; }
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      const body = Buffer.concat(chunks).toString("utf-8");
      try {
        const resp = await fetch("http://127.0.0.1:8092/v1/say", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        });
        const data = await resp.json();
        jsonResponse(res, data);
      } catch {
        res.writeHead(503); res.end("TTS unavailable");
      }
    },
  });

  // POST /api/mc/tts
  api.registerHttpRoute({
    path: "/api/mc/tts",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { res.writeHead(405); res.end(); return; }
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      let text: string;
      try {
        text = JSON.parse(Buffer.concat(chunks).toString("utf-8")).text;
      } catch { res.writeHead(400); res.end("Invalid JSON"); return; }
      if (!text) { res.writeHead(400); res.end("Missing text"); return; }
      try {
        const ttsResp = await fetch("http://127.0.0.1:8090/v1/audio/speech", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-local",
          },
          body: JSON.stringify({
            input: text,
            model: "tts-1",
            voice: "clone:cullen",
            response_format: "mp3",
          }),
        });
        if (!ttsResp.ok || !ttsResp.body) {
          res.writeHead(503); res.end("TTS synthesis failed"); return;
        }
        res.writeHead(200, { "Content-Type": "audio/mpeg" });
        const reader = ttsResp.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(value);
        }
        res.end();
      } catch {
        if (!res.headersSent) { res.writeHead(503); res.end("TTS unavailable"); }
      }
    },
  });

  // POST /api/mc/tts-push
  api.registerHttpRoute({
    path: "/api/mc/tts-push",
    auth: "gateway",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { res.writeHead(405); res.end(); return; }
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      try {
        const { audio } = JSON.parse(Buffer.concat(chunks).toString("utf-8"));
        if (!audio) { res.writeHead(400); res.end("Missing audio"); return; }
        broadcastSse("tts_mp3", JSON.stringify({ audio, mimeType: "audio/mpeg" }));
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
      } catch {
        res.writeHead(400);
        res.end("Invalid JSON");
      }
    },
  });

  // GET /api/mc/voice-stream (SSE)
  api.registerHttpRoute({
    path: "/api/mc/voice-stream",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "GET") { res.writeHead(405); res.end(); return; }
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "http://localhost:5173",
      });
      res.write(":\n\n");
      voiceSseClients.add(res);
      if (!voicePipeWs || voicePipeWs.readyState > 1) voicePipeConnect();
      res.write(`event: connected\ndata: ${JSON.stringify({ connected: voicePipeWsReady })}\n\n`);
      req.on("close", () => {
        voiceSseClients.delete(res);
      });
    },
  });

  // POST /api/mc/voice-send
  api.registerHttpRoute({
    path: "/api/mc/voice-send",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { res.writeHead(405); res.end(); return; }
      if (!voicePipeWs || !voicePipeWsReady) { res.writeHead(503); res.end("Voice pipe not connected"); return; }
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      const body = Buffer.concat(chunks);
      if (body.length > 65536) { res.writeHead(413); res.end("Payload too large"); return; }
      try {
        voicePipeWs.send(body);
      } catch {
        res.writeHead(503); res.end("Voice pipe send failed"); return;
      }
      res.writeHead(200);
      res.end("ok");
    },
  });

  // POST /api/mc/voice-cmd
  api.registerHttpRoute({
    path: "/api/mc/voice-cmd",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { res.writeHead(405); res.end(); return; }
      if (!voicePipeWs || !voicePipeWsReady) { res.writeHead(503); res.end("Voice pipe not connected"); return; }
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      const body = Buffer.concat(chunks).toString("utf-8");
      if (body.length > 1024) { res.writeHead(413); res.end("Payload too large"); return; }
      try {
        voicePipeWs.send(body);
      } catch {
        res.writeHead(503); res.end("Voice pipe send failed"); return;
      }
      jsonResponse(res, { ok: true });
    },
  });

  // Connect eagerly on startup
  voicePipeConnect();
}
