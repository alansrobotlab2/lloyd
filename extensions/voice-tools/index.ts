/**
 * index.ts — voice-tools OpenClaw plugin
 *
 * Connects to the standalone voice MCP server (voice_services.py) over SSE
 * transport at http://127.0.0.1:8094. The server runs as a systemd service
 * (lloyd-voice-mcp.service).
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { McpSseClient } from "../mcp-tools/mcp-sse-client.js";
import { request as httpsRequest } from "node:https";

const VOICE_MCP_URL = "http://127.0.0.1:8094";
const TTS_API_URL = "http://127.0.0.1:8090/v1/audio/speech";

function postToMc(port: number, path: string, body: object, token?: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Content-Length": String(Buffer.byteLength(data)),
    };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const req = httpsRequest(
      {
        hostname: "127.0.0.1",
        port,
        path,
        method: "POST",
        headers,
        rejectUnauthorized: false,
      },
      (res) => {
        res.resume(); // drain
        resolve();
      },
    );
    req.on("error", reject);
    req.end(data);
  });
}

const SKIP_PATTERNS = new Set(["NO_REPLY", "HEARTBEAT_OK"]);

function stripMarkdown(text: string): string {
  return text
    .replace(/#{1,6}\s+/g, '')       // headers
    .replace(/\*\*(.+?)\*\*/g, '$1') // bold
    .replace(/\*(.+?)\*/g, '$1')     // italic
    .replace(/_(.+?)_/g, '$1')       // italic underscore
    .replace(/`(.+?)`/g, '$1')       // inline code
    .trim();
}

/** Extract TTS summary: first paragraph (before \n\n), with fallback */
function extractFirstParagraph(text: string): string | null {
  if (!text || !text.trim()) return null;
  const trimmed = text.trim();

  // Skip known non-speech patterns
  if (SKIP_PATTERNS.has(trimmed)) return null;

  // Primary: split on double newline
  const idx = trimmed.indexOf("\n\n");
  if (idx > 0) {
    const first = trimmed.slice(0, idx).trim();
    return first ? stripMarkdown(first) : null;
  }

  // No \n\n: short text IS the summary
  if (trimmed.length <= 300) return stripMarkdown(trimmed);

  // Fallback: first 2 sentences or 300 chars at word boundary
  const sentenceMatch = trimmed.match(/^((?:[^.!?]*[.!?]){1,2})\s/);
  if (sentenceMatch) {
    return stripMarkdown(sentenceMatch[1].trim());
  }

  const boundary = trimmed.lastIndexOf(" ", 300);
  return stripMarkdown(trimmed.slice(0, boundary > 0 ? boundary : 300).trim());
}

export default function register(api: OpenClawPluginApi) {
  api.logger.info?.(
    "voice-tools: registering voice tools (proxied through voice MCP SSE server)",
  );

  const mcpClient = new McpSseClient(VOICE_MCP_URL);
  process.on("exit", () => mcpClient.destroy());

  // -- voice_last_utterance --

  api.registerTool({
    name: "voice_last_utterance",
    label: "Last Voice Utterance",
    description:
      "Get the last voice utterance heard by Lloyd. Returns transcript with speaker identification " +
      "(e.g. \"[Alan]: hello\"), raw transcript, timestamp, duration, and speaker segments. " +
      "Use this to see what was just said and who said it.",
    parameters: {
      type: "object" as const,
      properties: {},
    },
    async execute(_id: string, _params: Record<string, unknown>) {
      try {
        const content = await mcpClient.callTool("voice_last_utterance", {});
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return {
          content: [{ type: "text" as const, text: `voice_last_utterance error: ${err.message}` }],
        };
      }
    },
  });

  // -- voice_enroll_speaker --

  api.registerTool({
    name: "voice_enroll_speaker",
    label: "Enroll Speaker",
    description:
      "Enroll a speaker using audio from the last utterance. Creates a voice profile so Lloyd can " +
      "identify this person by name in future utterances. Takes effect immediately. " +
      "Example: voice_enroll_speaker({name: \"Alan\"})",
    parameters: {
      type: "object" as const,
      properties: {
        name: {
          type: "string" as const,
          description: "Name to assign to the speaker (e.g. \"Alan\", \"Sarah\")",
        },
        speaker_label: {
          type: "string" as const,
          description:
            "Optional diarization label (e.g. \"SPEAKER_00\") to isolate one speaker " +
            "from a multi-speaker utterance. If omitted, uses the full utterance audio.",
        },
      },
      required: ["name"] as string[],
    },
    async execute(
      _id: string,
      params: { name: string; speaker_label?: string },
    ) {
      try {
        const content = await mcpClient.callTool(
          "voice_enroll_speaker",
          params as Record<string, unknown>,
        );
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return {
          content: [{ type: "text" as const, text: `voice_enroll_speaker error: ${err.message}` }],
        };
      }
    },
  });

  // -- voice_list_speakers --

  api.registerTool({
    name: "voice_list_speakers",
    label: "List Speakers",
    description:
      "List all enrolled speaker profiles that Lloyd can identify by voice.",
    parameters: {
      type: "object" as const,
      properties: {},
    },
    async execute(_id: string, _params: Record<string, unknown>) {
      try {
        const content = await mcpClient.callTool("voice_list_speakers", {});
        const text = content.map((c) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return {
          content: [{ type: "text" as const, text: `voice_list_speakers error: ${err.message}` }],
        };
      }
    },
  });

  // -- voice_toggle --

  api.registerTool({
    name: "voice_toggle",
    label: "Toggle Voice Mode",
    description:
      "Toggle Lloyd's voice mode on or off. When disabled, the pipeline ignores " +
      "all microphone input. When re-enabled, listening resumes immediately. " +
      "Returns the new state (enabled or disabled).",
    parameters: {
      type: "object" as const,
      properties: {},
    },
    async execute(_id: string, _params: Record<string, unknown>) {
      try {
        const resp = await fetch("http://127.0.0.1:8092/v1/voice/toggle", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        const data = await resp.json();
        const enabled = data.voice_enabled;
        return {
          content: [{ type: "text" as const, text: `Voice mode ${enabled ? "enabled" : "disabled"}.` }],
        };
      } catch (err: any) {
        return {
          content: [{ type: "text" as const, text: `voice_toggle error: ${err.message}` }],
        };
      }
    },
  });

  // -- message_sending hook: extract first paragraph, synthesize TTS, push via MC SSE --

  const gwPort = (api as any).config?.gateway?.port ?? 18789;
  const gwToken = (api as any).config?.gateway?.auth?.token as string | undefined;

  api.on("message_sending", async (event) => {
    const content = event.content;
    if (!content) return;

    const summary = extractFirstParagraph(content);
    if (!summary) return;

    // Synthesize first paragraph via Qwen3-TTS and push MP3 to MC SSE
    try {
      const ttsResp = await fetch(TTS_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer sk-local",
        },
        body: JSON.stringify({
          input: summary,
          model: "tts-1",
          voice: "clone:cullen",
          response_format: "mp3",
        }),
      });
      if (!ttsResp.ok) {
        api.logger.warn?.(`voice-tools: TTS synthesis failed (${ttsResp.status})`);
        return;
      }
      const audioBuffer = await ttsResp.arrayBuffer();
      const base64Audio = Buffer.from(audioBuffer).toString("base64");
      await postToMc(gwPort, "/api/mc/tts-push", { audio: base64Audio }, gwToken);
      api.logger.info?.(`voice-tools: TTS pushed via SSE (${summary.length} chars, ${audioBuffer.byteLength} bytes)`);
    } catch (err: any) {
      api.logger.warn?.(`voice-tools: TTS delivery failed: ${err.message}`);
    }
  });

  api.logger.info?.("voice-tools: registered (3 tools + message_sending hook)");
}
