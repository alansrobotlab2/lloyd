/**
 * index.ts — voice-tools OpenClaw plugin
 *
 * Connects to the standalone voice MCP server (voice_services.py) over SSE
 * transport at http://127.0.0.1:8094. The server runs as a systemd service
 * (lloyd-voice-mcp.service).
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { McpSseClient } from "../mcp-tools/mcp-sse-client.js";

const VOICE_MCP_URL = "http://127.0.0.1:8094";

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

  // -- message_sending hook: extract <summary>, send to TTS, strip from display --

  const VOICE_TUI_URL = "http://127.0.0.1:8092";

  api.on("message_sending", async (event) => {
    const content = event.content;
    if (!content) return;

    const summaryRegex = /<summary>([\s\S]*?)<\/summary>/g;
    const matches: string[] = [];
    let m: RegExpExecArray | null;
    while ((m = summaryRegex.exec(content)) !== null) {
      const text = m[1].trim();
      if (text) matches.push(text);
    }

    if (matches.length === 0) return;

    // Send each summary block to voice TUI for TTS playback
    for (const text of matches) {
      try {
        await fetch(`${VOICE_TUI_URL}/v1/say`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        api.logger.info?.(`voice-tools: sent summary to TTS (${text.length} chars)`);
      } catch (err: any) {
        api.logger.warn?.(`voice-tools: TTS delivery failed: ${err.message}`);
      }
    }

    // Strip <summary> tags from displayed output
    let stripped = content.replace(/<summary>[\s\S]*?<\/summary>/g, "").trim();
    stripped = stripped.replace(/\n{3,}/g, "\n\n");

    if (stripped !== content) {
      return { content: stripped };
    }
  });

  api.logger.info?.("voice-tools: registered (3 tools + message_sending hook)");
}
