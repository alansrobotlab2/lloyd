/**
 * index.ts — toolkit-router: keyword-activated toolkit routing.
 *
 * Reads ~/obsidian/tools.md to build a keyword→toolkit map, then
 * intercepts before_prompt_build to return an enableTools list that
 * includes only the always-on core tools plus any toolkits whose
 * keywords match words in the user's prompt.
 *
 * Session-sticky: once a toolkit activates in a session, it stays
 * active for the rest of that session (prevents KV cache breakage
 * from tools appearing/disappearing between turns).
 *
 * Fail-open: if tools.md can't be parsed, no filtering is applied.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { readFileSync } from "fs";
import { join } from "path";

interface Toolkit {
  name: string;
  tools: string[];
  keywords: Set<string>;
}

const TOOLS_MD_PATH = join(
  process.env.HOME || "/home/alansrobotlab",
  "obsidian",
  "tools.md",
);
const RELOAD_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes
const SESSION_CLEANUP_MS = 2 * 60 * 60 * 1000; // 2 hours

let alwaysOnTools: string[] = [];
let toolkits: Toolkit[] = [];
let lastParsedAt = 0;

// Session-sticky: sessionKey -> { activated toolkit names, last activity }
const sessionState = new Map<
  string,
  { toolkits: Set<string>; lastSeen: number }
>();

function parseToolsMd(): boolean {
  try {
    const raw = readFileSync(TOOLS_MD_PATH, "utf-8");
    const newAlwaysOn: string[] = [];
    const newToolkits: Toolkit[] = [];

    // Split by ## headers — first element is preamble before any ##
    const sections = raw.split(/^## /m).slice(1);

    for (const section of sections) {
      const lines = section.split("\n");
      const name = lines[0].trim();

      if (name === "Always-On Core") {
        const toolsLine = lines.find((l) => l.startsWith("**Tools:**"));
        if (toolsLine) {
          const tools = toolsLine
            .replace("**Tools:**", "")
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean);
          newAlwaysOn.push(...tools);
        }
        continue;
      }

      if (name === "Ungrouped") {
        // Each line: **tool_name:** keyword1, keyword2, ...
        for (const line of lines.slice(1)) {
          const match = line.match(/^\*\*(\w+):\*\*\s*(.+)/);
          if (match) {
            const toolName = match[1];
            const keywords = match[2]
              .split(",")
              .map((k) => k.trim().toLowerCase())
              .filter(Boolean);
            newToolkits.push({
              name: `ungrouped:${toolName}`,
              tools: [toolName],
              keywords: new Set(keywords),
            });
          }
        }
        continue;
      }

      // Regular toolkit section
      const toolsLine = lines.find((l) => l.startsWith("**Tools:**"));
      const keywordsLine = lines.find((l) => l.startsWith("**Keywords:**"));

      if (toolsLine && keywordsLine) {
        const tools = toolsLine
          .replace("**Tools:**", "")
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean);
        const keywords = keywordsLine
          .replace("**Keywords:**", "")
          .split(",")
          .map((k) => k.trim().toLowerCase())
          .filter(Boolean);
        newToolkits.push({ name, tools, keywords: new Set(keywords) });
      }
    }

    alwaysOnTools = newAlwaysOn;
    toolkits = newToolkits;
    lastParsedAt = Date.now();
    return true;
  } catch (err: any) {
    console.warn(
      `[toolkit-router] Failed to parse ${TOOLS_MD_PATH}: ${err.message}`,
    );
    alwaysOnTools = [];
    toolkits = [];
    return false;
  }
}

function maybeReload(): void {
  if (Date.now() - lastParsedAt > RELOAD_INTERVAL_MS) {
    parseToolsMd();
  }
}

function cleanupSessions(): void {
  const now = Date.now();
  for (const [key, state] of sessionState) {
    if (now - state.lastSeen > SESSION_CLEANUP_MS) {
      sessionState.delete(key);
    }
  }
}

export default function register(api: OpenClawPluginApi) {
  // Initial parse
  const ok = parseToolsMd();
  if (ok) {
    const totalTools =
      toolkits.reduce((sum, tk) => sum + tk.tools.length, 0) +
      alwaysOnTools.length;
    const totalKeywords = toolkits.reduce(
      (sum, tk) => sum + tk.keywords.size,
      0,
    );
    console.log(
      `[toolkit-router] Loaded ${toolkits.length} toolkits, ` +
        `${alwaysOnTools.length} always-on tools, ` +
        `${totalTools} total tools, ${totalKeywords} total keywords`,
    );
  } else {
    console.warn(
      "[toolkit-router] Failed to load tools.md — running in fail-open mode (all tools enabled)",
    );
  }

  // Periodic session cleanup (every 30 min)
  setInterval(cleanupSessions, 30 * 60 * 1000);

  api.on("before_prompt_build", async (event: any, ctx: any) => {
    maybeReload();

    // Fail open if parse failed — no filtering
    if (alwaysOnTools.length === 0) return undefined;

    const prompt: string = event?.prompt || "";

    // Empty/missing prompt — return only always-on tools
    if (!prompt.trim()) {
      return { enableTools: [...alwaysOnTools, "prefill_context"] };
    }

    // Skip filtering for heartbeat/system prompts — allow all tools
    if (
      /\bHEARTBEAT(?:_OK|\.md)?\b/i.test(prompt) ||
      /\bPost-Compaction Audit\b/i.test(prompt)
    ) {
      return undefined;
    }

    // Strip sender metadata and timestamp framing before keyword matching
    let cleanPrompt = prompt
      .replace(/Sender \(untrusted metadata\):\s*```json[\s\S]*?```/g, "")
      .replace(/\[\w{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2} \w+\]/g, "")
      .replace(/\[message_id:\s*[^\]]*\]/g, "");

    // Tokenize prompt for O(1) keyword lookup
    const words = new Set(
      cleanPrompt
        .toLowerCase()
        .replace(/[^\w'-]/g, " ")
        .split(/\s+/)
        .filter(Boolean),
    );

    // Session key for sticky activations
    const sessionKey = ctx?.sessionKey || ctx?.sessionId || "default";

    // Get or create session state
    let session = sessionState.get(sessionKey);
    if (!session) {
      session = { toolkits: new Set(), lastSeen: Date.now() };
      sessionState.set(sessionKey, session);
    }
    session.lastSeen = Date.now();

    // Match toolkits by keywords
    const matchedNames: string[] = [];
    const matchedKeywordsLog: string[] = [];

    for (const tk of toolkits) {
      // Already activated in this session (sticky)
      if (session.toolkits.has(tk.name)) {
        matchedNames.push(tk.name);
        continue;
      }

      // Check for keyword match
      for (const kw of tk.keywords) {
        if (words.has(kw)) {
          matchedNames.push(tk.name);
          matchedKeywordsLog.push(`${tk.name}→"${kw}"`);
          session.toolkits.add(tk.name); // sticky activation
          break;
        }
      }
    }

    // Build enableTools set
    const enableTools = new Set(alwaysOnTools);
    enableTools.add("prefill_context"); // mcp-tools needs this for its own hook

    for (const name of matchedNames) {
      const tk = toolkits.find((t) => t.name === name);
      if (tk) {
        for (const tool of tk.tools) {
          enableTools.add(tool);
        }
      }
    }

    if (matchedKeywordsLog.length > 0) {
      console.debug(
        `[toolkit-router] Activated: ${matchedKeywordsLog.join(", ")} (session: ${sessionKey})`,
      );
    }

    const stickyCount = matchedNames.length - matchedKeywordsLog.length;
    if (stickyCount > 0) {
      console.debug(
        `[toolkit-router] ${stickyCount} sticky toolkit(s) from prior turns`,
      );
    }

    return { enableTools: [...enableTools] };
  });

  console.log("[toolkit-router] Plugin registered");
}
