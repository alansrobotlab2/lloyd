#!/usr/bin/env node
/**
 * dump-system-prompt.js — Reconstructs the COMPLETE system prompt
 * (framework boilerplate + workspace files + skills + tool descriptions)
 * into a single markdown file for analysis.
 *
 * Source: buildAgentSystemPrompt() in reply-Duq0R59W.js (OpenClaw v2026.2.26)
 */

const fs = require("fs");
const path = require("path");

const workspace = "/home/alansrobotlab/obsidian/agents/lloyd";
const outPath = "/home/alansrobotlab/.openclaw/docs/system-prompt-full.md";

const out = [];
let totalChars = 0;

// ── Metadata ──
out.push("# Complete System Prompt — Lloyd (main agent)");
out.push("");
out.push("> Reconstructed from `buildAgentSystemPrompt()` in");
out.push("> `reply-Duq0R59W.js` (OpenClaw v2026.2.26) + workspace files.");
out.push("> This is the full text sent as the system message to the LLM.");
out.push("");
out.push("---");
out.push("");

// ════════════════════════════════════════════════════════════════════════════
// PART 1: Framework boilerplate (reconstructed from source)
// ════════════════════════════════════════════════════════════════════════════

const boilerplate = [];

// ── Opening ──
boilerplate.push("You are a personal assistant running inside OpenClaw.");
boilerplate.push("");

// ── Tooling section ──
// These are the actual tools available to the main agent (Lloyd).
// Core tools are resolved from toolOrder + agent's allowed tools.
// External tools come from mcp-tools and voice-tools plugins.
const toolSummaries = {
  // Core (built-in) tools available to main agent:
  read: "Read file contents",
  write: "Create or overwrite files",
  edit: "Make precise edits to files",
  apply_patch: "Apply multi-file patches",
  grep: "Search file contents for patterns",
  find: "Find files by glob pattern",
  ls: "List directory contents",
  exec: "Run shell commands (pty available for TTY-required CLIs)",
  process: "Manage background exec sessions",
  web_search: "Search the web (Brave API)",
  web_fetch: "Fetch and extract readable content from a URL",
  browser: "Control web browser",
  canvas: "Present/eval/snapshot the Canvas",
  nodes: "List/describe/notify/camera/screen on paired nodes",
  cron: "Manage cron jobs and wake events (use for reminders; when scheduling a reminder, write the systemEvent text as something that will read like a reminder when it fires, and mention that it is a reminder depending on the time gap between setting and firing; include recent context in reminder text if appropriate)",
  message: "Send messages and channel actions",
  gateway: "Restart, apply config, or run updates on the running OpenClaw process",
  agents_list: "List OpenClaw agent ids allowed for sessions_spawn when runtime=\"subagent\" (not ACP harness ids)",
  sessions_list: "List other sessions (incl. sub-agents) with filters/last",
  sessions_history: "Fetch history for another session/sub-agent",
  sessions_send: "Send a message to another session/sub-agent",
  sessions_spawn: "Spawn an isolated sub-agent or ACP coding session (runtime=\"acp\" requires `agentId` unless `acp.defaultAgent` is configured; ACP harness ids follow acp.allowedAgents, not agents_list)",
  subagents: "List, steer, or kill sub-agent runs for this requester session",
  session_status: "Show a /status-equivalent status card (usage + time + Reasoning/Verbose/Elevated); use for model-use questions (📊 session_status); optional per-session model override",
  image: "Analyze an image with the configured image model",
  // External tools (from mcp-tools + voice-tools plugins):
  tag_search: "Search the Obsidian knowledge vault by tags",
  tag_explore: "Explore tag relationships in the vault",
  vault_overview: "Show vault statistics and structure",
  qmd_search: "Mandatory recall step: search the Obsidian knowledge vault",
  qmd_get: "Read a specific file from the Obsidian vault by relative path",
  memory_write: "Create or overwrite a file in the Obsidian vault",
  http_search: "Search the web using DuckDuckGo",
  http_fetch: "Fetch a public web page and extract its readable content",
  http_request: "Make a raw HTTP request (GET, POST, PUT, PATCH, DELETE, HEAD)",
  file_read: "Read a file from the filesystem",
  file_write: "Create or overwrite a file",
  file_edit: "Replace an exact string in a file",
  file_patch: "Apply a unified diff patch to one or more files",
  file_glob: "Find files matching a glob pattern",
  file_grep: "Search file contents with a regular expression",
  run_bash: "Execute a bash command and return combined stdout+stderr",
  bg_exec: "Start a background shell command and return a session ID",
  bg_process: "Manage background processes started by bg_exec",
  voice_last_utterance: "Get the last voice utterance heard by Lloyd",
  voice_enroll_speaker: "Enroll a speaker using audio from the last utterance",
  voice_list_speakers: "List all enrolled speaker profiles",
};

boilerplate.push("## Tooling");
boilerplate.push("Tool availability (filtered by policy):");
boilerplate.push("Tool names are case-sensitive. Call tools exactly as listed.");
for (const [name, desc] of Object.entries(toolSummaries)) {
  boilerplate.push(`- ${name}: ${desc}`);
}
boilerplate.push("TOOLS.md does not control tool availability; it is user guidance for how to use external tools.");
boilerplate.push("For long waits, avoid rapid poll loops: use exec with enough yieldMs or process(action=poll, timeout=<ms>).");
boilerplate.push("If a task is more complex or takes longer, spawn a sub-agent. Completion is push-based: it will auto-announce when done.");
boilerplate.push('For requests like "do this in codex/claude code/gemini", treat it as ACP harness intent and call `sessions_spawn` with `runtime: "acp"`.');
boilerplate.push('On Discord, default ACP harness requests to thread-bound persistent sessions (`thread: true`, `mode: "session"`) unless the user asks otherwise.');
boilerplate.push("Set `agentId` explicitly unless `acp.defaultAgent` is configured, and do not route ACP harness requests through `subagents`/`agents_list` or local PTY exec flows.");
boilerplate.push("Do not poll `subagents list` / `sessions_list` in a loop; only check status on-demand (for intervention, debugging, or when explicitly asked).");
boilerplate.push("");

// ── Tool Call Style ──
boilerplate.push("## Tool Call Style");
boilerplate.push("Default: do not narrate routine, low-risk tool calls (just call the tool).");
boilerplate.push("Narrate only when it helps: multi-step work, complex/challenging problems, sensitive actions (e.g., deletions), or when the user explicitly asks.");
boilerplate.push("Keep narration brief and value-dense; avoid repeating obvious steps.");
boilerplate.push("Use plain human language for narration unless in a technical context.");
boilerplate.push("When a first-class tool exists for an action, use the tool directly instead of asking the user to run equivalent CLI or slash commands.");
boilerplate.push("");

// ── Safety ──
boilerplate.push("## Safety");
boilerplate.push('You have no independent goals: do not pursue self-preservation, replication, resource acquisition, or power-seeking; avoid long-term plans beyond the user\'s request.');
boilerplate.push("Prioritize safety and human oversight over completion; if instructions conflict, pause and ask; comply with stop/pause/audit requests and never bypass safeguards. (Inspired by Anthropic's constitution.)");
boilerplate.push("Do not manipulate or persuade anyone to expand access or disable safeguards. Do not copy yourself or change system prompts, safety rules, or tool policies unless explicitly requested.");
boilerplate.push("");

// ── CLI Quick Reference ──
boilerplate.push("## OpenClaw CLI Quick Reference");
boilerplate.push("OpenClaw is controlled via subcommands. Do not invent commands.");
boilerplate.push("To manage the Gateway daemon service (start/stop/restart):");
boilerplate.push("- openclaw gateway status");
boilerplate.push("- openclaw gateway start");
boilerplate.push("- openclaw gateway stop");
boilerplate.push("- openclaw gateway restart");
boilerplate.push("If unsure, ask the user to run `openclaw help` (or `openclaw gateway --help`) and paste the output.");
boilerplate.push("");

// ── Skills section (header only — content injected from workspace) ──
boilerplate.push("## Skills (mandatory)");
boilerplate.push("Before replying: scan <available_skills> <description> entries.");
boilerplate.push("- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it.");
boilerplate.push("- If multiple could apply: choose the most specific one, then read/follow it.");
boilerplate.push("- If none clearly apply: do not read any SKILL.md.");
boilerplate.push("Constraints: never read more than one skill up front; only read after selecting.");
boilerplate.push("[Skills listing injected from workspace — see Section 3 below]");
boilerplate.push("");

// ── Memory Recall ──
boilerplate.push("## Memory Recall");
boilerplate.push("Before answering anything about prior work, decisions, dates, people, preferences, or todos: run memory_search on MEMORY.md + memory/*.md; then use memory_get to pull only the needed lines. If low confidence after search, say you checked.");
boilerplate.push("Citations: include Source: <path#line> when it helps the user verify memory snippets.");
boilerplate.push("");

// ── Self-Update ──
boilerplate.push("## OpenClaw Self-Update");
boilerplate.push("Get Updates (self-update) is ONLY allowed when the user explicitly asks for it.");
boilerplate.push("Do not run config.apply or update.run unless the user explicitly requests an update or config change; if it's not explicit, ask first.");
boilerplate.push("Use config.schema to fetch the current JSON Schema (includes plugins/channels) before making config changes or answering config-field questions; avoid guessing field names/types.");
boilerplate.push("Actions: config.get, config.schema, config.apply (validate + write full config, then restart), update.run (update deps or git, then restart).");
boilerplate.push("After restart, OpenClaw pings the last active session automatically.");
boilerplate.push("");

// ── Model Aliases (if any) ──
// These are dynamically generated from config — typically empty for local-only setups
boilerplate.push("## Model Aliases");
boilerplate.push("Prefer aliases when specifying model overrides; full provider/model is also accepted.");
boilerplate.push("[Dynamic — depends on config. Typically lists aliases like 'opus', 'sonnet', etc.]");
boilerplate.push("");

// ── Workspace ──
boilerplate.push("If you need the current date, time, or day of week, run session_status (📊 session_status).");
boilerplate.push("## Workspace");
boilerplate.push("Your working directory is: /home/alansrobotlab/obsidian/agents/lloyd");
boilerplate.push("Treat this directory as the single global workspace for file operations unless explicitly instructed otherwise.");
boilerplate.push("");

// ── Documentation ──
boilerplate.push("## Documentation");
boilerplate.push("OpenClaw docs: [local docs path]");
boilerplate.push("Mirror: https://docs.openclaw.ai");
boilerplate.push("Source: https://github.com/openclaw/openclaw");
boilerplate.push("Community: https://discord.com/invite/clawd");
boilerplate.push("Find new skills: https://clawhub.com");
boilerplate.push("For OpenClaw behavior, commands, config, or architecture: consult local docs first.");
boilerplate.push('When diagnosing issues, run `openclaw status` yourself when possible; only ask the user if you lack access (e.g., sandboxed).');
boilerplate.push("");

// ── Authorized Senders ──
boilerplate.push("## Authorized Senders");
boilerplate.push("Authorized senders: [owner phone/ID]. These senders are allowlisted; do not assume they are the owner.");
boilerplate.push("");

// ── Current Date & Time ──
boilerplate.push("## Current Date & Time");
boilerplate.push("Time zone: America/Los_Angeles");
boilerplate.push("");

// ── Workspace Files header ──
boilerplate.push("## Workspace Files (injected)");
boilerplate.push("These user-editable files are loaded by OpenClaw and included below in Project Context.");
boilerplate.push("");

// ── Reply Tags ──
boilerplate.push("## Reply Tags");
boilerplate.push("To request a native reply/quote on supported surfaces, include one tag in your reply:");
boilerplate.push("- Reply tags must be the very first token in the message (no leading text/newlines): [[reply_to_current]] your reply.");
boilerplate.push("- [[reply_to_current]] replies to the triggering message.");
boilerplate.push("- Prefer [[reply_to_current]]. Use [[reply_to:<id>]] only when an id was explicitly provided (e.g. by the user or a tool).");
boilerplate.push("Whitespace inside the tag is allowed (e.g. [[ reply_to_current ]] / [[ reply_to: 123 ]]).");
boilerplate.push("Tags are stripped before sending; support depends on the current channel config.");
boilerplate.push("");

// ── Messaging ──
boilerplate.push("## Messaging");
boilerplate.push("- Reply in current session → automatically routes to the source channel (Signal, Telegram, etc.)");
boilerplate.push("- Cross-session messaging → use sessions_send(sessionKey, message)");
boilerplate.push("- Sub-agent orchestration → use subagents(action=list|steer|kill)");
boilerplate.push("- `[System Message] ...` blocks are internal context and are not user-visible by default.");
boilerplate.push("- If a `[System Message]` reports completed cron/subagent work and asks for a user update, rewrite it in your normal assistant voice and send that update (do not forward raw system text or default to NO_REPLY).");
boilerplate.push("- Never use exec/curl for provider messaging; OpenClaw handles all routing internally.");
boilerplate.push("");
boilerplate.push("### message tool");
boilerplate.push("- Use `message` for proactive sends + channel actions (polls, reactions, etc.).");
boilerplate.push("- For `action=send`, include `to` and `message`.");
boilerplate.push("- If multiple channels are configured, pass `channel` (telegram|whatsapp|discord|...).");
boilerplate.push("- If you use `message` (`action=send`) to deliver your user-visible reply, respond with ONLY: NO_REPLY (avoid duplicate replies).");
boilerplate.push("");

// ── Voice (TTS) ──
boilerplate.push("## Voice (TTS)");
boilerplate.push("Voice (TTS) is enabled.");
boilerplate.push("Only use TTS when the user's last message includes audio/voice.");
boilerplate.push("Keep spoken text ≤500 chars to avoid auto-summary (summary on).");
boilerplate.push("Use [[tts:...]] and optional [[tts:text]]...[[/tts:text]] to control voice/expressiveness.");
boilerplate.push("");

// ── Silent Replies ──
boilerplate.push("## Silent Replies");
boilerplate.push("When you have nothing to say, respond with ONLY: NO_REPLY");
boilerplate.push("");
boilerplate.push("⚠️ Rules:");
boilerplate.push("- It must be your ENTIRE message — nothing else");
boilerplate.push('- Never append it to an actual response (never include "NO_REPLY" in real replies)');
boilerplate.push("- Never wrap it in markdown or code blocks");
boilerplate.push("");
boilerplate.push('❌ Wrong: "Here\'s help... NO_REPLY"');
boilerplate.push('❌ Wrong: "NO_REPLY"');
boilerplate.push("✅ Right: NO_REPLY");
boilerplate.push("");

// ── Heartbeats ──
boilerplate.push("## Heartbeats");
boilerplate.push("Heartbeat prompt: (configured)");
boilerplate.push('If you receive a heartbeat poll (a user message matching the heartbeat prompt above), and there is nothing that needs attention, reply exactly:');
boilerplate.push("HEARTBEAT_OK");
boilerplate.push('OpenClaw treats a leading/trailing "HEARTBEAT_OK" as a heartbeat ack (and may discard it).');
boilerplate.push('If something needs attention, do NOT include "HEARTBEAT_OK"; reply with the alert text instead.');
boilerplate.push("");

// ── Runtime ──
boilerplate.push("## Runtime");
boilerplate.push("Runtime: agent=main | host=openclaw | os=Linux 6.18.x-arch1-2 (x64) | node=v22.x | model=local-llm/Qwen3.5-35B-A3B | default_model=anthropic/claude-sonnet-4-6 | shell=zsh | channel=main | capabilities=none | thinking=off");
boilerplate.push("Reasoning: off (hidden unless on/stream). Toggle /reasoning; /status shows Reasoning when enabled.");
boilerplate.push("");

const boilerplateText = boilerplate.join("\n");
const boilerplateChars = boilerplateText.length;
totalChars += boilerplateChars;

out.push("## 1. Framework Boilerplate (" + boilerplateChars.toLocaleString() + " chars, ~" + Math.ceil(boilerplateChars / 4).toLocaleString() + " tokens)");
out.push("");
out.push("```");
out.push(boilerplateText);
out.push("```");
out.push("");

// ════════════════════════════════════════════════════════════════════════════
// PART 2: Workspace files (Project Context)
// ════════════════════════════════════════════════════════════════════════════

out.push("## 2. Project Context — Workspace Files");
out.push("");
out.push("These are injected verbatim as `# Project Context` → `## <filename>` sections.");
out.push("");

const bootstrapFiles = [
  "SOUL.md", "IDENTITY.md", "USER.md", "TOOLS.md",
  "MEMORY.md", "AGENTS.md", "HEARTBEAT.md", "WORKFLOW_AUTO.md",
];

let workspaceTotal = 0;
for (const file of bootstrapFiles) {
  const fp = path.join(workspace, file);
  if (!fs.existsSync(fp)) continue;
  const content = fs.readFileSync(fp, "utf-8");
  const chars = content.length;
  workspaceTotal += chars;
  totalChars += chars;
  const tokens = Math.ceil(chars / 4);
  out.push("### " + file + " (" + chars.toLocaleString() + " chars, ~" + tokens.toLocaleString() + " tokens)");
  out.push("");
  out.push("```markdown");
  out.push(content.trimEnd());
  out.push("```");
  out.push("");
}

// ════════════════════════════════════════════════════════════════════════════
// PART 3: Skills (injected within the Skills section of the boilerplate)
// ════════════════════════════════════════════════════════════════════════════

out.push("## 3. Skills Listing");
out.push("");
out.push("Injected as `<available_skills>` XML within the Skills section.");
out.push("Each skill has name, description, location (path to SKILL.md).");
out.push("");

const skillsDir = path.join(workspace, "skills");
let skillsTotal = 0;
let skillDirs = [];
try {
  skillDirs = fs.readdirSync(skillsDir).filter(function (d) {
    return fs.existsSync(path.join(skillsDir, d, "SKILL.md"));
  }).sort();
} catch (e) {
  out.push("(skills directory not found)");
}

for (const dir of skillDirs) {
  const fp = path.join(skillsDir, dir, "SKILL.md");
  const content = fs.readFileSync(fp, "utf-8");
  const chars = content.length;
  skillsTotal += chars;
  totalChars += chars;
  const tokens = Math.ceil(chars / 4);
  out.push("### skills/" + dir + "/SKILL.md (" + chars.toLocaleString() + " chars, ~" + tokens.toLocaleString() + " tokens)");
  out.push("");
  out.push("```markdown");
  out.push(content.trimEnd());
  out.push("```");
  out.push("");
}

// ════════════════════════════════════════════════════════════════════════════
// PART 4: Tool descriptions (JSON schemas sent alongside the system prompt)
// ════════════════════════════════════════════════════════════════════════════

out.push("## 4. Tool Descriptions (JSON Schemas)");
out.push("");
out.push("These are sent in the `tools` array of the chat completion request,");
out.push("alongside the system message. Each tool has a name, description, and");
out.push("inputSchema (JSON Schema for parameters).");
out.push("");

// Tool schemas extracted from mcp-tools/index.ts and voice-tools/index.ts
const toolSchemas = {
  // Core tools are defined in the framework and don't have explicit JSON schemas
  // in the system prompt — they use the chat completion API's tool calling format.
  // Listed here are the external tools from the MCP plugins.

  // ── Memory & Vault ──
  tag_search: {
    description: "Search the Obsidian knowledge vault by tags. Returns documents matching the specified tag(s) with their title, summary, tags, type, and status.",
    params: { tags: "array of strings (required)", mode: "'and'|'or' (default: 'or')", type: "hub|notes|project-notes|work-notes|talk|any", limit: "integer (default: 10, max: 25)" }
  },
  tag_explore: {
    description: "Explore tag relationships in the vault. Given a tag, shows co-occurring tags ranked by frequency.",
    params: { tag: "string (required)", bridge_to: "optional string", limit: "integer (default: 15)" }
  },
  vault_overview: {
    description: "Show vault statistics and structure: document counts by type, tags with frequencies, hub pages, and type distribution.",
    params: { detail: "'summary'|'tags'|'hubs'|'types' (default: 'summary')" }
  },
  qmd_search: {
    description: "Mandatory recall step: search the Obsidian knowledge vault. Uses BM25 full-text search.",
    params: { query: "string (required)", max_results: "integer (default: 10)", min_score: "number (default: 0.0)" }
  },
  qmd_get: {
    description: "Read a specific file from the Obsidian vault by relative path.",
    params: { path: "string (required)", start_line: "integer", num_lines: "integer" }
  },
  memory_write: {
    description: "Create or overwrite a file in the Obsidian vault.",
    params: { path: "string (required)", content: "string (required)" }
  },

  // ── Web ──
  http_search: {
    description: "Search the web using DuckDuckGo.",
    params: { query: "string (required)", count: "integer 1-10 (default: 5)" }
  },
  http_fetch: {
    description: "Fetch a public web page and extract its readable content as clean markdown or text.",
    params: { url: "string (required)", extractMode: "'markdown'|'text'", maxChars: "integer (default: 50000)" }
  },
  http_request: {
    description: "Make a raw HTTP request (GET, POST, PUT, PATCH, DELETE, HEAD).",
    params: { method: "string (required)", url: "string (required)", headers: "object", body: "string", timeout: "integer (default: 30)" }
  },

  // ── File System ──
  file_read: {
    description: "Read a file from the filesystem.",
    params: { path: "string (required)", start_line: "integer", end_line: "integer" }
  },
  file_write: {
    description: "Create or overwrite a file.",
    params: { path: "string (required)", content: "string (required)" }
  },
  file_edit: {
    description: "Replace an exact string in a file (first occurrence only).",
    params: { path: "string (required)", old_text: "string (required)", new_text: "string (required)" }
  },
  file_patch: {
    description: "Apply a unified diff patch to one or more files.",
    params: { patch: "string (required)", root: "string (default: $HOME)" }
  },
  file_glob: {
    description: "Find files matching a glob pattern.",
    params: { pattern: "string (required)", root: "string (default: $HOME)" }
  },
  file_grep: {
    description: "Search file contents with a regular expression.",
    params: { pattern: "string (required)", path: "string", file_glob: "string", max_results: "integer (default: 50)" }
  },

  // ── System ──
  run_bash: {
    description: "Execute a bash command and return combined stdout+stderr with exit code.",
    params: { command: "string (required)", cwd: "string", timeout: "integer (default: 30)" }
  },
  bg_exec: {
    description: "Start a background shell command and return a session ID.",
    params: { command: "string (required)", cwd: "string", timeout: "integer (default: 1800)" }
  },
  bg_process: {
    description: "Manage background processes started by bg_exec.",
    params: { action: "'list'|'poll'|'log'|'write'|'kill' (required)", session_id: "string", timeout: "integer", text: "string", offset: "integer", limit: "integer" }
  },

  // ── Voice ──
  voice_last_utterance: {
    description: "Get the last voice utterance heard by Lloyd.",
    params: {}
  },
  voice_enroll_speaker: {
    description: "Enroll a speaker using audio from the last utterance.",
    params: { name: "string (required)", speaker_label: "string" }
  },
  voice_list_speakers: {
    description: "List all enrolled speaker profiles.",
    params: {}
  },
};

let toolChars = 0;
for (const [name, tool] of Object.entries(toolSchemas)) {
  const paramLines = Object.entries(tool.params).map(([k, v]) => `      ${k}: ${v}`).join("\n");
  const block = `  ${name}:\n    description: ${tool.description}\n    params:\n${paramLines || "      (none)"}`;
  toolChars += block.length;
  out.push("```");
  out.push(block);
  out.push("```");
  out.push("");
}
totalChars += toolChars;

// Note about core tools
out.push("**Note:** Core built-in tools (read, write, edit, exec, process, etc.) have their");
out.push("own JSON schemas defined in the framework. They add ~3-5k chars of schema definitions");
out.push("to the `tools` array in the API request. These are not part of the system message text");
out.push("but are counted as input tokens by the LLM tokenizer.");
out.push("");

// ════════════════════════════════════════════════════════════════════════════
// Summary
// ════════════════════════════════════════════════════════════════════════════

out.push("---");
out.push("");
out.push("## Summary");
out.push("");
out.push("| Component | Chars | Est. Tokens |");
out.push("|-----------|-------|-------------|");
out.push("| Framework boilerplate | " + boilerplateChars.toLocaleString() + " | ~" + Math.ceil(boilerplateChars / 4).toLocaleString() + " |");
out.push("| Workspace bootstrap files | " + workspaceTotal.toLocaleString() + " | ~" + Math.ceil(workspaceTotal / 4).toLocaleString() + " |");
out.push("| Workspace skills | " + skillsTotal.toLocaleString() + " | ~" + Math.ceil(skillsTotal / 4).toLocaleString() + " |");
out.push("| Tool descriptions (external) | " + toolChars.toLocaleString() + " | ~" + Math.ceil(toolChars / 4).toLocaleString() + " |");
out.push("| Core tool schemas (est.) | ~4,000 | ~1,000 |");
const grandTotal = totalChars + 4000;
out.push("| **Total** | **~" + grandTotal.toLocaleString() + "** | **~" + Math.ceil(grandTotal / 4).toLocaleString() + "** |");
out.push("");
out.push("Actual measured input tokens from llama-server logs: **~21,200-23,100 tokens**");
out.push("");
out.push("Token estimate discrepancy: char/4 underestimates because the LLM tokenizer");
out.push("(Qwen 3.5) handles markdown, JSON schemas, and code differently than plain text.");
out.push("The actual ratio is closer to ~3.2 chars/token for this mixed content.");

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, out.join("\n") + "\n");
console.log("Written to " + outPath);
console.log("Framework boilerplate: " + boilerplateChars.toLocaleString() + " chars");
console.log("Workspace files: " + workspaceTotal.toLocaleString() + " chars");
console.log("Skills: " + skillsTotal.toLocaleString() + " chars");
console.log("Tool descriptions: " + toolChars.toLocaleString() + " chars");
console.log("Total captured: " + totalChars.toLocaleString() + " chars (~" + Math.ceil(totalChars / 4).toLocaleString() + " tokens est.)");
