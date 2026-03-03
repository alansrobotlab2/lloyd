/**
 * index.ts — mcp-tools: unified OpenClaw plugin for all MCP tools.
 *
 * Connects to the standalone MCP tool server (tool_services.py) over SSE
 * transport at http://127.0.0.1:8093. The server runs as a systemd service
 * (lloyd-tool-mcp.service).
 *
 * Tools (19): tag_search, tag_explore, vault_overview, qmd_search,
 *   qmd_get, memory_write, prefill_context, http_search, http_fetch,
 *   http_request, file_read, file_write, file_edit, file_patch, file_glob,
 *   file_grep, run_bash, bg_exec, bg_process
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { McpSseClient } from "./mcp-sse-client.js";

const TOOL_MCP_URL = "http://127.0.0.1:8093";

const PREFILL_HOOK_TIMEOUT_MS = 5_000;
const PREFILL_MAX_FAILURES = 3; // skip prefill after this many consecutive errors
const PREFILL_SKIP_COOLDOWN_MS = 120_000; // 2 min before retrying prefill
const RUN_BASH_TIMEOUT_MS = 120_000;
const WEB_TIMEOUT_MS = 20_000;
const MIN_QUERY_LENGTH = 12;

export default function register(api: OpenClawPluginApi) {
  const mcpClient = new McpSseClient(TOOL_MCP_URL);
  process.on("exit", () => mcpClient.destroy());

  // ── Helper: register a simple proxy tool ──────────────────────────────

  function proxyTool(
    name: string,
    label: string,
    description: string,
    parameters: object,
    timeoutMs?: number,
  ): void {
    api.registerTool({
      name,
      label,
      description,
      parameters,
      async execute(_id: string, params: any) {
        try {
          const content = await mcpClient.callTool(name, params, timeoutMs);
          const text =
            content.length > 0
              ? content.map((c: any) => c.text).join("")
              : "(no result)";
          return { content: [{ type: "text" as const, text }] };
        } catch (err: any) {
          return {
            content: [{ type: "text" as const, text: `${name} error: ${err.message}` }],
          };
        }
      },
    });
  }

  // ── Prefill hook (before_prompt_build) ────────────────────────────────
  //
  // Two-phase per-session prefill strategy:
  //   Turn 1: inject yesterday's + today's daily memory files (temporal grounding)
  //   Turn 2: inject semantic vault recall using the turn-1 query as search input
  //   Turn 3+: no prefill — history carries sufficient accumulated context
  //
  // Rationale: per-turn semantic prefill adds ~549 chars / ~285ms per turn.
  // By turn 5+ the conversation history already embeds the relevant vault docs
  // from earlier turns, so additional prefill produces diminishing returns.

  let prefillConsecutiveFailures = 0;
  let prefillSkipUntil = 0;

  // Per-session prefill state (in-process, cleared on session_end)
  const dailyNotesInjected = new Set<string>(); // sessions where turn-1 daily notes ran
  const semanticPrefillDone = new Set<string>(); // sessions where turn-2 semantic prefill ran
  const firstUserPrompt = new Map<string, string>(); // turn-1 prompt → used as turn-2 search query

  // Agents that don't need vault recall in their prefill
  const SKIP_PREFILL_AGENTS = new Set([
    "coder", "tester", "reviewer", "auditor", "operator", "orchestrator",
  ]);

  // ── Context profile classification (lightweight, regex-only) ──────────
  type ContextProfile = "chat" | "memory" | "code" | "research" | "ops" | "voice" | "heartbeat" | "default";

  const CHAT_RE = /^(hey|hi|hello|yo|sup|thanks|thank you|ok|sure|yes|no|yep|nah|nope|got it|cool|nice|great|perfect|sounds good|go ahead|do it|lol|haha|hmm|good morning|good night|gm|gn|👍|❤️|😂|💀)[\.\!\?]?$/i;
  const MEMORY_RE = /\b(remember|what did (?:we|i)|recall|last (?:time|session|week)|diary|journal|daily note|MEMORY\.md|vault (?:search|notes?))\b/i;
  const CODE_RE = /\b(implement|debug|refactor|fix (?:the |this )?(?:bug|error|code)|write (?:a |the )?(?:function|class|method|test|script)|add (?:a |the )?(?:feature|endpoint|method)|create (?:a |the )?(?:file|component|module))\b/i;
  const CODE_BLOCK_RE = /```/;
  const RESEARCH_RE = /\b(search for|look up|what is|what are|who is|find (?:out|info)|latest on|news about|how does .{3,} work)\b/i;
  const OPS_RE = /\b(restart|deploy|service|systemctl|docker|git (?:push|pull|merge|rebase)|clawdeck|task (?:board|backlog)|CI\/CD|build|release)\b/i;
  const VOICE_RE = /\b(say |speak |voice |read (?:this |it )?(?:aloud|out loud)|tts|text.to.speech|narrate)\b/i;
  // Cron-injected automation prompts and session-management instructions — vault
  // recall produces false positives from incidental keyword matches in boilerplate.
  const HEARTBEAT_RE = /\bHEARTBEAT(?:_OK|\.md)?\b|\bPost-Compaction Audit\b|\bWorkflow Recovery\b|\bExecute your Session Startup\b|\bnew session was started via\b/i;

  // Segment scope routing for semantic prefill.
  // Empty string = no scope restriction (search all segments).
  const PROFILE_SCOPE: Record<ContextProfile, string> = {
    memory:    "",
    research:  "knowledge,projects,work",
    default:   "",
    chat:      "",
    code:      "",
    ops:       "",
    voice:     "",
    heartbeat: "",
  };

  function classifyProfile(prompt: string): ContextProfile {
    const trimmed = prompt.trim();
    const lower = trimmed.toLowerCase();

    if (HEARTBEAT_RE.test(trimmed)) return "heartbeat";
    if (trimmed.length < 50 && CHAT_RE.test(trimmed)) return "chat";
    if (VOICE_RE.test(lower)) return "voice";
    if (CODE_BLOCK_RE.test(prompt) || CODE_RE.test(lower)) return "code";
    if (MEMORY_RE.test(lower)) return "memory";
    if (OPS_RE.test(lower)) return "ops";
    if (RESEARCH_RE.test(lower)) return "research";

    return "default";
  }

  // Fetch yesterday's and today's daily memory files from the vault.
  async function fetchDailyNotes(): Promise<string> {
    const now = new Date();
    const toPstDateStr = (d: Date): string => {
      const pst = new Date(d.toLocaleString("en-US", { timeZone: "America/Los_Angeles" }));
      const y = pst.getFullYear();
      const mo = String(pst.getMonth() + 1).padStart(2, "0");
      const dy = String(pst.getDate()).padStart(2, "0");
      return `${y}-${mo}-${dy}`;
    };

    const entries = [
      { date: toPstDateStr(now), label: "today" },
      { date: toPstDateStr(new Date(now.getTime() - 86_400_000)), label: "yesterday" },
    ];

    const results = await Promise.allSettled(
      entries.map(({ date }) =>
        mcpClient.callTool("qmd_get", { path: `agents/lloyd/memory/${date}.md` }, 3_000),
      ),
    );

    const sections: string[] = [];
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      const { date, label } = entries[i];
      if (r.status !== "fulfilled") continue;
      const raw = r.value
        .filter((b: any) => b.type === "text")
        .map((b: any) => b.text)
        .join("")
        .trim();
      // qmd_get returns JSON: {path, text} — unwrap if needed
      let text = raw;
      try { text = JSON.parse(raw)?.text ?? raw; } catch { /* use raw */ }
      if (text && !text.startsWith("File not found")) {
        sections.push(`--- ${date} (${label}) ---\n${text.trim()}`);
      }
    }

    if (sections.length === 0) return "";
    return `<daily_notes>\n${sections.join("\n\n")}\n</daily_notes>`;
  }

  api.on("before_prompt_build", async (event: any, ctx: any) => {
    const prompt: string = event.prompt ?? "";
    if (!prompt || prompt.length < MIN_QUERY_LENGTH) return;

    const agentId = ctx?.agentId;
    if (agentId && SKIP_PREFILL_AGENTS.has(agentId)) return;

    const profile = classifyProfile(prompt);
    if (profile === "heartbeat") return; // automated prompts never get prefill

    const sessionId = ctx?.sessionId ?? "";

    // ── Turn 1: inject daily memory files ─────────────────────────────────
    if (!dailyNotesInjected.has(sessionId)) {
      dailyNotesInjected.add(sessionId);
      // Save the first substantive prompt for semantic search on turn 2
      if (profile !== "chat") firstUserPrompt.set(sessionId, prompt);

      const dailyContext = await fetchDailyNotes();
      if (dailyContext) return { prependContext: dailyContext };
      return;
    }

    // ── Turn 2: semantic prefill using the turn-1 query ───────────────────
    if (!semanticPrefillDone.has(sessionId)) {
      semanticPrefillDone.add(sessionId);

      const searchQuery = firstUserPrompt.get(sessionId);
      if (!searchQuery) return; // turn-1 was chat/too-short — skip semantic prefill

      const now = Date.now();
      if (prefillSkipUntil > now) return;
      if (prefillSkipUntil > 0 && prefillSkipUntil <= now) {
        prefillSkipUntil = 0;
        prefillConsecutiveFailures = 0;
      }

      const prefillScope = PROFILE_SCOPE[classifyProfile(searchQuery)] ?? "";

      try {
        const content = await mcpClient.callTool(
          "prefill_context",
          { prompt: searchQuery, session_id: sessionId, scope: prefillScope },
          PREFILL_HOOK_TIMEOUT_MS,
        );

        prefillConsecutiveFailures = 0;

        const text = content
          .filter((b: any) => b.type === "text")
          .map((b: any) => b.text)
          .join("")
          .trim();
        if (text) return { prependContext: text };
      } catch (err: any) {
        prefillConsecutiveFailures++;
        if (prefillConsecutiveFailures >= PREFILL_MAX_FAILURES) {
          prefillSkipUntil = Date.now() + PREFILL_SKIP_COOLDOWN_MS;
          api.logger.warn?.(
            `mcp-tools prefill: ${prefillConsecutiveFailures} consecutive failures, skipping for ${PREFILL_SKIP_COOLDOWN_MS / 1000}s`,
          );
        } else {
          api.logger.warn?.(`mcp-tools prefill: ${err?.message}`);
        }
      }
      return;
    }

    // Turn 3+: no prefill — conversation history carries sufficient context
  });

  // ── Daily memory file (agent_end hook) ─────────────────────────────────

  const DAILY_MEMORY_PATH_PREFIX = "agents/lloyd/memory";

  function todayDateStr(): string {
    const now = new Date();
    const pst = new Date(
      now.toLocaleString("en-US", { timeZone: "America/Los_Angeles" }),
    );
    const y = pst.getFullYear();
    const m = String(pst.getMonth() + 1).padStart(2, "0");
    const d = String(pst.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  }

  function pstTimeStr(): string {
    return new Date().toLocaleString("en-US", {
      timeZone: "America/Los_Angeles",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  }

  api.on("session_end", async (event: any, ctx: any) => {
    // Clean up per-session prefill state
    const sid = event?.sessionId ?? ctx?.sessionId ?? "";
    if (sid) {
      dailyNotesInjected.delete(sid);
      semanticPrefillDone.delete(sid);
      firstUserPrompt.delete(sid);
    }

    // Only create daily files for the main agent
    if (ctx?.agentId && ctx.agentId !== "main") return;

    const dateStr = todayDateStr();
    const filePath = `${DAILY_MEMORY_PATH_PREFIX}/${dateStr}.md`;
    const time = pstTimeStr();
    const msgCount = event?.messageCount ?? 0;
    const durationMin = event?.durationMs
      ? Math.round(event.durationMs / 60_000)
      : null;

    const entry = [
      `\n### Session ${time} PST`,
      `- Messages: ${msgCount}${durationMin != null ? `, Duration: ~${durationMin}min` : ""}`,
      `- Session: ${event?.sessionId ?? ctx?.sessionId ?? "unknown"}`,
    ].join("\n");

    try {
      // Try to read existing file
      let existing = "";
      try {
        const content = await mcpClient.callTool(
          "qmd_get",
          { path: filePath },
          3000,
        );
        const raw = content.map((b: any) => b.text).join("");
        try {
          existing = JSON.parse(raw)?.text ?? "";
        } catch {
          existing = raw;
        }
      } catch {
        // File doesn't exist yet — that's fine
      }

      let fileContent: string;
      if (existing && existing.trim().length > 0) {
        fileContent = existing.trimEnd() + "\n" + entry + "\n";
      } else {
        fileContent = [
          "---",
          "segment: agents",
          "---",
          "",
          `# ${dateStr} Daily Notes`,
          "",
          "## Sessions",
          entry,
          "",
        ].join("\n");
      }

      await mcpClient.callTool(
        "memory_write",
        { path: filePath, content: fileContent },
        5000,
      );
      api.logger.info?.(`mcp-tools daily-memory: wrote ${filePath}`);
    } catch (err: any) {
      api.logger.warn?.(`mcp-tools daily-memory: ${err?.message}`);
    }
  });

  // ── Memory & vault tools ──────────────────────────────────────────────

  proxyTool(
    "tag_search",
    "Tag Search",
    "Search the Obsidian knowledge vault by tags. Returns documents matching the specified tag(s) " +
      "with their title, summary, tags, type, and status. Use AND mode to find documents at the " +
      "intersection of multiple topics, OR mode for broader searches. " +
      "Examples: tag_search([\"alfie\"]), tag_search([\"ai\", \"rag\"], \"and\"), tag_search([\"robotics\"], type=\"hub\").",
    {
      type: "object",
      properties: {
        tags: {
          type: "array",
          items: { type: "string" },
          description: "One or more tags to search for (without # prefix)",
        },
        mode: {
          type: "string",
          enum: ["and", "or"],
          description: "Match mode: 'and' = docs must have ALL tags, 'or' = docs with ANY tag (default: 'or')",
        },
        type: {
          type: "string",
          description: "Filter by document type: hub, notes, project-notes, work-notes, talk, or 'any' (default: 'any')",
        },
        limit: {
          type: "integer",
          description: "Max results to return (default: 10, max: 25)",
        },
      },
      required: ["tags"],
    },
  );

  proxyTool(
    "tag_explore",
    "Tag Explore",
    "Explore tag relationships in the vault. Given a tag, shows co-occurring tags ranked by " +
      "frequency. Optionally provide bridge_to to find documents that have BOTH tags. " +
      "Use this to discover connections between topics and navigate the vault's knowledge structure.",
    {
      type: "object",
      properties: {
        tag: {
          type: "string",
          description: "A tag to explore relationships for (without # prefix)",
        },
        bridge_to: {
          type: "string",
          description: "Optional second tag — shows documents that have BOTH tags (bridging documents)",
        },
        limit: {
          type: "integer",
          description: "Max related tags to show (default: 15)",
        },
      },
      required: ["tag"],
    },
  );

  proxyTool(
    "vault_overview",
    "Vault Overview",
    "Show vault statistics and structure: document counts by type, tags with frequencies, " +
      "hub pages (index pages), and type distribution. Use this to understand what's in " +
      "the vault before searching.",
    {
      type: "object",
      properties: {
        detail: {
          type: "string",
          enum: ["summary", "tags", "hubs", "types"],
          description: "What to show: 'summary' = overview, 'tags' = all tags with frequencies, 'hubs' = hub pages, 'types' = type breakdown (default: 'summary')",
        },
      },
    },
  );

  proxyTool(
    "qmd_search",
    "QMD Search",
    "Mandatory recall step: search the Obsidian knowledge vault before answering questions " +
      "about prior work, decisions, dates, people, preferences, or todos. Uses BM25 full-text " +
      "search across the indexed vault. Returns JSON with matching document paths, relevance " +
      "scores, line ranges, and snippets.",
    {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        max_results: { type: "integer", description: "Max results to return (default: 10)" },
        min_score: { type: "number", description: "Minimum relevance score threshold (default: 0.0)" },
      },
      required: ["query"],
    },
  );

  proxyTool(
    "qmd_get",
    "QMD Get",
    "Read a specific file from the Obsidian vault by relative path. " +
      "Use after qmd_search to pull only the needed lines and keep context small. " +
      "path: relative from vault root, e.g. 'projects/alfie/alfie.md'.",
    {
      type: "object",
      properties: {
        path: { type: "string", description: "Vault-relative file path" },
        start_line: { type: "integer", description: "First line to return (1-indexed, 0 = beginning)" },
        num_lines: { type: "integer", description: "Number of lines to read (0 = all remaining)" },
      },
      required: ["path"],
    },
  );

  proxyTool(
    "memory_write",
    "Memory Write",
    "Create or overwrite a file in the Obsidian vault. " +
      "path: vault-relative path, e.g. 'projects/alfie/notes.md'. Creates parent directories automatically.",
    {
      type: "object",
      properties: {
        path: { type: "string", description: "Vault-relative file path, e.g. 'projects/alfie/notes.md'" },
        content: { type: "string", description: "Text content to write" },
      },
      required: ["path", "content"],
    },
  );

  // ── Web tools ─────────────────────────────────────────────────────────

  proxyTool(
    "http_search",
    "HTTP Search",
    "Search the web using DuckDuckGo. Returns a list of results with title, URL, and snippet.",
    {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        count: {
          type: "integer",
          description: "Number of results to return (1-10, default 5)",
          minimum: 1,
          maximum: 10,
        },
      },
      required: ["query"],
    },
    WEB_TIMEOUT_MS,
  );

  api.registerTool({
    name: "http_fetch",
    label: "Fetch Web Page",
    description:
      "Fetch a public web page and extract its readable content as clean markdown or text. " +
      "Uses readability to strip boilerplate and returns the article/main body — ideal for reading " +
      "documentation, blog posts, and news articles. GET only; public URLs only (no 127.0.0.1). " +
      "For REST APIs, POST requests, auth headers, or local services, use http_request instead.",
    parameters: {
      type: "object" as const,
      properties: {
        url: {
          type: "string" as const,
          description: "The URL to fetch (http or https)",
        },
        extractMode: {
          type: "string" as const,
          enum: ["markdown", "text"],
          description: 'Extraction mode: "markdown" or "text" (default "markdown")',
        },
        maxChars: {
          type: "integer" as const,
          description: "Maximum characters to return (default 50000)",
          minimum: 1000,
          maximum: 200000,
        },
      },
      required: ["url"] as string[],
    },
    async execute(
      _id: string,
      params: { url: string; extractMode?: "markdown" | "text"; maxChars?: number },
    ) {
      // Translate camelCase LLM params → snake_case Python params
      const mcpArgs: Record<string, unknown> = { url: params.url };
      if (params.extractMode !== undefined) mcpArgs.extract_mode = params.extractMode;
      if (params.maxChars !== undefined) mcpArgs.max_chars = params.maxChars;

      try {
        const content = await mcpClient.callTool("http_fetch", mcpArgs, WEB_TIMEOUT_MS);
        const text = content.map((c: any) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `http_fetch error: ${err.message}` }] };
      }
    },
  });

  api.registerTool({
    name: "http_request",
    label: "HTTP Request",
    description:
      "Make a raw HTTP request (GET, POST, PUT, PATCH, DELETE, HEAD) and return the status code " +
      "and response body unprocessed. Use this for REST APIs, local services, and any endpoint " +
      "that needs custom headers, a request body, or non-GET methods. " +
      "127.0.0.1 (loopback) is allowed for local container services. " +
      "Other private/internal IPs are blocked. " +
      "For reading public web pages as readable text, use http_fetch instead.",
    parameters: {
      type: "object" as const,
      properties: {
        method: {
          type: "string" as const,
          enum: ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
          description: "HTTP method",
        },
        url: {
          type: "string" as const,
          description: "Full URL (http or https)",
        },
        headers: {
          type: "object" as const,
          description: "Optional request headers",
          additionalProperties: { type: "string" as const },
        },
        body: {
          type: "string" as const,
          description: "Optional request body string",
        },
        timeout: {
          type: "integer" as const,
          description: "Max seconds to wait (default 30, max 120)",
          minimum: 1,
          maximum: 120,
        },
      },
      required: ["method", "url"] as string[],
    },
    async execute(
      _id: string,
      params: { method: string; url: string; headers?: Record<string, string>; body?: string; timeout?: number },
    ) {
      try {
        const content = await mcpClient.callTool("http_request", params as Record<string, unknown>, WEB_TIMEOUT_MS);
        const text = content.map((c: any) => c.text).join("") || "(no result)";
        return { content: [{ type: "text" as const, text }] };
      } catch (err: any) {
        return { content: [{ type: "text" as const, text: `http_request error: ${err.message}` }] };
      }
    },
  });

  // ── File system tools ─────────────────────────────────────────────────

  proxyTool(
    "file_read",
    "Read File",
    "Read a file from the filesystem. path must be absolute or ~/relative (sandboxed to $HOME). " +
      "Supports optional line range via start_line / end_line (1-indexed).",
    {
      type: "object",
      properties: {
        path: { type: "string", description: "Absolute or ~/relative file path" },
        start_line: { type: "integer", description: "First line to return (1-indexed, 0 = beginning)" },
        end_line: { type: "integer", description: "Last line to return (0 = end of file)" },
      },
      required: ["path"],
    },
  );

  proxyTool(
    "file_write",
    "Write File",
    "Create or overwrite a file. path must be absolute or ~/relative (sandboxed to $HOME). " +
      "Creates parent directories automatically.",
    {
      type: "object",
      properties: {
        path: { type: "string", description: "Absolute or ~/relative file path" },
        content: { type: "string", description: "Text content to write" },
      },
      required: ["path", "content"],
    },
  );

  proxyTool(
    "file_edit",
    "Edit File",
    "Replace an exact string in a file (first occurrence only). " +
      "old_text must appear exactly once — provide more surrounding context if it appears multiple times. " +
      "path must be absolute or ~/relative (sandboxed to $HOME).",
    {
      type: "object",
      properties: {
        path: { type: "string", description: "Absolute or ~/relative file path" },
        old_text: { type: "string", description: "Exact text to find (must appear exactly once)" },
        new_text: { type: "string", description: "Replacement text" },
      },
      required: ["path", "old_text", "new_text"],
    },
  );

  proxyTool(
    "file_patch",
    "Apply Patch",
    "Apply a unified diff patch (like `diff -u` or `git diff` output) to one or more files. " +
      "Supports creating new files (--- /dev/null), deleting files (+++ /dev/null), and " +
      "modifying files with multi-hunk context-based matching. All paths must be within $HOME. " +
      "Returns a summary of operations (A = added, M = modified, D = deleted).",
    {
      type: "object",
      properties: {
        patch: { type: "string", description: "Unified diff text" },
        root: {
          type: "string",
          description: "Base directory for relative paths in the diff (default: $HOME)",
        },
      },
      required: ["patch"],
    },
  );

  proxyTool(
    "file_glob",
    "Glob Files",
    "Find files matching a glob pattern. Returns up to 200 matching paths relative to root. " +
      'Examples: file_glob("**/*.py"), file_glob("*.md", "~/obsidian")',
    {
      type: "object",
      properties: {
        pattern: { type: "string", description: 'Glob pattern, e.g. "**/*.py", "*.md"' },
        root: { type: "string", description: "Directory to search from (default: $HOME); must be within $HOME" },
      },
      required: ["pattern"],
    },
  );

  proxyTool(
    "file_grep",
    "Grep Files",
    "Search file contents with a regular expression. Returns matching lines with filename and line number. " +
      "path can be a directory (searched recursively) or a single file.",
    {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Python regex pattern to search for" },
        path: { type: "string", description: "File or directory to search (default: $HOME); must be within $HOME" },
        file_glob: { type: "string", description: 'Glob pattern to filter files (default "**/*")' },
        max_results: { type: "integer", description: "Maximum matching lines to return (default 50, max 200)" },
      },
      required: ["pattern"],
    },
  );

  // ── System tools ──────────────────────────────────────────────────────

  proxyTool(
    "run_bash",
    "Run Shell Command",
    "Execute a bash command and return combined stdout+stderr with exit code. " +
      "command is passed to bash -c. cwd must be within $HOME (default: $HOME). " +
      "timeout is max seconds to wait (default 30, max 120).",
    {
      type: "object",
      properties: {
        command: { type: "string", description: "Bash command string (passed to bash -c)" },
        cwd: { type: "string", description: "Working directory (absolute or ~/relative, must be within $HOME; default: $HOME)" },
        timeout: { type: "integer", description: "Max seconds to wait (default 30, max 120)" },
      },
      required: ["command"],
    },
    RUN_BASH_TIMEOUT_MS,
  );

  proxyTool(
    "bg_exec",
    "Background Execute",
    "Start a background shell command and return a session ID. " +
      "Use bg_process to poll output, read logs, write to stdin, or kill. " +
      "Ideal for long-running builds, servers, and watch processes.",
    {
      type: "object",
      properties: {
        command: { type: "string", description: "Bash command string (passed to bash -c)" },
        cwd: { type: "string", description: "Working directory (default: $HOME; must be within $HOME)" },
        timeout: { type: "integer", description: "Auto-kill timeout in seconds (default 1800, max 7200)" },
      },
      required: ["command"],
    },
    RUN_BASH_TIMEOUT_MS,
  );

  proxyTool(
    "bg_process",
    "Manage Background Process",
    "Manage background processes started by bg_exec. " +
      "Actions: list (show all sessions), poll (wait for new output), " +
      "log (get output buffer), write (send to stdin), kill (terminate).",
    {
      type: "object",
      properties: {
        action: {
          type: "string",
          enum: ["list", "poll", "log", "write", "kill"],
          description: "Action to perform",
        },
        session_id: { type: "string", description: "Session ID (required for poll/log/write/kill)" },
        timeout: { type: "integer", description: "Poll timeout in seconds (default 10, max 60)" },
        text: { type: "string", description: "Text to write to stdin (for write action)" },
        offset: { type: "integer", description: "Line offset for log action (0 = last N lines)" },
        limit: { type: "integer", description: "Max lines to return for log (default 100, max 500)" },
      },
      required: ["action"],
    },
    70_000, // poll can block up to 60s
  );

  api.logger.info?.("mcp-tools: registered 19 tools + prefill hook via single MCP server");
}
