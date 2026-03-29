/**
 * agents.ts — Agent config, tools/skills per agent, avatars, call log
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, writeFileSync, readdirSync, existsSync, statSync } from "fs";
import { readFile, writeFile, readdir, access, stat } from "fs/promises";
import { join, extname } from "path";
import { homedir } from "os";
import type { PluginContext, SkillInfo, SessionMessage, AgentCallLogEntry } from "./types.js";
import { jsonResponse, readBody, readFileOpt, parseJsonl } from "./helpers.js";
import { parseSkillDir } from "./skills.js";

// ── Tool groups & state ─────────────────────────────────────────────

const TOOL_GROUPS = [
  { source: "openclaw — sessions & agents", tools: ["sessions_list", "sessions_history", "sessions_send", "sessions_spawn", "subagents", "session_status", "agents_list", "message"] },
  { source: "openclaw — files & runtime", tools: ["read", "write", "edit", "apply_patch", "exec", "process"] },
  { source: "openclaw — web & memory", tools: ["web_search", "web_fetch", "memory_search", "memory_get"] },
  { source: "openclaw — system & media", tools: ["cron", "gateway", "nodes", "browser", "canvas", "image", "tts"] },
  { source: "mcp-tools", tools: ["mem_search", "mem_get", "mem_write", "tag_explore", "vault_overview", "prefill_context", "http_search", "http_fetch", "http_request", "file_read", "file_write", "file_edit", "file_patch", "file_glob", "file_grep", "run_bash", "bg_exec", "bg_process", "skills_search", "skills_get"] },
  { source: "mcp-tools — backlog", tools: ["backlog_boards", "backlog_tasks", "backlog_get_task", "backlog_write_task"] },
  { source: "mcp-tools — memory", tools: ["context_bundle", "get_profile", "get_facts", "add_fact", "get_relations", "add_relation", "detect_contradictions", "resolve_contradictions", "rebuild_index"] },
  { source: "mcp-tools — autonomy", tools: ["autonomy_tasks", "autonomy_get_task", "autonomy_write_task", "autonomy_run_task"] },
  { source: "voice-tools", tools: ["voice_last_utterance", "voice_enroll_speaker", "voice_list_speakers"] },
  { source: "thunderbird-tools", tools: ["email_accounts", "email_folders", "email_search", "email_read", "email_recent", "calendar_list", "calendar_events", "contacts_search", "contacts_get"] },
];

export { TOOL_GROUPS };

function loadToolsState(toolsFile: string): Record<string, boolean> {
  try {
    if (existsSync(toolsFile)) return JSON.parse(readFileSync(toolsFile, "utf-8"));
  } catch { /* non-fatal */ }
  return {};
}

function saveToolsState(toolsFile: string, state: Record<string, boolean>) {
  writeFileSync(toolsFile, JSON.stringify(state, null, 2) + "\n");
}

// ── Avatar discovery ────────────────────────────────────────────────

const AVATAR_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];
const AVATAR_MIME: Record<string, string> = {
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".png": "image/png", ".webp": "image/webp",
};
const AVATARS_DIR = join(homedir(), "obsidian", "agents", "lloyd", "avatars");

function discoverAvatarFlat(agentId: string): string | null {
  if (!agentId || /[\/\.]/.test(agentId)) return null;
  for (const ext of AVATAR_EXTENSIONS) {
    const filePath = join(AVATARS_DIR, agentId + ext);
    if (existsSync(filePath)) return filePath;
  }
  return null;
}

function discoverAvatarInDir(agentId: string): string | null {
  if (!agentId || /[\/\.]/.test(agentId)) return null;
  const avatarDir = join(homedir(), "obsidian", "agents", agentId, "avatar");
  if (!existsSync(avatarDir)) return null;
  let entries: string[];
  try { entries = readdirSync(avatarDir); } catch { return null; }
  for (const ext of AVATAR_EXTENSIONS) {
    const match = entries.find((e) => e.toLowerCase().endsWith(ext));
    if (match) return join(avatarDir, match);
  }
  return null;
}

function discoverAvatar(agentId: string, agentName?: string): string | null {
  return discoverAvatarFlat(agentId)
    || discoverAvatarInDir(agentId)
    || (agentName ? discoverAvatarFlat(agentName.toLowerCase()) : null)
    || (agentName ? discoverAvatarInDir(agentName.toLowerCase()) : null);
}

// ── Route registration ──────────────────────────────────────────────

export function registerAgentRoutes(
  ctx: PluginContext,
  skillDirs: { workspaceSkillsDirs: string[]; bundledSkillsDir: string },
) {
  const { api, rootDir, configFile, toolsFile } = ctx;
  const { workspaceSkillsDirs, bundledSkillsDir } = skillDirs;

  // Resolve workspace dir
  const workspaceDir = (() => {
    try {
      const cfg = JSON.parse(readFileSync(configFile, "utf-8"));
      const mainAgent = cfg.agents?.list?.find((a: any) => a.id === "main") || cfg.agents?.list?.[0];
      return mainAgent?.workspace?.replace(/^~/, homedir()) ?? join(rootDir, "workspaces/lloyd");
    } catch { return join(rootDir, "workspaces/lloyd"); }
  })();

  function countSessions(agentId: string): { total: number; active: number } {
    const dir = join(rootDir, `agents/${agentId}/sessions`);
    if (!existsSync(dir)) return { total: 0, active: 0 };
    const files = readdirSync(dir);
    let active = files.filter((f) => f.endsWith(".jsonl") && !f.includes(".reset.")).length;
    const total = files.filter((f) => f.includes(".jsonl")).length;
    let liveActive = 0;
    for (const [key, status] of ctx.agentSessionStates) {
      if (key.startsWith(`agent:${agentId}:`) && status.state !== "idle") liveActive++;
    }
    active = Math.max(active, liveActive);
    return { total, active };
  }

  // ── Tool hooks ────────────────────────────────────────────────────

  api.on("before_tool_call", async (event: any, hookCtx: any) => {
    if (ctx.activityResetTimer.value) { clearTimeout(ctx.activityResetTimer.value); ctx.activityResetTimer.value = null; }
    ctx.currentActivity.value = {
      type: "tool_call", toolName: event.toolName,
      startedAt: Date.now(), sessionKey: hookCtx?.sessionKey,
    };
    const state = loadToolsState(toolsFile);
    if (state[event.toolName] === false) {
      return { block: true, blockReason: `Tool "${event.toolName}" is disabled via Mission Control` };
    }
  });

  api.on("after_tool_call", async () => {
    ctx.activityResetTimer.value = setTimeout(() => {
      ctx.currentActivity.value = { type: "llm_thinking", startedAt: Date.now() };
      ctx.activityResetTimer.value = null;
    }, 600);
  });

  api.on("llm_input", async (event: any, hookCtx: any) => {
    ctx.currentActivity.value = {
      type: "llm_thinking", model: event?.model,
      startedAt: Date.now(), sessionKey: hookCtx?.sessionKey,
    };
  });

  api.on("agent_end", async () => {
    ctx.currentActivity.value = { type: "idle", startedAt: Date.now() };
  });

  // ── Agents endpoint cache ────────────────────────────────────────
  let agentsCache: { data: any; ts: number; configMtime: number } | null = null;
  const AGENTS_CACHE_TTL = 30_000; // 30 seconds

  // ── GET /api/mc/tools ─────────────────────────────────────────────

  api.registerHttpRoute({
    path: "/api/mc/tools",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const state = loadToolsState(toolsFile);
        const groups = TOOL_GROUPS.map((g) => ({
          source: g.source,
          tools: g.tools.map((name) => ({ name, enabled: state[name] !== false })),
        }));
        jsonResponse(res, { groups });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/tool-toggle
  api.registerHttpRoute({
    path: "/api/mc/tool-toggle",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { toolName, enabled } = body;
        if (!toolName || typeof enabled !== "boolean") { jsonResponse(res, { error: "Missing toolName or enabled (boolean)" }, 400); return; }
        const state = loadToolsState(toolsFile);
        if (enabled) delete state[toolName];
        else state[toolName] = false;
        saveToolsState(toolsFile, state);
        jsonResponse(res, { ok: true, toolName, enabled });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/agents
  api.registerHttpRoute({
    path: "/api/mc/agents",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        // TTL cache — invalidate on config file mtime change or after 30s
        const now = Date.now();
        let configMtime = 0;
        try { configMtime = (await stat(configFile)).mtimeMs; } catch { /* non-fatal */ }
        if (agentsCache && (now - agentsCache.ts < AGENTS_CACHE_TTL) && agentsCache.configMtime === configMtime) {
          jsonResponse(res, agentsCache.data);
          return;
        }

        const config = JSON.parse(await readFile(configFile, "utf-8"));
        const agentList: any[] = config.agents?.list || [];
        const defaults = config.agents?.defaults || {};

        const agents = await Promise.all(agentList.map(async (a: any) => {
          const id = a.id;
          const sessions = countSessions(id);
          const agentDir = join(rootDir, `agents/${id}/agent`);

          let primaryModel = defaults.model?.primary ?? null;
          if (a.model) primaryModel = typeof a.model === "string" ? a.model : (a.model.primary ?? primaryModel);

          const modelsPath = join(agentDir, "models.json");
          let modelCount = 0, enabledModels = 0;
          try {
            const m = JSON.parse(await readFile(modelsPath, "utf-8"));
            for (const p of Object.values<any>(m.providers || {})) {
              for (const model of (p.models || [])) { modelCount++; if (model.enabled !== false) enabledModels++; }
            }
          } catch { /* non-fatal — file missing or parse error */ }

          const toolsPath = join(agentDir, "tools.json");
          let disabledTools = 0;
          try {
            const t = JSON.parse(await readFile(toolsPath, "utf-8"));
            disabledTools = Object.values(t).filter((v) => v === false).length;
          } catch { /* non-fatal */ }

          const agentWorkspaceDir = a.workspace
            ? a.workspace.replace(/^~/, homedir())
            : (a.default || id === agentList[0]?.id)
              ? join(rootDir, "workspaces/lloyd")
              : join(rootDir, `workspaces/${id}`);

          const agentWorkspace: Record<string, string | null> = {};
          const workspaceFiles: { name: string; key: string; content: string | null }[] = [];
          try {
            const wsEntries = await readdir(agentWorkspaceDir);
            for (const entry of wsEntries) {
              if (!entry.endsWith(".md") || entry.startsWith(".")) continue;
              const fullPath = join(agentWorkspaceDir, entry);
              try {
                const entrySt = await stat(fullPath);
                if (!entrySt.isFile()) continue;
              } catch { continue; }
              const key = entry.replace(/\.md$/, "").toLowerCase();
              let content: string | null = null;
              try { content = await readFile(fullPath, "utf-8"); } catch { /* ok */ }
              agentWorkspace[key] = content;
              workspaceFiles.push({ name: entry, key, content });
            }
            workspaceFiles.sort((a, b) => a.name.localeCompare(b.name));
          } catch { /* agentWorkspaceDir doesn't exist */ }

          const identityRaw = agentWorkspace.identity;
          let identity: string | null = null;
          if (identityRaw) {
            let lines = identityRaw.split("\n");
            if (lines[0]?.trim() === "---") {
              const end = lines.findIndex((l: string, i: number) => i > 0 && l.trim() === "---");
              if (end !== -1) lines = lines.slice(end + 1);
            }
            const oneLiner = lines.find((l: string) => l.trim() && !l.startsWith("#"));
            if (oneLiner) identity = oneLiner.trim();
          }

          return {
            id, name: a.name ?? id,
            avatar: discoverAvatar(id, a.name) ? `/api/mc/agent-avatar?id=${id}&name=${encodeURIComponent(a.name ?? id)}` : null,
            identity, primaryModel,
            modelFallbacks: (typeof a.model === "object" && a.model?.fallbacks) || null,
            sessions, modelCount, enabledModels, disabledTools,
            toolsAllow: a.tools?.allow ?? null,
            skills: a.skills ?? null,
            maxConcurrent: defaults.maxConcurrent ?? null,
            subagentMaxConcurrent: defaults.subagents?.maxConcurrent ?? null,
            workspace: agentWorkspace, workspaceFiles,
            workspacePath: agentWorkspaceDir,
          };
        }));

        const workspace: Record<string, string | null> = {
          soul: readFileOpt(join(workspaceDir, "SOUL.md")),
          identity: readFileOpt(join(workspaceDir, "IDENTITY.md")),
          agents: readFileOpt(join(workspaceDir, "AGENTS.md")),
          memory: readFileOpt(join(workspaceDir, "MEMORY.md")),
        };

        const allToolGroups = TOOL_GROUPS.map((g) => ({ source: g.source, tools: g.tools }));
        const wsSkills = workspaceSkillsDirs.flatMap(d => parseSkillDir(d, configFile, api.logger));
        const bdSkills = parseSkillDir(bundledSkillsDir, configFile, api.logger);
        const allSkillNames = [...wsSkills, ...bdSkills].map((s) => s.name);

        const result = { agents, workspace, defaults, allToolGroups, allSkillNames };
        agentsCache = { data: result, ts: Date.now(), configMtime };
        jsonResponse(res, result);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/agent-tools-update
  api.registerHttpRoute({
    path: "/api/mc/agent-tools-update",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { agentId, tools } = body;
        if (!agentId || typeof agentId !== "string") { jsonResponse(res, { error: "Missing agentId" }, 400); return; }
        if (tools !== null && !Array.isArray(tools)) { jsonResponse(res, { error: "tools must be string[] or null" }, 400); return; }
        const config = JSON.parse(await readFile(configFile, "utf-8"));
        const agent = (config.agents?.list || []).find((a: any) => a.id === agentId);
        if (!agent) { jsonResponse(res, { error: `Agent '${agentId}' not found` }, 404); return; }
        if (tools === null) {
          if (agent.tools) { delete agent.tools.allow; if (Object.keys(agent.tools).length === 0) delete agent.tools; }
        } else {
          if (!agent.tools) agent.tools = {};
          agent.tools.allow = tools;
        }
        await writeFile(configFile, JSON.stringify(config, null, 2) + "\n");
        jsonResponse(res, { ok: true, agentId, tools });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/agent-skills-update
  api.registerHttpRoute({
    path: "/api/mc/agent-skills-update",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { agentId, skills } = body;
        if (!agentId || typeof agentId !== "string") { jsonResponse(res, { error: "Missing agentId" }, 400); return; }
        if (skills !== null && !Array.isArray(skills)) { jsonResponse(res, { error: "skills must be string[] or null" }, 400); return; }
        const config = JSON.parse(await readFile(configFile, "utf-8"));
        const agent = (config.agents?.list || []).find((a: any) => a.id === agentId);
        if (!agent) { jsonResponse(res, { error: `Agent '${agentId}' not found` }, 404); return; }
        if (skills === null) delete agent.skills;
        else agent.skills = skills;
        await writeFile(configFile, JSON.stringify(config, null, 2) + "\n");
        jsonResponse(res, { ok: true, agentId, skills });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/agent-file-save
  api.registerHttpRoute({
    path: "/api/mc/agent-file-save",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { agentId, fileName, content } = body;
        if (!agentId || !fileName || typeof content !== "string") { jsonResponse(res, { error: "Missing agentId, fileName, or content" }, 400); return; }
        if (!fileName.endsWith(".md") || fileName.includes("/") || fileName.includes("\\") || fileName.includes("..")) {
          jsonResponse(res, { error: "Invalid fileName — must be a simple .md filename" }, 400); return;
        }
        const config = JSON.parse(await readFile(configFile, "utf-8"));
        const agent = (config.agents?.list || []).find((a: any) => a.id === agentId);
        if (!agent) { jsonResponse(res, { error: `Agent '${agentId}' not found` }, 404); return; }
        const agentWorkspaceDir = agent.workspace
          ? agent.workspace.replace(/^~/, homedir())
          : join(rootDir, `workspaces/${agentId}`);
        const filePath = join(agentWorkspaceDir, fileName);
        if (!filePath.startsWith(agentWorkspaceDir)) { jsonResponse(res, { error: "Path traversal detected" }, 403); return; }
        await writeFile(filePath, content);
        jsonResponse(res, { ok: true, agentId, fileName });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/agent-call-log
  api.registerHttpRoute({
    path: "/api/mc/agent-call-log",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "/", "http://localhost");
        const agentId = url.searchParams.get("agentId");
        const limit = Math.min(100, parseInt(url.searchParams.get("limit") ?? "30", 10));
        if (!agentId) { jsonResponse(res, { error: "Missing agentId" }, 400); return; }
        if (!/^[a-zA-Z0-9_-]+$/.test(agentId)) { jsonResponse(res, { error: "Invalid agentId" }, 400); return; }

        const agentSessionsDir = join(rootDir, "agents", agentId, "sessions");
        try { await access(agentSessionsDir); } catch { jsonResponse(res, { entries: [] }); return; }

        const rawFiles = await readdir(agentSessionsDir);
        const sessionFilesWithMtime = await Promise.all(
          rawFiles
            .filter((f: string) => f.endsWith(".jsonl"))
            .map(async (f: string) => {
              try { return { name: f, mtime: (await stat(join(agentSessionsDir, f))).mtimeMs }; }
              catch { return { name: f, mtime: 0 }; }
            }),
        );
        const sessionFiles = sessionFilesWithMtime.sort(
          (a: { mtime: number }, b: { mtime: number }) => b.mtime - a.mtime,
        );

        if (sessionFiles.length === 0) { jsonResponse(res, { entries: [] }); return; }

        const lines = parseJsonl<SessionMessage>(join(agentSessionsDir, sessionFiles[0].name));
        const resultMap = new Map<string, { isError: boolean; preview: string }>();
        for (const line of lines) {
          if (line.type !== "message" || (line.message as any)?.role !== "toolResult") continue;
          const msg = line.message as any;
          const id = msg.toolCallId as string | undefined;
          if (!id) continue;
          const text = (msg.content ?? []).filter((c: any) => c.type === "text").map((c: any) => String(c.text)).join("").slice(0, 120);
          resultMap.set(id, { isError: msg.isError ?? false, preview: text });
        }

        const entries: AgentCallLogEntry[] = [];
        for (const line of lines) {
          if (line.type !== "message" || line.message?.role !== "assistant") continue;
          const msg = line.message as any;
          const ts = line.timestamp ?? new Date().toISOString();
          const toolCallsInMsg = (msg.content ?? []).filter((c: any) => c.type === "toolCall");

          if (msg.usage) {
            entries.push({
              ts, type: "llm",
              model: msg.model ?? undefined, provider: msg.provider ?? undefined,
              inputTokens: msg.usage.input, outputTokens: msg.usage.output,
              cost: msg.usage.cost?.total ?? undefined,
              hasToolCalls: toolCallsInMsg.length > 0,
            });
          }

          for (const item of toolCallsInMsg) {
            const result = resultMap.get((item as any).id);
            entries.push({
              ts, type: "tool",
              toolName: (item as any).name,
              args: (item as any).arguments ?? {},
              isError: result?.isError ?? false,
              resultPreview: result?.preview ?? "",
            });
          }
        }

        jsonResponse(res, { entries: entries.slice(-limit) });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/agent-avatar
  api.registerHttpRoute({
    path: "/api/mc/agent-avatar",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      const url = new URL(req.url || "/", "http://localhost");
      const id = url.searchParams.get("id");
      if (!id) { res.writeHead(400); res.end("Missing id"); return; }
      const name = url.searchParams.get("name") || undefined;
      const filePath = discoverAvatar(id, name);
      if (!filePath) { res.writeHead(404); res.end("No avatar found"); return; }
      try {
        const content = readFileSync(filePath);
        const ext = extname(filePath).toLowerCase();
        res.writeHead(200, {
          "Content-Type": AVATAR_MIME[ext] || "application/octet-stream",
          "Cache-Control": "public, max-age=3600",
        });
        res.end(content);
      } catch {
        res.writeHead(500); res.end("Error reading avatar");
      }
    },
  });

  // Stage name → agent id mapping for avatar lookup
  const STAGE_AVATAR_MAP: Record<string, string> = {
    plan: "planner",
    implement: "coder",
    review: "reviewer",
    test: "tester",
    research: "researcher",
    audit: "auditor",
  };

  const STAGES_DIR = join(homedir(), "obsidian", "agents", "worker", "stages");

  function parseYamlFrontmatter(raw: string): { meta: Record<string, string>; content: string } {
    const meta: Record<string, string> = {};
    if (!raw.startsWith("---")) return { meta, content: raw };
    const end = raw.indexOf("---", 3);
    if (end === -1) return { meta, content: raw };
    const frontmatter = raw.slice(3, end).trim();
    const content = raw.slice(end + 3).trim();
    for (const line of frontmatter.split("\n")) {
      const colon = line.indexOf(":");
      if (colon === -1) continue;
      const key = line.slice(0, colon).trim();
      const value = line.slice(colon + 1).trim();
      meta[key] = value;
    }
    return { meta, content };
  }

  // GET /api/mc/stages
  api.registerHttpRoute({
    path: "/api/mc/stages",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        let files: string[];
        try { files = await readdir(STAGES_DIR); } catch { jsonResponse(res, { stages: [] }); return; }

        const mdFiles = files.filter((f) => f.endsWith(".md") && f !== "index.md");
        const stages = await Promise.all(mdFiles.map(async (f) => {
          const stageName = f.replace(/\.md$/, "");
          let raw = "";
          try { raw = await readFile(join(STAGES_DIR, f), "utf-8"); } catch { /* skip */ }
          const { meta, content } = parseYamlFrontmatter(raw);

          const agentId = STAGE_AVATAR_MAP[stageName] ?? stageName;
          const avatarPath = discoverAvatar(agentId);
          const avatarUrl = avatarPath
            ? `/api/mc/agent-avatar?id=${encodeURIComponent(agentId)}&name=${encodeURIComponent(agentId)}`
            : null;

          return {
            name: meta.name ?? stageName,
            default_model: meta.default_model ?? "",
            signal: meta.signal ?? "STAGE_COMPLETE",
            content,
            avatar: avatarUrl,
          };
        }));

        // Sort in a logical pipeline order
        const ORDER = ["plan", "implement", "test", "review", "research", "audit"];
        stages.sort((a, b) => {
          const ai = ORDER.indexOf(a.name);
          const bi = ORDER.indexOf(b.name);
          if (ai === -1 && bi === -1) return a.name.localeCompare(b.name);
          if (ai === -1) return 1;
          if (bi === -1) return -1;
          return ai - bi;
        });

        jsonResponse(res, { stages });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/stage-save
  api.registerHttpRoute({
    path: "/api/mc/stage-save",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { name, content } = body;
        if (!name || typeof name !== "string") { jsonResponse(res, { error: "Missing name" }, 400); return; }
        if (typeof content !== "string") { jsonResponse(res, { error: "Missing content" }, 400); return; }
        // Validate: alphanumeric + hyphens only, no path traversal
        if (!/^[a-zA-Z0-9-]+$/.test(name)) { jsonResponse(res, { error: "Invalid stage name — alphanumeric and hyphens only" }, 400); return; }
        const filePath = join(STAGES_DIR, `${name}.md`);
        if (!filePath.startsWith(STAGES_DIR)) { jsonResponse(res, { error: "Path traversal detected" }, 403); return; }
        await writeFile(filePath, content);
        jsonResponse(res, { ok: true, name });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });
}
