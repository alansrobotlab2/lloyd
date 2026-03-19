/**
 * skills.ts — Skill discovery, validation, and content CRUD
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, writeFileSync, readdirSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import type { PluginContext, SkillInfo } from "./types.js";
import { jsonResponse, readBody } from "./helpers.js";

const YAML = require(join(homedir(), ".npm-global/lib/node_modules/openclaw/node_modules/yaml"));

// ── Skill directory resolution ──────────────────────────────────────

export function resolveSkillDirs(configFile: string): { workspaceSkillsDirs: string[]; bundledSkillsDir: string } {
  const dirs: string[] = [];
  try {
    const cfg = JSON.parse(readFileSync(configFile, "utf-8"));
    const mainAgent = cfg.agents?.list?.find((a: any) => a.id === "main") || cfg.agents?.list?.[0];
    const ws = mainAgent?.workspace?.replace(/^~/, homedir()) ?? null;
    dirs.push(ws ? join(ws, "skills") : join(homedir(), ".openclaw/state/workspaces/lloyd/skills"));
    const extraDirs: string[] = cfg.skills?.load?.extraDirs ?? [];
    for (const d of extraDirs) dirs.push(d.replace(/^~/, homedir()));
  } catch {
    dirs.push(join(homedir(), ".openclaw/state/workspaces/lloyd/skills"));
  }
  return {
    workspaceSkillsDirs: dirs,
    bundledSkillsDir: join(homedir(), ".npm-global/lib/node_modules/openclaw/skills"),
  };
}

// ── Binary / config checks ──────────────────────────────────────────

const binExistsCache = new Map<string, boolean>();

function hasBinary(bin: string): boolean {
  if (binExistsCache.has(bin)) return binExistsCache.get(bin)!;
  const parts = (process.env.PATH ?? "").split(require("path").delimiter).filter(Boolean);
  for (const part of parts) {
    try {
      const candidate = join(part, bin);
      require("fs").accessSync(candidate, require("fs").constants.X_OK);
      binExistsCache.set(bin, true);
      return true;
    } catch { /* non-fatal */ }
  }
  binExistsCache.set(bin, false);
  return false;
}

const DEFAULT_CONFIG_VALUES: Record<string, any> = {
  "browser.enabled": true,
  "browser.evaluateEnabled": true,
};

function isConfigPathTruthy(config: any, pathStr: string): boolean {
  const parts = pathStr.split(".");
  let current = config;
  for (const part of parts) {
    if (current == null || typeof current !== "object") return Boolean(DEFAULT_CONFIG_VALUES[pathStr]);
    current = current[part];
  }
  if (current == null) return Boolean(DEFAULT_CONFIG_VALUES[pathStr]);
  return Boolean(current);
}

function checkSkillConfigured(requires: SkillInfo["requires"], os: string[] | undefined, config: any): boolean {
  if (os && os.length > 0 && !os.includes(process.platform)) return false;
  if (!requires) return true;
  for (const bin of requires.bins ?? []) { if (!hasBinary(bin)) return false; }
  const anyBins = requires.anyBins ?? [];
  if (anyBins.length > 0 && !anyBins.some((b) => hasBinary(b))) return false;
  for (const envName of requires.env ?? []) { if (!process.env[envName]) return false; }
  for (const configPath of requires.config ?? []) { if (!isConfigPathTruthy(config, configPath)) return false; }
  return true;
}

// ── Skill parsing ───────────────────────────────────────────────────

export function parseSkillDir(dir: string, configFile: string, logger: any): SkillInfo[] {
  const skills: SkillInfo[] = [];
  if (!existsSync(dir)) return skills;
  let config: any = {};
  try { config = JSON.parse(readFileSync(configFile, "utf-8")); } catch { /* non-fatal */ }
  const skillEntries = config?.skills?.entries ?? {};
  for (const entry of readdirSync(dir)) {
    const skillFile = join(dir, entry, "SKILL.md");
    if (!existsSync(skillFile)) continue;
    try {
      const raw = readFileSync(skillFile, "utf-8");
      const fmMatch = raw.match(/^---\n([\s\S]*?)\n---/);
      if (!fmMatch) continue;
      const fm = YAML.parse(fmMatch[1]);
      const oc = fm?.metadata?.openclaw ?? fm?.metadata?.["openclaw"] ?? {};
      const skillKey = fm.name || entry;
      const skillConfig = skillEntries[skillKey] ?? skillEntries[entry] ?? {};
      const enabled = skillConfig.enabled !== false;
      const configured = checkSkillConfigured(oc.requires, oc.os, config);
      skills.push({
        name: skillKey,
        description: fm.description || "",
        emoji: oc.emoji,
        requires: oc.requires,
        os: oc.os,
        enabled, configured,
        location: skillFile,
      });
    } catch (e) { logger.debug?.(`mission-control: parseSkillDir skill parse: ${(e as Error).message}`); }
  }
  return skills.sort((a, b) => a.name.localeCompare(b.name));
}

// ── Deduplicated skill file lookup ──────────────────────────────────

function findSkillFile(skillName: string, workspaceSkillsDirs: string[], bundledSkillsDir: string): string | null {
  const dirs = [...workspaceSkillsDirs, bundledSkillsDir];
  for (const dir of dirs) {
    if (!existsSync(dir)) continue;
    for (const entry of readdirSync(dir)) {
      const skillFile = join(dir, entry, "SKILL.md");
      if (!existsSync(skillFile)) continue;
      try {
        const raw = readFileSync(skillFile, "utf-8");
        const fmMatch = raw.match(/^---\n([\s\S]*?)\n---/);
        if (!fmMatch) continue;
        const fm = YAML.parse(fmMatch[1]);
        const skillKey = fm.name || entry;
        if (skillKey === skillName || entry === skillName) return skillFile;
      } catch { /* non-fatal */ }
    }
  }
  return null;
}

// ── Route registration ──────────────────────────────────────────────

export function registerSkillRoutes(
  ctx: PluginContext,
  skillDirs: { workspaceSkillsDirs: string[]; bundledSkillsDir: string },
) {
  const { api, configFile } = ctx;
  const { workspaceSkillsDirs, bundledSkillsDir } = skillDirs;

  let skillsCache: { data: { workspace: SkillInfo[]; bundled: SkillInfo[] }; ts: number } | null = null;

  // GET /api/mc/skills
  api.registerHttpRoute({
    path: "/api/mc/skills",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const now = Date.now();
        if (!skillsCache || now - skillsCache.ts > 30_000) {
          const allWorkspace = workspaceSkillsDirs.flatMap(d => parseSkillDir(d, configFile, api.logger));
          const seen = new Set<string>();
          const workspace: SkillInfo[] = [];
          for (const s of allWorkspace) {
            if (!seen.has(s.name)) { seen.add(s.name); workspace.push(s); }
          }
          skillsCache = {
            data: {
              workspace: workspace.sort((a, b) => a.name.localeCompare(b.name)),
              bundled: parseSkillDir(bundledSkillsDir, configFile, api.logger),
            },
            ts: now,
          };
        }
        jsonResponse(res, skillsCache.data);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/skill-toggle
  api.registerHttpRoute({
    path: "/api/mc/skill-toggle",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "POST only" }, 405); return; }
      try {
        const body = JSON.parse(await readBody(req));
        const { skillName, enabled } = body;
        if (!skillName || typeof enabled !== "boolean") { jsonResponse(res, { error: "Missing skillName or enabled (boolean)" }, 400); return; }
        const config = JSON.parse(readFileSync(configFile, "utf-8"));
        if (!config.skills) config.skills = {};
        if (!config.skills.entries) config.skills.entries = {};
        if (enabled) {
          if (config.skills.entries[skillName]) {
            delete config.skills.entries[skillName].enabled;
            if (Object.keys(config.skills.entries[skillName]).length === 0) delete config.skills.entries[skillName];
          }
        } else {
          if (!config.skills.entries[skillName]) config.skills.entries[skillName] = {};
          config.skills.entries[skillName].enabled = false;
        }
        if (Object.keys(config.skills.entries).length === 0) delete config.skills.entries;
        if (Object.keys(config.skills).length === 0) delete config.skills;
        writeFileSync(configFile, JSON.stringify(config, null, 2) + "\n");
        skillsCache = null;
        jsonResponse(res, { ok: true, skillName, enabled });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET/POST /api/mc/skill-content
  api.registerHttpRoute({
    path: "/api/mc/skill-content",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method === "GET") {
        const url = new URL(req.url!, `http://localhost`);
        const skillName = url.searchParams.get("name");
        if (!skillName) { jsonResponse(res, { error: "Missing name parameter" }, 400); return; }
        const skillFile = findSkillFile(skillName, workspaceSkillsDirs, bundledSkillsDir);
        if (!skillFile) { jsonResponse(res, { error: "Skill not found" }, 404); return; }
        const raw = readFileSync(skillFile, "utf-8");
        jsonResponse(res, { content: raw, location: skillFile });
        return;
      }

      if (req.method === "POST") {
        try {
          const body = JSON.parse(await readBody(req));
          const { skillName, content } = body;
          if (!skillName || typeof content !== "string") { jsonResponse(res, { error: "Missing skillName or content" }, 400); return; }
          const skillFile = findSkillFile(skillName, workspaceSkillsDirs, bundledSkillsDir);
          if (!skillFile) { jsonResponse(res, { error: "Skill not found" }, 404); return; }
          writeFileSync(skillFile, content, "utf-8");
          skillsCache = null;
          jsonResponse(res, { ok: true });
        } catch (err: any) {
          jsonResponse(res, { error: err.message }, 500);
        }
        return;
      }

      jsonResponse(res, { error: "GET or POST only" }, 405);
    },
  });
}
