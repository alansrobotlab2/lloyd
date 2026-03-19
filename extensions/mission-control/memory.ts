/**
 * memory.ts — Vault browsing, search, graphs, save, and frontmatter operations
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, writeFileSync, readdirSync, existsSync, statSync } from "fs";
import { join } from "path";
import { execSync, execFileSync } from "child_process";
import { homedir } from "os";
import type { PluginContext, VaultDoc } from "./types.js";
import { jsonResponse, readBody, parseFrontmatter } from "./helpers.js";

// ── Constants ───────────────────────────────────────────────────────

const vaultRoot = join(homedir(), "obsidian");
const qmdBin = join(homedir(), ".bun/bin/qmd");
const VAULT_EXCLUDED = new Set([".obsidian", "templates", "images", ".trash"]);

// ── Vault Index Cache ───────────────────────────────────────────────

let vaultIndex: VaultDoc[] = [];
let vaultIndexTs = 0;
const VAULT_CACHE_MS = 60_000;

function walkVault(dir: string, rel: string, docs: VaultDoc[]) {
  let entries: string[];
  try { entries = readdirSync(dir); } catch { return; }
  for (const name of entries) {
    if (VAULT_EXCLUDED.has(name)) continue;
    const full = join(dir, name);
    const relPath = rel ? `${rel}/${name}` : name;
    let st;
    try { st = statSync(full); } catch { continue; }
    if (st.isDirectory()) {
      walkVault(full, relPath, docs);
    } else if (name.endsWith(".md") && st.size < 512_000) {
      try {
        const raw = readFileSync(full, "utf-8");
        const fm = parseFrontmatter(raw);
        docs.push({
          path: relPath,
          title: fm.title || name.replace(/\.md$/, ""),
          type: fm.type || "notes",
          tags: Array.isArray(fm.tags) ? fm.tags : [],
          summary: fm.summary || "",
          folder: rel || "",
        });
      } catch { /* skip unreadable */ }
    }
  }
}

function getVaultIndex(): VaultDoc[] {
  if (Date.now() - vaultIndexTs > VAULT_CACHE_MS) {
    const docs: VaultDoc[] = [];
    walkVault(vaultRoot, "", docs);
    vaultIndex = docs;
    vaultIndexTs = Date.now();
  }
  return vaultIndex;
}

// ── Route registration ──────────────────────────────────────────────

export function registerMemoryRoutes(ctx: PluginContext) {
  const { api } = ctx;

  // GET /api/mc/memory/stats
  api.registerHttpRoute({
    path: "/api/mc/memory/stats",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const docs = getVaultIndex();
        const types: Record<string, number> = {};
        const tagCounts: Record<string, number> = {};
        for (const doc of docs) {
          types[doc.type] = (types[doc.type] || 0) + 1;
          for (const tag of doc.tags) tagCounts[tag] = (tagCounts[tag] || 0) + 1;
        }
        const topTags = Object.entries(tagCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 20)
          .map(([tag, count]) => ({ tag, count }));

        jsonResponse(res, {
          docCount: docs.length,
          tagCount: Object.keys(tagCounts).length,
          types, topTags,
          lastRefresh: new Date(vaultIndexTs).toISOString(),
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/vault-graph
  api.registerHttpRoute({
    path: "/api/mc/vault-graph",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const docs = getVaultIndex();
        const nodes = docs.map((doc) => ({ id: doc.path, label: doc.title, type: doc.type, tags: doc.tags, folder: doc.folder }));

        const edges: { source: string; target: string; kind: string }[] = [];
        const pathSet = new Set(docs.map((d) => d.path));
        const pathByBasename = new Map<string, string>();
        for (const doc of docs) {
          const base = doc.path.replace(/\.md$/, "").split("/").pop()!.toLowerCase();
          pathByBasename.set(base, doc.path);
        }

        for (const doc of docs) {
          const fullPath = join(vaultRoot, doc.path);
          let raw: string;
          try { raw = readFileSync(fullPath, "utf-8"); } catch { continue; }
          const wikilinkRe = /\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]/g;
          let m: RegExpExecArray | null;
          while ((m = wikilinkRe.exec(raw)) !== null) {
            const target = m[1].trim().toLowerCase();
            let targetPath: string | undefined;
            if (pathSet.has(target + ".md")) targetPath = target + ".md";
            else targetPath = pathByBasename.get(target);
            if (targetPath && targetPath !== doc.path) {
              edges.push({ source: doc.path, target: targetPath, kind: "wikilink" });
            }
          }
        }

        const edgeKeys = new Set<string>();
        const uniqueEdges = edges.filter((e) => {
          const key = [e.source, e.target].sort().join("→");
          if (edgeKeys.has(key)) return false;
          edgeKeys.add(key);
          return true;
        });

        jsonResponse(res, { nodes, edges: uniqueEdges });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/tag-graph
  api.registerHttpRoute({
    path: "/api/mc/tag-graph",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const docs = getVaultIndex();
        const tagDocs = new Map<string, string[]>();
        for (const doc of docs) {
          for (const tag of doc.tags) {
            let arr = tagDocs.get(tag);
            if (!arr) { arr = []; tagDocs.set(tag, arr); }
            arr.push(doc.path);
          }
        }
        const nodes = Array.from(tagDocs.entries()).map(([tag, docPaths]) => ({ id: tag, label: tag, count: docPaths.length }));
        const edgeWeights = new Map<string, number>();
        for (const doc of docs) {
          const tags = doc.tags;
          for (let i = 0; i < tags.length; i++) {
            for (let j = i + 1; j < tags.length; j++) {
              const key = [tags[i], tags[j]].sort().join("→");
              edgeWeights.set(key, (edgeWeights.get(key) || 0) + 1);
            }
          }
        }
        const edges: { source: string; target: string; weight: number }[] = [];
        for (const [key, weight] of edgeWeights) {
          const [source, target] = key.split("→");
          edges.push({ source, target, weight });
        }
        jsonResponse(res, { nodes, edges });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/memory/search
  api.registerHttpRoute({
    path: "/api/mc/memory/search",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const query = url.searchParams.get("q") || "";
        const limit = Math.min(parseInt(url.searchParams.get("limit") || "10", 10), 30);
        if (!query || query.length < 2) { jsonResponse(res, { query, results: [] }); return; }

        const safeQuery = query.replace(/"/g, '\\"').replace(/\$/g, '\\$');
        const cmd = `${qmdBin} search "${safeQuery}" -c obsidian -n ${limit} --json 2>/dev/null`;
        let raw: string;
        try {
          raw = execSync(cmd, { encoding: "utf-8", timeout: 5000, maxBuffer: 1024 * 1024 });
        } catch { jsonResponse(res, { query, results: [] }); return; }

        const parsed = JSON.parse(raw);
        const docs = getVaultIndex();
        const summaryMap = new Map<string, string>();
        for (const d of docs) { if (d.summary) summaryMap.set(d.path, d.summary); }

        const results = (Array.isArray(parsed) ? parsed : []).map((r: any) => {
          const path = (r.file || "").replace(/^qmd:\/\/obsidian\//, "");
          return {
            path, title: r.title || "", score: r.score || 0,
            snippet: (r.snippet || "").replace(/@@ -\d+,?\d* @@[^\n]*\n?/g, "").trim().slice(0, 300),
            summary: summaryMap.get(path) || "",
          };
        });

        jsonResponse(res, { query, results });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/memory/tags
  api.registerHttpRoute({
    path: "/api/mc/memory/tags",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const limit = Math.min(parseInt(url.searchParams.get("limit") || "100", 10), 200);
        const docs = getVaultIndex();
        const tagCounts: Record<string, number> = {};
        for (const doc of docs) {
          for (const tag of doc.tags) tagCounts[tag] = (tagCounts[tag] || 0) + 1;
        }
        const tags = Object.entries(tagCounts).sort((a, b) => b[1] - a[1]).slice(0, limit).map(([tag, count]) => ({ tag, count }));
        jsonResponse(res, { tags });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/memory/tag-documents
  api.registerHttpRoute({
    path: "/api/mc/memory/tag-documents",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const tag = url.searchParams.get("tag");
        const limit = Math.min(parseInt(url.searchParams.get("limit") || "50", 10), 200);
        if (!tag) { jsonResponse(res, { error: "Missing tag parameter" }, 400); return; }

        const docs = getVaultIndex();
        const matches = docs
          .filter((d) => d.tags.includes(tag))
          .slice(0, limit)
          .map((d) => ({ path: d.path, title: d.title, type: d.type, summary: d.summary, folder: d.folder }));
        jsonResponse(res, { tag, count: matches.length, documents: matches });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/memory/browse
  api.registerHttpRoute({
    path: "/api/mc/memory/browse",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const browsePath = (url.searchParams.get("path") || "").replace(/^\/+|\/+$/g, "");
        const fullPath = join(vaultRoot, browsePath);
        if (!fullPath.startsWith(vaultRoot)) { jsonResponse(res, { error: "Invalid path" }, 400); return; }
        if (!existsSync(fullPath) || !statSync(fullPath).isDirectory()) { jsonResponse(res, { error: "Not a directory" }, 404); return; }

        const names = readdirSync(fullPath);
        const entries: any[] = [];
        for (const name of names) {
          if (VAULT_EXCLUDED.has(name) || name.startsWith(".")) continue;
          const fp = join(fullPath, name);
          let st;
          try { st = statSync(fp); } catch { continue; }

          if (st.isDirectory()) {
            let children = 0;
            try { children = readdirSync(fp).filter(n => !n.startsWith(".")).length; } catch { /* ok */ }
            entries.push({ name, type: "dir", children });
          } else if (name.endsWith(".md")) {
            let title = name.replace(/\.md$/, "");
            try {
              const head = readFileSync(fp, "utf-8").slice(0, 500);
              const fm = parseFrontmatter(head);
              if (fm.title) title = fm.title;
            } catch { /* ok */ }
            entries.push({ name, type: "file", size: st.size, title });
          }
        }
        entries.sort((a, b) => {
          if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
          return a.name.localeCompare(b.name);
        });
        jsonResponse(res, { path: browsePath, entries });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/memory/read
  api.registerHttpRoute({
    path: "/api/mc/memory/read",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const filePath = (url.searchParams.get("path") || "").replace(/^\/+/, "");
        const fullPath = join(vaultRoot, filePath);
        if (!fullPath.startsWith(vaultRoot)) { jsonResponse(res, { error: "Invalid path" }, 400); return; }

        let resolvedPath = fullPath;
        if (!existsSync(resolvedPath) || statSync(resolvedPath).isDirectory()) {
          const dir = join(resolvedPath, "..");
          const base = resolvedPath.split("/").pop()!.toLowerCase();
          let found = false;
          if (existsSync(dir)) {
            for (const entry of readdirSync(dir)) {
              if (entry.toLowerCase() === base) { resolvedPath = join(dir, entry); found = true; break; }
            }
          }
          if (!found) {
            try {
              const out = execFileSync('find', [vaultRoot, '-iname', base, '-type', 'f'], {
                encoding: "utf-8", timeout: 3000,
              }).trim().split('\n')[0] || '';
              if (out && out.startsWith(vaultRoot)) { resolvedPath = out; found = true; }
            } catch {}
          }
          if (!found || !existsSync(resolvedPath) || statSync(resolvedPath).isDirectory()) {
            jsonResponse(res, { error: "File not found" }, 404); return;
          }
        }

        const raw = readFileSync(resolvedPath, "utf-8");
        const fm = parseFrontmatter(raw);
        let content = raw;
        if (raw.startsWith("---")) {
          const end = raw.indexOf("\n---", 3);
          if (end !== -1) content = raw.slice(end + 4).trimStart();
        }
        const actualPath = resolvedPath.startsWith(vaultRoot) ? resolvedPath.slice(vaultRoot.length + 1) : filePath;
        jsonResponse(res, { path: actualPath, frontmatter: fm, content, lineCount: raw.split("\n").length });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // POST /api/mc/memory/save
  api.registerHttpRoute({
    path: "/api/mc/memory/save",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      if (req.method !== "POST") { jsonResponse(res, { error: "Method not allowed" }, 405); return; }
      try {
        const body = await readBody(req);
        const parsed = JSON.parse(body);
        const { path: filePath, content, frontmatter: fmEdits } = parsed;
        if (!filePath || typeof content !== "string") { jsonResponse(res, { error: "path and content required" }, 400); return; }

        const cleanPath = filePath.replace(/^\/+/, "");
        const fullPath = join(vaultRoot, cleanPath);
        if (!fullPath.startsWith(vaultRoot)) { jsonResponse(res, { error: "Invalid path" }, 400); return; }
        if (!existsSync(fullPath)) { jsonResponse(res, { error: "File not found" }, 404); return; }

        const raw = readFileSync(fullPath, "utf-8");
        let fmBlock = "";
        if (raw.startsWith("---")) {
          const end = raw.indexOf("\n---", 3);
          if (end !== -1) fmBlock = raw.slice(0, end + 4) + "\n";
        }

        if (fmEdits && typeof fmEdits === "object" && fmBlock) {
          const fmContent = fmBlock.slice(4, fmBlock.lastIndexOf("\n---"));
          const lines = fmContent.split("\n");
          const updatedLines: string[] = [];
          const handled = new Set<string>();

          for (const line of lines) {
            const match = line.match(/^(\w[\w-]*):\s*/);
            if (match && match[1] in fmEdits) {
              const key = match[1];
              handled.add(key);
              const val = fmEdits[key];
              if (Array.isArray(val)) {
                updatedLines.push(`${key}: [${val.map((v: string) => v.includes(" ") || v.includes(",") ? `"${v}"` : v).join(", ")}]`);
              } else if (val !== undefined && val !== null) {
                const sv = String(val);
                updatedLines.push(`${key}: ${sv.includes(":") || sv.includes("#") ? `"${sv}"` : sv}`);
              }
            } else {
              updatedLines.push(line);
            }
          }
          for (const [key, val] of Object.entries(fmEdits)) {
            if (handled.has(key) || val === undefined || val === null) continue;
            if (Array.isArray(val)) {
              updatedLines.push(`${key}: [${(val as string[]).map((v: string) => v.includes(" ") || v.includes(",") ? `"${v}"` : v).join(", ")}]`);
            } else {
              const sv = String(val);
              updatedLines.push(`${key}: ${sv.includes(":") || sv.includes("#") ? `"${sv}"` : sv}`);
            }
          }
          fmBlock = "---\n" + updatedLines.join("\n") + "\n---\n";
        }

        writeFileSync(fullPath, fmBlock + content, "utf-8");
        jsonResponse(res, { ok: true });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });
}
