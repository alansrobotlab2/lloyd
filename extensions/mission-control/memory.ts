/**
 * memory.ts - Vault browsing, search, graphs, save, and frontmatter operations
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, writeFileSync, readdirSync, existsSync, statSync } from "fs";
import { join } from "path";
import { execSync, execFileSync } from "child_process";
import { homedir } from "os";
import type { PluginContext, VaultDoc } from "./types.js";
import { jsonResponse, readBody, parseFrontmatter } from "./helpers.js";

// -- Next-gen memory pipeline paths --

const pipelineRoot = join(homedir(), "obsidian", "memory", "_pipeline");
const factsIndexPath = join(pipelineRoot, "facts-index.json");
const relationsIndexPath = join(pipelineRoot, "relations-index.json");
const factsDir = join(pipelineRoot, "facts");

// -- Entity/graph caches --

interface EntitySummary {
  name: string;
  factCount: number;
  categories: string[];
}
interface EntitiesCache { entities: EntitySummary[]; total: number; }

let entitiesCache: EntitiesCache | null = null;
let entitiesCacheTs = 0;

interface GraphNode { id: string; label: string; type: string; factCount?: number; }
interface GraphEdge { source: string; target: string; type: string; weight: number; }
interface EntityGraphCache { nodes: GraphNode[]; edges: GraphEdge[]; }

let entityGraphCache: EntityGraphCache | null = null;
let entityGraphCacheTs = 0;

const ENTITY_CACHE_MS = 60_000;

function loadEntities(): EntitiesCache {
  if (entitiesCache && Date.now() - entitiesCacheTs < ENTITY_CACHE_MS) return entitiesCache;
  if (!existsSync(factsIndexPath)) return { entities: [], total: 0 };
  const raw = JSON.parse(readFileSync(factsIndexPath, "utf-8"));
  const byEntity = new Map<string, { factCount: number; categories: Set<string> }>();
  for (const entry of (raw.facts || [])) {
    const name: string = entry.entity;
    if (!byEntity.has(name)) byEntity.set(name, { factCount: 0, categories: new Set() });
    const e = byEntity.get(name)!;
    e.factCount += entry.fact_count || 0;
    if (entry.category) e.categories.add(entry.category);
  }
  const entities: EntitySummary[] = Array.from(byEntity.entries())
    .map(([name, v]) => ({ name, factCount: v.factCount, categories: Array.from(v.categories) }))
    .sort((a, b) => b.factCount - a.factCount);
  entitiesCache = { entities, total: entities.length };
  entitiesCacheTs = Date.now();
  return entitiesCache;
}

function parseFactsFile(filePath: string): Array<{ fact: string; confidence: number; category: string; event_date?: string | null; id?: string }> {
  if (!existsSync(filePath)) return [];
  try {
    const raw = readFileSync(filePath, "utf-8");
    const fm = parseFrontmatter(raw);
    if (!Array.isArray(fm.facts)) return [];
    return fm.facts.map((f: any) => ({
      fact: f.fact || "",
      confidence: f.confidence ?? 1.0,
      category: f.category || fm.category || "",
      event_date: f.event_date ?? null,
      id: f.id,
    }));
  } catch { return []; }
}

function loadEntityGraph(): EntityGraphCache {
  if (entityGraphCache && Date.now() - entityGraphCacheTs < ENTITY_CACHE_MS) return entityGraphCache;
  if (!existsSync(relationsIndexPath)) return { nodes: [], edges: [] };
  const raw = JSON.parse(readFileSync(relationsIndexPath, "utf-8"));
  const relationships: Array<{ source: string; target: string; type: string; reason?: string; score?: number }> = raw.relationships || [];

  // Build unique nodes from all doc paths in the relations
  const nodeSet = new Map<string, { id: string; label: string; type: string }>();
  for (const rel of relationships) {
    for (const p of [rel.source, rel.target]) {
      if (!nodeSet.has(p)) {
        const label = p.split("/").pop()?.replace(/\.md$/, "") || p;
        nodeSet.set(p, { id: p, label, type: "document" });
      }
    }
  }

  // Deduplicate edges by source+target (keep highest score)
  const edgeMap = new Map<string, GraphEdge>();
  for (const rel of relationships) {
    if (rel.source === rel.target) continue; // skip self-loops
    const key = [rel.source, rel.target].sort().join("\x00");
    const existing = edgeMap.get(key);
    const weight = rel.score ?? 1;
    if (!existing || weight > existing.weight) {
      edgeMap.set(key, { source: rel.source, target: rel.target, type: rel.type || "wiki-link", weight });
    }
  }

  entityGraphCache = { nodes: Array.from(nodeSet.values()), edges: Array.from(edgeMap.values()) };
  entityGraphCacheTs = Date.now();
  return entityGraphCache;
}


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
        let dirSt: import("fs").Stats;
        try { dirSt = await stat(fullPath); } catch { jsonResponse(res, { error: "Not a directory" }, 404); return; }
        if (!dirSt.isDirectory()) { jsonResponse(res, { error: "Not a directory" }, 404); return; }

        const names = await readdir(fullPath);
        const entries: any[] = [];
        for (const name of names) {
          if (VAULT_EXCLUDED.has(name) || name.startsWith(".")) continue;
          const fp = join(fullPath, name);
          let st: import("fs").Stats;
          try { st = await stat(fp); } catch { continue; }

          if (st.isDirectory()) {
            let children = 0;
            try { children = (await readdir(fp)).filter((n: string) => !n.startsWith(".")).length; } catch { /* ok */ }
            entries.push({ name, type: "dir", children });
          } else if (name.endsWith(".md")) {
            let title = name.replace(/\.md$/, "");
            try {
              const head = (await readFile(fp, "utf-8")).slice(0, 500);
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
        let resolvedSt: import("fs").Stats | null = null;
        try { resolvedSt = await stat(resolvedPath); } catch { /* not found */ }
        if (!resolvedSt || resolvedSt.isDirectory()) {
          const dir = join(resolvedPath, "..");
          const base = resolvedPath.split("/").pop()!.toLowerCase();
          let found = false;
          let dirSt2: import("fs").Stats | null = null;
          try { dirSt2 = await stat(dir); } catch { /* ok */ }
          if (dirSt2) {
            try {
              const dirEntries = await readdir(dir);
              for (const entry of dirEntries) {
                if (entry.toLowerCase() === base) { resolvedPath = join(dir, entry); found = true; break; }
              }
            } catch { /* ok */ }
          }
          if (!found) {
            try {
              const out = execFileSync('find', [vaultRoot, '-iname', base, '-type', 'f'], {
                encoding: "utf-8", timeout: 3000,
              }).trim().split('\n')[0] || '';
              if (out && out.startsWith(vaultRoot)) { resolvedPath = out; found = true; }
            } catch {}
          }
          if (!found) { jsonResponse(res, { error: "File not found" }, 404); return; }
          try { resolvedSt = await stat(resolvedPath); } catch { jsonResponse(res, { error: "File not found" }, 404); return; }
          if (!resolvedSt || resolvedSt.isDirectory()) { jsonResponse(res, { error: "File not found" }, 404); return; }
        }

        const raw = await readFile(resolvedPath, "utf-8");
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
        try { await access(fullPath); } catch { jsonResponse(res, { error: "File not found" }, 404); return; }

        const raw = await readFile(fullPath, "utf-8");
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
  // GET /api/mc/entities
  api.registerHttpRoute({
    path: "/api/mc/entities",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const limit = Math.min(parseInt(url.searchParams.get("limit") || "500", 10), 1000);
        const data = loadEntities();
        jsonResponse(res, { entities: data.entities.slice(0, limit), total: data.total });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/entity?name=X
  api.registerHttpRoute({
    path: "/api/mc/entity",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const name = url.searchParams.get("name") || "";
        if (!name) { jsonResponse(res, { error: "Missing name parameter" }, 400); return; }

        // Collect all facts for this entity
        const entityDir = join(factsDir, name);
        const allFacts: Array<{ fact: string; confidence: number; category: string; event_date?: string | null }> = [];

        if (existsSync(entityDir) && statSync(entityDir).isDirectory()) {
          const files = readdirSync(entityDir).filter((f) => f.endsWith(".md"));
          for (const fname of files) {
            const facts = parseFactsFile(join(entityDir, fname));
            allFacts.push(...facts);
          }
        }

        // Also check for lowercase entity dirs
        const lowerName = name.toLowerCase();
        const lowerEntityDir = join(factsDir, lowerName);
        if (lowerName !== name && existsSync(lowerEntityDir) && statSync(lowerEntityDir).isDirectory()) {
          const files = readdirSync(lowerEntityDir).filter((f) => f.endsWith(".md"));
          for (const fname of files) {
            const facts = parseFactsFile(join(lowerEntityDir, fname));
            allFacts.push(...facts);
          }
        }

        // Look up relationships from relations-index (entities are mentioned via doc paths)
        const graph = loadEntityGraph();
        const entityLower = name.toLowerCase();
        const relatedEdges = graph.edges.filter((e) => {
          const srcMatch = e.source.toLowerCase().includes(entityLower);
          const tgtMatch = e.target.toLowerCase().includes(entityLower);
          return srcMatch || tgtMatch;
        }).slice(0, 50).map((e) => ({
          target: e.source.toLowerCase().includes(entityLower) ? e.target : e.source,
          type: e.type,
          score: e.weight,
        }));

        jsonResponse(res, { name, facts: allFacts, relationships: relatedEdges });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/entity-graph
  api.registerHttpRoute({
    path: "/api/mc/entity-graph",
    auth: "plugin",
    handler: async (_req: IncomingMessage, res: ServerResponse) => {
      try {
        const data = loadEntityGraph();
        jsonResponse(res, data);
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

}
