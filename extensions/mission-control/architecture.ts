/**
 * architecture.ts — File system browsing for Architecture tab
 * Provides endpoints for browsing ~/repos/openclaw/src/ and ~/.openclaw/extensions/
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, readdirSync, existsSync, statSync } from "fs";
import { readFile, access, readdir, stat } from "fs/promises";
import { join, extname, resolve, relative, dirname } from "path";
import { homedir } from "os";
import type { PluginContext } from "./types.js";
import { jsonResponse } from "./helpers.js";

// Allowed root directories for browsing
const ALLOWED_ROOTS = [
  join(homedir(), "repos", "openclaw", "src"),
  join(homedir(), ".openclaw", "extensions"),
];

interface FileEntry {
  name: string;
  path: string;
  type: "file" | "dir";
  size?: number;
  children?: number;
}

interface BrowseResult {
  path: string;
  entries: FileEntry[];
}

interface FileContentResult {
  path: string;
  content: string;
  language: string;
  lineCount: number;
}

function isPathAllowed(browsePath: string): boolean {
  const fullPath = join(browsePath);
  return ALLOWED_ROOTS.some(root => fullPath.startsWith(root));
}

function getLanguage(path: string): string {
  const ext = extname(path).toLowerCase();
  const map: Record<string, string> = {
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
    ".md": "markdown",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".css": "css",
    ".html": "html",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".toml": "toml",
    ".txt": "text",
  };
  return map[ext] || "text";
}

export function registerArchitectureRoutes(ctx: PluginContext) {
  const { api } = ctx;

  // GET /api/mc/architecture/browse
  api.registerHttpRoute({
    path: "/api/mc/architecture/browse",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        let browsePath = url.searchParams.get("path") || "";
        
        // Handle root paths
        if (!browsePath || browsePath === "/") {
          const entries: FileEntry[] = ALLOWED_ROOTS.map(root => ({
            name: root.split("/").pop() || root,
            path: root,
            type: "dir" as const,
          }));
          return jsonResponse(res, { path: "/", entries });
        }

        // Normalize and resolve path
        let fullPath: string;
        let responsePath: string;
        
        // If it's an absolute path, use directly
        if (browsePath.startsWith("/")) {
          fullPath = browsePath;
          responsePath = browsePath;
        } else {
          // Relative path — try to match against allowed roots
          const normalizedPath = browsePath.replace(/^\/+|\/+$/g, "");
          const matchingRoot = ALLOWED_ROOTS.find(root => root.endsWith(normalizedPath) || root === normalizedPath);
          fullPath = matchingRoot || join(homedir(), normalizedPath);
          responsePath = normalizedPath;
        }
        
        // Security check
        if (!ALLOWED_ROOTS.some(root => fullPath.startsWith(root + "/") || fullPath === root)) {
          return jsonResponse(res, { error: "Access denied: path not in allowed directories" }, 403);
        }

        let fullPathSt: import("fs").Stats;
        try { fullPathSt = await stat(fullPath); } catch { return jsonResponse(res, { error: "Path not found" }, 404); }
        if (!fullPathSt.isDirectory()) {
          return jsonResponse(res, { error: "Not a directory" }, 400);
        }

        const names = await readdir(fullPath);
        const entries: FileEntry[] = [];
        
        for (const name of names) {
          if (name.startsWith(".")) continue;
          const fp = join(fullPath, name);
          let entrySt: import("fs").Stats;
          try { entrySt = await stat(fp); } catch { continue; }

          if (entrySt.isDirectory()) {
            let children = 0;
            try { 
              children = (await readdir(fp)).filter((n: string) => !n.startsWith(".")).length; 
            } catch { /* ok */ }
            entries.push({ name, path: fp, type: "dir", children });
          } else if (entrySt.isFile() && entrySt.size < 1_000_000) { // Limit file size to 1MB
            entries.push({ name, path: fp, type: "file", size: entrySt.size });
          }
        }
        
        // Sort: directories first, then files, alphabetically
        entries.sort((a, b) => {
          if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
          return a.name.localeCompare(b.name);
        });

        jsonResponse(res, { path: responsePath, entries });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/architecture/read
  api.registerHttpRoute({
    path: "/api/mc/architecture/read",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        let filePath = url.searchParams.get("path") || "";
        
        let fullPath: string;
        let responsePath: string;
        if (filePath.startsWith("/")) {
          fullPath = filePath;
          responsePath = filePath;
        } else {
          const normalizedPath = filePath.replace(/^\/+|\/+$/g, "");
          fullPath = join(homedir(), normalizedPath);
          responsePath = normalizedPath;
        }
        
        // Security check
        if (!ALLOWED_ROOTS.some(root => fullPath.startsWith(root + "/") || fullPath === root)) {
          return jsonResponse(res, { error: "Access denied: path not in allowed directories" }, 403);
        }

        let readSt: import("fs").Stats;
        try { readSt = await stat(fullPath); } catch { return jsonResponse(res, { error: "File not found" }, 404); }
        if (!readSt.isFile()) {
          return jsonResponse(res, { error: "Not a file" }, 400);
        }

        if (readSt.size > 1_000_000) { // Limit to 1MB
          return jsonResponse(res, { error: "File too large (max 1MB)" }, 400);
        }

        const content = await readFile(fullPath, "utf-8");
        
        jsonResponse(res, {
          path: responsePath,
          content,
          language: getLanguage(filePath),
          lineCount: content.split("\n").length,
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message }, 500);
      }
    },
  });

  // GET /api/mc/architecture/graph
  api.registerHttpRoute({
    path: "/api/mc/architecture/graph",
    auth: "plugin",
    handler: async (req: IncomingMessage, res: ServerResponse) => {
      try {
        const url = new URL(req.url || "", "http://localhost");
        const filterPath = url.searchParams.get("path") || "";
        
        const extensionsRoot = join(homedir(), ".openclaw", "extensions", "mission-control");
        
        // Recursively find all .ts/.tsx files
        const allFiles: string[] = [];
        function walkDir(dir: string) {
          let entries: string[];
          try { entries = readdirSync(dir); } catch { return; }
          for (const name of entries) {
            if (name.startsWith(".") || name === "node_modules" || name === "dist-web" || name === "dist") continue;
            const fp = join(dir, name);
            let st: import("fs").Stats;
            try { st = statSync(fp); } catch { continue; }
            if (st.isDirectory()) {
              walkDir(fp);
            } else if (/\.(ts|tsx)$/.test(name) && !name.endsWith(".d.ts")) {
              allFiles.push(fp);
            }
          }
        }
        walkDir(extensionsRoot);
        
        // Parse imports from each file
        const importRegex = /import\s+(?:(?:type\s+)?(?:\{[^}]*\}|[\w*]+(?:\s*,\s*\{[^}]*\})?)\s+from\s+)?['"](\.\.?\/[^'"]+)['"]/g;
        
        interface NodeInfo { id: string; path: string; count: number; }
        interface LinkInfo { source: string; target: string; }
        
        const nodesMap = new Map<string, NodeInfo>();
        const links: LinkInfo[] = [];
        
        for (const filePath of allFiles) {
          const relPath = relative(extensionsRoot, filePath);
          let content: string;
          try { content = readFileSync(filePath, "utf-8"); } catch { continue; }
          
          const imports: string[] = [];
          let match: RegExpExecArray | null;
          const regex = new RegExp(importRegex.source, importRegex.flags);
          while ((match = regex.exec(content)) !== null) {
            imports.push(match[1]);
          }
          
          // Register node
          if (!nodesMap.has(relPath)) {
            nodesMap.set(relPath, { id: relPath, path: filePath, count: imports.length });
          }
          
          // Resolve imports to actual files
          for (const imp of imports) {
            const importDir = dirname(filePath);
            const resolved = resolve(importDir, imp);
            
            // Try extensions: .ts, .tsx, /index.ts, /index.tsx
            const candidates = [
              resolved + ".ts",
              resolved + ".tsx",
              join(resolved, "index.ts"),
              join(resolved, "index.tsx"),
              resolved, // exact match (rare)
            ];
            
            let targetPath: string | null = null;
            for (const c of candidates) {
              if (existsSync(c)) {
                targetPath = c;
                break;
              }
            }
            
            if (targetPath && targetPath.startsWith(extensionsRoot)) {
              const targetRel = relative(extensionsRoot, targetPath);
              if (!nodesMap.has(targetRel)) {
                nodesMap.set(targetRel, { id: targetRel, path: targetPath, count: 0 });
              }
              links.push({ source: relPath, target: targetRel });
            }
          }
        }
        
        let nodes = Array.from(nodesMap.values());
        let filteredLinks = links;
        
        // Apply path filter if provided
        if (filterPath) {
          const matchingNodeIds = new Set<string>();
          for (const node of nodes) {
            if (node.id.includes(filterPath) || node.path.includes(filterPath)) {
              matchingNodeIds.add(node.id);
            }
          }
          // Also include directly connected nodes
          const connectedIds = new Set(matchingNodeIds);
          for (const link of links) {
            if (matchingNodeIds.has(link.source)) connectedIds.add(link.target);
            if (matchingNodeIds.has(link.target)) connectedIds.add(link.source);
          }
          nodes = nodes.filter(n => connectedIds.has(n.id));
          const nodeIdSet = new Set(nodes.map(n => n.id));
          filteredLinks = links.filter(l => nodeIdSet.has(l.source) && nodeIdSet.has(l.target));
        }
        
        jsonResponse(res, {
          nodes,
          links: filteredLinks,
          totalImports: links.reduce((sum, _) => sum + 1, 0),
          totalNodes: nodes.length,
          totalLinks: filteredLinks.length,
        });
      } catch (err: any) {
        jsonResponse(res, { error: err.message || "Failed to build graph" }, 500);
      }
    },
  });
}
