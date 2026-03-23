/**
 * architecture.ts — File system browsing for Architecture tab
 * Provides endpoints for browsing ~/repos/openclaw/src/ and ~/.openclaw/extensions/
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, readdirSync, existsSync, statSync } from "fs";
import { join, extname } from "path";
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

        // Normalize path and check if allowed
        const normalizedPath = browsePath.replace(/^\/+|\/+$/g, "");
        let fullPath: string;
        
        // Check if it's one of the allowed roots
        const matchingRoot = ALLOWED_ROOTS.find(root => root.endsWith(normalizedPath) || root === normalizedPath);
        if (matchingRoot) {
          fullPath = matchingRoot;
        } else {
          // Try to find the root this path belongs to
          fullPath = join(homedir(), normalizedPath);
          if (!ALLOWED_ROOTS.some(root => fullPath.startsWith(root + "/") || fullPath === root)) {
            return jsonResponse(res, { error: "Access denied: path not in allowed directories" }, 403);
          }
        }

        if (!existsSync(fullPath)) {
          return jsonResponse(res, { error: "Path not found" }, 404);
        }

        const st = statSync(fullPath);
        if (!st.isDirectory()) {
          return jsonResponse(res, { error: "Not a directory" }, 400);
        }

        const names = readdirSync(fullPath);
        const entries: FileEntry[] = [];
        
        for (const name of names) {
          if (name.startsWith(".")) continue;
          const fp = join(fullPath, name);
          let entrySt;
          try { entrySt = statSync(fp); } catch { continue; }

          if (entrySt.isDirectory()) {
            let children = 0;
            try { 
              children = readdirSync(fp).filter(n => !n.startsWith(".")).length; 
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

        jsonResponse(res, { path: normalizedPath, entries });
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
        
        // Normalize path
        const normalizedPath = filePath.replace(/^\/+|\/+$/g, "");
        const fullPath = join(homedir(), normalizedPath);
        
        // Check if path is allowed
        if (!ALLOWED_ROOTS.some(root => fullPath.startsWith(root + "/") || fullPath === root)) {
          return jsonResponse(res, { error: "Access denied: path not in allowed directories" }, 403);
        }

        if (!existsSync(fullPath)) {
          return jsonResponse(res, { error: "File not found" }, 404);
        }

        const st = statSync(fullPath);
        if (!st.isFile()) {
          return jsonResponse(res, { error: "Not a file" }, 400);
        }

        if (st.size > 1_000_000) { // Limit to 1MB
          return jsonResponse(res, { error: "File too large (max 1MB)" }, 400);
        }

        const content = readFileSync(fullPath, "utf-8");
        
        jsonResponse(res, {
          path: normalizedPath,
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
        
        const graphUrl = `http://localhost:8100/graph${filterPath ? `?path=${encodeURIComponent(filterPath)}` : ""}`;
        
        const data = await fetch(graphUrl).then(r => r.json());
        jsonResponse(res, data);
      } catch (err: any) {
        jsonResponse(res, { error: err.message || "Failed to fetch graph data" }, 500);
      }
    },
  });
}
