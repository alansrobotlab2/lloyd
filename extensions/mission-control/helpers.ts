/**
 * helpers.ts — Shared utility functions for Mission Control plugin
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, existsSync, openSync, readSync, fstatSync, closeSync } from "fs";
import type { CacheEntry } from "./types.js";

// ── MIME types ──────────────────────────────────────────────────────

export const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
};

// ── Caching ─────────────────────────────────────────────────────────

export const CACHE_TTL = 5_000;
export const HEAVY_CACHE_TTL = 30_000;
const cache = new Map<string, CacheEntry<any>>();

export function cached<T>(key: string, fn: () => T, ttl: number = CACHE_TTL): T {
  const entry = cache.get(key);
  if (entry && Date.now() - entry.ts < ttl) return entry.data;
  const data = fn();
  cache.set(key, { data, ts: Date.now() });
  return data;
}

// ── JSONL parsing ───────────────────────────────────────────────────

const TAIL_CHUNK_SIZE = 8192; // 8KB chunks for tail-read

/**
 * Parse a JSONL file. When `limit` is provided, reads only the last N lines
 * using a backwards seek to avoid loading the entire file into memory.
 */
export function parseJsonl<T>(filePath: string, limit?: number): T[] {
  if (!existsSync(filePath)) return [];

  if (limit && limit > 0) {
    // Tail-read: seek backwards in chunks until we have enough newlines
    let fd: number | null = null;
    try {
      fd = openSync(filePath, "r");
      const fileSize = fstatSync(fd).size;
      if (fileSize === 0) return [];

      let pos = fileSize;
      let collectedLines: string[] = [];
      let leftover = "";

      while (pos > 0 && collectedLines.length <= limit) {
        const chunkSize = Math.min(TAIL_CHUNK_SIZE, pos);
        pos -= chunkSize;
        const buf = Buffer.allocUnsafe(chunkSize);
        readSync(fd, buf, 0, chunkSize, pos);
        const chunk = buf.toString("utf-8") + leftover;
        const parts = chunk.split("\n");
        // parts[0] may be a partial line — save for next iteration
        leftover = parts[0];
        // parts[1..] are complete lines (reversed order since we're going backwards)
        for (let i = parts.length - 1; i >= 1; i--) {
          const line = parts[i].trim();
          if (line) collectedLines.push(line);
          if (collectedLines.length > limit) break;
        }
      }
      // Don't forget the leftover (start of file)
      if (leftover.trim()) collectedLines.push(leftover.trim());

      // collectedLines is newest-first; take last `limit` and reverse to oldest-first
      const tail = collectedLines.slice(0, limit).reverse();
      const items: T[] = [];
      for (const line of tail) {
        try { items.push(JSON.parse(line)); } catch { /* skip malformed */ }
      }
      return items;
    } catch {
      // Fall through to full-read on any error
      if (fd !== null) { try { closeSync(fd); } catch { /* ignore */ } fd = null; }
    } finally {
      if (fd !== null) { try { closeSync(fd); } catch { /* ignore */ } }
    }
  }

  // Full read (no limit, or tail-read failed)
  const content = readFileSync(filePath, "utf-8");
  const lines = content.trim().split("\n").filter(Boolean);
  const items: T[] = [];
  for (const line of lines) {
    try { items.push(JSON.parse(line)); } catch { /* skip malformed JSONL line */ }
  }
  return items;
}

// ── HTTP helpers ────────────────────────────────────────────────────

const CORS_ORIGIN = "http://localhost:5173";

export function jsonResponse(res: ServerResponse, data: any, status = 200) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": CORS_ORIGIN,
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  });
  res.end(JSON.stringify(data));
}

export function readBody(req: IncomingMessage, maxBytes = 1_000_000): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > maxBytes) { req.destroy(); reject(new Error("Body too large")); }
      else chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

/** Handle CORS OPTIONS preflight. Returns true if handled (caller should return). */
export function handleCorsOptions(req: IncomingMessage, res: ServerResponse): boolean {
  if (req.method === "OPTIONS") {
    res.writeHead(200, {
      "Access-Control-Allow-Origin": CORS_ORIGIN,
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    });
    res.end();
    return true;
  }
  return false;
}

/** Guard for POST-only endpoints. Returns true if method is not POST (error sent). */
export function requirePost(req: IncomingMessage, res: ServerResponse): boolean {
  if (req.method !== "POST") {
    jsonResponse(res, { error: "Method not allowed" }, 405);
    return true;
  }
  return false;
}

// ── File helpers ────────────────────────────────────────────────────

export function readFileOpt(p: string): string | null {
  try { return existsSync(p) ? readFileSync(p, "utf-8") : null; } catch { return null; }
}

// ── Frontmatter parsing ─────────────────────────────────────────────

export function parseFrontmatter(raw: string): Record<string, any> {
  const fm: Record<string, any> = {};
  if (!raw.startsWith("---")) return fm;
  const end = raw.indexOf("\n---", 3);
  if (end === -1) return fm;
  const block = raw.slice(4, end);
  for (const line of block.split("\n")) {
    const m = line.match(/^(\w[\w_-]*):\s*(.*)/);
    if (!m) continue;
    const [, key, val] = m;
    if (val.startsWith("[") && val.endsWith("]")) {
      fm[key] = val.slice(1, -1).split(",").map((s: string) => s.trim().replace(/^["']|["']$/g, "")).filter(Boolean);
    } else {
      fm[key] = val.replace(/^["']|["']$/g, "").trim();
    }
  }
  return fm;
}
