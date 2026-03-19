/**
 * helpers.ts — Shared utility functions for Mission Control plugin
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync, existsSync } from "fs";
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

export function parseJsonl<T>(filePath: string, limit?: number): T[] {
  if (!existsSync(filePath)) return [];
  const content = readFileSync(filePath, "utf-8");
  const lines = content.trim().split("\n").filter(Boolean);
  const items: T[] = [];
  const start = limit ? Math.max(0, lines.length - limit) : 0;
  for (let i = start; i < lines.length; i++) {
    try {
      items.push(JSON.parse(lines[i]));
    } catch { /* skip malformed JSONL line */ }
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
