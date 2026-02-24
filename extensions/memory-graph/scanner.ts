/**
 * scanner.ts — Vault scanning and wiki-link extraction.
 *
 * Reads all markdown files in the Obsidian vault, parses frontmatter,
 * extracts wiki-links from body text, and builds a stem-to-path index
 * for link resolution. Compatible with autolink.py's link format.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, basename, extname } from "node:path";

// ── Types ──────────────────────────────────────────────────────────────

export interface ScannedNode {
  path: string;      // relative path from vault root, e.g. "projects/stompy/stompy-overview.md"
  stem: string;      // filename without .md
  title: string;     // from frontmatter or derived from stem
  summary: string;   // from frontmatter
  tags: string[];    // from frontmatter
}

export interface ScannedEdge {
  source: string;    // relative path of document containing the link
  target: string;    // relative path of linked document (resolved)
  alias: string | null; // display text if aliased
}

export interface ScanResult {
  nodes: ScannedNode[];
  edges: ScannedEdge[];
  stemIndex: Map<string, string[]>;  // stem -> [paths]
  brokenLinks: Map<string, string[]>; // source path -> [unresolved targets]
}

// ── Constants ──────────────────────────────────────────────────────────

const EXCLUDE_DIRS = new Set([".obsidian", "templates", "images", ".git", ".trash"]);
const MAX_FILE_SIZE = 512 * 1024; // 500KB

// Match [[target]] and [[target|alias]], excluding ![[embeds]] and [[text]](url)
const WIKILINK_RE = /(?<!!)\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\](?!\()/g;

// Frontmatter parsing (matches autolink.py patterns)
const FM_BLOCK = /^---\n([\s\S]*?)\n---/;
const FM_TITLE = /^title:\s*"?([^"\n]+)"?\s*$/m;
const FM_SUMMARY = /^summary:\s*"?([^"\n]*(?:"[^"]*"[^"\n]*)*[^"\n]*)"?\s*$/m;
const FM_TAGS = /^tags:\s*\[([^\]]*)\]/m;

// Protected zone patterns (code blocks, inline code, URLs, etc.)
const FENCED_CODE = /(?:^|\n)(```[^\n]*\n[\s\S]*?\n```|~~~[^\n]*\n[\s\S]*?\n~~~)/g;
const INLINE_CODE = /`[^`\n]+`/g;
const URL_PATTERN = /https?:\/\/\S+/g;
const MD_LINK = /\[[^\]]*\]\([^)]*\)/g;
const HEADING = /^#{1,6}\s.*$/gm;

// ── Frontmatter Parsing ───────────────────────────────────────────────

function parseFrontmatter(text: string): { title: string; summary: string; tags: string[]; bodyStart: number } {
  const fm = FM_BLOCK.exec(text);
  if (!fm) return { title: "", summary: "", tags: [], bodyStart: 0 };

  const block = fm[1];
  const bodyStart = fm[0].length;

  let title = "";
  const tm = FM_TITLE.exec(block);
  if (tm) title = tm[1].trim().replace(/^"|"$/g, "");

  let summary = "";
  const sm = FM_SUMMARY.exec(block);
  if (sm) summary = sm[1].trim().replace(/^"|"$/g, "");

  const tags: string[] = [];
  const tgm = FM_TAGS.exec(block);
  if (tgm) {
    for (const t of tgm[1].split(",")) {
      const trimmed = t.trim().replace(/^"|"$/g, "");
      if (trimmed) tags.push(trimmed);
    }
  }

  return { title, summary, tags, bodyStart };
}

// ── Protected Zone Detection ──────────────────────────────────────────

interface Zone { start: number; end: number }

function findProtectedZones(text: string, bodyStart: number): Zone[] {
  const zones: Zone[] = [];

  // Frontmatter itself
  if (bodyStart > 0) zones.push({ start: 0, end: bodyStart });

  // Fenced code blocks
  for (const m of text.matchAll(FENCED_CODE)) {
    zones.push({ start: m.index!, end: m.index! + m[0].length });
  }
  // Inline code
  for (const m of text.matchAll(INLINE_CODE)) {
    zones.push({ start: m.index!, end: m.index! + m[0].length });
  }
  // URLs
  for (const m of text.matchAll(URL_PATTERN)) {
    zones.push({ start: m.index!, end: m.index! + m[0].length });
  }
  // Markdown links
  for (const m of text.matchAll(MD_LINK)) {
    zones.push({ start: m.index!, end: m.index! + m[0].length });
  }
  // Headings
  for (const m of text.matchAll(HEADING)) {
    zones.push({ start: m.index!, end: m.index! + m[0].length });
  }

  // Sort and merge overlapping zones
  zones.sort((a, b) => a.start - b.start);
  const merged: Zone[] = [];
  for (const z of zones) {
    if (merged.length > 0 && z.start <= merged[merged.length - 1].end) {
      merged[merged.length - 1].end = Math.max(merged[merged.length - 1].end, z.end);
    } else {
      merged.push({ ...z });
    }
  }
  return merged;
}

function inZone(pos: number, endPos: number, zones: Zone[]): boolean {
  for (const z of zones) {
    if (pos < z.end && endPos > z.start) return true;
    if (z.start > endPos) break; // zones are sorted
  }
  return false;
}

// ── Wiki-link Extraction ──────────────────────────────────────────────

interface RawLink {
  target: string;
  alias: string | null;
}

function extractWikiLinks(text: string, bodyStart: number): RawLink[] {
  const zones = findProtectedZones(text, bodyStart);
  const links: RawLink[] = [];

  for (const m of text.matchAll(WIKILINK_RE)) {
    if (inZone(m.index!, m.index! + m[0].length, zones)) continue;
    links.push({
      target: m[1].trim(),
      alias: m[2]?.trim() ?? null,
    });
  }
  return links;
}

// ── File Discovery ────────────────────────────────────────────────────

function walkMarkdownFiles(dir: string, vaultRoot: string): string[] {
  const files: string[] = [];
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return files;
  }
  for (const entry of entries) {
    if (EXCLUDE_DIRS.has(entry)) continue;
    const fullPath = join(dir, entry);
    let stat;
    try {
      stat = statSync(fullPath);
    } catch {
      continue;
    }
    if (stat.isDirectory()) {
      files.push(...walkMarkdownFiles(fullPath, vaultRoot));
    } else if (stat.isFile() && extname(entry) === ".md" && stat.size <= MAX_FILE_SIZE) {
      files.push(relative(vaultRoot, fullPath));
    }
  }
  return files;
}

// ── Main Scanner ──────────────────────────────────────────────────────

export function scanVault(vaultPath: string): ScanResult {
  const files = walkMarkdownFiles(vaultPath, vaultPath);

  // Phase 1: Build nodes and stem index
  const nodes: ScannedNode[] = [];
  const stemIndex = new Map<string, string[]>();
  const fileContents = new Map<string, { text: string; bodyStart: number }>();

  for (const relPath of files) {
    const fullPath = join(vaultPath, relPath);
    let text: string;
    try {
      text = readFileSync(fullPath, "utf-8");
    } catch {
      continue;
    }

    const fm = parseFrontmatter(text);
    const stem = basename(relPath, ".md");
    const title = fm.title || stem.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

    nodes.push({
      path: relPath,
      stem,
      title,
      summary: fm.summary,
      tags: fm.tags,
    });

    fileContents.set(relPath, { text, bodyStart: fm.bodyStart });

    const existing = stemIndex.get(stem);
    if (existing) {
      existing.push(relPath);
    } else {
      stemIndex.set(stem, [relPath]);
    }
  }

  // Phase 2: Extract links and resolve targets
  const edges: ScannedEdge[] = [];
  const brokenLinks = new Map<string, string[]>();

  for (const node of nodes) {
    const content = fileContents.get(node.path);
    if (!content) continue;

    const rawLinks = extractWikiLinks(content.text, content.bodyStart);

    for (const link of rawLinks) {
      const resolved = resolveTarget(link.target, stemIndex, vaultPath);
      if (resolved) {
        // Skip self-links
        if (resolved === node.path) continue;
        edges.push({
          source: node.path,
          target: resolved,
          alias: link.alias,
        });
      } else {
        const broken = brokenLinks.get(node.path) ?? [];
        broken.push(link.target);
        brokenLinks.set(node.path, broken);
      }
    }
  }

  return { nodes, edges, stemIndex, brokenLinks };
}

// ── Target Resolution ─────────────────────────────────────────────────

function resolveTarget(
  target: string,
  stemIndex: Map<string, string[]>,
  _vaultPath: string,
): string | null {
  // 1. If target contains "/" treat as a path
  if (target.includes("/")) {
    const withMd = target.endsWith(".md") ? target : target + ".md";
    // Check if this path exists in the stem index (by scanning nodes)
    for (const paths of stemIndex.values()) {
      for (const p of paths) {
        if (p === withMd) return p;
      }
    }
    return null;
  }

  // 2. Exact stem match
  const paths = stemIndex.get(target);
  if (paths) {
    if (paths.length === 1) return paths[0];
    // Ambiguous — skip
    return null;
  }

  // 3. Case-insensitive stem match
  const lowerTarget = target.toLowerCase();
  for (const [stem, paths] of stemIndex) {
    if (stem.toLowerCase() === lowerTarget) {
      if (paths.length === 1) return paths[0];
      return null; // ambiguous
    }
  }

  return null;
}
