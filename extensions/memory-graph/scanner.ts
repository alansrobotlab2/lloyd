/**
 * scanner.ts — Vault scanning and frontmatter extraction.
 *
 * Reads all markdown files in the Obsidian vault, parses frontmatter
 * (title, summary, tags, type, status, folder), and builds indexes
 * for tag-based document retrieval.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, basename, extname, dirname } from "node:path";

// ── Types ──────────────────────────────────────────────────────────────

export interface DocMeta {
  path: string;       // relative path from vault root
  stem: string;       // filename without .md
  title: string;      // from frontmatter or derived from stem
  type: string;       // hub, notes, project-notes, work-notes, talk, etc.
  tags: string[];     // from frontmatter tags array
  summary: string;    // from frontmatter
  status: string;     // done, paused, active, reference, etc.
  folder: string;     // from frontmatter or derived from dirname
  importance: number; // computed score (set later by TagIndex)
}

export interface ScanResult {
  docs: DocMeta[];
  tagIndex: Map<string, string[]>;    // tag -> [paths]
  docIndex: Map<string, DocMeta>;     // path -> DocMeta
  stemIndex: Map<string, string[]>;   // stem -> [paths]
}

// ── Constants ──────────────────────────────────────────────────────────

const EXCLUDE_DIRS = new Set(["templates", "images"]);
const EXCLUDE_FILES = new Set(["tags.md"]);
const MAX_FILE_SIZE = 512 * 1024; // 500KB

// Frontmatter parsing
const FM_BLOCK = /^---\n([\s\S]*?)\n---/;
const FM_TITLE = /^title:\s*"?([^"\n]+)"?\s*$/m;
const FM_SUMMARY = /^summary:\s*"?([^"\n]*(?:"[^"]*"[^"\n]*)*[^"\n]*)"?\s*$/m;
const FM_TAGS = /^tags:\s*\[([^\]]*)\]/m;
const FM_TYPE = /^type:\s*"?(\S+)"?\s*$/m;
const FM_STATUS = /^status:\s*"?(\S+)"?\s*$/m;
const FM_FOLDER = /^folder:\s*"?([^"\n]+)"?\s*$/m;

// ── Frontmatter Parsing ───────────────────────────────────────────────

interface FrontmatterResult {
  title: string;
  summary: string;
  tags: string[];
  type: string;
  status: string;
  folder: string;
}

function parseFrontmatter(text: string): FrontmatterResult {
  const fm = FM_BLOCK.exec(text);
  if (!fm) return { title: "", summary: "", tags: [], type: "", status: "", folder: "" };

  const block = fm[1];

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

  let type = "";
  const tym = FM_TYPE.exec(block);
  if (tym) type = tym[1].trim().replace(/^"|"$/g, "");

  let status = "";
  const stm = FM_STATUS.exec(block);
  if (stm) status = stm[1].trim().replace(/^"|"$/g, "");

  let folder = "";
  const flm = FM_FOLDER.exec(block);
  if (flm) folder = flm[1].trim().replace(/^"|"$/g, "");

  return { title, summary, tags, type, status, folder };
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
    if (entry.startsWith(".")) continue;        // skip all hidden files/dirs
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
      if (EXCLUDE_FILES.has(entry)) continue;
      files.push(relative(vaultRoot, fullPath));
    }
  }
  return files;
}

// ── Main Scanner ──────────────────────────────────────────────────────

export function scanVault(vaultPath: string): ScanResult {
  const files = walkMarkdownFiles(vaultPath, vaultPath);

  const docs: DocMeta[] = [];
  const tagIndex = new Map<string, string[]>();
  const docIndex = new Map<string, DocMeta>();
  const stemIndex = new Map<string, string[]>();

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
    const folder = fm.folder || dirname(relPath);

    const doc: DocMeta = {
      path: relPath,
      stem,
      title,
      type: fm.type,
      tags: fm.tags,
      summary: fm.summary,
      status: fm.status,
      folder: folder === "." ? "" : folder,
      importance: 0, // computed later by TagIndex
    };

    docs.push(doc);
    docIndex.set(relPath, doc);

    // Build stem index
    const existing = stemIndex.get(stem);
    if (existing) {
      existing.push(relPath);
    } else {
      stemIndex.set(stem, [relPath]);
    }

    // Build tag index
    for (const tag of fm.tags) {
      const tagPaths = tagIndex.get(tag);
      if (tagPaths) {
        tagPaths.push(relPath);
      } else {
        tagIndex.set(tag, [relPath]);
      }
    }
  }

  return { docs, tagIndex, docIndex, stemIndex };
}
