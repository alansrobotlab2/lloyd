#!/usr/bin/env node
// vault-rename.js — Rename Obsidian vault to QMD-compatible paths
// Every name becomes: lowercase, alphanumeric + hyphen only, lowercase extension
//
// Usage:
//   node vault-rename.js                   # dry-run (show what would change)
//   node vault-rename.js --dry-run         # explicit dry-run
//   node vault-rename.js --apply           # actually rename
//   node vault-rename.js --apply --verbose # rename with per-file logging

import { readdirSync, renameSync, statSync } from 'fs';
import { join, dirname, basename } from 'path';

const VAULT_ROOT = '/home/alansrobotlab/obsidian';

// Parse CLI args
const args = process.argv.slice(2);
const apply   = args.includes('--apply');
const verbose = args.includes('--verbose');
const dryRun  = !apply;

// ─── Rename logic ─────────────────────────────────────────────────────────────

function toCompliant(name, isDir) {
  if (isDir) {
    const clean = name.toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
    return clean || name; // fallback: keep original if result would be empty
  }

  // For files: split off extension at the last dot (if not the first char)
  const lastDot = name.lastIndexOf('.');
  const hasExt  = lastDot > 0;
  const base     = hasExt ? name.slice(0, lastDot) : name;
  const ext      = hasExt ? name.slice(lastDot).toLowerCase() : '';

  const cleanBase = base.toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'untitled';

  return cleanBase + ext;
}

function isCompliant(name, isDir) {
  return toCompliant(name, isDir) === name;
}

// ─── Tree walk ────────────────────────────────────────────────────────────────

// Collect all renames bottom-up (deepest paths first so we rename
// files before their containing directories)
function collectRenames(dir) {
  const entries = readdirSync(dir, { withFileTypes: true });
  const renames = []; // { oldPath, newPath, isDir }

  for (const entry of entries) {
    // Skip hidden entries (.obsidian, .trash, .git, etc.)
    if (entry.name.startsWith('.')) continue;

    const oldPath = join(dir, entry.name);
    const isDir = entry.isDirectory();

    // Recurse first (depth-first = bottom-up)
    if (isDir) {
      renames.push(...collectRenames(oldPath));
    }

    const newName = toCompliant(entry.name, isDir);
    if (newName !== entry.name) {
      const newPath = join(dir, newName);
      renames.push({ oldPath, newPath, isDir, oldName: entry.name, newName });
    }
  }

  return renames;
}

// ─── Collision detection ──────────────────────────────────────────────────────

function detectCollisions(renames) {
  // Group by parent dir
  const byDir = new Map();
  for (const r of renames) {
    const parent = dirname(r.oldPath);
    if (!byDir.has(parent)) byDir.set(parent, []);
    byDir.get(parent).push(r);
  }

  const collisions = [];
  for (const [parent, group] of byDir) {
    const seen = new Map(); // newName → oldName
    // Also include already-compliant names in the same dir to detect conflicts
    try {
      const existing = readdirSync(parent, { withFileTypes: true })
        .filter(e => !e.name.startsWith('.') && !group.some(r => r.oldName === e.name))
        .map(e => toCompliant(e.name, e.isDirectory()));
      for (const n of existing) seen.set(n, `(existing: ${n})`);
    } catch {}

    for (const r of group) {
      if (seen.has(r.newName)) {
        collisions.push({
          dir: parent,
          conflict: r.newName,
          a: seen.get(r.newName),
          b: r.oldName,
        });
      } else {
        seen.set(r.newName, r.oldName);
      }
    }
  }
  return collisions;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

console.log(`\nVault: ${VAULT_ROOT}`);
console.log(`Mode:  ${dryRun ? 'DRY-RUN (no changes)' : 'APPLY'}\n`);

const renames = collectRenames(VAULT_ROOT);

if (renames.length === 0) {
  console.log('✓ All names already compliant — nothing to rename.');
  process.exit(0);
}

// Check collisions
const collisions = detectCollisions(renames);
if (collisions.length > 0) {
  console.error('COLLISION DETECTED — aborting:\n');
  for (const c of collisions) {
    console.error(`  In ${c.dir}:`);
    console.error(`    "${c.a}" and "${c.b}" both → "${c.conflict}"`);
  }
  console.error('\nResolve these manually before running --apply.');
  process.exit(1);
}

// Print rename plan
const dirs  = renames.filter(r => r.isDir);
const files = renames.filter(r => !r.isDir);

console.log(`${renames.length} renames needed  (${files.length} files, ${dirs.length} dirs)\n`);

if (dryRun || verbose) {
  for (const r of renames) {
    const tag  = r.isDir ? 'DIR ' : 'FILE';
    const rel  = r.oldPath.replace(VAULT_ROOT + '/', '');
    const relN = r.newPath.replace(VAULT_ROOT + '/', '');
    const dir  = dirname(rel);
    const prefix = dir === '.' ? '' : dir + '/';
    console.log(`  ${tag}  ${prefix}${r.oldName}  →  ${r.newName}`);
  }
  console.log('');
}

if (dryRun) {
  console.log('Run with --apply to execute these renames.');
  process.exit(0);
}

// Apply renames (already in bottom-up order from collectRenames)
let done = 0, errors = 0;
for (const r of renames) {
  try {
    renameSync(r.oldPath, r.newPath);
    done++;
    if (verbose) {
      const rel = r.oldPath.replace(VAULT_ROOT + '/', '');
      console.log(`  ✓  ${rel}  →  ${r.newName}`);
    }
  } catch (err) {
    errors++;
    console.error(`  ✗  ${r.oldPath}: ${err.message}`);
  }
}

console.log(`\n✓ Done: ${done} renamed${errors ? `, ${errors} errors` : ''}`);
console.log('\nNext steps:');
console.log('  1. Delete and rebuild QMD index:');
console.log('     rm ~/.openclaw/agents/main/qmd/xdg-cache/qmd/index.sqlite*');
console.log('     XDG_CONFIG_HOME=~/.openclaw/agents/main/qmd/xdg-config \\');
console.log('     XDG_CACHE_HOME=~/.openclaw/agents/main/qmd/xdg-cache \\');
console.log('     ~/.bun/install/global/node_modules/@tobilu/qmd/qmd update');
console.log('  2. Update Obsidian internal links');
console.log('  3. Restart OpenClaw');
