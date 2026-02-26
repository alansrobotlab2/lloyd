/**
 * format.ts — LLM context formatting for tag-based and unified results.
 *
 * Formats tag index data into structured text blocks that the LLM
 * can understand and act on in conversation.
 */

import type { DocMeta } from "./scanner.js";
import type { TagIndexStats } from "./tag-index.js";

// ── Helpers ───────────────────────────────────────────────────────────

function formatDocLine(doc: DocMeta, maxSummary = 120): string {
  const typePart = doc.type ? doc.type : "";
  const statusPart = doc.status ? doc.status : "";
  const badge = [typePart, statusPart].filter(Boolean).join("|");
  const badgeStr = badge ? ` [${badge}]` : "";
  const summary = doc.summary
    ? ` — ${doc.summary.length > maxSummary ? doc.summary.slice(0, maxSummary) + "..." : doc.summary}`
    : "";
  return `- ${doc.path}${badgeStr} "${doc.title}"${summary}\n  Tags: ${doc.tags.join(", ") || "(none)"}`;
}

// ── Unified Context (before_prompt_build) ─────────────────────────────

interface UnifiedCandidate {
  path: string;
  doc?: DocMeta;
  vectorScore: number;
  tagScore: number;
  snippet?: string;
  sources: Set<string>;
  finalScore: number;
  content?: string;
}

function sourceLabel(sources: Set<string>): string {
  const parts: string[] = [];
  if (sources.has("vector") || sources.has("glm")) parts.push("vector");
  if (sources.has("tag")) parts.push("tag");
  return parts.join("+");
}

export function formatUnifiedContext(
  tier1: UnifiedCandidate[],
  tier2: UnifiedCandidate[],
  budget: number,
): string {
  if (tier1.length === 0 && tier2.length === 0) return "";

  const lines: string[] = [
    "<memory_context>",
    "Pre-fetched memory context. Use this to answer directly if sufficient — use memory_search or tag_search only if you need more.\n",
  ];

  // Tier 1: full content
  for (const c of tier1) {
    const badge = c.doc
      ? [c.doc.type, c.doc.status].filter(Boolean).join("|")
      : "";
    const badgeStr = badge ? ` [${badge}]` : "";
    const title = c.doc?.title ?? c.path.split("/").pop()?.replace(/\.md$/, "") ?? c.path;

    lines.push(`--- ${c.path}${badgeStr} (score: ${c.finalScore.toFixed(2)}, via: ${sourceLabel(c.sources)}) ---`);
    lines.push(`"${title}"`);
    if (c.doc?.tags && c.doc.tags.length > 0) {
      lines.push(`Tags: ${c.doc.tags.join(", ")}`);
    }
    if (c.content) {
      lines.push(c.content);
    } else if (c.snippet) {
      lines.push(c.snippet);
    } else if (c.doc?.summary) {
      lines.push(c.doc.summary);
    }
    lines.push("");
  }

  // Tier 2: metadata only
  if (tier2.length > 0) {
    lines.push("**Also relevant** (use memory_get for full content):");
    for (const c of tier2) {
      const badge = c.doc
        ? [c.doc.type, c.doc.status].filter(Boolean).join("|")
        : "";
      const badgeStr = badge ? ` [${badge}]` : "";
      const title = c.doc?.title ?? c.path.split("/").pop()?.replace(/\.md$/, "") ?? c.path;
      const detail = c.doc?.summary
        ? ` — ${c.doc.summary.slice(0, 100)}`
        : c.snippet
          ? ` — ${c.snippet.slice(0, 100)}`
          : c.doc?.tags && c.doc.tags.length > 0
            ? ` — Tags: ${c.doc.tags.join(", ")}`
            : "";
      lines.push(`- ${c.path}${badgeStr} "${title}" (${c.finalScore.toFixed(2)})${detail}`);
    }
    lines.push("");
  }

  lines.push("</memory_context>");

  const full = lines.join("\n");
  if (full.length <= budget) return full;

  // Over budget: truncate and close tag
  return full.slice(0, budget - 40) + "\n[... truncated]\n</memory_context>";
}

// ── Tool Output: tag_search ──────────────────────────────────────────

export function formatTagSearchResult(
  tags: string[],
  mode: string,
  docs: DocMeta[],
  totalMatches: number,
  limit: number,
  unresolvedTags: string[],
  suggestions: Map<string, string[]>,
): string {
  const modeLabel = mode === "and" ? "ALL" : "ANY";
  const tagList = tags.map((t) => `"${t}"`).join(", ");

  if (docs.length === 0 && unresolvedTags.length > 0) {
    const lines = [`No documents found matching tags [${tagList}] (${modeLabel} mode).`];
    for (const tag of unresolvedTags) {
      const sugg = suggestions.get(tag);
      if (sugg && sugg.length > 0) {
        lines.push(`  Tag "${tag}" not found. Did you mean: ${sugg.join(", ")}?`);
      } else {
        lines.push(`  Tag "${tag}" not found in the vault.`);
      }
    }
    return lines.join("\n");
  }

  if (docs.length === 0) {
    return `No documents found matching tags [${tagList}] (${modeLabel} mode).`;
  }

  const lines: string[] = [
    `**Tag search: [${tagList}] (${modeLabel} mode)**`,
    `Showing ${docs.length} of ${totalMatches} matches:\n`,
  ];

  for (const doc of docs) {
    lines.push(formatDocLine(doc));
  }

  if (totalMatches > limit) {
    lines.push(`\n... and ${totalMatches - limit} more. Use a higher limit or narrow with AND mode.`);
  }

  return lines.join("\n");
}

// ── Tool Output: tag_explore ─────────────────────────────────────────

export function formatTagExploreResult(
  tag: string,
  tagDocCount: number,
  tagIdf: number,
  relatedTags: Array<{ tag: string; count: number; idf: number; docFreq: number }>,
  bridgeTag?: string,
  bridgeDocs?: DocMeta[],
): string {
  const lines: string[] = [
    `**Tag "${tag}"** — appears in ${tagDocCount} documents (IDF: ${tagIdf.toFixed(2)})\n`,
  ];

  if (relatedTags.length > 0) {
    lines.push("**Co-occurring tags:**");
    for (const rt of relatedTags) {
      lines.push(`  ${rt.tag} (${rt.count} docs together, ${rt.docFreq} total, IDF: ${rt.idf.toFixed(2)})`);
    }
    lines.push("");
  } else {
    lines.push("No co-occurring tags found.\n");
  }

  if (bridgeTag) {
    if (bridgeDocs && bridgeDocs.length > 0) {
      lines.push(`**Documents bridging "${tag}" and "${bridgeTag}":**`);
      for (const doc of bridgeDocs) {
        lines.push(formatDocLine(doc));
      }
    } else {
      lines.push(`No documents found with both "${tag}" and "${bridgeTag}" tags.`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

// ── Tool Output: vault_overview ──────────────────────────────────────

export function formatVaultOverview(
  stats: TagIndexStats,
  detail: string,
): string {
  if (stats.docCount === 0) {
    return "The vault is empty. No markdown files were found.";
  }

  if (detail === "tags") {
    return formatTagList(stats);
  }
  if (detail === "hubs") {
    return formatHubList(stats);
  }
  if (detail === "types") {
    return formatTypeBreakdown(stats);
  }

  // Default: summary
  const lines: string[] = [
    "**Vault Overview**\n",
    `Documents: ${stats.docCount} | Tags: ${stats.tagCount} | Hub pages: ${stats.hubPages.length}`,
    `Last indexed: ${new Date(stats.builtAt).toISOString()}\n`,
  ];

  // Type distribution
  const typeParts: string[] = [];
  const sorted = Object.entries(stats.typeDistribution).sort((a, b) => b[1] - a[1]);
  for (const [type, count] of sorted) {
    typeParts.push(`${type}: ${count}`);
  }
  lines.push(`**Type distribution:** ${typeParts.join(" | ")}\n`);

  // Top tags
  if (stats.topTags.length > 0) {
    const tagParts = stats.topTags.slice(0, 15).map((t) => `${t.tag} (${t.docFreq})`);
    lines.push(`**Top tags:** ${tagParts.join(" | ")}\n`);
  }

  // Top docs
  if (stats.topDocsByImportance.length > 0) {
    lines.push("**Top documents by importance:**");
    for (const doc of stats.topDocsByImportance.slice(0, 5)) {
      lines.push(formatDocLine(doc, 80));
    }
  }

  return lines.join("\n");
}

function formatTagList(stats: TagIndexStats): string {
  const lines: string[] = [
    `**All Tags** (${stats.tagCount} tags across ${stats.docCount} documents)\n`,
  ];
  for (const t of stats.topTags) {
    lines.push(`  ${t.tag} — ${t.docFreq} docs (IDF: ${t.idf.toFixed(2)})`);
  }
  return lines.join("\n");
}

function formatHubList(stats: TagIndexStats): string {
  if (stats.hubPages.length === 0) {
    return "No hub pages found in the vault.";
  }
  const lines: string[] = [
    `**Hub Pages** (${stats.hubPages.length} index pages)\n`,
  ];
  for (const doc of stats.hubPages) {
    lines.push(formatDocLine(doc, 100));
  }
  return lines.join("\n");
}

function formatTypeBreakdown(stats: TagIndexStats): string {
  const lines: string[] = [
    `**Document Types** (${stats.docCount} total documents)\n`,
  ];
  const sorted = Object.entries(stats.typeDistribution).sort((a, b) => b[1] - a[1]);
  for (const [type, count] of sorted) {
    const pct = ((count / stats.docCount) * 100).toFixed(1);
    lines.push(`  ${type}: ${count} (${pct}%)`);
  }
  return lines.join("\n");
}
