/**
 * format.ts — LLM context formatting for graph results.
 *
 * Formats graph data into structured text blocks that Lloyd
 * can understand and act on in conversation.
 */

import type { GraphNode, GraphPath, GraphStats } from "./graph.js";

// ── Hook Context (before_prompt_build) ────────────────────────────────

export function formatHookContext(
  seeds: Array<{ seedPath: string; neighbors: GraphNode[] }>,
): string {
  if (seeds.length === 0 || seeds.every((s) => s.neighbors.length === 0)) return "";

  const lines: string[] = [
    "<graph_context>",
    "Documents linked to your memory search results — these may provide additional relevant context.\n",
  ];

  for (const { seedPath, neighbors } of seeds) {
    if (neighbors.length === 0) continue;
    lines.push(`**Linked to "${seedPath}":**`);
    for (const n of neighbors.slice(0, 5)) {
      const pr = n.pageRank.toFixed(2);
      const desc = n.summary ? ` — ${n.summary.slice(0, 80)}` : "";
      lines.push(`- ${n.path} (authority: ${pr}) "${n.title}"${desc}`);
    }
    lines.push("");
  }

  lines.push("</graph_context>");
  return lines.join("\n");
}

// ── Tool Output: graph_neighbors ──────────────────────────────────────

export function formatNeighborsTool(
  topic: string,
  resolved: GraphNode[],
  inbound: GraphNode[],
  outbound: GraphNode[],
  extended: GraphNode[],
  hops: number,
): string {
  if (resolved.length === 0) {
    return `No graph node found matching "${topic}". The knowledge graph has ${
      extended.length === 0 ? "no links yet — run autolink.py to add wiki-links to your vault." : "nodes, but none matched your query."
    }`;
  }

  const node = resolved[0];
  const lines: string[] = [
    `Knowledge graph neighbors for "${node.title}" (${node.path}):`,
    `Authority: ${node.pageRank.toFixed(2)} | Inbound: ${node.inDegree} | Outbound: ${node.outDegree}`,
    "",
  ];

  if (inbound.length > 0) {
    lines.push("**Documents linking TO this page (inbound):**");
    for (const n of inbound.slice(0, 10)) {
      const desc = n.summary ? ` — ${n.summary.slice(0, 80)}` : "";
      lines.push(`  ${n.pageRank.toFixed(2)} | ${n.path} — "${n.title}"${desc}`);
    }
    lines.push("");
  }

  if (outbound.length > 0) {
    lines.push("**Documents this page links TO (outbound):**");
    for (const n of outbound.slice(0, 10)) {
      const desc = n.summary ? ` — ${n.summary.slice(0, 80)}` : "";
      lines.push(`  ${n.pageRank.toFixed(2)} | ${n.path} — "${n.title}"${desc}`);
    }
    lines.push("");
  }

  if (hops > 1 && extended.length > inbound.length + outbound.length) {
    const directPaths = new Set([
      ...inbound.map((n) => n.path),
      ...outbound.map((n) => n.path),
    ]);
    const hopNeighbors = extended.filter((n) => !directPaths.has(n.path));
    if (hopNeighbors.length > 0) {
      lines.push(`**${hops}-hop neighbors (through intermediaries):**`);
      for (const n of hopNeighbors.slice(0, 10)) {
        const desc = n.summary ? ` — ${n.summary.slice(0, 60)}` : "";
        lines.push(`  ${n.pageRank.toFixed(2)} | ${n.path} — "${n.title}"${desc}`);
      }
      lines.push("");
    }
  }

  if (inbound.length === 0 && outbound.length === 0) {
    lines.push("This document has no wiki-links to or from other documents yet.");
  }

  return lines.join("\n");
}

// ── Tool Output: graph_path ───────────────────────────────────────────

export function formatPathTool(
  fromTopic: string,
  toTopic: string,
  fromNode: GraphNode | null,
  toNode: GraphNode | null,
  paths: GraphPath[],
  nodeMap: Map<string, GraphNode>,
): string {
  if (!fromNode) return `No graph node found matching "${fromTopic}".`;
  if (!toNode) return `No graph node found matching "${toTopic}".`;

  if (paths.length === 0) {
    return `No path found between "${fromNode.title}" and "${toNode.title}" within 4 hops. These topics may not be connected in the knowledge graph yet.`;
  }

  const lines: string[] = [
    `Paths from "${fromNode.title}" to "${toNode.title}":\n`,
  ];

  for (let i = 0; i < paths.length; i++) {
    const p = paths[i];
    const chain = p.nodes
      .map((nodePath) => {
        const n = nodeMap.get(nodePath);
        return n ? `${n.title} (${n.path})` : nodePath;
      })
      .join(" → ");
    const scoreStr = p.score > 0 ? `, intermediate authority: ${p.score.toFixed(2)}` : "";
    lines.push(`**Path ${i + 1}** (${p.length} hops${scoreStr}):`);
    lines.push(`  ${chain}`);
    lines.push("");
  }

  return lines.join("\n");
}

// ── Tool Output: graph_stats ──────────────────────────────────────────

export function formatStatsTool(stats: GraphStats): string {
  if (stats.nodeCount === 0) {
    return "The knowledge graph is empty. No markdown files were found in the vault.";
  }

  const lines: string[] = [
    "**Knowledge Graph Statistics**\n",
    `Nodes: ${stats.nodeCount} documents`,
    `Edges: ${stats.edgeCount} wiki-links`,
    `Broken links: ${stats.brokenLinkCount}`,
    `Last built: ${new Date(stats.builtAt).toISOString()}`,
    "",
  ];

  if (stats.edgeCount === 0) {
    lines.push("No wiki-links found yet. Run autolink.py to add [[wikilinks]] throughout the vault.");
    return lines.join("\n");
  }

  if (stats.topByPageRank.length > 0) {
    lines.push("**Top 10 by Authority (PageRank):**");
    for (let i = 0; i < stats.topByPageRank.length; i++) {
      const n = stats.topByPageRank[i];
      lines.push(`  ${i + 1}. ${n.pageRank.toFixed(3)} | ${n.path} — "${n.title}" (in: ${n.inDegree}, out: ${n.outDegree})`);
    }
    lines.push("");
  }

  if (stats.topByInDegree.length > 0 && stats.topByInDegree[0].inDegree > 0) {
    lines.push("**Top 10 by Inbound Links (most referenced):**");
    for (let i = 0; i < stats.topByInDegree.length; i++) {
      const n = stats.topByInDegree[i];
      if (n.inDegree === 0) break;
      lines.push(`  ${i + 1}. ${n.inDegree} inbound | ${n.path} — "${n.title}"`);
    }
    lines.push("");
  }

  return lines.join("\n");
}
