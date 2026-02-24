/**
 * index.ts — memory-graph OpenClaw plugin.
 *
 * Adds graph-aware memory augmentation to Lloyd by parsing Obsidian
 * wiki-links into a traversable knowledge graph. Provides:
 *
 *  1. Transparent hook (before_prompt_build) — enriches context with
 *     1-hop neighbors of topics detected in the user's query.
 *  2. Explicit tools — graph_neighbors, graph_path, graph_stats.
 */

import { existsSync } from "node:fs";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { scanVault } from "./scanner.js";
import { KnowledgeGraph } from "./graph.js";
import {
  formatHookContext,
  formatNeighborsTool,
  formatPathTool,
  formatStatsTool,
} from "./format.js";

// ── Configuration ─────────────────────────────────────────────────────

const VAULT_PATH = process.env.HOME
  ? `${process.env.HOME}/obsidian`
  : "/home/alansrobotlab/obsidian";
const REFRESH_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes
const HOOK_MAX_NEIGHBORS = 5;
const MIN_QUERY_LENGTH = 8;

// ── State ─────────────────────────────────────────────────────────────

let graph: KnowledgeGraph = new KnowledgeGraph();
let refreshLock = false;

// ── Graph Build ───────────────────────────────────────────────────────

function buildGraph(logger?: { info?: (...args: any[]) => void; warn?: (...args: any[]) => void }): KnowledgeGraph {
  if (!existsSync(VAULT_PATH)) {
    logger?.warn?.(`memory-graph: vault not found at ${VAULT_PATH}`);
    return new KnowledgeGraph();
  }
  const scan = scanVault(VAULT_PATH);
  const g = KnowledgeGraph.fromScan(scan);
  logger?.info?.(
    `memory-graph: built ${g.nodeCount} nodes, ${g.edgeCount} edges, ${g.brokenLinkCount} broken links`,
  );
  return g;
}

async function refreshGraph(logger?: any): Promise<void> {
  if (refreshLock) return;
  refreshLock = true;
  try {
    const newGraph = buildGraph(logger);
    graph = newGraph; // atomic swap
  } finally {
    refreshLock = false;
  }
}

// ── Query Keyword Extraction ──────────────────────────────────────────

const FILLER_WORDS = new Set([
  "hey", "hi", "what", "do", "you", "know", "about", "my", "work", "with",
  "the", "a", "an", "is", "are", "was", "were", "can", "could", "would",
  "should", "tell", "me", "i", "want", "to", "how", "when", "where", "why",
  "who", "which", "that", "this", "these", "those", "it", "be", "been",
  "being", "have", "has", "had", "does", "did", "will", "shall", "may",
  "might", "some", "any", "all", "more", "very", "just", "your", "our",
  "their", "its", "there", "give", "get", "let", "make", "see", "look",
  "show", "find", "help", "think", "say", "please", "thanks", "ok", "so",
  "and", "or", "but", "in", "on", "at", "of", "for", "from", "up", "out",
  "if", "no", "not", "as", "into", "through", "during", "before", "after",
  "use", "using", "graph", "search", "memory", "related", "connected",
]);

function extractTopicKeywords(prompt: string): string[] {
  // Strip OpenClaw metadata envelope
  const match = prompt.match(/\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+[\d:]+\s+\w+\]\s+([\s\S]+)$/);
  const text = match ? match[1].trim() : prompt.trim();

  // Split into words and filter
  const words = text
    .toLowerCase()
    .replace(/[?!.,;:'"()\[\]{}]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !FILLER_WORDS.has(w));

  // Also look for multi-word phrases (bigrams from adjacent non-filler words)
  const phrases: string[] = [];
  const rawWords = text
    .replace(/[?!.,;:'"()\[\]{}]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
  for (let i = 0; i < rawWords.length - 1; i++) {
    if (
      !FILLER_WORDS.has(rawWords[i].toLowerCase()) &&
      !FILLER_WORDS.has(rawWords[i + 1].toLowerCase())
    ) {
      phrases.push(`${rawWords[i]} ${rawWords[i + 1]}`.toLowerCase());
    }
  }

  return [...phrases, ...words];
}

// ── Plugin Registration ───────────────────────────────────────────────

export default function register(api: OpenClawPluginApi) {
  // Build graph on load
  graph = buildGraph(api.logger);

  // Periodic refresh
  const timer = setInterval(() => refreshGraph(api.logger), REFRESH_INTERVAL_MS);
  // Ensure timer doesn't prevent process exit
  if (timer.unref) timer.unref();

  // ── Hook: before_prompt_build ────────────────────────────────────

  api.on(
    "before_prompt_build",
    async (event) => {
      if (graph.edgeCount === 0) return;

      const prompt = (event as any).prompt as string | undefined;
      if (!prompt || prompt.length < MIN_QUERY_LENGTH) return;

      const keywords = extractTopicKeywords(prompt);
      if (keywords.length === 0) return;

      // Resolve keywords to graph nodes, deduplicate
      const resolvedPaths = new Set<string>();
      const seeds: Array<{ seedPath: string; neighbors: import("./graph.js").GraphNode[] }> = [];

      for (const kw of keywords) {
        const nodes = graph.resolveTopic(kw);
        if (nodes.length === 0) continue;
        // Take the highest-PageRank match
        const best = nodes[0];
        if (resolvedPaths.has(best.path)) continue;
        resolvedPaths.add(best.path);

        const neighbors = graph.getNeighbors(best.path, 1, "both");
        if (neighbors.length > 0) {
          seeds.push({
            seedPath: best.path,
            neighbors: neighbors.slice(0, HOOK_MAX_NEIGHBORS),
          });
        }
        // Limit to 3 seed topics to keep context concise
        if (seeds.length >= 3) break;
      }

      const context = formatHookContext(seeds);
      if (context) {
        return { prependContext: context };
      }
    },
    { priority: 200 }, // run after memory-prefetch (default priority)
  );

  // ── Tool: graph_neighbors ────────────────────────────────────────

  api.registerTool({
    name: "graph_neighbors",
    description:
      "Find documents connected to a given topic in the Obsidian knowledge graph. " +
      "Returns notes that link to or from the specified document, ranked by authority (PageRank). " +
      "Use this to explore what's related to a topic beyond what memory_search returns.",
    parameters: {
      type: "object",
      properties: {
        topic: {
          type: "string",
          description: "A document path, filename stem, title, or tag to find neighbors for",
        },
        hops: {
          type: "integer",
          description: "Number of link hops to traverse (1-3, default 1)",
        },
        direction: {
          type: "string",
          enum: ["in", "out", "both"],
          description: "Link direction: 'in' (pages linking TO topic), 'out' (pages topic links TO), 'both' (default)",
        },
      },
      required: ["topic"],
    },
    async execute(_id: string, params: any) {
      const topic = params.topic as string;
      const hops = Math.min(Math.max((params.hops as number) || 1, 1), 3);
      const direction = (params.direction as "in" | "out" | "both") || "both";

      const resolved = graph.resolveTopic(topic);
      const mainNode = resolved[0] ?? null;

      let inbound: import("./graph.js").GraphNode[] = [];
      let outbound: import("./graph.js").GraphNode[] = [];
      let extended: import("./graph.js").GraphNode[] = [];

      if (mainNode) {
        const dir = graph.getDirectionalNeighbors(mainNode.path);
        inbound = dir.inbound;
        outbound = dir.outbound;
        extended = graph.getNeighbors(mainNode.path, hops, direction);
      }

      return {
        content: [{
          type: "text" as const,
          text: formatNeighborsTool(topic, resolved, inbound, outbound, extended, hops),
        }],
      };
    },
  });

  // ── Tool: graph_path ─────────────────────────────────────────────

  api.registerTool({
    name: "graph_path",
    description:
      "Find the relationship path between two topics in the knowledge graph. " +
      "Returns the shortest chain of linked documents connecting topic A to topic B, " +
      "ranked by authority of intermediate nodes. " +
      "Use this to understand how concepts relate through the vault's link structure.",
    parameters: {
      type: "object",
      properties: {
        from: {
          type: "string",
          description: "Starting topic (document path, stem, title, or tag)",
        },
        to: {
          type: "string",
          description: "Ending topic (document path, stem, title, or tag)",
        },
      },
      required: ["from", "to"],
    },
    async execute(_id: string, params: any) {
      const fromTopic = params.from as string;
      const toTopic = params.to as string;

      const fromResolved = graph.resolveTopic(fromTopic);
      const toResolved = graph.resolveTopic(toTopic);

      const fromNode = fromResolved[0] ?? null;
      const toNode = toResolved[0] ?? null;

      let paths: import("./graph.js").GraphPath[] = [];
      if (fromNode && toNode) {
        paths = graph.findPaths(fromNode.path, toNode.path);
      }

      return {
        content: [{
          type: "text" as const,
          text: formatPathTool(fromTopic, toTopic, fromNode, toNode, paths, graph.nodes),
        }],
      };
    },
  });

  // ── Tool: graph_stats ────────────────────────────────────────────

  api.registerTool({
    name: "graph_stats",
    description:
      "Show knowledge graph statistics: document count, link count, most connected " +
      "and highest-authority pages. Use this to understand the vault's link structure.",
    parameters: {
      type: "object",
      properties: {},
    },
    async execute() {
      const stats = graph.getStats();
      return {
        content: [{
          type: "text" as const,
          text: formatStatsTool(stats),
        }],
      };
    },
  });

  api.logger.info?.("memory-graph: registered (hook + 3 tools)");
}
