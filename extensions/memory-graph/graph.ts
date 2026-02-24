/**
 * graph.ts — In-memory knowledge graph with PageRank, BFS neighbors, and path discovery.
 *
 * Builds a directed graph from wiki-link edges, computes PageRank authority
 * scores, and supports neighbor traversal and shortest-path queries.
 */

import type { ScannedNode, ScannedEdge, ScanResult } from "./scanner.js";

// ── Types ──────────────────────────────────────────────────────────────

export interface GraphNode {
  path: string;
  stem: string;
  title: string;
  summary: string;
  tags: string[];
  pageRank: number;
  inDegree: number;
  outDegree: number;
}

export interface GraphPath {
  nodes: string[];        // ordered list of paths from start to end
  length: number;         // number of hops
  score: number;          // cumulative PageRank of intermediate nodes
}

export interface GraphStats {
  nodeCount: number;
  edgeCount: number;
  brokenLinkCount: number;
  topByPageRank: GraphNode[];
  topByInDegree: GraphNode[];
  builtAt: number;
}

// ── Knowledge Graph ───────────────────────────────────────────────────

export class KnowledgeGraph {
  nodes = new Map<string, GraphNode>();
  outEdges = new Map<string, Set<string>>();
  inEdges = new Map<string, Set<string>>();
  stemIndex = new Map<string, string[]>();
  brokenLinkCount = 0;
  builtAt = 0;

  get nodeCount(): number { return this.nodes.size; }
  get edgeCount(): number {
    let count = 0;
    for (const s of this.outEdges.values()) count += s.size;
    return count;
  }

  /**
   * Build the graph from a scan result.
   */
  static fromScan(scan: ScanResult): KnowledgeGraph {
    const g = new KnowledgeGraph();
    g.stemIndex = scan.stemIndex;
    g.builtAt = Date.now();

    // Count broken links
    for (const targets of scan.brokenLinks.values()) {
      g.brokenLinkCount += targets.length;
    }

    // Create nodes
    for (const n of scan.nodes) {
      g.nodes.set(n.path, {
        path: n.path,
        stem: n.stem,
        title: n.title,
        summary: n.summary,
        tags: n.tags,
        pageRank: 0,
        inDegree: 0,
        outDegree: 0,
      });
      g.outEdges.set(n.path, new Set());
      g.inEdges.set(n.path, new Set());
    }

    // Add edges
    for (const e of scan.edges) {
      if (!g.nodes.has(e.source) || !g.nodes.has(e.target)) continue;
      g.outEdges.get(e.source)!.add(e.target);
      g.inEdges.get(e.target)!.add(e.source);
    }

    // Compute degrees
    for (const [path, node] of g.nodes) {
      node.outDegree = g.outEdges.get(path)?.size ?? 0;
      node.inDegree = g.inEdges.get(path)?.size ?? 0;
    }

    // Compute PageRank
    g.computePageRank();

    return g;
  }

  // ── PageRank ──────────────────────────────────────────────────────

  private computePageRank(damping = 0.85, iterations = 20): void {
    const N = this.nodeCount;
    if (N === 0) return;

    const scores = new Map<string, number>();
    const initial = 1 / N;
    for (const path of this.nodes.keys()) {
      scores.set(path, initial);
    }

    for (let i = 0; i < iterations; i++) {
      const newScores = new Map<string, number>();

      // Sum contributions from dangling nodes (no outbound links)
      let danglingSum = 0;
      for (const [path, node] of this.nodes) {
        if (node.outDegree === 0) {
          danglingSum += scores.get(path)!;
        }
      }

      for (const [path] of this.nodes) {
        let sum = 0;
        const inbound = this.inEdges.get(path);
        if (inbound) {
          for (const src of inbound) {
            const srcNode = this.nodes.get(src)!;
            sum += scores.get(src)! / srcNode.outDegree;
          }
        }
        newScores.set(
          path,
          (1 - damping) / N + damping * (sum + danglingSum / N),
        );
      }

      // Copy new scores
      for (const [path, score] of newScores) {
        scores.set(path, score);
      }
    }

    // Normalize to [0, 1]
    let maxScore = 0;
    for (const s of scores.values()) {
      if (s > maxScore) maxScore = s;
    }
    if (maxScore > 0) {
      for (const [path, node] of this.nodes) {
        node.pageRank = scores.get(path)! / maxScore;
      }
    }
  }

  // ── Neighbor Retrieval ────────────────────────────────────────────

  getNeighbors(
    path: string,
    hops = 1,
    direction: "in" | "out" | "both" = "both",
  ): GraphNode[] {
    if (!this.nodes.has(path)) return [];

    const visited = new Set<string>([path]);
    let frontier = new Set<string>([path]);

    for (let hop = 0; hop < hops; hop++) {
      const nextFrontier = new Set<string>();
      for (const node of frontier) {
        if (direction === "out" || direction === "both") {
          const out = this.outEdges.get(node);
          if (out) for (const t of out) {
            if (!visited.has(t)) { nextFrontier.add(t); visited.add(t); }
          }
        }
        if (direction === "in" || direction === "both") {
          const inc = this.inEdges.get(node);
          if (inc) for (const s of inc) {
            if (!visited.has(s)) { nextFrontier.add(s); visited.add(s); }
          }
        }
      }
      frontier = nextFrontier;
      if (frontier.size === 0) break;
    }

    visited.delete(path);
    return Array.from(visited)
      .map((p) => this.nodes.get(p)!)
      .filter(Boolean)
      .sort((a, b) => b.pageRank - a.pageRank);
  }

  /**
   * Get immediate inbound/outbound neighbors separately (useful for tool output).
   */
  getDirectionalNeighbors(path: string): { inbound: GraphNode[]; outbound: GraphNode[] } {
    const inbound: GraphNode[] = [];
    const outbound: GraphNode[] = [];
    const inc = this.inEdges.get(path);
    if (inc) for (const s of inc) {
      const n = this.nodes.get(s);
      if (n) inbound.push(n);
    }
    const out = this.outEdges.get(path);
    if (out) for (const t of out) {
      const n = this.nodes.get(t);
      if (n) outbound.push(n);
    }
    inbound.sort((a, b) => b.pageRank - a.pageRank);
    outbound.sort((a, b) => b.pageRank - a.pageRank);
    return { inbound, outbound };
  }

  // ── Path Discovery ────────────────────────────────────────────────

  findPaths(startPath: string, endPath: string, maxHops = 4): GraphPath[] {
    if (startPath === endPath) return [];
    if (!this.nodes.has(startPath) || !this.nodes.has(endPath)) return [];

    const allPaths: string[][] = [];
    const queue: Array<{ node: string; path: string[] }> = [
      { node: startPath, path: [startPath] },
    ];

    // BFS — track visited per-path-length to allow multiple shortest paths
    const visitedAtDepth = new Map<string, number>();
    visitedAtDepth.set(startPath, 0);

    while (queue.length > 0) {
      const { node, path } = queue.shift()!;
      const depth = path.length - 1;

      if (depth >= maxHops) continue;
      // If we already found paths, don't explore deeper than shortest found
      if (allPaths.length > 0 && path.length >= allPaths[0].length) continue;

      // Explore both directions (undirected traversal for path finding)
      const neighbors = new Set<string>();
      const out = this.outEdges.get(node);
      if (out) for (const t of out) neighbors.add(t);
      const inc = this.inEdges.get(node);
      if (inc) for (const s of inc) neighbors.add(s);

      for (const neighbor of neighbors) {
        if (path.includes(neighbor)) continue; // no cycles

        const newPath = [...path, neighbor];

        if (neighbor === endPath) {
          allPaths.push(newPath);
          continue;
        }

        // Only visit if not seen at a shorter depth
        const prevDepth = visitedAtDepth.get(neighbor);
        if (prevDepth !== undefined && prevDepth < newPath.length - 1) continue;
        visitedAtDepth.set(neighbor, newPath.length - 1);

        queue.push({ node: neighbor, path: newPath });
      }
    }

    // Score paths by cumulative PageRank of intermediate nodes
    const scored: GraphPath[] = allPaths.map((p) => ({
      nodes: p,
      length: p.length - 1,
      score: p
        .slice(1, -1)
        .reduce((sum, n) => sum + (this.nodes.get(n)?.pageRank ?? 0), 0),
    }));

    // Sort by length first, then by score descending
    scored.sort((a, b) => a.length - b.length || b.score - a.score);
    return scored.slice(0, 5);
  }

  // ── Topic Resolution ──────────────────────────────────────────────

  /**
   * Map a user query string to graph node(s).
   * Tries: exact stem → exact path → case-insensitive stem → title substring → tag match.
   */
  resolveTopic(query: string): GraphNode[] {
    const q = query.trim();
    if (!q) return [];

    // 1. Exact path match
    const byPath = this.nodes.get(q) ?? this.nodes.get(q + ".md");
    if (byPath) return [byPath];

    // 2. Exact stem match
    const stemPaths = this.stemIndex.get(q);
    if (stemPaths) {
      const results = stemPaths
        .map((p) => this.nodes.get(p))
        .filter((n): n is GraphNode => n != null);
      if (results.length > 0) return results;
    }

    // 3. Case-insensitive stem match
    const lowerQ = q.toLowerCase();
    for (const [stem, paths] of this.stemIndex) {
      if (stem.toLowerCase() === lowerQ) {
        const results = paths
          .map((p) => this.nodes.get(p))
          .filter((n): n is GraphNode => n != null);
        if (results.length > 0) return results;
      }
    }

    // 4. Title substring match
    const titleMatches: GraphNode[] = [];
    for (const node of this.nodes.values()) {
      if (node.title.toLowerCase().includes(lowerQ)) {
        titleMatches.push(node);
      }
    }
    if (titleMatches.length > 0) {
      return titleMatches.sort((a, b) => b.pageRank - a.pageRank);
    }

    // 5. Tag match
    const tagMatches: GraphNode[] = [];
    for (const node of this.nodes.values()) {
      if (node.tags.some((t) => t.toLowerCase() === lowerQ)) {
        tagMatches.push(node);
      }
    }
    if (tagMatches.length > 0) {
      return tagMatches.sort((a, b) => b.pageRank - a.pageRank);
    }

    return [];
  }

  // ── Stats ─────────────────────────────────────────────────────────

  getStats(): GraphStats {
    const byPageRank = Array.from(this.nodes.values())
      .sort((a, b) => b.pageRank - a.pageRank)
      .slice(0, 10);
    const byInDegree = Array.from(this.nodes.values())
      .sort((a, b) => b.inDegree - a.inDegree || b.pageRank - a.pageRank)
      .slice(0, 10);

    return {
      nodeCount: this.nodeCount,
      edgeCount: this.edgeCount,
      brokenLinkCount: this.brokenLinkCount,
      topByPageRank: byPageRank,
      topByInDegree: byInDegree,
      builtAt: this.builtAt,
    };
  }
}
