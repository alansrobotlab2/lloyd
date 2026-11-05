// The Memory graph's data and layout, kept apart from EntityGraph so that the
// sidebar can start fetching on hover without pulling in three.js.
//
// Measured 2026-11-05: a click on Memory waited ~5 s for the EntityGraph chunk
// (queued behind the IDE's ~2,000-module Monaco preload under Vite dev), then
// 1.5 s for the graph, then ~7 s of visible force layout because the stored
// layout was discarded whenever the node set changed at all.

import { api, type EntityGraphData, type EntityGraphPositions } from "../api";

/** The EntityGraph chunk. One loader for React.lazy and for preloading, so
 *  both resolve to the same module request. */
export const importEntityGraph = () => import("../components/EntityGraph");

/** Reuse a fetched graph this long. EntityGraph keeps its own settled node
 *  objects on top of this; this tier only spares the network round-trip. */
const GRAPH_FRESH_MS = 5 * 60 * 1000;

let graphEntry: { at: number; promise: Promise<EntityGraphData> } | null = null;
let layoutEntry: Promise<EntityGraphPositions> | null = null;

/** The graph payload, one request at a time. A stale answer (the server's
 *  previous build, served while it rebuilds) is handed to the caller but not
 *  kept, so the next call asks again. */
export function fetchEntityGraph(force = false): Promise<EntityGraphData> {
  if (!force && graphEntry && Date.now() - graphEntry.at < GRAPH_FRESH_MS) {
    return graphEntry.promise;
  }
  const entry = { at: Date.now(), promise: api.entityGraph() };
  graphEntry = entry;
  entry.promise.then(
    (d) => { if (d.stale && graphEntry === entry) entry.at = 0; },
    () => { if (graphEntry === entry) graphEntry = null; },
  );
  return entry.promise;
}

/** The stored layout. Never rejects: no layout just means laying out afresh. */
export function fetchGraphLayout(): Promise<EntityGraphPositions> {
  if (!layoutEntry) {
    const p = api.entityGraphLayout()
      .then((r) => (r && typeof r.positions === "object" && r.positions) || {})
      .catch(() => {
        if (layoutEntry === p) layoutEntry = null;
        return {} as EntityGraphPositions;
      });
    layoutEntry = p;
  }
  return layoutEntry;
}

/** Store a settled layout for every browser. Fire and forget. */
export function saveGraphLayout(positions: EntityGraphPositions): void {
  layoutEntry = Promise.resolve(positions);
  api.saveEntityGraphLayout(positions).catch(() => {
    // Not worth surfacing: the next visit lays out afresh and saves again.
  });
}

let chunkPreload: Promise<unknown> | null = null;

/** Load the EntityGraph chunk (three.js and the force layout) ahead of need. */
export function preloadEntityGraphChunk(): Promise<unknown> {
  if (!chunkPreload) {
    chunkPreload = importEntityGraph().catch(() => { chunkPreload = null; });
  }
  return chunkPreload;
}

/** Everything the Memory tab needs before it can draw: the chunk, the graph
 *  and its layout. Called on hover over the nav item. */
export function preloadMemoryGraph(): void {
  void preloadEntityGraphChunk();
  fetchEntityGraph().catch(() => {});
  void fetchGraphLayout();
}

// -- Layout placement --

interface Placeable {
  id: string;
  x?: number; y?: number; z?: number;
  fx?: number; fy?: number; fz?: number;
}

interface LinkLike {
  source: string | { id: string };
  target: string | { id: string };
}

/**
 * - `settled`: every node has a position; all pinned, no simulation at all.
 * - `partial`: most do; those are pinned and only the newcomers settle.
 * - `fresh`: too few to keep the shape; known positions are only a head start.
 */
export type LayoutMode = "settled" | "partial" | "fresh";

/** Below this share of known nodes, pinning them would freeze a shape that no
 *  longer fits the graph, so the whole graph settles instead. */
export const MIN_PINNED_SHARE = 0.5;

const endId = (e: string | { id: string }) => (typeof e === "string" ? e : e.id);

/** Give every node a starting position from `positions`, seed the rest beside
 *  neighbours that have one, and say how much simulating is left to do.
 *  Mutates the nodes (they are the simulation's own objects). */
export function placeNodes(
  nodes: Placeable[],
  links: LinkLike[],
  positions: EntityGraphPositions,
  rng: () => number = Math.random,
): LayoutMode {
  if (!nodes.length) return "settled";
  let known = 0;
  for (const n of nodes) if (positions[n.id]) known++;
  const pin = known / nodes.length >= MIN_PINNED_SHARE;

  const placed = new Set<string>();
  for (const n of nodes) {
    const p = positions[n.id];
    if (p) {
      [n.x, n.y, n.z] = p;
      if (pin) [n.fx, n.fy, n.fz] = p;
      else n.fx = n.fy = n.fz = undefined;
      placed.add(n.id);
    } else {
      n.x = n.y = n.z = undefined;
      n.fx = n.fy = n.fz = undefined;
    }
  }
  if (known === nodes.length) return "settled";
  if (known === 0) return "fresh";

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const adj = new Map<string, string[]>();
  for (const l of links) {
    const s = endId(l.source), t = endId(l.target);
    if (!byId.has(s) || !byId.has(t)) continue;
    (adj.get(s) ?? adj.set(s, []).get(s)!).push(t);
    (adj.get(t) ?? adj.set(t, []).get(t)!).push(s);
  }
  const jitter = () => (rng() - 0.5) * 16;

  // Newcomers go beside their placed neighbours; a pass can place nodes whose
  // only neighbours were placed by the pass before, so chains fill inward.
  let pending = nodes.filter((n) => !placed.has(n.id));
  for (let pass = 0; pass < 6 && pending.length; pass++) {
    const next: Placeable[] = [];
    const placedThisPass: Placeable[] = [];
    for (const n of pending) {
      let sx = 0, sy = 0, sz = 0, k = 0;
      for (const m of adj.get(n.id) ?? []) {
        if (!placed.has(m)) continue;
        const o = byId.get(m)!;
        sx += o.x!; sy += o.y!; sz += o.z!; k++;
      }
      if (!k) { next.push(n); continue; }
      n.x = sx / k + jitter(); n.y = sy / k + jitter(); n.z = sz / k + jitter();
      placedThisPass.push(n);
    }
    if (!placedThisPass.length) break;
    for (const n of placedThisPass) placed.add(n.id);
    pending = next;
  }

  // Unconnected to anything placed: somewhere inside the existing shape.
  if (pending.length) {
    let r = 0;
    for (const id of placed) {
      const o = byId.get(id)!;
      r = Math.max(r, Math.hypot(o.x!, o.y!, o.z!));
    }
    r = r || 100;
    for (const n of pending) {
      n.x = (rng() - 0.5) * r; n.y = (rng() - 0.5) * r; n.z = (rng() - 0.5) * r;
    }
  }
  return pin ? "partial" : "fresh";
}

/** The nodes' current positions, rounded to 0.1 (the server rounds the same). */
export function collectPositions(nodes: Placeable[]): EntityGraphPositions {
  const out: EntityGraphPositions = {};
  const r1 = (v: number) => Math.round(v * 10) / 10;
  for (const n of nodes) {
    if (typeof n.x !== "number" || typeof n.y !== "number") continue;
    out[n.id] = [r1(n.x), r1(n.y), r1(n.z ?? 0)];
  }
  return out;
}
